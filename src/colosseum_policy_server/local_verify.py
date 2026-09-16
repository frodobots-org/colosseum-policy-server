"""Protobuf receipt checks and explicitly requested synthetic inference; no weights."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import hashlib
import math
import struct
from pathlib import Path
import re
import ssl

from . import colosseum_pb2 as pb
from google.protobuf.message import DecodeError
from .local_protocol import encode_control, decode_control

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed


class VerificationServer:
    def __init__(self, runtime_profiles, receipts: str | Path, *, download_seconds=3,
                 load_seconds=2, warmup_seconds=1, inference_seconds=.5):
        self.stages = [("downloading", download_seconds), ("loading", load_seconds),
                       ("warming_up", warmup_seconds)]
        self.inference_seconds = inference_seconds
        if any(not math.isfinite(v) or v < 0 for v in
               (download_seconds, load_seconds, warmup_seconds, inference_seconds)):
            raise ValueError("Simulation delays must be finite and nonnegative")
        self.runtime_profiles = sorted(set(runtime_profiles))
        if not self.runtime_profiles or any(not isinstance(p, str) or not p or len(p) > 100 for p in self.runtime_profiles):
            raise ValueError('Specify at least one valid runtime profile')
        self.receipts = Path(receipts)
        self.receipts.parent.mkdir(parents=True, exist_ok=True)

    def validate(self, request):
        if request.get('type') != 'prepare' or request.get('protocol_version') != 1:
            raise ValueError('Expected protocol v1 prepare')
        for field in ('run_id', 'preparation_id'):
            if not isinstance(request.get(field), str) or not 1 <= len(request[field]) <= 200:
                raise ValueError('Invalid preparation identity')
        model = request.get('model')
        if not isinstance(model, dict):
            raise ValueError('Missing model specification')
        if not isinstance(model.get('url'), str) or not re.fullmatch(r'https://huggingface\.co/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', model['url']):
            raise ValueError('Expected Hugging Face model URL')
        if not isinstance(model.get('revision'), str) or not re.fullmatch('[a-fA-F0-9]{40}', model['revision']):
            raise ValueError('Expected pinned model revision')
        if model.get('runtime_profile') not in self.runtime_profiles:
            raise ValueError('Unsupported runtime profile')
        if not isinstance(model.get('action_space'), str) or not model['action_space']:
            raise ValueError('Missing action space')
        for field, maximum in [('action_dim',256), ('control_hz',1000), ('max_horizon',1000)]:
            if type(model.get(field)) is not int or not 1 <= model[field] <= maximum:
                raise ValueError('Invalid action contract')
        if not isinstance(request.get('task'), dict):
            raise ValueError('Missing task')

    async def handle(self, ws):
        try:
            raw = await asyncio.wait_for(ws.recv(), 30)
            request = decode_control(raw)
            if not isinstance(request, dict):
                raise ValueError('Expected Protobuf control message')
            if request.get('type') == 'capabilities' and request.get('protocol_version') == 1:
                await ws.send(encode_control(dict(type='capabilities', protocol_version=1,
                    runtime_profiles=self.runtime_profiles, verification_only=True)))
                return
            self.validate(request)
            simulate = request.get('state') == 'simulate'
            if simulate and (not (request.get('test') is True or request['task'].get('robot_id') == 'test')
                             or request.get('verification_only') is not True):
                raise ValueError('Simulation requires test flag and verification_only')
            receipt = dict(type='received', protocol_version=1, verification_only=True,
                loaded=False, run_id=request['run_id'], preparation_id=request['preparation_id'],
                model=request['model'], transport='wss' if ws.transport.get_extra_info('ssl_object') else 'ws')
            # Persist only the relevant receipt, never headers, tokens, or arbitrary request keys.
            record = {**receipt, 'received_at':datetime.now(timezone.utc).isoformat()}
            with self.receipts.open('a', encoding='utf-8') as out:
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
                out.flush()
            await ws.send(encode_control(receipt))
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if simulate:
                await self.simulate(ws, request, receipt)
                return
            # Optional dummy observation check, with no model loading/action generation.
            raw = await asyncio.wait_for(ws.recv(), 10)
            frame, obs = self.observation(raw, request['run_id'], 1)
            ack = dict(type='observation_received',run_id=request['run_id'],observation_sequence=frame.sequence,
                       sensor_count=len(obs.sensors),state_count=len(obs.state),data_sha256=hashlib.sha256(frame.payload).hexdigest())
            with self.receipts.open('a', encoding='utf-8') as out:
                out.write(json.dumps(ack)+'\n')
            await ws.send(encode_control(ack))
            print(json.dumps(ack),flush=True)
        except (ValueError, TypeError, DecodeError, asyncio.TimeoutError):
            await ws.send(encode_control(dict(type='error', code='INVALID_PREPARATION')))
            await ws.close(code=1008)
        except ConnectionClosed:
            pass


    @staticmethod
    def observation(raw, run_id, sequence):
        if not isinstance(raw, bytes):
            raise ValueError('Binary observation required')
        frame = pb.RelayFrame.FromString(raw)
        if frame.type != pb.OBSERVATION or frame.protocol_version != 1 or frame.session_id != run_id or frame.sequence != sequence:
            raise ValueError('Invalid dummy observation frame')
        obs = pb.Observation.FromString(frame.payload)
        if not obs.state or not obs.sensors:
            raise ValueError('Missing dummy state or image')
        for tensor in obs.state.values():
            count = math.prod(tensor.shape)
            if not tensor.shape or count < 1 or tensor.dtype != pb.FLOAT32 or len(tensor.data) != count*4:
                raise ValueError('Invalid dummy tensor')
            if any(not math.isfinite(x[0]) for x in struct.iter_unpack('<f',tensor.data)):
                raise ValueError('Nonfinite dummy tensor')
        for sensor in obs.sensors:
            if sensor.encoding != pb.RAW_RGB or sensor.width < 1 or sensor.height < 1 or len(sensor.data) != sensor.width*sensor.height*3:
                raise ValueError('Invalid dummy RGB image')
        return frame, obs

    async def simulate(self, ws, request, receipt):
        identity = dict(run_id=request['run_id'], preparation_id=request['preparation_id'],
                        verification_only=True, loaded=False)
        for state, seconds in self.stages:
            started = asyncio.get_running_loop().time()
            while True:
                elapsed = asyncio.get_running_loop().time() - started
                percent = 100 if seconds == 0 else min(100, int(elapsed / seconds * 100))
                await ws.send(encode_control(dict(type='progress', state=state,
                    message=f'SIMULATED {percent}% ({elapsed:.1f}/{seconds:g}s)', **identity)))
                if elapsed >= seconds:
                    break
                try:
                    # Drain and reject early data instead of buffering stale observations
                    # for use after preparation. Closing forces a clean retry.
                    await asyncio.wait_for(ws.recv(), min(1, seconds - elapsed))
                except asyncio.TimeoutError:
                    pass
                else:
                    await ws.send(encode_control(dict(type='error', code='NOT_READY',
                        message='Wait for simulation_ready before sending observations',
                        run_id=request['run_id'])))
                    await ws.close(code=1008)
                    return
        await ws.send(encode_control({**receipt, 'type':'simulation_ready'}))
        sequence = 1
        while True:
            raw = await asyncio.wait_for(ws.recv(), 60)
            if isinstance(raw, bytes):
                close = pb.RelayFrame.FromString(raw)
                if close.type == pb.SESSION_CLOSE and close.protocol_version == 1 and close.session_id == request['run_id']:
                    return
            frame, obs = self.observation(raw, request['run_id'], sequence)
            if obs.instruction != request['task']['instruction'] or obs.control_step != sequence - 1:
                raise ValueError('Synthetic observation task/step mismatch')
            if not frame.deadline_ms:
                raise ValueError('Inference deadline required')
            if self.inference_seconds >= frame.deadline_ms / 1000:
                await asyncio.sleep(frame.deadline_ms / 1000)
                await ws.send(encode_control(dict(type='error', run_id=request['run_id'],
                    code='INFERENCE_TIMEOUT', message='Simulated inference exceeds request deadline')))
                return
            await asyncio.sleep(self.inference_seconds)
            model = request['model']
            plan = pb.ActionPlan(request_sequence=sequence, plan_id=sequence,
                start_step=obs.control_step, valid_until_step=obs.control_step,
                control_hz=model['control_hz'], actions=pb.Tensor(shape=[1,model['action_dim']],
                    dtype=pb.FLOAT32, data=struct.pack(f"<{model['action_dim']}f", *([0.0]*model['action_dim']))))
            await ws.send(pb.RelayFrame(protocol_version=1, type=pb.ACTION_PLAN,
                session_id=request['run_id'], sequence=sequence, payload=plan.SerializeToString()).SerializeToString())
            record = dict(type='simulated_action', run_id=request['run_id'], observation_sequence=sequence,
                sensor_count=len(obs.sensors), state_count=len(obs.state),
                data_sha256=hashlib.sha256(frame.payload).hexdigest(), action_shape=list(plan.actions.shape),
                instruction=obs.instruction, simulation=True, loaded=False)
            with self.receipts.open('a', encoding='utf-8') as out:
                out.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
            sequence += 1


def tls_context(certfile, keyfile):
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certfile, keyfile)
    return context


async def run(args):
    verifier = VerificationServer(args.runtime_profile, args.receipts,
        download_seconds=args.download_seconds, load_seconds=args.load_seconds,
        warmup_seconds=args.warmup_seconds, inference_seconds=args.inference_seconds)
    context = tls_context(args.certfile, args.keyfile) if args.certfile else None
    async with serve(verifier.handle, args.host, args.port,
                     ssl=context, max_size=32 * 1024 * 1024,
                     compression=None, ping_interval=10, ping_timeout=10):
        print(f'{"WSS" if context else "WS"} verification server listening on {args.host}:{args.port}; synthetic test inference available; no weights loaded', flush=True)
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description='Test model preparation and synthetic inference over Protobuf WS(S)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--certfile')
    parser.add_argument('--keyfile')
    parser.add_argument('--runtime-profile', action='append')
    parser.add_argument('--receipts', default='local-receipts.jsonl')
    for phase, default in [('download',3), ('load',2), ('warmup',1), ('inference',.5)]:
        parser.add_argument(f'--{phase}-seconds', type=float, default=default)
    args = parser.parse_args()
    if bool(args.certfile) != bool(args.keyfile):
        parser.error("Supply both --certfile and --keyfile for WSS")
    # Verification profiles describe synthetic transport contracts, not real runtimes.
    args.runtime_profile = args.runtime_profile or [
        "molmoact2-droid-v1",
        "verify-molmoact2-droid-v1",
        "verify-gr00t-n17-droid-v1",
        "verify-pi05-droid-v1",
        "verify-g05-droid-v1",
        "verify-lap-3b-v1",
    ]
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
