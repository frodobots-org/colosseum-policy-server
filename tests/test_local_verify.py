import json
import pytest
from colosseum_policy_server.local_protocol import encode_control, decode_control
from colosseum_policy_server.local_verify import VerificationServer


class Socket:
    def __init__(self, message):
        self.message = message
        self.sent = []
        self.transport = self
    def get_extra_info(self, key): return object()
    async def recv(self):
        if self.sent:
            from websockets.exceptions import ConnectionClosedOK
            raise ConnectionClosedOK(None, None)
        return encode_control(self.message)
    async def send(self, message): self.sent.append(decode_control(message))
    async def close(self, **kwargs): pass


def prepare():
    return dict(type='prepare',protocol_version=1,run_id='run-1',preparation_id='prep-1',task={},model=dict(
        url='https://huggingface.co/allenai/MolmoAct2-DROID',revision='a'*40,
        runtime_profile='molmoact2-droid-v1',action_space='joint_position',action_dim=8,control_hz=15,max_horizon=16))


async def test_receipt_never_claims_loaded_or_ready(tmp_path):
    service = VerificationServer(['molmoact2-droid-v1'], tmp_path/'receipts.jsonl')
    ws = Socket(prepare())
    await service.handle(ws)
    assert len(ws.sent) == 1 and ws.sent[0]['type'] == 'received'
    assert ws.sent[0]['loaded'] is False
    assert json.loads((tmp_path/'receipts.jsonl').read_text())['model'] == prepare()['model']


@pytest.mark.parametrize('field,value', [('revision','main'), ('runtime_profile','unknown'),('action_dim',0),('url','http://bad')])
async def test_invalid_model_has_no_receipt(tmp_path, field, value):
    service = VerificationServer(['molmoact2-droid-v1'], tmp_path/'receipts.jsonl')
    request = prepare()
    request['model'][field] = value
    ws = Socket(request)
    await service.handle(ws)
    assert ws.sent == [dict(type='error',code='INVALID_PREPARATION')]
    assert not (tmp_path/'receipts.jsonl').exists()


async def test_json_text_is_rejected(tmp_path):
    class TextSocket(Socket):
        async def recv(self): return json.dumps(self.message)
    service = VerificationServer(['molmoact2-droid-v1'], tmp_path/'receipts.jsonl')
    ws = TextSocket(prepare())
    await service.handle(ws)
    assert ws.sent == [dict(type='error', code='INVALID_PREPARATION')]
    assert not (tmp_path/'receipts.jsonl').exists()


async def test_simulation_rejects_physical_robot(tmp_path):
    request = {**prepare(), 'state':'simulate', 'verification_only':True, 'task':{'robot_id':'franka'}}
    ws = Socket(request)
    await VerificationServer(['molmoact2-droid-v1'], tmp_path/'receipts').handle(ws)
    assert ws.sent == [dict(type='error',code='INVALID_PREPARATION')]


async def test_simulation_progress_precedes_readiness(tmp_path):
    request = {**prepare(), 'state':'simulate', 'verification_only':True, 'task':{'robot_id':'test'}}
    ws = Socket(request)
    await VerificationServer(['molmoact2-droid-v1'], tmp_path/'receipts',
        download_seconds=0, load_seconds=0, warmup_seconds=0).handle(ws)
    assert [m['type'] for m in ws.sent] == ['received','progress','progress','progress','simulation_ready']
    assert [m['state'] for m in ws.sent[1:4]] == ['downloading','loading','warming_up']
    assert ws.sent[-1]['loaded'] is False and ws.sent[-1]['verification_only'] is True


async def test_early_observation_rejected_without_actions(tmp_path):
    import asyncio
    from websockets.asyncio.server import serve
    from websockets.asyncio.client import connect
    from colosseum_policy_server import colosseum_pb2 as pb
    request={**prepare(), 'state':'simulate','verification_only':True,'task':{'robot_id':'test'}}
    service=VerificationServer(['molmoact2-droid-v1'],tmp_path/'receipts',download_seconds=1)
    async with serve(service.handle,'127.0.0.1',0) as server:
        async with connect(f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}',proxy=None) as ws:
            await ws.send(encode_control(request))
            assert decode_control(await ws.recv())['type']=='received'
            assert decode_control(await ws.recv())['type']=='progress'
            await ws.send(pb.RelayFrame(protocol_version=1,type=pb.OBSERVATION,
                session_id='run-1',sequence=1).SerializeToString())
            error=decode_control(await asyncio.wait_for(ws.recv(),.5))
            assert error['code']=='NOT_READY'
    records=[json.loads(line) for line in (tmp_path/'receipts').read_text().splitlines()]
    assert len(records)==1 and records[0]['type']=='received'
