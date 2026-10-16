import argparse
import asyncio
import base64
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from ...sdk import AsyncColosseumPolicySDK
from . import CONFIG_VERSION, PROMPT_VERSION, SOURCE_REVISION
from .controller import Controller, ControllerConfig
from .planner import Astra, _SYSTEM, REASONING_EFFORT, MAX_OUTPUT_TOKENS, FRAME_MAX_W, IMAGE_DETAIL

PHOTO_INSTRUCTION = (
    "Navigate to the location and viewing direction shown in the reference photo. "
    "Declare done only when stopped at that location with a matching view. "
    "Identifying an object alone does not complete this mission. "
    "In this controller, look_around means one 30-degree left turn followed by a fresh front image; "
    "no labelled side views are collected. There is no speculative execution or automatic motor tuning. "
)
TELEMETRY_FIELDS = ("orientation", "speed", "battery", "rpm_1", "rpm_2", "rpm_3", "rpm_4")


def revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("COLOSSEUM_POLICY_REVISION", "unversioned")


def policy_identity(planner, config):
    source_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
    return dict(name="Astra", model=planner.model, policy_revision=revision(),
                source_sha256=source_hash.hexdigest(), prompt_version=PROMPT_VERSION,
                prompt_sha256=hashlib.sha256(planner.system_prompt.encode()).hexdigest(),
                configuration_version=CONFIG_VERSION, controller=asdict(config),
                extracted_from_revision=SOURCE_REVISION, goal_types=["image"],
                planner_settings=dict(reasoning=REASONING_EFFORT, max_output_tokens=MAX_OUTPUT_TOKENS,
                    frame_max_width=FRAME_MAX_W, image_detail=IMAGE_DETAIL, service_tier=planner.service_tier,
                    streaming=planner.streaming, timeout_s=planner.timeout_s, decision_freshness_s=8))


def permitted_observation(observation):
    front = observation.image("front_image")
    goal = observation.image("goal_image")
    if front is None or goal is None:
        raise ValueError("Astra requires a front camera and reference photo")
    now = time.time_ns()
    if not 0 <= (now - front.capture_time_ns) / 1e9 <= 2:
        raise ValueError("Front camera is stale")
    telemetry = {}
    for field in TELEMETRY_FIELDS:
        value = observation.raw_state.get(field)
        if value is not None and np.asarray(value).size == 1:
            number = float(np.asarray(value).reshape(-1)[0])
            if np.isfinite(number):
                telemetry[field] = number
    rear = observation.image("rear_image")
    rear_fresh = rear is not None and 0 <= (now - rear.capture_time_ns) / 1e9 <= 2
    return front, goal, telemetry, rear_fresh


class AstraPolicy:
    def __init__(self, planner=None, config=ControllerConfig()):
        self.planner = planner or Astra()
        self.planner.system_prompt = _SYSTEM + '\nCOLOSSEUM PHOTO-GOAL CONTRACT: ' + PHOTO_INSTRUCTION
        self.config = config
        self.controller = Controller(config)
        self.identity = policy_identity(self.planner, config)
        self.pending = None
        self.session = None
        self.decision = None
        self.failed = ""
        self.planned_at = None

    async def reset(self, session):
        if self.pending:
            self.pending.cancel()
        self.pending = None
        self.session = session
        self.controller = Controller(self.config)
        self.decision = None
        self.failed = ""

    async def action(self, observation):
        if observation.session_id != self.session:
            await self.reset(observation.session_id)
        front, goal, telemetry, rear_fresh = permitted_observation(observation)
        if self.pending and self.pending.done():
            try:
                self.decision = self.pending.result()
            except Exception:
                self.failed = "Astra inference failed"
                self.decision = None
            self.pending = None
            if self.decision is None:
                self.failed = "Astra returned no valid decision"
            elif time.monotonic() - self.planned_at > 8:
                self.failed = "Astra decision exceeded the observation freshness budget"
            else:
                self.controller.load(self.decision)
        if not self.failed and not self.controller.moves and not self.controller.done and self.pending is None:
            self.planned_at = time.monotonic()
            self.pending = asyncio.create_task(asyncio.to_thread(
                self.planner.decide, base64.b64encode(front.data).decode(),
                PHOTO_INSTRUCTION + observation.instruction, base64.b64encode(goal.data).decode(),
                list(self.controller.history), str(telemetry.get("orientation", "unavailable"))))
        action = (0.0, 0.0) if self.failed else self.controller.step(time.monotonic(), telemetry, rear_available=rear_fresh)
        self.failed = self.failed or self.controller.error
        metadata = dict(policy=self.identity, phase="failed" if self.failed else "thinking" if self.pending else "executing",
                        error=self.failed, caption=self.decision.thought if self.decision else "",
                        decision=asdict(self.decision) if self.decision else None)
        # Three control ticks cover ordinary network jitter; the runner still
        # expires this chunk after 300 ms and applies its own command limits.
        return np.asarray([action] * 3, dtype=np.float32), self.controller.done, metadata

    async def close(self):
        if self.pending:
            self.pending.cancel()
        self.planner.close()


async def run(config_path):
    import yaml
    settings = yaml.safe_load(Path(config_path).read_text())
    model = settings.get('model') or os.environ.get('ASTRA_MODEL')
    if not model:
        raise ValueError('Set an exact Astra model identifier in the policy configuration')
    policy = AstraPolicy(Astra(model=model), ControllerConfig(**settings.get('controller', {})))
    token = settings['token']
    if token.startswith('env:'):
        token = os.environ[token[4:]]
    sdk = AsyncColosseumPolicySDK(router_url=settings['url'], token=token)
    sdk.policy_id = "Astra"
    sdk.policy_metadata = policy.identity
    sdk.policy_revision = policy.identity['source_sha256']
    sdk.action_spaces = ("base_velocity",)
    sdk.control_hz = 10
    sdk.max_horizon = 10
    try:
        async with sdk:
            while True:
                observation = await sdk.get_obs()
                try:
                    actions, done, metadata = await policy.action(observation)
                except ValueError as exc:
                    actions, done, metadata = np.zeros((1, 2), dtype=np.float32), False, {"error": str(exc)}
                await sdk.send_action(actions, observation=observation, done=done, decision_metadata=metadata)
    finally:
        await policy.close()


def main():
    parser = argparse.ArgumentParser(description="Run Astra image-goal navigation through RoboColosseum")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    asyncio.run(run(args.config))


if __name__ == '__main__':
    main()
