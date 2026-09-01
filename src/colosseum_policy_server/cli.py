from __future__ import annotations

import argparse
import asyncio
import os

import numpy as np

from . import colosseum_pb2 as pb
from .server import PolicyServer
from .tensors import tensor_from_numpy


class ZeroPolicy:
    def metadata(self) -> pb.SessionReady:
        return pb.SessionReady(
            policy_id="demo-zero",
            policy_revision="0.1.0",
            action_spaces=["joint_position"],
            control_hz=15,
            max_horizon=8,
        )

    def infer(self, observation: pb.Observation, sequence: int) -> pb.ActionPlan:
        horizon = 8
        return pb.ActionPlan(
            request_sequence=sequence,
            plan_id=sequence,
            start_step=observation.control_step,
            valid_until_step=observation.control_step + horizon - 1,
            control_hz=15,
            actions=tensor_from_numpy(np.zeros((horizon, 8), dtype=np.float32)),
        )

    def reset(self, request: pb.ResetRequest) -> None:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a zero-action Colosseum demo policy")
    parser.add_argument("--router-url", default="ws://127.0.0.1:8443")
    parser.add_argument("--server-id", default="policy-demo-01")
    args = parser.parse_args()
    token = os.environ.get("COLOSSEUM_POLICY_TOKEN")
    if not token:
        parser.error("COLOSSEUM_POLICY_TOKEN is required")
    server = PolicyServer(
        router_url=args.router_url,
        token=token,
        server_id=args.server_id,
        policy_id="demo-zero",
        policy=ZeroPolicy(),
    )
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
