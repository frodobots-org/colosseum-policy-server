from __future__ import annotations

import numpy as np
import pytest

from colosseum_policy_server import Observation, RobotConfiguration
from colosseum_policy_server.demo_actions import make_gripper_action_chunk


def observation(joints: np.ndarray) -> Observation:
    return Observation(
        session_id="session-1",
        sequence=1,
        deadline_ms=1000,
        robot_time_ns=1,
        control_step=2,
        instruction="test",
        raw_state={"joint_position": joints},
        images=(),
        _received_monotonic_ns=1,
    )


def robot(*, has_gripper: bool = True, action_dim: int = 4) -> RobotConfiguration:
    return RobotConfiguration(
        robot_type="test-arm",
        joint_count=3,
        has_gripper=has_gripper,
        control_hz=15,
        action_spaces={"joint_position": action_dim},
    )


def test_gripper_chunk_holds_current_joints():
    joints = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    chunk = make_gripper_action_chunk(
        observation(joints), robot(), "joint_position", (0, 1, 0, 1)
    )

    assert chunk.shape == (4, 4)
    np.testing.assert_allclose(chunk[:, :3], np.tile(joints, (4, 1)))
    np.testing.assert_allclose(chunk[:, 3], [0, 1, 0, 1])


def test_gripper_chunk_requires_registered_gripper():
    with pytest.raises(ValueError, match="did not register a gripper"):
        make_gripper_action_chunk(
            observation(np.zeros(3)), robot(has_gripper=False), "joint_position", (0,)
        )


def test_gripper_chunk_validates_joint_count_and_action_space():
    with pytest.raises(ValueError, match="contains 2 values"):
        make_gripper_action_chunk(
            observation(np.zeros(2)), robot(), "joint_position", (0,)
        )
    with pytest.raises(ValueError, match="requires the joint_position"):
        make_gripper_action_chunk(
            observation(np.zeros(3)), robot(), "joint_velocity", (0,)
        )
    with pytest.raises(ValueError, match=r"joint_count \+ 1"):
        make_gripper_action_chunk(
            observation(np.zeros(3)), robot(action_dim=3), "joint_position", (0,)
        )
