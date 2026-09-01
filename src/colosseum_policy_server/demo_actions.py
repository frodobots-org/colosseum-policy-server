from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .sdk import Observation, RobotConfiguration


def make_gripper_action_chunk(
    observation: Observation,
    robot: RobotConfiguration,
    action_space: str,
    gripper_values: Iterable[float],
) -> np.ndarray:
    """Hold the current joints while applying a sequence of gripper commands."""

    if action_space != "joint_position":
        raise ValueError("the gripper demo requires the joint_position action space")
    if not robot.has_gripper:
        raise ValueError("the connected robot did not register a gripper")

    expected_dim = robot.action_spaces.get(action_space)
    if expected_dim != robot.joint_count + 1:
        raise ValueError(
            "the gripper demo expects joint_position action_dim to equal joint_count + 1"
        )

    joints = observation.state.joints
    if joints is None:
        raise ValueError("observation has no joint_position state")
    joints = np.asarray(joints, dtype=np.float32).reshape(-1)
    if joints.size != robot.joint_count:
        raise ValueError(
            f"joint_position contains {joints.size} values; robot registered {robot.joint_count} joints"
        )
    if not np.all(np.isfinite(joints)):
        raise ValueError("joint_position must contain finite values")

    gripper = np.asarray(tuple(gripper_values), dtype=np.float32)
    if gripper.ndim != 1 or gripper.size == 0:
        raise ValueError("gripper pattern must contain at least one value")
    if not np.all(np.isfinite(gripper)):
        raise ValueError("gripper pattern must contain finite values")

    chunk = np.empty((gripper.size, expected_dim), dtype=np.float32)
    chunk[:, : robot.joint_count] = joints
    chunk[:, robot.joint_count] = gripper
    return chunk
