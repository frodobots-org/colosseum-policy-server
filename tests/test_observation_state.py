from __future__ import annotations

import numpy as np

from colosseum_policy_server import ImageFrame, Observation


def test_observation_state_exposes_standard_policy_inputs():
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    joints = np.arange(7, dtype=np.float32)
    gripper = np.array([0.5], dtype=np.float32)
    observation = Observation(
        session_id="session-1",
        sequence=1,
        deadline_ms=1000,
        robot_time_ns=1,
        control_step=1,
        instruction="test",
        raw_state={"joint_position": joints, "gripper_position": gripper},
        images=(
            ImageFrame(
                image_id="head_image",
                encoding="RAW_RGB",
                capture_time_ns=1,
                width=3,
                height=2,
                data=rgb.tobytes(),
            ),
        ),
        _received_monotonic_ns=1,
    )

    np.testing.assert_array_equal(observation.state.joints, joints)
    np.testing.assert_array_equal(observation.state.gripper, gripper)
    np.testing.assert_array_equal(observation.state.head_image, rgb)
    assert observation.state.left_image is None
    assert observation.state.right_image is None
