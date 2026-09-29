import json

import numpy as np
import pytest

from colosseum_policy_server.pi05_contract import (
    ACTION_DIM,
    ACTION_HORIZON,
    DROID_DIM,
    LEROBOT_CHECKPOINT,
    LEROBOT_MODEL_SOURCE,
    QuantileStats,
    build_openpi_droid_prompt,
    droid_actions_from_lerobot,
    normalize_droid_state,
    prepare_droid_request,
    validate_server_metadata,
)


def stats() -> QuantileStats:
    state_q01 = np.r_[np.full(8, -1.0), np.zeros(24)].astype(np.float32)
    state_q99 = np.r_[np.full(8, 1.0), np.zeros(24)].astype(np.float32)
    action_q01 = np.r_[np.full(7, -1.0), np.array([0.0]), np.zeros(24)].astype(np.float32)
    action_q99 = np.r_[np.full(7, 1.0), np.array([1.0]), np.zeros(24)].astype(np.float32)
    return QuantileStats(state_q01, state_q99, action_q01, action_q99)


def payload() -> dict:
    image = np.full((12, 16, 3), 127, dtype=np.uint8)
    return {
        "observation/exterior_image_1_left": image,
        "observation/wrist_image_left": image.copy(),
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/gripper_position": np.array([0.5], dtype=np.float32),
        "prompt": "close the prop laptop",
    }


def metadata() -> dict:
    return {
        "checkpoint": LEROBOT_CHECKPOINT,
        "model_source": LEROBOT_MODEL_SOURCE,
        "state_dim": 32,
        "action_dim": 32,
        "droid_output_dim": 8,
        "action_horizon": 15,
        "action_space": "joint_velocity",
        "gripper_action_space": "position",
        "right_wrist_image": "masked",
    }


def test_droid_request_pads_state_and_keeps_right_wrist_absent():
    prepared = prepare_droid_request(payload(), stats())
    assert prepared.normalized_state.shape == (32,)
    np.testing.assert_array_equal(prepared.normalized_state[8:], np.zeros(24))
    assert "close the prop laptop" in prepared.prompt
    assert prepared.exterior_image.dtype == np.uint8


def test_same_noise_reference_contract_is_15_by_32_then_physical_15_by_8():
    raw = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
    raw[:, 7] = 1.0  # normalized gripper -> physical one under the fixture stats
    converted = droid_actions_from_lerobot(raw, stats())
    assert converted.shape == (ACTION_HORIZON, DROID_DIM)
    np.testing.assert_allclose(converted[:, :7], 0)
    np.testing.assert_allclose(converted[:, 7], 1)


@pytest.mark.parametrize("bad", [np.full((15, 8), 0.0), np.full((15, 32), np.nan)])
def test_raw_lerobot_action_contract_rejects_wrong_shape_and_nonfinite_values(bad):
    with pytest.raises(ValueError):
        droid_actions_from_lerobot(bad, stats())


def test_raw_lerobot_action_contract_rejects_nonzero_padding_after_unnormalization():
    raw = np.zeros((15, 32), dtype=np.float32)
    altered = QuantileStats(stats().state_q01, stats().state_q99, np.zeros(32), np.r_[np.ones(8), np.ones(24)])
    with pytest.raises(ValueError, match="padding"):
        droid_actions_from_lerobot(raw, altered)


def test_server_identity_must_be_the_isolated_lerobot_candidate():
    validate_server_metadata(metadata())
    invalid = metadata()
    invalid["checkpoint"] = "official-openpi-pi05_droid"
    with pytest.raises(ValueError, match="checkpoint"):
        validate_server_metadata(invalid)


def test_prompt_requires_exact_32_value_state_and_nonempty_task():
    with pytest.raises(ValueError):
        build_openpi_droid_prompt("", np.zeros(32))
    with pytest.raises(ValueError):
        build_openpi_droid_prompt("close", np.zeros(8))
