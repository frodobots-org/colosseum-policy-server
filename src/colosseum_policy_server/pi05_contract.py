"""Traceable DROID <-> LeRobot π0.5 conversion with no hardware dependency.

The converted LeRobot checkpoint retains OpenPI's 32-value padded state/action
layout.  DROID supplies and consumes only the first eight physical values:
seven joint values plus a gripper position.  The remaining dimensions are
padding and must remain zero after quantile de-normalisation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


LEROBOT_CHECKPOINT = "lerobot-pi05_droid-72824c0a93f00ce5bb8bedb7feb58953ba1da364"
LEROBOT_MODEL_SOURCE = "official-openpi-pi05-droid-pytorch-conversion"
STATE_DIM = 32
ACTION_DIM = 32
DROID_DIM = 8
ACTION_HORIZON = 15


@dataclass(frozen=True)
class QuantileStats:
    state_q01: np.ndarray
    state_q99: np.ndarray
    action_q01: np.ndarray
    action_q99: np.ndarray


@dataclass(frozen=True)
class PreparedPi05Input:
    normalized_state: np.ndarray
    prompt: str
    exterior_image: np.ndarray
    left_wrist_image: np.ndarray


def _vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (STATE_DIM,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {STATE_DIM}-value vector")
    return result


def load_quantile_stats(path: str | Path) -> QuantileStats:
    """Load the original OpenPI DROID q01/q99 statistics without fallbacks."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))["norm_stats"]
        state = document["state"]
        actions = document["actions"]
        stats = QuantileStats(
            _vector(state["q01"], "state.q01"),
            _vector(state["q99"], "state.q99"),
            _vector(actions["q01"], "actions.q01"),
            _vector(actions["q99"], "actions.q99"),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid OpenPI DROID quantile statistics: {path}") from exc
    if np.any(stats.state_q99[:DROID_DIM] <= stats.state_q01[:DROID_DIM]):
        raise ValueError("state q99 must exceed q01 for every DROID state value")
    if np.any(stats.action_q99[:7] <= stats.action_q01[:7]):
        raise ValueError("action q99 must exceed q01 for every DROID joint velocity")
    if not np.allclose(stats.state_q01[DROID_DIM:], 0) or not np.allclose(stats.state_q99[DROID_DIM:], 0):
        raise ValueError("OpenPI DROID state padding statistics must be zero")
    if not np.allclose(stats.action_q01[DROID_DIM:], 0) or not np.allclose(stats.action_q99[DROID_DIM:], 0):
        raise ValueError("OpenPI DROID action padding statistics must be zero")
    return stats


def _quantile_normalize(values: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    denominator = q99 - q01
    result = np.zeros_like(values, dtype=np.float32)
    varying = denominator != 0
    result[varying] = 2.0 * (values[varying] - q01[varying]) / denominator[varying] - 1.0
    if np.any(~varying & (values != q01)):
        raise ValueError("cannot normalize a non-padding value with zero quantile range")
    return result


def _quantile_unnormalize(values: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    return ((values + 1.0) * 0.5 * (q99 - q01) + q01).astype(np.float32)


def normalize_droid_state(state8: Any, stats: QuantileStats) -> np.ndarray:
    raw = np.asarray(state8, dtype=np.float32).reshape(-1)
    if raw.shape != (DROID_DIM,) or not np.isfinite(raw).all():
        raise ValueError("DROID π0.5 state must be seven finite joints plus one finite gripper")
    padded = np.zeros(STATE_DIM, dtype=np.float32)
    padded[:DROID_DIM] = raw
    return _quantile_normalize(padded, stats.state_q01, stats.state_q99)


def build_openpi_droid_prompt(instruction: str, normalized_state: Any) -> str:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("π0.5 instruction must be nonempty")
    state = _vector(normalized_state, "normalized state")
    bins = np.linspace(-1, 1, 257, dtype=np.float32)[:-1]
    tokens = np.digitize(state, bins=bins) - 1
    return f"Task: {instruction.strip()}, State: {' '.join(map(str, tokens))};\nAction: "


def prepare_droid_request(payload: Mapping[str, Any], stats: QuantileStats) -> PreparedPi05Input:
    """Validate one existing OpenPI DROID request and prepare LeRobot inputs.

    The caller deliberately supplies only the exterior and left wrist image.  An
    omitted right wrist is represented by PI05Policy's native missing-image mask.
    """
    try:
        exterior = np.asarray(payload["observation/exterior_image_1_left"])
        wrist = np.asarray(payload["observation/wrist_image_left"])
        joints = np.asarray(payload["observation/joint_position"], dtype=np.float32).reshape(-1)
        gripper = np.asarray(payload["observation/gripper_position"], dtype=np.float32).reshape(-1)
        instruction = payload["prompt"]
    except (KeyError, TypeError) as exc:
        raise ValueError("OpenPI DROID request is missing a required field") from exc
    for name, image in (("exterior", exterior), ("left wrist", wrist)):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or image.size == 0:
            raise ValueError(f"{name} image must be nonempty uint8 HWC RGB")
    if joints.shape != (7,) or gripper.shape != (1,):
        raise ValueError("OpenPI DROID request must contain seven joints and one gripper position")
    normalized = normalize_droid_state(np.concatenate((joints, gripper)), stats)
    return PreparedPi05Input(normalized, build_openpi_droid_prompt(instruction, normalized), exterior, wrist)


def droid_actions_from_lerobot(raw_actions: Any, stats: QuantileStats) -> np.ndarray:
    """Convert normalized ``15×32`` PI05 output to physical DROID ``15×8``."""
    raw = np.asarray(raw_actions, dtype=np.float32)
    if raw.shape != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(f"LeRobot π0.5 output must have shape ({ACTION_HORIZON}, {ACTION_DIM})")
    if not np.isfinite(raw).all():
        raise ValueError("LeRobot π0.5 output must be finite")
    physical = _quantile_unnormalize(raw, stats.action_q01, stats.action_q99)
    if not np.allclose(physical[:, DROID_DIM:], 0.0, atol=1e-6):
        raise ValueError("LeRobot π0.5 output changed OpenPI DROID padding dimensions")
    droid_actions = physical[:, :DROID_DIM]
    if np.any(np.abs(droid_actions[:, :7]) > 1.0):
        raise ValueError("LeRobot π0.5 normalized joint velocity is outside [-1, 1]")
    return droid_actions


def validate_server_metadata(metadata: Mapping[str, Any]) -> None:
    expected = {
        "checkpoint": LEROBOT_CHECKPOINT,
        "model_source": LEROBOT_MODEL_SOURCE,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "droid_output_dim": DROID_DIM,
        "action_horizon": ACTION_HORIZON,
        "action_space": "joint_velocity",
        "gripper_action_space": "position",
        "right_wrist_image": "masked",
    }
    missing_or_wrong = [name for name, value in expected.items() if metadata.get(name) != value]
    if missing_or_wrong:
        raise ValueError("pi05_lerobot server metadata is invalid: " + ", ".join(missing_or_wrong))
