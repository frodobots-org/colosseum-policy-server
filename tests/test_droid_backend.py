import base64
from dataclasses import replace

import numpy as np
import pytest

from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.droid_backend import DroidBackend, _json_decode
from colosseum_policy_server.local_runtime import LocalModel
from colosseum_policy_server.tensors import tensor_from_numpy


def model(adapter, action_space="joint_position", action_dim=8):
    return LocalModel.from_mapping({
        "name": adapter,
        "url": "https://huggingface.co/example/model",
        "revision": "a" * 40,
        "runtime_profile": f"{adapter}-v1",
        "action_space": action_space,
        "action_dim": action_dim,
        "control_hz": 15,
        "max_horizon": 15,
        "endpoint": "ws://127.0.0.1:9100",
        "backend_options": {"adapter": adapter},
    })


def observation():
    image = np.full((2, 3, 3), 9, dtype=np.uint8)
    return pb.Observation(
        instruction="close",
        state={
            "joint_position": tensor_from_numpy(np.arange(7, dtype=np.float32)),
            "gripper_position": tensor_from_numpy(np.array([0.25], dtype=np.float32)),
            "cartesian_position": tensor_from_numpy(np.arange(6, dtype=np.float32)),
        },
        sensors=[
            pb.Image(sensor_id="head_image", encoding=pb.RAW_RGB, width=3, height=2, data=image.tobytes()),
            pb.Image(sensor_id="left_image", encoding=pb.RAW_RGB, width=3, height=2, data=image.tobytes()),
        ],
    )


def test_builds_each_droid_request_schema_without_hardware():
    backend = DroidBackend({})
    source = backend._observation(observation(), require_cartesian=True)
    pi05 = backend._payload("pi05_lerobot", source)
    g05 = backend._payload("g05", source)
    assert pi05["observation/joint_position"].shape == (7,)
    assert g05["images"]["exterior_image"].shape == (3, 2, 3)
    assert g05["state"]["right_gripper"].tolist() == [0.75]
    pytest.importorskip("scipy")
    assert backend._payload("groot_n17", source)["state"]["eef_9d"].shape == (1, 1, 9)
    assert backend._payload("lap_3b", source)["observation"]["state"].shape == (10,)


def test_converts_joint_and_gripper_responses():
    backend = DroidBackend({})
    source = backend._observation(observation(), require_cartesian=False)
    pi05 = backend._actions("pi05_lerobot", {"actions": np.zeros((2, 8), dtype=np.float32)}, source, 8)
    groot = backend._actions("groot_n17", {"joint_position": np.zeros((1, 2, 7)), "gripper_position": np.ones((1, 2, 1))}, source, 8)
    g05 = backend._actions("g05", {"action": {"right_arm": np.zeros(7), "right_gripper": np.array([0.2])}}, source, 8)
    assert pi05.shape == (2, 8)
    assert groot.shape == (2, 8)
    assert g05[0, -1] == pytest.approx(0.8)


def test_molmo_array_decoder_rejects_invalid_payload():
    with pytest.raises(ValueError):
        _json_decode({"__numpy__": "bad", "dtype": "<f4", "shape": [1, 8]})
    valid = np.zeros((1, 8), dtype=np.float32)
    decoded = _json_decode({"__numpy__": base64.b64encode(valid.tobytes()).decode("ascii"), "dtype": valid.dtype.str, "shape": [1, 8]})
    assert np.array_equal(decoded, valid)


@pytest.mark.asyncio
async def test_rejects_unknown_adapter_before_connecting():
    item = replace(model("pi05_lerobot"), backend_options={"adapter": "unknown"})
    with pytest.raises(ValueError):
        await DroidBackend({}).infer(item, observation())
