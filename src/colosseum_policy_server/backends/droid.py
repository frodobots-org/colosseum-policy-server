"""DROID observation adapters for user-managed local model services.

This module contains no model implementation, weights, hardware imports, or
machine-specific paths.  It translates the Local Protocol observation into
the documented wire format expected by separately installed model services.
"""
from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from .. import colosseum_pb2 as pb
from .base import LocalModel
from ..tensors import tensor_to_numpy


_ADAPTERS = frozenset({"molmoact2", "pi05_lerobot", "groot_n17", "g05", "lap_3b"})


@dataclass(frozen=True)
class DroidObservation:
    external: np.ndarray
    wrist: np.ndarray
    joints: np.ndarray
    gripper: np.ndarray
    cartesian: np.ndarray
    instruction: str


def create_backend(options: Mapping[str, Any]) -> "DroidBackend":
    """Policy plugin factory registered as ``droid``."""
    return DroidBackend(options)


class DroidBackend:
    """Call a loopback model service selected by ``backend_options.adapter``."""

    def __init__(self, options: Mapping[str, Any]) -> None:
        options = dict(options)
        self.external_sensor = _nonempty(options.get("external_sensor", "head_image"), "external_sensor")
        self.wrist_sensor = _nonempty(options.get("wrist_sensor", "left_image"), "wrist_sensor")
        if self.external_sensor == self.wrist_sensor:
            raise ValueError("external_sensor and wrist_sensor must differ")
        self.http_timeout = _positive_float(options.get("http_timeout_seconds", 30), "http_timeout_seconds")

    async def infer(self, model: LocalModel, observation: pb.Observation) -> np.ndarray:
        adapter = model.backend_options.get("adapter")
        if adapter not in _ADAPTERS:
            raise ValueError("backend_options.adapter is not a supported DROID adapter")
        expected_space = {"pi05_lerobot": "joint_velocity", "lap_3b": "cartesian_position"}.get(adapter, "joint_position")
        expected_dim = 7 if adapter == "lap_3b" else 8
        if model.action_space != expected_space or model.action_dim != expected_dim:
            raise ValueError("model action space does not match the selected adapter")
        source = self._observation(observation, require_cartesian=adapter in {"groot_n17", "lap_3b"})
        if adapter == "molmoact2":
            return await asyncio.to_thread(self._molmo, model, source)
        payload = self._payload(adapter, source)
        if adapter == "groot_n17":
            raw = await asyncio.to_thread(self._groot, model.endpoint, payload)
        else:
            _, raw = await self._websocket(model.endpoint, payload)
        return self._actions(adapter, raw, source, model.action_dim)

    def _observation(self, observation: pb.Observation, *, require_cartesian: bool) -> DroidObservation:
        if not observation.instruction.strip():
            raise ValueError("observation instruction is empty")
        images = {image.sensor_id: image for image in observation.sensors}
        external = _rgb(images.get(self.external_sensor), self.external_sensor)
        wrist = _rgb(images.get(self.wrist_sensor), self.wrist_sensor)
        joints = _state(observation, "joint_position", (7,))
        gripper = _state(observation, "gripper_position", (1,))
        cartesian = _state(observation, "cartesian_position", (6,)) if require_cartesian else np.empty(0, np.float32)
        return DroidObservation(external, wrist, joints, gripper, cartesian, observation.instruction)

    def _payload(self, adapter: str, source: DroidObservation) -> dict[str, Any]:
        if adapter == "pi05_lerobot":
            return {
                "observation/exterior_image_1_left": source.external,
                "observation/wrist_image_left": source.wrist,
                "observation/joint_position": source.joints,
                "observation/gripper_position": source.gripper,
                "prompt": source.instruction,
            }
        if adapter == "groot_n17":
            xyz, rot6d = _pose(source.cartesian)
            return {
                "video": {"exterior_image_1_left": source.external[None, None], "wrist_image_left": source.wrist[None, None]},
                "state": {"eef_9d": np.concatenate((xyz, rot6d))[None, None], "joint_position": source.joints[None, None], "gripper_position": source.gripper[None, None]},
                "language": {"annotation.language.language_instruction": [[source.instruction]]},
            }
        if adapter == "g05":
            wrist = np.ascontiguousarray(source.wrist.transpose(2, 0, 1))
            return {
                "images": {"exterior_image": np.ascontiguousarray(source.external.transpose(2, 0, 1)), "wrist_image": wrist, "dummy_wrist_right": wrist},
                "state": {"right_arm": source.joints, "right_gripper": 1.0 - source.gripper},
                "task": source.instruction,
                "frequency": 15,
                "embodiment_type": "Droid_Franka",
            }
        xyz, rot6d = _pose(source.cartesian)
        cartesian = np.concatenate((xyz, rot6d)).astype(np.float32)
        return {
            "observation": {"base_0_rgb": source.external, "left_wrist_0_rgb": source.wrist, "cartesian_position": cartesian, "gripper_position": source.gripper, "joint_position": source.joints, "state": np.concatenate((cartesian, source.gripper)), "euler": source.cartesian[3:]},
            "prompt": source.instruction,
        }

    def _molmo(self, model: LocalModel, source: DroidObservation) -> np.ndarray:
        payload = {
            "external_cam": _json_array(source.external),
            "wrist_cam": _json_array(source.wrist),
            "instruction": source.instruction,
            "state": _json_array(np.concatenate((source.joints, source.gripper))),
            "num_steps": int(model.backend_options.get("num_steps", 10)),
        }
        request = Request(f"{model.endpoint.rstrip('/')}/act", data=json.dumps(payload, separators=(",", ":")).encode(), headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.http_timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise ValueError(f"model service returned HTTP {exc.code}") from exc
        except (URLError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("model service returned an invalid response") from exc
        if isinstance(raw, Mapping) and "error" in raw:
            raise ValueError("model service reported an inference error")
        if not isinstance(raw, Mapping) or "actions" not in raw:
            raise ValueError("model service response is missing actions")
        return _finite_matrix(_json_decode(raw["actions"]), model.action_dim)

    async def _websocket(self, endpoint: str, payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        try:
            import msgpack
            import websockets
        except ImportError as exc:
            raise RuntimeError("install colosseum-policy-server[droid] for WebSocket adapters") from exc
        def pack(value: Any) -> bytes:
            return msgpack.packb(value, default=_msgpack_default)
        def unpack(value: bytes) -> Any:
            return msgpack.unpackb(value, object_hook=_msgpack_object, raw=False, strict_map_key=False)
        async with websockets.connect(endpoint, compression=None, max_size=None, open_timeout=30, close_timeout=5) as socket:
            greeting = unpack(await asyncio.wait_for(socket.recv(), 30))
            await socket.send(pack(dict(payload)))
            response = unpack(await asyncio.wait_for(socket.recv(), 240))
        if not isinstance(greeting, Mapping) or not isinstance(response, Mapping):
            raise ValueError("model service response must be a mapping")
        return dict(greeting), dict(response)

    @staticmethod
    def _groot(endpoint: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            import msgpack
            import zmq
        except ImportError as exc:
            raise RuntimeError("install colosseum-policy-server[droid] for GR00T adapters") from exc
        context, socket = zmq.Context(), None
        try:
            socket = context.socket(zmq.REQ)
            socket.setsockopt(zmq.RCVTIMEO, 240000)
            socket.setsockopt(zmq.SNDTIMEO, 30000)
            socket.connect(endpoint)
            socket.send(msgpack.packb({"endpoint": "get_action", "data": {"observation": dict(payload), "options": None}}, default=_groot_default, use_bin_type=True))
            raw = msgpack.unpackb(socket.recv(), object_hook=_msgpack_object, raw=False, strict_map_key=False)
        finally:
            if socket is not None:
                socket.close(linger=0)
            context.term()
        if isinstance(raw, Mapping) and "error" in raw:
            raise ValueError("model service reported an inference error")
        if isinstance(raw, Mapping) and "action" in raw:
            raw = raw["action"]
        elif isinstance(raw, (list, tuple)) and raw:
            raw = raw[0]
        if not isinstance(raw, Mapping):
            raise ValueError("GR00T response is missing an action mapping")
        return dict(raw)

    def _actions(self, adapter: str, raw: Mapping[str, Any], source: DroidObservation, action_dim: int) -> np.ndarray:
        if "error" in raw:
            raise ValueError("model service reported an inference error")
        if adapter == "pi05_lerobot":
            actions = _finite_matrix(raw.get("actions"), action_dim)
            if np.any(np.abs(actions[:, :7]) > 1):
                raise ValueError("normalized joint velocities must be in [-1, 1]")
            return actions
        if adapter == "groot_n17":
            joints, gripper = raw.get("joint_position"), raw.get("gripper_position")
            joints = np.asarray(joints, dtype=np.float32)
            gripper = np.asarray(gripper, dtype=np.float32)
            if joints.ndim == 3 and joints.shape[0] == 1:
                joints = joints[0]
            if gripper.ndim == 3 and gripper.shape[0] == 1:
                gripper = gripper[0]
            return _finite_matrix(np.concatenate((joints, gripper), axis=-1), action_dim)
        if adapter == "g05":
            action = raw.get("action")
            if not isinstance(action, Mapping):
                raise ValueError("G05 response is missing action")
            arm = np.asarray(action.get("right_arm", action.get("joint_position")), dtype=np.float32).reshape(-1)
            value = action.get("right_gripper", action.get("gripper"))
            gripper = source.gripper if value is None else 1.0 - np.asarray(value, dtype=np.float32).reshape(-1)
            if arm.shape != (7,) or gripper.shape != (1,) or not np.isfinite(np.r_[arm, gripper]).all():
                raise ValueError("G05 response requires seven finite joints and one finite gripper value")
            return _finite_matrix(np.concatenate((arm, np.clip(gripper, 0, 1)))[None], action_dim)
        actions = _finite_matrix(raw.get("actions"), action_dim)
        return _lap_absolute(actions, source.cartesian)


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _positive_float(value: Any, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _rgb(image: pb.Image | None, sensor_id: str) -> np.ndarray:
    if image is None or image.encoding != pb.RAW_RGB or image.width < 1 or image.height < 1 or len(image.data) != image.width * image.height * 3:
        raise ValueError(f"invalid RGB sensor {sensor_id}")
    return np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, 3).copy()


def _state(observation: pb.Observation, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if name not in observation.state:
        raise ValueError(f"observation is missing {name}")
    value = np.asarray(tensor_to_numpy(observation.state[name]), dtype=np.float32).reshape(-1)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"invalid observation state {name}")
    return value


def _json_array(value: np.ndarray) -> dict[str, Any]:
    value = np.ascontiguousarray(value)
    return {"__numpy__": base64.b64encode(value.tobytes()).decode("ascii"), "dtype": value.dtype.str, "shape": list(value.shape)}


def _json_decode(value: Any) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value)
    if not isinstance(value, Mapping) or set(value) != {"__numpy__", "dtype", "shape"}:
        raise ValueError("model actions have an invalid array encoding")
    dtype, shape = np.dtype(value["dtype"]), value["shape"]
    if dtype.kind not in "biuf":
        raise ValueError("model actions require numeric array data")
    if not isinstance(shape, list) or not shape or any(type(item) is not int or item < 1 for item in shape) or int(np.prod(shape)) > 1_000_000:
        raise ValueError("model actions have an invalid shape")
    try:
        data = base64.b64decode(value["__numpy__"], validate=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("model actions have invalid base64") from exc
    if len(data) != int(np.prod(shape)) * dtype.itemsize:
        raise ValueError("model actions have invalid byte length")
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def _finite_matrix(value: Any, action_dim: int) -> np.ndarray:
    if value is None:
        raise ValueError("model service response is missing actions")
    actions = np.asarray(value, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != action_dim or not np.isfinite(actions).all():
        raise ValueError("model actions must be a finite two-dimensional action array")
    return actions


def _pose(cartesian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise RuntimeError("install colosseum-policy-server[droid] for Cartesian adapters") from exc
    matrix = Rotation.from_euler("xyz", cartesian[3:]).as_matrix()
    return cartesian[:3], np.concatenate((matrix[:, 0], matrix[:, 1])).astype(np.float32)


def _lap_absolute(actions: np.ndarray, cartesian: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise RuntimeError("install colosseum-policy-server[droid] for Cartesian adapters") from exc
    result = actions.copy()
    result[:, :3] += cartesian[:3]
    result[:, 3:6] = (Rotation.from_euler("xyz", cartesian[3:]) * Rotation.from_euler("xyz", actions[:, 3:6])).as_euler("xyz")
    result[:, 6] = 1.0 - result[:, 6]
    return result


def _msgpack_default(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in "OVc":
        raise ValueError("unsupported array dtype")
    if isinstance(value, np.ndarray):
        return {b"__ndarray__": True, b"data": value.tobytes(), b"dtype": value.dtype.str, b"shape": value.shape}
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"cannot MessagePack {type(value).__name__}")


def _msgpack_object(value: dict[Any, Any]) -> Any:
    value = {key.decode("ascii") if isinstance(key, bytes) else key: item for key, item in value.items()}
    if "nd" in value:
        dtype = np.dtype(value["type"])
        if dtype.kind in "OVc":
            raise ValueError("unsupported array dtype")
        array = np.frombuffer(value["data"], dtype=dtype)
        return array.reshape(tuple(value["shape"])).copy() if value["nd"] else array[0]
    if "__ndarray__" in value:
        if np.dtype(value["dtype"]).kind in "OVc":
            raise ValueError("unsupported array dtype")
        return np.frombuffer(value["data"], dtype=np.dtype(value["dtype"])).reshape(tuple(value["shape"])).copy()
    if "__npgeneric__" in value:
        return np.dtype(value["dtype"]).type(value["data"])
    return value


def _groot_default(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        if array.dtype.kind in "OVc":
            raise ValueError("unsupported array dtype")
        return {b"nd": isinstance(value, np.ndarray), b"type": array.dtype.str,
                b"shape": array.shape, b"data": array.tobytes()}
    raise TypeError("unsupported MessagePack value")
