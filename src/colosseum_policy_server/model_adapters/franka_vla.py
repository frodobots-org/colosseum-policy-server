"""Franka model adapters for user-managed VLA inference services.

This module contains no model implementation, weights, hardware imports, or
machine-specific paths.  It translates the Local Protocol observation into
the documented wire format expected by separately installed model services.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from .. import colosseum_pb2 as pb
from ..backends.base import LocalModel
from .observation import _rgb, _state
from ..numpy_wire import _json_array, _json_decode, _msgpack_default, _msgpack_object, _groot_default


_ADAPTERS = frozenset({"molmoact2", "pi05_lerobot", "groot_n17", "g05", "lap_3b"})


@dataclass(frozen=True)
class DroidObservation:
    external: np.ndarray
    wrist: np.ndarray
    joints: np.ndarray
    gripper: np.ndarray
    cartesian: np.ndarray
    instruction: str


def create_backend(options: Mapping[str, Any]) -> "FrankaVLAAdapter":
    """Policy plugin factory registered as ``droid``."""
    return FrankaVLAAdapter(options)


class FrankaVLAAdapter:
    """Call a loopback model service selected by ``backend_options.adapter``."""

    supported_robot_types = ('franka',)

    async def start_session(self, model, request):
        from .robots import session_robot
        if session_robot(model, request, {}) not in self.supported_robot_types:
            raise ValueError('Franka VLA model adapter does not support this robot_type')
        self.validate(model)

    def __init__(self, options: Mapping[str, Any]) -> None:
        options = dict(options)
        self.external_sensor = _nonempty(options.get("external_sensor", "head_image"), "external_sensor")
        self.wrist_sensor = _nonempty(options.get("wrist_sensor", "left_image"), "wrist_sensor")
        if self.external_sensor == self.wrist_sensor:
            raise ValueError("external_sensor and wrist_sensor must differ")
        self.http_timeout = _positive_float(options.get("http_timeout_seconds", 30), "http_timeout_seconds")

    def validate(self, model: LocalModel) -> None:
        adapter = model.backend_options.get("adapter")
        if adapter not in _ADAPTERS:
            raise ValueError("backend_options.adapter is not a supported DROID adapter")
        expected_space = {"pi05_lerobot": "joint_velocity", "lap_3b": "cartesian_position"}.get(adapter, "joint_position")
        expected_dim = 7 if adapter == "lap_3b" else 8
        if model.action_space != expected_space or model.action_dim != expected_dim:
            raise ValueError("model action space does not match the selected adapter")

    async def infer(self, model: LocalModel, observation: pb.Observation) -> np.ndarray:
        self.validate(model)
        adapter = model.backend_options["adapter"]
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
