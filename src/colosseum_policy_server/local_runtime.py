"""Loopback Local Protocol runtime backed by installed policy plugins."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from importlib import metadata
import ipaddress
from pathlib import Path
import time
import traceback
from typing import Any, Mapping, Protocol
from urllib.parse import urlsplit

import numpy as np
import yaml

from . import colosseum_pb2 as pb
from .local_protocol import decode_control, encode_control


class PolicyBackend(Protocol):
    async def infer(self, model: "LocalModel", observation: pb.Observation) -> np.ndarray: ...


def load_backend(name: str, options: Mapping[str, Any]) -> PolicyBackend:
    entries = metadata.entry_points()
    entries = entries.select(group="colosseum_policy_server.backends") if hasattr(entries, "select") else entries.get("colosseum_policy_server.backends", [])
    matches = [entry for entry in entries if entry.name == name]
    if len(matches) != 1:
        raise ValueError(f"backend {name!r} is not installed")
    backend = matches[0].load()(dict(options))
    if not callable(getattr(backend, "infer", None)):
        raise ValueError("backend must define async infer")
    return backend


@dataclass(frozen=True)
class LocalModel:
    name: str; url: str; revision: str; runtime_profile: str; action_space: str; action_dim: int; control_hz: int; max_horizon: int; endpoint: str
    launcher: tuple[str, ...] = (); backend_options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LocalModel":
        required = ("name", "url", "revision", "runtime_profile", "action_space", "action_dim", "control_hz", "max_horizon", "endpoint")
        if not all(key in value for key in required):
            raise ValueError("local model is incomplete")
        launcher, options = value.get("launcher", []), value.get("backend_options", {})
        if not isinstance(launcher, list) or not all(isinstance(arg, str) and arg for arg in launcher) or not isinstance(options, Mapping):
            raise ValueError("invalid model launcher or backend options")
        model = cls(str(value["name"]), str(value["url"]), str(value["revision"]), str(value["runtime_profile"]), str(value["action_space"]), int(value["action_dim"]), int(value["control_hz"]), int(value["max_horizon"]), str(value["endpoint"]), tuple(launcher), dict(options))
        endpoint = urlsplit(model.endpoint)
        try: is_loopback = endpoint.hostname is not None and ipaddress.ip_address(endpoint.hostname).is_loopback
        except ValueError: is_loopback = endpoint.hostname == "localhost"
        expected_dimension = 8 if model.action_space in {"joint_position", "joint_velocity"} else 7
        if (not model.name or not model.url.startswith("https://huggingface.co/") or len(model.revision) != 40 or model.action_space not in {"joint_position", "joint_velocity", "cartesian_position"} or model.action_dim != expected_dimension or model.control_hz < 1 or model.max_horizon < 1 or endpoint.scheme not in {"ws", "wss"} or not is_loopback):
            raise ValueError("invalid model contract")
        return model

    def public_spec(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in ("url", "revision", "runtime_profile", "action_space", "action_dim", "control_hz", "max_horizon")}


@dataclass(frozen=True)
class RuntimeConfig:
    host: str; port: int; backend_name: str; backend_options: Mapping[str, Any]; log_dir: Path; start_timeout_seconds: int; models: Mapping[str, LocalModel]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RuntimeConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        backend = raw.get("backend") if isinstance(raw, Mapping) else None
        if not isinstance(backend, Mapping) or not isinstance(backend.get("name"), str) or not isinstance(backend.get("options", {}), Mapping):
            raise ValueError("runtime config requires backend.name and backend.options")
        host, port = str(raw.get("host", "127.0.0.1")), raw.get("port", 8000)
        try: is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError: is_loopback = host == "localhost"
        if not is_loopback or type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("runtime host must be loopback and port must be valid")
        rows = raw.get("models", [])
        if not isinstance(rows, list) or not rows:
            raise ValueError("models must be nonempty")
        models = [LocalModel.from_mapping(row) for row in rows if isinstance(row, Mapping)]
        if len(models) != len(rows) or len({model.name for model in models}) != len(models):
            raise ValueError("models must be complete and uniquely named")
        timeout = raw.get("start_timeout_seconds", 180)
        if type(timeout) is not int or timeout < 1:
            raise ValueError("start_timeout_seconds must be positive")
        return cls(host, port, backend["name"], dict(backend.get("options", {})), Path(str(raw.get("log_dir", "local-policy-logs"))), timeout, {model.name: model for model in models})


class LocalServiceSupervisor:
    def __init__(self, config: RuntimeConfig): self.config, self.process, self.active, self.log = config, None, None, None
    async def activate(self, model: LocalModel) -> None:
        if not model.launcher: return
        await self.stop(); self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self.log = (self.config.log_dir / f"{model.name}-{time.strftime('%Y%m%d-%H%M%S')}.log").open("xb")
        self.process = await asyncio.create_subprocess_exec(*model.launcher, stdout=self.log, stderr=asyncio.subprocess.STDOUT, start_new_session=True); self.active = model
        await asyncio.sleep(.25)
        if self.process.returncode is not None: raise RuntimeError(f"launcher exited with status {self.process.returncode}")
    async def stop(self) -> None:
        process, self.process, self.active = self.process, None, None
        if process is not None and process.returncode is None: process.terminate(); await process.wait()
        if self.log is not None: self.log.close(); self.log = None


class LocalPolicyRuntime:
    def __init__(self, config: RuntimeConfig, backend: PolicyBackend | None = None, supervisor: LocalServiceSupervisor | None = None):
        self.config, self.backend, self.supervisor, self.lock = config, backend or load_backend(config.backend_name, config.backend_options), supervisor or LocalServiceSupervisor(config), asyncio.Lock()
    async def _error(self, ws, run_id: str, code: str, message: str) -> None:
        payload = pb.Error(code=code, message=message, retryable=False).SerializeToString()
        await ws.send(pb.RelayFrame(protocol_version=1, type=pb.ERROR, session_id=run_id, payload=payload).SerializeToString())
    async def handle(self, ws) -> None:
        run_id = ""
        try:
            request = decode_control(await asyncio.wait_for(ws.recv(), 30))
            if request.get("type") == "capabilities":
                await ws.send(encode_control({"type":"capabilities", "protocol_version":1, "runtime_profiles":sorted(model.runtime_profile for model in self.config.models.values()), "verification_only":False})); return
            if request.get("type") != "prepare" or not isinstance(request.get("run_id"), str) or not isinstance(request.get("model"), Mapping): raise ValueError("invalid preparation")
            run_id = request["run_id"]; matches = [model for model in self.config.models.values() if model.public_spec() == dict(request["model"])]
            if len(matches) != 1: raise ValueError("unregistered model")
            model = matches[0]
            async with self.lock:
                try: await self.supervisor.activate(model)
                except Exception: print(traceback.format_exc(), flush=True); await self._error(ws, run_id, "MODEL_START_FAILED", "Model startup failed"); return
                await ws.send(encode_control({"type":"ready", "run_id":run_id, "preparation_id":request.get("preparation_id", ""), "model":model.public_spec(), "loaded":True, "verification_only":False}))
                await self._serve(ws, run_id, model)
        except Exception: print(traceback.format_exc(), flush=True); await self._error(ws, run_id, "INVALID_REQUEST", "Invalid local policy request")
    async def _serve(self, ws, run_id: str, model: LocalModel) -> None:
        sequence = 1
        while True:
            try:
                frame = pb.RelayFrame.FromString(await asyncio.wait_for(ws.recv(), 90))
                if frame.type == pb.SESSION_CLOSE and frame.session_id == run_id: return
                if frame.type != pb.OBSERVATION or frame.session_id != run_id or frame.sequence != sequence or frame.deadline_ms < 1: raise ValueError("invalid observation")
                observation = pb.Observation.FromString(frame.payload)
            except Exception: await self._error(ws, run_id, "INVALID_REQUEST", "Invalid local policy request"); return
            try: actions = np.asarray(await asyncio.wait_for(self.backend.infer(model, observation), frame.deadline_ms / 1000), dtype=np.float32)
            except asyncio.TimeoutError: await self._error(ws, run_id, "INFERENCE_TIMEOUT", "Local inference exceeded its deadline"); return
            except Exception: print(traceback.format_exc(), flush=True); await self._error(ws, run_id, "INFERENCE_FAILED", "Local inference failed"); return
            if actions.ndim != 2 or not 1 <= actions.shape[0] <= model.max_horizon or actions.shape[1] != model.action_dim or not np.isfinite(actions).all(): await self._error(ws, run_id, "INVALID_REQUEST", "Backend returned an invalid action array"); return
            plan = pb.ActionPlan(request_sequence=sequence, plan_id=sequence, start_step=observation.control_step, valid_until_step=observation.control_step + len(actions) - 1, control_hz=model.control_hz, actions=pb.Tensor(shape=actions.shape, dtype=pb.FLOAT32, data=actions.tobytes()))
            await ws.send(pb.RelayFrame(protocol_version=1, type=pb.ACTION_PLAN, session_id=run_id, sequence=sequence, payload=plan.SerializeToString()).SerializeToString()); sequence += 1
