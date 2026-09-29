"""Robot-independent contract for local policy backend plugins."""
from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
from typing import Any, Mapping, Protocol
from urllib.parse import urlsplit

import numpy as np

from .. import colosseum_pb2 as pb


class PolicyBackend(Protocol):
    async def infer(self, model: "LocalModel", observation: pb.Observation) -> np.ndarray:
        """Return finite (horizon, action_dim) actions in the configured action space."""
        ...


@dataclass(frozen=True)
class LocalModel:
    name: str; url: str; revision: str; action_space: str; action_dim: int; control_hz: int; max_horizon: int; endpoint: str
    launcher: tuple[str, ...] = (); backend_options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LocalModel":
        required = ("name", "url", "revision", "action_space", "action_dim", "control_hz", "max_horizon", "endpoint")
        if not all(key in value for key in required):
            raise ValueError("local model is incomplete")
        launcher, options = value.get("launcher", []), value.get("backend_options", {})
        if not isinstance(launcher, list) or not all(isinstance(arg, str) and arg for arg in launcher) or not isinstance(options, Mapping):
            raise ValueError("invalid model launcher or backend options")
        model = cls(str(value["name"]), str(value["url"]), str(value["revision"]), str(value["action_space"]), int(value["action_dim"]), int(value["control_hz"]), int(value["max_horizon"]), str(value["endpoint"]), tuple(launcher), dict(options))
        endpoint = urlsplit(model.endpoint)
        try: is_loopback = endpoint.hostname is not None and ipaddress.ip_address(endpoint.hostname).is_loopback
        except ValueError: is_loopback = endpoint.hostname == "localhost"
        if (not model.name or not model.url.startswith("https://huggingface.co/") or len(model.revision) != 40 or not model.action_space or model.action_dim < 1 or model.control_hz < 1 or model.max_horizon < 1 or endpoint.scheme not in {"ws", "wss", "http", "tcp"} or not is_loopback or endpoint.port is None):
            raise ValueError("invalid model contract")
        return model

    def public_spec(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in ("url", "revision", "action_space", "action_dim", "control_hz", "max_horizon")}


