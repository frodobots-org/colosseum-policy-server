"""Stateful Inspect Robots LLM policy behind Colosseum's Local Protocol.

No robot driver is loaded here. The existing DROID client executes joint
positions. Optional Inspect dependencies are imported only on session startup.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np

from .base import LocalModel
from ..model_adapters.observation import _rgb, _state
from ..model_adapters.robots import session_robot, llm_robot_contract


def create_backend(options: Mapping[str, Any]) -> "InspectAgentBackend":
    return InspectAgentBackend(options)


class InspectAgentBackend:
    """One isolated agent per exclusive runtime session, with bounded actions."""

    def __init__(self, options: Mapping[str, Any], *, agent_factory=None):
        self.options = dict(options)
        self.agent_factory = agent_factory
        self.agent = None
        self._inflight = None
        self.model = None
        self.bound = False
        self.last_step = -1
        self.stop_reason = None
        self.stop_step = None
        self.stop_detail = None
        self.hold_target = None
        self.last_actions = None

    async def start_session(self, model: LocalModel, request: Mapping[str, Any]) -> None:
        if self.agent is not None or self._inflight is not None:
            raise RuntimeError("previous agent session has not closed")
        settings = {**self.options, **model.backend_options}
        self.robot = llm_robot_contract(session_robot(model, request, self.options))
        self.robot.validate(model)
        self.joint_count = self.robot.joint_count
        identity = urlsplit(model.url)
        typed = model.model_type == "llm"
        providers = {"gpt-": ("openai", "OPENAI_API_KEY"),
                     "grok-": ("x-ai", "XAI_API_KEY"),
                     "claude-": ("anthropic", "ANTHROPIC_API_KEY")}
        if typed:
            provider, key_env = next((value for prefix, value in providers.items()
                                      if model.name.startswith(prefix)), (None, None))
            if provider is None:
                raise ValueError("Unsupported LLM model name")
            identity = urlsplit(f"llm://{provider}/{model.name}")
        if model.url.startswith("https://huggingface.co/"):
            identity = urlsplit("llm://" + str(model.backend_options.get("agent_model", "")))
        wires = {"openai": "responses", "x-ai": "chat", "anthropic": "messages"}
        if identity.scheme != "llm" or identity.netloc not in wires or model.endpoint != "inprocess://inspect-agent":
            raise ValueError("inspect_agent requires an llm://provider/model identity")
        self.low = self._vector(settings, "joint_low", self.joint_count)
        self.high = self._vector(settings, "joint_high", self.joint_count)
        self.max_step = self._vector(settings, "joint_max_step", self.joint_count)
        if np.any(self.low >= self.high) or np.any(self.max_step <= 0):
            raise ValueError("joint bounds must increase and joint_max_step must be positive")
        self.open_value = settings.get("gripper_open_value")
        if type(self.open_value) is not int or self.open_value not in {0, 1}:
            raise ValueError("declare gripper_open_value as 0 or 1 for the RobotEnv")
        self.sensors = tuple(settings.get("sensors", ["head_image", "left_image"]))
        if not self.sensors or any(not isinstance(s, str) or not s for s in self.sensors) or len(set(self.sensors)) != len(self.sensors):
            raise ValueError("sensors must be nonempty unique sensor names")
        self.docs = str(settings.get("robot_notes", ""))
        if not self.docs.strip():
            raise ValueError("robot_notes must describe the rig and joint directions")
        self.log_dir = Path(settings.get("log_dir", "agent-logs"))
        self.log_id = uuid4().hex  # Never use a peer's run_id as a path.
        self.run_id = str(request["run_id"])
        self.model = model
        self.bound, self.last_step, self.stop_reason = False, -1, None
        self.stop_step = self.stop_detail = self.hold_target = self.last_actions = None
        factory = self.agent_factory
        if factory is None:
            from inspect_robots_agent import LLMAgentPolicy
            factory = LLMAgentPolicy
        # Keep endpoint and credentials provider-owned; keys are read from env
        # by Inspect, never carried in the public model contract or config logs.
        kwargs = {key: settings[key] for key in (
            "effort", "max_llm_calls", "max_output_tokens", "max_speed_frac",
            "images", "image_horizon", "max_retries", "backoff_s",
        ) if key in settings}
        kwargs.setdefault("effort", "low")
        kwargs.setdefault("max_llm_calls", 30)
        kwargs.setdefault("max_retries", 1)
        if typed:
            api_key = request.get("api_key")
            if not isinstance(api_key, str) or not api_key.strip():
                raise ValueError("Client must supply its API key for this LLM")
            # Per-agent credential environment; never mutate process env or
            # fall back to another Client's / the host's provider credentials.
            base_url = model.url.rstrip("/")
            # Inspect appends /messages; Anthropic URLs may be supplied at root.
            if provider == "anthropic" and not urlsplit(base_url).path.strip("/"):
                base_url += "/v1"
            kwargs.update(env={key_env: api_key}, base_url=base_url, api_key_env=key_env)
            auth = settings.get("api_auth", "api_key" if provider == "anthropic" else "bearer")
            if auth not in {"api_key", "bearer"}:
                raise ValueError("api_auth must be api_key or bearer")
            if provider == "anthropic" and auth == "bearer":
                from .llm_transport import BearerTransport
                kwargs["transport"] = BearerTransport(api_key)
        try:
            self.agent = factory(model=model.name if typed else f"{identity.netloc}/{identity.path.lstrip('/')}",
                                 wire=settings.get("api_format", wires[identity.netloc]), wire_capture=False, **kwargs)
            if typed and urlsplit(model.url).hostname != "api.openai.com":
                # Pinned Inspect enables explicit GPT cache breakpoints by model
                # name alone. Relays may reject them; keep stateless full history.
                client = getattr(self.agent, "_client", None)
                if hasattr(client, "_cache_anchors"):
                    client._cache_anchors = False
        except Exception:
            if "transport" in kwargs:
                kwargs["transport"].close()
            raise

    @staticmethod
    def _vector(settings, name, size):
        value = np.asarray(settings.get(name), dtype=np.float64)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain {size} finite numbers")
        return value

    async def infer(self, model, observation):
        if self.agent is None or model != self.model:
            raise RuntimeError("agent session was not prepared for this model")
        if self._inflight is not None:
            raise RuntimeError("previous inference is still running")
        # Inspect's HTTP clients are synchronous. Shield the worker so timeout
        # cannot detach a still-mutating conversation from session cleanup.
        task = asyncio.create_task(asyncio.to_thread(self._infer, observation))
        self._inflight = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                self._inflight = None

    def _infer(self, observation):
        from inspect_robots.embodiment import EmbodimentInfo
        from inspect_robots.scene import Scene
        from inspect_robots.spaces import ActionSemantics, Box, CameraSpec, ObservationSpace, StateField, StateSpec
        from inspect_robots.types import Observation

        if not observation.instruction.strip() or observation.control_step <= self.last_step:
            raise ValueError("instruction must be nonempty and control_step must advance")
        joints = _state(observation, self.robot.state_key, (self.joint_count,)).astype(np.float64)
        gripper = _state(observation, "gripper_position", (1,)).astype(np.float64)
        if np.any(joints < self.low) or np.any(joints > self.high) or np.any(gripper < 0) or np.any(gripper > 1):
            raise ValueError("observed state is outside the configured rig bounds")
        if self.hold_target is not None:
            return self._hold(joints, observation.control_step)
        raw = {item.sensor_id: item for item in observation.sensors}
        images = {name: _rgb(raw.get(name), name) for name in self.sensors}
        # Inspect uses 1=open. RobotEnv's polarity is explicitly configured.
        position = np.r_[joints, gripper if self.open_value == 1 else 1 - gripper]
        if not self.bound:
            cameras = tuple(CameraSpec(name, img.shape[0], img.shape[1]) for name, img in images.items())
            self.agent.bind(EmbodimentInfo(
                name=self.robot.name, control_hz=self.model.control_hz,
                action_space=Box(shape=(self.robot.action_dim,), low=np.r_[self.low, 0.], high=np.r_[self.high, 1.],
                    semantics=ActionSemantics(control_mode="joint_pos", gripper="binary",
                        dim_labels=tuple(f"joint{i + 1}" for i in range(self.joint_count)) + ("gripper",),
                        max_step=tuple(self.max_step) + (1.,))),
                observation_space=ObservationSpace(cameras=cameras,
                    state=StateSpec((StateField("joint_pos", (self.robot.action_dim,), "rad+normalized"),))),
                docs=self.docs + "\nThe final dimension is gripper: 0 closed, 1 open. "
                    "Motion may be truncated to max_horizon; use the next measured state to replan.",
            ))
            self.agent.reset(Scene(id=self.run_id, instruction=observation.instruction))
            self.bound = True
        obs = Observation(images=images, state={"joint_pos": position},
                          instruction=observation.instruction, extra={"env_step": observation.control_step})
        chunk = self.agent.act(obs)
        if chunk.meta.get("request_stop") or any(action.meta.get("request_stop") for action in chunk.actions):
            meta = next((a.meta for a in chunk.actions if a.meta.get("request_stop")), chunk.meta)
            self.stop_reason = meta.get("stop_reason", "give_up")
            if self.stop_reason not in {"done", "give_up"}:
                raise ValueError("invalid agent stop reason")
            self.stop_step = observation.control_step
            self.stop_detail = str(meta.get("stop_detail", ""))
            # Freeze measured joints, not the LLM's proposed stop action. Keep
            # the last executed binary gripper command: an obstructed closed
            # gripper's measured width can look open and must not reopen it.
            grip = float(gripper[0] > .5)
            if self.last_actions is not None:
                executed = min(observation.control_step - self.last_step, len(self.last_actions))
                grip = float(self.last_actions[executed - 1, -1])
            self.hold_target = np.r_[joints, grip].astype(np.float32)
            return self._hold(joints, observation.control_step)
        if chunk.control_hz is not None and chunk.control_hz != self.model.control_hz:
            raise ValueError("agent returned a different control frequency")
        actions = np.asarray([action.data for action in chunk.actions], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != self.robot.action_dim or not len(actions) or not np.isfinite(actions).all():
            raise ValueError("agent returned malformed actions")
        if np.any(actions < np.r_[self.low, 0.]) or np.any(actions > np.r_[self.high, 1.]):
            raise ValueError("agent actions exceed rig bounds")
        if np.any(np.abs(np.diff(np.vstack((position, actions)), axis=0)[:, :self.joint_count]) > self.max_step + 1e-7):
            raise ValueError("agent actions exceed per-step joint limits")
        # Match the existing client's binary gripper execution, including polarity.
        actions[:, -1] = (actions[:, -1] > .5).astype(float)
        if self.open_value == 0:
            actions[:, -1] = 1 - actions[:, -1]
        self.last_step = observation.control_step
        self.last_actions = actions[:self.model.max_horizon].astype(np.float32)
        return self.last_actions.copy()

    def _hold(self, joints, control_step):
        """Repeat one fixed target while retaining the per-step motion guard."""
        if np.any(np.abs(self.hold_target[:self.joint_count].astype(np.float64) - joints) > self.max_step + 1e-7):
            raise ValueError("hold target exceeds per-step joint limits from measured state")
        self.last_step = control_step
        return self.hold_target[None, :].copy()

    async def end_session(self):
        # A timed-out HTTP worker must finish before closing its client or
        # allowing another trial into this stateful backend.
        task = self._inflight
        if task is not None:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not task.cancelled():
                task.exception()  # Retrieve late failures after a timeout.
            self._inflight = None
        agent, self.agent = self.agent, None
        if agent is None:
            return
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            record = SimpleNamespace(scene_id="trial", epoch=0, metadata={})
            agent.on_trial_end(record, str(self.log_dir), self.log_id)
            (self.log_dir / f"{self.log_id}.json").write_text(json.dumps({
                "run_id": self.run_id, "model": self.model.public_spec(),
                "robot_type": self.robot.name,
                "stop_reason": self.stop_reason, "stop_step": self.stop_step,
                "stop_detail": self.stop_detail,
                "hold_target": self.hold_target.tolist() if self.hold_target is not None else None,
                "metadata": record.metadata,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            # Inspect 0.29 exposes close on its transport, not its policy.
            # Keep this version-specific access isolated here.
            client = getattr(agent, "_client", None)
            if client is not None:
                client.close()
            self.model = None
            self.hold_target = self.last_actions = None
