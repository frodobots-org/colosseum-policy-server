"""Rover LLM agent: real provider clients against scripted HTTP, no network or rover."""
import json

import numpy as np
import pytest

pytest.importorskip("inspect_robots_agent")
import httpx

from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.backends.auto import AutoBackend
from colosseum_policy_server.backends.base import LocalModel, PlanResult
from colosseum_policy_server.backends.rover_agent import (
    HeadingEstimate, RoverAgentBackend, _evicted, bearing_distance)
from colosseum_policy_server.local_protocol import decode_control, encode_control
from colosseum_policy_server.local_runtime import LocalPolicyRuntime, RuntimeConfig
from colosseum_policy_server.tensors import tensor_from_numpy

ROVER = "earth_rover_mini_plus"
URLS = {"anthropic": "https://api.anthropic.com/v1", "openai": "https://api.openai.com/v1", "x-ai": "https://api.x.ai/v1"}
NAMES = {"anthropic": "claude-fable-5-2", "openai": "gpt-6-astra", "x-ai": "grok-4.7"}
WIRES = {"anthropic": "messages", "openai": "responses", "x-ai": "chat"}
START = (30.48243713, 114.3026428)  # Wuhan checkpoint 1
GOAL = (30.48268318, 114.3026047)   # Wuhan checkpoint 2, about 28 m north


def model(provider="anthropic", **options):
    return LocalModel.from_mapping(dict(
        name=NAMES[provider], model_type="llm", url=URLS[provider], revision="rover-v1",
        endpoint="inprocess://inspect-agent", action_space="base_velocity", action_dim=2,
        control_hz=10, max_horizon=30, backend_options=options))


def request(**extra):
    return {"run_id": "run", "robot_type": ROVER, "api_key": "key",
            "navigation": {"timeout_s": 600, "stopped_s": 2}, **extra}


def observation(step=0, position=START, heading=0.0, goal=GOAL, goal_image=False, rear=False):
    obs = pb.Observation(instruction="Drive to the second checkpoint", control_step=step)
    navigation = np.array([position[0], position[1], heading, 0.0, 80.0, 0, 0, 0, 0], np.float64)
    obs.state["navigation_state"].CopyFrom(tensor_from_numpy(navigation))
    if goal is not None:
        obs.state["goal_gps"].CopyFrom(tensor_from_numpy(np.array(goal, np.float64)))
    names = ["front_image"] + (["rear_image"] if rear else []) + (["goal_image"] if goal_image else [])
    for name in names:
        obs.sensors.add(sensor_id=name, encoding=pb.RAW_RGB, width=4, height=3, data=bytes(36))
    return obs


def tool_reply(wire, name, arguments, call_id="call1"):
    if wire == "messages":
        return dict(content=[dict(type="tool_use", id=call_id, name=name, input=arguments)],
                    stop_reason="tool_use", usage={"input_tokens": 10, "output_tokens": 5})
    if wire == "responses":
        return dict(id="r1", status="completed", output=[dict(type="function_call", id="fc1", call_id=call_id,
                    name=name, arguments=json.dumps(arguments))], usage={})
    return dict(choices=[dict(message=dict(role="assistant", content=None, tool_calls=[
        dict(id=call_id, type="function", function=dict(name=name, arguments=json.dumps(arguments)))]))])


class Script:
    """Replies in order and keeps each request body."""

    def __init__(self, wire, *replies):
        self.wire, self.replies, self.bodies = wire, list(replies), []

    def __call__(self, http_request):
        self.bodies.append(json.loads(http_request.content))
        return httpx.Response(200, json=tool_reply(self.wire, *self.replies.pop(0)))


async def session(script, provider="anthropic", tmp_path=None, **options):
    if tmp_path is not None:
        options.setdefault("log_dir", str(tmp_path))
    backend = RoverAgentBackend({}, transport=httpx.MockTransport(script), clock=lambda: 100.0)
    await backend.start_session(model(provider, **options), request())
    return backend


DRIVE = {"forward": 0.3, "turn": 0.1, "seconds": 1.5, "note": "Path is clear ahead."}


@pytest.mark.parametrize("provider", ["anthropic", "openai", "x-ai"])
async def test_drive_returns_a_chunk_that_ends_stopped(provider, tmp_path):
    script = Script(WIRES[provider], ("drive", DRIVE))
    backend = await session(script, provider, tmp_path)
    actions = await backend.infer(backend.model, observation())
    assert actions.shape == (20, 2)  # 1.5 s at 10 Hz, then 0.5 s standing still
    assert np.allclose(actions[:15], [0.3, 0.1]) and not actions[15:].any()
    sent = json.dumps(script.bodies[0])
    assert "Drive to the second checkpoint" in sent and "m away" in sent
    assert "Time limit: 600 s" in sent and "about 600 s left" in sent
    # Messages API carries a base64 image block; the other wires a data URL.
    assert "data:image/png;base64," in sent or '"media_type": "image/png"' in sent
    await backend.end_session()


async def test_done_ends_the_trial_with_a_zero_plan(tmp_path):
    script = Script("messages", ("drive", DRIVE), ("done", {"note": "At the checkpoint."}))
    backend = await session(script, tmp_path=tmp_path)
    await backend.infer(backend.model, observation(0))
    result = await backend.infer(backend.model, observation(20, position=GOAL))
    assert isinstance(result, PlanResult) and result.done and not result.actions.any()
    assert backend.stop_reason == "done"
    await backend.end_session()
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert record["stop_reason"] == "done" and record["llm_calls"] == 2
    assert "base64" not in json.dumps(record)  # No camera frames in the local record.


async def test_give_up_also_ends_the_trial(tmp_path):
    backend = await session(Script("messages", ("give_up", {"note": "Blocked by a fence."})), tmp_path=tmp_path)
    result = await backend.infer(backend.model, observation())
    assert result.done and backend.stop_reason == "give_up"
    await backend.end_session()


async def test_out_of_range_command_is_rejected_and_retried(tmp_path):
    fast = {**DRIVE, "forward": 0.9}
    script = Script("messages", ("drive", fast), ("drive", DRIVE))
    backend = await session(script, tmp_path=tmp_path)
    actions = await backend.infer(backend.model, observation())
    assert np.allclose(actions[0], [0.3, 0.1])
    assert "Rejected: forward must be between -0.2 and 0.3" in json.dumps(script.bodies[1])
    await backend.end_session()


async def test_missing_note_is_rejected(tmp_path):
    script = Script("messages", ("drive", {**DRIVE, "note": ""}), ("drive", DRIVE))
    backend = await session(script, tmp_path=tmp_path)
    await backend.infer(backend.model, observation())
    assert "note is required" in json.dumps(script.bodies[1])
    await backend.end_session()


async def test_goal_direction_and_last_move_are_reported(tmp_path):
    east_goal = (START[0], START[1] + 0.0003)  # about 29 m east
    script = Script("messages", ("drive", DRIVE), ("drive", DRIVE))
    backend = await session(script, tmp_path=tmp_path)
    await backend.infer(backend.model, observation(0, heading=0.0, goal=east_goal))
    moved = (START[0] + 0.00001, START[1])
    await backend.infer(backend.model, observation(20, position=moved, heading=350.0, goal=east_goal))
    first, second = (json.dumps(body) for body in script.bodies)
    assert "90\\u00b0 to your right" in first or "90° to your right" in first
    assert "compass changed by -10" in second and "(left)" in second
    await backend.end_session()


async def test_photo_goal_stays_in_context_while_old_frames_are_elided(tmp_path):
    replies = [("drive", DRIVE)] * 4
    script = Script("messages", *replies)
    backend = await session(script, tmp_path=tmp_path, image_horizon=1)
    for step in range(4):
        await backend.infer(backend.model, observation(step * 20, goal=None, goal_image=True))
    last = json.dumps(script.bodies[-1])
    assert "Photo of the destination" in last
    assert last.count('"type": "image"') == 2  # the goal photo plus the newest frame
    assert "[camera frame elided]" in last
    await backend.end_session()


def test_eviction_marks_one_cache_anchor():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
    messages = [{"role": "system", "content": "s"}] + [
        {"role": "user", "content": [{"type": "text", "text": str(i)}, image]} for i in range(3)]
    view = _evicted(messages, 1, mark_anchor=True)
    assert [m.get("cache_anchor") for m in view] == [None, None, True, None]
    assert messages[1]["content"][1] is image  # The stored conversation is untouched.


async def test_rear_camera_is_sent_when_configured(tmp_path):
    script = Script("messages", ("drive", DRIVE))
    backend = await session(script, tmp_path=tmp_path, sensors=["front_image", "rear_image"])
    await backend.infer(backend.model, observation(rear=True))
    assert "rear_image:" in json.dumps(script.bodies[0])
    await backend.end_session()


@pytest.mark.parametrize("change,message", [
    (dict(action_dim=3), "2-D base_velocity"),
    (dict(action_space="joint_position"), "2-D base_velocity"),
    (dict(max_horizon=10), "shorter than the longest drive"),
])
async def test_contract_is_checked(change, message):
    item = LocalModel.from_mapping({**model().__dict__, "launcher": [], "backend_options": {}, **change})
    with pytest.raises(ValueError, match=message):
        await RoverAgentBackend({}).start_session(item, request())


async def test_requires_the_rover_and_a_client_key():
    with pytest.raises(ValueError, match="earth_rover_mini_plus"):
        await RoverAgentBackend({}).start_session(model(), request(robot_type="yam"))
    with pytest.raises(ValueError, match="API key"):
        await RoverAgentBackend({}).start_session(model(), request(api_key=""))
    with pytest.raises(ValueError, match="Claude Code login"):
        await RoverAgentBackend({}).start_session(model("openai"), request(auth_method="claude-code", api_key=""))


async def test_auto_backend_sends_rover_llms_to_the_rover_agent(tmp_path):
    backend = AutoBackend({"log_dir": str(tmp_path)})
    await backend.start_session(model(), request())
    assert isinstance(backend.active, RoverAgentBackend)
    assert backend.control_mode == "base_velocity"
    await backend.end_session()


def test_example_config_loads():
    from pathlib import Path
    config = RuntimeConfig.from_yaml(Path(__file__).parents[1] / "configs" / "local-runtime-rover-agent.yaml.example")
    assert {item.action_space for item in config.models.values()} == {"base_velocity"}
    assert "claude-fable-5-2" in config.models and "gpt-6-astra" in config.models


def test_heading_offset_is_learned_from_straight_moves():
    estimate = HeadingEstimate()
    # Driving due east (course 90°) while the compass reads 60°: 30° of bias.
    fixes = [(START[0], START[1] + i * 0.000012, 60.0) for i in range(5)]
    for before, after in zip(fixes, fixes[1:]):
        estimate.observe(before, after, {"forward": 0.3, "turn": 0.0})
    assert estimate.calibrated and abs(estimate.offset_deg - 15) < 2  # gain 0.5 moves halfway
    estimate.observe(fixes[-1], fixes[-1], {"forward": 0.0, "turn": 0.5})
    assert estimate.track == []  # A turn starts a new segment.


def test_bearing_distance_between_wuhan_checkpoints():
    bearing, distance = bearing_distance(*START, *GOAL)
    assert 27 < distance < 29 and (bearing > 350 or bearing < 10)


# Local Protocol: a done plan reaches the Client -------------------------------

class Socket:
    def __init__(self, messages): self.messages, self.sent = list(messages), []
    async def recv(self): return self.messages.pop(0)
    async def send(self, message): self.sent.append(message)


class FixedBackend:
    def __init__(self, result): self.result = result
    async def infer(self, model, observation): return self.result


def runtime_frames(tmp_path):
    item = model()
    config = RuntimeConfig("127.0.0.1", 8000, "unused", {}, tmp_path, 1, {item.name: item})
    prepare = encode_control({"type": "prepare", "protocol_version": 1, "run_id": "run",
                              "preparation_id": "prep", "model": item.public_spec()})
    frame = pb.RelayFrame(protocol_version=1, type=pb.OBSERVATION, session_id="run", sequence=1,
                          deadline_ms=1000, payload=pb.Observation(control_step=0).SerializeToString())
    return config, Socket([prepare, frame.SerializeToString()])


async def test_runtime_sends_done_plans(tmp_path):
    config, ws = runtime_frames(tmp_path)
    ws.messages.append(pb.RelayFrame(type=pb.SESSION_CLOSE, session_id="run").SerializeToString())
    await LocalPolicyRuntime(config, FixedBackend(PlanResult(np.zeros((1, 2)), done=True))).handle(ws)
    assert decode_control(ws.sent[0])["type"] == "ready"
    plan = pb.ActionPlan.FromString(pb.RelayFrame.FromString(ws.sent[1]).payload)
    assert plan.done and list(plan.actions.shape) == [1, 2]


async def test_runtime_rejects_a_done_plan_that_moves(tmp_path):
    config, ws = runtime_frames(tmp_path)
    await LocalPolicyRuntime(config, FixedBackend(PlanResult(np.ones((1, 2)), done=True))).handle(ws)
    error = pb.Error.FromString(pb.RelayFrame.FromString(ws.sent[1]).payload)
    assert error.code == "INVALID_REQUEST" and "zero actions" in error.message


async def test_runtime_still_accepts_plain_arrays(tmp_path):
    config, ws = runtime_frames(tmp_path)
    ws.messages.append(pb.RelayFrame(type=pb.SESSION_CLOSE, session_id="run").SerializeToString())
    await LocalPolicyRuntime(config, FixedBackend(np.full((3, 2), 0.2))).handle(ws)
    plan = pb.ActionPlan.FromString(pb.RelayFrame.FromString(ws.sent[1]).payload)
    assert not plan.done and list(plan.actions.shape) == [3, 2]


async def test_runtime_passes_client_navigation_to_the_agent(tmp_path):
    seen = {}

    class Recording:
        async def start_session(self, model, request): seen.update(request)
        async def infer(self, model, observation): return np.zeros((1, 2))

    item = model()
    config = RuntimeConfig("127.0.0.1", 8000, "unused", {}, tmp_path, 1, {item.name: item})
    prepare = encode_control({"type": "prepare", "protocol_version": 1, "run_id": "run", "preparation_id": "prep",
                              "model": item.public_spec(), "message": json.dumps({"client_options": {
                                  "navigation": {"timeout_s": 600, "stopped_s": 2}}})})
    close = pb.RelayFrame(type=pb.SESSION_CLOSE, session_id="run").SerializeToString()
    ws = Socket([prepare, close])
    await LocalPolicyRuntime(config, Recording()).handle(ws)
    assert seen["navigation"] == {"timeout_s": 600, "stopped_s": 2}
    assert decode_control(ws.sent[0])["control_mode"] == "base_velocity"
