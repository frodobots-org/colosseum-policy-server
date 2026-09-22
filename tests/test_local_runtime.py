import asyncio

import numpy as np
import pytest

from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.local_protocol import encode_control
from colosseum_policy_server.local_runtime import LocalModel, LocalPolicyRuntime, RuntimeConfig, load_backend


def model():
    return LocalModel.from_mapping({"name":"test", "url":"https://huggingface.co/example/model", "revision":"a" * 40, "runtime_profile":"test-v1", "action_space":"joint_position", "action_dim":8, "control_hz":15, "max_horizon":2, "endpoint":"ws://127.0.0.1:9100"})


def config(tmp_path):
    item = model()
    return RuntimeConfig("127.0.0.1", 8000, "unused", {}, tmp_path, 1, {item.name:item})


def prepare(): return encode_control({"type":"prepare", "protocol_version":1, "run_id":"run", "preparation_id":"prep", "model":model().public_spec()})
def observation(): return pb.RelayFrame(protocol_version=1, type=pb.OBSERVATION, session_id="run", sequence=1, deadline_ms=20, payload=pb.Observation(control_step=0).SerializeToString()).SerializeToString()


class Socket:
    def __init__(self, messages): self.messages, self.sent = list(messages), []
    async def recv(self): return self.messages.pop(0)
    async def send(self, message): self.sent.append(message)


def error_code(message): return pb.Error.FromString(pb.RelayFrame.FromString(message).payload).code


class Backend:
    def __init__(self, result=None, error=None, delay=0): self.result, self.error, self.delay = result, error, delay
    async def infer(self, model, observation):
        if self.delay: await asyncio.sleep(self.delay)
        if self.error: raise self.error
        return self.result


class FailingSupervisor:
    async def activate(self, model): raise RuntimeError("private launcher detail")


async def test_start_failure_is_sanitized(tmp_path):
    ws = Socket([prepare()])
    await LocalPolicyRuntime(config(tmp_path), Backend(), FailingSupervisor()).handle(ws)
    assert error_code(ws.sent[-1]) == "MODEL_START_FAILED"
    assert b"private launcher detail" not in ws.sent[-1]


@pytest.mark.parametrize(("backend", "code"), [(Backend(error=RuntimeError("private failure")), "INFERENCE_FAILED"), (Backend(delay=.05), "INFERENCE_TIMEOUT"), (Backend(result=np.zeros((1, 7))), "INVALID_REQUEST")])
async def test_inference_failures_are_stable(tmp_path, backend, code):
    ws = Socket([prepare(), observation()])
    await LocalPolicyRuntime(config(tmp_path), backend).handle(ws)
    assert error_code(ws.sent[-1]) == code


def test_backend_entry_point_factory_receives_options(monkeypatch):
    class Plugin:
        def __init__(self, options): self.options = options
        async def infer(self, model, observation): return np.zeros((1, 8))
    class Entry:
        name = "example"
        def load(self): return Plugin
    class Entries:
        def select(self, *, group): return [Entry()]
    monkeypatch.setattr("colosseum_policy_server.local_runtime.metadata.entry_points", lambda: Entries())
    assert load_backend("example", {"value":1}).options == {"value":1}


def test_model_rejects_non_loopback_endpoint():
    with pytest.raises(ValueError):
        LocalModel.from_mapping({"name":"test", "url":"https://huggingface.co/example/model", "revision":"a" * 40, "runtime_profile":"test-v1", "action_space":"joint_position", "action_dim":8, "control_hz":15, "max_horizon":2, "endpoint":"ws://192.168.1.1:9100"})
