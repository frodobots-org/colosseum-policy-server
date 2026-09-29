import asyncio

import numpy as np
import pytest

from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.local_protocol import encode_control
from colosseum_policy_server.local_runtime import LocalModel, LocalPolicyRuntime, RuntimeConfig, load_backend


def model():
    return LocalModel.from_mapping({"name":"test", "url":"https://huggingface.co/example/model", "revision":"a" * 40, "subfolder":"", "action_space":"joint_position", "action_dim":8, "control_hz":15, "max_horizon":2, "endpoint":"ws://127.0.0.1:9100"})


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
        LocalModel.from_mapping({"name":"test", "url":"https://huggingface.co/example/model", "revision":"a" * 40, "subfolder":"", "action_space":"joint_position", "action_dim":8, "control_hz":15, "max_horizon":2, "endpoint":"ws://example.invalid:9100"})


def test_model_accepts_loopback_http_endpoint():
    item = LocalModel.from_mapping({"name":"test", "url":"https://huggingface.co/example/model", "revision":"a" * 40, "subfolder":"", "action_space":"joint_position", "action_dim":8, "control_hz":15, "max_horizon":2, "endpoint":"http://127.0.0.1:9100"})
    assert item.endpoint == "http://127.0.0.1:9100"


@pytest.mark.parametrize('subfolder', [None, '', 'ignored-checkpoint'])
@pytest.mark.parametrize('profile', [None, '', 'unknown-legacy-profile'])
async def test_prepare_profile_is_optional_metadata(tmp_path, subfolder, profile):
    from colosseum_policy_server.local_protocol import decode_control
    spec = model().public_spec()
    if subfolder is not None:
        spec['subfolder'] = subfolder
    if profile is not None:
        spec['runtime_profile'] = profile
    request = encode_control({'type': 'prepare', 'protocol_version': 1,
        'run_id': 'run', 'preparation_id': 'prep', 'model': spec})
    close = pb.RelayFrame(protocol_version=1, type=pb.SESSION_CLOSE, session_id='run').SerializeToString()
    ws = Socket([request, observation(), close])
    await LocalPolicyRuntime(config(tmp_path), Backend(result=np.ones((1, 8)))).handle(ws)
    ready = decode_control(ws.sent[0])
    assert ready['type'] == 'ready'
    assert ready['model'] == decode_control(request)['model']
    reply = pb.RelayFrame.FromString(ws.sent[1])
    assert reply.type == pb.ACTION_PLAN
    assert list(pb.ActionPlan.FromString(reply.payload).actions.shape) == [1, 8]
    assert ready['model'] == spec


@pytest.mark.parametrize('filename', ['local-runtime.yaml.example', 'local-runtime-droid-example.yaml'])
def test_example_config_needs_no_profile(filename):
    from pathlib import Path
    cfg = RuntimeConfig.from_yaml(Path(__file__).parents[1] / 'configs' / filename)
    assert cfg.models


async def test_new_robot_six_joint_backend_returns_action_plan(tmp_path):
    from colosseum_policy_server.backends import LocalModel as PublicModel
    from colosseum_policy_server.local_protocol import decode_control
    item = PublicModel.from_mapping({
        'name': 'six-joint', 'url': 'https://huggingface.co/example/model',
        'revision': 'a' * 40, 'action_space': 'joint_position', 'action_dim': 6,
        'control_hz': 15, 'max_horizon': 2, 'endpoint': 'http://127.0.0.1:9100'})
    cfg = RuntimeConfig('127.0.0.1', 8000, 'unused', {}, tmp_path, 1, {item.name: item})
    request = encode_control({'type': 'prepare', 'protocol_version': 1,
        'run_id': 'run', 'preparation_id': 'prep', 'model': item.public_spec()})
    close = pb.RelayFrame(protocol_version=1, type=pb.SESSION_CLOSE, session_id='run').SerializeToString()
    ws = Socket([request, observation(), close])
    await LocalPolicyRuntime(cfg, Backend(result=np.ones((2, 6)))).handle(ws)
    assert decode_control(ws.sent[0])['type'] == 'ready'
    frame = pb.RelayFrame.FromString(ws.sent[1])
    assert frame.type == pb.ACTION_PLAN
    assert list(pb.ActionPlan.FromString(frame.payload).actions.shape) == [2, 6]


@pytest.mark.parametrize('robot', ['droid', 'yam', 'custom_robot'])
def test_robot_cli_selects_backend(tmp_path, monkeypatch, robot):
    import yaml
    from colosseum_policy_server.local_server import configured_runtime
    config_path = tmp_path/'runtime.yaml'
    item = model()
    config_path.write_text(yaml.safe_dump({'backend': {'name': 'custom'},
        'models': [{'name': item.name, **item.public_spec(), 'endpoint': item.endpoint}]}))
    assert configured_runtime(config_path).backend_name == 'custom'
    assert configured_runtime(config_path, robot).backend_name == robot
    from colosseum_policy_server import local_server
    selected = []
    async def capture(config):
        selected.append(config.backend_name)
    monkeypatch.setattr(local_server, 'run', capture)
    monkeypatch.setattr('sys.argv', ['colosseum-policy-local', '--robot', robot, '--config', str(config_path)])
    local_server.main()
    assert selected == [robot]
