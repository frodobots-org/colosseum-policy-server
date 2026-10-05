from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from colosseum_policy_server.backends.auto import AutoBackend
from colosseum_policy_server.backends.base import LocalModel
from colosseum_policy_server.backends.vla import VLABackend
from colosseum_policy_server.local_protocol import encode_control, decode_control
from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.local_runtime import LocalPolicyRuntime, RuntimeConfig


def model(adapter='molmoact2', **kwargs):
    return LocalModel.from_mapping(dict(name='test', url='https://huggingface.co/org/model',
        revision='a' * 40, action_space='joint_position', action_dim=8, control_hz=15,
        max_horizon=2, endpoint='http://127.0.0.1:9100',
        backend_options={'adapter': adapter, **kwargs}))


def test_robot_identity_roundtrip():
    request = dict(type='prepare', run_id='r', robot_type='yam')
    assert decode_control(encode_control(request))['robot_type'] == 'yam'


@pytest.mark.parametrize('robot', ['yam', 'so101'])
async def test_franka_model_cannot_be_selected_for_other_robots(robot):
    backend = AutoBackend({})
    with pytest.raises(ValueError, match='does not support robot_type'):
        await backend.start_session(model(), {'robot_type': robot})
    await backend.end_session()
    assert backend.active is None


async def test_contract_checked_during_preparation():
    backend = VLABackend({})
    with pytest.raises(ValueError, match='action space'):
        await backend.start_session(replace(model(), action_dim=6), {'robot_type': 'franka'})
    assert backend.active is None
    with pytest.raises(ValueError, match='does not match'):
        await backend.start_session(model(robot_type='yam'), {'robot_type': 'franka'})


async def test_missing_adapter_is_not_silently_droid():
    backend = VLABackend({})
    with pytest.raises(ValueError, match='not installed'):
        await backend.start_session(model('unknown-model-adapter'), {'robot_type': 'franka'})


async def test_yam_plugin_and_franka_share_vla_runtime(monkeypatch):
    """Synthetic plugin proves routing and lifecycle, not YAM hardware support."""
    events = []
    class YamModelAdapter:
        supported_robot_types = ('yam',)
        def validate(self, model):
            assert model.action_dim == 6
        async def start_session(self, model, request):
            events.append(('start', request['robot_type']))
        async def infer(self, model, observation):
            return np.zeros((1, 6), np.float32)
        async def end_session(self):
            events.append(('end',))
    class Entries(list):
        def select(self, **kwargs):
            assert kwargs == {'group': 'colosseum_policy_server.model_adapters'}
            return self
    from colosseum_policy_server.backends import vla
    monkeypatch.setattr(vla.metadata, 'entry_points', lambda: Entries([
        SimpleNamespace(name='yam-test', load=lambda: lambda options: YamModelAdapter())]))
    backend = AutoBackend({})
    item = replace(model('yam-test', robot_type='yam'), action_dim=6)
    await backend.start_session(item, {'robot_type': 'yam'})
    assert (await backend.infer(item, None)).shape == (1, 6)
    await backend.end_session()
    assert events == [('start', 'yam'), ('end',)]
    await backend.start_session(model(), {'robot_type': 'franka'})
    assert isinstance(backend.active, VLABackend)
    await backend.end_session()


@pytest.mark.parametrize('robot,ready', [('franka', True), ('yam', False)])
async def test_protobuf_preparation_checks_robot_before_ready(tmp_path, robot, ready):
    item = model()
    config = RuntimeConfig('127.0.0.1', 8000, 'auto', {}, tmp_path, 1, {item.name: item})
    class Socket:
        def __init__(self):
            self.sent = []
            self.messages = iter([
                encode_control(dict(type='prepare', protocol_version=1, run_id='r',
                                    robot_type=robot, model=item.public_spec())),
                pb.RelayFrame(type=pb.SESSION_CLOSE, session_id='r').SerializeToString(),
            ])
        async def recv(self):
            return next(self.messages)
        async def send(self, message):
            self.sent.append(message)
    socket = Socket()
    backend = AutoBackend({})
    await LocalPolicyRuntime(config, backend).handle(socket)
    assert len(socket.sent) == 1
    if ready:
        assert decode_control(socket.sent[0])['type'] == 'ready'
    else:
        frame = pb.RelayFrame.FromString(socket.sent[0])
        assert frame.type == pb.ERROR
        assert pb.Error.FromString(frame.payload).code == 'MODEL_START_FAILED'
    assert backend.active is None
