from dataclasses import replace

import numpy as np
import pytest

from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.backends.test_backend import TestBackend
from colosseum_policy_server.local_protocol import encode_control, decode_control
from colosseum_policy_server.local_runtime import LocalPolicyRuntime
from colosseum_policy_server.tensors import tensor_from_numpy
from test_local_runtime import Socket, config, error_code, model


async def test_loopback_requires_explicit_test_flag(tmp_path):
    ws = Socket([encode_control(dict(type='prepare', protocol_version=1,
        run_id='run', model=model().public_spec()))])
    await LocalPolicyRuntime(config(tmp_path), TestBackend({})).handle(ws)
    assert error_code(ws.sent[0]) == 'INVALID_REQUEST'


@pytest.mark.parametrize('invalid', ['missing', 'nan', 'dimension', 'velocity'])
async def test_invalid_echo_is_rejected(invalid):
    obs = pb.Observation()
    obs.state['joint_position'].CopyFrom(tensor_from_numpy(np.zeros(7, dtype=np.float32)))
    if invalid != 'missing':
        obs.state['gripper_position'].CopyFrom(tensor_from_numpy(np.array([np.nan if invalid == 'nan' else .5], dtype=np.float32)))
    item = replace(model(), action_dim=9) if invalid == 'dimension' else model()
    if invalid == 'velocity':
        item = replace(item, action_space='joint_velocity')
    with pytest.raises(ValueError):
        await TestBackend({}).infer(item, obs)


async def test_custom_bimanual_state_order():
    obs = pb.Observation()
    fields = ['left_joint', 'left_gripper', 'right_joint', 'right_gripper']
    for name, value in zip(fields, [[1., 2.], [.3], [4., 5.], [.6]]):
        obs.state[name].CopyFrom(tensor_from_numpy(np.array(value, dtype=np.float32)))
    actions = await TestBackend({'state_fields': fields}).infer(replace(model(), action_dim=6), obs)
    np.testing.assert_array_equal(actions, np.array([[1, 2, .3, 4, 5, .6]], dtype=np.float32))
