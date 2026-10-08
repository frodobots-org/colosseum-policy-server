from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import yaml

from colosseum_policy_server.yam_lingbot import LingBotYAMRuntime, validate_assets
from colosseum_policy_server.model_service import parse_args


@pytest.fixture
def assets(tmp_path):
    root = tmp_path / 'model'
    checkpoint = root / 'checkpoints/global_step_40000/hf_ckpt'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'model.safetensors').touch()
    (checkpoint / 'config.json').write_text('{}')
    (root / 'lingbotvla_cli.yaml').write_text(yaml.safe_dump({
        'train': {'chunk_size': 30}, 'data': {'cameras': ['camera_top', 'camera_wrist_left', 'camera_wrist_right']}}))
    robot = root / 'configs/robot_configs/molmoact2_yam_abs.yaml'
    robot.parent.mkdir(parents=True)
    robot.write_text((Path(__file__).parent / 'fixtures/lingbot_yam/robot.yaml').read_text())
    stats = root / 'assets/norm_stats/molmoact2_yam_abs.json'
    stats.parent.mkdir(parents=True)
    stats.write_text('{}')
    source = tmp_path / 'source'
    (source / 'deploy').mkdir(parents=True)
    (source / 'deploy/lingbot_vla_v2_policy.py').touch()
    processor = tmp_path / 'processor'
    processor.mkdir()
    for name in ('config.json', 'tokenizer_config.json', 'preprocessor_config.json'):
        (processor / name).write_text('{}')
    return checkpoint, source, processor, robot


def payload():
    return {'state': np.arange(14, dtype=np.float32) / 20, 'instruction': 'pick cup',
            **{name: np.full((5, 7, 3), i * 60, np.uint8)
               for i, name in enumerate(LingBotYAMRuntime.image_keys)}}


@pytest.fixture
def runtime(assets, monkeypatch):
    checkpoint, source, processor, robot = assets
    calls = {}
    class Policy:
        def __init__(self, **kwargs):
            calls['init'] = kwargs
            assert Path(kwargs['robot_norm_path']).is_file()
        def reset(self, robo_name):
            assert Path(f'configs/robot_configs/{robo_name}.yaml').is_file()
            calls['reset'] = robo_name
        def infer(self, observation):
            calls['observation'] = observation
            return {'action': np.tile(observation['observation.state'], (30, 1))}
    monkeypatch.setitem(sys.modules, 'deploy.lingbot_vla_v2_policy', SimpleNamespace(LingbotVLAv2Server=Policy))
    monkeypatch.syspath_prepend(str(source))
    for name in ('QWEN3VL_PATH', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        monkeypatch.setenv(name, '')
    cwd = Path.cwd()
    worker = LingBotYAMRuntime(checkpoint, source_root=source, processor=processor)
    assert Path.cwd() == cwd
    assert not (source / 'configs').exists()
    return worker, calls


def test_state_camera_order_absolute_actions_and_fresh_inference(runtime):
    worker, calls = runtime
    for offset in (0, .01):
        p = payload()
        p['state'] += offset
        np.testing.assert_array_equal(worker.infer(p)['actions'], np.tile(p['state'], (30, 1)))
    assert calls['init']['chunk_ret'] is True
    assert calls['init']['use_length'] == 30
    assert calls['init']['use_compile'] is False
    assert calls['reset'] == 'molmoact2_yam_abs'
    assert calls['observation']['task'] == 'pick cup'
    for i, name in enumerate(worker.camera_map.values()):
        np.testing.assert_array_equal(calls['observation'][name], np.full((5, 7, 3), i * 60, np.uint8))


@pytest.mark.parametrize('bad', ['state', 'gripper', 'image', 'instruction'])
def test_invalid_observation_rejected(runtime, bad):
    worker, calls = runtime
    p = payload()
    if bad == 'state': p['state'][0] = np.nan
    if bad == 'gripper': p['state'][6] = 1.1
    if bad == 'image': p['left_cam'] = np.zeros((4, 4, 3), np.float32)
    if bad == 'instruction': p['instruction'] = ''
    with pytest.raises(ValueError): worker.infer(p)
    assert 'observation' not in calls


@pytest.mark.parametrize('bad', ['shape', 'nan'])
def test_invalid_model_output_rejected(runtime, bad):
    worker, _ = runtime
    actions = np.zeros((29 if bad == 'shape' else 30, 14))
    if bad == 'nan': actions[0, 0] = np.nan
    worker.policy.infer = lambda _: {'action': actions}
    with pytest.raises(ValueError): worker.infer(payload())


@pytest.mark.parametrize('low,high', [(-1e-8, np.nextafter(np.float32(1), np.float32(2))),
                                    (-.1, 1.003670), (-10., 10.)])
def test_grippers_clipped_without_changing_arms_or_upstream(runtime, caplog, low, high):
    worker, _ = runtime
    actions = np.tile(payload()['state'], (30, 1))
    actions[14, 6] = low
    actions[15, 13] = high
    original = actions.copy()
    actions.setflags(write=False)
    worker.policy.infer = lambda _: {'action': actions}
    result = worker.infer(payload())['actions']
    expected = original.copy()
    expected[14, 6], expected[15, 13] = 0, 1
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(actions, original)
    assert 'action_index=14 left_gripper target=-' in caplog.text
    assert 'action_index=15 right_gripper' in caplog.text


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('column', [6, 13])
def test_nonfinite_grippers_still_rejected(runtime, value, column):
    worker, _ = runtime
    actions = np.zeros((30, 14), dtype=np.float32)
    actions[14, column] = value
    worker.policy.infer = lambda _: {'action': actions}
    with pytest.raises(ValueError):
        worker.infer(payload())


def test_cli_and_reject_relative_checkpoint(assets):
    checkpoint, source, processor, robot = assets
    argv = ['lingbot_v2', '--robot-type', 'yam', '--checkpoint', str(checkpoint),
            '--source-root', str(source), '--processor', str(processor), '--port', '8206']
    assert parse_args(argv).model == 'lingbot_v2'
    bad = argv.copy()
    bad[2] = 'franka'
    with pytest.raises(SystemExit): parse_args(bad)
    value = yaml.safe_load(robot.read_text())
    value['actions'][0]['action.arm.position']['subtract_state'] = True
    robot.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match='absolute YAM'): validate_assets(checkpoint, source, processor)
    with pytest.raises(SystemExit): parse_args(argv)
