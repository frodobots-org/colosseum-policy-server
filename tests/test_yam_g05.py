from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from colosseum_policy_server import model_service, yam_g05
from test_model_workers import Tensor


@pytest.fixture
def assets(tmp_path):
    ckpt = tmp_path / 'checkpoint'
    (ckpt / '.hydra').mkdir(parents=True)
    # RoboColosseum/G05-MolmoAct2-YAM @ 685add3f748f823c49889f092f6c3cceabcae062
    (ckpt / '.hydra/config.yaml').write_bytes(
        (Path(__file__).parent / 'fixtures/g05_yam_config.yaml').read_bytes())
    for name in ('model.pt', 'action_tokenizer.pt', 'dataset_stats.json'):
        (ckpt / name).touch()
    source = tmp_path / 'upstream'
    (source / 'scripts').mkdir(parents=True)
    (source / 'scripts/serve_policy.py').touch()
    processor = tmp_path / 'processor'
    processor.mkdir()
    # Published training metadata, upstream revision 26c91270ce33325a29990b3ff7ea7a4b47bd4ec1.
    parts = tmp_path / 'test-parts.yaml'
    parts.write_bytes((Path(__file__).parents[1] / 'configs/data/parts_meta/yam.yaml').read_bytes())
    return ckpt, source, processor, parts


@pytest.fixture
def runtime(assets, monkeypatch):
    ckpt, source, processor, parts = assets
    calls = {}
    cfg = SimpleNamespace(model=SimpleNamespace(
        model_arch=SimpleNamespace(hf_processor_path='old'),
        processor=SimpleNamespace(tokenizer_params=SimpleNamespace(pretrained_model_name_or_path='old'))))
    def load(stage, path, overrides):
        assert Path.cwd() == source
        assert (stage / 'model.pt').resolve() == ckpt / 'model.pt'
        assert (stage / 'hf_processor').resolve() == processor
        text = (stage / '.hydra/config.yaml').read_text()
        assert 'configs/data/parts_meta/yam.yaml' not in text and str(parts) in text
        assert overrides == ['eval_embodiment=yam', 'model.use_torch_compile=false']
        return cfg
    def setup(config, *, device):
        assert config.model.model_arch.hf_processor_path == str(processor)
        assert config.model.processor.tokenizer_params.pretrained_model_name_or_path == str(processor)
        return 'policy', 'processor'
    def build(raw, p):
        calls['raw'] = raw
        assert p == 'processor'
        return 'upstream observation'
    class Inferencer:
        def __init__(self, policy, p, *, device): assert policy == 'policy' and p == 'processor'
        def infer(self, observations):
            assert observations == ['upstream observation']
            # Stand-in for already postprocessed ABSOLUTE values. The bridge
            # must not add the observation again or invert either gripper.
            return [{key: Tensor(np.full((1, 32, end-start), value, np.float32))
                     for (key,start,end),value in zip(yam_g05.G05YAMRuntime.parts, [1.5, .2, -1.5, .8])}]
    monkeypatch.setattr(yam_g05.runpy, 'run_path', lambda path: dict(
        load_config_from_run_dir=load, filter_embodiment=lambda c,e: None,
        setup=setup, PolicyInferencer=Inferencer, build_obs_dict=build))
    cwd = Path.cwd()
    worker = yam_g05.G05YAMRuntime(ckpt, source_root=source, processor=processor,
                                  parts_meta=parts, native_attention=True)
    assert Path.cwd() == cwd
    yield worker, calls
    worker._stage.cleanup()


def payload():
    return dict(state=np.arange(14, dtype=np.float32)/20, instruction='pick cup', **{
        key: np.full((8,12,3), i*50, np.uint8) for i,key in enumerate(yam_g05.G05YAMRuntime.image_keys)})


def test_raw_inputs_and_absolute_output(runtime):
    worker, calls = runtime
    actions = worker.infer(payload())['actions']
    assert actions.shape == (32,14)
    np.testing.assert_allclose(actions[0], [1.5]*6+[.2]+[-1.5]*6+[.8])
    raw = calls['raw']
    assert raw['embodiment_type'] == 'yam' and raw['frequency'] == 30
    assert raw['task'] == 'pick cup'
    for i,key in enumerate(('head_rgb','left_wrist_rgb','right_wrist_rgb')):
        assert raw['images'][key].shape == (3,8,12)
        assert raw['images'][key].dtype == np.uint8
        assert np.all(raw['images'][key] == i*50)
    for key,start,end in worker.parts:
        np.testing.assert_array_equal(raw['state'][key], payload()['state'][start:end])


@pytest.mark.parametrize('bad', ['missing','absent','shape','nan','gripper'])
def test_reject_invalid_output(runtime, bad):
    worker, _ = runtime
    result = {k: np.zeros((1,32,e-s),np.float32) for k,s,e in worker.parts}
    if bad == 'missing': del result['left_arm']
    elif bad == 'absent': result['_absent_keys'] = {'right_gripper'}
    elif bad == 'shape': result['left_arm'] = np.zeros((1,32,9))
    elif bad == 'nan': result['left_arm'][0,0,0] = np.nan
    else: result['right_gripper'][0,0,0] = 1.1
    worker.inferencer.infer = lambda obs: [result]
    with pytest.raises(ValueError): worker.infer(payload())


@pytest.mark.parametrize('bad', ['state','gripper','image','instruction'])
def test_reject_invalid_input(runtime, bad):
    worker, _ = runtime
    data = payload()
    if bad == 'state': data['state'][0] = np.nan
    elif bad == 'gripper': data['state'][6] = -1
    elif bad == 'image': data['top_cam'] = np.zeros((3,8,12),np.uint8)
    else: data['instruction'] = ''
    with pytest.raises(ValueError): worker.infer(data)


def test_missing_original_recipe_rejected_before_import(assets):
    ckpt, source, processor, _ = assets
    with pytest.raises(FileNotFoundError, match='Missing training parts_meta'):
        yam_g05.G05YAMRuntime(ckpt, source_root=source, processor=processor)


def test_wrong_arm_transform_rejected(assets):
    ckpt, source, processor, parts = assets
    path = ckpt / '.hydra/config.yaml'
    cfg = yaml.safe_load(path.read_text())
    cfg['data']['processors']['yam']['action_state_transforms'] = []
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='relative arm transform'):
        yam_g05.validate_assets(ckpt, source, processor, parts)


def test_cli_http_dispatch(runtime, assets, monkeypatch):
    worker, _ = runtime
    ckpt, source, processor, parts = assets
    args = ['g05','--robot-type','yam','--checkpoint',str(ckpt),'--source-root',str(source),
            '--processor',str(processor),'--g05-parts-meta',str(parts),'--port','8205']
    assert model_service.parse_args(args).robot_type == 'yam'
    calls = []
    def create(*a, **kw):
        assert kw['parts_meta'] == parts
        return worker
    monkeypatch.setattr(yam_g05, 'G05YAMRuntime', create)
    class Server:
        def __init__(self, address, handler): assert address == ('127.0.0.1',8205)
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def serve_forever(self): calls.append(True)
    monkeypatch.setattr(model_service, 'HTTPServer', Server)
    model_service.main(args)
    assert calls == [True]


def test_parts_group_order_rejected(assets):
    ckpt, source, processor, parts = assets
    raw = yaml.safe_load(parts.read_text())
    raw['merge_spec'] = dict(reversed(list(raw['merge_spec'].items())))
    parts.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ValueError, match='group order'):
        yam_g05.validate_assets(ckpt, source, processor, parts)


def test_runtime_config_matches_g05_contract():
    from colosseum_policy_server.local_runtime import RuntimeConfig
    from colosseum_policy_server.backends.vla import load_model_adapter
    cfg = RuntimeConfig.from_yaml(Path(__file__).parents[1] / 'configs/local-runtime-yam-g05.yaml.example')
    model = cfg.models['g05-yam']
    assert model.max_horizon == 32 and model.backend_options['robot_type'] == 'yam'
    load_model_adapter(model.backend_options['adapter'], {}).validate(model)


def test_checkpoint_metadata_discovery(assets):
    ckpt, source, processor, parts = assets
    bundled = ckpt / 'configs/data/parts_meta/yam.yaml'
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(parts.read_bytes())
    assert yam_g05.validate_assets(ckpt, source, processor)[3] == bundled


def test_source_metadata_fallback(assets):
    ckpt, source, processor, parts = assets
    bundled = source / 'configs/data/parts_meta/yam.yaml'
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(parts.read_bytes())
    assert yam_g05.validate_assets(ckpt, source, processor)[3] == bundled


def test_tokenizer_aliases_survive_upstream_sidecar_updates(assets):
    # The actual upstream loader uses these three OmegaConf.update calls.
    # Keeping the checkpoint aliases caused _target_ to disappear at startup.
    OmegaConf = pytest.importorskip('omegaconf').OmegaConf
    ckpt, source, processor, parts = assets
    original = (ckpt / '.hydra/config.yaml').read_text()
    staged = yam_g05.stage_config_text(original, parts)
    cfg = OmegaConf.create(staged)
    saved = yaml.safe_load(original)['tokenizer']
    for key in ('tokenizer.vq_config.ckpt_dir', 'model.tokenizer.vq_config.ckpt_dir',
                'model.model_arch.AT_CONFIG.ckpt_dir'):
        if OmegaConf.select(cfg, key) is not None:
            OmegaConf.update(cfg, key, '/local/action_tokenizer.pt', merge=False)
    assert cfg.model.model_arch.action_tokenizer == saved['_target_']
    expected = dict(saved['vq_config'], ckpt_dir='/local/action_tokenizer.pt')
    for node in (cfg.tokenizer.vq_config, cfg.model.tokenizer.vq_config,
                 cfg.model.model_arch.AT_CONFIG):
        assert OmegaConf.to_container(node, resolve=True) == expected
    assert (ckpt / '.hydra/config.yaml').read_text() == original
