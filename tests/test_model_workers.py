import contextlib
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import ModuleType, SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from colosseum_policy_server import model_loading as loading, model_service as service
from colosseum_policy_server.numpy_wire import _json_array, _json_decode, _msgpack_default, _msgpack_object


def module(monkeypatch, name, **members):
    obj = ModuleType(name)
    obj.__dict__.update(members)
    monkeypatch.setitem(sys.modules, name, obj)
    return obj


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.dtype = self.value.dtype

    def to(self, *args, **kwargs):
        return self

    def float(self):
        return Tensor(self.value.astype(np.float32))

    def bool(self):
        return Tensor(self.value.astype(bool))

    def cpu(self):
        return self

    def detach(self):
        return self

    def is_floating_point(self):
        return self.value.dtype.kind == 'f'

    def numpy(self):
        return self.value

    def __getitem__(self, key):
        return Tensor(self.value[key])

    def __truediv__(self, value):
        return Tensor(self.value / value)


def fake_torch(monkeypatch):
    return module(monkeypatch, "torch", from_numpy=Tensor, float32="float32", bfloat16="bfloat16",
                  is_tensor=lambda v: isinstance(v, Tensor), inference_mode=contextlib.nullcontext,
                  no_grad=contextlib.nullcontext)


def test_pi05_loads_checkpoint_and_preserves_deployed_inputs(monkeypatch, tmp_path):
    fake_torch(monkeypatch)
    calls = {}

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls['tokenizer_load'] = path, kwargs
            return cls()

        def __call__(self, prompts, **kwargs):
            calls['prompt'] = prompts[0]
            return {'input_ids': Tensor([[1, 2]]), 'attention_mask': Tensor([[1, 1]])}

    class Policy:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls['load'] = path, kwargs
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def predict_action_chunk(self, batch, **kwargs):
            calls['batch'], calls['infer_options'] = batch, kwargs
            return Tensor(np.zeros((1, 15, 32), np.float32))

    module(monkeypatch, 'transformers', AutoTokenizer=Tokenizer)
    module(monkeypatch, 'lerobot')
    module(monkeypatch, 'lerobot.configs', PreTrainedConfig=SimpleNamespace(from_pretrained=lambda _: SimpleNamespace()))
    module(monkeypatch, 'lerobot.policies')
    module(monkeypatch, 'lerobot.policies.pi05', PI05Policy=Policy)
    module(monkeypatch, 'lerobot.utils')
    module(monkeypatch, 'lerobot.utils.constants', OBS_STATE='state', OBS_LANGUAGE_TOKENS='tokens', OBS_LANGUAGE_ATTENTION_MASK='mask')
    stats = {'norm_stats': {name: {'q01': [-1.] * 8 + [0.] * 24, 'q99': [1.] * 8 + [0.] * 24} for name in ['state', 'actions']}}
    stats['norm_stats']['actions']['q01'][7] = 0
    path = tmp_path / 'norm_stats.json'
    path.write_text(json.dumps(stats))
    runtime = loading.Pi05Runtime(tmp_path, tmp_path, path)
    image = np.full((2, 3, 3), 255, dtype=np.uint8)
    result = runtime.infer({'observation/joint_position': np.zeros(7), 'observation/gripper_position': [.5],
                            'observation/exterior_image_1_left': image, 'observation/wrist_image_left': image,
                            'prompt': ' close '})
    assert calls['tokenizer_load'][1] == {'local_files_only': True}
    assert calls['load'][1]['strict'] is True
    assert calls['load'][1]['config'].dtype == 'float32'
    assert calls['load'][1]['config'].compile_model is False
    assert calls['prompt'].startswith('Task: close, State: 128 128 ')
    assert calls['infer_options'] == {'num_steps': 10}
    batch = calls['batch']
    assert batch['state'].value.shape == (1, 32)
    np.testing.assert_array_equal(batch['state'].value[0, 8:], 0)
    np.testing.assert_array_equal(batch['observation.images.base_0_rgb'].value, np.ones((1, 3, 2, 3)))
    assert 'observation.images.right_wrist_0_rgb' not in batch
    np.testing.assert_allclose(result['actions'][:, -1], .5)
    assert result['actions'].shape == (15, 8)
    assert result['raw_actions_15x32'].shape == (15, 32)


def test_molmo_patch_is_scoped_idempotent_and_fails_closed(tmp_path):
    path = tmp_path / 'modeling_molmoact2.py'
    original = ('device=device,\n            dtype=torch.float32,\n            generator=generator,\n'
                'return value.detach().cpu().numpy().astype(np.float32, copy=False)')
    path.write_text(original)
    weight = tmp_path / 'model.safetensors'
    weight.write_bytes(b'untouched')
    loading.patch_molmo_bf16(tmp_path)
    assert 'source_tensor.dtype' in path.read_text()
    assert '.cpu().float().numpy()' in path.read_text()
    first = path.read_bytes()
    loading.patch_molmo_bf16(tmp_path)
    assert path.read_bytes() == first
    assert path.with_suffix('.py.before-colosseum-bf16').read_text() == original
    assert weight.read_bytes() == b'untouched'
    path.write_text('different upstream version')
    with pytest.raises(ValueError, match='does not match'):
        loading.patch_molmo_bf16(tmp_path)
    assert path.read_text() == 'different upstream version'


def native_molmo_bf16_source():
    return (Path(__file__).parent / 'fixtures/molmo_yam_native_bf16.txt').read_text()


def test_molmo_native_bf16_needs_no_patch_or_backup(tmp_path):
    path = tmp_path / 'modeling_molmoact2.py'
    original = native_molmo_bf16_source()
    path.write_text(original)
    for _ in range(2):
        loading.patch_molmo_bf16(tmp_path)
    assert path.read_text() == original
    assert not path.with_suffix('.py.before-colosseum-bf16').exists()
    assert all(loading._molmo_bf16_supported(original))


@pytest.mark.parametrize('before,after', [
    ('trajectory_dtype = action_expert.action_embed.weight.dtype', 'trajectory_dtype = torch.float32'),
    ('tensor = tensor.float()', 'tensor = tensor'),
])
def test_molmo_incomplete_native_fix_rejected_without_mutation(tmp_path, before, after):
    path = tmp_path / 'modeling_molmoact2.py'
    original = native_molmo_bf16_source().replace(before, after)
    path.write_text(original)
    with pytest.raises(ValueError, match='does not match'):
        loading.patch_molmo_bf16(tmp_path)
    assert path.read_text() == original
    assert not path.with_suffix('.py.before-colosseum-bf16').exists()


@pytest.mark.parametrize("native_bf16", [False, True])
@pytest.mark.parametrize("robot_type,dim,keys,tag", [
    ("franka", 8, ("external_cam", "wrist_cam"), "franka_droid"),
    ("yam", 14, ("top_cam", "left_cam", "right_cam"), "yam_dual_molmoact2"),
])
def test_molmo_load_and_real_predict_entry_use_deployed_precision(monkeypatch, tmp_path, robot_type, dim, keys, tag, native_bf16):
    fake_torch(monkeypatch)
    calls = {}
    (tmp_path / 'modeling_molmoact2.py').write_text(native_molmo_bf16_source() if native_bf16 else '# patched_bf16_dtype\n# patched_bf16_to_array')

    class Model:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls['load'] = path, kwargs
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def parameters(self):
            return iter([SimpleNamespace(dtype='bfloat16')])

        def predict_action(self, **kwargs):
            calls['infer'] = kwargs
            return SimpleNamespace(actions=Tensor(np.zeros((1, 3, dim))))

    module(monkeypatch, 'transformers', AutoModelForImageTextToText=Model,
           AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **k: 'processor'))
    module(monkeypatch, 'PIL', Image=SimpleNamespace(fromarray=lambda a: a))
    runtime = loading.MolmoRuntime(tmp_path, robot_type=robot_type)
    result = runtime.infer({'state': np.zeros(dim), 'instruction': 'close',
                            **{key: np.full((2, 2, 3), i, np.uint8) for i, key in enumerate(keys)}})
    assert calls['load'][1]['torch_dtype'] == 'bfloat16'
    assert calls['load'][1]['local_files_only'] is True
    assert calls['infer']['enable_cuda_graph'] is False
    assert calls['infer']['norm_tag'] == tag
    assert [image[0, 0, 0] for image in calls['infer']['images']] == list(range(len(keys)))
    assert calls['infer']['inference_action_mode'] == 'continuous'
    assert calls['infer']['num_steps'] == 10
    assert result['actions'].shape == (3, dim)


def test_groot_loader_keeps_deployed_embodiment_and_local_processor(monkeypatch, tmp_path):
    calls = {}
    processing = module(monkeypatch, 'gr00t.model.gr00t_n1d7.processing_gr00t_n1d7')
    module(monkeypatch, 'gr00t')
    module(monkeypatch, 'gr00t.model')
    module(monkeypatch, 'gr00t.model.gr00t_n1d7', processing_gr00t_n1d7=processing)
    module(monkeypatch, 'gr00t.data')
    module(monkeypatch, 'gr00t.data.embodiment_tags', EmbodimentTag=SimpleNamespace(resolve=lambda x: x))
    module(monkeypatch, 'gr00t.policy')
    module(monkeypatch, 'gr00t.policy.gr00t_policy', Gr00tPolicy=lambda **kwargs: calls.update(kwargs) or 'policy')
    module(monkeypatch, 'transformers', Qwen3VLProcessor=SimpleNamespace(from_pretrained=lambda p, **k: (p, k)))
    assert loading.load_groot(tmp_path, processor=tmp_path) == 'policy'
    assert calls['embodiment_tag'] == 'oxe_droid_relative_eef_relative_joint'
    assert calls['strict'] is True
    assert processing.build_processor('ignored', {}) == (str(tmp_path), {'local_files_only': True})


def test_lap_loader_preserves_flow_config_and_tensorflow_gpu_exclusion(monkeypatch, tmp_path):
    calls = []
    @dataclass
    class Model:
        stop_action_to_vlm_grad: bool = True
    @dataclass
    class Config:
        model: Model
    module(monkeypatch, 'tensorflow', config=SimpleNamespace(set_visible_devices=lambda *args: calls.append(args)))
    module(monkeypatch, 'lap')
    module(monkeypatch, 'lap.models')
    tokenization = module(monkeypatch, 'lap.models.tokenizer')
    config = SimpleNamespace(get_config=lambda name: Config(Model()))
    module(monkeypatch, 'lap.training', config=config)
    module(monkeypatch, 'lap.policies')
    module(monkeypatch, 'lap.policies.policy_config_adapter', create_trained_policy=lambda *args: args)
    result = loading.load_lap(tmp_path, tmp_path / 'tokenizer.model')
    assert calls == [([], 'GPU')]
    assert not result[0].model.stop_action_to_vlm_grad
    assert result[1] == str(tmp_path)
    assert tokenization.PALIGEMMA_TOKENIZER_MODEL_PATH == str(tmp_path / 'tokenizer.model')


def test_g05_invokes_installed_upstream_with_sdpa_and_restores_process_state(monkeypatch, tmp_path):
    vision = SimpleNamespace(_flash_attn_varlen='flash', _flash_attn_backend='flash')
    module(monkeypatch, 'g05')
    module(monkeypatch, 'g05.models')
    module(monkeypatch, 'g05.models.g05')
    module(monkeypatch, 'g05.models.g05.qwen35', vision=vision)
    calls = []
    monkeypatch.setattr(service.runpy, 'run_path', lambda *a, **k: calls.append((a, k, sys.argv.copy(), Path.cwd())))
    cwd, argv = Path.cwd(), sys.argv
    args = SimpleNamespace(source_root=tmp_path, checkpoint=tmp_path / 'model.pt', device='cuda',
                           host='127.0.0.1', port=9104, g05_native_attention=False)
    service.run_g05(args)
    assert vision._flash_attn_varlen is None and vision._flash_attn_backend == 'sdpa'
    assert calls[0][2][1:] == ['--ckpt_path', str(args.checkpoint), '--host', '127.0.0.1', '--port', '9104', '--device', 'cuda']
    assert calls[0][3] == tmp_path
    assert Path.cwd() == cwd and sys.argv is argv


def test_cli_rejects_invalid_assets_and_nonloopback_before_loading(tmp_path):
    for argv in [[], ['molmoact2', '--checkpoint', str(tmp_path), '--port', '0'],
                 ['molmoact2', '--checkpoint', str(tmp_path), '--port', '9101', '--host', '0.0.0.0'],
                 ['pi05_lerobot', '--checkpoint', str(tmp_path), '--port', '9112']]:
        with pytest.raises(SystemExit):
            service.parse_args(argv)


def test_worker_import_does_not_require_protocol_or_gpu_dependencies():
    code = '''
import sys
for name in ['google.protobuf', 'torch', 'transformers', 'tensorflow', 'jax', 'websockets']:
    sys.modules[name] = None
import colosseum_policy_server.model_service
assert 'colosseum_policy_server.colosseum_pb2' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], check=True)


def test_five_model_example_has_valid_contracts_and_worker_commands():
    from colosseum_policy_server.local_runtime import RuntimeConfig
    root = Path(__file__).resolve().parents[1]
    config = RuntimeConfig.from_yaml(root / 'configs/local-runtime-five-models-example.yaml')
    assert set(config.models) == {'molmoact2', 'pi05_lerobot', 'groot_n17', 'lap_3b', 'g05'}
    for name, model in config.models.items():
        assert model.launcher[1:4] == ('-m', 'colosseum_policy_server.model_service', name)
        assert model.backend_options['adapter'] == name
        assert model.launcher[model.launcher.index('--port') + 1] == model.endpoint.rsplit(':', 1)[1]
    assert config.models['pi05_lerobot'].action_space == 'joint_velocity'
    assert config.models['lap_3b'].action_dim == 7


@pytest.mark.parametrize('name', ['molmoact2', 'pi05_lerobot', 'groot_n17', 'lap_3b', 'g05'])
def test_cli_dispatches_all_workers_without_loading_models(monkeypatch, tmp_path, name):
    calls = []
    tokenizer = tmp_path / 'tokenizer'
    if name == 'lap_3b':
        tokenizer.write_text('fixture')
    else:
        tokenizer.mkdir()
    stats = tmp_path / 'stats.json'
    stats.write_text('{}')
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts/serve_policy.py').write_text('# fixture')
    checkpoint = tmp_path
    if name == 'g05':
        checkpoint = tmp_path / 'model.pt'
        checkpoint.write_text('fixture')
    monkeypatch.setattr(loading, 'MolmoRuntime', lambda *a, **k: calls.append('molmo') or object())
    monkeypatch.setattr(loading, 'Pi05Runtime', lambda *a, **k: calls.append('pi05') or object())
    monkeypatch.setattr(loading, 'load_groot', lambda *a, **k: calls.append('groot') or object())
    monkeypatch.setattr(loading, 'load_lap', lambda *a, **k: calls.append('lap') or SimpleNamespace(metadata={}))
    class Server:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def serve_forever(self): calls.append('serve')
        def run(self): calls.append('serve')
    monkeypatch.setattr(service, 'HTTPServer', Server)
    async def serve(*args): calls.append('serve')
    monkeypatch.setattr(service, 'serve_pi05', serve)
    monkeypatch.setattr(service, 'run_g05', lambda args: calls.extend(['g05', 'serve']))
    module(monkeypatch, 'gr00t')
    module(monkeypatch, 'gr00t.policy')
    module(monkeypatch, 'gr00t.policy.server_client', PolicyServer=Server)
    module(monkeypatch, 'openpi')
    module(monkeypatch, 'openpi.serving')
    module(monkeypatch, 'openpi.serving.websocket_policy_server', WebsocketPolicyServer=Server)
    service.main([name, '--checkpoint', str(checkpoint), '--tokenizer', str(tokenizer),
                  '--processor', str(tmp_path), '--stats', str(stats), '--source-root', str(tmp_path), '--port', '9100'])
    assert len(calls) == 2 and calls[-1] == 'serve'


def test_http_worker_roundtrip_and_sanitized_failure():
    class Runtime:
        def infer(self, request):
            assert request['external_cam'].dtype == np.uint8
            if request['instruction'] == 'fail':
                raise RuntimeError('secret/private/path')
            return {'actions': np.zeros((2, 8), np.float32)}
    server = service.HTTPServer(('127.0.0.1', 0), service.http_handler(Runtime()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f'http://127.0.0.1:{server.server_port}/act'
        payload = {'external_cam': _json_array(np.zeros((2, 3, 3), np.uint8)),
                   'wrist_cam': _json_array(np.zeros((2, 3, 3), np.uint8)),
                   'state': _json_array(np.zeros(8)), 'instruction': 'close'}
        with urlopen(Request(endpoint, data=json.dumps(payload).encode()), timeout=5) as response:
            actions = _json_decode(json.load(response)['actions'])
            assert actions.shape == (2, 8)
        payload['instruction'] = 'fail'
        with pytest.raises(HTTPError) as error:
            urlopen(Request(endpoint, data=json.dumps(payload).encode()), timeout=5)
        assert error.value.code == 500
        assert json.load(error.value) == {'error': 'Model inference failed'}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.asyncio
async def test_pi05_websocket_worker_roundtrip_and_sanitized_failure():
    msgpack = pytest.importorskip('msgpack')
    import websockets
    class Runtime:
        def infer(self, payload):
            if payload.get('fail'):
                raise ValueError('secret/private/path')
            assert payload['state'].shape == (8,)
            return {'actions': np.zeros((15, 8), np.float32)}
    handler = service.websocket_handler(Runtime(), {'action_horizon': 15})
    async with websockets.serve(handler, '127.0.0.1', 0) as server:
        port = server.sockets[0].getsockname()[1]
        async with websockets.connect(f'ws://127.0.0.1:{port}') as socket:
            assert msgpack.unpackb(await socket.recv()) == {'action_horizon': 15}
            await socket.send(msgpack.packb({'state': np.zeros(8)}, default=_msgpack_default))
            result = msgpack.unpackb(await socket.recv(), object_hook=_msgpack_object, strict_map_key=False)
            assert result['actions'].shape == (15, 8)
            await socket.send(msgpack.packb({'fail': True}))
            assert msgpack.unpackb(await socket.recv()) == {'error': 'Model inference failed'}


def test_molmo_mixed_native_and_legacy_fixes(tmp_path):
    path = tmp_path / 'modeling_molmoact2.py'
    original = native_molmo_bf16_source().replace('dtype=trajectory_dtype,', 'dtype=torch.float32,')
    path.write_text(original)
    loading.patch_molmo_bf16(tmp_path)
    assert all(loading._molmo_bf16_supported(path.read_text()))
    assert 'patched_bf16_dtype' in path.read_text()
    assert 'patched_bf16_to_array' not in path.read_text()
    assert path.with_suffix('.py.before-colosseum-bf16').read_text() == original
