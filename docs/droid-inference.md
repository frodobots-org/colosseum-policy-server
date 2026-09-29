# DROID model loading and inference

Policy Server includes model workers based on the deployed five-model workflow.
Install each upstream SDK and provide your own downloaded assets. Policy supplies
the launch entry point, weight loading, observation preparation, inference call,
action conversion and Local Protocol response. No unpublished site scripts or
control project are required. Model weights are not included.

Each model runs in its own environment because the SDK dependencies differ.
Policy receives `prepare`, starts the selected worker, sends observations through
its `droid` backend and returns validated actions. Workers bind to loopback.
Neither the worker nor the backend commands a robot.

## Actual loading and inference locations

| Model | Loading | Inference |
| --- | --- | --- |
| MolmoAct2 | `MolmoRuntime`: `AutoProcessor.from_pretrained`, `AutoModelForImageTextToText.from_pretrained` | `predict_action` |
| LeRobot π0.5 | `Pi05Runtime`: `PreTrainedConfig.from_pretrained`, `PI05Policy.from_pretrained` | `predict_action_chunk` and original OpenPI quantile conversion |
| GR00T N1.7 | `load_groot`: `Gr00tPolicy(model_path=...)` and local Cosmos processor | upstream `PolicyServer` calls the loaded policy's `get_action` |
| LAP-3B | `load_lap`: `create_trained_policy(config, checkpoint)` | upstream OpenPI server calls the loaded policy's `infer` |
| G05 | `run_g05`: installed upstream `scripts/serve_policy.py` loads the trained run and `PolicyInferencer` | upstream inferencer executes its model |

The first four loaders are in [model_loading.py](../src/colosseum_policy_server/model_loading.py).
The worker CLI, HTTP/WebSocket services and upstream server startup are in
[model_service.py](../src/colosseum_policy_server/model_service.py).
[pi05_contract.py](../src/colosseum_policy_server/pi05_contract.py) preserves the
deployed normalization, prompt and action projection. Neural-network architectures
remain installed SDK dependencies, not copied source. G05 must be obtained and
used under its own license.

## Install the Policy process

In a separate Python 3.10+ environment, from this checkout:

```bash
python -m pip install -e '.[droid]'
python -c 'from colosseum_policy_server.local_runtime import load_backend; print(type(load_backend("droid", {})).__name__)'
```

Expect `DroidBackend`; this check loads no model. `PYTHONPATH` alone does not
register the plugin: install the package.

## Prepare model environments

Use one environment per upstream SDK. These source commits and key dependency
versions were read from the deployed environments. They are reproduction
references; CUDA wheels must match your GPU. SDK source revisions are distinct
from checkpoint revisions.

| Worker | Public SDK and source commit | Deployed key dependencies |
| --- | --- | --- |
| Molmo | [allenai/molmoact2](https://github.com/allenai/molmoact2) `66b87e64efd99dfd103241418113955cf64dfa9c` | Python 3.12, torch 2.11.0+cu128, transformers 4.57.6 |
| π0.5 | [huggingface/lerobot](https://github.com/huggingface/lerobot) `8c894413c0967d83624d17a440afd70754fddb01` | Python 3.12, lerobot 0.6.2, torch 2.11.0+cu128, transformers 5.5.4 |
| GR00T | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) `51d4c89f72fda44cbf77285c6a8114b52676b8a1` | Python 3.12, torch 2.9.0+cu128, transformers 4.57.3 |
| LAP | [lihzha/lap](https://github.com/lihzha/lap) `3958d1466d5b92445b67de7d4202c19608ad4d56` | Python 3.11, torch 2.7.1, transformers 4.53.2, jax 0.5.3, flax 0.10.2, tensorflow 2.15.0 |
| G05 | [OpenGalaxea/GalaxeaVLA](https://github.com/OpenGalaxea/GalaxeaVLA) `89f2322b4ad016e192437adc1a2c253b05bab246` | Python 3.10, torch 2.7.1+cu128, transformers 4.57.1 |

Follow the installation instructions at the selected revision. Molmo and GR00T
document recursive submodules and `uv sync` (GR00T: `uv sync --python 3.12`).
LAP documents `GIT_LFS_SKIP_SMUDGE=1 uv sync` followed by
`GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .`. G05 documents
`uv sync --index-strategy unsafe-best-match`. Install LeRobot with its PI05 extra
and that revision's required Transformers implementation. Install the SDK, not
just its source directory on `PYTHONPATH`.

Then install this checkout into each worker environment **without resolving
Policy process dependencies**:

```bash
/path/to/model-env/bin/python -m pip install --no-deps -e .
/path/to/model-env/bin/python -m colosseum_policy_server.model_service --help
```

If that uv environment has no pip, use
`uv pip install --python /path/to/model-env/bin/python --no-deps -e .`.
Molmo uses upstream torch/transformers/Pillow/numpy and standard-library HTTP.
π0.5 also needs `msgpack>=1,<2` and `websockets>=15,<16` in its worker environment.
GR00T, LAP and G05 use their upstream transports. Preserve LAP's
TensorFlow/protobuf and LAP/G05 WebSocket versions: do not install `.[droid]`
in these model environments. Worker imports do not load Policy protobuf/SDK.
Do not blindly resynchronize an environment after installing the worker.

## Supply assets and launch a worker

These commands start real inference services when you run them. Replace every
`/path/to/...` with your local path; use each model's own environment.
Workers set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; obtain all weights
and gated tokenizer/processor assets first. No startup warmup is performed.

### MolmoAct2-DROID

Use the full [MolmoAct2-DROID](https://huggingface.co/allenai/MolmoAct2-DROID)
snapshot (`d8c1abd8a27d8e859455bbe514df2bcc617db0fb`), including processor,
`modeling_molmoact2.py` and `norm_stats.json` alongside weights.

```bash
colosseum-policy-model molmoact2 --checkpoint /path/to/molmo-snapshot --device cuda:0 --dtype bfloat16 --patch-molmo-bf16 --port 9101
```

Preserves deployed bfloat16, ten steps, `franka_droid` stats, continuous actions,
exterior-then-wrist images, language normalization, no depth reasoning, no CUDA
graph and no warmup. `--patch-molmo-bf16` applies the deployed trajectory-dtype
and tensor-to-NumPy fixes only to that snapshot's Python model source, keeping a
`.before-colosseum-bf16` backup. It edits no weights or unrelated HF caches;
unrecognized source fails before editing. Already patched source does not need
the flag. An instance input cast also preserves deployed bfloat16 behavior.
`--dtype float32` is an explicit alternative, not the deployed default.

### LeRobot π0.5 DROID

Supply the converted [lerobot/pi05_droid](https://huggingface.co/lerobot/pi05_droid)
checkpoint (`72824c0a93f00ce5bb8bedb7feb58953ba1da364`), local
`google/paligemma-3b-pt-224` tokenizer directory (reference
`35e4f46485b4d07967e7e9935bc3786aad50687c`), and original OpenPI
`pi05_droid/assets/droid/norm_stats.json`.

```bash
colosseum-policy-model pi05_lerobot --checkpoint /path/to/lerobot-pi05 --tokenizer /path/to/paligemma-tokenizer --stats /path/to/norm_stats.json --device cuda --num-steps 10 --port 9112
```

Requires the converted LeRobot checkpoint, not arbitrary OpenPI weights. It
loads strict float32 with compilation disabled, normalizes eight state values
to 32 padded values, builds the state-token prompt and supplies two BCHW RGB
images in [0,1]. The absent right wrist uses PI05's native missing-image mask.
`predict_action_chunk` returns finite 15×32 output; original quantile
unnormalization projects the first eight values: seven normalized joint
velocities and gripper position. Joint velocities outside [-1,1] are rejected.

### GR00T N1.7 DROID

Supply [GR00T-N1.7-DROID](https://huggingface.co/nvidia/GR00T-N1.7-DROID)
(`05e7cc97e40dbd33b0890c35cc0214fcb0547ab5`) and local
`nvidia/Cosmos-Reason2-2B` processor (reference
`9ce19a195e423419c349abfc86fd07178b230561`).

```bash
colosseum-policy-model groot_n17 --checkpoint /path/to/groot-snapshot --processor /path/to/cosmos-processor --device cuda --port 9103
```

Uses deployed `oxe_droid_relative_eef_relative_joint`, strict validation and
the local Qwen3VL processor override. Upstream `PolicyServer` owns ZeroMQ and
calls the loaded policy. Its decoded response becomes absolute N×8 actions.

### LAP-3B

Supply [LAP-3B](https://huggingface.co/lihzha/LAP-3B)
(`601db9c1ab4bcaf6dddb160c7b2dec589a67b730`) with checkpoint assets, plus
PaliGemma `tokenizer.model` from the tokenizer snapshot above.

```bash
colosseum-policy-model lap_3b --checkpoint /path/to/lap-snapshot --tokenizer /path/to/paligemma-tokenizer/tokenizer.model --port 9105
```

Uses `lap` flow config with `stop_action_to_vlm_grad=False`. TensorFlow stays
off GPU; JAX defaults to CUDA, preallocation off and platform allocator.
Upstream OpenPI calls `policy.infer`; the backend converts relative
xyz/Euler XYZ/gripper to absolute Cartesian N×7.

### G05

Install upstream under its license. Supply the full
[G05](https://huggingface.co/OpenGalaxea/G05) snapshot (reference
`e312be81e90c56a55bcb26b57429bd39a335b449`), retaining this run layout:

```text
g05-droid/
  .hydra/config.yaml
  checkpoints/model_state_dict.pt
  dataset_stats.json
  action_tokenizer.pt
  hf_processor/
```

The download places `action_tokenizer.pt` and `qwen3_5_2b_base_processor/` next
to `g05-droid/`. Create the same sidecars used by the deployment, if missing:

```bash
cd /path/to/g05-snapshot/g05-droid
ln -s ../action_tokenizer.pt action_tokenizer.pt
ln -s ../qwen3_5_2b_base_processor hf_processor
```

Upstream resolves these sidecars and reads the run config, statistics and
weights. For a customized config, check its processor/tokenizer paths point at
local assets. A lone `.pt` file is insufficient.

```bash
colosseum-policy-model g05 --source-root /path/to/GalaxeaVLA --checkpoint /path/to/g05-snapshot/g05-droid/checkpoints/model_state_dict.pt --device cuda --port 9104
```

Invokes installed upstream `scripts/serve_policy.py` from its source root and
preserves the deployed SDPA fallback. `--g05-native-attention` explicitly disables
that override. No G05 model source/config is vendored. The backend supplies
`Droid_Franka` observations at 15 Hz and returns the next joint target with the
deployed gripper inversion.

## Complete Local Policy startup

Copy [local-runtime-five-models-example.yaml](../configs/local-runtime-five-models-example.yaml)
to a local config. Replace every `/path/to/...` with worker interpreters, assets
and installed SDK paths. Identity, revision, action space and horizon must match
the Router assignment; strict contract checks remain enabled.

```bash
colosseum-policy-local --config /path/to/your-local-runtime.yaml
```

Run this in the Policy environment. On Client `prepare`, Policy starts the chosen
worker. On Observation, the model executes and Policy returns an ActionPlan.
Launcher lists are argv, not shell text: `~`, environment variables and wildcards
are not expanded. Remove a launcher only when managing that worker yourself.
A listening port indicates finished loading, not executed robot motion.

Default images are RAW_RGB `head_image` and `left_image`; configure
`backend.options.external_sensor`/`wrist_sensor` as needed. State needs seven
joints and gripper; GR00T and LAP additionally need xyz/Euler XYZ.

| Adapter | Transport | Returned action space |
| --- | --- | --- |
| `molmoact2` | HTTP `/act`, JSON NumPy | joint_position, N×8 |
| `pi05_lerobot` | WebSocket, OpenPI NumPy MessagePack | joint_velocity, 15×8 |
| `groot_n17` | ZeroMQ, GR00T MessagePack | joint_position, N×8 |
| `g05` | WebSocket, OpenPI NumPy MessagePack | joint_position, 1×8 |
| `lap_3b` | WebSocket, OpenPI NumPy MessagePack | cartesian_position, N×7 |

## Validation

Packaging preserves deployed model calls, precision, preprocessing and fixes.
Changes include configurable paths, loopback binding, scoped Molmo patching and
sanitized errors. Offline tests use fake SDKs, real NumPy preprocessing and fake
loopback services; they load no weights.

```bash
python -m pip install -e '.[dev,droid]'
python -m pytest -q
```

The existing deployment is the behavioral reference. This packaging change has
not rerun models on GPU; another machine's behavior depends on its SDKs/assets.
