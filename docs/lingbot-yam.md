# LingBot-VLA v2 YAM in Colosseum

The `lingbot_v2 --robot-type yam` worker uses upstream's Python inference class
behind Colosseum's loopback HTTP `/act` service. Use `adapter: lingbot_yam` in the
Local Policy Server. Client changes and the upstream WebSocket server are not
needed. Hardware execution remains in the Client.

Supported snapshot: `helen9975/lingbot-vla-v2-6b-molmoact2-yam` at
`af7529f2ea27ff6e77911a98f86c40cfb75d229e`, checkpoint
`checkpoints/global_step_40000/hf_ckpt`. ModelScope mirror:
`aarontung/lingbot-vla-v2-6b-molmoact2-yam` (private). Keep the complete snapshot
layout: the worker needs the root training YAML, YAM robot YAML and normalization
statistics. It validates absolute 14-D state/action ordering and the 30-step
chunk. Cameras map head/left/right to dataset top/left/right. Upstream handles
normalization and unnormalization; the worker does not add state to actions.

## Download (on the GPU host)

The YAM snapshot is approximately 25.52 GB. Supply a ModelScope token with access
to the private repository. The token is read without terminal echo.

```bash
cd ~/colosseum-policy-server
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_PYTHON_DOWNLOADS=never
uv run --no-project --python .venv/bin/python --with modelscope python -c 'from getpass import getpass; from pathlib import Path; from modelscope import snapshot_download; snapshot_download("aarontung/lingbot-vla-v2-6b-molmoact2-yam", local_dir=str(Path.home()/"colosseum-policy-server/models/lingbot-vla-v2-6b-molmoact2-yam"), token=getpass("ModelScope token: "))'
uv run --no-project --python .venv/bin/python --with modelscope python -c 'from pathlib import Path; from modelscope import snapshot_download; snapshot_download("Qwen/Qwen3-VL-4B-Instruct", local_dir=str(Path.home()/"colosseum-policy-server/models/Qwen3-VL-4B-Instruct"), allow_file_pattern=["*.json", "*.txt", "*.jinja", "*.model"])'
```

The separate Qwen download is config/tokenizer/processor assets, not weights.
Do not substitute the GR00T Qwen3-VL-2B processor. Runtime Hub access is disabled.

## Dedicated GPU environment (upstream reference)

This section requires GitHub access. The ModelScope downloads above and a PyPI
mirror do not make the complete upstream installer China-only: its LeRobot URL
and FlashAttention build/download path can still access GitHub. A verified
China-only source/dependency bundle is not included in this change. If that is a
deployment requirement, do not run this reference installer yet. An existing
`.venv-lingbot-yam` can be retained while that bundle is prepared.

Prerequisites: working NVIDIA CUDA environment and Conda. The upstream script
installs Python 3.12, torch 2.8.0, FlashAttention 2.8.3 and its depth dependencies.
Use an independent environment to preserve existing Molmo/GR00T environments.

```bash
cd ~/colosseum-policy-server
mkdir -p third_party
git clone --recursive https://github.com/robbyant/lingbot-vla-v2.git third_party/lingbot-vla-v2
cd third_party/lingbot-vla-v2
bash tools/create_train_env.sh --env-name lingbot-yam
eval "$(conda shell.bash hook)"
conda activate lingbot-yam
python -m pip install -e ~/colosseum-policy-server
command -v colosseum-policy-model
git rev-parse HEAD
```

Record the upstream commit with your deployment. To resume an interrupted
install, use the same script with `--env-name lingbot-yam --resume`.
No copying into the source checkout is necessary: the worker stages the robot
configuration temporarily for upstream reset and supplies an absolute stats path.

## Worker and synthetic test

```bash
conda activate lingbot-yam
colosseum-policy-model lingbot_v2 \
  --robot-type yam \
  --checkpoint ~/colosseum-policy-server/models/lingbot-vla-v2-6b-molmoact2-yam/checkpoints/global_step_40000/hf_ckpt \
  --source-root ~/colosseum-policy-server/third_party/lingbot-vla-v2 \
  --processor ~/colosseum-policy-server/models/Qwen3-VL-4B-Instruct \
  --port 8206
```

Default precision is bfloat16; compilation is off. Optional `--lingbot-compile`
enables upstream compilation. Upstream hardcodes CUDA; select a GPU with
`CUDA_VISIBLE_DEVICES` rather than `--device cuda:N`.

In another terminal, using the normal Policy Server environment:

```bash
cd ~/colosseum-policy-server
uv run --no-sync python scripts/check_yam_worker.py \
  --endpoint http://127.0.0.1:8206 --max-horizon 30 --timeout 120
```

Expected contract: `shape=(30, 14)`. This sends synthetic images/state only and
never commands hardware. First GPU inference may take longer than steady state.

## Local launcher

Copy `configs/local-runtime-yam-lingbot.yaml.example` to a local YAML. Replace
launcher[0] with the absolute executable printed by `command -v` above, and check
all asset paths. Model identity/revision/profile/subfolder must match the Router.
Stop the manually started worker before using the automatic launcher.

```bash
colosseum-policy-local --config configs/local-runtime-yam-lingbot.yaml
```

The launcher activates the worker when a matching session is prepared.

LingBot output validation clips all finite gripper targets to `[0, 1]`
(normalized opening units), logging each clipped original value, side and
action index. For example, `1.003670` becomes `1.0`. Nonfinite outputs and
incorrect action shapes still fail. Arm
targets and input-state validation are unchanged; this is not a change to
Client gripper calibration or motor force limits.

Local contract tests use a fake upstream policy and cover state/camera mapping, absolute
actions, validation and protocol routing. They do not establish CUDA compatibility,
real-checkpoint inference, thermal safety or robot task performance. This is an
intermediate 40K-step checkpoint; its model card reports no robot evaluation.

Sources: [upstream setup](https://github.com/robbyant/lingbot-vla-v2/blob/main/tools/create_train_env.sh),
[upstream policy](https://github.com/robbyant/lingbot-vla-v2/blob/main/deploy/lingbot_vla_v2_policy.py),
[checkpoint card](https://huggingface.co/helen9975/lingbot-vla-v2-6b-molmoact2-yam).
