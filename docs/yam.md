# MolmoAct2-BimanualYAM

The `molmoact2_yam` VLA adapter connects Colosseum's local protocol to the
[upstream YAM HTTP /act protocol](https://github.com/allenai/molmoact2/blob/6070080a20321b4f498ab30f28e1d09ac465edb7/examples/yam/host_server_yam.py).
It requires `robot_type: yam`, 14-D absolute `joint_position` at 30 Hz, three
RGB cameras, twelve joint positions and two normalized grippers. It does not
implement pi0.5 YAM. For LLM control, see the Inspect Agent section below.

## Model worker on the GPU host

Use a separate model environment with the dependencies from the
[official MolmoAct2 repository](https://github.com/allenai/molmoact2/tree/6070080a20321b4f498ab30f28e1d09ac465edb7).
Install this `colosseum-policy-server` package into that environment too.
Download a pinned `allenai/MolmoAct2-BimanualYAM` snapshot, including its remote
model Python files, processor/tokenizer assets and `norm_stats.json`. Point the
worker at that **local snapshot directory**. It loads offline, with
`trust_remote_code=True`; it does not download weights for you.

```bash
colosseum-policy-model molmoact2 --robot-type yam \
  --checkpoint /absolute/path/to/MolmoAct2-BimanualYAM-snapshot \
  --dtype float32 --port 8202
```

For bfloat16, use `--dtype bfloat16 --patch-molmo-bf16`. The patch backs up and
modifies only the selected snapshot's model source, and fails if it cannot
match the known upstream source patterns. Newer checkpoints may need an updated
patch; float32 avoids that patch requirement. Actual GPU/model loading has not
been validated by this change.

YAM inference uses `norm_tag=yam_dual_molmoact2`, images `[top, left, right]`,
continuous actions, no depth reasoning and no CUDA graph, following the
upstream example. Do not select `--robot-type yam` with DROID weights.

Alternatively, run the official `examples/yam/host_server_yam.py` service and
point the adapter at it. The default upstream script resolves a Hub snapshot
without pinning a revision, so make sure the worker's actual weights match the
revision registered on your Router. Our local worker makes that snapshot
selection explicit.

## Colosseum local server

From the `colosseum-policy-server` checkout:

```bash
uv sync
cp configs/local-runtime-yam.yaml.example configs/local-runtime-yam.yaml
# Edit model revision/contract to match your Router and actual snapshot.
uv run colosseum-policy-local --config configs/local-runtime-yam.yaml
```

The example uses the built-in `vla` backend; do not pass `--robot yam` (that
legacy option selects a backend entry point, not this model adapter).
A worker can be started separately, or via the optional absolute-path
`launcher` in the example configuration. The model service is loopback HTTP;
the Colosseum service is loopback WebSocket on port 8000. Forward port 8000 to
the robot host if needed.

The adapter converts the Client's split vectors into
`[left six joints, left gripper, right six joints, right gripper]` and sends
`top_cam`, `left_cam`, `right_cam`, `state`, `instruction`, `num_steps` using
json_numpy-compatible encoding. Actions stay in that same 14-value order.
Malformed dimensions, missing/duplicate cameras, nonfinite values, grippers
outside [0,1], and chunks longer than the configured horizon are rejected.

Tests cover mock-model loading, norm tag and camera ordering, real loopback HTTP
serialization, and local VLA routing. They do not prove GPU inference, measured
latency, motor operation or task success. See the
[Client YAM guide](../../colosseum-client/docs/yam.md) for hardware configuration
and shutdown behavior.

## Inspect Agent: Grok, Opus and GPT

The Colosseum Inspect Agent adapter supports the same YAM Client with
`grok-4.7`, `claude-opus-5-5`, and `gpt-6-astra` model identifiers. These reuse
the existing Chat Completions, Anthropic Messages, and Responses transports
respectively. This is adapter support; model availability, credentials and
relay compatibility must be checked against your actual provider.

```bash
uv sync --extra agent
cp configs/local-runtime-yam-agent.yaml.example configs/local-runtime-yam-agent.yaml
# Fill the twelve-joint limits, rig notes, and registered model identities.
uv run --no-sync colosseum-policy-local --config configs/local-runtime-yam-agent.yaml
```

Each entry uses 14-D `joint_position` at 30 Hz. `max_horizon: 150` caps a
returned trajectory at five seconds of nominal control time, not the LLM's
request frequency. Router YAM deployments must use the same contract; existing
8-D Franka model deployments cannot be reused as-is. This change does not
register or deploy Router entries.

The Client supplies API keys via its existing `llm_api_keys` configuration.
No model worker/GPU or `inspect-robots-yam` hardware plugin is needed for this
LLM path: the Colosseum Client owns I2RT and cameras. The Policy Server binds
Inspect's agent to the YAM action/state description.

Joint labels are `left_joint1` through `left_joint6`, then `left_gripper`,
`right_joint1` through `right_joint6`, then `right_gripper`. Both grippers are
continuous [0,1], 1=open. Bounds and per-step limits are packed in this order;
settings arrays remain twelve joint values, left then right. YAM requires
`gripper_open_value: 1` and all three camera roles.

When the agent requests done/give_up, measured joints form a fixed hold target
and each gripper retains its last executed command (or its measured opening
if no command has executed). Subsequent hold requests make no more LLM calls.
Operator scoring is still required; an agent stop does not establish success.
This powered hold during a trial differs from Client teardown, which releases
torque through I2RT close.

Offline tests exercise all three provider wire formats with the actual pinned
Inspect agent and mocked HTTP replies, including Protobuf action plans, both
arm indices, continuous grippers and stop handling. No live provider call or
physical YAM execution was performed.
