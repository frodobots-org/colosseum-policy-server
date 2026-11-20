# Earth Rover LLM agent

Cloud LLMs (Claude, GPT, Grok) can drive the Earth Rover Mini+ through the local
Policy Server, the same way they drive YAM, Franka and SO-101: the Router
assigns the model, the Robot Client runs the rover and sends observations to
`colosseum-policy-local`, and the Client supplies the provider's API key for
the trial.

The arm agents use Inspect Robots' joint tools. A rover has no joints, so
`backends/rover_agent.py` runs its own loop on Inspect Robots' provider clients
(Responses for GPT, Messages for Claude, Chat Completions for Grok, or the
Claude Code login for Claude). `backend.name: auto` sends every LLM assigned to
`robot_type: earth_rover_mini_plus` there.

## Stop and look

LLM calls take seconds, far slower than the rover's 10 Hz control loop. So the
rover never drives while the model thinks:

1. The model gets the front camera (and the rear one if configured), the GPS
   distance and direction to the goal, what its last move did by GPS and
   compass, and the time left.
2. It makes exactly one tool call:
   - `drive(forward, turn, seconds, note)`: up to `max_seconds`, then
     `settle_seconds` of zeros, so the next observation shows a stopped rover;
   - `done(note)`: it believes it has arrived;
   - `give_up(note)`: it is stuck or the goal is unsafe.
3. `done` and `give_up` send a zero `ActionPlan` with `done=True`, which ends
   the trial on the Client. Neither is a success claim: the operator scores the
   trial on the 0–4 navigation rubric as usual.

Out-of-range or malformed commands are returned to the model as errors, and it
gets three more tries before the inference fails. The Client's command sender
still stops the rover if no command arrives within `command_ttl_s`.

Heading comes from the compass, which can be tens of degrees off near
buildings. The agent learns an offset from the GPS course of straight drives of
at least 4 m (joined across consecutive moves), like the GeNIE baseline.

## Run it

```bash
uv sync --extra agent
cp configs/local-runtime-rover-agent.yaml.example configs/local-runtime-rover.yaml
uv run --no-sync colosseum-policy-local --config configs/local-runtime-rover.yaml
```

Robot Client (`robot.yaml`), in addition to the usual rover settings:

```yaml
policy_server_url: ws://127.0.0.1:8000
deadline_ms: 120000          # one turn can take several LLM calls
llm_api_keys:
  openai: env:OPENAI_API_KEY
  anthropic: env:ANTHROPIC_API_KEY
# or, for Claude through the operator's Claude subscription:
# llm_account_logins: {anthropic: claude-code}
```

With `policy_server_url` set, the Client asks the Router for local
assignments, and the Router only offers LLMs whose provider the Client has a key
for. Router deployments must match a `models` entry exactly: name, URL,
revision, `base_velocity`, 2 values, 10 Hz and `max_horizon`.

## Settings (`backend.options.rover`)

| Key | Default | Meaning |
|---|---|---|
| `max_forward` / `max_reverse` | 0.3 / 0.2 | normalized speed limits (the Client's `max_linear` scales them again) |
| `max_turn` | 0.6 | turning-rate limit, positive = left |
| `max_seconds` | 2.0 | longest single move |
| `settle_seconds` | 0.5 | zeros after each move |
| `arrive_m` | 8 | GPS distance at which the model is told it may call `done` |
| `sensors` | `[front_image]` | add `rear_image` if the Client records it |
| `image_horizon` | 2 | camera frames kept in the conversation; older ones are elided |
| `effort` | `low` | provider reasoning effort |
| `robot_notes` | built in | description of the rover given to the model |
| `log_dir` | `agent-logs` | one JSON record per trial: commands, notes, usage, transcript without images |

`max_horizon` in each model entry must cover `max_seconds + settle_seconds` at
10 Hz.

## Time limits

A trial ends at the task's `navigation.timeout_s`. Stop-and-look turns take
several seconds each, so give LLM tasks a few minutes more than the 180 s the
first Wuhan task allows a continuous policy. The agent is told the limit, and
how much is left, in every turn.
