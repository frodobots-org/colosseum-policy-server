# Colosseum Policy Server

For the deployed five-model loading and inference workflow, see
[DROID model setup](docs/droid-inference.md). This package provides the
`colosseum-policy-model` worker CLI, loading/inference entry points, protocol
adapters and action conversion. Install each upstream SDK in its own environment
and provide your own downloaded weights and tokenizer assets.

Serve your models for robot task evaluation on
[Robo Colosseum](https://frodobots-org.github.io/robo-colosseum/).

## Installation

Requirements: Python 3.10 or later, Git, and `uv` available on your PATH.
Run the following on the machine that will host the Policy Server:

```bash
git clone git@github.com:frodobots-org/colosseum-policy-server.git
cd colosseum-policy-server
uv sync
```

## Host a Local Policy Server

Use this option if you are an evaluator with your own local computer for
inference. Host the Policy Server on your computer, and connect the
[Robot Client](https://github.com/frodobots-org/colosseum-client) to it to run
evaluations using your own compute resources.

### Start the example server

This example simulates model preparation and inference without loading model weights.

From the `colosseum-policy-server` directory, run:

```bash
uv run colosseum-policy-verify
```

The server listens at `ws://127.0.0.1:8000`. Leave it running while you start the
Client. If the Client runs on another machine, use:

```bash
uv run colosseum-policy-verify --host 0.0.0.0 --port 8000
```

## Host a Remote Policy Server

Host your models on your own inference machine so evaluators can use your compute
resources and models to run evaluations. The
[Robot Client](https://github.com/frodobots-org/colosseum-client) exchanges
observations and actions with your Policy Server through the Router.

### Run the Example

Install the demo dependencies and create the Policy Server configuration:

```bash
uv sync --extra demo
cp configs/policy.yaml.example configs/policy.yaml
```

Edit it with your Router endpoint and **Policy Server token** from the Router admin:

```yaml
url: wss://router.example.com:8443
token: pol_replace_with_token_from_router_ui
```

Start the SDK example:

```bash
uv run --extra demo python examples/test_policy.py
```

The example connects to the Router and waits for observations. When an evaluator
starts a remote evaluation and the Router matches their Client to your Policy
Server, you will see task instructions, joint/gripper state, and image shapes in
the terminal. Camera images appear in OpenCV windows, so run this example with a
graphical display. Press `q` or Escape in an image window to exit.

The example also sends demo actions: it holds the observed joint positions and
alternates the gripper target between `1` and `0`. To only inspect incoming data,
run:

```bash
uv run --extra demo python examples/test_policy.py --no-enable-action
```

Observation-only mode sends no actions, so the Client may reach its inference
deadline. After checking the connection and observation format, replace the demo
policy with your own model using the SDK below.

### Connect your model

Install your model's dependencies in the same Python environment. Create
`serve_policy.py` in the repository root using this template:

```python
from colosseum_policy_server import ColosseumPolicySDK
from your_policy import load_model  # Replace with your own model loader.


def main():
    model = load_model()

    with ColosseumPolicySDK.from_yaml("configs/policy.yaml") as sdk:
        print("Connected to Router. Waiting for observations...", flush=True)
        while True:
            obs = sdk.get_obs()
            actions = model.infer(obs)
            sdk.send_action(actions, observation=obs)


if __name__ == "__main__":
    main()
```

`your_policy` is your own module, not a package included in this repository.
Replace the import and model-loading call with your implementation. Your
`model.infer(obs)` adapter should convert the observation to your model's inputs
and return a NumPy array shaped `(horizon, action_dim)` in the Client's agreed
action space.

For example, observations expose:

```python
obs.instruction        # Task instruction.
obs.state.head_image   # RGB uint8 image; None if absent.
obs.state.left_image
obs.state.right_image
obs.state.joints       # Robot joint positions.
obs.state.gripper      # Gripper state.
```

For JPEG/PNG observations, install the image-decoding dependency with
`uv sync --extra demo`.

The YAML-based SDK defaults to `joint_position`, 15 Hz, and a maximum 16-step
action horizon. These settings must suit your model and the Client. To customize
them, construct `ColosseumPolicySDK` directly with `router_url`, `token`,
`action_spaces`, `control_hz`, and `max_horizon` instead of using `from_yaml`.

After matching, `sdk.robot` exposes the Client's hardware specification and
`sdk.selected_action_space` identifies the selected action space. The SDK checks
action dimensions, horizon, finite values, and observation deadlines before
sending actions.

### Start the Policy Server

After implementing your model loader and inference adapter, run:

```bash
uv run python serve_policy.py
```

The script loads your model, connects to the Router, and waits for observations.
Keep it running while evaluators use your policy. Once the Router matches a
Client to your Policy Server, each observation is passed to your model and the
resulting actions are sent back through the Router.
