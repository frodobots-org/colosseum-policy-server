# Colosseum Policy Server

Pull-based Python SDK for running a policy behind a Colosseum Router. The policy host
doesn't need a public IP; it initiates an outbound WSS connection to the router.

## SDK usage

```python
from colosseum_policy_server import ColosseumPolicySDK


def main():
    sdk = ColosseumPolicySDK.from_yaml("configs/policy.yaml")

    with sdk:
        while True:
            obs = sdk.get_obs()
            actions = model.infer(obs)  # NumPy shape: (horizon, action_dim)
            sdk.send_action(actions)


main()
```

`configs/policy.yaml` contains only the Router URL and the Policy Server token:

```yaml
url: wss://router.example.com:8443
token: pol_replace_with_token_from_router_ui
```

Keep this file private and out of Git:

```bash
chmod 600 configs/policy.yaml
```

The Router derives `server_id` from the token record and uses the token name from the
admin UI as `policy_id`. The SDK defaults to `joint_position`, 15 Hz and a maximum
16-step action horizon.

`get_obs()` returns the latest unclaimed observation with NumPy robot state and images.
Its queue holds one item, so an observation that hasn't been claimed is
replaced when a newer one arrives. `send_action()` automatically correlates the action
chunk with the observation sequence and rejects duplicate, expired, non-finite or
oversized chunks.

After a Robot Client is matched, its registered hardware specification is available as:

```python
sdk.robot.robot_type
sdk.robot.joint_count
sdk.robot.has_gripper
sdk.robot.control_hz
sdk.robot.action_spaces       # e.g. {"joint_position": 8}
sdk.selected_action_space     # e.g. "joint_position"
```

`send_action()` enforces the selected action dimension. For example, if the Client
registers `joint_position: 8`, a chunk shaped `(16, 7)` or `(16, 9)` is rejected before
it reaches the robot.

For inference workers, retain the observation and pass it explicitly:

```python
obs = sdk.get_obs()
actions = run_inference(obs)
sdk.send_action(actions, observation=obs)
```

The blocking SDK runs WSS on a background event-loop thread. Applications that already
use asyncio can import `AsyncColosseumPolicySDK` and use `await get_obs()` / `await
send_action()` directly.

## Example

`examples/test_policy.py` prints instructions, joint/gripper state and image shapes in
the terminal. Camera frames are displayed in OpenCV windows; press `q` or
Escape to exit.

Policies can access standardized inputs directly. Missing cameras return `None`:

```python
obs.state.left_image    # RGB uint8 array shaped (height, width, 3)
obs.state.right_image
obs.state.head_image
obs.state.joints        # float32 array shaped (joint_count,)
obs.state.gripper       # float32 array shaped (1,)
```

```bash
uv sync --extra demo
uv run --extra demo python examples/test_policy.py
```

No action is sent by default. `--enable-action` holds the latest joint positions and
sends one second of a fixed gripper target per chunk, alternating between `1` and `0`:

```bash
uv run --extra demo python examples/test_policy.py --enable-action
```

The meaning and valid range of a gripper value are robot-specific. Confirm that the
Robot Client maps these values safely before using `--enable-action` on hardware.

Use `wss://` with a trusted TLS certificate outside local development.
