# DROID inference with external model services

This package includes a real `droid` backend, registered in the
`colosseum_policy_server.backends` entry-point group. It decodes protobuf
observations, builds model-specific requests, calls a loopback model service,
converts its response into DROID actions, and returns those actions to the
Local Protocol runtime. It does not import RobotEnv or send robot commands.

## Install and check without loading a model

From this repository in an isolated Python 3.10+ environment:

```bash
python -m pip install -e '.[droid]'
python -c 'from importlib.metadata import entry_points; print([(e.name, e.value) for e in entry_points(group="colosseum_policy_server.backends")])'
python -c 'from colosseum_policy_server.local_runtime import load_backend; print(type(load_backend("droid", {})).__name__)'
```

Expect `droid` pointing to `colosseum_policy_server.droid_backend:create_backend`
and `DroidBackend`. These two checks do not start a service or load weights.
Setting `PYTHONPATH` alone does not install entry-point metadata; reinstall the
package when upgrading from a version without this backend.

## Where model loading happens

The model service owns checkpoint loading, tokenizer/normalization assets,
device placement, and neural-network forward execution. Install that service
and its model assets separately in the environment required by its upstream
project. Model code, weights, and their licenses remain with those projects.
This repository provides the client-side inference integration, not vendored
implementations of five neural networks or automatic checkpoint downloads.

Start the model service separately, or supply `models[].launcher` as an argv
list for a command you have installed. The runtime starts that process on
`prepare` and waits for its loopback endpoint. An open port indicates transport
availability, not a completed warm-up inference. With no launcher, the operator
is responsible for starting the external service before evaluation.

## Supported service contracts

Choose `backend.name: droid` and one `backend_options.adapter` per model:

| Adapter | Endpoint | Required service response | Returned action space |
| --- | --- | --- | --- |
| `molmoact2` | HTTP `/act` | JSON `actions`, list or json_numpy array | joint_position, N x 8 |
| `pi05_lerobot` | WebSocket | OpenPI NumPy MessagePack greeting then `actions` | normalized joint_velocity, N x 8 |
| `groot_n17` | ZeroMQ `tcp://` | `get_action` response containing decoded joint_position and gripper_position | joint_position, N x 8 |
| `g05` | WebSocket | OpenPI NumPy MessagePack `action` with right_arm and optional right_gripper | joint_position, 1 x 8 |
| `lap_3b` | WebSocket | OpenPI NumPy MessagePack relative xyz/Euler XYZ/gripper `actions` | cartesian_position, N x 7 |

These are explicit wire contracts. An upstream server with another API needs
a separately installed service wrapper; a model name alone does not establish
compatibility. In particular, the pi05 LeRobot conversion service is not
included here. No G05 implementation, weights, or inference server is copied
into this package; users must obtain and use their service under its license.

Observations contain RAW_RGB `head_image` and `left_image` by default; rename
them with `backend.options.external_sensor` and `wrist_sensor`. State includes
seven joint positions and one gripper position. GR00T and LAP additionally need
Cartesian xyz plus Euler XYZ. GR00T responses must already be decoded absolute
joint targets. LAP relative rotations compose with the current orientation;
G05 and LAP gripper conventions are inverted when producing DROID actions.

## Configure and run

Copy `configs/local-runtime-droid-example.yaml` to your own local configuration.
Replace its placeholder model URL, revision, profile, endpoint, and horizon
with values matching both your service and the Router assignment. Use `ws://`
for WebSocket adapters and `tcp://` for GR00T. LAP requires action_dim 7 and
cartesian_position; pi05 requires joint_velocity and action_dim 8.

```bash
colosseum-policy-local --config configs/local-runtime-droid-example.yaml
```

The example is a schema reference, not a preconfigured model deployment.
After a real Client supplies an observation, the backend performs external
service inference. Model errors and missing/null actions are rejected instead
of being converted to scalar arrays. The runtime enforces horizon, dimensions,
finite values, timeouts and sanitized protobuf errors.

## Offline validation and limits

```bash
python -m pip install -e '.[dev,droid]'
python -m pytest -q
```

Tests use synthetic arrays and fake services; they neither load weights nor
access hardware. Protocol and conversion tests do not certify model quality,
checkpoint compatibility, or successful robot motion. Each installed service
requires its own inference acceptance test before hardware use.
