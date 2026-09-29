# Add a robot backend

A local backend translates a robot observation into a model request and converts
model output into robot actions. The runtime handles Protobuf, sessions, deadlines,
and action shape validation. A backend does not send commands to robot hardware.

## Code layout

- `src/colosseum_policy_server/backends/base.py`: public `PolicyBackend` interface
  and `LocalModel` configuration.
- `src/colosseum_policy_server/backends/droid.py`: built-in DROID service adapters.
- `src/colosseum_policy_server/local_runtime.py`: plugin loading and Local Protocol.
- `src/colosseum_policy_server/droid_backend.py`: compatibility imports for older users.

New plugins should import `LocalModel` and `PolicyBackend` from
`colosseum_policy_server.backends`. Explicit inheritance is optional.

## Contract

An installed Python entry point in `colosseum_policy_server.backends` names a
factory. The factory receives the YAML `backend.options` dictionary and returns
an object implementing:

```python
async def infer(self, model: LocalModel, observation: pb.Observation) -> np.ndarray:
    ...
```

`model` contains URL, pinned revision, action space, action dimension, control
frequency, maximum horizon, service endpoint, optional launcher argv, and per-model
`backend_options`. It is metadata, not a loaded neural network. Return a finite
NumPy array shaped `(horizon, model.action_dim)`, with
`1 <= horizon <= model.max_horizon`. Columns, units, coordinate frames, absolute
versus relative targets, and gripper conventions must agree with the Robot Client.
The shared runtime allows positive action dimensions and nonempty action-space
names; each backend must enforce its robot-specific contract.

`observation` is the protobuf `Observation`: `instruction`, named `state` tensors,
`sensors` with IDs/encodings/dimensions, `control_step`, and timestamps. Decode
state with `colosseum_policy_server.tensors.tensor_to_numpy`. Image decoding and
model preprocessing belong to the backend. DROID expects RAW_RGB; a new backend
must explicitly support whatever its Client sends.

## Minimal external plugin

Create a separate project with `pyproject.toml` and
`src/my_robot_backend/__init__.py`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-robot-backend"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["colosseum-policy-server", "numpy>=1.26"]

[project.entry-points."colosseum_policy_server.backends"]
my_robot = "my_robot_backend:create_backend"
```

This implementation template deliberately leaves the model-specific work explicit:

```python
import numpy as np
from colosseum_policy_server.backends import LocalModel
from colosseum_policy_server import colosseum_pb2 as pb


def create_backend(options):
    return MyRobotBackend(options)


class MyRobotBackend:
    def __init__(self, options):
        self.options = dict(options)
        # Initialize your service client or model cache here.

    async def infer(self, model: LocalModel, observation: pb.Observation) -> np.ndarray:
        if model.action_space != "joint_position" or model.action_dim != 6:
            raise ValueError("Expected six joint position targets")
        # Implement: decode observation, preprocess, call your model/service,
        # then convert output to six joint targets in the Client's units.
        # Return np.asarray(actions, dtype=np.float32), shaped (horizon, 6).
        raise NotImplementedError("Connect your model before running an evaluation")
```

The example uses six joints only to illustrate a different robot. Replace that
contract and implement the method for your hardware. Never substitute dummy
actions for model errors.

Install the plugin into the same environment as the local server, from its checkout:

```bash
uv pip install --python .venv/bin/python -e /path/to/my-robot-backend
.venv/bin/python -c 'from colosseum_policy_server.local_runtime import load_backend; print(type(load_backend("my_robot", {})).__name__)'
```

Use that environment directly when starting the server so a later environment
sync does not remove the separately installed plugin:

```bash
.venv/bin/colosseum-policy-local --config /path/to/my-robot.yaml
```

Example `my-robot.yaml` (replace URL, revision, endpoint, and action contract):

```yaml
host: 127.0.0.1
port: 8000
backend:
  name: my_robot
  options: {}
models:
  - name: my-model
    url: https://huggingface.co/example/model
    revision: "0000000000000000000000000000000000000000"
    action_space: joint_position
    action_dim: 6
    control_hz: 15
    max_horizon: 8
    endpoint: http://127.0.0.1:9100
    launcher: []
    backend_options: {}
```

Register matching model URL, revision, and action contract on the Router. The
Client forwards the Router's `prepare.model`; the runtime selects the matching
local entry and invokes your backend for observations. `runtime_profile` is
optional legacy metadata, echoed unchanged but not used for selection.
`subfolder` is also echoed but checkpoint selection by subfolder is not implemented.

## Model loading and lifecycle

A model service must be installed and started separately, or through the optional
`launcher` command. Alternatively, implement loading and caching within your
backend; include model URL and revision in cache identity. The current interface
only defines `infer`, with no `prepare`, `reset`, or `close` hook. The backend
instance survives across sessions; explicitly consider stateful policy reuse.

The runtime does not download weights or verify which checkpoint an external
service loaded. With a launcher it waits for a listening port; without a launcher
it does not probe the service. The `loaded=True` readiness reply therefore does
not establish completed weight loading or warm-up. Validate the service before
starting evaluations. Blocking model calls must not block the asyncio event loop;
thread offloading alone does not cancel underlying inference after a timeout.

## New robot integration and validation

1. Add/configure the Robot Client hardware adapter: sensor IDs, image encoding,
   state names/shapes, action execution and safety checks. A policy backend alone
   does not add hardware support to the Client.
2. Implement the backend input/output conversion and model loading/service client.
   Confirm joints, units, coordinate frames, gripper conventions and action space.
3. Test with synthetic observations and a fake model service: missing sensors,
   wrong shapes, invalid values, model errors and deadline failures.
4. Run Client → local runtime → real model service → action validation without
   enabling physical actuation first. Confirm the loaded checkpoint identity,
   cold-start readiness, latency, disconnect behavior and session reuse.
5. Validate robot execution separately using the Client's hardware safety process.

The real local runtime binds loopback WS. For another machine or WSS, provide a
separately configured secure transport/proxy; the verification server's TLS flags
are not available on `colosseum-policy-local`.
