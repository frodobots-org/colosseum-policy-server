# Local Runtime Plugins

`colosseum-policy-local --config FILE` serves only loopback Local Protocol
traffic. Real inference belongs to an installed entry point in the
`colosseum_policy_server.backends` group. The backend returns a finite
two-dimensional NumPy action array. Do not commit model weights, credentials,
hardware addresses, camera identifiers, or machine-specific launch paths.

The entry point factory receives `backend.options` and returns an object with
`async infer(model, observation)`. `model` is public metadata only and
`observation` is the standard protobuf message. The runtime binds only a
loopback host. It reports startup, timeout, inference, and action-validation
failures through stable Local Protocol `ERROR` codes without exposing backend
paths or environment details.

## DROID External Services

Install `colosseum-policy-server[droid]` to enable the built-in `droid`
backend. It translates the standard protobuf observation and calls a
user-managed loopback service selected by each model's
`backend_options.adapter`: `molmoact2`, `pi05_lerobot`, `groot_n17`, `g05`, or
`lap_3b`. The backend never includes model weights, caches, model repositories,
robot imports, camera identifiers, or a machine-specific launcher.

Use `configs/local-runtime-droid-example.yaml` as a schema reference. Replace
all example identity values and endpoints with your own installed service. A
model entry may provide an optional launcher argv list, which the local runtime
executes as a separate process and records in its configured log directory.
