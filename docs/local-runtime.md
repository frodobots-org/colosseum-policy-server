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
