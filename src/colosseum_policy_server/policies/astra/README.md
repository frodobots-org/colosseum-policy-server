# Astra

An Open-track image-goal baseline for Earth Rover Mini+. The reusable planner comes from `frodobots-astra-demo` revision `d83411fcf10692e1738d46b0c7702ee6ada4c0f0`; runtime imports stay entirely inside this package. No demo website, embedded rover SDK, LocalNav model or Jev service is required.

```sh
python -m pip install -e '.[astra]'
colosseum-policy-astra --config /path/to/astra.yaml
```

Copy `configs/astra.example.yaml`; set `COLOSSEUM_POLICY_TOKEN` and `OPENAI_API_KEY` in the environment. Use the policy token registered for deployment `Astra` in RoboColosseum. Set `url` to the router's WebSocket origin and select the exact model identifier available to your account. The package never loads the original demo's `.env`. `ASTRA_API_BASE`, `ASTRA_SERVICE_TIER`, `ASTRA_STREAM` and the planner's `ASTRA_*` prompt settings remain configurable. The existing explicit prompt-cache breakpoints are supported by the [Responses prompt caching documentation](https://developers.openai.com/api/docs/guides/prompt-caching).

Inputs are a fresh `front_image`, a sanitized `goal_image`, the mission instruction and an allowlist of non-positional telemetry. Photo goals receive no GPS or hidden judge coordinates. Turns use fresh compass feedback with configurable sign; reverse requires a fresh rear image. The controller derives bounded normalized `[linear, angular]` actions. The runner independently imposes qualified command limits, the speed cap, expiry, the 10 Hz control loop and stopping.

The adapter adds destination-and-heading instructions to the original prompt. It translates `look_around` to a single 30-degree left turn followed by a new observation; it omits the demo's motor auto-tuning and multi-camera scanning routines. Controller settings must be qualified on the actual rover. These are deliberate changes, identified by prompt `colosseum-photo-v1` and configuration version 1.

Model planning runs in a separate thread while the adapter returns zero commands; a decision older than eight seconds fails the policy. Motion responses contain three 10 Hz ticks (300 ms). `done` always carries zero motion. A model failure also returns zero motion with decision metadata so the runner can score the policy failure rather than silently dropping it.

The session handshake and decisions contain model identifier, Git commit, source hash, prompt version/hash, configuration version and controller settings. This records dirty working copies by their content hash; use committed code for published runs. The original source revision is also retained. No run is described as fine-tuned.

Tests use recorded response fixtures and synthetic photos, without API calls: `python -m pytest tests/test_astra.py tests/test_sdk.py tests/test_sync_sdk.py`.
