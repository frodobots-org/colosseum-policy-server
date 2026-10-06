"""Call a running YAM VLA HTTP worker with synthetic observations; no robot access."""
import argparse
import json
import time
from urllib.parse import urlsplit
from urllib.request import Request, ProxyHandler, build_opener

import numpy as np

from colosseum_policy_server.numpy_wire import _json_array, _json_decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', required=True, help='e.g. http://127.0.0.1:8202')
    parser.add_argument('--max-horizon', type=int, required=True)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    url = urlsplit(args.endpoint)
    if url.scheme != 'http' or url.hostname not in {'127.0.0.1', 'localhost'}:
        parser.error('Use a loopback HTTP worker or SSH forwarding')
    if args.max_horizon < 1 or args.timeout <= 0:
        parser.error('Horizon and timeout must be positive')
    state = np.zeros(14, np.float32)
    state[[6, 13]] = 1
    data = {name: _json_array(np.zeros((480, 640, 3), np.uint8))
            for name in ('top_cam', 'left_cam', 'right_cam')}
    data.update(state=_json_array(state), instruction='Hold the current position.', num_steps=10)
    start = time.monotonic()
    request = Request(args.endpoint.rstrip('/') + '/act', data=json.dumps(data).encode(),
                      headers={'Content-Type': 'application/json'})
    with build_opener(ProxyHandler({})).open(request, timeout=args.timeout) as response:
        result = json.load(response)
    actions = np.asarray(_json_decode(result['actions']), dtype=np.float32)
    if (actions.ndim != 2 or actions.shape[1] != 14 or not 1 <= len(actions) <= args.max_horizon
            or not np.isfinite(actions).all()):
        raise ValueError(f'Invalid YAM actions: shape={actions.shape}')
    if np.any(actions[:, [6, 13]] < 0) or np.any(actions[:, [6, 13]] > 1):
        raise ValueError('Gripper commands outside [0, 1]')
    print(f'PASS: shape={actions.shape}, elapsed={time.monotonic()-start:.2f}s; synthetic inputs only, no robot commands')


if __name__ == '__main__':
    main()
