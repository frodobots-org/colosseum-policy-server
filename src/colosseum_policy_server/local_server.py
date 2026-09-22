"""CLI for the real loopback-only Local Protocol runtime."""
from __future__ import annotations

import argparse
import asyncio

from websockets.asyncio.server import serve

from .local_runtime import LocalPolicyRuntime, RuntimeConfig


async def run(config: RuntimeConfig) -> None:
    runtime = LocalPolicyRuntime(config)
    async with serve(runtime.handle, config.host, config.port, compression=None, max_size=32 * 1024 * 1024,
                     ping_interval=20, ping_timeout=60):
        print(f"Local Policy Server listening on ws://{config.host}:{config.port}; real model runtime only", flush=True)
        try:
            await asyncio.Future()
        finally:
            await runtime.supervisor.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve registered local VLA models over Local Protocol v1")
    parser.add_argument("--config", required=True, help="YAML local runtime configuration")
    args = parser.parse_args()
    try:
        asyncio.run(run(RuntimeConfig.from_yaml(args.config)))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
