from __future__ import annotations

import asyncio
import inspect
import ssl
import time
from typing import Protocol

from websockets.asyncio.client import ClientConnection, connect

from . import colosseum_pb2 as pb

PROTOCOL_VERSION = 1


class Policy(Protocol):
    def metadata(self) -> pb.SessionReady: ...
    def infer(self, observation: pb.Observation, sequence: int) -> pb.ActionPlan: ...
    def reset(self, request: pb.ResetRequest) -> None: ...


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


class PolicyServer:
    def __init__(
        self,
        *,
        router_url: str,
        token: str,
        server_id: str,
        policy_id: str,
        policy: Policy,
    ) -> None:
        self.router_url = router_url
        self.token = token
        self.server_id = server_id
        self.policy_id = policy_id
        self.policy = policy
        self.connection: ClientConnection | None = None
        self.session_id = ""

    @staticmethod
    def _frame(message_type: int, payload=None, *, session_id: str = "", sequence: int = 0) -> pb.RelayFrame:
        return pb.RelayFrame(
            protocol_version=PROTOCOL_VERSION,
            type=message_type,
            session_id=session_id,
            sequence=sequence,
            sent_at_ns=time.time_ns(),
            payload=payload.SerializeToString() if payload is not None else b"",
        )

    async def run(self, *, ssl_context: ssl.SSLContext | None = None) -> None:
        options = {
            "additional_headers": {"Authorization": f"Bearer {self.token}"},
            "compression": None,
            "max_size": 32 * 1024 * 1024,
            "ping_interval": 10,
            "ping_timeout": 10,
        }
        if ssl_context is not None:
            options["ssl"] = ssl_context
        async with connect(self.router_url, **options) as connection:
            self.connection = connection
            hello = pb.Hello(role=pb.POLICY_SERVER, peer_id=self.server_id, policy_id=self.policy_id)
            await connection.send(self._frame(pb.HELLO, hello).SerializeToString())
            registered = await self._receive()
            if registered.type != pb.REGISTERED:
                raise RuntimeError("router did not register policy server")
            async for raw in connection:
                if isinstance(raw, str):
                    continue
                frame = pb.RelayFrame.FromString(raw)
                await self._handle(frame)

    async def _handle(self, frame: pb.RelayFrame) -> None:
        if self.connection is None:
            return
        if frame.type == pb.MATCHED:
            result = pb.MatchResult.FromString(frame.payload)
            self.session_id = result.session_id
            metadata = await _maybe_await(self.policy.metadata())
            await self.connection.send(
                self._frame(pb.SESSION_READY, metadata, session_id=self.session_id).SerializeToString()
            )
        elif frame.type == pb.OBSERVATION:
            if frame.session_id != self.session_id:
                return
            observation = pb.Observation.FromString(frame.payload)
            plan = await _maybe_await(self.policy.infer(observation, frame.sequence))
            plan.request_sequence = frame.sequence
            await self.connection.send(
                self._frame(pb.ACTION_PLAN, plan, session_id=self.session_id, sequence=frame.sequence).SerializeToString()
            )
        elif frame.type == pb.RESET:
            await _maybe_await(self.policy.reset(pb.ResetRequest.FromString(frame.payload)))
        elif frame.type == pb.SESSION_CLOSE:
            self.session_id = ""
        elif frame.type == pb.ERROR:
            error = pb.Error.FromString(frame.payload)
            raise RuntimeError(f"router error {error.code}: {error.message}")

    async def _receive(self) -> pb.RelayFrame:
        if self.connection is None:
            raise RuntimeError("not connected")
        raw = await self.connection.recv()
        if isinstance(raw, str):
            raise RuntimeError("binary protobuf frame required")
        return pb.RelayFrame.FromString(raw)
