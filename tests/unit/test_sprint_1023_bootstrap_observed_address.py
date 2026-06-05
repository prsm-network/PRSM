"""Sprint 1023 — bootstrap server reports each node its server-observed address.

Tier-1 audit gap-1 (F14 auto-advertise) FOUNDATION. The bootstrap server already
computes the observed client_ip during register (server.py:447-458) but never told
the node. Now register_ack carries observed_address = "<observed_ip>:<declared_port>",
so a node behind NAT / co-located with the server can learn how the world actually
sees it — without a manual PRSM_ADVERTISE_ADDRESS. This is a genuine connectivity
diagnostic on its own; wiring it as the discovery auto-advertise fallback (so the F14
co-location rewrite fires automatically) is the documented follow-on.
"""
from __future__ import annotations

import json
from typing import Any, Optional
from unittest.mock import AsyncMock

import pytest
import websockets

from prsm.bootstrap.client import BootstrapClient
from prsm.bootstrap.server import BootstrapServer
from prsm.bootstrap.config import BootstrapConfig


@pytest.mark.asyncio
async def test_register_ack_reports_observed_address():
    """The ack reports the server-seen source IP joined to the node's DECLARED
    listen port (a routable dial target, not the ephemeral WS source port)."""
    server = BootstrapServer(BootstrapConfig())
    mock_ws = AsyncMock()
    mock_ws.send = AsyncMock()
    mock_ws.remote_address = ("203.0.113.7", 54321)  # 54321 = ephemeral source port

    await server._handle_register(
        mock_ws,
        {"type": "register", "peer_id": "p1", "port": 9001,
         "capabilities": [], "version": "t"},
        "203.0.113.7",
    )

    sent = json.loads(mock_ws.send.call_args[0][0])
    assert sent["type"] == "register_ack"
    assert sent["observed_address"] == "203.0.113.7:9001"


class _StubServer:
    """Minimal real-WebSocket bootstrap stub whose register_ack carries
    observed_address (mirrors the canonical protocol)."""

    def __init__(self, *, observed_address: Optional[str] = "198.51.100.9:9001") -> None:
        self._server = None
        self.port: Optional[int] = None
        self._observed_address = observed_address

    async def __aenter__(self) -> "_StubServer":
        self._server = await websockets.serve(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    async def _handle(self, ws) -> None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "register":
                ack = {
                    "type": "register_ack",
                    "peer_id": msg.get("peer_id"),
                    "peers": [],
                    "heartbeat_interval": 30,
                    "server_time": "2026-06-05T00:00:00+00:00",
                }
                if self._observed_address is not None:
                    ack["observed_address"] = self._observed_address
                await ws.send(json.dumps(ack))
            elif msg.get("type") == "disconnect":
                return


@pytest.mark.asyncio
async def test_client_exposes_observed_address_from_ack():
    async with _StubServer(observed_address="198.51.100.9:9001") as srv:
        c = BootstrapClient(
            bootstrap_url=srv.url, node_id="probe", port=9001,
            capabilities=["x"], version="t",
        )
        await c.connect()
        try:
            assert c.observed_address == "198.51.100.9:9001"
        finally:
            await c.disconnect()


@pytest.mark.asyncio
async def test_client_observed_address_none_when_server_omits_it():
    """A pre-sprint-1023 server omits observed_address → the client property stays
    None (graceful, no crash)."""
    async with _StubServer(observed_address=None) as srv:
        c = BootstrapClient(
            bootstrap_url=srv.url, node_id="probe", port=9001,
            capabilities=["x"], version="t",
        )
        await c.connect()
        try:
            assert c.observed_address is None
        finally:
            await c.disconnect()


def test_client_observed_address_defaults_none_before_connect():
    c = BootstrapClient(
        bootstrap_url="ws://127.0.0.1:1", node_id="probe", port=9001,
        capabilities=["x"], version="t",
    )
    assert c.observed_address is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
