"""Sprint 1400 — bootstrap client reconnects + re-registers when the server drops/restarts.

Live 2026-07-07: restarting the bootstrap server silently de-registered sfo — its heartbeat loop set
_connected=False and BROKE, never reconnecting, so the registry entry lapsed until an operator restart
(direct peer links persisted, so the network stayed up, but discovery for new joiners was degraded).
Now the heartbeat loop is persistent: while disconnected it reconnects via connect() (re-registering).
"""
import asyncio
from unittest.mock import AsyncMock

import pytest

from prsm.bootstrap.client import BootstrapClient, BootstrapPeer


def _client(on_peers=None):
    return BootstrapClient(
        bootstrap_url="ws://x:8765", node_id="n" * 32, port=9002,
        capabilities=["compute"], version="t", on_peers_discovered=on_peers)


@pytest.mark.asyncio
async def test_reconnect_once_reregisters_and_refires_peers():
    fired = []
    c = _client(on_peers=lambda peers: fired.append(peers))
    peers = [BootstrapPeer(peer_id="p2", address="1.2.3.4", port=9001)]
    ws = AsyncMock()
    c._ws = ws
    c.connect = AsyncMock(return_value=peers)
    await c._reconnect_once()
    c.connect.assert_awaited_once()            # re-registered via connect()
    ws.close.assert_awaited_once()             # half-open socket cleaned up first
    assert c._peers == peers
    assert fired == [peers]                     # on_peers_discovered re-fired → auto-dial re-runs


@pytest.mark.asyncio
async def test_reconnect_once_tolerates_server_still_down():
    c = _client()
    c.connect = AsyncMock(side_effect=ConnectionError("down"))
    c._ws = None
    await c._reconnect_once()                    # must NOT raise
    assert c._connected is False


@pytest.mark.asyncio
async def test_heartbeat_loop_reconnects_instead_of_exiting():
    c = _client()
    c._connected = False                         # simulate a dropped/restarted server
    c._ws = None
    c.heartbeat_interval = 0.01
    reconnected = asyncio.Event()

    async def _rc():
        reconnected.set()
    c._reconnect_once = _rc

    task = asyncio.create_task(c._heartbeat_loop())
    try:
        # deterministic: the loop must ATTEMPT a reconnect (not exit) while disconnected
        await asyncio.wait_for(reconnected.wait(), timeout=3.0)
        assert not task.done()                   # loop did NOT exit on disconnect
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
