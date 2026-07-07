"""Sprint 1398 — peer auto-discovery runs even when the transport bootstrap SUCCEEDS.

_try_bootstrap_client (register with the bootstrap server + get_peers + auto-dial) used to run only
in degraded mode (transport bootstrap FAILED). So two nodes that BOTH reached the signaling server
never registered/discovered/dialed each other — operators had to POST /peers/connect by hand
(2026-07-07 live: us + sfo showed 0 auto-peers). Now it runs whenever bootstrap_nodes exist.
"""
import asyncio
from unittest.mock import MagicMock

import pytest

from prsm.node.discovery import PeerDiscovery


async def _noop():
    return


def _make_discovery(bootstrap_nodes):
    transport = MagicMock()
    transport.identity = MagicMock()
    transport.identity.node_id = "n" * 32
    transport.port = 9002
    d = PeerDiscovery(transport=transport, bootstrap_nodes=bootstrap_nodes)
    d._announce_loop = _noop
    d._maintenance_loop = _noop
    return d


async def _run_start(d):
    calls = []

    async def _boot():
        d.bootstrap_degraded_mode = False   # transport bootstrap SUCCEEDS
        return 1
    d.bootstrap = _boot

    async def _tbc():
        calls.append(True)
        return True
    d._try_bootstrap_client = _tbc

    await d.start()
    for t in d._tasks:
        t.cancel()
    return calls


@pytest.mark.asyncio
async def test_discovery_runs_despite_successful_transport_bootstrap():
    d = _make_discovery(["wss://boot:8765"])
    calls = await _run_start(d)
    assert calls == [True]              # registered + discovered + auto-dialed anyway


@pytest.mark.asyncio
async def test_skipped_when_no_bootstrap_nodes():
    d = _make_discovery([])
    calls = await _run_start(d)
    assert calls == []                  # nothing to register with → no-op
