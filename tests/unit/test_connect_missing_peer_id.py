"""connect_to_peer detects missing /p2p/ suffix (sprint 121).

Pre-fix: raw bootstrap URLs without /p2p/<peerID> got passed
to PrsmConnect at the C bridge, which rejected with cryptic
"invalid p2p multiaddr: failed to extract peer info". Operators
saw the symptom but not the cause.

Post-fix: connect_to_peer detects missing /p2p/ before calling
the C bridge, logs a clear actionable warning, returns None
gracefully.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from prsm.node.libp2p_transport import Libp2pTransport


def _transport():
    """Build a Libp2pTransport stub without invoking dlopen.
    We only need connect_to_peer behavior, not real C calls."""
    t = Libp2pTransport.__new__(Libp2pTransport)
    t._handle = 1  # non-negative so connect_to_peer doesn't early-exit
    t._lib = MagicMock()
    t._read_and_free = MagicMock(return_value=None)
    t._telemetry = {"error_count": 0, "connect_count": 0}
    t._peers = {}
    return t


@pytest.mark.asyncio
async def test_missing_p2p_suffix_returns_none(caplog):
    """A real libp2p multiaddr without /p2p/<peerID>: clear warning + None.

    sp479 — the warning is reserved for genuine libp2p multiaddrs that DO need
    /p2p/ (a /dns4/.../ws or /ip4/... address). ws:// / wss:// inputs take the
    documented BootstrapClient WS fallback and are demoted to debug (see the
    next test), so this case uses a real multiaddr."""
    t = _transport()
    with caplog.at_level(logging.WARNING):
        result = await t.connect_to_peer(
            "/dns4/bootstrap1.prsm-network.com/tcp/8765/ws",
        )
    assert result is None
    # Warning text should mention the missing suffix
    assert any(
        "/p2p/<peerID>" in rec.getMessage()
        and "bootstrap multiaddr" in rec.getMessage()
        for rec in caplog.records
    ), [r.getMessage() for r in caplog.records]
    # Telemetry incremented
    assert t._telemetry["error_count"] == 1
    # PrsmConnect was NOT called (short-circuited)
    t._lib.PrsmConnect.assert_not_called()


@pytest.mark.asyncio
async def test_ws_scheme_missing_p2p_is_debug_not_warning(caplog):
    """sp479 — a ws:// / wss:// bootstrap URL without /p2p/ is the documented
    BootstrapClient WS-fallback path: return None WITHOUT a warning or an
    error-count bump (it is not a misconfiguration)."""
    t = _transport()
    with caplog.at_level(logging.WARNING):
        result = await t.connect_to_peer("wss://bootstrap1.prsm-network.com:8765")
    assert result is None
    assert not any(
        "missing /p2p" in rec.getMessage() for rec in caplog.records
    ), [r.getMessage() for r in caplog.records]
    assert t._telemetry["error_count"] == 0
    t._lib.PrsmConnect.assert_not_called()


@pytest.mark.asyncio
async def test_with_p2p_suffix_proceeds_to_c_bridge():
    """When peer ID is present, connect proceeds normally."""
    t = _transport()
    addr = (
        "/dns4/bootstrap1.prsm-network.com/tcp/8765/ws/"
        "p2p/12D3KooWFakePeerIDForTest"
    )
    await t.connect_to_peer(addr)
    # PrsmConnect WAS called (no short-circuit)
    t._lib.PrsmConnect.assert_called_once()


@pytest.mark.asyncio
async def test_handle_negative_short_circuits():
    """Pre-existing behavior: handle<0 returns None without C call."""
    t = _transport()
    t._handle = -1
    result = await t.connect_to_peer(
        "/ip4/127.0.0.1/tcp/9001/ws/p2p/12D3KooWX",
    )
    assert result is None
    t._lib.PrsmConnect.assert_not_called()
