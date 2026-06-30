"""Sprint 1331 — backfill unit tests for the live-found transport fixes (sp1326).

The 2026-06-30 cross-cloud testnet GO surfaced these under load; they were committed with
live validation (real on-chain settlement) but no unit coverage. This backfills it:
  - _ws_max_size_bytes(): env-tunable frame cap (the 16MB→256MB fix; MESSAGE_TOO_BIG was THE
    cross-host blocker for Qwen's 152k-vocab logit payload).
  - NodeConfig ws ping interval/timeout env override (so a cross-cloud link survives an
    event-loop block during a cold slice-load).
  - _handle_incoming reconnect-REPLACE (last-writer-wins) instead of rejecting with
    "Already connected" — the asymmetric-firewall flap fix — and the identity-checked cleanup
    so a stale read-loop can't wipe the fresh replacement.
"""
from __future__ import annotations

import asyncio

import pytest
import websockets.exceptions

from prsm.node.config import NodeConfig
from prsm.node.identity import generate_node_identity
from prsm.node.transport import (
    MSG_HANDSHAKE,
    P2PMessage,
    WebSocketTransport,
    _ws_max_size_bytes,
)


# ── _ws_max_size_bytes (the MESSAGE_TOO_BIG fix) ──────────────────────────────

def test_ws_max_size_default_256mb(monkeypatch):
    monkeypatch.delenv("PRSM_WS_MAX_SIZE_MB", raising=False)
    assert _ws_max_size_bytes() == 256 * 1024 * 1024


def test_ws_max_size_env_override(monkeypatch):
    monkeypatch.setenv("PRSM_WS_MAX_SIZE_MB", "512")
    assert _ws_max_size_bytes() == 512 * 1024 * 1024


def test_ws_max_size_malformed_falls_back(monkeypatch):
    monkeypatch.setenv("PRSM_WS_MAX_SIZE_MB", "not-a-number")
    assert _ws_max_size_bytes() == 256 * 1024 * 1024


def test_ws_max_size_exceeds_old_16mb_cap(monkeypatch):
    """The regression guard: the default must comfortably exceed the old 16MB ceiling that
    a Qwen-7B (152k vocab) final-stage logit payload (~18MB) overflowed."""
    monkeypatch.delenv("PRSM_WS_MAX_SIZE_MB", raising=False)
    assert _ws_max_size_bytes() > 18 * 1024 * 1024


# ── ws ping env override (the cold-load-survival fix) ─────────────────────────

def test_ping_env_override(monkeypatch):
    monkeypatch.setenv("PRSM_WS_PING_INTERVAL_S", "600")
    monkeypatch.setenv("PRSM_WS_PING_TIMEOUT_S", "600")
    c = NodeConfig()
    assert c.ws_ping_interval == 600.0
    assert c.ws_ping_timeout == 600.0


def test_ping_defaults_unchanged(monkeypatch):
    monkeypatch.delenv("PRSM_WS_PING_INTERVAL_S", raising=False)
    monkeypatch.delenv("PRSM_WS_PING_TIMEOUT_S", raising=False)
    c = NodeConfig()
    assert c.ws_ping_interval == 20.0 and c.ws_ping_timeout == 10.0


def test_ping_malformed_keeps_default(monkeypatch):
    monkeypatch.setenv("PRSM_WS_PING_TIMEOUT_S", "abc")
    assert NodeConfig().ws_ping_timeout == 10.0


# ── reconnect-replace + identity-checked cleanup (the flap fix) ───────────────

class _FakeWS:
    """A fake inbound websocket: yields a signed handshake once, then blocks until close()d
    (then raises ConnectionClosed so the read loop exits into the cleanup)."""

    def __init__(self, handshake_json: str):
        self._handshake = handshake_json
        self._n = 0
        self.closed = False
        self.close_code = None
        self._gate = asyncio.Event()

    @property
    def remote_address(self):
        return ("10.0.0.9", 55555)

    async def recv(self):  # the handshake (read once by _handle_incoming)
        self._n += 1
        if self._n == 1:
            return self._handshake
        await self._gate.wait()
        raise websockets.exceptions.ConnectionClosed(None, None)

    def __aiter__(self):  # the read loop iterates the socket
        return self

    async def __anext__(self):
        await self._gate.wait()  # block until close() — then end the loop
        raise StopAsyncIteration

    async def send(self, data):  # the handshake ack
        return None

    async def close(self, code=1000, reason=""):
        self.closed = True
        self.close_code = code
        self._gate.set()


def _signed_handshake(peer_identity, listen_port=9001) -> str:
    msg = P2PMessage(
        msg_type=MSG_HANDSHAKE, sender_id=peer_identity.node_id,
        payload={"public_key": peer_identity.public_key_b64, "listen_port": listen_port})
    msg.sign(peer_identity)
    return msg.to_json()


async def _await_until(pred, tries=200, delay=0.01):
    for _ in range(tries):
        if pred():
            return True
        await asyncio.sleep(delay)
    return False


def test_reconnect_replaces_stale_peer_and_cleanup_is_identity_checked():
    async def _run():
        srv = WebSocketTransport(identity=generate_node_identity(display_name="server"))
        peer = generate_node_identity(display_name="worker")
        ws1, ws2 = _FakeWS(_signed_handshake(peer)), _FakeWS(_signed_handshake(peer))

        t1 = asyncio.create_task(srv._handle_incoming(ws1))
        assert await _await_until(
            lambda: srv.peers.get(peer.node_id) is not None
            and srv.peers[peer.node_id].websocket is ws1), "ws1 should register"

        # second inbound for the SAME peer_id → REPLACE (old "Already connected" path rejected it)
        t2 = asyncio.create_task(srv._handle_incoming(ws2))
        assert await _await_until(
            lambda: srv.peers.get(peer.node_id) is not None
            and srv.peers[peer.node_id].websocket is ws2), "ws2 should replace ws1"
        assert ws1.closed and ws1.close_code == 1012, "stale ws1 closed with 1012 (replaced)"

        # ws1's read loop now exits (it was closed) → its finally runs, but the identity check
        # must NOT wipe the live ws2 replacement installed under the same peer_id.
        await asyncio.sleep(0.1)
        assert srv.peers.get(peer.node_id) is not None, "ws1's stale cleanup wiped the live peer!"
        assert srv.peers[peer.node_id].websocket is ws2

        # teardown
        await ws2.close()
        for t in (t1, t2):
            try:
                await asyncio.wait_for(t, timeout=2)
            except (asyncio.TimeoutError, Exception):
                t.cancel()

    asyncio.run(_run())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
