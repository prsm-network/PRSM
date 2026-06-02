"""Sprint 936 — per-peer message rate limiting (P2P transport review, DoS).

The transport read-loop (WebSocketTransport._read_loop) parsed, deduped, and
dispatched every inbound message from a peer with NO per-peer rate limit. A
single connected peer could flood messages: each unique nonce grows _seen_nonces
(bounded only by the ~300s window, so rate*300s entries), each is parsed +
dispatched (CPU), and gossip re-propagates it (amplification) — a single-peer
liveness/memory DoS.

Fix: a per-peer token bucket. A peer gets `burst` tokens refilling at `rate`/sec;
messages over budget are dropped before dedup/dispatch (so a flood can't grow the
nonce set, burn dispatch CPU, or be amplified). Generous defaults so legitimate
gossip (a few msgs/sec/peer) is never affected.
"""
from __future__ import annotations

import pytest

from prsm.node.transport import WebSocketTransport, P2PMessage
from prsm.node.identity import generate_node_identity


def _transport(rate=10.0, burst=5.0):
    return WebSocketTransport(
        generate_node_identity("n"), host="127.0.0.1", port=0,
        peer_msg_rate=rate, peer_msg_burst=burst,
    )


# ── token bucket ──────────────────────────────────────────────────────────


def test_allows_up_to_burst_then_drops():
    t = _transport(rate=10.0, burst=5.0)
    now = 1000.0
    assert all(t._allow_peer_message("p1", now=now) for _ in range(5))
    assert t._allow_peer_message("p1", now=now) is False   # burst exhausted


def test_refills_over_time():
    t = _transport(rate=10.0, burst=5.0)
    now = 1000.0
    for _ in range(5):
        t._allow_peer_message("p1", now=now)
    assert t._allow_peer_message("p1", now=now) is False
    # 0.2s later at 10 tokens/s → exactly 2 more
    assert t._allow_peer_message("p1", now=now + 0.2) is True
    assert t._allow_peer_message("p1", now=now + 0.2) is True
    assert t._allow_peer_message("p1", now=now + 0.2) is False


def test_buckets_are_per_peer():
    t = _transport(rate=10.0, burst=2.0)
    now = 1000.0
    assert t._allow_peer_message("p1", now=now)
    assert t._allow_peer_message("p1", now=now)
    assert t._allow_peer_message("p1", now=now) is False   # p1 drained
    assert t._allow_peer_message("p2", now=now) is True    # p2 independent


def test_refill_caps_at_burst():
    t = _transport(rate=10.0, burst=5.0)
    # long idle → tokens cap at burst, not unbounded
    assert all(t._allow_peer_message("p1", now=1000.0 + i * 0.0) for i in range(5))
    # huge gap then drain: only `burst` allowed at once
    big = 1_000_000.0
    assert all(t._allow_peer_message("p1", now=big) for _ in range(5))
    assert t._allow_peer_message("p1", now=big) is False


# ── read-loop integration: a flood is dropped past the burst ──────────────


class _FakeWS:
    def __init__(self, raws):
        self._raws = raws

    async def __aiter__(self):
        for r in self._raws:
            yield r

    async def send(self, data):
        pass


@pytest.mark.asyncio
async def test_read_loop_drops_flood_past_burst():
    from prsm.node.transport import PeerConnection

    t = _transport(rate=1.0, burst=5.0)   # tiny refill so a fast flood is bounded
    dispatched = []

    async def handler(msg, peer):
        dispatched.append(msg.nonce)

    t.on_message("test_flood", handler)

    raws = []
    for i in range(50):
        m = P2PMessage(msg_type="test_flood", sender_id="attacker", payload={}, nonce=f"n{i}")
        raws.append(m.to_json())

    peer = PeerConnection(peer_id="attacker", address="x", websocket=_FakeWS(raws))
    await t._read_loop(peer)

    # The flood of 50 distinct messages is bounded to ~burst (refill over the
    # sub-millisecond loop is negligible at 1 token/sec).
    assert len(dispatched) <= 6, f"flood not bounded: {len(dispatched)} dispatched"
    assert len(dispatched) >= 1
