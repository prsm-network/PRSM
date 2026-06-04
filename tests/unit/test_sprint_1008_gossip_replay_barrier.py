"""Sprint 1008 — gossip-layer replay barrier (finding 4).

The P2P-substrate integrity hunt (workflow wbu7u2ftm, finding 4) confirmed that
GossipProtocol._handle_gossip had no replay barrier of its own — it relied on the
transport's nonce dedup, whose window is ~300s, while the gossip log retains
messages for 1h–24h. A captured, validly-signed gossip frame replayed AFTER the
300s transport window is therefore re-delivered to subscribers AND re-fanned into
the mesh (replay + amplification).

Legit gossip is unaffected: every publish() generates a FRESH nonce (and periodic
re-announces likewise), so only EXACT-nonce replays — which mesh redundancy
already tolerates being dropped — are caught.

Fix: a gossip-layer seen-nonce set with a TTL >= the log retention (default 24h),
bounded by an LRU cap, checked in _handle_gossip AFTER the digest special-casing
and BEFORE subscriber delivery / re-propagation.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.gossip import (
    GossipProtocol,
    _gossip_dedup_max,
    _gossip_dedup_window,
)
from prsm.node.transport import MSG_GOSSIP, P2PMessage, PeerConnection
from prsm.node.identity import generate_node_identity


def _gossip():
    t = MagicMock()
    t.identity = generate_node_identity("self")
    t.on_message = MagicMock()
    t.gossip = AsyncMock()
    return GossipProtocol(t)


def _peer(pid):
    return PeerConnection(peer_id=pid, address="1.1.1.1:9001", websocket=MagicMock())


def _msg(nonce, subtype="test_sub", ttl=3):
    return P2PMessage(
        msg_type=MSG_GOSSIP, sender_id="peerA",
        payload={"subtype": subtype, "data": {"x": 1}}, ttl=ttl, nonce=nonce,
    )


# ── the barrier primitive ───────────────────────────────────────────────────


def test_first_sight_not_replay_second_is():
    g = _gossip()
    assert g._is_replayed_gossip("N1", now=1000.0) is False
    assert g._is_replayed_gossip("N1", now=1100.0) is True  # within window


def test_distinct_nonces_not_replays():
    g = _gossip()
    assert g._is_replayed_gossip("A", now=1000.0) is False
    assert g._is_replayed_gossip("B", now=1000.0) is False


def test_nonce_forgotten_after_window(monkeypatch):
    monkeypatch.setenv("PRSM_GOSSIP_DEDUP_WINDOW_SEC", "100")
    g = _gossip()
    assert g._is_replayed_gossip("N1", now=1000.0) is False
    # 200s later — beyond the 100s window → forgotten, so re-acceptance is NOT a
    # replay drop (the message has aged out of the barrier).
    assert g._is_replayed_gossip("N1", now=1200.0) is False


def test_lru_cap_bounds_the_set(monkeypatch):
    monkeypatch.setenv("PRSM_GOSSIP_DEDUP_MAX", "10")
    monkeypatch.setenv("PRSM_GOSSIP_DEDUP_WINDOW_SEC", "100000")
    g = _gossip()
    for i in range(100):
        g._is_replayed_gossip(f"n{i}", now=1000.0 + i)
    assert len(g._seen_gossip_nonces) <= 10


def test_window_and_max_defaults_sane():
    assert _gossip_dedup_window() >= 3600.0   # >= the default retention
    assert _gossip_dedup_max() >= 1000


# ── _handle_gossip integration ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handle_gossip_delivers_once_drops_replay():
    g = _gossip()
    calls = []

    async def cb(subtype, data, origin):
        calls.append(data)

    g.subscribe("test_sub", cb)
    msg = _msg("N1")
    await g._handle_gossip(msg, _peer("peerA"))
    await g._handle_gossip(msg, _peer("peerA"))  # exact-nonce replay

    assert len(calls) == 1                       # delivered exactly once
    assert g.transport.gossip.await_count == 1   # re-fanned exactly once


@pytest.mark.asyncio
async def test_handle_gossip_fresh_nonce_delivered():
    g = _gossip()
    calls = []

    async def cb(subtype, data, origin):
        calls.append(data)

    g.subscribe("test_sub", cb)
    await g._handle_gossip(_msg("N1"), _peer("peerA"))
    await g._handle_gossip(_msg("N2"), _peer("peerA"))  # distinct nonce

    assert len(calls) == 2  # both delivered


@pytest.mark.asyncio
async def test_digest_messages_not_blocked_by_barrier():
    """Digest req/resp are special-cased BEFORE the barrier, so they are never
    dropped by it (the catch-up/sync mechanism must keep working)."""
    g = _gossip()
    seen = []
    g._handle_digest_request = AsyncMock(side_effect=lambda m, p: seen.append("req"))
    msg = P2PMessage(
        msg_type=MSG_GOSSIP, sender_id="peerA",
        payload={"subtype": "digest_request"}, ttl=3, nonce="DG1",
    )
    await g._handle_gossip(msg, _peer("peerA"))
    await g._handle_gossip(msg, _peer("peerA"))  # same nonce again
    assert seen == ["req", "req"]  # digest handled both times, not deduped
