"""Sprint 941 — discovery announce replay freshness (P2P transport review).

sp937 closed announce FORGERY (an attacker can't claim another node's id). The
residual: a captured GENUINE attested announce could be replayed after the
~300s transport nonce-dedup window to re-assert a stale address/capabilities —
e.g. revert a node whose address changed back to its old one. sp941 binds a
signed `announce_time` into the announce attestation and rejects any announce
whose timestamp is not strictly newer than the last accepted one for that peer,
so a replayed (older-or-equal) announce can't overwrite fresher state.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.discovery import PeerDiscovery, _attest_announce_payload
from prsm.node.transport import MSG_GOSSIP, P2PMessage, PeerConnection
from prsm.node.identity import generate_node_identity


def _discovery():
    t = MagicMock()
    t.identity = generate_node_identity("self")
    t.on_message = MagicMock()
    t.peer_count = 0
    t.gossip = AsyncMock(return_value=0)
    return PeerDiscovery(t)


def _peer(peer_id="relayer"):
    return PeerConnection(peer_id=peer_id, address="1.1.1.1:9001", websocket=MagicMock())


def _attested_announce(author, *, announce_time, address, nonce, subtype="discovery_announce"):
    payload = {"subtype": subtype, "address": address, "announce_time": announce_time}
    _attest_announce_payload(author, payload, nonce)
    return P2PMessage(msg_type=MSG_GOSSIP, sender_id=author.node_id, payload=payload, ttl=1, nonce=nonce)


@pytest.mark.asyncio
async def test_stale_announce_replay_cannot_revert_address():
    author = generate_node_identity("author")
    d = _discovery()
    peer = _peer()   # attested → authenticated regardless of relayer

    await d._handle_announce(_attested_announce(author, announce_time=200.0, address="B:9001", nonce="n1"), peer)
    assert d.known_peers[author.node_id].address == "B:9001"
    assert d.known_peers[author.node_id].last_announce_time == 200.0

    # Replay a genuine-but-OLDER announce (address A @ t=100) → rejected.
    await d._handle_announce(_attested_announce(author, announce_time=100.0, address="A:9001", nonce="n2"), peer)
    assert d.known_peers[author.node_id].address == "B:9001"   # NOT reverted

    # A genuinely newer announce (t=300) is accepted.
    await d._handle_announce(_attested_announce(author, announce_time=300.0, address="C:9001", nonce="n3"), peer)
    assert d.known_peers[author.node_id].address == "C:9001"
    assert d.known_peers[author.node_id].last_announce_time == 300.0


@pytest.mark.asyncio
async def test_equal_timestamp_replay_rejected():
    author = generate_node_identity("author")
    d = _discovery()
    peer = _peer()
    await d._handle_announce(_attested_announce(author, announce_time=500.0, address="X:9001", nonce="n1"), peer)
    # exact-same-timestamp replay (<=) → rejected, no state churn
    await d._handle_announce(_attested_announce(author, announce_time=500.0, address="Y:9001", nonce="n2"), peer)
    assert d.known_peers[author.node_id].address == "X:9001"


@pytest.mark.asyncio
async def test_capability_announce_replay_rejected():
    author = generate_node_identity("author")
    d = _discovery()
    peer = _peer()
    fresh = _attested_announce(author, announce_time=200.0, address="X:9001", nonce="c1", subtype="capability_announce")
    fresh.payload["capabilities"] = ["inference", "gpu"]
    _attest_announce_payload(author, fresh.payload, "c1")  # re-attest with caps included
    await d._handle_capability_announce(fresh, peer)
    assert d.known_peers[author.node_id].capabilities == ["inference", "gpu"]

    stale = _attested_announce(author, announce_time=100.0, address="X:9001", nonce="c2", subtype="capability_announce")
    stale.payload["capabilities"] = ["downgraded"]
    _attest_announce_payload(author, stale.payload, "c2")
    await d._handle_capability_announce(stale, peer)
    assert d.known_peers[author.node_id].capabilities == ["inference", "gpu"]   # replay rejected


@pytest.mark.asyncio
async def test_legacy_announce_without_timestamp_still_accepted():
    # Backward-compat: a direct announce with no announce_time is not treated as
    # a replay (sp937 already drops its forged/relayed variants).
    author = generate_node_identity("author")
    d = _discovery()
    payload = {"subtype": "discovery_announce", "address": "L:9001"}  # no announce_time, no attestation
    msg = P2PMessage(msg_type=MSG_GOSSIP, sender_id=author.node_id, payload=payload, ttl=1, nonce="n1")
    await d._handle_announce(msg, _peer(peer_id=author.node_id))   # direct → accepted by sp937
    assert d.known_peers[author.node_id].address == "L:9001"
    assert d.known_peers[author.node_id].last_announce_time == 0.0
