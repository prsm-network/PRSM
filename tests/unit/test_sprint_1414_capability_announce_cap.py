"""Sprint 1414 — the known_peers cap must also bound _handle_capability_announce.

The routing-table cap (``PRSM_MAX_KNOWN_PEERS``, default 2048) is the memory + eclipse-by-magnitude
defense: it stops one authenticated peer from minting unlimited keypairs (node_id =
sha256(pubkey)[:32], each passing the self-signature gate) and flooding ``known_peers`` without bound.

sp1005 enforced it on ``_handle_announce`` (discovery.py:1155) and ``_handle_peer_response`` (:1267),
and the libp2p sibling caps the same capability path. But ``_handle_capability_announce``'s
NEW-peer branch (discovery.py:1329) constructs a ``PeerInfo`` with NO cap check — so the cap was a
live, default-on defense with a hole straight through it. An attacker floods
``DISCOVERY_CAPABILITY_ANNOUNCE`` (each a self-signed announce for a freshly-minted key) and grows the
table unbounded: memory-exhaustion DoS, and the fabricated entries dominate
``find_peers_with_capability`` / GPU selection (eclipse-by-magnitude).

Authentication is not the barrier: ``_authenticated_announce_node_id`` accepts any announce whose
``sha256(origin_pubkey)[:32] == sender_id``, regardless of which connection relayed it. Minting keys
is free.

These tests flood the capability-announce path from distinct minted identities and assert the cap
holds — the sp1005 test's exact sibling, one handler over.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.discovery import (
    PeerDiscovery,
    _announce_signing_bytes,
    _max_known_peers,
)
from prsm.node.identity import generate_node_identity
from prsm.node.transport import MSG_GOSSIP, P2PMessage, PeerConnection


DISCOVERY_CAPABILITY_ANNOUNCE = "capability_announce"


def _discovery(name="self"):
    t = MagicMock()
    t.identity = generate_node_identity(name)
    t.on_message = MagicMock()
    t.peer_count = 0
    t.peers = {}
    t.send_to_peer = AsyncMock()
    return PeerDiscovery(t)


def _peer(peer_id):
    return PeerConnection(peer_id=peer_id, address="1.1.1.1:9001", websocket=MagicMock())


def _signed_capability_announce(identity, *, address, announce_time=100.0, relayer_id=None):
    """A self-signed capability announce, as _authenticated_announce_node_id accepts it.

    relayer_id defaults to the announcer itself (a direct announce); set it to a DIFFERENT id to
    model the flood — one connected peer relaying announces it minted for many identities. The
    self-signature over (node_id, payload, nonce) still authenticates each minted node_id."""
    payload = {
        "subtype": DISCOVERY_CAPABILITY_ANNOUNCE,
        "address": address,
        "announce_time": announce_time,
        "capabilities": ["inference"],
        "supported_backends": ["anthropic"],
        "gpu_available": True,
    }
    nonce = "cn-" + identity.node_id[:6]
    payload["origin_pubkey"] = identity.public_key_b64
    payload["origin_sig"] = identity.sign(
        _announce_signing_bytes(identity.node_id, payload, nonce))
    return P2PMessage(
        msg_type=MSG_GOSSIP,
        sender_id=identity.node_id,           # attested: sha256(pubkey)[:32] == sender_id
        payload=payload,
        ttl=1,
        nonce=nonce,
    )


@pytest.mark.asyncio
async def test_capability_announce_respects_the_known_peers_cap(monkeypatch):
    """The bug: 50 authenticated announces for distinct minted keys inflate the table past the cap."""
    monkeypatch.setenv("PRSM_MAX_KNOWN_PEERS", "2")
    assert _max_known_peers() == 2
    c = _discovery("c")

    for i in range(50):
        ident = generate_node_identity(f"flood{i}")
        msg = _signed_capability_announce(ident, address=f"5.5.5.{i % 250}:9001")
        await c._handle_capability_announce(msg, _peer(ident.node_id))

    assert len(c.known_peers) <= 2, (
        f"known_peers grew to {len(c.known_peers)} — the capability-announce path bypasses the cap "
        f"(memory DoS + eclipse-by-magnitude)"
    )


@pytest.mark.asyncio
async def test_one_relayer_cannot_flood_the_table_with_minted_peers(monkeypatch):
    """The realistic shape: a SINGLE connected peer relays announces it minted for many keys.
    Authentication passes (each announce is self-signed) — only the cap stops it."""
    monkeypatch.setenv("PRSM_MAX_KNOWN_PEERS", "3")
    c = _discovery("c")
    relayer = generate_node_identity("relayer")
    conn = _peer(relayer.node_id)

    for i in range(30):
        ident = generate_node_identity(f"minted{i}")
        msg = _signed_capability_announce(ident, address=f"6.6.6.{i % 250}:9001")
        await c._handle_capability_announce(msg, conn)   # all over ONE connection

    assert len(c.known_peers) <= 3


@pytest.mark.asyncio
async def test_already_tracked_peer_still_refreshes_at_cap(monkeypatch):
    """Regression: the cap gates only NEW node_ids — a peer already in the table must keep updating
    its capabilities even when the table is full (mirrors the sp1005 sibling contract). The existing-
    peer branch (discovery.py:1316) refreshes capabilities/backends/gpu; address is set only on
    creation, so this asserts the fields the update path actually touches."""
    monkeypatch.setenv("PRSM_MAX_KNOWN_PEERS", "1")
    c = _discovery("c")
    ident = generate_node_identity("resident")

    first = _signed_capability_announce(ident, address="7.7.7.7:9001", announce_time=100.0)
    first.payload["capabilities"] = ["inference"]
    first.payload["gpu_available"] = False
    first.payload["origin_sig"] = ident.sign(
        _announce_signing_bytes(ident.node_id, first.payload, first.nonce))
    await c._handle_capability_announce(first, _peer(ident.node_id))
    assert c.known_peers[ident.node_id].gpu_available is False

    # a fresh announce (newer time) from the SAME peer must still update, table full or not
    second = _signed_capability_announce(ident, address="7.7.7.7:9001", announce_time=200.0)
    second.payload["capabilities"] = ["inference", "embedding"]
    second.payload["gpu_available"] = True
    second.payload["origin_sig"] = ident.sign(
        _announce_signing_bytes(ident.node_id, second.payload, second.nonce))
    await c._handle_capability_announce(second, _peer(ident.node_id))

    assert c.known_peers[ident.node_id].gpu_available is True
    assert "embedding" in c.known_peers[ident.node_id].capabilities
    assert len(c.known_peers) == 1


@pytest.mark.asyncio
async def test_unauthenticated_capability_announce_is_still_dropped(monkeypatch):
    """The cap must not become the ONLY line of defense — an unattested announce is dropped before
    the cap is even consulted (sp937)."""
    monkeypatch.setenv("PRSM_MAX_KNOWN_PEERS", "2048")
    c = _discovery("c")
    ident = generate_node_identity("liar")
    msg = _signed_capability_announce(ident, address="9.9.9.9:9001")
    del msg.payload["origin_sig"]                       # strip the attestation
    # relayed by a DIFFERENT connection, so the direct-peer fallback can't authenticate it either
    await c._handle_capability_announce(msg, _peer(generate_node_identity("other").node_id))

    assert ident.node_id not in c.known_peers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
