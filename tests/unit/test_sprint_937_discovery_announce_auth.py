"""Sprint 937 — authenticate the discovery announce identity (P2P transport review).

Discovery announces travel as MSG_GOSSIP, which sp731 EXCLUDES from sender_id
re-binding, and there is no per-message signature check — so `msg.sender_id` on a
gossip announce is attacker-controlled. `_handle_announce` keyed `known_peers` on
that raw sender_id, so a connected peer could send an announce with
`sender_id=victim` and a malicious address → poison `known_peers[victim]` and
redirect anyone who later dials the victim to the attacker (eclipse / routing
redirect). `_handle_capability_announce` had the same flaw.

Fix (mirrors sp934): announces carry a self-verifying attestation (pubkey whose
sha256 is the node_id + a signature over the announce content). The handler
trusts the claimed node_id only if the attestation verifies; otherwise it accepts
it ONLY when it equals the handshake-authenticated `peer.peer_id` (a direct
announce); otherwise the announce is DROPPED. Raw, unauthenticated sender_id is
never trusted.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.node.discovery import (
    PeerDiscovery,
    _attest_announce_payload,
    _authenticated_announce_node_id,
)
from prsm.node.transport import MSG_GOSSIP, P2PMessage, PeerConnection
from prsm.node.identity import generate_node_identity


def _msg(sender_id, payload, nonce="n1"):
    return P2PMessage(
        msg_type=MSG_GOSSIP, sender_id=sender_id,
        payload={"subtype": "discovery_announce", **payload}, ttl=1, nonce=nonce,
    )


def _peer(peer_id):
    return PeerConnection(peer_id=peer_id, address="1.1.1.1:9001", websocket=MagicMock())


# ── the authentication decision ───────────────────────────────────────────


def test_direct_unattested_announce_uses_peer_id():
    announcer = generate_node_identity("announcer")
    msg = _msg(announcer.node_id, {"address": "5.5.5.5:9001"})
    # arrives directly from the announcer (handshake-authenticated)
    assert _authenticated_announce_node_id(msg, _peer(announcer.node_id)) == announcer.node_id


def test_forged_sender_id_is_dropped():
    victim = generate_node_identity("victim")
    attacker = generate_node_identity("attacker")
    # attacker forges sender_id=victim on its own connection, no attestation
    msg = _msg(victim.node_id, {"address": "6.6.6.6:9001"})
    assert _authenticated_announce_node_id(msg, _peer(attacker.node_id)) is None


def test_attested_announce_authenticates_across_relay():
    author = generate_node_identity("author")
    relayer = generate_node_identity("relayer")
    payload = {"subtype": "discovery_announce", "address": "5.5.5.5:9001"}
    _attest_announce_payload(author, payload, "n7")
    msg = P2PMessage(msg_type=MSG_GOSSIP, sender_id=author.node_id, payload=payload, ttl=1, nonce="n7")
    # relayed (peer is the relayer), but the attestation authenticates the author
    assert _authenticated_announce_node_id(msg, _peer(relayer.node_id)) == author.node_id


def test_attested_with_wrong_pubkey_is_dropped():
    victim = generate_node_identity("victim")
    attacker = generate_node_identity("attacker")
    payload = {"subtype": "discovery_announce", "address": "6.6.6.6:9001"}
    _attest_announce_payload(attacker, payload, "n8")   # signed by attacker
    msg = P2PMessage(msg_type=MSG_GOSSIP, sender_id=victim.node_id, payload=payload, ttl=1, nonce="n8")
    # claims to be victim but the pubkey hashes to attacker → not authenticated,
    # and sender_id != peer.peer_id → dropped
    assert _authenticated_announce_node_id(msg, _peer(attacker.node_id)) is None


def test_tampered_attested_content_is_dropped():
    author = generate_node_identity("author")
    relayer = generate_node_identity("relayer")
    payload = {"subtype": "discovery_announce", "address": "5.5.5.5:9001"}
    _attest_announce_payload(author, payload, "n9")
    payload["address"] = "9.9.9.9:9001"   # relayer tampers the address
    msg = P2PMessage(msg_type=MSG_GOSSIP, sender_id=author.node_id, payload=payload, ttl=1, nonce="n9")
    assert _authenticated_announce_node_id(msg, _peer(relayer.node_id)) is None


# ── handler integration ────────────────────────────────────────────────────


def _discovery():
    t = MagicMock()
    t.identity = generate_node_identity("self")
    t.on_message = MagicMock()
    t.peer_count = 0
    return PeerDiscovery(t)


@pytest.mark.asyncio
async def test_handle_announce_drops_forged_peer():
    victim = generate_node_identity("victim")
    attacker = generate_node_identity("attacker")
    d = _discovery()
    msg = _msg(victim.node_id, {"address": "6.6.6.6:9001"})
    await d._handle_announce(msg, _peer(attacker.node_id))
    assert victim.node_id not in d.known_peers   # forge did not poison the table


@pytest.mark.asyncio
async def test_handle_announce_records_direct_peer():
    announcer = generate_node_identity("announcer")
    d = _discovery()
    msg = _msg(announcer.node_id, {"address": "5.5.5.5:9001"})
    await d._handle_announce(msg, _peer(announcer.node_id))
    assert announcer.node_id in d.known_peers
    assert d.known_peers[announcer.node_id].address == "5.5.5.5:9001"


@pytest.mark.asyncio
async def test_handle_capability_announce_drops_forged_peer():
    victim = generate_node_identity("victim")
    attacker = generate_node_identity("attacker")
    d = _discovery()
    msg = _msg(victim.node_id, {"capabilities": ["inference"], "gpu_available": True})
    await d._handle_capability_announce(msg, _peer(attacker.node_id))
    assert victim.node_id not in d.known_peers
