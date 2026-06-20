"""Sprint 934 — authenticate the gossip `origin` field (P2P transport review).

`GossipProtocol._handle_gossip` delivered `origin = msg.payload.get("origin",
msg.sender_id)` to subscribers — i.e. it trusted an UNTRUSTED payload field for
the source identity that application handlers use for authorization (job
accept/confirm/cancel attribute work + payment by `origin`). The transport
authenticates the immediate `sender_id` (sp731 F64 binds it to the handshake
identity), but a connected peer could publish a message with its OWN authenticated
sender_id while setting `payload["origin"]` to a VICTIM's node_id — and every
subscriber would treat it as coming from the victim. Full node-ID impersonation
for the whole gossip marketplace.

Fix: the publisher signs an origin attestation over (subtype, data, origin,
origin_time, nonce) with its key and ships its pubkey in the payload; the receiver
trusts `origin` ONLY if (a) sha256(pubkey)==origin (node_id is the pubkey hash, so
the claim is self-verifying) AND (b) the signature verifies. Otherwise it falls
back to the authenticated `sender_id`. This survives multi-hop relay (the payload
+ nonce are preserved on forward) and is backward-compatible (old messages without
the attestation fall back to sender_id — safe, never the spoofed payload origin).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.gossip import GossipProtocol, build_gossip_origin_fields
from prsm.node.transport import MSG_GOSSIP, P2PMessage
from prsm.node.identity import generate_node_identity


def _gossip(receiver):
    transport = MagicMock()
    transport.identity = receiver
    transport.on_message = MagicMock()
    transport.gossip = AsyncMock(return_value=0)
    gp = GossipProtocol(transport)
    return gp


def _msg(sender_id, payload, nonce):
    return P2PMessage(
        msg_type=MSG_GOSSIP, sender_id=sender_id,
        payload={"subtype": "test_sub", **payload}, ttl=1, nonce=nonce,
    )


async def _deliver(gp, msg):
    seen = []

    async def cb(subtype, data, origin):
        seen.append(origin)

    gp._subscribers["test_sub"] = [cb]
    # Sp1182 — the fallback origin (when attestation is absent/invalid) is now
    # the handshake-authenticated peer.peer_id, not the raw msg.sender_id frame
    # field. An honest relayer's authenticated peer_id equals the sender_id it
    # stamps, so pass a peer bound to msg.sender_id (these tests don't spoof the
    # relayer identity itself; sp1182's own test covers the spoofed case).
    await gp._handle_gossip(msg, MagicMock(peer_id=msg.sender_id))
    return seen


@pytest.mark.asyncio
async def test_authenticated_origin_is_trusted():
    author = generate_node_identity("author")
    gp = _gossip(generate_node_identity("receiver"))
    data = {"x": 1}
    fields = build_gossip_origin_fields(author, "test_sub", data, origin_time=123.0, nonce="n1")
    msg = _msg(author.node_id, {"data": data, **fields}, "n1")
    seen = await _deliver(gp, msg)
    assert seen == [author.node_id]


@pytest.mark.asyncio
async def test_spoofed_origin_without_attestation_falls_back_to_sender():
    # THE ATTACK: attacker (authenticated sender) claims origin=victim with no
    # valid attestation → subscribers must see the ATTACKER, not the victim.
    attacker = generate_node_identity("attacker")
    victim = generate_node_identity("victim")
    gp = _gossip(generate_node_identity("receiver"))
    msg = _msg(attacker.node_id, {"data": {"x": 1}, "origin": victim.node_id, "origin_time": 123.0}, "n2")
    seen = await _deliver(gp, msg)
    assert seen == [attacker.node_id]
    assert victim.node_id not in seen


@pytest.mark.asyncio
async def test_forged_attestation_with_wrong_pubkey_rejected():
    # Attacker signs with its OWN key but claims origin=victim and ships its own
    # pubkey. sha256(attacker_pubkey) != victim node_id → node-id binding fails
    # → fall back to the authenticated sender.
    attacker = generate_node_identity("attacker")
    victim = generate_node_identity("victim")
    gp = _gossip(generate_node_identity("receiver"))
    data = {"x": 1}
    f = build_gossip_origin_fields(attacker, "test_sub", data, 123.0, "n3")
    f["origin"] = victim.node_id          # lie about origin, keep attacker's pubkey/sig
    msg = _msg(attacker.node_id, {"data": data, **f}, "n3")
    seen = await _deliver(gp, msg)
    assert seen == [attacker.node_id]


@pytest.mark.asyncio
async def test_tampered_data_breaks_attestation():
    # Attestation signed over data={"x":1}; attacker forwards it with data={"x":9}
    # → signature no longer verifies → fall back to sender.
    author = generate_node_identity("author")
    relayer = generate_node_identity("relayer")
    gp = _gossip(generate_node_identity("receiver"))
    f = build_gossip_origin_fields(author, "test_sub", {"x": 1}, 123.0, "n4")
    msg = _msg(relayer.node_id, {"data": {"x": 9}, **f}, "n4")   # data tampered
    seen = await _deliver(gp, msg)
    assert seen == [relayer.node_id]      # not author — attestation invalid


@pytest.mark.asyncio
async def test_relayed_message_preserves_authenticated_origin():
    # A relayed message (immediate sender = relayer, but valid author attestation)
    # must attribute to the AUTHOR — multi-hop relay must not break attribution.
    author = generate_node_identity("author")
    relayer = generate_node_identity("relayer")
    gp = _gossip(generate_node_identity("receiver"))
    data = {"x": 1}
    f = build_gossip_origin_fields(author, "test_sub", data, 123.0, "n5")
    msg = _msg(relayer.node_id, {"data": data, **f}, "n5")
    seen = await _deliver(gp, msg)
    assert seen == [author.node_id]


@pytest.mark.asyncio
async def test_legacy_message_without_attestation_uses_sender():
    # Backward-compat: an old node's message (no attestation, may carry a bare
    # 'origin') must fall back to the authenticated sender, never the payload.
    old = generate_node_identity("old")
    other = generate_node_identity("other")
    gp = _gossip(generate_node_identity("receiver"))
    msg = _msg(old.node_id, {"data": {"x": 1}, "origin": other.node_id}, "n6")
    seen = await _deliver(gp, msg)
    assert seen == [old.node_id]


@pytest.mark.asyncio
async def test_publish_emits_a_self_verifying_attestation():
    # End-to-end: a message produced by publish() must authenticate as its author
    # when received. Drives the real publish path + the receive path together.
    author = generate_node_identity("author")
    pub_transport = MagicMock()
    pub_transport.identity = author
    captured = {}

    async def _gossip_send(msg, fanout):
        captured["msg"] = msg
        return 1   # pretend it went to a peer (skip local-delivery shortcut)

    pub_transport.gossip = _gossip_send
    pub_transport.on_message = MagicMock()
    pub = GossipProtocol(pub_transport)
    await pub.publish("test_sub", {"hello": "world"})

    received = captured["msg"]
    gp = _gossip(generate_node_identity("receiver"))
    seen = await _deliver(gp, received)
    assert seen == [author.node_id]
