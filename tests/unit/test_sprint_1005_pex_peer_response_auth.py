"""Sprint 1005 — authenticate the PEX (peer-exchange) response, the sp937 sibling gap.

The P2P-substrate integrity hunt (workflow wbu7u2ftm) confirmed that
`PeerDiscovery._handle_peer_response` — the handler for DISCOVERY_PEER_RESPONSE,
the peer-exchange path by which a node learns peers from another node — had NO
origin authentication, even though sp937/sp941 had added it to `_handle_announce`
and `_handle_capability_announce`. It read `node_id`, `address`, and
`hardware_profile` straight from an unauthenticated gossip payload and wrote them
into `known_peers`. On the live WebSocket transport (the libp2p `.so` is absent on
the Linux fleet, so deploys pin PRSM_TRANSPORT_BACKEND=websocket) a connected peer
could therefore:
  - inject an attacker-chosen (node_id → attacker address) mapping, redirecting the
    victim's outbound dials to the attacker (ECLIPSE), and/or
  - inject a forged fat hardware_profile that flows into the Parallax allocation
    pool via dht_backed_pool_provider, and/or
  - grow `known_peers` without bound (memory DoS).

Fix: a PEX response entry must carry a portable, self-verifying credential
({node_id, signed_payload, nonce, origin_pubkey, origin_sig}) — exactly the
attestation an authenticated announce already produces — and the receiver verifies
it (node_id == sha256(pubkey)[:32] AND the signature over the announce content)
and DROPS any entry that fails, ingesting address/profile from the VERIFIED signed
payload (never the tamperable top-level fields). The producer re-emits the stored
credential per known peer; peers without one are not relayed. A `known_peers` size
cap (PRSM_MAX_KNOWN_PEERS) bounds memory + the eclipse magnitude.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.discovery import (
    PeerDiscovery,
    _announce_signing_bytes,
    _attest_announce_payload,
    _max_known_peers,
    _verify_peer_credential,
)
from prsm.node.transport import MSG_GOSSIP, P2PMessage, PeerConnection
from prsm.node.identity import generate_node_identity


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


def _credential(identity, address="5.5.5.5:9001", announce_time=100.0, hardware_profile=None):
    """Build the portable announce credential a node signs for itself."""
    signed_payload = {
        "subtype": "discovery_announce",
        "address": address,
        "announce_time": announce_time,
    }
    if hardware_profile is not None:
        signed_payload["hardware_profile"] = hardware_profile
    nonce = "nx-" + identity.node_id[:6]
    sig = identity.sign(_announce_signing_bytes(identity.node_id, signed_payload, nonce))
    return {
        "node_id": identity.node_id,
        "signed_payload": signed_payload,
        "nonce": nonce,
        "origin_pubkey": identity.public_key_b64,
        "origin_sig": sig,
    }


def _pex_response(relayer_id, entries):
    return P2PMessage(
        msg_type=MSG_GOSSIP,
        sender_id=relayer_id,
        payload={"subtype": "discovery_peer_response", "peers": entries},
        ttl=1,
        nonce="pexn",
    )


# ── the credential-verification primitive ──────────────────────────────────


def test_verify_accepts_valid_credential():
    b = generate_node_identity("b")
    assert _verify_peer_credential(_credential(b)) == b.node_id


def test_verify_rejects_tampered_signed_payload():
    b = generate_node_identity("b")
    cred = _credential(b, address="5.5.5.5:9001")
    cred["signed_payload"]["address"] = "9.9.9.9:9001"  # relayer tampers
    assert _verify_peer_credential(cred) is None


def test_verify_rejects_impersonation_pubkey_mismatch():
    victim = generate_node_identity("victim")
    attacker = generate_node_identity("attacker")
    cred = _credential(attacker)
    cred["node_id"] = victim.node_id  # claim victim, signed by attacker
    assert _verify_peer_credential(cred) is None


def test_verify_rejects_missing_fields():
    assert _verify_peer_credential(None) is None
    assert _verify_peer_credential({}) is None
    assert _verify_peer_credential({"node_id": "x"}) is None


# ── _handle_peer_response (consumer) ────────────────────────────────────────


@pytest.mark.asyncio
async def test_pex_ingests_valid_credential():
    c = _discovery("c")
    b = generate_node_identity("b")
    relayer = generate_node_identity("relayer")
    await c._handle_peer_response(
        _pex_response(relayer.node_id, [{"credential": _credential(b, address="5.5.5.5:9001")}]),
        _peer(relayer.node_id),
    )
    assert b.node_id in c.known_peers
    assert c.known_peers[b.node_id].address == "5.5.5.5:9001"


@pytest.mark.asyncio
async def test_pex_drops_unauthenticated_flat_entry():
    """The pre-fix wire shape: a flat {node_id, address} with no credential.
    An attacker forging it must NOT poison the table."""
    c = _discovery("c")
    attacker = generate_node_identity("attacker")
    forged = {"node_id": "deadbeef" * 4, "address": "6.6.6.6:9001"}
    await c._handle_peer_response(
        _pex_response(attacker.node_id, [forged]), _peer(attacker.node_id),
    )
    assert "deadbeef" * 4 not in c.known_peers
    assert len(c.known_peers) == 0


@pytest.mark.asyncio
async def test_pex_drops_tampered_credential():
    c = _discovery("c")
    b = generate_node_identity("b")
    attacker = generate_node_identity("attacker")
    cred = _credential(b, address="5.5.5.5:9001")
    cred["signed_payload"]["address"] = "6.6.6.6:9001"  # attacker re-points the address
    await c._handle_peer_response(
        _pex_response(attacker.node_id, [{"credential": cred}]), _peer(attacker.node_id),
    )
    assert b.node_id not in c.known_peers


@pytest.mark.asyncio
async def test_pex_reads_address_from_signed_payload_not_top_level():
    """Even with a valid credential, a tampered top-level address field must be
    ignored — the address comes from the SIGNED payload."""
    c = _discovery("c")
    b = generate_node_identity("b")
    relayer = generate_node_identity("relayer")
    entry = {"credential": _credential(b, address="5.5.5.5:9001"), "address": "9.9.9.9:9001"}
    await c._handle_peer_response(_pex_response(relayer.node_id, [entry]), _peer(relayer.node_id))
    assert c.known_peers[b.node_id].address == "5.5.5.5:9001"


@pytest.mark.asyncio
async def test_pex_drops_stale_replay_credential():
    c = _discovery("c")
    b = generate_node_identity("b")
    relayer = generate_node_identity("relayer")
    # First, a fresh credential at announce_time=200.
    await c._handle_peer_response(
        _pex_response(relayer.node_id, [{"credential": _credential(b, address="5.5.5.5:9001", announce_time=200.0)}]),
        _peer(relayer.node_id),
    )
    # Then a replayed older one (announce_time=100) trying to re-point the address.
    await c._handle_peer_response(
        _pex_response(relayer.node_id, [{"credential": _credential(b, address="7.7.7.7:9001", announce_time=100.0)}]),
        _peer(relayer.node_id),
    )
    assert c.known_peers[b.node_id].address == "5.5.5.5:9001"  # stale replay ignored


@pytest.mark.asyncio
async def test_pex_respects_known_peers_cap(monkeypatch):
    monkeypatch.setenv("PRSM_MAX_KNOWN_PEERS", "2")
    assert _max_known_peers() == 2
    c = _discovery("c")
    relayer = generate_node_identity("relayer")
    ids = []
    for i in range(4):
        ident = generate_node_identity(f"p{i}")
        ids.append(ident.node_id)
        await c._handle_peer_response(
            _pex_response(relayer.node_id, [{"credential": _credential(ident, address=f"5.5.5.{i}:9001")}]),
            _peer(relayer.node_id),
        )
    assert len(c.known_peers) <= 2  # cap holds despite 4 valid credentials


# ── _handle_peer_request (producer) ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_producer_emits_credential_and_skips_uncredentialed():
    a = _discovery("a")
    b = generate_node_identity("b")
    requester = generate_node_identity("requester")

    # A learns B via a genuine authenticated announce → stores B's credential.
    announce_payload = {
        "subtype": "discovery_announce",
        "address": "5.5.5.5:9001",
        "announce_time": 100.0,
    }
    _attest_announce_payload(b, announce_payload, "anonce")
    announce = P2PMessage(
        msg_type=MSG_GOSSIP, sender_id=b.node_id, payload=announce_payload, ttl=1, nonce="anonce",
    )
    await a._handle_announce(announce, _peer(generate_node_identity("relayer").node_id))
    assert b.node_id in a.known_peers
    assert a.known_peers[b.node_id].announce_credential is not None

    # A peer with NO credential (e.g. learned via legacy/direct-only path).
    from prsm.node.discovery import PeerInfo
    a.known_peers["uncredentialed"] = PeerInfo(node_id="uncredentialed", address="8.8.8.8:9001")

    await a._handle_peer_request(
        P2PMessage(msg_type=MSG_GOSSIP, sender_id=requester.node_id,
                   payload={"subtype": "discovery_peer_request", "max_peers": 20}, ttl=1, nonce="req"),
        _peer(requester.node_id),
    )
    a.transport.send_to_peer.assert_awaited()
    sent = a.transport.send_to_peer.await_args.args[1]
    emitted = sent.payload["peers"]
    node_ids = {_verify_peer_credential(e.get("credential")) for e in emitted}
    assert b.node_id in node_ids                 # B (credentialed) is relayed
    assert "uncredentialed" not in node_ids      # uncredentialed peer is NOT relayed
    assert None not in node_ids                   # every emitted entry verifies


@pytest.mark.asyncio
async def test_pex_round_trip_announce_to_relay():
    """End-to-end: B announces (authenticated) to A; A relays B via PEX to C; C
    learns B at B's signed address."""
    a = _discovery("a")
    c = _discovery("c")
    b = generate_node_identity("b")

    announce_payload = {"subtype": "discovery_announce", "address": "5.5.5.5:9001", "announce_time": 100.0}
    _attest_announce_payload(b, announce_payload, "anonce")
    announce = P2PMessage(msg_type=MSG_GOSSIP, sender_id=b.node_id, payload=announce_payload, ttl=1, nonce="anonce")
    await a._handle_announce(announce, _peer(generate_node_identity("rly").node_id))

    await a._handle_peer_request(
        P2PMessage(msg_type=MSG_GOSSIP, sender_id="cc",
                   payload={"subtype": "discovery_peer_request"}, ttl=1, nonce="req"),
        _peer("cc"),
    )
    relayed = a.transport.send_to_peer.await_args.args[1]
    await c._handle_peer_response(
        P2PMessage(msg_type=MSG_GOSSIP, sender_id=a.transport.identity.node_id,
                   payload=relayed.payload, ttl=1, nonce="pex"),
        _peer(a.transport.identity.node_id),
    )
    assert b.node_id in c.known_peers
    assert c.known_peers[b.node_id].address == "5.5.5.5:9001"
