"""Sprint 1246 — libp2p gossip-layer origin-authentication parity (Residual A).

The WebSocket GossipProtocol authenticates the ORIGIN of every gossip message
(sp934 _authenticate_origin: trust payload['origin'] only if its pubkey hashes to
the claimed node_id AND the signature verifies; else fall back to the relayer's
identity). Libp2pGossip did NOT — publish() attached no origin attestation and
_handle_gossip() handed callbacks the RAW (spoofable) envelope sender_id. Security-
critical consumers (settlement _on_ad: admit iff origin == announcer_id) were only
"secure by omission" — a forged ad failed the gate because the unauthenticated path
happened to surface a non-matching sender. This makes libp2p gossip secure BY DESIGN,
mirroring the WebSocket path. The discovery handlers already self-verify their `data`
(sp1086/1087/1097), so they are unaffected by the third-arg change.

Also (defense-in-depth): the connectivity-only PeerInfo entries (peer_join hydration,
DHT find_providers stub) are now explicitly node_id_authenticated=False so they can
never be trusted for sp1083 tier-binding — and the coarse class flag deliberately
stays False (flipping it would grant tier-trust to exactly those unauthenticated
entries, since the pool provider falls back to the class flag only when a per-entry
marker is None — see dht_backed_pool_provider).
"""
from __future__ import annotations

import asyncio
import json

import pytest

from prsm.node.gossip import _authenticate_origin, build_gossip_origin_fields
from prsm.node.identity import generate_node_identity
from prsm.node.libp2p_gossip import Libp2pGossip
from prsm.node.transport import MSG_GOSSIP, P2PMessage


def _transport(identity):
    """Minimal Libp2pTransport stand-in carrying a REAL signing identity and a
    PrsmPublish that captures the published envelope bytes."""
    class _Lib:
        def __init__(self):
            self.published = []

        def PrsmPublish(self, handle, topic, payload_bytes, length):
            self.published.append((topic, payload_bytes))
            return 0

        def PrsmSubscribe(self, handle, topic):
            return 0

    class _T:
        def __init__(self):
            self.identity = identity
            self._handle = 0
            self._lib = _Lib()

    return _T()


def _run(coro):
    return asyncio.run(coro)


# ── publish: attaches a verifiable sp934 origin attestation + nonce ──────────

def test_publish_attaches_origin_attestation_and_nonce():
    ident = generate_node_identity("publisher")
    g = Libp2pGossip(_transport(ident))
    data = {"announcer_id": ident.node_id, "batch_id": "b1"}

    assert _run(g.publish("settlement_batch_ad", data)) == 1
    topic, payload_bytes = g.transport._lib.published[-1]
    env = json.loads(payload_bytes.decode())

    # the sp934 envelope attestation fields + a fresh nonce must be present
    for k in ("origin", "origin_time", "origin_pubkey", "origin_sig", "nonce"):
        assert k in env, f"published envelope missing {k}"
    assert env["origin"] == ident.node_id

    # round-trip: the receiver's authenticator must recover the true author
    assert _authenticate_origin(env, env["nonce"], "some-relay") == ident.node_id


# ── _handle_gossip: hands callbacks the AUTHENTICATED origin ─────────────────

def _capture_cb():
    seen = {}

    async def cb(subtype, data, origin):
        seen["subtype"] = subtype
        seen["data"] = data
        seen["origin"] = origin

    return cb, seen


def _incoming(env, *, relay_sender):
    """A received gossip P2PMessage whose payload is the JSON envelope and whose
    frame sender_id is the relaying peer (the spoofable field)."""
    return P2PMessage(msg_type=MSG_GOSSIP, sender_id=relay_sender, payload=env)


def test_handle_gossip_delivers_authenticated_origin_for_valid_attestation():
    author = generate_node_identity("author")
    data = {"announcer_id": author.node_id, "v": 1}
    nonce = "n-valid-001"
    env = {"subtype": "settlement_batch_ad", "data": data, "nonce": nonce,
           **build_gossip_origin_fields(author, "settlement_batch_ad", data, 1700.0, nonce)}

    g = Libp2pGossip(_transport(generate_node_identity("receiver")))
    cb, seen = _capture_cb()
    g.subscribe("settlement_batch_ad", cb)
    # arrives RELAYED by a different peer — the authenticated origin must still be
    # the author, NOT the relay.
    _run(g._handle_gossip(_incoming(env, relay_sender="relay-peer"), None))

    assert seen["origin"] == author.node_id        # authenticated, not the relay
    assert seen["data"] == data


def test_handle_gossip_falls_back_to_relay_for_forged_origin():
    # THE security property: an envelope CLAIMING origin=victim but carrying no
    # valid attestation must NOT be attributed to the victim — the callback gets
    # the relayer's identity instead (so a settlement-ad origin==announcer_id gate
    # drops it).
    victim = generate_node_identity("victim")
    data = {"announcer_id": victim.node_id, "v": 1}
    forged = {"subtype": "settlement_batch_ad", "data": data, "nonce": "n",
              "origin": victim.node_id}  # no origin_pubkey/origin_sig

    g = Libp2pGossip(_transport(generate_node_identity("receiver")))
    cb, seen = _capture_cb()
    g.subscribe("settlement_batch_ad", cb)
    _run(g._handle_gossip(_incoming(forged, relay_sender="attacker-relay"), None))

    assert seen["origin"] == "attacker-relay"       # NOT the victim
    assert seen["origin"] != victim.node_id


def test_handle_gossip_drops_tampered_signature():
    # valid attestation but the signed data was tampered after signing → must not
    # authenticate as the author (falls back to relay).
    author = generate_node_identity("author")
    data = {"announcer_id": author.node_id, "v": 1}
    nonce = "n-tamper"
    env = {"subtype": "settlement_batch_ad", "data": data, "nonce": nonce,
           **build_gossip_origin_fields(author, "settlement_batch_ad", data, 1700.0, nonce)}
    env["data"] = {"announcer_id": author.node_id, "v": 999}  # tamper post-sign

    g = Libp2pGossip(_transport(generate_node_identity("receiver")))
    cb, seen = _capture_cb()
    g.subscribe("settlement_batch_ad", cb)
    _run(g._handle_gossip(_incoming(env, relay_sender="relay"), None))

    assert seen["origin"] == "relay"                # tamper detected → fallback


def test_settlement_ad_gate_semantics_forged_vs_legit():
    # Wire the actual settlement _on_ad gate semantics: legit (origin==announcer_id)
    # admitted; forged (origin!=announcer_id) dropped — proving the gossip-auth fix
    # makes the existing secure-by-omission gate secure-by-DESIGN.
    author = generate_node_identity("producer")
    admitted = []

    async def on_ad(subtype, data, origin):
        if data.get("announcer_id") and origin == data["announcer_id"]:
            admitted.append(data["batch_id"])

    g = Libp2pGossip(_transport(generate_node_identity("receiver")))
    g.subscribe("settlement_batch_ad", on_ad)

    legit_data = {"announcer_id": author.node_id, "batch_id": "ok"}
    nonce = "n-legit"
    legit = {"subtype": "settlement_batch_ad", "data": legit_data, "nonce": nonce,
             **build_gossip_origin_fields(author, "settlement_batch_ad", legit_data, 1.0, nonce)}
    forged = {"subtype": "settlement_batch_ad",
              "data": {"announcer_id": author.node_id, "batch_id": "bad"}, "nonce": "x",
              "origin": author.node_id}

    _run(g._handle_gossip(_incoming(legit, relay_sender="r1"), None))
    _run(g._handle_gossip(_incoming(forged, relay_sender="attacker"), None))

    assert "ok" in admitted          # legit ad admitted
    assert "bad" not in admitted     # forged ad dropped by the origin gate


def test_inbound_ledger_persists_authenticated_origin_nonce_and_attestation():
    # Review (w7x42hqne) parity fix: the inbound ledger write must mirror the
    # WebSocket GossipProtocol — persist the AUTHENTICATED origin (not the relay
    # sender_id), the signature-bound envelope nonce, and the sp961 attestation, so a
    # future digest catch-up consumer can re-verify authorship relayer-independently.
    author = generate_node_identity("author")
    data = {"announcer_id": author.node_id, "v": 1}
    nonce = "n-ledger"
    env = {"subtype": "settlement_batch_ad", "data": data, "nonce": nonce,
           **build_gossip_origin_fields(author, "settlement_batch_ad", data, 1700.0, nonce)}

    captured = {}

    class _Ledger:
        async def log_gossip(self, **kw):
            captured.update(kw)

    g = Libp2pGossip(_transport(generate_node_identity("receiver")))
    g.ledger = _Ledger()
    cb, _ = _capture_cb()
    g.subscribe("settlement_batch_ad", cb)
    _run(g._handle_gossip(_incoming(env, relay_sender="relay-peer"), None))

    assert captured["origin"] == author.node_id          # authenticated author, not relay
    assert captured["origin"] != "relay-peer"
    assert captured["nonce"] == nonce                     # signature-bound envelope nonce
    att = captured.get("attestation")
    assert att and att.get("origin_pubkey") and att.get("origin_sig")  # sp961 re-verify material


def test_discovery_callback_three_arg_unaffected():
    # A discovery-style callback (subtype, data, sender_id) that self-verifies its
    # `data` must keep receiving intact data; the third-arg change from raw
    # sender_id to authenticated origin must not break its arity or data.
    received = {}

    async def on_capability(subtype, data, sender_id):  # libp2p_discovery signature
        received["data"] = data

    g = Libp2pGossip(_transport(generate_node_identity("receiver")))
    g.subscribe("capability_announce", on_capability)
    env = {"subtype": "capability_announce", "data": {"node_id": "x", "caps": ["gpu"]},
           "nonce": "n"}
    _run(g._handle_gossip(_incoming(env, relay_sender="r"), None))
    assert received["data"] == {"node_id": "x", "caps": ["gpu"]}


# ── defense-in-depth: connectivity-only entries stay unauthenticated ─────────

def test_class_flag_stays_false_and_connectivity_entries_unauthenticated():
    # The coarse class flag must NOT be flipped to True: the pool provider uses it
    # only as the fallback for per-entry None, and the per-entry-None entries are
    # exactly the UNAUTHENTICATED connectivity/stub paths — flipping would grant
    # them tier-trust. Attested peers already carry per-entry True, so the flip has
    # no upside.
    from prsm.node.libp2p_discovery import Libp2pDiscovery
    assert Libp2pDiscovery.node_id_authenticated is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
