"""Sprint 1088 — authenticate bootstrap-relayed peers with a portable credential.

The bootstrap-server relay (`_hydrate_peers_from_bootstrap`) threaded each relayed peer's
(node_id, hardware_profile) into the Parallax pool keyed by an UNVERIFIED peer_id — so a
malicious bootstrap server, or a peer registering under a forged node_id, could inject an
(attacker_node_id -> victim_attestation) entry. This ports the sp1005 portable per-peer
credential to the bootstrap relay: the node signs a credential at registration, the server
relays it verbatim, and the consumer re-verifies it (sha256(pubkey)[:32]==node_id + sig)
before marking the peer authenticated + trusting the SIGNED hardware_profile.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.bootstrap.client import BootstrapPeer
from prsm.node.identity import generate_node_identity
from prsm.node.libp2p_discovery import Libp2pDiscovery


def _make_transport(identity):
    t = MagicMock()
    t.identity = identity
    t.port = 9001
    return t


def _discovery_with_hw(identity, hw):
    d = Libp2pDiscovery(_make_transport(identity))
    d._local_hardware_profile = hw
    d.set_local_capabilities(capabilities=["inference"], backends=["local"],
                             gpu_available=True)
    return d


_HW = {"tflops_fp16": 9.0, "ram_total_gb": 64.0, "gpu_vram_gb": 24.0,
       "gpu_name": "TEE Box", "attestation": "QUOTEB64"}


# ── producer: a node builds a verifiable credential for its own announce ─────────

def test_own_credential_is_verifiable():
    from prsm.node.discovery import _verify_peer_credential
    ident = generate_node_identity("me")
    d = _discovery_with_hw(ident, _HW)
    cred = d._own_announce_credential()
    assert cred is not None
    assert _verify_peer_credential(cred) == ident.node_id
    # the signed payload carries the hardware_profile (so the attestation is bound)
    assert cred["signed_payload"]["hardware_profile"] == _HW


# ── consumer: a credentialed bootstrap peer is authenticated + its signed hw used ──

def test_credentialed_bootstrap_peer_authenticated():
    provider = generate_node_identity("provider")
    # the provider builds its own credential (as it would at registration)
    cred = _discovery_with_hw(provider, _HW)._own_announce_credential()

    consumer = Libp2pDiscovery(_make_transport(generate_node_identity("consumer")))
    bp = BootstrapPeer(
        peer_id=provider.node_id, address="1.2.3.4", port=9001,
        capabilities=["inference"], hardware_profile=_HW, announce_credential=cred)
    consumer._hydrate_peers_from_bootstrap([bp])

    entry = consumer._capability_index[provider.node_id]
    assert entry.node_id_authenticated is True
    assert entry.hardware_profile == _HW


def test_uncredentialed_bootstrap_peer_unauthenticated():
    consumer = Libp2pDiscovery(_make_transport(generate_node_identity("consumer")))
    bp = BootstrapPeer(
        peer_id="p" * 32, address="1.2.3.4", port=9001,
        capabilities=["inference"], hardware_profile=_HW, announce_credential=None)
    consumer._hydrate_peers_from_bootstrap([bp])
    entry = consumer._capability_index["p" * 32]
    # kept for routing/liveness, but NOT authenticated → its attestation grants no tier
    assert entry.node_id_authenticated is False


def test_forged_credential_for_other_peer_unauthenticated():
    """A credential signed by the ATTACKER but relayed under the VICTIM's peer_id —
    sha256(attacker_pubkey)[:32] != victim peer_id, so it's not authenticated."""
    victim = generate_node_identity("victim")
    attacker = generate_node_identity("attacker")
    attacker_cred = _discovery_with_hw(attacker, _HW)._own_announce_credential()

    consumer = Libp2pDiscovery(_make_transport(generate_node_identity("consumer")))
    bp = BootstrapPeer(
        peer_id=victim.node_id, address="1.2.3.4", port=9001,
        hardware_profile=_HW, announce_credential=attacker_cred)
    consumer._hydrate_peers_from_bootstrap([bp])
    entry = consumer._capability_index[victim.node_id]
    assert entry.node_id_authenticated is False


def test_server_tampered_hardware_profile_overridden_by_signed():
    """A malicious server relays a TAMPERED hardware_profile alongside a genuine
    credential — the consumer must use the SIGNED hw from the credential, not the relay."""
    provider = generate_node_identity("provider")
    cred = _discovery_with_hw(provider, _HW)._own_announce_credential()
    tampered = dict(_HW, attestation="ATTACKER_QUOTE", tflops_fp16=999.0)

    consumer = Libp2pDiscovery(_make_transport(generate_node_identity("consumer")))
    bp = BootstrapPeer(
        peer_id=provider.node_id, address="1.2.3.4", port=9001,
        hardware_profile=tampered, announce_credential=cred)
    consumer._hydrate_peers_from_bootstrap([bp])
    entry = consumer._capability_index[provider.node_id]
    assert entry.node_id_authenticated is True
    assert entry.hardware_profile == _HW   # the signed profile, not the tampered relay


# ── pool builder: a per-peer authenticated marker overrides the coarse flag ─────

def test_authenticated_peer_uses_signed_capabilities(monkeypatch):
    """review MEDIUM-2 — a malicious server can't rewrite an authenticated peer's
    capabilities; the SIGNED ones win."""
    provider = generate_node_identity("provider")
    d = _discovery_with_hw(provider, _HW)   # signs capabilities=["inference"]
    cred = d._own_announce_credential()

    consumer = Libp2pDiscovery(_make_transport(generate_node_identity("consumer")))
    bp = BootstrapPeer(
        peer_id=provider.node_id, address="1.2.3.4", port=9001,
        capabilities=["storage", "evil"],   # server-rewritten (unsigned)
        hardware_profile=_HW, announce_credential=cred)
    consumer._hydrate_peers_from_bootstrap([bp])
    entry = consumer._capability_index[provider.node_id]
    assert entry.capabilities == ["inference"]   # the signed set, not the relay's


@pytest.mark.asyncio
async def test_authenticated_announce_upgrades_existing_unauth_entry():
    """review LOW-1 — an authenticated capability_announce upgrades a previously
    credential-less (unauthenticated) bootstrap entry to authenticated."""
    from prsm.node.discovery import _attest_announce_payload
    consumer = Libp2pDiscovery(_make_transport(generate_node_identity("consumer")))
    peer = generate_node_identity("peer")
    # seed an unauthenticated entry (as an uncredentialed bootstrap hydrate would)
    bp = BootstrapPeer(peer_id=peer.node_id, address="1.2.3.4", port=9001,
                       hardware_profile=_HW, announce_credential=None)
    consumer._hydrate_peers_from_bootstrap([bp])
    assert consumer._capability_index[peer.node_id].node_id_authenticated is False

    # now an authenticated gossip announce arrives from the same peer
    data = {"node_id": peer.node_id, "capabilities": ["inference"],
            "supported_backends": [], "gpu_available": True,
            "startup_timestamp": 1.0, "nonce": "z"}
    _attest_announce_payload(peer, data, "z")
    await consumer._on_capability("capability_announce", data, peer.node_id)
    assert consumer._capability_index[peer.node_id].node_id_authenticated is True


def test_pool_builder_honors_per_peer_marker_over_libp2p_default(monkeypatch):
    import prsm.node.dht_backed_pool_provider as mod
    from prsm.node.discovery import PeerInfo
    monkeypatch.setattr(mod, "verified_tier_attestation",
                        lambda blob, *, node_id=None: "tier-sgx")

    # a libp2p-style discovery (coarse node_id_authenticated=False) but a per-peer
    # AUTHENTICATED entry → the attestation must be honored
    discovery = MagicMock()
    discovery.node_id_authenticated = False
    authed = PeerInfo(node_id="a" * 32, address="1.2.3.4:9001",
                      hardware_profile=dict(_HW), node_id_authenticated=True)
    unauthed = PeerInfo(node_id="b" * 32, address="5.6.7.8:9001",
                        hardware_profile=dict(_HW), node_id_authenticated=False)
    discovery.get_known_peers = lambda: [authed, unauthed]

    provider = mod.build_dht_backed_pool_provider(_node_with(discovery))
    gpus = {g.node_id: g for g in provider()}
    assert gpus["a" * 32].tier_attestation == "tier-sgx"      # per-peer True honored
    assert gpus["b" * 32].tier_attestation == "tier-none"     # per-peer False gated


def _node_with(discovery):
    node = MagicMock()
    node.identity.node_id = "self"
    node.discovery = discovery
    node.stake_reader = None
    return node


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
