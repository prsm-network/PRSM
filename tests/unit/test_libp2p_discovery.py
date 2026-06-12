"""
Unit tests for prsm.node.libp2p_discovery.Libp2pDiscovery.

All tests use mocked transport / gossip objects — no Go library required.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.discovery import PeerInfo, _attest_announce_payload
from prsm.node.identity import generate_node_identity
from prsm.node.libp2p_discovery import Libp2pDiscovery


# ── Helpers ──────────────────────────────────────────────────────────────────


def _attested_capability(
    identity, *, caps=None, backends=None, gpu=False, startup=0.0, nonce="n1",
):
    """Sprint 1086 — a capability_announce carrying a valid origin attestation
    (node_id == sha256(pubkey)[:32] + a signature), the shape _on_capability now
    requires. node_id is the identity's real (pubkey-derived) node_id."""
    data = {
        "node_id": identity.node_id,
        "capabilities": caps or [],
        "supported_backends": backends or [],
        "gpu_available": gpu,
        "startup_timestamp": startup,
        "nonce": nonce,
    }
    _attest_announce_payload(identity, data, nonce)
    return data


def _attested_shard(identity, cid, *, nonce="s1"):
    """Sprint 1087 — an origin-attested shard_available announce."""
    data = {"cid": cid, "node_id": identity.node_id, "nonce": nonce}
    _attest_announce_payload(identity, data, nonce)
    return data


def _make_transport(node_id: str = "local-node-001") -> MagicMock:
    transport = MagicMock()
    transport.identity = MagicMock()
    transport.identity.node_id = node_id
    transport._handle = 0
    transport._lib = MagicMock()
    transport.connect_to_peer = AsyncMock(return_value=None)
    transport.dht_provide = AsyncMock(return_value=True)
    transport.dht_find_providers = AsyncMock(return_value=[])
    return transport


def _make_gossip() -> MagicMock:
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    gossip.publish = AsyncMock(return_value=1)
    return gossip


def _peer_info(
    node_id: str = "peer-001",
    gpu: bool = False,
    caps: list = None,
    backends: list = None,
) -> PeerInfo:
    return PeerInfo(
        node_id=node_id,
        address="127.0.0.1:9001",
        display_name="",
        roles=[],
        capabilities=caps or [],
        supported_backends=backends or [],
        gpu_available=gpu,
        last_seen=time.time(),
        last_capability_update=time.time(),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestCapabilityAnnounceAuth:
    """Sprint 1086 — _on_capability authenticates node_id before trusting the announce."""

    @pytest.mark.asyncio
    async def test_unattested_announce_dropped(self):
        d = Libp2pDiscovery(_make_transport())
        # legacy-shape announce with no origin attestation → dropped
        await d._on_capability("capability_announce", {
            "node_id": "spoofed-peer", "capabilities": ["inference"],
            "gpu_available": True,
        }, "spoofed-peer")
        assert d._capability_index == {}

    @pytest.mark.asyncio
    async def test_attacker_claiming_victim_node_id_dropped(self):
        """An attacker signs with their own key but claims the VICTIM's node_id —
        sha256(attacker_pubkey)[:32] != victim_node_id, so it's dropped."""
        d = Libp2pDiscovery(_make_transport())
        victim = generate_node_identity("victim")
        attacker = generate_node_identity("attacker")
        data = _attested_capability(attacker, caps=["inference"], gpu=True)
        data["node_id"] = victim.node_id   # claim the victim's id (sig won't match)
        await d._on_capability("capability_announce", data, attacker.node_id)
        assert victim.node_id not in d._capability_index
        assert d._capability_index == {}

    @pytest.mark.asyncio
    async def test_tampered_payload_dropped(self):
        """A relayer flips gpu_available after the announcer signed → signature breaks."""
        d = Libp2pDiscovery(_make_transport())
        peer = generate_node_identity("peer")
        data = _attested_capability(peer, caps=["inference"], gpu=False)
        data["gpu_available"] = True   # tamper post-signature
        await d._on_capability("capability_announce", data, peer.node_id)
        assert d._capability_index == {}

    @pytest.mark.asyncio
    async def test_announce_capabilities_emits_attested_payload(self):
        transport = _make_transport()
        # give the transport a real identity so the producer can actually sign
        ident = generate_node_identity("local")
        transport.identity = ident
        gossip = _make_gossip()
        d = Libp2pDiscovery(transport, gossip=gossip)
        await d.announce_capabilities()
        published = gossip.publish.await_args.args[1]
        assert published["node_id"] == ident.node_id
        assert published.get("origin_pubkey") and published.get("origin_sig")
        # round-trips through the verifier
        from prsm.node.libp2p_discovery import _authenticated_capability_node_id
        assert _authenticated_capability_node_id(published) == ident.node_id


class TestCapabilityIndex:
    """_on_capability updates the index and find_peers_with_gpu works."""

    @pytest.mark.asyncio
    async def test_capability_index(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport)

        peer = generate_node_identity("gpu-peer")
        data = _attested_capability(
            peer, caps=["inference"], backends=["local"], gpu=True)
        await discovery._on_capability("capability_announce", data, peer.node_id)

        gpu_peers = discovery.find_peers_with_gpu()
        assert len(gpu_peers) == 1
        assert gpu_peers[0].node_id == peer.node_id
        assert gpu_peers[0].gpu_available is True

    @pytest.mark.asyncio
    async def test_capability_index_update_existing(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport)

        peer = generate_node_identity("peer-001")
        # First announcement
        await discovery._on_capability(
            "capability_announce",
            _attested_capability(peer, caps=["inference"], gpu=False, nonce="a"),
            peer.node_id,
        )
        assert not discovery._capability_index[peer.node_id].gpu_available

        # Second announcement with GPU now available
        await discovery._on_capability(
            "capability_announce",
            _attested_capability(peer, caps=["inference"], gpu=True, nonce="b"),
            peer.node_id,
        )
        assert discovery._capability_index[peer.node_id].gpu_available is True

    @pytest.mark.asyncio
    async def test_non_gpu_peer_not_in_gpu_list(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport)

        await discovery._on_capability(
            "capability_announce",
            {"node_id": "cpu-peer", "capabilities": [], "supported_backends": [],
             "gpu_available": False},
            "cpu-peer",
        )

        assert discovery.find_peers_with_gpu() == []


class TestShardCache:
    """_on_shard_available populates the shard cache."""

    @pytest.mark.asyncio
    async def test_shard_cache(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport)

        peer = generate_node_identity("shard-peer-001")
        await discovery._on_shard_available(
            "shard_available", _attested_shard(peer, "QmTest123"), peer.node_id)

        assert "QmTest123" in discovery._shard_cache
        assert peer.node_id in discovery._shard_cache["QmTest123"]

    @pytest.mark.asyncio
    async def test_shard_cache_multiple_providers(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport)

        cid = "QmMultiProvider"
        for i in range(3):
            peer = generate_node_identity(f"peer-{i}")
            await discovery._on_shard_available(
                "shard_available", _attested_shard(peer, cid), peer.node_id)

        assert len(discovery._shard_cache[cid]) == 3

    @pytest.mark.asyncio
    async def test_shard_cache_no_duplicates(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport)

        cid = "QmDup"
        peer = generate_node_identity("shard-peer")
        for n in range(5):
            await discovery._on_shard_available(
                "shard_available", _attested_shard(peer, cid, nonce=f"s{n}"),
                peer.node_id)

        assert len(discovery._shard_cache[cid]) == 1

    @pytest.mark.asyncio
    async def test_unattested_shard_dropped(self) -> None:
        """sp1087 — a shard_available with no valid origin attestation is dropped."""
        discovery = Libp2pDiscovery(_make_transport())
        await discovery._on_shard_available(
            "shard_available", {"cid": "QmX", "node_id": "spoofed"}, "spoofed")
        assert discovery._shard_cache == {}

    @pytest.mark.asyncio
    async def test_shard_attacker_claims_victim_dropped(self) -> None:
        """An attacker claims the VICTIM provides a shard — sig won't match → dropped."""
        discovery = Libp2pDiscovery(_make_transport())
        victim = generate_node_identity("victim")
        attacker = generate_node_identity("attacker")
        data = _attested_shard(attacker, "QmY")
        data["node_id"] = victim.node_id
        await discovery._on_shard_available("shard_available", data, attacker.node_id)
        assert discovery._shard_cache == {}


class TestBootstrapDegraded:
    """When all bootstrap connections return None, status shows degraded."""

    @pytest.mark.asyncio
    async def test_bootstrap_degraded(self) -> None:
        transport = _make_transport()
        transport.connect_to_peer = AsyncMock(return_value=None)

        discovery = Libp2pDiscovery(
            transport,
            bootstrap_nodes=["ws://bootstrap1.example.com:8765",
                             "ws://bootstrap2.example.com:8765"],
        )

        connected = await discovery.bootstrap()

        assert connected == 0
        status = discovery.get_bootstrap_status()
        assert status["degraded"] is True
        assert status["connected"] == 0
        assert status["attempted"] == 2

    @pytest.mark.asyncio
    async def test_bootstrap_success(self) -> None:
        transport = _make_transport()
        mock_peer = MagicMock()
        mock_peer.peer_id = "remote-bootstrap-001"
        transport.connect_to_peer = AsyncMock(return_value=mock_peer)

        discovery = Libp2pDiscovery(
            transport,
            bootstrap_nodes=["ws://bootstrap1.example.com:8765"],
        )

        connected = await discovery.bootstrap()

        assert connected == 1
        status = discovery.get_bootstrap_status()
        assert status["degraded"] is False
        assert status["connected"] == 1

    @pytest.mark.asyncio
    async def test_no_bootstrap_nodes_not_degraded(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport, bootstrap_nodes=[])

        connected = await discovery.bootstrap()

        assert connected == 0
        assert discovery.get_bootstrap_status()["degraded"] is False


class TestFindPeersByCapability:
    """find_peers_by_capability with match_all=True requires all capabilities."""

    def test_find_peers_by_capability_match_all(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport)

        # Peer with both capabilities
        discovery._capability_index["peer-both"] = _peer_info(
            "peer-both", caps=["inference", "embedding"]
        )
        # Peer with only one capability
        discovery._capability_index["peer-one"] = _peer_info(
            "peer-one", caps=["inference"]
        )

        results = discovery.find_peers_by_capability(
            ["inference", "embedding"], match_all=True
        )

        ids = [p.node_id for p in results]
        assert "peer-both" in ids
        assert "peer-one" not in ids

    def test_find_peers_by_capability_match_any(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport)

        discovery._capability_index["peer-a"] = _peer_info("peer-a", caps=["inference"])
        discovery._capability_index["peer-b"] = _peer_info("peer-b", caps=["embedding"])
        discovery._capability_index["peer-c"] = _peer_info("peer-c", caps=["benchmark"])

        results = discovery.find_peers_by_capability(
            ["inference", "embedding"], match_all=False
        )

        ids = {p.node_id for p in results}
        assert "peer-a" in ids
        assert "peer-b" in ids
        assert "peer-c" not in ids

    def test_find_peers_case_insensitive(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport)

        discovery._capability_index["peer-upper"] = _peer_info(
            "peer-upper", caps=["Inference", "EMBEDDING"]
        )

        results = discovery.find_peers_by_capability(["inference", "embedding"])
        assert len(results) == 1

    def test_find_peers_with_backend(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport)

        discovery._capability_index["anthropic-peer"] = _peer_info(
            "anthropic-peer", backends=["anthropic", "openai"]
        )
        discovery._capability_index["local-peer"] = _peer_info(
            "local-peer", backends=["local"]
        )

        results = discovery.find_peers_with_backend("anthropic")
        assert len(results) == 1
        assert results[0].node_id == "anthropic-peer"


class TestStartSubscribes:
    """On start(), gossip.subscribe is called for capability_announce and shard_available."""

    @pytest.mark.asyncio
    async def test_start_subscribes_to_gossip(self) -> None:
        transport = _make_transport()
        # Make connect_to_peer succeed so bootstrap doesn't degrade
        transport.connect_to_peer = AsyncMock(return_value=MagicMock())
        gossip = _make_gossip()

        discovery = Libp2pDiscovery(
            transport,
            bootstrap_nodes=["ws://b1.example.com:8765"],
            gossip=gossip,
        )
        await discovery.start()

        subscribed_subtypes = {
            call.args[0] for call in gossip.subscribe.call_args_list
        }
        assert "capability_announce" in subscribed_subtypes
        assert "shard_available" in subscribed_subtypes

    @pytest.mark.asyncio
    async def test_start_without_gossip_no_error(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport, bootstrap_nodes=[])
        # Should not raise even with no gossip
        await discovery.start()


class TestAnnounceCapabilities:
    """announce_capabilities publishes the local capability payload via gossip."""

    @pytest.mark.asyncio
    async def test_announce_capabilities(self) -> None:
        transport = _make_transport()
        gossip = _make_gossip()

        discovery = Libp2pDiscovery(transport, gossip=gossip)
        discovery.set_local_capabilities(["inference"], ["anthropic"], gpu_available=True)

        result = await discovery.announce_capabilities()

        assert result == 1
        gossip.publish.assert_called_once()
        call_args = gossip.publish.call_args
        assert call_args.args[0] == "capability_announce"
        payload = call_args.args[1]
        assert payload["capabilities"] == ["inference"]
        assert payload["gpu_available"] is True

    @pytest.mark.asyncio
    async def test_announce_capabilities_no_gossip(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport)  # no gossip

        result = await discovery.announce_capabilities()
        assert result == 0


class TestPeerInfoReliability:
    """Tests for PeerInfo reliability tracking fields."""

    def test_reliability_score_new_peer(self):
        peer = PeerInfo(node_id="peer1", address="1.2.3.4:9000")
        assert peer.reliability_score == 1.0

    def test_reliability_score_all_successes(self):
        peer = PeerInfo(node_id="peer1", address="1.2.3.4:9000")
        peer.job_success_count = 10
        assert peer.reliability_score == 1.0

    def test_reliability_score_mixed(self):
        peer = PeerInfo(node_id="peer1", address="1.2.3.4:9000")
        peer.job_success_count = 2
        peer.job_failure_count = 1
        assert abs(peer.reliability_score - 0.6667) < 0.01

    def test_reliability_score_all_failures(self):
        peer = PeerInfo(node_id="peer1", address="1.2.3.4:9000")
        peer.job_failure_count = 5
        assert peer.reliability_score == 0.0

    def test_startup_timestamp_default(self):
        peer = PeerInfo(node_id="peer1", address="1.2.3.4:9000")
        assert peer.startup_timestamp == 0.0


def _make_discovery():
    """Create a Libp2pDiscovery with a mocked transport."""
    transport = MagicMock()
    transport.identity = MagicMock()
    transport.identity.node_id = "local_node"
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    d = Libp2pDiscovery(transport=transport, gossip=gossip)
    return d


class TestReliabilityMethods:
    """Tests for job success/failure recording."""

    def test_record_job_success(self):
        d = _make_discovery()
        d._capability_index["peer1"] = PeerInfo(node_id="peer1", address="")
        d.record_job_success("peer1")
        assert d._capability_index["peer1"].job_success_count == 1

    def test_record_job_failure(self):
        d = _make_discovery()
        d._capability_index["peer1"] = PeerInfo(node_id="peer1", address="")
        d.record_job_failure("peer1")
        assert d._capability_index["peer1"].job_failure_count == 1
        assert d._capability_index["peer1"].last_failure_time > 0

    def test_record_unknown_peer_is_noop(self):
        d = _make_discovery()
        d.record_job_success("unknown_peer")
        d.record_job_failure("unknown_peer")


class TestConditionalReset:
    """Tests for capability re-announcement with conditional reliability reset."""

    @pytest.mark.asyncio
    async def test_heartbeat_does_not_reset_reliability(self):
        d = _make_discovery()
        d._capability_index["peer1"] = PeerInfo(
            node_id="peer1", address="",
            startup_timestamp=1000.0,
            job_success_count=5, job_failure_count=3,
        )
        await d._on_capability("capability_announce", {
            "node_id": "peer1",
            "capabilities": [],
            "supported_backends": [],
            "gpu_available": False,
            "startup_timestamp": 1000.0,
        }, "peer1")
        assert d._capability_index["peer1"].job_failure_count == 3
        assert d._capability_index["peer1"].job_success_count == 5

    @pytest.mark.asyncio
    async def test_restart_resets_reliability(self):
        d = _make_discovery()
        peer = generate_node_identity("peer1")
        d._capability_index[peer.node_id] = PeerInfo(
            node_id=peer.node_id, address="",
            startup_timestamp=1000.0,
            job_success_count=5, job_failure_count=3,
        )
        await d._on_capability(
            "capability_announce",
            _attested_capability(peer, startup=2000.0), peer.node_id)
        assert d._capability_index[peer.node_id].job_failure_count == 0
        assert d._capability_index[peer.node_id].job_success_count == 0
        assert d._capability_index[peer.node_id].startup_timestamp == 2000.0

    @pytest.mark.asyncio
    async def test_capability_change_resets_reliability(self):
        d = _make_discovery()
        peer = generate_node_identity("peer1")
        d._capability_index[peer.node_id] = PeerInfo(
            node_id=peer.node_id, address="",
            capabilities=["compute"],
            startup_timestamp=1000.0,
            job_success_count=5, job_failure_count=3,
        )
        await d._on_capability(
            "capability_announce",
            _attested_capability(
                peer, caps=["compute", "gpu"], gpu=True, startup=1000.0),
            peer.node_id)
        assert d._capability_index[peer.node_id].job_failure_count == 0
        assert d._capability_index[peer.node_id].job_success_count == 0

    @pytest.mark.asyncio
    async def test_new_peer_announcement(self):
        d = _make_discovery()
        peer = generate_node_identity("new_peer")
        await d._on_capability(
            "capability_announce",
            _attested_capability(peer, caps=["storage"], startup=5000.0),
            peer.node_id)
        assert peer.node_id in d._capability_index
        assert d._capability_index[peer.node_id].startup_timestamp == 5000.0
        assert d._capability_index[peer.node_id].reliability_score == 1.0
