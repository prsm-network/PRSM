"""
Integration tests for compute/storage lifecycle over libp2p transport.

Uses a MockLibp2pTransport that routes messages between in-process nodes
with JSON serialization fidelity (mimics the FFI boundary).
"""
import asyncio
import json
import time
import uuid
import pytest
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

from prsm.node.transport import MSG_GOSSIP, MSG_DIRECT, P2PMessage, PeerConnection
from prsm.node.discovery import PeerInfo, _attest_announce_payload
from prsm.node.identity import generate_node_identity


def _attested_capability(identity, *, caps=None, backends=None, gpu=False,
                         startup=0.0, nonce="n1"):
    """Sprint 1086 — an origin-attested capability_announce (the shape
    Libp2pDiscovery._on_capability now requires)."""
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


class MockLibp2pTransport:
    """In-process transport that routes messages with JSON serialization fidelity."""

    def __init__(self, node_id: str, network: "MockNetwork"):
        # Sprint 1086 — a REAL signing identity so announce_capabilities can attest
        # (the producer now signs; the consumer verifies sha256(pubkey)==node_id).
        self.identity = generate_node_identity(node_id)
        self._handlers: Dict[str, List[Callable]] = {}
        self._network = network
        self._peers: Dict[str, PeerConnection] = {}

    def on_message(self, msg_type: str, handler: Callable) -> None:
        self._handlers.setdefault(msg_type, []).append(handler)

    async def send_to_peer(self, peer_id: str, msg: P2PMessage) -> bool:
        """Send via network with JSON round-trip for serialization fidelity."""
        target = self._network.get_transport(peer_id)
        if target is None:
            return False
        wire = json.dumps({
            "msg_type": msg.msg_type,
            "sender_id": msg.sender_id,
            "payload": msg.payload,
            "timestamp": msg.timestamp,
            "signature": msg.signature,
            "ttl": msg.ttl,
            "nonce": msg.nonce,
        })
        raw = json.loads(wire)
        reconstructed = P2PMessage(
            msg_type=raw["msg_type"],
            sender_id=raw["sender_id"],
            payload=raw["payload"],
            timestamp=raw["timestamp"],
            signature=raw["signature"],
            ttl=raw["ttl"],
            nonce=raw["nonce"],
        )
        peer = PeerConnection(peer_id=msg.sender_id, address="mock", websocket=None)
        for handler in target._handlers.get(msg.msg_type, []):
            await handler(reconstructed, peer)
        return True

    def sign(self, msg):
        pass

    async def connect_to_peer(self, address: str):
        """Stub — no real connection needed in tests."""
        return None


class MockNetwork:
    """Routes messages between MockLibp2pTransport instances."""

    def __init__(self):
        self._transports: Dict[str, MockLibp2pTransport] = {}

    def add_node(self, node_id: str) -> MockLibp2pTransport:
        t = MockLibp2pTransport(node_id, self)
        self._transports[node_id] = t
        return t

    def get_transport(self, node_id: str) -> Optional[MockLibp2pTransport]:
        return self._transports.get(node_id)


class MockGossip:
    """In-process gossip that broadcasts to all nodes with JSON fidelity."""

    def __init__(self, node_id: str, network: MockNetwork):
        self._node_id = node_id
        self._network = network
        self._callbacks: Dict[str, List[Callable]] = {}
        self.published: List[Dict[str, Any]] = []
        self.ledger = None

    def subscribe(self, subtype: str, callback: Callable) -> None:
        self._callbacks.setdefault(subtype, []).append(callback)

    async def publish(self, subtype: str, data: Dict[str, Any], ttl=None) -> int:
        wire = json.dumps({"subtype": subtype, "data": data, "sender_id": self._node_id})
        raw = json.loads(wire)

        self.published.append(raw)

        for node_id, transport in self._network._transports.items():
            gossip = _gossip_registry.get(node_id)
            if gossip:
                for cb in gossip._callbacks.get(subtype, []):
                    await cb(subtype, raw["data"], raw["sender_id"])
        return 1


_gossip_registry: Dict[str, MockGossip] = {}


def make_test_node(node_id: str, network: MockNetwork):
    """Create a test node with mock transport and gossip."""
    transport = network.add_node(node_id)
    gossip = MockGossip(node_id, network)
    _gossip_registry[node_id] = gossip
    return transport, gossip


# ── Test classes ──────────────────────────────────────────────────────────────


class TestComputeJobLifecycle:

    @pytest.mark.asyncio
    async def test_job_offer_accept_confirm_result_payment(self):
        _gossip_registry.clear()
        network = MockNetwork()
        req_transport, req_gossip = make_test_node("requester", network)
        prov_transport, prov_gossip = make_test_node("provider", network)

        messages_received = {"requester": [], "provider": []}

        async def on_job_offer(subtype, data, sender):
            messages_received["provider"].append(("job_offer", data))
            await prov_gossip.publish("job_accept", {
                "job_id": data["job_id"],
                "provider_id": "provider",
            })

        prov_gossip.subscribe("job_offer", on_job_offer)

        async def on_job_accept(subtype, data, sender):
            messages_received["requester"].append(("job_accept", data))
            await req_gossip.publish("job_confirm", {
                "job_id": data["job_id"],
                "provider_id": data["provider_id"],
                "requester_id": "requester",
            })

        req_gossip.subscribe("job_accept", on_job_accept)

        async def on_job_confirm(subtype, data, sender):
            messages_received["provider"].append(("job_confirm", data))
            await prov_gossip.publish("job_result", {
                "job_id": data["job_id"],
                "provider_id": "provider",
                "status": "completed",
                "result": {"output": "42"},
            })

        prov_gossip.subscribe("job_confirm", on_job_confirm)

        result_received = asyncio.Event()

        async def on_job_result(subtype, data, sender):
            messages_received["requester"].append(("job_result", data))
            result_received.set()

        req_gossip.subscribe("job_result", on_job_result)

        await req_gossip.publish("job_offer", {
            "job_id": "test_job_001",
            "job_type": "inference",
            "requester_id": "requester",
            "payload": {"prompt": "What is 6*7?"},
            "ftns_budget": 1.0,
        })

        await asyncio.wait_for(result_received.wait(), timeout=5.0)

        assert len(messages_received["provider"]) == 2
        assert messages_received["provider"][0][0] == "job_offer"
        assert messages_received["provider"][1][0] == "job_confirm"
        assert len(messages_received["requester"]) == 2
        assert messages_received["requester"][0][0] == "job_accept"
        assert messages_received["requester"][1][0] == "job_result"
        assert messages_received["requester"][1][1]["result"]["output"] == "42"


class TestCapabilityDiscovery:

    @pytest.mark.asyncio
    async def test_capability_index_populated(self):
        from prsm.node.libp2p_discovery import Libp2pDiscovery

        _gossip_registry.clear()
        network = MockNetwork()
        t_a, g_a = make_test_node("node_a", network)
        t_b, g_b = make_test_node("node_b", network)

        disc_a = Libp2pDiscovery(transport=t_a, gossip=g_a)
        disc_b = Libp2pDiscovery(transport=t_b, gossip=g_b)

        await disc_a.start()
        await disc_b.start()

        disc_a.set_local_capabilities(
            capabilities=["compute", "inference", "gpu"],
            backends=["local"],
            gpu_available=True,
        )
        await disc_a.announce_capabilities()

        disc_b.set_local_capabilities(
            capabilities=["storage", "pinning"],
            backends=[],
            gpu_available=False,
        )
        await disc_b.announce_capabilities()

        await asyncio.sleep(0.01)

        gpu_peers = disc_b.find_peers_with_gpu()
        assert len(gpu_peers) == 1
        assert gpu_peers[0].node_id == t_a.identity.node_id

        storage_peers = disc_a.find_peers_with_capability("storage")
        assert len(storage_peers) == 1
        assert storage_peers[0].node_id == t_b.identity.node_id

        assert all(p.node_id != t_a.identity.node_id
                   for p in disc_a.find_peers_with_capability("storage"))


class TestReliabilityTracking:

    @pytest.mark.asyncio
    async def test_reliability_degrades_and_resets(self):
        from prsm.node.libp2p_discovery import Libp2pDiscovery

        _gossip_registry.clear()
        network = MockNetwork()
        t_req, g_req = make_test_node("requester", network)
        t_prov, g_prov = make_test_node("provider", network)

        disc = Libp2pDiscovery(transport=t_req, gossip=g_req)
        await disc.start()

        prov = t_prov.identity   # the real provider identity (attests its announces)
        pid = prov.node_id
        await disc._on_capability(
            "capability_announce",
            _attested_capability(prov, caps=["compute", "gpu"], backends=["local"],
                                 gpu=True, startup=1000.0, nonce="a"),
            pid)

        disc.record_job_success(pid)
        disc.record_job_success(pid)
        disc.record_job_failure(pid)

        peer = disc._capability_index[pid]
        assert abs(peer.reliability_score - 0.6667) < 0.01

        # Same startup_timestamp — no reset, counters accumulate
        await disc._on_capability(
            "capability_announce",
            _attested_capability(prov, caps=["compute", "gpu"], backends=["local"],
                                 gpu=True, startup=1000.0, nonce="b"),
            pid)
        assert peer.job_failure_count == 1

        # New startup_timestamp — counters reset
        await disc._on_capability(
            "capability_announce",
            _attested_capability(prov, caps=["compute", "gpu"], backends=["local"],
                                 gpu=True, startup=2000.0, nonce="c"),
            pid)
        assert peer.job_failure_count == 0
        assert peer.job_success_count == 0
        assert peer.reliability_score == 1.0


class TestDirectP2PChallenge:

    @pytest.mark.asyncio
    async def test_challenge_and_proof_via_direct_p2p(self):
        _gossip_registry.clear()
        network = MockNetwork()
        challenger_transport, challenger_gossip = make_test_node("challenger", network)
        provider_transport, provider_gossip = make_test_node("provider", network)

        direct_messages = {"challenger": [], "provider": []}

        async def provider_direct_handler(msg: P2PMessage, peer: PeerConnection):
            subtype = msg.payload.get("subtype", "")
            direct_messages["provider"].append(subtype)
            if subtype == "storage_challenge":
                proof_msg = P2PMessage(
                    msg_type=MSG_DIRECT,
                    sender_id="provider",
                    payload={
                        "subtype": "storage_proof_response",
                        "proof": {"challenge_id": msg.payload["challenge"]["challenge_id"], "data": "merkle_proof_bytes"},
                        "challenge_id": msg.payload["challenge"]["challenge_id"],
                        "provider_id": "provider",
                    },
                )
                await provider_transport.send_to_peer("challenger", proof_msg)

        provider_transport.on_message(MSG_DIRECT, provider_direct_handler)

        proof_received = asyncio.Event()

        async def challenger_direct_handler(msg: P2PMessage, peer: PeerConnection):
            subtype = msg.payload.get("subtype", "")
            direct_messages["challenger"].append(subtype)
            if subtype == "storage_proof_response":
                proof_received.set()

        challenger_transport.on_message(MSG_DIRECT, challenger_direct_handler)

        challenge_msg = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id="challenger",
            payload={
                "subtype": "storage_challenge",
                "challenge": {
                    "challenge_id": "chal_test_001",
                    "cid": "QmTestContent123",
                    "nonce": "abc123",
                    "difficulty": 32,
                },
                "challenger_id": "challenger",
                "target_provider_id": "provider",
            },
        )
        sent = await challenger_transport.send_to_peer("provider", challenge_msg)
        assert sent is True

        await asyncio.wait_for(proof_received.wait(), timeout=5.0)

        assert "storage_challenge" in direct_messages["provider"]
        assert "storage_proof_response" in direct_messages["challenger"]
        assert len(challenger_gossip.published) == 0
        assert len(provider_gossip.published) == 0


class TestDirectP2PFallback:

    @pytest.mark.asyncio
    async def test_fallback_to_gossip_on_send_failure(self):
        _gossip_registry.clear()
        network = MockNetwork()
        challenger_transport, challenger_gossip = make_test_node("challenger", network)
        provider_transport, provider_gossip = make_test_node("provider", network)

        challenger_transport.send_to_peer = AsyncMock(return_value=False)

        gossip_received = asyncio.Event()
        received_data = {}

        async def on_gossip_challenge(subtype, data, sender):
            received_data.update(data)
            gossip_received.set()

        provider_gossip.subscribe("storage_challenge", on_gossip_challenge)

        challenge_payload = {
            "subtype": "storage_challenge",
            "challenge": {
                "challenge_id": "chal_fallback_001",
                "cid": "QmFallbackTest",
                "nonce": "xyz789",
                "difficulty": 32,
            },
            "challenger_id": "challenger",
            "target_provider_id": "provider",
        }

        msg = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id="challenger",
            payload=challenge_payload,
        )
        sent = await challenger_transport.send_to_peer("provider", msg)
        if not sent:
            await challenger_gossip.publish("storage_challenge", challenge_payload)

        await asyncio.wait_for(gossip_received.wait(), timeout=5.0)

        assert received_data["challenge"]["challenge_id"] == "chal_fallback_001"
        assert len(challenger_gossip.published) == 1
