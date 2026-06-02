"""Sprint 680 — discovery announcement carries a hardware_profile.

The DHT-backed GpuPoolProvider (sprint 681+) needs peer-advertised
hardware specs to construct ParallaxGPU entries. PRSM_PARALLAX_*
expects tflops_fp16 / memory_gb / memory_bandwidth_gbps / region /
gpu_name / device — none of which the current PeerInfo carries.

Sprint 680 plumbs an optional `hardware_profile` dict through:

  1. PeerInfo gains `hardware_profile: Optional[Dict[str, Any]] = None`
  2. PeerDiscovery.__init__ takes `local_hardware_profile`
  3. announce_self() includes it in the gossip payload
  4. _handle_announce parses it into the PeerInfo
  5. _handle_peer_request / _handle_peer_response propagate it

This sprint does NOT yet build the DHT-backed pool provider — that's
sprint 681. This sprint only opens the schema-pass-through path.
Pure-additive: no existing announcement-consuming peer breaks if
`hardware_profile` is absent (default None preserved everywhere).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_peer_info_carries_optional_hardware_profile():
    """The PeerInfo dataclass exposes a hardware_profile field that
    defaults to None — every existing construction site keeps its
    current shape."""
    from prsm.node.discovery import PeerInfo
    p = PeerInfo(node_id="abc", address="1.2.3.4:9001")
    assert p.hardware_profile is None
    p2 = PeerInfo(
        node_id="def",
        address="5.6.7.8:9001",
        hardware_profile={"tflops_fp16": 1.5, "memory_gb": 16.0},
    )
    assert p2.hardware_profile == {"tflops_fp16": 1.5, "memory_gb": 16.0}


def test_peer_discovery_accepts_local_hardware_profile():
    """PeerDiscovery.__init__ accepts a local_hardware_profile kwarg
    that gets stored for later announcement."""
    from prsm.node.discovery import PeerDiscovery
    transport = MagicMock()
    transport.identity = MagicMock()
    transport.identity.node_id = "myid"
    pd = PeerDiscovery(
        transport=transport,
        local_hardware_profile={"tflops_fp16": 4.6, "memory_gb": 16.0},
    )
    assert pd._local_hardware_profile == {
        "tflops_fp16": 4.6, "memory_gb": 16.0,
    }


def test_peer_discovery_defaults_local_hardware_profile_to_none():
    """No kwarg supplied → field is None. Pre-680 callers keep working."""
    from prsm.node.discovery import PeerDiscovery
    transport = MagicMock()
    transport.identity = MagicMock()
    transport.identity.node_id = "myid"
    pd = PeerDiscovery(transport=transport)
    assert pd._local_hardware_profile is None


@pytest.mark.asyncio
async def test_announce_self_includes_hardware_profile_when_set():
    """announce_self() payload carries hardware_profile under a
    dedicated key when the local field is populated."""
    from prsm.node.discovery import PeerDiscovery
    transport = MagicMock()
    transport.identity = MagicMock()
    transport.identity.node_id = "myid"
    transport.identity.display_name = "mynode"
    transport.port = 9001
    transport.peer_count = 0
    transport.gossip = AsyncMock(return_value=2)
    profile = {
        "tflops_fp16": 4.6, "memory_gb": 16.0,
        "memory_bandwidth_gbps": 273.0, "gpu_name": "Apple M4",
        "device": "mps", "region": "home",
    }
    pd = PeerDiscovery(
        transport=transport, local_hardware_profile=profile,
    )
    await pd.announce_self()
    sent_msg = transport.gossip.call_args[0][0]
    assert sent_msg.payload.get("hardware_profile") == profile


@pytest.mark.asyncio
async def test_announce_self_omits_hardware_profile_when_none():
    """When local_hardware_profile is None, no hardware_profile key
    leaks into the payload — keeps wire format clean for legacy
    peers that don't inspect this field yet."""
    from prsm.node.discovery import PeerDiscovery
    transport = MagicMock()
    transport.identity = MagicMock()
    transport.identity.node_id = "myid"
    transport.identity.display_name = "mynode"
    transport.port = 9001
    transport.peer_count = 0
    transport.gossip = AsyncMock(return_value=2)
    pd = PeerDiscovery(transport=transport)
    await pd.announce_self()
    sent_msg = transport.gossip.call_args[0][0]
    assert "hardware_profile" not in sent_msg.payload


@pytest.mark.asyncio
async def test_handle_announce_parses_hardware_profile_into_peer_info():
    """Incoming DISCOVERY_ANNOUNCE with hardware_profile sets the
    field on the freshly-stored PeerInfo."""
    from prsm.node.discovery import (
        PeerDiscovery, DISCOVERY_ANNOUNCE,
    )
    from prsm.node.transport import P2PMessage, MSG_GOSSIP
    transport = MagicMock()
    transport.identity = MagicMock()
    transport.identity.node_id = "myid"
    transport.port = 9001
    pd = PeerDiscovery(transport=transport)
    profile = {"tflops_fp16": 9.0, "memory_gb": 24.0, "region": "us-east-1"}
    msg = P2PMessage(
        msg_type=MSG_GOSSIP,
        sender_id="peerA",
        payload={
            "subtype": DISCOVERY_ANNOUNCE,
            "address": "10.0.0.5:9001",
            "capabilities": ["inference"],
            "hardware_profile": profile,
        },
        ttl=1,
    )
    peer = MagicMock()
    peer.address = "10.0.0.5:9001"
    peer.peer_id = "peerA"   # sp937 — direct announce from the handshake-authenticated peer
    await pd._handle_announce(msg, peer)
    info = pd.known_peers["peerA"]
    assert info.hardware_profile == profile


@pytest.mark.asyncio
async def test_handle_announce_tolerates_missing_hardware_profile():
    """Pre-680 peers send announcements without hardware_profile —
    receiver stores PeerInfo with hardware_profile=None, no
    KeyError."""
    from prsm.node.discovery import (
        PeerDiscovery, DISCOVERY_ANNOUNCE,
    )
    from prsm.node.transport import P2PMessage, MSG_GOSSIP
    transport = MagicMock()
    transport.identity = MagicMock()
    transport.identity.node_id = "myid"
    transport.port = 9001
    pd = PeerDiscovery(transport=transport)
    msg = P2PMessage(
        msg_type=MSG_GOSSIP,
        sender_id="peerLegacy",
        payload={
            "subtype": DISCOVERY_ANNOUNCE,
            "address": "10.0.0.6:9001",
            "capabilities": ["inference"],
        },
        ttl=1,
    )
    peer = MagicMock()
    peer.address = "10.0.0.6:9001"
    peer.peer_id = "peerLegacy"   # sp937 — direct announce from the handshake-authenticated peer
    await pd._handle_announce(msg, peer)
    assert pd.known_peers["peerLegacy"].hardware_profile is None


@pytest.mark.asyncio
async def test_handle_peer_response_propagates_hardware_profile():
    """When a peer-list response from another node carries
    hardware_profile for each peer, the receiver stores those
    profiles on its own PeerInfo entries — DHT propagation works."""
    from prsm.node.discovery import (
        PeerDiscovery, DISCOVERY_PEER_RESPONSE,
    )
    from prsm.node.transport import P2PMessage, MSG_GOSSIP
    transport = MagicMock()
    transport.identity = MagicMock()
    transport.identity.node_id = "myid"
    transport.port = 9001
    pd = PeerDiscovery(transport=transport)
    msg = P2PMessage(
        msg_type=MSG_GOSSIP,
        sender_id="gossipper",
        payload={
            "subtype": DISCOVERY_PEER_RESPONSE,
            "peers": [{
                "node_id": "remoteA",
                "address": "11.1.1.1:9001",
                "capabilities": ["inference"],
                "hardware_profile": {"tflops_fp16": 7.5, "memory_gb": 12.0},
            }],
        },
        ttl=1,
    )
    peer = MagicMock()
    peer.peer_id = "gossipper"
    await pd._handle_peer_response(msg, peer)
    assert pd.known_peers["remoteA"].hardware_profile == {
        "tflops_fp16": 7.5, "memory_gb": 12.0,
    }


@pytest.mark.asyncio
async def test_handle_peer_request_echoes_known_peer_hardware_profiles():
    """When asked for our peer list, we include each known peer's
    hardware_profile in the response — multi-hop DHT propagation."""
    from prsm.node.discovery import (
        PeerDiscovery, DISCOVERY_PEER_RESPONSE, PeerInfo,
    )
    from prsm.node.transport import P2PMessage, MSG_GOSSIP
    transport = MagicMock()
    transport.identity = MagicMock()
    transport.identity.node_id = "myid"
    transport.port = 9001
    transport.peers = {}
    transport.send_to_peer = AsyncMock(return_value=True)
    pd = PeerDiscovery(transport=transport)
    pd.known_peers["other"] = PeerInfo(
        node_id="other", address="9.9.9.9:9001",
        hardware_profile={"tflops_fp16": 3.0, "memory_gb": 8.0},
    )
    msg = P2PMessage(
        msg_type=MSG_GOSSIP,
        sender_id="asker",
        payload={"max_peers": 20},
        ttl=1,
    )
    peer = MagicMock()
    peer.peer_id = "asker"
    await pd._handle_peer_request(msg, peer)
    sent = transport.send_to_peer.call_args[0][1]
    peers = sent.payload["peers"]
    other_entry = next(p for p in peers if p["node_id"] == "other")
    assert other_entry["hardware_profile"] == {
        "tflops_fp16": 3.0, "memory_gb": 8.0,
    }
