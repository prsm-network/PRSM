"""Sprint 700 F46 fix — monotonic hardware_profile gossip propagation.

Live-attest of sprint 698 surfaced F46: when Lambda's A10 specs reached
NYC via direct DISCOVERY_ANNOUNCE, NYC correctly stored the profile.
But subsequent gossip from SFO (which hadn't yet received Lambda's
announce) carried a peer-list entry where Lambda's hardware_profile
was None. _handle_peer_response unconditionally REPLACED NYC's good
profile data with the stale None.

Sprint 700 fix: gossip from a non-authoritative source can ADD profile
data but cannot REMOVE it. The semantics is: gossip is monotonic-
improvement. If incoming peer-list entry has hardware_profile=None
and we ALREADY have one for this peer, preserve the existing value.

_handle_announce is unchanged: the announcer IS authoritative for its
own profile, so replacement (including with None) is semantically
correct there.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_peer_response_preserves_existing_hardware_profile_when_incoming_none():
    """NYC has Lambda's A10 profile (from direct announce). SFO gossips
    a peer-list entry for Lambda with hardware_profile=None (SFO hadn't
    received Lambda's announce yet). NYC must preserve its existing
    A10 profile, NOT overwrite with None."""
    from prsm.node.discovery import (
        PeerDiscovery, DISCOVERY_PEER_RESPONSE, PeerInfo,
    )
    from prsm.node.transport import P2PMessage, MSG_GOSSIP
    transport = MagicMock()
    transport.identity = MagicMock()
    transport.identity.node_id = "nyc-self"
    transport.port = 9001
    pd = PeerDiscovery(transport=transport)
    # Pre-populate Lambda's PeerInfo with A10 profile (as if from direct announce)
    a10_profile = {"tflops_fp16": 33.9, "gpu_vram_gb": 22.5, "gpu_name": "NVIDIA A10"}
    pd.known_peers["lambda"] = PeerInfo(
        node_id="lambda", address="146.235.193.143:9001",
        hardware_profile=a10_profile,
    )
    # Now SFO gossips a peer-list including Lambda with NO profile
    msg = P2PMessage(
        msg_type=MSG_GOSSIP,
        sender_id="sfo",
        payload={
            "subtype": DISCOVERY_PEER_RESPONSE,
            "peers": [{
                "node_id": "lambda",
                "address": "146.235.193.143:9001",
                "capabilities": ["compute"],
                # hardware_profile omitted (or None) — SFO doesn't have it yet
            }],
        },
        ttl=1,
    )
    peer = MagicMock()
    peer.peer_id = "sfo"
    await pd._handle_peer_response(msg, peer)
    # NYC must STILL have Lambda's A10 profile
    assert pd.known_peers["lambda"].hardware_profile == a10_profile, (
        "F46 fix: gossip from a non-authoritative peer must NOT "
        "downgrade hardware_profile to None when we already have one"
    )


@pytest.mark.asyncio
async def test_peer_response_accepts_hardware_profile_when_we_have_none():
    """We have no profile for Lambda yet; SFO gossips a peer-list with
    Lambda's A10 profile. We must ADOPT it (monotonic add)."""
    from prsm.node.discovery import (
        PeerDiscovery, DISCOVERY_PEER_RESPONSE, PeerInfo,
    )
    from prsm.node.transport import P2PMessage, MSG_GOSSIP
    transport = MagicMock()
    transport.identity = MagicMock()
    transport.identity.node_id = "nyc-self"
    transport.port = 9001
    pd = PeerDiscovery(transport=transport)
    # Pre-populate with NO profile
    pd.known_peers["lambda"] = PeerInfo(
        node_id="lambda", address="146.235.193.143:9001",
        hardware_profile=None,
    )
    a10_profile = {"tflops_fp16": 33.9, "gpu_vram_gb": 22.5}
    msg = P2PMessage(
        msg_type=MSG_GOSSIP,
        sender_id="sfo",
        payload={
            "subtype": DISCOVERY_PEER_RESPONSE,
            "peers": [{
                "node_id": "lambda",
                "address": "146.235.193.143:9001",
                "hardware_profile": a10_profile,
            }],
        },
        ttl=1,
    )
    peer = MagicMock()
    peer.peer_id = "sfo"
    await pd._handle_peer_response(msg, peer)
    assert pd.known_peers["lambda"].hardware_profile == a10_profile


@pytest.mark.asyncio
async def test_peer_response_replaces_when_both_present():
    """Both old and new profiles are non-None: replace with new (latest
    write wins). Avoids stuck-at-stale if a peer's hardware changes."""
    from prsm.node.discovery import (
        PeerDiscovery, DISCOVERY_PEER_RESPONSE, PeerInfo,
    )
    from prsm.node.transport import P2PMessage, MSG_GOSSIP
    transport = MagicMock()
    transport.identity = MagicMock()
    transport.identity.node_id = "nyc-self"
    transport.port = 9001
    pd = PeerDiscovery(transport=transport)
    old_profile = {"tflops_fp16": 33.9, "gpu_name": "NVIDIA A10"}
    pd.known_peers["lambda"] = PeerInfo(
        node_id="lambda", address="x:1", hardware_profile=old_profile,
    )
    new_profile = {"tflops_fp16": 67.0, "gpu_name": "NVIDIA H100"}
    msg = P2PMessage(
        msg_type=MSG_GOSSIP,
        sender_id="sfo",
        payload={
            "subtype": DISCOVERY_PEER_RESPONSE,
            "peers": [{
                "node_id": "lambda",
                "address": "x:1",
                "hardware_profile": new_profile,
            }],
        },
        ttl=1,
    )
    peer = MagicMock()
    peer.peer_id = "sfo"
    await pd._handle_peer_response(msg, peer)
    assert pd.known_peers["lambda"].hardware_profile == new_profile


@pytest.mark.asyncio
async def test_peer_response_new_peer_with_none_stays_none():
    """Brand-new peer in gossip with no profile: PeerInfo gets None.
    Pre-fix behavior preserved for peers we've never heard of."""
    from prsm.node.discovery import (
        PeerDiscovery, DISCOVERY_PEER_RESPONSE,
    )
    from prsm.node.transport import P2PMessage, MSG_GOSSIP
    transport = MagicMock()
    transport.identity = MagicMock()
    transport.identity.node_id = "nyc-self"
    transport.port = 9001
    pd = PeerDiscovery(transport=transport)
    # known_peers does NOT contain "newpeer"
    msg = P2PMessage(
        msg_type=MSG_GOSSIP,
        sender_id="sfo",
        payload={
            "subtype": DISCOVERY_PEER_RESPONSE,
            "peers": [{"node_id": "newpeer", "address": "x:1"}],
        },
        ttl=1,
    )
    peer = MagicMock()
    peer.peer_id = "sfo"
    await pd._handle_peer_response(msg, peer)
    assert pd.known_peers["newpeer"].hardware_profile is None


@pytest.mark.asyncio
async def test_handle_announce_replaces_authoritatively():
    """_handle_announce is unchanged: the announcer IS authoritative
    for its own profile. Replacement is correct semantics there.
    Pin this so the F46 fix doesn't accidentally apply monotonic
    semantics to the announce path."""
    from prsm.node.discovery import (
        PeerDiscovery, DISCOVERY_ANNOUNCE, PeerInfo,
    )
    from prsm.node.transport import P2PMessage, MSG_GOSSIP
    transport = MagicMock()
    transport.identity = MagicMock()
    transport.identity.node_id = "nyc-self"
    transport.port = 9001
    pd = PeerDiscovery(transport=transport)
    # Pre-populate with an OLD profile
    pd.known_peers["lambda"] = PeerInfo(
        node_id="lambda", address="x:1",
        hardware_profile={"tflops_fp16": 100.0, "gpu_name": "OLD"},
    )
    # New announce comes in with a NEW profile (different specs)
    new_profile = {"tflops_fp16": 33.9, "gpu_name": "NVIDIA A10"}
    msg = P2PMessage(
        msg_type=MSG_GOSSIP,
        sender_id="lambda",
        payload={
            "subtype": DISCOVERY_ANNOUNCE,
            "address": "146.235.193.143:9001",
            "hardware_profile": new_profile,
        },
        ttl=1,
    )
    p = MagicMock()
    p.address = "146.235.193.143:9001"
    p.peer_id = "lambda"   # sp937 — direct announce from the handshake-authenticated peer
    await pd._handle_announce(msg, p)
    # Authoritative announce REPLACES (latest write wins)
    assert pd.known_peers["lambda"].hardware_profile == new_profile
