"""
Unit tests for gossip persistence and late-joining node catch-up.

Tests the digest exchange mechanism that allows nodes to catch up on
missed gossip messages when they join the network after messages were
originally broadcast.
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from prsm.node.gossip import (
    GossipProtocol,
    GOSSIP_DIGEST_REQUEST,
    GOSSIP_DIGEST_RESPONSE,
    GOSSIP_JOB_OFFER,
    GOSSIP_CONTENT_ADVERTISE,
    GOSSIP_AGENT_ADVERTISE,
    GOSSIP_RETENTION_SECONDS,
)
from prsm.node.transport import P2PMessage, PeerConnection, MSG_GOSSIP


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_transport():
    """Create a mock WebSocket transport."""
    transport = MagicMock()
    transport.identity = MagicMock()
    transport.identity.node_id = "test_node_123"
    transport.peer_count = 0
    transport.on_message = MagicMock()
    transport.send_to_peer = AsyncMock(return_value=True)
    transport.gossip = AsyncMock(return_value=1)
    return transport


@pytest.fixture
def mock_ledger():
    """Create a mock ledger with gossip log storage."""
    ledger = MagicMock()
    ledger.log_gossip = AsyncMock()
    ledger.get_recent_gossip = AsyncMock(return_value=[])
    # sp1442 — the catch-up dedup now prefers the O(1) gossip_nonce_exists seek; default "not a
    # duplicate" so a fresh catch-up message is delivered (tests that simulate a dup set it True).
    ledger.gossip_nonce_exists = AsyncMock(return_value=False)
    ledger.prune_gossip_log = AsyncMock(return_value=0)
    return ledger


@pytest.fixture
def gossip_protocol(mock_transport, mock_ledger):
    """Create a gossip protocol instance with mocked dependencies."""
    gossip = GossipProtocol(
        transport=mock_transport,
        fanout=3,
        default_ttl=5,
        heartbeat_interval=30.0,
    )
    gossip.ledger = mock_ledger
    return gossip


@pytest.fixture
def mock_peer():
    """Create a mock peer connection."""
    return PeerConnection(
        peer_id="remote_node_456",
        address="ws://localhost:9002",
        websocket=MagicMock(),
        public_key_b64="test_public_key",
        display_name="Test Peer",
    )


# =============================================================================
# Test GOSSIP_RETENTION_SECONDS Configuration
# =============================================================================

class TestGossipRetentionConfig:
    """Tests for gossip retention configuration."""

    def test_retention_config_exists(self):
        """Verify retention configuration exists for key subtypes."""
        assert GOSSIP_JOB_OFFER in GOSSIP_RETENTION_SECONDS
        assert GOSSIP_CONTENT_ADVERTISE in GOSSIP_RETENTION_SECONDS
        assert GOSSIP_AGENT_ADVERTISE in GOSSIP_RETENTION_SECONDS

    def test_task_retention_is_one_hour(self):
        """Task-related messages should have 1 hour (3600s) retention."""
        assert GOSSIP_RETENTION_SECONDS.get("job_offer") == 3600
        assert GOSSIP_RETENTION_SECONDS.get("agent_task_offer") == 3600

    def test_content_retention_is_24_hours(self):
        """Content-related messages should have 24 hour (86400s) retention."""
        assert GOSSIP_RETENTION_SECONDS.get("content_advertise") == 86400
        assert GOSSIP_RETENTION_SECONDS.get("agent_advertise") == 86400

    def test_digest_exchange_not_stored(self):
        """Digest exchange messages should have 0 retention (not stored)."""
        assert GOSSIP_RETENTION_SECONDS.get("digest_request") == 0
        assert GOSSIP_RETENTION_SECONDS.get("digest_response") == 0


# =============================================================================
# Test Digest Request
# =============================================================================

class TestDigestRequest:
    """Tests for digest request functionality."""

    @pytest.mark.asyncio
    async def test_request_digest_sends_message(self, gossip_protocol, mock_transport):
        """Verify request_digest sends a properly formatted message."""
        # Set up the ledger to return some timestamps
        gossip_protocol.ledger.get_recent_gossip = AsyncMock(return_value=[
            {"nonce": "abc123", "subtype": "job_offer", "received_at": time.time() - 100}
        ])

        await gossip_protocol.request_digest("remote_node_456")

        # Verify send_to_peer was called
        mock_transport.send_to_peer.assert_called_once()
        call_args = mock_transport.send_to_peer.call_args
        
        # Verify the message was sent to the correct peer
        assert call_args[0][0] == "remote_node_456"
        
        # Verify the message is a gossip message with digest_request subtype
        msg = call_args[0][1]
        assert msg.msg_type == MSG_GOSSIP
        assert msg.payload["subtype"] == GOSSIP_DIGEST_REQUEST
        assert "timestamps" in msg.payload["data"]
        assert "requester_id" in msg.payload["data"]

    @pytest.mark.asyncio
    async def test_get_last_seen_timestamps(self, gossip_protocol):
        """Test getting last-seen timestamps from the ledger."""
        current_time = time.time()
        
        # Mock the ledger to return some messages
        # Note: _get_last_seen_timestamps queries for each subtype separately
        # and returns the max timestamp for each
        call_count = [0]
        
        async def mock_get_recent_gossip(since, subtypes=None):
            call_count[0] += 1
            if subtypes and subtypes[0] == "job_offer":
                return [
                    {"nonce": "msg1", "subtype": "job_offer", "received_at": current_time - 100},
                    {"nonce": "msg2", "subtype": "job_offer", "received_at": current_time - 50},
                ]
            elif subtypes and subtypes[0] == "content_advertise":
                return [
                    {"nonce": "msg3", "subtype": "content_advertise", "received_at": current_time - 200},
                ]
            return []
        
        gossip_protocol.ledger.get_recent_gossip = mock_get_recent_gossip

        timestamps = await gossip_protocol._get_last_seen_timestamps()

        # Should return the most recent timestamp for each subtype
        assert "job_offer" in timestamps
        assert timestamps["job_offer"] == current_time - 50
        assert "content_advertise" in timestamps
        assert timestamps["content_advertise"] == current_time - 200

    @pytest.mark.asyncio
    async def test_get_last_seen_timestamps_empty_ledger(self, gossip_protocol):
        """Test getting timestamps when ledger is empty."""
        gossip_protocol.ledger.get_recent_gossip = AsyncMock(return_value=[])

        timestamps = await gossip_protocol._get_last_seen_timestamps()

        # Should return empty dict when no messages
        assert timestamps == {}


# =============================================================================
# Test Digest Response Handling
# =============================================================================

class TestDigestResponse:
    """Tests for digest response handling."""

    @pytest.mark.asyncio
    async def test_handle_digest_request_responds_with_messages(self, gossip_protocol, mock_peer, mock_transport):
        """Test that digest request is answered with relevant messages."""
        current_time = time.time()
        
        # Mock ledger to return messages for specific subtypes
        async def mock_get_recent_gossip(since, subtypes=None):
            if subtypes is None:
                return []
            if subtypes[0] == "job_offer":
                return [
                    {
                        "nonce": "msg1",
                        "subtype": "job_offer",
                        "origin": "node_a",
                        "payload": {"job_id": "job_123"},
                        "received_at": current_time - 100,
                    }
                ]
            elif subtypes[0] == "content_advertise":
                return [
                    {
                        "nonce": "msg2",
                        "subtype": "content_advertise",
                        "origin": "node_b",
                        "payload": {"cid": "QmTest"},
                        "received_at": current_time - 50,
                    }
                ]
            # For catchup_subtypes that weren't in timestamps
            elif subtypes[0] in ("agent_advertise", "provenance_register"):
                return []
            return []
        
        gossip_protocol.ledger.get_recent_gossip = mock_get_recent_gossip

        # Create a digest request message
        msg = P2PMessage(
            msg_type=MSG_GOSSIP,
            sender_id="remote_node_456",
            payload={
                "subtype": GOSSIP_DIGEST_REQUEST,
                "data": {
                    "timestamps": {"job_offer": current_time - 200},
                    "requester_id": "remote_node_456",
                },
            },
        )

        await gossip_protocol._handle_digest_request(msg, mock_peer)

        # Verify response was sent
        mock_transport.send_to_peer.assert_called_once()
        call_args = mock_transport.send_to_peer.call_args
        
        # Verify the response contains the messages
        response_msg = call_args[0][1]
        assert response_msg.payload["subtype"] == GOSSIP_DIGEST_RESPONSE
        assert "messages" in response_msg.payload["data"]
        # Should have at least the job_offer message
        assert response_msg.payload["data"]["total_count"] >= 1

    @pytest.mark.asyncio
    async def test_handle_digest_request_includes_new_subtypes(self, gossip_protocol, mock_peer, mock_transport):
        """Test that digest request includes messages for subtypes the requester didn't mention."""
        current_time = time.time()
        
        # Mock ledger to return messages
        gossip_protocol.ledger.get_recent_gossip = AsyncMock(return_value=[
            {
                "nonce": "msg1",
                "subtype": "agent_advertise",
                "origin": "node_a",
                "payload": {"agent_id": "agent_123"},
                "received_at": current_time - 100,
            },
        ])

        # Create a digest request with no timestamps (new node)
        msg = P2PMessage(
            msg_type=MSG_GOSSIP,
            sender_id="remote_node_456",
            payload={
                "subtype": GOSSIP_DIGEST_REQUEST,
                "data": {
                    "timestamps": {},  # Empty - new node
                    "requester_id": "remote_node_456",
                },
            },
        )

        await gossip_protocol._handle_digest_request(msg, mock_peer)

        # Verify response was sent
        mock_transport.send_to_peer.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_digest_response_processes_messages(self, gossip_protocol, mock_peer):
        """Test that digest response messages are processed correctly."""
        current_time = time.time()
        
        # Track subscriber callbacks
        received_messages = []
        
        async def capture_message(subtype, data, origin):
            received_messages.append((subtype, data, origin))
        
        gossip_protocol.subscribe("job_offer", capture_message)
        gossip_protocol.subscribe("content_advertise", capture_message)

        # Create a digest response message
        msg = P2PMessage(
            msg_type=MSG_GOSSIP,
            sender_id="remote_node_456",
            payload={
                "subtype": GOSSIP_DIGEST_RESPONSE,
                "data": {
                    "messages": [
                        {
                            "nonce": "msg1",
                            "subtype": "job_offer",
                            "origin": "node_a",
                            "payload": {"job_id": "job_123"},
                            "received_at": current_time - 100,
                        },
                        {
                            "nonce": "msg2",
                            "subtype": "content_advertise",
                            "origin": "node_b",
                            "payload": {"cid": "QmTest"},
                            "received_at": current_time - 50,
                        },
                    ],
                    "total_count": 2,
                },
            },
        )

        # sp1442 — a digest_response is only processed if we solicited it (sent this peer a
        # digest_request). Simulate that outbound request so the SOLICITED path under test runs.
        gossip_protocol._pending_digest[mock_peer.peer_id] = time.time() + 3600
        await gossip_protocol._handle_digest_response(msg, mock_peer)

        # Verify messages were delivered to subscribers
        assert len(received_messages) == 2
        assert received_messages[0][0] == "job_offer"
        assert received_messages[1][0] == "content_advertise"

    @pytest.mark.asyncio
    async def test_handle_digest_response_stores_in_ledger(self, gossip_protocol, mock_peer):
        """Test that digest response messages are stored in the local ledger."""
        current_time = time.time()
        
        # Create a digest response message
        msg = P2PMessage(
            msg_type=MSG_GOSSIP,
            sender_id="remote_node_456",
            payload={
                "subtype": GOSSIP_DIGEST_RESPONSE,
                "data": {
                    "messages": [
                        {
                            "nonce": "msg1",
                            "subtype": "job_offer",
                            "origin": "node_a",
                            "payload": {"job_id": "job_123"},
                            "received_at": current_time - 100,
                        },
                    ],
                    "total_count": 1,
                },
            },
        )

        gossip_protocol._pending_digest[mock_peer.peer_id] = time.time() + 3600  # sp1442 — solicit
        await gossip_protocol._handle_digest_response(msg, mock_peer)

        # Verify message was stored in ledger
        gossip_protocol.ledger.log_gossip.assert_called_once()
        call_args = gossip_protocol.ledger.log_gossip.call_args
        assert call_args[1]["subtype"] == "job_offer"
        assert call_args[1]["nonce"] == "msg1"

    @pytest.mark.asyncio
    async def test_handle_digest_response_skips_duplicates(self, gossip_protocol, mock_peer):
        """Test that duplicate messages in digest response are skipped."""
        current_time = time.time()
        
        # sp1442 — _is_duplicate now uses the O(1) gossip_nonce_exists seek; simulate the nonce
        # already being in the log (a duplicate to skip).
        gossip_protocol.ledger.gossip_nonce_exists = AsyncMock(return_value=True)

        # Track subscriber callbacks
        received_messages = []
        
        async def capture_message(subtype, data, origin):
            received_messages.append((subtype, data, origin))
        
        gossip_protocol.subscribe("job_offer", capture_message)

        # Create a digest response with a duplicate message
        msg = P2PMessage(
            msg_type=MSG_GOSSIP,
            sender_id="remote_node_456",
            payload={
                "subtype": GOSSIP_DIGEST_RESPONSE,
                "data": {
                    "messages": [
                        {
                            "nonce": "msg1",  # Duplicate
                            "subtype": "job_offer",
                            "origin": "node_a",
                            "payload": {"job_id": "job_123"},
                            "received_at": current_time - 100,
                        },
                    ],
                    "total_count": 1,
                },
            },
        )

        gossip_protocol._pending_digest[mock_peer.peer_id] = time.time() + 3600  # sp1442 — solicit
        await gossip_protocol._handle_digest_response(msg, mock_peer)

        # Verify duplicate was skipped
        assert len(received_messages) == 0


# =============================================================================
# Test Peer Connection Handler
# =============================================================================

class TestPeerConnectionHandler:
    """Tests for peer connection event handling."""

    @pytest.mark.asyncio
    async def test_on_peer_connected_outbound_requests_digest(self, gossip_protocol, mock_transport):
        """Test that outbound peer connections trigger digest request."""
        # Create a peer_connected message for outbound connection
        msg = P2PMessage(
            msg_type="peer_connected",
            sender_id="remote_node_456",
            payload={"direction": "outbound"},
        )
        mock_peer = PeerConnection(
            peer_id="remote_node_456",
            address="ws://localhost:9002",
            websocket=MagicMock(),
        )

        # Mock request_digest to track calls
        gossip_protocol.request_digest = AsyncMock()

        await gossip_protocol._on_peer_connected(msg, mock_peer)

        # Verify digest request was sent
        gossip_protocol.request_digest.assert_called_once_with("remote_node_456")

    @pytest.mark.asyncio
    async def test_on_peer_connected_inbound_no_digest(self, gossip_protocol, mock_transport):
        """Test that inbound peer connections do not trigger digest request."""
        # Create a peer_connected message for inbound connection
        msg = P2PMessage(
            msg_type="peer_connected",
            sender_id="remote_node_456",
            payload={"direction": "inbound"},
        )
        mock_peer = PeerConnection(
            peer_id="remote_node_456",
            address="ws://localhost:9002",
            websocket=MagicMock(),
        )

        # Mock request_digest to track calls
        gossip_protocol.request_digest = AsyncMock()

        await gossip_protocol._on_peer_connected(msg, mock_peer)

        # Verify digest request was NOT sent
        gossip_protocol.request_digest.assert_not_called()


# =============================================================================
# Test Gossip Log Cleanup
# =============================================================================

class TestGossipLogCleanup:
    """Tests for gossip log cleanup functionality."""

    @pytest.mark.asyncio
    async def test_prune_gossip_log_calls_ledger(self, gossip_protocol):
        """Test that prune_gossip_log_by_retention calls the ledger."""
        gossip_protocol.ledger.prune_gossip_log = AsyncMock(return_value=5)

        pruned = await gossip_protocol._prune_gossip_log_by_retention()

        # Verify ledger prune was called
        gossip_protocol.ledger.prune_gossip_log.assert_called_once()
        assert pruned == 5

    @pytest.mark.asyncio
    async def test_prune_gossip_log_no_ledger(self, gossip_protocol):
        """Test that prune handles missing ledger gracefully."""
        gossip_protocol.ledger = None

        pruned = await gossip_protocol._prune_gossip_log_by_retention()

        # Should return 0 when no ledger
        assert pruned == 0


# =============================================================================
# Test Integration Scenarios
# =============================================================================

class TestIntegrationScenarios:
    """Integration tests for gossip persistence."""

    @pytest.mark.asyncio
    async def test_late_joining_node_catches_up(self, gossip_protocol, mock_transport, mock_peer):
        """Test complete flow: late-joining node receives missed messages."""
        current_time = time.time()
        
        # Track received catch-up messages
        catchup_messages = []
        
        async def capture_catchup(subtype, data, origin):
            catchup_messages.append((subtype, data, origin))
        
        gossip_protocol.subscribe("job_offer", capture_catchup)
        gossip_protocol.subscribe("content_advertise", capture_catchup)
        gossip_protocol.subscribe("agent_advertise", capture_catchup)

        # Mock _is_duplicate to return False (no duplicates)
        gossip_protocol._is_duplicate = AsyncMock(return_value=False)

        # Simulate receiving a digest response
        digest_response = P2PMessage(
            msg_type=MSG_GOSSIP,
            sender_id="remote_node_456",
            payload={
                "subtype": GOSSIP_DIGEST_RESPONSE,
                "data": {
                    "messages": [
                        {
                            "nonce": "msg1",
                            "subtype": "job_offer",
                            "origin": "node_a",
                            "payload": {"job_id": "job_123", "task": "compute"},
                            "received_at": current_time - 3600,
                        },
                        {
                            "nonce": "msg2",
                            "subtype": "content_advertise",
                            "origin": "node_b",
                            "payload": {"cid": "QmTest123", "size": 1024},
                            "received_at": current_time - 1800,
                        },
                        {
                            "nonce": "msg3",
                            "subtype": "agent_advertise",
                            "origin": "node_c",
                            "payload": {"agent_id": "agent_456", "capabilities": ["nlp"]},
                            "received_at": current_time - 900,
                        },
                    ],
                    "total_count": 3,
                },
            },
        )

        # sp1442 — the late-joiner is OUTBOUND (it dialed the peer and sent the digest request),
        # so the response is solicited and processed.
        gossip_protocol._pending_digest[mock_peer.peer_id] = time.time() + 3600
        await gossip_protocol._handle_digest_response(digest_response, mock_peer)

        # Verify all messages were received
        assert len(catchup_messages) == 3
        
        # Verify message types
        subtypes = [m[0] for m in catchup_messages]
        assert "job_offer" in subtypes
        assert "content_advertise" in subtypes
        assert "agent_advertise" in subtypes

    @pytest.mark.asyncio
    async def test_digest_exchange_does_not_propagate(self, gossip_protocol, mock_transport, mock_peer):
        """Test that digest exchange messages are not re-propagated."""
        # Create a digest request
        digest_request = P2PMessage(
            msg_type=MSG_GOSSIP,
            sender_id="remote_node_456",
            payload={
                "subtype": GOSSIP_DIGEST_REQUEST,
                "data": {
                    "timestamps": {},
                    "requester_id": "remote_node_456",
                },
            },
            ttl=5,
        )

        # Handle the digest request
        gossip_protocol.ledger.get_recent_gossip = AsyncMock(return_value=[])
        await gossip_protocol._handle_gossip(digest_request, mock_peer)

        # Verify the message was NOT re-propagated (gossip not called)
        mock_transport.gossip.assert_not_called()

        # Create a digest response
        digest_response = P2PMessage(
            msg_type=MSG_GOSSIP,
            sender_id="remote_node_456",
            payload={
                "subtype": GOSSIP_DIGEST_RESPONSE,
                "data": {
                    "messages": [],
                    "total_count": 0,
                },
            },
            ttl=5,
        )

        # Reset mock
        mock_transport.gossip.reset_mock()

        # Handle the digest response
        await gossip_protocol._handle_gossip(digest_response, mock_peer)

        # Verify the message was NOT re-propagated
        mock_transport.gossip.assert_not_called()


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_handle_digest_request_no_ledger(self, gossip_protocol, mock_peer, mock_transport):
        """Test digest request handling when ledger is not available."""
        gossip_protocol.ledger = None

        msg = P2PMessage(
            msg_type=MSG_GOSSIP,
            sender_id="remote_node_456",
            payload={
                "subtype": GOSSIP_DIGEST_REQUEST,
                "data": {
                    "timestamps": {},
                    "requester_id": "remote_node_456",
                },
            },
        )

        # Should not raise, just return early
        await gossip_protocol._handle_digest_request(msg, mock_peer)

        # Verify no response was sent
        mock_transport.send_to_peer.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_digest_response_empty_messages(self, gossip_protocol, mock_peer):
        """Test digest response with empty message list."""
        msg = P2PMessage(
            msg_type=MSG_GOSSIP,
            sender_id="remote_node_456",
            payload={
                "subtype": GOSSIP_DIGEST_RESPONSE,
                "data": {
                    "messages": [],
                    "total_count": 0,
                },
            },
        )

        # Should handle gracefully
        await gossip_protocol._handle_digest_response(msg, mock_peer)

        # No messages should be processed
        # (no exception raised is the test)

    @pytest.mark.asyncio
    async def test_handle_digest_response_malformed_message(self, gossip_protocol, mock_peer):
        """Test digest response with malformed messages."""
        msg = P2PMessage(
            msg_type=MSG_GOSSIP,
            sender_id="remote_node_456",
            payload={
                "subtype": GOSSIP_DIGEST_RESPONSE,
                "data": {
                    "messages": [
                        {"nonce": "msg1"},  # Missing subtype and payload
                        {"subtype": "job_offer"},  # Missing payload
                    ],
                    "total_count": 2,
                },
            },
        )

        # Should handle gracefully without crashing
        await gossip_protocol._handle_digest_response(msg, mock_peer)

    @pytest.mark.asyncio
    async def test_digest_response_limits_message_count(self, gossip_protocol, mock_peer, mock_transport):
        """Test that digest response limits to 100 messages."""
        current_time = time.time()
        
        # Create 150 mock messages
        mock_messages = [
            {
                "nonce": f"msg{i}",
                "subtype": "job_offer",
                "origin": "node_a",
                "payload": {"job_id": f"job_{i}"},
                "received_at": current_time - i,
            }
            for i in range(150)
        ]
        
        gossip_protocol.ledger.get_recent_gossip = AsyncMock(return_value=mock_messages)

        msg = P2PMessage(
            msg_type=MSG_GOSSIP,
            sender_id="remote_node_456",
            payload={
                "subtype": GOSSIP_DIGEST_REQUEST,
                "data": {
                    "timestamps": {},
                    "requester_id": "remote_node_456",
                },
            },
        )

        await gossip_protocol._handle_digest_request(msg, mock_peer)

        # Verify response was limited to 100 messages
        call_args = mock_transport.send_to_peer.call_args
        response_msg = call_args[0][1]
        assert response_msg.payload["data"]["total_count"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])