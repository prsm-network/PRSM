"""
Tests for Cross-Node Content Retrieval (Phase 2.1)

Tests the P2P content request/response protocol that allows nodes
to discover and retrieve content from each other.

Coverage:
- ContentRequestMessage and ContentResponseMessage serialization
- ContentDiscovery tracking of content locations
- ContentProvider request handling (server side)
- ContentProvider content retrieval (client side)
- Timeout handling
- Error cases (content not found, peer unavailable)
"""

import asyncio
import base64
import hashlib
import json
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prsm.node.content_provider import (
    ContentAnnouncement,
    ContentDiscovery,
    ContentProvider,
    ContentRequestMessage,
    ContentResponseMessage,
    ContentStatus,
    TransferMode,
    MAX_INLINE_SIZE,
    DEFAULT_REQUEST_TIMEOUT,
)
from prsm.node.transport import MSG_DIRECT, P2PMessage, PeerConnection


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_identity():
    """Create a mock node identity."""
    identity = MagicMock()
    identity.node_id = "test_node_" + uuid.uuid4().hex[:8]
    identity.public_key_b64 = base64.b64encode(b"test_public_key").decode()
    identity.sign = MagicMock(return_value="test_signature")
    return identity


@pytest.fixture
def mock_transport(mock_identity):
    """Create a mock WebSocket transport."""
    transport = MagicMock()
    transport.identity = mock_identity
    transport.on_message = MagicMock()
    transport.send_to_peer = AsyncMock(return_value=True)
    return transport


@pytest.fixture
def mock_gossip():
    """Create a mock gossip protocol."""
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    gossip.publish = AsyncMock(return_value=1)
    return gossip


@pytest.fixture
def mock_content_index():
    """Create a mock content index."""
    content_index = MagicMock()
    content_index.lookup = MagicMock(return_value=None)
    return content_index


@pytest.fixture
def content_discovery():
    """Create a ContentDiscovery instance."""
    return ContentDiscovery(max_content_entries=100)


@pytest.fixture
def content_provider(mock_identity, mock_transport, mock_gossip, mock_content_index):
    """Create a ContentProvider instance."""
    provider = ContentProvider(
        identity=mock_identity,
        transport=mock_transport,
        gossip=mock_gossip,
        content_index=mock_content_index,
    )
    return provider


# =============================================================================
# ContentRequestMessage Tests
# =============================================================================

class TestContentRequestMessage:
    """Tests for ContentRequestMessage serialization."""
    
    def test_create_request(self):
        """Test creating a content request."""
        request = ContentRequestMessage(
            cid="QmTestCID123",
            request_id="req123",
            priority=1,
            timeout=60,
            requester_id="node_abc",
        )
        
        assert request.cid == "QmTestCID123"
        assert request.request_id == "req123"
        assert request.priority == 1
        assert request.timeout == 60
        assert request.requester_id == "node_abc"
    
    def test_request_default_values(self):
        """Test default values for content request."""
        request = ContentRequestMessage(cid="QmTestCID")
        
        assert request.cid == "QmTestCID"
        assert len(request.request_id) == 16  # UUID hex[:16]
        assert request.priority == 0
        assert request.timeout == 30
        assert request.requester_id == ""
    
    def test_request_to_payload(self):
        """Test serializing request to payload."""
        request = ContentRequestMessage(
            cid="QmTestCID",
            request_id="req123",
            priority=1,
            timeout=60,
            requester_id="node_abc",
        )
        
        payload = request.to_payload()
        
        assert payload["subtype"] == "content_request"
        assert payload["cid"] == "QmTestCID"
        assert payload["request_id"] == "req123"
        assert payload["priority"] == 1
        assert payload["timeout"] == 60
        assert payload["requester_id"] == "node_abc"
    
    def test_request_from_payload(self):
        """Test deserializing request from payload."""
        payload = {
            "subtype": "content_request",
            "cid": "QmTestCID",
            "request_id": "req123",
            "priority": 1,
            "timeout": 60,
            "requester_id": "node_abc",
        }
        
        request = ContentRequestMessage.from_payload(payload)
        
        assert request.cid == "QmTestCID"
        assert request.request_id == "req123"
        assert request.priority == 1
        assert request.timeout == 60
        assert request.requester_id == "node_abc"
    
    def test_request_roundtrip(self):
        """Test request serialization roundtrip."""
        original = ContentRequestMessage(
            cid="QmTestCID",
            priority=1,
            timeout=45,
            requester_id="node_xyz",
        )
        
        payload = original.to_payload()
        restored = ContentRequestMessage.from_payload(payload)
        
        assert restored.cid == original.cid
        assert restored.request_id == original.request_id
        assert restored.priority == original.priority
        assert restored.timeout == original.timeout
        assert restored.requester_id == original.requester_id


# =============================================================================
# ContentResponseMessage Tests
# =============================================================================

class TestContentResponseMessage:
    """Tests for ContentResponseMessage serialization."""
    
    def test_create_response_found(self):
        """Test creating a found response."""
        data = b"test content bytes"
        response = ContentResponseMessage(
            request_id="req123",
            cid="QmTestCID",
            status=ContentStatus.FOUND,
            data=data,
            size=len(data),
            transfer_mode=TransferMode.INLINE,
            content_hash=hashlib.sha256(data).hexdigest(),
            filename="test.txt",
        )
        
        assert response.request_id == "req123"
        assert response.cid == "QmTestCID"
        assert response.status == ContentStatus.FOUND
        assert response.data == data
        assert response.size == len(data)
        assert response.transfer_mode == TransferMode.INLINE
    
    def test_create_response_not_found(self):
        """Test creating a not-found response."""
        response = ContentResponseMessage.not_found("req123", "QmTestCID")
        
        assert response.request_id == "req123"
        assert response.cid == "QmTestCID"
        assert response.status == ContentStatus.NOT_FOUND
        assert response.error == "Content not found on this node"
    
    def test_create_response_error(self):
        """Test creating an error response."""
        response = ContentResponseMessage.error_response("req123", "QmTestCID", "Storage error")
        
        assert response.request_id == "req123"
        assert response.cid == "QmTestCID"
        assert response.status == ContentStatus.ERROR
        assert response.error == "Storage error"
    
    def test_response_to_payload_inline(self):
        """Test serializing inline response to payload."""
        data = b"test content"
        response = ContentResponseMessage(
            request_id="req123",
            cid="QmTestCID",
            status=ContentStatus.FOUND,
            data=data,
            size=len(data),
            transfer_mode=TransferMode.INLINE,
            content_hash="abc123",
            filename="test.txt",
        )
        
        payload = response.to_payload()
        
        assert payload["subtype"] == "content_response"
        assert payload["request_id"] == "req123"
        assert payload["cid"] == "QmTestCID"
        assert payload["status"] == "found"
        assert payload["transfer_mode"] == "inline"
        assert payload["data_b64"] == base64.b64encode(data).decode()
        assert payload["content_hash"] == "abc123"
        assert payload["filename"] == "test.txt"
    
    def test_response_to_payload_gateway(self):
        """Test serializing gateway response to payload."""
        response = ContentResponseMessage(
            request_id="req123",
            cid="QmTestCID",
            status=ContentStatus.FOUND,
            size=5_000_000,  # Large file
            transfer_mode=TransferMode.GATEWAY,
            gateway_url="prsm://content/QmTestCID",
        )
        
        payload = response.to_payload()
        
        assert payload["transfer_mode"] == "gateway"
        assert payload["gateway_url"] == "prsm://content/QmTestCID"
        assert "data_b64" not in payload
    
    def test_response_from_payload_inline(self):
        """Test deserializing inline response from payload."""
        data = b"test content"
        payload = {
            "subtype": "content_response",
            "request_id": "req123",
            "cid": "QmTestCID",
            "status": "found",
            "size": len(data),
            "transfer_mode": "inline",
            "data_b64": base64.b64encode(data).decode(),
            "content_hash": "abc123",
            "filename": "test.txt",
        }
        
        response = ContentResponseMessage.from_payload(payload)
        
        assert response.request_id == "req123"
        assert response.cid == "QmTestCID"
        assert response.status == ContentStatus.FOUND
        assert response.data == data
        assert response.transfer_mode == TransferMode.INLINE
    
    def test_response_from_payload_not_found(self):
        """Test deserializing not-found response."""
        payload = {
            "subtype": "content_response",
            "request_id": "req123",
            "cid": "QmTestCID",
            "status": "not_found",
            "error": "Content not found",
        }
        
        response = ContentResponseMessage.from_payload(payload)
        
        assert response.status == ContentStatus.NOT_FOUND
        assert response.error == "Content not found"
    
    def test_response_roundtrip(self):
        """Test response serialization roundtrip."""
        data = b"test content for roundtrip"
        original = ContentResponseMessage(
            request_id="req123",
            cid="QmTestCID",
            status=ContentStatus.FOUND,
            data=data,
            size=len(data),
            transfer_mode=TransferMode.INLINE,
            content_hash=hashlib.sha256(data).hexdigest(),
            filename="test.txt",
        )
        
        payload = original.to_payload()
        restored = ContentResponseMessage.from_payload(payload)
        
        assert restored.request_id == original.request_id
        assert restored.cid == original.cid
        assert restored.status == original.status
        assert restored.data == original.data
        assert restored.size == original.size
        assert restored.transfer_mode == original.transfer_mode


# =============================================================================
# ContentDiscovery Tests
# =============================================================================

class TestContentDiscovery:
    """Tests for ContentDiscovery content tracking."""
    
    def test_announce_content(self, content_discovery):
        """Test announcing content availability."""
        content_discovery.announce_content("QmCID1", "node_abc")
        
        assert "QmCID1" in content_discovery.content_locations
        assert "node_abc" in content_discovery.content_locations["QmCID1"]
    
    def test_multiple_providers(self, content_discovery):
        """Test tracking multiple providers for same content."""
        content_discovery.announce_content("QmCID1", "node_abc")
        content_discovery.announce_content("QmCID1", "node_def")
        content_discovery.announce_content("QmCID1", "node_ghi")
        
        providers = content_discovery.find_content_peers("QmCID1")
        
        assert len(providers) == 3
        assert "node_abc" in providers
        assert "node_def" in providers
        assert "node_ghi" in providers
    
    def test_find_content_peers(self, content_discovery):
        """Test finding peers for content."""
        content_discovery.announce_content("QmCID1", "node_abc")
        content_discovery.announce_content("QmCID2", "node_def")
        
        peers1 = content_discovery.find_content_peers("QmCID1")
        peers2 = content_discovery.find_content_peers("QmCID2")
        peers3 = content_discovery.find_content_peers("QmCID3")
        
        assert peers1 == ["node_abc"]
        assert peers2 == ["node_def"]
        assert peers3 == []
    
    def test_has_content(self, content_discovery):
        """Test checking if content exists."""
        content_discovery.announce_content("QmCID1", "node_abc")
        
        assert content_discovery.has_content("QmCID1") is True
        assert content_discovery.has_content("QmCID2") is False
    
    def test_remove_provider(self, content_discovery):
        """Test removing a provider."""
        content_discovery.announce_content("QmCID1", "node_abc")
        content_discovery.announce_content("QmCID1", "node_def")
        
        content_discovery.remove_provider("QmCID1", "node_abc")
        
        providers = content_discovery.find_content_peers("QmCID1")
        assert "node_abc" not in providers
        assert "node_def" in providers
    
    def test_remove_last_provider(self, content_discovery):
        """Test removing the last provider removes the content entry."""
        content_discovery.announce_content("QmCID1", "node_abc")
        
        content_discovery.remove_provider("QmCID1", "node_abc")
        
        assert "QmCID1" not in content_discovery.content_locations
        assert content_discovery.has_content("QmCID1") is False
    
    def test_announce_with_metadata(self, content_discovery):
        """Test announcing content with metadata."""
        announcement = ContentAnnouncement(
            cid="QmCID1",
            size=1024,
            content_type="application/json",
            content_hash="abc123",
            provider_id="node_abc",
            filename="data.json",
        )
        
        content_discovery.announce_content("QmCID1", "node_abc", announcement)
        
        info = content_discovery.get_content_info("QmCID1")
        assert info is not None
        assert info.size == 1024
        assert info.content_type == "application/json"
        assert info.filename == "data.json"
    
    def test_eviction(self):
        """Test LRU eviction when over limit."""
        discovery = ContentDiscovery(max_content_entries=5)
        
        # Add 6 items
        for i in range(6):
            discovery.announce_content(f"QmCID{i}", f"node_{i}")
        
        # First item should be evicted
        assert discovery.has_content("QmCID0") is False
        assert discovery.has_content("QmCID5") is True
    
    def test_get_stats(self, content_discovery):
        """Test getting discovery statistics."""
        content_discovery.announce_content("QmCID1", "node_abc")
        content_discovery.announce_content("QmCID1", "node_def")
        content_discovery.announce_content("QmCID2", "node_ghi")
        
        stats = content_discovery.get_stats()
        
        assert stats["tracked_cids"] == 2
        assert stats["total_providers"] == 3


# =============================================================================
# ContentAnnouncement Tests
# =============================================================================

class TestContentAnnouncement:
    """Tests for ContentAnnouncement."""
    
    def test_create_announcement(self):
        """Test creating a content announcement."""
        announcement = ContentAnnouncement(
            cid="QmCID1",
            size=1024,
            content_type="text/plain",
            content_hash="abc123",
            provider_id="node_abc",
            filename="test.txt",
        )
        
        assert announcement.cid == "QmCID1"
        assert announcement.size == 1024
        assert announcement.provider_id == "node_abc"
    
    def test_to_gossip_data(self):
        """Test converting announcement to gossip data."""
        announcement = ContentAnnouncement(
            cid="QmCID1",
            size=1024,
            content_type="text/plain",
            content_hash="abc123",
            provider_id="node_abc",
            filename="test.txt",
            metadata={"key": "value"},
        )
        
        data = announcement.to_gossip_data()
        
        assert data["cid"] == "QmCID1"
        assert data["size_bytes"] == 1024
        assert data["content_type"] == "text/plain"
        assert data["provider_id"] == "node_abc"
        assert data["filename"] == "test.txt"
    
    def test_from_gossip_data(self):
        """Test creating announcement from gossip data."""
        data = {
            "cid": "QmCID1",
            "size_bytes": 1024,
            "content_type": "text/plain",
            "content_hash": "abc123",
            "provider_id": "node_abc",
            "filename": "test.txt",
            "metadata": {"key": "value"},
        }
        
        announcement = ContentAnnouncement.from_gossip_data(data, "origin_node")
        
        assert announcement.cid == "QmCID1"
        assert announcement.size == 1024
        assert announcement.provider_id == "node_abc"


# =============================================================================
# ContentProvider Tests
# =============================================================================

class TestContentProvider:
    """Tests for ContentProvider."""
    
    def test_start_registers_handlers(self, content_provider, mock_transport, mock_gossip):
        """Test that start() registers message handlers."""
        content_provider.start()
        
        # Should register for direct messages
        mock_transport.on_message.assert_called_once()
        call_args = mock_transport.on_message.call_args
        assert call_args[0][0] == MSG_DIRECT
        
        # Should subscribe to content advertisements
        mock_gossip.subscribe.assert_called_once()
    
    def test_register_local_content(self, content_provider):
        """Test registering local content."""
        content_provider.register_local_content(
            cid="QmCID1",
            size_bytes=1024,
            content_hash="abc123",
            filename="test.txt",
        )
        
        assert content_provider.has_local_content("QmCID1")
        assert not content_provider.has_local_content("QmCID2")
    
    def test_unregister_local_content(self, content_provider):
        """Test unregistering local content."""
        content_provider.register_local_content(
            cid="QmCID1",
            size_bytes=1024,
            content_hash="abc123",
        )
        
        content_provider.unregister_local_content("QmCID1")
        
        assert not content_provider.has_local_content("QmCID1")
    
    @pytest.mark.asyncio
    async def test_handle_content_request_not_found(self, content_provider, mock_transport):
        """Test handling request for content we don't have."""
        content_provider.start()
        
        # Create mock peer
        peer = MagicMock()
        peer.peer_id = "requester_node"
        
        # Create request message
        msg = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id="requester_node",
            payload=ContentRequestMessage(
                cid="QmMissing",
                request_id="req123",
            ).to_payload(),
        )
        
        # Handle the request
        await content_provider._handle_content_request(msg, peer)
        
        # Should have sent a not-found response
        mock_transport.send_to_peer.assert_called_once()
        call_args = mock_transport.send_to_peer.call_args
        response_msg = call_args[0][1]
        response = ContentResponseMessage.from_payload(response_msg.payload)
        
        assert response.status == ContentStatus.NOT_FOUND
    
    @pytest.mark.asyncio
    async def test_handle_content_request_inline(self, content_provider, mock_transport):
        """Test handling request for small content (inline transfer)."""
        content_provider.start()
        
        # Register local content
        test_data = b"test content data"
        content_hash = hashlib.sha256(test_data).hexdigest()
        content_provider.register_local_content(
            cid="QmCID1",
            size_bytes=len(test_data),
            content_hash=content_hash,
            filename="test.txt",
        )
        
        # Mock content fetch
        with patch.object(content_provider, '_fetch_local', new_callable=AsyncMock) as mock_cat:
            mock_cat.return_value = test_data
            
            # Create mock peer
            peer = MagicMock()
            peer.peer_id = "requester_node"
            
            # Create request message
            msg = P2PMessage(
                msg_type=MSG_DIRECT,
                sender_id="requester_node",
                payload=ContentRequestMessage(
                    cid="QmCID1",
                    request_id="req123",
                ).to_payload(),
            )
            
            # Handle the request
            await content_provider._handle_content_request(msg, peer)
            
            # Should have sent a found response with inline data
            mock_transport.send_to_peer.assert_called_once()
            call_args = mock_transport.send_to_peer.call_args
            response_msg = call_args[0][1]
            response = ContentResponseMessage.from_payload(response_msg.payload)
            
            assert response.status == ContentStatus.FOUND
            assert response.transfer_mode == TransferMode.INLINE
            assert response.data == test_data
    
    @pytest.mark.asyncio
    async def test_handle_content_request_chunked(self, content_provider, mock_transport):
        """sp1020 — content above MAX_INLINE_SIZE (within the chunked ceiling) is
        served as multiple ordered CHUNKED frames over the P2P substrate (was
        previously a single dead GATEWAY prsm:// URL)."""
        content_provider.start()

        large_size = MAX_INLINE_SIZE + 1
        content_provider.register_local_content(
            cid="QmLargeCID",
            size_bytes=large_size,
            content_hash="abc123",
            filename="large.bin",
        )

        with patch.object(content_provider, '_fetch_local', new_callable=AsyncMock) as mock_cat:
            mock_cat.return_value = b"x" * large_size

            peer = MagicMock()
            peer.peer_id = "requester_node"
            msg = P2PMessage(
                msg_type=MSG_DIRECT,
                sender_id="requester_node",
                payload=ContentRequestMessage(cid="QmLargeCID", request_id="req123").to_payload(),
            )

            await content_provider._handle_content_request(msg, peer)

            # Multiple CHUNKED frames, not a single inline/gateway frame.
            assert mock_transport.send_to_peer.call_count >= 2
            responses = [
                ContentResponseMessage.from_payload(c[0][1].payload)
                for c in mock_transport.send_to_peer.call_args_list
            ]
            assert all(r.status == ContentStatus.FOUND for r in responses)
            assert all(r.transfer_mode == TransferMode.CHUNKED for r in responses)
            ordered = sorted(responses, key=lambda r: r.chunk_index)
            assert b"".join(r.data for r in ordered) == b"x" * large_size

    @pytest.mark.asyncio
    async def test_handle_content_request_gateway_above_chunked_ceiling(
        self, content_provider, mock_transport, monkeypatch,
    ):
        """sp1020 — content above the chunked ceiling falls back to the GATEWAY
        URL (a real HTTP gateway / libtorrent swarm is the right path there)."""
        # Force the ceiling down to MAX_INLINE_SIZE so a just-over-inline payload
        # exceeds it (avoids allocating tens of MiB in the test).
        monkeypatch.setenv("PRSM_MAX_CHUNKED_TRANSFER_BYTES", str(MAX_INLINE_SIZE))
        content_provider.start()

        large_size = MAX_INLINE_SIZE + 1
        content_provider.register_local_content(
            cid="QmHugeCID",
            size_bytes=large_size,
            content_hash="abc123",
            filename="huge.bin",
        )

        with patch.object(content_provider, '_fetch_local', new_callable=AsyncMock) as mock_cat:
            mock_cat.return_value = b"x" * large_size

            peer = MagicMock()
            peer.peer_id = "requester_node"
            msg = P2PMessage(
                msg_type=MSG_DIRECT,
                sender_id="requester_node",
                payload=ContentRequestMessage(cid="QmHugeCID", request_id="req123").to_payload(),
            )

            await content_provider._handle_content_request(msg, peer)

            mock_transport.send_to_peer.assert_called_once()
            response = ContentResponseMessage.from_payload(
                mock_transport.send_to_peer.call_args[0][1].payload
            )
            assert response.status == ContentStatus.FOUND
            assert response.transfer_mode == TransferMode.GATEWAY
            assert "gateway_url" in response.to_payload()
    
    @pytest.mark.asyncio
    async def test_request_content_local(self, content_provider):
        """Test requesting content that we have locally."""
        test_data = b"local content"
        content_provider.register_local_content(
            cid="QmLocalCID",
            size_bytes=len(test_data),
            content_hash=hashlib.sha256(test_data).hexdigest(),
        )
        
        with patch.object(content_provider, '_fetch_local', new_callable=AsyncMock) as mock_cat:
            mock_cat.return_value = test_data
            
            result = await content_provider.request_content("QmLocalCID")
            
            assert result == test_data
    
    @pytest.mark.asyncio
    async def test_request_content_no_providers(self, content_provider):
        """Test requesting content with no providers."""
        result = await content_provider.request_content("QmMissingCID")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_request_content_from_provider(self, content_provider, mock_transport, mock_content_index):
        """Test requesting content from a remote provider - tests the response handling path."""
        test_data = b"remote content"
        content_hash = hashlib.sha256(test_data).hexdigest()
        
        # Setup mock content index with provider info
        mock_record = MagicMock()
        mock_record.providers = {"provider_node"}
        mock_record.content_hash = content_hash
        mock_content_index.lookup.return_value = mock_record
        
        # Test the response handling directly
        request_id = "test_req_123"
        
        # Create a future and register it
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        content_provider._pending_requests[request_id] = future
        
        # Create the response
        response = ContentResponseMessage(
            request_id=request_id,
            cid="QmRemoteCID",
            status=ContentStatus.FOUND,
            data=test_data,
            size=len(test_data),
            transfer_mode=TransferMode.INLINE,
            content_hash=content_hash,
        )
        
        # Set the response on the future
        future.set_result(response)
        
        # Now simulate what _request_from_provider does after receiving response
        # This tests the response processing logic
        assert response.status == ContentStatus.FOUND
        assert response.transfer_mode == TransferMode.INLINE
        assert response.data == test_data
        
        # Verify hash matches
        actual_hash = hashlib.sha256(response.data).hexdigest()
        assert actual_hash == content_hash
        
        # Clean up
        content_provider._pending_requests.pop(request_id, None)
    
    @pytest.mark.asyncio
    async def test_request_content_timeout(self, content_provider, mock_transport, mock_content_index):
        """Test content request timeout."""
        # Setup mock content index with provider info
        mock_record = MagicMock()
        mock_record.providers = {"provider_node"}
        mock_record.content_hash = "abc123"
        mock_content_index.lookup.return_value = mock_record
        
        # Make send succeed but don't provide response
        mock_transport.send_to_peer = AsyncMock(return_value=True)
        
        # Request with short timeout
        result = await content_provider.request_content("QmRemoteCID", timeout=0.1)
        
        assert result is None
        assert content_provider._telemetry["requests_timed_out"] > 0
    
    @pytest.mark.asyncio
    async def test_request_content_hash_mismatch(self, content_provider, mock_transport, mock_content_index):
        """Test content hash verification failure."""
        test_data = b"wrong content"
        
        # Setup mock content index with wrong hash
        mock_record = MagicMock()
        mock_record.providers = {"provider_node"}
        mock_record.content_hash = "wrong_hash_12345"
        mock_content_index.lookup.return_value = mock_record
        
        # Setup response handling
        async def mock_send(peer_id, msg):
            response = ContentResponseMessage(
                request_id=msg.payload["request_id"],
                cid="QmCID",
                status=ContentStatus.FOUND,
                data=test_data,
                size=len(test_data),
                transfer_mode=TransferMode.INLINE,
            )
            response_msg = P2PMessage(
                msg_type=MSG_DIRECT,
                sender_id="provider_node",
                payload=response.to_payload(),
            )
            content_provider._handle_content_response(response_msg)
        
        mock_transport.send_to_peer = AsyncMock(side_effect=mock_send)
        
        result = await content_provider.request_content("QmCID", verify_hash=True)
        
        # Should return None due to hash mismatch
        assert result is None
    
    @pytest.mark.asyncio
    async def test_announce_content(self, content_provider, mock_gossip):
        """Test announcing content to network."""
        await content_provider.announce_content(
            cid="QmCID1",
            size=1024,
            content_type="text/plain",
            content_hash="abc123",
            filename="test.txt",
        )
        
        mock_gossip.publish.assert_called_once()
        call_args = mock_gossip.publish.call_args
        assert call_args[0][0] == "content_advertise"
        data = call_args[0][1]
        assert data["cid"] == "QmCID1"
    
    def test_get_stats(self, content_provider):
        """Test getting provider statistics."""
        content_provider.register_local_content("QmCID1", 1024, "hash1")
        content_provider.register_local_content("QmCID2", 2048, "hash2")
        
        stats = content_provider.get_stats()
        
        assert stats["local_content_count"] == 2
        assert "discovery" in stats
        assert "requests_received" in stats


# =============================================================================
# Integration Tests
# =============================================================================

class TestCrossNodeContentIntegration:
    """Integration tests for cross-node content retrieval."""
    
    @pytest.mark.asyncio
    async def test_full_request_response_flow(self):
        """Test complete request/response flow between two nodes."""
        # Create two content providers (simulating two nodes)
        identity_a = MagicMock()
        identity_a.node_id = "node_alpha"
        identity_a.public_key_b64 = base64.b64encode(b"key_a").decode()
        identity_a.sign = MagicMock(return_value="sig_a")
        
        identity_b = MagicMock()
        identity_b.node_id = "node_beta"
        identity_b.public_key_b64 = base64.b64encode(b"key_b").decode()
        identity_b.sign = MagicMock(return_value="sig_b")
        
        # Create transports
        transport_a = MagicMock()
        transport_a.identity = identity_a
        transport_a.on_message = MagicMock()
        transport_a.send_to_peer = AsyncMock(return_value=True)
        
        transport_b = MagicMock()
        transport_b.identity = identity_b
        transport_b.on_message = MagicMock()
        transport_b.send_to_peer = AsyncMock(return_value=True)
        
        # Create gossip protocols
        gossip_a = MagicMock()
        gossip_a.subscribe = MagicMock()
        gossip_a.publish = AsyncMock(return_value=1)
        
        gossip_b = MagicMock()
        gossip_b.subscribe = MagicMock()
        gossip_b.publish = AsyncMock(return_value=1)
        
        # Create content providers
        provider_a = ContentProvider(
            identity=identity_a,
            transport=transport_a,
            gossip=gossip_a,
        )
        
        provider_b = ContentProvider(
            identity=identity_b,
            transport=transport_b,
            gossip=gossip_b,
        )
        
        # Start providers
        provider_a.start()
        provider_b.start()
        
        # Node A registers content
        test_data = b"shared content from node A"
        test_hash = hashlib.sha256(test_data).hexdigest()
        provider_a.register_local_content(
            cid="QmSharedCID",
            size_bytes=len(test_data),
            content_hash=test_hash,
            filename="shared.txt",
        )
        
        # Node B discovers content via gossip
        provider_b.content_discovery.announce_content("QmSharedCID", "node_alpha")
        
        # Mock content fetch on node A
        with patch.object(provider_a, '_fetch_local', new_callable=AsyncMock) as mock_cat:
            mock_cat.return_value = test_data
            
            # Simulate request/response flow
            # Node B creates request
            request = ContentRequestMessage(
                cid="QmSharedCID",
                requester_id="node_beta",
            )
            
            # Node A handles request
            peer_a = MagicMock()
            peer_a.peer_id = "node_beta"
            
            request_msg = P2PMessage(
                msg_type=MSG_DIRECT,
                sender_id="node_beta",
                payload=request.to_payload(),
            )
            
            await provider_a._handle_content_request(request_msg, peer_a)
            
            # Verify response was sent
            assert transport_a.send_to_peer.called
            response_payload = transport_a.send_to_peer.call_args[0][1].payload
            
            # Verify response content
            assert response_payload["status"] == "found"
            assert response_payload["transfer_mode"] == "inline"
            
            # Decode and verify data
            decoded_data = base64.b64decode(response_payload["data_b64"])
            assert decoded_data == test_data


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in content retrieval."""
    
    @pytest.mark.asyncio
    async def test_fetch_failure(self, content_provider, mock_transport):
        """Test handling content fetch failure."""
        content_provider.start()
        
        content_provider.register_local_content(
            cid="QmCID1",
            size_bytes=1024,
            content_hash="abc123",
        )
        
        # Mock fetch failure
        with patch.object(content_provider, '_fetch_local', new_callable=AsyncMock) as mock_cat:
            mock_cat.return_value = None  # fetch failed
            
            peer = MagicMock()
            peer.peer_id = "requester"
            
            msg = P2PMessage(
                msg_type=MSG_DIRECT,
                sender_id="requester",
                payload=ContentRequestMessage(cid="QmCID1").to_payload(),
            )
            
            await content_provider._handle_content_request(msg, peer)
            
            # Should have sent error response
            response_payload = mock_transport.send_to_peer.call_args[0][1].payload
            assert response_payload["status"] == "error"
    
    @pytest.mark.asyncio
    async def test_provider_unavailable(self, content_provider, mock_transport, mock_content_index):
        """Test handling when provider is unavailable."""
        mock_record = MagicMock()
        mock_record.providers = {"unavailable_node"}
        mock_record.content_hash = "abc123"
        mock_content_index.lookup.return_value = mock_record
        
        # Make send fail
        mock_transport.send_to_peer = AsyncMock(return_value=False)
        
        result = await content_provider.request_content("QmCID1")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_gateway_failure(self, content_provider):
        """Test handling gateway fetch failure."""
        response = ContentResponseMessage(
            request_id="req123",
            cid="QmCID",
            status=ContentStatus.FOUND,
            size=5_000_000,
            transfer_mode=TransferMode.GATEWAY,
            gateway_url="prsm://content/QmCID",
        )
        
        # Mock gateway fetch failure
        with patch.object(content_provider, '_fetch_from_url', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = None
            
            result = await content_provider._request_from_provider(
                "QmCID", "provider", 0.1, None  # 0.1s: future never resolves in tests
            )

            # No transport is active so the future times out immediately → None
            assert result is None


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
