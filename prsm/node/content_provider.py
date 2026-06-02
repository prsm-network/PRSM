"""
Content Provider
================

Handles cross-node content retrieval over the P2P network.

This module implements the content request/response protocol that allows
Node A to store content and Node B to discover and retrieve it.

Protocol Flow:
1. Node A uploads content and announces via gossip (GOSSIP_CONTENT_ADVERTISE)
2. ContentIndex on Node B records the advertisement
3. Node B requests content via ContentProvider.request_content()
4. Node A's ContentProvider handles the request and responds with data
5. Node B receives and validates the content

Message Types:
- ContentRequestMessage: Request content from a peer
- ContentResponseMessage: Response with content data or error
"""

import asyncio
import base64
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from prsm.node.transport import MSG_DIRECT, P2PMessage, PeerConnection, WebSocketTransport
from prsm.node.gossip import GOSSIP_CONTENT_ADVERTISE, GOSSIP_CONTENT_ACCESS, GossipProtocol
from prsm.node.identity import NodeIdentity


def sign_content_access_event(payload: Dict[str, Any], identity: NodeIdentity) -> Dict[str, Any]:
    """sp909 — sign a GOSSIP_CONTENT_ACCESS payload so subscribers can
    AUTHENTICATE it before crediting royalties (mirrors
    ledger_sync.broadcast_transaction). The signature covers the canonical
    (sorted-JSON) payload excluding the signature fields themselves;
    `_on_content_access` rebuilds the same canonical and verifies via
    verify_signature(origin_public_key, canonical, signature).

    Also reconciles the publisher/subscriber key: emits BOTH `cid` (legacy
    consumers) and `content_id` (what the subscriber reads) set to the same
    value, so the royalty path is no longer silently dead.
    """
    base = dict(payload)
    if "cid" in base and "content_id" not in base:
        base["content_id"] = base["cid"]
    canonical = {
        k: v for k, v in base.items()
        if k not in ("signature", "origin_public_key")
    }
    signature = identity.sign(json.dumps(canonical, sort_keys=True).encode())
    return {
        **canonical,
        "signature": signature,
        "origin_public_key": identity.public_key_b64,
    }

logger = logging.getLogger(__name__)

# Default timeouts and limits
DEFAULT_REQUEST_TIMEOUT = 30.0
MAX_INLINE_SIZE = 1_048_576  # 1MB - content larger than this uses gateway transfer
MAX_CONCURRENT_REQUESTS = 10


class ContentStatus(str, Enum):
    """Status of content retrieval."""
    FOUND = "found"
    NOT_FOUND = "not_found"
    ERROR = "error"
    TIMEOUT = "timeout"


class TransferMode(str, Enum):
    """How content is transferred."""
    INLINE = "inline"  # Base64 encoded in response
    GATEWAY = "gateway"  # Gateway URL provided (peer-supplied, may be PRSM-native or HTTP)


@dataclass
class ContentRequestMessage:
    """Request content from a peer.
    
    Sent via direct P2P message to request content that a peer has advertised.
    """
    cid: str  # PRSM content ID
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    priority: int = 0  # 0=normal, 1=high priority
    timeout: int = 30  # Request timeout in seconds
    requester_id: str = ""  # Set when sending
    
    def to_payload(self) -> Dict[str, Any]:
        """Convert to P2PMessage payload."""
        return {
            "subtype": "content_request",
            "cid": self.cid,
            "request_id": self.request_id,
            "priority": self.priority,
            "timeout": self.timeout,
            "requester_id": self.requester_id,
        }
    
    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ContentRequestMessage":
        """Create from P2PMessage payload."""
        return cls(
            cid=payload.get("cid", ""),
            request_id=payload.get("request_id", uuid.uuid4().hex[:16]),
            priority=payload.get("priority", 0),
            timeout=payload.get("timeout", 30),
            requester_id=payload.get("requester_id", ""),
        )


@dataclass
class ContentResponseMessage:
    """Response to content request.
    
    Contains either the content data (for small files) or a gateway URL
    (for large files), or an error if the content couldn't be retrieved.
    """
    request_id: str
    cid: str
    status: ContentStatus
    data: Optional[bytes] = None  # Content data if found and inline
    size: int = 0  # Content size in bytes
    error: Optional[str] = None  # Error message if status is "error"
    transfer_mode: Optional[TransferMode] = None
    gateway_url: Optional[str] = None  # Gateway URL for large files (peer-supplied)
    content_hash: Optional[str] = None  # SHA-256 hash for verification
    filename: Optional[str] = None
    
    def to_payload(self) -> Dict[str, Any]:
        """Convert to P2PMessage payload."""
        payload: Dict[str, Any] = {
            "subtype": "content_response",
            "request_id": self.request_id,
            "cid": self.cid,
            "status": self.status.value,
            "size": self.size,
        }
        
        if self.error:
            payload["error"] = self.error
        
        if self.transfer_mode:
            payload["transfer_mode"] = self.transfer_mode.value
        
        if self.data is not None and self.transfer_mode == TransferMode.INLINE:
            payload["data_b64"] = base64.b64encode(self.data).decode()
        
        if self.gateway_url:
            payload["gateway_url"] = self.gateway_url
        
        if self.content_hash:
            payload["content_hash"] = self.content_hash
        
        if self.filename:
            payload["filename"] = self.filename
        
        return payload
    
    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ContentResponseMessage":
        """Create from P2PMessage payload."""
        status_str = payload.get("status", "error")
        try:
            status = ContentStatus(status_str)
        except ValueError:
            status = ContentStatus.ERROR
        
        transfer_mode = None
        mode_str = payload.get("transfer_mode")
        if mode_str:
            try:
                transfer_mode = TransferMode(mode_str)
            except ValueError:
                pass
        
        data = None
        data_b64 = payload.get("data_b64")
        if data_b64 and transfer_mode == TransferMode.INLINE:
            try:
                data = base64.b64decode(data_b64)
            except Exception:
                pass
        
        return cls(
            request_id=payload.get("request_id", ""),
            cid=payload.get("cid", ""),
            status=status,
            data=data,
            size=payload.get("size", 0),
            error=payload.get("error"),
            transfer_mode=transfer_mode,
            gateway_url=payload.get("gateway_url"),
            content_hash=payload.get("content_hash"),
            filename=payload.get("filename"),
        )
    
    @classmethod
    def not_found(cls, request_id: str, cid: str) -> "ContentResponseMessage":
        """Create a not-found response."""
        return cls(
            request_id=request_id,
            cid=cid,
            status=ContentStatus.NOT_FOUND,
            error="Content not found on this node",
        )
    
    @classmethod
    def error_response(cls, request_id: str, cid: str, error: str) -> "ContentResponseMessage":
        """Create an error response."""
        return cls(
            request_id=request_id,
            cid=cid,
            status=ContentStatus.ERROR,
            error=error,
        )


@dataclass
class ContentAnnouncement:
    """Announce content availability via gossip.

    Broadcast when a node has content available for retrieval.
    """
    cid: str
    size: int
    content_type: str
    content_hash: str
    provider_id: str
    timestamp: float = field(default_factory=time.time)
    filename: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Phase 1.2: canonical creator-bound provenance hash (0x-prefixed
    # hex). Forwarded to remote nodes so replicated content can route
    # royalties through the on-chain RoyaltyDistributor. None when the
    # original uploader had no 0x creator_address.
    provenance_hash: Optional[str] = None

    def to_gossip_data(self) -> Dict[str, Any]:
        """Convert to gossip message data."""
        return {
            "cid": self.cid,
            "size_bytes": self.size,
            "content_type": self.content_type,
            "content_hash": self.content_hash,
            "provider_id": self.provider_id,
            "timestamp": self.timestamp,
            "filename": self.filename or "",
            "metadata": self.metadata,
            "provenance_hash": self.provenance_hash,
        }

    @classmethod
    def from_gossip_data(cls, data: Dict[str, Any], origin: str) -> "ContentAnnouncement":
        """Create from gossip message data."""
        return cls(
            cid=data.get("cid", ""),
            size=data.get("size_bytes", 0),
            content_type=data.get("content_type", ""),
            content_hash=data.get("content_hash", ""),
            provider_id=data.get("provider_id", origin),
            timestamp=data.get("timestamp", time.time()),
            filename=data.get("filename"),
            metadata=data.get("metadata", {}),
            provenance_hash=data.get("provenance_hash"),
        )


class ContentDiscovery:
    """Track which nodes have which content.
    
    Maintains a local index of content locations based on gossip
    advertisements. This is a lightweight alternative/supplement to
    the full ContentIndex for content provider lookups.
    """
    
    def __init__(self, max_content_entries: int = 10_000):
        self.max_content_entries = max_content_entries
        # cid -> set of peer_ids that have the content
        self.content_locations: Dict[str, Set[str]] = {}
        # cid -> ContentAnnouncement (metadata)
        self.content_metadata: Dict[str, ContentAnnouncement] = {}
        # LRU tracking
        self._access_order: Dict[str, float] = {}
    
    def announce_content(self, cid: str, peer_id: str, announcement: Optional[ContentAnnouncement] = None) -> None:
        """Record that a peer has content.
        
        Args:
            cid: Content identifier
            peer_id: Node ID of the peer with the content
            announcement: Optional metadata about the content
        """
        if cid not in self.content_locations:
            self.content_locations[cid] = set()
            self._evict_if_needed()
        
        self.content_locations[cid].add(peer_id)
        self._access_order[cid] = time.time()
        
        if announcement:
            self.content_metadata[cid] = announcement
    
    def remove_provider(self, cid: str, peer_id: str) -> None:
        """Remove a provider for a CID (e.g., when peer disconnects)."""
        if cid in self.content_locations:
            self.content_locations[cid].discard(peer_id)
            if not self.content_locations[cid]:
                del self.content_locations[cid]
                self.content_metadata.pop(cid, None)
                self._access_order.pop(cid, None)
    
    def find_content_peers(self, cid: str) -> List[str]:
        """Find peers that have the content.
        
        Returns peers ordered by most recently announced.
        """
        peers = self.content_locations.get(cid, set())
        if not peers:
            return []
        return list(peers)
    
    def has_content(self, cid: str) -> bool:
        """Check if any peer has the content."""
        return bool(self.content_locations.get(cid))
    
    def get_content_info(self, cid: str) -> Optional[ContentAnnouncement]:
        """Get metadata for content."""
        return self.content_metadata.get(cid)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get discovery statistics."""
        total_providers = sum(len(peers) for peers in self.content_locations.values())
        return {
            "tracked_cids": len(self.content_locations),
            "total_providers": total_providers,
            "metadata_entries": len(self.content_metadata),
        }
    
    def _evict_if_needed(self) -> None:
        """Evict oldest entries if over limit."""
        while len(self.content_locations) > self.max_content_entries:
            # Find oldest entry
            oldest_cid = min(self._access_order.items(), key=lambda x: x[1])[0]
            self.content_locations.pop(oldest_cid, None)
            self.content_metadata.pop(oldest_cid, None)
            self._access_order.pop(oldest_cid, None)


class ContentProvider:
    """Handles content requests from other nodes.
    
    This class is responsible for:
    1. Receiving and processing content requests from peers
    2. Retrieving content from the local ContentStore
    3. Sending content responses (inline or gateway URL)
    4. Requesting content from other peers when acting as a client
    
    Integration Points:
    - Transport: Receives MSG_DIRECT messages with content_request subtype
    - ContentStore: Retrieves content via the PRSM proprietary content store
    - Gossip: Announces content availability
    - ContentIndex: Looks up providers for content
    """

    def __init__(
        self,
        identity: NodeIdentity,
        transport: WebSocketTransport,
        gossip: GossipProtocol,
        content_index: Optional[Any] = None,
        content_discovery: Optional[ContentDiscovery] = None,
        default_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        bandwidth_limiter: Optional[Any] = None,
        content_economy: Optional[Any] = None,  # Phase 4: Payment processing
    ):
        self.identity = identity
        self.transport = transport
        self.gossip = gossip
        self.content_index = content_index
        self.content_discovery = content_discovery or ContentDiscovery()
        self.default_timeout = default_timeout
        
        # Bandwidth limiter for throttling content serving
        # This is typically the BandwidthLimiter from StorageProvider
        self.bandwidth_limiter = bandwidth_limiter
        
        # Content economy for payment processing (Phase 4)
        self.content_economy = content_economy
        
        # Local content we can serve (cid -> metadata)
        self._local_content: Dict[str, Dict[str, Any]] = {}
        
        # Pending requests: request_id -> asyncio.Future
        self._pending_requests: Dict[str, asyncio.Future] = {}
        
        # Semaphore to limit concurrent requests
        self._request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        # HTTP session for cross-node gateway fetches
        self._http_session: Optional[Any] = None

        # Sprint 427 (F7) — BT-infohash fallback for _fetch_local.
        # Set externally by Node init when the BitTorrent layer
        # is available. When wired, _fetch_local falls back to
        # ``content_retriever.fetch(infohash)`` for cids that are
        # in ``_local_content`` but structurally invalid for
        # ``ContentHash.from_hex`` (e.g., 40-char BT v1
        # infohashes). Pre-sprint-427 these cids returned None
        # from _fetch_local even though the bytes were locally
        # available via the BT swarm.
        self.content_retriever: Optional[Any] = None

        # Telemetry
        self._telemetry: Dict[str, Any] = {
            "requests_received": 0,
            "requests_served": 0,
            "requests_failed": 0,
            "bytes_served": 0,
            "requests_sent": 0,
            "requests_fulfilled": 0,
            "requests_timed_out": 0,
        }
    
    def start(self) -> None:
        """Register message handlers and gossip subscriptions."""
        # Register for direct content messages
        self.transport.on_message(MSG_DIRECT, self._handle_direct_message)
        
        # Subscribe to content advertisements
        self.gossip.subscribe(GOSSIP_CONTENT_ADVERTISE, self._on_content_advertise)
        
        logger.info("Content provider started — listening for content requests")
    
    async def stop(self) -> None:
        """Clean up resources."""
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

        # Cancel any pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()
    
    # ── Local Content Registration ─────────────────────────────────────
    
    def register_local_content(
        self,
        cid: str,
        size_bytes: int,
        content_hash: str,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register content that this node can serve.
        
        Args:
            cid: PRSM content identifier
            size_bytes: Size of the content
            content_hash: SHA-256 hash of the content
            filename: Optional filename
            metadata: Optional metadata dict
        """
        self._local_content[cid] = {
            "cid": cid,
            "size_bytes": size_bytes,
            "content_hash": content_hash,
            "filename": filename or cid,
            "metadata": metadata or {},
            "registered_at": time.time(),
        }
        
        logger.debug(f"Registered local content: {cid[:12]}... ({size_bytes} bytes)")
    
    def unregister_local_content(self, cid: str) -> None:
        """Unregister content that this node no longer serves."""
        self._local_content.pop(cid, None)
    
    def has_local_content(self, cid: str) -> bool:
        """Check if we have content locally."""
        return cid in self._local_content
    
    async def announce_content(self, cid: str, **kwargs) -> None:
        """Announce content availability to the network.
        
        Args:
            cid: Content identifier
            **kwargs: Additional announcement fields
        """
        announcement = ContentAnnouncement(
            cid=cid,
            size=kwargs.get("size", 0),
            content_type=kwargs.get("content_type", "application/octet-stream"),
            content_hash=kwargs.get("content_hash", ""),
            provider_id=self.identity.node_id,
            filename=kwargs.get("filename"),
            metadata=kwargs.get("metadata", {}),
            # Phase 1.2: forward provenance_hash so on-chain routing
            # survives the announce path in addition to the uploader's
            # direct GOSSIP_CONTENT_ADVERTISE publish.
            provenance_hash=kwargs.get("provenance_hash"),
        )
        
        await self.gossip.publish(GOSSIP_CONTENT_ADVERTISE, announcement.to_gossip_data())
    
    # ── Content Request Handling (Server Side) ──────────────────────────
    
    async def _handle_direct_message(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Route incoming direct messages to appropriate handlers."""
        subtype = msg.payload.get("subtype", "")
        
        if subtype == "content_request":
            await self._handle_content_request(msg, peer)
        elif subtype == "content_response":
            self._handle_content_response(msg)
    
    async def _handle_content_request(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Handle incoming content request from a peer.
        
        This is the server-side handler that:
        1. Checks if we have the content
        2. Retrieves it from the local ContentStore
        3. Sends the response
        """
        self._telemetry["requests_received"] += 1
        
        request = ContentRequestMessage.from_payload(msg.payload)
        cid = request.cid
        request_id = request.request_id
        
        logger.debug(f"Content request from {peer.peer_id[:8]}: {cid[:12]}...")
        
        # Check if we have this content
        content_info = self._local_content.get(cid)
        if not content_info:
            # We don't have it
            response = ContentResponseMessage.not_found(request_id, cid)
            await self._send_response(peer.peer_id, response)
            self._telemetry["requests_failed"] += 1
            return
        
        try:
            content_bytes = await self._fetch_local(cid)
            if content_bytes is None:
                response = ContentResponseMessage.error_response(
                    request_id, cid, "Failed to retrieve from ContentStore"
                )
                await self._send_response(peer.peer_id, response)
                self._telemetry["requests_failed"] += 1
                return
            
            size = len(content_bytes)
            
            # Apply bandwidth throttling before sending content
            if self.bandwidth_limiter:
                await self.bandwidth_limiter.throttle_upload(size)
            
            # Determine transfer mode
            if size <= MAX_INLINE_SIZE:
                # Send inline (base64 encoded)
                response = ContentResponseMessage(
                    request_id=request_id,
                    cid=cid,
                    status=ContentStatus.FOUND,
                    data=content_bytes,
                    size=size,
                    transfer_mode=TransferMode.INLINE,
                    content_hash=content_info.get("content_hash"),
                    filename=content_info.get("filename"),
                )
            else:
                # Provide gateway URL for large files
                gateway_url = f"prsm://content/{cid}"
                response = ContentResponseMessage(
                    request_id=request_id,
                    cid=cid,
                    status=ContentStatus.FOUND,
                    size=size,
                    transfer_mode=TransferMode.GATEWAY,
                    gateway_url=gateway_url,
                    content_hash=content_info.get("content_hash"),
                    filename=content_info.get("filename"),
                )
            
            await self._send_response(peer.peer_id, response)
            
            # Process payment for content access (Phase 4)
            if self.content_economy:
                try:
                    payment = await self.content_economy.process_content_access(
                        content_id=cid,
                        accessor_id=peer.peer_id,
                        content_metadata=self._resolve_payment_metadata(content_info),
                    )
                    if payment.status.value == "completed":
                        logger.debug(
                            f"Payment processed for {cid[:12]}... "
                            f"({payment.amount} FTNS from {peer.peer_id[:8]})"
                        )
                    elif payment.status.value == "pending_onchain":
                        # Phase 1.2: on-chain broadcast OK, receipt unknown —
                        # reconciliation handles this out-of-band.
                        logger.info(
                            f"Payment pending on-chain reconciliation for "
                            f"{cid[:12]}... ({payment.amount} FTNS from "
                            f"{peer.peer_id[:8]})"
                        )
                except Exception as e:
                    # Log but don't fail the transfer - payment issues handled separately
                    logger.warning(f"Payment processing failed for {cid[:12]}...: {e}")

            # Phase 1.3 Task 3b: publish GOSSIP_CONTENT_ACCESS so
            # source creators on other nodes can credit their shares
            # locally. This used to fire from ContentUploader's
            # (now-retired) server-side _handle_content_request. The
            # publish happens AFTER the payment try/except so it fires
            # regardless of whether the on-chain path succeeded or
            # fell back to local. Payload fields preserved from the
            # legacy uploader implementation.
            try:
                resolved = self._resolve_payment_metadata(content_info)
                await self.gossip.publish(GOSSIP_CONTENT_ACCESS, sign_content_access_event({
                    "cid": cid,
                    "accessor_id": peer.peer_id,
                    "creator_id": resolved.get("creator_id", ""),
                    "royalty_rate": resolved.get("royalty_rate", 0.01),
                    "parent_cids": resolved.get("parent_cids", []),
                    "timestamp": time.time(),
                    # Sp899 — unique per-access id so receivers credit
                    # this access event at most once (replay-safe).
                    "access_nonce": uuid.uuid4().hex,
                }, self.identity))
            except Exception as exc:
                logger.warning(
                    f"GOSSIP_CONTENT_ACCESS publish failed for "
                    f"{cid[:12]}...: {exc}"
                )

            self._telemetry["requests_served"] += 1
            self._telemetry["bytes_served"] += size
            
            logger.info(
                f"Served content {cid[:12]}... ({size} bytes) to {peer.peer_id[:8]}"
            )
            
        except Exception as e:
            logger.error(f"Error serving content {cid[:12]}...: {e}")
            response = ContentResponseMessage.error_response(request_id, cid, str(e))
            await self._send_response(peer.peer_id, response)
            self._telemetry["requests_failed"] += 1
    
    async def _send_response(self, peer_id: str, response: ContentResponseMessage) -> None:
        """Send a content response to a peer."""
        msg = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id=self.identity.node_id,
            payload=response.to_payload(),
        )
        await self.transport.send_to_peer(peer_id, msg)
    
    # ── Content Request (Client Side) ───────────────────────────────────
    
    async def request_content(
        self,
        cid: str,
        timeout: Optional[float] = None,
        verify_hash: bool = True,
        preferred_peer: Optional[str] = None,
    ) -> Optional[bytes]:
        """Request content from the network by CID.
        
        Looks up providers via the content index or discovery,
        sends a content request, and returns the content bytes.
        
        Args:
            cid: Content identifier to retrieve
            timeout: Seconds to wait for response (default: self.default_timeout)
            verify_hash: If True, verify SHA-256 hash matches
            preferred_peer: Optional specific peer to request from first
        
        Returns:
            Content bytes, or None if not found/timed out/failed
        """
        timeout = timeout or self.default_timeout
        
        # Check if we have it locally first
        # Sprint 484 (F24): propagate timeout into _fetch_local so
        # the BT-fallback path can't block indefinitely on hydrated
        # CIDs whose BT seed metadata was lost across restart.
        if cid in self._local_content:
            local_bytes = await self._fetch_local(cid, timeout=timeout)
            if local_bytes is not None:
                # sp923 — honor verify_hash on the local shortcut too. The
                # remote path verifies returned bytes against the expected
                # SHA-256 (_get_content_hash); the local shortcut previously
                # returned bytes UNVERIFIED, so a caller passing verify_hash
                # got an unchecked local copy (inconsistent with remote + misses
                # local disk corruption of this node's own content). When the
                # CID has no known content hash (e.g. a BitTorrent infohash, not
                # a SHA-256 content address) _get_content_hash returns None and
                # there is nothing to verify by hash — return as before.
                if verify_hash:
                    expected = self._get_content_hash(cid)
                    if expected and hashlib.sha256(local_bytes).hexdigest() != expected:
                        logger.warning(
                            "local content for %s failed SHA-256 verification "
                            "(possible disk corruption) — not serving the local "
                            "copy; trying network providers",
                            cid[:12],
                        )
                        local_bytes = None
                if local_bytes is not None:
                    logger.debug(f"Retrieved {cid[:12]}... locally")
                    return local_bytes
        
        # Find providers
        providers = self._find_providers(cid)
        if not providers:
            logger.debug(f"No providers found for {cid[:12]}...")
            return None
        
        # Remove ourselves from providers
        providers.discard(self.identity.node_id)
        
        # If preferred peer specified, try them first
        if preferred_peer and preferred_peer in providers:
            providers_list = [preferred_peer] + [p for p in providers if p != preferred_peer]
        else:
            providers_list = list(providers)
        
        # Get expected hash for verification
        expected_hash = self._get_content_hash(cid)
        
        # Try each provider
        for provider_id in providers_list:
            result = await self._request_from_provider(
                cid, provider_id, timeout, expected_hash if verify_hash else None
            )
            if result is not None:
                self._telemetry["requests_fulfilled"] += 1
                return result
        
        logger.warning(f"Failed to retrieve {cid[:12]}... from any provider")
        return None
    
    def _find_providers(self, cid: str) -> Set[str]:
        """Find providers for a CID from content index and discovery."""
        providers: Set[str] = set()
        
        # Check content index first
        if self.content_index:
            record = self.content_index.lookup(cid)
            if record and hasattr(record, 'providers'):
                providers.update(record.providers)
        
        # Also check local discovery
        discovery_peers = self.content_discovery.find_content_peers(cid)
        providers.update(discovery_peers)
        
        return providers
    
    def _get_content_hash(self, cid: str) -> Optional[str]:
        """Get expected content hash for verification."""
        # Check content index
        if self.content_index:
            record = self.content_index.lookup(cid)
            if record and hasattr(record, 'content_hash'):
                return record.content_hash
        
        # Check local discovery
        info = self.content_discovery.get_content_info(cid)
        if info:
            return info.content_hash
        
        return None
    
    async def _request_from_provider(
        self,
        cid: str,
        provider_id: str,
        timeout: float,
        expected_hash: Optional[str] = None,
    ) -> Optional[bytes]:
        """Request content from a specific provider.
        
        Returns content bytes or None on failure.
        """
        self._telemetry["requests_sent"] += 1
        
        request = ContentRequestMessage(
            cid=cid,
            timeout=int(timeout),
            requester_id=self.identity.node_id,
        )
        request_id = request.request_id
        
        # Create future for response
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_requests[request_id] = future
        
        try:
            # Send request
            msg = P2PMessage(
                msg_type=MSG_DIRECT,
                sender_id=self.identity.node_id,
                payload=request.to_payload(),
            )
            sent = await self.transport.send_to_peer(provider_id, msg)
            if not sent:
                logger.debug(f"Failed to send request to {provider_id[:8]}")
                return None
            
            # Wait for response
            response = await asyncio.wait_for(future, timeout=timeout)
            
        except asyncio.TimeoutError:
            self._telemetry["requests_timed_out"] += 1
            logger.debug(f"Request to {provider_id[:8]} timed out for {cid[:12]}...")
            return None
        except Exception as e:
            logger.debug(f"Request to {provider_id[:8]} failed: {e}")
            return None
        finally:
            self._pending_requests.pop(request_id, None)
        
        # Process response
        if response.status != ContentStatus.FOUND:
            logger.debug(f"Provider {provider_id[:8]} returned {response.status}")
            return None
        
        # Get content bytes
        content_bytes: Optional[bytes] = None
        
        if response.transfer_mode == TransferMode.INLINE and response.data is not None:
            content_bytes = response.data
        
        elif response.transfer_mode == TransferMode.GATEWAY and response.gateway_url:
            content_bytes = await self._fetch_from_url(response.gateway_url)
        
        if content_bytes is None:
            return None
        
        # Verify hash if provided
        if expected_hash:
            actual_hash = hashlib.sha256(content_bytes).hexdigest()
            if actual_hash != expected_hash:
                logger.warning(
                    f"Hash mismatch for {cid[:12]}... from {provider_id[:8]}: "
                    f"expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
                )
                return None
        
        logger.info(
            f"Retrieved {len(content_bytes)} bytes for {cid[:12]}... from {provider_id[:8]}"
        )
        return content_bytes
    
    def _handle_content_response(self, msg: P2PMessage) -> None:
        """Handle incoming content response (client side)."""
        response = ContentResponseMessage.from_payload(msg.payload)
        request_id = response.request_id
        
        future = self._pending_requests.get(request_id)
        if future and not future.done():
            future.set_result(response)
    
    # ── Gossip Handlers ─────────────────────────────────────────────────
    
    async def _on_content_advertise(
        self, subtype: str, data: Dict[str, Any], origin: str
    ) -> None:
        """Handle content advertisement from gossip."""
        announcement = ContentAnnouncement.from_gossip_data(data, origin)
        
        self.content_discovery.announce_content(
            announcement.cid,
            announcement.provider_id,
            announcement,
        )
        
        logger.debug(
            f"Discovered content {announcement.cid[:12]}... at {announcement.provider_id[:8]}"
        )
    
    # ── Content Store / Gateway Operations ─────────────────────────────

    async def _get_http_session(self) -> Any:
        """Lazily create and return a shared aiohttp session for gateway fetches."""
        if self._http_session is None:
            import aiohttp
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    async def _fetch_local(
        self, content_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[bytes]:
        """Retrieve content from the local ContentStore by content hash.

        Returns None (never raises) for any of: ContentStore unavailable,
        malformed CID hex, missing content, or underlying storage errors.

        Sprint 427 (F7): falls back to ``self.content_retriever.fetch``
        when the cid is structurally invalid for ``ContentHash.from_hex``
        OR ContentStore returns None, and the cid is in
        ``_local_content``. Closes the BT-infohash vs ContentHash storage
        split documented in
        ``docs/operations/2026-05-14-user-dogfood-findings.md`` F7.

        Sprint 484 (F24): propagates ``timeout`` to the BT-fallback
        path. Pre-fix, BT-fallback was unbounded; for hydrated CIDs
        (post-sprint-480 F22 fix) where BT publisher state was lost
        across restart, retrieve hung forever.
        """
        try:
            from prsm.storage import ContentHash, get_content_store
            from prsm.storage.exceptions import ContentNotFoundError, StorageError

            store = get_content_store()
            if store is None:
                logger.debug(
                    f"ContentStore not available, cannot retrieve {content_id[:12]}..."
                )
                return await self._fetch_local_via_bt(content_id, timeout=timeout)
            try:
                bytes_ = await store.retrieve_local(
                    ContentHash.from_hex(content_id),
                )
                if bytes_ is not None:
                    return bytes_
            except ValueError:
                # Malformed for ContentStore — fall through to BT.
                logger.debug(
                    f"cid not ContentHash-shaped: {content_id[:12]}... "
                    "— trying BT fallback"
                )
        except ContentNotFoundError:
            logger.debug(f"Content not found in ContentStore for {content_id[:12]}...")
        except (StorageError, OSError) as e:
            logger.error(f"ContentStore retrieve failed for {content_id[:12]}...: {e}")
        return await self._fetch_local_via_bt(content_id, timeout=timeout)

    async def _fetch_local_via_bt(
        self, content_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[bytes]:
        """Sprint 427 (F7): fetch via the wired ``ContentRetriever``.

        Fires only for cids that are in ``_local_content`` — guards
        against accidentally swarming the BT layer for unknown cids
        when the local-lookup path is meant to be a fast no-op.

        Sprint 484 (F24): propagates ``timeout`` to retriever.fetch().
        Pre-fix: timeout=None → BT swarm wait was unbounded. This
        regressed F8 single-node retrieve for content hydrated by
        sprint 480's F22 fix — `_local_content` re-populated post-
        restart, but BT-publisher's `_published_paths` is per-session
        so `local_publish_path()` returned None → fell through to
        `bt_requester.request_content(timeout=None)` → hung forever
        even though the user's curl had its own timeout. Operator-
        visible symptom: `/content/retrieve/{cid}` would never
        return for any hydrated CID.

        Returns None (never raises) for: no retriever wired, cid not
        in ``_local_content``, or retriever errors.
        """
        retriever = getattr(self, "content_retriever", None)
        if retriever is None:
            return None
        if content_id not in self._local_content:
            return None
        try:
            # Sprint 484 (F24): propagate timeout. asyncio.wait_for
            # is the defense-in-depth — even if retriever.fetch
            # has its own (None) default, wait_for forces a return.
            if timeout is not None:
                return await asyncio.wait_for(
                    retriever.fetch(
                        content_id, timeout=timeout,
                    ),
                    timeout=timeout,
                )
            return await retriever.fetch(content_id)
        except asyncio.TimeoutError:
            logger.debug(
                "BT fallback retrieve timed out for "
                "%s... (timeout=%ss)",
                content_id[:12], timeout,
            )
            return None
        except Exception as e:  # noqa: BLE001
            logger.error(
                f"BT fallback retrieve failed for {content_id[:12]}...: {e}"
            )
            return None

    async def _fetch_from_url(self, gateway_url: str) -> Optional[bytes]:
        """Fetch content via HTTP from a peer-supplied gateway URL.

        Used when a peer responds with TransferMode.GATEWAY instead of
        inline bytes. The URL scheme is whatever the peer publishes (e.g.
        ``prsm://`` for in-network gateways, ``http(s)://`` for HTTP-fronted
        nodes); only ``http``/``https`` are actually fetchable here.
        """
        if not gateway_url.startswith(("http://", "https://")):
            logger.debug(f"Skipping non-HTTP gateway URL: {gateway_url}")
            return None
        try:
            import aiohttp
            session = await self._get_http_session()
            async with session.get(
                gateway_url,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status == 200:
                    return await resp.read()
                else:
                    logger.debug(f"Gateway returned {resp.status} for {gateway_url}")
        except Exception as e:
            logger.error(f"Gateway fetch failed for {gateway_url}: {e}")
        return None
    
    # ── Phase 1.3 Task 3f: replica-serve delegation ───────────────────

    async def serve_on_behalf_of_replica(
        self,
        cid: str,
        request_id: str,
        peer: "PeerConnection",
        replica_size_bytes: int,
        gateway_url: str,
    ) -> bool:
        """Serve a CID that this provider does NOT have in _local_content
        but another subsystem (e.g., storage_provider) has pinned as a
        replica. Looks up creator metadata via content_index, calls
        content_economy.process_content_access to credit royalties,
        publishes GOSSIP_CONTENT_ACCESS, and sends a canonical
        ContentResponseMessage gateway response.

        Refuses to serve if content_index has no record for the CID —
        without creator metadata, we can't pay the creator, so we let
        the requester try another provider.

        Returns True if a response was sent, False if the serve was
        refused (no content_index record or other unrecoverable gap).
        Phase 1.3 Task 3f.
        """
        if self.content_index is None:
            logger.warning(
                f"serve_on_behalf_of_replica: no content_index configured; "
                f"refusing replica serve for {cid[:12]}..."
            )
            return False

        record = self.content_index.lookup(cid)
        if record is None:
            logger.warning(
                f"serve_on_behalf_of_replica: content_index has no record "
                f"for {cid[:12]}...; refusing replica serve (creator "
                f"metadata required for royalty payment)"
            )
            return False

        # Phase 1.3 Task 3g pass-5: refuse the serve when creator_id
        # is empty. A record whose only advertisement was a minimal
        # replica ad (no creator_id in payload) now surfaces
        # creator_id="" from the new-record path, replacing the
        # previous "default to origin peer id" lie. Without creator
        # metadata we can't pay royalties, so decline and let the
        # requester try another provider — mirrors the "no record"
        # refusal path above.
        if not record.creator_id:
            logger.warning(
                f"serve_on_behalf_of_replica: content_index record for "
                f"{cid[:12]}... has empty creator_id; refusing replica "
                f"serve (cannot pay royalties without creator metadata)"
            )
            return False

        # Build payment metadata directly from the ContentRecord —
        # we don't have a _local_content entry to resolve from, so we
        # go straight to the index fields. Mirrors the key mapping
        # _resolve_payment_metadata uses (parent_cids -> parent_cids).
        content_metadata = {
            "royalty_rate": record.royalty_rate,
            "creator_id": record.creator_id,
            "parent_cids": record.parent_cids,
            "provenance_hash": record.provenance_hash,
        }

        if self.content_economy is not None:
            try:
                payment = await self.content_economy.process_content_access(
                    content_id=cid,
                    accessor_id=peer.peer_id,
                    content_metadata=content_metadata,
                )
                if payment.status.value == "completed":
                    logger.debug(
                        f"Replica payment processed for {cid[:12]}... "
                        f"({payment.amount} FTNS from {peer.peer_id[:8]})"
                    )
                elif payment.status.value == "pending_onchain":
                    logger.info(
                        f"Replica payment pending on-chain reconciliation "
                        f"for {cid[:12]}... ({payment.amount} FTNS from "
                        f"{peer.peer_id[:8]})"
                    )
            except Exception as e:
                # Mirrors the inline serve path — payment failure is
                # logged but doesn't block the serve. The replica has
                # the bytes; payment is a separate concern.
                logger.warning(
                    f"Replica payment processing failed for {cid[:12]}...: {e}"
                )

        # Phase 1.3 Task 3b parity: publish GOSSIP_CONTENT_ACCESS so
        # source creators on other nodes can credit their shares. This
        # fires regardless of whether the payment call above succeeded
        # or fell back to local, matching the inline serve path.
        try:
            await self.gossip.publish(GOSSIP_CONTENT_ACCESS, sign_content_access_event({
                "cid": cid,
                "accessor_id": peer.peer_id,
                "creator_id": record.creator_id,
                "royalty_rate": record.royalty_rate,
                "parent_cids": record.parent_cids,
                "timestamp": time.time(),
                "access_nonce": uuid.uuid4().hex,  # sp899 replay-safe
            }, self.identity))
        except Exception as exc:
            logger.warning(
                f"GOSSIP_CONTENT_ACCESS publish failed for replica serve "
                f"of {cid[:12]}...: {exc}"
            )

        # Send the canonical response. Uses the same message shape
        # the inline gateway-transfer path emits so the requester's
        # ContentResponseMessage.from_payload() parses it as FOUND.
        response = ContentResponseMessage(
            request_id=request_id,
            cid=cid,
            status=ContentStatus.FOUND,
            size=replica_size_bytes,
            transfer_mode=TransferMode.GATEWAY,
            gateway_url=gateway_url,
            content_hash=record.content_hash,
            filename=record.filename or None,
        )
        await self._send_response(peer.peer_id, response)
        return True

    # ── Phase 1.2 test seam ────────────────────────────────────────────

    async def _fire_payment_for_test(
        self, cid: str, accessor_id: str
    ) -> Any:
        """Test seam: invoke the payment-on-access path directly. Used by
        tests/integration/test_onchain_provenance_e2e.py to verify
        provenance_hash forwarding without standing up the full P2P stack.

        Phase 1.3 Task 3b: mirrors the inline serve path by also
        publishing GOSSIP_CONTENT_ACCESS after the payment call, so
        tests for the gossip-after-serve contract exercise the same
        post-payment block as production.
        """
        if not self.content_economy:
            return None
        content_info = self._local_content.get(cid)
        if not content_info:
            return None
        resolved = self._resolve_payment_metadata(content_info)
        payment = await self.content_economy.process_content_access(
            content_id=cid,
            accessor_id=accessor_id,
            content_metadata=resolved,
        )
        try:
            await self.gossip.publish(GOSSIP_CONTENT_ACCESS, sign_content_access_event({
                "cid": cid,
                "accessor_id": accessor_id,
                "creator_id": resolved.get("creator_id", ""),
                "royalty_rate": resolved.get("royalty_rate", 0.01),
                "parent_cids": resolved.get("parent_cids", []),
                "timestamp": time.time(),
                "access_nonce": uuid.uuid4().hex,  # sp899 replay-safe
            }, self.identity))
        except Exception as exc:
            logger.warning(
                f"GOSSIP_CONTENT_ACCESS publish failed for "
                f"{cid[:12]}...: {exc}"
            )
        return payment

    def _resolve_payment_metadata(
        self, content_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build the content_metadata dict forwarded to
        content_economy.process_content_access from a _local_content record.

        Two-step lookup (top-level → nested → default) handles both
        register_local_content's nested shape and any legacy call site that
        set fields at the top level. Uses explicit None checks instead of
        `or`-chain coalescing so that semantically meaningful falsy values
        like royalty_rate=0.0 (free content) and creator_id="" are not
        silently replaced by defaults.
        """
        nested_metadata = content_info.get("metadata") or {}

        def _pick(key: str, default: Any) -> Any:
            value = content_info.get(key)
            if value is None:
                value = nested_metadata.get(key)
            if value is None:
                value = default
            return value

        return {
            "royalty_rate": _pick("royalty_rate", 0.01),
            "creator_id": _pick("creator_id", ""),
            "parent_cids": _pick("parent_cids", []),
            "provenance_hash": _pick("provenance_hash", None),
        }

    # ── Statistics ──────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics."""
        return {
            "local_content_count": len(self._local_content),
            "pending_requests": len(self._pending_requests),
            "discovery": self.content_discovery.get_stats(),
            **self._telemetry,
        }
