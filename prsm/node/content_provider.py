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
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
MAX_INLINE_SIZE = 1_048_576  # 1MB - content at/under this transfers INLINE (one frame)
MAX_CONCURRENT_REQUESTS = 10

# sp1020 — CHUNKED P2P transfer ceiling. Content between MAX_INLINE_SIZE and this
# bound is split into GATEWAY_FETCH_CHUNK_BYTES frames sent over the same P2P
# WebSocket substrate as INLINE (no libtorrent, no HTTP gateway). Above this bound
# the dead GATEWAY path still applies (a libtorrent swarm / real HTTP gateway is the
# right answer for truly huge content). The default is generous but finite — the
# whole body is buffered in memory on both ends, so this bounds peak memory; tune
# via PRSM_MAX_CHUNKED_TRANSFER_BYTES.
_DEFAULT_MAX_CHUNKED_TRANSFER_BYTES = 64 * 1024 * 1024  # 64 MiB

# sp1002 — GATEWAY content-fetch DoS bounds. A peer-supplied GATEWAY url is
# fetched over HTTP by the *retrieving* node; without an upper byte bound a
# malicious/over-large body OOMs the retriever (finding 7). The per-fetch cap
# is the advertised content size plus encoding slack (so "advertise tiny,
# serve huge" is rejected), and is itself bounded by this hard ceiling so an
# attacker advertising a giant size can never drive the cap unbounded. The
# default ceiling is generous (large legitimate content exists) but finite;
# operators can tune it via PRSM_MAX_GATEWAY_FETCH_BYTES.
_DEFAULT_GATEWAY_FETCH_CEILING_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
GATEWAY_FETCH_CHUNK_BYTES = 262_144  # 256 KiB streaming read granularity


def _gateway_fetch_hard_ceiling_bytes() -> int:
    """Hard upper bound (bytes) for a single GATEWAY HTTP fetch.

    Read from PRSM_MAX_GATEWAY_FETCH_BYTES at call time (so operators can
    tune without restart-coupling to import order); falls back to the
    2 GiB default on missing / non-positive / unparseable values.
    """
    raw = os.environ.get("PRSM_MAX_GATEWAY_FETCH_BYTES", "").strip()
    if not raw:
        return _DEFAULT_GATEWAY_FETCH_CEILING_BYTES
    try:
        val = int(raw)
    except (ValueError, TypeError):
        return _DEFAULT_GATEWAY_FETCH_CEILING_BYTES
    return val if val > 0 else _DEFAULT_GATEWAY_FETCH_CEILING_BYTES


def _classify_gateway_url(url: str, *, resolve=None):
    """sp1102 (Domain-06 review HIGH) — SSRF guard for the peer-supplied GATEWAY URL.

    A malicious provider can put any URL in a ContentResponseMessage. Before the
    retriever fetches it, validate: (1) scheme is http(s); (2) the host resolves ONLY
    to public IPs — reject any private / loopback / link-local / reserved / multicast /
    unspecified address (the AWS-metadata 169.254.169.254, 127.0.0.1 admin/RPC, RFC1918,
    etc.). Returns ``(safe: bool, reason: str, pinned_ips: list[str])``; the caller pins
    the connection to ``pinned_ips`` so a host that resolves safe here can't rebind to an
    internal address at connect time (DNS rebinding). ``resolve`` is injectable for tests.
    """
    import ipaddress
    import socket as _socket
    from urllib.parse import urlparse

    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return (False, "non-http-scheme", [])
    try:
        parsed = urlparse(url)
    except Exception:  # noqa: BLE001
        return (False, "unparseable", [])
    host = parsed.hostname
    if not host:
        return (False, "no-host", [])
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    _resolve = resolve or _socket.getaddrinfo
    try:
        infos = _resolve(host, port, type=_socket.SOCK_STREAM)
    except Exception:  # noqa: BLE001
        return (False, "resolve-failed", [])
    ips: List[str] = []
    for info in infos:
        try:
            ip = info[4][0].split("%")[0]  # strip any IPv6 scope id
            addr = ipaddress.ip_address(ip)
        except (ValueError, IndexError, TypeError):
            return (False, "bad-ip", [])
        if (
            addr.is_private or addr.is_loopback or addr.is_link_local
            or addr.is_reserved or addr.is_multicast or addr.is_unspecified
        ):
            return (False, f"blocked-ip:{ip}", [])
        ips.append(ip)
    if not ips:
        return (False, "no-ips", [])
    return (True, "ok", ips)


class _PinnedGatewayResolver:
    """aiohttp resolver that returns ONLY the pre-validated IPs for the gateway host,
    so the connection can't be rebound to an internal address between the safety check
    and connect. SNI/TLS still validate against the original hostname (from the URL)."""

    def __init__(self, ips: List[str]) -> None:
        self._ips = ips

    async def resolve(self, host: str, port: int = 0, family: int = 0):
        import socket as _socket
        out = []
        for ip in self._ips:
            fam = _socket.AF_INET6 if ":" in ip else _socket.AF_INET
            out.append({
                "hostname": host, "host": ip, "port": port,
                "family": fam, "proto": _socket.IPPROTO_TCP, "flags": 0,
            })
        return out

    async def close(self) -> None:
        return None


def _max_chunked_transfer_bytes() -> int:
    """sp1020 — upper bound (bytes) for CHUNKED P2P transfer, read at call time.

    Content above MAX_INLINE_SIZE and at/under this bound is chunked over the P2P
    substrate; above it the GATEWAY path applies. Falls back to the 64 MiB default
    on missing / non-positive / unparseable values.
    """
    raw = os.environ.get("PRSM_MAX_CHUNKED_TRANSFER_BYTES", "").strip()
    if not raw:
        return _DEFAULT_MAX_CHUNKED_TRANSFER_BYTES
    try:
        val = int(raw)
    except (ValueError, TypeError):
        return _DEFAULT_MAX_CHUNKED_TRANSFER_BYTES
    return val if val > 0 else _DEFAULT_MAX_CHUNKED_TRANSFER_BYTES


# sp1290 — STREAMING transfer ceiling. Content ABOVE the in-memory CHUNKED ceiling
# (_max_chunked_transfer_bytes) and at/under this bound is served by streaming ordered
# 256 KiB frames from the provider's on-disk staged file — with NO full-content
# materialization on EITHER side: the sender reads frame-by-frame from disk
# (_send_chunked_from_path), the requester writes frames straight to a destination file
# (request_content_to_file). This replaces the previously-DEAD GATEWAY response
# (prsm://content/{cid}, which nothing actually served) for large content, keeping big
# transfers decentralized + bounded + CID-verified. Tunable via
# PRSM_MAX_STREAMING_TRANSFER_BYTES (defaults to the 2 GiB gateway-fetch ceiling).
_DEFAULT_MAX_STREAMING_TRANSFER_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB


def _max_streaming_transfer_bytes() -> int:
    """sp1290 — upper bound (bytes) for the disk-streaming CHUNKED path, read at call
    time. Content above the in-memory CHUNKED ceiling and at/under this bound streams
    from/to disk; above it the GATEWAY fallback still applies. Falls back to the 2 GiB
    default on missing / non-positive / unparseable values. Always treated as at least
    the in-memory CHUNKED ceiling so the streaming band can never be empty/inverted."""
    raw = os.environ.get("PRSM_MAX_STREAMING_TRANSFER_BYTES", "").strip()
    val = _DEFAULT_MAX_STREAMING_TRANSFER_BYTES
    if raw:
        try:
            parsed = int(raw)
            if parsed > 0:
                val = parsed
        except (ValueError, TypeError):
            pass
    return max(val, _max_chunked_transfer_bytes())


class _StreamingSink:
    """sp1290 — receive-side sink that writes CHUNKED (or a single INLINE) frame
    straight to a destination file, bounded, with NO full-content buffering. Backs
    ``ContentProvider.request_content_to_file`` so content too large to hold in memory
    (the previously-dead >64 MiB GATEWAY case) transfers end-to-end over the P2P
    substrate. Frame validation mirrors the in-memory ``_accumulate_chunk`` guards
    (per-frame size cap, declared-count ceiling, pinned total, in-range index). The
    CID/hash check is run by the async caller AFTER completion via a streaming read,
    not here, so the (synchronous) message handler never blocks re-hashing a multi-GiB
    file. Frames are written at their ``index * chunk_size`` offset, so out-of-order
    delivery is handled."""

    def __init__(self, dest_path: Path, chunk_size: int, ceiling: int):
        self.dest_path = dest_path
        self.chunk_size = chunk_size
        self.ceiling = ceiling
        self.total: Optional[int] = None
        self.received: Set[int] = set()
        self.bytes_written = 0
        self.closed = False
        self._fh = open(dest_path, "wb")

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                self._fh.close()
            except OSError:
                pass

    def _fail(self) -> str:
        self.close()
        try:
            self.dest_path.unlink()
        except OSError:
            pass
        return "error"

    def add_inline(self, data: bytes) -> str:
        """Write a single-frame (INLINE) body and complete."""
        if len(data) > self.ceiling:
            return self._fail()
        try:
            self._fh.seek(0)
            self._fh.write(data)
        except OSError:
            return self._fail()
        self.bytes_written = len(data)
        self.close()
        return "complete"

    def add_frame(self, response) -> str:
        """Write one CHUNKED frame. Returns 'pending' | 'complete' | 'error'."""
        total = response.total_chunks
        index = response.chunk_index
        data = response.data
        if (not isinstance(total, int) or total <= 0
                or not isinstance(index, int) or not (0 <= index < total)
                or data is None):
            return self._fail()
        # Per-frame size cap (disk-amplification defense): a legitimate frame never
        # exceeds the protocol chunk size.
        if len(data) > self.chunk_size:
            return self._fail()
        # Declared COUNT must not imply a body over the streaming ceiling.
        if total * self.chunk_size > self.ceiling + self.chunk_size:
            return self._fail()
        if self.total is None:
            self.total = total
        elif self.total != total:
            # total is pinned by the first frame; a disagreeing total is malformed.
            return self._fail()
        if index in self.received:
            # Duplicate frame — idempotent, don't double-count bytes.
            return "complete" if len(self.received) >= self.total else "pending"
        try:
            self._fh.seek(index * self.chunk_size)
            self._fh.write(data)
        except OSError:
            return self._fail()
        self.received.add(index)
        self.bytes_written += len(data)
        if self.bytes_written > self.ceiling:
            return self._fail()
        if len(self.received) >= self.total:
            self.close()
            return "complete"
        return "pending"


class ContentStatus(str, Enum):
    """Status of content retrieval."""
    FOUND = "found"
    NOT_FOUND = "not_found"
    ERROR = "error"
    TIMEOUT = "timeout"


class TransferMode(str, Enum):
    """How content is transferred."""
    INLINE = "inline"  # Base64 encoded in a single response
    CHUNKED = "chunked"  # sp1020: split into ordered base64 frames over the P2P substrate
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
    # sp1020 — CHUNKED transfer: this frame's index + the total frame count for
    # the content. None for INLINE/GATEWAY (single-frame) responses.
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None

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

        # INLINE carries the whole body; CHUNKED carries this frame's slice. Both
        # ride the same base64-over-WebSocket path.
        if self.data is not None and self.transfer_mode in (TransferMode.INLINE, TransferMode.CHUNKED):
            payload["data_b64"] = base64.b64encode(self.data).decode()

        if self.transfer_mode == TransferMode.CHUNKED:
            if self.chunk_index is not None:
                payload["chunk_index"] = self.chunk_index
            if self.total_chunks is not None:
                payload["total_chunks"] = self.total_chunks

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
        if data_b64 and transfer_mode in (TransferMode.INLINE, TransferMode.CHUNKED):
            try:
                data = base64.b64decode(data_b64)
            except Exception:
                pass

        chunk_index = payload.get("chunk_index")
        total_chunks = payload.get("total_chunks")

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
            chunk_index=chunk_index if isinstance(chunk_index, int) else None,
            total_chunks=total_chunks if isinstance(total_chunks, int) else None,
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

        # sp1020 — CHUNKED reassembly buffers, keyed by request_id. Only ever
        # populated for OUR outstanding requests (a chunk for an unknown
        # request_id is dropped), so an attacker cannot drive unbounded growth.
        # Cleaned up alongside _pending_requests when the request completes.
        self._pending_chunks: Dict[str, Dict[str, Any]] = {}

        # sp1290 — receive-side streaming-to-file sinks, keyed by request_id, populated
        # only for our own request_content_to_file calls (a frame for an unknown
        # request_id is dropped). Each writes frames straight to disk instead of the
        # in-memory _pending_chunks buffer, so large content never materializes here.
        self._streaming_sinks: Dict[str, "_StreamingSink"] = {}

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
        self._pending_chunks.clear()  # sp1020 — drop any in-flight reassembly buffers
        for _sink in self._streaming_sinks.values():  # sp1290 — close any open file sinks
            _sink.close()
        self._streaming_sinks.clear()
    
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
            # sp1290 — LARGE content with an on-disk staged copy streams from disk in
            # CHUNKED frames WITHOUT materializing the whole file in memory (this is the
            # case that previously returned the dead prsm://content/{cid} GATEWAY URL).
            # The common path below (<= the in-memory CHUNKED ceiling, or no streamable
            # path) is unchanged — INLINE/CHUNKED/GATEWAY behave byte-identically.
            served = False
            size = 0
            stream_info = self._local_stream_info(cid)
            if stream_info is not None:
                spath, ssize = stream_info
                if _max_chunked_transfer_bytes() < ssize <= _max_streaming_transfer_bytes():
                    if self.bandwidth_limiter:
                        await self.bandwidth_limiter.throttle_upload(ssize)
                    await self._send_chunked_from_path(
                        peer.peer_id, request_id, cid, spath, ssize, content_info,
                    )
                    size = ssize
                    served = True

            if not served:
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
                    # Send inline (base64 encoded) — single frame.
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
                    await self._send_response(peer.peer_id, response)
                elif size <= _max_chunked_transfer_bytes():
                    # sp1020 — CHUNKED: split into ordered 256 KiB frames over the
                    # same P2P substrate (no libtorrent, no HTTP gateway). The
                    # requester reassembles + re-verifies against the CID/hash.
                    await self._send_chunked_response(
                        peer.peer_id, request_id, cid, content_bytes, content_info,
                    )
                else:
                    # Beyond BOTH the in-memory chunked ceiling AND the disk-streaming
                    # band (or no on-disk staged copy to stream) — fall back to the
                    # gateway URL (a real HTTP gateway / libtorrent swarm).
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

    async def _send_chunked_response(
        self,
        peer_id: str,
        request_id: str,
        cid: str,
        content_bytes: bytes,
        content_info: Dict[str, Any],
    ) -> None:
        """sp1020 — serve content above MAX_INLINE_SIZE as ordered CHUNKED frames
        over the P2P substrate. Each frame carries its 256 KiB slice + chunk_index
        + the total_chunks count + the full content size/hash, so the requester
        reassembles in index order and re-verifies the result against the CID/hash.
        The whole body was already bandwidth-throttled once by the caller."""
        size = len(content_bytes)
        chunk_size = GATEWAY_FETCH_CHUNK_BYTES
        total_chunks = (size + chunk_size - 1) // chunk_size
        content_hash = content_info.get("content_hash")
        filename = content_info.get("filename")
        for index in range(total_chunks):
            start = index * chunk_size
            response = ContentResponseMessage(
                request_id=request_id,
                cid=cid,
                status=ContentStatus.FOUND,
                data=content_bytes[start:start + chunk_size],
                size=size,
                transfer_mode=TransferMode.CHUNKED,
                content_hash=content_hash,
                filename=filename,
                chunk_index=index,
                total_chunks=total_chunks,
            )
            await self._send_response(peer_id, response)

    def _local_stream_info(self, cid: str) -> Optional[Tuple[Path, int]]:
        """sp1290 — return (path, size) of a locally-staged on-disk file for ``cid``
        WITHOUT reading it into memory, or None if no streamable single-file copy
        exists. Drives the disk-streaming send path for large content. Only the
        publisher's staged-file path qualifies (the libtorrent-free LocalContentPublisher
        / ContentPublisher staging that ``_fetch_local_via_bt`` already trusts); a
        ContentStore-only blob has no stable single-file path here and keeps the
        in-memory path."""
        pub = getattr(self, "content_publisher", None)
        if pub is None:
            return None
        try:
            staged = pub.local_publish_path(cid)
        except Exception:  # noqa: BLE001 — path probing must never break a request
            return None
        if staged is None:
            return None
        try:
            p = Path(staged)
            if p.is_file():
                return p, p.stat().st_size
        except OSError:
            return None
        return None

    async def _send_chunked_from_path(
        self,
        peer_id: str,
        request_id: str,
        cid: str,
        path: Path,
        size: int,
        content_info: Dict[str, Any],
    ) -> None:
        """sp1290 — serve a large local file as ordered CHUNKED frames streamed from
        disk, WITHOUT materializing the whole file in memory. The frame wire format is
        identical to ``_send_chunked_response`` (256 KiB slices + chunk_index +
        total_chunks + size/hash), so an in-memory requester reassembles it the same
        way and a ``request_content_to_file`` requester writes it straight to disk.
        Disk reads run in the default executor so the event loop is never blocked on
        IO; only one 256 KiB frame is resident at a time."""
        chunk_size = GATEWAY_FETCH_CHUNK_BYTES
        total_chunks = (size + chunk_size - 1) // chunk_size
        content_hash = content_info.get("content_hash")
        filename = content_info.get("filename")
        loop = asyncio.get_event_loop()

        fh = await loop.run_in_executor(None, lambda: open(path, "rb"))
        try:
            index = 0
            while index < total_chunks:
                data = await loop.run_in_executor(None, fh.read, chunk_size)
                if not data:
                    break
                response = ContentResponseMessage(
                    request_id=request_id,
                    cid=cid,
                    status=ContentStatus.FOUND,
                    data=data,
                    size=size,
                    transfer_mode=TransferMode.CHUNKED,
                    content_hash=content_hash,
                    filename=filename,
                    chunk_index=index,
                    total_chunks=total_chunks,
                )
                await self._send_response(peer_id, response)
                index += 1
        finally:
            await loop.run_in_executor(None, fh.close)

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

    def _get_expected_size(self, cid: str) -> Optional[int]:
        """Get the advertised content size (bytes) for a CID, if known.

        sp1002 — used to bound a GATEWAY fetch: a provider that advertised
        N bytes must not be able to then serve an unbounded body. Returns
        None when no positive size is known (caller falls back to the hard
        ceiling). The value is gossip-supplied (untrusted) but using it only
        ever TIGHTENS the cap below the hard ceiling, so a lie can shrink the
        bound (rejecting the lie's oversized body) but never enlarge it.
        """
        if self.content_index:
            record = self.content_index.lookup(cid)
            size = getattr(record, "size_bytes", None) if record else None
            if isinstance(size, int) and size > 0:
                return size
        info = self.content_discovery.get_content_info(cid)
        if info is not None:
            size = getattr(info, "size", None)
            if isinstance(size, int) and size > 0:
                return size
        return None

    @staticmethod
    def _cid_anchor_rejects(cid: str, content_bytes: bytes) -> bool:
        """Return True iff ``cid`` is a RECOMPUTABLE content identity AND
        ``content_bytes`` does NOT match it (a content-substitution attempt that
        must be rejected). Two anchors:

        * sp1003 — a canonical ContentHash (algorithm-prefixed sha256-family) CID:
          re-hash the bytes and compare.
        * sp1076 — a 40-hex BitTorrent v1 infohash CID: the infohash IS recomputable
          from the bytes (``prsm.core.torrent_infohash``), so re-derive it and
          compare. This closes the residual gap the sp1003 docstring noted — a
          malicious provider can no longer substitute bytes for an infohash CID
          (the only prior check, the gossip-supplied ``expected_hash``, is
          attacker-controllable). Both INLINE and CHUNKED fetches funnel through
          here (the chunked reassembly re-verifies via ``_request_from_provider``).

        Returns False (no opinion → caller falls back to the gossip-hash check) when
        ``cid`` is neither — i.e. not a recomputable identity.
        """
        ch_verdict = ContentProvider._contenthash_anchor_verdict(cid, content_bytes)
        if ch_verdict is not None:
            return ch_verdict
        return ContentProvider._infohash_anchor_rejects(cid, content_bytes)

    @staticmethod
    def _contenthash_anchor_verdict(cid: str, content_bytes: bytes):
        """sp1003 ContentHash anchor. Returns True/False for a CANONICAL ContentHash
        CID (reject / accept), or None when ``cid`` is not an unambiguous ContentHash
        (defer — strict round-trip + digest-length guard, so a BT infohash that
        happens to parse as an algorithm id falls through rather than false-rejects)."""
        try:
            from prsm.storage import ContentHash
        except Exception:  # pragma: no cover - storage always importable
            return None
        try:
            parsed = ContentHash.from_hex(cid)
        except (ValueError, TypeError):
            return None
        try:
            canonical = parsed.hex() == cid
            expected_len = len(
                ContentHash.from_data(b"", parsed.algorithm_id).digest
            )
        except (ValueError, TypeError):
            return None
        if not canonical or len(parsed.digest) != expected_len:
            return None
        recomputed = ContentHash.from_data(content_bytes, parsed.algorithm_id)
        return recomputed.digest != parsed.digest

    @staticmethod
    def _infohash_anchor_rejects(cid: str, content_bytes: bytes) -> bool:
        """sp1076 — True iff ``cid`` is a 40-hex BitTorrent v1 infohash and
        ``content_bytes`` does not re-derive to it. False (defer) for any other CID
        shape or on a computation error (fail-OPEN to the gossip-hash check, never a
        false reject of a non-infohash CID)."""
        if not (isinstance(cid, str) and len(cid) == 40):
            return False
        low = cid.lower()
        if any(c not in "0123456789abcdef" for c in low):
            return False
        try:
            import hashlib
            from prsm.core.torrent_infohash import compute_v1_infohash_single_file
            name = hashlib.sha256(content_bytes).hexdigest()
            return compute_v1_infohash_single_file(content_bytes, name) != low
        except Exception:  # noqa: BLE001 - can't recompute → defer, don't false-reject
            return False

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

        # sp1002 (finding 8) — track elapsed so the GATEWAY HTTP fetch below
        # runs under the caller's REMAINING timeout budget, not a fixed 120s
        # outside it (a slow-loris gateway otherwise tied up a worker for
        # 120s regardless of the caller's smaller timeout).
        _t0 = time.monotonic()

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
            self._pending_chunks.pop(request_id, None)  # sp1020 — drop any partial reassembly

        # Process response
        if response.status != ContentStatus.FOUND:
            logger.debug(f"Provider {provider_id[:8]} returned {response.status}")
            return None

        # Get content bytes
        content_bytes: Optional[bytes] = None

        # INLINE carries the body directly; CHUNKED arrives reassembled (see
        # _handle_content_response). Both then pass the same CID/hash verification.
        if response.transfer_mode in (TransferMode.INLINE, TransferMode.CHUNKED) and response.data is not None:
            content_bytes = response.data

        elif response.transfer_mode == TransferMode.GATEWAY and response.gateway_url:
            # sp1002 (findings 7+8) — bound the peer-supplied gateway fetch.
            # Cap bytes at the advertised size (+ encoding slack) bounded by
            # the hard ceiling, and give the HTTP fetch only the caller's
            # remaining timeout budget.
            expected_size = self._get_expected_size(cid)
            ceiling = _gateway_fetch_hard_ceiling_bytes()
            if expected_size:
                max_bytes = min(expected_size * 2 + 4096, ceiling)
            else:
                max_bytes = ceiling
            remaining = max(0.5, timeout - (time.monotonic() - _t0))
            content_bytes = await self._fetch_from_url(
                response.gateway_url,
                timeout=remaining,
                max_bytes=max_bytes,
            )

        if content_bytes is None:
            return None

        # sp1003 — CID-anchored integrity (content-substitution defense).
        # For a ContentHash-shaped CID (an algorithm-prefixed content
        # address) verify the returned bytes against the CID ITSELF. The
        # requester chose+trusts the CID, whereas `expected_hash` came from
        # an UNAUTHENTICATED gossip advertise and is attacker-controllable
        # (a malicious provider can advertise content_hash=sha256(evil),
        # serve evil, and pass the gossip-hash check below). The ContentStore
        # is content-addressed, so genuine content always re-hashes to its
        # CID. A 40/64-char BitTorrent infohash is NOT a recomputable content
        # address from inline bytes (documented residual gap in
        # docs/2026-06-04-content-data-plane-trust-anchors.md) — it fails the
        # round-trip/digest-length guard and falls through to the gossip-hash
        # check.
        if self._cid_anchor_rejects(cid, content_bytes):
            logger.warning(
                "CID-anchor mismatch for %s... from %s — refusing "
                "substituted content (bytes do not hash to the requested CID)",
                cid[:12], provider_id[:8],
            )
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
        if future is None or future.done():
            # Unknown / already-resolved request — drop. This also prevents an
            # attacker from buffering CHUNKED frames under a request_id we never
            # issued (unbounded-memory defense).
            self._pending_chunks.pop(request_id, None)
            self._streaming_sinks.pop(request_id, None)
            return

        # sp1290 — a request_content_to_file request routes its frames straight to a
        # disk sink (bounded, no in-memory buffering) instead of the _accumulate_chunk
        # path below. The future resolves with the dest Path on completion (the async
        # caller then streaming-verifies the CID/hash), or None on a malformed/failed
        # transfer. Non-streaming requests are unaffected.
        sink = self._streaming_sinks.get(request_id)
        if sink is not None:
            self._route_to_sink(request_id, response, sink, future)
            return

        if response.transfer_mode == TransferMode.CHUNKED:
            assembled = self._accumulate_chunk(request_id, response)
            if assembled is None:
                return  # more frames pending (or a malformed frame discarded it)
            # Resolve with one synthetic response carrying the full reassembled
            # body; _request_from_provider then re-verifies it against the CID/hash.
            self._pending_chunks.pop(request_id, None)
            if not future.done():
                future.set_result(ContentResponseMessage(
                    request_id=request_id,
                    cid=response.cid,
                    status=ContentStatus.FOUND,
                    data=assembled,
                    size=len(assembled),
                    transfer_mode=TransferMode.CHUNKED,
                    content_hash=response.content_hash,
                    filename=response.filename,
                ))
            return

        # Single-frame response (INLINE / GATEWAY / not-found / error).
        future.set_result(response)

    def _accumulate_chunk(
        self, request_id: str, response: ContentResponseMessage,
    ) -> Optional[bytes]:
        """sp1020 — buffer one CHUNKED frame; return the reassembled bytes once
        every frame has arrived, else None. Returns None and discards the
        in-flight transfer on any malformed or over-bound frame, so a misbehaving
        provider fails the request safely (future times out → None) rather than
        corrupting or OOMing the requester. Content integrity of a well-formed-but-
        wrong reassembly is caught downstream by the CID/hash check."""
        total = response.total_chunks
        index = response.chunk_index
        data = response.data
        if (not isinstance(total, int) or total <= 0
                or not isinstance(index, int) or not (0 <= index < total)
                or data is None):
            self._pending_chunks.pop(request_id, None)
            return None

        # Per-frame size cap: a legitimate frame never exceeds the protocol chunk
        # size. Without this a malicious provider declares a SMALL total_chunks
        # (passing the count guard below) then sends giant frames (the transport
        # permits multi-MiB messages), buffering many times the ceiling before the
        # post-reassembly integrity check can run — a memory-amplification OOM.
        if len(data) > GATEWAY_FETCH_CHUNK_BYTES:
            logger.warning(
                "CHUNKED frame for %s is %d bytes > the %d-byte chunk size — "
                "refusing (memory-amplification defense)",
                request_id, len(data), GATEWAY_FETCH_CHUNK_BYTES,
            )
            self._pending_chunks.pop(request_id, None)
            return None

        # The declared frame COUNT must not imply a body over the chunked ceiling.
        ceiling = _max_chunked_transfer_bytes()
        if total * GATEWAY_FETCH_CHUNK_BYTES > ceiling + GATEWAY_FETCH_CHUNK_BYTES:
            logger.warning(
                "CHUNKED transfer %s declares %d frames exceeding the ceiling "
                "— refusing", request_id, total,
            )
            self._pending_chunks.pop(request_id, None)
            return None

        buf = self._pending_chunks.get(request_id)
        if buf is None:
            buf = {"total": total, "chunks": {}, "bytes": 0}
            self._pending_chunks[request_id] = buf
        elif buf["total"] != total:
            # total_chunks is pinned by the first frame; a disagreeing total is a
            # malformed/inconsistent transfer — fail fast rather than re-keying the
            # buffer (which would stall the request to its full timeout).
            logger.warning(
                "CHUNKED transfer %s sent inconsistent total_chunks (%d != %d) "
                "— refusing", request_id, total, buf["total"],
            )
            self._pending_chunks.pop(request_id, None)
            return None

        if index not in buf["chunks"]:
            buf["bytes"] += len(data)
            # Running buffered-byte guard (defense-in-depth alongside the per-frame
            # + count caps): never hold more than one ceiling's worth per request.
            if buf["bytes"] > ceiling + GATEWAY_FETCH_CHUNK_BYTES:
                logger.warning(
                    "CHUNKED transfer %s exceeded the %d-byte buffer ceiling "
                    "— refusing", request_id, ceiling,
                )
                self._pending_chunks.pop(request_id, None)
                return None
        buf["chunks"][index] = data

        if len(buf["chunks"]) < total:
            return None
        return b"".join(buf["chunks"][i] for i in range(total))

    def _route_to_sink(self, request_id, response, sink, future) -> None:
        """sp1290 — feed one response frame to a streaming-to-file sink and resolve the
        request's future when the transfer completes (with the dest Path) or fails (with
        None). Cleans up the sink registration on any terminal outcome."""
        if response.status != ContentStatus.FOUND:
            sink.close()
            try:
                sink.dest_path.unlink()
            except OSError:
                pass
            self._streaming_sinks.pop(request_id, None)
            if not future.done():
                future.set_result(None)
            return

        if response.transfer_mode == TransferMode.CHUNKED:
            status = sink.add_frame(response)
        elif response.transfer_mode == TransferMode.INLINE and response.data is not None:
            status = sink.add_inline(response.data)
        else:
            # GATEWAY (or an unexpected mode) — a streaming sink can't consume it.
            status = sink._fail()

        if status == "pending":
            return
        self._streaming_sinks.pop(request_id, None)
        if not future.done():
            future.set_result(sink.dest_path if status == "complete" else None)

    async def request_content_to_file(
        self,
        cid: str,
        dest_path,
        timeout: Optional[float] = None,
        verify_hash: bool = True,
        preferred_peer: Optional[str] = None,
    ) -> Optional[Path]:
        """sp1290 — like ``request_content`` but streams the body to ``dest_path``
        instead of returning it in memory, so content too large to buffer (the
        previously-dead >64 MiB GATEWAY case) transfers end-to-end over the P2P
        substrate without materializing on either side. Returns the destination Path on
        success (CID/hash re-verified by a streaming read) or None on failure. The
        partial file is removed on any failure."""
        timeout = timeout or self.default_timeout
        dest_path = Path(dest_path)

        # Local shortcut: if we hold a streamable on-disk copy, stream-copy it.
        stream_info = self._local_stream_info(cid)
        if stream_info is not None:
            spath, _ = stream_info
            try:
                await self._copy_file_streaming(spath, dest_path)
            except OSError:
                dest_path.unlink(missing_ok=True)
            else:
                if await self._verify_file_cid(cid, dest_path, verify_hash):
                    logger.debug("Streamed %s locally → %s", cid[:12], dest_path)
                    return dest_path
                dest_path.unlink(missing_ok=True)

        providers = self._find_providers(cid)
        providers.discard(self.identity.node_id)
        if not providers:
            logger.debug("No providers found for %s (to-file)", cid[:12])
            return None
        if preferred_peer and preferred_peer in providers:
            providers_list = [preferred_peer] + [p for p in providers if p != preferred_peer]
        else:
            providers_list = list(providers)

        expected_hash = self._get_content_hash(cid) if verify_hash else None
        for provider_id in providers_list:
            result = await self._request_to_file_from_provider(
                cid, provider_id, timeout, dest_path, expected_hash, verify_hash,
            )
            if result is not None:
                self._telemetry["requests_fulfilled"] += 1
                return result
        return None

    async def _request_to_file_from_provider(
        self, cid, provider_id, timeout, dest_path, expected_hash, verify_hash,
    ) -> Optional[Path]:
        """sp1290 — send a content request to one provider and stream the response to
        ``dest_path`` via a _StreamingSink; verify (streaming) before returning."""
        self._telemetry["requests_sent"] += 1
        request = ContentRequestMessage(
            cid=cid, timeout=int(timeout), requester_id=self.identity.node_id,
        )
        request_id = request.request_id
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_requests[request_id] = future
        self._streaming_sinks[request_id] = _StreamingSink(
            dest_path, GATEWAY_FETCH_CHUNK_BYTES, _max_streaming_transfer_bytes(),
        )

        result_path: Optional[Path] = None
        try:
            msg = P2PMessage(
                msg_type=MSG_DIRECT,
                sender_id=self.identity.node_id,
                payload=request.to_payload(),
            )
            sent = await self.transport.send_to_peer(provider_id, msg)
            if not sent:
                logger.debug("Failed to send to-file request to %s", provider_id[:8])
            else:
                result_path = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._telemetry["requests_timed_out"] += 1
            logger.debug("To-file request to %s timed out for %s", provider_id[:8], cid[:12])
        except Exception as e:  # noqa: BLE001
            logger.debug("To-file request to %s failed: %s", provider_id[:8], e)
        finally:
            self._pending_requests.pop(request_id, None)
            leftover = self._streaming_sinks.pop(request_id, None)
            if leftover is not None:
                leftover.close()

        if result_path is None:
            Path(dest_path).unlink(missing_ok=True)
            return None
        if await self._verify_file_cid(cid, dest_path, verify_hash, expected_hash):
            return Path(dest_path)
        logger.warning("Streamed %s failed CID/hash verification — discarding", cid[:12])
        Path(dest_path).unlink(missing_ok=True)
        return None

    async def _copy_file_streaming(self, src: Path, dest: Path) -> None:
        """sp1290 — copy src→dest in 1 MiB blocks off the event loop (no full read)."""
        loop = asyncio.get_event_loop()

        def _copy():
            with open(src, "rb") as fin, open(dest, "wb") as fout:
                for block in iter(lambda: fin.read(1024 * 1024), b""):
                    fout.write(block)

        await loop.run_in_executor(None, _copy)

    async def _verify_file_cid(
        self, cid: str, path, verify_hash: bool, expected_hash: Optional[str] = None,
    ) -> bool:
        """sp1290 — streaming CID/hash verification of a received file (executor read,
        never blocks the loop, never materializes the body). Verifies the file's sha256
        against (a) a canonical sha256-family ContentHash CID — the strongest anchor,
        the requester chose the CID — else (b) the gossip-supplied ``expected_hash``
        when present. Accepts when neither anchor is available (e.g. a BitTorrent
        infohash CID, whose streaming re-derivation is a documented follow-on) — matching
        how unknown-hash content is treated elsewhere. Returns False on any IO error."""
        if not verify_hash:
            return True
        loop = asyncio.get_event_loop()

        def _sha256() -> Optional[str]:
            try:
                h = hashlib.sha256()
                with open(path, "rb") as fh:
                    for block in iter(lambda: fh.read(1024 * 1024), b""):
                        h.update(block)
                return h.hexdigest()
            except OSError:
                return None

        digest = await loop.run_in_executor(None, _sha256)
        if digest is None:
            return False

        # (a) canonical sha256-family ContentHash CID → the file's sha256 must equal
        # the CID's embedded digest (the sp1003 anchor, streaming-friendly).
        try:
            from prsm.storage import ContentHash
            parsed = ContentHash.from_hex(cid)
            if parsed.hex() == cid:
                algo = getattr(parsed, "algorithm", "") or ""
                if "sha256" in str(algo).lower() or parsed.digest.hex() == digest:
                    return parsed.digest.hex() == digest
        except Exception:  # noqa: BLE001 — not a ContentHash CID, fall through
            pass

        # (b) gossip-supplied expected hash, when known.
        if expected_hash:
            return digest == expected_hash
        return True

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

    def _open_gateway_session(self, pinned_ips: List[str]) -> Any:
        """sp1102 — async-context-manager session for ONE untrusted gateway fetch,
        pinned to the SSRF-validated IP(s) so the connection can't be rebound to an
        internal address (DNS rebinding) between the safety check and connect. A one-off
        session (not the shared pool) isolates untrusted fetches. Overridable in tests."""
        import aiohttp
        connector = aiohttp.TCPConnector(
            resolver=_PinnedGatewayResolver(pinned_ips), ttl_dns_cache=0)
        return aiohttp.ClientSession(connector=connector)

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
        if content_id not in getattr(self, "_local_content", {}):
            return None
        # sp1073 — direct-publisher shortcut FIRST (libtorrent-free). If a publisher is
        # wired and it locally published this cid as a Tier-A staged file, serve those
        # bytes with NO BT retriever. This is what lets a non-libtorrent node
        # (sp1071 LocalContentPublisher) self-fetch + serve its own uploads; it also
        # short-circuits the BT swarm for the libtorrent ContentPublisher's own Tier-A
        # content (the F8 shortcut, now reachable without a retriever).
        pub = getattr(self, "content_publisher", None)
        if pub is not None:
            try:
                staged = pub.local_publish_path(content_id)
                if staged is not None and staged.is_file():
                    return staged.read_bytes()
            except Exception:  # noqa: BLE001 - fall through to the retriever path
                pass
        retriever = getattr(self, "content_retriever", None)
        if retriever is None:
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

    async def _fetch_from_url(
        self,
        gateway_url: str,
        timeout: Optional[float] = None,
        max_bytes: Optional[int] = None,
    ) -> Optional[bytes]:
        """Fetch content via HTTP from a peer-supplied gateway URL.

        Used when a peer responds with TransferMode.GATEWAY instead of
        inline bytes. The URL scheme is whatever the peer publishes (e.g.
        ``prsm://`` for in-network gateways, ``http(s)://`` for HTTP-fronted
        nodes); only ``http``/``https`` are actually fetchable here.

        sp1002 — the URL and the served body are both attacker-controlled
        (ContentResponseMessage.gateway_url comes verbatim from the peer's
        response). Two DoS guards:

          - ``max_bytes`` (finding 7): refuse early when the advertised
            Content-Length already exceeds the cap, and stream the body in
            chunks, aborting the moment the accumulated size passes the cap.
            Prevents an unbounded / much-larger-than-advertised body from
            OOM-ing the retriever. Falls back to the hard ceiling when not
            supplied by the caller.
          - ``timeout`` (finding 8): the HTTP fetch runs under the caller's
            remaining retrieve budget, not a fixed 120s outside it.
        """
        # sp1102 (Domain-06 review HIGH) — SSRF guard. The gateway_url is attacker-
        # controllable (any peer can reply GATEWAY with an internal URL). Reject any
        # host that resolves to a private/loopback/link-local/reserved address BEFORE
        # issuing the request, pin the connection to the validated IP(s) (anti-rebinding),
        # and disable redirects (so a public URL can't 302 to an internal one).
        safe, reason, pinned_ips = _classify_gateway_url(gateway_url)
        if not safe:
            logger.warning(
                "Refusing GATEWAY fetch to unsafe URL %s (%s) — SSRF guard",
                gateway_url, reason,
            )
            return None
        cap = max_bytes if (max_bytes and max_bytes > 0) else _gateway_fetch_hard_ceiling_bytes()
        eff_timeout = timeout if (timeout and timeout > 0) else self.default_timeout
        try:
            import aiohttp
            async with self._open_gateway_session(pinned_ips) as session, session.get(
                gateway_url,
                timeout=aiohttp.ClientTimeout(total=eff_timeout),
                allow_redirects=False,
            ) as resp:
                if resp.status != 200:
                    logger.debug(f"Gateway returned {resp.status} for {gateway_url}")
                    return None
                # Fast-fail when the advertised length already exceeds the
                # cap (untrusted, but a cheap early reject for honest servers
                # that set it). Don't trust it for sizing — still stream-cap.
                declared = resp.headers.get("Content-Length")
                if declared is not None:
                    try:
                        if int(declared) > cap:
                            logger.warning(
                                "Gateway %s advertises %s bytes > cap %s — refusing fetch",
                                gateway_url, declared, cap,
                            )
                            return None
                    except (ValueError, TypeError):
                        pass
                buf = bytearray()
                async for chunk in resp.content.iter_chunked(GATEWAY_FETCH_CHUNK_BYTES):
                    buf.extend(chunk)
                    if len(buf) > cap:
                        logger.warning(
                            "Gateway %s exceeded byte cap %s mid-stream — aborting fetch",
                            gateway_url, cap,
                        )
                        return None
                return bytes(buf)
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
