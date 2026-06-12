"""
Bootstrap Server Data Models

Defines data structures for peer information, status tracking,
and bootstrap server metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import uuid4


class PeerStatus(Enum):
    """Status of a peer in the bootstrap network."""
    
    ACTIVE = "active"
    """Peer is actively connected and responding."""
    
    IDLE = "idle"
    """Peer is connected but not recently active."""
    
    UNREACHABLE = "unreachable"
    """Peer is not responding to health checks."""
    
    DISCONNECTED = "disconnected"
    """Peer has disconnected from the network."""
    
    BANNED = "banned"
    """Peer has been banned due to malicious behavior."""


@dataclass
class PeerInfo:
    """
    Information about a peer in the P2P network.
    
    Tracks connection details, capabilities, and status for each peer
    that connects to the bootstrap server.
    """
    
    peer_id: str
    """Unique identifier for the peer."""
    
    address: str
    """Network address (IP or hostname) of the peer."""
    
    port: int
    """Port number the peer is listening on."""
    
    public_key: Optional[str] = None
    """Peer's public key for encrypted communications."""
    
    status: PeerStatus = PeerStatus.ACTIVE
    """Current status of the peer."""
    
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """Timestamp when peer first connected."""
    
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """Timestamp of last activity from peer."""
    
    capabilities: List[str] = field(default_factory=list)
    """List of capabilities this peer offers (e.g., 'compute', 'storage')."""
    
    region: Optional[str] = None
    """Geographic region of the peer for routing optimization."""
    
    version: Optional[str] = None
    """PRSM client version of the peer."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the peer."""

    # Sprint 838 — hardware_profile relay. Cold-start joiners
    # connecting only to the bootstrap-server need to see real
    # fleet capacity (tflops_fp16, ram_total_gb, etc.) so the
    # DHT-backed pool can route through actual hardware. Pre-838
    # bootstrap only forwarded peer_id+address+capabilities; the
    # hw advertisement traveled in DISCOVERY_ANNOUNCE between
    # peers but never reached operators that hadn't directly
    # contacted the advertiser yet (NAT/firewall blocked). The
    # bootstrap caches whatever profile each peer sent at
    # registration time + re-broadcasts in peer-list responses.
    hardware_profile: Optional[Dict[str, Any]] = None
    """Hardware profile relayed from the peer's registration."""

    # Sprint 1088 — the peer's portable, self-verifying announce credential (signed at
    # registration). The server caches + re-broadcasts it verbatim so a receiver can
    # cryptographically authenticate the relayed (node_id, hardware_profile) instead of
    # trusting the server. The server never needs to verify it (end-to-end at the
    # consumer), but stores it as opaque relayable data.
    announce_credential: Optional[Dict[str, Any]] = None
    """Portable signed credential relayed from the peer's registration."""

    connection_count: int = 0
    """Number of times this peer has connected."""
    
    bytes_sent: int = 0
    """Total bytes sent to this peer."""
    
    bytes_received: int = 0
    """Total bytes received from this peer."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert peer info to dictionary for serialization."""
        out: Dict[str, Any] = {
            "peer_id": self.peer_id,
            "address": self.address,
            "port": self.port,
            "public_key": self.public_key,
            "status": self.status.value,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "capabilities": self.capabilities,
            "region": self.region,
            "version": self.version,
            "metadata": self.metadata,
            "connection_count": self.connection_count,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
        }
        # Sprint 838 — only emit hardware_profile when present;
        # keeps the wire format byte-identical for pre-838 peers
        # that don't advertise hw at all (legacy compatibility).
        if self.hardware_profile is not None:
            out["hardware_profile"] = self.hardware_profile
        # Sprint 1088 — only emit the credential when present (legacy byte-compat).
        if self.announce_credential is not None:
            out["announce_credential"] = self.announce_credential
        return out
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PeerInfo":
        """Create PeerInfo from dictionary."""
        return cls(
            peer_id=data["peer_id"],
            address=data["address"],
            port=data["port"],
            public_key=data.get("public_key"),
            status=PeerStatus(data.get("status", "active")),
            first_seen=datetime.fromisoformat(data["first_seen"]),
            last_seen=datetime.fromisoformat(data["last_seen"]),
            capabilities=data.get("capabilities", []),
            region=data.get("region"),
            version=data.get("version"),
            metadata=data.get("metadata", {}),
            connection_count=data.get("connection_count", 0),
            bytes_sent=data.get("bytes_sent", 0),
            bytes_received=data.get("bytes_received", 0),
            # Sprint 838 — defensive: accept hw_profile if present,
            # default to None for legacy serialized peers.
            hardware_profile=data.get("hardware_profile"),
            # Sprint 1088 — accept the relayed credential if present.
            announce_credential=data.get("announce_credential"),
        )
    
    @property
    def endpoint(self) -> str:
        """Get the full endpoint address for this peer."""
        return f"{self.address}:{self.port}"
    
    @property
    def websocket_url(self) -> str:
        """Get WebSocket URL for connecting to this peer."""
        return f"ws://{self.address}:{self.port}"
    
    def update_activity(self) -> None:
        """Update last_seen timestamp to now."""
        self.last_seen = datetime.now(timezone.utc)
    
    def is_stale(self, timeout_seconds: int = 300) -> bool:
        """Check if peer hasn't been seen recently."""
        elapsed = (datetime.now(timezone.utc) - self.last_seen).total_seconds()
        return elapsed > timeout_seconds


@dataclass
class BootstrapMetrics:
    """
    Metrics for the bootstrap server.
    
    Tracks performance and operational metrics for monitoring
    and health checks.
    """
    
    # Connection metrics
    total_connections: int = 0
    """Total number of connections since server start."""
    
    active_connections: int = 0
    """Current number of active peer connections."""
    
    failed_connections: int = 0
    """Number of failed connection attempts."""
    
    rejected_connections: int = 0
    """Number of rejected connections (rate limit, banned, etc.)."""
    
    # Peer metrics
    total_peers_served: int = 0
    """Total unique peers served since server start."""
    
    peers_by_region: Dict[str, int] = field(default_factory=dict)
    """Count of peers by geographic region."""
    
    peers_by_capability: Dict[str, int] = field(default_factory=dict)
    """Count of peers by capability type."""
    
    # Network metrics
    bytes_sent: int = 0
    """Total bytes sent to all peers."""
    
    bytes_received: int = 0
    """Total bytes received from all peers."""
    
    messages_processed: int = 0
    """Total messages processed."""
    
    # Performance metrics
    avg_response_time_ms: float = 0.0
    """Average response time in milliseconds."""
    
    uptime_seconds: float = 0.0
    """Server uptime in seconds."""
    
    # Health metrics
    last_health_check: Optional[datetime] = None
    """Timestamp of last health check."""
    
    health_check_failures: int = 0
    """Number of consecutive health check failures."""
    
    errors_count: int = 0
    """Total number of errors encountered."""
    
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """Server start time."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "total_connections": self.total_connections,
            "active_connections": self.active_connections,
            "failed_connections": self.failed_connections,
            "rejected_connections": self.rejected_connections,
            "total_peers_served": self.total_peers_served,
            "peers_by_region": self.peers_by_region,
            "peers_by_capability": self.peers_by_capability,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "messages_processed": self.messages_processed,
            "avg_response_time_ms": self.avg_response_time_ms,
            "uptime_seconds": self.uptime_seconds,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "health_check_failures": self.health_check_failures,
            "errors_count": self.errors_count,
            "start_time": self.start_time.isoformat(),
        }
    
    def update_uptime(self) -> None:
        """Update uptime calculation."""
        self.uptime_seconds = (datetime.now(timezone.utc) - self.start_time).total_seconds()
    
    def record_connection(self, success: bool = True, rejected: bool = False) -> None:
        """Record a connection attempt."""
        if rejected:
            self.rejected_connections += 1
        elif success:
            self.total_connections += 1
        else:
            self.failed_connections += 1
    
    def record_message(self, bytes_sent: int = 0, bytes_received: int = 0) -> None:
        """Record message processing."""
        self.messages_processed += 1
        self.bytes_sent += bytes_sent
        self.bytes_received += bytes_received


@dataclass
class BootstrapAnnouncement:
    """
    Announcement message sent to peers.
    
    Used to broadcast new peers joining the network or
    important network updates.
    """
    
    announcement_id: str = field(default_factory=lambda: str(uuid4()))
    """Unique identifier for this announcement."""
    
    announcement_type: str = "peer_join"
    """Type of announcement (peer_join, peer_leave, network_update)."""
    
    peer_id: Optional[str] = None
    """ID of the peer this announcement is about."""
    
    peer_endpoint: Optional[str] = None
    """Endpoint of the peer."""
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When this announcement was created."""
    
    ttl: int = 300
    """Time-to-live in seconds for this announcement."""
    
    signature: Optional[str] = None
    """Cryptographic signature of the announcement."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert announcement to dictionary."""
        return {
            "announcement_id": self.announcement_id,
            "announcement_type": self.announcement_type,
            "peer_id": self.peer_id,
            "peer_endpoint": self.peer_endpoint,
            "timestamp": self.timestamp.isoformat(),
            "ttl": self.ttl,
            "signature": self.signature,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BootstrapAnnouncement":
        """Create announcement from dictionary."""
        return cls(
            announcement_id=data.get("announcement_id", str(uuid4())),
            announcement_type=data.get("announcement_type", "peer_join"),
            peer_id=data.get("peer_id"),
            peer_endpoint=data.get("peer_endpoint"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(timezone.utc),
            ttl=data.get("ttl", 300),
            signature=data.get("signature"),
        )
    
    def is_expired(self) -> bool:
        """Check if announcement has expired."""
        elapsed = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return elapsed > self.ttl
