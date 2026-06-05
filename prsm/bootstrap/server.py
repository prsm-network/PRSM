"""
PRSM Bootstrap Server

Production-ready bootstrap server for the PRSM P2P network.
Handles peer discovery, connection management, and network bootstrapping.
"""

import asyncio
import json
import logging
import signal
import ssl
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import aiohttp

try:
    import websockets
    from websockets.server import serve, WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    websockets = None
    serve = None
    WebSocketServerProtocol = None
    WEBSOCKETS_AVAILABLE = False

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    uvicorn = None
    FastAPI = None
    HTTPException = None
    Request = None
    JSONResponse = None
    BaseModel = None
    FASTAPI_AVAILABLE = False

from .config import BootstrapConfig, get_bootstrap_config
from .models import (
    PeerInfo, PeerStatus, BootstrapMetrics, BootstrapAnnouncement
)

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter for peer connections."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if client is allowed to make a request."""
        now = time.time()
        # Clean old requests
        self.requests[client_id] = [
            t for t in self.requests[client_id]
            if now - t < self.window_seconds
        ]
        
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        self.requests[client_id].append(now)
        return True
    
    def reset(self, client_id: str) -> None:
        """Reset rate limit for a client."""
        self.requests.pop(client_id, None)


class PeerDatabase:
    """Persistent storage for peer information."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.peers: Dict[str, PeerInfo] = {}
        self._load()
    
    def _load(self) -> None:
        """Load peers from disk."""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    self.peers = {
                        peer_id: PeerInfo.from_dict(peer_data)
                        for peer_id, peer_data in data.get("peers", {}).items()
                    }
                logger.info(f"Loaded {len(self.peers)} peers from database")
            except Exception as e:
                logger.error(f"Failed to load peer database: {e}")
                self.peers = {}
    
    def save(self) -> None:
        """Save peers to disk."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "peers": {
                    peer_id: peer.to_dict()
                    for peer_id, peer in self.peers.items()
                },
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self.peers)} peers to database")
        except Exception as e:
            logger.error(f"Failed to save peer database: {e}")
    
    def add_peer(self, peer: PeerInfo) -> None:
        """Add or update a peer."""
        self.peers[peer.peer_id] = peer
    
    def remove_peer(self, peer_id: str) -> None:
        """Remove a peer."""
        self.peers.pop(peer_id, None)
    
    def get_peer(self, peer_id: str) -> Optional[PeerInfo]:
        """Get a peer by ID."""
        return self.peers.get(peer_id)
    
    def get_all_peers(self) -> List[PeerInfo]:
        """Get all peers."""
        return list(self.peers.values())
    
    def get_active_peers(self, timeout_seconds: int = 300) -> List[PeerInfo]:
        """Get peers that have been active recently."""
        now = datetime.now(timezone.utc)
        active = []
        for peer in self.peers.values():
            elapsed = (now - peer.last_seen).total_seconds()
            if elapsed < timeout_seconds and peer.status == PeerStatus.ACTIVE:
                active.append(peer)
        return active


class BootstrapServer:
    """
    Bootstrap server for P2P network.
    
    Provides peer discovery and connection services for the PRSM network.
    New peers connect to the bootstrap server to discover other peers
    and announce their presence to the network.
    """
    
    def __init__(self, config: Optional[BootstrapConfig] = None):
        """
        Initialize the bootstrap server.
        
        Args:
            config: Server configuration. Uses defaults if not provided.
        """
        self.config = config or get_bootstrap_config()
        
        # Peer management
        self.peers: Dict[str, PeerInfo] = {}
        self.connections: Dict[str, "WebSocketServerProtocol"] = {}
        self.ip_connections: Dict[str, int] = defaultdict(int)
        
        # Database for persistence
        if self.config.persist_peers:
            self.db = PeerDatabase(self.config.peer_db_path)
            self.peers = {p.peer_id: p for p in self.db.get_active_peers(self.config.peer_timeout * 2)}
        else:
            self.db = None
        
        # Rate limiting
        self.rate_limiter = RateLimiter(
            max_requests=self.config.rate_limit_requests,
            window_seconds=self.config.rate_limit_window
        )
        
        # Metrics
        self.metrics = BootstrapMetrics()
        
        # Server state
        self.server = None
        self.api_server = None
        self.running = False
        self.start_time: Optional[datetime] = None
        
        # Background tasks
        self._tasks: Set[asyncio.Task] = set()

        # Sprint 392: per-loop heartbeat registry. Each
        # background loop bumps its key on every successful
        # iteration; /health/detailed reads these to detect
        # silent loop failures (a loop that crashed will
        # have a stale or missing entry).
        self._loop_heartbeats: Dict[str, datetime] = {}

        # SSL context
        self._ssl_context: Optional[ssl.SSLContext] = None
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Bootstrap server initialized with config: {self.config.websocket_url}")
    
    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper(), logging.INFO),
            format=self.config.log_format
        )
    
    def _create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context for secure connections."""
        if not self.config.ssl_enabled:
            return None
        
        try:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(
                certfile=self.config.ssl_cert_path,
                keyfile=self.config.ssl_key_path
            )
            
            if self.config.require_client_cert and self.config.ssl_ca_path:
                context.load_verify_locations(cafile=self.config.ssl_ca_path)
                context.verify_mode = ssl.CERT_REQUIRED
            
            # Security settings
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.options |= ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3 | ssl.OP_NO_TLSv1
            
            logger.info("SSL context created successfully")
            return context
            
        except Exception as e:
            logger.error(f"Failed to create SSL context: {e}")
            if self.config.ssl_enabled:
                raise
            return None
    
    async def start(self) -> None:
        """Start the bootstrap server."""
        if not WEBSOCKETS_AVAILABLE:
            raise RuntimeError("websockets library not installed. Install with: pip install websockets")
        
        if self.running:
            logger.warning("Server is already running")
            return
        
        logger.info(f"Starting bootstrap server on {self.config.host}:{self.config.port}")
        
        self.running = True
        self.start_time = datetime.now(timezone.utc)
        
        # Create SSL context
        self._ssl_context = self._create_ssl_context()
        
        # Start WebSocket server
        try:
            self.server = await serve(
                self.handle_peer_connection,
                self.config.host,
                self.config.port,
                ssl=self._ssl_context,
                ping_interval=self.config.heartbeat_interval,
                ping_timeout=self.config.connection_timeout,
                max_size=self.config.max_message_size,
                close_timeout=self.config.connection_timeout,
            )
            logger.info(f"WebSocket server started on port {self.config.port}")
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            self.running = False
            raise
        
        # Start API server
        if FASTAPI_AVAILABLE:
            self._tasks.add(asyncio.create_task(self._run_api_server()))
        
        # Start background tasks
        self._tasks.add(asyncio.create_task(self._health_check_loop()))
        self._tasks.add(asyncio.create_task(self._peer_cleanup_loop()))
        
        if self.config.persist_peers and self.db:
            self._tasks.add(asyncio.create_task(self._peer_backup_loop()))
        
        if self.config.federation_enabled and self.config.federation_peers:
            self._tasks.add(asyncio.create_task(self._federation_sync_loop()))
        
        logger.info("Bootstrap server started successfully")
    
    async def stop(self) -> None:
        """Stop the bootstrap server gracefully."""
        if not self.running:
            return
        
        logger.info("Stopping bootstrap server...")
        self.running = False
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Close all connections
        for peer_id, ws in list(self.connections.items()):
            try:
                await ws.close(code=1001, reason="Server shutting down")
            except Exception:
                pass
        
        # Close WebSocket server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        # Save peer database
        if self.db:
            self.db.save()
        
        logger.info("Bootstrap server stopped")
    
    async def handle_peer_connection(
        self,
        websocket: "WebSocketServerProtocol",
        path: str = "/"
    ) -> None:
        """
        Handle a new peer connection.
        
        This is the main WebSocket handler that processes all peer messages.
        
        Args:
            websocket: WebSocket connection
            path: Request path (unused)
        """
        client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
        peer_id: Optional[str] = None
        
        # Check IP connection limit
        if self.ip_connections[client_ip] >= self.config.max_connections_per_ip:
            logger.warning(f"Connection limit reached for IP: {client_ip}")
            self.metrics.record_connection(rejected=True)
            await websocket.close(code=1008, reason="Connection limit reached")
            return
        
        # Check if IP is banned
        if client_ip in self.config.banned_ips:
            logger.warning(f"Rejected connection from banned IP: {client_ip}")
            self.metrics.record_connection(rejected=True)
            await websocket.close(code=1008, reason="IP banned")
            return
        
        self.ip_connections[client_ip] += 1
        
        try:
            async for message in websocket:
                try:
                    # Rate limiting
                    if not self.rate_limiter.is_allowed(client_ip):
                        await websocket.close(code=1008, reason="Rate limit exceeded")
                        return
                    
                    # Parse message
                    data = json.loads(message)
                    msg_type = data.get("type")
                    
                    if msg_type == "register":
                        peer_id = await self._handle_register(websocket, data, client_ip)
                    elif msg_type == "get_peers":
                        await self._handle_get_peers(websocket, data)
                    elif msg_type == "heartbeat":
                        await self._handle_heartbeat(websocket, data, peer_id)
                    elif msg_type == "announce":
                        await self._handle_announce(websocket, data, peer_id)
                    elif msg_type == "disconnect":
                        await self._handle_disconnect(websocket, data, peer_id)
                        break
                    else:
                        await self._send_error(websocket, f"Unknown message type: {msg_type}")
                    
                    self.metrics.record_message(
                        bytes_sent=len(message) if message else 0,
                        bytes_received=len(message) if message else 0
                    )
                    
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON message")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    await self._send_error(websocket, str(e))
        
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.ip_connections[client_ip] -= 1
            if peer_id:
                await self._handle_peer_disconnect(peer_id)
    
    async def _handle_register(
        self,
        websocket: "WebSocketServerProtocol",
        data: Dict[str, Any],
        client_ip: str
    ) -> str:
        """Handle peer registration."""
        # Validate required fields
        peer_id = data.get("peer_id")
        if not peer_id:
            await self._send_error(websocket, "peer_id is required")
            return ""
        
        port = data.get("port", 8000)
        capabilities = data.get("capabilities", [])
        version = data.get("version")
        region = data.get("region")
        public_key = data.get("public_key")
        # Sprint 838 — hardware_profile relay. Operators with a
        # locally-detected hw profile (sprint 681) opt to advertise
        # it via the registration payload; the bootstrap caches
        # the latest value + re-broadcasts in peer-list responses
        # so cold-start joiners can populate their DHT-backed pool
        # without waiting on direct DISCOVERY_ANNOUNCE that NAT
        # may block. Only accept dict shapes; reject other types
        # silently (defensive — don't break the handshake on a
        # buggy/malicious client).
        supplied_hw = data.get("hardware_profile")
        hardware_profile = (
            supplied_hw if isinstance(supplied_hw, dict) else None
        )

        # Sprint 566: honor a client-supplied `address` field when
        # present + non-empty string. Operators co-located with a
        # bootstrap-server bootstrap via loopback (sprint-460
        # invariant) but need to advertise their EXTERNAL IP to
        # remote peers. Without this, the server records `127.0.0.1`
        # as the address, which is loopback to every other peer.
        # Falls back to client_ip for legacy clients that don't
        # send the field. Non-string / empty values rejected
        # defensively to avoid storing garbage on a misbehaving
        # client.
        supplied_address = data.get("address")
        if (
            isinstance(supplied_address, str)
            and supplied_address.strip()
        ):
            effective_address = supplied_address.strip()
        else:
            effective_address = client_ip

        # Check if peer is banned
        if peer_id in self.config.banned_peers:
            await websocket.close(code=1008, reason="Peer banned")
            return ""

        # Create peer info. Sprint 838 — preserve the existing
        # cached hw_profile if the new registration didn't supply
        # one (legacy clients pre-838) so a re-registration after
        # a brief disconnect doesn't drop hw advertisement we
        # already had. New value overwrites old (operator may
        # have upgraded hardware).
        existing_hw = None
        existing = self.peers.get(peer_id)
        if existing is not None:
            existing_hw = existing.hardware_profile
        effective_hw = (
            hardware_profile if hardware_profile is not None
            else existing_hw
        )
        peer = PeerInfo(
            peer_id=peer_id,
            address=effective_address,
            port=port,
            public_key=public_key,
            status=PeerStatus.ACTIVE,
            capabilities=capabilities,
            region=region,
            version=version,
            connection_count=self.peers.get(peer_id, PeerInfo(peer_id, client_ip, port)).connection_count + 1,
            hardware_profile=effective_hw,
        )
        
        # Check max peers limit
        if len(self.peers) >= self.config.max_peers:
            # Remove oldest inactive peer
            oldest = min(
                (p for p in self.peers.values() if p.peer_id not in self.connections),
                key=lambda p: p.last_seen,
                default=None
            )
            if oldest:
                del self.peers[oldest.peer_id]
        
        # Store peer
        self.peers[peer_id] = peer
        self.connections[peer_id] = websocket
        
        if self.db:
            self.db.add_peer(peer)
        
        # Update metrics
        self.metrics.total_connections += 1
        self.metrics.active_connections = len(self.connections)
        self.metrics.total_peers_served = len(self.peers)
        
        # Update region/capability counts
        if region:
            self.metrics.peers_by_region[region] = self.metrics.peers_by_region.get(region, 0) + 1
        for cap in capabilities:
            self.metrics.peers_by_capability[cap] = self.metrics.peers_by_capability.get(cap, 0) + 1
        
        logger.info(f"Peer registered: {peer_id} from {client_ip}:{port}")
        
        # Send acknowledgment with peer list
        peer_list = await self.get_peer_list(exclude_peer=peer_id)
        response = {
            "type": "register_ack",
            "peer_id": peer_id,
            "peers": peer_list[:self.config.peer_list_size],
            "heartbeat_interval": self.config.heartbeat_interval,
            "server_time": datetime.now(timezone.utc).isoformat(),
            # sp1023 — tell the node how the world sees it: the server-observed
            # source IP joined to its DECLARED listen port (a routable dial target,
            # not the ephemeral WS source port). Lets a co-located / NAT'd node
            # learn its advertise address without a manual PRSM_ADVERTISE_ADDRESS.
            "observed_address": f"{client_ip}:{port}",
        }
        await websocket.send(json.dumps(response))
        
        # Broadcast new peer to existing peers
        await self._broadcast_peer_join(peer)
        
        return peer_id
    
    async def _handle_get_peers(
        self,
        websocket: "WebSocketServerProtocol",
        data: Dict[str, Any]
    ) -> None:
        """Handle request for peer list."""
        exclude_peer = data.get("exclude_peer")
        capabilities = data.get("capabilities", [])
        region = data.get("region")
        limit = min(data.get("limit", self.config.peer_list_size), self.config.peer_list_size)
        
        peer_list = await self.get_peer_list(
            exclude_peer=exclude_peer,
            capabilities=capabilities if capabilities else None,
            region=region
        )
        
        response = {
            "type": "peer_list",
            "peers": peer_list[:limit],
            "total": len(peer_list),
            "server_time": datetime.now(timezone.utc).isoformat(),
        }
        await websocket.send(json.dumps(response))
    
    async def _handle_heartbeat(
        self,
        websocket: "WebSocketServerProtocol",
        data: Dict[str, Any],
        peer_id: Optional[str]
    ) -> None:
        """Handle heartbeat message."""
        if not peer_id:
            await self._send_error(websocket, "Not registered")
            return
        
        if peer_id in self.peers:
            self.peers[peer_id].update_activity()
            
            # Update metrics if provided
            if "metrics" in data:
                peer_metrics = data["metrics"]
                self.peers[peer_id].bytes_sent = peer_metrics.get("bytes_sent", 0)
                self.peers[peer_id].bytes_received = peer_metrics.get("bytes_received", 0)
        
        response = {
            "type": "heartbeat_ack",
            "server_time": datetime.now(timezone.utc).isoformat(),
        }
        await websocket.send(json.dumps(response))
    
    async def _handle_announce(
        self,
        websocket: "WebSocketServerProtocol",
        data: Dict[str, Any],
        peer_id: Optional[str]
    ) -> None:
        """Handle peer announcement."""
        if not peer_id:
            await self._send_error(websocket, "Not registered")
            return
        
        announcement = BootstrapAnnouncement.from_dict(data)
        
        # Broadcast to other peers
        await self._broadcast_announcement(announcement, exclude_peer=peer_id)
        
        response = {
            "type": "announce_ack",
            "announcement_id": announcement.announcement_id,
        }
        await websocket.send(json.dumps(response))
    
    async def _handle_disconnect(
        self,
        websocket: "WebSocketServerProtocol",
        data: Dict[str, Any],
        peer_id: Optional[str]
    ) -> None:
        """Handle graceful disconnect."""
        if peer_id:
            await self._handle_peer_disconnect(peer_id)
        
        response = {"type": "disconnect_ack"}
        await websocket.send(json.dumps(response))
    
    async def _handle_peer_disconnect(self, peer_id: str) -> None:
        """Handle peer disconnection."""
        if peer_id in self.peers:
            self.peers[peer_id].status = PeerStatus.DISCONNECTED
            self.peers[peer_id].update_activity()
        
        if peer_id in self.connections:
            del self.connections[peer_id]
        
        self.metrics.active_connections = len(self.connections)
        
        logger.info(f"Peer disconnected: {peer_id}")
        
        # Broadcast peer leave
        announcement = BootstrapAnnouncement(
            announcement_type="peer_leave",
            peer_id=peer_id,
        )
        await self._broadcast_announcement(announcement)
    
    async def _send_error(
        self,
        websocket: "WebSocketServerProtocol",
        message: str
    ) -> None:
        """Send error message to peer."""
        response = {
            "type": "error",
            "message": message,
            "server_time": datetime.now(timezone.utc).isoformat(),
        }
        await websocket.send(json.dumps(response))
    
    async def _broadcast_peer_join(self, peer: PeerInfo) -> None:
        """Broadcast new peer join to all connected peers."""
        announcement = BootstrapAnnouncement(
            announcement_type="peer_join",
            peer_id=peer.peer_id,
            peer_endpoint=peer.endpoint,
        )
        await self._broadcast_announcement(announcement, exclude_peer=peer.peer_id)
    
    async def _broadcast_announcement(
        self,
        announcement: BootstrapAnnouncement,
        exclude_peer: Optional[str] = None
    ) -> None:
        """Broadcast announcement to all connected peers."""
        if not self.connections:
            return
        
        message = json.dumps(announcement.to_dict())
        tasks = []
        
        for peer_id, ws in self.connections.items():
            if peer_id == exclude_peer:
                continue
            tasks.append(ws.send(message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def get_peer_list(
        self,
        exclude_peer: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        region: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get list of known peers.
        
        Args:
            exclude_peer: Peer ID to exclude from list
            capabilities: Filter by capabilities
            region: Filter by region
            limit: Maximum number of peers to return
        
        Returns:
            List of peer information dictionaries
        """
        peers = []
        
        for peer in self.peers.values():
            # Skip excluded peer
            if peer.peer_id == exclude_peer:
                continue
            
            # Skip inactive peers
            if peer.status != PeerStatus.ACTIVE:
                continue
            
            # Skip stale peers
            if peer.is_stale(self.config.peer_timeout):
                continue
            
            # Filter by capabilities
            if capabilities:
                if not any(cap in peer.capabilities for cap in capabilities):
                    continue
            
            # Filter by region
            if region and peer.region != region:
                continue
            
            entry: Dict[str, Any] = {
                "peer_id": peer.peer_id,
                "address": peer.address,
                "port": peer.port,
                "capabilities": peer.capabilities,
                "region": peer.region,
                "version": peer.version,
            }
            # Sprint 838 — relay hardware_profile if cached.
            # Omit the key entirely for peers that haven't
            # advertised hw so pre-838 wire format stays byte-
            # identical for legacy fleets.
            if peer.hardware_profile is not None:
                entry["hardware_profile"] = peer.hardware_profile
            peers.append(entry)
        
        # Sort by last seen (most recent first)
        peers.sort(key=lambda p: self.peers[p["peer_id"]].last_seen, reverse=True)
        
        return peers[:limit]
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Return server health status.
        
        Returns:
            Health status dictionary
        """
        self.metrics.update_uptime()
        self.metrics.last_health_check = datetime.now(timezone.utc)
        
        return {
            "status": "healthy" if self.running else "stopped",
            "uptime_seconds": self.metrics.uptime_seconds,
            "active_connections": self.metrics.active_connections,
            "total_peers": len(self.peers),
            "total_connections": self.metrics.total_connections,
            "failed_connections": self.metrics.failed_connections,
            "messages_processed": self.metrics.messages_processed,
            "region": self.config.region,
            "version": "1.0.0",
            "server_time": datetime.now(timezone.utc).isoformat(),
        }
    
    async def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        while self.running:
            try:
                health = await self.health_check()
                logger.debug(f"Health check: {health['status']}, peers: {health['total_peers']}")
                self._record_loop_heartbeat("health_check_loop")
            except Exception as e:
                logger.error(f"Health check error: {e}")

            await asyncio.sleep(self.config.health_check_interval)

    async def _peer_cleanup_loop(self) -> None:
        """Periodic peer cleanup loop."""
        while self.running:
            try:
                now = datetime.now(timezone.utc)
                stale_peers = []

                for peer_id, peer in list(self.peers.items()):
                    elapsed = (now - peer.last_seen).total_seconds()

                    if elapsed > self.config.peer_timeout:
                        peer.status = PeerStatus.IDLE

                    if elapsed > self.config.peer_timeout * 3:
                        stale_peers.append(peer_id)

                # Remove stale peers
                for peer_id in stale_peers:
                    del self.peers[peer_id]
                    logger.debug(f"Removed stale peer: {peer_id}")

                if stale_peers:
                    logger.info(f"Cleaned up {len(stale_peers)} stale peers")
                self._record_loop_heartbeat("peer_cleanup")
            except Exception as e:
                logger.error(f"Peer cleanup error: {e}")

            await asyncio.sleep(self.config.peer_timeout)

    async def _peer_backup_loop(self) -> None:
        """Periodic peer database backup loop."""
        while self.running:
            try:
                if self.db:
                    self.db.save()
                self._record_loop_heartbeat("peer_backup")
            except Exception as e:
                logger.error(f"Peer backup error: {e}")

            await asyncio.sleep(self.config.peer_db_backup_interval)
    
    async def _federation_sync_loop(self) -> None:
        """Periodic federation sync loop."""
        while self.running:
            try:
                # Use aiohttp to fetch peers from federation peers
                async with aiohttp.ClientSession() as session:
                    for fed_peer in self.config.federation_peers:
                        try:
                            # Construct API URL
                            peer_url = f"{fed_peer.rstrip('/')}/peers"
                            async with session.get(peer_url, timeout=10) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    peers_data = data.get("peers", [])
                                    new_peers_count = 0
                                    
                                    for p_data in peers_data:
                                        peer_id = p_data.get("peer_id")
                                        if not peer_id:
                                            continue
                                            
                                        if peer_id not in self.peers:
                                            # Add newly discovered peer as idle
                                            peer = PeerInfo(
                                                peer_id=peer_id,
                                                address=p_data.get("address", ""),
                                                port=p_data.get("port", 8000),
                                                status=PeerStatus.IDLE,
                                                capabilities=p_data.get("capabilities", []),
                                                region=p_data.get("region"),
                                                version=p_data.get("version"),
                                            )
                                            self.peers[peer_id] = peer
                                            if self.db:
                                                self.db.add_peer(peer)
                                            new_peers_count += 1
                                            
                                    if new_peers_count > 0:
                                        logger.info(f"Federation sync: Added {new_peers_count} new peers from {fed_peer}")
                                    else:
                                        logger.debug(f"Federation sync: No new peers from {fed_peer}")
                                else:
                                    logger.warning(f"Federation sync failed with {fed_peer}: HTTP {response.status}")
                        except Exception as peer_err:
                            logger.error(f"Failed to sync with federation peer {fed_peer}: {peer_err}")
                self._record_loop_heartbeat("federation_sync")
            except Exception as e:
                logger.error(f"Federation sync error: {e}")

            await asyncio.sleep(self.config.federation_sync_interval)

    def _record_loop_heartbeat(self, name: str) -> None:
        """Sprint 392 — bump a loop's last-iteration timestamp
        in the heartbeat registry. /health/detailed reads
        this to detect silently-dead loops.
        """
        self._loop_heartbeats[name] = datetime.now(timezone.utc)

    def health_check_detailed(self) -> Dict[str, Any]:
        """Sprint 392 — per-subsystem readiness probe.

        Each background loop is classified relative to its
        expected iteration interval:
          age < 2 × interval → healthy
          2 ≤ age < 5 × interval → degraded
          age ≥ 5 × interval OR missing → stale

        Aggregate `status` mirrors the worst subsystem:
          all healthy → healthy
          any degraded but none stale → degraded
          any stale → unhealthy
        """
        now = datetime.now(timezone.utc)
        # Subsystem name → expected interval seconds. Pulled
        # from the same config knobs the loops await on.
        subsystem_intervals = {
            "peer_cleanup": float(self.config.peer_timeout),
            "peer_backup": float(self.config.peer_db_backup_interval),
            "federation_sync": float(self.config.federation_sync_interval),
            "health_check_loop": float(self.config.health_check_interval),
        }
        # Sprint 397 — loops whose enabling-config isn't met
        # (e.g., federation_sync when federation_peers is
        # empty) are NEVER instantiated at start() (see
        # server.py:292 for federation_sync). Surface those
        # as "disabled" instead of "stale" so a default
        # standalone bootstrap doesn't read as "unhealthy".
        def _loop_is_configured(name: str) -> bool:
            if name == "federation_sync":
                return bool(
                    self.config.federation_enabled
                    and self.config.federation_peers
                )
            if name == "peer_backup":
                return bool(
                    self.config.persist_peers and self.db
                )
            # peer_cleanup + health_check_loop always start
            return True

        subsystems: Dict[str, Dict[str, Any]] = {}
        worst = "healthy"
        for name, interval in subsystem_intervals.items():
            last = self._loop_heartbeats.get(name)
            if last is None:
                if not _loop_is_configured(name):
                    # Operator-disabled, not silently dead
                    subsystems[name] = {
                        "alive": False,
                        "status": "disabled",
                        "last_heartbeat_age_seconds": None,
                        "expected_interval_seconds": interval,
                    }
                    continue
                subsystems[name] = {
                    "alive": False,
                    "status": "stale",
                    "last_heartbeat_age_seconds": None,
                    "expected_interval_seconds": interval,
                }
                worst = "unhealthy"
                continue
            age = (now - last).total_seconds()
            if age < 2 * interval:
                status = "healthy"
            elif age < 5 * interval:
                status = "degraded"
            else:
                status = "stale"
            subsystems[name] = {
                "alive": status != "stale",
                "status": status,
                "last_heartbeat_age_seconds": age,
                "expected_interval_seconds": interval,
            }
            if status == "stale":
                worst = "unhealthy"
            elif status == "degraded" and worst == "healthy":
                worst = "degraded"

        # api_server: if /health/detailed is returning a
        # response, by definition the API server is alive.
        subsystems["api_server"] = {
            "alive": True,
            "status": "healthy",
            "last_heartbeat_age_seconds": 0.0,
            "expected_interval_seconds": None,
        }

        return {
            "status": worst,
            "subsystems": subsystems,
            "server_time": now.isoformat(),
        }

    def _render_prometheus_exposition(self) -> str:
        """Render BootstrapMetrics as Prometheus exposition text.

        Sprint 389 — extracted from inline `/prometheus`
        handler so both the explicit `/prometheus` path and
        the content-negotiated `/metrics` path share one
        renderer. Covers all flat counters/gauges plus the
        two dict fields (peers_by_region, peers_by_capability)
        as labeled gauges.
        """
        m = self.metrics.to_dict()
        lines: list[str] = []
        flat_metric_map = {
            "active_connections": ("gauge", "Number of active peer connections"),
            "total_connections": ("counter", "Total peer connections since start"),
            "failed_connections": ("counter", "Total failed connection attempts"),
            "rejected_connections": ("counter", "Total rejected connections (rate limit, banned, etc.)"),
            "messages_processed": ("counter", "Total messages processed"),
            "total_peers_served": ("counter", "Total unique peers served"),
            "bytes_sent": ("counter", "Total bytes sent to peers"),
            "bytes_received": ("counter", "Total bytes received from peers"),
            "avg_response_time_ms": ("gauge", "Rolling average response time in ms"),
            "uptime_seconds": ("gauge", "Server uptime in seconds"),
            "health_check_failures": ("counter", "Total health check failures"),
            "errors_count": ("counter", "Total errors"),
        }
        for key, (mtype, help_text) in flat_metric_map.items():
            if key in m:
                lines.append(f"# HELP prsm_bootstrap_{key} {help_text}")
                lines.append(f"# TYPE prsm_bootstrap_{key} {mtype}")
                lines.append(f"prsm_bootstrap_{key} {m[key]}")

        labeled_gauge_map = {
            "peers_by_region": ("region", "Count of peers by geographic region"),
            "peers_by_capability": ("capability", "Count of peers by advertised capability"),
        }

        def _escape_label_value(s: str) -> str:
            return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

        for key, (label_name, help_text) in labeled_gauge_map.items():
            label_dict = m.get(key) or {}
            if not label_dict:
                continue
            metric_name = f"prsm_bootstrap_{key}"
            lines.append(f"# HELP {metric_name} {help_text}")
            lines.append(f"# TYPE {metric_name} gauge")
            for raw_label, value in label_dict.items():
                escaped = _escape_label_value(str(raw_label))
                lines.append(
                    f'{metric_name}{{{label_name}="{escaped}"}} {value}'
                )

        # Sprint 394 — per-subsystem status + heartbeat age
        # as labeled gauges so Prometheus alerts can fire on
        # a specific stuck loop. Status encoded numerically
        # (Prometheus convention for enum metrics):
        #   0 = healthy, 1 = degraded, 2 = stale
        detailed = self.health_check_detailed()
        subsystems = detailed.get("subsystems") or {}
        if subsystems:
            status_metric = "prsm_bootstrap_subsystem_status"
            age_metric = (
                "prsm_bootstrap_subsystem_heartbeat_age_seconds"
            )
            lines.append(
                f"# HELP {status_metric} "
                f"Per-subsystem readiness "
                f"(0=healthy, 1=degraded, 2=stale)"
            )
            lines.append(f"# TYPE {status_metric} gauge")
            lines.append(
                f"# HELP {age_metric} "
                f"Seconds since last successful heartbeat "
                f"per subsystem"
            )
            lines.append(f"# TYPE {age_metric} gauge")
            status_encoding = {
                "healthy": 0,
                # Sprint 397 — "disabled" (operator opt-out)
                # encodes as 1 alongside "degraded" so
                # Prometheus alerts on `status >= 2` fire
                # only for silent-death/stale subsystems,
                # not for never-instantiated configured-off
                # loops. Mirrors sprint-395 operator-node
                # opt-out semantics.
                "degraded": 1,
                "disabled": 1,
                "stale": 2,
            }
            for sub_name, sub_data in subsystems.items():
                escaped = _escape_label_value(sub_name)
                sub_status_val = status_encoding.get(
                    sub_data.get("status"), 2,
                )
                lines.append(
                    f'{status_metric}{{subsystem="{escaped}"}}'
                    f' {sub_status_val}'
                )
                age = sub_data.get("last_heartbeat_age_seconds")
                if isinstance(age, (int, float)):
                    lines.append(
                        f'{age_metric}{{subsystem="{escaped}"}}'
                        f' {age}'
                    )

        nl = chr(10)
        return nl.join(lines) + nl

    def _build_api_app(self):
        """Construct + return the FastAPI app for the
        bootstrap server's HTTP control surface.

        Sprint 389 extracted this from `_run_api_server` so
        tests can wrap it in a TestClient without binding a
        port + running uvicorn.
        """
        if not FASTAPI_AVAILABLE:
            raise RuntimeError(
                "FastAPI not available; cannot build API app"
            )

        try:
            from importlib.metadata import version as _pkg_version
            _v = _pkg_version("prsm-network")
        except Exception:  # noqa: BLE001
            _v = "unknown"
        app = FastAPI(title="PRSM Bootstrap Server", version=_v)
        from fastapi import Request
        from fastapi.responses import PlainTextResponse, JSONResponse

        @app.get("/health")
        async def health():
            return await self.health_check()

        @app.get("/health/detailed")
        async def health_detailed():
            return self.health_check_detailed()

        @app.get("/metrics")
        async def metrics(request: Request):
            # Sprint 389 — content negotiation. Prometheus
            # scrape clients send Accept: text/plain or
            # application/openmetrics-text by default.
            # Anything else (incl. no Accept header) keeps
            # the pre-sprint-389 JSON behavior for
            # backwards-compat with operator polling scripts.
            accept = (request.headers.get("accept") or "").lower()
            prefers_prometheus = (
                "text/plain" in accept
                or "application/openmetrics-text" in accept
            )
            if prefers_prometheus:
                return PlainTextResponse(
                    self._render_prometheus_exposition(),
                    media_type="text/plain; version=0.0.4",
                )
            return JSONResponse(self.metrics.to_dict())

        @app.get("/prometheus")
        async def prometheus_metrics():
            return PlainTextResponse(
                self._render_prometheus_exposition(),
                media_type="text/plain; version=0.0.4",
            )

        @app.get("/peers")
        async def peers():
            return {"peers": await self.get_peer_list()}

        @app.get("/config")
        async def config():
            return self.config.to_dict()

        return app

    async def _run_api_server(self) -> None:
        """Run the HTTP API server for health checks and metrics."""
        if not FASTAPI_AVAILABLE:
            logger.warning("FastAPI not available, skipping API server")
            return

        app = self._build_api_app()
        config = uvicorn.Config(
            app,
            host=self.config.host,
            port=self.config.api_port,
            log_level=self.config.log_level.lower(),
        )
        server = uvicorn.Server(config)
        await server.serve()


async def run_bootstrap_server(config: Optional[BootstrapConfig] = None) -> None:
    """
    Run the bootstrap server.
    
    This is the main entry point for running a PRSM bootstrap server.
    
    Args:
        config: Server configuration. Uses defaults if not provided.
    """
    server = BootstrapServer(config)
    
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(server.stop())
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        await server.start()
        
        # Keep running until stopped
        while server.running:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await server.stop()


def main() -> None:
    """Main entry point for the bootstrap server CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    config = get_bootstrap_config()
    asyncio.run(run_bootstrap_server(config))


if __name__ == "__main__":
    main()
