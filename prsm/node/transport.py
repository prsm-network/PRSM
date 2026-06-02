"""
WebSocket P2P Transport
========================

Real peer-to-peer connectivity over WebSockets.
Handles incoming/outgoing connections, message routing,
handshake protocol, and connection health monitoring.
"""

import asyncio
import base64
import collections
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Dict, List, Optional, Set
from urllib.parse import urlparse

import websockets
import websockets.server
import websockets.client

from prsm.node.identity import NodeIdentity, verify_signature

def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    """Read a float from env, falling back to `default` on unset/invalid, and
    clamping to `minimum` (sp936 rate-limit knobs)."""
    raw = os.getenv(name)
    if raw is None:
        return max(minimum, float(default))
    try:
        return max(minimum, float(raw))
    except (TypeError, ValueError):
        return max(minimum, float(default))


if TYPE_CHECKING:
    from prsm.node.jurisdiction_filter import PeerJurisdictionFilter
    from prsm.node.transport_adapter import TransportAdapter


def _parse_uri_host_port(uri: str) -> tuple[str, int]:
    """Extract ``(host, port)`` from a ws:// or wss:// URI.

    Used by ``WebSocketTransport.connect_to_peer`` when routing through
    a non-direct TransportAdapter — the adapter takes host+port, not
    the full URI, because SOCKS proxies operate on TCP dest host/port
    and don't care about the wrapping websocket scheme.
    """
    parsed = urlparse(uri)
    host = parsed.hostname or ""
    # Fall back to scheme default (443 for wss, 80 for ws) if no explicit port.
    port = parsed.port if parsed.port is not None else (
        443 if parsed.scheme == "wss" else 80
    )
    return host, port

logger = logging.getLogger(__name__)

_HANDSHAKE_REASON_TAXONOMY: Dict[str, str] = {
    "Missing public key": "missing_public_key",
    "Missing signature": "missing_signature",
    "Sender identity mismatch": "sender_identity_mismatch",
    "Invalid signature": "invalid_signature",
    "Missing ack binding": "missing_ack_binding",
    "Ack nonce mismatch": "ack_nonce_mismatch",
    "Missing nonce": "missing_nonce",
    "Replay nonce": "replay_nonce",
    "Expected handshake": "unexpected_message_type",
    "Self-connection rejected": "self_connection_rejected",
    "Already connected": "already_connected",
    "Timeout": "timeout",
    "Invalid handshake ack type": "invalid_ack_type",
}

# Message type constants
MSG_HANDSHAKE = "handshake"
MSG_HANDSHAKE_ACK = "handshake_ack"
MSG_PING = "ping"
MSG_PONG = "pong"
MSG_GOSSIP = "gossip"
MSG_DIRECT = "direct"
MSG_PEER_CONNECTED = "peer_connected"
MSG_PEER_DISCONNECTED = "peer_disconnected"


@dataclass
class P2PMessage:
    """A signed message exchanged between peers."""
    msg_type: str
    sender_id: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    signature: str = ""
    ttl: int = 5
    nonce: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def to_bytes(self) -> bytes:
        """Serialize for signing/sending (excludes signature)."""
        data = {
            "msg_type": self.msg_type,
            "sender_id": self.sender_id,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "nonce": self.nonce,
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()

    def to_json(self) -> str:
        """Full serialization including signature."""
        data = {
            "msg_type": self.msg_type,
            "sender_id": self.sender_id,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "signature": self.signature,
            "ttl": self.ttl,
            "nonce": self.nonce,
        }
        return json.dumps(data, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> "P2PMessage":
        data = json.loads(raw)
        return cls(
            msg_type=data["msg_type"],
            sender_id=data["sender_id"],
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
            signature=data.get("signature", ""),
            ttl=data.get("ttl", 5),
            nonce=data.get("nonce", uuid.uuid4().hex[:16]),
        )

    def sign(self, identity: NodeIdentity) -> None:
        """Sign this message with the node's private key."""
        self.signature = identity.sign(self.to_bytes())


@dataclass
class PeerConnection:
    """Wraps a WebSocket connection with peer metadata."""
    peer_id: str
    address: str
    websocket: Any  # websockets connection object
    public_key_b64: str = ""
    display_name: str = ""
    roles: List[str] = field(default_factory=list)
    connected_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    outbound: bool = False  # True if we initiated the connection


# Type alias for message handler callbacks
MessageHandler = Callable[[P2PMessage, "PeerConnection"], Coroutine[Any, Any, None]]


class WebSocketTransport:
    """WebSocket-based P2P transport layer.

    Manages both a server (for incoming connections) and client connections
    (outgoing to other peers). Provides message routing and health monitoring.
    
    Thread Safety:
        All peer connection operations are protected by asyncio locks to prevent
        race conditions when multiple coroutines access shared state concurrently.
    """

    def __init__(
        self,
        identity: NodeIdentity,
        host: str = "0.0.0.0",
        port: int = 9001,
        nonce_window: float = 300.0,
        ws_ping_interval: float = 20.0,
        ws_ping_timeout: float = 10.0,
        handshake_timeout: float = 10.0,
        nonce_cleanup_interval: float = 60.0,
        transport_adapter: Optional["TransportAdapter"] = None,
        jurisdiction_filter: Optional["PeerJurisdictionFilter"] = None,
        peer_msg_rate: float = 100.0,
        peer_msg_burst: float = 200.0,
    ):
        self.identity = identity
        self.host = host
        self.port = port
        self.ws_ping_interval = ws_ping_interval
        self.ws_ping_timeout = ws_ping_timeout
        self.handshake_timeout = handshake_timeout
        self.nonce_cleanup_interval = nonce_cleanup_interval
        # R9 Phase 6.2: pluggable outbound-transport adapter. Default is
        # DirectAdapter (pre-R9 behavior, unchanged). Operators in censoring
        # jurisdictions inject SocksAdapter (Tor / V2Ray / Trojan / Shadow-
        # socks via local SOCKS5) at node wiring time. See
        # docs/2026-04-23-r9-transport-censorship-resistance-scoping.md §5.
        if transport_adapter is None:
            from prsm.node.transport_adapter import DirectAdapter
            transport_adapter = DirectAdapter()
        self._transport_adapter = transport_adapter
        # R9 Phase 6.3: optional peer-jurisdiction filter. When set,
        # connect_to_peer consults the filter BEFORE adapter.open_connection.
        # Blocked peers never reach the transport. Default None = no
        # filtering (pre-R9 behavior). Per R9 §8 the Foundation does not
        # ship default jurisdiction lists — operators configure this
        # based on their own threat model. See
        # docs/2026-04-23-r9-transport-censorship-resistance-scoping.md §6.
        self._jurisdiction_filter = jurisdiction_filter

        self.peers: Dict[str, PeerConnection] = {}  # peer_id -> connection
        self._handlers: Dict[str, List[MessageHandler]] = {}
        self._seen_nonces: Set[str] = set()
        self._nonce_window = nonce_window
        self._nonce_timestamps: Dict[str, float] = {}
        self._server = None
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # Thread safety locks for concurrent access to shared state
        self._peers_lock = asyncio.Lock()  # Protects self.peers dictionary
        self._nonces_lock = asyncio.Lock()  # Protects _seen_nonces and _nonce_timestamps

        # sp936 — per-peer message rate limit (token bucket). Without it a single
        # connected peer can flood messages: each unique nonce grows _seen_nonces
        # (bounded only by the ~300s window), each is parsed + dispatched, and
        # gossip re-propagates it (amplification) — a single-peer DoS. Defaults
        # are generous (legit gossip is a few msgs/sec/peer); env-tunable.
        self._peer_msg_rate = _env_float("PRSM_PEER_MSG_RATE", peer_msg_rate, minimum=1.0)
        self._peer_msg_burst = _env_float("PRSM_PEER_MSG_BURST", peer_msg_burst, minimum=1.0)
        self._peer_buckets: Dict[str, List[float]] = {}  # peer_id -> [tokens, last_refill]

        # Additive observability counters (must never alter protocol behavior)
        self._telemetry: Dict[str, Any] = {
            "handshake_success_total": 0,
            "handshake_failure_total": 0,
            "handshake_failure_reasons": collections.Counter(),
            "dispatch_success_total": 0,
            "dispatch_failure_total": 0,
            "dispatch_failure_reasons": collections.Counter(),
        }

    @staticmethod
    def _handshake_reason_label(reason: str) -> str:
        """Map free-form internal reason strings to bounded telemetry labels."""
        return _HANDSHAKE_REASON_TAXONOMY.get(reason, "other")

    def _record_handshake_outcome(self, *, success: bool, reason: str = "") -> None:
        """Best-effort telemetry sink for handshake validation outcomes."""
        try:
            if success:
                self._telemetry["handshake_success_total"] += 1
            else:
                self._telemetry["handshake_failure_total"] += 1
                label = self._handshake_reason_label(reason)
                self._telemetry["handshake_failure_reasons"][label] += 1
                logger.debug("telemetry_event=transport_handshake_auth outcome=failure reason=%s", label)
        except Exception:
            # Fail closed for telemetry internals only.
            pass

    def _record_dispatch_outcome(self, *, success: bool, reason: str = "") -> None:
        """Best-effort telemetry sink for internal message dispatch outcomes."""
        try:
            if success:
                self._telemetry["dispatch_success_total"] += 1
            else:
                self._telemetry["dispatch_failure_total"] += 1
                self._telemetry["dispatch_failure_reasons"][reason or "handler_exception"] += 1
                logger.debug(
                    "telemetry_event=transport_dispatch outcome=failure reason=%s",
                    reason or "handler_exception",
                )
        except Exception:
            pass

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Return a stable copy of transport telemetry counters for tests/debugging."""
        return {
            "handshake_success_total": int(self._telemetry["handshake_success_total"]),
            "handshake_failure_total": int(self._telemetry["handshake_failure_total"]),
            "handshake_failure_reasons": dict(self._telemetry["handshake_failure_reasons"]),
            "dispatch_success_total": int(self._telemetry["dispatch_success_total"]),
            "dispatch_failure_total": int(self._telemetry["dispatch_failure_total"]),
            "dispatch_failure_reasons": dict(self._telemetry["dispatch_failure_reasons"]),
        }

    @property
    def peer_count(self) -> int:
        """Return the number of connected peers (thread-safe read)."""
        # Note: This property is not async, so we can't use the lock here.
        # For accurate count in async context, use get_peer_count() instead.
        return len(self.peers)

    @property
    def peer_addresses(self) -> List[str]:
        """Return list of peer addresses (thread-safe snapshot)."""
        # Note: For accurate snapshot in async context, use get_peer_addresses() instead.
        return [p.address for p in self.peers.values()]

    async def get_peer_count(self) -> int:
        """Async method to get peer count with proper locking."""
        async with self._peers_lock:
            return len(self.peers)

    async def get_peer_addresses(self) -> List[str]:
        """Async method to get peer addresses with proper locking."""
        async with self._peers_lock:
            return [p.address for p in self.peers.values()]

    def get_peer(self, node_id: str) -> Optional["PeerConnection"]:
        """Return the PeerConnection for a node_id, or None if not connected.

        Sync snapshot — for Phase 2 RemoteShardDispatcher's peer-resolve
        step. Reads the peers dict without the async lock; callers should
        treat the returned object as a snapshot (may disconnect after).
        """
        return self.peers.get(node_id)

    # ── Message handler registration ─────────────────────────────

    def on_message(self, msg_type: str, handler: MessageHandler) -> None:
        """Register a handler for a specific message type."""
        self._handlers.setdefault(msg_type, []).append(handler)

    async def _dispatch(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Route a message to all registered handlers for its type.

        Sprint 731 F64 — bind `msg.sender_id` to the handshake-
        authenticated `peer.peer_id` for direct point-to-point
        messages (MSG_DIRECT, MSG_PEER_CONNECTED,
        MSG_PEER_DISCONNECTED). Generalizes sprint-730's per-
        chain-executor-handler bind to ALL MSG_DIRECT handlers
        (ledger_sync / compute_provider / storage_provider /
        content_provider / agent_registry — each registered its
        own MSG_DIRECT handler and each read msg.sender_id
        without crypto rebinding).

        MSG_GOSSIP is excluded: gossip relay legitimately carries
        the original sender's id, which differs from the
        relaying peer's id by design. The gossip protocol does
        its own provenance verification (per-message signature
        check against the original sender's known pubkey).
        Internal pseudo-messages (peer_connected /
        peer_disconnected) are constructed with peer.peer_id as
        sender_id already, so the bind is a no-op for them.
        """
        if (
            peer is not None
            and msg.msg_type != MSG_GOSSIP
        ):
            authentic = getattr(peer, "peer_id", None)
            if authentic:
                msg.sender_id = authentic
        handlers = self._handlers.get(msg.msg_type, [])
        for handler in handlers:
            try:
                await handler(msg, peer)
                self._record_dispatch_outcome(success=True)
            except Exception as e:
                logger.error(f"Handler error for {msg.msg_type}: {e}")
                self._record_dispatch_outcome(success=False, reason="handler_exception")

    # ── Server (incoming connections) ────────────────────────────

    async def start(self) -> None:
        """Start the WebSocket server and background tasks."""
        self._running = True
        # Sprint 627 — bump max_size from default 1MB to 16MB so
        # chain-executor logit payloads (vocab × seq_len × float32 +
        # base64 overhead) don't get silently dropped. gpt2 5-token
        # response = ~1.3MB after base64; larger models / longer prompts
        # need more headroom. 16MB ceiling lets us handle full Llama
        # vocab (128k) with reasonable sequence lengths.
        self._server = await websockets.server.serve(
            self._handle_incoming,
            self.host,
            self.port,
            ping_interval=self.ws_ping_interval,
            ping_timeout=self.ws_ping_timeout,
            max_size=16 * 1024 * 1024,
        )
        self._tasks.append(asyncio.create_task(self._nonce_cleanup_loop()))
        logger.info(f"P2P transport listening on ws://{self.host}:{self.port}")

    # sp953 — graceful shutdown must be BOUNDED. A peer's websocket close
    # handshake (and the server's wait_closed) can hang indefinitely when the
    # peer is gone or unresponsive — e.g. a requester that disconnects while the
    # server is mid-stream, so the close-ack never round-trips (it deadlocks:
    # the requester's close awaits the server's ack, but the server only stops
    # sending once its socket errors, which only happens once the requester's
    # socket is actually closed). An unbounded await there hangs node shutdown
    # forever (and hung the F54 disconnect-during-stream test). A legitimate
    # close on localhost/WAN completes in well under a second; only a stalled
    # handshake reaches this bound, at which point we force the socket closed.
    _SHUTDOWN_CLOSE_TIMEOUT_SECONDS = 2.0

    async def stop(self) -> None:
        """Gracefully shut down server and all peer connections.

        Bounded (sp953): no peer/server close can block shutdown beyond
        ``_SHUTDOWN_CLOSE_TIMEOUT_SECONDS``. Peers are snapshotted + cleared
        under the lock, then closed CONCURRENTLY outside it, so one stalled
        close neither holds the peers lock nor serializes behind the others —
        total peer-close latency is bounded by the timeout, not timeout×N."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

        # Snapshot + clear peers under the lock (so no concurrent send finds a
        # closing peer), then close concurrently + bounded OUTSIDE the lock.
        async with self._peers_lock:
            peers = list(self.peers.values())
            self.peers.clear()
        if peers:
            await asyncio.gather(
                *(self._bounded_close(p.websocket) for p in peers),
                return_exceptions=True,
            )

        if self._server:
            self._server.close()
            try:
                await asyncio.wait_for(
                    self._server.wait_closed(),
                    timeout=self._SHUTDOWN_CLOSE_TIMEOUT_SECONDS,
                )
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                # Server had in-flight connection handlers that didn't drain in
                # time (e.g. a handler blocked mid-stream). The listener is
                # already closed by .close(); abandon the stragglers.
                pass
            self._server = None

        logger.info("P2P transport stopped")

    @staticmethod
    def _force_close_transport(websocket: Any) -> None:
        """Force the underlying TCP transport closed (synchronous, instant).
        This makes any in-flight graceful close() resolve immediately instead of
        awaiting the websockets library's internal close_timeout."""
        for _attr in ("transport", "_transport"):
            _t = getattr(websocket, _attr, None)
            if _t is not None:
                try:
                    _t.close()
                except Exception:  # noqa: BLE001
                    pass
                break

    async def _bounded_close(self, websocket: Any) -> None:
        """Close a websocket without letting a stalled handshake hang shutdown
        (sp953). Try a GRACEFUL close first (sends the CLOSE frame so a
        responsive peer deregisters us cleanly), bounded by
        ``_SHUTDOWN_CLOSE_TIMEOUT_SECONDS``. If it stalls — a gone/busy peer
        whose close-ack never round-trips, where websockets' own close() would
        otherwise run to its internal ~10s close_timeout — force the underlying
        transport closed, which resolves the pending close() immediately.

        `asyncio.wait` is used (not `wait_for`) for the bound because websockets'
        close() resists cancellation; wait returns at the timeout regardless and
        we then force the transport rather than relying on cancelling close()."""
        close_task = asyncio.ensure_future(websocket.close())
        try:
            done, _pending = await asyncio.wait(
                {close_task}, timeout=self._SHUTDOWN_CLOSE_TIMEOUT_SECONDS,
            )
            if close_task not in done:
                # Graceful close stalled — force the transport (instant), which
                # unblocks the pending close(), then reap it.
                self._force_close_transport(websocket)
                try:
                    await asyncio.wait_for(
                        close_task,
                        timeout=self._SHUTDOWN_CLOSE_TIMEOUT_SECONDS,
                    )
                except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                    close_task.cancel()
        except Exception:  # noqa: BLE001
            self._force_close_transport(websocket)
            if not close_task.done():
                close_task.cancel()

    async def _handle_incoming(self, websocket: Any) -> None:
        """Handle a new incoming WebSocket connection."""
        peer = None
        try:
            # Wait for handshake
            raw = await asyncio.wait_for(websocket.recv(), timeout=self.handshake_timeout)
            msg = P2PMessage.from_json(raw)

            if msg.msg_type != MSG_HANDSHAKE:
                self._record_handshake_outcome(success=False, reason="Expected handshake")
                await websocket.close(1002, "Expected handshake")
                return

            peer_id = msg.sender_id
            if peer_id == self.identity.node_id:
                self._record_handshake_outcome(success=False, reason="Self-connection rejected")
                await websocket.close(1002, "Self-connection rejected")
                return

            ok, failure_reason = await self._validate_handshake_message(msg, require_ack_for=False)
            if not ok:
                await websocket.close(1002, failure_reason)
                return

            # Thread-safe check and add for peer connection
            async with self._peers_lock:
                if peer_id in self.peers:
                    self._record_handshake_outcome(success=False, reason="Already connected")
                    await websocket.close(1002, "Already connected")
                    return

            pub_key_b64 = msg.payload.get("public_key", "")

            peer = PeerConnection(
                peer_id=peer_id,
                address=f"{websocket.remote_address[0]}:{websocket.remote_address[1]}",
                websocket=websocket,
                public_key_b64=pub_key_b64,
                display_name=msg.payload.get("display_name", ""),
                roles=msg.payload.get("roles", []),
                outbound=False,
            )
            
            # Thread-safe peer addition
            async with self._peers_lock:
                # Double-check after acquiring lock (race condition prevention)
                if peer_id in self.peers:
                    self._record_handshake_outcome(success=False, reason="Already connected")
                    await websocket.close(1002, "Already connected")
                    return
                self.peers[peer_id] = peer
                peer_count = len(self.peers)

            # Send handshake acknowledgment
            ack = P2PMessage(
                msg_type=MSG_HANDSHAKE_ACK,
                sender_id=self.identity.node_id,
                payload={
                    "public_key": self.identity.public_key_b64,
                    "display_name": self.identity.display_name if hasattr(self.identity, 'display_name') else "",
                    "peer_count": peer_count,
                    # Bind ack to this exact handshake to prevent replay/downgrade.
                    "ack_for": msg.nonce,
                },
            )
            ack.sign(self.identity)
            await websocket.send(ack.to_json())

            logger.info(f"Peer connected (inbound): {peer_id[:8]}... from {peer.address}")
            self._record_handshake_outcome(success=True)
            await self._dispatch(
                P2PMessage(msg_type="peer_connected", sender_id=peer_id, payload={"direction": "inbound"}),
                peer,
            )

            # Message loop
            await self._read_loop(peer)

        except asyncio.TimeoutError:
            logger.debug("Incoming connection timed out during handshake")
            self._record_handshake_outcome(success=False, reason="Timeout")
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"Error handling incoming connection: {e}")
        finally:
            if peer:
                # Thread-safe peer removal
                async with self._peers_lock:
                    if peer.peer_id in self.peers:
                        del self.peers[peer.peer_id]
                        logger.info(f"Peer disconnected: {peer.peer_id[:8]}...")
                # Dispatch outside lock to prevent deadlock
                if peer:
                    await self._dispatch(
                        P2PMessage(msg_type="peer_disconnected", sender_id=peer.peer_id, payload={}),
                        peer,
                    )

    # ── Client (outgoing connections) ────────────────────────────

    async def connect_to_peer(self, address: str) -> Optional[PeerConnection]:
        """Initiate an outgoing connection to a peer at address (host:port).

        Addresses that already start with ``ws://`` or ``wss://`` are used
        as-is.  Plain ``host:port`` addresses default to ``ws://``, except
        when the port is 443 which implies TLS (``wss://``).
        """
        if address.startswith(("ws://", "wss://")):
            uri = address
        elif address.endswith(":443"):
            uri = f"wss://{address}"
        else:
            uri = f"ws://{address}"

        # R9 Phase 6.2: outbound connections flow through the configured
        # TransportAdapter. DirectAdapter (default) opens a native TCP
        # socket, preserving pre-R9 behavior. SocksAdapter routes through
        # a local SOCKS5 proxy (Tor / V2Ray / Trojan / Shadowsocks).
        # websockets supports `sock=` to accept a pre-connected socket.
        try:
            host, port = _parse_uri_host_port(uri)

            # R9 Phase 6.3: jurisdiction filter runs before any network I/O
            # so a blocked peer never gets a TCP SYN. Operators who
            # configured a filter (default None) can audit blocked peers
            # via the logged reason code.
            if self._jurisdiction_filter is not None:
                decision = self._jurisdiction_filter.evaluate(host)
                if not decision.allow:
                    logger.info(
                        "peer blocked by jurisdiction filter: host=%s reason=%s detected=%s",
                        host, decision.reason, decision.detected_jurisdiction,
                    )
                    return None

            adapter = self._transport_adapter
            # Sprint 627 — match server's 16MB max_size so large
            # chain-executor responses aren't dropped at the client.
            _WS_MAX_SIZE = 16 * 1024 * 1024
            if adapter.name == "direct":
                # Fast path: let websockets manage the connect itself.
                # Avoids the adapter round-trip for the common case.
                websocket = await websockets.client.connect(
                    uri, open_timeout=self.handshake_timeout,
                    max_size=_WS_MAX_SIZE,
                )
            else:
                sock = await adapter.open_connection(
                    host, port, timeout=self.handshake_timeout
                )
                websocket = await websockets.client.connect(
                    uri, sock=sock, open_timeout=self.handshake_timeout,
                    server_hostname=host,
                    max_size=_WS_MAX_SIZE,
                )

            # Send handshake
            hs = P2PMessage(
                msg_type=MSG_HANDSHAKE,
                sender_id=self.identity.node_id,
                payload={
                    "public_key": self.identity.public_key_b64,
                    "display_name": self.identity.display_name if hasattr(self.identity, 'display_name') else "",
                    "roles": [],
                },
            )
            hs.sign(self.identity)
            await websocket.send(hs.to_json())

            # Wait for ack
            raw = await asyncio.wait_for(websocket.recv(), timeout=self.handshake_timeout)
            ack = P2PMessage.from_json(raw)

            if ack.msg_type != MSG_HANDSHAKE_ACK:
                self._record_handshake_outcome(success=False, reason="Invalid handshake ack type")
                await websocket.close()
                return None

            ok, failure_reason = await self._validate_handshake_message(
                ack,
                require_ack_for=True,
                expected_ack_for=hs.nonce,
            )
            if not ok:
                self._record_handshake_outcome(success=False, reason=failure_reason)
                await websocket.close()
                return None

            peer = PeerConnection(
                peer_id=ack.sender_id,
                address=address,
                websocket=websocket,
                public_key_b64=ack.payload.get("public_key", ""),
                display_name=ack.payload.get("display_name", ""),
                outbound=True,
            )

            # Thread-safe check and add for peer connection
            async with self._peers_lock:
                if peer.peer_id in self.peers:
                    await websocket.close()
                    return self.peers[peer.peer_id]
                self.peers[peer.peer_id] = peer

            logger.info(f"Peer connected (outbound): {peer.peer_id[:8]}... at {address}")
            self._record_handshake_outcome(success=True)

            # Start reading in background
            task = asyncio.create_task(self._read_loop(peer))
            self._tasks.append(task)

            await self._dispatch(
                P2PMessage(msg_type="peer_connected", sender_id=peer.peer_id, payload={"direction": "outbound"}),
                peer,
            )
            return peer

        except Exception as e:
            logger.debug(f"Failed to connect to {address}: {e}")
            self._record_handshake_outcome(success=False, reason="other")
            return None

    @staticmethod
    def _derive_node_id_from_public_key(public_key_b64: str) -> Optional[str]:
        """Derive node_id from base64-encoded Ed25519 public key."""
        try:
            pub_bytes = base64.b64decode(public_key_b64)
        except Exception:
            return None
        return hashlib.sha256(pub_bytes).hexdigest()[:32]

    async def _validate_handshake_message(
        self,
        msg: P2PMessage,
        *,
        require_ack_for: bool,
        expected_ack_for: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Validate handshake/handshake-ack auth properties and reject weak/replay paths."""
        public_key_b64 = msg.payload.get("public_key", "") if isinstance(msg.payload, dict) else ""
        if not public_key_b64:
            self._record_handshake_outcome(success=False, reason="Missing public key")
            return False, "Missing public key"

        if not msg.signature:
            self._record_handshake_outcome(success=False, reason="Missing signature")
            return False, "Missing signature"

        expected_node_id = self._derive_node_id_from_public_key(public_key_b64)
        if not expected_node_id or expected_node_id != msg.sender_id:
            self._record_handshake_outcome(success=False, reason="Sender identity mismatch")
            return False, "Sender identity mismatch"

        if not verify_signature(public_key_b64, msg.to_bytes(), msg.signature):
            self._record_handshake_outcome(success=False, reason="Invalid signature")
            return False, "Invalid signature"

        if require_ack_for:
            ack_for = msg.payload.get("ack_for", "") if isinstance(msg.payload, dict) else ""
            if not ack_for:
                self._record_handshake_outcome(success=False, reason="Missing ack binding")
                return False, "Missing ack binding"
            if expected_ack_for is not None and ack_for != expected_ack_for:
                self._record_handshake_outcome(success=False, reason="Ack nonce mismatch")
                return False, "Ack nonce mismatch"

        if not msg.nonce:
            self._record_handshake_outcome(success=False, reason="Missing nonce")
            return False, "Missing nonce"

        # Replay protection applies to handshake and ack messages before promotion.
        async with self._nonces_lock:
            if msg.nonce in self._seen_nonces:
                self._record_handshake_outcome(success=False, reason="Replay nonce")
                return False, "Replay nonce"
            self._seen_nonces.add(msg.nonce)
            self._nonce_timestamps[msg.nonce] = time.time()

        return True, ""

    # ── Messaging ────────────────────────────────────────────────

    async def send_to_peer(self, peer_id: str, msg: P2PMessage) -> bool:
        """Send a message directly to a specific peer (thread-safe)."""
        async with self._peers_lock:
            peer = self.peers.get(peer_id)
            if not peer:
                return False
            # Get reference to websocket while holding lock
            websocket = peer.websocket
        
        try:
            msg.sign(self.identity)
            await websocket.send(msg.to_json())
            return True
        except Exception as e:
            logger.error(f"Failed to send to {peer_id[:8]}: {e}")
            return False

    async def broadcast(self, msg: P2PMessage) -> int:
        """Send a message to ALL connected peers (thread-safe)."""
        msg.sign(self.identity)
        raw = msg.to_json()
        
        # Get snapshot of peers under lock
        async with self._peers_lock:
            peers_snapshot = list(self.peers.values())
        
        sent = 0
        for peer in peers_snapshot:
            try:
                await peer.websocket.send(raw)
                sent += 1
            except Exception:
                pass
        return sent

    async def gossip(self, msg: P2PMessage, fanout: int = 3) -> int:
        """Send a message to a random subset of peers (gossip protocol, thread-safe)."""
        import random
        
        # Get snapshot of peers under lock
        async with self._peers_lock:
            targets = list(self.peers.values())
        
        if len(targets) > fanout:
            targets = random.sample(targets, fanout)

        msg.sign(self.identity)
        raw = msg.to_json()
        sent = 0
        for peer in targets:
            try:
                await peer.websocket.send(raw)
                sent += 1
            except Exception:
                pass
        return sent

    # ── Internal message loop ────────────────────────────────────

    def _allow_peer_message(self, peer_id: str, now: Optional[float] = None) -> bool:
        """sp936 — per-peer token bucket. Returns False when the peer has
        exceeded its message budget (caller drops the message). `burst` tokens
        refill at `rate`/sec, capped at `burst`. Sync + await-free, and each
        peer's bucket is touched only by that peer's read-loop, so no lock is
        needed."""
        now = time.time() if now is None else now
        bucket = self._peer_buckets.get(peer_id)
        if bucket is None:
            bucket = [self._peer_msg_burst, now]   # start full
            self._peer_buckets[peer_id] = bucket
        tokens = min(self._peer_msg_burst, bucket[0] + (now - bucket[1]) * self._peer_msg_rate)
        bucket[1] = now
        if tokens < 1.0:
            bucket[0] = tokens
            return False
        bucket[0] = tokens - 1.0
        return True

    async def _read_loop(self, peer: PeerConnection) -> None:
        """Read messages from a peer until disconnect."""
        try:
            async for raw in peer.websocket:
                try:
                    msg = P2PMessage.from_json(raw)
                    peer.last_seen = time.time()

                    # sp936 — drop messages from a peer exceeding its rate budget
                    # BEFORE dedup/dispatch, so a flood can't grow the nonce set,
                    # burn dispatch CPU, or be amplified by gossip re-propagation.
                    if not self._allow_peer_message(peer.peer_id):
                        self._telemetry["peer_rate_limited_total"] = (
                            self._telemetry.get("peer_rate_limited_total", 0) + 1
                        )
                        continue

                    # Thread-safe dedup by nonce
                    async with self._nonces_lock:
                        if msg.nonce in self._seen_nonces:
                            continue
                        self._seen_nonces.add(msg.nonce)
                        self._nonce_timestamps[msg.nonce] = time.time()

                    # Handle pings internally
                    if msg.msg_type == MSG_PING:
                        pong = P2PMessage(
                            msg_type=MSG_PONG,
                            sender_id=self.identity.node_id,
                            payload={"echo": msg.nonce},
                        )
                        pong.sign(self.identity)
                        await peer.websocket.send(pong.to_json())
                        continue

                    await self._dispatch(msg, peer)

                except json.JSONDecodeError:
                    logger.debug(f"Invalid JSON from {peer.peer_id[:8]}")
                except Exception as e:
                    logger.error(f"Error processing message from {peer.peer_id[:8]}: {e}")

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            # Thread-safe peer removal
            async with self._peers_lock:
                if peer.peer_id in self.peers:
                    del self.peers[peer.peer_id]
                    logger.info(f"Peer disconnected: {peer.peer_id[:8]}...")
            # sp936 — drop the peer's rate-limit bucket so the dict doesn't grow
            # with churned peers.
            self._peer_buckets.pop(peer.peer_id, None)
            # Dispatch outside lock to prevent deadlock
            await self._dispatch(
                P2PMessage(msg_type="peer_disconnected", sender_id=peer.peer_id, payload={}),
                peer,
            )

    async def _nonce_cleanup_loop(self) -> None:
        """Periodically clean up old nonces to prevent memory growth (thread-safe)."""
        while self._running:
            await asyncio.sleep(self.nonce_cleanup_interval)
            cutoff = time.time() - self._nonce_window
            
            # Thread-safe nonce cleanup
            async with self._nonces_lock:
                expired = [n for n, t in self._nonce_timestamps.items() if t < cutoff]
                for n in expired:
                    self._seen_nonces.discard(n)
                    del self._nonce_timestamps[n]
