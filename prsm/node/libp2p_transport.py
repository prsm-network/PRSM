"""
libp2p Transport Adapter
========================

Drop-in replacement for WebSocketTransport that delegates to a Go-based
libp2p shared library via ctypes FFI.  The Go daemon communicates back to
Python over a Unix Domain Socket (UDS) with a 4-byte length-prefixed JSON
protocol.

The shared library exposes 13 C functions (see libp2p/build/).
"""

import asyncio
import collections
import ctypes
import json
import logging
import os
import platform
import struct
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

from prsm.node.identity import NodeIdentity
from prsm.node.transport import P2PMessage, PeerConnection

logger = logging.getLogger(__name__)

# Type alias matching WebSocketTransport
MessageHandler = Callable[[P2PMessage, PeerConnection], Coroutine[Any, Any, None]]


class Libp2pTransportError(Exception):
    """Raised when the libp2p shared library cannot be loaded or returns an error."""


class Libp2pTransport:
    """libp2p-backed P2P transport layer.

    Provides the same public interface as ``WebSocketTransport`` so higher-level
    code can swap transports without changes.  Under the hood every network
    operation is forwarded to the Go shared library via ctypes.
    """

    # ── Construction ────────────────────────────────────────────

    def __init__(
        self,
        identity: NodeIdentity,
        host: str = "0.0.0.0",
        port: int = 9001,
        library_path: str = "",
        **kwargs: Any,
    ):
        self._identity = identity
        self.host = host
        self.port = port

        # Load shared library
        self._lib = self._load_library(library_path)
        self._setup_ctypes()

        # Runtime state
        self._handle: int = -1
        self._uds_path: str = ""
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._running = False

        # Sprint 1229 — application-level keepalive. A downstream chain
        # stage can idle for many minutes while upstream stages cold-load
        # multi-GB slices; without periodic traffic the libp2p stream is
        # reaped and the head's eventual dispatch fails with a transport
        # TimeoutError. A background task pings every connected peer every
        # ``_keepalive_interval`` seconds to keep the bidirectional stream
        # warm. Set PRSM_P2P_KEEPALIVE_INTERVAL_S<=0 to disable.
        self._keepalive_task: Optional[asyncio.Task] = None
        self._keepalive_interval: float = self._resolve_keepalive_interval()
        # Peers that dialed US (recorded from inbound frames). connect_to_peer
        # only records OUTBOUND dials in _peers; a node that just receives
        # dials (e.g. the pipeline head) would otherwise have nothing to ping.
        self._inbound_peer_ids: set = set()

        # Handler registry  (msg_type -> list[handler])
        self._handlers: Dict[str, List[MessageHandler]] = {}

        # Compatibility shim — keyed peer cache
        self._peers: Dict[str, PeerConnection] = {}

        # Telemetry counters (mirroring WebSocketTransport)
        self._telemetry: Dict[str, Any] = {
            "messages_sent": 0,
            "messages_received": 0,
            "publish_count": 0,
            "connect_count": 0,
            "error_count": 0,
            "dispatch_success_total": 0,
            "dispatch_failure_total": 0,
            "dispatch_failure_reasons": collections.Counter(),
            "keepalive_pings_sent": 0,
        }

    # ── Properties ──────────────────────────────────────────────

    @property
    def identity(self) -> NodeIdentity:
        return self._identity

    @property
    def peer_count(self) -> int:
        if self._handle < 0:
            return 0
        rc = self._lib.PrsmPeerCount(self._handle)
        return rc if rc >= 0 else 0

    @property
    def peer_addresses(self) -> List[str]:
        if self._handle < 0:
            return []
        ptr = self._lib.PrsmPeerList(self._handle)
        raw = self._read_and_free(ptr)
        if not raw:
            return []
        try:
            peers = json.loads(raw)
            return [p.get("addr", "") for p in peers if isinstance(p, dict)]
        except (json.JSONDecodeError, TypeError):
            return []

    @property
    def peers(self) -> Dict[str, PeerConnection]:
        """Compatibility shim — returns cached peer dict."""
        return self._peers

    def get_peer(self, node_id: str) -> Optional[PeerConnection]:
        """Return the PeerConnection for a node_id, or None if not connected.

        Sync snapshot for Phase 2 RemoteShardDispatcher's peer-resolve step.
        """
        return self._peers.get(node_id)

    # ── Lifecycle ───────────────────────────────────────────────

    async def start(self) -> None:
        """Start the Go libp2p daemon and UDS reader."""
        self._loop = asyncio.get_running_loop()

        # Build UDS path
        self._uds_path = os.path.join(
            tempfile.gettempdir(),
            f"prsm_p2p_{self._identity.node_id[:8]}_{uuid.uuid4().hex[:8]}.sock",
        )

        # Build 64-byte Ed25519 key (seed || public)
        seed = self._identity.private_key_bytes  # 32 bytes
        pub = self._identity.public_key_bytes     # 32 bytes
        ed25519_key_64 = seed + pub

        handle = self._lib.PrsmStart(
            ed25519_key_64,
            self.port,
            b"",  # no bootstrap addrs
            self._uds_path.encode("utf-8"),
        )
        if handle < 0:
            raise Libp2pTransportError("PrsmStart returned negative handle")
        self._handle = handle
        self._running = True

        # Give Go time to bind the UDS
        await asyncio.sleep(0.3)

        # Start UDS reader
        self._reader_task = asyncio.create_task(self._uds_reader())

        # Sprint 1229 — start the keepalive loop (no-op if disabled)
        if self._keepalive_interval > 0:
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())

        logger.info(
            "libp2p transport started  handle=%d  uds=%s  port=%d  "
            "keepalive=%.1fs",
            self._handle, self._uds_path, self.port,
            self._keepalive_interval,
        )

    async def stop(self) -> None:
        """Stop the Go daemon, cancel the reader, clean up."""
        self._running = False

        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
            self._keepalive_task = None

        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

        if self._handle >= 0:
            self._lib.PrsmStop(self._handle)
            self._handle = -1

        # Clean up UDS socket file
        if self._uds_path:
            try:
                os.unlink(self._uds_path)
            except FileNotFoundError:
                pass
            self._uds_path = ""

        self._peers.clear()
        logger.info("libp2p transport stopped")

    # ── Keepalive (Sprint 1229) ─────────────────────────────────

    @staticmethod
    def _resolve_keepalive_interval() -> float:
        """Seconds between keepalive pings. Default 30s — well under any
        cloud TCP / libp2p idle reaper, negligible overhead. ``<=0`` (or a
        garbage value resolving to non-positive) disables; an unparseable
        value falls back to the default."""
        raw = (os.environ.get("PRSM_P2P_KEEPALIVE_INTERVAL_S", "") or "").strip()
        if not raw:
            return 30.0
        try:
            val = float(raw)
        except ValueError:
            return 30.0
        return val if val > 0 else 0.0

    @staticmethod
    def _is_keepalive(raw: dict) -> bool:
        """True if an inbound UDS frame is a keepalive ping — in EITHER
        framing: a bare ``msg_type="keepalive_ping"`` frame, or the libp2p
        direct-protocol wrapper ``{"msg_type":"direct","data":<inner-json>}``
        where the inner message is a keepalive. Used to short-circuit before
        handler fan-out so the ping never errors in (and inflates the failure
        counters of) the registered direct-message handlers."""
        if raw.get("msg_type") == "keepalive_ping" or raw.get("type") == "keepalive_ping":
            return True
        p = raw.get("payload")
        if isinstance(p, dict) and p.get("subtype") == "keepalive_ping":
            return True
        data = raw.get("data")
        if isinstance(data, (str, bytes)):
            # Cheap guard: skip the json.loads for frames that can't be a
            # keepalive (avoids parsing every inbound direct message).
            marker = b"keepalive_ping" if isinstance(data, bytes) else "keepalive_ping"
            if marker not in data:
                return False
            try:
                inner = json.loads(data)
            except (json.JSONDecodeError, TypeError, ValueError):
                return False
            if isinstance(inner, dict):
                if inner.get("msg_type") == "keepalive_ping":
                    return True
                ip = inner.get("payload")
                if isinstance(ip, dict) and ip.get("subtype") == "keepalive_ping":
                    return True
        return False

    async def _ping_all_peers(self) -> None:
        """Ping every peer we know — OUTBOUND dials (``self._peers``) AND
        INBOUND-seen peers (``self._inbound_peer_ids``).

        The inbound set is load-bearing: a node that only RECEIVES dials (the
        pipeline head — workers dial into it) has an empty ``_peers`` and would
        otherwise warm nothing toward the very workers it must dispatch to. A
        ping opens (and the Go side closes) a short-lived ``/prsm/direct/1.0.0``
        stream, which resets the underlying connection's idle timer in BOTH
        directions, so the head's later dispatch reuses a live connection
        instead of hitting a reaped one. Point-to-point, never gossiped. A
        failure to one peer must never abort the sweep; an inbound-only peer
        that fails is pruned (it's gone). (Relies on off-event-loop slice
        loading — sp1227 — so this task actually gets scheduled mid-cold-load.)
        """
        targets = set(self._peers.keys()) | set(self._inbound_peer_ids)
        for peer_id in targets:
            try:
                ping = P2PMessage(
                    msg_type="keepalive_ping",
                    sender_id=self._identity.node_id,
                    payload={"subtype": "keepalive_ping", "ts": time.time()},
                )
                ok = await self.send_to_peer(peer_id, ping)
                if ok:
                    self._telemetry["keepalive_pings_sent"] += 1
                elif peer_id in self._inbound_peer_ids and peer_id not in self._peers:
                    self._inbound_peer_ids.discard(peer_id)
            except Exception as exc:  # noqa: BLE001 — never let one peer kill the loop
                logger.debug("keepalive ping to %s failed: %s", peer_id, exc)
                if peer_id not in self._peers:
                    self._inbound_peer_ids.discard(peer_id)

    async def _keepalive_loop(self) -> None:
        """Background task: ping all peers every ``_keepalive_interval`` s.

        Self-healing — the try/except is INSIDE the loop so a single bad sweep
        is logged and the loop keeps running. A resilience mechanism must not
        be permanently disabled by one transient error.
        """
        while self._running:
            try:
                await asyncio.sleep(self._keepalive_interval)
                if not self._running:
                    break
                await self._ping_all_peers()
            except asyncio.CancelledError:
                raise  # let cancellation propagate (stop() awaits it)
            except Exception as exc:  # noqa: BLE001 — a bad sweep must not kill keepalive
                logger.error("keepalive sweep error (continuing): %s", exc)

    # ── Peer operations ─────────────────────────────────────────

    async def connect_to_peer(self, address: str) -> Optional[PeerConnection]:
        """Connect to a peer by address (host:port, ws(s)://…, or /ip4/…).

        Multiaddr MUST include a `/p2p/<peerID>` suffix for libp2p
        to identify the remote. Operators using the canonical
        bootstrap registry get this automatically; raw URLs from
        config (e.g. `wss://bootstrap1.prsm-network.com:8765`)
        without the suffix fail at the C-bridge layer with a
        cryptic "invalid p2p multiaddr" message.

        We detect the missing suffix here and log a clear,
        actionable warning instead of the raw error.
        """
        if self._handle < 0:
            return None

        maddr = self._to_multiaddr(address)
        if "/p2p/" not in maddr:
            # Sprint 479: demote to debug for ws:// / wss:// inputs.
            # The canonical PRSM bootstrap fleet uses
            # BootstrapClient's register/heartbeat WS protocol —
            # NOT full libp2p. Libp2pDiscovery.bootstrap() calls
            # this method first as a libp2p-native probe and falls
            # back to _try_bootstrap_client on ws:// URLs (which
            # don't need /p2p/<peerID>). Emitting a WARNING for
            # ws:// inputs misled operators into thinking the
            # bootstrap config was broken when it was actually
            # working via the documented fallback.
            #
            # Real libp2p multiaddrs (/dns4/.../tcp/.../ws or
            # /ip4/...) STILL warn — those genuinely need /p2p
            # for the C-bridge connect to succeed.
            is_ws_fallback_path = (
                address.startswith("ws://")
                or address.startswith("wss://")
            )
            if is_ws_fallback_path:
                logger.debug(
                    "Libp2pTransport: skipping libp2p-native "
                    "connect for ws:// bootstrap %r — "
                    "BootstrapClient WS fallback will handle.",
                    address,
                )
            else:
                logger.warning(
                    "Libp2pTransport: bootstrap multiaddr %r is "
                    "missing /p2p/<peerID> suffix — libp2p connect "
                    "requires it. Use the bootstrap registry "
                    "(signed bootstrap.json) to resolve full "
                    "multiaddrs, or append /p2p/<id> manually if "
                    "the peer ID is known.",
                    maddr,
                )
                self._telemetry["error_count"] += 1
            return None

        ptr = self._lib.PrsmConnect(self._handle, maddr.encode("utf-8"))
        peer_id_str = self._read_and_free(ptr)
        if not peer_id_str:
            self._telemetry["error_count"] += 1
            return None

        peer = PeerConnection(
            peer_id=peer_id_str,
            address=address,
            websocket=None,  # no WS object in libp2p mode
            outbound=True,
        )
        self._peers[peer_id_str] = peer
        self._telemetry["connect_count"] += 1
        return peer

    # ── Messaging ───────────────────────────────────────────────

    async def send_to_peer(self, peer_id: str, msg: P2PMessage) -> bool:
        """Send a direct message to a specific peer."""
        if self._handle < 0:
            return False
        msg.sign(self._identity)
        data = msg.to_json().encode("utf-8")
        rc = self._lib.PrsmSend(
            self._handle,
            peer_id.encode("utf-8"),
            b"/prsm/direct/1.0.0",
            data,
            len(data),
        )
        if rc == 0:
            self._telemetry["messages_sent"] += 1
            return True
        self._telemetry["error_count"] += 1
        return False

    async def broadcast(self, msg: P2PMessage) -> int:
        """Broadcast via gossip (compatibility alias)."""
        return await self.gossip(msg)

    async def gossip(self, msg: P2PMessage, fanout: int = 3) -> int:
        """Publish a message over GossipSub."""
        if self._handle < 0:
            return 0

        msg.sign(self._identity)

        # Derive topic from payload subtype
        subtype = ""
        if isinstance(msg.payload, dict):
            subtype = msg.payload.get("subtype", msg.payload.get("type", "general"))
        if not subtype:
            subtype = "general"
        topic = f"prsm/{subtype}"

        data = msg.to_json().encode("utf-8")
        rc = self._lib.PrsmPublish(
            self._handle,
            topic.encode("utf-8"),
            data,
            len(data),
        )
        if rc == 0:
            self._telemetry["publish_count"] += 1
            self._telemetry["messages_sent"] += 1
            return 1
        self._telemetry["error_count"] += 1
        return 0

    # ── Handler registration ────────────────────────────────────

    def on_message(self, msg_type: str, handler: MessageHandler) -> None:
        """Register a callback for a given message type."""
        self._handlers.setdefault(msg_type, []).append(handler)

    # ── Telemetry ───────────────────────────────────────────────

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Return a stable copy of transport telemetry counters."""
        return {
            "messages_sent": int(self._telemetry["messages_sent"]),
            "messages_received": int(self._telemetry["messages_received"]),
            "publish_count": int(self._telemetry["publish_count"]),
            "connect_count": int(self._telemetry["connect_count"]),
            "error_count": int(self._telemetry["error_count"]),
            "dispatch_success_total": int(self._telemetry["dispatch_success_total"]),
            "dispatch_failure_total": int(self._telemetry["dispatch_failure_total"]),
            "dispatch_failure_reasons": dict(self._telemetry["dispatch_failure_reasons"]),
            "keepalive_pings_sent": int(self._telemetry["keepalive_pings_sent"]),
        }

    # ── DHT helpers ─────────────────────────────────────────────

    async def dht_provide(self, key: str) -> bool:
        if self._handle < 0:
            return False
        return self._lib.PrsmDHTProvide(self._handle, key.encode("utf-8")) == 0

    async def dht_find_providers(self, key: str, limit: int = 20) -> List[str]:
        if self._handle < 0:
            return []
        ptr = self._lib.PrsmDHTFindProviders(
            self._handle, key.encode("utf-8"), limit,
        )
        raw = self._read_and_free(ptr)
        if not raw:
            return []
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []

    async def get_nat_status(self) -> str:
        if self._handle < 0:
            return "unknown"
        ptr = self._lib.PrsmGetNATStatus(self._handle)
        return self._read_and_free(ptr) or "unknown"

    # ── UDS reader (data plane from Go) ─────────────────────────

    async def _uds_reader(self) -> None:
        """Connect to the Go daemon's UDS and read length-prefixed JSON frames."""
        try:
            reader, _ = await asyncio.open_unix_connection(self._uds_path)
            while self._running:
                # 4-byte big-endian length prefix
                header = await reader.readexactly(4)
                length = struct.unpack(">I", header)[0]
                payload = await reader.readexactly(length)
                try:
                    raw = json.loads(payload.decode("utf-8"))
                    await self._dispatch(raw)
                except (json.JSONDecodeError, Exception) as exc:
                    logger.debug("UDS frame decode error: %s", exc)
        except asyncio.IncompleteReadError:
            logger.debug("UDS reader: Go side closed the connection")
        except asyncio.CancelledError:
            raise  # let cancellation propagate
        except ConnectionRefusedError:
            logger.warning("UDS reader: could not connect to %s", self._uds_path)
        except Exception as exc:
            logger.error("UDS reader error: %s", exc)

    async def _dispatch(self, raw: dict) -> None:
        """Route an incoming UDS frame to registered handlers."""
        msg_type = raw.get("msg_type", raw.get("type", ""))
        if not msg_type:
            return

        self._telemetry["messages_received"] += 1

        # Sprint 1229 — remember peers that dial US so the keepalive sweep can
        # warm the reverse direction. The head never adds inbound dialers to
        # _peers; without this it would ping nothing toward its workers.
        sender_id = raw.get("sender_id", raw.get("from", ""))
        if sender_id and sender_id != self._identity.node_id:
            self._inbound_peer_ids.add(sender_id)

        # Sprint 1229 — a keepalive ping is pure connection-warming traffic;
        # short-circuit BEFORE handler fan-out (it is delivered over the
        # /prsm/direct/1.0.0 protocol, so it would otherwise reach every
        # registered direct-message handler, error on the wrapped payload, and
        # inflate dispatch_failure_total / the sp1217 runtime metrics).
        if self._is_keepalive(raw):
            return

        # Reconstruct P2PMessage.
        # sp1235 — normalize the wire payload to a dict. The libp2p UDS frame
        # carries the inner P2PMessage as a JSON STRING in `data` (send_to_peer/
        # gossip put msg.to_json() on the wire); parse it and use the inner
        # payload dict + metadata, matching WebSocketTransport.from_json so
        # handlers get an identical msg across backends. Routing stays on the
        # OUTER msg_type — unchanged. (Was: payload assigned the raw string →
        # every direct handler's payload.get(...) AttributeError'd; the gossip
        # handler json.loads'd it itself, hiding the asymmetry.)
        _payload = raw.get("payload", raw.get("data", {}))
        _sender = raw.get("sender_id", raw.get("from", ""))
        _sig = raw.get("signature", "")
        _ts = raw.get("timestamp", time.time())
        _ttl = raw.get("ttl", 5)
        _nonce = raw.get("nonce", uuid.uuid4().hex[:16])
        if isinstance(_payload, (str, bytes)):
            try:
                _inner = json.loads(_payload)
            except (json.JSONDecodeError, TypeError, ValueError):
                _inner = None
            if isinstance(_inner, dict):
                # match WebSocketTransport.from_json: payload = inner["payload"]
                _payload = _inner.get("payload", {})
                # adopt inner-message metadata (matches from_json / websocket).
                # NOTE: inbound-peer warming (sp1229, above) deliberately keeps
                # the OUTER libp2p sender_id — only the handler-facing msg below
                # uses the inner sender_id.
                _sender = _inner.get("sender_id", _sender)
                _sig = _inner.get("signature", _sig)
                _ts = _inner.get("timestamp", _ts)
                _ttl = _inner.get("ttl", _ttl)
                _nonce = _inner.get("nonce", _nonce)
            else:
                _payload = {}                       # malformed inner JSON → empty dict
        # sp1235 — guarantee handlers always get a dict (the P2PMessage payload
        # contract). Defends against a non-dict wire payload of ANY shape: a
        # JSON null/list/scalar in `data` (which skips the str block above), or
        # a non-dict inner "payload". Without this, such a payload reaches a
        # handler and its payload.get(...) errors into dispatch_failure_total.
        if not isinstance(_payload, dict):
            _payload = {}
        try:
            msg = P2PMessage(
                msg_type=msg_type,
                sender_id=_sender,
                payload=_payload,
                timestamp=_ts,
                signature=_sig,
                ttl=_ttl,
                nonce=_nonce,
            )
        except Exception as exc:
            logger.debug("Failed to reconstruct P2PMessage: %s", exc)
            return

        # Build a stub PeerConnection for the sender
        peer = self._peers.get(msg.sender_id) or PeerConnection(
            peer_id=msg.sender_id,
            address=raw.get("remote_addr", ""),
            websocket=None,
        )

        handlers = self._handlers.get(msg_type, [])
        for handler in handlers:
            try:
                await handler(msg, peer)
                self._telemetry["dispatch_success_total"] += 1
            except Exception as exc:
                logger.error("Handler error for %s: %s", msg_type, exc)
                self._telemetry["dispatch_failure_total"] += 1
                self._telemetry["dispatch_failure_reasons"][
                    type(exc).__name__
                ] += 1

    # ── Internal helpers ────────────────────────────────────────

    @staticmethod
    def _load_library(path: str) -> ctypes.CDLL:
        """Locate and load the Go shared library.

        Search order:
        1. Explicit *path* argument.
        2. ``libp2p/build/`` relative to the repository root (two levels up
           from this file).
        """
        if path:
            try:
                return ctypes.CDLL(path)
            except OSError as exc:
                raise Libp2pTransportError(
                    f"Cannot load shared library at {path}: {exc}"
                ) from exc

        # Auto-detect from repo layout
        system = platform.system().lower()
        machine = platform.machine().lower()

        if system == "darwin":
            ext = "dylib"
        elif system == "windows":
            ext = "dll"
        else:
            ext = "so"

        arch = "arm64" if machine in ("arm64", "aarch64") else "amd64"
        lib_name = f"libprsm_p2p_{system}_{arch}.{ext}"

        repo_root = Path(__file__).resolve().parent.parent.parent
        lib_path = repo_root / "libp2p" / "build" / lib_name

        try:
            return ctypes.CDLL(str(lib_path))
        except OSError as exc:
            raise Libp2pTransportError(
                f"Cannot load shared library at {lib_path}: {exc}"
            ) from exc

    def _setup_ctypes(self) -> None:
        """Declare argtypes / restype for every exported C symbol."""
        L = self._lib  # noqa: N806

        # PrsmStart(ed25519Key *C.char, listenPort C.int,
        #           bootstrapAddrs *C.char, udsPath *C.char) → C.int
        L.PrsmStart.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                ctypes.c_char_p, ctypes.c_char_p]
        L.PrsmStart.restype = ctypes.c_int

        # PrsmStop(handle C.int) → C.int
        L.PrsmStop.argtypes = [ctypes.c_int]
        L.PrsmStop.restype = ctypes.c_int

        # PrsmPeerCount(handle C.int) → C.int
        L.PrsmPeerCount.argtypes = [ctypes.c_int]
        L.PrsmPeerCount.restype = ctypes.c_int

        # PrsmConnect(handle C.int, multiaddr *C.char) → *C.char
        L.PrsmConnect.argtypes = [ctypes.c_int, ctypes.c_char_p]
        L.PrsmConnect.restype = ctypes.c_void_p

        # PrsmFree(ptr *C.char) → void
        L.PrsmFree.argtypes = [ctypes.c_void_p]
        L.PrsmFree.restype = None

        # PrsmPublish(handle, topic, data, dataLen) → C.int
        L.PrsmPublish.argtypes = [ctypes.c_int, ctypes.c_char_p,
                                  ctypes.c_char_p, ctypes.c_int]
        L.PrsmPublish.restype = ctypes.c_int

        # PrsmSubscribe(handle, topic) → C.int
        L.PrsmSubscribe.argtypes = [ctypes.c_int, ctypes.c_char_p]
        L.PrsmSubscribe.restype = ctypes.c_int

        # PrsmUnsubscribe(handle, topic) → C.int
        L.PrsmUnsubscribe.argtypes = [ctypes.c_int, ctypes.c_char_p]
        L.PrsmUnsubscribe.restype = ctypes.c_int

        # PrsmSend(handle, peerID, proto, data, dataLen) → C.int
        L.PrsmSend.argtypes = [ctypes.c_int, ctypes.c_char_p,
                               ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        L.PrsmSend.restype = ctypes.c_int

        # PrsmDHTProvide(handle, key) → C.int
        L.PrsmDHTProvide.argtypes = [ctypes.c_int, ctypes.c_char_p]
        L.PrsmDHTProvide.restype = ctypes.c_int

        # PrsmDHTFindProviders(handle, key, limit) → *C.char
        L.PrsmDHTFindProviders.argtypes = [ctypes.c_int, ctypes.c_char_p,
                                           ctypes.c_int]
        L.PrsmDHTFindProviders.restype = ctypes.c_void_p

        # PrsmGetNATStatus(handle) → *C.char
        L.PrsmGetNATStatus.argtypes = [ctypes.c_int]
        L.PrsmGetNATStatus.restype = ctypes.c_void_p

        # PrsmPeerList(handle) → *C.char
        L.PrsmPeerList.argtypes = [ctypes.c_int]
        L.PrsmPeerList.restype = ctypes.c_void_p

    def _read_and_free(self, ptr: Any) -> str:
        """Decode a C string returned by Go, call PrsmFree, return Python str."""
        if not ptr:
            return ""
        try:
            value = ctypes.string_at(ptr).decode("utf-8")
        except Exception:
            value = ""
        self._lib.PrsmFree(ptr)
        return value

    @staticmethod
    def _to_multiaddr(address: str) -> str:
        """Translate common address formats to libp2p multiaddr strings.

        Supported inputs:
        - ``wss://host:port`` or ``ws://host:port``
            → ``/{ip4|dns4}/host/tcp/port/ws`` based on whether
              host is an IPv4 literal or a DNS hostname
        - ``host:port`` → ``/{ip4|dns4}/host/udp/port/quic-v1``
        - ``/ip4/...``, ``/ip6/...``, ``/dns/...``, ``/dns4/...``,
          ``/dns6/...`` → passthrough
        """
        if (
            address.startswith("/ip4/")
            or address.startswith("/ip6/")
            or address.startswith("/dns/")
            or address.startswith("/dns4/")
            or address.startswith("/dns6/")
        ):
            return address

        if address.startswith("wss://") or address.startswith("ws://"):
            # Strip scheme
            without_scheme = address.split("://", 1)[1]
            host, _, port = without_scheme.rpartition(":")
            if not host:
                host = without_scheme
                port = "443" if address.startswith("wss") else "80"
            return f"/{Libp2pTransport._addr_proto(host)}/{host}/tcp/{port}/ws"

        # Plain host:port → QUIC
        if ":" in address:
            host, port = address.rsplit(":", 1)
            return (
                f"/{Libp2pTransport._addr_proto(host)}/{host}"
                f"/udp/{port}/quic-v1"
            )

        # Fallback — treat as hostname only
        return (
            f"/{Libp2pTransport._addr_proto(address)}/{address}"
            f"/udp/9001/quic-v1"
        )

    @staticmethod
    def _addr_proto(host: str) -> str:
        """Pick `ip4` or `dns4` based on whether host is an IPv4
        literal. libp2p's `/ip4/` requires a dotted-quad; DNS
        hostnames need `/dns4/`."""
        import ipaddress
        try:
            ipaddress.IPv4Address(host)
            return "ip4"
        except (ipaddress.AddressValueError, ValueError):
            return "dns4"
