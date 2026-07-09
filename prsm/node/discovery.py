"""
Peer Discovery
==============

Bootstrap-based peer discovery with gossip propagation.
Nodes connect to bootstrap peers, request their peer lists,
and periodically share their own presence on the network.
"""

import asyncio
import base64
import collections
import hashlib
import json
import logging
import os
import random
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from prsm.node.identity import verify_signature

from prsm.node.transport import (
    MSG_GOSSIP,
    P2PMessage,
    PeerConnection,
    WebSocketTransport,
)

logger = logging.getLogger(__name__)

# Discovery-specific message subtypes (carried in payload["subtype"])
DISCOVERY_ANNOUNCE = "discovery_announce"
DISCOVERY_PEER_REQUEST = "discovery_peer_request"
DISCOVERY_PEER_RESPONSE = "discovery_peer_response"
DISCOVERY_CAPABILITY_ANNOUNCE = "capability_announce"


@dataclass
class PeerInfo:
    """Lightweight peer descriptor shared during discovery."""
    node_id: str
    address: str
    display_name: str = ""
    roles: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)  # e.g. ["inference", "embedding", "benchmark"]
    supported_backends: List[str] = field(default_factory=list)  # e.g. ["anthropic", "openai", "local"]
    gpu_available: bool = False
    last_seen: float = field(default_factory=time.time)
    last_capability_update: float = field(default_factory=time.time)
    job_success_count: int = 0
    job_failure_count: int = 0
    last_failure_time: float = 0.0
    startup_timestamp: float = 0.0
    # sp941 — the signed announce_time of the last ACCEPTED announce from this
    # peer (monotonic replay defense). An announce whose announce_time is not
    # strictly newer is rejected, so a captured genuine announce replayed after
    # the transport nonce-dedup window cannot re-assert a stale address. 0.0
    # means no attested announce seen yet (legacy/no-timestamp announces).
    last_announce_time: float = 0.0
    # sp1005 — the portable, self-verifying credential from this peer's last
    # ATTESTED announce ({node_id, signed_payload, nonce, origin_pubkey,
    # origin_sig}). Stored so it can be RE-EMITTED via authenticated PEX
    # (_handle_peer_request) and re-verified by the receiver. None for peers
    # learned via a direct/legacy path with no attestation (not PEX-relayable).
    announce_credential: Optional[Dict[str, Any]] = None
    # Sprint 1088 — per-peer node_id authentication marker. None → "use the discovery
    # transport's default" (the coarse Libp2pDiscovery/PeerDiscovery.node_id_authenticated
    # the sp1085 pool builder reads). True/False overrides it for THIS peer: set True when
    # the peer's node_id was cryptographically authenticated (an attested gossip announce,
    # sp1086/1087, or a verified bootstrap-relay credential, sp1088), False when an entry
    # was learned WITHOUT authentication. The pool provider honors an attestation-derived
    # hardware tier only for authenticated peers (sp1083 binding is meaningless otherwise).
    node_id_authenticated: Optional[bool] = None
    # Sprint 680 — opt-in hardware advertisement. Carries serialized
    # HardwareProfile.to_dict() (or a subset). Consumed by the DHT-
    # backed GpuPoolProvider (sprint 681+) to construct ParallaxGPU
    # entries — peers without this field are excluded from the pool.
    hardware_profile: Optional[Dict[str, Any]] = None

    @property
    def reliability_score(self) -> float:
        """Compute reliability as success ratio. New peers get benefit of the doubt (1.0)."""
        total = self.job_success_count + self.job_failure_count
        if total == 0:
            return 1.0
        return self.job_success_count / total


def validate_bootstrap_address(address: str) -> Tuple[bool, str]:
    """Validate that a bootstrap address is well-formed.

    Returns (is_valid, reason).  Accepts formats:
      - wss://host:port  or  ws://host:port
      - host:port  (bare host:port, port must be numeric)
      - hostname only (accepted, uses default port)

    Rejects empty strings, whitespace-only, addresses with no parseable
    host, and URLs with unsupported schemes.
    """
    if not address or not address.strip():
        return False, "empty address"

    address = address.strip()

    # URL form
    if "://" in address:
        try:
            parsed = urlparse(address)
        except Exception as exc:
            return False, f"unparseable URL ({exc})"

        if parsed.scheme not in ("ws", "wss"):
            return False, f"unsupported scheme '{parsed.scheme}'"
        if not parsed.hostname:
            return False, "missing hostname"
        return True, ""

    # Bare host:port form
    host, sep, port_str = address.rpartition(":")
    if sep and host:
        try:
            port = int(port_str)
            if not (1 <= port <= 65535):
                return False, f"port {port} out of range"
        except ValueError:
            return False, f"non-numeric port '{port_str}'"
        return True, ""

    # Single token (hostname only, no port) — accept
    return True, ""


def _rewrite_co_located_address(
    peer_address: str,
    own_advertise: "str | None",
) -> str:
    """Sprint 781 — F14 fix: auto-loopback-rewrite for co-located peers.

    When a peer announces an address whose host portion matches our
    own announced address, we MUST be co-located (two daemons on the
    same host both reaching the world through the same external IP).
    The OS typically can't loopback to its own external IP through
    NAT (no NAT hairpin pinning), so direct dial fails.

    Fix: rewrite the dial target's host portion to ``127.0.0.1`` while
    preserving the port. The OS routes the localhost dial directly
    to the other daemon listening on that port.

    Returns the address unchanged when:
    - ``own_advertise`` is None (we don't know our own announced IP
      yet — can't detect co-location)
    - peer's host portion differs from ours (multi-host case)
    - peer_address is empty or starts with 0.0.0.0 (caller filtered
      it already)
    - peer_address has no port (no information to dial via loopback)
    """
    if not peer_address or peer_address.startswith("0.0.0.0"):
        return peer_address
    if own_advertise is None:
        return peer_address

    # Strip optional port suffix on own_advertise (legacy callers
    # may pass "ip:port"). We compare hosts only.
    own_host = own_advertise.split(":", 1)[0]

    if ":" not in peer_address:
        return peer_address

    peer_host, peer_port = peer_address.rsplit(":", 1)
    if peer_host != own_host:
        return peer_address

    return f"127.0.0.1:{peer_port}"


# ── sp937: authenticated discovery announces ──────────────────────────────
# Discovery announces are MSG_GOSSIP, which sp731 excludes from sender_id
# re-binding, and there is no per-message signature check — so a gossip
# announce's `sender_id` is attacker-controlled. The handlers keyed known_peers
# on it, letting a peer forge `sender_id=victim` + a malicious address to poison
# the routing table (eclipse). An announce now carries a self-verifying
# attestation (node_id == sha256(pubkey)[:32], plus a signature over the
# content); the handler trusts the claimed node_id only if it verifies, else
# accepts it only when it equals the handshake-authenticated peer.peer_id, else
# drops it. Mirrors the sp934 gossip-origin fix.

_ANNOUNCE_ATTEST_KEYS = ("origin_pubkey", "origin_sig")


def _announce_signing_bytes(node_id: str, payload: Dict[str, Any], nonce: str) -> bytes:
    """Canonical bytes an announcer signs / a receiver verifies. Covers the full
    announce content (minus the attestation fields) bound to node_id + nonce, so
    a relayer cannot tamper with the advertised address/capabilities."""
    content = {k: v for k, v in payload.items() if k not in _ANNOUNCE_ATTEST_KEYS}
    return json.dumps(
        {"node_id": node_id, "payload": content, "nonce": nonce},
        sort_keys=True, separators=(",", ":"),
    ).encode()


def _attest_announce_payload(identity, payload: Dict[str, Any], nonce: str) -> None:
    """Add a self-verifying origin attestation to a discovery announce payload."""
    payload["origin_pubkey"] = identity.public_key_b64
    payload["origin_sig"] = identity.sign(
        _announce_signing_bytes(identity.node_id, payload, nonce)
    )


def _authenticated_announce_node_id(msg: "P2PMessage", peer: "PeerConnection") -> Optional[str]:
    """Authenticated identity for a discovery announce, or None to DROP it."""
    claimed = msg.sender_id
    pubkey = msg.payload.get("origin_pubkey", "")
    sig = msg.payload.get("origin_sig", "")
    if pubkey and sig:
        try:
            derived = hashlib.sha256(base64.b64decode(pubkey)).hexdigest()[:32]
            if derived == claimed and verify_signature(
                pubkey, _announce_signing_bytes(claimed, msg.payload, msg.nonce), sig
            ):
                return claimed
        except Exception:
            pass
    # Not attested: accept only a DIRECT announce from the handshake-
    # authenticated peer; never the raw, unverified gossip sender_id.
    authenticated_peer = getattr(peer, "peer_id", None)
    if authenticated_peer and claimed == authenticated_peer:
        return claimed
    return None


def _announce_is_stale_replay(payload: Dict[str, Any], prev_announce_time: float) -> bool:
    """sp941 — True iff the announce carries a signed announce_time that is NOT
    strictly newer than the last ACCEPTED one for this peer — i.e. a replay of a
    captured genuine announce (e.g. after the transport nonce-dedup window). The
    timestamp is covered by the sp937 attestation, so it can't be forged. Legacy
    announces (no announce_time) are not treated as replays here — sp937 already
    drops their forged/relayed variants."""
    ts = payload.get("announce_time")
    if ts is None:
        return False
    try:
        return float(ts) <= float(prev_announce_time)
    except (TypeError, ValueError):
        return False


# ── sp1005: authenticated peer-exchange (PEX) ─────────────────────────────
# sp937/sp941 authenticated the node's OWN announce (_handle_announce /
# _handle_capability_announce) but the peer-EXCHANGE response
# (_handle_peer_response) — by which a node relays OTHER peers it knows — was
# left unauthenticated, so a connected peer could inject attacker-chosen
# (node_id → address / hardware_profile) entries into known_peers (eclipse +
# Parallax-pool poisoning + unbounded-memory DoS). The relayer is not the
# authority for the entries it forwards, so the fix is a PER-ENTRY portable
# credential: each peer signs a self-verifying record (its node_id derives from
# its pubkey, and a signature covers the announce content), which any relayer
# can forward and any receiver re-verifies. This reuses the sp937 announce
# attestation primitive (_announce_signing_bytes) — a credential is exactly the
# verbatim signed material of an authenticated announce, made portable.

_DEFAULT_MAX_KNOWN_PEERS = 2048


def _max_known_peers() -> int:
    """Upper bound on the known_peers routing table (memory + eclipse-magnitude
    defense). PRSM_MAX_KNOWN_PEERS overrides; falls back to the default on a
    missing / non-positive / unparseable value."""
    raw = os.environ.get("PRSM_MAX_KNOWN_PEERS", "").strip()
    if not raw:
        return _DEFAULT_MAX_KNOWN_PEERS
    try:
        val = int(raw)
    except (ValueError, TypeError):
        return _DEFAULT_MAX_KNOWN_PEERS
    return val if val > 0 else _DEFAULT_MAX_KNOWN_PEERS


def _build_announce_credential(node_id: str, payload: Dict[str, Any], nonce: str) -> Optional[Dict[str, Any]]:
    """Capture the portable, self-verifying credential from an ATTESTED announce
    so it can be re-emitted via PEX. Returns None when the announce carried no
    attestation (a direct/legacy peer we can't cryptographically vouch for)."""
    pubkey = payload.get("origin_pubkey")
    sig = payload.get("origin_sig")
    if not (pubkey and sig):
        return None
    signed_payload = {k: v for k, v in payload.items() if k not in _ANNOUNCE_ATTEST_KEYS}
    return {
        "node_id": node_id,
        "signed_payload": signed_payload,
        "nonce": nonce,
        "origin_pubkey": pubkey,
        "origin_sig": sig,
    }


def _verify_peer_credential(cred: Any) -> Optional[str]:
    """Verify a portable peer credential relayed via PEX. Returns the
    authenticated node_id, or None to DROP. Mirrors
    _authenticated_announce_node_id but for a self-contained relayed record
    (there is no connection-peer fallback — a PEX entry is never the relaying
    peer's own identity)."""
    if not isinstance(cred, dict):
        return None
    node_id = cred.get("node_id") or ""
    pubkey = cred.get("origin_pubkey") or ""
    sig = cred.get("origin_sig") or ""
    signed_payload = cred.get("signed_payload")
    nonce = cred.get("nonce") or ""
    if not (node_id and pubkey and sig and isinstance(signed_payload, dict)):
        return None
    try:
        derived = hashlib.sha256(base64.b64decode(pubkey)).hexdigest()[:32]
    except Exception:
        return None
    if derived != node_id:
        return None
    try:
        ok = verify_signature(
            pubkey, _announce_signing_bytes(node_id, signed_payload, nonce), sig
        )
    except Exception:
        return None
    return node_id if ok else None


def _join_bootstrap_address(address: str, port: int) -> str:
    """sp1026 — combine a bootstrap-peer's stored address + port into a dial
    target WITHOUT double-appending the port when ``address`` already carries one.

    An operator's ``PRSM_ADVERTISE_ADDRESS`` may be ``host:port`` (the documented
    sp566 format); the bootstrap server stores it verbatim, so the prior naive
    ``f"{address}:{port}"`` produced ``host:port:port`` — an undialable address that
    broke cross-host discovery (surfaced live at the Tier-1 bench). Handles bare
    IPv4/hostname and bracketed IPv6; bare unbracketed IPv6 is out of scope (the
    transport uses host:port / bracketed forms)."""
    if not address:
        return f"{address}:{port}"
    if address.startswith("["):  # bracketed IPv6: "[::1]" or "[::1]:9001"
        return address if "]:" in address else f"{address}:{port}"
    _host, sep, tail = address.rpartition(":")
    if sep and tail.isdigit():  # already "host:port"
        return address
    return f"{address}:{port}"


class PeerDiscovery:
    """Discovers and maintains connections to network peers.

    Strategy:
    1. Connect to bootstrap nodes on startup.
    2. Request their peer lists.
    3. Periodically announce ourselves via gossip.
    4. Maintain a target number of connections.
    """

    # Sprint 1085 — the WebSocket discovery authenticates a peer's node_id
    # (_authenticated_announce_node_id: sha256(pubkey)==node_id + ed25519 signature, or
    # a handshake-authenticated peer_id; unauthenticated announces are dropped). So the
    # DHT pool provider may honor an attestation-derived hardware tier for these peers.
    node_id_authenticated: bool = True

    def __init__(
        self,
        transport: WebSocketTransport,
        bootstrap_nodes: Optional[List[str]] = None,
        bootstrap_connect_timeout: float = 5.0,
        bootstrap_retry_attempts: int = 2,
        bootstrap_fallback_enabled: bool = True,
        bootstrap_fallback_nodes: Optional[List[str]] = None,
        bootstrap_validate_addresses: bool = True,
        bootstrap_backoff_base: float = 1.0,
        bootstrap_backoff_max: float = 8.0,
        target_peers: int = 8,
        announce_interval: float = 60.0,
        maintenance_interval: float = 30.0,
        peer_stale_timeout: float = 600.0,
        local_capabilities: Optional[List[str]] = None,
        local_backends: Optional[List[str]] = None,
        local_gpu_available: bool = False,
        local_hardware_profile: Optional[Dict[str, Any]] = None,
    ):
        # Default bootstrap node — the live PRSM bootstrap server.
        # Sprint 575 F29 — bootstrap1 → bootstrap-us DNS rename
        # (2026-05-19); old hostname no longer resolves.
        _DEFAULT_BOOTSTRAP = ["wss://bootstrap-us.prsm-network.com:8765"]

        self.transport = transport
        self.bootstrap_nodes = bootstrap_nodes if bootstrap_nodes is not None else _DEFAULT_BOOTSTRAP
        self.bootstrap_connect_timeout = max(1.0, float(bootstrap_connect_timeout))
        self.bootstrap_retry_attempts = max(1, int(bootstrap_retry_attempts))
        self.bootstrap_fallback_enabled = bootstrap_fallback_enabled
        self.bootstrap_fallback_nodes = bootstrap_fallback_nodes or []
        self.bootstrap_validate_addresses = bootstrap_validate_addresses
        self.bootstrap_backoff_base = max(0.1, float(bootstrap_backoff_base))
        self.bootstrap_backoff_max = max(self.bootstrap_backoff_base, float(bootstrap_backoff_max))
        self.target_peers = target_peers
        self.announce_interval = announce_interval
        self.maintenance_interval = maintenance_interval
        self.peer_stale_timeout = peer_stale_timeout
        self._local_capabilities = local_capabilities or []
        self._local_backends = local_backends or []
        self._local_gpu_available = local_gpu_available
        # Sprint 680 — local hardware profile, advertised via
        # DISCOVERY_ANNOUNCE when set. None → key omitted from
        # payload (legacy wire format preserved).
        self._local_hardware_profile: Optional[Dict[str, Any]] = (
            local_hardware_profile
        )

        # Startup bootstrap status (for first-run observability)
        self.bootstrap_degraded_mode: bool = False
        self.bootstrap_connected_count: int = 0
        self.bootstrap_attempted_nodes: List[str] = []
        self.bootstrap_success_node: Optional[str] = None
        self.bootstrap_failed_nodes: List[str] = []
        # sp1024 — the server-observed advertise address (from the sp1023
        # register_ack). Used as the own-advertise FALLBACK when
        # PRSM_ADVERTISE_ADDRESS is unset, so the F14 co-location rewrite + the
        # gossip announce get a routable value with zero manual config.
        self._observed_advertise: Optional[str] = None
        # Sprint 653 — separate tracking for the BootstrapClient WS
        # protocol probes (sprint 568+ fallback path). Pre-653 only
        # the P2P-handshake probe failures were tracked; bootstrap
        # servers don't speak P2P so every probe ended up in
        # bootstrap_failed_nodes regardless of whether the WS
        # protocol probe later succeeded. Operators reading
        # /bootstrap/status concluded bootstrap-eu/apac were down
        # when they were fully operational (F26).
        self.bootstrap_client_attempted_nodes: List[str] = []
        self.bootstrap_client_failed_nodes: List[str] = []

        # Bootstrap decision telemetry (additive, never alters behavior)
        self._bootstrap_telemetry: Dict[str, Any] = {
            "addresses_validated": 0,
            "addresses_rejected": 0,
            "rejected_reasons": collections.Counter(),
            "fallback_activated": False,
            "fallback_attempted": 0,
            "fallback_succeeded": False,
            "backoff_total_seconds": 0.0,
            "source_policy": "primary_only",
        }

        # Known peers (may not be connected)
        self.known_peers: Dict[str, PeerInfo] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Register message handlers
        self.transport.on_message(MSG_GOSSIP, self._handle_gossip)

    async def start(self) -> None:
        """Start discovery: bootstrap then run maintenance loops."""
        self._running = True
        self._bootstrap_client = None
        await self.bootstrap()

        # sp1398 — ALWAYS run the bootstrap-client protocol (register + get_peers + auto-dial), not
        # only when the P2P transport bootstrap FAILED. bootstrap() above merely opens a transport
        # connection to the signaling server; it does NOT register this node for discovery, pull the
        # server's peer list, or dial peer compute nodes. Gating discovery behind degraded_mode meant
        # two nodes that BOTH reached the bootstrap never learned about each other — operators had to
        # POST /peers/connect by hand after every start (2026-07-07 live: us + sfo, 0 auto-peers).
        if self.bootstrap_nodes:
            connected = await self._try_bootstrap_client()
            if connected and self.bootstrap_degraded_mode:
                self.bootstrap_degraded_mode = False
                self.bootstrap_connected_count = 1

        self._tasks.append(asyncio.create_task(self._announce_loop()))
        self._tasks.append(asyncio.create_task(self._maintenance_loop()))
        if self.bootstrap_degraded_mode:
            logger.warning(
                "Discovery started in DEGRADED local mode: bootstrap unavailable; "
                "peer discovery may be delayed until inbound peers or local announcements arrive"
            )
        else:
            logger.info(f"Discovery started with {len(self.bootstrap_nodes)} bootstrap node(s)")

    async def _try_bootstrap_client(self) -> bool:
        """Fall back to the bootstrap client protocol when P2P handshake fails.

        The bootstrap server speaks a simpler register/heartbeat protocol
        (not the P2P MSG_HANDSHAKE). This method uses BootstrapClient to
        register with the server and discover peers.
        """
        try:
            from prsm.bootstrap.client import BootstrapClient, BootstrapPeer
        except ImportError:
            logger.debug("Bootstrap client not available")
            return False

        for address in self.bootstrap_nodes:
            # Sprint 653 — track WS-protocol probe attempts alongside
            # the P2P attempts so /bootstrap/status can distinguish
            # "P2P-handshake failed" (expected; bootstrap servers
            # don't speak P2P) from "totally unreachable".
            self.bootstrap_client_attempted_nodes.append(address)
            try:
                logger.info(
                    "Trying bootstrap client protocol for %s", address
                )
                # Sprint 150 — version must match runtime package
                # version, not a stale literal. Pre-fix this was
                # hardcoded "0.24.0" even after shipping v1.x.
                import prsm as _prsm_pkg
                # Sprint 566: PRSM_ADVERTISE_ADDRESS lets co-located
                # operators bootstrap via loopback but still advertise
                # their external IP to remote peers.
                from prsm.node.libp2p_discovery import (
                    _resolve_advertise_address,
                )
                client = BootstrapClient(
                    bootstrap_url=address,
                    node_id=self.transport.identity.node_id,
                    port=getattr(self.transport, 'port', 8000),
                    capabilities=self._local_capabilities,
                    version=_prsm_pkg.__version__,
                    connect_timeout=self.bootstrap_connect_timeout,
                    advertise_address=_resolve_advertise_address(),
                    # Sprint 838 — advertise local hw_profile to
                    # bootstrap-server for relay (closes cold-start
                    # gossip gap; sp682 pool reads this on the
                    # receiving side).
                    hardware_profile=self._local_hardware_profile,
                )

                peers = await client.connect()
                await client.start_heartbeat()
                # Sprint 632 — periodic peer-list refresh closes the
                # race where bootstrap-server peer state grows after
                # this client's register-time call. Without refresh,
                # the client never re-fetches and operators have to
                # restart daemons in specific order to coax symmetric
                # discovery (sprint 630 live evidence). Interval
                # defaults to 2× heartbeat so refresh adds ~one
                # extra request per minute per client — cheap.
                _refresh_interval = float(
                    os.environ.get(
                        "PRSM_BOOTSTRAP_PEER_REFRESH_INTERVAL",
                        str(client.heartbeat_interval * 2),
                    )
                )
                await client.start_peer_refresh(
                    interval=_refresh_interval,
                )

                self._bootstrap_client = client
                # sp1024 — capture how the server saw us as the own-advertise
                # fallback (the env var still wins in _own_advertise()).
                if getattr(client, "observed_address", None):
                    self._observed_advertise = client.observed_address
                self.bootstrap_success_node = address
                # Sprint 653 — F26 fix: the address is reachable via
                # the WS protocol even if it failed P2P-handshake.
                # Remove it from the operator-facing failed_nodes
                # list so /bootstrap/status doesn't lie about
                # reachability. (bootstrap_failed_nodes still
                # accurately records the P2P-probe failure history
                # internally; this remove is purely the operator-UX
                # correction.)
                if address in self.bootstrap_failed_nodes:
                    self.bootstrap_failed_nodes.remove(address)

                # Feed discovered peers into known_peers (sp1009 — capped).
                self._ingest_bootstrap_peers(peers)

                logger.info(
                    "Bootstrap client connected to %s — "
                    "registered, heartbeat active, %d peer(s) discovered",
                    address, len(peers),
                )

                # Sprint 573 — auto-dial sweep: turn freshly-discovered
                # known peers into connected peers without waiting for
                # an operator to POST /peers/connect for each one. Best-
                # effort; failures logged but don't break the bootstrap
                # success path.
                try:
                    await self._auto_dial_sweep()
                except Exception as _sweep_exc:  # noqa: BLE001
                    logger.warning(
                        "auto-dial sweep raised: %s "
                        "(bootstrap registration still succeeded)",
                        _sweep_exc,
                    )

                return True

            except Exception as e:
                logger.debug(
                    "Bootstrap client failed for %s: %s", address, e
                )
                # Sprint 653 — record genuine WS-protocol failures
                # alongside the P2P-protocol failures.
                self.bootstrap_client_failed_nodes.append(address)
                continue

        return False

    async def stop(self) -> None:
        self._running = False

        # Disconnect bootstrap client if active
        if getattr(self, '_bootstrap_client', None):
            try:
                await self._bootstrap_client.disconnect()
            except Exception:
                pass
            self._bootstrap_client = None

        for task in self._tasks:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

    def _build_bootstrap_candidate_list(self) -> Tuple[List[str], List[str]]:
        """Build ordered candidate list: primary nodes first, then fallback.

        Returns:
            (valid_candidates, rejected_addresses) where rejected_addresses
            contains any addresses that failed validation.
        """
        candidates: List[str] = []
        rejected: List[str] = []

        # Phase 1: configured primary nodes
        for addr in self.bootstrap_nodes:
            if self.bootstrap_validate_addresses:
                ok, reason = validate_bootstrap_address(addr)
                self._bootstrap_telemetry["addresses_validated"] += 1
                if not ok:
                    self._bootstrap_telemetry["addresses_rejected"] += 1
                    self._bootstrap_telemetry["rejected_reasons"][reason] += 1
                    rejected.append(addr)
                    logger.warning(
                        "Bootstrap address rejected (validation): %s — %s",
                        addr, reason,
                    )
                    continue
            candidates.append(addr)

        # Phase 2: fallback nodes (only if feature flag enabled)
        if self.bootstrap_fallback_enabled and self.bootstrap_fallback_nodes:
            self._bootstrap_telemetry["source_policy"] = "primary_then_fallback"
            for addr in self.bootstrap_fallback_nodes:
                if addr in candidates:
                    continue  # Already in primary list, skip duplicate
                if self.bootstrap_validate_addresses:
                    ok, reason = validate_bootstrap_address(addr)
                    self._bootstrap_telemetry["addresses_validated"] += 1
                    if not ok:
                        self._bootstrap_telemetry["addresses_rejected"] += 1
                        self._bootstrap_telemetry["rejected_reasons"][reason] += 1
                        rejected.append(addr)
                        logger.warning(
                            "Fallback bootstrap address rejected (validation): %s — %s",
                            addr, reason,
                        )
                        continue
                candidates.append(addr)

        return candidates, rejected

    async def bootstrap(self) -> int:
        """Connect to bootstrap nodes and request their peer lists.

        Uses source ordering policy: configured nodes first, then trusted
        fallback nodes (when bootstrap_fallback_enabled is True).
        Applies address validation and exponential backoff between retries.
        """
        connected = 0
        self.bootstrap_connected_count = 0
        self.bootstrap_degraded_mode = False
        self.bootstrap_attempted_nodes = []
        self.bootstrap_success_node = None
        self.bootstrap_failed_nodes = []

        # Reset telemetry for this bootstrap attempt
        self._bootstrap_telemetry = {
            "addresses_validated": 0,
            "addresses_rejected": 0,
            "rejected_reasons": collections.Counter(),
            "fallback_activated": False,
            "fallback_attempted": 0,
            "fallback_succeeded": False,
            "backoff_total_seconds": 0.0,
            "source_policy": "primary_only",
        }

        candidates, rejected = self._build_bootstrap_candidate_list()
        primary_count = len([a for a in self.bootstrap_nodes if a in candidates])

        for idx, address in enumerate(candidates):
            is_fallback = idx >= primary_count
            if is_fallback and not self._bootstrap_telemetry["fallback_activated"]:
                self._bootstrap_telemetry["fallback_activated"] = True
                logger.info(
                    "Primary bootstrap nodes exhausted; activating fallback peers"
                )
            if is_fallback:
                self._bootstrap_telemetry["fallback_attempted"] += 1

            self.bootstrap_attempted_nodes.append(address)

            for attempt in range(1, self.bootstrap_retry_attempts + 1):
                try:
                    peer = await asyncio.wait_for(
                        self.transport.connect_to_peer(address),
                        timeout=self.bootstrap_connect_timeout,
                    )
                except asyncio.TimeoutError:
                    peer = None
                    logger.debug(
                        "Bootstrap timeout for %s (attempt %d/%d)",
                        address,
                        attempt,
                        self.bootstrap_retry_attempts,
                    )

                if peer:
                    connected = 1
                    self.bootstrap_success_node = address
                    if is_fallback:
                        self._bootstrap_telemetry["fallback_succeeded"] = True

                    # Request their peer list
                    req = P2PMessage(
                        msg_type=MSG_GOSSIP,
                        sender_id=self.transport.identity.node_id,
                        payload={
                            "subtype": DISCOVERY_PEER_REQUEST,
                            "max_peers": 20,
                        },
                    )
                    await self.transport.send_to_peer(peer.peer_id, req)
                    break

                # Exponential backoff between retries (not after last attempt)
                if attempt < self.bootstrap_retry_attempts:
                    backoff = min(
                        self.bootstrap_backoff_base * (2 ** (attempt - 1)),
                        self.bootstrap_backoff_max,
                    )
                    self._bootstrap_telemetry["backoff_total_seconds"] += backoff
                    await asyncio.sleep(backoff)

            if connected:
                break

            self.bootstrap_failed_nodes.append(address)

        self.bootstrap_connected_count = connected

        if connected:
            logger.info(
                "Bootstrap success via %s (attempted %d/%d candidates)%s",
                self.bootstrap_success_node,
                len(self.bootstrap_attempted_nodes),
                len(candidates),
                " [fallback]" if self._bootstrap_telemetry.get("fallback_succeeded") else "",
            )
        elif candidates:
            self.bootstrap_degraded_mode = True
            logger.warning(
                "Bootstrap unavailable after %d candidate(s), %d attempt(s) each; "
                "continuing in DEGRADED local mode",
                len(candidates),
                self.bootstrap_retry_attempts,
            )
        else:
            if rejected:
                self.bootstrap_degraded_mode = True
                logger.warning(
                    "All %d bootstrap address(es) rejected as malformed; "
                    "continuing in DEGRADED local mode",
                    len(rejected),
                )
            else:
                logger.info("No bootstrap nodes configured — this node is the first on the network")

        return connected

    def get_bootstrap_status(self) -> Dict[str, object]:
        """Return startup bootstrap state for node/CLI status reporting."""
        return {
            "configured_nodes": list(self.bootstrap_nodes),
            "attempted_nodes": list(self.bootstrap_attempted_nodes),
            "failed_nodes": list(self.bootstrap_failed_nodes),
            "success_node": self.bootstrap_success_node,
            "connected_count": self.bootstrap_connected_count,
            "degraded_mode": self.bootstrap_degraded_mode,
            "retry_attempts": self.bootstrap_retry_attempts,
            "connect_timeout_seconds": self.bootstrap_connect_timeout,
            "fallback_enabled": self.bootstrap_fallback_enabled,
            "fallback_activated": self._bootstrap_telemetry.get("fallback_activated", False),
            "fallback_succeeded": self._bootstrap_telemetry.get("fallback_succeeded", False),
            "addresses_rejected": self._bootstrap_telemetry.get("addresses_rejected", 0),
            "source_policy": self._bootstrap_telemetry.get("source_policy", "primary_only"),
            "bootstrap_client_active": (
                getattr(self, '_bootstrap_client', None) is not None
                and getattr(self._bootstrap_client, 'is_connected', False)
            ),
            # Sprint 653 — F26 fix: separate visibility for the WS-
            # protocol probe history so operators can tell
            # "P2P-probe failed (expected; bootstrap servers don't
            # speak P2P)" from "node genuinely unreachable".
            "bootstrap_client_attempted_nodes": list(
                self.bootstrap_client_attempted_nodes,
            ),
            "bootstrap_client_failed_nodes": list(
                self.bootstrap_client_failed_nodes,
            ),
        }

    def get_bootstrap_telemetry(self) -> Dict[str, Any]:
        """Return a stable copy of bootstrap decision telemetry for observability.

        This data is purely additive and never alters bootstrap behavior.
        """
        return {
            "addresses_validated": int(self._bootstrap_telemetry.get("addresses_validated", 0)),
            "addresses_rejected": int(self._bootstrap_telemetry.get("addresses_rejected", 0)),
            "rejected_reasons": dict(self._bootstrap_telemetry.get("rejected_reasons", {})),
            "fallback_activated": bool(self._bootstrap_telemetry.get("fallback_activated", False)),
            "fallback_attempted": int(self._bootstrap_telemetry.get("fallback_attempted", 0)),
            "fallback_succeeded": bool(self._bootstrap_telemetry.get("fallback_succeeded", False)),
            "backoff_total_seconds": float(self._bootstrap_telemetry.get("backoff_total_seconds", 0.0)),
            "source_policy": str(self._bootstrap_telemetry.get("source_policy", "primary_only")),
        }

    def _own_advertise(self) -> "str | None":
        """The node's own externally-visible advertise value, the single source
        for the gossip announce + the F14 co-location dial rewrite. Precedence:
        PRSM_ADVERTISE_ADDRESS (sp566) → the bootstrap-server-observed address
        (sp1023/sp1024) → None. May carry a ':port' suffix; the co-location
        rewrite strips it, and the announce builder takes the host part only."""
        from prsm.node.libp2p_discovery import _resolve_advertise_address
        return _resolve_advertise_address() or self._observed_advertise

    async def announce_self(self) -> int:
        """Broadcast our presence to the network."""
        # Sprint 756 — operator-controlled active-window scheduling.
        # If PRSM_ACTIVE_HOURS is set and we're currently OUTSIDE the
        # window, skip the announce. Existing peers' known-peer
        # caches will expire this node after peer_stale_timeout
        # (default 60s), cleanly removing us from the routing pool
        # for the duration. When the active window resumes, normal
        # announces re-add us to peers' caches. Daemon stays running
        # — operators can still query /status (loopback-only post-
        # sprint 748), claim earnings, etc.
        from prsm.node.schedule import is_currently_active
        if not is_currently_active():
            return 0
        # Sprint 773 — preemption gate. AWS/GCP spot operators that
        # have been signaled for preemption stop announcing so peers
        # evict from the routing pool inside the ~2min warning
        # window. AND semantics with the active-window gate above.
        from prsm.node.preemption import is_currently_preempted
        if is_currently_preempted():
            return 0
        # Sprint 570 F28: only include `address` when we have a real
        # externally-reachable value. transport.host is typically
        # "0.0.0.0" (bind-to-all) which is unreachable when gossiped;
        # recipients would overwrite the correct bootstrap-server-
        # supplied IP with 0.0.0.0:port. Prefer PRSM_ADVERTISE_ADDRESS
        # (sprint-566 env var); otherwise omit so _handle_announce's
        # fallback to peer.address (the WS source-connection IP) kicks
        # in — which IS routable for any inbound connection.
        advertise = self._own_advertise()
        payload = {
            "subtype": DISCOVERY_ANNOUNCE,
            "display_name": getattr(self.transport.identity, "display_name", ""),
            "roles": [],
            "capabilities": self._local_capabilities,
            "supported_backends": self._local_backends,
            "gpu_available": self._local_gpu_available,
            "peer_count": self.transport.peer_count,
        }
        if advertise:
            # Host-only + our local listen port — observed_address may already
            # carry a ':port' suffix, so strip it to avoid "ip:port:port".
            payload["address"] = f"{advertise.split(':', 1)[0]}:{self.transport.port}"
        # Sprint 680 — include hardware_profile only when locally
        # configured. Absent key preserves the pre-680 wire format
        # for peers that don't parse this field yet.
        if self._local_hardware_profile is not None:
            payload["hardware_profile"] = self._local_hardware_profile
        # sp941 — signed freshness stamp (monotonic replay defense). Set BEFORE
        # attesting so it's covered by the signature.
        payload["announce_time"] = time.time()
        # sp937 — sign the announce so receivers can authenticate our node_id
        # (and reject forgeries) across multi-hop relay.
        nonce = uuid.uuid4().hex[:16]
        _attest_announce_payload(self.transport.identity, payload, nonce)
        msg = P2PMessage(
            msg_type=MSG_GOSSIP,
            sender_id=self.transport.identity.node_id,
            payload=payload,
            nonce=nonce,
        )
        return await self.transport.gossip(msg, fanout=3)

    async def _auto_dial_sweep(self) -> None:
        """Sprint 573 — turn known_peers into connected peers.

        After bootstrap hydration (or any path that mass-populates
        ``self.known_peers``), iterate the registry and dial each
        peer that:
          - is not self
          - is not already in ``self.transport.peers``
          - has a non-bogus address (skip 0.0.0.0:* / empty per
            sprint-570 F28 defense-in-depth)

        Best-effort: each dial runs in its own try/except so one
        failed connection (NAT'd peer, stale registration, etc.)
        doesn't abort the rest of the sweep.

        Closes sprint-567 gap 2 + the operator ergonomic gap
        sprint-569 surfaced: post-bootstrap, ``known_count`` would
        be non-zero but ``connected_count`` stayed 0 until an
        operator manually called ``POST /peers/connect`` for every
        known peer. After sprint 573 the daemon does it itself.
        """
        own_id = self.transport.identity.node_id
        connected_ids = set(self.transport.peers.keys())
        for info in list(self.known_peers.values()):
            if info.node_id == own_id:
                continue
            if info.node_id in connected_ids:
                continue
            addr = info.address or ""
            if (
                not addr
                or addr.startswith("0.0.0.0:")
                or addr == "0.0.0.0"
                or addr.startswith(":")
            ):
                logger.debug(
                    "auto-dial sweep skipping %s — bogus address %r",
                    info.node_id[:8], addr,
                )
                continue
            # Sprint 781 — F14 fix: rewrite to loopback when the
            # peer's announced host matches our own (co-located
            # daemons can't NAT-hairpin to their shared external IP).
            dial_addr = _rewrite_co_located_address(
                addr, self._own_advertise(),
            )
            try:
                peer = await self.transport.connect_to_peer(dial_addr)
                if peer is None:
                    logger.debug(
                        "auto-dial sweep: connect_to_peer returned "
                        "None for %s (%s)",
                        info.node_id[:8], dial_addr,
                    )
                else:
                    logger.info(
                        "auto-dial sweep: connected to %s (%s)",
                        info.node_id[:8], dial_addr,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "auto-dial sweep: dial to %s (%s) raised %s: %s",
                    info.node_id[:8], addr, type(exc).__name__, exc,
                )

    async def maintain_connections(self) -> None:
        """Ensure we have enough peer connections, connecting to known peers if needed."""
        current = self.transport.peer_count
        if current >= self.target_peers:
            return

        # Try connecting to known but unconnected peers
        connected_ids = set(self.transport.peers.keys())
        candidates = [
            p for p in self.known_peers.values()
            if p.node_id not in connected_ids
            and p.node_id != self.transport.identity.node_id
        ]
        random.shuffle(candidates)

        needed = self.target_peers - current
        # Sprint 781 — F14 fix: same loopback-rewrite as
        # _auto_dial_sweep so periodic maintain doesn't keep
        # failing on co-located peers.
        own_advertise = self._own_advertise()
        for info in candidates[:needed]:
            dial_addr = _rewrite_co_located_address(
                info.address, own_advertise,
            )
            peer = await self.transport.connect_to_peer(dial_addr)
            if peer:
                logger.debug(f"Reconnected to known peer {info.node_id[:8]}...")

    def get_known_peers(self) -> List[PeerInfo]:
        """Return list of all known peers (connected or not)."""
        return list(self.known_peers.values())

    def find_peers_by_capability(
        self,
        required: List[str],
        match_all: bool = True,
    ) -> List[PeerInfo]:
        """Find peers that offer the required capabilities.

        Args:
            required: Capability strings to search for.
            match_all: If True, peer must have *all* required capabilities.
                       If False, peer must have *any* of them.

        Returns:
            Matching peers sorted by most-recently-seen first.
        """
        required_lower = {c.lower() for c in required}
        results: List[PeerInfo] = []
        for peer in self.known_peers.values():
            peer_caps = {c.lower() for c in peer.capabilities}
            if match_all:
                if required_lower <= peer_caps:
                    results.append(peer)
            else:
                if required_lower & peer_caps:
                    results.append(peer)
        results.sort(key=lambda p: p.last_seen, reverse=True)
        return results

    def find_peers_with_capability(self, capability: str) -> List[PeerInfo]:
        """Find peers that have a specific capability.

        Args:
            capability: The capability to search for (e.g., "inference", "embedding").

        Returns:
            List of peers with the specified capability, sorted by most-recently-seen.
        """
        capability_lower = capability.lower()
        results = [
            peer for peer in self.known_peers.values()
            if capability_lower in {c.lower() for c in peer.capabilities}
        ]
        results.sort(key=lambda p: p.last_seen, reverse=True)
        return results

    def find_peers_with_backend(self, backend: str) -> List[PeerInfo]:
        """Find peers that support a specific backend.

        Args:
            backend: The backend to search for (e.g., "anthropic", "openai", "local").

        Returns:
            List of peers with the specified backend support, sorted by most-recently-seen.
        """
        backend_lower = backend.lower()
        results = [
            peer for peer in self.known_peers.values()
            if backend_lower in {b.lower() for b in peer.supported_backends}
        ]
        results.sort(key=lambda p: p.last_seen, reverse=True)
        return results

    def find_peers_with_gpu(self) -> List[PeerInfo]:
        """Find peers that have GPU available.

        Returns:
            List of peers with GPU available, sorted by most-recently-seen.
        """
        results = [peer for peer in self.known_peers.values() if peer.gpu_available]
        results.sort(key=lambda p: p.last_seen, reverse=True)
        return results

    # ── Message handlers ─────────────────────────────────────────

    def _ingest_bootstrap_peers(self, peers) -> int:
        """sp1009 — feed bootstrap-discovered peers into known_peers, bounded by
        the same PRSM_MAX_KNOWN_PEERS cap that guards the announce + PEX paths.
        The bootstrap is TLS-verified (sp1006) but still semi-trusted, so a
        COMPROMISED bootstrap must not be able to flood known_peers without
        bound (memory DoS on cold start). An already-known peer always refreshes;
        a new id beyond the cap is dropped. sp838 relayed hardware_profile is
        preserved. Returns the number ingested."""
        ingested = 0
        for bp in peers:
            pid = getattr(bp, "peer_id", None)
            if not pid or pid == self.transport.identity.node_id:
                continue
            if pid not in self.known_peers and len(self.known_peers) >= _max_known_peers():
                continue
            bp_hw = getattr(bp, "hardware_profile", None)
            self.known_peers[pid] = PeerInfo(
                node_id=pid,
                # sp1026 — do NOT double-append the port: bp.address may already
                # be "host:port" (the operator's PRSM_ADVERTISE_ADDRESS).
                address=_join_bootstrap_address(bp.address, bp.port),
                capabilities=getattr(bp, "capabilities", []),
                hardware_profile=bp_hw if isinstance(bp_hw, dict) else None,
            )
            ingested += 1
        return ingested

    def record_job_success(self, node_id: str) -> None:
        """Sprint 1402 — record a successful job completion for a peer (reliability tracking).

        MUST exist on PeerDiscovery, not just Libp2pDiscovery: compute_requester._on_job_result
        calls self.discovery.record_job_success(provider) AFTER marking the job complete but BEFORE
        releasing payment. On the WebSocket transport (PeerDiscovery) the missing method raised
        AttributeError, killing the handler before the provider was paid — cross-node jobs settled to
        NO ONE (live 2026-07-07). PeerInfo already carries the counters."""
        peer = self.known_peers.get(node_id)
        if peer:
            peer.job_success_count += 1

    def record_job_failure(self, node_id: str) -> None:
        """Sprint 1402 — record a job failure/timeout for a peer (reliability tracking)."""
        peer = self.known_peers.get(node_id)
        if peer:
            peer.job_failure_count += 1
            peer.last_failure_time = time.time()

    async def _handle_gossip(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Handle discovery-related gossip messages."""
        subtype = msg.payload.get("subtype", "")

        if subtype == DISCOVERY_ANNOUNCE:
            await self._handle_announce(msg, peer)
        elif subtype == DISCOVERY_PEER_REQUEST:
            await self._handle_peer_request(msg, peer)
        elif subtype == DISCOVERY_PEER_RESPONSE:
            await self._handle_peer_response(msg, peer)
        elif subtype == DISCOVERY_CAPABILITY_ANNOUNCE:
            await self._handle_capability_announce(msg, peer)

    async def _handle_announce(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Record a peer announcement."""
        # sp937 — authenticate the announcer's node_id. A forged/unauthenticated
        # gossip sender_id is dropped so it can't poison the routing table.
        node_id = _authenticated_announce_node_id(msg, peer)
        if node_id is None:
            return
        # sp941 — reject a replayed (stale-timestamp) announce so it can't
        # re-assert an old address/capabilities after the dedup window.
        _prev = self.known_peers.get(node_id)
        if _prev is not None and _announce_is_stale_replay(msg.payload, _prev.last_announce_time):
            return
        # sp1005 — bound the routing table (memory + eclipse-magnitude defense).
        # Only NEW node_ids are subject to the cap; an already-tracked peer is
        # always allowed to refresh.
        if _prev is None and len(self.known_peers) >= _max_known_peers():
            logger.debug(
                "known_peers at cap (%d) — dropping new announce from %s",
                _max_known_peers(), node_id[:8],
            )
            return
        # Sprint 570 F28 defense-in-depth: ignore 0.0.0.0:* from
        # legacy pre-sprint-570 peers. The bind-to-all listen host
        # is not a routable advertise value — falling back to
        # peer.address (WS source-connection IP) gives us a real
        # reachable address for inbound connections.
        raw_address = msg.payload.get("address", "")
        if (
            not raw_address
            or raw_address == "0.0.0.0"
            or raw_address.startswith("0.0.0.0:")
        ):
            address = peer.address
        else:
            address = raw_address
        self.known_peers[node_id] = PeerInfo(
            node_id=node_id,
            address=address,
            display_name=msg.payload.get("display_name", ""),
            roles=msg.payload.get("roles", []),
            capabilities=msg.payload.get("capabilities", []),
            supported_backends=msg.payload.get("supported_backends", []),
            gpu_available=msg.payload.get("gpu_available", False),
            last_seen=time.time(),
            last_capability_update=time.time(),
            # Sprint 680 — optional. Pre-680 announcements omit the
            # key entirely; .get() returns None, matching the
            # dataclass default.
            hardware_profile=msg.payload.get("hardware_profile"),
            # sp941 — record the accepted announce_time for the next replay check.
            last_announce_time=float(msg.payload.get("announce_time") or 0.0),
            # sp1005 — capture the portable credential (when attested) so this
            # peer can be relayed via authenticated PEX.
            announce_credential=_build_announce_credential(node_id, msg.payload, msg.nonce),
        )
        # Re-gossip if TTL > 0
        if msg.ttl > 1:
            fwd = P2PMessage(
                msg_type=msg.msg_type,
                sender_id=msg.sender_id,
                payload=msg.payload,
                ttl=msg.ttl - 1,
                nonce=msg.nonce,  # same nonce so others dedup
            )
            await self.transport.gossip(fwd, fanout=2)

    async def _handle_peer_request(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Respond with our known peer list.

        sp1005 — every emitted entry carries the peer's own portable,
        self-verifying credential (captured from its authenticated announce),
        and the entry's identity/address/profile live INSIDE that signed
        credential, so a relayer cannot fabricate or tamper an entry. Peers we
        have no credential for (direct/legacy, never saw an attested announce)
        are NOT relayed — the receiver would drop them anyway, and relaying
        them unsigned would just reopen the eclipse vector.
        """
        # sp1005 — bound the response size; max_peers is attacker-controllable.
        try:
            max_peers = int(msg.payload.get("max_peers", 20))
        except (TypeError, ValueError):
            max_peers = 20
        max_peers = max(1, min(max_peers, 100))
        peers_data = []
        for info in self.known_peers.values():
            if len(peers_data) >= max_peers:
                break
            if info.announce_credential is None:
                continue
            peers_data.append({"credential": info.announce_credential})

        resp = P2PMessage(
            msg_type=MSG_GOSSIP,
            sender_id=self.transport.identity.node_id,
            payload={
                "subtype": DISCOVERY_PEER_RESPONSE,
                "peers": peers_data,
            },
        )
        await self.transport.send_to_peer(peer.peer_id, resp)

    async def _handle_peer_response(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Process a peer list (PEX) response.

        sp1005 — each entry MUST carry a portable, self-verifying credential
        (see _verify_peer_credential). Entries that fail verification — forged,
        tampered, or impersonating — are DROPPED, so a relaying peer can only
        forward records that the named peer itself legitimately signed (closing
        the eclipse + Parallax-pool-poisoning + memory-DoS vectors that the
        pre-fix unauthenticated handler opened). All ingested fields come from
        the VERIFIED signed payload, never the tamperable top-level entry.
        """
        peers_data = msg.payload.get("peers", [])
        ingested = 0
        for p in peers_data:
            cred = p.get("credential") if isinstance(p, dict) else None
            nid = _verify_peer_credential(cred)
            if nid is None or nid == self.transport.identity.node_id:
                continue
            signed = cred["signed_payload"]
            _prev = self.known_peers.get(nid)
            # sp941 — reject a replayed (stale-timestamp) credential so it can't
            # re-assert an old address/profile after the dedup window.
            if _prev is not None and _announce_is_stale_replay(signed, _prev.last_announce_time):
                continue
            # sp1005 — cap the routing table (memory + eclipse-magnitude). New
            # node_ids only; an already-tracked peer always refreshes.
            if _prev is None and len(self.known_peers) >= _max_known_peers():
                continue
            # Sprint 700 F46 — monotonic hardware_profile: a relayed entry may
            # ADD profile data but must not REMOVE it.
            incoming_hw = signed.get("hardware_profile")
            if incoming_hw is None and _prev is not None and _prev.hardware_profile is not None:
                incoming_hw = _prev.hardware_profile
            self.known_peers[nid] = PeerInfo(
                node_id=nid,
                address=signed.get("address", ""),
                display_name=signed.get("display_name", ""),
                roles=signed.get("roles", []),
                capabilities=signed.get("capabilities", []),
                supported_backends=signed.get("supported_backends", []),
                gpu_available=signed.get("gpu_available", False),
                last_seen=time.time(),
                last_capability_update=time.time(),
                hardware_profile=incoming_hw,
                last_announce_time=float(signed.get("announce_time") or 0.0),
                # Store the verified credential so this peer can be re-relayed.
                announce_credential=cred,
            )
            ingested += 1
        logger.debug(
            "PEX from %s: %d/%d entries authenticated + ingested",
            peer.peer_id[:8], ingested, len(peers_data),
        )

    async def _handle_capability_announce(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Handle capability announcement from a peer.

        Updates the peer's capability information in the known_peers dict.
        This allows late-joining nodes to receive capability updates.
        """
        # sp937 — authenticate the announcer's node_id (see _handle_announce).
        node_id = _authenticated_announce_node_id(msg, peer)
        if node_id is None:
            return
        # sp941 — reject a replayed (stale-timestamp) capability announce.
        _prev = self.known_peers.get(node_id)
        if _prev is not None and _announce_is_stale_replay(msg.payload, _prev.last_announce_time):
            return
        # sp1414 — bound the routing table on the capability-announce path too (memory +
        # eclipse-magnitude defense). sp1005 capped _handle_announce + _handle_peer_response,
        # but this handler's new-peer branch below inserted uncapped — so one authenticated peer
        # minting keypairs and flooding DISCOVERY_CAPABILITY_ANNOUNCE grew known_peers without
        # bound. Only NEW node_ids are gated; a tracked peer always refreshes (the branch below).
        if _prev is None and len(self.known_peers) >= _max_known_peers():
            logger.debug(
                "known_peers at cap (%d) — dropping new capability announce from %s",
                _max_known_peers(), node_id[:8],
            )
            return
        announce_time = float(msg.payload.get("announce_time") or 0.0)
        capabilities = msg.payload.get("capabilities", [])
        supported_backends = msg.payload.get("supported_backends", [])
        gpu_available = msg.payload.get("gpu_available", False)

        # Update existing peer info or create new entry
        if node_id in self.known_peers:
            existing = self.known_peers[node_id]
            existing.capabilities = capabilities
            existing.supported_backends = supported_backends
            existing.gpu_available = gpu_available
            existing.last_seen = time.time()
            existing.last_capability_update = time.time()
            existing.last_announce_time = announce_time
            logger.debug(
                f"Updated capabilities for peer {node_id[:8]}: "
                f"caps={capabilities}, backends={supported_backends}, gpu={gpu_available}"
            )
        else:
            # Create new peer entry with capability info
            self.known_peers[node_id] = PeerInfo(
                node_id=node_id,
                address=msg.payload.get("address", peer.address),
                display_name=msg.payload.get("display_name", ""),
                roles=msg.payload.get("roles", []),
                capabilities=capabilities,
                supported_backends=supported_backends,
                gpu_available=gpu_available,
                last_seen=time.time(),
                last_capability_update=time.time(),
                last_announce_time=announce_time,
            )
            logger.debug(
                f"Created new peer entry from capability announce: {node_id[:8]}"
            )

        # Re-gossip if TTL > 0
        if msg.ttl > 1:
            fwd = P2PMessage(
                msg_type=msg.msg_type,
                sender_id=msg.sender_id,
                payload=msg.payload,
                ttl=msg.ttl - 1,
                nonce=msg.nonce,
            )
            await self.transport.gossip(fwd, fanout=2)

    async def announce_capabilities(self) -> int:
        """Broadcast our capabilities to the network.

        This should be called on node startup and when capabilities change.
        Returns the number of peers the announcement was sent to.
        """
        cap_payload = {
            "subtype": DISCOVERY_CAPABILITY_ANNOUNCE,
            "node_id": self.transport.identity.node_id,
            "capabilities": self._local_capabilities,
            "supported_backends": self._local_backends,
            "gpu_available": self._local_gpu_available,
            "announce_time": time.time(),  # sp941 — signed freshness stamp (replay defense)
        }
        # sp937 — sign the capability announce (see announce_self).
        nonce = uuid.uuid4().hex[:16]
        _attest_announce_payload(self.transport.identity, cap_payload, nonce)
        msg = P2PMessage(
            msg_type=MSG_GOSSIP,
            sender_id=self.transport.identity.node_id,
            payload=cap_payload,
            nonce=nonce,
        )
        logger.info(
            f"Announcing capabilities: caps={self._local_capabilities}, "
            f"backends={self._local_backends}, gpu={self._local_gpu_available}"
        )
        return await self.transport.gossip(msg, fanout=3)

    def set_local_capabilities(
        self,
        capabilities: List[str],
        backends: List[str],
        gpu_available: bool = False,
    ) -> None:
        """Set the local node's capabilities.

        Args:
            capabilities: List of capabilities this node offers (e.g., ["inference", "embedding"]).
            backends: List of supported backends (e.g., ["anthropic", "openai", "local"]).
            gpu_available: Whether this node has GPU resources.
        """
        self._local_capabilities = capabilities
        self._local_backends = backends
        self._local_gpu_available = gpu_available
        logger.info(
            f"Set local capabilities: caps={capabilities}, backends={backends}, gpu={gpu_available}"
        )

    # ── Background loops ─────────────────────────────────────────

    async def _announce_loop(self) -> None:
        # Sprint 758 — graceful state-transition handling. Without
        # this, when an operator's active-window resumes at 22:00,
        # the node would wait up to `announce_interval` (default
        # 60s) before broadcasting again. Operators see "I set
        # my schedule to start at 22:00 but the node didn't appear
        # in the pool until 22:00:45". Fix: poll on a tighter
        # cadence + detect inactive→active transition + force an
        # immediate announce.
        #
        # Poll interval: min(announce_interval, 10s). Operators
        # with very long announce_interval still get a 10s-ish
        # detection latency; operators with short intervals see
        # no behavior change.
        from prsm.node.schedule import is_currently_active
        # Sprint 773 — preemption flag also short-circuits the loop.
        # Imported here (same lazy-import pattern as is_currently_active)
        # so test-time monkeypatches of the module-level helper land
        # in this scope on each iteration.
        from prsm.node.preemption import is_currently_preempted  # noqa: F401
        was_active = is_currently_active()
        poll_interval = min(self.announce_interval, 10.0)
        elapsed_since_announce = 0.0
        while self._running:
            await asyncio.sleep(poll_interval)
            elapsed_since_announce += poll_interval
            now_active = is_currently_active()
            transition_to_active = now_active and not was_active
            was_active = now_active
            # Three triggers for announce: (1) regular interval
            # elapsed, (2) transition from inactive → active
            # (immediate re-announce so peers re-add us), (3) we
            # ARE active (announce_self() skips internally if
            # somehow inactive — defensive idempotency).
            should_announce = (
                elapsed_since_announce >= self.announce_interval
                or transition_to_active
            )
            if not should_announce:
                continue
            if transition_to_active:
                logger.info(
                    "Sprint 758 — active window resumed; "
                    "forcing immediate announce."
                )
            try:
                await self.announce_self()
            except Exception as e:
                logger.error(f"Announce error: {e}")
            elapsed_since_announce = 0.0

    async def _maintenance_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.maintenance_interval)
            try:
                await self.maintain_connections()
                # Prune stale known peers
                cutoff = time.time() - self.peer_stale_timeout
                stale = [nid for nid, p in self.known_peers.items() if p.last_seen < cutoff]
                for nid in stale:
                    del self.known_peers[nid]
            except Exception as e:
                logger.error(f"Maintenance error: {e}")
