"""
Bootstrap Client

Connects a PRSM node to the bootstrap server for peer discovery.
The bootstrap server speaks a simple JSON protocol (register/heartbeat/get_peers),
while the node's P2P transport uses a different handshake protocol. This client
bridges the two by:

1. Registering the node with the bootstrap server via WSS
2. Receiving the peer list from the bootstrap server
3. Maintaining heartbeat to stay in the peer registry
4. Feeding discovered peer addresses back to the caller
"""

import asyncio
import json
import logging
import os
import ssl
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _build_bootstrap_ssl_context(bootstrap_url: str) -> Optional[ssl.SSLContext]:
    """sp1006 — SSL context for a bootstrap WSS connection.

    SECURE BY DEFAULT: full TLS verification (CERT_REQUIRED + hostname check)
    via ssl.create_default_context(). The pre-fix code disabled verification
    unconditionally (CERT_NONE), making the bootstrap connection MITM-able — an
    on-path attacker could impersonate the bootstrap and feed a joining node an
    all-attacker peer list (cold-start eclipse). The live fleet uses valid
    Let's Encrypt certs, so verifying by default is non-breaking.

    Dev / self-signed deployments opt in explicitly:
      - PRSM_BOOTSTRAP_TLS_CA_FILE=<pem>  pins a custom CA (still verifies).
      - PRSM_BOOTSTRAP_TLS_INSECURE=1     disables verification (loudly warned).

    Returns None for a non-wss URL (plain ws:// needs no TLS context).
    """
    if not bootstrap_url.startswith("wss://"):
        return None
    ctx = ssl.create_default_context()  # CERT_REQUIRED + check_hostname=True
    insecure = os.environ.get("PRSM_BOOTSTRAP_TLS_INSECURE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    ca_file = os.environ.get("PRSM_BOOTSTRAP_TLS_CA_FILE", "").strip()
    if insecure:
        logger.critical(
            "PRSM_BOOTSTRAP_TLS_INSECURE is set — bootstrap TLS verification is "
            "DISABLED. The bootstrap connection is MITM-able (cold-start eclipse "
            "risk); use this in dev only. Prefer PRSM_BOOTSTRAP_TLS_CA_FILE to pin "
            "a self-signed bootstrap's CA instead."
        )
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    elif ca_file:
        # Pin a custom / self-signed CA as the trust anchor — still verifies.
        ctx.load_verify_locations(cafile=ca_file)
    return ctx

try:
    import websockets
    import websockets.client
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False


@dataclass
class BootstrapPeer:
    """A peer discovered via the bootstrap server."""
    peer_id: str
    address: str
    port: int
    capabilities: List[str] = field(default_factory=list)
    region: Optional[str] = None
    version: Optional[str] = None
    # Sprint 838 — hw_profile relayed from the bootstrap. None
    # when the peer hasn't advertised one (pre-838 client OR
    # operator without local profile detection).
    hardware_profile: Optional[Dict[str, Any]] = None


class BootstrapClient:
    """Client that connects to a PRSM bootstrap server for peer discovery.

    Usage::

        client = BootstrapClient(
            bootstrap_url="wss://bootstrap1.prsm-network.com:8765",
            node_id="my-node-id",
            port=8000,
            capabilities=["compute", "storage"],
            version="0.24.0",
        )
        peers = await client.connect()
        # peers is a list of BootstrapPeer objects

        # Start background heartbeat (keeps registration alive)
        await client.start_heartbeat()

        # Later...
        await client.disconnect()
    """

    def __init__(
        self,
        bootstrap_url: str,
        node_id: str,
        port: int = 8000,
        capabilities: Optional[List[str]] = None,
        version: str = "0.3.2",
        region: Optional[str] = None,
        connect_timeout: float = 10.0,
        heartbeat_interval: int = 30,
        on_peers_discovered: Optional[Callable[[List[BootstrapPeer]], Any]] = None,
        advertise_address: Optional[str] = None,
        hardware_profile: Optional[Dict[str, Any]] = None,
    ):
        self.bootstrap_url = bootstrap_url
        self.node_id = node_id
        self.port = port
        self.capabilities = capabilities or []
        self.version = version
        self.region = region
        self.connect_timeout = connect_timeout
        self.heartbeat_interval = heartbeat_interval
        self.on_peers_discovered = on_peers_discovered
        # Sprint 838 — hardware_profile relay. When supplied, the
        # client advertises this profile in the registration
        # payload. Bootstrap caches + re-broadcasts to other
        # peers so cold-start joiners see real fleet capacity
        # without waiting on direct DISCOVERY_ANNOUNCE.
        self.hardware_profile = hardware_profile
        # Sprint 566: operator-supplied address the bootstrap-server
        # should record + advertise to other peers. When None, the
        # server falls back to the WS client_ip (pre-566 behavior).
        # Needed when the operator co-locates with a bootstrap-server
        # and bootstraps via loopback (sprint-460 invariant) — the
        # server would otherwise record `127.0.0.1` as the operator's
        # address, which is unreachable to remote peers.
        self.advertise_address = advertise_address

        self._ws: Optional[Any] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        # Sprint 632 — periodic peer-list refresh task. Closes the
        # sprint-630 race where bootstrap-server peer state grew
        # after register-time (server restart, late-joining peer)
        # and the client never re-fetched, requiring an operator
        # restart cascade to recover symmetric discovery.
        self._refresh_task: Optional[asyncio.Task] = None
        self._connected = False
        self._registered = False
        self._peers: List[BootstrapPeer] = []
        self._server_time: Optional[str] = None
        self._connect_time: Optional[float] = None
        # Sprint 319 — server-pushed announcements (peer_join /
        # peer_leave / system_notice / ...) get drained off the
        # socket during typed request/response cycles. Buffered
        # here for tests + future async-peer-update consumers.
        self._observed_announcements: List[Dict[str, Any]] = []

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None

    @property
    def is_registered(self) -> bool:
        return self._registered

    @property
    def discovered_peers(self) -> List[BootstrapPeer]:
        return list(self._peers)

    async def connect(self) -> List[BootstrapPeer]:
        """Connect to the bootstrap server and register this node.

        Returns:
            List of discovered peers from the bootstrap server.

        Raises:
            ImportError: If websockets is not installed.
            ConnectionError: If connection or registration fails.
        """
        if not _WS_AVAILABLE:
            raise ImportError(
                "websockets is required for bootstrap connectivity. "
                "Install with: pip install websockets"
            )

        # sp1006 — secure-by-default TLS (verify the bootstrap's cert). Dev /
        # self-signed deployments opt out via PRSM_BOOTSTRAP_TLS_CA_FILE (pin)
        # or PRSM_BOOTSTRAP_TLS_INSECURE=1 (disable, loudly warned).
        ssl_ctx = _build_bootstrap_ssl_context(self.bootstrap_url)

        try:
            logger.info(
                "Connecting to bootstrap server: %s", self.bootstrap_url
            )
            kwargs = {"open_timeout": self.connect_timeout}
            if ssl_ctx:
                kwargs["ssl"] = ssl_ctx

            self._ws = await websockets.client.connect(
                self.bootstrap_url, **kwargs
            )
            self._connected = True
            self._connect_time = time.monotonic()
            logger.info("WebSocket connected to bootstrap server")

        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to bootstrap server at "
                f"{self.bootstrap_url}: {e}"
            ) from e

        # Register
        try:
            register_msg = {
                "type": "register",
                "peer_id": self.node_id,
                "port": self.port,
                "capabilities": self.capabilities,
                "version": self.version,
                "region": self.region,
            }
            # Sprint 566: only include `address` when explicitly set
            # so legacy servers still see byte-identical messages.
            if self.advertise_address:
                register_msg["address"] = self.advertise_address
            # Sprint 838: similarly omit hardware_profile when not
            # supplied so pre-838 servers + pre-838 clients see
            # byte-identical register messages.
            if self.hardware_profile is not None:
                register_msg["hardware_profile"] = self.hardware_profile
            await self._ws.send(json.dumps(register_msg))
            logger.debug("Sent register message for node %s", self.node_id)

            # Wait for ack — sprint 319 — must filter by `type`
            # so a server pushing an announcement before the ack
            # doesn't get mis-parsed as the ack itself.
            data = await self._recv_typed(
                {"register_ack", "error"},
                timeout=self.connect_timeout,
            )

            if data.get("type") == "register_ack":
                self._registered = True
                self._server_time = data.get("server_time")

                # Update heartbeat interval from server
                server_hb = data.get("heartbeat_interval")
                if server_hb and isinstance(server_hb, (int, float)):
                    self.heartbeat_interval = int(server_hb)

                # Parse peer list
                self._peers = []
                for p in data.get("peers", []):
                    self._peers.append(BootstrapPeer(
                        peer_id=p.get("peer_id", ""),
                        address=p.get("address", ""),
                        port=p.get("port", 8000),
                        capabilities=p.get("capabilities", []),
                        region=p.get("region"),
                        version=p.get("version"),
                        # Sprint 838 — pick up relayed hardware
                        # profile when present; tolerates pre-838
                        # servers that omit the key.
                        hardware_profile=p.get("hardware_profile"),
                    ))

                logger.info(
                    "Registered with bootstrap server — peer_id=%s, "
                    "discovered %d peer(s), heartbeat=%ds",
                    self.node_id,
                    len(self._peers),
                    self.heartbeat_interval,
                )

                if self.on_peers_discovered and self._peers:
                    self.on_peers_discovered(self._peers)

                return self._peers

            elif data.get("type") == "error":
                raise ConnectionError(
                    f"Bootstrap registration rejected: {data.get('message')}"
                )
            else:
                raise ConnectionError(
                    f"Unexpected response from bootstrap: {data.get('type')}"
                )

        except asyncio.TimeoutError:
            raise ConnectionError(
                "Timed out waiting for register_ack from bootstrap server"
            )
        except Exception:
            await self.disconnect()
            raise

    async def start_heartbeat(self) -> None:
        """Start background heartbeat to keep the registration alive."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            return  # Already running

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.debug(
            "Heartbeat started (interval=%ds)", self.heartbeat_interval
        )

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to the bootstrap server."""
        while self._connected and self._ws:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                if not self._connected or not self._ws:
                    break

                heartbeat = {
                    "type": "heartbeat",
                    "peer_id": self.node_id,
                    "uptime": time.monotonic() - (self._connect_time or 0),
                }
                await self._ws.send(json.dumps(heartbeat))
                logger.debug("Heartbeat sent to bootstrap server")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Heartbeat failed: %s", e)
                self._connected = False
                break

    async def start_peer_refresh(self, interval: float = 60.0) -> None:
        """Sprint 632 — start background periodic peer-list refresh.

        Closes the race where bootstrap-server peer state grows
        after register-time (server restart wiped state then peers
        rejoined; peer joined the network after this client
        registered; etc.) and the client never re-fetches. Live
        evidence from sprint 630: fleet recovery required 3 daemon
        restarts in specific order to coax bootstrap-mediated
        symmetric discovery into working.

        Idempotent: a second call while the task is already running
        is a no-op (does not spawn a second task).

        ``on_peers_discovered`` fires only for NEWLY-discovered
        peers (delta vs. current ``self._peers``). Refreshes that
        return an identical list do NOT re-fire the callback —
        otherwise downstream auto-dial sweeps would re-attempt
        connections every interval.
        """
        if self._refresh_task and not self._refresh_task.done():
            return  # Idempotent
        self._refresh_task = asyncio.create_task(
            self._peer_refresh_loop(interval),
        )
        logger.debug(
            "Peer refresh started (interval=%.1fs)", interval,
        )

    async def _peer_refresh_loop(self, interval: float) -> None:
        """Periodically call get_peers; fire callback only on delta.

        First tick fires immediately (after a 0s yield so concurrent
        register-time setup completes first). Without this the operator
        waits a full `interval` for the FIRST refresh — which defeats
        the point when the register response was empty (sprint 630
        live failure mode).
        """
        # Sprint 632 — let the register-time callback chain settle
        # before we kick off the first refresh tick.
        await asyncio.sleep(0)
        first_tick = True
        while self._connected and self._ws:
            try:
                if not first_tick:
                    await asyncio.sleep(interval)
                first_tick = False
                if not self._connected or not self._ws:
                    break
                prev_ids = {p.peer_id for p in self._peers}
                # get_peers updates self._peers + fires callback (the
                # callback-fires-on-non-empty contract from sprint
                # 319). We need delta-only semantics here, so:
                # 1) temporarily swap the callback
                # 2) recompute the delta locally
                # 3) fire the real callback only for new peers
                real_cb = self.on_peers_discovered
                self.on_peers_discovered = None
                try:
                    peers = await self.get_peers()
                finally:
                    self.on_peers_discovered = real_cb
                new_peers = [
                    p for p in peers if p.peer_id not in prev_ids
                ]
                if new_peers and real_cb:
                    real_cb(new_peers)
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                logger.warning("Peer refresh failed: %s", e)
                # Don't break — refresh is best-effort; next tick
                # may succeed. If the WS itself died, the
                # `self._connected` check at top of loop will exit.

    async def get_peers(self) -> List[BootstrapPeer]:
        """Request an updated peer list from the bootstrap server.

        Sprint 319 — filter recv'd messages by `type == peer_list`
        so server-pushed announcements (peer_join/peer_leave)
        queued ahead of the response don't get mis-parsed as an
        empty peer list. Pre-fix `data.get("peers", [])` on a
        peer_join announcement silently returned [] and left the
        real peer_list response queued, misaligning every
        subsequent request/response.
        """
        if not self.is_connected or not self._ws:
            raise ConnectionError("Not connected to bootstrap server")

        await self._ws.send(json.dumps({"type": "get_peers"}))
        data = await self._recv_typed(
            {"peer_list", "error"},
            timeout=self.connect_timeout,
        )
        if data.get("type") == "error":
            raise ConnectionError(
                f"Bootstrap get_peers rejected: {data.get('message')}"
            )

        peers = []
        for p in data.get("peers", []):
            peers.append(BootstrapPeer(
                peer_id=p.get("peer_id", ""),
                address=p.get("address", ""),
                port=p.get("port", 8000),
                capabilities=p.get("capabilities", []),
                region=p.get("region"),
                version=p.get("version"),
                # Sprint 838 — relayed hw_profile, sibling
                # to the register-ack path above.
                hardware_profile=p.get("hardware_profile"),
            ))

        self._peers = peers
        if self.on_peers_discovered and peers:
            self.on_peers_discovered(peers)

        return peers

    async def _recv_typed(
        self,
        expected_types: set,
        timeout: float,
        max_drain: int = 32,
    ) -> Dict[str, Any]:
        """Sprint 319 — recv until a message whose `type` field
        matches one of ``expected_types`` arrives.

        Drains any messages whose `type` is not in the expected
        set into ``self._observed_announcements`` so they're not
        lost (peer_join/peer_leave announcements are valuable for
        future async-peer-update consumers).

        Bounded by ``max_drain`` to fail loud on pathological
        servers that never send the expected type — operators see
        a clear error instead of a silent hang.

        Raises:
            ConnectionError: if ``max_drain`` messages drained
                without a match.
            asyncio.TimeoutError: if recv exceeds ``timeout``.
        """
        if self._ws is None:
            raise ConnectionError("Not connected to bootstrap server")
        for _ in range(max_drain):
            raw = await asyncio.wait_for(
                self._ws.recv(), timeout=timeout,
            )
            data = json.loads(raw)
            if data.get("type") in expected_types:
                return data
            # Non-matching message — buffer (best-effort, capped)
            # so tests + future consumers can see what arrived
            self._observed_announcements.append(data)
            if len(self._observed_announcements) > 256:
                self._observed_announcements.pop(0)
        raise ConnectionError(
            f"Bootstrap server emitted {max_drain} messages "
            f"without one of {sorted(expected_types)}; aborting"
        )

    async def disconnect(self) -> None:
        """Cleanly disconnect from the bootstrap server."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Sprint 632 — cancel periodic peer refresh too. Without
        # this, the task would log "Peer refresh failed: ..." after
        # the WS is closed (best-effort survival inside the loop)
        # until the next interval boundary.
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            try:
                await self._ws.send(json.dumps({
                    "type": "disconnect",
                    "peer_id": self.node_id,
                }))
            except Exception:
                pass

            try:
                await self._ws.close()
            except Exception:
                pass

        self._ws = None
        self._connected = False
        self._registered = False
        logger.info("Disconnected from bootstrap server")

    def get_status(self) -> Dict[str, Any]:
        """Return current bootstrap client status."""
        return {
            "connected": self._connected,
            "registered": self._registered,
            "bootstrap_url": self.bootstrap_url,
            "node_id": self.node_id,
            "discovered_peers": len(self._peers),
            "heartbeat_interval": self.heartbeat_interval,
            "server_time": self._server_time,
            "uptime": (
                time.monotonic() - self._connect_time
                if self._connect_time else None
            ),
        }
