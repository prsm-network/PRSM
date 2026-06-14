"""
libp2p DHT Discovery Wrapper
=============================

Thin wrapper around ``Libp2pTransport`` that exposes the same public API
as ``PeerDiscovery`` (prsm/node/discovery.py).

Discovery is hybrid:
- **GossipSub** carries ephemeral ``capability_announce`` and
  ``shard_available`` messages so the network learns about peers in
  near-real-time.
- **Kademlia DHT** via ``PrsmDHTProvide`` / ``PrsmDHTFindProviders``
  stores durable content-provider records that survive node restarts.

Bootstrap uses ``transport.connect_to_peer()`` for each configured
bootstrap address; if zero succeed the status is set to degraded.
"""

import asyncio
import base64
import hashlib
import logging
import os
import secrets
import time
from typing import Any, Dict, List, Optional

from prsm.node.discovery import (
    PeerInfo,
    _announce_is_stale_replay,
    _announce_signing_bytes,
    _attest_announce_payload,
    _build_announce_credential,
    _join_bootstrap_address,
    _max_known_peers,
    _verify_peer_credential,
)

# sp1097 — bound on distinct providers cached per CID (the shard-cache analog of the
# _max_known_peers routing-table cap; prevents a Sybil flood from making one CID resolve
# to unbounded bogus providers / exhausting memory).
_MAX_SHARD_PROVIDERS = 64
from prsm.node.identity import verify_signature

logger = logging.getLogger(__name__)


def _resolve_stale_threshold() -> float:
    """Sprint 323 — read PRSM_PEER_STALE_SECONDS env (default
    600s). Fail-soft on garbage values: non-numeric or
    non-positive → return default.
    """
    try:
        val = float(os.getenv("PRSM_PEER_STALE_SECONDS", "600"))
        if val <= 0:
            return 600.0
        return val
    except (TypeError, ValueError):
        return 600.0


def _resolve_advertise_address() -> Optional[str]:
    """Sprint 566 — read PRSM_ADVERTISE_ADDRESS env var, with
    whitespace stripping + empty-string-as-unset semantics.

    Operators co-located with a bootstrap-server bootstrap via
    loopback (sprint-460 invariant) but need to tell the server
    what address remote peers should use. Without this env var,
    the bootstrap-server records the WS client_ip (= 127.0.0.1
    when bootstrapping via loopback), which is unreachable to
    remote peers.

    Returns None when unset or empty; the BootstrapClient then
    omits `address` from the register message and the server
    falls back to client_ip (pre-566 behavior).
    """
    raw = os.environ.get("PRSM_ADVERTISE_ADDRESS", "")
    stripped = raw.strip()
    return stripped or None


class BootstrapDead(Exception):
    """Sprint 564 — typed exception raised by ``_DeadBootstrapSentinel``
    when the poll loop calls ``get_peers`` on it.

    Pre-sprint-564 the sentinel had no ``get_peers`` method, so the
    poll loop's `await client.get_peers()` raised AttributeError on
    every tick — surrounding exception handler caught it but the log
    message read ``'_DeadBootstrapSentinel' object has no attribute
    'get_peers'``, which (a) leaked an internal class name into
    operator-visible logs, and (b) was indistinguishable from a real
    programming bug. ``BootstrapDead`` is the typed grep tag operators
    can alert on for "bootstrap is dead, retrying" specifically.
    """


class _DeadBootstrapSentinel:
    """Sprint 321 — placeholder installed in `_bootstrap_client`
    when a reconnect attempt fails so the poll loop's
    `while getattr(self, "_bootstrap_client", None) is not None`
    condition stays truthy and the next tick retries.

    Carries `is_connected=False` so the reconnect-on-drop branch
    keeps firing instead of treating the sentinel as a live
    client.

    Sprint 564: ``get_peers`` is now an async stub that raises
    ``BootstrapDead`` instead of yielding the bare AttributeError
    that fired pre-fix. The surrounding exception handler catches
    it the same way; the log message just stops leaking the
    internal class name.
    """
    is_connected = False

    async def get_peers(self):
        raise BootstrapDead(
            "dead-sentinel: previous bootstrap reconnect attempt "
            "failed; will retry on next tick"
        )


def _authenticated_capability_node_id(data: Dict[str, Any]) -> Optional[str]:
    """Sprint 1086/1087 — the cryptographically-authenticated node_id for a libp2p
    gossip announce (``capability_announce`` / ``shard_available``), or None to DROP it.
    Mirrors the WebSocket ``discovery._authenticated_announce_node_id``: the announce
    must carry a self-verifying origin attestation — ``sha256(origin_pubkey)[:32] ==
    node_id`` AND a valid signature over the canonical announce content bound to node_id
    + nonce. Unlike the WS path there is NO handshake-bound peer fallback (gossip is
    broadcast with no authenticated channel), so an unattested or mis-signed announce is
    dropped outright. This closes the sp1010 Residual A index/cache-poisoning vector: an
    attacker can no longer inject an arbitrary (node_id -> capabilities/attestation, or
    node_id -> shard-provider) entry under a victim's node_id."""
    claimed = data.get("node_id")
    if not claimed:
        return None
    pubkey = data.get("origin_pubkey", "")
    sig = data.get("origin_sig", "")
    nonce = data.get("nonce", "")
    if not (pubkey and sig):
        return None
    try:
        derived = hashlib.sha256(base64.b64decode(pubkey)).hexdigest()[:32]
        if derived == claimed and verify_signature(
                pubkey, _announce_signing_bytes(claimed, data, nonce), sig):
            return claimed
    except Exception:  # noqa: BLE001 — any parse/verify error → drop
        return None
    return None


class Libp2pDiscovery:
    """DHT-backed peer discovery layer for ``Libp2pTransport``.

    Provides the same public interface as ``PeerDiscovery`` so higher-level
    code can swap implementations without changes.
    """

    # Sprint 1085 — the libp2p gossip path does NOT authenticate a peer's claimed
    # node_id (sp1010 Residual A: it trusts a payload-supplied node_id without the
    # sha256(pubkey)==node_id + signature check the WebSocket PeerDiscovery enforces).
    # The DHT pool provider reads this to refuse honoring an attestation-derived
    # hardware tier from a libp2p-discovered peer (the sp1083 binding is void without an
    # authenticated node_id). Flip to True when libp2p origin-auth is ported.
    node_id_authenticated: bool = False

    def __init__(
        self,
        transport: Any,
        bootstrap_nodes: Optional[List[str]] = None,
        gossip: Optional[Any] = None,
        bootstrap_fallback_nodes: Optional[List[str]] = None,
        bootstrap_fallback_enabled: bool = True,
        local_hardware_profile: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            transport:       A ``Libp2pTransport`` instance.
            bootstrap_nodes: Primary multiaddr / host:port / ws(s)://
                             addresses (typically the canonical
                             ``wss://bootstrap1.prsm-network.com:8765``).
            gossip:          Optional ``Libp2pGossip`` for ephemeral
                             announcements.
            bootstrap_fallback_nodes:
                             Sprint 375 — multi-region backup hosts
                             tried in order when primary returns 0
                             connections. Closes the §7.29 honest-
                             scope SPOF: pre-sprint-375, libp2p-mode
                             operators ignored the EU+APAC fallback
                             list from NodeConfig and stuck in
                             degraded mode whenever the US droplet
                             was unreachable.
            bootstrap_fallback_enabled:
                             Operator-toggleable. When False, fallback
                             nodes are ignored (single-host posture
                             — backwards-compat for pre-375 deploys).
            **kwargs:        Ignored (drop-in compatibility).
        """
        self.transport = transport
        self.bootstrap_nodes: List[str] = bootstrap_nodes or []
        self.bootstrap_fallback_nodes: List[str] = (
            bootstrap_fallback_nodes or []
        )
        self.bootstrap_fallback_enabled: bool = (
            bootstrap_fallback_enabled
        )
        # Sprint 838 — operator's locally-detected hardware
        # profile (sprint 681). Forwarded to BootstrapClient
        # so the bootstrap-server caches + relays it to other
        # peers, closing the cold-start gossip gap.
        self._local_hardware_profile: Optional[Dict[str, Any]] = (
            local_hardware_profile
        )
        self.gossip = gossip

        # Peer capability index — node_id → PeerInfo
        self._capability_index: Dict[str, PeerInfo] = {}

        # Content shard cache — CID → list of peer_ids
        self._shard_cache: Dict[str, List[str]] = {}

        # Local node capabilities (set via set_local_capabilities)
        self._local_capabilities: List[str] = []
        self._local_backends: List[str] = []
        self._local_gpu_available: bool = False
        self._startup_timestamp: float = time.time()

        # Bootstrap status tracking. `discovered_peer_count` (sprint
        # 167) is the last-poll snapshot of how many peers the
        # bootstrap server surfaced; distinct from
        # _capability_index size (which monotonically grows).
        self._bootstrap_status: Dict[str, Any] = {
            "attempted": 0,
            "connected": 0,
            "degraded": False,
            "discovered_peer_count": 0,
            # Sprint 324 — operator-facing cumulative counters
            # for the new sprint-319-through-323 behaviors.
            # Never reset within process lifetime (process
            # restart is the natural reset boundary).
            "peer_join_events": 0,
            "peer_leave_events": 0,
            "stale_evictions": 0,
            "reconnect_attempts": 0,
            "reconnect_successes": 0,
        }

    # ── Lifecycle ────────────────────────────────────────────────

    async def start(self) -> None:
        """Bootstrap and subscribe to capability/shard gossip topics."""
        await self.bootstrap()

        if self.gossip is not None:
            self.gossip.subscribe("capability_announce", self._on_capability)
            self.gossip.subscribe("shard_available", self._on_shard_available)

        logger.info(
            "Libp2pDiscovery started — bootstrap: %s, degraded: %s",
            self._bootstrap_status["connected"],
            self._bootstrap_status["degraded"],
        )

    async def stop(self) -> None:
        """Cancel the bootstrap poll task (sprint 165), disconnect
        the BootstrapClient WebSocket (sprint 166), and let the
        transport handle the rest of teardown."""
        task = getattr(self, "_bootstrap_poll_task", None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Sprint 166 — close the bootstrap WebSocket cleanly so the
        # server sees a graceful unregister + we don't leak open
        # sockets across shutdown. Failures here are non-fatal —
        # we're tearing down anyway.
        client = getattr(self, "_bootstrap_client", None)
        if client is not None:
            try:
                await client.disconnect()
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Libp2pDiscovery: bootstrap client disconnect "
                    "raised: %s", exc,
                )
            self._bootstrap_client = None

    # ── Bootstrap ────────────────────────────────────────────────

    async def bootstrap(self) -> int:
        """Connect to each configured bootstrap node.

        Sprint 164 — two-stage strategy:
          1. Primary: ``transport.connect_to_peer(addr)`` for full
             libp2p P2P handshake. Works for multiaddr-style
             addresses (``/dns4/.../tcp/.../p2p/...``).
          2. Fallback: when primary returns 0 connected AND any
             configured address is a ``ws://`` or ``wss://`` URL,
             use the simpler BootstrapClient register/heartbeat
             protocol that the canonical PRSM bootstrap server
             actually serves on. The legacy ``PeerDiscovery`` had
             this fallback; ``Libp2pDiscovery`` was missing it,
             which left every operator using the canonical
             ``wss://bootstrap1.prsm-network.com:8765`` default
             stuck in degraded mode forever.

        Returns:
            Number of successfully connected nodes.
        """
        connected = 0
        candidates = self._candidate_bootstrap_addresses()
        self._bootstrap_status["attempted"] = len(candidates)
        # Sprint 375 — `active_url` records the candidate that
        # produced the first successful connect. Reset per
        # bootstrap() invocation so re-bootstrap reflects
        # current state.
        self._bootstrap_status["active_url"] = None

        for addr in candidates:
            try:
                peer = await self.transport.connect_to_peer(addr)
                if peer is not None:
                    connected += 1
                    if self._bootstrap_status[
                        "active_url"
                    ] is None:
                        self._bootstrap_status[
                            "active_url"
                        ] = addr
                    logger.debug("Bootstrap connected to %s", addr)
                else:
                    logger.debug("Bootstrap failed for %s (returned None)", addr)
            except Exception as exc:
                logger.debug("Bootstrap error for %s: %s", addr, exc)

        # Sprint 164 — BootstrapClient fallback for ws:// / wss://
        # URLs when primary libp2p path returned zero. Sprint 375
        # extends to consume the merged primary+fallback list.
        if connected == 0:
            wsocket_addrs = [
                a for a in candidates
                if a.startswith(("ws://", "wss://"))
            ]
            if wsocket_addrs and await self._try_bootstrap_client(
                wsocket_addrs,
            ):
                connected = 1
                # _try_bootstrap_client iterates internally and
                # sets _bootstrap_active_url on success via the
                # extended sprint-375 hook below.
                bc_url = getattr(
                    self, "_bootstrap_active_url", None,
                )
                if bc_url is not None:
                    self._bootstrap_status[
                        "active_url"
                    ] = bc_url

        self._bootstrap_status["connected"] = connected

        if connected == 0 and candidates:
            self._bootstrap_status["degraded"] = True
            logger.warning(
                "Libp2pDiscovery: all %d bootstrap node(s) "
                "unreachable; operating in degraded mode",
                len(candidates),
            )

        return connected

    def _candidate_bootstrap_addresses(self) -> List[str]:
        """Sprint 375 — merged primary + fallback list with
        dedup. Mirrors the pattern from PeerDiscovery
        (prsm/node/discovery.py:309-327) so libp2p + WebSocket
        discovery paths behave consistently when the canonical
        US droplet is unreachable.

        Order matters: primary tried first, fallback after.
        When fallback is disabled at construction, returns
        the primary list verbatim (backwards-compat for pre-
        sprint-375 single-host configs).
        """
        candidates: List[str] = []
        seen: set = set()
        for addr in self.bootstrap_nodes:
            if addr in seen:
                continue
            candidates.append(addr)
            seen.add(addr)
        if self.bootstrap_fallback_enabled:
            for addr in self.bootstrap_fallback_nodes:
                if addr in seen:
                    continue
                candidates.append(addr)
                seen.add(addr)
        return candidates

    async def _try_bootstrap_client(self, addrs: List[str]) -> bool:
        """Sprint 164 — BootstrapClient register/heartbeat fallback.

        The PRSM bootstrap server doesn't speak full libp2p — it
        runs a simpler register/heartbeat WebSocket protocol on
        ``wss://...:8765``. This method registers + starts heartbeat
        for the first reachable wss:// address.

        Mirrors the legacy ``PeerDiscovery._try_bootstrap_client``
        from prsm/node/discovery.py, which was the only place this
        fallback existed pre-sprint-164.

        Returns True on first successful registration, False
        otherwise. Does not raise — registration failures are
        logged at debug level and the next address is tried.
        """
        try:
            from prsm.bootstrap.client import BootstrapClient
        except ImportError:
            logger.debug(
                "BootstrapClient import failed; skipping fallback",
            )
            return False
        try:
            import prsm as _prsm_pkg
            _ver = _prsm_pkg.__version__
        except Exception:  # noqa: BLE001
            _ver = "unknown"
        for addr in addrs:
            try:
                logger.info(
                    "Libp2pDiscovery: BootstrapClient fallback "
                    "attempting %s", addr,
                )
                client = BootstrapClient(
                    bootstrap_url=addr,
                    node_id=self.transport.identity.node_id,
                    port=getattr(self.transport, "port", 9001),
                    capabilities=self._local_capabilities,
                    version=_ver,
                    advertise_address=_resolve_advertise_address(),
                    # Sprint 838 — advertise hw_profile so the
                    # bootstrap-server can relay it to other
                    # operators.
                    hardware_profile=self._local_hardware_profile,
                    # Sprint 1088 — a portable signed credential the server relays so
                    # receivers can authenticate node_id + the advertised hw_profile.
                    announce_credential=self._own_announce_credential(),
                )
                peers = await client.connect()
                await client.start_heartbeat()
                self._bootstrap_client = client
                # Sprint 375 — record the URL that produced
                # the successful registration so /bootstrap/
                # status surfaces it. Read back in bootstrap()
                # after _try_bootstrap_client returns True.
                self._bootstrap_active_url = addr
                self._hydrate_peers_from_bootstrap(peers)
                logger.info(
                    "Libp2pDiscovery: BootstrapClient connected via "
                    "%s — %d peer(s) discovered",
                    addr, len(peers),
                )
                # Sprint 165 — periodic peer-list polling so this
                # node sees newly-registered operators after its
                # own registration tick.
                self._bootstrap_poll_task = asyncio.create_task(
                    self._bootstrap_poll_loop(),
                )
                return True
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Libp2pDiscovery: BootstrapClient fallback "
                    "failed for %s: %s", addr, exc,
                )
                continue
        return False

    def _own_announce_credential(self) -> Optional[Dict[str, Any]]:
        """Sprint 1088 — build this node's portable, self-verifying credential for the
        bootstrap relay: a signed record binding its node_id (== sha256(pubkey)[:32]) to
        its advertised capabilities + hardware_profile. The bootstrap server relays it
        and other nodes re-verify it (``_verify_peer_credential``), so a malicious server
        — or a peer registering under a forged node_id — cannot get an unauthenticated
        (node_id -> attestation) entry honored in the Parallax pool. None when the
        identity can't sign (e.g. a stub transport in tests)."""
        identity = getattr(self.transport, "identity", None)
        if identity is None or not hasattr(identity, "sign"):
            return None
        payload: Dict[str, Any] = {
            "node_id": identity.node_id,
            "capabilities": list(self._local_capabilities or []),
        }
        if isinstance(self._local_hardware_profile, dict):
            payload["hardware_profile"] = self._local_hardware_profile
        nonce = secrets.token_hex(16)
        try:
            _attest_announce_payload(identity, payload, nonce)
            return _build_announce_credential(identity.node_id, payload, nonce)
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not build own announce credential: %s", exc)
            return None

    def _hydrate_peers_from_bootstrap(self, peers: List[Any]) -> None:
        """Sprint 165 — populate _capability_index from bootstrap-
        server peer payloads. Skips self (avoids self-edges in the
        capability graph).

        Sprint 167 — also tracks the last-poll discovered-peer
        count on _bootstrap_status so /status surfaces the live
        bootstrap-server view of the network (separate from
        `known` which conflates bootstrap + gossip sources).
        """
        own_id = self.transport.identity.node_id
        hydrated = 0
        for bp in peers:
            pid = getattr(bp, "peer_id", None)
            if not pid or pid == own_id:
                continue
            # Sprint 322 — thread bootstrap-reported capabilities
            # into PeerInfo so find_peers_with_capability /
            # find_peers_by_capability /
            # QueryOrchestrator-style selectors see the right
            # candidate set. Pre-fix this list was dropped and
            # every bootstrap-discovered peer appeared
            # capability-less to consumers.
            # Sprint 838 — bootstrap may have relayed each peer's
            # hardware_profile (cached from their registration).
            # Thread it through PeerInfo so the DHT-backed pool
            # (sp682) sees real fleet capacity for cold-start
            # joiners instead of sp836's conservative synthesis.
            bp_hw = getattr(bp, "hardware_profile", None)
            bp_hw = bp_hw if isinstance(bp_hw, dict) else None
            # Sprint 1088 — verify the peer's portable credential (relayed by the
            # bootstrap server). When it verifies AND matches the relayed peer_id, the
            # node_id is cryptographically authenticated → trust the SIGNED
            # hardware_profile (not the server's relayed copy, which a malicious server
            # could tamper) and mark the peer authenticated so the pool may honor its
            # attestation. An absent/invalid credential keeps the entry (for routing/
            # liveness) but marks it UNAUTHENTICATED → its attestation grants no tier.
            cred = getattr(bp, "announce_credential", None)
            authed_id = _verify_peer_credential(cred) if cred else None
            node_authenticated = bool(authed_id) and authed_id == pid
            # When authenticated, prefer the SIGNED capabilities + hardware_profile over
            # the server-relayed (unsigned) copies a malicious server could rewrite
            # (review MEDIUM-2). FRESHNESS NOTE (review MEDIUM-1): the bootstrap credential
            # vouches for IDENTITY, not recency — it carries no announce_time and is NOT
            # replay-checked here (unlike the sp1005 PEX path). That is intentional: the
            # attestation's own revocation/recency is enforced at the pool builder by
            # verified_tier_attestation (sp1060 CRL + sp1061-1083 TCB/QE), so a replayed
            # credential carrying a revoked/downlevel quote still resolves to tier-none.
            relayed_caps = list(getattr(bp, "capabilities", []) or [])
            caps = relayed_caps
            if node_authenticated:
                signed = cred.get("signed_payload") or {}
                signed_hw = signed.get("hardware_profile")
                if isinstance(signed_hw, dict):
                    bp_hw = signed_hw
                signed_caps = signed.get("capabilities")
                if isinstance(signed_caps, list):
                    caps = list(signed_caps)
            self._capability_index[pid] = PeerInfo(
                node_id=pid,
                # sp1097 (Domain-02 review F5) — use the sp1026 join helper instead of a
                # naive f"{addr}:{port}", which produced "host:port:port" (undialable, the
                # Tier-1-bench bug) when the operator advertised the documented host:port
                # form. The WS path was fixed in sp1026; this ports it to libp2p.
                address=_join_bootstrap_address(
                    getattr(bp, "address", "") or "", getattr(bp, "port", 0) or 0),
                capabilities=caps,
                last_seen=time.time(),
                last_capability_update=time.time(),
                hardware_profile=bp_hw,
                announce_credential=cred if node_authenticated else None,
                node_id_authenticated=node_authenticated,
            )
            hydrated += 1
        # Sprint 167 — track how many peers the bootstrap server
        # surfaced on the most recent poll. Distinct from
        # _capability_index size, which monotonically grows.
        self._bootstrap_status["discovered_peer_count"] = hydrated

    def _sweep_stale_peers(
        self, threshold_seconds: float,
    ) -> int:
        """Sprint 323 — evict peers whose `last_seen` is older
        than ``threshold_seconds``.

        Bootstrap-sourced peers get last_seen bumped on every
        successful peer_list poll; gossip-sourced peers get
        theirs bumped on every capability_announce. So any peer
        whose last_seen exceeds the threshold has been silent
        across all channels and is effectively gone — typically
        a hard crash with no peer_leave announcement.

        Defense in depth against:
          - Peer crashed hard (kernel panic / SIGKILL / network
            unplug): server eventually emits peer_leave but we
            may miss it if our bootstrap WebSocket was dropped
            during the announcement window.
          - Gossip-sourced peers that go silent (no peer_leave
            path at all).

        threshold_seconds=0 (or negative) evicts EVERY peer —
        useful as a forced cache reset.

        Returns: number of peers evicted (for logging /
        observability).
        """
        cutoff = time.time() - threshold_seconds
        # `<=` (not `<`) so threshold=0 evicts a peer whose
        # last_seen is exactly `now` — semantically "evict
        # everything when threshold is zero". With non-zero
        # thresholds the boundary case is irrelevant since
        # time.time() resolution exceeds the cutoff.
        stale_ids = [
            pid for pid, peer in self._capability_index.items()
            if peer.last_seen <= cutoff
        ]
        for pid in stale_ids:
            self._capability_index.pop(pid, None)
        # Sprint 324 — surface to operators via /bootstrap/status
        self._bootstrap_status["stale_evictions"] += len(stale_ids)
        return len(stale_ids)

    def _consume_bootstrap_announcements(self, client: Any) -> None:
        """Sprint 320 — process buffered server-pushed announcements
        from the BootstrapClient (sprint 319 wired the buffer).

        Handled announcement_types:
          - ``peer_join``: hydrate _capability_index eagerly so we
            see the peer at announcement cadence rather than waiting
            for the next bootstrap poll's peer_list.
          - ``peer_leave``: REMOVE the peer from _capability_index.
            Pre-sprint-320 the index only grew — peer_leave was a
            no-op, accumulating dead entries that downstream
            consumers would surface as live candidates.

        Forward-compat: unknown announcement_type values are
        ignored (do not raise).

        After processing, ``client._observed_announcements`` is
        cleared so the next poll tick doesn't re-process old
        events. Malformed entries are skipped per-item — one bad
        entry must not poison the whole batch.
        """
        buf = getattr(client, "_observed_announcements", None)
        if not buf:
            return
        own_id = self.transport.identity.node_id
        # Snapshot + clear up front so we don't race with new
        # announcements arriving while we process this batch
        snapshot = list(buf)
        buf.clear()
        for ann in snapshot:
            try:
                atype = ann.get("announcement_type")
                pid = ann.get("peer_id")
                if not pid or pid == own_id:
                    continue
                if atype == "peer_join":
                    # Sprint 322 — don't overwrite an existing
                    # entry. The bootstrap server's peer_list
                    # response is authoritative for capabilities;
                    # peer_join announcements only carry id +
                    # endpoint. A naive overwrite here would
                    # clobber the caps that
                    # _hydrate_peers_from_bootstrap filled in on
                    # the same poll tick. setdefault preserves
                    # the canonical entry + only adds new peers.
                    endpoint = ann.get("peer_endpoint", "")
                    self._capability_index.setdefault(
                        pid,
                        PeerInfo(
                            node_id=pid,
                            address=endpoint,
                            last_seen=time.time(),
                            last_capability_update=time.time(),
                        ),
                    )
                    self._bootstrap_status["peer_join_events"] += 1
                elif atype == "peer_leave":
                    self._capability_index.pop(pid, None)
                    self._bootstrap_status["peer_leave_events"] += 1
                # Unknown announcement_type: ignore (forward-compat)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Libp2pDiscovery: skipped malformed "
                    "announcement %r: %s", ann, exc,
                )

    async def _bootstrap_poll_loop(self) -> None:
        """Sprint 165 — periodic poll of bootstrap server for new
        peer registrations.

        Cadence is tuned via PRSM_BOOTSTRAP_POLL_SECONDS env (default
        60s). Polling runs until the task is cancelled (typically by
        node stop()). Per-tick failures are logged at debug and DO
        NOT terminate the loop — transient WebSocket dropouts get
        retried automatically.
        """
        try:
            interval = float(
                os.getenv("PRSM_BOOTSTRAP_POLL_SECONDS", "60"),
            )
            if interval <= 0:
                interval = 60.0
        except (TypeError, ValueError):
            interval = 60.0
        # Sprint 321 — re-read _bootstrap_client every tick so
        # reconnect-on-drop can swap a fresh client in without
        # the loop holding a stale local reference.
        while getattr(self, "_bootstrap_client", None) is not None:
            try:
                await asyncio.sleep(interval)
                client = getattr(self, "_bootstrap_client", None)
                if client is None:
                    break
                try:
                    peers = await client.get_peers()
                    self._hydrate_peers_from_bootstrap(peers)
                    # Sprint 320 — drain announcements buffered
                    # by _recv_typed during the get_peers
                    # exchange so peer_leave evicts departed
                    # peers and peer_join eagerly hydrates new
                    # ones.
                    self._consume_bootstrap_announcements(client)
                    # Sprint 323 — sweep stale entries for
                    # crashed peers that never sent
                    # peer_leave (kernel panic / SIGKILL /
                    # network unplug). Threshold defaults to
                    # 10 min via PRSM_PEER_STALE_SECONDS env.
                    evicted = self._sweep_stale_peers(
                        _resolve_stale_threshold(),
                    )
                    if evicted:
                        logger.info(
                            "Libp2pDiscovery: evicted %d stale "
                            "peer(s) past staleness threshold",
                            evicted,
                        )
                except Exception as exc:  # noqa: BLE001
                    # Sprint 321 — reconnect on a dropped
                    # WebSocket. If the client is still marked
                    # connected, this is a non-network error
                    # (parse / logic) — leave the client alone
                    # and retry on the next tick. Otherwise
                    # tear down + rebuild via the existing
                    # _try_bootstrap_client path so
                    # heartbeat/poll resume against a fresh
                    # socket.
                    if not getattr(client, "is_connected", True):
                        logger.warning(
                            "Libp2pDiscovery: bootstrap client "
                            "dropped (%s); attempting reconnect",
                            exc,
                        )
                        wsocket_addrs = [
                            a for a in self.bootstrap_nodes
                            if a.startswith(("ws://", "wss://"))
                        ]
                        # Detach the dead client so
                        # _try_bootstrap_client's success path
                        # can install the fresh one. Failure
                        # to reconnect leaves the slot empty;
                        # the outer-while condition then exits
                        # cleanly unless we re-arm it. We
                        # always re-arm with the (possibly
                        # failed) attempt's result so the loop
                        # keeps retrying.
                        self._bootstrap_client = None
                        if wsocket_addrs:
                            # Sprint 324 — operator-facing
                            # cumulative reconnect telemetry
                            self._bootstrap_status[
                                "reconnect_attempts"
                            ] += 1
                            ok = await self._try_bootstrap_client(
                                wsocket_addrs,
                            )
                            if ok:
                                self._bootstrap_status[
                                    "reconnect_successes"
                                ] += 1
                        # If reconnect succeeded, retry
                        # get_peers immediately so the next-
                        # poll wait doesn't add latency on top
                        # of the drop.
                        client = getattr(
                            self, "_bootstrap_client", None,
                        )
                        if client is not None and getattr(
                            client, "is_connected", False,
                        ):
                            try:
                                peers = await client.get_peers()
                                self._hydrate_peers_from_bootstrap(peers)
                                self._consume_bootstrap_announcements(client)
                            except Exception as retry_exc:  # noqa: BLE001
                                logger.debug(
                                    "Libp2pDiscovery: post-"
                                    "reconnect get_peers "
                                    "raised: %s", retry_exc,
                                )
                        else:
                            # Reconnect failed — keep the loop
                            # alive by re-arming the slot with
                            # the dead client so the outer
                            # while condition stays truthy.
                            # Next tick will try again.
                            self._bootstrap_client = client or (
                                # dead client object isn't
                                # discoverable here; we rely
                                # on _try_bootstrap_client's
                                # contract that it either
                                # installs a working client or
                                # leaves the slot empty. Put a
                                # sentinel marker that
                                # is_connected=False so we
                                # keep retrying.
                                _DeadBootstrapSentinel()
                            )
                    else:
                        logger.debug(
                            "Libp2pDiscovery: bootstrap poll "
                            "tick raised on connected client: "
                            "%s", exc,
                        )
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Libp2pDiscovery: bootstrap poll tick raised: %s",
                    exc,
                )

    # ── Peer query helpers (mirrors PeerDiscovery) ───────────────

    def get_known_peers(self) -> List[PeerInfo]:
        """Return all peers in the capability index."""
        return list(self._capability_index.values())

    def find_peers_by_capability(
        self,
        required: List[str],
        match_all: bool = True,
    ) -> List[PeerInfo]:
        """Find peers by required capabilities.

        Args:
            required:  Capability strings to search for.
            match_all: If True, peer must have ALL required capabilities.
                       If False, peer must have ANY of them.
        """
        required_lower = {c.lower() for c in required}
        results: List[PeerInfo] = []
        for peer in self._capability_index.values():
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
        """Find peers that have a specific capability."""
        cap_lower = capability.lower()
        results = [
            p for p in self._capability_index.values()
            if cap_lower in {c.lower() for c in p.capabilities}
        ]
        results.sort(key=lambda p: p.last_seen, reverse=True)
        return results

    def find_peers_with_backend(self, backend: str) -> List[PeerInfo]:
        """Find peers that support a specific backend."""
        backend_lower = backend.lower()
        results = [
            p for p in self._capability_index.values()
            if backend_lower in {b.lower() for b in p.supported_backends}
        ]
        results.sort(key=lambda p: p.last_seen, reverse=True)
        return results

    def find_peers_with_gpu(self) -> List[PeerInfo]:
        """Find peers that have GPU available."""
        results = [p for p in self._capability_index.values() if p.gpu_available]
        results.sort(key=lambda p: p.last_seen, reverse=True)
        return results

    # ── Local capabilities ───────────────────────────────────────

    def set_local_capabilities(
        self,
        capabilities: List[str],
        backends: List[str],
        gpu_available: bool = False,
    ) -> None:
        """Store this node's capabilities for announcement."""
        self._local_capabilities = capabilities
        self._local_backends = backends
        self._local_gpu_available = gpu_available

    async def announce_capabilities(self) -> int:
        """Publish local capabilities via GossipSub.

        Returns:
            1 if published successfully, 0 otherwise.
        """
        if self.gossip is None:
            return 0
        # Sprint 1086 — self-verifying origin attestation so receivers can authenticate
        # node_id (sha256(pubkey)==node_id + signature) rather than trusting the raw
        # gossip claim. A fresh nonce is bound into the signed content per announce.
        payload = {
            "node_id": self.transport.identity.node_id,
            "capabilities": self._local_capabilities,
            "supported_backends": self._local_backends,
            "gpu_available": self._local_gpu_available,
            "startup_timestamp": self._startup_timestamp,
            # sp1097 — signed monotonic timestamp so a receiver can reject a replayed
            # (stale) announce (parity with the WS sp941 defense).
            "announce_time": time.time(),
            "nonce": secrets.token_hex(16),
        }
        _attest_announce_payload(
            self.transport.identity, payload, payload["nonce"])
        return await self.gossip.publish("capability_announce", payload)

    def record_job_success(self, node_id: str) -> None:
        """Record a successful job completion for a peer."""
        peer = self._capability_index.get(node_id)
        if peer:
            peer.job_success_count += 1

    def record_job_failure(self, node_id: str) -> None:
        """Record a job failure/timeout for a peer."""
        peer = self._capability_index.get(node_id)
        if peer:
            peer.job_failure_count += 1
            peer.last_failure_time = time.time()

    # ── Content routing (dual: GossipSub + DHT) ──────────────────

    async def provide_content(self, cid: str) -> None:
        """Announce content availability via both GossipSub and DHT.

        - GossipSub ``shard_available`` for immediate propagation.
        - ``PrsmDHTProvide`` for durable, cross-restart DHT record.
        """
        node_id = self.transport.identity.node_id

        # 1. Ephemeral GossipSub announcement — sp1087: attested so receivers can
        # authenticate node_id (a peer can't claim a victim provides a shard, nor
        # advertise a shard it doesn't hold under a forged id).
        if self.gossip is not None:
            try:
                payload = {
                    "cid": cid, "node_id": node_id,
                    "announce_time": time.time(),   # sp1097 — replay defense
                    "nonce": secrets.token_hex(16),
                }
                _attest_announce_payload(
                    self.transport.identity, payload, payload["nonce"])
                await self.gossip.publish("shard_available", payload)
            except Exception as exc:
                logger.debug("GossipSub provide_content error: %s", exc)

        # 2. Durable DHT record
        try:
            await self.transport.dht_provide(cid)
        except Exception as exc:
            logger.debug("DHT provide error for %s: %s", cid, exc)

        # Update local shard cache
        providers = self._shard_cache.setdefault(cid, [])
        if node_id not in providers:
            providers.append(node_id)

    async def find_content_providers(
        self, cid: str, limit: int = 20
    ) -> List[PeerInfo]:
        """Find peers that hold *cid*.

        Checks local shard_cache first; falls back to DHT lookup.

        Returns:
            List of PeerInfo for peers believed to hold the content.
        """
        results: List[PeerInfo] = []

        # Fast path: local cache
        cached_ids = self._shard_cache.get(cid, [])
        for peer_id in cached_ids[:limit]:
            info = self._capability_index.get(peer_id)
            if info:
                results.append(info)

        if results:
            return results

        # Slow path: DHT
        try:
            peer_ids = await self.transport.dht_find_providers(cid, limit)
        except Exception as exc:
            logger.debug("DHT find_providers error for %s: %s", cid, exc)
            return []

        for peer_id in peer_ids:
            info = self._capability_index.get(peer_id)
            if info:
                results.append(info)
            else:
                # Create a minimal stub entry
                results.append(
                    PeerInfo(
                        node_id=peer_id,
                        address="",
                        last_seen=time.time(),
                        last_capability_update=time.time(),
                    )
                )

        return results

    # ── Status / telemetry ───────────────────────────────────────

    def get_bootstrap_status(self) -> Dict[str, Any]:
        """Return bootstrap connectivity status.

        Sprint 324 — extends with cumulative counters for the
        sprint-319-through-323 behaviors and a `client_state`
        field summarizing the BootstrapClient slot for fast
        operator triage.
        """
        # Sprint 324 — derive client_state for operator triage
        # without exposing the raw client object:
        #   - "none"          : no client installed
        #   - "dead"          : sentinel from a failed reconnect
        #   - "disconnected"  : client present but is_connected=False
        #   - "connected"     : client present and is_connected=True
        client = getattr(self, "_bootstrap_client", None)
        if client is None:
            client_state = "none"
        elif isinstance(client, _DeadBootstrapSentinel):
            client_state = "dead"
        elif getattr(client, "is_connected", False):
            client_state = "connected"
        else:
            client_state = "disconnected"
        return {
            "attempted": self._bootstrap_status["attempted"],
            "connected": self._bootstrap_status["connected"],
            "degraded": self._bootstrap_status["degraded"],
            "bootstrap_nodes": list(self.bootstrap_nodes),
            # Sprint 375 — surface fallback config + the
            # active candidate URL so operators see which
            # bootstrap host their node is actually using.
            # Closes the §7.29 honest-scope SPOF visibility
            # gap — pre-fix, operators had no way to tell
            # whether a degraded node was failing-primary
            # or failing-all.
            "bootstrap_fallback_nodes": list(
                self.bootstrap_fallback_nodes
            ),
            "bootstrap_fallback_enabled": (
                self.bootstrap_fallback_enabled
            ),
            "active_url": self._bootstrap_status.get(
                "active_url",
            ),
            # Sprint 167 — last-poll snapshot of peers the bootstrap
            # server knows about (excluding self).
            "discovered_peer_count": self._bootstrap_status.get(
                "discovered_peer_count", 0,
            ),
            # Sprint 324 — operator-facing cumulative counters
            "peer_join_events": self._bootstrap_status[
                "peer_join_events"
            ],
            "peer_leave_events": self._bootstrap_status[
                "peer_leave_events"
            ],
            "stale_evictions": self._bootstrap_status[
                "stale_evictions"
            ],
            "reconnect_attempts": self._bootstrap_status[
                "reconnect_attempts"
            ],
            "reconnect_successes": self._bootstrap_status[
                "reconnect_successes"
            ],
            "client_state": client_state,
        }

    def get_bootstrap_telemetry(self) -> Dict[str, Any]:
        """Return telemetry-compatible bootstrap data."""
        return {
            "addresses_validated": self._bootstrap_status["attempted"],
            "addresses_rejected": 0,
            "fallback_activated": False,
            "fallback_attempted": 0,
            "fallback_succeeded": False,
            "backoff_total_seconds": 0.0,
            "source_policy": "libp2p_dht",
        }

    # ── Internal callbacks ───────────────────────────────────────

    async def _on_capability(
        self, subtype: str, data: Dict[str, Any], sender_id: str
    ) -> None:
        """Update capability index from a ``capability_announce`` message.

        Resets reliability counters only on restart (new startup_timestamp)
        or capability change. Periodic heartbeats only refresh last_seen.

        Sprint 1086 — the announce's node_id must be CRYPTOGRAPHICALLY authenticated
        (origin attestation) before it's trusted; an unattested / mis-signed announce is
        dropped (never inserted into the capability index that feeds peer selection +
        the sp1083 attestation pool).
        """
        node_id = _authenticated_capability_node_id(data)
        if not node_id:
            logger.debug(
                "dropping unauthenticated capability_announce from sender=%s "
                "(no valid origin attestation)",
                (sender_id or "?")[:8],
            )
            return

        existing = self._capability_index.get(node_id)
        # sp1097 (Domain-02 review F1) — reject a replayed (stale-timestamp) announce so a
        # captured genuine envelope can't re-assert an old address/caps or refresh
        # last_seen for a dead/changed peer (parity with the WS sp941 defense).
        if existing is not None and _announce_is_stale_replay(
                data, existing.last_announce_time):
            return
        # sp1097 (F2) — bound the capability index (memory + eclipse-magnitude defense);
        # only NEW node_ids are subject to the cap (a tracked peer always refreshes). Sybil
        # node_ids are free, so without this the index grows unbounded under a flood.
        if existing is None and len(self._capability_index) >= _max_known_peers():
            logger.debug(
                "libp2p capability index at cap (%d) — dropping new announce from %s",
                _max_known_peers(), node_id[:8])
            return

        new_startup = data.get("startup_timestamp", 0.0)
        new_caps = {c.lower() for c in data.get("capabilities", [])}

        if existing is not None:
            old_startup = existing.startup_timestamp
            old_caps = {c.lower() for c in existing.capabilities}

            # Reset reliability only on restart or capability change
            if new_startup > old_startup or new_caps != old_caps:
                existing.job_success_count = 0
                existing.job_failure_count = 0
                existing.last_failure_time = 0.0
                existing.startup_timestamp = new_startup

            existing.capabilities = data.get("capabilities", existing.capabilities)
            existing.supported_backends = data.get(
                "supported_backends", existing.supported_backends
            )
            existing.gpu_available = data.get("gpu_available", existing.gpu_available)
            existing.last_seen = time.time()
            existing.last_capability_update = time.time()
            # sp1097 — record the accepted announce_time for the next replay check.
            existing.last_announce_time = float(data.get("announce_time") or 0.0)
            # sp1088 review LOW-1 — this announce is origin-authenticated (sp1086), so a
            # previously-unauthenticated entry (e.g. a credential-less bootstrap hydrate)
            # is UPGRADED to authenticated rather than left stuck at tier-none.
            existing.node_id_authenticated = True
        else:
            self._capability_index[node_id] = PeerInfo(
                node_id=node_id,
                address=data.get("address", ""),
                display_name=data.get("display_name", ""),
                roles=data.get("roles", []),
                capabilities=data.get("capabilities", []),
                supported_backends=data.get("supported_backends", []),
                gpu_available=data.get("gpu_available", False),
                last_seen=time.time(),
                last_capability_update=time.time(),
                startup_timestamp=new_startup,
                # sp1097 — seed the replay-defense timestamp from the signed announce.
                last_announce_time=float(data.get("announce_time") or 0.0),
                # sp1088 — this announce was origin-authenticated (sp1086), so the peer's
                # node_id is cryptographically established: mark it so the pool provider
                # may honor an attestation it carries.
                node_id_authenticated=True,
            )

    async def _on_shard_available(
        self, subtype: str, data: Dict[str, Any], sender_id: str
    ) -> None:
        """Update shard cache from a ``shard_available`` message.

        Sprint 1087 — the provider's node_id must be cryptographically authenticated
        (origin attestation) before it's cached; an unattested / mis-signed announce is
        dropped so an attacker can't poison the content-routing cache (claim a victim
        provides a shard, or advertise a shard under a forged node_id)."""
        cid = data.get("cid", "")
        if not cid:
            return
        node_id = _authenticated_capability_node_id(data)
        if not node_id:
            logger.debug(
                "dropping unauthenticated shard_available from sender=%s",
                (sender_id or "?")[:8],
            )
            return

        providers = self._shard_cache.setdefault(cid, [])
        if node_id not in providers:
            # sp1097 (Domain-02 review F2) — bound the per-CID providers list so a Sybil
            # flood (free-to-mint node_ids, each a valid signed announce) can't make one
            # CID resolve to unbounded bogus providers / exhaust memory.
            if len(providers) >= _MAX_SHARD_PROVIDERS:
                logger.debug(
                    "shard providers for %s at cap (%d) — dropping %s",
                    cid[:12], _MAX_SHARD_PROVIDERS, node_id[:8])
                return
            providers.append(node_id)
