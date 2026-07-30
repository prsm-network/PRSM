"""
Storage Provider
================

Pin-space contribution to the PRSM proprietary content store.
Auto-detects the local ContentStore, pledges configurable storage,
accepts pin requests, and earns FTNS rewards for storage proofs.

Includes challenge-response storage proof verification to ensure
providers actually store the content they claim to pin.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from datetime import datetime, timezone

from prsm.core.bandwidth_limiter import BandwidthLimiter

if TYPE_CHECKING:
    from prsm.node.config import NodeConfig
    from prsm.node.content_provider import ContentProvider

from prsm.node.config import is_active_now
from prsm.node.gossip import (
    GOSSIP_CONTENT_ADVERTISE,
    GOSSIP_PROOF_OF_STORAGE,
    GOSSIP_STORAGE_CONFIRM,
    GOSSIP_STORAGE_REQUEST,
    GossipProtocol,
)
from prsm.node.transport import MSG_DIRECT, P2PMessage, PeerConnection, WebSocketTransport
from prsm.node.identity import NodeIdentity
from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.node.storage_proofs import (
    StorageProofVerifier,
    StorageProver,
    StorageChallenge,
    StorageProof,
    ProofType,
    ChallengeStatus,
    ChallengeRecord,
    DEFAULT_CHALLENGE_TIMEOUT_MINUTES,
)

try:
    from prsm.node.libp2p_transport import Libp2pTransportError
except ImportError:
    Libp2pTransportError = OSError  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)

# Reward rate: FTNS per GB per epoch (1 hour)
STORAGE_REWARD_RATE = 0.1

# Default challenge configuration
DEFAULT_CHALLENGE_INTERVAL = 300.0  # 5 minutes between challenges
DEFAULT_CHALLENGE_DIFFICULTY_BYTES = 4096  # 4KB chunks to prove
MAX_CONCURRENT_CHALLENGES = 10  # Max pending challenges at once


@dataclass
class PinnedContent:
    """Tracks content pinned by this node."""
    cid: str
    size_bytes: int
    pinned_at: float = field(default_factory=time.time)
    requester_id: str = ""
    last_verified: float = field(default_factory=time.time)
    last_challenge_time: float = 0.0  # Last time this CID was challenged
    successful_challenges: int = 0  # Count of successful proof verifications
    failed_challenges: int = 0  # Count of failed proof verifications


@dataclass
class ChallengeConfig:
    """Configuration for storage proof challenges."""
    challenge_interval: float = DEFAULT_CHALLENGE_INTERVAL  # Seconds between challenges
    challenge_difficulty: int = DEFAULT_CHALLENGE_DIFFICULTY_BYTES  # Bytes to prove
    challenge_timeout_minutes: int = DEFAULT_CHALLENGE_TIMEOUT_MINUTES
    max_concurrent_challenges: int = MAX_CONCURRENT_CHALLENGES
    enable_challenges: bool = True  # Master switch for challenge system
    min_challenge_interval_per_cid: float = 60.0  # Min seconds between challenges for same CID


class StorageProvider:
    """Contributes pin space to the network and earns FTNS rewards.

    Requires a running PRSM ContentStore. Gracefully degrades if the
    store is not available — storage features are disabled but the
    node continues to operate.

    Includes challenge-response storage proof verification to ensure
    this provider (and remote providers) actually store the content
    they claim to pin.
    """

    def __init__(
        self,
        identity: NodeIdentity,
        gossip: GossipProtocol,
        ledger: LocalLedger,
        pledged_gb: float = 10.0,
        reward_interval: float = 3600.0,  # 1 hour
        challenge_config: Optional[ChallengeConfig] = None,
        config: Optional["NodeConfig"] = None,
        content_economy: Optional[Any] = None,  # Phase 4: Replication tracking
        transport: Optional[Any] = None,
        discovery: Optional[Any] = None,
        content_provider: Optional["ContentProvider"] = None,
    ):
        self.identity = identity
        self.gossip = gossip
        self.ledger = ledger
        self.transport = transport
        self.discovery = discovery
        self.pledged_gb = pledged_gb
        self.reward_interval = reward_interval
        self.config = config  # NodeConfig for scheduling checks
        self.content_economy = content_economy  # Phase 4: Update replication status on challenges

        # Phase 1.3 Task 3e: defer content_request serving to
        # ContentProvider when it has the CID, to avoid a dual-handler
        # race on MSG_DIRECT. storage_provider still handles pin-only
        # content (CIDs not tracked by ContentProvider).
        self._content_provider = content_provider

        # sp1484 — replication controls. Fetching on a REMOTE peer's request is the
        # point of pledging storage, so this is ON by default for a storage node
        # (pledging GB IS the opt-in); PRSM_REPLICATION_ENABLED=0 disables it.
        # `max_replica_bytes` bounds any SINGLE replica independently of the
        # gossiped size, which an attacker controls. The semaphore stops a burst of
        # storage requests from opening unbounded simultaneous fetches.
        import os as _os_env
        self.replication_enabled = (
            str(_os_env.environ.get("PRSM_REPLICATION_ENABLED", "1")).strip().lower()
            not in ("0", "false", "no")
        )
        try:
            self.max_replica_bytes = int(
                _os_env.environ.get("PRSM_MAX_REPLICA_BYTES", "")
                or 2 * 1024 * 1024 * 1024)
        except (TypeError, ValueError):
            self.max_replica_bytes = 2 * 1024 * 1024 * 1024
        try:
            _rep_conc = max(1, int(
                _os_env.environ.get("PRSM_REPLICATION_CONCURRENCY", "") or 2))
        except (TypeError, ValueError):
            _rep_conc = 2
        self._replication_sem = asyncio.Semaphore(_rep_conc)
        
        # Bandwidth limits (0 = unlimited)
        # Sprint 761 — operator-facing env vars exposing the
        # existing BandwidthLimiter mechanism. Pre-761 the limits
        # were hardcoded to 0 (unlimited); operators wanting to
        # cap their daemon's network usage (consumer ISPs with
        # metered/capped plans, gaming PC that shouldn't saturate
        # upload during the workday, etc.) had no knob. Now:
        # `PRSM_STORAGE_UPLOAD_MBPS=10` caps upload to 10 Mbps;
        # similarly for download. Default 0 = unlimited preserves
        # the pre-761 behavior for the existing operator fleet.
        # Non-float values safely default to 0 (rather than
        # failing daemon-start mysteriously).
        import os as _os
        def _resolve_mbps(env_name: str) -> float:
            raw = _os.environ.get(env_name, "").strip()
            if not raw:
                return 0.0
            try:
                val = float(raw)
            except ValueError:
                logger.warning(
                    f"{env_name}={raw!r} not a valid float; "
                    f"defaulting to 0 (unlimited)"
                )
                return 0.0
            if val < 0:
                logger.warning(
                    f"{env_name}={val} is negative; "
                    f"defaulting to 0 (unlimited)"
                )
                return 0.0
            return val

        self.upload_mbps_limit: float = _resolve_mbps(
            "PRSM_STORAGE_UPLOAD_MBPS"
        )
        self.download_mbps_limit: float = _resolve_mbps(
            "PRSM_STORAGE_DOWNLOAD_MBPS"
        )

        # Bandwidth limiter for throttling content serving
        self.bandwidth_limiter = BandwidthLimiter(
            upload_mbps=self.upload_mbps_limit,
            download_mbps=self.download_mbps_limit,
        )

        # Challenge configuration
        self.challenge_config = challenge_config or ChallengeConfig()

        self.storage_available = False
        self.pinned_content: Dict[str, PinnedContent] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self.ledger_sync = None  # Set by node.py after construction

        # Storage proof system: verifier challenges other providers; prover
        # answers challenges to this provider. StorageProofVerifier reads
        # shard bytes from the ContentStore.
        self._proof_verifier = StorageProofVerifier(
            challenge_timeout_minutes=self.challenge_config.challenge_timeout_minutes,
            max_pending_challenges=self.challenge_config.max_concurrent_challenges,
        )
        self._storage_prover: Optional[StorageProver] = None  # Initialized after ContentStore check

        # Track pending challenges issued by this provider
        self._pending_challenges: Dict[str, ChallengeRecord] = {}

        # Track challenges received by this provider (to answer)
        self._received_challenges: Dict[str, StorageChallenge] = {}

        # Provider reputation tracking (for remote providers we challenge)
        self._provider_reputation: Dict[str, float] = {}  # provider_id -> reputation score (0.0-1.0)

        # Deduplication for incoming storage challenges
        self._seen_challenge_ids: Dict[str, float] = {}

    @property
    def used_bytes(self) -> int:
        return sum(p.size_bytes for p in self.pinned_content.values())

    @property
    def used_gb(self) -> float:
        return round(self.used_bytes / (1024**3), 4)

    @property
    def available_gb(self) -> float:
        return max(0.0, self.pledged_gb - self.used_gb)

    async def start(self) -> None:
        """Detect ContentStore, register handlers, start reward and challenge loops."""
        self._running = True

        self.storage_available = await self._check_content_store()
        if self.storage_available:
            logger.info(f"ContentStore active, storage provider active ({self.pledged_gb}GB pledged)")

            self._storage_prover = StorageProver(
                identity=self.identity,
            )
            
            # Register gossip handlers
            self.gossip.subscribe(GOSSIP_STORAGE_REQUEST, self._on_storage_request)
            
            # Subscribe to challenge-related gossip messages
            self.gossip.subscribe("storage_challenge", self._on_storage_challenge)
            self.gossip.subscribe("storage_proof_response", self._on_storage_proof_response)

            # Register unified direct message handler
            self._register_direct_handler()

            # Start background tasks
            self._tasks.append(asyncio.create_task(self._reward_loop()))
            
            # Start challenge loop if enabled
            if self.challenge_config.enable_challenges:
                self._tasks.append(asyncio.create_task(self._challenge_loop()))
                self._tasks.append(asyncio.create_task(self._challenge_cleanup_loop()))
                logger.info(
                    f"Storage proof challenges enabled: interval={self.challenge_config.challenge_interval}s, "
                    f"difficulty={self.challenge_config.challenge_difficulty} bytes"
                )
        else:
            logger.warning(
                "ContentStore not available — "
                "storage features disabled. Ensure ContentStore is initialized."
            )

    async def stop(self) -> None:
        """Stop all background tasks and cleanup resources."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

        if self._storage_prover:
            await self._storage_prover.close()
            self._storage_prover = None

    async def _check_content_store(self) -> bool:
        """Check if ContentStore is available.

        Initializes the global ContentStore singleton lazily if one is not
        already present. Returns True when the store is reachable.
        """
        try:
            from prsm.storage import get_content_store, init_content_store
            store = get_content_store()
            if store is None:
                # Sprint 168 — thread node_id so manifests carry
                # correct owner attribution. Pre-fix the fallback
                # init defaulted node_id="" (empty string), breaking
                # downstream royalty + provenance attribution.
                store = init_content_store(
                    node_id=self.identity.node_id
                    if self.identity else "",
                )
            return store is not None
        except Exception:
            return False

    async def pin_content(self, cid: str) -> bool:
        """Pin content locally, FETCHING it first when we do not already hold it.

        sp1484 — until now this only "promoted" content that happened to be local:
        `exists_local(...)` False -> return False, without ever fetching the bytes.
        So `replicas=N` on every upload and CLI flag was DECORATIVE — no copy was
        ever placed on any other node, and published content died with the
        publisher's box. The handler around it already emitted STORAGE_CONFIRM +
        CONTENT_ADVERTISE on success, i.e. it advertised a pin that could never
        happen. This is what makes replication real.

        Safety properties (this method now pulls bytes on a REMOTE peer's request,
        so each one is load-bearing):
          * INTEGRITY — request_content_to_file re-verifies the CID with a streaming
            read and deletes the partial on mismatch, so a peer cannot poison us
            with bytes that are not the content we asked for.
          * OPERATOR CONSENT — a CID the operator's content filter blocks is never
            fetched or stored. A moderation decision must not be overridable by a
            stranger's gossip.
          * DECLARED SIZE IS UNTRUSTED — the caller's capacity check uses the
            gossiped `size_bytes`, which an attacker controls ("declare 1 byte,
            serve 10 GiB"). We therefore re-check the ACTUAL on-disk size after the
            fetch and discard it if it no longer fits. The transport's own
            gateway ceiling bounds the transfer itself.
          * BOUNDED CONCURRENCY — a burst of storage requests must not open
            unbounded simultaneous fetches; replication fetches are serialized
            through a semaphore.
        """
        if not self.storage_available:
            return False
        try:
            from prsm.storage import ContentHash, get_content_store

            store = get_content_store()
            if store is None:
                return False
            try:
                exists = await store.exists_local(ContentHash.from_hex(cid))
            except ValueError:
                # Malformed content hash hex — preserved legacy behavior.
                return False
            if exists:
                size = await self._get_content_size(cid)
                self.pinned_content[cid] = PinnedContent(cid=cid, size_bytes=size)
                return True

            # Not local — replicate it.
            size = await self._replicate_content(cid)
            if size is None:
                return False
            self.pinned_content[cid] = PinnedContent(cid=cid, size_bytes=size)
            return True
        except ValueError:
            return False
        except Exception as e:
            logger.error(f"Failed to pin {cid}: {e}")
            return False

    async def _replicate_content(self, cid: str) -> Optional[int]:
        """Fetch ``cid`` from the network and stage it so this node can SERVE it.

        Returns the stored size in bytes, or None if replication was refused or
        failed (in which case nothing is stored, pinned, confirmed or advertised).
        """
        if not self.replication_enabled:
            return None
        provider = getattr(self, "_content_provider", None)
        if provider is None or not hasattr(provider, "request_content_to_file"):
            logger.debug("sp1484 replication skipped for %s: no content provider", cid[:12])
            return None

        # Operator consent: never fetch or host content this operator blocked.
        blocked = getattr(self, "_content_filter_store", None)
        if blocked is not None:
            try:
                if blocked.is_cid_blocked(cid):
                    logger.info(
                        "sp1484 refusing to replicate %s — blocked by this "
                        "operator's content filter", cid[:12])
                    return None
            except Exception:  # noqa: BLE001 — a filter error must not force a store
                return None

        publisher = getattr(provider, "content_publisher", None)
        staging_dir = getattr(publisher, "staging_dir", None)
        if publisher is None or staging_dir is None:
            logger.debug(
                "sp1484 replication skipped for %s: no staging publisher", cid[:12])
            return None

        import hashlib
        import os as _os
        from pathlib import Path as _Path

        tmp_path = _Path(staging_dir) / f".replicating-{cid[:24]}.part"
        async with self._replication_sem:
            try:
                got = await provider.request_content_to_file(cid, tmp_path)
                if got is None:
                    return None

                actual = _os.path.getsize(tmp_path)
                # The gossiped size was untrusted; this one is measured.
                if actual > self.available_gb * (1024 ** 3):
                    logger.warning(
                        "sp1484 discarding replicated %s: actual %.1fMB exceeds "
                        "available %.1fGB (declared size was untrusted)",
                        cid[:12], actual / (1024 ** 2), self.available_gb)
                    return None
                if self.max_replica_bytes and actual > self.max_replica_bytes:
                    logger.warning(
                        "sp1484 discarding replicated %s: actual %.1fMB exceeds the "
                        "per-item replication cap %.1fMB",
                        cid[:12], actual / (1024 ** 2),
                        self.max_replica_bytes / (1024 ** 2))
                    return None

                # Content-address the staged file so the streaming serve path
                # (local_publish_path -> _local_stream_info) can find it.
                h = hashlib.sha256()
                with open(tmp_path, "rb") as fh:
                    for block in iter(lambda: fh.read(1024 * 1024), b""):
                        h.update(block)
                content_hash = h.hexdigest()
                final_path = _Path(staging_dir) / content_hash
                _os.replace(tmp_path, final_path)

                if not publisher.register_local_publish_tier_a(cid, content_hash):
                    logger.warning(
                        "sp1484 staged %s but re-registration failed", cid[:12])
                    return None
                provider.register_local_content(
                    cid=cid, size_bytes=actual, content_hash=content_hash,
                    filename=None, metadata={"tier": "A", "replica": True},
                )
                logger.info(
                    "sp1484 replicated %s (%.1fMB) — now serving a durable copy",
                    cid[:12], actual / (1024 ** 2))
                return actual
            except Exception as exc:  # noqa: BLE001 — never break the storage loop
                logger.warning("sp1484 replication of %s failed: %s", cid[:12], exc)
                return None
            finally:
                try:
                    _Path(tmp_path).unlink(missing_ok=True)
                except OSError:
                    pass

    async def _get_content_size(self, cid: str) -> int:
        """Get the size of stored content from ContentStore.

        Reads the manifest's total_size for *cid*. Returns 0 when the
        ContentStore is unavailable, the CID hex is malformed, or the
        content is not present locally.
        """
        try:
            from prsm.storage import ContentHash, get_content_store

            store = get_content_store()
            if store is None:
                return 0
            return await store.size_local(ContentHash.from_hex(cid))
        except ValueError:
            # Malformed content hash hex (bad algorithm prefix, etc.).
            return 0
        except Exception as e:
            logger.debug(f"_get_content_size failed for {cid[:12]}...: {e}")
            return 0

    async def verify_pin(self, cid: str) -> bool:
        """Verify that content exists in the local ContentStore."""
        if not self.storage_available:
            return False
        try:
            from prsm.storage import ContentHash, get_content_store

            store = get_content_store()
            if store is None:
                return False
            return await store.exists_local(ContentHash.from_hex(cid))
        except ValueError:
            # Malformed content hash hex.
            return False
        except Exception:
            return False

    # ── Gossip handlers ──────────────────────────────────────────

    async def _on_storage_request(self, subtype: str, data: Dict[str, Any], origin: str) -> None:
        """Handle a storage request from the network."""
        if not self._running or not self.storage_available:
            return

        # Check if we're within active hours
        if self.config and not is_active_now(self.config):
            logger.debug("Node is outside active hours, declining storage request")
            return

        cid = data.get("cid", "")
        size_bytes = data.get("size_bytes", 0)
        requester_id = data.get("requester_id", origin)

        if not cid:
            return

        # Don't re-pin content we already have
        if cid in self.pinned_content:
            return

        # Check if we have space
        size_gb = size_bytes / (1024**3) if size_bytes > 0 else 0.001
        if size_gb > self.available_gb:
            return

        # Pin the content
        success = await self.pin_content(cid)
        if success:
            self.pinned_content[cid].requester_id = requester_id

            # Confirm to network
            await self.gossip.publish(GOSSIP_STORAGE_CONFIRM, {
                "cid": cid,
                "provider_id": self.identity.node_id,
                "size_bytes": self.pinned_content[cid].size_bytes,
            })

            # Advertise that we can now serve this content
            await self.gossip.publish(GOSSIP_CONTENT_ADVERTISE, {
                "cid": cid,
                "filename": "",
                "size_bytes": self.pinned_content[cid].size_bytes,
                "content_hash": "",
                "creator_id": requester_id,
                "provider_id": self.identity.node_id,
                "created_at": self.pinned_content[cid].pinned_at,
                "metadata": {},
            })

            # Announce via DHT
            if self.discovery:
                try:
                    await self.discovery.provide_content(cid)
                except Exception as exc:
                    logger.debug("DHT provide_content failed for %s: %s", cid[:12], exc)

            logger.info(f"Pinned content {cid[:12]}... for {requester_id[:8]}")

    # ── Cleanup helpers ─────────────────────────────────────────────

    def _cleanup_seen_challenges(self) -> None:
        """Evict stale entries from the challenge deduplication set."""
        cutoff = time.time() - 600
        for cid in list(self._seen_challenge_ids.keys()):
            if self._seen_challenge_ids.get(cid, 0) < cutoff:
                self._seen_challenge_ids.pop(cid, None)

    # ── Direct message handling ──────────────────────────────────

    def _register_direct_handler(self) -> None:
        """Register unified handler for all direct P2P messages."""
        if self.transport is not None:
            self.transport.on_message(MSG_DIRECT, self._on_direct_message)

    async def _on_direct_message(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Dispatch incoming direct messages by subtype."""
        subtype = msg.payload.get("subtype", "")
        if subtype == "content_request":
            await self._on_direct_content_request(msg, peer)
        elif subtype == "storage_challenge":
            await self._on_storage_challenge(subtype, msg.payload, msg.sender_id)
        elif subtype == "storage_proof_response":
            await self._on_storage_proof_response(subtype, msg.payload, msg.sender_id)

    # ── Content serving ────────────────────────────────────────────

    def register_content_handler(self, transport: WebSocketTransport) -> None:
        """Register to handle direct content_request messages for pinned CIDs.

        .. deprecated::
            Use the ``transport`` constructor parameter instead.
            This method is kept for backward compatibility.
        """
        import warnings
        warnings.warn(
            "register_content_handler() is deprecated. Pass transport= to StorageProvider constructor.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.transport is None:
            self.transport = transport
        self._register_direct_handler()
        logger.info("Storage provider registered for content serving")

    async def _on_direct_content_request(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Serve pinned content via PRSM gateway URL.

        Phase 1.3 Task 3e: defer to ContentProvider when it has the CID
        in its _local_content — ContentProvider's serve path runs the
        canonical payment flow and emits the canonical response shape.
        storage_provider responds only for pinned CIDs that
        ContentProvider doesn't serve.

        Phase 1.3 Task 3f: replica-only pinned serves (CIDs pinned here
        but missing from ContentProvider._local_content) now delegate
        back to ContentProvider.serve_on_behalf_of_replica so the
        creator earns FTNS for the access and the on-chain event fires.
        Without this delegation, replica serves bypass on-chain
        royalties and silently violate the Phase 1 acceptance criterion
        ("anyone can independently verify a creator earned FTNS by
        reading Base mainnet logs").
        """
        subtype = msg.payload.get("subtype", "")
        if subtype != "content_request":
            return

        cid = msg.payload.get("cid", "")
        request_id = msg.payload.get("request_id", "")

        if cid not in self.pinned_content:
            return  # Let ContentProvider handle unknown CIDs

        # Defer to ContentProvider if it also has the CID — it runs the
        # canonical serve path with payment integration, and both
        # handlers firing on MSG_DIRECT would otherwise race (our
        # response skipping the payment await always arrived first).
        if (
            self._content_provider is not None
            and self._content_provider.has_local_content(cid)
        ):
            return

        # Phase 1.3 Task 3f: delegate to ContentProvider for the full
        # payment + gossip + canonical response pipeline. storage_provider
        # owns the gateway URL / pin metadata; ContentProvider owns the
        # royalty/settlement contract. Without this delegation, replica
        # serves bypass on-chain royalties and silently violate the
        # Phase 1 acceptance criterion.
        if self._content_provider is None:
            # Bootstrap misconfig — fall back to silent skip rather than
            # serving content for free.
            logger.warning(
                f"storage_provider has no content_provider reference; "
                f"cannot pay creator for replica serve of {cid[:12]}..."
            )
            return

        gateway_url = f"prsm://content/{cid}"
        served = await self._content_provider.serve_on_behalf_of_replica(
            cid=cid,
            request_id=request_id,
            peer=peer,
            replica_size_bytes=self.pinned_content[cid].size_bytes,
            gateway_url=gateway_url,
        )
        if not served:
            logger.debug(
                f"Replica serve declined for {cid[:12]}... (no content_index "
                f"record); requester will try another provider"
            )

    # ── Reward loop ──────────────────────────────────────────────

    async def _reward_loop(self) -> None:
        """Periodically prove storage and claim rewards."""
        while self._running:
            await asyncio.sleep(self.reward_interval)
            if not self.pinned_content:
                continue

            try:
                # Verify a sample of pins
                verified_count = 0
                for cid, content in list(self.pinned_content.items()):
                    if await self.verify_pin(cid):
                        content.last_verified = time.time()
                        verified_count += 1
                    else:
                        # Content no longer pinned — remove from tracking
                        del self.pinned_content[cid]

                if verified_count == 0:
                    continue

                # Calculate reward: FTNS per GB stored
                reward = round(self.used_gb * STORAGE_REWARD_RATE, 6)
                if reward > 0:
                    tx = await self.ledger.credit(
                        wallet_id=self.identity.node_id,
                        amount=reward,
                        tx_type=TransactionType.STORAGE_REWARD,
                        description=f"Storage reward: {self.used_gb:.4f}GB, {verified_count} CIDs verified",
                    )

                    # Broadcast earning via ledger sync
                    if self.ledger_sync:
                        try:
                            await self.ledger_sync.broadcast_transaction(tx)
                        except Exception:
                            pass

                    # Announce proof to network
                    await self.gossip.publish(GOSSIP_PROOF_OF_STORAGE, {
                        "provider_id": self.identity.node_id,
                        "verified_cids": verified_count,
                        "total_gb": self.used_gb,
                        "reward_ftns": reward,
                    })

                    logger.info(f"Storage reward: {reward} FTNS for {self.used_gb:.4f}GB ({verified_count} CIDs)")

            except Exception as e:
                logger.error(f"Reward loop error: {e}")

    # ── Storage Proof Challenge System ─────────────────────────────────────

    async def _challenge_loop(self) -> None:
        """Periodically issue storage challenges to verify pinned content.
        
        This loop challenges this provider's own pinned content to ensure
        we can prove storage, and can also challenge remote providers.
        """
        while self._running:
            await asyncio.sleep(self.challenge_config.challenge_interval)
            
            if not self.pinned_content:
                continue
                
            try:
                # Self-challenge: verify we can prove storage of our pinned content
                await self._self_challenge_pinned_content()
                
                # Clean up expired challenges
                expired_count = self._proof_verifier.cleanup_expired_challenges()
                if expired_count > 0:
                    logger.debug(f"Cleaned up {expired_count} expired challenges")
                    
            except Exception as e:
                logger.error(f"Challenge loop error: {e}")

    async def _challenge_cleanup_loop(self) -> None:
        """Periodically clean up expired challenges and update reputations."""
        while self._running:
            await asyncio.sleep(60.0)  # Check every minute
            
            try:
                # Process expired challenges
                await self._process_expired_challenges()

                # Evict stale dedup entries
                self._cleanup_seen_challenges()

            except Exception as e:
                logger.error(f"Challenge cleanup loop error: {e}")

    async def _compute_trusted_merkle_root(self, cid: str):
        """sp1253 — recompute the TRUSTED merkle root for a CID from the LOCAL content
        so proof verification binds to content we actually hold, NOT the
        provider-supplied (attacker-controllable) root. Returns None when the content
        isn't locally available → the verifier then fails closed. Uses the same
        MerkleProofGenerator (default chunk_size) the prover uses, so a genuine proof's
        root matches. Never raises (must not break the challenge loop)."""
        try:
            prover = self._storage_prover
            client = getattr(prover, "content_client", None) if prover else None
            if client is None or not hasattr(client, "get_content"):
                return None
            content = await client.get_content(cid)
            if not content:
                return None
            from prsm.node.storage_proofs import MerkleProofGenerator
            return MerkleProofGenerator().build_merkle_tree(content).root_hash
        except Exception as exc:  # noqa: BLE001
            logger.debug("sp1253: trusted merkle root unavailable for %s: %s",
                         (cid or "?")[:16], exc)
            return None

    async def _self_challenge_pinned_content(self) -> None:
        """Challenge ourselves to prove storage of pinned content.

        This ensures we can generate valid proofs for content we claim to store.
        """
        now = time.time()
        challenges_issued = 0
        proofs_verified = 0
        proofs_failed = 0
        
        for cid, content in list(self.pinned_content.items()):
            # Check if we should challenge this CID
            time_since_last = now - content.last_challenge_time
            if time_since_last < self.challenge_config.min_challenge_interval_per_cid:
                continue
                
            # Don't issue too many challenges at once
            if challenges_issued >= self.challenge_config.max_concurrent_challenges:
                break
            
            # Generate a self-challenge
            try:
                challenge = self._proof_verifier.generate_challenge(
                    cid=cid,
                    challenger_id=self.identity.node_id,
                    difficulty=self.challenge_config.challenge_difficulty,
                    proof_type=ProofType.MERKLE,
                )
                
                challenges_issued += 1
                content.last_challenge_time = now
                
                # If we have a storage prover, answer the challenge
                if self._storage_prover:
                    proof = await self._storage_prover.answer_challenge(challenge)
                    
                    if proof:
                        # sp1253 — verify against a root recomputed from the LOCAL
                        # content (not the provider-supplied root) so a proof can't be
                        # forged with an arbitrary tree.
                        expected_root = await self._compute_trusted_merkle_root(cid)
                        is_valid, error_msg = await self._proof_verifier.verify_proof(
                            proof=proof,
                            challenge=challenge,
                            expected_merkle_root=expected_root,
                        )
                        
                        if is_valid:
                            proofs_verified += 1
                            content.successful_challenges += 1
                            content.last_verified = time.time()
                            logger.debug(f"Self-challenge passed for {cid[:16]}...")
                            
                            # Update ContentEconomy replication status (Phase 4)
                            if self.content_economy:
                                await self.content_economy.update_replication_status(
                                    cid=cid,
                                    provider_id=self.identity.node_id,
                                    has_content=True,
                                )
                        else:
                            proofs_failed += 1
                            content.failed_challenges += 1
                            logger.warning(
                                f"Self-challenge FAILED for {cid[:16]}...: {error_msg}. "
                                f"Content may not be properly stored."
                            )
                            
                            # Record the failure
                            self._proof_verifier.record_provider_result(
                                provider_id=self.identity.node_id,
                                success=False,
                            )
                    else:
                        # Could not generate proof - content may not be available
                        proofs_failed += 1
                        content.failed_challenges += 1
                        logger.warning(
                            f"Could not generate proof for {cid[:16]}... - "
                            f"content may not be available in ContentStore"
                        )
                        
                        # Verify if content is actually pinned
                        is_pinned = await self.verify_pin(cid)
                        if not is_pinned:
                            logger.error(
                                f"Content {cid[:16]}... is no longer pinned - "
                                f"removing from tracking"
                            )
                            del self.pinned_content[cid]
                            
                            # Update ContentEconomy replication status (Phase 4)
                            if self.content_economy:
                                await self.content_economy.update_replication_status(
                                    cid=cid,
                                    provider_id=self.identity.node_id,
                                    has_content=False,
                                )
                            
            except Exception as e:
                logger.error(f"Error during self-challenge for {cid[:16]}...: {e}")
                
        if challenges_issued > 0:
            logger.info(
                f"Self-challenge round complete: {challenges_issued} challenges, "
                f"{proofs_verified} verified, {proofs_failed} failed"
            )

    async def _process_expired_challenges(self) -> None:
        """Process expired challenges and update provider reputations."""
        now = datetime.now(timezone.utc)
        expired_challenges = []
        
        for challenge_id, record in list(self._pending_challenges.items()):
            if record.challenge.is_expired() and record.status == ChallengeStatus.PENDING:
                expired_challenges.append((challenge_id, record))
                
        for challenge_id, record in expired_challenges:
            # Update status
            self._proof_verifier._update_challenge_status(challenge_id, ChallengeStatus.EXPIRED)
            
            # Record the failure
            provider_id = record.challenge.challenger_id  # Provider being challenged
            self._proof_verifier.record_provider_result(
                provider_id=provider_id,
                success=False,
                expired=True,
            )
            
            # Update reputation
            self._update_provider_reputation(provider_id, success=False, expired=True)
            
            # Remove from pending
            del self._pending_challenges[challenge_id]
            
            logger.warning(
                f"Challenge {challenge_id[:16]}... expired for provider {provider_id[:8]}"
            )

    async def issue_challenge_to_provider(
        self,
        provider_id: str,
        cid: str,
    ) -> Optional[StorageChallenge]:
        """Issue a storage challenge to a remote provider.
        
        Args:
            provider_id: ID of the provider to challenge
            cid: Content ID to challenge them on
            
        Returns:
            The challenge that was issued, or None if rate limited
        """
        # Check rate limiting
        if not self._proof_verifier.can_challenge(provider_id):
            logger.debug(f"Rate limited: cannot challenge provider {provider_id[:8]}")
            return None
            
        # Generate the challenge
        challenge = self._proof_verifier.generate_challenge(
            cid=cid,
            challenger_id=provider_id,  # The provider being challenged
            difficulty=self.challenge_config.challenge_difficulty,
            proof_type=ProofType.MERKLE,
        )
        
        # Track the challenge
        self._pending_challenges[challenge.challenge_id] = ChallengeRecord(
            challenge=challenge,
            status=ChallengeStatus.PENDING,
        )
        
        # Send challenge via direct P2P with gossip fallback
        challenge_payload = {
            "subtype": "storage_challenge",
            "challenge": challenge.to_dict(),
            "challenger_id": self.identity.node_id,
            "target_provider_id": provider_id,
        }
        if self.transport is not None:
            msg = P2PMessage(
                msg_type=MSG_DIRECT,
                sender_id=self.identity.node_id,
                payload=challenge_payload,
            )
            try:
                sent = await self.transport.send_to_peer(provider_id, msg)
                if not sent:
                    raise ConnectionError("direct send failed")
            except (ConnectionError, Libp2pTransportError, OSError) as exc:
                logger.debug("Direct challenge to %s failed (%s), falling back to gossip", provider_id[:8], exc)
                await self.gossip.publish("storage_challenge", challenge_payload)
        else:
            await self.gossip.publish("storage_challenge", challenge_payload)
        
        logger.info(
            f"Issued challenge {challenge.challenge_id[:16]}... to provider "
            f"{provider_id[:8]} for CID {cid[:16]}..."
        )
        
        return challenge

    async def _on_storage_challenge(
        self,
        subtype: str,
        data: Dict[str, Any],
        origin: str,
    ) -> None:
        """Handle an incoming storage challenge from another node.

        This is called when another node challenges us to prove storage.
        """
        # Deduplication guard
        challenge_id = data.get("challenge", {}).get("challenge_id", "")
        if challenge_id in self._seen_challenge_ids:
            logger.debug("Dropping duplicate challenge %s", challenge_id[:16])
            return
        if challenge_id:
            self._seen_challenge_ids[challenge_id] = time.time()

        if not self._running or not self._storage_prover:
            return

        # Check if this challenge is for us
        target_provider_id = data.get("target_provider_id", "")
        if target_provider_id != self.identity.node_id:
            return  # Not our challenge
            
        try:
            challenge_data = data.get("challenge", {})
            challenge = StorageChallenge.from_dict(challenge_data)
            
            # Check if we have this content
            cid = challenge.cid
            if cid not in self.pinned_content:
                logger.debug(f"Received challenge for unknown CID {cid[:16]}...")
                return
                
            # Answer the challenge
            proof = await self._storage_prover.answer_challenge(challenge)
            
            if proof:
                # Send the proof response via direct P2P with gossip fallback
                challenger_id = data.get("challenger_id", origin)
                proof_payload = {
                    "subtype": "storage_proof_response",
                    "proof": proof.to_dict(),
                    "challenge_id": challenge.challenge_id,
                    "provider_id": self.identity.node_id,
                }
                if self.transport is not None:
                    proof_msg = P2PMessage(
                        msg_type=MSG_DIRECT,
                        sender_id=self.identity.node_id,
                        payload=proof_payload,
                    )
                    try:
                        sent = await self.transport.send_to_peer(challenger_id, proof_msg)
                        if not sent:
                            raise ConnectionError("direct send failed")
                    except (ConnectionError, Libp2pTransportError, OSError) as exc:
                        logger.debug("Direct proof to %s failed (%s), falling back to gossip", challenger_id[:8], exc)
                        await self.gossip.publish("storage_proof_response", proof_payload)
                else:
                    await self.gossip.publish("storage_proof_response", proof_payload)
                
                # Update our tracking
                self.pinned_content[cid].last_challenge_time = time.time()
                self.pinned_content[cid].successful_challenges += 1
                
                logger.debug(
                    f"Answered challenge {challenge.challenge_id[:16]}... for CID {cid[:16]}..."
                )
            else:
                logger.warning(
                    f"Failed to answer challenge {challenge.challenge_id[:16]}... for CID {cid[:16]}..."
                )
                self.pinned_content[cid].failed_challenges += 1
                
        except Exception as e:
            logger.error(f"Error handling storage challenge: {e}")

    async def _on_storage_proof_response(
        self,
        subtype: str,
        data: Dict[str, Any],
        origin: str,
    ) -> None:
        """Handle an incoming storage proof response.
        
        This is called when a provider responds to our challenge.
        """
        if not self._running:
            return
            
        try:
            proof_data = data.get("proof", {})
            challenge_id = data.get("challenge_id", "")
            provider_id = data.get("provider_id", origin)
            
            # Look up the challenge
            record = self._pending_challenges.get(challenge_id)
            if not record:
                logger.debug(f"Received proof for unknown challenge {challenge_id[:16]}...")
                return
                
            if record.status != ChallengeStatus.PENDING:
                logger.debug(f"Challenge {challenge_id[:16]}... already resolved")
                return
                
            # Deserialize and verify the proof
            proof = StorageProof.from_dict(proof_data)

            # sp1253 — bind to a root recomputed from LOCAL content if we hold this
            # CID; else None → the verifier fails closed (we cannot independently verify
            # storage of content we have no trusted root for, so we must NOT grant a
            # forged-proof VERIFIED on the remote path).
            expected_root = await self._compute_trusted_merkle_root(
                record.challenge.shard_hash)
            is_valid, error_msg = await self._proof_verifier.verify_proof(
                proof=proof,
                challenge=record.challenge,
                expected_merkle_root=expected_root,
            )
            
            if is_valid:
                # Update the record
                record.status = ChallengeStatus.VERIFIED
                record.proof = proof
                record.verified_at = datetime.now(timezone.utc)
                
                # Record success
                self._proof_verifier.record_provider_result(
                    provider_id=provider_id,
                    success=True,
                )
                
                # Update reputation
                self._update_provider_reputation(provider_id, success=True)
                
                logger.info(
                    f"Verified proof from provider {provider_id[:8]} for "
                    f"challenge {challenge_id[:16]}..."
                )
            else:
                # Proof verification failed
                record.status = ChallengeStatus.FAILED
                record.proof = proof
                
                # Record failure
                self._proof_verifier.record_provider_result(
                    provider_id=provider_id,
                    success=False,
                )
                
                # Update reputation
                self._update_provider_reputation(provider_id, success=False)
                
                logger.warning(
                    f"Proof verification FAILED from provider {provider_id[:8]} for "
                    f"challenge {challenge_id[:16]}...: {error_msg}"
                )
                
        except Exception as e:
            logger.error(f"Error handling storage proof response: {e}")

    def _update_provider_reputation(
        self,
        provider_id: str,
        success: bool,
        expired: bool = False,
    ) -> None:
        """Update a provider's reputation based on challenge result.
        
        Uses an exponential moving average for reputation scoring.
        
        Args:
            provider_id: Provider to update
            success: Whether the challenge was successful
            expired: Whether the challenge expired (worse than failure)
        """
        # Get current reputation (default to 1.0 for new providers)
        current = self._provider_reputation.get(provider_id, 1.0)
        
        # Calculate reputation delta
        if success:
            # Small boost for successful proof
            delta = 0.02
        elif expired:
            # Larger penalty for expired (no response)
            delta = -0.15
        else:
            # Moderate penalty for failed proof
            delta = -0.10
            
        # Apply exponential moving average (alpha = 0.3)
        alpha = 0.3
        new_reputation = current + alpha * (delta - current + 1.0)
        
        # Clamp to [0.0, 1.0]
        new_reputation = max(0.0, min(1.0, new_reputation))
        
        self._provider_reputation[provider_id] = new_reputation
        
        # Log significant reputation changes
        if abs(new_reputation - current) > 0.05:
            logger.info(
                f"Provider {provider_id[:8]} reputation: {current:.2f} -> {new_reputation:.2f}"
            )

    def get_provider_reputation(self, provider_id: str) -> float:
        """Get the reputation score for a provider.
        
        Args:
            provider_id: Provider ID to look up
            
        Returns:
            Reputation score (0.0 to 1.0), defaults to 1.0 for unknown providers
        """
        return self._provider_reputation.get(provider_id, 1.0)

    async def verify_remote_provider(
        self,
        provider_id: str,
        cid: str,
    ) -> Tuple[bool, str]:
        """Verify that a remote provider is storing content they claim.
        
        Issues a challenge and waits for response.
        
        Args:
            provider_id: Provider to verify
            cid: Content ID to verify
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Issue the challenge
        challenge = await self.issue_challenge_to_provider(provider_id, cid)
        
        if not challenge:
            return False, "Rate limited - cannot issue challenge"
            
        # Wait for response (with timeout)
        timeout = self.challenge_config.challenge_timeout_minutes * 60
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            record = self._pending_challenges.get(challenge.challenge_id)
            
            if record and record.status == ChallengeStatus.VERIFIED:
                return True, "Provider verified successfully"
            elif record and record.status == ChallengeStatus.FAILED:
                return False, "Provider failed to provide valid proof"
            elif record and record.status == ChallengeStatus.EXPIRED:
                return False, "Challenge expired"
                
            await asyncio.sleep(1.0)
            
        return False, "Verification timed out"

    # ── Statistics ─────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return storage provider statistics."""
        # Get challenge statistics from verifier
        verifier_stats = self._proof_verifier.get_provider_stats(self.identity.node_id)
        
        return {
            "storage_available": self.storage_available,
            "pledged_gb": self.pledged_gb,
            "used_gb": self.used_gb,
            "available_gb": self.available_gb,
            "pinned_cids": len(self.pinned_content),
            "reward_rate": STORAGE_REWARD_RATE,
            # Challenge system stats
            "challenge_config": {
                "enabled": self.challenge_config.enable_challenges,
                "interval": self.challenge_config.challenge_interval,
                "difficulty": self.challenge_config.challenge_difficulty,
                "timeout_minutes": self.challenge_config.challenge_timeout_minutes,
            },
            "challenge_stats": {
                "total_challenges": verifier_stats.get("total_challenges", 0),
                "successful_proofs": verifier_stats.get("successful_proofs", 0),
                "failed_proofs": verifier_stats.get("failed_proofs", 0),
                "expired_challenges": verifier_stats.get("expired_challenges", 0),
            },
            "pending_challenges": len(self._pending_challenges),
            "tracked_providers": len(self._provider_reputation),
        }

    async def update_limits(
        self,
        pledged_gb: Optional[float] = None,
        upload_mbps_limit: Optional[float] = None,
        download_mbps_limit: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Update storage limits at runtime.
        
        Changes take effect immediately. This method allows live tuning
        of storage allocation and bandwidth limits without restarting
        the node.
        
        Args:
            pledged_gb: Maximum storage to pledge in GB (must be positive)
            upload_mbps_limit: Upload bandwidth limit in Mbps (0 = unlimited)
            download_mbps_limit: Download bandwidth limit in Mbps (0 = unlimited)
        
        Returns:
            Dict with updated limit values
        
        Raises:
            ValueError: If any value is invalid
        """
        if pledged_gb is not None:
            if pledged_gb <= 0:
                raise ValueError(f"pledged_gb must be positive, got {pledged_gb}")
            # Check if new pledge is less than current usage
            if pledged_gb < self.used_gb:
                logger.warning(
                    f"Reducing pledged_gb from {self.pledged_gb} to {pledged_gb} "
                    f"but {self.used_gb:.4f}GB is already in use. "
                    "New pledge will be effective after content expires."
                )
            self.pledged_gb = pledged_gb
            
        if upload_mbps_limit is not None:
            if upload_mbps_limit < 0:
                raise ValueError(f"upload_mbps_limit cannot be negative, got {upload_mbps_limit}")
            self.upload_mbps_limit = upload_mbps_limit
            
        if download_mbps_limit is not None:
            if download_mbps_limit < 0:
                raise ValueError(f"download_mbps_limit cannot be negative, got {download_mbps_limit}")
            self.download_mbps_limit = download_mbps_limit
        
        # Update the bandwidth limiter with new limits
        await self.bandwidth_limiter.update_limits(
            upload_mbps=self.upload_mbps_limit,
            download_mbps=self.download_mbps_limit,
        )
        
        logger.info(
            f"Updated storage limits: pledged={self.pledged_gb}GB, "
            f"upload={self.upload_mbps_limit}Mbps, download={self.download_mbps_limit}Mbps"
        )
        
        return {
            "pledged_gb": self.pledged_gb,
            "upload_mbps_limit": self.upload_mbps_limit,
            "download_mbps_limit": self.download_mbps_limit,
            "used_gb": self.used_gb,
            "available_gb": self.available_gb,
        }

    def get_pinned_content_stats(self) -> List[Dict[str, Any]]:
        """Return detailed statistics for each pinned content."""
        result = []
        for cid, content in self.pinned_content.items():
            result.append({
                "cid": cid,
                "size_bytes": content.size_bytes,
                "pinned_at": content.pinned_at,
                "requester_id": content.requester_id,
                "last_verified": content.last_verified,
                "last_challenge_time": content.last_challenge_time,
                "successful_challenges": content.successful_challenges,
                "failed_challenges": content.failed_challenges,
            })
        return result

    def get_provider_stats_summary(self) -> Dict[str, Dict[str, Any]]:
        """Return reputation and stats for all tracked providers."""
        result = {}
        for provider_id in self._provider_reputation:
            stats = self._proof_verifier.get_provider_stats(provider_id)
            result[provider_id] = {
                "reputation": self._provider_reputation[provider_id],
                "total_challenges": stats.get("total_challenges", 0),
                "successful_proofs": stats.get("successful_proofs", 0),
                "failed_proofs": stats.get("failed_proofs", 0),
                "expired_challenges": stats.get("expired_challenges", 0),
            }
        return result
