"""
Content Economy - FTNS Payment Integration for Content Access
==============================================================

Wires together content uploads, retrieval, and FTNS payments.
Implements the Phase 4 content economy requirements:

1. FTNS payment on content access
2. Royalty distribution (8% original, 1% derivative)
3. Replication guarantees with minimum replica tracking
4. Content retrieval marketplace with provider bidding
5. Vector DB integration for semantic search

Integration Points:
- ContentUploader: Triggers payments on access
- ContentProvider: Handles retrieval with payment
- LocalLedger: FTNS accounting
- OnChainFTNSLedger: Base mainnet FTNS transfers
- ContentIndex: Provider discovery
- VectorStore: Semantic indexing
"""

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.node.gossip import GossipProtocol

if TYPE_CHECKING:
    from prsm.node.identity import NodeIdentity
    from prsm.node.content_index import ContentIndex
    from prsm.economy.ftns_onchain import OnChainFTNSLedger

logger = logging.getLogger(__name__)


# ── Royalty Configuration ─────────────────────────────────────────────────

# Phase 4 spec: 8% original creator, 1% derivative creators
ORIGINAL_CREATOR_ROYALTY_RATE = 0.08  # 8% to original creator
DERIVATIVE_CREATOR_ROYALTY_RATE = 0.01  # 1% to each derivative creator
NETWORK_FEE_RATE = 0.02  # 2% network fee
MAX_ROYALTY_CHAIN_DEPTH = 5  # Max derivative chain depth to track

# Alternative pricing model (existing 70/25/5 split for backward compatibility)
LEGACY_DERIVATIVE_SHARE = 0.70
LEGACY_SOURCE_SHARE = 0.25
LEGACY_NETWORK_SHARE = 0.05


# ── On-chain provenance feature flag (Phase 1) ────────────────────────────
# When PRSM_ONCHAIN_PROVENANCE=1, _distribute_royalties first attempts an
# on-chain RoyaltyDistributor.distributeRoyalty call. On any failure, the
# call falls through to the existing local-ledger split so payments are
# never lost.
ONCHAIN_PROVENANCE_ENABLED = os.getenv(
    "PRSM_ONCHAIN_PROVENANCE", ""
).lower() in ("1", "true", "yes")
# T6 (post-2026-05-07): addresses + RPC resolution moves to
# `prsm.config.networks.resolve_endpoints()` so PRSM_NETWORK=testnet
# correctly retargets these clients at Base Sepolia (matching what
# `prsm join-testnet` writes into the env file). The constants below
# remain as legacy fallbacks for the env-var-only operator path.
from prsm.config.networks import resolve_endpoints as _resolve_endpoints
_endpoints = _resolve_endpoints()
PROVENANCE_REGISTRY_ADDRESS = _endpoints.provenance_registry or ""
ROYALTY_DISTRIBUTOR_ADDRESS = _endpoints.royalty_distributor or ""
DEFAULT_FTNS_TOKEN_ADDRESS = (
    _endpoints.ftns_token or "0x5276a3756C85f2E9e46f6D34386167a209aa16e5"
)


class RoyaltyModel(str, Enum):
    """Royalty distribution models."""
    PHASE4 = "phase4"  # 8% original, 1% derivative, 2% network
    LEGACY = "legacy"  # 70/25/5 split


class PaymentStatus(str, Enum):
    """Status of a content access payment."""
    PENDING = "pending"
    ESCROWED = "escrowed"
    COMPLETED = "completed"
    PENDING_ONCHAIN = "pending_onchain"  # broadcast OK, receipt unknown
    FAILED = "failed"
    REFUNDED = "refunded"


# ── Data Classes ───────────────────────────────────────────────────────────

@dataclass
class ContentAccessPayment:
    """Tracks a single content access payment."""
    payment_id: str
    content_id: str
    accessor_id: str
    creator_id: str
    amount: Decimal
    royalty_model: RoyaltyModel
    status: PaymentStatus = PaymentStatus.PENDING
    escrow_tx_hash: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    royalty_distributions: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ReplicationStatus:
    """Tracks replication status for a content ID."""
    content_id: str
    min_replicas: int
    current_replicas: int
    providers: Set[str] = field(default_factory=set)
    last_verified: float = field(default_factory=time.time)
    pending_requests: int = 0
    verification_failures: int = 0


@dataclass
class ProviderBid:
    """A bid from a storage provider for content retrieval."""
    provider_id: str
    content_id: str
    price_ftns: Decimal
    estimated_latency_ms: float
    available_bandwidth_mbps: float
    reputation_score: float
    bid_timestamp: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 60.0)


@dataclass
class RetrievalRequest:
    """A content retrieval request with bidding."""
    request_id: str
    content_id: str
    requester_id: str
    max_price_ftns: Decimal
    min_replicas_required: int = 1
    status: str = "open"  # open, bidding, fulfilled, failed
    bids: List[ProviderBid] = field(default_factory=list)
    selected_provider: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    bid_deadline: float = field(default_factory=lambda: time.time() + 10.0)


# ── Content Economy Manager ───────────────────────────────────────────────

class ContentEconomy:
    """Manages FTNS payments, royalties, and content marketplace.
    
    Responsibilities:
    1. Process FTNS payments on content access
    2. Distribute royalties to creators (original + derivative chain)
    3. Track and enforce replication guarantees
    4. Manage content retrieval marketplace with provider bidding
    5. Index content into vector store for semantic search
    """
    
    def __init__(
        self,
        identity: "NodeIdentity",
        ledger: LocalLedger,
        gossip: GossipProtocol,
        content_index: "ContentIndex",
        ftns_ledger: Optional["OnChainFTNSLedger"] = None,
        royalty_model: RoyaltyModel = RoyaltyModel.PHASE4,
        min_replicas: int = 3,
        replication_check_interval: float = 300.0,
        vector_store: Optional[Any] = None,
        embedding_fn: Optional[Callable] = None,
    ):
        self.identity = identity
        self.ledger = ledger
        self.gossip = gossip
        self.content_index = content_index
        self.ftns_ledger = ftns_ledger
        self.royalty_model = royalty_model
        self.min_replicas = min_replicas
        self.replication_check_interval = replication_check_interval
        self.vector_store = vector_store
        self.embedding_fn = embedding_fn
        
        # In-memory tracking
        self._pending_payments: Dict[str, ContentAccessPayment] = {}
        self._replication_status: Dict[str, ReplicationStatus] = {}
        self._retrieval_requests: Dict[str, RetrievalRequest] = {}
        self._provider_reputation: Dict[str, float] = {}
        
        # Multi-party escrow for batch settlements
        self._escrow: Optional[Any] = None  # Set via set_escrow()
        
        # Background tasks
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
    async def start(self) -> None:
        """Start background tasks for replication monitoring."""
        self._running = True
        
        # Start replication check loop
        self._tasks.append(asyncio.create_task(self._replication_monitor_loop()))
        
        # Start bid cleanup loop
        self._tasks.append(asyncio.create_task(self._bid_cleanup_loop()))
        
        # Register gossip handlers
        self.gossip.subscribe("retrieval_bid", self._on_retrieval_bid)
        self.gossip.subscribe("retrieval_fulfill", self._on_retrieval_fulfill)
        
        logger.info(
            f"Content economy started (royalty_model={self.royalty_model.value}, "
            f"min_replicas={self.min_replicas})"
        )
    
    def set_escrow(self, escrow: Any) -> None:
        """Set the multi-party escrow for batch settlements."""
        self._escrow = escrow
    
    async def stop(self) -> None:
        """Stop background tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        
    # ── Content Access Payment ─────────────────────────────────────────────
    
    async def process_content_access(
        self,
        content_id: str,
        accessor_id: str,
        content_metadata: Dict[str, Any],
    ) -> ContentAccessPayment:
        """Process FTNS payment for content access.

        Flow:
        1. Calculate total payment based on royalty_rate
        2. Lock FTNS in escrow (on-chain if available)
        3. Resolve provenance chain (original + derivatives)
        4. Distribute royalties according to model
        5. Record payment completion

        Args:
            content_id: Content identifier
            accessor_id: Node/user accessing the content
            content_metadata: Metadata including royalty_rate, creator_id, parent_cids

        Returns:
            ContentAccessPayment with status and distribution details
        """
        payment_id = f"pay-{uuid.uuid4().hex[:12]}"

        # Get royalty rate from metadata or default
        royalty_rate = content_metadata.get("royalty_rate", 0.01)
        # sp1077 (content-data-plane Gap B) — the advertise-lane royalty_rate is
        # forgeable (unauthenticated, first-writer-wins). For content with a REGISTERED
        # provenance_hash, override it with the on-chain-authoritative rate so a
        # malicious advertiser can't skew off-chain credit (pool-split weighting + the
        # payment amount). Unregistered/no-client → the bounded advertise value stands.
        _auth_rate = await self._authenticated_royalty_rate(content_metadata)
        if _auth_rate is not None:
            if _auth_rate != royalty_rate:
                logger.info(
                    "content %s: using on-chain-registered royalty_rate %.4f "
                    "(advertise lane claimed %.4f)",
                    content_id[:12], _auth_rate, royalty_rate)
            royalty_rate = _auth_rate
        creator_id = content_metadata.get("creator_id", "")
        parent_cids = content_metadata.get("parent_cids", [])

        # Calculate total payment amount
        # For Phase4: base access fee is the royalty_rate
        # For Legacy: royalty_rate is already the per-access fee
        total_amount = Decimal(str(royalty_rate))

        payment = ContentAccessPayment(
            payment_id=payment_id,
            content_id=content_id,
            accessor_id=accessor_id,
            creator_id=creator_id,
            amount=total_amount,
            royalty_model=self.royalty_model,
            status=PaymentStatus.PENDING,
        )

        self._pending_payments[payment_id] = payment

        try:
            # Step 1: Lock FTNS in escrow (on-chain) or debit local ledger
            if (
                self.ftns_ledger
                and accessor_id == self.identity.node_id
                and hasattr(self.ftns_ledger, "lock_escrow")
            ):
                # On-chain escrow for our own accesses
                escrow_result = await self.ftns_ledger.lock_escrow(
                    amount=float(total_amount),
                    purpose=f"content_access:{content_id}",
                )
                if escrow_result.get("success"):
                    payment.escrow_tx_hash = escrow_result.get("tx_hash")
                    payment.status = PaymentStatus.ESCROWED
                else:
                    # Fall back to local ledger debit
                    await self.ledger.debit(
                        wallet_id=accessor_id,
                        amount=float(total_amount),
                        tx_type=TransactionType.COMPUTE_PAYMENT,
                        description=f"Content access: {content_id[:12]}...",
                    )
            else:
                # Local ledger for cross-node or non-on-chain payments
                if accessor_id == self.identity.node_id:
                    await self.ledger.debit(
                        wallet_id=accessor_id,
                        amount=float(total_amount),
                        tx_type=TransactionType.COMPUTE_PAYMENT,
                        description=f"Content access: {content_id[:12]}...",
                    )
                # Remote accessors pay via their own node - we just track

            # Step 2: Distribute royalties
            distributions = await self._distribute_royalties(
                payment=payment,
                creator_id=creator_id,
                parent_cids=parent_cids,
                content_metadata=content_metadata,
            )
            payment.royalty_distributions = distributions
            
            # Step 3: Release escrow (if on-chain)
            if payment.escrow_tx_hash and self.ftns_ledger:
                # Sum of distributed amounts goes to creators
                total_distributed = sum(d["amount"] for d in distributions)
                await self.ftns_ledger.release_escrow(
                    escrow_id=payment.escrow_tx_hash,
                    recipient=creator_id,  # Simplified - real impl would multi-sig
                    amount=float(total_distributed),
                )
            
            # Phase 1.2: if any distribution was a broadcast_pending stub from
            # the on-chain branch, the chain tx may still settle. Surface that
            # to the API as PENDING_ONCHAIN so callers know to reconcile
            # manually rather than treating the payment as final.
            if any(
                d.get("type") == "broadcast_pending"
                for d in payment.royalty_distributions
            ):
                payment.status = PaymentStatus.PENDING_ONCHAIN
            else:
                payment.status = PaymentStatus.COMPLETED
            payment.completed_at = time.time()

            logger.info(
                f"Content access payment {payment.status.value}: {payment_id} "
                f"({total_amount} FTNS for {content_id[:12]}... by {accessor_id[:8]})"
            )
            
        except Exception as e:
            payment.status = PaymentStatus.FAILED
            payment.error = str(e)
            logger.error(f"Content access payment failed: {payment_id} - {e}")
            
            # Refund if escrowed
            if payment.escrow_tx_hash and self.ftns_ledger:
                try:
                    await self.ftns_ledger.release_escrow(
                        escrow_id=payment.escrow_tx_hash,
                        recipient=accessor_id,
                        amount=float(total_amount),
                    )
                    payment.status = PaymentStatus.REFUNDED
                except Exception:
                    pass
        
        return payment
    
    # ── On-chain provenance clients (Phase 1, lazy) ────────────────────────

    def _get_provenance_client(self):
        """Lazy-init ProvenanceRegistryClient. Returns None if disabled."""
        if not ONCHAIN_PROVENANCE_ENABLED:
            return None
        if not PROVENANCE_REGISTRY_ADDRESS:
            logger.warning(
                "PRSM_ONCHAIN_PROVENANCE=1 but PRSM_PROVENANCE_REGISTRY_ADDRESS not set"
            )
            return None
        if getattr(self, "_provenance_client", None) is None:
            try:
                from prsm.economy.web3.provenance_registry import (
                    ProvenanceRegistryClient,
                )
                rpc_url = _endpoints.rpc_url
                pk = os.getenv("FTNS_WALLET_PRIVATE_KEY")
                self._provenance_client = ProvenanceRegistryClient(
                    rpc_url=rpc_url,
                    contract_address=PROVENANCE_REGISTRY_ADDRESS,
                    private_key=pk,
                )
            except Exception as exc:
                logger.error(f"failed to init ProvenanceRegistryClient: {exc}")
                self._provenance_client = None
        return self._provenance_client

    _AUTH_RATE_CACHE_CAP = 4096

    async def _authenticated_provenance_record(self, content_metadata):
        """sp1077/1078 (content-data-plane Gap B) — the on-chain ProvenanceRegistry
        record for content with a registered provenance_hash, or None. The shared,
        cached AUTHORITATIVE source for the off-chain rate (sp1077) + creator (sp1078)
        authentication (the same source the on-chain royalty leg dispatches on, sp996).
        Returns None when there's no provenance_hash, no provenance client, the hash is
        unregistered, or the lookup fails. Cached per content_hash (bounded), so the hot
        retrieve path does at most one RPC per content (None cached too — a later
        registration is picked up after eviction/restart, a minor freshness tradeoff)."""
        import asyncio
        ph = content_metadata.get("provenance_hash") if content_metadata else None
        if not ph:
            return None
        try:
            content_hash = bytes.fromhex(str(ph).removeprefix("0x"))
        except (ValueError, AttributeError):
            return None
        if len(content_hash) != 32:
            return None

        cache = getattr(self, "_auth_prov_cache", None)
        if cache is None:
            cache = self._auth_prov_cache = {}
        if content_hash in cache:
            return cache[content_hash]

        record = None
        client = self._get_provenance_client()
        if client is not None:
            try:
                record = await asyncio.to_thread(client.get_content, content_hash)
            except Exception as exc:  # noqa: BLE001 - lookup failure → defer to advertise
                logger.debug("provenance get_content failed for %s: %s",
                             content_hash.hex()[:12], exc)
                record = None

        if len(cache) >= self._AUTH_RATE_CACHE_CAP:   # bounded (evict oldest)
            try:
                cache.pop(next(iter(cache)))
            except StopIteration:
                pass
        cache[content_hash] = record
        return record

    async def _authenticated_royalty_rate(self, content_metadata):
        """sp1077 — the AUTHORITATIVE royalty rate (royaltyRateBps/10000) for content
        with a registered provenance_hash, or None (caller keeps the sp1004-bounded
        advertise value). Stops a forged advertise rate from skewing off-chain credit
        for REGISTERED content."""
        record = await self._authenticated_provenance_record(content_metadata)
        if record is None:
            return None
        bps = int(record.royalty_rate_bps)
        # MAX_ROYALTY_RATE_BPS = 9800; reject out-of-range rather than trust a corrupt read.
        if 0 <= bps <= 9800:
            return bps / 10000.0
        return None

    async def _authenticated_creator(self, content_metadata):
        """sp1078 (Gap B creator leg) — the AUTHORITATIVE creator eth address (the
        registry's `creator`) for content with a registered provenance_hash, or None
        (caller keeps the forgeable advertise/own value). Used to override the
        advertise-lane creator_eth_address before crediting reputation, so a malicious
        advertiser can't poison the §14 reputation of a creator they don't control."""
        record = await self._authenticated_provenance_record(content_metadata)
        if record is None:
            return None
        creator = getattr(record, "creator", None)
        if isinstance(creator, str) and creator.startswith("0x") and len(creator) == 42:
            return creator
        return None

    def _get_royalty_distributor(self):
        """Lazy-init RoyaltyDistributorClient. Returns None if disabled."""
        if not ONCHAIN_PROVENANCE_ENABLED:
            return None
        if not ROYALTY_DISTRIBUTOR_ADDRESS:
            logger.warning(
                "PRSM_ONCHAIN_PROVENANCE=1 but PRSM_ROYALTY_DISTRIBUTOR_ADDRESS not set"
            )
            return None
        if getattr(self, "_royalty_client", None) is None:
            try:
                from prsm.economy.web3.royalty_distributor import (
                    RoyaltyDistributorClient,
                )
                rpc_url = _endpoints.rpc_url
                pk = os.getenv("FTNS_WALLET_PRIVATE_KEY")
                ftns_addr = os.getenv("FTNS_TOKEN_ADDRESS", DEFAULT_FTNS_TOKEN_ADDRESS)
                self._royalty_client = RoyaltyDistributorClient(
                    rpc_url=rpc_url,
                    distributor_address=ROYALTY_DISTRIBUTOR_ADDRESS,
                    ftns_token_address=ftns_addr,
                    private_key=pk,
                )
            except Exception as exc:
                logger.error(f"failed to init RoyaltyDistributorClient: {exc}")
                self._royalty_client = None
        return self._royalty_client

    def _serving_node_address(self) -> Optional[str]:
        """Return this node's on-chain 0x address if available."""
        if self.ftns_ledger and getattr(self.ftns_ledger, "_connected_address", None):
            return self.ftns_ledger._connected_address
        return None

    async def _try_onchain_distribute(
        self,
        payment: "ContentAccessPayment",
        content_metadata: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        """Attempt on-chain royalty distribution.

        Returns:
          - distributions list when on-chain settlement is final OR in flight
            (caller MUST NOT fall back to local payment).
          - None when no on-chain attempt was made or the attempt failed
            before broadcast (caller may safely fall back).

        Phase 1.1 Task 6: reads `provenance_hash` from content_metadata
        directly, replacing the broken `sha3_256(content_id_string)` path
        that never matched what the CLI registered. Phase 1.1 Task 4 error
        types make the broadcast/settle distinction explicit so a chain
        outage cannot trigger a double payment.
        """
        distributor = self._get_royalty_distributor()
        if distributor is None:
            return None

        serving_node_addr = self._serving_node_address()
        if not serving_node_addr:
            logger.debug("on-chain distribute skipped: no serving-node 0x address")
            return None

        # Read the canonical hash directly from upload metadata.
        provenance_hash = content_metadata.get("provenance_hash")
        if not provenance_hash:
            logger.debug(
                f"on-chain distribute skipped: content {payment.content_id[:12]}… "
                f"has no provenance_hash in metadata (use `prsm provenance register`)"
            )
            return None
        # Parse the metadata-supplied hash. A malformed value is a metadata
        # bug, not a payment failure — fall back to local instead of marking
        # the whole payment FAILED.
        try:
            if isinstance(provenance_hash, str):
                provenance_hash = bytes.fromhex(
                    provenance_hash[2:]
                    if provenance_hash.startswith("0x")
                    else provenance_hash
                )
            if not isinstance(provenance_hash, (bytes, bytearray)):
                raise ValueError("provenance_hash must be bytes or hex string")
            if len(provenance_hash) != 32:
                raise ValueError(
                    f"provenance_hash must be 32 bytes (got {len(provenance_hash)})"
                )
            provenance_hash = bytes(provenance_hash)
        except (ValueError, TypeError) as exc:
            logger.warning(
                f"on-chain distribute skipped: malformed provenance_hash for "
                f"{payment.content_id[:12]}…: {exc}"
            )
            return None

        gross_wei = int(Decimal(str(payment.amount)) * Decimal(10**18))
        if gross_wei <= 0:
            return None

        # Late imports so the symbols stay local to the on-chain branch.
        from prsm.economy.web3.royalty_distributor import (
            BroadcastFailedError,
            OnChainPendingError,
            OnChainRevertedError,
        )

        try:
            preview = await asyncio.to_thread(
                distributor.preview_split, provenance_hash, gross_wei
            )
            tx_hash, status = await asyncio.to_thread(
                distributor.distribute_royalty,
                provenance_hash,
                serving_node_addr,
                gross_wei,
            )
        except BroadcastFailedError as exc:
            logger.error(
                f"on-chain distribute pre-broadcast failure (safe fallback): {exc}"
            )
            return None
        except OnChainRevertedError as exc:
            logger.error(
                f"on-chain distribute reverted (safe fallback): {exc}"
            )
            return None
        except OnChainPendingError as exc:
            logger.error(
                f"on-chain distribute broadcast OK but receipt unknown — "
                f"NOT falling back. tx_hash={exc.tx_hash}. "
                f"Operator must reconcile manually."
            )
            # Return a single in-flight stub so the caller does NOT trigger
            # local fallback. This is the key fix for the double-pay race.
            return [
                {
                    "recipient_id": "onchain:in_flight",
                    "amount": float(payment.amount),
                    "type": "broadcast_pending",
                    "tx_hash": exc.tx_hash,
                }
            ]
        except Exception as exc:
            logger.error(
                f"on-chain distribute unexpected error (safe fallback): {exc}"
            )
            return None

        logger.info(
            f"on-chain royalty paid: hash={provenance_hash.hex()[:12]}… "
            f"gross={payment.amount} status={status.value} tx={tx_hash[:16]}…"
        )
        return [
            {
                "recipient_id": "onchain:creator",
                "amount": preview.creator_amount / 10**18,
                "type": "original_creator",
                "tx_hash": tx_hash,
            },
            {
                "recipient_id": "onchain:network",
                "amount": preview.network_amount / 10**18,
                "type": "network_fee",
                "tx_hash": tx_hash,
            },
            {
                "recipient_id": serving_node_addr,
                "amount": preview.serving_node_amount / 10**18,
                "type": "serving_node",
                "tx_hash": tx_hash,
            },
        ]

    async def _distribute_royalties(
        self,
        payment: ContentAccessPayment,
        creator_id: str,
        parent_cids: List[str],
        content_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Distribute royalties according to the configured model.

        Phase4 Model:
        - 8% to original creator (first in chain)
        - 1% to each derivative creator (up to MAX_ROYALTY_CHAIN_DEPTH)
        - 2% network fee
        - Remaining to serving node (Phase 1.1 fix; was direct_creator)

        Legacy Model:
        - 70% to derivative creator
        - 25% split among source creators
        - 5% network fee

        Phase 1.1 (on-chain): when PRSM_ONCHAIN_PROVENANCE=1 and
        content_metadata contains a `provenance_hash`, distribution
        happens via RoyaltyDistributor on Base mainnet. Pre-broadcast
        and reverted failures fall through to the local path so
        payments are never lost. Post-broadcast unknown short-circuits
        with an in-flight stub to prevent double payment.
        """
        # Phase 1.1: pass metadata so the on-chain branch can read provenance_hash.
        onchain = await self._try_onchain_distribute(payment, content_metadata or {})
        if onchain is not None:
            return onchain

        distributions = []
        total_amount = payment.amount

        if self.royalty_model == RoyaltyModel.PHASE4:
            # Resolve provenance chain
            provenance_chain = await self._resolve_provenance_chain(
                content_id=payment.content_id,
                parent_cids=parent_cids,
            )

            # Original creator gets 8%
            # Phase 1.3 Task 3g pass-6: the empty-string filter for
            # original_creator is load-bearing — _resolve_provenance_chain
            # may return chain.original_creator="" if all ancestors in the
            # chain were known only from minimal ads. Don't remove this check.
            if provenance_chain.original_creator:
                original_share = total_amount * Decimal(str(ORIGINAL_CREATOR_ROYALTY_RATE))
                distributions.append({
                    "recipient_id": provenance_chain.original_creator,
                    "amount": float(original_share),
                    "type": "original_creator",
                    "content_id": provenance_chain.original_content_id,
                })
                await self._credit_royalty(
                    recipient_id=provenance_chain.original_creator,
                    amount=float(original_share),
                    content_id=payment.content_id,
                    description=f"Original creator royalty (8%): {payment.content_id[:12]}...",
                )

            # Derivative creators get 1% each
            for i, derivative in enumerate(provenance_chain.derivative_creators[:MAX_ROYALTY_CHAIN_DEPTH]):
                derivative_creator_id = derivative.get("creator_id", "")
                if not derivative_creator_id:
                    # Phase 1.3 Task 3g pass-6: defense-in-depth. The
                    # source-of-truth filter lives in
                    # _resolve_provenance_chain, but a stale chain built
                    # before the fix (or a bug-resurfaced later) would
                    # otherwise silently credit an empty wallet.
                    continue
                derivative_share = total_amount * Decimal(str(DERIVATIVE_CREATOR_ROYALTY_RATE))
                distributions.append({
                    "recipient_id": derivative["creator_id"],
                    "amount": float(derivative_share),
                    "type": "derivative_creator",
                    "depth": derivative["depth"],
                    "content_id": derivative["content_id"],
                })
                await self._credit_royalty(
                    recipient_id=derivative["creator_id"],
                    amount=float(derivative_share),
                    content_id=payment.content_id,
                    description=f"Derivative royalty (1%, depth={derivative['depth']}): {payment.content_id[:12]}...",
                )

            # Network fee
            network_fee = total_amount * Decimal(str(NETWORK_FEE_RATE))
            distributions.append({
                "recipient_id": "system",
                "amount": float(network_fee),
                "type": "network_fee",
            })
            await self.ledger.credit(
                wallet_id="system",
                amount=float(network_fee),
                tx_type=TransactionType.CONTENT_ROYALTY,
                description=f"Network fee: {payment.content_id[:12]}...",
            )

            # Phase 1.1 Task 7: serving node gets the remainder, mirroring
            # the on-chain 8/2/90 split. Pre-Phase-1.1, this remainder went
            # to the direct creator, which silently under-paid the serving
            # node by ~90% on every payment that fell back to local.
            distributed = sum(d["amount"] for d in distributions)
            remainder = float(total_amount) - distributed
            if remainder > 0:
                serving_node_id = self.identity.node_id
                distributions.append({
                    "recipient_id": serving_node_id,
                    "amount": remainder,
                    "type": "serving_node",
                })
                await self._credit_royalty(
                    recipient_id=serving_node_id,
                    amount=remainder,
                    content_id=payment.content_id,
                    description=f"Serving node share: {payment.content_id[:12]}...",
                )

        else:  # Legacy model
            if parent_cids:
                # Derivative work - split royalties
                derivative_share = total_amount * Decimal(str(LEGACY_DERIVATIVE_SHARE))
                source_pool = total_amount * Decimal(str(LEGACY_SOURCE_SHARE))
                network_fee = total_amount * Decimal(str(LEGACY_NETWORK_SHARE))

                # Credit derivative creator (this node)
                distributions.append({
                    "recipient_id": creator_id,
                    "amount": float(derivative_share),
                    "type": "derivative_creator",
                })
                await self._credit_royalty(
                    recipient_id=creator_id,
                    amount=float(derivative_share),
                    content_id=payment.content_id,
                    description=f"Derivative royalty (70%): {payment.content_id[:12]}...",
                )

                # Split source pool among parent creators
                parent_creators = await self._resolve_parent_creators(parent_cids)
                if parent_creators:
                    per_parent = float(source_pool) / len(parent_creators)
                    for parent_creator_id in parent_creators:
                        distributions.append({
                            "recipient_id": parent_creator_id,
                            "amount": per_parent,
                            "type": "source_creator",
                        })
                        await self._credit_royalty(
                            recipient_id=parent_creator_id,
                            amount=per_parent,
                            content_id=payment.content_id,
                            description=f"Source royalty: {payment.content_id[:12]}...",
                        )
                else:
                    # No resolvable parents - derivative creator gets source pool too
                    distributions.append({
                        "recipient_id": creator_id,
                        "amount": float(source_pool),
                        "type": "unclaimed_source",
                    })
                    await self._credit_royalty(
                        recipient_id=creator_id,
                        amount=float(source_pool),
                        content_id=payment.content_id,
                        description=f"Unclaimed source pool: {payment.content_id[:12]}...",
                    )

                # Network fee
                distributions.append({
                    "recipient_id": "system",
                    "amount": float(network_fee),
                    "type": "network_fee",
                })
                await self.ledger.credit(
                    wallet_id="system",
                    amount=float(network_fee),
                    tx_type=TransactionType.CONTENT_ROYALTY,
                    description=f"Network fee: {payment.content_id[:12]}...",
                )
            else:
                # Original work - full royalty to creator
                distributions.append({
                    "recipient_id": creator_id,
                    "amount": float(total_amount),
                    "type": "original_creator",
                })
                await self._credit_royalty(
                    recipient_id=creator_id,
                    amount=float(total_amount),
                    content_id=payment.content_id,
                    description=f"Content royalty: {payment.content_id[:12]}...",
                )

        # Accumulate to escrow for batch on-chain settlement (Phase 4)
        if self._escrow:
            try:
                for dist in distributions:
                    if dist["recipient_id"] != "system":
                        await self._escrow.accumulate(
                            creator_id=dist["recipient_id"],
                            amount=dist["amount"],
                            source_content_id=payment.content_id,
                            accessor_id=payment.accessor_id,
                        )
            except Exception as e:
                logger.debug(f"Escrow accumulation failed: {e}")
        
        return distributions
    
    async def _credit_royalty(
        self,
        recipient_id: str,
        amount: float,
        content_id: str,
        description: str,
    ) -> None:
        """Credit royalty to a recipient via local ledger and optionally on-chain."""
        # Local ledger credit
        await self.ledger.credit(
            wallet_id=recipient_id,
            amount=amount,
            tx_type=TransactionType.CONTENT_ROYALTY,
            description=description,
        )
        
        # On-chain transfer if available and we know recipient's address
        # (simplified - real impl would need address resolution)
        if self.ftns_ledger and recipient_id == self.identity.node_id:
            # Self-royalty already credited locally
            pass
        # For remote recipients, they'll receive via their own ledger sync
    
    async def _resolve_provenance_chain(
        self,
        content_id: str,
        parent_cids: List[str],
    ) -> "ProvenanceChain":
        """Resolve the full provenance chain for royalty distribution."""
        chain = ProvenanceChain()

        # Track original creator (deepest ancestor)
        visited: Set[str] = set()

        async def trace_ancestors(current_cid: str, depth: int) -> None:
            if current_cid in visited or depth > MAX_ROYALTY_CHAIN_DEPTH:
                return
            visited.add(current_cid)

            record = self.content_index.lookup(current_cid)
            if not record:
                return

            if depth == 0 and not parent_cids:
                # This is the content itself - the direct creator
                chain.direct_creator = record.creator_id or ""

            # Phase 1.3 Task 3g pass-6: read ContentRecord's parent_cids
            # field defensively. The original code used a different name
            # (since canonicalized in Part D) and was masked by MagicMock
            # auto-attribute behavior (MagicMock returns a truthy child
            # mock for any attribute, which hid an AttributeError in
            # production). Using getattr keeps the defensive read.
            record_parents = getattr(record, "parent_cids", None) or []
            if record_parents:
                for parent_cid in record_parents:
                    parent_record = self.content_index.lookup(parent_cid)
                    if parent_record:
                        if depth == 0:
                            # First level parent - derivative relationship.
                            # Phase 1.3 Task 3g pass-6: skip entries with an
                            # empty creator_id. After Task 3g pass-5 made ""
                            # a valid ContentRecord.creator_id state (for
                            # records learned only from minimal replica
                            # ads), crediting empty-wallet recipients would
                            # lock derivative royalty shares in a phantom
                            # local-ledger row instead of paying the real
                            # creator.
                            if record.creator_id:
                                chain.derivative_creators.append({
                                    "creator_id": record.creator_id,
                                    "content_id": current_cid,
                                    "depth": depth + 1,
                                })
                        await trace_ancestors(parent_cid, depth + 1)
            else:
                # No parents = original creator
                if chain.original_creator is None and record.creator_id:
                    # Phase 1.3 Task 3g pass-6: only set original_creator
                    # when we know who it is. An empty string here would
                    # cause _distribute_royalties to credit an empty wallet
                    # for the 8% original-creator share.
                    chain.original_creator = record.creator_id
                    chain.original_content_id = current_cid

        # Start tracing from parents
        for parent_cid in parent_cids:
            await trace_ancestors(parent_cid, 0)

        # If no parents found, this is original content
        if not chain.original_creator:
            record = self.content_index.lookup(content_id)
            if record and record.creator_id:
                # Phase 1.3 Task 3g pass-6: only claim original creator
                # when the record has a real creator_id, not an
                # empty-string placeholder from a minimal ad.
                chain.original_creator = record.creator_id
                chain.original_content_id = content_id

        return chain

    async def _resolve_parent_creators(self, parent_cids: List[str]) -> List[str]:
        """Resolve creator IDs for parent content IDs."""
        creators = []
        for parent_cid in parent_cids:
            record = self.content_index.lookup(parent_cid)
            if record and record.creator_id and record.creator_id not in creators:
                creators.append(record.creator_id)
        return creators
    
    # ── Replication Management ─────────────────────────────────────────────
    
    async def track_content_upload(
        self,
        content_id: str,
        size_bytes: int,
        replicas_requested: int,
    ) -> ReplicationStatus:
        """Start tracking replication status for newly uploaded content."""
        status = ReplicationStatus(
            content_id=content_id,
            min_replicas=max(self.min_replicas, replicas_requested),
            current_replicas=1,  # We have it locally
            providers={self.identity.node_id},
        )
        self._replication_status[content_id] = status

        logger.info(
            f"Tracking replication for {content_id[:12]}... "
            f"(min={status.min_replicas}, current={status.current_replicas})"
        )

        return status

    async def update_replication_status(
        self,
        content_id: str,
        provider_id: str,
        has_content: bool,
    ) -> None:
        """Update replication status when a provider announces or removes content."""
        status = self._replication_status.get(content_id)
        if not status:
            # Auto-create tracking for known content
            record = self.content_index.lookup(content_id)
            if record:
                status = ReplicationStatus(
                    content_id=content_id,
                    min_replicas=self.min_replicas,
                    current_replicas=len(record.providers),
                    providers=record.providers.copy(),
                )
                self._replication_status[content_id] = status
            else:
                return

        if has_content:
            if provider_id not in status.providers:
                status.providers.add(provider_id)
                status.current_replicas = len(status.providers)
                status.last_verified = time.time()
        else:
            status.providers.discard(provider_id)
            status.current_replicas = len(status.providers)

        # Check if we need more replicas
        await self._check_replication_needs(content_id, status)

    async def _check_replication_needs(
        self,
        content_id: str,
        status: ReplicationStatus,
    ) -> None:
        """Request additional replicas if below minimum."""
        if status.current_replicas >= status.min_replicas:
            return

        needed = status.min_replicas - status.current_replicas
        if needed <= 0 or status.pending_requests >= needed:
            return

        # Request additional storage via gossip
        status.pending_requests += needed

        await self.gossip.publish("storage_request", {
            "content_id": content_id,
            "size_bytes": 0,  # Would need to look up
            "requester_id": self.identity.node_id,
            "replicas_needed": needed,
            "priority": "high" if status.current_replicas == 0 else "normal",
        })

        logger.info(
            f"Requested {needed} additional replicas for {content_id[:12]}... "
            f"(current={status.current_replicas}, min={status.min_replicas})"
        )

    async def _replication_monitor_loop(self) -> None:
        """Periodically check replication status and request more replicas if needed."""
        while self._running:
            await asyncio.sleep(self.replication_check_interval)

            try:
                for content_id, status in list(self._replication_status.items()):
                    # Decay pending requests over time
                    if status.pending_requests > 0:
                        status.pending_requests = max(0, status.pending_requests - 1)

                    await self._check_replication_needs(content_id, status)
                    
            except Exception as e:
                logger.error(f"Replication monitor error: {e}")
    
    # ── Content Retrieval Marketplace ───────────────────────────────────────
    
    async def request_content_retrieval(
        self,
        content_id: str,
        max_price_ftns: Decimal,
        timeout: float = 30.0,
    ) -> Optional[bytes]:
        """Request content from the network with marketplace bidding.

        Flow:
        1. Broadcast retrieval request with max price
        2. Collect bids from providers
        3. Select best provider (price + reputation + latency)
        4. Pay and retrieve content

        Args:
            content_id: Content identifier to retrieve
            max_price_ftns: Maximum willing to pay
            timeout: Seconds to wait for bids

        Returns:
            Content bytes, or None if not available
        """
        request_id = f"ret-{uuid.uuid4().hex[:12]}"

        request = RetrievalRequest(
            request_id=request_id,
            content_id=content_id,
            requester_id=self.identity.node_id,
            max_price_ftns=max_price_ftns,
            bid_deadline=time.time() + min(timeout / 3, 10.0),
        )
        self._retrieval_requests[request_id] = request

        # Broadcast retrieval request
        await self.gossip.publish("retrieval_request", {
            "request_id": request_id,
            "content_id": content_id,
            "requester_id": self.identity.node_id,
            "max_price_ftns": float(max_price_ftns),
            "deadline": request.bid_deadline,
        })
        
        # Wait for bids
        await asyncio.sleep(min(timeout / 3, 10.0))
        
        if not request.bids:
            logger.debug(f"No bids received for retrieval request {request_id}")
            request.status = "failed"
            return None
        
        # Select best provider
        selected_bid = self._select_best_bid(request.bids, max_price_ftns)
        if not selected_bid:
            logger.debug(f"No suitable bids for retrieval request {request_id}")
            request.status = "failed"
            return None
        
        request.selected_provider = selected_bid.provider_id
        request.status = "fulfilled"

        # Phase 1.2: look up canonical provenance_hash and the true
        # original creator from the content_index so the marketplace
        # retrieval path can actually route through the on-chain
        # RoyaltyDistributor. Before this lookup the metadata only had
        # the selected provider_id as creator_id, which disagrees with
        # the real original creator and always falls back to local.
        provenance_hash: Optional[str] = None
        index_creator_id: Optional[str] = None
        index_parents: List[str] = []
        if self.content_index is not None:
            lookup = getattr(self.content_index, "lookup", None)
            if callable(lookup):
                record = lookup(content_id)
                if record is not None:
                    provenance_hash = getattr(record, "provenance_hash", None)
                    index_creator_id = getattr(record, "creator_id", None)
                    index_parents = list(getattr(record, "parent_cids", []) or [])

        # Process payment
        payment = await self.process_content_access(
            content_id=content_id,
            accessor_id=self.identity.node_id,
            content_metadata={
                "royalty_rate": float(selected_bid.price_ftns),
                "creator_id": index_creator_id or selected_bid.provider_id,
                "parent_cids": index_parents,
                "provenance_hash": provenance_hash,
            },
        )
        
        # Phase 1.2: PENDING_ONCHAIN means the on-chain broadcast succeeded
        # but the receipt is unknown — the tx may still settle. Treat it as
        # a valid "proceed" state for retrieval (the chain will reconcile),
        # same as COMPLETED. Only hard failures return None.
        if payment.status not in (
            PaymentStatus.COMPLETED,
            PaymentStatus.PENDING_ONCHAIN,
        ):
            logger.error(f"Payment failed for retrieval {request_id}: {payment.error}")
            return None
        if payment.status == PaymentStatus.PENDING_ONCHAIN:
            logger.warning(
                f"Retrieval {request_id} proceeding with in-flight on-chain "
                f"payment (tx pending reconciliation)"
            )
        
        # Request content from selected provider
        # (ContentProvider will handle the actual retrieval)
        return None  # Actual retrieval handled by ContentProvider
    
    def _select_best_bid(
        self,
        bids: List[ProviderBid],
        max_price: Decimal,
    ) -> Optional[ProviderBid]:
        """Select the best bid based on price, reputation, and latency."""
        valid_bids = [
            b for b in bids
            if Decimal(str(b.price_ftns)) <= max_price
            and b.expires_at > time.time()
        ]
        
        if not valid_bids:
            return None
        
        # Score: lower price, higher reputation, lower latency
        def score(bid: ProviderBid) -> float:
            price_score = float(max_price - Decimal(str(bid.price_ftns))) / float(max_price) if max_price > 0 else 0
            rep_score = bid.reputation_score
            latency_score = max(0, 1 - (bid.estimated_latency_ms / 1000.0))  # Normalize to 0-1
            return price_score * 0.5 + rep_score * 0.3 + latency_score * 0.2
        
        return max(valid_bids, key=score)
    
    async def _on_retrieval_bid(
        self,
        subtype: str,
        data: Dict[str, Any],
        origin: str,
    ) -> None:
        """Handle a bid for a retrieval request."""
        request_id = data.get("request_id")
        if not request_id:
            return
        
        request = self._retrieval_requests.get(request_id)
        if not request or request.status != "open":
            return
        
        bid = ProviderBid(
            provider_id=origin,
            content_id=request.content_id,
            price_ftns=Decimal(str(data.get("price_ftns", 0))),
            estimated_latency_ms=data.get("estimated_latency_ms", 100.0),
            available_bandwidth_mbps=data.get("available_bandwidth_mbps", 100.0),
            reputation_score=self._provider_reputation.get(origin, 0.5),
            expires_at=data.get("expires_at", time.time() + 60.0),
        )
        
        request.bids.append(bid)
        logger.debug(f"Received bid from {origin[:8]} for request {request_id}")
    
    async def _on_retrieval_fulfill(
        self,
        subtype: str,
        data: Dict[str, Any],
        origin: str,
    ) -> None:
        """Handle content fulfillment from a provider."""
        request_id = data.get("request_id")
        # Actual content delivery handled by ContentProvider
        logger.debug(f"Retrieval fulfillment from {origin[:8]} for {request_id}")
    
    async def _bid_cleanup_loop(self) -> None:
        """Clean up expired retrieval requests."""
        while self._running:
            await asyncio.sleep(60.0)
            
            now = time.time()
            expired = [
                rid for rid, req in self._retrieval_requests.items()
                if req.bid_deadline < now and req.status == "open"
            ]
            
            for rid in expired:
                self._retrieval_requests[rid].status = "expired"
                del self._retrieval_requests[rid]
    
    # ── Vector DB Integration ───────────────────────────────────────────────
    
    async def index_content_embedding(
        self,
        content_id: str,
        content: bytes,
        metadata: Dict[str, Any],
    ) -> bool:
        """Index content into vector store for semantic search.

        Args:
            content_id: Content identifier
            content: Raw content bytes
            metadata: Content metadata (creator_id, royalty_rate, etc.)

        Returns:
            True if indexed successfully
        """
        if not self.vector_store or not self.embedding_fn:
            return False

        try:
            # Generate embedding
            text = content.decode("utf-8", errors="ignore").strip()
            if len(text) < 50:
                return False  # Too short for meaningful embedding

            embedding = await self.embedding_fn(text[:32_000])
            if embedding is None:
                return False

            # Store in vector DB
            await self.vector_store.upsert(
                content_id=content_id,
                embedding=embedding,
                metadata={
                    "creator_id": metadata.get("creator_id", ""),
                    "royalty_rate": metadata.get("royalty_rate", 0.01),
                    "content_type": metadata.get("content_type", "text"),
                    "filename": metadata.get("filename", ""),
                    **metadata,
                },
            )

            logger.debug(f"Indexed embedding for {content_id[:12]}...")
            return True

        except Exception as e:
            logger.warning(f"Failed to index embedding for {content_id[:12]}...: {e}")
            return False
    
    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.7,
        max_royalty_rate: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search content by semantic similarity.
        
        Args:
            query: Search query text
            limit: Maximum results to return
            min_similarity: Minimum similarity score
            max_royalty_rate: Optional maximum royalty rate filter
            
        Returns:
            List of content matches with metadata
        """
        if not self.vector_store or not self.embedding_fn:
            return []
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_fn(query)
            if query_embedding is None:
                return []
            
            # Search vector store
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                limit=limit,
                filters={
                    "min_similarity": min_similarity,
                    "max_royalty_rate": max_royalty_rate,
                } if max_royalty_rate else None,
            )
            
            return [
                {
                    "cid": r.content_cid,
                    "similarity": r.similarity_score,
                    "creator_id": r.creator_id,
                    "royalty_rate": r.royalty_rate,
                    "metadata": r.metadata,
                }
                for r in results
                if r.similarity_score >= min_similarity
            ]
            
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return []
    
    # ── Statistics ──────────────────────────────────────────────────────────
    
    def get_stats(self) -> Dict[str, Any]:
        """Get content economy statistics."""
        return {
            "pending_payments": len(self._pending_payments),
            "tracked_content": len(self._replication_status),
            "active_retrieval_requests": len([
                r for r in self._retrieval_requests.values()
                if r.status == "open"
            ]),
            "royalty_model": self.royalty_model.value,
            "min_replicas": self.min_replicas,
            "vector_store_enabled": self.vector_store is not None,
        }


@dataclass
class ProvenanceChain:
    """Resolved provenance chain for royalty distribution."""
    original_creator: Optional[str] = None
    original_content_id: Optional[str] = None
    derivative_creators: List[Dict[str, Any]] = field(default_factory=list)
    direct_creator: Optional[str] = None
