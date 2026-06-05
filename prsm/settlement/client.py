"""Phase 3.1 Task 6: BatchSettlementClient.

Off-chain driver that wraps:
  - ReceiptAccumulator (Task 4) — per-(requester, provider) queuing
  - Merkle tree + canonical encoding (Task 5) — Python↔Solidity parity
  - SettlementContractClient — abstracted on-chain surface

to produce the full commit → finalize lifecycle for a provider's
batched receipts.

The client is stateless about commit outcomes in two senses:
  1. If commit fails, receipts stay in the accumulator (caller retries
     via `commit_ready_batches` on the next poll).
  2. Finalization is driven by on-chain state (isFinalizable), not local
     timers — robust against clock drift between the provider host and
     Base's block timestamp.

Production deployment wires `SettlementContractClient` to web3.py's
async API against the deployed BatchSettlementRegistry + EscrowPool
contracts. Unit tests substitute an AsyncMock implementing the same
Protocol — the client has no direct dependency on web3.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

from prsm.settlement.accumulator import (
    BatchedReceipt,
    ReadyBatch,
    ReceiptAccumulator,
    TriggerReason,
)
from prsm.settlement.merkle import (
    batched_receipt_to_leaf,
    build_tree_and_proofs,
    hash_leaf,
)

logger = logging.getLogger(__name__)


class SettlementContractClient(Protocol):
    """Abstract surface of the BatchSettlementRegistry + EscrowPool
    contracts. Production implementation wraps web3.py; unit tests
    substitute AsyncMock."""

    async def commit_batch(
        self,
        provider_address: str,
        requester_address: str,
        merkle_root: bytes,
        receipt_count: int,
        total_value_ftns: int,
        tier_slash_rate_bps: int,
        consensus_group_id: bytes,
        metadata_uri: str,
    ) -> Tuple[bytes, int]:
        """Submit commitBatch; return (batch_id, commit_timestamp_unix).

        provider_address is redundant in the real Solidity wrapper
        (the contract uses msg.sender as the provider), but is passed
        through for mock/test implementations and for audit logging.
        The implementation parses the BatchCommitted event from the
        transaction receipt and returns the emitted batch_id + timestamp.

        tier_slash_rate_bps (Phase 7): basis-point slash rate snapshot,
        0-10000. 0 means this batch's provider hasn't bonded stake, so
        no slashing on successful challenge.

        consensus_group_id (Phase 7.1x §8.7): 32-byte identifier binding
        this batch to a k-of-n consensus dispatch. All-zero bytes means
        "not a consensus batch" — only DOUBLE_SPEND / INVALID_SIGNATURE
        challenges apply; CONSENSUS_MISMATCH is not targetable.
        """
        ...

    async def is_finalizable(self, batch_id: bytes) -> bool:
        """Mirrors BatchSettlementRegistry.isFinalizable(batchId)."""
        ...

    async def finalize_batch(self, batch_id: bytes) -> None:
        """Submit finalizeBatch(batch_id). Raises on tx revert."""
        ...

    async def get_batch_status(self, batch_id: bytes) -> int:
        """Current on-chain BatchStatus enum value:
           0=NONEXISTENT, 1=PENDING, 2=FINALIZED, 3=VOIDED."""
        ...

    async def get_committed_batch_for_tx(
        self, tx_hash: str
    ) -> Optional[Tuple[bytes, int]]:
        """Recover (batch_id, commit_timestamp) from a commitBatch tx that was
        broadcast but whose receipt was lost (the broadcast-but-unconfirmed
        case). Returns the BatchCommitted (batch_id, commit_timestamp) iff that tx
        mined and emitted the event, else None (still pending / dropped / reverted).

        Used by ``reconcile_pending_commits`` to ADOPT a batch that actually
        landed rather than re-committing the same receipts (which mints a distinct
        on-chain batchId and double-settles)."""
        ...


@dataclass(frozen=True)
class CommittedBatch:
    """Local record of a batch this client committed on chain.

    Retained after finalization for historical/audit purposes and — more
    importantly — so the client still has the leaf hashes needed to
    generate challenge proofs during the window, if someone challenges
    one of the provider's own batches.
    """
    batch_id: bytes                         # 32 bytes
    tx_hash: str                            # hex-encoded
    provider_address: str
    requester_address: str
    merkle_root: bytes
    receipt_count: int
    total_value_ftns: int
    commit_timestamp: int
    leaf_hashes: Tuple[bytes, ...]          # tuple for immutability
    trigger_reason: TriggerReason


@dataclass(frozen=True)
class FinalizedBatch:
    """Return shape of finalize_ready_batches."""
    batch_id: bytes
    tx_submitted: bool                      # False if already finalized etc.


@dataclass(frozen=True)
class PendingCommit:
    """A commit whose tx was broadcast but whose receipt was lost (the
    broadcast-but-unconfirmed case). The receipts have been REMOVED from the
    accumulator (so they can never be re-committed → no double settlement) and are
    retained here for ``reconcile_pending_commits``: if the original tx actually
    landed it is adopted into the tracked set; otherwise it stays quarantined for
    operator attention (never auto-re-committed)."""
    accumulator_key: Tuple                  # the AccumulatorKey of the popped batch
    tx_hash: str
    merkle_root: bytes
    leaf_hashes: Tuple[bytes, ...]
    receipt_count: int
    total_value_ftns: int
    provider_address: str
    requester_address: str
    trigger_reason: TriggerReason


class BatchSettlementClient:
    """Drives the Phase 3.1 commit → finalize lifecycle for one provider.

    Typical deployment: one BatchSettlementClient per provider node.
    The MarketplaceOrchestrator (Task 7) calls `accumulate` after each
    successful dispatch; a background task periodically calls
    `commit_ready_batches` and `finalize_ready_batches`.
    """

    def __init__(
        self,
        accumulator: ReceiptAccumulator,
        contract_client: SettlementContractClient,
        provider_address: str,
    ):
        self._accumulator = accumulator
        self._contract = contract_client
        self._provider = provider_address
        self._tracked: Dict[bytes, CommittedBatch] = {}
        self._finalized_ids: set = set()
        # sp1022 — commits broadcast but unconfirmed, keyed by tx_hash. Quarantined
        # (removed from the accumulator) so they can't be re-committed; resolved by
        # reconcile_pending_commits.
        self._pending_commits: Dict[str, PendingCommit] = {}

    # ── Accumulation ─────────────────────────────────────────────

    async def accumulate(self, br: BatchedReceipt) -> None:
        """Record a receipt. Caller (MarketplaceOrchestrator or a
        provider-side ComputeProvider hook) invokes this after each
        successful Phase 3 dispatch.

        Safety: the client's bound address must match EITHER the
        provider (earning side) OR the requester (spending side) on
        the receipt. Provider-side operators bind `provider_address`
        to their own address + receive receipts where provider_address
        matches; requester-side operators (e.g., MarketplaceOrchestrator
        in Phase 3.1 Task 7) bind the same parameter to their own
        address + receive receipts where requester_address matches.
        Both modes catch wrong-client routing while allowing either
        architectural pattern.
        """
        if (
            br.provider_address != self._provider
            and br.requester_address != self._provider
        ):
            raise ValueError(
                f"BatchSettlementClient bound to {self._provider!r} "
                f"refusing receipt with provider={br.provider_address!r} "
                f"requester={br.requester_address!r} — neither matches "
                f"the client's bound address"
            )
        self._accumulator.add(br)

    # ── Commit path ──────────────────────────────────────────────

    async def commit_ready_batches(self) -> List[CommittedBatch]:
        """Poll the accumulator for batches that crossed a threshold;
        commit each one on chain. On commit failure the receipts stay
        in the accumulator (no data loss); caller retries next poll.

        Returns the list of successfully-committed CommittedBatch
        records in the order processed.
        """
        committed: List[CommittedBatch] = []
        for ready in self._accumulator.ready_batches():
            try:
                record = await self._commit_one(ready)
            except Exception as exc:
                tx_hash = getattr(exc, "tx_hash", None)
                if tx_hash is not None:
                    # Broadcast-but-unconfirmed (OnChainPendingError carries
                    # tx_hash): the tx MAY still mine. Retrying would re-commit the
                    # same receipts as a DISTINCT on-chain batchId (the registry
                    # has no content dedup) → double settlement. Quarantine
                    # instead: remove from the accumulator so it can NEVER be
                    # re-committed, and record it for reconcile_pending_commits.
                    self._quarantine_pending(ready, tx_hash)
                    logger.error(
                        f"batch commit for key {ready.key} broadcast but "
                        f"UNCONFIRMED (tx {tx_hash}); quarantined to avoid "
                        f"double settlement — will reconcile, not retry"
                    )
                else:
                    # BroadcastFailedError (chain saw nothing) / OnChainRevertedError
                    # (atomic rollback) — safe to retry; receipts stay in the
                    # accumulator (popped only on success below).
                    logger.warning(
                        f"batch commit failed for key {ready.key}: "
                        f"{type(exc).__name__}: {exc} — receipts retained "
                        f"in accumulator for retry"
                    )
                continue

            # Pop only on success so failed commits naturally retry.
            self._accumulator.pop_batch(ready.key)
            self._tracked[record.batch_id] = record
            committed.append(record)

        return committed

    def _build_leaves_root(self, ready: ReadyBatch) -> Tuple[List[bytes], bytes]:
        """Canonical leaf hashes + Merkle root for a ready batch (deterministic).
        Shared by _commit_one and the pending-quarantine path."""
        leaf_hashes = [
            hash_leaf(batched_receipt_to_leaf(br)) for br in ready.batch.receipts
        ]
        tree = build_tree_and_proofs(leaf_hashes)
        return leaf_hashes, tree.root

    def _quarantine_pending(self, ready: ReadyBatch, tx_hash: str) -> None:
        """Remove a broadcast-but-unconfirmed batch from the accumulator and stash
        it for reconciliation. Popping is what guarantees no double-commit."""
        requester_address, provider_address, _group_id, _slash_bps = ready.key
        leaf_hashes, root = self._build_leaves_root(ready)
        self._accumulator.pop_batch(ready.key)
        self._pending_commits[tx_hash] = PendingCommit(
            accumulator_key=ready.key,
            tx_hash=tx_hash,
            merkle_root=root,
            leaf_hashes=tuple(leaf_hashes),
            receipt_count=len(leaf_hashes),
            total_value_ftns=ready.batch.total_value_ftns,
            provider_address=provider_address,
            requester_address=requester_address,
            trigger_reason=ready.trigger,
        )

    async def _commit_one(self, ready: ReadyBatch) -> CommittedBatch:
        """Convert receipts to canonical leaves, build tree, submit
        commitBatch. On success, returns the CommittedBatch record.

        Architectural note: this client is side-agnostic (provider-side
        batching or requester-side auditing both route through the same
        _commit_one code path). The on-chain contract enforces commit
        authority via `msg.sender`; the Python client trusts the
        accumulator's keyed batches."""
        # AccumulatorKey is (requester, provider, group_id, slash_rate_bps).
        requester_address, provider_address, group_id, slash_rate_bps = ready.key

        leaf_hashes, root = self._build_leaves_root(ready)

        batch_id, commit_ts = await self._contract.commit_batch(
            provider_address=provider_address,
            requester_address=requester_address,
            merkle_root=root,
            receipt_count=len(leaf_hashes),
            total_value_ftns=ready.batch.total_value_ftns,
            tier_slash_rate_bps=slash_rate_bps,
            consensus_group_id=group_id,
            metadata_uri="",
        )

        return CommittedBatch(
            batch_id=batch_id,
            tx_hash="",  # contract client populates if needed; not used by logic
            provider_address=provider_address,
            requester_address=requester_address,
            merkle_root=root,
            receipt_count=len(leaf_hashes),
            total_value_ftns=ready.batch.total_value_ftns,
            commit_timestamp=commit_ts,
            leaf_hashes=tuple(leaf_hashes),
            trigger_reason=ready.trigger,
        )

    # ── Finalize path ────────────────────────────────────────────

    async def finalize_ready_batches(self) -> List[FinalizedBatch]:
        """For each locally-tracked batch whose on-chain challenge
        window has elapsed, submit finalizeBatch. Idempotent: batches
        already finalized are skipped silently."""
        finalized: List[FinalizedBatch] = []
        # Snapshot keys so we can mutate during iteration if needed.
        for batch_id in list(self._tracked.keys()):
            if batch_id in self._finalized_ids:
                continue

            try:
                eligible = await self._contract.is_finalizable(batch_id)
            except Exception as exc:
                logger.warning(
                    f"isFinalizable query failed for batch {batch_id.hex()[:12]}…: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            if not eligible:
                continue

            try:
                await self._contract.finalize_batch(batch_id)
            except Exception as exc:
                logger.warning(
                    f"finalizeBatch tx failed for batch {batch_id.hex()[:12]}…: "
                    f"{type(exc).__name__}: {exc} — will retry next poll"
                )
                finalized.append(FinalizedBatch(
                    batch_id=batch_id, tx_submitted=False
                ))
                continue

            self._finalized_ids.add(batch_id)
            finalized.append(FinalizedBatch(
                batch_id=batch_id, tx_submitted=True
            ))

        return finalized

    # ── Introspection ────────────────────────────────────────────

    def tracked_batches(self) -> List[CommittedBatch]:
        """Snapshot of every batch this client committed on chain."""
        return list(self._tracked.values())

    def get_tracked(self, batch_id: bytes) -> Optional[CommittedBatch]:
        return self._tracked.get(batch_id)

    def pending_commits(self) -> List[PendingCommit]:
        """Snapshot of broadcast-but-unconfirmed commits awaiting reconciliation.
        Non-empty here means operator attention may be warranted: these receipts
        are NOT settled and NOT in the accumulator (they will not be re-committed)
        until reconcile_pending_commits resolves their tx's fate."""
        return list(self._pending_commits.values())

    async def reconcile_pending_commits(self) -> int:
        """Resolve quarantined broadcast-but-unconfirmed commits. For each, ask the
        contract whether its tx actually landed (get_committed_batch_for_tx). If it
        did, ADOPT the batch into the tracked set using the on-chain batch_id +
        commit_timestamp (so finalize/reconcile flows pick it up) and clear it from
        quarantine. If it did not land (still pending, dropped, or reverted) leave
        it quarantined — we never auto-re-commit, since a re-commit of a tx that is
        merely slow would double-settle. Returns the count newly adopted."""
        adopted = 0
        for tx_hash, pc in list(self._pending_commits.items()):
            try:
                result = await self._contract.get_committed_batch_for_tx(tx_hash)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"reconcile lookup failed for pending tx {tx_hash}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            if not result:
                continue  # not landed yet — stay quarantined (conservative)
            batch_id, commit_ts = result
            self._tracked[batch_id] = CommittedBatch(
                batch_id=batch_id,
                tx_hash=tx_hash,
                provider_address=pc.provider_address,
                requester_address=pc.requester_address,
                merkle_root=pc.merkle_root,
                receipt_count=pc.receipt_count,
                total_value_ftns=pc.total_value_ftns,
                commit_timestamp=commit_ts,
                leaf_hashes=pc.leaf_hashes,
                trigger_reason=pc.trigger_reason,
            )
            del self._pending_commits[tx_hash]
            adopted += 1
        return adopted

    def is_finalized_locally(self, batch_id: bytes) -> bool:
        """True if this client has seen finalizeBatch succeed. Does NOT
        reconcile against on-chain state (use get_batch_status for that)."""
        return batch_id in self._finalized_ids

    async def reconcile_finalized(self) -> int:
        """Query on-chain BatchStatus for each tracked batch and mark
        any that reached FINALIZED as locally finalized (status=2 per
        the Solidity enum). Useful after a restart, or when a third-party
        watchdog finalizes one of our batches per §2.4 of design.

        Returns count of batches newly reconciled as finalized."""
        count = 0
        for batch_id in self._tracked:
            if batch_id in self._finalized_ids:
                continue
            try:
                status = await self._contract.get_batch_status(batch_id)
            except Exception:
                continue
            if status == 2:  # FINALIZED per Solidity BatchStatus enum
                self._finalized_ids.add(batch_id)
                count += 1
        return count
