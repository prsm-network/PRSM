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
from prsm.settlement.state_store import (
    SettlementStateCorruptError,
    SettlementStateStore,
)
# BroadcastFailedError's class is defined unconditionally (its module guards the
# web3 import), so this adds no hard web3 dependency — and a module-level import
# (vs a lazy one that could swallow to False) removes the sp1052-review footgun
# where an import failure would reclassify a BroadcastFailed as a safe-retry.
from prsm.economy.web3.provenance_registry import BroadcastFailedError

logger = logging.getLogger(__name__)

_STATE_VERSION = 1


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

    async def find_committed_batch_by_root(
        self, provider_address: str, merkle_root: bytes,
    ) -> Optional[Tuple[bytes, int]]:
        """Recover (batch_id, commit_timestamp) for a commit that landed on chain
        by scanning BatchCommitted logs (provider indexed) for a matching
        merkle_root. Returns None if none is found / the scan is inconclusive.

        Optional in the Protocol: ``recover_committing_intents`` calls it via
        getattr and no-ops if a contract client doesn't implement it (sp1040)."""
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
class CommitIntent:
    """sp1040 (brick 2.6) — a write-ahead record of a batch this client is ABOUT
    to commit, written durably BEFORE the irreversible on-chain broadcast.

    If the process crashes after commitBatch confirms on chain (escrow locked)
    but before the post-commit persist records the batch in _tracked, this intent
    is what survives the restart. recover_committing_intents() scans the chain for
    a BatchCommitted with this merkle_root and adopts the landed batch into
    _tracked — without the intent there is NO way to learn the commit happened
    (a clean confirm leaves no quarantined tx_hash), and the escrow strands."""
    accumulator_key: Tuple
    merkle_root: bytes
    leaf_hashes: Tuple[bytes, ...]
    receipt_count: int
    total_value_ftns: int
    provider_address: str
    requester_address: str
    trigger_reason: TriggerReason


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
        state_store: Optional[SettlementStateStore] = None,
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
        # sp1040 (brick 2.6) — commit-intent write-ahead log, keyed by merkle_root
        # hex. An entry means "a commit for this batch was broadcast (or about to
        # be) and may have landed". Written BEFORE the irreversible broadcast and
        # cleared once the batch is promoted to _tracked; leftovers on restart are
        # resolved by recover_committing_intents (chain scan by root).
        self._committing: Dict[str, CommitIntent] = {}
        # sp1039 (brick 2.5) — durable post-commit state. None => in-memory only
        # (pre-1039 behavior, no filesystem touch). When set, the three dicts
        # above are rehydrated here and re-persisted after every mutation, so a
        # restart never strands an already-committed-on-chain batch. A corrupt
        # state file raises out of load() (loud, not silent-empty).
        self._store = state_store
        if self._store is not None:
            persisted = self._store.load()
            if persisted is not None:
                self._apply_state(persisted)

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
            leaf_hashes, root = self._build_leaves_root(ready)
            # sp1040 review #10 — key the WAL by (accumulator_key, root), NOT root
            # alone: two distinct ready batches that happen to share a merkle_root
            # would otherwise overwrite each other's intent and strand the first.
            intent_key = self._intent_key(ready.key, root)
            # sp1040 — WAL: record the intent durably BEFORE the irreversible
            # broadcast so a crash in the commit→persist window is recoverable.
            self._committing[intent_key] = self._build_intent(ready, leaf_hashes, root)
            self._persist()
            try:
                record = await self._commit_one(ready, leaf_hashes, root)
            except Exception as exc:
                tx_hash = getattr(exc, "tx_hash", None)
                if tx_hash is not None:
                    # Broadcast-but-unconfirmed (OnChainPendingError carries
                    # tx_hash): the tx MAY still mine. Retrying would re-commit the
                    # same receipts as a DISTINCT on-chain batchId (the registry
                    # has no content dedup) → double settlement. Quarantine
                    # instead: remove from the accumulator so it can NEVER be
                    # re-committed, and record it for reconcile_pending_commits.
                    # The intent is cleared because the durable _pending_commits
                    # record (by tx_hash) now owns recovery for this commit.
                    self._committing.pop(intent_key, None)
                    self._quarantine_pending(ready, tx_hash, leaf_hashes, root)
                    logger.error(
                        f"batch commit for key {ready.key} broadcast but "
                        f"UNCONFIRMED (tx {tx_hash}); quarantined to avoid "
                        f"double settlement — will reconcile, not retry"
                    )
                elif self._is_broadcast_failed(exc):
                    # sp1052 — BroadcastFailedError has NO tx_hash, but the
                    # send_raw_transaction throwing does NOT prove the tx didn't
                    # land (a dropped RPC response after the node accepted it —
                    # the mainnet lag class). Blindly re-committing the receipts
                    # would mint a DISTINCT on-chain batchId → double-settle. So
                    # KEEP the durable intent (root-based recovery marker) and POP
                    # the receipts from the accumulator so they can NEVER be blindly
                    # re-committed; recover_committing_intents adopts the batch if it
                    # actually landed, else the intent stays surfaced
                    # (status funds_in_flight) for operator attention.
                    self._accumulator.pop_batch(ready.key)
                    self._persist()  # intent already persisted above
                    logger.error(
                        f"batch commit for key {ready.key} BROADCAST FAILED "
                        f"({exc}); intent retained + receipts quarantined — "
                        f"recover-by-root will adopt if it landed, NOT blind-retry"
                    )
                else:
                    # OnChainRevertedError — the tx MINED and atomically reverted,
                    # so NOTHING was committed → safe to retry: drop the intent,
                    # receipts stay in the accumulator for the next poll.
                    self._committing.pop(intent_key, None)
                    self._persist()
                    logger.warning(
                        f"batch commit reverted for key {ready.key}: "
                        f"{type(exc).__name__}: {exc} — receipts retained "
                        f"in accumulator for retry"
                    )
                continue

            # Pop only on success so failed commits naturally retry. Promote the
            # batch and clear its intent together (one persist) — a later loop
            # error can't lose this one, and a crash before this persist leaves
            # the intent for chain-scan recovery.
            self._accumulator.pop_batch(ready.key)
            self._tracked[record.batch_id] = record
            self._committing.pop(intent_key, None)
            self._persist()
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

    @staticmethod
    def _is_broadcast_failed(exc: Exception) -> bool:
        """sp1052 — BroadcastFailedError (send_raw_transaction threw) MAY have
        landed (dropped RPC response), so it is quarantined-not-retried; an
        OnChainRevertedError is a confirmed atomic revert and is safe to retry."""
        return isinstance(exc, BroadcastFailedError)

    @staticmethod
    def _intent_key(accumulator_key: Tuple, root: bytes) -> str:
        """sp1040 #10 — a string WAL key unique per ready batch (root alone is
        not: identical content under two accumulator keys shares a root)."""
        requester, provider, group_id, slash_bps = accumulator_key
        return f"{root.hex()}|{requester}|{provider}|{group_id.hex()}|{slash_bps}"

    def _build_intent(
        self, ready: ReadyBatch, leaf_hashes: List[bytes], root: bytes,
    ) -> CommitIntent:
        """sp1040 — the write-ahead record for a batch about to be committed."""
        requester_address, provider_address, _group_id, _slash_bps = ready.key
        return CommitIntent(
            accumulator_key=ready.key,
            merkle_root=root,
            leaf_hashes=tuple(leaf_hashes),
            receipt_count=len(leaf_hashes),
            total_value_ftns=ready.batch.total_value_ftns,
            provider_address=provider_address,
            requester_address=requester_address,
            trigger_reason=ready.trigger,
        )

    def _quarantine_pending(
        self, ready: ReadyBatch, tx_hash: str,
        leaf_hashes: Optional[List[bytes]] = None, root: Optional[bytes] = None,
    ) -> None:
        """Remove a broadcast-but-unconfirmed batch from the accumulator and stash
        it for reconciliation. Popping is what guarantees no double-commit."""
        requester_address, provider_address, _group_id, _slash_bps = ready.key
        if leaf_hashes is None or root is None:
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
        self._persist()  # quarantine must survive a restart to be reconciled

    async def _commit_one(
        self, ready: ReadyBatch,
        leaf_hashes: Optional[List[bytes]] = None, root: Optional[bytes] = None,
    ) -> CommittedBatch:
        """Convert receipts to canonical leaves, build tree, submit
        commitBatch. On success, returns the CommittedBatch record.

        Architectural note: this client is side-agnostic (provider-side
        batching or requester-side auditing both route through the same
        _commit_one code path). The on-chain contract enforces commit
        authority via `msg.sender`; the Python client trusts the
        accumulator's keyed batches."""
        # AccumulatorKey is (requester, provider, group_id, slash_rate_bps).
        requester_address, provider_address, group_id, slash_rate_bps = ready.key

        if leaf_hashes is None or root is None:
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
            self._persist()
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

    def status(self) -> Dict[str, object]:
        """sp1051 — read-only lifecycle snapshot for operator observability. A
        non-zero ``pending_commits`` (broadcast-but-unconfirmed quarantine) or
        ``committing_intents`` (commit-intent WAL entries awaiting chain-scan
        recovery) means escrow may be mid-flight and warrants attention."""
        pending = len(self._pending_commits)
        intents = len(self._committing)
        return {
            "provider_address": self._provider,
            "write_capable": getattr(self._contract, "address", None) is not None,
            "durable_state": self._store is not None,
            "tracked_batches": len(self._tracked),
            "finalized_locally": len(self._finalized_ids),
            "pending_commits": pending,
            "committing_intents": intents,
            "funds_in_flight": bool(pending or intents),
        }

    def pending_commits(self) -> List[PendingCommit]:
        """Snapshot of broadcast-but-unconfirmed commits awaiting reconciliation.
        Non-empty here means operator attention may be warranted: these receipts
        are NOT settled and NOT in the accumulator (they will not be re-committed)
        until reconcile_pending_commits resolves their tx's fate."""
        return list(self._pending_commits.values())

    def committing_intents(self) -> List[CommitIntent]:
        """Snapshot of write-ahead commit intents not yet promoted/resolved
        (sp1040). A non-empty list after a clean run means a commit may have
        landed on chain but its local promote was lost to a crash — resolved by
        recover_committing_intents (chain scan)."""
        return list(self._committing.values())

    async def recover_committing_intents(self) -> int:
        """sp1040 (brick 2.6) — resolve write-ahead commit intents left over from
        a crash in the commit→persist window. For each intent, scan the chain for
        a BatchCommitted with this provider + merkle_root: if the commit ACTUALLY
        LANDED, adopt the batch into _tracked (so finalize proceeds) and clear the
        intent; if it did not land (or the scan is inconclusive) leave the intent
        for a later cycle — never assume-failed, since a re-commit of a slow tx
        would double-settle. Returns the count newly adopted.

        No-op (returns 0) if the contract client cannot scan by root — the backfill
        simply isn't available, but recovery never crashes."""
        scan = getattr(self._contract, "find_committed_batch_by_root", None)
        if scan is None:
            return 0
        # sp1040 #9 — scan by the address that actually sent commitBatch
        # (msg.sender == the settler key) when known; the BatchCommitted.provider
        # topic is msg.sender. The sp1036 funds-safety check already pins
        # key.address == provider_address, so these agree today, but using the
        # contract's own address is correct even if that ever decouples.
        msg_sender = getattr(self._contract, "address", None)
        adopted = 0
        for intent_key, intent in list(self._committing.items()):
            scan_provider = msg_sender or intent.provider_address
            try:
                result = await scan(scan_provider, intent.merkle_root)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"commit-intent recovery scan failed for {intent_key[:12]}…:"
                    f" {type(exc).__name__}: {exc}"
                )
                continue
            if not result:
                continue  # not landed / inconclusive — keep the intent (conservative)
            batch_id, commit_ts = result
            self._tracked[batch_id] = CommittedBatch(
                batch_id=batch_id,
                tx_hash="",
                provider_address=intent.provider_address,
                requester_address=intent.requester_address,
                merkle_root=intent.merkle_root,
                receipt_count=intent.receipt_count,
                total_value_ftns=intent.total_value_ftns,
                commit_timestamp=commit_ts,
                leaf_hashes=intent.leaf_hashes,
                trigger_reason=intent.trigger_reason,
            )
            del self._committing[intent_key]
            # sp1052 — pop the receipts from the accumulator on adopt. No-op for a
            # crash-orphaned intent (its receipts were popped on the original commit
            # success), but for a BroadcastFailed-quarantined intent this guarantees
            # the now-adopted batch can never be re-committed (double-settle).
            self._accumulator.pop_batch(intent.accumulator_key)
            self._persist()
            logger.info(
                f"recovered orphaned commit (root {intent.merkle_root.hex()[:12]}…) "
                f"as batch {batch_id.hex()[:12]}… — strand avoided"
            )
            adopted += 1
        return adopted

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
            self._persist()  # adoption (tracked+ / pending-) must be durable
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
                self._persist()
                count += 1
        return count

    # ── Durable state (sp1039, brick 2.5) ────────────────────────

    def _persist(self) -> None:
        """Atomically persist the three post-commit state dicts. No-op when no
        store is configured (preserves pre-1039 in-memory behavior).

        sp1039 review #6 — a persist failure (disk full / IO error) happens
        AFTER the in-memory (and possibly on-chain) state already changed, so the
        in-memory state is now ahead of disk. Log LOUDLY and re-raise rather than
        swallow: a silent divergence would strand on the next restart, and
        re-raising bounds the divergence (the caller / poll cycle stops piling up
        more in-memory-only committed batches and surfaces the failure)."""
        if self._store is None:
            return
        try:
            self._store.save(self._serialize_state())
        except Exception as exc:  # noqa: BLE001 — re-raised below after logging
            logger.error(
                "settlement state persist FAILED (%s: %s) — in-memory state is "
                "ahead of disk; a restart before the next successful persist "
                "would lose the un-persisted change. Check disk/permissions.",
                type(exc).__name__, exc,
            )
            raise

    def _serialize_state(self) -> dict:
        """JSON-able snapshot. bytes->hex, TriggerReason->value, tuple->list so
        the round-trip is byte/enum exact (finalize + challenge proofs depend on
        merkle_root / leaf_hashes / batch_id surviving unchanged)."""
        return {
            "version": _STATE_VERSION,
            "tracked": {
                bid.hex(): self._committed_to_dict(cb)
                for bid, cb in self._tracked.items()
            },
            "finalized_ids": sorted(b.hex() for b in self._finalized_ids),
            "pending_commits": {
                tx: self._pending_to_dict(pc)
                for tx, pc in self._pending_commits.items()
            },
            "committing": {
                rh: self._intent_to_dict(ci)
                for rh, ci in self._committing.items()
            },
        }

    def _apply_state(self, state: dict) -> None:
        """Rehydrate the three dicts from a serialized snapshot.

        sp1039 review #1/#4 — a valid-JSON but schema-incompatible file (a future
        version, a hand-edited/partially-restored file, a renamed/dropped key)
        must be as LOUD as syntactic corruption. Validate the version and wrap the
        per-record decode so any mismatch raises SettlementStateCorruptError —
        which the daemon wiring turns into "settlement OFF + ERROR log" instead of
        a silent empty rehydrate that would strand every committed batch."""
        version = state.get("version")
        if version != _STATE_VERSION:
            raise SettlementStateCorruptError(
                f"settlement state version {version!r} != expected "
                f"{_STATE_VERSION} — refusing to start with a mismatched schema "
                f"(would risk forgetting committed batches)"
            )
        try:
            self._tracked = {
                bytes.fromhex(bid): self._committed_from_dict(d)
                for bid, d in (state.get("tracked") or {}).items()
            }
            self._finalized_ids = {
                bytes.fromhex(h) for h in (state.get("finalized_ids") or [])
            }
            self._pending_commits = {
                tx: self._pending_from_dict(d)
                for tx, d in (state.get("pending_commits") or {}).items()
            }
            self._committing = {
                rh: self._intent_from_dict(d)
                for rh, d in (state.get("committing") or {}).items()
            }
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            raise SettlementStateCorruptError(
                f"settlement state schema is unreadable: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _committed_to_dict(cb: CommittedBatch) -> dict:
        return {
            "batch_id": cb.batch_id.hex(),
            "tx_hash": cb.tx_hash,
            "provider_address": cb.provider_address,
            "requester_address": cb.requester_address,
            "merkle_root": cb.merkle_root.hex(),
            "receipt_count": cb.receipt_count,
            "total_value_ftns": cb.total_value_ftns,
            "commit_timestamp": cb.commit_timestamp,
            "leaf_hashes": [h.hex() for h in cb.leaf_hashes],
            "trigger_reason": cb.trigger_reason.value,
        }

    @staticmethod
    def _committed_from_dict(d: dict) -> CommittedBatch:
        return CommittedBatch(
            batch_id=bytes.fromhex(d["batch_id"]),
            tx_hash=d["tx_hash"],
            provider_address=d["provider_address"],
            requester_address=d["requester_address"],
            merkle_root=bytes.fromhex(d["merkle_root"]),
            receipt_count=d["receipt_count"],
            total_value_ftns=d["total_value_ftns"],
            commit_timestamp=d["commit_timestamp"],
            leaf_hashes=tuple(bytes.fromhex(h) for h in d["leaf_hashes"]),
            trigger_reason=TriggerReason(d["trigger_reason"]),
        )

    @staticmethod
    def _pending_to_dict(pc: PendingCommit) -> dict:
        requester, provider, group_id, slash_bps = pc.accumulator_key
        return {
            "accumulator_key": [requester, provider, group_id.hex(), slash_bps],
            "tx_hash": pc.tx_hash,
            "merkle_root": pc.merkle_root.hex(),
            "leaf_hashes": [h.hex() for h in pc.leaf_hashes],
            "receipt_count": pc.receipt_count,
            "total_value_ftns": pc.total_value_ftns,
            "provider_address": pc.provider_address,
            "requester_address": pc.requester_address,
            "trigger_reason": pc.trigger_reason.value,
        }

    @staticmethod
    def _pending_from_dict(d: dict) -> PendingCommit:
        requester, provider, group_hex, slash_bps = d["accumulator_key"]
        return PendingCommit(
            accumulator_key=(requester, provider, bytes.fromhex(group_hex),
                             slash_bps),
            tx_hash=d["tx_hash"],
            merkle_root=bytes.fromhex(d["merkle_root"]),
            leaf_hashes=tuple(bytes.fromhex(h) for h in d["leaf_hashes"]),
            receipt_count=d["receipt_count"],
            total_value_ftns=d["total_value_ftns"],
            provider_address=d["provider_address"],
            requester_address=d["requester_address"],
            trigger_reason=TriggerReason(d["trigger_reason"]),
        )

    @staticmethod
    def _intent_to_dict(ci: CommitIntent) -> dict:
        requester, provider, group_id, slash_bps = ci.accumulator_key
        return {
            "accumulator_key": [requester, provider, group_id.hex(), slash_bps],
            "merkle_root": ci.merkle_root.hex(),
            "leaf_hashes": [h.hex() for h in ci.leaf_hashes],
            "receipt_count": ci.receipt_count,
            "total_value_ftns": ci.total_value_ftns,
            "provider_address": ci.provider_address,
            "requester_address": ci.requester_address,
            "trigger_reason": ci.trigger_reason.value,
        }

    @staticmethod
    def _intent_from_dict(d: dict) -> CommitIntent:
        requester, provider, group_hex, slash_bps = d["accumulator_key"]
        return CommitIntent(
            accumulator_key=(requester, provider, bytes.fromhex(group_hex),
                             slash_bps),
            merkle_root=bytes.fromhex(d["merkle_root"]),
            leaf_hashes=tuple(bytes.fromhex(h) for h in d["leaf_hashes"]),
            receipt_count=d["receipt_count"],
            total_value_ftns=d["total_value_ftns"],
            provider_address=d["provider_address"],
            requester_address=d["requester_address"],
            trigger_reason=TriggerReason(d["trigger_reason"]),
        )
