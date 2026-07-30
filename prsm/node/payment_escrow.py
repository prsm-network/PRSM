"""
Payment Escrow
==============

Escrow system for FTNS payments on compute jobs.

Flow:
1. Requester creates escrow by locking FTNS from their wallet
2. Providers execute the job
3. When consensus is reached, escrow distributes payment to winning provider(s)
4. If consensus fails or job times out, escrow refunds the requester

This ensures providers get paid for work and requesters only pay for
verified results.
"""

import asyncio
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from prsm.economy.batch_settlement import _looks_like_node_id
from prsm.node.local_ledger import LocalLedger, Transaction

logger = logging.getLogger(__name__)


class EscrowNotFoundError(KeyError):
    """Raised when a caller references a job_id with no escrow record."""


class EscrowAmountError(ValueError):
    """Raised when a release amount exceeds the escrowed amount."""


class EscrowAlreadyFinalizedError(RuntimeError):
    """Raised on cross-state transitions: release on a REFUNDED escrow or
    refund on a RELEASED escrow. Same-state repeats (double-release,
    double-refund) are idempotent no-ops with warnings, not errors — only
    *cross-state* transitions are illegal."""


class EscrowStatus(str, Enum):
    PENDING = "pending"           # Waiting for results
    RELEASED = "released"        # Payment distributed
    REFUNDED = "refunded"        # Money returned to requester
    DISPUTED = "disputed"        # Under dispute resolution


@dataclass
class EscrowEntry:
    """A single escrow for a compute job."""
    escrow_id: str
    job_id: str
    requester_id: str
    amount: float
    status: EscrowStatus = EscrowStatus.PENDING
    provider_winner: Optional[str] = None  # Who earned the payment
    tx_lock: Optional[str] = None          # Transaction that locked the funds
    tx_release: Optional[str] = None       # Transaction that released payment
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PaymentEscrow:
    """Manages escrow accounts for compute job payments.

    The escrow system ensures:
    - Requesters pre-commit FTNS before jobs run
    - Providers are guaranteed payment if they deliver valid results
    - Failed jobs refund the requester
    - Disputed results can trigger partial refunds
    """

    def __init__(
        self,
        ledger: LocalLedger,
        node_id: str,
        broadcast_transaction: Optional[Callable] = None,
        *,
        gossip_transaction: Optional[Callable] = None,
        default_timeout: Optional[float] = None,
        cleanup_interval: Optional[float] = None,
        on_cleanup_callback: Optional[Callable] = None,
    ):
        self.ledger = ledger
        self.node_id = node_id
        self.broadcast_tx = broadcast_transaction  # async func(tx)
        # sp1494 — the CROSS-NODE credit rail. Until now a release to a REMOTE
        # payee produced a credit that existed only on THIS node's ledger: the
        # payee's own node never learned of it, and broadcast_tx cannot help
        # because it resolves node_ids to None (sp1492). Set by node.py to
        # ledger_sync.broadcast_transaction. See _maybe_gossip for the gating.
        self.gossip_tx = gossip_transaction  # async func(tx)
        self._escrows: Dict[str, EscrowEntry] = {}
        # sp907 — per-job_id locks serialize release/refund/split so the
        # PENDING->terminal status transition is not a check-then-act race
        # across the ledger.transfer await. Without this, two concurrent
        # release/refund calls on the same job both observe PENDING during
        # their awaits and both pay out (escrow wallet goes negative =
        # FTNS minted from nothing). Single-event-loop asyncio.Lock is the
        # right serialization primitive here (the escrow state is in-memory).
        self._job_locks: Dict[str, asyncio.Lock] = {}
        self._tasks: List[asyncio.Task] = []
        self._running = False
        # Optional async callback invoked after each periodic cleanup
        # sweep with the cleaned-count. When wired with a
        # webhook-dispatcher callback (see Node init), operators get
        # an "escrow.leaked" event when stale escrows are reaped.
        self._on_cleanup_callback = on_cleanup_callback

        # Timeout for unreleased escrows. Resolution order:
        #   1) explicit constructor arg (wins)
        #   2) PRSM_ESCROW_TIMEOUT_SEC env var
        #   3) v1 default 3600s
        # Invalid env values (non-numeric, zero, negative) → fall
        # back to default; zero/negative would auto-expire every
        # escrow which is a footgun.
        import os as _os
        DEFAULT_TIMEOUT = 3600.0
        DEFAULT_CLEANUP_INTERVAL = 600.0

        def _resolve(arg, env_name, default):
            if arg is not None:
                return float(arg)
            raw = _os.getenv(env_name, "").strip()
            if not raw:
                return default
            try:
                v = float(raw)
                if v <= 0:
                    logger.warning(
                        "%s=%r non-positive; using default %s",
                        env_name, raw, default,
                    )
                    return default
                return v
            except ValueError:
                logger.warning(
                    "%s=%r not numeric; using default %s",
                    env_name, raw, default,
                )
                return default

        self.default_timeout = _resolve(
            default_timeout, "PRSM_ESCROW_TIMEOUT_SEC", DEFAULT_TIMEOUT,
        )
        self.cleanup_interval = _resolve(
            cleanup_interval,
            "PRSM_ESCROW_CLEANUP_INTERVAL_SEC",
            DEFAULT_CLEANUP_INTERVAL,
        )

    async def create_escrow(
        self,
        job_id: str,
        amount: float,
        requester_id: Optional[str] = None,
    ) -> Optional[EscrowEntry]:
        """Lock FTNS in escrow for a compute job.

        Returns the escrow entry, or None if insufficient balance.
        """
        requester = requester_id or self.node_id
        # sp1477 — enforce ONE live escrow per job_id, under the same sp907
        # per-job lock that release/refund/split hold. create_escrow does not
        # dedup by job_id, and every release/refund/split locates its escrow by
        # the FIRST-PENDING match; so a duplicate create (an idempotency retry
        # whose response was lost) would fund a SECOND escrow-<uuid> wallet that
        # a later release could pay out separately — a requester-side loss
        # (conservation holds, but the requester pays twice for one job). The
        # pre-check + lock make a same-job_id re-create return the existing live
        # escrow instead of locking a second hold. Held across the balance check
        # + transfer, so it also serializes concurrent same-job creates.
        async with self._job_lock(job_id):
            for e in self._escrows.values():
                if e.job_id == job_id and e.status == EscrowStatus.PENDING:
                    logger.warning(
                        "create_escrow: job %s already has a live PENDING escrow "
                        "%s; returning it (no second hold locked)",
                        job_id[:8], e.escrow_id[:8],
                    )
                    return e
            return await self._create_escrow_locked(job_id, amount, requester)

    async def _create_escrow_locked(
        self, job_id: str, amount: float, requester: str,
    ) -> Optional[EscrowEntry]:
        balance = await self.ledger.get_balance(requester)

        if balance < amount:
            logger.warning(
                f"Escrow rejected: {requester[:12]}... has {balance:.6f} < {amount:.6f}"
            )
            return None

        # Create escrow record. Note: we register the record
        # in `self._escrows` AFTER the funds transfer succeeds —
        # sprint 489 (F27) flipped this ordering. Pre-fix, the
        # record was added BEFORE the transfer; if transfer
        # raised anything other than ValueError (e.g.,
        # ConcurrentModificationError from dag_ledger), the
        # record stayed in _escrows but funds didn't move →
        # orphaned record. Now: register only on success.
        escrow = EscrowEntry(
            escrow_id=str(uuid.uuid4()),
            job_id=job_id,
            requester_id=requester,
            amount=amount,
        )

        # Lock funds: transfer from requester to escrow wallet
        escrow_wallet = f"escrow-{escrow.escrow_id}"
        try:
            tx = await self.ledger.transfer(
                from_wallet=requester,
                to_wallet=escrow_wallet,
                amount=amount,
                description=f"Escrow for job {job_id[:8]}",
            )
            escrow.tx_lock = tx.tx_id
            # Sprint 489 (F27 fix) — register record AFTER
            # successful transfer to prevent orphan-on-failure.
            self._escrows[escrow.escrow_id] = escrow
            logger.info(
                f"Escrow created: {escrow.escrow_id[:8]}... "
                f"locked {amount:.6f} FTNS for job {job_id[:8]}..."
            )
            # Broadcast escrow creation to network
            if self.broadcast_tx:
                try:
                    await self.broadcast_tx(tx)
                except Exception as exc:  # noqa: BLE001
                    # sp1489 — do NOT roll back: the local transfer already
                    # committed and reversing it here would reopen the TOCTOU
                    # this ordering exists to close. But do not stay SILENT
                    # either: this was a bare `pass`, so a broadcast that never
                    # landed left no trace anywhere, and an operator cannot
                    # reconcile a divergence they were never told about.
                    logger.error(
                        "escrow-create broadcast FAILED for job %s (escrow %s, "
                        "%.6f FTNS, tx %s): %s: %s — local ledger committed, "
                        "network/chain did NOT. Reconciliation required.",
                        job_id[:8], escrow.escrow_id[:8], amount, tx.tx_id,
                        type(exc).__name__, exc,
                    )
            return escrow
        except ValueError as e:
            logger.warning(f"Escrow transfer failed: {e}")
            return None
        except Exception as e:
            # Sprint 489 (F27 fix) — broaden exception catch.
            # Pre-fix only ValueError was handled; the
            # ConcurrentModificationError + BalanceLockError
            # paths from dag_ledger propagated up with the
            # escrow record half-registered. Now: any error
            # leaves the requester's funds intact + raises so
            # the caller (compute_requester.submit_job) knows
            # the submit failed.
            logger.warning(
                f"Escrow transfer raised {type(e).__name__}: {e}"
            )
            raise

    def _job_lock(self, job_id: str) -> asyncio.Lock:
        """sp907 — the per-job_id lock that serializes release/refund/split.
        `setdefault` is atomic w.r.t. the event loop (no await), so lazily
        creating the lock is itself race-free."""
        return self._job_locks.setdefault(job_id, asyncio.Lock())

    async def release_escrow(
        self,
        job_id: str,
        provider_id: str,
        consensus_reached: bool = True,
        partial_amount: Optional[float] = None,
    ) -> Optional[Transaction]:
        """Release escrow payment to the winning provider — sp907 lock wrapper.
        Holds the per-job lock across the whole body so a concurrent
        release/refund on the same job cannot interleave (the loser
        re-evaluates after the winner's terminal status is committed)."""
        async with self._job_lock(job_id):
            return await self._release_escrow_locked(
                job_id, provider_id, consensus_reached, partial_amount,
            )

    async def _release_escrow_locked(
        self,
        job_id: str,
        provider_id: str,
        consensus_reached: bool = True,
        partial_amount: Optional[float] = None,
    ) -> Optional[Transaction]:
        """Release escrow payment to the winning provider.

        If consensus was reached, full payment goes to provider.
        If consensus failed but partial work was done, can specify partial_amount.

        State-machine guards:
          - Double-release on an already-RELEASED escrow: idempotent no-op
            (warns, returns None). Never double-pays.
          - Release on a REFUNDED escrow: raises EscrowAlreadyFinalizedError.
          - No escrow at all for this job_id: returns None (preserves legacy
            'not found' behavior for backwards compat with existing callers).
        """
        # sp1477 — validate partial_amount sign/finiteness (the split path
        # already does this at _release_escrow_split_locked). Without it a
        # negative value bypasses the `escrow_balance < amount` guard below
        # (100 < -X is False) and, on the non-default LocalLedger backend, a
        # reverse transfer would raise the escrow balance + debit the provider
        # + over-refund the requester; NaN poisons every comparison. No
        # production caller passes partial_amount today (DAG default guards it
        # at the primitive) — this makes the module self-defend regardless of
        # backend/caller.
        if partial_amount is not None and (
            not math.isfinite(partial_amount) or partial_amount <= 0
        ):
            raise ValueError(
                f"partial_amount must be a positive finite number, "
                f"got {partial_amount!r}"
            )

        escrow_any = None
        escrow = None
        for e in self._escrows.values():
            if e.job_id == job_id:
                escrow_any = e
                if e.status == EscrowStatus.PENDING:
                    escrow = e
                    break

        if not escrow:
            if escrow_any is not None:
                if escrow_any.status == EscrowStatus.RELEASED:
                    logger.warning(
                        f"escrow for job {job_id[:8]}... already released; "
                        f"release_escrow is a no-op"
                    )
                    return None
                if escrow_any.status == EscrowStatus.REFUNDED:
                    raise EscrowAlreadyFinalizedError(
                        f"escrow for job {job_id!r} is already REFUNDED; "
                        f"cannot release"
                    )
            logger.warning(f"No pending escrow found for job {job_id[:8]}...")
            return None

        escrow_wallet = f"escrow-{escrow.escrow_id}"
        amount = partial_amount if partial_amount is not None else escrow.amount

        # Get escrow balance
        escrow_balance = await self.ledger.get_balance(escrow_wallet)
        if escrow_balance < amount:
            logger.warning(f"Escrow wallet has {escrow_balance:.6f}, trying to release {amount:.6f}")
            return None

        # Pay the provider
        try:
            tx = await self.ledger.transfer(
                from_wallet=escrow_wallet,
                to_wallet=provider_id,
                amount=amount,
                description=f"Payment for job {job_id[:8]} (consensus={'yes' if consensus_reached else 'partial'})",
            )
            escrow.provider_winner = provider_id
            escrow.tx_release = tx.tx_id
            escrow.status = EscrowStatus.RELEASED
            escrow.completed_at = time.time()

            # Refund remainder to requester if partial
            remainder = escrow_balance - amount
            refund_tx = None
            if remainder > 0:
                refund_tx = await self.ledger.transfer(
                    from_wallet=escrow_wallet,
                    to_wallet=escrow.requester_id,
                    amount=remainder,
                    description=f"Escrow refund for job {job_id[:8]}",
                )
                logger.info(
                    f"Refunded {remainder:.6f} FTNS to requester {escrow.requester_id[:12]}..."
                )

            # Broadcast to on-chain FTNS only AFTER local ledger
            # transfers have fully committed (no TOCTOU rollback risk).
            if self.broadcast_tx:
                try:
                    await self.broadcast_tx(tx)
                    if refund_tx:
                        await self.broadcast_tx(refund_tx)
                except Exception as exc:  # noqa: BLE001
                    # sp1489 — same reasoning as create: no rollback (the local
                    # release already committed and the provider did honest
                    # work), but this must NOT be silent. This is the payout
                    # leg — a swallowed failure here means the provider is paid
                    # locally and NOT on chain, which is exactly the divergence
                    # an operator needs to know about to reconcile.
                    logger.error(
                        "escrow-release broadcast FAILED for job %s (%.6f FTNS "
                        "-> provider %s, tx %s%s): %s: %s — local ledger "
                        "committed, on-chain did NOT. Reconciliation required.",
                        job_id[:8], amount, provider_id[:12], tx.tx_id,
                        f", refund {refund_tx.tx_id}" if refund_tx else "",
                        type(exc).__name__, exc,
                    )

            # sp1494 — tell the PAYEE's own node. Without this the credit exists
            # only here and the payee never learns they were paid.
            await self._maybe_gossip(tx, provider_id, broadcast=True)

            logger.info(
                f"Escrow released: {amount:.6f} FTNS -> {provider_id[:12]}... "
                f"for job {job_id[:8]}..."
            )
            return tx
        except ValueError as e:
            logger.warning(f"Escrow release failed: {e}")
            return None

    async def release_escrow_split(
        self,
        job_id: str,
        splits: List[tuple],
        consensus_reached: bool = True,
        broadcast: bool = True,
    ) -> Optional[List[Transaction]]:
        """Release an escrow to multiple recipients — sp907 lock wrapper.
        Serialized per job_id so concurrent split-releases (or a split
        racing a release/refund) cannot double-pay recipients.

        sp1374 — ``broadcast=False`` performs the local-ledger release WITHOUT
        the on-chain BatchSettlement broadcast. Used by the multi-stage settle
        path (credit_policy) when PRSM_MULTISTAGE_SETTLEMENT is on: the per-stage
        escrow commit (sp1324/1322) settles those shares on-chain via the
        registry, so a second on-chain broadcast here would DOUBLE-PAY."""
        async with self._job_lock(job_id):
            return await self._release_escrow_split_locked(
                job_id, splits, consensus_reached, broadcast=broadcast,
            )

    async def _release_escrow_split_locked(
        self,
        job_id: str,
        splits: List[tuple],
        consensus_reached: bool = True,
        broadcast: bool = True,
    ) -> Optional[List[Transaction]]:
        """Release an escrow to multiple recipients atomically.

        Closes the §4 step 6 settlement gap for QueryOrchestrator
        swarm queries: the prompter's compute budget escrow is
        distributed across the actual compute participants (one
        share per shard) + the aggregator coordination fee, rather
        than collapsed onto the prompter's own node.

        Args:
            job_id: Job identifier whose escrow to release.
            splits: List of (recipient_id, amount) tuples. Sum
                of amounts must be <= the escrow's total amount.
                Any remainder is refunded to the requester.
            consensus_reached: Whether the swarm reached consensus
                (recorded in transaction descriptions).

        Returns:
            List of ledger Transactions (one per recipient) on
            success, or None if no pending escrow / split-amount
            invariant violated.

        State-machine guards mirror ``release_escrow``: double-
        release on a RELEASED escrow is a no-op (warns, returns
        None); release on a REFUNDED escrow raises
        ``EscrowAlreadyFinalizedError``.
        """
        if not splits:
            logger.warning(
                f"release_escrow_split called with empty splits for "
                f"job {job_id[:8]}... — returning None"
            )
            return None
        # Validate split shape eagerly so a bad caller can't waste
        # the escrow lookup work.
        for entry in splits:
            if not (isinstance(entry, tuple) and len(entry) == 2):
                raise ValueError(
                    f"splits entries must be (recipient_id, amount) "
                    f"tuples, got {entry!r}"
                )
            recipient, amount = entry
            if not isinstance(recipient, str) or not recipient:
                raise ValueError(
                    f"split recipient_id must be a non-empty string, "
                    f"got {recipient!r}"
                )
            if not isinstance(amount, (int, float)) or amount <= 0:
                raise ValueError(
                    f"split amount must be positive number, got "
                    f"{amount!r} for recipient {recipient[:12]}..."
                )

        escrow_any = None
        escrow = None
        for e in self._escrows.values():
            if e.job_id == job_id:
                escrow_any = e
                if e.status == EscrowStatus.PENDING:
                    escrow = e
                    break
        if not escrow:
            if escrow_any is not None:
                if escrow_any.status == EscrowStatus.RELEASED:
                    logger.warning(
                        f"escrow for job {job_id[:8]}... already released; "
                        f"release_escrow_split is a no-op"
                    )
                    return None
                if escrow_any.status == EscrowStatus.REFUNDED:
                    raise EscrowAlreadyFinalizedError(
                        f"escrow for job {job_id!r} is already REFUNDED; "
                        f"cannot release"
                    )
            logger.warning(
                f"No pending escrow found for job {job_id[:8]}..."
            )
            return None

        total_split = sum(amount for _, amount in splits)
        if total_split > escrow.amount + 1e-9:  # tolerance for fp
            raise ValueError(
                f"sum of splits ({total_split}) exceeds escrow amount "
                f"({escrow.amount}) for job {job_id!r}"
            )

        escrow_wallet = f"escrow-{escrow.escrow_id}"
        escrow_balance = await self.ledger.get_balance(escrow_wallet)
        # sp997 — a REAL shortfall (beyond float tolerance) must still strand
        # (never release more than is held). BUT compute_split_amounts normalizes
        # the split to sum to the budget with float rounding that can land a few
        # ULPs OVER escrow_balance (~8% of multi-recipient forge splits, e.g.
        # 10.000000000000002 vs 10.0). The old `escrow_balance < total_split`
        # had NO tolerance (unlike the +1e-9 raise guard above), so that float
        # noise returned None → zero transfers, escrow stranded PENDING, every
        # provider paid 0, the requester refunded at timeout — while the forge
        # handler still reported the job COMPLETED + 200. Tolerate the same 1e-9
        # epsilon, and clamp the float overshoot out of the largest leg so the
        # per-leg transfers sum to exactly the available balance (no over-debit,
        # no spurious remainder). The clamp is sub-wei (~1e-15 FTNS).
        if escrow_balance + 1e-9 < total_split:
            logger.warning(
                f"Escrow wallet has {escrow_balance:.6f}, trying to "
                f"split-release {total_split:.6f} (shortfall exceeds fp "
                f"tolerance) — not releasing"
            )
            return None
        if total_split > escrow_balance:
            overshoot = total_split - escrow_balance
            idx_max = max(
                range(len(splits)), key=lambda i: splits[i][1],
            )
            _r, _a = splits[idx_max]
            splits = list(splits)
            splits[idx_max] = (_r, _a - overshoot)
            total_split = sum(amount for _, amount in splits)

        # Atomic-from-caller-view: any per-recipient transfer
        # failure triggers compensating reverse-transfers for the
        # legs that already succeeded, restoring the escrow wallet
        # to its pre-call state. Escrow stays PENDING so cleanup
        # or operator retry can re-attempt.
        txs = []
        succeeded_legs: List[tuple] = []  # (recipient, amount) per success
        for recipient, amount in splits:
            try:
                tx = await self.ledger.transfer(
                    from_wallet=escrow_wallet,
                    to_wallet=recipient,
                    amount=amount,
                    description=(
                        f"Split payment for job {job_id[:8]} "
                        f"(consensus={'yes' if consensus_reached else 'partial'})"
                    ),
                )
                txs.append(tx)
                succeeded_legs.append((recipient, amount))
            except ValueError as exc:
                logger.error(
                    f"Split payment failed for recipient "
                    f"{recipient[:12]}...: {exc}. Compensating "
                    f"{len(succeeded_legs)} prior leg(s) to restore "
                    f"escrow wallet."
                )
                # Compensate already-succeeded legs.
                compensated: List[Dict[str, Any]] = []
                leaked: List[Dict[str, Any]] = []
                for prior_recipient, prior_amount in succeeded_legs:
                    try:
                        await self.ledger.transfer(
                            from_wallet=prior_recipient,
                            to_wallet=escrow_wallet,
                            amount=prior_amount,
                            description=(
                                f"Atomic rollback: compensating reverse "
                                f"of split-leg for job {job_id[:8]} "
                                f"(failed at recipient "
                                f"{recipient[:12]}...)"
                            ),
                        )
                        compensated.append({
                            "recipient": prior_recipient,
                            "amount": prior_amount,
                        })
                    except Exception as comp_exc:
                        logger.error(
                            f"Atomic rollback FAILED for prior leg "
                            f"{prior_recipient[:12]}... amount "
                            f"{prior_amount}: {comp_exc}. Escrow "
                            f"is partially-released and unrecoverable "
                            f"without operator reconciliation."
                        )
                        leaked.append({
                            "recipient": prior_recipient,
                            "amount": prior_amount,
                        })
                # Always record the failed-leg history for audit.
                escrow.metadata.setdefault("rollback_history", []).append({
                    "failed_recipient": recipient,
                    "failed_amount": amount,
                    "failed_reason": str(exc),
                    "compensated": compensated,
                    "ts": time.time(),
                })
                if leaked:
                    escrow.metadata["partial_release_unrecoverable"] = True
                    escrow.metadata.setdefault(
                        "leaked_recipients", [],
                    ).extend(leaked)
                # Escrow stays PENDING for retry / cleanup paths.
                return None

        # Mark escrow released. provider_winner stores a stable
        # "split:N" marker since there's no single winner.
        escrow.provider_winner = f"split:{len(splits)}"
        escrow.tx_release = txs[0].tx_id  # head of the split chain
        escrow.status = EscrowStatus.RELEASED
        escrow.completed_at = time.time()
        # Stash the per-recipient breakdown in metadata so audit
        # trails can reconstruct the split without log replay.
        escrow.metadata.setdefault("splits", []).extend(
            {"recipient": r, "amount": a, "tx_id": txs[i].tx_id}
            for i, (r, a) in enumerate(splits)
        )

        # Refund remainder to requester if total_split < escrow.amount.
        remainder = escrow_balance - total_split
        refund_tx = None
        if remainder > 0:
            try:
                refund_tx = await self.ledger.transfer(
                    from_wallet=escrow_wallet,
                    to_wallet=escrow.requester_id,
                    amount=remainder,
                    description=(
                        f"Escrow split-release remainder for job "
                        f"{job_id[:8]}"
                    ),
                )
                logger.info(
                    f"Refunded {remainder:.6f} FTNS to requester "
                    f"{escrow.requester_id[:12]}..."
                )
            except ValueError as exc:
                logger.warning(
                    f"Remainder refund failed (non-fatal — splits "
                    f"already released): {exc}"
                )

        # Broadcast on-chain after local ledger commit. sp1374 — skipped when
        # broadcast=False (multi-stage: the per-stage escrow commit settles these
        # shares on-chain via the registry; a second broadcast here double-pays).
        if broadcast and self.broadcast_tx:
            for tx in list(txs) + ([refund_tx] if refund_tx else []):
                try:
                    await self.broadcast_tx(tx)
                except Exception as exc:  # noqa: BLE001
                    # sp1494 — the THIRD broadcast site; sp1489 loudened the other
                    # two and missed this one. No rollback (the local split already
                    # committed), but not silent either: the local ledger says paid
                    # while the chain does not.
                    logger.error(
                        "escrow-split broadcast FAILED for payee %s (tx %s, job "
                        "%s): %s: %s — local ledger committed, on-chain did NOT. "
                        "Reconciliation required.",
                        str(getattr(tx, "to_wallet", "?"))[:16],
                        getattr(tx, "tx_id", "?"), job_id[:8],
                        type(exc).__name__, exc,
                    )

        # sp1494 — cross-node credit rail. Paired off each tx's OWN to_wallet
        # rather than by index into `splits`, so a partially-failed split cannot
        # misattribute a credit to the wrong recipient.
        for tx in txs:
            await self._maybe_gossip(
                tx, str(getattr(tx, "to_wallet", "")), broadcast=broadcast)

        logger.info(
            f"Escrow split-released: {total_split:.6f} FTNS across "
            f"{len(splits)} recipients for job {job_id[:8]}..."
        )
        return txs

    async def _maybe_gossip(self, tx, payee: str, broadcast: bool) -> bool:
        """sp1494 — publish a release so the PAYEE's own node credits itself.

        Closes the silent cross-node strand: `release_escrow` moves funds to the
        payee on THIS node's ledger only. For a remote payee that credit is
        invisible — the payee's node never sees it, and it cannot even be
        reconciled, because the tx's parties are ``escrow-<uuid>`` and the payee,
        neither of which is this node's own id, so `get_transaction_history()`
        never returns it.

        GATED ON THE SAME AXIS AS ``broadcast``, which is the whole subtlety.
        ``broadcast`` is not really an "on-chain?" switch — it answers "does
        something ELSE settle this payee?":

          * ``broadcast=False`` (credit_policy.py with PRSM_MULTISTAGE_SETTLEMENT)
            means the payee is settled on chain by per-stage self-commit. Gossiping
            as well would credit them TWICE — once withdrawable off-chain, once on
            chain. So: no gossip.
          * ``broadcast=True`` with a REMOTE NODE_ID payee means nothing else pays
            them at all: broadcast_tx routes to BatchSettlementManager, whose
            _resolve_address rejects a 32-hex node_id (sp1492). This is exactly the
            strand, and exactly where gossip belongs.

        Deliberately NOT gossiped:
          * a ``0x`` payee — broadcast_tx CAN mirror that on chain, so gossiping too
            would double-pay;
          * our OWN node_id — a self-release is already local;
          * internal wallets (``escrow-…``) — not a payee at all.

        Gossip itself cannot double-credit: the receiver dedups on tx_id through
        three independent gates (seen-nonce, an atomic INSERT-OR-IGNORE nonce claim,
        and has_transaction), so even a duplicate broadcast applies once.
        """
        if tx is None or self.gossip_tx is None or not broadcast:
            return False
        payee = (payee or "").strip()
        if not _looks_like_node_id(payee) or payee == self.node_id:
            return False
        try:
            await self.gossip_tx(tx)
            return True
        except Exception as exc:  # noqa: BLE001
            # Non-fatal: the local release already committed and the payee can
            # still be reconciled. But NOT silent — sp1489's lesson: this is the
            # only rail that tells the payee they were paid.
            logger.error(
                "escrow-release GOSSIP FAILED for payee %s (tx %s): %s: %s — the "
                "payee's own node will NOT see this credit. Reconciliation "
                "required.",
                payee[:16], getattr(tx, "tx_id", "?"),
                type(exc).__name__, exc,
            )
            return False

    async def refund_escrow(self, job_id: str, reason: str = "") -> bool:
        """Refund escrow to the requester — sp907 lock wrapper.
        Serialized per job_id so a refund cannot interleave with a
        concurrent release of the same job (which would pay the provider
        AND refund the requester from one escrow)."""
        async with self._job_lock(job_id):
            return await self._refund_escrow_locked(job_id, reason)

    async def _refund_escrow_locked(self, job_id: str, reason: str = "") -> bool:
        """Refund escrow to the requester (job failed or cancelled).

        State-machine guards:
          - Double-refund on an already-REFUNDED escrow: idempotent no-op
            (warns, returns True). Never double-refunds.
          - Refund on a RELEASED escrow: raises EscrowAlreadyFinalizedError.
          - No escrow at all for this job_id: returns False (legacy).
        """
        escrow_any = None
        escrow = None
        for e in self._escrows.values():
            if e.job_id == job_id:
                escrow_any = e
                if e.status == EscrowStatus.PENDING:
                    escrow = e
                    break

        if not escrow:
            if escrow_any is not None:
                if escrow_any.status == EscrowStatus.REFUNDED:
                    logger.warning(
                        f"escrow for job {job_id[:8]}... already refunded; "
                        f"refund_escrow is a no-op"
                    )
                    return True
                if escrow_any.status == EscrowStatus.RELEASED:
                    raise EscrowAlreadyFinalizedError(
                        f"escrow for job {job_id!r} is already RELEASED; "
                        f"cannot refund"
                    )
            return False

        escrow_wallet = f"escrow-{escrow.escrow_id}"
        balance = await self.ledger.get_balance(escrow_wallet)

        if balance > 0:
            try:
                await self.ledger.transfer(
                    from_wallet=escrow_wallet,
                    to_wallet=escrow.requester_id,
                    amount=balance,
                    description=f"Escrow refund: {reason}",
                )
                logger.info(
                    f"Escrow refunded: {balance:.6f} FTNS -> {escrow.requester_id[:12]}... "
                    f"({reason})"
                )
            except ValueError:
                return False

        escrow.status = EscrowStatus.REFUNDED
        escrow.completed_at = time.time()
        return True

    async def cleanup_expired_escrows(self) -> int:
        """Refund any escrows that have exceeded the timeout."""
        now = time.time()
        cleaned = 0
        for escrow in list(self._escrows.values()):
            if (
                escrow.status == EscrowStatus.PENDING
                and now - escrow.created_at > self.default_timeout
            ):
                await self.refund_escrow(escrow.job_id, reason="Escrow timed out")
                cleaned += 1
        return cleaned

    async def periodic_cleanup(self) -> None:
        """Run cleanup every ``self.cleanup_interval`` seconds.
        Configurable via constructor arg or
        PRSM_ESCROW_CLEANUP_INTERVAL_SEC env var; default 600s.

        After each sweep, invokes ``on_cleanup_callback(cleaned)``
        if wired. Callback is best-effort: exceptions logged at
        WARN, never break the cleanup loop."""
        self._running = True
        while self._running:
            await asyncio.sleep(self.cleanup_interval)
            try:
                cleaned = await self.cleanup_expired_escrows()
                if cleaned:
                    logger.info(f"Cleaned up {cleaned} expired escrows")
                if self._on_cleanup_callback is not None:
                    try:
                        await self._on_cleanup_callback(cleaned)
                    except Exception as cb_exc:
                        logger.warning(
                            "PaymentEscrow on_cleanup_callback raised: %s",
                            cb_exc,
                        )
            except Exception as e:
                logger.error(f"Escrow cleanup error: {e}")

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

    def list_escrows_by_requester(
        self,
        requester_id: str,
        *,
        pending_only: bool = True,
    ) -> List[EscrowEntry]:
        """Return escrows owned by `requester_id`.

        By default returns only PENDING escrows — funds actively
        locked up + at risk + part of the requester's outstanding
        liability. Released / refunded escrows are accounting
        history, not current position; pass `pending_only=False`
        to include them.

        Address matching is case-insensitive (a wallet may be
        stored as checksummed 0xAb… and queried as lowercased
        0xab…; both must match).

        Public surface used by aggregate-source quoting in
        `prsm_balance_check` (audit-prep §7.23 honest-scope).
        """
        normalized = requester_id.lower()
        result: List[EscrowEntry] = []
        for e in self._escrows.values():
            if e.requester_id.lower() != normalized:
                continue
            if pending_only and e.status != EscrowStatus.PENDING:
                continue
            result.append(e)
        return result

    def get_escrow(self, job_id: str) -> Optional[EscrowEntry]:
        for e in self._escrows.values():
            if e.job_id == job_id:
                return e
        return None

    def get_by_escrow_id(self, escrow_id: str) -> Optional[EscrowEntry]:
        """Direct lookup by escrow_id (the unique primary key).
        Distinct from ``get_escrow(job_id)`` — multiple escrows
        could in principle share a job_id (though current
        operational practice gives each job exactly one). Useful
        for operators investigating a specific escrow_id from
        logs / tx receipts."""
        return self._escrows.get(escrow_id)

    def get_stats(self) -> Dict[str, Any]:
        statuses = {}
        for e in self._escrows.values():
            statuses[e.status.value] = statuses.get(e.status.value, 0) + 1

        total_locked = sum(
            e.amount for e in self._escrows.values() if e.status == EscrowStatus.PENDING
        )

        return {
            "total_escrows": len(self._escrows),
            "by_status": statuses,
            "total_locked_ftns": round(total_locked, 6),
        }
