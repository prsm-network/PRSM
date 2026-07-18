"""
Batch Settlement Manager
========================

Queues on-chain FTNS transfers and settles them in batches instead of
broadcasting every individual transaction to Base mainnet.

Why batch?
  - Base L2 costs ~$0.001-0.01 per tx.  At 1,000 jobs/day that's $1-10/day.
  - Netting opposing transfers (A→B 5, B→A 3 = settle 2) cuts volume further.
  - Single-tx settlement amortizes gas across many local operations.
  - The local DAG ledger remains the source of truth for instant balance checks.

Architecture:
  Local DAG Ledger  (instant, free, authoritative)
       |
       v
  BatchSettlementManager._queue   (pending on-chain transfers)
       |
       v  [periodic flush OR manual trigger OR threshold]
  OnChainFTNSLedger.transfer()    (real Base mainnet ERC-20 transfer)

Settlement modes:
  1. PERIODIC  — flush every N seconds (default: 600s / 10 min)
  2. THRESHOLD — flush when pending value exceeds N FTNS (default: 1.0)
  3. MANUAL    — flush on explicit API call (prsm ftns settle)
  4. ON_WITHDRAW — flush only when a user withdraws to external wallet
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class SettlementMode(str, Enum):
    """When to flush the settlement queue to on-chain."""
    PERIODIC = "periodic"         # Flush every N seconds
    THRESHOLD = "threshold"       # Flush when pending value exceeds limit
    MANUAL = "manual"             # Only flush on explicit trigger
    ON_WITHDRAW = "on_withdraw"   # Only when tokens leave the network


@dataclass
class PendingTransfer:
    """A queued on-chain transfer waiting for batch settlement."""
    tx_id: str
    from_wallet: str
    to_wallet: str            # resolved 0x address
    amount: float
    job_id: str
    queued_at: float = field(default_factory=time.time)
    description: str = ""


@dataclass
class SettlementResult:
    """Outcome of a batch settlement flush."""
    settled_count: int = 0
    total_amount: float = 0.0
    net_transfers: int = 0     # after netting
    tx_hashes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class BatchSettlementManager:
    """Queues on-chain FTNS transfers and settles in batches.

    Drop-in replacement for direct _on_chain_ftns_transfer calls.
    The local DAG ledger has already committed — this only controls
    WHEN the on-chain mirror broadcast happens.
    """

    def __init__(
        self,
        ftns_ledger,                       # OnChainFTNSLedger instance
        node_id: str,
        connected_address: Optional[str] = None,
        mode: SettlementMode = SettlementMode.PERIODIC,
        flush_interval: float = 600.0,     # 10 minutes
        flush_threshold: float = 1.0,      # 1.0 FTNS
        max_queue_size: int = 1000,
        persist_dir: Optional[str] = None,
    ):
        self._ftns_ledger = ftns_ledger
        self._node_id = node_id
        self._connected_address = connected_address
        self.mode = mode
        self.flush_interval = flush_interval
        self.flush_threshold = flush_threshold
        self.max_queue_size = max_queue_size

        # sp1476 — durable state. Without this, a process restart (every deploy)
        # silently dropped the queued + in-flight owed on-chain payouts, voiding
        # the sp914/sp917/sp1475 re-queue safety net. persist_dir=None → in-memory
        # only (tests / ephemeral). The flush path persists the queue-CLEAR before
        # broadcasting so a crash favors a recoverable loss over an irreversible
        # double-pay; a graceful stop()'s final flush + persist loses nothing.
        self._persist_path: Optional[Path] = None
        if persist_dir and persist_dir not in ("", ":memory:"):
            d = Path(persist_dir)
            d.mkdir(parents=True, exist_ok=True)
            self._persist_path = d / "batch_settlement.json"

        # Transfer queue
        self._queue: List[PendingTransfer] = []
        self._lock = asyncio.Lock()
        # sp930 — serialize reconcile_in_flight() against itself. flush() calls
        # it OUTSIDE self._lock (line 239) and a threshold-triggered flush can
        # run concurrently with the periodic one, so without this two reconciles
        # both iterate the same in-flight set and both re-queue a reverted tx →
        # the next flush broadcasts it twice → double-pay.
        self._reconcile_lock = asyncio.Lock()

        # Deduplication. sp1476 — bounded (was an unbounded set that grew
        # forever): a companion FIFO deque caps membership + makes it persistable.
        # It must persist so a restart does not re-enqueue an already-settled
        # tx_id → re-broadcast → double-pay.
        self._settled_ids: Set[str] = set()
        self._settled_order: Deque[str] = deque()
        self._max_settled_ids = 50000

        # Stats
        self._total_settled = 0
        self._total_gas_saved = 0  # number of transfers avoided by netting
        self._last_flush_at: float = 0.0
        self._settlement_history: List[SettlementResult] = []

        # sp917/sp1475 — in-flight (broadcast-but-unconfirmed) settlement
        # transfers awaiting receipt reconciliation. Each entry:
        # {from_addr, to_addr, amount, tx_hash, nonce, first_tracked_at,
        # attempts}. A reverted tx is re-queued; a confirmed one is dropped; an
        # ambiguous one is re-queued ONLY once nonce-advance + a receipt re-poll
        # PROVE it dead (sp1475, mirroring the sp1439/sp1474 withdraw reconciler)
        # — never silently dropped.
        self._in_flight: List[Dict[str, Any]] = []
        self._max_reconcile_attempts = 20  # sp1475 — observability only; no longer gates a drop
        self._max_in_flight = 1000
        # sp1475 — bounded retry for an on-chain REVERT (status==rejected). An
        # ERC-20 revert is atomic (no tokens moved) so re-queue is safe + recovers
        # the dominant transient cause (operator hot-wallet low on FTNS, self-heals
        # on top-up). A recipient that reverts _max_reject_retries times in a row is
        # dead-lettered (surfaced, not silently dropped, not looped forever).
        self._reject_retry_counts: Dict[str, int] = {}
        self._max_reject_retries = 5
        self._dead_letter: List[Dict[str, Any]] = []
        # sp1475 — a still-ambiguous in-flight tx (no receipt, nonce not yet
        # advanced) is kept + loudly surfaced past this wall-clock age, never
        # dropped (dropping risks losing a payout that later reverts).
        self._reconcile_stale_seconds = 3600.0

        # Background task
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False

        # sp1476 — restore any durably-persisted queue / in-flight / dedup so a
        # restart resumes owed payouts instead of dropping them. Best-effort.
        self._load()

    # ── Deduplication (bounded) ────────────────────────────────

    def _mark_settled(self, tx_id: str) -> None:
        """Record a settled tx_id for dedup, FIFO-bounded (sp1476)."""
        if not tx_id or tx_id in self._settled_ids:
            return
        self._settled_ids.add(tx_id)
        self._settled_order.append(tx_id)
        while len(self._settled_order) > self._max_settled_ids:
            old = self._settled_order.popleft()
            self._settled_ids.discard(old)

    # ── Durable persistence (sp1476) ───────────────────────────

    def _persist(self) -> None:
        """Atomically write the queue + in-flight + dedup + dead-letter so a
        restart can resume. No-op when persist_dir was not configured."""
        if self._persist_path is None:
            return
        try:
            snapshot = {
                "queue": [asdict(p) for p in self._queue],
                "in_flight": self._in_flight,
                "settled_ids": list(self._settled_order),
                "dead_letter": self._dead_letter,
            }
            tmp = self._persist_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(snapshot))
            tmp.replace(self._persist_path)   # atomic
        except (OSError, TypeError) as exc:
            logger.warning("BatchSettlement: persist failed (%s)", exc)

    def _load(self) -> None:
        if self._persist_path is None or not self._persist_path.exists():
            return
        try:
            raw = json.loads(self._persist_path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning(
                "BatchSettlement: load failed (%s); starting empty", exc)
            return
        for d in raw.get("queue", []):
            try:
                self._queue.append(PendingTransfer(**d))
            except (TypeError, KeyError):
                continue
        self._in_flight = [e for e in raw.get("in_flight", []) if isinstance(e, dict)]
        for tx_id in raw.get("settled_ids", []):
            self._mark_settled(tx_id)
        self._dead_letter = [e for e in raw.get("dead_letter", []) if isinstance(e, dict)]
        if self._queue or self._in_flight:
            logger.info(
                "BatchSettlement: restored %d queued + %d in-flight owed payout(s) "
                "from %s", len(self._queue), len(self._in_flight), self._persist_path)

    # ── Lifecycle ──────────────────────────────────────────────

    def start(self) -> None:
        """Start the periodic flush task (if mode is PERIODIC)."""
        if self._running:
            return
        self._running = True
        if self.mode == SettlementMode.PERIODIC:
            self._flush_task = asyncio.create_task(self._periodic_flush_loop())
            logger.info(
                f"BatchSettlement started: mode={self.mode.value}, "
                f"interval={self.flush_interval}s, threshold={self.flush_threshold} FTNS"
            )
        else:
            logger.info(f"BatchSettlement started: mode={self.mode.value}")

    async def stop(self) -> None:
        """Stop the flush task and settle any remaining queue."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush on shutdown
        if self._queue:
            logger.info(f"BatchSettlement stopping: flushing {len(self._queue)} pending transfers")
            await self.flush()

    # ── Enqueue ────────────────────────────────────────────────

    async def enqueue(self, transaction) -> bool:
        """Queue a transaction for batch settlement.

        This is the drop-in replacement for _on_chain_ftns_transfer.
        Called by PaymentEscrow.broadcast_tx after local ledger commit.

        Returns True if queued, False if skipped (dedup, wrong type, etc.)
        """
        if not self._ftns_ledger or not self._ftns_ledger._is_initialized:
            return False

        # Extract fields
        tx_id = getattr(transaction, "tx_id", "")
        from_wallet = getattr(transaction, "from_wallet", "")
        to_wallet = getattr(transaction, "to_wallet", "")
        amount = float(getattr(transaction, "amount", 0))

        if amount <= 0:
            return False

        # Dedup
        if tx_id in self._settled_ids:
            return False

        # Resolve to_wallet to 0x address
        target_address = self._resolve_address(to_wallet)
        if not target_address:
            logger.debug(
                f"BatchSettlement: skipping non-chain wallet {to_wallet[:20]}…"
            )
            return False

        # Queue it
        pending = PendingTransfer(
            tx_id=tx_id,
            from_wallet=from_wallet,
            to_wallet=target_address,
            amount=amount,
            job_id=tx_id,
            description=getattr(transaction, "description", ""),
        )

        async with self._lock:
            self._queue.append(pending)
            self._mark_settled(tx_id)
            self._persist()   # sp1476 — durable before we report success

        logger.debug(
            f"BatchSettlement: queued {amount:.6f} FTNS → {target_address[:12]}… "
            f"(queue: {len(self._queue)})"
        )

        # Auto-flush on threshold
        if self.mode in (SettlementMode.PERIODIC, SettlementMode.THRESHOLD):
            pending_total = sum(p.amount for p in self._queue)
            if pending_total >= self.flush_threshold:
                logger.info(
                    f"BatchSettlement: threshold reached ({pending_total:.4f} >= "
                    f"{self.flush_threshold}), flushing"
                )
                asyncio.create_task(self.flush())

        # Auto-flush on queue size
        if len(self._queue) >= self.max_queue_size:
            logger.info(f"BatchSettlement: max queue size reached, flushing")
            asyncio.create_task(self.flush())

        return True

    # ── Settlement / Flush ─────────────────────────────────────

    async def flush(self) -> SettlementResult:
        """Settle all pending transfers in a batch.

        Nets opposing transfers to minimize on-chain transactions:
          A→B 5.0, B→A 3.0 → net: A→B 2.0 (1 tx instead of 2)
        """
        start = time.time()
        result = SettlementResult()

        # sp917 — reconcile any prior in-flight settlement txs FIRST: a reverted
        # one is re-queued here, so it is then broadcast in this same flush.
        await self.reconcile_in_flight()

        async with self._lock:
            if not self._queue:
                return result

            pending = list(self._queue)
            self._queue.clear()
            # sp1476 — persist the CLEAR before broadcasting. If we crash mid-
            # flush, the persisted queue is already empty so these transfers are
            # NOT re-broadcast (double-pay); a broadcast one is captured in
            # _in_flight (persisted per-transfer below), a never-broadcast one is
            # re-queued + persisted at the end. Crash favors recoverable loss.
            self._persist()

        result.settled_count = len(pending)
        result.total_amount = sum(p.amount for p in pending)

        # Net transfers by (from_addr, to_addr) pair
        net_amounts = self._net_transfers(pending)
        result.net_transfers = len(net_amounts)
        self._total_gas_saved += result.settled_count - result.net_transfers

        logger.info(
            f"BatchSettlement: flushing {result.settled_count} transfers "
            f"→ {result.net_transfers} net on-chain txs "
            f"(saved {result.settled_count - result.net_transfers} gas txs)"
        )

        # Execute net transfers.
        # sp914 — a transfer that was NEVER broadcast (transfer() returns None,
        # or raises) must be RE-QUEUED, not silently dropped: the queue was
        # already cleared above, so without re-queue the owed on-chain payout
        # is lost forever and the payee is never paid. A transfer that WAS
        # broadcast but is unconfirmed ("pending", carries a real tx_hash) must
        # NOT be re-queued — it is in the mempool and a retry would DOUBLE-PAY.
        # A reverted ("rejected") transfer is a deterministic on-chain failure;
        # re-queuing would loop, so it is surfaced as an error instead.
        requeued = 0

        async def _mark_requeue(from_addr: str, to_addr: str, net_amount: float) -> None:
            # sp1476 — persist each never-broadcast / reverted / raised re-queue
            # IMMEDIATELY (append to the DURABLE self._queue + persist), not into a
            # loop-local list flushed only after the whole broadcast loop: a crash
            # mid-loop would otherwise silently, permanently lose the owed payout
            # (it would be in neither the restored queue nor in-flight). These
            # branches moved NO tokens (transfer() returned None, or an atomic
            # ERC-20 revert), so an incremental durable re-queue carries ZERO
            # double-pay risk.
            nonlocal requeued
            async with self._lock:
                self._queue.append(PendingTransfer(
                    tx_id=f"requeue-{uuid.uuid4().hex[:12]}",
                    from_wallet=from_addr,
                    to_wallet=to_addr,
                    amount=net_amount,
                    # sp930 — uuid, not int(time.time()): a whole-second job_id
                    # collides for two transfers in the same flush second and the
                    # on-chain audit table (job_id PK, INSERT OR REPLACE) silently
                    # overwrites the first row.
                    job_id=f"requeue-{uuid.uuid4().hex[:12]}",
                    description="re-queued: on-chain transfer never broadcast (sp914)",
                ))
                self._persist()
            requeued += 1

        for (from_addr, to_addr), net_amount in net_amounts.items():
            if net_amount <= 0:
                continue
            try:
                tx_record = await self._ftns_ledger.transfer(
                    job_id=f"batch-{uuid.uuid4().hex[:12]}",  # sp930 — collision-free (see _mark_requeue)
                    to_address=to_addr,
                    amount_ftns=net_amount,
                )
            except Exception as e:
                # An unexpected raise means we cannot confirm a broadcast →
                # treat as never-sent and re-queue for a safe retry.
                result.errors.append(f"{to_addr[:12]}: {e} (re-queued)")
                logger.error(f"BatchSettlement: transfer to {to_addr[:12]}… raised: {e}")
                await _mark_requeue(from_addr, to_addr, net_amount)
                continue

            status = getattr(tx_record, "status", None) if tx_record else None
            if tx_record and status == "confirmed":
                result.tx_hashes.append(tx_record.tx_hash)
                # sp1475 — a clean settle clears the transient-revert retry count
                # for this payee (the low-balance dry spell has ended).
                self._reject_retry_counts.pop(to_addr, None)
                logger.info(
                    f"BatchSettlement: {net_amount:.6f} FTNS → {to_addr[:12]}… "
                    f"confirmed (tx: {tx_record.tx_hash[:16]}…)"
                )
            elif tx_record and status == "pending":
                # Broadcast succeeded but unconfirmed — in-flight. Surface the
                # real tx_hash for reconciliation; do NOT re-queue (double-pay).
                # sp917 — track it so reconcile_in_flight() re-queues the owed
                # payout if it ultimately reverts (was a silent drop before).
                result.tx_hashes.append(tx_record.tx_hash)
                self._track_in_flight(
                    from_addr, to_addr, net_amount, tx_record.tx_hash,
                    nonce=getattr(tx_record, "nonce", None),
                )
                self._persist()   # sp1476 — durable in-flight (with tx_hash) so a
                #                    crash does NOT re-broadcast this landed payout
                result.errors.append(
                    f"Pending (unconfirmed, not re-queued): {net_amount:.6f} → "
                    f"{to_addr[:12]} tx={(tx_record.tx_hash or '')[:16]}"
                )
                logger.warning(
                    f"BatchSettlement: {net_amount:.6f} FTNS → {to_addr[:12]}… "
                    f"broadcast but UNCONFIRMED — left for reconciliation, NOT re-queued"
                )
            elif tx_record and status == "rejected":
                # sp1475 — an on-chain ERC-20 revert is ATOMIC (no tokens moved),
                # so the payout is still owed and re-queuing is safe (cannot
                # double-pay) — exactly as the reconcile path already does for a
                # reverted in-flight tx. The dominant cause is a transient operator
                # hot-wallet FTNS shortfall that self-heals on top-up; the old
                # "drop + error only" branch stranded EVERY provider in the batch.
                # Bound the pathological permanently-reverting recipient with a
                # per-payee retry cap → dead-letter (surfaced, never silent).
                n = self._reject_retry_counts.get(to_addr, 0) + 1
                self._reject_retry_counts[to_addr] = n
                if n <= self._max_reject_retries:
                    result.errors.append(
                        f"Rejected (reverted on-chain, re-queued {n}/"
                        f"{self._max_reject_retries}): {net_amount:.6f} → {to_addr[:12]}"
                    )
                    await _mark_requeue(from_addr, to_addr, net_amount)
                else:
                    self._dead_letter.append({
                        "to_addr": to_addr, "amount": net_amount,
                        "reason": "reverted on-chain "
                                  f"{n} times (exceeds {self._max_reject_retries})",
                    })
                    result.errors.append(
                        f"Rejected (reverted {n}×, DEAD-LETTERED — manual "
                        f"reconciliation): {net_amount:.6f} → {to_addr[:12]}"
                    )
                    logger.error(
                        "BatchSettlement: %.6f FTNS → %s reverted on-chain %d times "
                        "— DEAD-LETTERED (not re-queued). Operator must reconcile.",
                        net_amount, to_addr[:12], n,
                    )
            else:
                # tx_record is None → never broadcast → safe to retry.
                result.errors.append(
                    f"Never broadcast (re-queued): {net_amount:.6f} → {to_addr[:12]}"
                )
                await _mark_requeue(from_addr, to_addr, net_amount)

        # sp1476 — each never-broadcast / reverted re-queue was already appended
        # to the durable queue + persisted INCREMENTALLY inside the loop (see
        # _mark_requeue), so a crash mid-loop can no longer lose it. A final
        # persist captures the remaining state changes (dead-letters, cleared
        # reject-retry counts, confirmed drops).
        if requeued:
            logger.warning(
                f"BatchSettlement: re-queued {requeued} never-broadcast "
                f"transfer(s) for retry"
            )
        self._persist()

        result.duration_seconds = time.time() - start
        self._last_flush_at = time.time()
        self._total_settled += result.settled_count
        self._settlement_history.append(result)

        # Keep only last 100 settlement records
        if len(self._settlement_history) > 100:
            self._settlement_history = self._settlement_history[-100:]

        return result

    # ── In-flight reconciliation (sp917) ───────────────────────

    def _track_in_flight(
        self, from_addr: str, to_addr: str, amount: float, tx_hash: str,
        nonce: Optional[int] = None,
    ) -> None:
        """Record a broadcast-but-unconfirmed settlement transfer so
        reconcile_in_flight() can re-queue it if it ultimately reverts, or
        prove it dead via nonce-advance (sp1475). ``nonce`` is the tx's on-chain
        nonce (from tx_record.nonce) — required for the prove-dead recovery."""
        self._in_flight.append({
            "from_addr": from_addr,
            "to_addr": to_addr,
            "amount": float(amount),
            "tx_hash": tx_hash,
            "nonce": (int(nonce) if nonce is not None else None),
            "first_tracked_at": time.time(),
            "attempts": 0,
        })
        if len(self._in_flight) > self._max_in_flight:
            # Bound the tracker (defensive — entries normally drain as they
            # resolve). sp1475 — NEVER silently drop an un-reconciled owed
            # payout: surface the trimmed entries loudly so an operator can
            # reconcile them manually (mirrors the abandon-path logging).
            trimmed = self._in_flight[:-self._max_in_flight]
            self._in_flight = self._in_flight[-self._max_in_flight:]
            for e in trimmed:
                self._dead_letter.append({
                    "to_addr": e.get("to_addr"), "amount": e.get("amount"),
                    "tx_hash": e.get("tx_hash"),
                    "reason": "in-flight tracker overflow (>%d) — trimmed"
                              % self._max_in_flight,
                })
            logger.error(
                "BatchSettlement: in-flight tracker overflow — %d owed payout(s) "
                "trimmed + DEAD-LETTERED (manual reconciliation): %s",
                len(trimmed),
                ", ".join(f"{e.get('amount')}→{str(e.get('to_addr'))[:12]}" for e in trimmed),
            )

    async def reconcile_in_flight(self) -> Dict[str, int]:
        """Poll each in-flight settlement tx receipt and resolve it.

        confirmed → drop (the on-chain settlement landed);
        reverted  → RE-QUEUE the owed payout + drop (an ERC-20 transfer revert
                    is atomic — no tokens moved — so a re-queue cannot
                    double-pay; without this the payout was silently dropped);
        no receipt → keep for the next tick; after _max_reconcile_attempts the
                    tx is AMBIGUOUS (may still land) so it is dropped WITHOUT
                    auto-requeue (re-queuing a tx that could still confirm would
                    double-pay) and surfaced for manual operator reconciliation.
        """
        # sp930 — serialize against concurrent reconciles (two flushes can each
        # call this). Without it both iterate the same in-flight set and both
        # re-queue a reverted tx → double-pay.
        async with self._reconcile_lock:
            return await self._reconcile_in_flight_locked()

    async def _reconcile_in_flight_locked(self) -> Dict[str, int]:
        if not self._in_flight:
            return {"confirmed": 0, "reverted": 0, "still_pending": 0,
                    "abandoned": 0}

        w3 = getattr(self._ftns_ledger, "w3", None)
        loop = asyncio.get_running_loop()
        confirmed = reverted = still_pending = abandoned = requeued_dead = 0
        survivors: List[Dict[str, Any]] = []
        requeue: List[PendingTransfer] = []

        # sp930 — iterate a SNAPSHOT, not the live list. Entries that a
        # concurrent flush's _track_in_flight appends while we await receipt
        # polls are merged back (by identity) at the end so they are never
        # dropped — including the >max_in_flight trim edge where _track replaces
        # the list object out from under us.
        snapshot = list(self._in_flight)
        snapshot_ids = {id(e) for e in snapshot}

        for entry in snapshot:
            tx_hash = entry.get("tx_hash") or ""
            receipt = None
            polled = False
            if w3 is not None and tx_hash:
                h = tx_hash if tx_hash.startswith("0x") else "0x" + tx_hash
                try:
                    receipt = await loop.run_in_executor(
                        None, lambda hh=h: w3.eth.get_transaction_receipt(hh),
                    )
                    polled = True
                except Exception as exc:  # noqa: BLE001 — transient RPC; keep
                    logger.debug(
                        "BatchSettlement reconcile: receipt poll for %s "
                        "failed: %s", tx_hash[:18], exc,
                    )
            if polled and receipt is not None:
                if receipt.get("status") == 1:
                    confirmed += 1
                    continue   # landed — drop from in-flight
                # status 0 → reverted atomically → re-queue the owed payout
                reverted += 1
                requeue.append(PendingTransfer(
                    tx_id=f"requeue-{uuid.uuid4().hex[:12]}",
                    from_wallet=entry["from_addr"],
                    to_wallet=entry["to_addr"],
                    amount=entry["amount"],
                    job_id=f"requeue-{uuid.uuid4().hex[:12]}",  # sp930 — collision-free
                    description=(
                        "re-queued: batch settlement tx reverted on-chain "
                        "(sp917)"
                    ),
                ))
                continue
            # No receipt (still pending, dropped, or RPC error). sp1475 — NEVER
            # blindly drop after a fixed attempt budget (the old behavior lost a
            # genuinely-dead payout, and concurrent flushes inflated the count so
            # a still-pending tx was ABANDONED prematurely, defeating the sp917
            # revert recovery). Instead PROVE the tx dead via nonce-advance + a
            # receipt re-poll (mirrors the sp1439/sp1474 withdraw reconciler): a
            # DIFFERENT tx took the account's nonce slot AND this tx_hash has no
            # receipt → it can never mine → re-queue the owed payout (safe: a
            # never-landed tx moved no tokens). If not provably dead, KEEP it (a
            # tx that could still confirm must never be re-queued → double-pay).
            entry["attempts"] = entry.get("attempts", 0) + 1
            sender = self._connected_address or getattr(
                self._ftns_ledger, "_connected_address", None)
            entry_nonce = entry.get("nonce")
            proven_dead = False
            if w3 is not None and sender and entry_nonce is not None and tx_hash:
                try:
                    confirmed_nonce = await loop.run_in_executor(
                        None, lambda: w3.eth.get_transaction_count(sender, "latest"))
                except Exception:  # noqa: BLE001 — transient RPC; retry next tick
                    confirmed_nonce = None
                if confirmed_nonce is not None and int(confirmed_nonce) > int(entry_nonce):
                    hh = tx_hash if tx_hash.startswith("0x") else "0x" + tx_hash
                    try:
                        rp = await loop.run_in_executor(
                            None, lambda x=hh: w3.eth.get_transaction_receipt(x))
                    except Exception:  # noqa: BLE001 — not found → receipt absent
                        rp = None
                    if rp is None:
                        proven_dead = True
            if proven_dead:
                requeued_dead += 1
                requeue.append(PendingTransfer(
                    tx_id=f"requeue-{uuid.uuid4().hex[:12]}",
                    from_wallet=entry["from_addr"],
                    to_wallet=entry["to_addr"],
                    amount=entry["amount"],
                    job_id=f"requeue-{uuid.uuid4().hex[:12]}",  # sp930 — collision-free
                    description=(
                        "re-queued: settlement tx provably dead "
                        "(sp1475 nonce-advance + receipt-absent)"
                    ),
                ))
                continue
            # Not provably dead → keep tracking. Surface loudly if stale; NEVER drop.
            age = time.time() - entry.get("first_tracked_at", time.time())
            if age >= self._reconcile_stale_seconds:
                logger.error(
                    "BatchSettlement reconcile: tx %s STILL unconfirmed after "
                    "%.0fs (%d polls) and NOT provably dead — KEEPING for "
                    "reconciliation (never dropped). %.6f FTNS → %s",
                    tx_hash[:18], age, entry["attempts"], entry["amount"],
                    entry["to_addr"][:12],
                )
            still_pending += 1
            survivors.append(entry)

        # Preserve any entries a concurrent flush tracked during our awaits
        # (they are not in the snapshot we processed).
        concurrent = [e for e in self._in_flight if id(e) not in snapshot_ids]
        self._in_flight = survivors + concurrent
        if requeue:
            async with self._lock:
                self._queue.extend(requeue)
                self._persist()   # sp1476 — durable requeue + updated in-flight
            logger.warning(
                "BatchSettlement reconcile: re-queued %d reverted settlement "
                "transfer(s) for retry", len(requeue),
            )
        else:
            # sp1476 — persist the resolved in-flight set (confirmed dropped,
            # survivors kept) even when nothing was re-queued.
            self._persist()
        return {"confirmed": confirmed, "reverted": reverted,
                "still_pending": still_pending, "abandoned": abandoned,
                "requeued_dead": requeued_dead}

    # ── Netting ────────────────────────────────────────────────

    def _net_transfers(
        self, pending: List[PendingTransfer]
    ) -> Dict[Tuple[str, str], float]:
        """Net opposing transfers to minimize on-chain transactions.

        For self-compute (provider == requester), all transfers to the node's
        own on-chain address are aggregated into one transfer.

        For multi-node, opposing transfers between the same pair are netted:
          A→B 5.0 and B→A 3.0 becomes A→B 2.0
        """
        # Aggregate by to_address (most common case: all transfers
        # go from the node's wallet to the node's wallet for self-compute,
        # or to specific provider addresses)
        aggregated: Dict[str, float] = defaultdict(float)
        from_addr = self._connected_address or ""

        for p in pending:
            aggregated[p.to_wallet] += p.amount

        # Convert to (from, to) → amount format
        net: Dict[Tuple[str, str], float] = {}
        for to_addr, amount in aggregated.items():
            if amount > 0:
                net[(from_addr, to_addr)] = amount

        return net

    # ── Address Resolution ─────────────────────────────────────

    def _resolve_address(self, wallet_id: str) -> Optional[str]:
        """Resolve a wallet ID to an on-chain 0x address.

        Returns None if the wallet is internal-only (escrow, named wallets).
        """
        if wallet_id.startswith("0x") and len(wallet_id) >= 40:
            return wallet_id
        if wallet_id == self._node_id and self._connected_address:
            return self._connected_address
        # Internal wallets (escrow-xxx, system, etc.) → skip
        return None

    # ── Background Loop ────────────────────────────────────────

    async def _periodic_flush_loop(self) -> None:
        """Background task: flush the queue periodically."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                # sp917 — also flush when in-flight txs await reconciliation,
                # so a reverted settlement is re-queued even with an empty queue.
                if self._queue or self._in_flight:
                    await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"BatchSettlement: periodic flush error: {e}")

    # ── Stats & Monitoring ─────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return settlement queue stats for monitoring."""
        pending_total = sum(p.amount for p in self._queue)
        return {
            "mode": self.mode.value,
            "queue_size": len(self._queue),
            "pending_amount": round(pending_total, 6),
            "flush_interval": self.flush_interval,
            "flush_threshold": self.flush_threshold,
            "total_settled": self._total_settled,
            "gas_txs_saved": self._total_gas_saved,
            "in_flight": len(self._in_flight),
            # sp1475 — payouts that could not be settled automatically (reverted
            # past the retry cap, or trimmed on tracker overflow). Non-zero →
            # operator must reconcile these manually.
            "dead_letter_count": len(self._dead_letter),
            "last_flush_at": self._last_flush_at,
            "last_flush_ago": (
                round(time.time() - self._last_flush_at, 1)
                if self._last_flush_at else None
            ),
            "settlement_history_count": len(self._settlement_history),
        }

    def get_pending(self) -> List[Dict[str, Any]]:
        """Return pending transfers for inspection."""
        return [
            {
                "tx_id": p.tx_id[:12],
                "to": p.to_wallet[:12] + "…",
                "amount": round(p.amount, 6),
                "queued_at": p.queued_at,
                "age_seconds": round(time.time() - p.queued_at, 1),
            }
            for p in self._queue
        ]

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return recent settlement history."""
        return [
            {
                "settled_count": r.settled_count,
                "total_amount": round(r.total_amount, 6),
                "net_transfers": r.net_transfers,
                "tx_hashes": [h[:16] + "…" for h in r.tx_hashes],
                "errors": r.errors,
                "duration_seconds": round(r.duration_seconds, 2),
            }
            for r in self._settlement_history[-limit:]
        ]
