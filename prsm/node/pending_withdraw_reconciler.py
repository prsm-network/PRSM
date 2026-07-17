"""Sprint 916 — pending-withdraw reconciler.

Closes the dead-end exposed by sp914: a withdraw debits the off-chain wallet
BEFORE broadcasting the on-chain ERC-20 transfer. sp914 correctly made a
broadcast-but-unconfirmed tx return ``status="pending"`` WITHOUT refunding (the
tx is in the mempool and will likely confirm — refunding then would double-pay).
But if that pending tx later REVERTS on-chain, nothing refunds the off-chain
debit → the user permanently loses the FTNS. The only prior reconciliation
(``OnChainFTNSLedger._reconcile_pending_transactions``) runs at startup and only
updates tx *status*; it takes no corrective action.

This module records each pending withdraw (``job_id → wallet_id, amount, tx_hash,
nonce``) in a small bounded, persisted store, and a background reconciler polls the
receipt:

  * confirmed → resolve, no refund (the on-chain transfer succeeded);
  * reverted  → refund the off-chain debit, then resolve;
  * dropped   → (sp1439) the tx was evicted/dropped and can NEVER mine — proven when the
                escrow's CONFIRMED nonce has advanced strictly past this tx's nonce — so
                it never produces a receipt and would otherwise strand the debit forever;
                refund and resolve;
  * unconfirmed but still live → leave for the next tick.

The refund is EXACTLY-ONCE: it credits with ``idempotency_key="withdraw-refund:{job_id}"``,
so a reconciler restart / double-run credits the wallet exactly once (the credit's
deterministic ``tx_id`` collides on the transactions PRIMARY KEY on replay). sp1439 replaced
the earlier record_nonce-claim-before-credit design, whose two separate durable commits left
a crash-in-the-gap window that could burn the claim without ever crediting.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Local import kept lazy/local where the enum is used to avoid a heavy import
# chain at module load; the tx_type string is stable.
_REFUND_TX_TYPE_VALUE = "bridge_withdraw"


@dataclass
class WithdrawIntent:
    """A pending withdraw whose on-chain tx may still revert."""
    job_id: str
    wallet_id: str
    amount: float
    to_addr: str
    tx_hash: str
    recorded_at: float = field(default_factory=time.time)
    resolved: bool = False
    outcome: Optional[str] = None   # "confirmed" | "refunded" | "refunded_dropped"
    # sp1439 — the on-chain nonce this withdraw tx was signed at. Optional so intents
    # persisted before sp1439 still load (nonce=None → the reconciler can't prove the tx
    # is dropped and leaves it pending, exactly the pre-sp1439 behavior — no regression).
    nonce: Optional[int] = None
    # sp1474 — the debit was taken but the on-chain transfer definitively did NOT land
    # (never-broadcast / reverted) AND the endpoint's inline refund credit itself FAILED.
    # There is no on-chain tx to poll; the reconciler must simply retry the idempotent
    # refund until it lands, rather than leaving the debit stranded with only a log line.
    refund_owed: bool = False


class PendingWithdrawStore:
    """Bounded, optionally-persisted record of pending withdraws.

    persist_dir=None → in-memory only (tests / ephemeral). Otherwise a single
    JSON file under persist_dir survives restart so a reconciler can refund a
    revert that lands while the daemon was down. Resolved entries are pruned
    once the store exceeds max_entries (sp897 unbounded-disk discipline).
    """

    _FILE = "pending_withdraws.json"

    def __init__(self, persist_dir: Optional[str] = None, max_entries: int = 10_000):
        self._max = max(1, int(max_entries))
        self._path: Optional[Path] = None
        if persist_dir:
            d = Path(persist_dir)
            d.mkdir(parents=True, exist_ok=True)
            self._path = d / self._FILE
        # job_id → WithdrawIntent, insertion-ordered.
        self._intents: Dict[str, WithdrawIntent] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────
    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
            for d in raw.get("intents", []):
                try:
                    self._intents[d["job_id"]] = WithdrawIntent(**d)
                except (TypeError, KeyError):
                    continue
        except (OSError, ValueError) as exc:
            logger.warning("PendingWithdrawStore: load failed (%s); starting empty", exc)

    def _save(self) -> None:
        if not self._path:
            return
        try:
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(
                {"intents": [asdict(i) for i in self._intents.values()]}
            ))
            tmp.replace(self._path)   # atomic
        except OSError as exc:
            logger.warning("PendingWithdrawStore: save failed (%s)", exc)

    def _prune(self) -> None:
        if len(self._intents) <= self._max:
            return
        # Drop oldest RESOLVED entries first; never drop unresolved (those
        # still owe a reconciliation decision).
        resolved = [j for j, i in self._intents.items() if i.resolved]
        for job_id in resolved:
            if len(self._intents) <= self._max:
                break
            del self._intents[job_id]

    # ── api ──────────────────────────────────────────────────────
    def record(self, *, job_id: str, wallet_id: str, amount: float,
               to_addr: str, tx_hash: str, nonce: Optional[int] = None) -> None:
        if job_id in self._intents:
            return   # idempotent — a withdraw records its intent once
        self._intents[job_id] = WithdrawIntent(
            job_id=job_id, wallet_id=wallet_id, amount=float(amount),
            to_addr=to_addr, tx_hash=tx_hash,
            nonce=(int(nonce) if nonce is not None else None),
        )
        self._prune()
        self._save()

    def record_refund_owed(self, *, job_id: str, wallet_id: str,
                           amount: float, to_addr: str) -> None:
        """sp1474 — durably record a debit whose broadcast failed AND whose inline
        refund credit ALSO failed, so the reconciler retries the (idempotent) refund
        instead of stranding it on a log line. Idempotent per job_id."""
        if job_id in self._intents:
            return
        self._intents[job_id] = WithdrawIntent(
            job_id=job_id, wallet_id=wallet_id, amount=float(amount),
            to_addr=to_addr, tx_hash="", refund_owed=True,
        )
        self._prune()
        self._save()

    def unresolved(self) -> List[WithdrawIntent]:
        return [i for i in self._intents.values() if not i.resolved]

    def all(self) -> List[WithdrawIntent]:
        return list(self._intents.values())

    def mark_resolved(self, job_id: str, outcome: str) -> None:
        intent = self._intents.get(job_id)
        if intent is None:
            return
        intent.resolved = True
        intent.outcome = outcome
        self._prune()
        self._save()


async def reconcile_pending_withdraws(
    store: PendingWithdrawStore,
    *,
    get_receipt_status: Callable[[str], Awaitable[str]],
    refund: Callable[[WithdrawIntent], Awaitable[bool]],
    is_dropped: Optional[Callable[[WithdrawIntent], Awaitable[bool]]] = None,
) -> Dict[str, int]:
    """Resolve every unresolved withdraw intent.

    ``get_receipt_status(tx_hash)`` → "confirmed" | "reverted" | "pending"
    (anything else is treated as still-pending — leave for the next tick).
    ``refund(intent)`` performs the idempotent off-chain refund.
    ``is_dropped(intent)`` (sp1439, optional) → True iff a still-"pending" tx is PROVABLY
    dead (dropped/evicted and can never mine). Such a tx never produces a receipt, so the
    confirmed/reverted branches can't fire and the off-chain debit would otherwise be
    stranded forever; when it returns True we refund via the same idempotent path. Omitted
    (None) → the pre-sp1439 behavior (a stuck-pending tx stays pending).
    """
    confirmed = refunded = still_pending = dropped = 0
    for intent in list(store.unresolved()):
        # sp1474 — a refund-owed intent has NO on-chain tx to poll (the transfer
        # never landed and the inline refund failed). Just retry the idempotent
        # refund until it lands, then resolve.
        if getattr(intent, "refund_owed", False):
            try:
                await refund(intent)
            except Exception as exc:  # noqa: BLE001 — leave unresolved, retry next tick
                logger.error("reconcile: owed-refund for %s FAILED (will retry): %s",
                             intent.job_id, exc)
                still_pending += 1
                continue
            store.mark_resolved(intent.job_id, "refunded")
            refunded += 1
            continue
        try:
            status = await get_receipt_status(intent.tx_hash)
        except Exception as exc:  # noqa: BLE001 — transient RPC; retry next tick
            logger.debug("reconcile: receipt poll for %s failed: %s",
                         intent.job_id, exc)
            still_pending += 1
            continue
        if status == "confirmed":
            store.mark_resolved(intent.job_id, "confirmed")
            confirmed += 1
        elif status == "reverted":
            try:
                await refund(intent)
            except Exception as exc:  # noqa: BLE001 — leave unresolved, retry
                logger.error("reconcile: refund for %s FAILED (will retry): %s",
                             intent.job_id, exc)
                still_pending += 1
                continue
            store.mark_resolved(intent.job_id, "refunded")
            refunded += 1
        else:
            # sp1439 — still "pending": refund ONLY if it is provably dead (a dropped tx
            # that can never mine). A tx that could still land is left pending so we never
            # refund a debit whose transfer subsequently confirms (double-pay).
            dead = False
            if is_dropped is not None:
                try:
                    dead = bool(await is_dropped(intent))
                except Exception as exc:  # noqa: BLE001 — be conservative, retry next tick
                    logger.debug("reconcile: dropped-check for %s failed: %s",
                                 intent.job_id, exc)
                    dead = False
            if dead:
                try:
                    intent.outcome = "dropped"   # for the refund description
                    await refund(intent)
                except Exception as exc:  # noqa: BLE001 — leave unresolved, retry
                    logger.error("reconcile: dropped-refund for %s FAILED (will retry): %s",
                                 intent.job_id, exc)
                    still_pending += 1
                    continue
                store.mark_resolved(intent.job_id, "refunded_dropped")
                dropped += 1
            else:
                still_pending += 1
    if confirmed or refunded or dropped:
        logger.info(
            "pending-withdraw reconcile: %d confirmed, %d refunded, %d dropped-refunded, "
            "%d still pending", confirmed, refunded, dropped, still_pending,
        )
    return {"confirmed": confirmed, "refunded": refunded,
            "dropped_refunded": dropped, "still_pending": still_pending}


def resolve_pending_withdraw_reconciler_config_from_env() -> tuple[bool, float]:
    """(enabled, interval_seconds). Enabled by DEFAULT — this is a money-safety
    net and is cheap when idle (it no-ops on an empty store). Disable with
    ``PRSM_PENDING_WITHDRAW_RECONCILER_ENABLED=0``. Interval defaults to 300s,
    clamped to a 60s floor.
    """
    enabled_raw = os.environ.get(
        "PRSM_PENDING_WITHDRAW_RECONCILER_ENABLED", "1",
    ).strip().lower()
    enabled = enabled_raw not in ("0", "false", "no", "off")

    interval = 300.0
    interval_raw = os.environ.get(
        "PRSM_PENDING_WITHDRAW_RECONCILER_INTERVAL_S", "",
    ).strip()
    if interval_raw:
        try:
            interval = float(interval_raw)
        except ValueError:
            logger.warning(
                "PRSM_PENDING_WITHDRAW_RECONCILER_INTERVAL_S=%r invalid; "
                "defaulting to 300s", interval_raw,
            )
    if interval < 60.0:
        interval = 60.0
    return enabled, interval


class PendingWithdrawReconciler:
    """Background worker that polls pending withdraw receipts and refunds
    reverts. Co-locates the on-chain receipt poll (via the FTNS ledger's web3)
    with the off-chain refund (via the local ledger), keeping the pure
    reconcile loop testable in isolation.
    """

    def __init__(
        self,
        *,
        store: PendingWithdrawStore,
        ftns_ledger,
        local_ledger,
        interval_seconds: float = 300.0,
    ):
        self._store = store
        self._ftns_ledger = ftns_ledger
        self._local_ledger = local_ledger
        self.interval_seconds = max(60.0, float(interval_seconds))
        self._running = False
        self.confirmed_total = 0
        self.refunded_total = 0

    async def _get_receipt_status(self, tx_hash: str) -> str:
        """Poll the chain for a single tx receipt → confirmed/reverted/pending."""
        w3 = getattr(self._ftns_ledger, "w3", None)
        if w3 is None or not tx_hash:
            return "pending"
        h = tx_hash if tx_hash.startswith("0x") else "0x" + tx_hash
        loop = asyncio.get_running_loop()
        receipt = await loop.run_in_executor(
            None, lambda: w3.eth.get_transaction_receipt(h),
        )
        if receipt is None:
            return "pending"
        return "confirmed" if receipt.get("status") == 1 else "reverted"

    async def _refund(self, intent: WithdrawIntent) -> bool:
        """Refund the off-chain debit EXACTLY ONCE across restarts / double-runs.

        sp1439 (audit) — idempotency is the CREDIT itself (``idempotency_key`` makes
        ``tx_id`` deterministic, so a replay collides on the transactions PRIMARY KEY and
        is a no-op). The old design claimed a separate ``record_nonce`` BEFORE crediting;
        a crash in the gap between the two durable commits burned the claim while no credit
        landed, and on restart the burned claim made ``_refund`` return without ever
        crediting — a permanent lost refund. Folding idempotency into the credit removes
        that window: a crash before the credit commits simply re-runs the same idempotent
        credit on the next tick and the wallet is made whole exactly once."""
        from prsm.node.local_ledger import TransactionType
        await self._local_ledger.credit(
            wallet_id=intent.wallet_id,
            amount=intent.amount,
            tx_type=TransactionType.BRIDGE_WITHDRAW,
            idempotency_key=f"withdraw-refund:{intent.job_id}",
            description=(
                f"bridge withdraw REFUND (reconciler: on-chain tx {intent.outcome or 'reverted'}) "
                f"job={intent.job_id} tx={intent.tx_hash}"
            ),
        )
        logger.warning(
            "reconcile: REFUNDED %s FTNS to %s — withdraw %s did not deliver on-chain",
            intent.amount, intent.wallet_id, intent.job_id,
        )
        return True

    async def _is_dropped(self, intent: WithdrawIntent) -> bool:
        """sp1439 — is this pending tx PROVABLY dead (dropped/evicted and can never mine)?

        Deterministic + double-pay-safe: a tx can only be refunded once it is IMPOSSIBLE
        for it to still land. That is precisely when the escrow's CONFIRMED nonce has
        advanced STRICTLY past this tx's nonce — a different tx took the slot, so this
        tx_hash is permanently 'nonce too low'. If our tx had itself confirmed, the
        confirmed-status branch would have caught it before this check. A legacy intent
        with no recorded nonce can't be proven dead, so it stays pending (no regression)."""
        if intent.nonce is None:
            return False
        w3 = getattr(self._ftns_ledger, "w3", None)
        sender = getattr(self._ftns_ledger, "_connected_address", None)
        if w3 is None or not sender:
            return False
        loop = asyncio.get_running_loop()
        try:
            confirmed_nonce = await loop.run_in_executor(
                None, lambda: w3.eth.get_transaction_count(sender, "latest"))
        except Exception as exc:  # noqa: BLE001 — transient RPC; retry next tick
            logger.debug("reconcile: confirmed-nonce poll for %s failed: %s",
                         intent.job_id, exc)
            return False
        if int(confirmed_nonce) <= int(intent.nonce):
            return False
        # sp1474 — the nonce slot is taken, but that alone does NOT prove OUR tx
        # died: the taker could BE our tx, having CONFIRMED in the gap between the
        # earlier get_receipt_status read (which returned pending) and this nonce
        # read — or the two reads may hit RPC replicas at different block heights.
        # Refunding then double-pays a landed withdraw. Re-poll THIS tx's receipt
        # last: if it now exists, our tx confirmed/reverted → NOT dropped (let the
        # next tick's confirmed/reverted branch handle it). Only a permanently-null
        # receipt (a DIFFERENT tx took the slot) is provably dead → refund.
        h = intent.tx_hash if intent.tx_hash.startswith("0x") else "0x" + intent.tx_hash
        try:
            receipt = await loop.run_in_executor(
                None, lambda: w3.eth.get_transaction_receipt(h))
        except Exception:  # noqa: BLE001 — receipt not found / transient → treat as absent
            receipt = None
        if receipt is not None:
            # Our tx is in a block after all — do not refund it.
            return False
        return True

    async def reconcile_once(self) -> Dict[str, int]:
        out = await reconcile_pending_withdraws(
            self._store,
            get_receipt_status=self._get_receipt_status,
            refund=self._refund,
            is_dropped=self._is_dropped,   # sp1439 — refund provably-dead dropped txs
        )
        self.confirmed_total += out["confirmed"]
        self.refunded_total += out["refunded"] + out.get("dropped_refunded", 0)
        return out

    async def run_forever(self) -> None:
        self._running = True
        logger.info("PendingWithdrawReconciler launched (interval=%.0fs)",
                    self.interval_seconds)
        while self._running:
            try:
                await asyncio.sleep(self.interval_seconds)
                if self._store.unresolved():
                    await self.reconcile_once()
            except asyncio.CancelledError:
                return
            except Exception as exc:  # noqa: BLE001 — never crash the daemon
                logger.warning("PendingWithdrawReconciler tick failed: %s", exc)

    async def stop(self) -> None:
        self._running = False
