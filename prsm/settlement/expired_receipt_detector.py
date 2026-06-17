"""Sprint 1148 — off-chain EXPIRED detector (the 5th and FINAL on-chain-actionable
settlement class, completing DOUBLE_SPEND=0 / INVALID_SIGNATURE=1 / NO_ESCROW=2 /
EXPIRED=3 / CONSENSUS_MISMATCH=5).

THE CONDITION. EXPIRED is protocol HYGIENE, not malice: a committed leaf whose
``executed_at_unix`` is OLDER than the batch's per-batch lookback window should not be paid
from escrow. On-chain ``challengeReceipt(..., ReasonCode.EXPIRED, ...)`` ->
``_handleExpired(leaf, b.lookbackWindowSecondsAtCommit)`` (BatchSettlementRegistry.sol
840-850) is a PURE time check:

    age = block.timestamp - leaf.executedAtUnix
    lookback = lookbackWindowSecondsAtCommit
    if (age <= lookback) revert ReceiptNotExpired(leaf.executedAtUnix, lookback)
    return true

So a leaf is EXPIRED-challengeable IFF ``age > lookback`` (STRICT GREATER; ``age <=
lookback`` is NOT expired). EXPIRED does NOT slash the provider's stake (the slash gate fires
only for DOUBLE_SPEND/INVALID_SIGNATURE/CONSENSUS_MISMATCH); it invalidates the challenged
leaf value (``b.invalidatedValueFTNS += leaf.valueFtns``) so finalize won't draw it from
escrow. ANY observer can challenge (no ``msg.sender`` restriction); ``auxData`` is UNUSED.

CRITICAL SAFETY (anti-griefing) — WHY FAIL-CLOSED PROTECTS HONEST PROVIDERS. An EXPIRED
finding leads to a challenge that DENIES the provider payment for that leaf. A FALSE EXPIRED
finding therefore wrongly DENIES an HONEST provider. So this detector:

  1. matches ``_handleExpired`` EXACTLY — it flags ONLY ``age > lookback`` (strict; a leaf
     AT the boundary ``age == lookback`` is NEVER flagged, mirroring the on-chain revert);
  2. is FAIL-CLOSED on the lookback: if the per-batch lookback is missing / ``None`` / ``<=
     0`` / unknown, NO leaf in that batch is flagged (a finding would lead to a challenge that
     could grief an honest provider, so we refuse to flag without a trustworthy lookback —
     the lookback is the chain-anchored ``lookbackWindowSecondsAtCommit``);
  3. is FAIL-CLOSED per receipt: any receipt that can't be turned into a canonical leaf
     (malformed / observed-only) is SKIPPED + logged, never flagged — a skipped receipt is
     simply not a finding (safe: no false EXPIRED).

CONSERVATIVE SAFETY MARGIN. The detector evaluates ``now - executed_at_unix`` at SCAN time,
but the on-chain check uses ``block.timestamp`` at SUBMIT time. Age only GROWS over time, so
a leaf flagged here stays expired later — the only risk is flagging a leaf RIGHT AT the
boundary (``age`` barely over ``lookback`` now) that a clock-skew between this process and the
chain could make ``age <= lookback`` at submit. ``safety_margin_seconds`` (default 0) lets a
caller require ``age > lookback + margin`` for an extra cushion; the audit engine passes a
small positive margin. Default 0 keeps the detector's bare semantics EXACTLY the on-chain
check (the assembler + dry-run gate are the other two belts before any irreversible tx).

PURITY. No web3, no network, no I/O, no signing. Consumes verified batches (each carrying
``batch_id``, order-preserved ``receipts``, and the chain-anchored
``lookback_window_seconds``) and returns findings. It does NOT broadcast, sign, or slash —
assembling + DRY-RUNNING the challenge is the audit-engine brick; the broadcast stays the
separate user-gated submitter path. Mirrors ``double_spend_detector`` /
``consensus_mismatch_detector``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

from prsm.economy.web3.stake_manager import ReasonCode
from prsm.settlement.merkle import ReceiptLeaf, batched_receipt_to_leaf

logger = logging.getLogger(__name__)

# The BatchSettlementRegistry ReasonCode this detector maps to (== 3).
_ON_CHAIN_EXPIRED: int = int(ReasonCode.EXPIRED)

# Stable string tag mirroring the sibling detectors' value style.
_REASON_TAG: str = "expired"


@dataclass(frozen=True)
class ExpiredReceiptFinding:
    """A single committed leaf whose age exceeds its batch's per-batch lookback window.

    ``batch_id`` + ``target_index`` are the coordinates the assembler needs to build the
    merkle proof; ``leaf`` is the canonical ``ReceiptLeaf`` (the SAME the merkle tree commits
    + the contract time-checks); ``age`` / ``lookback`` record the exact strict comparison
    (``age > lookback``) that made this leaf EXPIRED-challengeable. EXPIRED does NOT slash —
    this is hygiene, not malice — so the finding is informational+actionable, never a slash."""

    batch_id: bytes
    target_index: int
    leaf: ReceiptLeaf
    age: int
    lookback: int
    on_chain_reason: int = _ON_CHAIN_EXPIRED

    def to_dict(self) -> dict:
        return {
            "reason": _REASON_TAG,
            "batch_id": self.batch_id.hex(),
            "target_index": self.target_index,
            "executed_at_unix": int(self.leaf.executed_at_unix),
            "age": self.age,
            "lookback": self.lookback,
            "on_chain_reason": self.on_chain_reason,
        }


def detect_expired_receipts(
    batches: Iterable,
    *,
    now: int,
    safety_margin_seconds: int = 0,
) -> List[ExpiredReceiptFinding]:
    """Find every committed leaf that is STRICTLY expired (``age > lookback + margin``).

    Each batch must expose ``batch_id`` (bytes), ``receipts`` (order-preserved
    ``BatchedReceipt`` list), and ``lookback_window_seconds`` (the chain-anchored
    ``lookbackWindowSecondsAtCommit`` for that batch). For each leaf the canonical
    ``ReceiptLeaf`` is built (``batched_receipt_to_leaf`` — the SAME canonicalization the
    merkle tree commits + the contract time-checks), ``age = now - leaf.executed_at_unix``,
    and the leaf is flagged IFF ``age > lookback + safety_margin_seconds`` (STRICT, mirrors
    the on-chain ``_handleExpired`` ``age > lookback``).

    FAIL-CLOSED (anti-griefing — a finding leads to a challenge that DENIES an honest
    provider's payment):
      - a batch whose ``lookback_window_seconds`` is ``None`` / ``<= 0`` / unknown flags NO
        leaf (we refuse to flag without a trustworthy chain-anchored lookback);
      - a receipt that can't be canonicalized into a leaf is SKIPPED + logged (never flagged).

    Returns findings sorted by ``(batch_id, target_index)`` — input-order-independent.
    """
    findings: List[ExpiredReceiptFinding] = []
    now_i = int(now)
    margin = int(safety_margin_seconds)

    for batch in batches:
        batch_id = bytes(getattr(batch, "batch_id"))
        lookback_raw = getattr(batch, "lookback_window_seconds", None)
        # FAIL-CLOSED on the lookback: missing/None/<=0/unparseable -> never flag this batch.
        try:
            lookback = int(lookback_raw) if lookback_raw is not None else 0
        except (TypeError, ValueError):
            logger.debug(
                "expired detector: unparseable lookback for batch %s (%r); skipping batch "
                "(fail-closed)", batch_id.hex(), lookback_raw,
            )
            continue
        if lookback <= 0:
            logger.debug(
                "expired detector: non-positive/unknown lookback (%d) for batch %s; "
                "skipping batch (fail-closed — never grief an honest provider)",
                lookback, batch_id.hex(),
            )
            continue

        threshold = lookback + margin
        for target_index, br in enumerate(getattr(batch, "receipts", []) or []):
            # PER-RECEIPT FAIL-CLOSED: a malformed/observed-only receipt whose leaf can't be
            # built is skipped + logged (never flagged) — a skipped receipt is simply not a
            # finding (safe: no false EXPIRED).
            try:
                leaf = batched_receipt_to_leaf(br)
            except Exception as exc:  # noqa: BLE001 — fail-closed, never abort the scan
                logger.debug(
                    "expired detector: skipping unparseable receipt (batch=%s idx=%d): %s",
                    batch_id.hex(), target_index, exc,
                )
                continue
            age = now_i - int(leaf.executed_at_unix)
            # STRICT: age > lookback (+ conservative margin). age <= lookback is NOT expired
            # — the exact on-chain ``_handleExpired`` semantics (boundary is not flagged).
            if age > threshold:
                findings.append(ExpiredReceiptFinding(
                    batch_id=batch_id,
                    target_index=target_index,
                    leaf=leaf,
                    age=age,
                    lookback=lookback,
                    on_chain_reason=_ON_CHAIN_EXPIRED,
                ))

    findings.sort(key=lambda f: (f.batch_id, f.target_index))
    return findings


__all__ = [
    "ExpiredReceiptFinding",
    "detect_expired_receipts",
]
