"""Sprint 1383 — challenge auto-defense (sp1411: the verdict now branches on the on-chain reason).

When a ``ReceiptChallenged`` lands against one of this node's committed batches (surfaced by the
sp1382 SettlementChallengeWatcher), self-verify the RETAINED §7 receipt for the challenged leaf and
tell the operator which case they are in.

**sp1411 — why the original verdict was unsafe.** It classified purely on
``ChallengeReport.receipt_ok``: receipt verifies → "BAD-FAITH … no stake should be lost". But:

  1. ``BatchSettlementRegistry.challengeReceipt`` does ``if (!proven) revert ChallengeNotProven``
     *before* ``emit ReceiptChallenged``. **Every challenge that reaches this callback was already
     proven on-chain**, and the registry has already done ``b.invalidatedValueFTNS += leaf.valueFtns``
     — the provider will not be paid for that receipt, whatever we conclude offline.
  2. The §7 verifier only proves an invalid settler signature and broken activation-chain grounds.
     It says NOTHING about DOUBLE_SPEND, NO_ESCROW, EXPIRED or CONSENSUS_MISMATCH. A DOUBLE_SPEND is
     a *validly signed* receipt replayed into two batches: it verifies, so ``receipt_ok`` is True.

So a DOUBLE_SPEND or CONSENSUS_MISMATCH — both of which slash the provider's bond on-chain — was
logged as "appears BAD-FAITH … no stake should be lost" and bucketed ``bad_faith``. The documented
alert target ``prsm_challenge_defense_legitimate_total`` stayed at 0 for two of the three slashing
reason codes, so an operator who wired their pager to it (as sp1384 instructs) was actively
reassured while their stake burned.

The verdict now comes from ``challenge.reason_code`` (carried on ChallengeEvent since sp1381), and
the §7 self-verify is demoted to *evidence* attached to the log line — never a reason to stand down:

  - **SLASHING** (DOUBLE_SPEND / INVALID_SIGNATURE / CONSENSUS_MISMATCH) — the three reasons the
    registry slashes on. Stake at risk. ERROR.
  - **GRIEFABLE** (NO_ESCROW) — value invalidated, no slash. This ground is the *requester's
    attestation*, not a cryptographic proof (the contract itself notes the "griefing risk if slash
    triggered"), so this is where genuine bad faith lives. ERROR; dispute off-chain.
  - **EXPIRED** — value invalidated, no slash. Protocol hygiene: the receipt fell outside the
    batch's committed lookback window. ERROR; commit sooner.
  - **UNKNOWN** — an unrecognised code (a contract upgrade). Fail-safe to stake-at-risk. Never
    reassure on a code we do not understand.

This stays a read-only, offline self-assessment (no tx, no on-chain response — on-chain challenges
are self-proving). It wires in as the watcher's ``on_new_challenge`` callback. The store is resolved
lazily (via a getter) because the watcher is built before the receipt store during node startup.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# BatchSettlementRegistry.ReasonCode — contracts/contracts/BatchSettlementRegistry.sol:218
REASON_DOUBLE_SPEND = 0
REASON_INVALID_SIGNATURE = 1
REASON_NO_ESCROW = 2
REASON_EXPIRED = 3
REASON_MALFORMED = 4
REASON_CONSENSUS_MISMATCH = 5

_REASON_NAMES = {
    REASON_DOUBLE_SPEND: "DOUBLE_SPEND",
    REASON_INVALID_SIGNATURE: "INVALID_SIGNATURE",
    REASON_NO_ESCROW: "NO_ESCROW",
    REASON_EXPIRED: "EXPIRED",
    REASON_MALFORMED: "MALFORMED",
    REASON_CONSENSUS_MISMATCH: "CONSENSUS_MISMATCH",
}

#: The three reasons ``challengeReceipt`` slashes the provider's bond on. Mirrors the
#: ``reason == DOUBLE_SPEND || INVALID_SIGNATURE || CONSENSUS_MISMATCH`` guard in the registry.
#: NO_ESCROW is requester-attestation-based (griefing risk) and EXPIRED is hygiene — neither slashes.
SLASHING_REASONS = frozenset({
    REASON_DOUBLE_SPEND, REASON_INVALID_SIGNATURE, REASON_CONSENSUS_MISMATCH,
})


class DefenseVerdict(str, Enum):
    """What an on-chain-proven challenge means for this operator."""
    SLASHING = "slashing"     # stake at risk + value invalidated
    GRIEFABLE = "griefable"   # NO_ESCROW: value invalidated, no slash, requester attestation
    EXPIRED = "expired"       # value invalidated, no slash, protocol hygiene
    UNKNOWN = "unknown"       # unrecognised reason code → fail-safe to stake-at-risk


def classify_challenge(challenge: Any) -> DefenseVerdict:
    """Map an on-chain ``ReceiptChallenged`` to its consequence for this operator.

    Fail-safe: anything we do not recognise (a missing/None/non-int reason, a future contract's new
    code) is UNKNOWN, which the caller treats as stake-at-risk. Never reassure by default."""
    code = getattr(challenge, "reason_code", None)
    if isinstance(code, bool) or not isinstance(code, int):
        return DefenseVerdict.UNKNOWN
    if code in SLASHING_REASONS:
        return DefenseVerdict.SLASHING
    if code == REASON_NO_ESCROW:
        return DefenseVerdict.GRIEFABLE
    if code == REASON_EXPIRED:
        return DefenseVerdict.EXPIRED
    return DefenseVerdict.UNKNOWN


class ChallengeDefenseStats:
    """Sprint 1384 — running tallies of auto-defense verdicts, a metric source. ``record`` matches
    the ``build_challenge_auto_defense`` ``on_verdict(challenge, report_or_None)`` signature.

    sp1411 — the buckets are keyed on the CHALLENGE's on-chain reason, not on whether our retained
    receipt happened to verify:

      - ``legitimate``  — SLASHING (+ UNKNOWN, fail-safe). **The alert target**: stake at risk.
      - ``bad_faith``   — NO_ESCROW, the requester-attestation ground. The harassment signal.
      - ``expired``     — EXPIRED. Value invalidated, no slash.
      - ``no_receipt``  — we could not self-assess (no store / no retained receipt / verify error).
        This is an assessment-COVERAGE counter, orthogonal to the three verdict buckets above: a
        slashing challenge with no retained receipt increments both ``legitimate`` and
        ``no_receipt``, because the pager must fire whether or not we kept the receipt.
    """

    def __init__(self) -> None:
        self._bad_faith = 0
        self._legitimate = 0
        self._expired = 0
        self._no_receipt = 0

    def record(self, challenge: Any, report: Any = None) -> None:
        verdict = classify_challenge(challenge)
        if verdict is DefenseVerdict.GRIEFABLE:
            self._bad_faith += 1
        elif verdict is DefenseVerdict.EXPIRED:
            self._expired += 1
        else:  # SLASHING, or UNKNOWN → fail-safe onto the alert target
            self._legitimate += 1
        if report is None:
            self._no_receipt += 1

    @property
    def bad_faith(self) -> int:
        return self._bad_faith

    @property
    def legitimate(self) -> int:
        return self._legitimate

    @property
    def expired(self) -> int:
        return self._expired

    @property
    def no_receipt(self) -> int:
        return self._no_receipt


def build_challenge_auto_defense(
    receipt_store_getter: Any,
    *,
    on_verdict: Optional[Callable[[Any, Any], None]] = None,
) -> Callable[[Any], None]:
    """Return an ``on_new_challenge(challenge_event)`` callback.

    It classifies the challenge by its on-chain reason code, self-verifies the retained §7 receipt
    for the challenged leaf as supporting EVIDENCE, and logs the consequence.
    ``on_verdict(challenge, report_or_None)`` is invoked for programmatic hooks/metrics — always,
    including when we cannot self-assess, so the operator's alert still fires.
    """
    def _defend(challenge: Any) -> None:
        batch = getattr(challenge, "batch_id", "?")
        verdict = classify_challenge(challenge)
        code = getattr(challenge, "reason_code", None)
        reason = getattr(challenge, "reason", None) or _REASON_NAMES.get(code, f"UNKNOWN({code})")
        leaf_hex = (getattr(challenge, "receipt_leaf_hash", "") or "").strip()

        report, evidence = _self_assess(receipt_store_getter, leaf_hex)
        _log_verdict(verdict, batch, reason, evidence)
        _safe_verdict(on_verdict, challenge, report)

    return _defend


def _self_assess(receipt_store_getter: Any, leaf_hex: str):
    """Return ``(report_or_None, evidence_str)``. Never raises — a failed self-assessment must not
    stop the verdict from being logged and recorded."""
    store = receipt_store_getter() if callable(receipt_store_getter) else receipt_store_getter
    if store is None:
        return None, "no retained-receipt store wired, so the receipt could not be self-verified"
    try:
        leaf = bytes.fromhex(leaf_hex[2:] if leaf_hex.startswith("0x") else leaf_hex)
    except Exception:  # noqa: BLE001 — a malformed leaf is still a real challenge
        return None, f"the challenged leaf {leaf_hex!r} is unparseable, so it could not be looked up"
    try:
        record = store.get(leaf)
    except Exception as exc:  # noqa: BLE001
        return None, f"the retained-receipt lookup errored: {exc}"
    if record is None:
        return None, (
            f"no retained §7 receipt for leaf {leaf_hex} (evicted, or not this node's batch)"
        )
    try:
        from prsm.settlement.challenge_verifier import (
            verify_inference_receipt_for_challenge,
        )
        report = verify_inference_receipt_for_challenge(
            record.inference_receipt,
            settler_public_key_b64=record.settler_public_key_b64,
            stage_public_keys=record.stage_public_keys)
    except Exception as exc:  # noqa: BLE001 — never let self-verify crash the watcher
        return None, f"self-verify errored for leaf {leaf_hex}: {exc}"

    if getattr(report, "receipt_ok", False):
        return report, (
            f"the retained §7 receipt for leaf {leaf_hex} VERIFIES (served at "
            f"GET /settlement/receipt/leaf/{leaf_hex} for independent confirmation) — note this "
            f"only refutes signature-class grounds, not the reason proven on chain"
        )
    proven = [f for f in getattr(report, "findings", []) if getattr(f, "proven", False)]
    grounds = ", ".join(str(getattr(f, "reason", "?")) for f in proven) or "unknown"
    return report, (
        f"the retained §7 receipt for leaf {leaf_hex} FAILS verification (grounds: {grounds})"
    )


def _log_verdict(verdict: DefenseVerdict, batch: Any, reason: str, evidence: str) -> None:
    if verdict is DefenseVerdict.SLASHING:
        logger.error(
            "auto-defense: challenge %s on batch %s is on-chain-PROVEN and SLASHABLE. The registry "
            "emits ReceiptChallenged only after proving the ground, and this reason slashes the "
            "provider's bond. STAKE IS AT RISK and the leaf's value is already invalidated. "
            "Evidence: %s.", reason, batch, evidence)
    elif verdict is DefenseVerdict.GRIEFABLE:
        logger.error(
            "auto-defense: challenge NO_ESCROW on batch %s invalidated the leaf's value — you will "
            "NOT be paid for that receipt. No stake is slashed. This ground is the REQUESTER'S "
            "ATTESTATION, not a cryptographic proof, so it is the griefing vector: dispute it "
            "off-chain. Evidence: %s.", batch, evidence)
    elif verdict is DefenseVerdict.EXPIRED:
        logger.error(
            "auto-defense: challenge EXPIRED on batch %s invalidated the leaf's value — you will "
            "NOT be paid for that receipt. No stake is slashed; the receipt fell outside the "
            "batch's committed lookback window, so commit sooner. Evidence: %s.", batch, evidence)
    else:
        logger.error(
            "auto-defense: challenge %s on batch %s carries an UNRECOGNISED reason code — treating "
            "it as STAKE AT RISK (fail-safe). Evidence: %s.", reason, batch, evidence)


def _safe_verdict(cb: Optional[Callable[[Any, Any], None]], challenge: Any, report: Any) -> None:
    if cb is None:
        return
    try:
        cb(challenge, report)
    except Exception as exc:  # noqa: BLE001 — a bad hook must not break the watcher
        logger.warning("auto-defense on_verdict callback failed: %s", exc)
