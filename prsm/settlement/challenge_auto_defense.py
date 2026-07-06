"""Sprint 1383 — challenge auto-defense.

When a ``ReceiptChallenged`` lands against one of this node's committed batches (surfaced by the
sp1382 SettlementChallengeWatcher), automatically self-verify the RETAINED §7 receipt for the
challenged leaf and tell the operator which case they're in:

  - **BAD-FAITH** — the retained receipt VERIFIES (``ChallengeReport.receipt_ok``). The challenge is
    refutable; the receipt is already served at ``GET /settlement/receipt/leaf/{leaf}`` (sp1305) so
    the challenger / any third party can independently confirm it. Logged WARNING (actionable but
    reassuring — no stake should be lost).
  - **LEGITIMATE** — the retained receipt FAILS verification. The challenge is real; stake is at
    risk and contesting is futile. Logged ERROR so the operator reacts.
  - **NO RECEIPT** — nothing retained for that leaf (evicted, or not this node's). Can't self-assess.

This is a read-only, offline self-assessment (no tx, no on-chain response — on-chain challenges are
self-proving). It wires in as the watcher's ``on_new_challenge`` callback. The store is resolved
lazily (via a getter) because the watcher is built before the receipt store during node startup.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def build_challenge_auto_defense(
    receipt_store_getter: Any,
    *,
    on_verdict: Optional[Callable[[Any, Any], None]] = None,
) -> Callable[[Any], None]:
    """Return an ``on_new_challenge(challenge_event)`` callback that self-verifies the retained
    receipt for the challenged leaf. ``receipt_store_getter`` is the InferenceReceiptStore or a
    zero-arg callable returning it (or None). ``on_verdict(challenge, report_or_None)`` is invoked
    with the ChallengeReport (or None when no receipt was retained) for programmatic hooks/metrics.
    """
    def _defend(challenge: Any) -> None:
        batch = getattr(challenge, "batch_id", "?")
        leaf_hex = (getattr(challenge, "receipt_leaf_hash", "") or "").strip()
        store = receipt_store_getter() if callable(receipt_store_getter) else receipt_store_getter
        if store is None:
            logger.info(
                "auto-defense: no retained-receipt store — cannot self-assess challenge on "
                "batch %s.", batch)
            return
        try:
            leaf = bytes.fromhex(leaf_hex[2:] if leaf_hex.startswith("0x") else leaf_hex)
        except Exception:  # noqa: BLE001 — a malformed leaf is not actionable
            logger.warning(
                "auto-defense: challenge on batch %s has an unparseable leaf %r — skipping.",
                batch, leaf_hex)
            return
        record = store.get(leaf)
        if record is None:
            logger.warning(
                "auto-defense: no retained §7 receipt for challenged leaf %s (batch %s) — cannot "
                "self-assess (evicted, or not this node's batch).", leaf_hex, batch)
            _safe_verdict(on_verdict, challenge, None)
            return
        try:
            from prsm.settlement.challenge_verifier import (
                verify_inference_receipt_for_challenge,
            )
            report = verify_inference_receipt_for_challenge(
                record.inference_receipt,
                settler_public_key_b64=record.settler_public_key_b64,
                stage_public_keys=record.stage_public_keys)
        except Exception as exc:  # noqa: BLE001 — never let self-verify crash the watcher
            logger.warning(
                "auto-defense: self-verify errored for leaf %s (batch %s): %s",
                leaf_hex, batch, exc)
            return
        if getattr(report, "receipt_ok", False):
            logger.warning(
                "auto-defense: challenge on batch %s appears BAD-FAITH — the retained §7 receipt "
                "for leaf %s VERIFIES. It is served at GET /settlement/receipt/leaf/%s for "
                "independent verification; no stake should be lost.", batch, leaf_hex, leaf_hex)
        else:
            proven = [f for f in getattr(report, "findings", []) if getattr(f, "proven", False)]
            grounds = ", ".join(str(getattr(f, "reason", "?")) for f in proven) or "unknown"
            logger.error(
                "auto-defense: challenge on batch %s is LEGITIMATE — the retained §7 receipt for "
                "leaf %s FAILS verification (grounds: %s). Stake is at risk; contesting is futile.",
                batch, leaf_hex, grounds)
        _safe_verdict(on_verdict, challenge, report)

    return _defend


def _safe_verdict(cb: Optional[Callable[[Any, Any], None]], challenge: Any, report: Any) -> None:
    if cb is None:
        return
    try:
        cb(challenge, report)
    except Exception as exc:  # noqa: BLE001 — a bad hook must not break the watcher
        logger.warning("auto-defense on_verdict callback failed: %s", exc)
