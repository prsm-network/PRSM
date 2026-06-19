"""Sprint 1166 — user-gated challenge BROADCAST (the capstone of the fraud-defense loop).

The prepare bridge (challenge_preparer, sp1163-1165) is deliberately KEYLESS + never-broadcast:
it assembles + read-only-dry-runs a challenge and stops at the exact ``challengeReceipt`` call
args. This module is the FINAL step — actually filing a prepared challenge — as an EXPLICITLY
gated action. Broadcasting a challenge spends gas + slashes a provider's staked bond, so per
the standing rule the ASSISTANT never broadcasts: it builds this tool; the OPERATOR runs it
with their OWN key.

``gated_broadcast_prepared`` is the testable core:
  - it ALWAYS runs the read-only dry-run first (the verify-before-broadcast pre-flight);
  - it broadcasts ONLY when ``broadcast_enabled`` is True AND (the dry-run passed OR ``force``);
  - otherwise it NEVER calls the submitter's ``submit`` — the default (broadcast_enabled=False)
    is read-only, and a failed dry-run refuses to broadcast (anti-griefing: a would-revert
    challenge wastes gas + can't slash anyway).

The ``submitter`` is INJECTED (the CLI builds the keyed one — ChallengeSubmitter for reasons
0-3, ConsensusChallengeSubmitter for reason 5; both expose ``dry_run`` + ``submit`` over a
prepared challenge's ``to_call_args()``). This core never builds a submitter, never holds a
key, and never reaches a sign/broadcast path unless explicitly enabled by its caller.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["BroadcastOutcome", "gated_broadcast_prepared"]


@dataclass(frozen=True)
class BroadcastOutcome:
    """The outcome of a gated broadcast attempt. ``broadcast`` is whether a tx was actually
    sent; ``refused_reason`` explains a non-broadcast (read-only / dry-run-failed). Always
    carries the read-only dry-run verdict so the operator sees the pre-flight result."""

    broadcast: bool
    dry_run_ok: Optional[bool]
    dry_run_reason: Optional[str]
    tx_hash_hex: Optional[str] = None
    refused_reason: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "broadcast": self.broadcast,
            "dry_run_ok": self.dry_run_ok,
            "dry_run_reason": self.dry_run_reason,
            "tx_hash_hex": self.tx_hash_hex,
            "refused_reason": self.refused_reason,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


def gated_broadcast_prepared(
    prepared: Any,
    *,
    submitter: Any,
    broadcast_enabled: bool,
    require_dry_run_ok: bool = True,
    force: bool = False,
) -> BroadcastOutcome:
    """Dry-run a prepared challenge and, ONLY if explicitly enabled + safe, broadcast it.

    Order (fail-closed):
      1. ALWAYS dry-run first (read-only static call) — capture the verdict.
      2. If ``broadcast_enabled`` is False -> refuse (read-only mode); NEVER call submit.
      3. If the dry-run failed and ``require_dry_run_ok`` and not ``force`` -> refuse; NEVER
         call submit (a would-revert challenge wastes gas + can't slash).
      4. Otherwise broadcast via ``submitter.submit(prepared.challenge)`` (USER-GATED — the
         caller enabled it with the operator's key).

    ``submitter`` must expose ``dry_run(challenge)`` (-> verdict with ``.would_succeed`` /
    ``.revert_reason``) and ``submit(challenge)`` (-> result with ``.tx_hash_hex`` / ``.success``
    / ``.error_*``). NEVER raises — this core ENFORCES the invariant itself (it wraps the
    injected dry_run/submit calls), so even a non-conforming submitter yields a clean,
    fail-closed outcome rather than an exception escaping a bond-slashing path."""
    challenge = prepared.challenge
    # Dry-run is the verify-before-broadcast pre-flight. A dry_run that ERRORS is treated as a
    # FAILED pre-flight (fail-closed): the broadcast gate below then refuses unless --force.
    try:
        verdict = submitter.dry_run(challenge)
        dry_ok = bool(getattr(verdict, "would_succeed", False))
        dry_reason = getattr(verdict, "revert_reason", None)
    except Exception as exc:  # noqa: BLE001 — enforce never-raise; a bad dry_run fails closed
        dry_ok = False
        dry_reason = f"dry-run raised {type(exc).__name__}: {exc}"

    if not broadcast_enabled:
        return BroadcastOutcome(
            broadcast=False, dry_run_ok=dry_ok, dry_run_reason=dry_reason,
            refused_reason="broadcast not enabled (read-only / dry-run only)",
        )

    if require_dry_run_ok and not dry_ok and not force:
        return BroadcastOutcome(
            broadcast=False, dry_run_ok=dry_ok, dry_run_reason=dry_reason,
            refused_reason=(
                f"dry-run failed ({dry_reason}); refusing to broadcast a challenge that "
                "would revert (use force only if you are certain the dry-run is a false "
                "negative — e.g. RPC lag)"
            ),
        )

    # USER-GATED broadcast — the caller enabled it with the operator's key.
    logger.warning(
        "challenge_broadcaster: BROADCASTING a %s challenge for batch %s (reason %s) — this "
        "spends gas and may slash a provider's bond.",
        getattr(prepared, "reason", "?"), getattr(prepared, "batch_id_hex", "?"),
        getattr(prepared, "on_chain_reason", "?"),
    )
    try:
        result = submitter.submit(challenge)
    except Exception as exc:  # noqa: BLE001 — enforce never-raise even on a non-conforming
        # submitter; a broadcast that errored out of submit() is reported as NOT broadcast.
        logger.error(
            "challenge_broadcaster: submit raised %s: %s", type(exc).__name__, exc)
        return BroadcastOutcome(
            broadcast=False, dry_run_ok=dry_ok, dry_run_reason=dry_reason,
            refused_reason="submit raised", error_type=type(exc).__name__,
            error_message=str(exc),
        )
    return BroadcastOutcome(
        broadcast=bool(getattr(result, "success", False)),
        dry_run_ok=dry_ok,
        dry_run_reason=dry_reason,
        tx_hash_hex=getattr(result, "tx_hash_hex", None),
        refused_reason=None,
        error_type=getattr(result, "error_type", None),
        error_message=getattr(result, "error_message", None),
    )
