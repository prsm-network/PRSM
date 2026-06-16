"""Active challenge/dispute WATCHER (challenge/dispute brick 5).

Bricks 1-4 are pure, injectable, one-module-per-concern:
  - challenge_verifier.verify_inference_receipt_for_challenge(...) -> ChallengeReport
        (.on_chain_actionable() = proven findings that map to a BatchSettlementRegistry
        ReasonCode, e.g. INVALID_SIGNATURE == 1);
  - challenge_assembler.assemble_invalid_signature_challenge(...) -> InvalidSignatureChallenge
        (the concrete challengeReceipt(batchId, leaf, merkleProof, ReasonCode, auxData) inputs);
  - invalid_signature_submitter.InvalidSignatureChallengeSubmitter
        (.dry_run(...) = READ-ONLY static eth_call; .submit(...) = USER-GATED broadcast).

THE GAP this brick closes: nothing AUTOMATICALLY runs the verifier against committed
batches, so slashable receipts were only caught by a manual check. This watcher makes the
dispute mechanism an ACTIVE monitor — it pulls units from a pluggable async source,
verifies each, and for every on-chain-actionable finding it ASSEMBLES + DRY-RUNS the
challenge and SURFACES it for an operator to act on.

HARD SECURITY INVARIANTS (this is the money/security path):
  - The watcher NEVER broadcasts / signs / spends gas / slashes. It only verifies +
    DRY-RUNS (static eth_call) + surfaces actionable challenges. It NEVER calls the
    submitter's ``submit`` (the user-gated broadcast). The actual broadcast stays separate
    and user-gated.
  - Fail-closed, never-raise: a verifier/assembler/dry-run error on ONE unit is recorded as
    an error entry and MUST NOT abort the whole scan or crash the node.
  - The SOURCE of receipts (which committed batches + their off-chain receipts/leaves) is a
    SEPARATE concern (on-chain batches store only a merkle root, not the leaves). It is a
    PLUGGABLE injected async source so the watcher is correct regardless of where receipts
    ultimately come from (local state store, gossip, DHT). This keeps the watcher PURE +
    unit-testable with a fake source.

The watcher ORCHESTRATES the three pure modules; it does not reimplement verify / assemble
/ dry-run logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
)

from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.challenge_assembler import (
    REASON_INVALID_SIGNATURE,
    InvalidSignatureChallenge,
    assemble_invalid_signature_challenge,
)
from prsm.settlement.challenge_verifier import (
    ChallengeFinding,
    verify_inference_receipt_for_challenge,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WatchUnit:
    """One unit to check, carrying everything the pure bricks need.

    The off-chain inference receipt + settler key feed the VERIFIER; the order-preserved
    committed batch receipts + the target index feed the ASSEMBLER (which builds the merkle
    proof against the batch's leaf order). These are distinct off-chain artifacts (a §7
    InferenceReceipt vs the ShardExecutionReceipts the batch root was built over), so the
    unit binds them together for the watcher. ``stage_public_keys`` is optional and only
    affects the (off-chain) activation-chain checks — the on-chain-actionable
    INVALID_SIGNATURE ground does not need it."""

    batch_id: bytes
    inference_receipt: Any
    settler_public_key_b64: str
    batch_receipts: List[BatchedReceipt]
    target_index: int
    stage_public_keys: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class ActionableChallenge:
    """A surfaced, ASSEMBLED + DRY-RUN challenge an operator could choose to broadcast.

    ``challenge`` is the concrete challengeReceipt inputs (leaf + merkle proof + auxData).
    ``dry_run_ok`` is the read-only verdict — True means the on-chain call would currently
    be accepted; False (with ``dry_run_revert_reason``) means the operator sees the
    actionable fraud but knows the on-chain call would currently revert. Broadcasting it
    remains a SEPARATE, user-gated step — the watcher never does it."""

    batch_id: bytes
    target_index: int
    finding: ChallengeFinding
    challenge: InvalidSignatureChallenge
    dry_run_ok: bool
    dry_run_revert_reason: Optional[str] = None

    @property
    def on_chain_reason(self) -> Optional[int]:
        return self.finding.on_chain_reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": "0x" + bytes(self.batch_id).hex(),
            "target_index": self.target_index,
            "finding": self.finding.to_dict(),
            "reason_code": self.challenge.reason_code,
            "dry_run_ok": self.dry_run_ok,
            "dry_run_revert_reason": self.dry_run_revert_reason,
        }


@dataclass(frozen=True)
class ScanError:
    """A per-unit failure recorded WITHOUT aborting the scan (fail-closed)."""

    batch_id: Optional[bytes]
    target_index: Optional[int]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": (
                "0x" + bytes(self.batch_id).hex() if self.batch_id is not None else None
            ),
            "target_index": self.target_index,
            "message": self.message,
        }


@dataclass(frozen=True)
class ScanResult:
    """The structured outcome of a scan: surfaced actionable challenges + per-unit errors.
    Clean receipts contribute nothing."""

    actionable: List[ActionableChallenge] = field(default_factory=list)
    errors: List[ScanError] = field(default_factory=list)
    units_scanned: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "units_scanned": self.units_scanned,
            "actionable": [a.to_dict() for a in self.actionable],
            "errors": [e.to_dict() for e in self.errors],
        }


# A pluggable async receipt source: a zero-arg callable returning an async iterator of
# WatchUnits. Injected so the watcher never hard-wires a network source.
ReceiptSource = Callable[[], AsyncIterator[WatchUnit]]


class ChallengeWatcher:
    """Active monitor that turns the off-chain dispute mechanism into a scanner.

    Injected dependencies (keeps the watcher pure + unit-testable):
      - ``source``: a zero-arg callable yielding WatchUnits (local store / gossip / DHT —
        the watcher does not care);
      - ``dry_run_client``: an object exposing ``dry_run(InvalidSignatureChallenge) ->
        DryRunResult`` (the InvalidSignatureChallengeSubmitter, or a fake). Its
        user-gated ``submit`` is NEVER called from here.
    """

    def __init__(self, *, source: ReceiptSource, dry_run_client: Any) -> None:
        self._source = source
        self._dry_run_client = dry_run_client

    async def scan(self) -> ScanResult:
        """Pull every unit from the source, verify it, and for each on-chain-actionable
        finding assemble + DRY-RUN the challenge and surface it. Per-unit errors are caught
        and recorded; the scan never raises and never broadcasts."""
        actionable: List[ActionableChallenge] = []
        errors: List[ScanError] = []
        units_scanned = 0

        async for unit in self._source():
            units_scanned += 1
            try:
                self._scan_unit(unit, actionable, errors)
            except Exception as exc:  # noqa: BLE001 — fail-closed: one bad unit never
                # aborts the scan or crashes the node.
                logger.warning(
                    "challenge-watcher: unit (batch=%s, idx=%s) failed: %s: %s",
                    _hex(getattr(unit, "batch_id", None)),
                    getattr(unit, "target_index", None),
                    type(exc).__name__, exc,
                )
                errors.append(ScanError(
                    batch_id=getattr(unit, "batch_id", None),
                    target_index=getattr(unit, "target_index", None),
                    message=f"{type(exc).__name__}: {exc}",
                ))

        result = ScanResult(
            actionable=actionable, errors=errors, units_scanned=units_scanned)
        logger.info(
            "challenge-watcher scan complete: %d unit(s) scanned, %d actionable "
            "challenge(s) surfaced (%d dry-run-OK / %d would-revert), %d error(s). "
            "NOTE: nothing was broadcast — broadcasting is a separate user-gated step.",
            units_scanned, len(actionable),
            sum(1 for a in actionable if a.dry_run_ok),
            sum(1 for a in actionable if not a.dry_run_ok),
            len(errors),
        )
        return result

    def _scan_unit(
        self, unit: WatchUnit, actionable: List[ActionableChallenge],
        errors: List[ScanError],
    ) -> None:
        """Verify one unit and append any surfaced ActionableChallenges. Raises on a bad
        unit (caught + recorded by ``scan``)."""
        report = verify_inference_receipt_for_challenge(
            unit.inference_receipt,
            settler_public_key_b64=unit.settler_public_key_b64,
            stage_public_keys=unit.stage_public_keys,
        )
        # Clean receipt → nothing surfaced.
        for finding in report.on_chain_actionable():
            # FAIL-CLOSED: the assembler only produces INVALID_SIGNATURE (reason 1)
            # challenges. on_chain_actionable() returns only findings whose
            # on_chain_reason is set, and today INVALID_SIGNATURE is the SOLE such
            # ground — but if a future ChallengeReason is given a different on-chain
            # reason code, blindly assembling it as INVALID_SIGNATURE would mis-encode
            # the dispute. Skip it as a recorded error rather than surface a wrong
            # challenge; a valid INVALID_SIGNATURE finding in the same unit still
            # surfaces (per-finding, not whole-unit).
            if finding.on_chain_reason != REASON_INVALID_SIGNATURE:
                errors.append(ScanError(
                    batch_id=unit.batch_id,
                    target_index=unit.target_index,
                    message=(
                        f"unsupported on-chain reason {finding.on_chain_reason} "
                        f"({finding.reason.value}); the watcher only assembles "
                        f"INVALID_SIGNATURE ({REASON_INVALID_SIGNATURE}) — not surfaced"
                    ),
                ))
                continue
            # Assemble the concrete challengeReceipt inputs (leaf + proof + auxData).
            challenge = assemble_invalid_signature_challenge(
                batch_id=unit.batch_id,
                batch_receipts=unit.batch_receipts,
                target_index=unit.target_index,
            )
            # DRY-RUN ONLY — read-only static eth_call. NEVER submit()/broadcast.
            dry = self._dry_run_client.dry_run(challenge)
            actionable.append(ActionableChallenge(
                batch_id=unit.batch_id,
                target_index=unit.target_index,
                finding=finding,
                challenge=challenge,
                dry_run_ok=bool(dry.would_succeed),
                dry_run_revert_reason=getattr(dry, "revert_reason", None),
            ))


def _hex(b: Optional[bytes]) -> Optional[str]:
    return "0x" + bytes(b).hex() if b is not None else None
