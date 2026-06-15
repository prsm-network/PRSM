"""Off-chain challenge/dispute VERIFIER for §7 inference receipts.

The on-chain BatchSettlementRegistry already has the challenge machinery
(challengeReceipt with reason codes DOUBLE_SPEND / INVALID_SIGNATURE / NO_ESCROW /
EXPIRED / CONSENSUS_MISMATCH, and StakeBond slashing on a proven challenge). What was
missing is the OFF-CHAIN side: a verifier that, given a receipt and the relevant public
keys, recomputes the verify-time checks and produces an actionable verdict — "is there
provable fraud in this receipt, on what grounds, and is it on-chain-actionable?" — so a
would-be challenger (or an auditor / reputation system) can decide whether to act.

This module composes the existing verify-time primitives:
  - the settler signature over InferenceReceipt.signing_payload (receipt.verify_receipt),
    which maps directly to the on-chain INVALID_SIGNATURE reason; and
  - the per-stage signed activation chain (sp1107-1113 stage_activation_proof.*): broken
    internal links, an invalid per-stage signature, a topology the signed stages don't
    support (F2), or a chain spliced from a different request — each an off-chain fraud
    ground (no single on-chain reason code maps 1:1, but they justify an off-chain
    reputation action and inform whether an on-chain INVALID_SIGNATURE / CONSENSUS_
    MISMATCH challenge is warranted).

It is intentionally PURE (no web3): it returns a verdict; assembling + broadcasting an
on-chain challengeReceipt transaction (leaf + merkleProof + auxData) is a follow-on brick
that consumes this verdict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ChallengeReason(str, Enum):
    """Grounds on which a receipt can be disputed. ``on_chain_reason`` (below) records
    which, if any, maps to a BatchSettlementRegistry ReasonCode."""

    INVALID_SETTLER_SIGNATURE = "invalid_settler_signature"
    INVALID_ACTIVATION_CHAIN = "invalid_activation_chain"        # broken links / bad sig
    ACTIVATION_TOPOLOGY_MISMATCH = "activation_topology_mismatch"  # F2 — asserted ≠ signed
    ACTIVATION_CHAIN_WRONG_REQUEST = "activation_chain_wrong_request"  # spliced chain


# The BatchSettlementRegistry ReasonCode that an off-chain ground maps to, when one
# exists (else None → off-chain reputation action only). INVALID_SIGNATURE == 1.
_ON_CHAIN_REASON: Dict[ChallengeReason, Optional[int]] = {
    ChallengeReason.INVALID_SETTLER_SIGNATURE: 1,  # ReasonCode.INVALID_SIGNATURE
    ChallengeReason.INVALID_ACTIVATION_CHAIN: None,
    ChallengeReason.ACTIVATION_TOPOLOGY_MISMATCH: None,
    ChallengeReason.ACTIVATION_CHAIN_WRONG_REQUEST: None,
}


@dataclass(frozen=True)
class ChallengeFinding:
    reason: ChallengeReason
    proven: bool          # True iff the fraud is cryptographically demonstrated here
    detail: str
    on_chain_reason: Optional[int] = None  # BatchSettlementRegistry ReasonCode, or None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason": self.reason.value,
            "proven": self.proven,
            "detail": self.detail,
            "on_chain_reason": self.on_chain_reason,
        }


@dataclass(frozen=True)
class ChallengeReport:
    """The verdict. ``receipt_ok`` is True iff NO ground was proven. ``findings`` lists
    every ground that was CHECKED (proven or not) so a caller sees what was + wasn't
    verifiable (e.g. an activation chain present but no stage keys supplied)."""

    receipt_ok: bool
    findings: List[ChallengeFinding] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def proven_findings(self) -> List[ChallengeFinding]:
        return [f for f in self.findings if f.proven]

    def on_chain_actionable(self) -> List[ChallengeFinding]:
        return [f for f in self.proven_findings() if f.on_chain_reason is not None]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_ok": self.receipt_ok,
            "findings": [f.to_dict() for f in self.findings],
            "notes": list(self.notes),
        }


def verify_inference_receipt_for_challenge(
    receipt: Any, *,
    settler_public_key_b64: str,
    stage_public_keys: Optional[Dict[str, str]] = None,
) -> ChallengeReport:
    """Run the off-chain dispute checks on an ``InferenceReceipt``. Returns a
    ChallengeReport: receipt_ok=False iff any ground is PROVEN.

    Checks:
      1. INVALID_SETTLER_SIGNATURE — the settler's Ed25519 signature must verify over the
         receipt's signing_payload under ``settler_public_key_b64``. A failure is the
         on-chain-actionable INVALID_SIGNATURE ground.
      2. The per-stage activation chain (when present): internal links + (with
         ``stage_public_keys``) per-stage signatures + topology cross-check + request_id
         binding. Each failure is an off-chain fraud ground.

    ``stage_public_keys`` is {node_id: pubkey_b64}; omit it to skip per-stage signature
    authenticity (internal-link + topology + request_id checks still run)."""
    from prsm.compute.inference.receipt import verify_receipt

    findings: List[ChallengeFinding] = []
    notes: List[str] = []

    # 1. Settler signature over the whole receipt (maps to on-chain INVALID_SIGNATURE).
    settler_ok = False
    try:
        settler_ok = verify_receipt(receipt, public_key_b64=settler_public_key_b64)
    except Exception as exc:  # noqa: BLE001 — verification must never raise into caller
        notes.append(f"settler signature verification raised: {exc!r}")
    findings.append(ChallengeFinding(
        reason=ChallengeReason.INVALID_SETTLER_SIGNATURE,
        proven=not settler_ok,
        detail=(
            "settler signature does not verify over the receipt's signing payload"
            if not settler_ok else "settler signature valid"
        ),
        on_chain_reason=_ON_CHAIN_REASON[ChallengeReason.INVALID_SETTLER_SIGNATURE],
    ))

    # 2. Per-stage signed activation chain (off-chain grounds).
    chain = getattr(receipt, "stage_activation_chain", None)
    if chain is not None:
        from prsm.compute.inference.stage_activation_proof import (
            summarize_stage_activation_chain,
        )
        summary = summarize_stage_activation_chain(
            chain,
            stage_public_keys=stage_public_keys,
            topology_assignment=getattr(receipt, "topology_assignment", None),
            expected_request_id=getattr(receipt, "request_id", None),
        )
        # internal links broken → spliced/reordered/substituted intermediate
        links_ok = bool(summary.get("internal_links_ok"))
        sigs_checked = bool(summary.get("signatures_checked"))
        sigs_ok = bool(summary.get("signatures_valid")) if sigs_checked else True
        chain_invalid = (not links_ok) or (sigs_checked and not sigs_ok)
        findings.append(ChallengeFinding(
            reason=ChallengeReason.INVALID_ACTIVATION_CHAIN,
            proven=chain_invalid,
            detail=(
                f"links_ok={links_ok}, signatures_checked={sigs_checked}, "
                f"signatures_valid={summary.get('signatures_valid')}; "
                f"{summary.get('internal_links_detail')}"
            ),
            on_chain_reason=_ON_CHAIN_REASON[ChallengeReason.INVALID_ACTIVATION_CHAIN],
        ))
        if not sigs_checked:
            notes.append(
                "activation chain present but stage_public_keys not supplied — per-stage "
                "signature authenticity NOT verified (internal links + topology + "
                "request_id were)"
            )
        # topology cross-check (F2) — only when a topology was asserted
        if "topology_consistent" in summary:
            findings.append(ChallengeFinding(
                reason=ChallengeReason.ACTIVATION_TOPOLOGY_MISMATCH,
                proven=not bool(summary.get("topology_consistent")),
                detail=str(summary.get("topology_detail")),
                on_chain_reason=_ON_CHAIN_REASON[
                    ChallengeReason.ACTIVATION_TOPOLOGY_MISMATCH],
            ))
        # request_id binding (anti-splice) — only when checked
        if "request_id_bound" in summary:
            findings.append(ChallengeFinding(
                reason=ChallengeReason.ACTIVATION_CHAIN_WRONG_REQUEST,
                proven=not bool(summary.get("request_id_bound")),
                detail=(
                    "chain request_id does not match the receipt's (spliced from a "
                    "different inference)"
                    if not summary.get("request_id_bound")
                    else "chain bound to this receipt's request_id"
                ),
                on_chain_reason=_ON_CHAIN_REASON[
                    ChallengeReason.ACTIVATION_CHAIN_WRONG_REQUEST],
            ))

    receipt_ok = not any(f.proven for f in findings)
    return ChallengeReport(receipt_ok=receipt_ok, findings=findings, notes=notes)
