"""Sprint 292 — privacy-claim verification public API.

Vision §7 says PRSM provides "TEE-attested compute" as a
structural property of the protocol. The local executor
infrastructure for this is in place (sprint 287-288 verified
DPNoiseInjector is wired + PrivacyBudgetTracker enforces ε),
but every receipt produced today carries a
``DEV-ONLY-SW-TEE:``-prefixed software-stub attestation. Until
real hardware-attestation backends (Intel ASP / AMD KDS /
Apple SEP) wire in, end-users have no way to distinguish a
production-attested receipt from the software fallback.

This module exposes that truth via a public API so callers
can make trust decisions:

  is_dev_only_attestation(blob)
    Public predicate — matches the docstring intent at
    executor.py:602-606 ("verifiers MUST reject any
    attestation starting with the DEV-ONLY prefix as a
    confidentiality proof").

  verify_receipt_privacy_claim(receipt, *,
        require_hardware_attestation=False,
        require_dp_noise=False,
        identity=None, public_key_b64=None)
    Composite check returning a PrivacyVerification record.
    Validates signature + DP-noise + hardware-attestation
    quality + multi-stage envelope presence. Surfaces
    expected vs actual epsilon mismatch for callers that
    need to detect tier↔ε desync.

The HTTP endpoint + MCP tool wrapping this primitive ship
in sprint 293 (out of sprint-292 scope). This sprint focuses
on the pure verification surface that those wrappers consume.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from prsm.compute.inference.executor import (
    SOFTWARE_TEE_ATTESTATION_PREFIX,
)
from prsm.compute.inference.receipt import (
    InferenceReceipt, verify_receipt,
)
from prsm.compute.tee.models import PrivacyLevel


def is_dev_only_attestation(blob: Optional[bytes]) -> bool:
    """True iff ``blob`` carries the DEV-ONLY software-fallback
    prefix. Use as a hard predicate: any True return means the
    attestation is NOT a production confidentiality proof,
    regardless of what the receipt's ``tee_type`` field claims.

    Defensive: None, empty, and short-bytes inputs return
    False (they're "missing", not "dev-only"). Callers that
    require a hardware attestation should separately check
    that the blob is present + sufficient length.
    """
    if blob is None or not isinstance(blob, bytes):
        return False
    return blob.startswith(SOFTWARE_TEE_ATTESTATION_PREFIX)


@dataclass
class PrivacyVerification:
    """Composite result of verifying a receipt's privacy
    claims. ``ok=True`` means every check the caller asked
    for passed; ``reasons`` enumerates the failures + any
    informational warnings."""

    ok: bool
    reasons: List[str] = field(default_factory=list)
    signature_valid: bool = False
    dp_noise_applied: bool = False
    hardware_attested: bool = False
    multi_stage_envelope_present: bool = False
    # sp1236-B — real per-stage verification verdict for a multi-stage chain:
    # None = not a multi-stage envelope OR no attestation_registry supplied
    # (structural-only, back-compat); True/False = every stage cryptographically
    # vendor_verified + REPORT_DATA-bound to its node (fail-closed aggregate).
    multi_stage_verified: Optional[bool] = None
    # Diagnostic fields (always populated for caller
    # inspection; surfacing the values is half the point of
    # this API).
    privacy_tier: str = ""
    epsilon_spent: float = 0.0
    expected_epsilon: float = 0.0
    # Sprint 293 — backend-supplied attestation detail.
    # `attestation_vendor` is the result of running the
    # attestation blob through the sprint-293 backend
    # registry (intel-sgx / intel-tdx / amd-sev-snp /
    # software-fallback / unknown). `attestation_vendor_data`
    # carries vendor-specific parsed fields (MRENCLAVE_hex,
    # MRSIGNER_hex, TCB level, etc.) so callers can pin
    # against expected values out-of-band.
    attestation_vendor: str = "unknown"
    attestation_vendor_data: Dict[str, Any] = field(
        default_factory=dict,
    )
    attestation_vendor_verified: bool = False
    # Sprint 297 — §7 capstone integrity fields. True iff
    # the receipt carried a structurally + semantically valid
    # ActivationNoiseTrace / TopologyAssignment. Absent
    # field → True by default (nothing to check); strict
    # callers pass require_* flags to flip ok=False when the
    # field is required but missing/invalid.
    activation_noise_trace_valid: bool = True
    topology_structurally_valid: bool = True
    topology_distinct_from_history: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "reasons": list(self.reasons),
            "signature_valid": self.signature_valid,
            "dp_noise_applied": self.dp_noise_applied,
            "hardware_attested": self.hardware_attested,
            "multi_stage_envelope_present": (
                self.multi_stage_envelope_present
            ),
            "multi_stage_verified": self.multi_stage_verified,
            "privacy_tier": self.privacy_tier,
            "epsilon_spent": self.epsilon_spent,
            "expected_epsilon": self.expected_epsilon,
            "attestation_vendor": self.attestation_vendor,
            "attestation_vendor_data": dict(
                self.attestation_vendor_data,
            ),
            "attestation_vendor_verified": (
                self.attestation_vendor_verified
            ),
            "activation_noise_trace_valid": (
                self.activation_noise_trace_valid
            ),
            "topology_structurally_valid": (
                self.topology_structurally_valid
            ),
            "topology_distinct_from_history": (
                self.topology_distinct_from_history
            ),
        }


def _privacy_tier_str(tier: Any) -> str:
    """Coerce the receipt's privacy_tier (might be enum or
    string depending on round-trip path) to a string."""
    if hasattr(tier, "value"):
        return tier.value
    return str(tier)


def _expected_epsilon_for(tier: Any) -> float:
    """Look up the canonical ε for the tier. Returns 0.0 for
    NONE, inf for unrecognized tiers (caller should still
    surface, but won't trigger a mismatch reason)."""
    try:
        if isinstance(tier, PrivacyLevel):
            level = tier
        else:
            level = PrivacyLevel(_privacy_tier_str(tier))
    except ValueError:
        return float("inf")
    if level == PrivacyLevel.NONE:
        return 0.0
    return PrivacyLevel.config_for_level(level).epsilon


def verify_receipt_privacy_claim(
    receipt: InferenceReceipt,
    *,
    require_hardware_attestation: bool = False,
    require_dp_noise: bool = False,
    require_activation_dp_trace: bool = False,
    require_topology_rotation: bool = False,
    identity: Any = None,
    public_key_b64: Optional[str] = None,
    topology_history: Any = None,
    expected_anti_repeat_window: int = 3,
    attestation_registry: Any = None,
) -> PrivacyVerification:
    """Run all privacy-claim checks against a receipt.

    Default posture (no ``require_*`` flags): permissive —
    returns ok=True for any receipt that has a valid
    signature, regardless of hardware-attestation quality or
    DP-noise presence. The diagnostic fields surface the
    truth so callers can apply their own policy.

    Strict posture: callers who require hardware attestation
    or DP noise pass the respective ``require_*`` flag, and
    a failure on any required check flips ``ok=False`` with
    a clear reason.
    """
    reasons: List[str] = []
    tier_str = _privacy_tier_str(receipt.privacy_tier)
    expected_eps = _expected_epsilon_for(receipt.privacy_tier)

    # ── Signature check ─────────────────────────────────
    signature_valid = False
    if not receipt.settler_signature:
        reasons.append(
            "receipt is unsigned (settler_signature empty)"
        )
    elif identity is None and public_key_b64 is None:
        reasons.append(
            "no verifier key supplied (pass identity= or "
            "public_key_b64=)"
        )
    else:
        try:
            signature_valid = verify_receipt(
                receipt,
                identity=identity,
                public_key_b64=public_key_b64,
            )
        except Exception as exc:  # noqa: BLE001
            reasons.append(
                f"signature verification raised: {exc}"
            )
        if not signature_valid and (
            identity is not None or public_key_b64 is not None
        ):
            # Add the reason only if we actually tried and
            # the result was a clean False (not from a missing
            # input — that already added its own reason).
            if receipt.settler_signature:
                # Avoid duplicating the "unsigned" reason
                already_reported = any(
                    "verifi" in r for r in reasons
                )
                if not already_reported:
                    reasons.append(
                        "signature failed cryptographic "
                        "verification"
                    )

    # ── DP-noise applied check ──────────────────────────
    # Convention: epsilon_spent > 0 iff DP noise was applied.
    # For privacy_tier=NONE this is correctly False.
    dp_noise_applied = receipt.epsilon_spent > 0.0
    if require_dp_noise and not dp_noise_applied:
        reasons.append(
            "DP noise was NOT applied "
            f"(epsilon_spent={receipt.epsilon_spent}; "
            f"privacy_tier={tier_str!r})"
        )

    # Surface tier↔ε mismatch as a non-fatal informational
    # reason. Caller decides what to do with the diagnostic.
    if (
        dp_noise_applied
        and expected_eps not in (0.0, float("inf"))
        and abs(receipt.epsilon_spent - expected_eps) > 1e-9
    ):
        reasons.append(
            f"epsilon_spent mismatch with privacy_tier: "
            f"expected ε={expected_eps} for "
            f"tier={tier_str!r}, got "
            f"ε={receipt.epsilon_spent}"
        )

    # ── Attestation quality ─────────────────────────────
    attestation = receipt.tee_attestation
    is_dev_only = is_dev_only_attestation(attestation)
    has_attestation = bool(attestation)
    hardware_attested = has_attestation and not is_dev_only
    if require_hardware_attestation:
        if not has_attestation:
            reasons.append(
                "no TEE attestation present "
                "(tee_attestation is empty)"
            )
        elif is_dev_only:
            reasons.append(
                "attestation is DEV-ONLY software fallback "
                "— not a production confidentiality proof. "
                "Hardware TEE backends (Intel ASP / AMD KDS "
                "/ Apple SEP) ship in a future sprint."
            )

    # Sprint 293 — run the attestation through the backend
    # registry to get vendor + parsed fields. Defensive: any
    # backend failure surfaces as vendor="unknown" with an
    # error reason but doesn't crash the primary
    # verification.
    attestation_vendor = "unknown"
    attestation_vendor_data: Dict[str, Any] = {}
    attestation_vendor_verified = False
    if has_attestation:
        try:
            from prsm.compute.inference.attestation_backends import (  # noqa: E501
                verify_attestation as _verify_attest,
            )
            backend_result = _verify_attest(attestation)
            attestation_vendor = backend_result.vendor
            attestation_vendor_data = (
                backend_result.vendor_data
            )
            attestation_vendor_verified = (
                backend_result.vendor_verified
            )
            if backend_result.error and (
                require_hardware_attestation
            ):
                reasons.append(
                    f"attestation backend: "
                    f"{backend_result.error}"
                )
        except Exception as exc:  # noqa: BLE001
            attestation_vendor = "unknown"
            if require_hardware_attestation:
                reasons.append(
                    f"attestation backend raised: {exc}"
                )

    # ── Multi-stage envelope presence ───────────────────
    # Phase 3.x.7 RpcChainExecutor wraps single-stage
    # attestation with a multi-stage envelope. Detect that
    # here so callers can surface "this inference used the
    # cross-host attestation path."
    multi_stage_present = False
    multi_stage_verified: Optional[bool] = None
    try:
        from prsm.compute.inference.multi_stage_attestation import (
            is_multi_stage_attestation as _is_ms,
            verify_stage_attestations as _ms_verify,
        )
        multi_stage_present = _is_ms(attestation) and not is_dev_only
        # sp1236-B — when the caller supplies the real backend registry, run
        # every stage of the proven cross-host chain through cryptographic
        # verification + REPORT_DATA->node binding (sp1236), instead of the
        # permissive "non-empty + not-DEV-ONLY" heuristic. No registry → still
        # structural-only (multi_stage_verified stays None), back-compat.
        if multi_stage_present and attestation_registry is not None:
            _exp_count = None
            _topo = getattr(receipt, "topology_assignment", None)
            _sc = getattr(_topo, "stage_count", None) if _topo is not None else None
            if isinstance(_sc, int) and _sc > 0:
                _exp_count = _sc
            _all_ok, _ = _ms_verify(
                attestation,
                registry=attestation_registry,
                expected_stage_count=_exp_count,
            )
            multi_stage_verified = bool(_all_ok)
            if require_hardware_attestation and not multi_stage_verified:
                reasons.append(
                    "multi-stage attestation failed cryptographic "
                    "verification / node-binding"
                )
    except Exception as exc:  # noqa: BLE001
        multi_stage_present = False
        if require_hardware_attestation and attestation_registry is not None:
            reasons.append(f"multi-stage attestation check raised: {exc}")

    # sp1236-B — when the multi-stage chain was really verified, that verdict
    # (not the permissive heuristic) is the hardware-attestation signal.
    if multi_stage_verified is not None:
        hardware_attested = multi_stage_verified

    # ── Sprint 297: activation_noise_trace integrity ────
    activation_noise_trace_valid = True  # default-true when absent
    trace = getattr(
        receipt, "activation_noise_trace", None,
    )
    tier_enum: Any = receipt.privacy_tier
    if not hasattr(tier_enum, "value"):
        # Coerce string back to enum for the verifier
        try:
            from prsm.compute.tee.models import (
                PrivacyLevel as _PL,
            )
            tier_enum = _PL(str(tier_enum))
        except Exception:  # noqa: BLE001
            tier_enum = None
    if trace is not None:
        try:
            from prsm.compute.inference.activation_dp import (
                verify_activation_noise_trace,
            )
            if tier_enum is None:
                trace_ok, trace_reason = (
                    False, "could not resolve tier enum"
                )
            else:
                trace_ok, trace_reason = (
                    verify_activation_noise_trace(
                        trace, expected_tier=tier_enum,
                    )
                )
            activation_noise_trace_valid = trace_ok
            if not trace_ok:
                reasons.append(
                    f"activation_noise_trace invalid: "
                    f"{trace_reason}"
                )
            else:
                # sp1001 — cross-check the receipt's claimed epsilon_spent against
                # the trace's recorded total. The honest executor injects the full
                # tier ε on every stage and sets epsilon_spent to the same tier
                # ceiling, so epsilon_spent == trace total on an honest receipt. A
                # divergence means the CHARGED DP budget and the ACTUALLY-INJECTED
                # noise disagree — a privacy over-claim (e.g. epsilon_spent=8.0
                # charged while the signed trace records 0.0 noise). Both fields are
                # signed, so this is the operator self-signing a contradictory pair.
                _claimed = float(receipt.epsilon_spent)
                _traced = float(
                    getattr(trace, "total_epsilon_spent", 0.0) or 0.0
                )
                if abs(_claimed - _traced) > 1e-6:
                    activation_noise_trace_valid = False
                    reasons.append(
                        f"epsilon_spent ({_claimed}) disagrees with the "
                        f"activation_noise_trace total ({_traced}) — the charged "
                        f"DP budget and the injected activation noise are "
                        f"inconsistent (privacy over-claim)"
                    )
        except Exception as exc:  # noqa: BLE001
            activation_noise_trace_valid = False
            reasons.append(
                f"activation_noise_trace check raised: "
                f"{exc}"
            )
    elif require_activation_dp_trace:
        activation_noise_trace_valid = False
        reasons.append(
            "activation_noise_trace missing on receipt "
            "(require_activation_dp_trace=True)"
        )

    # ── Sprint 297: topology_assignment integrity ───────
    topology_structurally_valid = True  # default-true when absent
    topology_distinct_from_history = True
    topo = getattr(receipt, "topology_assignment", None)
    if topo is not None:
        try:
            from prsm.compute.inference.topology_rotation import (
                verify_topology_sequence,
            )
            # Structural check: a single-element sequence,
            # zero rotation window. Catches dup nodes +
            # missing positions.
            struct_ok, struct_reason = (
                verify_topology_sequence(
                    [topo],
                    expected_anti_repeat_window=0,
                )
            )
            topology_structurally_valid = struct_ok
            if not struct_ok:
                reasons.append(
                    f"topology structurally invalid: "
                    f"{struct_reason}"
                )
            # History distinctness check (if history supplied)
            if topology_history is not None:
                try:
                    in_history = topology_history.contains(
                        topo,
                    )
                except Exception:  # noqa: BLE001
                    in_history = False
                topology_distinct_from_history = (
                    not in_history
                )
                if in_history:
                    reasons.append(
                        "topology repeats an entry in "
                        "supplied history (rotation violated)"
                    )
        except Exception as exc:  # noqa: BLE001
            topology_structurally_valid = False
            reasons.append(
                f"topology check raised: {exc}"
            )
    elif require_topology_rotation:
        topology_structurally_valid = False
        reasons.append(
            "topology_assignment missing on receipt "
            "(require_topology_rotation=True)"
        )

    # ── Final ok ─────────────────────────────────────────
    # ok = signature_valid AND (no failed required checks)
    failed_required = False
    if require_hardware_attestation and not hardware_attested:
        failed_required = True
    if require_dp_noise and not dp_noise_applied:
        failed_required = True
    # sp1001 — a PRESENT but detectably-INVALID activation_noise_trace is FATAL on
    # EVERY path, not only when require_activation_dp_trace=True. The /compute/
    # receipt/verify endpoint, the MCP tool, and the SDK all default that flag
    # OFF, so previously a receipt whose own signed trace contradicted its
    # privacy-tier/epsilon claim (an over-claim: tier↔ε desync, charged-ε vs
    # injected-noise mismatch, or zero-noise on a private tier) still returned
    # ok=True. An HONEST trace always validates (the executor injects the full
    # tier ε every stage), so this never false-rejects honest receipts. NOTE:
    # activation_noise_trace_valid is False ONLY when a trace was present and
    # failed, OR when it is absent AND require_activation_dp_trace=True — both are
    # correctly fatal; an absent trace with the flag off keeps it True (no-op).
    if not activation_noise_trace_valid:
        failed_required = True
    if require_topology_rotation and (
        not topology_structurally_valid
        or not topology_distinct_from_history
    ):
        failed_required = True
    ok = signature_valid and not failed_required

    return PrivacyVerification(
        ok=ok,
        reasons=reasons,
        signature_valid=signature_valid,
        dp_noise_applied=dp_noise_applied,
        hardware_attested=hardware_attested,
        multi_stage_envelope_present=multi_stage_present,
        multi_stage_verified=multi_stage_verified,
        privacy_tier=tier_str,
        epsilon_spent=float(receipt.epsilon_spent),
        expected_epsilon=(
            expected_eps if expected_eps != float("inf") else 0.0
        ),
        attestation_vendor=attestation_vendor,
        attestation_vendor_data=attestation_vendor_data,
        attestation_vendor_verified=(
            attestation_vendor_verified
        ),
        activation_noise_trace_valid=(
            activation_noise_trace_valid
        ),
        topology_structurally_valid=(
            topology_structurally_valid
        ),
        topology_distinct_from_history=(
            topology_distinct_from_history
        ),
    )
