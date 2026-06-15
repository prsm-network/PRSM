"""Sprint 305 — TEE-only execution policy primitive.

Vision §7 Enterprise Confidentiality Mode layer 3 (the
attestation-quality gate). Builds on the §7 attestation
backend registry (sprints 293-297) to give enterprises a
declarative policy:

  "this job must run on a real TEE, not software fallback"
  "this job must run on Intel SGX or AMD SEV-SNP only"
  "this job must run on a TEE with full cryptographic
   signature-chain verification (DCAP-equivalent)"

The policy operates on the existing
AttestationVerificationResult shape. Effective tier of an
attestation:

  unknown / errored      → NONE
  software-fallback      → SOFTWARE
  real vendor +
    vendor_verified=F    → HARDWARE_UNVERIFIED
  real vendor +
    vendor_verified=T    → HARDWARE_VERIFIED

Policy satisfaction: effective_tier >= min_attestation_tier
AND (allowed_vendors is None OR vendor in allowed_vendors).

The policy primitive is pure. Sprint 305a wires it into
the /compute/inference dispatch path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Set

from prsm.compute.inference.attestation_backends import (
    AttestationVerificationResult,
    verify_attestation,
)


# Hardware-vendor strings produced by the §7 registry.
_REAL_HARDWARE_VENDORS: frozenset = frozenset({
    "intel-sgx", "intel-tdx", "amd-sev-snp", "apple-sep",
})


class AttestationTier(str, Enum):
    NONE = "none"
    SOFTWARE = "software"
    HARDWARE_UNVERIFIED = "hardware_unverified"
    HARDWARE_VERIFIED = "hardware_verified"


_TIER_ORDER = (
    AttestationTier.NONE,
    AttestationTier.SOFTWARE,
    AttestationTier.HARDWARE_UNVERIFIED,
    AttestationTier.HARDWARE_VERIFIED,
)


def tier_rank(tier: AttestationTier) -> int:
    return _TIER_ORDER.index(tier)


class PolicyStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"


# ── TEEPolicy ─────────────────────────────────────────


@dataclass
class TEEPolicy:
    """Declarative attestation-quality gate.

    min_attestation_tier
        The lowest effective tier accepted. Default NONE
        (no gating). Set SOFTWARE to require at least a
        parseable software-fallback attestation;
        HARDWARE_UNVERIFIED to require a real hardware TEE
        (structural parse only — current production state);
        HARDWARE_VERIFIED to require full crypto chain.

    allowed_vendors
        Optional vendor allowlist. None → no allowlist
        check. Empty set → admit nothing (loud-failure
        protection against accidental empty-list misconfig).
        Non-empty set → vendor must be in the set.

    require_signature_chain
        Tighter form of HARDWARE_VERIFIED — also requires
        the backend's signature_chain_ok flag to be True.
        Layered on top of the tier check, not in lieu of.
    """

    min_attestation_tier: AttestationTier = (
        AttestationTier.NONE
    )
    allowed_vendors: Optional[Set[str]] = None
    require_signature_chain: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_attestation_tier": (
                self.min_attestation_tier.value
            ),
            "allowed_vendors": (
                sorted(self.allowed_vendors)
                if self.allowed_vendors is not None
                else None
            ),
            "require_signature_chain": (
                self.require_signature_chain
            ),
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "TEEPolicy":
        tier_raw = d.get("min_attestation_tier", "none")
        try:
            tier = AttestationTier(tier_raw)
        except ValueError:
            raise ValueError(
                f"unknown attestation tier {tier_raw!r}"
            )
        allowed = d.get("allowed_vendors")
        if allowed is not None:
            allowed = set(allowed)
        return cls(
            min_attestation_tier=tier,
            allowed_vendors=allowed,
            require_signature_chain=bool(
                d.get("require_signature_chain", False),
            ),
        )

    def stricter_of(self, other: "TEEPolicy") -> "TEEPolicy":
        """Sprint 1114 — combine two policies into the one that satisfies BOTH (used to
        merge an operator-mandated floor with a requester's policy). The result requires:
          - the HIGHER min_attestation_tier of the two,
          - the INTERSECTION of the vendor allowlists (None ∩ S = S; both set →
            intersection, which may be empty → admit nothing, the loud-misconfig guard),
          - signature-chain if EITHER requires it.
        Neither policy is weakened — the merge can only tighten."""
        higher = (
            self.min_attestation_tier
            if tier_rank(self.min_attestation_tier)
            >= tier_rank(other.min_attestation_tier)
            else other.min_attestation_tier
        )
        if self.allowed_vendors is None:
            vendors = other.allowed_vendors
        elif other.allowed_vendors is None:
            vendors = self.allowed_vendors
        else:
            vendors = self.allowed_vendors & other.allowed_vendors
        return TEEPolicy(
            min_attestation_tier=higher,
            allowed_vendors=vendors,
            require_signature_chain=(
                self.require_signature_chain or other.require_signature_chain
            ),
        )


def operator_floor_policy_from_env(env: Optional[Dict[str, str]] = None) -> Optional["TEEPolicy"]:
    """Sprint 1114 — the OPERATOR-mandated TEE floor, read from the environment:
      - PRSM_MIN_ATTESTATION_TIER (none|software|hardware_unverified|hardware_verified)
      - PRSM_TEE_ALLOWED_VENDORS (comma-separated allowlist; optional)
      - PRSM_TEE_REQUIRE_SIGNATURE_CHAIN (truthy → require the full crypto chain)
    Returns None when the operator set no floor (preserving the prior requester-opt-in
    behavior). A floor of tier 'none' with no other constraint also returns None (it
    gates nothing). An unparseable tier is treated as no floor + is the caller's job to
    surface; we fail SAFE toward not silently weakening (return None, the operator's
    intent was a floor — so we log loudly at the call site)."""
    import os

    e = env if env is not None else os.environ
    tier_raw = (e.get("PRSM_MIN_ATTESTATION_TIER") or "").strip().lower()
    vendors_raw = (e.get("PRSM_TEE_ALLOWED_VENDORS") or "").strip()
    require_chain = (e.get("PRSM_TEE_REQUIRE_SIGNATURE_CHAIN") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    if not tier_raw and not vendors_raw and not require_chain:
        return None
    tier = AttestationTier.NONE
    if tier_raw:
        try:
            tier = AttestationTier(tier_raw)
        except ValueError:
            raise ValueError(
                f"PRSM_MIN_ATTESTATION_TIER={tier_raw!r} is not a valid tier "
                f"(none|software|hardware_unverified|hardware_verified)"
            )
    vendors = (
        {v.strip() for v in vendors_raw.split(",") if v.strip()}
        if vendors_raw else None
    )
    # A pure tier=none floor with no vendor/chain constraint gates nothing → None.
    if tier == AttestationTier.NONE and vendors is None and not require_chain:
        return None
    return TEEPolicy(
        min_attestation_tier=tier,
        allowed_vendors=vendors,
        require_signature_chain=require_chain,
    )


# ── PolicyResult ─────────────────────────────────────


@dataclass
class PolicyResult:
    status: PolicyStatus
    effective_tier: AttestationTier
    min_required_tier: AttestationTier
    vendor: str = ""
    diagnostic: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "effective_tier": self.effective_tier.value,
            "min_required_tier": (
                self.min_required_tier.value
            ),
            "vendor": self.vendor,
            "diagnostic": self.diagnostic,
            "error": self.error,
        }


# ── Tier derivation ──────────────────────────────────


def effective_tier_from_result(
    result: AttestationVerificationResult,
) -> AttestationTier:
    """Map an AttestationVerificationResult to its
    effective tier."""
    if result.error:
        return AttestationTier.NONE
    if not result.vendor or result.vendor == "unknown":
        return AttestationTier.NONE
    if result.vendor == "software-fallback":
        return AttestationTier.SOFTWARE
    if result.vendor in _REAL_HARDWARE_VENDORS:
        if result.vendor_verified:
            return AttestationTier.HARDWARE_VERIFIED
        return AttestationTier.HARDWARE_UNVERIFIED
    return AttestationTier.NONE


# ── Evaluation ───────────────────────────────────────


def evaluate_attestation_result(
    result: AttestationVerificationResult,
    policy: TEEPolicy,
) -> PolicyResult:
    """Pure check: does this AttestationVerificationResult
    satisfy this policy?"""
    effective = effective_tier_from_result(result)
    required = policy.min_attestation_tier

    # Vendor allowlist — empty set fails loud; None allows
    # any.
    if policy.allowed_vendors is not None:
        if not policy.allowed_vendors:
            return PolicyResult(
                status=PolicyStatus.FAIL,
                effective_tier=effective,
                min_required_tier=required,
                vendor=result.vendor or "",
                diagnostic=(
                    "policy.allowed_vendors is an empty "
                    "set; admitting no vendor (loud-fail "
                    "on operator misconfig)"
                ),
            )
        if result.vendor not in policy.allowed_vendors:
            return PolicyResult(
                status=PolicyStatus.FAIL,
                effective_tier=effective,
                min_required_tier=required,
                vendor=result.vendor or "",
                diagnostic=(
                    f"vendor {result.vendor!r} not in "
                    f"allowlist "
                    f"{sorted(policy.allowed_vendors)}"
                ),
            )

    # Tier check
    if tier_rank(effective) < tier_rank(required):
        return PolicyResult(
            status=PolicyStatus.FAIL,
            effective_tier=effective,
            min_required_tier=required,
            vendor=result.vendor or "",
            diagnostic=(
                f"effective tier {effective.value!r} "
                f"below required {required.value!r}"
            ),
        )

    # Optional tighter signature-chain check (layered on
    # top of the tier gate).
    if (
        policy.require_signature_chain
        and not result.signature_chain_ok
    ):
        return PolicyResult(
            status=PolicyStatus.FAIL,
            effective_tier=effective,
            min_required_tier=required,
            vendor=result.vendor or "",
            diagnostic=(
                "require_signature_chain=True but "
                "result.signature_chain_ok is False"
            ),
        )

    return PolicyResult(
        status=PolicyStatus.PASS,
        effective_tier=effective,
        min_required_tier=required,
        vendor=result.vendor or "",
        diagnostic=(
            f"effective={effective.value} satisfies "
            f"required={required.value}"
        ),
    )


def evaluate_attestation_blob(
    blob: Optional[bytes],
    policy: TEEPolicy,
) -> PolicyResult:
    """Run the §7 registry on `blob` to derive an
    AttestationVerificationResult, then evaluate the
    policy on the result. Garbage / None / unknown blobs
    produce an unknown-vendor result → policy passes only
    when min_attestation_tier=NONE."""
    if blob is None:
        result = AttestationVerificationResult(
            vendor="unknown",
            error="attestation blob is None",
        )
    elif not isinstance(blob, (bytes, bytearray)):
        result = AttestationVerificationResult(
            vendor="unknown",
            error=(
                f"attestation blob must be bytes, got "
                f"{type(blob).__name__}"
            ),
        )
    else:
        try:
            result = verify_attestation(bytes(blob))
        except Exception as e:  # noqa: BLE001
            result = AttestationVerificationResult(
                vendor="unknown",
                error=f"verify_attestation raised: {e}",
            )
    return evaluate_attestation_result(result, policy)
