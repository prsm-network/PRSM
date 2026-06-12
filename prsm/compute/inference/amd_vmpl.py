"""Sprint 1080 — AMD SEV-SNP VMPL / guest-policy enforcement (Tier 3 hardening).

A validly-SIGNED SEV-SNP report can still come from a less-privileged VMPL or a
DEBUG-enabled guest (which is inspectable, not confidential). The report's VMPL
(0x30), GUEST_POLICY (0x08) and PLATFORM_INFO (0x40) are inside the signed region
(report[:0x2A0]), so they're authenticated by the report signature — this enforces a
configurable policy over them, the AMD parallel to Intel QE-identity / TCB-recency.

GUEST_POLICY bits (AMD SEV-SNP ABI): SMT=16, RESERVED=17 (must-be-1), MIGRATE_MA=18,
DEBUG=19. Default policy: VMPL must be <= 0 (the most-privileged guest measurement)
and DEBUG must be disabled.

HONEST SCOPE (same as the AMD TCB parsers): the offsets + bit positions follow the
AMD spec but are self-fixture-validated — confirm against a REAL SEV-SNP report before
relying on this dimension in prod (a layout error fails closed — false rejects, never
a silent accept).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

_POLICY_OFF = 0x08      # GUEST_POLICY (u64 LE)
_VMPL_OFF = 0x30        # VMPL (u32 LE)
_PLATFORM_INFO_OFF = 0x40  # PLATFORM_INFO (u64 LE)

_BIT_SMT = 1 << 16
_BIT_MIGRATE_MA = 1 << 18
_BIT_DEBUG = 1 << 19


class SnpPolicyError(ValueError):
    """The SEV-SNP report's VMPL/guest-policy is unreadable or violates the configured
    policy. Raised (not swallowed) so the verifier fails closed."""


@dataclass(frozen=True)
class SnpReportFlags:
    vmpl: int
    policy: int            # raw GUEST_POLICY u64
    platform_info: int     # raw PLATFORM_INFO u64
    debug: bool
    migrate_ma: bool
    smt: bool


def parse_snp_report_flags(report: bytes) -> SnpReportFlags:
    """Extract VMPL + the GUEST_POLICY flags from a SEV-SNP report."""
    if len(report) < _PLATFORM_INFO_OFF + 8:
        raise SnpPolicyError(
            f"report too short ({len(report)} bytes) for VMPL/policy fields")
    (policy,) = struct.unpack_from("<Q", report, _POLICY_OFF)
    (vmpl,) = struct.unpack_from("<I", report, _VMPL_OFF)
    (platform_info,) = struct.unpack_from("<Q", report, _PLATFORM_INFO_OFF)
    return SnpReportFlags(
        vmpl=vmpl, policy=policy, platform_info=platform_info,
        debug=bool(policy & _BIT_DEBUG),
        migrate_ma=bool(policy & _BIT_MIGRATE_MA),
        smt=bool(policy & _BIT_SMT),
    )


@dataclass(frozen=True)
class SnpVmplPolicy:
    """Configurable SEV-SNP VMPL / guest-policy floor.

    ``max_vmpl`` — the report's VMPL must be <= this (default 0: the most-privileged
    guest context; a report from VMPL>0 is a less-privileged measurement). The two
    confidentiality-critical guest-policy checks: ``require_debug_disabled`` (a
    debug-enabled guest is inspectable, not confidential) and the optional
    ``forbid_migrate_ma`` / ``forbid_smt`` for deployments that disallow them."""
    max_vmpl: int = 0
    require_debug_disabled: bool = True
    forbid_migrate_ma: bool = False
    forbid_smt: bool = False


def default_snp_vmpl_policy() -> SnpVmplPolicy:
    return SnpVmplPolicy()


def verify_snp_vmpl_policy(flags: SnpReportFlags, policy: SnpVmplPolicy) -> None:
    """Raise ``SnpPolicyError`` unless ``flags`` satisfy ``policy``. Returns None on OK."""
    if flags.vmpl > policy.max_vmpl:
        raise SnpPolicyError(
            f"report VMPL {flags.vmpl} > max allowed {policy.max_vmpl} "
            f"(less-privileged guest context)")
    if policy.require_debug_disabled and flags.debug:
        raise SnpPolicyError(
            "guest policy has DEBUG enabled — the guest is inspectable, not confidential")
    if policy.forbid_migrate_ma and flags.migrate_ma:
        raise SnpPolicyError("guest policy allows a migration agent (MIGRATE_MA) — forbidden")
    if policy.forbid_smt and flags.smt:
        raise SnpPolicyError("guest policy allows SMT — forbidden")


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def snp_vmpl_policy_from_env(environ):
    """Build a policy from env, or None when not enabled. Enabled by
    ``PRSM_AMD_SEV_SNP_VMPL_POLICY`` truthy; overrides: ``PRSM_AMD_SEV_SNP_MAX_VMPL``
    (int, default 0), ``PRSM_AMD_SEV_SNP_ALLOW_DEBUG`` (truthy → don't require
    debug-disabled), ``PRSM_AMD_SEV_SNP_FORBID_MIGRATE_MA`` / ``_FORBID_SMT`` (truthy)."""
    if not _truthy(environ.get("PRSM_AMD_SEV_SNP_VMPL_POLICY", "")):
        return None
    raw_max = (environ.get("PRSM_AMD_SEV_SNP_MAX_VMPL", "") or "").strip()
    if raw_max:
        try:
            max_vmpl = int(raw_max)
        except ValueError:
            raise ValueError(
                f"PRSM_AMD_SEV_SNP_MAX_VMPL must be an int (got {raw_max!r})")
        if max_vmpl < 0 or max_vmpl > 3:
            raise ValueError("PRSM_AMD_SEV_SNP_MAX_VMPL must be in [0, 3]")
    else:
        max_vmpl = 0
    return SnpVmplPolicy(
        max_vmpl=max_vmpl,
        require_debug_disabled=not _truthy(environ.get("PRSM_AMD_SEV_SNP_ALLOW_DEBUG", "")),
        forbid_migrate_ma=_truthy(environ.get("PRSM_AMD_SEV_SNP_FORBID_MIGRATE_MA", "")),
        forbid_smt=_truthy(environ.get("PRSM_AMD_SEV_SNP_FORBID_SMT", "")),
    )
