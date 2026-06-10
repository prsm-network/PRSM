"""Sprint 1064 — AMD SEV-SNP TCB-recency enforcement (Tier 3, parallel to Intel sp1061-1063).

A validly-SIGNED SEV-SNP report can still come from a DOWNLEVEL (known-vulnerable)
or DOWNGRADED TCB. AMD's model differs from Intel's: the TCB the VCEK was issued for
is encoded directly in the VCEK certificate (AMD OID extensions under
1.3.6.1.4.1.3704.1.3), and the report carries a REPORTED_TCB field. So recency here
is two checks once the VCEK chain is verified to the ARK:

  (a) BINDING — the report's REPORTED_TCB must equal the VCEK's issued TCB. A report
      claiming a TCB its signing key wasn't issued for is a spoof/downgrade.
  (b) FLOOR — each of the 4 TCB components (bootloader, tee, snp, microcode) must
      meet a configurable minimum (the recency bar).

No external signed collateral is needed (unlike Intel's TCB-Info JSON) — the VCEK
cert is already chain-verified. Enforcement is OPT-IN via PRSM_AMD_SEV_SNP_MIN_TCB
(absent → no TCB check, unchanged).

TCB_VERSION (8 bytes, little-endian): byte0=BOOTLOADER, byte1=TEE, bytes2-5=reserved,
byte6=SNP, byte7=MICROCODE.

HONEST SCOPE: the SPL-extension encoding + REPORTED_TCB offset must be validated
against a REAL AMD VCEK cert + report before relying on the TCB dimension in prod
(the same engine-correct-vs-live-verified line as Intel sp1061-1063). The parser
accepts the SPL value as a DER INTEGER or OCTET STRING (last byte) to be tolerant of
the real encoding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cryptography import x509
from cryptography.x509.oid import ObjectIdentifier
from pyasn1.codec.der.decoder import decode as der_decode

# AMD SEV-SNP VCEK TCB component extension OIDs.
_BL_OID = "1.3.6.1.4.1.3704.1.3.1"      # blSPL  (bootloader)
_TEE_OID = "1.3.6.1.4.1.3704.1.3.2"     # teeSPL (tee)
_SNP_OID = "1.3.6.1.4.1.3704.1.3.3"     # snpSPL (snp)
_UCODE_OID = "1.3.6.1.4.1.3704.1.3.8"   # ucodeSPL (microcode)

_REPORTED_TCB_OFF = 0x180   # REPORTED_TCB (TCB_VERSION, 8 bytes) in the v2 report


class SnpTcbError(ValueError):
    """The TCB level is unreadable, the report/VCEK TCB don't bind, or the encoding
    is unexpected. Raised (not swallowed) so the verifier fails closed — an
    unverifiable TCB must NOT be treated as acceptable."""


@dataclass(frozen=True)
class SnpTcb:
    bootloader: int
    tee: int
    snp: int
    microcode: int


def parse_reported_tcb(report: bytes) -> SnpTcb:
    """Extract REPORTED_TCB (the TCB the report attests to) from a SEV-SNP report."""
    if len(report) < _REPORTED_TCB_OFF + 8:
        raise SnpTcbError(
            f"report too short ({len(report)} bytes) for REPORTED_TCB @0x180")
    tcb = report[_REPORTED_TCB_OFF:_REPORTED_TCB_OFF + 8]
    return SnpTcb(bootloader=tcb[0], tee=tcb[1], snp=tcb[6], microcode=tcb[7])


def _ext_spl(cert: x509.Certificate, oid: str) -> int:
    try:
        ext = cert.extensions.get_extension_for_oid(ObjectIdentifier(oid))
    except x509.ExtensionNotFound:
        raise SnpTcbError(f"VCEK missing TCB extension {oid}")
    raw = ext.value.value if isinstance(ext.value, x509.UnrecognizedExtension) else ext.value
    try:
        decoded, _ = der_decode(bytes(raw))
    except Exception as exc:  # noqa: BLE001
        raise SnpTcbError(f"VCEK TCB extension {oid} is not valid DER: {exc}")
    # Real AMD VCEKs encode the SPL as a single octet 0-255; accept a DER INTEGER or
    # a single-byte OCTET STRING. Reject anything outside that range / shape (sp1065
    # review #1) so a maliciously multi-byte/negative/out-of-range encoding can't slip
    # an odd value into the binding/floor logic.
    try:
        value = int(decoded)
    except Exception:  # noqa: BLE001
        try:
            b = bytes(decoded)
        except Exception:  # noqa: BLE001
            raise SnpTcbError(f"VCEK TCB extension {oid} has an unexpected type")
        if len(b) != 1:
            raise SnpTcbError(
                f"VCEK TCB extension {oid} octet string must be 1 byte, got {len(b)}")
        value = b[0]
    if not (0 <= value <= 255):
        raise SnpTcbError(f"VCEK TCB extension {oid} SPL out of range 0-255: {value}")
    return value


def parse_vcek_tcb(vcek_cert: x509.Certificate) -> SnpTcb:
    """Read the 4 TCB component SVNs the VCEK was issued for, from its AMD extensions."""
    return SnpTcb(
        bootloader=_ext_spl(vcek_cert, _BL_OID),
        tee=_ext_spl(vcek_cert, _TEE_OID),
        snp=_ext_spl(vcek_cert, _SNP_OID),
        microcode=_ext_spl(vcek_cert, _UCODE_OID),
    )


def check_tcb_binding(report_tcb: SnpTcb, vcek_tcb: SnpTcb) -> None:
    """Raise unless the report's REPORTED_TCB equals the VCEK's issued TCB. A
    mismatch means the report claims a TCB its (chain-verified) signing key was not
    issued for — a spoof/downgrade."""
    if report_tcb != vcek_tcb:
        raise SnpTcbError(
            f"report REPORTED_TCB {report_tcb} != VCEK issued TCB {vcek_tcb} "
            f"(spoof/downgrade)")


@dataclass(frozen=True)
class SnpTcbPolicy:
    """Per-component minimum TCB floor. A TCB is acceptable iff every component meets
    or exceeds its minimum (fails closed below any)."""
    min_bootloader: int
    min_tee: int
    min_snp: int
    min_microcode: int

    def is_acceptable(self, tcb: SnpTcb) -> bool:
        return (tcb.bootloader >= self.min_bootloader
                and tcb.tee >= self.min_tee
                and tcb.snp >= self.min_snp
                and tcb.microcode >= self.min_microcode)


def default_snp_tcb_policy() -> SnpTcbPolicy:
    """A zero floor — binding is still enforced, but no recency minimum. Operators
    set PRSM_AMD_SEV_SNP_MIN_TCB to AMD's published current TCB to enforce recency."""
    return SnpTcbPolicy(0, 0, 0, 0)


def snp_tcb_policy_from_env(environ) -> Optional[SnpTcbPolicy]:
    """Build a policy from ``PRSM_AMD_SEV_SNP_MIN_TCB`` ("bl,tee,snp,microcode").
    Unset → None (TCB enforcement OFF, unchanged behavior). Setting it (even
    "0,0,0,0") turns enforcement ON (binding + the given floor)."""
    raw = (environ.get("PRSM_AMD_SEV_SNP_MIN_TCB", "") or "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("PRSM_AMD_SEV_SNP_MIN_TCB must be 4 comma-separated ints "
                         "(bootloader,tee,snp,microcode)")
    try:
        bl, tee, snp, uc = (int(p) for p in parts)
    except ValueError:
        raise ValueError("PRSM_AMD_SEV_SNP_MIN_TCB components must be integers")
    if any(x < 0 for x in (bl, tee, snp, uc)):
        # a negative floor is a foot-gun (looks like enforcement, actually passes
        # everything for that component) — reject so it can't silently weaken (sp1065 #5)
        raise ValueError("PRSM_AMD_SEV_SNP_MIN_TCB components must be >= 0")
    return SnpTcbPolicy(min_bootloader=bl, min_tee=tee, min_snp=snp, min_microcode=uc)
