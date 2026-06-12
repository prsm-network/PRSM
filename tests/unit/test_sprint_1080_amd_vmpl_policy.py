"""Sprint 1080 — AMD SEV-SNP VMPL / guest-policy enforcement (Tier 3, AMD parallel to QE-identity).

A validly-signed SEV-SNP report can still come from a less-privileged VMPL or a
DEBUG-enabled guest (not confidential). The VMPL (0x30), GUEST_POLICY (0x08), and
PLATFORM_INFO (0x40) are inside the signed region, so they're authenticated; this
enforces a configurable policy over them: the report must be from an acceptable VMPL
(default 0, the most-privileged guest context) with debugging disabled.
"""
from __future__ import annotations

import struct

import pytest

from prsm.compute.inference.amd_vmpl import (
    SnpPolicyError,
    SnpVmplPolicy,
    default_snp_vmpl_policy,
    parse_snp_report_flags,
    snp_vmpl_policy_from_env,
    verify_snp_vmpl_policy,
)

_POLICY_OFF = 0x08
_VMPL_OFF = 0x30
_PLATFORM_INFO_OFF = 0x40
# GUEST_POLICY bits
_BIT_SMT = 1 << 16
_BIT_MIGRATE_MA = 1 << 18
_BIT_DEBUG = 1 << 19
_RESERVED_BIT17 = 1 << 17   # must-be-1 per AMD spec


def _report(vmpl=0, debug=False, migrate_ma=False, smt=False, platform_info=0):
    r = bytearray(1184)
    struct.pack_into("<I", r, 0, 2)               # version 2
    policy = _RESERVED_BIT17                       # bit 17 reserved must-be-1
    if smt:
        policy |= _BIT_SMT
    if migrate_ma:
        policy |= _BIT_MIGRATE_MA
    if debug:
        policy |= _BIT_DEBUG
    struct.pack_into("<Q", r, _POLICY_OFF, policy)
    struct.pack_into("<I", r, _VMPL_OFF, vmpl)
    struct.pack_into("<Q", r, _PLATFORM_INFO_OFF, platform_info)
    return bytes(r)


# ── parsing ─────────────────────────────────────────────────────────────────────

def test_parse_flags():
    f = parse_snp_report_flags(_report(vmpl=0, debug=True, smt=True))
    assert f.vmpl == 0 and f.debug is True and f.smt is True and f.migrate_ma is False


def test_parse_vmpl_and_migrate():
    f = parse_snp_report_flags(_report(vmpl=2, migrate_ma=True))
    assert f.vmpl == 2 and f.migrate_ma is True and f.debug is False


def test_short_report_raises():
    with pytest.raises(SnpPolicyError):
        parse_snp_report_flags(b"\x00" * 0x20)


# ── policy ──────────────────────────────────────────────────────────────────────

def test_default_policy_accepts_vmpl0_no_debug():
    verify_snp_vmpl_policy(parse_snp_report_flags(_report(vmpl=0, debug=False)),
                           default_snp_vmpl_policy())   # no raise


def test_debug_enabled_rejected_by_default():
    with pytest.raises(SnpPolicyError):
        verify_snp_vmpl_policy(parse_snp_report_flags(_report(debug=True)),
                               default_snp_vmpl_policy())


def test_higher_vmpl_rejected_by_default():
    with pytest.raises(SnpPolicyError):
        verify_snp_vmpl_policy(parse_snp_report_flags(_report(vmpl=1)),
                               default_snp_vmpl_policy())   # max_vmpl default 0


def test_higher_vmpl_allowed_when_configured():
    p = SnpVmplPolicy(max_vmpl=2, require_debug_disabled=True)
    verify_snp_vmpl_policy(parse_snp_report_flags(_report(vmpl=2)), p)   # no raise


def test_forbid_migrate_ma():
    p = SnpVmplPolicy(max_vmpl=0, require_debug_disabled=True, forbid_migrate_ma=True)
    with pytest.raises(SnpPolicyError):
        verify_snp_vmpl_policy(parse_snp_report_flags(_report(migrate_ma=True)), p)


def test_forbid_smt():
    p = SnpVmplPolicy(max_vmpl=0, require_debug_disabled=True, forbid_smt=True)
    with pytest.raises(SnpPolicyError):
        verify_snp_vmpl_policy(parse_snp_report_flags(_report(smt=True)), p)


# ── env ─────────────────────────────────────────────────────────────────────────

def test_policy_from_env_off_when_unset():
    assert snp_vmpl_policy_from_env({}) is None


def test_policy_from_env_defaults_on_flag():
    p = snp_vmpl_policy_from_env({"PRSM_AMD_SEV_SNP_VMPL_POLICY": "1"})
    assert p is not None and p.max_vmpl == 0 and p.require_debug_disabled is True


def test_policy_from_env_overrides():
    p = snp_vmpl_policy_from_env({
        "PRSM_AMD_SEV_SNP_VMPL_POLICY": "1",
        "PRSM_AMD_SEV_SNP_MAX_VMPL": "3",
        "PRSM_AMD_SEV_SNP_FORBID_MIGRATE_MA": "1"})
    assert p.max_vmpl == 3 and p.forbid_migrate_ma is True


def test_policy_from_env_bad_max_vmpl_raises():
    with pytest.raises(ValueError):
        snp_vmpl_policy_from_env({"PRSM_AMD_SEV_SNP_VMPL_POLICY": "1",
                                  "PRSM_AMD_SEV_SNP_MAX_VMPL": "notint"})


# ── end-to-end through AMDSEVSNPBackend ─────────────────────────────────────────

def test_backend_rejects_debug_guest():
    """A signed report with DEBUG set is rejected when the VMPL policy is configured."""
    import datetime as _dt
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
    from prsm.compute.inference.amd_sev_snp import AMDSEVSNPBackend

    def _mk(subj, issuer, pub, priv, ca):
        return (x509.CertificateBuilder()
                .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subj)]))
                .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer)]))
                .public_key(pub).serial_number(x509.random_serial_number())
                .not_valid_before(_dt.datetime(2025, 1, 1)).not_valid_after(_dt.datetime(2035, 1, 1))
                .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
                .sign(priv, hashes.SHA384()))

    ark_priv = ec.generate_private_key(ec.SECP384R1())
    ask_priv = ec.generate_private_key(ec.SECP384R1())
    vcek_priv = ec.generate_private_key(ec.SECP384R1())
    ark = _mk("ARK", "ARK", ark_priv.public_key(), ark_priv, True)
    ask = _mk("ASK", "ARK", ask_priv.public_key(), ark_priv, True)
    vcek = _mk("VCEK", "ASK", vcek_priv.public_key(), ask_priv, False)

    report = bytearray(_report(vmpl=0, debug=True))   # DEBUG enabled
    der = vcek_priv.sign(bytes(report[:0x2A0]), ec.ECDSA(hashes.SHA384()))
    r, s = decode_dss_signature(der)
    report[0x2A0:0x2A0 + 72] = r.to_bytes(48, "little") + b"\x00" * 24
    report[0x2A0 + 72:0x2A0 + 144] = s.to_bytes(48, "little") + b"\x00" * 24
    chain = vcek.public_bytes(serialization.Encoding.PEM) + ask.public_bytes(serialization.Encoding.PEM)
    env = b"PRSMSNP1" + struct.pack("<I", 1184) + bytes(report) + chain
    ark_pem = ark.public_bytes(serialization.Encoding.PEM)

    res = AMDSEVSNPBackend(trusted_root_pem=ark_pem,
                           vmpl_policy=default_snp_vmpl_policy()).verify(env)
    assert res.vendor_verified is False
    assert "vmpl" in (res.error or "").lower() or "debug" in (res.error or "").lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
