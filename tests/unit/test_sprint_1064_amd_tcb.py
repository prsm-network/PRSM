"""Sprint 1064 — AMD SEV-SNP TCB-recency enforcement (Tier 3, the AMD parallel to sp1061-1063).

Unlike Intel (a separately-signed TCB-Info JSON with levels), AMD encodes the TCB
the VCEK was issued for directly IN the VCEK certificate (AMD OID extensions under
1.3.6.1.4.1.3704.1.3), and the attestation report carries a REPORTED_TCB field
(offset 0x180, an 8-byte TCB_VERSION: byte0=bootloader, byte1=tee, byte6=snp,
byte7=microcode). So AMD TCB-recency = (a) BIND the report's REPORTED_TCB to the
chain-verified VCEK's issued TCB (anti-spoof/downgrade), and (b) enforce a
configurable minimum floor per component. No external signed collateral needed —
the VCEK cert is already chain-verified to the ARK.
"""
from __future__ import annotations

import datetime as _dt

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID, ObjectIdentifier
from pyasn1.type import univ
from pyasn1.codec.der.encoder import encode

from prsm.compute.inference.amd_tcb import (
    SnpTcb,
    SnpTcbError,
    SnpTcbPolicy,
    check_tcb_binding,
    default_snp_tcb_policy,
    parse_reported_tcb,
    parse_vcek_tcb,
    snp_tcb_policy_from_env,
)

_BL_OID = "1.3.6.1.4.1.3704.1.3.1"
_TEE_OID = "1.3.6.1.4.1.3704.1.3.2"
_SNP_OID = "1.3.6.1.4.1.3704.1.3.3"
_UCODE_OID = "1.3.6.1.4.1.3704.1.3.8"


def _report_with_tcb(bl, tee, snp, uc, length=1184):
    r = bytearray(length)
    r[0:4] = (2).to_bytes(4, "little")          # version 2
    tcb = bytearray(8)
    tcb[0], tcb[1], tcb[6], tcb[7] = bl, tee, snp, uc
    r[0x180:0x188] = tcb
    return bytes(r)


def _vcek_cert(bl, tee, snp, uc):
    priv = ec.generate_private_key(ec.SECP384R1())
    b = (x509.CertificateBuilder()
         .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "AMD VCEK")]))
         .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "AMD ASK")]))
         .public_key(priv.public_key()).serial_number(x509.random_serial_number())
         .not_valid_before(_dt.datetime(2025, 1, 1))
         .not_valid_after(_dt.datetime(2035, 1, 1)))
    for oid, val in ((_BL_OID, bl), (_TEE_OID, tee), (_SNP_OID, snp), (_UCODE_OID, uc)):
        b = b.add_extension(
            x509.UnrecognizedExtension(ObjectIdentifier(oid), encode(univ.Integer(val))),
            critical=False)
    return b.sign(priv, hashes.SHA384())


def test_parse_reported_tcb():
    tcb = parse_reported_tcb(_report_with_tcb(3, 0, 20, 209))
    assert tcb == SnpTcb(bootloader=3, tee=0, snp=20, microcode=209)


def test_parse_vcek_tcb():
    tcb = parse_vcek_tcb(_vcek_cert(3, 0, 20, 209))
    assert tcb == SnpTcb(bootloader=3, tee=0, snp=20, microcode=209)


def test_short_report_raises():
    with pytest.raises(SnpTcbError):
        parse_reported_tcb(b"\x00" * 0x100)


def test_vcek_missing_tcb_ext_raises():
    priv = ec.generate_private_key(ec.SECP384R1())
    cert = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "no ext")]))
            .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "no ext")]))
            .public_key(priv.public_key()).serial_number(1)
            .not_valid_before(_dt.datetime(2025, 1, 1))
            .not_valid_after(_dt.datetime(2035, 1, 1))
            .sign(priv, hashes.SHA384()))
    with pytest.raises(SnpTcbError):
        parse_vcek_tcb(cert)


def test_binding_match_ok():
    rep = parse_reported_tcb(_report_with_tcb(3, 0, 20, 209))
    vcek = parse_vcek_tcb(_vcek_cert(3, 0, 20, 209))
    check_tcb_binding(rep, vcek)  # no raise


def test_binding_mismatch_raises():
    """A report claiming a TCB different from its VCEK's issued TCB is a
    spoof/downgrade and must be rejected."""
    rep = parse_reported_tcb(_report_with_tcb(3, 0, 20, 209))
    vcek = parse_vcek_tcb(_vcek_cert(3, 0, 20, 100))  # microcode differs
    with pytest.raises(SnpTcbError):
        check_tcb_binding(rep, vcek)


def test_policy_floor_accept_and_reject():
    p = SnpTcbPolicy(min_bootloader=3, min_tee=0, min_snp=20, min_microcode=209)
    assert p.is_acceptable(SnpTcb(3, 0, 20, 209)) is True
    assert p.is_acceptable(SnpTcb(3, 0, 20, 210)) is True   # higher ok
    assert p.is_acceptable(SnpTcb(3, 0, 20, 208)) is False  # one below floor
    assert p.is_acceptable(SnpTcb(2, 0, 20, 209)) is False


def test_default_policy_is_binding_only_floor_zero():
    p = default_snp_tcb_policy()
    assert p.is_acceptable(SnpTcb(0, 0, 0, 0)) is True


def test_policy_from_env_off_when_unset():
    assert snp_tcb_policy_from_env({}) is None


def test_policy_from_env_parses_four_components():
    p = snp_tcb_policy_from_env({"PRSM_AMD_SEV_SNP_MIN_TCB": "3,0,20,209"})
    assert (p.min_bootloader, p.min_tee, p.min_snp, p.min_microcode) == (3, 0, 20, 209)


def test_policy_from_env_bad_format_raises():
    with pytest.raises(ValueError):
        snp_tcb_policy_from_env({"PRSM_AMD_SEV_SNP_MIN_TCB": "3,0,20"})


def test_policy_from_env_negative_floor_raises():
    """sp1065 review #5 — a negative floor (looks like enforcement, passes all) is rejected."""
    with pytest.raises(ValueError):
        snp_tcb_policy_from_env({"PRSM_AMD_SEV_SNP_MIN_TCB": "3,0,20,-1"})


def test_ext_spl_out_of_range_rejected():
    """sp1065 review #1 — a multi-byte/out-of-range SPL encoding is rejected (not
    silently truncated to a plausible value)."""
    cert = _vcek_cert(3, 0, 20, 256)  # 256 > 255
    with pytest.raises(SnpTcbError):
        parse_vcek_tcb(cert)


def test_ext_spl_multibyte_octet_string_rejected():
    priv = ec.generate_private_key(ec.SECP384R1())
    from pyasn1.type import univ as _u
    multi = encode(_u.OctetString(hexValue="00ff"))  # 2-byte octet string
    cert = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "AMD VCEK")]))
            .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "AMD ASK")]))
            .public_key(priv.public_key()).serial_number(1)
            .not_valid_before(_dt.datetime(2025, 1, 1)).not_valid_after(_dt.datetime(2035, 1, 1))
            .add_extension(x509.UnrecognizedExtension(ObjectIdentifier(_BL_OID), multi), critical=False)
            .add_extension(x509.UnrecognizedExtension(ObjectIdentifier(_TEE_OID), encode(univ.Integer(0))), critical=False)
            .add_extension(x509.UnrecognizedExtension(ObjectIdentifier(_SNP_OID), encode(univ.Integer(20))), critical=False)
            .add_extension(x509.UnrecognizedExtension(ObjectIdentifier(_UCODE_OID), encode(univ.Integer(209))), critical=False)
            .sign(priv, hashes.SHA384()))
    with pytest.raises(SnpTcbError):
        parse_vcek_tcb(cert)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
