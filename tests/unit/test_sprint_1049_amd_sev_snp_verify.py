"""Sprint 1049 — real AMD SEV-SNP attestation verification (Tier 3, parallel to Intel DCAP).

Closes the §7 attestation_vendor_verified=false gap for AMD SEV-SNP. The verifier:
  1. parses the SEV-SNP ATTESTATION_REPORT (v2, 1184 bytes) — measurement (0x90,48),
     report_data (0x50,64), chip_id (0x1A0,64), signature (0x2A0,512);
  2. verifies the report's ECDSA-P384 / SHA-384 signature over report[:0x2A0] using
     the VCEK public key (AMD stores r,s LITTLE-ENDIAN in 72-byte slots);
  3. verifies the VCEK → ASK → ARK X.509 chain (P-384) up to a CONFIGURED ARK root
     (operator pins AMD's real ARK; vendor_verified is only as strong as that anchor),
     reusing the same hardened path-validation as Intel (CA-flag, validity,
     name-chaining, keyCertSign, pathLen).

The SEV-SNP report carries no embedded certs (unlike Intel's quote), so PRSM wraps
report + VCEK/ASK chain in an envelope: b"PRSMSNP1" | report_len(u32 LE) | report |
chain_pem. Hermetic: the test builds a self-consistent ARK→ASK→VCEK chain + a
VCEK-signed report, fully offline, no SEV hardware, no AMD KDS.
"""
from __future__ import annotations

import datetime as _dt
import struct

import pytest
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

_REPORT_LEN = 1184
_SIG_OFF = 0x2A0          # 672
_MEAS_OFF = 0x90          # 144
_RDATA_OFF = 0x50         # 80
_CHIPID_OFF = 0x1A0       # 416


def _mk_cert(subj, issuer, subj_pub, issuer_priv, *, ca, not_before=None, not_after=None):
    nvb = not_before or _dt.datetime(2025, 1, 1)
    nva = not_after or (nvb + _dt.timedelta(days=3650))
    return (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subj)]))
            .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer)]))
            .public_key(subj_pub).serial_number(x509.random_serial_number())
            .not_valid_before(nvb).not_valid_after(nva)
            .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
            .sign(issuer_priv, hashes.SHA384()))


def build_sev_snp_envelope(*, ark_priv=None, measurement=b"\x5a" * 48,
                           report_data=b"\x11" * 64, chip_id=b"\x3c" * 64):
    if ark_priv is None:
        ark_priv = ec.generate_private_key(ec.SECP384R1())
    ask_priv = ec.generate_private_key(ec.SECP384R1())
    vcek_priv = ec.generate_private_key(ec.SECP384R1())
    ark = _mk_cert("AMD ARK", "AMD ARK", ark_priv.public_key(), ark_priv, ca=True)
    ask = _mk_cert("AMD ASK", "AMD ARK", ask_priv.public_key(), ark_priv, ca=True)
    vcek = _mk_cert("AMD VCEK", "AMD ASK", vcek_priv.public_key(), ask_priv, ca=False)

    report = bytearray(_REPORT_LEN)
    struct.pack_into("<I", report, 0, 2)             # version 2
    report[_MEAS_OFF:_MEAS_OFF + 48] = measurement
    report[_RDATA_OFF:_RDATA_OFF + 64] = report_data
    report[_CHIPID_OFF:_CHIPID_OFF + 64] = chip_id
    # VCEK signs report[:0x2A0] (P-384 / SHA-384); store r,s LITTLE-ENDIAN in 72B slots
    der = vcek_priv.sign(bytes(report[:_SIG_OFF]), ec.ECDSA(hashes.SHA384()))
    r, s = decode_dss_signature(der)
    report[_SIG_OFF:_SIG_OFF + 72] = r.to_bytes(48, "little") + b"\x00" * 24
    report[_SIG_OFF + 72:_SIG_OFF + 144] = s.to_bytes(48, "little") + b"\x00" * 24

    chain_pem = (vcek.public_bytes(serialization.Encoding.PEM)
                 + ask.public_bytes(serialization.Encoding.PEM))
    envelope = b"PRSMSNP1" + struct.pack("<I", _REPORT_LEN) + bytes(report) + chain_pem
    ark_pem = ark.public_bytes(serialization.Encoding.PEM)
    return envelope, ark_pem, {"measurement": measurement.hex(), "chip_id": chip_id.hex()}


# ── tests ─────────────────────────────────────────────────────────────────────

def test_sev_snp_verifies_self_consistent_envelope():
    from prsm.compute.inference.amd_sev_snp import AMDSEVSNPBackend
    env, ark_pem, expected = build_sev_snp_envelope()
    res = AMDSEVSNPBackend(trusted_root_pem=ark_pem).verify(env)
    assert res.vendor == "amd-sev-snp"
    assert res.structural_parse_ok is True
    assert res.vendor_verified is True
    assert res.signature_chain_ok is True
    assert res.error is None
    assert res.vendor_data["measurement_hex"] == expected["measurement"]
    assert res.vendor_data["chip_id_hex"] == expected["chip_id"]


def test_sev_snp_rejects_tampered_report():
    from prsm.compute.inference.amd_sev_snp import AMDSEVSNPBackend
    env, ark_pem, _ = build_sev_snp_envelope()
    b = bytearray(env)
    # flip a byte inside the measurement region (8 magic + 4 len + 0x90 offset)
    b[8 + 4 + _MEAS_OFF] ^= 0xFF
    res = AMDSEVSNPBackend(trusted_root_pem=ark_pem).verify(bytes(b))
    assert res.vendor_verified is False
    assert res.error and "signature" in res.error.lower()


def test_sev_snp_rejects_wrong_ark_root():
    from prsm.compute.inference.amd_sev_snp import AMDSEVSNPBackend
    env, _good, _ = build_sev_snp_envelope()
    _e2, other_ark, _ = build_sev_snp_envelope()
    res = AMDSEVSNPBackend(trusted_root_pem=other_ark).verify(env)
    assert res.vendor_verified is False
    assert res.error and ("chain" in res.error.lower() or "root" in res.error.lower()
                          or "issuer" in res.error.lower())


def test_sev_snp_rejects_non_ca_issuer_in_chain():
    """Forgery guard (the Intel CRITICAL, shared path-validator): an attacker with
    a genuine non-CA VCEK can't use it to 'issue' a forged signer cert."""
    from prsm.compute.inference.amd_sev_snp import AMDSEVSNPBackend
    ark_priv = ec.generate_private_key(ec.SECP384R1())
    ask_priv = ec.generate_private_key(ec.SECP384R1())
    vcek_priv = ec.generate_private_key(ec.SECP384R1())     # genuine VCEK (ca=False)
    forged_priv = ec.generate_private_key(ec.SECP384R1())   # attacker
    ark = _mk_cert("ARK", "ARK", ark_priv.public_key(), ark_priv, ca=True)
    ask = _mk_cert("ASK", "ARK", ask_priv.public_key(), ark_priv, ca=True)
    vcek = _mk_cert("VCEK", "ASK", vcek_priv.public_key(), ask_priv, ca=False)
    forged = _mk_cert("FORGED", "VCEK", forged_priv.public_key(), vcek_priv, ca=False)

    report = bytearray(_REPORT_LEN)
    struct.pack_into("<I", report, 0, 2)
    report[_MEAS_OFF:_MEAS_OFF + 48] = b"\x99" * 48   # attacker-chosen measurement
    der = forged_priv.sign(bytes(report[:_SIG_OFF]), ec.ECDSA(hashes.SHA384()))
    r, s = decode_dss_signature(der)
    report[_SIG_OFF:_SIG_OFF + 72] = r.to_bytes(48, "little") + b"\x00" * 24
    report[_SIG_OFF + 72:_SIG_OFF + 144] = s.to_bytes(48, "little") + b"\x00" * 24
    chain = (forged.public_bytes(serialization.Encoding.PEM)
             + vcek.public_bytes(serialization.Encoding.PEM)
             + ask.public_bytes(serialization.Encoding.PEM))
    env = b"PRSMSNP1" + struct.pack("<I", _REPORT_LEN) + bytes(report) + chain
    res = AMDSEVSNPBackend(
        trusted_root_pem=ark.public_bytes(serialization.Encoding.PEM)).verify(env)
    assert res.vendor_verified is False           # non-CA VCEK can't be an issuer


def test_sev_snp_requires_trusted_root():
    from prsm.compute.inference.amd_sev_snp import AMDSEVSNPBackend
    with pytest.raises(ValueError):
        AMDSEVSNPBackend(trusted_root_pem=b"")


def test_sev_snp_rejects_non_envelope_blob():
    from prsm.compute.inference.amd_sev_snp import AMDSEVSNPBackend
    env, ark_pem, _ = build_sev_snp_envelope()
    res = AMDSEVSNPBackend(trusted_root_pem=ark_pem).verify(b"\x02\x00\x00\x00rawreport")
    assert res.vendor_verified is False
    assert res.error is not None


def test_sev_snp_handles_truncated_envelope():
    from prsm.compute.inference.amd_sev_snp import AMDSEVSNPBackend
    env, ark_pem, _ = build_sev_snp_envelope()
    res = AMDSEVSNPBackend(trusted_root_pem=ark_pem).verify(env[:50])
    assert res.vendor_verified is False
    assert res.error is not None


def test_sev_snp_rejects_unsupported_version():
    """sp1049 review — a non-v2 report (different field layout) must be rejected,
    not silently mis-parsed (parity with Intel's version gate)."""
    from prsm.compute.inference.amd_sev_snp import AMDSEVSNPBackend
    env, ark_pem, _ = build_sev_snp_envelope()
    b = bytearray(env)
    # report version field is at: 8 (magic) + 4 (len) + 0
    struct.pack_into("<I", b, 8 + 4 + 0, 3)   # claim v3
    res = AMDSEVSNPBackend(trusted_root_pem=ark_pem).verify(bytes(b))
    assert res.vendor_verified is False
    assert res.error and "version" in res.error.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
