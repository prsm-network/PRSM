"""Sprint 1079 — Intel SGX QE-Identity pinning (Tier 3 hardening).

Closes the "is the QE a genuine Intel Quoting Enclave" gap in the DCAP chain: verify
the QE report's MRSIGNER/ISVPRODID/ISVSVN against Intel's signed QE Identity (same
shape + signer as the TCB-Info). Hermetic tests (built QE report + signed QE Identity)
plus a GOLDEN test against the REAL Intel QE Identity + the real TCB Signing cert.
"""
from __future__ import annotations

import json
import os
import struct

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

from prsm.compute.inference.qe_identity import (
    QeIdentityError,
    parse_and_verify_qe_identity,
    parse_qe_report_identity,
    qe_tcb_status,
    verify_qe_identity,
)

_MRSIGNER_OFF = 0x80
_ISVPRODID_OFF = 0x100
_ISVSVN_OFF = 0x102
_FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures", "attestation")

MRSIGNER = bytes.fromhex("8c" * 32)


def _qe_report(mrsigner=MRSIGNER, isvprodid=1, isvsvn=8):
    r = bytearray(384)
    r[_MRSIGNER_OFF:_MRSIGNER_OFF + 32] = mrsigner
    struct.pack_into("<H", r, _ISVPRODID_OFF, isvprodid)
    struct.pack_into("<H", r, _ISVSVN_OFF, isvsvn)
    return bytes(r)


def _raw_sig(priv, msg):
    r, s = decode_dss_signature(priv.sign(msg, ec.ECDSA(hashes.SHA256())))
    return (r.to_bytes(32, "big") + s.to_bytes(32, "big")).hex()


def _signer_cert(priv):
    import datetime as _dt
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    return (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Intel SGX TCB Signing")]))
            .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Intel SGX TCB Signing")]))
            .public_key(priv.public_key()).serial_number(1)
            .not_valid_before(_dt.datetime(2025, 1, 1)).not_valid_after(_dt.datetime(2035, 1, 1))
            .sign(priv, hashes.SHA256()))


def _signed_identity(priv, *, mrsigner=MRSIGNER, isvprodid=1, levels=None):
    if levels is None:
        levels = [{"tcb": {"isvsvn": 8}, "tcbStatus": "UpToDate"},
                  {"tcb": {"isvsvn": 4}, "tcbStatus": "OutOfDate"}]
    ei = {"id": "QE", "version": 2, "mrsigner": mrsigner.hex(),
          "isvprodid": isvprodid, "tcbLevels": levels}
    b = json.dumps(ei, separators=(",", ":")).encode()
    return b'{"enclaveIdentity":' + b + b',"signature":"' + _raw_sig(priv, b).encode() + b'"}'


# ── QE report parsing ───────────────────────────────────────────────────────────

def test_parse_qe_report_identity():
    qe = parse_qe_report_identity(_qe_report(MRSIGNER, 1, 9))
    assert qe.mrsigner == MRSIGNER and qe.isvprodid == 1 and qe.isvsvn == 9


def test_short_qe_report_raises():
    with pytest.raises(QeIdentityError):
        parse_qe_report_identity(b"\x00" * 0x100)


# ── QE Identity parse + verify ──────────────────────────────────────────────────

def test_parse_and_verify_good_signature():
    priv = ec.generate_private_key(ec.SECP256R1())
    ident = parse_and_verify_qe_identity(_signed_identity(priv), _signer_cert(priv))
    assert ident.mrsigner == MRSIGNER and ident.isvprodid == 1
    assert len(ident.tcb_levels) == 2


def test_tampered_identity_fails_signature():
    priv = ec.generate_private_key(ec.SECP256R1())
    blob = _signed_identity(priv).replace(b'"UpToDate"', b'"Revoked!!"')
    with pytest.raises(QeIdentityError):
        parse_and_verify_qe_identity(blob, _signer_cert(priv))


def test_wrong_signer_fails():
    priv = ec.generate_private_key(ec.SECP256R1())
    other = ec.generate_private_key(ec.SECP256R1())
    with pytest.raises(QeIdentityError):
        parse_and_verify_qe_identity(_signed_identity(priv), _signer_cert(other))


# ── verification: MRSIGNER / ISVPRODID / status ─────────────────────────────────

def test_genuine_qe_passes():
    priv = ec.generate_private_key(ec.SECP256R1())
    ident = parse_and_verify_qe_identity(_signed_identity(priv), _signer_cert(priv))
    verify_qe_identity(parse_qe_report_identity(_qe_report(isvsvn=8)), ident)  # no raise


def test_wrong_mrsigner_rejected():
    priv = ec.generate_private_key(ec.SECP256R1())
    ident = parse_and_verify_qe_identity(_signed_identity(priv), _signer_cert(priv))
    rogue = parse_qe_report_identity(_qe_report(mrsigner=bytes.fromhex("ff" * 32)))
    with pytest.raises(QeIdentityError):
        verify_qe_identity(rogue, ident)


def test_wrong_isvprodid_rejected():
    priv = ec.generate_private_key(ec.SECP256R1())
    ident = parse_and_verify_qe_identity(_signed_identity(priv), _signer_cert(priv))
    with pytest.raises(QeIdentityError):
        verify_qe_identity(parse_qe_report_identity(_qe_report(isvprodid=99)), ident)


def test_downlevel_isvsvn_rejected():
    priv = ec.generate_private_key(ec.SECP256R1())
    ident = parse_and_verify_qe_identity(_signed_identity(priv), _signer_cert(priv))
    # isvsvn=4 → OutOfDate level → not acceptable by default
    with pytest.raises(QeIdentityError):
        verify_qe_identity(parse_qe_report_identity(_qe_report(isvsvn=4)), ident)


def test_status_derivation():
    priv = ec.generate_private_key(ec.SECP256R1())
    ident = parse_and_verify_qe_identity(_signed_identity(priv), _signer_cert(priv))
    assert qe_tcb_status(parse_qe_report_identity(_qe_report(isvsvn=8)), ident) == "UpToDate"
    assert qe_tcb_status(parse_qe_report_identity(_qe_report(isvsvn=5)), ident) == "OutOfDate"
    assert qe_tcb_status(parse_qe_report_identity(_qe_report(isvsvn=0)), ident) == "OutOfDate"


# ── GOLDEN: real Intel QE Identity + real TCB Signing cert ──────────────────────

def test_real_intel_qe_identity_parses_and_verifies():
    from cryptography import x509
    blob = open(os.path.join(_FIX, "intel_qe_identity.json"), "rb").read()
    signer = x509.load_pem_x509_certificate(
        open(os.path.join(_FIX, "intel_tcb_signing.pem"), "rb").read())
    ident = parse_and_verify_qe_identity(blob, signer)   # real ECDSA-P256 sig verifies
    assert ident.isvprodid == 1
    assert len(ident.tcb_levels) == 6                    # real Intel QE Identity has 6 levels
    # a QE at the top published isvsvn is UpToDate; one below all is OutOfDate
    top = ident.tcb_levels[0]
    assert qe_tcb_status(parse_qe_report_identity(
        _qe_report(mrsigner=ident.mrsigner, isvsvn=top.isvsvn)), ident) == top.status


def test_real_intel_qe_identity_tamper_fails():
    from cryptography import x509
    blob = open(os.path.join(_FIX, "intel_qe_identity.json"), "rb").read()
    signer = x509.load_pem_x509_certificate(
        open(os.path.join(_FIX, "intel_tcb_signing.pem"), "rb").read())
    tampered = blob.replace(b'"isvprodid":1', b'"isvprodid":9', 1)
    with pytest.raises(QeIdentityError):
        parse_and_verify_qe_identity(tampered, signer)


# ── end-to-end through IntelDCAPBackend (QE pinning gates vendor_verified) ───────

def test_dcap_backend_rejects_rogue_qe_and_accepts_genuine():
    """The QE report in build_self_consistent_quote has MRSIGNER=0xdd…, ISVPRODID=0,
    ISVSVN=0. A QE Identity matching that → genuine (vendor_verified True). A QE
    Identity with a different MRSIGNER → rogue QE → vendor_verified False."""
    import sys
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization
    sys.path.insert(0, os.path.dirname(__file__))
    from test_sprint_1044_intel_dcap_verify import build_self_consistent_quote
    from prsm.compute.inference.intel_dcap import IntelDCAPBackend

    quote, root_pem, _expected, _root_priv = build_self_consistent_quote()
    qe_mrsigner = bytes.fromhex("dd" * 32)   # the assembler's QE report MRSIGNER

    signer_priv = ec.generate_private_key(ec.SECP256R1())
    signer_pem = _signer_cert(signer_priv).public_bytes(serialization.Encoding.PEM)
    genuine = _signed_identity(signer_priv, mrsigner=qe_mrsigner, isvprodid=0,
                               levels=[{"tcb": {"isvsvn": 0}, "tcbStatus": "UpToDate"}])
    rogue = _signed_identity(signer_priv, mrsigner=bytes.fromhex("ee" * 32), isvprodid=0,
                             levels=[{"tcb": {"isvsvn": 0}, "tcbStatus": "UpToDate"}])

    ok = IntelDCAPBackend(trusted_root_pem=root_pem, tcb_signer_cert_pem=signer_pem,
                          qe_identity_json=genuine).verify(quote)
    assert ok.vendor_verified is True
    assert ok.vendor_data.get("qe_tcb_status") == "UpToDate"

    bad = IntelDCAPBackend(trusted_root_pem=root_pem, tcb_signer_cert_pem=signer_pem,
                           qe_identity_json=rogue).verify(quote)
    assert bad.vendor_verified is False
    assert "qe" in (bad.error or "").lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
