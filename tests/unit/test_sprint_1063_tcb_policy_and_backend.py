"""Sprint 1063 — TCB status policy + IntelDCAPBackend TCB enforcement (bricks 4+5).

Brick 4: ``TcbPolicy`` — a configurable status floor (default: accept UpToDate +
the *Needed variants, reject OutOfDate*/Revoked). Brick 5: wire it into
IntelDCAPBackend so a validly-SIGNED quote from a DOWNLEVEL (out-of-date) TEE is
rejected (vendor_verified=False) when TCB-Info is configured — and the SAME quote
passes when the platform is up to date. Without TCB-Info configured, behavior is
unchanged (signature-chain only).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.x509.oid import NameOID, ObjectIdentifier

sys.path.insert(0, os.path.dirname(__file__))
from test_sprint_1044_intel_dcap_verify import _mk_cert, _assemble_quote
from test_sprint_1061_pck_sgx_extension import _sgx_ext_der, _SGX_EXT_OID

from prsm.compute.inference.sgx_tcb import TcbPolicy, default_tcb_policy
from prsm.compute.inference.intel_dcap import IntelDCAPBackend

FMSPC = "00906ed50000"


# ── brick 4: policy ────────────────────────────────────────────────────────────

def test_default_policy_accepts_uptodate_rejects_outofdate_and_revoked():
    p = default_tcb_policy()
    assert p.is_acceptable("UpToDate") is True
    assert p.is_acceptable("SWHardeningNeeded") is True
    assert p.is_acceptable("ConfigurationNeeded") is True
    assert p.is_acceptable("OutOfDate") is False
    assert p.is_acceptable("OutOfDateConfigurationNeeded") is False
    assert p.is_acceptable("Revoked") is False


def test_custom_policy_only_uptodate():
    p = TcbPolicy(allowed_statuses=frozenset({"UpToDate"}))
    assert p.is_acceptable("UpToDate") is True
    assert p.is_acceptable("SWHardeningNeeded") is False


def test_unknown_status_is_rejected():
    """An unrecognized status fails closed (never silently accepted)."""
    assert default_tcb_policy().is_acceptable("SomethingNew") is False


# ── brick 5: end-to-end backend enforcement ─────────────────────────────────────

def _raw_sig(priv, msg):
    r, s = decode_dss_signature(priv.sign(msg, ec.ECDSA(hashes.SHA256())))
    return (r.to_bytes(32, "big") + s.to_bytes(32, "big")).hex()


def _signed_tcb_info(priv, levels):
    obj = {"id": "SGX", "version": 3, "fmspc": FMSPC, "pceId": "0000",
           "tcbType": 0, "tcbEvaluationDataNumber": 15, "tcbLevels": levels}
    b = json.dumps(obj, separators=(",", ":")).encode()
    return b'{"tcbInfo":' + b + b',"signature":"' + _raw_sig(priv, b).encode() + b'"}'


def _level(svns, pcesvn, status):
    return {"tcb": {"sgxtcbcomponents": [{"svn": v} for v in svns], "pcesvn": pcesvn},
            "tcbStatus": status}


def _pck_with_sgx_ext(subject, issuer, subject_pub, issuer_priv, *, comp_svns, pcesvn):
    der = _sgx_ext_der(comp_svns, pcesvn, FMSPC)
    return (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject)]))
            .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer)]))
            .public_key(subject_pub).serial_number(x509.random_serial_number())
            .not_valid_before(_dt.datetime(2025, 1, 1))
            .not_valid_after(_dt.datetime(2035, 1, 1))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.UnrecognizedExtension(ObjectIdentifier(_SGX_EXT_OID), der),
                           critical=False)
            .sign(issuer_priv, hashes.SHA256()))


def _build_quote_and_roots(*, platform_svns, platform_pcesvn):
    root_priv = ec.generate_private_key(ec.SECP256R1())
    inter_priv = ec.generate_private_key(ec.SECP256R1())
    pck_priv = ec.generate_private_key(ec.SECP256R1())
    ak_priv = ec.generate_private_key(ec.SECP256R1())
    root_c = _mk_cert("Test SGX Root CA", "Test SGX Root CA", root_priv.public_key(), root_priv, ca=True)
    inter_c = _mk_cert("Test Intermediate CA", "Test SGX Root CA", inter_priv.public_key(), root_priv, ca=True)
    pck_c = _pck_with_sgx_ext("Test PCK Leaf", "Test Intermediate CA", pck_priv.public_key(),
                              inter_priv, comp_svns=platform_svns, pcesvn=platform_pcesvn)
    quote = _assemble_quote(mrenclave=b"\xaa" * 32, mrsigner=b"\xbb" * 32,
                            ak_priv=ak_priv, qe_signer_priv=pck_priv, chain_certs=[pck_c, inter_c])
    return quote, root_c.public_bytes(serialization.Encoding.PEM)


def _tcb_info_and_signer():
    signer_priv = ec.generate_private_key(ec.SECP256R1())
    signer_cert = _mk_cert("Intel TCB Signing", "Intel TCB Signing",
                           signer_priv.public_key(), signer_priv, ca=False)
    levels = [_level([10] * 16, 12, "UpToDate"), _level([8] * 16, 10, "OutOfDate")]
    blob = _signed_tcb_info(signer_priv, levels)
    return blob, signer_cert.public_bytes(serialization.Encoding.PEM)


def test_no_tcb_info_configured_is_chain_only_unchanged():
    quote, root_pem = _build_quote_and_roots(platform_svns=[1] * 16, platform_pcesvn=1)
    res = IntelDCAPBackend(trusted_root_pem=root_pem).verify(quote)
    assert res.vendor_verified is True  # downlevel platform, but no TCB check configured


def test_uptodate_platform_passes_with_tcb_enforcement():
    quote, root_pem = _build_quote_and_roots(platform_svns=[10] * 16, platform_pcesvn=12)
    tcb_blob, signer_pem = _tcb_info_and_signer()
    res = IntelDCAPBackend(trusted_root_pem=root_pem, tcb_info_json=tcb_blob,
                           tcb_signer_cert_pem=signer_pem).verify(quote)
    assert res.vendor_verified is True
    assert res.vendor_data.get("tcb_status") == "UpToDate"


def test_outofdate_platform_is_rejected_by_tcb_enforcement():
    quote, root_pem = _build_quote_and_roots(platform_svns=[8] * 16, platform_pcesvn=10)
    tcb_blob, signer_pem = _tcb_info_and_signer()
    res = IntelDCAPBackend(trusted_root_pem=root_pem, tcb_info_json=tcb_blob,
                           tcb_signer_cert_pem=signer_pem).verify(quote)
    assert res.vendor_verified is False
    assert res.vendor_data.get("tcb_status") == "OutOfDate"
    assert "tcb" in (res.error or "").lower()


def test_tampered_tcb_info_fails_closed():
    quote, root_pem = _build_quote_and_roots(platform_svns=[10] * 16, platform_pcesvn=12)
    tcb_blob, signer_pem = _tcb_info_and_signer()
    tampered = tcb_blob.replace(b'"OutOfDate"', b'"UpToDate!"')
    res = IntelDCAPBackend(trusted_root_pem=root_pem, tcb_info_json=tampered,
                           tcb_signer_cert_pem=signer_pem).verify(quote)
    assert res.vendor_verified is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
