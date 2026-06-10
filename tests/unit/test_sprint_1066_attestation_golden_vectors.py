"""Sprint 1066 — REAL-collateral golden-vector tests (Tier 3, retires the honest-scope caveat).

The sp1044-1065 attestation engines were hermetically tested against self-built
fixtures, with a standing caveat: the parsers/offsets/encodings must be validated
against REAL Intel + AMD production collateral before trusting them in prod. This
suite does that, using real material fetched from Intel PCS / AMD KDS and committed
under tests/fixtures/attestation/ (all PUBLIC trust anchors / signed collateral — no
secrets). Provenance: tests/fixtures/attestation/README.md.

Validated against REAL data:
  - Intel signed TCB-Info (fmspc 00906ed50000, v3, 21 levels): parse + ECDSA-P256
    signature-verify against the real Intel SGX TCB Signing cert (sp1062).
  - Intel PCK chain (Platform CA -> Root CA, ECDSA): the shared path-validator (sp1049).
  - Intel PCK Platform CRL (57 real revocations): parse + signature-valid under the
    Platform CA (sp1060).
  - AMD Milan ASK -> ARK chain (RSA-4096 / RSA-PSS): caught + fixed sp1066 — the
    validator hardcoded ec.ECDSA and would have REJECTED the real AMD chain on
    hardware; now dispatches on issuer key type.

``now`` is pinned inside the fixtures' validity windows so the suite is deterministic
and won't rot when the live collateral rotates.
"""
from __future__ import annotations

import datetime as _dt
import os
import re

import pytest
from cryptography import x509

from prsm.compute.inference.sgx_tcb import (
    PckTcb, TcbInfoError, evaluate_tcb_status, parse_and_verify_tcb_info,
)
from prsm.compute.inference.x509_path import _find_valid_crl, verify_cert_chain

_FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures", "attestation")
# inside every fixture's validity window (certs: years; CRL: 2026-06-10..07-10)
_NOW = _dt.datetime(2026, 6, 15, tzinfo=_dt.timezone.utc)


def _read(name):
    with open(os.path.join(_FIX, name), "rb") as f:
        return f.read()


def _certs(name):
    t = _read(name).decode()
    return [x509.load_pem_x509_certificate((c + "\n").encode())
            for c in re.findall(r'-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----', t, re.S)]


# ── Intel TCB-Info (real signed collateral) ─────────────────────────────────────

def test_real_intel_tcb_info_parses_and_verifies():
    blob = _read("intel_tcb_info_00906ed50000.json")
    signer = _certs("intel_tcb_signing.pem")[0]
    assert "Intel SGX TCB Signing" in signer.subject.rfc4514_string()
    info = parse_and_verify_tcb_info(blob, signer)
    assert info.fmspc.hex() == "00906ed50000"
    assert len(info.tcb_levels) == 21               # real Intel collateral has 21 levels
    assert len(info.tcb_levels[0].comp_svns) == 16  # 16 sgxtcbcomponents each


def test_real_intel_tcb_info_status_evaluation():
    """A platform meeting the top published level gets that level's real status."""
    blob = _read("intel_tcb_info_00906ed50000.json")
    info = parse_and_verify_tcb_info(blob, _certs("intel_tcb_signing.pem")[0])
    top = info.tcb_levels[0]
    pck = PckTcb(fmspc=info.fmspc, comp_svns=list(top.comp_svns), pcesvn=top.pcesvn)
    assert evaluate_tcb_status(pck, info) == top.status
    # an all-zero platform is below every real level → conservative OutOfDate
    low = PckTcb(fmspc=info.fmspc, comp_svns=[0] * 16, pcesvn=0)
    assert evaluate_tcb_status(low, info) == "OutOfDate"


def test_real_intel_tcb_info_tamper_fails_signature():
    """Mutating any byte of the real signed tcbInfo breaks the real signature."""
    blob = _read("intel_tcb_info_00906ed50000.json")
    signer = _certs("intel_tcb_signing.pem")[0]
    tampered = blob.replace(b'"SWHardeningNeeded"', b'"UpToDate........"', 1)
    assert tampered != blob
    with pytest.raises(TcbInfoError):
        parse_and_verify_tcb_info(tampered, signer)


def test_real_intel_tcb_info_wrong_signer_fails():
    blob = _read("intel_tcb_info_00906ed50000.json")
    wrong = _certs("intel_sgx_root_ca.pem")[0]   # a real cert, but not the TCB signer
    with pytest.raises(TcbInfoError):
        parse_and_verify_tcb_info(blob, wrong)


# ── Intel PCK chain + CRL (real) ────────────────────────────────────────────────

def test_real_intel_pck_chain_verifies():
    platform_ca = _certs("intel_sgx_platform_ca.pem")[0]
    root_ca = _certs("intel_sgx_root_ca.pem")[0]
    ok, err = verify_cert_chain([platform_ca], root_ca, now=_NOW)
    assert ok is True, err


def test_real_intel_pck_crl_is_accepted_and_signed_by_platform_ca():
    platform_ca = _certs("intel_sgx_platform_ca.pem")[0]
    crl = x509.load_pem_x509_crl(_read("intel_pck_platform.crl.pem"))
    found = _find_valid_crl([crl], platform_ca, _NOW)
    assert found is not None                      # valid + signature-valid under Platform CA
    assert len(list(crl)) > 0                     # real CRL has revocations


# ── AMD chain (real RSA-PSS — the sp1066 regression) ────────────────────────────

def test_real_amd_milan_chain_verifies_rsa_pss():
    """Regression for sp1066: real AMD ARK/ASK are RSA-4096 (RSA-PSS). A validator
    that hardcodes ec.ECDSA rejects them — this asserts the real chain verifies."""
    ask, ark = _certs("amd_milan_ask_ark.pem")
    assert "SEV-Milan" in ask.subject.rfc4514_string()
    assert "ARK-Milan" in ark.subject.rfc4514_string()
    ok, err = verify_cert_chain([ask], ark, now=_NOW)
    assert ok is True, err


def test_real_amd_ark_self_signature_checked():
    """The validator verifies the trusted root's own (RSA-PSS) self-signature; a real
    ARK is self-consistent."""
    ask, ark = _certs("amd_milan_ask_ark.pem")
    ok, err = verify_cert_chain([ask], ark, now=_NOW)
    assert ok is True, err


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
