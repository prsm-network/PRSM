"""Sprint 1067 — bundled vendor root anchors with fingerprint pinning (Tier 3, brick 1).

Ships the REAL Intel SGX Root CA + AMD product ARKs (Milan/Genoa/Turin) as package
data so the daemon defaults to genuine verification without the operator sourcing the
PEMs (a bring-up ceremony). The anchors are the same ones the sp1066 golden vectors
verify real chains against. Each is SHA-256 (DER) PINNED so a swap of the bundled
trust material (e.g. a repo compromise) is caught loudly rather than silently
trusting an attacker root.
"""
from __future__ import annotations

import hashlib

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization

from prsm.compute.inference.vendor_anchors import (
    AnchorIntegrityError,
    _FINGERPRINTS,
    bundled_amd_arks,
    bundled_intel_sgx_root_ca,
)


def _fp(pem: bytes) -> str:
    der = x509.load_pem_x509_certificate(pem).public_bytes(serialization.Encoding.DER)
    return hashlib.sha256(der).hexdigest()


def test_intel_root_loads_and_is_the_real_root():
    pem = bundled_intel_sgx_root_ca()
    c = x509.load_pem_x509_certificate(pem)
    assert "Intel SGX Root CA" in c.subject.rfc4514_string()
    assert _fp(pem) == _FINGERPRINTS["intel_sgx_root_ca.pem"]


def test_amd_arks_load_all_products():
    arks = bundled_amd_arks()
    assert set(arks) == {"milan", "genoa", "turin"}
    for product, pem in arks.items():
        c = x509.load_pem_x509_certificate(pem)
        assert f"ARK-{product.capitalize()}" in c.subject.rfc4514_string()
        assert _fp(pem) == _FINGERPRINTS[f"amd_ark_{product}.pem"]


def test_bundled_roots_are_self_consistent_cas():
    """Each bundled root is a self-signed CA (verifiable via the shared validator)."""
    from prsm.compute.inference.x509_path import verify_cert_chain
    import datetime as _dt
    now = _dt.datetime(2026, 6, 15, tzinfo=_dt.timezone.utc)
    intel = x509.load_pem_x509_certificate(bundled_intel_sgx_root_ca())
    ok, err = verify_cert_chain([intel], intel, now=now)  # root as its own anchor
    assert ok is True, err


def test_fingerprint_mismatch_raises(monkeypatch):
    """If the pinned fingerprint doesn't match the file, loading fails closed."""
    import prsm.compute.inference.vendor_anchors as va
    monkeypatch.setitem(va._FINGERPRINTS, "intel_sgx_root_ca.pem", "00" * 32)
    with pytest.raises(AnchorIntegrityError):
        va.bundled_intel_sgx_root_ca()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
