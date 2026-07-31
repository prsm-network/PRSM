"""Sprint 1495 — a paid TEE dispatch must get a REAL attestation, not any truthy value.

The check was `if not receipt.get("tee_attestation"): raise` — PRESENCE only. It
confirmed the field existed, never that it attested to anything. So a provider
could satisfy a requester's PAID `require_tee` with the string "yes", or with the
dev-only software-fallback blob this repo ships for local testing.

That is the hollowest possible defence: the requester pays a TEE-tier price for a
confidentiality guarantee nobody checked, and it looks like it is working right up
until it matters. The repo already had the real machinery — verify_attestation
(Intel SGX DCAP + AMD SEV-SNP, cryptographic vendor_verified) and the tiered
tee_policy engine, used by three other call sites. The paid path just ignored it.
"""
from __future__ import annotations

import logging

import pytest

from prsm.compute.inference.attestation_backends import (
    AttestationVerificationResult,
)
from prsm.compute.remote_dispatcher import (
    MissingAttestationError,
    _enforce_tee_attestation,
)

NODE = "d437aa67d99cff4a6a17179f5c731b77"


def _patch(monkeypatch, vendor, verified=False, error=None):
    """Force verify_attestation to report a given tier."""
    res = AttestationVerificationResult(
        vendor=vendor, vendor_verified=verified, error=error)
    monkeypatch.setattr(
        "prsm.compute.inference.attestation_backends.verify_attestation",
        lambda blob: res)


# ── the hole this closes ────────────────────────────────────────────

def test_an_arbitrary_truthy_string_no_longer_passes():
    """★ THE bug. 'yes' satisfied the old presence check completely."""
    with pytest.raises(MissingAttestationError, match="not decodable as bytes"):
        _enforce_tee_attestation("yes", 0, NODE)


def test_the_dev_SOFTWARE_FALLBACK_blob_is_REJECTED(monkeypatch):
    """★ The tracked regression: this repo ships a software-fallback attestation
    for local testing, and it used to satisfy a paid TEE requirement."""
    _patch(monkeypatch, "software-fallback")
    with pytest.raises(MissingAttestationError, match="software-fallback"):
        _enforce_tee_attestation(b"\x01" * 64, 0, NODE)


def test_an_unknown_vendor_is_rejected(monkeypatch):
    _patch(monkeypatch, "unknown")
    with pytest.raises(MissingAttestationError, match="below the required"):
        _enforce_tee_attestation(b"\x01" * 64, 0, NODE)


def test_a_verifier_error_is_rejected(monkeypatch):
    """Fails toward refusing — an attestation we could not check is not proof."""
    _patch(monkeypatch, "intel-sgx", error="signature chain broken")
    with pytest.raises(MissingAttestationError, match="below the required"):
        _enforce_tee_attestation(b"\x01" * 64, 0, NODE)


def test_a_raising_verifier_is_rejected(monkeypatch):
    def boom(blob):
        raise RuntimeError("parser exploded")
    monkeypatch.setattr(
        "prsm.compute.inference.attestation_backends.verify_attestation", boom)
    with pytest.raises(MissingAttestationError, match="below the required"):
        _enforce_tee_attestation(b"\x01" * 64, 0, NODE)


# ── still accepts genuine hardware ──────────────────────────────────

def test_real_sgx_hardware_is_accepted_even_without_vendor_roots(monkeypatch, caplog):
    """★ The deliberate calibration. Minimum is HARDWARE_UNVERIFIED, not
    HARDWARE_VERIFIED, so a genuine SGX quote still works on an operator who has
    not configured vendor root CAs — but the gap is LOGGED."""
    _patch(monkeypatch, "intel-sgx", verified=False)
    with caplog.at_level(logging.WARNING):
        _enforce_tee_attestation(b"\x01" * 64, 3, NODE)
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "HARDWARE_UNVERIFIED" in msg
    assert "NOT cryptographically verified" in msg


def test_fully_verified_hardware_is_accepted_quietly(monkeypatch, caplog):
    _patch(monkeypatch, "amd-sev-snp", verified=True)
    with caplog.at_level(logging.WARNING):
        _enforce_tee_attestation(b"\x01" * 64, 0, NODE)
    assert not [r for r in caplog.records if "HARDWARE_UNVERIFIED" in r.getMessage()]


# ── the missing-field case still works ──────────────────────────────

def test_a_missing_attestation_still_raises():
    for empty in (None, b"", ""):
        with pytest.raises(MissingAttestationError, match="no\n?\\s*tee_attestation|no tee_attestation"):
            _enforce_tee_attestation(empty, 0, NODE)


# ── encoding tolerance, without weakening the gate ──────────────────

def test_hex_and_base64_receipts_are_decoded(monkeypatch):
    """Receipts cross a JSON boundary, so bytes arrive as text."""
    _patch(monkeypatch, "intel-sgx", verified=True)
    _enforce_tee_attestation("0x" + "01" * 64, 0, NODE)
    import base64
    _enforce_tee_attestation(base64.b64encode(b"\x01" * 64).decode(), 0, NODE)


# ── operator can tighten, and a bad value fails safe ────────────────

def test_operator_can_require_full_verification(monkeypatch):
    _patch(monkeypatch, "intel-sgx", verified=False)
    monkeypatch.setenv("PRSM_TEE_MIN_TIER", "hardware_verified")
    with pytest.raises(MissingAttestationError, match="hardware_verified"):
        _enforce_tee_attestation(b"\x01" * 64, 0, NODE)


def test_a_garbage_min_tier_falls_back_to_the_SAFE_default(monkeypatch):
    """★ A typo in the env must not silently downgrade to accepting anything."""
    _patch(monkeypatch, "software-fallback")
    monkeypatch.setenv("PRSM_TEE_MIN_TIER", "lol-whatever")
    with pytest.raises(MissingAttestationError):
        _enforce_tee_attestation(b"\x01" * 64, 0, NODE)


# ── wired into the dispatch path ────────────────────────────────────

def test_the_dispatch_path_CALLS_the_enforcer():
    """★ Binding test — an enforcer nothing calls leaves the presence check live."""
    import inspect

    from prsm.compute.remote_dispatcher import RemoteShardDispatcher

    src = inspect.getsource(RemoteShardDispatcher)
    assert "_enforce_tee_attestation(" in src
    assert 'require_tee_attestation and not receipt.get("tee_attestation")' not in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
