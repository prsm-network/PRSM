"""Sprint 433 — live-verified §7 receipt-verification roundtrip.

The §7 Private Inference truth-surfacing claim says: a caller can
construct an `InferenceReceipt`, sign it with the settler's
ed25519 key, and an independent verifier (via
`POST /compute/receipt/verify`) can confirm:

  1. Signature is cryptographically valid (or not, on tamper)
  2. DP-noise was applied per the claimed tier
  3. Hardware attestation is present (vendor verification is
     honest-scope deferred — `attestation_vendor_verified=False`
     until real DCAP/KDS keys are wired)
  4. Multi-stage envelope is present
  5. Activation-noise-trace is structurally valid
  6. Topology assignment is structurally valid + history-distinct

Sprint 433 was the priority-#4 verification-campaign sprint:
ran the full path end-to-end against a live daemon. The verify
endpoint:

- Honest receipt → `ok=true`, all checks pass
- Tampered receipt (one field flipped) → `signature_valid=false`

These pins capture the schema-level invariants the roundtrip
proved. They don't replace the live test (the local schema can
shift without the endpoint's parser noticing) but they pin the
in-process bytes-exact path.
"""
from __future__ import annotations

import base64
import dataclasses
from decimal import Decimal

import pytest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from prsm.compute.inference.models import (
    ContentTier, InferenceReceipt,
)
from prsm.compute.inference.privacy_verification import (
    verify_receipt_privacy_claim,
)
from prsm.compute.tee.models import PrivacyLevel, TEEType


def _make_signed_receipt():
    """Build a receipt and sign it. Returns (receipt, pubkey)."""
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    pub_bytes = pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    base = InferenceReceipt(
        job_id="job-test",
        request_id="req-test",
        model_id="test-model-v1",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.STANDARD,
        epsilon_spent=8.0,  # matches expected ε for standard tier
        tee_type=TEEType.NONE,
        tee_attestation=b"stub-attestation",
        output_hash=b"\x00" * 32,
        duration_seconds=1.0,
        cost_ftns=Decimal("0.50"),
        settler_signature=b"",
        settler_node_id="test-settler",
    )
    sig = priv.sign(base.signing_payload())
    signed = dataclasses.replace(base, settler_signature=sig)
    return signed, pub_bytes


def test_honest_receipt_verifies_clean():
    """The headline §7 claim: an honestly-signed receipt
    passes the independent verification path with all checks
    green. Sprint 433 live-verified this against a running
    daemon; this pin keeps the in-process schema honest."""
    receipt, pubkey = _make_signed_receipt()
    result = verify_receipt_privacy_claim(
        receipt,
        public_key_b64=base64.b64encode(pubkey).decode(),
    )
    assert result.signature_valid is True
    assert result.ok is True
    # `reasons` should be empty when ok=true.
    assert result.reasons == []
    assert result.dp_noise_applied is True
    # sp1256 — this fixture builds a SINGLE-stage receipt (tee_attestation is a stub,
    # not a per-stage chain envelope), so multi_stage_envelope_present is correctly
    # False. The original `is True` assertion reflected the pre-sp1236 permissive logic
    # where ANY non-empty attestation counted as multi-stage; the TEE attestation work
    # (sp1236+) tightened is_multi_stage_attestation to detect only real envelopes.
    assert result.multi_stage_envelope_present is False


def test_tampered_receipt_signature_fails():
    """Tampering ANY signed field must break the signature.
    Sprint 433's live test flipped `epsilon_spent` from 0.5
    to 99.9 — the verifier returned `signature_valid=false`.
    This pins the same defense against in-process replay
    attacks."""
    receipt, pubkey = _make_signed_receipt()
    # Tamper: change epsilon_spent (which is part of
    # signing_payload). The signature was generated over
    # the original bytes; flipping a byte must break verify.
    tampered = dataclasses.replace(
        receipt, epsilon_spent=99.9,
    )
    result = verify_receipt_privacy_claim(
        tampered, public_key_b64=base64.b64encode(pubkey).decode(),
    )
    assert result.signature_valid is False
    assert any(
        "signature" in r.lower() for r in result.reasons
    ), (
        "tampered receipt verification must surface a "
        "signature failure in `reasons`"
    )
    assert result.ok is False


def test_tampered_output_hash_breaks_signature():
    """Defensive: tampering the output_hash (one of the
    cryptographically-bound fields) must break the signature.
    Pre-3.x.1, output_hash wasn't part of the signed payload
    so an attacker could swap outputs without invalidating
    the receipt — that's the bug the §7 design fixes."""
    receipt, pubkey = _make_signed_receipt()
    tampered = dataclasses.replace(
        receipt, output_hash=b"\xff" * 32,  # different output
    )
    result = verify_receipt_privacy_claim(
        tampered, public_key_b64=base64.b64encode(pubkey).decode(),
    )
    assert result.signature_valid is False


def test_wrong_pubkey_fails_verification():
    """Verifying with the wrong public key must fail.
    Sanity check: this is what would happen if an attacker
    forged a settler_node_id but the caller checked the
    signature against the legitimate settler's key."""
    receipt, _correct_pubkey = _make_signed_receipt()
    # Generate a different keypair
    other_priv = Ed25519PrivateKey.generate()
    other_pub = other_priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    result = verify_receipt_privacy_claim(
        receipt, public_key_b64=base64.b64encode(other_pub).decode(),
    )
    assert result.signature_valid is False


def test_attestation_vendor_verified_honest_scope_deferral():
    """§7 honest-scope: real cryptographic vendor verification
    of TEE attestation chains is deferred. The verifier
    surfaces `attestation_vendor_verified=False` to make the
    deferral visible — callers must not interpret a
    structurally-valid attestation as cryptographically
    verified by the vendor."""
    receipt, pubkey = _make_signed_receipt()
    result = verify_receipt_privacy_claim(
        receipt,
        public_key_b64=base64.b64encode(pubkey).decode(),
    )
    # Stub attestation bytes (b"stub-attestation") cannot
    # match a real Intel ASP / AMD KDS parse → vendor stays
    # `unknown` and unverified.
    assert result.attestation_vendor_verified is False


def test_signing_payload_excludes_signature_field():
    """The signing payload MUST NOT include the signature
    field (would be circular). This is the cryptographic
    invariant that makes the signing scheme well-formed."""
    receipt, _ = _make_signed_receipt()
    payload = receipt.signing_payload()
    # signature bytes are random per generation; check they
    # don't appear verbatim in the canonical bytes
    assert receipt.settler_signature not in payload
    # signing_payload is deterministic for fixed input
    assert payload == receipt.signing_payload()
