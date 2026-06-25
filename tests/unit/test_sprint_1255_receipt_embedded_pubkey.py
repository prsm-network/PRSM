"""Sprint 1255 — embed the settler pubkey in the InferenceReceipt so it's
INDEPENDENTLY verifiable offline (the §7 "trust the math, not the provider" thesis).

Surfaced by the live front-door validation: `verify-receipt` required the signer's
pubkey to be fetched out-of-band, which undercuts independent third-party
verification. The receipt now carries `settler_pubkey_b64`; verify_receipt falls back
to it when no pubkey is supplied, BUT binds it to settler_node_id
(node_id == sha256(pubkey)[:32], which IS signed) so a swapped embedded pubkey is
rejected. The field is NOT in signing_payload — pre-1255 canonical bytes are unchanged.
"""
from __future__ import annotations

import base64
import dataclasses
import hashlib

import pytest

from prsm.compute.inference.models import ContentTier, InferenceReceipt
from prsm.compute.inference.receipt import sign_receipt, verify_receipt
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import generate_node_identity, node_id_for_public_key


def _unsigned():
    return InferenceReceipt(
        job_id="j1", request_id="r1", model_id="Qwen/Qwen2.5-7B-Instruct",
        content_tier=ContentTier.A, privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0, tee_type=TEEType.SOFTWARE, tee_attestation=b"",
        output_hash=hashlib.sha256(b"Paris").digest(),
        duration_seconds=0.1, cost_ftns="0.12",
        settler_signature=b"", settler_node_id="",
    )


def test_sign_embeds_settler_pubkey():
    ident = generate_node_identity("settler")
    r = sign_receipt(_unsigned(), ident)
    assert r.settler_pubkey_b64 == ident.public_key_b64
    assert r.settler_node_id == ident.node_id


def test_verify_with_embedded_pubkey_no_arg():
    # THE fix: a genuine receipt verifies offline with NO pubkey supplied.
    ident = generate_node_identity("settler")
    r = sign_receipt(_unsigned(), ident)
    assert verify_receipt(r) is True            # uses embedded pubkey + binding


def test_verify_explicit_pubkey_still_works():
    # back-compat: supplying the pubkey explicitly still verifies.
    ident = generate_node_identity("settler")
    r = sign_receipt(_unsigned(), ident)
    assert verify_receipt(r, public_key_b64=ident.public_key_b64) is True
    assert verify_receipt(r, identity=ident) is True


def test_swapped_embedded_pubkey_rejected_by_node_id_binding():
    # attacker swaps the embedded pubkey to THEIR key while keeping the victim's
    # settler_node_id → binding (sha256(their_pub)[:32] != victim node_id) → REJECT.
    victim = generate_node_identity("victim")
    attacker = generate_node_identity("attacker")
    r = sign_receipt(_unsigned(), victim)
    assert node_id_for_public_key(base64.b64decode(attacker.public_key_b64)) != victim.node_id
    tampered = dataclasses.replace(r, settler_pubkey_b64=attacker.public_key_b64)
    assert verify_receipt(tampered) is False


def test_no_embedded_and_no_arg_returns_false():
    # a pre-1255 receipt (no embedded pubkey) with no pubkey supplied can't verify.
    ident = generate_node_identity("settler")
    r = sign_receipt(_unsigned(), ident)
    stripped = dataclasses.replace(r, settler_pubkey_b64=None)
    assert verify_receipt(stripped) is False
    # ...but still verifies when the pubkey is supplied explicitly (back-compat)
    assert verify_receipt(stripped, public_key_b64=ident.public_key_b64) is True


def test_signing_payload_byte_identical_with_or_without_pubkey():
    # the embedded pubkey must NOT change the canonical signing bytes (so pre-1255
    # receipts + verifiers stay byte-compatible, and the field can't be a tamper vector
    # via the signature path — it's guarded by the node_id binding instead).
    ident = generate_node_identity("settler")
    r = sign_receipt(_unsigned(), ident)
    without = dataclasses.replace(r, settler_pubkey_b64=None)
    assert r.signing_payload() == without.signing_payload()


def test_to_dict_roundtrip_preserves_pubkey_and_verifies():
    ident = generate_node_identity("settler")
    r = sign_receipt(_unsigned(), ident)
    restored = InferenceReceipt.from_dict(r.to_dict())
    assert restored.settler_pubkey_b64 == ident.public_key_b64
    assert verify_receipt(restored) is True


def test_tampered_signed_field_still_fails_with_embedded_pubkey():
    # binding holding doesn't rescue a tampered SIGNED field — the signature fails.
    ident = generate_node_identity("settler")
    r = sign_receipt(_unsigned(), ident)
    tampered = dataclasses.replace(r, output_hash=hashlib.sha256(b"London").digest())
    assert verify_receipt(tampered) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
