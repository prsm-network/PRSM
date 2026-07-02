"""Sprint 1349 — Tier B/C paid-decrypt consumer arc, brick 1: the offline reconstruct primitive.

Content ciphertext is served freely; the content key is deposited on-chain and released only on
verified payment (KeyDistribution). This brick is the pure, offline core of the consumer
orchestration — wrap the content key to the buyer (publisher/deposit side) and reconstruct the
plaintext from the released key + retrieved ciphertext (consumer side) — so the money/chain wiring
(later bricks) builds on a proven crypto foundation. Fail-loud: wrong key / tamper → raise.
"""
from __future__ import annotations

import dataclasses

import pytest

from prsm.enterprise.recipient_encryption import (
    EnterpriseRecipient,
    generate_recipient_keypair,
)
from prsm.storage.encryption import encrypt, generate_key
from prsm.storage.paid_unlock import (
    PaidUnlockError,
    reconstruct_paid_content,
    wrap_content_key_for_deposit,
)


def _setup(plaintext=b"proprietary NADA dataset rows - Tier B"):
    content_key = generate_key()
    content = encrypt(plaintext, content_key)          # freely-served AES-GCM ciphertext
    buyer_priv, buyer_pub = generate_recipient_keypair()
    wrapped = wrap_content_key_for_deposit(            # the on-chain-deposited encrypted_key
        content_key, [EnterpriseRecipient(identifier="buyer-1", x25519_pubkey_b64=buyer_pub)])
    return content, wrapped, buyer_priv, plaintext


# ── the happy path: pay (simulated) → released key + ciphertext → plaintext ───

def test_round_trip_reconstructs_plaintext():
    content, wrapped, buyer_priv, plaintext = _setup()
    assert reconstruct_paid_content(wrapped, buyer_priv, content) == plaintext


def test_multiple_designated_buyers_each_unlock():
    content_key = generate_key()
    content = encrypt(b"shared licensed dataset", content_key)
    a_priv, a_pub = generate_recipient_keypair()
    b_priv, b_pub = generate_recipient_keypair()
    wrapped = wrap_content_key_for_deposit(content_key, [
        EnterpriseRecipient(identifier="a", x25519_pubkey_b64=a_pub),
        EnterpriseRecipient(identifier="b", x25519_pubkey_b64=b_pub)])
    assert reconstruct_paid_content(wrapped, a_priv, content) == b"shared licensed dataset"
    assert reconstruct_paid_content(wrapped, b_priv, content) == b"shared licensed dataset"


# ── fail-loud: never return garbage ───────────────────────────────────────────

def test_wrong_buyer_key_raises():
    content, wrapped, _buyer_priv, _ = _setup()
    other_priv, _ = generate_recipient_keypair()       # paid for the wrong content / not a buyer
    with pytest.raises(PaidUnlockError, match="unwrap"):
        reconstruct_paid_content(wrapped, other_priv, content)


def test_tampered_ciphertext_raises():
    content, wrapped, buyer_priv, _ = _setup()
    tampered = dataclasses.replace(
        content, ciphertext=bytes([content.ciphertext[0] ^ 0xFF]) + content.ciphertext[1:])
    with pytest.raises(PaidUnlockError, match="tampered|decryption failed"):
        reconstruct_paid_content(wrapped, buyer_priv, tampered)


def test_malformed_wrapped_key_raises():
    content, _wrapped, buyer_priv, _ = _setup()
    with pytest.raises(PaidUnlockError, match="malformed"):
        reconstruct_paid_content(b"}{ not json", buyer_priv, content)


def test_wrap_requires_a_recipient():
    with pytest.raises(PaidUnlockError, match="recipient"):
        wrap_content_key_for_deposit(generate_key(), [])


def test_wrapped_key_is_json_bytes_not_the_raw_key():
    """The deposited encrypted_key must NOT leak the content key in the clear."""
    content_key = generate_key()
    _, pub = generate_recipient_keypair()
    wrapped = wrap_content_key_for_deposit(
        content_key, [EnterpriseRecipient(identifier="b", x25519_pubkey_b64=pub)])
    assert content_key.key_bytes not in wrapped        # the raw key never appears
    import json
    assert "manifest" in json.loads(wrapped)           # it's the recipient-wrap envelope


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
