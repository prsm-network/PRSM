"""Sprint 1357 — F1 redesign R1: commitment + fetch-and-verify (pure, offline building blocks).

The B5 F1 critical: the wrapped key must not live in world-readable on-chain storage. R1 adds the
primitives for the fix — key_commitment (only this goes on-chain) and fetch_and_verify_wrapped_key
(the consumer fetches the key off-chain and verifies it against the commitment before trusting it),
so a lying publisher can't serve a wrong-but-usable key. These land green without changing the flow.
"""
from __future__ import annotations

import pytest

from prsm.economy.web3.key_acquisition import (
    KeyCommitmentMismatchError,
    KeyNotReleasedError,
    fetch_and_verify_wrapped_key,
)
from prsm.enterprise.recipient_encryption import (
    EnterpriseRecipient,
    generate_recipient_keypair,
)
from prsm.storage.encryption import encrypt, generate_key
from prsm.storage.paid_unlock import (
    key_commitment,
    reconstruct_paid_content,
    verify_key_commitment,
    wrap_content_key_for_deposit,
)

_CH = bytes.fromhex("ab" * 32)


# ── key_commitment / verify_key_commitment ────────────────────────────────────

def test_commitment_is_deterministic_32_bytes():
    w = b"some wrapped key bytes"
    c = key_commitment(w)
    assert isinstance(c, bytes) and len(c) == 32
    assert key_commitment(w) == c                      # deterministic


def test_commitment_changes_with_the_key():
    assert key_commitment(b"key-A") != key_commitment(b"key-B")


def test_verify_commitment_true_only_for_the_matching_key():
    w = b"wrapped"
    assert verify_key_commitment(w, key_commitment(w)) is True
    assert verify_key_commitment(b"other", key_commitment(w)) is False


# ── fetch_and_verify_wrapped_key ──────────────────────────────────────────────

def test_fetch_and_verify_returns_key_when_commitment_matches():
    wrapped = b"the real wrapped key"
    got = fetch_and_verify_wrapped_key(lambda ch: wrapped, _CH, key_commitment(wrapped))
    assert got == wrapped


def test_fetch_and_verify_rejects_a_wrong_key():
    wrapped = b"the real wrapped key"
    commitment = key_commitment(wrapped)
    with pytest.raises(KeyCommitmentMismatchError, match="WRONG key"):
        fetch_and_verify_wrapped_key(lambda ch: b"a substituted key", _CH, commitment)


def test_fetch_and_verify_empty_fetch_is_key_not_released():
    with pytest.raises(KeyNotReleasedError, match="no key"):
        fetch_and_verify_wrapped_key(lambda ch: None, _CH, key_commitment(b"x"))


def test_fetch_and_verify_rejects_malformed_commitment():
    with pytest.raises(KeyCommitmentMismatchError, match="32 bytes"):
        fetch_and_verify_wrapped_key(lambda ch: b"k", _CH, b"short")


# ── end-to-end: commit → fetch+verify → reconstruct (ties to B1) ──────────────

def test_commitment_flow_reconstructs_plaintext():
    plaintext = b"proprietary rows behind the binding gate"
    content_key = generate_key()
    content = encrypt(plaintext, content_key)
    buyer_priv, buyer_pub = generate_recipient_keypair()
    wrapped = wrap_content_key_for_deposit(
        content_key, [EnterpriseRecipient(identifier="b", x25519_pubkey_b64=buyer_pub)])

    commitment = key_commitment(wrapped)               # deposited on-chain (only this)
    # consumer fetches the wrapped key off-chain (simulated) and verifies against the commitment
    fetched = fetch_and_verify_wrapped_key(lambda ch: wrapped, _CH, commitment)
    assert reconstruct_paid_content(fetched, buyer_priv, content) == plaintext


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
