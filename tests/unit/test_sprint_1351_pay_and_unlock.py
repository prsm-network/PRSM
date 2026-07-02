"""Sprint 1351 (+1359 F1 redesign) — brick 3: pay -> unlock orchestration.

Composes fee-settlement (injectable) + brick 2 (fetch the wrapped key off-chain + verify vs the
on-chain commitment) + retrieval + brick 1 (reconstruct). Tested end-to-end with REAL crypto; the
wrapped key comes from an injected fetch (simulating the payment-gated serve endpoint, sp1358).
"""
from __future__ import annotations

import pytest

from prsm.economy.paid_content import pay_and_unlock
from prsm.economy.web3.key_acquisition import (
    KeyCommitmentMismatchError,
    KeyNotReleasedError,
)
from prsm.enterprise.recipient_encryption import (
    EnterpriseRecipient,
    generate_recipient_keypair,
)
from prsm.storage.encryption import encrypt, generate_key
from prsm.storage.paid_unlock import (
    PaidUnlockError,
    key_commitment,
    wrap_content_key_for_deposit,
)

_CH = bytes.fromhex("cd" * 32)


def _fixture(plaintext=b"paywalled dataset rows"):
    content_key = generate_key()
    content = encrypt(plaintext, content_key)
    buyer_priv, buyer_pub = generate_recipient_keypair()
    wrapped = wrap_content_key_for_deposit(
        content_key, [EnterpriseRecipient(identifier="buyer", x25519_pubkey_b64=buyer_pub)])
    return content, wrapped, key_commitment(wrapped), buyer_priv, plaintext


def _call(content, wrapped, commitment, buyer_priv, *, settle_fee=None,
          fetch=None, retrieve=None):
    return pay_and_unlock(
        content_hash=_CH, recipient_privkey_b64=buyer_priv, commitment=commitment,
        fetch_wrapped_key=fetch if fetch is not None else (lambda ch: wrapped),
        retrieve_content=retrieve if retrieve is not None else (lambda ch: content),
        settle_fee=settle_fee)


# ── happy path ────────────────────────────────────────────────────────────────

def test_full_flow_pays_then_fetches():
    content, wrapped, commitment, buyer_priv, plaintext = _fixture()
    order = []
    out = _call(content, wrapped, commitment, buyer_priv,
                settle_fee=lambda: order.append("pay"),
                fetch=lambda ch: (order.append("fetch"), wrapped)[1])
    assert out == plaintext
    assert order == ["pay", "fetch"]                   # fee settled BEFORE fetching the key


def test_already_paid_no_settle():
    content, wrapped, commitment, buyer_priv, plaintext = _fixture()
    out = _call(content, wrapped, commitment, buyer_priv, settle_fee=None)
    assert out == plaintext


def test_authoritative_commitment_read_after_payment():
    # sp1363 (R5 MEDIUM): when no commitment is supplied, it's read via fetch_commitment AFTER
    # settle (the on-chain release is gated on payment) — never trusted from the serve path.
    content, wrapped, commitment, buyer_priv, plaintext = _fixture()
    order = []
    out = pay_and_unlock(
        content_hash=_CH, recipient_privkey_b64=buyer_priv,
        fetch_wrapped_key=lambda ch: wrapped, retrieve_content=lambda ch: content,
        fetch_commitment=lambda: (order.append("read-commitment"), commitment)[1],
        settle_fee=lambda: order.append("pay"))
    assert out == plaintext
    assert order == ["pay", "read-commitment"]         # commitment read AFTER payment


def test_missing_both_commitment_and_fetch_raises():
    content, wrapped, _commitment, buyer_priv, _ = _fixture()
    with pytest.raises(PaidUnlockError, match="commitment or fetch_commitment"):
        pay_and_unlock(content_hash=_CH, recipient_privkey_b64=buyer_priv,
                       fetch_wrapped_key=lambda ch: wrapped, retrieve_content=lambda ch: content)


# ── fail-loud at each stage ───────────────────────────────────────────────────

def test_unpaid_fetch_returns_nothing_surfaces_key_not_released():
    content, wrapped, commitment, buyer_priv, _ = _fixture()
    # the payment-gated endpoint serves nothing to an unpaid fetcher
    with pytest.raises(KeyNotReleasedError, match="no key"):
        _call(content, wrapped, commitment, buyer_priv, fetch=lambda ch: None)


def test_wrong_served_key_fails_commitment_check():
    content, wrapped, commitment, buyer_priv, _ = _fixture()
    with pytest.raises(KeyCommitmentMismatchError, match="WRONG key"):
        _call(content, wrapped, commitment, buyer_priv, fetch=lambda ch: b"a substituted key")


def test_ciphertext_not_retrievable_raises():
    content, wrapped, commitment, buyer_priv, _ = _fixture()
    with pytest.raises(PaidUnlockError, match="not retrievable"):
        _call(content, wrapped, commitment, buyer_priv, retrieve=lambda ch: None)


def test_wrong_buyer_key_raises():
    content, wrapped, commitment, _buyer_priv, _ = _fixture()
    other_priv, _ = generate_recipient_keypair()
    with pytest.raises(PaidUnlockError, match="unwrap"):
        _call(content, wrapped, commitment, other_priv)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
