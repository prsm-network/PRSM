"""Sprint 1351 — Tier B/C paid-decrypt consumer arc, brick 3: pay -> unlock orchestration.

Composes the fee-settlement (injectable) + brick 2 (acquire the released key) + retrieval + brick 1
(reconstruct) into the flagship paid-content action. Tested end-to-end with REAL crypto (B1) and
the REAL acquisition (B2) over a mocked KeyDistribution client — the live chain is user-gated.
"""
from __future__ import annotations

import pytest

from prsm.economy.paid_content import pay_and_unlock
from prsm.economy.web3.key_acquisition import KeyNotReleasedError
from prsm.economy.web3.key_distribution import PaymentNotVerifiedError
from prsm.enterprise.recipient_encryption import (
    EnterpriseRecipient,
    generate_recipient_keypair,
)
from prsm.storage.encryption import encrypt, generate_key
from prsm.storage.paid_unlock import PaidUnlockError, wrap_content_key_for_deposit

_CH = bytes.fromhex("cd" * 32)
_BUYER_ETH = "0x" + "22" * 20


class _Ev:
    def __init__(self, content_hash, recipient, encrypted_key):
        self.content_hash, self.recipient, self.encrypted_key = (
            content_hash, recipient, encrypted_key)


class _FakeClient:
    def __init__(self, *, released=None, release_error=None, emit=None, order=None):
        self._released = list(released or [])
        self._release_error = release_error
        self._emit = emit
        self._order = order

    def latest_block(self):
        return 100

    def get_key_released_events(self, fb, tb, *, argument_filters=None):
        return list(self._released)

    def release(self, ch, recipient):
        if self._order is not None:
            self._order.append("release")
        if self._release_error is not None:
            raise self._release_error
        if self._emit is not None:
            self._released.append(self._emit)
        return ("0xtx", None)


def _fixture(plaintext=b"paywalled dataset rows"):
    content_key = generate_key()
    content = encrypt(plaintext, content_key)
    buyer_priv, buyer_pub = generate_recipient_keypair()
    wrapped = wrap_content_key_for_deposit(
        content_key, [EnterpriseRecipient(identifier="buyer", x25519_pubkey_b64=buyer_pub)])
    return content, wrapped, buyer_priv, plaintext


def _call(client, buyer_priv, content, *, settle_fee=None):
    return pay_and_unlock(
        content_hash=_CH, recipient=_BUYER_ETH, recipient_privkey_b64=buyer_priv,
        key_client=client, retrieve_content=lambda ch: content, settle_fee=settle_fee)


# ── happy path ────────────────────────────────────────────────────────────────

def test_full_flow_pays_then_unlocks():
    content, wrapped, buyer_priv, plaintext = _fixture()
    order = []
    client = _FakeClient(emit=_Ev(_CH, _BUYER_ETH, wrapped), order=order)  # not-yet-released
    out = _call(client, buyer_priv, content, settle_fee=lambda: order.append("pay"))
    assert out == plaintext
    assert order == ["pay", "release"]                 # fee settled BEFORE release


def test_already_paid_no_settle_no_rerelease():
    content, wrapped, buyer_priv, plaintext = _fixture()
    order = []
    client = _FakeClient(released=[_Ev(_CH, _BUYER_ETH, wrapped)], order=order)  # already released
    out = _call(client, buyer_priv, content, settle_fee=None)                    # already paid
    assert out == plaintext
    assert order == []                                 # no settle, no re-release (idempotent)


# ── fail-loud at each stage ───────────────────────────────────────────────────

def test_unpaid_release_surfaces_pay_first():
    content, _wrapped, buyer_priv, _ = _fixture()
    client = _FakeClient(release_error=PaymentNotVerifiedError("PaymentNotVerified"))
    with pytest.raises(KeyNotReleasedError, match="PAY the release fee"):
        _call(client, buyer_priv, content, settle_fee=None)


def test_ciphertext_not_retrievable_raises():
    _content, wrapped, buyer_priv, _ = _fixture()
    client = _FakeClient(released=[_Ev(_CH, _BUYER_ETH, wrapped)])
    with pytest.raises(PaidUnlockError, match="not retrievable"):
        pay_and_unlock(content_hash=_CH, recipient=_BUYER_ETH, recipient_privkey_b64=buyer_priv,
                       key_client=client, retrieve_content=lambda ch: None)


def test_wrong_buyer_key_raises():
    content, wrapped, _buyer_priv, _ = _fixture()
    other_priv, _ = generate_recipient_keypair()
    client = _FakeClient(released=[_Ev(_CH, _BUYER_ETH, wrapped)])
    with pytest.raises(PaidUnlockError, match="unwrap"):
        _call(client, other_priv, content, settle_fee=None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
