"""Sprint 1352 — Tier B/C paid-decrypt arc, brick 4 (PUBLISHER side): publish_paid_content.

Makes a dataset Tier B/C paid-access: encrypt (freely-served ciphertext) → wrap the content key
to the buyer(s) → KeyDistribution.deposit_key (naming the royalty verifier + fee) → serve the
ciphertext. The capstone test is the FULL round trip publisher→consumer: publish, then a buyer
pays + unlocks back to the exact plaintext — tying bricks 1-4 end to end over real crypto and a
stateful mocked KeyDistribution client.
"""
from __future__ import annotations

import hashlib

import pytest

from prsm.economy.paid_content import pay_and_unlock, publish_paid_content
from prsm.economy.web3.key_distribution import KeyNotFoundError
from prsm.enterprise.recipient_encryption import (
    EnterpriseRecipient,
    generate_recipient_keypair,
)
from prsm.storage.encryption import encrypt, generate_key
from prsm.storage.paid_unlock import (
    PaidUnlockError,
    deserialize_encrypted_content,
    serialize_encrypted_content,
)

_VERIFIER = "0x" + "ab" * 20
_FEE = 10 ** 18


class _Ev:
    def __init__(self, content_hash, recipient, encrypted_key):
        self.content_hash, self.recipient, self.encrypted_key = (
            content_hash, recipient, encrypted_key)


class _FakeKD:
    """Stateful KeyDistribution: deposit stores the key by content_hash; release emits it."""

    def __init__(self, order=None):
        self._deposits = {}
        self._released = []
        self.deposit_calls = []
        self.release_calls = []
        self._order = order

    def deposit_key(self, ch, encrypted_key, royalty, fee):
        if self._order is not None:
            self._order.append("deposit")
        self.deposit_calls.append((bytes(ch), bytes(encrypted_key), royalty, int(fee)))
        self._deposits[bytes(ch)] = bytes(encrypted_key)
        return ("0xdeposit", None)

    def latest_block(self):
        return 100

    def get_key_released_events(self, fb, tb, *, argument_filters=None):
        return list(self._released)

    def release(self, ch, recipient):
        self.release_calls.append((bytes(ch), recipient))
        key = self._deposits.get(bytes(ch))
        if key is None:
            raise KeyNotFoundError("no deposit")
        self._released.append(_Ev(bytes(ch), recipient, key))
        return ("0xrelease", None)


# ── serialization envelope ────────────────────────────────────────────────────

def test_encrypted_content_serialize_round_trip():
    payload = encrypt(b"hello dataset", generate_key())
    back = deserialize_encrypted_content(serialize_encrypted_content(payload))
    assert back.ciphertext == payload.ciphertext
    assert back.iv == payload.iv and back.auth_tag == payload.auth_tag


def test_deserialize_garbage_raises():
    with pytest.raises(PaidUnlockError, match="malformed"):
        deserialize_encrypted_content(b"}{ not json")


# ── publish_paid_content ──────────────────────────────────────────────────────

def test_publish_deposits_then_serves():
    order = []
    kd = _FakeKD(order=order)
    _, pub = generate_recipient_keypair()
    served = {}
    res = publish_paid_content(
        plaintext=b"proprietary NADA rows",
        recipients=[EnterpriseRecipient(identifier="b", x25519_pubkey_b64=pub)],
        royalty_verifier_address=_VERIFIER, release_fee_ftns_wei=_FEE, key_client=kd,
        publish_ciphertext=lambda ch, ct: (order.append("publish"), served.__setitem__(bytes(ch), ct)))
    assert order == ["deposit", "publish"]                         # deposit BEFORE serving
    ch, wrapped, verifier, fee = kd.deposit_calls[0]
    assert ch == res["content_hash"] == hashlib.sha256(res["ciphertext"]).digest()
    assert verifier == _VERIFIER and fee == _FEE
    assert wrapped and b"manifest" in wrapped                      # the wrapped key, not raw
    assert bytes(res["content_hash"]) in served                    # ciphertext served under ch


def test_publish_requires_recipient_and_positive_fee():
    kd = _FakeKD()
    _, pub = generate_recipient_keypair()
    with pytest.raises(ValueError, match="recipient"):
        publish_paid_content(plaintext=b"x", recipients=[], royalty_verifier_address=_VERIFIER,
                             release_fee_ftns_wei=_FEE, key_client=kd,
                             publish_ciphertext=lambda ch, ct: None)
    with pytest.raises(ValueError, match="fee"):
        publish_paid_content(
            plaintext=b"x", recipients=[EnterpriseRecipient(identifier="b", x25519_pubkey_b64=pub)],
            royalty_verifier_address=_VERIFIER, release_fee_ftns_wei=0, key_client=kd,
            publish_ciphertext=lambda ch, ct: None)


# ── ★ the capstone: full publish → pay → unlock round trip (bricks 1-4) ────────

def test_full_publish_to_unlock_round_trip():
    kd = _FakeKD()
    served = {}
    buyer_priv, buyer_pub = generate_recipient_keypair()   # X25519 decrypt identity
    buyer_eth = "0x" + "33" * 20                           # on-chain payment/release identity
    plaintext = b"the proprietary dataset only paying buyers can read"

    res = publish_paid_content(
        plaintext=plaintext,
        recipients=[EnterpriseRecipient(identifier="buyer", x25519_pubkey_b64=buyer_pub)],
        royalty_verifier_address=_VERIFIER, release_fee_ftns_wei=_FEE, key_client=kd,
        publish_ciphertext=lambda ch, ct: served.__setitem__(bytes(ch), ct))
    ch = res["content_hash"]

    # A DIFFERENT party (no buyer key) holding only the ciphertext learns nothing useful:
    assert plaintext not in served[bytes(ch)]

    # The buyer pays (no-op settle here — B2's tests cover the gate) and unlocks:
    out = pay_and_unlock(
        content_hash=ch, recipient=buyer_eth, recipient_privkey_b64=buyer_priv, key_client=kd,
        retrieve_content=lambda c: deserialize_encrypted_content(served[bytes(c)]),
        settle_fee=lambda: None)
    assert out == plaintext
    assert kd.deposit_calls and kd.release_calls               # deposited then released


def test_non_buyer_cannot_unlock_published_content():
    kd = _FakeKD()
    served = {}
    _buyer_priv, buyer_pub = generate_recipient_keypair()
    other_priv, _ = generate_recipient_keypair()
    res = publish_paid_content(
        plaintext=b"secret", recipients=[EnterpriseRecipient(identifier="b", x25519_pubkey_b64=buyer_pub)],
        royalty_verifier_address=_VERIFIER, release_fee_ftns_wei=_FEE, key_client=kd,
        publish_ciphertext=lambda ch, ct: served.__setitem__(bytes(ch), ct))
    with pytest.raises(PaidUnlockError, match="unwrap"):
        pay_and_unlock(content_hash=res["content_hash"], recipient="0x" + "44" * 20,
                       recipient_privkey_b64=other_priv, key_client=kd,
                       retrieve_content=lambda c: deserialize_encrypted_content(served[bytes(c)]),
                       settle_fee=lambda: None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
