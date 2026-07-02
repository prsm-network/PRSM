"""Sprint 1352 (+1359 F1 redesign) — brick 4 PUBLISHER side: publish_paid_content.

Makes a dataset Tier B/C paid-access: encrypt (freely-served ciphertext) → wrap the key to the
buyer(s) → deposit ONLY the sha256 COMMITMENT on-chain (never the wrapped key — the F1 fix) → retain
the wrapped key for the payment-gated serve. The capstone is the full round trip: publish, then a
buyer pays + FETCHES the wrapped key (verified against the commitment) + unlocks to the plaintext.
"""
from __future__ import annotations

import hashlib

import pytest

from prsm.economy.paid_content import pay_and_unlock, publish_paid_content
from prsm.enterprise.recipient_encryption import (
    EnterpriseRecipient,
    generate_recipient_keypair,
)
from prsm.storage.encryption import encrypt, generate_key
from prsm.storage.paid_unlock import (
    PaidUnlockError,
    deserialize_encrypted_content,
    key_commitment,
    serialize_encrypted_content,
)

_VERIFIER = "0x" + "ab" * 20
_FEE = 10 ** 18


class _FakeKD:
    """Records deposits; the wrapped key is NOT deposited (only the commitment)."""

    def __init__(self, order=None):
        self.deposit_calls = []
        self._order = order

    def deposit_key(self, ch, deposited_bytes, royalty, fee):
        if self._order is not None:
            self._order.append("deposit")
        self.deposit_calls.append((bytes(ch), bytes(deposited_bytes), royalty, int(fee)))
        return ("0xdeposit", None)


def _publish(plaintext, buyer_pub, kd, *, served, retained, order=None):
    return publish_paid_content(
        plaintext=plaintext,
        recipients=[EnterpriseRecipient(identifier="buyer", x25519_pubkey_b64=buyer_pub)],
        royalty_verifier_address=_VERIFIER, release_fee_ftns_wei=_FEE, key_client=kd,
        publish_ciphertext=lambda ch, ct: ((order.append("publish") if order is not None else None),
                                           served.__setitem__(bytes(ch), ct)),
        retain_wrapped_key=lambda ch, wk, fee: retained.__setitem__(
            bytes(ch), {"wrapped_key": wk, "fee_wei": fee}))


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

def test_publish_deposits_commitment_not_the_key():
    order, served, retained = [], {}, {}
    kd = _FakeKD(order=order)
    _, pub = generate_recipient_keypair()
    res = _publish(b"proprietary NADA rows", pub, kd, served=served, retained=retained, order=order)

    assert order == ["deposit", "publish"]                         # deposit BEFORE serving
    ch, deposited, verifier, fee = kd.deposit_calls[0]
    # the DEPOSITED bytes are the 32-byte commitment, NOT the wrapped key
    assert deposited == res["commitment"] == key_commitment(res["wrapped_key"])
    assert len(deposited) == 32
    assert b"manifest" not in deposited                            # the wrapped key never on-chain
    assert verifier == _VERIFIER and fee == _FEE
    assert ch == res["content_hash"] == hashlib.sha256(res["ciphertext"]).digest()
    assert bytes(res["content_hash"]) in served                    # ciphertext served
    assert retained[bytes(ch)]["wrapped_key"] == res["wrapped_key"]  # wrapped key retained locally


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


# ── ★ the capstone: full publish → pay → FETCH+verify → unlock round trip ─────

def test_full_publish_to_unlock_round_trip():
    kd = _FakeKD()
    served, retained = {}, {}
    buyer_priv, buyer_pub = generate_recipient_keypair()
    plaintext = b"the proprietary dataset only paying buyers can read"

    res = _publish(plaintext, buyer_pub, kd, served=served, retained=retained)
    ch = res["content_hash"]

    # a party holding only the ciphertext learns nothing:
    assert plaintext not in served[bytes(ch)]

    # the buyer pays (no-op settle) and FETCHES the wrapped key from the retained store, which is
    # verified against the on-chain commitment inside pay_and_unlock:
    out = pay_and_unlock(
        content_hash=ch, recipient_privkey_b64=buyer_priv, commitment=res["commitment"],
        fetch_wrapped_key=lambda c: retained[bytes(c)]["wrapped_key"],
        retrieve_content=lambda c: deserialize_encrypted_content(served[bytes(c)]),
        settle_fee=lambda: None)
    assert out == plaintext


def test_non_buyer_cannot_unlock_published_content():
    kd = _FakeKD()
    served, retained = {}, {}
    _buyer_priv, buyer_pub = generate_recipient_keypair()
    other_priv, _ = generate_recipient_keypair()
    res = _publish(b"secret", buyer_pub, kd, served=served, retained=retained)
    with pytest.raises(PaidUnlockError, match="unwrap"):
        pay_and_unlock(
            content_hash=res["content_hash"], recipient_privkey_b64=other_priv,
            commitment=res["commitment"],
            fetch_wrapped_key=lambda c: retained[bytes(c)]["wrapped_key"],
            retrieve_content=lambda c: deserialize_encrypted_content(served[bytes(c)]),
            settle_fee=lambda: None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
