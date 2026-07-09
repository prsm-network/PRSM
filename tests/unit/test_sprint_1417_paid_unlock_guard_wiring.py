"""Sprint 1417 — prove pay_and_unlock_content actually INVOKES its consumer money guards.

The paid-content consumer flow (``PRSMClient.pay_and_unlock_content``) protects the buyer with two
guards before it pays a non-refundable on-chain fee:

  * assert_fee_matches_deposit — refuse to pay a fee that doesn't match the on-chain deposit (a
    wrong/unconfirmable fee is pulled with NO unlock and NO refund — pure buyer fund loss).
  * assert_publisher_controls_payee — refuse to pay when the fee payee (registry creator) is not the
    key depositor (publisher): the squatting signature, where the fee reaches someone who can't
    unlock the content.

Both PURE FUNCTIONS have killing tests (sp1361 / sp1365). What was untested is the WIRING: deleting
BOTH call sites in prsm/sdk/client.py leaves all 114 paid-content tests green. That is the sp1412
shape at the invocation level — the guard exists, is tested in isolation, and a refactor could drop
its call while CI stays green and the protection silently vanishes from the live path.

These tests drive the real ``pay_and_unlock_content`` with injected clients arranged so each guard
WOULD raise if reached, and assert it does. Delete the call and the flow proceeds past the guard —
the raise never happens — and the test fails. (sp1417 also added the ``_creator_reader`` injection
seam so the squat guard's invocation is reachable under test; without it the reader construction
failed on the unresolved rpc and the guard silently skipped.)
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from prsm.economy.paid_content import FeeMismatchError, SquatMismatchError
from prsm.economy.web3.key_distribution import KeyDeposit
from prsm.enterprise.recipient_encryption import (
    EnterpriseRecipient,
    generate_recipient_keypair,
)
from prsm.sdk.client import PRSMClient
from prsm.storage.encryption import encrypt, generate_key
from prsm.storage.paid_unlock import key_commitment, wrap_content_key_for_deposit

_CH = bytes.fromhex("cd" * 32)
_FEE = 10 ** 18
_PUBLISHER = "0xPUB0000000000000000000000000000000000001"


def _fixture():
    content_key = generate_key()
    content = encrypt(b"paywalled dataset", content_key)
    buyer_priv, buyer_pub = generate_recipient_keypair()
    wrapped = wrap_content_key_for_deposit(
        content_key, [EnterpriseRecipient(identifier="b", x25519_pubkey_b64=buyer_pub)])
    commitment = key_commitment(wrapped)

    paid = {"ok": False}
    vc = MagicMock()
    vc.address = "0x" + "33" * 20
    vc.verify_payment.return_value = False
    vc.pay_for_access.side_effect = lambda *a: paid.__setitem__("ok", True)

    def fetch(_ch):
        return wrapped if paid["ok"] else None

    return content, buyer_priv, commitment, vc, fetch


def _run(*, deposit, creator_reader, fee_wei=_FEE):
    content, buyer_priv, commitment, vc, fetch = _fixture()
    kc = MagicMock()
    kc.get_deposit.return_value = deposit
    client = PRSMClient()
    try:
        return asyncio.run(client.pay_and_unlock_content(
            _CH, requester_key="0x" + "01" * 32, x25519_privkey_b64=buyer_priv,
            fee_wei=fee_wei, verifier_address="0x" + "ab" * 20, commitment=commitment,
            _verifier_client=vc, _content=content, _fetch_wrapped_key=fetch,
            _key_client=kc, _creator_reader=creator_reader))
    finally:
        asyncio.run(client.close())


_MATCHING_DEPOSIT = KeyDeposit(
    publisher=_PUBLISHER, royalty="0xroy", release_fee_ftns_wei=_FEE, active=True)


# ── the fee guard must be invoked ─────────────────────────────────────────


def test_pay_and_unlock_invokes_the_fee_guard():
    """THE killing test for the fee-guard CALL: the on-chain deposit fee (5 FTNS) differs from the
    fee about to be paid (_FEE). Only assert_fee_matches_deposit stands between this and a
    non-refundable payment. Remove the call and the flow proceeds — no FeeMismatchError."""
    mismatched = KeyDeposit(
        publisher=_PUBLISHER, royalty="0xroy", release_fee_ftns_wei=5 * 10 ** 18, active=True)
    # a creator_reader that AGREES (creator == publisher), so the squat guard can't be what raises
    with pytest.raises(FeeMismatchError):
        _run(deposit=mismatched, creator_reader=lambda _c: _PUBLISHER)


def test_pay_and_unlock_fee_guard_fails_closed_on_unconfirmable_deposit():
    """The fee guard fails CLOSED: an unreadable deposit (RPC error) must abort, not pay blind."""
    kc = MagicMock()
    kc.get_deposit.side_effect = RuntimeError("rpc down")
    content, buyer_priv, commitment, vc, fetch = _fixture()
    client = PRSMClient()
    try:
        with pytest.raises(FeeMismatchError):
            asyncio.run(client.pay_and_unlock_content(
                _CH, requester_key="0x" + "01" * 32, x25519_privkey_b64=buyer_priv,
                fee_wei=_FEE, verifier_address="0x" + "ab" * 20, commitment=commitment,
                _verifier_client=vc, _content=content, _fetch_wrapped_key=fetch,
                _key_client=kc, _creator_reader=lambda _c: _PUBLISHER))
    finally:
        asyncio.run(client.close())


# ── the squat guard must be invoked ───────────────────────────────────────


def test_pay_and_unlock_invokes_the_squat_guard():
    """THE killing test for the squat-guard CALL: the fee matches, but the registry creator (fee
    payee) is a DIFFERENT identity from the key depositor (publisher). Only
    assert_publisher_controls_payee stands between this and paying a squatter."""
    squatter = "0xSQUATTER000000000000000000000000000000A"
    with pytest.raises(SquatMismatchError):
        _run(deposit=_MATCHING_DEPOSIT, creator_reader=lambda _c: squatter)


def test_pay_and_unlock_squat_guard_rejects_zero_creator():
    """A content with no registered creator has an unknown fee payee — refuse to pay."""
    with pytest.raises(SquatMismatchError):
        _run(deposit=_MATCHING_DEPOSIT, creator_reader=lambda _c: "0x" + "0" * 40)


# ── the happy path still unlocks (guards not over-broad) ──────────────────


def test_matching_fee_and_creator_unlocks():
    """Regression: fee matches AND creator == publisher → both guards pass, the content decrypts."""
    out = _run(deposit=_MATCHING_DEPOSIT, creator_reader=lambda _c: _PUBLISHER)
    assert out == b"paywalled dataset"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
