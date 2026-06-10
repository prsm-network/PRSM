"""Sprint 1055 — PaymentAuthorizationVerifier (requester-payment, brick 2).

Fail-closed verification of a signed PaymentAuthorization before a provider serves
+ settles a paid inference. Checks (in order): signature recovers to the named
requester; the auth pins THIS provider; the request-hash binds the auth to THIS
work; not expired; price within the authorized ceiling; nonce unseen (durable);
and (when a balance reader is wired) the requester's on-chain EscrowPool balance
covers the price. Any failure raises ``AuthorizationRejected`` with a reason — the
caller rejects the paid job. The nonce is remembered only on full success, so a
transient insufficient-balance leaves the auth retryable after a top-up.

See docs/2026-06-10-requester-payment-model-proposal.md.
"""
from __future__ import annotations

import asyncio

import pytest
from eth_account import Account

from prsm.settlement.payment_authorization import sign_payment_authorization
from prsm.settlement.payment_authorization_verifier import (
    AuthorizationRejected,
    InMemoryNonceStore,
    DurableNonceStore,
    PaymentAuthorizationVerifier,
)

REQUESTER_KEY = "0x" + "11" * 32
PROVIDER_KEY = "0x" + "22" * 32
REQUESTER = Account.from_key(REQUESTER_KEY).address
PROVIDER = Account.from_key(PROVIDER_KEY).address
OTHER = Account.from_key("0x" + "33" * 32).address
ONE_FTNS = 10**18
REQUEST_HASH = "0x" + "cd" * 32


def _payload(**over):
    p = {
        "requester": REQUESTER,
        "provider": PROVIDER,
        "max_spend_wei": ONE_FTNS,
        "job_nonce": "0x" + "ab" * 32,
        "expiry_unix": 2_000_000_000,
        "request_hash": REQUEST_HASH,
    }
    p.update(over)
    return p


def _signed(payload=None, key=REQUESTER_KEY):
    payload = payload or _payload()
    return payload, sign_payment_authorization(payload, key)


def _verifier(*, balance_reader=None, nonce_store=None, now=None):
    return PaymentAuthorizationVerifier(
        provider_address=PROVIDER,
        nonce_store=nonce_store or InMemoryNonceStore(),
        balance_reader=balance_reader,
        now=now or (lambda: 1_000_000_000.0),
    )


def _run(coro):
    return asyncio.run(coro)


def test_happy_path_returns_requester():
    payload, sig = _signed()
    v = _verifier()
    out = _run(v.verify(payload, sig, request_hash=REQUEST_HASH, quoted_price_wei=ONE_FTNS))
    assert out.lower() == REQUESTER.lower()


def test_signer_mismatch_rejected():
    """A provider can't name a requester who didn't sign (provider-key sig)."""
    payload, sig = _signed(key=PROVIDER_KEY)
    v = _verifier()
    with pytest.raises(AuthorizationRejected) as ei:
        _run(v.verify(payload, sig, request_hash=REQUEST_HASH, quoted_price_wei=ONE_FTNS))
    assert ei.value.reason == "signer-mismatch"


def test_provider_pin_rejected():
    """Auth signed for a DIFFERENT provider can't be replayed against this node."""
    payload, sig = _signed(_payload(provider=OTHER))
    v = _verifier()
    with pytest.raises(AuthorizationRejected) as ei:
        _run(v.verify(payload, sig, request_hash=REQUEST_HASH, quoted_price_wei=ONE_FTNS))
    assert ei.value.reason == "provider-mismatch"


def test_request_hash_binding_rejected():
    payload, sig = _signed()
    v = _verifier()
    with pytest.raises(AuthorizationRejected) as ei:
        _run(v.verify(payload, sig, request_hash="0x" + "00" * 32, quoted_price_wei=ONE_FTNS))
    assert ei.value.reason == "request-hash-mismatch"


def test_expired_rejected():
    payload, sig = _signed(_payload(expiry_unix=999))
    v = _verifier(now=lambda: 1000.0)
    with pytest.raises(AuthorizationRejected) as ei:
        _run(v.verify(payload, sig, request_hash=REQUEST_HASH, quoted_price_wei=ONE_FTNS))
    assert ei.value.reason == "expired"


def test_exceeds_max_spend_rejected():
    payload, sig = _signed()  # max_spend_wei == ONE_FTNS
    v = _verifier()
    with pytest.raises(AuthorizationRejected) as ei:
        _run(v.verify(payload, sig, request_hash=REQUEST_HASH, quoted_price_wei=ONE_FTNS + 1))
    assert ei.value.reason == "exceeds-max-spend"


def test_nonce_replay_rejected():
    payload, sig = _signed()
    v = _verifier()
    _run(v.verify(payload, sig, request_hash=REQUEST_HASH, quoted_price_wei=ONE_FTNS))
    with pytest.raises(AuthorizationRejected) as ei:
        _run(v.verify(payload, sig, request_hash=REQUEST_HASH, quoted_price_wei=ONE_FTNS))
    assert ei.value.reason == "nonce-replayed"


def test_bad_signature_rejected():
    payload = _payload()
    v = _verifier()
    with pytest.raises(AuthorizationRejected) as ei:
        _run(v.verify(payload, b"\x00" * 10, request_hash=REQUEST_HASH, quoted_price_wei=ONE_FTNS))
    assert ei.value.reason == "bad-signature"


def test_insufficient_escrow_balance_rejected():
    async def balance_reader(addr):
        return ONE_FTNS - 1
    payload, sig = _signed()
    v = _verifier(balance_reader=balance_reader)
    with pytest.raises(AuthorizationRejected) as ei:
        _run(v.verify(payload, sig, request_hash=REQUEST_HASH, quoted_price_wei=ONE_FTNS))
    assert ei.value.reason == "insufficient-escrow-balance"


def test_sufficient_escrow_balance_passes():
    async def balance_reader(addr):
        assert addr.lower() == REQUESTER.lower()
        return ONE_FTNS
    payload, sig = _signed()
    v = _verifier(balance_reader=balance_reader)
    out = _run(v.verify(payload, sig, request_hash=REQUEST_HASH, quoted_price_wei=ONE_FTNS))
    assert out.lower() == REQUESTER.lower()


def test_insufficient_balance_does_not_burn_nonce():
    """A rejection on balance must leave the nonce retryable after a top-up."""
    calls = {"n": 0}

    async def balance_reader(addr):
        calls["n"] += 1
        return ONE_FTNS if calls["n"] > 1 else 0  # poor first, funded second

    payload, sig = _signed()
    store = InMemoryNonceStore()
    v = _verifier(balance_reader=balance_reader, nonce_store=store)
    with pytest.raises(AuthorizationRejected):
        _run(v.verify(payload, sig, request_hash=REQUEST_HASH, quoted_price_wei=ONE_FTNS))
    # second attempt after "top-up" succeeds — nonce was not consumed
    out = _run(v.verify(payload, sig, request_hash=REQUEST_HASH, quoted_price_wei=ONE_FTNS))
    assert out.lower() == REQUESTER.lower()


def test_durable_nonce_store_persists_across_instances(tmp_path):
    """A nonce remembered by one DurableNonceStore is seen by a fresh one on the
    same file — a restart can't allow a replay."""
    path = tmp_path / "nonces.json"
    s1 = DurableNonceStore(path)
    n = "0x" + "ab" * 32
    assert s1.seen(n) is False
    s1.remember(n)
    s2 = DurableNonceStore(path)  # fresh instance, same file
    assert s2.seen(n) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
