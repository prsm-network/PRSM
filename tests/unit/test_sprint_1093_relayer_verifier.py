"""Sprint 1093 — RelayerAuthorizationVerifier (requester-payment relayer, brick 3).

Combines the sp1091 delegated-auth chain verifier + the sp1092 cumulative budget store +
the sp1055 nonce store into the stateful, fail-closed verifier the ingress calls: a
funder signs ONE PaymentDelegation; a relayer signs per-request PaymentAuthorizations;
the provider verifies the chain + binds request_hash + price + anti-replay + escrow +
delegation budget, and returns the FUNDING requester (whose escrow pays).
"""
from __future__ import annotations

import pytest
from eth_account import Account

from prsm.settlement.payment_authorization import (
    canonical_request_hash, inference_request_fields, sign_payment_authorization,
)
from prsm.settlement.payment_delegation import sign_payment_delegation
from prsm.settlement.payment_authorization_verifier import (
    AuthorizationRejected, RelayerAuthorizationVerifier,
)

_CHAIN = 8453
_FUNDER = Account.create()
_RELAYER = Account.create()
_PROVIDER = Account.create()


def _rhash():
    return canonical_request_hash(inference_request_fields(
        model_id="gpt2", prompt="hi", max_tokens=8, privacy_tier="none", content_tier="A"))


def _delegation(*, max_total=10**18, expiry=9999999999, nonce="0x" + "cd" * 32):
    return {"requester": _FUNDER.address, "relayer": _RELAYER.address,
            "max_total_spend_wei": max_total, "delegation_nonce": nonce,
            "expiry_unix": expiry}


def _auth(*, max_spend=10**17, expiry=9999999999, job_nonce="0x" + "ab" * 32, rhash=None):
    return {"requester": _FUNDER.address, "provider": _PROVIDER.address,
            "max_spend_wei": max_spend, "job_nonce": job_nonce,
            "expiry_unix": expiry, "request_hash": rhash or _rhash()}


def _signed(auth, deleg):
    return (sign_payment_authorization(auth, _RELAYER.key, chain_id=_CHAIN),
            sign_payment_delegation(deleg, _FUNDER.key, chain_id=_CHAIN))


def _verifier(**kw):
    return RelayerAuthorizationVerifier(
        provider_address=_PROVIDER.address, chain_id=_CHAIN, now=lambda: 100.0, **kw)


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def test_valid_relayer_chain_returns_funder():
    a, d = _auth(), _delegation()
    asig, dsig = _signed(a, d)
    funder = _run(_verifier().verify(
        a, asig, d, dsig, request_hash=a["request_hash"], quoted_price_wei=10**16))
    assert funder.lower() == _FUNDER.address.lower()


def test_rejects_wrong_provider():
    a, d = _auth(), _delegation()
    asig, dsig = _signed(a, d)
    v = RelayerAuthorizationVerifier(
        provider_address=Account.create().address, chain_id=_CHAIN, now=lambda: 100.0)
    with pytest.raises(AuthorizationRejected) as ei:
        _run(v.verify(a, asig, d, dsig, request_hash=a["request_hash"], quoted_price_wei=1))
    assert ei.value.reason == "provider-mismatch"


def test_rejects_request_hash_mismatch():
    a, d = _auth(), _delegation()
    asig, dsig = _signed(a, d)
    with pytest.raises(AuthorizationRejected) as ei:
        _run(_verifier().verify(a, asig, d, dsig,
                                request_hash="0x" + "99" * 32, quoted_price_wei=1))
    assert ei.value.reason == "request-hash-mismatch"


def test_rejects_price_over_max_spend():
    a, d = _auth(max_spend=10**16), _delegation()
    asig, dsig = _signed(a, d)
    with pytest.raises(AuthorizationRejected) as ei:
        _run(_verifier().verify(a, asig, d, dsig,
                                request_hash=a["request_hash"], quoted_price_wei=10**17))
    assert ei.value.reason == "exceeds-max-spend"


def test_rejects_replayed_nonce():
    a, d = _auth(), _delegation()
    asig, dsig = _signed(a, d)
    v = _verifier()
    _run(v.verify(a, asig, d, dsig, request_hash=a["request_hash"], quoted_price_wei=10**16))
    with pytest.raises(AuthorizationRejected) as ei:
        _run(v.verify(a, asig, d, dsig, request_hash=a["request_hash"], quoted_price_wei=10**16))
    assert ei.value.reason == "nonce-replayed"


def test_rejects_when_delegation_budget_exhausted():
    """Cumulative spend across requests can't exceed the delegation cap."""
    v = _verifier()
    d = _delegation(max_total=10**16)   # 0.01 FTNS total
    # each request's per-request ceiling == the cap (passes the chain check); the
    # CUMULATIVE of two requests is what exceeds the delegation budget.
    a1 = _auth(max_spend=10**16, job_nonce="0x" + "01" * 32)
    asig1, dsig = _signed(a1, d)
    _run(v.verify(a1, asig1, d, dsig, request_hash=a1["request_hash"], quoted_price_wei=10**16))
    # second request (fresh nonce) is rejected — budget exhausted
    a2 = _auth(max_spend=10**16, job_nonce="0x" + "02" * 32)
    asig2, _ = _signed(a2, d)
    with pytest.raises(AuthorizationRejected) as ei:
        _run(v.verify(a2, asig2, d, dsig, request_hash=a2["request_hash"], quoted_price_wei=10**16))
    assert ei.value.reason == "delegation-budget-exceeded"


def test_rejects_auth_not_signed_by_relayer():
    a, d = _auth(), _delegation()
    stranger = Account.create()
    asig = sign_payment_authorization(a, stranger.key, chain_id=_CHAIN)
    dsig = sign_payment_delegation(d, _FUNDER.key, chain_id=_CHAIN)
    with pytest.raises(AuthorizationRejected) as ei:
        _run(_verifier().verify(a, asig, d, dsig,
                                request_hash=a["request_hash"], quoted_price_wei=1))
    assert ei.value.reason == "delegated-auth"


def test_insufficient_escrow_rejected_and_nonce_not_burned():
    a, d = _auth(), _delegation()
    asig, dsig = _signed(a, d)

    async def poor(_addr):
        return 0
    v = _verifier(balance_reader=poor)
    with pytest.raises(AuthorizationRejected) as ei:
        _run(v.verify(a, asig, d, dsig, request_hash=a["request_hash"], quoted_price_wei=10**16))
    assert ei.value.reason == "insufficient-escrow-balance"
    # nonce + budget not consumed → a later (funded) retry of the same request works
    async def rich(_addr):
        return 10**18
    v2 = RelayerAuthorizationVerifier(
        provider_address=_PROVIDER.address, chain_id=_CHAIN, now=lambda: 100.0,
        balance_reader=rich, nonce_store=v._nonce_store, budget_store=v._budget_store)
    funder = _run(v2.verify(a, asig, d, dsig,
                            request_hash=a["request_hash"], quoted_price_wei=10**16))
    assert funder.lower() == _FUNDER.address.lower()


def test_concurrent_same_nonce_only_one_succeeds():
    """sp1093 review #2 — the internal lock serializes verify() so two concurrent
    requests with the SAME job_nonce can't both pass the seen()->remember() window
    (the balance read is an await point between them)."""
    import asyncio
    a, d = _auth(), _delegation()
    asig, dsig = _signed(a, d)

    async def slow_balance(_addr):
        await asyncio.sleep(0.01)   # yield the loop between seen() and remember()
        return 10**18

    v = _verifier(balance_reader=slow_balance)

    async def both():
        async def one():
            try:
                return await v.verify(a, asig, d, dsig,
                                      request_hash=a["request_hash"], quoted_price_wei=10**16)
            except AuthorizationRejected as exc:
                return exc.reason
        return await asyncio.gather(one(), one())

    results = _run(both())
    successes = [r for r in results if isinstance(r, str) and r.lower() == _FUNDER.address.lower()]
    rejects = [r for r in results if r == "nonce-replayed"]
    assert len(successes) == 1 and len(rejects) == 1   # exactly one served, one replayed


def test_release_reservation_frees_budget_for_next_request():
    """sp1095 — capture: releasing a reservation's unused remainder lets the freed
    budget serve a later request that would otherwise exceed the delegation cap."""
    v = _verifier()
    d = _delegation(max_total=10**16)
    a1 = _auth(max_spend=10**16, job_nonce="0x" + "01" * 32)   # reserves the full cap
    asig1, dsig = _signed(a1, d)
    _run(v.verify(a1, asig1, d, dsig, request_hash=a1["request_hash"], quoted_price_wei=10**16))
    # budget now fully consumed → a second request (ceiling 5·10^15) is rejected
    a2 = _auth(max_spend=5 * 10**15, job_nonce="0x" + "02" * 32)
    asig2, _ = _signed(a2, d)
    with pytest.raises(AuthorizationRejected):
        _run(v.verify(a2, asig2, d, dsig, request_hash=a2["request_hash"], quoted_price_wei=5 * 10**15))
    # the first job actually settled cheap (10^15) → release the 0.9·10^16 remainder
    _run(v.release_reservation(d["delegation_nonce"], 10**16 - 10**15))
    # consumed is now 10^15; the second request (5·10^15) fits in the freed budget
    funder = _run(v.verify(a2, asig2, d, dsig,
                           request_hash=a2["request_hash"], quoted_price_wei=5 * 10**15))
    assert funder.lower() == _FUNDER.address.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
