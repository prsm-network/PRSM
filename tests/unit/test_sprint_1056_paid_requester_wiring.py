"""Sprint 1056 — paid-requester ingress wiring (requester-payment, brick 3).

The /compute/inference glue that turns a signed PaymentAuthorization in the request
body into an authenticated requester address threaded into the existing
``accumulate_settled_inference_receipt`` settle call. Two pure pieces tested here:

  * ``canonical_request_hash`` / ``inference_request_fields`` — the request-binding
    digest BOTH the requester (brick 4) and the provider compute, so they agree on
    what work the authorization covers.
  * ``resolve_paid_requester`` — the fail-closed ingress resolver: no auth in the
    body → None (unchanged self-pay path); auth present but no verifier wired →
    reject (a paid job must not silently downgrade); auth present + verifier →
    return the authenticated requester or propagate AuthorizationRejected.

The api.py edits (the per-job carrier + the two settle-site reads) are exercised
by the existing inference path; this brick keeps the decision logic hermetic.
"""
from __future__ import annotations

import asyncio

import pytest
from eth_account import Account

from prsm.settlement.payment_authorization import (
    canonical_request_hash,
    inference_request_fields,
    sign_payment_authorization,
)
from prsm.settlement.payment_authorization_verifier import (
    AuthorizationRejected,
    InMemoryNonceStore,
    PaymentAuthorizationVerifier,
)
from prsm.settlement.client_wiring import resolve_paid_requester

REQUESTER_KEY = "0x" + "11" * 32
PROVIDER_KEY = "0x" + "22" * 32
REQUESTER = Account.from_key(REQUESTER_KEY).address
PROVIDER = Account.from_key(PROVIDER_KEY).address
ONE_FTNS = 10**18


def _run(coro):
    return asyncio.run(coro)


def _fields():
    return inference_request_fields(
        model_id="gpt2", prompt="hello", max_tokens=16,
        privacy_tier="standard", content_tier="A",
    )


def _auth_and_sig(rh, **over):
    payload = {
        "requester": REQUESTER,
        "provider": PROVIDER,
        "max_spend_wei": ONE_FTNS,
        "job_nonce": "0x" + "ab" * 32,
        "expiry_unix": 2_000_000_000,
        "request_hash": rh,
    }
    payload.update(over)
    sig = sign_payment_authorization(payload, REQUESTER_KEY)
    return {"payload": payload, "signature": "0x" + sig.hex()}


def _verifier():
    return PaymentAuthorizationVerifier(
        provider_address=PROVIDER,
        nonce_store=InMemoryNonceStore(),
        now=lambda: 1_000_000_000.0,
    )


def test_canonical_request_hash_is_stable_and_order_independent():
    a = canonical_request_hash(inference_request_fields(
        model_id="gpt2", prompt="hi", max_tokens=8,
        privacy_tier="none", content_tier="A"))
    b = canonical_request_hash(inference_request_fields(
        content_tier="A", privacy_tier="none", max_tokens=8,
        prompt="hi", model_id="gpt2"))
    assert a == b
    assert a.startswith("0x") and len(a) == 66


def test_canonical_request_hash_changes_with_params():
    base = canonical_request_hash(_fields())
    diff = canonical_request_hash(inference_request_fields(
        model_id="gpt2", prompt="hello", max_tokens=17,  # changed
        privacy_tier="standard", content_tier="A"))
    assert base != diff


def test_no_auth_returns_none_self_pay_unchanged():
    out = _run(resolve_paid_requester(
        verifier=_verifier(), payment_authorization=None,
        request_hash=canonical_request_hash(_fields()), quoted_price_wei=ONE_FTNS))
    assert out is None


def test_auth_present_but_no_verifier_is_rejected():
    rh = canonical_request_hash(_fields())
    with pytest.raises(AuthorizationRejected) as ei:
        _run(resolve_paid_requester(
            verifier=None, payment_authorization=_auth_and_sig(rh),
            request_hash=rh, quoted_price_wei=ONE_FTNS))
    assert ei.value.reason == "verifier-not-wired"


def test_valid_auth_returns_authenticated_requester():
    rh = canonical_request_hash(_fields())
    out = _run(resolve_paid_requester(
        verifier=_verifier(), payment_authorization=_auth_and_sig(rh),
        request_hash=rh, quoted_price_wei=ONE_FTNS))
    assert out.lower() == REQUESTER.lower()


def test_request_hash_mismatch_propagates_rejection():
    """Provider computes the hash of the ACTUAL request; an auth signed over a
    different request is rejected (caller computed rh from the real body)."""
    signed_over = canonical_request_hash(_fields())
    actual_rh = canonical_request_hash(inference_request_fields(
        model_id="gpt2", prompt="DIFFERENT", max_tokens=16,
        privacy_tier="standard", content_tier="A"))
    with pytest.raises(AuthorizationRejected) as ei:
        _run(resolve_paid_requester(
            verifier=_verifier(), payment_authorization=_auth_and_sig(signed_over),
            request_hash=actual_rh, quoted_price_wei=ONE_FTNS))
    assert ei.value.reason == "request-hash-mismatch"


def test_malformed_auth_object_rejected():
    rh = canonical_request_hash(_fields())
    with pytest.raises(AuthorizationRejected) as ei:
        _run(resolve_paid_requester(
            verifier=_verifier(), payment_authorization={"payload": {}},  # no signature
            request_hash=rh, quoted_price_wei=ONE_FTNS))
    assert ei.value.reason == "malformed-authorization"


def test_accumulate_skips_when_value_exceeds_signed_max_spend():
    """sp1058 review #3 — defense-in-depth ceiling: a settled value above the
    requester's signed max_spend must NOT be booked on-chain against their escrow."""
    from unittest.mock import AsyncMock
    from prsm.settlement.client_wiring import accumulate_settled_inference_receipt

    class _Receipt:  # minimal non-multi-stage receipt (split helper returns None)
        topology_assignment = None
        privacy_proof = None

    client = AsyncMock()
    out = _run(accumulate_settled_inference_receipt(
        client=client, identity=object(), provider_address=PROVIDER,
        receipt=_Receipt(), release_ftns="2.0", job_id="job-x",
        requester_address=REQUESTER,
        max_spend_wei=ONE_FTNS,  # signed ceiling 1 FTNS; release 2 FTNS > ceiling
    ))
    assert out == "skipped:exceeds-max-spend"
    client.accumulate.assert_not_awaited()


def test_accumulate_within_max_spend_is_not_blocked_by_the_guard():
    """A value within the signed ceiling passes the guard (it then proceeds to the
    adapter, which may skip for other hermetic reasons — the guard itself doesn't
    fire)."""
    from unittest.mock import AsyncMock
    from prsm.settlement.client_wiring import accumulate_settled_inference_receipt

    class _Receipt:
        topology_assignment = None
        privacy_proof = None

    client = AsyncMock()
    out = _run(accumulate_settled_inference_receipt(
        client=client, identity=object(), provider_address=PROVIDER,
        receipt=_Receipt(), release_ftns="0.5", job_id="job-y",
        requester_address=REQUESTER, max_spend_wei=ONE_FTNS,
    ))
    assert out != "skipped:exceeds-max-spend"


def test_paid_requester_carrier_is_bounded():
    """sp1058 review #2 — the in-flight carrier evicts oldest entries past the cap
    so non-settling paid jobs can't leak memory."""
    import prsm.node.api as api

    class _Node:
        pass

    node = _Node()
    cap = api._PAID_REQUESTER_CARRIER_CAP
    for i in range(cap + 10):
        api._record_paid_requester(node, f"job-{i}", {"requester": REQUESTER, "max_spend_wei": ONE_FTNS})
    assert len(node._paid_requester_by_job) <= cap
    # the oldest entries were evicted; the most recent survive + pop correctly
    info = api._take_paid_requester(node, f"job-{cap + 9}")
    assert info is not None and info["requester"].lower() == REQUESTER.lower()
    assert api._take_paid_requester(node, "job-0") is None  # evicted
    assert api._take_paid_requester(node, "never-recorded") is None


def _run(coro):
    return asyncio.run(coro)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
