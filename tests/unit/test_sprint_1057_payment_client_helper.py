"""Sprint 1057 — requester-side payment helper (requester-payment, brick 4).

The client side a paying requester uses: build + sign a PaymentAuthorization bound
to a specific inference request, ready to drop into the /compute/inference body as
``{"payment_authorization": {...}}``. The decisive test is the END-TO-END interop:
what this helper produces must verify against the provider-side verifier (bricks
1-3) AND its request_hash must equal the hash the provider computes from the same
request body. Funding (EscrowPool.deposit) reuses the sprint-1041 client and is not
re-tested here.
"""
from __future__ import annotations

import asyncio

import pytest
from eth_account import Account

from prsm.settlement.payment_authorization import (
    canonical_request_hash, inference_request_fields,
)
from prsm.settlement.payment_authorization_verifier import (
    AuthorizationRejected, InMemoryNonceStore, PaymentAuthorizationVerifier,
)
from prsm.settlement.payment_client import build_payment_authorization

REQUESTER_KEY = "0x" + "11" * 32
REQUESTER = Account.from_key(REQUESTER_KEY).address
PROVIDER = Account.from_key("0x" + "22" * 32).address
ONE_FTNS = 10**18


def _run(coro):
    return asyncio.run(coro)


_REQ = dict(model_id="gpt2", prompt="hello world", max_tokens=16,
            privacy_tier="standard", content_tier="A")


def _build(**over):
    kw = dict(
        requester_key=REQUESTER_KEY, provider_address=PROVIDER,
        max_spend_ftns=1.0, expiry_unix=2_000_000_000, **_REQ)
    kw.update(over)
    return build_payment_authorization(**kw)


def test_produces_embeddable_object():
    auth = _build()
    assert set(auth) >= {"payload", "signature"}
    assert auth["signature"].startswith("0x")
    p = auth["payload"]
    assert p["requester"].lower() == REQUESTER.lower()
    assert p["provider"].lower() == PROVIDER.lower()
    assert p["max_spend_wei"] == ONE_FTNS


def test_request_hash_matches_provider_computation():
    """The helper binds the SAME request_hash the provider derives from the body."""
    auth = _build()
    provider_rh = canonical_request_hash(inference_request_fields(**_REQ))
    assert auth["payload"]["request_hash"] == provider_rh


def test_end_to_end_verifies_against_provider_verifier():
    auth = _build()
    rh = canonical_request_hash(inference_request_fields(**_REQ))
    v = PaymentAuthorizationVerifier(
        provider_address=PROVIDER, nonce_store=InMemoryNonceStore(),
        now=lambda: 1_000_000_000.0)
    out = _run(v.verify(
        auth["payload"], auth["signature"], request_hash=rh, quoted_price_wei=ONE_FTNS))
    assert out.lower() == REQUESTER.lower()


def test_nonce_is_unique_per_build():
    a, b = _build(), _build()
    assert a["payload"]["job_nonce"] != b["payload"]["job_nonce"]


def test_explicit_nonce_is_honored():
    n = "0x" + "ab" * 32
    auth = _build(job_nonce=n)
    assert auth["payload"]["job_nonce"] == n


def test_tampered_body_fails_request_hash_binding():
    """An auth built for one request must not verify against a different request."""
    auth = _build()
    different_rh = canonical_request_hash(inference_request_fields(
        model_id="gpt2", prompt="DIFFERENT", max_tokens=16,
        privacy_tier="standard", content_tier="A"))
    v = PaymentAuthorizationVerifier(
        provider_address=PROVIDER, nonce_store=InMemoryNonceStore(),
        now=lambda: 1_000_000_000.0)
    with pytest.raises(AuthorizationRejected) as ei:
        _run(v.verify(auth["payload"], auth["signature"],
                      request_hash=different_rh, quoted_price_wei=ONE_FTNS))
    assert ei.value.reason == "request-hash-mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
