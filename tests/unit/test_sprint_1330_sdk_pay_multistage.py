"""Sprint 1330 (S5) — PRSMClient.pay_and_infer_multistage: client glue for the paid big-model path.

Runs quote → sign one per-stage auth over the quoted (price-based) payees → POST the paid
request. Tested with a mocked transport: the inference POST carries a per-stage auth whose
payee_set_hash matches the quoted payees, the quote is attached to the result, and the
not-multi-stage / not-settleable cases raise clearly.
"""
from __future__ import annotations

import asyncio

import pytest
from eth_account import Account

from prsm.sdk.client import PRSMClient
from prsm.settlement.per_stage_payment_authorization import compute_payee_set_hash

_KEY = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
_REQ = Account.from_key(_KEY).address
_A = "0x" + "a1" * 20
_B = "0x" + "b2" * 20
_SHARE = 140000000000000000  # 0.14 FTNS


def _client(quote, result=None):
    c = PRSMClient(base_url="http://x")
    c._posts = []

    async def _fake_post(path, data):
        c._posts.append((path, data))
        return quote if "quote-multistage" in path else (result or {})
    c._post = _fake_post
    return c


def _settleable_quote():
    return {
        "multi_stage": True, "settleable": True, "stage_count": 2,
        "price_ftns": "0.28",
        "payee_set_hash": "0x" + compute_payee_set_hash([(_A, _SHARE), (_B, _SHARE)]).hex(),
        "payees": [[_A, _SHARE], [_B, _SHARE]],
    }


def test_signs_quoted_payees_and_posts_paid_request():
    c = _client(_settleable_quote(), {"success": True, "output": "The", "receipt": {}})
    out = asyncio.run(c.pay_and_infer_multistage(
        "hi", requester_key=_KEY, model_id="qwen", max_tokens=8, chain_id=8453))

    # quote was requested first, then the paid inference
    paths = [p for p, _ in c._posts]
    assert paths == ["/compute/inference/quote-multistage", "/compute/inference"]

    infer_body = [d for p, d in c._posts if p == "/compute/inference"][0]
    auth = infer_body["per_stage_payment_authorization"]
    # signed over the EXACT quoted payees → hash matches (so the serve gate accepts)
    assert auth["payload"]["payee_set_hash"] == \
        "0x" + compute_payee_set_hash([(_A, _SHARE), (_B, _SHARE)]).hex()
    assert auth["payload"]["requester"].lower() == _REQ.lower()
    # the quote is surfaced on the result
    assert out["multistage_quote"]["price_ftns"] == "0.28"
    assert out["multistage_quote"]["stage_count"] == 2
    assert out["output"] == "The"


def test_not_multistage_raises():
    c = _client({"multi_stage": False, "reason": "routes to a single node"})
    with pytest.raises(ValueError, match="does not route multi-stage"):
        asyncio.run(c.pay_and_infer_multistage("hi", requester_key=_KEY, model_id="gpt2"))


def test_not_settleable_raises():
    c = _client({"multi_stage": True, "settleable": False,
                 "reason": "one or more stage nodes have no registered on-chain payee"})
    with pytest.raises(ValueError, match="not settleable"):
        asyncio.run(c.pay_and_infer_multistage("hi", requester_key=_KEY, model_id="qwen"))


def test_receipt_verify_flag_surfaces():
    # a bad receipt → receipt_verified False (never raises)
    c = _client(_settleable_quote(), {"success": True, "output": "x", "receipt": {"bad": 1}})
    out = asyncio.run(c.pay_and_infer_multistage(
        "hi", requester_key=_KEY, model_id="qwen", verify_pubkey_b64="not-a-real-key"))
    assert out["receipt_verified"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
