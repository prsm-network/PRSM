"""Sprint 1332 (S5) — PRSMClient.pay_and_infer_multistage_stream: paid multi-stage STREAMING.

The streaming twin of pay_and_infer_multistage: quote → sign one per-stage auth over the quoted
(price-based) payees → stream /compute/inference/stream. Tested with a mocked transport: the
streamed body carries a per-stage auth matching the quoted payees, tokens pass through, the quote
is attached to the terminal result, and not-multi-stage / not-settleable raise clearly.
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


def _settleable_quote():
    return {
        "multi_stage": True, "settleable": True, "stage_count": 2, "price_ftns": "0.28",
        "payee_set_hash": "0x" + compute_payee_set_hash([(_A, _SHARE), (_B, _SHARE)]).hex(),
        "payees": [[_A, _SHARE], [_B, _SHARE]],
    }


def _client(quote, events):
    c = PRSMClient(base_url="http://x")
    c._captured_body = None

    async def _fake_post(path, data):
        return quote

    async def _fake_stream(body):
        c._captured_body = body
        for ev in events:
            yield ev

    c._post = _fake_post
    c._stream_events = _fake_stream
    return c


async def _collect(agen):
    return [ev async for ev in agen]


def test_streams_tokens_and_attaches_quote_to_result():
    events = [
        {"type": "token", "text_delta": "He"},
        {"type": "token", "text_delta": "llo"},
        {"type": "result", "success": True, "output": "Hello", "receipt": {}},
    ]
    c = _client(_settleable_quote(), events)
    got = asyncio.run(_collect(c.pay_and_infer_multistage_stream(
        "hi", requester_key=_KEY, model_id="qwen", max_tokens=8, chain_id=8453)))

    assert [e["text_delta"] for e in got if e["type"] == "token"] == ["He", "llo"]
    result = [e for e in got if e["type"] == "result"][0]
    assert result["multistage_quote"]["price_ftns"] == "0.28"
    assert result["multistage_quote"]["stage_count"] == 2
    # the streamed body carried the per-stage auth signed over the EXACT quoted payees
    auth = c._captured_body["per_stage_payment_authorization"]
    assert auth["payload"]["payee_set_hash"] == \
        "0x" + compute_payee_set_hash([(_A, _SHARE), (_B, _SHARE)]).hex()
    assert auth["payload"]["requester"].lower() == _REQ.lower()


def test_receipt_verify_on_result():
    events = [{"type": "result", "success": True, "output": "x", "receipt": {"bad": 1}}]
    c = _client(_settleable_quote(), events)
    got = asyncio.run(_collect(c.pay_and_infer_multistage_stream(
        "hi", requester_key=_KEY, model_id="qwen", verify_pubkey_b64="not-a-key")))
    result = [e for e in got if e["type"] == "result"][0]
    assert result["receipt_verified"] is False


def test_not_multistage_raises():
    c = _client({"multi_stage": False, "reason": "single node"}, [])
    with pytest.raises(ValueError, match="does not route multi-stage"):
        asyncio.run(_collect(c.pay_and_infer_multistage_stream(
            "hi", requester_key=_KEY, model_id="gpt2")))


def test_not_settleable_raises():
    c = _client({"multi_stage": True, "settleable": False, "reason": "unmapped payee"}, [])
    with pytest.raises(ValueError, match="not settleable"):
        asyncio.run(_collect(c.pay_and_infer_multistage_stream(
            "hi", requester_key=_KEY, model_id="qwen")))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
