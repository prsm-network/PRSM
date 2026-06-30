"""Sprint 1310 — PRSMClient.pay_and_infer_stream(): paid STREAMING inference.

The streaming twin of pay_and_infer (sp1189). The server streaming path already verifies
+ records + settles a PaymentAuthorization (sp1056, api.py:9145); the only gap was the SDK
client, which now signs an auth + attaches it to the /compute/inference/stream body, and
optionally verifies the receipt on the terminal result event.

Reuses the sp820 SSE shim shape; provider_address is passed explicitly to skip /info.
"""
from __future__ import annotations

import json

import pytest
from eth_account import Account

pytestmark = pytest.mark.asyncio


def _sse(event_type, data):
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


class _FakeContent:
    def __init__(self, body): self._b = body
    def __aiter__(self): return self._aiter()
    async def _aiter(self):
        for line in self._b.split(b"\n"):
            yield line + b"\n"


class _FakeResp:
    def __init__(self, status, body_text):
        self.status = status
        self._b = body_text.encode("utf-8")
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    @property
    def content(self): return _FakeContent(self._b)
    async def read(self): return self._b


class _FakeSession:
    def __init__(self, resp): self._r = resp; self.post_calls = []
    def post(self, url, json=None, headers=None, timeout=None):
        self.post_calls.append({"url": url, "json": json})
        return self._r
    async def close(self): pass


def _client(resp):
    from prsm.sdk.client import PRSMClient
    c = PRSMClient("http://node:8000")
    c._session = _FakeSession(resp)
    return c


_KEY = Account.create().key.hex()
PROVIDER = "0x" + "11" * 20


async def test_method_exists():
    from prsm.sdk.client import PRSMClient
    assert callable(PRSMClient().pay_and_infer_stream)


async def test_posts_payment_auth_to_stream_endpoint():
    c = _client(_FakeResp(200, _sse("result", {
        "success": True, "output": "ok", "ftns_charged": "0.1", "receipt": {}})))
    async for _e in c.pay_and_infer_stream(
            prompt="Hi", requester_key=_KEY, provider_address=PROVIDER):
        pass
    call = c._session.post_calls[0]
    assert "/compute/inference/stream" in call["url"]
    body = call["json"]
    # the signed authorization is attached, shaped {payload, signature}
    auth = body["payment_authorization"]
    assert isinstance(auth, dict) and "payload" in auth and "signature" in auth
    assert auth["signature"].startswith("0x")
    # the request body still carries the canonical inference fields
    for k in ("prompt", "model_id", "budget_ftns", "max_tokens",
              "privacy_tier", "content_tier"):
        assert k in body


async def test_auth_payload_binds_provider_and_request():
    c = _client(_FakeResp(200, _sse("result", {
        "success": True, "output": "", "ftns_charged": "0", "receipt": {}})))
    async for _e in c.pay_and_infer_stream(
            prompt="Hello", requester_key=_KEY, provider_address=PROVIDER,
            model_id="gpt2", max_tokens=8, max_spend_ftns=2.0):
        pass
    payload = c._session.post_calls[0]["json"]["payment_authorization"]["payload"]
    assert payload["provider"].lower() == PROVIDER.lower()
    assert payload["requester"].lower() == Account.from_key(_KEY).address.lower()
    assert int(payload["max_spend_wei"]) == 2 * 10 ** 18


async def test_yields_token_then_result():
    body = (_sse("token", {"sequence_index": 0, "text_delta": "Hel"})
            + _sse("token", {"sequence_index": 1, "text_delta": "lo"})
            + _sse("result", {"success": True, "output": "Hello",
                              "ftns_charged": "0.2", "receipt": {}}))
    c = _client(_FakeResp(200, body))
    evs = [ev async for ev in c.pay_and_infer_stream(
        prompt="Hi", requester_key=_KEY, provider_address=PROVIDER)]
    assert [e["type"] for e in evs] == ["token", "token", "result"]
    assert evs[-1]["output"] == "Hello"


async def test_verify_pubkey_adds_receipt_verified_on_result():
    c = _client(_FakeResp(200, _sse("result", {
        "success": True, "output": "ok", "ftns_charged": "0.1", "receipt": {}})))
    evs = [ev async for ev in c.pay_and_infer_stream(
        prompt="Hi", requester_key=_KEY, provider_address=PROVIDER,
        verify_pubkey_b64="QUJD")]
    result = [e for e in evs if e["type"] == "result"][0]
    assert "receipt_verified" in result            # hook fired
    assert isinstance(result["receipt_verified"], bool)


async def test_no_verify_pubkey_leaves_result_unaugmented():
    c = _client(_FakeResp(200, _sse("result", {
        "success": True, "output": "ok", "ftns_charged": "0.1", "receipt": {}})))
    evs = [ev async for ev in c.pay_and_infer_stream(
        prompt="Hi", requester_key=_KEY, provider_address=PROVIDER)]
    result = [e for e in evs if e["type"] == "result"][0]
    assert "receipt_verified" not in result        # no verification requested


async def test_non_200_yields_error_event():
    c = _client(_FakeResp(402, '{"detail":"payment authorization rejected"}'))
    evs = [ev async for ev in c.pay_and_infer_stream(
        prompt="Hi", requester_key=_KEY, provider_address=PROVIDER)]
    assert len(evs) == 1
    assert evs[0]["type"] == "error" and evs[0]["status"] == 402
    assert "rejected" in evs[0]["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
