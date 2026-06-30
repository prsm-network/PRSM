"""Sprint 1310 — `prsm compute pay-infer --stream`.

The CLI surface for paid STREAMING inference (pay_and_infer_stream). Mirrors the
unary pay-infer CLI (sp1192) but consumes the SSE event stream: tokens print live,
then a charge/verify footer. Mocks PRSMClient with an async-generator stream method.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from prsm.cli import main

_TEST_KEY = "0x" + "11" * 32
_PROVIDER = "0x" + "22" * 20


@pytest.fixture
def runner():
    return CliRunner()


def _fake_client_class(events):
    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def pay_and_infer_stream(self, prompt, **kwargs):
            for ev in events:
                yield ev

        async def close(self):
            pass

    return _FakeClient


def _patch(monkeypatch, events):
    monkeypatch.setenv("PRIVATE_KEY", _TEST_KEY)
    monkeypatch.setattr("prsm.sdk.client.PRSMClient", _fake_client_class(events))


def _invoke(runner, *extra):
    return runner.invoke(main, [
        "compute", "pay-infer", "--prompt", "Hi", "--stream",
        "--provider-address", _PROVIDER, "--network", "testnet", *extra,
    ])


def test_stream_prints_tokens_then_footer(runner, monkeypatch):
    _patch(monkeypatch, [
        {"type": "token", "text_delta": "Hel"},
        {"type": "token", "text_delta": "lo"},
        {"type": "result", "success": True, "output": "Hello",
         "ftns_charged": "0.2", "receipt": {}},
    ])
    r = _invoke(runner)
    assert r.exit_code == 0, r.output
    assert "Hello" in r.output            # streamed tokens
    assert "0.2 FTNS" in r.output         # charge footer
    assert "settled from your escrow" in r.output


def test_stream_receipt_verified_footer(runner, monkeypatch):
    _patch(monkeypatch, [
        {"type": "token", "text_delta": "ok"},
        {"type": "result", "success": True, "output": "ok",
         "ftns_charged": "0.1", "receipt": {}, "receipt_verified": True},
    ])
    r = _invoke(runner, "--verify-pubkey-b64", "QUJD")
    assert r.exit_code == 0, r.output
    assert "receipt_verified" in r.output and "yes" in r.output


def test_stream_error_event_exits_1(runner, monkeypatch):
    _patch(monkeypatch, [
        {"type": "error", "status": 402, "detail": "payment authorization rejected"},
    ])
    r = _invoke(runner)
    assert r.exit_code == 1
    assert "rejected" in r.output.lower()


def test_stream_no_terminal_event_exits_1(runner, monkeypatch):
    # stream ends with only tokens, no result/error → treated as an error
    _patch(monkeypatch, [{"type": "token", "text_delta": "partial"}])
    r = _invoke(runner)
    assert r.exit_code == 1
    assert "no terminal event" in r.output.lower()


def test_stream_json_format_emits_result_event(runner, monkeypatch):
    import json
    _patch(monkeypatch, [
        {"type": "token", "text_delta": "x"},
        {"type": "result", "success": True, "output": "x", "ftns_charged": "0.1"},
    ])
    r = _invoke(runner, "--format", "json")
    assert r.exit_code == 0, r.output
    # the terminal result event is emitted as JSON (tokens not printed in json mode)
    parsed = json.loads(r.output.strip().splitlines()[-1])
    assert parsed["type"] == "result" and parsed["output"] == "x"


def test_no_key_exits_1(runner, monkeypatch):
    monkeypatch.delenv("PRIVATE_KEY", raising=False)
    monkeypatch.delenv("FTNS_WALLET_PRIVATE_KEY", raising=False)
    monkeypatch.setattr("prsm.sdk.client.PRSMClient", _fake_client_class([]))
    r = _invoke(runner)
    assert r.exit_code == 1
    assert "signing key" in r.output.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
