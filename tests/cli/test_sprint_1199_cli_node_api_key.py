"""Sprint 1199 — the CLI authenticates daemon calls with the node API key.

NodeAuthMiddleware protects /compute/, /ftns/, /wallet/, ... when the operator
sets PRSM_NODE_API_KEY (the common day-one public-bind posture, made easy by the
sp1195 auto-provision). Before this fix the CLI sent only a user-login JWT, so
`compute infer` / `compute pay-infer` / `wallet faucet` all 401'd against a keyed
node. Now they send the node key (X-API-Key for raw-httpx calls; Bearer via
PRSMClient for pay-infer) when PRSM_NODE_API_KEY is set, and nothing when unset.
"""
from __future__ import annotations

import httpx
import pytest
from click.testing import CliRunner

from prsm.cli import main, _node_api_key_headers


@pytest.fixture
def runner():
    return CliRunner()


# ── the helper ───────────────────────────────────────────────────────────────────────

def test_helper_emits_x_api_key_when_set(monkeypatch):
    monkeypatch.setenv("PRSM_NODE_API_KEY", "abc123")
    assert _node_api_key_headers() == {"X-API-Key": "abc123"}


def test_helper_empty_when_unset(monkeypatch):
    monkeypatch.delenv("PRSM_NODE_API_KEY", raising=False)
    assert _node_api_key_headers() == {}


# ── compute infer (unary) sends the key ──────────────────────────────────────────────

def test_compute_infer_sends_node_key(runner, monkeypatch):
    monkeypatch.setenv("PRSM_NODE_API_KEY", "nodekey")
    captured = {}

    def _fake_post(endpoint, json=None, timeout=None, headers=None):
        captured["headers"] = headers or {}

        class _R:
            status_code = 200

            def json(self_inner):
                return {"output": "x", "ftns_charged": 0}
        return _R()
    monkeypatch.setattr(httpx, "post", _fake_post)
    r = runner.invoke(main, ["compute", "infer", "--prompt", "hi"])
    assert r.exit_code == 0, r.output
    assert captured["headers"].get("X-API-Key") == "nodekey"


def test_compute_infer_no_key_no_header(runner, monkeypatch):
    monkeypatch.delenv("PRSM_NODE_API_KEY", raising=False)
    captured = {}

    def _fake_post(endpoint, json=None, timeout=None, headers=None):
        captured["headers"] = headers or {}

        class _R:
            status_code = 200

            def json(self_inner):
                return {"output": "x"}
        return _R()
    monkeypatch.setattr(httpx, "post", _fake_post)
    r = runner.invoke(main, ["compute", "infer", "--prompt", "hi"])
    assert r.exit_code == 0, r.output
    assert "X-API-Key" not in captured["headers"]


# ── wallet faucet sends the key (/ftns/ is protected) ────────────────────────────────

def test_wallet_faucet_sends_node_key(runner, monkeypatch):
    monkeypatch.setenv("PRSM_NODE_API_KEY", "nodekey")
    captured = {}

    def _fake_post(url, json=None, timeout=None, headers=None):
        captured["headers"] = headers or {}

        class _R:
            status_code = 200

            def json(self_inner):
                return {"dispensed_ftns": 100, "tx_hash": "0x1",
                        "recipient": "0x" + "44" * 20, "faucet_address": "0xF",
                        "network": "base-sepolia"}
        return _R()
    monkeypatch.setattr(httpx, "post", _fake_post)
    r = runner.invoke(
        main, ["wallet", "faucet", "--address", "0x" + "44" * 20])
    assert r.exit_code == 0, r.output
    assert captured["headers"].get("X-API-Key") == "nodekey"


# ── pay-infer threads the key into PRSMClient (Bearer) ───────────────────────────────

def test_pay_infer_passes_node_key_to_client(runner, monkeypatch):
    monkeypatch.setenv("PRSM_NODE_API_KEY", "nodekey")
    monkeypatch.setenv("PRIVATE_KEY", "0x" + "11" * 32)
    captured = {}

    class _FakeClient:
        def __init__(self, *a, **kw):
            captured["api_key"] = kw.get("api_key")

        async def pay_and_infer(self, prompt, **kwargs):
            return {"success": True, "output": "x"}

        async def close(self):
            pass
    monkeypatch.setattr("prsm.sdk.client.PRSMClient", _FakeClient)
    r = runner.invoke(main, ["compute", "pay-infer", "--prompt", "hi"])
    assert r.exit_code == 0, r.output
    assert captured["api_key"] == "nodekey"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
