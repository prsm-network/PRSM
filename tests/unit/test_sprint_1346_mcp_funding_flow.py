"""Sprint 1346 — MCP front-door: make the paid-inference flow self-guiding + fix a 402 bug.

The compute-flagship analog of sp1345. prsm_inference is the self-serve inference tool, but when
it failed on insufficient FTNS the model got a dead-end error (and — a real bug — a 402
insufficient-funds actually surfaced "Inference failed: None", because a FastAPI HTTPException
lands as {"detail": ...} not {"error": ...} and the handler only read "error"). Now the real
reason is surfaced AND an actionable funding hint points the model at prsm_faucet →
prsm_local_balance → retry, so the faucet → inference flow is self-guiding through MCP.
"""
from __future__ import annotations

import asyncio

import prsm.mcp_server as mcp


def _run(coro):
    return asyncio.run(coro)


# ── _funding_hint detection ───────────────────────────────────────────────────

def test_funding_hint_detects_insufficient_variants():
    assert "prsm_faucet" in mcp._funding_hint("Insufficient FTNS balance to lock escrow")
    assert "prsm_faucet" in mcp._funding_hint("402 Payment Required")
    assert "prsm_faucet" in mcp._funding_hint("not enough funds for this request")
    assert "prsm_faucet" in mcp._funding_hint("escrow balance too low")


def test_funding_hint_silent_for_unrelated_errors():
    assert mcp._funding_hint("model 'xyz' not found") == ""
    assert mcp._funding_hint("privacy tier not permitted") == ""
    assert mcp._funding_hint(None) == ""
    assert mcp._funding_hint("") == ""


# ── the 402 bug fix + hint on the real error surface ──────────────────────────

def test_inference_402_detail_surfaced_not_none_with_hint(monkeypatch):
    async def _fake(method, path, data=None, **k):
        # FastAPI HTTPException(402) body shape — what the node actually returns
        return {"detail": "Insufficient FTNS balance to lock escrow for inference"}
    monkeypatch.setattr(mcp, "_call_node_api", _fake)
    out = _run(mcp.handle_prsm_inference({"prompt": "hi", "budget_ftns": 1.0}))
    assert "Insufficient FTNS balance" in out         # the REAL reason (bug: was "None")
    assert "failed: None" not in out                  # the pre-fix dead-end is gone
    assert "prsm_faucet" in out and "prsm_local_balance" in out  # self-guiding


def test_inference_error_key_also_gets_hint(monkeypatch):
    async def _fake(method, path, data=None, **k):
        return {"error": "insufficient escrow balance for the requested budget"}
    monkeypatch.setattr(mcp, "_call_node_api", _fake)
    out = _run(mcp.handle_prsm_inference({"prompt": "hi", "budget_ftns": 1.0}))
    assert "prsm_faucet" in out


def test_inference_nonfunding_error_has_no_funding_hint(monkeypatch):
    async def _fake(method, path, data=None, **k):
        return {"detail": "model 'ghost-70b' not found"}
    monkeypatch.setattr(mcp, "_call_node_api", _fake)
    out = _run(mcp.handle_prsm_inference({"prompt": "hi", "budget_ftns": 1.0}))
    assert "model 'ghost-70b' not found" in out       # surfaced (not None)
    assert "prsm_faucet" not in out                   # not misapplied to unrelated errors


def test_inference_success_is_clean(monkeypatch):
    async def _fake(method, path, data=None, **k):
        return {"success": True, "output": "Paris",
                "receipt": {"job_id": "j1", "cost_ftns": 0.1, "model_id": "m"}}
    monkeypatch.setattr(mcp, "_call_node_api", _fake)
    out = _run(mcp.handle_prsm_inference({"prompt": "capital of France?", "budget_ftns": 1.0}))
    assert "Paris" in out
    assert "prsm_faucet" not in out


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
