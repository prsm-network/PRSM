"""Sprint 1347 — sweep MCP action tools for the 402/detail swallow bug (sp1346 class).

_call_node_api returns resp.json() regardless of HTTP status, so a 4xx (FastAPI HTTPException →
{"detail": ...}) lands as a dict. Handlers that read success-path keys without guarding render
fake-success / blank output. This adds a shared _api_error() guard and fixes the two flagship
ACTION handlers found vulnerable: prsm_upload_dataset (rendered "Dataset Published" on a 4xx) and
prsm_analyze (rendered an empty "PRSM Analysis Result" on a 402).
"""
from __future__ import annotations

import asyncio

import prsm.mcp_server as mcp


def _run(coro):
    return asyncio.run(coro)


# ── the shared guard ──────────────────────────────────────────────────────────

def test_api_error_extracts_detail_error_and_success_false():
    assert mcp._api_error({"detail": "Payload Too Large"}) == "Payload Too Large"
    assert mcp._api_error({"error": "boom"}) == "boom"
    assert mcp._api_error({"success": False, "message": "nope"}) == "nope"
    assert mcp._api_error("not a dict").startswith("unexpected")


def test_api_error_none_for_success():
    assert mcp._api_error({"success": True, "output": "x"}) is None
    assert mcp._api_error({"shard_count": 4}) is None          # no error keys → success
    assert mcp._api_error({"response": "answer", "route": "swarm"}) is None


# ── prsm_upload_dataset no longer fake-succeeds on a 4xx ───────────────────────

def test_upload_dataset_surfaces_4xx_not_fake_success(monkeypatch):
    async def _fake(method, path, data=None, **k):
        return {"detail": "text size 20000000 bytes exceeds PRSM_MAX_UPLOAD_BYTES cap"}
    monkeypatch.setattr(mcp, "_call_node_api", _fake)
    out = _run(mcp.handle_prsm_upload_dataset({"dataset_id": "d1", "title": "T"}))
    assert "upload failed" in out.lower()
    assert "exceeds PRSM_MAX_UPLOAD_BYTES" in out               # real reason surfaced
    assert "Dataset Published" not in out                       # the fake success is gone


def test_upload_dataset_success_still_renders(monkeypatch):
    async def _fake(method, path, data=None, **k):
        return {"shard_count": 4, "cid": "bafy"}
    monkeypatch.setattr(mcp, "_call_node_api", _fake)
    out = _run(mcp.handle_prsm_upload_dataset({"dataset_id": "d1", "title": "T"}))
    assert "Dataset Published" in out and "Shards: 4" in out


# ── prsm_analyze surfaces a 402 (with funding hint) instead of empty output ───

def test_analyze_surfaces_402_with_funding_hint(monkeypatch):
    async def _fake(method, path, data=None, **k):
        return {"detail": "Insufficient FTNS balance to lock escrow"}
    monkeypatch.setattr(mcp, "_call_node_api", _fake)
    out = _run(mcp.handle_prsm_analyze({"query": "analyze X", "budget_ftns": 10.0}))
    assert "rejected" in out.lower()
    assert "Insufficient FTNS balance" in out
    assert "prsm_faucet" in out                                 # money path → funding hint
    assert "PRSM Analysis Result" not in out                    # no empty fake result


def test_analyze_nonfunding_4xx_has_no_funding_hint(monkeypatch):
    async def _fake(method, path, data=None, **k):
        return {"detail": "privacy tier not permitted for this content"}
    monkeypatch.setattr(mcp, "_call_node_api", _fake)
    out = _run(mcp.handle_prsm_analyze({"query": "x", "budget_ftns": 10.0}))
    assert "privacy tier not permitted" in out
    assert "prsm_faucet" not in out


def test_analyze_success_renders_result(monkeypatch):
    async def _fake(method, path, data=None, **k):
        return {"response": "The answer is 42.", "route": "swarm", "job_id": "j1"}
    monkeypatch.setattr(mcp, "_call_node_api", _fake)
    out = _run(mcp.handle_prsm_analyze({"query": "meaning?", "budget_ftns": 10.0}))
    assert "PRSM Analysis Result" in out and "42" in out


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
