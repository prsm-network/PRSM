"""Sprint 1348 — fix the fake-success bug CLASS at the source: _call_node_api raises on 4xx/5xx.

sp1346/1347 hand-fixed a few handlers. This eliminates the whole class: _call_node_api used to
return resp.json() regardless of HTTP status, so a 4xx (FastAPI HTTPException → {"detail": ...})
reached every handler as a plain dict and any that read success keys rendered fake-success/blank.
Now a non-2xx status raises NodeAPIError, carrying the real reason (+ the sp1346 funding hint in
__str__), so every handler's error surfaces via its own except or the call_tool dispatch.
"""
from __future__ import annotations

import asyncio

import pytest

import prsm.mcp_server as mcp


# ── _raise_for_status ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("code", [200, 201, 202, 204, 302])
def test_no_raise_on_success_status(code):
    mcp._raise_for_status(code, {"success": True, "output": "x"})  # must not raise


def test_raises_on_4xx_detail():
    with pytest.raises(mcp.NodeAPIError) as ei:
        mcp._raise_for_status(402, {"detail": "Insufficient FTNS balance to lock escrow"})
    assert ei.value.status == 402
    assert "Insufficient FTNS balance" in ei.value.message


def test_raises_on_4xx_error_key():
    with pytest.raises(mcp.NodeAPIError) as ei:
        mcp._raise_for_status(422, {"error": "bad parameter"})
    assert "bad parameter" in ei.value.message


def test_raises_on_4xx_with_no_error_keys_falls_back_to_status():
    with pytest.raises(mcp.NodeAPIError) as ei:
        mcp._raise_for_status(500, {"unexpected": "shape"})
    assert "HTTP 500" in str(ei.value)


def test_raises_on_text_body():
    with pytest.raises(mcp.NodeAPIError) as ei:
        mcp._raise_for_status(503, "service unavailable")
    assert "service unavailable" in ei.value.message


# ── NodeAPIError.__str__ carries the sp1346 funding hint ──────────────────────

def test_error_str_folds_in_funding_hint():
    s = str(mcp.NodeAPIError(402, "Insufficient FTNS balance to lock escrow"))
    assert "HTTP 402" in s and "prsm_faucet" in s


def test_error_str_no_hint_for_unrelated():
    s = str(mcp.NodeAPIError(404, "model not found"))
    assert "HTTP 404" in s and "prsm_faucet" not in s


# ── end-to-end: a RAISED error surfaces (never fake-success) ──────────────────

def _raiser(status, message):
    async def _fake(method, path, data=None, **k):
        raise mcp.NodeAPIError(status, message)
    return _fake


def test_analyze_surfaces_raised_402_with_hint(monkeypatch):
    monkeypatch.setattr(mcp, "_call_node_api",
                        _raiser(402, "Insufficient FTNS balance to lock escrow"))
    out = asyncio.run(mcp.handle_prsm_analyze({"query": "x", "budget_ftns": 10.0}))
    assert "Insufficient FTNS balance" in out
    assert "prsm_faucet" in out                    # via NodeAPIError.__str__
    assert "PRSM Analysis Result" not in out       # no empty fake result


def test_upload_dataset_surfaces_raised_413_not_fake_success(monkeypatch):
    monkeypatch.setattr(mcp, "_call_node_api", _raiser(413, "text size exceeds the cap"))
    out = asyncio.run(mcp.handle_prsm_upload_dataset({"dataset_id": "d", "title": "T"}))
    assert "exceeds the cap" in out
    assert "Dataset Published" not in out          # the fake success is gone


def test_get_dataset_surfaces_raised_retrieve_error(monkeypatch):
    # find_and_fetch analog through MCP: a raised retrieve error must surface, not preview blank
    async def _fake(method, path, data=None, **k):
        if "search" in path:
            return {"results": [{"cid": "bafy1", "filename": "f"}]}
        raise mcp.NodeAPIError(404, "cid not found on any provider")
    monkeypatch.setattr(mcp, "_call_node_api", _fake)
    out = asyncio.run(mcp.handle_prsm_get_dataset({"query": "x"}))
    assert "not found" in out.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
