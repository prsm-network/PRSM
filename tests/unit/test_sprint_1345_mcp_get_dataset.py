"""Sprint 1345 — MCP front-door: prsm_get_dataset (find -> fetch -> verify) + semantic browsing.

The MCP server (the Vision's front door — "the model routes") let an LLM SEARCH datasets but had
NO retrieve tool, so the §4 flagship (a researcher's LLM finds + fetches a dataset) was broken at
the front door. prsm_get_dataset closes it: resolve a query/cid, retrieve, integrity-verify the
bytes, surface creator + on-chain provenance, and return a text preview the model can reason over.
Plus semantic=true browsing exposes the sp1344 embedding search.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib

import prsm.mcp_server as mcp

_CREATOR = "0x" + "a" * 40
_PROV = "0x" + "cd" * 32


def _retrieve_ok(raw=b"the dataset text", *, content_hash=None,
                 creator=_CREATOR, prov=_PROV, filename="nada.csv"):
    return {
        "status": "success", "data": base64.b64encode(raw).decode(),
        "content_hash": content_hash or hashlib.sha256(raw).hexdigest(),
        "creator_eth_address": creator, "provenance_hash": prov,
        "filename": filename, "size_bytes": len(raw)}


def _api(routes):
    async def _fake(method, path, *a, **k):
        for key, val in routes.items():
            if key in path:
                return val
        raise RuntimeError(f"unexpected path {path}")
    return _fake


def _run(coro):
    return asyncio.run(coro)


# ── prsm_get_dataset: the flagship find -> fetch -> verify ─────────────────────

def test_get_dataset_by_query_verifies_and_previews(monkeypatch):
    raw = b"household food insecurity survey rows across east africa"
    monkeypatch.setattr(mcp, "_call_node_api", _api({
        "/content/search?": {"results": [
            {"cid": "bafy1", "filename": "nada.csv",
             "creator_eth_address": _CREATOR, "provenance_hash": _PROV}]},
        "/content/retrieve/": _retrieve_ok(raw)}))
    out = _run(mcp.handle_prsm_get_dataset({"query": "nutrition"}))
    assert "bafy1" in out
    assert "integrity: VERIFIED" in out
    assert _CREATOR in out            # creator surfaced
    assert _PROV in out               # provenance surfaced
    assert "food insecurity" in out   # text preview the model can read


def test_get_dataset_by_cid_skips_search(monkeypatch):
    called = {"search": False}

    async def _fake(method, path, *a, **k):
        if "search" in path:
            called["search"] = True
            return {"results": []}
        return _retrieve_ok(b"hello world")
    monkeypatch.setattr(mcp, "_call_node_api", _fake)
    out = _run(mcp.handle_prsm_get_dataset({"cid": "bafyX"}))
    assert called["search"] is False
    assert "bafyX" in out and "VERIFIED" in out


def test_integrity_failed_on_tampered_bytes(monkeypatch):
    tampered = _retrieve_ok(b"honest", content_hash=hashlib.sha256(b"different").hexdigest())
    monkeypatch.setattr(mcp, "_call_node_api", _api({"/content/retrieve/": tampered}))
    out = _run(mcp.handle_prsm_get_dataset({"cid": "c"}))
    assert "integrity: FAILED" in out


def test_no_query_or_cid_is_actionable():
    out = _run(mcp.handle_prsm_get_dataset({}))
    assert "Provide `query`" in out


def test_no_match_points_to_browse(monkeypatch):
    monkeypatch.setattr(mcp, "_call_node_api", _api({"/content/search?": {"results": []}}))
    out = _run(mcp.handle_prsm_get_dataset({"query": "nothing here"}))
    assert "No dataset found" in out and "prsm_list_datasets" in out


def test_semantic_query_uses_semantic_endpoint(monkeypatch):
    paths = []

    async def _fake(method, path, *a, **k):
        paths.append(path)
        return {"results": [{"cid": "s1"}], "semantic_available": True} if "semantic" in path \
            else _retrieve_ok(b"x")
    monkeypatch.setattr(mcp, "_call_node_api", _fake)
    _run(mcp.handle_prsm_get_dataset({"query": "concept", "semantic": True}))
    assert any("/content/search/semantic" in p for p in paths)   # search leg used semantic


def test_semantic_unavailable_is_clear(monkeypatch):
    monkeypatch.setattr(mcp, "_call_node_api", _api({
        "/content/search/semantic": {"semantic_available": False, "results": []}}))
    out = _run(mcp.handle_prsm_get_dataset({"query": "x", "semantic": True}))
    assert "Semantic search unavailable" in out


def test_binary_content_not_previewed_as_text(monkeypatch):
    raw = b"\xff\xfe\x00\x01\x02binary-blob"
    monkeypatch.setattr(mcp, "_call_node_api", _api({"/content/retrieve/": _retrieve_ok(raw)}))
    out = _run(mcp.handle_prsm_get_dataset({"cid": "c"}))
    assert "binary content" in out


def test_not_retrievable_surfaces_status(monkeypatch):
    monkeypatch.setattr(mcp, "_call_node_api", _api({
        "/content/retrieve/": {"status": "not_found", "providers_tried": 3}}))
    out = _run(mcp.handle_prsm_get_dataset({"cid": "ghost"}))
    assert "not retrievable" in out and "not_found" in out


# ── prsm_list_datasets semantic option ────────────────────────────────────────

def test_list_datasets_semantic_switches_endpoint(monkeypatch):
    seen = {}

    async def _fake(method, path, *a, **k):
        seen["path"] = path
        return {"results": [], "semantic_available": True}
    monkeypatch.setattr(mcp, "_call_node_api", _fake)
    _run(mcp.handle_prsm_list_datasets({"search": "topic", "semantic": True}))
    assert "/content/search/semantic" in seen["path"]


# ── registration ──────────────────────────────────────────────────────────────

def test_tool_registered():
    assert "prsm_get_dataset" in mcp.TOOL_HANDLERS
    assert any(t.name == "prsm_get_dataset" for t in mcp.TOOLS)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
