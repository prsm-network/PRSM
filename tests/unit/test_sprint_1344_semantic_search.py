"""Sprint 1344 — semantic (embedding-similarity) content search: the consumer surface.

The top-k cosine primitive (_SemanticIndex.find_top_k) + the query-embed (_embedding_fn) already
existed — built for the QueryOrchestrator, never exposed to a data consumer. sp1344 wires them
into ContentUploader.semantic_search + GET /content/search/semantic + SDK/CLI, so a consumer can
find CONCEPTUALLY related content, not just keyword matches. Honestly gated: no embedding function
wired → [] + semantic_available False (keyword /content/search stays the always-available path).
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.content_uploader import ContentUploader

_CREATOR = "0x" + "a" * 40
_PROV = "0x" + "cd" * 32


class _FakeSemIndex:
    """Controls find_top_k output (the algorithm itself is tested elsewhere)."""

    def __init__(self, triples):
        self._triples = triples  # list of (cid, similarity, creator_id)

    def find_top_k(self, embedding, k):
        return self._triples[:k]


class _Rec:
    def __init__(self, filename, creator=None, prov=None, metadata=None):
        self.filename = filename
        self.creator_eth_address = creator
        self.provenance_hash = prov
        self.metadata = metadata or {}


class _FakeIndex:
    def __init__(self, records):
        self._records = records

    def lookup(self, cid):
        return self._records.get(cid)


async def _embed(text):
    return [1.0, 0.0, 0.0]  # any non-None vector; find_top_k is faked


def _uploader(triples, records, embedding_fn=_embed):
    up = ContentUploader.__new__(ContentUploader)
    up._embedding_fn = embedding_fn
    up._semantic_index = _FakeSemIndex(triples)
    up.content_index = _FakeIndex(records)
    up.uploaded_content = {}
    return up


# ── ContentUploader.semantic_search ──────────────────────────────────────────

def test_semantic_search_ranks_and_decorates():
    up = _uploader(
        [("cid1", 0.95, "cr1"), ("cid2", 0.60, "cr2")],
        {"cid1": _Rec("nada.csv", _CREATOR, _PROV, {"title": "NADA Nutrition"})})
    rows = asyncio.run(up.semantic_search("household food insecurity", top_k=10))
    assert [r["cid"] for r in rows] == ["cid1", "cid2"]        # similarity order preserved
    assert rows[0]["similarity"] == 0.95
    assert rows[0]["creator_eth_address"] == _CREATOR          # decorated from the index record
    assert rows[0]["provenance_hash"] == _PROV
    assert rows[0]["metadata"] == {"title": "NADA Nutrition"}
    assert rows[1]["filename"] is None                         # cid2 had no record → None-safe


def test_min_similarity_filters():
    up = _uploader([("cid1", 0.95, "cr1"), ("cid2", 0.60, "cr2")], {})
    rows = asyncio.run(up.semantic_search("q", top_k=10, min_similarity=0.8))
    assert [r["cid"] for r in rows] == ["cid1"]               # 0.60 dropped


def test_unavailable_without_embedding_fn():
    up = _uploader([("cid1", 0.9, "c")], {}, embedding_fn=None)
    assert asyncio.run(up.semantic_search("q")) == []


def test_unembeddable_query_returns_empty():
    async def _none(text):
        return None
    up = _uploader([("cid1", 0.9, "c")], {}, embedding_fn=_none)
    assert asyncio.run(up.semantic_search("q")) == []


def test_embed_exception_is_empty_not_crash():
    async def _boom(text):
        raise RuntimeError("model down")
    up = _uploader([("cid1", 0.9, "c")], {}, embedding_fn=_boom)
    assert asyncio.run(up.semantic_search("q")) == []


# ── /content/search/semantic endpoint ────────────────────────────────────────

def _client(uploader):
    node = MagicMock()
    node.content_uploader = uploader
    return TestClient(create_api_app(node, enable_security=False),
                      raise_server_exceptions=False)


def test_endpoint_returns_rows_and_available():
    body = _client(_uploader(
        [("cid1", 0.9, "cr")], {"cid1": _Rec("nada.csv", _CREATOR, _PROV)}
    )).get("/content/search/semantic", params={"q": "nutrition"}).json()
    assert body["semantic_available"] is True
    assert body["count"] == 1
    assert body["results"][0]["cid"] == "cid1"
    assert body["results"][0]["creator_eth_address"] == _CREATOR


def test_endpoint_semantic_unavailable_when_no_uploader():
    node = MagicMock()
    node.content_uploader = None
    body = TestClient(create_api_app(node, enable_security=False),
                      raise_server_exceptions=False).get(
        "/content/search/semantic", params={"q": "x"}).json()
    assert body["semantic_available"] is False
    assert body["results"] == []


def test_endpoint_top_k_bounds():
    resp = _client(_uploader([], {})).get(
        "/content/search/semantic", params={"q": "x", "top_k": 0})
    assert resp.status_code == 422


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
