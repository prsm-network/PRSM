"""Sprint 1340 — richer advertise metadata so topic search actually works.

Consumer-path #1 follow-on: sp1339 proved /content/search is network-wide, but a creator could
only make content findable via its FILENAME (a dataset named "data.bin" is unfindable by topic).
This adds optional title/description/tags to the upload — they flow into the content record's
metadata + the GOSSIP_CONTENT_ADVERTISE payload, and ContentIndex._index_keywords tokenizes the
string AND list-of-string values, so a peer finds the content by TOPIC network-wide.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.content_index import ContentIndex
from prsm.node.gossip import GOSSIP_CONTENT_ADVERTISE


class _FakeGossip:
    def subscribe(self, subtype, handler):
        pass


def _fed(metadata, *, filename="data.bin", cid="bafy-nada"):
    """A ContentIndex fed one remote advertisement carrying `metadata`."""
    idx = ContentIndex(gossip=_FakeGossip())
    ad = {"cid": cid, "provider_id": "peer", "filename": filename, "metadata": metadata}
    asyncio.run(idx._on_content_advertise(GOSSIP_CONTENT_ADVERTISE, ad, origin="peer"))
    return idx


# ── the core property: topic search over title / description / tags ───────────

def test_topic_search_via_title_and_description():
    idx = _fed({"title": "NADA Nutrition Survey",
                "description": "Household food security in East Africa"})
    assert idx.search("nutrition")[0].cid == "bafy-nada"    # title word
    assert idx.search("household")[0].cid == "bafy-nada"    # description word
    assert idx.search("africa")[0].cid == "bafy-nada"


def test_topic_search_via_tags_list():
    """tags is a LIST — _index_keywords must tokenize list-of-string values (sp1340)."""
    idx = _fed({"tags": ["kenya", "nutrition", "poverty"]}, filename="opaque.bin")
    assert idx.search("kenya")[0].cid == "bafy-nada"
    assert idx.search("poverty")[0].cid == "bafy-nada"


def test_topic_search_and_across_fields():
    idx = _fed({"title": "NADA Nutrition Survey", "tags": ["kenya", "poverty"]})
    # a title word AND a tag word must intersect to the same CID
    assert [r.cid for r in idx.search("nutrition kenya")] == ["bafy-nada"]


def test_non_string_metadata_values_are_ignored_not_indexed():
    idx = _fed({"replicas": 3, "nested": {"x": "y"}, "title": "findme"})
    assert idx.search("findme")[0].cid == "bafy-nada"   # str value indexed
    assert idx.search("3") == []                        # int value not indexed (also <2 chars)


# ── the upload endpoint threads the descriptive fields into the metadata ──────

def test_upload_threads_topic_metadata_into_uploader():
    node = MagicMock()
    captured = {}

    async def _cap(**kw):
        captured.update(kw)
        return SimpleNamespace(cid="bafy-x", content_hash="h", filename=kw.get("filename"),
                               size_bytes=1, metadata_uri=None)

    node.content_uploader.upload_text = _cap
    node.content_uploader.content_publisher = object()  # pass the preflight
    node._content_fingerprint_registry = None
    cli = TestClient(create_api_app(node, enable_security=False), raise_server_exceptions=False)
    cli.post("/content/upload", json={
        "text": "hello", "filename": "data.bin",
        "title": "NADA Nutrition Survey", "description": "food security",
        "tags": ["kenya", "nutrition"]})
    assert captured["metadata"] == {
        "title": "NADA Nutrition Survey", "description": "food security",
        "tags": ["kenya", "nutrition"]}


def test_upload_without_descriptive_fields_passes_none_metadata():
    node = MagicMock()
    captured = {}

    async def _cap(**kw):
        captured.update(kw)
        return SimpleNamespace(cid="bafy-x", content_hash="h", filename=kw.get("filename"),
                               size_bytes=1, metadata_uri=None)

    node.content_uploader.upload_text = _cap
    node.content_uploader.content_publisher = object()
    node._content_fingerprint_registry = None
    cli = TestClient(create_api_app(node, enable_security=False), raise_server_exceptions=False)
    cli.post("/content/upload", json={"text": "hi", "filename": "f.txt"})
    assert captured["metadata"] is None   # backward-compatible: no metadata when unset


def test_tags_count_cap_rejected():
    """A pathological tags list is rejected by the model (DoS/gossip-payload guard)."""
    node = MagicMock()
    node.content_uploader.content_publisher = object()
    cli = TestClient(create_api_app(node, enable_security=False), raise_server_exceptions=False)
    resp = cli.post("/content/upload", json={
        "text": "hi", "filename": "f.txt", "tags": [f"t{i}" for i in range(33)]})
    assert resp.status_code == 422   # exceeds max_length=32


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
