"""Sprint 1338 — /content/retrieve surfaces verifiable provenance + creator attribution.

The retrieve handler already RESOLVED the creator_eth_address + provenance_hash (for the §14
reputation record), but ContentRetrieveResponse dropped them at the response boundary — so a
data consumer got the bytes + integrity hash but could NOT see who created the dataset it
fetched or its provenance commitment (the "verifiable provenance and creator attribution" the
Vision positions PRSM on). These now flow through to the consumer.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


class _FakeContentProvider:
    def get_stats(self):
        return {"providers_attempted": 0}

    async def request_content(self, cid, timeout=30.0, verify_hash=True):
        return b"the dataset bytes"


class _Rec:
    """A real record object (not MagicMock) so unset attrs are truly None."""

    def __init__(self, creator=None, provenance=None):
        self.content_hash = "sha256-abc"
        self.filename = "nada-dataset.bin"
        self.creator_eth_address = creator
        self.provenance_hash = provenance


class _Idx:
    def __init__(self, rec):
        self._rec = rec

    def lookup(self, cid):
        return self._rec


def _client(rec):
    node = MagicMock()
    node.identity.node_id = "n"
    node.ftns_ledger = MagicMock()
    node.ftns_ledger._connected_address = "0xop"
    node._creator_reputation_tracker = None
    node._content_filter_store = None
    node.content_economy = None  # skip the sp1078 authenticated-creator override
    node.content_provider = _FakeContentProvider()
    node.content_index = _Idx(rec)
    node.content_uploader.uploaded_content = {}
    return TestClient(create_api_app(node, enable_security=False),
                      raise_server_exceptions=False)


def test_retrieve_surfaces_creator_and_provenance():
    creator = "0x" + "a" * 40
    provenance = "0x" + "cd" * 32
    body = _client(_Rec(creator=creator, provenance=provenance)).get(
        "/content/retrieve/bafy-nada").json()
    assert body["status"] == "success"
    assert body["creator_eth_address"] == creator
    assert body["provenance_hash"] == provenance
    # the integrity + data surface is unchanged (no regression)
    assert body["content_hash"] == "sha256-abc"
    assert body["data"]  # base64 bytes present


def test_retrieve_provenance_none_for_unattributed_content():
    """Content that predates creator threading → the fields are present but None
    (backward-compatible: a consumer distinguishes 'no attribution' from 'attributed')."""
    body = _client(_Rec(creator=None, provenance=None)).get(
        "/content/retrieve/bafy-legacy").json()
    assert body["status"] == "success"
    assert body["creator_eth_address"] is None
    assert body["provenance_hash"] is None


def test_response_model_declares_the_new_fields():
    """The fields are part of the response contract even on the error/not-found paths."""
    from pydantic import BaseModel
    # a not_found response still validates against the model with the new optional fields
    class _NoneProvider:
        def get_stats(self):
            return {}

        async def request_content(self, cid, **kwargs):
            return None

    node = MagicMock()
    node.identity.node_id = "n"
    node.ftns_ledger = MagicMock()
    node.ftns_ledger._connected_address = "0xop"
    node._creator_reputation_tracker = None
    node._content_filter_store = None
    node.content_economy = None
    node.content_provider = _NoneProvider()
    node.content_index = _Idx(_Rec())
    node.content_uploader.uploaded_content = {}
    body = TestClient(create_api_app(node, enable_security=False),
                      raise_server_exceptions=False).get("/content/retrieve/x").json()
    assert body["status"] == "not_found"
    assert "creator_eth_address" in body and body["creator_eth_address"] is None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
