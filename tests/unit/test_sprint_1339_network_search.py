"""Sprint 1339 — verify /content/search returns NETWORK-wide results (+ surface attribution).

Consumer-path verification #1: does a non-operator's content search see the whole network, or
only the node's own uploads? ANSWER (proven here): network-wide. ContentIndex subscribes to
GOSSIP_CONTENT_ADVERTISE and indexes each PEER's advertisement (filename + string metadata) into
the keyword index that search() reads — so a node finds content it learned purely via gossip,
never uploaded. These tests guard that property from regression.

The verification also surfaced a gap (fixed here): search result rows dropped
creator_eth_address + provenance_hash (parity with the sp1338 retrieve surface) — so a consumer
couldn't see verifiable creator/provenance at DISCOVERY time. Now they're on each hit.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.content_index import ContentIndex
from prsm.node.gossip import GOSSIP_CONTENT_ADVERTISE

_CREATOR = "0x" + "a" * 40
_PROV = "0x" + "cd" * 32


class _FakeGossip:
    def __init__(self):
        self.handlers = {}

    def subscribe(self, subtype, handler):
        self.handlers[subtype] = handler


def _advertise(cid, filename, *, creator=_CREATOR, provenance=_PROV,
               metadata=None, provider="remote-peer-1"):
    return {
        "cid": cid, "provider_id": provider, "filename": filename,
        "content_hash": "h-" + cid, "creator_id": provider,
        "creator_eth_address": creator, "provenance_hash": provenance,
        "metadata": metadata or {},
    }


def _fed_index(*ads):
    """A ContentIndex populated ONLY by simulated remote gossip advertisements."""
    idx = ContentIndex(gossip=_FakeGossip())

    async def _feed():
        for ad in ads:
            await idx._on_content_advertise(GOSSIP_CONTENT_ADVERTISE, ad, origin=ad["provider_id"])
    asyncio.run(_feed())
    return idx


# ── the core property: search is network-wide (gossip-fed) ────────────────────

def test_search_finds_content_learned_via_gossip():
    """A REMOTE peer's dataset — this node never uploaded it — is keyword-findable."""
    idx = _fed_index(_advertise("bafy-nada", "nada-nutrition-survey-africa.csv"))
    assert [r.cid for r in idx.search("nutrition")] == ["bafy-nada"]
    assert idx.search("survey")[0].cid == "bafy-nada"
    rec = idx.search("africa")[0]
    assert "remote-peer-1" in rec.providers          # provider is the remote peer
    assert rec.creator_eth_address == _CREATOR       # attribution rode the advertisement
    assert rec.provenance_hash == _PROV


def test_search_indexes_string_metadata_not_only_filename():
    idx = _fed_index(_advertise("bafy-x", "data.bin",
                                metadata={"title": "Kenya Health Records"}, provider="peer-2"))
    assert idx.search("kenya")[0].cid == "bafy-x"    # found via metadata, opaque filename


def test_search_spans_multiple_peers_with_and_semantics():
    idx = _fed_index(
        _advertise("cid-a", "climate-data-brazil.nc", provider="peer-a"),
        _advertise("cid-b", "climate-data-india.nc", provider="peer-b"))
    assert {r.cid for r in idx.search("climate")} == {"cid-a", "cid-b"}   # both peers
    assert [r.cid for r in idx.search("climate brazil")] == ["cid-a"]     # AND narrows


def test_start_wires_the_gossip_subscription():
    """The subscription that MAKES it network-fed is actually installed."""
    g = _FakeGossip()
    ContentIndex(gossip=g).start()
    assert GOSSIP_CONTENT_ADVERTISE in g.handlers


# ── consumer boundary: the /content/search API returns gossip-fed hits + attribution ──

def _search_client(idx):
    node = MagicMock()
    node.content_index = idx
    node._creator_reputation_tracker = None   # → tier NEW, no filtering
    node._creator_stake_client = None
    node._content_filter_store = None
    return TestClient(create_api_app(node, enable_security=False),
                      raise_server_exceptions=False)


def test_search_endpoint_returns_gossip_fed_result_with_attribution():
    idx = _fed_index(_advertise("bafy-nada", "nada-nutrition-survey.csv"))
    body = _search_client(idx).get("/content/search", params={"q": "nutrition"}).json()
    rows = body["results"] if isinstance(body, dict) and "results" in body else body
    hit = next(r for r in rows if r["cid"] == "bafy-nada")
    # sp1339 — verifiable creator + provenance surfaced at DISCOVERY time
    assert hit["creator_eth_address"] == _CREATOR
    assert hit["provenance_hash"] == _PROV


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
