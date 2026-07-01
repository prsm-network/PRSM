"""Sprint 1343 — periodic content re-advertise so LATE-JOINING nodes get the catalog.

Gossip catch-up (GOSSIP_DIGEST_REQUEST/RESPONSE) only spans the ~24h gossip-log retention, so
without a re-advertise cadence content goes DARK after 24h and a node that joins later can never
discover it. ContentUploader.readvertise_all re-publishes each locally-hosted item's
GOSSIP_CONTENT_ADVERTISE — including the sp1340 topic metadata — so a late-joiner's ContentIndex
upserts + can search it. These tests prove the re-advertised payload is complete + topic-findable.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from prsm.node.content_index import ContentIndex
from prsm.node.content_uploader import ContentUploader, UploadedContent
from prsm.node.gossip import GOSSIP_CONTENT_ADVERTISE

_CREATOR = "0x" + "a" * 40
_PROV = "0x" + "cd" * 32


class _CapGossip:
    """Captures published ads; can be told to fail one cid (fail-soft test)."""

    def __init__(self, fail_cid=None):
        self.published = []
        self.fail_cid = fail_cid

    async def publish(self, subtype, data):
        if self.fail_cid and data.get("cid") == self.fail_cid:
            raise RuntimeError("gossip down for this one")
        self.published.append((subtype, data))

    def subscribe(self, *a, **k):
        pass


def _uploader(gossip, contents):
    """A ContentUploader with just the attributes readvertise_all touches (bypass __init__)."""
    up = ContentUploader.__new__(ContentUploader)
    up.gossip = gossip
    up.identity = SimpleNamespace(node_id="host-A")
    up.uploaded_content = {c.content_id: c for c in contents}
    return up


def _item(cid, filename="data.bin", *, metadata=None):
    return UploadedContent(
        content_id=cid, filename=filename, size_bytes=10, content_hash="h-" + cid,
        creator_id="host-A", creator_eth_address=_CREATOR, provenance_hash=_PROV,
        metadata=metadata or {})


# ── UploadedContent now retains the sp1340 metadata ───────────────────────────

def test_uploaded_content_retains_metadata():
    u = UploadedContent(content_id="c", filename="f", size_bytes=1, content_hash="h",
                        creator_id="n", metadata={"title": "T"})
    assert u.metadata == {"title": "T"}
    # default is an empty dict (backward-safe for every other construction site)
    assert UploadedContent(content_id="c", filename="f", size_bytes=1,
                           content_hash="h", creator_id="n").metadata == {}


# ── readvertise_all re-publishes the full, topic-searchable payload ───────────

def test_readvertise_republishes_all_with_metadata_and_attribution():
    g = _CapGossip()
    up = _uploader(g, [_item("cid1", "nada.csv",
                             metadata={"title": "NADA Nutrition Survey", "tags": ["kenya"]})])
    n = asyncio.run(up.readvertise_all())
    assert n == 1
    subtype, data = g.published[0]
    assert subtype == GOSSIP_CONTENT_ADVERTISE
    assert data["cid"] == "cid1"
    assert data["metadata"] == {"title": "NADA Nutrition Survey", "tags": ["kenya"]}
    assert data["creator_eth_address"] == _CREATOR
    assert data["provenance_hash"] == _PROV
    assert data["provider_id"] == "host-A"


def test_late_joiner_discovers_readvertised_content_by_topic():
    """END-TO-END: a fresh node (missed the original ad) indexes the re-advertised payload and
    finds the content by TOPIC — the whole point of the re-advertise."""
    g = _CapGossip()
    up = _uploader(g, [_item("cid1", "opaque.bin",
                             metadata={"title": "NADA Nutrition Survey", "tags": ["kenya"]})])
    asyncio.run(up.readvertise_all())
    _, ad = g.published[0]

    late_joiner_index = ContentIndex(gossip=_CapGossip())
    asyncio.run(late_joiner_index._on_content_advertise(
        GOSSIP_CONTENT_ADVERTISE, ad, origin="host-A"))
    assert late_joiner_index.search("nutrition")[0].cid == "cid1"   # title
    assert late_joiner_index.search("kenya")[0].cid == "cid1"       # tag
    rec = late_joiner_index.search("nutrition")[0]
    assert rec.creator_eth_address == _CREATOR                      # attribution rode along


def test_readvertise_is_fail_soft_per_item():
    g = _CapGossip(fail_cid="bad")
    up = _uploader(g, [_item("good1"), _item("bad"), _item("good2")])
    n = asyncio.run(up.readvertise_all())
    assert n == 2  # the two good ones published; the bad one didn't abort the sweep
    assert {d["cid"] for _, d in g.published} == {"good1", "good2"}


def test_readvertise_empty_is_zero_no_publish():
    g = _CapGossip()
    assert asyncio.run(_uploader(g, []).readvertise_all()) == 0
    assert g.published == []


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
