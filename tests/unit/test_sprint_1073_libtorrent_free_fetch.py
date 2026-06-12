"""Sprint 1073 — libtorrent-free content fetch (completes the sp1070-1071 decoupling).

After sp1071 a default operator without libtorrent can PUBLISH Tier-A content
(LocalContentPublisher), but ContentUploader._fetch_content still required the
BitTorrent retriever (content_retriever) — with no libtorrent that's None, so a
non-libtorrent node could not retrieve content at all (its own OR others').

This routes _fetch_content through ContentProvider.request_content when the BT
retriever is absent. request_content already handles BOTH self-published content (the
local_content shortcut) AND cross-node fetch (P2P INLINE/CHUNKED, sp1020) — both
libtorrent-free. The libtorrent-present path (content_retriever set) is unchanged.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.content_uploader import ContentUploader


def _uploader(*, retriever=None, provider=None):
    """Build a ContentUploader with just the fetch dependencies stubbed."""
    up = ContentUploader.__new__(ContentUploader)   # bypass heavy __init__
    up.content_retriever = retriever
    up._content_provider = provider
    return up


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def test_uses_bt_retriever_when_present():
    """libtorrent path unchanged: when content_retriever is set, it's used."""
    retr = MagicMock()
    retr.fetch = AsyncMock(return_value=b"bt-bytes")
    prov = MagicMock()
    prov.request_content = AsyncMock(return_value=b"should-not-be-used")
    up = _uploader(retriever=retr, provider=prov)
    out = _run(up._fetch_content("cid-1"))
    assert out == b"bt-bytes"
    retr.fetch.assert_awaited_once_with("cid-1")
    prov.request_content.assert_not_awaited()


def test_falls_back_to_content_provider_without_libtorrent():
    """No BT retriever → fetch routes through ContentProvider.request_content
    (handles self-published + cross-node, libtorrent-free)."""
    prov = MagicMock()
    prov.request_content = AsyncMock(return_value=b"p2p-bytes")
    up = _uploader(retriever=None, provider=prov)
    out = _run(up._fetch_content("cid-2"))
    assert out == b"p2p-bytes"
    prov.request_content.assert_awaited_once_with("cid-2")


def test_content_provider_miss_returns_none():
    prov = MagicMock()
    prov.request_content = AsyncMock(return_value=None)
    up = _uploader(retriever=None, provider=prov)
    assert _run(up._fetch_content("missing")) is None


def test_content_provider_error_is_caught():
    prov = MagicMock()
    prov.request_content = AsyncMock(side_effect=RuntimeError("p2p down"))
    up = _uploader(retriever=None, provider=prov)
    assert _run(up._fetch_content("cid")) is None   # never raises


def test_no_retriever_no_provider_returns_none():
    up = _uploader(retriever=None, provider=None)
    assert _run(up._fetch_content("cid")) is None


# ── end-to-end: publish (no libtorrent) → ContentProvider self-fetch returns bytes ──

def test_end_to_end_local_publish_then_self_fetch(tmp_path):
    """The full libtorrent-free loop: LocalContentPublisher stages + computes the CID,
    the CID is registered with ContentProvider, and ContentProvider._fetch_local
    serves the staged bytes via the publisher shortcut (no libtorrent, no retriever)."""
    from prsm.node.local_content_publisher import LocalContentPublisher
    from prsm.node.content_provider import ContentProvider

    data = b"libtorrent-free end to end content " * 10
    pub = LocalContentPublisher(staging_dir=tmp_path, node_id="n")
    published = _run(pub.publish(data, provenance_id=""))
    cid = published.torrent_infohash

    # ContentProvider with the publisher wired (as node.initialize does), no retriever.
    prov = ContentProvider.__new__(ContentProvider)
    prov._local_content = {}
    prov.content_publisher = pub
    prov.content_retriever = None
    # register the published content (as ContentUploader does post-publish)
    prov.register_local_content(
        cid=cid, size_bytes=len(data),
        content_hash=__import__("hashlib").sha256(data).hexdigest())

    got = _run(prov._fetch_local(cid))
    assert got == data   # served from the staged file, no libtorrent involved


def test_fetched_tier_a_plaintext_passes_through(tmp_path):
    """A non-bundle (Tier-A) blob from the provider is returned as-is."""
    prov = MagicMock()
    prov.request_content = AsyncMock(return_value=b"plain tier-A bytes")
    up = _uploader(retriever=None, provider=prov)
    assert _run(up._fetch_content("cid-a")) == b"plain tier-A bytes"


def test_fetch_rejects_blob_not_matching_cid():
    """sp1075 review (premise) — a provider returning bytes whose v1 infohash != the
    requested 40-hex CID is rejected (binds fetched bytes to the CID)."""
    import hashlib
    from prsm.core.torrent_infohash import compute_v1_infohash_single_file
    real = b"the authentic content bytes"
    cid = compute_v1_infohash_single_file(real, hashlib.sha256(real).hexdigest())
    prov = MagicMock()
    prov.request_content = AsyncMock(return_value=b"DIFFERENT substituted bytes")
    up = _uploader(retriever=None, provider=prov)
    assert _run(up._fetch_content(cid)) is None   # mismatch → rejected


def test_fetch_accepts_blob_matching_cid():
    import hashlib
    from prsm.core.torrent_infohash import compute_v1_infohash_single_file
    real = b"the authentic content bytes"
    cid = compute_v1_infohash_single_file(real, hashlib.sha256(real).hexdigest())
    prov = MagicMock()
    prov.request_content = AsyncMock(return_value=real)
    up = _uploader(retriever=None, provider=prov)
    assert _run(up._fetch_content(cid)) == real


def test_fetched_tier_bc_bundle_is_decrypted(tmp_path):
    """sp1075 — a fetched Tier-B/C bundle is auto-detected + decrypted to plaintext."""
    from prsm.storage.content_store import ContentStore
    from prsm.node.local_content_publisher import LocalContentPublisher
    from prsm.compute.inference.models import ContentTier
    import prsm.storage as storage_mod

    store = ContentStore(data_dir=str(tmp_path / "store"), node_id="n")
    pub = LocalContentPublisher(staging_dir=tmp_path / "staging", content_store=store)
    data = b"encrypted-then-bundled content " * 100
    out = _run(pub.publish(data, provenance_id="", tier=ContentTier.B))
    bundle = out.staged_path.read_bytes()

    prov = MagicMock()
    prov.request_content = AsyncMock(return_value=bundle)
    up = _uploader(retriever=None, provider=prov)
    # _decode_bundle_if_present resolves the store via get_content_store()
    prev = storage_mod._content_store
    storage_mod._content_store = store
    try:
        got = _run(up._fetch_content(out.torrent_infohash))
    finally:
        storage_mod._content_store = prev
    assert got == data   # decrypted plaintext, not the bundle


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
