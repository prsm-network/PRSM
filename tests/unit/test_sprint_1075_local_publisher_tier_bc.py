"""Sprint 1075 — LocalContentPublisher Tier B/C via the artifact bundle (brick 2).

The proof that encrypted (Tier B/C) content publishes + round-trips with NO
libtorrent: encrypt via the ContentStore → bundle the artifacts into one blob (sp1074)
→ stage + identify it exactly like a Tier-A single file (sp1070 infohash) → and
unbundle_and_decrypt recovers the original plaintext. So a default operator can
publish encrypted content and it's servable over the single-blob P2P substrate.
"""
from __future__ import annotations

import hashlib

import pytest

from prsm.compute.inference.models import ContentTier
from prsm.core.torrent_infohash import compute_v1_infohash_single_file
from prsm.node.artifact_bundle import is_bundle
from prsm.node.local_content_publisher import (
    LocalContentPublisher, unbundle_and_decrypt)
from prsm.storage.content_store import ContentStore


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def _store(tmp_path):
    return ContentStore(data_dir=str(tmp_path / "store"), node_id="test-node")


def test_tier_b_publish_stages_a_bundle_with_tier_a_infohash(tmp_path):
    store = _store(tmp_path)
    pub = LocalContentPublisher(staging_dir=tmp_path / "staging", content_store=store)
    data = b"secret encrypted content " * 200
    out = _run(pub.publish(data, provenance_id="0xprov", tier=ContentTier.B))

    blob = out.staged_path.read_bytes()
    assert is_bundle(blob)                         # staged file is a bundle, not plaintext
    assert blob != data                            # encrypted, not the plaintext
    # CID is the Tier-A single-file infohash OF THE BUNDLE
    name = hashlib.sha256(blob).hexdigest()
    assert out.torrent_infohash == compute_v1_infohash_single_file(blob, name)
    assert out.manifest.infohash == out.torrent_infohash


def test_tier_b_roundtrip_unbundle_decrypt_recovers_plaintext(tmp_path):
    store = _store(tmp_path)
    pub = LocalContentPublisher(staging_dir=tmp_path / "staging", content_store=store)
    data = b"the quick brown fox jumps over the lazy dog. " * 500
    out = _run(pub.publish(data, provenance_id="", tier=ContentTier.B))

    blob = out.staged_path.read_bytes()
    recovered = _run(unbundle_and_decrypt(blob, store))
    assert recovered == data                       # full encrypt→bundle→unbundle→decrypt loop


def test_tier_c_also_supported(tmp_path):
    store = _store(tmp_path)
    pub = LocalContentPublisher(staging_dir=tmp_path / "staging", content_store=store)
    data = b"tier C erasure-coded content " * 300
    out = _run(pub.publish(data, provenance_id="", tier=ContentTier.C))
    recovered = _run(unbundle_and_decrypt(out.staged_path.read_bytes(), store))
    assert recovered == data


def test_local_publish_path_shortcut_works_for_tier_bc(tmp_path):
    """The infohash → staged-bundle shortcut works for B/C too (self-fetch path)."""
    store = _store(tmp_path)
    pub = LocalContentPublisher(staging_dir=tmp_path / "staging", content_store=store)
    out = _run(pub.publish(b"x" * 5000, provenance_id="", tier=ContentTier.B))
    assert pub.local_publish_path(out.torrent_infohash) == out.staged_path


def test_tier_b_without_content_store_raises(tmp_path):
    from prsm.storage import close_content_store
    close_content_store()   # ensure no global store is resolvable
    pub = LocalContentPublisher(staging_dir=tmp_path / "staging", content_store=None)
    with pytest.raises(RuntimeError):
        _run(pub.publish(b"secret", provenance_id="", tier=ContentTier.B))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
