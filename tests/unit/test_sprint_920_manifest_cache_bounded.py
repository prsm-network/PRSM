"""Sprint 920 — bound the in-memory manifest cache (content data-plane review).

The content data-plane review found ContentStore._manifest_cache grows UNBOUNDED:
every store_local / store_local_with_artifacts inserts an entry and nothing
evicts, so repeated Tier-B/C publishes (e.g. POST /content/upload/shard with
distinct payloads) accumulate ciphertext + key-shares + manifests in memory until
OOM (the sp897 unbounded-resource class).

Fix: an LRU bound (OrderedDict + max-entries). Eviction is SAFE because the
in-memory cache is only a local fast-path — retrieve_local raises
ContentNotFoundError on a miss and ContentProvider falls back to the seeded
BitTorrent torrent (manifest + key-shares + shards persist on disk), so an
evicted entry costs a BT round-trip, never data loss.
"""
from __future__ import annotations

import pytest

from prsm.storage.content_store import ContentStore
from prsm.storage.exceptions import ContentNotFoundError


def _store(tmp_path, max_entries=3):
    return ContentStore(
        data_dir=str(tmp_path), node_id="n", manifest_cache_max=max_entries,
    )


def test_cache_put_bounds_and_evicts_oldest(tmp_path):
    s = _store(tmp_path, 3)
    for i in range(5):
        s._cache_put(f"k{i}", (b"", [], object()))
    keys = list(s._manifest_cache.keys())
    assert len(keys) == 3
    assert keys == ["k2", "k3", "k4"]   # oldest (k0, k1) evicted


def test_cache_get_touches_lru(tmp_path):
    s = _store(tmp_path, 3)
    for i in range(3):
        s._cache_put(f"k{i}", (b"", [], object()))
    assert s._cache_get("k0") is not None       # touch oldest → most-recent
    s._cache_put("k3", (b"", [], object()))      # evicts the new-oldest (k1)
    keys = set(s._manifest_cache.keys())
    assert keys == {"k0", "k2", "k3"}            # k0 survived, k1 evicted


def test_cache_get_absent_returns_none(tmp_path):
    s = _store(tmp_path, 3)
    assert s._cache_get("nope") is None


def test_default_cache_max_is_positive(tmp_path):
    s = ContentStore(data_dir=str(tmp_path), node_id="n")
    assert s._manifest_cache_max > 0


@pytest.mark.asyncio
async def test_eviction_forces_not_found_locally(tmp_path):
    """An evicted manifest → retrieve_local raises ContentNotFoundError (the
    BT-fallback-safe behavior); recent entries remain retrievable + correct."""
    s = _store(tmp_path, 2)
    h0 = await s.store_local(b"content-zero")
    await s.store_local(b"content-one")
    h2 = await s.store_local(b"content-two")     # evicts h0 (oldest)
    with pytest.raises(ContentNotFoundError):
        await s.retrieve_local(h0)
    assert await s.retrieve_local(h2) == b"content-two"


@pytest.mark.asyncio
async def test_store_with_artifacts_also_bounded(tmp_path):
    s = _store(tmp_path, 2)
    for i in range(4):
        await s.store_local_with_artifacts(f"artifact-{i}".encode())
    assert len(s._manifest_cache) <= 2
