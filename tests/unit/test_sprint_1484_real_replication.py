"""Sprint 1484 — make replication REAL (pin_content must fetch, not just promote).

`StorageProvider.pin_content` previously only "promoted" content that happened to
be local: `exists_local(...)` False -> return False, without ever fetching. So
`replicas=N` on every upload and CLI flag was DECORATIVE — no copy was ever placed
on another node, and published content died with the publisher's box. Worse, the
handler around it emitted STORAGE_CONFIRM + CONTENT_ADVERTISE on "success", i.e. it
advertised a pin that could never happen.

This now pulls bytes on a REMOTE peer's request, so the safety properties are
load-bearing and tested adversarially:
  * integrity — a poisoned/failed fetch stores nothing
  * operator consent — a filter-blocked CID is never fetched or stored
  * the gossiped size is UNTRUSTED — "declare 1 byte, serve 10 GiB" must be caught
    by re-checking the ACTUAL on-disk size
  * bounded concurrency — a burst of requests must not open unbounded fetches
"""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.storage_provider import StorageProvider

pytestmark = pytest.mark.asyncio

CID = "cid-remote-dataset"
# A REAL ContentHash hex — the format carries an algorithm-id prefix byte, so a
# bare 64-char hex string is rejected by ContentHash.from_hex (as it should be).
VALID_CID = "01db538533ffc72f423cb180698d002573a70bcf3c5875de29add9e7ef6785b701"


def _provider(tmp_path, *, payload=b"hello world", blocked=False,
              pledged_gb=10.0, fetch_ok=True):
    sp = StorageProvider.__new__(StorageProvider)
    sp.storage_available = True
    sp.pledged_gb = pledged_gb
    sp.pinned_content = {}
    sp.identity = SimpleNamespace(node_id="node-me")
    sp.replication_enabled = True
    sp.max_replica_bytes = 2 * 1024**3
    sp._replication_sem = asyncio.Semaphore(2)

    staging = tmp_path / "staging"
    staging.mkdir(exist_ok=True)

    publisher = MagicMock()
    publisher.staging_dir = staging
    publisher.register_local_publish_tier_a = MagicMock(return_value=True)

    async def _fetch(cid, dest_path, **kw):
        if not fetch_ok:
            return None
        Path(dest_path).write_bytes(payload)
        return Path(dest_path)

    cp = MagicMock()
    cp.request_content_to_file = AsyncMock(side_effect=_fetch)
    cp.content_publisher = publisher
    cp.register_local_content = MagicMock()
    sp._content_provider = cp

    sp._content_filter_store = (
        SimpleNamespace(is_cid_blocked=lambda c: True) if blocked else None)
    return sp, cp, publisher, staging


async def test_replicates_content_it_does_not_have():
    """★ The core fix: a storage request for absent content FETCHES and stores it."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sp, cp, publisher, staging = _provider(tmp, payload=b"A" * 4096)
        size = await sp._replicate_content(CID)
        assert size == 4096
        cp.request_content_to_file.assert_awaited_once()
        # Staged content-addressed so the streaming serve path can find it.
        expected_hash = hashlib.sha256(b"A" * 4096).hexdigest()
        assert (staging / expected_hash).is_file()
        publisher.register_local_publish_tier_a.assert_called_once_with(CID, expected_hash)
        cp.register_local_content.assert_called_once()
        assert cp.register_local_content.call_args.kwargs["size_bytes"] == 4096


async def test_failed_fetch_stores_nothing():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sp, cp, publisher, staging = _provider(tmp, fetch_ok=False)
        assert await sp._replicate_content(CID) is None
        publisher.register_local_publish_tier_a.assert_not_called()
        cp.register_local_content.assert_not_called()
        assert list(staging.iterdir()) == []      # no partial left behind


async def test_blocked_cid_is_never_fetched_or_stored():
    """★ Operator consent: a moderation decision must not be overridable by a
    stranger's gossip. The fetch must not even be attempted."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sp, cp, publisher, staging = _provider(tmp, blocked=True)
        assert await sp._replicate_content(CID) is None
        cp.request_content_to_file.assert_not_awaited()
        assert list(staging.iterdir()) == []


async def test_actual_size_over_capacity_is_discarded():
    """★ The gossiped size is attacker-controlled ("declare 1 byte, serve 10 GiB").
    The ACTUAL on-disk size must be re-checked against capacity after the fetch."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # 1 MiB payload but the node has ~0 capacity left.
        sp, cp, publisher, staging = _provider(
            tmp, payload=b"B" * (1024 * 1024), pledged_gb=0.0)
        assert await sp._replicate_content(CID) is None
        publisher.register_local_publish_tier_a.assert_not_called()
        assert list(staging.iterdir()) == []      # discarded, not stored


async def test_actual_size_over_per_item_cap_is_discarded():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sp, cp, publisher, staging = _provider(tmp, payload=b"C" * 8192)
        sp.max_replica_bytes = 4096          # smaller than the fetched object
        assert await sp._replicate_content(CID) is None
        assert list(staging.iterdir()) == []


async def test_replication_can_be_disabled():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sp, cp, publisher, staging = _provider(tmp)
        sp.replication_enabled = False
        assert await sp._replicate_content(CID) is None
        cp.request_content_to_file.assert_not_awaited()


async def test_no_content_provider_is_a_safe_noop():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sp, cp, publisher, staging = _provider(tmp)
        sp._content_provider = None
        assert await sp._replicate_content(CID) is None


async def test_concurrent_replications_are_bounded():
    """★ A burst of storage requests must not open unbounded simultaneous fetches."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sp, cp, publisher, staging = _provider(tmp, payload=b"D" * 128)
        sp._replication_sem = asyncio.Semaphore(2)
        live = {"now": 0, "peak": 0}

        async def _slow(cid, dest_path, **kw):
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
            await asyncio.sleep(0.02)
            Path(dest_path).write_bytes(b"D" * 128)
            live["now"] -= 1
            return Path(dest_path)

        cp.request_content_to_file = AsyncMock(side_effect=_slow)
        await asyncio.gather(*[sp._replicate_content(f"{CID}-{i}") for i in range(8)])
        assert live["peak"] <= 2, f"unbounded concurrency: peak={live['peak']}"


async def test_registration_failure_does_not_claim_success():
    """If the staged file cannot be re-registered we must NOT report a stored size —
    otherwise the caller confirms + advertises a copy we cannot actually serve."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sp, cp, publisher, staging = _provider(tmp)
        publisher.register_local_publish_tier_a = MagicMock(return_value=False)
        assert await sp._replicate_content(CID) is None
        cp.register_local_content.assert_not_called()


async def test_pin_content_promotes_local_without_fetching():
    """Legacy behavior preserved: content already local is promoted, not re-fetched."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sp, cp, publisher, staging = _provider(tmp)
        sp._get_content_size = AsyncMock(return_value=999)
        fake_store = SimpleNamespace(exists_local=AsyncMock(return_value=True))
        import prsm.storage as storage_mod
        orig = storage_mod.get_content_store
        storage_mod.get_content_store = lambda: fake_store
        try:
            ok = await sp.pin_content(VALID_CID)
        finally:
            storage_mod.get_content_store = orig
        assert ok is True
        assert sp.pinned_content[VALID_CID].size_bytes == 999
        cp.request_content_to_file.assert_not_awaited()


async def test_pin_content_REPLICATES_when_content_is_absent():
    """★ THE integration point this sprint changes: pin_content on content we do
    NOT have must trigger a replication fetch and then pin it. Without this the
    suite would pass with pin_content reverted to promote-only — verified: an
    earlier RED check went green precisely because no test covered this wiring."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sp, cp, publisher, staging = _provider(tmp, payload=b"E" * 2048)
        fake_store = SimpleNamespace(exists_local=AsyncMock(return_value=False))
        import prsm.storage as storage_mod
        orig = storage_mod.get_content_store
        storage_mod.get_content_store = lambda: fake_store
        try:
            ok = await sp.pin_content(VALID_CID)
        finally:
            storage_mod.get_content_store = orig
        assert ok is True, "absent content must be REPLICATED, not refused"
        cp.request_content_to_file.assert_awaited_once()
        assert sp.pinned_content[VALID_CID].size_bytes == 2048


async def test_pin_content_refuses_when_replication_fails():
    """A failed replication must NOT pin — otherwise the caller emits
    STORAGE_CONFIRM + CONTENT_ADVERTISE for a copy we do not hold."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sp, cp, publisher, staging = _provider(tmp, fetch_ok=False)
        fake_store = SimpleNamespace(exists_local=AsyncMock(return_value=False))
        import prsm.storage as storage_mod
        orig = storage_mod.get_content_store
        storage_mod.get_content_store = lambda: fake_store
        try:
            ok = await sp.pin_content(VALID_CID)
        finally:
            storage_mod.get_content_store = orig
        assert ok is False
        assert VALID_CID not in sp.pinned_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
