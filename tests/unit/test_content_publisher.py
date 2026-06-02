"""Unit tests for ContentPublisher + ContentRetriever (PR 2a Tier A + PR 2b Tier B/C)."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from prsm.compute.inference.models import ContentTier
from prsm.node.content_publisher import (
    ContentPublisher,
    ContentRetriever,
    PublishedContent,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeManifest:
    infohash: str
    name: str
    total_size: int


@dataclass
class _FakeDownloadResult:
    success: bool
    error: Optional[str] = None


class _FakeBitTorrentProvider:
    """Captures seed_content calls; returns a deterministic manifest.

    Mirrors the real BT provider for the publish path: handles both
    single-file (Tier A) and directory (Tier B/C) seed targets, and
    stashes the staged bytes in :attr:`seeded` so the retriever fake
    can serve them back on the next fetch.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.next_return: Optional[_FakeManifest] = None
        # infohash -> {filename: bytes} (single-element dict for Tier A)
        self.seeded: Dict[str, Dict[str, bytes]] = {}

    async def seed_content(
        self,
        path: Path,
        name: Optional[str] = None,
        provenance_id: Optional[str] = None,
        piece_length: int = 262144,
    ) -> Optional[_FakeManifest]:
        self.calls.append({
            "path": Path(path),
            "name": name,
            "provenance_id": provenance_id,
            "piece_length": piece_length,
        })
        if self.next_return is not None:
            return self.next_return

        # Synthesize a manifest from the path's content. Hash a stable
        # representation so the same staged content yields the same infohash.
        path = Path(path)
        if path.is_dir():
            files = sorted(p for p in path.iterdir() if p.is_file())
            payload_map = {p.name: p.read_bytes() for p in files}
            digest_input = b"".join(
                fname.encode("utf-8") + b"\x00" + body
                for fname, body in sorted(payload_map.items())
            )
            total_size = sum(len(b) for b in payload_map.values())
        else:
            payload_map = {path.name: path.read_bytes()}
            digest_input = path.read_bytes()
            total_size = len(digest_input)

        infohash = hashlib.sha256(digest_input + b"::infohash").hexdigest()[:40]
        self.seeded[infohash] = payload_map
        return _FakeManifest(
            infohash=infohash,
            name=name or path.name,
            total_size=total_size,
        )


class _FakeBitTorrentRequester:
    """Writes the requested files into save_path; returns success.

    If ``provider`` is supplied, files are pulled from
    ``provider.seeded[infohash]`` (matching the publish-side fake).
    Otherwise falls back to the legacy ``payload_for`` dict (used by
    standalone retriever-only tests).
    """

    def __init__(self, provider: Optional[_FakeBitTorrentProvider] = None) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.payload_for: Dict[str, bytes] = {}
        self.next_result: Optional[_FakeDownloadResult] = None
        self.extra_files: int = 0  # set >0 to simulate multi-file torrents
        self.provider = provider

    async def request_content(
        self,
        infohash: str,
        save_path: Path,
        timeout: Optional[float] = None,
        progress_callback: Any = None,
    ) -> _FakeDownloadResult:
        self.calls.append({
            "infohash": infohash,
            "save_path": Path(save_path),
            "timeout": timeout,
        })
        if self.next_result is not None:
            return self.next_result

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Real BT clients drop multi-file torrents inside a subdirectory
        # named after the torrent. Mirror that behaviour for Tier B/C.
        if self.provider is not None and infohash in self.provider.seeded:
            files = self.provider.seeded[infohash]
            if len(files) == 1:
                # Tier A — write the single file at the save_path root.
                fname, payload = next(iter(files.items()))
                (save_path / fname).write_bytes(payload)
            else:
                # Tier B/C — drop into a subdirectory.
                subdir = save_path / infohash
                subdir.mkdir(parents=True, exist_ok=True)
                for fname, payload in files.items():
                    (subdir / fname).write_bytes(payload)
            return _FakeDownloadResult(success=True)

        # Legacy single-file path (retriever-only tests).
        payload = self.payload_for.get(infohash, b"")
        (save_path / "content.bin").write_bytes(payload)
        for i in range(self.extra_files):
            (save_path / f"extra-{i}.bin").write_bytes(b"x")
        return _FakeDownloadResult(success=True)


# ---------------------------------------------------------------------------
# ContentPublisher
# ---------------------------------------------------------------------------


class TestContentPublisher:
    @pytest.fixture
    def staging_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "staging"

    @pytest.fixture
    def provider(self) -> _FakeBitTorrentProvider:
        return _FakeBitTorrentProvider()

    @pytest.fixture
    def publisher(
        self, provider: _FakeBitTorrentProvider, staging_dir: Path
    ) -> ContentPublisher:
        return ContentPublisher(bt_provider=provider, staging_dir=staging_dir)

    @pytest.mark.asyncio
    async def test_publish_tier_a_returns_handle(
        self, publisher: ContentPublisher, provider: _FakeBitTorrentProvider
    ) -> None:
        data = b"hello PRSM"
        result = await publisher.publish(data, provenance_id="0xprov-1")

        assert isinstance(result, PublishedContent)
        assert result.torrent_infohash != ""
        assert result.staged_path.exists()
        assert result.staged_path.read_bytes() == data
        assert result.manifest.total_size == len(data)
        assert len(provider.calls) == 1
        assert provider.calls[0]["provenance_id"] == "0xprov-1"

    @pytest.mark.asyncio
    async def test_publish_uses_sha256_filename(
        self, publisher: ContentPublisher
    ) -> None:
        data = b"deterministic"
        result = await publisher.publish(data, provenance_id="0xp")

        expected_name = hashlib.sha256(data).hexdigest()
        assert result.staged_path.name == expected_name

    @pytest.mark.asyncio
    async def test_publish_idempotent_for_same_bytes(
        self, publisher: ContentPublisher, provider: _FakeBitTorrentProvider
    ) -> None:
        data = b"same content"
        first = await publisher.publish(data, provenance_id="0xp")
        second = await publisher.publish(data, provenance_id="0xp")

        # Same staged path; same infohash (synthesized from content).
        assert first.staged_path == second.staged_path
        assert first.torrent_infohash == second.torrent_infohash
        # Two seed_content calls — BT layer is responsible for its own dedup.
        assert len(provider.calls) == 2

    @pytest.mark.asyncio
    async def test_publish_tier_b_round_trip(
        self,
        provider: _FakeBitTorrentProvider,
        staging_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Tier B publishes via ContentStore + BT, retriever decrypts back to plaintext."""
        from prsm.storage import ContentStore

        store = ContentStore(data_dir=str(tmp_path / "store"), node_id="test-node")
        pub = ContentPublisher(
            bt_provider=provider, staging_dir=staging_dir, content_store=store
        )
        requester = _FakeBitTorrentRequester(provider=provider)
        ret = ContentRetriever(
            bt_requester=requester,
            cache_dir=tmp_path / "cache",
            content_store=store,
        )

        data = b"tier-B encrypted payload " * 50
        result = await pub.publish(
            data, provenance_id="0xprov-B", tier=ContentTier.B
        )

        assert isinstance(result, PublishedContent)
        assert result.staged_path.is_dir()
        # Multi-file artefact layout — the directory must contain at
        # minimum manifest.bin, keyshares.json, and one shard file.
        names = {p.name for p in result.staged_path.iterdir()}
        assert "manifest.bin" in names
        assert "keyshares.json" in names
        assert any(n.startswith("shard-") for n in names)

        plaintext = await ret.fetch(result.torrent_infohash)
        assert plaintext == data

    @pytest.mark.asyncio
    async def test_publish_tier_c_round_trip(
        self,
        provider: _FakeBitTorrentProvider,
        staging_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Tier C round-trip — ContentStore picks erasure mode based on size."""
        from prsm.storage import ContentStore

        store = ContentStore(data_dir=str(tmp_path / "store"), node_id="test-node")
        pub = ContentPublisher(
            bt_provider=provider, staging_dir=staging_dir, content_store=store
        )
        requester = _FakeBitTorrentRequester(provider=provider)
        ret = ContentRetriever(
            bt_requester=requester,
            cache_dir=tmp_path / "cache",
            content_store=store,
        )

        # Larger payload to encourage multi-shard layout.
        data = b"tier-C erasure-coded payload " * 1000
        result = await pub.publish(
            data, provenance_id="0xprov-C", tier=ContentTier.C
        )

        assert result.staged_path.is_dir()
        plaintext = await ret.fetch(result.torrent_infohash)
        assert plaintext == data

    @pytest.mark.asyncio
    async def test_publish_tier_b_without_content_store_raises(
        self,
        provider: _FakeBitTorrentProvider,
        staging_dir: Path,
    ) -> None:
        """Without ContentStore (and no global singleton), Tier B publish raises."""
        from prsm.storage import close_content_store

        close_content_store()  # ensure no global is hanging around
        pub = ContentPublisher(bt_provider=provider, staging_dir=staging_dir)
        with pytest.raises(RuntimeError, match="ContentStore"):
            await pub.publish(b"x", provenance_id="0xp", tier=ContentTier.B)

    @pytest.mark.asyncio
    async def test_publish_seed_failure_cleans_up_staged_file(
        self,
        publisher: ContentPublisher,
        provider: _FakeBitTorrentProvider,
        staging_dir: Path,
    ) -> None:
        # Sprint 495 (F37) reversed the old "preserve for retry" behaviour:
        # under sustained BT seed failure, preserved staged files accumulated
        # (a 60s soak leaked 36,337 files). publish() now UNLINKS the staged
        # file on a seed failure (raise OR None-return) to bound staging-dir
        # growth — see prsm/node/content_publisher.py:317-343 + the
        # authoritative test_sprint_495_f37_staging_cleanup.py. (This test was
        # stale: it still asserted the pre-F37 preserve behaviour.)
        provider.next_return = None  # default behaviour
        # Override provider to return None (simulating seed failure).

        class _FailingProvider(_FakeBitTorrentProvider):
            async def seed_content(self, *args: Any, **kwargs: Any) -> None:
                self.calls.append(kwargs)
                return None

        failing = _FailingProvider()
        pub = ContentPublisher(bt_provider=failing, staging_dir=staging_dir)
        data = b"will fail to seed"

        with pytest.raises(RuntimeError, match="seed_content returned None"):
            await pub.publish(data, provenance_id="0xp")

        # F37: the staged file is cleaned up on seed failure (not leaked).
        staged = staging_dir / hashlib.sha256(data).hexdigest()
        assert not staged.exists()

    @pytest.mark.asyncio
    async def test_publish_custom_name_passed_through(
        self, publisher: ContentPublisher, provider: _FakeBitTorrentProvider
    ) -> None:
        await publisher.publish(
            b"x", provenance_id="0xp", name="dataset-v1"
        )
        assert provider.calls[0]["name"] == "dataset-v1"


# ---------------------------------------------------------------------------
# ContentRetriever
# ---------------------------------------------------------------------------


class TestContentRetriever:
    @pytest.fixture
    def cache_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "cache"

    @pytest.fixture
    def requester(self) -> _FakeBitTorrentRequester:
        return _FakeBitTorrentRequester()

    @pytest.fixture
    def retriever(
        self, requester: _FakeBitTorrentRequester, cache_dir: Path
    ) -> ContentRetriever:
        return ContentRetriever(bt_requester=requester, cache_dir=cache_dir)

    @pytest.mark.asyncio
    async def test_fetch_returns_payload(
        self,
        retriever: ContentRetriever,
        requester: _FakeBitTorrentRequester,
    ) -> None:
        infohash = "deadbeef"
        requester.payload_for[infohash] = b"the bytes we asked for"

        result = await retriever.fetch(infohash)

        assert result == b"the bytes we asked for"
        assert len(requester.calls) == 1
        assert requester.calls[0]["infohash"] == infohash

    @pytest.mark.asyncio
    async def test_fetch_propagates_timeout(
        self,
        retriever: ContentRetriever,
        requester: _FakeBitTorrentRequester,
    ) -> None:
        requester.payload_for["x"] = b""
        await retriever.fetch("x", timeout=12.5)
        assert requester.calls[0]["timeout"] == 12.5

    @pytest.mark.asyncio
    async def test_fetch_failure_raises(
        self,
        retriever: ContentRetriever,
        requester: _FakeBitTorrentRequester,
    ) -> None:
        requester.next_result = _FakeDownloadResult(
            success=False, error="peer timeout"
        )
        with pytest.raises(RuntimeError, match="peer timeout"):
            await retriever.fetch("nope")

    @pytest.mark.asyncio
    async def test_fetch_empty_dir_raises(
        self,
        retriever: ContentRetriever,
        requester: _FakeBitTorrentRequester,
    ) -> None:
        # success=True but the requester writes no files.
        class _EmptyRequester(_FakeBitTorrentRequester):
            async def request_content(self, *args: Any, **kwargs: Any):
                save_path = Path(kwargs.get("save_path") or args[1])
                save_path.mkdir(parents=True, exist_ok=True)
                return _FakeDownloadResult(success=True)

        empty = _EmptyRequester()
        ret = ContentRetriever(bt_requester=empty, cache_dir=retriever.cache_dir)
        with pytest.raises(FileNotFoundError):
            await ret.fetch("empty-result")

    @pytest.mark.asyncio
    async def test_fetch_tier_bc_missing_manifest_raises(
        self,
        retriever: ContentRetriever,
        requester: _FakeBitTorrentRequester,
    ) -> None:
        """Multi-file torrents without manifest.bin / keyshares.json are
        treated as malformed Tier-B/C drops and surface a clear error."""
        requester.payload_for["malformed"] = b"primary"
        # extra_files>0 produces multi-file layout but without the
        # Tier-B/C-specific manifest.bin / keyshares.json artefacts.
        requester.extra_files = 2
        with pytest.raises(RuntimeError, match="manifest.bin|keyshares.json"):
            await retriever.fetch("malformed")
