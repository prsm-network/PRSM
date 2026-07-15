"""Sprint 1071 — LocalContentPublisher: libtorrent-free Tier-A content publishing.

The PRSM upload path (``ContentUploader._publish_content``) requires a
``ContentPublisher`` that computes the content CID via libtorrent's
``create_torrent``. A default operator without libtorrent gets ``content_publisher
is None`` → upload fails. This class is a drop-in publisher that needs NO libtorrent:
it stages the bytes content-addressed and computes the SAME canonical v1 infohash in
pure Python (sp1070), so the CID is byte-identical to what a libtorrent (``v1_only``)
node assigns the same bytes — no network fragmentation.

It only computes identity + stages; serving + cross-node fetch already go through the
``ContentProvider`` P2P substrate (INLINE/CHUNKED, sp1020 — also libtorrent-free),
which ``ContentUploader`` populates after a successful publish. It mirrors the
``ContentPublisher`` surface the uploader/retriever use (``publish``,
``local_publish_path``, ``register_local_publish_tier_a``) so it's a drop-in.

Scope: Tier A (public single-file) only — the common default-operator case. Tier B/C
(encrypted multi-file, ContentStore + Shamir + multi-file torrent) still needs the
BitTorrent/ContentStore path and raises a clear error here.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional

from prsm.compute.inference.models import ContentTier
from prsm.core.bittorrent_manifest import FileEntry, PieceInfo, TorrentManifest
from prsm.core.torrent_infohash import (
    DEFAULT_PIECE_LENGTH, compute_v1_infohash_single_file,
    compute_v1_infohash_single_file_from_path)
from prsm.node.artifact_bundle import bundle_artifacts, unbundle_artifacts
from prsm.node.content_publisher import (
    PublishedContent, _deserialize_key_shares, _serialize_key_shares)

logger = logging.getLogger(__name__)


async def unbundle_and_decrypt(blob: bytes, content_store) -> bytes:
    """Inverse of LocalContentPublisher's Tier-B/C publish: unbundle the artifacts
    from a fetched single blob (sp1074) and decrypt via the ContentStore. The
    retrieval path calls this when ``artifact_bundle.is_bundle(blob)`` is True; a
    non-bundle (Tier-A plaintext) blob is returned to the caller as-is."""
    encrypted_manifest, keyshares_json, shard_blobs = unbundle_artifacts(blob)
    key_shares = _deserialize_key_shares(keyshares_json)
    return await content_store.retrieve_with_artifacts(
        encrypted_manifest=encrypted_manifest,
        key_shares=key_shares,
        shard_blobs=shard_blobs,
    )


class LocalContentPublisher:
    """Pure-Python Tier-A publisher (no libtorrent). Drop-in for ContentPublisher."""

    def __init__(self, staging_dir, *, node_id: str = "",
                 piece_length: int = DEFAULT_PIECE_LENGTH,
                 content_store=None) -> None:
        self.staging_dir = Path(staging_dir).expanduser()
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self._node_id = node_id
        self._piece_length = int(piece_length)
        self._content_store = content_store
        # infohash → staged Path, so ContentRetriever's F8 local shortcut works.
        self._published_paths: Dict[str, Path] = {}

    def local_publish_path(self, infohash: str) -> Optional[Path]:
        return self._published_paths.get(infohash)

    def register_local_publish_tier_a(self, infohash: str, content_hash: str) -> bool:
        """Re-register a staged Tier-A file (hydration after restart) — mirrors
        ContentPublisher. Returns False if the staged file is gone."""
        staged_path = self.staging_dir / content_hash
        if not staged_path.is_file():
            return False
        self._published_paths[infohash] = staged_path
        return True

    async def publish(
        self,
        data: bytes,
        *,
        provenance_id: str,
        tier: ContentTier = ContentTier.A,
        name: Optional[str] = None,
        replication_factor: int = 3,
    ) -> PublishedContent:
        """Stage ``data`` content-addressed and return its PublishedContent with the
        pure-Python v1 infohash as the CID. Tier A only."""
        if tier is ContentTier.A:
            # Content-addressed staging filename = sha256 hex (identical to
            # ContentPublisher._publish_tier_a, so the torrent NAME — and thus the
            # infohash — matches a libtorrent node's for the same bytes).
            blob = data
        else:
            # sp1074 — Tier B/C: encrypt via the ContentStore (no libtorrent), then
            # BUNDLE the artifacts into ONE blob distributed + identified exactly like
            # a Tier-A single file. Per-publish (AES key rotates), so the CID is the
            # bundle's sha256, not the plaintext's.
            blob = await self._encrypt_and_bundle(data, replication_factor)

        staged_filename = hashlib.sha256(blob).hexdigest()
        staged_path = self.staging_dir / staged_filename
        if not staged_path.exists():
            tmp_path = staged_path.with_suffix(".tmp")
            tmp_path.write_bytes(blob)
            tmp_path.replace(staged_path)   # atomic — a reader never sees a partial file

        infohash = compute_v1_infohash_single_file(
            blob, staged_filename, self._piece_length)
        manifest = self._build_manifest(blob, staged_filename, infohash, provenance_id)
        self._published_paths[infohash] = staged_path

        logger.info(
            "Content published (tier=%s, no libtorrent): infohash=%s name=%s size=%d "
            "provenance_id=%s", tier.value, infohash, staged_filename, len(blob),
            provenance_id)
        return PublishedContent(
            torrent_infohash=infohash, staged_path=staged_path, manifest=manifest)

    async def publish_from_path(
        self,
        src_path,
        *,
        provenance_id: str,
        tier: ContentTier = ContentTier.A,
        name: Optional[str] = None,
    ) -> PublishedContent:
        """sp1457 — STREAMING Tier-A publish of a file too large to hold in memory.

        The in-memory ``publish(data: bytes)`` materializes the whole blob (sha256 for the
        content-addressed staged name, the v1 infohash, and every manifest piece all iterate
        ``data``), so a >100 MiB dataset can't be published through it. This stages ``src_path``
        content-addressed by a STREAMING sha256 (block-copied, never fully read into memory) and
        computes the SAME canonical v1 infohash via the streaming primitive — byte-identical to
        ``publish(bytes)`` for the same content, with a bounded memory footprint. The staged file
        is registered so the ContentProvider streaming-send path (sp1290) serves it.

        Tier A (public single-file) only — Tier B/C encrypt-from-file is a follow-on (raises).
        Raises ``FileNotFoundError`` if ``src_path`` isn't a file."""
        if tier is not ContentTier.A:
            raise NotImplementedError(
                "publish_from_path supports Tier A only; Tier B/C large-file encrypt-from-path "
                "is a follow-on. Use publish(data) for encrypted content that fits in memory.")
        src = Path(src_path).expanduser()
        if not src.is_file():
            raise FileNotFoundError(f"publish_from_path: {src} is not a file")
        loop = asyncio.get_event_loop()

        def _stage() -> "tuple[str, Path]":
            # content-addressed name = streaming sha256 (matches publish(bytes)).
            h = hashlib.sha256()
            with open(src, "rb") as fh:
                for block in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(block)
            staged_filename = h.hexdigest()
            staged_path = self.staging_dir / staged_filename
            if not staged_path.exists():
                tmp_path = staged_path.with_suffix(".tmp")
                with open(src, "rb") as fin, open(tmp_path, "wb") as fout:
                    for block in iter(lambda: fin.read(1024 * 1024), b""):
                        fout.write(block)
                tmp_path.replace(staged_path)   # atomic — a reader never sees a partial file
            return staged_filename, staged_path

        staged_filename, staged_path = await loop.run_in_executor(None, _stage)
        infohash = await loop.run_in_executor(
            None, lambda: compute_v1_infohash_single_file_from_path(
                staged_path, staged_filename, self._piece_length))
        manifest = await loop.run_in_executor(
            None, lambda: self._build_manifest_from_path(
                staged_path, staged_filename, infohash, provenance_id))
        self._published_paths[infohash] = staged_path
        logger.info(
            "Content published (tier=A, streaming, no libtorrent): infohash=%s name=%s "
            "size=%d provenance_id=%s", infohash, staged_filename,
            manifest.total_size, provenance_id)
        return PublishedContent(
            torrent_infohash=infohash, staged_path=staged_path, manifest=manifest)

    def _build_manifest_from_path(self, path, name: str, infohash: str,
                                  provenance_id: str) -> TorrentManifest:
        """Streaming equivalent of ``_build_manifest`` — reads ``path`` in piece_length blocks
        so a multi-GiB file's manifest is built without materializing it."""
        pl = self._piece_length
        pieces = []
        total = 0
        with open(path, "rb") as fh:
            index = 0
            for chunk in iter(lambda: fh.read(pl), b""):
                pieces.append(PieceInfo(
                    index=index, hash=hashlib.sha1(chunk).hexdigest(), size=len(chunk)))
                total += len(chunk)
                index += 1
        return TorrentManifest(
            infohash=infohash, name=name, total_size=total, piece_length=pl, pieces=pieces,
            files=[FileEntry(path=name, size_bytes=total)],
            created_by_node_id=self._node_id, provenance_id=provenance_id or None)

    async def _encrypt_and_bundle(self, data: bytes, replication_factor: int) -> bytes:
        """Tier B/C: encrypt+shard via the ContentStore and bundle the artifacts into
        one canonical blob (sp1074). Reuses the EXACT ContentStore crypto the BT path
        uses (no crypto reimplementation)."""
        store = self._resolve_content_store()
        artefacts = await store.store_local_with_artifacts(
            data, replication_factor=replication_factor)
        keyshares_json = _serialize_key_shares(artefacts.key_shares)
        shard_blobs = [Path(p).read_bytes() for p in artefacts.shard_paths]
        return bundle_artifacts(
            artefacts.encrypted_manifest, keyshares_json, shard_blobs)

    def _resolve_content_store(self):
        if self._content_store is not None:
            return self._content_store
        from prsm.storage import get_content_store
        store = get_content_store()
        if store is None:
            raise RuntimeError(
                "LocalContentPublisher: Tier B/C publish requires a ContentStore. "
                "Initialise via prsm.storage.init_content_store() or pass "
                "content_store= to the constructor.")
        return store

    def _build_manifest(self, data: bytes, name: str, infohash: str,
                        provenance_id: str) -> TorrentManifest:
        pl = self._piece_length
        pieces = []
        if data:
            for index, off in enumerate(range(0, len(data), pl)):
                chunk = data[off:off + pl]
                pieces.append(PieceInfo(
                    index=index, hash=hashlib.sha1(chunk).hexdigest(), size=len(chunk)))
        return TorrentManifest(
            infohash=infohash,
            name=name,
            total_size=len(data),
            piece_length=pl,
            pieces=pieces,
            files=[FileEntry(path=name, size_bytes=len(data))],
            created_by_node_id=self._node_id,
            provenance_id=provenance_id or None,
        )
