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

import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional

from prsm.compute.inference.models import ContentTier
from prsm.core.bittorrent_manifest import FileEntry, PieceInfo, TorrentManifest
from prsm.core.torrent_infohash import (
    DEFAULT_PIECE_LENGTH, compute_v1_infohash_single_file)
from prsm.node.content_publisher import PublishedContent

logger = logging.getLogger(__name__)


class LocalContentPublisher:
    """Pure-Python Tier-A publisher (no libtorrent). Drop-in for ContentPublisher."""

    def __init__(self, staging_dir, *, node_id: str = "",
                 piece_length: int = DEFAULT_PIECE_LENGTH) -> None:
        self.staging_dir = Path(staging_dir).expanduser()
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self._node_id = node_id
        self._piece_length = int(piece_length)
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
        if tier is not ContentTier.A:
            raise NotImplementedError(
                "LocalContentPublisher (no libtorrent) supports Tier A (public) "
                "only; Tier B/C encrypted publishing needs the BitTorrent/ContentStore "
                "path — install libtorrent and use ContentPublisher")

        # Content-addressed staging filename = sha256 hex (identical to
        # ContentPublisher._publish_tier_a, so the torrent NAME — and thus the
        # infohash — matches a libtorrent node's for the same bytes).
        staged_filename = hashlib.sha256(data).hexdigest()
        staged_path = self.staging_dir / staged_filename
        if not staged_path.exists():
            tmp_path = staged_path.with_suffix(".tmp")
            tmp_path.write_bytes(data)
            tmp_path.replace(staged_path)   # atomic — a reader never sees a partial file

        infohash = compute_v1_infohash_single_file(
            data, staged_filename, self._piece_length)
        manifest = self._build_manifest(data, staged_filename, infohash, provenance_id)
        self._published_paths[infohash] = staged_path

        logger.info(
            "Content published (tier=A, no libtorrent): infohash=%s name=%s size=%d "
            "provenance_id=%s", infohash, staged_filename, len(data), provenance_id)
        return PublishedContent(
            torrent_infohash=infohash, staged_path=staged_path, manifest=manifest)

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
