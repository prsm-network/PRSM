"""
ContentStore — public API facade for the PRSM native storage module.

Ties together BlobStore (local file I/O), ShardEngine (split/reassemble),
and KeyManager (AES-256-GCM + Shamir) into a single high-level interface.

Usage::

    store = ContentStore(data_dir="/var/prsm/storage", node_id="node-1")
    content_hash = await store.store_local(raw_bytes, replication_factor=3)
    data = await store.retrieve_local(content_hash)
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# sp920 — default ceiling on the in-memory manifest cache (LRU). Each entry is
# ciphertext + Shamir key-shares + a manifest (~few KB); 4096 ≈ low-MB. Tunable
# via PRSM_MANIFEST_CACHE_MAX. Eviction is safe: a miss falls back to the seeded
# BitTorrent torrent (artefacts persist on disk).
_DEFAULT_MANIFEST_CACHE_MAX = 4096

from prsm.storage.blob_store import BlobStore
from prsm.storage.exceptions import ContentNotFoundError
from prsm.storage.key_manager import KeyManager
from prsm.storage.models import ContentHash, KeyShare, ShardManifest
from prsm.storage.shard_engine import ShardEngine


# ---------------------------------------------------------------------------
# StorageArtifacts (returned by store_local_with_artifacts / consumed by
# retrieve_with_artifacts) — used by the publish/fetch layer to wire
# encrypted Tier-B/C content through external transport (BitTorrent).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StorageArtifacts:
    """All on-disk + in-memory artefacts produced by an encrypted store.

    Used by external distribution layers (e.g. BitTorrent) that need to
    package the full set of bytes required to later reassemble + decrypt
    the content on a different node.

    Attributes
    ----------
    content_hash:
        Content-addressed identifier of the *original* (plaintext) bytes.
    encrypted_manifest:
        AES-256-GCM ciphertext of the serialised :class:`ShardManifest`.
        Distributed alongside the shards so a recipient can recover the
        manifest *without* querying the origin node.
    key_shares:
        Shamir secret-shared encryption-key shares. The recipient needs
        ``threshold`` of these to reconstruct the AES key.
    shard_paths:
        Absolute filesystem paths of every shard blob written to the
        local BlobStore. The publisher is responsible for staging /
        copying these into a torrent directory (the ContentStore does not
        own the BitTorrent staging area).
    manifest:
        The plaintext :class:`ShardManifest` — kept for the publisher's
        bookkeeping (e.g. inspecting shard order). Should NOT be
        distributed alongside the encrypted manifest, since doing so
        would defeat the encryption.
    """
    content_hash: ContentHash
    encrypted_manifest: bytes
    key_shares: List[KeyShare]
    shard_paths: List[str]
    manifest: ShardManifest


# ---------------------------------------------------------------------------
# Threshold policy helpers
# ---------------------------------------------------------------------------

def _select_threshold_params(num_shards: int) -> Tuple[int, int]:
    """Return (threshold, num_shares) based on the number of shards.

    Policy:
        <10 shards  -> 3-of-5
        10-99 shards -> 5-of-8
        100+ shards -> 7-of-12
    """
    if num_shards < 10:
        return 3, 5
    elif num_shards < 100:
        return 5, 8
    else:
        return 7, 12


# ---------------------------------------------------------------------------
# ContentStore
# ---------------------------------------------------------------------------

class ContentStore:
    """
    High-level content storage facade.

    Orchestrates sharding, encryption, and manifest management for locally
    stored content.  Network transport and discovery are accepted as optional
    parameters for future use but are not exercised in the local-only methods.

    Parameters
    ----------
    data_dir:
        Root directory used by the underlying BlobStore.
    node_id:
        Logical identifier for this storage node (used in manifest metadata).
    shard_threshold:
        Content <= this many bytes is stored as a single shard.
        Default: 1 MiB (1_048_576 bytes).
    shard_size:
        Target shard size in bytes for content that exceeds *shard_threshold*.
        Default: 256 KiB (262_144 bytes).
    transport:
        Optional network transport handle (reserved for future P2P use).
    discovery:
        Optional peer-discovery handle (reserved for future P2P use).
    """

    def __init__(
        self,
        data_dir: str,
        node_id: str = "",
        shard_threshold: int = 1_048_576,
        shard_size: int = 262_144,
        transport: Optional[Any] = None,
        discovery: Optional[Any] = None,
        manifest_cache_max: Optional[int] = None,
    ) -> None:
        self.data_dir = data_dir
        self.node_id = node_id
        self.transport = transport
        self.discovery = discovery

        self.blob_store = BlobStore(data_dir=data_dir)
        self.shard_engine = ShardEngine(
            blob_store=self.blob_store,
            shard_threshold=shard_threshold,
            shard_size=shard_size,
        )
        self.key_manager = KeyManager()

        # sp920 — LRU-bounded so repeated publishes can't grow it to OOM.
        if manifest_cache_max is None:
            try:
                manifest_cache_max = int(os.environ.get(
                    "PRSM_MANIFEST_CACHE_MAX", _DEFAULT_MANIFEST_CACHE_MAX,
                ))
            except (TypeError, ValueError):
                manifest_cache_max = _DEFAULT_MANIFEST_CACHE_MAX
        self._manifest_cache_max = max(1, int(manifest_cache_max))

        # content_hash hex -> (ciphertext, key_shares, manifest). OrderedDict so
        # _cache_put/_cache_get can evict the least-recently-used entry.
        self._manifest_cache: "OrderedDict[str, Tuple[bytes, List[KeyShare], ShardManifest]]" = OrderedDict()

    def _cache_put(self, key: str, value) -> None:
        """sp920 — insert into the manifest cache, marking it most-recently-used
        and evicting the least-recently-used entry once over the bound. Eviction
        is safe: a later miss falls back to the seeded BitTorrent torrent."""
        self._manifest_cache[key] = value
        self._manifest_cache.move_to_end(key)
        while len(self._manifest_cache) > self._manifest_cache_max:
            self._manifest_cache.popitem(last=False)

    def _cache_get(self, key: str):
        """sp920 — fetch + mark most-recently-used (LRU touch). None if absent."""
        if key not in self._manifest_cache:
            return None
        self._manifest_cache.move_to_end(key)
        return self._manifest_cache[key]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _store_and_encrypt(
        self,
        data: bytes,
        owner_node_id: str,
        replication_factor: int,
        visibility: str = "public",
    ) -> Tuple[ContentHash, ShardManifest, bytes, List[KeyShare]]:
        """Shard *data*, encrypt the manifest, and return all artefacts.

        Returns
        -------
        (content_hash, manifest, ciphertext, key_shares)
            *content_hash*  — content-addressed identifier of the original data.
            *manifest*      — ShardManifest describing the stored shards.
            *ciphertext*    — AES-256-GCM encrypted serialised manifest.
            *key_shares*    — Shamir key shares for the encryption key.
        """
        manifest = await self.shard_engine.split(
            data,
            owner_node_id=owner_node_id,
            replication_factor=replication_factor,
            visibility=visibility,
        )
        content_hash = manifest.content_hash

        # Determine threshold parameters from number of shards.
        num_shards = len(manifest.shard_hashes)
        threshold, num_shares = _select_threshold_params(num_shards)

        manifest_bytes = manifest.to_json().encode("utf-8")
        ciphertext, key_shares = self.key_manager.encrypt_manifest(
            manifest_bytes,
            content_hash,
            threshold=threshold,
            num_shares=num_shares,
        )

        return content_hash, manifest, ciphertext, key_shares

    def _decode_manifest(self, ciphertext: bytes, key_shares: List[KeyShare]) -> ShardManifest:
        """Reconstruct key from *key_shares* and decrypt the manifest ciphertext."""
        manifest_bytes = self.key_manager.decrypt_manifest(ciphertext, key_shares)
        return ShardManifest.from_json(manifest_bytes.decode("utf-8"))

    # ------------------------------------------------------------------
    # Public local API
    # ------------------------------------------------------------------

    async def store_local(
        self,
        data: bytes,
        replication_factor: int = 3,
    ) -> ContentHash:
        """Store *data* locally and cache the encrypted manifest in memory.

        Shards are written to the BlobStore; the manifest is encrypted and
        kept in :attr:`_manifest_cache` keyed by the content hash hex string.

        Parameters
        ----------
        data:
            Raw bytes to store.
        replication_factor:
            Desired network replication factor embedded in the manifest metadata.

        Returns
        -------
        ContentHash
            Content-addressed identifier for the stored data.
        """
        content_hash, manifest, ciphertext, key_shares = await self._store_and_encrypt(
            data,
            owner_node_id=self.node_id,
            replication_factor=replication_factor,
        )
        self._cache_put(content_hash.hex(), (ciphertext, key_shares, manifest))
        return content_hash

    async def retrieve_local(self, content_hash: ContentHash) -> bytes:
        """Retrieve content by *content_hash* from the local store.

        Uses the cached manifest to locate and reassemble shards.

        Raises
        ------
        ContentNotFoundError
            If the content hash is not present in :attr:`_manifest_cache`.
        """
        cache_key = content_hash.hex()
        entry = self._cache_get(cache_key)   # sp920 — LRU touch on use
        if entry is None:
            raise ContentNotFoundError(cache_key)

        ciphertext, key_shares, manifest = entry
        return await self.shard_engine.reassemble(manifest)

    async def exists_local(self, content_hash: ContentHash) -> bool:
        """Return ``True`` if *content_hash* is present in the manifest cache."""
        return content_hash.hex() in self._manifest_cache

    async def size_local(self, content_hash: ContentHash) -> int:
        """Return the total content size in bytes for *content_hash*.

        Reads the cached manifest's ``total_size`` field (set when the
        content was sharded and stored). Returns 0 if the content is not
        present in the local manifest cache — callers that care about the
        distinction between "absent" and "empty" should first check
        :meth:`exists_local`.
        """
        cache_key = content_hash.hex()
        entry = self._manifest_cache.get(cache_key)
        if entry is None:
            return 0
        _ciphertext, _key_shares, manifest = entry
        return manifest.total_size

    # ------------------------------------------------------------------
    # Artifact-aware API (used by ContentPublisher / ContentRetriever for
    # cross-node distribution via BitTorrent). The artifact-aware path
    # exposes the same encrypt-then-shard pipeline as :meth:`store_local`
    # but returns every byte the caller needs to reassemble + decrypt on
    # a different node — bypassing :attr:`_manifest_cache` (which is
    # in-memory only, single-node).
    # ------------------------------------------------------------------

    async def store_local_with_artifacts(
        self,
        data: bytes,
        replication_factor: int = 3,
    ) -> StorageArtifacts:
        """Store *data* and return every artefact a remote node would need.

        Same encrypt-then-shard pipeline as :meth:`store_local`, but
        returns the encrypted manifest, key shares, and shard paths
        instead of caching them in :attr:`_manifest_cache`.

        Used by :class:`prsm.node.content_publisher.ContentPublisher` for
        Tier-B/C publishing, where the artefacts are staged into a
        BitTorrent directory and seeded to the network.

        The shards are also written to the local BlobStore as a
        side-effect so the publisher's own node can serve fetch requests
        without re-downloading via BitTorrent.
        """
        content_hash, manifest, ciphertext, key_shares = await self._store_and_encrypt(
            data,
            owner_node_id=self.node_id,
            replication_factor=replication_factor,
        )
        # Cache the manifest so subsequent local retrievals on this node
        # skip the BlobStore round-trip — keeps store_local + this method
        # observationally interchangeable from the local-fetch side. sp920 —
        # LRU-bounded.
        self._cache_put(content_hash.hex(), (ciphertext, key_shares, manifest))

        shard_paths = [
            self.blob_store._path_for(shard_hash)
            for shard_hash in manifest.shard_hashes
        ]

        return StorageArtifacts(
            content_hash=content_hash,
            encrypted_manifest=ciphertext,
            key_shares=key_shares,
            shard_paths=shard_paths,
            manifest=manifest,
        )

    async def retrieve_with_artifacts(
        self,
        encrypted_manifest: bytes,
        key_shares: List[KeyShare],
        shard_blobs: List[bytes],
    ) -> bytes:
        """Reassemble + decrypt content from a remotely-supplied artefact set.

        Inverse of :meth:`store_local_with_artifacts`. Used by
        :class:`prsm.node.content_publisher.ContentRetriever` after a
        BitTorrent fetch has materialised the artefact files locally.

        Parameters
        ----------
        encrypted_manifest:
            AES-256-GCM ciphertext from
            :attr:`StorageArtifacts.encrypted_manifest`.
        key_shares:
            Shamir key shares — at least :attr:`KeyShare.threshold` of them.
        shard_blobs:
            Raw shard bytes in the same order as the manifest's
            ``shard_hashes``. The caller is responsible for fetching the
            shards (e.g. by reading every shard-{N}.bin in the torrent
            directory) and supplying them in deterministic order.

        Returns
        -------
        bytes
            The original plaintext content.

        Raises
        ------
        KeyReconstructionError
            If fewer than ``threshold`` key shares are supplied or any
            share is corrupt.
        ManifestError
            If decrypting the manifest yields invalid JSON.
        ShardIntegrityError
            If any shard's recomputed hash does not match the manifest.
        """
        # Decrypt the manifest first — it tells us shard order, count,
        # and erasure parameters.
        manifest = self._decode_manifest(encrypted_manifest, key_shares)

        # Stage the shards back into the local BlobStore. ShardEngine
        # reads via `BlobStore.retrieve(shard_hash)`, so we need the
        # bytes addressable by their content hash on this node before
        # calling reassemble.
        if len(shard_blobs) != len(manifest.shard_hashes):
            from prsm.storage.exceptions import ShardIntegrityError
            raise ShardIntegrityError(
                f"Shard count mismatch: manifest expects "
                f"{len(manifest.shard_hashes)}, got {len(shard_blobs)}"
            )

        # BlobStore.store() is content-addressed: re-storing bytes whose
        # hash matches an existing blob is a no-op. We rely on that to
        # both (a) verify each shard's integrity (the stored hash must
        # match the manifest's expected hash) and (b) populate the
        # local blob store so reassemble() can read them back.
        for shard_blob, expected_hash in zip(shard_blobs, manifest.shard_hashes):
            stored_hash = await self.blob_store.store(shard_blob)
            if stored_hash != expected_hash:
                from prsm.storage.exceptions import ShardIntegrityError
                raise ShardIntegrityError(
                    f"Shard hash mismatch: expected {expected_hash.hex()}, "
                    f"got {stored_hash.hex()}"
                )

        return await self.shard_engine.reassemble(manifest)

    async def delete_local(self, content_hash: ContentHash) -> None:
        """Delete all shards and the manifest cache entry for *content_hash*.

        Shard blobs are removed from the BlobStore; the manifest cache entry
        is dropped.  No-op if *content_hash* is not present.
        """
        cache_key = content_hash.hex()
        if cache_key not in self._manifest_cache:
            return

        _ciphertext, _key_shares, manifest = self._manifest_cache.pop(cache_key)
        for shard_hash in manifest.shard_hashes:
            await self.blob_store.delete(shard_hash)
