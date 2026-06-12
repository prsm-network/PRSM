"""Sprint 1074 — Tier B/C artifact bundle codec (libtorrent-free encrypted content, brick 1).

Tier B/C (encrypted) content is today a MULTI-FILE torrent (manifest.bin +
keyshares.json + shard-NNNN.bin), which (a) needs libtorrent and (b) is in fact
BROKEN on libtorrent 2.0.12 (the create_torrent directory branch adds bare relative
paths with no root name → "missing name"). Since Tier B/C CIDs are per-publish anyway
(the AES key rotates), there's no dedup/compat to preserve, so the clean libtorrent-
free design bundles the artifacts into ONE canonical blob, identified by the proven
Tier-A single-file infohash and served over the single-blob P2P substrate.

This brick is the reversible codec: (encrypted_manifest, keyshares_json, shard_blobs)
<-> one blob. A magic header lets the retriever detect a bundle vs Tier-A plaintext.
"""
from __future__ import annotations

import os

import pytest

from prsm.node.artifact_bundle import (
    BUNDLE_MAGIC,
    ArtifactBundleError,
    bundle_artifacts,
    is_bundle,
    unbundle_artifacts,
)


def test_roundtrip():
    manifest = os.urandom(4000)
    keyshares = b'[{"share_index":0,"share_data":"ab"}]'
    shards = [os.urandom(1000), os.urandom(262144 + 7), b""]
    blob = bundle_artifacts(manifest, keyshares, shards)
    m2, k2, s2 = unbundle_artifacts(blob)
    assert m2 == manifest
    assert k2 == keyshares
    assert s2 == shards


def test_zero_shards_roundtrip():
    blob = bundle_artifacts(b"manifest", b"[]", [])
    m, k, s = unbundle_artifacts(blob)
    assert m == b"manifest" and k == b"[]" and s == []


def test_is_bundle_detects_magic():
    blob = bundle_artifacts(b"m", b"k", [b"s"])
    assert is_bundle(blob) is True
    assert blob.startswith(BUNDLE_MAGIC)


def test_is_bundle_false_for_plaintext():
    assert is_bundle(b"just some plaintext tier-A content") is False
    assert is_bundle(b"") is False
    assert is_bundle(BUNDLE_MAGIC[:-1]) is False   # too short / partial magic


def test_deterministic():
    """Same artifacts → same bundle bytes (so the CID is stable for a given encrypt)."""
    a = bundle_artifacts(b"m", b"k", [b"s1", b"s2"])
    b = bundle_artifacts(b"m", b"k", [b"s1", b"s2"])
    assert a == b


def test_truncated_blob_raises():
    blob = bundle_artifacts(b"manifest", b"keys", [os.urandom(500)])
    with pytest.raises(ArtifactBundleError):
        unbundle_artifacts(blob[:20])   # cut mid-structure


def test_wrong_magic_raises():
    with pytest.raises(ArtifactBundleError):
        unbundle_artifacts(b"NOTABUND" + b"\x00" * 20)


def test_large_shard_count():
    shards = [bytes([i % 256]) * (i + 1) for i in range(50)]
    blob = bundle_artifacts(b"m", b"k", shards)
    _, _, s = unbundle_artifacts(blob)
    assert s == shards


# ── sp1075 review hardening (untrusted-blob DoS / routing) ───────────────────────

import struct as _struct  # noqa: E402


def test_huge_shard_count_rejected():
    """HIGH-1 — an absurd shard_count is rejected before the loop (no CPU burn)."""
    blob = (BUNDLE_MAGIC + _struct.pack(">I", 0) + _struct.pack(">I", 0)
            + _struct.pack(">I", 0xFFFFFFFF))
    with pytest.raises(ArtifactBundleError):
        unbundle_artifacts(blob)


def test_empty_shard_amplification_rejected():
    """HIGH-1 — count claims more shards than the remaining bytes can hold."""
    blob = (BUNDLE_MAGIC + _struct.pack(">I", 0) + _struct.pack(">I", 0)
            + _struct.pack(">I", 1000))   # 1000 shards but zero bytes follow
    with pytest.raises(ArtifactBundleError):
        unbundle_artifacts(blob)


def test_huge_keyshares_rejected():
    """MEDIUM-1 — an oversized keyshares part is rejected before json.loads."""
    blob = BUNDLE_MAGIC + _struct.pack(">I", 0) + _struct.pack(">I", 5 * 1024 * 1024)
    with pytest.raises(ArtifactBundleError):
        unbundle_artifacts(blob)


def test_is_bundle_magic_only_is_false():
    """MEDIUM-2 — magic alone (or a non-fitting header) is NOT a bundle, so Tier-A
    plaintext that merely starts with the magic falls through to plaintext."""
    assert is_bundle(BUNDLE_MAGIC) is False                       # no length field
    assert is_bundle(BUNDLE_MAGIC + _struct.pack(">I", 9999)) is False  # mlen doesn't fit
    # a real bundle still detects as one
    assert is_bundle(bundle_artifacts(b"m", b"k", [b"s"])) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
