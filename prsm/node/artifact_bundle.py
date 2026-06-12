"""Sprint 1074 — Tier B/C encrypted-content artifact bundle codec (libtorrent-free).

Tier B/C content (encrypted manifest + Shamir key shares + N encrypted shards) is
today distributed as a MULTI-FILE torrent — which needs libtorrent AND is broken on
libtorrent 2.0.12 (the create_torrent directory branch produces "missing name").
Because Tier B/C CIDs are per-publish (the AES key rotates, so the same plaintext
never yields the same CID), there is no content-dedup/compat to preserve. So the
libtorrent-free design bundles the artifacts into ONE canonical blob, identified by
the proven Tier-A single-file v1 infohash (sp1070) and served over the single-blob
P2P substrate (ContentProvider INLINE/CHUNKED) — making Tier B/C work end-to-end
without libtorrent.

Wire format (all integers big-endian u32):

    BUNDLE_MAGIC (8 bytes)
    manifest_len   | manifest bytes
    keyshares_len  | keyshares (JSON) bytes
    shard_count    | for each shard: shard_len | shard bytes

The magic header lets a retriever distinguish a bundled Tier-B/C blob from raw
Tier-A plaintext. (Key-share colocation matches today's behaviour — the proper
per-share distribution is the KeyDistribution.sol follow-on; this codec does not
change that trust property, it only changes the on-wire container.)
"""
from __future__ import annotations

import struct
from typing import List, Tuple

BUNDLE_MAGIC = b"PRSMBND1"
_U32 = struct.Struct(">I")
# sp1075 review HIGH-2 — bound parts COHERENTLY with the 64 MiB CHUNKED transport
# ceiling, not a meaningless 16 GiB above it (any blob reaching this parser already
# passed the transport cap). 128 MiB gives headroom without being dead protection.
_MAX_PART = 128 * 1024 * 1024
# sp1075 review MEDIUM-1 — keyshares JSON is KB-scale; cap it tightly so a malicious
# blob can't feed a 64 MiB JSON-bomb to json.loads downstream.
_MAX_KEYSHARES = 4 * 1024 * 1024
# sp1075 review HIGH-1 — a real bundle has O(10s) of shards (replication × erasure);
# an unbounded shard_count + empty shards is a CPU/memory amplification DoS.
_MAX_SHARDS = 4096


class ArtifactBundleError(ValueError):
    """The blob is not a well-formed PRSM artifact bundle (bad magic / truncated /
    inconsistent length). Raised so a caller fails closed rather than decrypting
    garbage."""


def is_bundle(blob: bytes) -> bool:
    """True iff ``blob`` is a structurally-plausible Tier-B/C bundle. Cheap, total —
    never raises.

    sp1075 review MEDIUM-2 — requires the magic AND a readable first length field that
    fits within the blob, not just the 8-byte magic prefix. This narrows the
    false-positive window where Tier-A plaintext that merely BEGINS with the magic
    would be misrouted to decrypt and then silently lost. (A coincidental prefix that
    isn't a real bundle now falls through to plaintext.)"""
    if not isinstance(blob, (bytes, bytearray)):
        return False
    blob = bytes(blob)
    if blob[:len(BUNDLE_MAGIC)] != BUNDLE_MAGIC:
        return False
    off = len(BUNDLE_MAGIC)
    if len(blob) < off + 4:
        return False
    mlen = _U32.unpack(blob[off:off + 4])[0]
    # the manifest length must at least fit in what's left after its own prefix
    return mlen <= len(blob) - (off + 4)


def bundle_artifacts(
    encrypted_manifest: bytes, keyshares_json: bytes, shard_blobs: List[bytes],
) -> bytes:
    """Encode the Tier-B/C artifacts into one canonical, reversible blob."""
    out = bytearray(BUNDLE_MAGIC)
    out += _U32.pack(len(encrypted_manifest))
    out += encrypted_manifest
    out += _U32.pack(len(keyshares_json))
    out += keyshares_json
    out += _U32.pack(len(shard_blobs))
    for shard in shard_blobs:
        out += _U32.pack(len(shard))
        out += bytes(shard)
    return bytes(out)


def _read(blob: bytes, off: int, n: int) -> Tuple[bytes, int]:
    end = off + n
    if n < 0 or end > len(blob):
        raise ArtifactBundleError(
            f"artifact bundle truncated: need {n} bytes at offset {off}, have "
            f"{len(blob) - off}")
    return blob[off:end], end


def _read_u32(blob: bytes, off: int) -> Tuple[int, int]:
    raw, off = _read(blob, off, 4)
    val = _U32.unpack(raw)[0]
    if val > _MAX_PART:
        raise ArtifactBundleError(f"artifact bundle part length {val} exceeds bound")
    return val, off


def unbundle_artifacts(blob: bytes) -> Tuple[bytes, bytes, List[bytes]]:
    """Decode a bundle back into ``(encrypted_manifest, keyshares_json,
    shard_blobs)``. Raises ``ArtifactBundleError`` on bad magic / truncation /
    inconsistent lengths (fail closed)."""
    if not is_bundle(blob):
        raise ArtifactBundleError("not a PRSM artifact bundle (bad/absent magic)")
    off = len(BUNDLE_MAGIC)
    mlen, off = _read_u32(blob, off)
    manifest, off = _read(blob, off, mlen)
    klen, off = _read_u32(blob, off)
    if klen > _MAX_KEYSHARES:   # review MEDIUM-1 — no 64 MiB JSON-bomb into json.loads
        raise ArtifactBundleError(
            f"keyshares part {klen} exceeds bound {_MAX_KEYSHARES}")
    keyshares, off = _read(blob, off, klen)
    count, off = _read_u32(blob, off)
    # review HIGH-1 — bound the shard count BEFORE the loop. _MAX_SHARDS is the
    # principled cap; the remaining-bytes check kills the empty-shard amplification
    # (each shard needs ≥4 bytes of length prefix, so count can't exceed remaining/4).
    if count > _MAX_SHARDS:
        raise ArtifactBundleError(f"shard_count {count} exceeds bound {_MAX_SHARDS}")
    if count * 4 > len(blob) - off:
        raise ArtifactBundleError(
            f"shard_count {count} exceeds remaining bytes (amplification)")
    shards: List[bytes] = []
    for _ in range(count):
        slen, off = _read_u32(blob, off)
        shard, off = _read(blob, off, slen)
        shards.append(shard)
    if off != len(blob):
        raise ArtifactBundleError(
            f"artifact bundle has {len(blob) - off} trailing bytes (corrupt)")
    return manifest, keyshares, shards
