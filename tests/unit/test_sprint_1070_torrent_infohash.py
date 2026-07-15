"""Sprint 1070 — pure-Python BitTorrent v1 infohash (decouple content-identity from libtorrent).

The PRSM content identifier (CID) is the BitTorrent infohash (content_publisher.py:
``torrent_infohash`` → provenance + on-chain RoyaltyDistributor). Today that infohash is
computed by libtorrent's ``create_torrent``, so a default operator WITHOUT libtorrent can't
publish at all (content_publisher is None → upload fails). This brick computes the SAME
canonical v1 (BEP-3) infohash in pure Python (bencode + sha1), so a non-libtorrent node
produces a byte-identical CID — no network migration, no libtorrent dependency for the
identity.

The load-bearing property is byte-exact compatibility with libtorrent. These golden tests
assert equality against the REAL libtorrent (forced to v1_only, the simple BEP-3 info dict)
when it's installed; the pure path is also self-checked offline.
"""
from __future__ import annotations

import hashlib

import pytest

from prsm.core.torrent_infohash import compute_v1_infohash_single_file

try:
    import libtorrent as lt
    HAS_LT = True
except ImportError:  # pragma: no cover
    HAS_LT = False


def _lt_v1_infohash(data: bytes, name: str, piece_length: int = 262144) -> str:
    """Reference: libtorrent's v1_only single-file infohash (the canonical CID)."""
    import os
    import tempfile
    d = tempfile.mkdtemp()
    fp = os.path.join(d, name)
    with open(fp, "wb") as f:
        f.write(data)
    fs = lt.file_storage()
    fs.add_file(name, len(data))
    t = lt.create_torrent(fs, piece_size=piece_length, flags=lt.create_torrent.v1_only)
    lt.set_piece_hashes(t, d)
    tb = lt.bencode(t.generate())
    ih = str(lt.torrent_info(lt.bdecode(tb)).info_hash())
    os.remove(fp)
    os.rmdir(d)
    return ih


# ── pure-python self-consistency (no libtorrent needed) ─────────────────────────

def test_returns_40_hex_chars():
    ih = compute_v1_infohash_single_file(b"hello prsm", "name")
    assert len(ih) == 40 and all(c in "0123456789abcdef" for c in ih)


def test_deterministic():
    a = compute_v1_infohash_single_file(b"abc" * 1000, "f")
    b = compute_v1_infohash_single_file(b"abc" * 1000, "f")
    assert a == b


def test_different_content_different_infohash():
    a = compute_v1_infohash_single_file(b"one", "f")
    b = compute_v1_infohash_single_file(b"two", "f")
    assert a != b


def test_name_is_part_of_identity():
    """The torrent name is in the info dict, so it changes the infohash (matches BEP-3)."""
    a = compute_v1_infohash_single_file(b"same", "name-a")
    b = compute_v1_infohash_single_file(b"same", "name-b")
    assert a != b


# ── GOLDEN: byte-exact match with real libtorrent (v1_only) ─────────────────────

@pytest.mark.skipif(not HAS_LT, reason="libtorrent not installed")
@pytest.mark.parametrize("label,size", [
    ("small", 18),
    ("sub-piece", 100_000),
    ("exact-1-piece", 262144),
    ("just-over-1-piece", 262145),
    ("multi-piece", 262144 * 3 + 5000),
])
def test_matches_libtorrent_v1(label, size):
    import os
    data = os.urandom(size) if size else b""
    name = hashlib.sha256(data).hexdigest()   # PRSM stages Tier-A files under their sha256
    assert compute_v1_infohash_single_file(data, name) == _lt_v1_infohash(data, name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ── sp1457: streaming-from-path infohash is byte-identical to the in-memory version ──

import hashlib as _hashlib

from prsm.core.torrent_infohash import (
    compute_v1_infohash_single_file_from_path,
    DEFAULT_PIECE_LENGTH,
)


import pytest as _pytest


@_pytest.mark.parametrize("size", [
    0,                              # empty
    1,                              # tiny
    DEFAULT_PIECE_LENGTH - 1,       # just under one piece
    DEFAULT_PIECE_LENGTH,           # exactly one piece
    DEFAULT_PIECE_LENGTH + 1,       # just over → 2 pieces
    DEFAULT_PIECE_LENGTH * 3 + 7,   # several pieces, non-aligned tail
])
def test_streaming_infohash_matches_in_memory(tmp_path, size):
    data = bytes((i * 31 + 7) % 256 for i in range(size))
    f = tmp_path / "blob.bin"
    f.write_bytes(data)
    # Default name = sha256(bytes).hex() — the PRSM canonical single-file name, matching
    # content_provider._infohash_anchor_rejects.
    name = _hashlib.sha256(data).hexdigest()
    expected = compute_v1_infohash_single_file(data, name)
    assert compute_v1_infohash_single_file_from_path(f) == expected


def test_streaming_infohash_honors_explicit_name(tmp_path):
    data = b"named-torrent-content" * 1000
    f = tmp_path / "blob.bin"
    f.write_bytes(data)
    assert (compute_v1_infohash_single_file_from_path(f, name="dataset.parquet")
            == compute_v1_infohash_single_file(data, "dataset.parquet"))


def test_streaming_infohash_rejects_bad_piece_length(tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(b"x")
    with _pytest.raises(ValueError):
        compute_v1_infohash_single_file_from_path(f, piece_length=0)
