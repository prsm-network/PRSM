"""Sprint 1070 — pure-Python BitTorrent v1 (BEP-3) infohash computation.

The PRSM content identifier (CID) is the BitTorrent infohash that
``content_publisher.py`` returns from ``ContentPublisher.publish`` (used as the
provenance ``cid`` + the on-chain ``RoyaltyDistributor`` ``contentHash``). Today it
is produced by libtorrent's ``create_torrent``, which makes libtorrent a hard
dependency of the upload path — a default operator without it can't publish at all.

This module computes the SAME canonical v1 infohash in pure Python so a
non-libtorrent node produces a byte-identical CID (no network migration). The v1
infohash is ``sha1(bencode(info))`` where ``info`` is the BEP-3 single-file dict
``{length, name, piece length, pieces}``. This matches libtorrent EXACTLY when
libtorrent is asked for a v1-only torrent (``create_torrent.v1_only``) — verified
byte-for-byte by the sprint-1070 golden tests against the real library. (libtorrent
2.x defaults to HYBRID v1+v2 torrents whose info dict carries extra v2 fields, so the
libtorrent seed path must be pinned to v1_only for the two to agree — done in the
provider's seed call.)

Only ``bencodepy`` (already a PRSM dependency, used by ``bittorrent_manifest``) and
``hashlib`` are needed — no libtorrent, no native code.
"""
from __future__ import annotations

import hashlib
from typing import Dict

import bencodepy

# The canonical PRSM piece length (matches BitTorrentProvider.seed_content's default).
DEFAULT_PIECE_LENGTH = 262144  # 256 KiB


def _piece_hashes(data: bytes, piece_length: int) -> bytes:
    """Concatenated 20-byte SHA-1 of each ``piece_length`` slice (BEP-3 'pieces')."""
    if not data:
        return b""
    return b"".join(
        hashlib.sha1(data[i:i + piece_length]).digest()
        for i in range(0, len(data), piece_length)
    )


def build_single_file_info_dict(
    data: bytes, name: str, piece_length: int = DEFAULT_PIECE_LENGTH,
) -> Dict[bytes, object]:
    """Return the BEP-3 single-file ``info`` dict (the structure whose bencode is
    SHA-1'd to get the v1 infohash). Exposed so the publisher can also embed the
    canonical metadata; keys are bytes to match bencode's wire form."""
    if piece_length <= 0:
        raise ValueError(f"piece_length must be positive (got {piece_length})")
    return {
        b"length": len(data),
        b"name": name.encode("utf-8"),
        b"piece length": int(piece_length),
        b"pieces": _piece_hashes(data, piece_length),
    }


def compute_v1_infohash_single_file(
    data: bytes, name: str, piece_length: int = DEFAULT_PIECE_LENGTH,
) -> str:
    """The canonical v1 BitTorrent infohash (40-hex SHA-1) for a single-file torrent
    of ``data`` named ``name`` — byte-identical to libtorrent's ``v1_only`` output
    (sp1070 golden-verified). This is the PRSM CID a non-libtorrent publisher uses."""
    info = build_single_file_info_dict(data, name, piece_length)
    return hashlib.sha1(bencodepy.encode(info)).hexdigest()
