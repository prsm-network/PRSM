"""Sprint 1071 — LocalContentPublisher: libtorrent-free Tier-A publishing (brick 2).

A drop-in for ContentPublisher.publish() that needs NO libtorrent: it stages the
bytes content-addressed (same sha256 filename the BT publisher uses), computes the
canonical v1 infohash in pure Python (sp1070), and returns a PublishedContent whose
``torrent_infohash`` is BYTE-IDENTICAL to what a libtorrent (v1_only) node would
compute — so a default operator can publish and the CID matches the network. The
existing ContentUploader → ContentProvider registration then serves it over the
sp1020 P2P substrate; cross-node fetch uses ContentProvider.request_content (also
libtorrent-free). Tier B/C (encrypted multi-file) still needs the BT/ContentStore
path — out of scope for this brick (raises a clear error).
"""
from __future__ import annotations

import hashlib

import pytest

from prsm.compute.inference.models import ContentTier
from prsm.node.local_content_publisher import LocalContentPublisher

try:
    import libtorrent as lt
    HAS_LT = True
except ImportError:  # pragma: no cover
    HAS_LT = False


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def _lt_v1_infohash(data: bytes, name: str, piece_length: int = 262144) -> str:
    import os
    import tempfile
    d = tempfile.mkdtemp()
    fp = os.path.join(d, name)
    open(fp, "wb").write(data)
    fs = lt.file_storage()
    fs.add_file(name, len(data))
    t = lt.create_torrent(fs, piece_size=piece_length, flags=lt.create_torrent.v1_only)
    lt.set_piece_hashes(t, d)
    ih = str(lt.torrent_info(lt.bdecode(lt.bencode(t.generate()))).info_hash())
    os.remove(fp)
    os.rmdir(d)
    return ih


def test_publish_returns_pure_python_infohash(tmp_path):
    pub = LocalContentPublisher(staging_dir=tmp_path, node_id="node-1")
    data = b"hello prsm public content"
    out = _run(pub.publish(data, provenance_id="0xabc"))
    from prsm.core.torrent_infohash import compute_v1_infohash_single_file
    expected = compute_v1_infohash_single_file(data, hashlib.sha256(data).hexdigest())
    assert out.torrent_infohash == expected
    assert out.staged_path.exists() and out.staged_path.read_bytes() == data
    assert out.manifest.infohash == expected
    assert out.manifest.total_size == len(data)
    assert out.manifest.provenance_id == "0xabc"


@pytest.mark.skipif(not HAS_LT, reason="libtorrent not installed")
@pytest.mark.parametrize("size", [50, 262144, 262144 * 2 + 99])
def test_published_cid_matches_libtorrent_node(tmp_path, size):
    """The decisive interop: a libtorrent node and this no-libtorrent publisher must
    assign the SAME CID to the same bytes (no network fragmentation)."""
    import os
    pub = LocalContentPublisher(staging_dir=tmp_path)
    data = os.urandom(size)
    out = _run(pub.publish(data, provenance_id=""))
    assert out.torrent_infohash == _lt_v1_infohash(data, hashlib.sha256(data).hexdigest())


def test_idempotent_restage(tmp_path):
    pub = LocalContentPublisher(staging_dir=tmp_path)
    data = b"same bytes"
    a = _run(pub.publish(data, provenance_id=""))
    b = _run(pub.publish(data, provenance_id=""))
    assert a.torrent_infohash == b.torrent_infohash
    assert a.staged_path == b.staged_path


def test_local_publish_path_shortcut(tmp_path):
    """ContentRetriever's F8 shortcut calls local_publish_path(infohash)."""
    pub = LocalContentPublisher(staging_dir=tmp_path)
    out = _run(pub.publish(b"x" * 100, provenance_id=""))
    assert pub.local_publish_path(out.torrent_infohash) == out.staged_path
    assert pub.local_publish_path("0" * 40) is None


def test_manifest_pieces_cover_content(tmp_path):
    pub = LocalContentPublisher(staging_dir=tmp_path)
    data = b"y" * (262144 * 2 + 10)   # 3 pieces
    out = _run(pub.publish(data, provenance_id=""))
    assert len(out.manifest.pieces) == 3
    assert sum(p.size for p in out.manifest.pieces) == len(data)
    assert out.manifest.pieces[0].index == 0


def test_tier_bc_raises_clear_error(tmp_path):
    pub = LocalContentPublisher(staging_dir=tmp_path)
    with pytest.raises(NotImplementedError):
        _run(pub.publish(b"secret", provenance_id="", tier=ContentTier.B))


@pytest.mark.skipif(not HAS_LT, reason="libtorrent not installed")
@pytest.mark.parametrize("size", [50, 262144, 262144 * 3 + 777])
def test_real_bittorrent_client_path_matches_pure_python(tmp_path, size):
    """AUTHORITATIVE interop golden: the REAL BitTorrentClient.create_torrent path
    (v1_only + basename, sp1071) must assign the SAME CID as the pure-Python
    LocalContentPublisher for the same bytes. The brick-1/2 helpers used a simplified
    libtorrent call; this exercises the actual production code path to catch any
    divergence (it caught the full-path-vs-basename bug)."""
    import asyncio
    import os
    from prsm.core.bittorrent_client import BitTorrentClient
    from prsm.core.torrent_infohash import compute_v1_infohash_single_file

    data = os.urandom(size)
    name = hashlib.sha256(data).hexdigest()
    staged = tmp_path / name
    staged.write_bytes(data)

    async def _bt():
        c = BitTorrentClient()
        await c.initialize()
        res = await c.create_torrent(path=str(staged))
        return res.infohash

    bt_ih = asyncio.run(_bt())
    assert bt_ih == compute_v1_infohash_single_file(data, name)


def test_no_libtorrent_import():
    """The module must not require libtorrent (that's the whole point)."""
    import prsm.node.local_content_publisher as mod
    import inspect
    src = inspect.getsource(mod)
    assert "import libtorrent" not in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
