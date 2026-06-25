"""Sprint 1269 — BitTorrent endpoints confine paths to a data root (audit round 5, HIGH/MED).

create_torrent/add accepted an arbitrary local content_path and seeded it to the PUBLIC
swarm (any authenticated user could exfiltrate the node's private keys, /etc/passwd, etc.);
add/start_download accepted an arbitrary save_path (arbitrary-path WRITE). Fix: _safe_bt_path
resolves the user path (FOLLOWING symlinks) and confines it to an allowed torrent data root
(node's <data_dir>/torrents, or PRSM_BT_DATA_ROOT), rejecting traversal / symlink escape with
HTTP 403.
"""
from __future__ import annotations

import os

import pytest
from fastapi import HTTPException

from prsm.interface.api.routers.bittorrent_router import _safe_bt_path


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / "torrents"
    r.mkdir()
    monkeypatch.setenv("PRSM_BT_DATA_ROOT", str(r))
    return r


def test_path_inside_root_allowed(root):
    f = root / "dataset.bin"
    f.write_text("x")
    assert _safe_bt_path(str(f), must_exist=True) == f.resolve()


def test_subdir_inside_root_allowed(root):
    sub = root / "a" / "b"
    sub.mkdir(parents=True)
    f = sub / "c.bin"
    f.write_text("x")
    assert _safe_bt_path(str(f), must_exist=True) == f.resolve()


def test_traversal_escape_blocked(root):
    with pytest.raises(HTTPException) as ei:
        _safe_bt_path(str(root / ".." / ".." / "etc" / "passwd"), must_exist=False)
    assert ei.value.status_code == 403


def test_absolute_outside_root_blocked(root):
    with pytest.raises(HTTPException) as ei:
        _safe_bt_path("/etc/passwd", must_exist=False)
    assert ei.value.status_code == 403


def test_symlink_escape_blocked(root, tmp_path):
    secret = tmp_path / "secret.key"
    secret.write_text("PRIVATE")
    link = root / "innocent"
    link.symlink_to(secret)                      # symlink INSIDE root → OUTSIDE target
    with pytest.raises(HTTPException) as ei:
        _safe_bt_path(str(link), must_exist=True)
    assert ei.value.status_code == 403


def test_missing_path_when_must_exist(root):
    with pytest.raises(HTTPException) as ei:
        _safe_bt_path(str(root / "nope.bin"), must_exist=True)
    assert ei.value.status_code == 400


def test_nonexistent_save_path_inside_root_ok(root):
    # a download target that doesn't exist yet is fine as long as it's within the root
    p = _safe_bt_path(str(root / "downloads" / "out.bin"), must_exist=False)
    assert str(p).startswith(str(root.resolve()))


def test_multiple_roots_via_pathsep(tmp_path, monkeypatch):
    r1 = tmp_path / "r1"; r2 = tmp_path / "r2"
    r1.mkdir(); r2.mkdir()
    monkeypatch.setenv("PRSM_BT_DATA_ROOT", os.pathsep.join([str(r1), str(r2)]))
    f = r2 / "x.bin"; f.write_text("x")
    assert _safe_bt_path(str(f), must_exist=True) == f.resolve()   # second allowed root works


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
