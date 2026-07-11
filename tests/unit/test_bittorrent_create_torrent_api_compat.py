"""Sprint 330 — BitTorrentClient.create_torrent compat with
libtorrent 2.0.12+ API.

Pre-fix `create_torrent` called `t.set_priv(True)` and
`t.get_torrent_info()` directly. libtorrent 2.0.12+ removed
both methods from the `create_torrent` object — `set_priv`
became a `priv` attribute and `get_torrent_info` was dropped
entirely (callers now bencode-decode the generated payload
into a fresh `torrent_info`).

PRSM nodes running on machines with libtorrent 2.0.12+
hit AttributeError on every torrent creation. The fix
uses feature-detection: try the old method first for
back-compat, fall back to the new API path.

Tests below patch `prsm.core.bittorrent_client.lt` with
controlled mocks to exercise both API branches
deterministically without requiring two versions of
libtorrent installed.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prsm.core.bittorrent_client import (
    BitTorrentClient,
    BitTorrentResult,
)


def _fake_create_torrent_old_api():
    """Return a MagicMock simulating the OLD (≤2.0.11) API:
    has `set_priv` method + `get_torrent_info` method."""
    t = MagicMock()
    # Configure the methods explicitly so the hasattr() checks
    # in production code see them
    t.set_priv = MagicMock()
    t.get_torrent_info = MagicMock(return_value=MagicMock(
        info_hash=MagicMock(return_value="OLDHASH"),
    ))
    t.set_comment = MagicMock()
    t.generate = MagicMock(return_value={"_": "entry"})
    return t


def _fake_create_torrent_new_api():
    """Return a MagicMock simulating the NEW (≥2.0.12) API:
    NO `set_priv` method, NO `get_torrent_info` method —
    only `priv` attribute."""
    t = MagicMock(spec=["set_comment", "generate", "priv"])
    t.set_comment = MagicMock()
    t.generate = MagicMock(return_value={"_": "entry"})
    return t


@pytest.fixture
def tmp_file(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_bytes(b"hello, libtorrent")
    return f


def _patched_lt(create_torrent_factory, *, expect_new_api=False):
    """Build a MagicMock module with create_torrent +
    set_piece_hashes + bencode + torrent_info wired to
    deterministic behaviors."""
    lt_mod = MagicMock()
    lt_mod.file_storage = MagicMock(return_value=MagicMock())
    lt_mod.create_torrent = MagicMock(
        return_value=create_torrent_factory(),
    )
    lt_mod.set_piece_hashes = MagicMock()
    lt_mod.bencode = MagicMock(return_value=b"BENCODED")
    lt_mod.bdecode = MagicMock(return_value={"_": "decoded"})
    # If the new-api path is taken, torrent_info(bdecode(...))
    # is what produces the result
    if expect_new_api:
        new_info = MagicMock()
        new_info.info_hash = MagicMock(return_value="NEWHASH")
        lt_mod.torrent_info = MagicMock(return_value=new_info)
    return lt_mod


# ── Old API path (libtorrent ≤2.0.11) ─────────────────────


def test_create_torrent_uses_set_priv_when_method_exists(tmp_file):
    """When `t.set_priv` exists, the code must call it
    (back-compat)."""
    lt_mod = _patched_lt(_fake_create_torrent_old_api)
    client = BitTorrentClient()
    client._initialized = True
    client._loop = asyncio.new_event_loop()
    try:
        with patch("prsm.core.bittorrent_client.LT_AVAILABLE", True), \
             patch("prsm.core.bittorrent_client.lt", lt_mod):
            result = client._loop.run_until_complete(
                client.create_torrent(tmp_file, private=True),
            )
        assert result.success
        # set_priv was called with True
        ct = lt_mod.create_torrent.return_value
        ct.set_priv.assert_called_once_with(True)
        # get_torrent_info was used (old API)
        ct.get_torrent_info.assert_called_once()
        assert result.infohash == "OLDHASH"
    finally:
        client._loop.close()


# ── New API path (libtorrent 2.0.12+) ─────────────────────


def test_create_torrent_falls_back_to_priv_attr(tmp_file):
    """When `t.set_priv` is absent, the code must set
    `t.priv = True` (new API)."""
    lt_mod = _patched_lt(
        _fake_create_torrent_new_api, expect_new_api=True,
    )
    client = BitTorrentClient()
    client._initialized = True
    client._loop = asyncio.new_event_loop()
    try:
        with patch("prsm.core.bittorrent_client.LT_AVAILABLE", True), \
             patch("prsm.core.bittorrent_client.lt", lt_mod):
            result = client._loop.run_until_complete(
                client.create_torrent(tmp_file, private=True),
            )
        assert result.success, f"failed: {result.error}"
        ct = lt_mod.create_torrent.return_value
        # priv attribute was set (new API)
        assert ct.priv is True
    finally:
        client._loop.close()


def test_create_torrent_uses_torrent_info_fallback_on_new_api(
    tmp_file,
):
    """When `t.get_torrent_info` is absent, the code must
    bdecode + torrent_info(bencoded) to derive the infohash."""
    lt_mod = _patched_lt(
        _fake_create_torrent_new_api, expect_new_api=True,
    )
    client = BitTorrentClient()
    client._initialized = True
    client._loop = asyncio.new_event_loop()
    try:
        with patch("prsm.core.bittorrent_client.LT_AVAILABLE", True), \
             patch("prsm.core.bittorrent_client.lt", lt_mod):
            result = client._loop.run_until_complete(
                client.create_torrent(tmp_file),
            )
        assert result.success
        # bdecode + torrent_info were used (new API path)
        lt_mod.bdecode.assert_called_once_with(b"BENCODED")
        lt_mod.torrent_info.assert_called_once()
        assert result.infohash == "NEWHASH"
    finally:
        client._loop.close()


# ── private flag honored across both paths ───────────────


def test_create_torrent_does_not_set_priv_when_not_private(
    tmp_file,
):
    """If private=False, neither set_priv nor t.priv should
    fire."""
    lt_mod = _patched_lt(_fake_create_torrent_old_api)
    client = BitTorrentClient()
    client._initialized = True
    client._loop = asyncio.new_event_loop()
    try:
        with patch("prsm.core.bittorrent_client.LT_AVAILABLE", True), \
             patch("prsm.core.bittorrent_client.lt", lt_mod):
            result = client._loop.run_until_complete(
                client.create_torrent(tmp_file, private=False),
            )
        assert result.success
        ct = lt_mod.create_torrent.return_value
        ct.set_priv.assert_not_called()
    finally:
        client._loop.close()
