"""Sprint 923 — honor verify_hash on the local-content shortcut (content review #4).

ContentProvider.request_content(cid, verify_hash=True) verified returned bytes
against the expected SHA-256 on the REMOTE provider path (_request_from_provider)
but the LOCAL shortcut returned bytes unverified. A caller passing
verify_hash=True (api.py:9022, prsm_data_loader.py:235) thus got an unchecked
local copy — inconsistent with the remote path and blind to local disk
corruption of this node's own content.

Fix: the local shortcut now verifies too. For a CID with no known content hash
(e.g. a BitTorrent infohash, not a SHA-256 content address) _get_content_hash
returns None → nothing to verify → returned as before (the forward-secrecy /
non-content-addressed case). A local copy that fails verification is not served;
the request falls through to network providers (availability-preserving).
"""
from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.content_provider import ContentProvider


def _cp():
    cp = ContentProvider(
        identity=MagicMock(node_id="n"),
        transport=MagicMock(),
        gossip=MagicMock(),
    )
    cp._find_providers = MagicMock(return_value=set())   # no remote providers
    return cp


def _local(cp, cid, data, expected_hash):
    cp._local_content = {cid: object()}
    cp._fetch_local = AsyncMock(return_value=data)
    cp._get_content_hash = MagicMock(return_value=expected_hash)


@pytest.mark.asyncio
async def test_local_verify_returns_matching_content():
    cp = _cp()
    data = b"hello world"
    _local(cp, "cid1", data, hashlib.sha256(data).hexdigest())
    assert await cp.request_content("cid1", verify_hash=True) == data


@pytest.mark.asyncio
async def test_local_verify_rejects_corrupted_content():
    cp = _cp()
    # Local bytes don't match the expected hash → not served; no providers → None
    _local(cp, "cid1", b"CORRUPTED", hashlib.sha256(b"original").hexdigest())
    assert await cp.request_content("cid1", verify_hash=True) is None


@pytest.mark.asyncio
async def test_local_infohash_cid_no_known_hash_returns_bytes():
    # _get_content_hash None (BT infohash / non-content-addressed) → nothing to
    # verify by SHA-256 → local bytes returned as before.
    cp = _cp()
    _local(cp, "infohash", b"torrent-bytes", None)
    assert await cp.request_content("infohash", verify_hash=True) == b"torrent-bytes"


@pytest.mark.asyncio
async def test_local_verify_false_skips_check():
    cp = _cp()
    _local(cp, "cid1", b"whatever", "deadbeef" * 8)   # would mismatch
    assert await cp.request_content("cid1", verify_hash=False) == b"whatever"
