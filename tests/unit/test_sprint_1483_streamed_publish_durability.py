"""Sprint 1483 — a streamed (large-file) publish must be DISCOVERABLE and DURABLE.

The user-readiness re-assessment found the flagship large-dataset path broken in a
quiet way: `/content/upload-stream` staged the bytes correctly but only called
`ContentProvider.register_local_content` — an IN-MEMORY dict. So a multi-GB publish
was:

  * UNDISCOVERABLE — no GOSSIP_CONTENT_ADVERTISE, so no other node's ContentIndex
    ever learned the CID existed; and
  * NON-DURABLE — no provenance row, so `_hydrate_from_db` had nothing to replay and
    the registration vanished on the next daemon restart.

Publishing a multi-GB dataset and having it silently disappear is a work-loss
failure for the user even though no FTNS moves. The prior endpoint test asserted
only `register_local_content`, which is exactly why this shipped.
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.content_uploader import UploadedContent
from prsm.node.gossip import GOSSIP_CONTENT_ADVERTISE

pytestmark = pytest.mark.asyncio

CID = "bafyStreamedBigDataset"
CONTENT_HASH = "a" * 64


def _uploader():
    """A ContentUploader with only what register_streamed_publish touches, so the
    REAL method runs (no re-implementation in the test)."""
    from prsm.node.content_uploader import ContentUploader
    u = ContentUploader.__new__(ContentUploader)
    u.identity = SimpleNamespace(node_id="node-abc")
    u.gossip = SimpleNamespace(publish=AsyncMock())
    u.uploaded_content = {}
    u.creator_address = "0x" + "c" * 40
    u._persist_provenance = AsyncMock()
    return u


async def test_streamed_publish_is_advertised_and_persisted():
    """★ The core fix: both the gossip advertise AND the provenance persist fire."""
    u = _uploader()
    rec = await u.register_streamed_publish(
        content_id=CID, filename="big.parquet", size_bytes=5 * 1024**3,
        content_hash=CONTENT_HASH, provenance_hash="0xdead",
    )
    # Discoverable.
    u.gossip.publish.assert_awaited_once()
    topic, payload = u.gossip.publish.await_args.args
    assert topic == GOSSIP_CONTENT_ADVERTISE
    assert payload["cid"] == CID
    assert payload["size_bytes"] == 5 * 1024**3
    assert payload["content_hash"] == CONTENT_HASH
    assert payload["provider_id"] == "node-abc"
    # Durable.
    u._persist_provenance.assert_awaited_once()
    assert u._persist_provenance.await_args.args[0].content_id == CID
    assert isinstance(rec, UploadedContent)


async def test_registered_item_enrolls_in_the_readvertise_sweep():
    """★ Being in `uploaded_content` is what lets sp1343's periodic re-advertise
    reach LATE-JOINING nodes after the 24h gossip-retention window. Without it a
    streamed dataset goes dark to anyone who joins later."""
    from prsm.node.content_uploader import ContentUploader
    u = _uploader()
    await u.register_streamed_publish(
        content_id=CID, filename="big.parquet", size_bytes=1024,
        content_hash=CONTENT_HASH,
    )
    assert CID in u.uploaded_content
    u.gossip.publish.reset_mock()
    n = await ContentUploader.readvertise_all(u)
    assert n == 1
    assert u.gossip.publish.await_args.args[0] == GOSSIP_CONTENT_ADVERTISE


async def test_advertise_payload_uses_the_shared_builder():
    """The payload must come from _build_content_advertise_payload so its shape
    cannot drift between the JSON and streaming publish routes."""
    from prsm.node.content_uploader import ContentUploader
    u = _uploader()
    await u.register_streamed_publish(
        content_id=CID, filename="f.bin", size_bytes=7,
        content_hash=CONTENT_HASH, metadata={"tier": "A"},
    )
    sent = u.gossip.publish.await_args.args[1]
    expected = ContentUploader._build_content_advertise_payload(u, u.uploaded_content[CID])
    assert sent == expected


async def test_creator_eth_address_is_carried_for_royalties():
    """★ A typo'd attribute name would silently drop the royalty destination from
    every streamed publish. Assert it is actually populated."""
    u = _uploader()
    await u.register_streamed_publish(
        content_id=CID, filename="f", size_bytes=1, content_hash=CONTENT_HASH)
    assert u.gossip.publish.await_args.args[1]["creator_eth_address"] == "0x" + "c" * 40


async def test_reregistration_is_idempotent_not_duplicated():
    u = _uploader()
    first = await u.register_streamed_publish(
        content_id=CID, filename="f", size_bytes=1, content_hash=CONTENT_HASH)
    await u.register_streamed_publish(
        content_id=CID, filename="f", size_bytes=1, content_hash=CONTENT_HASH)
    assert len(u.uploaded_content) == 1
    # created_at is preserved across re-registration (not reset to "now").
    assert u.uploaded_content[CID].created_at == first.created_at


async def test_gossip_failure_does_not_fail_the_publish():
    """The bytes are already staged — a transient gossip failure must not raise."""
    u = _uploader()
    u.gossip.publish = AsyncMock(side_effect=RuntimeError("gossip down"))
    rec = await u.register_streamed_publish(
        content_id=CID, filename="f", size_bytes=1, content_hash=CONTENT_HASH)
    assert rec.content_id == CID
    u._persist_provenance.assert_awaited_once()   # persistence still happened


async def test_persist_failure_does_not_fail_the_publish():
    u = _uploader()
    u._persist_provenance = AsyncMock(side_effect=RuntimeError("db down"))
    rec = await u.register_streamed_publish(
        content_id=CID, filename="f", size_bytes=1, content_hash=CONTENT_HASH)
    assert rec.content_id == CID
    u.gossip.publish.assert_awaited_once()        # advertise still happened


# ───────────────────── the endpoint wiring ─────────────────────

def _app(uploader):
    node = MagicMock()
    node.identity.node_id = "node-abc"
    node.ftns_ledger = None
    node.content_uploader = uploader

    # Mirror the REAL publish_from_path return shape: the endpoint reads
    # result.torrent_infohash for the CID and result.staged_path.name for the
    # content-addressed sha256 hex (NOT manifest.content_id).
    publisher = MagicMock()
    publisher.publish_from_path = AsyncMock(return_value=SimpleNamespace(
        torrent_infohash=CID,
        staged_path=SimpleNamespace(name=CONTENT_HASH),
        manifest=SimpleNamespace(total_size=1234),
    ))
    node.content_provider.content_publisher = publisher
    node.content_provider.register_local_content = MagicMock()
    node._content_filter_store = None
    return TestClient(create_api_app(node, enable_security=False),
                      raise_server_exceptions=False), node


def test_endpoint_reports_advertised_and_durable_honestly():
    """★ The response tells a publisher whether their multi-GB dataset is actually
    discoverable + will survive a restart, instead of leaving them to assume it."""
    uploader = MagicMock()
    uploader.register_streamed_publish = AsyncMock()
    client, node = _app(uploader)
    r = client.post("/content/upload-stream?filename=big.parquet", content=b"x" * 64)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["advertised"] is True and body["durable"] is True
    uploader.register_streamed_publish.assert_awaited_once()


def test_endpoint_degrades_honestly_without_an_uploader():
    """No uploader → the publish still succeeds (bytes are staged) but must NOT
    claim to be advertised or durable."""
    client, node = _app(None)
    r = client.post("/content/upload-stream?filename=big.parquet", content=b"x" * 64)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["advertised"] is False and body["durable"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
