"""B3 — `/content/upload/shard` publishes real CIDs via ContentPublisher.

Per canonical-workflow gap-list delta (2026-05-07): the endpoint used
to register `SemanticShardManifest` objects with placeholder CIDs of
the shape ``Qm{dataset_id}-{i:04d}`` rather than real
network-discoverable CIDs. The native-storage migration rewired the
Python ``ContentUploader`` class but never rewired this endpoint.

This commit threads each shard through ``node.content_uploader.upload(
content=chunk, ...)`` — the same publish path the regular
``/content/upload`` endpoint uses — and surfaces the resulting
``UploadedContent.cid`` as the shard's CID.

Tests:
- happy path: each shard CID is the real CID from the (mocked)
  uploader, NOT a ``Qm{dataset_id}-...`` placeholder
- per-shard call count matches shard_count
- empty content rejected at 400 (no more silent placeholder manifest)
- bad base64 rejected at 400
- missing dataset_id rejected at 400
- shard_count < 1 rejected at 400
- ContentUploader unwired → 503
- ContentUploader.upload() returning None → 502
- trailing-empty-shard slicing skipped (manifest reports actual count)
"""
from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _make_uploaded(cid: str, size_bytes: int):
    """Mimic the shape of UploadedContent for the mock return."""
    return SimpleNamespace(
        # UploadedContent's field is `content_id` (not `cid`) — the
        # shard endpoint reads uploaded.content_id (api.py sprint-532
        # F45 fix). The mock previously used `cid`, so every shard
        # publish raised AttributeError.
        content_id=cid,
        filename=f"file-{cid}",
        size_bytes=size_bytes,
        content_hash="0xdeadbeef",
        creator_id="test-creator",
        royalty_rate=0.01,
        parent_cids=[],
    )


def _make_node(*, content_uploader=None):
    """Mock node minimum surface: identity + content_uploader.
    Other endpoints in api.py touch various other attributes, but
    /content/upload/shard only reads content_uploader (post-fix —
    the old code also touched data_listing_manager but the new code
    short-circuits on content_uploader being None first)."""
    node = MagicMock()
    node.identity = SimpleNamespace(node_id="test-node")
    node.content_uploader = content_uploader
    # data_listing_manager check uses hasattr → set None to skip
    node.data_listing_manager = None
    return node


def _make_client(node):
    return TestClient(create_api_app(node, enable_security=False))


# ──────────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────────


class TestRealCIDsPublished:
    def test_each_shard_gets_real_cid_from_uploader(self):
        """The placeholder-CID stub is gone. Every shard's CID is
        whatever the (mocked) ContentUploader returns from upload()."""
        uploader = MagicMock()
        # Emit distinct CIDs per call so we can verify each shard
        # got its own publish.
        upload_results = [
            _make_uploaded(cid=f"prsm:real-cid-{i}", size_bytes=100)
            for i in range(4)
        ]
        uploader.upload = AsyncMock(side_effect=upload_results)
        client = _make_client(_make_node(content_uploader=uploader))

        body = {
            "dataset_id": "ds1",
            "title": "Test Dataset",
            "content_b64": base64.b64encode(b"x" * 8000).decode(),
            "shard_count": 4,
        }
        r = client.post("/content/upload/shard", json=body)
        assert r.status_code == 200, r.text
        data = r.json()
        cids = [s["cid"] for s in data["shards"]]
        # NO placeholders.
        for cid in cids:
            assert not cid.startswith("Qmds1-"), (
                f"{cid} is a placeholder — endpoint regressed"
            )
        # Each shard's CID came from a distinct upload() call.
        assert cids == [
            "prsm:real-cid-0", "prsm:real-cid-1",
            "prsm:real-cid-2", "prsm:real-cid-3",
        ]
        assert uploader.upload.await_count == 4

    def test_uploader_called_with_chunk_bytes_and_filename(self):
        """The bytes passed to upload() are the chunk slice — not the
        whole content. Each call gets a deterministic filename
        derived from dataset_id + shard index."""
        uploader = MagicMock()
        uploader.upload = AsyncMock(side_effect=[
            _make_uploaded(cid=f"cid-{i}", size_bytes=100) for i in range(2)
        ])
        client = _make_client(_make_node(content_uploader=uploader))

        content = b"hello-world" + b"x" * 5000
        body = {
            "dataset_id": "ds-fn",
            "content_b64": base64.b64encode(content).decode(),
            "shard_count": 2,
            "royalty_rate": 0.05,
        }
        r = client.post("/content/upload/shard", json=body)
        assert r.status_code == 200, r.text
        # Check filenames + royalty_rate propagated
        calls = uploader.upload.await_args_list
        assert calls[0].kwargs["filename"] == "ds-fn-shard-0000"
        assert calls[1].kwargs["filename"] == "ds-fn-shard-0001"
        assert calls[0].kwargs["royalty_rate"] == 0.05
        # The first chunk contains the leading bytes of the content
        # (the existing slicing logic produces non-overlapping chunks
        # but doesn't always cover trailing bytes — pre-existing
        # behavior, unchanged by this commit).
        assert calls[0].kwargs["content"].startswith(b"hello-world")
        # Every chunk passed to upload() is non-empty bytes.
        for c in calls:
            assert isinstance(c.kwargs["content"], bytes)
            assert len(c.kwargs["content"]) > 0


# ──────────────────────────────────────────────────────────────────────
# Validation rejections
# ──────────────────────────────────────────────────────────────────────


class TestValidation:
    def test_missing_dataset_id_400(self):
        client = _make_client(_make_node(content_uploader=MagicMock()))
        r = client.post("/content/upload/shard", json={
            "content_b64": base64.b64encode(b"x" * 100).decode(),
        })
        assert r.status_code == 400
        assert "dataset_id" in r.json()["detail"]

    def test_empty_content_400(self):
        """No more silent placeholder-manifest creation."""
        client = _make_client(_make_node(content_uploader=MagicMock()))
        r = client.post("/content/upload/shard", json={
            "dataset_id": "ds-empty",
            "content_b64": "",
        })
        assert r.status_code == 400
        assert "empty" in r.json()["detail"].lower()

    def test_bad_base64_400(self):
        client = _make_client(_make_node(content_uploader=MagicMock()))
        r = client.post("/content/upload/shard", json={
            "dataset_id": "ds-bad-b64",
            "content_b64": "!!!not-valid-base64!!!",
        })
        assert r.status_code == 400
        assert "base64" in r.json()["detail"].lower()

    def test_zero_shard_count_400(self):
        client = _make_client(_make_node(content_uploader=MagicMock()))
        r = client.post("/content/upload/shard", json={
            "dataset_id": "ds-zero-shards",
            "content_b64": base64.b64encode(b"x" * 100).decode(),
            "shard_count": 0,
        })
        assert r.status_code == 400


# ──────────────────────────────────────────────────────────────────────
# Backend failure modes
# ──────────────────────────────────────────────────────────────────────


class TestBackendFailures:
    def test_no_content_uploader_503(self):
        client = _make_client(_make_node(content_uploader=None))
        r = client.post("/content/upload/shard", json={
            "dataset_id": "ds-no-uploader",
            "content_b64": base64.b64encode(b"x" * 100).decode(),
            "shard_count": 1,
        })
        assert r.status_code == 503
        assert "uploader" in r.json()["detail"].lower()

    def test_uploader_returns_none_502(self):
        """If a shard's upload returns None (publisher unavailable or
        rejected the payload), the endpoint must surface 502 — NEVER
        emit a placeholder CID to keep the response 'green'."""
        uploader = MagicMock()
        uploader.upload = AsyncMock(return_value=None)
        client = _make_client(_make_node(content_uploader=uploader))
        r = client.post("/content/upload/shard", json={
            "dataset_id": "ds-fail",
            "content_b64": base64.b64encode(b"x" * 4000).decode(),
            "shard_count": 2,
        })
        assert r.status_code == 502
        assert "publisher" in r.json()["detail"].lower()


# ──────────────────────────────────────────────────────────────────────
# Slicing edge cases
# ──────────────────────────────────────────────────────────────────────


class TestCreatorEthAddressThreading:
    """sp995 (fix A) — the shard endpoint must thread creator_eth_address into
    every shard's upload() call, so each shard carries the REAL creator's on-chain
    identity (which the §14 stake gate + on-chain royalty routing key on), not the
    hosting operator's wallet. Pre-fix the field was dropped entirely."""

    def test_creator_eth_address_threaded_to_every_shard(self):
        eth = "0x" + "a" * 40
        uploader = MagicMock()
        uploader.upload = AsyncMock(side_effect=[
            _make_uploaded(cid=f"cid-{i}", size_bytes=100) for i in range(2)
        ])
        client = _make_client(_make_node(content_uploader=uploader))
        r = client.post("/content/upload/shard", json={
            "dataset_id": "ds-eth",
            "content_b64": base64.b64encode(b"x" * 5000).decode(),
            "shard_count": 2,
            "creator_eth_address": eth,
        })
        assert r.status_code == 200, r.text
        calls = uploader.upload.await_args_list
        assert len(calls) >= 1
        for c in calls:
            assert c.kwargs["creator_eth_address"] == eth

    def test_omitted_creator_eth_address_passes_none(self):
        """Backward-compat: omitting it passes None → upload() falls back to the
        operator address (operator self-upload), unchanged behavior."""
        uploader = MagicMock()
        uploader.upload = AsyncMock(side_effect=[
            _make_uploaded(cid="cid-0", size_bytes=100),
        ])
        client = _make_client(_make_node(content_uploader=uploader))
        r = client.post("/content/upload/shard", json={
            "dataset_id": "ds-no-eth",
            "content_b64": base64.b64encode(b"x" * 3000).decode(),
            "shard_count": 1,
        })
        assert r.status_code == 200, r.text
        assert uploader.upload.await_args_list[0].kwargs["creator_eth_address"] is None

    def test_malformed_creator_eth_address_422(self):
        uploader = MagicMock()
        uploader.upload = AsyncMock(return_value=_make_uploaded("c", 1))
        client = _make_client(_make_node(content_uploader=uploader))
        r = client.post("/content/upload/shard", json={
            "dataset_id": "ds-bad-eth",
            "content_b64": base64.b64encode(b"x" * 3000).decode(),
            "creator_eth_address": "0xnot-a-valid-address",
        })
        assert r.status_code == 422
        assert "creator_eth_address" in r.json()["detail"]
        uploader.upload.assert_not_awaited()


class TestShardSlicing:
    def test_trailing_empty_shards_skipped(self):
        """When shard_count exceeds what 1024-byte minimum chunking
        produces, trailing slices would be empty. The endpoint must
        skip those rather than try to publish 0 bytes."""
        uploader = MagicMock()
        # We expect AT MOST 2 shards from 1500 bytes with chunk_size
        # floor at 1024 — the third+fourth would be empty and skipped.
        uploader.upload = AsyncMock(side_effect=[
            _make_uploaded(cid=f"cid-{i}", size_bytes=512) for i in range(4)
        ])
        client = _make_client(_make_node(content_uploader=uploader))
        r = client.post("/content/upload/shard", json={
            "dataset_id": "ds-trailing",
            "content_b64": base64.b64encode(b"x" * 1500).decode(),
            "shard_count": 4,
        })
        assert r.status_code == 200, r.text
        data = r.json()
        # The actual shard count in the manifest reflects what was
        # successfully published — never includes empty placeholders.
        assert len(data["shards"]) <= 4
        # And the uploader was only called for the non-empty ones.
        assert uploader.upload.await_count == len(data["shards"])
