"""Cap shard upload payload size on /content/upload/shard
(sprint 103).

Companion to /content/upload size cap (sprint 102) — same
DoS vector via the shard endpoint, higher default ceiling
since shard endpoint natively chunks.
"""
from __future__ import annotations

import base64
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._webhook_log = None
    cu = AsyncMock()
    cu.upload = AsyncMock(
        return_value=MagicMock(
            content_id="cid1", filename="shard.bin",
            size_bytes=100, content_hash="0xCAFE",
        ),
    )
    node.content_uploader = cu
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def _post(node, content_bytes):
    encoded = base64.b64encode(content_bytes).decode("ascii")
    return _client(node).post(
        "/content/upload/shard",
        json={
            "dataset_id": "test",
            "content_b64": encoded,
            "shard_count": 2,
        },
    )


class TestShardSizeCap:
    def test_default_100mb_rejects_101mb(self):
        # The global http_body_size_limit_middleware (sprint 742)
        # caps ALL request bodies at PRSM_HTTP_MAX_BODY_BYTES
        # (default 1 MiB) BEFORE any endpoint runs, so a 101MB body is
        # intercepted globally before the shard endpoint's own cap. To
        # exercise the shard-specific cap in isolation, raise the
        # global cap above the request size. (Operators wanting large
        # shard uploads must likewise raise PRSM_HTTP_MAX_BODY_BYTES —
        # documented in the activation runbook.)
        with patch.dict(
            os.environ,
            {"PRSM_HTTP_MAX_BODY_BYTES": str(512 * 1024 * 1024)},
            clear=False,
        ):
            os.environ.pop("PRSM_MAX_SHARD_UPLOAD_BYTES", None)
            big = b"x" * (101 * 1024 * 1024)
            resp = _post(_node(), big)
        assert resp.status_code == 413
        assert "PRSM_MAX_SHARD_UPLOAD_BYTES" in resp.json()["detail"]

    def test_env_override_smaller(self):
        with patch.dict(
            os.environ, {"PRSM_MAX_SHARD_UPLOAD_BYTES": "1024"},
        ):
            resp = _post(_node(), b"x" * 2048)
        assert resp.status_code == 413

    def test_invalid_env_falls_back(self):
        with patch.dict(
            os.environ, {"PRSM_MAX_SHARD_UPLOAD_BYTES": "not-a-num"},
        ):
            # Default 100MB; small payload passes pre-cap
            resp = _post(_node(), b"hello world")
        # Either 200 (passes) or 502 (downstream — accepted past cap)
        assert resp.status_code != 413


class TestShardUploadNotImplemented:
    """Sprint 921 (content data-plane review #3) — a single shard chunk that
    still exceeds the uploader's internal sharding threshold makes
    content_uploader.upload() raise NotImplementedError (the sp491 F29 guard).
    The /content/upload/shard handler must surface that as 413 with the
    actionable detail (mirroring /content/upload at api.py:8906), NOT a generic
    500."""

    def test_chunk_over_threshold_returns_413_not_500(self):
        node = _node()
        node.content_uploader.upload = AsyncMock(
            side_effect=NotImplementedError(
                "content exceeds sharding threshold; split via a larger "
                "shard_count or /content/upload/shard"
            ),
        )
        resp = _post(node, b"hello world")
        assert resp.status_code == 413
        assert "threshold" in resp.json()["detail"].lower()
