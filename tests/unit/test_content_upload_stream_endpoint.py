"""Sprint 1457 — POST /content/upload-stream: streaming large-content publish endpoint.

Closes the publish half of the large-content gap (the retrieve half is complete). The JSON
/content/upload caps at ~10 MiB and materializes the body; this streams the request body
straight to a temp file, stages it via LocalContentPublisher.publish_from_path (byte-identical
CID to an in-memory publish), registers it with the ContentProvider so the sp1290 streaming-send
path serves it, and returns the CID. Tier A (public) only.
"""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from prsm.core.torrent_infohash import compute_v1_infohash_single_file
from prsm.node.api import create_api_app
from prsm.node.local_content_publisher import LocalContentPublisher

_BODY = b"large tier-a dataset payload " * 100_000  # ~2.9 MB → multi-piece, exceeds JSON cap


def _node(tmp_path, *, with_publisher=True):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._content_filter_store = None
    node.content_uploader = None
    if with_publisher:
        publisher = LocalContentPublisher(tmp_path / "stage", node_id="test-node")
        node.content_provider.content_publisher = publisher
        node.content_provider.register_local_content = MagicMock()
        node._publisher = publisher  # test handle
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False),
                      raise_server_exceptions=False)


def test_upload_stream_publishes_canonical_cid_and_registers(tmp_path):
    node = _node(tmp_path)
    resp = _client(node).post("/content/upload-stream", content=_BODY,
                              params={"filename": "data.bin"})
    assert resp.status_code == 200, resp.text
    j = resp.json()
    # CID is the canonical v1 infohash — byte-identical to an in-memory publish of the same bytes.
    expected_cid = compute_v1_infohash_single_file(
        _BODY, hashlib.sha256(_BODY).hexdigest())
    assert j["cid"] == expected_cid
    assert j["content_hash"] == hashlib.sha256(_BODY).hexdigest()
    assert j["size_bytes"] == len(_BODY)
    # Registered with the provider (so the streaming-send path serves it) under the right CID.
    node.content_provider.register_local_content.assert_called_once()
    ka = node.content_provider.register_local_content.call_args.kwargs
    assert ka["cid"] == expected_cid and ka["size_bytes"] == len(_BODY)
    # Staged + servable via the publisher's local shortcut.
    assert node._publisher.local_publish_path(expected_cid) is not None


def test_upload_stream_501_without_streaming_publisher(tmp_path):
    node = _node(tmp_path, with_publisher=False)
    node.content_provider.content_publisher = object()   # no publish_from_path
    resp = _client(node).post("/content/upload-stream", content=b"data")
    assert resp.status_code == 501


def test_upload_stream_503_without_provider(tmp_path):
    node = _node(tmp_path)
    node.content_provider = None
    resp = _client(node).post("/content/upload-stream", content=b"data")
    assert resp.status_code == 503


def test_upload_stream_empty_body_422(tmp_path):
    node = _node(tmp_path)
    resp = _client(node).post("/content/upload-stream", content=b"")
    assert resp.status_code == 422


def test_upload_stream_over_cap_413(tmp_path, monkeypatch):
    monkeypatch.setenv("PRSM_MAX_STREAMING_UPLOAD_BYTES", "1024")
    node = _node(tmp_path)
    resp = _client(node).post("/content/upload-stream", content=b"x" * 4096)
    assert resp.status_code == 413
    node.content_provider.register_local_content.assert_not_called()


def test_upload_stream_content_filter_blocks_451(tmp_path):
    node = _node(tmp_path)
    node._content_filter_store = MagicMock()
    node._content_filter_store.is_cid_blocked = MagicMock(return_value=True)
    resp = _client(node).post("/content/upload-stream", content=_BODY)
    assert resp.status_code == 451
    node.content_provider.register_local_content.assert_not_called()
