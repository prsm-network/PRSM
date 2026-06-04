"""Sprint 1016 — /compute/inference/stream works end-to-end with the mock executor.

Integration test: wire the REAL build_mock_streaming_executor onto the API app
and POST to /compute/inference/stream, asserting genuine SSE token events + a
terminal result event — proving the mock makes the streaming endpoint functional
on a single node (no real GPUs). Mirrors the node/app harness in
test_inference_stream_job_history_wiring.py.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from prsm.compute.inference.mock_streaming import (
    MOCK_MODEL_ID,
    build_mock_streaming_executor,
)
from prsm.node.api import create_api_app
from prsm.node.identity import generate_node_identity
from prsm.node.job_history import JobHistoryStore, JobStatus


def _node():
    ident = generate_node_identity("itest-stream")
    node = MagicMock()
    node.identity = ident
    node.ftns_ledger = None
    node._payment_escrow = None
    node.privacy_budget = None
    node._receipt_store = None
    node._job_history = JobHistoryStore()
    node.inference_executor = build_mock_streaming_executor(ident)
    return node


def _client(node):
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def _post_stream(client, model_id=MOCK_MODEL_ID, privacy_tier="none"):
    return client.post("/compute/inference/stream", json={
        "prompt": "hello world",
        "model_id": model_id,
        "budget_ftns": 1.0,
        "privacy_tier": privacy_tier,
    })


def test_stream_endpoint_emits_token_and_result_events():
    node = _node()
    resp = _post_stream(_client(node))
    assert resp.status_code == 200
    body = "\n".join(line for line in resp.iter_lines() if line)
    # Genuine SSE token events flowed, then a terminal result.
    assert "event: token" in body, body[:500]
    assert "event: result" in body, body[:500]
    # No error event on the happy path.
    assert "event: error" not in body


def test_stream_endpoint_records_completed_history():
    node = _node()
    resp = _post_stream(_client(node))
    assert resp.status_code == 200
    list(resp.iter_lines())  # drain
    completed = [
        r for r in node._job_history.list(
            status_filter=JobStatus.COMPLETED, limit=10, offset=0,
        )
        if r.route == "inference_stream"
    ]
    assert len(completed) == 1
    # The synthetic mock output landed in history.
    assert completed[0].response  # non-empty streamed output


def test_stream_endpoint_unknown_model_no_crash():
    node = _node()
    resp = _post_stream(_client(node), model_id="no-such-model")
    # The request is accepted (200 SSE) but the stream surfaces a failure/error
    # event rather than crashing the endpoint.
    assert resp.status_code in (200, 400, 422)
    if resp.status_code == 200:
        body = "\n".join(line for line in resp.iter_lines() if line)
        assert "event: result" in body or "event: error" in body
