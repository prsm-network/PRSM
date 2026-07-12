"""POST /compute/inference — per-requester rate limiting via
PRSM_INFERENCE_MAX_RPS_PER_REQUESTER. Independent bucket from
/compute/forge so operators can tune the two endpoints separately.
"""
from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.rate_limiter import get_or_build_bucket, reset_global_bucket


def _node():
    """Bare node — inference_executor None so the rate-cap check fires
    BEFORE the 503 inference-not-configured check (same precedence
    as forge's rate-cap-before-agent_forge ordering)."""
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.inference_executor = None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


@pytest.fixture(autouse=True)
def _reset_buckets():
    """Reset all rate-limit buckets between tests (pytest fixture
    fires for both module-level tests AND class methods)."""
    reset_global_bucket()
    yield
    reset_global_bucket()


# ──────────────────────────────────────────────────────────────────────
# Cap enforcement
# ──────────────────────────────────────────────────────────────────────


class TestInferenceRateLimit:
    def test_burst_then_429(self):
        """cap=2/sec, burst=2: 3rd request rejects with 429."""
        node = _node()
        with patch.dict(os.environ, {
            "PRSM_INFERENCE_MAX_RPS_PER_REQUESTER": "2",
        }):
            # sp1434 — freeze the bucket clock so no token refills mid-test. Same wall-clock flake
            # (and same fix) as the /compute/forge test: without this the assertion silently
            # required all 3 round-trips inside 0.5s (burst=2 refills at 2 tok/sec), so a loaded CI
            # runner refilled a token and the 3rd request sailed past the limiter into the
            # inference_executor-is-None 503 → "assert 503 == 429". The endpoint fetches this SAME
            # process-global bucket (name="inference"; rate unchanged → get_or_build_bucket returns
            # this instance), so freezing _now here holds inside the handler.
            frozen = time.time()
            bucket = get_or_build_bucket(2.0, name="inference")
            bucket._now = lambda: frozen
            client = _client(node)
            for _ in range(2):
                resp = client.post(
                    "/compute/inference",
                    json={"prompt": "x", "model_id": "m"},
                )
                # Through to 503 (inference_executor None) — cap
                # not exceeded.
                assert resp.status_code == 503
            # Fail LOUD if the endpoint stopped using the bucket we froze.
            assert bucket._buckets, (
                "endpoint did not consume the frozen 'inference' bucket — test no longer measures "
                "what it thinks it does"
            )
            resp = client.post(
                "/compute/inference",
                json={"prompt": "x", "model_id": "m"},
            )
            assert resp.status_code == 429
            assert "/compute/inference" in resp.json()["detail"]

    def test_no_env_disables_limit(self):
        node = _node()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_INFERENCE_MAX_RPS_PER_REQUESTER", None)
            client = _client(node)
            for _ in range(10):
                resp = client.post(
                    "/compute/inference",
                    json={"prompt": "x", "model_id": "m"},
                )
                assert resp.status_code == 503  # never 429

    def test_invalid_env_disables_limit(self):
        node = _node()
        with patch.dict(os.environ, {
            "PRSM_INFERENCE_MAX_RPS_PER_REQUESTER": "not_a_number",
        }):
            client = _client(node)
            for _ in range(10):
                resp = client.post(
                    "/compute/inference",
                    json={"prompt": "x", "model_id": "m"},
                )
                assert resp.status_code == 503  # never 429


# ──────────────────────────────────────────────────────────────────────
# Independent buckets — forge vs inference
# ──────────────────────────────────────────────────────────────────────


class TestStreamSharesInferenceBucket:
    """/compute/inference/stream uses the same "inference" bucket
    as /compute/inference — single requester's combined RPS across
    both endpoints is capped together."""

    def test_stream_endpoint_429s_after_burst(self):
        """cap=2/sec, burst=2: third stream request rejects."""
        node = _node()
        with patch.dict(os.environ, {
            "PRSM_INFERENCE_MAX_RPS_PER_REQUESTER": "2",
        }):
            client = _client(node)
            for _ in range(2):
                resp = client.post(
                    "/compute/inference/stream",
                    json={"prompt": "x", "model_id": "m"},
                )
                assert resp.status_code == 503  # cap not exceeded
            resp = client.post(
                "/compute/inference/stream",
                json={"prompt": "x", "model_id": "m"},
            )
            assert resp.status_code == 429
            assert "/compute/inference/stream" in resp.json()["detail"]

    def test_stream_drains_inference_bucket(self):
        """Hitting /compute/inference 1× drains 1 token from the
        inference bucket; subsequent /compute/inference/stream
        request sees only burst-1 capacity available."""
        node = _node()
        with patch.dict(os.environ, {
            "PRSM_INFERENCE_MAX_RPS_PER_REQUESTER": "2",
        }):
            client = _client(node)
            # Drain 2 from inference (cap reached).
            for _ in range(2):
                resp = client.post(
                    "/compute/inference",
                    json={"prompt": "x", "model_id": "m"},
                )
                assert resp.status_code == 503
            # Stream request now hits 429 (shared bucket exhausted).
            resp = client.post(
                "/compute/inference/stream",
                json={"prompt": "x", "model_id": "m"},
            )
            assert resp.status_code == 429


class TestSeparateBuckets:
    def test_named_buckets_independent_at_module_level(self):
        """Direct test of the rate_limiter module: buckets keyed
        by name are independent. Drains one without affecting
        the other."""
        from prsm.node.rate_limiter import get_or_build_bucket

        reset_global_bucket()
        forge_bucket = get_or_build_bucket(2.0, name="forge")
        inference_bucket = get_or_build_bucket(2.0, name="inference")

        # Different objects.
        assert forge_bucket is not inference_bucket

        # Drain forge bucket fully.
        for _ in range(2):
            assert forge_bucket.try_consume("req-1") is True
        assert forge_bucket.try_consume("req-1") is False

        # Inference bucket is still full for the same requester.
        assert inference_bucket.try_consume("req-1") is True
        assert inference_bucket.try_consume("req-1") is True
        assert inference_bucket.try_consume("req-1") is False

    def test_default_name_preserves_legacy_aliases(self):
        """get_or_build_bucket(rate) without a name uses
        '_default' bucket and updates the legacy module-level
        _GLOBAL_BUCKET / _GLOBAL_RATE aliases for backwards
        compatibility."""
        import prsm.node.rate_limiter as rl

        reset_global_bucket()
        bucket = rl.get_or_build_bucket(3.0)
        assert bucket is rl._GLOBAL_BUCKET
        assert rl._GLOBAL_RATE == 3.0
