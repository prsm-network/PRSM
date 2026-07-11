"""POST /compute/forge — per-requester rate limiting.

DoS-protection feature: operator sets
``PRSM_FORGE_MAX_RPS_PER_REQUESTER`` to cap requests per second
per requester (token-bucket with burst). Default unset → no
limit, v1 behavior preserved bit-identically.

Token-bucket algorithm:
- Each requester gets a bucket initialized to burst tokens.
- Each request consumes 1 token.
- Tokens refill at `rate` tokens/sec, capped at `burst`.
- Empty bucket → 429 with Retry-After.

For testing simplicity the rate limiter is exposed as a
testable module + tests target it directly. Endpoint
integration is verified separately.
"""
from __future__ import annotations

import time

import pytest

from prsm.node.rate_limiter import SimpleTokenBucket


# ──────────────────────────────────────────────────────────────────────
# SimpleTokenBucket
# ──────────────────────────────────────────────────────────────────────


class TestSimpleTokenBucket:
    def test_initial_burst_consumed_immediately(self):
        """Fresh bucket allows `burst` immediate requests."""
        bucket = SimpleTokenBucket(rate=1.0, burst=5)
        for _ in range(5):
            assert bucket.try_consume("req-1") is True
        # 6th in same instant exceeds burst.
        assert bucket.try_consume("req-1") is False

    def test_separate_requesters_have_separate_buckets(self):
        bucket = SimpleTokenBucket(rate=1.0, burst=2)
        assert bucket.try_consume("req-a") is True
        assert bucket.try_consume("req-a") is True
        assert bucket.try_consume("req-a") is False
        # req-b gets its own fresh bucket.
        assert bucket.try_consume("req-b") is True
        assert bucket.try_consume("req-b") is True
        assert bucket.try_consume("req-b") is False

    def test_tokens_refill_over_time(self):
        """After 1 second at rate=2/sec, 2 fresh tokens available."""
        bucket = SimpleTokenBucket(rate=2.0, burst=2)
        # Drain.
        bucket.try_consume("r")
        bucket.try_consume("r")
        assert bucket.try_consume("r") is False
        # Simulate 1 second elapsed.
        bucket._now = lambda: time.time() + 1.0
        # Should have refilled 2 tokens.
        assert bucket.try_consume("r") is True
        assert bucket.try_consume("r") is True
        # Cap stays at burst.
        assert bucket.try_consume("r") is False

    def test_partial_refill(self):
        """At rate=10/sec after 0.5s, 5 tokens refilled."""
        bucket = SimpleTokenBucket(rate=10.0, burst=5)
        for _ in range(5):
            bucket.try_consume("r")
        bucket._now = lambda: time.time() + 0.5
        # 5 fresh tokens.
        for _ in range(5):
            assert bucket.try_consume("r") is True
        assert bucket.try_consume("r") is False

    def test_retry_after_returns_seconds_until_one_token(self):
        bucket = SimpleTokenBucket(rate=2.0, burst=2)
        bucket.try_consume("r")
        bucket.try_consume("r")
        # Bucket empty; need 0.5s for 1 token at rate=2/sec.
        retry = bucket.retry_after("r")
        assert 0.4 <= retry <= 0.6

    def test_retry_after_zero_when_tokens_available(self):
        bucket = SimpleTokenBucket(rate=1.0, burst=1)
        # Fresh bucket has 1 token.
        assert bucket.retry_after("r") == 0.0


# ──────────────────────────────────────────────────────────────────────
# Endpoint integration
# ──────────────────────────────────────────────────────────────────────


import os
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.job_history import JobHistoryStore
from prsm.node.rate_limiter import get_or_build_bucket, reset_global_bucket


@pytest.fixture(autouse=True)
def _reset_buckets():
    """The endpoint's token buckets are PROCESS-GLOBAL. Wipe them either
    side of every test so this file neither inherits a drained "forge"
    bucket from another module nor leaves one behind. Mirrors the fixture
    in tests/unit/test_compute_inference_rate_limit.py, which also builds
    a "forge"-named global bucket — today we only survive that collision
    because this filename sorts first.
    """
    reset_global_bucket()
    yield
    reset_global_bucket()


def _node_minimal():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node._job_history = JobHistoryStore()
    node.agent_forge = None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


class TestForgeRateLimit:
    def test_burst_then_limit(self):
        """With cap=2/sec burst=2, 3rd request rejects with 429."""
        node = _node_minimal()
        with patch.dict(
            os.environ, {"PRSM_FORGE_MAX_RPS_PER_REQUESTER": "2"},
        ):
            # Pin the bucket's clock. /compute/forge fetches this SAME
            # process-global bucket (name="forge"; the rate is unchanged, so
            # get_or_build_bucket hands back this instance rather than
            # rebuilding) — freezing _now here means ZERO refill mid-test.
            #
            # sp1418: without this the assertion silently required all three
            # round-trips to finish inside 0.5s of WALL CLOCK. burst=2 refills
            # at 2 tok/sec, so tokens_at_req3 = 2*(t3-t1); a >=0.5s stall on a
            # loaded CI runner refills a token, the 3rd request sails past the
            # limiter into the agent_forge-is-None 503, and the test fails
            # "assert 503 == 429" — which is exactly what CI hit. Freezing the
            # clock also STRENGTHENS the test: it now proves burst-then-429
            # with no refill possible at all.
            frozen = time.time()
            bucket = get_or_build_bucket(2.0, name="forge")
            bucket._now = lambda: frozen
            client = _client(node)
            for _ in range(2):
                resp = client.post(
                    "/compute/forge",
                    json={"query": "test"},
                )
                # Through to 503 (agent_forge None) — rate-cap
                # not exceeded.
                assert resp.status_code == 503
            # Fail LOUD if the endpoint stopped using the bucket we froze
            # (renamed bucket / changed rate parsing), rather than silently
            # decaying back into the wall-clock race above.
            assert bucket._buckets, (
                "endpoint did not consume the frozen 'forge' bucket — this "
                "test is no longer measuring what it thinks it is"
            )
            # Third hits 429.
            resp = client.post(
                "/compute/forge",
                json={"query": "test"},
            )
            assert resp.status_code == 429
            assert "rate" in resp.json()["detail"].lower()

    def test_no_env_disables_limit(self):
        """Without PRSM_FORGE_MAX_RPS_PER_REQUESTER set, no rate
        limiting — many requests in a row all reach 503 (agent
        forge missing) without 429."""
        node = _node_minimal()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_FORGE_MAX_RPS_PER_REQUESTER", None)
            client = _client(node)
            for _ in range(10):
                resp = client.post(
                    "/compute/forge",
                    json={"query": "test"},
                )
                assert resp.status_code == 503  # never 429

    def test_invalid_env_disables_limit(self):
        """Non-numeric env value falls back to no limiting."""
        node = _node_minimal()
        with patch.dict(
            os.environ,
            {"PRSM_FORGE_MAX_RPS_PER_REQUESTER": "not_a_number"},
        ):
            client = _client(node)
            for _ in range(10):
                resp = client.post(
                    "/compute/forge",
                    json={"query": "test"},
                )
                assert resp.status_code == 503  # never 429
