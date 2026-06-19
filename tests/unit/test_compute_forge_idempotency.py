"""POST /compute/forge — Idempotency-Key header handling.

Operators retrying a failed POST due to network blip must not
double-charge their escrow. The forge endpoint reads the
optional ``Idempotency-Key`` header and, when seen previously,
returns the cached job's status without locking a new escrow
or running compute.

Tests focus on the entry-level idempotency check + the
register-on-write path. The full forge pipeline is mocked so
these tests don't require a real Ring 1-10 stack.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from prsm.node.api import create_api_app
from prsm.node.job_history import (
    JobHistoryRecord, JobHistoryStore, JobStatus,
)


def _node(*, with_history=True, with_agent_forge=False):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node._job_history = JobHistoryStore() if with_history else None
    if not with_agent_forge:
        node.agent_forge = None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def _forge_endpoint(app):
    """The /compute/forge handler coroutine. We call it DIRECTLY in
    the concurrency tests below: the conftest `mock_http_requests`
    autouse fixture patches httpx.AsyncClient, so a TestClient /
    ASGITransport request never reaches the ASGI app — calling the
    endpoint coroutine bypasses HTTP entirely while still exercising
    the real event-loop interleaving at the create_escrow await."""
    for r in app.routes:
        if (
            getattr(r, "path", None) == "/compute/forge"
            and "POST" in (getattr(r, "methods", None) or set())
        ):
            return r.endpoint
    raise AssertionError("/compute/forge route not found")


def _make_forge_request():
    """A minimal Starlette Request for direct endpoint invocation
    (the handler reads request.client for the rate-limit key)."""
    scope = {
        "type": "http", "method": "POST", "path": "/compute/forge",
        "headers": [], "client": ("127.0.0.1", 55555),
        "query_string": b"", "scheme": "http", "server": ("t", 80),
    }

    async def _receive():
        return {
            "type": "http.request",
            "body": b'{"query":"q","budget_ftns":10.0}',
            "more_body": False,
        }
    return Request(scope, _receive)


# ──────────────────────────────────────────────────────────────────────
# Idempotent replay — header present + key seen before
# ──────────────────────────────────────────────────────────────────────


class TestIdempotentReplay:
    def test_returns_cached_job_when_key_seen_before(self):
        """Pre-seed JobHistoryStore + idempotency index. Forge POST
        with the same Idempotency-Key returns the cached job
        without invoking agent_forge (which is None — so any
        execution attempt would 503)."""
        node = _node()
        # Pre-seed a completed job with idempotency mapping.
        completed = JobHistoryRecord(
            job_id="forge-cached-abc",
            query="prior query",
            status=JobStatus.COMPLETED,
            started_at=time.time() - 10,
            completed_at=time.time(),
            response="prior response",
        )
        node._job_history.put_with_idempotency(
            completed, idempotency_key="abc-123",
        )

        # Forge with same key → cache hit, no execute.
        # Even though agent_forge is None (would 503 normally),
        # the idempotency check returns BEFORE that 503.
        resp = _client(node).post(
            "/compute/forge",
            json={"query": "anything"},
            headers={"Idempotency-Key": "abc-123"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "idempotent_replay"
        assert body["job_id"] == "forge-cached-abc"
        assert body["history"]["status"] == "completed"
        assert body["history"]["response"] == "prior response"

    def test_no_header_falls_through_to_normal_forge(self):
        """Without Idempotency-Key header, the endpoint proceeds
        as v1 — agent_forge being None hits the normal 503."""
        node = _node()
        resp = _client(node).post(
            "/compute/forge",
            json={"query": "test"},
            # No Idempotency-Key header.
        )
        assert resp.status_code == 503
        assert "Agent forge not initialized" in resp.json()["detail"]

    def test_unknown_key_falls_through_to_normal_forge(self):
        """Idempotency-Key present but key never seen → fall
        through to normal forge logic."""
        node = _node()
        # No pre-seeded mapping for this key.
        resp = _client(node).post(
            "/compute/forge",
            json={"query": "test"},
            headers={"Idempotency-Key": "never-seen-key"},
        )
        # Hits the 503 since agent_forge is None — proves we
        # didn't short-circuit.
        assert resp.status_code == 503

    def test_no_history_store_disables_idempotency(self):
        """Without JobHistoryStore wired, idempotency check is
        skipped (no cache to consult). Falls through to normal
        forge logic."""
        node = _node(with_history=False)
        resp = _client(node).post(
            "/compute/forge",
            json={"query": "test"},
            headers={"Idempotency-Key": "abc-123"},
        )
        # 503 from agent_forge None.
        assert resp.status_code == 503


# ──────────────────────────────────────────────────────────────────────
# Robustness
# ──────────────────────────────────────────────────────────────────────


class TestIdempotencyRobustness:
    def test_lookup_raising_falls_through(self):
        """If lookup_by_idempotency_key raises (corrupted index,
        disk error), the endpoint must NOT 500 — log + continue
        with normal forge logic."""
        node = _node()
        node._job_history.lookup_by_idempotency_key = MagicMock(
            side_effect=RuntimeError("index corrupt"),
        )
        resp = _client(node).post(
            "/compute/forge",
            json={"query": "test"},
            headers={"Idempotency-Key": "abc-123"},
        )
        # Falls through to normal 503 (agent_forge None).
        assert resp.status_code == 503

    def test_cached_record_pointer_dangling_returns_normal_forge(self):
        """Index points to a job_id but the record is LRU-evicted +
        not on disk — treat as cache miss + proceed normally."""
        node = _node()
        # Manually insert a dangling index entry.
        node._job_history._idempotency_index["abc-123"] = "forge-evicted"
        # No record actually present in store.
        resp = _client(node).post(
            "/compute/forge",
            json={"query": "test"},
            headers={"Idempotency-Key": "abc-123"},
        )
        # Falls through; not a cache hit since record is missing.
        assert resp.status_code == 503


# ──────────────────────────────────────────────────────────────────────
# Sp1177 — concurrent same-key requests must not double-lock escrow
# ──────────────────────────────────────────────────────────────────────


class TestConcurrentIdempotency:
    """Sp1177 — two CONCURRENT POSTs with the same Idempotency-Key
    must lock escrow + run compute exactly ONCE. Pre-sp1177 the key
    was registered only AFTER `await create_escrow`, so two
    concurrent requests both missed the cache and both locked an
    escrow (2x charge for one logical request). The fix reserves the
    key SYNCHRONOUSLY before the create_escrow await, so the second
    request replays instead of charging."""

    async def test_concurrent_same_key_locks_escrow_once(self):
        node = MagicMock()
        node.identity.node_id = "test-node"
        node._job_history = JobHistoryStore()
        # agent_forge present (passes the early not-initialized 503),
        # WITHOUT dispatch_query so the legacy run() path is taken; run
        # raises right after escrow so the test doesn't need the full
        # Ring 1-10 stack — we only assert the escrow-lock COUNT.
        node.agent_forge = MagicMock(spec=["run"])
        node.agent_forge.run = AsyncMock(
            side_effect=RuntimeError("halt after escrow (test)"),
        )
        # No per-requester rate limiter (so the 2nd concurrent request
        # isn't throttled before reaching the escrow path).
        node._forge_rate_limiter = None

        locked = []

        async def _create_escrow(*, job_id, amount, requester_id):
            locked.append(job_id)
            # Park so the concurrent duplicate is scheduled and observes
            # the reserved key before this call returns.
            await asyncio.sleep(0.05)
            e = MagicMock()
            e.escrow_id = "esc-" + job_id
            e.amount = amount
            return e

        node._payment_escrow = MagicMock()
        node._payment_escrow.create_escrow = _create_escrow
        node._payment_escrow.refund_escrow = AsyncMock(return_value=True)

        app = create_api_app(node, enable_security=False)
        ep = _forge_endpoint(app)
        results = await asyncio.gather(
            ep(
                request=_make_forge_request(),
                body={"query": "q", "budget_ftns": 10.0},
                idempotency_key="dup-key",
            ),
            ep(
                request=_make_forge_request(),
                body={"query": "q", "budget_ftns": 10.0},
                idempotency_key="dup-key",
            ),
            return_exceptions=True,
        )

        # The fix: exactly ONE escrow locked for the duplicate pair.
        assert len(locked) == 1, (
            f"escrow locked {len(locked)} times for one logical request "
            f"(double-charge regression): {locked}"
        )
        # And exactly one call took the idempotent-replay path (the
        # duplicate that observed the reserved key); the other reached
        # compute (and raised the test's halt-after-escrow sentinel).
        replayed = sum(
            1 for r in results
            if isinstance(r, dict)
            and r.get("status") == "idempotent_replay"
        )
        assert replayed == 1, (
            f"expected exactly one idempotent_replay; got results={results}"
        )

    async def test_distinct_keys_lock_separate_escrows(self):
        """Control: two concurrent requests with DIFFERENT keys are
        NOT duplicates — both legitimately lock their own escrow."""
        node = MagicMock()
        node.identity.node_id = "test-node"
        node._job_history = JobHistoryStore()
        node.agent_forge = MagicMock(spec=["run"])
        node.agent_forge.run = AsyncMock(
            side_effect=RuntimeError("halt after escrow (test)"),
        )
        node._forge_rate_limiter = None

        locked = []

        async def _create_escrow(*, job_id, amount, requester_id):
            locked.append(job_id)
            await asyncio.sleep(0.05)
            e = MagicMock()
            e.escrow_id = "esc-" + job_id
            return e

        node._payment_escrow = MagicMock()
        node._payment_escrow.create_escrow = _create_escrow
        node._payment_escrow.refund_escrow = AsyncMock(return_value=True)

        app = create_api_app(node, enable_security=False)
        ep = _forge_endpoint(app)
        await asyncio.gather(
            ep(
                request=_make_forge_request(),
                body={"query": "q", "budget_ftns": 10.0},
                idempotency_key="key-A",
            ),
            ep(
                request=_make_forge_request(),
                body={"query": "q", "budget_ftns": 10.0},
                idempotency_key="key-B",
            ),
            return_exceptions=True,
        )

        assert len(locked) == 2, (
            f"distinct keys must each lock an escrow; got {locked}"
        )
