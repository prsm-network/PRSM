"""Sprint 1425 — collaboration persistence + agent allowances were dead on the DEFAULT ledger.

save_task/save_review/save_query, load_active_*, delete_collab_record, and grant/get/revoke_
agent_allowance were defined ONLY on LocalLedger. But `Node` builds a raw `DAGLedger` by default
(config.ledger_type='dag'), so on every default node:

  * AgentCollaboration._persist_task/_persist_review/_persist_query guard with
    `hasattr(self.ledger, "save_task")` — always False on a raw DAGLedger — so they SILENTLY
    return. Every collaboration task/review/query was in-memory only and LOST on restart
    (_load_active_* also short-circuited on the missing method). AgentCollaboration is constructed
    AND started on the node, and _persist_task is called from 5+ points on the live path.
  * the mounted /agents/{id}/allowance endpoints call node.ledger.grant_agent_allowance /
    revoke_agent_allowance, which raise AttributeError -> HTTP 500 on every default node.

Same drift class as sp966/sp1419. Fixed by moving the surface into LedgerNodeServicesMixin (shared
by BOTH ledgers), so the backends carry it identically by construction. agent_debit stays on
LocalLedger (needs _debit_locked/_write_lock; zero production callers).
"""
from __future__ import annotations

import re

import pytest

from prsm.node.dag_ledger import DAGLedger
from prsm.node.ledger_node_services import LedgerNodeServicesMixin
from prsm.node.local_ledger import LocalLedger

_COLLAB_ALLOWANCE_METHODS = [
    "save_task", "save_review", "save_query",
    "load_active_tasks", "load_active_reviews", "load_active_queries",
    "delete_collab_record",
    "grant_agent_allowance", "get_agent_allowance", "revoke_agent_allowance",
]


@pytest.fixture
async def dag_ledger(tmp_path):
    """A REAL DAGLedger — exactly what Node builds on the default config."""
    led = DAGLedger(str(tmp_path / "dag.db"), verify_signatures=False)
    await led.initialize()
    yield led


@pytest.fixture
async def local_ledger(tmp_path):
    led = LocalLedger(str(tmp_path / "local.db"))
    await led.initialize()
    yield led


class TestCollabPersistenceOnTheDefaultLedger:
    async def test_task_round_trips_on_a_real_dag_ledger(self, dag_ledger):
        """Pre-1425 this raised AttributeError; AgentCollaboration swallowed it and persisted
        nothing, losing the task on restart."""
        await dag_ledger.save_task({
            "task_id": "t1", "requester_agent_id": "a1", "requester_node_id": "n1",
            "title": "hello", "status": "open", "created_at": 100,
        })
        active = await dag_ledger.load_active_tasks()
        assert [t["task_id"] for t in active] == ["t1"], (
            "a task saved to the default ledger did not survive a reload — collaboration state is "
            "lost on every restart"
        )
        assert active[0]["title"] == "hello"

        # completed tasks drop out of the active set; delete removes the row.
        await dag_ledger.save_task({
            "task_id": "t1", "requester_agent_id": "a1", "requester_node_id": "n1",
            "status": "completed", "created_at": 100,
        })
        assert await dag_ledger.load_active_tasks() == []
        await dag_ledger.delete_collab_record("collab_tasks", "task_id", "t1")

    async def test_review_and_query_round_trip_on_a_real_dag_ledger(self, dag_ledger):
        await dag_ledger.save_review({
            "review_id": "r1", "submitter_agent_id": "a1", "submitter_node_id": "n1",
            "status": "pending", "created_at": 1,
        })
        await dag_ledger.save_query({
            "query_id": "q1", "requester_agent_id": "a1", "requester_node_id": "n1",
            "max_responses": 5, "responses": [], "created_at": 1,
        })
        assert [r["review_id"] for r in await dag_ledger.load_active_reviews()] == ["r1"]
        assert [q["query_id"] for q in await dag_ledger.load_active_queries()] == ["q1"]


class TestAgentAllowanceOnTheDefaultLedger:
    async def test_grant_get_revoke_round_trip_on_a_real_dag_ledger(self, dag_ledger):
        """Pre-1425 the /agents/{id}/allowance endpoints 500'd here (AttributeError)."""
        await dag_ledger.grant_agent_allowance("principal-1", "agent-1", 10.0, epoch_hours=24.0)
        got = await dag_ledger.get_agent_allowance("agent-1")
        assert got is not None and got["allowance"] == 10.0 and got["remaining"] == 10.0
        assert got["revoked"] is False

        assert await dag_ledger.revoke_agent_allowance("principal-1", "agent-1") is True
        assert (await dag_ledger.get_agent_allowance("agent-1"))["revoked"] is True
        # unknown agent -> None (not a crash)
        assert await dag_ledger.get_agent_allowance("nope") is None


class TestBothBackendsBehaveIdentically:
    async def test_allowance_behaviour_matches_across_ledgers(self, dag_ledger, local_ledger):
        for led in (dag_ledger, local_ledger):
            await led.grant_agent_allowance("p", "a", 7.5)
            got = await led.get_agent_allowance("a")
            assert (got["allowance"], got["remaining"], got["revoked"]) == (7.5, 7.5, False), (
                f"{type(led).__name__} diverged on allowance behaviour"
            )

    def test_all_ten_methods_resolve_to_the_shared_mixin(self):
        """Not just "present" — present via the SINGLE source, so the two can never drift."""
        for m in _COLLAB_ALLOWANCE_METHODS:
            for led_cls in (DAGLedger, LocalLedger):
                owner = getattr(led_cls, m).__qualname__.split(".")[0]
                assert owner == "LedgerNodeServicesMixin", (
                    f"{led_cls.__name__}.{m} resolves to {owner}, not the shared mixin — it can "
                    f"drift between the two backends again"
                )


class TestInterfaceConformance:
    """Pin the CLASS of bug: every ledger method the live callers use must exist on BOTH ledgers."""

    def test_every_ledger_method_agent_collaboration_calls_exists_on_both(self):
        src = open("prsm/node/agent_collaboration.py").read()
        called = set(re.findall(r"self\.ledger\.([a-zA-Z_][a-zA-Z0-9_]*)", src))
        assert called, "parsed zero ledger calls — the regex has gone blind"
        missing = {
            cls.__name__: sorted(m for m in called if not hasattr(cls, m))
            for cls in (DAGLedger, LocalLedger)
        }
        assert not any(missing.values()), (
            f"AgentCollaboration calls ledger methods a default-buildable ledger lacks — its "
            f"persistence is silently dead on those nodes: {missing}"
        )

    def test_allowance_api_surface_exists_on_both(self):
        """The /agents/{id}/allowance endpoints call these on node.ledger; both backends need them."""
        for m in ("grant_agent_allowance", "get_agent_allowance", "revoke_agent_allowance"):
            assert hasattr(DAGLedger, m) and hasattr(LocalLedger, m), (
                f"{m} missing on a ledger Node can build — the allowance endpoint 500s there"
            )
