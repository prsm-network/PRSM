"""Sprint 1038 — settlement commit/finalize/reconcile poll cycle (brick 2).

run_settlement_poll_cycle drives ONE cycle of the on-chain settlement client's
commit/finalize/reconcile methods (which existed but were never invoked). A
background task in node.start() runs it on an interval when settlement is
opted-in. Each phase is isolated (one failing does not block the others) and the
cycle NEVER raises (it runs detached).

Order: reconcile broadcast-but-unconfirmed commits (adopt any that landed) ->
commit ready batches -> finalize past the challenge window -> reconcile
on-chain-finalized.

With the default view-only client (no funded settler key) the commit/finalize
phases raise and are recorded as errors — the cycle is inert until the key
ceremony. DURABLE batch state (brick 2.5) is required before that ceremony.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock


def _client(**over):
    c = MagicMock()
    c.recover_committing_intents = over.get(
        "recover_committing_intents", AsyncMock(return_value=None))
    c.reconcile_pending_commits = over.get(
        "reconcile_pending_commits", AsyncMock(return_value=None))
    c.commit_ready_batches = over.get(
        "commit_ready_batches", AsyncMock(return_value=None))
    c.finalize_ready_batches = over.get(
        "finalize_ready_batches", AsyncMock(return_value=None))
    c.reconcile_finalized = over.get(
        "reconcile_finalized", AsyncMock(return_value=None))
    return c


def _run(client):
    from prsm.settlement.client_wiring import run_settlement_poll_cycle
    return asyncio.run(run_settlement_poll_cycle(client))


def test_runs_all_phases():
    c = _client()
    res = _run(c)
    c.recover_committing_intents.assert_awaited_once()
    c.reconcile_pending_commits.assert_awaited_once()
    c.commit_ready_batches.assert_awaited_once()
    c.finalize_ready_batches.assert_awaited_once()
    c.reconcile_finalized.assert_awaited_once()
    assert set(res.keys()) == {
        "recover", "reconcile_pending", "commit", "finalize", "reconcile_finalized",
    }
    assert all(v == "ok" for v in res.values())


def test_phase_order_is_recover_reconcile_commit_finalize_reconcile():
    order = []
    c = _client(
        recover_committing_intents=AsyncMock(
            side_effect=lambda: order.append("recover")),
        reconcile_pending_commits=AsyncMock(
            side_effect=lambda: order.append("reconcile_pending")),
        commit_ready_batches=AsyncMock(
            side_effect=lambda: order.append("commit")),
        finalize_ready_batches=AsyncMock(
            side_effect=lambda: order.append("finalize")),
        reconcile_finalized=AsyncMock(
            side_effect=lambda: order.append("reconcile_finalized")),
    )
    _run(c)
    assert order == [
        "recover", "reconcile_pending", "commit", "finalize", "reconcile_finalized",
    ]


def test_one_phase_error_does_not_block_others():
    c = _client(commit_ready_batches=AsyncMock(side_effect=RuntimeError("boom")))
    res = _run(c)
    assert res["commit"].startswith("error:")
    # the other three still ran + report ok
    c.reconcile_pending_commits.assert_awaited_once()
    c.finalize_ready_batches.assert_awaited_once()
    c.reconcile_finalized.assert_awaited_once()
    assert res["finalize"] == "ok" and res["reconcile_finalized"] == "ok"


def test_never_raises_when_all_phases_fail():
    c = _client(
        recover_committing_intents=AsyncMock(side_effect=RuntimeError("r")),
        reconcile_pending_commits=AsyncMock(side_effect=RuntimeError("a")),
        commit_ready_batches=AsyncMock(side_effect=RuntimeError("b")),
        finalize_ready_batches=AsyncMock(side_effect=RuntimeError("c")),
        reconcile_finalized=AsyncMock(side_effect=RuntimeError("d")),
    )
    res = _run(c)  # must not raise
    assert all(v.startswith("error:") for v in res.values())


def test_missing_method_is_isolated_not_fatal():
    c = _client()
    del c.finalize_ready_batches   # simulate an incomplete client
    res = _run(c)
    assert res["finalize"].startswith("error:")
    assert res["commit"] == "ok"   # others unaffected


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
