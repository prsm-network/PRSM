"""sp1448 — per-stage settlement: a commit that BROADCASTS but does not cleanly confirm
(OnChainPendingError / BroadcastFailedError) must NOT double-settle on the next drain, and a
commit that MINES-AND-REVERTS must stay retryable (not stranded).

The sp1446 test proved the crash-AFTER-clean-commit re-drain is dropped by the sp1436 durable
`_committed_escrow_ids` ledger. But that ledger was armed ONLY on the clean-commit success tail
(client.py, post-`_commit_one`). The adversarial money-audit (workflow w754cz4mr) found the
uncertain-fate windows uncovered:

  * OnChainPendingError (receipt-wait timed out; the tx MAY mine) — the receipts are quarantined but
    the escrow id is NOT recorded, so the still-staged task re-drains and commits a SECOND on-chain
    batch for the same share → double escrow release.
  * BroadcastFailedError (send threw; the tx MAY have landed) — same.
  * A crash between the on-chain commit and the success-tail persist — same, across a restart.
  * The per-stage cycle never ran recover_committing_intents / reconcile_pending_commits, so a
    landed-but-unconfirmed batch was never adopted into `_tracked` → escrow stranded, never finalized.

sp1448 arms the ledger BEFORE the irreversible broadcast (same durable write as the WAL intent),
UNARMS it only on a confirmed atomic revert (safe to retry), discards a staged task the client
already owns (stops the receiver store re-injecting it), and wires the recovery/adoption phases into
the per-stage cycle. This test proves all of that on a REAL BatchSettlementClient.

Money assertions — never weaken.
"""
from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError,
    OnChainPendingError,
    OnChainRevertedError,
)
from prsm.settlement.client_wiring import run_per_stage_commit_cycle
from prsm.settlement.per_stage_receiver_store import (
    commit_staged_task,
    drain_and_commit_staged,
)

# Reuse the fully-signed staged-task + real-client fixtures from the sp1446 suite.
from tests.unit.test_sprint_1446_per_stage_double_settle_closed import (
    _client,
    _mock_contract,
    _staged_task,
)


def _run(coro):
    return asyncio.run(coro)


def _fail_first_then_ok_contract(exc):
    """First commit_batch raises `exc` (the tx may/does land); later calls return a DISTINCT batch id
    (so any second on-chain commit for the same share is observable as commit_batch.call_count == 2)."""
    mock = AsyncMock()
    n = {"i": 0}

    async def commit(**kw):
        n["i"] += 1
        if n["i"] == 1:
            raise exc
        return (hashlib.sha256(f"b{n['i']}".encode()).digest(), 1_700_000_000 + n["i"])

    mock.commit_batch.side_effect = commit
    mock.is_finalizable.return_value = False
    mock.finalize_batch.return_value = None
    mock.get_batch_status.return_value = 1
    mock.find_committed_batch_by_root = None
    return mock


# ── #1/#4: broadcast-but-unconfirmed commit must not double-settle on re-drain ──


def test_onchain_pending_commit_does_not_double_settle_on_redrain(tmp_path):
    store, task = _staged_task(tmp_path)
    contract = _fail_first_then_ok_contract(OnChainPendingError("receipt wait timed out", tx_hash="0x" + "de" * 32))
    client = _client(tmp_path / "state.json", contract)

    # Cycle 1: broadcast succeeds but the receipt could not be confirmed. The tx MAY (here does) land.
    r1 = _run(commit_staged_task(task, client=client))
    assert not r1.committed  # quarantined, not a clean commit
    assert contract.commit_batch.call_count == 1
    # sp1448: the id is armed pre-broadcast even though the confirm failed.
    assert client.has_committed_escrow_id(task.local_escrow_id)

    # Cycle 2: the still-staged task re-drains. accumulate() must DROP it — no second on-chain batch.
    r2 = _run(commit_staged_task(task, client=client))
    assert contract.commit_batch.call_count == 1, (
        "a broadcast-but-unconfirmed share was re-committed as a second on-chain batch — DOUBLE SETTLE")
    assert not r2.committed


def test_broadcast_failed_commit_does_not_double_settle_on_redrain(tmp_path):
    store, task = _staged_task(tmp_path)
    contract = _fail_first_then_ok_contract(BroadcastFailedError("send_raw response dropped"))
    client = _client(tmp_path / "state.json", contract)

    r1 = _run(commit_staged_task(task, client=client))
    assert not r1.committed
    assert contract.commit_batch.call_count == 1
    assert client.has_committed_escrow_id(task.local_escrow_id)

    r2 = _run(commit_staged_task(task, client=client))
    assert contract.commit_batch.call_count == 1, (
        "a broadcast-failed (maybe-landed) share was re-committed as a second on-chain batch — DOUBLE SETTLE")
    assert not r2.committed


def test_pending_commit_dedup_survives_a_restart(tmp_path):
    """The pre-broadcast arming is DURABLE: a fresh client rehydrated from the same state store drops
    the re-delivery, so a crash between broadcast and the success-tail persist cannot double-settle."""
    store, task = _staged_task(tmp_path)
    state_path = tmp_path / "state.json"

    contract1 = _fail_first_then_ok_contract(OnChainPendingError("timeout", tx_hash="0x" + "de" * 32))
    _run(commit_staged_task(task, client=_client(state_path, contract1)))
    assert contract1.commit_batch.call_count == 1

    # "restart": a brand-new client + a fresh (always-succeeds) contract over the SAME persisted state.
    contract2 = _mock_contract()
    r = _run(commit_staged_task(task, client=_client(state_path, contract2)))
    assert not r.committed
    assert contract2.commit_batch.call_count == 0, (
        "after a restart, a broadcast-but-unconfirmed share committed a second on-chain batch — DOUBLE SETTLE")


# ── the regression guard: a CONFIRMED revert must stay retryable, not stranded ──


def test_reverted_commit_is_retryable_not_stranded(tmp_path):
    store, task = _staged_task(tmp_path)
    contract = _fail_first_then_ok_contract(OnChainRevertedError("execution reverted"))
    client = _client(tmp_path / "state.json", contract)

    # Cycle 1: the commit mines and atomically reverts — NOTHING was committed. Pre-broadcast arming
    # must be UNDONE (else the share is permanently blocked from re-committing = stranded).
    r1 = _run(commit_staged_task(task, client=client))
    assert not r1.committed
    assert not client.has_committed_escrow_id(task.local_escrow_id), (
        "a reverted (never-landed) commit left the share armed — it can never retry (STRANDED FUNDS)")

    # Cycle 2: the retry commits the share exactly once.
    r2 = _run(commit_staged_task(task, client=client))
    assert r2.committed, r2.reason
    assert contract.commit_batch.call_count == 2


# ── #1 belt-and-suspenders: a share the client already OWNS is discarded, not re-injected ──


def test_owned_share_task_is_discarded_not_re_injected(tmp_path):
    store, task = _staged_task(tmp_path)
    contract = _mock_contract()
    client = _client(tmp_path / "state.json", contract)
    # Simulate a prior broadcast-but-unconfirmed commit that armed the ledger for this share.
    client._record_committed_escrow_id(task.local_escrow_id)
    assert client.has_committed_escrow_id(task.local_escrow_id)

    # Drain: commit_staged_task drops the re-delivery at accumulate (records=[], committed=False) —
    # but the client OWNS the id, so drain must DISCARD the staged task to stop it re-injecting.
    _run(drain_and_commit_staged(store, client_for_node=lambda _n: client))
    remaining = [s.local_escrow_id for s in store.all_staged()]
    assert task.local_escrow_id not in remaining, (
        "an already-owned (committed/in-flight) share stayed staged — it will re-inject and double-settle")
    assert contract.commit_batch.call_count == 0, "an owned share fired a redundant second on-chain commit"


# ── #3: the per-stage cycle must run the recover/reconcile adoption phases ──


def test_per_stage_commit_cycle_runs_recovery_phases(monkeypatch):
    monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", "1")
    recover = AsyncMock(return_value=0)
    reconcile = AsyncMock(return_value=0)
    client = SimpleNamespace(
        recover_committing_intents=recover,
        reconcile_pending_commits=reconcile,
    )
    store = SimpleNamespace(all_staged=lambda: [])  # nothing staged — recovery must STILL run
    node = SimpleNamespace(
        _onchain_per_stage_settlement_client=client,
        _settlement_per_stage_receiver_store=store,
    )
    r = _run(run_per_stage_commit_cycle(node))
    recover.assert_awaited_once()
    reconcile.assert_awaited_once()
    assert r["per_stage_commit"] == "ok:nothing-staged"
