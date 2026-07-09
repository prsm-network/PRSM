"""Sprint 1040 — commit-intent write-ahead log + chain-scan recovery (brick 2.6).

Closes the HIGH finding from the sp1039 review: there was a strand window in
commit_ready_batches where the on-chain commitBatch CONFIRMS (escrow locked)
before the local _persist() records it in _tracked. A crash in that window left
the batch committed on chain but invisible to the client forever — never
finalized, escrow STRANDED — with NO recovery path (reconcile_pending_commits
needs a quarantined tx_hash; a clean confirm produces none).

Brick 2.6:
  1. Write a durable commit-INTENT (keyed by merkle_root) BEFORE the irreversible
     broadcast. On success it is cleared as the batch is promoted to _tracked.
  2. recover_committing_intents() scans the chain (BatchCommitted, provider
     indexed) for each leftover intent's merkle_root; if the commit actually
     landed it is ADOPTED into _tracked (so finalize proceeds) and the intent is
     cleared. If it did not land, the intent is kept (conservative — never assume
     a possibly-pending commit failed).

This suite drives the intent lifecycle, the crash-recovery adoption (the actual
#2 fix), the conservative not-found behavior, durability of intents across a
restart, graceful no-op when the contract lacks the scan method, and the poll
cycle running the recover phase first.
"""
from __future__ import annotations

import hashlib
from base64 import b64encode
from unittest.mock import AsyncMock

import pytest

from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.settlement.accumulator import (
    AccumulatorConfig,
    BatchedReceipt,
    ReceiptAccumulator,
)
from prsm.settlement.client import BatchSettlementClient


ONE_FTNS = 10**18
PROVIDER_ADDR = "0x" + "b" * 40
REQUESTER_A = "0x" + "a" * 40


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def _make_batched(shard_index: int = 0) -> BatchedReceipt:
    receipt = ShardExecutionReceipt(
        job_id=f"job-{shard_index}",
        shard_index=shard_index,
        provider_id="provider-id-string",
        provider_pubkey_b64=b64encode(b"pubkey-raw").decode(),
        output_hash=hashlib.sha256(f"out-{shard_index}".encode()).hexdigest(),
        executed_at_unix=1700000000 + shard_index,
        signature=b64encode(b"sig-raw").decode(),
    )
    return BatchedReceipt(
        receipt=receipt, requester_address=REQUESTER_A,
        provider_address=PROVIDER_ADDR, value_ftns=ONE_FTNS,
        local_escrow_id=f"escrow-{shard_index}",
    )


def _acc():
    return ReceiptAccumulator(AccumulatorConfig(count_threshold=1))


def _store(tmp_path):
    from prsm.settlement.state_store import SettlementStateStore
    return SettlementStateStore(tmp_path / "settlement_state.json")


def _contract(commit_return=None):
    m = AsyncMock()
    if commit_return is None:
        n = {"i": 0}

        async def commit(**kw):
            n["i"] += 1
            return (hashlib.sha256(f"b{n['i']}".encode()).digest(), 1700000000 + n["i"])
        m.commit_batch.side_effect = commit
    else:
        m.commit_batch.return_value = commit_return
    m.is_finalizable.return_value = False
    m.finalize_batch.return_value = None
    m.get_batch_status.return_value = 1
    return m


# ── Intent lifecycle ──────────────────────────────────────────────────


def test_intent_is_persisted_before_broadcast(tmp_path):
    """The intent must be on disk BEFORE commit_batch is invoked, else a crash
    during the broadcast has nothing to recover from."""
    store = _store(tmp_path)
    seen = {}

    contract = _contract()

    async def commit_checking(**kw):
        # at broadcast time the intent for this root must already be tracked
        seen["intent_present"] = len(client._committing) == 1
        return (hashlib.sha256(b"bid").digest(), 1700000999)
    contract.commit_batch.side_effect = commit_checking

    client = BatchSettlementClient(_acc(), contract, PROVIDER_ADDR, state_store=store)
    _run(client.accumulate(_make_batched()))
    _run(client.commit_ready_batches())
    assert seen["intent_present"] is True


def test_successful_commit_clears_intent(tmp_path):
    store = _store(tmp_path)
    client = BatchSettlementClient(_acc(), _contract(), PROVIDER_ADDR, state_store=store)
    _run(client.accumulate(_make_batched()))
    _run(client.commit_ready_batches())
    assert client.committing_intents() == []
    # and a restart sees no leftover intent
    c2 = BatchSettlementClient(_acc(), _contract(), PROVIDER_ADDR, state_store=store)
    assert c2.committing_intents() == []


def test_crash_between_commit_and_persist_is_recovered_by_chain_scan(tmp_path):
    """THE #2 fix: commit confirms on chain, the promote-persist 'crashes', and a
    restart recovers the committed batch by scanning the chain for its root."""
    store = _store(tmp_path)
    landed_bid = hashlib.sha256(b"landed").digest()
    contract = _contract(commit_return=(landed_bid, 1700001111))

    # store.save: 1st call (intent) ok; 2nd call (promote) raises -> 'crash'
    real_save = store.save
    calls = {"n": 0}

    def flaky(state):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("crash before promote persist")
        return real_save(state)
    c1 = BatchSettlementClient(_acc(), contract, PROVIDER_ADDR, state_store=store)
    _run(c1.accumulate(_make_batched()))
    # sp1409 — accumulate() now persists the receipt too, so install the flaky save
    # AFTER it: `calls` must count only the COMMIT path's saves (1 = the sp1040 intent
    # write-ahead, 2 = the post-commit promote, which is the crash this test simulates).
    store.save = flaky  # type: ignore[assignment]
    with pytest.raises(OSError):
        _run(c1.commit_ready_batches())

    # On disk now: the intent is recorded, the batch is NOT in _tracked.
    store2 = _store(tmp_path)
    recover_contract = _contract()
    recover_contract.find_committed_batch_by_root = AsyncMock(
        return_value=(landed_bid, 1700001111))
    c2 = BatchSettlementClient(_acc(), recover_contract, PROVIDER_ADDR, state_store=store2)
    assert len(c2.committing_intents()) == 1      # the leftover intent survived
    assert c2.get_tracked(landed_bid) is None     # not yet adopted

    adopted = _run(c2.recover_committing_intents())
    assert adopted == 1
    assert c2.get_tracked(landed_bid) is not None  # STRAND AVOIDED
    assert c2.committing_intents() == []           # intent cleared after adoption


def test_recover_keeps_intent_when_not_landed(tmp_path):
    """Conservative: a scan that finds nothing must NOT drop the intent (the
    commit could still be pending; assuming-failed could later double-commit)."""
    store = _store(tmp_path)
    contract = _contract(commit_return=(hashlib.sha256(b"x").digest(), 1))
    real_save = store.save
    calls = {"n": 0}

    def flaky(state):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("crash")
        return real_save(state)
    c1 = BatchSettlementClient(_acc(), contract, PROVIDER_ADDR, state_store=store)
    _run(c1.accumulate(_make_batched()))
    # sp1409 — accumulate() now persists the receipt too, so install the flaky save
    # AFTER it: `calls` must count only the COMMIT path's saves (1 = the sp1040 intent
    # write-ahead, 2 = the post-commit promote, which is the crash this test simulates).
    store.save = flaky  # type: ignore[assignment]
    with pytest.raises(OSError):
        _run(c1.commit_ready_batches())

    store2 = _store(tmp_path)
    rc = _contract()
    rc.find_committed_batch_by_root = AsyncMock(return_value=None)  # not landed
    c2 = BatchSettlementClient(_acc(), rc, PROVIDER_ADDR, state_store=store2)
    assert _run(c2.recover_committing_intents()) == 0
    assert len(c2.committing_intents()) == 1   # kept


def test_recover_is_noop_when_contract_lacks_scan_method(tmp_path):
    """A Protocol contract client without find_committed_batch_by_root must not
    crash recovery — it just can't backfill (returns 0)."""
    store = _store(tmp_path)
    contract = _contract(commit_return=(hashlib.sha256(b"y").digest(), 1))
    real_save = store.save
    calls = {"n": 0}

    def flaky(state):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("crash")
        return real_save(state)
    c1 = BatchSettlementClient(_acc(), contract, PROVIDER_ADDR, state_store=store)
    _run(c1.accumulate(_make_batched()))
    # sp1409 — accumulate() now persists the receipt too, so install the flaky save
    # AFTER it: `calls` must count only the COMMIT path's saves (1 = the sp1040 intent
    # write-ahead, 2 = the post-commit promote, which is the crash this test simulates).
    store.save = flaky  # type: ignore[assignment]
    with pytest.raises(OSError):
        _run(c1.commit_ready_batches())

    store2 = _store(tmp_path)
    bare = AsyncMock(spec=["commit_batch", "is_finalizable", "finalize_batch",
                           "get_batch_status", "get_committed_batch_for_tx"])
    c2 = BatchSettlementClient(_acc(), bare, PROVIDER_ADDR, state_store=store2)
    assert _run(c2.recover_committing_intents()) == 0
    assert len(c2.committing_intents()) == 1   # still there, not lost


def test_intent_key_is_unique_per_accumulator_key_not_just_root(tmp_path):
    """sp1040 review #10 — two ready batches sharing a merkle_root but under
    different accumulator keys must get DISTINCT WAL keys (else the second
    overwrites the first's intent and strands it)."""
    root = b"\x07" * 32
    k_a = (REQUESTER_A, PROVIDER_ADDR, b"\x00" * 32, 0)
    k_b = ("0x" + "e" * 40, PROVIDER_ADDR, b"\x00" * 32, 0)  # different requester
    ka = BatchSettlementClient._intent_key(k_a, root)
    kb = BatchSettlementClient._intent_key(k_b, root)
    assert ka != kb


def test_recovery_scans_by_msg_sender_when_available(tmp_path):
    """sp1040 review #9 — the scan must filter by the address that sent
    commitBatch (msg.sender == contract.address), not the earning provider."""
    store = _store(tmp_path)
    contract = _contract(commit_return=(hashlib.sha256(b"z").digest(), 1))
    real_save = store.save
    calls = {"n": 0}

    def flaky(state):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("crash")
        return real_save(state)
    c1 = BatchSettlementClient(_acc(), contract, PROVIDER_ADDR, state_store=store)
    _run(c1.accumulate(_make_batched()))
    # sp1409 — accumulate() now persists the receipt too, so install the flaky save
    # AFTER it: `calls` must count only the COMMIT path's saves (1 = the sp1040 intent
    # write-ahead, 2 = the post-commit promote, which is the crash this test simulates).
    store.save = flaky  # type: ignore[assignment]
    with pytest.raises(OSError):
        _run(c1.commit_ready_batches())

    store2 = _store(tmp_path)
    rc = _contract()
    rc.address = "0x" + "5" * 40  # the settler key's address (msg.sender)
    rc.find_committed_batch_by_root = AsyncMock(return_value=None)
    c2 = BatchSettlementClient(_acc(), rc, PROVIDER_ADDR, state_store=store2)
    _run(c2.recover_committing_intents())
    # called with msg.sender (contract.address), not intent.provider_address
    called_provider = rc.find_committed_batch_by_root.call_args[0][0]
    assert called_provider == "0x" + "5" * 40


def test_poll_cycle_runs_recover_phase_first(tmp_path):
    """run_settlement_poll_cycle must drive recovery, and before commit (so a
    recovered batch is finalized in the same cycle)."""
    from prsm.settlement.client_wiring import run_settlement_poll_cycle
    order = []
    c = AsyncMock()
    c.recover_committing_intents = AsyncMock(side_effect=lambda: order.append("recover"))
    c.reconcile_pending_commits = AsyncMock(side_effect=lambda: order.append("recon_p"))
    c.commit_ready_batches = AsyncMock(side_effect=lambda: order.append("commit"))
    c.finalize_ready_batches = AsyncMock(side_effect=lambda: order.append("finalize"))
    c.reconcile_finalized = AsyncMock(side_effect=lambda: order.append("recon_f"))
    res = _run(run_settlement_poll_cycle(c))
    assert order[0] == "recover"
    assert order.index("recover") < order.index("commit")
    assert "recover" in res


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
