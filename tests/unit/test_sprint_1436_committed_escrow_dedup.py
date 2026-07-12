"""sp1436 (money-path audit #2) — the DURABLE committed-escrow-id ledger.

The gap this closes
-------------------
sp973's dedup lives on the per-batch ``PendingBatch.seen_escrow_ids`` set. That set
protects a receipt *only while its batch is still PENDING* — the moment the batch is
committed on-chain and popped from the accumulator, the set is gone with it. So a
receipt re-delivered AFTER its batch committed (a crash in the commit→discard window,
or an upstream retry that races the commit) sails past the per-batch dedup, builds a
FRESH batch, and gets committed a SECOND time under a distinct on-chain batchId. The
registry has no content-level dedup, so that is a real double-settle: the provider is
paid twice / the requester's escrow is drained twice for one unit of work.

sp1409 already proved the *pending* dedup survives a restart
(``test_seen_escrow_ids_survive_restart``). This file proves the *committed* dedup:
once an escrow id has been settled on-chain, re-delivering it never produces a second
``commit_batch`` — in-process, across a restart, and without over-deduping distinct work.

These are money assertions. Per CLAUDE.md they must never be weakened to pass; a red
here means the product double-settles.
"""
from __future__ import annotations

import hashlib
from base64 import b64encode

import pytest

import prsm.settlement.client as client_mod
from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.settlement.accumulator import AccumulatorConfig, BatchedReceipt, ReceiptAccumulator
from prsm.settlement.client import BatchSettlementClient
from prsm.settlement.state_store import SettlementStateStore
from unittest.mock import AsyncMock


ONE_FTNS = 10**18
PROVIDER_ADDR = "0x" + "b" * 40
REQUESTER_A = "0x" + "a" * 40


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def _make_batched(shard_index: int = 0, value_ftns: int = ONE_FTNS) -> BatchedReceipt:
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
        receipt=receipt,
        requester_address=REQUESTER_A,
        provider_address=PROVIDER_ADDR,
        value_ftns=value_ftns,
        local_escrow_id=f"escrow-{shard_index}",
        tier_slash_rate_bps=0,
        consensus_group_id=b"\x00" * 32,
    )


def _mock_contract():
    """A unique on-chain batch_id per commit — so a double-settle shows up as a SECOND
    commit_batch call (call_count == 2), never a silent overwrite."""
    mock = AsyncMock()
    n = {"i": 0}

    async def commit_default(**kw):
        n["i"] += 1
        return (hashlib.sha256(f"batch-{n['i']}".encode()).digest(), 1700000000 + n["i"])

    mock.commit_batch.side_effect = commit_default
    mock.is_finalizable.return_value = False
    mock.finalize_batch.return_value = None
    mock.get_batch_status.return_value = 1
    mock.find_committed_batch_by_root = None  # no chain-scan recovery
    return mock


def _store(tmp_path):
    return SettlementStateStore(tmp_path / "settlement_state.json")


def _client(store, contract=None, acc=None):
    return BatchSettlementClient(
        acc if acc is not None else ReceiptAccumulator(AccumulatorConfig(count_threshold=1)),
        contract if contract is not None else _mock_contract(),
        PROVIDER_ADDR,
        state_store=store,
    )


# ── 1. In-process: a re-delivery after commit never settles twice ────────────


def test_redelivery_after_commit_is_dropped_no_second_settle(tmp_path):
    contract = _mock_contract()
    c = _client(_store(tmp_path), contract=contract)

    _run(c.accumulate(_make_batched(0)))
    assert len(_run(c.commit_ready_batches(force=True))) == 1
    assert contract.commit_batch.call_count == 1

    # The commit→discard window: the same receipt is re-delivered.
    _run(c.accumulate(_make_batched(0)))
    # It must NOT sit pending as a fresh batch...
    assert c._accumulator.total_receipt_count() == 0, (
        "the re-delivered, already-committed receipt was staged for a SECOND batch"
    )
    # ...and a subsequent commit poll must produce no new on-chain batch.
    assert _run(c.commit_ready_batches(force=True)) == []
    assert contract.commit_batch.call_count == 1, "DOUBLE SETTLE: escrow committed twice"


def test_committed_escrow_id_is_recorded_in_the_durable_ledger(tmp_path):
    """The mechanism, asserted directly: committing promotes the escrow id into the
    durable set that accumulate() consults."""
    c = _client(_store(tmp_path))
    _run(c.accumulate(_make_batched(0)))
    assert "escrow-0" not in c._committed_escrow_ids  # not yet committed
    _run(c.commit_ready_batches(force=True))
    assert "escrow-0" in c._committed_escrow_ids


# ── 2. Across a restart: the ledger is durable ──────────────────────────────


def test_redelivery_after_commit_survives_restart(tmp_path):
    """The real crash shape: commit on c1, process dies, c2 restarts from the same
    state store, and the upstream redelivers the already-committed receipt."""
    store = _store(tmp_path)
    c1 = _client(store)
    _run(c1.accumulate(_make_batched(0)))
    assert len(_run(c1.commit_ready_batches(force=True))) == 1

    contract2 = _mock_contract()
    c2 = _client(store, contract=contract2)  # "restart" over the same store
    assert "escrow-0" in c2._committed_escrow_ids, "the durable ledger did not survive restart"

    _run(c2.accumulate(_make_batched(0)))  # re-delivery after restart
    assert c2._accumulator.total_receipt_count() == 0
    assert _run(c2.commit_ready_batches(force=True)) == []
    contract2.commit_batch.assert_not_called()


# ── 3. No over-dedup: distinct work still settles ───────────────────────────


def test_distinct_escrow_id_still_settles_after_a_commit(tmp_path):
    """The ledger must reject only RE-delivered ids, never a genuinely new receipt."""
    contract = _mock_contract()
    c = _client(_store(tmp_path), contract=contract)

    _run(c.accumulate(_make_batched(0)))
    assert len(_run(c.commit_ready_batches(force=True))) == 1

    _run(c.accumulate(_make_batched(1)))  # different escrow id → real new work
    assert len(_run(c.commit_ready_batches(force=True))) == 1
    assert contract.commit_batch.call_count == 2, "a distinct receipt was wrongly deduped"


def test_receipt_without_escrow_id_is_not_deduped(tmp_path):
    """A receipt with no local_escrow_id can't be content-deduped; it must pass through
    (the dedup key is the escrow id, and an empty id is never recorded)."""
    contract = _mock_contract()
    c = _client(_store(tmp_path), contract=contract)

    br = BatchedReceipt(
        receipt=_make_batched(0).receipt,
        requester_address=REQUESTER_A,
        provider_address=PROVIDER_ADDR,
        value_ftns=ONE_FTNS,
        local_escrow_id="",
        tier_slash_rate_bps=0,
        consensus_group_id=b"\x00" * 32,
    )
    _run(c.accumulate(br))
    assert len(_run(c.commit_ready_batches(force=True))) == 1
    assert "" not in c._committed_escrow_ids  # empty id never recorded


# ── 4. Bounded FIFO: the ledger cannot grow without limit ───────────────────


def test_ledger_is_bounded_fifo_evicting_oldest(tmp_path, monkeypatch):
    """The ledger caps at _MAX_COMMITTED_ESCROW_IDS and evicts the OLDEST id first.
    Eviction is far past any real re-delivery window, so it can't re-open the hole in
    practice — but the bound must exist, or a long-lived node OOMs."""
    monkeypatch.setattr(client_mod, "_MAX_COMMITTED_ESCROW_IDS", 3)
    c = _client(_store(tmp_path))

    for i in range(5):  # commit escrow-0..escrow-4, cap = 3
        _run(c.accumulate(_make_batched(i)))
        _run(c.commit_ready_batches(force=True))

    assert list(c._committed_escrow_ids) == ["escrow-2", "escrow-3", "escrow-4"]
    assert len(c._committed_escrow_ids) == 3


# ── 5. Backward compatibility: an older state file loads ────────────────────


def test_state_without_committed_escrow_ids_key_loads_empty(tmp_path):
    """A node upgrading across sp1436 has an on-disk state that predates the key. It must
    load (the key is additive/optional, _STATE_VERSION unchanged) with an empty ledger —
    NOT hard-fail via SettlementStateCorruptError."""
    store = _store(tmp_path)
    c1 = _client(store)
    _run(c1.accumulate(_make_batched(0)))  # writes a state file WITHOUT the sp1436 key…

    # …simulate the pre-upgrade file by stripping the key back out.
    raw = store.load()
    assert raw is not None
    raw.pop("committed_escrow_ids", None)
    store.save(raw)

    c2 = _client(store)  # must not raise
    assert list(c2._committed_escrow_ids) == []
    # And it still behaves — a genuinely new commit records into the (now-empty) ledger.
    _run(c2.commit_ready_batches(force=True))
    assert "escrow-0" in c2._committed_escrow_ids


def test_committed_escrow_ids_round_trip_is_order_preserving(tmp_path):
    """FIFO eviction depends on insertion order surviving serialize→load."""
    store = _store(tmp_path)
    c1 = _client(store)
    for i in range(3):
        _run(c1.accumulate(_make_batched(i)))
        _run(c1.commit_ready_batches(force=True))
    assert list(c1._committed_escrow_ids) == ["escrow-0", "escrow-1", "escrow-2"]

    c2 = _client(store)
    assert list(c2._committed_escrow_ids) == ["escrow-0", "escrow-1", "escrow-2"]
