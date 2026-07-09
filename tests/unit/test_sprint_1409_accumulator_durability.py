"""Sprint 1409 — the ReceiptAccumulator's pending receipts must survive a restart.

sp1039 made the three POST-commit dicts durable (_tracked / _pending_commits /
_finalized_ids), reasoning that "on-chain money is ALREADY escrowed for every entry in
these dicts, so losing them on a daemon restart STRANDS that money."

It never covered the stage BEFORE the commit. ``ReceiptAccumulator._pending`` is
in-memory only; ``_serialize_state`` omits it and ``client_wiring`` rebuilds a fresh
empty accumulator on every start. A provider serves jobs, its receipts accumulate toward
the commit thresholds (default: 1000 receipts / 1h / 100 FTNS), and any restart in that
window — a deploy, an OOM, a crash — silently DROPS them. The work was done, the receipt
was signed, and the requester's on-chain escrow is never drawn: the provider is simply
never paid on-chain for up to an hour of earnings.

sp1409 persists the pending batches alongside the post-commit state and rehydrates them.

THE TRAP (why this is not a one-line change): the sp1040 write-ahead log persists a
CommitIntent *while the batch is still in the accumulator* (client.py: `self._committing
[intent_key] = ...; self._persist()`, then broadcast). So a crash in that window leaves
BOTH the intent and the pending batch on disk. Restoring both would let the commit phase
re-commit receipts that `recover_committing_intents` may separately adopt as a landed
batch — a second on-chain batchId for the same receipts, i.e. a DOUBLE SETTLE (the exact
class sp1040/sp1052 were written to prevent).

Invariant: a batch is either in the accumulator OR captured by a commit intent, never
both. The intent wins — it owns recovery for those receipts.
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
from prsm.settlement.state_store import SettlementStateStore


ONE_FTNS = 10**18
PROVIDER_ADDR = "0x" + "b" * 40
REQUESTER_A = "0x" + "a" * 40
GROUP = bytes(range(32))


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def _make_batched(shard_index: int = 0, value_ftns: int = ONE_FTNS,
                  slash_bps: int = 0, group_id: bytes = b"\x00" * 32) -> BatchedReceipt:
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
        tier_slash_rate_bps=slash_bps,
        consensus_group_id=group_id,
    )


def _mock_contract():
    mock = AsyncMock()
    n = {"i": 0}

    async def commit_default(**kw):
        n["i"] += 1
        return (hashlib.sha256(f"batch-{n['i']}".encode()).digest(), 1700000000 + n["i"])

    mock.commit_batch.side_effect = commit_default
    mock.is_finalizable.return_value = False
    mock.finalize_batch.return_value = None
    mock.get_batch_status.return_value = 1
    # No root-scan capability → recover_committing_intents is a no-op (returns 0).
    mock.find_committed_batch_by_root = None
    return mock


def _store(tmp_path):
    return SettlementStateStore(tmp_path / "settlement_state.json")


def _client(store, acc=None, contract=None):
    """A client whose accumulator does NOT auto-fire (default thresholds), so
    receipts sit pending — the window this sprint is about."""
    return BatchSettlementClient(
        acc if acc is not None else ReceiptAccumulator(),
        contract if contract is not None else _mock_contract(),
        PROVIDER_ADDR,
        state_store=store,
    )


def _only_pending(client):
    keys = client._accumulator.pending_keys()
    assert len(keys) == 1, f"expected exactly one pending batch, got {keys}"
    return client._accumulator.peek_batch(keys[0])


# ── 1. The money bug: pending receipts must survive a restart ─────────────


def test_pending_receipts_survive_restart(tmp_path):
    store = _store(tmp_path)
    c1 = _client(store)
    _run(c1.accumulate(_make_batched(0)))
    _run(c1.accumulate(_make_batched(1)))
    assert c1._accumulator.total_receipt_count() == 2

    c2 = _client(store)   # "restart": a fresh client over the same store

    assert c2._accumulator.total_receipt_count() == 2, "the provider's earnings were dropped"
    assert c2._accumulator.total_pending_value_ftns() == 2 * ONE_FTNS


def test_pending_batch_started_at_survives_so_the_time_trigger_still_fires(tmp_path):
    """started_at_unix drives the TIME threshold. If it reset on restart, a restarting
    node would push its commit an hour further out every time."""
    store = _store(tmp_path)
    c1 = _client(store)
    _run(c1.accumulate(_make_batched(0)))
    started = _only_pending(c1).started_at_unix

    c2 = _client(store)

    assert _only_pending(c2).started_at_unix == started


def test_pending_receipt_fields_survive_byte_exact(tmp_path):
    """The merkle leaf is built from these fields — a lossy round-trip would produce a
    different root and break every challenge proof."""
    store = _store(tmp_path)
    c1 = _client(store)
    original = _make_batched(3, value_ftns=7 * ONE_FTNS + 12345, slash_bps=250, group_id=GROUP)
    _run(c1.accumulate(original))

    c2 = _client(store)
    restored = _only_pending(c2).receipts[0]

    assert restored == original          # frozen dataclass → full structural equality
    assert restored.consensus_group_id == GROUP
    assert restored.value_ftns == 7 * ONE_FTNS + 12345
    assert restored.tier_slash_rate_bps == 250
    assert restored.receipt.signature == original.receipt.signature
    assert restored.receipt.output_hash == original.receipt.output_hash


def test_accumulate_persists_immediately(tmp_path):
    """A crash one instruction after accumulate() must not lose the receipt, so the
    write happens on accumulate — not only at commit time."""
    store = _store(tmp_path)
    c1 = _client(store)
    _run(c1.accumulate(_make_batched(0)))

    on_disk = store.load()
    assert on_disk["pending_batches"], "accumulate() did not persist the receipt"
    assert len(on_disk["pending_batches"][0]["receipts"]) == 1


def test_batch_grouping_key_survives_restart(tmp_path):
    """group_id + slash_bps are batch-level on-chain invariants; they key the batch."""
    store = _store(tmp_path)
    c1 = _client(store)
    _run(c1.accumulate(_make_batched(0, slash_bps=250, group_id=GROUP)))

    c2 = _client(store)
    key = c2._accumulator.pending_keys()[0]

    assert key == (REQUESTER_A, PROVIDER_ADDR, GROUP, 250)


def test_seen_escrow_ids_survive_restart(tmp_path):
    """sp973 dedup: re-delivering a receipt after a restart must not double-count value."""
    store = _store(tmp_path)
    c1 = _client(store)
    _run(c1.accumulate(_make_batched(0)))

    c2 = _client(store)
    _run(c2.accumulate(_make_batched(0)))   # same local_escrow_id → no-op

    assert c2._accumulator.total_receipt_count() == 1
    assert c2._accumulator.total_pending_value_ftns() == ONE_FTNS


# ── 2. The trap: never restore a batch a commit intent already owns ──────


def test_restore_skips_a_batch_owned_by_a_commit_intent(tmp_path):
    """Reproduces the sp1040 WAL window: the intent is persisted while the batch is
    still in the accumulator. Restoring both would double-settle."""
    store = _store(tmp_path)
    c1 = _client(store)
    _run(c1.accumulate(_make_batched(0)))

    # Exactly what commit_ready_batches does before the irreversible broadcast:
    ready = c1._accumulator.ready_batches(at_unix=2**62)[0]
    leaf_hashes, root = c1._build_leaves_root(ready)
    c1._committing[c1._intent_key(ready.key, root)] = c1._build_intent(ready, leaf_hashes, root)
    c1._persist()
    # ...crash here, before the batch is popped.

    c2 = _client(store)

    assert c2._committing, "the commit intent must survive (it owns recovery)"
    assert c2._accumulator.total_receipt_count() == 0, (
        "the batch was restored alongside its commit intent — the commit phase would "
        "re-commit these receipts as a distinct on-chain batchId: DOUBLE SETTLE"
    )


def test_intent_owned_batch_is_never_recommitted_after_restart(tmp_path):
    """The end-to-end money assertion for the trap above: no second commitBatch."""
    store = _store(tmp_path)
    c1 = _client(store)
    _run(c1.accumulate(_make_batched(0)))
    ready = c1._accumulator.ready_batches(at_unix=2**62)[0]
    leaf_hashes, root = c1._build_leaves_root(ready)
    c1._committing[c1._intent_key(ready.key, root)] = c1._build_intent(ready, leaf_hashes, root)
    c1._persist()

    contract2 = _mock_contract()
    c2 = _client(store, contract=contract2)
    committed = _run(c2.commit_ready_batches(force=True))

    assert committed == []
    contract2.commit_batch.assert_not_called()


def test_committed_batch_is_not_restored_as_pending(tmp_path):
    """After a clean commit the batch is popped; a restart must not resurrect it."""
    store = _store(tmp_path)
    acc = ReceiptAccumulator(AccumulatorConfig(count_threshold=1))
    c1 = _client(store, acc=acc)
    _run(c1.accumulate(_make_batched(0)))
    assert len(_run(c1.commit_ready_batches())) == 1

    contract2 = _mock_contract()
    c2 = _client(store, contract=contract2)

    assert c2._accumulator.total_receipt_count() == 0
    assert _run(c2.commit_ready_batches(force=True)) == []
    contract2.commit_batch.assert_not_called()


# ── 3. Compatibility ─────────────────────────────────────────────────────


def test_state_file_without_pending_batches_key_loads(tmp_path):
    """An existing node's on-disk state predates this key. It must load, not hard-fail —
    _apply_state raises SettlementStateCorruptError on schema mismatch, which would turn
    settlement OFF for every upgrading operator."""
    store = _store(tmp_path)
    c1 = _client(store)
    _run(c1.accumulate(_make_batched(0)))
    state = store.load()
    del state["pending_batches"]          # simulate a pre-sp1409 file
    store.save(state)

    c2 = _client(store)                   # must not raise

    assert c2._accumulator.total_receipt_count() == 0


def test_no_store_keeps_in_memory_behavior(tmp_path):
    """store=None is the documented opt-out; it must write nothing and still accumulate."""
    c = BatchSettlementClient(ReceiptAccumulator(), _mock_contract(), PROVIDER_ADDR)
    _run(c.accumulate(_make_batched(0)))
    assert c._accumulator.total_receipt_count() == 1
    assert not list(tmp_path.iterdir())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
