"""Phase 3.1 Task 4 — ReceiptAccumulator unit tests.

Verifies:
  - Basic add + pop contract.
  - Per-(requester, provider) key isolation.
  - Three-trigger logic: count, time, value.
  - First-trigger-wins semantics.
  - started_at_unix is set on first append only.
  - total_value_ftns accumulates correctly.
  - Empty batches don't appear in ready_batches.
  - pop_batch removes; peek_batch doesn't.
  - Config validation rejects non-positive thresholds.
  - Cross-key aggregation methods (total_receipt_count, etc.).
"""
from __future__ import annotations

import hashlib
from typing import Tuple

import pytest

from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.settlement.accumulator import (
    AccumulatorConfig,
    BatchedReceipt,
    PendingBatch,
    ReadyBatch,
    ReceiptAccumulator,
    TriggerReason,
)


# Constants shared across tests.
ONE_FTNS = 10**18
DEFAULT_CONFIG = AccumulatorConfig(
    count_threshold=1000,
    time_threshold_seconds=3600,
    value_threshold_ftns=100 * 10**18,
)

# Sample Ethereum-like addresses.
ADDR_REQUESTER_A = "0x" + "a" * 40
ADDR_REQUESTER_B = "0x" + "b" * 40
ADDR_PROVIDER_X = "0x" + "1" * 40
ADDR_PROVIDER_Y = "0x" + "2" * 40


def _make_receipt(
    job_id: str = "job-1",
    shard_index: int = 0,
    output_hash: str = None,
    executed_at_unix: int = 1700000000,
) -> ShardExecutionReceipt:
    """Construct a valid-shape ShardExecutionReceipt. Signature is
    synthetic; we don't verify it in accumulator tests."""
    if output_hash is None:
        output_hash = hashlib.sha256(f"{job_id}:{shard_index}".encode()).hexdigest()
    return ShardExecutionReceipt(
        job_id=job_id,
        shard_index=shard_index,
        provider_id="provider-id-hex",
        provider_pubkey_b64="pubkey_b64",
        output_hash=output_hash,
        executed_at_unix=executed_at_unix,
        signature="sig_b64",
    )


_escrow_counter = [0]


def _make_batched(
    job_id: str = "job-1",
    shard_index: int = 0,
    requester: str = ADDR_REQUESTER_A,
    provider: str = ADDR_PROVIDER_X,
    value_ftns: int = ONE_FTNS,
    escrow_id: str = None,
) -> BatchedReceipt:
    # sp973 — each real dispatch has a UNIQUE local_escrow_id; default to a fresh
    # one per call so a batch of distinct receipts isn't collapsed by the new
    # per-escrow-id dedup. Tests exercising the dedup pass an explicit id.
    if escrow_id is None:
        _escrow_counter[0] += 1
        escrow_id = f"escrow-{_escrow_counter[0]}"
    return BatchedReceipt(
        receipt=_make_receipt(job_id=job_id, shard_index=shard_index),
        requester_address=requester,
        provider_address=provider,
        value_ftns=value_ftns,
        local_escrow_id=escrow_id,
    )


# ── Basic contract ────────────────────────────────────────────────


def test_empty_accumulator_has_no_ready_batches():
    acc = ReceiptAccumulator()
    assert acc.ready_batches() == []
    assert acc.pending_keys() == []
    assert acc.total_receipt_count() == 0
    assert acc.total_pending_value_ftns() == 0


def test_add_creates_batch_for_new_pair():
    acc = ReceiptAccumulator()
    br = _make_batched()
    acc.add(br, at_unix=1000)

    key = (ADDR_REQUESTER_A, ADDR_PROVIDER_X, b"\x00" * 32, 0)
    assert key in acc.pending_keys()
    batch = acc.peek_batch(key)
    assert batch is not None
    assert batch.count == 1
    assert batch.total_value_ftns == ONE_FTNS
    assert batch.started_at_unix == 1000


def test_add_appends_to_existing_batch():
    acc = ReceiptAccumulator()
    for i in range(3):
        acc.add(_make_batched(shard_index=i), at_unix=1000 + i)

    key = (ADDR_REQUESTER_A, ADDR_PROVIDER_X, b"\x00" * 32, 0)
    batch = acc.peek_batch(key)
    assert batch.count == 3
    assert batch.total_value_ftns == 3 * ONE_FTNS
    # started_at_unix is pinned to the first append's timestamp.
    assert batch.started_at_unix == 1000


def test_add_does_not_move_started_at_on_subsequent_appends():
    acc = ReceiptAccumulator()
    acc.add(_make_batched(shard_index=0), at_unix=500)
    acc.add(_make_batched(shard_index=1), at_unix=2000)  # much later
    key = (ADDR_REQUESTER_A, ADDR_PROVIDER_X, b"\x00" * 32, 0)
    assert acc.peek_batch(key).started_at_unix == 500


# ── Per-pair key isolation ────────────────────────────────────────


def test_different_providers_tracked_in_separate_batches():
    acc = ReceiptAccumulator()
    acc.add(_make_batched(provider=ADDR_PROVIDER_X), at_unix=1000)
    acc.add(_make_batched(provider=ADDR_PROVIDER_Y), at_unix=1000)

    keys = set(acc.pending_keys())
    assert (ADDR_REQUESTER_A, ADDR_PROVIDER_X, b"\x00" * 32, 0) in keys
    assert (ADDR_REQUESTER_A, ADDR_PROVIDER_Y, b"\x00" * 32, 0) in keys
    assert len(keys) == 2


def test_different_requesters_tracked_in_separate_batches():
    acc = ReceiptAccumulator()
    acc.add(_make_batched(requester=ADDR_REQUESTER_A), at_unix=1000)
    acc.add(_make_batched(requester=ADDR_REQUESTER_B), at_unix=1000)
    assert len(acc.pending_keys()) == 2


def test_cross_pair_totals_aggregate_correctly():
    acc = ReceiptAccumulator()
    acc.add(_make_batched(provider=ADDR_PROVIDER_X, value_ftns=2 * ONE_FTNS), at_unix=1000)
    acc.add(_make_batched(provider=ADDR_PROVIDER_X, value_ftns=3 * ONE_FTNS), at_unix=1000)
    acc.add(_make_batched(provider=ADDR_PROVIDER_Y, value_ftns=5 * ONE_FTNS), at_unix=1000)
    assert acc.total_receipt_count() == 3
    assert acc.total_pending_value_ftns() == 10 * ONE_FTNS


# ── Trigger logic: count ──────────────────────────────────────────


def test_count_trigger_fires_at_threshold():
    cfg = AccumulatorConfig(count_threshold=3, time_threshold_seconds=3600, value_threshold_ftns=10**30)
    acc = ReceiptAccumulator(cfg)
    for i in range(3):
        acc.add(_make_batched(shard_index=i), at_unix=1000)

    ready = acc.ready_batches(at_unix=1100)
    assert len(ready) == 1
    assert ready[0].trigger == TriggerReason.COUNT


def test_count_trigger_does_not_fire_below_threshold():
    cfg = AccumulatorConfig(count_threshold=3, time_threshold_seconds=3600, value_threshold_ftns=10**30)
    acc = ReceiptAccumulator(cfg)
    for i in range(2):
        acc.add(_make_batched(shard_index=i), at_unix=1000)
    assert acc.ready_batches(at_unix=1100) == []


def test_count_trigger_fires_past_threshold():
    cfg = AccumulatorConfig(count_threshold=2, time_threshold_seconds=3600, value_threshold_ftns=10**30)
    acc = ReceiptAccumulator(cfg)
    for i in range(5):  # well above threshold
        acc.add(_make_batched(shard_index=i), at_unix=1000)
    ready = acc.ready_batches(at_unix=1100)
    assert ready[0].trigger == TriggerReason.COUNT


# ── Trigger logic: time ───────────────────────────────────────────


def test_time_trigger_fires_after_elapsed_seconds():
    cfg = AccumulatorConfig(count_threshold=10000, time_threshold_seconds=60, value_threshold_ftns=10**30)
    acc = ReceiptAccumulator(cfg)
    acc.add(_make_batched(), at_unix=1000)

    # 30 seconds: not yet.
    assert acc.ready_batches(at_unix=1030) == []

    # 60 seconds: ready.
    ready = acc.ready_batches(at_unix=1060)
    assert len(ready) == 1
    assert ready[0].trigger == TriggerReason.TIME


def test_time_trigger_uses_first_receipt_timestamp():
    cfg = AccumulatorConfig(count_threshold=10000, time_threshold_seconds=30, value_threshold_ftns=10**30)
    acc = ReceiptAccumulator(cfg)
    acc.add(_make_batched(shard_index=0), at_unix=1000)  # started_at = 1000
    acc.add(_make_batched(shard_index=1), at_unix=1020)  # doesn't reset clock

    # At t=1030, 30 seconds since first receipt.
    ready = acc.ready_batches(at_unix=1030)
    assert len(ready) == 1
    assert ready[0].trigger == TriggerReason.TIME


# ── Trigger logic: value ──────────────────────────────────────────


def test_value_trigger_fires_at_threshold():
    cfg = AccumulatorConfig(count_threshold=10000, time_threshold_seconds=3600, value_threshold_ftns=10 * ONE_FTNS)
    acc = ReceiptAccumulator(cfg)
    acc.add(_make_batched(value_ftns=5 * ONE_FTNS), at_unix=1000)
    acc.add(_make_batched(value_ftns=5 * ONE_FTNS), at_unix=1000)  # cumulative = 10 FTNS

    ready = acc.ready_batches(at_unix=1100)
    assert len(ready) == 1
    assert ready[0].trigger == TriggerReason.VALUE


def test_value_trigger_does_not_fire_below_threshold():
    cfg = AccumulatorConfig(count_threshold=10000, time_threshold_seconds=3600, value_threshold_ftns=10 * ONE_FTNS)
    acc = ReceiptAccumulator(cfg)
    acc.add(_make_batched(value_ftns=5 * ONE_FTNS), at_unix=1000)
    acc.add(_make_batched(value_ftns=3 * ONE_FTNS), at_unix=1000)  # cumulative = 8 FTNS
    assert acc.ready_batches(at_unix=1100) == []


# ── First-trigger-wins semantics ──────────────────────────────────


def test_count_trigger_wins_when_count_hits_first():
    cfg = AccumulatorConfig(count_threshold=2, time_threshold_seconds=3600, value_threshold_ftns=10 * ONE_FTNS)
    acc = ReceiptAccumulator(cfg)
    acc.add(_make_batched(value_ftns=ONE_FTNS), at_unix=1000)
    acc.add(_make_batched(value_ftns=ONE_FTNS), at_unix=1000)
    # Count = 2 (hit threshold), value = 2 FTNS (below 10 threshold)
    ready = acc.ready_batches(at_unix=1100)
    assert ready[0].trigger == TriggerReason.COUNT


def test_time_trigger_reported_when_time_is_only_fired():
    cfg = AccumulatorConfig(count_threshold=10000, time_threshold_seconds=60, value_threshold_ftns=10 * ONE_FTNS)
    acc = ReceiptAccumulator(cfg)
    acc.add(_make_batched(value_ftns=ONE_FTNS), at_unix=1000)
    # Count = 1 (below 10000), value = 1 FTNS (below 10), time elapsed = 60.
    ready = acc.ready_batches(at_unix=1060)
    assert len(ready) == 1
    assert ready[0].trigger == TriggerReason.TIME


# ── pop_batch + readiness ─────────────────────────────────────────


def test_pop_batch_removes_batch():
    acc = ReceiptAccumulator()
    acc.add(_make_batched(), at_unix=1000)
    key = (ADDR_REQUESTER_A, ADDR_PROVIDER_X, b"\x00" * 32, 0)

    popped = acc.pop_batch(key)
    assert popped is not None
    assert popped.count == 1
    assert acc.peek_batch(key) is None
    assert acc.pending_keys() == []


def test_pop_batch_returns_none_for_unknown_key():
    acc = ReceiptAccumulator()
    assert acc.pop_batch((ADDR_REQUESTER_A, ADDR_PROVIDER_X, b"\x00" * 32, 0)) is None


def test_ready_batches_after_pop_are_empty():
    cfg = AccumulatorConfig(count_threshold=1, time_threshold_seconds=3600, value_threshold_ftns=10**30)
    acc = ReceiptAccumulator(cfg)
    acc.add(_make_batched(), at_unix=1000)
    assert len(acc.ready_batches(at_unix=1100)) == 1
    acc.pop_batch((ADDR_REQUESTER_A, ADDR_PROVIDER_X, b"\x00" * 32, 0))
    assert acc.ready_batches(at_unix=1100) == []


def test_multiple_ready_batches_all_returned():
    cfg = AccumulatorConfig(count_threshold=1, time_threshold_seconds=3600, value_threshold_ftns=10**30)
    acc = ReceiptAccumulator(cfg)
    acc.add(_make_batched(provider=ADDR_PROVIDER_X), at_unix=1000)
    acc.add(_make_batched(provider=ADDR_PROVIDER_Y), at_unix=1000)
    ready = acc.ready_batches(at_unix=1100)
    assert len(ready) == 2


def test_only_triggered_batches_in_ready_list():
    cfg = AccumulatorConfig(count_threshold=2, time_threshold_seconds=3600, value_threshold_ftns=10**30)
    acc = ReceiptAccumulator(cfg)
    # Pair X: count=1 (below threshold)
    acc.add(_make_batched(provider=ADDR_PROVIDER_X), at_unix=1000)
    # Pair Y: count=2 (at threshold)
    acc.add(_make_batched(provider=ADDR_PROVIDER_Y, shard_index=0), at_unix=1000)
    acc.add(_make_batched(provider=ADDR_PROVIDER_Y, shard_index=1), at_unix=1000)

    ready = acc.ready_batches(at_unix=1100)
    assert len(ready) == 1
    assert ready[0].key == (ADDR_REQUESTER_A, ADDR_PROVIDER_Y, b"\x00" * 32, 0)


# ── Config validation ─────────────────────────────────────────────


def test_config_rejects_zero_count_threshold():
    with pytest.raises(ValueError, match="count_threshold"):
        AccumulatorConfig(count_threshold=0, time_threshold_seconds=60, value_threshold_ftns=1)


def test_config_rejects_negative_time_threshold():
    with pytest.raises(ValueError, match="time_threshold_seconds"):
        AccumulatorConfig(count_threshold=1, time_threshold_seconds=-1, value_threshold_ftns=1)


def test_config_rejects_zero_value_threshold():
    with pytest.raises(ValueError, match="value_threshold_ftns"):
        AccumulatorConfig(count_threshold=1, time_threshold_seconds=60, value_threshold_ftns=0)


def test_default_config_values_match_design_doc():
    cfg = AccumulatorConfig()
    assert cfg.count_threshold == 1000
    assert cfg.time_threshold_seconds == 3600
    assert cfg.value_threshold_ftns == 100 * 10**18


# ── Aggregate helpers ─────────────────────────────────────────────


def test_pending_keys_excludes_empty_batches():
    """After all receipts for a pair have been popped, the pair is gone
    from pending_keys."""
    acc = ReceiptAccumulator()
    acc.add(_make_batched(provider=ADDR_PROVIDER_X), at_unix=1000)
    acc.add(_make_batched(provider=ADDR_PROVIDER_Y), at_unix=1000)
    acc.pop_batch((ADDR_REQUESTER_A, ADDR_PROVIDER_X, b"\x00" * 32, 0))
    keys = acc.pending_keys()
    assert len(keys) == 1
    assert keys[0] == (ADDR_REQUESTER_A, ADDR_PROVIDER_Y, b"\x00" * 32, 0)


def test_readd_to_previously_popped_key_starts_fresh():
    """If a batch is popped and new receipts come in for that pair,
    started_at_unix is set to the new first append's timestamp."""
    acc = ReceiptAccumulator()
    acc.add(_make_batched(), at_unix=1000)
    key = (ADDR_REQUESTER_A, ADDR_PROVIDER_X, b"\x00" * 32, 0)
    acc.pop_batch(key)
    acc.add(_make_batched(shard_index=1), at_unix=5000)
    assert acc.peek_batch(key).started_at_unix == 5000


# ── Phase 7 + 7.1x key extensions ──────────────────────────────────


def _batched_with_extras(
    tier_slash_rate_bps: int = 0,
    consensus_group_id: bytes = b"\x00" * 32,
    shard_index: int = 0,
):
    """Helper: _make_batched + override the Phase 7 / Phase 7.1x fields."""
    base = _make_batched(shard_index=shard_index)
    return BatchedReceipt(
        receipt=base.receipt,
        requester_address=base.requester_address,
        provider_address=base.provider_address,
        value_ftns=base.value_ftns,
        local_escrow_id=base.local_escrow_id,
        tier_slash_rate_bps=tier_slash_rate_bps,
        consensus_group_id=consensus_group_id,
    )


def test_different_consensus_group_ids_batch_separately():
    """Two receipts from the same provider for the same requester but
    with DIFFERENT consensus_group_ids MUST land in different batches.
    A single on-chain Batch struct carries one group_id; mixing would
    break the CONSENSUS_MISMATCH binding invariant."""
    acc = ReceiptAccumulator()
    group_a = b"\x01" * 32
    group_b = b"\x02" * 32
    acc.add(
        _batched_with_extras(consensus_group_id=group_a, shard_index=0),
        at_unix=1000,
    )
    acc.add(
        _batched_with_extras(consensus_group_id=group_b, shard_index=1),
        at_unix=1000,
    )
    keys = acc.pending_keys()
    assert len(keys) == 2
    assert (ADDR_REQUESTER_A, ADDR_PROVIDER_X, group_a, 0) in keys
    assert (ADDR_REQUESTER_A, ADDR_PROVIDER_X, group_b, 0) in keys


def test_different_tier_slash_rates_batch_separately():
    """Similarly, a provider who re-bonded at a different tier between
    receipts produces receipts with different tier_slash_rate_bps.
    Those can't share a batch either (Batch struct carries one rate)."""
    acc = ReceiptAccumulator()
    acc.add(_batched_with_extras(tier_slash_rate_bps=5000, shard_index=0),
            at_unix=1000)
    acc.add(_batched_with_extras(tier_slash_rate_bps=10000, shard_index=1),
            at_unix=1000)
    keys = acc.pending_keys()
    assert len(keys) == 2
    slashes = {k[3] for k in keys}
    assert slashes == {5000, 10000}


def test_zero_group_id_receipts_batch_together_legacy():
    """Pre-Phase-7.1x receipts (no consensus dispatch) all carry
    group_id = zero bytes. They should continue to batch together by
    (requester, provider) as in Phase 3.1 — the new key just degenerates."""
    acc = ReceiptAccumulator()
    for i in range(3):
        acc.add(_batched_with_extras(shard_index=i), at_unix=1000 + i)
    keys = acc.pending_keys()
    assert len(keys) == 1
    batch = acc.peek_batch(keys[0])
    assert batch.count == 3


def test_batch_level_slash_rate_captured_on_first_append():
    """PendingBatch.tier_slash_rate_bps + consensus_group_id are pinned
    on the first receipt and surface via the batch for audit/commit
    payload construction."""
    acc = ReceiptAccumulator()
    group = b"\xab" * 32
    acc.add(
        _batched_with_extras(
            tier_slash_rate_bps=10000,
            consensus_group_id=group,
        ),
        at_unix=1000,
    )
    key = (ADDR_REQUESTER_A, ADDR_PROVIDER_X, group, 10000)
    batch = acc.peek_batch(key)
    assert batch.tier_slash_rate_bps == 10000
    assert batch.consensus_group_id == group


def test_batched_receipt_rejects_bad_slash_rate():
    from prsm.settlement.accumulator import BatchedReceipt
    from prsm.compute.shard_receipt import ShardExecutionReceipt
    base_receipt = ShardExecutionReceipt(
        job_id="j", shard_index=0, provider_id="p",
        provider_pubkey_b64="k", output_hash="o",
        executed_at_unix=1, signature="s",
    )
    with pytest.raises(ValueError, match="tier_slash_rate_bps"):
        BatchedReceipt(
            receipt=base_receipt,
            requester_address="0x00", provider_address="0x01",
            value_ftns=1, local_escrow_id="e",
            tier_slash_rate_bps=10001,
        )


def test_batched_receipt_rejects_bad_group_id_length():
    from prsm.settlement.accumulator import BatchedReceipt
    from prsm.compute.shard_receipt import ShardExecutionReceipt
    base_receipt = ShardExecutionReceipt(
        job_id="j", shard_index=0, provider_id="p",
        provider_pubkey_b64="k", output_hash="o",
        executed_at_unix=1, signature="s",
    )
    with pytest.raises(ValueError, match="consensus_group_id"):
        BatchedReceipt(
            receipt=base_receipt,
            requester_address="0x00", provider_address="0x01",
            value_ftns=1, local_escrow_id="e",
            consensus_group_id=b"\x00" * 31,  # 31 bytes, not 32
        )


# ── sp973: per-batch local_escrow_id dedup (defense-in-depth) ──────────────


def test_duplicate_escrow_id_counted_once_in_batch():
    """A receipt with a local_escrow_id already in the batch is a no-op — it must
    not inflate count or total_value_ftns (guards a future replay into add())."""
    b = PendingBatch()
    b.append(_make_batched(escrow_id="e1", value_ftns=ONE_FTNS), 100)
    b.append(_make_batched(escrow_id="e1", value_ftns=ONE_FTNS), 100)  # duplicate
    assert b.count == 1
    assert b.total_value_ftns == ONE_FTNS


def test_distinct_escrow_ids_both_counted():
    b = PendingBatch()
    b.append(_make_batched(escrow_id="e1", value_ftns=ONE_FTNS), 100)
    b.append(_make_batched(escrow_id="e2", value_ftns=ONE_FTNS), 100)
    assert b.count == 2
    assert b.total_value_ftns == 2 * ONE_FTNS


def test_accumulator_add_dedups_duplicate_escrow_id():
    """End-to-end through the accumulator: a duplicate escrow id does not advance
    the count threshold."""
    from prsm.settlement.accumulator import AccumulatorConfig
    acc = ReceiptAccumulator(AccumulatorConfig(count_threshold=2))
    t = 1700000000  # fixed clock; evaluate readiness at the same instant so the
    acc.add(_make_batched(escrow_id="e1"), at_unix=t)  # TIME threshold never fires
    acc.add(_make_batched(escrow_id="e1"), at_unix=t)  # dup → still 1 → not ready
    assert acc.ready_batches(at_unix=t) == []
    acc.add(_make_batched(escrow_id="e2"), at_unix=t)  # distinct → 2 → ready
    assert len(acc.ready_batches(at_unix=t)) == 1
