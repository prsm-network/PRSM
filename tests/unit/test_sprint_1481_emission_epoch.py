"""Sprint 1481 — emission epoch builder: exactly-once attribution + exact allocation.

The middle link of the pool → earner rail. Two properties carry the money:

  1. **Exactly-once attribution** — a finalized batch may contribute to ONE epoch.
     Counting it twice pays a provider twice for the same work out of a shared pot,
     which necessarily underpays everyone else.
  2. **Exact allocation** — the entitlements must sum to the pot EXACTLY. Under-sum
     strands emission in the pool; over-sum publishes a root whose leaves exceed the
     reserved total, and the contract then reverts EpochOverdrawn part-way through —
     the last earners simply cannot claim.

Both are tested adversarially (re-scan, overlapping RPC ranges, boundary
timestamps, pathological pot sizes), not just on the happy path.
"""
from __future__ import annotations

import pytest
from eth_utils import to_checksum_address

from prsm.settlement.emission_epoch import (
    FinalizedBatch,
    allocate_pro_rata,
    build_emission_epoch,
    plan_to_reward_epoch,
)
from prsm.settlement.reward_epoch import verify_reward_proof

A = to_checksum_address("0x" + "a1" * 20)
B = to_checksum_address("0x" + "b2" * 20)
C = to_checksum_address("0x" + "c3" * 20)
POT = 1_000 * 10**18


def batch(bid, provider, value, ts=1000):
    return FinalizedBatch(
        batch_id=bid, provider=provider, final_value_wei=value, finalize_timestamp=ts
    )


# ───────────────────── exact allocation ─────────────────────

def test_allocation_sums_to_pot_exactly():
    alloc = allocate_pro_rata({A: 1, B: 1, C: 1}, 100)
    assert sum(alloc.values()) == 100          # 100/3 does not divide evenly
    assert sorted(alloc.values()) == [33, 33, 34]


@pytest.mark.parametrize("pot", [1, 2, 3, 7, 999, 10**18 + 1, 12345678901234567])
@pytest.mark.parametrize("weights", [
    {A: 1, B: 1},
    {A: 1, B: 2, C: 7},
    {A: 999999, B: 1},
    {A: 10**24, B: 3, C: 17},
])
def test_allocation_never_loses_or_creates_wei(pot, weights):
    """★ Across pathological pot/weight combinations the split is exact and every
    share is strictly positive (a zero leaf is unclaimable on chain)."""
    alloc = allocate_pro_rata(weights, pot)
    assert sum(alloc.values()) == pot
    assert all(v > 0 for v in alloc.values())


def test_tiny_pot_drops_providers_rather_than_emitting_zero_leaves():
    """A pot smaller than the provider count cannot pay everyone; it must pay a
    subset in full rather than mint unclaimable zero-amount leaves."""
    alloc = allocate_pro_rata({A: 1, B: 1, C: 1}, 2)
    assert sum(alloc.values()) == 2
    assert all(v > 0 for v in alloc.values())
    assert len(alloc) == 2


def test_allocation_is_proportional():
    alloc = allocate_pro_rata({A: 1, B: 3}, 100)
    assert alloc[A] == 25 and alloc[B] == 75


def test_allocation_is_deterministic_regardless_of_dict_order():
    """Two parties rebuilding the same epoch must get identical roots, so the
    tie-breaking must not depend on insertion order."""
    a1 = allocate_pro_rata({A: 1, B: 1, C: 1}, 100)
    a2 = allocate_pro_rata({C: 1, B: 1, A: 1}, 100)
    assert a1 == a2


def test_zero_pot_and_zero_weights_yield_nothing():
    assert allocate_pro_rata({A: 1}, 0) == {}
    assert allocate_pro_rata({A: 0, B: 0}, 100) == {}


def test_negative_pot_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        allocate_pro_rata({A: 1}, -1)


# ───────────────── exactly-once attribution ─────────────────

def test_already_consumed_batches_are_skipped():
    """★ The core double-pay guard: a batch paid by a previous epoch must not be
    counted again, even if the caller re-scans the same block range."""
    batches = [batch("0xaa", A, 100), batch("0xbb", B, 100)]
    plan = build_emission_epoch(
        epoch_id=2, batches=batches, pot_wei=POT,
        consumed_batch_ids={"0xaa"},
    )
    assert plan.skipped_already_consumed == ["0xaa"]
    assert set(plan.contribution_wei) == {B}
    assert [e.account for e in plan.entries] == [B]
    assert plan.entries[0].amount_wei == POT   # B alone takes the whole pot


def test_consumed_id_matching_is_case_insensitive():
    """Batch ids are hex; a 0xAA/0xaa mismatch must not defeat the guard."""
    plan_batches = [batch("0xAABB", A, 100), batch("0xcc", B, 100)]
    plan = build_emission_epoch(
        epoch_id=2, batches=plan_batches, pot_wei=POT,
        consumed_batch_ids={"0xaabb"},
    )
    assert plan.skipped_already_consumed == ["0xAABB"]
    assert set(plan.contribution_wei) == {B}


def test_duplicate_batch_within_one_scan_counted_once():
    """★ An RPC returning overlapping ranges must not inflate a provider's share."""
    batches = [batch("0xaa", A, 100), batch("0xaa", A, 100), batch("0xbb", B, 100)]
    plan = build_emission_epoch(epoch_id=1, batches=batches, pot_wei=POT)
    # A contributed 100 (not 200) -> equal split with B.
    assert plan.contribution_wei[A] == 100
    assert plan.entries[0].amount_wei == plan.entries[1].amount_wei


def test_consumed_ids_returned_for_watermarking():
    """The caller persists these; without them the next epoch cannot dedup."""
    batches = [batch("0xaa", A, 100), batch("0xbb", B, 50)]
    plan = build_emission_epoch(epoch_id=1, batches=batches, pot_wei=POT)
    assert sorted(plan.consumed_batch_ids) == ["0xaa", "0xbb"]


def test_two_sequential_epochs_never_double_pay():
    """★ End-to-end of the guard: feed the SAME batch list to epoch 2 while passing
    epoch 1's consumed ids. Epoch 2 must find nothing to pay."""
    batches = [batch("0xaa", A, 100), batch("0xbb", B, 100)]
    e1 = build_emission_epoch(epoch_id=1, batches=batches, pot_wei=POT)
    consumed = set(e1.consumed_batch_ids)
    with pytest.raises(ValueError, match="no eligible finalized batches"):
        build_emission_epoch(
            epoch_id=2, batches=batches, pot_wei=POT, consumed_batch_ids=consumed)


def test_fully_invalidated_batch_pays_nothing_but_is_consumed():
    """A batch whose value was fully challenged away earned nothing — but it must
    still be marked consumed so it is not re-examined forever."""
    batches = [batch("0xaa", A, 0), batch("0xbb", B, 100)]
    plan = build_emission_epoch(epoch_id=1, batches=batches, pot_wei=POT)
    assert A not in plan.contribution_wei
    assert "0xaa" in plan.consumed_batch_ids
    assert plan.entries[0].account == B


# ───────────────────── windowing ─────────────────────

def test_window_bounds_are_half_open():
    """[start, end) — a batch exactly on `end` belongs to the NEXT epoch, so a
    boundary batch is never counted in both."""
    batches = [
        batch("0xaa", A, 100, ts=999),    # before window
        batch("0xbb", B, 100, ts=1000),   # start, inclusive
        batch("0xcc", C, 100, ts=2000),   # end, exclusive
    ]
    plan = build_emission_epoch(
        epoch_id=1, batches=batches, pot_wei=POT,
        window_start=1000, window_end=2000,
    )
    assert set(plan.contribution_wei) == {B}
    # The excluded ones are NOT consumed — a later epoch must still be able to pay them.
    assert plan.consumed_batch_ids == ["0xbb"]


def test_adjacent_windows_partition_batches_exactly_once():
    """★ Two adjacent half-open windows must together count each batch exactly once."""
    batches = [batch(f"0x{i:02x}", A, 100, ts=1000 + i) for i in range(10)]
    e1 = build_emission_epoch(
        epoch_id=1, batches=batches, pot_wei=POT, window_start=1000, window_end=1005)
    e2 = build_emission_epoch(
        epoch_id=2, batches=batches, pot_wei=POT, window_start=1005, window_end=1010,
        consumed_batch_ids=set(e1.consumed_batch_ids))
    assert len(e1.consumed_batch_ids) == 5
    assert len(e2.consumed_batch_ids) == 5
    assert set(e1.consumed_batch_ids).isdisjoint(set(e2.consumed_batch_ids))
    assert set(e1.consumed_batch_ids) | set(e2.consumed_batch_ids) == {
        b.batch_id for b in batches}


# ───────────────── integration with the tree ─────────────────

def test_plan_becomes_a_publishable_epoch_whose_leaves_sum_to_the_pot():
    """★ The join point: the tree's total must equal the pot, or the contract's
    reserve and the leaves disagree and late claimers revert EpochOverdrawn."""
    batches = [batch("0xaa", A, 7), batch("0xbb", B, 11), batch("0xcc", C, 13)]
    plan = build_emission_epoch(epoch_id=5, batches=batches, pot_wei=POT)
    assert plan.total_allocated_wei == POT

    epoch = plan_to_reward_epoch(plan)
    assert epoch.total_amount_wei == POT
    for e in epoch.entries:
        assert verify_reward_proof(
            5, e.account, e.amount_wei, epoch.proofs[e.account], epoch.merkle_root)


def test_empty_batch_set_refuses_to_plan():
    with pytest.raises(ValueError, match="no eligible finalized batches"):
        build_emission_epoch(epoch_id=1, batches=[], pot_wei=POT)


def test_pot_too_small_for_anyone_is_refused():
    """Rather than publish an epoch nobody can claim from."""
    with pytest.raises(ValueError, match="too small"):
        build_emission_epoch(epoch_id=1, batches=[batch("0xaa", A, 100)], pot_wei=0)


def test_provider_with_many_batches_gets_the_sum():
    batches = [batch("0xa1", A, 30), batch("0xa2", A, 70), batch("0xb1", B, 100)]
    plan = build_emission_epoch(epoch_id=1, batches=batches, pot_wei=POT)
    assert plan.contribution_wei[A] == 100
    assert plan.entries[0].amount_wei == plan.entries[1].amount_wei  # 50/50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
