"""Phase 3.1 Task 6 — BatchSettlementClient unit tests.

Uses an AsyncMock implementing SettlementContractClient so the tests
don't depend on web3/RPC. Sepolia integration testing is gated behind
a separate test file + env var (Task 6 plan), out of scope here.
"""
from __future__ import annotations

import hashlib
from base64 import b64encode
from unittest.mock import AsyncMock
from typing import Tuple

import pytest

from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.settlement.accumulator import (
    AccumulatorConfig,
    BatchedReceipt,
    ReceiptAccumulator,
    TriggerReason,
)
from prsm.settlement.client import (
    BatchSettlementClient,
    CommittedBatch,
    FinalizedBatch,
)


ONE_FTNS = 10**18
PROVIDER_ADDR = "0x" + "b" * 40
OTHER_PROVIDER_ADDR = "0x" + "c" * 40
REQUESTER_A = "0x" + "a" * 40
REQUESTER_B = "0x" + "d" * 40


def _make_batched(
    shard_index: int = 0,
    requester: str = REQUESTER_A,
    provider: str = PROVIDER_ADDR,
    value_ftns: int = ONE_FTNS,
    escrow_id: str = None,
) -> BatchedReceipt:
    receipt = ShardExecutionReceipt(
        job_id=f"job-{shard_index}",
        shard_index=shard_index,
        provider_id="provider-id-string",
        provider_pubkey_b64=b64encode(b"pubkey-raw").decode(),
        output_hash=hashlib.sha256(
            f"out-{shard_index}".encode()
        ).hexdigest(),
        executed_at_unix=1700000000 + shard_index,
        signature=b64encode(b"sig-raw").decode(),
    )
    return BatchedReceipt(
        receipt=receipt,
        requester_address=requester,
        provider_address=provider,
        value_ftns=value_ftns,
        local_escrow_id=escrow_id if escrow_id is not None else f"escrow-{shard_index}",
    )


def _make_mock_contract(commit_return=None, default_status: int = 1):
    """Mock SettlementContractClient with AsyncMock methods.

    commit_return: tuple (batch_id_bytes, commit_ts) or callable producing
                   one per invocation. Defaults to a deterministic per-
                   call batchId so multiple commits produce distinct ids.
    """
    mock = AsyncMock()

    if commit_return is None:
        # Produce distinct batch_ids across calls.
        _call_counter = {"n": 0}

        async def commit_default(**kwargs):
            # kwargs includes provider_address, requester_address,
            # merkle_root, receipt_count, total_value_ftns, metadata_uri.
            _call_counter["n"] += 1
            bid = hashlib.sha256(
                f"mock-batch-{_call_counter['n']}".encode()
            ).digest()
            return (bid, 1700000000 + _call_counter["n"])

        mock.commit_batch.side_effect = commit_default
    else:
        mock.commit_batch.return_value = commit_return

    mock.is_finalizable.return_value = False
    mock.finalize_batch.return_value = None
    mock.get_batch_status.return_value = default_status
    return mock


def _run(coro):
    """Helper to run a coroutine synchronously (tests not async-marked)."""
    import asyncio
    return asyncio.run(coro)


# ── Accumulation ──────────────────────────────────────────────────


def test_client_forwards_accumulate_to_internal_accumulator():
    acc = ReceiptAccumulator()
    contract = _make_mock_contract()
    client = BatchSettlementClient(acc, contract, PROVIDER_ADDR)

    br = _make_batched()
    _run(client.accumulate(br))

    assert acc.total_receipt_count() == 1


def test_client_rejects_receipt_for_wrong_provider():
    """Safety check: the client's bound address must match EITHER
    the receipt's provider_address OR requester_address. A receipt
    for a provider different from the client AND a requester the
    client doesn't match raises rather than silently accumulating."""
    acc = ReceiptAccumulator()
    contract = _make_mock_contract()
    client = BatchSettlementClient(acc, contract, PROVIDER_ADDR)

    # Receipt's provider is not our bound addr, and requester is also
    # not our bound addr — must reject.
    br = _make_batched(
        provider=OTHER_PROVIDER_ADDR,
        requester=REQUESTER_A,  # also != PROVIDER_ADDR
    )
    with pytest.raises(ValueError, match="refusing receipt"):
        _run(client.accumulate(br))
    assert acc.total_receipt_count() == 0


def test_client_accepts_receipt_when_bound_as_requester():
    """Requester-side binding: client's bound address is the requester's
    Ethereum address; receipts where requester_address matches accumulate
    cleanly. This is the Task 7 orchestrator hook's pattern."""
    acc = ReceiptAccumulator()
    contract = _make_mock_contract()
    # Bind the client to REQUESTER_A instead of a provider.
    client = BatchSettlementClient(acc, contract, REQUESTER_A)

    # Receipt from various providers, all paid by REQUESTER_A.
    br = _make_batched(
        requester=REQUESTER_A,
        provider=OTHER_PROVIDER_ADDR,
    )
    _run(client.accumulate(br))
    assert acc.total_receipt_count() == 1


# ── Commit path ───────────────────────────────────────────────────


def test_commit_ready_batches_returns_empty_when_nothing_ready():
    acc = ReceiptAccumulator()
    contract = _make_mock_contract()
    client = BatchSettlementClient(acc, contract, PROVIDER_ADDR)

    committed = _run(client.commit_ready_batches())
    assert committed == []
    contract.commit_batch.assert_not_awaited()


def test_commit_ready_batches_commits_one_batch_at_count_threshold():
    cfg = AccumulatorConfig(
        count_threshold=3,
        time_threshold_seconds=3600,
        value_threshold_ftns=10**30,  # never triggers in test
    )
    acc = ReceiptAccumulator(cfg)
    contract = _make_mock_contract()
    client = BatchSettlementClient(acc, contract, PROVIDER_ADDR)

    for i in range(3):
        _run(client.accumulate(_make_batched(shard_index=i)))

    committed = _run(client.commit_ready_batches())
    assert len(committed) == 1

    rec = committed[0]
    assert rec.receipt_count == 3
    assert rec.total_value_ftns == 3 * ONE_FTNS
    assert rec.provider_address == PROVIDER_ADDR
    assert rec.requester_address == REQUESTER_A
    assert rec.trigger_reason == TriggerReason.COUNT
    assert len(rec.leaf_hashes) == 3

    # Mock was called with the right arguments.
    contract.commit_batch.assert_awaited_once()
    kwargs = contract.commit_batch.await_args.kwargs
    assert kwargs["requester_address"] == REQUESTER_A
    assert kwargs["receipt_count"] == 3
    assert kwargs["total_value_ftns"] == 3 * ONE_FTNS
    assert len(kwargs["merkle_root"]) == 32

    # Accumulator drained.
    assert acc.total_receipt_count() == 0


def test_commit_pops_accumulator_only_on_success():
    """If the on-chain commit raises, receipts stay in the accumulator
    so they retry on the next poll — no data loss."""
    cfg = AccumulatorConfig(count_threshold=1, time_threshold_seconds=3600, value_threshold_ftns=10**30)
    acc = ReceiptAccumulator(cfg)
    contract = _make_mock_contract()

    # Make commit fail the first time, succeed the second.
    call_log = {"n": 0}

    async def flaky_commit(**kwargs):
        call_log["n"] += 1
        if call_log["n"] == 1:
            raise RuntimeError("RPC temporary glitch")
        return (hashlib.sha256(b"retry-success").digest(), 1700000000)

    contract.commit_batch.side_effect = flaky_commit

    client = BatchSettlementClient(acc, contract, PROVIDER_ADDR)
    _run(client.accumulate(_make_batched()))

    # First poll: commit fails.
    committed = _run(client.commit_ready_batches())
    assert committed == []
    # Accumulator still has the receipt.
    assert acc.total_receipt_count() == 1

    # Second poll: commit succeeds.
    committed = _run(client.commit_ready_batches())
    assert len(committed) == 1
    assert acc.total_receipt_count() == 0


def test_commit_multiple_pair_batches_handled_independently():
    """Two ready batches from distinct (requester, provider) pairs get
    committed in one poll cycle."""
    cfg = AccumulatorConfig(count_threshold=1, time_threshold_seconds=3600, value_threshold_ftns=10**30)
    acc = ReceiptAccumulator(cfg)
    contract = _make_mock_contract()
    client = BatchSettlementClient(acc, contract, PROVIDER_ADDR)

    _run(client.accumulate(_make_batched(requester=REQUESTER_A)))
    _run(client.accumulate(_make_batched(requester=REQUESTER_B)))

    committed = _run(client.commit_ready_batches())
    assert len(committed) == 2
    addrs = {c.requester_address for c in committed}
    assert addrs == {REQUESTER_A, REQUESTER_B}


def test_commit_distinct_batch_ids_across_commits():
    """Two batches with identical RECEIPT content (same shard_index → same
    job/output hashes) still get distinct batchIds from the contract
    (per-provider sequence counter). Verifies the mock's default side_effect
    produces distinct ids.

    The two receipts carry DISTINCT escrow ids — a genuinely separate escrow
    allocation for identical logical work. Reusing the SAME escrow id here
    would (correctly, since sp1436) be dropped as a re-delivery/double-settle,
    which is a different concern from batchId uniqueness. The escrow id, not
    the receipt content, is the settlement dedup key."""
    cfg = AccumulatorConfig(count_threshold=1, time_threshold_seconds=3600, value_threshold_ftns=10**30)
    acc = ReceiptAccumulator(cfg)
    contract = _make_mock_contract()
    client = BatchSettlementClient(acc, contract, PROVIDER_ADDR)

    _run(client.accumulate(_make_batched(shard_index=0, requester=REQUESTER_A, escrow_id="escrow-a")))
    c1 = _run(client.commit_ready_batches())

    _run(client.accumulate(_make_batched(shard_index=0, requester=REQUESTER_A, escrow_id="escrow-b")))
    c2 = _run(client.commit_ready_batches())

    assert c1[0].batch_id != c2[0].batch_id


# ── Finalize path ─────────────────────────────────────────────────


def test_finalize_ready_batches_submits_when_on_chain_eligible():
    cfg = AccumulatorConfig(count_threshold=1, time_threshold_seconds=3600, value_threshold_ftns=10**30)
    acc = ReceiptAccumulator(cfg)
    contract = _make_mock_contract()
    client = BatchSettlementClient(acc, contract, PROVIDER_ADDR)

    _run(client.accumulate(_make_batched()))
    _run(client.commit_ready_batches())  # commits 1 batch

    # Before window elapses, contract says not finalizable.
    contract.is_finalizable.return_value = False
    assert _run(client.finalize_ready_batches()) == []
    contract.finalize_batch.assert_not_awaited()

    # After window, contract says finalizable.
    contract.is_finalizable.return_value = True
    result = _run(client.finalize_ready_batches())
    assert len(result) == 1
    assert result[0].tx_submitted is True
    contract.finalize_batch.assert_awaited_once()

    # Tracking records the finalization locally.
    assert client.is_finalized_locally(client.tracked_batches()[0].batch_id)


def test_finalize_ready_batches_idempotent_after_success():
    """After a successful finalize, subsequent polls skip this batch."""
    cfg = AccumulatorConfig(count_threshold=1, time_threshold_seconds=3600, value_threshold_ftns=10**30)
    acc = ReceiptAccumulator(cfg)
    contract = _make_mock_contract()
    client = BatchSettlementClient(acc, contract, PROVIDER_ADDR)

    _run(client.accumulate(_make_batched()))
    _run(client.commit_ready_batches())

    contract.is_finalizable.return_value = True
    _run(client.finalize_ready_batches())
    contract.finalize_batch.reset_mock()

    # Second call: batch already locally finalized; skipped.
    _run(client.finalize_ready_batches())
    contract.finalize_batch.assert_not_awaited()


def test_finalize_retries_on_tx_failure():
    """If finalizeBatch reverts, the batch stays un-finalized locally so
    it's picked up again on the next poll."""
    cfg = AccumulatorConfig(count_threshold=1, time_threshold_seconds=3600, value_threshold_ftns=10**30)
    acc = ReceiptAccumulator(cfg)
    contract = _make_mock_contract()
    client = BatchSettlementClient(acc, contract, PROVIDER_ADDR)

    _run(client.accumulate(_make_batched()))
    _run(client.commit_ready_batches())

    contract.is_finalizable.return_value = True
    contract.finalize_batch.side_effect = RuntimeError("temporary RPC error")

    result = _run(client.finalize_ready_batches())
    assert len(result) == 1
    assert result[0].tx_submitted is False
    # Not marked locally-finalized on failure.
    assert not client.is_finalized_locally(result[0].batch_id)

    # Fix the flake; next poll succeeds.
    contract.finalize_batch.side_effect = None
    contract.finalize_batch.return_value = None
    result = _run(client.finalize_ready_batches())
    assert len(result) == 1
    assert result[0].tx_submitted is True


# ── Introspection + reconciliation ────────────────────────────────


def test_tracked_batches_returned_correctly():
    cfg = AccumulatorConfig(count_threshold=1, time_threshold_seconds=3600, value_threshold_ftns=10**30)
    acc = ReceiptAccumulator(cfg)
    contract = _make_mock_contract()
    client = BatchSettlementClient(acc, contract, PROVIDER_ADDR)

    _run(client.accumulate(_make_batched(requester=REQUESTER_A)))
    _run(client.accumulate(_make_batched(requester=REQUESTER_B)))
    committed = _run(client.commit_ready_batches())

    tracked = client.tracked_batches()
    assert len(tracked) == 2
    assert {c.batch_id for c in tracked} == {c.batch_id for c in committed}

    # get_tracked lookup
    for c in committed:
        assert client.get_tracked(c.batch_id) == c


def test_reconcile_finalized_picks_up_watchdog_finalizations():
    """Scenario: a third-party watchdog (§2.4 of design) finalized a batch
    while this client was offline. reconcile_finalized syncs local state
    by querying on-chain BatchStatus."""
    cfg = AccumulatorConfig(count_threshold=1, time_threshold_seconds=3600, value_threshold_ftns=10**30)
    acc = ReceiptAccumulator(cfg)
    contract = _make_mock_contract()
    client = BatchSettlementClient(acc, contract, PROVIDER_ADDR)

    _run(client.accumulate(_make_batched()))
    _run(client.commit_ready_batches())

    # On-chain status = 2 (FINALIZED).
    contract.get_batch_status.return_value = 2
    count = _run(client.reconcile_finalized())
    assert count == 1
    assert client.is_finalized_locally(client.tracked_batches()[0].batch_id)

    # Second reconcile: already marked locally; no new reconciliations.
    count = _run(client.reconcile_finalized())
    assert count == 0


def test_reconcile_ignores_non_finalized_statuses():
    cfg = AccumulatorConfig(count_threshold=1, time_threshold_seconds=3600, value_threshold_ftns=10**30)
    acc = ReceiptAccumulator(cfg)
    contract = _make_mock_contract()
    client = BatchSettlementClient(acc, contract, PROVIDER_ADDR)

    _run(client.accumulate(_make_batched()))
    _run(client.commit_ready_batches())

    contract.get_batch_status.return_value = 1  # PENDING
    count = _run(client.reconcile_finalized())
    assert count == 0


def test_get_tracked_returns_none_for_unknown_id():
    acc = ReceiptAccumulator()
    contract = _make_mock_contract()
    client = BatchSettlementClient(acc, contract, PROVIDER_ADDR)

    fake_id = hashlib.sha256(b"unknown").digest()
    assert client.get_tracked(fake_id) is None
    assert not client.is_finalized_locally(fake_id)


# ── Merkle-root integrity across commit ──────────────────────────


def test_commit_passes_correct_merkle_root_to_contract():
    """The merkle_root passed to commit_batch matches what Task 5's
    build_tree_and_proofs produces from the batch's leaf hashes."""
    from prsm.settlement.merkle import (
        batched_receipt_to_leaf,
        build_tree_and_proofs,
        hash_leaf,
    )

    cfg = AccumulatorConfig(count_threshold=2, time_threshold_seconds=3600, value_threshold_ftns=10**30)
    acc = ReceiptAccumulator(cfg)
    contract = _make_mock_contract()
    client = BatchSettlementClient(acc, contract, PROVIDER_ADDR)

    batched = [
        _make_batched(shard_index=0),
        _make_batched(shard_index=1),
    ]
    for br in batched:
        _run(client.accumulate(br))

    committed = _run(client.commit_ready_batches())
    assert len(committed) == 1

    # Independently compute what the merkle root SHOULD be.
    leaves = [batched_receipt_to_leaf(br) for br in batched]
    leaf_hashes = [hash_leaf(l) for l in leaves]
    expected_tree = build_tree_and_proofs(leaf_hashes)

    # Client's stored leaf hashes match.
    assert committed[0].leaf_hashes == tuple(leaf_hashes)

    # Contract was called with the matching root.
    kwargs = contract.commit_batch.await_args.kwargs
    assert kwargs["merkle_root"] == expected_tree.root


# ── Phase 7 + 7.1x extensions flow through commit_batch ───────────────


def _make_batched_with_extras(
    shard_index: int = 0,
    tier_slash_rate_bps: int = 0,
    consensus_group_id: bytes = b"\x00" * 32,
) -> BatchedReceipt:
    """Variant of _make_batched that overrides the Phase 7 / 7.1x fields."""
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


def test_commit_passes_tier_slash_rate_and_group_id_defaults():
    """Legacy (non-Phase-7) callers who don't populate the new fields
    get through with zero defaults, matching the pre-Phase-7.1x
    behaviour of non-consensus DOUBLE_SPEND-only batches."""
    cfg = AccumulatorConfig(
        count_threshold=2, time_threshold_seconds=3600,
        value_threshold_ftns=10**30,
    )
    acc = ReceiptAccumulator(cfg)
    contract = _make_mock_contract()
    client = BatchSettlementClient(acc, contract, PROVIDER_ADDR)

    for i in range(2):
        _run(client.accumulate(_make_batched(shard_index=i)))
    _run(client.commit_ready_batches())

    kwargs = contract.commit_batch.await_args.kwargs
    assert kwargs["tier_slash_rate_bps"] == 0
    assert kwargs["consensus_group_id"] == b"\x00" * 32


def test_commit_passes_tier_slash_rate_from_receipts():
    """When the orchestrator populates tier_slash_rate_bps on the
    receipts (Phase 7 snapshot at bond time), commit_batch receives
    that value — enabling the on-chain slash hook to fire on
    successful challenges."""
    cfg = AccumulatorConfig(
        count_threshold=2, time_threshold_seconds=3600,
        value_threshold_ftns=10**30,
    )
    acc = ReceiptAccumulator(cfg)
    contract = _make_mock_contract()
    client = BatchSettlementClient(acc, contract, PROVIDER_ADDR)

    for i in range(2):
        _run(client.accumulate(
            _make_batched_with_extras(shard_index=i, tier_slash_rate_bps=5000)
        ))
    _run(client.commit_ready_batches())

    kwargs = contract.commit_batch.await_args.kwargs
    assert kwargs["tier_slash_rate_bps"] == 5000


def test_commit_passes_consensus_group_id_from_receipts():
    """When receipts carry a non-zero consensus_group_id (Phase 7.1x
    k-of-n dispatch), commit_batch receives it. Without this the
    CONSENSUS_MISMATCH handler rejects any challenge against the
    resulting batch because `b.consensus_group_id == 0` short-circuits."""
    cfg = AccumulatorConfig(
        count_threshold=2, time_threshold_seconds=3600,
        value_threshold_ftns=10**30,
    )
    acc = ReceiptAccumulator(cfg)
    contract = _make_mock_contract()
    client = BatchSettlementClient(acc, contract, PROVIDER_ADDR)

    group = b"\xcc" * 32
    for i in range(2):
        _run(client.accumulate(
            _make_batched_with_extras(
                shard_index=i, tier_slash_rate_bps=10000,
                consensus_group_id=group,
            )
        ))
    _run(client.commit_ready_batches())

    kwargs = contract.commit_batch.await_args.kwargs
    assert kwargs["consensus_group_id"] == group
    assert kwargs["tier_slash_rate_bps"] == 10000


def test_commit_separates_batches_with_different_group_ids():
    """The accumulator-key extension means receipts from the same
    (requester, provider) pair with DIFFERENT consensus_group_ids
    land in separate batches; commit_ready_batches produces one
    CommittedBatch per group."""
    cfg = AccumulatorConfig(
        count_threshold=1, time_threshold_seconds=3600,
        value_threshold_ftns=10**30,
    )
    acc = ReceiptAccumulator(cfg)
    contract = _make_mock_contract()
    client = BatchSettlementClient(acc, contract, PROVIDER_ADDR)

    _run(client.accumulate(
        _make_batched_with_extras(shard_index=0, consensus_group_id=b"\x01" * 32)
    ))
    _run(client.accumulate(
        _make_batched_with_extras(shard_index=1, consensus_group_id=b"\x02" * 32)
    ))
    committed = _run(client.commit_ready_batches())
    assert len(committed) == 2
    # Each call got its own group_id.
    group_ids_seen = {
        call.kwargs["consensus_group_id"]
        for call in contract.commit_batch.await_args_list
    }
    assert group_ids_seen == {b"\x01" * 32, b"\x02" * 32}
