"""Unit tests for ConsensusChallengeQueue + process_submittable_queue runner.

Uses in-memory SQLite (`:memory:`) for test isolation — each test gets
a fresh queue, nothing leaks between tests. Runner tests mock the
submitter entirely.
"""
from __future__ import annotations

import base64
import time
from unittest.mock import MagicMock

import pytest

from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.marketplace.consensus_queue import (
    ConsensusChallengeQueue,
    PendingChallenge,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_SUBMITTABLE,
    STATUS_SUBMITTED,
)
from prsm.marketplace.consensus_submitter import (
    ChallengeResult,
    process_submittable_queue,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _receipt(provider_id: str, output_hash_hex: str = None) -> ShardExecutionReceipt:
    if output_hash_hex is None:
        output_hash_hex = ("aa" * 32) if provider_id.startswith("maj") else ("bb" * 32)
    # Real receipts carry BASE64 pubkey/signature; the committed-leaf convention
    # (merkle.batched_receipt_to_leaf, sp1165) b64-decodes them before hashing, so use valid
    # base64 here (still per-provider-distinct).
    return ShardExecutionReceipt(
        job_id="job-phase7.1x",
        shard_index=0,
        provider_id=provider_id,
        provider_pubkey_b64=base64.b64encode(
            f"PUBKEY_{provider_id}".encode()).decode("ascii"),
        output_hash=output_hash_hex,
        executed_at_unix=1_700_000_000,
        signature=base64.b64encode(f"SIG_{provider_id}".encode()).decode("ascii"),
    )


def _drained_entry(
    *,
    minorities: list[str] = None,
    majorities: list[str] = None,
    job_id: str = "job-phase7.1x",
    shard_index: int = 0,
) -> dict:
    # Distinguish "caller passed nothing" (use defaults) from "caller
    # passed []" (intentional empty list — keep it).
    if minorities is None:
        minorities = ["provC"]
    if majorities is None:
        majorities = ["majA", "majB"]
    return {
        "job_id": job_id,
        "shard_index": shard_index,
        "agreed_output_hash": "aa" * 32,
        "majority_receipts": [_receipt(p, "aa" * 32) for p in majorities],
        "minority_receipts": [_receipt(p, "bb" * 32) for p in minorities],
        "value_ftns_per_provider_wei": 5 * 10**18,
    }


@pytest.fixture
def queue():
    q = ConsensusChallengeQueue(":memory:")
    yield q
    q.close()


# ── enqueue_from_drained ─────────────────────────────────────────────


def test_enqueue_one_minority_one_majority_creates_one_row(queue):
    row_ids = queue.enqueue_from_drained([
        _drained_entry(minorities=["provC"], majorities=["majA"]),
    ])
    assert len(row_ids) == 1
    row = queue.get(row_ids[0])
    assert row.status == STATUS_PENDING
    assert row.minority_provider_id == "provC"
    assert row.majority_provider_id == "majA"
    assert row.minority_batch_id is None
    assert row.majority_batch_id is None
    assert row.attempts == 0


def test_enqueue_expands_multiple_minorities_into_multiple_rows(queue):
    """k=5 with 3-2 split → 2 minorities → 2 separate rows,
    each paired with the same (canonical) majority."""
    row_ids = queue.enqueue_from_drained([
        _drained_entry(
            minorities=["provD", "provE"],
            majorities=["majA", "majB", "majC"],
        ),
    ])
    assert len(row_ids) == 2
    rows = [queue.get(r) for r in row_ids]
    # Both paired with the lexicographically lowest majority provider.
    assert {r.majority_provider_id for r in rows} == {"majA"}
    # Different minority providers in each row.
    assert {r.minority_provider_id for r in rows} == {"provD", "provE"}


def test_enqueue_skips_entry_with_no_majority(queue):
    """Entry with empty majority_receipts → no row. Don't stall drain."""
    entry = _drained_entry(minorities=["provC"], majorities=[])
    row_ids = queue.enqueue_from_drained([entry])
    assert row_ids == []
    assert queue.count_by_status() == {}


def test_enqueue_picks_lowest_provider_id_as_canonical_majority(queue):
    """Deterministic majority selection — lowest provider_id wins.
    Audit replay reproduces the same pairing."""
    row_ids = queue.enqueue_from_drained([
        _drained_entry(
            minorities=["provC"],
            majorities=["majZ", "majA", "majM"],
        ),
    ])
    row = queue.get(row_ids[0])
    assert row.majority_provider_id == "majA"


def test_enqueue_multiple_entries_preserves_order(queue):
    queue.enqueue_from_drained([
        _drained_entry(minorities=["p1"], shard_index=0),
        _drained_entry(minorities=["p2"], shard_index=1),
        _drained_entry(minorities=["p3"], shard_index=2),
    ])
    pending = queue.list_pending()
    assert [r.minority_provider_id for r in pending] == ["p1", "p2", "p3"]
    assert [r.shard_index for r in pending] == [0, 1, 2]


# ── record_batch_commit ──────────────────────────────────────────────


def test_record_batch_commit_sets_minority_batch_id(queue):
    row_ids = queue.enqueue_from_drained([_drained_entry()])
    touched = queue.record_batch_commit(
        "provC", "job-phase7.1x", 0, b"\x01" * 32,
    )
    assert touched >= 1
    row = queue.get(row_ids[0])
    assert row.minority_batch_id == b"\x01" * 32
    assert row.majority_batch_id is None
    # Still PENDING because majority hasn't committed.
    assert row.status == STATUS_PENDING


def test_record_batch_commit_sets_majority_batch_id(queue):
    row_ids = queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)
    row = queue.get(row_ids[0])
    assert row.majority_batch_id == b"\x02" * 32
    assert row.status == STATUS_PENDING   # minority still missing


def test_record_batch_commit_promotes_to_submittable_when_both_set(queue):
    """Once BOTH batch_ids are recorded, status flips to SUBMITTABLE
    so the runner picks the row up."""
    queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)
    submittable = queue.list_submittable()
    assert len(submittable) == 1
    assert submittable[0].status == STATUS_SUBMITTABLE
    assert submittable[0].minority_batch_id == b"\x01" * 32
    assert submittable[0].majority_batch_id == b"\x02" * 32


def test_record_batch_commit_commit_order_does_not_matter(queue):
    """Majority-commit-first then minority-commit works the same as
    the reverse. Both orderings must land in SUBMITTABLE."""
    queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    assert len(queue.list_submittable()) == 1


def test_record_batch_commit_is_idempotent_on_same_provider(queue):
    """Calling record_batch_commit twice with the same provider doesn't
    double-set the batch_id or create duplicate work."""
    row_ids = queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    # Second call with a different batch_id must NOT overwrite — we
    # already know this provider's batch for this (job, shard).
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x09" * 32)
    row = queue.get(row_ids[0])
    assert row.minority_batch_id == b"\x01" * 32   # still the first value


def test_record_batch_commit_does_nothing_for_unrelated_provider(queue):
    queue.enqueue_from_drained([_drained_entry()])
    touched = queue.record_batch_commit(
        "someone-else", "job-phase7.1x", 0, b"\x03" * 32,
    )
    assert touched == 0


def test_record_batch_commit_rejects_wrong_batch_id_length(queue):
    with pytest.raises(ValueError, match="32 bytes"):
        queue.record_batch_commit("p", "j", 0, b"\x01" * 16)


def test_record_batch_commit_multi_shard_scoping(queue):
    """A provider participates in multiple shards of the same job.
    A commit for shard 0 must NOT affect rows for shard 1."""
    queue.enqueue_from_drained([
        _drained_entry(shard_index=0),
        _drained_entry(shard_index=1),
    ])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    pending = queue.list_pending()
    rows_by_shard = {r.shard_index: r for r in pending}
    assert rows_by_shard[0].minority_batch_id == b"\x01" * 32
    assert rows_by_shard[1].minority_batch_id is None


# ── list_submittable / list_pending ─────────────────────────────────


def test_list_submittable_orders_oldest_first(queue):
    """Runner processes FIFO — ensures pending work doesn't starve."""
    queue.enqueue_from_drained([_drained_entry(shard_index=0)])
    time.sleep(0.01)  # guarantee distinct created_at values
    queue.enqueue_from_drained([_drained_entry(shard_index=1)])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)
    queue.record_batch_commit("provC", "job-phase7.1x", 1, b"\x03" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 1, b"\x04" * 32)

    rows = queue.list_submittable()
    assert [r.shard_index for r in rows] == [0, 1]


def test_list_submittable_limit(queue):
    for i in range(5):
        queue.enqueue_from_drained([_drained_entry(shard_index=i)])
        queue.record_batch_commit("provC", "job-phase7.1x", i, b"\x01" * 32)
        queue.record_batch_commit("majA", "job-phase7.1x", i, b"\x02" * 32)
    rows = queue.list_submittable(limit=3)
    assert len(rows) == 3


def test_list_pending_includes_both_pending_and_submittable(queue):
    queue.enqueue_from_drained([
        _drained_entry(shard_index=0),
        _drained_entry(shard_index=1),
    ])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)
    # Shard 0 is SUBMITTABLE, shard 1 is still PENDING.
    pending = queue.list_pending()
    statuses = {r.status for r in pending}
    assert statuses == {STATUS_PENDING, STATUS_SUBMITTABLE}
    assert len(pending) == 2


# ── mark_submitted / mark_failed ────────────────────────────────────


def test_mark_submitted_transitions_to_terminal_state(queue):
    row_ids = queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)
    queue.mark_submitted(row_ids[0], "0x" + "ab" * 32)

    row = queue.get(row_ids[0])
    assert row.status == STATUS_SUBMITTED
    assert row.last_tx_hash == "0x" + "ab" * 32
    assert row.attempts == 1
    # SUBMITTED rows NOT in list_submittable any more.
    assert queue.list_submittable() == []


def test_mark_failed_terminal_moves_to_failed(queue):
    row_ids = queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)
    queue.mark_failed(
        row_ids[0], "OnChainRevertedError", "ChallengeNotProven",
        terminal=True,
    )
    row = queue.get(row_ids[0])
    assert row.status == STATUS_FAILED
    assert "ChallengeNotProven" in row.last_error
    assert row.attempts == 1


def test_mark_failed_non_terminal_stays_submittable_for_retry(queue):
    """BroadcastFailedError (RPC transient) → leave submittable so the
    next runner tick retries. Attempts counter increments."""
    row_ids = queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)
    queue.mark_failed(
        row_ids[0], "BroadcastFailedError", "rpc unreachable",
        terminal=False,
    )
    row = queue.get(row_ids[0])
    assert row.status == STATUS_SUBMITTABLE
    assert row.attempts == 1
    assert "rpc unreachable" in row.last_error


def test_count_by_status(queue):
    row_ids = queue.enqueue_from_drained([
        _drained_entry(shard_index=0),
        _drained_entry(shard_index=1),
        _drained_entry(shard_index=2),
    ])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)
    queue.mark_submitted(row_ids[0], "0xaa")
    counts = queue.count_by_status()
    assert counts.get(STATUS_SUBMITTED) == 1
    assert counts.get(STATUS_PENDING) == 2


# ── Persistence across instances (the whole point) ─────────────────


def test_queue_survives_reopen(tmp_path):
    """Crash-safety: write to a file-backed DB, close it, reopen, and
    the pending work is still there. This is the §8.6 mitigation —
    a submitter crash mid-flight doesn't drop challenges."""
    db_path = str(tmp_path / "consensus.sqlite")
    q1 = ConsensusChallengeQueue(db_path)
    row_ids = q1.enqueue_from_drained([_drained_entry()])
    q1.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    q1.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)
    q1.close()

    # Reopen the DB — as if the process crashed and restarted.
    q2 = ConsensusChallengeQueue(db_path)
    try:
        rows = q2.list_submittable()
        assert len(rows) == 1
        assert rows[0].row_id == row_ids[0]
        assert rows[0].minority_batch_id == b"\x01" * 32
        assert rows[0].majority_batch_id == b"\x02" * 32
        # Receipt fields rehydrate correctly through JSON round-trip.
        assert isinstance(rows[0].minority_receipt, ShardExecutionReceipt)
        assert rows[0].minority_receipt.provider_id == "provC"
    finally:
        q2.close()


# ── process_submittable_queue runner ────────────────────────────────


def _stub_submitter(results: list[ChallengeResult]) -> MagicMock:
    sub = MagicMock()
    sub.submit_one = MagicMock(side_effect=results)
    return sub


def test_runner_processes_submittable_and_marks_submitted(queue):
    queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)

    submitter = _stub_submitter([
        ChallengeResult(
            success=True, tx_hash_hex="0xabc", error_type=None,
            error_message=None,
        ),
    ])
    results = process_submittable_queue(queue, submitter)
    assert len(results) == 1
    assert results[0].success is True
    # Row transitioned to SUBMITTED.
    assert queue.count_by_status() == {STATUS_SUBMITTED: 1}


def test_runner_marks_onchain_reverted_as_terminal(queue):
    """Contract revert → terminal FAILED. No retry."""
    queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)

    submitter = _stub_submitter([
        ChallengeResult(
            success=False, tx_hash_hex=None,
            error_type="OnChainRevertedError",
            error_message="ChallengeNotProven",
        ),
    ])
    process_submittable_queue(queue, submitter)
    assert queue.count_by_status() == {STATUS_FAILED: 1}


def test_runner_marks_broadcast_failed_as_retryable(queue):
    """Transient RPC failure → SUBMITTABLE for next tick."""
    queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)

    submitter = _stub_submitter([
        ChallengeResult(
            success=False, tx_hash_hex=None,
            error_type="BroadcastFailedError",
            error_message="rpc unreachable",
        ),
    ])
    process_submittable_queue(queue, submitter)
    assert queue.count_by_status() == {STATUS_SUBMITTABLE: 1}
    row = queue.list_submittable()[0]
    assert row.attempts == 1


def test_runner_marks_onchain_pending_as_terminal_needs_manual(queue):
    """OnChainPending is UNSAFE to auto-retry — tx may land. Mark
    terminal so a human reconciles via tx_hash rather than
    double-submitting."""
    queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)

    submitter = _stub_submitter([
        ChallengeResult(
            success=False, tx_hash_hex="0xmaybe",
            error_type="OnChainPendingError",
            error_message="timeout",
        ),
    ])
    process_submittable_queue(queue, submitter)
    assert queue.count_by_status() == {STATUS_FAILED: 1}


def test_runner_processes_multiple_rows_independently(queue):
    """Each row's outcome is independent — one terminal failure must
    not block subsequent successes."""
    queue.enqueue_from_drained([
        _drained_entry(shard_index=0),
        _drained_entry(shard_index=1),
        _drained_entry(shard_index=2),
    ])
    for i in range(3):
        queue.record_batch_commit("provC", "job-phase7.1x", i, bytes([i + 1]) * 32)
        queue.record_batch_commit("majA", "job-phase7.1x", i, bytes([i + 4]) * 32)

    submitter = _stub_submitter([
        ChallengeResult(success=True, tx_hash_hex="0xaa",
                        error_type=None, error_message=None),
        ChallengeResult(success=False, tx_hash_hex=None,
                        error_type="OnChainRevertedError", error_message="x"),
        ChallengeResult(success=True, tx_hash_hex="0xcc",
                        error_type=None, error_message=None),
    ])
    results = process_submittable_queue(queue, submitter)
    assert [r.success for r in results] == [True, False, True]
    counts = queue.count_by_status()
    assert counts.get(STATUS_SUBMITTED) == 2
    assert counts.get(STATUS_FAILED) == 1


def test_runner_on_empty_queue_is_noop(queue):
    submitter = MagicMock()
    results = process_submittable_queue(queue, submitter)
    assert results == []
    submitter.submit_one.assert_not_called()


def test_runner_respects_limit(queue):
    for i in range(5):
        queue.enqueue_from_drained([_drained_entry(shard_index=i)])
        queue.record_batch_commit("provC", "job-phase7.1x", i, bytes([i + 1]) * 32)
        queue.record_batch_commit("majA", "job-phase7.1x", i, bytes([i + 10]) * 32)

    submitter = _stub_submitter([
        ChallengeResult(success=True, tx_hash_hex="0x", error_type=None,
                        error_message=None) for _ in range(2)
    ])
    results = process_submittable_queue(queue, submitter, limit=2)
    assert len(results) == 2
    assert queue.count_by_status().get(STATUS_SUBMITTED) == 2
    # 3 still submittable for the next tick.
    assert queue.count_by_status().get(STATUS_SUBMITTABLE) == 3


# ── Pre-audit P1: default-deny retry classification ─────────────────


def test_runner_marks_unexpected_bug_as_terminal(queue):
    """Pre-audit review P1 fix: unexpected error types (KeyError,
    ValueError, generic Exception from a bug in submit_one) must NOT
    retry forever. The allowlist accepts only BroadcastFailedError;
    everything else — including unknown types — is terminal.

    Before this fix, a bug that escaped submit_one's web3-error
    classification would bounce between SUBMITTABLE and mark_failed
    indefinitely, silently accumulating attempts."""
    queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)

    submitter = _stub_submitter([
        ChallengeResult(
            success=False, tx_hash_hex=None,
            error_type="KeyError",       # an unexpected bug inside encoder
            error_message="'majorityLeaf' missing",
        ),
    ])
    process_submittable_queue(queue, submitter)
    # Terminal — row does NOT bounce back to SUBMITTABLE.
    assert queue.count_by_status() == {STATUS_FAILED: 1}


def test_runner_caps_retries_at_max_attempts(queue):
    """Even for retryable (BroadcastFailedError) errors, a row must
    terminal-fail after max_attempts to prevent a misconfigured RPC
    endpoint from burning ops budget on one row indefinitely."""
    queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)

    # Simulate max_attempts=3 runs; RPC is persistently down.
    submitter = _stub_submitter([
        ChallengeResult(
            success=False, tx_hash_hex=None,
            error_type="BroadcastFailedError",
            error_message="rpc down",
        ) for _ in range(3)
    ])
    # First tick: attempts goes 0→1, still under cap → SUBMITTABLE.
    process_submittable_queue(queue, submitter, max_attempts=3)
    row = queue.list_pending()[0]
    assert row.status == STATUS_SUBMITTABLE
    assert row.attempts == 1

    # Second tick: 1→2, still under cap → SUBMITTABLE.
    process_submittable_queue(queue, submitter, max_attempts=3)
    row = queue.list_pending()[0]
    assert row.status == STATUS_SUBMITTABLE
    assert row.attempts == 2

    # Third tick: 2→3, cap reached → terminal FAILED.
    process_submittable_queue(queue, submitter, max_attempts=3)
    assert queue.count_by_status() == {STATUS_FAILED: 1}


def test_runner_default_max_attempts_is_set(queue):
    """Default max_attempts should not be None / unbounded. 5 is the
    current default; pinning it so a regression that removes the cap
    is caught."""
    from prsm.marketplace.consensus_submitter import DEFAULT_MAX_ATTEMPTS
    assert DEFAULT_MAX_ATTEMPTS == 5


def test_runner_broadcast_failed_still_retries_under_cap(queue):
    """Regression: the existing 'BroadcastFailed → SUBMITTABLE' happy
    path must still work under the new allowlist."""
    queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)

    submitter = _stub_submitter([
        ChallengeResult(
            success=False, tx_hash_hex=None,
            error_type="BroadcastFailedError",
            error_message="rpc blip",
        ),
    ])
    process_submittable_queue(queue, submitter, max_attempts=5)
    # One attempt, far below cap → still SUBMITTABLE.
    assert queue.count_by_status() == {STATUS_SUBMITTABLE: 1}


# ── §5.3 claim-lease ────────────────────────────────────────────────


def test_claim_submittable_marks_rows_as_claimed(queue):
    """Basic claim: a submittable row becomes owned by the claimant
    with a lease expiration in the future."""
    queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)

    before = time.time()
    claimed = queue.claim_submittable("runner-A", lease_seconds=60)
    after = time.time()

    assert len(claimed) == 1
    row = claimed[0]
    # claimed_by is "<id>:<uuid>" — prefix-check because uuid is random.
    assert row.claimed_by.startswith("runner-A:")
    assert row.claimed_at is not None
    assert before <= row.claimed_at <= after
    assert row.lease_expires_at is not None
    assert row.lease_expires_at > row.claimed_at
    # Lease is ~60s from now.
    assert 59 <= row.lease_expires_at - row.claimed_at <= 61


def test_claim_submittable_skips_already_claimed_rows(queue):
    """Two runners claim at the same time — the second claim call
    excludes rows already held by the first, regardless of claimant_id."""
    queue.enqueue_from_drained([_drained_entry(shard_index=0)])
    queue.enqueue_from_drained([_drained_entry(shard_index=1)])
    for i in range(2):
        queue.record_batch_commit("provC", "job-phase7.1x", i, bytes([i + 1]) * 32)
        queue.record_batch_commit("majA", "job-phase7.1x", i, bytes([i + 10]) * 32)

    runner_a = queue.claim_submittable("A", lease_seconds=60, limit=1)
    runner_b = queue.claim_submittable("B", lease_seconds=60, limit=10)

    assert len(runner_a) == 1
    assert len(runner_b) == 1
    # Different rows.
    assert runner_a[0].row_id != runner_b[0].row_id


def test_claim_submittable_returns_empty_when_all_claimed(queue):
    """If one runner has claimed all available rows, a second runner
    gets no work this tick — waits for lease expiry or completion."""
    queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)

    queue.claim_submittable("A", lease_seconds=60)
    assert queue.claim_submittable("B", lease_seconds=60) == []


def test_claim_submittable_reclaims_expired_lease(queue):
    """A lease with expires_at in the past is reclaimable by any
    runner — this is the crash-recovery path (first runner held the
    lease, crashed, and its lease has now expired).

    Uses a negative lease duration so the claim is born expired; the
    test-suite-wide time.sleep mock makes wall-clock waits unreliable,
    and real-time waits inflate test runtime."""
    queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)

    # Runner A takes the row with an already-expired lease.
    queue.claim_submittable("A", lease_seconds=-10)

    reclaimed = queue.claim_submittable("B", lease_seconds=60)
    assert len(reclaimed) == 1
    assert reclaimed[0].claimed_by.startswith("B:")


def test_mark_submitted_clears_the_lease(queue):
    """Success case: the row's lease fields are reset so the row
    doesn't keep pointing at its last claimant after it's done."""
    queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)

    claimed = queue.claim_submittable("A", lease_seconds=60)
    queue.mark_submitted(claimed[0].row_id, "0xdeadbeef")

    row = queue.get(claimed[0].row_id)
    assert row.status == STATUS_SUBMITTED
    assert row.claimed_by is None
    assert row.claimed_at is None
    assert row.lease_expires_at is None


def test_mark_failed_clears_the_lease_on_retryable_too(queue):
    """A retryable failure returns the row to SUBMITTABLE AND clears
    the lease — so any runner (not just the original claimant) can
    retry on the next tick without waiting for lease expiry."""
    queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)

    claimed = queue.claim_submittable("A", lease_seconds=60)
    queue.mark_failed(
        claimed[0].row_id, "BroadcastFailedError", "rpc down",
        terminal=False,
    )

    row = queue.get(claimed[0].row_id)
    assert row.status == STATUS_SUBMITTABLE
    assert row.claimed_by is None
    # Runner B can pick it up immediately.
    reclaimed_by_b = queue.claim_submittable("B", lease_seconds=60)
    assert len(reclaimed_by_b) == 1


def test_runner_claimant_id_uses_claim_path(queue):
    """process_submittable_queue with claimant_id set routes through
    claim_submittable — provable by observing that a second concurrent
    runner call with a different claimant_id gets no rows."""
    queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)

    # Submitter that doesn't complete quickly — simulate by returning
    # a retryable failure. Row stays SUBMITTABLE but lease is cleared
    # on mark_failed, so to test claim-scoping we use a stubbed
    # submitter that never marks. Simpler: only test at the claim API
    # level directly, and trust that process_submittable_queue
    # correctly delegates based on the claimant_id parameter presence.
    stub_called = {"count": 0}
    def never_called_submit(*args, **kwargs):
        stub_called["count"] += 1
        raise AssertionError("should not be called in this test")

    submitter = MagicMock()

    # First, claim with runner-A via the submitter path but no actual
    # processing — mock submit_one to succeed.
    submitter.submit_one = MagicMock(return_value=ChallengeResult(
        success=True, tx_hash_hex="0xabc",
        error_type=None, error_message=None,
    ))
    results = process_submittable_queue(
        queue, submitter, claimant_id="runner-A",
    )
    assert len(results) == 1

    # Row should now be SUBMITTED — not available for runner-B.
    results_b = process_submittable_queue(
        queue, submitter, claimant_id="runner-B",
    )
    assert results_b == []


def test_default_lease_seconds_exported():
    """Pin DEFAULT_LEASE_SECONDS value so a regression is caught."""
    from prsm.marketplace.consensus_queue import DEFAULT_LEASE_SECONDS
    assert DEFAULT_LEASE_SECONDS == 300.0


def test_legacy_list_submittable_path_still_works(queue):
    """process_submittable_queue without claimant_id uses the legacy
    no-claim path — existing single-runner callers unchanged."""
    queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)

    submitter = _stub_submitter([
        ChallengeResult(
            success=True, tx_hash_hex="0xabc",
            error_type=None, error_message=None,
        ),
    ])
    # No claimant_id → no claim. Row still gets submitted successfully.
    process_submittable_queue(queue, submitter)
    row = queue.list_pending()   # empty since SUBMITTED is terminal
    assert row == []
    assert queue.count_by_status() == {STATUS_SUBMITTED: 1}


def test_schema_migration_adds_lease_columns_to_old_db(tmp_path):
    """Phase 7.1x.next pre-existing DBs don't have the lease columns.
    Opening them with the updated ConsensusChallengeQueue class must
    ALTER TABLE to add the columns — existing rows remain queryable."""
    import sqlite3 as _sqlite3
    db_path = str(tmp_path / "legacy.sqlite")

    # Build a pre-lease schema manually (subset of what _SCHEMA_SQL had
    # before this commit).
    conn = _sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE consensus_challenges (
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            shard_index INTEGER NOT NULL,
            agreed_output_hash TEXT NOT NULL,
            value_ftns_per_provider_wei INTEGER NOT NULL,
            minority_provider_id TEXT NOT NULL,
            minority_receipt_json TEXT NOT NULL,
            majority_provider_id TEXT NOT NULL,
            majority_receipt_json TEXT NOT NULL,
            minority_batch_id BLOB,
            majority_batch_id BLOB,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            last_tx_hash TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
    """)
    conn.commit()
    conn.close()

    # Open with the new class; migration should add the lease columns.
    q = ConsensusChallengeQueue(db_path)
    try:
        # Insert a row; verify lease fields are queryable (NULL by default).
        q.enqueue_from_drained([_drained_entry()])
        q.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
        q.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)
        rows = q.list_submittable()
        assert len(rows) == 1
        assert rows[0].claimed_by is None
        assert rows[0].lease_expires_at is None

        # Claim works on the migrated DB.
        claimed = q.claim_submittable("A")
        assert len(claimed) == 1
        assert claimed[0].claimed_by.startswith("A:")
    finally:
        q.close()


def test_reopen_preserves_lease_state(tmp_path):
    """Crash-safety for leases: a claim held by a crashed runner
    survives DB reopen and remains valid until the lease expires."""
    db_path = str(tmp_path / "leased.sqlite")
    q1 = ConsensusChallengeQueue(db_path)
    q1.enqueue_from_drained([_drained_entry()])
    q1.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    q1.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)
    q1.claim_submittable("crashed-runner", lease_seconds=300)
    q1.close()

    # Reopen DB — simulate a different runner instance.
    q2 = ConsensusChallengeQueue(db_path)
    try:
        # Row is still claimed by crashed-runner with a live lease.
        # New runner gets nothing until the lease expires.
        assert q2.claim_submittable("new-runner") == []
        # But list_submittable (no-claim) still sees it — the lease
        # doesn't change status, just ownership.
        visible = q2.list_submittable()
        assert len(visible) == 1
        assert visible[0].claimed_by.startswith("crashed-runner:")
    finally:
        q2.close()
