"""Sprint 1488 — the remote-dispatch adapter must fail closed, not default.

The adapter bridging TensorParallelExecutor to RemoteShardDispatcher read the
three fields that decide WHO gets paid, HOW MUCH, and WHICH ESCROW via
`assignment.get(k, <default>)`. Each default is a distinct money bug, not a
graceful fallback:

  job_id -> ""              every concurrent job's shard n collapses onto
                            ":shard:n", so one job's release settles another's
  escrow_amount -> 1.0      invents a price nobody quoted or agreed to
  node_id -> ""             dispatches to no one while still creating an escrow

A missing field means the caller is broken. The only safe response is to dispatch
nothing: an unpaid provider or a mis-settled escrow cannot be undone from here,
while refusing costs one failed job.
"""
from __future__ import annotations

import pytest

from prsm.compute.remote_dispatcher import (
    InvalidDispatchAssignment,
    _escrow_job_id,
    validate_dispatch_assignment,
)


def _ok(**over):
    a = {"node_id": "peer-1", "job_id": "job-abc", "escrow_amount_ftns": 2.5}
    a.update(over)
    return a


def test_a_complete_assignment_passes_through():
    assert validate_dispatch_assignment(_ok()) == ("peer-1", "job-abc", 2.5)


# ── the escrow-key collision ────────────────────────────────────────

def test_missing_job_id_is_REFUSED_not_defaulted():
    """★ THE collision. With job_id="" the escrow key is ':shard:0' — which every
    other concurrent job also produces, so one job's release settles another's
    escrow."""
    a = _ok()
    del a["job_id"]
    with pytest.raises(InvalidDispatchAssignment, match="no job_id"):
        validate_dispatch_assignment(a)


def test_empty_and_whitespace_job_id_are_both_refused():
    for bad in ("", "   ", None):
        with pytest.raises(InvalidDispatchAssignment, match="no job_id"):
            validate_dispatch_assignment(_ok(job_id=bad))


def test_escrow_key_builder_itself_refuses_an_empty_job_id():
    """Defence in depth: even a caller that skips validation cannot mint the
    colliding key."""
    with pytest.raises(InvalidDispatchAssignment, match="EMPTY job_id"):
        _escrow_job_id("", 0)


def test_escrow_keys_stay_distinct_across_jobs_and_shards():
    keys = {_escrow_job_id(j, s) for j in ("job-a", "job-b") for s in (0, 1)}
    assert len(keys) == 4


# ── the invented price ──────────────────────────────────────────────

def test_missing_escrow_amount_is_REFUSED_not_defaulted_to_one_ftns():
    """★ The amount must come from the accepted quote. A default silently pays
    something nobody agreed to — in either direction."""
    a = _ok()
    del a["escrow_amount_ftns"]
    with pytest.raises(InvalidDispatchAssignment, match="no escrow_amount_ftns"):
        validate_dispatch_assignment(a)


def test_zero_or_negative_escrow_is_refused():
    for bad in (0, -1.0):
        with pytest.raises(InvalidDispatchAssignment, match="must be positive"):
            validate_dispatch_assignment(_ok(escrow_amount_ftns=bad))


def test_non_finite_escrow_is_refused():
    """NaN/Inf survive float() and poison every downstream balance — sp1468."""
    for bad in (float("nan"), float("inf"), float("-inf"), "nan", "inf"):
        with pytest.raises(InvalidDispatchAssignment, match="not finite"):
            validate_dispatch_assignment(_ok(escrow_amount_ftns=bad))


def test_unparseable_escrow_is_refused():
    with pytest.raises(InvalidDispatchAssignment, match="not a number"):
        validate_dispatch_assignment(_ok(escrow_amount_ftns="free"))


def test_a_numeric_string_amount_is_accepted():
    """Assignments cross a JSON boundary, so a quoted number is legitimate."""
    assert validate_dispatch_assignment(_ok(escrow_amount_ftns="2.5"))[2] == 2.5


# ── the missing payee ───────────────────────────────────────────────

def test_missing_node_id_is_refused_before_any_escrow_exists():
    """Creating an escrow for a dispatch that can never happen strands the funds."""
    a = _ok()
    del a["node_id"]
    with pytest.raises(InvalidDispatchAssignment, match="no node_id"):
        validate_dispatch_assignment(a)


def test_node_id_is_checked_before_the_amount():
    """Ordering matters for the operator-facing message: report the structural
    problem, not a derived one."""
    with pytest.raises(InvalidDispatchAssignment, match="no node_id"):
        validate_dispatch_assignment({"escrow_amount_ftns": 0})


# ── the adapter really uses it ──────────────────────────────────────

def test_the_node_adapter_calls_the_validator():
    """★ Binding test: the validator is worthless if the live adapter still uses
    .get() defaults. Assert the defaults are GONE from the call site."""
    import inspect

    import prsm.node.node as node_mod

    src = inspect.getsource(node_mod)
    i = src.index("_tensor_remote_dispatch")
    body = src[i:i + 3000]
    assert "validate_dispatch_assignment(assignment)" in body
    assert 'assignment.get("job_id", "")' not in body
    assert 'assignment.get("node_id", "")' not in body
    assert 'assignment.get("escrow_amount_ftns", 1.0)' not in body


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
