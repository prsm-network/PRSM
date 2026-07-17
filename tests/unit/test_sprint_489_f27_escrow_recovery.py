"""Sprint 489 — F27 escrow leak fixes.

F27 had THREE compounding root causes:

1. `payment_escrow.create_escrow` registered the in-memory
   record BEFORE the funds transfer. If transfer raised
   anything other than ValueError (e.g.,
   ConcurrentModificationError from dag_ledger), the record
   stayed in `_escrows` but funds didn't move → orphaned
   record on the in-memory side.

2. `payment_escrow.create_escrow` only caught ValueError.
   Other exceptions propagated up with the escrow record
   half-registered.

3. `dag_ledger.submit_transaction` did NOT call
   `await self._db.commit()` at the end of transfer-style
   transactions. The wallet_balances UPDATE and
   dag_transactions INSERT were buffered. The
   `_seed_welcome_grant` startup hook deletes + rebuilds
   wallet_balances from dag_transactions, so any
   uncommitted writes were lost on daemon restart.

Sprint 489 shipped:
  (a) Flipped ordering in create_escrow — record registered
      AFTER successful transfer.
  (b) Broadened exception catch in create_escrow + re-raise
      on non-ValueError (so caller knows to back off).
  (c) Explicit `await self._db.commit()` at end of
      submit_transaction.
  (d) New admin endpoint `/admin/escrow/recover-orphans`
      for operator-actionable orphan recovery without
      database edits.

Live-verified: 27 orphan escrows totaling 1081 FTNS
recovered via the admin endpoint; balance survived daemon
restart at 1084.11 (proving (c) committed durably).
Full concurrency suite (sprint 487) now 4/4 PASS.

These pins defend all four invariants in source.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_create_escrow_registers_record_after_transfer():
    """Source pin: in payment_escrow.create_escrow, the
    `self._escrows[escrow.escrow_id] = escrow` line must
    appear AFTER `tx = await self.ledger.transfer(...)`.
    Pre-fix, registration came FIRST → orphaned record on
    transfer failure."""
    src = (
        REPO_ROOT / "prsm" / "node" / "payment_escrow.py"
    ).read_text()
    create_idx = src.find("async def create_escrow")
    assert create_idx >= 0
    body = src[create_idx:create_idx + 5000]
    # Find both lines.
    register_idx = body.find(
        "self._escrows[escrow.escrow_id] = escrow"
    )
    transfer_idx = body.find(
        "await self.ledger.transfer("
    )
    assert register_idx >= 0, "register line missing"
    assert transfer_idx >= 0, "transfer line missing"
    assert register_idx > transfer_idx, (
        "F27 regression: _escrows registration must come "
        "AFTER ledger.transfer succeeds, not before"
    )


def test_create_escrow_catches_non_value_error_exceptions():
    """Source pin: create_escrow must handle ALL exception
    types, not just ValueError. Pre-fix, ConcurrentModification
    Error propagated up with funds half-transferred."""
    src = (
        REPO_ROOT / "prsm" / "node" / "payment_escrow.py"
    ).read_text()
    create_idx = src.find("async def create_escrow")
    body = src[create_idx:create_idx + 5000]
    # Must have the broad except clause.
    assert "except Exception as e:" in body, (
        "create_escrow must catch broad Exception for F27 "
        "defensive cleanup — pre-fix only ValueError was "
        "handled"
    )


def test_submit_transaction_commits_explicitly():
    """Source pin: dag_ledger.submit_transaction must call
    `await self._db.commit()` after the balance-credit
    step for transfer-style transactions. Pre-fix, only
    no-source `credit` transactions committed; transfers
    relied on a connection-close commit that was lost
    across daemon restart (when `_seed_welcome_grant`
    DELETE+REBUILD wallet_balances)."""
    src = (
        REPO_ROOT / "prsm" / "node" / "dag_ledger.py"
    ).read_text()
    submit_idx = src.find("async def submit_transaction")
    assert submit_idx >= 0
    # Find the post-credit commit by its sprint 489 marker. Bound the search to the ACTUAL method
    # (up to the next class-method def) rather than a fixed char window, so unrelated additions to
    # submit_transaction (e.g. the sp1468 amount guard) don't push the marker out of range.
    _end = src.find("\n    async def ", submit_idx + len("async def submit_transaction"))
    body = src[submit_idx:_end if _end != -1 else len(src)]
    assert "Sprint 489 (F27 fix) — durability barrier" in body, (
        "F27 regression: sprint 489 durability-barrier "
        "commit marker missing from submit_transaction"
    )
    # The commit call itself must be present.
    assert "await self._db.commit()" in body


def test_admin_orphan_recovery_endpoint_exists():
    """Source pin: /admin/escrow/recover-orphans endpoint
    must exist for operator-actionable recovery. Without
    this, operators whose daemon crashed mid-test would
    have FTNS permanently locked with no path short of
    a database edit."""
    api_src = (
        REPO_ROOT / "prsm" / "node" / "api.py"
    ).read_text()
    assert (
        '"/admin/escrow/recover-orphans"' in api_src
        or "/admin/escrow/recover-orphans" in api_src
    )
    # The handler must accept a `dry_run` flag for safety.
    idx = api_src.find("recover-orphans")
    region = api_src[idx:idx + 5000]
    assert "dry_run" in region, (
        "recovery endpoint must default to dry_run for "
        "operator safety — direct execution against a "
        "running daemon is destructive"
    )


def test_admin_orphan_recovery_returns_canonical_envelope():
    """The endpoint response shape — dashboards consume it."""
    api_src = (
        REPO_ROOT / "prsm" / "node" / "api.py"
    ).read_text()
    idx = api_src.find("recover-orphans")
    region = api_src[idx:idx + 5000]
    for field in (
        '"dry_run"',
        '"scanned"',
        '"recoverable"',
        '"refunded"',
        '"total_ftns_recovered"',
        '"errors"',
    ):
        assert field in region, (
            f"recovery response missing field: {field}"
        )


def test_admin_orphan_recovery_marker_in_source():
    """The fix's sprint marker must remain in source so
    a code archaeologist can trace why this exists."""
    api_src = (
        REPO_ROOT / "prsm" / "node" / "api.py"
    ).read_text()
    assert "Sprint 489 (F27 recovery)" in api_src
