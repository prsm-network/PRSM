"""Sprint 1475 — BatchSettlementManager money-path hardening (audit wf_530e5cd6).

The batch-settlement adversarial audit (queue → net → on-chain payout) found that
an owed provider payout could exit settlement without being paid or preserved.
Fixes (mirroring the sp1439/sp1474 withdraw reconciler — never silently drop):

  #1 A reverted ("rejected") net settlement was DROPPED (error only), not re-queued
     — yet an ERC-20 revert is atomic (no tokens moved), so it is still owed. The
     dominant cause is a transient operator hot-wallet FTNS shortfall. Now re-queued
     (bounded per-payee), then dead-lettered. (covered in test_sprint_914)
  #2 An ambiguous in-flight tx was DROPPED after a fixed attempt budget with no
     nonce-based proof-of-dead. Now kept until nonce-advance + receipt-absent PROVE
     it dead → re-queue; otherwise kept forever. (covered in test_sprint_917)
  #3 Concurrent threshold flushes inflated the per-call attempts counter and
     ABANDONED a still-pending tx prematurely. The attempts counter no longer gates
     any drop → concurrent flushes cannot strand a payout. (this file)
  #4 The in-flight tracker overflow trim silently discarded the oldest un-reconciled
     entry. Now dead-lettered + logged, never silent. (this file)
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from prsm.economy.batch_settlement import (
    BatchSettlementManager,
    PendingTransfer,
    SettlementMode,
)

pytestmark = pytest.mark.asyncio

_FROM = "0x" + "a" * 40
_TO = "0x" + "b" * 40


class _FakeEth:
    """No receipt for any tx, nonce never advances → nothing is provable-dead."""
    def get_transaction_receipt(self, h):
        from web3.exceptions import TransactionNotFound
        raise TransactionNotFound(h)

    def get_transaction_count(self, addr, block):
        return 0


def _mgr():
    led = MagicMock()
    led.w3 = SimpleNamespace(eth=_FakeEth())
    return BatchSettlementManager(
        ftns_ledger=led, node_id="n1", connected_address=_FROM,
        mode=SettlementMode.MANUAL,
    )


# ─────────────────── #3 concurrent flushes don't prematurely abandon ────────

async def test_many_concurrent_reconciles_never_abandon_a_pending_payout():
    """★ #3 — the OLD code bumped `attempts` per reconcile CALL and dropped the
    entry at _max_reconcile_attempts, so N concurrent threshold-flushes (each
    calls reconcile at its start) could abandon a still-pending settlement within
    seconds → the sp917 revert recovery is defeated and the payout is lost. Now
    the attempts counter gates NO drop; a not-provably-dead entry is always kept."""
    mgr = _mgr()
    mgr._track_in_flight(_FROM, _TO, 5.0, "0xpending", nonce=10)  # nonce not advanced (fake returns 0)
    # Fire far more concurrent reconciles than the old _max_reconcile_attempts budget.
    await asyncio.gather(*[mgr.reconcile_in_flight()
                           for _ in range(mgr._max_reconcile_attempts * 3)])
    assert len(mgr._in_flight) == 1, "a still-pending payout must never be abandoned"
    assert mgr._queue == []           # never auto-requeued (double-pay safe)
    assert mgr._dead_letter == []


async def test_attempts_counter_does_not_gate_a_drop():
    """The attempts counter is observability-only now — even far past the old
    budget, a not-provably-dead entry survives."""
    mgr = _mgr()
    mgr._track_in_flight(_FROM, _TO, 3.0, "0xstuck")   # no nonce → never provable dead
    for _ in range(mgr._max_reconcile_attempts + 10):
        await mgr.reconcile_in_flight()
    assert len(mgr._in_flight) == 1
    assert mgr._in_flight[0]["attempts"] > mgr._max_reconcile_attempts


# ─────────────────── #4 tracker-overflow trim is dead-lettered, not silent ──

async def test_in_flight_overflow_trim_dead_letters_not_silent():
    """★ #4 — overflowing the in-flight tracker past _max_in_flight used to drop
    the oldest un-reconciled owed payout with NO log and NO recovery. Now the
    trimmed entries are dead-lettered (surfaced) so an operator can reconcile."""
    mgr = _mgr()
    mgr._max_in_flight = 3
    for i in range(5):
        mgr._track_in_flight(_FROM, f"0x{i:040x}", float(i + 1), f"0xhash{i}", nonce=i)
    assert len(mgr._in_flight) == 3          # bounded
    assert len(mgr._dead_letter) == 2        # the 2 oldest were surfaced, not lost
    assert mgr.get_stats()["dead_letter_count"] == 2


# ─────────────────── stats surface ─────────────────────────────────────────

async def test_stats_exposes_in_flight_and_dead_letter():
    mgr = _mgr()
    mgr._track_in_flight(_FROM, _TO, 5.0, "0xtx", nonce=1)
    stats = mgr.get_stats()
    assert stats["in_flight"] == 1
    assert stats["dead_letter_count"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
