"""Sprint 1128 — finalize-countdown ETA is lag-resilient (display-only money-path fix).

The phase-1 (deposit-commit) summary in scripts/settlement_sepolia_e2e.py printed the
finalize countdown by trusting ``seconds_until_finalizable`` ALONE:

    secs = await contract.seconds_until_finalizable(batch_id)
    eta = "now" if secs <= 0 else f"~{secs/3600:.1f}h (at {secs}s)"

But the public Base-Sepolia RPC (sepolia.base.org) is load-balanced. The view call
fired immediately after the commit tx can hit a REPLICA THAT HASN'T YET SEEN the
just-committed batch. For a batch the registry does not yet know about (status=0
NONEXISTENT) the contract returns ``isFinalizable=False`` AND
``secondsUntilFinalizable=0`` — so the naive logic read secs=0 -> eta="now" and told
the operator the batch was finalizable IMMEDIATELY, nudging a pointless early finalize.

The three on-chain states to disambiguate:
  - Not-yet-visible / nonexistent (LAG): status=0, isFinalizable=False, secs=0.
  - Pending, window ACTIVE:             status=1, isFinalizable=False, secs>0.
  - Genuinely finalizable (elapsed):    status=1, isFinalizable=True,  secs=0.

The fix mirrors the sp1047/1048 deposit-balance lag-poll: poll past replica lag and
use ``is_finalizable()`` as the source of truth for whether the window has genuinely
elapsed. Return "now" ONLY when is_finalizable() is True.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from scripts.settlement_sepolia_e2e import _resolve_finalizable_eta


BATCH_ID = bytes.fromhex(
    "a184621c74d1b37eec72fced3f2c482c8abaeefd475bdf8b12b5843aab8f3300"
)


class _FakeContract:
    """Minimal fake of Web3SettlementContractClient's view surface.

    ``secs_seq`` / ``finalizable_seq`` are per-call sequences (the last value is
    repeated once exhausted), modelling a load-balanced RPC whose replicas catch up
    across successive reads.
    """

    def __init__(self, secs_seq, finalizable_seq, *, raise_on=None):
        self._secs_seq = list(secs_seq)
        self._fin_seq = list(finalizable_seq)
        self._raise_on = raise_on
        self.secs_calls = 0
        self.fin_calls = 0

    async def seconds_until_finalizable(self, batch_id):
        i = self.secs_calls
        self.secs_calls += 1
        if self._raise_on == "secs":
            raise RuntimeError("RPC error: replica unavailable")
        return self._secs_seq[min(i, len(self._secs_seq) - 1)]

    async def is_finalizable(self, batch_id):
        i = self.fin_calls
        self.fin_calls += 1
        if self._raise_on == "is_finalizable":
            raise RuntimeError("RPC error: replica unavailable")
        return self._fin_seq[min(i, len(self._fin_seq) - 1)]


async def _noop_sleep(_seconds):
    # Inject in place of asyncio.sleep so tests don't actually wait.
    return None


# ── (a) LAG REGRESSION — THE bug ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_replica_lag_does_not_report_false_now():
    """First N reads hit a stale replica (secs=0, isFinalizable=False); then the
    batch becomes visible (secs=86212, still PENDING). The helper must poll past the
    lag and return the REAL ~23.9h countdown — NOT 'now'."""
    # 2 lagged reads, then the visible PENDING value.
    contract = _FakeContract(
        secs_seq=[0, 0, 86212],
        finalizable_seq=[False, False, False],
    )
    eta = await _resolve_finalizable_eta(
        contract, BATCH_ID, retries=6, interval_s=0.0, sleep=_noop_sleep
    )
    assert eta != "now"
    assert "now" != eta
    assert eta == "~23.9h (at 86212s)"
    # It actually had to poll past the two stale reads.
    assert contract.secs_calls >= 3


@pytest.mark.asyncio
async def test_naive_logic_would_have_reported_false_now():
    """Red-proof: the OLD inline shape ('secs<=0 -> now') reports a FALSE 'now' on
    the very first lagged read. Documents exactly the bug the helper fixes."""
    contract = _FakeContract(
        secs_seq=[0, 0, 86212],
        finalizable_seq=[False, False, False],
    )
    secs = await contract.seconds_until_finalizable(BATCH_ID)
    naive_eta = "now" if secs <= 0 else f"~{secs/3600:.1f}h (at {secs}s)"
    assert naive_eta == "now"  # the bug: claims finalizable while replica is behind


# ── (b) GENUINE-NOW ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_genuine_now_when_finalizable_true():
    contract = _FakeContract(secs_seq=[0], finalizable_seq=[True])
    eta = await _resolve_finalizable_eta(
        contract, BATCH_ID, retries=6, interval_s=0.0, sleep=_noop_sleep
    )
    assert eta == "now"


# ── (c) ACTIVE WINDOW ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_active_window_returns_real_countdown():
    contract = _FakeContract(secs_seq=[86400], finalizable_seq=[False])
    eta = await _resolve_finalizable_eta(
        contract, BATCH_ID, retries=6, interval_s=0.0, sleep=_noop_sleep
    )
    assert eta == "~24.0h (at 86400s)"


# ── (d) EXTREME LAG ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_extreme_lag_returns_honest_fallback_not_now():
    """Every poll returns the lagged signature (secs=0, isFinalizable=False). After
    the bounded retries the helper must NOT claim 'now' — it returns an honest
    fallback that does not assert finalizability."""
    contract = _FakeContract(
        secs_seq=[0, 0, 0, 0, 0, 0, 0, 0],
        finalizable_seq=[False, False, False, False, False, False, False, False],
    )
    eta = await _resolve_finalizable_eta(
        contract, BATCH_ID, retries=4, interval_s=0.0, sleep=_noop_sleep
    )
    assert eta != "now"
    assert "replica lag" in eta
    assert "finalize" in eta  # points the operator at --phase finalize to re-check
    # It exhausted the bounded retries.
    assert contract.secs_calls >= 4


@pytest.mark.asyncio
async def test_rpc_error_returns_honest_fallback_no_crash():
    """Preserve the existing except-branch behaviour: an RPC error yields an honest
    fallback string (no crash, no false 'now')."""
    contract = _FakeContract(secs_seq=[0], finalizable_seq=[False], raise_on="secs")
    eta = await _resolve_finalizable_eta(
        contract, BATCH_ID, retries=6, interval_s=0.0, sleep=_noop_sleep
    )
    assert eta != "now"
    assert isinstance(eta, str) and eta
