"""Sprint 939 — heartbeat scheduler consecutive-miss escalation (storage review).

A storage provider that misses heartbeats becomes slashable (permissionless
slash_for_missing_heartbeat + a 70% challenger bounty) once its on-chain grace
window elapses — so a sustained RPC/connectivity outage marches an HONEST
operator toward losing its stake. The scheduler previously retried each missed
tick quietly (cumulative failure_count only), giving the operator no clear
"slashing imminent" signal.

sp939 adds a consecutive-failure streak: after CRITICAL_CONSECUTIVE_FAILURES in a
row (~3/4 of the grace window at the grace/4 auto-tune cadence — about one
interval before the stake becomes slashable), the scheduler logs CRITICAL and
fires an optional on_critical alert hook so the operator can intervene. A success
resets the streak. This is the off-chain mitigation of the (deploy-gated)
permissionless-slash griefing risk.
"""
from __future__ import annotations

import pytest

from prsm.economy.web3.heartbeat_scheduler import HeartbeatScheduler
from prsm.economy.web3.provenance_registry import BroadcastFailedError


class _Client:
    def __init__(self, fail: bool = True):
        self.fail = fail

    def record_heartbeat(self):
        if self.fail:
            raise BroadcastFailedError("rpc down")
        return ("0xabc", "confirmed")


def _scheduler(client, on_critical=None):
    # explicit interval → no auto-tune (client needs no grace method)
    return HeartbeatScheduler(client, interval_seconds=900, on_critical=on_critical)


@pytest.mark.asyncio
async def test_consecutive_failures_escalate_to_critical():
    alerts = []
    s = _scheduler(_Client(fail=True), on_critical=lambda n: alerts.append(n))
    await s.tick()
    assert s.consecutive_failures == 1 and alerts == []
    await s.tick()
    assert s.consecutive_failures == 2 and alerts == []
    await s.tick()
    assert s.consecutive_failures == 3 and alerts == [3]   # threshold reached
    await s.tick()
    assert s.consecutive_failures == 4 and alerts == [3, 4]  # keeps alerting while down
    # cumulative counter still tracks every failure
    assert s.failure_count == 4


@pytest.mark.asyncio
async def test_success_resets_the_streak():
    alerts = []
    c = _Client(fail=True)
    s = _scheduler(c, on_critical=lambda n: alerts.append(n))
    await s.tick()
    await s.tick()              # 2 failures, below threshold → no alert
    assert alerts == []
    c.fail = False
    await s.tick()              # success
    assert s.consecutive_failures == 0
    assert s.success_count == 1
    c.fail = True
    await s.tick()
    await s.tick()              # 2 fresh failures
    assert alerts == []         # streak was reset; 2 < threshold
    await s.tick()              # 3rd consecutive → escalate
    assert alerts == [3]


@pytest.mark.asyncio
async def test_on_critical_exception_does_not_crash_the_loop():
    def boom(_n):
        raise RuntimeError("alert sink down")
    s = _scheduler(_Client(fail=True), on_critical=boom)
    for _ in range(3):
        await s.tick()          # must not raise despite the failing hook
    assert s.consecutive_failures == 3


@pytest.mark.asyncio
async def test_no_critical_without_a_hook_is_safe():
    # on_critical is optional — escalation still logs CRITICAL, never crashes.
    s = _scheduler(_Client(fail=True), on_critical=None)
    for _ in range(4):
        await s.tick()
    assert s.consecutive_failures == 4
