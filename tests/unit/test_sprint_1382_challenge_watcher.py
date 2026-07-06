"""Sprint 1382 — background settlement-challenge watcher.

Baselines pre-existing challenges (no alarm storm), then alerts once per NEW challenge; exposes an
active-challenge count for metrics. The client is faked so no chain is touched.
"""
import asyncio

import pytest

from prsm.economy.web3.settlement_challenge_watcher import SettlementChallengeWatcher


class _Chal:
    def __init__(self, bid, tx="0xtx"):
        self.batch_id = bid
        self.receipt_leaf_hash = "0xleaf-" + bid
        self.tx_hash = tx
        self.reason = "INVALID_SIGNATURE"
        self.invalidated_value_ftns = 0.5
        self.challenger = "0xcc"
        self.batch_status = 1


class _Client:
    """Returns a scripted challenge list per successive call (last entry repeats)."""

    def __init__(self, per_call):
        self._per_call = list(per_call)
        self.calls = 0

    def get_challenges_for_provider(self, _op, lookback_blocks=None):
        r = self._per_call[min(self.calls, len(self._per_call) - 1)]
        self.calls += 1
        return r


def test_baseline_seeds_without_alerting_then_alerts_new():
    alerts = []
    w = SettlementChallengeWatcher(
        _Client([[_Chal("b1")], [_Chal("b1"), _Chal("b2")]]),
        "0xop", on_new_challenge=alerts.append)
    asyncio.run(w.tick())                              # baseline: b1 pre-existing → no alert
    assert alerts == [] and w.active_challenge_count == 1
    asyncio.run(w.tick())                              # b1 known, b2 is new → one alert
    assert len(alerts) == 1 and alerts[0].batch_id == "b2"
    assert w.active_challenge_count == 2


def test_dedup_no_realert_on_same_challenge():
    alerts = []
    w = SettlementChallengeWatcher(
        _Client([[_Chal("b1")], [_Chal("b1")], [_Chal("b1")]]),
        "0xop", on_new_challenge=alerts.append)
    asyncio.run(w.tick())                              # baseline b1
    asyncio.run(w.tick())
    asyncio.run(w.tick())
    assert alerts == []                                # b1 never re-alerted


def test_new_after_empty_baseline_alerts():
    alerts = []
    w = SettlementChallengeWatcher(
        _Client([[], [_Chal("b9")]]), "0xop", on_new_challenge=alerts.append)
    asyncio.run(w.tick())                              # baseline: none
    asyncio.run(w.tick())                              # b9 new → alert
    assert [c.batch_id for c in alerts] == ["b9"]


def test_bad_callback_does_not_break_watcher():
    def boom(_c):
        raise RuntimeError("callback blew up")
    w = SettlementChallengeWatcher(
        _Client([[], [_Chal("b1")]]), "0xop", on_new_challenge=boom)
    asyncio.run(w.tick())
    asyncio.run(w.tick())                              # dispatch swallows the callback error
    assert w.active_challenge_count == 1               # tick still completed


def test_invalid_interval_rejected():
    with pytest.raises(ValueError):
        SettlementChallengeWatcher(_Client([[]]), "0xop", poll_interval_sec=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
