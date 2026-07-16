"""Sprint 1464 — ATOMIC AML check-and-reserve (closes the shared_redis check→record race).

The AML tier limit on /wallet/onramp/execute read the rolling total (total_usd_for_user) and later
recorded the reservation (record → zadd) as TWO separate round-trips. In shared_redis (multi-replica)
mode, N concurrent onramps all read the same under-limit total, all minted, all recorded → a user
could structure PAST their tier limit by fanning concurrent requests across replicas.

RedisRollingTotal.try_reserve does the windowed-total read + limit compare + conditional zadd inside
ONE WATCH/MULTI transaction, so a concurrent modification aborts the EXEC → retry with the fresh
total. FiatComplianceRing.try_reserve delegates to it (shared_redis) or preserves record()+True
(process_local). Verified with a REAL fake Redis (WATCH/MULTI-capable), including a threaded
contention test proving no over-limit structuring.
"""
from __future__ import annotations

import threading

import pytest

fakeredis = pytest.importorskip("fakeredis")

from prsm.economy.web3.fiat_compliance_ring import (  # noqa: E402
    FiatComplianceRing,
    RedisRollingTotal,
)

_WINDOW = 86400


def _client(server=None):
    return fakeredis.FakeStrictRedis(
        server=server or fakeredis.FakeServer(), decode_responses=True)


def _roll(server=None):
    return RedisRollingTotal(_client(server))


# ── RedisRollingTotal.try_reserve — atomic core ───────────────────────────────

def test_reserve_under_limit_records():
    roll = _roll()
    ok = roll.try_reserve(
        user_id="u1", dedup_key="intent-1", entry_id="e1", usd_amount=600.0,
        timestamp=1000.0, limit_usd=1000.0, window_sec=_WINDOW, now=1000.0)
    assert ok is True
    assert roll.total_usd_for_user("u1", window_sec=_WINDOW, now=1000.0) == 600.0


def test_reserve_that_would_breach_is_rejected_and_records_nothing():
    roll = _roll()
    assert roll.try_reserve(
        user_id="u1", dedup_key="i1", entry_id="e1", usd_amount=600.0,
        timestamp=1000.0, limit_usd=1000.0, window_sec=_WINDOW, now=1000.0) is True
    # 600 + 500 = 1100 > 1000 → rejected, total unchanged.
    assert roll.try_reserve(
        user_id="u1", dedup_key="i2", entry_id="e2", usd_amount=500.0,
        timestamp=1001.0, limit_usd=1000.0, window_sec=_WINDOW, now=1001.0) is False
    assert roll.total_usd_for_user("u1", window_sec=_WINDOW, now=1001.0) == 600.0
    # 600 + 300 = 900 <= 1000 → accepted.
    assert roll.try_reserve(
        user_id="u1", dedup_key="i3", entry_id="e3", usd_amount=300.0,
        timestamp=1002.0, limit_usd=1000.0, window_sec=_WINDOW, now=1002.0) is True
    assert roll.total_usd_for_user("u1", window_sec=_WINDOW, now=1002.0) == 900.0


def test_reserve_exactly_at_limit_is_allowed():
    roll = _roll()
    assert roll.try_reserve(
        user_id="u", dedup_key="i", entry_id="e", usd_amount=1000.0,
        timestamp=1.0, limit_usd=1000.0, window_sec=_WINDOW, now=1.0) is True


def test_reserve_same_dedup_key_is_idempotent_not_double_counted():
    # Re-reserving the SAME intent replaces its contribution (no inflation) — models a PENDING that is
    # re-observed; the deduped total must not grow to 1200.
    roll = _roll()
    assert roll.try_reserve(
        user_id="u", dedup_key="intent-X", entry_id="e1", usd_amount=600.0,
        timestamp=10.0, limit_usd=1000.0, window_sec=_WINDOW, now=10.0) is True
    assert roll.try_reserve(
        user_id="u", dedup_key="intent-X", entry_id="e2", usd_amount=600.0,
        timestamp=11.0, limit_usd=1000.0, window_sec=_WINDOW, now=11.0) is True
    assert roll.total_usd_for_user("u", window_sec=_WINDOW, now=11.0) == 600.0


def test_distinct_users_have_independent_limits():
    roll = _roll()
    assert roll.try_reserve(user_id="a", dedup_key="i", entry_id="e", usd_amount=900.0,
                            timestamp=1.0, limit_usd=1000.0, window_sec=_WINDOW, now=1.0) is True
    assert roll.try_reserve(user_id="b", dedup_key="i", entry_id="e", usd_amount=900.0,
                            timestamp=1.0, limit_usd=1000.0, window_sec=_WINDOW, now=1.0) is True


def test_concurrent_reservations_never_exceed_the_limit():
    # ★ THE FIX: 40 threads each try to reserve $100 against a $1000 limit on a SHARED fake Redis.
    # Atomic WATCH/MULTI ⇒ AT MOST 10 succeed and the recorded total NEVER exceeds the limit. Without
    # atomicity (separate read-then-record) many more would slip through the check→record window.
    server = fakeredis.FakeServer()
    limit, amount, n = 1000.0, 100.0, 40
    results = []
    lock = threading.Lock()

    def worker(i):
        roll = RedisRollingTotal(_client(server))       # each thread its own client, shared server
        try:
            ok = roll.try_reserve(
                user_id="burst", dedup_key=f"intent-{i}", entry_id=f"e{i}",
                usd_amount=amount, timestamp=100.0 + i, limit_usd=limit,
                window_sec=_WINDOW, now=100.0 + i, max_retries=200)
            with lock:
                results.append(ok)
        except RuntimeError:
            with lock:
                results.append("error")               # retry-exhaustion fails CLOSED (safe)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    successes = sum(1 for r in results if r is True)
    final_total = RedisRollingTotal(_client(server)).total_usd_for_user(
        "burst", window_sec=_WINDOW, now=100.0 + n)
    assert successes <= 10, f"{successes} reservations succeeded — structured past the limit!"
    assert final_total <= limit, f"recorded total {final_total} exceeds the ${limit} limit"
    assert final_total == successes * amount            # recorded total is exactly the winners
    assert successes >= 1                               # not trivially rejecting everything


# ── FiatComplianceRing.try_reserve — the ring wrapper ─────────────────────────

def test_ring_shared_redis_atomic_reserve_and_local_mirror():
    roll = _roll()
    ring = FiatComplianceRing(shared_total=roll)
    ok = ring.try_reserve(user_id="u", dedup_key="intent-1", usd_amount=400.0,
                          limit_usd=1000.0, timestamp=5.0)
    assert ok is True
    # authoritative shared total reflects it, AND it's mirrored to the local audit deque.
    assert roll.total_usd_for_user("u", window_sec=_WINDOW, now=5.0) == 400.0
    assert ring.count() == 1
    assert ring.recent(user_id="u")[0].metadata.get("intent_id") == "intent-1"


def test_ring_shared_redis_reserve_breach_records_nothing():
    roll = _roll()
    ring = FiatComplianceRing(shared_total=roll)
    assert ring.try_reserve(user_id="u", dedup_key="i1", usd_amount=800.0,
                            limit_usd=1000.0, timestamp=5.0) is True
    assert ring.try_reserve(user_id="u", dedup_key="i2", usd_amount=500.0,
                            limit_usd=1000.0, timestamp=6.0) is False
    assert roll.total_usd_for_user("u", window_sec=_WINDOW, now=6.0) == 800.0
    assert ring.count() == 1                            # the rejected reserve left NO local entry


def test_ring_shared_no_double_write_to_shared_store():
    # try_reserve must NOT also go through record()'s shared-mirror (that would double the shared zadd).
    roll = _roll()
    ring = FiatComplianceRing(shared_total=roll)
    ring.try_reserve(user_id="u", dedup_key="i", usd_amount=300.0, limit_usd=1000.0, timestamp=5.0)
    assert roll.total_usd_for_user("u", window_sec=_WINDOW, now=5.0) == 300.0   # 300, not 600


def test_ring_process_local_reserve_preserves_record_behavior():
    # No shared store → process_local: records + returns True (the API's pre-mint tier check gates
    # this mode; single-replica semantics unchanged).
    ring = FiatComplianceRing()                         # no shared_total
    assert ring.try_reserve(user_id="u", dedup_key="i", usd_amount=999999.0,
                            limit_usd=1000.0) is True     # default timestamp = now (within the window)
    assert ring.count() == 1
    assert ring.total_usd_for_user("u") == 999999.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
