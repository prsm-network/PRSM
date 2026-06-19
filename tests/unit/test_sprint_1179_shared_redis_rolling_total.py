"""Sprint 1179 — SHARED (cross-replica) Redis-backed fiat AML rolling total.

The sp1175 finding: the per-user AML tier limit was enforced against the
PROCESS-LOCAL FiatComplianceRing, so a multi-replica deployment let a user
fanned across N replicas transact up to N× their limit. sp1175 shipped the
fail-closed `strict_shared` lever (deny rather than under-enforce). sp1179 is
the REAL fix: an opt-in `shared_redis` mode where the rolling total lives in a
shared Redis store, so the limit is enforced GLOBALLY.

The correctness proof is a DIFFERENTIAL test: identical event sequences fed to
the in-memory ring and the Redis-backed ring must yield byte-equal totals
(incl. the sp968 intent dedup, the sp1176 EXPIRED $0 supersession, no-intent
each-counts, and rolling-window expiry). Plus API-level tests proving
shared_redis ENFORCES the global total and FAILS CLOSED when Redis is down.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.economy.web3.coinbase_waas_client import CoinbaseWaaSClient
from prsm.economy.web3.fiat_compliance_ring import (
    FiatComplianceRing,
    RedisRollingTotal,
)
from prsm.economy.web3.kyc_client import KYCClient
from prsm.node.api import create_api_app


# ── Minimal faithful sync fake of the sorted-set commands ─────────────
# RedisRollingTotal uses zadd / zrangebyscore(withscores) /
# zremrangebyscore / expire. A legitimate test double for the injected
# client — it exercises RedisRollingTotal's real accounting logic.


def _parse_bound(b):
    s = str(b)
    excl = s.startswith("(")
    if excl:
        s = s[1:]
    if s == "-inf":
        return float("-inf"), False
    if s in ("+inf", "inf"):
        return float("inf"), False
    return float(s), excl


def _in_range(score, lo, lo_excl, hi, hi_excl):
    if lo_excl and not (score > lo):
        return False
    if not lo_excl and not (score >= lo):
        return False
    if hi_excl and not (score < hi):
        return False
    if not hi_excl and not (score <= hi):
        return False
    return True


class _FakeSortedSetRedis:
    def __init__(self):
        self._z = {}      # key -> {member: score}
        self._ttl = {}    # key -> ttl (observed, not enforced)
        self.fail = False  # flip True to simulate Redis unreachable

    def _maybe_fail(self):
        if self.fail:
            raise RuntimeError("simulated redis down")

    def zadd(self, key, mapping):
        self._maybe_fail()
        d = self._z.setdefault(key, {})
        for m, s in mapping.items():
            d[m] = float(s)
        return len(mapping)

    def expire(self, key, ttl):
        self._maybe_fail()
        self._ttl[key] = ttl
        return True

    def zremrangebyscore(self, key, min_, max_):
        self._maybe_fail()
        d = self._z.get(key, {})
        lo, lo_excl = _parse_bound(min_)
        hi, hi_excl = _parse_bound(max_)
        gone = [m for m, s in d.items()
                if _in_range(s, lo, lo_excl, hi, hi_excl)]
        for m in gone:
            del d[m]
        return len(gone)

    def zrangebyscore(self, key, min_, max_, withscores=False):
        self._maybe_fail()
        d = self._z.get(key, {})
        lo, lo_excl = _parse_bound(min_)
        hi, hi_excl = _parse_bound(max_)
        items = [(m, s) for m, s in d.items()
                 if _in_range(s, lo, lo_excl, hi, hi_excl)]
        items.sort(key=lambda x: (x[1], x[0]))  # score asc, then member
        return items if withscores else [m for m, _ in items]


def _shared_ring(fake):
    return FiatComplianceRing(
        shared_total=RedisRollingTotal(fake, window_sec=86400),
    )


def _feed(ring, events):
    for ev in events:
        ring.record(
            kind=ev["kind"],
            user_id=ev["user"],
            usd_amount=ev["usd"],
            ftns_amount=0.0,
            status=ev["status"],
            metadata=ev.get("md", {}),
            timestamp=ev["ts"],
        )


# ── RedisRollingTotal correctness ─────────────────────────────────────


class TestRedisRollingTotalUnit:
    def test_records_and_sums_single_event(self):
        fake = _FakeSortedSetRedis()
        rrt = RedisRollingTotal(fake, window_sec=86400)
        import time as _t
        now = _t.time()
        rrt.record(
            user_id="alice", dedup_key="intent-1", entry_id="e1",
            usd_amount=500.0, timestamp=now - 10,
        )
        assert rrt.total_usd_for_user(
            "alice", window_sec=86400, now=now,
        ) == 500.0

    def test_read_propagates_redis_error(self):
        """A read error must RAISE so the tier check can fail closed."""
        fake = _FakeSortedSetRedis()
        rrt = RedisRollingTotal(fake)
        fake.fail = True
        with pytest.raises(Exception):
            rrt.total_usd_for_user("alice", window_sec=86400, now=1.0)

    def test_write_error_is_logged_not_raised(self):
        """record() is best-effort — a write error must NOT raise."""
        fake = _FakeSortedSetRedis()
        rrt = RedisRollingTotal(fake)
        fake.fail = True
        # Must not raise.
        rrt.record(
            user_id="alice", dedup_key="i", entry_id="e",
            usd_amount=1.0, timestamp=1.0,
        )


# ── DIFFERENTIAL: shared-redis total == in-memory total ───────────────


class TestDifferentialEquivalence:
    """Identical event sequences → identical totals on both backends."""

    def _assert_equiv(self, events, users):
        import time as _t
        now = _t.time()
        # Stamp ts relative to now so both backends see the same window.
        stamped = [{**ev, "ts": now + ev["ts_off"]} for ev in events]
        mem = FiatComplianceRing()                      # process-local
        shared = _shared_ring(_FakeSortedSetRedis())    # redis-backed
        _feed(mem, stamped)
        _feed(shared, stamped)
        for u in users:
            m = mem.total_usd_for_user(u)
            s = shared.total_usd_for_user(u)
            assert m == pytest.approx(s), (
                f"user {u}: in-memory={m} != shared={s}"
            )

    def test_simple_single_onramp(self):
        self._assert_equiv(
            [{"kind": "onramp_execute", "user": "alice", "usd": 500.0,
              "status": "PENDING", "md": {"intent_id": "i1"},
              "ts_off": -10}],
            ["alice"],
        )

    def test_intent_supersession_pending_then_confirmed(self):
        # sp968 — PENDING reservation then CONFIRMED settle, same intent
        # → counts ONCE (latest by timestamp wins).
        self._assert_equiv(
            [
                {"kind": "onramp_execute", "user": "a", "usd": 500.0,
                 "status": "PENDING", "md": {"intent_id": "i1"},
                 "ts_off": -100},
                {"kind": "onramp_execute", "user": "a", "usd": 480.0,
                 "status": "CONFIRMED", "md": {"intent_id": "i1"},
                 "ts_off": -10},
            ],
            ["a"],
        )

    def test_expired_zero_supersedes(self):
        # sp1176 — abandoned intent's EXPIRED $0 entry supersedes the
        # PENDING reservation → contributes 0.
        self._assert_equiv(
            [
                {"kind": "onramp_execute", "user": "a", "usd": 500.0,
                 "status": "PENDING", "md": {"intent_id": "i1"},
                 "ts_off": -100},
                {"kind": "onramp_execute", "user": "a", "usd": 0.0,
                 "status": "EXPIRED", "md": {"intent_id": "i1"},
                 "ts_off": -10},
            ],
            ["a"],
        )

    def test_no_intent_entries_each_count(self):
        # offramp / legacy (no intent_id) each count individually.
        self._assert_equiv(
            [
                {"kind": "offramp_execute", "user": "a", "usd": 300.0,
                 "status": "CONFIRMED", "ts_off": -10},
                {"kind": "offramp_execute", "user": "a", "usd": 200.0,
                 "status": "CONFIRMED", "ts_off": -8},
            ],
            ["a"],
        )

    def test_window_expiry_excludes_old(self):
        self._assert_equiv(
            [
                {"kind": "onramp_execute", "user": "a", "usd": 999.0,
                 "status": "CONFIRMED", "md": {"intent_id": "old"},
                 "ts_off": -100000},  # well outside the 24h window
                {"kind": "onramp_execute", "user": "a", "usd": 25.0,
                 "status": "CONFIRMED", "md": {"intent_id": "new"},
                 "ts_off": -50},
            ],
            ["a"],
        )

    def test_quotes_excluded_gasless_excluded(self):
        # Only *_execute kinds burn the limit (sp885). Quotes + gasless
        # + kyc must not count, on BOTH backends.
        self._assert_equiv(
            [
                {"kind": "onramp_quote", "user": "a", "usd": 9999.0,
                 "status": "PENDING_COMMISSION", "ts_off": -10},
                {"kind": "gasless_transfer_execute", "user": "a",
                 "usd": 0.0, "status": "DONE", "ts_off": -10},
                {"kind": "onramp_execute", "user": "a", "usd": 75.0,
                 "status": "CONFIRMED", "md": {"intent_id": "i1"},
                 "ts_off": -10},
            ],
            ["a"],
        )

    def test_multi_user_mixed(self):
        self._assert_equiv(
            [
                {"kind": "onramp_execute", "user": "a", "usd": 500.0,
                 "status": "PENDING", "md": {"intent_id": "a1"},
                 "ts_off": -30},
                {"kind": "onramp_execute", "user": "a", "usd": 510.0,
                 "status": "CONFIRMED", "md": {"intent_id": "a1"},
                 "ts_off": -10},
                {"kind": "offramp_execute", "user": "b", "usd": 200.0,
                 "status": "CONFIRMED", "ts_off": -20},
                {"kind": "onramp_execute", "user": "b", "usd": 100.0,
                 "status": "CONFIRMED", "md": {"intent_id": "b1"},
                 "ts_off": -15},
            ],
            ["a", "b", "nobody"],
        )


# ── API-level: shared_redis ENFORCES global + FAILS CLOSED on Redis down


class _FakeWaasBackend:
    def create_wallet(self, user_id, email):
        return {"wallet_id": f"w-{user_id}",
                "address": "0x" + "11" * 20, "network": "base-mainnet"}


class _FakeKYCBackend:
    def initiate_session(self, user_id, email, level):
        return {"vendor_ref": f"persona-{user_id}",
                "session_url": f"https://persona.example/v/{user_id}",
                "status": "INITIATED"}


def _waas_with(user_id):
    c = CoinbaseWaaSClient(
        api_key_name="k", api_key_private="p", backend=_FakeWaasBackend(),
    )
    c.provision_wallet(user_id=user_id, email="a@x.io")
    return c


def _kyc_verified(user_id, level):
    kyc = KYCClient(vendor="persona", api_key="k", backend=_FakeKYCBackend())
    kyc.initiate(user_id=user_id, email="a@x.io", level=level)
    kyc.update_status(user_id, "VERIFIED")
    return kyc


def _client(*, waas, kyc, ring):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node._coinbase_waas_client = waas
    node._kyc_client = kyc
    node._aerodrome_client = None
    node._fiat_compliance_ring = ring
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


class TestSharedRedisApiEnforcement:
    def test_shared_redis_enforces_global_total_over_limit(
        self, monkeypatch,
    ):
        """A pre-seeded $950 GLOBAL total + a $100 onramp exceeds the
        $1k basic limit → 403 tier_limit_exceeded (enforced via the
        shared store, not a per-pod view)."""
        monkeypatch.setenv("PRSM_FIAT_TIER_LIMIT_MODE", "shared_redis")
        ring = _shared_ring(_FakeSortedSetRedis())
        import time as _t
        ring.record(
            kind="onramp_execute", user_id="alice", usd_amount=950.0,
            ftns_amount=0.0, status="CONFIRMED",
            metadata={"intent_id": "prior"}, timestamp=_t.time() - 60,
        )
        resp = _client(
            waas=_waas_with("alice"), kyc=_kyc_verified("alice", "basic"),
            ring=ring,
        ).post(
            "/wallet/onramp/execute",
            json={"usd_amount": 100.0, "destination_user_id": "alice"},
        )
        assert resp.status_code == 403
        assert resp.json()["detail"]["error"] == "tier_limit_exceeded"

    def test_shared_redis_allows_within_global_limit(self, monkeypatch):
        monkeypatch.setenv("PRSM_FIAT_TIER_LIMIT_MODE", "shared_redis")
        ring = _shared_ring(_FakeSortedSetRedis())
        import time as _t
        ring.record(
            kind="onramp_execute", user_id="alice", usd_amount=950.0,
            ftns_amount=0.0, status="CONFIRMED",
            metadata={"intent_id": "prior"}, timestamp=_t.time() - 60,
        )
        resp = _client(
            waas=_waas_with("alice"), kyc=_kyc_verified("alice", "basic"),
            ring=ring,
        ).post(
            "/wallet/onramp/execute",
            json={"usd_amount": 40.0, "destination_user_id": "alice"},
        )
        assert resp.status_code == 200  # 950 + 40 < 1000

    def test_shared_redis_fails_closed_when_redis_down(self, monkeypatch):
        """Redis unreachable → the rolling read raises → FAIL CLOSED
        (503), NOT a fail-open under-enforcement off a 0.0 total."""
        monkeypatch.setenv("PRSM_FIAT_TIER_LIMIT_MODE", "shared_redis")
        fake = _FakeSortedSetRedis()
        fake.fail = True
        ring = _shared_ring(fake)
        resp = _client(
            waas=_waas_with("alice"), kyc=_kyc_verified("alice", "basic"),
            ring=ring,
        ).post(
            "/wallet/onramp/execute",
            json={"usd_amount": 50.0, "destination_user_id": "alice"},
        )
        assert resp.status_code == 503
        assert resp.json()["detail"]["error"] == (
            "tier_limit_enforcement_unavailable"
        )

    def test_shared_redis_misconfig_no_backend_fails_closed(
        self, monkeypatch,
    ):
        """mode=shared_redis but the ring has NO shared backend wired
        (misconfig) → fail closed rather than silently under-enforce."""
        monkeypatch.setenv("PRSM_FIAT_TIER_LIMIT_MODE", "shared_redis")
        ring = FiatComplianceRing()  # no shared_total
        resp = _client(
            waas=_waas_with("alice"), kyc=_kyc_verified("alice", "basic"),
            ring=ring,
        ).post(
            "/wallet/onramp/execute",
            json={"usd_amount": 50.0, "destination_user_id": "alice"},
        )
        assert resp.status_code == 503
