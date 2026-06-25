"""Sprint 1249 — close the Redis wrapper-misuse sweep: core/auth/rate_limiter.py.

The last instance of the sp1247 systemic bug. RateLimiter held the RedisClient
WRAPPER (get_redis_client()) and called raw commands (.zadd/.zcard/.get/.setex/...)
directly on it across ~13 sites — each raised AttributeError, swallowed by a
fail-open/fail-safe except, so the sliding-window limiter, IP reputation, and
blacklist all silently degraded. (This limiter is NOT wired into the live request
path — only audit_checklist references it — so this is the lowest-impact site, but
closing it eliminates the bug so future wiring can't resurrect it.)

Fix: a `_live_redis()` helper (sp1247 resolve_async_redis) used at every site.
"""
from __future__ import annotations

import pytest

from prsm.core.auth.rate_limiter import RateLimiter


class _Inner:
    """Stands in for the real async Redis (the wrapper's inner .redis_client)."""
    def __init__(self, *, zcard=0, get_val=None):
        self._zcard = zcard
        self._get = get_val

    async def zremrangebyscore(self, *a, **k):
        return 0

    async def zcard(self, *a, **k):
        return self._zcard

    async def zrange(self, *a, **k):
        return [("member", 1000.0)]

    async def zadd(self, *a, **k):
        return 1

    async def expire(self, *a, **k):
        return True

    async def get(self, *a, **k):
        return self._get

    async def setex(self, *a, **k):
        return True


class _Wrapper:
    """Stands in for RedisClient — NO raw commands; inner .redis_client; .connected."""
    def __init__(self, *, connected, inner):
        self.connected = connected
        self.redis_client = inner


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def _limiter(wrapper):
    rl = RateLimiter()
    rl.redis_client = wrapper
    return rl


def test_sliding_window_blocks_over_limit_via_inner_client():
    # connected wrapper, inner reports 5 in-window entries vs limit 3 → blocked.
    # Pre-fix: wrapper.zremrangebyscore raised → except → (True,0,0) fail-open.
    rl = _limiter(_Wrapper(connected=True, inner=_Inner(zcard=5)))
    allowed, count, _ = _run(rl._check_sliding_window("k", limit=3, window=60))
    assert allowed is False        # enforced via the inner client, NOT failed open
    assert count == 5


def test_sliding_window_allows_under_limit_via_inner_client():
    rl = _limiter(_Wrapper(connected=True, inner=_Inner(zcard=2)))
    allowed, count, _ = _run(rl._check_sliding_window("k", limit=3, window=60))
    assert allowed is True
    assert count == 2


def test_sliding_window_disconnected_wrapper_allows():
    # disconnected wrapper → resolve None → the documented no-Redis allow path.
    rl = _limiter(_Wrapper(connected=False, inner=None))
    allowed, count, remaining = _run(rl._check_sliding_window("k", limit=3, window=60))
    assert (allowed, count, remaining) == (True, 0, 0)


def test_ip_reputation_reads_inner_client():
    # a stored low reputation must be read via the inner client (pre-fix the wrapper
    # .get raised → except → always returned 1.0 "good", disabling reputation gating).
    rl = _limiter(_Wrapper(connected=True, inner=_Inner(get_val="0.2")))
    assert _run(rl._get_ip_reputation("1.2.3.4")) == pytest.approx(0.2)


def test_ip_reputation_disconnected_returns_neutral_good():
    rl = _limiter(_Wrapper(connected=False, inner=None))
    assert _run(rl._get_ip_reputation("1.2.3.4")) == 1.0


def test_live_redis_helper_resolves_inner_and_none():
    inner = _Inner()
    assert _limiter(_Wrapper(connected=True, inner=inner))._live_redis() is inner
    assert _limiter(_Wrapper(connected=False, inner=inner))._live_redis() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
