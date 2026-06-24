"""Sprint 1247 — Redis rate-limiting fail-open bug: wrapper vs inner client.

The live mainnet pay-infer run surfaced `'RedisClient' object has no attribute
'pipeline'`. Three security components held the RedisClient WRAPPER
(get_redis_client()) but called `.pipeline()` on it — the wrapper has no
.pipeline (its inner `.redis_client` is the real async Redis). The AttributeError
was caught by an `except` that FAILS OPEN (allow the request), so rate limiting was
silently DISABLED on the Redis backend:
  - prsm/node/api_hardening.py        RateLimiter._check_rate_limit_redis
  - prsm/core/auth/middleware.py      rate-limit middleware
  - prsm/core/auth/enhanced_middleware.py  rate-limit + anomaly detection

Fix: a shared resolve_async_redis() that returns the live inner async Redis (or
None when disconnected), applied at every site; and the api_hardening limiter now
DEGRADES to its local in-memory limiter on a Redis error instead of failing open.
"""
from __future__ import annotations

import pytest

from prsm.core.redis_client import resolve_async_redis


# ── resolve_async_redis ──────────────────────────────────────────────────────

class _Raw:
    """Looks like a raw redis.asyncio.Redis — it HAS .pipeline."""
    def pipeline(self):
        return object()


class _Wrapper:
    """Looks like prsm.core.redis_client.RedisClient — NO top-level .pipeline;
    inner `.redis_client` is the real client; `.connected` tracks liveness."""
    def __init__(self, *, connected, inner):
        self.connected = connected
        self.redis_client = inner


def test_resolve_none():
    assert resolve_async_redis(None) is None


def test_resolve_raw_client_passthrough():
    raw = _Raw()
    assert resolve_async_redis(raw) is raw          # already pipeline-capable


def test_resolve_wrapper_connected_returns_inner():
    inner = _Raw()
    assert resolve_async_redis(_Wrapper(connected=True, inner=inner)) is inner


def test_resolve_wrapper_disconnected_returns_none():
    assert resolve_async_redis(_Wrapper(connected=False, inner=_Raw())) is None


def test_resolve_wrapper_connected_but_inner_none_returns_none():
    assert resolve_async_redis(_Wrapper(connected=True, inner=None)) is None


# ── api_hardening RateLimiter: the live-path limiter that failed open ─────────

def _limiter(limit=3):
    from prsm.node.api_hardening import RateLimitConfig, RateLimiter
    cfg = RateLimitConfig()
    cfg.enabled = True
    cfg.requests_per_minute = limit
    return RateLimiter(cfg)


class _Pipe:
    """Async redis pipeline: command methods queue (sync, return self); execute()
    is awaited. zcard is the 2nd queued op, so execute() returns [_, count]."""
    def __init__(self, count_seq, raise_on_execute=False):
        self._counts = list(count_seq)
        self._raise = raise_on_execute

    def zremrangebyscore(self, *a, **k): return self
    def zcard(self, *a, **k): return self
    def zadd(self, *a, **k): return self
    def expire(self, *a, **k): return self

    async def execute(self):
        if self._raise:
            raise ConnectionError("redis blip")
        return [None, self._counts.pop(0)]


class _Inner:
    def __init__(self, pipe):
        self._pipe = pipe

    def pipeline(self):
        return self._pipe


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def test_real_wrapper_shape_no_longer_fails_open_uses_inner_and_blocks():
    # THE regression: a RedisClient-shaped wrapper (connected, inner with a working
    # pipeline, NO top-level .pipeline). Pre-fix: wrapper.pipeline() → AttributeError
    # → except → allow-all. Post-fix: resolve to inner, enforce the limit.
    rl = _limiter(limit=3)
    # zcard reports 5 prior requests in the window, well over the limit of 3. A
    # fail-open path would ignore the count and allow; the fix reads the inner
    # pipeline's count and blocks.
    rl.redis_client = _Wrapper(connected=True, inner=_Inner(_Pipe([5])))
    res = _run(rl.check_rate_limit("clientA", "default"))
    assert res.allowed is False        # blocked — NOT failed open


def test_wrapper_under_limit_allowed_via_inner():
    rl = _limiter(limit=5)
    rl.redis_client = _Wrapper(connected=True, inner=_Inner(_Pipe([2])))  # under limit
    res = _run(rl.check_rate_limit("clientB", "default"))
    assert res.allowed is True


def test_disconnected_wrapper_degrades_to_local_and_enforces():
    # connected=False → resolve returns None → local limiter path → still enforces.
    rl = _limiter(limit=2)
    rl.redis_client = _Wrapper(connected=False, inner=None)
    allowed = [_run(rl.check_rate_limit("clientC", "default")).allowed for _ in range(4)]
    assert allowed[:2] == [True, True]
    assert allowed[2] is False          # local limiter blocks past the limit (no fail-open)


def test_redis_error_degrades_to_local_not_fail_open():
    # inner pipeline raises mid-op → must fall back to the LOCAL limiter (enforced),
    # NOT fail open (allow-all). Pre-fix the except returned allowed=True.
    rl = _limiter(limit=1)
    rl.redis_client = _Wrapper(connected=True, inner=_Inner(_Pipe([], raise_on_execute=True)))
    first = _run(rl.check_rate_limit("clientD", "default"))
    second = _run(rl.check_rate_limit("clientD", "default"))
    assert first.allowed is True        # 1st under local limit
    assert second.allowed is False      # 2nd blocked by local fallback — not fail-open


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
