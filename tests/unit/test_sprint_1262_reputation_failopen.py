"""Sprint 1262 — IP-reputation lookup must not grant MAX trust when Redis is unavailable.

_get_ip_reputation returned 1.0 ("good reputation") both when the Redis backend was absent
(_live_redis() is None) and on ANY error. Reputation gates a block (score <
ip_reputation_threshold → blocked), so returning the MAXIMUM on a backend outage is a
fail-OPEN: a Redis blip silently elevates every IP to perfect trust and bypasses the
low-reputation block. (LOW severity today — the reputation block is one layer and the
overall limiter already degrades to a local limiter on Redis error per sp1247 — but the
semantics are wrong and dangerous if reputation is ever wired as a continuous multiplier.)

Fix: on backend-unavailable/error, return a NEUTRAL "unknown" reputation (the new-IP
default, not the maximum) so an outage can't elevate trust — while not self-DoSing every IP.
Strict operators can opt into hard fail-closed (block) via
PRSM_RATE_LIMIT_REPUTATION_FAIL_CLOSED.
"""
from __future__ import annotations

import pytest

from prsm.core.auth.rate_limiter import (
    RateLimiter,
    _NEUTRAL_REPUTATION,
    _reputation_fail_closed,
)

STRICT = "PRSM_RATE_LIMIT_REPUTATION_FAIL_CLOSED"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(STRICT, raising=False)


class _RaisingRedis:
    async def get(self, *a, **k):
        raise RuntimeError("redis down")


class _ScoreRedis:
    def __init__(self, score):
        self._score = score

    async def get(self, *a, **k):
        return self._score

    async def setex(self, *a, **k):
        return True


def _rl(monkeypatch, redis):
    rl = RateLimiter()
    monkeypatch.setattr(rl, "_live_redis", lambda: redis)
    return rl


# ── the neutral constant is sane ────────────────────────────────────────────────

def test_neutral_reputation_is_not_max_and_not_blocking():
    rl = RateLimiter()
    assert _NEUTRAL_REPUTATION < 1.0                       # not MAX trust
    assert _NEUTRAL_REPUTATION >= rl.ip_reputation_threshold   # but not self-blocking


# ── default (graceful) behavior: neutral, never 1.0 ──────────────────────────────

@pytest.mark.asyncio
async def test_redis_none_returns_neutral_not_max(monkeypatch):
    rl = _rl(monkeypatch, None)              # _live_redis() → None (backend absent)
    score = await rl._get_ip_reputation("1.2.3.4")
    assert score == _NEUTRAL_REPUTATION
    assert score != 1.0


@pytest.mark.asyncio
async def test_redis_error_returns_neutral_not_max(monkeypatch):
    rl = _rl(monkeypatch, _RaisingRedis())
    score = await rl._get_ip_reputation("1.2.3.4")
    assert score == _NEUTRAL_REPUTATION
    assert score != 1.0


# ── strict opt-in: hard fail-closed (block) ──────────────────────────────────────

@pytest.mark.asyncio
async def test_strict_mode_redis_none_fails_closed(monkeypatch):
    monkeypatch.setenv(STRICT, "1")
    assert _reputation_fail_closed() is True
    rl = _rl(monkeypatch, None)
    score = await rl._get_ip_reputation("1.2.3.4")
    assert score < rl.ip_reputation_threshold            # < 0.3 → blocked


@pytest.mark.asyncio
async def test_strict_mode_error_fails_closed(monkeypatch):
    monkeypatch.setenv(STRICT, "1")
    rl = _rl(monkeypatch, _RaisingRedis())
    score = await rl._get_ip_reputation("1.2.3.4")
    assert score < rl.ip_reputation_threshold


# ── the happy path is unchanged ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_existing_score_returned_unchanged(monkeypatch):
    rl = _rl(monkeypatch, _ScoreRedis("0.5"))
    assert await rl._get_ip_reputation("1.2.3.4") == 0.5


@pytest.mark.asyncio
async def test_new_ip_gets_neutral_default(monkeypatch):
    rl = _rl(monkeypatch, _ScoreRedis(None))   # no stored score → new IP
    assert await rl._get_ip_reputation("1.2.3.4") == _NEUTRAL_REPUTATION


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
