"""Sprint 1187 — fix the rate-limit middleware init race (day-one-live blocker #8).

APIHardening.apply_middleware() only registers RateLimitMiddleware `if self.rate_limiter`,
but self.rate_limiter was constructed inside the ASYNC initialize() (create_task'd, not
awaited) before apply_middleware() ran when the event loop was already running — so in the
async-server path rate limiting (DDoS protection on money/compute endpoints) was SILENTLY
never installed. Fix: construct the components SYNCHRONOUSLY (idempotent) so
apply_middleware always registers the middleware; initialize() keeps doing the async
warm-up (redis/JWT) on the already-constructed instances.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI

from prsm.node.api_hardening import (
    APIHardening,
    APISecurityConfig,
    RateLimitMiddleware,
)


def _hardening(**cfg):
    base = dict(enable_rate_limiting=True, enable_jwt_auth=False, enable_websocket=False)
    base.update(cfg)
    return APIHardening(FastAPI(), APISecurityConfig(**base))


def _middleware_classes(app):
    return [m.cls for m in app.user_middleware]


def test_apply_middleware_registers_ratelimit_without_initialize_first():
    """The race: apply_middleware() runs BEFORE the async initialize() completes (loop
    already running → initialize is create_task'd). Rate-limit middleware must STILL be
    registered."""
    h = _hardening()
    assert h.rate_limiter is None          # initialize() has NOT run yet
    h.apply_middleware()                   # must construct + register regardless
    assert h.rate_limiter is not None
    assert RateLimitMiddleware in _middleware_classes(h.app)


def test_rate_limiting_disabled_registers_nothing():
    h = _hardening(enable_rate_limiting=False)
    h.apply_middleware()
    assert RateLimitMiddleware not in _middleware_classes(h.app)
    assert h.rate_limiter is None


def test_initialize_then_apply_is_idempotent_single_instance():
    """When initialize() DOES run first (loop not running path), apply_middleware must not
    re-construct a second rate_limiter — it reuses the warmed-up instance."""
    h = _hardening()
    asyncio.run(h.initialize())
    rl = h.rate_limiter
    assert rl is not None
    h.apply_middleware()
    assert h.rate_limiter is rl             # same instance, not re-constructed
    assert RateLimitMiddleware in _middleware_classes(h.app)


def test_initialize_still_warms_up_after_sync_construct():
    """apply_middleware() first (constructs), then initialize() warms up the SAME limiter
    (redis backend attr set) without replacing it."""
    h = _hardening()
    h.apply_middleware()
    rl = h.rate_limiter
    asyncio.run(h.initialize())
    assert h.rate_limiter is rl             # warm-up didn't replace the instance
    assert hasattr(rl, "redis_client")      # async initialize() ran on it


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
