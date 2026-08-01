"""Sprint 1499 — quote validation was one-sided: a ceiling with no floor.

`request_quote` checked only `quoted_price > listing.price_per_shard_ftns`. Three
distinct bad values slipped through, because every comparison against them is
False:

    nan > 0.05  -> False     (verified)
    -1  > 0.05  -> False     (verified)
    0   > 0.05  -> False

Each fails differently:
  NaN       poisons downstream balances. DAGLedger's sp1468 gate catches it, but
            only AFTER the orchestrator has committed to this provider.
  negative  is a reverse transfer by intent.
  zero      is the interesting one: dag_ledger.py explicitly says "amount == 0 is
            a harmless no-op — left to outer endpoints", and this IS the outer
            endpoint that never checked. The provider performs real work against
            an escrow holding nothing.

`min_price_per_shard_ftns` already existed and was enforced on LISTINGS by
EligibilityFilter — but never on the QUOTE, which is the number actually escrowed.
The same asymmetry sp1498 closed on the ceiling side.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.marketplace.price_handshake import (
    PriceNegotiator,
    PriceQuoteRejected,
)


def _listing(price=0.05):
    l = MagicMock()
    l.provider_id = "d437aa67d99cff4a6a17179f5c731b77"
    l.listing_id = "listing-1"
    l.price_per_shard_ftns = price
    return l


def _negotiator(quoted):
    """A negotiator whose provider answers with `quoted`."""
    n = object.__new__(PriceNegotiator)
    n.identity = MagicMock(node_id="requester")
    n.transport = MagicMock()
    n.transport.send_to_peer = AsyncMock()
    n.default_timeout = 5.0
    n._pending = {}

    async def _send(peer, msg):
        rid = msg.payload["request_id"]
        fut = n._pending.get(rid)
        if fut and not fut.done():
            fut.set_result({
                "subtype": "shard_price_quote_ack",
                "quoted_price_ftns": quoted,
            })
    n.transport.send_to_peer = _send
    return n


def _quote(quoted, floor=0.0, listing_price=0.05):
    n = _negotiator(quoted)
    return asyncio.run(n.request_quote(
        listing=_listing(listing_price), shard_index=0, shard_size_bytes=1024,
        max_acceptable_price_ftns=1.0, min_acceptable_price_ftns=floor))


# ── the three values that slipped through ───────────────────────────

def test_the_comparison_really_does_let_them_through():
    """★ Why a ceiling alone is not validation — the premise, demonstrated."""
    assert (float("nan") > 0.05) is False
    assert (-1.0 > 0.05) is False
    assert (0.0 > 0.05) is False


def test_a_ZERO_quote_is_rejected():
    """★ THE fix. dag_ledger treats amount==0 as a no-op 'left to outer
    endpoints' — this is that endpoint. Otherwise the provider works for free
    against an escrow holding nothing."""
    r = _quote(0.0)
    assert isinstance(r, PriceQuoteRejected)
    assert r.reason == "quote_not_positive"


def test_a_NEGATIVE_quote_is_rejected():
    r = _quote(-5.0)
    assert isinstance(r, PriceQuoteRejected)
    assert r.reason == "quote_not_positive"


def test_a_NAN_quote_is_rejected():
    """★ Caught here rather than deep in the ledger, so the orchestrator never
    commits to this provider."""
    r = _quote(float("nan"))
    assert isinstance(r, PriceQuoteRejected)
    assert r.reason == "quote_not_finite"


def test_an_INFINITE_quote_is_rejected():
    r = _quote(float("inf"))
    assert isinstance(r, PriceQuoteRejected)
    assert r.reason == "quote_not_finite"


# ── the requester's floor ───────────────────────────────────────────

def test_a_quote_below_the_requesters_floor_is_rejected():
    """min_price_per_shard_ftns was enforced on listings but never on the quote —
    the number actually escrowed."""
    r = _quote(0.001, floor=0.01)
    assert isinstance(r, PriceQuoteRejected)
    assert r.reason == "quote_below_floor"


PRICE_REJECTIONS = {"quote_not_positive", "quote_not_finite",
                    "quote_below_floor", "quote_exceeds_listing"}


def test_a_quote_exactly_at_the_floor_passes_the_PRICE_gates():
    """The floor is inclusive — a provider matching it exactly is honest.

    (The handshake still rejects afterwards on the sp972 signature check, since
    this fixture sends an unsigned quote; what matters here is that it is not
    rejected on PRICE grounds.)"""
    r = _quote(0.01, floor=0.01)
    assert getattr(r, "reason", None) not in PRICE_REJECTIONS


def test_the_ceiling_check_still_works():
    """The pre-existing upper bound must survive."""
    r = _quote(0.09, listing_price=0.05)
    assert isinstance(r, PriceQuoteRejected)
    assert r.reason == "quote_exceeds_listing"


def test_a_normal_quote_passes_the_PRICE_gates():
    """A well-formed price in range must not be rejected on price grounds."""
    r = _quote(0.04, floor=0.01, listing_price=0.05)
    assert getattr(r, "reason", None) not in PRICE_REJECTIONS


# ── the orchestrator supplies the floor ─────────────────────────────

def test_the_orchestrator_passes_the_policy_floor():
    """★ Binding test — a floor parameter nothing supplies defaults to 0.0 and
    leaves the zero-quote path open."""
    import inspect

    from prsm.marketplace.orchestrator import MarketplaceOrchestrator

    src = inspect.getsource(MarketplaceOrchestrator._dispatch_one_shard)
    assert "min_acceptable_price_ftns=policy.min_price_per_shard_ftns" in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
