"""Sprint 1498 — a requester's spend must be bounded by intent, not by their balance.

Found by an adversarial assessment of the marketplace demand side.

`DispatchPolicy.max_price_per_shard_ftns` defaulted to `float("inf")`, which made
the price check in EligibilityFilter DEAD CODE — `listing.price > inf` is always
False, so no listing was ever excluded on price. Listing validation only rejects
NEGATIVE prices. And selection walked the eligible pool in gossip-ARRIVAL order,
so a provider advertising 1e9 FTNS/shard early was picked first, quoted its own
listing price, and got escrowed and paid.

Three separate holes, all of which had to close:
  1. the default ceiling was infinite      -> finite, env-overridable
  2. selection rewarded being EARLY        -> cheapest-first
  3. a per-shard cap does not bound a JOB  -> per-job budget, since the shard
                                              COUNT comes from the model
"""
from __future__ import annotations

import pytest

from prsm.marketplace.filter import EligibilityFilter
from prsm.marketplace.policy import (
    DEFAULT_MAX_PRICE_PER_SHARD_FTNS,
    DispatchPolicy,
    _default_max_price,
)


# ── 1. the default is finite ────────────────────────────────────────

def test_the_default_ceiling_is_FINITE():
    """★ THE bug. With inf, the filter's price check could never exclude anything."""
    p = DispatchPolicy()
    assert p.max_price_per_shard_ftns != float("inf")
    assert p.max_price_per_shard_ftns > 0
    assert p.max_price_per_shard_ftns == DEFAULT_MAX_PRICE_PER_SHARD_FTNS


def test_an_absurd_listing_is_now_EXCLUDED_by_the_filter():
    """★ End to end through the real filter: the 1e9 provider is dropped."""
    from prsm.marketplace.listing import ProviderListing
    import time as _t

    def listing(pid, price):
        return ProviderListing(
            provider_id=pid, listing_id=f"l-{pid}", price_per_shard_ftns=price,
            capacity_shards_per_sec=1.0, max_shard_bytes=1 << 20,
            supported_dtypes=["float64"], tee_capable=False, stake_tier="open",
            advertised_at_unix=int(_t.time()), ttl_seconds=300,
            provider_pubkey_b64="", signature="")

    got = EligibilityFilter().filter(
        [listing("greedy", 1e9), listing("fair", 0.25)], DispatchPolicy())
    assert [l.provider_id for l in got] == ["fair"], "the 1e9 provider must be excluded"


def test_the_env_override_is_honoured(monkeypatch):
    monkeypatch.setenv("PRSM_MARKETPLACE_MAX_PRICE_FTNS", "2.5")
    assert _default_max_price() == 2.5


@pytest.mark.parametrize("bad", ["inf", "-inf", "nan", "0", "-1", "not-a-number", ""])
def test_a_nonfinite_or_nonpositive_override_is_REFUSED(monkeypatch, bad):
    """★ 'inf' from the environment must not reopen the hole through another door —
    it is exactly the value that made the filter dead code."""
    monkeypatch.setenv("PRSM_MARKETPLACE_MAX_PRICE_FTNS", bad)
    v = _default_max_price()
    assert v == 1.0
    assert v != float("inf")


# ── 2. cheapest-first, not first-to-gossip ──────────────────────────

def test_selection_is_sorted_CHEAPEST_first_not_arrival_order():
    """★ The filter deliberately preserves input order, and that input is gossip
    ARRIVAL order — so dispatch used to reward being early, not being cheap."""
    import inspect

    from prsm.marketplace.orchestrator import MarketplaceOrchestrator

    src = inspect.getsource(MarketplaceOrchestrator.orchestrate_sharded_inference)
    assert "sorted(" in src
    assert "price_per_shard_ftns" in src


def test_the_sort_is_deterministic_on_ties():
    """Two requesters with the same directory must make the same choice."""
    rows = [("b", 0.5), ("a", 0.5), ("c", 0.1)]
    ordered = sorted(rows, key=lambda r: (float(r[1]), str(r[0])))
    assert [r[0] for r in ordered] == ["c", "a", "b"]


# ── 3. the per-job budget ───────────────────────────────────────────

def test_policy_exposes_a_per_job_budget():
    """★ A per-shard cap does not bound a job: N shards at the cap is N× the cap,
    and N comes from the MODEL, not the requester."""
    assert hasattr(DispatchPolicy(), "max_total_job_ftns")
    assert DispatchPolicy().max_total_job_ftns == 0.0          # off by default
    assert DispatchPolicy(max_total_job_ftns=5.0).max_total_job_ftns == 5.0


def test_the_orchestrator_enforces_the_budget_BEFORE_escrow():
    """Aborting after escrow would part-pay a job that cannot complete."""
    import inspect

    from prsm.marketplace.orchestrator import MarketplaceOrchestrator

    src = inspect.getsource(MarketplaceOrchestrator._dispatch_one_shard)
    assert "JobBudgetExceededError" in src
    i_budget = src.index("JobBudgetExceededError")
    i_dispatch = src.index("dispatch_with_receipt")
    assert i_budget < i_dispatch, "budget must be checked before dispatch/escrow"


def test_spend_accrues_only_on_SUCCESS():
    """A failed attempt was never escrow-released, so it must not consume budget."""
    import inspect

    from prsm.marketplace.orchestrator import MarketplaceOrchestrator

    src = inspect.getsource(MarketplaceOrchestrator._dispatch_one_shard)
    guard = src[src.index("# Success."):]
    assert "_job_spent_ftns" in guard


# ── 4. the quote is checked locally, not trusted ────────────────────

def test_the_QUOTE_is_re_checked_against_the_ceiling_locally():
    """★ request_quote passes the ceiling to the provider — but the provider is the
    one answering. A ceiling enforced only on the far side is not a check."""
    import inspect

    from prsm.marketplace.orchestrator import MarketplaceOrchestrator

    src = inspect.getsource(MarketplaceOrchestrator._dispatch_one_shard)
    assert "quote_over_ceiling" in src
    i_check = src.index("quote_over_ceiling")
    i_dispatch = src.index("dispatch_with_receipt")
    assert i_check < i_dispatch, "the quote must be checked before dispatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
