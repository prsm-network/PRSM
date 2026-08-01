"""Phase 3 Task 4: DispatchPolicy.

Requester-side policy that drives EligibilityFilter. All fields have
safe defaults so a caller who supplies `DispatchPolicy()` gets a
permissive-but-sane filter: any non-expired listing passes except the
anti-loss-leader price floor.

Defaults rationale (docs/2026-04-20-phase3-marketplace-design.md §8.4):
  - min_price_per_shard_ftns = 0.01 FTNS — a provider advertising less
    than this is either running a loss-leader attack (attract requests,
    degrade service) or misconfigured. The default rejects both.
  - min_reputation_score = 0.0 — allows new providers in (their score
    is 0.5 neutral per ReputationTracker); raises to 0.5 to gate on
    "proven decent" once you have history.
  - required_dtype = "float64" — matches Phase 2's executor contract.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _default_max_price() -> float:
    """sp1498 — the per-shard sanity ceiling, overridable by the operator.

    Env ``PRSM_MARKETPLACE_MAX_PRICE_FTNS``. A non-finite or non-positive value is
    REFUSED rather than honoured, because "inf" is exactly the value that made the
    price filter dead code in the first place — accepting it from the environment
    would reopen the hole through a different door.
    """
    raw = (os.environ.get("PRSM_MARKETPLACE_MAX_PRICE_FTNS") or "").strip()
    if not raw:
        return 1.0
    try:
        v = float(raw)
    except ValueError:
        return 1.0
    if v != v or v in (float("inf"), float("-inf")) or v <= 0:
        return 1.0
    return v


#: Per-shard price ceiling applied when a caller does not set one. NOT a market
#: price — a bound that stops an unconfigured requester being drained.
DEFAULT_MAX_PRICE_PER_SHARD_FTNS: float = _default_max_price()


@dataclass(frozen=True)
class DispatchPolicy:
    """Policy for marketplace provider selection.

    Consumed by EligibilityFilter.filter() to narrow a directory's
    listing set down to a policy-compliant subset.
    """
    # sp1498 — a FINITE default. This was float("inf"), which made the price check
    # in EligibilityFilter.filter dead code: `listing.price > inf` is always False,
    # so no listing was ever excluded on price. Listing validation only rejects
    # NEGATIVE prices, and selection walks the eligible list in gossip-arrival
    # order — so a provider advertising 1e9 FTNS/shard early was selected first,
    # quoted its own listing price (the quote ceiling compares against the
    # listing, not the requester's policy), and got escrowed and paid. The spend
    # was bounded by the requester's balance, not by anything they intended.
    #
    # DEFAULT_MAX_PRICE_PER_SHARD_FTNS is a sanity ceiling, not a market price:
    # it exists so that an unconfigured requester cannot be drained. Callers who
    # genuinely want to pay more must say so explicitly.
    max_price_per_shard_ftns: float = DEFAULT_MAX_PRICE_PER_SHARD_FTNS
    #: sp1498 — total FTNS this job may spend across ALL shards. 0 disables the
    #: check. A per-shard cap does not bound a job, because the shard COUNT comes
    #: from the model rather than from the requester: N shards at the cap costs N
    #: times the cap. Enforced against agreed quotes, before escrow.
    max_total_job_ftns: float = 0.0
    min_price_per_shard_ftns: float = 0.01
    require_tee: bool = False
    min_stake_tier: str = "open"
    min_reputation_score: float = 0.0
    required_dtype: str = "float64"
    min_capacity_shards_per_sec: float = 0.0
    max_timeout_seconds: float = 30.0
    require_unique_providers: bool = False
    # Phase 7.1 Task 5: redundant-execution verification (Tier B). When
    # consensus_mode is None (the default), dispatch is single-provider
    # as in Phase 7. When set, the orchestrator routes the shard to k
    # providers in parallel and decides agreement by output-hash voting.
    #   - "majority": winning group >= (k // 2) + 1 (default)
    #   - "unanimous": all k must respond AND all must agree
    #   - "byzantine": reserved — raises NotImplementedError until 7.1x
    consensus_mode: Optional[str] = None
    consensus_k: int = 3
    # sp906 — staking priority access. A requester's staking lock confers
    # a priority boost (>= 0) that biases provider selection toward
    # higher-capacity (faster) providers. 0.0 = no boost (default;
    # behavior identical to pre-sp906). Populated by the caller from the
    # requester's StakingBenefits.priority_boost.
    requester_priority_boost: float = 0.0
