"""Sprint 926 — validate advertised capacity in verify_listing (compute review #5).

The compute/inference review found a provider can advertise an arbitrary
``capacity_shards_per_sec`` (self-reported, never validated). verify_listing
checked ttl/price/dtypes/signature but NOT capacity, and _select_top_k boosts a
requester-priority-boosted ranking by capacity relative to the fastest in the
set — so a provider claiming an absurd capacity gets the full boost. (The boost
is bounded — priority_boost ≤ 0.5 and cap_norm ≤ 1 → ≤1.5× advantage, only on
boosted requests — so this is a bounded bias, not monopolization; the residual
"claim-max-then-under-deliver" needs latency-into-reputation, a scoring-policy
change queued as a follow-on.)

Fix: verify_listing rejects a negative capacity (nonsensical + would make the
selection cap_norm negative) or an implausibly large one
(> PRSM_MAX_LISTING_CAPACITY_SHARDS_PER_SEC), bounding the claim at ingestion;
_select_top_k additionally clamps capacity to >= 0 defensively.
"""
from __future__ import annotations

import pytest

from prsm.marketplace.listing import sign_listing, verify_listing
from prsm.node.identity import generate_node_identity


def _make(identity, **overrides):
    kw = dict(
        capacity_shards_per_sec=10.0,
        max_shard_bytes=10 * 1024 * 1024,
        supported_dtypes=["float64"],
        price_per_shard_ftns=0.05,
        tee_capable=False,
        stake_tier="standard",
        ttl_seconds=300,
    )
    kw.update(overrides)
    return sign_listing(identity=identity, **kw)


def test_negative_capacity_rejected():
    led = generate_node_identity()
    assert verify_listing(_make(led, capacity_shards_per_sec=-1.0)) is False


def test_absurd_capacity_rejected(monkeypatch):
    monkeypatch.setenv("PRSM_MAX_LISTING_CAPACITY_SHARDS_PER_SEC", "1000")
    led = generate_node_identity()
    assert verify_listing(_make(led, capacity_shards_per_sec=1_000_000.0)) is False


def test_normal_capacity_accepted():
    led = generate_node_identity()
    assert verify_listing(_make(led, capacity_shards_per_sec=50.0)) is True


def test_capacity_at_cap_accepted(monkeypatch):
    monkeypatch.setenv("PRSM_MAX_LISTING_CAPACITY_SHARDS_PER_SEC", "1000")
    led = generate_node_identity()
    assert verify_listing(_make(led, capacity_shards_per_sec=1000.0)) is True


def test_select_top_k_clamps_negative_capacity():
    # Defense-in-depth: even if a negative-capacity listing reached selection,
    # the cap_norm must not go negative (which would scramble the score).
    from prsm.marketplace.orchestrator import MarketplaceOrchestrator

    a = generate_node_identity()
    b = generate_node_identity()
    good = _make(a, capacity_shards_per_sec=10.0)
    neg = _make(b, capacity_shards_per_sec=-5.0)
    # Should not raise + should return a deterministic ordering.
    out = MarketplaceOrchestrator._select_top_k([good, neg], k=2, priority_boost=0.5)
    assert len(out) == 2
