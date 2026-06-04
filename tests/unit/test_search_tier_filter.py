"""Sprint 289 — search filtering by creator tier.

Vision §14 mitigation item (4): "Users can filter search
results by creator reputation tier."

Adds two query params to /content/search:
  min_tier:    "low" | "medium" | "high"  (filter to ≥ tier)
  exclude_new: bool                       (hide cold-start)

Default: no filtering — everything shows including TIER_NEW.
Power users opt in to min_tier=medium for proven creators
only.

Tier rank for comparison:
  new    = -1  (cold-start, no signal)
  low    =  1
  medium =  2
  high   =  3

`min_tier=low` ≥ rank 1 → excludes NEW (rank -1)
`min_tier=medium` ≥ rank 2 → excludes NEW + LOW
`min_tier=high` ≥ rank 3 → only HIGH

Each result row now carries `creator_tier` so callers see
the tier even when not filtering.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from prsm.marketplace.creator_reputation import (
    CreatorReputationTracker,
    TIER_NEW, TIER_LOW, TIER_MEDIUM, TIER_HIGH,
)
from prsm.node.api import create_api_app


def _seed_tier(t, creator, target_tier):
    """Seed the tracker so `creator` lands in `target_tier`."""
    if target_tier == TIER_NEW:
        # Under MIN_SAMPLES_FOR_SCORE
        for i in range(3):
            t.record_access(
                creator_id=creator, purchaser_id=f"p{i}",
                content_id=f"c{i}",
            )
    elif target_tier == TIER_LOW:
        # 15 distinct, no repeats → score ≈ 0.36 → LOW
        for i in range(15):
            t.record_access(
                creator_id=creator, purchaser_id=f"p{i}",
                content_id=f"c{i}",
            )
    elif target_tier == TIER_MEDIUM:
        # 100 distinct, no repeats → score 0.6 → exactly at
        # MEDIUM threshold
        for i in range(100):
            t.record_access(
                creator_id=creator, purchaser_id=f"p{i}",
                content_id=f"c{i}",
            )
    elif target_tier == TIER_HIGH:
        # 50 distinct, all repeat → high score
        for i in range(50):
            t.record_access(
                creator_id=creator, purchaser_id=f"p{i}",
                content_id="c0",
            )
            t.record_access(
                creator_id=creator, purchaser_id=f"p{i}",
                content_id="c1",
            )


class _FakeIndex:
    """Returns canned search results. Each record has
    creator_id matching one of the tiered creators."""

    def __init__(self, creators_by_cid):
        self.creators_by_cid = creators_by_cid

    def search(self, q, limit=20):
        results = []
        for cid, creator in self.creators_by_cid.items():
            r = MagicMock()
            r.cid = cid
            r.filename = f"{cid}.bin"
            r.size_bytes = 1024
            r.content_hash = "sha256-deadbeef"
            r.creator_id = creator
            # sp995 (decision A) — the search tier-read keys on the creator ETH
            # address (the identity reputation is accumulated + stake-gated
            # under), so the record must carry it. This test seeds reputation
            # under `creator`, so the record's eth-address identity is `creator`.
            r.creator_eth_address = creator
            r.providers = []
            r.created_at = 0
            r.metadata = {}
            r.royalty_rate = 0.1
            r.parent_cids = []
            results.append(r)
        return results[:limit]


def _client(tracker=None, results=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._creator_reputation_tracker = tracker
    node.content_index = _FakeIndex(results or {})
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


# ── Default: no filtering shows all ──────────────────────


def test_default_search_returns_all_tiers():
    t = CreatorReputationTracker()
    _seed_tier(t, "creator_new", TIER_NEW)
    _seed_tier(t, "creator_low", TIER_LOW)
    _seed_tier(t, "creator_high", TIER_HIGH)
    results = {
        "cid_n": "creator_new",
        "cid_l": "creator_low",
        "cid_h": "creator_high",
    }
    resp = _client(t, results).get("/content/search?q=foo")
    body = resp.json()
    assert body["count"] == 3
    cids = sorted(r["cid"] for r in body["results"])
    assert cids == ["cid_h", "cid_l", "cid_n"]


def test_results_now_carry_creator_tier():
    """Each row gets creator_tier so callers see the tier
    even without filtering."""
    t = CreatorReputationTracker()
    _seed_tier(t, "creator_high", TIER_HIGH)
    resp = _client(t, {"cid_h": "creator_high"}).get(
        "/content/search?q=foo",
    )
    body = resp.json()
    assert body["results"][0]["creator_tier"] == TIER_HIGH


def test_results_carry_tier_when_tracker_unwired():
    """No tracker → tier defaults to TIER_NEW (consistent
    with the tracker's own cold-start semantics)."""
    resp = _client(None, {"cid_x": "creator_x"}).get(
        "/content/search?q=foo",
    )
    body = resp.json()
    assert body["results"][0]["creator_tier"] == TIER_NEW


def test_search_tier_reads_eth_address_not_node_id():
    """sp995 (fix C) regression — reputation is RECORDED and stake-gated under the
    creator ETH address, so the search tier-read must use it, NOT the node_id
    (creator_id). A record whose creator_id (node_id) differs from its
    creator_eth_address must still surface the HIGH tier earned under the eth
    address. Under the old `tracker.tier_for(r.creator_id)` read this returned
    TIER_NEW → the creator was wrongly excluded by min_tier=high."""
    eth = "0x" + "a" * 40
    t = CreatorReputationTracker()
    _seed_tier(t, eth, TIER_HIGH)  # reputation earned under the ETH address

    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._creator_reputation_tracker = t
    node._creator_stake_client = None  # uncommissioned → gate passes through
    rec = MagicMock()
    rec.cid = "cid_h"
    rec.filename = "f.bin"
    rec.size_bytes = 1
    rec.content_hash = "sha"
    rec.creator_id = "node-id-distinct-from-eth"  # node_id != eth address
    rec.creator_eth_address = eth
    rec.providers = []
    rec.created_at = 0
    rec.metadata = {}
    rec.royalty_rate = 0.1
    rec.parent_cids = []
    idx = MagicMock()
    idx.search = lambda q, limit=20: [rec]
    node.content_index = idx
    client = TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )

    resp = client.get("/content/search?q=foo&min_tier=high")
    body = resp.json()
    assert [r["cid"] for r in body["results"]] == ["cid_h"]
    assert body["results"][0]["creator_tier"] == TIER_HIGH


# ── min_tier filter ──────────────────────────────────────


def test_min_tier_low_excludes_new():
    t = CreatorReputationTracker()
    _seed_tier(t, "creator_new", TIER_NEW)
    _seed_tier(t, "creator_low", TIER_LOW)
    _seed_tier(t, "creator_high", TIER_HIGH)
    results = {
        "cid_n": "creator_new",
        "cid_l": "creator_low",
        "cid_h": "creator_high",
    }
    resp = _client(t, results).get(
        "/content/search?q=foo&min_tier=low",
    )
    body = resp.json()
    cids = sorted(r["cid"] for r in body["results"])
    # NEW excluded
    assert "cid_n" not in cids
    assert cids == ["cid_h", "cid_l"]


def test_min_tier_medium_excludes_low_and_new():
    t = CreatorReputationTracker()
    _seed_tier(t, "creator_new", TIER_NEW)
    _seed_tier(t, "creator_low", TIER_LOW)
    _seed_tier(t, "creator_medium", TIER_MEDIUM)
    _seed_tier(t, "creator_high", TIER_HIGH)
    results = {
        "cid_n": "creator_new",
        "cid_l": "creator_low",
        "cid_m": "creator_medium",
        "cid_h": "creator_high",
    }
    resp = _client(t, results).get(
        "/content/search?q=foo&min_tier=medium",
    )
    body = resp.json()
    cids = sorted(r["cid"] for r in body["results"])
    assert cids == ["cid_h", "cid_m"]


def test_min_tier_high_only_high():
    t = CreatorReputationTracker()
    _seed_tier(t, "creator_low", TIER_LOW)
    _seed_tier(t, "creator_medium", TIER_MEDIUM)
    _seed_tier(t, "creator_high", TIER_HIGH)
    results = {
        "cid_l": "creator_low",
        "cid_m": "creator_medium",
        "cid_h": "creator_high",
    }
    resp = _client(t, results).get(
        "/content/search?q=foo&min_tier=high",
    )
    body = resp.json()
    cids = sorted(r["cid"] for r in body["results"])
    assert cids == ["cid_h"]


def test_min_tier_invalid_value_422():
    resp = _client().get("/content/search?q=foo&min_tier=bogus")
    assert resp.status_code == 422


# ── exclude_new flag ─────────────────────────────────────


def test_exclude_new_drops_cold_start_creators():
    t = CreatorReputationTracker()
    _seed_tier(t, "creator_new", TIER_NEW)
    _seed_tier(t, "creator_low", TIER_LOW)
    results = {
        "cid_n": "creator_new",
        "cid_l": "creator_low",
    }
    resp = _client(t, results).get(
        "/content/search?q=foo&exclude_new=true",
    )
    body = resp.json()
    cids = [r["cid"] for r in body["results"]]
    assert cids == ["cid_l"]


def test_exclude_new_false_includes_new():
    t = CreatorReputationTracker()
    _seed_tier(t, "creator_new", TIER_NEW)
    results = {"cid_n": "creator_new"}
    resp = _client(t, results).get(
        "/content/search?q=foo&exclude_new=false",
    )
    body = resp.json()
    assert len(body["results"]) == 1


# ── Count reflects filtered results ──────────────────────


def test_count_reflects_post_filter():
    t = CreatorReputationTracker()
    _seed_tier(t, "creator_new", TIER_NEW)
    _seed_tier(t, "creator_high", TIER_HIGH)
    results = {
        "cid_n": "creator_new",
        "cid_h": "creator_high",
    }
    resp = _client(t, results).get(
        "/content/search?q=foo&min_tier=medium",
    )
    body = resp.json()
    assert body["count"] == 1


# ── Backwards compat: existing fields still present ──────


def test_existing_fields_preserved():
    t = CreatorReputationTracker()
    _seed_tier(t, "creator_high", TIER_HIGH)
    resp = _client(t, {"cid_h": "creator_high"}).get(
        "/content/search?q=foo",
    )
    r0 = resp.json()["results"][0]
    # Spot-check the pre-sprint-289 schema is preserved
    for k in [
        "cid", "filename", "size_bytes", "content_hash",
        "creator_id", "providers", "created_at", "metadata",
        "royalty_rate", "parent_cids",
    ]:
        assert k in r0
