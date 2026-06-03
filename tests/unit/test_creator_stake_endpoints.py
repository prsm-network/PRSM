"""Sprint 290 — creator stake HTTP + MCP surface + tier-gate
integration with creator-reputation endpoints + /content/
search.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.marketplace.creator_reputation import (
    CreatorReputationTracker,
    TIER_HIGH, TIER_MEDIUM, TIER_LOW, TIER_NEW,
)
from prsm.marketplace.creator_stake_client import (
    CreatorStakeClient, MIN_HIGH_TIER_STAKE_WEI,
)
from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_creator_stake,
)
from prsm.node.api import create_api_app


def _client(rep_tracker=None, stake_client=None,
            content_index=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._creator_reputation_tracker = rep_tracker
    node._creator_stake_client = stake_client
    if content_index is not None:
        node.content_index = content_index
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def _seed_high(t, creator="alice"):
    """Seed creator into TIER_HIGH (50 distinct + all repeat)."""
    for i in range(50):
        t.record_access(
            creator_id=creator, purchaser_id=f"p{i}",
            content_id="c0",
        )
        t.record_access(
            creator_id=creator, purchaser_id=f"p{i}",
            content_id="c1",
        )


# ── GET /marketplace/creator-stake/{id} ──────────────────


def test_stake_balance_503_when_unwired():
    resp = _client(stake_client=None).get(
        "/marketplace/creator-stake/alice",
    )
    assert resp.status_code == 503


def test_stake_balance_zero_for_unknown():
    s = CreatorStakeClient()
    resp = _client(stake_client=s).get(
        "/marketplace/creator-stake/nobody",
    )
    body = resp.json()
    assert body["balance_wei"] == 0
    assert body["high_tier_eligible"] is False


def test_stake_balance_after_stake():
    s = CreatorStakeClient()
    s.stake("alice", amount_wei=MIN_HIGH_TIER_STAKE_WEI)
    resp = _client(stake_client=s).get(
        "/marketplace/creator-stake/alice",
    )
    body = resp.json()
    assert body["balance_wei"] == MIN_HIGH_TIER_STAKE_WEI
    assert body["high_tier_eligible"] is True
    assert body["min_high_tier_stake_wei"] == MIN_HIGH_TIER_STAKE_WEI


# ── POST /marketplace/creator-stake/stake ────────────────


def test_stake_post_503_when_unwired():
    resp = _client(stake_client=None).post(
        "/marketplace/creator-stake/stake",
        json={"creator_id": "alice", "amount_wei": 100},
    )
    assert resp.status_code == 503


def test_stake_post_happy_path():
    s = CreatorStakeClient()
    resp = _client(stake_client=s).post(
        "/marketplace/creator-stake/stake",
        json={"creator_id": "alice", "amount_wei": 500},
    )
    assert resp.status_code == 200
    assert s.stake_balance("alice") == 500


def test_stake_post_422_zero_amount():
    s = CreatorStakeClient()
    resp = _client(stake_client=s).post(
        "/marketplace/creator-stake/stake",
        json={"creator_id": "alice", "amount_wei": 0},
    )
    assert resp.status_code == 422


def test_stake_post_422_missing_creator():
    s = CreatorStakeClient()
    resp = _client(stake_client=s).post(
        "/marketplace/creator-stake/stake",
        json={"amount_wei": 100},
    )
    assert resp.status_code == 422


# ── POST /marketplace/creator-stake/slash ────────────────


def test_slash_post_happy_path():
    s = CreatorStakeClient()
    s.stake("alice", amount_wei=1000)
    resp = _client(stake_client=s).post(
        "/marketplace/creator-stake/slash",
        json={
            "creator_id": "alice",
            "amount_wei": 400,
            "reason": "confirmed spam",
        },
    )
    assert resp.status_code == 200
    assert s.stake_balance("alice") == 600


def test_slash_post_422_missing_reason():
    s = CreatorStakeClient()
    s.stake("alice", amount_wei=1000)
    resp = _client(stake_client=s).post(
        "/marketplace/creator-stake/slash",
        json={"creator_id": "alice", "amount_wei": 100},
    )
    assert resp.status_code == 422


# ── Tier-gate integration with reputation row ────────────


def test_creator_row_high_tier_demoted_without_stake():
    """Score-based tier is HIGH but no stake → row surfaces
    MEDIUM (stake-gated)."""
    t = CreatorReputationTracker()
    s = CreatorStakeClient()
    _seed_high(t)
    # Pre-check: score-only tier
    assert t.tier_for("alice") == TIER_HIGH
    # With stake gate: demoted
    resp = _client(rep_tracker=t, stake_client=s).get(
        "/marketplace/creator-reputation/alice",
    )
    body = resp.json()
    assert body["tier"] == TIER_MEDIUM


def test_creator_row_high_tier_held_with_stake():
    t = CreatorReputationTracker()
    s = CreatorStakeClient()
    _seed_high(t)
    s.stake("alice", amount_wei=MIN_HIGH_TIER_STAKE_WEI)
    resp = _client(rep_tracker=t, stake_client=s).get(
        "/marketplace/creator-reputation/alice",
    )
    body = resp.json()
    assert body["tier"] == TIER_HIGH


def test_creator_row_lower_tiers_unaffected_by_stake():
    """LOW stays LOW even with massive stake."""
    t = CreatorReputationTracker()
    s = CreatorStakeClient()
    # 15 distinct, no repeats → LOW
    for i in range(15):
        t.record_access(
            creator_id="alice", purchaser_id=f"p{i}",
            content_id=f"c{i}",
        )
    s.stake("alice", amount_wei=MIN_HIGH_TIER_STAKE_WEI * 100)
    resp = _client(rep_tracker=t, stake_client=s).get(
        "/marketplace/creator-reputation/alice",
    )
    body = resp.json()
    assert body["tier"] == TIER_LOW


def test_creator_row_no_stake_client_passthrough():
    """When stake_client unwired, tier is score-based only
    (preserves sprint 287/288 behavior)."""
    t = CreatorReputationTracker()
    _seed_high(t)
    resp = _client(rep_tracker=t, stake_client=None).get(
        "/marketplace/creator-reputation/alice",
    )
    body = resp.json()
    assert body["tier"] == TIER_HIGH


# ── Tier-gate integration with /content/search ───────────


class _FakeIndex:
    def __init__(self, creators_by_cid):
        self.creators_by_cid = creators_by_cid
    def search(self, q, limit=20):
        results = []
        for cid, creator in self.creators_by_cid.items():
            r = MagicMock()
            r.cid = cid
            r.filename = f"{cid}.bin"
            r.size_bytes = 1024
            r.content_hash = "sha"
            r.creator_id = creator
            # sp978 (decision A) — the stake gate keys on creator_eth_address.
            # The fixture stakes by this same identity (see the stake() calls),
            # so a record's stake-identity matches what was bonded.
            r.creator_eth_address = creator
            r.providers = []
            r.created_at = 0
            r.metadata = {}
            r.royalty_rate = 0.1
            r.parent_cids = []
            results.append(r)
        return results[:limit]


def test_search_min_tier_high_excludes_unstaked():
    """HIGH-score creator without stake is demoted →
    min_tier=high filter excludes them."""
    t = CreatorReputationTracker()
    s = CreatorStakeClient()
    _seed_high(t, "creator_no_stake")
    _seed_high(t, "creator_staked")
    s.stake(
        "creator_staked",
        amount_wei=MIN_HIGH_TIER_STAKE_WEI,
    )
    idx = _FakeIndex({
        "cid_a": "creator_no_stake",
        "cid_b": "creator_staked",
    })
    resp = _client(
        rep_tracker=t, stake_client=s, content_index=idx,
    ).get("/content/search?q=foo&min_tier=high")
    body = resp.json()
    cids = [r["cid"] for r in body["results"]]
    assert cids == ["cid_b"]


# ── MCP tool ─────────────────────────────────────────────


def test_mcp_tool_registered():
    assert "prsm_creator_stake" in TOOL_HANDLERS


class TestMcp:
    @pytest.mark.asyncio
    async def test_missing_action(self):
        r = await handle_prsm_creator_stake({})
        assert "action" in r.lower()

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        r = await handle_prsm_creator_stake(
            {"action": "explode"},
        )
        assert "must be" in r.lower()

    @pytest.mark.asyncio
    async def test_balance_action(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "creator_id": "alice",
                "balance_wei": 500,
                "high_tier_eligible": False,
                "min_high_tier_stake_wei": 1000,
            }),
        ) as mock_call:
            r = await handle_prsm_creator_stake({
                "action": "balance", "creator_id": "alice",
            })
        args = mock_call.await_args[0]
        assert args[1] == "/marketplace/creator-stake/alice"
        assert "alice" in r
        assert "500" in r

    @pytest.mark.asyncio
    async def test_balance_requires_creator_id(self):
        r = await handle_prsm_creator_stake(
            {"action": "balance"},
        )
        assert "creator_id" in r

    @pytest.mark.asyncio
    async def test_stake_action_happy_path(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "creator_id": "alice",
                "balance_wei": 500,
                "high_tier_eligible": False,
            }),
        ) as mock_call:
            r = await handle_prsm_creator_stake({
                "action": "stake",
                "creator_id": "alice",
                "amount_wei": 500,
            })
        args = mock_call.await_args[0]
        assert args[0] == "POST"
        assert args[1] == "/marketplace/creator-stake/stake"
        assert args[2]["creator_id"] == "alice"
        assert args[2]["amount_wei"] == 500
        assert "500" in r

    @pytest.mark.asyncio
    async def test_stake_action_requires_amount(self):
        r = await handle_prsm_creator_stake({
            "action": "stake", "creator_id": "alice",
        })
        assert "amount_wei" in r

    @pytest.mark.asyncio
    async def test_slash_action_happy_path(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "creator_id": "alice",
                "balance_wei": 600,
                "slashed_wei": 400,
            }),
        ) as mock_call:
            r = await handle_prsm_creator_stake({
                "action": "slash",
                "creator_id": "alice",
                "amount_wei": 400,
                "reason": "confirmed spam",
            })
        args = mock_call.await_args[0]
        assert args[1] == "/marketplace/creator-stake/slash"
        assert args[2]["reason"] == "confirmed spam"
        assert "400" in r

    @pytest.mark.asyncio
    async def test_slash_requires_reason(self):
        r = await handle_prsm_creator_stake({
            "action": "slash",
            "creator_id": "alice",
            "amount_wei": 400,
        })
        assert "reason" in r
