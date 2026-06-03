"""PROTECTED_PREFIXES covers /admin/, /wallet/, /content/arbitration/
(sprint 138).

Pre-fix: when an operator set PRSM_NODE_API_KEY (the documented
mechanism for protecting their node), /admin/heartbeat/trigger,
/admin/distribution/trigger, and /wallet/royalty/claim stayed
unprotected. Network-adjacent attackers could spend operator gas.

Post-fix: the prefix list covers all sensitive write endpoints.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from prsm.api.auth_middleware import (
    PROTECTED_PREFIXES, NodeAuthMiddleware, hash_api_key,
)
from starlette.requests import Request


def _is_protected(path: str) -> bool:
    return any(path.startswith(p) for p in PROTECTED_PREFIXES)


class TestAdminProtected:
    def test_heartbeat_trigger_protected(self):
        assert _is_protected("/admin/heartbeat/trigger")

    def test_distribution_trigger_protected(self):
        assert _is_protected("/admin/distribution/trigger")

    def test_admin_history_endpoints_protected(self):
        # /admin/* read endpoints — protected too because
        # operators may not want raw event-log enumeration
        # exposed unauthenticated
        for p in (
            "/admin/webhook-history",
            "/admin/slash-history",
            "/admin/heartbeat-history",
            "/admin/distribution-history",
            "/admin/earnings-summary",
        ):
            assert _is_protected(p), f"{p} should be protected"


class TestWalletProtected:
    def test_royalty_claim_protected(self):
        assert _is_protected("/wallet/royalty/claim")

    def test_offramp_quote_protected(self):
        assert _is_protected("/wallet/offramp/quote")

    def test_balance_endpoints_protected(self):
        # Read-side wallet info is financial-adjacent PII
        assert _is_protected("/wallet/spend")
        assert _is_protected("/wallet/escrows")


class TestArbitrationProtected:
    def test_preview_resolution_protected(self):
        assert _is_protected("/content/arbitration/preview-resolution")


class TestCreatorStakeProtected:
    """sp970 — the creator-stake MUTATION endpoints write the in-memory balance
    that gates HIGH creator-tier eligibility (search ranking + tier label).
    Unprotected, any reachable caller could grant a creator free HIGH tier
    (/stake) or grief a competitor's tier (/slash). They join /staking/ +
    /wallet/ in the protected set (enforced when PRSM_NODE_API_KEY is configured;
    dev-mode no-key is unaffected). NOTE: the deeper gap — the gate has no
    on-chain teeth (in-memory only, no CreatorStakeRegistry backend) — is a
    separate creator-stake commissioning item; this closes the auth vector now."""

    def test_creator_stake_mutation_protected(self):
        assert _is_protected("/marketplace/creator-stake/stake")
        assert _is_protected("/marketplace/creator-stake/slash")

    def test_creator_stake_read_protected_via_prefix(self):
        assert _is_protected("/marketplace/creator-stake/0xCreator")


class TestUnchanged:
    def test_public_endpoints_still_pass(self):
        # Sanity: /health, / etc. not in protected list
        for p in ("/", "/health", "/status", "/openapi.json"):
            assert not _is_protected(p), f"{p} should NOT be protected"

    def test_pre_existing_protections_intact(self):
        # The original three guarded endpoints
        assert _is_protected("/settler/register")
        assert _is_protected("/content/upload")
        assert _is_protected("/content/upload/shard")  # via startswith
        assert _is_protected("/compute/forge")


class TestSprint139Protected:
    """Audit-pass coverage of remaining sensitive endpoints."""

    def test_bridge_protected(self):
        assert _is_protected("/bridge/deposit")
        assert _is_protected("/bridge/withdraw")
        assert _is_protected("/bridge/transactions")

    def test_ftns_faucet_protected(self):
        assert _is_protected("/ftns/faucet")

    def test_ledger_transfer_protected(self):
        assert _is_protected("/ledger/transfer")

    def test_staking_protected(self):
        for p in (
            "/staking/stake", "/staking/unstake",
            "/staking/claim-rewards",
            "/staking/withdraw/abc123",
            "/staking/cancel-unstake/abc123",
        ):
            assert _is_protected(p), f"{p} should be protected"

    def test_agents_actions_protected(self):
        for p in (
            "/agents/abc/allowance",
            "/agents/abc/pause",
            "/agents/abc/resume",
        ):
            assert _is_protected(p), f"{p} should be protected"

    def test_node_resources_put_protected(self):
        assert _is_protected("/node/resources")

    def test_settlement_protected(self):
        assert _is_protected("/settlement/flush")

    def test_compute_endpoints_broadened(self):
        for p in (
            "/compute/submit",
            "/compute/cancel/job-abc",
            "/compute/inference",
            "/compute/cleanup-stale",
        ):
            assert _is_protected(p), f"{p} should be protected"
