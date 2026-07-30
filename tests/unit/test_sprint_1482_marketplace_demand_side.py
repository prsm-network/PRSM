"""Sprint 1482 — marketplace DEMAND side: ingest listings by default + make supply visible.

sp1459 activated the supply side (nodes broadcast signed listings), but the
MarketplaceDirectory that INGESTS them was constructed in exactly one place —
inside `_build_query_orchestrator_or_none`, behind PRSM_QUERY_ORCHESTRATOR_ENABLED
(default OFF). So on a default node nothing subscribed to
GOSSIP_MARKETPLACE_LISTING and every advertised listing was dropped on the floor:
the marketplace was live-but-empty AND live-but-unread. Live proof at assessment
time: both operator daemons reported `"query_orchestrator_state":"disabled"`.

sp1482 hoists the directory to a top-level node subsystem built UNCONDITIONALLY,
and adds GET /marketplace/providers so the provider set is observable at all.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from prsm.marketplace.directory import MarketplaceDirectory
from prsm.node.api import create_api_app
from prsm.node.gossip import GOSSIP_MARKETPLACE_LISTING


class _FakeGossip:
    """Records subscriptions so we can assert the directory actually listens."""

    def __init__(self):
        self.handlers = {}

    def subscribe(self, topic, handler):
        self.handlers.setdefault(topic, []).append(handler)

    def emit(self, topic, payload):
        for h in self.handlers.get(topic, []):
            h(payload)


def test_directory_subscribes_to_the_listing_topic():
    """The whole demand side rests on this one subscription existing."""
    g = _FakeGossip()
    MarketplaceDirectory(g)
    assert GOSSIP_MARKETPLACE_LISTING in g.handlers
    assert len(g.handlers[GOSSIP_MARKETPLACE_LISTING]) == 1


# ───────────────── node wiring: built unconditionally ─────────────────

def _node_module():
    import prsm.node.node as node_mod
    return node_mod


def test_directory_is_hoisted_out_of_the_query_orchestrator_block():
    """★ The regression this sprint fixes. The directory must be constructed in
    initialize() — NOT only inside the QO block — or a default node ingests
    nothing. Asserted structurally against the source so a future refactor that
    pushes it back behind the QO gate fails loudly."""
    import inspect
    src = inspect.getsource(_node_module())

    init_marker = "# ── Marketplace DEMAND side (sp1482) ─"
    assert init_marker in src, (
        "the unconditional demand-side construction block is gone — a default "
        "node would stop ingesting marketplace listings"
    )
    # The construction must NOT be gated on the query-orchestrator env flag.
    block_start = src.index(init_marker)
    block = src[block_start:block_start + 1600]
    assert "MarketplaceDirectory(self.gossip)" in block
    # Strip comments first: the block's own docs MENTION the env var to explain
    # the bug being fixed. The property is that no CODE gates on it.
    code_only = "\n".join(
        ln for ln in block.splitlines() if not ln.strip().startswith("#")
    )
    assert "PRSM_QUERY_ORCHESTRATOR_ENABLED" not in code_only, (
        "the demand side must not be gated behind the query orchestrator again"
    )


def test_query_orchestrator_reuses_the_hoisted_directory():
    """★ Two MarketplaceDirectory instances would BOTH subscribe to the listing
    topic and keep divergent provider sets — the QO would select against a
    directory that /marketplace/providers never shows. The QO must reuse the
    hoisted instance."""
    import inspect
    src = inspect.getsource(_node_module())
    assert 'getattr(self, "marketplace_directory", None)' in src, (
        "the QO builder no longer reuses the hoisted directory — it would create "
        "a second, divergent subscriber"
    )


# ───────────────── the /marketplace/providers view ─────────────────

def _listing(provider_id, price, *, verified=False, tier="T1", ttl=3600,
             advertised_at=None, dtypes=("fp16",)):
    """A ProviderListing-shaped stand-in. has_verified_stake_binding is the real
    contract the route consumes (it does the crypto), so it is stubbed here."""
    return SimpleNamespace(
        listing_id=f"lst-{provider_id}",
        provider_id=provider_id,
        provider_pubkey_b64="pk",
        capacity_shards_per_sec=2.0,
        max_shard_bytes=1024,
        supported_dtypes=list(dtypes),
        price_per_shard_ftns=price,
        tee_capable=False,
        stake_tier=tier,
        advertised_at_unix=advertised_at or int(time.time()),
        ttl_seconds=ttl,
        signature="sig",
        stake_eth_address="0x" + "a" * 40 if verified else None,
        stake_binding_sig="0xsig" if verified else None,
        has_verified_stake_binding=lambda v=verified: v,
    )


class _StubDirectory:
    def __init__(self, listings):
        self._l = list(listings)

    def list_active_providers(self, at_unix=None):
        return list(self._l)


def _client(directory):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node.marketplace_directory = directory
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_providers_route_lists_ingested_listings():
    c = _client(_StubDirectory([
        _listing("prov-a", 0.5),
        _listing("prov-b", 0.25, verified=True),
    ]))
    r = c.get("/marketplace/providers")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 2
    assert body["total_active"] == 2
    assert body["verified_stake_count"] == 1
    ids = [p["provider_id"] for p in body["providers"]]
    # Verified-stake listings sort first (they are the ones weighted by REAL stake).
    assert ids[0] == "prov-b"


def test_providers_route_marks_unverified_tier_claims():
    """★ A self-asserted stake_tier is meaningless for selection weight unless the
    binding verifies (sp1457/sp1463). The row must surface both together so a
    reader is never misled by an unbacked T4 claim."""
    c = _client(_StubDirectory([_listing("liar", 1.0, tier="T4", verified=False)]))
    row = c.get("/marketplace/providers").json()["providers"][0]
    assert row["stake_tier"] == "T4"
    assert row["verified_stake_binding"] is False
    # An unverified binding must not leak an address that implies backing.
    assert row["stake_eth_address"] is None


def test_verified_stake_only_filter():
    c = _client(_StubDirectory([
        _listing("plain", 0.5),
        _listing("bound", 0.75, verified=True),
    ]))
    body = c.get("/marketplace/providers?verified_stake_only=true").json()
    assert [p["provider_id"] for p in body["providers"]] == ["bound"]
    assert body["count"] == 1
    assert body["total_active"] == 2   # unfiltered total still reported


def test_providers_route_reports_expiry():
    now = int(time.time())
    c = _client(_StubDirectory([_listing("p", 0.5, ttl=600, advertised_at=now)]))
    row = c.get("/marketplace/providers").json()["providers"][0]
    assert row["expires_at_unix"] == now + 600


def test_empty_directory_is_an_empty_list_not_an_error():
    """An empty marketplace is a normal state (nobody advertising yet), and must
    read as empty rather than as a failure."""
    body = _client(_StubDirectory([])).get("/marketplace/providers").json()
    assert body["providers"] == [] and body["count"] == 0


def test_missing_directory_is_503_with_an_actionable_reason():
    c = _client(None)
    r = c.get("/marketplace/providers")
    assert r.status_code == 503
    assert "gossip" in r.json()["detail"].lower()


def test_limit_is_validated():
    c = _client(_StubDirectory([_listing("p", 0.5)]))
    assert c.get("/marketplace/providers?limit=0").status_code == 422
    assert c.get("/marketplace/providers?limit=10001").status_code == 422


def test_malformed_binding_is_treated_as_unverified_not_a_500():
    """A hostile listing whose binding check raises must not take the route down."""
    bad = _listing("bad", 0.5, verified=True)
    def _boom():
        raise ValueError("corrupt binding")
    bad.has_verified_stake_binding = _boom
    row = _client(_StubDirectory([bad])).get("/marketplace/providers").json()["providers"][0]
    assert row["verified_stake_binding"] is False


# ───────────────── auth posture ─────────────────

def test_providers_route_is_a_public_read():
    """Consistent with its /marketplace/reputation sibling: the listings are
    already signed public gossip, so serving them discloses nothing new — and
    discoverability is the point of the demand side."""
    from prsm.api.auth_middleware import is_protected_path
    assert is_protected_path("/marketplace/providers") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
