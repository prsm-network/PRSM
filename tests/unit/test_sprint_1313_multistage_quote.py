"""Sprint 1313 (S1) — paid multi-stage QUOTE / topology-preview.

The first piece of live big-model paid settlement (Design A): a read-only quote that
previews the planned stage→node topology + the (eth_payee, share_wei) SET the requester
must authorize, so a paying requester can sign a per-stage PaymentAuthorization for the
EXACT set the serve will pay.

Pins: the shared topology helper (drift-proof quote==serve), and the
/compute/inference/quote-multistage endpoint — including a ROUND-TRIP proving the quoted
payee_set_hash equals what build_per_stage_payment_authorization signs (so the requester
can authorize exactly the quoted set).
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from prsm.compute.inference.topology_rotation import (
    TopologyAssignment,
    topology_from_chain_stages,
)
from prsm.node.api import create_api_app

_ETH_A = "0x" + "a1" * 20
_ETH_B = "0x" + "b2" * 20


# ── shared topology helper (drift-proof) ─────────────────────────────────────

def test_topology_from_chain_stages():
    t = topology_from_chain_stages(["nodeA", "nodeB", "nodeC"])
    assert isinstance(t, TopologyAssignment)
    assert t.stage_count == 3 and t.slots_per_stage == 1
    assert t.positions == {(0, 0): "nodeA", (1, 0): "nodeB", (2, 0): "nodeC"}


def test_topology_from_chain_stages_empty():
    t = topology_from_chain_stages([])
    assert t.stage_count == 0 and t.positions == {}


# ── quote endpoint ───────────────────────────────────────────────────────────

class _FakeExec:
    def __init__(self, topology):
        self._t = topology

    async def plan_topology(self, req):
        return self._t

    async def estimate_cost(self, req):
        # sp1328 — the quote now binds total_value_wei to the deterministic PRICE
        # (estimate_cost), not budget_ftns. These tests quote at budget_ftns=1.0, so a price
        # of 1.0 keeps their total==1e18 / 0.5-each assertions valid.
        from decimal import Decimal
        return Decimal("1.0")


class _FakeWalletMap:
    def __init__(self, mapping):
        self._m = mapping

    def resolve(self, rid):
        return self._m.get(rid, rid)   # pass-through (unmapped) like the real one


def _client(executor, monkeypatch, wallet=None):
    if wallet is not None:
        monkeypatch.setattr(
            "prsm.node.compute_wallet_map.ComputeWalletMap.from_env",
            classmethod(lambda cls, *a, **k: wallet))
    node = MagicMock()
    node.identity.node_id = "requester-node"
    node.inference_executor = executor
    return TestClient(create_api_app(node, enable_security=False),
                      raise_server_exceptions=False)


def _post(client, **body):
    body.setdefault("model_id", "qwen2.5-72b")
    body.setdefault("prompt", "Hello")
    body.setdefault("budget_ftns", 1.0)
    return client.post("/compute/inference/quote-multistage", json=body)


def test_no_plan_support_returns_not_multistage(monkeypatch):
    r = _post(_client(object(), monkeypatch))   # executor without plan_topology
    assert r.status_code == 200
    assert r.json()["multi_stage"] is False


def test_single_node_plan_returns_not_multistage(monkeypatch):
    r = _post(_client(_FakeExec(None), monkeypatch))   # plan_topology → None
    assert r.json()["multi_stage"] is False


def test_two_node_quote_settleable(monkeypatch):
    topo = topology_from_chain_stages(["nodeA", "nodeB"])
    wallet = _FakeWalletMap({"nodeA": _ETH_A, "nodeB": _ETH_B})
    r = _post(_client(_FakeExec(topo), monkeypatch, wallet=wallet), budget_ftns=1.0)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["multi_stage"] is True and d["settleable"] is True
    assert d["stage_count"] == 2
    assert d["total_value_wei"] == 10 ** 18
    # 1 FTNS split over 2 nodes = 0.5 each
    payees = {addr.lower(): share for addr, share in d["payees"]}
    assert payees[_ETH_A.lower()] == 5 * 10 ** 17
    assert payees[_ETH_B.lower()] == 5 * 10 ** 17
    assert d["payee_set_hash"].startswith("0x") and len(d["payee_set_hash"]) == 66


def test_unmapped_node_not_settleable(monkeypatch):
    topo = topology_from_chain_stages(["nodeA", "nodeB"])
    wallet = _FakeWalletMap({"nodeA": _ETH_A})   # nodeB unmapped → pass-through
    r = _post(_client(_FakeExec(topo), monkeypatch, wallet=wallet))
    d = r.json()
    assert d["multi_stage"] is True and d["settleable"] is False
    assert "no registered on-chain payee" in d["reason"]


def test_quote_hash_matches_what_requester_would_sign(monkeypatch):
    """The KEY S1 property: the quoted payee_set_hash == the hash the requester signs via
    build_per_stage_payment_authorization for the same set — so the requester can authorize
    exactly the quoted set + the server's verifier accepts it."""
    from eth_account import Account
    from prsm.settlement.payment_client import build_per_stage_payment_authorization

    topo = topology_from_chain_stages(["nodeA", "nodeB"])
    wallet = _FakeWalletMap({"nodeA": _ETH_A, "nodeB": _ETH_B})
    quote = _post(_client(_FakeExec(topo), monkeypatch, wallet=wallet),
                  budget_ftns=1.0, model_id="qwen2.5-72b", prompt="Hello",
                  max_tokens=8).json()

    # requester rebuilds the auth from the quote's payees (wei → FTNS)
    payees_ftns = [(addr, Decimal(share) / (Decimal(10) ** 18))
                   for addr, share in quote["payees"]]
    auth = build_per_stage_payment_authorization(
        requester_key=Account.create().key.hex(), payees=payees_ftns,
        model_id="qwen2.5-72b", prompt="Hello", max_tokens=8,
        privacy_tier="none", content_tier="A", expiry_unix=9999999999)
    assert auth["payload"]["payee_set_hash"] == quote["payee_set_hash"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
