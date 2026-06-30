"""Sprint 1328 — bind the multi-stage quote to the deterministic PRICE, not the budget.

The 2026-06-30 cross-cloud testnet GO surfaced this live: the quote echoed ``budget_ftns`` as the
total, but the serve settles ``receipt.cost_ftns == estimate_cost`` (cost_per_layer × num_layers,
deterministic). The per-stage auth commits to EXACT (payee, share) pairs, so when budget != cost
the settle-time payee_set_hash didn't match the signed one and the fail-closed gate rejected every
paid multi-stage inference (we had to set budget==cost by hand). Now the quote returns the price
(estimate_cost) as the total + treats budget_ftns as a CAP. Since estimate_cost is the IDENTICAL
value the serve charges, the signed shares match the settled shares for any budget >= price.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from prsm.compute.inference.topology_rotation import topology_from_chain_stages
from prsm.node.api import create_api_app

_ETH_A = "0x" + "a1" * 20
_ETH_B = "0x" + "b2" * 20


class _FakeExec:
    """plan_topology + estimate_cost, the two surfaces the quote needs."""
    def __init__(self, topology, cost):
        self._t = topology
        self._cost = Decimal(str(cost))

    async def plan_topology(self, req):
        return self._t

    async def estimate_cost(self, req):
        return self._cost


class _FakeWalletMap:
    def __init__(self, mapping):
        self._m = mapping

    def resolve(self, rid):
        return self._m.get(rid, rid)


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
    body.setdefault("model_id", "qwen2.5-7b")
    body.setdefault("prompt", "Hello")
    return client.post("/compute/inference/quote-multistage", json=body)


def test_quote_total_is_the_price_not_the_budget(monkeypatch):
    """THE fix: total_value_wei == estimate_cost (0.28), regardless of a larger budget (1.0)."""
    topo = topology_from_chain_stages(["nodeA", "nodeB"])
    wallet = _FakeWalletMap({"nodeA": _ETH_A, "nodeB": _ETH_B})
    r = _post(_client(_FakeExec(topo, "0.28"), monkeypatch, wallet=wallet), budget_ftns=1.0)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["settleable"] is True
    assert d["price_ftns"] == "0.28"
    assert d["total_value_wei"] == 28 * 10 ** 16  # 0.28 FTNS, NOT the 1.0 budget
    # 0.28 split over 2 nodes = 0.14 each
    payees = {a.lower(): s for a, s in d["payees"]}
    assert payees[_ETH_A.lower()] == 14 * 10 ** 16
    assert payees[_ETH_B.lower()] == 14 * 10 ** 16


def test_round_trip_hash_matches_what_requester_signs(monkeypatch):
    """The quoted payee_set_hash == what build_per_stage_payment_authorization signs over the
    PRICE-based shares — so the auth the requester signs matches what the serve settles."""
    from eth_account import Account
    from prsm.settlement.payment_client import build_per_stage_payment_authorization

    topo = topology_from_chain_stages(["nodeA", "nodeB"])
    wallet = _FakeWalletMap({"nodeA": _ETH_A, "nodeB": _ETH_B})
    q = _post(_client(_FakeExec(topo, "0.28"), monkeypatch, wallet=wallet), budget_ftns=1.0).json()
    payees_ftns = [(a, Decimal(s) / (Decimal(10) ** 18)) for a, s in q["payees"]]
    auth = build_per_stage_payment_authorization(
        requester_key=Account.create().key.hex(), payees=payees_ftns,
        model_id="qwen2.5-7b", prompt="Hello", max_tokens=8,
        privacy_tier="none", content_tier="A", expiry_unix=9999999999)
    assert auth["payload"]["payee_set_hash"] == q["payee_set_hash"]


def test_price_above_budget_is_not_settleable(monkeypatch):
    """budget_ftns is a CAP — a price above it is rejected (mirrors the serve's budget<cost gate)."""
    topo = topology_from_chain_stages(["nodeA", "nodeB"])
    wallet = _FakeWalletMap({"nodeA": _ETH_A, "nodeB": _ETH_B})
    r = _post(_client(_FakeExec(topo, "5.0"), monkeypatch, wallet=wallet), budget_ftns=1.0)
    d = r.json()
    assert d["multi_stage"] is True and d["settleable"] is False
    assert "exceeds budget" in d["reason"]
    assert d["price_ftns"] == "5.0" and d["budget_ftns"] == "1.0"


def test_price_equal_budget_settleable(monkeypatch):
    topo = topology_from_chain_stages(["nodeA", "nodeB"])
    wallet = _FakeWalletMap({"nodeA": _ETH_A, "nodeB": _ETH_B})
    d = _post(_client(_FakeExec(topo, "1.0"), monkeypatch, wallet=wallet), budget_ftns=1.0).json()
    assert d["settleable"] is True and d["total_value_wei"] == 10 ** 18


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
