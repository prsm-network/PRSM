"""Sprint 1186 — real readiness probe (day-one-live blocker #9).

/health returned 200 unconditionally even when the node could not serve its primary path
(inference dark), so a load balancer would route traffic to a broken node. /health stays a
LIVENESS probe (process up → 200; restart-me semantics). This adds a READINESS probe
(/health/ready + /readyz, ungated for the LB): 200 when the node can serve inference, 503
otherwise — the correct "should I receive traffic" gate (failing readiness drains traffic,
not restarts the pod). Cheap (no RPC); body carries coarse non-sensitive subsystem flags.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.readiness import compute_readiness
from prsm.node.api import create_api_app


# ── compute_readiness: cheap, inference-gated ────────────────────────────────────────

def test_ready_when_inference_executor_present():
    node = SimpleNamespace(inference_executor=object(), settlement_client=None,
                           ftns_ledger=None)
    ready, detail = compute_readiness(node)
    assert ready is True
    assert detail["ready"] is True
    assert detail["subsystems"]["inference"] is True


def test_not_ready_when_inference_executor_absent():
    node = SimpleNamespace(inference_executor=None,
                           _onchain_settlement_client=None,
                           ftns_ledger=None)
    ready, detail = compute_readiness(node)
    assert ready is False
    assert detail["ready"] is False
    assert "inference" in (detail.get("reason") or "").lower()
    assert detail["subsystems"]["inference"] is False


def test_detail_surfaces_coarse_subsystem_flags_only():
    # sp1217 — settlement reads the REAL node attr (_onchain_settlement_client),
    # not the never-set settlement_client.
    node = SimpleNamespace(inference_executor=object(),
                           _onchain_settlement_client=object(),
                           ftns_ledger=object())
    _, detail = compute_readiness(node)
    subs = detail["subsystems"]
    # coarse booleans only — no addresses / topology leaked (safe for an ungated probe)
    assert subs == {"inference": True, "settlement": True, "ftns_ledger": True}


def test_compute_readiness_never_raises_on_odd_node():
    ready, detail = compute_readiness(object())  # no attrs at all
    assert ready is False and detail["ready"] is False


def test_settlement_reads_real_node_attr_not_legacy_name():
    """sp1217 regression — the readiness 'settlement' boolean must reflect the
    node's actual on-chain settlement client (_onchain_settlement_client). A
    node carrying ONLY the legacy/never-set 'settlement_client' name must report
    settlement=False (the bug was reading that name → always False)."""
    legacy_only = SimpleNamespace(inference_executor=object(),
                                  settlement_client=object(),  # legacy, ignored
                                  ftns_ledger=object())
    _, detail = compute_readiness(legacy_only)
    assert detail["subsystems"]["settlement"] is False

    real = SimpleNamespace(inference_executor=object(),
                           _onchain_settlement_client=object(),
                           ftns_ledger=object())
    _, detail2 = compute_readiness(real)
    assert detail2["subsystems"]["settlement"] is True


# ── endpoints: 200/503 + ungated ─────────────────────────────────────────────────────

def _client(*, inference):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.inference_executor = object() if inference else None
    node.settlement_client = None
    node.ftns_ledger = None
    return TestClient(create_api_app(node, enable_security=False),
                      raise_server_exceptions=False)


def test_ready_endpoint_200_when_serving():
    for path in ("/health/ready", "/readyz"):
        r = _client(inference=True).get(path)
        assert r.status_code == 200, path
        assert r.json()["ready"] is True


def test_ready_endpoint_503_when_inference_dark():
    for path in ("/health/ready", "/readyz"):
        r = _client(inference=False).get(path)
        assert r.status_code == 503, path
        assert r.json()["ready"] is False


def test_liveness_health_stays_200_regardless():
    # /health is liveness — always 200 even when not ready (so the LB drains traffic
    # via readiness instead of k8s RESTARTING a merely-not-ready pod).
    r = _client(inference=False).get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
