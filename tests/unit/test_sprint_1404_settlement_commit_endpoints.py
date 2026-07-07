"""Sprint 1404 — operator-triggered on-chain settlement commit/finalize driver endpoints.

The on-chain client (sp1403 target) has commit_ready_batches/finalize_ready_batches but nothing drove
them (no auto-loop). These admin endpoints are the controlled trigger for the funded-ceremony runbook:
the operator commits + finalizes explicitly, verifying each step. OFF (503/"off") without a client.
"""
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.settlement.client import CommittedBatch, FinalizedBatch


def _client(settlement_client):
    node = MagicMock()
    node._onchain_settlement_client = settlement_client
    node._operator_address = "0xREQ"
    return TestClient(create_api_app(node, enable_security=False))


def test_status_and_commit_off_without_client():
    c = _client(None)
    assert c.get("/admin/settlement/onchain/status").json()["onchain_settlement"] == "off"
    assert c.post("/admin/settlement/onchain/commit-ready").status_code == 503
    assert c.post("/admin/settlement/onchain/finalize-ready").status_code == 503


def test_commit_ready_returns_committed_batches():
    sc = MagicMock()
    cb = CommittedBatch(
        batch_id=b"\x01" * 32, tx_hash="0xabc", provider_address="0xPROV",
        requester_address="0xREQ", merkle_root=b"\x02" * 32, receipt_count=1,
        total_value_ftns=10 ** 18, commit_timestamp=123, leaf_hashes=[b"\x04" * 32],
        trigger_reason="size")
    sc.commit_ready_batches = AsyncMock(return_value=[cb])
    r = _client(sc).post("/admin/settlement/onchain/commit-ready")
    assert r.status_code == 200
    body = r.json()["committed"]
    assert len(body) == 1
    assert body[0]["batch_id"] == "01" * 32
    assert body[0]["tx_hash"] == "0xabc"
    assert body[0]["provider_address"] == "0xPROV"
    assert body[0]["total_value_ftns"] == str(10 ** 18)


def test_finalize_ready_returns_finalized():
    sc = MagicMock()
    fb = FinalizedBatch(batch_id=b"\x03" * 32, tx_submitted="0xfin")
    sc.finalize_ready_batches = AsyncMock(return_value=[fb])
    r = _client(sc).post("/admin/settlement/onchain/finalize-ready")
    assert r.status_code == 200
    assert r.json()["finalized"][0]["batch_id"] == "03" * 32


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
