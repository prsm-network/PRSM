"""POST /compute/cleanup-stale — manual trigger for the
periodic-cleanup task that auto-refunds expired escrows.

Operators sometimes need to force-cleanup without waiting for
the 10-min cleanup loop:
  - Just lowered PRSM_ESCROW_TIMEOUT_SEC and want immediate effect
  - Stuck escrows blocking a downstream operation
  - Drain before maintenance / restart

Returns the number of escrows refunded.
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.payment_escrow import (
    EscrowEntry, EscrowStatus, PaymentEscrow,
)


def _ledger():
    led = MagicMock()
    led.get_balance = AsyncMock(return_value=100.0)
    tx = MagicMock()
    tx.tx_id = "tx-stub"
    led.transfer = AsyncMock(return_value=tx)
    led.create_wallet = AsyncMock()
    return led


def _node(escrow_svc):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node._payment_escrow = escrow_svc
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


class TestCleanupStaleAvailability:
    def test_503_when_escrow_not_wired(self):
        node = _node(None)
        resp = _client(node).post("/compute/cleanup-stale")
        assert resp.status_code == 503


class TestCleanupStaleHappyPath:
    def test_cleans_expired_escrows(self):
        escrow = PaymentEscrow(
            ledger=_ledger(), node_id="test-node",
            # 30s window: the stale entries (1h old) are unambiguously expired, while the
            # "fresh" entry created just below stays well within the window for the whole
            # test (node + TestClient build + POST). A tight timeout here was flaky — the
            # setup latency alone could exceed it, expiring the "fresh" escrow too.
            default_timeout=30.0,
        )
        # Seed two stale + one fresh.
        old = time.time() - 3600.0
        for jid in ("j1", "j2"):
            entry = EscrowEntry(
                escrow_id=f"esc-{jid}", job_id=jid,
                requester_id="req", amount=1.0,
                status=EscrowStatus.PENDING,
                created_at=old,
            )
            escrow._escrows[entry.escrow_id] = entry
        # Fresh one (won't be cleaned).
        fresh = EscrowEntry(
            escrow_id="esc-fresh", job_id="j-fresh",
            requester_id="req", amount=1.0,
            status=EscrowStatus.PENDING,
            created_at=time.time(),
        )
        escrow._escrows[fresh.escrow_id] = fresh

        node = _node(escrow)
        resp = _client(node).post("/compute/cleanup-stale")
        assert resp.status_code == 200
        body = resp.json()
        assert body["cleaned"] == 2
        # Fresh escrow still PENDING.
        assert fresh.status == EscrowStatus.PENDING

    def test_no_stale_escrows_returns_zero(self):
        escrow = PaymentEscrow(
            ledger=_ledger(), node_id="test-node",
            default_timeout=86400.0,  # 1 day — nothing expires
        )
        entry = EscrowEntry(
            escrow_id="e1", job_id="j1",
            requester_id="req", amount=1.0,
            status=EscrowStatus.PENDING,
            created_at=time.time(),
        )
        escrow._escrows[entry.escrow_id] = entry
        node = _node(escrow)
        resp = _client(node).post("/compute/cleanup-stale")
        body = resp.json()
        assert body["cleaned"] == 0


class TestCleanupStaleFailures:
    @pytest.mark.asyncio
    async def test_cleanup_raising_returns_502(self):
        """If cleanup_expired_escrows raises, surface as 502."""
        escrow = MagicMock()
        escrow.cleanup_expired_escrows = AsyncMock(
            side_effect=RuntimeError("ledger down"),
        )
        node = _node(escrow)
        resp = _client(node).post("/compute/cleanup-stale")
        assert resp.status_code == 502
        assert "ledger down" in resp.json()["detail"]
