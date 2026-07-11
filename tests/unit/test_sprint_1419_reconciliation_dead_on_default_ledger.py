"""Sprint 1419 — balance reconciliation was DEAD on every node, by default.

`Node._initialize()` builds `DAGLedger` whenever `config.ledger_type == "dag"` — and that is the
DEFAULT (`prsm/node/config.py`: `ledger_type: str = "dag"`). It then hands that object to
`LedgerSync`, whose `_run_reconciliation()` and `_handle_balance_request()` both call
`self.ledger.get_recent_tx_ids(...)`.

`DAGLedger` never had that method. Only `LocalLedger` (the non-default legacy ledger) and
`DAGLedgerAdapter` (which is dead code — never instantiated anywhere in prsm/) do. So on a default
node every reconciliation cycle raised AttributeError, and `_reconciliation_loop` swallowed it:

    try:
        await self._run_reconciliation()
    except Exception as e:
        logger.error(f"Reconciliation error: {e}")

Net effect: the node never sent balance proofs, and never answered a peer's balance_request. The
subsystem was inert on the entire live network and nothing failed — same shape as sp1412/sp1178/
sp1411/F54: a defense that cannot fire is indistinguishable from one that does not exist.

`LedgerSync.__init__` is annotated `ledger: LocalLedger`, which is why this type-checked: the
annotation was simply wrong about what production passes.

The third test is the one that matters long-term — it pins the whole INTERFACE rather than this one
method, so the next method LedgerSync grows cannot silently miss the default ledger.
"""
from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.dag_ledger import DAGLedger, TransactionType
from prsm.node.identity import generate_node_identity
from prsm.node.ledger_sync import LedgerSync
from prsm.node.local_ledger import LocalLedger


def _ledger_sync_ledger_methods() -> set:
    """Every `self.ledger.<method>` LedgerSync actually calls."""
    src = open("prsm/node/ledger_sync.py").read()
    return set(re.findall(r"self\.ledger\.([a-zA-Z_]+)", src))


@pytest.fixture
async def dag_ledger(tmp_path):
    """A REAL DAGLedger — exactly what Node builds on the default config."""
    ledger = DAGLedger(str(tmp_path / "dag.db"), verify_signatures=False)
    await ledger.initialize()
    yield ledger


def _sync(ledger, identity, transport):
    return LedgerSync(
        identity=identity,
        gossip=MagicMock(),
        ledger=ledger,
        transport=transport,
        reconciliation_interval=300.0,
    )


class TestReconciliationOnTheDefaultLedger:
    async def test_reconciliation_sends_a_balance_request_on_the_default_dag_ledger(
        self, dag_ledger,
    ):
        """The node must actually emit a balance proof request.

        Pre-1419 this raised AttributeError: 'DAGLedger' object has no attribute
        'get_recent_tx_ids' — and the caller swallowed it, so reconciliation silently never ran on
        any default node.
        """
        identity = generate_node_identity("recon-node")
        await dag_ledger.create_wallet(identity.node_id, "recon-node")

        transport = MagicMock()
        transport.peers = {"peer-abc": MagicMock()}
        transport.send_to_peer = AsyncMock(return_value=True)

        sync = _sync(dag_ledger, identity, transport)

        # Call _run_reconciliation DIRECTLY (not the loop, which swallows).
        await sync._run_reconciliation()

        assert transport.send_to_peer.await_count == 1, (
            "no balance_request was sent — reconciliation did not run"
        )
        _peer, msg = transport.send_to_peer.await_args.args
        assert msg.payload["subtype"] == "balance_request"
        assert "recent_tx_ids" in msg.payload, (
            "balance_request carries no recent_tx_ids — the whole point of the proof"
        )
        assert isinstance(msg.payload["recent_tx_ids"], list)

    async def test_balance_request_from_a_peer_gets_answered_on_the_default_dag_ledger(
        self, dag_ledger,
    ):
        """The other half: a peer asks us for a balance proof and we must answer.

        Same missing method, so pre-1419 a default node ignored every peer's balance_request.
        """
        identity = generate_node_identity("responder")
        await dag_ledger.create_wallet(identity.node_id, "responder")

        transport = MagicMock()
        transport.peers = {}
        transport.send_to_peer = AsyncMock(return_value=True)
        sync = _sync(dag_ledger, identity, transport)

        req = MagicMock()
        req.payload = {
            "subtype": "balance_request",
            "request_id": "req-1",
            "requester_balance": 0.0,
            "recent_tx_ids": [],
        }
        peer = MagicMock()
        peer.peer_id = "peer-abc"

        await sync._handle_balance_request(req, peer)

        assert transport.send_to_peer.await_count == 1, (
            "peer's balance_request went unanswered"
        )
        _pid, resp = transport.send_to_peer.await_args.args
        assert resp.payload["subtype"] == "balance_response"
        assert resp.payload["request_id"] == "req-1"
        assert isinstance(resp.payload["recent_tx_ids"], list)

    async def test_recent_tx_ids_actually_reflects_the_dag(self, dag_ledger):
        """Not just "the method exists" — it must return this wallet's real transactions.

        A stub returning [] would pass the two tests above while still making the balance proof
        worthless, so assert against real DAG state.
        """
        identity = generate_node_identity("txn-node")
        await dag_ledger.create_wallet(identity.node_id, "txn-node")
        await dag_ledger.create_wallet("system", "PRSM Network")
        await dag_ledger.credit(
            identity.node_id, 5.0, TransactionType.REWARD, description="test credit",
        )

        ids = await dag_ledger.get_recent_tx_ids(identity.node_id, limit=20)

        assert isinstance(ids, list) and ids, "no tx ids returned for a wallet that has a credit"
        assert all(isinstance(i, str) for i in ids)
        history = await dag_ledger.get_transaction_history(identity.node_id, 20)
        assert ids == [t.tx_id for t in history], (
            "get_recent_tx_ids disagrees with the DAG's own transaction history"
        )


class TestLedgerInterfaceConformance:
    """The class-level fix: pin the INTERFACE, not the one method that happened to be missing."""

    def test_every_ledger_method_ledger_sync_calls_exists_on_both_ledger_impls(self):
        """`Node` can construct EITHER ledger (config.ledger_type), and hands either to LedgerSync.

        So every method LedgerSync calls on `self.ledger` must exist on BOTH. This is the test that
        would have caught sp1419 the day get_recent_tx_ids was added to LocalLedger alone — and it
        will catch the next one. LedgerSync's `ledger: LocalLedger` annotation does NOT protect us:
        the default node passes a DAGLedger, which is not a subclass of it.
        """
        called = _ledger_sync_ledger_methods()
        assert called, "parsed zero ledger methods — this test has gone blind, fix the regex"

        missing = {
            impl.__name__: sorted(m for m in called if not hasattr(impl, m))
            for impl in (DAGLedger, LocalLedger)
        }
        assert not any(missing.values()), (
            f"LedgerSync calls ledger methods that a ledger Node can actually build does not "
            f"have — that subsystem is silently dead on those nodes: {missing}"
        )
