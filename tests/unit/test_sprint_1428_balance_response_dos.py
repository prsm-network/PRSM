"""Sprint 1428 — the now-live reconciliation could be DoS'd by an unbounded balance_response.

sp1419 activated LedgerSync's reconciliation on every default node (it was AttributeError-dead
before). That turned on `_handle_balance_response`, which:
  - processes ANY peer's balance_response with NO request_id correlation (solicited or not), and
  - iterates `recent_tx_ids` with NO length cap, doing up to 2 awaited SQLite lookups per element
    on the single shared ledger connection.

A legitimate response carries at most 20 ids (the request path caps at `limit=20`). But a malicious
connected peer can send one unsolicited balance_response with millions of fabricated ids → millions
of serialized DB lookups that monopolize the ledger connection and starve concurrent
transfers/credits/nonce-claims — a resource-exhaustion DoS of the MONEY path, reachable on every
peer-connected node.

Fix: (1) drop responses whose request_id we never sent (uncorrelated/unsolicited), and (2) cap the
processed id count. Found by the sp1427 money-path adversarial audit (finding #5).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.ledger_sync import LedgerSync


def _sync():
    ledger = MagicMock()
    ledger.has_transaction = AsyncMock(return_value=True)   # "we've seen it" → cheap path
    ledger.has_seen_nonce = AsyncMock(return_value=True)
    ledger.get_balance = AsyncMock(return_value=0.0)
    ledger.get_recent_tx_ids = AsyncMock(return_value=[])
    transport = MagicMock()
    transport.peers = {}
    transport.send_to_peer = AsyncMock(return_value=True)
    return LedgerSync(
        identity=MagicMock(node_id="me"),
        gossip=MagicMock(),
        ledger=ledger,
        transport=transport,
        reconciliation_interval=300.0,
    ), ledger


def _response(request_id: str, tx_ids):
    msg = MagicMock()
    msg.payload = {
        "subtype": "balance_response",
        "request_id": request_id,
        "responder_balance": 1.0,
        # sp1467 — the drift-bearing field is now directed_tx_ids (txs the responder directed AT us).
        # The sp1428 protections (request_id correlation + length cap) apply to it unchanged.
        "directed_tx_ids": tx_ids,
    }
    return msg


class TestUnsolicitedResponseIsDropped:
    async def test_response_to_a_request_we_never_sent_is_ignored(self):
        sync, ledger = _sync()
        peer = MagicMock(peer_id="attacker")
        # No request was ever sent, so this request_id is unknown → drop without touching the ledger.
        await sync._handle_balance_response(
            _response("never-sent-id", ["a" * 40] * 1000), peer,
        )
        assert ledger.has_transaction.await_count == 0, (
            "an unsolicited balance_response was processed — an attacker can trigger the DB storm "
            "without us ever asking them anything"
        )


class TestOversizedListIsCapped:
    async def test_a_giant_tx_id_list_does_not_produce_unbounded_db_lookups(self):
        sync, ledger = _sync()
        peer = MagicMock(peer_id="attacker")
        rid = sync._register_outstanding_request()  # simulate having sent a request
        # A hostile response with a million ids...
        await sync._handle_balance_response(_response(rid, ["x" * 40] * 1_000_000), peer)
        # ...must not translate into a million DB lookups. The legit request path caps at 20;
        # allow generous headroom but assert it is BOUNDED, not linear in the attacker's input.
        assert ledger.has_transaction.await_count <= 256, (
            f"processed {ledger.has_transaction.await_count} ids — the balance_response loop is "
            f"still unbounded; a large message DoS's the ledger connection"
        )


class TestSolicitedResponseStillWorks:
    async def test_a_normal_solicited_response_is_processed(self):
        sync, ledger = _sync()
        ledger.has_transaction = AsyncMock(return_value=False)   # "not seen" → counts as missing
        ledger.has_seen_nonce = AsyncMock(return_value=False)
        peer = MagicMock(peer_id="honest")
        rid = sync._register_outstanding_request()
        await sync._handle_balance_response(_response(rid, ["a" * 40, "b" * 40, "c" * 40]), peer)
        assert ledger.has_transaction.await_count == 3, "a legitimate response must still reconcile"
        assert sync._discrepancies_found == 1

    async def test_a_request_id_is_single_use(self):
        """A consumed request_id must not let a second (replayed) response through."""
        sync, ledger = _sync()
        peer = MagicMock(peer_id="honest")
        rid = sync._register_outstanding_request()
        await sync._handle_balance_response(_response(rid, ["a" * 40]), peer)
        first = ledger.has_transaction.await_count
        # replay the same response
        await sync._handle_balance_response(_response(rid, ["a" * 40] * 500), peer)
        assert ledger.has_transaction.await_count == first, (
            "a request_id was reusable — a replayed response re-triggers processing"
        )


class TestOutstandingIdsAreBounded:
    def test_the_outstanding_id_set_cannot_grow_without_bound(self):
        """Even the tracking set must be bounded, or the fix trades one leak for another."""
        sync, _ = _sync()
        for _ in range(10_000):
            sync._register_outstanding_request()
        # implementation detail, but must be bounded well under the number registered
        assert len(sync._outstanding_request_ids) <= 1024
