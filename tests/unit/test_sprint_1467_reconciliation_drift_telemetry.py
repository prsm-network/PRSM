"""Sprint 1467 — reconciliation telemetry alarms ONLY on real drift (to_wallet == self).

The DEFAULT ledger is per-node: each node stores only txs it is a wallet-party to, and cross-node
value moves through escrow. The old reconciliation compared a peer's ENTIRE recent-tx set against our
ledger, so the peer's own escrow-funding / third-party txs (which never involve us) were reported as a
permanent, benign ~20/cycle "not seen locally" — a healthy steady state that read as a standing
anomaly and buried any real drift.

sp1467: the responder sends only `directed_tx_ids` (txs it directed AT the requester, to_wallet ==
requester); the requester alarms only when one of THOSE is missing (a genuine incoming-payment drift,
incl. an sp1466-class stranding). A legacy peer (no directed_tx_ids) no longer raises the alarm.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.ledger_sync import LedgerSync

pytestmark = pytest.mark.asyncio


def _requester(node_id="me"):
    ledger = MagicMock()
    ledger.get_balance = AsyncMock(return_value=0.0)
    transport = MagicMock()
    transport.peers = {}
    transport.send_to_peer = AsyncMock(return_value=True)
    ls = LedgerSync(
        identity=MagicMock(node_id=node_id), gossip=MagicMock(),
        ledger=ledger, transport=transport, reconciliation_interval=300.0)
    return ls, ledger


def _resp(request_id, *, directed=None, legacy_recent=None):
    payload = {"subtype": "balance_response", "request_id": request_id, "responder_balance": 3.0}
    if directed is not None:
        payload["directed_tx_ids"] = directed
    if legacy_recent is not None:
        payload["recent_tx_ids"] = legacy_recent
    m = MagicMock()
    m.payload = payload
    return m


# ── Responder side: pre-filter to txs directed at the requester ────────────────

async def test_responder_sends_only_txs_directed_at_the_requester():
    ledger = MagicMock()
    ledger.get_balance = AsyncMock(return_value=5.0)

    def _tx(tid, to):
        return SimpleNamespace(tx_id=tid, to_wallet=to)

    # Responder's recent txs: one payment TO the requester + its own escrow-funding + a third-party.
    ledger.get_transaction_history = AsyncMock(return_value=[
        _tx("pay-to-requester", "requester-node"),
        _tx("my-escrow-funding", "escrow-abc"),
        _tx("pay-to-third-party", "other-node"),
    ])
    transport = MagicMock()
    transport.send_to_peer = AsyncMock(return_value=True)
    ls = LedgerSync(
        identity=MagicMock(node_id="responder"), gossip=MagicMock(),
        ledger=ledger, transport=transport)

    req = MagicMock()
    req.sender_id = "requester-node"
    req.payload = {"subtype": "balance_request", "request_id": "r1"}
    peer = MagicMock(peer_id="requester-node")

    await ls._handle_balance_request(req, peer)
    _pid, resp = transport.send_to_peer.await_args.args
    # ONLY the tx directed at the requester — NOT the responder's own escrow/third-party txs.
    assert resp.payload["directed_tx_ids"] == ["pay-to-requester"]


# ── Requester side: alarm ONLY on missing incoming payments ────────────────────

async def test_benign_no_directed_txs_raises_no_discrepancy():
    # The live steady state: the peer has nothing directed at us → empty list → NO alarm (the whole
    # point of the fix; previously this class produced a permanent ~20/cycle "not seen locally").
    ls, ledger = _requester()
    ledger.has_transaction = AsyncMock(return_value=False)
    rid = ls._register_outstanding_request()
    await ls._handle_balance_response(_resp(rid, directed=[]), MagicMock(peer_id="peer"))
    assert ls._discrepancies_found == 0
    assert ledger.has_transaction.await_count == 0        # nothing to check → no DB work


async def test_missing_incoming_payment_is_flagged_as_drift():
    # A payment the responder directed AT US that we DON'T have → real drift → 1 discrepancy.
    ls, ledger = _requester()
    ledger.has_transaction = AsyncMock(return_value=False)   # we don't have it
    rid = ls._register_outstanding_request()
    await ls._handle_balance_response(
        _resp(rid, directed=["owed-payment-1", "owed-payment-2"]), MagicMock(peer_id="peer"))
    assert ls._discrepancies_found == 1                   # incremented once (missing > 0)
    assert ledger.has_transaction.await_count == 2        # checked both directed ids


async def test_present_incoming_payment_is_not_flagged():
    # A payment directed at us that we DO have → not drift → no alarm.
    ls, ledger = _requester()
    ledger.has_transaction = AsyncMock(return_value=True)    # we have it
    rid = ls._register_outstanding_request()
    await ls._handle_balance_response(
        _resp(rid, directed=["already-have-it"]), MagicMock(peer_id="peer"))
    assert ls._discrepancies_found == 0


async def test_drift_check_ignores_has_seen_nonce_so_a_stranding_is_visible():
    # sp1466 stranding: nonce seen but tx not stored, for a payment directed at us → MUST surface as
    # drift (the old code's has_seen_nonce suppression would have HIDDEN it).
    ls, ledger = _requester()
    ledger.has_transaction = AsyncMock(return_value=False)
    ledger.has_seen_nonce = AsyncMock(return_value=True)    # nonce seen (stranded) — must NOT suppress
    rid = ls._register_outstanding_request()
    await ls._handle_balance_response(
        _resp(rid, directed=["stranded-payment"]), MagicMock(peer_id="peer"))
    assert ls._discrepancies_found == 1


async def test_legacy_peer_without_directed_tx_ids_does_not_alarm():
    # A pre-sp1467 responder sends only the unfiltered recent_tx_ids — we can't classify them, so we
    # do NOT alarm on that benign noise.
    ls, ledger = _requester()
    ledger.has_transaction = AsyncMock(return_value=False)
    rid = ls._register_outstanding_request()
    await ls._handle_balance_response(
        _resp(rid, legacy_recent=["a" * 40, "b" * 40]), MagicMock(peer_id="legacy-peer"))
    assert ls._discrepancies_found == 0
    assert ledger.has_transaction.await_count == 0        # legacy list never scanned


async def test_unsolicited_response_still_dropped_sp1428_preserved():
    ls, ledger = _requester()
    ledger.has_transaction = AsyncMock(return_value=False)
    # never registered this request_id → dropped before any ledger work.
    await ls._handle_balance_response(
        _resp("never-sent", directed=["x"] * 1000), MagicMock(peer_id="attacker"))
    assert ledger.has_transaction.await_count == 0
    assert ls._discrepancies_found == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
