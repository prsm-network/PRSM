"""Sprint 916 — pending-withdraw reconciler (closes the sp914 revert gap).

sp914 made a withdraw whose on-chain tx broadcasts-but-times-out return a
"pending" status WITHOUT refunding the off-chain debit (correct — the tx is in
the mempool and will likely confirm; refunding then would double-pay). But the
investigation found the dead-end: if that pending tx later REVERTS on-chain, the
off-chain debit (taken BEFORE broadcast) is NEVER refunded → the user
permanently loses the FTNS. The only existing reconciliation
(OnChainFTNSLedger._reconcile_pending_transactions) runs at startup and merely
updates tx STATUS — it takes no corrective action.

sp916 closes the loop: a PendingWithdrawStore records each pending withdraw
(job_id → wallet_id, amount, tx_hash); a reconciler polls the receipt and, on
revert, refunds the off-chain debit IDEMPOTENTLY (record_nonce atomic-claim per
sp898/911, so a reconciler restart / double-run cannot double-refund). A
confirmed tx resolves with no refund.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
# Import TestClient at MODULE TOP (collection time) so its httpx.Client base is
# bound to the REAL class — the autouse mock_http_requests conftest fixture
# patches httpx.Client per-test, and a TestClient imported under that patch
# would inherit the mock (the 541 withdraw tests import it at top for the same
# reason).
from fastapi.testclient import TestClient

from prsm.node.pending_withdraw_reconciler import (
    PendingWithdrawStore,
    PendingWithdrawReconciler,
    reconcile_pending_withdraws,
    resolve_pending_withdraw_reconciler_config_from_env,
)


# ── PendingWithdrawStore ─────────────────────────────────────────────────


def _rec(store, job_id="withdraw-1", wallet="w1", amount=10.0):
    store.record(job_id=job_id, wallet_id=wallet, amount=amount,
                 to_addr="0x" + "b" * 40, tx_hash="0x" + "d" * 64)


def test_record_then_unresolved():
    s = PendingWithdrawStore(persist_dir=None)
    _rec(s, "withdraw-1")
    _rec(s, "withdraw-2")
    jobs = {i.job_id for i in s.unresolved()}
    assert jobs == {"withdraw-1", "withdraw-2"}


def test_mark_resolved_drops_from_unresolved():
    s = PendingWithdrawStore(persist_dir=None)
    _rec(s, "withdraw-1")
    s.mark_resolved("withdraw-1", "confirmed")
    assert s.unresolved() == []


def test_record_same_job_id_is_idempotent():
    s = PendingWithdrawStore(persist_dir=None)
    _rec(s, "withdraw-1", amount=10.0)
    _rec(s, "withdraw-1", amount=10.0)
    assert len(s.unresolved()) == 1


def test_persistence_roundtrip(tmp_path):
    d = str(tmp_path / "pw")
    s1 = PendingWithdrawStore(persist_dir=d)
    _rec(s1, "withdraw-1", wallet="alice", amount=42.0)
    # A fresh store from the same dir must recover the unresolved intent.
    s2 = PendingWithdrawStore(persist_dir=d)
    got = s2.unresolved()
    assert len(got) == 1
    assert got[0].wallet_id == "alice"
    assert got[0].amount == 42.0


def test_resolved_state_persists(tmp_path):
    d = str(tmp_path / "pw")
    s1 = PendingWithdrawStore(persist_dir=d)
    _rec(s1, "withdraw-1")
    s1.mark_resolved("withdraw-1", "refunded")
    s2 = PendingWithdrawStore(persist_dir=d)
    assert s2.unresolved() == []   # resolved survives restart → no re-refund


def test_bounded_prunes_resolved():
    s = PendingWithdrawStore(persist_dir=None, max_entries=5)
    for i in range(20):
        _rec(s, f"withdraw-{i}")
        s.mark_resolved(f"withdraw-{i}", "confirmed")
    # Resolved entries are pruned so the store can't grow unbounded.
    assert len(s.all()) <= 5


# ── reconcile_pending_withdraws (pure core) ──────────────────────────────


@pytest.mark.asyncio
async def test_confirmed_resolves_without_refund():
    s = PendingWithdrawStore(persist_dir=None)
    _rec(s, "withdraw-1")
    refund = AsyncMock(return_value=True)
    out = await reconcile_pending_withdraws(
        s,
        get_receipt_status=AsyncMock(return_value="confirmed"),
        refund=refund,
    )
    assert out["confirmed"] == 1
    refund.assert_not_called()
    assert s.unresolved() == []


@pytest.mark.asyncio
async def test_reverted_refunds_and_resolves():
    s = PendingWithdrawStore(persist_dir=None)
    _rec(s, "withdraw-1", wallet="alice", amount=10.0)
    refund = AsyncMock(return_value=True)
    out = await reconcile_pending_withdraws(
        s,
        get_receipt_status=AsyncMock(return_value="reverted"),
        refund=refund,
    )
    assert out["refunded"] == 1
    refund.assert_awaited_once()
    intent = refund.await_args.args[0]
    assert intent.wallet_id == "alice" and intent.amount == 10.0
    assert s.unresolved() == []


@pytest.mark.asyncio
async def test_pending_left_unresolved():
    s = PendingWithdrawStore(persist_dir=None)
    _rec(s, "withdraw-1")
    refund = AsyncMock(return_value=True)
    out = await reconcile_pending_withdraws(
        s,
        get_receipt_status=AsyncMock(return_value="pending"),
        refund=refund,
    )
    assert out["still_pending"] == 1
    refund.assert_not_called()
    assert len(s.unresolved()) == 1   # retried on the next tick


@pytest.mark.asyncio
async def test_second_pass_does_not_double_refund():
    s = PendingWithdrawStore(persist_dir=None)
    _rec(s, "withdraw-1")
    refund = AsyncMock(return_value=True)
    poll = AsyncMock(return_value="reverted")
    await reconcile_pending_withdraws(s, get_receipt_status=poll, refund=refund)
    await reconcile_pending_withdraws(s, get_receipt_status=poll, refund=refund)
    refund.assert_awaited_once()   # resolved after pass 1 → not refunded twice


# ── PendingWithdrawReconciler.refund idempotency (sp1439: idempotency_key on credit) ──


@pytest.mark.asyncio
async def test_refund_is_idempotency_key_gated_against_double_credit():
    # sp1439 — idempotency moved from a record_nonce PRE-CLAIM (crash-unsafe: a crash
    # between the claim commit and the credit commit burned the claim and lost the refund
    # forever) INTO the credit itself via idempotency_key="withdraw-refund:{job_id}". Two
    # invocations (e.g. across a reconciler restart that lost the resolved mark) must credit
    # the wallet EXACTLY once — the real ledger dedups on the deterministic tx_id.
    local_ledger = MagicMock()
    credited = []          # (wallet, amount) actually applied
    seen_keys = set()

    async def credit(*, wallet_id, amount, tx_type, idempotency_key=None, description=""):
        if idempotency_key is not None and idempotency_key in seen_keys:
            return None    # exactly-once no-op on replay (mirrors the tx_id PRIMARY-KEY dedup)
        if idempotency_key is not None:
            seen_keys.add(idempotency_key)
        credited.append((wallet_id, float(amount)))
        return MagicMock()

    local_ledger.credit = AsyncMock(side_effect=credit)

    rec = PendingWithdrawReconciler(
        store=PendingWithdrawStore(persist_dir=None),
        ftns_ledger=MagicMock(),
        local_ledger=local_ledger,
    )
    intent = MagicMock(job_id="withdraw-1", wallet_id="alice", amount=10.0,
                       tx_hash="0xdead", outcome="reverted")
    first = await rec._refund(intent)
    second = await rec._refund(intent)
    # Both succeed (never a stranded refund), and the wallet is credited EXACTLY once.
    assert first is True and second is True
    assert credited == [("alice", 10.0)]
    # Both calls carried the per-job idempotency key that makes the real ledger exactly-once.
    assert seen_keys == {"withdraw-refund:withdraw-1"}


# ── env config ───────────────────────────────────────────────────────────


def test_env_config_defaults(monkeypatch):
    monkeypatch.delenv("PRSM_PENDING_WITHDRAW_RECONCILER_ENABLED", raising=False)
    monkeypatch.delenv("PRSM_PENDING_WITHDRAW_RECONCILER_INTERVAL_S", raising=False)
    enabled, interval = resolve_pending_withdraw_reconciler_config_from_env()
    assert isinstance(enabled, bool)
    assert interval >= 60.0   # clamped to a sane floor


def test_env_config_can_disable(monkeypatch):
    monkeypatch.setenv("PRSM_PENDING_WITHDRAW_RECONCILER_ENABLED", "0")
    enabled, _ = resolve_pending_withdraw_reconciler_config_from_env()
    assert enabled is False


# ── integration: /wallet/withdraw records the intent on pending ──────────


@pytest.mark.asyncio
async def test_withdraw_pending_records_intent_for_reconciler(tmp_path):
    """The seam between sp914 (pending status) and sp916 (reconciler): a
    withdraw whose on-chain broadcast is unconfirmed must record a reconcilable
    intent so a later revert can be refunded. A confirmed withdraw records
    nothing."""
    from prsm.node.dag_ledger import DAGLedger, TransactionType
    from prsm.node.api import create_api_app

    db = str(tmp_path / "ledger.db")
    led = DAGLedger(db_path=db)
    await led.initialize()
    await led.credit(wallet_id="alice", amount=10.0,
                     tx_type=TransactionType.WELCOME_GRANT, description="setup")

    def _ftns(status):
        f = MagicMock()
        f._connected_address = "0x" + "4a" * 20
        f.contract_address = "0x" + "52" * 20
        f.chain_id = 8453
        rec = MagicMock()
        rec.tx_hash = "0x" + "ab" * 32
        rec.status = status
        rec.block_number = None
        f.transfer = AsyncMock(return_value=rec)
        return f

    store = PendingWithdrawStore(persist_dir=None)
    node = MagicMock()
    node.ledger = led
    node.ftns_ledger = _ftns("pending")
    node.identity = MagicMock()
    node.identity.node_id = "alice"
    node._pending_withdraw_store = store

    client = TestClient(create_api_app(node, enable_security=False))
    r = client.post("/wallet/withdraw", json={
        "amount_ftns": 1.5, "to_eth_address": "0x" + "cb" * 20,
    })
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "pending"

    unresolved = store.unresolved()
    assert len(unresolved) == 1
    assert unresolved[0].wallet_id == "alice"
    assert unresolved[0].amount == 1.5
    assert unresolved[0].tx_hash == "0x" + "ab" * 32
    await led._db.close()


@pytest.mark.asyncio
async def test_withdraw_confirmed_records_no_intent(tmp_path):
    from prsm.node.dag_ledger import DAGLedger, TransactionType
    from prsm.node.api import create_api_app

    db = str(tmp_path / "ledger.db")
    led = DAGLedger(db_path=db)
    await led.initialize()
    await led.credit(wallet_id="alice", amount=10.0,
                     tx_type=TransactionType.WELCOME_GRANT, description="setup")

    f = MagicMock()
    f._connected_address = "0x" + "4a" * 20
    f.contract_address = "0x" + "52" * 20
    f.chain_id = 8453
    rec = MagicMock(tx_hash="0x" + "ab" * 32, status="confirmed", block_number=1)
    f.transfer = AsyncMock(return_value=rec)

    store = PendingWithdrawStore(persist_dir=None)
    node = MagicMock()
    node.ledger = led
    node.ftns_ledger = f
    node.identity = MagicMock()
    node.identity.node_id = "alice"
    node._pending_withdraw_store = store

    client = TestClient(create_api_app(node, enable_security=False))
    r = client.post("/wallet/withdraw", json={
        "amount_ftns": 1.5, "to_eth_address": "0x" + "cb" * 20,
    })
    assert r.status_code == 200
    assert r.json()["status"] == "confirmed"
    assert store.unresolved() == []   # confirmed → nothing to reconcile
    await led._db.close()
