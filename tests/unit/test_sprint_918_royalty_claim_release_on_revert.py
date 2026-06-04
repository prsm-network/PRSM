"""Sprint 918 — release the sp911 royalty idempotency claim on definitive failure.

sp911 made content-royalty dispatch idempotent by atomically CLAIMING a
``royalty_dispatch_key(settlement_key, cid)`` nonce BEFORE the on-chain tx, so a
re-delivery of the same settlement cannot double-pay. But the claim was never
RELEASED on a definitive non-payment outcome: on ``OnChainRevertedError`` (the
chain rolled the tx back atomically — no payment) or ``BroadcastFailedError``
(the tx never reached the network), the claim stayed set, so a legitimate retry
hit ``skipped_already_dispatched`` → the royalty was PERMANENTLY LOST. The code
even documents "safe to retry / fall back" for these cases, which the held claim
made impossible.

sp918 fixes it: a ``release_nonce`` ledger primitive + a pure
``keys_to_release(results, settlement_key)`` helper that returns the claim keys
SAFE to release — ONLY ``reverted`` (atomic rollback) + ``failed``
(BroadcastFailedError, never sent). It MUST NOT release ``pending`` (the tx is
in the mempool and may still confirm — releasing then re-dispatching would
DOUBLE-PAY), ``sent``, ``error`` (unknown/generic exception — fail-safe), or
``skipped_*``. The dispatch is split so ``failed`` means BroadcastFailedError
specifically (definitively never sent) vs ``error`` for an unknown exception.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.economy.onchain_content_royalty import (
    dispatch_content_access_royalties,
    royalty_dispatch_key,
    keys_to_release,
)
from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError,
    OnChainPendingError,
    OnChainRevertedError,
)


def _record(cid, h="ab" * 32):
    r = MagicMock()
    r.content_hash = h
    # sp996 — on-chain dispatch keys on provenance_hash (the registry key), not
    # content_hash. Mirror the registered hash (else MagicMock auto-attr →
    # skipped_unregistered).
    r.provenance_hash = h
    r.royalty_rate = 0.05
    return r


def _index(recs):
    idx = MagicMock()
    idx.lookup = lambda cid: recs.get(cid)
    return idx


def _client(returns=None, side_effect=None):
    c = MagicMock()
    if side_effect is not None:
        c.distribute_royalty = MagicMock(side_effect=side_effect)
    else:
        c.distribute_royalty = MagicMock(return_value=returns or ("0xtx", "confirmed"))
    return c


def _dispatch(client, claimed, settle="settle-1", cids=("c1",)):
    recs = {c: _record(c) for c in cids}
    return dispatch_content_access_royalties(
        shards=list(cids), content_index=_index(recs), royalty_client=client,
        serving_node_address="0xop", gross_per_shard_wei=1000,
        settlement_key=settle, claim_fn=lambda k: claimed.get(k, False),
    )


# ── dispatch status split: BroadcastFailedError vs generic ───────────────


def test_broadcast_failed_is_status_failed():
    out = _dispatch(_client(side_effect=BroadcastFailedError("never sent")),
                    {royalty_dispatch_key("settle-1", "c1"): True})
    assert out[0].status == "failed"


def test_generic_exception_is_status_error_not_failed():
    # An unknown exception is NOT a definitively-never-sent failure → distinct
    # 'error' status so its claim is NOT released (fail-safe vs double-pay).
    out = _dispatch(_client(side_effect=RuntimeError("who knows")),
                    {royalty_dispatch_key("settle-1", "c1"): True})
    assert out[0].status == "error"


def test_reverted_is_status_reverted():
    out = _dispatch(_client(side_effect=OnChainRevertedError("revert")),
                    {royalty_dispatch_key("settle-1", "c1"): True})
    assert out[0].status == "reverted"


# ── keys_to_release — the safety-critical selector ───────────────────────


def _result(cid, status):
    return MagicMock(cid=cid, status=status)


def test_keys_to_release_includes_reverted_and_failed():
    results = [_result("c1", "reverted"), _result("c2", "failed")]
    keys = set(keys_to_release(results, "s"))
    assert keys == {royalty_dispatch_key("s", "c1"), royalty_dispatch_key("s", "c2")}


def test_keys_to_release_excludes_pending_sent_error_skipped():
    # CRITICAL: a pending tx is in the mempool and may still confirm — its
    # claim MUST be kept (releasing + re-dispatching would double-pay).
    results = [
        _result("c1", "pending"),
        _result("c2", "sent"),
        _result("c3", "error"),
        _result("c4", "skipped_already_dispatched"),
        _result("c5", "skipped_no_record"),
    ]
    assert keys_to_release(results, "s") == []


def test_keys_to_release_none_settlement_key_is_empty():
    assert keys_to_release([_result("c1", "reverted")], None) == []


# ── release_nonce on both ledger backends ────────────────────────────────


@pytest.mark.asyncio
async def test_local_ledger_release_nonce_frees_the_claim():
    from prsm.node.local_ledger import LocalLedger
    led = LocalLedger(":memory:")
    await led.initialize()
    assert await led.record_nonce("k1", "op") is True
    assert await led.record_nonce("k1", "op") is False   # claimed
    assert await led.release_nonce("k1") is True          # released
    assert await led.record_nonce("k1", "op") is True     # re-claimable
    # releasing an absent nonce is a harmless no-op
    assert await led.release_nonce("nope") is False


@pytest.mark.asyncio
async def test_dag_ledger_release_nonce_frees_the_claim():
    from prsm.node.dag_ledger import DAGLedger
    led = DAGLedger(db_path=":memory:")
    await led.initialize()
    assert await led.record_nonce("k1", "op") is True
    assert await led.record_nonce("k1", "op") is False
    assert await led.release_nonce("k1") is True
    assert await led.record_nonce("k1", "op") is True
    if led._db is not None:
        await led._db.close()


# ── end-to-end: reverted royalty CAN be retried; pending CANNOT ──────────


@pytest.mark.asyncio
async def test_reverted_claim_released_allows_redispatch():
    """The sp918 fix: a reverted dispatch's claim is released, so a retry of the
    same settlement RE-DISPATCHES (instead of skipped_already_dispatched)."""
    from prsm.node.local_ledger import LocalLedger
    led = LocalLedger(":memory:")
    await led.initialize()
    settle, cid = "settle-1", "c1"
    key = royalty_dispatch_key(settle, cid)

    claimed = {key: await led.record_nonce(key, "op")}
    r1 = _dispatch(_client(side_effect=OnChainRevertedError("revert")), claimed,
                   settle=settle, cids=(cid,))
    assert r1[0].status == "reverted"

    for k in keys_to_release(r1, settle):
        await led.release_nonce(k)

    claimed2 = {key: await led.record_nonce(key, "op")}
    assert claimed2[key] is True   # claim was released → re-claimable
    client2 = _client(("0xtx", "confirmed"))
    r2 = _dispatch(client2, claimed2, settle=settle, cids=(cid,))
    assert r2[0].status == "sent"  # FIXED — re-dispatched, royalty not lost
    assert client2.distribute_royalty.call_count == 1


@pytest.mark.asyncio
async def test_pending_claim_not_released_blocks_redispatch_no_double_pay():
    """SAFETY: a PENDING dispatch's claim is NOT released, so a retry is
    skipped_already_dispatched and the in-flight tx is never re-broadcast."""
    from prsm.node.local_ledger import LocalLedger
    led = LocalLedger(":memory:")
    await led.initialize()
    settle, cid = "settle-2", "c1"
    key = royalty_dispatch_key(settle, cid)

    claimed = {key: await led.record_nonce(key, "op")}
    r1 = _dispatch(_client(side_effect=OnChainPendingError("unconfirmed", "0xpend")),
                   claimed, settle=settle, cids=(cid,))
    assert r1[0].status == "pending"

    released = keys_to_release(r1, settle)
    assert released == []                       # pending → NOT released
    for k in released:
        await led.release_nonce(k)

    claimed2 = {key: await led.record_nonce(key, "op")}
    assert claimed2[key] is False               # claim still held
    client2 = _client(("0xtx", "confirmed"))
    r2 = _dispatch(client2, claimed2, settle=settle, cids=(cid,))
    assert r2[0].status == "skipped_already_dispatched"
    assert client2.distribute_royalty.call_count == 0   # NO double-pay
