"""Sprint 1043 — state-file-independent finalize-by-batch-id (harness resilience).

The sp1042 two-phase proof bridges deposit-commit → finalize via a local durable
state file. If that file is lost over the multi-day gap (different machine, wiped
~/.prsm, etc.), the committed batch — real escrow already locked on-chain — would
be un-finalizable through the harness. ``run_finalize_by_batch_id`` recovers it
straight from chain state: read the batch (provider/requester/status), finalize it
directly, verify it reached FINALIZED. Keyed on the authoritative on-chain status
(idempotent: already-FINALIZED is success; never finalizes a not-yet/voided/absent
batch).
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

ONE_FTNS = 10**18
REQ = "0x" + "a" * 40
PROV = "0x" + "b" * 40
BID = b"\xbb" * 32


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def _batch(status, total=ONE_FTNS):
    return {"provider": PROV, "requester": REQ, "merkle_root": b"\x01" * 32,
            "receipt_count": 1, "total_value_ftns": total, "commit_timestamp": 1700000000,
            "status": status, "escrow_pool_at_commit": "0x" + "e" * 40}


def test_finalize_by_batch_id_happy():
    from prsm.settlement.e2e_proof import run_finalize_by_batch_id
    contract = AsyncMock()
    contract.get_batch.return_value = _batch(1)          # PENDING
    contract.is_finalizable.return_value = True
    contract.get_batch_status.return_value = 2           # FINALIZED after finalize
    escrow = AsyncMock()
    escrow.ftns_balance_of.side_effect = [0, ONE_FTNS]
    escrow.balance_of.return_value = 0
    report = _run(run_finalize_by_batch_id(
        contract_client=contract, escrow_client=escrow, batch_id=BID))
    assert report["exists"] is True
    assert report["finalized"] is True and report["success"] is True
    assert report["recipient_ftns_delta"] == ONE_FTNS
    contract.finalize_batch.assert_awaited_once_with(BID)


def test_finalize_by_batch_id_already_finalized_is_idempotent():
    from prsm.settlement.e2e_proof import run_finalize_by_batch_id
    contract = AsyncMock()
    contract.get_batch.return_value = _batch(2)          # already FINALIZED
    escrow = AsyncMock()
    escrow.ftns_balance_of.return_value = ONE_FTNS
    escrow.balance_of.return_value = 0
    report = _run(run_finalize_by_batch_id(
        contract_client=contract, escrow_client=escrow, batch_id=BID))
    assert report["success"] is True and report["finalized"] is True
    contract.finalize_batch.assert_not_awaited()         # no double-finalize
    contract.is_finalizable.assert_not_awaited()


def test_finalize_by_batch_id_not_yet_finalizable():
    from prsm.settlement.e2e_proof import run_finalize_by_batch_id
    contract = AsyncMock()
    contract.get_batch.return_value = _batch(1)
    contract.is_finalizable.return_value = False
    contract.seconds_until_finalizable.return_value = 9000
    escrow = AsyncMock()
    report = _run(run_finalize_by_batch_id(
        contract_client=contract, escrow_client=escrow, batch_id=BID))
    assert report["finalizable"] is False and report["success"] is False
    assert report["seconds_until_finalizable"] == 9000
    contract.finalize_batch.assert_not_awaited()


def test_finalize_by_batch_id_nonexistent():
    from prsm.settlement.e2e_proof import run_finalize_by_batch_id
    contract = AsyncMock()
    contract.get_batch.return_value = _batch(0)          # NONEXISTENT
    escrow = AsyncMock()
    report = _run(run_finalize_by_batch_id(
        contract_client=contract, escrow_client=escrow, batch_id=BID))
    assert report["exists"] is False and report["success"] is False
    contract.finalize_batch.assert_not_awaited()


def test_finalize_by_batch_id_voided():
    from prsm.settlement.e2e_proof import run_finalize_by_batch_id
    contract = AsyncMock()
    contract.get_batch.return_value = _batch(3)          # VOIDED
    escrow = AsyncMock()
    report = _run(run_finalize_by_batch_id(
        contract_client=contract, escrow_client=escrow, batch_id=BID))
    assert report["success"] is False
    contract.finalize_batch.assert_not_awaited()


def test_finalize_by_batch_id_finalize_tx_fails():
    """finalize_batch reverts → status stays PENDING → success False (not a lie)."""
    from prsm.settlement.e2e_proof import run_finalize_by_batch_id
    contract = AsyncMock()
    contract.get_batch.return_value = _batch(1)
    contract.is_finalizable.return_value = True
    contract.get_batch_status.return_value = 1           # still PENDING after attempt
    escrow = AsyncMock()
    escrow.ftns_balance_of.side_effect = [0, 0]
    escrow.balance_of.return_value = ONE_FTNS
    report = _run(run_finalize_by_batch_id(
        contract_client=contract, escrow_client=escrow, batch_id=BID))
    assert report["success"] is False


def test_finalize_by_batch_id_propagates_revert():
    """A reverted finalize tx (OnChainRevertedError) propagates — it must NOT be
    swallowed into a false success."""
    from prsm.economy.web3.provenance_registry import OnChainRevertedError
    from prsm.settlement.e2e_proof import run_finalize_by_batch_id
    contract = AsyncMock()
    contract.get_batch.return_value = _batch(1)
    contract.is_finalizable.return_value = True
    contract.finalize_batch.side_effect = OnChainRevertedError("tx reverted: 0xdead")
    escrow = AsyncMock()
    escrow.ftns_balance_of.return_value = 0
    with pytest.raises(OnChainRevertedError):
        _run(run_finalize_by_batch_id(
            contract_client=contract, escrow_client=escrow, batch_id=BID))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
