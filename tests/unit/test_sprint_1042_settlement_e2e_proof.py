"""Sprint 1042 — settlement deposit→commit→finalize end-to-end proof harness.

The capstone that composes every settlement brick (EscrowPoolClient deposit +
BatchSettlementClient accumulate/commit/finalize + the durable state) into the
one provable loop that activates the rail. On Base Sepolia the challenge window
is 3 days, so a LIVE proof is inherently two-phase — deposit+commit now, finalize
after the window — bridged by the durable state_store (brick 2.5). These tests
drive the composition with mocks (no live chain): the receipt builder, the
deposit→commit phase, the finalize→verify phase (finalizable + not-yet), and the
mainnet hard-guard.
"""
from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock

import pytest

from prsm.settlement.client import CommittedBatch, FinalizedBatch
from prsm.settlement.accumulator import TriggerReason


ONE_FTNS = 10**18
REQ = "0x" + "a" * 40
PROV = "0x" + "b" * 40


def _committed(batch_id=b"\xbb" * 32, value=ONE_FTNS):
    return CommittedBatch(
        batch_id=batch_id, tx_hash="0xcommit", provider_address=PROV,
        requester_address=REQ, merkle_root=b"\xcd" * 32, receipt_count=1,
        total_value_ftns=value, commit_timestamp=1700000000,
        leaf_hashes=(b"\x01" * 32,), trigger_reason=TriggerReason.VALUE,
    )


# ── receipt builder ──────────────────────────────────────────────────────────


def test_build_signed_batched_receipt_produces_valid_leaf():
    from prsm.node.identity import generate_node_identity, verify_signature
    from prsm.settlement.e2e_proof import build_signed_batched_receipt
    from prsm.settlement.merkle import batched_receipt_to_leaf, hash_leaf
    from prsm.compute.shard_receipt import build_receipt_signing_payload

    ident = generate_node_identity(display_name="proof-settler")
    br = build_signed_batched_receipt(
        identity=ident, job_id="job-1", output_hash=hashlib.sha256(b"out").hexdigest(),
        executed_at_unix=1700000000, requester_address=REQ, provider_address=PROV,
        value_wei=ONE_FTNS, local_escrow_id="esc-1",
    )
    # builds a valid, hashable Merkle leaf (raises if the receipt is malformed)
    leaf_hash = hash_leaf(batched_receipt_to_leaf(br))
    assert len(leaf_hash) == 32
    # the signature is real and verifies against the embedded pubkey
    payload = build_receipt_signing_payload(
        job_id="job-1", shard_index=0,
        output_hash=hashlib.sha256(b"out").hexdigest(), executed_at_unix=1700000000)
    assert verify_signature(br.receipt.provider_pubkey_b64, payload, br.receipt.signature)
    assert br.requester_address == REQ and br.provider_address == PROV
    assert br.value_ftns == ONE_FTNS


# ── deposit + commit phase ───────────────────────────────────────────────────


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def test_run_deposit_and_commit_happy():
    from prsm.settlement.e2e_proof import run_deposit_and_commit
    escrow = AsyncMock()
    escrow.deposit.return_value = "0xdeposit"
    escrow.balance_of.return_value = ONE_FTNS
    settlement = AsyncMock()
    settlement.commit_ready_batches.return_value = [_committed()]
    report = _run(run_deposit_and_commit(
        escrow_client=escrow, settlement_client=settlement,
        batched_receipt=object(), deposit_amount_wei=ONE_FTNS, requester_address=REQ,
    ))
    escrow.deposit.assert_awaited_once_with(ONE_FTNS)
    settlement.accumulate.assert_awaited_once()
    settlement.commit_ready_batches.assert_awaited_once()
    assert report["deposit_tx"] == "0xdeposit"
    assert report["escrow_balance"] == ONE_FTNS
    assert report["batch_id"] == (b"\xbb" * 32).hex()


def test_run_deposit_and_commit_raises_if_escrow_underfunded():
    from prsm.settlement.e2e_proof import run_deposit_and_commit
    escrow = AsyncMock()
    escrow.deposit.return_value = "0xdeposit"
    escrow.balance_of.return_value = 0          # deposit didn't land
    settlement = AsyncMock()
    with pytest.raises(RuntimeError, match="escrow"):
        _run(run_deposit_and_commit(
            escrow_client=escrow, settlement_client=settlement,
            batched_receipt=object(), deposit_amount_wei=ONE_FTNS, requester_address=REQ,
        ))
    settlement.commit_ready_batches.assert_not_awaited()


def test_run_deposit_and_commit_raises_if_not_committed():
    from prsm.settlement.e2e_proof import run_deposit_and_commit
    escrow = AsyncMock()
    escrow.deposit.return_value = "0xdeposit"
    escrow.balance_of.return_value = ONE_FTNS
    settlement = AsyncMock()
    settlement.commit_ready_batches.return_value = []   # nothing committed
    with pytest.raises(RuntimeError, match="commit"):
        _run(run_deposit_and_commit(
            escrow_client=escrow, settlement_client=settlement,
            batched_receipt=object(), deposit_amount_wei=ONE_FTNS, requester_address=REQ,
        ))


# ── finalize + verify phase ──────────────────────────────────────────────────


def test_run_finalize_and_verify_not_yet_finalizable():
    from unittest.mock import MagicMock
    from prsm.settlement.e2e_proof import run_finalize_and_verify
    contract = AsyncMock()
    contract.is_finalizable.return_value = False
    contract.get_batch_status.return_value = 1          # PENDING
    contract.seconds_until_finalizable.return_value = 12345
    settlement = AsyncMock()
    settlement.is_finalized_locally = MagicMock(return_value=False)  # sync method
    escrow = AsyncMock()
    report = _run(run_finalize_and_verify(
        settlement_client=settlement, contract_client=contract, escrow_client=escrow,
        batch_id=b"\xbb" * 32, requester_address=REQ, recipient_address=PROV,
    ))
    assert report["finalizable"] is False
    assert report["success"] is False
    assert report["seconds_until_finalizable"] == 12345
    settlement.finalize_ready_batches.assert_not_awaited()   # never finalize early


def test_run_finalize_and_verify_happy_value_transfer():
    from unittest.mock import MagicMock
    from prsm.settlement.e2e_proof import run_finalize_and_verify
    bid = b"\xbb" * 32
    contract = AsyncMock()
    contract.is_finalizable.return_value = True
    # PENDING on the pre-check, FINALIZED after finalize (models the on-chain transition)
    contract.get_batch_status.side_effect = [1, 2]
    settlement = AsyncMock()
    settlement.is_finalized_locally = MagicMock(return_value=False)
    settlement.finalize_ready_batches.return_value = [
        FinalizedBatch(batch_id=bid, tx_submitted=True)]
    escrow = AsyncMock()
    # recipient WALLET FTNS rises by the settled amount; requester escrow drains
    escrow.ftns_balance_of.side_effect = [0, ONE_FTNS]      # before, after (rose)
    escrow.balance_of.return_value = 0                       # requester escrow drained
    report = _run(run_finalize_and_verify(
        settlement_client=settlement, contract_client=contract, escrow_client=escrow,
        batch_id=bid, requester_address=REQ, recipient_address=PROV,
        settle_confirm_delay_s=0,
    ))
    assert report["finalizable"] is True
    assert report["finalized"] is True
    assert report["recipient_ftns_delta"] == ONE_FTNS
    assert report["requester_escrow_after"] == 0
    assert report["success"] is True
    settlement.finalize_ready_batches.assert_awaited_once()


def test_run_finalize_and_verify_success_keyed_on_chain_status_not_wallet_delta():
    """sp1042 review #1/#2 — success must follow the on-chain FINALIZED status,
    not a wallet delta (which is cross-attributed across batches / self-pay)."""
    from unittest.mock import MagicMock
    from prsm.settlement.e2e_proof import run_finalize_and_verify
    bid = b"\xbb" * 32
    contract = AsyncMock()
    contract.is_finalizable.return_value = True
    contract.get_batch_status.side_effect = [1, 2]          # PENDING -> FINALIZED
    settlement = AsyncMock()
    settlement.is_finalized_locally = MagicMock(return_value=False)
    settlement.finalize_ready_batches.return_value = [
        FinalizedBatch(batch_id=bid, tx_submitted=True)]
    escrow = AsyncMock()
    escrow.ftns_balance_of.return_value = 0   # wallet never rises (delta 0)
    escrow.balance_of.return_value = 0
    report = _run(run_finalize_and_verify(
        settlement_client=settlement, contract_client=contract, escrow_client=escrow,
        batch_id=bid, requester_address=REQ, recipient_address=PROV,
        settle_confirm_attempts=2, settle_confirm_delay_s=0,
    ))
    assert report["success"] is True   # FINALIZED on-chain even with zero wallet delta


def test_run_finalize_and_verify_idempotent_when_already_finalized():
    """sp1042 review #5/#9 — a batch already finalized (by a sibling's bulk
    finalize, a 3rd party, or a prior run) is SUCCESS, never re-finalized or
    falsely reported 'not finalizable yet'."""
    from unittest.mock import MagicMock
    from prsm.settlement.e2e_proof import run_finalize_and_verify
    bid = b"\xbb" * 32
    contract = AsyncMock()
    settlement = AsyncMock()
    settlement.is_finalized_locally = MagicMock(return_value=True)   # already done
    escrow = AsyncMock()
    escrow.ftns_balance_of.return_value = ONE_FTNS
    escrow.balance_of.return_value = 0
    report = _run(run_finalize_and_verify(
        settlement_client=settlement, contract_client=contract, escrow_client=escrow,
        batch_id=bid, requester_address=REQ, recipient_address=PROV,
    ))
    assert report["success"] is True and report["finalized"] is True
    settlement.finalize_ready_batches.assert_not_awaited()  # no double-finalize
    contract.is_finalizable.assert_not_awaited()            # short-circuited


def test_run_finalize_and_verify_polls_past_stale_rpc_read():
    """sp1045 — a finalize tx mines but the RPC read replica lags: the first
    post-finalize balance read is STALE (pre-settle), a later poll sees the
    settled balance. The reported delta must reflect the settled state, not the
    stale read (the bug the live Sepolia proof exposed)."""
    from unittest.mock import MagicMock
    from prsm.settlement.e2e_proof import run_finalize_and_verify
    bid = b"\xbb" * 32
    contract = AsyncMock()
    contract.is_finalizable.return_value = True
    contract.get_batch_status.side_effect = [1, 2]
    settlement = AsyncMock()
    settlement.is_finalized_locally = MagicMock(return_value=False)
    settlement.finalize_ready_batches.return_value = [
        FinalizedBatch(batch_id=bid, tx_submitted=True)]
    escrow = AsyncMock()
    # before=0, then a STALE read (still 0), then the settled read (rose)
    escrow.ftns_balance_of.side_effect = [0, 0, ONE_FTNS]
    escrow.balance_of.return_value = 0
    report = _run(run_finalize_and_verify(
        settlement_client=settlement, contract_client=contract, escrow_client=escrow,
        batch_id=bid, requester_address=REQ, recipient_address=PROV,
        settle_confirm_attempts=4, settle_confirm_delay_s=0,
    ))
    assert report["success"] is True
    assert report["recipient_ftns_delta"] == ONE_FTNS   # polled past the stale read


def test_run_finalize_and_verify_reports_failure_if_not_finalized_onchain():
    from unittest.mock import MagicMock
    from prsm.settlement.e2e_proof import run_finalize_and_verify
    bid = b"\xbb" * 32
    contract = AsyncMock()
    contract.is_finalizable.return_value = True
    contract.get_batch_status.return_value = 1   # stays PENDING (finalize tx failed)
    settlement = AsyncMock()
    settlement.is_finalized_locally = MagicMock(return_value=False)
    settlement.finalize_ready_batches.return_value = [
        FinalizedBatch(batch_id=bid, tx_submitted=False)]  # tx failed
    escrow = AsyncMock()
    escrow.ftns_balance_of.side_effect = [0, 0]
    escrow.balance_of.return_value = ONE_FTNS
    report = _run(run_finalize_and_verify(
        settlement_client=settlement, contract_client=contract, escrow_client=escrow,
        batch_id=bid, requester_address=REQ, recipient_address=PROV,
    ))
    assert report["success"] is False


# ── mainnet hard-guard ───────────────────────────────────────────────────────


def test_mainnet_guard_refuses_8453_without_ack():
    from prsm.settlement.e2e_proof import assert_chain_allowed
    with pytest.raises(SystemExit):
        assert_chain_allowed(chain_id=8453, mainnet_acknowledged=False)


def test_mainnet_guard_allows_8453_with_ack():
    from prsm.settlement.e2e_proof import assert_chain_allowed
    assert_chain_allowed(chain_id=8453, mainnet_acknowledged=True)  # no raise


def test_mainnet_guard_allows_testnet_without_ack():
    from prsm.settlement.e2e_proof import assert_chain_allowed
    assert_chain_allowed(chain_id=84532, mainnet_acknowledged=False)  # no raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
