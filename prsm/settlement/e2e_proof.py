"""Sprint 1042 — settlement deposit→commit→finalize end-to-end proof (brick 3b).

The capstone that composes the whole on-chain settlement rail into the one
provable loop that ACTIVATES it: deposit FTNS into EscrowPool → accumulate a
signed receipt → commit the batch on-chain → (after the challenge window)
finalize → EscrowPool.settleFromRequester moves the FTNS from the requester's
escrow to the provider's wallet.

On Base Sepolia the challenge window is 3 days (same as mainnet — testnet does
NOT shorten it), so a LIVE proof is inherently TWO-PHASE:
  1. ``run_deposit_and_commit`` — fund + commit now (the batch + its on-chain
     batchId are persisted by the durable state_store, brick 2.5).
  2. ``run_finalize_and_verify`` — ~3 days later, finalize + assert the recipient
     wallet rose and the requester escrow drained. The client rehydrates the
     tracked batch from disk, so the two phases survive a restart between them.

A single-key run (requester == provider == recipient) is a valid MECHANICAL proof
(funds leave self's escrow, return to self's wallet — net zero, but the full
on-chain path executes). A two-key run proves real cross-party value transfer.

These helpers are pure composition over already-constructed clients so they can be
exercised with mocks (no live chain) and reused by the operator script.
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Base chain ids — the live proof must default to testnet and refuse mainnet
# unless explicitly acknowledged (real money).
BASE_MAINNET_CHAIN_ID = 8453
BASE_SEPOLIA_CHAIN_ID = 84532


def assert_chain_allowed(*, chain_id: int, mainnet_acknowledged: bool) -> None:
    """Hard safety gate: this harness sends real settlement transactions, so it
    must refuse Base mainnet (chainId 8453) unless the operator explicitly
    acknowledged it. Exits the process (SystemExit) rather than returning, so a
    misconfigured PRSM_NETWORK can never silently hit mainnet."""
    if chain_id == BASE_MAINNET_CHAIN_ID and not mainnet_acknowledged:
        raise SystemExit(
            "REFUSING to run the settlement e2e proof on Base MAINNET "
            "(chainId 8453) without explicit acknowledgement. This sends real "
            "settlement transactions. Pass --i-understand-mainnet only if you "
            "have verified the RPC, contract addresses and signer key. The "
            "default + intended target is Base Sepolia (chainId 84532)."
        )


def build_signed_batched_receipt(
    *,
    identity: Any,
    job_id: str,
    output_hash: str,
    executed_at_unix: int,
    requester_address: str,
    provider_address: str,
    value_wei: int,
    local_escrow_id: str,
    shard_index: int = 0,
    tee_attestation: Any = None,
    tee_type: Any = None,
):
    """Build a challenge-defensible signed ``BatchedReceipt``: sign the canonical
    shard payload with ``identity`` (the same preimage the on-chain
    INVALID_SIGNATURE handler checks) and wrap it with the settlement metadata.

    Happy-path commit/finalize do NOT verify the signature on-chain (only a
    challenge does), but we sign honestly so the committed batch is defensible if
    challenged within the window."""
    from prsm.compute.shard_receipt import (
        ShardExecutionReceipt,
        build_receipt_signing_payload,
        tee_attestation_audit_dict,
    )
    from prsm.settlement.accumulator import BatchedReceipt

    payload = build_receipt_signing_payload(
        job_id=job_id, shard_index=shard_index,
        output_hash=output_hash, executed_at_unix=executed_at_unix,
    )
    signature_b64 = identity.sign(payload)
    receipt = ShardExecutionReceipt(
        job_id=job_id,
        shard_index=shard_index,
        provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64,
        output_hash=output_hash,
        executed_at_unix=executed_at_unix,
        signature=signature_b64,
        # sp1238 — optional audit-only attestation carriage (default None →
        # unchanged; NOT in the signing payload / leaf, so zero consensus impact).
        tee_attestation=tee_attestation_audit_dict(tee_attestation, tee_type),
    )
    return BatchedReceipt(
        receipt=receipt,
        requester_address=requester_address,
        provider_address=provider_address,
        value_ftns=int(value_wei),
        local_escrow_id=local_escrow_id,
    )


async def run_deposit_and_commit(
    *,
    escrow_client: Any,
    settlement_client: Any,
    batched_receipt: Any,
    deposit_amount_wei: int,
    requester_address: str,
    deposit_confirm_attempts: int = 6,
    deposit_confirm_delay_s: float = 2.0,
) -> dict:
    """Phase 1: fund the requester's escrow, accumulate the signed receipt, and
    commit the batch on-chain. Returns a report dict.

    sp1048 — IDEMPOTENT + lag-tolerant deposit. A prior run can deposit on-chain
    then fail before committing (e.g. an RPC-replica-lag false 'underfunded' on the
    post-deposit read); re-depositing would DOUBLE-fund. So: if the requester
    escrow already covers the amount, SKIP the deposit and go straight to commit.
    Otherwise deposit, then POLL the escrow balance past replica lag before
    concluding underfunded (a single immediate read can hit a stale replica that
    hasn't seen the just-mined deposit)."""
    amount = int(deposit_amount_wei)
    escrow_balance = int(await escrow_client.balance_of(requester_address))
    if escrow_balance >= amount:
        # already funded by a prior (commit-less) run — don't double-deposit
        deposit_tx = f"skipped:escrow-already-funded({escrow_balance})"
    else:
        deposit_tx = await escrow_client.deposit(amount)
        import asyncio as _asyncio
        for attempt in range(max(1, deposit_confirm_attempts)):
            escrow_balance = int(await escrow_client.balance_of(requester_address))
            if escrow_balance >= amount:
                break
            if attempt < deposit_confirm_attempts - 1 and deposit_confirm_delay_s > 0:
                await _asyncio.sleep(deposit_confirm_delay_s)
        if escrow_balance < amount:
            raise RuntimeError(
                f"escrow underfunded after deposit: balance {escrow_balance} < "
                f"deposited {amount} (tx {deposit_tx}) — even after polling past "
                f"RPC lag; check the deposit tx on-chain before retrying"
            )

    await settlement_client.accumulate(batched_receipt)
    committed = await settlement_client.commit_ready_batches()
    if not committed:
        raise RuntimeError(
            "commit_ready_batches returned no batch — the accumulator threshold "
            "may not have been crossed, or the commit was quarantined/failed"
        )
    batch = committed[0]
    # sp1042 review #11 — the escrow must cover the COMMITTED batch value, else
    # finalize will revert (settleFromRequester InsufficientBalance). The script
    # sets deposit == batch value, but guard the reusable helper anyway.
    if escrow_balance < int(batch.total_value_ftns):
        raise RuntimeError(
            f"escrow balance {escrow_balance} < committed batch value "
            f"{batch.total_value_ftns} — finalize would revert; deposit more"
        )
    report = {
        "deposit_tx": deposit_tx,
        "escrow_balance": escrow_balance,
        "batch_id": batch.batch_id.hex(),
        "merkle_root": batch.merkle_root.hex(),
        "commit_timestamp": batch.commit_timestamp,
        "total_value_ftns": batch.total_value_ftns,
    }
    logger.info("deposit+commit done: batch %s committed (value %s wei)",
                report["batch_id"][:12], report["total_value_ftns"])
    return report


async def run_two_party_deposit_and_commit(
    *,
    escrow_client: Any,
    settlement_client: Any,
    verifier: Any,
    payment_authorization: dict,
    request_fields: dict,
    provider_address: str,
    identity: Any,
    job_id: str,
    output_hash: str,
    executed_at_unix: int,
    value_wei: int,
    local_escrow_id: str,
    deposit_amount_wei: int,
    deposit_confirm_attempts: int = 6,
    deposit_confirm_delay_s: float = 2.0,
) -> dict:
    """Sprint 1058 (brick 5) — the TWO-PARTY analogue of ``run_deposit_and_commit``.

    Requester A signs a PaymentAuthorization; provider B (this caller) verifies it
    FAIL-CLOSED (proving A authorized the spend for THIS request), then funds A's
    escrow, builds a batched receipt naming A as requester + B as provider, and
    commits. finalize (phase 2, reuse ``run_finalize_and_verify`` /
    ``run_finalize_by_batch_id``) then draws from A's escrow into B's wallet — real
    cross-party value transfer.

    A rejected authorization raises ``AuthorizationRejected`` BEFORE any deposit or
    commit, so a forged/expired/over-spend auth never moves money. The verified
    requester address (not a caller-supplied one) is what the receipt + deposit use.
    """
    from prsm.settlement.payment_authorization import (
        canonical_request_hash, inference_request_fields,
    )

    request_hash = canonical_request_hash(inference_request_fields(**request_fields))
    payload = payment_authorization["payload"]
    signature = payment_authorization["signature"]
    # FAIL-CLOSED verify — raises AuthorizationRejected before touching the chain.
    requester = await verifier.verify(
        payload, signature, request_hash=request_hash, quoted_price_wei=value_wei,
    )

    batched = build_signed_batched_receipt(
        identity=identity, job_id=job_id, output_hash=output_hash,
        executed_at_unix=executed_at_unix, requester_address=requester,
        provider_address=provider_address, value_wei=value_wei,
        local_escrow_id=local_escrow_id,
    )
    report = await run_deposit_and_commit(
        escrow_client=escrow_client, settlement_client=settlement_client,
        batched_receipt=batched, deposit_amount_wei=deposit_amount_wei,
        requester_address=requester,
        deposit_confirm_attempts=deposit_confirm_attempts,
        deposit_confirm_delay_s=deposit_confirm_delay_s,
    )
    report["requester_address"] = requester
    report["provider_address"] = provider_address
    logger.info("two-party deposit+commit: batch %s requester=%s provider=%s",
                report["batch_id"][:12], requester, provider_address)
    return report


async def run_finalize_and_verify(
    *,
    settlement_client: Any,
    contract_client: Any,
    escrow_client: Any,
    batch_id: bytes,
    requester_address: str,
    recipient_address: str,
    settle_confirm_attempts: int = 6,
    settle_confirm_delay_s: float = 2.0,
) -> dict:
    """Phase 2: finalize this batch (if the challenge window elapsed) and verify it
    SETTLED. ``success`` is keyed on the AUTHORITATIVE on-chain batch status
    (FINALIZED), NOT on a wallet delta (sp1042 review #1/#2/#5/#9): with one wallet
    receiving several batches — e.g. self-pay, or multiple tracked batches — a
    global wallet diff is cross-attributed and would both inflate one batch's delta
    and falsely fail the others. Idempotent + batch-scoped: an already-finalized
    batch (settled here earlier in the loop, by a third party, or in a prior run)
    counts as SUCCESS, not "not finalizable yet". Never finalizes early.

    The recipient wallet delta + requester escrow are still reported, as
    informational evidence (exact per-batch attribution isn't possible from a
    shared wallet, so they don't gate success)."""
    if await _is_finalized(settlement_client, contract_client, batch_id):
        recipient_now = int(await escrow_client.ftns_balance_of(recipient_address))
        escrow_now = int(await escrow_client.balance_of(requester_address))
        return {
            "finalizable": True, "finalized": True, "finalize_submitted": False,
            "recipient_ftns_before": recipient_now, "recipient_ftns_after": recipient_now,
            "recipient_ftns_delta": 0, "requester_escrow_after": escrow_now,
            "success": True, "note": "already finalized",
        }

    finalizable = bool(await contract_client.is_finalizable(batch_id))
    if not finalizable:
        seconds_left = await _seconds_until_finalizable(contract_client, batch_id)
        return {
            "finalizable": False, "finalized": False, "success": False,
            "seconds_until_finalizable": seconds_left,
        }

    recipient_before = int(await escrow_client.ftns_balance_of(recipient_address))
    # finalize_ready_batches finalizes ALL currently-eligible tracked batches +
    # persists _finalized_ids; idempotent across this per-batch loop.
    await settlement_client.finalize_ready_batches()
    finalized = await _is_finalized(settlement_client, contract_client, batch_id)

    # sp1045 — settlement-confirmation poll. A fresh finalize tx mines, but RPC
    # read REPLICAS lag, so an immediate balance read can show the PRE-settle state
    # (escrow not yet drained / wallet unchanged) even though settleFromRequester
    # landed. The success verdict already follows the on-chain FINALIZED status, so
    # this only SHARPENS the advisory balances — avoiding alarming-but-wrong
    # "delta 0 / escrow not drained" output (which would be terrifying mid mainnet
    # ceremony). Poll until the recipient wallet reflects the settlement, bounded.
    recipient_after = int(await escrow_client.ftns_balance_of(recipient_address))
    requester_escrow_after = int(await escrow_client.balance_of(requester_address))
    if finalized:
        import asyncio as _asyncio
        for attempt in range(max(1, settle_confirm_attempts)):
            if recipient_after > recipient_before:
                break   # settlement reflected on the read replica
            if attempt < settle_confirm_attempts - 1 and settle_confirm_delay_s > 0:
                await _asyncio.sleep(settle_confirm_delay_s)
            recipient_after = int(await escrow_client.ftns_balance_of(recipient_address))
            requester_escrow_after = int(await escrow_client.balance_of(requester_address))

    report = {
        "finalizable": True,
        "finalized": finalized,
        "finalize_submitted": finalized,
        "recipient_ftns_before": recipient_before,
        "recipient_ftns_after": recipient_after,
        "recipient_ftns_delta": recipient_after - recipient_before,
        "requester_escrow_after": requester_escrow_after,
        "success": bool(finalized),
    }
    logger.info("finalize+verify: batch %s success=%s escrow_after=%s",
                batch_id.hex()[:12], finalized, requester_escrow_after)
    return report


async def run_finalize_by_batch_id(
    *,
    contract_client: Any,
    escrow_client: Any,
    batch_id: bytes,
) -> dict:
    """sp1043 — STATE-FILE-INDEPENDENT finalize: recover + finalize a committed
    batch straight from chain state, for when the local durable state was lost
    between the two phases (the committed escrow would otherwise be un-finalizable
    via the harness). Reads the batch (provider/requester/status), finalizes it
    directly via the contract, and verifies it reached FINALIZED.

    Idempotent + safe: an already-FINALIZED batch is success (no re-finalize); an
    absent (NONEXISTENT) / VOIDED / not-yet-finalizable batch is never finalized.
    Success is keyed on the authoritative on-chain status, not a wallet delta."""
    batch = await contract_client.get_batch(batch_id)
    status = int(batch["status"])
    if status == 0:
        return {"exists": False, "success": False, "error": "batch not found on chain"}
    requester = batch["requester"]
    recipient = batch["provider"]   # settleFromRequester pays the provider
    if status == 3:
        return {"exists": True, "finalized": False, "success": False,
                "error": "batch is VOIDED"}
    if status == 2:
        recipient_now = int(await escrow_client.ftns_balance_of(recipient))
        escrow_now = int(await escrow_client.balance_of(requester))
        return {"exists": True, "finalizable": True, "finalized": True,
                "recipient_ftns_after": recipient_now, "recipient_ftns_delta": 0,
                "requester_escrow_after": escrow_now, "success": True,
                "note": "already finalized"}

    # status == 1 PENDING
    finalizable = bool(await contract_client.is_finalizable(batch_id))
    if not finalizable:
        seconds_left = await _seconds_until_finalizable(contract_client, batch_id)
        return {"exists": True, "finalizable": False, "finalized": False,
                "success": False, "seconds_until_finalizable": seconds_left}

    recipient_before = int(await escrow_client.ftns_balance_of(recipient))
    await contract_client.finalize_batch(batch_id)   # raises OnChainRevertedError on revert
    final_status = int(await contract_client.get_batch_status(batch_id))
    finalized = (final_status == 2)
    recipient_after = int(await escrow_client.ftns_balance_of(recipient))
    requester_escrow_after = int(await escrow_client.balance_of(requester))
    report = {
        "exists": True, "finalizable": True, "finalized": finalized,
        "recipient_ftns_before": recipient_before,
        "recipient_ftns_after": recipient_after,
        "recipient_ftns_delta": recipient_after - recipient_before,
        "requester_escrow_after": requester_escrow_after,
        "success": bool(finalized),
    }
    logger.info("finalize-by-id: batch %s success=%s escrow_after=%s",
                batch_id.hex()[:12], finalized, requester_escrow_after)
    return report


async def _is_finalized(settlement_client: Any, contract_client: Any,
                        batch_id: bytes) -> bool:
    """True if this batch is FINALIZED — locally recorded OR on-chain status==2
    (authoritative; covers third-party finalizes + restarts)."""
    try:
        if settlement_client.is_finalized_locally(batch_id):
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        return int(await contract_client.get_batch_status(batch_id)) == 2
    except Exception:  # noqa: BLE001
        return False


async def _seconds_until_finalizable(contract_client: Any, batch_id: bytes):
    """Best-effort remaining-window read; None if the contract client can't tell."""
    fn = getattr(contract_client, "seconds_until_finalizable", None)
    if fn is None:
        return None
    try:
        return int(await fn(batch_id))
    except Exception:  # noqa: BLE001
        return None
