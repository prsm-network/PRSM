"""Unit tests for Phase 7.1x — ConsensusChallengeSubmitter.

Mocks web3 entirely. Verifies:
  - ReceiptLeafFields conversion from Python ShardExecutionReceipt
    matches the CANONICAL committed-leaf convention (merkle.batched_receipt_to_leaf):
    job_id/provider_id via keccak(utf8), pubkey/signature via keccak(b64decode).
  - auxData encoding matches _handleConsensusMismatch's abi.decode layout.
  - submit_one returns success/failure uniformly, never raises.
  - submit_batch runs through all attempts even when some fail.
  - Drain API on the orchestrator hands the submitter work cleanly.
"""
from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import pytest

from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.marketplace.consensus_submitter import (
    ChallengeAttempt,
    ChallengeResult,
    ConsensusChallengeSubmitter,
    DEFAULT_CHALLENGE_GAS,
    ReceiptLeafFields,
)
from prsm.economy.web3.stake_manager import ReasonCode


# ── Helpers ──────────────────────────────────────────────────────────


# Real receipts carry BASE64 pubkey/signature (32-byte ed25519 key, 64-byte sig), which is why
# the committed-leaf convention (merkle.batched_receipt_to_leaf) b64-decodes them before
# hashing. Use valid base64 here so from_python_receipt's keccak(b64decode(...)) matches.
_PUBKEY_RAW = b"provider-ed25519-pubkey-32 bytes"   # any bytes; b64-roundtripped below
_SIG_RAW = b"ed25519-signature-placeholder-bytes"   # any bytes; b64-roundtripped below
_PUBKEY_B64 = base64.b64encode(_PUBKEY_RAW).decode("ascii")
_SIG_B64 = base64.b64encode(_SIG_RAW).decode("ascii")


def _make_receipt(
    provider_id: str = "provA",
    output_hash: bytes = b"\x01" * 32,
) -> ShardExecutionReceipt:
    return ShardExecutionReceipt(
        job_id="job-phase7.1",
        shard_index=0,
        provider_id=provider_id,
        provider_pubkey_b64=_PUBKEY_B64,
        output_hash=output_hash.hex(),
        executed_at_unix=1_700_000_000,
        signature=_SIG_B64,
    )


@pytest.fixture
def mock_web3():
    with patch(
        "prsm.marketplace.consensus_submitter.Web3"
    ) as MockWeb3:
        w3 = MagicMock()
        MockWeb3.return_value = w3
        MockWeb3.HTTPProvider.return_value = MagicMock()
        MockWeb3.to_checksum_address.side_effect = lambda x: x
        yield w3, MockWeb3


def _make_submitter(mock_web3, gas_budget=None):
    w3, _ = mock_web3
    contract = MagicMock()
    w3.eth.contract.return_value = contract

    account = MagicMock()
    account.address = "0xChallenger"
    account.key = b"\x33" * 32
    with patch(
        "prsm.marketplace.consensus_submitter.Account.from_key",
        return_value=account,
    ):
        kwargs = {
            "rpc_url": "http://localhost:8545",
            "registry_address": "0xRegistry",
            "private_key": "0x" + "33" * 32,
        }
        if gas_budget is not None:
            kwargs["gas_budget"] = gas_budget
        sub = ConsensusChallengeSubmitter(**kwargs)
    return sub, contract, w3


def _make_attempt():
    # Sprint 347 — ReceiptLeafFields gained signing_message_hash
    # to match the on-chain ReceiptLeaf struct (L2 audit C-INT-01).
    leaf_majority = ReceiptLeafFields(
        job_id_hash=b"\x11" * 32,
        shard_index=0,
        provider_id_hash=b"\x22" * 32,
        provider_pubkey_hash=b"\x33" * 32,
        output_hash=b"\xaa" * 32,
        executed_at_unix=1_700_000_000,
        value_ftns_wei=10 * 10**18,
        signature_hash=b"\x44" * 32,
        signing_message_hash=b"\x88" * 32,
    )
    leaf_minority = ReceiptLeafFields(
        job_id_hash=b"\x11" * 32,   # same job
        shard_index=0,               # same shard
        provider_id_hash=b"\x55" * 32,
        provider_pubkey_hash=b"\x66" * 32,
        output_hash=b"\xbb" * 32,    # different output
        executed_at_unix=1_700_000_000,
        value_ftns_wei=10 * 10**18,
        signature_hash=b"\x77" * 32,
        signing_message_hash=b"\x99" * 32,
    )
    return ChallengeAttempt(
        minority_batch_id=b"\x01" * 32,
        minority_leaf=leaf_minority,
        minority_proof=[],
        majority_batch_id=b"\x02" * 32,
        majority_leaf=leaf_majority,
        majority_proof=[],
    )


def _stub_happy_send(w3, contract):
    contract.functions.challengeReceipt.return_value.build_transaction.return_value = {
        "to": "0xRegistry", "data": "0x", "gas": DEFAULT_CHALLENGE_GAS,
        "gasPrice": 1, "nonce": 0, "chainId": 8453,
    }
    w3.eth.get_transaction_count.return_value = 0
    w3.eth.gas_price = 1_000_000_000
    w3.eth.chain_id = 8453
    signed = MagicMock()
    signed.raw_transaction = b"raw"
    w3.eth.account.sign_transaction.return_value = signed
    w3.eth.send_raw_transaction.return_value = b"\xcc" * 32
    receipt = MagicMock()
    receipt.status = 1
    w3.eth.wait_for_transaction_receipt.return_value = receipt


# ── ReceiptLeafFields conversion ─────────────────────────────────────


def test_receipt_leaf_fields_from_python_receipt_hashes_correctly():
    """Must match the CANONICAL committed-leaf convention (merkle.batched_receipt_to_leaf),
    which is what the on-chain merkleRoot is folded over: job_id + provider_id are keccak256
    of their UTF-8 bytes, but pubkey + signature are keccak256 of their BASE64-DECODED bytes
    (NOT the utf8 of the base64 string). output_hash is a raw 32-byte hex (NOT re-hashed).

    sp1165 fix: pre-fix this hashed keccak(utf8(b64)) for pubkey/sig, diverging from the
    committed convention -> every consensus challenge reverted InvalidMerkleProof on-chain."""
    from eth_utils import keccak

    receipt = _make_receipt(output_hash=b"\xde\xad\xbe\xef" + b"\x00" * 28)
    fields = ReceiptLeafFields.from_python_receipt(
        receipt, value_ftns_wei=5 * 10**18,
    )
    assert fields.job_id_hash == keccak(b"job-phase7.1")
    assert fields.provider_id_hash == keccak(b"provA")
    # pubkey + signature hash the BASE64-DECODED bytes (committed convention)
    assert fields.provider_pubkey_hash == keccak(base64.b64decode(_PUBKEY_B64))
    assert fields.provider_pubkey_hash == keccak(_PUBKEY_RAW)
    assert fields.output_hash == b"\xde\xad\xbe\xef" + b"\x00" * 28
    assert fields.signature_hash == keccak(base64.b64decode(_SIG_B64))
    assert fields.signature_hash == keccak(_SIG_RAW)
    assert fields.value_ftns_wei == 5 * 10**18
    assert fields.shard_index == 0
    assert fields.executed_at_unix == 1_700_000_000


def test_receipt_leaf_fields_rejects_wrong_output_hash_length():
    """output_hash MUST decode to 32 bytes. The contract expects bytes32."""
    bad_receipt = ShardExecutionReceipt(
        job_id="j", shard_index=0,
        provider_id="p", provider_pubkey_b64="x",
        output_hash="deadbeef",   # only 4 bytes, not 32
        executed_at_unix=1, signature="s",
    )
    with pytest.raises(ValueError, match="32 bytes"):
        ReceiptLeafFields.from_python_receipt(
            bad_receipt, value_ftns_wei=1,
        )


def test_receipt_leaf_fields_to_tuple_matches_contract_argument_order():
    """The tuple must be ordered (jobIdHash, shardIndex, providerIdHash,
    providerPubkeyHash, outputHash, executedAtUnix, valueFtns,
    signatureHash, signingMessageHash) — matches ReceiptLeaf struct
    field order in the ABI. Sprint 347 added signingMessageHash at
    position 8 (L2 audit C-INT-01)."""
    fields = ReceiptLeafFields(
        job_id_hash=b"\x01" * 32, shard_index=7,
        provider_id_hash=b"\x02" * 32, provider_pubkey_hash=b"\x03" * 32,
        output_hash=b"\x04" * 32, executed_at_unix=1234,
        value_ftns_wei=10**18, signature_hash=b"\x05" * 32,
        signing_message_hash=b"\x06" * 32,
    )
    t = fields.to_tuple()
    assert t[0] == b"\x01" * 32      # jobIdHash
    assert t[1] == 7                  # shardIndex
    assert t[2] == b"\x02" * 32      # providerIdHash
    assert t[3] == b"\x03" * 32      # providerPubkeyHash
    assert t[4] == b"\x04" * 32      # outputHash
    assert t[5] == 1234               # executedAtUnix
    assert t[6] == 10**18             # valueFtns
    assert t[7] == b"\x05" * 32      # signatureHash
    assert t[8] == b"\x06" * 32      # signingMessageHash


# ── auxData encoding ─────────────────────────────────────────────────


def test_aux_encoding_matches_handler_layout(mock_web3):
    """The auxData MUST decode back to
    (conflictingBatchId, majorityProof, majorityLeaf) per
    _handleConsensusMismatch. Round-trip via eth_abi."""
    from eth_abi import decode as abi_decode

    sub, _, _ = _make_submitter(mock_web3)
    attempt = _make_attempt()
    encoded = sub._encode_aux(attempt)

    # Sprint 347 — ReceiptLeaf gained signingMessageHash.
    (batch_id, proof, leaf_tuple) = abi_decode(
        [
            "bytes32",
            "bytes32[]",
            "(bytes32,uint32,bytes32,bytes32,bytes32,uint64,uint128,bytes32,bytes32)",
        ],
        encoded,
    )
    assert batch_id == attempt.majority_batch_id
    assert list(proof) == attempt.majority_proof
    assert leaf_tuple == attempt.majority_leaf.to_tuple()


# ── submit_one happy path ────────────────────────────────────────────


def test_submit_one_success_returns_tx_hash(mock_web3):
    sub, contract, w3 = _make_submitter(mock_web3)
    _stub_happy_send(w3, contract)

    result = sub.submit_one(_make_attempt())
    assert isinstance(result, ChallengeResult)
    assert result.success is True
    assert result.tx_hash_hex is not None
    assert result.tx_hash_hex.startswith("0x")
    assert result.error_type is None


def test_submit_one_uses_default_gas_budget(mock_web3):
    """Gas budget defaults to DEFAULT_CHALLENGE_GAS (1M), which is well
    above Phase 7 §8.7's MIN_SLASH_GAS floor (150_000)."""
    sub, contract, w3 = _make_submitter(mock_web3)
    _stub_happy_send(w3, contract)
    sub.submit_one(_make_attempt())

    # build_transaction was called with the gas budget in overrides.
    call_args = contract.functions.challengeReceipt.return_value.build_transaction.call_args
    tx_overrides = call_args.args[0]
    assert tx_overrides["gas"] == DEFAULT_CHALLENGE_GAS
    assert DEFAULT_CHALLENGE_GAS > 150_000   # pins that we're above the floor


def test_submit_one_respects_custom_gas_budget(mock_web3):
    sub, contract, w3 = _make_submitter(mock_web3, gas_budget=500_000)
    _stub_happy_send(w3, contract)
    sub.submit_one(_make_attempt())
    tx_overrides = contract.functions.challengeReceipt.return_value.build_transaction.call_args.args[0]
    assert tx_overrides["gas"] == 500_000


def test_submit_one_passes_consensus_mismatch_reason_code(mock_web3):
    sub, contract, w3 = _make_submitter(mock_web3)
    _stub_happy_send(w3, contract)
    attempt = _make_attempt()
    sub.submit_one(attempt)

    # Inspect the registry call's positional args.
    call_args = contract.functions.challengeReceipt.call_args
    # Positional: (batchId, leaf, proof, reason, auxData)
    assert call_args.args[0] == attempt.minority_batch_id
    assert call_args.args[1] == attempt.minority_leaf.to_tuple()
    assert call_args.args[2] == attempt.minority_proof
    assert call_args.args[3] == int(ReasonCode.CONSENSUS_MISMATCH)
    # auxData is bytes — non-empty
    assert isinstance(call_args.args[4], (bytes, bytearray))
    assert len(call_args.args[4]) > 0


# ── submit_one failure surfaces ─────────────────────────────────────


def test_submit_one_returns_failure_on_broadcast_error(mock_web3):
    """Broadcast-level failures (RPC down, bad tx) surface as failure
    results rather than raising. The caller's queue-drain loop must
    never abort on one broken attempt."""
    sub, contract, w3 = _make_submitter(mock_web3)
    contract.functions.challengeReceipt.return_value.build_transaction.return_value = {
        "to": "0xRegistry", "data": "0x", "gas": 1_000_000,
        "gasPrice": 1, "nonce": 0, "chainId": 8453,
    }
    w3.eth.get_transaction_count.return_value = 0
    w3.eth.gas_price = 1_000_000_000
    w3.eth.chain_id = 8453
    signed = MagicMock()
    signed.raw_transaction = b"raw"
    w3.eth.account.sign_transaction.return_value = signed
    w3.eth.send_raw_transaction.side_effect = Exception("RPC unreachable")

    result = sub.submit_one(_make_attempt())
    assert result.success is False
    assert result.error_type == "BroadcastFailedError"
    assert "RPC unreachable" in result.error_message


def test_submit_one_returns_failure_on_reverted_tx(mock_web3):
    """Contract revert (e.g., ChallengeNotProven, InsufficientGasForSlash)
    → receipt.status=0 → OnChainRevertedError surfaced as failure."""
    sub, contract, w3 = _make_submitter(mock_web3)
    contract.functions.challengeReceipt.return_value.build_transaction.return_value = {
        "to": "0xRegistry", "data": "0x", "gas": 1_000_000,
        "gasPrice": 1, "nonce": 0, "chainId": 8453,
    }
    w3.eth.get_transaction_count.return_value = 0
    w3.eth.gas_price = 1_000_000_000
    w3.eth.chain_id = 8453
    signed = MagicMock()
    signed.raw_transaction = b"raw"
    w3.eth.account.sign_transaction.return_value = signed
    w3.eth.send_raw_transaction.return_value = b"\xcc" * 32
    receipt = MagicMock()
    receipt.status = 0   # revert
    w3.eth.wait_for_transaction_receipt.return_value = receipt

    result = sub.submit_one(_make_attempt())
    assert result.success is False
    assert result.error_type == "OnChainRevertedError"


def test_submit_one_returns_failure_on_pending_timeout(mock_web3):
    """Broadcast succeeded but receipt timed out → OnChainPendingError
    (do NOT retry — tx may still land; UNSAFE to submit a replacement)."""
    sub, contract, w3 = _make_submitter(mock_web3)
    contract.functions.challengeReceipt.return_value.build_transaction.return_value = {
        "to": "0xRegistry", "data": "0x", "gas": 1_000_000,
        "gasPrice": 1, "nonce": 0, "chainId": 8453,
    }
    w3.eth.get_transaction_count.return_value = 0
    w3.eth.gas_price = 1_000_000_000
    w3.eth.chain_id = 8453
    signed = MagicMock()
    signed.raw_transaction = b"raw"
    w3.eth.account.sign_transaction.return_value = signed
    w3.eth.send_raw_transaction.return_value = b"\xcc" * 32
    w3.eth.wait_for_transaction_receipt.side_effect = Exception("timeout")

    result = sub.submit_one(_make_attempt())
    assert result.success is False
    assert result.error_type == "OnChainPendingError"
    # tx_hash IS captured for manual reconciliation.
    assert result.tx_hash_hex is not None


def test_submit_one_does_not_raise_on_unexpected_exception(mock_web3):
    """A bug inside our code (non-web3 exception) should still produce
    a failure result, not abort the caller's drain loop. Log loudly."""
    sub, contract, w3 = _make_submitter(mock_web3)
    # Force _encode_aux to explode via a patched abi_encode.
    with patch(
        "prsm.marketplace.consensus_submitter.abi_encode",
        side_effect=KeyError("bug in encoder"),
    ):
        result = sub.submit_one(_make_attempt())
    assert result.success is False
    assert result.error_type == "KeyError"
    assert "bug in encoder" in result.error_message


# ── submit_batch ─────────────────────────────────────────────────────


def test_submit_batch_runs_every_attempt(mock_web3):
    sub, contract, w3 = _make_submitter(mock_web3)
    _stub_happy_send(w3, contract)

    attempts = [_make_attempt() for _ in range(3)]
    results = sub.submit_batch(attempts)
    assert len(results) == 3
    assert all(r.success for r in results)
    # 3 calls to challengeReceipt.
    assert contract.functions.challengeReceipt.call_count == 3


def test_submit_batch_does_not_abort_on_mid_failure(mock_web3):
    """One failing attempt must not block the remaining attempts."""
    sub, contract, w3 = _make_submitter(mock_web3)

    # First succeeds, second's broadcast fails, third succeeds.
    contract.functions.challengeReceipt.return_value.build_transaction.return_value = {
        "to": "0xRegistry", "data": "0x", "gas": 1_000_000,
        "gasPrice": 1, "nonce": 0, "chainId": 8453,
    }
    w3.eth.get_transaction_count.return_value = 0
    w3.eth.gas_price = 1_000_000_000
    w3.eth.chain_id = 8453
    signed = MagicMock()
    signed.raw_transaction = b"raw"
    w3.eth.account.sign_transaction.return_value = signed

    good_hash = b"\xaa" * 32
    send_calls = [good_hash, Exception("rpc down"), good_hash]
    w3.eth.send_raw_transaction.side_effect = send_calls

    receipt = MagicMock()
    receipt.status = 1
    w3.eth.wait_for_transaction_receipt.return_value = receipt

    results = sub.submit_batch([_make_attempt() for _ in range(3)])
    assert [r.success for r in results] == [True, False, True]
    assert results[1].error_type == "BroadcastFailedError"


# ── Orchestrator drain API ───────────────────────────────────────────


def test_orchestrator_drain_returns_and_clears_queue():
    """MarketplaceOrchestrator.drain_consensus_minority_queue pops all
    pending challenge entries and leaves the queue empty."""
    from prsm.marketplace.orchestrator import MarketplaceOrchestrator

    identity = MagicMock()
    orch = MarketplaceOrchestrator(
        identity=identity,
        directory=MagicMock(),
        eligibility_filter=MagicMock(),
        reputation=MagicMock(),
        price_negotiator=MagicMock(),
        remote_dispatcher=MagicMock(),
    )
    # Hand-seed the queue with 2 entries.
    orch.consensus_minority_queue.extend([
        {"job_id": "j1", "shard_index": 0},
        {"job_id": "j2", "shard_index": 1},
    ])

    drained = orch.drain_consensus_minority_queue()
    assert len(drained) == 2
    assert drained[0]["job_id"] == "j1"
    assert drained[1]["job_id"] == "j2"
    assert orch.consensus_minority_queue == []


def test_orchestrator_drain_is_idempotent_when_empty():
    """Calling drain on an empty queue returns [] without error."""
    from prsm.marketplace.orchestrator import MarketplaceOrchestrator

    orch = MarketplaceOrchestrator(
        identity=MagicMock(),
        directory=MagicMock(),
        eligibility_filter=MagicMock(),
        reputation=MagicMock(),
        price_negotiator=MagicMock(),
        remote_dispatcher=MagicMock(),
    )
    assert orch.drain_consensus_minority_queue() == []
    # Calling twice is safe.
    assert orch.drain_consensus_minority_queue() == []


def test_orchestrator_drain_preserves_fifo_order():
    """Multiple queued entries come back in insertion order so the
    submitter processes oldest-first."""
    from prsm.marketplace.orchestrator import MarketplaceOrchestrator

    orch = MarketplaceOrchestrator(
        identity=MagicMock(),
        directory=MagicMock(),
        eligibility_filter=MagicMock(),
        reputation=MagicMock(),
        price_negotiator=MagicMock(),
        remote_dispatcher=MagicMock(),
    )
    orch.consensus_minority_queue.extend([
        {"job_id": f"job-{i}", "shard_index": i} for i in range(5)
    ])
    drained = orch.drain_consensus_minority_queue()
    assert [e["job_id"] for e in drained] == [
        f"job-{i}" for i in range(5)
    ]
