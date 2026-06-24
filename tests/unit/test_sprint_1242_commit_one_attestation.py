"""Sprint 1242 (TEE Tier-3 roadmap F final glue) — _commit_one computes the batch
attestation commitment from its receipts' tee_attestation dicts and passes it to
commit_batch. Closes the off-chain loop end-to-end in the live commit flow.

The pieces it wires were shipped earlier: batch_attestation_commitment (sp1241,
weakest-link bytes32) + the client routing to commitBatchWithAttestation (sp1241).
This test asserts the commit path actually computes + forwards the commitment.
"""

import hashlib
from base64 import b64encode
from unittest.mock import AsyncMock

from prsm.compute.shard_receipt import (
    ShardExecutionReceipt,
    batch_attestation_commitment,
)
from prsm.settlement.accumulator import (
    AccumulatorConfig,
    BatchedReceipt,
    ReceiptAccumulator,
)
from prsm.settlement.client import BatchSettlementClient

ONE_FTNS = 10 ** 18
PROVIDER_ADDR = "0x" + "b" * 40
REQUESTER_A = "0x" + "a" * 40


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def _att(tee_type, *, is_dev_only=False, is_multi_stage=True):
    return {"attestation_b64": "x", "is_dev_only": is_dev_only,
            "is_multi_stage": is_multi_stage, "tee_type": tee_type}


def _make_batched(shard_index=0, tee_attestation=None):
    receipt = ShardExecutionReceipt(
        job_id=f"job-{shard_index}", shard_index=shard_index,
        provider_id="provider-id-string",
        provider_pubkey_b64=b64encode(b"pubkey-raw").decode(),
        output_hash=hashlib.sha256(f"out-{shard_index}".encode()).hexdigest(),
        executed_at_unix=1700000000 + shard_index,
        signature=b64encode(b"sig-raw").decode(),
        tee_attestation=tee_attestation,
    )
    return BatchedReceipt(
        receipt=receipt, requester_address=REQUESTER_A,
        provider_address=PROVIDER_ADDR, value_ftns=ONE_FTNS,
        local_escrow_id=f"escrow-{shard_index}")


def _store(tmp_path):
    from prsm.settlement.state_store import SettlementStateStore
    return SettlementStateStore(tmp_path / "settlement_state.json")


def _capturing_contract():
    m = AsyncMock()
    captured = {}

    async def commit(**kw):
        captured["attestation_commitment"] = kw.get("attestation_commitment")
        return (hashlib.sha256(b"bid").digest(), 1700000999)

    m.commit_batch.side_effect = commit
    m.is_finalizable.return_value = False
    m.finalize_batch.return_value = None
    m.get_batch_status.return_value = 1
    return m, captured


def _client(tmp_path, threshold=1):
    contract, captured = _capturing_contract()
    client = BatchSettlementClient(
        ReceiptAccumulator(AccumulatorConfig(count_threshold=threshold)),
        contract, PROVIDER_ADDR, state_store=_store(tmp_path))
    return client, captured


def test_commit_one_forwards_weakest_link_commitment(tmp_path):
    client, captured = _client(tmp_path, threshold=2)
    a1, a2 = _att("sgx"), _att("sgx")
    _run(client.accumulate(_make_batched(0, a1)))
    _run(client.accumulate(_make_batched(1, a2)))
    _run(client.commit_ready_batches())

    expected = batch_attestation_commitment([a1, a2])
    assert captured["attestation_commitment"] == expected
    assert expected != b"\x00" * 32   # a real attested batch


def test_commit_one_zero_commitment_when_no_attestation(tmp_path):
    client, captured = _client(tmp_path, threshold=1)
    _run(client.accumulate(_make_batched(0)))  # tee_attestation=None (legacy)
    _run(client.commit_ready_batches())
    assert captured["attestation_commitment"] == b"\x00" * 32


def test_commit_one_weakest_link_one_software_stage(tmp_path):
    # an all-SGX batch and a (SGX + software) batch must yield DIFFERENT
    # commitments — the software stage collapses the batch tier.
    client, captured = _client(tmp_path, threshold=2)
    a_sgx, a_soft = _att("sgx"), _att("software")
    _run(client.accumulate(_make_batched(0, a_sgx)))
    _run(client.accumulate(_make_batched(1, a_soft)))
    _run(client.commit_ready_batches())
    assert captured["attestation_commitment"] == batch_attestation_commitment([a_sgx, a_soft])
    assert captured["attestation_commitment"] != batch_attestation_commitment([a_sgx, a_sgx])
