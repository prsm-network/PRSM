"""sp1446 — the per-stage (big-model paid multi-stage) commit path's documented double-settle is
CLOSED by the sp1436 durable committed-escrow-id ledger.

`drain_and_commit_staged` / `commit_staged_task` (Design A: each stage node self-commits its own
share) previously carried a documented DEFERRED money bug (sp1431): the receiver-store discard was
the ONLY idempotency for a committed task, so a crash AFTER the on-chain commit lands but BEFORE the
discard persists left the task staged, and the next drain re-committed the same receipt under a
distinct on-chain batchId → DOUBLE-SETTLE. The docstring named the fix: "a committed-escrow-id
ledger."

sp1436 shipped exactly that on BatchSettlementClient: `accumulate()` drops a receipt whose
`local_escrow_id` is already in the durable `_committed_escrow_ids` set (persisted via the client's
state store). The per-stage share-batch carries a STABLE `local_escrow_id = f"{job}::stage::{node}"`
(per_stage_settlement_split), and the per-stage client is wired WITH a durable state store
(resolve_per_stage_settlement_client). So the crash-before-discard re-drain is now dropped at
accumulate — no second on-chain commit. This test proves it end-to-end (in-process AND across a
"restart" with a fresh client rehydrated from the persisted state), on a REAL BatchSettlementClient.

Money assertion — never weaken.
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import time
from decimal import Decimal
from unittest.mock import AsyncMock

from eth_account import Account
from eth_utils import keccak

from prsm.compute.inference.models import ContentTier, InferenceReceipt
from prsm.compute.inference.topology_rotation import TopologyAssignment
from prsm.compute.shard_receipt import build_receipt_signing_payload, per_stage_leaf_job_id
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.compute_wallet_map import ComputeWalletMap
from prsm.node.identity import generate_node_identity
from prsm.settlement.accumulator import AccumulatorConfig, ReceiptAccumulator
from prsm.settlement.client import BatchSettlementClient
from prsm.settlement.per_stage_payment_authorization import (
    DEFAULT_CHAIN_ID,
    compute_payee_set_hash,
    sign_per_stage_authorization,
)
from prsm.settlement.per_stage_receiver_store import (
    PerStageReceiverStore,
    commit_staged_task,
    ingest_routed_task,
)
from prsm.settlement.per_stage_routing import build_per_stage_settlement_tasks, payee_set_from_tasks
from prsm.settlement.per_stage_settlement_split import NodeSignatureMaterial
from prsm.settlement.state_store import SettlementStateStore

_WEI = 10 ** 18
_REQ_KEY = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
_REQ_ADDR = Account.from_key(_REQ_KEY).address
_PAYEE_A = "0x" + "a1" * 20
_TOTAL = 10 * _WEI + 1


def _run(coro):
    return asyncio.run(coro)


def _mock_contract():
    """A unique on-chain batch_id per commit so a double-settle shows up as a SECOND commit_batch."""
    mock = AsyncMock()
    n = {"i": 0}

    async def commit(**kw):
        n["i"] += 1
        return (hashlib.sha256(f"b{n['i']}".encode()).digest(), 1_700_000_000 + n["i"])

    mock.commit_batch.side_effect = commit
    mock.is_finalizable.return_value = False
    mock.finalize_batch.return_value = None
    mock.get_batch_status.return_value = 1
    mock.find_committed_batch_by_root = None
    return mock


def _staged_task(tmp_path):
    """Build one node's fully-signed, authorized staged share-task (node A) in a receiver store."""
    id_a = generate_node_identity(display_name="stage0")
    id_b = generate_node_identity(display_name="stage1")
    out = hashlib.sha256(b"out-1446").digest()
    topo = TopologyAssignment(
        positions={(0, 0): id_a.node_id, (1, 0): id_b.node_id}, stage_count=2, slots_per_stage=1)
    ir0 = InferenceReceipt(
        job_id="job-1446", request_id="req-1446", model_id="qwen2.5-72b",
        content_tier=ContentTier.A, privacy_tier=PrivacyLevel.NONE, epsilon_spent=0.0,
        tee_type=TEEType.SOFTWARE, tee_attestation=b"att", output_hash=out, duration_seconds=1.0,
        cost_ftns=Decimal("1.0"), settler_signature=b"sig", settler_node_id=id_a.node_id,
        topology_assignment=topo)
    job = per_stage_leaf_job_id(ir0.request_id)
    outhex = out.hex()

    def _sig(identity, stage):
        payload = build_receipt_signing_payload(
            job_id=job, shard_index=stage, output_hash=outhex, executed_at_unix=1_700_000_000)
        return NodeSignatureMaterial(
            pubkey_b64=identity.public_key_b64, signature=identity.sign(payload),
            stage_index=stage, output_hash=outhex, executed_at_unix=1_700_000_000)

    ir = dataclasses.replace(ir0, per_stage_settlement_signatures={
        id_a.node_id: _sig(id_a, 0), id_b.node_id: _sig(id_b, 1)})
    wallet = ComputeWalletMap.from_mapping({id_a.node_id: _PAYEE_A, id_b.node_id: "0x" + "b2" * 20})
    bare = build_per_stage_settlement_tasks(
        receipt=ir, total_value_wei=_TOTAL, requester_address=_REQ_ADDR, wallet_map=wallet)
    payees = payee_set_from_tasks(bare)
    auth_payload = {
        "requester": _REQ_ADDR, "payee_set_hash": "0x" + compute_payee_set_hash(payees).hex(),
        "total_max_spend_wei": _TOTAL, "job_nonce": "0x" + keccak(b"n-1446").hex(),
        "expiry_unix": int(time.time()) + 86400, "request_hash": "0x" + keccak(b"r-1446").hex()}
    auth = {"payload": auth_payload,
            "signature": "0x" + sign_per_stage_authorization(auth_payload, _REQ_KEY, chain_id=DEFAULT_CHAIN_ID).hex()}
    tasks = [dataclasses.replace(t, payment_authorization=auth) for t in bare]
    store = PerStageReceiverStore(tmp_path / "rx.json")
    for t in tasks:
        ingest_routed_task(store, t, payees=payees, my_node_id=t.node_id)
    # all_staged() returns StagedSettlementTask wrappers (commit_staged_task takes the wrapper).
    staged_a = next(s for s in store.all_staged() if s.task.node_id == id_a.node_id)
    return store, staged_a


def _client(state_path, contract):
    return BatchSettlementClient(
        ReceiptAccumulator(AccumulatorConfig(count_threshold=1)),
        contract, _PAYEE_A, state_store=SettlementStateStore(state_path))


# ── the money assertion: a crash-before-discard re-commit does NOT double-settle ──


def test_re_commit_of_a_committed_stage_task_does_not_double_settle(tmp_path):
    store, task = _staged_task(tmp_path)
    contract = _mock_contract()
    client = _client(tmp_path / "state.json", contract)

    # First commit lands on-chain (sp1436 records the escrow id in the durable ledger + persists).
    r1 = _run(commit_staged_task(task, client=client))
    assert r1.committed, r1.reason
    assert contract.commit_batch.call_count == 1

    # Simulate the crash window: the on-chain commit landed but the receiver-store DISCARD never
    # persisted, so the SAME task is re-committed on the next drain. sp1436's accumulate() drops it.
    r2 = _run(commit_staged_task(task, client=client))
    assert not r2.committed, "the already-committed stage share was re-committed — DOUBLE SETTLE"
    assert contract.commit_batch.call_count == 1, "second on-chain commit fired — double settle"


def test_dedup_survives_a_node_restart(tmp_path):
    """The durable ledger persists: a FRESH client rehydrated from the same state store still drops
    the re-delivered share — the real crash+restart shape."""
    store, task = _staged_task(tmp_path)
    state_path = tmp_path / "state.json"

    contract1 = _mock_contract()
    _run(commit_staged_task(task, client=_client(state_path, contract1)))
    assert contract1.commit_batch.call_count == 1

    # "restart": a brand-new client over the SAME persisted state store.
    contract2 = _mock_contract()
    r = _run(commit_staged_task(task, client=_client(state_path, contract2)))
    assert not r.committed
    contract2.commit_batch.assert_not_called()
