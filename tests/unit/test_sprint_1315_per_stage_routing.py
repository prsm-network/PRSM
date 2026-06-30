"""Sprint 1315 (S3a) — per-stage settlement task routing (build_per_stage_settlement_tasks).

Turns a settled MULTI-STAGE receipt (carrying per_stage_settlement_signatures, sp1314/S2)
into N routable per-node settlement TASKS — each = the node's challenge-defensible
BatchedReceipt (from the brick-1 splitter) + the requester's per-stage auth. Pure (no
transport, no commit — those are S3b). Integration-tested through the REAL splitter +
ComputeWalletMap so it's not a mock-only check.
"""
from __future__ import annotations

import dataclasses
import hashlib
from decimal import Decimal

from eth_utils import is_address

from prsm.compute.inference.models import ContentTier, InferenceReceipt
from prsm.compute.inference.topology_rotation import TopologyAssignment
from prsm.compute.shard_receipt import (
    build_receipt_signing_payload,
    per_stage_leaf_job_id,
)
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.compute_wallet_map import ComputeWalletMap
from prsm.node.identity import generate_node_identity
from prsm.settlement.per_stage_routing import (
    PerStageSettlementTask,
    build_per_stage_settlement_tasks,
)
from prsm.settlement.per_stage_settlement_split import NodeSignatureMaterial

_WEI = 10 ** 18
_PAYEE_A = "0x" + "a1" * 20
_PAYEE_B = "0x" + "b2" * 20
_REQUESTER = "0x" + "11" * 20
_AUTH = {"payload": {"requester": _REQUESTER}, "signature": "0x" + "cd" * 65}


def _make_ir(*, node_a_id, node_b_id, single=False, **over):
    output_hash = hashlib.sha256(b"the multi-stage output").digest()
    positions = ({(0, 0): node_a_id} if single
                 else {(0, 0): node_a_id, (1, 0): node_b_id})
    topo = TopologyAssignment(
        positions=positions, stage_count=(1 if single else 2), slots_per_stage=1)
    base = dict(
        job_id="job-1315", request_id="req-1315", model_id="qwen2.5-72b",
        content_tier=ContentTier.A, privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0, tee_type=TEEType.SOFTWARE, tee_attestation=b"att",
        output_hash=output_hash, duration_seconds=1.0, cost_ftns=Decimal("1.0"),
        settler_signature=b"sig", settler_node_id=node_a_id, topology_assignment=topo,
    )
    base.update(over)
    return InferenceReceipt(**base)


def _sig(identity, *, job_id, stage_index, output_hex, executed):
    payload = build_receipt_signing_payload(
        job_id=job_id, shard_index=stage_index,
        output_hash=output_hex, executed_at_unix=executed)
    return NodeSignatureMaterial(
        pubkey_b64=identity.public_key_b64, signature=identity.sign(payload),
        stage_index=stage_index, output_hash=output_hex, executed_at_unix=executed)


def _two_node_receipt():
    id_a = generate_node_identity(display_name="stage0")
    id_b = generate_node_identity(display_name="stage1")
    ir0 = _make_ir(node_a_id=id_a.node_id, node_b_id=id_b.node_id)
    out = ir0.output_hash.hex()
    job = per_stage_leaf_job_id(ir0.request_id)
    sigs = {
        id_a.node_id: _sig(id_a, job_id=job, stage_index=0, output_hex=out, executed=1_700_000_000),
        id_b.node_id: _sig(id_b, job_id=job, stage_index=1, output_hex=out, executed=1_700_000_000),
    }
    ir = dataclasses.replace(ir0, per_stage_settlement_signatures=sigs)
    wallet = ComputeWalletMap.from_mapping({id_a.node_id: _PAYEE_A, id_b.node_id: _PAYEE_B})
    return ir, id_a, id_b, wallet


# ── happy path ───────────────────────────────────────────────────────────────

def test_builds_one_task_per_stage_node():
    ir, id_a, id_b, wallet = _two_node_receipt()
    tasks = build_per_stage_settlement_tasks(
        receipt=ir, total_value_wei=10 * _WEI + 1, requester_address=_REQUESTER,
        per_stage_authorization=_AUTH, wallet_map=wallet)
    assert tasks is not None and len(tasks) == 2
    by_node = {t.node_id: t for t in tasks}
    assert set(by_node) == {id_a.node_id, id_b.node_id}
    # conserving split (non-divisible total → remainder distributed)
    assert sum(t.share_wei for t in tasks) == 10 * _WEI + 1
    # each task's BatchedReceipt is stamped to that node's OWN resolved eth payee
    assert by_node[id_a.node_id].batched_receipt.provider_address.lower() == _PAYEE_A.lower()
    assert by_node[id_b.node_id].batched_receipt.provider_address.lower() == _PAYEE_B.lower()
    for t in tasks:
        assert is_address(t.batched_receipt.provider_address)
        assert t.payment_authorization is _AUTH      # the per-stage auth rides along


def test_task_serialization_round_trip():
    ir, _a, _b, wallet = _two_node_receipt()
    tasks = build_per_stage_settlement_tasks(
        receipt=ir, total_value_wei=2 * _WEI, requester_address=_REQUESTER,
        per_stage_authorization=_AUTH, wallet_map=wallet)
    t = tasks[0]
    d = t.to_dict()
    assert d["node_id"] == t.node_id
    assert d["share_wei"] == str(t.share_wei)
    assert d["payment_authorization"] == _AUTH
    t2 = PerStageSettlementTask.from_dict(d)
    assert t2.node_id == t.node_id
    assert t2.share_wei == t.share_wei
    assert t2.batched_receipt.provider_address == t.batched_receipt.provider_address
    assert t2.payment_authorization == _AUTH


# ── fail-closed (single-payee fallback) ──────────────────────────────────────

def test_none_when_receipt_has_no_per_stage_signatures():
    ir, _a, _b, wallet = _two_node_receipt()
    ir = dataclasses.replace(ir, per_stage_settlement_signatures=None)
    assert build_per_stage_settlement_tasks(
        receipt=ir, total_value_wei=_WEI, requester_address=_REQUESTER,
        wallet_map=wallet) is None


def test_none_for_single_node_topology():
    id_a = generate_node_identity(display_name="solo")
    ir0 = _make_ir(node_a_id=id_a.node_id, node_b_id=id_a.node_id, single=True)
    out = ir0.output_hash.hex(); job = per_stage_leaf_job_id(ir0.request_id)
    sigs = {id_a.node_id: _sig(id_a, job_id=job, stage_index=0, output_hex=out, executed=1)}
    ir = dataclasses.replace(ir0, per_stage_settlement_signatures=sigs)
    assert build_per_stage_settlement_tasks(
        receipt=ir, total_value_wei=_WEI, requester_address=_REQUESTER,
        wallet_map=ComputeWalletMap.from_mapping({id_a.node_id: _PAYEE_A})) is None


def test_none_when_a_node_unmapped():
    ir, id_a, _id_b, _wallet = _two_node_receipt()
    # only node A mapped → node B has no eth payee → fail-closed
    partial = ComputeWalletMap.from_mapping({id_a.node_id: _PAYEE_A})
    assert build_per_stage_settlement_tasks(
        receipt=ir, total_value_wei=_WEI, requester_address=_REQUESTER,
        wallet_map=partial) is None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
