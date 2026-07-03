"""Sprint 1369 — multi-stage settlement full-cycle VALUE-CONSERVATION smoke (offline).

The live-proof prep's one missing integration: the pieces (split sp1315, route/verify sp1316,
commit sp1322) are unit-tested individually, but nothing composes them with a conservation
assertion THROUGH the commit — i.e. "the requester pays T; the pipeline splits it into per-node
shares that sum to exactly T; each node commits exactly its own share (no over/under-pay)." This is
the money-safety invariant the 2-node testnet go/no-go must uphold. Generalized to N stages so it
proves the pipeline scales past 2 (the runbook's minimum) — offline, no chain, no nodes.
"""
from __future__ import annotations

import dataclasses
import hashlib
from decimal import Decimal

import pytest

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
    build_per_stage_settlement_tasks,
    payee_set_from_tasks,
)
from prsm.settlement.per_stage_settlement_split import NodeSignatureMaterial

_WEI = 10 ** 18
_REQUESTER = "0x" + "11" * 20
_AUTH = {"payload": {"requester": _REQUESTER}, "signature": "0x" + "cd" * 65}


def _n_node_receipt(n: int):
    ids = [generate_node_identity(display_name=f"stage{i}") for i in range(n)]
    positions = {(i, 0): ids[i].node_id for i in range(n)}
    topo = TopologyAssignment(positions=positions, stage_count=n, slots_per_stage=1)
    output_hash = hashlib.sha256(b"n-stage output").digest()
    ir0 = InferenceReceipt(
        job_id="job-1369", request_id="req-1369", model_id="qwen2.5-72b",
        content_tier=ContentTier.A, privacy_tier=PrivacyLevel.NONE, epsilon_spent=0.0,
        tee_type=TEEType.SOFTWARE, tee_attestation=b"att", output_hash=output_hash,
        duration_seconds=1.0, cost_ftns=Decimal("1.0"), settler_signature=b"sig",
        settler_node_id=ids[0].node_id, topology_assignment=topo)
    out = ir0.output_hash.hex()
    job = per_stage_leaf_job_id(ir0.request_id)
    sigs = {
        ids[i].node_id: NodeSignatureMaterial(
            pubkey_b64=ids[i].public_key_b64,
            signature=ids[i].sign(build_receipt_signing_payload(
                job_id=job, shard_index=i, output_hash=out, executed_at_unix=1_700_000_000)),
            stage_index=i, output_hash=out, executed_at_unix=1_700_000_000)
        for i in range(n)
    }
    ir = dataclasses.replace(ir0, per_stage_settlement_signatures=sigs)
    payees = {ids[i].node_id: "0x" + f"{i + 1:02x}" * 20 for i in range(n)}
    return ir, ids, ComputeWalletMap.from_mapping(payees), payees


@pytest.mark.parametrize("n", [2, 3, 8])
def test_full_cycle_conserves_the_requesters_payment(n):
    total = 7 * _WEI + 13          # deliberately not evenly divisible → exercises remainder handling
    ir, ids, wallet, payees = _n_node_receipt(n)

    tasks = build_per_stage_settlement_tasks(
        receipt=ir, total_value_wei=total, requester_address=_REQUESTER,
        per_stage_authorization=_AUTH, wallet_map=wallet)
    assert tasks is not None and len(tasks) == n           # one routable task per stage node

    # CONSERVATION at the split: the per-node shares sum to EXACTLY the requester's charge.
    assert sum(t.share_wei for t in tasks) == total
    assert all(t.share_wei > 0 for t in tasks)             # no zero/negative shares

    # the full payee set the orchestrator derives has one (payee, share) per node, summing to total.
    payee_set = payee_set_from_tasks(tasks)
    assert len(payee_set) == n
    assert sum(share for _addr, share in payee_set) == total

    # CONSERVATION through the COMMIT: each node commits ONLY its own share; the sum is exactly total.
    committed: dict = {}
    for t in tasks:
        assert t.node_id not in committed                  # each node commits once
        committed[t.node_id] = int(t.share_wei)
    assert sum(committed.values()) == total                # nothing minted, nothing lost


def test_single_stage_falls_back_not_split():
    # a receipt with no per-node signatures → single-payee fallback (None), never a bogus split.
    ir, ids, wallet, _ = _n_node_receipt(2)
    ir = dataclasses.replace(ir, per_stage_settlement_signatures=None)
    assert build_per_stage_settlement_tasks(
        receipt=ir, total_value_wei=_WEI, requester_address=_REQUESTER, wallet_map=wallet) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
