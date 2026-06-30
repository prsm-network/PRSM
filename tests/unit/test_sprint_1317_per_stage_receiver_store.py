"""Sprint 1317 (S3b-2) — node-side receiver store + ingest for routed per-stage settlement tasks.

A stage node receives its routed task (+ the full payee set), FAIL-CLOSED verifies its own
membership (sp1316 gate), and STAGES the authorized task for its own client to commit later
(S3b-3). Tested through the REAL splitter + REAL signed auth: ingest accepts + stages an
authorized task, is idempotent on the per-node ``local_escrow_id`` (no double-stage), rejects +
stages NOTHING on a bad auth / misroute, and the store round-trips on disk.
"""
from __future__ import annotations

import dataclasses
import hashlib
import time
from decimal import Decimal

from eth_account import Account
from eth_utils import keccak

from prsm.compute.inference.models import ContentTier, InferenceReceipt
from prsm.compute.inference.topology_rotation import TopologyAssignment
from prsm.compute.shard_receipt import (
    build_receipt_signing_payload,
    per_stage_leaf_job_id,
)
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.compute_wallet_map import ComputeWalletMap
from prsm.node.identity import generate_node_identity
from prsm.settlement.per_stage_payment_authorization import (
    DEFAULT_CHAIN_ID,
    compute_payee_set_hash,
    sign_per_stage_authorization,
)
from prsm.settlement.per_stage_receiver_store import (
    PerStageReceiverStore,
    StagedSettlementTask,
    ingest_routed_task,
)
from prsm.settlement.per_stage_routing import (
    build_per_stage_settlement_tasks,
    payee_set_from_tasks,
)
from prsm.settlement.per_stage_settlement_split import NodeSignatureMaterial

_WEI = 10 ** 18
_REQ_KEY = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
_REQ_ADDR = Account.from_key(_REQ_KEY).address
_PAYEE_A = "0x" + "a1" * 20
_PAYEE_B = "0x" + "b2" * 20
_TOTAL = 10 * _WEI + 1


def _make_ir(*, node_a_id, node_b_id):
    out = hashlib.sha256(b"the multi-stage output").digest()
    topo = TopologyAssignment(
        positions={(0, 0): node_a_id, (1, 0): node_b_id}, stage_count=2, slots_per_stage=1)
    return InferenceReceipt(
        job_id="job-1317", request_id="req-1317", model_id="qwen2.5-72b",
        content_tier=ContentTier.A, privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0, tee_type=TEEType.SOFTWARE, tee_attestation=b"att",
        output_hash=out, duration_seconds=1.0, cost_ftns=Decimal("1.0"),
        settler_signature=b"sig", settler_node_id=node_a_id, topology_assignment=topo)


def _sig(identity, *, job_id, stage_index, output_hex, executed):
    payload = build_receipt_signing_payload(
        job_id=job_id, shard_index=stage_index, output_hash=output_hex, executed_at_unix=executed)
    return NodeSignatureMaterial(
        pubkey_b64=identity.public_key_b64, signature=identity.sign(payload),
        stage_index=stage_index, output_hash=output_hex, executed_at_unix=executed)


def _signed_auth(payees, *, total_cap=None, expiry_delta=86400, key=_REQ_KEY):
    # Mirror the REAL wire shape (build_per_stage_payment_authorization): all hash fields are
    # 0x-hex STRINGS so the auth is JSON-safe end-to-end (HTTP transport + on-disk persistence).
    payload = {
        "requester": _REQ_ADDR,
        "payee_set_hash": "0x" + compute_payee_set_hash(payees).hex(),
        "total_max_spend_wei": _TOTAL if total_cap is None else total_cap,
        "job_nonce": "0x" + keccak(b"nonce-1317").hex(),
        "expiry_unix": int(time.time()) + expiry_delta,
        "request_hash": "0x" + keccak(b"req-1317").hex(),
    }
    sig = sign_per_stage_authorization(payload, key, chain_id=DEFAULT_CHAIN_ID)
    return {"payload": payload, "signature": "0x" + sig.hex()}  # wire shape: hex string


def _routed(*, auth_kwargs=None):
    id_a = generate_node_identity(display_name="stage0")
    id_b = generate_node_identity(display_name="stage1")
    ir0 = _make_ir(node_a_id=id_a.node_id, node_b_id=id_b.node_id)
    out = ir0.output_hash.hex(); job = per_stage_leaf_job_id(ir0.request_id)
    sigs = {
        id_a.node_id: _sig(id_a, job_id=job, stage_index=0, output_hex=out, executed=1_700_000_000),
        id_b.node_id: _sig(id_b, job_id=job, stage_index=1, output_hex=out, executed=1_700_000_000),
    }
    ir = dataclasses.replace(ir0, per_stage_settlement_signatures=sigs)
    wallet = ComputeWalletMap.from_mapping({id_a.node_id: _PAYEE_A, id_b.node_id: _PAYEE_B})
    bare = build_per_stage_settlement_tasks(
        receipt=ir, total_value_wei=_TOTAL, requester_address=_REQ_ADDR, wallet_map=wallet)
    payees = payee_set_from_tasks(bare)
    auth = _signed_auth(payees, **(auth_kwargs or {}))
    tasks = [dataclasses.replace(t, payment_authorization=auth) for t in bare]
    return tasks, payees


def _store(tmp_path, **kw):
    return PerStageReceiverStore(tmp_path / "receiver.json", **kw)


# ── ingest accepts + stages an authorized task ───────────────────────────────

def test_ingest_accepts_and_stages_authorized_task(tmp_path):
    tasks, payees = _routed()
    store = _store(tmp_path)
    res = ingest_routed_task(store, tasks[0], payees=payees, my_node_id=tasks[0].node_id)
    assert res.accepted, res.reason
    assert res.local_escrow_id == tasks[0].batched_receipt.local_escrow_id
    staged = store.get(res.local_escrow_id)
    assert staged is not None
    assert staged.task.node_id == tasks[0].node_id
    assert staged.payees == payees


def test_both_nodes_stage_independently(tmp_path):
    tasks, payees = _routed()
    store = _store(tmp_path)
    for t in tasks:
        assert ingest_routed_task(store, t, payees=payees, my_node_id=t.node_id).accepted
    assert len(store.all_staged()) == 2
    # distinct idempotency keys ({job}::stage::{node}) for the two nodes
    assert len({s.local_escrow_id for s in store.all_staged()}) == 2


# ── idempotent: a re-delivery never double-stages ────────────────────────────

def test_redelivery_is_idempotent_no_double_stage(tmp_path):
    tasks, payees = _routed()
    store = _store(tmp_path)
    r1 = ingest_routed_task(store, tasks[0], payees=payees)
    r2 = ingest_routed_task(store, tasks[0], payees=payees)  # duplicate delivery
    assert r1.accepted and r2.accepted
    assert r1.local_escrow_id == r2.local_escrow_id
    assert len(store.all_staged()) == 1  # one entry, not two


# ── fail-closed: a rejected task stages NOTHING ──────────────────────────────

def test_no_auth_rejected_nothing_staged(tmp_path):
    tasks, payees = _routed()
    store = _store(tmp_path)
    bare = dataclasses.replace(tasks[0], payment_authorization=None)
    res = ingest_routed_task(store, bare, payees=payees)
    assert not res.accepted
    assert store.all_staged() == []


def test_expired_auth_rejected_nothing_staged(tmp_path):
    tasks, payees = _routed(auth_kwargs={"expiry_delta": -1})
    store = _store(tmp_path)
    assert not ingest_routed_task(store, tasks[0], payees=payees).accepted
    assert store.all_staged() == []


def test_misrouted_task_rejected_nothing_staged(tmp_path):
    tasks, payees = _routed()
    store = _store(tmp_path)
    # deliver node-A's task but claim to be node B → misroute, refuse to stage
    res = ingest_routed_task(store, tasks[0], payees=payees, my_node_id=tasks[1].node_id)
    assert not res.accepted and "misrouted" in res.reason
    assert store.all_staged() == []


# ── store lifecycle ──────────────────────────────────────────────────────────

def test_discard_after_commit(tmp_path):
    tasks, payees = _routed()
    store = _store(tmp_path)
    res = ingest_routed_task(store, tasks[0], payees=payees)
    assert store.discard(res.local_escrow_id) is True
    assert store.get(res.local_escrow_id) is None
    assert store.discard(res.local_escrow_id) is False  # idempotent


def test_persistence_round_trip(tmp_path):
    tasks, payees = _routed()
    s1 = _store(tmp_path)
    for t in tasks:
        ingest_routed_task(s1, t, payees=payees)
    # reload from disk
    s2 = _store(tmp_path)
    assert len(s2.all_staged()) == 2
    by_key = {s.local_escrow_id: s for s in s2.all_staged()}
    orig = {s.local_escrow_id: s for s in s1.all_staged()}
    for k, rec in orig.items():
        assert by_key[k].task.share_wei == rec.task.share_wei
        assert by_key[k].task.batched_receipt.provider_address == rec.task.batched_receipt.provider_address
        assert by_key[k].payees == rec.payees


def test_malformed_disk_entry_skipped_not_raised(tmp_path):
    tasks, payees = _routed()
    s1 = _store(tmp_path)
    ingest_routed_task(s1, tasks[0], payees=payees)
    import json
    p = tmp_path / "receiver.json"
    raw = json.loads(p.read_text())
    raw["staged"]["corrupt::stage::x"] = {"task": {"not": "valid"}, "payees": [], "staged_at": 1}
    p.write_text(json.dumps(raw))
    s2 = _store(tmp_path)  # must not raise
    assert len(s2.all_staged()) == 1  # the good entry survives, the corrupt one skipped


def test_staged_round_trip_dataclass():
    tasks, payees = _routed()
    rec = StagedSettlementTask(task=tasks[0], payees=payees, staged_at=123)
    rec2 = StagedSettlementTask.from_dict(rec.to_dict())
    assert rec2.task.node_id == rec.task.node_id
    assert rec2.payees == rec.payees
    assert rec2.staged_at == 123


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
