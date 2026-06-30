"""Sprint 1320 (S3b-3a) — node-side commit driver: drain the receiver store → per-node commit.

Each stage node commits its OWN staged share-batch on its OWN settlement client (Design A:
msg.sender == provider). The driver RE-VERIFIES membership against the STORED FULL payee set at
commit time (so an auth that expired between stage + commit is caught), accumulates the
node-signed BatchedReceipt, drives commit, and discards on success. FAIL-SOFT: a gate reject /
client bind-mismatch / commit error leaves the share unsettled (retryable), never raises.

Tested with a FAKE client (records accumulate + commit, NO chain) through the REAL splitter +
REAL signed auth. Money path is gated behind a real funded client + a testnet 2-node proof
before mainnet (not exercised here).
"""
from __future__ import annotations

import asyncio
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
    commit_staged_task,
    drain_and_commit_staged,
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
    out = hashlib.sha256(b"out-1320").digest()
    topo = TopologyAssignment(
        positions={(0, 0): node_a_id, (1, 0): node_b_id}, stage_count=2, slots_per_stage=1)
    return InferenceReceipt(
        job_id="job-1320", request_id="req-1320", model_id="qwen2.5-72b",
        content_tier=ContentTier.A, privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0, tee_type=TEEType.SOFTWARE, tee_attestation=b"att",
        output_hash=out, duration_seconds=1.0, cost_ftns=Decimal("1.0"),
        settler_signature=b"sig", settler_node_id=node_a_id, topology_assignment=topo)


def _sig(identity, *, job_id, stage_index, output_hex):
    payload = build_receipt_signing_payload(
        job_id=job_id, shard_index=stage_index, output_hash=output_hex, executed_at_unix=1_700_000_000)
    return NodeSignatureMaterial(
        pubkey_b64=identity.public_key_b64, signature=identity.sign(payload),
        stage_index=stage_index, output_hash=output_hex, executed_at_unix=1_700_000_000)


def _signed_auth(payees, *, expiry_delta=86400):
    payload = {
        "requester": _REQ_ADDR,
        "payee_set_hash": "0x" + compute_payee_set_hash(payees).hex(),
        "total_max_spend_wei": _TOTAL,
        "job_nonce": "0x" + keccak(b"n-1320").hex(),
        "expiry_unix": int(time.time()) + expiry_delta,
        "request_hash": "0x" + keccak(b"r-1320").hex(),
    }
    sig = sign_per_stage_authorization(payload, _REQ_KEY, chain_id=DEFAULT_CHAIN_ID)
    return {"payload": payload, "signature": "0x" + sig.hex()}


def _staged(tmp_path, *, expiry_delta=86400):
    """Build a receiver store with both nodes' tasks staged. Returns (store, node_a, node_b)."""
    id_a = generate_node_identity(display_name="stage0")
    id_b = generate_node_identity(display_name="stage1")
    ir0 = _make_ir(node_a_id=id_a.node_id, node_b_id=id_b.node_id)
    out = ir0.output_hash.hex(); job = per_stage_leaf_job_id(ir0.request_id)
    sigs = {
        id_a.node_id: _sig(id_a, job_id=job, stage_index=0, output_hex=out),
        id_b.node_id: _sig(id_b, job_id=job, stage_index=1, output_hex=out),
    }
    ir = dataclasses.replace(ir0, per_stage_settlement_signatures=sigs)
    wallet = ComputeWalletMap.from_mapping({id_a.node_id: _PAYEE_A, id_b.node_id: _PAYEE_B})
    bare = build_per_stage_settlement_tasks(
        receipt=ir, total_value_wei=_TOTAL, requester_address=_REQ_ADDR, wallet_map=wallet)
    payees = payee_set_from_tasks(bare)
    auth = _signed_auth(payees, expiry_delta=expiry_delta)
    tasks = [dataclasses.replace(t, payment_authorization=auth) for t in bare]
    store = PerStageReceiverStore(tmp_path / "rx.json")
    for t in tasks:
        ingest_routed_task(store, t, payees=payees, my_node_id=t.node_id)
    return store, id_a.node_id, id_b.node_id


class _FakeClient:
    """Records accumulate + commits the accumulated receipt on the next commit poll (no chain)."""

    def __init__(self, *, fail_accumulate=False, fail_commit=False):
        self._pending = []
        self.committed = []
        self.accumulated = []
        self._fail_accumulate = fail_accumulate
        self._fail_commit = fail_commit

    async def accumulate(self, br):
        if self._fail_accumulate:
            raise ValueError("bound-address mismatch")
        self.accumulated.append(br)
        self._pending.append(br)

    async def commit_ready_batches(self):
        if self._fail_commit:
            raise RuntimeError("rpc down")
        out = [SimpleBatch(br.local_escrow_id) for br in self._pending]
        self.committed.extend(out)
        self._pending = []
        return out


class SimpleBatch:
    def __init__(self, lid):
        self.local_escrow_id = lid


# ── happy path: drain commits both, discards both ───────────────────────────

def test_drain_commits_and_discards(tmp_path):
    store, na, nb = _staged(tmp_path)
    clients = {na: _FakeClient(), nb: _FakeClient()}
    results = asyncio.run(drain_and_commit_staged(
        store, client_for_node=lambda nid: clients[nid]))
    assert len(results) == 2
    assert all(r.committed for r in results), [r.reason for r in results]
    # each node's own client accumulated exactly its own share-batch
    assert len(clients[na].accumulated) == 1 and len(clients[nb].accumulated) == 1
    # committed tasks are discarded → store drained
    assert store.all_staged() == []


def test_redrain_after_commit_is_noop(tmp_path):
    store, na, nb = _staged(tmp_path)
    clients = {na: _FakeClient(), nb: _FakeClient()}
    asyncio.run(drain_and_commit_staged(store, client_for_node=lambda nid: clients[nid]))
    again = asyncio.run(drain_and_commit_staged(store, client_for_node=lambda nid: clients[nid]))
    assert again == []  # nothing left staged


# ── fail-soft: a commit error leaves the task staged for retry ───────────────

def test_commit_error_retains_task(tmp_path):
    store, na, nb = _staged(tmp_path)
    # node A's client fails to commit; node B's succeeds
    clients = {na: _FakeClient(fail_commit=True), nb: _FakeClient()}
    results = asyncio.run(drain_and_commit_staged(
        store, client_for_node=lambda nid: clients[nid]))
    by_node = {r.node_id: r for r in results}
    assert by_node[na].committed is False and "commit error" in by_node[na].reason
    assert by_node[nb].committed is True
    # A retained (retryable), B discarded
    remaining = {s.task.node_id for s in store.all_staged()}
    assert remaining == {na}


def test_accumulate_bind_mismatch_is_fail_soft(tmp_path):
    store, na, nb = _staged(tmp_path)
    clients = {na: _FakeClient(fail_accumulate=True), nb: _FakeClient(fail_accumulate=True)}
    results = asyncio.run(drain_and_commit_staged(
        store, client_for_node=lambda nid: clients[nid]))
    assert all(not r.committed for r in results)
    assert len(store.all_staged()) == 2  # nothing committed → nothing discarded


# ── money-safety: an auth that expired between stage + commit is REJECTED ─────

def test_expired_between_stage_and_commit_rejected(tmp_path):
    # stage while valid (the ingest gate passes), then commit AFTER expiry → re-check fails.
    store, na, nb = _staged(tmp_path, expiry_delta=120)
    clients = {na: _FakeClient(), nb: _FakeClient()}
    # commit at a time PAST the auth expiry
    future = time.time() + 10_000
    results = asyncio.run(drain_and_commit_staged(
        store, client_for_node=lambda nid: clients[nid], now_unix=future))
    assert all(not r.committed and "auth re-check failed" in r.reason for r in results)
    # nothing accumulated on any client (the gate ran BEFORE accumulate)
    assert clients[na].accumulated == [] and clients[nb].accumulated == []
    # tasks retained (not discarded) — but they will keep failing; operator surfaces this
    assert len(store.all_staged()) == 2


# ── single-task driver direct ────────────────────────────────────────────────

def test_commit_staged_task_direct(tmp_path):
    store, na, _nb = _staged(tmp_path)
    staged = [s for s in store.all_staged() if s.task.node_id == na][0]
    client = _FakeClient()
    res = asyncio.run(commit_staged_task(staged, client=client))
    assert res.committed and res.node_id == na
    assert res.committed_batch is not None
    assert res.local_escrow_id == staged.local_escrow_id


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
