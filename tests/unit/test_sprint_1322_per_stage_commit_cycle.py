"""Sprint 1322 (S3b-3b) — node-side per-stage commit cycle (run_per_stage_commit_cycle).

The node-side scheduler: drain the receiver store + commit each staged share on the node's own
settlement client, run from the settlement poll loop. GATED (PRSM_MULTISTAGE_SETTLEMENT) +
fail-soft. Tests every skip state + a real drain with a fake client + store.
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import time
from decimal import Decimal
from types import SimpleNamespace

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
from prsm.settlement.client_wiring import run_per_stage_commit_cycle
from prsm.settlement.per_stage_payment_authorization import (
    DEFAULT_CHAIN_ID,
    compute_payee_set_hash,
    sign_per_stage_authorization,
)
from prsm.settlement.per_stage_receiver_store import (
    PerStageReceiverStore,
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


def _stage_into(store):
    id_a = generate_node_identity(display_name="stage0")
    id_b = generate_node_identity(display_name="stage1")
    out = hashlib.sha256(b"out-1322").digest()
    topo = TopologyAssignment(
        positions={(0, 0): id_a.node_id, (1, 0): id_b.node_id}, stage_count=2, slots_per_stage=1)
    ir0 = InferenceReceipt(
        job_id="job-1322", request_id="req-1322", model_id="qwen2.5-72b",
        content_tier=ContentTier.A, privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0, tee_type=TEEType.SOFTWARE, tee_attestation=b"att",
        output_hash=out, duration_seconds=1.0, cost_ftns=Decimal("1.0"),
        settler_signature=b"sig", settler_node_id=id_a.node_id, topology_assignment=topo)
    job = per_stage_leaf_job_id(ir0.request_id); oh = out.hex()

    def _sig(idn, idx):
        p = build_receipt_signing_payload(job_id=job, shard_index=idx, output_hash=oh, executed_at_unix=1_700_000_000)
        return NodeSignatureMaterial(pubkey_b64=idn.public_key_b64, signature=idn.sign(p),
                                     stage_index=idx, output_hash=oh, executed_at_unix=1_700_000_000)
    ir = dataclasses.replace(ir0, per_stage_settlement_signatures={
        id_a.node_id: _sig(id_a, 0), id_b.node_id: _sig(id_b, 1)})
    wallet = ComputeWalletMap.from_mapping({id_a.node_id: _PAYEE_A, id_b.node_id: _PAYEE_B})
    bare = build_per_stage_settlement_tasks(
        receipt=ir, total_value_wei=_TOTAL, requester_address=_REQ_ADDR, wallet_map=wallet)
    payees = payee_set_from_tasks(bare)
    payload = {
        "requester": _REQ_ADDR, "payee_set_hash": "0x" + compute_payee_set_hash(payees).hex(),
        "total_max_spend_wei": _TOTAL, "job_nonce": "0x" + keccak(b"n").hex(),
        "expiry_unix": int(time.time()) + 86400, "request_hash": "0x" + keccak(b"r").hex()}
    auth = {"payload": payload,
            "signature": "0x" + sign_per_stage_authorization(payload, _REQ_KEY, chain_id=DEFAULT_CHAIN_ID).hex()}
    for t in (dataclasses.replace(t, payment_authorization=auth) for t in bare):
        ingest_routed_task(store, t, payees=payees, my_node_id=t.node_id)


class _FakeClient:
    def __init__(self, *, fail=False):
        self._pending = []; self.committed = []; self._fail = fail
    async def accumulate(self, br):
        self._pending.append(br)
    async def commit_ready_batches(self):
        if self._fail:
            raise RuntimeError("private_key required")  # view-only client analog
        out = list(self._pending); self.committed.extend(out); self._pending = []
        return [SimpleNamespace(local_escrow_id=br.local_escrow_id) for br in out]


def _node(tmp_path, *, client):
    # sp1329 — run_per_stage_commit_cycle now resolves a DEDICATED per-stage client
    # (count_threshold=1) cached on _onchain_per_stage_settlement_client.
    store = PerStageReceiverStore(tmp_path / "rx.json")
    return SimpleNamespace(_settlement_per_stage_receiver_store=store,
                           _onchain_per_stage_settlement_client=client), store


# ── skip states ───────────────────────────────────────────────────────────────

def test_skipped_when_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("PRSM_MULTISTAGE_SETTLEMENT", raising=False)
    node, _ = _node(tmp_path, client=_FakeClient())
    r = asyncio.run(run_per_stage_commit_cycle(node))
    assert r["per_stage_commit"] == "skipped:disabled"


def test_skipped_no_client(tmp_path, monkeypatch):
    monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", "1")
    node, _ = _node(tmp_path, client=None)
    r = asyncio.run(run_per_stage_commit_cycle(node))
    assert r["per_stage_commit"] == "skipped:no-client"


def test_nothing_staged(tmp_path, monkeypatch):
    monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", "1")
    node, _ = _node(tmp_path, client=_FakeClient())
    r = asyncio.run(run_per_stage_commit_cycle(node))
    assert r["per_stage_commit"] == "ok:nothing-staged"


# ── real drain ──────────────────────────────────────────────────────────────

def test_commits_staged_and_drains(tmp_path, monkeypatch):
    monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", "1")
    client = _FakeClient()
    node, store = _node(tmp_path, client=client)
    _stage_into(store)
    assert len(store.all_staged()) == 2
    r = asyncio.run(run_per_stage_commit_cycle(node))
    assert r["per_stage_commit"] == "committed 2/2"
    assert len(client.committed) == 2
    assert store.all_staged() == []  # drained


def test_view_only_client_is_inert_tasks_retained(tmp_path, monkeypatch):
    monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", "1")
    client = _FakeClient(fail=True)  # commit raises (view-only analog)
    node, store = _node(tmp_path, client=client)
    _stage_into(store)
    r = asyncio.run(run_per_stage_commit_cycle(node))
    assert r["per_stage_commit"] == "committed 0/2"
    assert len(store.all_staged()) == 2  # nothing committed → all retained for a funded retry


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
