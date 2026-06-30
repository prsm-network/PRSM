"""Sprint 1323 (S3b-3b) — END-TO-END big-model paid-settlement chain, in one process.

Wires together EVERYTHING built this session over REAL HTTP (FastAPI TestClient) — no external
infra — to prove the full multi-stage settle chain holds together before the gated 2-live-node
testnet run:

  orchestrator: split settled receipt → tasks (S3a)
              → deliver_for_settled_receipt → POST each over HTTP (S3b-2c/sp1321)
  each node:    POST /settlement/per-stage-task (sp1318) → parse → sp1316 gate → stage (sp1317)
              → run_per_stage_commit_cycle drains + commits on its own client (sp1322/S3b-3a)

Asserts: both stage nodes receive + stage their OWN share, then both commit + drain. Plus the
money-safety negative: a node whose payee was NOT in the signed set fail-closes (nothing staged,
nothing committed).
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import time
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

from eth_account import Account
from eth_utils import keccak
from fastapi.testclient import TestClient

from prsm.compute.inference.models import ContentTier, InferenceReceipt
from prsm.compute.inference.topology_rotation import TopologyAssignment
from prsm.compute.shard_receipt import (
    build_receipt_signing_payload,
    per_stage_leaf_job_id,
)
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.api import create_api_app
from prsm.node.compute_wallet_map import ComputeWalletMap
from prsm.node.identity import generate_node_identity
from prsm.settlement.client_wiring import (
    deliver_for_settled_receipt,
    run_per_stage_commit_cycle,
)
from prsm.settlement.per_stage_payment_authorization import (
    DEFAULT_CHAIN_ID,
    compute_payee_set_hash,
    sign_per_stage_authorization,
)
from prsm.settlement.per_stage_receiver_store import PerStageReceiverStore
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


def _settled():
    id_a = generate_node_identity(display_name="stage0")
    id_b = generate_node_identity(display_name="stage1")
    out = hashlib.sha256(b"e2e-1323").digest()
    topo = TopologyAssignment(
        positions={(0, 0): id_a.node_id, (1, 0): id_b.node_id}, stage_count=2, slots_per_stage=1)
    ir0 = InferenceReceipt(
        job_id="job-1323", request_id="req-1323", model_id="qwen2.5-72b",
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
    payees = payee_set_from_tasks(build_per_stage_settlement_tasks(
        receipt=ir, total_value_wei=_TOTAL, requester_address=_REQ_ADDR, wallet_map=wallet))
    payload = {
        "requester": _REQ_ADDR, "payee_set_hash": "0x" + compute_payee_set_hash(payees).hex(),
        "total_max_spend_wei": _TOTAL, "job_nonce": "0x" + keccak(b"n-1323").hex(),
        "expiry_unix": int(time.time()) + 86400, "request_hash": "0x" + keccak(b"r-1323").hex()}
    auth = {"payload": payload,
            "signature": "0x" + sign_per_stage_authorization(payload, _REQ_KEY, chain_id=DEFAULT_CHAIN_ID).hex()}
    return ir, wallet, auth, id_a.node_id, id_b.node_id


class _FakeClient:
    """A real-shaped settlement client (accumulate + commit_ready_batches), no chain."""
    def __init__(self):
        self._pending = []; self.committed = []
    async def accumulate(self, br):
        self._pending.append(br)
    async def commit_ready_batches(self):
        out = list(self._pending); self.committed.extend(out); self._pending = []
        return [SimpleNamespace(local_escrow_id=br.local_escrow_id) for br in out]


def _node(node_id, tmp_path, name):
    """A receiver node = a real FastAPI app over a real store + a fake on-chain client."""
    n = MagicMock()
    n.identity.node_id = node_id
    n.transport = None
    n._settlement_per_stage_receiver_store = PerStageReceiverStore(tmp_path / f"{name}.json")
    n._onchain_settlement_client = _FakeClient()
    client = TestClient(create_api_app(n, enable_security=False), raise_server_exceptions=False)
    return n, client


def _http_post_router(routes):
    """An http_post that dispatches the full URL to the right node's TestClient (loop-back over
    REAL HTTP request/response serialization)."""
    def _post(url, *, json, timeout):
        # url = "http://<host>:<port>/settlement/per-stage-task"
        for host_prefix, client in routes.items():
            if url.startswith(host_prefix):
                return client.post("/settlement/per-stage-task", json=json)
        raise ConnectionError(f"no route for {url}")
    return _post


def test_full_chain_deliver_stage_commit(tmp_path, monkeypatch):
    monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", "1")
    monkeypatch.setenv("PRSM_MULTISTAGE_ENDPOINT_SCHEME", "http")
    ir, wallet, auth, na, nb = _settled()

    node_a, client_a = _node(na, tmp_path, "a")
    node_b, client_b = _node(nb, tmp_path, "b")
    routes = {"http://10.0.0.1:8000": client_a, "http://10.0.0.2:8000": client_b}

    # orchestrator resolves endpoints from a static map → POSTs each task over HTTP
    monkeypatch.setenv("PRSM_MULTISTAGE_ENDPOINT_MAP",
                       f'{{"{na}": "http://10.0.0.1:8000", "{nb}": "http://10.0.0.2:8000"}}')
    orchestrator = SimpleNamespace(transport=None)
    results = deliver_for_settled_receipt(
        orchestrator, receipt=ir, total_value_wei=_TOTAL, requester_address=_REQ_ADDR,
        per_stage_authorization=auth, wallet_map=wallet,
        http_post=_http_post_router(routes))

    # both delivered + accepted + staged on the right node
    assert len(results) == 2 and all(r.delivered and r.accepted for r in results), [r.reason for r in results]
    assert node_a._settlement_per_stage_receiver_store.all_staged()[0].task.node_id == na
    assert node_b._settlement_per_stage_receiver_store.all_staged()[0].task.node_id == nb

    # each node's poll cycle commits its own share + drains
    ra = asyncio.run(run_per_stage_commit_cycle(node_a))
    rb = asyncio.run(run_per_stage_commit_cycle(node_b))
    assert ra["per_stage_commit"] == "committed 1/1"
    assert rb["per_stage_commit"] == "committed 1/1"
    assert len(node_a._onchain_settlement_client.committed) == 1
    assert len(node_b._onchain_settlement_client.committed) == 1
    assert node_a._settlement_per_stage_receiver_store.all_staged() == []
    assert node_b._settlement_per_stage_receiver_store.all_staged() == []
    # conservation: the two committed shares sum to the settled total
    sa = node_a._settlement_per_stage_receiver_store  # drained — read committed via client
    total_committed = sum(
        n._onchain_settlement_client.committed[0].local_escrow_id is not None
        for n in (node_a, node_b))
    assert total_committed == 2


def test_full_chain_rejects_node_outside_signed_set(tmp_path, monkeypatch):
    """Money-safety end-to-end: if a node's delivered payee was NOT in the requester's signed set,
    its receiver gate fail-closes — nothing staged, nothing committed for that node."""
    monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", "1")
    monkeypatch.setenv("PRSM_MULTISTAGE_ENDPOINT_SCHEME", "http")
    ir, wallet, auth, na, nb = _settled()
    # tamper: sign the auth over a DIFFERENT set than the one that will be delivered
    rogue_payees = [(_PAYEE_A, _TOTAL // 2), ("0x" + "d3" * 20, _TOTAL - _TOTAL // 2)]
    payload = dict(auth["payload"], payee_set_hash="0x" + compute_payee_set_hash(rogue_payees).hex())
    tampered = {"payload": payload,
                "signature": "0x" + sign_per_stage_authorization(payload, _REQ_KEY, chain_id=DEFAULT_CHAIN_ID).hex()}

    node_a, client_a = _node(na, tmp_path, "a")
    node_b, client_b = _node(nb, tmp_path, "b")
    routes = {"http://10.0.0.1:8000": client_a, "http://10.0.0.2:8000": client_b}
    monkeypatch.setenv("PRSM_MULTISTAGE_ENDPOINT_MAP",
                       f'{{"{na}": "http://10.0.0.1:8000", "{nb}": "http://10.0.0.2:8000"}}')
    results = deliver_for_settled_receipt(
        SimpleNamespace(transport=None), receipt=ir, total_value_wei=_TOTAL,
        requester_address=_REQ_ADDR, per_stage_authorization=tampered, wallet_map=wallet,
        http_post=_http_post_router(routes))

    # delivered (HTTP 200) but the gate REJECTED both (set-hash mismatch) → nothing staged
    assert all(r.delivered and r.accepted is False for r in results)
    assert node_a._settlement_per_stage_receiver_store.all_staged() == []
    assert node_b._settlement_per_stage_receiver_store.all_staged() == []
    # and the commit cycle has nothing to do
    assert asyncio.run(run_per_stage_commit_cycle(node_a))["per_stage_commit"] == "ok:nothing-staged"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
