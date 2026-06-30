"""Sprint 1318 (S3b-2b) — HTTP delivery endpoint + gated node-side receiver-store resolver.

POST /settlement/per-stage-task lets a settling orchestrator deliver each stage node its routed
task (+ the full payee set). The node FAIL-CLOSED verifies its own membership (sp1316 gate) then
STAGES the authorized task (sp1317 store) — no on-chain commit here. Gated by
PRSM_MULTISTAGE_SETTLEMENT (default-off → 503). Tests the parser, the gated resolver, and the
endpoint end-to-end through the REAL splitter + REAL signed auth.
"""
from __future__ import annotations

import dataclasses
import hashlib
import time
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
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
    multistage_settlement_enabled,
    resolve_per_stage_receiver_store,
)
from prsm.settlement.per_stage_payment_authorization import (
    DEFAULT_CHAIN_ID,
    compute_payee_set_hash,
    sign_per_stage_authorization,
)
from prsm.settlement.per_stage_receiver_store import (
    PerStageReceiverStore,
    parse_delivery_request,
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
        job_id="job-1318", request_id="req-1318", model_id="qwen2.5-72b",
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


def _signed_auth(payees):
    payload = {
        "requester": _REQ_ADDR,
        "payee_set_hash": "0x" + compute_payee_set_hash(payees).hex(),
        "total_max_spend_wei": _TOTAL,
        "job_nonce": "0x" + keccak(b"nonce-1318").hex(),
        "expiry_unix": int(time.time()) + 86400,
        "request_hash": "0x" + keccak(b"req-1318").hex(),
    }
    sig = sign_per_stage_authorization(payload, _REQ_KEY, chain_id=DEFAULT_CHAIN_ID)
    return {"payload": payload, "signature": "0x" + sig.hex()}


def _routed():
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
    auth = _signed_auth(payees)
    tasks = [dataclasses.replace(t, payment_authorization=auth) for t in bare]
    return tasks, payees


def _body(task, payees):
    return {"task": task.to_dict(), "payees": [[p, str(s)] for p, s in payees]}


# ── the gate flag ─────────────────────────────────────────────────────────────

def test_gate_default_off(monkeypatch):
    monkeypatch.delenv("PRSM_MULTISTAGE_SETTLEMENT", raising=False)
    assert multistage_settlement_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "YES", "on"])
def test_gate_truthy(monkeypatch, val):
    monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", val)
    assert multistage_settlement_enabled() is True


# ── parse_delivery_request ─────────────────────────────────────────────────────

def test_parse_round_trips_a_real_delivery():
    tasks, payees = _routed()
    task, parsed = parse_delivery_request(_body(tasks[0], payees))
    assert task.node_id == tasks[0].node_id
    assert parsed == payees


@pytest.mark.parametrize("bad", [
    [], "x", {"payees": []}, {"task": {}, "payees": "notalist"},
    {"task": "notobj", "payees": []}, {"task": {"bad": 1}, "payees": []},
    {"task": {}, "payees": [["addr"]]},
])
def test_parse_rejects_malformed(bad):
    tasks, payees = _routed()
    # for the {"task": {}} cases supply a parseable task to isolate the payees failure
    if isinstance(bad, dict) and bad.get("task") == {}:
        bad = {**bad, "task": tasks[0].to_dict()}
    with pytest.raises(ValueError):
        parse_delivery_request(bad)


# ── gated resolver ─────────────────────────────────────────────────────────────

def test_resolver_none_when_gate_off(monkeypatch):
    monkeypatch.delenv("PRSM_MULTISTAGE_SETTLEMENT", raising=False)
    node = MagicMock()
    node._settlement_per_stage_receiver_store = None
    assert resolve_per_stage_receiver_store(node) is None


def test_resolver_builds_and_caches_when_on(monkeypatch, tmp_path):
    monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", "1")
    monkeypatch.setenv("PRSM_MULTISTAGE_RECEIVER_STORE_FILE", str(tmp_path / "rx.json"))
    node = MagicMock()
    node._settlement_per_stage_receiver_store = None
    s1 = resolve_per_stage_receiver_store(node)
    assert isinstance(s1, PerStageReceiverStore)
    s2 = resolve_per_stage_receiver_store(node)
    assert s2 is s1  # cached, not rebuilt


# ── the endpoint ───────────────────────────────────────────────────────────────

def _client(*, gate_on, my_node_id, tmp_path, monkeypatch):
    if gate_on:
        monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", "1")
    else:
        monkeypatch.delenv("PRSM_MULTISTAGE_SETTLEMENT", raising=False)
    node = MagicMock()
    node.identity.node_id = my_node_id
    # attach a real receiver store (the resolver returns it when the gate is on)
    node._settlement_per_stage_receiver_store = PerStageReceiverStore(tmp_path / "rx.json")
    client = TestClient(create_api_app(node, enable_security=False),
                        raise_server_exceptions=False)
    return client, node


def test_endpoint_503_when_gate_off(tmp_path, monkeypatch):
    tasks, payees = _routed()
    client, _ = _client(gate_on=False, my_node_id=tasks[0].node_id, tmp_path=tmp_path, monkeypatch=monkeypatch)
    r = client.post("/settlement/per-stage-task", json=_body(tasks[0], payees))
    assert r.status_code == 503


def test_endpoint_accepts_and_stages(tmp_path, monkeypatch):
    tasks, payees = _routed()
    client, node = _client(gate_on=True, my_node_id=tasks[0].node_id, tmp_path=tmp_path, monkeypatch=monkeypatch)
    r = client.post("/settlement/per-stage-task", json=_body(tasks[0], payees))
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["accepted"] is True
    assert d["local_escrow_id"] == tasks[0].batched_receipt.local_escrow_id
    # actually staged in the node's store
    assert node._settlement_per_stage_receiver_store.get(d["local_escrow_id"]) is not None


def test_endpoint_rejects_misrouted_task(tmp_path, monkeypatch):
    tasks, payees = _routed()
    # node identity is node B but we POST node A's task
    client, node = _client(gate_on=True, my_node_id=tasks[1].node_id, tmp_path=tmp_path, monkeypatch=monkeypatch)
    r = client.post("/settlement/per-stage-task", json=_body(tasks[0], payees))
    assert r.status_code == 200
    d = r.json()
    assert d["accepted"] is False and "misrouted" in d["reason"]
    assert node._settlement_per_stage_receiver_store.all_staged() == []


def test_endpoint_rejects_unauthorized(tmp_path, monkeypatch):
    tasks, payees = _routed()
    client, node = _client(gate_on=True, my_node_id=tasks[0].node_id, tmp_path=tmp_path, monkeypatch=monkeypatch)
    # strip the auth → fail-closed reject
    body = _body(dataclasses.replace(tasks[0], payment_authorization=None), payees)
    r = client.post("/settlement/per-stage-task", json=body)
    assert r.status_code == 200
    assert r.json()["accepted"] is False
    assert node._settlement_per_stage_receiver_store.all_staged() == []


def test_endpoint_422_on_malformed_body(tmp_path, monkeypatch):
    tasks, _payees = _routed()
    client, _ = _client(gate_on=True, my_node_id=tasks[0].node_id, tmp_path=tmp_path, monkeypatch=monkeypatch)
    r = client.post("/settlement/per-stage-task", json={"task": tasks[0].to_dict(), "payees": "notalist"})
    assert r.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
