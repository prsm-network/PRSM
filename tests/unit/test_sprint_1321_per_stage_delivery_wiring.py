"""Sprint 1321 (S3b-3b) — node-aware orchestrator delivery wiring.

build_per_stage_endpoint_resolver maps node_id → base URL (static map wins over the transport-peer
fallback; None on miss so the FAIL-SOFT sender records undelivered). deliver_for_settled_receipt
is the gated node entrypoint: split a settled multi-stage receipt + deliver each per-node task to
its stage node, returning [] when PRSM_MULTISTAGE_SETTLEMENT is off.
"""
from __future__ import annotations

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
from prsm.settlement.client_wiring import (
    build_per_stage_endpoint_resolver,
    deliver_for_settled_receipt,
)
from prsm.settlement.per_stage_payment_authorization import (
    DEFAULT_CHAIN_ID,
    compute_payee_set_hash,
    sign_per_stage_authorization,
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


# ── endpoint resolver ─────────────────────────────────────────────────────────

class _FakeTransport:
    def __init__(self, peers):
        self._peers = peers  # node_id -> address "host:port"

    def get_peer(self, node_id):
        addr = self._peers.get(node_id)
        return SimpleNamespace(address=addr) if addr else None


def test_resolver_static_map_wins(monkeypatch):
    monkeypatch.setenv("PRSM_MULTISTAGE_ENDPOINT_MAP",
                       '{"nodeA": "https://a.example:9000"}')
    node = SimpleNamespace(transport=_FakeTransport({"nodeA": "10.0.0.1:8000"}))
    resolve = build_per_stage_endpoint_resolver(node)
    assert resolve("nodeA") == "https://a.example:9000"  # static beats transport


def test_resolver_transport_fallback(monkeypatch):
    monkeypatch.delenv("PRSM_MULTISTAGE_ENDPOINT_MAP", raising=False)
    monkeypatch.setenv("PRSM_MULTISTAGE_ENDPOINT_SCHEME", "http")
    node = SimpleNamespace(transport=_FakeTransport({"nodeB": "10.0.0.2:8001"}))
    resolve = build_per_stage_endpoint_resolver(node)
    assert resolve("nodeB") == "http://10.0.0.2:8001"


def test_resolver_none_on_miss(monkeypatch):
    monkeypatch.delenv("PRSM_MULTISTAGE_ENDPOINT_MAP", raising=False)
    node = SimpleNamespace(transport=_FakeTransport({}))
    assert build_per_stage_endpoint_resolver(node)("ghost") is None


def test_resolver_none_when_no_backends(monkeypatch):
    monkeypatch.delenv("PRSM_MULTISTAGE_ENDPOINT_MAP", raising=False)
    node = SimpleNamespace(transport=None)
    assert build_per_stage_endpoint_resolver(node)("anyone") is None


def test_resolver_port_override(monkeypatch):
    monkeypatch.delenv("PRSM_MULTISTAGE_ENDPOINT_MAP", raising=False)
    monkeypatch.setenv("PRSM_MULTISTAGE_ENDPOINT_SCHEME", "http")
    monkeypatch.setenv("PRSM_MULTISTAGE_ENDPOINT_PORT", "8443")
    node = SimpleNamespace(transport=_FakeTransport({"n": "host:8000"}))
    assert build_per_stage_endpoint_resolver(node)("n") == "http://host:8443"


def test_resolver_malformed_static_map_falls_back(monkeypatch):
    monkeypatch.setenv("PRSM_MULTISTAGE_ENDPOINT_MAP", "{not json")
    monkeypatch.setenv("PRSM_MULTISTAGE_ENDPOINT_SCHEME", "http")
    node = SimpleNamespace(transport=_FakeTransport({"n": "host:8000"}))
    assert build_per_stage_endpoint_resolver(node)("n") == "http://host:8000"


# ── deliver_for_settled_receipt (gated) ───────────────────────────────────────

def _settled_with_auth():
    id_a = generate_node_identity(display_name="stage0")
    id_b = generate_node_identity(display_name="stage1")
    out = hashlib.sha256(b"out-1321").digest()
    topo = TopologyAssignment(
        positions={(0, 0): id_a.node_id, (1, 0): id_b.node_id}, stage_count=2, slots_per_stage=1)
    ir0 = InferenceReceipt(
        job_id="job-1321", request_id="req-1321", model_id="qwen2.5-72b",
        content_tier=ContentTier.A, privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0, tee_type=TEEType.SOFTWARE, tee_attestation=b"att",
        output_hash=out, duration_seconds=1.0, cost_ftns=Decimal("1.0"),
        settler_signature=b"sig", settler_node_id=id_a.node_id, topology_assignment=topo)
    job = per_stage_leaf_job_id(ir0.request_id); oh = out.hex()

    def _sig(idn, idx):
        p = build_receipt_signing_payload(job_id=job, shard_index=idx, output_hash=oh, executed_at_unix=1_700_000_000)
        return NodeSignatureMaterial(pubkey_b64=idn.public_key_b64, signature=idn.sign(p),
                                     stage_index=idx, output_hash=oh, executed_at_unix=1_700_000_000)
    sigs = {id_a.node_id: _sig(id_a, 0), id_b.node_id: _sig(id_b, 1)}
    ir = dataclasses.replace(ir0, per_stage_settlement_signatures=sigs)
    wallet = ComputeWalletMap.from_mapping({id_a.node_id: _PAYEE_A, id_b.node_id: _PAYEE_B})
    payees = payee_set_from_tasks(build_per_stage_settlement_tasks(
        receipt=ir, total_value_wei=_TOTAL, requester_address=_REQ_ADDR, wallet_map=wallet))
    payload = {
        "requester": _REQ_ADDR, "payee_set_hash": "0x" + compute_payee_set_hash(payees).hex(),
        "total_max_spend_wei": _TOTAL, "job_nonce": "0x" + keccak(b"n").hex(),
        "expiry_unix": int(time.time()) + 86400, "request_hash": "0x" + keccak(b"r").hex()}
    auth = {"payload": payload,
            "signature": "0x" + sign_per_stage_authorization(payload, _REQ_KEY, chain_id=DEFAULT_CHAIN_ID).hex()}
    return ir, wallet, auth, id_a.node_id, id_b.node_id


class _Resp:
    status_code = 200
    def __init__(self, body): self._b = body
    def json(self): return self._b


def test_deliver_gated_off_returns_empty(monkeypatch):
    monkeypatch.delenv("PRSM_MULTISTAGE_SETTLEMENT", raising=False)
    ir, wallet, auth, _a, _b = _settled_with_auth()
    node = SimpleNamespace(transport=_FakeTransport({}))
    out = deliver_for_settled_receipt(
        node, receipt=ir, total_value_wei=_TOTAL, requester_address=_REQ_ADDR,
        per_stage_authorization=auth, wallet_map=wallet,
        http_post=lambda *a, **k: _Resp({"accepted": True}))
    assert out == []


def test_deliver_gated_on_fans_out(monkeypatch):
    monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", "1")
    monkeypatch.setenv("PRSM_MULTISTAGE_ENDPOINT_SCHEME", "http")
    ir, wallet, auth, na, nb = _settled_with_auth()
    node = SimpleNamespace(transport=_FakeTransport({na: "10.0.0.1:8000", nb: "10.0.0.2:8000"}))
    seen = []
    def _post(url, *, json, timeout):
        seen.append(url)
        return _Resp({"accepted": True, "reason": "ok", "local_escrow_id": json["task"]["batched_receipt"]["local_escrow_id"]})
    out = deliver_for_settled_receipt(
        node, receipt=ir, total_value_wei=_TOTAL, requester_address=_REQ_ADDR,
        per_stage_authorization=auth, wallet_map=wallet, http_post=_post)
    assert len(out) == 2 and all(r.delivered and r.accepted for r in out)
    assert sorted(seen) == ["http://10.0.0.1:8000/settlement/per-stage-task",
                            "http://10.0.0.2:8000/settlement/per-stage-task"]


def test_deliver_unmapped_node_undelivered(monkeypatch):
    monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", "1")
    monkeypatch.setenv("PRSM_MULTISTAGE_ENDPOINT_SCHEME", "http")
    ir, wallet, auth, na, nb = _settled_with_auth()
    node = SimpleNamespace(transport=_FakeTransport({na: "10.0.0.1:8000"}))  # nb unmapped
    out = deliver_for_settled_receipt(
        node, receipt=ir, total_value_wei=_TOTAL, requester_address=_REQ_ADDR,
        per_stage_authorization=auth, wallet_map=wallet,
        http_post=lambda *a, **k: _Resp({"accepted": True, "reason": "ok", "local_escrow_id": "x"}))
    by_node = {r.node_id: r for r in out}
    assert by_node[na].delivered is True
    assert by_node[nb].delivered is False and "no resolvable endpoint" in by_node[nb].reason


def test_deliver_self_task_ingested_in_process(monkeypatch, tmp_path):
    """sp1327 — the orchestrator's OWN stage task is ingested IN-PROCESS (local receiver store),
    NOT self-HTTP-POSTed (that re-entrant POST to localhost:8000 deadlocked inside the inference
    request handler during the live GO). Only FOREIGN tasks go over HTTP."""
    monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", "1")
    monkeypatch.setenv("PRSM_MULTISTAGE_ENDPOINT_SCHEME", "http")
    from prsm.settlement.per_stage_receiver_store import PerStageReceiverStore
    ir, wallet, auth, na, nb = _settled_with_auth()
    store = PerStageReceiverStore(tmp_path / "rx.json")
    node = SimpleNamespace(
        identity=SimpleNamespace(node_id=na),                 # the orchestrator IS the head
        transport=_FakeTransport({nb: "10.0.0.2:8000"}),
        _settlement_per_stage_receiver_store=store)
    posted = []

    def _post(url, *, json, timeout):
        posted.append(url)
        return _Resp({"accepted": True, "reason": "ok",
                      "local_escrow_id": json["task"]["batched_receipt"]["local_escrow_id"]})

    out = deliver_for_settled_receipt(
        node, receipt=ir, total_value_wei=_TOTAL, requester_address=_REQ_ADDR,
        per_stage_authorization=auth, wallet_map=wallet, http_post=_post)
    by_node = {r.node_id: r for r in out}
    # head (na): ingested in-process, accepted, staged locally — NOT posted
    assert by_node[na].accepted is True and "in-process" in by_node[na].reason
    assert [s.task.node_id for s in store.all_staged()] == [na]
    # worker (nb): the only HTTP delivery
    assert posted == ["http://10.0.0.2:8000/settlement/per-stage-task"]
    assert by_node[nb].delivered is True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
