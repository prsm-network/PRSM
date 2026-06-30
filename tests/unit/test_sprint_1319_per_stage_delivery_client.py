"""Sprint 1319 (S3b-2c) — orchestrator-side SENDER for routed per-stage settlement tasks.

The settling node splits a settled multi-stage receipt (S3a) and POSTs each node its task + the
full payee set (sp1318 endpoint). FAIL-SOFT: a miss leaves a stage undelivered (unpaid), never
raises. Tested against a LOOP-BACK fake transport that drives the REAL receiver
(ingest_routed_task) so send→receive is proven end-to-end, plus the failure paths (transport
error, non-200, unmapped node, not-multistage).
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
from prsm.settlement.per_stage_delivery_client import (
    DeliveryResult,
    deliver_per_stage_task,
    deliver_settled_multistage_tasks,
)
from prsm.settlement.per_stage_payment_authorization import (
    DEFAULT_CHAIN_ID,
    compute_payee_set_hash,
    sign_per_stage_authorization,
)
from prsm.settlement.per_stage_receiver_store import (
    PerStageReceiverStore,
    ingest_routed_task,
    parse_delivery_request,
)
from prsm.settlement.per_stage_settlement_split import NodeSignatureMaterial

_WEI = 10 ** 18
_REQ_KEY = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
_REQ_ADDR = Account.from_key(_REQ_KEY).address
_PAYEE_A = "0x" + "a1" * 20
_PAYEE_B = "0x" + "b2" * 20
_TOTAL = 10 * _WEI + 1


def _make_ir(*, node_a_id, node_b_id):
    out = hashlib.sha256(b"out-1319").digest()
    topo = TopologyAssignment(
        positions={(0, 0): node_a_id, (1, 0): node_b_id}, stage_count=2, slots_per_stage=1)
    return InferenceReceipt(
        job_id="job-1319", request_id="req-1319", model_id="qwen2.5-72b",
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


def _settled_receipt():
    """A settled multi-stage receipt carrying per-stage sigs + the wallet map + a signed auth over
    the resulting payee set (built after the split learns the resolved payees + shares)."""
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
    return ir, wallet, id_a.node_id, id_b.node_id


def _auth_over(payees):
    payload = {
        "requester": _REQ_ADDR,
        "payee_set_hash": "0x" + compute_payee_set_hash(payees).hex(),
        "total_max_spend_wei": _TOTAL,
        "job_nonce": "0x" + keccak(b"n-1319").hex(),
        "expiry_unix": int(time.time()) + 86400,
        "request_hash": "0x" + keccak(b"r-1319").hex(),
    }
    sig = sign_per_stage_authorization(payload, _REQ_KEY, chain_id=DEFAULT_CHAIN_ID)
    return {"payload": payload, "signature": "0x" + sig.hex()}


class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


class _LoopbackTransport:
    """A fake http_post that drives the REAL receiver endpoint logic (parse + ingest) against a
    per-node store, so the send path is proven against the actual receive path (no network)."""

    def __init__(self, store_for_node, node_id_for_url):
        self._store_for_node = store_for_node
        self._node_id_for_url = node_id_for_url
        self.calls = []

    def __call__(self, url, *, json, timeout):
        self.calls.append(url)
        my_node = self._node_id_for_url(url)
        task, payees = parse_delivery_request(json)  # 422-equivalent would raise → here always valid
        res = ingest_routed_task(self._store_for_node(my_node), task, payees=payees, my_node_id=my_node)
        return _Resp(200, {"accepted": res.accepted, "reason": res.reason,
                           "local_escrow_id": res.local_escrow_id})


# ── end-to-end loop-back: both nodes receive + stage their own share ─────────

def test_fan_out_delivers_and_both_nodes_stage(tmp_path):
    ir, wallet, na, nb = _settled_receipt()
    # endpoints + per-node stores
    url_a, url_b = "http://a:8000", "http://b:8000"
    stores = {na: PerStageReceiverStore(tmp_path / "a.json"),
              nb: PerStageReceiverStore(tmp_path / "b.json")}
    node_to_url = {na: url_a, nb: url_b}
    # the transport receives the FULL endpoint URL (base + /settlement/per-stage-task)
    transport = _LoopbackTransport(
        lambda nid: stores[nid], lambda u: na if u.startswith(url_a) else nb)

    # build the auth over the actual split payee set
    from prsm.settlement.per_stage_routing import (
        build_per_stage_settlement_tasks,
        payee_set_from_tasks,
    )
    tmp_tasks = build_per_stage_settlement_tasks(
        receipt=ir, total_value_wei=_TOTAL, requester_address=_REQ_ADDR, wallet_map=wallet)
    auth = _auth_over(payee_set_from_tasks(tmp_tasks))

    results = deliver_settled_multistage_tasks(
        receipt=ir, total_value_wei=_TOTAL, requester_address=_REQ_ADDR,
        endpoint_for_node=lambda nid: node_to_url.get(nid),
        per_stage_authorization=auth, wallet_map=wallet, http_post=transport)

    assert len(results) == 2
    assert all(r.delivered and r.accepted for r in results), [r.reason for r in results]
    # each node actually staged its own share
    assert stores[na].all_staged()[0].task.node_id == na
    assert stores[nb].all_staged()[0].task.node_id == nb
    assert len(transport.calls) == 2


# ── not-multistage / fail-closed split → nothing delivered ───────────────────

def test_no_per_stage_signatures_delivers_nothing(tmp_path):
    ir, wallet, _na, _nb = _settled_receipt()
    ir = dataclasses.replace(ir, per_stage_settlement_signatures=None)
    results = deliver_settled_multistage_tasks(
        receipt=ir, total_value_wei=_TOTAL, requester_address=_REQ_ADDR,
        endpoint_for_node=lambda nid: "http://x", wallet_map=wallet,
        http_post=lambda *a, **k: _Resp(200, {}))
    assert results == []


# ── unmapped endpoint → recorded undelivered, never raises ───────────────────

def test_unmapped_endpoint_recorded_undelivered(tmp_path):
    ir, wallet, na, nb = _settled_receipt()
    from prsm.settlement.per_stage_routing import (
        build_per_stage_settlement_tasks,
        payee_set_from_tasks,
    )
    auth = _auth_over(payee_set_from_tasks(build_per_stage_settlement_tasks(
        receipt=ir, total_value_wei=_TOTAL, requester_address=_REQ_ADDR, wallet_map=wallet)))
    # only node A has an endpoint; B unmapped
    results = deliver_settled_multistage_tasks(
        receipt=ir, total_value_wei=_TOTAL, requester_address=_REQ_ADDR,
        endpoint_for_node=lambda nid: "http://a:8000" if nid == na else None,
        per_stage_authorization=auth, wallet_map=wallet,
        http_post=lambda *a, **k: _Resp(200, {"accepted": True, "reason": "ok", "local_escrow_id": "x"}))
    by_node = {r.node_id: r for r in results}
    assert by_node[na].delivered is True
    assert by_node[nb].delivered is False and "no resolvable endpoint" in by_node[nb].reason


# ── single-delivery transport failures (fail-soft) ───────────────────────────

class _Task:
    node_id = "n0"
    def to_dict(self):
        return {"node_id": "n0"}


def test_transport_error_is_fail_soft():
    def _boom(*a, **k):
        raise ConnectionError("refused")
    r = deliver_per_stage_task("http://x", _Task(), [(_PAYEE_A, 1)], http_post=_boom)
    assert isinstance(r, DeliveryResult)
    assert r.delivered is False and r.accepted is None and "transport error" in r.reason


def test_non_200_is_fail_soft():
    r = deliver_per_stage_task("http://x", _Task(), [(_PAYEE_A, 1)],
                               http_post=lambda *a, **k: _Resp(503, {"detail": "gate off"}))
    assert r.delivered is False and r.status_code == 503 and "gate off" in r.reason


def test_rejected_delivery_surfaces_accepted_false():
    r = deliver_per_stage_task("http://x", _Task(), [(_PAYEE_A, 1)],
                               http_post=lambda *a, **k: _Resp(200, {"accepted": False, "reason": "misrouted"}))
    assert r.delivered is True and r.accepted is False and r.reason == "misrouted"


def test_post_body_shape():
    captured = {}
    def _cap(url, *, json, timeout):
        captured["url"] = url; captured["json"] = json
        return _Resp(200, {"accepted": True, "reason": "ok"})
    deliver_per_stage_task("http://h:1/", _Task(), [(_PAYEE_A, 5), (_PAYEE_B, 7)], http_post=_cap)
    assert captured["url"] == "http://h:1/settlement/per-stage-task"
    assert captured["json"]["payees"] == [[_PAYEE_A, "5"], [_PAYEE_B, "7"]]
    assert captured["json"]["task"] == {"node_id": "n0"}


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
