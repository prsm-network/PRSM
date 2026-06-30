"""Sprint 1316 (S3b-1) — node-side receiver gate for a routed per-stage settlement task.

In Design A each stage node commits its OWN per-node batch (``msg.sender == provider``). Before
it accumulates/commits anything on-chain, the node must confirm the requester's signed per-stage
authorization actually authorizes paying ITS OWN ``(payee, share)`` as a member of the committed
payee set. The requester signs over the WHOLE set's hash, so the node needs the full set
(``payee_set_from_tasks``) even though it holds only its own task — exactly what
``verify_per_stage_authorization`` (sp1172) takes (full ``payees`` + own ``(payee, share)``).

Tested end-to-end through the REAL splitter + REAL signed auth (not a mock): build the routable
tasks, derive the set, sign over it, attach, and assert each node's gate AUTHORIZES — plus the
fail-closed paths (no auth / tampered share / wrong signer / expired / over-cap).
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
from prsm.settlement.per_stage_routing import (
    build_per_stage_settlement_tasks,
    payee_set_from_tasks,
    verify_routed_settlement_task,
)
from prsm.settlement.per_stage_settlement_split import NodeSignatureMaterial

_WEI = 10 ** 18
_REQ_KEY = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
_REQ_ADDR = Account.from_key(_REQ_KEY).address
_PAYEE_A = "0x" + "a1" * 20
_PAYEE_B = "0x" + "b2" * 20
_TOTAL = 10 * _WEI + 1  # non-divisible → exercises the remainder


def _make_ir(*, node_a_id, node_b_id, **over):
    out = hashlib.sha256(b"the multi-stage output").digest()
    topo = TopologyAssignment(
        positions={(0, 0): node_a_id, (1, 0): node_b_id}, stage_count=2, slots_per_stage=1)
    base = dict(
        job_id="job-1316", request_id="req-1316", model_id="qwen2.5-72b",
        content_tier=ContentTier.A, privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0, tee_type=TEEType.SOFTWARE, tee_attestation=b"att",
        output_hash=out, duration_seconds=1.0, cost_ftns=Decimal("1.0"),
        settler_signature=b"sig", settler_node_id=node_a_id, topology_assignment=topo)
    base.update(over)
    return InferenceReceipt(**base)


def _sig(identity, *, job_id, stage_index, output_hex, executed):
    payload = build_receipt_signing_payload(
        job_id=job_id, shard_index=stage_index, output_hash=output_hex, executed_at_unix=executed)
    return NodeSignatureMaterial(
        pubkey_b64=identity.public_key_b64, signature=identity.sign(payload),
        stage_index=stage_index, output_hash=output_hex, executed_at_unix=executed)


def _signed_auth_for(payees, *, total_cap=None, expiry_delta=86400, key=_REQ_KEY,
                     requester=_REQ_ADDR, request_bound=True, chain_id=DEFAULT_CHAIN_ID):
    payload = {
        "requester": requester,
        "payee_set_hash": compute_payee_set_hash(payees),
        "total_max_spend_wei": _TOTAL if total_cap is None else total_cap,
        "job_nonce": keccak(b"nonce-1316"),
        "expiry_unix": int(time.time()) + expiry_delta,
        "request_hash": keccak(b"req-1316") if request_bound else (b"\x00" * 32),
    }
    return {"payload": payload, "signature": sign_per_stage_authorization(payload, key, chain_id=chain_id)}


def _routed_tasks(*, auth_kwargs=None, total=_TOTAL):
    """Build the real per-node tasks, derive the set, sign the auth over it, attach. Returns
    (tasks_with_auth, payees, auth)."""
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
        receipt=ir, total_value_wei=total, requester_address=_REQ_ADDR, wallet_map=wallet)
    payees = payee_set_from_tasks(bare)
    auth = _signed_auth_for(payees, **(auth_kwargs or {}))
    tasks = [dataclasses.replace(t, payment_authorization=auth) for t in bare]
    return tasks, payees, auth


# ── the gate AUTHORIZES each node's own routed share ─────────────────────────

def test_each_node_gate_authorizes_its_own_share():
    tasks, payees, _auth = _routed_tasks()
    assert len(tasks) == 2
    for t in tasks:
        v = verify_routed_settlement_task(t, payees=payees)
        assert v.authorized, v.reason


def test_payee_set_covers_every_task_and_conserves():
    tasks, payees, _ = _routed_tasks()
    assert {p for p, _ in payees} == {_PAYEE_A, _PAYEE_B}
    assert sum(s for _, s in payees) == _TOTAL
    assert dict(payees)[_PAYEE_A] == [t for t in tasks if t.batched_receipt.provider_address == _PAYEE_A][0].share_wei


# ── fail-closed: never authorize, never raise ────────────────────────────────

def test_none_auth_rejected_not_raised():
    tasks, payees, _ = _routed_tasks()
    bare = dataclasses.replace(tasks[0], payment_authorization=None)
    v = verify_routed_settlement_task(bare, payees=payees)
    assert not v.authorized and "no per-stage authorization" in v.reason


def test_tampered_share_rejected():
    # node tries to claim a LARGER share than the set authorizes → membership fails.
    tasks, payees, _ = _routed_tasks()
    t = tasks[0]
    inflated = dataclasses.replace(t, share_wei=t.share_wei + 1)
    assert not verify_routed_settlement_task(inflated, payees=payees).authorized


def test_wrong_payees_set_rejected():
    # verify against a DIFFERENT set than was signed → set-hash mismatch.
    tasks, _payees, _ = _routed_tasks()
    rogue = [(_PAYEE_A, 1), ("0x" + "d3" * 20, _TOTAL - 1)]
    assert not verify_routed_settlement_task(tasks[0], payees=rogue).authorized


def test_expired_auth_rejected():
    tasks, payees, _ = _routed_tasks(auth_kwargs={"expiry_delta": -1})
    assert not verify_routed_settlement_task(tasks[0], payees=payees).authorized


def test_over_cap_auth_rejected():
    tasks, payees, _ = _routed_tasks(auth_kwargs={"total_cap": _TOTAL - 1})
    assert not verify_routed_settlement_task(tasks[0], payees=payees).authorized


def test_wrong_signer_rejected():
    other = Account.create().key.hex()
    tasks, payees, _ = _routed_tasks(auth_kwargs={"key": other})  # requester=_REQ_ADDR but signed by other
    assert not verify_routed_settlement_task(tasks[0], payees=payees).authorized


def test_unbound_request_hash_rejected():
    tasks, payees, _ = _routed_tasks(auth_kwargs={"request_bound": False})
    assert not verify_routed_settlement_task(tasks[0], payees=payees).authorized


def test_malformed_signature_shape_rejected_not_raised():
    tasks, payees, _ = _routed_tasks()
    bad = dataclasses.replace(
        tasks[0],
        payment_authorization={"payload": tasks[0].payment_authorization["payload"],
                               "signature": "0xdeadbeef"})  # too short → InvalidSignatureFormat
    v = verify_routed_settlement_task(bad, payees=payees)
    assert not v.authorized and "malformed" in v.reason.lower()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
