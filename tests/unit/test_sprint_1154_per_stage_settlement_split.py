"""Sprint 1154 — per-node on-chain settlement SPLITTER (BRICK 1 of the
on-chain per-stage payee arc).

The deployed BatchSettlementRegistry pays the WHOLE batch value to ONE
provider (commitBatch sets b.provider = msg.sender; finalizeBatch settles
finalValue to b.provider). There is NO per-leaf payee. So on-chain
per-stage payment = one share-batch PER topology node, each committed BY
that node, valued at its share. This brick produces those per-node
``BatchedReceipt``s (the downstream bricks have each node commit its own).

This module is the PURE, money-safe, offline-testable CORE:

  - ``compute_per_stage_shares`` — the value-conservation primitive: EQUAL
    integer-wei split with EXACT conservation (sum == total, no over-pay,
    deterministic remainder).
  - ``split_receipt_to_per_node_batched_receipts`` — the splitter. Mirrors
    sp1031 gating (None/single-payee fallback when topology absent /
    malformed / <2 distinct nodes), FAIL-CLOSED on any missing per-node
    signature (never fabricate a payee). For >=2 nodes with all signatures
    present, builds one challenge-defensible ``BatchedReceipt`` per node.

Tests:
  (a) CONSERVATION — divisible + non-divisible + remainder-dominated +
      near-uint128 totals all conserve EXACTLY in integer wei.
  (b) CHALLENGE-DEFENSIBILITY — each per-node leaf rebuilds
      (batched_receipt_to_leaf) and the consumed node signature verifies
      under that node's pubkey over the leaf signing_message_hash ==
      build_receipt_signing_payload (an on-chain INVALID_SIGNATURE
      challenge would FAIL); binding provider_id == sha256(pubkey)[:32].
  (c) VALUE SPLIT — the per-node shares in the receipt path sum to total.
  (d) FAIL-CLOSED — topology absent / single-node / malformed / a missing
      node_signature -> None.
  (e) PURITY — the splitter consumes only provided sig material + pubkeys
      (works with NO private keys available); it never signs.
"""
from __future__ import annotations

import base64
import hashlib
from decimal import Decimal

import pytest

from prsm.compute.shard_receipt import build_receipt_signing_payload
from prsm.compute.inference.topology_rotation import TopologyAssignment
from prsm.node.identity import generate_node_identity, verify_signature
from prsm.settlement.merkle import batched_receipt_to_leaf
from prsm.settlement.per_stage_settlement_split import (
    NodeSignatureMaterial,
    StageSettlementShare,
    compute_per_stage_shares,
    split_receipt_to_per_node_batched_receipts,
)

_MAX_UINT128 = 2 ** 128 - 1
_WEI = 10 ** 18


# ── helpers ───────────────────────────────────────────────────────────

def _make_ir(*, node_a_id, node_b_id, output_hash=None, **over):
    from prsm.compute.inference.models import InferenceReceipt, ContentTier
    from prsm.compute.tee.models import PrivacyLevel, TEEType
    if output_hash is None:
        output_hash = hashlib.sha256(b"the multi-stage output").digest()
    topo = TopologyAssignment(
        positions={(0, 0): node_a_id, (1, 0): node_b_id},
        stage_count=2,
        slots_per_stage=1,
    )
    defaults = dict(
        job_id="job-1154",
        request_id="req-1154",
        model_id="gpt2",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=b"att",
        output_hash=output_hash,
        duration_seconds=1.0,
        cost_ftns=Decimal("1.0"),
        settler_signature=b"parallax-signature-bytes",
        settler_node_id=node_a_id,
        topology_assignment=topo,
    )
    defaults.update(over)
    return InferenceReceipt(**defaults)


def _sig_material_for(identity, *, job_id, stage_index, output_hash_hex,
                      executed_at_unix):
    """A node signs build_receipt_signing_payload for ITS stage and hands the
    splitter the material (pubkey + sig + the leaf-defining fields)."""
    payload = build_receipt_signing_payload(
        job_id=job_id,
        shard_index=stage_index,
        output_hash=output_hash_hex,
        executed_at_unix=executed_at_unix,
    )
    return NodeSignatureMaterial(
        pubkey_b64=identity.public_key_b64,
        signature=identity.sign(payload),
        stage_index=stage_index,
        output_hash=output_hash_hex,
        executed_at_unix=executed_at_unix,
    )


# ── (a) CONSERVATION ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    "total,n",
    [
        (9, 3),                  # divisible
        (10, 3),                 # non-divisible (remainder 1)
        (11, 3),                 # non-divisible (remainder 2)
        (1, 3),                  # remainder-dominated: 1 wei across 3
        (0, 3),                  # zero total
        (2, 5),
        (7, 1),                  # single node gets all
        (_WEI, 3),
        (_MAX_UINT128, 3),       # near-uint128
        (_MAX_UINT128, 7),
        (_MAX_UINT128 - 1, 4),
    ],
)
def test_compute_shares_conserves_exactly(total, n):
    node_ids = [f"node{i}" for i in range(n)]
    shares = compute_per_stage_shares(total, node_ids)
    assert len(shares) == n
    # exact integer conservation
    assert sum(s for _, s in shares) == total
    # every share >= 0
    assert all(s >= 0 for _, s in shares)
    # no node over-paid: no share exceeds ceil(total/n). Integer ceil — a
    # float math.ceil(total/n) loses precision near uint128.
    ceil_share = -(-total // n) if total > 0 else 0
    assert all(s <= ceil_share for _, s in shares)
    # node_ids preserved in stable order
    assert [nid for nid, _ in shares] == node_ids


def test_compute_shares_remainder_is_deterministic_and_front_loaded():
    # 10 / 3 -> [4, 3, 3] (the first (total % n)=1 node gets the +1)
    shares = compute_per_stage_shares(10, ["a", "b", "c"])
    assert shares == [("a", 4), ("b", 3), ("c", 3)]
    # 11 / 3 -> [4, 4, 3]
    shares = compute_per_stage_shares(11, ["a", "b", "c"])
    assert shares == [("a", 4), ("b", 4), ("c", 3)]
    # 1 / 3 -> [1, 0, 0]
    shares = compute_per_stage_shares(1, ["a", "b", "c"])
    assert shares == [("a", 1), ("b", 0), ("c", 0)]
    # deterministic across repeated calls
    assert compute_per_stage_shares(10, ["a", "b", "c"]) == \
        compute_per_stage_shares(10, ["a", "b", "c"])


def test_compute_shares_rejects_bad_inputs():
    with pytest.raises(ValueError):
        compute_per_stage_shares(-1, ["a", "b"])
    with pytest.raises(ValueError):
        compute_per_stage_shares(10, [])
    with pytest.raises(ValueError):
        compute_per_stage_shares(2 ** 128, ["a", "b"])   # beyond uint128
    with pytest.raises((ValueError, TypeError)):
        compute_per_stage_shares(1.5, ["a", "b"])        # not an int
    with pytest.raises((ValueError, TypeError)):
        compute_per_stage_shares(True, ["a", "b"])       # bool rejected


# ── (b)+(c) CHALLENGE-DEFENSIBILITY + VALUE SPLIT ────────────────────

def test_splits_two_distinct_nodes_into_defensible_leaves():
    id_a = generate_node_identity(display_name="stage0")
    id_b = generate_node_identity(display_name="stage1")
    ir = _make_ir(node_a_id=id_a.node_id, node_b_id=id_b.node_id)
    output_hex = ir.output_hash.hex()
    executed = 1_700_000_000
    node_sigs = {
        id_a.node_id: _sig_material_for(
            id_a, job_id=ir.job_id, stage_index=0,
            output_hash_hex=output_hex, executed_at_unix=executed,
        ),
        id_b.node_id: _sig_material_for(
            id_b, job_id=ir.job_id, stage_index=1,
            output_hash_hex=output_hex, executed_at_unix=executed,
        ),
    }
    total = 10 * _WEI + 1   # non-divisible by 2 to exercise remainder
    shares = split_receipt_to_per_node_batched_receipts(
        receipt=ir,
        total_value_wei=total,
        node_signatures=node_sigs,
        requester_address="0x" + "11" * 20,
    )
    assert shares is not None
    assert len(shares) == 2
    assert all(isinstance(s, StageSettlementShare) for s in shares)

    # (c) VALUE SPLIT: shares sum to total EXACTLY
    assert sum(s.share_wei for s in shares) == total
    assert {s.node_id for s in shares} == {id_a.node_id, id_b.node_id}

    id_by_node = {id_a.node_id: id_a, id_b.node_id: id_b}
    for share in shares:
        br = share.batched_receipt
        ident = id_by_node[share.node_id]
        # provider binding: provider_id == node_id == sha256(pubkey)[:32]
        assert br.receipt.provider_id == share.node_id
        derived = hashlib.sha256(
            base64.b64decode(br.receipt.provider_pubkey_b64)
        ).hexdigest()[:32]
        assert derived == share.node_id
        # value_ftns is THIS node's share
        assert br.value_ftns == share.share_wei
        # (b) CHALLENGE-DEFENSIBILITY: rebuild the leaf, verify the node sig
        # over leaf.signing_message_hash == build_receipt_signing_payload.
        leaf = batched_receipt_to_leaf(br)
        expected = build_receipt_signing_payload(
            job_id=ir.job_id,
            shard_index=br.receipt.shard_index,
            output_hash=br.receipt.output_hash,
            executed_at_unix=br.receipt.executed_at_unix,
        )
        assert leaf.signing_message_hash == expected
        # the consumed node signature verifies under that node's pubkey over
        # the leaf signing message -> an on-chain INVALID_SIGNATURE challenge
        # (succeeds iff verify==False) would FAIL: the leaf is defensible.
        assert verify_signature(
            br.receipt.provider_pubkey_b64,
            leaf.signing_message_hash,
            br.receipt.signature,
        ) is True
        # double-check it's THIS node's pubkey
        assert br.receipt.provider_pubkey_b64 == ident.public_key_b64


def test_distinct_shard_indices_per_node():
    id_a = generate_node_identity()
    id_b = generate_node_identity()
    ir = _make_ir(node_a_id=id_a.node_id, node_b_id=id_b.node_id)
    output_hex = ir.output_hash.hex()
    node_sigs = {
        id_a.node_id: _sig_material_for(
            id_a, job_id=ir.job_id, stage_index=0,
            output_hash_hex=output_hex, executed_at_unix=42),
        id_b.node_id: _sig_material_for(
            id_b, job_id=ir.job_id, stage_index=1,
            output_hash_hex=output_hex, executed_at_unix=42),
    }
    shares = split_receipt_to_per_node_batched_receipts(
        receipt=ir, total_value_wei=8 * _WEI, node_signatures=node_sigs,
        requester_address="0x" + "ab" * 20,
    )
    by_node = {s.node_id: s for s in shares}
    assert by_node[id_a.node_id].batched_receipt.receipt.shard_index == 0
    assert by_node[id_b.node_id].batched_receipt.receipt.shard_index == 1


# ── (d) FAIL-CLOSED ──────────────────────────────────────────────────

def _full_sigs(ir, id_a, id_b):
    output_hex = ir.output_hash.hex()
    return {
        id_a.node_id: _sig_material_for(
            id_a, job_id=ir.job_id, stage_index=0,
            output_hash_hex=output_hex, executed_at_unix=7),
        id_b.node_id: _sig_material_for(
            id_b, job_id=ir.job_id, stage_index=1,
            output_hash_hex=output_hex, executed_at_unix=7),
    }


def test_fail_closed_topology_absent():
    id_a = generate_node_identity()
    id_b = generate_node_identity()
    ir = _make_ir(node_a_id=id_a.node_id, node_b_id=id_b.node_id,
                  topology_assignment=None)
    assert split_receipt_to_per_node_batched_receipts(
        receipt=ir, total_value_wei=_WEI,
        node_signatures=_full_sigs(ir, id_a, id_b),
        requester_address="0x" + "11" * 20,
    ) is None


def test_fail_closed_single_node_topology():
    id_a = generate_node_identity()
    only = TopologyAssignment(
        positions={(0, 0): id_a.node_id, (1, 0): id_a.node_id},
        stage_count=2, slots_per_stage=1,
    )
    ir = _make_ir(node_a_id=id_a.node_id, node_b_id=id_a.node_id,
                  topology_assignment=only)
    output_hex = ir.output_hash.hex()
    sigs = {id_a.node_id: _sig_material_for(
        id_a, job_id=ir.job_id, stage_index=0,
        output_hash_hex=output_hex, executed_at_unix=7)}
    assert split_receipt_to_per_node_batched_receipts(
        receipt=ir, total_value_wei=_WEI, node_signatures=sigs,
        requester_address="0x" + "11" * 20,
    ) is None


def test_fail_closed_malformed_topology():
    id_a = generate_node_identity()
    id_b = generate_node_identity()

    class _Bad:
        positions = "not-a-dict"

    ir = _make_ir(node_a_id=id_a.node_id, node_b_id=id_b.node_id,
                  topology_assignment=_Bad())
    assert split_receipt_to_per_node_batched_receipts(
        receipt=ir, total_value_wei=_WEI,
        node_signatures=_full_sigs(ir, id_a, id_b),
        requester_address="0x" + "11" * 20,
    ) is None

    # empty positions dict -> malformed
    empty = TopologyAssignment(positions={}, stage_count=0, slots_per_stage=0)
    ir2 = _make_ir(node_a_id=id_a.node_id, node_b_id=id_b.node_id,
                   topology_assignment=empty)
    assert split_receipt_to_per_node_batched_receipts(
        receipt=ir2, total_value_wei=_WEI,
        node_signatures=_full_sigs(ir2, id_a, id_b),
        requester_address="0x" + "11" * 20,
    ) is None


def test_fail_closed_missing_node_signature():
    id_a = generate_node_identity()
    id_b = generate_node_identity()
    ir = _make_ir(node_a_id=id_a.node_id, node_b_id=id_b.node_id)
    output_hex = ir.output_hash.hex()
    # only node A's signature present; B's missing -> NEVER a partial payee set
    partial = {
        id_a.node_id: _sig_material_for(
            id_a, job_id=ir.job_id, stage_index=0,
            output_hash_hex=output_hex, executed_at_unix=7),
    }
    assert split_receipt_to_per_node_batched_receipts(
        receipt=ir, total_value_wei=_WEI, node_signatures=partial,
        requester_address="0x" + "11" * 20,
    ) is None


# ── (e) PURITY ───────────────────────────────────────────────────────

def test_purity_no_private_keys_needed():
    """The splitter consumes only the provided sig material + pubkeys; it has
    no signing key. We sign OUTSIDE the splitter (the worker-emission brick's
    job), discard the private keys, and the splitter still builds defensible
    leaves from the public material alone."""
    id_a = generate_node_identity()
    id_b = generate_node_identity()
    ir = _make_ir(node_a_id=id_a.node_id, node_b_id=id_b.node_id)
    output_hex = ir.output_hash.hex()
    node_sigs = _full_sigs(ir, id_a, id_b)
    # the material carries only b64 pubkeys + b64 signatures — no private keys.
    for mat in node_sigs.values():
        assert isinstance(mat.pubkey_b64, str)
        assert isinstance(mat.signature, str)
        assert not hasattr(mat, "private_key")
        assert not hasattr(mat, "private_key_bytes")
    shares = split_receipt_to_per_node_batched_receipts(
        receipt=ir, total_value_wei=2 * _WEI, node_signatures=node_sigs,
        requester_address="0x" + "11" * 20,
    )
    assert shares is not None and len(shares) == 2
    # StageSettlementShare must be frozen (immutable money object)
    with pytest.raises(Exception):
        shares[0].share_wei = 0  # type: ignore[misc]
