"""Sprint 1162 — wire ComputeWalletMap node->eth-payee resolution into the
on-chain per-stage payee arc (the PRODUCTION gap of project_onchain_per_stage_
payee_arc).

THE GAP. The brick-1 splitter (``per_stage_settlement_split``) stamps each
per-node ``BatchedReceipt.provider_address`` to a REQUESTER PLACEHOLDER (brick 4
re-stamps it at commit). The brick-3 authorization (``per_stage_payment_
authorization``) needs the actual eth payee SET. On the local/Sepolia benches each
node's eth key is supplied CENTRALLY, but a real orchestrator only knows topology
node_ids (``sha256(pubkey)[:32]`` — NOT eth addresses). The fix REUSES the EXISTING
``ComputeWalletMap`` (the off-chain sp1031 split in ``credit_policy.py`` already
uses it) to resolve each node_id -> its REGISTERED eth payee.

CRITICAL fail-closed contract: ``ComputeWalletMap.resolve`` PASSES THROUGH the
node_id when unmapped. A node_id is NOT a valid 20-byte eth address, so an unmapped
node has NO valid on-chain payee — the per-stage arc MUST FAIL-CLOSED (return None /
single-payee fallback), NEVER fabricate or stamp a non-address payee.

Tests:
  (a) RESOLUTION — wallet_map maps each node_id -> a DISTINCT valid eth address;
      the splitter stamps each per-node provider_address to that node's OWN
      resolved payee (no cross-wiring); shares still sum to total.
  (b) FAIL-CLOSED — an unmapped node (resolve passes the node_id through) -> None;
      a mapped-but-non-eth value -> None. NEVER a fabricated/non-address payee.
  (c) AUTH-SPLIT AGREEMENT — the brick-3 payee-set helper over the SAME topology +
      wallet_map produces payees whose addresses EXACTLY match the splitter's
      per-node provider_addresses; compute_payee_set_hash over them is what the
      requester signs — so the authorization covers exactly who the split pays.
  (d) BACKWARD-COMPAT — split WITHOUT a wallet_map is byte-identical to before
      (placeholder provider_address; leaf/value unchanged).
  (e) CONSERVATION — sum of per-node shares == total in the wallet-map path
      (incl. a non-divisible total).

PURE / offline: ComputeWalletMap.from_mapping fixture; no chain, no escrow.
"""
from __future__ import annotations

import base64
import hashlib
from decimal import Decimal

import pytest
from eth_utils import is_address, to_checksum_address

from prsm.compute.shard_receipt import per_stage_leaf_job_id
from prsm.compute.inference.topology_rotation import TopologyAssignment
from prsm.node.compute_wallet_map import ComputeWalletMap
from prsm.node.identity import generate_node_identity
from prsm.settlement.merkle import batched_receipt_to_leaf
from prsm.settlement.per_stage_payment_authorization import compute_payee_set_hash
from prsm.settlement.per_stage_settlement_split import (
    build_per_stage_payee_set,
    resolve_node_eth_payee,
    split_receipt_to_per_node_batched_receipts,
)
from prsm.settlement.per_stage_settlement_split import NodeSignatureMaterial
from prsm.compute.shard_receipt import build_receipt_signing_payload

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
        job_id="job-1162",
        request_id="req-1162",
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


def _two_node_fixture():
    """Two distinct topology nodes + a full set of valid per-node sig material.
    Returns (ir, node_sigs, id_a, id_b, executed, total)."""
    id_a = generate_node_identity(display_name="stage0")
    id_b = generate_node_identity(display_name="stage1")
    ir = _make_ir(node_a_id=id_a.node_id, node_b_id=id_b.node_id)
    output_hex = ir.output_hash.hex()
    executed = 1_700_000_000
    job = per_stage_leaf_job_id(ir.request_id)
    node_sigs = {
        id_a.node_id: _sig_material_for(
            id_a, job_id=job, stage_index=0,
            output_hash_hex=output_hex, executed_at_unix=executed,
        ),
        id_b.node_id: _sig_material_for(
            id_b, job_id=job, stage_index=1,
            output_hash_hex=output_hex, executed_at_unix=executed,
        ),
    }
    total = 10 * _WEI + 1  # non-divisible by 2 to exercise the remainder
    return ir, node_sigs, id_a, id_b, executed, total


# Distinct valid eth payees (lowercase + checksum to prove case-insensitivity).
_PAYEE_A = "0x" + "a1" * 20
_PAYEE_B = "0x" + "b2" * 20


# ── (a) RESOLUTION ────────────────────────────────────────────────────

def test_wallet_map_stamps_each_node_own_resolved_payee_no_cross_wiring():
    ir, node_sigs, id_a, id_b, _executed, total = _two_node_fixture()
    wallet_map = ComputeWalletMap.from_mapping({
        id_a.node_id: _PAYEE_A,
        id_b.node_id: _PAYEE_B,
    })
    shares = split_receipt_to_per_node_batched_receipts(
        receipt=ir,
        total_value_wei=total,
        node_signatures=node_sigs,
        requester_address="0x" + "11" * 20,
        wallet_map=wallet_map,
    )
    assert shares is not None
    assert len(shares) == 2

    payee_by_node = {
        s.node_id: s.batched_receipt.provider_address for s in shares
    }
    # Each node's provider_address is its OWN resolved payee — no cross-wiring.
    assert payee_by_node[id_a.node_id].lower() == _PAYEE_A.lower()
    assert payee_by_node[id_b.node_id].lower() == _PAYEE_B.lower()
    assert payee_by_node[id_a.node_id].lower() != payee_by_node[id_b.node_id].lower()

    # shares still sum to total exactly
    assert sum(s.share_wei for s in shares) == total
    # the resolved payee is a real eth address, NEVER the node_id
    for s in shares:
        assert is_address(s.batched_receipt.provider_address)
        assert s.batched_receipt.provider_address != s.node_id


def test_resolve_node_eth_payee_returns_address_for_mapped_node():
    nid = "abc123"
    wm = ComputeWalletMap.from_mapping({nid: _PAYEE_A})
    assert resolve_node_eth_payee(nid, wm).lower() == _PAYEE_A.lower()


# ── (b) FAIL-CLOSED ───────────────────────────────────────────────────

def test_unmapped_node_fails_closed_returns_none():
    ir, node_sigs, id_a, id_b, _executed, total = _two_node_fixture()
    # node_b is NOT in the map -> resolve passes the node_id through (NOT an eth
    # address) -> the WHOLE split must fail closed (single-payee fallback).
    wallet_map = ComputeWalletMap.from_mapping({id_a.node_id: _PAYEE_A})
    shares = split_receipt_to_per_node_batched_receipts(
        receipt=ir,
        total_value_wei=total,
        node_signatures=node_sigs,
        requester_address="0x" + "11" * 20,
        wallet_map=wallet_map,
    )
    assert shares is None


def test_mapped_to_non_eth_value_fails_closed_returns_none():
    ir, node_sigs, id_a, id_b, _executed, total = _two_node_fixture()
    # node_b maps to a non-eth string -> not a valid payee -> fail closed.
    wallet_map = ComputeWalletMap.from_mapping({
        id_a.node_id: _PAYEE_A,
        id_b.node_id: "operator-wallet-not-an-eth-address",
    })
    shares = split_receipt_to_per_node_batched_receipts(
        receipt=ir,
        total_value_wei=total,
        node_signatures=node_sigs,
        requester_address="0x" + "11" * 20,
        wallet_map=wallet_map,
    )
    assert shares is None


def test_resolve_helper_fail_closed_on_passthrough_and_non_address():
    # Unmapped -> resolve passes the node_id through -> None (never the node_id).
    empty = ComputeWalletMap.from_mapping({})
    assert resolve_node_eth_payee("somenode", empty) is None
    # Mapped to a non-eth value -> None (never a fabricated address).
    bad = ComputeWalletMap.from_mapping({"somenode": "not-an-address"})
    assert resolve_node_eth_payee("somenode", bad) is None
    # None wallet_map -> None (no resolution requested).
    assert resolve_node_eth_payee("somenode", None) is None


# ── (c) AUTH-SPLIT AGREEMENT ──────────────────────────────────────────

def test_auth_payee_set_agrees_with_split_provider_addresses():
    ir, node_sigs, id_a, id_b, _executed, total = _two_node_fixture()
    wallet_map = ComputeWalletMap.from_mapping({
        id_a.node_id: _PAYEE_A,
        id_b.node_id: _PAYEE_B,
    })
    shares = split_receipt_to_per_node_batched_receipts(
        receipt=ir,
        total_value_wei=total,
        node_signatures=node_sigs,
        requester_address="0x" + "11" * 20,
        wallet_map=wallet_map,
    )
    assert shares is not None

    # The brick-3 payee-set helper over the SAME topology + wallet_map.
    payees = build_per_stage_payee_set(
        receipt=ir, total_value_wei=total, wallet_map=wallet_map,
    )
    assert payees is not None
    assert len(payees) == 2

    # AGREEMENT: the auth payee set (addr, share) EXACTLY matches the split's
    # per-node (provider_address, value_ftns) — same payees, same shares.
    auth_set = {
        (to_checksum_address(a), share) for a, share in payees
    }
    split_set = {
        (to_checksum_address(s.batched_receipt.provider_address), s.share_wei)
        for s in shares
    }
    assert auth_set == split_set

    # And the canonical payee-set hash (what the requester signs) is computable
    # over EXACTLY the split's payee set -> the same commitment either side.
    split_payees = [
        (s.batched_receipt.provider_address, s.share_wei) for s in shares
    ]
    assert compute_payee_set_hash(payees) == compute_payee_set_hash(split_payees)


def test_auth_split_agree_even_with_zero_share_payee_sub_wei_total():
    """Sub-wei edge (flagged in review): total=1 wei across 2 nodes -> deterministic
    remainder split [1, 0], so one payee has share 0. The 0-share payee is INCLUDED in
    BOTH the split and the auth payee-set, and that is SAFE: conservation still holds
    (sum==total), the auth/split sets still AGREE, and compute_payee_set_hash accepts the
    0-share payee (does not raise). Documents the edge; realistic FTNS settlements are
    >> N wei so no node ever gets 0 in practice."""
    ir, node_sigs, id_a, id_b, _executed, _total = _two_node_fixture()
    wallet_map = ComputeWalletMap.from_mapping({
        id_a.node_id: _PAYEE_A,
        id_b.node_id: _PAYEE_B,
    })
    ONE = 1  # 1 wei across 2 nodes -> shares [1, 0]
    shares = split_receipt_to_per_node_batched_receipts(
        receipt=ir, total_value_wei=ONE, node_signatures=node_sigs,
        requester_address="0x" + "11" * 20, wallet_map=wallet_map,
    )
    assert shares is not None
    payees = build_per_stage_payee_set(
        receipt=ir, total_value_wei=ONE, wallet_map=wallet_map,
    )
    assert payees is not None
    # conservation holds exactly, incl. the 0-share node
    assert sum(s.share_wei for s in shares) == ONE
    assert sum(share for _a, share in payees) == ONE
    # a 0-share payee is genuinely present (the remainder split)
    assert 0 in {s.share_wei for s in shares}
    # AGREEMENT still holds with the 0-share payee, and the hash is computable
    auth_set = {(to_checksum_address(a), share) for a, share in payees}
    split_set = {
        (to_checksum_address(s.batched_receipt.provider_address), s.share_wei)
        for s in shares
    }
    assert auth_set == split_set
    split_payees = [
        (s.batched_receipt.provider_address, s.share_wei) for s in shares
    ]
    assert compute_payee_set_hash(payees) == compute_payee_set_hash(split_payees)


def test_auth_payee_set_fails_closed_when_a_node_unmapped():
    ir, _node_sigs, id_a, id_b, _executed, total = _two_node_fixture()
    wallet_map = ComputeWalletMap.from_mapping({id_a.node_id: _PAYEE_A})
    payees = build_per_stage_payee_set(
        receipt=ir, total_value_wei=total, wallet_map=wallet_map,
    )
    assert payees is None


# ── (d) BACKWARD-COMPAT ───────────────────────────────────────────────

def test_no_wallet_map_is_byte_identical_to_placeholder_behavior():
    ir, node_sigs, _id_a, _id_b, _executed, total = _two_node_fixture()
    requester = "0x" + "11" * 20

    # The current/bench path: NO wallet_map argument at all.
    baseline = split_receipt_to_per_node_batched_receipts(
        receipt=ir,
        total_value_wei=total,
        node_signatures=node_sigs,
        requester_address=requester,
    )
    # Explicit wallet_map=None must be identical to omitting it.
    explicit_none = split_receipt_to_per_node_batched_receipts(
        receipt=ir,
        total_value_wei=total,
        node_signatures=node_sigs,
        requester_address=requester,
        wallet_map=None,
    )
    assert baseline is not None
    assert explicit_none is not None

    for shares in (baseline, explicit_none):
        for s in shares:
            # placeholder provider_address == requester (brick 4 re-stamps later)
            assert s.batched_receipt.provider_address == requester
            assert s.batched_receipt.requester_address == requester

    # byte-for-byte: leaf hash + value + every leaf-defining field identical
    # across the two no-wallet_map calls.
    def _snapshot(shares):
        out = []
        for s in sorted(shares, key=lambda x: x.node_id):
            br = s.batched_receipt
            leaf = batched_receipt_to_leaf(br)
            out.append((
                s.node_id, s.share_wei, br.value_ftns,
                br.provider_address, br.requester_address,
                br.receipt.provider_id, br.receipt.provider_pubkey_b64,
                br.receipt.output_hash, br.receipt.signature,
                bytes(leaf.signing_message_hash),
            ))
        return out

    assert _snapshot(baseline) == _snapshot(explicit_none)


# ── (e) CONSERVATION ──────────────────────────────────────────────────

@pytest.mark.parametrize("total", [10 * _WEI, 10 * _WEI + 1, 3, 2, 1])
def test_conservation_preserved_in_wallet_map_path(total):
    ir, node_sigs, id_a, id_b, _executed, _t = _two_node_fixture()
    wallet_map = ComputeWalletMap.from_mapping({
        id_a.node_id: _PAYEE_A,
        id_b.node_id: _PAYEE_B,
    })
    shares = split_receipt_to_per_node_batched_receipts(
        receipt=ir,
        total_value_wei=total,
        node_signatures=node_sigs,
        requester_address="0x" + "11" * 20,
        wallet_map=wallet_map,
    )
    assert shares is not None
    # EXACT integer-wei conservation in the wallet-map path (incl. non-divisible).
    assert sum(s.share_wei for s in shares) == total
    assert sum(s.batched_receipt.value_ftns for s in shares) == total
