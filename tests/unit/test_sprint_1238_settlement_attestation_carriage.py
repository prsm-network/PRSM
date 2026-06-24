"""Sprint 1238 (TEE Tier-3 roadmap C) — carry the attestation to the settlement
boundary.

The settlement adapters built ShardExecutionReceipt with tee_attestation=None
(hardcoded — the inference receipt's attestation is raw bytes, the field wants a
dict). So the attestation never survived past inference into the settlement
receipt. This sprint adds tee_attestation_audit_dict() and threads it through the
three builders (inference_adapter, e2e_proof, per_stage_settlement_split via
NodeSignatureMaterial).

CRITICAL invariant: tee_attestation is a Phase-2.1 schema reservation — NOT in
build_receipt_signing_payload nor batched_receipt_to_leaf — so carrying it must
have ZERO signature / leaf / consensus impact. The leaf-invariance test proves it.
"""

import base64
import hashlib

from prsm.compute.shard_receipt import tee_attestation_audit_dict
from prsm.compute.tee.models import TEEType

ONE_FTNS = 10 ** 18
REQ = "0x" + "a" * 40
PROV = "0x" + "b" * 40
MS_BLOB = b"PRSM-MS-ATT-V1:" + b"\x01" * 16
DEV_BLOB = b"DEV-ONLY-SW-TEE:" + b"\x02" * 16


# ── the audit-dict helper ──────────────────────────────────────────

def test_audit_dict_none_for_empty():
    assert tee_attestation_audit_dict(None) is None
    assert tee_attestation_audit_dict(b"") is None


def test_audit_dict_structure_and_roundtrip():
    d = tee_attestation_audit_dict(MS_BLOB, TEEType.SGX)
    assert d["tee_type"] == "sgx"
    assert d["is_multi_stage"] is True
    assert d["is_dev_only"] is False
    # the raw attestation round-trips out of the audit dict
    assert base64.b64decode(d["attestation_b64"]) == MS_BLOB


def test_audit_dict_flags_dev_only():
    d = tee_attestation_audit_dict(DEV_BLOB, TEEType.SOFTWARE)
    assert d["is_dev_only"] is True
    assert d["is_multi_stage"] is False
    assert d["tee_type"] == "software"


def test_audit_dict_tee_type_optional():
    d = tee_attestation_audit_dict(MS_BLOB)
    assert "tee_type" not in d


# ── e2e_proof threads it (opt-in, default None) ────────────────────

def _build(tee_attestation=None, tee_type=None, identity=None):
    from prsm.node.identity import generate_node_identity
    from prsm.settlement.e2e_proof import build_signed_batched_receipt
    ident = identity or generate_node_identity(display_name="sp1238")
    return build_signed_batched_receipt(
        identity=ident, job_id="job-1",
        output_hash=hashlib.sha256(b"out").hexdigest(),
        executed_at_unix=1700000000, requester_address=REQ, provider_address=PROV,
        value_wei=ONE_FTNS, local_escrow_id="esc-1",
        tee_attestation=tee_attestation, tee_type=tee_type)


def test_e2e_proof_default_is_none():
    br = _build()
    assert br.receipt.tee_attestation is None


def test_e2e_proof_carries_attestation_dict():
    br = _build(tee_attestation=MS_BLOB, tee_type=TEEType.SGX)
    assert isinstance(br.receipt.tee_attestation, dict)
    assert br.receipt.tee_attestation["is_multi_stage"] is True
    assert base64.b64decode(br.receipt.tee_attestation["attestation_b64"]) == MS_BLOB


# ── CONSENSUS-SAFETY: leaf + signing payload are invariant ─────────

def test_leaf_and_signature_invariant_to_attestation():
    from prsm.settlement.merkle import batched_receipt_to_leaf, hash_leaf
    from prsm.node.identity import generate_node_identity

    ident = generate_node_identity(display_name="sp1238-inv")  # SAME key for both
    br_none = _build(identity=ident)
    br_attested = _build(tee_attestation=MS_BLOB, tee_type=TEEType.SGX, identity=ident)
    # signature is over the canonical payload (no tee_attestation) → identical
    assert br_none.receipt.signature == br_attested.receipt.signature
    # the merkle leaf must be byte-identical → zero consensus impact
    assert (hash_leaf(batched_receipt_to_leaf(br_none))
            == hash_leaf(batched_receipt_to_leaf(br_attested)))


# ── per-stage path carries it via NodeSignatureMaterial ────────────

def test_node_signature_material_has_optional_attestation_field():
    from prsm.settlement.per_stage_settlement_split import NodeSignatureMaterial
    mat = NodeSignatureMaterial(
        pubkey_b64="pk", signature="sig", stage_index=0,
        output_hash="0" * 64, executed_at_unix=1700000000)
    assert mat.tee_attestation is None  # default unchanged
    d = tee_attestation_audit_dict(MS_BLOB, TEEType.SGX)
    mat2 = NodeSignatureMaterial(
        pubkey_b64="pk", signature="sig", stage_index=1,
        output_hash="0" * 64, executed_at_unix=1700000000, tee_attestation=d)
    assert mat2.tee_attestation == d
