"""Sprint 1156 — worker stage-settlement-signature EMISSION + orchestrator
assembly (BRICK 2 of the on-chain per-stage payee arc).

Brick 1 (sp1154 ``per_stage_settlement_split``) CONSUMES per-node
``NodeSignatureMaterial`` — each topology node's challenge-defensible
shard-payload signature over ``build_receipt_signing_payload(job_id,
stage_index, output_hash, executed_at)`` (a 32-byte keccak that IS the
on-chain leaf ``signingMessageHash``). Brick 2 makes that INPUT real: each
WORKER, when it signs its ``RunLayerSliceResponse``, ALSO emits that
challenge-defensible settlement signature so the orchestrator can assemble
the per-node material the splitter consumes.

WHY a NEW signature (not the activation-proof sig): the per-stage
``StageActivationProof`` signature signs ``StageActivationProof.signing_bytes``
— a VARIABLE-LENGTH UTF-8 payload — so it is NOT a valid on-chain leaf sig
(the contract Ed25519 verifier checks a sig over the 32-byte leaf hash). The
settlement leaf sig must be over ``build_receipt_signing_payload`` = a 32-byte
keccak. So the worker ALSO signs THAT for its stage. It is a SEPARATE,
self-securing artifact — NOT part of the stage-response ``signing_payload``,
so adding it CANNOT change the existing stage signature bytes.

Tests:
  (a) BYTE-EQUIVALENCE — a response signed WITHOUT the settlement-sig inputs
      produces a byte-identical ``signing_payload`` + the same
      ``stage_signature_b64`` as before; the new fields default None / omit
      from ``to_dict``; a round-trip is unchanged. Pinned: the signing payload
      bytes for a fixed input are equal with vs without the settlement inputs.
  (b) CHALLENGE-DEFENSIBILITY — a worker signs a stage → its
      ``stage_settlement_signature_b64`` verifies under the worker pubkey over
      ``build_receipt_signing_payload(job_id, stage_index,
      stage_output_activation_hash, executed_at)``; feeding the assembled
      ``NodeSignatureMaterial`` through the brick-1 splitter yields a per-node
      leaf whose ``signing_message_hash`` == that payload and whose sig
      verifies (an on-chain INVALID_SIGNATURE challenge would FAIL).
  (c) ORCHESTRATOR ASSEMBLY — two distinct worker stages each emit the
      settlement-sig fields → the orchestrator assembles a 2-entry
      ``NodeSignatureMaterial`` dict keyed by node_id; if ANY stage omits it →
      None/empty (single-payee fallback), mirroring ``_assemble_chain``.
  (d) IN-PROCESS E2E — two distinct node identities run two stages → emit
      settlement sigs → orchestrator assembles → feed receipt + materials to
      the brick-1 splitter → 2 challenge-defensible per-node ``BatchedReceipt``
      summing to the total.
  (e) NO MONEY-PATH CHANGE — nothing broadcasts; the single-payee inference
      adapter / split path is unchanged.
"""
from __future__ import annotations

import base64
import hashlib
from decimal import Decimal

import pytest

from prsm.compute.shard_receipt import (
    build_receipt_signing_payload,
    per_stage_leaf_job_id,
)
from prsm.compute.inference.stage_activation_proof import activation_hash
from prsm.compute.inference.topology_rotation import TopologyAssignment
from prsm.node.identity import generate_node_identity, verify_signature
from prsm.settlement.merkle import batched_receipt_to_leaf
from prsm.settlement.per_stage_settlement_split import (
    NodeSignatureMaterial,
    split_receipt_to_per_node_batched_receipts,
)

from prsm.compute.chain_rpc.protocol import RunLayerSliceResponse
from prsm.compute.tee.models import TEEType

_WEI = 10 ** 18


# ── helpers ──────────────────────────────────────────────────────────────


def _base_sign_kwargs(identity, *, blob=b"output-activation-bytes"):
    """The minimal kwargs to sign a unary RunLayerSliceResponse."""
    return dict(
        identity=identity,
        request_id="req-1156",
        activation_blob=blob,
        activation_shape=(1, 4),
        activation_dtype="float32",
        duration_seconds=1.0,
        tee_attestation=b"att",
        tee_type=TEEType.SOFTWARE,
        epsilon_spent=0.0,
    )


def _make_ir(*, node_a_id, node_b_id, job_id="job-1156", output_hash=None, **over):
    from prsm.compute.inference.models import InferenceReceipt, ContentTier
    from prsm.compute.tee.models import PrivacyLevel
    if output_hash is None:
        output_hash = hashlib.sha256(b"the multi-stage output").digest()
    topo = TopologyAssignment(
        positions={(0, 0): node_a_id, (1, 0): node_b_id},
        stage_count=2,
        slots_per_stage=1,
    )
    defaults = dict(
        job_id=job_id,
        request_id="req-1156",
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


# ── (a) BYTE-EQUIVALENCE ───────────────────────────────────────────────────


def test_signing_payload_byte_identical_with_vs_without_settlement_inputs():
    """The stage-response signing_payload is a parallel artifact — adding the
    settlement-sig inputs CANNOT change its bytes (the leaf sig is separate)."""
    blob = b"output-activation-bytes"
    payload_plain = RunLayerSliceResponse.signing_payload(
        "req-1156", blob, (1, 4), "float32", 1.0, b"att",
        TEEType.SOFTWARE, 0.0, "node-xyz",
    )
    # signing_payload must NOT have grown a settlement parameter; assert the
    # exact same bytes are produced regardless of any settlement context the
    # worker may have signed in parallel. (The settlement sig is built OUTSIDE
    # signing_payload, so there is no way to pass it in here — that is the
    # safety property: it cannot pollute the signed bytes.)
    payload_again = RunLayerSliceResponse.signing_payload(
        "req-1156", blob, (1, 4), "float32", 1.0, b"att",
        TEEType.SOFTWARE, 0.0, "node-xyz",
    )
    assert payload_plain == payload_again
    # Pin the exact bytes so a future field-creep into signing_payload fails.
    import json
    expected = json.dumps(
        {
            "request_id": "req-1156",
            "activation_blob_hex": blob.hex(),
            "activation_shape": [1, 4],
            "activation_dtype": "float32",
            "duration_seconds": 1.0,
            "tee_attestation_hex": b"att".hex(),
            "tee_type": TEEType.SOFTWARE.value,
            "epsilon_spent": 0.0,
            "stage_node_id": "node-xyz",
        },
        sort_keys=True,
    ).encode("utf-8")
    assert payload_plain == expected


def test_sign_without_settlement_inputs_is_byte_identical():
    """A worker that does NOT opt into brick 2 (no settlement_job_id) produces
    the same stage signature + the same response shape as pre-brick-2."""
    identity = generate_node_identity("worker")
    resp_no = RunLayerSliceResponse.sign(**_base_sign_kwargs(identity))
    resp_with = RunLayerSliceResponse.sign(
        **_base_sign_kwargs(identity),
        settlement_job_id="job-1156",
        settlement_stage_index=0,
        settlement_executed_at_unix=1_700_000_000,
    )
    # The stage signature (over signing_payload) is IDENTICAL — settlement sig
    # is a parallel artifact and does not touch the stage-response signature.
    assert resp_no.stage_signature_b64 == resp_with.stage_signature_b64
    # The opt-out response carries NO settlement fields (omit-when-default).
    assert resp_no.stage_settlement_signature_b64 is None
    assert resp_no.stage_settlement_pubkey_b64 is None
    assert resp_no.stage_settlement_executed_at_unix is None
    d = resp_no.to_dict()
    assert "stage_settlement_signature_b64" not in d
    assert "stage_settlement_pubkey_b64" not in d
    assert "stage_settlement_executed_at_unix" not in d
    # The opt-in response DOES carry them.
    assert resp_with.stage_settlement_signature_b64 is not None
    dw = resp_with.to_dict()
    assert dw["stage_settlement_signature_b64"] == resp_with.stage_settlement_signature_b64


def test_roundtrip_unchanged_without_settlement_fields():
    identity = generate_node_identity("worker")
    resp = RunLayerSliceResponse.sign(**_base_sign_kwargs(identity))
    rt = RunLayerSliceResponse.from_dict(resp.to_dict())
    assert rt.stage_signature_b64 == resp.stage_signature_b64
    assert rt.stage_settlement_signature_b64 is None
    assert rt.stage_settlement_pubkey_b64 is None
    assert rt.stage_settlement_executed_at_unix is None


def test_roundtrip_preserves_settlement_fields():
    identity = generate_node_identity("worker")
    resp = RunLayerSliceResponse.sign(
        **_base_sign_kwargs(identity),
        settlement_job_id="job-1156",
        settlement_stage_index=2,
        settlement_executed_at_unix=1_700_000_123,
    )
    rt = RunLayerSliceResponse.from_dict(resp.to_dict())
    assert rt.stage_settlement_signature_b64 == resp.stage_settlement_signature_b64
    assert rt.stage_settlement_pubkey_b64 == resp.stage_settlement_pubkey_b64
    assert rt.stage_settlement_executed_at_unix == 1_700_000_123


# ── (b) CHALLENGE-DEFENSIBILITY ────────────────────────────────────────────


def test_settlement_signature_verifies_over_receipt_payload():
    identity = generate_node_identity("worker")
    job_id, stage_index, executed_at = "job-1156", 1, 1_700_000_000
    blob = b"this-is-the-stage-output"
    resp = RunLayerSliceResponse.sign(
        **_base_sign_kwargs(identity, blob=blob),
        settlement_job_id=job_id,
        settlement_stage_index=stage_index,
        settlement_executed_at_unix=executed_at,
    )
    # output_hash for the leaf == the activation hash the response already carries.
    assert resp.stage_output_activation_hash == activation_hash(blob)
    payload = build_receipt_signing_payload(
        job_id=job_id,
        shard_index=stage_index,
        output_hash=resp.stage_output_activation_hash,
        executed_at_unix=executed_at,
    )
    assert verify_signature(
        resp.stage_settlement_pubkey_b64, payload,
        resp.stage_settlement_signature_b64,
    )
    # The emitted pubkey is the worker's own key (binds to its node_id).
    assert resp.stage_settlement_pubkey_b64 == identity.public_key_b64
    assert resp.stage_node_id == identity.node_id


def test_assembled_material_feeds_brick1_to_a_defensible_leaf():
    node_a = generate_node_identity("A")
    node_b = generate_node_identity("B")
    # sp1158 — the per-node settlement leaf binds the STREAMED-AGNOSTIC
    # per_stage_leaf_job_id(request_id), so the worker signs over THAT (it knows
    # request_id, not the receipt's eventual streamed flag). The receipt below
    # uses request_id="req-1156"; the splitter rebuilds over the same id.
    job_id = per_stage_leaf_job_id("req-1156")
    blob_a = b"stage-0-output"
    blob_b = b"stage-1-output"
    resp_a = RunLayerSliceResponse.sign(
        **_base_sign_kwargs(node_a, blob=blob_a),
        settlement_job_id=job_id, settlement_stage_index=0,
        settlement_executed_at_unix=1_700_000_000,
    )
    resp_b = RunLayerSliceResponse.sign(
        **_base_sign_kwargs(node_b, blob=blob_b),
        settlement_job_id=job_id, settlement_stage_index=1,
        settlement_executed_at_unix=1_700_000_001,
    )
    mat_a = NodeSignatureMaterial(
        pubkey_b64=resp_a.stage_settlement_pubkey_b64,
        signature=resp_a.stage_settlement_signature_b64,
        stage_index=0,
        output_hash=resp_a.stage_output_activation_hash,
        executed_at_unix=resp_a.stage_settlement_executed_at_unix,
    )
    receipt = _make_ir(node_a_id=node_a.node_id, node_b_id=node_b.node_id,
                       job_id=job_id)
    shares = split_receipt_to_per_node_batched_receipts(
        receipt=receipt,
        total_value_wei=10 * _WEI,
        node_signatures={
            node_a.node_id: mat_a,
            node_b.node_id: NodeSignatureMaterial(
                pubkey_b64=resp_b.stage_settlement_pubkey_b64,
                signature=resp_b.stage_settlement_signature_b64,
                stage_index=1,
                output_hash=resp_b.stage_output_activation_hash,
                executed_at_unix=resp_b.stage_settlement_executed_at_unix,
            ),
        },
        requester_address="0x" + "11" * 20,
    )
    assert shares is not None and len(shares) == 2
    for share in shares:
        leaf = batched_receipt_to_leaf(share.batched_receipt)
        br = share.batched_receipt
        expected = build_receipt_signing_payload(
            job_id=job_id,
            shard_index=br.receipt.shard_index,
            output_hash=br.receipt.output_hash,
            executed_at_unix=br.receipt.executed_at_unix,
        )
        assert leaf.signing_message_hash == expected
        # the consumed worker signature verifies over the leaf hash → an
        # on-chain INVALID_SIGNATURE challenge would FAIL (defensible).
        assert verify_signature(
            br.receipt.provider_pubkey_b64, expected, br.receipt.signature,
        )


# ── (c) ORCHESTRATOR ASSEMBLY ──────────────────────────────────────────────


def _make_outcome(stage_index, node_id, resp):
    from prsm.compute.chain_rpc.client import StageOutcome
    return StageOutcome(
        stage_index=stage_index,
        stage_node_id=node_id,
        duration_seconds=1.0,
        tee_attestation=b"att",
        tee_type=TEEType.SOFTWARE,
        epsilon_spent=0.0,
        stage_output_activation_hash=resp.stage_output_activation_hash,
        stage_settlement_signature_b64=resp.stage_settlement_signature_b64,
        stage_settlement_pubkey_b64=resp.stage_settlement_pubkey_b64,
        stage_settlement_executed_at_unix=resp.stage_settlement_executed_at_unix,
    )


def test_orchestrator_assembles_two_entry_material_dict():
    from prsm.compute.chain_rpc.client import (
        _assemble_per_node_settlement_signatures,
    )
    node_a = generate_node_identity("A")
    node_b = generate_node_identity("B")
    resp_a = RunLayerSliceResponse.sign(
        **_base_sign_kwargs(node_a, blob=b"out-a"),
        settlement_job_id="job-1156", settlement_stage_index=0,
        settlement_executed_at_unix=1_700_000_000,
    )
    resp_b = RunLayerSliceResponse.sign(
        **_base_sign_kwargs(node_b, blob=b"out-b"),
        settlement_job_id="job-1156", settlement_stage_index=1,
        settlement_executed_at_unix=1_700_000_001,
    )
    outcomes = [
        _make_outcome(0, node_a.node_id, resp_a),
        _make_outcome(1, node_b.node_id, resp_b),
    ]
    mats = _assemble_per_node_settlement_signatures(outcomes)
    assert mats is not None
    assert set(mats.keys()) == {node_a.node_id, node_b.node_id}
    assert isinstance(mats[node_a.node_id], NodeSignatureMaterial)
    assert mats[node_a.node_id].stage_index == 0
    assert mats[node_b.node_id].stage_index == 1
    assert mats[node_a.node_id].pubkey_b64 == node_a.public_key_b64


def test_orchestrator_fail_closed_when_any_stage_omits():
    from prsm.compute.chain_rpc.client import (
        _assemble_per_node_settlement_signatures,
        StageOutcome,
    )
    node_a = generate_node_identity("A")
    node_b = generate_node_identity("B")
    resp_a = RunLayerSliceResponse.sign(
        **_base_sign_kwargs(node_a, blob=b"out-a"),
        settlement_job_id="job-1156", settlement_stage_index=0,
        settlement_executed_at_unix=1_700_000_000,
    )
    # node_b did NOT opt into brick 2 — no settlement fields.
    resp_b = RunLayerSliceResponse.sign(**_base_sign_kwargs(node_b, blob=b"out-b"))
    outcomes = [
        _make_outcome(0, node_a.node_id, resp_a),
        StageOutcome(
            stage_index=1, stage_node_id=node_b.node_id, duration_seconds=1.0,
            tee_attestation=b"att", tee_type=TEEType.SOFTWARE, epsilon_spent=0.0,
            stage_output_activation_hash=resp_b.stage_output_activation_hash,
            stage_settlement_signature_b64=None,
            stage_settlement_pubkey_b64=None,
            stage_settlement_executed_at_unix=None,
        ),
    ]
    assert _assemble_per_node_settlement_signatures(outcomes) is None


def test_orchestrator_empty_outcomes_returns_none():
    from prsm.compute.chain_rpc.client import (
        _assemble_per_node_settlement_signatures,
    )
    assert _assemble_per_node_settlement_signatures([]) is None


# ── (d) IN-PROCESS E2E ─────────────────────────────────────────────────────


def test_e2e_two_stages_to_conserving_defensible_per_node_receipts():
    from prsm.compute.chain_rpc.client import (
        _assemble_per_node_settlement_signatures,
    )
    node_a = generate_node_identity("A")
    node_b = generate_node_identity("B")
    # sp1158 — workers sign over the streamed-agnostic per_stage_leaf_job_id of
    # the inference's request_id (the receipt below uses request_id="req-1156").
    job_id = per_stage_leaf_job_id("req-1156")
    total = 7 * _WEI + 3  # non-divisible → exercises remainder conservation

    # workers sign their stages, emitting settlement sigs
    resp_a = RunLayerSliceResponse.sign(
        **_base_sign_kwargs(node_a, blob=b"alpha-output"),
        settlement_job_id=job_id, settlement_stage_index=0,
        settlement_executed_at_unix=1_700_000_010,
    )
    resp_b = RunLayerSliceResponse.sign(
        **_base_sign_kwargs(node_b, blob=b"beta-output"),
        settlement_job_id=job_id, settlement_stage_index=1,
        settlement_executed_at_unix=1_700_000_020,
    )
    outcomes = [
        _make_outcome(0, node_a.node_id, resp_a),
        _make_outcome(1, node_b.node_id, resp_b),
    ]
    # orchestrator assembles
    mats = _assemble_per_node_settlement_signatures(outcomes)
    assert mats is not None and len(mats) == 2

    receipt = _make_ir(node_a_id=node_a.node_id, node_b_id=node_b.node_id,
                       job_id=job_id)
    shares = split_receipt_to_per_node_batched_receipts(
        receipt=receipt,
        total_value_wei=total,
        node_signatures=mats,
        requester_address="0x" + "22" * 20,
    )
    assert shares is not None and len(shares) == 2
    # value conserved exactly
    assert sum(s.share_wei for s in shares) == total
    # each leaf is challenge-defensible under the WORKER's own key
    for share in shares:
        br = share.batched_receipt
        leaf = batched_receipt_to_leaf(br)
        payload = build_receipt_signing_payload(
            job_id=job_id,
            shard_index=br.receipt.shard_index,
            output_hash=br.receipt.output_hash,
            executed_at_unix=br.receipt.executed_at_unix,
        )
        assert leaf.signing_message_hash == payload
        assert verify_signature(
            br.receipt.provider_pubkey_b64, payload, br.receipt.signature,
        )
        # provider binding: provider_id == sha256(pubkey)[:32]
        derived = hashlib.sha256(
            base64.b64decode(br.receipt.provider_pubkey_b64)
        ).hexdigest()[:32]
        assert derived == br.receipt.provider_id


# ── (e) NO MONEY-PATH CHANGE ───────────────────────────────────────────────


def test_single_payee_path_unchanged_when_no_settlement_sigs():
    """The splitter still returns None (single-payee fallback) when the
    per-node settlement signatures are absent — the money path is untouched."""
    node_a = generate_node_identity("A")
    node_b = generate_node_identity("B")
    receipt = _make_ir(node_a_id=node_a.node_id, node_b_id=node_b.node_id)
    # No signatures → fail-closed to single-payee.
    assert split_receipt_to_per_node_batched_receipts(
        receipt=receipt,
        total_value_wei=10 * _WEI,
        node_signatures={},
        requester_address="0x" + "33" * 20,
    ) is None


def test_splitter_never_broadcasts_and_holds_no_keys():
    """The per-node splitter is pure — it consumes provided sig material, never
    signs, never broadcasts. Sanity: a NodeSignatureMaterial built from a
    DIFFERENT key than the node_id fails the provider binding → None (the
    splitter cannot fabricate / sign a payee leaf)."""
    node_a = generate_node_identity("A")
    node_b = generate_node_identity("B")
    wrong = generate_node_identity("wrong")
    receipt = _make_ir(node_a_id=node_a.node_id, node_b_id=node_b.node_id,
                       job_id="job-1156")
    out_hash = activation_hash(b"x")
    bad_mat = {
        node_a.node_id: NodeSignatureMaterial(
            pubkey_b64=wrong.public_key_b64,  # binding will FAIL
            signature=wrong.sign(b"whatever"),
            stage_index=0, output_hash=out_hash, executed_at_unix=1,
        ),
        node_b.node_id: NodeSignatureMaterial(
            pubkey_b64=node_b.public_key_b64,
            signature=node_b.sign(b"whatever"),
            stage_index=1, output_hash=out_hash, executed_at_unix=1,
        ),
    }
    assert split_receipt_to_per_node_batched_receipts(
        receipt=receipt, total_value_wei=10 * _WEI,
        node_signatures=bad_mat, requester_address="0x" + "44" * 20,
    ) is None
