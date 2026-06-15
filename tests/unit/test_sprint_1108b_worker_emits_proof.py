"""Sprint 1108 brick 3 — workers populate the self-securing per-stage activation proof
in RunLayerSliceResponse, and it survives the wire round-trip and verifies.

(The model receptacle is sp1107; the verifier is sp1108. This brick has the worker
PRODUCE the proof in its signed response so the orchestrator can later collect it.)
"""
from __future__ import annotations

import pytest

from prsm.compute.chain_rpc.protocol import RunLayerSliceResponse, parse_message, encode_message
from prsm.compute.inference.stage_activation_proof import (
    StageActivationProof,
    activation_hash,
    verify_stage_proof,
)
from prsm.compute.tee.models import TEEType
from prsm.node.identity import generate_node_identity


class _Anchor:
    def __init__(self, mapping):
        self._m = mapping

    def lookup(self, node_id):
        return self._m.get(node_id)


def _sign(identity, *, req="req-1", inp=b"input-act", out=b"output-act", stage=2):
    return RunLayerSliceResponse.sign(
        identity=identity,
        request_id=req,
        activation_blob=out,
        activation_shape=(len(out),),
        activation_dtype="uint8",
        duration_seconds=0.5,
        tee_attestation=b"att",
        tee_type=TEEType.SOFTWARE,
        epsilon_spent=0.0,
        stage_input_blob=inp,
        chain_stage_index=stage,
    )


def test_sign_populates_proof_fields():
    idn = generate_node_identity("worker")
    resp = _sign(idn, inp=b"in", out=b"out", stage=3)
    assert resp.stage_input_activation_hash == activation_hash(b"in")
    assert resp.stage_output_activation_hash == activation_hash(b"out")
    assert resp.stage_activation_proof_b64


def test_emitted_proof_verifies_against_the_workers_key():
    idn = generate_node_identity("worker")
    resp = _sign(idn, req="req-7", inp=b"in", out=b"out", stage=3)
    proof = StageActivationProof(
        stage_index=3, stage_node_id=idn.node_id,
        input_activation_hash=resp.stage_input_activation_hash,
        output_activation_hash=resp.stage_output_activation_hash,
        stage_signature_b64=resp.stage_activation_proof_b64)
    anchor = _Anchor({idn.node_id: idn.public_key_b64})
    assert verify_stage_proof(proof, "req-7", anchor=anchor) is True
    # wrong request id → fails (the proof binds the request)
    assert verify_stage_proof(proof, "req-OTHER", anchor=anchor) is False


def test_proof_survives_wire_round_trip():
    idn = generate_node_identity("worker")
    resp = _sign(idn, stage=1)
    rebuilt = parse_message(encode_message(resp))
    assert rebuilt.stage_input_activation_hash == resp.stage_input_activation_hash
    assert rebuilt.stage_output_activation_hash == resp.stage_output_activation_hash
    assert rebuilt.stage_activation_proof_b64 == resp.stage_activation_proof_b64


def test_no_proof_when_inputs_not_supplied_is_byte_identical_pre_1108():
    """A response built WITHOUT the new params must omit the fields entirely (the
    existing dispatch paths / pre-1108 peers see byte-identical wire bytes)."""
    idn = generate_node_identity("worker")
    resp = RunLayerSliceResponse.sign(
        identity=idn, request_id="r", activation_blob=b"out",
        activation_shape=(3,), activation_dtype="uint8", duration_seconds=0.1,
        tee_attestation=b"a", tee_type=TEEType.SOFTWARE, epsilon_spent=0.0)
    d = resp.to_dict()
    assert "stage_activation_proof_b64" not in d
    assert "stage_input_activation_hash" not in d
    assert resp.stage_activation_proof_b64 is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
