"""Sprint 927 — bind the request INPUT into the stage-response signature
(compute/inference review #4: streaming/incremental replay).

RunLayerSliceResponse.signing_payload committed the stage to its request_id +
OUTPUT activation + attestation/timing, but NOT to the INPUT it processed nor the
decode iteration. The KV-cache is keyed on request_id, so prefill + every
incremental token in a decode session SHARE one request_id — meaning a malicious
non-tail stage could REPLAY a prior iteration's signed response (e.g. its prefill
output) on a later incremental step and the signature still verified, silently
corrupting the inference output (undetectable, unlike returning garbage which the
signature check would not bless).

Fix (sp927): commit a per-iteration INPUT commitment in the response signature —
sha256 of the inline input activation, or the streamed input's manifest
payload_sha256. The executor supplies the EXPECTED commitment externally at verify
(mirroring expected_stage_node_id), computed from the request IT sent — which
differs every iteration — so a replayed response (bound to a different input)
fails verification. Omit-when-None preserves byte-equivalence: callers that pass
None (prefill / non-decode / pre-fix) produce identical signed bytes.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Optional

from prsm.compute.chain_rpc.protocol import (
    RunLayerSliceResponse,
    request_input_commitment,
)
from prsm.compute.tee.models import TEEType
from prsm.node.identity import generate_node_identity


class FakeAnchor:
    def __init__(self, registered: Optional[Dict[str, str]] = None):
        self.registered: Dict[str, str] = dict(registered or {})

    def lookup(self, node_id: str) -> Optional[str]:
        return self.registered.get(node_id)


def _signed(stage, *, input_commitment=None):
    return RunLayerSliceResponse.sign(
        identity=stage, request_id="req-1", activation_blob=b"output-bytes",
        activation_shape=(1, 4), activation_dtype="float16",
        duration_seconds=0.1, tee_attestation=b"\x01" * 32,
        tee_type=TEEType.SGX, epsilon_spent=0.0,
        input_commitment=input_commitment,
    )


def _anchor(stage):
    return FakeAnchor({stage.node_id: stage.public_key_b64})


# ── the input-commitment binding ─────────────────────────────────────────


def test_matching_input_commitment_verifies():
    stage = generate_node_identity("stage")
    resp = _signed(stage, input_commitment="hash-of-input-X")
    assert resp.verify_with_anchor(
        _anchor(stage), expected_stage_node_id=stage.node_id,
        expected_input_commitment="hash-of-input-X",
    ) is True


def test_replay_on_different_input_rejected():
    # The attack: a response signed for input X is replayed on an iteration
    # whose input is Y → the executor verifies against Y's commitment → reject.
    stage = generate_node_identity("stage")
    resp = _signed(stage, input_commitment="hash-of-input-X")
    assert resp.verify_with_anchor(
        _anchor(stage), expected_stage_node_id=stage.node_id,
        expected_input_commitment="hash-of-input-Y",
    ) is False


def test_none_commitment_is_backward_compatible():
    # Prefill / non-decode / pre-fix: signed + verified with no commitment.
    stage = generate_node_identity("stage")
    resp = _signed(stage, input_commitment=None)
    assert resp.verify_with_anchor(
        _anchor(stage), expected_stage_node_id=stage.node_id,
    ) is True


def test_signing_payload_omits_commitment_when_none_byte_equivalence():
    base = dict(
        request_id="r", activation_blob=b"a", activation_shape=(1,),
        activation_dtype="float32", duration_seconds=0.1,
        tee_attestation=b"\x01" * 4, tee_type=TEEType.SGX,
        epsilon_spent=0.0, stage_node_id="node",
    )
    p_none = RunLayerSliceResponse.signing_payload(
        base["request_id"], base["activation_blob"], base["activation_shape"],
        base["activation_dtype"], base["duration_seconds"],
        base["tee_attestation"], base["tee_type"], base["epsilon_spent"],
        base["stage_node_id"],
    )
    assert b"input_commitment" not in p_none   # byte-equivalent with pre-fix
    p_set = RunLayerSliceResponse.signing_payload(
        base["request_id"], base["activation_blob"], base["activation_shape"],
        base["activation_dtype"], base["duration_seconds"],
        base["tee_attestation"], base["tee_type"], base["epsilon_spent"],
        base["stage_node_id"], input_commitment="hash-of-input-X",
    )
    assert b"input_commitment" in p_set


# ── the request_input_commitment helper (inline vs streamed) ─────────────


def test_request_input_commitment_inline_is_sha256():
    assert request_input_commitment(b"abc", None) == hashlib.sha256(b"abc").hexdigest()


def test_request_input_commitment_streamed_is_manifest_sha():
    class _Manifest:
        payload_sha256 = "deadbeefcafe"
    assert request_input_commitment(b"", _Manifest()) == "deadbeefcafe"


def test_request_input_commitment_differs_per_input():
    # The property the replay defense relies on: distinct inputs → distinct
    # commitments (so consecutive decode iterations can't share one).
    assert request_input_commitment(b"token-5-hidden", None) != \
        request_input_commitment(b"token-6-hidden", None)
