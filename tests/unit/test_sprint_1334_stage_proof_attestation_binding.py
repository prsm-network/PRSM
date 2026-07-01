"""Sprint 1334 (TEE Tier-B) — bind the stage's TEE attestation into StageActivationProof.

The proof signed the (request, stage, node, in/out activation hashes) but NOT the
tee_attestation — so a node could swap its attestation without invalidating its own signature.
This binds sha256(attestation) into signing_bytes, gated by PRSM_STAGE_PROOF_BIND_ATTESTATION
(default off → byte-identical to pre-1334; on → the swap/strip now breaks the signature).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json

from prsm.compute.inference.stage_activation_proof import (
    StageActivationChain,
    StageActivationProof,
    activation_hash,
    stage_proof_bind_attestation_enabled,
    tee_attestation_hash_for,
    verify_stage_proof,
)
from prsm.node.identity import generate_node_identity

_REQ = "req-1334"
_DOMAIN = "prsm-stage-activation-proof-v1"


class _Anchor:
    def __init__(self, m):
        self._m = m

    def lookup(self, nid):
        return self._m.get(nid)


def _signed(idn, *, att_hash="", idx=0, in_h="aa", out_h="bb"):
    p = StageActivationProof(
        stage_index=idx, stage_node_id=idn.node_id,
        input_activation_hash=in_h, output_activation_hash=out_h,
        tee_attestation_hash=att_hash)
    return dataclasses.replace(p, stage_signature_b64=idn.sign(p.signing_bytes(_REQ)))


# ── omit-when-default: byte-identical to pre-1334 when unbound ─────────────────

def test_unbound_signing_bytes_byte_identical_to_pre_1334():
    p = StageActivationProof(0, "node", "aa", "bb")
    expected = "\n".join([_DOMAIN, "req", "0", "node", "aa", "bb"]).encode("utf-8")
    assert p.signing_bytes("req") == expected


def test_unbound_to_dict_omits_field():
    assert "tee_attestation_hash" not in StageActivationProof(0, "node", "aa", "bb").to_dict()


def test_from_dict_without_field_is_unbound():
    d = {"stage_index": 0, "stage_node_id": "n", "input_activation_hash": "a",
         "output_activation_hash": "b", "stage_signature_b64": "s"}
    assert StageActivationProof.from_dict(d).tee_attestation_hash == ""


def test_unbound_chain_stable_hash_matches_pre_1334():
    p = StageActivationProof(0, "n", "aa", "bb", stage_signature_b64="s")
    chain = StageActivationChain(request_id="r", proofs=[p])
    old_canon = json.dumps({"request_id": "r", "proofs": [{
        "stage_index": 0, "stage_node_id": "n", "input_activation_hash": "aa",
        "output_activation_hash": "bb", "stage_signature_b64": "s"}]}, sort_keys=True)
    assert chain.stable_hash() == hashlib.sha256(old_canon.encode("utf-8")).hexdigest()


# ── bound: labeled line + round-trip ──────────────────────────────────────────

def test_bound_signing_bytes_appends_labeled_line():
    p = StageActivationProof(0, "node", "aa", "bb", tee_attestation_hash="deadbeef")
    assert p.signing_bytes("req").decode("utf-8").endswith("\ntee_attestation_hash=deadbeef")


def test_bound_to_dict_includes_field_and_roundtrips():
    p = StageActivationProof(0, "node", "aa", "bb", tee_attestation_hash="deadbeef")
    d = p.to_dict()
    assert d["tee_attestation_hash"] == "deadbeef"
    assert StageActivationProof.from_dict(d).tee_attestation_hash == "deadbeef"


# ── gate + hash helper ────────────────────────────────────────────────────────

def test_gate_default_off(monkeypatch):
    monkeypatch.delenv("PRSM_STAGE_PROOF_BIND_ATTESTATION", raising=False)
    assert stage_proof_bind_attestation_enabled() is False


def test_hash_helper_off_by_default(monkeypatch):
    monkeypatch.delenv("PRSM_STAGE_PROOF_BIND_ATTESTATION", raising=False)
    assert tee_attestation_hash_for(b"real-quote") == ""


def test_hash_helper_on_returns_sha256(monkeypatch):
    monkeypatch.setenv("PRSM_STAGE_PROOF_BIND_ATTESTATION", "1")
    assert tee_attestation_hash_for(b"real-quote") == hashlib.sha256(b"real-quote").hexdigest()


def test_hash_helper_empty_attestation_always_empty(monkeypatch):
    monkeypatch.setenv("PRSM_STAGE_PROOF_BIND_ATTESTATION", "1")
    assert tee_attestation_hash_for(b"") == ""
    assert tee_attestation_hash_for(None) == ""


# ── the security property: sign/verify + tamper/strip ─────────────────────────

def test_bound_proof_signs_and_verifies():
    idn = generate_node_identity(display_name="w")
    p = _signed(idn, att_hash=activation_hash(b"quote-A"))
    assert verify_stage_proof(p, _REQ, anchor=_Anchor({idn.node_id: idn.public_key_b64})) is True


def test_swapping_attestation_breaks_signature():
    """THE binding: a node signs bound to quote-A; swapping in a different attestation's hash
    invalidates its signature (the head, recomputing the swapped hash, verifies False)."""
    idn = generate_node_identity(display_name="w")
    p = _signed(idn, att_hash=activation_hash(b"quote-A"))
    anchor = _Anchor({idn.node_id: idn.public_key_b64})
    assert verify_stage_proof(p, _REQ, anchor=anchor) is True
    tampered = dataclasses.replace(p, tee_attestation_hash=activation_hash(b"quote-B"))
    assert verify_stage_proof(tampered, _REQ, anchor=anchor) is False


def test_stripping_binding_breaks_signature():
    """Downgrade defense: a bound proof can't have its attestation tie stripped (→ unbound)
    without breaking the signature."""
    idn = generate_node_identity(display_name="w")
    p = _signed(idn, att_hash=activation_hash(b"quote-A"))
    stripped = dataclasses.replace(p, tee_attestation_hash="")
    assert verify_stage_proof(
        stripped, _REQ, anchor=_Anchor({idn.node_id: idn.public_key_b64})) is False


def test_unbound_proof_still_verifies_unbound():
    """Back-compat: an unbound proof (pre-1334 / gate off) signs + verifies as before."""
    idn = generate_node_identity(display_name="w")
    p = _signed(idn, att_hash="")
    assert verify_stage_proof(p, _REQ, anchor=_Anchor({idn.node_id: idn.public_key_b64})) is True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
