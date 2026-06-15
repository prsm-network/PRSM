"""Sprint 1124 — Domain-03 review F6: bind layer_range + model_id into the HandoffToken.

The settler token authorized "stage K of request R" but not WHICH layers or model the
worker runs, so a relay could pair a valid token with a request for a different slice.
Now both are committed in the signing payload (tampering invalidates the signature) and
the worker enforces the binding (binds_request).
"""
from __future__ import annotations

import pytest

from prsm.compute.chain_rpc.protocol import HandoffToken
from prsm.node.identity import generate_node_identity


class _Anchor:
    def __init__(self, m):
        self._m = m

    def lookup(self, nid):
        return self._m.get(nid)


def _settler():
    idn = generate_node_identity("settler")
    return idn, _Anchor({idn.node_id: idn.public_key_b64})


def _token(idn, *, layer_range=(0, 6), model_id="gpt2"):
    return HandoffToken.sign(
        identity=idn, request_id="req-1", chain_stage_index=0, chain_total_stages=2,
        deadline_unix=9999999999.0, layer_range=layer_range, model_id=model_id)


# ── signature covers the new fields ────────────────────────────────────────────────

def test_bound_token_verifies():
    idn, anchor = _settler()
    assert _token(idn).verify_with_anchor(anchor) is True


def test_tampering_layer_range_breaks_signature():
    import dataclasses
    idn, anchor = _settler()
    tok = _token(idn, layer_range=(0, 6))
    tampered = dataclasses.replace(tok, layer_range=(0, 12))  # swap the slice
    assert tampered.verify_with_anchor(anchor) is False


def test_tampering_model_id_breaks_signature():
    import dataclasses
    idn, anchor = _settler()
    tok = _token(idn, model_id="gpt2")
    tampered = dataclasses.replace(tok, model_id="llama-70b")
    assert tampered.verify_with_anchor(anchor) is False


# ── binds_request enforcement ────────────────────────────────────────────────────────

def test_binds_request_matches():
    idn, _ = _settler()
    tok = _token(idn, layer_range=(0, 6), model_id="gpt2")
    assert tok.binds_request(layer_range=(0, 6), model_id="gpt2") is True


def test_binds_request_rejects_wrong_layer_range():
    idn, _ = _settler()
    tok = _token(idn, layer_range=(0, 6), model_id="gpt2")
    assert tok.binds_request(layer_range=(6, 12), model_id="gpt2") is False


def test_binds_request_rejects_wrong_model():
    idn, _ = _settler()
    tok = _token(idn, layer_range=(0, 6), model_id="gpt2")
    assert tok.binds_request(layer_range=(0, 6), model_id="evil-model") is False


def test_legacy_token_without_binding_passes_enforce_when_present():
    """Back-compat: a token minted without the binding (pre-1124 settler) is accepted
    for any layer_range/model — the worker can't enforce what the settler didn't sign."""
    idn, _ = _settler()
    legacy = HandoffToken.sign(
        identity=idn, request_id="r", chain_stage_index=0, chain_total_stages=1,
        deadline_unix=9999999999.0)  # no layer_range/model_id
    assert legacy.layer_range is None
    assert legacy.binds_request(layer_range=(3, 9), model_id="anything") is True


# ── wire round-trip + back-compat ──────────────────────────────────────────────────

def test_bound_token_wire_round_trips():
    idn, anchor = _settler()
    tok = _token(idn, layer_range=(2, 8), model_id="distilgpt2")
    rebuilt = HandoffToken.from_dict(tok.to_dict())
    assert rebuilt.layer_range == (2, 8)
    assert rebuilt.model_id == "distilgpt2"
    assert rebuilt.verify_with_anchor(anchor) is True


def test_legacy_token_byte_identical_payload():
    """A token minted WITHOUT the binding must produce a signing payload with no
    layer_range/model_id keys (byte-identical to a pre-1124 token)."""
    payload = HandoffToken.signing_payload(
        "r", "settler-node", 0, 1, 1.0)
    assert b"layer_range" not in payload
    assert b"model_id" not in payload


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
