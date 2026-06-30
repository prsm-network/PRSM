"""Sprint 1314 (big-model paid settlement, S2) — carry per-node settlement-signature
material on the §7 InferenceReceipt.

The per-stage splitter needs each node's challenge-defensible shard-payload signature, which
the RPC chain executor assembles onto the ChainExecutionResult (chain_rpc/client.py:838) but
which was DROPPED at receipt-build — so the settle path couldn't see it. S2 threads it onto
the receipt, CARRIED out-of-band (NOT in signing_payload, self-securing), so it survives to
where the splitter runs. Default None ⇒ pre-1314 receipts are byte-identical-signed.
"""
from __future__ import annotations

from decimal import Decimal

from prsm.compute.inference.models import ContentTier, InferenceReceipt
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.settlement.per_stage_settlement_split import NodeSignatureMaterial


def _mat(stage, tag):
    return NodeSignatureMaterial(
        pubkey_b64=f"PK-{tag}", signature=f"SIG-{tag}", stage_index=stage,
        output_hash=tag * 64, executed_at_unix=1000 + stage, tee_attestation=None)


def _receipt(**over):
    base = dict(
        job_id="job-1", request_id="req-1", model_id="qwen2.5-72b",
        content_tier=ContentTier.A, privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0, tee_type=TEEType.SOFTWARE,
        tee_attestation=b"att", output_hash=b"\xab" * 32,
        duration_seconds=1.0, cost_ftns=Decimal("0.12"), settler_node_id="s" * 32,
    )
    base.update(over)
    return InferenceReceipt(**base)


# ── back-compat: absent → byte-identical ─────────────────────────────────────

def test_absent_field_omitted_from_to_dict():
    assert "per_stage_settlement_signatures" not in _receipt().to_dict()


def test_signing_payload_UNCHANGED_by_the_carried_field():
    """THE money-path-safety property: the per-stage signatures are CARRIED, not SIGNED — so
    a receipt with them produces byte-identical signing bytes to one without (existing
    receipts + their settler signatures are completely unaffected)."""
    sigs = {"a" * 32: _mat(0, "a"), "b" * 32: _mat(1, "b")}
    r_without = _receipt()
    r_with = _receipt(per_stage_settlement_signatures=sigs)
    assert r_with.signing_payload() == r_without.signing_payload()


def test_old_serialized_receipt_parses(monkeypatch=None):
    d = _receipt().to_dict()           # no per_stage_settlement_signatures key
    r = InferenceReceipt.from_dict(d)
    assert r.per_stage_settlement_signatures is None


# ── round-trip when present ──────────────────────────────────────────────────

def test_to_dict_serializes_each_material():
    sigs = {"a" * 32: _mat(0, "a"), "b" * 32: _mat(1, "b")}
    d = _receipt(per_stage_settlement_signatures=sigs).to_dict()
    ser = d["per_stage_settlement_signatures"]
    assert ser["a" * 32] == _mat(0, "a").to_dict()
    assert ser["b" * 32]["pubkey_b64"] == "PK-b"
    assert ser["a" * 32]["stage_index"] == 0


def test_from_dict_reconstructs_typed_material():
    sigs = {"a" * 32: _mat(0, "a"), "b" * 32: _mat(1, "b")}
    d = _receipt(per_stage_settlement_signatures=sigs).to_dict()
    r2 = InferenceReceipt.from_dict(d)
    got = r2.per_stage_settlement_signatures
    assert isinstance(got["a" * 32], NodeSignatureMaterial)
    assert got["a" * 32] == _mat(0, "a")     # frozen dataclass equality
    assert got["b" * 32] == _mat(1, "b")


def test_full_round_trip_preserves_everything():
    sigs = {"a" * 32: _mat(0, "a"), "b" * 32: _mat(1, "b")}
    r = _receipt(per_stage_settlement_signatures=sigs)
    r2 = InferenceReceipt.from_dict(r.to_dict())
    assert r2.per_stage_settlement_signatures == sigs
    # and the signature/verification surface is unchanged
    assert r2.signing_payload() == r.signing_payload()


def test_material_to_from_dict_round_trip():
    m = _mat(2, "c")
    assert NodeSignatureMaterial.from_dict(m.to_dict()) == m


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
