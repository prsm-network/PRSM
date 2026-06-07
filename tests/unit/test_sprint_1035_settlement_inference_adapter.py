"""Sprint 1035 — InferenceReceipt → BatchedReceipt settlement adapter.

First key-free brick of on-chain cross-node settlement (the Tier-1 completion).
The parallax InferenceReceipt and the settlement-layer ShardExecutionReceipt are
DISJOINT types; this adapter bridges them into a BatchedReceipt that the
ReceiptAccumulator + BatchSettlementRegistry consume.

Load-bearing facts the adapter must honor (scouted against the real code):
- InferenceReceipt.output_hash is raw bytes → ShardExecutionReceipt.output_hash
  must be the 64-char hex digest.
- The parallax settler_signature signs InferenceReceipt.signing_payload, a
  DIFFERENT preimage than build_receipt_signing_payload (the shard payload the
  on-chain leaf binds). Copying it would yield a leaf that fails an on-chain
  INVALID_SIGNATURE challenge → wrongful slash. So the settling node RE-SIGNS.
- Single-payee: the local identity must BE the receipt's settler (per-stage
  payee mapping across topology nodes is a follow-on).

THE money test: batched_receipt_to_leaf(adapter_output) must build a stable,
challenge-defensible leaf (signature verifies over the leaf's signing-message
preimage), so a committed inference receipt can actually be defended on-chain.
"""
from __future__ import annotations

import base64
import hashlib
from decimal import Decimal


def _ident(name="n1"):
    from prsm.node.identity import generate_node_identity
    return generate_node_identity(display_name=name)


def _make_ir(identity, **over):
    from prsm.compute.inference.models import InferenceReceipt, ContentTier
    from prsm.compute.tee.models import PrivacyLevel, TEEType
    defaults = dict(
        job_id="job-1",
        request_id="req-1",
        model_id="gpt2",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=b"att",
        output_hash=hashlib.sha256(b"the output").digest(),  # 32 raw bytes
        duration_seconds=1.0,
        cost_ftns=Decimal("1.0"),
        settler_signature=b"parallax-signature-bytes",
        settler_node_id=identity.node_id,
    )
    defaults.update(over)
    return InferenceReceipt(**defaults)


_WEI = 10 ** 18
_KW = dict(
    requester_address="0x" + "11" * 20,
    provider_address="0x" + "22" * 20,
    value_ftns=_WEI,
    local_escrow_id="job-1",
    executed_at_unix=1_700_000_000,
)


def test_builds_batched_receipt_with_inner_shard_receipt():
    from prsm.settlement.inference_adapter import (
        inference_receipt_to_batched_receipt,
    )
    ident = _ident()
    ir = _make_ir(ident)
    br = inference_receipt_to_batched_receipt(receipt=ir, identity=ident, **_KW)
    r = br.receipt
    assert r.job_id == "job-1"
    assert r.shard_index == 0
    assert r.provider_id == ident.node_id
    assert r.provider_pubkey_b64 == ident.public_key_b64
    assert r.output_hash == ir.output_hash.hex()          # raw bytes → hex
    assert r.executed_at_unix == 1_700_000_000
    base64.b64decode(r.signature)                          # valid base64
    assert br.value_ftns == _WEI
    assert br.requester_address == _KW["requester_address"]
    assert br.provider_address == _KW["provider_address"]
    assert br.local_escrow_id == "job-1"


def test_resigns_over_shard_payload_not_parallax_sig():
    from prsm.settlement.inference_adapter import (
        inference_receipt_to_batched_receipt,
    )
    from prsm.compute.shard_receipt import build_receipt_signing_payload
    from prsm.node.identity import verify_signature
    ident = _ident()
    ir = _make_ir(ident)
    br = inference_receipt_to_batched_receipt(receipt=ir, identity=ident, **_KW)
    r = br.receipt
    payload = build_receipt_signing_payload(
        job_id=r.job_id, shard_index=r.shard_index,
        output_hash=r.output_hash, executed_at_unix=r.executed_at_unix,
    )
    # The signature verifies over the SHARD payload (defensible leaf) ...
    assert verify_signature(r.provider_pubkey_b64, payload, r.signature) is True
    # ... and is NOT merely the copied parallax settler_signature.
    assert r.signature != base64.b64encode(ir.settler_signature).decode()


def test_leaf_is_reconstructable_and_stable():
    """The produced BatchedReceipt yields a valid on-chain leaf (no raise) with a
    stable 32-byte hash — so a committed inference receipt can be defended."""
    from prsm.settlement.inference_adapter import (
        inference_receipt_to_batched_receipt,
    )
    from prsm.settlement.merkle import batched_receipt_to_leaf, hash_leaf
    ident = _ident()
    br = inference_receipt_to_batched_receipt(
        receipt=_make_ir(ident), identity=ident, **_KW,
    )
    h1 = hash_leaf(batched_receipt_to_leaf(br))
    assert isinstance(h1, (bytes, bytearray)) and len(h1) == 32
    # deterministic for identical inputs
    br2 = inference_receipt_to_batched_receipt(
        receipt=_make_ir(ident), identity=ident, **_KW,
    )
    assert hash_leaf(batched_receipt_to_leaf(br2)) == h1


def test_provider_binding_holds():
    from prsm.settlement.inference_adapter import (
        inference_receipt_to_batched_receipt,
    )
    ident = _ident()
    br = inference_receipt_to_batched_receipt(
        receipt=_make_ir(ident), identity=ident, **_KW,
    )
    r = br.receipt
    derived = hashlib.sha256(
        base64.b64decode(r.provider_pubkey_b64)
    ).hexdigest()[:32]
    assert r.provider_id == derived       # on-chain provider binding verifies


def test_refuses_mismatched_settler():
    import pytest
    from prsm.settlement.inference_adapter import (
        inference_receipt_to_batched_receipt,
    )
    signer, other = _ident("a"), _ident("b")
    ir = _make_ir(signer)                 # settler == signer
    with pytest.raises(ValueError):
        inference_receipt_to_batched_receipt(
            receipt=ir, identity=other, **_KW,   # other != settler
        )


def test_requires_identity():
    import pytest
    from prsm.settlement.inference_adapter import (
        inference_receipt_to_batched_receipt,
    )
    ir = _make_ir(_ident())
    with pytest.raises(ValueError):
        inference_receipt_to_batched_receipt(receipt=ir, identity=None, **_KW)


def test_tier_slash_and_consensus_passthrough():
    from prsm.settlement.inference_adapter import (
        inference_receipt_to_batched_receipt,
    )
    ident = _ident()
    br = inference_receipt_to_batched_receipt(
        receipt=_make_ir(ident), identity=ident, **_KW,
    )
    assert br.tier_slash_rate_bps == 0
    assert br.consensus_group_id == b"\x00" * 32
    br2 = inference_receipt_to_batched_receipt(
        receipt=_make_ir(ident), identity=ident,
        tier_slash_rate_bps=500, consensus_group_id=b"\x01" * 32, **_KW,
    )
    assert br2.tier_slash_rate_bps == 500
    assert br2.consensus_group_id == b"\x01" * 32


# ── Fail-fast boundary validations (adversarial-review hardening) ──────────────

def _adapt(**over):
    from prsm.settlement.inference_adapter import (
        inference_receipt_to_batched_receipt,
    )
    ident = over.pop("identity", None) or _ident()
    receipt = over.pop("receipt", None) or _make_ir(ident)
    kw = dict(_KW)
    kw.update(over)
    return inference_receipt_to_batched_receipt(
        receipt=receipt, identity=ident, **kw,
    )


def test_rejects_non_32byte_output_hash():
    import pytest
    ident = _ident()
    with pytest.raises(ValueError):
        _adapt(identity=ident, receipt=_make_ir(ident, output_hash=b"short"))


def test_rejects_hex_str_output_hash():
    """A hex str (not raw bytes) must give a clean domain error, not AttributeError."""
    import pytest
    ident = _ident()
    with pytest.raises(ValueError):
        _adapt(identity=ident, receipt=_make_ir(ident, output_hash="ab" * 32))


def test_rejects_lossy_value_ftns():
    """Decimal/float/bool must NOT be silently int()-truncated (wrong settled amount)."""
    import pytest
    from decimal import Decimal
    for bad in (Decimal("1.9"), 1.0, True, "100"):
        with pytest.raises((ValueError, TypeError)):
            _adapt(value_ftns=bad)


def test_rejects_out_of_range_value_ftns():
    import pytest
    for bad in (-1, 2 ** 128):
        with pytest.raises(ValueError):
            _adapt(value_ftns=bad)


def test_accepts_zero_value_ftns():
    """0 wei is a valid (refund-equivalent) amount and must not be rejected."""
    br = _adapt(value_ftns=0)
    assert br.value_ftns == 0


def test_rejects_bad_addresses():
    import pytest
    for bad in ("0xnothexvalue", "", "0x123", "11" * 20, "0x" + "zz" * 20):
        with pytest.raises(ValueError):
            _adapt(requester_address=bad)
        with pytest.raises(ValueError):
            _adapt(provider_address=bad)


def test_rejects_empty_local_escrow_id():
    import pytest
    with pytest.raises(ValueError):
        _adapt(local_escrow_id="")


def test_rejects_out_of_range_executed_at_unix():
    import pytest
    for bad in (-1, 2 ** 64):
        with pytest.raises(ValueError):
            _adapt(executed_at_unix=bad)


def test_rejects_empty_settler():
    """An empty settler_node_id means the payee is undeterminable — refuse."""
    import pytest
    ident = _ident()
    with pytest.raises(ValueError):
        _adapt(identity=ident, receipt=_make_ir(ident, settler_node_id=""))


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
