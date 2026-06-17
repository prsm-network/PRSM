"""Sprint 1141 — §7-Brick-1: the InferenceReceiptStore (compute-integrity retention).

The settlement-receipt data plane (Bricks A-E.2) detects ON-CHAIN settlement fraud
(INVALID_SIGNATURE / DOUBLE_SPEND) by retaining the committed BatchedReceipts. The §7
follow-on extends fraud detection to COMPUTE-INTEGRITY (faked / spliced distributed
inference) via the EXISTING ChallengeWatcher + verify_inference_receipt_for_challenge —
which need the §7 ``InferenceReceipt`` (settler signature + stage_activation_chain), NOT
just the settlement ``BatchedReceipt``.

THE GAP these tests close: the adapter (``inference_receipt_to_batched_receipt``)
RE-SIGNS the shard payload and produces a ``BatchedReceipt``; the §7-specific fields
(stage_activation_chain, the §7 settler_signature, request_id, prompt_hash) do NOT
survive into settlement. So an observer holding a committed batch leaf has no way to
fetch the §7 receipt to run the verifier on it.

THE LINKING-KEY CRUX: an observer sees a committed batch with ``leaf_hashes`` (Brick
A/E.1). To run the §7 verifier on the receipt at a leaf, it must look up the §7
``InferenceReceipt`` by THAT leaf hash. The producer, at adapt time, has the
``BatchedReceipt`` -> its leaf -> ``hash_leaf(batched_receipt_to_leaf(batched))``. So we
retain the §7 ``InferenceReceipt`` keyed by THAT leaf_hash — identical to what the
committed batch will carry — so the lookup works.

These tests drive the REAL adapter (not a mock) and prove:
  (a) put/get round-trips a RetainedInferenceReceipt byte-exact;
  (b) the LEAF-HASH KEY crux: store under hash_leaf(batched_receipt_to_leaf(adapted)),
      look up by that same leaf hash;
  (c) prune drops expired / max_entries evicts oldest;
  (d) DEFAULT-OFF / money-path: adapt+accumulate with NO store is byte-for-byte
      unchanged, and a store whose put() raises does NOT break adapt/accumulate;
  (e) the retained receipt + key feed the EXISTING §7 verifier.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
from decimal import Decimal

import pytest


# ── builders (no network) ──────────────────────────────────────────────


def _ident(name="n1"):
    from prsm.node.identity import generate_node_identity
    return generate_node_identity(display_name=name)


def _make_signed_ir(identity, **over):
    """A genuine, SIGNED §7 InferenceReceipt (settler_signature verifies under the
    identity's pubkey). Mirrors the sp1035 builder but signs the receipt so the §7
    verifier's INVALID_SETTLER_SIGNATURE ground can be exercised."""
    from prsm.compute.inference.models import InferenceReceipt, ContentTier
    from prsm.compute.tee.models import PrivacyLevel, TEEType
    from prsm.compute.inference.receipt import sign_receipt
    defaults = dict(
        job_id="job-1",
        request_id="req-1",
        model_id="gpt2",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=b"att-bytes",
        output_hash=hashlib.sha256(b"the output").digest(),  # 32 raw bytes
        duration_seconds=1.0,
        cost_ftns=Decimal("1.0"),
        settler_node_id=identity.node_id,
    )
    defaults.update(over)
    unsigned = InferenceReceipt(**defaults)
    return sign_receipt(unsigned, identity)


_WEI = 10 ** 18
_ADAPT_KW = dict(
    requester_address="0x" + "11" * 20,
    provider_address="0x" + "22" * 20,
    value_ftns=_WEI,
    local_escrow_id="job-1",
    executed_at_unix=1_700_000_000,
)


def _adapt(receipt, identity):
    from prsm.settlement.inference_adapter import (
        inference_receipt_to_batched_receipt,
    )
    return inference_receipt_to_batched_receipt(
        receipt=receipt, identity=identity, **_ADAPT_KW
    )


def _leaf_hash_of(batched):
    from prsm.settlement.merkle import batched_receipt_to_leaf, hash_leaf
    return hash_leaf(batched_receipt_to_leaf(batched))


# ── (a) put/get round-trips byte-exact ──────────────────────────────────


def test_put_get_roundtrip_byte_exact(tmp_path):
    from prsm.settlement.inference_receipt_store import InferenceReceiptStore

    ident = _ident()
    ir = _make_signed_ir(ident)
    leaf_hash = b"\xab" * 32
    store = InferenceReceiptStore(tmp_path / "ir_store.json")
    store.put(
        leaf_hash,
        ir,
        settler_public_key_b64=ident.public_key_b64,
        stage_public_keys=None,
    )

    got = store.get(leaf_hash)
    assert got is not None
    assert got.leaf_hash == leaf_hash
    assert got.settler_public_key_b64 == ident.public_key_b64
    # The §7-specific bytes survive byte-exact (this is the whole point of the brick).
    assert got.inference_receipt.settler_signature == ir.settler_signature
    assert got.inference_receipt.output_hash == ir.output_hash
    assert got.inference_receipt.request_id == ir.request_id
    assert got.inference_receipt.settler_node_id == ir.settler_node_id
    # to_dict equality proves every field (incl. stage_activation_chain when present).
    assert got.inference_receipt.to_dict() == ir.to_dict()


def test_roundtrip_survives_reload_from_disk(tmp_path):
    from prsm.settlement.inference_receipt_store import InferenceReceiptStore

    ident = _ident()
    ir = _make_signed_ir(ident)
    leaf_hash = b"\xcd" * 32
    path = tmp_path / "ir_store.json"
    store = InferenceReceiptStore(path)
    store.put(leaf_hash, ir, settler_public_key_b64=ident.public_key_b64)

    # Reload from a fresh instance — proves atomic persistence + from_dict.
    reloaded = InferenceReceiptStore(path)
    got = reloaded.get(leaf_hash)
    assert got is not None
    assert got.inference_receipt.to_dict() == ir.to_dict()
    assert got.settler_public_key_b64 == ident.public_key_b64
    assert got.leaf_hash == leaf_hash


# ── (b) THE LEAF-HASH KEY CRUX (drives the REAL adapter) ─────────────────


def test_leaf_hash_key_crux_observer_lookup(tmp_path):
    """Adapt a §7 InferenceReceipt -> BatchedReceipt; compute the SAME leaf hash an
    observer would derive from the committed batch leaf; store the §7 receipt under it;
    assert lookup by that leaf hash returns it. Proves an observer holding a committed
    batch leaf can fetch the §7 receipt to run the §7 verifier."""
    from prsm.settlement.inference_receipt_store import InferenceReceiptStore

    ident = _ident()
    ir = _make_signed_ir(ident)
    batched = _adapt(ir, ident)
    # The observer's view: hash_leaf(batched_receipt_to_leaf(committed leaf)).
    leaf_hash = _leaf_hash_of(batched)

    store = InferenceReceiptStore(tmp_path / "ir_store.json")
    store.put(leaf_hash, ir, settler_public_key_b64=ident.public_key_b64)

    # Independent recomputation of the leaf hash (an observer re-deriving from the
    # committed batch) must hit the SAME key.
    observer_leaf_hash = _leaf_hash_of(batched)
    assert observer_leaf_hash == leaf_hash
    got = store.get(observer_leaf_hash)
    assert got is not None
    assert got.inference_receipt.to_dict() == ir.to_dict()


# ── (c) prune + max_entries eviction ─────────────────────────────────────


def test_prune_drops_expired_keeps_fresh(tmp_path):
    from prsm.settlement.inference_receipt_store import InferenceReceiptStore

    clock = {"t": 1000.0}
    ident = _ident()
    store = InferenceReceiptStore(
        tmp_path / "ir_store.json", now=lambda: clock["t"]
    )
    old = _make_signed_ir(ident, job_id="old", request_id="old")
    store.put(b"\x01" * 32, old, settler_public_key_b64=ident.public_key_b64)

    clock["t"] = 1000.0 + 10_000  # advance well past retention
    fresh = _make_signed_ir(ident, job_id="fresh", request_id="fresh")
    store.put(b"\x02" * 32, fresh, settler_public_key_b64=ident.public_key_b64)

    dropped = store.prune(retention_seconds=5_000)
    assert dropped == 1
    assert store.get(b"\x01" * 32) is None       # expired -> dropped
    assert store.get(b"\x02" * 32) is not None    # fresh -> kept


def test_max_entries_evicts_oldest(tmp_path):
    from prsm.settlement.inference_receipt_store import InferenceReceiptStore

    ident = _ident()
    store = InferenceReceiptStore(tmp_path / "ir_store.json", max_entries=2)
    ir = _make_signed_ir(ident)
    store.put(b"\x01" * 32, ir, settler_public_key_b64=ident.public_key_b64)
    store.put(b"\x02" * 32, ir, settler_public_key_b64=ident.public_key_b64)
    store.put(b"\x03" * 32, ir, settler_public_key_b64=ident.public_key_b64)

    assert store.get(b"\x01" * 32) is None        # oldest evicted
    assert store.get(b"\x02" * 32) is not None
    assert store.get(b"\x03" * 32) is not None


def test_never_raises_on_malformed_on_disk_entry(tmp_path):
    """A corrupt retained entry is skipped+logged, never raised — a single bad entry
    can't take down the producer (retention guards no funds)."""
    import json
    from prsm.settlement.inference_receipt_store import InferenceReceiptStore

    ident = _ident()
    ir = _make_signed_ir(ident)
    path = tmp_path / "ir_store.json"
    store = InferenceReceiptStore(path)
    store.put(b"\x01" * 32, ir, settler_public_key_b64=ident.public_key_b64)

    raw = json.loads(path.read_text())
    # Inject a malformed entry alongside the good one.
    raw["receipts"]["deadbeef"] = {"garbage": True}
    path.write_text(json.dumps(raw))

    reloaded = InferenceReceiptStore(path)  # must NOT raise
    assert reloaded.get(b"\x01" * 32) is not None  # good entry survives


# ── (d) DEFAULT-OFF / money-path safety ──────────────────────────────────


def test_default_off_adapt_byte_for_byte_unchanged(tmp_path):
    """The adapt path with NO store (the default) produces a BatchedReceipt identical
    to one produced without any §7 retention wiring."""
    ident = _ident()
    ir = _make_signed_ir(ident)

    # baseline: direct adapter call (what the money path did before this brick)
    baseline = _adapt(ir, ident)

    # adapt-and-retain with NO store (default None) -> must be byte-identical
    from prsm.settlement.inference_receipt_store import (
        adapt_and_retain_inference_receipt,
    )
    produced = adapt_and_retain_inference_receipt(
        receipt=ir, identity=ident, store=None, **_ADAPT_KW
    )
    from prsm.settlement.published_batch_store import _batched_receipt_to_dict
    assert _batched_receipt_to_dict(produced) == _batched_receipt_to_dict(baseline)


def test_retention_failure_never_breaks_adapt(tmp_path):
    """A store whose put() raises must NOT break adapt — the money path is sacred."""
    from prsm.settlement.inference_receipt_store import (
        adapt_and_retain_inference_receipt,
    )

    class _ExplodingStore:
        def put(self, *a, **k):
            raise RuntimeError("disk on fire")

    ident = _ident()
    ir = _make_signed_ir(ident)
    baseline = _adapt(ir, ident)

    produced = adapt_and_retain_inference_receipt(
        receipt=ir, identity=ident, store=_ExplodingStore(), **_ADAPT_KW
    )
    from prsm.settlement.published_batch_store import _batched_receipt_to_dict
    # The BatchedReceipt is returned, byte-identical to baseline, despite the explosion.
    assert _batched_receipt_to_dict(produced) == _batched_receipt_to_dict(baseline)


def test_adapt_and_retain_stores_under_leaf_hash(tmp_path):
    """When a store IS provided, adapt_and_retain stores the §7 receipt under exactly
    hash_leaf(batched_receipt_to_leaf(produced)) — the observer lookup key."""
    from prsm.settlement.inference_receipt_store import (
        InferenceReceiptStore,
        adapt_and_retain_inference_receipt,
    )

    ident = _ident()
    ir = _make_signed_ir(ident)
    store = InferenceReceiptStore(tmp_path / "ir_store.json")
    produced = adapt_and_retain_inference_receipt(
        receipt=ir,
        identity=ident,
        store=store,
        settler_public_key_b64=ident.public_key_b64,
        **_ADAPT_KW,
    )
    leaf_hash = _leaf_hash_of(produced)
    got = store.get(leaf_hash)
    assert got is not None
    assert got.inference_receipt.to_dict() == ir.to_dict()
    assert got.settler_public_key_b64 == ident.public_key_b64


# ── (e) the retained data feeds the EXISTING §7 verifier ─────────────────


def test_retained_receipt_feeds_existing_verifier_genuine(tmp_path):
    from prsm.settlement.inference_receipt_store import InferenceReceiptStore
    from prsm.settlement.challenge_verifier import (
        verify_inference_receipt_for_challenge,
    )

    ident = _ident()
    ir = _make_signed_ir(ident)
    batched = _adapt(ir, ident)
    leaf_hash = _leaf_hash_of(batched)
    store = InferenceReceiptStore(tmp_path / "ir_store.json")
    store.put(leaf_hash, ir, settler_public_key_b64=ident.public_key_b64)

    retained = store.get(leaf_hash)
    report = verify_inference_receipt_for_challenge(
        retained.inference_receipt,
        settler_public_key_b64=retained.settler_public_key_b64,
    )
    assert report.receipt_ok is True  # genuine receipt: no ground proven


def test_retained_receipt_feeds_existing_verifier_tampered(tmp_path):
    import dataclasses
    from prsm.settlement.inference_receipt_store import InferenceReceiptStore
    from prsm.settlement.challenge_verifier import (
        ChallengeReason,
        verify_inference_receipt_for_challenge,
    )

    ident = _ident()
    ir = _make_signed_ir(ident)
    # Tamper a signed field AFTER signing: the settler signature no longer verifies.
    tampered = dataclasses.replace(
        ir, output_hash=hashlib.sha256(b"a DIFFERENT output").digest()
    )
    batched = _adapt(tampered, ident)
    leaf_hash = _leaf_hash_of(batched)
    store = InferenceReceiptStore(tmp_path / "ir_store.json")
    store.put(leaf_hash, tampered, settler_public_key_b64=ident.public_key_b64)

    retained = store.get(leaf_hash)
    report = verify_inference_receipt_for_challenge(
        retained.inference_receipt,
        settler_public_key_b64=retained.settler_public_key_b64,
    )
    assert report.receipt_ok is False
    proven = {f.reason for f in report.findings if f.proven}
    assert ChallengeReason.INVALID_SETTLER_SIGNATURE in proven
