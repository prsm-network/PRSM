"""Sprint 1142 — §7-Brick-2: PUBLISH §7 InferenceReceipts in the receipt-set blob.

§7-Brick-1 (sp1141, ``inference_receipt_store.py``) retains the §7 ``InferenceReceipt``
keyed by the committed-batch leaf hash so a PRODUCER can later serve it. Data-plane
Brick B (sp1136, ``receipt_set_blob.py``) serializes the committed ``BatchedReceipt`` set
into a canonical, CID-addressable blob and gives the observer the on-chain-root
CROSS-CHECK gate (``verify_receipt_set_against_root``). This brick joins the two: it lets
the §7 receipts RIDE in the SAME published blob as the ``BatchedReceipts`` (one fetch),
as an OPTIONAL section keyed by leaf hash, so an observer can run the §7 verifier on a
peer provider's committed batch.

TRUST MODEL (why an UNTRUSTED §7 section is safe):
  - The ``BatchedReceipts`` are cross-checked against the on-chain merkle root
    (``verify_receipt_set_against_root``, UNCHANGED) — the leaf SET is root-anchored.
  - A §7 receipt is matched to a ``BatchedReceipt`` by LEAF HASH, but it is NOT trusted by
    inclusion: its CONTENT is RE-VERIFIED by ``verify_inference_receipt_for_challenge``
    (settler signature over the signing_payload + activation-chain checks). A forged §7
    receipt either fails the signature re-check OR (if the settler signed a fraudulent
    chain) is itself the provable fraud the verifier surfaces.
  - So the §7 section is UNTRUSTED-but-safe: include it, re-verify on use. It does NOT
    enter the merkle-root cross-check (the §7 settler signature is a DIFFERENT preimage
    than the on-chain leaf; the sp1141/sp1035 adapter re-signs the shard payload).

These tests are PURE / offline (no network):
  (a) serialize_enriched_receipt_set(pb, None) is BYTE-IDENTICAL to
      serialize_receipt_set(pb) — backward-compat: no new key when there are no §7
      receipts.
  (b) round-trip: serialize_enriched -> deserialize_enriched recovers the PublishedBatch
      byte-exact AND the §7 map byte-exact (settler_signature + stage_activation_chain
      survive).
  (c) OLD deserializer still works: deserialize_receipt_set(enriched_blob) returns the
      correct PublishedBatch (ignores the §7 section) — old observers keep functioning.
  (d) CROSS-CHECK UNCHANGED: verify_receipt_set_against_root(pb, root) is identically
      True (genuine) / False (tampered) whether the blob carried §7 receipts or not.
  (e) TRUST: a deserialized §7 receipt feeds the EXISTING §7 verifier — genuine ->
      receipt_ok True; a §7 receipt with a tampered chain / bad settler signature -> the
      verifier surfaces the fraud.
  (f) enrich_from_store maps the batch leaf hashes to the retained §7 receipts (drives
      the REAL adapter + the §7-1 store).
  (g) bounded parse: an oversized / malformed §7 section -> ValueError, no crash.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from base64 import b64encode
from decimal import Decimal

import pytest

from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.challenge_verifier import (
    ChallengeReason,
    verify_inference_receipt_for_challenge,
)
from prsm.settlement.inference_receipt_store import (
    InferenceReceiptStore,
    RetainedInferenceReceipt,
)
from prsm.settlement.merkle import (
    batched_receipt_to_leaf,
    build_tree_and_proofs,
    hash_leaf,
)
from prsm.settlement.published_batch_store import PublishedBatch
from prsm.settlement.receipt_set_blob import (
    deserialize_enriched_receipt_set,
    deserialize_receipt_set,
    enrich_from_store,
    serialize_enriched_receipt_set,
    serialize_receipt_set,
    verify_receipt_set_against_root,
)


ONE_FTNS = 10**18
PROVIDER_ADDR = "0x" + "b" * 40
REQUESTER_A = "0x" + "a" * 40
_WEI = 10**18
_ADAPT_KW = dict(
    requester_address="0x" + "11" * 20,
    provider_address="0x" + "22" * 20,
    value_ftns=_WEI,
    local_escrow_id="job-1",
    executed_at_unix=1_700_000_000,
)


# ── settlement-leaf builders (mirror sp1136) ──────────────────────────────


def _make_batched(shard_index: int = 0, *, with_tee: bool = False) -> BatchedReceipt:
    tee = None
    if with_tee:
        tee = {
            "quote": b64encode(f"quote-{shard_index}".encode()).decode(),
            "nested": {"vendor": "intel", "level": shard_index},
        }
    receipt = ShardExecutionReceipt(
        job_id=f"job-{shard_index}",
        shard_index=shard_index,
        provider_id="provider-id-string",
        provider_pubkey_b64=b64encode(b"pubkey-raw").decode(),
        output_hash=hashlib.sha256(f"out-{shard_index}".encode()).hexdigest(),
        executed_at_unix=1700000000 + shard_index,
        signature=b64encode(f"sig-raw-{shard_index}".encode()).decode(),
        tee_attestation=tee,
    )
    return BatchedReceipt(
        receipt=receipt,
        requester_address=REQUESTER_A,
        provider_address=PROVIDER_ADDR,
        value_ftns=ONE_FTNS + shard_index,
        local_escrow_id=f"escrow-{shard_index}",
        tier_slash_rate_bps=250,
        consensus_group_id=b"\x07" * 32,
    )


def _producer_leaf_hashes(receipts):
    return [hash_leaf(batched_receipt_to_leaf(br)) for br in receipts]


def _producer_root(receipts) -> bytes:
    return build_tree_and_proofs(_producer_leaf_hashes(receipts)).root


def _make_pb(receipts, *, batch_label: bytes = b"batch") -> PublishedBatch:
    return PublishedBatch(
        batch_id=hashlib.sha256(batch_label).digest(),
        merkle_root=_producer_root(receipts),
        commit_timestamp=1700001234,
        receipts=list(receipts),
    )


# ── §7 InferenceReceipt builders (mirror sp1141, drive the REAL adapter) ──


def _ident(name="n1"):
    from prsm.node.identity import generate_node_identity
    return generate_node_identity(display_name=name)


def _make_signed_ir(identity, **over):
    """A genuine, SIGNED §7 InferenceReceipt (settler_signature verifies under the
    identity's pubkey)."""
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
        output_hash=hashlib.sha256(b"the output").digest(),
        duration_seconds=1.0,
        cost_ftns=Decimal("1.0"),
        settler_node_id=identity.node_id,
    )
    defaults.update(over)
    unsigned = InferenceReceipt(**defaults)
    return sign_receipt(unsigned, identity)


def _adapt(receipt, identity):
    from prsm.settlement.inference_adapter import (
        inference_receipt_to_batched_receipt,
    )
    return inference_receipt_to_batched_receipt(
        receipt=receipt, identity=identity, **_ADAPT_KW
    )


def _retained_for(ir, identity, leaf_hash, *, stage_public_keys=None):
    """A RetainedInferenceReceipt as §7-1 would produce it (without going through a
    store), so we can build a §7 map for the blob directly."""
    return RetainedInferenceReceipt(
        leaf_hash=bytes(leaf_hash),
        inference_receipt=ir,
        settler_public_key_b64=identity.public_key_b64,
        stage_public_keys=stage_public_keys,
        retained_at=1_700_000_500,
    )


# ── (a) backward-compat: enriched(None) == plain ──────────────────────────


def test_enriched_none_is_byte_identical_to_plain():
    receipts = [_make_batched(i, with_tee=(i % 2 == 0)) for i in range(3)]
    pb = _make_pb(receipts, batch_label=b"compat")

    plain = serialize_receipt_set(pb)
    assert serialize_enriched_receipt_set(pb, None) == plain
    assert serialize_enriched_receipt_set(pb, inference_receipts={}) == plain
    # No "inference_receipts" key is emitted when the section is empty/absent.
    assert b"inference_receipts" not in serialize_enriched_receipt_set(pb, None)


# ── (b) round-trip: PublishedBatch + §7 map both byte-exact ───────────────


def test_enriched_roundtrip_recovers_batch_and_s7_map():
    receipts = [_make_batched(i, with_tee=True) for i in range(3)]
    pb = _make_pb(receipts, batch_label=b"rt")

    # Build genuine §7 receipts for two of the three leaves, keyed by the SAME leaf
    # hash an observer derives from the committed batch.
    ident = _ident()
    s7_map = {}
    leaf_hashes = _producer_leaf_hashes(receipts)
    for i in (0, 2):
        ir = _make_signed_ir(ident, job_id=f"job-{i}", request_id=f"req-{i}")
        s7_map[leaf_hashes[i]] = _retained_for(ir, ident, leaf_hashes[i])

    blob = serialize_enriched_receipt_set(pb, inference_receipts=s7_map)
    rebuilt_pb, rebuilt_s7 = deserialize_enriched_receipt_set(blob)

    # PublishedBatch recovers byte-exact.
    assert rebuilt_pb == pb
    for orig, loaded in zip(pb.receipts, rebuilt_pb.receipts):
        assert loaded.receipt.signature == orig.receipt.signature
        assert loaded.receipt.tee_attestation == orig.receipt.tee_attestation

    # §7 map recovers byte-exact: keys are leaf-hash bytes, values survive byte-exact.
    assert set(rebuilt_s7.keys()) == set(s7_map.keys())
    for k, orig_rec in s7_map.items():
        loaded_rec = rebuilt_s7[k]
        assert loaded_rec.to_dict() == orig_rec.to_dict()
        # The §7-load-bearing bytes survive: settler_signature + activation chain.
        assert (
            loaded_rec.inference_receipt.settler_signature
            == orig_rec.inference_receipt.settler_signature
        )
        assert (
            loaded_rec.inference_receipt.stage_activation_chain
            == orig_rec.inference_receipt.stage_activation_chain
        )
        assert loaded_rec.settler_public_key_b64 == orig_rec.settler_public_key_b64


def test_enriched_deserialize_empty_section_yields_empty_map():
    receipts = [_make_batched(0)]
    pb = _make_pb(receipts, batch_label=b"empty-s7")
    blob = serialize_enriched_receipt_set(pb, None)
    rebuilt_pb, rebuilt_s7 = deserialize_enriched_receipt_set(blob)
    assert rebuilt_pb == pb
    assert rebuilt_s7 == {}


# ── (c) OLD deserializer still works on an enriched blob ──────────────────


def test_old_deserializer_ignores_s7_section():
    """An old observer (pre-§7-Brick-2) runs deserialize_receipt_set on the enriched
    blob and still recovers the correct PublishedBatch, ignoring the §7 section."""
    receipts = [_make_batched(i, with_tee=(i == 0)) for i in range(2)]
    pb = _make_pb(receipts, batch_label=b"old-reader")

    ident = _ident()
    leaf_hashes = _producer_leaf_hashes(receipts)
    s7_map = {
        leaf_hashes[0]: _retained_for(
            _make_signed_ir(ident), ident, leaf_hashes[0]
        )
    }
    enriched = serialize_enriched_receipt_set(pb, inference_receipts=s7_map)

    # The EXISTING (Brick B) deserializer, unmodified contract, must still work.
    old_view = deserialize_receipt_set(enriched)
    assert isinstance(old_view, PublishedBatch)
    assert old_view == pb
    assert old_view.merkle_root == pb.merkle_root
    assert len(old_view.receipts) == 2


# ── (d) CROSS-CHECK UNCHANGED with or without the §7 section ──────────────


def test_cross_check_identical_with_and_without_s7():
    receipts = [_make_batched(i, with_tee=True) for i in range(5)]
    pb = _make_pb(receipts, batch_label=b"xcheck")
    root = pb.merkle_root

    ident = _ident()
    leaf_hashes = _producer_leaf_hashes(receipts)
    s7_map = {
        leaf_hashes[0]: _retained_for(_make_signed_ir(ident), ident, leaf_hashes[0]),
        leaf_hashes[3]: _retained_for(_make_signed_ir(ident), ident, leaf_hashes[3]),
    }

    # GENUINE: True both ways.
    plain_pb = deserialize_receipt_set(serialize_receipt_set(pb))
    enriched_pb, _ = deserialize_enriched_receipt_set(
        serialize_enriched_receipt_set(pb, inference_receipts=s7_map)
    )
    assert verify_receipt_set_against_root(plain_pb, root) is True
    assert verify_receipt_set_against_root(enriched_pb, root) is True

    # TAMPERED: False both ways. Tamper a BatchedReceipt and rebuild both blobs.
    tampered = list(receipts)
    tampered[2] = dataclasses.replace(
        receipts[2],
        receipt=dataclasses.replace(
            receipts[2].receipt,
            output_hash=hashlib.sha256(b"forged").hexdigest(),
        ),
    )
    # Keep the ORIGINAL (genuine) root; only the receipts are altered.
    tampered_pb = dataclasses.replace(pb, receipts=tampered)
    plain_tampered = deserialize_receipt_set(serialize_receipt_set(tampered_pb))
    enriched_tampered, _ = deserialize_enriched_receipt_set(
        serialize_enriched_receipt_set(tampered_pb, inference_receipts=s7_map)
    )
    assert verify_receipt_set_against_root(plain_tampered, root) is False
    assert verify_receipt_set_against_root(enriched_tampered, root) is False


# ── (e) TRUST: deserialized §7 receipt re-verifies (genuine + fraud) ──────


def test_deserialized_s7_receipt_verifies_genuine():
    receipts = [_make_batched(0)]
    pb = _make_pb(receipts, batch_label=b"trust-genuine")
    leaf_hash = _producer_leaf_hashes(receipts)[0]

    ident = _ident()
    ir = _make_signed_ir(ident)
    s7_map = {leaf_hash: _retained_for(ir, ident, leaf_hash)}

    blob = serialize_enriched_receipt_set(pb, inference_receipts=s7_map)
    _, rebuilt_s7 = deserialize_enriched_receipt_set(blob)

    rec = rebuilt_s7[leaf_hash]
    report = verify_inference_receipt_for_challenge(
        rec.inference_receipt,
        settler_public_key_b64=rec.settler_public_key_b64,
    )
    assert report.receipt_ok is True  # untrusted-but-re-verified -> genuine


def test_deserialized_s7_receipt_surfaces_fraud():
    """A §7 receipt whose signed field was tampered AFTER signing: the settler
    signature no longer verifies, so the re-verify on the UNTRUSTED §7 section surfaces
    the fraud (untrusted-but-safe: inclusion grants no trust)."""
    receipts = [_make_batched(0)]
    pb = _make_pb(receipts, batch_label=b"trust-fraud")
    leaf_hash = _producer_leaf_hashes(receipts)[0]

    ident = _ident()
    ir = _make_signed_ir(ident)
    tampered = dataclasses.replace(
        ir, output_hash=hashlib.sha256(b"a DIFFERENT output").digest()
    )
    s7_map = {leaf_hash: _retained_for(tampered, ident, leaf_hash)}

    blob = serialize_enriched_receipt_set(pb, inference_receipts=s7_map)
    _, rebuilt_s7 = deserialize_enriched_receipt_set(blob)

    rec = rebuilt_s7[leaf_hash]
    report = verify_inference_receipt_for_challenge(
        rec.inference_receipt,
        settler_public_key_b64=rec.settler_public_key_b64,
    )
    assert report.receipt_ok is False
    proven = {f.reason for f in report.findings if f.proven}
    assert ChallengeReason.INVALID_SETTLER_SIGNATURE in proven


# ── (f) enrich_from_store drives the REAL adapter + §7-1 store ────────────


def test_enrich_from_store_maps_leaf_hashes_to_retained(tmp_path):
    """Drive the REAL adapter: build §7 receipts, adapt to BatchedReceipts, retain in a
    real InferenceReceiptStore keyed by leaf hash, build a PublishedBatch over those
    same BatchedReceipts, then prove enrich_from_store recovers the §7 map keyed by the
    batch's leaf hashes."""
    ident = _ident()

    irs = [_make_signed_ir(ident, job_id=f"j{i}", request_id=f"r{i}") for i in range(3)]
    batched = [_adapt(ir, ident) for ir in irs]
    leaf_hashes = [hash_leaf(batched_receipt_to_leaf(br)) for br in batched]

    store = InferenceReceiptStore(tmp_path / "ir_store.json")
    # Retain §7 receipts for leaves 0 and 2 only (leaf 1 has no §7 receipt -> absent).
    for i in (0, 2):
        store.put(leaf_hashes[i], irs[i], settler_public_key_b64=ident.public_key_b64)

    pb = PublishedBatch(
        batch_id=hashlib.sha256(b"enrich").digest(),
        merkle_root=_producer_root(batched),
        commit_timestamp=1700009999,
        receipts=list(batched),
    )

    s7_map = enrich_from_store(pb, store)
    # Only the present ones are collected, keyed by the batch's leaf hashes.
    assert set(s7_map.keys()) == {leaf_hashes[0], leaf_hashes[2]}
    assert s7_map[leaf_hashes[0]].inference_receipt.to_dict() == irs[0].to_dict()
    assert s7_map[leaf_hashes[2]].inference_receipt.to_dict() == irs[2].to_dict()

    # End-to-end: the enriched blob round-trips and each §7 receipt re-verifies genuine.
    blob = serialize_enriched_receipt_set(pb, inference_receipts=s7_map)
    rebuilt_pb, rebuilt_s7 = deserialize_enriched_receipt_set(blob)
    assert verify_receipt_set_against_root(rebuilt_pb, pb.merkle_root) is True
    for lh, rec in rebuilt_s7.items():
        report = verify_inference_receipt_for_challenge(
            rec.inference_receipt,
            settler_public_key_b64=rec.settler_public_key_b64,
        )
        assert report.receipt_ok is True


# ── (g) bounded parse: oversized / malformed §7 section -> ValueError ─────


def test_deserialize_enriched_rejects_oversized_blob():
    huge = b'{"x":"' + b"a" * (64 * 1024 * 1024) + b'"}'
    with pytest.raises(ValueError):
        deserialize_enriched_receipt_set(huge)


def test_deserialize_enriched_rejects_non_object_s7_section():
    receipts = [_make_batched(0)]
    pb = _make_pb(receipts, batch_label=b"bad-s7-type")
    raw = json.loads(serialize_receipt_set(pb).decode("utf-8"))
    raw["inference_receipts"] = ["not", "an", "object"]  # must be a dict
    with pytest.raises(ValueError):
        deserialize_enriched_receipt_set(json.dumps(raw).encode("utf-8"))


def test_deserialize_enriched_rejects_oversized_s7_section_count():
    """A §7 section with absurdly many entries is rejected before reconstruction
    (JSON-bomb / oversized-section defense)."""
    receipts = [_make_batched(0)]
    pb = _make_pb(receipts, batch_label=b"too-many-s7")
    raw = json.loads(serialize_receipt_set(pb).decode("utf-8"))
    # Far beyond any real batch (a batch is ~1000 receipts).
    raw["inference_receipts"] = {f"{i:064x}": {"garbage": True} for i in range(200_001)}
    with pytest.raises(ValueError):
        deserialize_enriched_receipt_set(json.dumps(raw).encode("utf-8"))


def test_deserialize_enriched_rejects_malformed_s7_entry():
    """A structurally-malformed §7 entry -> ValueError (never a silent mis-parse)."""
    receipts = [_make_batched(0)]
    pb = _make_pb(receipts, batch_label=b"malformed-s7")
    leaf_hash = _producer_leaf_hashes(receipts)[0]
    raw = json.loads(serialize_receipt_set(pb).decode("utf-8"))
    raw["inference_receipts"] = {leaf_hash.hex(): {"garbage": True}}
    with pytest.raises(ValueError):
        deserialize_enriched_receipt_set(json.dumps(raw).encode("utf-8"))


def test_deserialize_enriched_rejects_non_hex_s7_key():
    receipts = [_make_batched(0)]
    pb = _make_pb(receipts, batch_label=b"bad-key")
    ident = _ident()
    leaf_hash = _producer_leaf_hashes(receipts)[0]
    rec = _retained_for(_make_signed_ir(ident), ident, leaf_hash)
    raw = json.loads(serialize_receipt_set(pb).decode("utf-8"))
    raw["inference_receipts"] = {"not-hex-key!!": rec.to_dict()}
    with pytest.raises(ValueError):
        deserialize_enriched_receipt_set(json.dumps(raw).encode("utf-8"))
