"""Sprint 1136 — settlement-receipt data plane BRICK B: canonical receipt-set
BLOB + the on-chain-root CROSS-CHECK primitive.

Brick A (sp1135) retains the ordered receipt set for a committed batch
(PublishedBatch). Brick B adds the two PURE primitives that let a producer
SERVE that set and an observer SAFELY consume it (design §3 gate 2, §7 Brick B):

  (1) a canonical, CID-addressable receipt-set BLOB: serialize_receipt_set ->
      deterministic bytes, receipt_set_cid -> the content CID via the existing
      v1 BT-infohash mechanism, deserialize_receipt_set -> bounded round-trip.
  (2) the cross-check gate: verify_receipt_set_against_root recomputes the merkle
      root from a (fetched, untrusted) receipt set via the EXACT producer pipeline
      and compares it to the on-chain merkleRoot; verify_leaf_hashes_against_root
      is the cheap leaf-only double-spend path.

These tests are PURE / offline (no network, no I/O beyond bytes <-> objects):
  (a) serialize -> deserialize round-trips a PublishedBatch byte-exact (incl.
      nested receipt signature + tee_attestation).
  (b) receipt_set_cid is DETERMINISTIC (same blob -> same CID) and CHANGES if any
      receipt byte changes.
  (c) CROSS-CHECK FIDELITY: build a batch, compute its root the SAME way the
      producer does; verify_receipt_set_against_root + verify_leaf_hashes_against_root
      are True against that root.
  (d) TAMPER REJECTION: mutate one receipt field / reorder the set -> the gate
      returns False against the original root.
  (e) malformed / oversized blob -> deserialize raises ValueError, no crash.
  (f) END-TO-END with the real producer path: drive a BatchSettlementClient commit,
      take CommittedBatch.merkle_root + the retained PublishedBatch, and assert the
      cross-check passes — proving the gate matches a REAL commit.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from base64 import b64encode
from unittest.mock import AsyncMock

import pytest

from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.core.torrent_infohash import compute_v1_infohash_single_file
from prsm.settlement.accumulator import (
    AccumulatorConfig,
    BatchedReceipt,
    ReceiptAccumulator,
)
from prsm.settlement.client import BatchSettlementClient
from prsm.settlement.merkle import (
    batched_receipt_to_leaf,
    build_tree_and_proofs,
    hash_leaf,
)
from prsm.settlement.published_batch_store import (
    PublishedBatch,
    PublishedBatchStore,
)
from prsm.settlement.receipt_set_blob import (
    deserialize_receipt_set,
    receipt_set_cid,
    serialize_receipt_set,
    verify_leaf_hashes_against_root,
    verify_receipt_set_against_root,
)


ONE_FTNS = 10**18
PROVIDER_ADDR = "0x" + "b" * 40
REQUESTER_A = "0x" + "a" * 40


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
    """The EXACT producer leaf pipeline (client._build_leaves_root)."""
    return [hash_leaf(batched_receipt_to_leaf(br)) for br in receipts]


def _producer_root(receipts) -> bytes:
    """Recompute the merkle root exactly as client._build_leaves_root does."""
    return build_tree_and_proofs(_producer_leaf_hashes(receipts)).root


def _make_pb(receipts, *, batch_label: bytes = b"batch") -> PublishedBatch:
    return PublishedBatch(
        batch_id=hashlib.sha256(batch_label).digest(),
        merkle_root=_producer_root(receipts),
        commit_timestamp=1700001234,
        receipts=list(receipts),
    )


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def _make_mock_contract():
    mock = AsyncMock()
    _n = {"c": 0}

    async def commit_default(**kwargs):
        _n["c"] += 1
        bid = hashlib.sha256(f"mock-batch-{_n['c']}".encode()).digest()
        return (bid, 1700000000 + _n["c"])

    mock.commit_batch.side_effect = commit_default
    mock.is_finalizable.return_value = False
    mock.finalize_batch.return_value = None
    mock.get_batch_status.return_value = 1
    return mock


# ── (a) serialize -> deserialize byte-exact round-trip ────────────


def test_serialize_deserialize_roundtrips_byte_exact():
    receipts = [_make_batched(0, with_tee=True), _make_batched(1, with_tee=False)]
    pb = _make_pb(receipts, batch_label=b"batch-A")

    blob = serialize_receipt_set(pb)
    assert isinstance(blob, (bytes, bytearray))

    rebuilt = deserialize_receipt_set(blob)
    assert isinstance(rebuilt, PublishedBatch)
    assert rebuilt == pb
    assert rebuilt.batch_id == pb.batch_id
    assert rebuilt.merkle_root == pb.merkle_root
    assert rebuilt.commit_timestamp == pb.commit_timestamp
    assert len(rebuilt.receipts) == 2
    for orig, loaded in zip(pb.receipts, rebuilt.receipts):
        assert loaded == orig
        assert loaded.receipt.signature == orig.receipt.signature
        assert loaded.receipt.tee_attestation == orig.receipt.tee_attestation
        assert loaded.consensus_group_id == orig.consensus_group_id


def test_serialize_is_deterministic_byte_identical():
    """Identical input -> byte-identical blob (canonical, sort_keys, compact)."""
    receipts = [_make_batched(i, with_tee=(i % 2 == 0)) for i in range(4)]
    pb1 = _make_pb(receipts, batch_label=b"det")
    pb2 = _make_pb(receipts, batch_label=b"det")
    assert serialize_receipt_set(pb1) == serialize_receipt_set(pb2)


# ── (b) CID determinism + sensitivity ─────────────────────────────


def test_receipt_set_cid_is_deterministic():
    receipts = [_make_batched(0, with_tee=True), _make_batched(1)]
    blob = serialize_receipt_set(_make_pb(receipts))
    cid1 = receipt_set_cid(blob)
    cid2 = receipt_set_cid(blob)
    assert cid1 == cid2
    # And matches the canonical v1 infohash the ContentProvider would compute.
    assert cid1 == compute_v1_infohash_single_file(blob, name="receipt-set")


def test_receipt_set_cid_changes_when_a_receipt_byte_changes():
    base = [_make_batched(0, with_tee=True), _make_batched(1)]
    cid_base = receipt_set_cid(serialize_receipt_set(_make_pb(base)))

    mutated = list(base)
    mutated[0] = dataclasses.replace(
        base[0],
        receipt=dataclasses.replace(
            base[0].receipt,
            output_hash=hashlib.sha256(b"DIFFERENT-OUTPUT").hexdigest(),
        ),
    )
    cid_mutated = receipt_set_cid(serialize_receipt_set(_make_pb(mutated)))
    assert cid_mutated != cid_base


# ── (c) CROSS-CHECK FIDELITY (the crux) ───────────────────────────


def test_verify_receipt_set_against_producer_root_is_true():
    receipts = [_make_batched(i, with_tee=(i % 2 == 0)) for i in range(5)]
    producer_root = _producer_root(receipts)
    pb = _make_pb(receipts, batch_label=b"fidelity")

    assert verify_receipt_set_against_root(pb, producer_root) is True
    # And it equals the root the PublishedBatch carries.
    assert verify_receipt_set_against_root(pb, pb.merkle_root) is True


def test_verify_leaf_hashes_against_producer_root_is_true():
    receipts = [_make_batched(i) for i in range(5)]
    producer_root = _producer_root(receipts)
    leaf_hashes = _producer_leaf_hashes(receipts)
    assert verify_leaf_hashes_against_root(leaf_hashes, producer_root) is True


def test_single_leaf_set_cross_check():
    receipts = [_make_batched(0, with_tee=True)]
    root = _producer_root(receipts)
    pb = _make_pb(receipts, batch_label=b"single")
    assert verify_receipt_set_against_root(pb, root) is True
    assert verify_leaf_hashes_against_root(_producer_leaf_hashes(receipts), root) is True


# ── (d) TAMPER REJECTION ──────────────────────────────────────────


def test_mutated_receipt_field_fails_against_original_root():
    receipts = [_make_batched(i, with_tee=(i % 2 == 0)) for i in range(4)]
    original_root = _producer_root(receipts)

    tampered = list(receipts)
    tampered[2] = dataclasses.replace(
        receipts[2],
        receipt=dataclasses.replace(
            receipts[2].receipt,
            output_hash=hashlib.sha256(b"forged-output").hexdigest(),
        ),
    )
    forged_pb = _make_pb(tampered, batch_label=b"forged")
    # The gate rejects the altered set against the GENUINE on-chain root.
    assert verify_receipt_set_against_root(forged_pb, original_root) is False
    # Leaf-only path rejects too.
    assert verify_leaf_hashes_against_root(
        _producer_leaf_hashes(tampered), original_root
    ) is False


def test_reordered_set_fails_against_original_root():
    """Order is load-bearing for the merkle build; a reordered set must fail.

    Note on the sorted-pair (OpenZeppelin) convention the producer uses
    (merkle._hash_pair sorts the two children before hashing): a swap WITHIN a
    single bottom pair — e.g. leaves [0,1] -> [1,0] — yields the SAME root, because
    that pair's hash is order-independent by construction. That is a genuine,
    intended property of the tree, not a gap in the gate. To prove the gate
    actually rejects a reordering, we use a rotation that moves a leaf ACROSS pair
    boundaries (changing which leaves are paired), which provably changes the root.
    """
    receipts = [_make_batched(i, with_tee=True) for i in range(4)]
    original_root = _producer_root(receipts)

    # Rotate left by one: leaf 0 moves from pair (0,1) to pair (3,0) — a true
    # cross-boundary reorder that changes the tree shape and thus the root.
    reordered = [receipts[1], receipts[2], receipts[3], receipts[0]]
    assert _producer_root(reordered) != original_root  # sanity: root truly differs
    reordered_pb = _make_pb(reordered, batch_label=b"reorder")
    assert verify_receipt_set_against_root(reordered_pb, original_root) is False


# ── (e) malformed / oversized blob -> ValueError ──────────────────


def test_deserialize_rejects_non_json_bytes():
    with pytest.raises(ValueError):
        deserialize_receipt_set(b"\x00\x01\x02not-json")


def test_deserialize_rejects_oversized_blob():
    """A size cap bounds the parse (JSON-bomb defense) — oversized -> ValueError."""
    huge = b'{"x":"' + b"a" * (64 * 1024 * 1024) + b'"}'
    with pytest.raises(ValueError):
        deserialize_receipt_set(huge)


def test_deserialize_rejects_structurally_invalid_json():
    """Valid JSON but not a PublishedBatch dict -> ValueError, never a mis-parse."""
    with pytest.raises(ValueError):
        deserialize_receipt_set(json.dumps({"not": "a batch"}).encode())


# ── (f) END-TO-END with the REAL producer commit path ─────────────


def test_cross_check_matches_a_real_client_commit(tmp_path):
    cfg = AccumulatorConfig(count_threshold=3, time_threshold_seconds=3600,
                            value_threshold_ftns=10**30)
    acc = ReceiptAccumulator(cfg)
    contract = _make_mock_contract()
    pub_store = PublishedBatchStore(tmp_path / "pub.json")
    client = BatchSettlementClient(
        acc, contract, PROVIDER_ADDR, published_batch_store=pub_store,
    )

    originals = [_make_batched(i, with_tee=True) for i in range(3)]
    for br in originals:
        _run(client.accumulate(br))
    committed = _run(client.commit_ready_batches())
    assert len(committed) == 1
    rec = committed[0]

    retained = pub_store.get(rec.batch_id)
    assert retained is not None

    # The cross-check gate passes against the REAL committed merkle root.
    assert verify_receipt_set_against_root(retained, rec.merkle_root) is True

    # And the full serve/fetch/verify cycle: serialize -> CID -> deserialize ->
    # cross-check, exactly what an observer would do over the wire.
    blob = serialize_receipt_set(retained)
    cid = receipt_set_cid(blob)
    assert cid == compute_v1_infohash_single_file(blob, name="receipt-set")
    fetched = deserialize_receipt_set(blob)
    assert verify_receipt_set_against_root(fetched, rec.merkle_root) is True

    # A tampered fetched set fails against the same committed root.
    bad_receipts = list(fetched.receipts)
    bad_receipts[0] = dataclasses.replace(
        fetched.receipts[0],
        receipt=dataclasses.replace(
            fetched.receipts[0].receipt,
            output_hash=hashlib.sha256(b"attacker").hexdigest(),
        ),
    )
    bad_pb = dataclasses.replace(fetched, receipts=bad_receipts)
    assert verify_receipt_set_against_root(bad_pb, rec.merkle_root) is False
