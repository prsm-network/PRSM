"""Sprint 1135 — settlement-receipt data plane BRICK A: producer retention store.

The producer discards the ordered receipt set after commit (commit_ready_batches
pops the batch; CommittedBatch keeps only leaf_hashes). Brick A retains the full
ordered BatchedReceipt set keyed by batch_id so a node can later self-verify
(recompute the committed merkle root from the retained receipts) and — future
bricks — publish them for cross-node auditing.

These tests are pure / offline (tmp_path, no network):
  (a) put→get round-trips a PublishedBatch byte-exact (incl. nested
      ShardExecutionReceipt signature + tee_attestation).
  (b) ROUND-TRIP FIDELITY (the crux): a retained batch reloaded from disk
      recomputes to the SAME merkle root that was committed.
  (c) prune drops entries past commit_timestamp + retention, keeps fresh ones.
  (d) max_batches bound evicts the oldest.
  (e) MONEY-PATH-SAFE: the client with no store commits exactly as before; with a
      store it retains the committed set; a store whose put() raises does NOT break
      the commit.
"""
from __future__ import annotations

import hashlib
from base64 import b64encode
from unittest.mock import AsyncMock

import pytest

from prsm.compute.shard_receipt import ShardExecutionReceipt
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


def _root_of(receipts) -> bytes:
    """Recompute the merkle root exactly as client._build_leaves_root does."""
    leaf_hashes = [hash_leaf(batched_receipt_to_leaf(br)) for br in receipts]
    return build_tree_and_proofs(leaf_hashes).root


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


# ── (a) byte-exact round-trip ─────────────────────────────────────


def test_put_get_roundtrips_published_batch_byte_exact(tmp_path):
    store = PublishedBatchStore(tmp_path / "published.json")
    receipts = [_make_batched(0, with_tee=True), _make_batched(1, with_tee=False)]
    root = _root_of(receipts)
    batch_id = hashlib.sha256(b"batch-A").digest()

    store.put(batch_id, receipts, root, commit_timestamp=1700001234)

    got = store.get(batch_id)
    assert got is not None
    assert isinstance(got, PublishedBatch)
    assert got.batch_id == batch_id
    assert got.merkle_root == root
    assert got.commit_timestamp == 1700001234
    assert len(got.receipts) == 2
    # Field-for-field equality of the wrapper + nested receipt (incl.
    # signature bytes and the tee_attestation dict).
    for orig, loaded in zip(receipts, got.receipts):
        assert loaded == orig
        assert loaded.receipt.signature == orig.receipt.signature
        assert loaded.receipt.tee_attestation == orig.receipt.tee_attestation
        assert loaded.consensus_group_id == orig.consensus_group_id
        assert loaded.tier_slash_rate_bps == orig.tier_slash_rate_bps


def test_to_dict_from_dict_roundtrips(tmp_path):
    receipts = [_make_batched(3, with_tee=True)]
    root = _root_of(receipts)
    pb = PublishedBatch(
        batch_id=hashlib.sha256(b"x").digest(),
        merkle_root=root,
        commit_timestamp=42,
        receipts=receipts,
    )
    rebuilt = PublishedBatch.from_dict(pb.to_dict())
    assert rebuilt == pb


# ── (b) ROUND-TRIP FIDELITY — recompute the committed root ────────


def test_reloaded_receipts_recompute_the_committed_root(tmp_path):
    """The crux: store a batch, drop the in-memory store, reload from disk in a
    FRESH store instance on the same path, recompute the merkle root from the
    reloaded ordered receipts, and assert it equals the originally committed
    root. This proves retained receipts rebuild the EXACT committed leaves."""
    path = tmp_path / "published.json"
    receipts = [_make_batched(i, with_tee=(i % 2 == 0)) for i in range(5)]
    original_root = _root_of(receipts)

    store = PublishedBatchStore(path)
    batch_id = hashlib.sha256(b"fidelity-batch").digest()
    store.put(batch_id, receipts, original_root, commit_timestamp=1700009999)

    # Fresh instance, same path — forces a disk reload (no shared memory).
    reloaded_store = PublishedBatchStore(path)
    reloaded = reloaded_store.get(batch_id)
    assert reloaded is not None

    recomputed_root = _root_of(reloaded.receipts)
    assert recomputed_root == original_root
    # And the stored root matches too (the store didn't mangle it).
    assert reloaded.merkle_root == original_root


# ── (c) prune by retention ────────────────────────────────────────


def test_prune_drops_expired_keeps_fresh(tmp_path):
    clock = {"t": 1_000_000}
    store = PublishedBatchStore(tmp_path / "p.json", now=lambda: clock["t"])

    old = _make_batched(0)
    fresh = _make_batched(1)
    store.put(hashlib.sha256(b"old").digest(), [old], _root_of([old]),
              commit_timestamp=1_000_000)
    store.put(hashlib.sha256(b"fresh").digest(), [fresh], _root_of([fresh]),
              commit_timestamp=1_000_500)

    # Advance the clock so 'old' is past retention but 'fresh' is not.
    clock["t"] = 1_000_700
    dropped = store.prune(retention_seconds=300)  # cutoff = now-300 = 1_000_400

    assert dropped == 1
    assert store.get(hashlib.sha256(b"old").digest()) is None
    assert store.get(hashlib.sha256(b"fresh").digest()) is not None

    # Survives a reload (prune persisted).
    reloaded = PublishedBatchStore(tmp_path / "p.json", now=lambda: clock["t"])
    assert reloaded.get(hashlib.sha256(b"old").digest()) is None
    assert reloaded.get(hashlib.sha256(b"fresh").digest()) is not None


# ── (d) max_batches bound evicts oldest ───────────────────────────


def test_max_batches_evicts_oldest(tmp_path):
    store = PublishedBatchStore(tmp_path / "b.json", max_batches=2)
    ids = []
    for i in range(3):
        br = _make_batched(i)
        bid = hashlib.sha256(f"batch-{i}".encode()).digest()
        ids.append(bid)
        store.put(bid, [br], _root_of([br]), commit_timestamp=1700000000 + i)

    # Oldest (i=0) evicted; the two newest retained.
    assert store.get(ids[0]) is None
    assert store.get(ids[1]) is not None
    assert store.get(ids[2]) is not None
    assert len(store.all_batches()) == 2


# ── (e) MONEY-PATH-SAFE ───────────────────────────────────────────


def test_client_with_no_store_commits_exactly_as_before():
    """Default published_batch_store=None => commit behaves byte-for-byte as the
    pre-1135 client: same CommittedBatch, accumulator drained, no filesystem."""
    cfg = AccumulatorConfig(count_threshold=3, time_threshold_seconds=3600,
                            value_threshold_ftns=10**30)
    acc = ReceiptAccumulator(cfg)
    contract = _make_mock_contract()
    client = BatchSettlementClient(acc, contract, PROVIDER_ADDR)

    for i in range(3):
        _run(client.accumulate(_make_batched(i)))
    committed = _run(client.commit_ready_batches())

    assert len(committed) == 1
    assert committed[0].receipt_count == 3
    assert acc.total_receipt_count() == 0


def test_client_with_store_retains_committed_receipts(tmp_path):
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
    assert retained.merkle_root == rec.merkle_root
    assert retained.commit_timestamp == rec.commit_timestamp
    assert len(retained.receipts) == 3
    # Order preserved and receipts byte-exact (signature + tee).
    for orig, loaded in zip(originals, retained.receipts):
        assert loaded == orig
    # The retained receipts recompute the committed root.
    assert _root_of(retained.receipts) == rec.merkle_root


def test_store_put_failure_does_not_break_commit(tmp_path):
    """Retention is best-effort: a store.put() that raises must NOT break the
    money path. The CommittedBatch is still returned + tracked + accumulator
    drained, exactly as if no store were present."""
    cfg = AccumulatorConfig(count_threshold=2, time_threshold_seconds=3600,
                            value_threshold_ftns=10**30)
    acc = ReceiptAccumulator(cfg)
    contract = _make_mock_contract()

    class ExplodingStore:
        def put(self, *args, **kwargs):
            raise RuntimeError("disk on fire")

    client = BatchSettlementClient(
        acc, contract, PROVIDER_ADDR, published_batch_store=ExplodingStore(),
    )
    for i in range(2):
        _run(client.accumulate(_make_batched(i)))

    committed = _run(client.commit_ready_batches())

    # The commit succeeded despite the store blowing up.
    assert len(committed) == 1
    rec = committed[0]
    assert rec.receipt_count == 2
    assert acc.total_receipt_count() == 0
    # And the batch is tracked (money-path state intact).
    assert rec.batch_id in client._tracked


def test_malformed_disk_entry_is_skipped_not_raised(tmp_path):
    """A corrupt/oversized entry on disk is skipped + logged, never raised — a
    bad retained entry can't take down the producer."""
    import json
    path = tmp_path / "pub.json"
    good = _make_batched(0)
    store = PublishedBatchStore(path)
    gid = hashlib.sha256(b"good").digest()
    store.put(gid, [good], _root_of([good]), commit_timestamp=1700000000)

    # Inject a malformed entry directly into the on-disk JSON.
    raw = json.loads(path.read_text())
    raw["batches"]["zz-not-hex"] = {"garbage": True}
    path.write_text(json.dumps(raw))

    reloaded = PublishedBatchStore(path)  # must not raise
    assert reloaded.get(gid) is not None
    # The malformed key is simply absent.
    assert all(pb.batch_id == gid for pb in reloaded.all_batches())
