"""Sprint 1164 — observer foreign-batch receipt availability for challenge preparation.

THE GAP this closes: sp1163 made the fraud-defense system operable end-to-end, but its
receipt source was the producer's/requester's OWN PublishedBatchStore. The PRIMARY
fraud-defense actor is an OBSERVER auditing OTHER providers' batches — and those verified
foreign-batch receipts lived only in the in-memory VerifiedBatchCache during a scan, never
persisted, so an out-of-daemon operator could DETECT foreign fraud but not PREPARE a
challenge for it afterward.

sp1164: the audit loop, after a batch PASSES the on-chain-root cross-check (the trust gate),
best-effort PERSISTS its ordered receipts into a (separate) PublishedBatchStore using the
TRUSTED on-chain root. The prepare CLI composes its OWN-batch lookup with this verified-batch
lookup (composite_receipt_lookup), so observer findings on third-party batches now prepare.

Trust property preserved: only a CROSS-CHECK-PASSED set is persisted (a forged set fails the
check -> ok=False -> not persisted), and the persisted merkle_root is the TRUSTED on-chain
root the receipts provably recompute to.
"""
from __future__ import annotations

import pytest

from prsm.compute.shard_receipt import (
    ShardExecutionReceipt,
    build_receipt_signing_payload,
)
from prsm.node.identity import generate_node_identity
from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.challenge_preparer import (
    composite_receipt_lookup,
    prepare_challenge_from_finding,
    published_batch_receipt_lookup,
)
from prsm.settlement.invalid_signature_submitter import DryRunResult
from prsm.settlement.merkle import (
    batched_receipt_to_leaf,
    build_tree_and_proofs,
    hash_leaf,
)
from prsm.settlement.published_batch_store import PublishedBatch, PublishedBatchStore
from prsm.settlement.receipt_set_blob import serialize_receipt_set
from prsm.settlement.settlement_audit_engine import (
    ObservedBatch,
    OwnStakeSelector,
    SettlementAuditEngine,
    VerifiedBatchCache,
)
from prsm.settlement.settlement_audit_loop import SettlementAuditLoop
from prsm.settlement.settlement_audit_report import AuditFindingRecord

ZERO_GROUP = b"\x00" * 32
ME = "0x" + "a" * 40


# ── builders (mirror the sp1139 loop test) ──────────────────────────────────────────


def _receipt(identity, *, job="job-1", shard=0, valid=True, out=None,
             provider="0x" + "2" * 40):
    out = out or ("ab" * 32)
    if valid:
        sig = identity.sign(build_receipt_signing_payload(
            job_id=job, shard_index=shard, output_hash=out, executed_at_unix=1000))
    else:
        sig = identity.sign(build_receipt_signing_payload(
            job_id=job, shard_index=shard, output_hash=("cd" * 32), executed_at_unix=1000))
    rec = ShardExecutionReceipt(
        job_id=job, shard_index=shard, provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64, output_hash=out,
        executed_at_unix=1000, signature=sig)
    return BatchedReceipt(
        receipt=rec, requester_address=ME, provider_address=provider,
        value_ftns=10 ** 18, local_escrow_id="e1", consensus_group_id=ZERO_GROUP)


def _root_for(receipts):
    return build_tree_and_proofs(
        [hash_leaf(batched_receipt_to_leaf(br)) for br in receipts]).root


class _FakeEnumerator:
    def __init__(self, observed):
        self._observed = list(observed)

    async def enumerate_committed_batches(self, from_block, to_block, *, provider=None):
        return list(self._observed)


class _FakeAd:
    def __init__(self, cid):
        self.cid = cid


class _FakeAdIndex:
    def __init__(self, mapping):
        self._ads = {bytes(b).hex(): _FakeAd(c) for b, c in mapping.items()}

    def get(self, batch_id):
        return self._ads.get(bytes(batch_id).hex())


class _FakeFetcher:
    def __init__(self, blobs):
        self._blobs = dict(blobs)

    async def fetch(self, cid):
        return self._blobs[cid]


class _FakeDryRunClient:
    def __init__(self):
        from unittest.mock import AsyncMock, MagicMock
        self.submit = MagicMock(name="submit")
        self.broadcast = AsyncMock(name="broadcast")

    def dry_run(self, challenge):
        return DryRunResult(would_succeed=True)


def _build_loop(observed, blobs, *, verified_batch_store=None):
    cache = VerifiedBatchCache()
    engine = SettlementAuditEngine(cache, dry_run_client=_FakeDryRunClient())
    return SettlementAuditLoop(
        enumerator=_FakeEnumerator(observed),
        selector=OwnStakeSelector(my_address=ME),
        ad_index=_FakeAdIndex({b.batch_id: b.cid for b in observed}),
        fetcher=_FakeFetcher(blobs),
        cache=cache,
        engine=engine,
        verified_batch_store=verified_batch_store,
    )


# ── composite_receipt_lookup ─────────────────────────────────────────────────────────


def test_composite_receipt_lookup_first_hit_wins_and_falls_through():
    a = generate_node_identity("a")
    own = [_receipt(a, shard=0)]
    verified = [_receipt(a, shard=1), _receipt(a, shard=2)]
    own_lk = lambda bid: own if bid == "own" else None       # noqa: E731
    ver_lk = lambda bid: verified if bid == "ver" else None   # noqa: E731
    composite = composite_receipt_lookup(own_lk, ver_lk)
    assert composite("own") == own            # first source hit
    assert composite("ver") == verified       # falls through to the second source
    assert composite("missing") is None       # neither -> None


def test_composite_receipt_lookup_skips_none_lookups_and_empty_results():
    a = generate_node_identity("a")
    hit = [_receipt(a)]
    empty_lk = lambda bid: []                 # noqa: E731 — empty result falls through
    hit_lk = lambda bid: hit                  # noqa: E731
    composite = composite_receipt_lookup(None, empty_lk, hit_lk)
    assert composite("x") == hit
    assert composite_receipt_lookup() ("x") is None  # no lookups -> None


# ── the loop persists a CROSS-CHECK-PASSED foreign batch ─────────────────────────────


@pytest.mark.asyncio
async def test_loop_persists_verified_foreign_batch_with_trusted_root(tmp_path):
    a = generate_node_identity("a")
    receipts = [_receipt(a, shard=0), _receipt(a, shard=1)]
    bid = b"\x71" * 32
    root = _root_for(receipts)
    pb = PublishedBatch(batch_id=bid, merkle_root=root, commit_timestamp=1000,
                        receipts=receipts)
    blob = serialize_receipt_set(pb)
    observed = ObservedBatch(batch_id=bid, provider_address="0x" + "2" * 40,
                             requester_address=ME, merkle_root=root,
                             consensus_group_id=ZERO_GROUP, cid="cid1")

    store = PublishedBatchStore(tmp_path / "verified.json")
    loop = _build_loop([observed], {"cid1": blob}, verified_batch_store=store)
    result = await loop.run_once(0, 100)
    assert result.verified == 1

    # the verified foreign batch is now persisted with the TRUSTED on-chain root + receipts
    got = store.get(bid)
    assert got is not None
    assert got.merkle_root == root
    assert len(got.receipts) == 2

    # and it now feeds the prepare bridge (an observer can prepare a challenge for it)
    lookup = published_batch_receipt_lookup(store)
    assert lookup(bid.hex()) is not None
    assert len(lookup(bid.hex())) == 2


@pytest.mark.asyncio
async def test_loop_does_NOT_persist_a_batch_that_fails_cross_check(tmp_path):
    a = generate_node_identity("a")
    receipts = [_receipt(a, shard=0)]
    bid = b"\x72" * 32
    pb = PublishedBatch(batch_id=bid, merkle_root=_root_for(receipts),
                        commit_timestamp=1000, receipts=receipts)
    blob = serialize_receipt_set(pb)
    # observed with a WRONG on-chain root -> the cross-check fails -> ok=False -> NOT cached/persisted
    observed = ObservedBatch(batch_id=bid, provider_address="0x" + "2" * 40,
                             requester_address=ME, merkle_root=b"\xff" * 32,
                             consensus_group_id=ZERO_GROUP, cid="cid1")
    store = PublishedBatchStore(tmp_path / "verified.json")
    loop = _build_loop([observed], {"cid1": blob}, verified_batch_store=store)
    result = await loop.run_once(0, 100)
    assert result.verified == 0
    assert result.rejected == 1
    assert store.get(bid) is None  # a forged/mismatched set is NEVER persisted


@pytest.mark.asyncio
async def test_loop_without_verified_store_is_unchanged(tmp_path):
    # default (no store) -> no crash, run completes, nothing persisted (backward compat).
    a = generate_node_identity("a")
    receipts = [_receipt(a, shard=0)]
    bid = b"\x73" * 32
    root = _root_for(receipts)
    pb = PublishedBatch(batch_id=bid, merkle_root=root, commit_timestamp=1000,
                        receipts=receipts)
    blob = serialize_receipt_set(pb)
    observed = ObservedBatch(batch_id=bid, provider_address="0x" + "2" * 40,
                             requester_address=ME, merkle_root=root,
                             consensus_group_id=ZERO_GROUP, cid="cid1")
    loop = _build_loop([observed], {"cid1": blob})  # verified_batch_store=None
    result = await loop.run_once(0, 100)
    assert result.verified == 1  # still verifies + scans; just doesn't persist


# ── the factory threads the store ────────────────────────────────────────────────────


def test_factory_threads_verified_batch_store_to_the_loop():
    from prsm.settlement.settlement_audit_wiring import (
        BlockCursor,
        build_settlement_audit_components,
    )

    sentinel = object()

    class _Gossip:
        def subscribe(self, *a, **k):
            return None

    bundle = build_settlement_audit_components(
        enabled=True, gossip=_Gossip(), fetcher=_FakeFetcher({}),
        contract_client=_FakeEnumerator([]), my_address=ME, group_ids=set(),
        dry_run_client=_FakeDryRunClient(), cursor=BlockCursor(path=":memory:"),
        verified_batch_store=sentinel,
    )
    assert bundle is not None
    assert bundle.audit_loop._verified_batch_store is sentinel


# ── end-to-end: observer prepares a foreign-batch challenge from the verified store ──


@pytest.mark.asyncio
async def test_observer_prepares_invalid_sig_challenge_from_verified_store(tmp_path):
    # a foreign provider committed a batch with a fraudulent (non-verifying) receipt; the
    # observer audits it (root cross-check passes -> persisted), then prepares the challenge.
    a, b = generate_node_identity("a"), generate_node_identity("b")
    receipts = [_receipt(a, shard=0, valid=True), _receipt(b, shard=1, valid=False)]
    bid = b"\x74" * 32
    root = _root_for(receipts)
    pb = PublishedBatch(batch_id=bid, merkle_root=root, commit_timestamp=1000,
                        receipts=receipts)
    observed = ObservedBatch(batch_id=bid, provider_address="0x" + "2" * 40,
                             requester_address=ME, merkle_root=root,
                             consensus_group_id=ZERO_GROUP, cid="cid1")
    verified_store = PublishedBatchStore(tmp_path / "verified.json")
    loop = _build_loop([observed], {"cid1": serialize_receipt_set(pb)},
                       verified_batch_store=verified_store)
    await loop.run_once(0, 100)

    # the operator's own store is EMPTY; the verified store has the foreign batch.
    own_store = PublishedBatchStore(tmp_path / "own.json")
    lookup = composite_receipt_lookup(
        published_batch_receipt_lookup(own_store),
        published_batch_receipt_lookup(verified_store))

    rec = AuditFindingRecord(reason="invalid_signature", batch_id_hex=bid.hex(),
                             target_index=1, on_chain_reason=1,
                             detail={"receipt_index": 1})
    prepared = prepare_challenge_from_finding(record=rec, receipt_lookup=lookup)
    assert prepared.on_chain_reason == 1
    assert prepared.challenge.reason_code == 1  # prepared from the OBSERVED foreign batch
