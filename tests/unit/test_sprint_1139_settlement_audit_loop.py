"""Sprint 1139 — settlement-receipt data plane BRICK E.1: the OBSERVE LOOP logic.

Bricks A-D are committed (see docs/2026-06-16-settlement-receipt-data-plane-design.md):
  - A retains each committed batch's ordered receipt set as a ``PublishedBatch``;
  - B canonicalizes it to a CID-addressable blob + the on-chain-root cross-check gate;
  - C indexes the (untrusted) availability pointers (``SettlementBatchAdIndex.get``);
  - D is the PURE audit engine (selectors + ``VerifiedBatchCache`` trust gate +
    ``SettlementAuditEngine`` dry-run-gated detectors).

Brick E.1 is the orchestration LOGIC that ties them into ONE ``run_once``:
  enumerate (trusted on-chain anchor) -> select -> for each: look up the untrusted ad
  pointer -> fetch the blob by CID -> deserialize -> cross-check-and-cache (cache.ingest)
  -> drive the dry-run-gated engine scans. It is PURE/injectable (enumerator, selector,
  ad_index, fetcher, cache, engine all injected), NEVER broadcasts/signs/slashes, and is
  FAIL-CLOSED per batch (a fetch/deserialize/ingest error skips THAT batch + counts it,
  never aborts the run). The LIVE daemon wiring/scheduling/announce-on-commit is Brick
  E.2 (operator/testbed-gated) and is OUT OF SCOPE here.

These tests inject EVERYTHING — no network, no chain, no disk:
  - a ``FakeEnumerator`` returning a fixed list of ``ObservedBatch`` (the trusted worklist);
  - a ``FakeAdIndex`` mapping batch_id -> ad (carrying a ``.cid``);
  - a ``FakeFetcher`` mapping cid -> blob bytes (or raising / returning garbage);
  - the REAL ``OwnStakeSelector`` + ``VerifiedBatchCache`` + ``SettlementAuditEngine``.

And the CHAIN ENUMERATOR (``Web3SettlementContractClient.enumerate_committed_batches``)
is exercised against a MOCK contract (fake ``BatchCommitted`` logs + fake ``batches()``
struct returns) — pure-read, chunked scan, no tx.

Coverage:
  (a) HAPPY PATH — two selected batches both verify+cache; a planted cross-provider
      double-spend surfaces in the result; counts correct;
  (b) TRUST PRESERVED — a root-mismatching fetched set is rejected (counted), NOT cached,
      NOT scanned; a forged double-spend planted in it does NOT surface;
  (c) SELECTOR APPLIED — a dropped batch is never fetched;
  (d) NO CID — a selected batch with no ad-index entry is skipped_no_cid, no fetch, no crash;
  (e) FAIL-CLOSED — a fetcher that raises / returns malformed bytes for one batch counts
      that batch rejected, the run does NOT raise, other batches still process;
  (f) NO-BROADCAST — the engine's dry_run_client submit/broadcast are NEVER called;
  (g) BUDGET — ``max_fetches_per_run`` caps the number of fetches;
  (h) ENUMERATOR — the chunked BatchCommitted scan + ``batches()`` reads build correct
      ``ObservedBatch`` (batch_id/provider/requester/merkle_root/consensus_group_id).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.compute.shard_receipt import (
    ShardExecutionReceipt,
    build_receipt_signing_payload,
)
from prsm.node.identity import generate_node_identity
from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.invalid_signature_submitter import DryRunResult
from prsm.settlement.merkle import (
    batched_receipt_to_leaf,
    build_tree_and_proofs,
    hash_leaf,
)
from prsm.settlement.published_batch_store import PublishedBatch
from prsm.settlement.receipt_set_blob import serialize_receipt_set
from prsm.settlement.settlement_audit_engine import (
    ObservedBatch,
    OwnStakeSelector,
    SettlementAuditEngine,
    VerifiedBatchCache,
)
from prsm.settlement.settlement_audit_loop import (
    AuditRunResult,
    SettlementAuditLoop,
)

ZERO_GROUP = b"\x00" * 32
ME = "0x" + "a" * 40


# ── builders ─────────────────────────────────────────────────────────


def _receipt(identity, *, job="job-1", shard=0, valid=True, out=None,
             requester=ME, provider="0x" + "2" * 40,
             group=ZERO_GROUP, value=10 ** 18):
    out = out or ("ab" * 32)
    payload = build_receipt_signing_payload(
        job_id=job, shard_index=shard, output_hash=out, executed_at_unix=1000)
    sig = identity.sign(payload)
    rec = ShardExecutionReceipt(
        job_id=job, shard_index=shard, provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64, output_hash=out,
        executed_at_unix=1000, signature=sig)
    return BatchedReceipt(
        receipt=rec, requester_address=requester, provider_address=provider,
        value_ftns=value, local_escrow_id="e1", consensus_group_id=group)


def _root_for(receipts):
    leaf_hashes = [hash_leaf(batched_receipt_to_leaf(br)) for br in receipts]
    return build_tree_and_proofs(leaf_hashes).root


def _published(batch_id, receipts):
    return PublishedBatch(
        batch_id=batch_id, merkle_root=_root_for(receipts),
        commit_timestamp=1000, receipts=list(receipts))


def _observed(batch_id, receipts, *, provider="0x" + "2" * 40,
              requester=ME, group=ZERO_GROUP, root=None, cid=None):
    return ObservedBatch(
        batch_id=batch_id, provider_address=provider,
        requester_address=requester,
        merkle_root=root if root is not None else _root_for(receipts),
        consensus_group_id=group, cid=cid)


# ── fakes (injectable, no network) ───────────────────────────────────


class FakeEnumerator:
    """Returns a fixed worklist of ObservedBatch; records the (from, to) it was asked."""

    def __init__(self, observed):
        self._observed = list(observed)
        self.calls = []

    async def enumerate_committed_batches(self, from_block, to_block, *, provider=None):
        self.calls.append((from_block, to_block, provider))
        return list(self._observed)


class _FakeAd:
    def __init__(self, cid):
        self.cid = cid


class FakeAdIndex:
    """Maps batch_id_hex -> ad (carrying .cid). get() returns None for unknown ids."""

    def __init__(self, mapping):
        # mapping: batch_id(bytes) -> cid(str) | None (None => an ad with no cid)
        self._ads = {}
        for bid, cid in mapping.items():
            self._ads[bytes(bid).hex()] = _FakeAd(cid) if cid is not None else _FakeAd(None)

    def get(self, batch_id):
        return self._ads.get(bytes(batch_id).hex())


class FakeFetcher:
    """Async fetch(cid)->bytes. Records every cid fetched. A cid mapped to an Exception
    instance raises it; a cid mapped to bytes returns those bytes; unknown -> b"" (garbage)."""

    def __init__(self, blobs):
        self._blobs = dict(blobs)
        self.fetched = []

    async def fetch(self, cid):
        self.fetched.append(cid)
        val = self._blobs.get(cid)
        if isinstance(val, Exception):
            raise val
        if val is None:
            return b"not-a-valid-blob"
        return val


class _FakeDryRunClient:
    """Dry-run-only client injected into the engine. submit/broadcast MUST never fire."""

    def __init__(self, would_succeed=True):
        self._would_succeed = would_succeed
        self.dry_run_calls = []
        self.submit = MagicMock(name="submit")
        self.broadcast = AsyncMock(name="broadcast")

    def dry_run(self, challenge):
        self.dry_run_calls.append(challenge)
        return DryRunResult(would_succeed=self._would_succeed)


def _loop(*, enumerator, ad_index, fetcher, cache, engine, selector=None,
          max_fetches_per_run=100):
    return SettlementAuditLoop(
        enumerator=enumerator,
        selector=selector or OwnStakeSelector(my_address=ME),
        ad_index=ad_index,
        fetcher=fetcher,
        cache=cache,
        engine=engine,
        max_fetches_per_run=max_fetches_per_run,
    )


# ── (a) HAPPY PATH ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_happy_path_both_verify_and_double_spend_surfaces():
    a = generate_node_identity("a")
    shared = _receipt(a, shard=0, job="shared-job")
    b1 = [shared, _receipt(a, shard=1, job="filler-1")]
    b2 = [shared, _receipt(a, shard=1, job="filler-2")]
    blob1 = serialize_receipt_set(_published(b"\x01" * 32, b1))
    blob2 = serialize_receipt_set(_published(b"\x02" * 32, b2))

    obs = [
        _observed(b"\x01" * 32, b1, provider="0x" + "1" * 40),
        _observed(b"\x02" * 32, b2, provider="0x" + "3" * 40),
    ]
    fetcher = FakeFetcher({"cid-1": blob1, "cid-2": blob2})
    ad_index = FakeAdIndex({b"\x01" * 32: "cid-1", b"\x02" * 32: "cid-2"})
    cache = VerifiedBatchCache(max_batches=8)
    fake = _FakeDryRunClient(would_succeed=True)
    engine = SettlementAuditEngine(cache, dry_run_client=fake)
    loop = _loop(enumerator=FakeEnumerator(obs), ad_index=ad_index,
                 fetcher=fetcher, cache=cache, engine=engine)

    result = await loop.run_once(from_block=0, to_block=10)

    assert isinstance(result, AuditRunResult)
    assert result.enumerated == 2
    assert result.selected == 2
    assert result.fetched == 2
    assert result.verified == 2
    assert result.rejected == 0
    assert result.skipped_no_cid == 0
    # both cached
    assert len(cache.verified_batches()) == 2
    # the planted cross-provider double-spend surfaces
    assert len(result.double_spend_findings) == 1
    f = result.double_spend_findings[0]
    assert f.finding.leaf_hash == hash_leaf(batched_receipt_to_leaf(shared))
    assert {b"\x01" * 32, b"\x02" * 32} == {occ.batch_id for occ in f.finding.occurrences}
    # never broadcast
    fake.submit.assert_not_called()
    fake.broadcast.assert_not_called()


# ── (b) TRUST PRESERVED ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_trust_preserved_root_mismatch_rejected_not_scanned():
    """A fetched blob that recomputes to a DIFFERENT root than observed.merkle_root is
    rejected (counted), NOT cached, NOT scanned — and a forged double-spend planted in it
    does NOT surface."""
    a = generate_node_identity("a")
    # genuine batch \x01 anchored correctly
    genuine = [_receipt(a, shard=0, job="genuine")]
    blob_genuine = serialize_receipt_set(_published(b"\x01" * 32, genuine))

    # batch \x02: a FORGED set sharing batch \x01's leaf (would be a double-spend IF
    # trusted), but observed.merkle_root is the WRONG root so the cross-check fails.
    forged = [_receipt(a, shard=0, job="genuine")]  # same leaf as genuine
    blob_forged = serialize_receipt_set(_published(b"\x02" * 32, forged))
    obs_forged = ObservedBatch(
        batch_id=b"\x02" * 32, provider_address="0x" + "3" * 40,
        requester_address=ME, merkle_root=b"\xde\xad" * 16,  # does NOT match forged
        consensus_group_id=ZERO_GROUP)

    obs = [_observed(b"\x01" * 32, genuine, provider="0x" + "1" * 40), obs_forged]
    fetcher = FakeFetcher({"cid-1": blob_genuine, "cid-2": blob_forged})
    ad_index = FakeAdIndex({b"\x01" * 32: "cid-1", b"\x02" * 32: "cid-2"})
    cache = VerifiedBatchCache(max_batches=8)
    engine = SettlementAuditEngine(cache)
    loop = _loop(enumerator=FakeEnumerator(obs), ad_index=ad_index,
                 fetcher=fetcher, cache=cache, engine=engine)

    result = await loop.run_once(from_block=0, to_block=10)

    assert result.enumerated == 2
    assert result.selected == 2
    assert result.fetched == 2
    assert result.verified == 1            # only the genuine one
    assert result.rejected == 1            # the forged/root-mismatch one
    # only the genuine batch is cached; the forged one cached NOTHING
    assert {cb.batch_id for cb in cache.verified_batches()} == {b"\x01" * 32}
    # the forged double-spend NEVER surfaces (the un-cross-checked set never reached detection)
    assert result.double_spend_findings == []


# ── (c) SELECTOR APPLIED ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_selector_dropped_batch_is_never_fetched():
    a = generate_node_identity("a")
    mine = [_receipt(a, shard=0, job="mine")]
    not_mine = [_receipt(a, shard=0, job="theirs")]
    blob_mine = serialize_receipt_set(_published(b"\x01" * 32, mine))
    blob_not = serialize_receipt_set(_published(b"\x02" * 32, not_mine))

    obs = [
        _observed(b"\x01" * 32, mine, requester=ME),
        _observed(b"\x02" * 32, not_mine, requester="0x" + "b" * 40),  # dropped
    ]
    fetcher = FakeFetcher({"cid-1": blob_mine, "cid-2": blob_not})
    ad_index = FakeAdIndex({b"\x01" * 32: "cid-1", b"\x02" * 32: "cid-2"})
    cache = VerifiedBatchCache(max_batches=8)
    engine = SettlementAuditEngine(cache)
    loop = _loop(enumerator=FakeEnumerator(obs), ad_index=ad_index,
                 fetcher=fetcher, cache=cache, engine=engine)

    result = await loop.run_once(from_block=0, to_block=10)

    assert result.enumerated == 2
    assert result.selected == 1
    assert result.fetched == 1
    assert fetcher.fetched == ["cid-1"]    # cid-2 was NEVER fetched
    assert {cb.batch_id for cb in cache.verified_batches()} == {b"\x01" * 32}


# ── (d) NO CID ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_selected_batch_with_no_ad_is_skipped_no_cid():
    a = generate_node_identity("a")
    has_ad = [_receipt(a, shard=0, job="has-ad")]
    no_ad = [_receipt(a, shard=0, job="no-ad")]
    blob = serialize_receipt_set(_published(b"\x01" * 32, has_ad))

    obs = [
        _observed(b"\x01" * 32, has_ad),
        _observed(b"\x02" * 32, no_ad),    # no ad-index entry
        _observed(b"\x03" * 32, no_ad),    # ad present but .cid is None
    ]
    fetcher = FakeFetcher({"cid-1": blob})
    ad_index = FakeAdIndex({b"\x01" * 32: "cid-1", b"\x03" * 32: None})
    cache = VerifiedBatchCache(max_batches=8)
    engine = SettlementAuditEngine(cache)
    loop = _loop(enumerator=FakeEnumerator(obs), ad_index=ad_index,
                 fetcher=fetcher, cache=cache, engine=engine)

    result = await loop.run_once(from_block=0, to_block=10)

    assert result.enumerated == 3
    assert result.selected == 3
    assert result.skipped_no_cid == 2
    assert result.fetched == 1
    assert fetcher.fetched == ["cid-1"]
    assert result.verified == 1


# ── (e) FAIL-CLOSED ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fail_closed_fetch_raises_and_malformed_bytes_other_batches_still_processed():
    a = generate_node_identity("a")
    good = [_receipt(a, shard=0, job="good")]
    blob_good = serialize_receipt_set(_published(b"\x03" * 32, good))

    obs = [
        _observed(b"\x01" * 32, good),     # fetch raises
        _observed(b"\x02" * 32, good),     # fetch returns garbage -> deserialize ValueError
        _observed(b"\x03" * 32, good),     # genuine, still processed
    ]
    fetcher = FakeFetcher({
        "cid-1": RuntimeError("network down"),
        "cid-2": b"\xff\xfe not json",
        "cid-3": blob_good,
    })
    ad_index = FakeAdIndex({b"\x01" * 32: "cid-1", b"\x02" * 32: "cid-2",
                            b"\x03" * 32: "cid-3"})
    cache = VerifiedBatchCache(max_batches=8)
    engine = SettlementAuditEngine(cache)
    loop = _loop(enumerator=FakeEnumerator(obs), ad_index=ad_index,
                 fetcher=fetcher, cache=cache, engine=engine)

    # the run must NOT raise
    result = await loop.run_once(from_block=0, to_block=10)

    assert result.enumerated == 3
    assert result.selected == 3
    assert result.fetched == 3
    assert result.rejected == 2            # the raising fetch + the malformed bytes
    assert result.verified == 1            # only batch \x03
    assert {cb.batch_id for cb in cache.verified_batches()} == {b"\x03" * 32}


# ── (f) NO-BROADCAST ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_once_never_broadcasts_or_submits():
    a, bident = generate_node_identity("a"), generate_node_identity("b")
    shared = _receipt(a, shard=0, job="shared")
    b1 = [shared, _receipt(bident, shard=1, job="x")]
    b2 = [shared, _receipt(a, shard=1, job="y")]
    blob1 = serialize_receipt_set(_published(b"\x01" * 32, b1))
    blob2 = serialize_receipt_set(_published(b"\x02" * 32, b2))

    obs = [
        _observed(b"\x01" * 32, b1, provider="0x" + "a" * 40),
        _observed(b"\x02" * 32, b2, provider="0x" + "c" * 40),
    ]
    fetcher = FakeFetcher({"cid-1": blob1, "cid-2": blob2})
    ad_index = FakeAdIndex({b"\x01" * 32: "cid-1", b"\x02" * 32: "cid-2"})
    cache = VerifiedBatchCache(max_batches=8)
    fake = _FakeDryRunClient(would_succeed=True)
    engine = SettlementAuditEngine(cache, dry_run_client=fake)
    loop = _loop(enumerator=FakeEnumerator(obs), ad_index=ad_index,
                 fetcher=fetcher, cache=cache, engine=engine)

    await loop.run_once(from_block=0, to_block=10)

    # dry-run was exercised; submit/broadcast NEVER
    assert len(fake.dry_run_calls) >= 1
    fake.submit.assert_not_called()
    fake.broadcast.assert_not_called()


# ── (g) BUDGET ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_max_fetches_per_run_caps_fetches():
    a = generate_node_identity("a")
    obs = []
    blobs = {}
    ad_map = {}
    for i in range(5):
        bid = bytes([i + 1]) * 32
        receipts = [_receipt(a, shard=0, job=f"job-{i}")]
        obs.append(_observed(bid, receipts))
        cid = f"cid-{i}"
        blobs[cid] = serialize_receipt_set(_published(bid, receipts))
        ad_map[bid] = cid
    fetcher = FakeFetcher(blobs)
    ad_index = FakeAdIndex(ad_map)
    cache = VerifiedBatchCache(max_batches=16)
    engine = SettlementAuditEngine(cache)
    loop = _loop(enumerator=FakeEnumerator(obs), ad_index=ad_index,
                 fetcher=fetcher, cache=cache, engine=engine,
                 max_fetches_per_run=2)

    result = await loop.run_once(from_block=0, to_block=10)

    assert result.enumerated == 5
    assert result.selected == 5
    assert result.fetched == 2             # capped
    assert len(fetcher.fetched) == 2
    assert result.verified == 2


# ── (h) ENUMERATOR ───────────────────────────────────────────────────


def _struct(provider, requester, merkle_root, consensus_group_id):
    """A fake batches() struct return in the BATCH_SETTLEMENT_REGISTRY_ABI field order."""
    return [
        provider,                  # 0 provider
        requester,                 # 1 requester
        bytes(merkle_root),        # 2 merkleRoot
        1,                         # 3 receiptCount
        1000,                      # 4 totalValueFTNS
        0,                         # 5 invalidatedValueFTNS
        1000,                      # 6 commitTimestamp
        1,                         # 7 status
        0,                         # 8 tier_slash_rate_bps
        bytes(consensus_group_id), # 9 consensus_group_id
        0, 0, 0,                   # 10-12
        "0x" + "0" * 40,           # 13 escrowPoolAtCommit
        "0x" + "0" * 40,           # 14 stakeBondAtCommit
        "0x" + "0" * 40,           # 15 signatureVerifierAtCommit
        "",                        # 16 metadataURI
    ]


@pytest.mark.asyncio
async def test_enumerate_committed_batches_builds_observed_from_logs_and_struct():
    from prsm.economy.web3.batch_settlement_contract_client import (
        Web3SettlementContractClient,
    )

    provider = "0x" + "1" * 40
    requester = "0x" + "2" * 40
    root1 = b"\x11" * 32
    root2 = b"\x22" * 32
    group1 = b"\xaa" * 32
    group2 = b"\x00" * 32
    bid1 = b"\x01" * 32
    bid2 = b"\x02" * 32

    structs = {
        bid1: _struct(provider, requester, root1, group1),
        bid2: _struct(provider, requester, root2, group2),
    }

    # Build a client WITHOUT touching __init__ (no real RPC), then wire a mock contract.
    client = Web3SettlementContractClient.__new__(Web3SettlementContractClient)

    # mock web3.eth.block_number
    client.web3 = MagicMock()
    client.web3.eth.block_number = 20_000  # head; from=0..20000 spans 3 chunked windows

    contract = MagicMock()

    # contract.functions.batches(batch_id).call() -> the struct
    def batches(batch_id):
        call_obj = MagicMock()
        call_obj.call.return_value = structs[bytes(batch_id)]
        return call_obj
    contract.functions.batches.side_effect = batches

    # contract.events.BatchCommitted().create_filter(...).get_all_entries() -> logs in window
    fake_logs_window_1 = [
        {"args": {"batchId": bid1, "provider": provider}},
        {"args": {"batchId": bid2, "provider": provider}},
    ]

    event_obj = MagicMock()

    def create_filter(*, from_block, to_block, argument_filters=None):
        flt = MagicMock()
        # only the FIRST window carries the logs; subsequent windows are empty
        flt.get_all_entries.return_value = fake_logs_window_1 if from_block == 0 else []
        # record the filter args so we can assert provider filtering + chunking
        create_filter.calls.append((from_block, to_block, argument_filters))
        return flt
    create_filter.calls = []
    event_obj.create_filter.side_effect = create_filter
    contract.events.BatchCommitted.return_value = event_obj

    client.contract = contract

    observed = await client.enumerate_committed_batches(0, 20_000, provider=provider)

    by_id = {bytes(o.batch_id): o for o in observed}
    assert set(by_id) == {bid1, bid2}
    o1 = by_id[bid1]
    assert o1.provider_address == provider
    assert o1.requester_address == requester
    assert o1.merkle_root == root1
    assert o1.consensus_group_id == group1
    assert o1.cid is None                  # loop fills cid from the ad index
    o2 = by_id[bid2]
    assert o2.merkle_root == root2
    assert o2.consensus_group_id == group2

    # CHUNKED: 0..20000 in 9000-block windows => windows 0-8999, 9000-17999, 18000-20000
    windows = [(f, t) for (f, t, _) in create_filter.calls]
    assert windows == [(0, 8999), (9000, 17999), (18000, 20000)]
    # provider filter was applied on every window
    assert all(af == {"provider": provider} for (_, _, af) in create_filter.calls)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
