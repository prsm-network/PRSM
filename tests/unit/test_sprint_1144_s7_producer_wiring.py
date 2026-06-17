"""Sprint 1144 — §7 PRODUCER daemon wiring (producer-side analog of E.2).

Closes the two gaps that left the §7 publish path dangling:

  (1) DANGLING CID: ``SettlementBatchAnnouncer.announce`` gossiped an ad carrying a
      ``receipt_set_cid(blob)`` but NEVER published the blob anywhere — so an observer
      fetching that cid via ``ContentProvider.request_content`` got nothing. The producer
      must PUBLISH the blob so the announced cid resolves. The robust way to guarantee
      ad-cid == published-content-cid is to USE THE PUBLISHER-RETURNED INFOHASH as the ad
      cid (the v1 infohash depends on the content NAME, so a separately-computed
      ``receipt_set_cid`` can mismatch the publisher's staged name).

  (2) §7 NOT PUBLISHED: the published blob was the PLAIN ``serialize_receipt_set``, not the
      ENRICHED blob (``serialize_enriched_receipt_set``) carrying the retained §7
      ``InferenceReceipts``. When an ``InferenceReceiptStore`` is wired, the announcer must
      enrich (``enrich_from_store``) so the §7 receipts ride in the SAME published blob.

DEFAULT-OFF is sacred: with no ``content_publisher`` and no ``inference_receipt_store``
(the defaults), ``SettlementBatchAnnouncer`` behaves EXACTLY as before — plain blob, legacy
``receipt_set_cid``, no publish — so node startup + the adapt/commit money path are
byte-for-byte unchanged. The whole announce/publish is BEST-EFFORT (wrapped, never raises).

These tests are OFFLINE (no network). FAKES:
  - ``_FakeContentPublisher`` records every published blob and returns a deterministic
    infohash for it (so we can assert ad-cid == publisher-returned-infohash for that exact
    blob, and an observer can re-fetch the published bytes by that cid);
  - ``_FakeGossip`` records the ads;
  - a REAL ``InferenceReceiptStore`` (tmp file) + a REAL ``PublishedBatch``.

  (a) CID RESOLVES: announce() with a publisher PUBLISHES the blob AND the ad cid EQUALS
      the publisher-returned infohash for that exact blob; an observer deserializes the
      published blob and recovers the PublishedBatch (round-trip end to end).
  (b) ENRICHED: with an InferenceReceiptStore holding §7 receipts for the batch leaves, the
      published blob is the ENRICHED blob (deserialize_enriched recovers the §7 map); with
      no §7 store the published blob is plain (deserialize_receipt_set works, no §7 key).
  (c) CID MATCH UNDER ENRICHMENT: the ad cid for an enriched blob == the publisher infohash
      of that enriched blob (still resolves).
  (d) BEST-EFFORT: a publisher whose publish() raises, and an enrich/serialize error, are
      caught — announce returns False, never raises.
  (e) factory: build_settlement_audit_components(enabled=True, content_publisher=fake,
      inference_receipt_store=store, ...) builds an announcer that, on announce, enriches +
      publishes; enabled=False => None bundle (default-off).
  (f) DEFAULT-OFF: build_settlement_audit_components with no content_publisher/store still
      works (legacy announcer, plain blob, legacy receipt_set_cid).
"""
from __future__ import annotations

import hashlib
from base64 import b64encode
from decimal import Decimal

import pytest

from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.node.gossip import GOSSIP_SETTLEMENT_BATCH_AVAILABLE
from prsm.node.identity import generate_node_identity
from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.inference_receipt_store import InferenceReceiptStore
from prsm.settlement.merkle import (
    batched_receipt_to_leaf,
    build_tree_and_proofs,
    hash_leaf,
)
from prsm.settlement.published_batch_store import PublishedBatch
from prsm.settlement.receipt_set_blob import (
    INFERENCE_RECEIPTS_KEY,
    deserialize_enriched_receipt_set,
    deserialize_receipt_set,
    receipt_set_cid,
    serialize_enriched_receipt_set,
    serialize_receipt_set,
)
from prsm.settlement.settlement_gossip import (
    SettlementBatchAdvertisement,
    SettlementBatchAnnouncer,
)


ONE_FTNS = 10**18
PROVIDER_ADDR = "0x" + "b" * 40
REQUESTER_A = "0x" + "a" * 40


# ── fixtures (mirror sp1137/sp1142) ───────────────────────────────────────


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


def _make_signed_ir(identity, **over):
    """A genuine, SIGNED §7 InferenceReceipt (mirrors sp1142)."""
    from prsm.compute.inference.models import ContentTier, InferenceReceipt
    from prsm.compute.inference.receipt import sign_receipt
    from prsm.compute.tee.models import PrivacyLevel, TEEType
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
    return sign_receipt(InferenceReceipt(**defaults), identity)


class _FakeStore:
    """Minimal stand-in for PublishedBatchStore.get(batch_id)."""

    def __init__(self):
        self._d = {}

    def put(self, pb: PublishedBatch) -> None:
        self._d[bytes(pb.batch_id).hex()] = pb

    def get(self, batch_id: bytes):
        return self._d.get(bytes(batch_id).hex())


class _FakeGossip:
    """Records publish() calls + exposes a producer identity for announcer_id."""

    def __init__(self, identity=None):
        self.published = []
        self._subscribers = {}
        self.transport = type("T", (), {"identity": identity})()

    async def publish(self, subtype, data, ttl=None):
        self.published.append((subtype, data))
        return 0

    def subscribe(self, subtype, handler):
        self._subscribers.setdefault(subtype, []).append(handler)


class _FakeContentPublisher:
    """Records every published blob and returns a DETERMINISTIC infohash for it.

    The infohash is the sha256 hex of the blob bytes prefixed so it is unambiguously the
    publisher's identity for THIS blob (and distinct from the legacy receipt_set_cid,
    which uses the v1 BitTorrent infohash under RECEIPT_SET_CONTENT_NAME). This lets a
    test assert ad-cid == publisher-returned-infohash for the exact published blob, and
    re-fetch the bytes by that cid (the staged content)."""

    def __init__(self):
        self.published = []          # list[(blob, kwargs)]
        self._by_cid = {}            # cid -> blob (the "staged content")

    def _infohash(self, blob: bytes) -> str:
        return "fakeih" + hashlib.sha256(bytes(blob)).hexdigest()

    async def publish(self, data: bytes, **kwargs):
        cid = self._infohash(data)
        self.published.append((bytes(data), kwargs))
        self._by_cid[cid] = bytes(data)
        return type("Pub", (), {"torrent_infohash": cid})()

    def fetch(self, cid: str):
        """An observer fetching the staged content by cid (the resolve step)."""
        return self._by_cid.get(cid)


class _BoomPublisher:
    """A publisher whose publish() always raises (best-effort coverage)."""

    async def publish(self, data: bytes, **kwargs):
        raise RuntimeError("staging disk full")


# ── (a) CID RESOLVES: publish + ad-cid == publisher infohash + round-trip ──


@pytest.mark.asyncio
async def test_announce_publishes_blob_and_ad_cid_is_publisher_infohash():
    receipts = [_make_batched(i, with_tee=(i % 2 == 0)) for i in range(4)]
    pb = _make_pb(receipts, batch_label=b"resolve-batch")
    store = _FakeStore()
    store.put(pb)
    producer = generate_node_identity("producer")
    gossip = _FakeGossip(identity=producer)
    publisher = _FakeContentPublisher()

    announcer = SettlementBatchAnnouncer(store, gossip, content_publisher=publisher)
    assert await announcer.announce(pb.batch_id) is True

    # The blob was PUBLISHED (no longer dangling).
    assert len(publisher.published) == 1
    published_blob, _kw = publisher.published[0]
    # No §7 store wired -> the published blob is the PLAIN serialize_receipt_set.
    assert published_blob == serialize_receipt_set(pb)

    # The ad cid EQUALS the publisher-returned infohash for that exact blob.
    assert len(gossip.published) == 1
    subtype, data = gossip.published[0]
    assert subtype == GOSSIP_SETTLEMENT_BATCH_AVAILABLE
    ad = SettlementBatchAdvertisement.from_gossip_data(data)
    assert ad.cid == publisher._infohash(published_blob)

    # AN OBSERVER FETCH RESOLVES: fetch by the ad cid yields the published bytes, which
    # deserialize back to the PublishedBatch (end-to-end round trip).
    fetched = publisher.fetch(ad.cid)
    assert fetched == published_blob
    recovered = deserialize_receipt_set(fetched)
    assert recovered == pb


# ── (b) ENRICHED with a §7 store; plain without ────────────────────────────


@pytest.mark.asyncio
async def test_announce_publishes_enriched_blob_when_s7_store_present(tmp_path):
    receipts = [_make_batched(i, with_tee=True) for i in range(3)]
    pb = _make_pb(receipts, batch_label=b"enriched-batch")
    store = _FakeStore()
    store.put(pb)
    leaf_hashes = _producer_leaf_hashes(receipts)

    # A REAL InferenceReceiptStore holding §7 receipts for two of the three leaves, keyed
    # by the SAME leaf hash an observer derives from the committed batch.
    ir_store = InferenceReceiptStore(tmp_path / "ir.json")
    ident = generate_node_identity("settler")
    for i in (0, 2):
        ir = _make_signed_ir(ident, job_id=f"job-{i}", request_id=f"req-{i}")
        ir_store.put(
            leaf_hashes[i], ir, settler_public_key_b64=ident.public_key_b64,
        )

    producer = generate_node_identity("producer")
    gossip = _FakeGossip(identity=producer)
    publisher = _FakeContentPublisher()
    announcer = SettlementBatchAnnouncer(
        store, gossip,
        content_publisher=publisher,
        inference_receipt_store=ir_store,
    )
    assert await announcer.announce(pb.batch_id) is True

    published_blob, _kw = publisher.published[0]
    # The published blob carries the §7 section.
    assert INFERENCE_RECEIPTS_KEY.encode() in published_blob
    rebuilt_pb, s7_map = deserialize_enriched_receipt_set(published_blob)
    assert rebuilt_pb == pb
    assert set(s7_map.keys()) == {leaf_hashes[0], leaf_hashes[2]}
    # The §7 settler signature survived the publish round-trip.
    for k in (leaf_hashes[0], leaf_hashes[2]):
        assert s7_map[k].inference_receipt.settler_signature
        assert s7_map[k].settler_public_key_b64 == ident.public_key_b64


@pytest.mark.asyncio
async def test_announce_publishes_plain_blob_when_no_s7_store():
    receipts = [_make_batched(i) for i in range(2)]
    pb = _make_pb(receipts, batch_label=b"plain-batch")
    store = _FakeStore()
    store.put(pb)
    gossip = _FakeGossip(identity=generate_node_identity("producer"))
    publisher = _FakeContentPublisher()
    announcer = SettlementBatchAnnouncer(store, gossip, content_publisher=publisher)

    assert await announcer.announce(pb.batch_id) is True
    published_blob, _kw = publisher.published[0]
    # Plain blob: NO §7 section, deserialize_receipt_set works.
    assert INFERENCE_RECEIPTS_KEY.encode() not in published_blob
    assert published_blob == serialize_receipt_set(pb)
    assert deserialize_receipt_set(published_blob) == pb


# ── (c) CID MATCH UNDER ENRICHMENT: ad cid == publisher infohash of enriched ─


@pytest.mark.asyncio
async def test_enriched_ad_cid_equals_publisher_infohash_of_enriched_blob(tmp_path):
    receipts = [_make_batched(i, with_tee=True) for i in range(3)]
    pb = _make_pb(receipts, batch_label=b"enriched-cid")
    store = _FakeStore()
    store.put(pb)
    leaf_hashes = _producer_leaf_hashes(receipts)

    ir_store = InferenceReceiptStore(tmp_path / "ir.json")
    ident = generate_node_identity("settler")
    ir = _make_signed_ir(ident)
    ir_store.put(leaf_hashes[1], ir, settler_public_key_b64=ident.public_key_b64)

    gossip = _FakeGossip(identity=generate_node_identity("producer"))
    publisher = _FakeContentPublisher()
    announcer = SettlementBatchAnnouncer(
        store, gossip,
        content_publisher=publisher,
        inference_receipt_store=ir_store,
    )
    assert await announcer.announce(pb.batch_id) is True

    published_blob, _kw = publisher.published[0]
    _, data = gossip.published[0]
    ad = SettlementBatchAdvertisement.from_gossip_data(data)
    # The ad cid resolves to the ENRICHED blob via the publisher.
    assert ad.cid == publisher._infohash(published_blob)
    assert publisher.fetch(ad.cid) == published_blob
    # And it is genuinely the enriched blob (carries the §7 receipt).
    _pb, s7 = deserialize_enriched_receipt_set(publisher.fetch(ad.cid))
    assert leaf_hashes[1] in s7


# ── (d) BEST-EFFORT: publish error + enrich error are swallowed ────────────


@pytest.mark.asyncio
async def test_announce_returns_false_when_publish_raises():
    receipts = [_make_batched(0)]
    pb = _make_pb(receipts, batch_label=b"boom")
    store = _FakeStore()
    store.put(pb)
    gossip = _FakeGossip(identity=generate_node_identity("producer"))
    announcer = SettlementBatchAnnouncer(store, gossip, content_publisher=_BoomPublisher())

    # A publish failure must NEVER raise into the (money-path) caller; returns False.
    assert await announcer.announce(pb.batch_id) is False
    # And nothing was gossiped (the cid would have been dangling).
    assert gossip.published == []


@pytest.mark.asyncio
async def test_announce_returns_false_when_enrich_raises(tmp_path):
    receipts = [_make_batched(0)]
    pb = _make_pb(receipts, batch_label=b"enrich-boom")
    store = _FakeStore()
    store.put(pb)
    gossip = _FakeGossip(identity=generate_node_identity("producer"))
    publisher = _FakeContentPublisher()

    class _BoomStore:
        def get(self, leaf_hash):
            raise RuntimeError("§7 store read failed")

    announcer = SettlementBatchAnnouncer(
        store, gossip,
        content_publisher=publisher,
        inference_receipt_store=_BoomStore(),
    )
    # enrich_from_store calls store.get -> raises; announce swallows it, returns False.
    assert await announcer.announce(pb.batch_id) is False
    # Best-effort: no half-published ad, no publish.
    assert gossip.published == []
    assert publisher.published == []


# ── (e) factory: enabled wires enrich+publish; disabled => None ────────────


@pytest.mark.asyncio
async def test_factory_builds_enriching_publishing_announcer(tmp_path):
    from prsm.settlement.settlement_audit_wiring import (
        BlockCursor,
        build_settlement_audit_components,
    )

    receipts = [_make_batched(i, with_tee=True) for i in range(3)]
    pb = _make_pb(receipts, batch_label=b"factory-batch")
    store = _FakeStore()
    store.put(pb)
    leaf_hashes = _producer_leaf_hashes(receipts)

    ir_store = InferenceReceiptStore(tmp_path / "ir.json")
    ident = generate_node_identity("settler")
    ir = _make_signed_ir(ident)
    ir_store.put(leaf_hashes[0], ir, settler_public_key_b64=ident.public_key_b64)

    gossip = _FakeGossip(identity=generate_node_identity("producer"))
    publisher = _FakeContentPublisher()

    bundle = build_settlement_audit_components(
        enabled=True,
        gossip=gossip,
        fetcher=object(),
        contract_client=object(),
        my_address=PROVIDER_ADDR,
        group_ids=None,
        dry_run_client=object(),
        cursor=BlockCursor(path=":memory:"),
        store=store,
        content_publisher=publisher,
        inference_receipt_store=ir_store,
    )
    assert bundle is not None
    assert bundle.announcer is not None

    assert await bundle.announcer.announce(pb.batch_id) is True
    published_blob, _kw = publisher.published[0]
    # Factory-built announcer enriched + published.
    assert INFERENCE_RECEIPTS_KEY.encode() in published_blob
    _, data = gossip.published[0]
    ad = SettlementBatchAdvertisement.from_gossip_data(data)
    assert ad.cid == publisher._infohash(published_blob)
    _pb, s7 = deserialize_enriched_receipt_set(publisher.fetch(ad.cid))
    assert leaf_hashes[0] in s7


def test_factory_disabled_returns_none():
    from prsm.settlement.settlement_audit_wiring import (
        BlockCursor,
        build_settlement_audit_components,
    )

    bundle = build_settlement_audit_components(
        enabled=False,
        gossip=_FakeGossip(),
        fetcher=object(),
        contract_client=object(),
        my_address=PROVIDER_ADDR,
        group_ids=None,
        dry_run_client=object(),
        cursor=BlockCursor(path=":memory:"),
        store=_FakeStore(),
        content_publisher=_FakeContentPublisher(),
        inference_receipt_store=object(),
    )
    assert bundle is None


# ── (f) DEFAULT-OFF: no publisher/store => legacy plain-blob behavior ──────


@pytest.mark.asyncio
async def test_default_off_legacy_announcer_no_publish_legacy_cid():
    receipts = [_make_batched(i) for i in range(3)]
    pb = _make_pb(receipts, batch_label=b"legacy-batch")
    store = _FakeStore()
    store.put(pb)
    gossip = _FakeGossip(identity=generate_node_identity("producer"))

    # No content_publisher, no inference_receipt_store -> EXACTLY the sp1137 behavior.
    announcer = SettlementBatchAnnouncer(store, gossip)
    assert await announcer.announce(pb.batch_id) is True

    _, data = gossip.published[0]
    ad = SettlementBatchAdvertisement.from_gossip_data(data)
    # Legacy cid (receipt_set_cid over the PLAIN blob) — may dangle without a publisher,
    # but is byte-for-byte the prior behavior.
    assert ad.cid == receipt_set_cid(serialize_receipt_set(pb))


@pytest.mark.asyncio
async def test_factory_default_off_legacy_announcer(tmp_path):
    """build_settlement_audit_components with no content_publisher/inference_receipt_store
    still builds a legacy (plain-blob, legacy-cid) announcer."""
    from prsm.settlement.settlement_audit_wiring import (
        BlockCursor,
        build_settlement_audit_components,
    )

    receipts = [_make_batched(i) for i in range(2)]
    pb = _make_pb(receipts, batch_label=b"factory-legacy")
    store = _FakeStore()
    store.put(pb)
    gossip = _FakeGossip(identity=generate_node_identity("producer"))

    bundle = build_settlement_audit_components(
        enabled=True,
        gossip=gossip,
        fetcher=object(),
        contract_client=object(),
        my_address=PROVIDER_ADDR,
        group_ids=None,
        dry_run_client=object(),
        cursor=BlockCursor(path=":memory:"),
        store=store,
    )
    assert bundle is not None
    assert bundle.announcer is not None
    assert await bundle.announcer.announce(pb.batch_id) is True
    _, data = gossip.published[0]
    ad = SettlementBatchAdvertisement.from_gossip_data(data)
    assert ad.cid == receipt_set_cid(serialize_receipt_set(pb))
