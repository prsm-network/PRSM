"""Sprint 1137 — settlement-receipt data plane BRICK C: the availability
ANNOUNCE / OBSERVE layer (design doc 2026-06-16 §5-7, Brick C).

Brick A (sp1135) retains each committed batch's ordered receipt set as a
``PublishedBatch``; Brick B (sp1136) turns that set into a canonical CID-addressable
blob + the on-chain-root cross-check gate. Brick C adds the thin, AUTHENTICATED,
pull-on-demand availability layer: a producer ANNOUNCES that a batch exists + where
to fetch it (a tiny ad), and an observer INDEXES that ad as an UNTRUSTED POINTER so
it can later fetch the blob (Brick D/E) and run the Brick B cross-check.

TRUST MODEL under test: the ad is an untrusted pointer. Gossip origin-auth (sp934)
only proves WHO gossiped it — for spam/replay accounting + dedup — it does NOT make
the pointer correct (correctness is established LATER by the Brick B root check).
So the index must:
  - record pointers, never treat an ad as authoritative;
  - dedup by batch_id (first-seen wins) so a later / forged-origin re-ad for a known
    batch_id can't overwrite the first entry;
  - bound its size (evict oldest);
  - drop a malformed ad or one whose origin does NOT authenticate, WITHOUT crashing
    and WITHOUT evicting/corrupting an existing good entry (sp1010 fail-safe).

These tests are OFFLINE: a FAKE gossip records publish() calls and faithfully
mirrors the real receive path — it runs the REAL ``_authenticate_origin`` over a
P2PMessage payload before dispatching to the subscribed handler, so the index's
authentication is exercised exactly as it would be on the wire (no network).

  (a) ad to_gossip_data -> from_gossip_data round-trips byte-exact (bytes + optional
      leaf_hashes).
  (b) announcer reads a store entry and publishes ONE ad on the subtype carrying the
      right batch_id / merkle_root / CID (CID == receipt_set_cid(serialize_receipt_set)).
  (c) index ingests an authenticated ad -> get(batch_id) returns it.
  (d) DEDUP: a second ad for the same batch_id (esp. from a DIFFERENT origin) does
      NOT overwrite the first.
  (e) FAIL-SAFE: a malformed ad payload and an ad whose origin does not authenticate
      are dropped without raising and without evicting/corrupting a good entry.
  (f) BOUND: max_ads evicts the oldest.
  (g) ADDITIVE: GOSSIP_RETENTION_SECONDS still contains every prior subtype + the new
      one (no clobber).
"""
from __future__ import annotations

import hashlib
from base64 import b64encode

import pytest

from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.node.gossip import (
    GOSSIP_RETENTION_SECONDS,
    GOSSIP_SETTLEMENT_BATCH_AVAILABLE,
    _authenticate_origin,
    build_gossip_origin_fields,
)
from prsm.node.identity import generate_node_identity
from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.merkle import (
    batched_receipt_to_leaf,
    build_tree_and_proofs,
    hash_leaf,
)
from prsm.settlement.published_batch_store import PublishedBatch
from prsm.settlement.receipt_set_blob import receipt_set_cid, serialize_receipt_set
from prsm.settlement.settlement_gossip import (
    SettlementBatchAdIndex,
    SettlementBatchAdvertisement,
    SettlementBatchAnnouncer,
)


ONE_FTNS = 10**18
PROVIDER_ADDR = "0x" + "b" * 40
REQUESTER_A = "0x" + "a" * 40


# ── fixtures ──────────────────────────────────────────────────────────


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


def _ad_data(ad: SettlementBatchAdvertisement, announcer_identity) -> dict:
    """Gossip envelope for an ad: the pointer fields + the announcer's gossip
    node_id (announcer_id), exactly as SettlementBatchAnnouncer builds it."""
    data = ad.to_gossip_data()
    data["announcer_id"] = announcer_identity.node_id
    return data


def _producer_root(receipts) -> bytes:
    return build_tree_and_proofs(
        [hash_leaf(batched_receipt_to_leaf(br)) for br in receipts]
    ).root


def _make_pb(receipts, *, batch_label: bytes = b"batch") -> PublishedBatch:
    return PublishedBatch(
        batch_id=hashlib.sha256(batch_label).digest(),
        merkle_root=_producer_root(receipts),
        commit_timestamp=1700001234,
        receipts=list(receipts),
    )


class _FakeStore:
    """Minimal stand-in for PublishedBatchStore.get(batch_id)."""

    def __init__(self):
        self._d = {}

    def put(self, pb: PublishedBatch) -> None:
        self._d[bytes(pb.batch_id).hex()] = pb

    def get(self, batch_id: bytes):
        return self._d.get(bytes(batch_id).hex())


class _FakeGossip:
    """Records publish() calls and faithfully mirrors the real receive path.

    publish(subtype, data) -> records (subtype, data).
    subscribe(subtype, handler) -> registers handler.
    deliver_from(identity, subtype, data, *, tamper=None, drop_attestation=False)
        builds a real signed origin attestation (sp934) via the live
        build_gossip_origin_fields, then runs the REAL _authenticate_origin over the
        payload (exactly as GossipProtocol._handle_gossip does) and dispatches the
        AUTHENTICATED origin to the subscribed handler. This exercises the index's
        authentication on the genuine wire format with no network.
    """

    def __init__(self, identity=None):
        self.published = []
        self._subscribers = {}
        # mirror GossipProtocol.transport.identity so the announcer can bind its
        # own gossip node_id into the ad (announcer_id) — the index verifies the
        # authenticated gossip origin against it.
        self.transport = type("T", (), {"identity": identity})()

    async def publish(self, subtype, data, ttl=None):
        self.published.append((subtype, data))
        return 0

    def subscribe(self, subtype, handler):
        self._subscribers.setdefault(subtype, []).append(handler)

    async def deliver_from(
        self, identity, subtype, data, *, relay_sender=None, tamper=None,
        drop_attestation=False,
    ):
        """Simulate a received, origin-signed gossip message reaching subscribers.

        ``identity`` is the attested author (the producer). ``relay_sender`` (if
        given) is the immediate transport peer that forwarded it — used to model a
        relayed / forged ad: when the attestation is stripped, _authenticate_origin
        falls back to the relaying sender, which does NOT match the producer the ad
        claims to come from.
        """
        nonce = "nonce-" + hashlib.sha256(
            repr((identity.node_id, subtype, sorted(data.items()))).encode()
        ).hexdigest()[:12]
        fields = build_gossip_origin_fields(identity, subtype, data, 123.0, nonce)
        if drop_attestation:
            fields = {"origin": identity.node_id}  # no pubkey/sig -> cannot authenticate
        payload = {"subtype": subtype, "data": tamper if tamper is not None else data, **fields}
        # immediate transport sender (may differ from the attested author on relay)
        sender_id = (relay_sender or identity).node_id
        authed_origin = _authenticate_origin(payload, nonce, sender_id)
        for cb in self._subscribers.get(subtype, []):
            await cb(subtype, payload["data"], authed_origin)


# ── (a) ad serialization round-trip ───────────────────────────────────


def test_advertisement_round_trips_byte_exact():
    batch_id = hashlib.sha256(b"batch-a").digest()
    merkle_root = hashlib.sha256(b"root-a").digest()
    leaves = [hashlib.sha256(f"leaf-{i}".encode()).digest() for i in range(3)]
    ad = SettlementBatchAdvertisement(
        batch_id=batch_id,
        provider_address=PROVIDER_ADDR,
        merkle_root=merkle_root,
        cid="a" * 40,
        leaf_hashes=leaves,
        commit_timestamp=1700001234,
    )
    data = ad.to_gossip_data()
    # gossip data must be JSON-native (hex strings for bytes)
    assert isinstance(data["batch_id"], str)
    assert isinstance(data["merkle_root"], str)
    assert all(isinstance(h, str) for h in data["leaf_hashes"])

    back = SettlementBatchAdvertisement.from_gossip_data(data)
    assert back.batch_id == batch_id
    assert back.merkle_root == merkle_root
    assert back.provider_address == PROVIDER_ADDR
    assert back.cid == "a" * 40
    assert back.leaf_hashes == leaves
    assert back.commit_timestamp == 1700001234
    assert back == ad


def test_advertisement_round_trips_without_leaf_hashes():
    ad = SettlementBatchAdvertisement(
        batch_id=hashlib.sha256(b"b").digest(),
        provider_address=PROVIDER_ADDR,
        merkle_root=hashlib.sha256(b"r").digest(),
        cid="c" * 40,
        leaf_hashes=None,
        commit_timestamp=42,
    )
    back = SettlementBatchAdvertisement.from_gossip_data(ad.to_gossip_data())
    assert back == ad
    assert back.leaf_hashes is None


# ── (b) announcer publishes exactly one correct ad ─────────────────────


@pytest.mark.asyncio
async def test_announcer_publishes_one_correct_ad():
    receipts = [_make_batched(i, with_tee=(i % 2 == 0)) for i in range(4)]
    pb = _make_pb(receipts, batch_label=b"announce-batch")
    store = _FakeStore()
    store.put(pb)
    producer = generate_node_identity("producer")
    gossip = _FakeGossip(identity=producer)
    announcer = SettlementBatchAnnouncer(store, gossip)

    await announcer.announce(pb.batch_id)

    assert len(gossip.published) == 1
    subtype, data = gossip.published[0]
    assert subtype == GOSSIP_SETTLEMENT_BATCH_AVAILABLE
    assert data["announcer_id"] == producer.node_id

    expected_cid = receipt_set_cid(serialize_receipt_set(pb))
    ad = SettlementBatchAdvertisement.from_gossip_data(data)
    assert ad.batch_id == pb.batch_id
    assert ad.merkle_root == pb.merkle_root
    assert ad.cid == expected_cid
    assert ad.commit_timestamp == pb.commit_timestamp
    # leaf hashes (when inlined) must equal the producer leaf pipeline
    expected_leaves = [hash_leaf(batched_receipt_to_leaf(br)) for br in receipts]
    assert ad.leaf_hashes == expected_leaves


@pytest.mark.asyncio
async def test_announcer_missing_batch_is_noop_no_publish():
    store = _FakeStore()
    gossip = _FakeGossip()
    announcer = SettlementBatchAnnouncer(store, gossip)
    # batch not in store -> best-effort no-op, must not raise, must not publish
    await announcer.announce(hashlib.sha256(b"missing").digest())
    assert gossip.published == []


@pytest.mark.asyncio
async def test_announcer_swallows_publish_error():
    receipts = [_make_batched(0)]
    pb = _make_pb(receipts, batch_label=b"err-batch")
    store = _FakeStore()
    store.put(pb)

    class _BoomGossip(_FakeGossip):
        async def publish(self, subtype, data, ttl=None):
            raise RuntimeError("network down")

    announcer = SettlementBatchAnnouncer(store, _BoomGossip())
    # best-effort: a publish error must NOT raise into the caller (money path safe)
    await announcer.announce(pb.batch_id)


# ── (c) index ingests an authenticated ad ──────────────────────────────


@pytest.mark.asyncio
async def test_index_ingests_authenticated_ad():
    receipts = [_make_batched(i) for i in range(3)]
    pb = _make_pb(receipts, batch_label=b"idx-batch")
    store = _FakeStore()
    store.put(pb)
    producer = generate_node_identity("producer")
    gossip = _FakeGossip(identity=producer)
    index = SettlementBatchAdIndex(gossip)

    # producer announces; relay the recorded ad data through the authenticated path
    announcer = SettlementBatchAnnouncer(store, gossip)
    await announcer.announce(pb.batch_id)
    _, ad_data = gossip.published[-1]

    await gossip.deliver_from(producer, GOSSIP_SETTLEMENT_BATCH_AVAILABLE, ad_data)

    got = index.get(pb.batch_id)
    assert got is not None
    assert got.batch_id == pb.batch_id
    assert got.cid == receipt_set_cid(serialize_receipt_set(pb))
    assert got.merkle_root == pb.merkle_root
    assert len(index.all_ads()) == 1


# ── (d) dedup: first-seen wins, even across origins ────────────────────


@pytest.mark.asyncio
async def test_dedup_second_ad_does_not_overwrite():
    batch_id = hashlib.sha256(b"dedup-batch").digest()
    gossip = _FakeGossip()
    index = SettlementBatchAdIndex(gossip)
    honest = generate_node_identity("honest")
    attacker = generate_node_identity("attacker")

    good = SettlementBatchAdvertisement(
        batch_id=batch_id, provider_address=PROVIDER_ADDR,
        merkle_root=hashlib.sha256(b"good-root").digest(), cid="g" * 40,
        leaf_hashes=None, commit_timestamp=100,
    )
    forged = SettlementBatchAdvertisement(
        batch_id=batch_id, provider_address="0x" + "f" * 40,
        merkle_root=hashlib.sha256(b"FORGED-root").digest(), cid="f" * 40,
        leaf_hashes=None, commit_timestamp=999,
    )

    await gossip.deliver_from(honest, GOSSIP_SETTLEMENT_BATCH_AVAILABLE, _ad_data(good, honest))
    # a DIFFERENT authenticated origin re-ads the same batch_id with different data
    await gossip.deliver_from(attacker, GOSSIP_SETTLEMENT_BATCH_AVAILABLE, _ad_data(forged, attacker))

    stored = index.get(batch_id)
    assert stored.cid == "g" * 40
    assert stored.merkle_root == good.merkle_root  # first-seen retained
    assert len(index.all_ads()) == 1


# ── (e) fail-safe: malformed + unauthenticated drops, no corruption ────


@pytest.mark.asyncio
async def test_failsafe_malformed_and_unauthenticated_dropped():
    good_batch = hashlib.sha256(b"good").digest()
    gossip = _FakeGossip()
    index = SettlementBatchAdIndex(gossip)
    honest = generate_node_identity("honest")
    attacker = generate_node_identity("attacker")

    good = SettlementBatchAdvertisement(
        batch_id=good_batch, provider_address=PROVIDER_ADDR,
        merkle_root=hashlib.sha256(b"r").digest(), cid="g" * 40,
        leaf_hashes=None, commit_timestamp=100,
    )
    await gossip.deliver_from(honest, GOSSIP_SETTLEMENT_BATCH_AVAILABLE, _ad_data(good, honest))
    assert index.get(good_batch) is not None

    # 1) malformed payload (missing required fields / garbage) -> dropped, no raise
    await gossip.deliver_from(honest, GOSSIP_SETTLEMENT_BATCH_AVAILABLE, {"junk": "x"})
    await gossip.deliver_from(honest, GOSSIP_SETTLEMENT_BATCH_AVAILABLE,
                              {"batch_id": "not-hex-zz", "merkle_root": "00", "cid": "c",
                               "provider_address": PROVIDER_ADDR, "commit_timestamp": 1,
                               "announcer_id": honest.node_id})

    # 2) an ad whose origin does NOT authenticate (attestation stripped, relayed by a
    #    DIFFERENT peer) -> authenticated origin falls back to the relaying sender,
    #    which does not match the embedded announcer_id -> dropped.
    forged = SettlementBatchAdvertisement(
        batch_id=hashlib.sha256(b"forged").digest(), provider_address="0x" + "f" * 40,
        merkle_root=hashlib.sha256(b"fr").digest(), cid="f" * 40,
        leaf_hashes=None, commit_timestamp=5,
    )
    relayer = generate_node_identity("relayer")
    await gossip.deliver_from(
        attacker, GOSSIP_SETTLEMENT_BATCH_AVAILABLE, _ad_data(forged, attacker),
        relay_sender=relayer, drop_attestation=True,
    )

    # existing good entry survived intact; no forged/malformed pointer leaked in
    survivor = index.get(good_batch)
    assert survivor is not None
    assert survivor.cid == "g" * 40
    assert index.get(hashlib.sha256(b"forged").digest()) is None
    assert len(index.all_ads()) == 1


@pytest.mark.asyncio
async def test_unauthenticated_origin_ad_is_dropped_when_provider_unbound():
    """An ad whose authenticated origin cannot be established (no attestation, so
    gossip falls back to the bare transport sender that does not bind the attested
    author) must be treated fail-safe: dropped, not recorded as a pointer."""
    gossip = _FakeGossip()
    index = SettlementBatchAdIndex(gossip)
    spoofer = generate_node_identity("spoofer")
    relayer = generate_node_identity("relayer")
    ad = SettlementBatchAdvertisement(
        batch_id=hashlib.sha256(b"unauth").digest(), provider_address="0x" + "9" * 40,
        merkle_root=hashlib.sha256(b"m").digest(), cid="9" * 40,
        leaf_hashes=None, commit_timestamp=1,
    )
    await gossip.deliver_from(
        spoofer, GOSSIP_SETTLEMENT_BATCH_AVAILABLE, _ad_data(ad, spoofer),
        relay_sender=relayer, drop_attestation=True,
    )
    assert index.all_ads() == []


# ── (f) bound: evict oldest ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_index_bound_evicts_oldest():
    gossip = _FakeGossip()
    index = SettlementBatchAdIndex(gossip, max_ads=3)
    producer = generate_node_identity("producer")

    ads = []
    for i in range(5):
        ad = SettlementBatchAdvertisement(
            batch_id=hashlib.sha256(f"b{i}".encode()).digest(),
            provider_address=PROVIDER_ADDR,
            merkle_root=hashlib.sha256(f"r{i}".encode()).digest(),
            cid=f"{i:040x}", leaf_hashes=None, commit_timestamp=i,
        )
        ads.append(ad)
        await gossip.deliver_from(producer, GOSSIP_SETTLEMENT_BATCH_AVAILABLE, _ad_data(ad, producer))

    assert len(index.all_ads()) == 3
    # oldest two evicted
    assert index.get(ads[0].batch_id) is None
    assert index.get(ads[1].batch_id) is None
    assert index.get(ads[2].batch_id) is not None
    assert index.get(ads[4].batch_id) is not None


# ── (g) additive: retention dict unchanged for prior subtypes ──────────


def test_retention_additive_new_subtype_present_priors_intact():
    assert GOSSIP_SETTLEMENT_BATCH_AVAILABLE == "settlement_batch_available"
    assert GOSSIP_SETTLEMENT_BATCH_AVAILABLE in GOSSIP_RETENTION_SECONDS
    # challenge window + margin (>= 2 days)
    assert GOSSIP_RETENTION_SECONDS[GOSSIP_SETTLEMENT_BATCH_AVAILABLE] >= 2 * 86400
    # all prior subtypes still present, no clobber
    for prior in (
        "content_advertise", "content_access", "provenance_register",
        "job_offer", "payment_confirm", "agent_advertise", "bittorrent_announce",
        "heartbeat",
    ):
        assert prior in GOSSIP_RETENTION_SECONDS
