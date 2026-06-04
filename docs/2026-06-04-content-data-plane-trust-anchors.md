# Content data-plane trust anchors — hunt findings, fixes, and residual decisions

**Sprints 1002–1004.** The content-data-plane integrity adversarial hunt
(workflow `w8gvkhelm`, 5 dimensions, default-refute with reachability +
crypto/origin-auth backstop discipline) confirmed 8 findings about whether a
malicious peer can serve wrong bytes for a CID, poison a peer's content index
with false metadata, forge provenance/dedup, corrupt shard reassembly, or
exhaust a retriever. This doc records what shipped autonomously, and the
**residual gaps that need a design decision** (on-chain ceremony or a
retrieval-routing/protocol change) before they can close — mirroring the
sp1000 storage-slashing-gap pattern.

## The two trust lanes (context)

Content metadata reaches a node over two gossip lanes with very different
trust properties:

- **Provenance lane** (`_on_provenance_register` → `_verified_upsert`, sp964):
  fully authenticated — ed25519 signature + `node_id == sha256(pubkey)[:32]`
  creator-binding + first-writer-wins, persisted to the durable ledger. This
  is the authoritative creator/royalty record and backs the **on-chain**
  royalty leg.
- **Advertise lane** (`_on_content_advertise`): **unauthenticated**. Any peer
  may broadcast a `GOSSIP_CONTENT_ADVERTISE` for any CID. sp934 authenticates
  *who* sent it (the origin node_id) but does **not** bind the *claim* — the
  advertiser need not hold the content, nor be its creator. This lane
  populates the in-memory `ContentRecord` (providers, `content_hash`,
  `creator_eth_address`, `royalty_rate`, `provenance_hash`, size, filename).

The findings all stem from trusting advertise-lane data for integrity or
credit decisions.

## Shipped (autonomous, reversible)

### sp1002 — GATEWAY fetch DoS bounds (findings 7, 8)
`ContentProvider._fetch_from_url` now caps bytes (stream + abort; fast-fail on
an oversized `Content-Length`) and runs under the caller's remaining timeout,
not a fixed 120s. The cap is the advertised size (×2 + slack) bounded by a
hard ceiling (`PRSM_MAX_GATEWAY_FETCH_BYTES`, default 2 GiB). Closes the OOM
and slow-loris vectors on the gateway lane. (INLINE is already bounded by
`MAX_INLINE_SIZE` = 1 MiB.)

### sp1003 — CID-anchored content-substitution defense (findings 1, 2, 4)
The remote fetch lane verified returned bytes against `expected_hash` =
`ContentRecord.content_hash`, which is advertise-lane data and therefore
**attacker-controllable**: a malicious provider advertises
`content_hash = sha256(evil)` (first-writer-wins locks it in), becomes a
routed provider, serves `evil`, and the check passes — or advertises
`content_hash=""` and the check is skipped entirely.

Fix: when the CID is an unambiguous **`ContentHash`** content-address
(round-trips through `from_hex` AND carries the algorithm's canonical digest
length), `_request_from_provider` re-hashes the returned bytes and checks them
against the **CID itself** (which the requester chose and trusts), rejecting on
mismatch — regardless of the gossip `content_hash`. The `ContentStore` is
content-addressed, so genuine content always re-hashes to its CID (cipher or
plaintext — the CID addresses whatever bytes were stored), making this
non-regressing. Closes substitution for all `ContentHash`-addressed content.

## ★ RESIDUAL GAP A — BitTorrent-infohash CIDs are not verifiable from inline bytes

A CID published by the BitTorrent upload path is a **v1 infohash** =
`sha1(bencode(info-dict))`, where the info-dict carries `piece_length`,
`name`, and the per-piece hashes. It is content-derived but **cannot be
reconstructed from the raw inline bytes alone** — `ContentResponseMessage`
carries only `data`/`content_hash`/`size`/`filename`, not the torrent
metadata. So for a BT-infohash CID on the INLINE/GATEWAY lane there is **no
trustworthy integrity anchor**: the sp1003 CID-anchor correctly defers (no
false reject), and the only remaining check is the attacker-controllable
gossip `content_hash`. An empty gossip hash → bytes accepted unverified.

This is pinned by `test_bt_infohash_residual_gap_documented` in
`tests/unit/test_sprint_1003_cid_anchored_integrity.py` (it asserts the
current accept-unverified behaviour; the assertion flips when the gap closes).

The piece-verifying path (`ContentRetriever.fetch`, libtorrent) DOES anchor on
the infohash, but is gated to `_local_content` (this node's own content) to
avoid swarming the BT layer for arbitrary CIDs.

**Design decision (retrieval routing / wire protocol):**
- **Option A — route remote BT-infohash fetches through the piece-verifying
  swarm.** For a deliberately-requested CID, joining the swarm by infohash and
  letting libtorrent verify pieces against the CID is the correct content-
  addressing guarantee. Cost: heavier than the INLINE fast-path; needs the
  torrent manifest discoverable for remote CIDs; risks regressing the tuned
  F7/F8 single-node retrieve path — wants a multi-node test bench to verify.
- **Option B — carry the torrent info-dict (or `piece_length`+`name`) in the
  response** so the retriever can recompute the infohash from the inline
  bytes and check it against the CID. Smaller change, but grows the wire
  message and only covers single-file v1 torrents cleanly.
- **Option C — deprecate BT-infohash CIDs in favour of `ContentHash` content-
  addresses end-to-end** (already self-verifying per sp1003). Largest change;
  cleanest long-term trust model.

**Recommendation:** Option A is the true content-addressing fix and reuses the
existing piece-verifying retriever; gate it behind a multi-node verification of
the F7/F8 retrieve path. Until one lands, BT-infohash remote content on the
INLINE/GATEWAY lane is integrity-unverified and operators should treat
BT-served bytes as best-effort.

## ★ RESIDUAL GAP B — advertise-lane credit fields are unauthenticated (findings 3, 5, 6)

The unauthenticated advertise lane lets any peer create/overwrite an in-memory
record's `creator_eth_address`, `royalty_rate`, and `provenance_hash`. Impact
analysis (what is and is NOT backstopped):

- **On-chain royalty leg — BACKSTOPPED (sp996).** `onchain_content_royalty`
  dispatches on `record.provenance_hash` to the on-chain `ProvenanceRegistry`,
  which pays the **registered** creator at the **registered** rate. A poisoned
  `provenance_hash` either is unregistered (→ `skipped_unregistered`, no
  payout) or is a hash legitimately registered by its true creator (→ pays the
  true creator, not the attacker). No on-chain ETH theft.
- **Off-chain credit / pool-splits / §14 reputation — NOT backstopped.** The
  in-memory `creator_eth_address` feeds the retrieve→`record_access`
  reputation auto-record, and `royalty_rate` weights multi-shard pool splits.
  A first/sole advertiser sets these to attacker-chosen values; a later
  advertise can bump a default-`0.01` `royalty_rate` to skew split weighting.
  (`record_access` additionally requires the operator's on-chain address, so
  the reputation poison is reachable only where that is wired.)

**sp1004 ships the bounded, clearly-correct part:** clamp advertise-ingested
`royalty_rate` to a sane non-negative finite bound and reject NaN/inf/negative
sizes, so absolute-insanity values (negative rates, `0.98` vs others' `0.01`,
NaN) cannot enter the index. This bounds the damage but does **not** stop a
within-bounds relative-skew.

**Design decision (the complete fix):** the credit-bearing fields should defer
to the **authenticated** source. Two shapes:
- **Option A — bind off-chain credit to the provenance ledger.** When a record
  has an authenticated provenance entry, off-chain readers (`record_access`,
  pool-split weighting) use the ledger's creator/rate, not the advertise-lane
  copy. Keeps the advertise lane as a routing hint only.
- **Option B — sign the advertise.** Extend `GOSSIP_CONTENT_ADVERTISE` with a
  creator signature over the credit fields, verified like the provenance lane,
  so a non-creator cannot set them. Larger protocol change; advertises are
  currently unsigned and high-fanout.

**Recommendation:** Option A — the authoritative data already lives in the
signed, first-writer-wins ledger; route off-chain credit through it rather
than the gossip copy. No new wire format, no new signatures on the hot path.

## Not-a-bug (refuted or already-backstopped, recorded for audit)

- **Provenance/dedup forgery (finding 6's provenance half):** the *ledger*
  provenance record is signature-bound + first-writer-wins (sp964/965); the
  fingerprint dedup is first-creator-wins (sp441). Only the advertise-lane
  in-memory copy is forgeable, and only for off-chain credit (Gap B).
- **Shard-reassembly integrity:** the hunt's shard dimension surfaced no
  confirmed unbacked path — reassembled tier-A/B/C content is checked against
  its content hash, and erasure shards are verified on decode
  (`shard_engine` `ContentHash.from_data` re-check). No fix required.
