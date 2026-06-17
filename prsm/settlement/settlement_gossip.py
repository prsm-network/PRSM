"""Sprint 1137 — settlement-receipt data plane BRICK C: the availability
ANNOUNCE / OBSERVE layer (design doc 2026-06-16 §5-7, Brick C).

Brick A (sp1135, ``published_batch_store.py``) retains each committed batch's
ordered receipt set as a ``PublishedBatch``. Brick B (sp1136, ``receipt_set_blob.py``)
turns that set into a canonical, CID-addressable BLOB + the on-chain-root cross-check
gate. Brick C adds the thin layer that lets an observer LEARN which batches exist and
WHERE to fetch them — without flooding the receipt data itself onto the wire.

PULL-ON-DEMAND, NOT FLOOD: the producer gossips a tiny AD (an untrusted pointer:
``batch_id``, ``provider_address``, ``merkle_root``, the receipt-set ``cid``, and
optionally the inlined ``leaf_hashes``). An observer indexes the pointer and FETCHES
the actual blob later (Brick D/E), then verifies it against the on-chain root via
Brick B (``verify_receipt_set_against_root``). The ad is never authoritative.

TRUST MODEL (design §6): gossip origin-auth (sp934) proves only WHO gossiped the ad —
useful for spam/replay accounting + dedup — it does NOT make the pointer CORRECT.
Correctness is established later by the on-chain root cross-check. So:
  - the index records pointers only; it never treats an ad as authoritative;
  - DEDUP key is ``batch_id``, FIRST-SEEN WINS: a later ad for a known batch_id (even
    from a different, authenticated origin) is ignored, so a forged-origin re-ad can
    never overwrite/corrupt the first-seen pointer;
  - the index is BOUNDED (``max_ads``, evict oldest) so an attacker can't grow it;
  - FAIL-SAFE (sp1010 libp2p origin-auth gap): an ad whose origin cannot be
    authenticated as the producer it claims to be — or a malformed payload — is
    DROPPED (logged), never crashing and never evicting/corrupting an existing entry.

ORIGIN BINDING: the producer embeds its own gossip node_id as ``announcer_id`` in the
ad envelope. The gossip layer authenticates the origin (sp934) and hands the index the
AUTHENTICATED origin; the index admits the ad only when that authenticated origin
equals the embedded ``announcer_id``. A relayed/forged ad whose attestation does not
verify falls back to the relaying transport sender, which won't match ``announcer_id``
— so it is dropped (untrusted-pointer model: better to drop than to record a pointer
we cannot attribute, since the dedup slot is first-seen-wins and precious).

OPT-IN / ADDITIVE: constructing a ``SettlementBatchAnnouncer`` or ``SettlementBatchAdIndex``
is the only thing that touches the new subtype. Merely importing this module changes no
default node behavior on the wire (Brick E wires them). The announcer is best-effort:
a serialize/publish failure is logged and never raised into the (money-path) caller.

This module REUSES Brick A's store + Brick B's ``serialize_receipt_set`` /
``receipt_set_cid``; it reinvents none of that.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from prsm.node.gossip import GOSSIP_SETTLEMENT_BATCH_AVAILABLE
from prsm.settlement.merkle import batched_receipt_to_leaf, hash_leaf
from prsm.settlement.published_batch_store import PublishedBatch
from prsm.settlement.receipt_set_blob import receipt_set_cid, serialize_receipt_set

logger = logging.getLogger(__name__)

# Defensive bound on inlined leaf hashes parsed from an UNTRUSTED ad (a real batch is
# ~1000 receipts; this rejects an adversarial multi-million-leaf payload before it can
# blow up memory). The blob fetched later (Brick B) has its own, separate cap.
_MAX_AD_LEAF_HASHES = 1_000_000
_DEFAULT_MAX_ADS = 50_000


@dataclass(frozen=True)
class SettlementBatchAdvertisement:
    """An UNTRUSTED availability pointer for a committed batch.

    Carries just enough for an observer to fetch + verify the receipt set later:
      - ``batch_id``     — the dedup key (bytes);
      - ``provider_address`` — the on-chain provider (ETH address string);
      - ``merkle_root``  — the committed root (bytes), for a quick pre-fetch sanity
        check against the chain (authoritative verification is still the Brick B
        cross-check over the fetched blob);
      - ``cid``          — the receipt-set blob CID (Brick B), the fetch handle;
      - ``leaf_hashes``  — optional inlined leaf hashes (bytes), enabling the cheap
        leaf-only double-spend path without fetching the full blob;
      - ``commit_timestamp`` — for ordering / GC.

    ``to_gossip_data`` / ``from_gossip_data`` are byte-exact inverses (bytes fields
    hex-encoded so the payload is JSON-native for the gossip wire).
    """
    batch_id: bytes
    provider_address: str
    merkle_root: bytes
    cid: str
    leaf_hashes: Optional[List[bytes]]
    commit_timestamp: int

    def to_gossip_data(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "batch_id": bytes(self.batch_id).hex(),
            "provider_address": self.provider_address,
            "merkle_root": bytes(self.merkle_root).hex(),
            "cid": self.cid,
            "commit_timestamp": int(self.commit_timestamp),
        }
        if self.leaf_hashes is not None:
            data["leaf_hashes"] = [bytes(h).hex() for h in self.leaf_hashes]
        return data

    @classmethod
    def from_gossip_data(cls, data: Dict[str, Any]) -> "SettlementBatchAdvertisement":
        """Inverse of ``to_gossip_data``. Raises ``ValueError`` on a malformed /
        oversized payload (the index catches + drops; never silently mis-parses)."""
        if not isinstance(data, dict):
            raise ValueError(f"ad payload must be an object, got {type(data).__name__}")
        try:
            batch_id = bytes.fromhex(data["batch_id"])
            merkle_root = bytes.fromhex(data["merkle_root"])
            provider_address = data["provider_address"]
            cid = data["cid"]
            commit_timestamp = int(data["commit_timestamp"])
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"malformed settlement batch ad: {exc}") from exc
        if not isinstance(provider_address, str) or not isinstance(cid, str):
            raise ValueError("ad provider_address/cid must be strings")

        leaf_hashes: Optional[List[bytes]] = None
        raw_leaves = data.get("leaf_hashes")
        if raw_leaves is not None:
            if not isinstance(raw_leaves, list):
                raise ValueError("ad leaf_hashes must be a list")
            if len(raw_leaves) > _MAX_AD_LEAF_HASHES:
                raise ValueError(
                    f"ad leaf_hashes count {len(raw_leaves)} exceeds cap "
                    f"{_MAX_AD_LEAF_HASHES}"
                )
            try:
                leaf_hashes = [bytes.fromhex(h) for h in raw_leaves]
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"ad leaf_hashes not hex: {exc}") from exc

        return cls(
            batch_id=batch_id,
            provider_address=provider_address,
            merkle_root=merkle_root,
            cid=cid,
            leaf_hashes=leaf_hashes,
            commit_timestamp=commit_timestamp,
        )


class SettlementBatchAnnouncer:
    """OPT-IN producer-side component: announce that a committed batch is available.

    ``announce(batch_id)`` reads the ``PublishedBatch`` from the (Brick A) store,
    serializes it (Brick B ``serialize_receipt_set``), computes the receipt-set CID,
    builds the ad (inlining the producer leaf hashes so observers get the cheap
    double-spend path for free), embeds the announcer's gossip node_id, and gossips it
    on ``GOSSIP_SETTLEMENT_BATCH_AVAILABLE``.

    BEST-EFFORT: only does anything when explicitly called; a missing batch is a
    no-op; a serialize/publish error is logged and NEVER raised into the caller (so a
    money-path caller that fires-and-forgets an announce can't be broken by it).
    """

    def __init__(
        self,
        store: Any,
        gossip: Any,
        *,
        cid_fn: Callable[[bytes], str] = receipt_set_cid,
        inline_leaf_hashes: bool = True,
    ):
        self._store = store
        self._gossip = gossip
        self._cid_fn = cid_fn
        self._inline_leaf_hashes = inline_leaf_hashes

    def _announcer_id(self) -> Optional[str]:
        """The producer's own gossip node_id (bound into the ad as announcer_id so an
        observer can attribute the ad to its origin). None if the gossip exposes no
        identity (then the ad is unattributable and observers will drop it fail-safe)."""
        try:
            return self._gossip.transport.identity.node_id
        except Exception:
            return None

    async def announce(self, batch_id: bytes) -> bool:
        """Publish ONE availability ad for ``batch_id``. Returns True iff an ad was
        published. Best-effort: never raises."""
        try:
            pb: Optional[PublishedBatch] = self._store.get(batch_id)
            if pb is None:
                logger.debug(
                    "SettlementBatchAnnouncer: batch %s not in store; no ad",
                    bytes(batch_id).hex(),
                )
                return False

            blob = serialize_receipt_set(pb)
            cid = self._cid_fn(blob)
            leaf_hashes = (
                [hash_leaf(batched_receipt_to_leaf(br)) for br in pb.receipts]
                if self._inline_leaf_hashes
                else None
            )
            ad = SettlementBatchAdvertisement(
                batch_id=pb.batch_id,
                provider_address=_sole_provider(pb),
                merkle_root=pb.merkle_root,
                cid=cid,
                leaf_hashes=leaf_hashes,
                commit_timestamp=pb.commit_timestamp,
            )
            data = ad.to_gossip_data()
            announcer_id = self._announcer_id()
            if announcer_id is not None:
                data["announcer_id"] = announcer_id
            await self._gossip.publish(GOSSIP_SETTLEMENT_BATCH_AVAILABLE, data)
            logger.debug(
                "SettlementBatchAnnouncer: announced batch %s (cid=%s)",
                pb.batch_id.hex(), cid,
            )
            return True
        except Exception as exc:
            # Best-effort: an announce failure must never break a caller (it carries
            # no funds — the actual availability is re-established on the next
            # announce, and the on-chain commit is unaffected).
            logger.warning(
                "SettlementBatchAnnouncer: announce of %s failed (%s); skipped",
                bytes(batch_id).hex() if batch_id else "?", exc,
            )
            return False


def _sole_provider(pb: PublishedBatch) -> str:
    """The provider address for a batch. A batch is one (requester, provider) pair
    (accumulator design), so all receipts share a provider_address; take the first
    (empty string for an empty set — an empty set never produces a real ad anyway)."""
    if pb.receipts:
        return pb.receipts[0].provider_address
    return ""


class SettlementBatchAdIndex:
    """OPT-IN observer-side component: index availability ads as UNTRUSTED POINTERS.

    On construction it subscribes to ``GOSSIP_SETTLEMENT_BATCH_AVAILABLE``. Each ad is
    authenticated (origin must bind to the embedded ``announcer_id``), parsed, deduped
    by ``batch_id`` (FIRST-SEEN WINS), and stored bounded by ``max_ads`` (evict oldest).
    A malformed ad or one whose origin does not authenticate is DROPPED (logged) without
    crashing and without evicting/corrupting an existing entry.

    The stored ad is a POINTER only — never authoritative. A consumer (Brick D) fetches
    the blob by ``cid`` and verifies it against the on-chain root before trusting it.
    """

    def __init__(self, gossip: Any, *, max_ads: int = _DEFAULT_MAX_ADS):
        if max_ads <= 0:
            raise ValueError(f"max_ads must be positive (got {max_ads})")
        self._gossip = gossip
        self._max_ads = max_ads
        # OrderedDict[batch_id_hex -> SettlementBatchAdvertisement], insertion-ordered
        # (oldest first) so popitem(last=False) evicts the oldest. First-seen wins:
        # a known batch_id is never re-inserted, so a forged re-ad can't overwrite it.
        self._ads: "OrderedDict[str, SettlementBatchAdvertisement]" = OrderedDict()
        gossip.subscribe(GOSSIP_SETTLEMENT_BATCH_AVAILABLE, self._on_ad)

    # ── public read API ──────────────────────────────────────────────

    def get(self, batch_id: bytes) -> Optional[SettlementBatchAdvertisement]:
        return self._ads.get(bytes(batch_id).hex())

    def all_ads(self) -> List[SettlementBatchAdvertisement]:
        """All indexed pointers, oldest-first."""
        return list(self._ads.values())

    # ── gossip handler ───────────────────────────────────────────────

    async def _on_ad(self, subtype: str, data: Dict[str, Any], origin: str) -> None:
        """Ingest one availability ad. ``origin`` is the gossip-AUTHENTICATED origin
        (sp934) — we admit the ad only if it binds to the embedded announcer_id. Never
        raises; a bad/unauthenticated ad is dropped without touching existing entries."""
        try:
            announcer_id = data.get("announcer_id") if isinstance(data, dict) else None
            # Fail-safe origin gate: the ad must claim an announcer and the gossip layer
            # must have authenticated the origin AS that announcer. A relayed/forged ad
            # whose attestation didn't verify falls back to the relaying sender (!=
            # announcer_id) and is dropped — it can't claim the precious dedup slot.
            if not announcer_id or origin != announcer_id:
                logger.debug(
                    "SettlementBatchAdIndex: dropping ad with unauthenticated origin "
                    "(origin=%s announcer_id=%s)", origin, announcer_id,
                )
                return

            ad = SettlementBatchAdvertisement.from_gossip_data(data)
        except Exception as exc:
            logger.debug("SettlementBatchAdIndex: dropping malformed ad (%s)", exc)
            return

        key = bytes(ad.batch_id).hex()
        # DEDUP — first-seen wins. A second ad for a known batch_id (even from another
        # authenticated origin) is IGNORED, so a forged-origin re-ad cannot overwrite
        # or evict the first-seen pointer.
        if key in self._ads:
            logger.debug(
                "SettlementBatchAdIndex: batch %s already indexed; ignoring re-ad",
                key,
            )
            return

        self._ads[key] = ad
        # BOUND — evict oldest beyond the cap.
        while len(self._ads) > self._max_ads:
            evicted_key, _ = self._ads.popitem(last=False)
            logger.debug(
                "SettlementBatchAdIndex: evicted oldest ad %s (over max_ads=%d)",
                evicted_key, self._max_ads,
            )
