"""
Content Index
=============

In-memory index of content available across the PRSM network.
Populated by GOSSIP_CONTENT_ADVERTISE messages — each node that
uploads or pins content broadcasts an advertisement, and every
receiving node upserts a record here.

Supports keyword search over filenames and metadata, and tracks
which nodes can serve each CID (providers).
"""

import asyncio
import base64
import hashlib
import json
import logging
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from prsm.node.gossip import (
    GOSSIP_CONTENT_ADVERTISE,
    GOSSIP_PROVENANCE_QUERY,
    GOSSIP_PROVENANCE_REGISTER,
    GOSSIP_PROVENANCE_RESPONSE,
    GossipProtocol,
)
from prsm.node.identity import verify_signature

logger = logging.getLogger(__name__)

MAX_INDEXED_CIDS = 10_000

# sp1004 — the GOSSIP_CONTENT_ADVERTISE lane is UNAUTHENTICATED (any peer can
# advertise any CID; sp934 authenticates the sender, not the claim). The
# in-memory royalty_rate weights off-chain multi-shard pool splits and
# size_bytes feeds the sp1002 gateway-fetch cap, so a non-finite / negative /
# absurd value entering the index is a real off-chain poisoning / DoS vector.
# These bounds reject absolute insanity on ingest; the residual within-bounds
# relative-skew + creator-binding gap is documented in
# docs/2026-06-04-content-data-plane-trust-anchors.md (Gap B).
DEFAULT_ROYALTY_RATE = 0.01
# Upper bound matches the on-chain ProvenanceRegistry max (9800 bps = 0.98),
# so no legitimately-registerable rate is rejected.
MAX_ADVERTISE_ROYALTY_RATE = 0.98


def _sane_royalty_rate(raw: Any, *, fallback: float) -> float:
    """Coerce an advertise-supplied royalty_rate to a safe finite value in
    [0, MAX_ADVERTISE_ROYALTY_RATE], or return ``fallback`` when it is
    non-numeric / non-finite / out of range."""
    try:
        rate = float(raw)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(rate) or rate < 0.0 or rate > MAX_ADVERTISE_ROYALTY_RATE:
        return fallback
    return rate


def _sane_size_bytes(raw: Any) -> int:
    """Coerce an advertise-supplied size_bytes to a non-negative int, or 0
    ("unknown") when it is non-integer / non-finite / negative."""
    try:
        if isinstance(raw, float) and not math.isfinite(raw):
            return 0
        n = int(raw)
    except (TypeError, ValueError, OverflowError):
        return 0
    return n if n >= 0 else 0


@dataclass
class ContentRecord:
    """A piece of content known to the network."""
    cid: str
    filename: str
    size_bytes: int
    content_hash: str
    creator_id: str
    providers: Set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    royalty_rate: float = 0.01
    parent_cids: List[str] = field(default_factory=list)
    embedding_id: Optional[str] = None
    near_duplicate_of: Optional[str] = None
    # Phase 1.2: canonical creator-bound provenance hash (0x-prefixed
    # hex). Set from the gossip advertisement so peers discovering
    # content remotely can still route royalties on-chain.
    provenance_hash: Optional[str] = None
    # Sprint 244 — creator's on-chain ETH address. Captured at
    # upload time via /content/upload?creator_eth_address=... and
    # propagated via gossip advertise (sprint TBD wires the wire
    # format). Used as destination for on-chain royalty when the
    # leg ships. None when not supplied (v1 backwards-compat).
    creator_eth_address: Optional[str] = None


class ContentIndex:
    """Network-wide content index built from gossip advertisements.

    Maintains an LRU-evicted map of CID → ContentRecord and a keyword
    index for search.  Thread-safety is not required because the event
    loop is single-threaded.
    """

    def __init__(
        self,
        gossip: GossipProtocol,
        max_indexed_cids: int = MAX_INDEXED_CIDS,
        ledger: Optional[Any] = None,
    ):
        self.gossip = gossip
        self.max_indexed_cids = max_indexed_cids
        self.ledger = ledger  # Optional LocalLedger for durable provenance storage
        # OrderedDict for LRU eviction — most-recently-touched at the end
        self._records: OrderedDict[str, ContentRecord] = OrderedDict()
        # keyword → set of CIDs that match
        self._keyword_index: Dict[str, Set[str]] = {}
        # Pending cross-node provenance lookups: cid → Future[dict]
        self._pending_provenance: Dict[str, asyncio.Future] = {}

    def start(self) -> None:
        """Subscribe to content and provenance gossip."""
        self.gossip.subscribe(GOSSIP_CONTENT_ADVERTISE, self._on_content_advertise)
        self.gossip.subscribe(GOSSIP_PROVENANCE_REGISTER, self._on_provenance_register)
        self.gossip.subscribe(GOSSIP_PROVENANCE_QUERY, self._on_provenance_query)
        self.gossip.subscribe(GOSSIP_PROVENANCE_RESPONSE, self._on_provenance_response)
        logger.info("Content index started — listening for advertisements and provenance")

    # ── Gossip handler ────────────────────────────────────────────

    async def _on_content_advertise(
        self, subtype: str, data: Dict[str, Any], origin: str
    ) -> None:
        """Upsert a content record from a gossip advertisement."""
        cid = data.get("cid", "")
        if not cid:
            return

        provider_id = data.get("provider_id", origin)

        if cid in self._records:
            # Existing record — add the new provider and backfill any
            # empty/default fields from the new advertisement. Never
            # clobber a populated value.
            record = self._records[cid]
            record.providers.add(provider_id)
            keyword_changed = self._backfill_record_from_advertise(
                record, data, origin
            )
            if keyword_changed:
                # A minimal ad created this record with empty filename
                # and/or empty metadata, so _index_keywords added
                # nothing on that path. Now that the backfill filled in
                # real values, re-run it so search() can find the CID
                # via the new filename tokens and metadata values. The
                # helper is additive (adds CIDs to keyword sets), so
                # this safely populates the new entries without needing
                # to prune stale ones — there are none.
                self._index_keywords(record)
            self._records.move_to_end(cid)
        else:
            # New record
            record = ContentRecord(
                cid=cid,
                filename=data.get("filename", ""),
                # sp1004 — sanitize unauthenticated advertise numerics.
                size_bytes=_sane_size_bytes(data.get("size_bytes", 0)),
                content_hash=data.get("content_hash", ""),
                # Phase 1.3 Task 3g pass-5: empty string signals
                # "unknown creator" so backfill can repair without
                # ambiguity. The previous origin-peer-id fallback was
                # a lie and broke legitimate self-hosting uploader
                # detection — ContentUploader publishes creator_id ==
                # provider_id == self.identity.node_id, so the
                # "creator_id in record.providers" heuristic fired on
                # every legitimate upload and silently overwrote the
                # original creator when a second full ad arrived.
                creator_id=data.get("creator_id", ""),
                providers={provider_id},
                created_at=data.get("created_at", time.time()),
                metadata=data.get("metadata", {}),
                # sp1004 — reject non-finite / negative / absurd rate.
                royalty_rate=_sane_royalty_rate(
                    data.get("royalty_rate", DEFAULT_ROYALTY_RATE),
                    fallback=DEFAULT_ROYALTY_RATE,
                ),
                parent_cids=data.get("parent_cids", []),
                embedding_id=data.get("embedding_id"),
                near_duplicate_of=data.get("near_duplicate_of"),
                provenance_hash=data.get("provenance_hash"),
                creator_eth_address=data.get("creator_eth_address"),
            )
            self._records[cid] = record
            self._index_keywords(record)
            self._evict_if_needed()

        logger.debug(f"Content index: {cid[:12]}... now has {len(self._records[cid].providers)} provider(s)")

    def _backfill_record_from_advertise(
        self,
        record: ContentRecord,
        data: Dict[str, Any],
        origin: str,
    ) -> bool:
        """Backfill empty/default fields on an existing ContentRecord
        from a new advertisement payload. Never clobbers populated
        values.

        "Empty" is field-specific:
         - strings: "" or None → backfill
         - dicts:   {} → backfill
         - lists:   [] → backfill
         - creator_id: "" or equal to origin (the fallback used when
           the advertise payload omits creator_id; see new-record path
           above at ``creator_id=data.get("creator_id", origin)``) →
           backfill when the new payload has a different, non-empty
           value.
         - royalty_rate: 0.01 → backfill when the new payload has a
           non-default, non-0.01 value (best-effort: we can't
           distinguish explicit 0.01 from default 0.01, so the first
           advertisement to set a non-default wins).
         - size_bytes: 0 → backfill when the new payload has a
           positive size. A real content record always has positive
           bytes, so 0 means "unknown / not in the incoming ad".
         - None for provenance_hash / embedding_id / near_duplicate_of
           → backfill

        Returns True if any keyword-affecting field (filename,
        metadata) was backfilled, so the caller knows whether to
        re-run _index_keywords().

        Phase 1.3 Task 3g — generalization of the Phase 1.2 Task 9
        provenance_hash-only backfill after codex pass 3 flagged the
        single-field version as P2. Without this, nodes that first
        learn about a CID from a minimal replica advertisement get
        stuck with creator_id = origin-peer-id, empty filename, empty
        parent_cids, and default royalty_rate forever — causing
        Task 3f's serve_on_behalf_of_replica to pay royalties to the
        wrong creator.

        Phase 1.3 Task 3g follow-up — codex pass 4 added the
        keyword-index re-index trigger (without it, a record created
        from a minimal ad and later filename-backfilled stayed
        permanently invisible to search()) and added size_bytes to
        the backfill list.
        """
        keyword_affecting_change = False

        # String fields that backfill on empty string or None.
        for field_name in ("filename", "content_hash"):
            current = getattr(record, field_name)
            if not current:
                incoming = data.get(field_name)
                if incoming:
                    setattr(record, field_name, incoming)
                    if field_name == "filename":
                        keyword_affecting_change = True

        # creator_id: backfill only when empty. The new-record path
        # now uses "" as the fallback for minimal ads (Phase 1.3
        # Task 3g pass-5 fix), so an empty string is the unambiguous
        # signal that the record came from a replica-like ad without
        # creator metadata. A populated creator_id is never
        # overwritten — Phase 1 uses the on-chain registry for the
        # authoritative creator when a provenance_hash is present,
        # and the local creator_id is informational plus used as the
        # fallback payment routing. The previous
        # "creator_id in record.providers" heuristic was a lie: in
        # production, ContentUploader publishes
        # creator_id == provider_id == self.identity.node_id, so the
        # heuristic fired on every legitimate self-hosting uploader
        # and silently overwrote the original creator when a second
        # full ad arrived from a different peer (e.g. two nodes
        # serving the same file, gossip relay re-advertise).
        if not record.creator_id:
            incoming_creator = data.get("creator_id")
            if incoming_creator:
                record.creator_id = incoming_creator

        # metadata dict: backfill when empty.
        if not record.metadata:
            incoming_metadata = data.get("metadata")
            if incoming_metadata:
                record.metadata = incoming_metadata
                keyword_affecting_change = True

        # parent_cids list: backfill when empty.
        if not record.parent_cids:
            incoming_parents = data.get("parent_cids")
            if incoming_parents:
                record.parent_cids = list(incoming_parents)

        # size_bytes: backfill 0 when a later ad carries a positive
        # size. A real content record always has positive bytes, so 0
        # means "unknown / not in the incoming ad". Not
        # keyword-affecting.
        if record.size_bytes == 0:
            incoming_size = _sane_size_bytes(data.get("size_bytes"))
            if incoming_size:  # non-zero (and now guaranteed non-negative int)
                record.size_bytes = incoming_size

        # royalty_rate: backfill default 0.01 when the new payload
        # carries a non-default value. Best-effort — we can't
        # distinguish an explicit 0.01 from a default 0.01, so if the
        # first advertisement legitimately set royalty_rate=0.01 a
        # later non-default ad will overwrite it. That is safer than
        # the alternative (replica minimal ads permanently pinning
        # royalty_rate to the 0.01 default).
        if record.royalty_rate == DEFAULT_ROYALTY_RATE:
            incoming_rate = data.get("royalty_rate")
            if incoming_rate is not None:
                # sp1004 — sanitize before clobbering the default. An
                # out-of-range / non-finite advertise leaves the default
                # intact rather than poisoning the rate.
                sane = _sane_royalty_rate(
                    incoming_rate, fallback=DEFAULT_ROYALTY_RATE,
                )
                if sane != DEFAULT_ROYALTY_RATE:
                    record.royalty_rate = sane

        # Optional fields that are None by default.
        for field_name in (
            "embedding_id",
            "near_duplicate_of",
            "provenance_hash",
            # sp995 (fix B) — a record first created from a minimal replica/
            # announce advertise (which omits creator_eth_address) had no way to
            # learn it when the uploader's full advertise arrived later, so the
            # §14 stake gate keyed on None → wrongly demoted a stake-eligible
            # HIGH creator. Backfilling it here repairs the record on the full ad
            # regardless of advertise ordering. (First-non-None-wins, like the
            # other optional fields — a later minimal ad can't clobber it.)
            "creator_eth_address",
        ):
            if getattr(record, field_name) is None:
                incoming = data.get(field_name)
                if incoming is not None:
                    setattr(record, field_name, incoming)

        return keyword_affecting_change

    async def _on_provenance_register(
        self, subtype: str, data: Dict[str, Any], origin: str
    ) -> None:
        """Persist a provenance registration to the local ledger.

        sp964: the broadcast PROVENANCE_REGISTER carries the original signed
        record, so we fully authenticate it (signature + pubkey↔creator binding)
        and enforce first-writer-wins before persisting — a peer can no longer
        forge or hijack the §14 first-creator-wins provenance record.
        """
        await self._verified_upsert(data, require_signature=True)

    @staticmethod
    def _provenance_authentic(record: Dict[str, Any]) -> bool:
        """True iff `record` carries a valid signature whose pubkey binds to the
        claimed creator_id. Mirrors gossip._authenticate_origin: node_id ==
        sha256(pubkey)[:32], and the ed25519 signature verifies over the canonical
        json of the record WITHOUT its `signature` field (exactly what
        ContentUploader signs). Returns False on any missing field / parse error."""
        creator_id = record.get("creator_id") or ""
        pubkey = record.get("creator_public_key") or ""
        sig = record.get("signature") or ""
        if not (creator_id and pubkey and sig):
            return False
        try:
            derived = hashlib.sha256(base64.b64decode(pubkey)).hexdigest()[:32]
        except Exception:
            return False
        if derived != creator_id:
            return False  # pubkey does not bind to the claimed creator_id
        try:
            signed = json.dumps(
                {k: v for k, v in record.items() if k != "signature"},
                sort_keys=True,
            ).encode()
        except Exception:
            return False
        return verify_signature(pubkey, signed, sig)

    async def _verified_upsert(
        self, record: Dict[str, Any], *, require_signature: bool
    ) -> None:
        """Guarded persistence for a gossip-received provenance record.

        - require_signature (REGISTER path): the original signed record is on the
          wire, so reject anything whose signature/creator-binding fails.
        - first-writer-wins (BOTH paths): a gossip record can NEVER change an
          existing cid's creator_id/creator_pubkey/royalty_rate. Same-creator
          re-registrations and metadata updates are allowed; new cids register
          normally. The RESPONSE path carries a lossy re-serialized stored row
          whose signature can't be reconstructed, so it relies on first-writer-
          wins only (full response-path crypto needs verbatim signed-form
          storage — a documented follow-on)."""
        if not self.ledger:
            return
        cid = record.get("cid")
        if not cid:
            return
        if require_signature and not self._provenance_authentic(record):
            logger.warning(
                "Rejecting provenance register for %s: signature/creator "
                "binding invalid (sp964)", str(cid)[:12],
            )
            return
        # First-writer-wins: never let a gossip record reassign an established
        # creator. Use the LEDGER's local-only lookup (NOT self.get_provenance,
        # which would broadcast a cross-node query and could recurse/hang).
        try:
            existing = await self.ledger.get_provenance(cid)
        except Exception:
            existing = None
        if (
            existing
            and existing.get("creator_id")
            and existing.get("creator_id") != record.get("creator_id")
        ):
            logger.warning(
                "Rejecting provenance for %s: first-writer-wins (existing "
                "creator %s != %s) (sp964)", str(cid)[:12],
                str(existing.get("creator_id"))[:12],
                str(record.get("creator_id"))[:12],
            )
            return
        try:
            await self.ledger.upsert_provenance(record)
        except Exception as exc:
            logger.warning(f"Failed to persist provenance for {str(cid)[:12]}: {exc}")

    async def _on_provenance_query(
        self, subtype: str, data: Dict[str, Any], origin: str
    ) -> None:
        """Respond to a cross-node provenance query if we have the record locally."""
        cid = data.get("cid", "")
        requester_id = data.get("requester_id", "")
        if not cid or not self.ledger:
            return
        try:
            # sp965 — prefer the VERBATIM signed record so the requester can
            # cryptographically verify authorship; fall back to the lossy typed
            # row only for pre-sp965 rows that have no verbatim form stored.
            record = await self.ledger.get_signed_provenance(cid)
            if record is None:
                record = await self.ledger.get_provenance(cid)
            if record:
                await self.gossip.publish(GOSSIP_PROVENANCE_RESPONSE, {
                    "cid": cid,
                    "for_requester": requester_id,
                    "provenance": record,
                })
                logger.debug(f"Answered provenance query for {cid[:12]}...")
        except Exception as exc:
            logger.warning(f"Error handling provenance query for {cid[:12]}: {exc}")

    async def _on_provenance_response(
        self, subtype: str, data: Dict[str, Any], origin: str
    ) -> None:
        """Handle a provenance response — persist it and resolve any pending query."""
        cid = data.get("cid", "")
        provenance = data.get("provenance", {})
        if not cid or not provenance:
            return
        # sp965 — the responder now re-serves the VERBATIM signed record, so the
        # response is cryptographically verifiable. Gate BOTH persistence AND the
        # caller-facing future on authenticity: a forged response is ignored
        # entirely (never cached, never handed to the get_provenance() caller —
        # the caller waits for an honest answer or times out to None). An
        # unverifiable record from a pre-sp965 (lossy) responder is likewise not
        # trusted; full interop is restored once peers upgrade.
        authentic = self._provenance_authentic(provenance)
        if not authentic:
            logger.debug(
                "Ignoring unverifiable provenance response for %s "
                "(no valid signature/creator binding)", str(cid)[:12],
            )
            return
        # Persist (first-writer-wins still applies — can't reassign a creator).
        if self.ledger:
            await self._verified_upsert(provenance, require_signature=True)
        # Resolve any pending async get_provenance() call with the verified record.
        future = self._pending_provenance.get(cid)
        if future and not future.done():
            future.set_result(provenance)

    # ── Public queries ────────────────────────────────────────────

    def lookup(self, cid: str) -> Optional[ContentRecord]:
        """Look up a content record by CID."""
        return self._records.get(cid)

    def search(self, query: str, limit: int = 20) -> List[ContentRecord]:
        """Keyword search over filenames and metadata values.

        Returns records whose filename or metadata contain *all* query
        words (AND semantics).  Results are ordered most-recent first.
        """
        words = self._tokenize(query)
        if not words:
            return []

        # Intersect CID sets for each keyword
        matching_cids: Optional[Set[str]] = None
        for word in words:
            cids = self._keyword_index.get(word, set())
            if matching_cids is None:
                matching_cids = set(cids)
            else:
                matching_cids &= cids

        if not matching_cids:
            return []

        # Collect records, most-recently-advertised first
        results: List[ContentRecord] = []
        for cid in reversed(self._records):
            if cid in matching_cids:
                results.append(self._records[cid])
                if len(results) >= limit:
                    break
        return results

    def get_providers(self, cid: str) -> Set[str]:
        """Return the set of node IDs that can serve this CID."""
        record = self._records.get(cid)
        return record.providers if record else set()

    async def get_provenance(
        self, cid: str, timeout: float = 5.0
    ) -> Optional[Dict[str, Any]]:
        """Return the provenance record for a CID.

        Resolution order:
        1. Local SQLite ledger (instant, survives restarts).
        2. Cross-node gossip query: broadcasts GOSSIP_PROVENANCE_QUERY and
           waits up to *timeout* seconds for a GOSSIP_PROVENANCE_RESPONSE.
           The response is persisted locally so subsequent calls are instant.

        Returns None if no record is found within the timeout.
        """
        # 1. Check local ledger first
        if self.ledger:
            try:
                record = await self.ledger.get_provenance(cid)
                if record:
                    return record
            except Exception as exc:
                logger.warning(f"Ledger provenance lookup failed for {cid[:12]}: {exc}")

        # 2. Broadcast a query and wait for a response from any peer
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None  # No event loop — can't do async query

        future: asyncio.Future = loop.create_future()
        self._pending_provenance[cid] = future
        try:
            await self.gossip.publish(GOSSIP_PROVENANCE_QUERY, {
                "cid": cid,
                "requester_id": "",  # origin is set automatically by gossip layer
            })
            return await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
        except asyncio.TimeoutError:
            logger.debug(f"Provenance query timed out for {cid[:12]}...")
            return None
        except Exception as exc:
            logger.warning(f"Provenance query failed for {cid[:12]}: {exc}")
            return None
        finally:
            self._pending_provenance.pop(cid, None)

    def get_stats(self) -> Dict[str, Any]:
        """Index statistics for the status endpoint."""
        unique_providers: Set[str] = set()
        for rec in self._records.values():
            unique_providers |= rec.providers
        return {
            "indexed_cids": len(self._records),
            "unique_providers": len(unique_providers),
            "keyword_entries": len(self._keyword_index),
        }

    # ── Internal helpers ──────────────────────────────────────────

    def _index_keywords(self, record: ContentRecord) -> None:
        """Add keywords from a record's filename and metadata."""
        text_parts = [record.filename]
        for v in record.metadata.values():
            if isinstance(v, str):
                text_parts.append(v)

        cid = record.cid
        for word in self._tokenize(" ".join(text_parts)):
            self._keyword_index.setdefault(word, set()).add(cid)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercase split, stripping common punctuation."""
        import re
        return [w for w in re.split(r"[\s_\-./\\]+", text.lower()) if len(w) >= 2]

    def _evict_if_needed(self) -> None:
        """Remove the oldest entries when the index exceeds the cap."""
        while len(self._records) > self.max_indexed_cids:
            evicted_cid, evicted_record = self._records.popitem(last=False)
            # Clean up keyword index references
            for word in self._tokenize(evicted_record.filename):
                kw_set = self._keyword_index.get(word)
                if kw_set:
                    kw_set.discard(evicted_cid)
                    if not kw_set:
                        del self._keyword_index[word]
