"""Sprint 1138 — settlement-receipt data plane BRICK D: the observer AUDIT ENGINE.

Bricks A/B/C built the retention + serve + cross-check + untrusted-pointer-index
plumbing (see docs/2026-06-16-settlement-receipt-data-plane-design.md):
  - A (sp1135) ``published_batch_store.PublishedBatch``: the retained ordered receipt
    set for a committed batch (batch_id, merkle_root, commit_timestamp, receipts).
  - B (sp1136) ``receipt_set_blob.verify_receipt_set_against_root``: THE observer trust
    gate — recompute the merkle root from a (fetched, UNTRUSTED) receipt set via the
    EXACT producer pipeline and compare it to the trusted on-chain ``merkleRoot``.
  - C (sp1137) ``settlement_gossip``: the untrusted-pointer ad index.

Brick D is the PURE/injectable OFFLINE audit engine that turns observed, chain-anchored
batch metadata + fetched receipt sets into ACTIONABLE dispute findings — and NEVER
broadcasts, signs, or slashes. The network loop that drives enumerate→select→fetch→cache
is a separate brick (Brick E); Brick D is fully offline-testable with injected data.

THREE PARTS:

  1. SELECTORS (``AuditSelector`` protocol). The audit POLICY is two incentive-aligned
     selectors that need NO cooperation from the suspect:
       - ``OwnStakeSelector(my_address)`` — audit batches whose ``requester_address``
         is me (my stake is what gets settled, so I want to catch fraud against it);
       - ``ConsensusGroupSelector(my_group_ids)`` — audit batches whose
         ``consensus_group_id`` is a group I co-executed (I have the context to judge).
     The interface is pluggable; more selectors can be added later.

  2. ``VerifiedBatchCache`` — THE TRUST GATE (design §3 crux). NO fetched receipt set may
     reach ``detect_double_spends`` / the assemblers unless it FIRST passed
     ``verify_receipt_set_against_root`` against the trusted on-chain ``merkle_root``.
     ``ingest`` verifies FIRST: on pass it caches a ``CommittedBatch`` record (leaf hashes
     recomputed from the fetched receipts) AND keeps the order-preserved receipts for the
     assembler; on FAIL it returns False and caches NOTHING. This is exactly what makes
     untrusted observed data safe to feed into the slashing-path detectors.

  3. ``SettlementAuditEngine`` — runs the two detectors over the VERIFIED cache:
       - ``scan_double_spends``: ``detect_double_spends`` over the verified
         ``CommittedBatch`` records; for each finding, optionally assemble the
         DOUBLE_SPEND challenge from the cached receipts + conflicting leaf hashes and
         DRY-RUN it (read-only) to attach ``dry_run_ok``.
       - ``scan_invalid_signatures``: for every verified batch, every receipt index, try
         ``assemble_invalid_signature_challenge`` — which FAIL-FASTS (raises ValueError
         with "VERIFIES") when the shard signature is GOOD. A returned challenge therefore
         == an actionable bad-shard-signature; the "VERIFIES" ValueError == clean (skip).
         Optionally dry-run each assembled challenge.
     Both scans are FAIL-CLOSED per-item: one bad batch/receipt is skipped (or recorded),
     never aborts the scan, never raises out.

HARD CONSTRAINTS honoured: PURE/injectable (the dry-run client + the selector criteria are
injected). The engine NEVER broadcasts/signs/slashes — it only DRY-RUNS (read-only
eth_call) and surfaces actionable challenges; the user-gated ``submit`` stays separate.
No money-path change. The §7 InferenceReceipt verify path (ChallengeWatcher) is OUT OF
SCOPE — the InferenceReceipt is not retained; auditing it is a follow-on that first needs
InferenceReceipt retention (analogous to Brick A for shard receipts).
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.challenge_assembler import assemble_invalid_signature_challenge
from prsm.settlement.client import CommittedBatch
from prsm.settlement.double_spend_assembler import assemble_double_spend_challenge
from prsm.settlement.double_spend_detector import (
    DoubleSpendFinding,
    detect_double_spends,
)
from prsm.settlement.merkle import batched_receipt_to_leaf, hash_leaf
from prsm.settlement.published_batch_store import PublishedBatch
from prsm.settlement.receipt_set_blob import verify_receipt_set_against_root

logger = logging.getLogger(__name__)

# The all-zero "no consensus group" id (accumulator.BatchedReceipt default). A batch with
# this group id is NOT a consensus receipt, so ConsensusGroupSelector must ignore it.
_ZERO_GROUP_ID = b"\x00" * 32


# ── on-chain-anchored observed metadata ──────────────────────────────


@dataclass(frozen=True)
class ObservedBatch:
    """The on-chain-anchored metadata for a committed batch an observer has SEEN.

    Every field here is READ FROM THE CHAIN (the BatchSettlementRegistry ``Batch``
    struct: requester + consensus_group_id + provider + merkleRoot) — so ``merkle_root``
    is the TRUSTED root the ``VerifiedBatchCache`` cross-checks a fetched receipt set
    against. ``cid`` is the (untrusted) Brick-C pointer to where the receipt-set blob can
    be fetched; it is metadata only and is never trusted for content."""

    batch_id: bytes
    provider_address: str
    requester_address: str
    merkle_root: bytes
    consensus_group_id: bytes = _ZERO_GROUP_ID
    cid: Optional[str] = None


# ── selectors ─────────────────────────────────────────────────────────


class AuditSelector(Protocol):
    """Pluggable audit-policy interface: pick, from a stream of observed batches, the
    ones worth fetching + auditing. Implementations MUST be pure (no network/chain) — the
    selection criteria are injected at construction."""

    def select(self, observed: Iterable[ObservedBatch]) -> List[ObservedBatch]:
        ...


def _eq_addr(a: str, b: str) -> bool:
    """Case-insensitive eth-address compare (addresses are 0x-hex; checksum casing must
    not cause a false miss)."""
    return (a or "").lower() == (b or "").lower()


class OwnStakeSelector:
    """Keep batches whose ``requester_address`` == my address (the batches that settle
    MY stake — incentive-aligned, needs no cooperation from the suspect)."""

    def __init__(self, my_address: str):
        if not my_address:
            raise ValueError("my_address is required")
        self._my_address = my_address

    def select(self, observed: Iterable[ObservedBatch]) -> List[ObservedBatch]:
        return [b for b in observed if _eq_addr(b.requester_address, self._my_address)]


class ConsensusGroupSelector:
    """Keep batches whose ``consensus_group_id`` is one of my groups (a group I
    co-executed — I have the execution context to judge it). Ignores the all-zero
    "no group" id (a non-consensus receipt is never selected by group)."""

    def __init__(self, my_group_ids: "set[bytes]"):
        # Normalize to bytes; drop the zero-group id if it was passed (never selectable).
        self._my_group_ids = {
            bytes(g) for g in (my_group_ids or set()) if bytes(g) != _ZERO_GROUP_ID
        }

    def select(self, observed: Iterable[ObservedBatch]) -> List[ObservedBatch]:
        out: List[ObservedBatch] = []
        for b in observed:
            gid = bytes(b.consensus_group_id)
            if gid == _ZERO_GROUP_ID:
                continue
            if gid in self._my_group_ids:
                out.append(b)
        return out


# ── the trust-gated cache ─────────────────────────────────────────────


class VerifiedBatchCache:
    """Bounded cache of batches whose fetched receipt set PASSED the on-chain-root
    cross-check. THE TRUST GATE: ``ingest`` verifies a fetched ``PublishedBatch`` against
    the trusted on-chain ``merkle_root`` BEFORE caching anything — a set that fails is
    rejected and caches NOTHING, so nothing forged/altered/reordered can ever reach the
    slashing-path detectors.

    For a verified batch it caches BOTH:
      - a ``CommittedBatch`` record (the shape ``detect_double_spends`` consumes:
        batch_id, provider_address, merkle_root, leaf_hashes recomputed from the fetched
        receipts in order);
      - the order-preserved ``BatchedReceipt`` list (the preimages the
        INVALID_SIGNATURE / DOUBLE_SPEND assemblers need).

    Bounded by ``max_batches`` (insertion-ordered; evict OLDEST on overflow)."""

    def __init__(self, *, max_batches: int = 10_000):
        if max_batches <= 0:
            raise ValueError(f"max_batches must be positive (got {max_batches})")
        self._max_batches = max_batches
        # batch_id_hex -> (CommittedBatch, ordered receipts). Insertion order == eviction
        # order (oldest first), so popitem(last=False) drops the oldest.
        self._entries: "OrderedDict[str, Tuple[CommittedBatch, List[BatchedReceipt]]]" = (
            OrderedDict()
        )

    def ingest(self, observed: ObservedBatch, fetched: PublishedBatch) -> bool:
        """VERIFY FIRST, then cache. Recompute the merkle root from ``fetched.receipts``
        via the producer pipeline and compare to the TRUSTED ``observed.merkle_root``. On
        pass: build + cache the ``CommittedBatch`` record + keep the receipts, return True.
        On FAIL (forged / altered / reordered / wrong-root): cache NOTHING, return False.

        Never raises on a bad fetched set — a verify error is treated as a rejection
        (fail-closed) so a single malformed fetch can't break the ingest loop."""
        try:
            ok = verify_receipt_set_against_root(fetched, observed.merkle_root)
        except Exception as exc:  # noqa: BLE001 — a bad fetched set is a rejection, not a crash
            logger.warning(
                "VerifiedBatchCache: cross-check raised for batch %s (%s); rejecting",
                bytes(observed.batch_id).hex(), exc,
            )
            return False
        if not ok:
            logger.debug(
                "VerifiedBatchCache: REJECTED batch %s — recomputed root != on-chain root",
                bytes(observed.batch_id).hex(),
            )
            return False

        receipts = list(fetched.receipts)  # order-preserved copy (== leaf order)
        leaf_hashes = tuple(
            hash_leaf(batched_receipt_to_leaf(br)) for br in receipts
        )
        committed = CommittedBatch(
            batch_id=bytes(observed.batch_id),
            tx_hash="",  # not known to an observer; not needed by the detectors
            provider_address=observed.provider_address,
            requester_address=observed.requester_address,
            merkle_root=bytes(observed.merkle_root),
            receipt_count=len(receipts),
            total_value_ftns=sum(int(br.value_ftns) for br in receipts),
            commit_timestamp=int(fetched.commit_timestamp),
            leaf_hashes=leaf_hashes,
            trigger_reason=None,  # observer-side record; trigger reason is producer-only
        )
        key = committed.batch_id.hex()
        if key in self._entries:
            del self._entries[key]
        self._entries[key] = (committed, receipts)
        while len(self._entries) > self._max_batches:
            evicted, _ = self._entries.popitem(last=False)
            logger.debug(
                "VerifiedBatchCache: evicted oldest batch %s (over max_batches=%d)",
                evicted, self._max_batches,
            )
        return True

    def verified_batches(self) -> List[CommittedBatch]:
        """All cached (verified) batches as ``CommittedBatch`` records (for the detector),
        oldest-first."""
        return [cb for cb, _ in self._entries.values()]

    def receipts_for(self, batch_id: bytes) -> Optional[List[BatchedReceipt]]:
        """The order-preserved receipts for a verified batch (for the assembler), or
        None if not cached."""
        entry = self._entries.get(bytes(batch_id).hex())
        return list(entry[1]) if entry is not None else None

    def leaf_hashes_for(self, batch_id: bytes) -> Optional[Tuple[bytes, ...]]:
        """The order-preserved leaf hashes for a verified batch (for the DOUBLE_SPEND
        conflicting-batch proof), or None if not cached."""
        entry = self._entries.get(bytes(batch_id).hex())
        return entry[0].leaf_hashes if entry is not None else None


# ── findings ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DoubleSpendAuditFinding:
    """An actionable double-spend finding from the verified cache. Wraps the pure
    ``DoubleSpendFinding`` with the optional read-only dry-run verdict
    (``dry_run_ok``: True/False if a dry-run client was injected, else None)."""

    finding: DoubleSpendFinding
    dry_run_ok: Optional[bool] = None
    dry_run_reason: Optional[str] = None


@dataclass(frozen=True)
class InvalidSignatureAuditFinding:
    """An actionable bad-shard-signature finding: the receipt at ``receipt_index`` in
    ``batch_id`` has a committed signature that does NOT verify (the assembler returned a
    challenge rather than fail-fasting). ``dry_run_ok`` is the optional read-only verdict."""

    batch_id: bytes
    receipt_index: int
    dry_run_ok: Optional[bool] = None
    dry_run_reason: Optional[str] = None


# ── the engine ────────────────────────────────────────────────────────


class SettlementAuditEngine:
    """Offline audit engine over a ``VerifiedBatchCache``. Runs the double-spend +
    invalid-signature detectors and optionally read-only DRY-RUNS each actionable finding
    against current chain state. NEVER broadcasts/signs/slashes."""

    def __init__(self, cache: VerifiedBatchCache, *, dry_run_client: Any = None):
        self._cache = cache
        self._dry_run_client = dry_run_client

    def _dry_run(self, challenge) -> Tuple[Optional[bool], Optional[str]]:
        """Read-only pre-flight via the injected client's ``dry_run`` (eth_call — no tx,
        no gas, no state change). Returns (ok, reason). No client → (None, None). A
        raising dry-run is swallowed (the finding is still actionable; the dry-run is just
        advisory)."""
        if self._dry_run_client is None:
            return None, None
        try:
            result = self._dry_run_client.dry_run(challenge)
        except Exception as exc:  # noqa: BLE001 — dry-run is advisory; never abort a scan
            logger.warning("dry-run raised (%s); leaving verdict unknown", exc)
            return None, f"dry-run error: {exc}"
        return bool(result.would_succeed), getattr(result, "revert_reason", None)

    def scan_double_spends(self) -> List[DoubleSpendAuditFinding]:
        """Detect double-spent leaves across the VERIFIED cache. For each finding, if a
        dry-run client is set, assemble the DOUBLE_SPEND challenge from the cached
        receipts + the conflicting batch's leaf hashes and dry-run it (read-only).
        FAIL-CLOSED: any error returns the findings gathered so far without raising; a
        per-finding assembly/dry-run error leaves ``dry_run_ok`` unknown but still surfaces
        the finding. NEVER broadcasts."""
        try:
            raw_findings = detect_double_spends(self._cache.verified_batches())
        except Exception as exc:  # noqa: BLE001 — fail-closed: a detector blowup is not actionable
            logger.error("scan_double_spends: detector raised (%s); returning []", exc)
            return []

        out: List[DoubleSpendAuditFinding] = []
        for finding in raw_findings:
            dry_ok: Optional[bool] = None
            dry_reason: Optional[str] = None
            if self._dry_run_client is not None:
                challenge = self._try_assemble_double_spend(finding)
                if challenge is not None:
                    dry_ok, dry_reason = self._dry_run(challenge)
            out.append(DoubleSpendAuditFinding(
                finding=finding, dry_run_ok=dry_ok, dry_run_reason=dry_reason))
        return out

    def _try_assemble_double_spend(self, finding: DoubleSpendFinding):
        """Assemble a DOUBLE_SPEND challenge for two DISTINCT-batch occurrences of the
        finding's leaf, sourcing the target's receipts + the conflicting batch's leaf
        hashes from the verified cache. Returns the challenge, or None if it can't be
        assembled (no distinct-batch pair, missing cache data, or a fail-fast). Never
        raises out."""
        try:
            first = finding.occurrences[0]
            conflicting = None
            for occ in finding.occurrences[1:]:
                if bytes(occ.batch_id) != bytes(first.batch_id):
                    conflicting = occ
                    break
            if conflicting is None:
                return None  # purely intra-batch — not expressible inter-batch
            target_receipts = self._cache.receipts_for(first.batch_id)
            conflicting_leaves = self._cache.leaf_hashes_for(conflicting.batch_id)
            if target_receipts is None or conflicting_leaves is None:
                return None
            return assemble_double_spend_challenge(
                target_batch_id=bytes(first.batch_id),
                target_batch_receipts=target_receipts,
                target_index=first.leaf_index,
                conflicting_batch_id=bytes(conflicting.batch_id),
                conflicting_leaf_hashes=list(conflicting_leaves),
                conflicting_index=conflicting.leaf_index,
            )
        except Exception as exc:  # noqa: BLE001 — assembly fail-fast/error → no dry-run, still surfaced
            logger.debug("double-spend assembly skipped (%s)", exc)
            return None

    def scan_invalid_signatures(self) -> List[InvalidSignatureAuditFinding]:
        """For every verified batch, every receipt index, try the INVALID_SIGNATURE
        assembler — which FAIL-FASTS (raises ValueError "...VERIFIES...") when the shard
        signature is GOOD. A returned challenge == an actionable bad signature at that
        index; the "VERIFIES" ValueError == clean (skip). If a dry-run client is set, each
        assembled challenge is dry-run (read-only). FAIL-CLOSED per-item: a single
        batch/receipt that raises a NON-fail-fast error is skipped, never aborts the scan.
        NEVER broadcasts."""
        out: List[InvalidSignatureAuditFinding] = []
        for cb in self._cache.verified_batches():
            receipts = self._cache.receipts_for(cb.batch_id)
            if not receipts:
                continue
            for idx in range(len(receipts)):
                try:
                    challenge = assemble_invalid_signature_challenge(
                        batch_id=cb.batch_id,
                        batch_receipts=receipts,
                        target_index=idx,
                    )
                except ValueError:
                    # Fail-fast (signature VERIFIES, or index issue) == clean / not
                    # actionable for this index. Skip.
                    continue
                except Exception as exc:  # noqa: BLE001 — fail-closed: skip a flaky item, keep scanning
                    logger.warning(
                        "scan_invalid_signatures: assembler raised for batch %s idx %d "
                        "(%s); skipping", cb.batch_id.hex(), idx, exc,
                    )
                    continue
                dry_ok, dry_reason = self._dry_run(challenge)
                out.append(InvalidSignatureAuditFinding(
                    batch_id=cb.batch_id, receipt_index=idx,
                    dry_run_ok=dry_ok, dry_run_reason=dry_reason))
        return out
