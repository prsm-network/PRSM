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
from base64 import b64decode
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from eth_utils import keccak

from prsm.marketplace.consensus_submitter import ChallengeAttempt, ReceiptLeafFields
from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.challenge_assembler import assemble_invalid_signature_challenge
from prsm.settlement.challenge_verifier import (
    ChallengeReason,
    verify_inference_receipt_for_challenge,
)
from prsm.settlement.client import CommittedBatch
from prsm.settlement.consensus_mismatch_detector import (
    ConsensusMismatchFinding,
    DetectorBatch,
    detect_consensus_mismatches,
)
from prsm.settlement.double_spend_assembler import assemble_double_spend_challenge
from prsm.settlement.double_spend_detector import (
    DoubleSpendFinding,
    detect_double_spends,
)
from prsm.settlement.expired_assembler import assemble_expired_challenge
from prsm.settlement.expired_receipt_detector import (
    ExpiredReceiptFinding,
    detect_expired_receipts,
)
from prsm.settlement.inference_receipt_store import RetainedInferenceReceipt
from prsm.settlement.merkle import (
    batched_receipt_to_leaf,
    build_merkle_proof,
    hash_leaf,
)
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
    # sp1148 — the chain-anchored ``lookbackWindowSecondsAtCommit`` (Batch struct idx 10) the
    # EXPIRED detector time-checks against. ADDITIVE default (0 == unknown) so every existing
    # ObservedBatch constructor still works; 0/unknown is treated FAIL-CLOSED by the detector
    # (never flagged), so an old call site that doesn't supply it can never produce a false
    # EXPIRED finding.
    lookback_window_seconds: int = 0


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
        # batch_id_hex -> {leaf_hash bytes -> RetainedInferenceReceipt}. The OPTIONAL §7
        # (sp1143) section retained ONLY for leaves of a ROOT-VERIFIED batch. Evicted in
        # lockstep with ``_entries`` (same key), so a §7 map never outlives its batch.
        self._inference_receipts: "OrderedDict[str, Dict[bytes, RetainedInferenceReceipt]]" = (
            OrderedDict()
        )
        # sp1145 — batch_id_hex -> observed.consensus_group_id (32 bytes). ADDITIVE: the
        # ``CommittedBatch`` record the double-spend detector consumes carries no group_id,
        # so the CONSENSUS_MISMATCH detector needs this retained separately to group co-group
        # batches by (group_id, job, shard) across providers. Defaults to the zero-group id
        # when an observed batch carries none, so a zero-group batch is never co-grouped.
        # Evicted in lockstep with ``_entries`` (same key).
        self._consensus_group_ids: "OrderedDict[str, bytes]" = OrderedDict()
        # sp1148 — batch_id_hex -> observed.lookback_window_seconds (the chain-anchored
        # ``lookbackWindowSecondsAtCommit``). ADDITIVE, mirrors ``_consensus_group_ids``: the
        # ``CommittedBatch`` record the detectors consume carries no lookback, so the EXPIRED
        # detector needs this retained separately to time-check each leaf against the EXACT
        # per-batch on-chain window. Defaults to 0 (unknown) when an observed batch carries
        # none — and 0 is FAIL-CLOSED in the detector (never flagged). Evicted in lockstep.
        self._lookback_window_seconds: "OrderedDict[str, int]" = OrderedDict()

    def ingest(
        self,
        observed: ObservedBatch,
        fetched: PublishedBatch,
        inference_receipts: Optional[Dict[bytes, RetainedInferenceReceipt]] = None,
    ) -> bool:
        """VERIFY FIRST, then cache. Recompute the merkle root from ``fetched.receipts``
        via the producer pipeline and compare to the TRUSTED ``observed.merkle_root``. On
        pass: build + cache the ``CommittedBatch`` record + keep the receipts, return True.
        On FAIL (forged / altered / reordered / wrong-root): cache NOTHING, return False.

        ``inference_receipts`` (sp1143, OPTIONAL/back-compat — None leaves behaviour
        UNCHANGED) is the §7 map {leaf_hash bytes -> RetainedInferenceReceipt} that rode in
        the SAME blob (Brick-2). On a VERIFIED ingest it is retained, but ONLY for the
        leaf_hashes that are actually in THIS batch's leaf set — a §7 entry keyed by an
        unknown leaf is dropped (it cannot be bound to a committed leaf). The §7 map is
        ONLY retained AFTER the root cross-check passes, so §7 receipts reach detection
        only behind the trust gate.

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
            self._inference_receipts.pop(key, None)
            self._consensus_group_ids.pop(key, None)
            self._lookback_window_seconds.pop(key, None)
        self._entries[key] = (committed, receipts)
        # sp1145 — retain the chain-anchored consensus_group_id for co-group detection.
        self._consensus_group_ids[key] = bytes(observed.consensus_group_id)
        # sp1148 — retain the chain-anchored per-batch lookback window for EXPIRED detection.
        self._lookback_window_seconds[key] = int(
            getattr(observed, "lookback_window_seconds", 0) or 0
        )

        # Retain the §7 map ONLY for leaves that are actually in this verified batch's
        # leaf set (an entry keyed by an unknown leaf can never bind to a committed leaf,
        # so it is dropped here). Always set a (possibly empty) map so the eviction key set
        # stays in lockstep with ``_entries``.
        batch_leaf_set = set(leaf_hashes)
        retained_s7: Dict[bytes, RetainedInferenceReceipt] = {}
        if inference_receipts:
            for lh, rec in inference_receipts.items():
                lh_b = bytes(lh)
                if lh_b in batch_leaf_set:
                    retained_s7[lh_b] = rec
        self._inference_receipts[key] = retained_s7

        while len(self._entries) > self._max_batches:
            evicted, _ = self._entries.popitem(last=False)
            self._inference_receipts.pop(evicted, None)
            self._consensus_group_ids.pop(evicted, None)
            self._lookback_window_seconds.pop(evicted, None)
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

    def consensus_group_id_for(self, batch_id: bytes) -> Optional[bytes]:
        """The chain-anchored ``consensus_group_id`` for a verified batch (sp1145), or
        None if the batch is not cached. Used by ``scan_consensus_mismatches`` to group
        co-group batches across providers."""
        return self._consensus_group_ids.get(bytes(batch_id).hex())

    def lookback_window_seconds_for(self, batch_id: bytes) -> Optional[int]:
        """The chain-anchored ``lookbackWindowSecondsAtCommit`` for a verified batch
        (sp1148), or None if the batch is not cached. Used by ``scan_expired_receipts`` to
        time-check each leaf against the EXACT per-batch on-chain window."""
        return self._lookback_window_seconds.get(bytes(batch_id).hex())

    def inference_receipts_for(
        self, batch_id: bytes,
    ) -> Dict[bytes, RetainedInferenceReceipt]:
        """The retained §7 receipts ({leaf_hash bytes -> RetainedInferenceReceipt}) for a
        verified batch, or an empty dict if the batch is not cached / carried no §7
        section. (sp1143 — returns a copy so a caller can't mutate the cache.)"""
        return dict(self._inference_receipts.get(bytes(batch_id).hex(), {}))


# ── EXPIRED detector input (sp1148) ───────────────────────────────────


@dataclass(frozen=True)
class _ExpiredDetectorBatch:
    """One root-verified batch fed to ``detect_expired_receipts``: order-preserved receipts +
    the chain-anchored per-batch ``lookback_window_seconds`` (0/unknown -> fail-closed)."""

    batch_id: bytes
    receipts: List[BatchedReceipt]
    lookback_window_seconds: int


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


@dataclass(frozen=True)
class InferenceReceiptAuditFinding:
    """A §7 compute-integrity audit record (sp1143) for ONE retained §7 InferenceReceipt
    matched to a verified-batch leaf.

    ``bound`` — whether the §7 receipt's ``settler_public_key_b64`` BOUND to the
    root-verified leaf (``keccak(b64decode(pubkey)) == leaf.provider_pubkey_hash``, the
    same hash merkle builds). An UNBOUND record (``bound=False``, ``proven=False``,
    ``reason=None``) is NOT trusted: its verdict is never run/surfaced (the ATTACK-A
    defence — a producer cannot mask/fabricate fraud via a substituted key).

    For a BOUND receipt, one finding is emitted per CHALLENGE GROUND the §7 verifier
    checked; ``proven`` is True iff that ground was cryptographically demonstrated, and
    ``on_chain_reason`` carries the BatchSettlementRegistry ReasonCode when the ground is
    on-chain-actionable (INVALID_SETTLER_SIGNATURE -> 1; activation-chain grounds None).
    ``dry_run_ok`` is the optional read-only verdict for an on-chain-actionable ground."""

    batch_id: bytes
    leaf_hash: bytes
    bound: bool
    proven: bool = False
    reason: Optional[ChallengeReason] = None
    on_chain_reason: Optional[int] = None
    detail: Optional[str] = None
    dry_run_ok: Optional[bool] = None
    dry_run_reason: Optional[str] = None


@dataclass(frozen=True)
class ConsensusMismatchAuditFinding:
    """An actionable co-group disagreement (sp1145) from the verified cache. Wraps the pure
    ``ConsensusMismatchFinding`` with the assembled (never-broadcast) ``ChallengeAttempt``
    (reusing ``marketplace.consensus_submitter``: minority leaf+proof from the challenged
    batch, majority leaf+proof from the conflicting batch) and the optional read-only dry-run
    verdict (``dry_run_ok``: True/False if a dry-run client was injected AND assembly
    succeeded, else None)."""

    finding: ConsensusMismatchFinding
    attempt: Optional[ChallengeAttempt] = None
    dry_run_ok: Optional[bool] = None
    dry_run_reason: Optional[str] = None


@dataclass(frozen=True)
class ExpiredReceiptAuditFinding:
    """An actionable EXPIRED finding (sp1148 — the 5th + FINAL on-chain reason) from the
    verified cache. Wraps the pure ``ExpiredReceiptFinding`` with the assembled
    (never-broadcast) ``ExpiredChallenge`` and the optional read-only dry-run verdict
    (``dry_run_ok``: True/False if a dry-run client was injected AND assembly succeeded, else
    None). EXPIRED is hygiene, not malice — this never slashes."""

    finding: ExpiredReceiptFinding
    challenge: Any = None
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

    def scan_consensus_mismatches(self) -> List[ConsensusMismatchAuditFinding]:
        """Detect co-group provider DISAGREEMENT across the VERIFIED cache (sp1145 — the
        3rd on-chain-actionable fraud class). Builds the detector input from ONLY the
        root-verified batches (their retained ``consensus_group_id`` + provider + receipts),
        runs ``detect_consensus_mismatches`` (which matches the on-chain
        ``_handleConsensusMismatch`` reject-set EXACTLY), and for each finding assembles a
        ``ChallengeAttempt`` (reusing ``marketplace.consensus_submitter``
        ``ReceiptLeafFields`` + merkle proofs from the verified batches: minority leaf+proof
        from the challenged batch, majority leaf+proof from the conflicting batch). If a
        dry-run client is injected AND assembly succeeded, dry-run it (read-only) and attach
        ``dry_run_ok``.

        FAIL-CLOSED: a detector blowup returns ``[]`` (a non-actionable scan); a per-finding
        assembly/dry-run error leaves ``attempt``/``dry_run_ok`` unknown but STILL surfaces
        the finding so the (now manual-review) disagreement is not silently dropped. NEVER
        calls ``submit_one``/broadcast — assembly + optional read-only dry-run only; the
        actual broadcast stays the separate user-gated submitter/queue path."""
        try:
            detector_batches = self._build_consensus_detector_input()
            raw_findings = detect_consensus_mismatches(detector_batches)
        except Exception as exc:  # noqa: BLE001 — fail-closed: a build/detector blowup is not actionable
            logger.error(
                "scan_consensus_mismatches: build/detector raised (%s); returning []", exc
            )
            return []

        out: List[ConsensusMismatchAuditFinding] = []
        for finding in raw_findings:
            attempt = self._try_assemble_consensus_mismatch(finding)
            dry_ok: Optional[bool] = None
            dry_reason: Optional[str] = None
            if attempt is not None and self._dry_run_client is not None:
                dry_ok, dry_reason = self._dry_run(attempt)
            out.append(ConsensusMismatchAuditFinding(
                finding=finding, attempt=attempt,
                dry_run_ok=dry_ok, dry_run_reason=dry_reason))
        return out

    def _build_consensus_detector_input(self) -> List[DetectorBatch]:
        """Build the ``detect_consensus_mismatches`` input from the VERIFIED cache: one
        ``DetectorBatch`` per root-verified batch carrying its retained
        ``consensus_group_id`` (zero-group when none) + provider + order-preserved receipts.
        A batch with missing receipts is skipped (it can't carry a leaf to compare)."""
        batches: List[DetectorBatch] = []
        for cb in self._cache.verified_batches():
            receipts = self._cache.receipts_for(cb.batch_id)
            if not receipts:
                continue
            group_id = self._cache.consensus_group_id_for(cb.batch_id)
            if group_id is None:
                group_id = b"\x00" * 32
            batches.append(DetectorBatch(
                batch_id=bytes(cb.batch_id),
                provider_address=cb.provider_address,
                consensus_group_id=bytes(group_id),
                receipts=receipts,
            ))
        return batches

    def _try_assemble_consensus_mismatch(
        self, finding: ConsensusMismatchFinding,
    ) -> Optional[ChallengeAttempt]:
        """Assemble the CONSENSUS_MISMATCH ``ChallengeAttempt`` for one finding, reusing
        ``marketplace.consensus_submitter`` ``ReceiptLeafFields`` + merkle proofs from the
        verified batches. The minority leaf+proof come from the CHALLENGED (minority) batch;
        the majority leaf+proof come from the CONFLICTING (majority) batch. Returns the
        attempt, or None if it can't be assembled (missing cache data or an encoding
        fail-fast). Never raises out — assembly failure leaves the finding surfaced without
        a dry-run verdict."""
        # sp1170 money-safety: an AMBIGUOUS finding (no strict majority — a tie/split where the
        # honest side is undeterminable off-chain) is surfaced for MANUAL review only. NEVER
        # auto-assemble a slash for it (the on-chain check is symmetric — it would slash
        # whichever side is challenged). Leave attempt=None.
        if getattr(finding, "ambiguous", False):
            return None
        try:
            minority_receipts = self._cache.receipts_for(finding.minority_batch_id)
            majority_receipts = self._cache.receipts_for(finding.majority_batch_id)
            minority_leaves = self._cache.leaf_hashes_for(finding.minority_batch_id)
            majority_leaves = self._cache.leaf_hashes_for(finding.majority_batch_id)
            if (minority_receipts is None or majority_receipts is None
                    or minority_leaves is None or majority_leaves is None):
                return None

            min_idx = finding.minority_leaf_index
            maj_idx = finding.majority_leaf_index
            minority_leaf = ReceiptLeafFields.from_python_receipt(
                minority_receipts[min_idx].receipt,
                value_ftns_wei=int(minority_receipts[min_idx].value_ftns),
            )
            majority_leaf = ReceiptLeafFields.from_python_receipt(
                majority_receipts[maj_idx].receipt,
                value_ftns_wei=int(majority_receipts[maj_idx].value_ftns),
            )
            minority_proof = build_merkle_proof(list(minority_leaves), min_idx)
            majority_proof = build_merkle_proof(list(majority_leaves), maj_idx)
            return ChallengeAttempt(
                minority_batch_id=bytes(finding.minority_batch_id),
                minority_leaf=minority_leaf,
                minority_proof=minority_proof,
                majority_batch_id=bytes(finding.majority_batch_id),
                majority_leaf=majority_leaf,
                majority_proof=majority_proof,
            )
        except Exception as exc:  # noqa: BLE001 — assembly fail-fast/error → no attempt, still surfaced
            logger.debug("consensus-mismatch assembly skipped (%s)", exc)
            return None

    def scan_expired_receipts(
        self, *, now: int, safety_margin_seconds: int = 60,
    ) -> List[ExpiredReceiptAuditFinding]:
        """Detect STALE (EXPIRED) leaves across the VERIFIED cache (sp1148 — the 5th + FINAL
        on-chain-actionable reason). Builds the detector input from ONLY the root-verified
        batches (their order-preserved receipts + the retained chain-anchored per-batch
        ``lookback_window_seconds``), runs ``detect_expired_receipts`` (which matches the
        on-chain ``_handleExpired`` ``age > lookback`` STRICTLY and is FAIL-CLOSED on a
        non-positive/unknown lookback + per-receipt errors), and for each finding assembles
        an ``ExpiredChallenge`` from the cached receipts. If a dry-run client is injected AND
        assembly succeeded, dry-run it (read-only) and attach ``dry_run_ok``.

        ``safety_margin_seconds`` (default 60) is a CONSERVATIVE cushion: the detector flags
        only ``age > lookback + margin`` so a leaf flagged at scan time stays expired at the
        (later) submit time despite clock skew between this process and ``block.timestamp`` —
        age only grows, so the only risk a margin guards against is flagging a leaf RIGHT at
        the boundary. The assembler then re-checks STRICTLY (``age > lookback``, no margin)
        against the same ``now`` and the dry-run gate confirms against the real chain.

        FAIL-CLOSED: a detector/build blowup returns ``[]`` (a non-actionable scan); a
        per-finding assembly/dry-run error leaves ``challenge``/``dry_run_ok`` unknown but
        STILL surfaces the finding. NEVER broadcasts — assembly + optional read-only dry-run
        only; the irreversible tx stays the separate USER-GATED submitter path (any observer
        key — EXPIRED has no on-chain msg.sender restriction)."""
        try:
            detector_batches = self._build_expired_detector_input()
            raw_findings = detect_expired_receipts(
                detector_batches, now=int(now),
                safety_margin_seconds=int(safety_margin_seconds),
            )
        except Exception as exc:  # noqa: BLE001 — fail-closed: a build/detector blowup is not actionable
            logger.error(
                "scan_expired_receipts: build/detector raised (%s); returning []", exc
            )
            return []

        out: List[ExpiredReceiptAuditFinding] = []
        for finding in raw_findings:
            challenge = self._try_assemble_expired(finding, now=int(now))
            dry_ok: Optional[bool] = None
            dry_reason: Optional[str] = None
            if challenge is not None and self._dry_run_client is not None:
                dry_ok, dry_reason = self._dry_run(challenge)
            out.append(ExpiredReceiptAuditFinding(
                finding=finding, challenge=challenge,
                dry_run_ok=dry_ok, dry_run_reason=dry_reason))
        return out

    def _build_expired_detector_input(self) -> List["_ExpiredDetectorBatch"]:
        """Build the ``detect_expired_receipts`` input from the VERIFIED cache: one batch per
        root-verified batch carrying its order-preserved receipts + the retained chain-anchored
        ``lookback_window_seconds`` (0/unknown when none — the detector is FAIL-CLOSED on it).
        A batch with missing receipts is skipped (no leaf to time-check)."""
        batches: List[_ExpiredDetectorBatch] = []
        for cb in self._cache.verified_batches():
            receipts = self._cache.receipts_for(cb.batch_id)
            if not receipts:
                continue
            lookback = self._cache.lookback_window_seconds_for(cb.batch_id)
            batches.append(_ExpiredDetectorBatch(
                batch_id=bytes(cb.batch_id),
                receipts=receipts,
                lookback_window_seconds=int(lookback) if lookback is not None else 0,
            ))
        return batches

    def _try_assemble_expired(self, finding: ExpiredReceiptFinding, *, now: int):
        """Assemble the EXPIRED ``ExpiredChallenge`` for one finding from the cached receipts.
        Re-checks STRICTLY via ``assemble_expired_challenge`` (which fail-fasts unless ``age >
        lookback``), so a finding that is no longer provably expired never yields a challenge.
        Returns the challenge, or None on missing cache data / a fail-fast / an error (never
        raises out — assembly failure leaves the finding surfaced without a dry-run verdict)."""
        try:
            receipts = self._cache.receipts_for(finding.batch_id)
            if not receipts:
                return None
            return assemble_expired_challenge(
                batch_id=bytes(finding.batch_id),
                batch_receipts=receipts,
                target_index=finding.target_index,
                now=int(now),
                lookback_window_seconds=int(finding.lookback),
            )
        except Exception as exc:  # noqa: BLE001 — assembly fail-fast/error → no dry-run, still surfaced
            logger.debug("expired assembly skipped (%s)", exc)
            return None

    def scan_inference_receipts(self) -> List[InferenceReceiptAuditFinding]:
        """Run the §7 (compute-integrity) challenge verifier over the §7 receipts retained
        for each VERIFIED batch (sp1143). NEVER broadcasts; FAIL-CLOSED per item.

        ATTACK-A defence — KEY BINDING FIRST. The §7 receipt's ``settler_public_key_b64``
        comes from the UNTRUSTED blob; ``verify_inference_receipt_for_challenge`` only
        attests self-consistency under WHATEVER key it is given. So before trusting any
        verdict, this BINDS the key to a chain-anchored source: it finds the verified
        ``BatchedReceipt`` for the leaf, computes that leaf's ``provider_pubkey_hash`` via
        ``batched_receipt_to_leaf`` (the SAME ``keccak(b64decode(pubkey))`` merkle commits),
        and requires it EQUALS ``keccak(b64decode(settler_public_key_b64))``. (The adapter
        re-signs the shard payload with the settler identity, so the committed leaf
        provider_pubkey_hash IS keccak(the settler pubkey).) If the binding FAILS, the
        record is surfaced UNBOUND (``bound=False``, no verdict) and the verifier is NOT
        run — an attacker cannot mask/fabricate fraud via a substituted key.

        For a BOUND receipt: run the verifier; emit one finding per PROVEN ground (carrying
        ``on_chain_reason``); for an on-chain-actionable INVALID_SETTLER_SIGNATURE ground,
        if a dry-run client is set, reuse the ``scan_invalid_signatures`` assemble+dry-run
        path on that leaf (read-only) to attach ``dry_run_ok``. If the verifier reports a
        clean receipt (no proven ground), emit a single clean record (``bound=True``,
        ``proven=False``). A raising bind/verify is recorded (``bound=False`` /
        ``proven=False``) and skipped — never aborts the scan, never raises out, NEVER
        broadcasts."""
        out: List[InferenceReceiptAuditFinding] = []
        for cb in self._cache.verified_batches():
            s7_map = self._cache.inference_receipts_for(cb.batch_id)
            if not s7_map:
                continue
            receipts = self._cache.receipts_for(cb.batch_id) or []
            # Map leaf_hash -> (receipt, index) for the binding lookup.
            leaf_to_receipt: Dict[bytes, Tuple[BatchedReceipt, int]] = {}
            for idx, br in enumerate(receipts):
                try:
                    lh = hash_leaf(batched_receipt_to_leaf(br))
                except Exception as exc:  # noqa: BLE001 — a bad receipt can't bind anything
                    logger.warning(
                        "scan_inference_receipts: leaf recompute failed for batch %s "
                        "idx %d (%s); skipping receipt", cb.batch_id.hex(), idx, exc,
                    )
                    continue
                leaf_to_receipt[lh] = (br, idx)

            for leaf_hash, rec in s7_map.items():
                try:
                    self._scan_one_inference_receipt(
                        out, cb.batch_id, leaf_hash, rec, leaf_to_receipt,
                    )
                except Exception as exc:  # noqa: BLE001 — fail-closed per item; never abort
                    logger.warning(
                        "scan_inference_receipts: item raised for batch %s leaf %s "
                        "(%s); recording unbound+skipped",
                        cb.batch_id.hex(), bytes(leaf_hash).hex(), exc,
                    )
                    out.append(InferenceReceiptAuditFinding(
                        batch_id=bytes(cb.batch_id), leaf_hash=bytes(leaf_hash),
                        bound=False, proven=False,
                        detail=f"audit raised: {type(exc).__name__}: {exc}",
                    ))
        return out

    def _scan_one_inference_receipt(
        self,
        out: List[InferenceReceiptAuditFinding],
        batch_id: bytes,
        leaf_hash: bytes,
        rec: RetainedInferenceReceipt,
        leaf_to_receipt: Dict[bytes, Tuple[BatchedReceipt, int]],
    ) -> None:
        """Bind-then-verify ONE retained §7 receipt; append its finding(s) to ``out``.
        Raising is caught by the caller (fail-closed per item)."""
        lh = bytes(leaf_hash)
        pair = leaf_to_receipt.get(lh)
        if pair is None:
            # No verified BatchedReceipt for this leaf -> cannot bind. UNBOUND.
            out.append(InferenceReceiptAuditFinding(
                batch_id=bytes(batch_id), leaf_hash=lh, bound=False, proven=False,
                detail="no verified leaf to bind the §7 settler key to",
            ))
            return
        br, idx = pair

        # ── ATTACK-A KEY BINDING: keccak(b64decode(settler_pubkey)) == committed leaf
        #    provider_pubkey_hash (the SAME hash merkle.batched_receipt_to_leaf commits).
        leaf = batched_receipt_to_leaf(br)
        settler_pubkey_b64 = rec.settler_public_key_b64
        bound = False
        try:
            settler_hash = keccak(b64decode(settler_pubkey_b64))
            bound = settler_hash == leaf.provider_pubkey_hash
        except Exception as exc:  # noqa: BLE001 — a malformed key cannot bind
            logger.debug(
                "scan_inference_receipts: settler key hash failed for batch %s leaf %s "
                "(%s); unbound", bytes(batch_id).hex(), lh.hex(), exc,
            )
            bound = False

        if not bound:
            # NOT bound to the committed provider — do NOT run/trust the verdict.
            out.append(InferenceReceiptAuditFinding(
                batch_id=bytes(batch_id), leaf_hash=lh, bound=False, proven=False,
                detail=(
                    "settler_public_key_b64 does not hash to the committed leaf "
                    "provider_pubkey_hash (unbound — verdict not trusted)"
                ),
            ))
            return

        # ── BOUND: run the §7 verifier (re-verify on use). ──
        report = verify_inference_receipt_for_challenge(
            rec.inference_receipt,
            settler_public_key_b64=settler_pubkey_b64,
            stage_public_keys=rec.stage_public_keys,
        )
        proven = report.proven_findings()
        if not proven:
            # Bound + clean — surface a single clean record (no fraud trusted-but-clean).
            out.append(InferenceReceiptAuditFinding(
                batch_id=bytes(batch_id), leaf_hash=lh, bound=True, proven=False,
                detail="bound §7 receipt verifies (no proven ground)",
            ))
            return

        for f in proven:
            dry_ok: Optional[bool] = None
            dry_reason: Optional[str] = None
            # On-chain-actionable INVALID_SETTLER_SIGNATURE -> reuse the invalid-signature
            # assemble + dry-run path on this leaf (read-only). NEVER broadcasts.
            if (
                self._dry_run_client is not None
                and f.on_chain_reason is not None
                and f.reason == ChallengeReason.INVALID_SETTLER_SIGNATURE
            ):
                challenge = self._try_assemble_invalid_signature(batch_id, idx)
                if challenge is not None:
                    dry_ok, dry_reason = self._dry_run(challenge)
            out.append(InferenceReceiptAuditFinding(
                batch_id=bytes(batch_id), leaf_hash=lh, bound=True, proven=True,
                reason=f.reason, on_chain_reason=f.on_chain_reason, detail=f.detail,
                dry_run_ok=dry_ok, dry_run_reason=dry_reason,
            ))

    def _try_assemble_invalid_signature(self, batch_id: bytes, idx: int):
        """Assemble the INVALID_SIGNATURE challenge for the leaf at ``idx`` in
        ``batch_id`` from the verified cache (the SAME path scan_invalid_signatures uses).
        Returns the challenge, or None on a fail-fast/missing-data/error (never raises)."""
        try:
            receipts = self._cache.receipts_for(batch_id)
            if not receipts:
                return None
            return assemble_invalid_signature_challenge(
                batch_id=bytes(batch_id),
                batch_receipts=receipts,
                target_index=idx,
            )
        except Exception as exc:  # noqa: BLE001 — assembly fail-fast/error → no dry-run
            logger.debug(
                "scan_inference_receipts: invalid-sig assembly skipped for batch %s "
                "idx %d (%s)", bytes(batch_id).hex(), idx, exc,
            )
            return None
