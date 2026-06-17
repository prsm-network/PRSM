"""Sprint 1139 — settlement-receipt data plane BRICK E.1: the OBSERVE-LOOP logic.

Bricks A-D are complete + committed (see
``docs/2026-06-16-settlement-receipt-data-plane-design.md``):
  - A retains each committed batch's ordered receipt set (``PublishedBatch``);
  - B canonicalizes it to a CID-addressable blob + the on-chain-root CROSS-CHECK gate
    (``deserialize_receipt_set`` / ``verify_receipt_set_against_root``);
  - C indexes the UNTRUSTED availability pointers (``SettlementBatchAdIndex.get`` -> ad
    carrying ``.cid``);
  - D is the PURE/injectable audit engine (selectors + ``VerifiedBatchCache`` trust gate +
    the dry-run-gated ``SettlementAuditEngine``).

Brick E.1 is the orchestration LOGIC that ties them into ONE ``run_once`` pass:

    observed = await enumerator.enumerate_committed_batches(from_block, to_block)  # TRUSTED anchor
    selected = selector.select(observed)                                          # audit policy
    for b in selected (up to max_fetches_per_run):
        ad = ad_index.get(b.batch_id)                                             # UNTRUSTED pointer
        if ad is None or not ad.cid:  -> skipped_no_cid; continue
        try:
            blob = await fetcher.fetch(ad.cid)                                    # UNTRUSTED bytes
            pb   = deserialize_receipt_set(blob)                                  # bounded parse
            ok   = cache.ingest(<b with cid filled>, pb)                          # CROSS-CHECK gate
        except Exception:  -> rejected; log; continue                            # FAIL-CLOSED per batch
    ds = engine.scan_double_spends(); inv = engine.scan_invalid_signatures()      # dry-run-gated
    return AuditRunResult(...)

The crucial trust property: detection is fed ONLY via ``cache.ingest`` — which cross-checks
the fetched set against the TRUSTED on-chain ``merkle_root`` FIRST and caches nothing on
mismatch. So an un-cross-checked / forged / altered batch is NEVER scanned, and a forged
double-spend planted in such a set never surfaces.

HARD CONSTRAINTS honoured:
  - PURE / injectable: enumerator, selector, ad_index, fetcher, cache, engine are ALL
    injected; this module has no hard ContentProvider/web3 dependency and is fully
    offline-testable with fakes;
  - NEVER broadcasts / signs / slashes: it only drives the dry-run-gated engine scans;
  - FAIL-CLOSED per batch: any fetch / deserialize / ingest error skips THAT batch, counts
    it as rejected, logs it, and continues — it never aborts the run or raises out;
  - BOUNDED: a per-run fetch budget (``max_fetches_per_run``) caps work per pass.

OUT OF SCOPE (Brick E.2, operator/testbed-gated): the live ``node.initialize()`` daemon
wiring, scheduling, and announce-on-commit. This module is the pure pass; wiring it into a
running node is a separate, gated brick.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any, List, Optional, Protocol

from prsm.settlement.receipt_set_blob import deserialize_enriched_receipt_set
from prsm.settlement.settlement_audit_engine import (
    ConsensusMismatchAuditFinding,
    DoubleSpendAuditFinding,
    ExpiredReceiptAuditFinding,
    InferenceReceiptAuditFinding,
    InvalidSignatureAuditFinding,
    ObservedBatch,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_FETCHES_PER_RUN = 256


# ── injectable component interfaces (Protocols — fakes satisfy them) ──


class CommittedBatchEnumerator(Protocol):
    """The TRUSTED on-chain worklist source. Mirrors
    ``Web3SettlementContractClient.enumerate_committed_batches``."""

    async def enumerate_committed_batches(
        self, from_block: int, to_block: int, *, provider: Optional[str] = None,
    ) -> List[ObservedBatch]:
        ...


class BlobFetcher(Protocol):
    """Async fetch a blob by CID. Wraps e.g. ``ContentProvider.request_content`` so the
    loop has no hard ContentProvider dependency and is offline-testable."""

    async def fetch(self, cid: str) -> bytes:
        ...


# ── per-run result ───────────────────────────────────────────────────


@dataclass(frozen=True)
class AuditRunResult:
    """The outcome of ONE ``run_once`` pass. Counts are observability; the findings lists
    are the actionable (dry-run-gated, never-broadcast) dispute candidates."""

    enumerated: int = 0
    selected: int = 0
    fetched: int = 0
    verified: int = 0
    rejected: int = 0
    skipped_no_cid: int = 0
    double_spend_findings: List[DoubleSpendAuditFinding] = field(default_factory=list)
    invalid_sig_findings: List[InvalidSignatureAuditFinding] = field(default_factory=list)
    # sp1143 — §7 compute-integrity findings from scan_inference_receipts (bound+verified
    # §7 receipts of root-verified batches). ``inference_receipt_finding_count`` mirrors
    # ``len(inference_receipt_findings)`` as the observability counter for this scan.
    inference_receipt_findings: List[InferenceReceiptAuditFinding] = field(
        default_factory=list
    )
    inference_receipt_finding_count: int = 0
    # sp1145 — CONSENSUS_MISMATCH findings from scan_consensus_mismatches (co-group provider
    # disagreement: same non-zero group, same job+shard, different providers, different
    # outputs — across root-verified batches). ``consensus_mismatch_finding_count`` mirrors
    # ``len(consensus_mismatch_findings)`` as the observability counter for this scan.
    consensus_mismatch_findings: List[ConsensusMismatchAuditFinding] = field(
        default_factory=list
    )
    consensus_mismatch_finding_count: int = 0
    # sp1148 — EXPIRED findings from scan_expired_receipts (stale leaves: age > the per-batch
    # lookback, across root-verified batches; STRICT on-chain match; never broadcasts).
    # ``expired_finding_count`` mirrors ``len(expired_findings)`` as the observability counter.
    expired_findings: List[ExpiredReceiptAuditFinding] = field(default_factory=list)
    expired_finding_count: int = 0


# ── the loop ──────────────────────────────────────────────────────────


class SettlementAuditLoop:
    """PURE/injectable orchestrator for ONE observe pass (Brick E.1).

    enumerate (trusted) -> select (policy) -> fetch (untrusted) -> deserialize (bounded)
    -> cross-check-and-cache (the trust gate) -> dry-run-gated engine scans. NEVER
    broadcasts; FAIL-CLOSED per batch; bounded by ``max_fetches_per_run``."""

    def __init__(
        self,
        *,
        enumerator: CommittedBatchEnumerator,
        selector: Any,
        ad_index: Any,
        fetcher: BlobFetcher,
        cache: Any,
        engine: Any,
        max_fetches_per_run: int = _DEFAULT_MAX_FETCHES_PER_RUN,
    ) -> None:
        if max_fetches_per_run <= 0:
            raise ValueError(
                f"max_fetches_per_run must be positive (got {max_fetches_per_run})"
            )
        self._enumerator = enumerator
        self._selector = selector
        self._ad_index = ad_index
        self._fetcher = fetcher
        self._cache = cache
        self._engine = engine
        self._max_fetches_per_run = int(max_fetches_per_run)

    async def run_once(
        self, from_block: int, to_block: int, *, provider: Optional[str] = None,
    ) -> AuditRunResult:
        """Run one enumerate→select→fetch→cross-check→scan pass over
        [from_block, to_block]. Returns the per-run counts + the dry-run-gated findings.
        Never broadcasts; per-batch fail-closed; bounded fetch budget."""
        observed = await self._enumerator.enumerate_committed_batches(
            from_block, to_block, provider=provider,
        )
        enumerated = len(observed)
        selected = self._selector.select(observed)

        fetched = 0
        verified = 0
        rejected = 0
        skipped_no_cid = 0

        for b in selected:
            if fetched >= self._max_fetches_per_run:
                # Budget exhausted — stop fetching this pass (remaining batches roll
                # into a future pass). Bounded work per run.
                logger.debug(
                    "SettlementAuditLoop: fetch budget %d reached; deferring remaining",
                    self._max_fetches_per_run,
                )
                break

            ad = self._ad_index.get(b.batch_id)
            cid = getattr(ad, "cid", None) if ad is not None else None
            if not cid:
                skipped_no_cid += 1
                logger.debug(
                    "SettlementAuditLoop: no ad/cid for batch %s; skipping",
                    bytes(b.batch_id).hex(),
                )
                continue

            fetched += 1
            try:
                blob = await self._fetcher.fetch(cid)
                # Brick-2 enriched parse: (PublishedBatch, {leaf_hash -> §7 receipt}). A
                # plain (no-§7) blob yields an empty §7 map — fully back-compatible.
                pb, section7_map = deserialize_enriched_receipt_set(blob)
                # Fill the (untrusted) cid into the observed metadata so the cached
                # record knows where it came from. The trust gate is cache.ingest, which
                # cross-checks pb against the TRUSTED on-chain merkle_root FIRST and only
                # retains the §7 map (for THIS batch's leaves) once that cross-check passes.
                observed_with_cid = replace(b, cid=cid)
                ok = self._cache.ingest(
                    observed_with_cid, pb, inference_receipts=section7_map
                )
            except Exception as exc:  # noqa: BLE001 — fail-closed per batch; never abort the run
                rejected += 1
                logger.warning(
                    "SettlementAuditLoop: batch %s rejected (cid=%s): %s",
                    bytes(b.batch_id).hex(), cid, exc,
                )
                continue

            if ok:
                verified += 1
            else:
                # Cross-check failed — the set did NOT recompute to the on-chain root,
                # so it was NOT cached and will NOT be scanned (trust preserved).
                rejected += 1
                logger.debug(
                    "SettlementAuditLoop: batch %s failed on-chain-root cross-check; "
                    "not cached", bytes(b.batch_id).hex(),
                )

        # Drive the dry-run-gated detectors over the VERIFIED cache only. NEVER broadcasts.
        double_spend_findings = self._engine.scan_double_spends()
        invalid_sig_findings = self._engine.scan_invalid_signatures()
        # sp1143 — §7 compute-integrity scan: bind-then-verify the retained §7 receipts of
        # root-verified batches (NEVER broadcasts; fail-closed per item; key-bound only).
        inference_receipt_findings = list(self._engine.scan_inference_receipts())
        # sp1145 — CONSENSUS_MISMATCH scan: co-group provider disagreement over root-verified
        # batches (NEVER broadcasts; assemble + optional read-only dry-run only; the actual
        # broadcast stays the separate user-gated submitter/queue path).
        consensus_mismatch_findings = list(self._engine.scan_consensus_mismatches())
        # sp1148 — EXPIRED scan: stale leaves (age > the per-batch chain-anchored lookback) over
        # root-verified batches. STRICT on-chain match + fail-closed on unknown lookback +
        # conservative safety margin; NEVER broadcasts (assemble + optional read-only dry-run
        # only — the irreversible tx stays the separate user-gated submitter path). ``now`` is
        # the observer's wall clock; the dry-run gate confirms against the real block.timestamp.
        expired_findings = list(self._engine.scan_expired_receipts(now=int(time.time())))

        return AuditRunResult(
            enumerated=enumerated,
            selected=len(selected),
            fetched=fetched,
            verified=verified,
            rejected=rejected,
            skipped_no_cid=skipped_no_cid,
            double_spend_findings=list(double_spend_findings),
            invalid_sig_findings=list(invalid_sig_findings),
            inference_receipt_findings=inference_receipt_findings,
            inference_receipt_finding_count=len(inference_receipt_findings),
            consensus_mismatch_findings=consensus_mismatch_findings,
            consensus_mismatch_finding_count=len(consensus_mismatch_findings),
            expired_findings=expired_findings,
            expired_finding_count=len(expired_findings),
        )
