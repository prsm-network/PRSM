"""Sprint 1145 — off-chain CONSENSUS_MISMATCH detector (the 3rd on-chain-actionable
settlement fraud class, after INVALID_SIGNATURE + DOUBLE_SPEND).

THE ATTACK. Providers in a k-of-n CONSENSUS GROUP redundantly execute the SAME unit of
work (same job, same shard) and commit CONFLICTING outputs. The honest majority output is
correct; a minority provider that committed a different output for the same work is either
faulty or adversarial. On-chain the BatchSettlementRegistry punishes this via
``challengeReceipt(..., ReasonCode.CONSENSUS_MISMATCH, ...)``.

THE SOUND SIGNAL. Each input ``DetectorBatch`` carries (batch_id, provider_address,
consensus_group_id, ordered receipts). A leaf's identity within a consensus group is
(job_id, shard_index): two providers in the SAME group executing the SAME (job, shard) MUST
produce the SAME output. A PAIR of leaves with the same non-zero group_id and the same
(job_id, shard_index) but DIFFERENT output_hash, committed by DIFFERENT providers in
DISTINCT batches, is an UNAMBIGUOUS disagreement.

SOUNDNESS (this is the slashing path — a false positive would wrongly accuse an HONEST
provider). The detector matches the on-chain ``_handleConsensusMismatch`` reject conditions
EXACTLY (BatchSettlementRegistry.sol). A pair is flagged IFF ALL hold:
  1. distinct batch ids (``minority_batch_id != majority_batch_id``);
  2. NON-ZERO consensus_group_id on the (minority) batch;
  3. SAME non-zero consensus_group_id on BOTH batches;
  4. DIFFERENT providers (same-provider disagreement is DOUBLE_SPEND territory);
  5. SAME jobIdHash;
  6. SAME shardIndex;
  7. DIFFERENT outputHash (genuine disagreement).
A pair that violates ANY condition is NEVER flagged — no fuzzy/heuristic matching. job/shard
identity is compared via the SAME canonical leaf fields (``batched_receipt_to_leaf``) the
merkle tree commits and the contract compares, so the off-chain signal is byte-for-byte the
on-chain comparison.

PURITY. This module is intentionally PURE: no web3, no network, no I/O, no signing. It
consumes already-verified receipt sets and returns findings. It does NOT broadcast, sign, or
slash — assembling + DRY-RUNNING the on-chain challenge (reusing
``marketplace.consensus_submitter``) is the audit-engine brick, and the actual broadcast
stays the separate user-gated submitter/queue path. This mirrors the
``double_spend_detector`` → ``settlement_audit_engine`` split.

DETERMINISM. Findings are returned in a stable, input-order-independent order: sorted by
(consensus_group_id, jobIdHash, shardIndex), and within a (group, job, shard) the two sides
are sorted by (batch_id, provider) so the "minority"/"majority" labelling is deterministic.
(The on-chain ``_handleConsensusMismatch`` is symmetric in which side is challenged; the
audit engine challenges the chosen ``minority_*`` side. Labelling is purely positional here.)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from prsm.economy.web3.stake_manager import ReasonCode
from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.merkle import batched_receipt_to_leaf

logger = logging.getLogger(__name__)

# The all-zero "no consensus group" id. A batch with this group id did NOT opt into
# consensus, so it can NEVER be a CONSENSUS_MISMATCH side (on-chain reject condition #2).
_ZERO_GROUP_ID = b"\x00" * 32

# The BatchSettlementRegistry ReasonCode this detector maps to.
_ON_CHAIN_CONSENSUS_MISMATCH: int = int(ReasonCode.CONSENSUS_MISMATCH)

# Stable string tag mirroring double_spend_detector / challenge_verifier value style.
_REASON_TAG: str = "consensus_mismatch"


@dataclass(frozen=True)
class DetectorBatch:
    """One verified, co-group batch the detector compares against its peers.

    ``batch_id`` + ``provider_address`` + ``consensus_group_id`` are READ FROM THE CHAIN
    (the trusted ``Batch`` struct); ``receipts`` are the order-preserved ``BatchedReceipt``
    list that PASSED the on-chain-root cross-check (the verified cache's
    ``receipts_for``). The detector never sees an un-cross-checked receipt set."""

    batch_id: bytes
    provider_address: str
    consensus_group_id: bytes
    receipts: List[BatchedReceipt] = field(default_factory=list)


@dataclass(frozen=True)
class ConsensusMismatchFinding:
    """A single co-group disagreement: two providers in the SAME non-zero consensus group
    committed DIFFERENT outputs for the SAME (job, shard).

    The two sides are labelled positionally (sorted by batch_id then provider) — the
    on-chain check is symmetric in which side is challenged, so "minority"/"majority" here
    is deterministic-labelling only; the audit engine challenges the ``minority_*`` side and
    proves the ``majority_*`` leaf in the conflicting batch.

    ``job_id_hash`` / ``shard_index`` / ``output_hash``es are the canonical leaf fields
    (``batched_receipt_to_leaf``) — the SAME the merkle tree commits and the contract
    compares — so the finding is on-chain-actionable byte-for-byte."""

    consensus_group_id: bytes
    job_id_hash: bytes
    shard_index: int

    minority_batch_id: bytes
    minority_provider: str
    minority_receipt: BatchedReceipt
    minority_leaf_index: int
    minority_output_hash: bytes

    majority_batch_id: bytes
    majority_provider: str
    majority_receipt: BatchedReceipt
    majority_leaf_index: int
    majority_output_hash: bytes

    on_chain_reason: int = _ON_CHAIN_CONSENSUS_MISMATCH

    def to_dict(self) -> Dict[str, object]:
        return {
            "reason": _REASON_TAG,
            "consensus_group_id": self.consensus_group_id.hex(),
            "job_id_hash": self.job_id_hash.hex(),
            "shard_index": self.shard_index,
            "minority": {
                "batch_id": self.minority_batch_id.hex(),
                "provider": self.minority_provider,
                "leaf_index": self.minority_leaf_index,
                "output_hash": self.minority_output_hash.hex(),
            },
            "majority": {
                "batch_id": self.majority_batch_id.hex(),
                "provider": self.majority_provider,
                "leaf_index": self.majority_leaf_index,
                "output_hash": self.majority_output_hash.hex(),
            },
            "on_chain_reason": self.on_chain_reason,
        }


@dataclass(frozen=True)
class _Side:
    """One committed leaf occupying a (group, job, shard) slot — a candidate side."""

    batch_id: bytes
    provider: str
    receipt: BatchedReceipt
    leaf_index: int
    output_hash: bytes


def detect_consensus_mismatches(
    batches: Iterable[DetectorBatch],
) -> List[ConsensusMismatchFinding]:
    """Find every co-group provider disagreement across the given verified batches.

    Walks every leaf of every batch, derives the canonical ``(consensus_group_id,
    job_id_hash, shard_index)`` key + ``output_hash`` via ``batched_receipt_to_leaf`` (the
    SAME canonicalization merkle + the contract use), and groups candidate sides by that
    key. For each key, emits ONE finding per pair of sides that satisfies the FULL on-chain
    reject-set:

      - NON-ZERO group_id (zero-group leaves are never keyed);
      - DIFFERENT providers;
      - DIFFERENT output_hash;
      - DISTINCT batch ids.

    The (group_id, job, shard) key already binds the SAME non-zero group_id and the SAME
    job+shard for every member, so conditions on group/job/shard equality are structural.

    SOUND: a pair that agrees on output, comes from the same provider, the same batch, a
    zero group, a different group, a different job, or a different shard is NEVER flagged.
    DETERMINISTIC: sides are sorted by (batch_id, provider) before pairing; findings are
    sorted by (group_id, job_id_hash, shard_index) — independent of input order."""
    # (group_id, job_id_hash, shard_index) -> list[_Side], in deterministic order.
    by_key: Dict[Tuple[bytes, bytes, int], List[_Side]] = {}

    for batch in batches:
        group_id = bytes(batch.consensus_group_id)
        # On-chain reject condition #2: non-consensus (zero-group) batches can NEVER be a
        # CONSENSUS_MISMATCH side. Skip entirely.
        if group_id == _ZERO_GROUP_ID:
            continue
        batch_id = bytes(batch.batch_id)
        provider = batch.provider_address
        for leaf_index, br in enumerate(batch.receipts):
            # Canonical leaf — the SAME fields the merkle tree commits + the contract
            # compares (jobIdHash, shardIndex, outputHash). PER-RECEIPT FAIL-CLOSED: a
            # single malformed/observed receipt (whose leaf can't be built) is skipped +
            # logged, so it can't suppress detection across the rest of the scan; a
            # skipped receipt simply isn't a side (safe — no false positive).
            try:
                leaf = batched_receipt_to_leaf(br)
            except Exception as exc:  # noqa: BLE001 — fail-closed, never abort the scan
                logger.debug(
                    "consensus-mismatch detector: skipping unparseable receipt "
                    "(batch=%s idx=%d): %s", batch_id.hex(), leaf_index, exc,
                )
                continue
            key = (group_id, bytes(leaf.job_id_hash), int(leaf.shard_index))
            by_key.setdefault(key, []).append(
                _Side(
                    batch_id=batch_id,
                    provider=provider,
                    receipt=br,
                    leaf_index=leaf_index,
                    output_hash=bytes(leaf.output_hash),
                )
            )

    findings: List[ConsensusMismatchFinding] = []
    for (group_id, job_id_hash, shard_index), sides in by_key.items():
        if len(sides) < 2:
            continue
        # Deterministic side ordering -> deterministic minority/majority labelling.
        ordered = sorted(sides, key=lambda s: (s.batch_id, s.provider, s.leaf_index))
        n = len(ordered)
        for i in range(n):
            for j in range(i + 1, n):
                lo, hi = ordered[i], ordered[j]
                # FULL on-chain reject-set (every condition must hold):
                if lo.batch_id == hi.batch_id:
                    continue  # distinct batches (reject condition #1)
                if lo.provider == hi.provider:
                    continue  # different providers (reject condition #4)
                if lo.output_hash == hi.output_hash:
                    continue  # genuine disagreement (reject condition #7)
                findings.append(
                    ConsensusMismatchFinding(
                        consensus_group_id=group_id,
                        job_id_hash=job_id_hash,
                        shard_index=shard_index,
                        minority_batch_id=lo.batch_id,
                        minority_provider=lo.provider,
                        minority_receipt=lo.receipt,
                        minority_leaf_index=lo.leaf_index,
                        minority_output_hash=lo.output_hash,
                        majority_batch_id=hi.batch_id,
                        majority_provider=hi.provider,
                        majority_receipt=hi.receipt,
                        majority_leaf_index=hi.leaf_index,
                        majority_output_hash=hi.output_hash,
                        on_chain_reason=_ON_CHAIN_CONSENSUS_MISMATCH,
                    )
                )

    findings.sort(key=lambda f: (f.consensus_group_id, f.job_id_hash, f.shard_index,
                                 f.minority_batch_id, f.majority_batch_id))
    return findings


__all__ = [
    "DetectorBatch",
    "ConsensusMismatchFinding",
    "detect_consensus_mismatches",
]
