"""Off-chain DOUBLE-SPEND detector for §7 settlement batches (challenge BRICK 6).

THE ATTACK. A provider commits the SAME inference receipt in two different
settlement batches — or twice within one batch — to get paid twice for one unit of
work. On-chain the BatchSettlementRegistry punishes this via
``challengeReceipt(..., ReasonCode.DOUBLE_SPEND, ...)``.

THE SOUND SIGNAL. ``CommittedBatch`` (prsm/settlement/client.py) retains, per
committed batch, the tuple of merkle ``leaf_hashes`` alongside ``batch_id`` and
``provider_address``. A merkle leaf is a hash of a receipt's canonical fields, so
the SAME receipt always produces the SAME leaf hash, and DISTINCT receipts produce
DISTINCT leaf hashes. Therefore a leaf hash that appears at MORE THAN ONE position
— across two committed batches OR twice within one batch — is an UNAMBIGUOUS
double-commit. This is detectable from a collection of ``CommittedBatch`` records
ALONE; the underlying receipts (NOT retained post-commit) are not needed.

SOUNDNESS (this is the slashing path — a false positive would wrongly accuse an
HONEST provider). The detector flags a finding ONLY when a byte-identical leaf hash
occupies >=2 DISTINCT positions. There is no fuzzy/heuristic matching: leaves are
compared for exact byte equality. Two receipts that differ in any field hash to
different leaves and therefore can NEVER collide into a finding. A leaf appearing at
exactly one position is NOT flagged.

PURITY. This module is intentionally PURE: no web3, no network, no I/O. It consumes
already-fetched ``CommittedBatch`` records and returns findings. It does NOT
broadcast, sign, or slash — assembling + submitting the on-chain DOUBLE_SPEND
challenge is a follow-on brick (exactly as INVALID_SIGNATURE's brick-1 verifier in
challenge_verifier.py precedes its assembler + submitter bricks).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from prsm.economy.web3.stake_manager import ReasonCode
from prsm.settlement.client import CommittedBatch

# The BatchSettlementRegistry ReasonCode this detector maps to (== 0).
_ON_CHAIN_DOUBLE_SPEND: int = int(ReasonCode.DOUBLE_SPEND)

# Stable string tag mirroring challenge_verifier.ChallengeReason value style.
_REASON_TAG: str = "double_spend"


@dataclass(frozen=True)
class LeafOccurrence:
    """One position at which a duplicated leaf hash was committed.

    ``batch_id`` + ``leaf_index`` together identify the exact committed leaf (the
    coordinates a downstream challenge assembler needs to build a merkle proof);
    ``provider_address`` records whom the commit pays (whom to challenge)."""

    batch_id: bytes
    leaf_index: int
    provider_address: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id.hex(),
            "leaf_index": self.leaf_index,
            "provider_address": self.provider_address,
        }


@dataclass(frozen=True)
class DoubleSpendFinding:
    """A single duplicated merkle leaf committed at >=2 distinct positions.

    ``leaf_hash`` is the byte-identical leaf that recurs; ``occurrences`` lists
    every position it was committed at (ordered by batch order then leaf index).
    ``on_chain_reason`` is ``int(ReasonCode.DOUBLE_SPEND)`` (== 0), the
    BatchSettlementRegistry reason code this finding is actionable under."""

    leaf_hash: bytes
    occurrences: List[LeafOccurrence] = field(default_factory=list)
    on_chain_reason: int = _ON_CHAIN_DOUBLE_SPEND

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason": _REASON_TAG,
            "leaf_hash": self.leaf_hash.hex(),
            "occurrences": [occ.to_dict() for occ in self.occurrences],
            "on_chain_reason": self.on_chain_reason,
        }


def detect_double_spends(
    batches: Iterable[CommittedBatch],
) -> List[DoubleSpendFinding]:
    """Find every merkle leaf double-committed across (or within) the given batches.

    Builds a map ``leaf_hash -> [LeafOccurrence, ...]`` by walking every leaf of
    every batch in input order, then emits ONE ``DoubleSpendFinding`` per leaf hash
    that occupies >=2 distinct positions. This covers BOTH inter-batch duplicates
    (the same leaf in two different batches) AND intra-batch double-commits (the
    same leaf at two indices within one batch).

    SOUND: only a byte-identical leaf at >=2 positions is flagged; a leaf at exactly
    one position is never flagged, so distinct receipts (which hash to distinct
    leaves) cannot produce a finding.

    DETERMINISTIC: findings are returned sorted by ``leaf_hash`` (ascending); within
    each finding, occurrences preserve input order (batch order, then leaf index),
    which is already (batch, index) ascending as built.
    """
    # Preserve first-seen-position order within each leaf's occurrence list by
    # iterating batches/leaves in input order. A plain dict preserves insertion
    # order, so occurrences come out ordered by (batch index, leaf index).
    occurrences_by_leaf: Dict[bytes, List[LeafOccurrence]] = {}
    for batch in batches:
        batch_id = batch.batch_id
        provider = batch.provider_address
        for leaf_index, leaf_hash in enumerate(batch.leaf_hashes):
            occ = LeafOccurrence(
                batch_id=batch_id,
                leaf_index=leaf_index,
                provider_address=provider,
            )
            occurrences_by_leaf.setdefault(leaf_hash, []).append(occ)

    findings: List[DoubleSpendFinding] = []
    for leaf_hash, occurrences in occurrences_by_leaf.items():
        if len(occurrences) >= 2:
            findings.append(
                DoubleSpendFinding(
                    leaf_hash=leaf_hash,
                    occurrences=occurrences,
                    on_chain_reason=_ON_CHAIN_DOUBLE_SPEND,
                )
            )

    # Deterministic, input-order-independent ordering of findings.
    findings.sort(key=lambda f: f.leaf_hash)
    return findings
