"""Sprint 1265 — contribution-proof verifiers reject forged work (security audit round 4).

Two proof-of-contribution verifiers in ContributorManager rubber-stamped attacker-supplied
data, which gates FTNS-earning eligibility / contributor tier (anti-speculation):

  - _verify_peer_review accepted ANY review dict with the right fields — so a submitter could
    fabricate their own "peer reviews" (self-authored, duplicated) to pass the data-proof's
    "min 2 verified reviews" gate (finding #5).
  - _verify_computation_hash accepted ANY 64-hex string as a valid compute proof ("for demo
    purposes, assume valid") (finding #4).

Fix:
  - _verify_peer_review now rejects self-reviews (reviewer_id == submitter), duplicate
    reviewers within a submission, and empty review hashes — fully closing the
    self-fabrication vector with the submitter context that _verify_data_proof already has.
  - _verify_computation_hash gains a fail-closed STRICT gate (PRSM_CONTRIBUTOR_PROOF_STRICT):
    real cryptographic binding to the node's signed §7 inference/compute receipts is a
    larger follow-on, so strict mode REJECTS unbindable compute proofs; default keeps the
    prior behavior but logs a loud warning that the proof is not cryptographically verified.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from prsm.economy.tokenomics.contributor_manager import ContributorManager


def _mgr():
    return ContributorManager(db_session=MagicMock())


def _review(reviewer_id, *, review_hash="abc123", rating=4):
    return {"reviewer_id": reviewer_id, "review_hash": review_hash, "rating": rating}


# ── peer-review authentication (finding #5) ──────────────────────────────────────

@pytest.mark.asyncio
async def test_peer_review_rejects_self_review():
    mgr = _mgr()
    seen = set()
    # reviewer is the submitter themselves → forged self-review
    assert await mgr._verify_peer_review(_review("alice"), submitter_id="alice", seen=seen) is False


@pytest.mark.asyncio
async def test_peer_review_rejects_duplicate_reviewer():
    mgr = _mgr()
    seen = set()
    assert await mgr._verify_peer_review(_review("bob"), submitter_id="alice", seen=seen) is True
    # same reviewer again in the same submission → padding → reject
    assert await mgr._verify_peer_review(_review("bob"), submitter_id="alice", seen=seen) is False


@pytest.mark.asyncio
async def test_peer_review_rejects_empty_hash():
    mgr = _mgr()
    assert await mgr._verify_peer_review(_review("bob", review_hash=""), submitter_id="alice", seen=set()) is False


@pytest.mark.asyncio
async def test_peer_review_accepts_distinct_third_party():
    mgr = _mgr()
    assert await mgr._verify_peer_review(_review("carol"), submitter_id="alice", seen=set()) is True


@pytest.mark.asyncio
async def test_peer_review_rejects_bad_rating():
    mgr = _mgr()
    assert await mgr._verify_peer_review(_review("bob", rating=9), submitter_id="alice", seen=set()) is False


# ── data-proof integration: self-fabricated reviews can't pass the min-2 gate ─────

def _data_proof(reviews):
    return {
        "dataset_hash": "d" * 64,
        "peer_reviews": reviews,
        "quality_metrics": {"completeness": 0.9, "accuracy": 0.9, "uniqueness": 0.8, "usefulness": 0.7},
    }


@pytest.mark.asyncio
async def test_data_proof_rejects_all_self_reviews():
    mgr = _mgr()
    # submitter forges two "peer reviews" authored by themselves → 0 verified → reject
    proof = _data_proof([_review("alice"), _review("alice", review_hash="z")])
    with pytest.raises(ValueError, match="Insufficient verified peer reviews"):
        await mgr._verify_data_proof("alice", proof)


@pytest.mark.asyncio
async def test_data_proof_accepts_two_distinct_third_parties():
    mgr = _mgr()
    proof = _data_proof([_review("bob"), _review("carol")])
    result = await mgr._verify_data_proof("alice", proof)
    assert result.value > Decimal("0")


# ── compute-hash strict fail-closed gate (finding #4) ────────────────────────────

@pytest.mark.asyncio
async def test_compute_hash_strict_mode_rejects(monkeypatch):
    monkeypatch.setenv("PRSM_CONTRIBUTOR_PROOF_STRICT", "1")
    mgr = _mgr()
    # a well-shaped-but-unbindable hash must be REJECTED under strict mode
    assert await mgr._verify_computation_hash("a" * 64, 100) is False


@pytest.mark.asyncio
async def test_compute_hash_default_preserves_shape_checks(monkeypatch):
    monkeypatch.delenv("PRSM_CONTRIBUTOR_PROOF_STRICT", raising=False)
    mgr = _mgr()
    assert await mgr._verify_computation_hash("a" * 64, 100) is True   # default permissive (warns)
    assert await mgr._verify_computation_hash("short", 100) is False   # bad shape still rejected
    assert await mgr._verify_computation_hash("a" * 64, 0) is False    # non-positive work rejected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
