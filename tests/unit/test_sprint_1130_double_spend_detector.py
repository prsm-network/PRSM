"""sprint 1130 — off-chain DOUBLE-SPEND detector (challenge/dispute BRICK 6).

The attack: a provider commits the SAME inference receipt in two different
settlement batches (or twice within one batch) to get paid twice for one unit of
work. A merkle leaf is a hash of a receipt's canonical fields, so the SAME receipt
produces the SAME leaf hash. A leaf hash appearing at >1 position — across two
committed batches OR twice within one batch — is an UNAMBIGUOUS double-commit,
detectable from a collection of CommittedBatch records ALONE.

The detector is SOUND: it flags ONLY a byte-identical leaf at >=2 distinct
positions. Distinct receipts have distinct leaves, so honest providers are NEVER
accused (this is the slashing path — a false positive = an unjust slash).

These tests are written FIRST (red on import error) and operate purely on small
in-memory CommittedBatch records — NO network.
"""
from __future__ import annotations

from prsm.settlement.accumulator import TriggerReason
from prsm.settlement.client import CommittedBatch
from prsm.economy.web3.stake_manager import ReasonCode

from prsm.settlement.double_spend_detector import (
    DoubleSpendFinding,
    LeafOccurrence,
    detect_double_spends,
)

PROV = "0x" + "aa" * 20
PROV2 = "0x" + "bb" * 20
REQ = "0x" + "cc" * 20


def _batch(batch_id: bytes, leaves, *, provider: str = PROV) -> CommittedBatch:
    """Build a minimal CommittedBatch with the given leaf hashes."""
    return CommittedBatch(
        batch_id=batch_id,
        tx_hash="0x" + batch_id.hex(),
        provider_address=provider,
        requester_address=REQ,
        merkle_root=b"\xcd" * 32,
        receipt_count=len(leaves),
        total_value_ftns=1000,
        commit_timestamp=1700000000,
        leaf_hashes=tuple(leaves),
        trigger_reason=TriggerReason.COUNT,
    )


def test_same_leaf_in_two_batches_is_one_finding_naming_both():
    """(a) The SAME leaf hash in two different batches -> exactly one finding
    naming both batch_ids, on_chain_reason == 0 (DOUBLE_SPEND)."""
    dup = b"\x11" * 32
    b1 = _batch(b"\x01" * 32, [dup, b"\x02" * 32])
    b2 = _batch(b"\x03" * 32, [b"\x04" * 32, dup])

    findings = detect_double_spends([b1, b2])

    assert len(findings) == 1
    f = findings[0]
    assert isinstance(f, DoubleSpendFinding)
    assert f.leaf_hash == dup
    assert f.on_chain_reason == int(ReasonCode.DOUBLE_SPEND)
    assert f.on_chain_reason == 0
    named = {occ.batch_id for occ in f.occurrences}
    assert named == {b"\x01" * 32, b"\x03" * 32}
    assert len(f.occurrences) == 2
    # the leaf_index of each occurrence is recorded
    by_batch = {occ.batch_id: occ.leaf_index for occ in f.occurrences}
    assert by_batch[b"\x01" * 32] == 0
    assert by_batch[b"\x03" * 32] == 1


def test_all_distinct_leaves_yield_no_findings_including_near_misses():
    """(b) SOUNDNESS — all-distinct leaves across many batches -> NO findings.
    Most important test: include near-miss leaves that differ by ONE byte and
    confirm they do NOT collide into a finding (no false positive)."""
    base = b"\x00" * 31
    # Two leaves that differ only in their final byte (a near miss).
    near_a = base + b"\x01"
    near_b = base + b"\x02"
    # A leaf that differs only in its FIRST byte from near_a.
    near_c = b"\x01" + b"\x00" * 30 + b"\x01"
    b1 = _batch(b"\x01" * 32, [near_a, near_b])
    b2 = _batch(b"\x02" * 32, [near_c, b"\x55" * 32])
    b3 = _batch(b"\x03" * 32, [b"\x66" * 32, b"\x77" * 32, b"\x88" * 32])

    findings = detect_double_spends([b1, b2, b3])

    assert findings == []


def test_same_leaf_twice_within_single_batch_is_flagged():
    """(c) The same leaf appearing TWICE within a SINGLE batch -> flagged
    (intra-batch double-commit)."""
    dup = b"\x11" * 32
    b1 = _batch(b"\x01" * 32, [dup, b"\x02" * 32, dup])

    findings = detect_double_spends([b1])

    assert len(findings) == 1
    f = findings[0]
    assert f.leaf_hash == dup
    assert f.on_chain_reason == 0
    # both occurrences are in the same batch but at distinct leaf indices
    assert all(occ.batch_id == b"\x01" * 32 for occ in f.occurrences)
    indices = sorted(occ.leaf_index for occ in f.occurrences)
    assert indices == [0, 2]


def test_leaf_appearing_three_times_across_three_batches_one_finding_three_occurrences():
    """(d) A leaf appearing 3x across 3 batches -> one finding with three
    occurrences."""
    dup = b"\x99" * 32
    b1 = _batch(b"\x01" * 32, [dup])
    b2 = _batch(b"\x02" * 32, [b"\x02" * 32, dup])
    b3 = _batch(b"\x03" * 32, [dup, b"\x03" * 32])

    findings = detect_double_spends([b1, b2, b3])

    assert len(findings) == 1
    f = findings[0]
    assert f.leaf_hash == dup
    assert len(f.occurrences) == 3
    assert {occ.batch_id for occ in f.occurrences} == {
        b"\x01" * 32, b"\x02" * 32, b"\x03" * 32
    }


def test_empty_and_single_unique_batch_no_findings_no_crash():
    """(e) Empty input / single batch with unique leaves -> no findings, no
    crash."""
    assert detect_double_spends([]) == []
    b1 = _batch(b"\x01" * 32, [b"\x02" * 32, b"\x03" * 32, b"\x04" * 32])
    assert detect_double_spends([b1]) == []


def test_finding_to_dict_hexes_bytes_and_is_serializable():
    """to_dict() serializes the duplicated leaf + occurrences with bytes hexed."""
    import json

    dup = b"\x11" * 32
    b1 = _batch(b"\x01" * 32, [dup])
    b2 = _batch(b"\x02" * 32, [dup])

    findings = detect_double_spends([b1, b2])
    d = findings[0].to_dict()

    assert d["leaf_hash"] == dup.hex()
    assert d["on_chain_reason"] == 0
    assert d["reason"] == "double_spend"
    batch_ids = {occ["batch_id"] for occ in d["occurrences"]}
    assert batch_ids == {(b"\x01" * 32).hex(), (b"\x02" * 32).hex()}
    # fully JSON serializable (no raw bytes anywhere)
    json.dumps(d)


def test_deterministic_ordering_across_multiple_duplicates():
    """Multiple duplicated leaves -> findings emitted in a deterministic order
    (by leaf_hash), and occurrences ordered by (batch order, leaf_index)."""
    dup1 = b"\xaa" * 32
    dup2 = b"\x22" * 32  # sorts before dup1
    b1 = _batch(b"\x01" * 32, [dup1, dup2])
    b2 = _batch(b"\x02" * 32, [dup2, dup1])

    findings = detect_double_spends([b1, b2])
    assert len(findings) == 2
    # deterministic: sorted by leaf_hash ascending
    assert [f.leaf_hash for f in findings] == sorted([dup1, dup2])

    # run twice — identical result
    findings2 = detect_double_spends([b1, b2])
    assert [f.leaf_hash for f in findings2] == [f.leaf_hash for f in findings]
    # occurrences for each finding ordered by (batch index, leaf index)
    for f in findings:
        seq = [(occ.batch_id, occ.leaf_index) for occ in f.occurrences]
        assert seq == sorted(seq, key=lambda t: (t[0], t[1]))


def test_occurrence_records_provider_address():
    """A LeafOccurrence carries the provider_address so a downstream challenge
    assembler knows whom to challenge."""
    dup = b"\x11" * 32
    b1 = _batch(b"\x01" * 32, [dup], provider=PROV)
    b2 = _batch(b"\x02" * 32, [dup], provider=PROV)

    findings = detect_double_spends([b1, b2])
    assert all(occ.provider_address == PROV for occ in findings[0].occurrences)
