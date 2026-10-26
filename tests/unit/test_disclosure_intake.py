"""Sprint 300 — DisclosureIntake + payout composer.

PRSM does not run a bug bounty program and does not pay for
vulnerability reports. The payout composer was built for a
proposed program (Vision §14 mitigation item 3) that was never
launched.

This module gives operators a DIRECT-CONTACT path for
security researchers: sensitive findings, anonymous
submissions, and time-critical disclosures during active
incidents.

Scope this sprint:
  DisclosureSeverity — severity bands
  DisclosureStatus — workflow: received → triaged →
                     confirmed|rejected → awarded
  DisclosureRecord — persistent record (filesystem-backed,
                     same pattern as sprint 272 TakedownNoticeRing)
  DisclosureIntake — submit / update_status / list / get
  compose_disclosure_payout_tx — Safe-uploadable ERC-20 transfer
                                 (composer-only, mirrors
                                 sprint 299 insurance-fund
                                 recovery composer)
"""
from __future__ import annotations

import json

import pytest

from prsm.economy.web3.disclosure_intake import (
    DEFAULT_PAYOUT_BANDS_FTNS,
    DisclosureIntake,
    DisclosureRecord,
    DisclosureSeverity,
    DisclosureStatus,
    compose_disclosure_payout_tx,
)


# ── Severity / status enums ──────────────────────────────


def test_severity_values():
    """Severity bands."""
    assert DisclosureSeverity.CRITICAL.value == "critical"
    assert DisclosureSeverity.HIGH.value == "high"
    assert DisclosureSeverity.MEDIUM.value == "medium"
    assert DisclosureSeverity.LOW.value == "low"
    assert (
        DisclosureSeverity.INFORMATIONAL.value
        == "informational"
    )


def test_status_values():
    assert DisclosureStatus.RECEIVED.value == "received"
    assert DisclosureStatus.TRIAGED.value == "triaged"
    assert DisclosureStatus.CONFIRMED.value == "confirmed"
    assert DisclosureStatus.REJECTED.value == "rejected"
    assert DisclosureStatus.AWARDED.value == "awarded"
    assert DisclosureStatus.DUPLICATE.value == "duplicate"
    assert (
        DisclosureStatus.OUT_OF_SCOPE.value == "out_of_scope"
    )


def test_default_payout_bands_in_ftns():
    """Proposed bands (proposal only; never activated) map
    severity → FTNS amount."""
    bands = DEFAULT_PAYOUT_BANDS_FTNS
    assert bands[DisclosureSeverity.CRITICAL] >= 1_000_000
    assert (
        bands[DisclosureSeverity.HIGH]
        < bands[DisclosureSeverity.CRITICAL]
    )
    assert (
        bands[DisclosureSeverity.MEDIUM]
        < bands[DisclosureSeverity.HIGH]
    )
    assert (
        bands[DisclosureSeverity.LOW]
        < bands[DisclosureSeverity.MEDIUM]
    )
    assert bands[DisclosureSeverity.INFORMATIONAL] >= 0


# ── DisclosureRecord round-trip ──────────────────────────


def test_record_to_dict_round_trip():
    r = DisclosureRecord(
        disclosure_id="d-1",
        timestamp=100.0,
        severity=DisclosureSeverity.HIGH,
        summary="reentrancy in RoyaltyDistributor.claim()",
        affected_contracts=["royalty_distributor"],
        researcher_contact="alice@example.com",
        status=DisclosureStatus.RECEIVED,
        details_b64="dGVzdA==",
        triage_notes="",
        payout_ftns=0,
        payout_tx_hash=None,
    )
    d = r.to_dict()
    restored = DisclosureRecord.from_dict(d)
    assert restored == r


def test_record_severity_enum_round_trip():
    """Severity persists as enum value string."""
    r = DisclosureRecord(
        disclosure_id="d-2",
        timestamp=100.0,
        severity=DisclosureSeverity.CRITICAL,
        summary="x",
        affected_contracts=[],
        researcher_contact="x",
        status=DisclosureStatus.AWARDED,
    )
    d = r.to_dict()
    assert d["severity"] == "critical"
    assert d["status"] == "awarded"


# ── DisclosureIntake.submit ──────────────────────────────


def test_submit_assigns_id_and_timestamp():
    intake = DisclosureIntake()
    r = intake.submit(
        severity=DisclosureSeverity.HIGH,
        summary="overflow in stake amount",
        affected_contracts=["stake_bond"],
        researcher_contact="alice@example.com",
        details="long writeup",
    )
    assert r.disclosure_id
    assert r.timestamp > 0
    assert r.status == DisclosureStatus.RECEIVED


def test_submit_validates_required_fields():
    intake = DisclosureIntake()
    with pytest.raises(ValueError):
        intake.submit(
            severity=DisclosureSeverity.HIGH,
            summary="",  # empty
            affected_contracts=["x"],
            researcher_contact="x",
            details="x",
        )
    with pytest.raises(ValueError):
        intake.submit(
            severity=DisclosureSeverity.HIGH,
            summary="x",
            affected_contracts=["x"],
            researcher_contact="",  # empty
            details="x",
        )


def test_submit_caps_details_size():
    """Details may be large but unbounded uploads are a DoS
    vector. v1 caps at 256KB."""
    intake = DisclosureIntake()
    huge = "x" * (1024 * 1024)  # 1MB
    with pytest.raises(ValueError, match="details"):
        intake.submit(
            severity=DisclosureSeverity.HIGH,
            summary="x",
            affected_contracts=["x"],
            researcher_contact="x",
            details=huge,
        )


def test_submit_caps_summary_size():
    intake = DisclosureIntake()
    huge_summary = "x" * 10_000
    with pytest.raises(ValueError, match="summary"):
        intake.submit(
            severity=DisclosureSeverity.HIGH,
            summary=huge_summary,
            affected_contracts=["x"],
            researcher_contact="x",
            details="x",
        )


def test_submit_caps_affected_contracts_count():
    intake = DisclosureIntake()
    too_many = [f"contract_{i}" for i in range(100)]
    with pytest.raises(ValueError, match="affected_contracts"):
        intake.submit(
            severity=DisclosureSeverity.HIGH,
            summary="x",
            affected_contracts=too_many,
            researcher_contact="x",
            details="x",
        )


# ── Status workflow ──────────────────────────────────────


def test_update_status_happy_path():
    intake = DisclosureIntake()
    r = intake.submit(
        severity=DisclosureSeverity.MEDIUM,
        summary="x",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
    )
    updated = intake.update_status(
        r.disclosure_id,
        new_status=DisclosureStatus.TRIAGED,
        triage_notes="legitimate; investigating",
    )
    assert updated.status == DisclosureStatus.TRIAGED
    assert "investigating" in updated.triage_notes


def test_update_status_invalid_transition_to_received():
    """Once moved out of RECEIVED, can't move back to
    RECEIVED. Workflow integrity."""
    intake = DisclosureIntake()
    r = intake.submit(
        severity=DisclosureSeverity.HIGH,
        summary="x",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
    )
    intake.update_status(
        r.disclosure_id, new_status=DisclosureStatus.TRIAGED,
    )
    with pytest.raises(ValueError, match="transition"):
        intake.update_status(
            r.disclosure_id,
            new_status=DisclosureStatus.RECEIVED,
        )


def test_update_status_terminal_states_cannot_change():
    """AWARDED, REJECTED, DUPLICATE, OUT_OF_SCOPE are
    terminal. Once set, status can't be moved."""
    intake = DisclosureIntake()
    r = intake.submit(
        severity=DisclosureSeverity.HIGH,
        summary="x",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
    )
    intake.update_status(
        r.disclosure_id,
        new_status=DisclosureStatus.TRIAGED,
    )
    intake.update_status(
        r.disclosure_id,
        new_status=DisclosureStatus.REJECTED,
    )
    with pytest.raises(ValueError, match="terminal"):
        intake.update_status(
            r.disclosure_id,
            new_status=DisclosureStatus.CONFIRMED,
        )


def test_update_status_unknown_id_raises():
    intake = DisclosureIntake()
    with pytest.raises(ValueError, match="not found"):
        intake.update_status(
            "no-such-id",
            new_status=DisclosureStatus.TRIAGED,
        )


def test_award_records_payout_amount():
    """When status transitions to AWARDED, the payout_ftns
    field captures the amount awarded — for audit trail +
    later payout-composer use."""
    intake = DisclosureIntake()
    r = intake.submit(
        severity=DisclosureSeverity.CRITICAL,
        summary="x",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
    )
    intake.update_status(
        r.disclosure_id,
        new_status=DisclosureStatus.TRIAGED,
    )
    intake.update_status(
        r.disclosure_id,
        new_status=DisclosureStatus.CONFIRMED,
    )
    awarded = intake.update_status(
        r.disclosure_id,
        new_status=DisclosureStatus.AWARDED,
        payout_ftns=1_500_000,
    )
    assert awarded.payout_ftns == 1_500_000


# ── Listing + lookup ─────────────────────────────────────


def test_list_returns_newest_first():
    intake = DisclosureIntake()
    r1 = intake.submit(
        severity=DisclosureSeverity.LOW,
        summary="first",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
        timestamp=100.0,
    )
    r2 = intake.submit(
        severity=DisclosureSeverity.HIGH,
        summary="second",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
        timestamp=200.0,
    )
    listed = intake.list()
    assert listed[0].disclosure_id == r2.disclosure_id
    assert listed[1].disclosure_id == r1.disclosure_id


def test_list_filter_by_severity():
    intake = DisclosureIntake()
    intake.submit(
        severity=DisclosureSeverity.LOW,
        summary="low",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
    )
    intake.submit(
        severity=DisclosureSeverity.CRITICAL,
        summary="critical",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
    )
    filtered = intake.list(
        severity=DisclosureSeverity.CRITICAL,
    )
    assert len(filtered) == 1
    assert filtered[0].summary == "critical"


def test_list_filter_by_status():
    intake = DisclosureIntake()
    r1 = intake.submit(
        severity=DisclosureSeverity.HIGH,
        summary="a",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
    )
    intake.submit(
        severity=DisclosureSeverity.HIGH,
        summary="b",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
    )
    intake.update_status(
        r1.disclosure_id,
        new_status=DisclosureStatus.TRIAGED,
    )
    received = intake.list(
        status=DisclosureStatus.RECEIVED,
    )
    assert len(received) == 1
    triaged = intake.list(
        status=DisclosureStatus.TRIAGED,
    )
    assert len(triaged) == 1


def test_get_by_id_happy_path():
    intake = DisclosureIntake()
    r = intake.submit(
        severity=DisclosureSeverity.MEDIUM,
        summary="x",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
    )
    fetched = intake.get(r.disclosure_id)
    assert fetched.disclosure_id == r.disclosure_id


def test_get_unknown_id_returns_none():
    intake = DisclosureIntake()
    assert intake.get("no-such-id") is None


def test_count():
    intake = DisclosureIntake()
    assert intake.count() == 0
    intake.submit(
        severity=DisclosureSeverity.LOW,
        summary="x",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
    )
    assert intake.count() == 1


# ── Filesystem persistence ───────────────────────────────


def test_persistence_round_trip(tmp_path):
    intake1 = DisclosureIntake(persist_dir=tmp_path)
    r = intake1.submit(
        severity=DisclosureSeverity.HIGH,
        summary="reentrancy",
        affected_contracts=["royalty_distributor"],
        researcher_contact="alice@example.com",
        details="full POC",
        timestamp=100.0,
    )
    intake2 = DisclosureIntake(persist_dir=tmp_path)
    assert intake2.count() == 1
    fetched = intake2.get(r.disclosure_id)
    assert fetched.summary == "reentrancy"


def test_persistence_corrupt_file_fail_soft(tmp_path):
    (tmp_path / "garbage.json").write_text("{not valid json")
    intake = DisclosureIntake(persist_dir=tmp_path)
    assert intake.count() == 0


def test_persistence_partial_corrupt_skips_only_bad(tmp_path):
    # One valid record + one corrupt file → 1 record loaded
    valid = DisclosureRecord(
        disclosure_id="good",
        timestamp=100.0,
        severity=DisclosureSeverity.LOW,
        summary="x",
        affected_contracts=[],
        researcher_contact="x",
        status=DisclosureStatus.RECEIVED,
    )
    (tmp_path / "good.json").write_text(
        json.dumps(valid.to_dict()),
    )
    (tmp_path / "bad.json").write_text("not json")
    intake = DisclosureIntake(persist_dir=tmp_path)
    assert intake.count() == 1
    assert intake.get("good") is not None


# ── from_env factory ─────────────────────────────────────


def test_from_env_no_dir(monkeypatch):
    monkeypatch.delenv(
        "PRSM_DISCLOSURE_INTAKE_DIR", raising=False,
    )
    intake = DisclosureIntake.from_env()
    assert intake._persist_dir is None


def test_from_env_with_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "PRSM_DISCLOSURE_INTAKE_DIR", str(tmp_path),
    )
    intake = DisclosureIntake.from_env()
    assert intake._persist_dir == tmp_path


# ── compose_disclosure_payout_tx ─────────────────────────


def test_compose_payout_happy_path():
    intake = DisclosureIntake()
    r = intake.submit(
        severity=DisclosureSeverity.CRITICAL,
        summary="x",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
    )
    intake.update_status(
        r.disclosure_id,
        new_status=DisclosureStatus.TRIAGED,
    )
    intake.update_status(
        r.disclosure_id,
        new_status=DisclosureStatus.CONFIRMED,
    )
    intake.update_status(
        r.disclosure_id,
        new_status=DisclosureStatus.AWARDED,
        payout_ftns=1_500_000,
    )
    tx = compose_disclosure_payout_tx(
        intake=intake,
        disclosure_id=r.disclosure_id,
        recipient="0x" + "ab" * 20,
        ftns_token_address="0x" + "11" * 20,
        chain_id=8453,
    )
    assert tx["action"] == "disclosure_payout"
    assert tx["disclosure_id"] == r.disclosure_id
    assert tx["recipient"] == "0x" + "ab" * 20
    # amount in wei = payout_ftns × 10^18
    assert tx["amount_wei"] == str(
        1_500_000 * (10 ** 18),
    )
    # ERC-20 transfer to token contract
    assert tx["to"] == "0x" + "11" * 20
    assert tx["data"].startswith("0xa9059cbb")
    assert "warning" in tx
    assert "basescan" in tx["explorer_url"].lower()


def test_compose_payout_rejects_non_awarded():
    """Can't compose payout for a disclosure that isn't
    AWARDED — defends premature payment."""
    intake = DisclosureIntake()
    r = intake.submit(
        severity=DisclosureSeverity.HIGH,
        summary="x",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
    )
    with pytest.raises(ValueError, match="AWARDED"):
        compose_disclosure_payout_tx(
            intake=intake,
            disclosure_id=r.disclosure_id,
            recipient="0x" + "ab" * 20,
            ftns_token_address="0x" + "11" * 20,
        )


def test_compose_payout_unknown_disclosure_raises():
    intake = DisclosureIntake()
    with pytest.raises(ValueError, match="not found"):
        compose_disclosure_payout_tx(
            intake=intake,
            disclosure_id="no-such-id",
            recipient="0x" + "ab" * 20,
            ftns_token_address="0x" + "11" * 20,
        )


def test_compose_payout_zero_amount_rejected():
    """payout_ftns=0 (set by the status update) means no
    payout — composer should refuse."""
    intake = DisclosureIntake()
    r = intake.submit(
        severity=DisclosureSeverity.LOW,
        summary="x",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
    )
    intake.update_status(
        r.disclosure_id,
        new_status=DisclosureStatus.TRIAGED,
    )
    intake.update_status(
        r.disclosure_id,
        new_status=DisclosureStatus.CONFIRMED,
    )
    intake.update_status(
        r.disclosure_id,
        new_status=DisclosureStatus.AWARDED,
        payout_ftns=0,  # zero
    )
    with pytest.raises(
        ValueError, match="payout_ftns=0|nothing to pay out",
    ):
        compose_disclosure_payout_tx(
            intake=intake,
            disclosure_id=r.disclosure_id,
            recipient="0x" + "ab" * 20,
            ftns_token_address="0x" + "11" * 20,
        )


def test_compose_payout_validates_recipient():
    intake = DisclosureIntake()
    r = intake.submit(
        severity=DisclosureSeverity.CRITICAL,
        summary="x",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
    )
    intake.update_status(
        r.disclosure_id, DisclosureStatus.TRIAGED,
    )
    intake.update_status(
        r.disclosure_id, DisclosureStatus.CONFIRMED,
    )
    intake.update_status(
        r.disclosure_id, DisclosureStatus.AWARDED,
        payout_ftns=1_500_000,
    )
    with pytest.raises(ValueError):
        compose_disclosure_payout_tx(
            intake=intake,
            disclosure_id=r.disclosure_id,
            recipient="not-an-address",
            ftns_token_address="0x" + "11" * 20,
        )


def test_compose_payout_records_tx_hash_field_writable():
    """After Safe execution, operator can record the
    on-chain tx hash for audit."""
    intake = DisclosureIntake()
    r = intake.submit(
        severity=DisclosureSeverity.HIGH,
        summary="x",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
    )
    intake.update_status(
        r.disclosure_id, DisclosureStatus.TRIAGED,
    )
    intake.update_status(
        r.disclosure_id, DisclosureStatus.CONFIRMED,
    )
    intake.update_status(
        r.disclosure_id, DisclosureStatus.AWARDED,
        payout_ftns=100_000,
    )
    updated = intake.record_payout_tx(
        r.disclosure_id, tx_hash="0xdeadbeef",
    )
    assert updated.payout_tx_hash == "0xdeadbeef"


# ── Persistence after status change ──────────────────────


def test_status_update_persists_to_disk(tmp_path):
    intake1 = DisclosureIntake(persist_dir=tmp_path)
    r = intake1.submit(
        severity=DisclosureSeverity.MEDIUM,
        summary="x",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
    )
    intake1.update_status(
        r.disclosure_id,
        new_status=DisclosureStatus.TRIAGED,
        triage_notes="investigating",
    )
    # Re-load: status persisted
    intake2 = DisclosureIntake(persist_dir=tmp_path)
    fetched = intake2.get(r.disclosure_id)
    assert fetched.status == DisclosureStatus.TRIAGED
    assert "investigating" in fetched.triage_notes
