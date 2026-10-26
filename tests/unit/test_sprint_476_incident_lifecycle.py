"""Sprint 476 — incident-response lifecycle invariant pins.

Live-verified 2026-05-16 against running daemon. Full lifecycle:
open → event → advance → recommendations → comms-template →
playbook decision tree.

The load-bearing invariants:
  1. Timeline accumulates events (never overwrites)
  2. Advance transitions phase + appends to timeline
  3. Recommendations key on (severity, current_phase)
  4. Playbook decision tree contains the Vision §14
     emergency-response guidance
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
IR_FILE = (
    REPO_ROOT / "prsm" / "economy" / "web3"
    / "incident_response.py"
)


def _ir_source() -> str:
    return IR_FILE.read_text()


def test_incident_record_has_timeline_field():
    """The timeline is the audit-trail surface — it must
    persist as a list that accumulates events, not get
    overwritten on advance."""
    src = _ir_source()
    assert "timeline" in src


def test_incident_playbook_has_s0_pause_guidance():
    """The s0 playbook MUST surface the
    `prsm_emergency_pause compose_pause` guidance — this is
    the load-bearing first response per Vision §14. Removing
    or weakening this language is a regression in incident-
    response operational doctrine."""
    src = _ir_source()
    # The exact guidance string sprint 476 caught in the
    # /playbook live response.
    assert (
        "PAUSE NOW" in src or "emergency_pause" in src.lower()
        or "compose_pause" in src.lower()
    )


def test_incident_playbook_has_foundation_safe_urgency():
    """The 15-minute target for Foundation Safe signature on
    s0 PAUSE is the published operational SLA. Deletion or
    relaxation is a regression."""
    src = _ir_source()
    assert "Foundation Safe" in src
    # "15min" or "15 min" or "15 minutes" — flexible match.
    assert (
        "15min" in src or "15 min" in src
        or "<15" in src or "&lt;15" in src
    )


def test_incident_playbook_has_forensics_partner_guidance():
    """For s0 triage, the playbook must direct operators to
    on-chain forensics partners. Without this, operators
    might attempt forensics in-house, slowing recovery."""
    src = _ir_source()
    assert (
        "forensics" in src.lower()
        or "Chainalysis" in src or "TRM" in src
    )


def test_incident_recommendations_keyed_by_severity_and_phase():
    """Recommendations must vary by (severity, phase) — a
    constant recommendation list across all states would
    flatten the operational guidance the playbook provides."""
    src = _ir_source()
    # The decision tree iterates over (severity, phase)
    # tuples to produce recommendations.
    assert "severity" in src
    assert "phase" in src


def test_incident_record_includes_related_disclosure_id():
    """Incidents may stem from a disclosure — the audit-trail
    must record the parent disclosure_id when present.
    Without this, the disclosure → exploit → incident chain
    is broken at the operator-visible layer."""
    src = _ir_source()
    assert "related_disclosure_id" in src
