"""Sprint 300 — disclosure intake HTTP + MCP surface.

POST /admin/disclosure/submit           — receive new disclosure
GET  /admin/disclosure                  — list (filtered)
GET  /admin/disclosure/{id}             — single record
POST /admin/disclosure/{id}/update      — workflow transition
POST /admin/disclosure/{id}/compose-payout — Safe-uploadable
                                              payout tx
POST /admin/disclosure/{id}/record-payout-tx — close audit

The submit endpoint is intentionally accessible (researchers
may be anonymous). Workflow + payout endpoints are admin-only;
payout itself is composer-only (Foundation Safe gates execution).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.economy.web3.disclosure_intake import (
    DisclosureIntake, DisclosureSeverity, DisclosureStatus,
)
from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_disclosure,
)
from prsm.node.api import create_api_app


def _client(intake=None, ftns_token_address=None,
            chain_id=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._disclosure_intake = intake
    node._disclosure_ftns_token_address = (
        ftns_token_address
    )
    node._disclosure_chain_id = chain_id
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


# ── HTTP: submit ─────────────────────────────────────────


def test_submit_503_when_unwired():
    resp = _client(None).post(
        "/admin/disclosure/submit",
        json={
            "severity": "high",
            "summary": "x",
            "affected_contracts": ["x"],
            "researcher_contact": "alice@example.com",
            "details": "x",
        },
    )
    assert resp.status_code == 503


def test_submit_happy_path():
    intake = DisclosureIntake()
    resp = _client(intake).post(
        "/admin/disclosure/submit",
        json={
            "severity": "high",
            "summary": "reentrancy in claim()",
            "affected_contracts": ["royalty_distributor"],
            "researcher_contact": "alice@example.com",
            "details": "POC here",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "received"
    assert body["disclosure_id"]
    assert intake.count() == 1


def test_submit_422_missing_severity():
    resp = _client(DisclosureIntake()).post(
        "/admin/disclosure/submit",
        json={
            "summary": "x",
            "affected_contracts": ["x"],
            "researcher_contact": "x",
        },
    )
    assert resp.status_code == 422


def test_submit_422_invalid_severity():
    resp = _client(DisclosureIntake()).post(
        "/admin/disclosure/submit",
        json={
            "severity": "EXPLODE",
            "summary": "x",
            "affected_contracts": ["x"],
            "researcher_contact": "x",
        },
    )
    assert resp.status_code == 422


def test_submit_422_empty_summary():
    resp = _client(DisclosureIntake()).post(
        "/admin/disclosure/submit",
        json={
            "severity": "high",
            "summary": "",
            "affected_contracts": ["x"],
            "researcher_contact": "x",
        },
    )
    assert resp.status_code == 422


# ── HTTP: list + get ─────────────────────────────────────


def test_list_503_when_unwired():
    resp = _client(None).get("/admin/disclosure")
    assert resp.status_code == 503


def test_list_empty():
    resp = _client(DisclosureIntake()).get(
        "/admin/disclosure",
    )
    body = resp.json()
    assert body["records"] == []
    assert body["count"] == 0


def test_list_populated_newest_first():
    intake = DisclosureIntake()
    intake.submit(
        severity=DisclosureSeverity.LOW,
        summary="first",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
        timestamp=100.0,
    )
    intake.submit(
        severity=DisclosureSeverity.HIGH,
        summary="second",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
        timestamp=200.0,
    )
    resp = _client(intake).get("/admin/disclosure")
    body = resp.json()
    assert body["records"][0]["summary"] == "second"


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
        summary="crit",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
    )
    resp = _client(intake).get(
        "/admin/disclosure?severity=critical",
    )
    body = resp.json()
    assert body["count"] == 1
    assert body["records"][0]["summary"] == "crit"


def test_get_one_404_unknown():
    resp = _client(DisclosureIntake()).get(
        "/admin/disclosure/no-such-id",
    )
    assert resp.status_code == 404


def test_get_one_happy_path():
    intake = DisclosureIntake()
    r = intake.submit(
        severity=DisclosureSeverity.HIGH,
        summary="x",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
    )
    resp = _client(intake).get(
        f"/admin/disclosure/{r.disclosure_id}",
    )
    body = resp.json()
    assert body["disclosure_id"] == r.disclosure_id


# ── HTTP: status transition ──────────────────────────────


def test_update_status_503_when_unwired():
    resp = _client(None).post(
        "/admin/disclosure/x/update",
        json={"new_status": "triaged"},
    )
    assert resp.status_code == 503


def test_update_status_happy_path():
    intake = DisclosureIntake()
    r = intake.submit(
        severity=DisclosureSeverity.HIGH,
        summary="x",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
    )
    resp = _client(intake).post(
        f"/admin/disclosure/{r.disclosure_id}/update",
        json={
            "new_status": "triaged",
            "triage_notes": "investigating",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "triaged"
    assert "investigating" in body["triage_notes"]


def test_update_status_404_unknown():
    resp = _client(DisclosureIntake()).post(
        "/admin/disclosure/no-such-id/update",
        json={"new_status": "triaged"},
    )
    assert resp.status_code == 404


def test_update_status_422_invalid_status():
    intake = DisclosureIntake()
    r = intake.submit(
        severity=DisclosureSeverity.HIGH,
        summary="x",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
    )
    resp = _client(intake).post(
        f"/admin/disclosure/{r.disclosure_id}/update",
        json={"new_status": "MADE_UP"},
    )
    assert resp.status_code == 422


def test_update_status_422_invalid_transition():
    intake = DisclosureIntake()
    r = intake.submit(
        severity=DisclosureSeverity.HIGH,
        summary="x",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
    )
    # RECEIVED → REJECTED is invalid (must go through TRIAGED
    # first per workflow)... actually that's not what
    # update_status enforces — terminal states from any
    # transition are allowed. Let me use the back-to-RECEIVED
    # case which IS rejected.
    intake.update_status(
        r.disclosure_id, DisclosureStatus.TRIAGED,
    )
    resp = _client(intake).post(
        f"/admin/disclosure/{r.disclosure_id}/update",
        json={"new_status": "received"},
    )
    assert resp.status_code == 422


# ── HTTP: compose-payout ─────────────────────────────────


def _awarded_disclosure(intake):
    r = intake.submit(
        severity=DisclosureSeverity.CRITICAL,
        summary="critical exploit",
        affected_contracts=["royalty_distributor"],
        researcher_contact="alice@example.com",
        details="x",
    )
    intake.update_status(
        r.disclosure_id, DisclosureStatus.TRIAGED,
    )
    intake.update_status(
        r.disclosure_id, DisclosureStatus.CONFIRMED,
    )
    intake.update_status(
        r.disclosure_id,
        DisclosureStatus.AWARDED,
        payout_ftns=1_500_000,
    )
    return r


def test_compose_payout_503_when_unwired():
    resp = _client(None).post(
        "/admin/disclosure/x/compose-payout",
        json={"recipient": "0x" + "ab" * 20},
    )
    assert resp.status_code == 503


def test_compose_payout_503_without_token_address():
    """If the operator hasn't wired the FTNS token address,
    payout composition can't proceed."""
    intake = DisclosureIntake()
    r = _awarded_disclosure(intake)
    resp = _client(
        intake, ftns_token_address=None,
    ).post(
        f"/admin/disclosure/{r.disclosure_id}/compose-payout",
        json={"recipient": "0x" + "ab" * 20},
    )
    assert resp.status_code == 503


def test_compose_payout_happy_path():
    intake = DisclosureIntake()
    r = _awarded_disclosure(intake)
    resp = _client(
        intake,
        ftns_token_address="0x" + "11" * 20,
        chain_id=8453,
    ).post(
        f"/admin/disclosure/{r.disclosure_id}/compose-payout",
        json={"recipient": "0x" + "ab" * 20},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["action"] == "disclosure_payout"
    assert body["recipient"] == "0x" + "ab" * 20
    assert body["amount_ftns"] == 1_500_000


def test_compose_payout_422_invalid_recipient():
    intake = DisclosureIntake()
    r = _awarded_disclosure(intake)
    resp = _client(
        intake,
        ftns_token_address="0x" + "11" * 20,
    ).post(
        f"/admin/disclosure/{r.disclosure_id}/compose-payout",
        json={"recipient": "not-an-address"},
    )
    assert resp.status_code == 422


def test_compose_payout_422_non_awarded():
    intake = DisclosureIntake()
    r = intake.submit(
        severity=DisclosureSeverity.HIGH,
        summary="x",
        affected_contracts=["x"],
        researcher_contact="x",
        details="x",
    )
    resp = _client(
        intake,
        ftns_token_address="0x" + "11" * 20,
    ).post(
        f"/admin/disclosure/{r.disclosure_id}/compose-payout",
        json={"recipient": "0x" + "ab" * 20},
    )
    assert resp.status_code == 422


# ── HTTP: record-payout-tx ───────────────────────────────


def test_record_payout_tx_happy_path():
    intake = DisclosureIntake()
    r = _awarded_disclosure(intake)
    resp = _client(intake).post(
        f"/admin/disclosure/{r.disclosure_id}/record-payout-tx",
        json={"tx_hash": "0xdeadbeef"},
    )
    assert resp.status_code == 200
    assert intake.get(r.disclosure_id).payout_tx_hash == (
        "0xdeadbeef"
    )


def test_record_payout_tx_404_unknown():
    resp = _client(DisclosureIntake()).post(
        "/admin/disclosure/no-such-id/record-payout-tx",
        json={"tx_hash": "0xdeadbeef"},
    )
    assert resp.status_code == 404


def test_record_payout_tx_422_empty_hash():
    intake = DisclosureIntake()
    r = _awarded_disclosure(intake)
    resp = _client(intake).post(
        f"/admin/disclosure/{r.disclosure_id}/record-payout-tx",
        json={"tx_hash": ""},
    )
    assert resp.status_code == 422


# ── MCP tool ─────────────────────────────────────────────


def test_mcp_tool_registered():
    assert "prsm_disclosure" in TOOL_HANDLERS


@pytest.mark.asyncio
async def test_mcp_missing_action():
    r = await handle_prsm_disclosure({})
    assert "action" in r.lower()


@pytest.mark.asyncio
async def test_mcp_unknown_action():
    r = await handle_prsm_disclosure(
        {"action": "explode"},
    )
    assert "must be" in r.lower()


@pytest.mark.asyncio
async def test_mcp_list_renders_table():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "records": [
                {
                    "disclosure_id": "abcdef0123456789",
                    "timestamp": 100.0,
                    "severity": "critical",
                    "summary": "reentrancy in claim()",
                    "affected_contracts": ["royalty_distributor"],
                    "researcher_contact": "alice@example.com",
                    "status": "awarded",
                    "details_b64": "",
                    "triage_notes": "",
                    "payout_ftns": 1_000_000,
                    "payout_tx_hash": "0xabc",
                },
            ],
            "count": 1,
        }),
    ):
        r = await handle_prsm_disclosure(
            {"action": "list"},
        )
    assert "abcdef01" in r  # short id
    assert "critical" in r.lower()
    assert "reentrancy" in r
    assert "awarded" in r.lower()


@pytest.mark.asyncio
async def test_mcp_list_filter_by_severity():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "records": [], "count": 0,
        }),
    ) as mock_call:
        await handle_prsm_disclosure({
            "action": "list",
            "severity": "critical",
        })
    path = mock_call.await_args[0][1]
    assert "severity=critical" in path


@pytest.mark.asyncio
async def test_mcp_lookup_renders_detail():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "disclosure_id": "d-1",
            "timestamp": 100.0,
            "severity": "high",
            "summary": "overflow",
            "affected_contracts": ["stake_bond"],
            "researcher_contact": "alice@example.com",
            "status": "triaged",
            "details_b64": "ZGV0YWlscw==",
            "triage_notes": "investigating",
            "payout_ftns": 0,
            "payout_tx_hash": None,
        }),
    ) as mock_call:
        r = await handle_prsm_disclosure({
            "action": "lookup",
            "disclosure_id": "d-1",
        })
    args = mock_call.await_args[0]
    assert args[1] == "/admin/disclosure/d-1"
    assert "d-1" in r
    assert "overflow" in r
    assert "stake_bond" in r
    assert "triaged" in r.lower()


@pytest.mark.asyncio
async def test_mcp_submit_happy_path():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "disclosure_id": "new-id-123",
            "timestamp": 100.0,
            "severity": "high",
            "summary": "x",
            "affected_contracts": [],
            "researcher_contact": "x",
            "status": "received",
            "details_b64": "",
            "triage_notes": "",
            "payout_ftns": 0,
            "payout_tx_hash": None,
        }),
    ) as mock_call:
        r = await handle_prsm_disclosure({
            "action": "submit",
            "severity": "high",
            "summary": "reentrancy",
            "affected_contracts": ["royalty_distributor"],
            "researcher_contact": "alice@example.com",
            "details": "POC",
        })
    args = mock_call.await_args[0]
    assert args[0] == "POST"
    assert args[1] == "/admin/disclosure/submit"
    assert "new-id-123" in r
    assert "received" in r.lower()


@pytest.mark.asyncio
async def test_mcp_submit_requires_severity():
    r = await handle_prsm_disclosure({
        "action": "submit",
        "summary": "x",
        "affected_contracts": ["x"],
        "researcher_contact": "x",
    })
    assert "severity" in r.lower()


@pytest.mark.asyncio
async def test_mcp_update_action():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "disclosure_id": "d-1",
            "timestamp": 100.0,
            "severity": "high",
            "summary": "x",
            "affected_contracts": [],
            "researcher_contact": "x",
            "status": "triaged",
            "details_b64": "",
            "triage_notes": "investigating",
            "payout_ftns": 0,
            "payout_tx_hash": None,
        }),
    ) as mock_call:
        r = await handle_prsm_disclosure({
            "action": "update",
            "disclosure_id": "d-1",
            "new_status": "triaged",
            "triage_notes": "investigating",
        })
    args = mock_call.await_args[0]
    assert args[1] == "/admin/disclosure/d-1/update"
    assert "triaged" in r.lower()


@pytest.mark.asyncio
async def test_mcp_compose_payout_action():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "action": "disclosure_payout",
            "disclosure_id": "d-1",
            "to": "0x" + "11" * 20,
            "data": "0xa9059cbb" + "0" * 128,
            "value": "0",
            "recipient": "0x" + "ab" * 20,
            "amount_wei": str(1_500_000 * (10 ** 18)),
            "amount_ftns": 1_500_000,
            "severity": "critical",
            "summary": "critical exploit",
            "chain_id": 8453,
            "warning": "DESTRUCTIVE...",
            "explorer_url": (
                "https://basescan.org/address/0x"
                + "11" * 20
            ),
            "instructions": "1) Open Safe UI...",
        }),
    ) as mock_call:
        r = await handle_prsm_disclosure({
            "action": "compose_payout",
            "disclosure_id": "d-1",
            "recipient": "0x" + "ab" * 20,
        })
    args = mock_call.await_args[0]
    assert args[1] == "/admin/disclosure/d-1/compose-payout"
    # Renders ⚠ block + WARNING + payload + recipient +
    # amount in FTNS
    assert "WARNING" in r.upper() or "destructive" in r.lower()
    assert "1,500,000" in r or "1500000" in r
    assert "0x" + "ab" * 20 in r


@pytest.mark.asyncio
async def test_mcp_record_payout_tx_action():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "disclosure_id": "d-1",
            "timestamp": 100.0,
            "severity": "critical",
            "summary": "x",
            "affected_contracts": [],
            "researcher_contact": "x",
            "status": "awarded",
            "details_b64": "",
            "triage_notes": "",
            "payout_ftns": 1_500_000,
            "payout_tx_hash": "0xdeadbeef",
        }),
    ) as mock_call:
        r = await handle_prsm_disclosure({
            "action": "record_payout_tx",
            "disclosure_id": "d-1",
            "tx_hash": "0xdeadbeef",
        })
    args = mock_call.await_args[0]
    assert args[1] == (
        "/admin/disclosure/d-1/record-payout-tx"
    )
    assert "0xdeadbeef" in r
