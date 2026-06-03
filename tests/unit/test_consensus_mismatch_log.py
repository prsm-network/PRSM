"""Sprint 957 (sp928 on-chain-slash groundwork) — persistent CONSENSUS_MISMATCH
evidence store.

The ComputeResultSampler catches a single-provider liar by re-execution and
builds CONSENSUS_MISMATCH evidence, but its challenge_sink was None — the
evidence was logged and lost. An autonomous on-chain slash is unsound from a
single node (StakeBond.slash is slasher-only; the open challengeReceipt rail
re-verifies a Merkle proof the off-chain single-provider path can't produce), so
this records the evidence in a bounded, opt-in-persistent, operator-reviewable
log instead — the corpus a future authority-gated on-chain bridge will consume.

Mirrors the audited SlashEventRing (slash_event_log.py): bounded deque,
PRSM_CONSENSUS_MISMATCH_LOG_DIR opt-in persistence, append/recent/count, plus an
async `record(evidence)` adapter matching the sampler's challenge_sink signature.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.consensus_mismatch_log import (
    ConsensusMismatchEntry,
    ConsensusMismatchLog,
)


def _evidence(job="job-1", accused="prov-bad", **kw):
    e = {
        "reason": "CONSENSUS_MISMATCH",
        "job_id": job,
        "accused_provider_id": accused,
        "accused_output_hash": "0xBAD",
        "majority_output_hash": "0xGOOD",
        "witness_provider_ids": ["w1", "w2"],
        "accused_bonded": True,
        "accused_stake_wei": 5_000_000_000_000_000_000,
        "accused_operator_address": "0xabc",
    }
    e.update(kw)
    return e


def test_append_and_recent():
    log = ConsensusMismatchLog()
    log.append(ConsensusMismatchEntry(
        timestamp=1.0, job_id="j1", accused_provider_id="p1",
        accused_output_hash="0x1", majority_output_hash="0x2",
        witness_provider_ids=("w",), accused_bonded=True,
        accused_stake_wei=1, accused_operator_address="0xa",
    ))
    out = log.recent(limit=10)
    assert len(out) == 1 and out[0].job_id == "j1"
    assert log.count() == 1


def test_maxlen_eviction():
    log = ConsensusMismatchLog(max_entries=3)
    for i in range(5):
        log.append(ConsensusMismatchEntry(
            timestamp=float(i), job_id=f"j{i}", accused_provider_id="p",
            accused_output_hash="0x1", majority_output_hash="0x2",
            witness_provider_ids=(), accused_bonded=False,
            accused_stake_wei=0, accused_operator_address=None,
        ))
    assert log.count() == 3
    # recent is newest-first; oldest two evicted.
    jobs = [e.job_id for e in log.recent(limit=10)]
    assert jobs == ["j4", "j3", "j2"]


def test_record_builds_entry_from_evidence():
    log = ConsensusMismatchLog()
    asyncio.run(log.record(_evidence()))
    assert log.count() == 1
    e = log.recent(limit=1)[0]
    assert e.accused_provider_id == "prov-bad"
    assert e.accused_bonded is True
    assert e.accused_stake_wei == 5_000_000_000_000_000_000
    assert e.accused_operator_address == "0xabc"
    assert e.witness_provider_ids == ("w1", "w2")


def test_record_rejects_non_consensus_mismatch():
    log = ConsensusMismatchLog()
    asyncio.run(log.record({"reason": "SOMETHING_ELSE", "job_id": "x"}))
    assert log.count() == 0  # ignored, not recorded


def test_record_never_raises_into_caller():
    """The sink must never raise into the sampler — malformed evidence is
    swallowed (logged), not propagated."""
    log = ConsensusMismatchLog()
    # Missing required keys → must not raise.
    asyncio.run(log.record({"reason": "CONSENSUS_MISMATCH"}))
    # A wholly bad payload type → must not raise.
    asyncio.run(log.record("not-a-dict"))  # type: ignore[arg-type]
    assert log.count() in (0, 1)  # tolerant; the point is no exception escaped


def test_persist_reload_roundtrip(tmp_path):
    d = tmp_path / "cm-evidence"
    log1 = ConsensusMismatchLog(persist_dir=d)
    asyncio.run(log1.record(_evidence(job="persisted-job", accused="prov-x")))
    # JSONL/JSON files written.
    assert any(d.glob("*.json"))
    # New instance reloads from disk.
    log2 = ConsensusMismatchLog(persist_dir=d)
    assert log2.count() == 1
    e = log2.recent(limit=1)[0]
    assert e.job_id == "persisted-job"
    assert e.accused_provider_id == "prov-x"
    assert e.accused_bonded is True


def test_persist_corrupt_file_skipped(tmp_path):
    d = tmp_path / "cm-evidence"
    d.mkdir()
    (d / "corrupt.json").write_text("{not json")
    # Must not raise; corrupt file skipped.
    log = ConsensusMismatchLog(persist_dir=d)
    assert log.count() == 0


# ── GET /admin/consensus-mismatch-evidence ─────────────────────────────────


def _node(*, with_log=True):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._webhook_log = None
    node._slash_event_log = None
    node._consensus_mismatch_log = ConsensusMismatchLog() if with_log else None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def test_endpoint_503_when_not_wired():
    node = _node(with_log=False)
    resp = _client(node).get("/admin/consensus-mismatch-evidence")
    assert resp.status_code == 503


def test_endpoint_returns_recent_entries():
    node = _node()
    asyncio.run(node._consensus_mismatch_log.record(_evidence(
        job="job-99", accused="prov-bad")))
    resp = _client(node).get("/admin/consensus-mismatch-evidence")
    body = resp.json()
    assert body["total"] == 1
    e = body["entries"][0]
    assert e["job_id"] == "job-99"
    assert e["accused_provider_id"] == "prov-bad"
    assert e["accused_bonded"] is True
    assert e["accused_stake_wei"] == 5_000_000_000_000_000_000


def test_endpoint_provider_filter():
    node = _node()
    asyncio.run(node._consensus_mismatch_log.record(_evidence(accused="0xMINE")))
    asyncio.run(node._consensus_mismatch_log.record(_evidence(accused="0xOTHER")))
    resp = _client(node).get(
        "/admin/consensus-mismatch-evidence?provider=0xMINE")
    body = resp.json()
    assert len(body["entries"]) == 1
    assert body["entries"][0]["accused_provider_id"] == "0xMINE"


def test_endpoint_invalid_limit_422():
    node = _node()
    resp = _client(node).get("/admin/consensus-mismatch-evidence?limit=0")
    assert resp.status_code == 422
