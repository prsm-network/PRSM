"""Sprint 925 — bound two compute-store DoS vectors (compute/inference review #6,#7).

#7 ReceiptStore.list_for_node_ids globbed + parsed EVERY receipt file on disk
per call, unbounded — and it's reachable via the PUBLIC, zero-cost
/devices/earnings endpoint, so a node with a large receipt dir suffers O(N)
CPU/memory per request (amplification). Fix: cap the results (default via
PRSM_RECEIPT_LIST_MAX, overridable per-call), logging when truncated.

#6 JobHistoryStore.response was an unbounded str persisted to disk — a single
job with a 1MB+ response bloats memory + the .json file. Fix: cap the response
field at put (default via PRSM_JOB_HISTORY_MAX_RESPONSE_BYTES) with a truncation
marker. (The disk-file COUNT bound is a separate, lower-urgency item — job
submission costs FTNS and each file is now size-capped; the get()-from-disk
durability feature means files are intentionally retained, so it needs a
prune-with-archive design, queued.)
"""
from __future__ import annotations

import pytest

from prsm.node.receipt_store import ReceiptStore
from prsm.node.job_history import JobHistoryStore, JobHistoryRecord, JobStatus


# ── #7 ReceiptStore.list_for_node_ids cap ────────────────────────────────


def _seed_receipts(store, n, node="n"):
    for i in range(n):
        store.put(f"job-{i}", {"job_id": f"job-{i}", "settler_node_id": node})


def test_list_caps_results_disk(tmp_path):
    s = ReceiptStore(persist_dir=tmp_path / "rcpts")
    _seed_receipts(s, 5)
    assert len(s.list_for_node_ids(["n"], max_results=2)) == 2


def test_list_default_returns_all_under_cap_disk(tmp_path):
    s = ReceiptStore(persist_dir=tmp_path / "rcpts")
    _seed_receipts(s, 5)
    assert len(s.list_for_node_ids(["n"])) == 5   # default cap >> 5


def test_list_caps_results_in_memory():
    s = ReceiptStore()   # no persist_dir → in-memory scan path
    _seed_receipts(s, 5)
    assert len(s.list_for_node_ids(["n"], max_results=3)) == 3


def test_list_empty_node_ids_still_empty(tmp_path):
    s = ReceiptStore(persist_dir=tmp_path / "rcpts")
    _seed_receipts(s, 3)
    assert s.list_for_node_ids([]) == []


# ── #6 JobHistory response field cap ─────────────────────────────────────


def _rec(job_id, response):
    return JobHistoryRecord(
        job_id=job_id, query="q", status=JobStatus.COMPLETED,
        started_at=0.0, response=response,
    )


def test_oversized_response_truncated_on_put(tmp_path, monkeypatch):
    monkeypatch.setenv("PRSM_JOB_HISTORY_MAX_RESPONSE_BYTES", "100")
    s = JobHistoryStore(persist_dir=tmp_path / "jh")
    s.put(_rec("j1", "x" * 5000))
    got = s.get("j1")
    assert len(got.response) < 5000
    assert "truncated" in got.response.lower()


def test_small_response_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("PRSM_JOB_HISTORY_MAX_RESPONSE_BYTES", "100")
    s = JobHistoryStore(persist_dir=tmp_path / "jh")
    s.put(_rec("j2", "short response"))
    assert s.get("j2").response == "short response"


def test_none_response_is_fine(tmp_path):
    s = JobHistoryStore(persist_dir=tmp_path / "jh")
    s.put(_rec("j3", None))
    assert s.get("j3").response is None
