"""Sprint 1149 — operator-facing settlement-audit findings surface.

These tests pin the NEW operator surface that closes the "fraud detected but
operationally invisible" gap (settlement_audit_wiring.run_once_tick only logged a
DEBUG repr). The new ``settlement_audit_report`` module is PURE; the new
``SettlementAuditFindingsStore`` mirrors PublishedBatchStore (bounded / atomic /
never-raises / dedups); the scheduler change is ADDITIVE (idle ticks unchanged,
cursor + retry semantics unchanged, never broadcasts).

NO network. Fixtures build an ``AuditRunResult`` with findings across ALL 5 classes
(incl. a §7 bound+proven AND an UNBOUND record) offline.
"""
from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

import pytest

from prsm.settlement.consensus_mismatch_detector import ConsensusMismatchFinding
from prsm.settlement.double_spend_detector import DoubleSpendFinding, LeafOccurrence
from prsm.settlement.expired_receipt_detector import ExpiredReceiptFinding
from prsm.settlement.settlement_audit_engine import (
    ConsensusMismatchAuditFinding,
    DoubleSpendAuditFinding,
    ExpiredReceiptAuditFinding,
    InferenceReceiptAuditFinding,
    InvalidSignatureAuditFinding,
)
from prsm.settlement.settlement_audit_loop import AuditRunResult
from prsm.settlement.settlement_audit_report import (
    AuditFindingRecord,
    AuditFindingsSummary,
    SettlementAuditFindingsStore,
    finding_key,
    summarize_audit_result,
)
from prsm.settlement.settlement_audit_wiring import (
    BlockCursor,
    SettlementAuditScheduler,
)


# ── fixtures ──────────────────────────────────────────────────────────


_BID_DS = bytes.fromhex("aa") * 32
_BID_DS2 = bytes.fromhex("bb") * 32
_BID_SIG = bytes.fromhex("cc") * 32
_BID_CM_MIN = bytes.fromhex("dd") * 32
_BID_CM_MAJ = bytes.fromhex("ee") * 32
_BID_EXP = bytes.fromhex("ff") * 32
_BID_S7 = bytes.fromhex("17") * 32


def _make_leaf():
    from prsm.settlement.merkle import ReceiptLeaf

    # Build a minimal ReceiptLeaf with whatever fields the dataclass needs.
    import inspect

    sig = inspect.signature(ReceiptLeaf)
    kwargs = {}
    for name, p in sig.parameters.items():
        if p.default is not inspect.Parameter.empty:
            continue
        ann = p.annotation
        if ann is int or ann == "int":
            kwargs[name] = 0
        elif ann is bytes or ann == "bytes":
            kwargs[name] = b"\x00" * 32
        elif ann is str or ann == "str":
            kwargs[name] = "0x" + "00" * 20
        else:
            kwargs[name] = b"\x00" * 32
    # Force a known executed_at for render assertions if present.
    if "executed_at_unix" in sig.parameters:
        kwargs["executed_at_unix"] = 1000
    return ReceiptLeaf(**kwargs)


def make_full_result() -> AuditRunResult:
    """An AuditRunResult carrying one finding in each of the 5 classes, plus a §7
    UNBOUND record that MUST NOT surface."""
    ds = DoubleSpendAuditFinding(
        finding=DoubleSpendFinding(
            leaf_hash=b"\x01" * 32,
            occurrences=[
                LeafOccurrence(batch_id=_BID_DS, leaf_index=3, provider_address="0xprovA"),
                LeafOccurrence(batch_id=_BID_DS2, leaf_index=7, provider_address="0xprovB"),
            ],
        ),
        dry_run_ok=True,
        dry_run_reason=None,
    )
    sig = InvalidSignatureAuditFinding(
        batch_id=_BID_SIG, receipt_index=2, dry_run_ok=False, dry_run_reason="reverted"
    )
    cm = ConsensusMismatchAuditFinding(
        finding=ConsensusMismatchFinding(
            consensus_group_id=b"\x09" * 32,
            job_id_hash=b"\x0a" * 32,
            shard_index=1,
            minority_batch_id=_BID_CM_MIN,
            minority_provider="0xmin",
            minority_receipt=None,
            minority_leaf_index=4,
            minority_output_hash=b"\x0b" * 32,
            majority_batch_id=_BID_CM_MAJ,
            majority_provider="0xmaj",
            majority_receipt=None,
            majority_leaf_index=5,
            majority_output_hash=b"\x0c" * 32,
        ),
        attempt=None,
        dry_run_ok=None,
        dry_run_reason=None,
    )
    exp = ExpiredReceiptAuditFinding(
        finding=ExpiredReceiptFinding(
            batch_id=_BID_EXP,
            target_index=6,
            leaf=_make_leaf(),
            age=5000,
            lookback=3600,
        ),
        challenge=None,
        dry_run_ok=True,
        dry_run_reason=None,
    )
    s7_bound = InferenceReceiptAuditFinding(
        batch_id=_BID_S7,
        leaf_hash=b"\x07" * 32,
        bound=True,
        proven=True,
        reason=None,
        on_chain_reason=1,
        detail="settler sig did not verify",
        dry_run_ok=True,
        dry_run_reason=None,
    )
    s7_unbound = InferenceReceiptAuditFinding(
        batch_id=_BID_S7,
        leaf_hash=b"\x08" * 32,
        bound=False,
        proven=False,
        reason=None,
        on_chain_reason=None,
    )
    return AuditRunResult(
        enumerated=10,
        selected=6,
        fetched=6,
        verified=6,
        rejected=0,
        skipped_no_cid=0,
        double_spend_findings=[ds],
        invalid_sig_findings=[sig],
        inference_receipt_findings=[s7_bound, s7_unbound],
        inference_receipt_finding_count=2,
        consensus_mismatch_findings=[cm],
        consensus_mismatch_finding_count=1,
        expired_findings=[exp],
        expired_finding_count=1,
    )


# ── (a) summarize maps all 5 classes ───────────────────────────────────


def test_summarize_maps_all_five_classes():
    summary = summarize_audit_result(make_full_result())
    # double_spend(1) + invalid_signature(1) + consensus_mismatch(1) + expired(1)
    # + inference_receipt(1 bound; unbound dropped) == 5
    assert summary.total == 5
    reasons = sorted(r.reason for r in summary.records)
    assert reasons == [
        "consensus_mismatch",
        "double_spend",
        "expired",
        "inference_receipt",
        "invalid_signature",
    ]
    assert summary.double_spend_count == 1
    assert summary.invalid_signature_count == 1
    assert summary.consensus_mismatch_count == 1
    assert summary.expired_count == 1
    assert summary.inference_receipt_count == 1

    # Each class surfaces its CANONICAL on-chain ReasonCode so an operator sees the exact
    # code to challenge with (not None read off a pure finding that lacks the field).
    # On-chain enum: DOUBLE_SPEND=0, INVALID_SIGNATURE=1, EXPIRED=3, CONSENSUS_MISMATCH=5.
    by_reason = {r.reason: r for r in summary.records}
    assert by_reason["double_spend"].on_chain_reason == 0
    assert by_reason["invalid_signature"].on_chain_reason == 1
    assert by_reason["expired"].on_chain_reason == 3
    assert by_reason["consensus_mismatch"].on_chain_reason == 5


def test_summarize_empty_result_no_records():
    summary = summarize_audit_result(AuditRunResult())
    assert summary.total == 0
    assert summary.records == []


# ── (b) §7 UNBOUND never surfaced ───────────────────────────────────────


def test_section7_unbound_never_surfaced():
    summary = summarize_audit_result(make_full_result())
    s7 = [r for r in summary.records if r.reason == "inference_receipt"]
    assert len(s7) == 1
    # The surfaced §7 record is the bound+proven one (leaf 0x07..), not unbound (0x08..)
    assert s7[0].detail.get("leaf_hash") == ("07" * 32)
    assert s7[0].detail.get("bound") is True
    assert s7[0].detail.get("proven") is True
    # The unbound leaf hash must appear NOWHERE.
    blob = summary.render()
    assert ("08" * 32) not in blob


def test_section7_only_unbound_yields_nothing():
    res = AuditRunResult(
        inference_receipt_findings=[
            InferenceReceiptAuditFinding(
                batch_id=_BID_S7, leaf_hash=b"\x08" * 32, bound=False, proven=False
            )
        ],
        inference_receipt_finding_count=1,
    )
    summary = summarize_audit_result(res)
    assert summary.total == 0
    assert summary.inference_receipt_count == 0


# ── (c) no content leak ─────────────────────────────────────────────────


def test_no_content_leak_only_hashes_ids_indices_verdicts():
    summary = summarize_audit_result(make_full_result())
    forbidden = ("prompt", "output", "plaintext", "content", "input_text", "response")
    for r in summary.records:
        d = r.to_dict()
        # every value is a hash/id/index/verdict scalar or a nested dict of those;
        # no free-text content field names.
        flat = _flatten_keys(d)
        for k in flat:
            assert k.lower() not in forbidden, f"leaked field {k!r} in {r.reason}"
        # detail values: no arbitrary long plaintext blobs (hex/int/bool/short ids only)
        for v in _flatten_values(d):
            if isinstance(v, str):
                # allowed strings are hex, addresses (0x...), short reason tags, or hex digests
                assert _looks_like_id_or_hash_or_tag(v), f"suspicious string {v!r}"


def _flatten_keys(d, acc=None):
    acc = acc if acc is not None else []
    if isinstance(d, dict):
        for k, v in d.items():
            acc.append(str(k))
            _flatten_keys(v, acc)
    elif isinstance(d, list):
        for v in d:
            _flatten_keys(v, acc)
    return acc


def _flatten_values(d, acc=None):
    acc = acc if acc is not None else []
    if isinstance(d, dict):
        for v in d.values():
            _flatten_values(v, acc)
    elif isinstance(d, list):
        for v in d:
            _flatten_values(v, acc)
    else:
        acc.append(d)
    return acc


def _looks_like_id_or_hash_or_tag(s: str) -> bool:
    # hex digest / 0x-address / known reason tags / short verdict strings
    body = s[2:] if s.startswith("0x") else s
    if body and all(c in "0123456789abcdefABCDEF" for c in body):
        return True
    if s in (
        "double_spend",
        "invalid_signature",
        "consensus_mismatch",
        "expired",
        "inference_receipt",
    ):
        return True
    # short advisory verdict reasons (e.g. "reverted") are allowed; cap length so
    # arbitrary plaintext content can't sneak through.
    return len(s) <= 64


# ── (d) render() is operator-readable ───────────────────────────────────


def test_render_includes_batch_ids_reasons_verdicts_and_submit_note():
    summary = summarize_audit_result(make_full_result())
    text = summary.render()
    # batch ids present
    assert (_BID_SIG.hex()) in text
    assert (_BID_EXP.hex()) in text
    # reasons present
    assert "double_spend" in text
    assert "invalid_signature" in text
    assert "consensus_mismatch" in text
    assert "expired" in text
    assert "inference_receipt" in text
    # dry-run verdict surfaced
    assert "dry_run" in text.lower()
    # the user-gated submit note
    low = text.lower()
    assert "user-gated" in low or "user gated" in low
    assert "submit" in low
    assert "own key" in low or "your own key" in low or "separately" in low


# ── (e) store dedups by finding_key ─────────────────────────────────────


def test_store_dedups_by_finding_key_across_ticks(tmp_path):
    store = SettlementAuditFindingsStore(tmp_path / "findings.json")
    summary = summarize_audit_result(make_full_result())
    store.record_summary(summary)
    n_first = len(store.all_findings())
    assert n_first == 5
    # Re-scanning a still-PENDING set re-detects the SAME findings -> NO duplicates.
    store.record_summary(summarize_audit_result(make_full_result()))
    assert len(store.all_findings()) == n_first
    # finding_key is the dedup tuple
    keys = {finding_key(r) for r in store.all_findings()}
    assert len(keys) == n_first


# ── (f) bounded + atomic reloadable + never-raises ──────────────────────


def test_store_bounded_evicts_oldest(tmp_path):
    store = SettlementAuditFindingsStore(tmp_path / "f.json", max_entries=2)
    for i in range(5):
        rec = AuditFindingRecord(
            reason="expired",
            batch_id_hex=(f"{i:02x}" * 32),
            target_index=i,
            on_chain_reason=3,
            dry_run_ok=None,
            dry_run_reason=None,
            detail={},
        )
        store.record_summary(
            AuditFindingsSummary(records=[rec])
        )
    assert len(store.all_findings()) == 2
    # newest two kept
    kept = {r.batch_id_hex for r in store.all_findings()}
    assert kept == {("03" * 32), ("04" * 32)}


def test_store_atomic_persist_reloadable(tmp_path):
    path = tmp_path / "f.json"
    store = SettlementAuditFindingsStore(path)
    store.record_summary(summarize_audit_result(make_full_result()))
    reloaded = SettlementAuditFindingsStore(path)
    assert len(reloaded.all_findings()) == 5
    keys_a = {finding_key(r) for r in store.all_findings()}
    keys_b = {finding_key(r) for r in reloaded.all_findings()}
    assert keys_a == keys_b


def test_store_never_raises_on_write_error(tmp_path, monkeypatch):
    store = SettlementAuditFindingsStore(tmp_path / "f.json")

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(store._store, "save", boom)
    # must not propagate
    store.record_summary(summarize_audit_result(make_full_result()))
    # in-memory still usable
    assert len(store.all_findings()) == 5


def test_store_prune(tmp_path):
    store = SettlementAuditFindingsStore(tmp_path / "f.json", now=lambda: 10_000.0)
    store.record_summary(summarize_audit_result(make_full_result()))
    assert len(store.all_findings()) == 5
    dropped = store.prune(retention_seconds=0)
    assert dropped == 5
    assert store.all_findings() == []


# ── (g) scheduler: additive, idle unchanged, retry unchanged ────────────


class _FakeLoop:
    def __init__(self, result=None, raises=False):
        self._result = result
        self._raises = raises
        self.calls = 0

    async def run_once(self, from_block, to_block, **kw):
        self.calls += 1
        if self._raises:
            raise RuntimeError("scan boom")
        return self._result


class _RecordingStore:
    def __init__(self, raises=False):
        self.calls = []
        self._raises = raises

    def record_summary(self, summary):
        self.calls.append(summary)
        if self._raises:
            raise OSError("persist boom")


def _scheduler(loop, head, *, cursor, store=None, on_findings=None):
    async def head_fn():
        return head

    return SettlementAuditScheduler(
        loop,
        head_fn,
        cursor=cursor,
        interval_s=0.0,
        findings_store=store,
        on_findings=on_findings,
    )


def test_scheduler_tick_with_findings_warns_persists_and_advances(tmp_path, caplog):
    loop = _FakeLoop(result=make_full_result())
    cursor = BlockCursor(path=":memory:")
    store = _RecordingStore()
    seen = []
    sched = _scheduler(loop, 100, cursor=cursor, store=store, on_findings=seen.append)
    with caplog.at_level(logging.WARNING):
        asyncio.run(sched.run_once_tick())
    assert loop.calls == 1
    assert cursor.load() == 100  # advanced
    assert len(store.calls) == 1  # persisted
    assert len(seen) == 1  # callback fired
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("dispute candidate" in r.getMessage().lower() for r in warnings)


def test_scheduler_tick_no_findings_no_warning_advances(tmp_path, caplog):
    loop = _FakeLoop(result=AuditRunResult(enumerated=3))
    cursor = BlockCursor(path=":memory:")
    store = _RecordingStore()
    sched = _scheduler(loop, 50, cursor=cursor, store=store)
    with caplog.at_level(logging.DEBUG):
        asyncio.run(sched.run_once_tick())
    assert cursor.load() == 50  # advanced
    assert store.calls == []  # nothing persisted for an idle tick
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("dispute candidate" in r.getMessage().lower() for r in warnings)


def test_scheduler_run_once_raises_cursor_not_advanced_store_untouched():
    loop = _FakeLoop(raises=True)
    cursor = BlockCursor(path=":memory:")
    store = _RecordingStore()
    sched = _scheduler(loop, 100, cursor=cursor, store=store)
    asyncio.run(sched.run_once_tick())
    assert cursor.load() == -1  # NOT advanced (retry next tick)
    assert store.calls == []  # not written


def test_scheduler_store_persist_failure_does_not_break_tick():
    loop = _FakeLoop(result=make_full_result())
    cursor = BlockCursor(path=":memory:")
    store = _RecordingStore(raises=True)  # record_summary raises
    sched = _scheduler(loop, 100, cursor=cursor, store=store)
    # must not raise; cursor still advances
    asyncio.run(sched.run_once_tick())
    assert cursor.load() == 100


def test_scheduler_callback_failure_does_not_break_tick():
    loop = _FakeLoop(result=make_full_result())
    cursor = BlockCursor(path=":memory:")

    def boom(_summary):
        raise RuntimeError("cb boom")

    sched = _scheduler(loop, 100, cursor=cursor, on_findings=boom)
    asyncio.run(sched.run_once_tick())
    assert cursor.load() == 100


# ── (h) never broadcast ─────────────────────────────────────────────────


def test_scheduler_never_broadcasts(tmp_path):
    """The findings surface must be READ-ONLY. Two complementary checks:

    (1) STRUCTURAL / non-vacuous: assert — at the SOURCE level — that neither the report
        module nor the scheduler's run_once_tick surface contains any broadcast/sign/
        transaction CALL site. This would FAIL the moment someone added e.g.
        ``submitter.submit(...)`` to the surface. (The words 'broadcast'/'submit'/'sign'
        appear in the operator advisory note + docstrings as PROSE — never as a ``.submit(``
        / ``.broadcast(`` call — so we match CALL-SYNTAX patterns, not the bare words.)
    (2) FUNCTIONAL: a tick with findings persists them via the read-only store (the surface
        works) and advances the cursor."""
    import inspect
    import re
    from prsm.settlement import settlement_audit_report as report_mod
    from prsm.settlement.settlement_audit_wiring import SettlementAuditScheduler

    forbidden = re.compile(
        r"\.(submit|broadcast|sign|sign_transaction|send_raw_transaction|"
        r"sendRawTransaction|build_transaction|buildTransaction|transact)\s*\(|\beth_send"
    )
    report_src = inspect.getsource(report_mod)
    tick_src = inspect.getsource(SettlementAuditScheduler.run_once_tick)
    for label, src in (("report_module", report_src), ("run_once_tick", tick_src)):
        m = forbidden.search(src)
        assert m is None, (
            f"{label} contains a forbidden broadcast/sign/transaction call site "
            f"{m.group(0)!r} — the audit findings surface must stay READ-ONLY "
            "(the challenge submit is user-gated, separate)"
        )

    # (2) functional read-only persist
    loop = _FakeLoop(result=make_full_result())
    cursor = BlockCursor(path=":memory:")
    store = SettlementAuditFindingsStore(tmp_path / "f.json")
    sched = _scheduler(loop, 100, cursor=cursor, store=store)
    asyncio.run(sched.run_once_tick())
    assert len(store.all_findings()) == 5
    assert cursor.load() == 100
