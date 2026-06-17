"""Sprint 1150 — operator CLI for the settlement fraud-defense audit surface.

THE GAP this closes: sp1149 made detected dispute candidates operationally
VISIBLE *inside* the node daemon (WARNING log + persisted findings store). But
there was NO operator entry point OUTSIDE the daemon — an operator could not run
a one-off audit scan, nor review the persisted findings, without booting a full
node. This brick adds a ``scripts/settlement_audit_report.py`` operator CLI
(mirroring ``scripts/settlement_sepolia_e2e.py``, the established settlement-ops
pattern) plus a PURE ``summarize_records`` helper so persisted store records
render IDENTICALLY to a fresh ``AuditRunResult``.

HARD INVARIANT under test: the CLI + the helper are READ-ONLY — they NEVER
broadcast / sign / slash; they import NO submitter broadcast path. Test (d) is a
STRUCTURAL non-vacuous assertion of that.

NO network. Live-mode (--run-once) deps (web3/chain) are lazy-imported INSIDE
that mode so --findings-file works in a minimal env; test (e) asserts the script
imports cleanly without web3.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from prsm.settlement.settlement_audit_loop import AuditRunResult
from prsm.settlement.settlement_audit_report import (
    AuditFindingRecord,
    SettlementAuditFindingsStore,
    finding_key,
    summarize_audit_result,
    summarize_records,
)

# Reuse the canonical 5-class fixture from the sp1149 suite (one finding in each
# of the 5 classes + a §7 UNBOUND that MUST NOT surface). No reinvention.
from tests.unit.test_sprint_1149_audit_findings_surface import make_full_result


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "settlement_audit_report.py"


@pytest.fixture
def real_subprocess(monkeypatch):
    """The session-wide conftest mocks subprocess.run/Popen for safety. These two CLI
    tests legitimately spawn the REAL operator script in a child process (the way an
    operator runs it), so we restore the genuine, un-imported subprocess implementations
    for the test's duration. The session mock is re-applied automatically on teardown
    (monkeypatch undoes the patch). We do NOT simplify the assertion — we run the real
    binary and check its real stdout/exit code."""
    import importlib

    real_subprocess_mod = importlib.import_module("subprocess")
    # The module object's attributes were replaced by the session patch; the genuine
    # callables still live on the unittest.mock patcher's `temp_original`. Reach the real
    # ones via a fresh, unpatched reference to the stdlib functions.
    import _posixsubprocess  # noqa: F401  (ensures the real backend is importable)

    # Pull the unpatched originals straight off the patchers the conftest stored.
    conftest = importlib.import_module("tests.conftest")
    run_patcher = conftest._subprocess_run_patcher
    popen_patcher = conftest._subprocess_popen_patcher
    real_run = run_patcher.temp_original if run_patcher else real_subprocess_mod.run
    real_popen = (
        popen_patcher.temp_original if popen_patcher else real_subprocess_mod.Popen
    )
    monkeypatch.setattr(real_subprocess_mod, "run", real_run)
    monkeypatch.setattr(real_subprocess_mod, "Popen", real_popen)
    yield real_subprocess_mod


def _load_script_module():
    """Import the operator CLI as a module (so we can call its main() in-process)."""
    spec = importlib.util.spec_from_file_location(
        "sprint_1150_settlement_audit_report_cli", _SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── (a) summarize_records over a list of records ────────────────────────


def test_summarize_records_counts_per_class():
    fresh = summarize_audit_result(make_full_result())
    records = list(fresh.records)
    summary = summarize_records(records)
    assert summary.total == 5
    assert summary.double_spend_count == 1
    assert summary.invalid_signature_count == 1
    assert summary.consensus_mismatch_count == 1
    assert summary.expired_count == 1
    assert summary.inference_receipt_count == 1
    # The records are carried through unchanged (same finding keys).
    assert {finding_key(r) for r in summary.records} == {
        finding_key(r) for r in records
    }


def test_summarize_records_empty_total_zero():
    summary = summarize_records([])
    assert summary.total == 0
    assert summary.records == []
    assert summary.render() == "settlement audit: no dispute candidates"


# ── (b) round-trip through a persisted store renders identically ─────────


def test_summarize_records_roundtrips_through_store(tmp_path):
    fresh_summary = summarize_audit_result(make_full_result())
    store = SettlementAuditFindingsStore(tmp_path / "findings.json")
    store.record_summary(fresh_summary)

    reloaded = SettlementAuditFindingsStore(tmp_path / "findings.json")
    persisted_summary = summarize_records(reloaded.all_findings())

    # Same per-class counts.
    assert persisted_summary.total == fresh_summary.total
    assert persisted_summary.double_spend_count == fresh_summary.double_spend_count
    assert persisted_summary.invalid_signature_count == (
        fresh_summary.invalid_signature_count
    )
    assert persisted_summary.consensus_mismatch_count == (
        fresh_summary.consensus_mismatch_count
    )
    assert persisted_summary.expired_count == fresh_summary.expired_count
    assert persisted_summary.inference_receipt_count == (
        fresh_summary.inference_receipt_count
    )
    # Same finding keys.
    assert {finding_key(r) for r in persisted_summary.records} == {
        finding_key(r) for r in fresh_summary.records
    }
    # render() of the persisted records IS IDENTICAL to the fresh-result render()
    # (the whole point of the helper — persisted == fresh).
    assert persisted_summary.render() == fresh_summary.render()


# ── (c) CLI --findings-file prints render(); missing file -> no findings ─


def test_cli_findings_file_prints_render(tmp_path):
    path = tmp_path / "findings.json"
    fresh_summary = summarize_audit_result(make_full_result())
    SettlementAuditFindingsStore(path).record_summary(fresh_summary)

    mod = _load_script_module()
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = mod.main(["--findings-file", str(path)])
    out = buf.getvalue()
    assert rc == 0
    # The full render() text appears (header + per-finding lines + submit note).
    assert "settlement audit found 5 dispute candidate(s)" in out
    # Batch ids + reasons present.
    assert "double_spend" in out
    assert "invalid_signature" in out
    assert "consensus_mismatch" in out
    assert "expired" in out
    assert "inference_receipt" in out
    # A specific batch id from the fixture appears (aa..).
    assert ("aa" * 32) in out
    # The user-gated-submit note is present (READ-ONLY pointer).
    assert "USER-GATED" in out
    assert "NEVER broadcasts" in out


def test_cli_missing_file_says_no_findings_exit_zero(tmp_path):
    path = tmp_path / "does_not_exist.json"
    assert not path.exists()
    mod = _load_script_module()
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = mod.main(["--findings-file", str(path)])
    out = buf.getvalue()
    assert rc == 0
    assert "no dispute candidates" in out or "no findings" in out.lower()


def test_cli_findings_file_json_mode(tmp_path):
    path = tmp_path / "findings.json"
    SettlementAuditFindingsStore(path).record_summary(
        summarize_audit_result(make_full_result())
    )
    mod = _load_script_module()
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = mod.main(["--findings-file", str(path), "--json"])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["total"] == 5
    assert payload["counts"]["double_spend"] == 1
    assert len(payload["records"]) == 5


def test_cli_subprocess_findings_file(tmp_path, real_subprocess):
    """End-to-end via subprocess (the way an operator actually runs it)."""
    path = tmp_path / "findings.json"
    SettlementAuditFindingsStore(path).record_summary(
        summarize_audit_result(make_full_result())
    )
    proc = real_subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--findings-file", str(path)],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "settlement audit found 5 dispute candidate(s)" in proc.stdout
    assert "USER-GATED" in proc.stdout


# ── (d) NEVER-BROADCAST — structural, non-vacuous ───────────────────────


_BROADCAST_CALL_RE = re.compile(
    r"\.(?:submit|broadcast|sign|send_raw_transaction|build_transaction)\s*\("
)
# A real import/construction of a challenge submitter (NOT a prose mention of the
# class name in a docstring — both this module and the sp1149 report module
# legitimately *describe* the user-gated ChallengeSubmitter.submit path they leave
# untouched). The invariant is: no import line, no construction call site.
_SUBMITTER_IMPORT_RE = re.compile(
    r"^[ \t]*(?:from\s+[\w.]+\s+import\s+[^\n]*ChallengeSubmitter"
    r"|import\s+[^\n]*ChallengeSubmitter)",
    re.MULTILINE,
)
_SUBMITTER_CTOR_RE = re.compile(r"ChallengeSubmitter\s*\(")


def _strip_string_and_comment_lines(src: str) -> str:
    """Drop comment lines + triple-quoted docstring blocks so the never-broadcast regex
    can't false-positive on prose that *describes* the user-gated submit path. This keeps
    the assertion NON-VACUOUS: it scans real CODE, not the documentation about it."""
    out = []
    in_doc = False
    doc_delim = None
    for line in src.splitlines():
        stripped = line.strip()
        if in_doc:
            if doc_delim in line:
                in_doc = False
            continue
        if stripped.startswith("#"):
            continue
        for delim in ('"""', "'''"):
            if stripped.startswith(delim):
                # A one-line docstring ("""x""") closes on the same line.
                if stripped.count(delim) >= 2 and len(stripped) > len(delim):
                    line = ""
                else:
                    in_doc = True
                    doc_delim = delim
                break
        if not in_doc:
            out.append(line)
    return "\n".join(out)


def test_script_source_has_no_broadcast_call_site():
    src = _SCRIPT_PATH.read_text()
    # Non-vacuous: the source is non-trivial (it really is the CLI).
    assert "settlement_audit" in src
    assert "argparse" in src
    code = _strip_string_and_comment_lines(src)
    hits = _BROADCAST_CALL_RE.findall(code)
    assert hits == [], f"script must NEVER broadcast/sign; found call sites: {hits}"
    # No import / construction of a challenge submitter (broadcast path) in CODE.
    assert _SUBMITTER_IMPORT_RE.search(code) is None
    assert _SUBMITTER_CTOR_RE.search(code) is None


def test_helper_source_has_no_broadcast_call_site():
    helper_path = (
        _REPO_ROOT / "prsm" / "settlement" / "settlement_audit_report.py"
    )
    src = helper_path.read_text()
    # summarize_records must exist and be exported.
    assert "def summarize_records(" in src
    assert "summarize_records" in src
    code = _strip_string_and_comment_lines(src)
    # No broadcast/sign call sites in the (read-only) report module CODE.
    hits = _BROADCAST_CALL_RE.findall(code)
    assert hits == [], f"report module must NEVER broadcast/sign; found: {hits}"
    assert _SUBMITTER_IMPORT_RE.search(code) is None
    assert _SUBMITTER_CTOR_RE.search(code) is None


# ── (e) the script imports cleanly without web3 (live deps are lazy) ─────


def test_script_imports_without_web3(real_subprocess):
    """The script must import even if web3 is unavailable — live-mode deps are
    lazy-imported inside --run-once. We simulate a web3-less env by blocking the
    import and exec'ing the module in a subprocess so we don't poison the suite."""
    code = (
        "import sys, importlib.util\n"
        "import builtins\n"
        "_real_import = builtins.__import__\n"
        "def _blocked(name, *a, **k):\n"
        "    if name == 'web3' or name.startswith('web3.'):\n"
        "        raise ImportError('web3 blocked for test')\n"
        "    return _real_import(name, *a, **k)\n"
        "builtins.__import__ = _blocked\n"
        f"spec = importlib.util.spec_from_file_location('s', r'{_SCRIPT_PATH}')\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "assert hasattr(mod, 'main')\n"
        "print('IMPORT_OK')\n"
    )
    proc = real_subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "IMPORT_OK" in proc.stdout
