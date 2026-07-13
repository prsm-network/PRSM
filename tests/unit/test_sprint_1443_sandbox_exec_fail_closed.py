"""sp1443 (compute-execution / sandbox audit) — the subprocess "sandbox" RCE.

An adversarial audit of the compute-execution layer found the WASM executor's isolation SOUND
(consume_fuel + epoch wall-clock deadline + memory set_limits; the guest gets no host access), but
the SEPARATE subprocess backend — SandboxManager.execute_safely — ran untrusted `.py` as a plain
child of the daemon user with NO OS isolation: no netns, no seccomp, no chroot, full host PATH. Its
block_network / allowed_domains flags are advisory JSON no execution code reads. Reachable (by
design) from the authenticated /integrations/import + /integrations/security/scan flows with
enable_sandbox defaulting True and _should_run_sandbox auto-running any benign-looking payload — so a
file-read + urllib-exfil `.py` that evades the regex vuln scan = host-secret exfiltration / SSRF /
RCE. (The node API doesn't currently mount the integrations router, so it's a critical LATENT
landmine; the fix makes it fail-safe for any current or future caller.)

  #1 (CRITICAL) — fixed FAIL-CLOSED: execute_safely refuses to run untrusted content unless the
     operator explicitly opts in (PRSM_SANDBOX_ALLOW_UNISOLATED_EXEC, only behind real external
     isolation). Default = static-scan only, no execution.
  #2 (HIGH) — when opted in, the run is in a NEW SESSION with rlimits so a forked grandchild can't
     survive the timeout as a runaway (killpg reaps the group) and a fork/alloc bomb is bounded.

Money/security assertions (CLAUDE.md: never weaken to pass).
"""
from __future__ import annotations

import subprocess as _subprocess

import pytest

from prsm.core.integrations.security.sandbox_manager import SandboxManager


# ── Finding #1 — fail-closed: untrusted code is NOT executed by default ───────


@pytest.mark.requires_halmos  # bypass the session-wide subprocess mock — need REAL exec here
async def test_execute_safely_does_not_run_untrusted_code_by_default(tmp_path, monkeypatch):
    """The CRITICAL invariant: with no explicit opt-in, submitting a .py must NOT execute it — the
    old backend ran it as the daemon user (host-secret exfil / SSRF / RCE). Uses REAL subprocess so
    a broken gate would ACTUALLY run the payload and create the sentinel (a genuine RCE probe)."""
    monkeypatch.delenv("PRSM_SANDBOX_ALLOW_UNISOLATED_EXEC", raising=False)
    sentinel = tmp_path / "it_ran.txt"
    script = tmp_path / "evil.py"
    script.write_text(f"open({str(sentinel)!r}, 'w').write('pwned')\n")

    mgr = SandboxManager()
    result = await mgr.execute_safely(str(script), {})

    assert result.status == "skipped_no_isolation"
    assert not sentinel.exists(), "untrusted code EXECUTED with no isolation — RCE still open"


@pytest.mark.requires_halmos  # need REAL exec
async def test_execute_safely_runs_only_when_operator_opts_in(tmp_path, monkeypatch):
    monkeypatch.setenv("PRSM_SANDBOX_ALLOW_UNISOLATED_EXEC", "1")
    sentinel = tmp_path / "it_ran.txt"
    script = tmp_path / "job.py"
    script.write_text(f"open({str(sentinel)!r}, 'w').write('ok')\n")

    mgr = SandboxManager()
    result = await mgr.execute_safely(str(script), {})

    assert result.status == "completed"
    assert sentinel.exists(), "opt-in execution should actually run the content"


async def test_non_python_content_is_only_validated_never_executed(tmp_path, monkeypatch):
    monkeypatch.delenv("PRSM_SANDBOX_ALLOW_UNISOLATED_EXEC", raising=False)
    blob = tmp_path / "data.bin"
    blob.write_text("not code")
    mgr = SandboxManager()
    result = await mgr.execute_safely(str(blob), {})
    assert result.success and "non-executable" in result.output


# ── Finding #2 — opt-in run is in a new session with rlimits ──────────────────


async def test_opt_in_exec_wires_new_session_rlimits_and_drops_host_path(tmp_path, monkeypatch):
    """The opt-in path must pass a preexec_fn (new session + rlimits so killpg can reap a runaway
    grandchild and a fork/alloc bomb is bounded) and NOT inherit the host PATH."""
    monkeypatch.setenv("PRSM_SANDBOX_ALLOW_UNISOLATED_EXEC", "1")
    captured = {}

    class _FakeProc:
        returncode = 0
        pid = 4242

        def communicate(self, timeout=None):
            return ("out", "")

    def _fake_popen(cmd, **kwargs):
        captured.update(kwargs)
        return _FakeProc()

    monkeypatch.setattr(_subprocess, "Popen", _fake_popen)
    script = tmp_path / "s.py"
    script.write_text("print(1)\n")

    mgr = SandboxManager()
    result = await mgr.execute_safely(str(script), {})

    assert result.status == "completed"
    assert callable(captured.get("preexec_fn")), (
        "no preexec_fn → no new-session/rlimits, so a runaway grandchild survives (Finding 2)")
    assert captured.get("env", {}).get("PATH") == "", "host PATH must not be inherited"


@pytest.mark.requires_halmos  # need REAL exec
async def test_opt_in_infinite_loop_times_out_and_is_killed(tmp_path, monkeypatch):
    """End-to-end exercise of the timeout→killpg branch with a real subprocess."""
    monkeypatch.setenv("PRSM_SANDBOX_ALLOW_UNISOLATED_EXEC", "1")
    script = tmp_path / "loop.py"
    script.write_text("while True:\n    pass\n")
    mgr = SandboxManager()
    result = await mgr.execute_safely(str(script), {"timeout": 1})
    assert result.status == "timeout"
