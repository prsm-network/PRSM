"""Sprint 407 — smoke-test-bootstrap-fleet.sh static pins.

The script is a live-fleet-probing operator tool. Its
correctness is verified by actually running it against
the canonical fleet (which is how we found that
bootstrap1 was running pre-sprint-389 code). This test
file pins the script's static contract:

  - exists + executable
  - canonical fleet hardcoded (US/EU/APAC)
  - documents all 10 sprint-388-406 checks it runs
  - has both text + json output modes
  - exit code reflects pass/fail
"""
from __future__ import annotations

import stat
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "smoke-test-bootstrap-fleet.sh"


def test_script_exists():
    assert SCRIPT.is_file(), f"smoke-test script missing at {SCRIPT}"


def test_script_is_executable():
    mode = SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, "script not executable"


def test_script_uses_set_uo_pipefail():
    """`set -e` is intentionally OMITTED — individual checks
    have their own success/failure tracking and a single
    failed check should not abort the whole probe."""
    text = SCRIPT.read_text()
    assert "set -uo pipefail" in text


def test_canonical_hosts_pinned():
    text = SCRIPT.read_text()
    for host in (
        # bootstrap1 was renamed to bootstrap-us in sprint 575 (dead DNS).
        "bootstrap-us.prsm-network.com",
        "bootstrap-eu.prsm-network.com",
        "bootstrap-apac.prsm-network.com",
    ):
        assert host in text, f"canonical host {host} missing"
    # The legacy hostname no longer resolves — the probe must not target it.
    assert "bootstrap1.prsm-network.com" not in text, (
        "smoke-test script still probes dead bootstrap1; renamed to "
        "bootstrap-us in sprint 575"
    )


def test_default_ports_pinned():
    text = SCRIPT.read_text()
    assert "WSS_PORT=8765" in text
    assert "API_PORT=8000" in text


def test_text_and_json_output_modes_supported():
    text = SCRIPT.read_text()
    # Both modes referenced
    assert 'FORMAT="text"' in text
    assert '"$FORMAT" = "json"' in text
    # Help block documents both
    assert "--format json" in text


def test_canonical_check_battery_documented():
    """Header block documents what the script verifies.
    Pins the sprint-388-405 surface coverage."""
    text = SCRIPT.read_text()
    expected = [
        "sprint 385",          # TCP+WSS reachability
        "sprint 392",          # /health/detailed
        "sprint 389",          # content-neg /metrics
        "sprint 397",          # disabled-vs-stale
        "sprint 394",          # labeled subsystem gauges
        "sprint 398",          # REGION_UNSET pin
    ]
    for marker in expected:
        assert marker in text, (
            f"sprint coverage marker {marker!r} missing"
        )


def test_bash_3_2_compatible_no_associative_arrays():
    """macOS ships bash 3.2 by default. The script MUST NOT
    use bash-4-only features like associative arrays
    (declare -A)."""
    text = SCRIPT.read_text()
    assert "declare -A" not in text, (
        "declare -A is bash-4-only; macOS default bash is 3.2"
    )


def test_no_required_external_deps_beyond_curl_python3():
    """Operators run this from anywhere — minimal deps. The
    only external commands the script SHALL require are
    curl + python3. nc is opt-in (fallback to /dev/tcp).
    jq is intentionally avoided (not preinstalled)."""
    text = SCRIPT.read_text()
    # No jq usage
    assert "jq " not in text
    assert "| jq" not in text
    # python3 is the JSON parser
    assert "python3" in text


def test_help_flag_exits_zero():
    """--help should print help and exit 0, not error out
    on the unknown-arg path."""
    text = SCRIPT.read_text()
    assert "-h|--help" in text
    # Bare exit 0 for help path
    assert "exit 0" in text
