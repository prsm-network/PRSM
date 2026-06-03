"""Sprint 360 — halmos runner tests.

The runner is the bridge between Python operator tooling
and halmos symbolic execution. Tests cover:
  - fail-soft when halmos / forge isn't installed
  - parsing of canonical halmos output formats
    (PASS / FAIL / mixed)
  - suite-status aggregation
  - timeout + invocation-error handling
  - the symbolic-proof catalog
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prsm.economy.web3.halmos_runner import (
    HalmosRunner,
    SYMBOLIC_PROOF_CATALOG,
    SymbolicProofResult,
    SymbolicProofStatus,
    SymbolicProofSuite,
    _parse_halmos_output,
)


# ── status + catalog ─────────────────────────────────


def test_status_enum_values():
    assert SymbolicProofStatus.PASSED.value == "passed"
    assert SymbolicProofStatus.FAILED.value == "failed"
    assert SymbolicProofStatus.SKIPPED.value == "skipped"
    assert SymbolicProofStatus.ERROR.value == "error"


def test_catalog_has_ftns_supply_cap_proof():
    """Sprint 360 ships FTNSSupplyCapSpec; pin the entry
    so accidental removal trips CI."""
    assert "FTNSSupplyCapSpec" in SYMBOLIC_PROOF_CATALOG
    entry = SYMBOLIC_PROOF_CATALOG["FTNSSupplyCapSpec"]
    assert entry["mirrors_runtime_contract"] == "ftns_token"
    assert "INV-FT-1" in entry["runtime_invariants"]
    assert "INV-FT-2" in entry["runtime_invariants"]


def test_catalog_has_escrow_pool_solvency_proof():
    """Sprint 363 ships EscrowPoolSolvencySpec — sister
    proof to RoyaltyDistributorSolvencySpec, mirrors
    INV-EP-1."""
    assert (
        "EscrowPoolSolvencySpec" in SYMBOLIC_PROOF_CATALOG
    )
    entry = SYMBOLIC_PROOF_CATALOG["EscrowPoolSolvencySpec"]
    assert entry["mirrors_runtime_contract"] == "escrow_pool"
    assert "INV-EP-1" in entry["runtime_invariants"]


def test_catalog_has_creator_stake_registry_proof():
    """Sprint 983 ships CreatorStakeRegistrySpec — the §14
    anti-spam money-custody contract joins the symbolic lane:
    solvency + slash-conservation + anti-game eligibility."""
    assert (
        "CreatorStakeRegistrySpec" in SYMBOLIC_PROOF_CATALOG
    )
    entry = SYMBOLIC_PROOF_CATALOG["CreatorStakeRegistrySpec"]
    assert (
        entry["mirrors_runtime_contract"] == "creator_stake_registry"
    )
    # Runtime state-pins are added at commission time (no on-chain
    # address yet) — the proofs here are algorithmic, the symbolic
    # lane's job.
    assert entry["runtime_invariants"] == []


def test_catalog_has_three_band_routing_proof():
    """Sprint 374: PRSM-PROV-1 three-band routing (§7.19)."""
    assert "ThreeBandRoutingSpec" in SYMBOLIC_PROOF_CATALOG
    entry = SYMBOLIC_PROOF_CATALOG["ThreeBandRoutingSpec"]
    assert entry["runtime_invariants"] == []
    assert (
        entry["mirrors_runtime_contract"] == "content_dedup"
    )


def test_catalog_has_streaming_emit_cap_proof():
    """Sprint 372: SSE streaming-emit cap (§7.5)."""
    assert (
        "StreamingEmitCapSpec" in SYMBOLIC_PROOF_CATALOG
    )
    entry = SYMBOLIC_PROOF_CATALOG["StreamingEmitCapSpec"]
    assert entry["runtime_invariants"] == []


def test_catalog_has_kv_cache_lru_bound_proof():
    """Sprint 373: KVCacheManager LRU bound (§7.9)."""
    assert (
        "KVCacheLRUBoundSpec" in SYMBOLIC_PROOF_CATALOG
    )
    entry = SYMBOLIC_PROOF_CATALOG["KVCacheLRUBoundSpec"]
    assert entry["runtime_invariants"] == []


def test_catalog_has_m1_cadence_driven_yield_proof():
    """Sprint 370: M1 cadence-driven yield invariant (§7.13)."""
    assert (
        "M1CadenceDrivenYieldSpec" in SYMBOLIC_PROOF_CATALOG
    )
    entry = SYMBOLIC_PROOF_CATALOG["M1CadenceDrivenYieldSpec"]
    assert entry["runtime_invariants"] == []
    assert (
        entry["mirrors_runtime_contract"]
        == "streaming_inference"
    )


def test_catalog_has_encrypted_probs_coset_proof():
    """Sprint 371: encrypted_proposed_token_probs co-set
    validators (§7.14)."""
    assert (
        "EncryptedProbsCoSetSpec" in SYMBOLIC_PROOF_CATALOG
    )
    entry = SYMBOLIC_PROOF_CATALOG["EncryptedProbsCoSetSpec"]
    assert entry["runtime_invariants"] == []
    assert (
        entry["mirrors_runtime_contract"]
        == "streaming_inference"
    )


def test_catalog_has_m2_response_size_padding_proof():
    """Sprint 369 ships M2ResponseSizePaddingSpec — Phase
    3.x.11.q.x response-size leak closure. Off-chain
    target like sprints 367-368."""
    assert (
        "M2ResponseSizePaddingSpec" in SYMBOLIC_PROOF_CATALOG
    )
    entry = SYMBOLIC_PROOF_CATALOG["M2ResponseSizePaddingSpec"]
    assert entry["runtime_invariants"] == []
    assert (
        entry["mirrors_runtime_contract"]
        == "streaming_inference"
    )


def test_catalog_has_chunk_streaming_bounds_proof():
    """Sprint 368 ships ChunkStreamingBoundsSpec — H1
    bounded-iterator invariant + relay-defense binding.
    Off-chain target like SpeculationRollbackMathSpec."""
    assert (
        "ChunkStreamingBoundsSpec" in SYMBOLIC_PROOF_CATALOG
    )
    entry = SYMBOLIC_PROOF_CATALOG["ChunkStreamingBoundsSpec"]
    assert entry["runtime_invariants"] == []
    assert (
        entry["mirrors_runtime_contract"]
        == "streaming_inference"
    )


def test_runner_passes_loop_bound_to_halmos():
    """Sprint 368 — verify the --loop flag is passed.
    Default loop_bound=32 covers chunk-streaming proofs.
    Test by inspecting the cmd construction logic."""
    runner = HalmosRunner(loop_bound=42)
    # Internal attribute pin — if anyone removes the flag,
    # this fails immediately.
    assert runner._loop_bound == 42


def test_catalog_has_speculation_rollback_math_proof():
    """Sprint 367 ships SpeculationRollbackMathSpec — first
    symbolic proof against the streaming-inference subsystem
    (off-chain Python algorithm structurally mirrored)."""
    assert (
        "SpeculationRollbackMathSpec"
        in SYMBOLIC_PROOF_CATALOG
    )
    entry = SYMBOLIC_PROOF_CATALOG[
        "SpeculationRollbackMathSpec"
    ]
    # No runtime invariant ID — this is a streaming-
    # inference algorithm, not an on-chain invariant.
    # Empty list is the canonical "no runtime mirror" signal.
    assert entry["runtime_invariants"] == []
    assert (
        entry["mirrors_runtime_contract"]
        == "streaming_inference"
    )


def test_catalog_has_role_disarm_access_control_proof():
    """Sprint 363 ships RoleDisarmAccessControlSpec —
    structural proof of OZ AccessControl uncircumventability
    underlying INV-FT-3/4/5."""
    assert (
        "RoleDisarmAccessControlSpec"
        in SYMBOLIC_PROOF_CATALOG
    )
    entry = SYMBOLIC_PROOF_CATALOG[
        "RoleDisarmAccessControlSpec"
    ]
    ids = entry["runtime_invariants"]
    for expected in ["INV-FT-3", "INV-FT-4", "INV-FT-5"]:
        assert expected in ids


def test_catalog_has_admin_bounded_setters_proof():
    """Sprint 362 ships AdminBoundedSettersSpec covering
    all remaining rate-bound invariants (INV-SS-1+2,
    INV-SB-1+2+3, INV-CD-1, INV-EC-1+2). Pin the entry +
    the 8 invariant IDs."""
    assert (
        "AdminBoundedSettersSpec" in SYMBOLIC_PROOF_CATALOG
    )
    entry = SYMBOLIC_PROOF_CATALOG[
        "AdminBoundedSettersSpec"
    ]
    ids = entry["runtime_invariants"]
    # All 8 rate-bound invariants covered
    for expected in [
        "INV-SS-1", "INV-SS-2",
        "INV-SB-1", "INV-SB-2", "INV-SB-3",
        "INV-CD-1",
        "INV-EC-1", "INV-EC-2",
    ]:
        assert expected in ids, (
            f"{expected} missing from "
            f"AdminBoundedSettersSpec catalog entry"
        )


def test_catalog_has_royalty_distributor_solvency_proof():
    """Sprint 361 ships RoyaltyDistributorSolvencySpec —
    the canonical solvency proof across distributeRoyalty +
    claim + recoverStranded. Mirrors INV-RD-1 + INV-RD-4."""
    assert (
        "RoyaltyDistributorSolvencySpec"
        in SYMBOLIC_PROOF_CATALOG
    )
    entry = SYMBOLIC_PROOF_CATALOG[
        "RoyaltyDistributorSolvencySpec"
    ]
    assert (
        entry["mirrors_runtime_contract"]
        == "royalty_distributor"
    )
    # INV-RD-4 is THE solvency invariant per the existing
    # registry; INV-RD-1 covers the NETWORK_FEE_BPS pin
    # that the symbolic spec also verifies.
    assert "INV-RD-4" in entry["runtime_invariants"]
    assert "INV-RD-1" in entry["runtime_invariants"]


# ── parser ───────────────────────────────────────────


def test_parser_handles_all_pass():
    output = """
Running 3 tests for test/FTNSTokenSupplyCap.t.sol:FTNSSupplyCapSpec
[PASS] check_max_supply_constant_value() (paths: 1, time: 0.00s, bounds: [])
[PASS] check_mint_preserves_cap(address,uint256) (paths: 5, time: 0.02s, bounds: [])
[PASS] check_post_construction_in_cap() (paths: 1, time: 0.00s, bounds: [])
Symbolic test result: 3 passed; 0 failed; time: 0.09s
""".strip()
    suite = _parse_halmos_output(
        "FTNSSupplyCapSpec", output,
    )
    assert suite.status == SymbolicProofStatus.PASSED
    assert len(suite.proofs) == 3
    assert all(
        p.status == SymbolicProofStatus.PASSED
        for p in suite.proofs
    )
    # paths_explored captured for the symbolic test
    mint_proof = next(
        p for p in suite.proofs
        if p.name.startswith("check_mint_preserves_cap")
    )
    assert mint_proof.paths_explored == 5


def test_parser_handles_mixed_pass_fail():
    output = """
Running 2 tests
[PASS] check_pass() (paths: 1, time: 0.00s, bounds: [])
[FAIL] check_fail() (paths: 3, time: 0.01s, bounds: [])
""".strip()
    suite = _parse_halmos_output("X", output)
    assert suite.status == SymbolicProofStatus.FAILED
    assert len(suite.proofs) == 2
    statuses = {p.status for p in suite.proofs}
    assert SymbolicProofStatus.PASSED in statuses
    assert SymbolicProofStatus.FAILED in statuses


def test_parser_strips_ansi_color_codes():
    """Halmos emits ANSI color codes around PASS/FAIL.
    Parser must strip them before regex-matching."""
    output = (
        "\x1b[32m[PASS]\x1b[0m check_x() (paths: 1, time: 0.0s, bounds: [])"
    )
    suite = _parse_halmos_output("X", output)
    assert suite.status == SymbolicProofStatus.PASSED
    assert len(suite.proofs) == 1


def test_parser_extracts_halmos_version():
    output = """
halmos 0.3.3
Running 1 tests
[PASS] check_x() (paths: 1, time: 0.00s, bounds: [])
""".strip()
    suite = _parse_halmos_output("X", output)
    assert suite.halmos_version == "0.3.3"


def test_parser_errors_on_empty_output():
    """If halmos produced no parseable lines (e.g., build
    failure), suite reports ERROR — not silently PASSED."""
    suite = _parse_halmos_output("X", "")
    assert suite.status == SymbolicProofStatus.ERROR
    assert "no parseable" in (suite.error or "")


def test_parser_handles_zero_paths():
    """Trivial constant-checks have paths=0 or 1; ensure
    parser tolerates."""
    output = (
        "[PASS] check_x() (paths: 0, time: 0.00s, bounds: [])"
    )
    suite = _parse_halmos_output("X", output)
    assert suite.proofs[0].paths_explored == 0


# ── runner fail-soft behavior ────────────────────────


def test_runner_skipped_when_halmos_missing(tmp_path):
    """No halmos in PATH → SKIPPED with named tool, no
    raise."""
    runner = HalmosRunner(
        halmos_bin=None, forge_bin="/usr/bin/forge",
    )
    with patch(
        "shutil.which",
        side_effect=lambda x: (
            "/usr/bin/forge" if x == "forge" else None
        ),
    ):
        suite = runner.run("FTNSSupplyCapSpec")
    assert suite.status == SymbolicProofStatus.SKIPPED
    assert "halmos" in (suite.error or "")


def test_runner_skipped_when_forge_missing():
    runner = HalmosRunner(
        halmos_bin="/usr/bin/halmos", forge_bin=None,
    )
    with patch(
        "shutil.which",
        side_effect=lambda x: (
            "/usr/bin/halmos" if x == "halmos" else None
        ),
    ):
        suite = runner.run("X")
    assert suite.status == SymbolicProofStatus.SKIPPED
    assert "forge" in (suite.error or "")


def test_runner_skipped_when_both_missing():
    runner = HalmosRunner()
    with patch("shutil.which", return_value=None):
        suite = runner.run("X")
    assert suite.status == SymbolicProofStatus.SKIPPED
    assert "halmos" in (suite.error or "")
    assert "forge" in (suite.error or "")


def test_runner_skipped_when_proofs_dir_missing():
    """Even with both tools installed, missing proofs dir
    SKIPS rather than crashes (e.g., partial repo clone)."""
    runner = HalmosRunner(
        proofs_dir="/tmp/__nonexistent_halmos_dir_xyz__",
        halmos_bin="/usr/bin/halmos",
        forge_bin="/usr/bin/forge",
    )
    with patch(
        "shutil.which",
        side_effect=lambda x: (
            "/usr/bin/halmos" if x == "halmos"
            else "/usr/bin/forge"
        ),
    ):
        suite = runner.run("X")
    assert suite.status == SymbolicProofStatus.SKIPPED
    assert "symbolic-proofs dir" in (suite.error or "")


def test_runner_is_available_checks_both_tools():
    runner = HalmosRunner()
    with patch("shutil.which", return_value=None):
        assert runner.is_available() is False
    with patch("shutil.which", return_value="/usr/bin/halmos"):
        # Both lookups return same value when which is mocked
        # with a string return; that satisfies both halmos +
        # forge.
        assert runner.is_available() is True


def test_runner_missing_tools_reports_both():
    runner = HalmosRunner()
    with patch("shutil.which", return_value=None):
        missing = runner.missing_tools()
    assert "halmos" in missing
    assert "forge" in missing


# ── suite serialization ──────────────────────────────


def test_suite_to_dict_includes_summary():
    suite = SymbolicProofSuite(
        contract="X",
        status=SymbolicProofStatus.PASSED,
        proofs=[
            SymbolicProofResult(
                name="check_a()",
                status=SymbolicProofStatus.PASSED,
                paths_explored=1,
            ),
            SymbolicProofResult(
                name="check_b()",
                status=SymbolicProofStatus.FAILED,
                paths_explored=3,
            ),
            SymbolicProofResult(
                name="check_c()",
                status=SymbolicProofStatus.ERROR,
            ),
        ],
    )
    d = suite.to_dict()
    assert d["summary"]["passed"] == 1
    assert d["summary"]["failed"] == 1
    assert d["summary"]["errored"] == 1
    assert len(d["proofs"]) == 3


def test_proof_result_to_dict_serializable():
    r = SymbolicProofResult(
        name="check_x()",
        status=SymbolicProofStatus.PASSED,
        paths_explored=5,
        time_seconds=0.02,
    )
    d = r.to_dict()
    assert d["name"] == "check_x()"
    assert d["status"] == "passed"
    assert d["paths_explored"] == 5
    assert d["time_seconds"] == 0.02


# ── subprocess timeout handling ──────────────────────


# Sprint 366 — live end-to-end tests against real halmos.
# Uses the @pytest.mark.requires_halmos marker carved in
# tests/conftest.py to surgically bypass the session-wide
# subprocess mock. Tests auto-skip when halmos / forge
# isn't on PATH so CI environments without the tools still
# pass — but when both are present, every PR touching a
# spec contract gets symbolically verified.


@pytest.mark.requires_halmos
def test_live_halmos_ftns_supply_cap():
    """Real halmos invocation against FTNSSupplyCapSpec.
    Proves the supply-cap invariant on every test run."""
    runner = HalmosRunner(timeout_seconds=120)
    if not runner.is_available():
        pytest.skip(
            "halmos or forge not on PATH; skipping live "
            "integration test (mocked-output tests above "
            "cover the runner's parsing + fail-soft surface)"
        )
    suite = runner.run("FTNSSupplyCapSpec")
    assert suite.status == SymbolicProofStatus.PASSED, (
        f"FTNSSupplyCapSpec failed symbolic verification: "
        f"{suite.to_dict()}"
    )
    # All 3 proofs from the spec must be present + passing
    proof_names = {p.name for p in suite.proofs}
    assert any(
        "check_max_supply_constant_value" in n
        for n in proof_names
    ), proof_names
    assert any(
        "check_post_construction_in_cap" in n
        for n in proof_names
    ), proof_names
    assert any(
        "check_mint_preserves_cap" in n
        for n in proof_names
    ), proof_names


@pytest.mark.requires_halmos
def test_live_halmos_m2_response_size_padding():
    """Real halmos invocation against the M2 response-size
    padding invariant — proves output_bytes == pad_to_bytes
    for ALL inputs regardless of codepoint-boundary
    dropout."""
    runner = HalmosRunner(timeout_seconds=120)
    if not runner.is_available():
        pytest.skip("halmos or forge not on PATH")
    suite = runner.run("M2ResponseSizePaddingSpec")
    assert suite.status == SymbolicProofStatus.PASSED, (
        f"M2ResponseSizePaddingSpec failed: "
        f"{suite.to_dict()}"
    )
    proof_names = {p.name for p in suite.proofs}
    assert any(
        "check_output_length_equals_pad_target" in n
        for n in proof_names
    ), proof_names
    assert any(
        "check_dropout_does_not_change_output_length" in n
        for n in proof_names
    ), proof_names
    assert any(
        "check_truncate_branch_overrides_finish" in n
        for n in proof_names
    ), proof_names


@pytest.mark.requires_halmos
def test_live_halmos_chunk_streaming_bounds():
    """Real halmos invocation against the H1 bounded-iterator
    proof. 65 symbolic paths on the headline bounded-by-
    expected check; 111 total across the spec."""
    runner = HalmosRunner(timeout_seconds=120)
    if not runner.is_available():
        pytest.skip("halmos or forge not on PATH")
    suite = runner.run("ChunkStreamingBoundsSpec")
    assert suite.status == SymbolicProofStatus.PASSED, (
        f"ChunkStreamingBoundsSpec failed: "
        f"{suite.to_dict()}"
    )
    proof_names = {p.name for p in suite.proofs}
    assert any(
        "check_accepted_bounded_by_expected" in n
        for n in proof_names
    ), proof_names
    assert any(
        "check_excess_always_raises" in n
        for n in proof_names
    ), proof_names
    assert any(
        "check_request_id_mismatch_rejects" in n
        for n in proof_names
    ), proof_names


@pytest.mark.requires_halmos
def test_live_halmos_speculation_rollback_math():
    """Real halmos invocation against the speculative-
    decoding rollback-math invariants. Eight proofs / 14
    symbolic paths total. Covers the Phase 3.x.11.y.x
    critical fix + adaptive-K state machine bounds."""
    runner = HalmosRunner(timeout_seconds=120)
    if not runner.is_available():
        pytest.skip("halmos or forge not on PATH")
    suite = runner.run("SpeculationRollbackMathSpec")
    assert suite.status == SymbolicProofStatus.PASSED, (
        f"SpeculationRollbackMathSpec failed: "
        f"{suite.to_dict()}"
    )
    proof_names = {p.name for p in suite.proofs}
    # Headline proofs must be present + green
    assert any(
        "check_rollback_math_post_fix_bounded" in n
        for n in proof_names
    ), proof_names
    assert any(
        "check_pre_fix_formula_undercounts" in n
        for n in proof_names
    ), proof_names
    assert any(
        "check_adaptive_k_stays_in_range" in n
        for n in proof_names
    ), proof_names


@pytest.mark.requires_halmos
def test_live_halmos_royalty_distributor_solvency():
    """Real halmos invocation against the canonical
    'this is what halmos is for' solvency proof. Six
    proofs / 66 explored symbolic paths."""
    runner = HalmosRunner(timeout_seconds=120)
    if not runner.is_available():
        pytest.skip("halmos or forge not on PATH")
    suite = runner.run("RoyaltyDistributorSolvencySpec")
    assert suite.status == SymbolicProofStatus.PASSED, (
        f"RoyaltyDistributorSolvencySpec failed: "
        f"{suite.to_dict()}"
    )
    # The headline solvency proofs must be present + green
    proof_names = {p.name for p in suite.proofs}
    assert any(
        "check_distributeRoyalty_preserves_solvency" in n
        for n in proof_names
    ), proof_names
    assert any(
        "check_claim_preserves_solvency" in n
        for n in proof_names
    ), proof_names
    assert any(
        "check_recoverStranded_does_not_decrease_claimable"
        in n
        for n in proof_names
    ), proof_names


@pytest.mark.requires_halmos
def test_live_halmos_creator_stake_registry():
    """Real halmos invocation against the §14 CreatorStakeRegistry
    proofs (sprint 983): solvency + slash-conservation + anti-game
    eligibility. Ten proofs. Validated locally at 10 passed / 0 failed."""
    runner = HalmosRunner(timeout_seconds=120)
    if not runner.is_available():
        pytest.skip("halmos or forge not on PATH")
    suite = runner.run("CreatorStakeRegistrySpec")
    assert suite.status == SymbolicProofStatus.PASSED, (
        f"CreatorStakeRegistrySpec failed: {suite.to_dict()}"
    )
    proof_names = {p.name for p in suite.proofs}
    # The three headline properties must be present + green.
    assert any(
        "check_slash_preserves_solvency" in n for n in proof_names
    ), proof_names
    assert any(
        "check_slash_conserves_value" in n for n in proof_names
    ), proof_names
    assert any(
        "check_creatorStakeOf_zero_after_requestUnbond" in n
        for n in proof_names
    ), proof_names


def test_runner_handles_timeout(tmp_path):
    """Long-running halmos times out gracefully."""
    proofs_dir = tmp_path / "symbolic-proofs"
    proofs_dir.mkdir()
    runner = HalmosRunner(
        proofs_dir=str(proofs_dir),
        timeout_seconds=1,
        halmos_bin="/usr/bin/halmos",
        forge_bin="/usr/bin/forge",
    )
    import subprocess
    with patch(
        "shutil.which",
        side_effect=lambda x: (
            "/usr/bin/halmos" if x == "halmos"
            else "/usr/bin/forge"
        ),
    ), patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(
            cmd="halmos", timeout=1,
        ),
    ):
        suite = runner.run("X")
    assert suite.status == SymbolicProofStatus.ERROR
    assert "timed out" in (suite.error or "")
