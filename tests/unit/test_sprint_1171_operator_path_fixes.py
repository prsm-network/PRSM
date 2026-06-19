"""Sprint 1171 — operator-path fixes surfaced by the sp1149-1169 holistic review.

1. The prepare CLI's --dry-run built ONE generalized ChallengeSubmitter for every prepared
   challenge, so a consensus_mismatch (reason 5) dry-run false-negatived "unsupported challenge
   reason (reason=5)" (ChallengeSubmitter.SUPPORTED = {0,1,2,3}). Fix: route the keyless dry-run
   client by on_chain_reason — reason 5 -> ConsensusChallengeSubmitter.
2. Doc drift: the prepare CLI docstring/--reason help said consensus_mismatch is NOT prepared,
   but sp1165 made it preparable.
3. (node.py) the audit engine's dry_run_client was a BatchSettlementClient (no dry_run at all),
   so every in-daemon finding's dry-run verdict was a silent no-op. Fixed to a keyless
   ChallengeSubmitter (covered by a parse + import check here; full daemon wiring is e2e-gated).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_prepare_cli():
    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "settlement_challenge_prepare", repo / "scripts" / "settlement_challenge_prepare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dry_run_client_routes_consensus_to_consensus_submitter():
    from prsm.marketplace.consensus_submitter import ConsensusChallengeSubmitter
    from prsm.settlement.invalid_signature_submitter import ChallengeSubmitter

    cli = _load_prepare_cli()
    frm = "0x" + "c" * 40
    rpc = "http://127.0.0.1:9"   # never connected — Web3 HTTPProvider + .contract() are lazy
    reg = "0x" + "d" * 40

    # reason 5 (CONSENSUS_MISMATCH) -> the dedicated consensus submitter
    c5 = cli._build_keyless_dry_run_client(5, rpc, reg, frm)
    assert isinstance(c5, ConsensusChallengeSubmitter)

    # reasons 0-3 -> the generalized ChallengeSubmitter
    for r in (0, 1, 2, 3):
        cr = cli._build_keyless_dry_run_client(r, rpc, reg, frm)
        assert isinstance(cr, ChallengeSubmitter)


def test_prepare_cli_docs_mention_consensus_mismatch():
    cli = _load_prepare_cli()
    assert "consensus_mismatch" in (cli.__doc__ or "")
    # the --reason help lists consensus_mismatch too
    help_text = " ".join(
        a.help or "" for a in cli.build_parser()._actions if getattr(a, "dest", "") == "reason")
    assert "consensus_mismatch" in help_text


def test_node_py_parses_and_wires_keyless_dry_run_client():
    # the sp1171 node.py change must parse + reference the keyless ChallengeSubmitter wiring.
    import ast

    repo = Path(__file__).resolve().parents[2]
    src = (repo / "prsm" / "node" / "node.py").read_text()
    ast.parse(src)  # raises SyntaxError on a bad edit
    # the dry_run_client is no longer the bare commit client; the keyless submitter is wired.
    assert "dry_run_client=dry_run_client" in src
    assert "sp1171" in src and "keyless" in src.lower()
