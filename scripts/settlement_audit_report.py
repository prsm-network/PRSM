#!/usr/bin/env python3
"""Sprint 1150 — operator CLI for the settlement fraud-defense AUDIT surface.

THE GAP this closes: the settlement fraud-defense system is CODE-COMPLETE (all 5
on-chain reason codes — DOUBLE_SPEND, INVALID_SIGNATURE, NO_ESCROW, EXPIRED,
CONSENSUS_MISMATCH — plus the §7 compute-integrity class) and, since sp1149,
operationally VISIBLE *inside* the node daemon (the SettlementAuditScheduler
WARNING-logs + persists dispute candidates when PRSM_SETTLEMENT_AUDIT is on). But
there was NO operator entry point OUTSIDE the daemon: an operator could not run a
one-off audit scan, nor review the persisted findings, without booting the full
node. This script is that entry point — mirroring scripts/settlement_sepolia_e2e.py
(the established settlement-ops pattern).

READ-ONLY by construction. This CLI NEVER broadcasts, signs, or slashes: it only
LOADS the persisted findings store and renders it. It constructs NO ChallengeSubmitter
and reaches NO broadcast/sign path. Disputing a candidate is a SEPARATE, USER-GATED
action the operator takes with their own key via the challenge submitter — the render()
output always carries that pointer.

MODES (argparse):
  --findings-file PATH   Load the persisted SettlementAuditFindingsStore at PATH
                         (default: $PRSM_SETTLEMENT_AUDIT_FINDINGS_FILE, else
                         ~/.prsm/settlement_audit_findings.json — the node path),
                         summarize_records(store.all_findings()), print render().
                         Missing/empty file -> clean "no dispute candidates" (exit 0).
                         This is the default mode.
  --run-once             LIVE scanning is driven by the NODE DAEMON, not this CLI: a
                         standalone scan needs a chain client + ContentProvider +
                         selector that the node builds in initialize(), and
                         reconstructing them here would duplicate node wiring. So this
                         flag prints the supported live-scan path (run the node with
                         PRSM_SETTLEMENT_AUDIT=1) and exits non-zero rather than faking a
                         scan. Review what the daemon persisted with --findings-file.
  --json                 Print summary.to_dict() as JSON (for tooling) instead of the
                         human render().

Single-line shell, review the persisted findings:
  cd ~/Documents/GitHub/PRSM && python scripts/settlement_audit_report.py --findings-file ~/.prsm/settlement_audit_findings.json

Exit codes: 0 success (incl. clean "no findings"); 1 runtime failure;
2 --run-once (live scanning is via the node daemon — actionable message, no stack trace).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# NOTE: this CLI is READ-ONLY. It imports the sp1149 report module (pure summarize
# + render + the persisted store) ONLY. It never imports/constructs a challenge
# submitter and never reaches a broadcast/sign/finalize path.
from prsm.settlement.settlement_audit_report import (  # noqa: E402
    SettlementAuditFindingsStore,
    summarize_records,
)


def _default_findings_file() -> str:
    """The findings-store path node.py uses: $PRSM_SETTLEMENT_AUDIT_FINDINGS_FILE if
    set, else ~/.prsm/settlement_audit_findings.json (the documented node default)."""
    env = (os.environ.get("PRSM_SETTLEMENT_AUDIT_FINDINGS_FILE", "") or "").strip()
    if env:
        return env
    return str(Path.home() / ".prsm" / "settlement_audit_findings.json")


def _emit(summary, as_json: bool) -> None:
    """Print the summary — either the operator render() (default) or to_dict() JSON.

    render() already carries the explicit user-gated-submit pointer, so the operator
    reading the output knows this surface NEVER broadcasts and that disputing is a
    separate deliberate action taken with their own key."""
    if as_json:
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    else:
        print(summary.render())


def _run_findings_file(args) -> int:
    """READ-ONLY review mode: load the persisted store + render it. A missing/empty
    file is NOT an error — it prints the clean "no dispute candidates" line (exit 0)."""
    path = Path(args.findings_file)
    # SettlementAuditFindingsStore NEVER-RAISES on a missing/unreadable file: it simply
    # loads empty. So a missing file naturally yields an empty summary -> "no dispute
    # candidates". (We do not pre-check existence; the store's own never-raise discipline
    # is the contract we lean on.)
    store = SettlementAuditFindingsStore(path)
    summary = summarize_records(store.all_findings())
    _emit(summary, args.json)
    return 0


def _run_once(args) -> int:
    """LIVE scanning is driven by the NODE DAEMON, not this CLI.

    A standalone scan needs a real chain client + ContentProvider + selector inputs that
    the node builds in initialize(); reconstructing them in a bare script would duplicate
    node wiring (and risk a subtly-misconfigured scan). Rather than fake a scan or dump a
    stack trace, this surfaces the supported live-scan path (the daemon) and exits non-zero.
    This CLI's supported, read-only capability is the persisted-findings review
    (--findings-file). Either way the CLI never broadcasts/signs."""
    print(
        "ERROR: standalone --run-once live scanning is not supported by this CLI.\n"
        "The node daemon is the live-scan driver: run the node with "
        "PRSM_SETTLEMENT_AUDIT=1 (it drives the read-only, never-broadcast audit on its "
        "block-cursor schedule and persists dispute candidates to "
        "$PRSM_SETTLEMENT_AUDIT_FINDINGS_FILE), then review them here with --findings-file.",
        file=sys.stderr,
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="settlement_audit_report.py",
        description=(
            "READ-ONLY operator surface for the settlement fraud-defense audit. Reviews "
            "persisted dispute candidates (or drives one read-only dry-run scan). NEVER "
            "broadcasts/signs/slashes — disputing a candidate is a separate, user-gated "
            "action taken with your own key via the challenge submitter."
        ),
    )
    p.add_argument(
        "--findings-file",
        default=_default_findings_file(),
        help=(
            "Path to the persisted SettlementAuditFindingsStore "
            "(default: $PRSM_SETTLEMENT_AUDIT_FINDINGS_FILE, else "
            "~/.prsm/settlement_audit_findings.json). Missing/empty -> 'no dispute "
            "candidates' (exit 0)."
        ),
    )
    p.add_argument(
        "--run-once",
        action="store_true",
        help=(
            "Print the supported live-scan path and exit non-zero: standalone live "
            "scanning is NOT supported by this CLI — the node daemon "
            "(PRSM_SETTLEMENT_AUDIT=1) drives the read-only scan. Use --findings-file "
            "to review what it persisted."
        ),
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print summary.to_dict() as JSON instead of the human render().",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.run_once:
        return _run_once(args)
    return _run_findings_file(args)


if __name__ == "__main__":
    raise SystemExit(main())
