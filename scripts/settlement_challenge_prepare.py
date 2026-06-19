#!/usr/bin/env python3
"""Sprint 1163 — operator CLI: PREPARE a settlement challenge from a persisted audit finding.

THE GAP this closes: the fraud-defense pipeline DEAD-ENDED at the findings surface. The
sp1150 CLI (settlement_audit_report.py) shows "here are N dispute candidates" and stops —
there was no command to turn a persisted ``AuditFindingRecord`` into the concrete on-chain
``challengeReceipt`` inputs an operator broadcasts. This CLI is that bridge: it loads the
persisted findings, re-sources each batch's receipts from a ``PublishedBatchStore``, ASSEMBLES
the right challenge (via prsm.settlement.challenge_preparer), optionally runs a READ-ONLY
dry-run, and prints the exact call args ready for the operator to broadcast.

NEVER broadcasts / signs / slashes. This CLI is KEYLESS by construction:
  - assembly is pure + offline (no chain, no key);
  - the optional --dry-run is a READ-ONLY static eth_call that needs only a ``from`` ADDRESS
    (NOT a private key) — it reuses ChallengeSubmitter.dry_run via its injection seam.
Broadcasting a prepared challenge is a SEPARATE, USER-GATED action the operator takes with
their OWN key via the challenge submitter, after reviewing the printed call args + dry-run
verdict. This script reaches NO ``submit`` / sign / broadcast path.

Reasons it prepares (== ChallengeSubmitter.SUPPORTED_CHALLENGE_REASONS): invalid_signature(1),
double_spend(0), no_escrow(2), expired(3). It does NOT prepare consensus_mismatch (its own
consensus_submitter) or §7 inference_receipt (off-chain reputation, no clean on-chain reason)
— those are skipped with a clear message.

RECEIPT SOURCE: a persisted finding carries only hashes/ids/indices (no receipt preimages),
so the assemblers' ordered receipt set is re-sourced from TWO PublishedBatchStore files,
composed (first hit wins):
  - --published-batches-file: the producer's/requester's OWN batches (the SAME store node.py
    writes on commit — default $PRSM_SETTLEMENT_AUDIT_STORE_FILE else ~/.prsm/published_batches.json);
  - --verified-batches-file: FOREIGN batches an observer audited + cross-checked on-chain (sp1164;
    default $PRSM_SETTLEMENT_VERIFIED_BATCHES_FILE else ~/.prsm/settlement_verified_batches.json).
A finding whose batch is in neither store is skipped MissingBatchReceipts (e.g. a foreign batch
this node never audited — re-fetch its receipt-set blob into the verified store first).

MODES (argparse):
  --findings-file PATH         persisted SettlementAuditFindingsStore (default the node path)
  --published-batches-file P   producer's/requester's OWN batches (the receipt source)
  --verified-batches-file P    observer-verified FOREIGN batches (the receipt source; sp1164)
  --batch-id 0x..              prepare only findings for this batch id (else: all)
  --reason TAG                 prepare only findings with this reason tag (else: all)
  --target-index N             prepare only the finding at this leaf/receipt index
  --dry-run                    ALSO run the read-only on-chain pre-flight (needs --rpc-url,
                               --registry-address, --from-address)
  --rpc-url / --registry-address / --from-address   for --dry-run (keyless, read-only)
  --json                       machine-readable output instead of the human render

Single-line shell, offline assemble (own + observer-verified batches, the defaults):
  cd ~/Documents/GitHub/PRSM && python scripts/settlement_challenge_prepare.py

Single-line shell, with the read-only on-chain dry-run (KEYLESS — a from-address, not a key):
  cd ~/Documents/GitHub/PRSM && python scripts/settlement_challenge_prepare.py --dry-run --rpc-url "$RPC" --registry-address 0xF8BEEb4362222b50109b6034767322B31aA92449 --from-address 0xYourChallengerAddress

Exit codes: 0 (incl. clean "nothing to prepare"); 1 runtime failure.
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

from prsm.settlement.challenge_preparer import (  # noqa: E402
    composite_receipt_lookup,
    dry_run_prepared,
    prepare_findings,
    published_batch_receipt_lookup,
)
from prsm.settlement.published_batch_store import PublishedBatchStore  # noqa: E402
from prsm.settlement.settlement_audit_report import (  # noqa: E402
    SettlementAuditFindingsStore,
)

_SUBMIT_NOTE = (
    "NOTE: this CLI NEVER broadcasts/signs/slashes — it only ASSEMBLES (+ optionally a "
    "READ-ONLY dry-run). To file a prepared challenge, broadcast it SEPARATELY with your "
    "OWN key via the challenge submitter, after reviewing the call args + dry-run verdict."
)


def _default_findings_file() -> str:
    env = (os.environ.get("PRSM_SETTLEMENT_AUDIT_FINDINGS_FILE", "") or "").strip()
    return env or str(Path.home() / ".prsm" / "settlement_audit_findings.json")


def _default_published_batches_file() -> str:
    # the SAME env + default node.py uses for the producer's PublishedBatchStore
    # (PRSM_SETTLEMENT_AUDIT_STORE_FILE), so the CLI reads the file the daemon writes.
    env = (os.environ.get("PRSM_SETTLEMENT_AUDIT_STORE_FILE", "") or "").strip()
    return env or str(Path.home() / ".prsm" / "published_batches.json")


def _default_verified_batches_file() -> str:
    # the SAME env + default node.py uses for the observer's verified-foreign-batch store
    # (sp1164), so the CLI reads the file the daemon's audit loop writes.
    env = (os.environ.get("PRSM_SETTLEMENT_VERIFIED_BATCHES_FILE", "") or "").strip()
    return env or str(Path.home() / ".prsm" / "settlement_verified_batches.json")


def _select(records, args):
    """Filter the loaded findings by the optional --batch-id / --reason / --target-index."""
    want_batch = (args.batch_id or "").lower().removeprefix("0x") or None
    out = []
    for r in records:
        if want_batch is not None and \
                str(r.batch_id_hex).lower().removeprefix("0x") != want_batch:
            continue
        if args.reason and r.reason != args.reason:
            continue
        if args.target_index is not None and r.target_index != args.target_index:
            continue
        out.append(r)
    return out


def _build_keyless_dry_run_client(rpc_url: str, registry_address: str, from_address: str):
    """A KEYLESS read-only dry-run client: a static eth_call needs only a ``from`` address,
    not a private key. Reuses ChallengeSubmitter.dry_run via its (web3, registry, account)
    injection seam — ``account`` is a from-address-only stub, so NO key is ever needed/held
    and NO broadcast/sign path is reachable."""
    from web3 import Web3

    from prsm.settlement.invalid_signature_submitter import (
        CHALLENGE_RECEIPT_ABI,
        ChallengeSubmitter,
    )

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    registry = w3.eth.contract(
        address=Web3.to_checksum_address(registry_address), abi=CHALLENGE_RECEIPT_ABI)

    class _FromOnly:
        address = Web3.to_checksum_address(from_address)

    return ChallengeSubmitter(web3=w3, registry=registry, account=_FromOnly())


def _maybe_dry_run(prepared_list, args):
    """If --dry-run, read-only pre-flight each prepared challenge. Returns
    {batch_id_hex+idx -> verdict-dict}. Requires --rpc-url/--registry-address/--from-address."""
    if not args.dry_run:
        return {}
    missing = [n for n, v in (
        ("--rpc-url", args.rpc_url), ("--registry-address", args.registry_address),
        ("--from-address", args.from_address)) if not v]
    if missing:
        raise SystemExit(
            f"--dry-run requires {', '.join(missing)} (read-only; a from-address, not a key)")
    client = _build_keyless_dry_run_client(
        args.rpc_url, args.registry_address, args.from_address)
    verdicts = {}
    for p in prepared_list:
        v = dry_run_prepared(p, client)  # READ-ONLY static call; never submit
        verdicts[(p.batch_id_hex, p.target_index)] = {
            "would_succeed": bool(getattr(v, "would_succeed", False)),
            "revert_reason": getattr(v, "revert_reason", None),
        }
    return verdicts


def _emit(prepared_list, skipped, verdicts, as_json: bool) -> None:
    if as_json:
        out = {
            "prepared": [
                {**p.to_dict(), "dry_run": verdicts.get((p.batch_id_hex, p.target_index))}
                for p in prepared_list
            ],
            "skipped": [s.to_dict() for s in skipped],
            "note": _SUBMIT_NOTE,
        }
        print(json.dumps(out, indent=2, sort_keys=True))
        return
    if not prepared_list and not skipped:
        print("settlement challenge prepare: no findings to prepare")
        return
    print(f"prepared {len(prepared_list)} challenge(s); skipped {len(skipped)}")
    for p in prepared_list:
        v = verdicts.get((p.batch_id_hex, p.target_index))
        verdict = (
            "" if v is None
            else f"  dry_run={'OK' if v['would_succeed'] else 'FAIL'}"
                 + (f" ({v['revert_reason']})" if v.get("revert_reason") else ""))
        print(f"  + reason={p.reason} on_chain_reason={p.on_chain_reason} "
              f"batch_id=0x{p.batch_id_hex} target_index={p.target_index}{verdict}")
    for s in skipped:
        print(f"  - SKIPPED reason={s.reason} batch_id=0x{s.batch_id_hex} "
              f"target_index={s.target_index}: {s.message}")
    print(_SUBMIT_NOTE)


def _run(args) -> int:
    findings_store = SettlementAuditFindingsStore(Path(args.findings_file))
    records = _select(findings_store.all_findings(), args)
    # Source receipts from the producer's/requester's OWN store AND the observer's
    # verified-foreign-batch store (sp1164) — first-hit wins — so findings on third-party
    # batches the observer audited prepare too.
    own_lookup = published_batch_receipt_lookup(
        PublishedBatchStore(Path(args.published_batches_file)))
    verified_lookup = published_batch_receipt_lookup(
        PublishedBatchStore(Path(args.verified_batches_file)))
    lookup = composite_receipt_lookup(own_lookup, verified_lookup)
    prepared_list, skipped = prepare_findings(records, lookup)
    verdicts = _maybe_dry_run(prepared_list, args)
    _emit(prepared_list, skipped, verdicts, args.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="settlement_challenge_prepare.py",
        description=(
            "PREPARE a settlement challengeReceipt from a persisted audit finding. "
            "KEYLESS + NEVER broadcasts — assembles (+ optional read-only dry-run); "
            "disputing is a separate, user-gated broadcast with your own key."),
    )
    p.add_argument("--findings-file", default=_default_findings_file(),
                   help="persisted SettlementAuditFindingsStore (default the node path)")
    p.add_argument("--published-batches-file", default=_default_published_batches_file(),
                   help="PublishedBatchStore file — the producer's/requester's OWN batches "
                        "(default $PRSM_SETTLEMENT_AUDIT_STORE_FILE)")
    p.add_argument("--verified-batches-file", default=_default_verified_batches_file(),
                   help="verified-foreign-batch store — batches an observer audited (sp1164; "
                        "default $PRSM_SETTLEMENT_VERIFIED_BATCHES_FILE). Composed with "
                        "--published-batches-file so observer findings prepare too.")
    p.add_argument("--batch-id", default=None,
                   help="prepare only findings for this batch id (hex, 0x optional)")
    p.add_argument("--reason", default=None,
                   help="prepare only findings with this reason tag "
                        "(invalid_signature/double_spend/no_escrow/expired)")
    p.add_argument("--target-index", type=int, default=None,
                   help="prepare only the finding at this leaf/receipt index")
    p.add_argument("--dry-run", action="store_true",
                   help="ALSO run the read-only on-chain pre-flight (keyless; needs "
                        "--rpc-url/--registry-address/--from-address)")
    p.add_argument("--rpc-url", default=os.environ.get("PRSM_RPC_URL"),
                   help="JSON-RPC endpoint for --dry-run (read-only)")
    p.add_argument("--registry-address", default=None,
                   help="BatchSettlementRegistry address for --dry-run")
    p.add_argument("--from-address", default=None,
                   help="the address you would broadcast from (a from-address, NOT a key; "
                        "for NO_ESCROW this MUST be the requester)")
    p.add_argument("--json", action="store_true",
                   help="machine-readable output instead of the human render")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — uniform non-zero, no stack spew
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
