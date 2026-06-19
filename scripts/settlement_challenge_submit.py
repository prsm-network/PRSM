#!/usr/bin/env python3
"""Sprint 1166 — operator CLI: BROADCAST a prepared settlement challenge (USER-GATED).

This is the FINAL step of the fraud-defense loop. The prepare CLI
(scripts/settlement_challenge_prepare.py) is keyless + never-broadcast — it assembles +
read-only-dry-runs a challenge and stops at the call args. This CLI files it on-chain.

Broadcasting a challengeReceipt SPENDS GAS and SLASHES a provider's staked bond — it is
IRREVERSIBLE. Per the standing rule the assistant NEVER broadcasts: the OPERATOR runs this
with their OWN key. Three gates make an accidental slash hard:
  1. DEFAULT IS DRY-RUN ONLY. Without --broadcast it only prints the read-only dry-run verdict.
  2. --broadcast also requires --i-understand-this-broadcasts-and-may-slash (explicit ack).
  3. It refuses to broadcast a challenge whose dry-run FAILS (would revert / can't slash)
     unless --force is passed.
Plus the same Base-mainnet hard-guard as the other settlement-ops scripts (refuses chainId
8453 without --i-understand-mainnet).

It targets exactly ONE finding (--batch-id + --reason + --target-index are REQUIRED — no batch
broadcast). Receipts are re-sourced from the producer's own + the observer's verified store
(same as the prepare CLI). The challenger key comes from $PRSM_CHALLENGER_KEY (NEVER a flag).
For NO_ESCROW the key MUST be the requester's (the on-chain handler checks msg.sender).

Single-line shell — dry-run only (default, read-only):
  cd ~/Documents/GitHub/PRSM && PRSM_CHALLENGER_KEY=0x... python scripts/settlement_challenge_submit.py --rpc-url "$RPC" --registry-address 0xF8BEEb4362222b50109b6034767322B31aA92449 --batch-id 0x<id> --reason invalid_signature --target-index 1

Single-line shell — actually FILE the challenge (operator-run, irreversible):
  cd ~/Documents/GitHub/PRSM && PRSM_CHALLENGER_KEY=0x... python scripts/settlement_challenge_submit.py --rpc-url "$RPC" --registry-address 0xF8BEEb4362222b50109b6034767322B31aA92449 --batch-id 0x<id> --reason invalid_signature --target-index 1 --broadcast --i-understand-this-broadcasts-and-may-slash

Exit codes: 0 (dry-run shown OR broadcast succeeded); 1 runtime/refused (incl. the Base-mainnet
hard-guard, which aborts via SystemExit before any send); 2 misconfiguration (bad args / no key).
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

from prsm.settlement.challenge_broadcaster import gated_broadcast_prepared  # noqa: E402
from prsm.settlement.challenge_preparer import (  # noqa: E402
    composite_receipt_lookup,
    prepare_challenge_from_finding,
    published_batch_receipt_lookup,
)
from prsm.settlement.consensus_mismatch_assembler import (  # noqa: E402
    REASON_CONSENSUS_MISMATCH,
)
from prsm.settlement.e2e_proof import assert_chain_allowed  # noqa: E402
from prsm.settlement.published_batch_store import PublishedBatchStore  # noqa: E402
from prsm.settlement.settlement_audit_report import (  # noqa: E402
    SettlementAuditFindingsStore,
)


def _default_findings_file() -> str:
    env = (os.environ.get("PRSM_SETTLEMENT_AUDIT_FINDINGS_FILE", "") or "").strip()
    return env or str(Path.home() / ".prsm" / "settlement_audit_findings.json")


def _default_published_batches_file() -> str:
    env = (os.environ.get("PRSM_SETTLEMENT_AUDIT_STORE_FILE", "") or "").strip()
    return env or str(Path.home() / ".prsm" / "published_batches.json")


def _default_verified_batches_file() -> str:
    env = (os.environ.get("PRSM_SETTLEMENT_VERIFIED_BATCHES_FILE", "") or "").strip()
    return env or str(Path.home() / ".prsm" / "settlement_verified_batches.json")


def _select_one(records, args):
    """Find the SINGLE finding matching --batch-id + --reason + --target-index."""
    want_batch = args.batch_id.lower().removeprefix("0x")
    matches = [
        r for r in records
        if str(r.batch_id_hex).lower().removeprefix("0x") == want_batch
        and r.reason == args.reason
        and r.target_index == args.target_index
    ]
    return matches


def _build_keyed_submitter(reason: int, rpc_url: str, registry_address: str, key: str):
    """Build the keyed submitter for this reason: the generalized ChallengeSubmitter for
    reasons 0-3, the ConsensusChallengeSubmitter for reason 5 (CONSENSUS_MISMATCH). Both
    expose dry_run + submit over a prepared challenge's to_call_args()."""
    if reason == REASON_CONSENSUS_MISMATCH:
        from prsm.marketplace.consensus_submitter import ConsensusChallengeSubmitter
        return ConsensusChallengeSubmitter(
            rpc_url=rpc_url, registry_address=registry_address, private_key=key)
    from prsm.settlement.invalid_signature_submitter import ChallengeSubmitter
    return ChallengeSubmitter(
        rpc_url=rpc_url, registry_address=registry_address, private_key=key)


def _run(args) -> int:
    key = (os.environ.get("PRSM_CHALLENGER_KEY", "") or "").strip()
    if not key:
        print("ERROR: set PRSM_CHALLENGER_KEY to the challenger signing key (env only, never "
              "a flag). For NO_ESCROW it MUST be the requester's key.", file=sys.stderr)
        return 2
    if not args.rpc_url or not args.registry_address:
        print("ERROR: --rpc-url and --registry-address are required.", file=sys.stderr)
        return 2
    # Fail FAST (before any chain connection) if --broadcast lacks the explicit ack.
    if args.broadcast and not args.i_understand_this_broadcasts_and_may_slash:
        print("ERROR: --broadcast also requires --i-understand-this-broadcasts-and-may-slash "
              "(it spends gas + slashes a provider bond, irreversibly).", file=sys.stderr)
        return 2

    # 1. select the ONE finding + source its receipts (own + verified stores)
    records = SettlementAuditFindingsStore(Path(args.findings_file)).all_findings()
    matches = _select_one(records, args)
    if len(matches) != 1:
        print(f"ERROR: expected exactly ONE finding for batch_id={args.batch_id} "
              f"reason={args.reason} target_index={args.target_index}; found {len(matches)}.",
              file=sys.stderr)
        return 2
    lookup = composite_receipt_lookup(
        published_batch_receipt_lookup(PublishedBatchStore(Path(args.published_batches_file))),
        published_batch_receipt_lookup(PublishedBatchStore(Path(args.verified_batches_file))),
    )

    # 2. assemble (pure)
    prepared = prepare_challenge_from_finding(record=matches[0], receipt_lookup=lookup)

    # 3. build the keyed submitter + HARD-GUARD mainnet before any broadcast
    submitter = _build_keyed_submitter(
        prepared.on_chain_reason, args.rpc_url, args.registry_address, key)
    chain_id = int(submitter.web3.eth.chain_id)
    assert_chain_allowed(chain_id=chain_id, mainnet_acknowledged=args.i_understand_mainnet)

    # 4. gated broadcast: default is DRY-RUN ONLY; broadcast needs --broadcast + the ack flag
    # (both already validated above — the ack is checked fail-fast before any chain connection).
    broadcast_enabled = bool(args.broadcast and args.i_understand_this_broadcasts_and_may_slash)

    outcome = gated_broadcast_prepared(
        prepared, submitter=submitter, broadcast_enabled=broadcast_enabled, force=args.force)

    out = {"prepared": prepared.to_dict(), "chain_id": chain_id,
           "outcome": outcome.to_dict()}
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(f"reason={prepared.reason} on_chain_reason={prepared.on_chain_reason} "
              f"batch_id=0x{prepared.batch_id_hex} target_index={prepared.target_index} "
              f"chain_id={chain_id}")
        print(f"  dry_run={'OK' if outcome.dry_run_ok else 'FAIL'}"
              + (f" ({outcome.dry_run_reason})" if outcome.dry_run_reason else ""))
        if outcome.broadcast:
            print(f"  ✅ BROADCAST — tx {outcome.tx_hash_hex}")
        else:
            print(f"  (not broadcast) {outcome.refused_reason}")
            if not broadcast_enabled:
                print("  To file it: re-run with --broadcast "
                      "--i-understand-this-broadcasts-and-may-slash")
    # exit 0 when a dry-run was shown OR the broadcast succeeded; 1 on a failed broadcast
    if broadcast_enabled and not outcome.broadcast:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="settlement_challenge_submit.py",
        description=(
            "USER-GATED broadcast of a prepared settlement challenge. Default is DRY-RUN ONLY; "
            "broadcasting spends gas + slashes a bond (irreversible) and requires explicit "
            "flags + $PRSM_CHALLENGER_KEY. The assistant never runs this — the operator does."),
    )
    p.add_argument("--findings-file", default=_default_findings_file())
    p.add_argument("--published-batches-file", default=_default_published_batches_file())
    p.add_argument("--verified-batches-file", default=_default_verified_batches_file())
    p.add_argument("--batch-id", required=True, help="batch id of the finding (hex, 0x ok)")
    p.add_argument("--reason", required=True,
                   help="finding reason tag (invalid_signature/double_spend/no_escrow/"
                        "expired/consensus_mismatch)")
    p.add_argument("--target-index", type=int, required=True,
                   help="leaf/receipt index of the finding")
    p.add_argument("--rpc-url", default=os.environ.get("PRSM_RPC_URL"))
    p.add_argument("--registry-address", default=None,
                   help="BatchSettlementRegistry address")
    p.add_argument("--broadcast", action="store_true",
                   help="actually broadcast (default: dry-run only). Also requires "
                        "--i-understand-this-broadcasts-and-may-slash.")
    p.add_argument("--i-understand-this-broadcasts-and-may-slash", action="store_true",
                   dest="i_understand_this_broadcasts_and_may_slash",
                   help="explicit acknowledgement that --broadcast files an irreversible, "
                        "bond-slashing transaction")
    p.add_argument("--i-understand-mainnet", action="store_true",
                   dest="i_understand_mainnet",
                   help="acknowledge Base mainnet (chainId 8453); refused otherwise")
    p.add_argument("--force", action="store_true",
                   help="broadcast even if the dry-run fails (only for a verified false "
                        "negative, e.g. RPC lag)")
    p.add_argument("--json", action="store_true")
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
