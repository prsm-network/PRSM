#!/usr/bin/env python3
"""Sprint 1042 — Base-Sepolia settlement deposit→commit→finalize proof harness.

Drives the on-chain settlement rail end to end against Base Sepolia (chainId
84532): deposit FTNS into EscrowPool → accumulate a signed receipt → commit the
batch on-chain → (after the 3-day challenge window) finalize → assert the
recipient wallet FTNS rose and the requester escrow drained.

Because the challenge window is 3 days (same on testnet as mainnet), the live
proof is TWO-PHASE. The committed batch is persisted to a durable state file
(brick 2.5), so the two phases survive a restart / multi-day gap:

  PHASE 1 (now):
    PRSM_NETWORK=testnet PRIVATE_KEY=0x... python scripts/settlement_sepolia_e2e.py --phase deposit-commit --amount-ftns 1
  PHASE 2 (after the challenge window):
    PRSM_NETWORK=testnet PRIVATE_KEY=0x... python scripts/settlement_sepolia_e2e.py --phase finalize

A single key is a self-pay MECHANICAL proof (requester==provider==recipient): the
full on-chain path executes; funds leave self's escrow and return to self's
wallet. Optional: lower the challenge window first (owner-only, snapshotted at
commit) to finalize sooner, e.g. setChallengeWindowSeconds(3600).

TWO-PARTY proof (requester-payment, sprint 1058) — REAL cross-party value transfer:
requester A signs an EIP-712 PaymentAuthorization, provider B verifies it + commits;
finalize draws from A's escrow into B's wallet. Pass --two-party with --requester-key
(or env REQUESTER_PRIVATE_KEY) as A; PRIVATE_KEY is provider/settler B. A must hold
the FTNS to deposit (B's deposit funds A's escrow in this harness). Phase 2 finalize
is identical (the committed batch already names A as requester / B as provider):

  PHASE 1: PRSM_NETWORK=testnet PRIVATE_KEY=0x<B> REQUESTER_PRIVATE_KEY=0x<A> python scripts/settlement_sepolia_e2e.py --phase deposit-commit --two-party --amount-ftns 1
  PHASE 2: PRSM_NETWORK=testnet PRIVATE_KEY=0x<B> python scripts/settlement_sepolia_e2e.py --phase finalize

Env: PRIVATE_KEY (or FTNS_WALLET_PRIVATE_KEY / PRSM_DEPLOYER_PRIVATE_KEY) — the
signer; PRSM_NETWORK=testnet selects Base Sepolia. Mainnet (chainId 8453) is
REFUSED unless --i-understand-mainnet is passed (it sends real transactions).

Exit codes: 0 success/committed; 1 runtime failure; 2 env/validation error.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _read_key() -> str:
    for name in ("PRIVATE_KEY", "FTNS_WALLET_PRIVATE_KEY", "PRSM_DEPLOYER_PRIVATE_KEY"):
        v = (os.environ.get(name, "") or "").strip()
        if v:
            return v if v.startswith("0x") else "0x" + v
    print("ERROR: set PRIVATE_KEY (or FTNS_WALLET_PRIVATE_KEY / "
          "PRSM_DEPLOYER_PRIVATE_KEY) to the funded signer key", file=sys.stderr)
    raise SystemExit(2)


def _default_state_file() -> str:
    from pathlib import Path
    return str(Path.home() / ".prsm" / "settlement_e2e_state.json")


async def _amain(args) -> int:
    from prsm.config.networks import resolve_endpoints
    from prsm.economy.web3.escrow_pool_client import EscrowPoolClient
    from prsm.economy.web3.batch_settlement_contract_client import (
        Web3SettlementContractClient,
    )
    from prsm.settlement.accumulator import AccumulatorConfig, ReceiptAccumulator
    from prsm.settlement.client import BatchSettlementClient
    from prsm.settlement.state_store import SettlementStateStore
    from prsm.settlement.e2e_proof import (
        assert_chain_allowed,
        build_signed_batched_receipt,
        run_deposit_and_commit,
        run_finalize_and_verify,
        run_finalize_by_batch_id,
        run_two_party_deposit_and_commit,
    )
    from prsm.node.identity import generate_node_identity
    from eth_account import Account
    from web3 import Web3

    network = (os.environ.get("PRSM_NETWORK", "") or "testnet").strip() or "testnet"
    endpoints = resolve_endpoints(network=network)
    for field in ("rpc_url", "escrow_pool", "ftns_token", "settlement_registry"):
        if not getattr(endpoints, field, None):
            print(f"ERROR: {field} not configured for network '{network}'",
                  file=sys.stderr)
            return 2

    # Authoritative chain id from the live RPC; hard-guard mainnet.
    w3 = Web3(Web3.HTTPProvider(endpoints.rpc_url))
    chain_id = w3.eth.chain_id
    assert_chain_allowed(chain_id=chain_id, mainnet_acknowledged=args.i_understand_mainnet)
    # sp1042 review #8 — redact the RPC URL: managed providers embed an API key in
    # the path/query. Print only scheme+host (+ the authoritative chainId).
    from urllib.parse import urlsplit
    _u = urlsplit(endpoints.rpc_url)
    print(f"network={network} chainId={chain_id} rpc={_u.scheme}://{_u.netloc}")

    key = _read_key()
    signer = Account.from_key(key).address
    # Two-party (requester-payment) mode: A=requester (--requester-key /
    # REQUESTER_PRIVATE_KEY) signs the auth + funds escrow; B=signer is provider.
    requester_key = None
    requester_addr = signer
    if args.two_party:
        rk = (args.requester_key or os.environ.get("REQUESTER_PRIVATE_KEY", "") or "").strip()
        if not rk:
            print("ERROR: --two-party requires --requester-key or REQUESTER_PRIVATE_KEY "
                  "(the paying requester A; PRIVATE_KEY is provider B)", file=sys.stderr)
            return 2
        requester_key = rk if rk.startswith("0x") else "0x" + rk
        requester_addr = Account.from_key(requester_key).address
        if requester_addr.lower() == signer.lower():
            print("ERROR: --two-party requester and provider must differ (that's "
                  "just self-pay; drop --two-party)", file=sys.stderr)
            return 2
        print(f"requester(A)={requester_addr}  provider/settler(B)={signer}")
    else:
        print(f"signer={signer} (self-pay: requester==provider==recipient)")

    # Escrow client signs as the REQUESTER (A funds + approves their own escrow);
    # in self-pay A==B so this is the same key.
    escrow = EscrowPoolClient(
        rpc_url=endpoints.rpc_url, escrow_pool_address=endpoints.escrow_pool,
        ftns_token_address=endpoints.ftns_token,
        private_key=requester_key or key)
    contract = Web3SettlementContractClient(
        rpc_url=endpoints.rpc_url, contract_address=endpoints.settlement_registry,
        private_key=key)
    # count_threshold=1 → a single receipt is immediately a ready batch.
    settlement = BatchSettlementClient(
        accumulator=ReceiptAccumulator(AccumulatorConfig(count_threshold=1)),
        contract_client=contract, provider_address=signer,
        state_store=SettlementStateStore(args.state_file))

    amount_wei = int(round(args.amount_ftns * (10 ** 18)))
    if amount_wei <= 0:
        print("ERROR: --amount-ftns must be positive", file=sys.stderr)
        return 2

    if args.phase in ("deposit-commit", "full"):
        # sp1042 review #6 — idempotency: a committed-but-unfinalized batch already
        # in the durable state means phase 1 ran before. Re-depositing/committing
        # would double-deposit + create a second batch. Refuse + point at finalize.
        existing = [
            b for b in settlement.tracked_batches()
            if not settlement.is_finalized_locally(b.batch_id)
        ]
        if existing:
            print(f"ERROR: {len(existing)} committed-but-unfinalized batch(es) "
                  f"already in {args.state_file}; phase 1 already ran. Run "
                  f"--phase finalize after the challenge window (re-deposit would "
                  f"double-spend). Delete the state file only if you know the "
                  f"prior batch is dead.", file=sys.stderr)
            return 1
        identity = generate_node_identity(display_name="sepolia-e2e-settler")
        job_id = f"e2e-{int(time.time())}"
        if args.two_party:
            # Requester A signs an EIP-712 PaymentAuthorization bound to a synthetic
            # request; provider B verifies it (balance pre-flight OFF — escrow is
            # funded by the deposit that follows; finalize enforces funding), then
            # deposits A's escrow + commits naming A→B.
            from prsm.settlement.payment_client import build_payment_authorization
            from prsm.settlement.payment_authorization import inference_request_fields
            from prsm.settlement.payment_authorization_verifier import (
                InMemoryNonceStore, PaymentAuthorizationVerifier,
            )
            request_fields = inference_request_fields(
                model_id="gpt2", prompt=f"two-party-proof-{job_id}", max_tokens=16,
                privacy_tier="standard", content_tier="A")
            auth = build_payment_authorization(
                requester_key=requester_key, provider_address=signer,
                max_spend_ftns=args.amount_ftns, expiry_unix=int(time.time()) + 86400,
                **request_fields)
            verifier = PaymentAuthorizationVerifier(
                provider_address=signer, nonce_store=InMemoryNonceStore(),
                chain_id=chain_id, balance_reader=None)
            print(f"requester A signing auth + depositing {args.amount_ftns} FTNS, "
                  f"provider B committing batch …")
            report = await run_two_party_deposit_and_commit(
                escrow_client=escrow, settlement_client=settlement,
                verifier=verifier, payment_authorization=auth,
                request_fields=request_fields, provider_address=signer,
                identity=identity, job_id=job_id,
                output_hash=hashlib.sha256(job_id.encode()).hexdigest(),
                executed_at_unix=int(time.time()), value_wei=amount_wei,
                local_escrow_id=job_id, deposit_amount_wei=amount_wei)
            print(f"  verified requester={report['requester_address']} "
                  f"provider={report['provider_address']}")
        else:
            receipt = build_signed_batched_receipt(
                identity=identity, job_id=job_id,
                output_hash=hashlib.sha256(job_id.encode()).hexdigest(),
                executed_at_unix=int(time.time()),
                requester_address=signer, provider_address=signer,
                value_wei=amount_wei, local_escrow_id=job_id)
            print(f"depositing {args.amount_ftns} FTNS + committing batch …")
            report = await run_deposit_and_commit(
                escrow_client=escrow, settlement_client=settlement,
                batched_receipt=receipt, deposit_amount_wei=amount_wei,
                requester_address=signer)
        print(f"  deposit_tx={report['deposit_tx']}")
        print(f"  batch_id={report['batch_id']}")
        print(f"  escrow_balance={report['escrow_balance']} wei")
        batch_id = bytes.fromhex(report["batch_id"])
        if args.phase == "deposit-commit":
            # sp1043 — print the REAL countdown read from chain (the window is
            # snapshotted per-batch; don't hardcode a guess).
            try:
                secs = await contract.seconds_until_finalizable(batch_id)
                eta = "now" if secs <= 0 else f"~{secs/3600:.1f}h (at {secs}s)"
            except Exception:  # noqa: BLE001
                eta = "the challenge window"
            print(f"PHASE 1 done. Batch finalizable in {eta}. Re-run with "
                  f"--phase finalize then (same machine / --state-file), or "
                  f"--phase finalize --batch-id {report['batch_id']} from anywhere.")
            return 0

    if args.phase == "full":
        deadline = time.time() + args.poll_timeout_s
        print(f"polling isFinalizable (timeout {args.poll_timeout_s}s) …")
        while time.time() < deadline:
            if await contract.is_finalizable(batch_id):
                break
            await asyncio.sleep(args.poll_interval_s)

    # sp1043 — state-file-independent recovery: finalize a known batch_id straight
    # from chain state (no dependency on the local durable state file surviving
    # the multi-day gap).
    if args.batch_id:
        bid = bytes.fromhex(args.batch_id[2:] if args.batch_id.startswith("0x")
                            else args.batch_id)
        report = await run_finalize_by_batch_id(
            contract_client=contract, escrow_client=escrow, batch_id=bid)
        print(f"batch {bid.hex()[:12]}…: {report}")
        if report.get("success"):
            return 0
        secs = report.get("seconds_until_finalizable")
        if report.get("finalizable") is False and secs is not None:
            print(f"  not finalizable yet (~{secs/3600:.1f}h remaining)")
        return 1

    # sp1042 review #7 — adopt any commit that landed on-chain but isn't in
    # _tracked: a quarantined broadcast-but-unconfirmed commit (reconcile) or a
    # crash-orphaned intent (recover). Without this, that deposit's batch is
    # invisible to phase 2 and never finalizes → the deposit strands.
    try:
        adopted_p = await settlement.reconcile_pending_commits()
        adopted_r = await settlement.recover_committing_intents()
        if adopted_p or adopted_r:
            print(f"adopted {adopted_p} pending + {adopted_r} orphaned commit(s) "
                  f"into tracked before finalize")
    except Exception as exc:  # noqa: BLE001 - best-effort adoption
        print(f"WARN: reconcile/recover failed (continuing): {exc}", file=sys.stderr)

    # finalize phase: finalize every tracked-but-unfinalized batch
    tracked = [
        b for b in settlement.tracked_batches()
        if not settlement.is_finalized_locally(b.batch_id)
    ]
    if not tracked:
        print("no committed-but-unfinalized batch in state file "
              f"({args.state_file}); nothing to finalize", file=sys.stderr)
        return 1

    overall_ok = True
    for b in tracked:
        report = await run_finalize_and_verify(
            settlement_client=settlement, contract_client=contract,
            escrow_client=escrow, batch_id=b.batch_id,
            requester_address=b.requester_address, recipient_address=b.provider_address)
        print(f"batch {b.batch_id.hex()[:12]}…: {report}")
        if not report["finalizable"]:
            secs = report.get("seconds_until_finalizable")
            print(f"  not finalizable yet"
                  + (f" (~{secs}s remaining)" if secs is not None else ""))
            overall_ok = False
        elif not report["success"]:
            overall_ok = False

    return 0 if overall_ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("deposit-commit", "finalize", "full"),
                        default="deposit-commit",
                        help="deposit-commit (phase 1), finalize (phase 2), or "
                             "full (commit then poll+finalize — only viable if the "
                             "window is short)")
    parser.add_argument("--amount-ftns", type=float, default=1.0,
                        help="FTNS to deposit + settle (default 1.0)")
    parser.add_argument("--two-party", action="store_true",
                        help="requester-payment cross-party proof (sprint 1058): "
                             "requester A (--requester-key / REQUESTER_PRIVATE_KEY) "
                             "signs an EIP-712 auth, provider B (PRIVATE_KEY) verifies "
                             "+ commits; finalize draws A->B. Phase 1 only; phase 2 "
                             "finalize is identical.")
    parser.add_argument("--requester-key", default=None,
                        help="[--two-party] paying requester A's private key (or env "
                             "REQUESTER_PRIVATE_KEY); must differ from PRIVATE_KEY (B)")
    parser.add_argument("--state-file", default=_default_state_file(),
                        help="durable settlement state file bridging the two phases")
    parser.add_argument("--batch-id", default=None,
                        help="[finalize] finalize this specific batch straight from "
                             "chain state, no state file needed (recovery path if the "
                             "state file was lost between the two phases)")
    parser.add_argument("--poll-timeout-s", type=int, default=600,
                        help="[full] max seconds to poll isFinalizable")
    parser.add_argument("--poll-interval-s", type=int, default=15,
                        help="[full] seconds between isFinalizable polls")
    parser.add_argument("--i-understand-mainnet", action="store_true",
                        help="DANGER: allow Base mainnet (chainId 8453). Sends real "
                             "transactions. Default + intended target is Base Sepolia.")
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_amain(args))
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
