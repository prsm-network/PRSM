#!/usr/bin/env python3
"""Front-door e2e validation on Base Sepolia (sprint 1200) — the single-command proof.

Runs the day-one-live front door against a LIVE node + the live Base-Sepolia chain:

    faucet (optional) -> deposit -> pay-infer

and asserts each step on-chain (faucet recipient balance rose, escrow funded, the paid
inference returned real output + a signed receipt and was ACCEPTED, not 402). Mirrors
scripts/settlement_sepolia_e2e.py. TESTNET-ONLY: refuses any chain but Base Sepolia.

Prereqs (operator, in another terminal):
    cd ~/Documents/GitHub/PRSM && pip install -e '.[ml]'
    PRSM_NETWORK=testnet PRSM_INFERENCE_EXECUTOR=local PRSM_REQUESTER_PAYMENT=1 \
      PRSM_ONCHAIN_SETTLEMENT=1 FTNS_WALLET_PRIVATE_KEY=0x... prsm node start

Then (requester):
    PRIVATE_KEY=0x... python scripts/frontdoor_e2e_sepolia.py --prompt "Hello, PRSM"
    # add --faucet-address 0xFRESH to also prove the new-user faucet on-ramp.

Exit 0 on overall success, 1 on a failed assertion / runtime error, 2 on bad env.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from urllib.parse import urlparse

# Allow running as a bare script (python scripts/...) without an install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read_key() -> str:
    for name in ("PRIVATE_KEY", "FTNS_WALLET_PRIVATE_KEY", "PRSM_DEPLOYER_PRIVATE_KEY"):
        v = (os.environ.get(name) or "").strip()
        if v:
            return v if v.startswith("0x") else "0x" + v
    print("ERROR: set PRIVATE_KEY (or FTNS_WALLET_PRIVATE_KEY) to the requester's "
          "wallet key.", file=sys.stderr)
    raise SystemExit(2)


def _redact(url: str) -> str:
    try:
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}"
    except Exception:
        return "<rpc>"


def main() -> int:
    ap = argparse.ArgumentParser(description="Front-door e2e validation on Base Sepolia.")
    ap.add_argument("--prompt", default="Hello from a real PRSM user")
    ap.add_argument("--amount-ftns", type=float, default=5.0,
                    help="FTNS to deposit into the EscrowPool (default 5.0)")
    ap.add_argument("--budget", type=float, default=1.0,
                    help="FTNS budget / max-spend for the inference (default 1.0)")
    ap.add_argument("--model", default="distilgpt2",
                    help="model_id (default distilgpt2 — what `PRSM_INFERENCE_EXECUTOR=local` "
                         "serves out of the box; pass the model your node actually loads)")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--faucet-address", default=None,
                    help="If set, first dispense testnet FTNS to this address (proves "
                         "the new-user faucet on-ramp).")
    ap.add_argument("--faucet-amount", type=float, default=None)
    ap.add_argument("--api-url", default="http://127.0.0.1:8000",
                    help="The running PRSM node's API URL (default loopback).")
    ap.add_argument("--verify-pubkey-b64", default=None,
                    help="Operator public key (base64) — verifies the receipt signature.")
    ap.add_argument("--confirm-attempts", type=int, default=8)
    ap.add_argument("--confirm-delay-s", type=float, default=2.0)
    args = ap.parse_args()

    try:
        from eth_account import Account
        from web3 import Web3
        from prsm.config.networks import resolve_endpoints
        from prsm.economy.web3.escrow_pool_client import EscrowPoolClient
        from prsm.economy.web3.ftns_balance_reader import ReadOnlyFTNSBalanceReader
        from prsm.sdk.client import PRSMClient
        from prsm.node.frontdoor_e2e import (
            run_all, assert_testnet_only, FrontDoorValidationError,
            BASE_SEPOLIA_CHAIN_ID,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: missing dependency ({exc}). Install web3 + eth-account + the repo.",
              file=sys.stderr)
        return 2

    network = (os.environ.get("PRSM_NETWORK") or "testnet").strip()
    try:
        ep = resolve_endpoints(network)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not resolve endpoints for network={network!r}: {exc}",
              file=sys.stderr)
        return 2
    if not (ep.rpc_url and ep.escrow_pool and ep.ftns_token):
        print(f"ERROR: network {network!r} is missing rpc/escrow/ftns config.",
              file=sys.stderr)
        return 2

    key = _read_key()
    try:
        requester_address = Account.from_key(key).address
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: bad private key: {exc}", file=sys.stderr)
        return 2

    # Verify the LIVE chain is Base Sepolia (defense-in-depth with the harness guard).
    try:
        w3 = Web3(Web3.HTTPProvider(ep.rpc_url, request_kwargs={"timeout": 20}))
        live_chain = int(w3.eth.chain_id)
        assert_testnet_only(chain_id=live_chain)
    except FrontDoorValidationError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not read chainId from {_redact(ep.rpc_url)}: {exc}",
              file=sys.stderr)
        return 2

    print(f"network={network} chainId={live_chain} rpc={_redact(ep.rpc_url)}")
    print(f"requester={requester_address}  api={args.api_url}")
    if args.faucet_address:
        print(f"faucet -> {args.faucet_address}")

    node_api_key = (os.environ.get("PRSM_NODE_API_KEY") or "").strip()
    client = PRSMClient(base_url=args.api_url, api_key=node_api_key)
    escrow_client = EscrowPoolClient(ep.rpc_url, ep.escrow_pool, ep.ftns_token)
    balance_reader = ReadOnlyFTNSBalanceReader(rpc_url=ep.rpc_url, token_address=ep.ftns_token)

    def faucet_post(body: dict) -> dict:
        import httpx
        headers = {"X-API-Key": node_api_key} if node_api_key else {}
        r = httpx.post(f"{args.api_url}/ftns/faucet/onchain", json=body,
                       timeout=180.0, headers=headers)
        if r.status_code != 200:
            try:
                detail = r.json().get("detail", r.text[:300])
            except Exception:
                detail = r.text[:300]
            return {"detail": f"HTTP {r.status_code}: {detail}"}
        return r.json()

    payinfer_kwargs = {
        "model_id": args.model,
        "max_tokens": args.max_tokens,
        "budget_ftns": args.budget,
        "max_spend_ftns": args.budget,
        "chain_id": BASE_SEPOLIA_CHAIN_ID,
    }

    async def _go():
        try:
            return await run_all(
                chain_id=live_chain,
                client=client, escrow_client=escrow_client, balance_reader=balance_reader,
                requester_key=key, requester_address=requester_address,
                prompt=args.prompt, deposit_amount_ftns=args.amount_ftns,
                faucet_post=(faucet_post if args.faucet_address else None),
                faucet_recipient=args.faucet_address, faucet_amount_ftns=args.faucet_amount,
                verify_pubkey_b64=args.verify_pubkey_b64, payinfer_kwargs=payinfer_kwargs,
                confirm_attempts=args.confirm_attempts, confirm_delay_s=args.confirm_delay_s,
            )
        finally:
            await client.close()

    try:
        report = asyncio.run(_go())
    except FrontDoorValidationError as exc:
        print(f"\nFAIL — {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR — {exc}", file=sys.stderr)
        return 1

    print("\n── front-door e2e report ──")
    for s in report["steps"]:
        mark = "PASS" if s.get("ok") else "FAIL"
        print(f"  [{mark}] {s['step']}: {s.get('note', '')}")
        if s["step"] == "pay-infer":
            print(f"         output: {s.get('output')!r}  charged: {s.get('ftns_charged')} "
                  f"receipt_verified: {s.get('receipt_verified')}")
    ok = bool(report.get("success"))
    print(f"\noverall: {'PASS — the front door works end-to-end on Base Sepolia' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
