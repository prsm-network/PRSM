#!/usr/bin/env python3
"""Sprint 1406 — deposit FTNS into the EscrowPool for the requester (payer) side of an on-chain
settlement. The requester's escrow is what finalizeBatch draws to pay the provider, so it MUST be
funded before finalize. Approve (only if allowance is short) + deposit, in one call.

Network-aware: escrow pool / FTNS / chainId / RPC resolve from PRSM_NETWORK (testnet → Base Sepolia
84532, mainnet → Base 8453). Chain guard refuses a connected chainId != the resolved one.

Usage:
  # read-only preview (no key, no tx): current escrow + wallet FTNS balance
  PRSM_NETWORK=mainnet python scripts/deposit_escrow.py --check

  # deposit 1 FTNS (key from env ONLY — never on the command line)
  PRSM_NETWORK=mainnet ESCROW_DEPOSIT_KEY=0x<payer key> python scripts/deposit_escrow.py 1.0

Env:
  ESCROW_DEPOSIT_KEY   the PAYER's private key (gas + msg.sender). Required for a deposit, not --check.
  PRSM_NETWORK         testnet | mainnet (default per config).
  BASE_RPC_URL         optional RPC override.
"""
import asyncio
import os
import sys


def _resolve():
    from prsm.config.networks import resolve_endpoints, get_network_config
    e = resolve_endpoints()
    cfg = get_network_config()
    escrow = str(getattr(cfg, "escrow_pool", "") or "")
    return (escrow, str(e.ftns_token), int(e.chain_id),
            os.environ.get("BASE_RPC_URL", "") or str(getattr(e, "rpc_url", "") or ""))


async def main() -> int:
    escrow_addr, ftns_addr, chain_id, rpc = _resolve()
    if not escrow_addr:
        print("ERROR: no escrow_pool for the resolved network", file=sys.stderr)
        return 2
    check = "--check" in sys.argv
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]

    from prsm.economy.web3.escrow_pool_client import EscrowPoolClient
    key = (os.environ.get("ESCROW_DEPOSIT_KEY", "") or "").strip() if not check else None
    client = EscrowPoolClient(
        rpc_url=rpc, escrow_pool_address=escrow_addr,
        ftns_token_address=ftns_addr, private_key=key)

    # chain guard
    connected = await asyncio.to_thread(lambda: int(client.web3.eth.chain_id))
    if connected != chain_id:
        print(f"ERROR: connected chainId {connected} != resolved {chain_id} "
              f"(set PRSM_NETWORK + matching RPC)", file=sys.stderr)
        return 2

    who = client.address if getattr(client, "address", None) else None
    if check:
        if not who:
            who = (os.environ.get("PAYER_ADDRESS", "") or "").strip() or None
        if not who:
            print("--check needs PAYER_ADDRESS=0x... (or a key) to read a balance", file=sys.stderr)
            return 2
        esc = await client.balance_of(who)
        wal = await client.ftns_balance_of(who)
        print(f"payer {who} @ chainId {chain_id}")
        print(f"  escrow deposited : {esc/1e18:.4f} FTNS  (escrow pool {escrow_addr})")
        print(f"  wallet FTNS      : {wal/1e18:.4f} FTNS")
        return 0

    if not key:
        print("ERROR: ESCROW_DEPOSIT_KEY required to deposit (never pass a key on argv)",
              file=sys.stderr)
        return 2
    if not pos:
        print("ERROR: pass the amount to deposit, e.g. `... deposit_escrow.py 1.0`", file=sys.stderr)
        return 2
    amount_wei = int(round(float(pos[0]) * (10 ** 18)))
    before = await client.balance_of(who)
    print(f"payer {who}: escrow {before/1e18:.4f} → depositing {pos[0]} FTNS on chainId {chain_id}…")
    tx = await client.deposit(amount_wei)
    after = await client.balance_of(who)
    print(f"  deposit tx: {tx}")
    print(f"  escrow now: {after/1e18:.4f} FTNS (+{(after-before)/1e18:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
