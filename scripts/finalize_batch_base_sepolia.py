#!/usr/bin/env python3
"""Finalize a settlement batch on Base (network-aware; sp1327 follow-up).

Standalone, repo-light (web3 + eth-account + prsm.config.networks) — run it AFTER the
registry's challenge window elapses to settle a PENDING batch: ``finalizeBatch(batchId)``
draws the requester's escrow → pays the batch's provider (the settler) its committed share.

The registry/FTNS/chainId are resolved from ``PRSM_NETWORK`` (``testnet`` → Base Sepolia
84532; ``mainnet`` → the F bundle on Base 8453), so the SAME script finalizes both the
testnet per-stage GO batches AND the mainnet production canary. RPC from BASE_RPC_URL
(mainnet) / BASE_SEPOLIA_RPC_URL (testnet), else a sensible default.

The 2-node cross-cloud testnet GO (2026-06-30) left two PENDING batches:
  HEAD   ca30cbc632a0ebee337c4b18c5c6c5e6399505d2618e2ca8c24bb1d5ff57271c  → Settler-A 0xBbEB…
  WORKER f98566d929da386d373c7072cc14ee5958d25cc015c06cf610fc977b64d868c3  → Settler-B 0x2010…

Usage (run once per batch, with that batch's settler key in env — NEVER on the CLI):
    SETTLER_KEY=0x<Settler-A key> python scripts/finalize_batch_base_sepolia.py \
        ca30cbc632a0ebee337c4b18c5c6c5e6399505d2618e2ca8c24bb1d5ff57271c
    SETTLER_KEY=0x<Settler-B key> python scripts/finalize_batch_base_sepolia.py \
        f98566d929da386d373c7072cc14ee5958d25cc015c06cf610fc977b64d868c3

Env:
  SETTLER_KEY            REQUIRED — the batch's provider/settler private key (gas + msg.sender).
  BASE_SEPOLIA_RPC_URL   optional — defaults to https://sepolia.base.org (PAYG avoids rate limits).

Read-only preview (no key, no tx): pass --check to print isFinalizable + secondsUntilFinalizable.

Chain guard: refuses any connected chainId != the PRSM_NETWORK-resolved chainId
(testnet → 84532 Base Sepolia; mainnet → 8453 Base) — set PRSM_NETWORK + the matching RPC.
"""
from __future__ import annotations

import os
import sys
import time

# Network-aware: resolve the registry/FTNS/chainId from PRSM_NETWORK
# (testnet → Base Sepolia 84532; mainnet → the F bundle on Base 8453). Falls back to
# the Base Sepolia constants if prsm.config isn't importable.
def _resolve_network():
    try:
        from prsm.config.networks import resolve_endpoints
        e = resolve_endpoints()
        return (str(e.settlement_registry), str(e.ftns_token), int(e.chain_id),
                str(getattr(e, "rpc_url", "") or ""))
    except Exception:
        return ("0xF8BEEb4362222b50109b6034767322B31aA92449",
                "0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a", 84532, "https://sepolia.base.org")


REGISTRY, FTNS, CHAIN_ID, _RPC_DEFAULT = _resolve_network()

_ABI = [
    {"inputs": [{"name": "batchId", "type": "bytes32"}], "name": "finalizeBatch",
     "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "batchId", "type": "bytes32"}], "name": "isFinalizable",
     "outputs": [{"name": "", "type": "bool"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "batchId", "type": "bytes32"}], "name": "secondsUntilFinalizable",
     "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "batchId", "type": "bytes32"}], "name": "batches",
     "outputs": [{"name": "", "type": "address"}, {"name": "", "type": "address"},
                 {"name": "", "type": "bytes32"}, {"name": "", "type": "uint8"},
                 {"name": "", "type": "uint256"}, {"name": "", "type": "uint256"},
                 {"name": "", "type": "uint256"}, {"name": "", "type": "uint8"}] + [
                 {"name": "", "type": "uint256"}] * 9,
     "stateMutability": "view", "type": "function"},
]
_ERC20 = [{"inputs": [{"name": "a", "type": "address"}], "name": "balanceOf",
           "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view",
           "type": "function"}]


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check_only = "--check" in sys.argv
    if not args:
        print("usage: SETTLER_KEY=0x.. python scripts/finalize_batch_base_sepolia.py <batchId-hex> [--check]",
              file=sys.stderr)
        return 2
    bid_hex = args[0][2:] if args[0].startswith("0x") else args[0]
    try:
        bid = bytes.fromhex(bid_hex)
        assert len(bid) == 32
    except Exception:
        print(f"ERROR: batchId must be 32-byte hex, got {args[0]!r}", file=sys.stderr)
        return 2

    from web3 import Web3
    rpc = (os.environ.get("BASE_RPC_URL", "").strip()
           or os.environ.get("BASE_SEPOLIA_RPC_URL", "").strip()
           or _RPC_DEFAULT)
    w3 = Web3(Web3.HTTPProvider(rpc))
    if w3.eth.chain_id != CHAIN_ID:
        print(f"ERROR: connected chainId {w3.eth.chain_id} != the PRSM_NETWORK-resolved chainId "
              f"{CHAIN_ID}; refusing (set PRSM_NETWORK + matching RPC).", file=sys.stderr)
        return 1
    reg = w3.eth.contract(address=Web3.to_checksum_address(REGISTRY), abi=_ABI)
    ftns = w3.eth.contract(address=Web3.to_checksum_address(FTNS), abi=_ERC20)

    b = reg.functions.batches(bid).call()
    provider, requester, value, status = b[0], b[1], int(b[4]), int(b[7])
    finalizable = reg.functions.isFinalizable(bid).call()
    secs = reg.functions.secondsUntilFinalizable(bid).call()
    statuses = {0: "NONE", 1: "PENDING", 2: "FINALIZED", 3: "SLASHED"}
    print(f"batch {bid.hex()}")
    print(f"  provider(settler): {provider}  value: {value/1e18} FTNS  status: {statuses.get(status, status)}")
    print(f"  isFinalizable: {finalizable}  secondsUntilFinalizable: {secs} (~{secs/3600:.1f}h)")
    if status == 2:
        print("  already FINALIZED — nothing to do.")
        return 0
    if status != 1:
        print(f"  status is not PENDING — cannot finalize.", file=sys.stderr)
        return 1
    if not finalizable:
        print(f"  NOT YET FINALIZABLE — wait ~{secs/3600:.1f}h (challenge window).")
        return 0 if check_only else 1
    if check_only:
        print("  ready to finalize. Re-run without --check (with SETTLER_KEY) to settle.")
        return 0

    key = os.environ.get("SETTLER_KEY", "").strip()
    if not key:
        print("ERROR: SETTLER_KEY env unset (the batch's provider key). Never pass it on the CLI.",
              file=sys.stderr)
        return 2
    if not key.startswith("0x"):
        key = "0x" + key
    from eth_account import Account
    acct = Account.from_key(key)
    if acct.address.lower() != provider.lower():
        print(f"ERROR: SETTLER_KEY address {acct.address} != batch provider {provider}. "
              f"Use THIS batch's settler key.", file=sys.stderr)
        return 1

    before = ftns.functions.balanceOf(Web3.to_checksum_address(provider)).call()
    tx = reg.functions.finalizeBatch(bid).build_transaction({
        "from": acct.address,
        "nonce": w3.eth.get_transaction_count(acct.address, "pending"),
        "gas": 500_000,
        "gasPrice": w3.eth.gas_price,
        "chainId": CHAIN_ID,
    })
    signed = w3.eth.account.sign_transaction(tx, acct.key)
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    txh = w3.eth.send_raw_transaction(raw)
    print(f"  finalize tx: {txh.hex()} — waiting for receipt...")
    rcpt = w3.eth.wait_for_transaction_receipt(txh, timeout=180)
    if rcpt.status != 1:
        print(f"  REVERTED (status 0) — tx {txh.hex()}", file=sys.stderr)
        return 1
    # Poll the post-tx state with retries — a single immediate read can hit an RPC node
    # that hasn't yet applied the block, falsely showing PENDING / an unchanged balance.
    after_status, after_bal = status, before
    for _ in range(10):
        time.sleep(3)
        after_status = int(reg.functions.batches(bid).call()[7])
        after_bal = ftns.functions.balanceOf(Web3.to_checksum_address(provider)).call()
        if after_status == 2:
            break
    ok = after_status == 2
    print(f"  {'✅ FINALIZED' if ok else '⚠️ tx mined but status still ' + statuses.get(after_status, str(after_status))}"
          f" — batch status: {statuses.get(after_status, after_status)}")
    delta = (after_bal - before) / 1e18
    # Self-pay (provider==requester) settles within escrow and the provider's wallet rises by
    # the share; a normal settlement also credits the provider. Either way report the delta.
    print(f"  provider FTNS wallet: {before/1e18} → {after_bal/1e18} (+{delta} FTNS)")
    print(f"  tx: {txh.hex()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
