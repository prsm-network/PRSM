#!/usr/bin/env python3
"""
Generate the Foundation Safe transaction that PAUSES FTNS emissions by calling
EmissionController.pauseMinting().

Pre-launch posture (see docs/2026-06-05-pause-emissions-ceremony.md): with the
network not yet live and no participants to reward, emissions should be OFF until
launch. pauseMinting() is owner-only and reversible (resumeMinting() turns
emissions back on at launch). Pausing hard-blocks the permissionless
CompensationDistributor.pullAndDistribute() mint path (mintAuthorized is
whenNotPaused), keeping circulating supply at the genesis 100M.

This is the LEAST consequential of the economy ceremonies: a no-argument,
owner-only, reversible CALL that HALTS minting (a conservative action). It carries
no role/address/amount that could be fat-fingered — the only thing to confirm is
that the target is the EmissionController and the method is pauseMinting().

Modes:

  prep      — print Safe-UI parameters + write the import JSON (offline; default)
  preflight — query on-chain owner()==Safe AND paused()==false (GATE-1 preconditions)
  verify    — query on-chain paused()==true (GATE-4 post-execution check)

Usage:

  # PREP — generate the tx parameters before Safe signing (no RPC needed)
  python3 scripts/safe-tx-pause-emissions.py prep

  # PREFLIGHT — confirm preconditions before signing
  export BASE_RPC_URL=https://base-rpc.publicnode.com   # mainnet.base.org 403s from some networks
  python3 scripts/safe-tx-pause-emissions.py preflight

  # VERIFY — confirm emissions paused post-Safe-execution
  python3 scripts/safe-tx-pause-emissions.py verify

Default env values:
  EMISSION_CONTROLLER_ADDRESS = 0x13A0D76895c196B795b94fe843F76B6e145AeaAE (canonical, on-chain-verified)
  SAFE_ADDRESS                = 0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791 (Foundation Safe — owner)
  BASE_RPC_URL                = https://mainnet.base.org
  CHAIN_ID                    = 8453

Exit codes:
  0 = prep ok / preflight preconditions met / verify shows emissions PAUSED
  1 = preflight precondition NOT met / verify shows emissions NOT paused
  2 = invalid input (malformed address)
  3 = RPC error
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from eth_utils import keccak, to_bytes, to_checksum_address
from web3 import Web3


# ── Selectors (proven from source, not hand-copied) ─────────────────────────

# EmissionController.pauseMinting() — owner-only, no args
PAUSE_MINTING_SELECTOR = "0x" + keccak(text="pauseMinting()").hex()[:8]
# EmissionController.paused() view → bool
PAUSED_SELECTOR = "0x" + keccak(text="paused()").hex()[:8]
# Ownable.owner() view → address
OWNER_SELECTOR = "0x" + keccak(text="owner()").hex()[:8]

DEFAULT_EMISSION = "0x13A0D76895c196B795b94fe843F76B6e145AeaAE"
DEFAULT_SAFE = "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791"
DEFAULT_RPC = "https://mainnet.base.org"
DEFAULT_CHAIN_ID = 8453


def _resolve_addresses() -> dict:
    emission = os.environ.get("EMISSION_CONTROLLER_ADDRESS", DEFAULT_EMISSION).strip()
    safe = os.environ.get("SAFE_ADDRESS", DEFAULT_SAFE).strip()
    for label, addr in [("EmissionController", emission), ("Safe", safe)]:
        if not Web3.is_address(addr):
            print(f"ERROR: {label} address {addr!r} is not a valid Ethereum address", file=sys.stderr)
            raise SystemExit(2)
    return {"emission": to_checksum_address(emission), "safe": to_checksum_address(safe)}


def _connect_rpc() -> Web3:
    rpc_url = os.environ.get("BASE_RPC_URL", DEFAULT_RPC)
    chain_id = int(os.environ.get("CHAIN_ID", str(DEFAULT_CHAIN_ID)))
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        print(f"ERROR: cannot reach RPC at {rpc_url}", file=sys.stderr)
        raise SystemExit(3)
    rpc_chain = w3.eth.chain_id
    if rpc_chain != chain_id:
        print(
            f"ERROR: RPC chainId={rpc_chain} != expected {chain_id}. "
            f"BASE_RPC_URL likely points at the wrong network.",
            file=sys.stderr,
        )
        raise SystemExit(3)
    return w3


def _read_address(w3: Web3, to_addr: str, selector: str) -> str:
    result = w3.eth.call({"to": to_addr, "data": selector})
    return to_checksum_address("0x" + result.hex()[-40:])


def _read_bool(w3: Web3, to_addr: str, selector: str) -> bool:
    result = w3.eth.call({"to": to_addr, "data": selector})
    return int.from_bytes(result, "big") != 0


def cmd_prep() -> int:
    addrs = _resolve_addresses()
    calldata = PAUSE_MINTING_SELECTOR  # no args → calldata is just the 4-byte selector

    print("=" * 70)
    print("Foundation Safe transaction — PAUSE FTNS emissions (pauseMinting)")
    print("=" * 70)
    print()
    print("Pre-launch posture: halt minting until the network has participants and")
    print("the reward pools are split. Owner-only, reversible via resumeMinting().")
    print()
    print("─── Selector (proven from source) ──────────────────────────────────")
    print(f"  pauseMinting() selector:  {PAUSE_MINTING_SELECTOR}")
    print()
    print("─── Resolved addresses ──────────────────────────────────────────────")
    print(f"  EmissionController (target): {addrs['emission']}")
    print(f"  Foundation Safe (sender):    {addrs['safe']}  (must be owner)")
    print()
    print("─── Safe Transaction Builder UI parameters ─────────────────────────")
    print("  Open https://app.safe.global → your Foundation Safe")
    print("  → New transaction → Transaction Builder → Add new transaction")
    print()
    print(f"  To address:           {addrs['emission']}")
    print(f"  ETH value:            0")
    print(f"  ABI fragment:         function pauseMinting()")
    print(f"  (no arguments to fill)")
    print()
    print("─── Or, raw calldata mode in Safe UI ────────────────────────────────")
    print(f"  To:                   {addrs['emission']}")
    print(f"  Value:                0")
    print(f"  Data:                 {calldata}   (exactly 10 chars incl. 0x)")
    print(f"  Operation:            CALL (NOT DelegateCall)")
    print()

    out_path = Path("/tmp") / f"safe-tx-pause-emissions-{addrs['emission'][:10]}.json"
    safe_tx_builder_json = {
        "version": "1.0",
        "chainId": str(DEFAULT_CHAIN_ID),
        "createdAt": int(__import__("time").time() * 1000),
        "meta": {
            "name": "Pause FTNS emissions (EmissionController.pauseMinting)",
            "description": (
                "Pre-launch: halt minting until participants exist + pools are "
                "split. Owner-only, reversible via resumeMinting(). See "
                "docs/2026-06-05-pause-emissions-ceremony.md."
            ),
            "txBuilderVersion": "1.16.5",
            "createdFromSafeAddress": addrs["safe"],
            "createdFromOwnerAddress": "",
            "checksum": "",
        },
        "transactions": [
            {
                "to": addrs["emission"],
                "value": "0",
                "data": calldata,
                "contractMethod": {"inputs": [], "name": "pauseMinting", "payable": False},
                "contractInputsValues": {},
            }
        ],
    }
    out_path.write_text(json.dumps(safe_tx_builder_json, indent=2))
    print(f"─── Safe Transaction Builder import JSON ────────────────────────────")
    print(f"  Written to: {out_path}")
    print(f"  In Safe UI → Transaction Builder → click '...' → Drag and drop")
    print(f"  this JSON file to import the tx.")
    print()
    print("─── Before signing ─────────────────────────────────────────────────")
    print("  python3 scripts/safe-tx-pause-emissions.py preflight")
    print("─── After Safe execution, verify ───────────────────────────────────")
    print("  python3 scripts/safe-tx-pause-emissions.py verify")
    print()
    print("=" * 70)
    return 0


def cmd_preflight() -> int:
    """GATE 1 — confirm owner()==Safe AND paused()==false before signing."""
    addrs = _resolve_addresses()
    w3 = _connect_rpc()
    print("Pre-flight checks on Base mainnet (must both pass)...")
    print(f"  EmissionController:  {addrs['emission']}")
    print(f"  Foundation Safe:     {addrs['safe']}")
    print()
    try:
        owner = _read_address(w3, addrs["emission"], OWNER_SELECTOR)
        paused = _read_bool(w3, addrs["emission"], PAUSED_SELECTOR)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: pre-flight call failed: {e}", file=sys.stderr)
        return 3

    ok_owner = owner.lower() == addrs["safe"].lower()
    ok_unpaused = not paused
    print(f"  [{'PASS' if ok_owner else 'FAIL'}] owner() == Foundation Safe   (owner is {owner})")
    print(f"  [{'PASS' if ok_unpaused else 'FAIL'}] paused() == false            (currently paused={paused})")
    print()
    if ok_owner and ok_unpaused:
        print("✅ Pre-conditions met — the Safe can pauseMinting() and it won't revert.")
        return 0
    if not ok_owner:
        print("❌ The Safe is NOT the owner — pauseMinting() would revert. Abort.")
    if not ok_unpaused:
        print("❌ Emissions are ALREADY paused — pauseMinting() would revert AlreadyPaused.")
        print("   (Nothing to do; emissions are already off.)")
    return 1


def cmd_verify() -> int:
    """GATE 4 — confirm paused()==true post-execution."""
    addrs = _resolve_addresses()
    w3 = _connect_rpc()
    print(f"Verifying emissions paused on Base mainnet...")
    print(f"  EmissionController:  {addrs['emission']}")
    print()
    try:
        paused = _read_bool(w3, addrs["emission"], PAUSED_SELECTOR)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: paused() call failed: {e}", file=sys.stderr)
        return 3
    if paused:
        print("✅ EmissionController.paused() == true — emissions are PAUSED.")
        print("   pullAndDistribute() now reverts MintingIsPaused; supply held at genesis.")
        print("   Reverse at launch with resumeMinting() (owner-only Safe tx).")
        return 0
    print("❌ EmissionController.paused() == false — emissions are NOT paused.")
    print("   The Safe tx may not be executed yet (check pending txs in the Safe UI),")
    print("   or RPC state is lagging — wait ~1-2 blocks and re-run.")
    return 1


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("prep", "preflight", "verify"):
        print(__doc__, file=sys.stderr)
        print("\nERROR: must specify 'prep', 'preflight', or 'verify' as first arg", file=sys.stderr)
        return 2
    if sys.argv[1] == "prep":
        return cmd_prep()
    if sys.argv[1] == "preflight":
        return cmd_preflight()
    return cmd_verify()


if __name__ == "__main__":
    sys.exit(main())
