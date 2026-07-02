#!/usr/bin/env python3
"""sp1364 — LIVE on-chain smoke of the deployed ContentAccessVerifier (Base Sepolia).

Exercises the REAL deployed contracts with REAL testnet FTNS — the one path no unit/e2e test covers
(they mock the chain). One key acts as both creator and payer, so only one funded address is needed:

  1. register a fresh content in the ProvenanceRegistryV2 the CAV reads (so getCreatorAndRate
     resolves the fee payee),
  2. approve + payForAccess on the deployed ContentAccessVerifier,
  3. confirm verifyPayment(payer, contentHash, fee) == true (the gate the key-serve endpoint uses),
  4. confirm the creator was credited (claimable == fee),
  5. claim() and confirm the pool drains.

Requires (in your OWN shell — the key never goes through chat):
  SMOKE_KEY             a funded Base Sepolia key (>= FEE FTNS + a little ETH for gas).
  BASE_SEPOLIA_RPC_URL  optional (default https://sepolia.base.org).
  FEE_FTNS              optional fee in FTNS (default 0.01).

Usage:
  SMOKE_KEY=0x… python3 scripts/smoke_content_access_verifier_sepolia.py
"""
from __future__ import annotations

import os
import secrets
import sys
from decimal import Decimal

RPC = (os.environ.get("BASE_SEPOLIA_RPC_URL") or "https://sepolia.base.org").strip()
KEY = (os.environ.get("SMOKE_KEY") or "").strip()
FEE = int(Decimal(os.environ.get("FEE_FTNS", "0.01")) * (Decimal(10) ** 18))

# Deployed 2026-07-02 (see prsm/deployments/contract_addresses.json base-sepolia).
CAV = "0x99264Bca75d63DB9b8B5C7C1e2ECBf78d133905a"
REGISTRY = "0xCBe377Ae09fdD5F63875Aa5313C65A3C8C073731"
FTNS = "0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a"
CHAIN_ID = 84532


def main() -> int:
    if not KEY:
        print("Set SMOKE_KEY to a funded Base Sepolia key (FTNS + ETH).", file=sys.stderr)
        return 2

    from eth_account import Account

    from prsm.economy.web3.content_access_verifier import ContentAccessVerifierClient
    from prsm.economy.web3.provenance_registry_v2 import ProvenanceRegistryV2Client

    addr = Account.from_key(KEY).address
    reg = ProvenanceRegistryV2Client(rpc_url=RPC, contract_address=REGISTRY, private_key=KEY)
    cav = ContentAccessVerifierClient(RPC, CAV, FTNS, private_key=KEY, expected_chain_id=CHAIN_ID)

    ch = secrets.token_bytes(32)     # fresh → avoids AlreadyRegistered / KeyAlreadyDeposited
    print(f"payer/creator: {addr}")
    print(f"CAV:           {CAV}")
    print(f"registry:      {REGISTRY}")
    print(f"content_hash:  0x{ch.hex()}")
    print(f"fee:           {FEE} wei ({Decimal(FEE) / (Decimal(10) ** 18)} FTNS)")

    print("\n[1/5] register content in the registry (creator = payer)…")
    # 9800 bps = the registry's MAX creator rate (10000 - 200 network fee). The CAV credits the
    # FULL fee to the creator regardless — it only uses the creator from getCreatorAndRate, not
    # the rate — so this value is cosmetic for the smoke; it just has to be a valid registry rate.
    reg.register_content_v2(ch, 9800, "smoke://cav")
    creator, rate = reg.contract.functions.getCreatorAndRate(ch).call()
    print(f"      getCreatorAndRate → creator={creator} rate={rate}bps")
    assert creator.lower() == addr.lower(), f"creator {creator} != payer {addr}"

    before = int(cav.token.functions.balanceOf(addr).call())
    print(f"\n[2/5] payForAccess (approve if needed + pay {FEE} wei)…  wallet FTNS={before}")
    cav.pay_for_access(ch, FEE)

    print("[3/5] verifyPayment(payer, contentHash, fee)…")
    ok = cav.verify_payment(addr, ch, FEE)
    print(f"      verifyPayment → {ok}")
    assert ok is True, "verifyPayment returned false after paying"

    print("[4/5] creator credited (claimable == fee)…")
    claimable = cav.claimable(addr)
    print(f"      claimable(creator) → {claimable}")
    assert claimable == FEE, f"claimable {claimable} != fee {FEE}"

    print("[5/5] claim()…")
    cav.claim()
    after_claimable = cav.claimable(addr)
    assert after_claimable == 0, f"claimable {after_claimable} != 0 after claim"
    print(f"      claimable(creator) after claim → {after_claimable}")

    print("\n✅ LIVE SMOKE PASSED — register → payForAccess → verifyPayment==true → creator "
          "credited → claim, all on-chain against the deployed ContentAccessVerifier.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
