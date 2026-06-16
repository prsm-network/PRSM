# Two-Party Base-Sepolia Settlement Run — Verified Runbook (2026-06-16)

Cross-party (requester-payment) proof of the on-chain settlement rail on **Base Sepolia
(chainId 84532)**: requester **A** signs an EIP-712 PaymentAuthorization and funds A's own
escrow; provider/settler **B** verifies it and commits a batch naming **A→B**; after the
challenge window, B finalizes, draining A's escrow into B's wallet via
`EscrowPool.settleFromRequester(A, B, amount)`.

> Real test FTNS moves and an **irreversible on-chain batch** is committed. The assistant
> prepares + verifies **read-only**; the **operator runs** every key-bearing command (they
> hold the keys). Verified end-to-end via a 5-agent workflow + an independent read-only
> on-chain corroboration on 2026-06-16.

## Verified on-chain config (corroborated twice, read-only)

| Item | Value |
|------|-------|
| chainId | `84532` (live `eth_chainId`) |
| FTNS token | `0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a` (symbol `FTNS`, 18 dec; 170-byte minimal-proxy, delegates correctly) |
| EscrowPool | `0xaa28b5818242608e04C1773c3e34bF7bFfb96248` (2810 bytes) |
| SettlementRegistry | `0xF8BEEb4362222b50109b6034767322B31aA92449` (9488 bytes) |
| **Challenge window** | **`challengeWindowSeconds() = 86400` = 24h** (NOT the 3-day docstring default; the deployed registry's live value + the per-batch on-chain snapshot are authoritative) |
| Candidate requester A | `0xCCAc7b21695De068979b1ca47B0cfBD328654220` — testnet deployer/treasury EOA, **100,002,060 FTNS** on chain |

## 1. Prerequisites — keys, funding, A ≠ B

| Role | Env var | Must hold | Why |
|------|---------|-----------|-----|
| **A = requester** | `REQUESTER_PRIVATE_KEY` | **test FTNS** (≥ `--amount-ftns`) **AND** Base-Sepolia **ETH** | A signs the EIP-712 auth and **self-deposits A's own FTNS into A's own escrow** (`EscrowPool.deposit` is a self-deposit — there is no `depositFor`). A pays gas for `approve` (if allowance short) + `deposit`. (`e2e_proof.py:124-141`, escrow client signed by `requester_key`=A at `settlement_sepolia_e2e.py:131-134`.) |
| **B = provider/settler** | `PRIVATE_KEY` | Base-Sepolia **ETH** only (NO FTNS) | B verifies A's auth, sends the commit tx (phase 1) and the finalize tx (phase 2). (`settlement_sepolia_e2e.py:135-137`.) |

- **FTNS goes on A, never B.** The script docstring line 27 ("B's deposit funds A's escrow") is **stale/wrong** — the code does A-deposits-into-A's-escrow. Do not fund B with FTNS.
- **A ≠ B is enforced** (equal → exit 2, `settlement_sepolia_e2e.py:121-124`).
- **Both A and B need Base-Sepolia ETH for gas** — there is **no pre-flight gas check** (a gasless signer just fails at broadcast: safe, no double-spend, but it fails).

## 2. Phase 1 — deposit + commit (operator runs; A and B keys)

```
cd ~/Documents/GitHub/PRSM && PRSM_NETWORK=testnet PRIVATE_KEY=0x<B> REQUESTER_PRIVATE_KEY=0x<A> python scripts/settlement_sepolia_e2e.py --phase deposit-commit --two-party --amount-ftns 1
```

Reads chainId from the live RPC and hard-refuses mainnet; A signs the auth (provider=B); B's verifier checks recovered-signer==A and provider==B (fail-closed, before any chain write); A approves+deposits the FTNS (idempotent — skips if already funded, polls past RPC replica lag, sp1047/1048); B commits one batch naming **A→B**. Refuses if a committed-but-unfinalized batch already exists in the state file (anti-double-spend, exit 1).

Expected (exit 0): prints `batch_id=<64-hex>`, `escrow_balance=1000000000000000000 wei`, and `PHASE 1 done. Batch finalizable in ~24.0h (at 86400s)`.

**CRITICAL: copy the printed `batch_id` and save it off-machine now.** State file defaults to `~/.prsm/settlement_e2e_state.json`.

## 3. Wait — the challenge window

**24 hours (86,400s).** Finalizing early reverts by construction (`isFinalizable` reads the per-batch on-chain snapshot). Testnet does not shorten it; only the owner can lower it, and only for batches committed *after* the change.

## 4. Phase 2 — finalize (operator runs; B key only)

```
cd ~/Documents/GitHub/PRSM && PRSM_NETWORK=testnet PRIVATE_KEY=0x<B> python scripts/settlement_sepolia_e2e.py --phase finalize
```

A's key is NOT needed (the committed batch already names A→B). Rehydrates the state file, runs reconcile/recover (adopt-not-recommit), finalizes; `finalizeBatch` → `EscrowPool.settleFromRequester(A, B, amount)`. Success is keyed on the on-chain **FINALIZED** status (not a lag-prone wallet delta).

**Recovery variant** (state file lost between phases — uses the saved batch_id, no state file needed):
```
cd ~/Documents/GitHub/PRSM && PRSM_NETWORK=testnet PRIVATE_KEY=0x<B> python scripts/settlement_sepolia_e2e.py --phase finalize --batch-id 0x<batch_id>
```

## 5. Read-only verification (assistant runs after each phase; NO keys)

- **After Phase 1:** confirm batch status `1` (PENDING), `finalizable False`, `secs_left ~86400`, A's escrow `≥ 1e18 wei`.
- **After Phase 2:** confirm batch status `2` (FINALIZED), A's escrow dropped by the settled amount, B's wallet FTNS rose by it.

(Via `Web3SettlementContractClient.get_batch_status/is_finalizable/seconds_until_finalizable` + `EscrowPoolClient.balance_of/ftns_balance_of`, read-only with `private_key=None`.)

## 6. Safety caveats

1. **Never run Phase 1 twice** — a second run double-deposits + mints a second batch (no content dedup; the state-file guard catches the common case but treat it as a real double-spend risk). Run Phase 1 exactly once.
2. **Save the printed `batch_id` off-machine** before the 24h gap — it's the recovery key if the state file is lost.
3. **Never `PRSM_NETWORK=mainnet`, never `--i-understand-mainnet`** — always set `PRSM_NETWORK=testnet` explicitly (the script refuses mainnet without the flag, but `networks.py` `DEFAULT_NETWORK="mainnet"` is a latent footgun for other callers).
4. **Keep the state file** between phases; run Phase 2 on the same machine when possible (a corrupt state file fails closed loudly, never silently empty).
5. **Fund both A and B with Base-Sepolia ETH** for gas.

## 7. Open decision (operator input)

- **Confirm A:** use candidate `0xCCAc7b21695De068979b1ca47B0cfBD328654220` (deployer/treasury, has FTNS) as requester A, or designate another FTNS-funded wallet.
- **Supply B:** a second, distinct Base-Sepolia-funded EOA (provider/settler) holding ETH for gas, no FTNS. A and B must differ. The private keys go into the command env when the operator runs it — never shared with the assistant.
