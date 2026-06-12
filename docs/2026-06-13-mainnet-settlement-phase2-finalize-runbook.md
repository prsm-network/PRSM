# Mainnet Settlement — Phase-2 Finalize Runbook

**Batch:** `7b6490c9aa80ccecd916d5cc588a734b133cf04e3f96f5bed0c9b4680c1cd687`
**Network:** Base mainnet (chainId 8453)
**Runnable from:** **2026-06-13 18:05:05 UTC** (the 3-day challenge window closes)
**Who runs it:** the operator (you) — this sends a real mainnet transaction. The assistant
never signs/broadcasts mainnet txs; it only verifies read-only before and after.

---

## What this does

Phase 1 (committed 2026-06-10 18:05:05 UTC) committed a 1-FTNS batch to the on-chain
`BatchSettlementRegistry` with a 3-day challenge window. This Phase-2 step finalizes that
batch: it calls `finalizeBatch`, which runs `settleFromRequester` to move 1 FTNS out of the
`EscrowPool` into the provider's wallet. This is the **self-pay** proof (provider == requester
== the settler `0xF7d88c…`), so the FTNS returns to the same wallet and the net is zero — the
point is to prove the deposit→commit→**finalize** rail works end-to-end on mainnet.

---

## Pre-flight state (verified read-only 2026-06-12 ~21:17 UTC)

| Field | Value |
|---|---|
| Batch status | **1 = PENDING** (committed, not finalized, not voided) ✅ |
| `isFinalizable` | `False` — **~20.8h** remaining at check time |
| Provider / requester | `0xF7d88c943B048dAd2e5178E40DaaD545dB3311c2` (self-pay) |
| merkle_root | `1676c5de…` (matches the durable state file) ✅ |
| total value | 1.0 FTNS |
| `escrow_pool_at_commit` | `0x526D40C08524670846ab811C95691845374122aF` (wired) ✅ |
| EscrowPool.balanceOf(settler) | **1.0 FTNS** → drains to provider on finalize, then 0 |
| FTNS wallet (settler) | 1.0 FTNS → returns to 2.0 after self-pay finalize |
| settler ETH (gas) | ~0.0015 ETH — sufficient for one cheap Base finalize tx |

Contracts (Base mainnet): registry `0x48fFab641b9D638F312FFA776818756a326F995B`,
escrow `0x526D40C08524670846ab811C95691845374122aF`,
FTNS `0x5276a3756C85f2E9e46f6D34386167a209aa16e5`.

---

## Prerequisites

1. `PRIVATE_KEY` = the **settler** key for `0xF7d88c943B048dAd2e5178E40DaaD545dB3311c2`
   (the funded EOA that committed Phase 1). It pays ~0.0015 ETH gas + holds the escrow.
2. The repo checked out, deps installed (`web3`, `eth-account`).
3. Run **after 2026-06-13 18:05:05 UTC**. Running earlier prints
   `not finalizable yet (~Xh remaining)` and exits non-zero (no harm — gated on the real
   on-chain `isFinalizable`).

---

## The command (single line — paste-ready)

Replace `0x<SETTLER_KEY>` with the `0xF7d88c…` private key. The `--batch-id` form finalizes
straight from chain state (no dependency on the local state file surviving the multi-day gap).

```
BASE_RPC_URL=https://base-rpc.publicnode.com PRSM_NETWORK=mainnet PRIVATE_KEY=0x<SETTLER_KEY> python scripts/settlement_sepolia_e2e.py --phase finalize --i-understand-mainnet --batch-id 7b6490c9aa80ccecd916d5cc588a734b133cf04e3f96f5bed0c9b4680c1cd687 --state-file ~/.prsm/settlement_mainnet_state.json
```

`BASE_RPC_URL=https://base-rpc.publicnode.com` overrides the default `mainnet.base.org`
(which can 403 some clients; publicnode is confirmed working for both reads and
`eth_sendRawTransaction`). `--i-understand-mainnet` is the required mainnet guard — the script
refuses chainId 8453 without it.

---

## Expected output (success)

```
network=mainnet chainId=8453 rpc=https://base-rpc.publicnode.com
batch 7b6490c9aa80…: {'finalizable': True, 'success': True, 'status': 'FINALIZED', ...}
```

Exit code `0`. **Success is keyed on the on-chain batch status becoming FINALIZED (2)** — not
on an immediate wallet-balance delta, because Base public RPCs can serve a lagging replica
right after the tx lands (the harness polls past the lag, sp1045/sp1048; success was vindicated
on the Sepolia proof despite a stale delta read).

---

## Post-run verification (assistant runs these read-only afterward)

After you report the tx landed, the assistant confirms on-chain (read-only, no key needed):
- `get_batch(7b6490c9…).status == 2` (FINALIZED)
- `EscrowPool.balanceOf(0xF7d88c…) == 0` (escrow drained via `settleFromRequester`)
- `FTNS.balanceOf(0xF7d88c…)` back up by 1 FTNS (self-pay returns to the same wallet)

---

## Troubleshooting

- **`not finalizable yet (~Xh remaining)`** — the window hasn't closed; wait until
  2026-06-13 18:05:05 UTC and re-run. Harmless.
- **RPC 403 / connection error** — the `BASE_RPC_URL=…publicnode.com` override above already
  avoids `mainnet.base.org`; if publicnode is rate-limiting, swap in another Base RPC
  (e.g. `https://base.llamarpc.com`) via the same `BASE_RPC_URL=` prefix.
- **`insufficient funds for gas`** — top up the settler EOA with a little Base ETH and re-run
  (finalize is idempotent — a second run on an already-FINALIZED batch is a no-op success).
- **Already finalized** — re-running reports `status=FINALIZED, success=True` and exits 0
  (idempotent; `run_finalize_by_batch_id` short-circuits on a finalized batch).
- The assistant will NOT run this command (it never signs mainnet txs). Run it yourself; the
  assistant verifies the result read-only.
