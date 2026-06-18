# Per-Stage Base-Sepolia Bench — Operator Runbook (2026-06-18)

The operator-run counterpart to the **sp1159 local-hardhat bench**
(`tests/integration/test_sprint_1159_per_stage_local_bench.py`), which already
validated bricks 1-4 of the on-chain per-stage payee arc against a **real EVM**.
This runbook drives the **same proven flow** against **Base Sepolia (chainId 84532)**
with the operator's own funded keys, plus finalizes the **pending two-party batch**.

> Real test FTNS moves and **irreversible on-chain batches** are committed. The
> assistant **prepares + verifies read-only**; the **operator runs** every
> key-bearing command (they hold the keys; no key is ever shared with the assistant,
> embedded in the harness, or logged). Grounded in a read-only on-chain
> corroboration on 2026-06-18.

## Verified on-chain config (read-only, 2026-06-18 — do NOT invent addresses)

| Item | Value |
|------|-------|
| chainId | `84532` (live `eth_chainId`; the bench resolves this and hard-refuses mainnet) |
| FTNS token | `0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a` |
| EscrowPool | `0xaa28b5818242608e04C1773c3e34bF7bFfb96248` |
| SettlementRegistry | `0xF8BEEb4362222b50109b6034767322B31aA92449` |
| Challenge window | `challengeWindowSeconds() = 86400` = **24h** (per-batch on-chain snapshot is authoritative) |

These are resolved automatically from `prsm.config.networks` (TESTNET) when
`PRSM_NETWORK=testnet`. Confirm them yourself with the read-only verify (GATE 1).

---

## Part A — Finalize the PENDING two-party batch (now finalizable)

A two-party batch committed earlier is **PENDING with the window elapsed**
(`isFinalizable = TRUE`):

| Field | Value |
|-------|-------|
| batch_id | `a184621c74d1b37eec72fced3f2c482c8abaeefd475bdf8b12b5843aab8f3300` |
| provider (B, recipient) | `0xF7d88c943B048dAd2e5178E40DaaD545dB3311c2` |
| requester (A, escrow source) | `0xCCAc7b21695De068979b1ca47B0cfBD328654220` |
| status | `1` PENDING, `isFinalizable = TRUE` |
| escrow | A's escrow holds 1 FTNS; B's wallet 0 |

`finalizeBatch` settles from the **recorded** requester(A) escrow to the
**recorded** provider(B) **regardless of caller** — so **any funded Sepolia key**
can finalize (B is the natural caller). Use the existing, proven tool:

**GATE 1 — pre-flight read-only verify (assistant, NO keys):** confirm
`status=1`, `isFinalizable=TRUE`, A's escrow `>= 1 FTNS`, B's wallet `0`.

**GATE 2 — operator finalizes (any funded Sepolia key in `PRIVATE_KEY`):**

```
cd ~/Documents/GitHub/PRSM && PRSM_NETWORK=testnet PRIVATE_KEY=0x<funded-key> python scripts/settlement_sepolia_e2e.py --phase finalize --batch-id 0xa184621c74d1b37eec72fced3f2c482c8abaeefd475bdf8b12b5843aab8f3300
```

The key needs Base-Sepolia **ETH for gas only** (it does not pay the FTNS — that
comes from A's recorded escrow). Success is keyed on the on-chain **FINALIZED**
status, not a lag-prone wallet delta.

**GATE 3 — read-only post-verify:** `status=2` (FINALIZED), A's escrow dropped by
1 FTNS, B's wallet `0xF7d8…11c2` rose by 1 FTNS.

---

## Part B — The per-stage bench (deposit → per-node commit → per-node finalize)

Pays **each topology node its conserving share on-chain**: the requester deposits
the TOTAL into its own escrow; each node commits + finalizes its OWN share-batch
(`msg.sender` ⇒ on-chain provider == that node). Reuses brick 1
(`per_stage_settlement_split`), brick 4 (`per_stage_commit`), and the settlement
clients exactly as sp1159 proved on a real EVM — nothing is reinvented.

### B.1 Keys to fund (all from the operator env; never embedded/shared)

| Role | Env var | Must hold | Why |
|------|---------|-----------|-----|
| **Requester A** | `PRSM_REQUESTER_KEY` (or `REQUESTER_PRIVATE_KEY`) | test **FTNS** (≥ `--amount-ftns`) **AND** Base-Sepolia **ETH** | A self-deposits the TOTAL into A's own escrow (`EscrowPool.deposit` is a self-deposit) + pays gas for approve/deposit. Candidate: `0xCCAc7b21695De068979b1ca47B0cfBD328654220` already holds test FTNS. |
| **Nodes (≥ 2)** | `PRSM_NODE_KEYS` (comma-sep) or `PRSM_NODE_KEY_0,_1,…` | Base-Sepolia **ETH** only (no FTNS) | Each node commits (phase 1) + finalizes (phase 2) ITS share-batch with its OWN key. The bench needs **≥ 2** node keys (below 2, brick 1 falls back to single-payee). |

- **A must differ from every node key** (a node paying itself is self-pay, not a
  per-stage proof) — the bench refuses it (exit 2).
- **Faucet:** Base-Sepolia ETH from a public faucet for each key's gas; test FTNS
  for A via #testnet-faucet on Discord (or reuse `0xCCAc7b21…`).
- **How many nodes:** the bench builds one share-batch per node; the sp1159 bench
  used **3**. The total is split conserving (equal split; the remainder lands +1
  wei on the first nodes), so e.g. 1 FTNS across 3 nodes pays 0.333… each, summing
  to exactly 1 FTNS.

### B.2 GATE 1 — pre-flight read-only verify (assistant, NO keys)

```
cd ~/Documents/GitHub/PRSM && PRSM_NETWORK=testnet python scripts/per_stage_sepolia_bench.py --phase verify
```

Confirms the script resolves the deployed addresses (registry / escrow / FTNS
above) on the live chain. With no phase-1 state yet it reports address resolution
only (safe). `--phase verify` is the **default** — the safe, keyless default.

### B.3 GATE 2a — PHASE 1: deposit + per-node commit (operator; A + node keys)

```
cd ~/Documents/GitHub/PRSM && PRSM_NETWORK=testnet PRSM_REQUESTER_KEY=0x<A> PRSM_NODE_KEYS=0x<n0>,0x<n1>,0x<n2> python scripts/per_stage_sepolia_bench.py --phase deposit-commit --amount-ftns 1
```

Reads chainId from the live RPC and hard-refuses mainnet. Echoes the requester +
each node **address** (never the key). Assembles the multi-stage receipt + each
node's per-stage leaf signature + the brick-1 split (offline, conserving); the
requester deposits the TOTAL (idempotent — skips if already funded, polls past RPC
replica lag); each node commits its share-batch. Refuses if phase-1 state already
exists (anti-double-spend, exit 1).

Expected (exit 0): prints each node's `batch_id` and saves them to the state file
(`~/.prsm/per_stage_sepolia_bench_state.json`). **Save the printed batch_ids
off-machine now.**

### B.4 Wait — the challenge window (24h)

Each batch snapshots the window at commit. Finalizing early reverts by
construction (`isFinalizable` reads the per-batch on-chain snapshot).

### B.5 GATE 2b — PHASE 2: per-node finalize (operator; node keys only)

```
cd ~/Documents/GitHub/PRSM && PRSM_NETWORK=testnet PRSM_NODE_KEYS=0x<n0>,0x<n1>,0x<n2> python scripts/per_stage_sepolia_bench.py --phase finalize
```

The **requester key is NOT needed** (the committed batches already name their
requester + node providers). Rehydrates the state file, finalizes each node's
batch straight from chain state (`run_finalize_by_batch_id` — idempotent, never
finalizes early). Each `finalizeBatch` ⇒ `EscrowPool.settleFromRequester(A, node, share)`.

### B.6 GATE 3 — read-only post-verify (assistant, NO keys)

```
cd ~/Documents/GitHub/PRSM && PRSM_NETWORK=testnet python scripts/per_stage_sepolia_bench.py --phase verify
```

Confirms, per node-batch: `status=2` (FINALIZED), each node's wallet rose by
**exactly** its conserving share, and the sum of shares == the deposited total
(no FTNS created/lost) — the same money-safety properties sp1159 asserted on a
real EVM, now on Sepolia.

---

## Safety caveats

1. **Testnet-default + mainnet hard-guard.** Always set `PRSM_NETWORK=testnet`.
   The bench refuses chainId 8453 without `--i-understand-mainnet` (sends real
   transactions). Never pass that flag here.
2. **Never run Phase 1 twice** — a second run double-deposits + mints second
   batches. The state-file guard catches the common case; treat it as a real
   double-spend risk and run Phase 1 exactly once.
3. **Save the printed batch_ids off-machine** before the 24h gap — they're the
   recovery key if the state file is lost. Keep the state file between phases.
4. **Fund A with FTNS + ETH; fund each node with ETH only.** No pre-flight gas
   check — a gasless signer just fails at broadcast (safe, no double-spend).
5. **All keys come from your env, never the assistant.** The harness echoes only
   addresses; it never logs a key, and contains no key-shaped literal in source.
