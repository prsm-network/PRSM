# Runbook — Mainnet Activation for Big-Model Multi-Stage Paid Settlement

Turns on trustless per-node on-chain settlement for big-model (multi-host) paid inference on **Base
mainnet**. This is the deliberate, user-gated ceremony after the testnet go/no-go passed.

**Status going in.** The 2-node testnet go/no-go is a GO — both per-stage share-batches committed +
finalized on Base Sepolia, conservation verified on-chain (0.14 + 0.14 = the requester's 0.28). The
offline full-cycle conservation smoke (sp1369, N=2/3/8) is green, and the preflight
(`python -m prsm.settlement.go_live_preflight multistage`) resolves the live mainnet registry. So
the machinery is proven; this activates it on mainnet with a low-value canary FIRST.

**Irreversibility + division of labor.** Every mainnet commit/finalize broadcast moves real FTNS and
is **irreversible** — those are the OPERATOR's to sign. The assistant preps + verifies read-only;
the operator signs. **Keys live ONLY in each host's env — never in chat/command history** (a per-node
settler key + the requester key are involved).

**Live mainnet addresses** (chain 8453, auto-resolved from `PRSM_NETWORK=mainnet`):
- Settlement registry `0x12a01F6C487d765af389bC7D95D90b3136a391F2`
- EscrowPool `0x4e93a04b3A0C5063FE584980e6c2B1429495EEa1`
- FTNS `0x5276a3756C85f2E9e46f6D34386167a209aa16e5`

---

## 0. Preconditions (all already true)

- Testnet 2-node go/no-go: **GO** (committed + finalized, conservation held).
- `PRSM_MULTISTAGE_SETTLEMENT` machinery is default-OFF everywhere until this activation.
- Offline conservation smoke green; preflight targets the mainnet registry.

## 1. Provision (the money + identity setup)

- **Two production stage nodes** (hosts A + B) that reach each other, repo + `pip install -e '.[ml]'`,
  a Base **mainnet** RPC (`BASE_RPC_URL`, PAYG to avoid the public `eth_getLogs` cap).
- **Per-node mainnet settler key**, each controlling its operator address, each funded with a little
  **mainnet ETH** (gas) — and its address is what the wallet map pays.
- **The requester `R`** with **mainnet FTNS deposited into escrow** on the live EscrowPool (enough
  for the canary + a margin; `prsm wallet deposit`).
- Fund all with only a small amount for the canary; open up after it verifies.

## 2. The wallet map (THE #1 misconfig — same as testnet)

> Each node's payee in the wallet map MUST equal that node's settler key's address. A mismatch
> fail-closes the node's gate and that stage never settles.

`wallet_map.json` (identical on both hosts), pointed at by `PRSM_COMPUTE_WALLET_MAP_FILE`:
```json
{ "<NODE_A_node_id>": "<A_settler address>", "<NODE_B_node_id>": "<B_settler address>" }
```

## 3. Per-node mainnet environment

```bash
export PRSM_NETWORK=mainnet
export BASE_RPC_URL="https://base-mainnet.g.alchemy.com/v2/<KEY>"
export PRSM_MULTISTAGE_SETTLEMENT=1
export PRSM_ONCHAIN_SETTLEMENT=1
export PRSM_COMPUTE_WALLET_MAP_FILE=/path/to/wallet_map.json
export PRSM_MULTISTAGE_ENDPOINT_SCHEME=https   # or http on a trusted private subnet
export PRSM_MULTISTAGE_ENDPOINT_MAP='{"<A_node_id>":"https://nodeA:8000","<B_node_id>":"https://nodeB:8000"}'
export PRSM_SETTLEMENT_POLL_INTERVAL_S=60      # lower than default 600 for the canary; raise after
# Node A: FTNS_WALLET_PRIVATE_KEY=0x<A_settler> ; PRSM_OPERATOR_ADDRESS=0x<A_settler addr>
# Node B: FTNS_WALLET_PRIVATE_KEY=0x<B_settler> ; PRSM_OPERATOR_ADDRESS=0x<B_settler addr>
```

> The settlement client refuses to start if `FTNS_WALLET_PRIVATE_KEY`'s address ≠
> `PRSM_OPERATOR_ADDRESS` — a mismatch fails loudly at start, before any money moves.

## 4. Preflight gate — MUST be GO on each node before the canary

```bash
python -m prsm.settlement.go_live_preflight multistage      # read-only/offline
```
Every FAIL must be cleared: `settler_key` (funded key present), `provider_address`,
`onchain_settlement`, `multistage_settlement`, `compute_wallet_map`, `client_build`. It must resolve
the mainnet registry `0x12a01F6C…391F2` (active, not paused) and report `client_build: PASS`.

## 5. The canary — ONE low-value multi-stage paid inference

Keep the value tiny (e.g. **0.02 FTNS** total). Follow the testnet flow (that runbook §5), on mainnet:

1. **Quote** `POST /compute/inference/quote-multistage` → confirm `settleable == true` and `payees`
   are exactly `A_settler` + `B_settler` (wallet map resolved). If false → fix §2.
2. **Requester signs** the per-stage authorization over the QUOTED payee set
   (`build_per_stage_payment_authorization`); assert `payload.payee_set_hash == quote.payee_set_hash`.
3. **Paid inference** to Node A (head) with the auth in the body → each node's
   `/settlement/per-stage-task` reports `accepted` + stages its own task (not `misrouted` /
   `auth re-check failed`).
4. **Per-node commit** (poll loop, ≤ `PRSM_SETTLEMENT_POLL_INTERVAL_S`): each node commits its share
   on Base mainnet with its OWN key (`msg.sender == own settler`). Capture both commit tx hashes +
   batchIds.
5. **Finalize** after the challenge window (~24h): each `finalizeBatch` draws `R`'s escrow, paying
   each node its share.

## 6. Verification (read-only — the assistant runs this)

- **Conservation:** the two committed shares sum to the quoted total (≈ 0.02 FTNS).
- **Self-commit:** each batch's on-chain `provider == that node's settler`, not R, not the other node.
- **Escrow:** after both finalize, `R`'s escrow fell by ≈ the total; each settler balance rose by its
  share (minus gas).
- **Independence:** (optional, on a second canary) stop Node B before its commit → Node A still
  commits + finalizes its own share; R is drawn ONLY for Node A's stage (never an over-draw).

## 7. GO / NO-GO to open up

**GO** iff all hold: quote settleable + correct payees; both accepted + staged; both committed
(provider == own settler on mainnet); both finalized with conservation; escrow drawn by the total.
Then raise `PRSM_SETTLEMENT_POLL_INTERVAL_S` back up and open big-model paid inference to real
traffic.

**NO-GO** first checks (same as testnet): `settleable == false` → wallet map; `accepted == false` /
`auth re-check failed` → served payees ≠ signed set (re-quote/re-sign) or wallet-map payee ≠ settler
key; node won't start → key ≠ operator address; `committed 0/N` → view-only client / RPC unreachable.

## 8. Rollback (fully reversible, non-destructive)

Set `PRSM_MULTISTAGE_SETTLEMENT=0` (and restart) on the production nodes → the per-stage delivery
hook returns `[]`, no stage node self-commits, and paid multi-stage inference reverts to the
single-payee / local-accumulation path. Nothing on-chain is undone (finalized canary batches stand);
you simply stop originating new per-stage commits. No migration, no contract change — it's an env flag.

---

## Assistant vs operator

- **Assistant (me):** this runbook; the preflight run; all read-only verification (§6); help debug
  the quote / delivery / commit logs live.
- **Operator (you):** provision hosts + keys + escrow; set env (keys stay in your shell); sign the
  mainnet commit + finalize broadcasts. The autonomous loop pauses at every irreversible signature.

---

## Appendix — COMPUTE-LAYER PREREQUISITE (found 2026-07-03 during the live canary)

The settlement stack activated cleanly on mainnet (both nodes preflight ✅ GO, funded, peered,
wallet map resolving). The canary was blocked NOT by settlement but by the parallax **compute**
layer failing to produce a 2-stage split on the two nodes. Root cause, diagnosed via
`GET /admin/parallax/pool/snapshot`:

1. **Pool membership.** `gpu_count: 1` — the head's parallax pool contained only itself. A
   `/peers/connect` transport link does NOT put the peer in the head's GPU pool; the pool is
   `dht-backed` and reads peer `hardware_profile`s, which only propagate over **libp2p discovery**
   (`libp2p_discovery.py:452`). The default WebSocket transport doesn't propagate them, and the only
   pool kinds are `dht-backed` / `static-empty` (no static-file pool to list nodes explicitly).
2. **Advertised capacity.** The head advertised a synthetic default profile
   (`layer_capacity: 16, memory_gb: 80, device: cuda, region: local-region`). gpt2's 12 layers fit
   one node → no reason to split. `PRSM_PARALLAX_LAYER_CAPACITY_OVERRIDE` / `MEMORY_GB_OVERRIDE` /
   `PRSM_PARALLAX_REGION` did NOT change the advertised profile (confirmed ignored by the tiler).

**To run the live 2-stage canary, the compute layer must first be made to shard across the two
nodes**, independent of settlement. Options for a future attempt: (a) run the nodes on **libp2p**
transport so hardware_profiles propagate → the peer enters the pool; (b) supply a custom
`PRSM_HARDWARE_PROFILE_FILE` with a low `layer_capacity` (< model layers) on each node so the model
can't fit one; (c) use GPU nodes with real limited VRAM (how the earlier multi-host proofs got their
splits naturally). This is a parallax-scheduler task, orthogonal to the (already-proven) settlement
loop. The 2-node **testnet** full loop (commit + finalize + conservation) already passed, so the
settlement design is validated; this is purely the mainnet compute-topology plumbing.
