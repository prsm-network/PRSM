# Plan — Multi-Stage Settlement Mainnet Canary on Lambda GPU Nodes

The next attempt at the live 2-node mainnet multi-stage settlement canary, on **Lambda GPU
instances** with a **real model too big for one GPU**. This plan folds in everything learned on the
2026-07-03/04 GCP attempt (settlement validated GO; compute-topology fix landed + live-validated to
`gpu_count: 2`) plus the Lambda operational findings from the earlier big-model proofs.

## Why GPU + a big model (the core fix for what blocked the CPU run)

On the CPU VMs the settlement stack was GO and the static-file pool put both nodes in the head's
pool (`gpu_count: 2`), but the **router served single-node** because gpt2 fit one node — and
synthetic "too big" models landed in allocate-empty (embedding+lm_head+KV overhead dominates tiny
models). **A real model that physically cannot fit one GPU removes that ambiguity: the router has NO
single-node option, so it MUST build the 2-stage chain.** That's exactly how the earlier
Qwen-7B/14B/72B multi-host proofs got their splits (limited VRAM = unavoidable split).

**Model + hardware:** **Qwen2.5-14B-Instruct** (~28 GB fp16) on **2× A10 (24 GB each)**. 28 GB
doesn't fit one 24 GB card → forced 2-stage split, and the pieces fit two (14 GB/node). (A100-40GB
also works; then use a bigger model, e.g. Qwen-32B, to stay over one card.)

## Status going in (already done)

- **Settlement: validated GO** on mainnet — preflight passes on real nodes against the live registry
  `0x12a01F6C…391F2` / EscrowPool `0x4e93a04b…EEa1` / FTNS `0x5276…16e5`.
- **Wallets funded:** Settler A `0xBbEB…C9a0`, Settler B `0x2010…0e26` (ETH for gas), requester
  `0xF7d88c94…11c2` (holds 1.0 FTNS; no escrow deposit made yet). Each settler key derives its exact
  address (verified).
- **Compute-topology fix (sp1371):** static-file GPU pool provider, live-proven to `gpu_count: 2`.

---

## 1. Provision 2 Lambda GPU instances

- Two `gpu_1x_a10` (24 GB) instances (or `gpu_1x_a100`), same region.
- **Network — THE Lambda gotcha:** the Lambda PUBLIC firewall blocks inbound `:9001`. Bootstrap the
  P2P + settlement-task delivery over the **shared PRIVATE net (`10.19.0.0/16`)**, not the public IPs.
  Confirm each node's private IP (`ip -4 addr`) and use those in the endpoint map + peer connect.
- Repo + deps on each: `pip install -e '.[ml]' && pip install -e '.[blockchain]'` (the ml extra
  alone omits web3/eth-account — settlement needs blockchain too; learned on the GCP run).

## 2. Pre-stage the model (avoid the cold-fetch stall)

- **Serial shard cold-fetch mid-stage stalls the chain** (TransportTimeout). **Pre-download the model
  in PARALLEL on both nodes BEFORE launching the chain**, then run on WARM nodes. If a box hangs its
  HF download + drops: kill-stuck → resume-fetch → re-`/peers/connect`.
- Cold-load timeouts: `PRSM_PARALLAX_CHAIN_DEADLINE_S=600`, `PRSM_CHAIN_UNARY_EXECUTION_TIMEOUT_S=600`.
  14B cold-load uses the sp1227 meta-device skeleton (already in the code).

## 3. Static-file pool with REAL VRAM (recommended)

Use the sp1371 static-file pool — deterministic, no dependency on libp2p hardware_profile
propagation (which the default WebSocket transport lacks). With a real big model the `memory_gb` is
the **real GPU VRAM** (no synthetic tuning): 24 GB/node vs a 28 GB model → forced split.

`parallax_pool.json` (identical on both, node_ids from `GET /info` after start):
```json
{"gpus":[
  {"node_id":"<A_node_id>","region":"canary","layer_capacity":48,"memory_gb":24.0,"stake_amount":1000000000000000000,"device":"cuda"},
  {"node_id":"<B_node_id>","region":"canary","layer_capacity":48,"memory_gb":24.0,"stake_amount":1000000000000000000,"device":"cuda"}
]}
```
Same `region` on both (one allocation group). Set `PRSM_PARALLAX_GPU_POOL_KIND=static-file` +
`PRSM_PARALLAX_GPU_POOL_FILE=$HOME/parallax_pool.json`. (Alternative: the dht-backed pool via the
bootstrap flow, proven on Lambda for the 72B run — but static-file removes the discovery-propagation
variable.)

## 4. Trust: anchor the nodes OR mock for the canary

The PRODUCTION trust `filter_pool` drops an **unanchored** static-pool peer (learned on GCP — node B
got filtered, leaving 1 node → allocate-empty). Two options:
- **Fast (canary):** `PRSM_PARALLAX_TRUST_STACK_KIND=mock` (accept-all anchor). Proves the split +
  settlement; skips anchoring. Fine for a first live money-path run.
- **Production-fidelity:** keep production trust and properly anchor both nodes (publisher-key-anchor
  + real StakeBond stake), as the earlier dht-pool Lambda runs did. Heavier; do after the mock run.

Keep `PRSM_PARALLAX_STAKE_ELIGIBILITY=advisory` + `PRSM_PARALLAX_TIER_GATE=advisory` (from
`operator-parallax.env`) so eligibility/tier don't filter.

## 5. Per-node environment

Source `config/parallax/operator-parallax.env` (sets `PRSM_INFERENCE_EXECUTOR=parallax` +
executor/runner kinds), then override:
```bash
export PRSM_NETWORK=mainnet
export BASE_RPC_URL="https://base-mainnet.g.alchemy.com/v2/<KEY>"    # PAYG (getLogs cap)
export PRSM_MULTISTAGE_SETTLEMENT=1
export PRSM_ONCHAIN_SETTLEMENT=1
export PRSM_SETTLEMENT_POLL_INTERVAL_S=60
export PRSM_COMPUTE_WALLET_MAP_FILE=$HOME/wallet_map.json
export PRSM_PARALLAX_GPU_POOL_KIND=static-file
export PRSM_PARALLAX_GPU_POOL_FILE=$HOME/parallax_pool.json
export PRSM_PARALLAX_TRUST_STACK_KIND=mock         # or production + anchoring (§4)
export PRSM_PARALLAX_HF_MODEL_ID=Qwen/Qwen2.5-14B-Instruct
export PRSM_PARALLAX_HF_DEVICE=cuda
export PRSM_PARALLAX_SLICE_LOAD=1                   # each node loads only its layer slice
export PRSM_PARALLAX_CHAIN_DEADLINE_S=600
export PRSM_CHAIN_UNARY_EXECUTION_TIMEOUT_S=600
export PRSM_API_HOST=0.0.0.0                        # settlement /per-stage-task delivery between nodes
export PRSM_ALLOW_INSECURE_PUBLIC_BIND=1            # (/info stays loopback-gated; delivery isn't gated)
export PRSM_MULTISTAGE_ENDPOINT_SCHEME=http
export PRSM_MULTISTAGE_ENDPOINT_MAP='{"<A_node_id>":"http://<A_private_ip>:8000","<B_node_id>":"http://<B_private_ip>:8000"}'
# Node A: FTNS_WALLET_PRIVATE_KEY=<Settler A key> ; PRSM_OPERATOR_ADDRESS=0xBbEB…C9a0
# Node B: FTNS_WALLET_PRIVATE_KEY=<Settler B key> ; PRSM_OPERATOR_ADDRESS=0x2010…0e26
```
Keys live ONLY in each host's env (or a `chmod 600` file the node sources) — never in chat/argv.
`wallet_map.json`: `{"<A_node_id>":"0xBbEB…C9a0","<B_node_id>":"0x2010…0e26"}` (payee == settler
key's address — the #1 misconfig).

Ops notes from the GCP run: the node CLI has **no `--host` flag** (bind via `PRSM_API_HOST`); to
restart, **kill by port** (`pkill -f "prsm.cli node start"` self-matches the driving SSH session →
255), e.g. `kill $(ss -ltnp 'sport = :8000' | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+')`.

## 6. Bring up + verify BEFORE spending

1. Start both nodes; `GET /info` → capture the two `node_id`s → fill `parallax_pool.json` +
   `PRSM_MULTISTAGE_ENDPOINT_MAP` + `wallet_map.json`; restart.
2. `/peers/connect` node A → `<B_private_ip>:9001`.
3. **`GET /admin/parallax/pool/snapshot` → `gpu_count: 2`** (both nodes, region canary, device cuda).
4. `python -m prsm.settlement.go_live_preflight multistage` on each → **✅ GO**.
5. **`quote-multistage`** with a budget covering the cost (start high, e.g. `budget_ftns: 100`, then
   read the real cost) → expect **`multi_stage: true, settleable: true, stage_count: 2`**, payees ==
   `[Settler A, Settler B]`. This is the milestone the CPU run couldn't reach — a real 14B on 24 GB
   cards forces it. (Remember: too-low budget fails the budget gate and misleadingly reports "single
   node".)

## 7. The canary (real money — operator signs)

1. Requester `0xF7d88c94…11c2` deposits ~0.5 FTNS into escrow (covers the cost + margin).
2. Run `scripts/multistage_settlement_canary.py` (`REQUESTER_KEY` + `HEAD_URL` from env,
   `MODEL=Qwen/Qwen2.5-14B-Instruct`, `BUDGET_FTNS` ≥ the quoted cost): quote → sign the per-stage
   auth (bound to the quoted payee set) → paid inference.
3. Each node self-commits its share on Base mainnet (poll loop, `msg.sender == own settler`) →
   capture both batchIds + commit txs.
4. After the ~24 h challenge window, finalize → EscrowPool draws the requester's escrow.
5. **Verify read-only (§6 of the activation runbook):** conservation (shares sum to the total),
   self-commit (provider == own settler), escrow drawn by the total.

## 8. Division of labor

- **Assistant:** drive provisioning/setup over SSH, wire the pool + settlement config, run the
  preflight + `pool/snapshot` + quote, all read-only verification, help debug the chain/commit logs.
- **Operator (you):** provision Lambda + fund keys + deposit escrow; sign the mainnet commit +
  finalize broadcasts. The loop pauses at every irreversible signature.

**GO/NO-GO to open up:** quote `settleable + stage_count:2`; both nodes commit their own share
on-chain; both finalize with conservation; escrow drawn by the total. Then anchor the nodes (§4
production path) and raise the poll interval before real traffic.
