# Operator guide — serve a big model as a multi-stage GPU node

How to run a GPU node (or a pinned cluster) that serves a **model too big for one card**, sharding it
across nodes and getting **trustlessly paid per-node** for the slice it runs — the productionized
version of the 2026-07-04 mainnet canary.

**What changed (sp1375–1378).** Proving the canary took a 6-layer manual archaeology. That's now
folded into the node:
- **sp1375** — `.[ml]` pins numpy<2 + transformers<5, so the install doesn't ship a real-model-breaking stack.
- **sp1376** — the node **auto-derives the model catalog** from the HF config (no hand-editing).
- **sp1377** — the node **auto-stages the model** into the local registry at start (no manual `stage_hf_model.py`).
- **sp1378** — `gen_cluster_config.py` **generates the static pool + anchor** for a cluster (no hand-authoring).

So a single-node operator sets a few env vars; a cluster operator runs one extra command. Keys stay in
each host's shell — never in argv or chat.

---

## 1. Install (one command, deps now correct)

```bash
pip install -e '.[ml]' && pip install -e '.[blockchain]'
```
On a GPU image with a preinstalled torch, use a `--system-site-packages` venv to inherit it (the pins
keep numpy/transformers compatible). If pip trips on a broken system `flatbuffers`
(`invalid-installed-package`), `sudo rm -rf /usr/lib/python3/dist-packages/flatbuffers*` and retry.

## 2. Single big-model GPU node

Set these and start — the node auto-catalogs + auto-stages the model:
```bash
export PRSM_INFERENCE_EXECUTOR=parallax
export PRSM_PARALLAX_HF_MODEL_ID=Qwen/Qwen2.5-14B-Instruct   # any HF causal-LM
export PRSM_PARALLAX_HF_DEVICE=cuda
export PRSM_PARALLAX_SLICE_LOAD=1
export PRSM_MODEL_REGISTRY_ROOT=$HOME/prsm-registry
export PRSM_PARALLAX_MODEL_CATALOG_FILE=$HOME/PRSM/config/parallax/model_catalog.json
# (source config/parallax/operator-parallax.env for the executor/runner kinds)
prsm node start
```
Pre-download the weights once (`huggingface-cli download <model>` or a `snapshot_download`) so the
first request doesn't cold-fetch mid-chain. A single node serves the model whole when it fits its
VRAM; to force a multi-stage split you need ≥2 nodes (below).

## 3. Multi-node cluster (a model too big for one card)

**a. Collect each node's identity.** Start each node, then grab its `node_id` (`GET /info`) and its
base64 pubkey (`~/.prsm/identity.json`). Write `nodes.json`:
```json
[
  {"node_id": "<A_node_id>", "pubkey": "<A_pubkey_b64>"},
  {"node_id": "<B_node_id>", "pubkey": "<B_pubkey_b64>"}
]
```

**b. Generate the pool + anchor** (sp1378) — `memory_gb` is the split lever: set it BELOW the model's
single-node footprint so it can't fit one card (Qwen-14B on 24 GB A10s → `40`; the allocator over-
estimates, so 30–60 forces a clean 24+24 split while the real per-node load ~16 GB fits):
```bash
python scripts/gen_cluster_config.py --nodes nodes.json --memory-gb 40 --region cluster --out-dir .
```
Drop `parallax_pool.json` + `anchor.json` on **every** node, and add to each node's env:
```bash
export PRSM_PARALLAX_GPU_POOL_KIND=static-file
export PRSM_PARALLAX_GPU_POOL_FILE=$HOME/parallax_pool.json
export PRSM_PUBLISHER_KEY_ANCHOR_KIND=static
export PRSM_PUBLISHER_KEY_ANCHOR_FILE=$HOME/anchor.json
export PRSM_PARALLAX_TRUST_STACK_KIND=mock   # or production + real StakeBond stake per node
export PRSM_API_HOST=0.0.0.0 PRSM_ALLOW_INSECURE_PUBLIC_BIND=1   # settlement /per-stage-task delivery
```

**c. Connect the peers.** On a shared private subnet, `POST /peers/connect {"address":"<peer_priv_ip>:9001"}`.
If the nodes can't reach each other on `:9001`/`:8000` (e.g. Lambda cross-region — all inter-node TCP
blocked but `:22` open), bridge with an SSH tunnel from the head:
`ssh -N -L 9101:localhost:9001 -L 8100:localhost:8000 <peer>`, then peer-connect to `127.0.0.1:9101`
and point the settlement endpoint map at `http://localhost:8100` for the worker.

**d. Verify before spending:** `GET /admin/parallax/pool/snapshot` → `gpu_count == N`; `quote-multistage`
→ `multi_stage: true, stage_count: N` (budget must cover the cost, else the budget gate reads as
"single node").

## 4. Settlement (get paid per node)

Add the settlement env + the wallet map (each node's payee == its own settler key's address), run the
preflight to GO, then the canary. This is unchanged from the activation runbook — see
`docs/2026-07-04-multistage-settlement-gpu-canary-plan.md` §5–7 for the full flow (settler keys,
wallet map, `PRSM_MULTISTAGE_SETTLEMENT=1`, `go_live_preflight multistage`, the escrow deposit,
`scripts/multistage_settlement_canary.py`, commit → finalize). sp1374 ensures only the trustless
per-stage escrow commit settles (no legacy double-pay).

## What an operator no longer does by hand

| Was manual on the canary | Now |
|---|---|
| downgrade numpy + transformers | pinned in `.[ml]` (sp1375) |
| add the model to the catalog | auto-derived from HF config (sp1376) |
| run `stage_hf_model.py` | auto-staged at node-start (sp1377) |
| hand-write pool + anchor per node | `gen_cluster_config.py` (sp1378) |

Still operator-supplied (by nature): the funded settler keys, the requester escrow, the networking
(private subnet or tunnel), and — for production trust — real on-chain stake per node.
