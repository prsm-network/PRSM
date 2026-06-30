# Runbook — 2-Node Testnet Go/No-Go for Big-Model (Multi-Stage) Paid Settlement

**Purpose.** Validate the LIVE multi-stage paid-settlement chain (sprints S1–S3b-3b, sp1313–1324)
end-to-end across **two separate hosts** on **Base Sepolia**, before any mainnet activation. This
is the design's mandated go/no-go gate (`docs/2026-06-30-big-model-paid-settlement-design.md` §7).

**What this proves that unit + in-process tests don't.** The settlement-side machinery is fully
unit-tested and the whole chain is proven in-process over real HTTP (`sp1323`,
`tests/integration/test_sprint_1323_per_stage_e2e.py`). The on-chain per-node commit/finalize is
proven on Base Sepolia by the `sp1160` bench (`docs/2026-06-18-per-stage-sepolia-bench-runbook.md`).
The ONLY thing not yet exercised together: a real cross-host **HTTP task delivery** →
**node-side fail-closed gate** → **each node's own funded `BatchSettlementClient` broadcasting**
its share on a live chain, across two untrusted hosts. That is what this runbook validates.

**Safety posture.** Everything is gated behind `PRSM_MULTISTAGE_SETTLEMENT` (default-off). This
runbook touches **testnet only** (`PRSM_NETWORK=testnet`). The settlement clients carry a hard
mainnet guard (they refuse chainId 8453 unless explicitly acknowledged). No mainnet, no
hardware-wallet ceremony. Operator holds all keys; nothing is embedded, logged, or shared.

---

## 0. Topology + roles

```
                 requester R (a wallet, funds escrow + signs the per-stage auth)
                        │  paid multi-stage inference request
                        ▼
   ┌──────────────────────── HOST 1 ────────────────────────┐   ┌──────── HOST 2 ────────┐
   │ Node A  (stage 0 / head / settler)                      │   │ Node B (stage 1 / worker)│
   │  - serves stage 0, settles the §7 receipt               │   │  - serves stage 1        │
   │  - ORCHESTRATOR: splits + ROUTES per-node tasks (S3a/2c) │   │  - receives its task     │
   │  - commits ITS OWN share (own funded settler key)       │   │  - commits ITS OWN share │
   └─────────────────────────────────────────────────────────┘   └──────────────────────────┘
```

- A 14B (or 7B) model sharded across 2 hosts → `stage_count == 2`, two distinct stage node_ids.
- Each node commits **its own** per-node batch (`msg.sender == that node == provider`, Design A).
- The requester's escrow is drawn per-stage at each node's `finalizeBatch` (per-stage independence).

You need **three Base-Sepolia-funded EOAs** (each with a little testnet ETH for gas + the
requester pre-funded with testnet FTNS for the escrow deposit):
- `R` — the paying requester.
- `A_settler` — Node A's settler key (its operator/provider address).
- `B_settler` — Node B's settler key (its operator/provider address).

---

## 1. Prerequisites

- Two hosts that can reach each other over the network (a private subnet is fine; note each node's
  reachable base URL, e.g. `http://10.0.0.1:8000` and `http://10.0.0.2:8000`). The settlement-task
  delivery is a normal HTTP POST to `/settlement/per-stage-task`.
- Base Sepolia RPC (`BASE_SEPOLIA_RPC_URL`) reachable from both hosts. A PAYG RPC (Alchemy/etc.)
  avoids the free-tier `eth_getLogs` cap.
- The PRSM repo checked out on both hosts; `pip install -e '.[ml]'`.
- Testnet settlement addresses resolve automatically from `PRSM_NETWORK=testnet`
  (`prsm.config.networks.resolve_endpoints()`); confirm with
  `PRSM_NETWORK=testnet python -c "from prsm.config.networks import resolve_endpoints as r; e=r(); print(e.rpc_url, e.settlement_registry)"`.
- The requester `R` has deposited (or will deposit) testnet FTNS into its escrow on the testnet
  EscrowPool. (`prsm wallet deposit` / the EscrowPool client, exactly as in the single-stage path.)

---

## 2. The node_id → payee wallet map (THE #1 misconfiguration risk)

Each node self-commits with `msg.sender == its settler key`. The S3a splitter stamps each per-node
`BatchedReceipt.provider_address` to the **payee the wallet map resolves for that node_id**, and
the node-side gate verifies that payee is in the requester's signed set. So:

> **Each node's payee in the wallet map MUST equal that node's `FTNS_WALLET_PRIVATE_KEY` address.**
> If they differ, the node's gate fail-closes (set-hash/membership mismatch) and that stage never
> settles. This is the most common setup error — check it first.

1. Get each node's `node_id` (after the nodes start, from `GET /info` or the startup log — it's the
   32-hex NodeIdentity id).
2. Write a wallet map JSON, deployed to BOTH hosts (identical content), e.g. `wallet_map.json`:
   ```json
   {
     "<NODE_A_node_id>": "<A_settler address>",
     "<NODE_B_node_id>": "<B_settler address>"
   }
   ```
3. Point `PRSM_COMPUTE_WALLET_MAP_FILE` at it on both hosts. (`ComputeWalletMap.from_env()` reads
   this; it is what `deliver_for_settled_receipt` + the splitter use.)

---

## 3. Per-node environment

**Common to both nodes (and the orchestrator/Node A):**

```bash
export PRSM_NETWORK=testnet
export BASE_SEPOLIA_RPC_URL="https://base-sepolia.g.alchemy.com/v2/<KEY>"
export PRSM_MULTISTAGE_SETTLEMENT=1          # the S3b gate (default-off elsewhere)
export PRSM_ONCHAIN_SETTLEMENT=1             # build a real (funded) BatchSettlementClient
export PRSM_COMPUTE_WALLET_MAP_FILE=/path/to/wallet_map.json
# delivery endpoint resolution (orchestrator side). Static map wins; else the WS transport peer.
export PRSM_MULTISTAGE_ENDPOINT_SCHEME=http  # or https if the nodes serve TLS
export PRSM_MULTISTAGE_ENDPOINT_MAP='{"<NODE_A_node_id>":"http://10.0.0.1:8000","<NODE_B_node_id>":"http://10.0.0.2:8000"}'
# optional: where each node persists its staged tasks (default ~/.prsm/per_stage_receiver.json)
export PRSM_MULTISTAGE_RECEIVER_STORE_FILE=$HOME/.prsm/per_stage_receiver.json
```

**Node A (host 1):**
```bash
export FTNS_WALLET_PRIVATE_KEY=0x<A_settler key>     # controls Node A's operator/provider address
export PRSM_OPERATOR_ADDRESS=0x<A_settler address>   # = FTNS_WALLET_PRIVATE_KEY's address
```

**Node B (host 2):**
```bash
export FTNS_WALLET_PRIVATE_KEY=0x<B_settler key>
export PRSM_OPERATOR_ADDRESS=0x<B_settler address>
```

> The settlement client refuses to start if `FTNS_WALLET_PRIVATE_KEY`'s address does not equal
> `provider_address` (it would otherwise settle to the wrong party) — so a mismatch here fails
> loudly at node start, before any money moves.

Keys live ONLY in each host's env. Never embed, log, or pass them through the chat/command history.

---

## 4. Bring the nodes up + connect them

1. Start Node A and Node B (`prsm node start` on each host). Confirm each is serving:
   `curl http://<host>:8000/readyz` (model loaded) and `curl http://<host>:8000/info` (note the
   `node_id` + `operator_address`).
2. Connect the peers so the head can route the cross-host inference chain:
   `curl -X POST http://10.0.0.1:8000/peers/connect -d '{"address":"10.0.0.2:8000", ...}'`
   (use the same bootstrap you use for any 2-node big-model run — see the multihost proof recipe).
3. Confirm `PRSM_MULTISTAGE_SETTLEMENT` is live on each node:
   `curl http://<host>:8000/status` should show on-chain settlement enabled; the node settlement
   poll loop now also drives `run_per_stage_commit_cycle` each iteration (sp1322).

---

## 5. The paid multi-stage flow (the actual test)

### 5.1 Quote — preview the payee set (sp1313)
```bash
curl -s -X POST http://10.0.0.1:8000/compute/inference/quote-multistage \
  -H 'content-type: application/json' \
  -d '{"model_id":"Qwen/Qwen2.5-14B-Instruct","prompt":"The capital of France is","max_tokens":8,"budget_ftns":1.0}'
```
Expect `{"multi_stage": true, "settleable": true, "stage_count": 2, "payees": [[A_settler, share],
[B_settler, share]], "payee_set_hash": "0x…", "total_value_wei": 1000000000000000000}`.

- **GO check:** `settleable == true` and the two `payees` are exactly `A_settler` + `B_settler`
  (i.e. the wallet map resolved correctly). If `settleable == false`, fix §2.

### 5.2 Requester signs the per-stage authorization (sp1312)
The requester `R` signs over the **quoted** payee set (so the served payees match the signed set):
```python
from prsm.settlement.payment_client import build_per_stage_payment_authorization
from decimal import Decimal
payees_ftns = [(addr, Decimal(share)/Decimal(10**18)) for addr, share in quote["payees"]]
auth = build_per_stage_payment_authorization(
    requester_key="0x<R key>", payees=payees_ftns,
    model_id="Qwen/Qwen2.5-14B-Instruct", prompt="The capital of France is",
    max_tokens=8, privacy_tier="none", content_tier="A", expiry_unix=<now + 1h>)
assert auth["payload"]["payee_set_hash"] == quote["payee_set_hash"]   # binds to the quote
```

### 5.3 Paid multi-stage inference (sp1324 ingress + serve + S3a/2c delivery)
POST the inference to Node A (the head) with the per-stage auth in the body:
```bash
curl -s -X POST http://10.0.0.1:8000/compute/inference \
  -H 'content-type: application/json' \
  -d '{"model_id":"Qwen/Qwen2.5-14B-Instruct","prompt":"The capital of France is","max_tokens":8,
       "per_stage_payment_authorization": <auth from 5.2>}'
```
What happens (watch the Node A + Node B logs):
1. Node A serves the cross-host chain → a signed §7 receipt carrying per-stage signatures (S2).
2. Ingress authenticated `R` from the per-stage auth (sp1324; 402 if signer ≠ requester).
3. Post-settle hook splits the receipt → 2 per-node tasks (S3a) and **POSTs each** to its node's
   `/settlement/per-stage-task` (sp1321/2c). Node A log: `sp1324 multi-stage settlement delivery
   (job=…): 2 task(s), 2 accepted`.
4. Each node's `/settlement/per-stage-task` runs the fail-closed gate (sp1316) and **stages** the
   task (sp1317). Node B log: an `accepted` delivery; its receiver store now holds 1 staged task.

- **GO check:** both nodes report `accepted` for their own task (not `misrouted` / not
  `auth re-check failed`). A reject here means the served payee set drifted from the signed set
  (topology changed mid-serve) → re-quote + re-sign (§5.1–5.2).

### 5.4 Per-node on-chain commit (sp1320/3a, driven by the poll loop sp1322)
Within one settlement poll interval (`PRSM_SETTLEMENT_POLL_INTERVAL_S`, default 600s — lower it for
the test, e.g. `=30`), each node's `run_per_stage_commit_cycle` drains its staged task and commits
its share-batch on-chain with its OWN funded key. Node logs: `per-stage commit cycle: committed
1/1`; the receiver store drains.

- **GO check:** each node committed exactly its own batch on Base Sepolia (`msg.sender ==
  that node's settler`). Capture both commit tx hashes + batchIds from the logs.

### 5.5 Finalize after the challenge window (~24h on the deployed registry)
After the challenge window elapses, each node's poll loop drives `finalize_ready_batches` → each
`finalizeBatch` settles `EscrowPool.settleFromRequester(R, node, share)` — drawing `R`'s escrow,
paying each node its share. (Same mechanics the `sp1160` bench + the single-stage path use.)

---

## 6. Verification (read-only)

- **Conservation:** the two committed share values sum to the quoted `total_value_wei` (1 FTNS).
- **Self-commit:** each batch's on-chain `provider == that node's settler address`
  (`msg.sender`), NOT the requester, NOT the other node.
- **Escrow:** after both finalize, `R`'s escrow fell by ~1 FTNS; `A_settler` + `B_settler` balances
  each rose by their share (minus gas).
- **Independence:** if you deliberately stop Node B before its commit, Node A still commits +
  finalizes its own share, and `R` is drawn ONLY for Node A's stage (the documented v1 per-stage
  independence — a missing node leaves its stage unpaid, never an over-draw).
- Read batch state with the existing read-only tooling (the `sp1160` bench `--phase verify`, or the
  `BatchSettlementClient` status), pointed at the testnet registry.

---

## 7. GO / NO-GO

**GO** (proceed to plan mainnet activation) iff ALL hold:
1. Quote `settleable == true`, payees == the two settler addresses.
2. Both nodes `accepted` + staged their own routed task.
3. Both nodes committed their own share on Base Sepolia (`provider == own settler`).
4. Both finalized; conservation holds (shares sum to total); `R`'s escrow drawn by the total.
5. The independence check behaves (a downed node → its stage unpaid, no over-draw).

**NO-GO** triggers + first thing to check:
- Quote `settleable == false` → wallet map (§2): a node_id has no payee.
- Delivery `accepted == false` / `auth re-check failed` → served payees ≠ signed set (topology
  drift) → re-quote/re-sign; or the wallet-map payee ≠ the node's settler key (§2).
- Node won't start with settlement on → `FTNS_WALLET_PRIVATE_KEY` address ≠ `PRSM_OPERATOR_ADDRESS`
  (§3 funds-safety guard).
- Commit cycle stays `committed 0/N` → the client is view-only (no/!matching key) or the RPC is
  unreachable.

---

## 8. After a GO — mainnet activation (USER-GATED, irreversible — out of scope here)

Do NOT do these autonomously. When the testnet go/no-go passes, mainnet activation is a deliberate
operator ceremony:
1. Provision each production stage node with a funded **mainnet** settler key controlling its
   operator address; set the production `PRSM_COMPUTE_WALLET_MAP_FILE`.
2. Set `PRSM_MULTISTAGE_SETTLEMENT=1` + `PRSM_ONCHAIN_SETTLEMENT=1` on the production nodes,
   `PRSM_NETWORK=mainnet`, and the production `PRSM_MULTISTAGE_ENDPOINT_MAP`.
3. Run a single low-value canary multi-stage paid inference, verify per §6 on mainnet, THEN open up.
   The mainnet commit/finalize broadcasts are irreversible — they are the operator's to sign.

---

## Appendix — sprint → step map

| Step | Implements | Sprint |
|---|---|---|
| Quote / payee-set preview | `POST /compute/inference/quote-multistage` | S1 (sp1313) |
| Per-stage receipt signatures | carried on the §7 receipt | S2 (sp1314) |
| Split settled receipt → tasks | `build_per_stage_settlement_tasks` | S3a (sp1315) |
| Node-side fail-closed gate | `verify_routed_settlement_task` | S3b-1 (sp1316) |
| Receiver store (idempotent stage) | `PerStageReceiverStore` / `ingest_routed_task` | S3b-2 (sp1317) |
| Delivery endpoint | `POST /settlement/per-stage-task` | S3b-2b (sp1318) |
| Orchestrator sender | `deliver_settled_multistage_tasks` | S3b-2c (sp1319) |
| Node-side commit driver | `commit_staged_task` / `drain_and_commit_staged` | S3b-3a (sp1320) |
| Endpoint resolver + entrypoint | `build_per_stage_endpoint_resolver` / `deliver_for_settled_receipt` | S3b-3b (sp1321) |
| Poll-loop commit cycle | `run_per_stage_commit_cycle` | S3b-3b (sp1322) |
| In-process e2e proof | `tests/integration/test_sprint_1323_per_stage_e2e.py` | (sp1323) |
| Paid request ingress + hook | `_resolve_paid_requester_or_402` + post-settle hook | S3b ingress (sp1324) |
| On-chain commit/finalize on Sepolia (per-node) | `scripts/per_stage_sepolia_bench.py` | (sp1160) |
