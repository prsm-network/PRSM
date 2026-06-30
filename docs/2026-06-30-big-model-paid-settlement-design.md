# Design Pass — Live Big-Model (Multi-Stage) Paid Settlement

**Status:** DESIGN (no code). Scopes the deferred, money-path-sensitive build that lets a
paying requester settle cross-host **multi-stage** (big-model sliced) inference on-chain,
paying **each stage node** from the requester's escrow. Single-stage paid settlement is
already live; multi-stage on-chain settlement is explicitly skipped today
(`accumulate_settled_inference_receipt` → `"skipped:multi-stage-deferred"`,
`prsm/settlement/client_wiring.py:273-281`).

**Why it matters:** cross-host sliced inference (7B/14B/72B proven) is PRSM's differentiator.
Today you can *run* it and *pay for single-node* inference, but you cannot *pay* for the
flagship multi-host path. This closes that gap.

---

## 1. What already exists (assets to wire, not invent)

The per-stage subsystem is ~80% built + unit-tested; almost nothing here is greenfield.

| Brick | Symbol | File | State |
|---|---|---|---|
| Off-chain split | `split_release_across_stages` | `economy/credit_policy.py:168` | LIVE (off-chain only) |
| Per-node splitter | `split_receipt_to_per_node_batched_receipts` | `settlement/per_stage_settlement_split.py:304` | tested, **unwired** |
| Payee-set builder | `build_per_stage_payee_set` | `settlement/per_stage_settlement_split.py:220` | tested |
| Auth verifier | `verify_per_stage_authorization` | `settlement/per_stage_payment_authorization.py:308` | tested |
| Auth builder (client) | `build_per_stage_payment_authorization` | `settlement/payment_client.py` | **DONE sp1312** |
| Per-node commit orchestration | `commit_per_node_share_batches` | `settlement/per_stage_commit.py:162` | tested, **unwired** |
| Accumulator (multi-provider) | `AccumulatorKey` keyed by provider | `settlement/accumulator.py:157` | LIVE (already N-provider capable) |
| Worker per-stage signatures | `per_stage_settlement_signatures` field | `compute/inference/parallax_executor.py:151` | plumbed |

Conservation is already guaranteed by the splitter (`sum(share_wei) == total_value_wei`),
and each per-node `BatchedReceipt` is **signed by that node's own key** (challenge-defensible).

---

## 2. The crux: `b.provider = msg.sender`

`BatchSettlementRegistry.commitBatch` sets `b.provider = msg.sender`
(`BatchSettlementRegistry.sol:501`); `finalizeBatch` settles
`EscrowPool.settleFromRequester(requester, b.provider, finalValue)` (line 675); a slash hits
`b.provider`. There is **no `commitBatchFor(provider, …)`**. So a single orchestrator
**cannot** commit N batches for N different providers — each batch's provider is whoever
sent the tx. This forces one of two designs.

### Design A — distributed self-commit (RECOMMENDED)

Each stage node commits **its own** per-node batch (so `msg.sender == that node == provider`).

```
requester ──quote──▶ orchestrator (returns planned payee set {node_eth: share})
requester ──sign per-stage auth (sp1312) over payee_set_hash + cap──▶
requester ──paid multi-stage request {prompt, per_stage_payment_authorization}──▶ orchestrator
orchestrator: serve cross-host  ▶ §7 receipt + per-stage worker sigs
orchestrator: split_receipt_to_per_node_batched_receipts ▶ N per-node BatchedReceipts (each node-signed)
orchestrator: ROUTE (per-node BatchedReceipt + the per-stage auth) ──▶ each stage node
each stage node: verify_per_stage_authorization(authorizes MY (payee, share)) ▶ accumulate MY receipt ▶ commit MY batch (msg.sender = me)
each stage node: after challenge window ▶ finalizeBatch ▶ settleFromRequester(requester, me, my_share)  [draws requester escrow]
```

- **No contract change.** Reuses the live accumulator + poll loop + the tested
  `commit_per_node_share_batches` **running on each node**.
- Trust-faithful: each node commits its own signed work, exactly like single-stage.
- **Cost:** every stage node must run settlement with a **funded settler key** (gas) and
  accept routed per-node receipts. Routing is new plumbing.

### Design B — contract `commitBatchFor(provider, providerSig, …)`

Add a registry function letting a designated committer submit a batch on behalf of a
provider, authenticated by the provider's signature (the per-node `BatchedReceipt` is
already node-signed). One orchestrator commits all N.

- **Requires a contract change → a fresh bundle + Foundation migration** (the F-activation
  ceremony all over again). Heavy.
- Centralizes commit (the orchestrator can censor/stall a node's settlement); complicates
  the slash/challenge model (who is challengeable when a third party committed?).
- Only worth it if per-node settler keys prove operationally infeasible.

### Recommendation

**Design A.** No contract change, preserves the trustless per-node-signed model, and reuses
the already-tested bricks. Accept the per-node-settler-key operational cost for v1; revisit
B only if running funded keys on every stage node proves impractical. (A relayer/sponsored
model — sp1311 — does **not** remove this: the on-chain commit tx is each node's own;
`msg.sender` must be the provider.)

---

## 3. Money-path safety invariants (non-negotiable, all already patterns in the codebase)

1. **Conservation:** `sum(per-stage shares) == total settled value`, exact (splitter guarantees).
2. **Idempotency:** per-node `local_escrow_id = f"{job_id}::stage::{node_id}"` (already in the
   splitter) so a re-route / retry can't double-commit.
3. **Fail-closed auth:** a node commits a paid share ONLY if
   `verify_per_stage_authorization` authorizes its exact `(payee, share_wei)` (sp1172 gate).
4. **Dry-run before broadcast:** reuse the existing dry-run path; never broadcast a commit
   that would revert.
5. **Default-OFF:** gate behind an env flag (e.g. `PRSM_MULTISTAGE_SETTLEMENT`) so the proven
   single-stage path is byte-for-byte unchanged until explicitly enabled.
6. **Off-chain/on-chain consistency:** use the SAME split helper both ledgers use (already
   the case) so the on-chain per-stage shares match the off-chain release.

---

## 4. Partial-failure semantics (v1 decision)

**Per-stage independence** (recommended for v1): each node settles its own stage; the
requester's escrow is drawn per-stage at each finalize. If a node never commits
(offline / no key), that stage is unpaid on-chain — and the requester is **not** drawn for
it (no finalize → no draw). On-chain is the authoritative ledger; the off-chain release is
best-effort (same posture as single-stage today). **Atomic all-or-nothing** (escrow-lock →
fan-out → settle-or-refund) is a stronger guarantee but needs an escrow-lock primitive +
coordination — **defer to v2**; document the v1 weakness (a requester can pay for a subset of
stages if some nodes don't settle).

---

## 5. Sprint sequence (dependency-ordered)

Each ships behind the default-off flag, fully tested, money-path-gated.

- **S1 — Requester quote / topology preview.** A read-only endpoint that returns the planned
  payee set `{node_eth: share}` for a multi-stage request, so the requester can sign the
  per-stage auth (sp1312) for the EXACT set the provider will serve. (Without this the
  requester can't form a valid `payee_set_hash`.) Risk: LOW (read-only).
- **S2 — Orchestrator split + route.** On a settled multi-stage receipt, run the splitter →
  N node-signed `BatchedReceipt`s, and route each (+ the per-stage auth) to its node over the
  P2P substrate (extend the settlement gossip/announce, `settlement_gossip.py`). Risk: MED
  (new routing; no on-chain effect yet — nodes just receive).
- **S3 — Per-node accumulate + commit (unwire the skip).** On each stage node, consume a
  routed per-node receipt: verify the auth gate, accumulate to the node's own client, let the
  poll loop commit + finalize (msg.sender = node). Replace the `"skipped:multi-stage-deferred"`
  guard with this routed path (still default-off). Risk: **HIGH** (money path; this is the
  core). Gates: conservation test, idempotency test, fail-closed auth test, dry-run.
- **S4 — Per-node settler-key provisioning + funding runbook.** Operator guidance: each stage
  node needs a funded settler key bound to its provider address (reuse the sp1301 go-live
  preflight, extended per-node). Risk: LOW (ops/docs).
- **S5 — SDK/CLI paid-multistage surface.** `PRSMClient.pay_and_infer_multistage` (quote →
  build per-stage auth → send) + a CLI flag. Risk: LOW (client glue).

**Prereqs that gate S2/S3:** worker `per_stage_settlement_signatures` must be POPULATED in
production runs (confirm the field is filled end-to-end, not just declared); the proven
big-model runs must be re-exercised with settlement on.

---

## 6. Open questions (resolve before S3)

1. **Quote authenticity:** must the quoted payee set be signed by the provider so the
   requester isn't tricked into authorizing wrong payees? (Likely yes — bind the quote.)
2. **Topology drift:** if the served topology differs from the quote (a node drops mid-serve
   → re-allocation), the `payee_set_hash` won't match → settlement fails the auth gate. Need a
   re-quote / re-auth path, or commit the topology at request time.
3. **Challenge window × N batches:** N independent challenge windows; a slash hits one node.
   Confirm the per-node challenge/slash semantics compose (they should — each batch is
   independent).
4. **Gas economics:** each stage node pays its own commit+finalize gas (~2 txs). At 72B/10
   nodes that's 20 txs per inference. Acceptable on Base (cheap), but note it.
5. **Escrow headroom:** the requester's escrow must cover the FULL `total_max_spend` before
   any stage finalizes (all draw from the same escrow). The auth's `total_max_spend_wei` caps
   it; the quote/preflight should check escrow ≥ that.

---

## 7. Validation plan

- **Local:** the existing per-stage benches (`test_sprint_1159_per_stage_local_bench`,
  `scripts/per_stage_sepolia_bench.py`) already exercise split → per-node commit; extend to
  the routed path.
- **Testnet (Base Sepolia):** a 2-node multi-stage paid run end-to-end (quote → auth →
  serve → split → route → 2 nodes self-commit + finalize → assert each node's wallet rose by
  its share and the requester's escrow fell by the total). This is the go/no-go before
  mainnet, mirroring the single-stage two-party Sepolia proof.
- **Mainnet:** gated behind the flag + a Foundation-aware rollout, only after the testnet
  proof + a conservation/partial-failure adversarial review.

---

## 8. Verdict

**Recommended: Design A (distributed self-commit), 5 sprints (S1–S5), default-off, money-path
gated.** No contract change. The cryptographic + accounting bricks already exist and are
tested; the build is **wiring + routing + per-node settlement activation**, not invention.
The single genuinely-hard sprint is **S3** (the money path) — it earns the full safety-gate
treatment (conservation/idempotency/fail-closed/dry-run tests + a testnet two-node proof
before mainnet). S1/S2/S4/S5 are low-to-medium risk. Do NOT collapse S3 into a quick change.
