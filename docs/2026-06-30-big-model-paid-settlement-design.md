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

- **S1 — Requester quote / topology preview. ✅ DONE (sp1313, 500dc34a).** Risk was MED
  (not LOW): a correct quote must come from the SERVE path or it can't match (the verifier is
  fail-closed). Implemented `ParallaxScheduledExecutor.plan_topology` (runs `_pre_execute_gates`
  = filter→allocate→route WITHOUT executing) + `POST /compute/inference/quote-multistage` →
  `build_per_stage_payee_set` → `{multi_stage, settleable, payees, payee_set_hash,
  total_value_wei}`. DRIFT-SAFE: one shared `topology_from_chain_stages` helper builds the
  stage→node positions for BOTH serve + quote (behavior-preserving refactor of
  TopologyAwareChainExecutor). Round-trip TESTED: quoted `payee_set_hash` == what
  `build_per_stage_payment_authorization` signs. 7 TDD + 302 regression.
- **S2 — Carry per-node settlement signatures on the receipt. ✅ DONE (sp1314, b8a64ef3).**
  REFINED from "split + route": code review showed the *foundational* gap is that the per-node
  signature material (which the RPC executor assembles, chain_rpc/client.py:838) was DROPPED
  at receipt-build, so the settle path never saw it. S2 threads
  `per_stage_settlement_signatures` onto `InferenceReceipt` (carried out-of-band, NOT in
  signing_payload → pre-1314 receipts byte-identical-signed; self-securing). 7 TDD + 1283
  regression. The split+route+commit are now consolidated into **S3** (routing to a
  non-consuming receiver would be inert, so route+receive+commit ship together).
- **S3 — Split + route + per-node accumulate + commit (consolidated; the money path).** On a
  settled multi-stage receipt: run the splitter (brick 1) on the now-carried per-node
  signatures (S2) → N node-signed `BatchedReceipt`s; ROUTE each (+ the per-stage auth) to its
  node over the P2P substrate; each node verifies the auth gate (brick 3b), accumulates to its
  OWN client, and the poll loop commits + finalizes (msg.sender = node). Replace the
  `"skipped:multi-stage-deferred"` guard. Still default-off. Risk: **HIGH** (money path + new
  transport — 2 sub-sprints: S3a route/receive, S3b per-node commit). Gates:
  conservation, idempotency (`{job_id}::stage::{node}`), fail-closed auth, dry-run, + a
  testnet 2-node proof before mainnet.
  - **S3a — split → routable tasks. ✅ DONE (sp1315, 0476c694).** Pure core (no transport, no
    commit). `prsm/settlement/per_stage_routing.py`: `PerStageSettlementTask` (frozen: node_id,
    share_wei, node-signed `BatchedReceipt`, per-stage auth; JSON round-trip reusing
    `published_batch_store` helpers) + `build_per_stage_settlement_tasks()` reads the
    sp1314-carried `receipt.per_stage_settlement_signatures`, runs the tested splitter, wraps
    each conserving share into a routable task. Fail-closed `None` exactly where the splitter
    falls back to single-payee (no sigs / <2 nodes / unmapped node). Integration-tested through
    the REAL splitter + `ComputeWalletMap`. 5 TDD + 135 per-stage regression.
  - **S3b — transport delivery + per-node verify + accumulate + commit (the money path).**
    Route each task to its node over the P2P substrate, each node verifies the auth gate +
    accumulates to its OWN client, poll loop commits + finalizes (msg.sender = node); replace
    the `"skipped:multi-stage-deferred"` guard. HIGH risk → full safety gates + testnet 2-node
    proof before mainnet. Sub-bricks:
    - **S3b-1 — node-side RECEIVER gate. ✅ DONE (sp1316, 6fed0de2).** Pure verify (no transport,
      no chain). Key shape: the requester signs over the WHOLE set's `payee_set_hash` but a node
      holds only its own task → `payee_set_from_tasks` derives the full `(payee, share)` set (the
      orchestrator has every task) to route alongside each node's task, and
      `verify_routed_settlement_task` runs the tested sp1172 verifier fail-closed for the node's
      OWN membership BEFORE any commit. Identical money-safety invariants pre-route + on-chain
      (signer/set-hash/membership/cap/expiry/request-binding). Returns `authorized=False` (never
      raises) on no-auth / malformed payload / bad sig shape / any invariant fail. 10 TDD through
      the REAL splitter + REAL signed auth.
    - **S3b-2 — receiver store + ingest (the receive half). ✅ DONE (sp1317, 798714a5).**
      `prsm/settlement/per_stage_receiver_store.py`: `PerStageReceiverStore` (bounded, GC-able,
      atomically-persisted; keyed by the per-node `local_escrow_id` `{job_id}::stage::{node_id}`
      — the SAME idempotency key the splitter stamps + the accumulator dedups on, so a
      re-delivery refreshes, never double-stages → never a double-commit; mirrors
      `IssuedAuthorizationStore` discipline) + `ingest_routed_task` (gate-then-stage: rejects +
      stages NOTHING on misroute / unauthorized membership / stage failure; never raises on a
      logical reject). Finding: the per-stage auth's WIRE shape is all-hex-strings (signature +
      every bytes32 hash) which JSON persistence depends on — the sp1316 gate tolerates raw
      bytes (verifier normalizes) but the store needs the wire shape; a persist failure is
      fail-soft-logged. Pure (no transport, no chain). 10 TDD through the REAL splitter + REAL
      signed auth. **Substrate decision: deliver over the existing settlement HTTP surface on
      `node/api.py` (matches sp1305 `/settlement/receipt/leaf` + the paid single-stage path
      where the requester POSTs the auth), NOT the inference chain_rpc (settlement is a
      post-inference concern — coupling it into the layer-slice server is the wrong fit).** The
      thin HTTP delivery endpoint (`POST /settlement/per-stage-task`) + the orchestrator-side
      client send is the next wiring brick (S3b-2b).
    - **S3b-2b — HTTP delivery endpoint + gated receiver-store resolver (the receive side).
      ✅ DONE (sp1318, 14988710).** `POST /settlement/per-stage-task` (`node/api.py`) → parse
      `{task, payees}` (`parse_delivery_request`, 422 on malformed) → `ingest_routed_task` into
      the node's store → `{accepted, reason, local_escrow_id}`. Gated by
      `PRSM_MULTISTAGE_SETTLEMENT` (`client_wiring.multistage_settlement_enabled`; default-off →
      503 + proven single-stage path byte-unchanged). `client_wiring.resolve_per_stage_receiver_store`
      lazily builds + caches the node's `PerStageReceiverStore`
      (`PRSM_MULTISTAGE_RECEIVER_STORE_FILE` or `~/.prsm/per_stage_receiver.json`); None when
      gated off / build fails (never a silent accept). NO on-chain commit. 20 TDD (endpoint
      end-to-end through the REAL splitter + REAL signed auth).
    - **S3b-2c — orchestrator-side send (the send half). ✅ DONE (sp1319, 28fb25ec).**
      `prsm/settlement/per_stage_delivery_client.py`: `deliver_per_stage_task(peer_url, task,
      payees)` POSTs one task (FAIL-SOFT — never raises; transport error / non-200 / non-JSON →
      `delivered=False`; on 200 surfaces `{accepted, reason, local_escrow_id}`; injectable
      `http_post`, mirrors `receipt_challenge_client`) + `deliver_settled_multistage_tasks(...)`
      builds tasks (S3a) → `[]` when not a settleable multi-stage receipt → derives the payee set
      once → POSTs each via `endpoint_for_node(node_id)` (an unmapped node = recorded undelivered,
      never a raise). FAIL-SOFT rationale: a miss leaves a stage unpaid on-chain (the v1
      per-stage-independence weakness, the SAFE failure). 8 TDD incl. a loop-back transport
      driving the REAL receiver (send→receive proven without a network). No chain effect.
      Remaining glue: the post-settlement HOOK that supplies the real peer-registry
      `endpoint_for_node` resolver — wired with S3b-3 (both fire post-settle).
    - **S3b-3 — per-node accumulate + commit + finalize (the on-chain money path).** On an
      authorized staged task, drive the node's own `BatchSettlementClient` (msg.sender = node);
      replace the `"skipped:multi-stage-deferred"` guard. Dry-run before broadcast; testnet
      2-node go/no-go before mainnet (mainnet commit stays user-gated).
      - **S3b-3a — node-side commit DRIVER. ✅ DONE (sp1320, 48d760d8).** `commit_staged_task`
        + `drain_and_commit_staged` on the receiver store (client duck-typed). RE-VERIFIES
        membership against the STORED FULL payee set at commit time (re-runs the sp1316 gate →
        catches an auth that EXPIRED between stage + commit, before accumulate), then accumulates
        the node-signed `BatchedReceipt` + drives `commit_ready_batches`; discards on a landed
        commit (idempotent — accumulator `local_escrow_id` dedup is the 2nd guard), retains on a
        non-commit (retryable). FAIL-SOFT (gate reject / bind-mismatch / commit error →
        `committed=False`, never raises). KEY divergence from the bench
        `commit_per_node_share_batches`: a node holds ONLY its own share, so it must verify
        against the stored full set (reconstructing from a lone share fails the set-hash check) —
        that's why the store retains `staged.payees`. 6 TDD with a fake client (no chain) incl.
        the expired-between-stage-and-commit money-safety reject. The actual broadcast stays the
        client's concern (view-only = no broadcast).
      - **S3b-3b — the live wiring (mostly DONE; one ingress brick + validation remain).**
        - **Delivery wiring ✅ DONE (sp1321, fae7c224).** `client_wiring.build_per_stage_endpoint_resolver(node)`
          (node_id→Optional[base_url]; static map `PRSM_MULTISTAGE_ENDPOINT_MAP` wins over the
          transport-peer fallback; None-on-miss) + `deliver_for_settled_receipt(node, ...)` (gated
          node entrypoint → fans out via S3b-2c). 9 TDD.
        - **Node-side poll-loop commit cycle ✅ DONE (sp1322, b0193c6f).**
          `client_wiring.run_per_stage_commit_cycle(node)` drains the receiver store + commits
          each staged share on the node's own client; wired into the node settlement poll loop
          (node.py) as an isolated never-raises step. Gated + fail-soft; INERT on a view-only
          client (tasks accumulate until a funded key, mirroring the single-stage cycle). 5 TDD.
        - **Paid-multi-stage request INGRESS + post-settle delivery hook ✅ DONE (sp1324,
          a5a9aff0).** A paid multi-stage request carries `per_stage_payment_authorization`;
          `_resolve_paid_requester_or_402` (gate-on) authenticates the requester
          (`recover_per_stage_signer == payload.requester`, 402 on mismatch/malformed) + carries
          the auth in the paid info dict `{multi_stage, requester, max_spend_wei,
          per_stage_authorization}`. The post-settle hook (api.py settle site) fires
          `deliver_for_settled_receipt(node, ..., total_value_wei = release_to_operator * 1e18)`
          — conservation-correct (release_to_operator is the FULL settled value; both the
          off-chain split and the S3a splitter distribute it conservingly). The FULL money gate
          runs fail-closed at each node's commit (sp1316), since the served payee set isn't known
          until routed; FAIL-SOFT delivery. The `"skipped:multi-stage-deferred"` guard STAYS (it
          correctly prevents single-payee over-booking) — its comment now flags the multi-stage
          path as wired separately at the post-settle hook. 7 TDD + 60 regression
          (single-stage sp1056 unchanged). **★ The big-model paid path is now wired end-to-end in
          code.**
        - **VALIDATION.** **In-process end-to-end ✅ DONE (sp1323, 8a7185e1):**
          `tests/integration/test_sprint_1323_per_stage_e2e.py` wires the WHOLE chain over REAL
          HTTP (FastAPI TestClient) in one process — orchestrator split → `deliver_for_settled_receipt`
          → POST each → 2 receiver nodes (real apps + stores) → gate → stage →
          `run_per_stage_commit_cycle` commits 1/1 + drains; plus the money-safety reject (auth
          signed over a different set → each node's gate fail-closes, nothing staged/committed).
          This de-risks the live run. **★★★ 2-NODE CROSS-CLOUD TESTNET GO ACHIEVED (2026-06-30,
          Base Sepolia chainId 84532): the full chain ran end-to-end across two DIFFERENT cloud
          providers over the public internet** — cross-host 2-stage Qwen-7B inference (signed §7
          chain, linked activation hashes) → multi-stage quote → escrow → per-stage auth → paid
          serve+settle → post-settle delivery (head IN-PROCESS, worker over an SSH reverse tunnel
          since its inbound was firewall-blocked) → both nodes gate-verify + stage → **each node
          self-committed its own share on-chain (on-chain provider == own settler, msg.sender)**:
          head batch `ca30cbc6…` (Settler-A 0xBbEB…, 0.14 FTNS), worker batch `f98566d9…`
          (Settler-B 0x2010…, 0.14 FTNS), both PENDING + requester-bound; **conservation
          0.14+0.14 == 0.28 FTNS (settled cost) verified on-chain.** Finalize pending the
          challenge window. Three real bugs found+fixed live (sp1326/1327, commit ce265706): the
          16MB WS frame cap (too small for Qwen's 152k vocab → MESSAGE_TOO_BIG killed every
          cross-host dispatch — THE actual blocker, not the firewall), the head self-delivery
          re-entrancy (self-HTTP-POST deadlock → ingest in-process), and transport reconnect
          robustness (ping-timeout env + reconnect-replace for asymmetric-firewall fleets). Two
          design findings: quote-vs-settled-value binding (exact-share auth needs budget==cost or
          a binding/cap-based quote), and the 1.0 FTNS batch threshold being too high for small
          per-stage shares. **REAL-EVM proof of the routed path ✅ DONE (sp1325, 0b46e91f):** `test_sp1325_per_stage_ROUTED_onchain_payout_e2e` (in the sp1159 local-EVM
          bench) runs the DISTRIBUTED self-commit chain — S3a split → per-node gate+stage
          (S3b-1/2) → `drain_and_commit_staged` on each node's OWN real `BatchSettlementClient`
          (S3b-3a) → finalize — against real deployed contracts on a local hardhat chain, and
          asserts self-commit (provider == own settler) + conservation + exact payout + escrow
          drained by the total. So a REAL client broadcasting the routed path is now proven; the
          live 2-host Base Sepolia run adds only real network hosts + a real testnet RPC over
          this. **Still gated, infra-dependent:** a **2-live-node testnet go/no-go before
          mainnet** (now low-risk — protocol + money path proven on a real EVM). The
          2-live-node deployment + the mainnet activation are user-gated (irreversible — the
          autonomous loop pauses there). **The operator go/no-go procedure is written:
          `docs/2026-06-30-big-model-paid-settlement-2node-testnet-runbook.md`** (topology + 3
          funded EOAs, the wallet-map==settler-key invariant, per-node env, the full
          quote→sign→infer→deliver→commit→finalize flow, read-only verification, GO/NO-GO
          criteria, and mainnet activation as a separate user-gated ceremony).
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
