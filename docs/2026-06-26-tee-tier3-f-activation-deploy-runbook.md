# TEE Tier-3 Roadmap F — On-Chain Attestation Commitment Activation Runbook

**Sprint lineage:** sp1240 (contract) · sp1241 (weakest-link commitment + client ABI routing) · sp1242 (commit glue) · this runbook (sp1286 deploy-prep)
**Status:** PREP COMPLETE. The mainnet activation is a **gated Foundation ceremony** — the assistant does **not** sign or broadcast. This runbook + the validated verification script (`contracts/scripts/verify-attestation-commitment-deployment.js`) are the operator-facing deliverables.
**Author of prep:** automated (Claude). **Owner of execution:** PRSM Foundation 2-of-3 multisig signers.

---

## 0. What this activates (and what it does NOT change)

sp1240 added, to `BatchSettlementRegistry.sol`, an **additive** path that commits a batch's
TEE-attestation measurement on-chain:

- `commitBatchWithAttestation(address,bytes32,uint256,uint256,uint16,bytes32,string,bytes32)`
  — selector `0xfa4ce156`. Identical to `commitBatch` in every consensus-relevant way; the
  only additions are the stored `attestationCommitment` + a **separate** event
  `BatchAttestationCommitted(bytes32 indexed batchId, address indexed provider, bytes32 attestationCommitment)`
  (topic0 `0xec923112ccc386fa91e7116abfe5da0211d8908195bb5d41e644c8a0c79222e3`).
- Legacy `commitBatch` (selector `0x95e5ccec`) is **byte-identical** and still present — un-upgraded
  clients keep working with zero changes.

**Consensus invariance is the load-bearing property.** `attestationCommitment` is NOT in the
`batchId` keccak preimage (`_commitBatch`), so the two functions return the *same* `batchId` for the
same economic inputs. This is the on-chain analogue of the sp1238 leaf-invariance unit test and was
**proven on a live deploy** during prep:

```
commitBatch                → batchId 0x365b1764768f9ed872d2b702e41f7d9747043301f41e12c696aa66ac0728a7e0
commitBatchWithAttestation → batchId 0x365b1764768f9ed872d2b702e41f7d9747043301f41e12c696aa66ac0728a7e0   ✓ identical
```

Activating F therefore changes **no** settlement economics: same `batchId`, same `merkleRoot`, same
challenge / finalize / slash semantics. It only adds an auditor-correlatable on-chain pointer from a
batch to the TEE tier/measurement its receipts claimed.

---

## 1. ★ The structural finding that shapes this ceremony

`BatchSettlementRegistry` is **`Ownable2Step, Pausable` — NOT a proxy.** Solidity contracts are
immutable, so "deploy the upgraded registry" is **not** an in-place implementation swap. Worse, the
two consensus-critical cross-wires are **immutable** (post-L2-audit HIGH-6/HIGH-7):

- `EscrowPool.settlementRegistry` — constructor-only.
- `StakeBond.slasher` — constructor-only.

The **live** EscrowPool (`0x…` wired to the live registry `0x48fFab641b9D638F312FFA776818756a326F995B`)
can therefore **never be re-pointed** at a new registry. A new registry's `setEscrowPool` would make
the *new* registry know the *old* escrow, but the old escrow would still settle to the old registry —
so a side-deployed registry would never receive the production settlement flow.

**Conclusion: activating F on the production settlement path requires a *fresh audit-bundle*
(new registry + new EscrowPool + new StakeBond) and a migration cutover — not a one-contract deploy.**
`scripts/deploy-audit-bundle.js` already builds the sp1240-inclusive bundle (the source carries
`commitBatchWithAttestation` today), so no script change is needed; the cost is the *migration*, not
the deploy.

---

## 2. ★ Recommendation: sequence F **after** E, in one migration

Roadmap **E** (real SGX/SEV-SNP hardware quote generation, `SgxTEERuntime`/`SevSnpTEERuntime`,
sp1243) is **hardware-validation-pending**. Until E lands, the only attestation a node can commit is
the **dev-only software blob** (`SoftwareTEERuntime.get_attestation_bytes`). Committing dev-only
measurements on-chain — via a disruptive mainnet migration (7-day stake unbond, pending-batch drain,
client re-pointing) — buys nothing and front-runs the value.

**Therefore:**

| Track | When | Risk | Who |
|---|---|---|---|
| **A — Sepolia rehearsal + client-routing proof** | NOW | none (testnet) | assistant-prep + operator runs |
| **B — Mainnet bundle migration** | **defer until E hardware-validation lands**, then activate E+F together | high (migration) | Foundation 2-of-3 (gated) |

Track A fully exercises F (the contract path + sp1241/sp1242 client routing) with zero mainnet risk.
Track B is the irreversible, multisig-gated step and should ride the **same** migration that activates
E so the chain only ever migrates once.

---

## 3. Track A — Base Sepolia rehearsal (do this now)

Proves: the sp1240 contract deploys, the verify script gates correctly, and the settlement client
routes `commitBatchWithAttestation` when `supports_attestation` is on (sp1241/sp1242).

### 3.1 Env (Sepolia)

```bash
cd contracts
export PRIVATE_KEY="0x…"                     # Sepolia deployer hot key (testnet funds only)
export BASE_SEPOLIA_RPC_URL="https://sepolia.base.org"
export FTNS_TOKEN_ADDRESS="0x…"              # a Sepolia FTNS (or deploy-mock-ftns.js)
export FOUNDATION_RESERVE_WALLET="0x…"       # a CONTRACT (StakeBond rejects EOAs); a test Safe or any deployed contract
export AUTO_VERIFY=1                          # Basescan verify (needs ETHERSCAN/BASESCAN key)
```

### 3.2 Deploy + verify

```bash
# 1. deploy the sp1240-inclusive bundle
npx hardhat run scripts/deploy-audit-bundle.js --network base-sepolia
#    → writes deployments/audit-bundle-base-sepolia-<ts>.json

# 2. cross-wire verification (existing)
AUDIT_BUNDLE_MANIFEST=deployments/audit-bundle-base-sepolia-<ts>.json \
  npx hardhat run scripts/verify-audit-bundle-deployment.js --network base-sepolia

# 3. ★ sp1240 surface + consensus-invariance verification (NEW — this prep)
AUDIT_BUNDLE_MANIFEST=deployments/audit-bundle-base-sepolia-<ts>.json \
  REQUIRE_INVARIANCE_DRYRUN=1 \
  npx hardhat run scripts/verify-attestation-commitment-deployment.js --network base-sepolia
```

**Acceptance:** script 3 prints `✓ selector 0xfa4ce156 present`, `✓ legacy commitBatch … present`,
and `✓ batchId INVARIANT to attestation`, exiting `0`. (Validated end-to-end on a local node during
prep: positive deploy → exit 0; non-registry address → exit 1.)

### 3.3 Client-routing proof (sp1241/sp1242)

Point a settlement client at the Sepolia registry with attestation routing enabled, commit a batch
carrying receipts that include `tee_attestation_audit_dict()`, and confirm:

- the client calls `commitBatchWithAttestation` (not `commitBatch`) — `supports_attestation` true +
  non-zero `batch_attestation_commitment()`;
- a `BatchAttestationCommitted` event fires with the off-chain-recomputed weakest-link commitment
  (auditor-reproducible via `batch_attestation_commitment(receipts)`);
- the batch finalizes/settles **identically** to a legacy batch (invariance holds in practice).

---

## 4. Track B — Mainnet bundle migration (GATED; defer until E)

Run only when roadmap E hardware-validation has produced real hardware quotes AND the Foundation has
scheduled the migration window. Mirrors `docs/2026-04-30-post-audit-deploy-ceremony-runbook.md`; the
deltas specific to F are called out.

### 4.1 Known mainnet addresses

| Role | Address |
|---|---|
| Foundation 2-of-3 Safe (owner of all Ownable contracts) | `0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791` |
| Deployer hot key (mechanical wiring under tested invariants) | `0xF7d88c943B048dAd2e5178E40DaaD545dB3311c2` |
| **Live** BatchSettlementRegistry (pre-sp1240, to be retired) | `0x48fFab641b9D638F312FFA776818756a326F995B` |
| FTNS token (Base mainnet) | per `contracts/DEPLOYMENT_GUIDE.md` |

### 4.2 Pre-flight: migration hazards (start T-8d because of the 7-day unbond)

1. **StakeBond stake migration.** `StakeBond.slasher` is immutable → the new bond is a new contract.
   Providers must `requestUnbond` on the old bond (**7-day `UNBOND_DELAY_SECONDS`**) and re-stake on
   the new one. **This is the long pole — begin operator comms ≥ 8 days before cutover.**
2. **Old-registry pending batches.** Every PENDING batch in the old registry must pass its challenge
   window and `finalizeBatch` (or be allowed to expire) **before** the old escrow is drained. Do not
   cut over with unsettled value in the old registry.
3. **Client re-pointing.** Every settlement client + watchdog must move to the new
   registry/escrow/bond addresses atomically at cutover (config push, not a rolling deploy that
   straddles both).

### 4.3 Mainnet deploy (deployer hot key)

**Precondition (review C1):** broadcast to mainnet ONLY via `--network base` (name `base`, chainId
`8453`). NEVER use a fork/alias network whose `url` points at a live RPC — fork rehearsals run
`hardhat node --fork $BASE_RPC_URL` then deploy with `--network localhost`. sp1293 hardened
`deploy-audit-bundle.js` + `transfer-ownership.js` to key the mainnet guards off the CONNECTED chainId
(not the network name) and to **fail closed** on a real-mainnet chainId under a non-mainnet, non-local
name — so a `base-fork`-style slip can no longer silently disable the mock-verifier ban / foundation≠
deployer check / production-verifier default. Confirm the connected chainId is `8453` before signing.

```bash
cd contracts
export PRIVATE_KEY="0x…"                     # deployer hot key 0xF7d88c94…11c2
export BASE_RPC_URL="https://mainnet.base.org"
export FTNS_TOKEN_ADDRESS="0x…"              # LIVE FTNS
export FOUNDATION_RESERVE_WALLET="0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791"   # Foundation Safe
# DO NOT set USE_MOCK_VERIFIER — the script deploys the production Ed25519Verifier on mainnet.
export AUTO_VERIFY=1

npx hardhat run scripts/deploy-audit-bundle.js --network base
#   → new registry + escrow + bond, cross-wired, owner == deployer (pre-handoff)
```

### 4.4 Verify BEFORE handoff (still under deployer ownership)

```bash
MAN=deployments/audit-bundle-base-<ts>.json

# cross-wires
AUDIT_BUNDLE_MANIFEST=$MAN \
  npx hardhat run scripts/verify-audit-bundle-deployment.js --network base

# sp1240 surface + on-chain consensus-invariance (NEW)
AUDIT_BUNDLE_MANIFEST=$MAN REQUIRE_INVARIANCE_DRYRUN=1 \
  npx hardhat run scripts/verify-attestation-commitment-deployment.js --network base
```

Both must exit `0`. The invariance dry run is `eth_call` only — it writes no state and is safe to run
against the fresh, unpaused mainnet registry.

### 4.5 Foundation Safe ownership handoff (Ownable2Step — 2 signing steps)

1. **Deployer:** `transfer-ownership.js` sets `pendingOwner = Foundation Safe` on the 3 new contracts.
   ```bash
   FOUNDATION_MULTISIG=0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791 \
     AUDIT_BUNDLE_MANIFEST=$MAN \
     npx hardhat run scripts/transfer-ownership.js --network base
   ```
   **(review M3) Run this IMMEDIATELY before the scheduled Foundation signing slot — transfer + accept
   in the same short session, not days apart.** `transferOwnership` only sets `pendingOwner`; the
   deployer hot key STILL fully owns all 3 contracts (pause / setFoundationReserveWallet / slash-routing)
   until `owner()`==Safe is reconfirmed in step 3. Let the 7-day unbond + pending-batch drain elapse
   while ownership is simply deployer-owned (its natural pre-handoff state); minimise the
   post-transfer/pre-accept hot-key window. If the Safe ceremony is delayed/aborted after this step,
   treat the hot key as load-bearing and, if it's suspected lost, immediately re-`transferOwnership` to
   a fresh deployer-controlled holding address to invalidate the prior `pendingOwner`.
2. **Generate the Safe batch (sp1291, offline — no key, no RPC):** turn the deploy manifest into a
   ready-to-import Safe{Wallet} Transaction Builder bundle of the three `acceptOwnership()` calls.
   ```bash
   node scripts/build-f-activation-safe-txs.js \
     --manifest $MAN \
     --safe-address 0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791
   #   → deployments/safe-batch-acceptOwnership-f-activation-<network>-<ts>.json
   ```
   It auto-fills the 3 Ownable addresses from the manifest (Registry/EscrowPool/StakeBond — the
   verifier is not Ownable), verifies the `acceptOwnership()` selector `0x79ba5097`, and warns if the
   Safe ≠ the known Foundation 2-of-3 on Base mainnet. The full deploy→generate→transferOwnership→
   accept→`owner()`==Safe ceremony was **dress-rehearsed end-to-end on a local node** (executing the
   generated txs from the Safe completes the Ownable2Step handoff on all 3).
3. **Foundation Safe (2-of-3, gated):** import that JSON in the Safe UI → Transaction Builder and sign
   + execute the batch (`acceptOwnership()` on each of the 3 new contracts; same shape as the proven
   `contracts/deployments/safe-batch-acceptOwnership-mainnet-2026-05-07.json`). After acceptance,
   re-run §4.4 with `EXPECTED_OWNER=0x91b0e6F8…5791` to confirm `owner()` == Safe.

### 4.6 Cutover

Order matters — **re-point + SOAK on the new bundle BEFORE retiring the old one**, so the old bundle
stays a hot fallback until the new path is proven (review M5). The steps below close the review's H1/H2
+ M1/M2/M5 gaps; they need no contract change (both old contracts are Pausable + Foundation-owned).

1. **Stop new old-registry work + stop old-pool deposits (M2).** Disable the deposit code path/UI for
   the OLD EscrowPool and announce the freeze to clients + integration partners. Keep `withdraw()`
   OPEN. Do NOT `pause()` the EscrowPool (that would also block `withdraw()` and trap funds).
2. **Drain legit pending, THEN freeze the old registry (H1).** Finalize/expire all legitimately-PENDING
   old batches first (`finalizeBatch` is `whenNotPaused`), then the **Foundation 2-of-3 calls
   `pause()` on the OLD registry** (`0x48fFab…995B`). A paused registry rejects new `commitBatch`, so
   the drain reading becomes DURABLE — no new PENDING batch can appear after the scan (closes the
   TOCTOU). `_effectiveElapsed` credits paused time, so the pause never robs a challenger of their window.
3. **Gate on the drain check (sp1292 + sp1294 hardening).** Now durable:
   ```bash
   OLD_REGISTRY_ADDRESS=0x48fFab641b9D638F312FFA776818756a326F995B \
     FROM_BLOCK=45687572 \
     BASE_RPC_URL=<PAYG endpoint> \
     npx hardhat run scripts/verify-f-activation-cutover-readiness.js --network base
   # exit 0 REQUIRES: 0 PENDING + scan-complete (provider-sequence reconciled) + old registry PAUSED.
   ```
4. **Account for residual escrow (H2) — NOT a hard gate.** The old EscrowPool has NO admin drain (by
   design — anti owner-drain, L2 HIGH-6), so `totalEscrowedBalance()` may be non-zero and that value is
   **recoverable-not-stranded**: each depositor recovers it via their own `withdraw()`. So snapshot
   residual depositors (the check prints the balance + you can enumerate `Deposited`/`Withdrawn`
   events), run a self-service-withdraw comms campaign, and **keep the old EscrowPool UNPAUSED forever**
   so recovery stays available. Do NOT block cutover on a zero escrow balance, and do NOT add an admin
   drain (it would reintroduce L2 HIGH-6).
5. **Re-stake with OVERLAP (M1).** Providers bond on the NEW StakeBond and have it ACTIVE before
   `requestUnbond` on the old bond; the old registry must already be paused (step 2) before any
   old-bond withdraw, so no batch can be committed against now-un-slashable stake (avoids the
   `SlashSwallowed` window during the 7-day unbond).
6. **Re-point clients + SOAK (M5).** Both the registry re-point and the attestation flip are pure
   **config** (sp1299) — no code change, no redeploy. Push the new registry to all settlement clients
   atomically (config push, not a straddling rolling deploy):
   ```bash
   PRSM_SETTLEMENT_REGISTRY_ADDRESS=0x12a01F6C487d765af389bC7D95D90b3136a391F2   # new bundle
   # (and, for any consumer that reads them) PRSM_ESCROW_POOL_ADDRESS / PRSM_STAKE_BOND_ADDRESS
   ```
   then restart the daemon (the address resolves once at startup). Run a mainnet canary —
   commit+finalize one batch through the new bundle, assert identical settlement — and soak healthy
   BEFORE retiring the old bundle. The old bundle stays a hot fallback; if the new bundle misbehaves,
   revert `PRSM_SETTLEMENT_REGISTRY_ADDRESS` to the old (still-owned, still-functional) bundle.
7. **Flip `supports_attestation` on (LAST)** so clients route `commitBatchWithAttestation` (sp1241):
   ```bash
   PRSM_SETTLEMENT_SUPPORTS_ATTESTATION=1   # sp1299 — config flip, default OFF
   ```
   This is FAIL-SAFE: `client_wiring` confirms the bound registry actually exposes
   `commitBatchWithAttestation` (selector `0xfa4ce156`, ABI-derived bytecode probe) before enabling;
   if it can't confirm (e.g. the registry env still points at the OLD bundle), it logs a warning and
   stays on legacy `commitBatch` rather than sending reverting txs. So do step 6 (re-point) first.
   With E live, the committed measurement is now a real hardware quote.
8. (Optional, F's headline payoff) wire an indexer/Forta filter on `BatchAttestationCommitted` topic0
   `0xec923112ccc386fa91e7116abfe5da0211d8908195bb5d41e644c8a0c79222e3`.

---

## 5. What the assistant did vs. what is gated

**Done (this prep, autonomous):**
- Authored + **validated** `scripts/verify-attestation-commitment-deployment.js` end-to-end on a
  local node — positive deploy → exit 0 with on-chain invariance proof; non-registry → exit 1.
- Confirmed `deploy-audit-bundle.js` already emits the sp1240-capable registry (no script change).
- Authored + **validated** `scripts/build-f-activation-safe-txs.js` (sp1291) — the offline generator
  that turns a deploy manifest into the Foundation's importable Safe `acceptOwnership` batch.
  Dress-rehearsed the FULL ownership ceremony end-to-end on a local node (deploy → generate →
  transferOwnership → execute the generated txs from the Safe → `owner()`==Safe on all 3).
- Authored + **validated** `scripts/verify-f-activation-cutover-readiness.js` (sp1292) — the read-only
  pre-cutover gate that scans the old registry for unsettled PENDING batches (validated on a local
  node: 2 pending→exit 1, 1→exit 1, 0→exit 0 drained).
- **Dress-rehearsed the FULL ceremony against a Base MAINNET FORK** (real live FTNS token + real
  Foundation Safe): deploy → both verify scripts → Safe-tx generation → transferOwnership →
  impersonated-Safe `acceptOwnership` → `owner()`==Safe on all 3. Strictly stronger than Sepolia
  (real contracts). Confirmed the live registry `0x48fFab…995B` is pre-sp1240 (upgrade needed).
- **Ran an adversarial pre-mainnet review** of the whole migration (6 dimensions → 12 confirmed / 16
  dismissed). Verdict: NO-GO as-written → GO-WITH-FIXES; contract layer + sp1240 consensus-neutrality
  sound. Landed all confirmed fixes (script/config only, no contract change): C1 chainId-based deploy
  guards + base-fork landmine (sp1293), H3 EscrowPool verify args (sp1293), H1 pause-durable cutover
  gate + H2 escrow-balance read + M4 FROM_BLOCK/completeness (sp1294), and M1/M2/M3/M5 procedure
  (this runbook: freeze-before-scan, recoverable-not-stranded escrow, transfer+accept same session,
  soak-before-retire, rollback). E-before-F sequencing confirmed + strengthened.
- Authored this runbook, including the immutable-cross-wire finding and the E-before-F sequencing.

**Gated (Foundation / hardware — assistant must NOT do autonomously):**
- Roadmap **E** hardware-validation on a real SGX/SEV-SNP VM (precondition for a *meaningful* F).
- Signing/broadcasting the mainnet deployer txs.
- The Foundation 2-of-3 `acceptOwnership` ceremony.
- The migration cutover (stake re-bond, pending-batch drain, client re-pointing).

---

## 5.5 ★ LIVE MAINNET EXECUTION RECORD (2026-06-29)

Roadmap E hardware-validation completed first (real AMD SEV-SNP on a GCP N2D Milan
Confidential VM — real 5408B quote verified to the genuine AMD ARK, `vendor_verified=True`,
node-bound; sp1296/sp1297). With E satisfied, the gated mainnet F migration was executed live
on Base mainnet (chainId 8453). Phases 1–3 are **complete**; Phase 4 (cutover) remains scheduled.

**New F-capable bundle (sp1240-inclusive), live on Base mainnet** — manifest
`deployments/audit-bundle-base-1782757967581.json`:
- BatchSettlementRegistry: `0x12a01F6C487d765af389bC7D95D90b3136a391F2`
- EscrowPool:              `0x4e93a04b3A0C5063FE584980e6c2B1429495EEa1`
- StakeBond:               `0x21B5de0f65B9273A715C6a02b7085a8ABE8adA72`
- Ed25519Verifier:         `0x9d369312bf3b502Bc07c5859a18f7158c22A31e1`
- Deployer (hot key):      `0xF7d88c943B048dAd2e5178E40DaaD545dB3311c2`
- Foundation 2-of-3 Safe:  `0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791`
- Live FTNS:               `0x5276a3756C85f2E9e46f6D34386167a209aa16e5`
- Old (pre-sp1240) registry: `0x48fFab641b9D638F312FFA776818756a326F995B` (still owned/live — Phase-4 fallback)

- **Phase 1 — deploy ✅** (§4.3). All 4 contracts deployed + cross-wired; `owner()`==deployer (pre-handoff).
- **Phase 2 — verify ✅** (§4.4). Cross-wires match; sp1240 surface present; on-chain `batchId`
  invariance dry run `0xfeb5f4a2eced…` (`commitBatch` == `commitBatchWithAttestation` → zero
  consensus impact). All 4 contracts **Basescan source-verified**. (sp1298 fixed StakeBond verify args.)
- **Phase 3 — ownership handoff ✅** (§4.5). `transfer-ownership.js` set `pendingOwner`==Safe on all 3
  (`ownership-transfer-base-1782758519227.json`):
  EscrowPool `0x617a4940…e468e`, Registry `0xf7cafd04…105f1`, StakeBond `0xb15cda34…b003`.
  Foundation 2-of-3 executed the generated `acceptOwnership` batch
  (`deployments/f-activation-safe-acceptownership-base.json`, selector `0x79ba5097`).
  Post-accept read-only verify: `owner()`==Safe on all 3, `pendingOwner` cleared on all 3.
- **Phase 4 — cutover ⏳ GATED/SCHEDULED** (§4.6). Pause old registry → drain PENDING batches (needs a
  PAYG RPC for the `eth_getLogs` scan) → re-stake with 7-day unbond → re-point clients → soak →
  flip `supports_attestation`. Keep the old bundle live as fallback until the new path soaks healthy.

---

## 6. Abort criteria

- §4.4 verification fails (cross-wire mismatch OR invariance dry run shows differing `batchId`) →
  **abort**, do not hand off. A differing `batchId` means attestation leaked into the consensus
  preimage — a contract-level regression, not an ops issue.
- Old-registry PENDING batches cannot all be finalized/expired before the window → **postpone**
  cutover; never drain the old escrow with unsettled value.
- E hardware-validation not yet complete → **do not run Track B**; Track A (Sepolia) only.
- (review C1) Connected chainId is a real mainnet (8453/1/137) under a non-`base`/non-local network
  name, OR the network name is `base`/`mainnet` but the chainId is not mainnet → **abort** (the deploy
  scripts now throw on this; do not override).
- (review H1) The cutover drain check (§4.6 step 3) reports the old registry is NOT paused at scan time
  → **abort** the cutover; pause the old registry first so the drain reading is durable.
- (review M5) The new bundle fails its post-cutover soak/canary (§4.6 step 6) → **roll back**: revert
  client config to the old (still-owned, still-functional) bundle; do NOT drain the old escrow or
  start old-bond unbonding until the new path has soaked healthy. Because the new bundle's cross-wires
  are immutable, a botched new bundle is repaired by re-deploying a fresh bundle (back to §4.3), not in
  place — keeping the old bundle live as a fallback until then is mandatory.
```
