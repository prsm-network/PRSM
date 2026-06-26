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
2. **Foundation Safe (2-of-3, gated):** batch-call `acceptOwnership()` on each of the 3 new contracts
   (pattern: `contracts/deployments/safe-batch-acceptOwnership-mainnet-2026-05-07.json`). After
   acceptance, re-run §4.4 with `EXPECTED_OWNER=0x91b0e6F8…5791` to confirm `owner()` == Safe.

### 4.6 Cutover

- Finalize/expire old-registry pending batches; confirm old escrow drained.
- Confirm new-bond stake quorum re-established.
- Push the new addresses to all settlement clients **and** flip `supports_attestation` on so clients
  route `commitBatchWithAttestation` (sp1241). With E live, the committed measurement is now a real
  hardware quote.
- (Optional, F's headline payoff) wire an indexer/Forta filter on `BatchAttestationCommitted` topic0
  `0xec923112ccc386fa91e7116abfe5da0211d8908195bb5d41e644c8a0c79222e3`.

---

## 5. What the assistant did vs. what is gated

**Done (this prep, autonomous):**
- Authored + **validated** `scripts/verify-attestation-commitment-deployment.js` end-to-end on a
  local node — positive deploy → exit 0 with on-chain invariance proof; non-registry → exit 1.
- Confirmed `deploy-audit-bundle.js` already emits the sp1240-capable registry (no script change).
- Authored this runbook, including the immutable-cross-wire finding and the E-before-F sequencing.

**Gated (Foundation / hardware — assistant must NOT do autonomously):**
- Roadmap **E** hardware-validation on a real SGX/SEV-SNP VM (precondition for a *meaningful* F).
- Signing/broadcasting the mainnet deployer txs.
- The Foundation 2-of-3 `acceptOwnership` ceremony.
- The migration cutover (stake re-bond, pending-batch drain, client re-pointing).

---

## 6. Abort criteria

- §4.4 verification fails (cross-wire mismatch OR invariance dry run shows differing `batchId`) →
  **abort**, do not hand off. A differing `batchId` means attestation leaked into the consensus
  preimage — a contract-level regression, not an ops issue.
- Old-registry PENDING batches cannot all be finalized/expired before the window → **postpone**
  cutover; never drain the old escrow with unsettled value.
- E hardware-validation not yet complete → **do not run Track B**; Track A (Sepolia) only.
```
