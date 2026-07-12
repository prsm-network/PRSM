# DOUBLE_SPEND-fix (sp1429) Migration Runbook — BatchSettlementRegistry

**Status:** assistant-prepped; **mainnet execution is GATED** (Foundation 2-of-3 ceremony).
**Prepared:** 2026-07-12. **Prep sprint:** sp1435.

This activates the sp1429 fix to `BatchSettlementRegistry._handleDoubleSpend` on Base mainnet. It is
structurally a repeat of the F-migration (`docs/2026-06-26-tee-tier3-f-activation-deploy-runbook.md`)
and reuses that ceremony's exact, mainnet-proven tooling. Read that runbook alongside this one; only
the DELTAS are spelled out here.

---

## 0. What this activates (and what it does NOT change)

- **Activates:** the sp1429 fix. `_handleDoubleSpend` previously had NEITHER guard its sibling
  `_handleConsensusMismatch` enforces — no self-reference block and no first-committer binding — so
  ANY address could `challengeReceipt(id, leaf, proof, DOUBLE_SPEND, abi.encode(id, proof))` against
  a legitimate batch and invalidate the provider's payment, slash its stake, and collect the 70%
  bounty (self-reference variant), or commit a copycat batch and challenge the honest first committer
  (copycat variant). See `project_money_path_audit_sp1428_1431` and the two SECURITY tests in
  `contracts/test/BatchSettlementChallenge.test.js`.
- **Does NOT change:** the ABI/interface, the batchId keccak preimage, the challenge/finalize/slash
  semantics for every OTHER reason code, or the attestation surface (roadmap F). The new bundle is
  the SAME source as the live F bundle **plus** this handler fix — a drop-in with an identical
  interface. So settlement clients need only a config re-point (no code change), and the sp1240
  attestation feature carries over unchanged.
- **Exposure today is LOW** (only self-pay canary batches exist; no funded third-party providers have
  committed a PENDING batch), but this MUST be migrated before any real bonded provider settles — the
  receipt-invalidation half fires even with no stake bond.

## 1. Why a fresh bundle (not a 1-contract swap)

`BatchSettlementRegistry` is `Ownable` (Ownable2Step), NOT a proxy, so the fixed bytecode can only go
live at a NEW address. And the reverse cross-wires are **immutable by deliberate hardening**:
`EscrowPool.settlementRegistry` (L2 audit HIGH-6) and `StakeBond.slasher` (HIGH-7) are `immutable`
with no setter. So the existing EscrowPool/StakeBond CANNOT be re-pointed to a new registry — a new
registry needs a NEW EscrowPool + NEW StakeBond wired to it. This is exactly the F situation, so we
deploy the full audit bundle again. (The registry's OWN refs — `setEscrowPool`/`setStakeBond`/
`setSignatureVerifier` — are owner-settable, but that direction is not the blocker.)

## 2. Migrate-FROM addresses (current live F bundle)

| contract | current (migrate FROM) |
|---|---|
| settlement_registry | `0x12a01F6C487d765af389bC7D95D90b3136a391F2`  ← the buggy one |
| escrow_pool | `0x4e93a04b3A0C5063FE584980e6c2B1429495EEa1` |
| stake_bond | `0x21B5de0f65B9273A715C6a02b7085a8ABE8adA72` |
| signature_verifier | `0x9d369312bf3b502Bc07c5859a18f7158c22A31e1` (production Ed25519Verifier) |
| FTNS token | `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` (unchanged; the new escrow/bond re-use it) |
| Foundation 2-of-3 Safe | `0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791` (ownership target) |

Source of truth: `prsm/config/networks.py` (MAINNET). The pre-F bundle is already RETIRED there.

## 3. Pre-flight hazards (start T-8d because of the 7-day unbond)

Identical to F §4.2. The one that bites: any provider bonded on the OLD `StakeBond` must re-bond on
the NEW one with OVERLAP before `requestUnbond` on the old, and the OLD registry must be PAUSED before
any old-bond withdraw (so no batch can be committed against now-un-slashable stake). Today the old
StakeBond stake is ~0 (per the F cutover record), so this is likely a no-op — CONFIRM with a live read
before cutover.

## 4. Fork rehearsal — ✅ DONE (assistant-rehearsed 2026-07-12, sp1435)

**RESULT: GO.** The full ceremony was dress-rehearsed against a LOCAL fork of Base mainnet
(`--fork` at block 48550403, real FTNS + Safe state), all green:

- `deploy-audit-bundle.js` → clean deploy + cross-wire; post-deploy invariants all ✅
  (`escrow.settlementRegistry`/`registry.escrowPool`/`stakeBond.slasher`/`foundationReserveWallet`).
- `verify-audit-bundle-deployment.js` → "All on-chain state matches manifest."
- `verify-attestation-commitment-deployment.js` → sp1240 selector present + `batchId` INVARIANT to
  attestation (roadmap F carries over unchanged).
- ★ `verify-doublespend-fix-active.js` → **exit 0** on the freshly-deployed registry (self-ref +
  copycat blocked, legit intact). **NEGATIVE CONTROL: the SAME script against the OLD live registry
  `0x12a01F6C…` exits 1 — the self-reference AND copycat attacks ACTUALLY SUCCEED on the real
  deployed mainnet contract (forked state).** This is on-chain confirmation the vuln is live today
  and that the fresh bundle closes it.
- `transfer-ownership.js` → `pendingOwner` set to the Foundation Safe on all 3 (Registry/EscrowPool/
  StakeBond); the 2-of-3 `acceptOwnership` is the gated step (F-rehearsed; `build-f-activation-safe-txs.js`
  is bundle-agnostic).

The commands below are the reproducible recipe (operator re-runs with their own Base archive RPC).

### 4.x Reproduce (operator — needs a Base archive RPC)

The F ceremony scripts (deploy/verify/Safe-tx/cutover) are already mainnet-fork-rehearsed and
hardened (sp1293-1295). The ONE new thing here is proving the DS-fix is live, which sp1435 added +
validated. Rehearse the whole thing on a fork:

```bash
cd contracts
# 1. forked node (NOT the live RPC — sp1293 C1: base-fork must be a LOCAL fork)
npx hardhat node --fork "$BASE_RPC_URL" &          # BASE_RPC_URL = a Base archive endpoint

# 2. deploy the fixed bundle (same tooling as F)
FTNS_TOKEN_ADDRESS=0x5276a3756C85f2E9e46f6D34386167a209aa16e5 \
FOUNDATION_RESERVE_WALLET=0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791 \
  npx hardhat run scripts/deploy-audit-bundle.js --network localhost
#   → writes deployments/audit-bundle-localhost-<ts>.json

# 3. cross-wire + consensus-invariance verification (existing F scripts)
AUDIT_BUNDLE_MANIFEST=deployments/audit-bundle-localhost-<ts>.json \
  npx hardhat run scripts/verify-audit-bundle-deployment.js --network localhost
AUDIT_BUNDLE_MANIFEST=deployments/audit-bundle-localhost-<ts>.json \
  npx hardhat run scripts/verify-attestation-commitment-deployment.js --network localhost

# 4. ★ NEW (sp1435) — prove the DOUBLE_SPEND fix is ACTIVE on the deployed registry
AUDIT_BUNDLE_MANIFEST=deployments/audit-bundle-localhost-<ts>.json \
  npx hardhat run scripts/verify-doublespend-fix-active.js --network localhost
#   exit 0 REQUIRES: self-reference reverts + copycat-after reverts + legit double-spend still succeeds.
#   As a negative control, run it against the OLD registry 0x12a01F6C… → it MUST exit 1.
```

`verify-doublespend-fix-active.js` is validated as a DISCRIMINATING test (sp1435): exit 0 on the
fixed contract, exit 1 on a de-guarded one (self-ref + copycat succeed). It commits throwaway
single-leaf batches (`tier_slash_rate_bps=0`, no stake touched) and uses `eth_call` (no state
written by the challenges) to check the three cases.

## 5. Mainnet deploy (GATED — deployer hot key)

Same as F §4.3. `deploy-audit-bundle.js` fail-closes on a mainnet chainId under a non-mainnet
`--network` name (sp1293 C1) and deploys the PRODUCTION Ed25519Verifier (do NOT set
`USE_MOCK_VERIFIER`). It writes `deployments/audit-bundle-base-<ts>.json` with every address + the
cross-wire tx hashes.

```bash
FTNS_TOKEN_ADDRESS=0x5276a3756C85f2E9e46f6D34386167a209aa16e5 \
FOUNDATION_RESERVE_WALLET=0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791 \
  npx hardhat run scripts/deploy-audit-bundle.js --network base
```

## 6. Verify BEFORE handoff (still deployer-owned)

```bash
M=deployments/audit-bundle-base-<ts>.json
AUDIT_BUNDLE_MANIFEST=$M npx hardhat run scripts/verify-audit-bundle-deployment.js --network base
AUDIT_BUNDLE_MANIFEST=$M npx hardhat run scripts/verify-attestation-commitment-deployment.js --network base
# ★ the fix is live on the NEW registry:
AUDIT_BUNDLE_MANIFEST=$M npx hardhat run scripts/verify-doublespend-fix-active.js --network base
```
Then Basescan-verify all four contracts. **Do NOT proceed to handoff unless
verify-doublespend-fix-active exits 0** — that is the whole point of this migration.

## 7. Foundation Safe ownership handoff (Ownable2Step — 2 signing steps)

Identical to F §4.5, reusing `build-f-activation-safe-txs.js` (offline Safe-batch generator) — it is
bundle-agnostic (reads the new manifest, emits the Foundation's importable `acceptOwnership` batch).
Deployer `transferOwnership(Safe)` on each of Registry/EscrowPool/StakeBond → Foundation 2-of-3
executes `acceptOwnership`. Confirm `owner()==Safe` + `pendingOwner()==0` on all three.

## 8. Cutover (GATED)

Identical to F §4.6 — re-point + SOAK on the new bundle BEFORE retiring the old, keep the old
EscrowPool UNPAUSED forever (recoverable-not-stranded residual; the old pool holds ~0.12 FTNS).

```bash
# drain gate on the OLD (now-to-be-retired) F registry — must be PAUSED + 0 PENDING first
OLD_REGISTRY_ADDRESS=0x12a01F6C487d765af389bC7D95D90b3136a391F2 \
  FROM_BLOCK=<F-registry deploy block> BASE_RPC_URL=<PAYG endpoint> \
  npx hardhat run scripts/verify-f-activation-cutover-readiness.js --network base
```
Then the pure-config re-point (sp1299 — no code change): set
`PRSM_SETTLEMENT_REGISTRY_ADDRESS` (+ `PRSM_ESCROW_POOL_ADDRESS`/`PRSM_STAKE_BOND_ADDRESS` for any
consumer that reads them) to the NEW bundle across settlement clients, and update
`prsm/config/networks.py` MAINNET + `operator-parallax.env` (retire the `0x12a01F6C…` addresses,
mirroring the pre-F retirement block already in networks.py). Foundation 2-of-3 `pause()` the OLD
registry once drained.

## 9. Assistant-done vs GATED

**Assistant-done (this prep, sp1435):**
- The fix itself: sp1429 (source + the two attack regression tests) — CI green.
- `scripts/verify-doublespend-fix-active.js` — new, validated discriminating on-chain proof of the fix.
- This runbook. All ceremony tooling (deploy/verify/Safe-tx/cutover) already exists + is F-mainnet-proven.

**GATED (operator / Foundation):**
- The fork rehearsal (§4) — needs a Base archive RPC.
- Mainnet deploy (§5), verify (§6), ownership handoff (§7, Foundation 2-of-3 signing), cutover (§8).
  Operator division per prior ceremonies: assistant read-only/offline; user signs + executes.

## 10. Abort criteria

- `verify-doublespend-fix-active.js` exits non-zero on the new registry → the deploy did not carry
  the fix; do NOT hand off. Investigate the compiled artifact.
- Any cross-wire mismatch, or `batchId`-invariance failing → abort (as F).
- Old-registry drain check not clean (PENDING > 0 or not paused) → do NOT retire the old registry;
  the new bundle can still be cut over to (it's additive), but keep the old as a hot fallback.
