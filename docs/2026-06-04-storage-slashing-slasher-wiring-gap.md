# Storage-slashing wiring gap — StorageSlashing is not StakeBond's slasher

**Sprint 1000.** Found by the slashing-correctness adversarial hunt (workflow
wimmyp9rk, HIGH-confidence). This is an **on-chain wiring defect**: a contract change
+ a Foundation authorization are required to fix it. This doc records the defect, why
the autonomous response stopped at observability + a regression test, and the
**design decision the Foundation must make** to complete the fix.

## The defect

`StorageSlashing.submitProofFailure` and `slashForMissingHeartbeat` call
`stakeBond.slash(...)` **directly** (StorageSlashing.sol:207, 243). But:

- `StakeBond.slasher` is **immutable**, set once at construction, with the setter
  explicitly removed (StakeBond.sol:156, 252, 474-477 — HIGH-7 anti-drain rationale).
- `StakeBond.slash` hard-gates the caller: `if (msg.sender != slasher) revert
  CallerNotSlasher(...)` (StakeBond.sol:566-568).
- On the deployed Base-mainnet fleet, `StakeBond.slasher` == the
  **BatchSettlementRegistry** (`0x48fFab…`), asserted at deploy
  (deploy-audit-bundle.js:220). The single StakeBond is `0xD4C6584B…`; StorageSlashing
  (`0x0e9cAfad…`) points at that same StakeBond but is **not** and (no setter) can
  **never** be its slasher.

So **every** storage proof-failure / missing-heartbeat slash reverts with
`CallerNotSlasher`. The whole tx rolls back — including the `slashRecorded = true`
write (StorageSlashing.sol:205, 241) that happens before the slash call — so nothing
lands.

## Impact

**Slash-EVASION** (the dual of "never falsely slash an honest provider"): a
provably-dishonest storage provider (failed proof-of-retrievability, or a missed
heartbeat past the grace window) **cannot be slashed on mainnet**. The storage-tier
stake deterrent is entirely non-functional. No honest provider is harmed (this is not
a false-slash), but the economic deterrent backing storage reliability is absent.

Reachable today: the off-chain path is live (`prsm/storage/proof.py` escalates a
non-verified proof → `StorageSlashingClient.submit_proof_failure` builds + signs +
sends the real tx, which reverts).

## Why CI missed it

`contracts/test/StorageSlashing.test.js` injected `MockStakeBondSlasher`, whose
`slash()` has **no caller check** (MockStakeBondSlasher.sol:26-33). The integration
against the real immutable-slasher StakeBond was never exercised. sp1000 adds a
regression (`describe("slasher-wiring vs REAL StakeBond (sp1000)")`) that deploys the
real StakeBond and asserts the `CallerNotSlasher` revert — it will go green when the
fix below lands.

## Autonomous response shipped (sp1000)

This is a live-bond contract; redeploying or changing it is a governance ceremony, not
an autonomous edit. So sp1000 shipped only the autonomous, reversible parts:

1. **Observability** — `proof.py._escalate` previously swallowed the revert at a
   generic ERROR. It now emits a CRITICAL, actionable message when the failure is the
   slasher-wiring revert ("the storage-tier stake deterrent is NON-FUNCTIONAL until
   fixed on-chain"), so operators are not blind to a silently non-functional deterrent.
2. **Regression test** — pins the `CallerNotSlasher` revert against the real StakeBond
   (closing the CI gap) and will verify the eventual fix.
3. **This doc** — the design decision below.

## ★ DESIGN DECISION (Foundation) — two options to complete the fix

**Option A — StakeBond authorized-slasher allowlist (in-contract, requires StakeBond
redeploy + bond migration).** Add `mapping(address => bool) public authorizedSlasher`
+ owner-gated `addAuthorizedSlasher / removeAuthorizedSlasher`, and change the gate to
`if (msg.sender != slasher && !authorizedSlasher[msg.sender]) revert CallerNotSlasher(...)`.
Then the Foundation Safe calls `addAuthorizedSlasher(StorageSlashing)`. Preserves the
HIGH-7 immutable-primary-slasher invariant (only the Safe can add; only contract
slashers should be added). **Cost:** StakeBond holds live provider bonds, so this is a
redeploy + bond-migration ceremony — heavy.

**Option B — StorageSlashing-via-BSR adapter (no StakeBond change).** Route the storage
slash through the BatchSettlementRegistry (the existing immutable slasher) — e.g. a
BSR entry point that StorageSlashing calls, which then calls `stakeBond.slash`. Needs a
StorageSlashing (and possibly BSR) redeploy but **no change to the live-bond StakeBond**
— lighter on the most sensitive contract.

**Recommendation:** Option B avoids touching the live-bond StakeBond, but couples
StorageSlashing to the BSR's interface. Option A is cleaner long-term (a first-class
multi-slasher model) but migrates live bonds. The Foundation should choose based on
appetite for a StakeBond migration vs. a BSR-adapter coupling. Either is an irreversible
on-chain ceremony (deploy + Safe tx), gated like the other commissioning ceremonies.

Until one lands, the storage-tier slash is non-functional and the operator log will
emit the CRITICAL wiring-gap warning whenever a storage slash is attempted.
