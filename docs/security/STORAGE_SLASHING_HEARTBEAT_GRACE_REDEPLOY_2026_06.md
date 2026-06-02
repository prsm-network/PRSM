# StorageSlashing heartbeat-grace redeploy (sp940)

**Status:** source change landed + hardhat-verified; **mainnet redeploy is a
Foundation Safe multisig ceremony — NOT yet executed.** The live deployed
`StorageSlashing` contract is unchanged until this redeploy runs.

## The vulnerability (storage-slashing review, 2026-06-02)

`StorageSlashing.slashForMissingHeartbeat(provider)` is **permissionless** and
the caller is credited as challenger for a **70% bounty** (`StakeBond.slash`).
Pre-sp940 a provider became slashable the instant **one** grace window lapsed
(`block.timestamp > lastHeartbeat + heartbeatGraceSeconds`). So an HONEST
operator whose RPC/connectivity blips for slightly over the grace window can be
slashed by anyone, who profits (~70% of the bonded stake). Availability ≠ malice;
this is honest-operator griefing with a built-in economic incentive.

## The fix (this change)

`StorageSlashing.sol` now requires the provider to miss **`slashGraceMultiplier`
whole grace windows** (default **2**) before a heartbeat slash applies:

```
expiry = lastHeartbeat + heartbeatGraceSeconds * slashGraceMultiplier
```

- `slashGraceMultiplier` is governance-settable via `setSlashGraceMultiplier`
  (onlyOwner), bounded `[MIN_SLASH_GRACE_MULTIPLIER=1, MAX_SLASH_GRACE_MULTIPLIER=10]`,
  default 2. A transient one-window blip no longer slashes.
- Constructor signature is **unchanged** (the default is set in the body), so
  `scripts/deploy-phase7-storage.js` redeploys with no script edits.
- Verified: `contracts/test/StorageSlashing.test.js` — 27 passing (3 slash tests
  updated to the 2× threshold; 3 new: one-window buffer is NOT slashable, the
  setter retunes the threshold, bounds + onlyOwner enforced).

**Off-chain half (already shipped, sp939):** `HeartbeatScheduler` escalates to
CRITICAL + an `on_critical` alert hook after `CRITICAL_CONSECUTIVE_FAILURES`
consecutive missed ticks, so an operator gets a "slashing imminent" signal and
can intervene before the (now 2×) window elapses.

## Redeploy runbook (Foundation Safe ceremony)

1. **Audit + verify** the diff (financial contract): re-run the hardhat suite
   (`cd contracts && npx hardhat test test/StorageSlashing.test.js`), `slither`,
   and the formal-verification suite. **Add a halmos invariant** for
   `setSlashGraceMultiplier` to `symbolic-proofs/test/AdminBoundedSetters.t.sol`
   (the existing spec does not yet enumerate it) — prove it reverts outside
   `[MIN, MAX]`.
2. **Deploy** the new `StorageSlashing` via `scripts/deploy-phase7-storage.js`
   (constructor args unchanged: stakeBond, authorizedVerifier, grace, owner).
3. **Repoint authority:** `StakeBond` authorizes the slasher contract — point its
   authorized-slasher to the NEW `StorageSlashing` address (and de-authorize the
   old one) so only the patched contract can slash. This is the load-bearing
   migration step; sequence it so there is no window with two authorized slashers
   or none.
4. **Update config:** the operator-facing `deployment-config.json` /
   `prsm/economy/web3` address resolver + any watcher must point at the new
   address.
5. **Verify on-chain:** confirm `slashGraceMultiplier() == 2`, `heartbeatGrace`
   matches, owner is the Foundation Safe, and a missing-heartbeat slash reverts
   at 1× grace and succeeds only at 2×.

## Optional companion hardening (deferred, not in this change)

A **per-provider slash throttle** (`mapping(address => uint64) lastSlashAt` +
`minSlashInterval`, rejecting repeat slashes of the same provider within a
window) further bounds repeat-harvesting. Left out to keep this diff minimal +
auditable; the `slashRecorded` dedup already prevents double-slashing the same
heartbeat window, and the 2× multiplier closes the primary transient-blip
griefing. Add it in a follow-up redeploy if the threat model warrants.
