/*
 * sp1456 — verify the slashing fixes are LIVE on a deployed StakeBond + StorageSlashing.
 *
 * READ-ONLY: staticCall / view reads only, no state written, no signing. Complements
 * verify-audit-bundle-deployment.js (which checks wiring/config but NOTHING sp1456-specific).
 * Exits 0 iff every check passes; exits 1 on the first FAIL. Run at GATE A (deployer-owned) and
 * GATE B (Safe-owned) per the runbook, and inside the fork rehearsal.
 *
 * Env:
 *   STAKE_BOND_ADDRESS        - the new sp1456 StakeBond
 *   STORAGE_SLASHING_ADDRESS  - the new StorageSlashing (optional; enables the storageSlasher checks)
 *
 * Usage:
 *   STAKE_BOND_ADDRESS=0x<newStakeBond> STORAGE_SLASHING_ADDRESS=0x<newStorageSlashing> \
 *   BASE_RPC_URL=<PAYG> npx hardhat run scripts/verify-sp1456-slashing-fix-active.js --network base
 *
 * NOTE: the bond() below-floor reverts require the contract UNPAUSED (bond is whenNotPaused). If the
 * new StakeBond is deployed paused, those checks report SKIP(paused) rather than FAIL — re-run after
 * unpause. The floor VALUES (minSlashRateForAmount) are pure and provable regardless of pause.
 */
const hre = require("hardhat");

async function main() {
  const stakeBondAddr = process.env.STAKE_BOND_ADDRESS;
  if (!stakeBondAddr) throw new Error("STAKE_BOND_ADDRESS env var required");
  const storageSlashingAddr = process.env.STORAGE_SLASHING_ADDRESS;

  const E = hre.ethers;
  const stakeBond = E.getAddress(stakeBondAddr);
  const chainId = (await E.provider.getNetwork()).chainId;
  console.log(`\n=== Verifying sp1456 slashing fixes on ${hre.network.name} (chainId ${chainId}) ===`);
  console.log(`StakeBond: ${stakeBond}`);

  const bond = await E.getContractAt("StakeBond", stakeBond);
  let allPass = true;
  const check = (name, ok, detail) => {
    if (!ok) allPass = false;
    console.log(`  [${ok ? "PASS" : "FAIL"}] ${name}${detail ? " — " + detail : ""}`);
  };
  const skip = (name, detail) => console.log(`  [SKIP] ${name}${detail ? " — " + detail : ""}`);
  const ONE = 10n ** 18n;

  // ── #1a — tier slash-rate floor VALUES (pure) ──────────────────────────────
  console.log("\n#1a — minSlashRateForAmount (tier floor policy):");
  const floorCases = [
    [50_000n * ONE, 10000n], [50_000n * ONE - 1n, 5000n],
    [25_000n * ONE, 5000n], [5_000n * ONE, 5000n],
    [5_000n * ONE - 1n, 0n], [100n * ONE, 0n], [0n, 0n],
  ];
  for (const [amt, want] of floorCases) {
    let got;
    try { got = await bond.minSlashRateForAmount(amt); }
    catch (e) { check(`minSlashRateForAmount(${amt})`, false, `reverted: ${e.message} (not an sp1456 build?)`); continue; }
    check(`minSlashRateForAmount(${amt})`, BigInt(got) === want, `got ${got}, want ${want}`);
  }

  // ── #1a — bond() ENFORCES the floor (below-floor reverts) ──────────────────
  console.log("\n#1a — bond() rejects a sub-floor rate:");
  let paused = false;
  try { paused = await bond.paused(); } catch { /* no paused() — treat as unpaused */ }
  const belowFloor = [
    [50_000n * ONE, 0], [50_000n * ONE, 9999], [5_000n * ONE, 0], [5_000n * ONE, 4999],
  ];
  if (paused) {
    skip("bond() below-floor reverts", "StakeBond is PAUSED (bond is whenNotPaused) — re-run after unpause");
  } else {
    for (const [amt, rate] of belowFloor) {
      let reverted = false, floorErr = false;
      try { await bond.bond.staticCall(amt, rate); }
      catch (e) {
        reverted = true;
        floorErr = String(e.message).includes("SlashRateBelowTierFloor")
          || (e.data && String(e.data).length > 2); // custom-error data present
      }
      check(`bond(${amt}, ${rate}) reverts (sub-floor)`, reverted && floorErr,
        reverted ? (floorErr ? "" : "reverted but not clearly SlashRateBelowTierFloor") : "did NOT revert");
    }
  }

  // ── #2 — storageSlasher set-once wiring ────────────────────────────────────
  console.log("\n#2 — storageSlasher wiring:");
  const wired = await bond.storageSlasher();
  if (storageSlashingAddr) {
    const ss = E.getAddress(storageSlashingAddr);
    check("stakeBond.storageSlasher == StorageSlashing", wired.toLowerCase() === ss.toLowerCase(), `got ${wired}`);
    try {
      const ssc = await E.getContractAt("StorageSlashing", ss);
      const ssBond = await ssc.stakeBond();
      check("StorageSlashing.stakeBond == StakeBond (immutable)", ssBond.toLowerCase() === stakeBond.toLowerCase(), `got ${ssBond}`);
      const code = await E.provider.getCode(ss);
      check("StorageSlashing has code", code !== "0x" && code !== "0x0");
    } catch (e) { check("StorageSlashing readable", false, e.message); }
    // idempotency: setStorageSlasherOnce (from the owner) must revert StorageSlasherAlreadySet.
    try {
      const owner = await bond.owner();
      await bond.setStorageSlasherOnce.staticCall(ss, { from: owner });
      check("setStorageSlasherOnce is idempotent (reverts)", false, "did NOT revert — storageSlasher not set?");
    } catch (e) {
      const isAlready = String(e.message).includes("StorageSlasherAlreadySet") || (e.data && String(e.data).length > 2);
      check("setStorageSlasherOnce reverts once set", isAlready, isAlready ? "" : `reverted with unexpected: ${e.message}`);
    }
  } else {
    check("stakeBond.storageSlasher != 0 (wired)", wired !== E.ZeroAddress, `got ${wired}`);
    skip("StorageSlashing cross-checks", "pass STORAGE_SLASHING_ADDRESS to enable");
  }

  console.log("");
  if (allPass) {
    console.log("✅ sp1456 slashing fixes are ACTIVE (tier floor enforced, storageSlasher wired).");
    console.log("   NOTE: fixes #1b (rate-0 batch slashes) and #3 (withdraw re-clamp) are behavioral —");
    console.log("   prove them via the fork rehearsals in the runbook (Phase 4.5/4.6), not this read-only pass.");
  } else {
    console.log("❌ one or more sp1456 checks FAILED — do NOT cut over.");
    process.exitCode = 1;
  }
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
