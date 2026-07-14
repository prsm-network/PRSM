/*
 * sp1456 — READ-ONLY mainnet pre-flight (runbook Phase 0.2-0.5). NO writes, NO signing, NO keys.
 * Pulls the LIVE on-chain facts needed to resolve the open questions + preserve the right params
 * before any redeploy. Run against Base mainnet with BASE_RPC_URL set (in contracts/.env):
 *
 *   npx hardhat run scripts/preflight-sp1456-cutover.js --network base
 *
 * Overridable addresses (defaults = the live June-2026 F bundle + phase7 storage):
 *   OLD_STAKE_BOND, OLD_BSR, OLD_ESCROW, OLD_STORAGE_SLASHING, FOUNDATION_SAFE
 */
const hre = require("hardhat");

const D = {
  OLD_STAKE_BOND: "0x21B5de0f65B9273A715C6a02b7085a8ABE8adA72",
  OLD_BSR: "0x12a01F6C487d765af389bC7D95D90b3136a391F2",
  OLD_ESCROW: "0x4e93a04b3A0C5063FE584980e6c2B1429495EEa1",
  OLD_STORAGE_SLASHING: "0x0e9cAfadCCCe0987C773B5FdFF295c2Aa6F03337",
  FOUNDATION_SAFE: "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791",
};
const A = (k) => hre.ethers.getAddress(process.env[k] || D[k]);

async function main() {
  const E = hre.ethers;
  const net = await E.provider.getNetwork();
  console.log(`\n=== sp1456 read-only pre-flight — ${hre.network.name} (chainId ${net.chainId}) ===`);
  if (net.chainId !== 8453n) console.log(`  ⚠️  NOT Base mainnet (8453) — reads reflect ${net.chainId}.`);
  const safe = A("FOUNDATION_SAFE");
  const eq = (a, b) => a && b && a.toLowerCase() === b.toLowerCase();

  const sb = await E.getContractAt("StakeBond", A("OLD_STAKE_BOND"));
  const bsr = await E.getContractAt("BatchSettlementRegistry", A("OLD_BSR"));

  console.log(`\n[0.2] LIVE PARAMETERS TO PRESERVE at redeploy:`);
  const unbond = await sb.unbondDelaySeconds();
  const window = await bsr.challengeWindowSeconds();
  console.log(`  StakeBond.unbondDelaySeconds     = ${unbond}  (${Number(unbond) / 86400} d)  → UNBOND_DELAY_SECONDS`);
  console.log(`  BSR.challengeWindowSeconds       = ${window}  (${Number(window) / 86400} d)  → CHALLENGE_WINDOW_SECONDS`);
  try { console.log(`  BSR.settlementLookbackWindowSeconds = ${await bsr.settlementLookbackWindowSeconds()}`); } catch {}
  console.log(`  invariant unbondDelay >= challengeWindow: ${unbond >= window ? "OK" : "❌ VIOLATED"}`);

  console.log(`\n[0.3] OWNERSHIP (expect Safe ${safe}):`);
  for (const [name, addr] of [["StakeBond", A("OLD_STAKE_BOND")], ["BSR", A("OLD_BSR")], ["EscrowPool", A("OLD_ESCROW")], ["StorageSlashing", A("OLD_STORAGE_SLASHING")]]) {
    try {
      const c = await E.getContractAt("StakeBond", addr); // any Ownable ABI works for owner()
      const owner = await c.owner();
      console.log(`  ${name.padEnd(16)} owner = ${owner}  ${eq(owner, safe) ? "✓ Safe" : "⚠️ NOT Safe"}`);
    } catch (e) { console.log(`  ${name.padEnd(16)} owner read failed: ${e.message.slice(0, 80)}`); }
  }

  console.log(`\n[0.3b] OLD BSR paused? (a paused old BSR blocks new commits during cutover):`);
  try { console.log(`  BSR.paused() = ${await bsr.paused()}`); } catch (e) { console.log(`  paused() read failed: ${e.message.slice(0, 60)}`); }

  console.log(`\n[0.5] STORAGE-SLASHING DRIFT (does the live StorageSlashing point at THIS StakeBond?):`);
  try {
    const ss = await E.getContractAt("StorageSlashing", A("OLD_STORAGE_SLASHING"));
    const ssBond = await ss.stakeBond();
    console.log(`  StorageSlashing.stakeBond = ${ssBond}`);
    console.log(`  == live settlement StakeBond ${A("OLD_STAKE_BOND")}?  ${eq(ssBond, A("OLD_STAKE_BOND")) ? "yes" : "NO → storage slashing already mis-wired/DEAD on mainnet"}`);
  } catch (e) { console.log(`  StorageSlashing read failed: ${e.message.slice(0, 80)}`); }

  console.log(`\n[sp1456 build check] Is the live StakeBond pre-sp1456 (expected)?`);
  try { const s = await sb.storageSlasher(); console.log(`  storageSlasher() = ${s}  (⚠️ live StakeBond ALREADY has sp1456 surface?)`); }
  catch { console.log(`  storageSlasher() reverts → live StakeBond is PRE-sp1456 (as expected; needs redeploy).`); }
  try { await sb.minSlashRateForAmount(0n); console.log(`  minSlashRateForAmount() exists → ⚠️ already sp1456?`); }
  catch { console.log(`  minSlashRateForAmount() reverts → confirms PRE-sp1456.`); }

  console.log(`\nNEXT (migration weight, needs the F-bundle BSR deploy block as FROM_BLOCK):`);
  console.log(`  OLD_REGISTRY_ADDRESS=${A("OLD_BSR")} FROM_BLOCK=<BSR deploy block> \\`);
  console.log(`    npx hardhat run scripts/verify-f-activation-cutover-readiness.js --network base`);
  console.log(`\nDone (read-only).`);
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
