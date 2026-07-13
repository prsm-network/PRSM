/*
 * sp1456 — StorageSlashing-ONLY redeploy.
 *
 * Why this exists: StorageSlashing._stakeBond is IMMUTABLE, so the live StorageSlashing can never
 * point at the new (sp1456) StakeBond — it MUST be redeployed. But `deploy-phase7-storage.js` also
 * mints a fresh KeyDistribution, which on mainnet would ORPHAN the live encrypted-key/royalty state
 * (KeyDistribution 0x51AF73Aa… holds real deposits). This script deploys ONLY StorageSlashing,
 * pointing at the new StakeBond, and touches nothing else.
 *
 * Required env vars:
 *   STAKE_BOND_ADDRESS   - the NEW sp1456 StakeBond (from the audit-bundle redeploy manifest)
 *   AUTHORIZED_VERIFIER  - the storage proof verifier (see runbook open-question #5; the F manifest
 *                          used the Foundation Safe — confirm the intended prover before mainnet)
 * Optional:
 *   HEARTBEAT_GRACE_SECONDS - default 86400 (1 day); preserve the live value if it was retuned
 *
 * Usage:
 *   STAKE_BOND_ADDRESS=0x<newStakeBond> AUTHORIZED_VERIFIER=0x<prover> \
 *   BASE_RPC_URL=<PAYG> MAINNET_PRIVATE_KEY=0x<deployer> \
 *   npx hardhat run scripts/deploy-storage-slashing-only.js --network base
 *
 * NOTE: after this deploy, wire it with scripts/set-storage-slasher.js (StakeBond.setStorageSlasherOnce)
 * WHILE the deployer still owns the new StakeBond, then transfer StorageSlashing ownership to the Safe.
 */
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const network = hre.network.name;
  const stakeBondAddress = process.env.STAKE_BOND_ADDRESS;
  const verifierAddress = process.env.AUTHORIZED_VERIFIER;
  if (!stakeBondAddress) throw new Error("STAKE_BOND_ADDRESS env var required (the NEW sp1456 StakeBond)");
  if (!verifierAddress) throw new Error("AUTHORIZED_VERIFIER env var required (the storage proof verifier)");
  const heartbeatGrace = BigInt(process.env.HEARTBEAT_GRACE_SECONDS || "86400");

  const stakeBondChecksum = hre.ethers.getAddress(stakeBondAddress);
  const verifierChecksum = hre.ethers.getAddress(verifierAddress);

  const [deployer] = await hre.ethers.getSigners();
  const balance = await hre.ethers.provider.getBalance(deployer.address);
  const chainId = (await hre.ethers.provider.getNetwork()).chainId;

  console.log(`\n=== sp1456 StorageSlashing-only deploy to ${network} (chainId ${chainId}) ===`);
  console.log(`Deployer:            ${deployer.address}`);
  console.log(`Deployer balance:    ${hre.ethers.formatEther(balance)} ETH`);
  console.log(`New StakeBond:       ${stakeBondChecksum}`);
  console.log(`Authorized verifier: ${verifierChecksum}`);
  console.log(`Heartbeat grace:     ${heartbeatGrace}s`);
  if (balance === 0n) throw new Error("Deployer has zero balance");

  const stakeBondCode = await hre.ethers.provider.getCode(stakeBondChecksum);
  if (stakeBondCode === "0x" || stakeBondCode === "0x0") {
    throw new Error(`no contract at STAKE_BOND_ADDRESS ${stakeBondChecksum} on ${network}`);
  }
  // Guard: the target StakeBond must be an sp1456 build (exposes minSlashRateForAmount + storageSlasher),
  // else we would wire storage slashing to an OLD StakeBond that has no set-once slasher path.
  const stakeBond = await hre.ethers.getContractAt("StakeBond", stakeBondChecksum);
  try {
    await stakeBond.minSlashRateForAmount(0n); // sp1456-only view
    const existing = await stakeBond.storageSlasher();
    if (existing !== hre.ethers.ZeroAddress) {
      throw new Error(
        `StakeBond.storageSlasher is ALREADY set to ${existing} — it is set-once. Deploying a new ` +
        `StorageSlashing now would be un-wireable. Investigate before proceeding.`);
    }
  } catch (e) {
    if (String(e.message).includes("set-once")) throw e;
    throw new Error(
      `STAKE_BOND_ADDRESS ${stakeBondChecksum} does not look like an sp1456 StakeBond ` +
      `(minSlashRateForAmount reverted: ${e.message}). Point at the NEW StakeBond.`);
  }

  console.log("\nDeploying StorageSlashing…");
  const Slashing = await hre.ethers.getContractFactory("StorageSlashing");
  const slashing = await Slashing.deploy(
    stakeBondChecksum, verifierChecksum, heartbeatGrace, deployer.address);
  await slashing.waitForDeployment();
  const storageSlashingAddr = await slashing.getAddress();
  console.log(`   StorageSlashing: ${storageSlashingAddr}`);

  // Post-deploy invariant checks (immutable refs — verify NOW, they can't be fixed later).
  const wiredBond = await slashing.stakeBond();
  const wiredVerifier = await slashing.authorizedVerifier();
  const wiredGrace = await slashing.heartbeatGraceSeconds();
  console.log("\nPost-deploy invariant checks…");
  console.log(`   stakeBond:            ${wiredBond}`);
  console.log(`   authorizedVerifier:   ${wiredVerifier}`);
  console.log(`   heartbeatGraceSeconds:${wiredGrace}`);
  if (wiredBond.toLowerCase() !== stakeBondChecksum.toLowerCase()) {
    throw new Error(`stakeBond wiring mismatch: ${wiredBond} != ${stakeBondChecksum}`);
  }
  if (wiredVerifier.toLowerCase() !== verifierChecksum.toLowerCase()) {
    throw new Error(`authorizedVerifier wiring mismatch`);
  }

  const manifest = {
    bundle: "sp1456-storage-slashing-only",
    network,
    chainId: chainId.toString(),
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    params: {
      stakeBond: stakeBondChecksum,
      authorizedVerifier: verifierChecksum,
      heartbeatGraceSeconds: heartbeatGrace.toString(),
    },
    contracts: { StorageSlashing: storageSlashingAddr },
  };
  const outPath = path.join(
    __dirname, "..", "deployments",
    `sp1456-storage-slashing-only-${network}-${Date.now()}.json`);
  fs.writeFileSync(outPath, JSON.stringify(manifest, null, 2));
  console.log(`\nManifest: ${outPath}`);
  console.log("\nNEXT: run scripts/set-storage-slasher.js to wire StakeBond.setStorageSlasherOnce(" +
    `${storageSlashingAddr}) — BEFORE handing StakeBond ownership to the Safe.`);
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
