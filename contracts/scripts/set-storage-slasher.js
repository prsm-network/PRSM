/*
 * sp1456 — wire StakeBond.setStorageSlasherOnce(StorageSlashing).
 *
 * ⚠️ HIGHEST-RISK STEP IN THE CEREMONY. `storageSlasher` is SET-ONCE, forever. A wrong target
 * permanently bricks storage-fault slashing (every StorageSlashing.slash reverts CallerNotSlasher),
 * fixable ONLY by redeploying StakeBond (cascading BSR/EscrowPool/StorageSlashing). Triple-check.
 *
 * This script NEVER blind-sends: it first reads the current storageSlasher + owner and:
 *   - if storageSlasher already == the target  → prints SUCCESS (idempotent no-op), exits 0.
 *   - if storageSlasher is a DIFFERENT non-zero → ABORTS (already set, would revert; investigate).
 *   - if the connected signer IS the owner      → sends setStorageSlasherOnce(target), then verifies.
 *   - if the signer is NOT the owner (e.g. the Safe already owns it) → prints the exact calldata to
 *     drop into the Safe Transaction Builder and exits WITHOUT sending.
 *
 * Required env:
 *   STAKE_BOND_ADDRESS        - the NEW sp1456 StakeBond
 *   STORAGE_SLASHING_ADDRESS  - the NEW StorageSlashing (from deploy-storage-slashing-only.js)
 *
 * Usage (deployer still owns StakeBond — the intended order):
 *   STAKE_BOND_ADDRESS=0x<newStakeBond> STORAGE_SLASHING_ADDRESS=0x<newStorageSlashing> \
 *   BASE_RPC_URL=<PAYG> MAINNET_PRIVATE_KEY=0x<deployer> \
 *   npx hardhat run scripts/set-storage-slasher.js --network base
 */
const hre = require("hardhat");

async function main() {
  const stakeBondAddr = process.env.STAKE_BOND_ADDRESS;
  const storageSlashingAddr = process.env.STORAGE_SLASHING_ADDRESS;
  if (!stakeBondAddr) throw new Error("STAKE_BOND_ADDRESS env var required");
  if (!storageSlashingAddr) throw new Error("STORAGE_SLASHING_ADDRESS env var required");

  const stakeBond = hre.ethers.getAddress(stakeBondAddr);
  const target = hre.ethers.getAddress(storageSlashingAddr);
  const [signer] = await hre.ethers.getSigners();
  const chainId = (await hre.ethers.provider.getNetwork()).chainId;

  console.log(`\n=== sp1456 setStorageSlasherOnce on ${hre.network.name} (chainId ${chainId}) ===`);
  console.log(`StakeBond:        ${stakeBond}`);
  console.log(`Target slasher:   ${target}`);
  console.log(`Connected signer: ${signer.address}`);

  // Sanity: target must actually be a StorageSlashing that references THIS StakeBond (else you would
  // set-once a bad address and permanently brick the path).
  const ss = await hre.ethers.getContractAt("StorageSlashing", target);
  const ssBond = await ss.stakeBond();
  if (ssBond.toLowerCase() !== stakeBond.toLowerCase()) {
    throw new Error(
      `ABORT: StorageSlashing(${target}).stakeBond() == ${ssBond}, NOT ${stakeBond}. The set-once ` +
      `wiring must point at a StorageSlashing whose immutable stakeBond IS this StakeBond.`);
  }

  const bond = await hre.ethers.getContractAt("StakeBond", stakeBond);
  const current = await bond.storageSlasher();
  const owner = await bond.owner();
  console.log(`Current storageSlasher: ${current}`);
  console.log(`StakeBond owner:        ${owner}`);

  if (current !== hre.ethers.ZeroAddress) {
    if (current.toLowerCase() === target.toLowerCase()) {
      console.log(`\n✅ storageSlasher ALREADY == target — idempotent no-op, nothing to do.`);
      return;
    }
    throw new Error(
      `ABORT: storageSlasher is already set to ${current} (set-once). It cannot be changed to ` +
      `${target}. If ${current} is wrong, StakeBond must be redeployed.`);
  }

  // Build calldata once (used either to send or to hand to the Safe).
  const iface = new hre.ethers.Interface(["function setStorageSlasherOnce(address) external"]);
  const data = iface.encodeFunctionData("setStorageSlasherOnce", [target]);

  if (owner.toLowerCase() !== signer.address.toLowerCase()) {
    console.log(`\nThe connected signer is NOT the StakeBond owner. setStorageSlasherOnce is onlyOwner.`);
    console.log(`Execute it from the owner (${owner}) — e.g. the Safe Transaction Builder — with:`);
    console.log(`   to:    ${stakeBond}`);
    console.log(`   value: 0`);
    console.log(`   data:  ${data}`);
    console.log(`\n(No transaction sent.) After the owner executes it, re-run this script to verify.`);
    return;
  }

  console.log(`\nSigner IS the owner — sending setStorageSlasherOnce(${target})…`);
  const tx = await bond.setStorageSlasherOnce(target);
  console.log(`   tx: ${tx.hash}`);
  await tx.wait();

  const after = await bond.storageSlasher();
  if (after.toLowerCase() !== target.toLowerCase()) {
    throw new Error(`POST-CHECK FAILED: storageSlasher == ${after}, expected ${target}`);
  }
  console.log(`\n✅ storageSlasher == ${after}. Storage-fault slashing is now authorized.`);
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
