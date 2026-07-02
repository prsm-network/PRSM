/*
 * ContentAccessVerifier post-deploy verification (READ-ONLY — no signing, no gas).
 *
 * Re-confirms a deployed ContentAccessVerifier is wired correctly, before operators/consumers
 * point at it. Checks: bytecode present, ftns() + registry() match the expected addresses, a fresh
 * verifyPayment(anyone) is false, and totalClaimable() reads. Exit 0 = all good, 1 = mismatch.
 *
 * Required env:
 *   CAV_ADDRESS       — the deployed ContentAccessVerifier address to check.
 *   FTNS_ADDRESS      — the FTNS address it should be wired to.
 *   REGISTRY_ADDRESS  — the ProvenanceRegistry address it should be wired to.
 *
 * Usage (against Base mainnet, read-only — no PRIVATE_KEY needed):
 *   BASE_RPC_URL=https://... CAV_ADDRESS=0x... \
 *   FTNS_ADDRESS=0x5276a3756C85f2E9e46f6D34386167a209aa16e5 \
 *   REGISTRY_ADDRESS=0xe0cedDA354f99526c7fbb9b9651e12aDB2180dbf \
 *     npx hardhat run scripts/verify-content-access-verifier.js --network base
 */
const hre = require("hardhat");

function addr(name) {
  const v = (process.env[name] || "").trim();
  if (!hre.ethers.isAddress(v)) throw new Error(`${name} must be a valid address (got "${v}")`);
  return hre.ethers.getAddress(v);
}

async function main() {
  const cavAddr = addr("CAV_ADDRESS");
  const ftns = addr("FTNS_ADDRESS");
  const registry = addr("REGISTRY_ADDRESS");
  const chainId = (await hre.ethers.provider.getNetwork()).chainId;

  console.log(`\n=== Verifying ContentAccessVerifier @ ${cavAddr} (chain ${chainId}) ===`);

  const code = await hre.ethers.provider.getCode(cavAddr);
  if (code === "0x") { console.error(`❌ no bytecode at ${cavAddr}`); process.exitCode = 1; return; }

  const cav = await hre.ethers.getContractAt("ContentAccessVerifier", cavAddr);
  const wiredFtns = await cav.ftns();
  const wiredRegistry = await cav.registry();
  const paid = await cav.verifyPayment(cavAddr, "0x" + "00".repeat(32), 1n);
  const total = await cav.totalClaimable();

  const okFtns = wiredFtns.toLowerCase() === ftns.toLowerCase();
  const okReg = wiredRegistry.toLowerCase() === registry.toLowerCase();
  console.log(`  bytecode:              present ✓`);
  console.log(`  ftns():                ${wiredFtns} ${okFtns ? "✓" : "✗ (expected " + ftns + ")"}`);
  console.log(`  registry():            ${wiredRegistry} ${okReg ? "✓" : "✗ (expected " + registry + ")"}`);
  console.log(`  verifyPayment(random): ${paid} ${paid === false ? "✓" : "✗ (expected false)"}`);
  console.log(`  totalClaimable():      ${total}`);

  if (okFtns && okReg && paid === false) {
    console.log(`\n✅ ContentAccessVerifier is correctly wired.\n`);
  } else {
    console.error(`\n❌ ContentAccessVerifier wiring MISMATCH — do NOT use this address.\n`);
    process.exitCode = 1;
  }
}

main().catch((err) => { console.error(err); process.exitCode = 1; });
