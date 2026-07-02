/*
 * ContentAccessVerifier deploy script (Tier B/C paid-decrypt, sp1353 + F1 redesign sp1357-1363).
 *
 * Deploys the production IRoyaltyPaymentVerifier that gates KeyDistribution.release: a consumer
 * pays the release fee via payForAccess, which credits the content's REGISTERED creator (looked up
 * on-chain via ProvenanceRegistry) and records the payment so the payment-gated key-serve endpoint
 * can confirm it. The contract is NOT Ownable and holds no funds at deploy — a plain deploy, no
 * ownership ceremony.
 *
 * Constructor: ContentAccessVerifier(address _ftns, address _registry)
 *   _ftns     — the FTNS ERC-20 the fee is paid in.
 *   _registry — the ProvenanceRegistry whose getCreatorAndRate(contentHash) yields the fee payee.
 *               USE THE SAME REGISTRY THE RoyaltyDistributor USES (Base mainnet: the V2 registry),
 *               so access fees credit the same authoritative creator.
 *
 * Required env:
 *   FTNS_ADDRESS      — the FTNS token address (explicit, no default: wrong token = stuck fees).
 *   REGISTRY_ADDRESS  — the ProvenanceRegistry address (explicit: wrong registry = wrong payee).
 * Optional env:
 *   AUTO_VERIFY=1     — verify on Basescan after deploy (needs ETHERSCAN_API_KEY — Etherscan's V2
 *                       unified API covers Base + Base Sepolia with one etherscan.io key).
 *
 * Canonical Base mainnet values (chain 8453), for reference — still pass them explicitly:
 *   FTNS_ADDRESS=0x5276a3756C85f2E9e46f6D34386167a209aa16e5
 *   REGISTRY_ADDRESS=0xe0cedDA354f99526c7fbb9b9651e12aDB2180dbf   (ProvenanceRegistry V2)
 *
 * Usage:
 *   # Dry-run (local hardhat) — validates the script + post-deploy invariants, no gas:
 *   FTNS_ADDRESS=0x... REGISTRY_ADDRESS=0x... \
 *     npx hardhat run scripts/deploy-content-access-verifier.js --network hardhat
 *
 *   # Base Sepolia (testnet) — needs a ProvenanceRegistry deployed there first
 *   # (deploy-provenance-registry-v2.js --network base-sepolia):
 *   PRIVATE_KEY=0x... BASE_SEPOLIA_RPC_URL=https://... \
 *   FTNS_ADDRESS=0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a REGISTRY_ADDRESS=0x<sepolia-registry> \
 *   AUTO_VERIFY=1 ETHERSCAN_API_KEY=... \
 *     npx hardhat run scripts/deploy-content-access-verifier.js --network base-sepolia
 *
 *   # Base mainnet:
 *   PRIVATE_KEY=0x... BASE_RPC_URL=https://... \
 *   FTNS_ADDRESS=0x5276a3756C85f2E9e46f6D34386167a209aa16e5 \
 *   REGISTRY_ADDRESS=0xe0cedDA354f99526c7fbb9b9651e12aDB2180dbf \
 *   AUTO_VERIFY=1 ETHERSCAN_API_KEY=... \
 *     npx hardhat run scripts/deploy-content-access-verifier.js --network base
 *
 * After deploy: put the address in prsm/deployments/contract_addresses.json under
 * <network>.content_access_verifier and set PRSM_CONTENT_ACCESS_VERIFIER for operators/consumers.
 */
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

function requireAddr(name) {
  const v = (process.env[name] || "").trim();
  if (!hre.ethers.isAddress(v)) {
    throw new Error(`${name} env var must be a valid address (got "${v}")`);
  }
  return hre.ethers.getAddress(v);
}

async function main() {
  const network = hre.network.name;
  const chainId = (await hre.ethers.provider.getNetwork()).chainId;

  // PRSM is Base-native. Block Ethereum mainnet (chain 1) — never an intended target.
  if (chainId === 1n) {
    throw new Error(
      "Deploy to Ethereum mainnet (chain 1) is BLOCKED — PRSM is Base-native. " +
      "Use --network base (chain 8453) or base-sepolia (84532).");
  }

  const ftns = requireAddr("FTNS_ADDRESS");
  const registry = requireAddr("REGISTRY_ADDRESS");

  const [deployer] = await hre.ethers.getSigners();
  const balance = await hre.ethers.provider.getBalance(deployer.address);

  console.log(`\n=== Deploying ContentAccessVerifier to ${network} (chain ${chainId}) ===`);
  console.log(`Deployer:          ${deployer.address}`);
  console.log(`Deployer balance:  ${hre.ethers.formatEther(balance)} ETH`);
  console.log(`FTNS:              ${ftns}`);
  console.log(`Registry:          ${registry}`);
  if (balance === 0n && network !== "hardhat") throw new Error("Deployer has zero balance");

  // ── Deploy ─────────────────────────────────────────────────────────
  console.log(`\n[1/1] Deploying ContentAccessVerifier…`);
  const Factory = await hre.ethers.getContractFactory("ContentAccessVerifier");
  const cav = await Factory.deploy(ftns, registry);
  await cav.waitForDeployment();
  const address = await cav.getAddress();
  console.log(`   ContentAccessVerifier: ${address}`);

  // ── Post-deploy invariant checks ───────────────────────────────────
  console.log(`\nPost-deploy invariant checks…`);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  let code = "0x";
  for (let i = 0; i < 15; i++) {
    code = await hre.ethers.provider.getCode(address);
    if (code !== "0x") { if (i > 0) console.log(`   read replica caught up after ${i + 1}s`); break; }
    await sleep(1000);
  }
  if (code === "0x") throw new Error(`getCode(${address}) empty after 15s — verify manually.`);

  const wiredFtns = await cav.ftns();
  const wiredRegistry = await cav.registry();
  if (wiredFtns.toLowerCase() !== ftns.toLowerCase())
    throw new Error(`ftns() ${wiredFtns} != ${ftns} — ABORT`);
  if (wiredRegistry.toLowerCase() !== registry.toLowerCase())
    throw new Error(`registry() ${wiredRegistry} != ${registry} — ABORT`);
  console.log(`   ftns():     ${wiredFtns} ✓`);
  console.log(`   registry(): ${wiredRegistry} ✓`);

  // A fresh verifier must report no payment for anyone + an empty pool.
  const dummy = "0x" + "00".repeat(32);
  const paidUnknown = await cav.verifyPayment(deployer.address, dummy, 1n);
  if (paidUnknown !== false) throw new Error("verifyPayment(unknown) returned true — ABORT");
  const total = await cav.totalClaimable();
  if (total !== 0n) throw new Error(`totalClaimable ${total} != 0 on a fresh deploy — ABORT`);
  console.log(`   verifyPayment(unknown): false ✓`);
  console.log(`   totalClaimable(): 0 ✓`);

  // ── Manifest ───────────────────────────────────────────────────────
  const manifest = {
    bundle: "tier-bc-content-access-verifier",
    network, chainId: chainId.toString(),
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    contracts: { ContentAccessVerifier: address },
    constructorArgs: { ftns, registry },
    postDeployNotes: [
      "ContentAccessVerifier is NOT Ownable and holds no funds at deploy — no ownership ceremony.",
      "Publishers name THIS address as the royalty verifier in KeyDistribution.deposit_key.",
      "Copy the address into prsm/deployments/contract_addresses.json under " +
      "<network>.content_access_verifier and set PRSM_CONTENT_ACCESS_VERIFIER for " +
      "operators (PRSM_PAID_KEY_SERVE nodes) and consumers.",
    ],
  };
  const outDir = path.join(__dirname, "..", "deployments");
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(outDir, `content-access-verifier-${network}-${Date.now()}.json`);
  fs.writeFileSync(outFile, JSON.stringify(manifest, null, 2));
  console.log(`\nManifest saved → ${outFile}`);

  // ── Basescan verification ──────────────────────────────────────────
  if (process.env.AUTO_VERIFY === "1" &&
      ["base", "base-sepolia", "sepolia"].includes(network)) {
    console.log(`\nVerifying on block explorer…`);
    try {
      await hre.run("verify:verify", { address, constructorArguments: [ftns, registry] });
      console.log(`   ContentAccessVerifier verified`);
    } catch (e) {
      console.warn(`   verify failed (non-fatal): ${e.message.split("\n")[0]}`);
    }
  }

  console.log(`\n${"=".repeat(60)}`);
  console.log(`✅ DEPLOY COMPLETE`);
  console.log(`${"=".repeat(60)}`);
  console.log(`Network:                ${network} (chain ${chainId})`);
  console.log(`ContentAccessVerifier:  ${address}`);
  console.log(`\nNext steps:`);
  console.log(`  1. Add to prsm/deployments/contract_addresses.json: ` +
              `${network}.content_access_verifier = "${address}"`);
  console.log(`  2. Set PRSM_CONTENT_ACCESS_VERIFIER=${address} for consumers + serve nodes.`);
  console.log(`  3. Serve operators also set PRSM_PAID_KEY_SERVE=1 + PRSM_PAID_KEY_STORE_FILE=<path>.`);
  console.log(`  4. Run scripts/verify-content-access-verifier.js against ${address} to re-confirm.`);
  console.log(`${"=".repeat(60)}\n`);
}

main().catch((err) => { console.error(err); process.exitCode = 1; });
