/*
 * sp1291 — Safe{Wallet} Transaction Builder bundle for the TEE Tier-3 roadmap-F
 * activation handoff: the Foundation 2-of-3 multisig accepting Ownable2Step ownership
 * of the freshly-deployed audit bundle (BatchSettlementRegistry + EscrowPool + StakeBond).
 *
 * This is the turnkey form of the gated multisig step in
 * docs/2026-06-26-tee-tier3-f-activation-deploy-runbook.md §4.5. After the deployer hot
 * key runs deploy-audit-bundle.js (producing a manifest) + transfer-ownership.js (setting
 * each contract's pendingOwner = the Foundation Safe), this script reads that manifest and
 * emits a Safe Transaction Builder JSON the Foundation imports + signs 2-of-3 — one batch,
 * three acceptOwnership() calls. NO private keys, NO RPC, NO signing happen here: it is a
 * pure offline JSON generator (the assistant never signs the gated ceremony).
 *
 * acceptOwnership() is the Ownable2Step no-arg method; its selector is 0x79ba5097
 * (keccak256("acceptOwnership()")[:4]) — matches the proven 2026-05-07 mainnet batch
 * (deployments/safe-batch-acceptOwnership-mainnet-2026-05-07.json).
 *
 * Required args:
 *   --manifest      path to the audit-bundle-<network>-*.json from deploy-audit-bundle.js
 *   --safe-address  the Foundation 2-of-3 Safe (must equal each contract's pendingOwner)
 *
 * Optional args:
 *   --out           output path (default: deployments/safe-batch-acceptOwnership-f-activation-<network>-<ts>.json)
 *   --description   free-form Safe bundle description
 *
 * Usage:
 *   node scripts/build-f-activation-safe-txs.js \
 *     --manifest deployments/audit-bundle-base-1788000000000.json \
 *     --safe-address 0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791
 *
 * Exit codes: 0 ok; 1 bad/missing args; 2 bad address/manifest.
 */
const { ethers } = require("ethers");
const fs = require("fs");
const path = require("path");

const CHAIN_IDS = {
  base: 8453,
  "base-mainnet": 8453,
  mainnet: 1,
  "base-sepolia": 84532,
  sepolia: 11155111,
  localhost: 31337,
  hardhat: 31337,
};

// Ownable2Step acceptOwnership() selector. Derived (not hardcoded blindly) below and
// asserted to equal this, so a future ethers change can't silently emit a wrong selector.
const ACCEPT_OWNERSHIP_SELECTOR = "0x79ba5097";

// The Ownable2Step contracts in the audit bundle whose ownership the Safe must accept.
// The SignatureVerifier (Ed25519Verifier) is NOT Ownable, so it is intentionally absent
// — matching transfer-ownership.js and the 2026-05-07 mainnet batch.
const OWNABLE_BUNDLE_CONTRACTS = [
  "BatchSettlementRegistry",
  "EscrowPool",
  "StakeBond",
];

function parseArgs() {
  const args = {};
  const argv = process.argv.slice(2);
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith("--")) {
      const key = argv[i].slice(2);
      const val = argv[i + 1];
      if (!val || val.startsWith("--")) {
        console.error(`ERROR: --${key} requires a value`);
        process.exit(1);
      }
      args[key] = val;
      i++;
    }
  }
  return args;
}

function main() {
  const args = parseArgs();
  for (const r of ["manifest", "safe-address"]) {
    if (!args[r]) {
      console.error(`ERROR: missing required arg --${r}`);
      console.error("Run: node scripts/build-f-activation-safe-txs.js --manifest <path> --safe-address 0x...");
      process.exit(1);
    }
  }

  // Verify the acceptOwnership selector independently of any hardcoding.
  const iface = new ethers.Interface(["function acceptOwnership() external"]);
  const selector = iface.getFunction("acceptOwnership").selector;
  if (selector !== ACCEPT_OWNERSHIP_SELECTOR) {
    console.error(`ERROR: computed acceptOwnership selector ${selector} != expected ${ACCEPT_OWNERSHIP_SELECTOR}`);
    process.exit(2);
  }

  if (!fs.existsSync(args.manifest)) {
    console.error(`ERROR: --manifest ${args.manifest} does not exist`);
    process.exit(2);
  }
  const manifest = JSON.parse(fs.readFileSync(args.manifest, "utf8"));
  const contracts = manifest.contracts || {};
  const network = manifest.network || "unknown";
  const chainId = manifest.chainId || String(CHAIN_IDS[network] || "");
  if (!chainId) {
    console.error(`ERROR: cannot resolve chainId (manifest.network=${network}); pass a manifest from deploy-audit-bundle.js`);
    process.exit(2);
  }

  let safeAddress;
  try {
    safeAddress = ethers.getAddress(args["safe-address"]);
  } catch {
    console.error(`ERROR: --safe-address is not a valid address: ${args["safe-address"]}`);
    process.exit(2);
  }

  // Belt-and-suspenders: on Base mainnet the Safe should be the known Foundation 2-of-3.
  const KNOWN_FOUNDATION_SAFE = "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791";
  if ((chainId === "8453" || network === "base" || network === "base-mainnet")
      && safeAddress.toLowerCase() !== KNOWN_FOUNDATION_SAFE.toLowerCase()) {
    console.error(
      `WARNING: --safe-address ${safeAddress} != the known Foundation 2-of-3 ` +
      `${KNOWN_FOUNDATION_SAFE} on Base mainnet — re-verify before signing.`,
    );
  }

  const transactions = [];
  for (const name of OWNABLE_BUNDLE_CONTRACTS) {
    const addr = contracts[name];
    if (!addr) {
      console.error(`ERROR: manifest missing contracts.${name} — is this an audit-bundle manifest?`);
      process.exit(2);
    }
    let checksummed;
    try {
      checksummed = ethers.getAddress(addr);
    } catch {
      console.error(`ERROR: contracts.${name} is not a valid address: ${addr}`);
      process.exit(2);
    }
    transactions.push({
      to: checksummed,
      value: "0",
      data: ACCEPT_OWNERSHIP_SELECTOR,
      contractMethod: { inputs: [], name: "acceptOwnership", payable: false },
      contractInputsValues: null,
      _comment: name,
    });
  }

  const description =
    args.description ||
    `Foundation Safe (2-of-3) accepts Ownable2Step pendingOwner for the TEE Tier-3 roadmap-F ` +
    `audit-bundle re-deploy (BatchSettlementRegistry + EscrowPool + StakeBond). Run AFTER the ` +
    `deployer hot key called transferOwnership(safe) on each (transfer-ownership.js). Verify each ` +
    `contract's pendingOwner == ${safeAddress} before signing. Manifest: ${path.basename(args.manifest)}.`;

  const bundle = {
    version: "1.0",
    chainId: String(chainId),
    createdAt: Math.floor(Date.now() / 1000) * 1000,
    meta: {
      name: "PRSM TEE Tier-3 F-activation acceptOwnership",
      description,
      txBuilderVersion: "1.16.5",
      createdFromSafeAddress: safeAddress,
      createdFromOwnerAddress: "",
      checksum: "",
    },
    transactions,
  };

  const outPath = args.out || path.join(
    path.dirname(args.manifest),
    `safe-batch-acceptOwnership-f-activation-${network}-${Date.now()}.json`,
  );
  fs.writeFileSync(outPath, JSON.stringify(bundle, null, 2));

  console.log(`\nSafe acceptOwnership bundle written → ${outPath}`);
  console.log(`  Network:   ${network} (chainId ${chainId})`);
  console.log(`  Safe:      ${safeAddress}`);
  console.log(`  Selector:  ${ACCEPT_OWNERSHIP_SELECTOR} (acceptOwnership(), verified)`);
  console.log(`  Contracts (${transactions.length}):`);
  for (const t of transactions) {
    console.log(`    - ${t._comment}: ${t.to}`);
  }
  console.log("\nNext steps:");
  console.log("  1. Confirm each contract's pendingOwner() == the Safe (verify-audit-bundle-deployment.js).");
  console.log("  2. Import this JSON in the Safe UI → Transaction Builder.");
  console.log("  3. Foundation 2-of-3 signs + executes the batch.");
  console.log("  4. Re-run verify-audit-bundle-deployment.js with EXPECTED_OWNER=<safe> to confirm owner()==Safe.");
}

main();
