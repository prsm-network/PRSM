/*
 * sp1456 — build the Safe Transaction Builder batch for acceptOwnership() on the FOUR new contracts:
 * BatchSettlementRegistry, EscrowPool, StakeBond (from the audit-bundle redeploy manifest) AND
 * StorageSlashing (from the storage-only manifest or STORAGE_SLASHING_ADDRESS). The existing
 * build-f-activation-safe-txs.js emits only the first three; this closes the StorageSlashing gap.
 *
 * PURE OFFLINE: no keys, no RPC, no signing. Emits a JSON to import into the Foundation Safe → Apps →
 * Transaction Builder. The 2-of-3 council then signs + executes all four acceptOwnership calls
 * atomically. acceptOwnership() is the Ownable2Step no-arg method; selector 0x79ba5097 (derived +
 * asserted below). Prerequisite: the deployer already ran transferOwnership(Safe) on each contract
 * (sets pendingOwner = Safe); this batch is the Safe accepting.
 *
 * Env:
 *   AUDIT_BUNDLE_MANIFEST     - path to deployments/audit-bundle-base-<ts>.json (new BSR/Escrow/StakeBond)
 *   STORAGE_SLASHING_ADDRESS  - the new StorageSlashing (OR set STORAGE_MANIFEST to its manifest)
 *   STORAGE_MANIFEST          - (alt) path to the sp1456-storage-slashing-only manifest
 *   SAFE_ADDRESS              - the Foundation Safe (default 0x91b0e6F8…5791)
 *   OUT                       - output path (default deployments/safe-batch-acceptOwnership-sp1456-<ts>.json)
 *
 * Usage:
 *   AUDIT_BUNDLE_MANIFEST=deployments/audit-bundle-base-<ts>.json \
 *   STORAGE_SLASHING_ADDRESS=0x<newStorageSlashing> \
 *   node scripts/build-sp1456-safe-acceptownership.js
 */
const fs = require("fs");
const path = require("path");
const { ethers } = require("ethers");

const ACCEPT_OWNERSHIP_SELECTOR = "0x79ba5097";
const DEFAULT_SAFE = "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791";

function main() {
  // Derive + assert the selector so an ethers change can't silently emit the wrong bytes.
  const iface = new ethers.Interface(["function acceptOwnership() external"]);
  const selector = iface.getFunction("acceptOwnership").selector;
  if (selector !== ACCEPT_OWNERSHIP_SELECTOR) {
    throw new Error(`computed acceptOwnership selector ${selector} != expected ${ACCEPT_OWNERSHIP_SELECTOR}`);
  }

  const manifestPath = process.env.AUDIT_BUNDLE_MANIFEST;
  if (!manifestPath) throw new Error("AUDIT_BUNDLE_MANIFEST env var required");
  if (!fs.existsSync(manifestPath)) throw new Error(`manifest ${manifestPath} not found`);
  const m = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  const chainId = String(m.chainId || "");
  if (!chainId) throw new Error("manifest missing chainId");

  // Resolve StorageSlashing from env or its own manifest.
  let storageSlashing = process.env.STORAGE_SLASHING_ADDRESS;
  if (!storageSlashing && process.env.STORAGE_MANIFEST) {
    const sm = JSON.parse(fs.readFileSync(process.env.STORAGE_MANIFEST, "utf8"));
    storageSlashing = sm.contracts && sm.contracts.StorageSlashing;
  }
  if (!storageSlashing) {
    throw new Error("set STORAGE_SLASHING_ADDRESS or STORAGE_MANIFEST (the new StorageSlashing address)");
  }

  const c = m.contracts || {};
  const targets = [
    ["BatchSettlementRegistry", c.BatchSettlementRegistry],
    ["EscrowPool", c.EscrowPool],
    ["StakeBond", c.StakeBond],
    ["StorageSlashing", storageSlashing],
  ];
  for (const [name, addr] of targets) {
    if (!addr) throw new Error(`manifest/env missing address for ${name}`);
  }

  const transactions = targets.map(([name, addr]) => ({
    to: ethers.getAddress(addr),
    value: "0",
    data: ACCEPT_OWNERSHIP_SELECTOR,
    contractMethod: { inputs: [], name: "acceptOwnership", payable: false },
    contractInputsValues: null,
    // annotation for the human reviewer in the Safe UI
    _contract: name,
  }));

  const safe = ethers.getAddress(process.env.SAFE_ADDRESS || DEFAULT_SAFE);
  const bundle = {
    version: "1.0",
    chainId,
    createdAt: Date.now(),
    meta: {
      name: "PRSM sp1456 slashing-fix acceptOwnership (BSR + EscrowPool + StakeBond + StorageSlashing)",
      description: "Foundation Safe accepts ownership of the 4 redeployed sp1456 contracts (Ownable2Step).",
      txBuilderVersion: "1.16.5",
      createdFromSafeAddress: safe,
    },
    transactions,
  };

  const outPath = process.env.OUT || path.join(
    __dirname, "..", "deployments", `safe-batch-acceptOwnership-sp1456-${Date.now()}.json`);
  fs.writeFileSync(outPath, JSON.stringify(bundle, null, 2));
  console.log(`Wrote Safe acceptOwnership batch (${transactions.length} txs) → ${outPath}`);
  for (const [name, addr] of targets) console.log(`  acceptOwnership() → ${name} ${ethers.getAddress(addr)}`);
  console.log(`\nImport into Safe ${safe} → Apps → Transaction Builder. 2-of-3 sign + execute.`);
  console.log(`Prerequisite: transferOwnership(Safe) already run on each (pendingOwner == Safe).`);
}

main();
