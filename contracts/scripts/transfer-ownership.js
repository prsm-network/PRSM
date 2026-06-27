/*
 * Phase 1.3 + 7 + 7.1 + 8 + 7-storage — final ownership transfer.
 *
 * Reads the four deploy-script manifests + transfers ownership of every
 * Ownable contract from the deployer hot key to the Foundation 2-of-3
 * multi-sig in a single sweep. Runs LAST in the mainnet-day ceremony,
 * after every cross-wire setter has fired and every invariant has been
 * checked under deployer ownership.
 *
 * Why two-phase rather than constructor-set: setters like
 * Registry.setEscrowPool / Registry.setStakeBond /
 * Registry.setSignatureVerifier / setFoundationReserveWallet are
 * owner-only. If initialOwner were the multi-sig, every cross-wire tx
 * would need a 2-of-3 signature ceremony — the audit-bundle alone has
 * 4 cross-wires (post-HIGH-6/HIGH-7 immutable
 * EscrowPool.settlementRegistry + StakeBond.slasher) + owner-only
 * setters across 4 contracts. The two-phase model lets the deployer
 * hot key do the mechanical wiring under hardhat-tested invariants,
 * THEN hands ownership over to the multi-sig in a single commit-and-
 * verify step at the end.
 *
 * Required env vars:
 *   FOUNDATION_MULTISIG          - the 2-of-3 multi-sig that will
 *                                   own all contracts post-transfer
 *   AUDIT_BUNDLE_MANIFEST        - path to audit-bundle-<network>-*.json
 *                                   (produced by deploy-audit-bundle.js)
 *
 * Optional env vars (skip-if-absent):
 *   PHASE8_MANIFEST              - path to phase8-emission-<network>-*.json
 *   PHASE7_STORAGE_MANIFEST      - path to phase7-storage-<network>-*.json
 *
 * Usage:
 *   FOUNDATION_MULTISIG=0x... \
 *     AUDIT_BUNDLE_MANIFEST=contracts/deployments/audit-bundle-base-1234.json \
 *     PHASE8_MANIFEST=contracts/deployments/phase8-emission-base-1234.json \
 *     PHASE7_STORAGE_MANIFEST=contracts/deployments/phase7-storage-base-1234.json \
 *     npx hardhat run scripts/transfer-ownership.js --network base
 *
 * The script:
 *   1. Reads each manifest.
 *   2. For every Ownable contract address found, reads owner() —
 *      if owner() != deployer, abort (already transferred or
 *      unexpected state).
 *   3. Calls transferOwnership(FOUNDATION_MULTISIG) on each.
 *   4. Re-reads owner() and asserts it matches.
 *   5. Writes a transfer manifest with all (contract, oldOwner,
 *      newOwner, txHash) tuples.
 *
 * Idempotent: re-running after a successful transfer is a no-op
 * (each pre-check fails because the deployer no longer owns the
 * contract; script aborts cleanly with a clear message).
 *
 * Honest scope: this script does NOT touch FTNSToken (Phase 1.3,
 * has DEFAULT_ADMIN_ROLE rather than Ownable single-owner) or
 * Provenance/Royalty (Phase 1.3, AccessControl-based). Those role
 * grants/revocations are a separate ceremony, documented in the
 * Phase 1.3 deploy runbook.
 */
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

function readManifest(envVar, label) {
  const p = process.env[envVar];
  if (!p) {
    return null;
  }
  if (!fs.existsSync(p)) {
    throw new Error(`${envVar} points to ${p} which does not exist`);
  }
  const raw = fs.readFileSync(p, "utf8");
  const manifest = JSON.parse(raw);
  console.log(`Loaded ${label} manifest: ${p}`);
  return manifest;
}

async function transferOne(contractName, address, multisig, deployer) {
  const checksum = hre.ethers.getAddress(address);
  const code = await hre.ethers.provider.getCode(checksum);
  if (code === "0x" || code === "0x0") {
    throw new Error(`${contractName} at ${checksum}: no contract code`);
  }
  // L2 audit MEDIUM B-OWNABLE-1 fix: contracts now use Ownable2Step.
  // transferOwnership only sets pendingOwner; the multisig must call
  // acceptOwnership separately to complete the handoff. This script
  // performs only the deployer-side transferOwnership step and
  // verifies pendingOwner — the multisig's acceptOwnership ceremony
  // is a separate Foundation Safe operation documented in the runbook.
  const ownable = new hre.ethers.Contract(
    checksum,
    [
      "function owner() view returns (address)",
      "function pendingOwner() view returns (address)",
      "function transferOwnership(address newOwner)",
    ],
    deployer,
  );
  const currentOwner = await ownable.owner();
  if (currentOwner.toLowerCase() === multisig.toLowerCase()) {
    console.log(
      `   ⏭  ${contractName}: already owned by multi-sig (${currentOwner}); skipping`,
    );
    return null;
  }
  if (currentOwner.toLowerCase() !== deployer.address.toLowerCase()) {
    throw new Error(
      `${contractName} at ${checksum}: current owner ${currentOwner} ` +
      `is neither deployer (${deployer.address}) nor multi-sig ` +
      `(${multisig}) — aborting to avoid trampling unexpected state`,
    );
  }
  // If pendingOwner is already set to the target multisig, this
  // ceremony has been run before but the multisig hasn't yet
  // accepted. Skip the redundant on-chain call (idempotent re-runs).
  const currentPending = await ownable.pendingOwner();
  if (currentPending.toLowerCase() === multisig.toLowerCase()) {
    console.log(
      `   ⏭  ${contractName}: pendingOwner already set to multi-sig — ` +
      `awaiting Foundation Safe acceptOwnership; skipping`,
    );
    return {
      contractName,
      address: checksum,
      oldOwner: currentOwner,
      pendingOwner: multisig,
      txHash: null,
      blockNumber: null,
      note: "pending — multisig must acceptOwnership",
    };
  }
  console.log(
    `   → ${contractName} (${checksum}): pendingOwner → ${multisig}`,
  );
  const tx = await ownable.transferOwnership(multisig);
  const rcpt = await tx.wait();
  // Verify post-transfer: owner should still be deployer; pendingOwner
  // should now be the multisig.
  //
  // The state reads use the eventual-consistency wrapper to absorb
  // the Base RPC propagation race that bit every deploy + ownership
  // ceremony in the 2026-05-07 sprint: tx.wait() returns once the tx
  // is mined, but a follow-on getter sometimes returns the PRE-tx
  // value because the state indexer hasn't caught up. The receipt is
  // canonical (the tx is on-chain regardless of indexer state), so
  // re-reading the getter for ~6 seconds total handles the lag
  // without reverting genuinely-correct ceremonies.
  const { waitForAddressEquals } = require("./_lib/eventual-state");
  const newOwner = await waitForAddressEquals(
    () => ownable.owner(),
    deployer.address,
    {
      errorPrefix:
        `${contractName} at ${checksum}: post-transfer owner mismatch ` +
        `(expected deployer ${deployer.address}, Ownable2Step semantics)`,
    },
  );
  const pending = await waitForAddressEquals(
    () => ownable.pendingOwner(),
    multisig,
    {
      errorPrefix:
        `${contractName} at ${checksum}: post-transfer pendingOwner ` +
        `mismatch (expected ${multisig})`,
    },
  );
  console.log(`     ✅  tx ${tx.hash} (block ${rcpt.blockNumber})`);
  console.log(
    `     ⏳  Foundation Safe must call acceptOwnership() to complete handoff`,
  );
  return {
    contractName,
    address: checksum,
    oldOwner: currentOwner,
    pendingOwner: multisig,
    txHash: tx.hash,
    blockNumber: rcpt.blockNumber,
    note: "pending — multisig must acceptOwnership",
  };
}

async function main() {
  const network = hre.network.name;
  const multisig = process.env.FOUNDATION_MULTISIG;
  if (!multisig) throw new Error("FOUNDATION_MULTISIG env var required");
  const multisigChecksum = hre.ethers.getAddress(multisig);

  console.log(`\n=== Ownership transfer to Foundation multi-sig ===`);
  console.log(`Network:           ${network}`);
  console.log(`Foundation multi-sig: ${multisigChecksum}`);

  const [deployer] = await hre.ethers.getSigners();
  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log(`Deployer:          ${deployer.address}`);
  console.log(`Deployer balance:  ${hre.ethers.formatEther(balance)} ETH`);

  // sp1293 (C1, F-migration review): key mainnet guards off the CONNECTED chainId,
  // not the network NAME, so a fork/alias entry (e.g. base-fork: chainId 8453 + live
  // RPC) can't escape the multisig≠deployer + multisig-is-contract checks. Fail closed
  // on a name↔chainId mismatch.
  const chainId = (await hre.ethers.provider.getNetwork()).chainId;
  const MAINNET_CHAIN_IDS = new Set([8453n, 1n, 137n]);
  const isMainnetByChainId = MAINNET_CHAIN_IDS.has(chainId);
  const isNamedMainnet = network === "base" || network === "mainnet";
  const isLocalNode = network === "localhost" || network === "hardhat";
  if (isMainnetByChainId && !isNamedMainnet && !isLocalNode) {
    throw new Error(
      `Refusing to run: connected to mainnet chainId ${chainId} under non-mainnet ` +
      `network name "${network}" (fork/alias footgun). Use a local forked node + ` +
      `--network localhost for rehearsals.`,
    );
  }
  if (isNamedMainnet && !isMainnetByChainId) {
    throw new Error(`Network name "${network}" implies mainnet but chainId is ${chainId}.`);
  }
  const isMainnet = isMainnetByChainId || isNamedMainnet;
  if (
    isMainnet
    && multisigChecksum.toLowerCase() === deployer.address.toLowerCase()
  ) {
    throw new Error(
      "FOUNDATION_MULTISIG must not equal deployer on mainnet. " +
      "Use the Foundation 2-of-3 multi-sig.",
    );
  }
  if (multisigChecksum === hre.ethers.ZeroAddress) {
    throw new Error("FOUNDATION_MULTISIG is the zero address");
  }
  // Sanity: multi-sig must be a contract on mainnet (single-EOA
  // ownership defeats the safety property the multi-sig provides).
  // On testnet, allow EOA for rehearsal flexibility.
  if (isMainnet) {
    const mscode = await hre.ethers.provider.getCode(multisigChecksum);
    if (mscode === "0x" || mscode === "0x0") {
      throw new Error(
        `FOUNDATION_MULTISIG ${multisigChecksum} is an EOA on mainnet — ` +
        `expected a deployed multi-sig contract (Safe). Hot wallets ` +
        `cannot replace 2-of-3 multi-sig ownership for Foundation contracts.`,
      );
    }
  }

  // ── Load manifests ──────────────────────────────────────────────────
  const audit = readManifest("AUDIT_BUNDLE_MANIFEST", "audit bundle");
  if (!audit) throw new Error("AUDIT_BUNDLE_MANIFEST env var required");
  const phase8 = readManifest("PHASE8_MANIFEST", "Phase 8");
  const phase7s = readManifest("PHASE7_STORAGE_MANIFEST", "Phase 7-storage");

  // Sanity: manifests must agree on network.
  for (const [label, m] of [
    ["audit-bundle", audit],
    ["phase8", phase8],
    ["phase7-storage", phase7s],
  ]) {
    if (m && m.network !== network) {
      throw new Error(
        `${label} manifest network=${m.network} does not match ` +
        `--network=${network}`,
      );
    }
  }

  // ── Build the transfer plan ─────────────────────────────────────────
  // Each entry: [contractName, addressFromManifest].
  // Mock contracts (MockERC20, MockSignatureVerifier) are NOT in the
  // transfer plan — they're testnet-only and not Ownable / not
  // Foundation-managed.
  const plan = [];
  if (audit.contracts.EscrowPool) {
    plan.push(["EscrowPool", audit.contracts.EscrowPool]);
  }
  if (audit.contracts.BatchSettlementRegistry) {
    plan.push([
      "BatchSettlementRegistry",
      audit.contracts.BatchSettlementRegistry,
    ]);
  }
  if (audit.contracts.StakeBond) {
    plan.push(["StakeBond", audit.contracts.StakeBond]);
  }
  // SignatureVerifier (Ed25519Verifier on mainnet) is NOT Ownable —
  // it's stateless. Skip.

  if (phase8) {
    if (phase8.contracts.EmissionController) {
      plan.push(["EmissionController", phase8.contracts.EmissionController]);
    }
    if (phase8.contracts.CompensationDistributor) {
      plan.push([
        "CompensationDistributor",
        phase8.contracts.CompensationDistributor,
      ]);
    }
  }
  if (phase7s) {
    if (phase7s.contracts.StorageSlashing) {
      plan.push(["StorageSlashing", phase7s.contracts.StorageSlashing]);
    }
    if (phase7s.contracts.KeyDistribution) {
      plan.push(["KeyDistribution", phase7s.contracts.KeyDistribution]);
    }
  }

  console.log(`\nTransfer plan (${plan.length} contracts):`);
  for (const [name, addr] of plan) {
    console.log(`   ${name}: ${addr}`);
  }

  // ── Execute ─────────────────────────────────────────────────────────
  console.log(`\nExecuting transfers…`);
  const transferred = [];
  const skipped = [];
  for (const [name, addr] of plan) {
    const result = await transferOne(name, addr, multisigChecksum, deployer);
    if (result === null) {
      skipped.push({ contractName: name, address: addr });
    } else {
      transferred.push(result);
    }
  }

  // ── Manifest ────────────────────────────────────────────────────────
  const manifest = {
    bundle: "ownership-transfer",
    network,
    chainId: (await hre.ethers.provider.getNetwork()).chainId.toString(),
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    foundationMultisig: multisigChecksum,
    sourceManifests: {
      auditBundle: process.env.AUDIT_BUNDLE_MANIFEST || null,
      phase8: process.env.PHASE8_MANIFEST || null,
      phase7Storage: process.env.PHASE7_STORAGE_MANIFEST || null,
    },
    transferred,
    skipped,
  };

  const outDir = path.join(__dirname, "..", "deployments");
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(
    outDir,
    `ownership-transfer-${network}-${Date.now()}.json`,
  );
  fs.writeFileSync(outFile, JSON.stringify(manifest, null, 2));
  console.log(`\nManifest saved → ${outFile}`);

  console.log(
    `\n✅ Ownership transferred: ${transferred.length} ` +
    `contracts; skipped (already-multi-sig): ${skipped.length}`,
  );
  if (transferred.length === 0 && skipped.length === plan.length) {
    console.log(
      "(All contracts were already owned by the multi-sig — script " +
      "is idempotent on re-runs.)",
    );
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
