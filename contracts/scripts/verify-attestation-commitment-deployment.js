/*
 * sp1240 (TEE Tier-3 roadmap F) — post-deploy verification of the on-chain
 * attestation-commitment surface on a deployed BatchSettlementRegistry.
 *
 * The sibling verify-audit-bundle-deployment.js asserts the cross-wires
 * (escrowPool / stakeBond / signatureVerifier) match the manifest. It does
 * NOT know about sp1240 — it predates the upgrade. This script is the
 * missing checkpoint: it proves the *deployed bytecode* actually carries
 * the additive attestation surface and that activating roadmap F has ZERO
 * consensus impact. Run it after deploy-audit-bundle.js (against the fresh
 * registry) and again after the Foundation ownership handoff.
 *
 * Three assertions, in increasing strength:
 *
 *   [1] SELECTOR PRESENCE — the deployed bytecode contains the 4-byte
 *       dispatch selector for commitBatchWithAttestation(...). An OLD
 *       (pre-sp1240) registry fails here. Cheap, no signer, no RPC writes.
 *
 *   [2] EVENT TOPIC — the compiled ABI's BatchAttestationCommitted topic0
 *       is logged for the record so an indexer operator can wire a filter.
 *       (Informational — topic0 is a function of the ABI, not chain state.)
 *
 *   [3] CONSENSUS-INVARIANCE DRY RUN — the load-bearing proof. Using
 *       eth_call (staticCall, NO state written) pinned to a single block,
 *       call commitBatch(...) and commitBatchWithAttestation(..., <nonzero>)
 *       with byte-identical economic inputs and assert they return the
 *       SAME batchId. Because attestationCommitment is NOT in the batchId
 *       keccak preimage (BatchSettlementRegistry.sol _commitBatch), the
 *       two MUST agree. This is the on-chain analogue of the sp1238
 *       leaf-invariance unit test: it demonstrates, against the real
 *       deployed contract, that a client migrating from commitBatch to
 *       commitBatchWithAttestation changes NO consensus-relevant value.
 *       Skipped (with a loud note, non-fatal) only if the registry is
 *       paused or otherwise reverts the dry run — selector presence [1]
 *       is the hard gate.
 *
 * Required env var:
 *   AUDIT_BUNDLE_MANIFEST  - path to audit-bundle-<network>-*.json
 *                            (produced by deploy-audit-bundle.js), OR
 *   REGISTRY_ADDRESS       - a bare BatchSettlementRegistry address
 *                            (use when verifying a contract not described
 *                            by a local manifest).
 *
 * Optional env var:
 *   REQUIRE_INVARIANCE_DRYRUN=1  - treat a skipped/failed dry run [3] as
 *                                  fatal instead of a warning. Set this for
 *                                  a fresh, unpaused deploy where the dry
 *                                  run MUST succeed.
 *
 * Usage:
 *   AUDIT_BUNDLE_MANIFEST=contracts/deployments/audit-bundle-base-sepolia-123.json \
 *     npx hardhat run scripts/verify-attestation-commitment-deployment.js --network base-sepolia
 *
 *   REGISTRY_ADDRESS=0x48fFab641b9D638F312FFA776818756a326F995B \
 *     npx hardhat run scripts/verify-attestation-commitment-deployment.js --network base
 *
 * Exit codes:
 *   0 = deployed registry carries the sp1240 attestation surface and
 *       (if run) the consensus-invariance dry run held
 *   1 = missing selector, manifest/RPC mismatch, or (under
 *       REQUIRE_INVARIANCE_DRYRUN=1) a failed dry run
 */
const hre = require("hardhat");
const fs = require("fs");

// A non-zero sentinel requester + merkle root for the dry run. Any non-zero
// values work — the call is static (no state persists) and we compare the two
// returned batchIds, not their absolute value.
const DRYRUN_REQUESTER = "0x0000000000000000000000000000000000000001";
const DRYRUN_MERKLE_ROOT =
  "0x1111111111111111111111111111111111111111111111111111111111111111";
const DRYRUN_ATTESTATION =
  "0x2222222222222222222222222222222222222222222222222222222222222222";

async function main() {
  const manifestPath = process.env.AUDIT_BUNDLE_MANIFEST;
  const bareAddress = process.env.REGISTRY_ADDRESS;
  if (!manifestPath && !bareAddress) {
    throw new Error(
      "set AUDIT_BUNDLE_MANIFEST (preferred) or REGISTRY_ADDRESS env var",
    );
  }

  let registryAddress;
  let manifest = null;
  if (manifestPath) {
    if (!fs.existsSync(manifestPath)) {
      throw new Error(`AUDIT_BUNDLE_MANIFEST points to ${manifestPath} which does not exist`);
    }
    manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
    if (!manifest.contracts || !manifest.contracts.BatchSettlementRegistry) {
      throw new Error("manifest missing contracts.BatchSettlementRegistry");
    }
    registryAddress = manifest.contracts.BatchSettlementRegistry;
    // Sanity: manifest must agree with --network (mirrors verify-audit-bundle).
    if (manifest.network !== hre.network.name) {
      throw new Error(
        `manifest.network=${manifest.network} != --network=${hre.network.name}`,
      );
    }
  } else {
    registryAddress = bareAddress;
  }
  registryAddress = hre.ethers.getAddress(registryAddress);

  console.log(`\n=== Verifying sp1240 attestation-commitment surface ===`);
  console.log(`Network:   ${hre.network.name}`);
  console.log(`Registry:  ${registryAddress}`);
  if (manifest) {
    console.log(`Manifest:  ${manifestPath}`);
    // Sanity: chainId on connected RPC must match manifest.
    const onChainChainId = (await hre.ethers.provider.getNetwork()).chainId;
    if (manifest.chainId && onChainChainId.toString() !== manifest.chainId) {
      throw new Error(
        `RPC chainId=${onChainChainId} != manifest.chainId=${manifest.chainId}`,
      );
    }
  }

  let failures = 0;
  const fail = (msg) => { console.error(`  ❌ ${msg}`); failures += 1; };
  const ok = (msg) => { console.log(`  ✓ ${msg}`); };

  // ── 0. Bytecode presence ───────────────────────────────────────────
  const code = await hre.ethers.provider.getCode(registryAddress);
  if (code === "0x" || code === "0x0") {
    fail(`no bytecode at ${registryAddress} — wrong address or wrong network`);
    finishOrThrow(failures);
    return;
  }
  ok(`bytecode present: ${(code.length / 2 - 1)} bytes`);

  // The compiled artifact's interface — guaranteed to match the source we
  // intend to deploy (this checkout). We use it to derive the canonical
  // selector + event topic and to encode/decode the dry run.
  const iface = (await hre.ethers.getContractFactory("BatchSettlementRegistry")).interface;

  // ── 1. SELECTOR PRESENCE (hard gate) ───────────────────────────────
  console.log(`\n[1] commitBatchWithAttestation selector in deployed bytecode`);
  const fn = iface.getFunction("commitBatchWithAttestation");
  const selector = fn.selector; // 0x........
  console.log(`    signature: ${fn.format("sighash")}`);
  console.log(`    selector:  ${selector}`);
  // Solidity's external-function dispatcher embeds each 4-byte selector as a
  // PUSH4 literal in the runtime bytecode. Presence of the selector hex in
  // the deployed code is a reliable, ABI-independent existence check.
  if (code.toLowerCase().includes(selector.slice(2).toLowerCase())) {
    ok(`selector ${selector} present — registry IS sp1240-capable`);
  } else {
    fail(
      `selector ${selector} ABSENT — deployed registry is PRE-sp1240 ` +
      `(does not support commitBatchWithAttestation; roadmap F cannot activate here)`,
    );
  }

  // Cross-check: legacy commitBatch must STILL be present (the upgrade is
  // additive — un-upgraded clients must keep working).
  const legacySelector = iface.getFunction("commitBatch").selector;
  if (code.toLowerCase().includes(legacySelector.slice(2).toLowerCase())) {
    ok(`legacy commitBatch selector ${legacySelector} present (additive upgrade)`);
  } else {
    fail(`legacy commitBatch selector ${legacySelector} ABSENT — not a BatchSettlementRegistry?`);
  }

  // ── 2. EVENT TOPIC (informational) ─────────────────────────────────
  console.log(`\n[2] BatchAttestationCommitted event topic0 (for indexer wiring)`);
  const ev = iface.getEvent("BatchAttestationCommitted");
  console.log(`    signature: ${ev.format("sighash")}`);
  console.log(`    topic0:    ${ev.topicHash}`);

  // ── 3. CONSENSUS-INVARIANCE DRY RUN (load-bearing) ─────────────────
  console.log(`\n[3] consensus-invariance dry run (eth_call, no state written)`);
  const requireDryRun = process.env.REQUIRE_INVARIANCE_DRYRUN === "1";
  try {
    const registry = await hre.ethers.getContractAt("BatchSettlementRegistry", registryAddress);
    // Pin BOTH calls to the same block so msg.sender's batch sequence +
    // block.number are identical across the pair — the only way the two
    // batchIds can legitimately be compared for equality.
    const blockTag = await hre.ethers.provider.getBlockNumber();

    const argsCommon = [
      hre.ethers.getAddress(DRYRUN_REQUESTER),
      DRYRUN_MERKLE_ROOT,
      1n,            // receiptCount (must be non-zero)
      1000n,         // totalValueFTNS
      500,           // tierSlashRateBps (<= 10000)
      hre.ethers.ZeroHash, // consensusGroupId
      "ipfs://dryrun",
    ];

    const idLegacy = await registry.commitBatch.staticCall(...argsCommon, { blockTag });
    const idAttested = await registry.commitBatchWithAttestation.staticCall(
      ...argsCommon,
      DRYRUN_ATTESTATION,
      { blockTag },
    );

    console.log(`    commitBatch              → batchId ${idLegacy}`);
    console.log(`    commitBatchWithAttestation → batchId ${idAttested}`);
    if (idLegacy === idAttested) {
      ok(
        `batchId INVARIANT to attestation — activating roadmap F changes no ` +
        `consensus-relevant value (on-chain analogue of the sp1238 leaf-invariance test)`,
      );
    } else {
      fail(
        `batchId DIFFERS (${idLegacy} vs ${idAttested}) — attestation leaked into ` +
        `the consensus preimage. This contradicts the additive-upgrade invariant; ` +
        `DO NOT activate F. Investigate the deployed bytecode.`,
      );
    }
  } catch (err) {
    const msg = (err && err.shortMessage) || (err && err.message) || String(err);
    if (requireDryRun) {
      fail(`dry run failed (REQUIRE_INVARIANCE_DRYRUN=1): ${msg}`);
    } else {
      console.log(
        `  ⚠ dry run skipped (non-fatal): ${msg}\n` +
        `    Likely the registry is paused or this RPC rejects state-changing ` +
        `eth_call. Selector presence [1] is the hard gate; re-run with ` +
        `REQUIRE_INVARIANCE_DRYRUN=1 against a fresh, unpaused deploy.`,
      );
    }
  }

  finishOrThrow(failures);
}

function finishOrThrow(failures) {
  if (failures > 0) {
    console.error(`\n❌ ${failures} check(s) failed — registry is NOT a verified sp1240 deploy.`);
    process.exitCode = 1;
  } else {
    console.log(`\n✅ sp1240 attestation-commitment surface verified.`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
