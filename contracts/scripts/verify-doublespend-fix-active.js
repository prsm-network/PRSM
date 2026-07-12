/*
 * sp1435 — post-deploy verification that the sp1429 _handleDoubleSpend fix is
 * LIVE on a deployed BatchSettlementRegistry.
 *
 * The DS-fix is BEHAVIORAL, not a new selector, so unlike
 * verify-attestation-commitment-deployment.js it cannot be proven by bytecode
 * inspection alone. This script EXERCISES the challenge path against the
 * deployed contract and asserts the two attack variants revert while the
 * legitimate double-spend still slashes — the on-chain analogue of the two
 * SECURITY tests in test/BatchSettlementChallenge.test.js.
 *
 * It commits a few THROWAWAY single-leaf batches (permissionless commitBatch,
 * tier_slash_rate_bps=0 so no stake is touched), then uses eth_call
 * (staticCall — NO state written by the challenges) to check:
 *
 *   [A] SELF-REFERENCE — challengeReceipt(id, leaf, [], DOUBLE_SPEND,
 *       encode(id, [])) MUST revert. On the OLD (buggy) registry it SUCCEEDS
 *       (the caller-verified proof re-verifies against the same root), letting
 *       anyone invalidate + slash any honest provider. FIXED ⇒ reverts.
 *
 *   [B] COPYCAT-AFTER — commit a copycat batch LATER, then challenge the
 *       earlier victim citing the later copycat. MUST revert (first-committer-
 *       wins). OLD ⇒ succeeds (slashes the honest first committer).
 *
 *   [C] LEGIT — challenge the LATER duplicate citing the EARLIER original.
 *       MUST succeed (the fix must not break real double-spend detection).
 *
 * A registry that fails [A] or [B] (attack does NOT revert) or [C] (legit
 * challenge reverts) exits 1. All correct ⇒ exit 0.
 *
 * Run on a mainnet FORK during the dress rehearsal, or against the freshly-
 * deployed mainnet registry BEFORE cutover (commits throwaway batches on the
 * not-yet-live registry). Do NOT run against the OLD live registry except to
 * demonstrate it FAILS (the discriminating negative control).
 *
 * Required env (one of):
 *   AUDIT_BUNDLE_MANIFEST  - path to audit-bundle-<network>-*.json
 *   REGISTRY_ADDRESS       - a bare BatchSettlementRegistry address
 *
 * Usage:
 *   REGISTRY_ADDRESS=0x... npx hardhat run scripts/verify-doublespend-fix-active.js --network hardhat
 */
const hre = require("hardhat");
const fs = require("fs");

const ONE_FTNS = hre.ethers.parseUnits("1", 18);
const DOUBLE_SPEND = 0;

function makeLeaf(overrides = {}) {
  return {
    jobIdHash: hre.ethers.keccak256(hre.ethers.toUtf8Bytes("verify-ds-job")),
    shardIndex: 0,
    providerIdHash: hre.ethers.keccak256(hre.ethers.toUtf8Bytes("verify-ds-provider")),
    providerPubkeyHash: hre.ethers.keccak256(hre.ethers.toUtf8Bytes("verify-ds-pubkey")),
    outputHash: hre.ethers.keccak256(hre.ethers.toUtf8Bytes("verify-ds-output")),
    executedAtUnix: Math.floor(Date.now() / 1000),
    valueFtns: ONE_FTNS,
    signatureHash: hre.ethers.keccak256(hre.ethers.toUtf8Bytes("verify-ds-sig")),
    signingMessageHash: hre.ethers.keccak256(hre.ethers.toUtf8Bytes("verify-ds-msg")),
    ...overrides,
  };
}

function hashLeaf(leaf) {
  return hre.ethers.keccak256(
    hre.ethers.AbiCoder.defaultAbiCoder().encode(
      ["tuple(bytes32,uint32,bytes32,bytes32,bytes32,uint64,uint128,bytes32,bytes32)"],
      [[
        leaf.jobIdHash, leaf.shardIndex, leaf.providerIdHash, leaf.providerPubkeyHash,
        leaf.outputHash, leaf.executedAtUnix, leaf.valueFtns, leaf.signatureHash,
        leaf.signingMessageHash,
      ]],
    ),
  );
}

async function commitBatch(registry, signer, requester, root, value, label) {
  const tx = await registry
    .connect(signer)
    .commitBatch(requester, root, 1, value, 0, hre.ethers.ZeroHash, label);
  const rc = await tx.wait();
  return rc.logs.find((l) => l.fragment && l.fragment.name === "BatchCommitted").args[0];
}

function aux(conflictId) {
  return hre.ethers.AbiCoder.defaultAbiCoder().encode(["bytes32", "bytes32[]"], [conflictId, []]);
}

async function main() {
  const manifestPath = process.env.AUDIT_BUNDLE_MANIFEST;
  const bareAddress = process.env.REGISTRY_ADDRESS;
  let registryAddress;
  if (process.env.SELF_DEPLOY === "1") {
    // Local / fork self-validation: deploy a fresh bare registry from the CURRENT source and
    // verify against it. challengeReceipt needs no escrow/stake wiring for the DOUBLE_SPEND path
    // (tier_slash_rate_bps=0 skips the slash), so a bare registry is sufficient to prove the fix.
    const [d] = await hre.ethers.getSigners();
    const Reg = await hre.ethers.getContractFactory("BatchSettlementRegistry");
    const reg = await Reg.deploy(d.address, 30n * 24n * 3600n); // 30d = MAX_CHALLENGE_WINDOW; challenge stays in-window
    await reg.waitForDeployment();
    registryAddress = await reg.getAddress();
    console.log(`SELF_DEPLOY: deployed a fresh BatchSettlementRegistry at ${registryAddress}`);
  } else if (manifestPath) {
    if (!fs.existsSync(manifestPath)) throw new Error(`manifest ${manifestPath} not found`);
    const m = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
    if (m.network && m.network !== hre.network.name) {
      throw new Error(`manifest.network=${m.network} != --network=${hre.network.name}`);
    }
    registryAddress = (m.contracts && m.contracts.BatchSettlementRegistry) || m.settlement_registry;
    if (!registryAddress) throw new Error("manifest missing BatchSettlementRegistry address");
  } else if (bareAddress) {
    registryAddress = hre.ethers.getAddress(bareAddress);
  } else {
    throw new Error("set AUDIT_BUNDLE_MANIFEST or REGISTRY_ADDRESS");
  }

  const [deployer, requesterSigner, attackerSigner] = await hre.ethers.getSigners();
  const requester = (requesterSigner || deployer).address;
  const attacker = attackerSigner || deployer;

  console.log(`\n=== Verifying sp1429 DOUBLE_SPEND fix is LIVE ===`);
  console.log(`Network:  ${hre.network.name} (chainId ${(await hre.ethers.provider.getNetwork()).chainId})`);
  console.log(`Registry: ${registryAddress}`);

  const registry = await hre.ethers.getContractAt("BatchSettlementRegistry", registryAddress);

  const leaf = makeLeaf();
  const root = hashLeaf(leaf);

  // Two batches with the SAME receipt: batch1 first (the "original"), batch2 later (the copycat).
  const id1 = await commitBatch(registry, deployer, requester, root, leaf.valueFtns, "verify-ds-1");
  const id2 = await commitBatch(registry, attacker, requester, root, leaf.valueFtns, "verify-ds-2");
  console.log(`Committed throwaway batches:\n  batch1 (earlier) ${id1}\n  batch2 (later)   ${id2}`);

  let failures = 0;
  const check = (name, ok, detail) => {
    console.log(`  [${ok ? "PASS" : "FAIL"}] ${name}${detail ? " — " + detail : ""}`);
    if (!ok) failures++;
  };

  // [A] self-reference MUST revert.
  let reverted = false;
  try {
    await registry.challengeReceipt.staticCall(id1, leaf, [], DOUBLE_SPEND, aux(id1));
  } catch (e) { reverted = true; }
  check("[A] self-referential DOUBLE_SPEND reverts", reverted,
    reverted ? "attack blocked" : "SELF-REFERENCE SUCCEEDS — fix NOT active (buggy registry)");

  // [B] copycat-after: challenge the EARLIER batch1 citing the LATER batch2 MUST revert.
  reverted = false;
  try {
    await registry.challengeReceipt.staticCall(id1, leaf, [], DOUBLE_SPEND, aux(id2));
  } catch (e) { reverted = true; }
  check("[B] copycat-after (earlier cites later) reverts", reverted,
    reverted ? "honest first-committer protected" : "COPYCAT SUCCEEDS — fix NOT active");

  // [C] legit: challenge the LATER batch2 citing the EARLIER batch1 MUST succeed.
  let succeeded = false;
  try {
    await registry.challengeReceipt.staticCall(id2, leaf, [], DOUBLE_SPEND, aux(id1));
    succeeded = true;
  } catch (e) { succeeded = false; }
  check("[C] legit later-cites-earlier double-spend still succeeds", succeeded,
    succeeded ? "real double-spend detection intact" : "LEGIT CHALLENGE REVERTS — over-correction");

  console.log("");
  if (failures === 0) {
    console.log("✅ sp1429 DOUBLE_SPEND fix is ACTIVE on this registry (self-ref + copycat blocked, legit intact).");
    process.exit(0);
  } else {
    console.error(`❌ ${failures} check(s) failed — this registry does NOT carry the sp1429 fix. DO NOT cut over to it.`);
    process.exit(1);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
