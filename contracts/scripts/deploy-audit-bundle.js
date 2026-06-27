/*
 * Bundled deploy for the Phase 3.1 + 7 + 7.1 audit scope.
 *
 * Deploys in the order required by the audit-bundle coordinator
 * (docs/2026-04-21-audit-bundle-coordinator.md §7):
 *
 *   1. EscrowPool                   (Phase 3.1 substrate)
 *   2. BatchSettlementRegistry       (Phase 3.1 + 7 + 7.1 baked in)
 *   3. Signature verifier            (Mock on testnet; production Ed25519 on mainnet)
 *   4. StakeBond                    (Phase 7)
 *   5. Cross-wire (six setter calls)
 *
 * Required env vars:
 *   FTNS_TOKEN_ADDRESS             - ERC20 token for escrow + staking
 *   FOUNDATION_RESERVE_WALLET      - recipient for foundation slash share
 *   SIGNATURE_VERIFIER_ADDRESS     - (optional) pre-deployed verifier; script
 *                                     deploys MockSignatureVerifier if absent
 *
 * Optional env vars:
 *   CHALLENGE_WINDOW_SECONDS       - registry init (default 86400 = 1 day)
 *   UNBOND_DELAY_SECONDS           - stake-bond init (default 604800 = 7 days)
 *   AUTO_VERIFY                    - 1 to auto-verify on Basescan
 *
 * Usage:
 *   npx hardhat run scripts/deploy-audit-bundle.js --network hardhat
 *   npx hardhat run scripts/deploy-audit-bundle.js --network base-sepolia
 *   npx hardhat run scripts/deploy-audit-bundle.js --network base
 *
 * The script produces contracts/deployments/audit-bundle-<network>-<ts>.json
 * with every deployed address + cross-wire tx hash for mainnet-day handoff.
 */
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const network = hre.network.name;
  const ftnsAddress = process.env.FTNS_TOKEN_ADDRESS;
  const foundationWallet = process.env.FOUNDATION_RESERVE_WALLET;
  const preDeployedVerifier = process.env.SIGNATURE_VERIFIER_ADDRESS;
  const challengeWindow = BigInt(process.env.CHALLENGE_WINDOW_SECONDS || "86400");
  const unbondDelay = BigInt(process.env.UNBOND_DELAY_SECONDS || "604800");

  if (!ftnsAddress) throw new Error("FTNS_TOKEN_ADDRESS env var required");
  if (!foundationWallet) throw new Error("FOUNDATION_RESERVE_WALLET env var required");

  console.log(`\n=== Deploying Phase 3.1+7+7.1 audit bundle to ${network} ===`);

  // ── Preflight ───────────────────────────────────────────────────────
  const ftnsChecksum = hre.ethers.getAddress(ftnsAddress);
  const foundationChecksum = hre.ethers.getAddress(foundationWallet);
  if (ftnsChecksum === hre.ethers.ZeroAddress) throw new Error("FTNS_TOKEN_ADDRESS is zero");
  if (foundationChecksum === hre.ethers.ZeroAddress) throw new Error("FOUNDATION_RESERVE_WALLET is zero");

  const code = await hre.ethers.provider.getCode(ftnsChecksum);
  if (code === "0x" || code === "0x0") {
    throw new Error(`no contract at FTNS_TOKEN_ADDRESS ${ftnsChecksum} on ${network}`);
  }

  const [deployer] = await hre.ethers.getSigners();
  const balance = await hre.ethers.provider.getBalance(deployer.address);
  const chainId = (await hre.ethers.provider.getNetwork()).chainId;

  console.log(`Deployer:            ${deployer.address}`);
  console.log(`Deployer balance:    ${hre.ethers.formatEther(balance)} ETH`);
  console.log(`Chain id:            ${chainId}`);
  console.log(`FTNS token:          ${ftnsChecksum}`);
  console.log(`Foundation wallet:   ${foundationChecksum}`);
  console.log(`Challenge window:    ${challengeWindow}s`);
  console.log(`Unbond delay:        ${unbondDelay}s`);

  if (balance === 0n) throw new Error("Deployer has zero balance");

  // Mainnet production-verifier behavior: deploy the production Ed25519Verifier
  // by default. Operator can override with SIGNATURE_VERIFIER_ADDRESS (e.g., to
  // share a verifier across multiple registries). USE_MOCK_VERIFIER=1 is an
  // escape hatch only valid on non-mainnet networks.
  //
  // sp1293 (C1, F-migration review): key the mainnet guards off the CONNECTED
  // chainId, not the network NAME. A fork/alias network entry (e.g. `base-fork`
  // has chainId 8453 + a live RPC url) would otherwise run with isMainnet=false
  // against REAL mainnet, silently disabling the mock-verifier ban + the
  // foundation≠deployer check + the production-verifier default. Fail closed: a
  // real-mainnet chainId under a non-mainnet network name is a footgun → abort;
  // a mainnet name under a non-mainnet chainId is equally inconsistent → abort.
  const MAINNET_CHAIN_IDS = new Set([8453n, 1n, 137n]); // Base, Ethereum, Polygon
  const isMainnetByChainId = MAINNET_CHAIN_IDS.has(chainId);
  const isNamedMainnet = network === "base" || network === "mainnet";
  // localhost/hardhat are LOCAL nodes (incl. `hardhat node --fork`, which may report a
  // forked chainId like 8453) — they can never reach a live chain, so they're the
  // sanctioned fork-rehearsal path: allowed, and treated as mainnet so the rehearsal
  // exercises the production path. A mainnet chainId under any OTHER non-mainnet name
  // means a named alias is pointed at a live RPC (the base-fork footgun) → fail closed.
  const isLocalNode = network === "localhost" || network === "hardhat";
  if (isMainnetByChainId && !isNamedMainnet && !isLocalNode) {
    throw new Error(
      `Refusing to deploy: connected to mainnet chainId ${chainId} under non-mainnet ` +
      `network name "${network}" (fork/alias footgun). For a fork rehearsal use ` +
      `\`hardhat node --fork $BASE_RPC_URL\` + --network localhost, never a named ` +
      `network whose url points at a live RPC.`
    );
  }
  if (isNamedMainnet && !isMainnetByChainId) {
    throw new Error(
      `Network name "${network}" implies mainnet but the connected chainId is ${chainId}. ` +
      `Refusing to deploy under an inconsistent name↔chainId.`
    );
  }
  const isMainnet = isMainnetByChainId || isNamedMainnet;
  const useMockVerifier = process.env.USE_MOCK_VERIFIER === "1";
  if (isMainnet && useMockVerifier) {
    throw new Error(
      "USE_MOCK_VERIFIER=1 is forbidden on mainnet. MockSignatureVerifier is " +
      "test-only (verify() returns a flag set by anyone)."
    );
  }

  // Foundation wallet may equal deployer on testnet only. On mainnet the wallet
  // should be the 2-of-3 multi-sig. Belt-and-suspenders check for mainnet.
  if (isMainnet && foundationChecksum.toLowerCase() === deployer.address.toLowerCase()) {
    throw new Error(
      `FOUNDATION_RESERVE_WALLET must not equal deployer on mainnet. ` +
      `Use the Foundation 2-of-3 multi-sig per PRSM-GOV-1 §8.`
    );
  }

  const deployments = {};
  const txHashes = {};

  // ── 1. BatchSettlementRegistry ─────────────────────────────────────
  // L2 audit HIGH-6 (B-CROSS-1): EscrowPool.settlementRegistry is now
  // immutable (constructor-only), so Registry MUST deploy first and its
  // address is passed into EscrowPool's constructor.
  console.log("\n[1/5] Deploying BatchSettlementRegistry…");
  const Registry = await hre.ethers.getContractFactory("BatchSettlementRegistry");
  const registry = await Registry.deploy(deployer.address, challengeWindow);
  await registry.waitForDeployment();
  deployments.BatchSettlementRegistry = await registry.getAddress();
  console.log(`   BatchSettlementRegistry: ${deployments.BatchSettlementRegistry}`);

  // ── 2. EscrowPool ───────────────────────────────────────────────────
  console.log("\n[2/5] Deploying EscrowPool…");
  const EscrowPool = await hre.ethers.getContractFactory("EscrowPool");
  const escrow = await EscrowPool.deploy(
    deployer.address,
    ftnsChecksum,
    deployments.BatchSettlementRegistry,
  );
  await escrow.waitForDeployment();
  deployments.EscrowPool = await escrow.getAddress();
  console.log(`   EscrowPool:          ${deployments.EscrowPool}`);

  // ── 3. Signature verifier ──────────────────────────────────────────
  let verifierAddress;
  let verifierKind;
  if (preDeployedVerifier) {
    verifierAddress = hre.ethers.getAddress(preDeployedVerifier);
    verifierKind = "pre-deployed";
    console.log(`\n[3/5] Using pre-deployed verifier at ${verifierAddress}`);
  } else if (useMockVerifier) {
    console.log("\n[3/5] Deploying MockSignatureVerifier (TEST-ONLY)…");
    const Verifier = await hre.ethers.getContractFactory("MockSignatureVerifier");
    const verifier = await Verifier.deploy();
    await verifier.waitForDeployment();
    verifierAddress = await verifier.getAddress();
    verifierKind = "mock";
  } else {
    console.log("\n[3/5] Deploying production Ed25519Verifier…");
    const Verifier = await hre.ethers.getContractFactory("Ed25519Verifier");
    const verifier = await Verifier.deploy();
    await verifier.waitForDeployment();
    verifierAddress = await verifier.getAddress();
    verifierKind = "ed25519";
  }
  deployments.SignatureVerifier = verifierAddress;
  console.log(`   SignatureVerifier:   ${verifierAddress}  (${verifierKind})`);

  // ── 4. StakeBond ────────────────────────────────────────────────────
  // L2 audit HIGH-7 (B-CROSS-3): StakeBond.slasher is now immutable —
  // the registry address is passed into the constructor. Registry was
  // already deployed in step 1 (post-HIGH-6 ordering).
  console.log("\n[4/5] Deploying StakeBond…");
  const StakeBond = await hre.ethers.getContractFactory("StakeBond");
  const stakeBond = await StakeBond.deploy(
    deployer.address,
    ftnsChecksum,
    unbondDelay,
    deployments.BatchSettlementRegistry,
  );
  await stakeBond.waitForDeployment();
  deployments.StakeBond = await stakeBond.getAddress();
  console.log(`   StakeBond:           ${deployments.StakeBond}`);

  // ── 5. Cross-wire ───────────────────────────────────────────────────
  console.log("\n[5/5] Cross-wiring…");

  // EscrowPool.settlementRegistry was wired in EscrowPool's constructor
  // (immutable post-HIGH-6). StakeBond.slasher was wired in StakeBond's
  // constructor (immutable post-HIGH-7). Only the reverse pointers +
  // verifier + foundation-wallet wiring remain.
  let tx;

  tx = await registry.setEscrowPool(deployments.EscrowPool);
  await tx.wait();
  txHashes.registry_setEscrowPool = tx.hash;
  console.log(`   Registry.setEscrowPool → escrow (${tx.hash.slice(0, 10)}…)`);

  tx = await registry.setSignatureVerifier(verifierAddress);
  await tx.wait();
  txHashes.registry_setSignatureVerifier = tx.hash;
  console.log(`   Registry.setSignatureVerifier → verifier (${tx.hash.slice(0, 10)}…)`);

  tx = await registry.setStakeBond(deployments.StakeBond);
  await tx.wait();
  txHashes.registry_setStakeBond = tx.hash;
  console.log(`   Registry.setStakeBond → stakeBond (${tx.hash.slice(0, 10)}…)`);

  tx = await stakeBond.setFoundationReserveWallet(foundationChecksum);
  await tx.wait();
  txHashes.stakeBond_setFoundationReserveWallet = tx.hash;
  console.log(`   StakeBond.setFoundationReserveWallet → foundation (${tx.hash.slice(0, 10)}…)`);

  // ── Post-deploy invariant checks ───────────────────────────────────
  // Each getter is wrapped in `waitForAddressEquals` to absorb the
  // Base RPC propagation race observed in the 2026-05-07 mainnet
  // sprint: `tx.wait()` returns once the tx is mined, but the state
  // indexer can lag the receipt indexer by 1-3 seconds, returning
  // 0x0 / pre-tx values on an immediate-next read. The cross-wires
  // above (`registry.setEscrowPool`, etc.) all confirmed via
  // `tx.wait()`; re-reading the getter with retries is the right
  // way to confirm state without falsely failing a correct ceremony.
  console.log("\nPost-deploy invariant checks…");
  const { waitForAddressEquals } = require("./_lib/eventual-state");

  const checkAddr = async (label, read, expected) => {
    try {
      const got = await waitForAddressEquals(read, expected, {
        errorPrefix: `${label} mismatch (expected ${expected})`,
      });
      console.log(`   ✅ ${label}: ${got}`);
    } catch (err) {
      console.log(`   ❌ ${label}: ${err.message}`);
      throw err;
    }
  };

  await checkAddr("escrow.settlementRegistry", () => escrow.settlementRegistry(), deployments.BatchSettlementRegistry);
  await checkAddr("registry.escrowPool", () => registry.escrowPool(), deployments.EscrowPool);
  await checkAddr("registry.signatureVerifier", () => registry.signatureVerifier(), verifierAddress);
  await checkAddr("registry.stakeBond", () => registry.stakeBond(), deployments.StakeBond);
  await checkAddr("stakeBond.slasher", () => stakeBond.slasher(), deployments.BatchSettlementRegistry);
  await checkAddr("stakeBond.foundationReserveWallet", () => stakeBond.foundationReserveWallet(), foundationChecksum);

  // ── Manifest ────────────────────────────────────────────────────────
  const manifest = {
    bundle: "audit-bundle",
    phases: ["3.1", "7", "7.1"],
    network,
    chainId: chainId.toString(),
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    params: {
      challengeWindowSeconds: challengeWindow.toString(),
      unbondDelaySeconds: unbondDelay.toString(),
      verifierKind,
    },
    contracts: {
      ...deployments,
      FTNSToken: ftnsChecksum,
      FoundationReserveWallet: foundationChecksum,
    },
    crossWireTxHashes: txHashes,
  };

  const outDir = path.join(__dirname, "..", "deployments");
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(outDir, `audit-bundle-${network}-${Date.now()}.json`);
  fs.writeFileSync(outFile, JSON.stringify(manifest, null, 2));
  console.log(`\nManifest saved → ${outFile}`);

  // ── Optional Basescan verification ─────────────────────────────────
  const verifyEnabled = process.env.AUTO_VERIFY === "1";
  const isBase = network === "base" || network === "base-sepolia";
  if (verifyEnabled && isBase) {
    console.log("\nVerifying on Basescan…");
    const targets = [
      // sp1293 (H3, F-migration review): the 3rd ctor arg is the REGISTRY (immutable
      // settlementRegistry), not ZeroAddress — the constructor reverts on a zero
      // registry, so the wrong arg makes the Basescan source-verification silently fail
      // (the creation bytecode can never match). Must mirror the actual deploy (L137-141).
      { name: "EscrowPool", address: deployments.EscrowPool, args: [deployer.address, ftnsChecksum, deployments.BatchSettlementRegistry] },
      { name: "BatchSettlementRegistry", address: deployments.BatchSettlementRegistry, args: [deployer.address, challengeWindow] },
      { name: "StakeBond", address: deployments.StakeBond, args: [deployer.address, ftnsChecksum, unbondDelay] },
    ];
    if (verifierKind === "ed25519") {
      targets.push({ name: "Ed25519Verifier", address: verifierAddress, args: [] });
    } else if (verifierKind === "mock") {
      targets.push({ name: "MockSignatureVerifier", address: verifierAddress, args: [] });
    }
    for (const t of targets) {
      try {
        await hre.run("verify:verify", { address: t.address, constructorArguments: t.args });
        console.log(`   ${t.name} verified`);
      } catch (e) {
        console.warn(`   ${t.name} verify failed (non-fatal): ${e.message.split("\n")[0]}`);
      }
    }
  } else if (isBase) {
    console.log("\nSkipping verification (set AUTO_VERIFY=1 to enable).");
  }

  console.log("\n✅ Audit bundle deployment complete.");
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
