// Sprint 1481 — full-cycle rehearsal of the pool -> earner rail.
//
//   deploy -> fund -> publish epoch root -> every earner claims -> assert invariants
//
// Runs unchanged on two targets:
//
//   npx hardhat run scripts/rehearse-reward-pool.js
//       Self-contained on the in-process hardhat network: deploys a fresh FTNS
//       proxy and rehearses against it. Proves the whole cycle with NO external
//       dependency, so the rail is verifiable before any testnet key exists.
//
//   npx hardhat run scripts/rehearse-reward-pool.js --network base-sepolia
//       Live testnet. Requires a funded signer AND an FTNS token address:
//         PRIVATE_KEY=0x...            (NOTE the 0x prefix — hardhat.config's
//                                       pkAccounts() skips keyed networks without it)
//         REHEARSAL_FTNS_ADDRESS=0x... (an ERC-20 the signer holds a balance of;
//                                       omit to deploy a fresh test token)
//
// The epoch entitlements come from the SAME off-chain builder the production job
// uses (prsm/settlement/emission_epoch.py -> reward_epoch.py) via a committed
// fixture, so this rehearses the real pipeline rather than a JS re-implementation.
//
// REFUSES to run against mainnet: this deploys contracts and moves tokens, and the
// mainnet deployment is a governance ceremony, not a script.
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

const FTNS = (n) => hre.ethers.parseEther(String(n));

async function main() {
  const { ethers, upgrades } = hre;
  const network = hre.network.name;
  const chainId = (await ethers.provider.getNetwork()).chainId;

  if (chainId === 8453n || chainId === 1n) {
    throw new Error(
      `refusing to rehearse on chain ${chainId} — this deploys contracts and moves ` +
      `tokens. Mainnet deployment is a Foundation Safe ceremony, not a script.`
    );
  }

  const signers = await ethers.getSigners();
  if (signers.length === 0) {
    throw new Error(
      "no signer available. For a keyed network set PRIVATE_KEY=0x<64 hex> " +
      "(the 0x prefix is required — hardhat.config's pkAccounts() skips the " +
      "network without it)."
    );
  }
  const deployer = signers[0];
  const isLocal = chainId === 31337n;

  console.log(`\n=== OperatorRewardPool rehearsal — ${network} (chain ${chainId}) ===`);
  console.log(`Deployer: ${deployer.address}`);
  const nativeBal = await ethers.provider.getBalance(deployer.address);
  console.log(`Native balance: ${ethers.formatEther(nativeBal)}`);
  if (nativeBal === 0n) {
    throw new Error(
      `deployer ${deployer.address} has no native balance — fund it before ` +
      `rehearsing (Base Sepolia faucet: https://www.alchemy.com/faucets/base-sepolia)`
    );
  }

  // ── 1. Token ────────────────────────────────────────────────────────
  let ftns;
  const preset = process.env.REHEARSAL_FTNS_ADDRESS;
  if (preset) {
    console.log(`\n[1/6] Using existing FTNS at ${preset}`);
    ftns = await ethers.getContractAt("FTNSTokenSimple", preset);
  } else {
    console.log(`\n[1/6] Deploying a fresh FTNS (rehearsal token)…`);
    const Token = await ethers.getContractFactory("FTNSTokenSimple");
    ftns = await upgrades.deployProxy(
      Token, [deployer.address, deployer.address],
      { initializer: "initialize", kind: "uups" }
    );
    await ftns.waitForDeployment();
    console.log(`      FTNS: ${await ftns.getAddress()}`);
  }

  // ── 2. Pool ─────────────────────────────────────────────────────────
  console.log(`\n[2/6] Deploying OperatorRewardPool…`);
  const Pool = await ethers.getContractFactory("OperatorRewardPool");
  const pool = await Pool.deploy(
    await ftns.getAddress(),
    deployer.address,   // owner  — on mainnet this is the Foundation Safe
    deployer.address    // publisher — on mainnet this is the epoch-job key
  );
  await pool.waitForDeployment();
  const poolAddr = await pool.getAddress();
  console.log(`      Pool: ${poolAddr}`);
  console.log(`      MIN_CLAIM_WINDOW: ${await pool.MIN_CLAIM_WINDOW()}s`);

  // ── 3. Entitlements from the REAL off-chain builder ─────────────────
  const fixturePath = path.join(
    __dirname, "..", "test", "fixtures", "emission_epoch_from_batches.json");
  const plan = JSON.parse(fs.readFileSync(fixturePath, "utf8"));
  const pot = BigInt(plan.total_amount_wei);
  console.log(`\n[3/6] Epoch ${plan.epoch_id} from ${fixturePath.split("/").pop()}`);
  console.log(`      root:    ${plan.merkle_root}`);
  console.log(`      pot:     ${ethers.formatEther(pot)} FTNS`);
  console.log(`      earners: ${plan.entries.length}`);
  if (BigInt(plan.total_amount_wei) !== pot) throw new Error("fixture total mismatch");

  // On a live network the fixture's earners are hardhat's deterministic signers,
  // which nobody controls — that is fine: claims are permissionless and pay the
  // LEAF's account, so the deployer submits on their behalf and we verify the
  // recipients' balances rather than needing their keys.
  // ── 4. Fund ─────────────────────────────────────────────────────────
  console.log(`\n[4/6] Funding the pool with ${ethers.formatEther(pot)} FTNS…`);
  const bal = await ftns.balanceOf(deployer.address);
  if (bal < pot) {
    throw new Error(
      `deployer holds ${ethers.formatEther(bal)} FTNS but the epoch pot is ` +
      `${ethers.formatEther(pot)} — fund the signer or set REHEARSAL_FTNS_ADDRESS ` +
      `to a token it holds.`
    );
  }
  await (await ftns.transfer(poolAddr, pot)).wait();
  console.log(`      pool balance: ${ethers.formatEther(await ftns.balanceOf(poolAddr))} FTNS`);

  // Solvency gate proof: publishing MORE than is backed must revert.
  console.log(`\n      [check] over-publishing is refused…`);
  try {
    await pool.publishEpoch(999, plan.merkle_root, pot + 1n);
    throw new Error("SOLVENCY GATE FAILED — an unbacked epoch was published");
  } catch (e) {
    if (String(e.message).includes("SOLVENCY GATE FAILED")) throw e;
    console.log(`      ✓ refused (InsufficientBacking)`);
  }

  // ── 5. Publish ──────────────────────────────────────────────────────
  console.log(`\n[5/6] Publishing epoch ${plan.epoch_id}…`);
  const pubTx = await pool.publishEpoch(plan.epoch_id, plan.merkle_root, pot);
  const pubRcpt = await pubTx.wait();
  console.log(`      tx: ${pubRcpt.hash}`);
  const onChain = await pool.epochs(plan.epoch_id);
  if (onChain[0].toLowerCase() !== plan.merkle_root.toLowerCase()) {
    throw new Error("published root does not match the builder's root");
  }
  console.log(`      ✓ on-chain root matches the off-chain builder`);
  console.log(`      totalReserved: ${ethers.formatEther(await pool.totalReserved())} FTNS`);

  // ── 6. Claim every entitlement ──────────────────────────────────────
  console.log(`\n[6/6] Claiming…`);
  let paid = 0n;
  for (const e of plan.entries) {
    const amount = BigInt(e.amount_wei);
    const before = await ftns.balanceOf(e.account);
    if (!(await pool.isClaimable(plan.epoch_id, e.account, amount, e.proof))) {
      throw new Error(`isClaimable=false for ${e.account} — proof/root mismatch`);
    }
    const rcpt = await (await pool.claim(
      plan.epoch_id, e.account, amount, e.proof)).wait();
    const after = await ftns.balanceOf(e.account);
    if (after - before !== amount) {
      throw new Error(`payout mismatch for ${e.account}: ${after - before} != ${amount}`);
    }
    paid += amount;
    console.log(`      ✓ ${e.account} +${ethers.formatEther(amount)} FTNS (gas tx ${rcpt.hash.slice(0, 12)}…)`);

    // Double-claim must revert.
    try {
      await pool.claim(plan.epoch_id, e.account, amount, e.proof);
      throw new Error(`DOUBLE-CLAIM SUCCEEDED for ${e.account}`);
    } catch (err) {
      if (String(err.message).includes("DOUBLE-CLAIM SUCCEEDED")) throw err;
    }
  }

  // ── Invariants ──────────────────────────────────────────────────────
  const leftReserved = await pool.totalReserved();
  const leftBalance = await ftns.balanceOf(poolAddr);
  console.log(`\n=== Invariants ===`);
  console.log(`  distributed:    ${ethers.formatEther(paid)} FTNS`);
  console.log(`  pot:            ${ethers.formatEther(pot)} FTNS`);
  console.log(`  totalReserved:  ${ethers.formatEther(leftReserved)} (expect 0)`);
  console.log(`  pool balance:   ${ethers.formatEther(leftBalance)} (expect 0)`);
  if (paid !== pot) throw new Error(`distributed ${paid} != pot ${pot}`);
  if (leftReserved !== 0n) throw new Error(`reserved not drained: ${leftReserved}`);
  if (leftBalance !== 0n) throw new Error(`dust stranded in pool: ${leftBalance}`);
  console.log(`  ✓ pot distributed to the wei; nothing stranded; no double-claim`);

  if (!isLocal) {
    const outDir = path.join(__dirname, "..", "deployments");
    fs.mkdirSync(outDir, { recursive: true });
    const out = path.join(outDir, `operator-reward-pool-${network}-${Date.now()}.json`);
    fs.writeFileSync(out, JSON.stringify({
      network, chainId: chainId.toString(),
      contracts: { OperatorRewardPool: poolAddr, FTNSToken: await ftns.getAddress() },
      owner: deployer.address, rootPublisher: deployer.address,
      rehearsedEpoch: plan.epoch_id, potWei: pot.toString(),
      note: "REHEARSAL deployment (testnet). Mainnet deploy is a Safe ceremony.",
    }, null, 2));
    console.log(`\nDeployment record: ${out}`);
    console.log(`\nNext: set PRSM_REWARD_POOL_ADDRESS=${poolAddr} and try`);
    console.log(`      prsm node claim-emissions --dry-run --manifest <manifest.json>`);
  }
  console.log(`\n✅ Rehearsal complete on ${network}.\n`);
}

main().then(() => process.exit(0)).catch((e) => {
  console.error(`\n❌ Rehearsal failed: ${e.message}\n`);
  process.exit(1);
});
