/*
 * Sprint 976/977 — CreatorStakeRegistry deploy script (Vision §14 anti-spam
 * creator stake, on-chain teeth).
 *
 * Deploys CreatorStakeRegistry. Sepolia is the intended REHEARSAL target — it
 * exercises the full stake / requestUnbond / withdraw / slash / drain round-trip
 * against a real chain WITHOUT committing to the mainnet deploy. The mainnet
 * (Base, chain 8453) deploy + the 2-of-3 Foundation-Safe acceptOwnership is the
 * irreversible commissioning ceremony and is NOT performed by this script alone
 * — see docs/<date>-creator-stake-registry-sepolia-runbook.md.
 *
 * Constructor args (env):
 *   CREATOR_STAKE_OWNER        - initialOwner. For a rehearsal this is the
 *                                deployer; for mainnet, deploy as deployer then
 *                                transferOwnership() to the Foundation Safe,
 *                                which calls acceptOwnership() (Ownable2Step).
 *   FTNS_TOKEN_ADDRESS         - the FTNS ERC-20 (Base mainnet canonical, or a
 *                                MockERC20 on a testnet).
 *   CREATOR_STAKE_UNBOND_DELAY - seconds (1d..30d); defaults to 7 days.
 *   CREATOR_STAKE_SLASHER      - immutable slasher (the governance/Foundation
 *                                slash authority). Non-zero REQUIRED.
 *
 * Optional:
 *   AUTO_VERIFY=1  - verify on Basescan/Etherscan after deploy.
 *
 * Usage:
 *   # Dry-run on local Hardhat (validates script + post-deploy invariants, no gas):
 *   FTNS_TOKEN_ADDRESS=0x... CREATOR_STAKE_SLASHER=0x... \
 *     npx hardhat run scripts/deploy-creator-stake-registry.js --network hardhat
 *
 *   # Sepolia rehearsal:
 *   PRIVATE_KEY=0x... SEPOLIA_RPC_URL=https://... \
 *   FTNS_TOKEN_ADDRESS=0x... CREATOR_STAKE_SLASHER=0x... \
 *   AUTO_VERIFY=1 ETHERSCAN_API_KEY=... \
 *     npx hardhat run scripts/deploy-creator-stake-registry.js --network sepolia
 *
 * After a successful mainnet deploy, copy the printed address into
 * prsm/deployments/contract_addresses.json under <network>.creator_stake_registry
 * and set CREATOR_STAKE_REGISTRY_ADDRESS in the operator env (the Python backend
 * reads it). The mainnet deploy itself is governance-gated — do NOT remove the
 * base/mainnet guards below without a ratified resolution.
 */
const hre = require("hardhat");
const { assertSignerMatchesOwner } = require("./deploy-guards");
// sp994 — block-pinned reads so GATE-4 post-deploy invariant checks are immune to
// load-balanced-RPC read-replica lag (a stale read right after the irreversible
// deploy could fake a mismatch, or worse a false pass). See rpc-stable-read.js.
const { readAt } = require("./rpc-stable-read");

const DAY = 24 * 60 * 60;

async function main() {
  const network = hre.network.name;

  // Mainnet deploys are governance-gated. PRSM is Base-native; Ethereum mainnet
  // is never an intended target. Base mainnet requires an explicit ratified
  // resolution (none yet for CreatorStakeRegistry) — this script is for the
  // Sepolia REHEARSAL + a dry-run; flipping to base must be a separate,
  // council-ratified PR that removes this guard deliberately.
  if (network === "mainnet" || network === "base") {
    throw new Error(
      `Deploy to ${network} is BLOCKED for CreatorStakeRegistry: no ratified ` +
      `mainnet authorization yet. Use --network sepolia (rehearsal) or ` +
      `--network hardhat (dry-run). The Base mainnet deploy + 2-of-3 ` +
      `Foundation-Safe acceptOwnership is the irreversible commissioning ` +
      `ceremony — see the Sepolia runbook + obtain a council resolution first.`
    );
  }

  const ftnsAddress = process.env.FTNS_TOKEN_ADDRESS;
  const slasher = process.env.CREATOR_STAKE_SLASHER;
  const unbondDelay = parseInt(
    process.env.CREATOR_STAKE_UNBOND_DELAY || String(7 * DAY), 10
  );
  if (!ftnsAddress) throw new Error("FTNS_TOKEN_ADDRESS is required");
  if (!slasher) throw new Error("CREATOR_STAKE_SLASHER is required (non-zero)");

  const [deployer] = await hre.ethers.getSigners();
  const owner = process.env.CREATOR_STAKE_OWNER || deployer.address;
  const balance = await hre.ethers.provider.getBalance(deployer.address);
  const chainId = (await hre.ethers.provider.getNetwork()).chainId;

  console.log(`\n=== Deploying CreatorStakeRegistry to ${network} ===`);
  console.log(`Deployer:      ${deployer.address}`);
  console.log(`Balance:       ${hre.ethers.formatEther(balance)} ETH`);
  console.log(`Chain id:      ${chainId}`);
  console.log(`initialOwner:  ${owner}`);
  console.log(`FTNS token:    ${ftnsAddress}`);
  console.log(`Unbond delay:  ${unbondDelay}s (${unbondDelay / DAY}d)`);
  console.log(`Slasher:       ${slasher}`);
  if (balance === 0n && network !== "hardhat") {
    throw new Error("Deployer has zero balance");
  }

  // sp990 (readiness-audit ceremony-breaker fix): on a LIVE network the actual
  // signer MUST equal the intended owner — else the WRONG KEY is about to sign an
  // irreversible mainnet deploy (e.g. a stale MAINNET_PRIVATE_KEY in contracts/.env
  // winning over an inline PRIVATE_KEY, since base signs via
  // MAINNET_PRIVATE_KEY || PRIVATE_KEY). Fail LOUD before the deploy. Tested in
  // test/deploy-guards.test.js. ALLOW_OWNER_DEPLOYER_MISMATCH=1 opts out.
  assertSignerMatchesOwner({
    network,
    deployerAddress: deployer.address,
    owner,
  });

  console.log(`\n[1/1] Deploying CreatorStakeRegistry…`);
  const Reg = await hre.ethers.getContractFactory("CreatorStakeRegistry");
  const reg = await Reg.deploy(owner, ftnsAddress, unbondDelay, slasher);
  await reg.waitForDeployment();
  const addr = await reg.getAddress();
  // Block the deploy tx mined in — all post-deploy reads are pinned to it so a
  // lagging read-replica cannot fake a mismatch (it errors on the block until
  // synced; readAt retries) nor return a stale false-pass.
  const deployBlock = (await reg.deploymentTransaction().wait()).blockNumber;
  console.log(`   CreatorStakeRegistry: ${addr} (deploy block ${deployBlock})`);

  // ── Post-deploy invariant checks (block-pinned; immune to read-replica lag) ──
  console.log(`\nPost-deploy invariant checks…`);
  const onOwner = await readAt(deployBlock, (bt) => reg.owner({ blockTag: bt }));
  const onFtns = await readAt(deployBlock, (bt) => reg.ftns({ blockTag: bt }));
  const onSlasher = await readAt(deployBlock, (bt) => reg.slasher({ blockTag: bt }));
  const onDelay = await readAt(deployBlock, (bt) => reg.unbondDelaySeconds({ blockTag: bt }));
  console.log(`   owner:    ${onOwner}`);
  console.log(`   ftns:     ${onFtns}`);
  console.log(`   slasher:  ${onSlasher}`);
  console.log(`   delay:    ${onDelay}`);
  if (onOwner.toLowerCase() !== owner.toLowerCase()) {
    throw new Error(`owner mismatch: ${onOwner} != ${owner}`);
  }
  if (onFtns.toLowerCase() !== ftnsAddress.toLowerCase()) {
    throw new Error(`ftns mismatch: ${onFtns} != ${ftnsAddress}`);
  }
  if (onSlasher.toLowerCase() !== slasher.toLowerCase()) {
    throw new Error(`slasher mismatch: ${onSlasher} != ${slasher}`);
  }
  if (onDelay !== BigInt(unbondDelay)) {
    throw new Error(`delay mismatch: ${onDelay} != ${unbondDelay}`);
  }
  // creatorStakeOf for an unstaked address must be 0 (responsive + correct ABI).
  const zero = await readAt(deployBlock,
    (bt) => reg.creatorStakeOf(hre.ethers.ZeroAddress, { blockTag: bt }));
  if (zero !== 0n) throw new Error(`fresh creatorStakeOf != 0: ${zero}`);
  console.log(`   ✓ invariants OK`);

  if (process.env.AUTO_VERIFY === "1" && network !== "hardhat") {
    console.log(`\nVerifying on explorer…`);
    try {
      await hre.run("verify:verify", {
        address: addr,
        constructorArguments: [owner, ftnsAddress, unbondDelay, slasher],
      });
      console.log(`   ✓ verified`);
    } catch (e) {
      console.log(`   verify failed (verify manually): ${e.message}`);
    }
  }

  const cfgName = network === "base" ? "MAINNET" : network;
  console.log(`\n=== DEPLOY COMPLETE ===`);
  console.log(`CreatorStakeRegistry @ ${addr} on ${network} (chain ${chainId})`);
  console.log(`For mainnet: FIRST transferOwnership(FoundationSafe) → 2-of-3 Safe acceptOwnership.`);
  console.log(`Activation (the RUNTIME source of truth — sp981/sp990):`);
  console.log(`  Record this address in prsm/config/networks.py under the ${cfgName}`);
  console.log(`  config's creator_stake_registry field; CreatorStakeClient.from_env`);
  console.log(`  reads it via resolve_endpoints (no separate env var needed).`);
  console.log(`  NOTE: prsm/deployments/contract_addresses.json is informational ONLY —`);
  console.log(`  no runtime code reads it for the creator-stake gate.`);
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
