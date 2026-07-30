// Sprint 1481 — deploy OperatorRewardPool (the pool -> earner rail).
//
//   DRY RUN (no broadcast, recommended first):
//     npx hardhat run scripts/deploy-operator-reward-pool.js --network base-fork
//
//   MAINNET (irreversible; requires an explicit confirmation):
//     CONFIRM_MAINNET_DEPLOY=yes \
//     REWARD_POOL_OWNER=0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791 \
//     REWARD_POOL_PUBLISHER=0x<epoch-job-key-address> \
//     npx hardhat run scripts/deploy-operator-reward-pool.js --network base
//
// Deploying this contract ACTIVATES NOTHING on its own — emissions only reach it
// once the Foundation Safe calls CompensationDistributor.setPoolAddresses (a
// separate 2-of-3 ceremony; build its payload with
// scripts/build-setpooladdresses-safe-tx.js). So this step is safe to do ahead of
// the signing ceremony, and it produces the address that ceremony needs.
//
// Run it under tmux if you are driving from a phone/mosh session: an interruption
// between broadcast and the deployment-record write leaves you unsure whether the
// contract exists.
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

// Base mainnet FTNS + the Foundation Safe (verified on-chain).
const FTNS_MAINNET = "0x5276a3756C85f2E9e46f6D34386167a209aa16e5";
const FOUNDATION_SAFE = "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791";

async function main() {
  const { ethers } = hre;
  const net = await ethers.provider.getNetwork();
  const chainId = net.chainId;
  const isMainnet = chainId === 8453n;

  console.log(`\n=== Deploy OperatorRewardPool — ${hre.network.name} (chain ${chainId}) ===`);

  if (isMainnet && process.env.CONFIRM_MAINNET_DEPLOY !== "yes") {
    throw new Error(
      "refusing to deploy to Base MAINNET without CONFIRM_MAINNET_DEPLOY=yes. " +
      "Dry-run on --network base-fork first."
    );
  }

  const [deployer] = await ethers.getSigners();
  if (!deployer) {
    throw new Error(
      "no signer — set PRIVATE_KEY=0x<64 hex> (the 0x prefix is required; " +
      "hardhat.config's pkAccounts() skips the network without it)."
    );
  }

  // OWNER should be the Foundation Safe, not the deploying EOA: the owner can
  // pause and re-point the publisher, so leaving it on a hot key would put those
  // controls behind a single compromised key. Ownable2Step means a later transfer
  // needs the Safe to acceptOwnership, so getting this right at construction
  // avoids a second ceremony.
  const owner = ethers.getAddress(process.env.REWARD_POOL_OWNER || FOUNDATION_SAFE);
  // PUBLISHER is the automated epoch-job key. Deliberately NOT the owner: it
  // publishes roots but cannot move funds (verified by the contract's tests).
  const publisher = ethers.getAddress(
    process.env.REWARD_POOL_PUBLISHER || deployer.address);
  const ftns = ethers.getAddress(process.env.REWARD_POOL_FTNS || FTNS_MAINNET);

  const bal = await ethers.provider.getBalance(deployer.address);
  console.log(`Deployer : ${deployer.address}  (${ethers.formatEther(bal)} ETH)`);
  console.log(`FTNS     : ${ftns}`);
  console.log(`Owner    : ${owner}${owner.toLowerCase() === FOUNDATION_SAFE.toLowerCase()
    ? "  (Foundation Safe ✓)" : "  ⚠️  NOT the Foundation Safe"}`);
  console.log(`Publisher: ${publisher}${publisher.toLowerCase() === owner.toLowerCase()
    ? "  ⚠️  same as owner — the epoch key should be separate" : "  (separate from owner ✓)"}`);
  if (bal === 0n) throw new Error("deployer has no ETH for gas");

  // Sanity: the FTNS address must actually be an ERC-20 on this chain, or every
  // future claim reverts and the mistake is only discovered by an earner.
  const code = await ethers.provider.getCode(ftns);
  if (code === "0x") throw new Error(`no contract at FTNS address ${ftns} on chain ${chainId}`);

  console.log(`\nDeploying…`);
  const Pool = await ethers.getContractFactory("OperatorRewardPool");
  const pool = await Pool.deploy(ftns, owner, publisher);
  await pool.waitForDeployment();
  const addr = await pool.getAddress();
  const tx = pool.deploymentTransaction();
  console.log(`  OperatorRewardPool: ${addr}`);
  console.log(`  tx: ${tx && tx.hash}`);

  // Read back from chain rather than trusting the constructor args we sent.
  console.log(`\nVerifying on-chain state…`);
  console.log(`  ftns()            : ${await pool.ftns()}`);
  console.log(`  owner()           : ${await pool.owner()}`);
  console.log(`  rootPublisher()   : ${await pool.rootPublisher()}`);
  console.log(`  paused()          : ${await pool.paused()}`);
  console.log(`  MIN_CLAIM_WINDOW  : ${await pool.MIN_CLAIM_WINDOW()}s`);
  console.log(`  totalReserved()   : ${await pool.totalReserved()}`);

  const outDir = path.join(__dirname, "..", "deployments");
  fs.mkdirSync(outDir, { recursive: true });
  const out = path.join(outDir, `operator-reward-pool-${hre.network.name}-${Date.now()}.json`);
  fs.writeFileSync(out, JSON.stringify({
    network: hre.network.name, chainId: chainId.toString(),
    contracts: { OperatorRewardPool: addr, FTNSToken: ftns },
    owner, rootPublisher: publisher, deployer: deployer.address,
    txHash: tx && tx.hash,
    nextStep: "Foundation Safe (2-of-3) must call CompensationDistributor." +
              "setPoolAddresses to point operatorPool here — build the payload with " +
              "scripts/build-setpooladdresses-safe-tx.js",
  }, null, 2));
  console.log(`\nDeployment record: ${out}`);

  console.log(`\nNEXT — nothing flows yet. Emissions reach this contract only after the`);
  console.log(`Safe ceremony. Build the exact Transaction Builder payload with:`);
  console.log(`  REWARD_POOL_ADDRESS=${addr} \\`);
  console.log(`    npx hardhat run scripts/build-setpooladdresses-safe-tx.js --network base\n`);
}

main().then(() => process.exit(0)).catch((e) => {
  console.error(`\n❌ ${e.message}\n`);
  process.exit(1);
});
