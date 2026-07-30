// Sprint 1481 — build the Foundation Safe payload for
// CompensationDistributor.setPoolAddresses(creator, operator, grant).
//
//   REWARD_POOL_ADDRESS=0x<deployed OperatorRewardPool> \
//     npx hardhat run scripts/build-setpooladdresses-safe-tx.js --network base
//
// READ-ONLY: signs nothing, sends nothing. It emits a Safe Transaction Builder
// JSON you import at app.safe.global, plus the raw calldata and a decoded diff so
// each hardware-wallet signer can verify what they are approving.
//
// THE FOOTGUN THIS EXISTS TO PREVENT
// ---------------------------------
// setPoolAddresses sets ALL THREE pools in one call. To point `operator` at the
// new OperatorRewardPool you must also pass `creator` and `grant` — and passing
// the wrong values silently REDIRECTS creator and grant emissions to an
// unintended address. (Zeros revert, which is the only case the contract catches
// for you.) So this script READS the current creator/grant on-chain and preserves
// them by default; overriding either one requires an explicit env var, and the
// printed diff shows exactly which slots change.
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

const CD_MAINNET = "0xa9551F5a3AeAB39cc8315AcD8caC2886Bd04f244";
const FOUNDATION_SAFE = "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791";

const CD_ABI = [
  "function owner() view returns (address)",
  "function creatorPool() view returns (address)",
  "function operatorPool() view returns (address)",
  "function grantPool() view returns (address)",
  "function setPoolAddresses(address creator, address operator, address grant)",
];

async function main() {
  const { ethers } = hre;
  const net = await ethers.provider.getNetwork();
  const cdAddr = ethers.getAddress(process.env.COMPENSATION_DISTRIBUTOR || CD_MAINNET);

  const newOperator = process.env.REWARD_POOL_ADDRESS;
  if (!newOperator) {
    throw new Error(
      "set REWARD_POOL_ADDRESS=0x<deployed OperatorRewardPool> (deploy it first " +
      "with scripts/deploy-operator-reward-pool.js)"
    );
  }
  const operator = ethers.getAddress(newOperator);

  const cd = new ethers.Contract(cdAddr, CD_ABI, ethers.provider);
  const [owner, curCreator, curOperator, curGrant] = await Promise.all([
    cd.owner(), cd.creatorPool(), cd.operatorPool(), cd.grantPool(),
  ]);

  // Preserve unless explicitly overridden — see the header.
  const creator = ethers.getAddress(process.env.CREATOR_POOL || curCreator);
  const grant = ethers.getAddress(process.env.GRANT_POOL || curGrant);

  console.log(`\n=== setPoolAddresses payload — ${hre.network.name} (chain ${net.chainId}) ===`);
  console.log(`CompensationDistributor: ${cdAddr}`);
  console.log(`owner (must sign)      : ${owner}${
    owner.toLowerCase() === FOUNDATION_SAFE.toLowerCase()
      ? "  (Foundation Safe ✓)" : "  ⚠️  NOT the known Safe"}`);

  // Sanity: the new operator pool must be a contract on THIS chain. Pointing
  // emissions at an EOA or a wrong-chain address would strand every payout.
  const code = await ethers.provider.getCode(operator);
  if (code === "0x") {
    throw new Error(
      `no contract at REWARD_POOL_ADDRESS ${operator} on chain ${net.chainId} — ` +
      `refusing to build a payload that would point emissions at a non-contract`);
  }

  console.log(`\nDIFF (what the signers are approving):`);
  const row = (label, before, after) => {
    const changed = before.toLowerCase() !== after.toLowerCase();
    console.log(`  ${label.padEnd(9)} ${changed ? "CHANGE" : "keep  "}  ${before}`);
    if (changed) console.log(`            ${" ".repeat(6)}  ${after}   <-- NEW`);
  };
  row("creator", curCreator, creator);
  row("operator", curOperator, operator);
  row("grant", curGrant, grant);
  const changes = [
    curCreator.toLowerCase() !== creator.toLowerCase(),
    curOperator.toLowerCase() !== operator.toLowerCase(),
    curGrant.toLowerCase() !== grant.toLowerCase(),
  ].filter(Boolean).length;
  if (changes !== 1) {
    console.log(`\n  ⚠️  ${changes} slots change. The intended ceremony changes exactly ONE`);
    console.log(`      (operator). Re-check CREATOR_POOL / GRANT_POOL overrides.`);
  }

  const iface = new ethers.Interface(CD_ABI);
  const data = iface.encodeFunctionData("setPoolAddresses", [creator, operator, grant]);
  console.log(`\nRaw calldata (verify this on the Ledger screen):`);
  console.log(`  to   : ${cdAddr}`);
  console.log(`  value: 0`);
  console.log(`  data : ${data}`);

  // Safe Transaction Builder batch format — importable at app.safe.global.
  const batch = {
    version: "1.0",
    chainId: net.chainId.toString(),
    createdAt: 0,
    meta: {
      name: "PRSM — point operatorPool at OperatorRewardPool",
      description:
        "CompensationDistributor.setPoolAddresses: redirect the OPERATOR pool to the " +
        "deployed OperatorRewardPool so emitted FTNS can reach individual earners. " +
        "creator + grant are preserved at their current values.",
      txBuilderVersion: "1.16.5",
    },
    transactions: [{
      to: cdAddr,
      value: "0",
      data: null,
      contractMethod: {
        inputs: [
          { name: "creator", type: "address", internalType: "address" },
          { name: "operator", type: "address", internalType: "address" },
          { name: "grant", type: "address", internalType: "address" },
        ],
        name: "setPoolAddresses",
        payable: false,
      },
      contractInputsValues: { creator, operator, grant },
    }],
  };

  const outDir = path.join(__dirname, "..", "deployments");
  fs.mkdirSync(outDir, { recursive: true });
  const out = path.join(outDir, `safe-tx-setPoolAddresses-${net.chainId}.json`);
  fs.writeFileSync(out, JSON.stringify(batch, null, 2));
  console.log(`\nSafe Transaction Builder batch: ${out}`);

  console.log(`\nCEREMONY (2-of-3 — needs a browser + two hardware wallets):`);
  console.log(`  1. app.safe.global -> Safe ${owner} -> Apps -> Transaction Builder`);
  console.log(`  2. Import the JSON above.`);
  console.log(`  3. EACH signer verifies on the DEVICE SCREEN, not the browser:`);
  console.log(`       to   == ${cdAddr}`);
  console.log(`       data == the calldata printed above`);
  console.log(`  4. Two signers approve; execute.`);
  console.log(`  5. Confirm on-chain: operatorPool() == ${operator}`);
  console.log(`\n  Nothing flows until the pool is FUNDED and an epoch root is published.\n`);
}

main().then(() => process.exit(0)).catch((e) => {
  console.error(`\n❌ ${e.message}\n`);
  process.exit(1);
});
