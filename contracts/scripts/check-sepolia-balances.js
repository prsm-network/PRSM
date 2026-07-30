// Sprint 1481 — Base Sepolia balance check for rehearsal funding.
//
// Reports native ETH (gas) and ERC-20 FTNS for each address, and says plainly
// which one can actually run the OperatorRewardPool rehearsal.
//
//   npx hardhat run scripts/check-sepolia-balances.js --network base-sepolia
//   ADDRESSES=0xabc...,0xdef... npx hardhat run scripts/check-sepolia-balances.js --network base-sepolia
//   FTNS_ADDRESS=0x... (override the token; defaults to the deployed Sepolia FTNS)
//
// With no ADDRESSES it checks the configured signer (PRIVATE_KEY) plus any
// known Sepolia deployer addresses. Read-only: signs nothing, sends nothing.
const hre = require("hardhat");

// Deployed Base Sepolia FTNS (contracts/deployments/phase1-ftns-base-sepolia-*.json)
const DEFAULT_FTNS = "0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a";

const ERC20_ABI = [
  { name: "balanceOf", type: "function", stateMutability: "view",
    inputs: [{ name: "a", type: "address" }], outputs: [{ type: "uint256" }] },
  { name: "symbol", type: "function", stateMutability: "view",
    inputs: [], outputs: [{ type: "string" }] },
  { name: "decimals", type: "function", stateMutability: "view",
    inputs: [], outputs: [{ type: "uint8" }] },
];

// Enough ETH to deploy 2 contracts + ~8 txs on Base Sepolia, generously rounded.
const MIN_GAS_ETH = 0.005;

async function main() {
  const { ethers } = hre;
  const net = await ethers.provider.getNetwork();
  if (net.chainId !== 84532n) {
    throw new Error(
      `expected Base Sepolia (84532), got ${net.chainId}. ` +
      `Re-run with --network base-sepolia`
    );
  }

  const targets = [];
  if (process.env.ADDRESSES) {
    for (const a of process.env.ADDRESSES.split(",")) {
      const t = a.trim();
      if (t) targets.push({ address: ethers.getAddress(t), label: "ADDRESSES" });
    }
  } else {
    for (const s of await ethers.getSigners()) {
      targets.push({ address: s.address, label: "configured signer (PRIVATE_KEY)" });
    }
    for (const a of ["0x55d2B5623426BC65534C472b5987Cbb871619C74",
                     "0xF7d88c943B048dAd2e5178E40DaaD545dB3311c2"]) {
      if (!targets.some((t) => t.address.toLowerCase() === a.toLowerCase())) {
        targets.push({ address: ethers.getAddress(a), label: "known deployer" });
      }
    }
  }
  if (targets.length === 0) {
    throw new Error(
      "no addresses to check — pass ADDRESSES=0x…,0x… or configure PRIVATE_KEY"
    );
  }

  const tokenAddr = ethers.getAddress(process.env.FTNS_ADDRESS || DEFAULT_FTNS);
  const token = new ethers.Contract(tokenAddr, ERC20_ABI, ethers.provider);
  let symbol = "FTNS", decimals = 18n, tokenLive = true;
  try {
    symbol = await token.symbol();
    decimals = BigInt(await token.decimals());
  } catch {
    tokenLive = false;
  }

  console.log(`\n=== Base Sepolia balances (chain ${net.chainId}) ===`);
  console.log(`Token: ${tokenAddr}` + (tokenLive ? ` (${symbol})` : ` — NOT an ERC-20 here`));
  console.log("");

  const rows = [];
  for (const t of targets) {
    const eth = await ethers.provider.getBalance(t.address);
    let bal = 0n;
    if (tokenLive) {
      try { bal = await token.balanceOf(t.address); } catch { bal = 0n; }
    }
    rows.push({ ...t, eth, bal });
    const ethStr = ethers.formatEther(eth);
    const tokStr = tokenLive ? ethers.formatUnits(bal, decimals) : "n/a";
    const gasOk = Number(ethStr) >= MIN_GAS_ETH;
    console.log(`${t.address}`);
    console.log(`   ${gasOk ? "✅" : "❌"} gas : ${ethStr} ETH${gasOk ? "" : `  (need ~${MIN_GAS_ETH})`}`);
    console.log(`   ${bal > 0n ? "✅" : "  "} ${symbol.padEnd(4)}: ${tokStr}`);
    console.log(`      (${t.label})`);
    console.log("");
  }

  // Compare token balances against the ACTUAL pot the rehearsal will move — a
  // non-zero balance is not the same as a sufficient one, and recommending the
  // live token on a balance below the pot sends the operator into a failing run.
  let potWei = 0n;
  try {
    potWei = BigInt(require("../test/fixtures/emission_epoch_from_batches.json")
      .total_amount_wei);
  } catch { /* fixture optional */ }

  const fundable = rows.filter((r) => Number(ethers.formatEther(r.eth)) >= MIN_GAS_ETH);
  console.log("=== Rehearsal readiness ===");
  if (potWei > 0n) {
    console.log(`Epoch pot to be distributed: ${ethers.formatUnits(potWei, decimals)} ${symbol}\n`);
  }
  if (fundable.length === 0) {
    console.log(`❌ No address has >= ${MIN_GAS_ETH} ETH for gas.`);
    console.log(`   Fund one at https://www.alchemy.com/faucets/base-sepolia`);
    console.log(`   (it must be the PRIVATE_KEY address to run the rehearsal directly)`);
  } else {
    for (const r of fundable) {
      console.log(`✅ ${r.address} can pay gas.`);
      if (potWei > 0n && r.bal >= potWei) {
        console.log(`   Holds enough ${symbol} for the pot — you MAY reuse the live token:`);
        console.log(`      REHEARSAL_FTNS_ADDRESS=${tokenAddr}`);
      } else {
        const held = ethers.formatUnits(r.bal, decimals);
        console.log(`   Holds ${held} ${symbol} < pot — do NOT set REHEARSAL_FTNS_ADDRESS.`);
        console.log(`   Leave it unset and the rehearsal deploys a fresh test token it`);
        console.log(`   fully funds itself. That is the CORRECT way to rehearse: it`);
        console.log(`   exercises the identical code path without needing real balances.`);
      }
      console.log(`\n   Run it:`);
      console.log(`      npx hardhat run scripts/rehearse-reward-pool.js --network base-sepolia`);
    }
  }
  console.log("");
  console.log("To check your OTHER wallets (addresses are public — safe to paste):");
  console.log("   ADDRESSES=0xaaa...,0xbbb... npx hardhat run \\");
  console.log("     scripts/check-sepolia-balances.js --network base-sepolia");
  console.log("");
}

main().then(() => process.exit(0)).catch((e) => {
  console.error(`\n❌ ${e.message}\n`);
  process.exit(1);
});
