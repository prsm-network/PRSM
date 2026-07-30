// Sprint 1481 — emit OperatorRewardPool leaf/root GOLDEN VECTORS from the EVM.
//
// These pin Python<->Solidity parity for the pool -> earner rail. The vectors are
// consumed by tests/unit/test_sprint_1481_reward_epoch.py; regenerate with
//     npx hardhat run scripts/gen_reward_epoch_goldens.js
// ONLY when the leaf encoding changes deliberately - and change the contract, the
// Python builder and those goldens in the SAME commit, or epochs publish fine and
// every claim reverts InvalidProof.
const { ethers, upgrades } = require("hardhat");
function hashPair(a,b){const [x,y]=a.toLowerCase()<=b.toLowerCase()?[a,b]:[b,a];return ethers.keccak256(ethers.concat([x,y]));}
function buildRoot(l){let L=[...l];while(L.length>1){const n=[];for(let i=0;i<L.length;i+=2){n.push(i+1<L.length?hashPair(L[i],L[i+1]):L[i]);}L=n;}return L[0];}
async function main(){
  const [owner, publisher, , , , , treasury] = await ethers.getSigners();
  const Token = await ethers.getContractFactory("FTNSTokenSimple");
  const ftns = await upgrades.deployProxy(Token,[owner.address,treasury.address],{initializer:"initialize",kind:"uups"});
  const Pool = await ethers.getContractFactory("OperatorRewardPool");
  const pool = await Pool.deploy(await ftns.getAddress(), owner.address, publisher.address);
  await pool.waitForDeployment();

  // Fixed, non-signer addresses so the vectors are stable across machines.
  const cases = [
    [1n,  "0x1111111111111111111111111111111111111111", 1n],
    [1n,  "0xabc0000000000000000000000000000000000001", 1000000000000000000n],
    [42n, "0xdead000000000000000000000000000000000000", 123456789012345678901n],
    [0n,  "0x0000000000000000000000000000000000000001", 2n**255n],
  ];
  const out = { leaves: [], tree: null };
  for (const [e,a,amt] of cases) {
    out.leaves.push({epoch_id:e.toString(), account:a, amount_wei:amt.toString(),
                     leaf: await pool.leafHash(e,a,amt)});
  }
  // A 3-leaf tree (odd layer -> exercises the odd-promotion rule) built from
  // on-chain leaf hashes, sorted by address the way the Python builder sorts.
  const treeEpoch = 7n;
  const entries = [
    {account:"0x1111111111111111111111111111111111111111", amount:"100"},
    {account:"0x2222222222222222222222222222222222222222", amount:"250"},
    {account:"0x3333333333333333333333333333333333333333", amount:"50"},
  ];
  const leaves = [];
  for (const en of entries) leaves.push(await pool.leafHash(treeEpoch, en.account, BigInt(en.amount)));
  out.tree = {epoch_id: treeEpoch.toString(), entries, leaves, root: buildRoot(leaves)};
  console.log("GOLDEN_JSON_START");
  console.log(JSON.stringify(out, null, 2));
  console.log("GOLDEN_JSON_END");
}
main().catch(e=>{console.error(e);process.exit(1);});
