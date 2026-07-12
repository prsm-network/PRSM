// Team D — D3 PoC: Unbond/withdraw vs slash race under governance-misconfigured
// (unbondDelay < challengeWindow). Demonstrates a provider can withdraw their
// stake before a still-valid challenge fires, evading slashing entirely.
//
// Stated invariant (team-prompt §Stated invariants #2):
//   "Unbonding delay is non-zero and respects in-flight slashes. A pending
//    slash must block unbond completion until resolved."
//
// Reality: there is NO contract-level invariant binding unbondDelay >=
// challengeWindow, nor any "pending slash" tracking. With governance-allowed
// values (unbondDelay = 1 day MIN, challengeWindow = 30 days MAX), a provider
// can:
//   1. Commit a batch
//   2. Immediately requestUnbond
//   3. Wait 1 day (unbond delay)
//   4. withdraw — stake fully recovered
//   5. Challenger fires DOUBLE_SPEND at day 5 — challenge SUCCEEDS but slash
//      fails silently (try/catch swallows NotSlashable on WITHDRAWN status)
// Provider walks away with full stake despite proven misbehavior.

const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

describe("[Team D] D3 — Unbond/withdraw races slash under misconfigured windows", function () {
  let registry, pool, stakeBond, token;
  let owner, provider, requester, challenger;
  const ONE_FTNS = ethers.parseUnits("1", 18);
  const STAKE = ONE_FTNS * 10000n; // 10K FTNS provider stake
  const ESCROW = ONE_FTNS * 1000n;

  // Worst-case governance values: short unbond, long challenge window.
  const UNBOND_DELAY = 1 * 24 * 60 * 60; // 1 day MIN
  const CHALLENGE_WINDOW = 30 * 24 * 60 * 60; // 30 day MAX
  const TIER_BPS = 5000; // 50% slash rate

  beforeEach(async function () {
    [owner, provider, requester, challenger] = await ethers.getSigners();

    const Token = await ethers.getContractFactory("MockERC20");
    token = await Token.deploy();
    await token.waitForDeployment();

    const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
    registry = await Registry.deploy(owner.address, CHALLENGE_WINDOW);
    await registry.waitForDeployment();

    const Pool = await ethers.getContractFactory("EscrowPool");
    pool = await Pool.deploy(owner.address, await token.getAddress(), await registry.getAddress());
    await pool.waitForDeployment();

    const StakeBond = await ethers.getContractFactory("StakeBond");
    stakeBond = await StakeBond.deploy(owner.address, await token.getAddress(), UNBOND_DELAY, await registry.getAddress());
    await stakeBond.waitForDeployment();

    // Wire
    await registry.connect(owner).setEscrowPool(await pool.getAddress());
    await registry.connect(owner).setStakeBond(await stakeBond.getAddress());

    // Fund
    await token.mint(provider.address, STAKE);
    await token.mint(requester.address, ESCROW);
    await token.connect(provider).approve(await stakeBond.getAddress(), STAKE);
    await token.connect(requester).approve(await pool.getAddress(), ESCROW);
    await pool.connect(requester).deposit(ESCROW);
    await stakeBond.connect(provider).bond(STAKE, TIER_BPS);
  });

  it("provider withdraws stake before slash window closes — slash silently no-ops", async function () {
    // Build minimal merkle tree with 1 leaf so the challenge can prove inclusion.
    const leaf = {
      jobIdHash: ethers.keccak256(ethers.toUtf8Bytes("job-1")),
      shardIndex: 0,
      providerIdHash: ethers.keccak256(ethers.toUtf8Bytes("provider")),
      providerPubkeyHash: ethers.keccak256(ethers.toUtf8Bytes("pubkey")),
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("output")),
      executedAtUnix: (await time.latest()) - 100,
      valueFtns: ethers.parseUnits("10", 18),
      signatureHash: ethers.keccak256(ethers.toUtf8Bytes("sig")),
      signingMessageHash: ethers.keccak256(ethers.toUtf8Bytes("signing-msg")),
    };
    const leafType =
      "tuple(bytes32 jobIdHash,uint32 shardIndex,bytes32 providerIdHash,bytes32 providerPubkeyHash,bytes32 outputHash,uint64 executedAtUnix,uint128 valueFtns,bytes32 signatureHash,bytes32 signingMessageHash)";
    const leafEnc = ethers.AbiCoder.defaultAbiCoder().encode([leafType], [leaf]);
    const leafHash = ethers.keccak256(leafEnc);
    const merkleRoot = leafHash; // single leaf == root

    // Commit two batches (provider double-spends the same receipt).
    const tx1 = await registry.connect(provider).commitBatch(
      requester.address, merkleRoot, 1, leaf.valueFtns, TIER_BPS, ethers.ZeroHash, "ipfs://batch1"
    );
    const r1 = await tx1.wait();
    const batchId1 = r1.logs.find(l => l.fragment && l.fragment.name === "BatchCommitted").args[0];

    // Move forward 1 block before second commit so block.number differs
    // (batchId derivation includes block.number).
    await ethers.provider.send("evm_mine", []);

    const tx2 = await registry.connect(provider).commitBatch(
      requester.address, merkleRoot, 1, leaf.valueFtns, TIER_BPS, ethers.ZeroHash, "ipfs://batch2"
    );
    const r2 = await tx2.wait();
    const batchId2 = r2.logs.find(l => l.fragment && l.fragment.name === "BatchCommitted").args[0];

    // Provider immediately initiates unbond. Post-fix, requestUnbond
    // clamps unbond_eligible_at to AT LEAST now + challengeWindowSeconds.
    await stakeBond.connect(provider).requestUnbond();
    const stakeAfterUnbondReq = await stakeBond.stakes(provider.address);

    // 1 day passes — local unbondDelay elapsed; challenge-window-floor
    // has NOT.
    await time.increase(UNBOND_DELAY + 60);

    // Post-fix: withdraw MUST revert because unbond_eligible_at sits at
    // the 30-day mark (challenge window), not the 1-day mark.
    await expect(
      stakeBond.connect(provider).withdraw()
    ).to.be.revertedWithCustomError(stakeBond, "UnbondDelayNotElapsed");

    const stake = await stakeBond.stakes(provider.address);
    expect(stake.status).to.equal(2); // UNBONDING (NOT WITHDRAWN)
    expect(stake.amount).to.equal(STAKE);

    // Challenger fires DOUBLE_SPEND at day 5 — slash now SUCCEEDS
    // because stake is still UNBONDING (slashable).
    // sp1429 first-committer-wins: challenge the LATER duplicate (batchId2) citing the EARLIER
    // original (batchId1). Both are the provider's, so this slashes the same provider — the guard
    // requires the cited conflict to be strictly earlier.
    const auxData = ethers.AbiCoder.defaultAbiCoder().encode(
      ["bytes32", "bytes32[]"],
      [batchId1, []]
    );

    const fdReserveBefore = await stakeBond.foundationReserveBalance();
    const challengerBountyBefore = await stakeBond.slashedBountyPayable(challenger.address);

    await expect(
      registry.connect(challenger).challengeReceipt(
        batchId2, leaf, [], 0 /* DOUBLE_SPEND */, auxData
      )
    ).to.emit(stakeBond, "Slashed");

    // Slashing actually fired: foundation reserve + challenger bounty grew.
    const fdReserveAfter = await stakeBond.foundationReserveBalance();
    const challengerBountyAfter = await stakeBond.slashedBountyPayable(challenger.address);
    const slashAmount = (STAKE * BigInt(TIER_BPS)) / 10000n;
    const expectedBounty = (slashAmount * 7000n) / 10000n;
    const expectedReserve = slashAmount - expectedBounty;
    expect(challengerBountyAfter - challengerBountyBefore).to.equal(expectedBounty);
    expect(fdReserveAfter - fdReserveBefore).to.equal(expectedReserve);
  });
});
