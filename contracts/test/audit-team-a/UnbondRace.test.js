const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

// Team A — Economic Audit
//
// Finding A-02 PoC: stake-bond UNBOND-vs-CHALLENGE race.
//
// The contract pair (BatchSettlementRegistry + StakeBond) has no on-chain
// constraint linking unbondDelaySeconds >= challengeWindowSeconds.
// MIN_UNBOND_DELAY_SECONDS = 1 day. challengeWindow MIN = 1 hour, default
// 3 days, MAX 30 days. If governance ever sets unbondDelay shorter than
// challengeWindow (or simply leaves both at independent defaults that
// happen to satisfy unbondDelay < challengeWindow), a malicious provider
// can:
//   1. commitBatch with fraudulent receipts at T=0
//   2. requestUnbond at T=0
//   3. withdraw stake at T = unbondDelay (state -> WITHDRAWN, amount = 0)
//   4. challenger discovers fraud at T < challengeWindow
//   5. challengeReceipt invalidates receipt but slash() reverts
//      NotSlashable (status == WITHDRAWN); reverts swallowed by try/catch.
//   6. provider keeps full stake, batch is invalidated (no payment) but
//      provider was never going to get paid for fraud anyway — the
//      ECONOMIC PENALTY (slash) is lost.
//
// Net: the slashing-during-UNBONDING defense (StakeBond.sol:338) only
// closes the race if unbondDelay >= challengeWindow. The contracts do
// NOT enforce that relationship, and the parameter ranges allow
// misconfiguration.

describe("AUDIT-TEAM-A — A02 Unbond/Challenge race lets attacker dodge slash", function () {
  let registry, escrowPool, token, stakeBond;
  let owner, attacker, requester, challenger, foundation;
  const ONE_FTNS = ethers.parseUnits("1", 18);
  const PREMIUM_STAKE = 25_000n * ONE_FTNS;
  const PREMIUM_SLASH_BPS = 10000; // 100%

  // === MISCONFIGURATION SCENARIO ===
  // unbondDelay = 1 day (the MIN allowed by StakeBond)
  // challengeWindow = 7 days (a perfectly reasonable governance choice)
  const UNBOND_DELAY = 1 * 24 * 60 * 60;     // 1 day
  const CHALLENGE_WINDOW = 7 * 24 * 60 * 60; // 7 days

  function makeLeaf(overrides = {}) {
    return {
      jobIdHash: ethers.keccak256(ethers.toUtf8Bytes("job-attacker")),
      shardIndex: 0,
      providerIdHash: ethers.keccak256(ethers.toUtf8Bytes("attacker-pid")),
      providerPubkeyHash: ethers.keccak256(ethers.toUtf8Bytes("pubkey")),
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("malicious-output")),
      executedAtUnix: Math.floor(Date.now() / 1000),
      valueFtns: ONE_FTNS,
      signatureHash: ethers.keccak256(ethers.toUtf8Bytes("sig")),
      signingMessageHash: ethers.keccak256(ethers.toUtf8Bytes("signing-msg")),
      ...overrides,
    };
  }

  function hashLeaf(leaf) {
    const encoded = ethers.AbiCoder.defaultAbiCoder().encode(
      ["tuple(bytes32,uint32,bytes32,bytes32,bytes32,uint64,uint128,bytes32,bytes32)"],
      [[
        leaf.jobIdHash, leaf.shardIndex, leaf.providerIdHash,
        leaf.providerPubkeyHash, leaf.outputHash, leaf.executedAtUnix,
        leaf.valueFtns, leaf.signatureHash, leaf.signingMessageHash,
      ]],
    );
    return ethers.keccak256(encoded);
  }

  beforeEach(async function () {
    [owner, attacker, requester, challenger, foundation] = await ethers.getSigners();

    const Token = await ethers.getContractFactory("MockERC20");
    token = await Token.deploy();
    await token.waitForDeployment();

    const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
    registry = await Registry.deploy(owner.address, CHALLENGE_WINDOW);
    await registry.waitForDeployment();

    const Pool = await ethers.getContractFactory("EscrowPool");
    escrowPool = await Pool.deploy(owner.address, await token.getAddress(), await registry.getAddress());
    await escrowPool.waitForDeployment();

    await registry.connect(owner).setEscrowPool(await escrowPool.getAddress());

    const Bond = await ethers.getContractFactory("StakeBond");
    stakeBond = await Bond.deploy(owner.address, await token.getAddress(), UNBOND_DELAY, await registry.getAddress());
    await stakeBond.waitForDeployment();

    // L4 self-audit MED-4: foundation wallet must be a contract.
    const FoundationStub = await ethers.getContractFactory("MockERC20");
    const foundationStub = await FoundationStub.deploy();
    await foundationStub.waitForDeployment();
    await stakeBond.connect(owner).setFoundationReserveWallet(await foundationStub.getAddress());
    await registry.connect(owner).setStakeBond(await stakeBond.getAddress());

    await token.mint(requester.address, ONE_FTNS * 1000n);
    await token.connect(requester).approve(await escrowPool.getAddress(), ONE_FTNS * 1000n);
    await escrowPool.connect(requester).deposit(ONE_FTNS * 1000n);

    await token.mint(attacker.address, PREMIUM_STAKE * 2n);
    await token.connect(attacker).approve(await stakeBond.getAddress(), PREMIUM_STAKE * 2n);
    await stakeBond.connect(attacker).bond(PREMIUM_STAKE, PREMIUM_SLASH_BPS);
  });

  it("REGRESSION: post-fix, requestUnbond clamps unbond_eligible_at to slasher.challengeWindowSeconds — attacker cannot withdraw before window closes", async function () {
    // ---- Step 1: commit a fraudulent batch (DOUBLE_SPEND prep).
    const leaf = makeLeaf();
    const root = hashLeaf(leaf);

    const tx1 = await registry.connect(attacker).commitBatch(
      requester.address, root, 1, leaf.valueFtns, PREMIUM_SLASH_BPS, ethers.ZeroHash, "b1"
    );
    const r1 = await tx1.wait();
    const batchId1 = r1.logs.find(l => l.fragment && l.fragment.name === "BatchCommitted").args[0];

    const tx2 = await registry.connect(attacker).commitBatch(
      requester.address, root, 1, leaf.valueFtns, PREMIUM_SLASH_BPS, ethers.ZeroHash, "b2"
    );
    const r2 = await tx2.wait();
    const batchId2 = r2.logs.find(l => l.fragment && l.fragment.name === "BatchCommitted").args[0];

    // ---- Step 2: requestUnbond. Post-fix, unbond_eligible_at is clamped
    //              to AT LEAST now + challengeWindowSeconds, regardless
    //              of the local unbondDelaySeconds. The clamp comes from
    //              StakeBond reading slasher.challengeWindowSeconds().
    const requestUnbondTx = await stakeBond.connect(attacker).requestUnbond();
    const requestUnbondBlock = await ethers.provider.getBlock(requestUnbondTx.blockNumber);
    const stake = await stakeBond.stakes(attacker.address);
    const expectedFloor = BigInt(requestUnbondBlock.timestamp) + BigInt(CHALLENGE_WINDOW);
    expect(stake.unbond_eligible_at).to.equal(expectedFloor);

    // ---- Step 3: try to withdraw at T = UNBOND_DELAY + 1 (= 1 day + 1s).
    //              Pre-fix this would succeed. Post-fix it MUST revert
    //              because unbond_eligible_at is at the 7-day mark.
    await time.increase(UNBOND_DELAY + 1);
    await expect(
      stakeBond.connect(attacker).withdraw()
    ).to.be.revertedWithCustomError(stakeBond, "UnbondDelayNotElapsed");

    // Attacker still has the stake locked.
    const stillBonded = await stakeBond.stakes(attacker.address);
    expect(stillBonded.amount).to.equal(PREMIUM_STAKE);
    expect(stillBonded.status).to.equal(2); // UNBONDING

    // ---- Step 4: at T = UNBOND_DELAY + 1 (= 1 day + 1s), challenger
    //              discovers fraud and lands a DOUBLE_SPEND challenge.
    //              The slasher fires successfully because the stake is
    //              still UNBONDING (status==2), not WITHDRAWN.
    // sp1429 first-committer-wins: challenge the LATER duplicate (batchId2) citing the EARLIER
    // original (batchId1). Both are the attacker's, so this slashes the attacker either way — but
    // the guard only slashes the later duplicate (protecting the honest first committer), so the
    // conflict must be the strictly-earlier batch.
    const auxData = ethers.AbiCoder.defaultAbiCoder().encode(
      ["bytes32", "bytes32[]"],
      [batchId1, []]
    );

    const reserveBefore = await stakeBond.foundationReserveBalance();
    const bountyBefore = await stakeBond.slashedBountyPayable(challenger.address);

    await expect(
      registry.connect(challenger).challengeReceipt(
        batchId2, leaf, [], 0, auxData  // 0 = DOUBLE_SPEND
      )
    ).to.emit(stakeBond, "Slashed");

    // ---- Step 5: assert the slashing fired (the fix's economic
    //              defense). Reserve grew by 30%; bounty grew by 70%.
    const reserveAfter = await stakeBond.foundationReserveBalance();
    const bountyAfter = await stakeBond.slashedBountyPayable(challenger.address);
    const expectedSlash = PREMIUM_STAKE; // 100% slash rate
    const expectedBountyDelta = (expectedSlash * 7000n) / 10000n;
    const expectedReserveDelta = expectedSlash - expectedBountyDelta;

    expect(reserveAfter - reserveBefore).to.equal(expectedReserveDelta);
    expect(bountyAfter - bountyBefore).to.equal(expectedBountyDelta);

    // Stake fully drained by slash.
    const stakeFinal = await stakeBond.stakes(attacker.address);
    expect(stakeFinal.amount).to.equal(0n);
  });
});
