const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

// Phase 7 Task 3 — BatchSettlementRegistry wires challenge-success
// paths to call StakeBond.slash. Only DOUBLE_SPEND and INVALID_SIGNATURE
// trigger slashing; NO_ESCROW (griefing risk) and EXPIRED (hygiene)
// invalidate the receipt but do NOT touch stake.

describe("BatchSettlementRegistry — Phase 7 slashing integration", function () {
  let registry, escrowPool, token, verifier, stakeBond;
  let owner, provider, requester, challenger, foundation, other;
  const ONE_FTNS = ethers.parseUnits("1", 18);
  const DEFAULT_WINDOW = 3 * 24 * 60 * 60;
  const DEFAULT_UNBOND_DELAY = 7 * 24 * 60 * 60;
  const PREMIUM_STAKE = 25_000n * ONE_FTNS;
  const PREMIUM_SLASH_BPS = 10000; // 100%
  const STANDARD_SLASH_BPS = 5000; // 50%

  function makeLeaf(overrides = {}) {
    return {
      jobIdHash: ethers.keccak256(ethers.toUtf8Bytes("job-1")),
      shardIndex: 0,
      providerIdHash: ethers.keccak256(ethers.toUtf8Bytes("provider-id")),
      providerPubkeyHash: ethers.keccak256(ethers.toUtf8Bytes("pubkey")),
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("output")),
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
    [owner, provider, requester, challenger, foundation, other] =
      await ethers.getSigners();

    const Token = await ethers.getContractFactory("MockERC20");
    token = await Token.deploy();
    await token.waitForDeployment();

    const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
    registry = await Registry.deploy(owner.address, DEFAULT_WINDOW);
    await registry.waitForDeployment();

    const Pool = await ethers.getContractFactory("EscrowPool");
    escrowPool = await Pool.deploy(
      owner.address, await token.getAddress(), await registry.getAddress()
    );
    await escrowPool.waitForDeployment();

    await registry.connect(owner).setEscrowPool(await escrowPool.getAddress());

    const Verifier = await ethers.getContractFactory("MockSignatureVerifier");
    verifier = await Verifier.deploy();
    await verifier.waitForDeployment();
    await registry.connect(owner).setSignatureVerifier(await verifier.getAddress());

    // Phase 7: deploy StakeBond with the registry as immutable slasher
    // (HIGH-7). Registry already deployed above.
    const Bond = await ethers.getContractFactory("StakeBond");
    stakeBond = await Bond.deploy(
      owner.address, await token.getAddress(), DEFAULT_UNBOND_DELAY, await registry.getAddress()
    );
    await stakeBond.waitForDeployment();

    // L4 self-audit MED-4: foundation wallet must be a contract.
    const FoundationStub = await ethers.getContractFactory("MockERC20");
    const foundationStub = await FoundationStub.deploy();
    await foundationStub.waitForDeployment();
    await stakeBond.connect(owner).setFoundationReserveWallet(await foundationStub.getAddress());
    await registry.connect(owner).setStakeBond(await stakeBond.getAddress());

    // Fund + approve the requester's escrow.
    await token.mint(requester.address, ONE_FTNS * 1000n);
    await token.connect(requester).approve(
      await escrowPool.getAddress(), ONE_FTNS * 1000n
    );
    await escrowPool.connect(requester).deposit(ONE_FTNS * 1000n);

    // Fund the provider + bond premium stake.
    await token.mint(provider.address, PREMIUM_STAKE * 2n);
    await token.connect(provider).approve(
      await stakeBond.getAddress(), PREMIUM_STAKE * 2n
    );
    await stakeBond.connect(provider).bond(PREMIUM_STAKE, PREMIUM_SLASH_BPS);
  });

  async function commitOneLeafBatch(leaf, tierSlashRateBps = PREMIUM_SLASH_BPS, metadata = "") {
    const root = hashLeaf(leaf);
    const tx = await registry.connect(provider).commitBatch(
      requester.address, root, 1, leaf.valueFtns, tierSlashRateBps, ethers.ZeroHash, metadata
    );
    const r = await tx.wait();
    return r.logs.find(
      (l) => l.fragment && l.fragment.name === "BatchCommitted"
    ).args[0];
  }

  describe("DOUBLE_SPEND triggers slash", function () {
    it("slashes the provider on successful DOUBLE_SPEND challenge", async function () {
      const sharedLeaf = makeLeaf({ valueFtns: ONE_FTNS });
      const root = hashLeaf(sharedLeaf);

      const tx1 = await registry.connect(provider).commitBatch(requester.address, root, 1, sharedLeaf.valueFtns, PREMIUM_SLASH_BPS, ethers.ZeroHash, "b1");
      const id1 = (await tx1.wait()).logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];

      const tx2 = await registry.connect(provider).commitBatch(requester.address, root, 1, sharedLeaf.valueFtns, PREMIUM_SLASH_BPS, ethers.ZeroHash, "b2");
      const id2 = (await tx2.wait()).logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];

      const aux = ethers.AbiCoder.defaultAbiCoder().encode(
        ["bytes32", "bytes32[]"], [id1, []]
      );

      // Challenge id2 citing id1 as the conflicting batch.
      const stakeBefore = await stakeBond.stakeOf(provider.address);
      expect(stakeBefore.amount).to.equal(PREMIUM_STAKE);

      await expect(
        registry.connect(challenger).challengeReceipt(id2, sharedLeaf, [], 0, aux)
      ).to.emit(stakeBond, "Slashed");

      // 100% slash of premium stake.
      const stakeAfter = await stakeBond.stakeOf(provider.address);
      expect(stakeAfter.amount).to.equal(0);

      // Bounty credited to challenger (70%).
      const expectedBounty = (PREMIUM_STAKE * 7000n) / 10000n;
      expect(await stakeBond.slashedBountyPayable(challenger.address)).to.equal(expectedBounty);

      // Foundation reserve credited (30%).
      const expectedFoundation = PREMIUM_STAKE - expectedBounty;
      expect(await stakeBond.foundationReserveBalance()).to.equal(expectedFoundation);
    });

    it("slashes at the committed tier rate, not the current stake rate", async function () {
      // Provider bonds at PREMIUM (100%) then commits a batch at STANDARD
      // (50%) rate — e.g., intentional tier-downgrade in the batch to try
      // dodging full slash. Snapshot at commit time still uses STANDARD.
      const sharedLeaf = makeLeaf();
      const root = hashLeaf(sharedLeaf);

      const tx1 = await registry.connect(provider).commitBatch(requester.address, root, 1, sharedLeaf.valueFtns, STANDARD_SLASH_BPS, ethers.ZeroHash, "b1");
      const id1 = (await tx1.wait()).logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];
      const tx2 = await registry.connect(provider).commitBatch(requester.address, root, 1, sharedLeaf.valueFtns, STANDARD_SLASH_BPS, ethers.ZeroHash, "b2");
      const id2 = (await tx2.wait()).logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];

      const aux = ethers.AbiCoder.defaultAbiCoder().encode(
        ["bytes32", "bytes32[]"], [id1, []]
      );
      await registry.connect(challenger).challengeReceipt(id2, sharedLeaf, [], 0, aux);

      // Provider's bond's tier_slash_rate_bps snapshot was PREMIUM at bond time.
      // StakeBond.slash uses the BOND's tier, not the batch's tier. So
      // 100% slash regardless of the batch's committed rate.
      // (Design §3.2: slash rate determined by the tier the provider
      //  claimed AT BOND TIME, snapshotted into the stake.)
      const stake = await stakeBond.stakeOf(provider.address);
      expect(stake.amount).to.equal(0);
    });
  });

  describe("INVALID_SIGNATURE triggers slash", function () {
    it("slashes when the verifier returns false", async function () {
      await verifier.setResult(false); // signature "invalid"

      const pubkey = ethers.toUtf8Bytes("pubkey-bytes");
      const signature = ethers.toUtf8Bytes("signature-bytes");
      const leaf = makeLeaf({
        providerPubkeyHash: ethers.keccak256(pubkey),
        signatureHash: ethers.keccak256(signature),
      });
      const batchId = await commitOneLeafBatch(leaf);

      const aux = ethers.AbiCoder.defaultAbiCoder().encode(
        ["bytes", "bytes"],
        [pubkey, signature]
      );

      const stakeBefore = await stakeBond.stakeOf(provider.address);
      expect(stakeBefore.amount).to.equal(PREMIUM_STAKE);

      await expect(
        registry.connect(challenger).challengeReceipt(batchId, leaf, [], 1, aux)
      ).to.emit(stakeBond, "Slashed");

      const stakeAfter = await stakeBond.stakeOf(provider.address);
      expect(stakeAfter.amount).to.equal(0);
    });
  });

  describe("NO_ESCROW does NOT trigger slash", function () {
    it("invalidates the receipt but leaves stake untouched", async function () {
      const leaf = makeLeaf();
      const batchId = await commitOneLeafBatch(leaf);

      const stakeBefore = await stakeBond.stakeOf(provider.address);
      expect(stakeBefore.amount).to.equal(PREMIUM_STAKE);

      // Requester attests NO_ESCROW.
      await registry.connect(requester).challengeReceipt(batchId, leaf, [], 2, "0x");

      // Receipt invalidated...
      const batch = await registry.getBatch(batchId);
      expect(batch.invalidatedValueFTNS).to.equal(leaf.valueFtns);

      // ...but stake untouched.
      const stakeAfter = await stakeBond.stakeOf(provider.address);
      expect(stakeAfter.amount).to.equal(PREMIUM_STAKE);
      expect(await stakeBond.foundationReserveBalance()).to.equal(0);
    });
  });

  describe("EXPIRED does NOT trigger slash", function () {
    it("invalidates an old receipt but leaves stake untouched", async function () {
      const blockTime = await time.latest();
      const oldTimestamp = blockTime - 30 * 24 * 60 * 60 - 60 * 60; // >30d old
      const leaf = makeLeaf({ executedAtUnix: oldTimestamp });
      const batchId = await commitOneLeafBatch(leaf);

      await registry.connect(challenger).challengeReceipt(batchId, leaf, [], 3, "0x");

      const stake = await stakeBond.stakeOf(provider.address);
      expect(stake.amount).to.equal(PREMIUM_STAKE); // untouched
    });
  });

  describe("slashing is best-effort — disabled scenarios preserve receipt invalidation", function () {
    it("does not revert when stakeBond is unset", async function () {
      // Disconnect the stakeBond.
      await registry.connect(owner).setStakeBond(ethers.ZeroAddress);

      const sharedLeaf = makeLeaf();
      const root = hashLeaf(sharedLeaf);
      const tx1 = await registry.connect(provider).commitBatch(requester.address, root, 1, sharedLeaf.valueFtns, PREMIUM_SLASH_BPS, ethers.ZeroHash, "b1");
      const id1 = (await tx1.wait()).logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];
      const tx2 = await registry.connect(provider).commitBatch(requester.address, root, 1, sharedLeaf.valueFtns, PREMIUM_SLASH_BPS, ethers.ZeroHash, "b2");
      const id2 = (await tx2.wait()).logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];

      const aux = ethers.AbiCoder.defaultAbiCoder().encode(
        ["bytes32", "bytes32[]"], [id1, []]
      );

      // Challenge succeeds; receipt invalidated; no slash.
      await expect(
        registry.connect(challenger).challengeReceipt(id2, sharedLeaf, [], 0, aux)
      ).to.emit(registry, "ReceiptChallenged");

      const batch = await registry.getBatch(id2);
      expect(batch.invalidatedValueFTNS).to.equal(sharedLeaf.valueFtns);
      // Stake untouched.
      const stake = await stakeBond.stakeOf(provider.address);
      expect(stake.amount).to.equal(PREMIUM_STAKE);
    });

    it("sp1456: still slashes the provider's BOND when the batch tier_slash_rate_bps is 0", async function () {
      // A PROPERLY-BONDED provider (PREMIUM_STAKE @ PREMIUM_SLASH_BPS in beforeEach) commits the
      // fraudulent batch with a self-declared tier_slash_rate_bps = 0. Pre-sp1456 the challengeReceipt
      // gate `b.tier_slash_rate_bps > 0` skipped the slash ENTIRELY, so a bonded provider could
      // self-disable slashing by committing at rate 0 — a full slash-evasion. The slash AMOUNT is
      // computed from the provider's ON-CHAIN bonded rate (not the caller-supplied batch rate), so the
      // challenge must now slash the bond regardless of the batch's declared rate.
      const sharedLeaf = makeLeaf();
      const root = hashLeaf(sharedLeaf);

      const tx1 = await registry.connect(provider).commitBatch(requester.address, root, 1, sharedLeaf.valueFtns, 0, ethers.ZeroHash, "b1");
      const id1 = (await tx1.wait()).logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];
      const tx2 = await registry.connect(provider).commitBatch(requester.address, root, 1, sharedLeaf.valueFtns, 0, ethers.ZeroHash, "b2");
      const id2 = (await tx2.wait()).logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];

      const aux = ethers.AbiCoder.defaultAbiCoder().encode(
        ["bytes32", "bytes32[]"], [id1, []]
      );

      await registry.connect(challenger).challengeReceipt(id2, sharedLeaf, [], 0, aux);
      // sp1456: the bond IS slashed at the BOND's rate (100% for premium) — no longer swallowed.
      const stake = await stakeBond.stakeOf(provider.address);
      expect(stake.amount).to.equal(0);
    });

    it("does not revert when provider has no stake", async function () {
      // Provider without any stake commits a batch (incorrectly claiming
      // a tier). Challenge invalidates receipt; slash attempt fails
      // inside try/catch; challenge itself succeeds.
      const freshProvider = other;
      const leaf = makeLeaf();
      const root = hashLeaf(leaf);

      const tx1 = await registry.connect(freshProvider).commitBatch(requester.address, root, 1, leaf.valueFtns, PREMIUM_SLASH_BPS, ethers.ZeroHash, "b1");
      const id1 = (await tx1.wait()).logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];
      const tx2 = await registry.connect(freshProvider).commitBatch(requester.address, root, 1, leaf.valueFtns, PREMIUM_SLASH_BPS, ethers.ZeroHash, "b2");
      const id2 = (await tx2.wait()).logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];

      const aux = ethers.AbiCoder.defaultAbiCoder().encode(
        ["bytes32", "bytes32[]"], [id1, []]
      );

      await expect(
        registry.connect(challenger).challengeReceipt(id2, leaf, [], 0, aux)
      ).to.emit(registry, "ReceiptChallenged");

      const batch = await registry.getBatch(id2);
      expect(batch.invalidatedValueFTNS).to.equal(leaf.valueFtns);
    });
  });

  describe("governance: setStakeBond", function () {
    it("owner can update the stake bond address (must be a contract)", async function () {
      // L4 self-audit MED-7: setter rejects EOA targets.
      const Stub = await ethers.getContractFactory("MockERC20");
      const stub = await Stub.deploy();
      await stub.waitForDeployment();
      const newAddr = await stub.getAddress();
      await expect(registry.connect(owner).setStakeBond(newAddr))
        .to.emit(registry, "StakeBondUpdated")
        .withArgs(await stakeBond.getAddress(), newAddr);
      expect(await registry.stakeBond()).to.equal(newAddr);
    });

    it("non-owner cannot update the stake bond address", async function () {
      await expect(
        registry.connect(provider).setStakeBond(other.address)
      ).to.be.reverted;
    });

    it("REGRESSION (L4 self-audit MED-7 / D-03): rejects EOA target", async function () {
      await expect(
        registry.connect(owner).setStakeBond(other.address)
      ).to.be.revertedWithCustomError(registry, "SetterTargetNotContract");
    });
  });

  describe("commitBatch validates tier_slash_rate_bps bounds", function () {
    it("reverts on rate > 10000 bps", async function () {
      await expect(
        registry.connect(provider).commitBatch(requester.address,
          ethers.keccak256(ethers.toUtf8Bytes("x")),
          1, ONE_FTNS, 10001, ethers.ZeroHash, "")
      ).to.be.revertedWithCustomError(registry, "InvalidSlashRateBps");
    });
  });
});
