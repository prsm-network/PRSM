const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

// Sprint 976 — CreatorStakeRegistry: the §14 anti-spam creator stake gets
// on-chain teeth. A creator bonds real FTNS as collateral; spam → slash. Stake
// counts toward HIGH-tier eligibility ONLY while BONDED (requesting unbond drops
// eligibility immediately, closing the stake→tier→spam→unstake game). Mirrors
// StakeBond's security patterns (Ownable2Step, ReentrancyGuard, Pausable,
// immutable ftns + slasher, transferFrom-on-stake, unbond-delay withdraw,
// slasher-gated slash → Foundation reserve).

describe("CreatorStakeRegistry", function () {
  let reg, token;
  let owner, creator, other, slasher, reserve;
  const ONE = ethers.parseUnits("1", 18);
  const DELAY = 7 * 24 * 60 * 60; // 7 days
  const MIN_HIGH = 1000n * ONE;

  beforeEach(async function () {
    [owner, creator, other, slasher, reserve] = await ethers.getSigners();
    const Token = await ethers.getContractFactory("MockERC20");
    token = await Token.deploy();
    await token.waitForDeployment();
    const Reg = await ethers.getContractFactory("CreatorStakeRegistry");
    reg = await Reg.deploy(
      owner.address, await token.getAddress(), DELAY, slasher.address
    );
    await reg.waitForDeployment();
    await token.mint(creator.address, 10_000n * ONE);
    await token.connect(creator).approve(await reg.getAddress(), 10_000n * ONE);
  });

  describe("constructor", function () {
    it("sets owner, token, delay, slasher", async function () {
      expect(await reg.owner()).to.equal(owner.address);
      expect(await reg.ftns()).to.equal(await token.getAddress());
      expect(await reg.unbondDelaySeconds()).to.equal(DELAY);
      expect(await reg.slasher()).to.equal(slasher.address);
    });
  });

  describe("stake", function () {
    it("pulls FTNS via transferFrom and records the bonded balance", async function () {
      await expect(reg.connect(creator).stake(MIN_HIGH))
        .to.emit(reg, "CreatorStaked");
      expect(await reg.creatorStakeOf(creator.address)).to.equal(MIN_HIGH);
      expect(await token.balanceOf(await reg.getAddress())).to.equal(MIN_HIGH);
    });

    it("accumulates across multiple stakes", async function () {
      await reg.connect(creator).stake(600n * ONE);
      await reg.connect(creator).stake(500n * ONE);
      expect(await reg.creatorStakeOf(creator.address)).to.equal(1100n * ONE);
    });

    it("reverts on zero amount", async function () {
      await expect(reg.connect(creator).stake(0)).to.be.reverted;
    });
  });

  describe("unbond + withdraw", function () {
    beforeEach(async function () {
      await reg.connect(creator).stake(MIN_HIGH);
    });

    it("creatorStakeOf drops to 0 immediately on requestUnbond (no tier while exiting)", async function () {
      await reg.connect(creator).requestUnbond();
      expect(await reg.creatorStakeOf(creator.address)).to.equal(0n);
    });

    it("withdraw before delay reverts; succeeds after delay and returns funds", async function () {
      await reg.connect(creator).requestUnbond();
      await expect(reg.connect(creator).withdraw()).to.be.reverted;
      await time.increase(DELAY + 1);
      const before = await token.balanceOf(creator.address);
      await reg.connect(creator).withdraw();
      expect(await token.balanceOf(creator.address)).to.equal(before + MIN_HIGH);
    });

    it("withdraw without requestUnbond reverts", async function () {
      await expect(reg.connect(creator).withdraw()).to.be.reverted;
    });
  });

  describe("slash", function () {
    beforeEach(async function () {
      await reg.connect(creator).stake(MIN_HIGH);
    });

    it("slasher reduces the creator's bonded stake and credits the Foundation reserve", async function () {
      await expect(reg.connect(slasher).slash(creator.address, 400n * ONE, "spam"))
        .to.emit(reg, "CreatorSlashed");
      expect(await reg.creatorStakeOf(creator.address)).to.equal(600n * ONE);
      expect(await reg.foundationReserveBalance()).to.equal(400n * ONE);
    });

    it("non-slasher cannot slash", async function () {
      await expect(
        reg.connect(other).slash(creator.address, 100n * ONE, "x")
      ).to.be.reverted;
      await expect(
        reg.connect(owner).slash(creator.address, 100n * ONE, "x")
      ).to.be.reverted;
    });

    it("cannot slash more than the bonded amount", async function () {
      await expect(
        reg.connect(slasher).slash(creator.address, MIN_HIGH + 1n, "x")
      ).to.be.reverted;
    });
  });

  describe("foundation reserve drain", function () {
    it("owner drains accrued slash proceeds to the reserve wallet", async function () {
      await reg.connect(creator).stake(MIN_HIGH);
      await reg.connect(slasher).slash(creator.address, 400n * ONE, "spam");
      await reg.connect(owner).setFoundationReserveWallet(reserve.address);
      const before = await token.balanceOf(reserve.address);
      await reg.connect(owner).drainFoundationReserve();
      expect(await token.balanceOf(reserve.address)).to.equal(before + 400n * ONE);
      expect(await reg.foundationReserveBalance()).to.equal(0n);
    });

    it("non-owner cannot drain", async function () {
      await expect(reg.connect(other).drainFoundationReserve()).to.be.reverted;
    });
  });

  describe("pausable", function () {
    it("owner can pause; staking reverts while paused", async function () {
      await reg.connect(owner).pause();
      await expect(reg.connect(creator).stake(MIN_HIGH)).to.be.reverted;
      await reg.connect(owner).unpause();
      await reg.connect(creator).stake(MIN_HIGH);
      expect(await reg.creatorStakeOf(creator.address)).to.equal(MIN_HIGH);
    });
  });
});
