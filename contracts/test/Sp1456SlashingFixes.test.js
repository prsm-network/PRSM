const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

// sp1456 — fixes for the 3 slashing-evasion defects the StakeBond audit (workflow wu7qhwsz3) found.
// Each test reproduces the confirmed exploit and asserts it is now blocked. DEPLOY-GATED: these fix
// deployed Base-mainnet contracts; a governance redeploy ceremony is required to activate them.
describe("sp1456 — StakeBond slashing-evasion fixes", function () {
  const ONE = ethers.parseUnits("1", 18);
  const OPEN_STAKE = 100n * ONE;          // < 5k → open tier, no rate floor
  const STANDARD_STAKE = 5_000n * ONE;    // ≥5k → floor 5000
  const PREMIUM_STAKE = 25_000n * ONE;    // ≥25k → floor 5000
  const CRITICAL_STAKE = 50_000n * ONE;   // ≥50k → floor 10000
  const UNBOND_DELAY = 24 * 60 * 60;      // 1 day
  const CHALLENGE_WINDOW = 3 * 24 * 60 * 60; // 3 days

  // ── #1a — bond() binds the slash rate to a tier-derived floor ──────────────
  describe("#1 bond-side — slash rate floored by tier (rate-0-at-critical evasion)", function () {
    let bond, token, owner, provider, slasher;
    beforeEach(async function () {
      [owner, provider, slasher] = await ethers.getSigners();
      const Token = await ethers.getContractFactory("MockERC20");
      token = await Token.deploy();
      const Bond = await ethers.getContractFactory("StakeBond");
      bond = await Bond.deploy(owner.address, await token.getAddress(), UNBOND_DELAY, slasher.address);
      await token.mint(provider.address, CRITICAL_STAKE * 3n);
      await token.connect(provider).approve(await bond.getAddress(), CRITICAL_STAKE * 3n);
    });

    it("EXPLOIT BLOCKED: a critical-tier stake CANNOT bond at rate 0 (was fully unslashable)", async function () {
      await expect(bond.connect(provider).bond(CRITICAL_STAKE, 0))
        .to.be.revertedWithCustomError(bond, "SlashRateBelowTierFloor").withArgs(0, 10000);
    });
    it("a critical-tier stake cannot bond below 100% (10000 bps)", async function () {
      await expect(bond.connect(provider).bond(CRITICAL_STAKE, 5000))
        .to.be.revertedWithCustomError(bond, "SlashRateBelowTierFloor").withArgs(5000, 10000);
    });
    it("a standard/premium-tier stake cannot bond below 50% (5000 bps)", async function () {
      await expect(bond.connect(provider).bond(STANDARD_STAKE, 0))
        .to.be.revertedWithCustomError(bond, "SlashRateBelowTierFloor").withArgs(0, 5000);
      await expect(bond.connect(provider).bond(PREMIUM_STAKE, 4999))
        .to.be.revertedWithCustomError(bond, "SlashRateBelowTierFloor").withArgs(4999, 5000);
    });
    it("legitimate bonds at/above the tier floor still succeed", async function () {
      await expect(bond.connect(provider).bond(CRITICAL_STAKE, 10000)).to.not.be.reverted;
    });
    it("open-tier (<5k) stake may still bond at rate 0 — untrusted, no tier privileges", async function () {
      await expect(bond.connect(provider).bond(OPEN_STAKE, 0)).to.not.be.reverted;
      expect(await bond.effectiveTier(provider.address)).to.equal("open");
    });
    it("minSlashRateForAmount mirrors the tier thresholds", async function () {
      expect(await bond.minSlashRateForAmount(CRITICAL_STAKE)).to.equal(10000);
      expect(await bond.minSlashRateForAmount(PREMIUM_STAKE)).to.equal(5000);
      expect(await bond.minSlashRateForAmount(STANDARD_STAKE)).to.equal(5000);
      expect(await bond.minSlashRateForAmount(OPEN_STAKE)).to.equal(0);
    });
  });

  // ── #2 — storageSlasher set-once authorization ─────────────────────────────
  describe("#2 storage-fault slasher authorization (storage slashes were all reverting)", function () {
    let bond, token, owner, provider, bsrSlasher, storageSlasher, attacker, challenger;
    beforeEach(async function () {
      [owner, provider, bsrSlasher, storageSlasher, attacker, challenger] = await ethers.getSigners();
      const Token = await ethers.getContractFactory("MockERC20");
      token = await Token.deploy();
      const Bond = await ethers.getContractFactory("StakeBond");
      bond = await Bond.deploy(owner.address, await token.getAddress(), UNBOND_DELAY, bsrSlasher.address);
      await token.mint(provider.address, CRITICAL_STAKE);
      await token.connect(provider).approve(await bond.getAddress(), CRITICAL_STAKE);
      await bond.connect(provider).bond(CRITICAL_STAKE, 10000);
    });

    it("owner wires the storage slasher exactly once; re-point is rejected", async function () {
      await expect(bond.connect(owner).setStorageSlasherOnce(storageSlasher.address))
        .to.emit(bond, "StorageSlasherSet").withArgs(storageSlasher.address);
      await expect(bond.connect(owner).setStorageSlasherOnce(attacker.address))
        .to.be.revertedWithCustomError(bond, "StorageSlasherAlreadySet");
      expect(await bond.storageSlasher()).to.equal(storageSlasher.address);
    });
    it("only the owner can wire it, and not to address(0)", async function () {
      await expect(bond.connect(attacker).setStorageSlasherOnce(storageSlasher.address)).to.be.reverted;
      await expect(bond.connect(owner).setStorageSlasherOnce(ethers.ZeroAddress))
        .to.be.revertedWithCustomError(bond, "ZeroAddress");
    });
    it("EXPLOIT BLOCKED: after wiring, the storage slasher CAN slash (was CallerNotSlasher revert)", async function () {
      // Before wiring: the storage slasher is unauthorized (only the immutable BSR slasher is).
      await expect(bond.connect(storageSlasher).slash(provider.address, challenger.address, ethers.ZeroHash))
        .to.be.revertedWithCustomError(bond, "CallerNotSlasher");
      await bond.connect(owner).setStorageSlasherOnce(storageSlasher.address);
      // After wiring: the storage slash succeeds and forfeits 100% of the critical bond.
      await expect(bond.connect(storageSlasher).slash(provider.address, challenger.address, ethers.ZeroHash))
        .to.emit(bond, "Slashed");
      expect((await bond.stakeOf(provider.address)).amount).to.equal(0);
    });
    it("a random address still cannot slash", async function () {
      await bond.connect(owner).setStorageSlasherOnce(storageSlasher.address);
      await expect(bond.connect(attacker).slash(provider.address, challenger.address, ethers.ZeroHash))
        .to.be.revertedWithCustomError(bond, "CallerNotSlasher");
    });
  });

  // ── #3 — withdraw() re-clamps against batches committed after requestUnbond ─
  describe("#3 withdraw re-clamp (commit-after-requestUnbond evasion, real BSR slasher)", function () {
    let registry, bond, token, owner, provider;
    beforeEach(async function () {
      [owner, provider] = await ethers.getSigners();
      const Token = await ethers.getContractFactory("MockERC20");
      token = await Token.deploy();
      const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
      registry = await Registry.deploy(owner.address, CHALLENGE_WINDOW);
      const Bond = await ethers.getContractFactory("StakeBond");
      bond = await Bond.deploy(owner.address, await token.getAddress(), UNBOND_DELAY, await registry.getAddress());
      await token.mint(provider.address, PREMIUM_STAKE);
      await token.connect(provider).approve(await bond.getAddress(), PREMIUM_STAKE);
      await bond.connect(provider).bond(PREMIUM_STAKE, 10000);
    });

    it("EXPLOIT BLOCKED: cannot withdraw while a batch committed AFTER requestUnbond is still challengeable", async function () {
      await bond.connect(provider).requestUnbond();
      const eligibleAt = (await bond.stakeOf(provider.address)).unbond_eligible_at;

      // Commit a (to-be-fraudulent) batch AFTER requestUnbond — raises lastPendingBatchExpiry well past
      // the frozen unbond_eligible_at. In the OLD code this batch's window was never re-clamped.
      await time.increase(2 * 24 * 60 * 60); // +2 days (still before eligibility at ~3 days)
      await registry.connect(provider).commitBatch(
        owner.address, ethers.keccak256(ethers.toUtf8Bytes("fraud-root")),
        1, ONE, 10000, ethers.ZeroHash, "ipfs://x");
      const batchExpiry = await registry.lastPendingBatchExpiry(provider.address);
      expect(batchExpiry).to.be.gt(eligibleAt); // the batch outlives the frozen snapshot

      // Advance PAST the frozen unbond_eligible_at but BEFORE the batch's challenge window closes.
      await time.increaseTo(eligibleAt + 10n);
      // OLD code: block.timestamp >= unbond_eligible_at → withdraw succeeds (slash later swallowed).
      // NEW code: re-clamp against the live pending-expiry floor → revert.
      await expect(bond.connect(provider).withdraw())
        .to.be.revertedWithCustomError(bond, "UnbondDelayNotElapsed");

      // Once the batch's window has fully elapsed, withdraw is allowed.
      await time.increaseTo(batchExpiry + 10n);
      await expect(bond.connect(provider).withdraw()).to.not.be.reverted;
    });

    it("control: with no batch committed after requestUnbond, withdraw succeeds at eligibility", async function () {
      await bond.connect(provider).requestUnbond();
      const eligibleAt = (await bond.stakeOf(provider.address)).unbond_eligible_at;
      await time.increaseTo(eligibleAt + 10n);
      await expect(bond.connect(provider).withdraw()).to.not.be.reverted;
    });
  });
});
