const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

// Phase 7-storage Task 3 — StorageSlashing.sol.
// Per docs/2026-04-22-phase7-storage-design-plan.md §6 Task 3.

describe("StorageSlashing — slasher for Phase 7-storage", function () {
  let slashing, mockStake;
  let owner, verifier, provider, challenger, anyone, other;

  const DEFAULT_GRACE = 7 * 24 * 60 * 60; // 7 days

  async function deploy(grace = DEFAULT_GRACE) {
    [owner, verifier, provider, challenger, anyone, other] =
      await ethers.getSigners();

    const Mock = await ethers.getContractFactory("MockStakeBondSlasher");
    mockStake = await Mock.deploy();

    const Slashing = await ethers.getContractFactory("StorageSlashing");
    slashing = await Slashing.deploy(
      await mockStake.getAddress(),
      verifier.address,
      grace,
      owner.address
    );

    return { slashing, mockStake };
  }

  const SHARD_ID =
    "0x1111111111111111111111111111111111111111111111111111111111111111";
  const EVIDENCE =
    "0x2222222222222222222222222222222222222222222222222222222222222222";

  // -------------------------------------------------------------------------
  // Construction
  // -------------------------------------------------------------------------

  describe("construction", function () {
    it("rejects zero StakeBond address", async function () {
      [owner, verifier] = await ethers.getSigners();
      const Slashing = await ethers.getContractFactory("StorageSlashing");
      await expect(
        Slashing.deploy(
          ethers.ZeroAddress,
          verifier.address,
          DEFAULT_GRACE,
          owner.address
        )
      ).to.be.revertedWithCustomError(Slashing, "InvalidAddress");
    });

    it("rejects zero verifier address", async function () {
      [owner] = await ethers.getSigners();
      const Mock = await ethers.getContractFactory("MockStakeBondSlasher");
      const m = await Mock.deploy();
      const Slashing = await ethers.getContractFactory("StorageSlashing");
      await expect(
        Slashing.deploy(
          await m.getAddress(),
          ethers.ZeroAddress,
          DEFAULT_GRACE,
          owner.address
        )
      ).to.be.revertedWithCustomError(Slashing, "InvalidAddress");
    });

    it("rejects grace below minimum", async function () {
      [owner, verifier] = await ethers.getSigners();
      const Mock = await ethers.getContractFactory("MockStakeBondSlasher");
      const m = await Mock.deploy();
      const Slashing = await ethers.getContractFactory("StorageSlashing");
      await expect(
        Slashing.deploy(
          await m.getAddress(),
          verifier.address,
          60, // < 1 hour
          owner.address
        )
      ).to.be.revertedWithCustomError(Slashing, "GraceOutOfRange");
    });

    it("rejects grace above maximum", async function () {
      [owner, verifier] = await ethers.getSigners();
      const Mock = await ethers.getContractFactory("MockStakeBondSlasher");
      const m = await Mock.deploy();
      const Slashing = await ethers.getContractFactory("StorageSlashing");
      await expect(
        Slashing.deploy(
          await m.getAddress(),
          verifier.address,
          31 * 24 * 60 * 60, // > 30 days
          owner.address
        )
      ).to.be.revertedWithCustomError(Slashing, "GraceOutOfRange");
    });
  });

  // -------------------------------------------------------------------------
  // Heartbeats
  // -------------------------------------------------------------------------

  describe("heartbeat", function () {
    it("records provider self-heartbeat", async function () {
      await deploy();
      await slashing.connect(provider).recordHeartbeat();
      const ts = await slashing.lastHeartbeat(provider.address);
      expect(ts).to.be.gt(0);
    });

    it("emits HeartbeatRecorded", async function () {
      await deploy();
      await expect(slashing.connect(provider).recordHeartbeat()).to.emit(
        slashing,
        "HeartbeatRecorded"
      );
    });

    it("each heartbeat overwrites the previous timestamp", async function () {
      await deploy();
      await slashing.connect(provider).recordHeartbeat();
      const t1 = await slashing.lastHeartbeat(provider.address);
      await time.increase(1000);
      await slashing.connect(provider).recordHeartbeat();
      const t2 = await slashing.lastHeartbeat(provider.address);
      expect(t2).to.be.gt(t1);
    });
  });

  // -------------------------------------------------------------------------
  // submitProofFailure
  // -------------------------------------------------------------------------

  describe("submitProofFailure", function () {
    it("rejects non-verifier caller", async function () {
      await deploy();
      await expect(
        slashing
          .connect(anyone)
          .submitProofFailure(
            provider.address,
            SHARD_ID,
            EVIDENCE,
            challenger.address
          )
      ).to.be.revertedWithCustomError(slashing, "NotAuthorizedVerifier");
    });

    it("rejects zero provider address", async function () {
      await deploy();
      await expect(
        slashing
          .connect(verifier)
          .submitProofFailure(
            ethers.ZeroAddress,
            SHARD_ID,
            EVIDENCE,
            challenger.address
          )
      ).to.be.revertedWithCustomError(slashing, "InvalidAddress");
    });

    it("calls stakeBond.slash with the correct arguments", async function () {
      await deploy();
      await slashing
        .connect(verifier)
        .submitProofFailure(
          provider.address,
          SHARD_ID,
          EVIDENCE,
          challenger.address
        );
      expect(await mockStake.slashCount()).to.equal(1);
      const rec = await mockStake.slashes(0);
      expect(rec.provider).to.equal(provider.address);
      expect(rec.challenger).to.equal(challenger.address);
    });

    it("emits ProofFailureSlashed with shardId + challenger + evidence", async function () {
      await deploy();
      await expect(
        slashing
          .connect(verifier)
          .submitProofFailure(
            provider.address,
            SHARD_ID,
            EVIDENCE,
            challenger.address
          )
      )
        .to.emit(slashing, "ProofFailureSlashed");
    });

    it("prevents double-slash on the same (provider, shardId, evidence)", async function () {
      await deploy();
      await slashing
        .connect(verifier)
        .submitProofFailure(
          provider.address,
          SHARD_ID,
          EVIDENCE,
          challenger.address
        );
      await expect(
        slashing
          .connect(verifier)
          .submitProofFailure(
            provider.address,
            SHARD_ID,
            EVIDENCE,
            challenger.address
          )
      ).to.be.revertedWithCustomError(slashing, "AlreadySlashed");
    });

    it("allows a second slash with different evidence hash", async function () {
      await deploy();
      const evidence2 =
        "0x3333333333333333333333333333333333333333333333333333333333333333";
      await slashing
        .connect(verifier)
        .submitProofFailure(
          provider.address,
          SHARD_ID,
          EVIDENCE,
          challenger.address
        );
      await slashing
        .connect(verifier)
        .submitProofFailure(
          provider.address,
          SHARD_ID,
          evidence2,
          challenger.address
        );
      expect(await mockStake.slashCount()).to.equal(2);
    });

    it("self-slash pattern: challenger == provider passes through (StakeBond routes to Foundation)", async function () {
      await deploy();
      await slashing
        .connect(verifier)
        .submitProofFailure(
          provider.address,
          SHARD_ID,
          EVIDENCE,
          provider.address
        );
      const rec = await mockStake.slashes(0);
      expect(rec.challenger).to.equal(rec.provider);
    });
  });

  // -------------------------------------------------------------------------
  // slashForMissingHeartbeat
  // -------------------------------------------------------------------------

  describe("slashForMissingHeartbeat", function () {
    it("reverts when no heartbeat has ever been recorded", async function () {
      await deploy();
      await expect(
        slashing.connect(anyone).slashForMissingHeartbeat(provider.address)
      ).to.be.revertedWithCustomError(slashing, "HeartbeatNotRecorded");
    });

    it("reverts when still within the grace period", async function () {
      await deploy();
      await slashing.connect(provider).recordHeartbeat();
      await time.increase(DEFAULT_GRACE - 100);
      await expect(
        slashing.connect(anyone).slashForMissingHeartbeat(provider.address)
      ).to.be.revertedWithCustomError(slashing, "HeartbeatNotExpired");
    });

    it("succeeds past the grace period — caller is credited as challenger", async function () {
      await deploy();
      await slashing.connect(provider).recordHeartbeat();
      await time.increase(2 * DEFAULT_GRACE + 100); // sp940 — slashable after 2 grace windows (default multiplier)

      await slashing
        .connect(anyone)
        .slashForMissingHeartbeat(provider.address);
      const rec = await mockStake.slashes(0);
      expect(rec.provider).to.equal(provider.address);
      expect(rec.challenger).to.equal(anyone.address);
    });

    it("prevents double-slash for the same heartbeat window", async function () {
      await deploy();
      await slashing.connect(provider).recordHeartbeat();
      await time.increase(2 * DEFAULT_GRACE + 100); // sp940 — slashable after 2 grace windows (default multiplier)
      await slashing
        .connect(anyone)
        .slashForMissingHeartbeat(provider.address);
      await expect(
        slashing.connect(other).slashForMissingHeartbeat(provider.address)
      ).to.be.revertedWithCustomError(slashing, "AlreadySlashed");
    });

    it("emits HeartbeatMissingSlashed", async function () {
      await deploy();
      await slashing.connect(provider).recordHeartbeat();
      await time.increase(2 * DEFAULT_GRACE + 100); // sp940 — slashable after 2 grace windows (default multiplier)
      await expect(
        slashing.connect(anyone).slashForMissingHeartbeat(provider.address)
      ).to.emit(slashing, "HeartbeatMissingSlashed");
    });

    // sp940 — honest-operator recovery buffer: one missed grace window is NOT
    // slashable (default multiplier = 2), so a transient blip can't be griefed.
    it("does NOT slash after only one grace window (sp940 buffer)", async function () {
      await deploy();
      expect(await slashing.slashGraceMultiplier()).to.equal(2n);
      await slashing.connect(provider).recordHeartbeat();
      await time.increase(DEFAULT_GRACE + 100); // past 1 window, within the 2x threshold
      await expect(
        slashing.connect(anyone).slashForMissingHeartbeat(provider.address)
      ).to.be.revertedWithCustomError(slashing, "HeartbeatNotExpired");
    });

    it("setSlashGraceMultiplier retunes the threshold (1x → slashable after one window)", async function () {
      await deploy();
      await slashing.connect(owner).setSlashGraceMultiplier(1);
      await slashing.connect(provider).recordHeartbeat();
      await time.increase(DEFAULT_GRACE + 100);
      await slashing.connect(anyone).slashForMissingHeartbeat(provider.address);
      const rec = await mockStake.slashes(0);
      expect(rec.provider).to.equal(provider.address);
    });

    it("setSlashGraceMultiplier rejects out-of-range + is onlyOwner", async function () {
      await deploy();
      await expect(
        slashing.connect(owner).setSlashGraceMultiplier(0)
      ).to.be.revertedWithCustomError(slashing, "MultiplierOutOfRange");
      await expect(
        slashing.connect(owner).setSlashGraceMultiplier(11)
      ).to.be.revertedWithCustomError(slashing, "MultiplierOutOfRange");
      await expect(
        slashing.connect(anyone).setSlashGraceMultiplier(3)
      ).to.be.reverted; // Ownable
    });
  });

  // -------------------------------------------------------------------------
  // Governance
  // -------------------------------------------------------------------------

  describe("governance", function () {
    it("setAuthorizedVerifier is onlyOwner", async function () {
      await deploy();
      await expect(
        slashing.connect(other).setAuthorizedVerifier(other.address)
      ).to.be.revertedWithCustomError(slashing, "OwnableUnauthorizedAccount");
    });

    it("setAuthorizedVerifier rejects zero address", async function () {
      await deploy();
      await expect(
        slashing.connect(owner).setAuthorizedVerifier(ethers.ZeroAddress)
      ).to.be.revertedWithCustomError(slashing, "InvalidAddress");
    });

    it("setAuthorizedVerifier emits event and takes effect", async function () {
      await deploy();
      await slashing.connect(owner).setAuthorizedVerifier(other.address);
      expect(await slashing.authorizedVerifier()).to.equal(other.address);

      // Old verifier is no longer authorized.
      await expect(
        slashing
          .connect(verifier)
          .submitProofFailure(
            provider.address,
            SHARD_ID,
            EVIDENCE,
            challenger.address
          )
      ).to.be.revertedWithCustomError(slashing, "NotAuthorizedVerifier");
    });

    it("setHeartbeatGrace validates bounds", async function () {
      await deploy();
      await expect(
        slashing.connect(owner).setHeartbeatGrace(60)
      ).to.be.revertedWithCustomError(slashing, "GraceOutOfRange");
      await expect(
        slashing.connect(owner).setHeartbeatGrace(31 * 24 * 60 * 60)
      ).to.be.revertedWithCustomError(slashing, "GraceOutOfRange");
    });

    it("setHeartbeatGrace is onlyOwner", async function () {
      await deploy();
      await expect(
        slashing.connect(other).setHeartbeatGrace(2 * 24 * 60 * 60)
      ).to.be.revertedWithCustomError(slashing, "OwnableUnauthorizedAccount");
    });
  });

  // -------------------------------------------------------------------------
  // sp1000 — slasher-wiring regression (against the REAL StakeBond).
  //
  // The rest of this suite uses MockStakeBondSlasher, whose slash() has NO
  // caller check, so it never exercises the on-chain reality: StakeBond's
  // slasher is IMMUTABLE (set once at construction, no setter per HIGH-7), and on
  // the deployed fleet it is the BatchSettlementRegistry — NOT StorageSlashing.
  // So every storage proof-failure / missing-heartbeat slash reverts with
  // CallerNotSlasher and the misbehaving provider is never penalized (the
  // storage-tier stake deterrent is non-functional). These tests pin that revert
  // against a real StakeBond + will verify the eventual on-chain fix (a StakeBond
  // authorized-slasher allowlist, or StorageSlashing-via-BSR-adapter — see
  // docs/2026-06-04-storage-slashing-slasher-wiring-gap.md).
  // -------------------------------------------------------------------------
  describe("slasher-wiring vs REAL StakeBond (sp1000)", function () {
    async function deployWithRealStakeBond() {
      const [o, ver, prov, chal, bsrStandin] = await ethers.getSigners();
      const Token = await ethers.getContractFactory("MockERC20");
      const token = await Token.deploy();
      await token.waitForDeployment();
      const StakeBond = await ethers.getContractFactory("StakeBond");
      // Immutable slasher = a stand-in for the BatchSettlementRegistry — anything
      // that is NOT the StorageSlashing we deploy next. 1-day unbond delay (min).
      const stakeBond = await StakeBond.deploy(
        o.address, await token.getAddress(), 24 * 60 * 60, bsrStandin.address,
      );
      await stakeBond.waitForDeployment();
      const Slashing = await ethers.getContractFactory("StorageSlashing");
      const sl = await Slashing.deploy(
        await stakeBond.getAddress(), ver.address, DEFAULT_GRACE, o.address,
      );
      await sl.waitForDeployment();
      return { sl, stakeBond, ver, prov, chal };
    }

    it("submitProofFailure reverts CallerNotSlasher (proof-failure slash never lands)", async function () {
      const { sl, stakeBond, ver, prov, chal } = await deployWithRealStakeBond();
      await expect(
        sl.connect(ver).submitProofFailure(
          prov.address, SHARD_ID, EVIDENCE, chal.address,
        )
      ).to.be.revertedWithCustomError(stakeBond, "CallerNotSlasher");
    });

    it("slashForMissingHeartbeat reverts CallerNotSlasher once the window elapses", async function () {
      const { sl, stakeBond, ver, prov } = await deployWithRealStakeBond();
      // Provider records its own heartbeat, then time advances past the slash
      // window (grace * slashGraceMultiplier(=2)) so the missing-heartbeat slash
      // is attempted (and reverts on the caller check).
      await sl.connect(prov).recordHeartbeat();
      await time.increase(DEFAULT_GRACE * 2 + 10);
      await expect(
        sl.connect(prov).slashForMissingHeartbeat(prov.address)
      ).to.be.revertedWithCustomError(stakeBond, "CallerNotSlasher");
    });
  });
});
