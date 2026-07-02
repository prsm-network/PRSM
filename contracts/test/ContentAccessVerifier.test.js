const { expect } = require("chai");
const { ethers } = require("hardhat");

// Sprint 1353 — ContentAccessVerifier: the production IRoyaltyPaymentVerifier for KeyDistribution
// (Tier B/C paid-decrypt). A consumer pays the release fee here; that records the payment so
// KeyDistribution.release confirms it, and credits the fee to the content's registered creator.
describe("ContentAccessVerifier — Tier B/C paid-access gate", function () {
  let ftns, registry, verifier, keyDist;
  let owner, creator, publisher, consumer, other;

  const CH = ethers.keccak256(ethers.toUtf8Bytes("nada-tier-b-dataset"));
  const FEE = ethers.parseUnits("1", 18); // 1 FTNS
  const ENCRYPTED_KEY = "0x" + "ab".repeat(48); // the wrapped content-key bytes

  async function deploy() {
    [owner, creator, publisher, consumer, other] = await ethers.getSigners();

    ftns = await (await ethers.getContractFactory("MockERC20")).deploy();
    registry = await (await ethers.getContractFactory("MockProvenanceRegistry")).deploy();
    verifier = await (await ethers.getContractFactory("ContentAccessVerifier")).deploy(
      await ftns.getAddress(), await registry.getAddress());

    await registry.setCreator(CH, creator.address, 10000); // creator owns the content
    await ftns.mint(consumer.address, ethers.parseUnits("100", 18));
    await ftns.connect(consumer).approve(await verifier.getAddress(), ethers.MaxUint256);
  }
  beforeEach(deploy);

  // ── payForAccess + verifyPayment ────────────────────────────────────────────

  it("records the payment, credits the creator, and emits", async function () {
    await expect(verifier.connect(consumer).payForAccess(CH, FEE))
      .to.emit(verifier, "AccessPaid")
      .withArgs(consumer.address, CH, FEE, creator.address);

    expect(await verifier.verifyPayment(consumer.address, CH, FEE)).to.equal(true);
    expect(await verifier.claimable(creator.address)).to.equal(FEE);
    expect(await verifier.totalClaimable()).to.equal(FEE);
    // the fee actually left the consumer + is held by the verifier
    expect(await ftns.balanceOf(await verifier.getAddress())).to.equal(FEE);
  });

  it("verifyPayment is true ONLY for the exact (payer, content, fee) tuple", async function () {
    await verifier.connect(consumer).payForAccess(CH, FEE);
    expect(await verifier.verifyPayment(consumer.address, CH, FEE)).to.equal(true);
    expect(await verifier.verifyPayment(other.address, CH, FEE)).to.equal(false); // wrong payer
    expect(await verifier.verifyPayment(consumer.address, CH, FEE + 1n)).to.equal(false); // wrong fee
    const CH2 = ethers.keccak256(ethers.toUtf8Bytes("other"));
    expect(await verifier.verifyPayment(consumer.address, CH2, FEE)).to.equal(false); // wrong content
  });

  it("reverts on unregistered content (fee can't be paid to a nonexistent creator)", async function () {
    const CH2 = ethers.keccak256(ethers.toUtf8Bytes("unregistered"));
    await expect(verifier.connect(consumer).payForAccess(CH2, FEE))
      .to.be.revertedWithCustomError(verifier, "ContentNotRegistered");
  });

  it("reverts on zero fee", async function () {
    await expect(verifier.connect(consumer).payForAccess(CH, 0))
      .to.be.revertedWithCustomError(verifier, "ZeroFee");
  });

  it("reverts (no state change) when the consumer hasn't funded/approved", async function () {
    await expect(verifier.connect(other).payForAccess(CH, FEE)).to.be.reverted; // no balance/approval
    expect(await verifier.verifyPayment(other.address, CH, FEE)).to.equal(false);
  });

  // ── claim (pull payment) ────────────────────────────────────────────────────

  it("creator claims the accumulated fees; non-creator + double-claim revert", async function () {
    await verifier.connect(consumer).payForAccess(CH, FEE);
    const before = await ftns.balanceOf(creator.address);

    await expect(verifier.connect(creator).claim())
      .to.emit(verifier, "RoyaltyClaimed").withArgs(creator.address, FEE);
    expect(await ftns.balanceOf(creator.address)).to.equal(before + FEE);
    expect(await verifier.claimable(creator.address)).to.equal(0n);
    expect(await verifier.totalClaimable()).to.equal(0n);

    await expect(verifier.connect(creator).claim())
      .to.be.revertedWithCustomError(verifier, "NothingToClaim"); // already withdrawn
    await expect(verifier.connect(other).claim())
      .to.be.revertedWithCustomError(verifier, "NothingToClaim"); // never a payee
  });

  // ── ★ end-to-end: KeyDistribution.release is gated on payForAccess ──────────

  it("★ gates a real KeyDistribution key release on payment", async function () {
    keyDist = await (await ethers.getContractFactory("KeyDistribution")).deploy(owner.address);
    // publisher deposits the wrapped key, naming THIS verifier + the fee
    await keyDist.connect(publisher).depositKey(
      CH, ENCRYPTED_KEY, await verifier.getAddress(), FEE);

    // before payment: release reverts (PaymentNotVerified)
    await expect(keyDist.connect(consumer).release(CH, consumer.address)).to.be.reverted;

    // consumer pays the exact fee here
    await verifier.connect(consumer).payForAccess(CH, FEE);

    // now the key releases to the consumer, surfacing the wrapped key in the event
    await expect(keyDist.connect(consumer).release(CH, consumer.address))
      .to.emit(keyDist, "KeyReleased")
      .withArgs(CH, consumer.address, ENCRYPTED_KEY);
  });
});
