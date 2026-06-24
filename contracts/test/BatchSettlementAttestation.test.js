// Sprint 1240 (TEE Tier-3 roadmap F) — on-chain attestation commitment.
// commitBatchWithAttestation records a bytes32 commitment to a batch's TEE
// attestation tier/measurement as ADDITIVE metadata: it must NOT change batchId,
// merkleRoot, or any challenge/finalize/slash semantics (sp1238 keeps attestation
// out of the leaf). Legacy commitBatch stays byte-for-byte unchanged.
const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

describe("BatchSettlementRegistry — attestation commitment (sp1240)", function () {
  let registry, escrowPool, token, owner, provider, requester;
  const ONE_FTNS = ethers.parseUnits("1", 18);
  const DEFAULT_WINDOW = 3 * 24 * 60 * 60;
  const INITIAL_ESCROW = ONE_FTNS * 1000n;
  const ROOT = ethers.keccak256(ethers.toUtf8Bytes("att-test-root"));
  const ATT = ethers.keccak256(ethers.toUtf8Bytes("tier=HARDWARE_VERIFIED|mrenclave=..."));
  const COUNT = 10n;

  beforeEach(async function () {
    [owner, provider, requester] = await ethers.getSigners();
    const Token = await ethers.getContractFactory("MockERC20");
    token = await Token.deploy();
    await token.waitForDeployment();
    const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
    registry = await Registry.deploy(owner.address, DEFAULT_WINDOW);
    await registry.waitForDeployment();
    const Pool = await ethers.getContractFactory("EscrowPool");
    escrowPool = await Pool.deploy(
      owner.address, await token.getAddress(), await registry.getAddress());
    await escrowPool.waitForDeployment();
    await registry.connect(owner).setEscrowPool(await escrowPool.getAddress());
    await token.mint(requester.address, INITIAL_ESCROW);
    await token.connect(requester).approve(await escrowPool.getAddress(), INITIAL_ESCROW);
    await escrowPool.connect(requester).deposit(INITIAL_ESCROW);
  });

  function batchIdOf(r) {
    return r.logs.find((l) => l.fragment && l.fragment.name === "BatchCommitted").args[0];
  }
  function attEvent(r) {
    return r.logs.find((l) => l.fragment && l.fragment.name === "BatchAttestationCommitted");
  }

  it("stores the attestation commitment and emits BatchAttestationCommitted", async function () {
    const tx = await registry.connect(provider).commitBatchWithAttestation(
      requester.address, ROOT, COUNT, ONE_FTNS, 0, ethers.ZeroHash, "", ATT);
    const r = await tx.wait();
    const batchId = batchIdOf(r);

    const batch = await registry.getBatch(batchId);
    expect(batch.attestationCommitment).to.equal(ATT);

    const ev = attEvent(r);
    expect(ev, "BatchAttestationCommitted emitted").to.not.be.undefined;
    expect(ev.args[0]).to.equal(batchId);
    expect(ev.args[1]).to.equal(provider.address);
    expect(ev.args[2]).to.equal(ATT);
  });

  it("does NOT include the attestation commitment in batchId derivation", async function () {
    // CONSENSUS INVARIANT: batchId = keccak256(abi.encode(provider, requester,
    // root, count, block.number, sequence)) — attestation MUST NOT be in the
    // preimage, else existing deployed batchIds would shift.
    const seq = await registry.providerBatchSequence(provider.address); // 0, fresh
    const tx = await registry.connect(provider).commitBatchWithAttestation(
      requester.address, ROOT, COUNT, ONE_FTNS, 0, ethers.ZeroHash, "", ATT);
    const r = await tx.wait();
    const onchainId = batchIdOf(r);

    const expectedId = ethers.keccak256(
      ethers.AbiCoder.defaultAbiCoder().encode(
        ["address", "address", "bytes32", "uint256", "uint256", "uint256"],
        [provider.address, requester.address, ROOT, COUNT, r.blockNumber, seq]));
    expect(onchainId).to.equal(expectedId);
  });

  it("zero commitment stores bytes32(0) and emits NO attestation event", async function () {
    const tx = await registry.connect(provider).commitBatchWithAttestation(
      requester.address, ROOT, COUNT, ONE_FTNS, 0, ethers.ZeroHash, "", ethers.ZeroHash);
    const r = await tx.wait();
    const batch = await registry.getBatch(batchIdOf(r));
    expect(batch.attestationCommitment).to.equal(ethers.ZeroHash);
    expect(attEvent(r), "no attestation event for zero commitment").to.be.undefined;
  });

  it("legacy commitBatch is unchanged: zero attestation, no attestation event", async function () {
    const tx = await registry.connect(provider).commitBatch(
      requester.address, ROOT, COUNT, ONE_FTNS, 0, ethers.ZeroHash, "");
    const r = await tx.wait();
    // still emits BatchCommitted + records the batch normally
    const batchId = batchIdOf(r);
    const batch = await registry.getBatch(batchId);
    expect(batch.merkleRoot).to.equal(ROOT);
    expect(batch.attestationCommitment).to.equal(ethers.ZeroHash); // default
    expect(attEvent(r), "legacy path emits no attestation event").to.be.undefined;
  });

  it("legacy and attested commits from the same provider behave identically but for the attestation", async function () {
    // Same inputs, sequential commits → batchIds differ ONLY by sequence (not
    // the attestation); the attested one carries the commitment, the legacy one
    // carries zero. Proves the feature is purely additive.
    const txA = await registry.connect(provider).commitBatch(
      requester.address, ROOT, COUNT, ONE_FTNS, 0, ethers.ZeroHash, "");
    const rA = await txA.wait();
    const txB = await registry.connect(provider).commitBatchWithAttestation(
      requester.address, ROOT, COUNT, ONE_FTNS, 0, ethers.ZeroHash, "", ATT);
    const rB = await txB.wait();

    const bA = await registry.getBatch(batchIdOf(rA));
    const bB = await registry.getBatch(batchIdOf(rB));
    expect(batchIdOf(rA)).to.not.equal(batchIdOf(rB));   // distinct (sequence)
    expect(bA.merkleRoot).to.equal(bB.merkleRoot);
    expect(bA.totalValueFTNS).to.equal(bB.totalValueFTNS);
    expect(bA.attestationCommitment).to.equal(ethers.ZeroHash);
    expect(bB.attestationCommitment).to.equal(ATT);
  });

  it("finalizes an attested batch normally (attestation does not affect finalize)", async function () {
    // Integration: an attested batch goes through the SAME finalize path as a
    // standard one — proving attestationCommitment is inert in finalize/payout.
    const tx = await registry.connect(provider).commitBatchWithAttestation(
      requester.address, ROOT, COUNT, ONE_FTNS, 0, ethers.ZeroHash, "", ATT);
    const r = await tx.wait();
    const batchId = batchIdOf(r);

    await time.increase(DEFAULT_WINDOW + 1);          // past the challenge window
    const before = await token.balanceOf(provider.address);
    await registry.connect(provider).finalizeBatch(batchId);
    const after = await token.balanceOf(provider.address);

    expect(after - before).to.equal(ONE_FTNS);         // full value paid out
    const batch = await registry.getBatch(batchId);
    expect(batch.status).to.equal(2);                  // FINALIZED
    expect(batch.attestationCommitment).to.equal(ATT); // commitment intact post-finalize
  });
});
