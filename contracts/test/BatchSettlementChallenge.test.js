const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

// Phase 3.1 Task 3 — challengeReceipt + 4-of-5 reason-code handlers.
// (MALFORMED reserved, not implementable in Task 3 scope.)

describe("BatchSettlementRegistry — challengeReceipt", function () {
  let registry, escrowPool, token, verifier;
  let owner, provider, requester, challenger, watchdog, otherRequester;
  const ONE_FTNS = ethers.parseUnits("1", 18);
  const DEFAULT_WINDOW = 3 * 24 * 60 * 60; // 3 days
  const DEFAULT_LOOKBACK = 30 * 24 * 60 * 60; // 30 days

  // Helper: build a ReceiptLeaf struct + compute its keccak256 hash.
  function makeLeaf(overrides = {}) {
    const leaf = {
      jobIdHash: ethers.keccak256(ethers.toUtf8Bytes("job-1")),
      shardIndex: 0,
      providerIdHash: ethers.keccak256(ethers.toUtf8Bytes("provider-id")),
      providerPubkeyHash: ethers.keccak256(ethers.toUtf8Bytes("pubkey-bytes")),
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("output-bytes")),
      executedAtUnix: Math.floor(Date.now() / 1000),
      valueFtns: ONE_FTNS,
      signatureHash: ethers.keccak256(ethers.toUtf8Bytes("signature-bytes")),
      signingMessageHash: ethers.keccak256(ethers.toUtf8Bytes("signing-msg")),
      ...overrides,
    };
    return leaf;
  }

  function hashLeaf(leaf) {
    const encoded = ethers.AbiCoder.defaultAbiCoder().encode(
      [
        "tuple(bytes32,uint32,bytes32,bytes32,bytes32,uint64,uint128,bytes32,bytes32)",
      ],
      [
        [
          leaf.jobIdHash,
          leaf.shardIndex,
          leaf.providerIdHash,
          leaf.providerPubkeyHash,
          leaf.outputHash,
          leaf.executedAtUnix,
          leaf.valueFtns,
          leaf.signatureHash,
          leaf.signingMessageHash,
        ],
      ]
    );
    return ethers.keccak256(encoded);
  }

  // Minimal single-leaf Merkle tree: root == leafHash, proof == [].
  function singleLeafMerkle(leaf) {
    return { root: hashLeaf(leaf), proof: [] };
  }

  beforeEach(async function () {
    [owner, provider, requester, challenger, watchdog, otherRequester] =
      await ethers.getSigners();

    const Token = await ethers.getContractFactory("MockERC20");
    token = await Token.deploy();
    await token.waitForDeployment();

    // Post-HIGH-6 deploy ordering: registry first, then pool.
    const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
    registry = await Registry.deploy(owner.address, DEFAULT_WINDOW);
    await registry.waitForDeployment();

    const Pool = await ethers.getContractFactory("EscrowPool");
    escrowPool = await Pool.deploy(
      owner.address,
      await token.getAddress(),
      await registry.getAddress()
    );
    await escrowPool.waitForDeployment();

    await registry.connect(owner).setEscrowPool(await escrowPool.getAddress());

    const Verifier = await ethers.getContractFactory("MockSignatureVerifier");
    verifier = await Verifier.deploy();
    await verifier.waitForDeployment();
    await registry.connect(owner).setSignatureVerifier(await verifier.getAddress());

    await token.mint(requester.address, ONE_FTNS * 1000n);
    await token
      .connect(requester)
      .approve(await escrowPool.getAddress(), ONE_FTNS * 1000n);
    await escrowPool.connect(requester).deposit(ONE_FTNS * 1000n);
  });

  // Helper: commit a batch containing exactly one leaf, return {batchId, leaf}.
  async function commitOneLeafBatch(leafOverrides = {}) {
    const leaf = makeLeaf(leafOverrides);
    const { root } = singleLeafMerkle(leaf);

    const tx = await registry
      .connect(provider)
      .commitBatch(requester.address, root, 1, leaf.valueFtns,  0, ethers.ZeroHash, "");
    const r = await tx.wait();
    const batchId = r.logs.find(
      (l) => l.fragment && l.fragment.name === "BatchCommitted"
    ).args[0];

    return { batchId, leaf };
  }

  describe("DOUBLE_SPEND", function () {
    it("invalidates a receipt that appears in two batches", async function () {
      // Same receipt committed in two batches (real attack scenario:
      // malicious provider tries to claim the same work twice).
      const sharedLeaf = makeLeaf({ valueFtns: ONE_FTNS * 5n });
      const root = hashLeaf(sharedLeaf);

      // First batch.
      const tx1 = await registry
        .connect(provider)
        .commitBatch(requester.address, root, 1, sharedLeaf.valueFtns,  0, ethers.ZeroHash, "b1");
      const id1 = (await tx1.wait()).logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];

      // Second batch with the same receipt.
      const tx2 = await registry
        .connect(provider)
        .commitBatch(requester.address, root, 1, sharedLeaf.valueFtns,  0, ethers.ZeroHash, "b2");
      const id2 = (await tx2.wait()).logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];
      expect(id2).to.not.equal(id1);

      const aux = ethers.AbiCoder.defaultAbiCoder().encode(
        ["bytes32", "bytes32[]"],
        [id1, []]
      );

      // Challenge the second batch, proving the receipt is also in batch 1.
      await expect(
        registry
          .connect(challenger)
          .challengeReceipt(id2, sharedLeaf, [], 0 /* DOUBLE_SPEND */, aux)
      )
        .to.emit(registry, "ReceiptChallenged")
        .withArgs(
          id2,
          hashLeaf(sharedLeaf),
          challenger.address,
          0,
          sharedLeaf.valueFtns
        );

      // The invalidated value is recorded.
      const batch = await registry.getBatch(id2);
      expect(batch.invalidatedValueFTNS).to.equal(sharedLeaf.valueFtns);
    });

    it("SECURITY: rejects a self-referential DOUBLE_SPEND (a batch cannot be its own conflict)", async function () {
      // Money-path audit #1, variant A. Pre-fix, citing THIS batch as its own
      // conflict re-verified the caller-verified proof against the same root, so
      // ANY receipt in ANY pending batch could be "double-spend" challenged
      // against itself — invalidating payment + slashing the honest provider +
      // paying the challenger the 70% bounty.
      const leaf = makeLeaf({ valueFtns: ONE_FTNS * 5n });
      const root = hashLeaf(leaf);
      const tx = await registry
        .connect(provider)
        .commitBatch(requester.address, root, 1, leaf.valueFtns, 0, ethers.ZeroHash, "self");
      const id = (await tx.wait()).logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];
      // Cite THIS batch as its own conflict.
      const aux = ethers.AbiCoder.defaultAbiCoder().encode(
        ["bytes32", "bytes32[]"],
        [id, []]
      );
      await expect(
        registry.connect(challenger).challengeReceipt(id, leaf, [], 0, aux)
      ).to.be.revertedWithCustomError(registry, "ChallengeNotProven");
      expect((await registry.getBatch(id)).invalidatedValueFTNS).to.equal(0);
    });

    it("SECURITY: rejects a copycat batch committed AFTER the victim (first-committer-wins)", async function () {
      // Money-path audit #1, variant B. An attacker commits a throwaway batch
      // copying the victim's leaf AFTER the victim's batch (commitBatch is
      // permissionless), then challenges the HONEST provider's earlier batch
      // citing the copycat. Pre-fix this slashed the honest first committer.
      // The first-committer-wins guard requires the cited conflict to be
      // STRICTLY EARLIER, so this reverts.
      const sharedLeaf = makeLeaf({ valueFtns: ONE_FTNS * 5n });
      const root = hashLeaf(sharedLeaf);

      const txVictim = await registry
        .connect(provider)
        .commitBatch(requester.address, root, 1, sharedLeaf.valueFtns, 0, ethers.ZeroHash, "victim");
      const idVictim = (await txVictim.wait()).logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];

      // Attacker's copycat, committed LATER, same leaf.
      const txCopycat = await registry
        .connect(challenger)
        .commitBatch(requester.address, root, 1, sharedLeaf.valueFtns, 0, ethers.ZeroHash, "copycat");
      const idCopycat = (await txCopycat.wait()).logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];

      const aux = ethers.AbiCoder.defaultAbiCoder().encode(
        ["bytes32", "bytes32[]"],
        [idCopycat, []]
      );
      // Challenge the VICTIM's earlier batch citing the later copycat → must revert.
      await expect(
        registry.connect(challenger).challengeReceipt(idVictim, sharedLeaf, [], 0, aux)
      ).to.be.revertedWithCustomError(registry, "ChallengeNotProven");
      expect((await registry.getBatch(idVictim)).invalidatedValueFTNS).to.equal(0);
    });

    it("reverts when the conflicting batch does not exist", async function () {
      const { batchId, leaf } = await commitOneLeafBatch();
      const fakeOtherId = ethers.keccak256(
        ethers.toUtf8Bytes("nonexistent-batch")
      );
      const aux = ethers.AbiCoder.defaultAbiCoder().encode(
        ["bytes32", "bytes32[]"],
        [fakeOtherId, []]
      );
      await expect(
        registry
          .connect(challenger)
          .challengeReceipt(batchId, leaf, [], 0, aux)
      ).to.be.revertedWithCustomError(registry, "ConflictingBatchNotCommitted");
    });

    it("reverts when the receipt isn't actually in the conflicting batch", async function () {
      // Two batches, but different receipts — challenger falsely claims overlap.
      const leafA = makeLeaf({ jobIdHash: ethers.keccak256(ethers.toUtf8Bytes("job-a")) });
      const leafB = makeLeaf({ jobIdHash: ethers.keccak256(ethers.toUtf8Bytes("job-b")) });
      const rootA = hashLeaf(leafA);
      const rootB = hashLeaf(leafB);

      const tx1 = await registry
        .connect(provider)
        .commitBatch(requester.address, rootA, 1, leafA.valueFtns,  0, ethers.ZeroHash, "");
      const idA = (await tx1.wait()).logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];

      const tx2 = await registry
        .connect(provider)
        .commitBatch(requester.address, rootB, 1, leafB.valueFtns,  0, ethers.ZeroHash, "");
      const idB = (await tx2.wait()).logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];

      // Claim leafA is in batch B (it isn't).
      const aux = ethers.AbiCoder.defaultAbiCoder().encode(
        ["bytes32", "bytes32[]"],
        [idB, []]
      );
      await expect(
        registry
          .connect(challenger)
          .challengeReceipt(idA, leafA, [], 0, aux)
      ).to.be.revertedWithCustomError(registry, "ChallengeNotProven");
    });
  });

  describe("INVALID_SIGNATURE", function () {
    it("invalidates when the verifier returns false", async function () {
      await verifier.setResult(false); // signature "invalid" per mock

      const pubkey = ethers.toUtf8Bytes("pubkey-bytes");
      const signature = ethers.toUtf8Bytes("signature-bytes");
      const leaf = makeLeaf({
        providerPubkeyHash: ethers.keccak256(pubkey),
        signatureHash: ethers.keccak256(signature),
      });
      const root = hashLeaf(leaf);

      const tx = await registry
        .connect(provider)
        .commitBatch(requester.address, root, 1, leaf.valueFtns,  0, ethers.ZeroHash, "");
      const batchId = (await tx.wait()).logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];

      // Post-C-INT-01 remediation: auxData is (publicKey, signature) only.
      // signingMessageHash is bound in the leaf, not supplied by challenger.
      const aux = ethers.AbiCoder.defaultAbiCoder().encode(
        ["bytes", "bytes"],
        [pubkey, signature]
      );

      await expect(
        registry
          .connect(challenger)
          .challengeReceipt(batchId, leaf, [], 1 /* INVALID_SIGNATURE */, aux)
      ).to.emit(registry, "ReceiptChallenged");

      const batch = await registry.getBatch(batchId);
      expect(batch.invalidatedValueFTNS).to.equal(leaf.valueFtns);
    });

    it("reverts when the verifier returns true (sig is actually valid)", async function () {
      await verifier.setResult(true); // signature "valid" — challenge is wrong

      const pubkey = ethers.toUtf8Bytes("pubkey-bytes");
      const signature = ethers.toUtf8Bytes("signature-bytes");
      const leaf = makeLeaf({
        providerPubkeyHash: ethers.keccak256(pubkey),
        signatureHash: ethers.keccak256(signature),
      });
      const { batchId } = await commitOneLeafBatchWithLeaf(leaf);

      const aux = ethers.AbiCoder.defaultAbiCoder().encode(
        ["bytes", "bytes"],
        [pubkey, signature]
      );

      await expect(
        registry.connect(challenger).challengeReceipt(batchId, leaf, [], 1, aux)
      ).to.be.revertedWithCustomError(registry, "ChallengeNotProven");
    });

    it("reverts when pubkey doesn't match the committed hash", async function () {
      await verifier.setResult(false);

      const correctPubkey = ethers.toUtf8Bytes("correct");
      const wrongPubkey = ethers.toUtf8Bytes("wrong");
      const leaf = makeLeaf({
        providerPubkeyHash: ethers.keccak256(correctPubkey),
      });
      const { batchId } = await commitOneLeafBatchWithLeaf(leaf);

      const aux = ethers.AbiCoder.defaultAbiCoder().encode(
        ["bytes", "bytes"],
        [
          wrongPubkey, // mismatch
          ethers.toUtf8Bytes("signature-bytes"),
        ]
      );
      await expect(
        registry.connect(challenger).challengeReceipt(batchId, leaf, [], 1, aux)
      ).to.be.revertedWithCustomError(registry, "ChallengeNotProven");
    });

    it("reverts when verifier is not configured", async function () {
      await registry.connect(owner).setSignatureVerifier(ethers.ZeroAddress);
      const { batchId, leaf } = await commitOneLeafBatch();
      const aux = ethers.AbiCoder.defaultAbiCoder().encode(
        ["bytes", "bytes"],
        [ethers.toUtf8Bytes("p"), ethers.toUtf8Bytes("s")]
      );
      await expect(
        registry.connect(challenger).challengeReceipt(batchId, leaf, [], 1, aux)
      ).to.be.revertedWithCustomError(registry, "VerifierNotConfigured");
    });
  });

  describe("NO_ESCROW", function () {
    it("invalidates when batch requester challenges", async function () {
      const { batchId, leaf } = await commitOneLeafBatch();

      await expect(
        registry
          .connect(requester) // the named requester on the batch
          .challengeReceipt(batchId, leaf, [], 2 /* NO_ESCROW */, "0x")
      ).to.emit(registry, "ReceiptChallenged");

      const batch = await registry.getBatch(batchId);
      expect(batch.invalidatedValueFTNS).to.equal(leaf.valueFtns);
    });

    it("reverts when a non-requester challenges", async function () {
      const { batchId, leaf } = await commitOneLeafBatch();

      await expect(
        registry
          .connect(otherRequester) // not the named requester
          .challengeReceipt(batchId, leaf, [], 2, "0x")
      ).to.be.revertedWithCustomError(registry, "CallerNotRequester");
    });
  });

  describe("EXPIRED", function () {
    it("invalidates a receipt older than the lookback window", async function () {
      // EXPIRED is a tight test: the leaf's executedAtUnix must be
      // older than the lookback (30 days) at CHALLENGE time, but the
      // batch itself must still be within the challenge window (3 days).
      // We achieve this by setting the leaf's timestamp to block.timestamp
      // minus (lookback + safety margin) BEFORE commit — so at commit the
      // leaf is already "stale" from the lookback's perspective.
      const blockTime = await time.latest();
      const oldTimestamp = blockTime - DEFAULT_LOOKBACK - 60 * 60; // >30 days old at commit
      const leaf = makeLeaf({ executedAtUnix: oldTimestamp });
      const { batchId } = await commitOneLeafBatchWithLeaf(leaf);

      await expect(
        registry.connect(challenger).challengeReceipt(batchId, leaf, [], 3 /* EXPIRED */, "0x")
      ).to.emit(registry, "ReceiptChallenged");
    });

    it("reverts on a receipt within the lookback window", async function () {
      // Leaf must use hardhat's block.timestamp (not Date.now) because
      // prior tests in the suite may have advanced block.timestamp.
      const blockTime = await time.latest();
      const leaf = makeLeaf({ executedAtUnix: blockTime });
      const { batchId } = await commitOneLeafBatchWithLeaf(leaf);

      await expect(
        registry.connect(challenger).challengeReceipt(batchId, leaf, [], 3, "0x")
      ).to.be.revertedWithCustomError(registry, "ReceiptNotExpired");
    });
  });

  describe("MALFORMED — reserved", function () {
    it("reverts with MalformedReasonNotImplemented", async function () {
      const { batchId, leaf } = await commitOneLeafBatch();
      await expect(
        registry.connect(challenger).challengeReceipt(batchId, leaf, [], 4, "0x")
      ).to.be.revertedWithCustomError(registry, "MalformedReasonNotImplemented");
    });
  });

  describe("validation", function () {
    it("reverts on a bad merkle proof", async function () {
      // Commit with root A; challenge with leaf B that doesn't hash to A.
      const leafA = makeLeaf({ jobIdHash: ethers.keccak256(ethers.toUtf8Bytes("A")) });
      const leafB = makeLeaf({ jobIdHash: ethers.keccak256(ethers.toUtf8Bytes("B")) });
      const rootA = hashLeaf(leafA);

      const tx = await registry
        .connect(provider)
        .commitBatch(requester.address, rootA, 1, leafA.valueFtns,  0, ethers.ZeroHash, "");
      const id = (await tx.wait()).logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];

      await expect(
        registry
          .connect(requester)
          .challengeReceipt(id, leafB, [], 2, "0x")
      ).to.be.revertedWithCustomError(registry, "InvalidMerkleProof");
    });

    it("reverts on double-challenge of the same receipt", async function () {
      const { batchId, leaf } = await commitOneLeafBatch();

      // First challenge succeeds.
      await registry.connect(requester).challengeReceipt(batchId, leaf, [], 2, "0x");

      // Second challenge on the same leaf reverts.
      await expect(
        registry.connect(requester).challengeReceipt(batchId, leaf, [], 2, "0x")
      ).to.be.revertedWithCustomError(registry, "ReceiptAlreadyInvalidated");
    });

    it("reverts when the challenge window has elapsed", async function () {
      const { batchId, leaf } = await commitOneLeafBatch();
      await time.increase(DEFAULT_WINDOW + 60);

      await expect(
        registry.connect(requester).challengeReceipt(batchId, leaf, [], 2, "0x")
      ).to.be.revertedWithCustomError(registry, "ChallengeWindowElapsed");
    });

    it("reverts on unknown batchId", async function () {
      const fakeId = ethers.keccak256(ethers.toUtf8Bytes("fake"));
      const leaf = makeLeaf();
      await expect(
        registry.connect(challenger).challengeReceipt(fakeId, leaf, [], 2, "0x")
      ).to.be.revertedWithCustomError(registry, "BatchNotFound");
    });

    it("reverts on already-finalized batch", async function () {
      const { batchId, leaf } = await commitOneLeafBatch();
      await time.increase(DEFAULT_WINDOW);
      await registry.finalizeBatch(batchId);

      await expect(
        registry.connect(requester).challengeReceipt(batchId, leaf, [], 2, "0x")
      ).to.be.revertedWithCustomError(registry, "BatchNotPending");
    });
  });

  describe("finalizeBatch with invalidations", function () {
    it("excludes invalidated receipts' values from final payment", async function () {
      // Commit a batch with a single leaf worth 10 FTNS.
      const leaf = makeLeaf({ valueFtns: ONE_FTNS * 10n });
      const { batchId } = await commitOneLeafBatchWithLeaf(leaf, /* batchValue */ ONE_FTNS * 10n);

      // Invalidate it.
      await registry.connect(requester).challengeReceipt(batchId, leaf, [], 2, "0x");

      // Wait for window.
      await time.increase(DEFAULT_WINDOW);

      const providerBalBefore = await token.balanceOf(provider.address);
      const escrowBalBefore = await escrowPool.balanceOf(requester.address);

      await expect(registry.finalizeBatch(batchId))
        .to.emit(registry, "BatchFinalized")
        .withArgs(
          batchId,
          provider.address,
          0, // finalValue = total(10) - invalidated(10) = 0
          ONE_FTNS * 10n, // invalidatedValueFTNS
          anyUint()
        );

      // No transfer occurred since finalValue=0 skips escrow path.
      expect(await token.balanceOf(provider.address)).to.equal(providerBalBefore);
      expect(await escrowPool.balanceOf(requester.address)).to.equal(escrowBalBefore);
    });
  });

  describe("governance surface", function () {
    it("owner can update signature verifier", async function () {
      await expect(
        registry.connect(owner).setSignatureVerifier(ethers.ZeroAddress)
      )
        .to.emit(registry, "SignatureVerifierUpdated")
        .withArgs(await verifier.getAddress(), ethers.ZeroAddress);
    });

    it("non-owner cannot update signature verifier", async function () {
      await expect(
        registry.connect(challenger).setSignatureVerifier(ethers.ZeroAddress)
      ).to.be.reverted;
    });

    it("owner can update lookback window within bounds", async function () {
      const newWindow = 60 * 24 * 60 * 60; // 60 days
      await expect(
        registry.connect(owner).setSettlementLookbackWindow(newWindow)
      )
        .to.emit(registry, "SettlementLookbackUpdated")
        .withArgs(DEFAULT_LOOKBACK, newWindow);
      expect(await registry.settlementLookbackWindowSeconds()).to.equal(newWindow);
    });

    it("reverts on lookback below minimum", async function () {
      await expect(
        registry.connect(owner).setSettlementLookbackWindow(60 * 60)
      ).to.be.revertedWithCustomError(registry, "InvalidLookbackWindow");
    });

    it("reverts on lookback above maximum", async function () {
      await expect(
        registry.connect(owner).setSettlementLookbackWindow(500 * 24 * 60 * 60)
      ).to.be.revertedWithCustomError(registry, "InvalidLookbackWindow");
    });
  });

  // Helper used by several tests — commits a batch containing exactly one
  // caller-specified leaf.
  async function commitOneLeafBatchWithLeaf(leaf, batchValue = null) {
    const root = hashLeaf(leaf);
    const value = batchValue || leaf.valueFtns;
    const tx = await registry
      .connect(provider)
      .commitBatch(requester.address, root, 1, value,  0, ethers.ZeroHash, "");
    const r = await tx.wait();
    const batchId = r.logs.find(
      (l) => l.fragment && l.fragment.name === "BatchCommitted"
    ).args[0];
    return { batchId, leaf };
  }
});

function anyUint() {
  return (value) => typeof value === "bigint" || typeof value === "number";
}
