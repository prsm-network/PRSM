// Sprint 1481 — OperatorRewardPool: the pool -> earner rail.
//
// The user-readiness re-assessment (wf_58eae474) found the token economy's real
// blocker was not the multi-sig ceremony (which had already executed) but MISSING
// CODE: nothing routed emitted FTNS from the pools to individual earners.
//
// These tests attack the money properties, not the happy path:
//   * solvency      — a published epoch must be fully backed; claims never strand
//   * exactly-once  — one claim per (epoch, account), no double-spend
//   * proof binding — leaves bound to epochId + account + amount; no replay/forgery
//   * role split    — the publisher key cannot move funds; owner cannot touch
//                     reserved entitlements
const { expect } = require("chai");
const { ethers, upgrades } = require("hardhat");

// OZ sorted-pair merkle over double-hashed leaves — the SAME convention as
// prsm/settlement/merkle.py and MerkleProof.sol.
function hashPair(a, b) {
  const [x, y] = a.toLowerCase() <= b.toLowerCase() ? [a, b] : [b, a];
  return ethers.keccak256(ethers.concat([x, y]));
}

function leafHash(epochId, account, amount) {
  const inner = ethers.keccak256(
    ethers.AbiCoder.defaultAbiCoder().encode(
      ["uint256", "address", "uint256"],
      [epochId, account, amount]
    )
  );
  return ethers.keccak256(ethers.concat([inner]));
}

function buildRoot(leaves) {
  if (leaves.length === 0) throw new Error("empty");
  let layer = [...leaves];
  while (layer.length > 1) {
    const next = [];
    for (let i = 0; i < layer.length; i += 2) {
      next.push(i + 1 < layer.length ? hashPair(layer[i], layer[i + 1]) : layer[i]);
    }
    layer = next;
  }
  return layer[0];
}

function buildProof(leaves, index) {
  const proof = [];
  let layer = [...leaves];
  let idx = index;
  while (layer.length > 1) {
    const next = [];
    for (let i = 0; i < layer.length; i += 2) {
      if (i + 1 < layer.length) {
        if (i === idx || i + 1 === idx) {
          proof.push(i === idx ? layer[i + 1] : layer[i]);
          idx = next.length;
        }
        next.push(hashPair(layer[i], layer[i + 1]));
      } else {
        if (i === idx) idx = next.length;
        next.push(layer[i]);
      }
    }
    layer = next;
  }
  return proof;
}

const FTNS = (n) => ethers.parseEther(String(n));

describe("OperatorRewardPool", function () {
  let ftns, pool, owner, publisher, alice, bob, carol, outsider, treasury;

  beforeEach(async function () {
    [owner, publisher, alice, bob, carol, outsider, treasury] =
      await ethers.getSigners();
    // FTNSTokenSimple is a UUPS proxy; initialize(owner, treasury) mints the
    // initial supply to `treasury`, so the pool is funded from there.
    const Token = await ethers.getContractFactory("FTNSTokenSimple");
    ftns = await upgrades.deployProxy(
      Token, [owner.address, treasury.address],
      { initializer: "initialize", kind: "uups" }
    );
    await ftns.waitForDeployment();

    const Pool = await ethers.getContractFactory("OperatorRewardPool");
    pool = await Pool.deploy(
      await ftns.getAddress(), owner.address, publisher.address
    );
    await pool.waitForDeployment();
  });

  async function fund(amount) {
    await ftns.connect(treasury).transfer(await pool.getAddress(), amount);
  }

  // Standard 3-earner epoch used by most tests.
  function epochFixture(epochId = 1n) {
    const entries = [
      { account: alice.address, amount: FTNS(100) },
      { account: bob.address, amount: FTNS(250) },
      { account: carol.address, amount: FTNS(50) },
    ];
    const leaves = entries.map((e) => leafHash(epochId, e.account, e.amount));
    const root = buildRoot(leaves);
    const total = entries.reduce((a, e) => a + e.amount, 0n);
    const proofs = entries.map((_, i) => buildProof(leaves, i));
    return { entries, leaves, root, total, proofs, epochId };
  }

  describe("leaf parity", function () {
    it("on-chain leafHash matches the off-chain double-hash convention", async function () {
      const expected = leafHash(7n, alice.address, FTNS(3));
      expect(await pool.leafHash(7n, alice.address, FTNS(3))).to.equal(expected);
    });
  });

  describe("solvency", function () {
    it("refuses to publish an epoch the pool cannot fully back", async function () {
      const f = epochFixture();
      await fund(f.total - 1n); // one wei short
      await expect(
        pool.connect(publisher).publishEpoch(f.epochId, f.root, f.total)
      ).to.be.revertedWithCustomError(pool, "InsufficientBacking");
    });

    it("refuses a second epoch that would double-spend the first epoch's backing", async function () {
      const f1 = epochFixture(1n);
      await fund(f1.total); // exactly enough for ONE epoch
      await pool.connect(publisher).publishEpoch(1n, f1.root, f1.total);

      const f2 = epochFixture(2n);
      await expect(
        pool.connect(publisher).publishEpoch(2n, f2.root, f2.total)
      ).to.be.revertedWithCustomError(pool, "InsufficientBacking");
    });

    it("★ every earner can claim in full even if others claim first (no stranding)", async function () {
      const f = epochFixture();
      await fund(f.total);
      await pool.connect(publisher).publishEpoch(f.epochId, f.root, f.total);

      for (let i = 0; i < f.entries.length; i++) {
        const e = f.entries[i];
        await pool.connect(outsider).claim(f.epochId, e.account, e.amount, f.proofs[i]);
        expect(await ftns.balanceOf(e.account)).to.equal(e.amount);
      }
      expect(await ftns.balanceOf(await pool.getAddress())).to.equal(0n);
      expect(await pool.totalReserved()).to.equal(0n);
    });
  });

  describe("exactly-once claiming", function () {
    it("rejects a second claim by the same account", async function () {
      const f = epochFixture();
      await fund(f.total);
      await pool.connect(publisher).publishEpoch(f.epochId, f.root, f.total);
      await pool.connect(alice).claim(f.epochId, alice.address, f.entries[0].amount, f.proofs[0]);
      await expect(
        pool.connect(alice).claim(f.epochId, alice.address, f.entries[0].amount, f.proofs[0])
      ).to.be.revertedWithCustomError(pool, "AlreadyClaimed");
      expect(await ftns.balanceOf(alice.address)).to.equal(f.entries[0].amount);
    });

    it("an epoch root cannot be republished/overwritten", async function () {
      const f = epochFixture();
      await fund(f.total * 3n);
      await pool.connect(publisher).publishEpoch(f.epochId, f.root, f.total);
      await expect(
        pool.connect(publisher).publishEpoch(f.epochId, f.root, f.total)
      ).to.be.revertedWithCustomError(pool, "EpochAlreadyPublished");
    });
  });

  describe("proof binding", function () {
    it("rejects an inflated amount with an otherwise-valid proof", async function () {
      const f = epochFixture();
      await fund(f.total);
      await pool.connect(publisher).publishEpoch(f.epochId, f.root, f.total);
      await expect(
        pool.connect(alice).claim(f.epochId, alice.address, FTNS(999), f.proofs[0])
      ).to.be.revertedWithCustomError(pool, "InvalidProof");
    });

    it("rejects a claim redirected to another account", async function () {
      const f = epochFixture();
      await fund(f.total);
      await pool.connect(publisher).publishEpoch(f.epochId, f.root, f.total);
      await expect(
        pool.connect(outsider).claim(f.epochId, outsider.address, f.entries[0].amount, f.proofs[0])
      ).to.be.revertedWithCustomError(pool, "InvalidProof");
    });

    it("★ a proof from epoch 1 cannot be replayed against epoch 2 (epochId is in the leaf)", async function () {
      const f1 = epochFixture(1n);
      await fund(f1.total * 2n);
      await pool.connect(publisher).publishEpoch(1n, f1.root, f1.total);
      // Publish epoch 2 with the IDENTICAL root — leaves still differ by epochId.
      await pool.connect(publisher).publishEpoch(2n, f1.root, f1.total);
      await expect(
        pool.connect(alice).claim(2n, alice.address, f1.entries[0].amount, f1.proofs[0])
      ).to.be.revertedWithCustomError(pool, "InvalidProof");
    });

    it("rejects claims against an unpublished epoch", async function () {
      const f = epochFixture();
      await expect(
        pool.connect(alice).claim(99n, alice.address, f.entries[0].amount, f.proofs[0])
      ).to.be.revertedWithCustomError(pool, "EpochNotPublished");
    });

    it("★ an over-summed root cannot drain another epoch's reserved funds", async function () {
      // A malicious/buggy publisher declares totalAmount smaller than the leaves sum.
      const epochId = 1n;
      const entries = [
        { account: alice.address, amount: FTNS(100) },
        { account: bob.address, amount: FTNS(100) },
      ];
      const leaves = entries.map((e) => leafHash(epochId, e.account, e.amount));
      const root = buildRoot(leaves);
      const understated = FTNS(120); // < 200 actual
      await fund(FTNS(1000)); // plenty of raw balance sitting in the pool
      await pool.connect(publisher).publishEpoch(epochId, root, understated);

      await pool.connect(alice).claim(epochId, alice.address, FTNS(100), buildProof(leaves, 0));
      // Bob's leaf is valid, but honoring it would exceed the declared reserve.
      await expect(
        pool.connect(bob).claim(epochId, bob.address, FTNS(100), buildProof(leaves, 1))
      ).to.be.revertedWithCustomError(pool, "EpochOverdrawn");
    });
  });

  describe("role separation", function () {
    it("only the publisher may publish", async function () {
      const f = epochFixture();
      await fund(f.total);
      await expect(
        pool.connect(owner).publishEpoch(f.epochId, f.root, f.total)
      ).to.be.revertedWithCustomError(pool, "NotPublisher");
      await expect(
        pool.connect(outsider).publishEpoch(f.epochId, f.root, f.total)
      ).to.be.revertedWithCustomError(pool, "NotPublisher");
    });

    it("★ the publisher key cannot move funds", async function () {
      await fund(FTNS(100));
      await expect(
        pool.connect(publisher).sweepSurplus(publisher.address, FTNS(100))
      ).to.be.revertedWithCustomError(pool, "OwnableUnauthorizedAccount");
      await expect(
        pool.connect(publisher).setRootPublisher(outsider.address)
      ).to.be.revertedWithCustomError(pool, "OwnableUnauthorizedAccount");
    });

    it("★ the owner cannot sweep funds reserved for earners", async function () {
      const f = epochFixture();
      await fund(f.total);
      await pool.connect(publisher).publishEpoch(f.epochId, f.root, f.total);
      // Entire balance is reserved -> zero surplus.
      expect(await pool.surplus()).to.equal(0n);
      await expect(
        pool.connect(owner).sweepSurplus(owner.address, 1n)
      ).to.be.revertedWithCustomError(pool, "NoSurplus");
    });

    it("owner may sweep only the unreserved surplus", async function () {
      const f = epochFixture();
      await fund(f.total + FTNS(10));
      await pool.connect(publisher).publishEpoch(f.epochId, f.root, f.total);
      expect(await pool.surplus()).to.equal(FTNS(10));
      await expect(
        pool.connect(owner).sweepSurplus(owner.address, FTNS(11))
      ).to.be.revertedWithCustomError(pool, "NoSurplus");
      await pool.connect(owner).sweepSurplus(owner.address, FTNS(10));
      expect(await pool.surplus()).to.equal(0n);
      // ...and the reserved funds are still there for the earners.
      await pool.connect(alice).claim(f.epochId, alice.address, f.entries[0].amount, f.proofs[0]);
      expect(await ftns.balanceOf(alice.address)).to.equal(f.entries[0].amount);
    });
  });

  describe("pause + reclaim", function () {
    it("pausing halts publish and claim", async function () {
      const f = epochFixture();
      await fund(f.total);
      await pool.connect(owner).setPaused(true);
      await expect(
        pool.connect(publisher).publishEpoch(f.epochId, f.root, f.total)
      ).to.be.revertedWithCustomError(pool, "IsPaused");
      await pool.connect(owner).setPaused(false);
      await pool.connect(publisher).publishEpoch(f.epochId, f.root, f.total);
      await pool.connect(owner).setPaused(true);
      await expect(
        pool.connect(alice).claim(f.epochId, alice.address, f.entries[0].amount, f.proofs[0])
      ).to.be.revertedWithCustomError(pool, "IsPaused");
    });

    it("★ unclaimed funds cannot be reclaimed before the immutable window elapses", async function () {
      const f = epochFixture();
      await fund(f.total);
      await pool.connect(publisher).publishEpoch(f.epochId, f.root, f.total);
      await expect(
        pool.connect(owner).reclaimUnclaimed(f.epochId)
      ).to.be.revertedWithCustomError(pool, "ClaimWindowOpen");

      // Just before the boundary it is still protected.
      const window = await pool.MIN_CLAIM_WINDOW();
      await ethers.provider.send("evm_increaseTime", [Number(window) - 60]);
      await ethers.provider.send("evm_mine", []);
      await expect(
        pool.connect(owner).reclaimUnclaimed(f.epochId)
      ).to.be.revertedWithCustomError(pool, "ClaimWindowOpen");
    });

    it("after the window, the remainder becomes sweepable surplus", async function () {
      const f = epochFixture();
      await fund(f.total);
      await pool.connect(publisher).publishEpoch(f.epochId, f.root, f.total);
      await pool.connect(alice).claim(f.epochId, alice.address, f.entries[0].amount, f.proofs[0]);

      const window = await pool.MIN_CLAIM_WINDOW();
      await ethers.provider.send("evm_increaseTime", [Number(window) + 1]);
      await ethers.provider.send("evm_mine", []);

      const remaining = f.total - f.entries[0].amount;
      await expect(pool.connect(owner).reclaimUnclaimed(f.epochId))
        .to.emit(pool, "UnclaimedReclaimed")
        .withArgs(f.epochId, remaining);
      expect(await pool.totalReserved()).to.equal(0n);
      expect(await pool.surplus()).to.equal(remaining);
      await expect(
        pool.connect(owner).reclaimUnclaimed(f.epochId)
      ).to.be.revertedWithCustomError(pool, "AlreadyReclaimed");
    });
  });

  describe("views", function () {
    it("isClaimable tracks state without sending a transaction", async function () {
      const f = epochFixture();
      await fund(f.total);
      await pool.connect(publisher).publishEpoch(f.epochId, f.root, f.total);
      expect(await pool.isClaimable(
        f.epochId, alice.address, f.entries[0].amount, f.proofs[0])).to.equal(true);
      await pool.connect(alice).claim(f.epochId, alice.address, f.entries[0].amount, f.proofs[0]);
      expect(await pool.isClaimable(
        f.epochId, alice.address, f.entries[0].amount, f.proofs[0])).to.equal(false);
    });
  });
});
