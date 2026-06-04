const { expect } = require("chai");
const {
  assertSignerMatchesOwner,
} = require("../scripts/deploy-guards");

// sp990 — the wrong-signer guard that closes the readiness-audit ceremony-breaker
// (hardhat's `base` network signs via MAINNET_PRIVATE_KEY || PRIVATE_KEY, so a
// stale .env MAINNET_PRIVATE_KEY can silently sign an irreversible mainnet deploy
// from the wrong EOA). The guard aborts BEFORE any on-chain tx when the resolved
// signer differs from the intended owner.

const A = "0x" + "a".repeat(40);
const B = "0x" + "b".repeat(40);

describe("assertSignerMatchesOwner (sp990 wrong-signer guard)", function () {
  it("throws on a LIVE network when deployer != owner", function () {
    expect(() =>
      assertSignerMatchesOwner({
        network: "base",
        deployerAddress: B, // a different EOA actually signs
        owner: A, // intended owner
        env: { CREATOR_STAKE_OWNER: A },
      })
    ).to.throw(/Signer\/owner mismatch/);
  });

  it("includes the actual signer + intended owner + the likely cause in the message", function () {
    let msg = "";
    try {
      assertSignerMatchesOwner({
        network: "base",
        deployerAddress: B,
        owner: A,
        env: { CREATOR_STAKE_OWNER: A },
      });
    } catch (e) {
      msg = e.message;
    }
    expect(msg).to.contain(B); // the wrong signer is named
    expect(msg).to.contain(A); // the intended owner is named
    expect(msg).to.contain("MAINNET_PRIVATE_KEY"); // points at the root cause
  });

  it("does NOT throw when deployer == owner (case-insensitive)", function () {
    expect(() =>
      assertSignerMatchesOwner({
        network: "base",
        deployerAddress: A.toUpperCase().replace("0X", "0x"),
        owner: A,
        env: { CREATOR_STAKE_OWNER: A },
      })
    ).to.not.throw();
  });

  it("is a no-op on the in-process hardhat network (dry-runs use a non-owner addr)", function () {
    expect(() =>
      assertSignerMatchesOwner({
        network: "hardhat",
        deployerAddress: B,
        owner: A,
        env: { CREATOR_STAKE_OWNER: A },
      })
    ).to.not.throw();
  });

  it("is a no-op on localhost", function () {
    expect(() =>
      assertSignerMatchesOwner({
        network: "localhost",
        deployerAddress: B,
        owner: A,
        env: { CREATOR_STAKE_OWNER: A },
      })
    ).to.not.throw();
  });

  it("is a no-op when CREATOR_STAKE_OWNER is unset (owner defaults to deployer)", function () {
    expect(() =>
      assertSignerMatchesOwner({
        network: "base",
        deployerAddress: B,
        owner: B,
        env: {},
      })
    ).to.not.throw();
  });

  it("respects the ALLOW_OWNER_DEPLOYER_MISMATCH=1 opt-out", function () {
    expect(() =>
      assertSignerMatchesOwner({
        network: "base",
        deployerAddress: B,
        owner: A,
        env: { CREATOR_STAKE_OWNER: A, ALLOW_OWNER_DEPLOYER_MISMATCH: "1" },
      })
    ).to.not.throw();
  });
});
