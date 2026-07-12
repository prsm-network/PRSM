// Sprint 1433 — FTNSBridge.bridgeOut must CHECK the return value of transferFrom.
//
// A bridge that ignores a false-returning transferFrom would burn tokens it never received and
// mint the bridged amount on the destination chain: mint-from-nothing fund loss. FTNSTokenSimple
// reverts on failure today, but a bridge is exactly where a non-reverting-ERC20 swap becomes a
// vulnerability — so bridgeOut now reverts TransferFailed if transferFrom returns false.
//
// This is also bridgeOut's FIRST test (the function had zero coverage; found by the sp1432 SAST
// enablement + flagged as a real fix).

const { expect } = require("chai");
const { ethers, upgrades } = require("hardhat");

describe("FTNSBridge — bridgeOut transferFrom return-value check (sp1433)", function () {
  async function deploy(tokenContractName) {
    const signers = await ethers.getSigners();
    const [admin, feeRecipient, user, v1, v2, v3] = signers;

    const Token = await ethers.getContractFactory(tokenContractName);
    const token = await Token.deploy();
    await token.waitForDeployment();

    const BridgeSecurity = await ethers.getContractFactory("BridgeSecurity");
    const bridgeSecurity = await upgrades.deployProxy(
      BridgeSecurity,
      [admin.address, 2, [v1.address, v2.address, v3.address]],
      { initializer: "initialize", kind: "uups" }
    );
    await bridgeSecurity.waitForDeployment();

    const FTNSBridge = await ethers.getContractFactory("FTNSBridge");
    const bridge = await upgrades.deployProxy(
      FTNSBridge,
      [
        admin.address,
        await token.getAddress(),
        await bridgeSecurity.getAddress(),
        feeRecipient.address,
        1n,                              // minBridgeAmount
        ethers.parseEther("1000000"),    // maxBridgeAmount
        0,                               // bridgeFeeBps (no fee → no mint path)
      ],
      { initializer: "initialize", kind: "uups" }
    );
    await bridge.waitForDeployment();

    return { bridge, token, user };
  }

  it("reverts TransferFailed when the token's transferFrom returns false (no burn/mint from nothing)", async function () {
    // MockFalseTransferToken reports ample balance + allowance but transferFrom returns false —
    // the exact "lying" non-reverting ERC20 the return-value check must catch. Chain 1 (Ethereum)
    // is supported by default in initialize.
    const { bridge, user } = await deploy("MockFalseTransferToken");

    await expect(
      bridge.connect(user).bridgeOut(ethers.parseEther("10"), 1)
    ).to.be.revertedWithCustomError(bridge, "TransferFailed");
  });
});
