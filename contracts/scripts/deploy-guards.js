// SPDX-License-Identifier: MIT
//
// sp990 — deploy-time safety guards, extracted as pure functions so they can be
// unit-tested in isolation (the deploy scripts that use them run under
// `hardhat run` and can't be imported without side effects).

/**
 * Assert that the EOA actually signing a LIVE deploy equals the intended owner.
 *
 * The CreatorStakeRegistry commissioning ceremony deploys as the deployer EOA
 * (CREATOR_STAKE_OWNER = that EOA) and then transfers ownership to the Foundation
 * Safe. If the resolved signer differs from CREATOR_STAKE_OWNER, the WRONG KEY is
 * about to sign an irreversible mainnet deploy — most commonly because hardhat's
 * `base` network resolves accounts via `MAINNET_PRIVATE_KEY || PRIVATE_KEY`, so a
 * stale MAINNET_PRIVATE_KEY in contracts/.env silently wins over an inline
 * PRIVATE_KEY the operator set on the command line.
 *
 * Throws (aborting before any on-chain tx) on mismatch. No-op on the in-process
 * hardhat/localhost networks, when CREATOR_STAKE_OWNER is unset (owner defaults to
 * the deployer, so they trivially match), or when ALLOW_OWNER_DEPLOYER_MISMATCH=1
 * is set to deliberately deploy with owner != deployer (e.g. owner set directly to
 * a Safe).
 *
 * @param {object} opts
 * @param {string} opts.network         hre.network.name
 * @param {string} opts.deployerAddress the EOA the signer actually resolved to
 * @param {string} opts.owner           the resolved initialOwner (CREATOR_STAKE_OWNER || deployer)
 * @param {object} [opts.env]           env map (defaults to process.env; injectable for tests)
 */
function assertSignerMatchesOwner({ network, deployerAddress, owner, env = process.env }) {
  if (network === "hardhat" || network === "localhost") return;
  if (!env.CREATOR_STAKE_OWNER) return;
  if (env.ALLOW_OWNER_DEPLOYER_MISMATCH === "1") return;
  if (String(deployerAddress).toLowerCase() !== String(owner).toLowerCase()) {
    throw new Error(
      `Signer/owner mismatch: this deploy will be SIGNED by ${deployerAddress} ` +
      `but CREATOR_STAKE_OWNER is ${owner}. On a live deploy these MUST match ` +
      `(deploy as the owner EOA, then transferOwnership to the Safe). Likely ` +
      `cause: the '${network}' network resolved a different key than intended — ` +
      `e.g. a stale MAINNET_PRIVATE_KEY in contracts/.env wins over an inline ` +
      `PRIVATE_KEY (base uses MAINNET_PRIVATE_KEY || PRIVATE_KEY). Set ` +
      `MAINNET_PRIVATE_KEY to your deployer key, or pass ` +
      `ALLOW_OWNER_DEPLOYER_MISMATCH=1 to deploy with a different owner on purpose.`
    );
  }
}

module.exports = { assertSignerMatchesOwner };
