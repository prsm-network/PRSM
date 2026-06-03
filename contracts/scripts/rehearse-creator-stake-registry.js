/*
 * Sprint 980 — CreatorStakeRegistry Sepolia REHEARSAL execution script.
 *
 * Drives the full value-movement lifecycle against a deployed CreatorStakeRegistry
 * and asserts every economic invariant with an explicit PASS/FAIL line. This is the
 * step that de-risks the irreversible Base mainnet ceremony: it proves the DEPLOYED
 * BYTECODE behaves exactly as the audited source + unit tests claim, against a real
 * RPC and real signing — not just the in-process Hardhat VM.
 *
 * It is NOT a deploy script (that is deploy-creator-stake-registry.js) and it never
 * touches mainnet — base/mainnet are hard-blocked below. Run it on:
 *   --network hardhat   one-shot dry-run, full lifecycle incl. the post-delay
 *                       withdraw via evm_increaseTime (no real time waited).
 *   --network sepolia   the real rehearsal. evm_increaseTime is unavailable on a
 *                       live chain, so the time-gated withdraw is a SECOND phase:
 *                       phase 1 runs everything else; you re-run with
 *                       REHEARSAL_PHASE=withdraw after the unbond delay elapses.
 *
 * Self-contained by default: if FTNS_TOKEN_ADDRESS / REGISTRY_ADDRESS are unset it
 * deploys a MockERC20 (testnet FTNS stand-in) + a fresh registry (deployer = owner
 * = slasher) using the 1-DAY MINIMUM unbond delay so the phase-2 wait is as short
 * as the contract allows. To rehearse against an already-deployed registry, pass
 * REGISTRY_ADDRESS (+ FTNS_TOKEN_ADDRESS). For a single-key testnet, one account
 * plays creator/slasher/owner — the contract permits this and the value math is
 * identical; pass CREATOR_PRIVATE_KEY to rehearse the role separation faithfully.
 *
 * Env:
 *   REGISTRY_ADDRESS      attach to this registry instead of deploying fresh.
 *   FTNS_TOKEN_ADDRESS    attach to this ERC-20 (else a MockERC20 is deployed).
 *   CREATOR_PRIVATE_KEY   optional distinct creator signer (else = deployer).
 *   REHEARSAL_PHASE       "withdraw" → run ONLY the post-delay withdraw assertion
 *                         (real-testnet phase 2). Requires REGISTRY_ADDRESS +
 *                         FTNS_TOKEN_ADDRESS from the phase-1 printout.
 *
 * Usage:
 *   npx hardhat run scripts/rehearse-creator-stake-registry.js --network hardhat
 *   PRIVATE_KEY=0x... SEPOLIA_RPC_URL=https://... \
 *     npx hardhat run scripts/rehearse-creator-stake-registry.js --network sepolia
 *   # ...wait out the unbond delay, then:
 *   PRIVATE_KEY=0x... SEPOLIA_RPC_URL=https://... \
 *   REGISTRY_ADDRESS=0x... FTNS_TOKEN_ADDRESS=0x... REHEARSAL_PHASE=withdraw \
 *     npx hardhat run scripts/rehearse-creator-stake-registry.js --network sepolia
 */
const hre = require("hardhat");

const DAY = 24 * 60 * 60;
const ONE = 10n ** 18n;
const MIN_HIGH = 1000n * ONE; // mirrors MIN_HIGH_TIER_STAKE_WEI

let _pass = 0;
let _fail = 0;

function ok(name, cond, detail) {
  if (cond) {
    _pass += 1;
    console.log(`   ✓ ${name}${detail ? ` — ${detail}` : ""}`);
  } else {
    _fail += 1;
    console.log(`   ✗ FAIL: ${name}${detail ? ` — ${detail}` : ""}`);
  }
}

/** Assert a tx reverts. Returns true if it did. */
async function reverts(promiseFactory) {
  try {
    const tx = await promiseFactory();
    await tx.wait();
    return false;
  } catch (_e) {
    return true;
  }
}

/** Try to fast-forward chain time. Returns false on a live chain (no cheatcode). */
async function tryTimeTravel(seconds) {
  try {
    await hre.network.provider.send("evm_increaseTime", [seconds]);
    await hre.network.provider.send("evm_mine", []);
    return true;
  } catch (_e) {
    return false;
  }
}

async function main() {
  const network = hre.network.name;
  if (network === "mainnet" || network === "base") {
    throw new Error(
      `Rehearsal on ${network} is BLOCKED — this script is for --network sepolia ` +
      `(rehearsal) or --network hardhat (dry-run) only. Mainnet is the irreversible ` +
      `ceremony; rehearse first.`
    );
  }

  const [deployer] = await hre.ethers.getSigners();
  const chainId = (await hre.ethers.provider.getNetwork()).chainId;
  console.log(`\n=== CreatorStakeRegistry REHEARSAL on ${network} (chain ${chainId}) ===`);
  console.log(`Deployer/owner/slasher: ${deployer.address}`);

  // Creator signer: distinct key if provided, else the deployer (single-key rehearsal).
  let creator = deployer;
  if (process.env.CREATOR_PRIVATE_KEY) {
    creator = new hre.ethers.Wallet(process.env.CREATOR_PRIVATE_KEY, hre.ethers.provider);
    console.log(`Creator:                ${creator.address} (distinct key)`);
  } else {
    console.log(`Creator:                ${creator.address} (= deployer; single-key rehearsal)`);
  }

  // A "stranger" signer for the non-slasher-rejection probe: must be neither the
  // creator nor (later) the slasher. On hardhat there are 20 funded signers; on a
  // single-key testnet there is none, so that one probe is skipped (it is exhaustively
  // unit-tested anyway). NOTE: the probe MUST come from a genuine non-slasher, else a
  // single-key run would actually execute the slash and corrupt the withdraw amount.
  const allSigners = await hre.ethers.getSigners();
  let stranger = allSigners.find(
    (s) => s.address.toLowerCase() !== creator.address.toLowerCase() &&
           s.address.toLowerCase() !== deployer.address.toLowerCase()
  ) || null;

  // ── Resolve / deploy the token + registry ────────────────────────────────
  const MockERC20 = await hre.ethers.getContractFactory("MockERC20");
  let token;
  if (process.env.FTNS_TOKEN_ADDRESS) {
    token = MockERC20.attach(process.env.FTNS_TOKEN_ADDRESS);
    console.log(`FTNS token (attached):  ${process.env.FTNS_TOKEN_ADDRESS}`);
  }

  const Reg = await hre.ethers.getContractFactory("CreatorStakeRegistry");
  let reg;
  if (process.env.REGISTRY_ADDRESS) {
    reg = Reg.attach(process.env.REGISTRY_ADDRESS);
    console.log(`Registry (attached):    ${process.env.REGISTRY_ADDRESS}`);
    if (!token) {
      const onFtns = await reg.ftns();
      token = MockERC20.attach(onFtns);
      console.log(`FTNS token (from reg):  ${onFtns}`);
    }
  }

  // ── Phase 2 (withdraw-only): real-testnet post-delay step ────────────────
  if (process.env.REHEARSAL_PHASE === "withdraw") {
    if (!reg) throw new Error("REHEARSAL_PHASE=withdraw requires REGISTRY_ADDRESS");
    console.log(`\n── PHASE 2: post-delay withdraw ──`);
    const s = await reg.stakes(creator.address);
    const now = BigInt((await hre.ethers.provider.getBlock("latest")).timestamp);
    if (s.status !== 2n) {
      throw new Error(
        `creator is not UNBONDING (status=${s.status}); run phase 1 first (it must ` +
        `leave the creator in UNBONDING with funds remaining).`
      );
    }
    if (now < s.unbondEligibleAt) {
      throw new Error(
        `unbond delay not elapsed: now=${now} < eligibleAt=${s.unbondEligibleAt} ` +
        `(${Number(s.unbondEligibleAt - now)}s left). Wait, then re-run.`
      );
    }
    const before = await token.balanceOf(creator.address);
    const owed = s.amount;
    const tx = await reg.connect(creator).withdraw();
    await tx.wait();
    const after = await token.balanceOf(creator.address);
    ok("withdraw after delay returns the remaining bonded FTNS",
      after - before === owed, `returned ${hre.ethers.formatEther(owed)} FTNS`);
    const sAfter = await reg.stakes(creator.address);
    ok("status flips to WITHDRAWN", sAfter.status === 3n, `status=${sAfter.status}`);
    return summarize();
  }

  // Fresh deploy if nothing was attached.
  if (!token) {
    console.log(`\n[deploy] MockERC20 (testnet FTNS stand-in)…`);
    token = await MockERC20.deploy();
    await token.waitForDeployment();
    console.log(`   token: ${await token.getAddress()}`);
  }
  if (!reg) {
    const delay = DAY; // 1-day minimum → shortest possible phase-2 wait
    console.log(`\n[deploy] CreatorStakeRegistry (owner=slasher=deployer, delay=${delay}s/1d)…`);
    reg = await Reg.deploy(
      deployer.address, await token.getAddress(), delay, deployer.address
    );
    await reg.waitForDeployment();
    console.log(`   registry: ${await reg.getAddress()}`);
  }

  const regAddr = await reg.getAddress();
  const tokenAddr = await token.getAddress();
  const delay = await reg.unbondDelaySeconds();
  const slasher = await reg.slasher();
  const slasherIsDeployer = slasher.toLowerCase() === deployer.address.toLowerCase();

  // Fund + approve the creator for 2× MIN (so we can slash MIN and still withdraw MIN).
  console.log(`\n── PHASE 1: bond / slash / drain / unbond lifecycle ──`);
  const need = 2n * MIN_HIGH;
  const haveCreator = await token.balanceOf(creator.address);
  if (haveCreator < need) {
    // MockERC20.mint is open in tests; on a real custom FTNS this would be a faucet/transfer.
    const mintTx = await token.mint(creator.address, need - haveCreator);
    await mintTx.wait();
  }
  const apprTx = await token.connect(creator).approve(regAddr, need);
  await apprTx.wait();

  // INV-1: stake pulls FTNS and records the bonded balance.
  const regBalBefore = await token.balanceOf(regAddr);
  await (await reg.connect(creator).stake(need)).wait();
  ok("stake: creatorStakeOf == bonded amount",
    (await reg.creatorStakeOf(creator.address)) === need,
    `${hre.ethers.formatEther(need)} FTNS`);
  ok("stake: registry token balance increased by the staked amount",
    (await token.balanceOf(regAddr)) - regBalBefore === need);

  // INV-4: slash (while BONDED) reduces stake + credits the Foundation reserve.
  if (slasherIsDeployer) {
    await (await reg.connect(deployer).slash(creator.address, MIN_HIGH, "rehearsal-spam")).wait();
    ok("slash: bonded stake reduced by the slashed amount",
      (await reg.creatorStakeOf(creator.address)) === MIN_HIGH);
    ok("slash: foundationReserveBalance credited",
      (await reg.foundationReserveBalance()) === MIN_HIGH);
    // Probe non-slasher rejection from a genuine stranger (NOT the creator — in a
    // single-key run creator == slasher, so that probe would actually slash).
    const strangerNonSlasher =
      stranger && stranger.address.toLowerCase() !== slasher.toLowerCase()
        ? stranger : null;
    if (strangerNonSlasher) {
      ok("slash: rejected from a non-slasher",
        await reverts(() => reg.connect(strangerNonSlasher).slash(creator.address, 1n, "x")));
    } else {
      console.log(`   (non-slasher-rejection probe skipped — no distinct non-slasher ` +
        `account available; covered by the unit suite)`);
    }
  } else {
    console.log(`   (slash steps skipped — slasher ${slasher} is not the deployer; ` +
      `run with the slasher key to rehearse slash/drain)`);
  }

  // INV-6: drain accrued reserve to a CONTRACT wallet (sp979 WalletNotContract guard).
  if (slasherIsDeployer) {
    ok("setFoundationReserveWallet: rejects an EOA (sp979 WalletNotContract)",
      await reverts(() => reg.connect(deployer).setFoundationReserveWallet(creator.address)));
    // Use the token contract itself as the contract-address stand-in for the Safe.
    await (await reg.connect(deployer).setFoundationReserveWallet(tokenAddr)).wait();
    const reserveBefore = await token.balanceOf(tokenAddr);
    const accrued = await reg.foundationReserveBalance();
    await (await reg.connect(deployer).drainFoundationReserve()).wait();
    ok("drain: reserve wallet received the accrued proceeds",
      (await token.balanceOf(tokenAddr)) - reserveBefore === accrued);
    ok("drain: foundationReserveBalance zeroed",
      (await reg.foundationReserveBalance()) === 0n);
  }

  // INV-2: requestUnbond drops eligibility to 0 immediately.
  await (await reg.connect(creator).requestUnbond()).wait();
  ok("requestUnbond: creatorStakeOf drops to 0 immediately (no tier while exiting)",
    (await reg.creatorStakeOf(creator.address)) === 0n);

  // INV-3: withdraw before the delay reverts.
  ok("withdraw before delay reverts",
    await reverts(() => reg.connect(creator).withdraw()));

  // INV-5: pause freezes mutating ops (sp979). Smoke: stake reverts while paused;
  // the exhaustive 5-function matrix is covered by the unit suite.
  if (slasherIsDeployer) {
    await (await reg.connect(deployer).pause()).wait();
    ok("pause: stake reverts while paused (sp979 whenNotPaused)",
      await reverts(() => reg.connect(creator).stake(1n)));
    ok("pause: requestUnbond reverts while paused (sp979 whenNotPaused)",
      await reverts(() => reg.connect(creator).requestUnbond()));
    await (await reg.connect(deployer).unpause()).wait();
  }

  // INV-7: withdraw after the delay returns the remaining bonded FTNS.
  const traveled = await tryTimeTravel(Number(delay) + 1);
  if (traveled) {
    const s = await reg.stakes(creator.address);
    const owed = s.amount;
    const before = await token.balanceOf(creator.address);
    await (await reg.connect(creator).withdraw()).wait();
    ok("withdraw after delay returns the remaining bonded FTNS",
      (await token.balanceOf(creator.address)) - before === owed,
      `returned ${hre.ethers.formatEther(owed)} FTNS`);
    ok("status flips to WITHDRAWN",
      (await reg.stakes(creator.address)).status === 3n);
  } else {
    const s = await reg.stakes(creator.address);
    console.log(`\n   ⏳ Live chain — cannot fast-forward time. The creator is UNBONDING;`);
    console.log(`      withdraw is eligible at unix ${s.unbondEligibleAt}.`);
    console.log(`      After that, run PHASE 2 to finish the rehearsal:`);
    console.log(`        REGISTRY_ADDRESS=${regAddr} \\`);
    console.log(`        FTNS_TOKEN_ADDRESS=${tokenAddr} \\`);
    console.log(`        REHEARSAL_PHASE=withdraw \\`);
    console.log(`          npx hardhat run scripts/rehearse-creator-stake-registry.js --network ${network}`);
  }

  return summarize();
}

function summarize() {
  console.log(`\n=== REHEARSAL RESULT: ${_pass} passed, ${_fail} failed ===`);
  if (_fail > 0) {
    throw new Error(
      `${_fail} invariant(s) FAILED — the deployed bytecode does not match the ` +
      `audited contract. Do NOT proceed to the mainnet ceremony.`
    );
  }
  console.log(`All asserted invariants hold. Safe to proceed to the gated ceremony steps.`);
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
