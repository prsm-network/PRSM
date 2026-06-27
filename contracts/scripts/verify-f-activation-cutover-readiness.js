/*
 * sp1292 — pre-cutover safety check for the TEE Tier-3 roadmap-F migration.
 *
 * Activating F on the production settlement path is a FRESH audit-bundle deploy + a
 * cutover (the registry is Ownable2Step/immutable-cross-wired, not a proxy — see
 * docs/2026-06-26-tee-tier3-f-activation-deploy-runbook.md). §4.2/§4.6 of that runbook
 * require that the OLD registry's PENDING batches all finalize before clients are
 * re-pointed + the old escrow drained — cutting over with unsettled value in the old
 * registry would strand it. No existing verify script checks this; this one does.
 *
 * Read-only (no key, no signing): scans the old registry's BatchCommitted events, then
 * reads each batch's AUTHORITATIVE on-chain status via the public `batches()` getter,
 * and reports how many are still PENDING (the cutover gate) — splitting them into
 * "window elapsed (finalizable now)" vs "still in challenge window (must wait)".
 *
 * Required env:
 *   OLD_REGISTRY_ADDRESS   the live BatchSettlementRegistry being retired
 *                          (Base mainnet today: 0x48fFab641b9D638F312FFA776818756a326F995B)
 * The hard cutover gate (exit 0) requires ALL of: zero PENDING batches; the scan is
 * complete (per-provider event count == providerBatchSequence); on mainnet the old
 * registry is paused() (so the reading is durable — sp1294 H1); and, if
 * REQUIRE_ESCROW_DRAINED=1, the old EscrowPool residual is 0. Residual escrow + stake are
 * otherwise reported INFORMATIONALLY (recoverable-not-stranded — EscrowPool has no admin
 * drain by design, so funds are always recoverable via depositor withdraw()).
 *
 * Optional env:
 *   FROM_BLOCK             event-scan start block. REQUIRED on mainnet (a from-0 scan on a
 *                          capped RPC risks silent undercount) — set to the deploy block.
 *   SCAN_CHUNK_BLOCKS      blocks per queryFilter page (default 50000)
 *                          NOTE: public free-tier RPCs (e.g. https://mainnet.base.org)
 *                          cap eth_getLogs at ~10 blocks — scanning the full history
 *                          there is infeasible. Point BASE_RPC_URL at a PAYG endpoint
 *                          (Alchemy/Infura/QuickNode) for a fast, complete scan.
 *   PERMIT_UNPAUSED=1      skip the mainnet paused() requirement (testnet / report-only)
 *   REQUIRE_ESCROW_DRAINED=1  also hard-gate on old EscrowPool totalEscrowedBalance==0
 *   ALLOW_PENDING=1        report-only: exit 0 even if the gate is not met
 *
 * Usage:
 *   OLD_REGISTRY_ADDRESS=0x48fFab641b9D638F312FFA776818756a326F995B \
 *     FROM_BLOCK=45687572 \
 *     npx hardhat run scripts/verify-f-activation-cutover-readiness.js --network base
 *
 * Exit codes:
 *   0 = hard gate met (0 PENDING + scan complete + paused-on-mainnet + escrow-if-required)
 *       → safe to cut over (or ALLOW_PENDING=1 report-only)
 *   1 = gate not met (pending remain / not paused / scan incomplete) / bad config
 */
const hre = require("hardhat");

const STATUS = { 0: "NONEXISTENT", 1: "PENDING", 2: "FINALIZED", 3: "VOIDED" };

async function main() {
  const addrRaw = process.env.OLD_REGISTRY_ADDRESS;
  if (!addrRaw) throw new Error("OLD_REGISTRY_ADDRESS env var required");
  const registryAddr = hre.ethers.getAddress(addrRaw);
  const chunk = parseInt(process.env.SCAN_CHUNK_BLOCKS || "50000", 10);
  const allowPending = process.env.ALLOW_PENDING === "1";          // report-only
  const permitUnpaused = process.env.PERMIT_UNPAUSED === "1";       // skip the H1 pause gate
  const requireEscrowDrained = process.env.REQUIRE_ESCROW_DRAINED === "1"; // make escrow a hard gate

  const net = await hre.ethers.provider.getNetwork();
  const chainId = net.chainId;
  const isMainnet = new Set([8453n, 1n, 137n]).has(chainId);

  // sp1294 (M4): on mainnet, refuse the default FROM_BLOCK=0 — a from-genesis scan on a
  // capped RPC risks silently undercounting PENDING batches. Force an explicit anchor.
  const fromBlockRaw = process.env.FROM_BLOCK;
  if (isMainnet && (fromBlockRaw === undefined || fromBlockRaw.trim() === "")) {
    throw new Error(
      "FROM_BLOCK is required on mainnet — set it to the registry's deploy block. " +
      "Refusing a from-0 scan that could silently undercount pending batches.",
    );
  }
  const fromBlock = parseInt(fromBlockRaw || "0", 10);

  const code = await hre.ethers.provider.getCode(registryAddr);
  if (code === "0x" || code === "0x0") {
    throw new Error(`no bytecode at OLD_REGISTRY_ADDRESS ${registryAddr} on ${hre.network.name}`);
  }

  const registry = await hre.ethers.getContractAt("BatchSettlementRegistry", registryAddr);
  const latest = await hre.ethers.provider.getBlockNumber();
  const now = BigInt((await hre.ethers.provider.getBlock(latest)).timestamp);

  console.log(`\n=== F-activation cutover readiness — old registry drain check ===`);
  console.log(`Network:   ${hre.network.name} (chainId ${chainId})`);
  console.log(`Registry:  ${registryAddr}`);

  // sp1294 (H1): the drain reading is only DURABLE if the old registry is paused — else a
  // permissionless commitBatch can land a fresh PENDING batch AFTER this scan (TOCTOU). On
  // mainnet, require paused() unless PERMIT_UNPAUSED=1 (testnet/report-only).
  let paused = null;
  try { paused = await registry.paused(); } catch { paused = null; }
  console.log(`Paused:    ${paused === null ? "unknown" : paused}${isMainnet && !paused && !permitUnpaused ? "  ← must be paused for a durable gate" : ""}`);
  console.log(`Scanning BatchCommitted blocks ${fromBlock}..${latest} (chunk ${chunk})`);

  // ── Collect committed batchIds + per-provider event counts (for completeness) ──
  const filter = registry.filters.BatchCommitted();
  const batchIds = new Set();
  const providerEventCounts = new Map();
  for (let start = fromBlock; start <= latest; start += chunk) {
    const end = Math.min(start + chunk - 1, latest);
    let events;
    try {
      events = await registry.queryFilter(filter, start, end);
    } catch (e) {
      throw new Error(
        `queryFilter ${start}..${end} failed (${e.shortMessage || e.message}). ` +
        `Lower SCAN_CHUNK_BLOCKS or use a PAYG RPC + FROM_BLOCK=deploy block.`,
      );
    }
    for (const ev of events) {
      batchIds.add(ev.args.batchId);
      const p = ev.args.provider;
      providerEventCounts.set(p, (providerEventCounts.get(p) || 0) + 1);
    }
  }
  console.log(`Committed batches found: ${batchIds.size} (providers: ${providerEventCounts.size})`);

  // sp1294 (M4): completeness reconciliation — each provider's BatchCommitted count must
  // equal its on-chain providerBatchSequence (the authoritative monotonic per-provider
  // commit counter). A shortfall proves the event scan missed commits → fail closed.
  let scanComplete = true;
  for (const [p, cnt] of providerEventCounts) {
    let seq = null;
    try { seq = Number(await registry.providerBatchSequence(p)); } catch { seq = null; }
    if (seq !== null && cnt < seq) {
      scanComplete = false;
      console.log(`  ❌ scan INCOMPLETE for provider ${p.slice(0, 10)}…: found ${cnt} events but providerBatchSequence=${seq}`);
    }
  }

  // ── Read each batch's authoritative status (handles any path to FINALIZED/VOIDED) ──
  let pendingElapsed = 0;
  let pendingInWindow = 0;
  const pendingSample = [];
  for (const id of batchIds) {
    const b = await registry.batches(id);
    const status = Number(b.status);
    if (status !== 1) continue; // not PENDING → already terminal, doesn't block cutover
    const commitTs = BigInt(b.commitTimestamp);
    const window = BigInt(b.challengeWindowSecondsAtCommit);
    const earliestFinalize = commitTs + window; // pauses only EXTEND this, so it's a lower bound
    if (now >= earliestFinalize) pendingElapsed += 1; else pendingInWindow += 1;
    if (pendingSample.length < 10) {
      pendingSample.push({
        id, finalizable: now >= earliestFinalize,
        secondsLeft: now >= earliestFinalize ? 0 : Number(earliestFinalize - now),
      });
    }
  }
  const totalPending = pendingElapsed + pendingInWindow;
  console.log(`\nPENDING (unsettled) batches: ${totalPending}`);
  console.log(`  - window elapsed (finalize now): ${pendingElapsed}`);
  console.log(`  - still in challenge window:      ${pendingInWindow}`);
  for (const p of pendingSample) {
    console.log(`    ${p.id.slice(0, 18)}…  ${p.finalizable ? "FINALIZE NOW" : `wait ~${p.secondsLeft}s`}`);
  }
  if (totalPending > pendingSample.length) console.log(`    … and ${totalPending - pendingSample.length} more`);

  // sp1294 (H2): read the OLD EscrowPool residual balance. This is INFORMATIONAL by default,
  // NOT a drain gate — EscrowPool has no admin drain by design (anti owner-drain, L2 HIGH-6),
  // so residual requester escrow is recoverable indefinitely via each depositor's own
  // withdraw(); the old pool must stay UNPAUSED forever for that. Set REQUIRE_ESCROW_DRAINED=1
  // only if you've separately driven it to 0 and want a hard assertion.
  let escrowResidual = null;
  try {
    const escrowAddr = await registry.escrowPool();
    if (escrowAddr && escrowAddr !== hre.ethers.ZeroAddress) {
      const escrow = await hre.ethers.getContractAt("EscrowPool", escrowAddr);
      escrowResidual = await escrow.totalEscrowedBalance();
      console.log(`\nOld EscrowPool ${escrowAddr}:\n  totalEscrowedBalance = ${escrowResidual}  (informational — recoverable via depositor withdraw(), NOT a drain gate)`);
    }
  } catch (e) {
    console.log(`\n  ⚠ could not read old EscrowPool balance: ${e.shortMessage || e.message}`);
  }

  // ── Gate decision ──
  const pendingOk = totalPending === 0;
  const pausedOk = permitUnpaused || !isMainnet || paused === true;
  const escrowOk = !requireEscrowDrained || (escrowResidual !== null && escrowResidual === 0n);
  const hardGateOk = pendingOk && pausedOk && escrowOk && scanComplete;

  if (hardGateOk) {
    console.log(
      `\n✅ Old registry has ZERO PENDING batches${isMainnet && paused ? " and is PAUSED (durable gate)" : ""}. ` +
      `NOTE: this gate covers registry batch settlement ONLY — residual old-EscrowPool balance ` +
      `(${escrowResidual === null ? "unread" : escrowResidual}) + StakeBond stake are SEPARATE and ` +
      `recoverable-not-stranded; account for them per runbook §4.6 before re-pointing clients.`,
    );
    process.exitCode = 0;
    return;
  }

  console.log("\n❌ NOT ready to cut over:");
  if (!scanComplete) console.log("  - event scan is INCOMPLETE (provider-sequence mismatch) — cannot assert drained");
  if (!pendingOk) console.log(`  - ${totalPending} PENDING batch(es) remain — finalize window-elapsed ones (finalizeBatch) + wait out the rest`);
  if (!pausedOk) console.log("  - old registry is NOT paused — the drain reading is not durable (a new commitBatch can land after this scan). Foundation Safe should pause() the old registry first (PERMIT_UNPAUSED=1 for a non-authoritative report)");
  if (requireEscrowDrained && !escrowOk) console.log(`  - REQUIRE_ESCROW_DRAINED: old EscrowPool residual ${escrowResidual} != 0`);
  process.exitCode = allowPending ? 0 : 1;
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
