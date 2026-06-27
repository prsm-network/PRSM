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
 * Optional env:
 *   FROM_BLOCK             event-scan start block (default 0; set to the registry's
 *                          deploy block to scan faster on mainnet)
 *   SCAN_CHUNK_BLOCKS      blocks per queryFilter page (default 50000)
 *                          NOTE: public free-tier RPCs (e.g. https://mainnet.base.org)
 *                          cap eth_getLogs at ~10 blocks — scanning the full history
 *                          there is infeasible. Point BASE_RPC_URL at a PAYG endpoint
 *                          (Alchemy/Infura/QuickNode) and set FROM_BLOCK to the
 *                          registry's deploy block for a fast, complete scan.
 *   ALLOW_PENDING=1        report-only: exit 0 even if PENDING batches remain
 *
 * Usage:
 *   OLD_REGISTRY_ADDRESS=0x48fFab641b9D638F312FFA776818756a326F995B \
 *     FROM_BLOCK=12345678 \
 *     npx hardhat run scripts/verify-f-activation-cutover-readiness.js --network base
 *
 * Exit codes:
 *   0 = old registry has ZERO pending batches → safe to cut over (or ALLOW_PENDING=1)
 *   1 = pending batches remain (NOT safe to cut over) / bad config
 */
const hre = require("hardhat");

const STATUS = { 0: "NONEXISTENT", 1: "PENDING", 2: "FINALIZED", 3: "VOIDED" };

async function main() {
  const addrRaw = process.env.OLD_REGISTRY_ADDRESS;
  if (!addrRaw) throw new Error("OLD_REGISTRY_ADDRESS env var required");
  const registryAddr = hre.ethers.getAddress(addrRaw);
  const fromBlock = parseInt(process.env.FROM_BLOCK || "0", 10);
  const chunk = parseInt(process.env.SCAN_CHUNK_BLOCKS || "50000", 10);
  const allowPending = process.env.ALLOW_PENDING === "1";

  const code = await hre.ethers.provider.getCode(registryAddr);
  if (code === "0x" || code === "0x0") {
    throw new Error(`no bytecode at OLD_REGISTRY_ADDRESS ${registryAddr} on ${hre.network.name}`);
  }

  const registry = await hre.ethers.getContractAt("BatchSettlementRegistry", registryAddr);
  const latest = await hre.ethers.provider.getBlockNumber();
  const now = BigInt((await hre.ethers.provider.getBlock(latest)).timestamp);

  console.log(`\n=== F-activation cutover readiness — old registry drain check ===`);
  console.log(`Network:   ${hre.network.name}`);
  console.log(`Registry:  ${registryAddr}`);
  console.log(`Scanning BatchCommitted events blocks ${fromBlock}..${latest} (chunk ${chunk})`);

  // ── Collect every committed batchId (chunked to respect RPC log-range caps) ──
  const filter = registry.filters.BatchCommitted();
  const batchIds = new Set();
  for (let start = fromBlock; start <= latest; start += chunk) {
    const end = Math.min(start + chunk - 1, latest);
    let events;
    try {
      events = await registry.queryFilter(filter, start, end);
    } catch (e) {
      throw new Error(
        `queryFilter ${start}..${end} failed (${e.shortMessage || e.message}). ` +
        `Lower SCAN_CHUNK_BLOCKS or set FROM_BLOCK to the registry's deploy block.`,
      );
    }
    for (const ev of events) batchIds.add(ev.args.batchId);
  }
  console.log(`Committed batches found: ${batchIds.size}`);

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
    if (now >= earliestFinalize) {
      pendingElapsed += 1;
    } else {
      pendingInWindow += 1;
    }
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
    console.log(
      `    ${p.id.slice(0, 18)}…  ${p.finalizable ? "FINALIZE NOW" : `wait ~${p.secondsLeft}s`}`,
    );
  }
  if (totalPending > pendingSample.length) {
    console.log(`    … and ${totalPending - pendingSample.length} more`);
  }

  if (totalPending === 0) {
    console.log(`\n✅ Old registry is DRAINED (0 pending) — safe to cut over to the new bundle.`);
    process.exitCode = 0;
    return;
  }

  console.log(
    `\n${allowPending ? "ℹ" : "❌"} ${totalPending} pending batch(es) remain. Finalize the ` +
    `window-elapsed ones now (finalizeBatch) + wait out the rest BEFORE re-pointing clients / ` +
    `draining the old escrow (runbook §4.6).`,
  );
  process.exitCode = allowPending ? 0 : 1;
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
