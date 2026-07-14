/*
 * sp1456 — build the Safe Transaction Builder batches for the OLD-bundle cutover pause + its rollback.
 * PURE OFFLINE: no keys, no RPC, no signing. Emits importable JSON for the Foundation Safe (2-of-3).
 *
 * These are fully pre-buildable NOW (the OLD BSR address is known) — unlike the acceptOwnership batch
 * (build-sp1456-safe-acceptownership.js), which needs the post-deploy new addresses.
 *
 * Emits TWO files:
 *   1. …-pause-old-bsr…    — cutover step 3C.2: pause the OLD BSR so it rejects new commitBatch. Run
 *                            ONLY after the drain gate is clean (0 pending) AND before any old-bond
 *                            withdraw (it closes the unpatched defect-#3 slash-swallow window).
 *   2. …-unpause-old-bsr…  — ROLLBACK: unpause the OLD BSR to restore it as the hot fallback if the
 *                            new bundle fails soak. `_effectiveElapsed` credits paused time, so a
 *                            pause/unpause never robs a challenger of their window.
 *
 * Env (all optional; defaults = live June-2026 F bundle):
 *   OLD_BSR       - the OLD BatchSettlementRegistry to pause (default 0x12a01F6C…391F2)
 *   SAFE_ADDRESS  - the Foundation Safe (default 0x91b0e6F8…5791)
 *   CHAIN_ID      - default 8453 (Base mainnet)
 *   OUT_DIR       - output directory (default deployments/)
 *
 * Usage:  node scripts/build-sp1456-safe-pause.js
 */
const fs = require("fs");
const path = require("path");
const { ethers } = require("ethers");

const DEFAULT_OLD_BSR = "0x12a01F6C487d765af389bC7D95D90b3136a391F2";
const DEFAULT_SAFE = "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791";

// Derive + assert the selectors so an ethers change can't silently emit wrong bytes.
function selector(sig) {
  const s = new ethers.Interface([`function ${sig} external`]).getFunction(sig.split("(")[0]).selector;
  return s;
}

function batch({ chainId, safe, to, name, fn, description }) {
  return {
    version: "1.0",
    chainId: String(chainId),
    createdAt: 0, // deterministic — no Date.now() so re-runs diff cleanly (fill in at import if needed)
    meta: {
      name,
      description,
      txBuilderVersion: "1.16.5",
      createdFromSafeAddress: safe,
    },
    transactions: [{
      to: ethers.getAddress(to),
      value: "0",
      data: fn.selector,
      contractMethod: { inputs: [], name: fn.name, payable: false },
      contractInputsValues: null,
    }],
  };
}

function main() {
  const PAUSE = { name: "pause", selector: selector("pause()") };
  const UNPAUSE = { name: "unpause", selector: selector("unpause()") };
  if (PAUSE.selector !== "0x8456cb59") throw new Error(`pause selector ${PAUSE.selector} != 0x8456cb59`);
  if (UNPAUSE.selector !== "0x3f4ba83a") throw new Error(`unpause selector ${UNPAUSE.selector} != 0x3f4ba83a`);

  const chainId = process.env.CHAIN_ID || "8453";
  const safe = ethers.getAddress(process.env.SAFE_ADDRESS || DEFAULT_SAFE);
  const oldBsr = ethers.getAddress(process.env.OLD_BSR || DEFAULT_OLD_BSR);
  const outDir = process.env.OUT_DIR || path.join(__dirname, "..", "deployments");

  const files = [
    ["safe-batch-sp1456-pause-old-bsr.json", batch({
      chainId, safe, to: oldBsr, fn: PAUSE,
      name: "PRSM sp1456 cutover — PAUSE old BatchSettlementRegistry",
      description: `Foundation Safe (2-of-3) pauses the OLD BSR ${oldBsr} during the sp1456 cutover. ` +
        `Run ONLY after the drain gate is clean (0 pending batches) and BEFORE any old-bond withdraw ` +
        `(closes the unpatched defect-#3 slash-swallow window). Reversible via the unpause batch. ` +
        `Do NOT pause the old EscrowPool (no admin drain → traps funds).`,
    })],
    ["safe-batch-sp1456-unpause-old-bsr-ROLLBACK.json", batch({
      chainId, safe, to: oldBsr, fn: UNPAUSE,
      name: "PRSM sp1456 ROLLBACK — UNPAUSE old BatchSettlementRegistry",
      description: `ROLLBACK: Foundation Safe (2-of-3) unpauses the OLD BSR ${oldBsr} to restore it as ` +
        `the hot fallback if the new bundle fails soak. _effectiveElapsed credits paused time, so this ` +
        `never shortens a challenger's window.`,
    })],
  ];

  for (const [fname, b] of files) {
    const p = path.join(outDir, fname);
    fs.writeFileSync(p, JSON.stringify(b, null, 2));
    console.log(`Wrote ${p}`);
    console.log(`   ${b.transactions[0].contractMethod.name}() → ${b.transactions[0].to}  data ${b.transactions[0].data}`);
  }
  console.log(`\nImport into Safe ${safe} → Apps → Transaction Builder. 2-of-3 sign + execute.`);
  console.log(`pause: cutover step (after drain gate, before any old withdraw). unpause: rollback only.`);
}

main();
