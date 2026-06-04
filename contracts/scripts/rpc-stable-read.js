// SPDX-License-Identifier: MIT
//
// sp992 — resilient on-chain reads for live, load-balanced RPCs. Extracted as a
// pure, injectable function so the lag path (which the in-process hardhat network
// cannot reproduce) is unit-tested. Used by the rehearsal driver.

/**
 * Read on-chain state resiliently against a load-balanced public RPC. After a
 * write, ethers' tx.wait() returns when the tx is mined, but a subsequent eth_call
 * can be served by a read-replica node that is a block behind — returning the
 * STALE pre-write value (or "0x" → a thrown decode error). That makes a correct
 * contract look broken (e.g. "slash didn't reduce the stake" even though the slash
 * tx mined). `stable()` calls the read repeatedly and returns only once TWO
 * CONSECUTIVE reads agree, so a lone stale/lagging read can't fool a caller.
 *
 * It is UNBIASED — it stabilises on whatever the consistent on-chain value is,
 * never compared against an expected result. On a synced node (incl. the
 * in-process hardhat network) the first two back-to-back reads agree and it returns
 * immediately. The thunk MUST return a comparable scalar (BigInt / number / string)
 * — for struct getters, project the field inside the thunk.
 *
 * @param {() => Promise<any>} thunk      the read (e.g. () => reg.creatorStakeOf(a))
 * @param {object} [opts]
 * @param {(ms:number)=>Promise<void>} [opts.sleep]  injectable sleep (tests pass a no-op)
 * @param {number} [opts.maxTries=40]     attempts before giving up
 * @param {number} [opts.changeDelayMs=1500]  wait after a value changes (propagation)
 * @param {number} [opts.errorDelayMs=2000]   wait after a thrown read (replica lag)
 */
async function stable(thunk, opts = {}) {
  const sleep = opts.sleep || ((ms) => new Promise((r) => setTimeout(r, ms)));
  const maxTries = opts.maxTries || 40;
  const changeDelayMs = opts.changeDelayMs ?? 1500;
  const errorDelayMs = opts.errorDelayMs ?? 2000;
  let prev = null;
  let have = false;
  for (let i = 0; i < maxTries; i++) {
    let cur;
    try {
      cur = await thunk();
    } catch (_e) {
      // lagging replica (e.g. BAD_DATA "0x") — wait for propagation + retry.
      have = false;
      await sleep(errorDelayMs);
      continue;
    }
    if (have && cur === prev) return cur; // two consecutive agreeing reads
    if (have && cur !== prev) await sleep(changeDelayMs); // value moved → let it propagate
    prev = cur;
    have = true;
  }
  if (have) return prev;
  throw new Error(
    "stable(): RPC reads never stabilised — the endpoint is too lagged for a " +
    "read-after-write rehearsal. Re-run with a dedicated BASE_SEPOLIA_RPC_URL " +
    "(Alchemy/Infura/QuickNode), not the public load-balanced default."
  );
}

/**
 * Read on-chain state PINNED to a specific block — the deterministic cure for
 * load-balanced-RPC read-after-write lag. After a write mined in block B, reading
 * state at `{ blockTag: B }` always returns the post-write value: state at a fixed
 * past block is immutable. A replica whose head is still below B cannot serve block
 * B — a compliant node errors "block not found" — so we retry until a replica has
 * synced to B. Unlike stable()'s two-agree heuristic, this is immune to UNIFORM lag
 * (all replicas a block behind right after the write), which is what actually broke
 * the live rehearsal.
 *
 * @param {number} blockNumber           the block a write was mined in (receipt.blockNumber)
 * @param {(blockTag:number)=>Promise<any>} makeCall  the pinned read, e.g.
 *        (bt) => reg.creatorStakeOf(addr, { blockTag: bt })
 * @param {object} [opts] {sleep, maxTries, errorDelayMs}
 */
async function readAt(blockNumber, makeCall, opts = {}) {
  const sleep = opts.sleep || ((ms) => new Promise((r) => setTimeout(r, ms)));
  const maxTries = opts.maxTries || 40;
  const errorDelayMs = opts.errorDelayMs ?? 2000;
  let lastErr;
  for (let i = 0; i < maxTries; i++) {
    try {
      return await makeCall(blockNumber);
    } catch (e) {
      // replica head < blockNumber ("block not found"), or transient — wait + retry.
      lastErr = e;
      await sleep(errorDelayMs);
    }
  }
  throw new Error(
    `readAt(${blockNumber}): no replica synced to that block after ${maxTries} ` +
    `tries (last: ${lastErr && lastErr.message}). Use a dedicated BASE_SEPOLIA_RPC_URL.`
  );
}

module.exports = { stable, readAt };
