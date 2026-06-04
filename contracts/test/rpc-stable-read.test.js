const { expect } = require("chai");
const { stable } = require("../scripts/rpc-stable-read");

// sp992 — `stable()` defeats load-balanced-RPC read-after-write staleness (a
// replica a block behind returns the pre-write value, making a correct contract
// look broken). It returns only once two consecutive reads agree. The in-process
// hardhat network can't reproduce the lag, so the lag paths are unit-tested here
// with a scripted thunk + an injected no-op sleep (fast).

const NO_SLEEP = { sleep: () => Promise.resolve() };

// A thunk that yields a scripted sequence of values; `THROW` entries throw.
const THROW = Symbol("throw");
function scripted(seq) {
  let i = 0;
  return async () => {
    const v = seq[Math.min(i, seq.length - 1)];
    i += 1;
    if (v === THROW) throw new Error("BAD_DATA (simulated replica lag)");
    return v;
  };
}

describe("stable() — resilient read-after-write (sp992)", function () {
  it("returns immediately when the first two reads agree (synced node)", async function () {
    const t = scripted([5n, 5n, 5n]);
    expect(await stable(t, NO_SLEEP)).to.equal(5n);
  });

  it("ignores a single stale read and returns the post-write value", async function () {
    // Classic lag: read1 = stale pre-write (0), then the write propagates and
    // every later read is the new value (1000). Must return 1000n, not 0n.
    const t = scripted([0n, 1000n, 1000n]);
    expect(await stable(t, NO_SLEEP)).to.equal(1000n);
  });

  it("rides out a flapping replica until two consecutive reads agree", async function () {
    // bounce: new, stale, new, new → first agreeing consecutive pair is (new,new).
    const t = scripted([1000n, 0n, 1000n, 1000n]);
    expect(await stable(t, NO_SLEEP)).to.equal(1000n);
  });

  it("retries through a thrown read (BAD_DATA '0x') then stabilises", async function () {
    const t = scripted([THROW, THROW, 7n, 7n]);
    expect(await stable(t, NO_SLEEP)).to.equal(7n);
  });

  it("does NOT return a value that only ever appeared once (no false-stable)", async function () {
    // 0,0 agree first — the post-write value 1000 appears too late to be confirmed
    // within maxTries=3, so a low-try run stabilises on the consistent early value.
    // (Proves stability is required: a lone differing read is never trusted.)
    const t = scripted([1000n, 0n, 0n]);
    expect(await stable(t, { ...NO_SLEEP, maxTries: 3 })).to.equal(0n);
  });

  it("throws a clear, actionable error when reads never stabilise", async function () {
    // Always-throwing replica → never gets a successful read.
    const t = scripted([THROW]);
    let msg = "";
    try {
      await stable(t, { ...NO_SLEEP, maxTries: 4 });
    } catch (e) {
      msg = e.message;
    }
    expect(msg).to.contain("never stabilised");
    expect(msg).to.contain("BASE_SEPOLIA_RPC_URL"); // points at the fix
  });

  it("works for string values (e.g. addresses), not just BigInt", async function () {
    const addr = "0x" + "a".repeat(40);
    const t = scripted([addr, addr]);
    expect(await stable(t, NO_SLEEP)).to.equal(addr);
  });
});
