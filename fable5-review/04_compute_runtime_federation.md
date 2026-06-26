# Fable 5 Review — Domain 04: Compute runtime, sandbox, federation & performance

Adversarial, cross-cutting review. See `00_INDEX.md` for shared framing + output format.

## Context

Beyond inference (Domain 03), the compute layer includes: a WASM/SPRK execution runtime
(operator-supplied compute units run sandboxed, intended via Wasmtime), federated-learning
coordination, and the performance/scalability/collaboration subsystems. The security-critical
piece here is **sandbox isolation** — untrusted operator code/models must not escape the
sandbox or exfiltrate data; everything else is correctness + resource-safety.

### Read
- The WASM/SPRK runtime: `prsm/compute/wasm/` (profiler + runtime), `prsm/compute/plugins/`,
  and any Wasmtime/sandbox host-function boundary
- `prsm/compute/federation/` (federated learning: aggregation, participant coordination,
  any secure-aggregation / DP claims)
- `prsm/compute/performance/`, `prsm/compute/scalability/`, `prsm/compute/collaboration/`,
  `prsm/compute/chronos/`, `prsm/compute/diffing/`, `prsm/compute/scheduling/` (the non-inference
  scheduling parts)

### Invariants — confirm or break
1. **Sandbox isolation.** Untrusted compute units cannot: read host filesystem/env/secrets,
   make arbitrary network calls, exhaust host memory/CPU beyond configured limits, or call
   host functions outside the allowed surface. Audit every host-function / import exposed to
   the guest and the resource limits (fuel/memory/time). Find an escape or an unbounded
   resource the guest controls.
2. **Federated-learning integrity.** A malicious participant can't poison the aggregate beyond
   the model's stated robustness, forge another participant's contribution, or learn data it
   shouldn't (if secure-aggregation / DP is claimed, verify the claim holds — or flag the gap
   honestly). Check the aggregation math + the participant-authentication.
3. **No untrusted-input → host code-path.** Anything deserialized from a peer/operator (model
   bytes, compute-unit bytecode, task params, aggregation updates) is validated before use;
   no pickle/eval/dynamic-import on untrusted data; size/shape bounds enforced.
4. **Resource accounting honesty.** Performance/scalability accounting (tflops, memory,
   capacity) that feeds routing or payment can't be gamed by a self-reported value — confirm
   where these are trusted and whether that trust is justified or backstopped.
5. **Fail-closed on the security-critical paths.** A sandbox-setup error, a missing limit, or
   a malformed compute unit must refuse to run (not run unsandboxed). Find a fail-OPEN path.

### Hunt list
- `subprocess` / `os.system` / `eval` / `exec` / `pickle.loads` / dynamic `import` on any data
  that originated off-box.
- Wasmtime config: is fuel/epoch-interruption + memory limit + WASI capability set actually
  applied, or is there a path that instantiates without them?
- Temp-file / path handling for staged models or compute units (path traversal, symlink).
- Concurrency in aggregation / scheduling: races that drop or double-count contributions.
- The boundary between "trusted local" and "untrusted remote" code — is it crisp, and does any
  remote-supplied artifact cross it without validation?

Follow the `00_INDEX.md` output format. Report only.
