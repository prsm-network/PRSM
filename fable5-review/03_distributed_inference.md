# Fable 5 Review — Domain 03: Distributed inference & §7 verifiable compute

Adversarial, cross-cutting review. See `00_INDEX.md` for shared framing + output format.

## Context

PRSM serves model inference single-node and split across multiple hosts ("parallax": a model
is partitioned into layer-slice stages, each stage run by a different operator). The §7
promise is **verifiable inference**: a signed receipt lets anyone confirm who computed which
stage and that the activation chain wasn't tampered — without trusting the orchestrator or any
single operator. Stage-to-stage dispatch uses an RPC wire protocol (`chain_rpc`). This domain
is the correctness + receipt-integrity surface (the attestation tier-gate is Domain 01; the
WASM sandbox + perf is Domain 04; settlement of receipts is Domain 01).

### Read
- `prsm/compute/inference/` — the EXECUTION path: chain executors (e.g. `RpcChainExecutor`,
  `LocalHuggingFaceChainExecutor`, `ParallaxScheduledExecutor`), layer-slice runners,
  `pipeline_receipt.py`, the InferenceReceipt build/verify + `topology_assignment`, model
  staging, `content_tier_gate.py` (Tier B/C decrypt-in-TEE), result consensus
- `prsm/compute/parallax_scheduling/` — routing/DP path selection, GPU pool, the
  ParallaxGPU/topology types (the trust gate `trust_adapter.py` is Domain 01)
- `prsm/compute/chain_rpc/` — the stage dispatch wire protocol (server gate, client, handoff
  token, response binding)
- `prsm/compute/scheduling/`, `prsm/compute/query_orchestrator/`

### Invariants — confirm or break
1. **Receipt chain integrity.** The activation hash chain is the load-bearing primitive: stage
   K's output_activation_hash == stage K+1's input_activation_hash; prompt_hash == stage 0
   input; output_hash == last stage output; the orchestrator signature covers the canonical
   payload. A MITM swapping any intermediate activation, or an operator forging a stage they
   didn't run, must break verification. Find a tamper that still verifies.
2. **Topology attribution can't be forged.** `topology_assignment` (who ran which stage) is
   verifiable by rebuilding the chain definition's stable hash. Confirm an operator can't claim
   credit for (or settlement from) a stage it didn't run, and the per-stage signatures bind the
   stage node_id to the registered pubkey (anchor-verified).
3. **Stage-response binding (chain_rpc).** A stage response must be bound to the expected peer
   (`sender_id == expected_sender`) and the request (request_id / handoff token), so a forged
   or misattributed response is dropped. Find a response-injection or cross-request confusion.
4. **Handoff-token / upstream-token integrity.** The token authorizing a downstream stage is
   signed, deadline-bounded, and anchor-verified; it can't be replayed to a different stage,
   after its deadline, or against a different settler. Find a token replay/forge.
5. **Confidential-content decrypt gate.** Tier B/C content decrypts only inside an attested TEE
   context (`open_tier_b`/content_tier_gate requires `ctx.is_attested`). Confirm no path
   decrypts confidential content in a software context, and the gate's `is_attested` derives
   from the real execution-time TEE type (not a self-claim).
6. **Default-safe + fail-closed.** The bare default serves no real inference unless opted in;
   a privacy_tier ≥ standard request on a software runtime is refused (or warns under the
   documented advisory bypass). Confirm no silent downgrade to software for confidential work.

### Hunt list
- Canonicalization mismatches between sign-time and verify-time of the receipt/payload (sorted
  keys, separators, float/int, encoding) that let a tampered payload recover a valid signature.
- The single-node vs multi-host code paths diverging in what they sign/verify.
- Activation (de)serialization: shape/size mismatch, untrusted-tensor handling, OOM.
- Anything that trusts an operator-supplied field (claimed stage, claimed topology, claimed
  cost) without binding it to a signature or the chain definition.

Follow the `00_INDEX.md` output format. Report only.
