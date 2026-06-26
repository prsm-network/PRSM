# Fable 5 — PRSM Comprehensive Review (scoped set)

A collectively-exhaustive set of **independent, scoped** review prompts. Each covers one
coherent surface of the PRSM codebase with its own review lens. Pass them to Fable 5 **one
at a time** (each is self-contained — Fable 5 has no other context). Triage each report
before moving on. Together they cover the whole `prsm/` package (~370k LOC) with minimal
overlap; where two domains touch the same file, the seam is noted in each prompt.

## The domains (suggested order = highest stakes first)

| # | File | Surface | Primary lens |
|---|------|---------|--------------|
| 01 | `01_money_and_trust.md` | settlement + requester-payment/relayer + attestation verify/gate + money-path web3 clients + inference-ingress settle sites | money-safety + crypto-trust |
| 02 | `02_p2p_transport_discovery.md` | transport (WS + libp2p), discovery, gossip, PEX, bootstrap, NAT, heartbeat, `prsm/network`, `prsm/bootstrap` | distributed-systems correctness + network-trust-boundary security |
| 03 | `03_distributed_inference.md` | `compute/inference` (execution), `parallax_scheduling` (routing), `chain_rpc` (wire protocol), `scheduling`, `query_orchestrator`, result-consensus | inference correctness + §7 receipt-chain integrity |
| 04 | `04_compute_runtime_federation.md` | WASM/SPRK sandbox, `compute/performance`, `federation` (federated learning), `scalability`, `collaboration`, `chronos`, `diffing`, `plugins` | sandbox isolation + federation correctness + resource safety |
| 05 | `05_economy_tokenomics_contracts.md` | `economy/{tokenomics,blockchain,web3(non-settlement),payments,governance,economics,pricing}`, `emission`, `governance`, `marketplace` | economic-invariant correctness + contract-client safety |
| 06 | `06_content_storage_data.md` | `data/`, `storage/`, the content/torrent/encryption code in `node` + `core` (Tier A/B/C, CID/infohash, Shamir, AES-GCM) | data-integrity + content-crypto |
| 07 | `07_api_interface_operator.md` | `interface/` (api/public/onboarding/dashboard), `node` API endpoints + lifecycle + health + admin, `cli_modules`, `api`, `sdk` | API-security + input-validation + operator UX |
| 08 | `08_core_security_crypto_enterprise.md` | `core/{cryptography,auth,security,privacy,config,errors,validation,caching,monitoring,integrations}`, `security/`, `observability`, `enterprise/` | foundational-security primitives + CSO cross-cutting |

## Shared conventions (apply to EVERY prompt)

**Framing.** You are an adversarial reviewer with full read access to the repo. Default to
skeptical — try to *break* each invariant, not confirm it. The codebase has had per-change
("per-sprint") reviews already, so **do NOT re-do per-function reviews** — your value is the
**cross-module / end-to-end / architectural** layer those couldn't see. **Do NOT boil the
ocean even within your domain** — go deep on the security/correctness-critical flows named in
the prompt; skim the rest. Prefer "here is the exact line and the exact exploit" over breadth.

**Output format (use this in every report).** For each finding:
- **Severity**: CRITICAL / HIGH / MEDIUM / LOW
- **Location**: `path/file.py:line`
- **Invariant violated** (one named in the prompt, or a new one you identify)
- **Concrete exploit / failure scenario**: step-by-step, attacker capability stated
- **Why it survives existing defenses** (the prior reviews + tests)

Then give an explicit verdict on EACH numbered invariant in the prompt: **HOLDS** /
**VIOLATED** / **CONDITIONAL** (state the condition), with the 1-2 lines of code that decide
it — a clean "HOLDS, here's the enforcing line" is as valuable as a finding. End with: the
single highest-priority fix before this surface carries production load, and anything you
could not determine from static reading that needs a live test. **Do not fix anything — report only.**

## How to use the results

Run a domain → hand the report back to the PRSM maintainer / assistant → each finding is
triaged (confirmed or refuted against the code), and the real ones are folded the same way
the per-sprint adversarial-review findings were. Scoped + independent means you can run them
in any order, in parallel, and re-run a single domain after fixes without redoing the rest.
