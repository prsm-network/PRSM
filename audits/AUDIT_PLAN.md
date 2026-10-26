# PRSM Comprehensive Audit Plan

**Document version:** v1.1 (2026-05-05)
**Changelog:** v1.1 adds §3.10 surface×layer matrix, L6f infrastructure pen-test, expanded L3 decision criteria, §6.6 critical-path timeline, §11 risk-transfer (insurance), §12 disclosure & coordinated-vulnerability policy.
**Owner:** Ryne Schultz (Foundation founder) until Foundation Director of Engineering hired
**Status target:** "Audited" attestation that survives external due diligence by sophisticated investors, exchange listings, and post-incident forensics.

---

## 1. Why this document exists

A single freeze tag (`cumulative-audit-prep-20260504-h`) is a snapshot of code,
not an attestation of safety. "Audited" is not a binary state — it is the
**output of a layered process** spanning multiple disciplines, multiple time
horizons, and multiple independent reviewers.

This document is the master plan. Every individual audit artifact in this repo
(the AI multi-team prompts under `audits/team-prompts/`, the cumulative
audit-prep doc, the threat model, the engineering-audit memos) is a row in the
matrix below. None of them, alone, constitutes "the audit."

External readers — investors, auditors, exchanges, regulators — should be able
to read this doc and immediately understand:

1. What the full attack surface is.
2. What layers of defense have been applied to each part of that surface.
3. What gaps remain.
4. What the residual risk is after all layers.

---

## 2. Security objective

> No single key compromise, no single contract bug, no single off-chain service
> outage, and no single legal/regulatory event can result in loss of user funds,
> permanent loss of treasury control, or systemic protocol failure.

This implies defense-in-depth across nine surfaces (§3) and eleven audit layers
(§4). Each layer must be independently verifiable.

---

## 3. The full attack surface

| # | Surface | What's at risk | Adversary |
|---|---------|----------------|-----------|
| **3.1** | **On-chain Solidity** — 14 contracts, ~5,200 LoC | Treasury funds, FTNS supply integrity, payment-split correctness | Anyone with an RPC connection |
| **3.2** | **Off-chain Python services** — `prsm/inference/`, `prsm/chain_rpc/`, `prsm/streaming/` | User compute integrity, content provenance, billing accuracy | Malicious relay node, malicious model host, network MITM |
| **3.3** | **Cryptographic primitives** — Ed25519Lib, Sha512, EIP-712 signing | All signature-gated paths (one bug compromises everything that uses it) | Cryptanalysis, signature forgery, replay |
| **3.4** | **Operational security** — deploy ceremony, key management, multi-sig hygiene | Deployer-window backdoor, signer-key compromise | Hardware-wallet phishing, supply-chain compromise of signing tools |
| **3.5** | **Governance** — Foundation Safe, council voting, emission control, dispute resolution | Treasury redirection, malicious upgrade, denial of legitimate operations | Rogue council member, signer collusion, vote manipulation |
| **3.6** | **Infrastructure** — bootstrap nodes, transport layer, storage layer | Network availability, content censorship, partition attacks | DDoS, ISP-level censorship, sybil attack on DHT |
| **3.7** | **Economic / game-theory** — tokenomics, incentives, MEV surface | Long-run protocol viability, fair compensation, market manipulation | Whales, MEV bots, coordinated griefers |
| **3.8** | **Legal / regulatory** — securities classification, KYC compliance, jurisdiction | Foundation enforcement action, contributor liability, ability to operate | Regulators (SEC, CFTC, FinCEN, FATF analogues), private litigation |
| **3.9** | **Supply chain** — npm/pip dependencies, Hardhat plugins, build pipeline, CI | Malicious code injection, build reproducibility | Dependency-confusion, typosquatting, compromised maintainer accounts |

### 3.10 Surface × layer coverage matrix

The claim of "defense-in-depth" is verifiable iff every surface in §3.1–3.9 is
covered by ≥3 layers. The matrix below proves this. Cell legend: ✅ primary
coverage · ◯ supporting/partial · blank = not covered.

| Surface | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 |
|---------|----|----|----|----|----|----|----|----|----|----|-----|-----|
| 3.1 On-chain Solidity | ✅ | ✅ | ✅ | ◯ | ✅ |   |   |   |   | ✅ | ✅ | ✅ |
| 3.2 Off-chain Python | ✅ | ✅ |   |   |   | ✅ |   |   |   | ◯ | ◯ | ✅ |
| 3.3 Cryptographic primitives | ✅ | ◯ | ◯ | ✅ | ◯ | ◯ |   |   |   | ◯ |   | ✅ |
| 3.4 Operational security |   |   |   |   |   |   | ✅ |   |   |   | ◯ | ◯ |
| 3.5 Governance |   |   |   |   |   |   | ◯ | ◯ | ✅ |   |   | ◯ |
| 3.6 Infrastructure |   | ◯ |   |   |   | ◯ | ✅ |   |   | ◯ | ✅ | ◯ |
| 3.7 Economic / game-theory |   |   |   |   |   |   |   | ✅ |   |   | ◯ | ◯ |
| 3.8 Legal / regulatory |   |   |   |   |   |   |   |   | ✅ |   |   | ◯ |
| 3.9 Supply chain |   | ✅ |   |   |   |   | ◯ |   |   | ◯ |   | ◯ |

**Audit:** every surface has ≥3 layers of coverage. Two cases warrant attention:

- **3.4 OpSec** has only L6 + L10/L11 supporting. Acceptable because L6 itself
  has six sub-layers (L6a–L6f) executing independent reviews.
- **3.8 Legal** has only L8 + L11 supporting. Acceptable because legal
  defense-in-depth is achieved across multiple counsel engagements (Cayman +
  US securities), not multiple in-house layers.

**Gap closed in this revision:** Surface 3.6 (Infrastructure) was previously
under-covered; L6f was added to address bootstrap node, transport, and
storage-layer availability + pen-testing.

---

## 4. The eleven audit layers

Each surface (§3.1–3.9) is covered by one or more of these layers. No surface
is covered by fewer than three layers. The layers are listed in increasing
order of cost and decreasing order of frequency — L0 runs continuously, L11
runs annually.

| Layer | Name | Frequency | Cost | Status |
|-------|------|-----------|------|--------|
| **L0** | Internal review gates (per-phase tags) | Per-task | $0 | ✅ Continuous |
| **L1** | Static + symbolic tooling | Per-merge | $0 | 🟡 Manual; CI integration pending |
| **L2** | AI multi-team adversarial review | Per-major-release | $30–80/run | 🟡 Phase A drafted, awaiting fan-out |
| **L3** | Cryptographic spec-level review | Per-crypto-change | $5–30K | 🔴 Pending |
| **L4** | External smart-contract audit | Per-major-release | $20–80K | 🔴 Pending (gates #31, #40) |
| **L5** | Off-chain ML supply-chain audit | Annual | $30–60K | 🔴 Pending |
| **L6** | Operational security audit | Annual + per-ceremony | $0–20K | ✅ Multi-sig + deploy ceremony done; ongoing |
| **L7** | Economic / game-theory audit | Pre-mainnet + per-emission-change | $20–50K | 🟡 Internal modeling done; external review pending |
| **L8** | Legal / regulatory review | Pre-mainnet + per-jurisdiction-expansion | $15–40K/jurisdiction | 🟡 Counsel briefing memo done; engagement pending |
| **L9** | Public bug bounty | Continuous post-mainnet | $50K–500K pool (proposed) | 🔴 Proposal only; not launched |
| **L10** | Continuous on-chain monitoring + IR | Continuous | $200/mo + on-call | 🟡 Forta wired; alert routing pending |
| **L11** | Annual re-audit on cumulative changes | Annual | $40–80K | 🔴 Future state |

Legend: ✅ Done · 🟡 In progress · 🔴 Pending

---

## 5. Per-layer detail

### L0 — Internal review gates

**What:** Every phase ends with a `*-merge-ready` or `*-audit-prep` git tag.
Each tag is a freeze point with a code-review record.

**Who/how:** Lead engineer + LLM independent reviewer per phase. Tag pinned in
`docs/2026-04-27-cumulative-audit-prep.md`.

**Coverage:** Surfaces 3.1, 3.2.

**Gating:** Required before any L1+ audit. ✅ Currently enforced.

**Artifacts:** ~140 phase tags, all listed in TaskList.

---

### L1 — Static + symbolic tooling

**What:** Industry-standard automated analysis run over the pinned commit.

**Tools:**
- **Slither** — pattern-based static analysis (Solidity)
- **Aderyn** — Rust-native static analyzer (complementary patterns)
- **Mythril** — symbolic execution (deeper, slower)
- **Halmos** — bounded symbolic verification of stated invariants
- **Echidna** — property-based fuzzing
- **Wake / Slang** — alternative pattern matchers (cross-check)
- **pip-audit / npm audit / cargo audit** — dependency vuln scan
- **Bandit / Semgrep** — Python static analysis
- **gitleaks / TruffleHog** — secrets scan in git history

**Coverage:** Surfaces 3.1, 3.2, 3.9.

**Output:** `audits/findings/tooling/<tool>-<date>.{json,md}`

**Gating:** All findings triaged before L2 begins (avoid teams re-finding what
tools already found).

**Cost:** Engineer time only.

**Status (2026-05-05):** **Wired.** Workflow at
`.github/workflows/solidity-static-analysis.yml`. Per-PR fast tier
(Slither + Aderyn) blocks merge on `high` severity; weekly slow tier
(Mythril + Halmos + Echidna) soft-fails with artifacts. Halmos +
Echidna are scaffolded but no-op until property contracts are
authored — see `audits/L1-static-tooling/README.md` §5–§6 for the
priority list of 4 invariants to write next. Slither config:
`contracts/slither.config.json`.

---

### L2 — AI multi-team adversarial review

**What:** Four parallel AI agent teams attack from orthogonal lenses
(economic, access control, signature/crypto, state-composition). Synthesis
pass produces consolidated findings.

**Coverage:** Surface 3.1 fully, 3.3 partially.

**Detail:** See `audits/README.md` (Phase A scope doc) and
`audits/team-prompts/team-{a,b,c,d}.md`.

**Output:** `audits/findings/team-{a,b,c,d}-findings.md` →
`audits/findings/consolidated.md`.

**Gating:** Must complete before L4 contest is purchased (so the contest
brief includes our own pre-found bugs).

**Cost:** $30–80 in API per run. Re-runnable per major release.

**Status:** Prompts drafted, awaiting user review + fan-out.

---

### L3 — Cryptographic spec-level review

**What:** Specialized review of `Ed25519Lib.sol` (887 LoC hand-rolled curve
math) and `Sha512.sol` (328 LoC hash) against RFC 8032 / FIPS 180-4. This is
**not** the same as a smart-contract audit — it requires cryptographer skills,
not Solidity skills.

**Coverage:** Surface 3.3 deeply.

**Why separate from L4:** Code4rena auditors are not cryptographers. A bug in
hand-rolled Ed25519 would not be caught by a contest. The standard practice
for hand-rolled crypto is engagement of a specialist firm (e.g., Trail of
Bits, NCC Group, Cure53, Kudelski Security).

**Status (2026-05-05):** Decision ratified — engage specialist audit on
existing Ed25519Lib + Sha512. See `audits/decisions/L3-ed25519-decision.md`.
RFP packet drafted at `audits/rfp/L3-ed25519-crypto-rfp.md`. Pre-engagement
artifacts complete (RFC 8032 + FIPS 180-4 vectors + upstream port diff +
caller-assumptions memo at `audits/findings/L3-crypto/`). Awaiting vendor
outreach to Trail of Bits + NCC Group.

**Decision criteria (replace vs audit):**

The choice between replacing Ed25519Lib.sol and auditing it must be made
against five concrete criteria:

| Criterion | Replace if… | Audit if… |
|-----------|-------------|-----------|
| **Audit history** | Candidate has ≥1 published independent audit | No suitable candidate exists |
| **Gas cost** | Candidate ≤ 110% of current Ed25519Lib gas usage | Current is already gas-optimal |
| **License** | MIT / Apache 2.0 compatible | License blockers on all candidates |
| **Maintenance** | Active commits in last 12 months | All candidates abandoned |
| **EVM/Base compat** | Compiles with our Solidity version, runs on Base | Compatibility blockers |

**Replacement-candidate longlist** (per L3 task #330 — full evaluation
produces the decision memo):

- **Native EVM precompile** — neither Ethereum mainnet nor Base ship an Ed25519
  precompile as of this writing. ECDSA/secp256k1 has `ecrecover` (precompile
  0x01); Ed25519 does not. RIP-7212 is secp256r1 (different curve), not Ed25519.
  **Conclusion: not currently available.**
- **Witnet `WitnetVRF` / Ed25519 lib** — Witnet has on-chain Ed25519
  verification used by their oracle network. Audit history exists.
- **Daimo's Ed25519** — Daimo is a passkey wallet; their Solidity verifier has
  some audit history.
- **Sismo's crypto libs** — Sismo Connect ZK proofs use related primitives;
  unclear if they ship an Ed25519 verifier directly.
- **Chronicle Labs (Scribe)** — uses BLS12-381, **not** Ed25519. Excluded.
- **OpenZeppelin** — does NOT ship a generic Ed25519 verifier as of OZ v5.x.
- **Solady** — gas-optimized library; does not ship Ed25519 as of latest.
- **The current PRSM Ed25519Lib** — keep but commission specialist audit.

**Vendor candidates for specialist audit (if we keep current lib):**
- **Trail of Bits** — Cryptography practice (audited ZCash, Zoom, etc.)
- **NCC Group** — Cryptography Services group
- **Cure53** — Crypto + appsec; have audited Signal, Tutanota, etc.
- **Kudelski Security** — Specialized crypto firm
- **Least Authority** — ZCash + Tezos crypto auditing
- **Trail of Bits or NCC** are the realistic top-tier picks; cost $15–30K for
  this scope (Ed25519Lib + Sha512, ~1,300 LoC of math).

**Contest-based review (Code4rena):** Insufficient for this surface.
Smart-contract auditors are not cryptographers; standard contest format will
miss subtle curve-arithmetic bugs.

**Output:** Either (a) replacement PR + delta-audit of integration changes
(~$2K), or (b) full specialist crypto audit report (~$15–30K).

**Gating:** Required before any production use of `Ed25519Verifier` paths
(currently used by `BatchSettlementRegistry`).

**Status:** **Decision drafted 2026-05-05** in
[`decisions/L3-ed25519-decision.md`](./decisions/L3-ed25519-decision.md).
Outcome: engage specialist crypto audit on existing Ed25519Lib (no viable
replacement exists; verifier is hot-swappable so reversibility risk is low).
Pending Foundation council ratification.

---

### L4 — External smart-contract audit

**What:** Paid human review of the audit-bundle scope.

**Options matrix:**

| Vendor | Format | Cost | Wall-clock | Strengths |
|--------|--------|------|------------|-----------|
| **Code4rena** | Public contest | $20–40K | 2 weeks | Broad crowd; finds creative bugs |
| **Sherlock** | Fixed-fee contest | $30–50K | 3 weeks | Judge protocol; payout for missed findings |
| **Cantina** | Marketplace | $25–60K | 2–4 weeks | Specialist matching |
| **Trail of Bits** | Solo firm | $80–150K | 4–6 weeks | Reputation; report carries weight with regulators |
| **OpenZeppelin** | Solo firm | $80–120K | 4–6 weeks | Known to OZ-pattern users |
| **ConsenSys Diligence** | Solo firm | $80–150K | 4–6 weeks | Reputation; Ethereum-native |
| **Spearbit / Cantina elite** | Solo or pair | $40–100K | 3–4 weeks | Reputable middle-tier |

**Coverage:** Surface 3.1.

**Existing artifacts:** `docs/2026-04-23-auditor-shortlist-and-rfp.md` already
has shortlist + RFP template.

**Gating:** Required to clear tasks #31 (Phase 7 Task 9) and #40 (Phase 7.1
Task 9). Without this layer, the audit-bundle stack stays in pre-deploy state.

**Recommendation:** Two-stage —
1. **Code4rena contest** ($30K) using L2 + L3 findings as starting brief.
2. **Trail of Bits or Spearbit pair-review** ($60K) targeting whatever the
   contest didn't shake out, plus the integration surface across contracts.

Total: ~$90K. Comparable to a single solo-firm engagement but with cross-
checking by independent groups.

**Status (2026-05-05):** RFP packets drafted. Awaiting founder
execution of vendor outreach.

- **Code4rena contest packet:** `audits/rfp/L4-code4rena-contest-scope.md`
  (~$40K target pool; 14-day public contest; in-scope = 9 contracts,
  ~2,300 SLOC; out-of-scope = Ed25519Lib + Sha512 = L3).
- **Firm RFP refresh:** `audits/rfp/L4-firm-rfp-addendum-20260505.md`
  layered on top of the original 2026-04-23 shortlist doc. Pulls in
  L1-wired + L2-closed + MEDIUM-cleanup state and narrows the firm scope
  to cross-contract composition + Phase 7-storage + Phase 8 + UUPS
  upgrade surface (~$60-80K).

Next action: **founder sends RFPs to top-3 firms + opens Code4rena
intake conversation.**

---

### L5 — Off-chain ML supply-chain audit

**What:** Adversarial review of `prsm/inference/`, `prsm/chain_rpc/`,
`prsm/streaming/`. Threat focus: malicious relay/host nodes, activation
extraction, receipt forgery, KV-cache leak, tier-C constant-time guarantees.

**Coverage:** Surface 3.2.

**Why specialist:** Standard smart-contract audit firms don't have ML
supply-chain expertise. The right vendor profile is "ML systems security"
firms or academic groups (see e.g., Stanford CRFM, MIT CSAIL labs that audit
ML systems).

**Vendor candidates:**
- **Trail of Bits** (has ML practice — "ML Assurance")
- **NCC Group** (has cryptography + ML practice)
- **Galois** (formal-methods firm, has ML safety work)
- Academic engagement (Berkeley RDI, CMU CyLab) — slower, often cheaper

**Coverage scope:**
- TEE attestation aggregation policy
- Receipt-format extension and Ed25519 sign/verify on Python side
- Tier C constant-time padding (FixedRateStreamingRunner, BatchedTrailingShardedExecutor)
- Per-iteration attestation chain integrity
- KV-cache isolation across concurrent requests
- Manifest DHT tampering surface

**Threat-model docs already produced (input to auditor):**
- `docs/2026-04-22-r3-threat-model.md`
- `docs/2026-04-30-phase3.x.11-threat-model-addendum.md`

**Gating:** Must complete before any user runs Tier C inference against
unfamiliar relay nodes at scale.

**Status (2026-05-05):** RFP drafted at
`audits/rfp/L5-ml-supply-chain-audit-rfp.md`. Tier 1 scope ~28K LoC
(security-load-bearing); Tier 2 ~7K LoC. Budget $40K-$80K. Recipients:
Trail of Bits ML Assurance + NCC Group ML (commercial track) + Berkeley
RDI + CMU CyLab (academic track). Awaiting founder vendor outreach.

**Status:** Pending.

---

### L6 — Operational security audit

**What:** Audit of the operational layer that holds the keys, runs the deploy
ceremonies, and rotates signers.

**Sub-components:**

**L6a — Multi-sig setup audit (✅ done 2026-05-03/04):**
- 3 hardware wallets (Ledger / Trezor / OneKey) — verified end-to-end
- Foundation Safe deployed as 2-of-3 — verified
- Phase 4 round-trip signing on all 3 devices — verified
- Stored at `~/.prsm-foundation-private/Multi-Sig_Addresses.txt`, chmod 600

**L6b — Deploy ceremony audit (✅ executed 2026-05-04):**
- Phase 1.3 Task 8 deployed
- Lessons codified in `docs/2026-05-04-task8-deploy-ceremony-lessons.md`
- Hardening: `pkAccounts()` validation, ceremony lessons, sweep-deployer
  script, audit-bundle deployment verifier
- **Outstanding:** Sweep-deployer script tested on Phase 1.3, but
  `verify-audit-bundle-deployment.js` only used for #31/#40 deploy

**L6c — Key rotation runbook (🔴 pending):**
- What if a hardware wallet is lost? Documented procedure to rotate signers.
- What if the deployer key is suspected compromised before ownership transfer?
  Mid-ceremony abort + redeploy.
- What if a Safe owner's seed is suspected leaked? Multi-sig rotation transaction.

**L6d — Insider-threat / signer collusion model (🔴 pending):**
- 2-of-3 Safe means 2 signers can collude. Documented governance constraint
  (e.g., 3 signers in 3 different jurisdictions, 1 must be Foundation Director).
- Bus-factor: D7 deputy-founder protocol exists (task #134 done).
- Disaster recovery: founder incapacitation procedure.

**L6e — Continuous ops hygiene:**
- Quarterly review of signer addresses, hardware status, seed-storage location,
  emergency contacts, treasury floor compliance, contract operational state,
  and Forta routing health.
- Template: `docs/security/L6E_OPS_HYGIENE_REVIEW.md` (shipped 2026-05-05).
- Per-quarter signed instances archive in `docs/security/L6E-reviews/YYYY-Qx.md`.
- Annual key-rotation drill folds into the Q4 review each year.
- Infrastructure-side operational surface remains in
  `docs/PRODUCTION_OPERATIONS_MANUAL.md` (kubectl, deploys, etc.).

**L6f — Network infrastructure pen-test (🔴 pending):**
- Bootstrap-node availability: DDoS resilience of `wss://bootstrap-us.prsm-network.com:8765`
  and any subsequent bootstrap nodes the Foundation operates.
- Transport-layer adapter security: Direct/SOCKS adapter robustness under
  hostile network conditions (R9 Phase 6.2 surface).
- DHT poisoning resistance: Manifest DHT and Profile DHT (Phase 3.x.5) under
  sybil + eclipse-attack scenarios.
- Storage-layer availability: erasure-coded shard recovery under partial
  network failure (Phase 7-storage).
- Rate-limiting + connection-liveness exhaustion testing (Phase 6 Task 5).
- TLS configuration audit (cert pinning, protocol versions).
- Vendor candidates: NCC Group, Bishop Fox, Doyensec, IOActive — standard
  network/protocol pen-test firms. Engagement ~$20–50K, 2–3 weeks.

**Coverage:** Surfaces 3.4 (operational keys/ceremony) + 3.6 (infrastructure
availability + protocol robustness).

**Output:** Operational security report (L6a–e) + infrastructure pen-test
report (L6f). Quarterly cadence on L6e; per-major-release on L6f.

**Status (2026-05-05):** L6a + L6b done; L6c (key rotation runbook) +
L6d (insider-threat policy) + L6e (ops-hygiene template + review
archive directory) shipped 2026-05-05. First L6e review window
2026-07-01 → 2026-07-14. **L6f RFP drafted at
`audits/rfp/L6f-infrastructure-pentest-rfp.md`** — recipients NCC
Group + Bishop Fox + Doyensec; budget $20K-$50K, 2-3 weeks. Awaiting
founder vendor outreach.

---

### L7 — Economic / game-theory audit

**What:** External review of tokenomics, emission schedule, MEV exposure, and
incentive alignment.

**Coverage:** Surfaces 3.5 (governance economics) + 3.7 (economic / game-theory).

**Existing internal artifacts:**
- `docs/2026-04-21-prsm-tok-1-ftns-tokenomics.md` — Tokenomics spec
- `docs/2026-04-21-prsm-gov-1-foundation-governance-charter.md` — Governance
- Phase B Epoch 1 rate-calibration simulator (task #131 done)
- POL parameterization, halving schedule, Foundation revenue model — all done

**Why external review:** Internal modeling is necessary but biased toward our
own assumptions. An adversarial economic review tests the protocol against
agent-based simulation, MEV strategies, and worst-case market scenarios.

**Vendor candidates:**
- **Gauntlet** — DeFi economic risk modeling (Aave, Compound, etc.)
- **Chaos Labs** — DeFi simulation + risk management
- **BlockScience** — token-engineering academic firm
- Academic engagement (e.g., MIT DCI, Stanford engineering)

**Output:** Economic risk report. Updates required when emission parameters or
revenue model materially change.

**Gating:** Should complete before mainnet liquidity event (DEX listing, CEX
listing, Prismatica equity raise).

**Status (2026-05-05):** RFP drafted at
`audits/rfp/L7-economic-game-theory-audit-rfp.md`. Budget $50K-$100K.
Recipients: Gauntlet + Chaos Labs (commercial track) + BlockScience
(CADCAD modeling) + MIT DCI (academic track). 7 specific quantitative
questions enumerated for vendor scope. Awaiting founder vendor outreach.

---

### L8 — Legal / regulatory review

**What:** Counsel review of securities classification (Howey), KYC obligations,
jurisdiction strategy, and Foundation entity structure.

**Coverage:** Surface 3.8.

**Existing internal artifacts:**
- `docs/2026-04-23-prsm-policy-jurisdiction-1.md` — Jurisdiction policy
- Foundation jurisdiction scoping doc (task #104 done)
- Legal counsel shortlist + RFP template (task #105 done)
- Pre-mainnet legal scope packet (task #135 done)

**Engagement model:** Two counsel — (1) Cayman counsel for Foundation
governance; (2) US securities counsel for Prismatica + FTNS classification.

**Gating items:**
- Reg D 506(c) Prismatica equity offering: counsel-cleared private placement memorandum
- FTNS classification opinion (utility vs security): required before any
  fund-to-public path
- KYC vendor decision: §8.1 memo done (task #194)
- FATF/OFAC sanctions screening: implementation pending

**Status (2026-05-05):** RFP packet drafted at
`audits/rfp/L8-legal-counsel-rfp.md`. Two tracks:
- **Track A (Cayman):** $30K-$80K. Recipients: Mourant + Walkers + Ogier.
- **Track B (US securities):** $50K-$150K. Recipients: Cooley +
  Lowenstein + DLx Law.
Awaiting founder execution of vendor outreach.

---

### L9 — Public bug bounty

> **Status (2026-10-26): Proposal only — not launched.** PRSM does not
> run a bug bounty program and does not pay for vulnerability reports.
> The pool sizing and payout figures in this section and in the L9
> design spec were proposed figures and are not commitments.

**What:** Continuous post-mainnet adversarial coverage from the global
researcher community.

**Coverage:** All on-chain + off-chain surfaces (3.1, 3.2, 3.3, 3.6).

**Vendor options:**
- **Immunefi** — DeFi-native, largest researcher community, payout in protocol
  treasury or USDC
- **HackerOne** — broader market, less crypto-native
- **Self-hosted** — Foundation runs own program; flexible but lower discovery rate

**Recommended:** Immunefi for protocol/contract surface, supplemented by
self-hosted disclosure channel (security@prsm.dev) for off-chain reports.

**Pool sizing (industry norms):**
- Critical: $50K–$1M (proportional to TVL at risk)
- High: $10K–$50K
- Medium: $2K–$10K
- Low: $500–$2K

**Pre-mainnet:** Pool can start small ($50K total) and scale with TVL.

**Gating:** Cannot launch until L4 first-pass complete (otherwise the bounty
becomes the audit, which is more expensive than just doing the audit).

**Status (2026-05-05):** Design spec drafted at
`audits/rfp/L9-bug-bounty-program-design.md`. Two-channel structure
(Immunefi for on-chain ~$50K + self-hosted security@ for off-chain
~$25K). Activation gated on L4 contest + firm pair-review completion;
target activation 2026-08-15.

---

### L10 — Continuous on-chain monitoring + IR

**What:** Real-time detection of anomalies on deployed contracts +
pre-rehearsed incident response.

**Components:**

**L10a — Forta detection bots (🟡 partially wired):**
- Codebase: `ops/monitoring/forta-bots/`
- Contracts populated: FTNSToken, ProvenanceRegistry, RoyaltyDistributor (Base
  Mainnet, post 2026-05-04)
- Detectors covered: ERC-20 anomalous transfer, role grants/revokes, pause
  events, royalty distribution failure, ownership transfer
- **Outstanding:** Phase 7+8 contracts (BatchSettlementRegistry, EscrowPool,
  EmissionController, CompensationDistributor) addresses null pending deploy
- **Outstanding:** Alert routing to PagerDuty + Foundation council Slack
- Vendor decision memo: `docs/2026-04-26-vendor-decision-forta-vs-tenderly.md`

**L10b — Exploit response playbook (✅ exists):**
- `docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md`
- Per-contract pause-transaction templates pre-staged (task #138 done)
- Public + private annex (task #133 done)

**L10c — On-call rotation (🔴 pending):**
- Pre-mainnet: founder = on-call. Acceptable for Phase 1.3 only.
- Post-mainnet: minimum 2 on-call engineers.
- 24/7 coverage required once TVL > $1M.

**L10d — Public security policy:**
- `SECURITY.md` in repo root with disclosure email + PGP key.
- **Outstanding:** confirm policy file present + linked from README.

**Coverage:** All on-chain surfaces, partial off-chain.

**Gating:** L10a + L10b are required before mainnet TVL > $0. Currently
acceptable for Phase 1.3 only because TVL is foundation-treasury 2% accumulating
slowly.

**Status:** Forta wired but not fully populated; alert routing pending.

---

### L11 — Annual re-audit on cumulative changes

**What:** Each calendar year, freeze a new audit-prep tag and re-run L2–L4
against the diff since the last audit.

**Coverage:** All surfaces.

**Gating model:** No emission-controller change, no governance-charter
amendment, and no cross-contract wire change can be deployed without a fresh
audit on the changed surface.

**Status:** Future state. First annual cycle: 2027.

---

## 6. Pre-mainnet gating sequence

The chart below shows what must complete before each gate. Mainnet is split
into stages because not all stages need all layers.

### Gate A — Treasury layer live (Phase 1.3 Task 8)

✅ **CLEARED 2026-05-04.**

Required layers: L0, L6a, L6b. Why so few? Phase 1.3 contracts are
non-Ownable post-handoff with immutable cross-wires. Worst-case bug = the 2%
fee stream is misrouted; recoverable by deploying replacement and redirecting
RoyaltyDistributor (which IS owned by Safe).

**Residual risk accepted at this gate:** Smart-contract bug in
ProvenanceRegistry / RoyaltyDistributor not caught by L0. Mitigation:
contracts are non-upgradeable, attack surface is small (~230 LoC combined),
deployed value at risk is bounded (only 2% fee stream).

### Gate B — Audit-bundle stack live (#31, #40)

🔴 **PENDING.**

Required layers: L0 ✅, L1 🟡, L2 🟡, L3 🔴, L4 🔴, L6 ✅, L10a 🟡.

This is the largest gating point. Audit-bundle includes:
- EscrowPool (held funds — direct value at risk)
- BatchSettlementRegistry (788 LoC, central nexus)
- StakeBond (slashable stake — direct value at risk)
- SignatureVerifier + Ed25519Lib (crypto primitive — one bug = full break)

**Cannot clear without L3 and L4.** L2 (AI multi-team) is ready to start.

### Gate C — Phase 8 emission live

🔴 **FUTURE.**

Required layers: L0, L1, L2, L4, L7 (economic audit specifically required
because emission parameters are economic mechanisms).

### Gate D — Public bounty + DEX liquidity

🔴 **FUTURE.**

Required layers: All of L0–L8, plus L9 launch + L10c on-call rotation.

### Gate E — Prismatica fundraising close

🔴 **FUTURE.**

Required layers: L8 (legal — Reg D 506(c) compliance) is the binding constraint.

### 6.6 Critical-path timeline (from now to Gate B clearance)

The chart below is a working calendar from "today" (the date this plan was
authored) to the earliest realistic Gate B clearance (audit-bundle stack live).
Wall-clock weeks. Activities on the same row run in parallel.

| Week | Critical path (long-pole) | Parallel work | Output / gate |
|------|---------------------------|---------------|---------------|
| **W1** | L3 Ed25519 evaluation → decision memo | L2 fan-out (4 teams) + synthesis | L3 decision; L2 consolidated findings |
| **W2** | L1 CI integration (Slither/Aderyn/Mythril/Halmos/Echidna in CI) | L3 execution begins (replace OR engage specialist firm) | L1 reports per-PR; L3 in motion |
| **W3** | L4 RFP send to top-3 + Code4rena scoping call | L6c/L6d runbooks drafted | RFP responses incoming |
| **W4** | L4 contest setup (scope freeze, prize pool fund, judge protocol selection) | L7 economic-audit scoping (Gauntlet/Chaos Labs); L8 counsel engagement starts | L4 contest start date booked |
| **W5–W6** | L4 Code4rena contest active period | L5 ML supply-chain audit kickoff; L6f infrastructure pen-test scoping; L10a alert routing | Findings stream in mid-contest |
| **W7** | L4 contest closes; finding triage begins | L5 + L7 mid-engagement | Contest report draft |
| **W8** | L4 remediation sprint (fix High/Critical); L3 specialist report (if path b) | L7 final report; L8 counsel opinions land | Code-frozen post-fix |
| **W9** | L4 second-pass review (Spearbit pair-review or solo-firm follow-up) | L5 final report; L6f pen-test active | Confirmation pass |
| **W10** | L4 final sign-off; L9 Immunefi listing draft (proposed; not launched) | L10a alert routing live; L11 first-cycle baseline established | Pre-Gate-B checklist |
| **W11** | Final preflight: re-run L1 + L2 + verify-audit-bundle-deployment | L9 bounty soft-launch with low pool (proposed; not launched) | All-green gate state |
| **W12** | **Gate B ceremony — audit-bundle stack deploys to Base mainnet** | — | #31, #40 closed |

**Critical-path bottleneck:** L4 contest (~4 weeks: setup + run + remediate +
second-pass). Everything else can run alongside or before. Reducing this
bottleneck is the highest-leverage timeline lever.

**Branching: what if a layer fails to clear?**

- **L4 critical finding** → reset to W7 remediation, push Gate B by 2–4 weeks.
  Re-run AI multi-team (L2) on remediated code. No shortcut.
- **L3 specialist audit finds break** → either replace lib (week+ delay) or
  rework integration. Gate B blocks until resolved.
- **L7 economic finding** → if affects emission parameters, Phase 8 deploy
  blocks. Phase 7/7.1 (Gate B scope) does NOT block on L7 since audit-bundle
  stack is economic-mechanism-neutral.
- **L8 legal finding** → may force jurisdiction change or KYC/sanctions
  rework. Worst-case: Foundation entity restructure (months, not weeks).
- **L5 ML-audit finding** → blocks Tier C inference rollout but does not block
  Gate B (audit-bundle is on-chain-only).

**Cost timing:** L4 ($30–90K) is paid at W4. L3 ($15–30K if path b) is paid at
W2. L5 ($30–60K) at W5. L7 ($20–50K) at W4. L8 ($15–40K) at W4. **Peak burn:
weeks 4–6, ~$110–250K cash out across overlapping engagements.**

**Pre-funding gate:** Before W4 begins, Foundation must have 3 months of
audit-engagement runway (~$300K) on hand. If not, sequence engagements
serially instead of parallel — adds 4–8 weeks but reduces peak burn.

---

## 7. Post-mainnet ongoing

| Cadence | Layer | What happens |
|---------|-------|--------------|
| Continuous | L10 | Forta alerts → on-call paged → IR playbook |
| Per-PR | L0, L1 | Internal review + automated tooling on every merge |
| Per-major-release | L2 | AI multi-team re-run on changed surface |
| Per-crypto-change | L3 | Specialist review |
| Per-emission-change | L7 | Economic re-validation |
| Per-jurisdiction-expansion | L8 | Counsel review |
| Continuous | L9 | Bug bounty open-ended (proposed only; not launched) |
| Quarterly | L6e | Operational hygiene review |
| Annually | L11 | Full cumulative re-audit |

---

## 8. Cost envelope

| Layer | One-time | Annual |
|-------|----------|--------|
| L0 | $0 | $0 |
| L1 | $0 (engineer time) | $0 |
| L2 | $30–80 per run | $200–500 (4× yr) |
| L3 | $5–30K (or $0 if replace) | $0 if no crypto change |
| L4 | $30–90K | $40–80K (annual re-audit) |
| L5 | $30–60K | $30–60K |
| L6 | $0–20K (operational, mostly internal) | $5K |
| L7 | $20–50K | $10K |
| L8 | $15–40K per jurisdiction | $5K/jurisdiction |
| L9 | $50K seed pool (proposed) | Pool grows with TVL |
| L10 | $200/mo Forta + on-call setup | $5K |
| L11 | — | $40–80K |

**Pre-mainnet total (one-time):** ~$200K–400K depending on choices in L3 + L4.
**Annual ongoing:** ~$100K–200K + bounty pool (proposed; L9 not launched).

These are unsubsidized estimates. **Cost-reduction levers:**
- Replace Ed25519Lib with battle-tested alternative → L3 → $0
- Code4rena contest only (skip solo firm) → L4 → $30K
- Academic engagement for L5/L7 → 50–70% discount, slower wall-clock
- Foundation grant programs (e.g., Ethereum Foundation, Optimism Retro
  Funding) may subsidize parts of L4–L7 for legitimate public-good protocols

---

## 9. Status & next actions

### What's done
- L0 (continuous, ~140 phase tags)
- L6a (multi-sig setup verified end-to-end)
- L6b (deploy ceremony executed for Phase 1.3 Task 8)
- L8 prep (counsel briefing memo, jurisdiction policy, RFP templates)
- L10b (exploit-response playbook + pre-staged pause transactions)
- Threat model, Tokenomics, Governance Charter, Risk Register inputs

### What's drafted, awaiting trigger
- L2 Phase A (4 team prompts ready in `audits/team-prompts/`)
- L4 RFP (auditor shortlist done, RFPs not sent)
- L5 vendor candidates (no engagement)
- L7 vendor candidates (no engagement)
- L8 counsel candidates (no engagement)
- L9 program design (no Immunefi listing)

### Next-action priority list
1. **Decision: Replace Ed25519Lib?** Evaluate viable battle-tested alternatives.
   If yes, save L3 cost. If no, engage L3 vendor (highest priority — bug here
   breaks everything else).
2. **Fan out L2 Phase A.** Already gated only on user review of prompts.
3. **CI integration of L1 tooling.** Convert manual runs into per-PR automation.
4. **L6c key-rotation runbook.** Document signer-loss + key-compromise procedures.
5. **L4 RFP send.** Code4rena scoping + 2 solo-firm shortlist conversations.
6. **L10a alert routing.** PagerDuty + Slack integration on existing Forta bots.
7. **L8 counsel engagement.** Cayman + US securities counsel.
8. **L7 economic review.** Gauntlet/Chaos Labs/BlockScience scoping.
9. **L5 ML supply-chain audit.** Trail of Bits or academic engagement.
10. **L9 bounty program design.** Immunefi listing draft.

---

## 10. Residual risk after all layers

Even with all 11 layers fully executed, the following residual risks persist
and must be acknowledged:

| Residual risk | Why it persists | Mitigation strategy |
|---------------|-----------------|---------------------|
| Zero-day in Solidity compiler | Solc bug affects all contracts globally | Pin solc version; track CVEs |
| Zero-day in OpenZeppelin libs | Same; OZ is widely deployed | Track OZ security advisories |
| Hardware wallet firmware compromise | All 3 brands have had CVEs | Diversification across vendors |
| 2-of-3 signer collusion | Inherent to threshold cryptography | Geographic + organizational diversity of signers |
| Off-chain Python remote-code-execution | Always possible in any complex codebase | Sandbox node binary; minimize attack surface |
| Regulatory action on a specific jurisdiction | Outside our control | Multi-jurisdiction strategy; jurisdiction policy doc |
| Fundamental cryptographic break | Quantum computing on Ed25519 | R6 post-quantum trigger-watch memo (#53 done) |
| Black-swan economic scenario | Cannot fully model novel attacks | Pause mechanism + halving-pause governance lever |

This list is not exhaustive but represents the categories that remain after
defense-in-depth. The protocol design assumes these can occur and constrains
blast-radius accordingly (immutability where possible, multi-sig where
mutable, time-locked where critical, pausable where reversible-recovery
matters).

---

## 11. Risk-transfer layer (insurance)

Audit posture reduces probability of loss; insurance transfers a portion of
remaining loss exposure. The two are complementary, not substitutes.

### Available coverage products (2026 market reality)

The DeFi insurance market shrank materially over 2022–2025. Surviving products:

| Product | Type | Coverage profile | Notes |
|---------|------|------------------|-------|
| **Nexus Mutual** | Mutual member-pooled | Smart-contract failure, oracle, custodian | Most established; coverage capacity per protocol limited; KYC required |
| **Sherlock** | Audit + cover combined | Bound to Sherlock-audited protocols only | Tightly coupled to L4 vendor choice |
| **Unslashed Finance** | Discretionary mutual | Smart-contract + slashing | Smaller capacity; permissionless coverage |
| **Risk Harbor** | Parametric | Stablecoin depeg, specific event triggers | Narrower applicability |
| **InsurAce** | Mutual | Smart-contract + custodian | Smaller capacity; available |

### Foundation-level self-insurance

A portion of the 2% network fee accumulating to the Foundation Safe acts as a
**de facto self-insurance reserve** absent an external coverage purchase. The
question is: what fraction of the treasury is earmarked for incident response
vs operations?

**Recommended policy** (not yet ratified by Foundation council):
- Treasury reserve floor: **$2M USD-equivalent** in Foundation Safe at all
  times before any non-emergency disbursement.
- Treasury reserve target: **$10M USD-equivalent** before any external
  coverage purchase is entertained.
- Below floor: cover purchase on Nexus Mutual sized to bridge the gap.

### Engagement decision

Pre-mainnet (now → Gate B): **No external cover purchase recommended.** TVL is
small enough that self-insurance from Foundation reserve is sufficient.

Post-Gate-B + post-Phase-8 emission: **Re-evaluate quarterly.** Trigger for
purchasing external cover: any single user position exceeding 1% of treasury
reserve.

Post-DEX-listing: **Mandatory.** External cover at minimum 2× max-single-user
position. This is required to maintain credible TVL backstop for institutional
participants.

**Status (2026-05-05):** Policy drafted at
`docs/governance/2026-05-05-treasury-reserve-and-risk-transfer-policy.md`
(PRSM-POL-1). 14 sections covering reserve floors ($2M / $10M target),
risk-transfer triggers (1% single-user position OR $5M TVL OR
50%-treasury-depletion-on-payout), insurance-product evaluation,
disbursement authorization (founder→council escalation tiers), annual
allocation, reporting cadence, breach protocol, amendment process. **5
open ratification questions enumerated for council deliberation.**
Awaiting Foundation council 2-of-3 ratification at next governance
agenda.

---

## 12. Disclosure & coordinated-vulnerability policy

When a finding exists, who is told what, when? This section sets the policy
that security researchers, audit firms, integrators, and the public can
expect.

### Pre-mainnet (now → Gate B)

- **Audit findings** (L2/L3/L4/L5): held confidential by the engaged auditor
  and Foundation engineering. Public disclosure deferred until Gate B + 30
  days, at which point the consolidated audit report is published with all
  resolved findings annotated.
- **Self-discovered findings** (L0/L1): triaged internally; no external
  disclosure required pre-mainnet.
- **Researcher-discovered findings**: PRSM does not run a bug bounty
  program and does not pay for vulnerability reports; researchers who
  discover issues should email `security@prsm-network.com` (to be set
  up — action item) with PGP-encrypted reports. Acknowledgment within
  48 hours.

### Post-mainnet (Gate B onwards)

**Coordinated disclosure window: 90 days standard.**

- Day 0: Researcher reports finding to `security@prsm-network.com`.
- Day 0–2: Foundation engineering acknowledges receipt + assigns severity.
- Day 2–14: Triage + reproduction. If confirmed, fix development begins.
- Day 14–60: Fix development + audit of the fix.
- Day 60–75: Coordinated deployment window. Foundation deploys fix; known
  integrators are notified privately ≥72 hours before public disclosure.
- Day 75–90: Public disclosure on GitHub Security Advisory + blog post +
  reporter credit (with consent).
- Day 90: Hard cap. Even if fix incomplete, public disclosure happens at day
  90 to prevent indefinite suppression.

### Critical-severity exception

For any finding with **active exploitation** OR **>50% TVL at imminent risk**:

- Hour 0: Pause-transaction templates from `EXPLOIT_RESPONSE_PLAYBOOK.md`
  executed by Foundation Safe within ≤2 hours.
- Hour 0–24: Private notification to known integrators + Forta alert subscribers.
- Hour 24–48: Public disclosure on Foundation Twitter/blog + GitHub advisory.
- Day 7+: Post-mortem published.

The 90-day window does not apply to actively-exploited bugs.

### Bounty payout criteria

> **Status (2026-10-26): Proposal only — no payouts are offered.** PRSM
> does not run a bug bounty program and does not pay for vulnerability
> reports. The scale and payout terms below were proposed for the L9
> layer, were never activated, and are not commitments.

Per Immunefi standard scale (see L9 §5):

- Critical: $50K–$1M (proportional to TVL at risk and quality of report)
- High: $10K–$50K
- Medium: $2K–$10K
- Low: $500–$2K
- Informational: $0–$500 (or swag)

Payout in USDC from Foundation Safe within 14 days of fix deployment.

### Duplicate handling

First valid report receives credit. Subsequent identical reports receive
acknowledgment. Determined by submission timestamp on `security@prsm-network.com`
inbox.

### Public artifacts

- `SECURITY.md` in repo root (action item — verify present + linked from
  README) — published security email + PGP key + reporting guidelines.
- GitHub Security Advisory (GHSA) — used for all post-mainnet disclosures.
- Foundation blog — post-mortem narrative for Critical findings.
- Public hall-of-fame — researchers credited (with consent) at
  `audits/hall-of-fame.md` (not yet published).

**Status:** Policy drafted here. Operationalization (security email, PGP key,
SECURITY.md presence) is part of L10d (action item).

---

## Appendix A — Cross-reference of existing artifacts

| Artifact | Layer it serves |
|----------|-----------------|
| `docs/2026-04-22-r3-threat-model.md` | L2, L4, L5 input |
| `docs/2026-04-30-phase3.x.11-threat-model-addendum.md` | L2, L5 input |
| `docs/2026-04-27-cumulative-audit-prep.md` | L2, L4 input (auditor brief) |
| `docs/2026-04-21-audit-bundle-coordinator.md` | L4 scope spec |
| `docs/2026-04-23-auditor-shortlist-and-rfp.md` | L4 vendor selection |
| `docs/2026-04-21-prsm-tok-1-ftns-tokenomics.md` | L7 input |
| `docs/2026-04-21-prsm-gov-1-foundation-governance-charter.md` | L6, L7, L8 input |
| `docs/2026-04-23-prsm-policy-jurisdiction-1.md` | L8 input |
| `docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md` | L10b |
| `docs/PRODUCTION_OPERATIONS_MANUAL.md` | L6e |
| `docs/2026-05-04-task8-deploy-ceremony-lessons.md` | L6b |
| `docs/2026-04-30-deploy-ceremony-dry-run-audit.md` | L6 |
| `docs/2026-04-30-multisig-action-plan-engineering-audit.md` | L6 |
| `docs/2026-04-26-vendor-decision-forta-vs-tenderly.md` | L10a |
| `ops/monitoring/forta-bots/` | L10a implementation |
| `audits/README.md` | L2 Phase A scope |
| `audits/team-prompts/team-{a,b,c,d}.md` | L2 Phase A execution |
| `contracts/scripts/verify-audit-bundle-deployment.js` | L6b post-handoff verifier |
| `scripts/foundation-safe-health-check.py` | L6e periodic check |
| `scripts/sweep-deployer.py` | L6b ceremony tooling |

This index is the answer to "where is the audit?" — it points at the corpus.
The corpus is the audit. This document is the map.
