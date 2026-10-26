# L9 Public Bug Bounty Program — Design Specification

> **Status (2026-10-26): Proposal only — never activated.** PRSM does
> not run a bug bounty program and does not pay for vulnerability
> reports. All pool sizes and payout figures in this document were
> proposed figures and are not commitments.

**Type:** Bug bounty program design (not a vendor RFP)
**Issuing organization:** PRSM Foundation (Cayman Islands nonprofit)
**Issued:** 2026-05-05
**Activation:** **POST-L4** (after Code4rena contest + firm pair-review
findings are remediated). Cannot launch before L4 clears or the bounty
becomes the audit (more expensive than running L4 directly).

**Primary contact:** schultzryne@gmail.com / security@prsm.network
**PGP:** see SECURITY.md
**Repository:** https://github.com/Ryno2390/PRSM

---

## 1. Program summary (TL;DR)

Continuous post-mainnet adversarial coverage from the global researcher
community. Two-channel structure:

- **Channel A (Immunefi):** smart-contract + protocol surface — pays
  in FTNS-equivalent USDC from Foundation treasury. ~$50K initial
  pool, scaling with TVL.
- **Channel B (self-hosted at `security@prsm.network`):** off-chain
  surface (Python ML pipeline, infrastructure, web/SDK) — flat-fee
  bounties paid from Foundation treasury, smaller pool ($25K
  initial).

The two-channel split mirrors how the audit layers split (L4 = on-chain,
L5 = off-chain ML, L6f = infrastructure). Researchers self-route their
finding to the right channel.

---

## 2. Why this is a separate document (not a vendor RFP)

L9 is structurally different from L3 / L4 / L5 / L6f / L7 / L8:

- **L3-L8** are paid engagements with specific firms; we send RFPs.
- **L9** is a public, ongoing program; we publish a *spec* that
  researchers self-onboard into.

Immunefi is a *platform* (not an auditor) — the engagement model is
a one-time platform-onboarding fee + ongoing pool funding, not an
audit RFP. So this doc is the design + go-live spec, not an RFP.

---

## 3. Activation gating

**Cannot launch before:**

1. ✅ L1 SAST clean (achieved 2026-05-05; CI workflow live).
2. ✅ L2 multi-team review CLOSED (1 CRIT + 7 HIGH + 7 MEDIUM
   remediated 2026-05-05).
3. ⏳ L3 cryptographic specialist engagement complete + remediated.
4. ⏳ L4 Code4rena contest complete + High/Critical findings
   remediated.
5. ⏳ L5 ML supply-chain audit complete + remediated.
6. ⏳ Phase 1.3 v2 RoyaltyDistributor redeploy complete (HIGH-1
   fix: Ownable2Step + totalClaimable accumulator + recoverStranded;
   Pausable + D-04 pull-payment). No burn — the split is
   creator / network-fee / serving-node.

**Why these gates:** if a researcher submits a finding that's already
known to the audit layers, we pay them for re-finding what we already
fixed. That's economic waste. Run the paid layers first; bounty
catches what they missed.

**Target activation:** 2026-08-15 (after L4 contest period 2026-06-09
to 2026-06-23 + firm pair-review 2026-06-09 to 2026-07-07 + remediation
sprint 2026-07-07 to 2026-08-04 + 2-week safety margin).

---

## 4. Channel A — Immunefi (on-chain protocol surface)

### 4.1 Scope

**In-scope contracts** (matches L4 contest scope, post-remediation):

| Contract | Address (mainnet) | Notes |
|----------|-------------------|-------|
| `EscrowPool.sol` | (deployed at L4-clear) | Per-requester escrow |
| `StakeBond.sol` | (deployed at L4-clear) | Provider stake |
| `BatchSettlementRegistry.sol` | (deployed at L4-clear) | Receipt commit + challenge |
| `RoyaltyDistributor.sol` v2 | `0xfEa9aeB99e02FDb799E2Df3C9195Dc4e5323df7e` | Pull-payment; creator/network-fee/node split (no burn) |
| `FTNSTokenSimple.sol` | `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` | UUPS proxy |
| `EmissionController.sol` | (deployed at L4-clear) | Halving emission |
| `CompensationDistributor.sol` | (deployed at L4-clear) | Pull-based reward distributor |
| `StorageSlashing.sol` | (deployed at L4-clear) | Storage-proof slasher |
| `KeyDistribution.sol` | (deployed at L4-clear) | Threshold key distribution |
| `ProvenanceRegistry.sol` | `0xdF470BFa9eF310B196801D5105468515d0069915` | Content registry |
| `Ed25519Verifier.sol` | (deployed at L4-clear) | L3-audited Ed25519 wrapper |

**Out of scope:**
- Code4rena contest scope (overlap: pre-bounty, in-scope; post-bounty,
  duplicate findings get reduced payouts).
- Off-chain Python (Channel B).
- Front-end / SDK (Channel B).
- Foundation operational security (key ceremony, etc.).
- Already-known issues from L1-L5 audits (we publish a known-issues
  list at activation).

### 4.2 Severity & payouts

Per Immunefi standard (proportional to TVL at risk):

| Severity | Payout USD | Definition |
|----------|------------|------------|
| Critical | $50K – $250K | Direct theft of any FTNS; permanent locking of any user balance; ability to forge slashing against an honest provider |
| High     | $10K – $50K  | Indirect / conditional fund loss; auth bypass; griefing breaking core promise |
| Medium   | $2K – $10K   | Operational disruption recoverable via governance |
| Low      | $500 – $2K   | Edge case; gas optimization with quantifiable cost; minor information leak |

**Pool sizing:** Initial $50K Foundation-funded pool; scales with TVL
per Immunefi's standard ratio (~10% of TVL or $50K, whichever is
higher). Maximum single payout capped at $250K initially; revisit at
$10M TVL.

**Payout currency:** USDC on Base mainnet OR FTNS at oracle-pegged USD
rate (researcher's choice). No vesting; immediate post-validation
release.

### 4.3 Submission flow

Standard Immunefi flow:

1. Researcher submits via Immunefi dashboard.
2. Immunefi triage team validates against scope.
3. Foundation reviews (founder + deputy-founder; 7-day SLA).
4. If valid: severity classified + payout tier confirmed.
5. Foundation prepares fix.
6. Disclosure window: 90 days OR fix-deployed-+-30-days (whichever
   sooner) — coordinated with researcher.
7. Public report on Immunefi after disclosure window expires.

### 4.4 Researcher rules

- **No mainnet exploitation.** All testing on local fork or testnet.
- **No PII access without coordination.** Foundation and authorized
  signers only.
- **No DDoS or stress-testing the live bootstrap.** That's L6f
  scope; not bounty-eligible.
- **No social engineering of Foundation signers.** Out of scope.
- **First-to-find rule.** Duplicate submissions: only the first
  legitimate report receives the full payout; subsequent reports
  receive a fixed $500 acknowledgment.
- **First fixer wins ties.** If two researchers submit within 24h,
  first to provide a working PoC + fix wins.

---

## 5. Channel B — Self-hosted (off-chain surface)

### 5.1 Why self-hosted

Immunefi specializes in on-chain DeFi; their researchers don't
typically have ML systems security background. PRSM's off-chain
ML pipeline (L5 scope) is better served by direct disclosure to
`security@prsm.network`, with a separate (smaller) Foundation pool.

### 5.2 Scope

| Path | Notes |
|------|-------|
| `prsm/compute/inference/` | TensorParallel + Parallax executors, sampling, streaming |
| `prsm/compute/chain_rpc/` | RPC, handoff tokens, layer stage servers |
| `prsm/compute/streaming/` | Autoregressive + sharded + speculation runners |
| `prsm/compute/manifest_dht/` | Manifest DHT |
| Bootstrap node (`bootstrap-us.prsm-network.com:8765`) | Limited — DDoS scenarios are L6f scope |
| Web/SDK / `ai-concierge` | If deployed publicly |

### 5.3 Severity & payouts

| Severity | Payout USD | Definition |
|----------|------------|------------|
| Critical | $25K – $50K  | Activation extraction by malicious relay; receipt forgery |
| High     | $5K – $25K   | KV-cache leak across users; Tier C constant-time bypass |
| Medium   | $1K – $5K    | Operational issue; non-critical info leak |
| Low      | $250 – $1K   | Edge case |

**Pool sizing:** Initial $25K Foundation-funded; scales with usage
volume. Single-payout cap $50K initially.

### 5.4 Submission flow

1. Researcher emails `security@prsm.network` with PGP-encrypted
   report.
2. Foundation acknowledges within 48h.
3. Foundation reviews + classifies severity + replies with payout
   estimate within 14 days.
4. If valid: payout in USDC or FTNS within 30 days post-fix-deploy.
5. Disclosure window: same 90-day / fix-+-30-days standard.
6. Public report on `audits/findings/L9-bounty/` after
   disclosure.

---

## 6. Operational requirements (pre-launch)

Before activation, Foundation must have:

- [ ] L1-L5 audits closed + remediated (gating per §3)
- [ ] Foundation Safe funded with bounty pool ($50K Channel A + $25K
      Channel B = $75K initial)
- [ ] PGP key for `security@prsm.network` published in `SECURITY.md`
      (already done — task #342)
- [ ] Triage rotation defined: founder primary + deputy-founder
      secondary; 7-day SLA on Channel A, 48h ack + 14-day review on
      Channel B
- [ ] Public bounty page on Immunefi (Foundation creates account,
      publishes scope + payout tiers + rules)
- [ ] Public disclosure-policy page on prsm.network (links to
      `SECURITY.md` + this design spec)
- [ ] Forta monitoring (L10a) live so Foundation has independent
      visibility into the protocol while researchers probe (helps
      validate submissions vs. distinguish from real attacks)
- [ ] Council consensus that bounty payouts < $50K are at founder
      discretion; ≥ $50K requires 2-of-3 council sign-off

---

## 7. Post-launch operations

**Quarterly review:**
- TVL-vs-pool ratio (scale up if TVL grows; the bounty must remain
  ≥ 10% of TVL at risk)
- Severity tier review (adjust based on actual finding patterns)
- Researcher feedback / disputed-classifications log
- Public report cadence

**Annual review:**
- Channel A vs Channel B effectiveness (which finds more / better
  issues per dollar)
- Pool reserve sufficiency
- Disclosure-window calibration

**On any new contract deploy** (post-L11 annual re-audit; new
features):
- Add to scope, announce 30 days before going live
- Update Immunefi page

---

## 8. Companion documents

- `SECURITY.md` — vulnerability disclosure policy + PGP key
- `docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md` — incident response
  posture (what happens if bounty researcher reports a CRITICAL)
- `audits/AUDIT_PLAN.md` — master plan; L9 is one layer
- `audits/rfp/L4-code4rena-contest-scope.md` — pre-bounty audit scope
  (to avoid duplicate-finding exposure)

---

## 9. Status

**Status (2026-05-05):** Design spec drafted. Activation gated on
L4 contest + firm pair-review completion (target: 2026-08-15).

**Status (2026-10-26):** Proposal only; never activated. PRSM does not
run a bug bounty program and does not pay for vulnerability reports.

**Next actions:**
1. ☐ Complete L1-L5 audit pipeline.
2. ☐ Council ratification of $75K initial bounty pool funding.
3. ☐ Foundation creates Immunefi account + onboards.
4. ☐ Publish bounty page + announce activation.

---

*See `audits/AUDIT_PLAN.md` §5 L9 for the strategic rationale behind
treating bug bounty as a continuous post-launch defense layer.*
