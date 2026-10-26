# PRSM Audit Pipeline RFP Index

**Status (2026-05-06, post PRSM-POL-2):** All vendor-track RFPs drafted as of 2026-05-05 and **DEFERRED — funding-gated** per PRSM-POL-2 (Resource-Constrained Audit Strategy). RFP packets remain in tree as audit-trail evidence and as ready-to-fire artifacts when §5 revisit triggers fire. **L8 Track B (US securities counsel) is the sole non-deferred layer** — it remains a hard gate triggered by the first Prismatica investor solicitation regardless of broader budget posture.

> **Why deferred:** Combined budget is $280K–$645K. PRSM is currently a sole-founder, open-source, pre-funding project. The implicit "all audits clear before mainnet deploy" stance had become a project-freeze policy. PRSM-POL-2 (`docs/governance/2026-05-06-resource-constrained-audit-strategy.md`) ratified by PRSM-CR-2026-05-06-5 adopts agent-teams self-audit + OZ Pausable + TVL caps + 14-day public review window as the substitute mitigation framework for L3 and L4. Other layers are deferred with revisit triggers (POL-2 §5).

**Outreach packet:** [`outreach-emails.md`](outreach-emails.md) — one email per recipient (~22 firms across 8 RFPs), ready to copy/paste **once a §5 revisit trigger fires**. Do NOT execute outreach pre-trigger.

**Master plan:** `audits/AUDIT_PLAN.md` v1.1 (retains the layer-by-layer technical scoping; POL-2 governs scheduling/gating)
**Audit-strategy policy:** `docs/governance/2026-05-06-resource-constrained-audit-strategy.md` (PRSM-POL-2)
**Disclosure policy:** `SECURITY.md`

---

## RFP packets in this directory

| Layer | Packet | Recipients | Budget | Wall-clock | Status under PRSM-POL-2 |
|-------|--------|------------|--------|-----------|--------|
| **L3** | [L3-ed25519-crypto-rfp.md](L3-ed25519-crypto-rfp.md) | Trail of Bits Crypto + NCC Group Crypto | $15K-$30K | 2-4 weeks | **DEFERRED — funding-gated.** Substitute: agent-teams self-audit per POL-2 §4.1 |
| **L4** | [L4-code4rena-contest-scope.md](L4-code4rena-contest-scope.md) | Code4rena (public contest) | ~$40K | 14-day contest | **DEFERRED — first-to-fire if any audit budget materializes** (cheapest-per-dollar coverage) |
| **L4** | [L4-firm-rfp-addendum-20260505.md](L4-firm-rfp-addendum-20260505.md) | Trail of Bits + Spearbit-Cantina + OpenZeppelin (firm pair-review) | $60K-$80K | 3 weeks | **DEFERRED — funding-gated.** Substitute: agent-teams self-audit + OZ Pausable + TVL caps + 14-day public review per POL-2 §4 |
| **L5** | [L5-ml-supply-chain-audit-rfp.md](L5-ml-supply-chain-audit-rfp.md) | Trail of Bits ML + NCC Group ML + Berkeley RDI + CMU CyLab | $40K-$80K | 4-6 weeks | **DEFERRED — Tier C public availability also deferred.** |
| **L6f** | [L6f-infrastructure-pentest-rfp.md](L6f-infrastructure-pentest-rfp.md) | NCC Group + Bishop Fox + Doyensec | $20K-$50K | 2-4 weeks | **DEFERRED.** Bootstrap runs in alpha posture; L6f required only before any "production-grade" SLA claim. |
| **L7** | [L7-economic-game-theory-audit-rfp.md](L7-economic-game-theory-audit-rfp.md) | Gauntlet + Chaos Labs + BlockScience + MIT DCI | $50K-$100K | 4-8 weeks | **DEFERRED — DEX listing also deferred.** No CEX/DEX listing application before L7. |
| **L8 Track A** | [L8-legal-counsel-rfp.md](L8-legal-counsel-rfp.md) | Mourant + Walkers + Ogier (Cayman) | $30K-$80K | 8 weeks + retainer | **DEFERRED.** Re-fires when Foundation hires non-founder personnel OR Foundation real-asset custody event. |
| **L8 Track B** | [L8-legal-counsel-rfp.md](L8-legal-counsel-rfp.md) | Cooley + Lowenstein + DLx Law (US securities) | $50K-$150K | 8 weeks + retainer | **NOT DEFERRED — HARD GATE RETAINED.** Fires before any Reg D 506(c) Prismatica solicitation. Personal-liability legal exposure for founder. |
| **L9** | [L9-bug-bounty-program-design.md](L9-bug-bounty-program-design.md) | Immunefi (Channel A) + self-hosted security@prsm.network (Channel B) | $50K + $25K initial pool (proposal only; not active) | Continuous | **GATE FLIPPED PER POL-2:** Channel B (security@prsm.network) is **active now**, pre-L4, as a disclosure channel only (no payouts; see `SECURITY.md` for current channel status). Channel A (Immunefi) deferred. |

**Total combined budget envelope: ~$280K - $645K.** Per POL-2, this envelope is held until §5 revisit triggers fire (e.g., first VC term sheet, FTNS market-cap > $1M, EscrowPool+StakeBond TVL > $50K, etc.).

---

## Sequencing — under PRSM-POL-2

POL-2 replaces the original "fire all RFPs in parallel now" plan with a **trigger-driven** schedule. RFPs fire when their §5 revisit trigger fires, not on a calendar schedule:

| Trigger | Fires |
|---|---|
| First serious VC engagement (term sheet or signed LOI) | L4 firm + L7 economic immediately |
| FTNS market cap > $1M | L4 firm + L7 within 30 days |
| EscrowPool + StakeBond combined TVL > $50K | L4 firm within 60 days |
| Tier C public availability requested | L5 ML supply chain before any non-self-hosted Tier C deployment |
| **First Prismatica investor solicitation** | **L8 Track B fires regardless of broader budget situation** |
| Foundation hires non-founder personnel | L8 Track A within 60 days |
| First disclosed-but-unremediated HIGH finding via security@ | Whatever audit layer covers the finding within 30 days |
| Any cash availability for audits before any §5 trigger | **L4 Code4rena first** (cheapest, broadest coverage) |
| Annual review of POL-2 (every 2026-05-06) | Council formally revisits all deferred layers |

**Critical:** until a trigger fires, founder bandwidth on these RFPs should be zero. The drafts are done; firing is fast (90 minutes for the full 22-email outreach packet) when the moment arrives.

---

## What's been done internally (no RFP needed)

| Layer | Status | Artifact |
|-------|--------|----------|
| L0 | ✅ Continuous internal review | (no artifact) |
| L1 | ✅ Wired in CI | `.github/workflows/solidity-static-analysis.yml` + `audits/L1-static-tooling/README.md` |
| L2 | ✅ AI multi-team review closed | `audits/findings/consolidated.md` (1 CRIT + 7 HIGH + 7 MEDIUM all remediated) |
| L6a | ✅ Foundation Safe deployed | Phase 1.3 Task 8 (2026-05-04) |
| L6b | ✅ Hardware wallet diversity | Ledger + Trezor + OneKey verified |
| L6c | ✅ Key rotation runbook | `docs/security/KEY_ROTATION_RUNBOOK.md` |
| L6d | ✅ Insider threat policy | `docs/security/INSIDER_THREAT_AND_COLLUSION_POLICY.md` |
| L6e | ✅ Continuous ops hygiene template + archive dir | `docs/security/L6E_OPS_HYGIENE_REVIEW.md` + `docs/security/L6E-reviews/` (first review window 2026-07-01 → 2026-07-14) |
| L10a | ✅ Forta routing wired | Discord + Slack + PagerDuty + email; smoke-test CLI |
| L10d | ✅ SECURITY.md + PGP | Published 2026-05-05 |
| L11 (policy) | ✅ Treasury policy ratified 2026-05-06 | `docs/governance/2026-05-05-treasury-reserve-and-risk-transfer-policy.md` (PRSM-POL-1 v1.0) + Council Resolution `docs/governance/PRSM-CR-2026-05-06-1.md` |

---

## Cross-references

- **Master plan:** `audits/AUDIT_PLAN.md` v1.1
- **Decisions:** `audits/decisions/` (L3 ratified)
- **Findings:** `audits/findings/` (L2 consolidated, L3 pre-engagement,
  L1 ongoing CI artifacts)
- **Tooling:** `audits/L1-static-tooling/README.md`
- **Disclosure:** `SECURITY.md` + `docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md`
- **Governance:** `docs/governance/2026-05-05-treasury-reserve-and-risk-transfer-policy.md`

---

## Status update protocol

When a vendor proposal arrives:

1. Update the relevant `Lx-*-rfp.md` with proposal-receipt date + vendor.
2. Add to `audits/AUDIT_PLAN.md` Lx status block.
3. After engagement letter signed: cross-link engagement letter from
   the RFP doc.
4. After report received: file under
   `audits/findings/Lx-<vendor>-<date>/` and update master plan.

When a vendor declines or is unavailable:

1. Note decline reason in the RFP doc (one line).
2. Move to next firm in the recipient list.
3. If all listed firms decline, escalate to council for next-step
   decision.

---

*This index is maintained by the Foundation founder until DoE hire.*
