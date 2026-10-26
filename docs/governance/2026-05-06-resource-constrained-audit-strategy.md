# PRSM-POL-2 — Resource-Constrained Audit Strategy

**Policy ID:** PRSM-POL-2
**Version:** 1.0
**Issued:** 2026-05-06
**Status:** Ratified by PRSM-CR-2026-05-06-5
**Replaces:** Implicit "external audit gates all mainnet deploys" stance documented across `audits/AUDIT_PLAN.md` v1.1, `audits/rfp/README.md`, and tasks #31 / #40.
**Supersedes for the layers it covers; coexists with PRSM-POL-1** (Treasury Reserve & Risk-Transfer Policy) — POL-1 governs treasury custody and disbursement; POL-2 governs audit-engagement scheduling and mainnet-deploy gating in the absence of audit clearance.

---

## 1. Purpose

This policy formally acknowledges that the full-stack external-audit envelope identified in `audits/AUDIT_PLAN.md` v1.1 — combined budget **$280K–$645K** across Layers L3, L4, L4-Code4rena, L5, L6f, L7, L8, and L9 — is out of reach for PRSM as a sole-founder, open-source, pre-funding project.

The previous implicit policy was "all external audits clear before mainnet deploy." Under that policy, PRSM cannot ship the audit-bundle (EscrowPool, StakeBond, SettlementRegistry, SignatureVerifier), Phase 8 emission stack, Phase 7-storage contracts, or PublisherKeyAnchor to mainnet. The policy effectively freezes the project pending capital that has not arrived and may not arrive without first shipping a working, in-use protocol.

This document records the council's deliberate decision to **accept residual audit risk in exchange for shipping**, and defines the minimum-viable mitigations + revisit triggers that make the trade-off explicit, reviewable, and reversible.

## 2. Authority

Adopted under the authority granted by:

- `docs/2026-04-21-prsm-gov-1-foundation-governance-charter.md` — Foundation governance charter.
- `audits/AUDIT_PLAN.md` v1.1 §5, which identifies the founder/council as the body responsible for relaxing layer-specific gates when justified by resource constraints and risk profile.
- The 2026-05-06 mainnet bring-up Path A trace (`docs/2026-05-06-canonical-workflow-base-mainnet-trace.md`) and the FTNS role-migration ceremony (PRSM-CR-2026-05-06-3) demonstrating that the *non-audit-gated* surface is in active production use.

At the present council size (1), council unanimity (1-of-1) is the operative quorum.

## 3. Layer-by-layer treatment

| Layer | Original gate | New treatment under POL-2 | Required pre-condition |
|---|---|---|---|
| **L3** Ed25519 crypto audit | Pre-mainnet audit-bundle clear | **Deferred — funding-gated.** Substituted by agent-teams self-audit run before deploy. | Self-audit findings filed at `audits/findings/L3-self/`; CRIT/HIGH remediated. |
| **L4** Smart-contract audit (firm) | Mainnet deploy of audit-bundle | **Deferred — funding-gated.** Substituted by agent-teams self-audit + OZ Pausable (already wired, see HIGH-3 remediation task #348) + TVL caps (§5) + public GitHub review window. | Self-audit findings + Pausable verification + 14-day public review window with no unaddressed CRIT/HIGH. |
| **L4** Code4rena public contest | (parallel with L4 firm) | **Deferred — funding-gated.** First external audit to fire if any audit budget materializes (~$40K, cheapest-per-dollar coverage). | Funding event. |
| **L5** ML supply-chain audit | Tier C public availability | **Deferred — Tier C public availability also deferred.** Tier C remains available for self-hosted use; public-availability gate is held until L5 clears. | L5 vendor engagement OR explicit founder decision to launch Tier C public without audit. |
| **L6f** Infrastructure pen-test | Production-grade bootstrap | **Deferred.** Bootstrap infrastructure runs in alpha posture (`bootstrap1.prsm-network.com` already live per memory; addition of bootstrap2+ requires no L6f gate). | None for alpha; L6f required before any public claim of "production-grade" SLA. |
| **L7** Economic / game-theory audit | DEX listing | **Deferred — DEX listing also deferred.** FTNS is and remains illiquid until L7 clears. No CEX/DEX listing application before L7 engagement. | L7 vendor engagement before any listing application. |
| **L8 Track A** Cayman counsel | Foundation operating posture | **Deferred but legitimate.** Foundation operating without ongoing Cayman counsel is acceptable while Foundation footprint is small (sole founder, no employees, no real-asset custody beyond Safe). Re-engages when Foundation hires or scales. | Foundation hires first non-founder personnel OR Foundation-real-asset custody event. |
| **L8 Track B** US securities counsel | Reg D 506(c) Prismatica raise | **NOT deferred — hard gate retained.** This is a personal-liability legal exposure for the founder, not just a project-risk item. No Reg D 506(c) solicitation under any framing (warm intro, public website CTA, investor deck distribution to non-pre-existing-relationships) before counsel review. | Engagement letter signed with US securities counsel; counsel-reviewed solicitation materials; counsel sign-off on the specific solicitation activity. |
| **L9** Bug bounty | (gated post-L4) | **Gate flipped: now firing pre-L4.** A `security@prsm.network` channel is already live (L10d task #342). POL-2 ratifies that the security@ email is the active disclosure surface in lieu of Immunefi pending funding. | Disclosure policy in `SECURITY.md` (already published) governs incoming reports. |
| **L11** Treasury policy | (separate) | Already ratified via PRSM-POL-1 + PRSM-CR-2026-05-06-1. POL-2 does not modify. | n/a |

> **Status note (2026-10-26), L9 row:** No bug bounty program was launched. The security@ address is a disclosure channel only and does not pay bounties (see `SECURITY.md` for current reporting channels; GHSA preferred); amounts referenced in L9 planning were proposals only.

## 4. Mitigations adopted in lieu of external audits

For each deferred layer, POL-2 records the substitute mitigations. These are not equivalent to a paid external audit but represent good-faith effort to surface obvious issues before going live.

### 4.1 Self-audit via agent-teams skill (L3 + L4)

The internal AI multi-team review (tasks #325–#327) successfully closed 1 CRITICAL + 7 HIGH findings through agent-teams parallel review. POL-2 ratifies this approach as the substitute external-audit modality for L3 and L4 pre-mainnet review:

- **Trigger:** before any mainnet deploy of an audit-bundle, Phase 8, Phase 7-storage, or PublisherKeyAnchor contract.
- **Artifact:** a fresh agent-teams audit report filed at `audits/findings/L<N>-self-<date>/` covering the contract under review.
- **Pass criterion:** 0 unremediated CRITICAL findings; HIGH findings either remediated or accepted-with-recorded-rationale.
- **Limitations explicitly acknowledged:** agent-teams self-audit does not catch the same finding classes a human firm review would (subtle invariant violations, novel cryptographic mistakes, complex multi-contract interaction bugs). Operators of any contract deployed under POL-2 should treat it as alpha software.

### 4.2 OZ Pausable on all mainnet contracts

Per HIGH-3 remediation (task #348), all audit-bundle contracts now inherit OZ Pausable. The Foundation Safe holds PAUSER_ROLE post-deploy. This gives the council a documented kill-switch independent of L4 audit clearance.

### 4.3 TVL caps for alpha-mainnet contracts

Each contract deployed under POL-2 is governed by a soft TVL cap. The council sets initial caps at deploy time and revisits them quarterly:

| Contract | Initial alpha cap | Revisit |
|---|---|---|
| EscrowPool | $10K total locked stake | Quarterly OR 80% utilization |
| StakeBond | $10K total locked stake | Quarterly OR 80% utilization |
| SettlementRegistry | n/a (read-only batch state) | n/a |
| Phase 8 emission contracts | n/a (capped by `MAX_SUPPLY` already on-chain) | n/a |

If a cap is exceeded, the council shall (a) investigate the rate of growth, (b) decide to expand the cap, (c) pause the contract, or (d) accelerate the funding-gated audit timeline.

### 4.4 14-day public GitHub review window

Before any audit-bundle mainnet deploy under POL-2, the council shall:

1. Open a tracking issue on GitHub describing the contract, the deploy address, the agent-teams findings disposition, and the planned cap.
2. Hold the issue open for **14 calendar days** with a top-of-README link inviting public review.
3. At day 14, if no unaddressed CRIT/HIGH disclosure has arrived via security@prsm.network or the issue, proceed with deploy.
4. If a CRIT/HIGH disclosure does arrive, defer deploy and remediate before re-opening the window.

This is not an audit. It is a transparency mechanism — anyone reading the README can verify the council deliberately invited scrutiny.

### 4.5 Transparent disclosure

The contract's verified Basescan source code, the agent-teams self-audit report, the GitHub review issue, and the deploy-time TVL cap shall all be cross-linked from `prsm/deployments/contract_addresses.json` and from `audits/AUDIT_PLAN.md`. A future auditor (paid or not) shall be able to reconstruct what was reviewed, by whom, when, and what risks were accepted.

## 5. Revisit triggers

POL-2 is not a permanent decision. The following events shall trigger the council to revisit and potentially fire the originally-planned external audits:

| Trigger | Action |
|---|---|
| First serious VC engagement (term sheet or signed letter of intent) | Fire L4 firm + L7 economic immediately. Their reports become diligence artifacts. |
| FTNS market cap (any liquidity venue) > $1M | Fire L4 firm + L7 economic within 30 days. |
| EscrowPool + StakeBond combined TVL > $50K | Fire L4 firm within 60 days. Optionally raise cap. |
| Tier C public availability requested (founder decision) | Fire L5 ML supply chain before any non-self-hosted Tier C deployment. |
| First Prismatica investor solicitation | L8 Track B fires regardless of broader budget situation. (Per §3 — hard gate.) |
| Foundation hires non-founder personnel | L8 Track A fires within 60 days. |
| First disclosed-but-unremediated HIGH finding via security@ | Whatever audit layer covers the finding fires within 30 days. |
| Annual review of POL-2 (every 2026-05-06) | Council formally revisits all deferred layers and the deferred-vs-fire decision per layer. |

## 6. Risk acceptance — explicit

The council accepts that under POL-2:

1. **Mainnet contracts deployed without L4 audit may have undiscovered bugs that move funds wrongly.** OZ Pausable + TVL caps + agent-teams self-audit + 14-day public window are mitigations, not equivalent insurance. If a bug occurs, the Foundation eats the loss up to the cap.
2. **The economic / tokenomics design has not been adversarially simulated.** Halving curve, slashing dynamics, and serving-node incentive alignment have been internally reviewed but not third-party stress-tested. If FTNS becomes liquid (post-DEX listing, which is itself L7-gated), an attacker could grief tokenomics in ways the design did not anticipate.
3. **The off-chain ML pipeline has not been adversarially reviewed.** Tier C TEE attestation, model loading, and prompt-injection surfaces are internally hardened but not third-party probed.
4. **The bootstrap infrastructure has not been pen-tested.** Bootstrap nodes carry SLA risk for the network's discoverability layer.
5. **Public bug bounty pool is small.** A self-hosted security@ channel pays out per-finding from Foundation Safe at council discretion (per PRSM-POL-1 §5). This is significantly lower-friction than Immunefi but provides smaller incentive for serious researchers.

   > **Status note (2026-10-26):** No bug bounty program was launched. The security@ address is a disclosure channel only and does not pay bounties (see `SECURITY.md` for current reporting channels; GHSA preferred); the per-finding payout described here was a proposal only and was never activated.

The council judges these accepted risks **lower in expected impact than the alternative risk of indefinite project freeze pending unfunded audits**, while PRSM is small enough that catastrophic loss is bounded by §4.3 caps + Pausable.

## 7. Periodic review

POL-2 is reviewed annually on its issuance anniversary (next: 2026-05-06 → 2027-05-06) by the council, and immediately upon any §5 trigger event.

The annual review shall produce one of:

- **Reaffirmation** (no material change to PRSM's funding posture or scale).
- **Layer-by-layer adjustment** (some deferred layers fire; others remain deferred).
- **Full retirement** (PRSM is now well-funded; full external-audit envelope fires).

Reviews are documented as PRSM-POL-2.<minor> revisions and ratified via accompanying PRSM-CR-* resolutions.

## 8. Adoption

Adopted by Council Resolution PRSM-CR-2026-05-06-5 on 2026-05-06.

> **Signed:** Ryne Schultz, Founder
>
> **Date:** 2026-05-06
