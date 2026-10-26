# Vendor Outreach Email Packet

**Purpose:** Ready-to-send drafts for the 8 vendor-track RFPs in
`audits/rfp/`. One section per RFP; one email block per recipient.

**Author:** Ryne Schultz, Founder, PRSM Foundation
**From:** schultzryne@gmail.com (Cc: security@prsm.network)
**PGP:** see [`SECURITY.md`](../../SECURITY.md)

---

## How to use this packet

1. **Verify the recipient address before sending.** Most large
   firms route cold inquiries through web intake forms rather than
   public email. For each recipient I list either (a) a known
   intake address, or (b) the firm's intake URL with a note "verify
   contact path." When in doubt, use the intake form — it routes to
   business-development faster than a generic info@ inbox.
2. **Send each RFP's emails in the same business day.** Parallel
   solicitation is disclosed in each email; sending them on
   different days creates information asymmetry between recipients.
3. **Track replies in `audits/rfp/README.md` per the status-update
   protocol** (§ "Status update protocol" at the bottom of that
   file).
4. **Attach nothing.** Each email links the relevant RFP via the
   public GitHub URL. Firms expect to read the scope on their own
   time; attaching PDFs reduces the response rate.

**Repository URL convention:**
`https://github.com/Ryno2390/PRSM/blob/main/audits/rfp/<file>.md`

---

## Sequencing (from `audits/rfp/README.md`)

Send order over ~2 business days:

**Day 1 morning** — L4 (firm + Code4rena), L3 (gates audit-bundle
clear)
**Day 1 afternoon** — L8 Track B (US securities), L8 Track A (Cayman)
**Day 2 morning** — L5 (ML), L7 (economic)
**Day 2 afternoon** — L6f (infrastructure)

L9 (Immunefi bounty) is **NOT** in this packet — its activation is
gated POST-L4 clear. Send the L9 outreach email post-2026-08-15
target.

> **Status (2026-10-26):** Proposal only; L9 was never activated.
> PRSM does not run a bug bounty program and does not pay for vulnerability reports.

---

# §1 — L4 Firm Pair-Review (audit-bundle smart contracts)

**RFP:** `audits/rfp/L4-firm-rfp-addendum-20260505.md` + base RFP
`docs/2026-04-23-auditor-shortlist-and-rfp.md`
**Budget:** $60K – $80K
**Window:** 3 weeks, target 2026-06-09 → 2026-07-07
**Recipients:** Trail of Bits + Spearbit/Cantina + OpenZeppelin

---

### To: **Trail of Bits** (intake form: <https://www.trailofbits.com/contact/>)

**Subject:** RFP — PRSM audit-bundle Solidity review (Base mainnet, ~2,300 SLOC, $60K-$80K, 3 weeks)

Hi Trail of Bits team,

I'm Ryne Schultz, founder of the PRSM Foundation (Cayman Islands
nonprofit). PRSM is a decentralized research / storage / modeling
protocol with smart contracts deployed on Base mainnet — see
<https://github.com/Ryno2390/PRSM>.

We are issuing a firm pair-review RFP for our audit-bundle (~2,300
SLOC across 9 contracts: EscrowPool, StakeBond, BatchSettlementRegistry,
RoyaltyDistributor v2, EmissionController, CompensationDistributor,
StorageSlashing, KeyDistribution, Ed25519Verifier wrapper). We've
already completed an internal multi-team adversarial review (1
CRITICAL + 7 HIGH + 7 MEDIUM all remediated) and a pre-engagement
artifact pack so your reviewers can start with a clean baseline.

**Full RFP:**
<https://github.com/Ryno2390/PRSM/blob/main/audits/rfp/L4-firm-rfp-addendum-20260505.md>

Budget envelope: $60K-$80K. Target window: 2026-06-09 → 2026-07-07.

We are also soliciting proposals from Spearbit/Cantina and OpenZeppelin
in parallel. Our preference is one of you for the firm slot + the
public Code4rena contest as concurrent coverage.

Trail of Bits is on our shortlist specifically for your prior work on
Base / OP Stack chains and your published Ed25519 reviews (separate L3
crypto-specialist engagement may also interest your cryptography
practice — see `audits/rfp/L3-ed25519-crypto-rfp.md`).

Could we schedule a 30-minute scoping call this week or next? I can
share a Calendly or pick a time that works for your team.

Thanks,
Ryne Schultz
Founder, PRSM Foundation
schultzryne@gmail.com / security@prsm.network
PGP: see SECURITY.md

---

### To: **Spearbit / Cantina** (intake: <https://cantina.xyz/welcome> or ops@spearbit.com — verify contact path)

**Subject:** RFP — PRSM audit-bundle Solidity review (Base mainnet, ~2,300 SLOC, $60K-$80K, 3 weeks)

Hi Spearbit team,

I'm Ryne Schultz, founder of the PRSM Foundation. PRSM is a
decentralized research / storage / modeling protocol on Base mainnet
— see <https://github.com/Ryno2390/PRSM>.

We're issuing a firm pair-review RFP for our pre-mainnet audit-bundle
(~2,300 SLOC across 9 contracts). We've already completed an internal
multi-team adversarial review (1 CRITICAL + 7 HIGH + 7 MEDIUM all
remediated) and a pre-engagement artifact pack.

**Full RFP:**
<https://github.com/Ryno2390/PRSM/blob/main/audits/rfp/L4-firm-rfp-addendum-20260505.md>

Budget envelope: $60K-$80K. Target window: 2026-06-09 → 2026-07-07.

We are also soliciting proposals from Trail of Bits and OpenZeppelin
in parallel.

Spearbit is on our shortlist specifically for your Base-mainnet
experience and the Cantina marketplace's flexibility on senior-
reviewer pairings. We're open to either a Spearbit-direct engagement
or a curated Cantina pairing if you'd like to propose specific
reviewers.

Could we schedule a 30-minute scoping call this week or next?

Thanks,
Ryne Schultz
Founder, PRSM Foundation
schultzryne@gmail.com / security@prsm.network
PGP: see SECURITY.md

---

### To: **OpenZeppelin Security** (security-services@openzeppelin.com — verify on <https://www.openzeppelin.com/security-audits>)

**Subject:** RFP — PRSM audit-bundle Solidity review (Base mainnet, ~2,300 SLOC, $60K-$80K, 3 weeks)

Hi OpenZeppelin Security team,

I'm Ryne Schultz, founder of the PRSM Foundation. PRSM is a
decentralized research / storage / modeling protocol on Base
mainnet — see <https://github.com/Ryno2390/PRSM>.

We're issuing a firm pair-review RFP for our audit-bundle (~2,300
SLOC across 9 contracts). The codebase makes heavy use of OpenZeppelin
patterns — Ownable2Step, AccessControlUpgradeable, UUPSUpgradeable,
Pausable, ReentrancyGuard. We've completed an internal multi-team
adversarial review (1 CRITICAL + 7 HIGH + 7 MEDIUM all remediated).

**Full RFP:**
<https://github.com/Ryno2390/PRSM/blob/main/audits/rfp/L4-firm-rfp-addendum-20260505.md>

Budget envelope: $60K-$80K. Target window: 2026-06-09 → 2026-07-07.

We are also soliciting proposals from Trail of Bits and
Spearbit/Cantina in parallel.

OpenZeppelin is on our shortlist specifically for your fluency with
the OZ libraries we depend on — recent issues we caught internally
(e.g., B-OWNABLE-1 single-step Ownable migration) are exactly the
kind of pattern your reviewers would have flagged.

Could we schedule a 30-minute scoping call this week or next?

Thanks,
Ryne Schultz
Founder, PRSM Foundation
schultzryne@gmail.com / security@prsm.network
PGP: see SECURITY.md

---

# §2 — L4 Code4rena Public Contest

**RFP:** `audits/rfp/L4-code4rena-contest-scope.md`
**Budget:** ~$40K target prize pool
**Window:** 14-day public contest (target 2026-06-09 → 2026-06-23)
**Recipient:** Code4rena intake

---

### To: **Code4rena** — intake@code4rena.com

**Subject:** Contest scoping — PRSM audit-bundle (Base mainnet, ~2,300 SLOC, ~$40K pool)

Hi Code4rena team,

I'd like to scope a public contest for the PRSM audit-bundle —
~2,300 SLOC across 9 Solidity contracts deploying to Base mainnet.

PRSM is a decentralized research / storage / modeling protocol; the
in-scope contracts cover an escrow pool, provider stake bond, batch
settlement registry, royalty distributor (v2 pull-payment),
emission controller, compensation distributor, storage slashing,
key distribution, and an Ed25519 verifier wrapper.

**Pre-engagement state:**
- Internal multi-team adversarial review CLOSED — 1 CRITICAL + 7 HIGH
  + 7 MEDIUM all remediated.
- L1 static + symbolic tooling (Slither + Aderyn + Mythril) wired in
  CI; clean as of 2026-05-05.
- Concurrent L3 crypto-specialist engagement on the Ed25519 + SHA-512
  primitives (separate scope).
- Concurrent firm pair-review (Trail of Bits / Spearbit / OZ).

**Scoping packet (in-scope files, known-non-bugs, attack-surface
priority list):**
<https://github.com/Ryno2390/PRSM/blob/main/audits/rfp/L4-code4rena-contest-scope.md>

**Target:** ~$40K prize pool, 14-day contest window 2026-06-09 →
2026-06-23 (negotiable based on your scheduling).

What's your typical onboarding flow + lead time from "let's do it" to
contest live? Happy to do a scoping call.

Thanks,
Ryne Schultz
Founder, PRSM Foundation
schultzryne@gmail.com

---

# §3 — L3 Cryptographic Specialist Audit (Ed25519 + SHA-512)

**RFP:** `audits/rfp/L3-ed25519-crypto-rfp.md`
**Budget:** $15K – $30K
**Window:** 2-4 weeks, target 2026-05-26 start
**Recipients:** Trail of Bits Cryptography + NCC Group Cryptography

---

### To: **Trail of Bits Cryptography practice** (intake form: <https://www.trailofbits.com/contact/> — note "Cryptography" in routing)

**Subject:** RFP — Ed25519 + SHA-512 Solidity verifier review (~1,274 LoC, $15K-$30K, 2-4 weeks)

Hi Trail of Bits Cryptography team,

I'm Ryne Schultz, founder of the PRSM Foundation. We're seeking a
**cryptography-specialist firm** (not generic smart-contract review)
for a focused audit of an on-chain Ed25519 + SHA-512 implementation.

**Scope (~1,274 LoC):**
- `Ed25519Lib.sol` — 887 LoC, hand-rolled Solidity port of
  `chengwenxi/Ed25519` (Apache-2.0)
- `Sha512.sol` — 328 LoC
- `Ed25519Verifier.sol` — 59 LoC integration wrapper

**Why specialist:** the verifier is on the INVALID_SIGNATURE
challenge path of our batch-settlement protocol. A signature
malleability bug, compression edge case, or curve-arithmetic bug
maps directly to economic loss. We need reviewers with prior
published work on Ed25519 / Curve25519 / FIPS 180-4 — your team's
record on similar primitives is exactly the fit.

**Pre-engagement work already done** so you can start clean:
- RFC 8032 + FIPS 180-4 test-vector harness
- Upstream port diff vs `chengwenxi/Ed25519`
- Caller-assumptions memo

**Full RFP:**
<https://github.com/Ryno2390/PRSM/blob/main/audits/rfp/L3-ed25519-crypto-rfp.md>

Budget: $15K-$30K. Target start: 2026-05-26. We are also soliciting
NCC Group Cryptography Services in parallel.

Could we schedule a 30-minute scoping call this week?

Thanks,
Ryne Schultz
Founder, PRSM Foundation
schultzryne@gmail.com / security@prsm.network
PGP: see SECURITY.md

---

### To: **NCC Group Cryptography Services** (intake: <https://www.nccgroup.com/us/contact-us/> — note "Cryptography Services")

**Subject:** RFP — Ed25519 + SHA-512 Solidity verifier review (~1,274 LoC, $15K-$30K, 2-4 weeks)

Hi NCC Group Cryptography Services team,

I'm Ryne Schultz, founder of the PRSM Foundation. We're seeking a
cryptography-specialist firm for a focused review of an on-chain
Ed25519 + SHA-512 verifier (~1,274 LoC of curve-arithmetic + hash
code) deploying to Base mainnet.

The verifier is on our INVALID_SIGNATURE challenge path; a
malleability bug, compression edge case, or curve-arithmetic mistake
maps directly to economic loss. We need reviewers with published
prior work on Ed25519 / Curve25519 — NCC Cryptography Services'
record on similar primitives is the fit.

**Scope:**
- `Ed25519Lib.sol` — 887 LoC, Solidity port of `chengwenxi/Ed25519`
  (Apache-2.0)
- `Sha512.sol` — 328 LoC
- `Ed25519Verifier.sol` — 59 LoC integration wrapper

**Pre-engagement artifacts ready:** RFC 8032 + FIPS 180-4 test-vector
harness, upstream port diff, caller-assumptions memo.

**Full RFP:**
<https://github.com/Ryno2390/PRSM/blob/main/audits/rfp/L3-ed25519-crypto-rfp.md>

Budget: $15K-$30K. Target start: 2026-05-26. We are soliciting Trail
of Bits Cryptography in parallel.

Could we schedule a 30-minute scoping call this week?

Thanks,
Ryne Schultz
Founder, PRSM Foundation
schultzryne@gmail.com / security@prsm.network
PGP: see SECURITY.md

---

# §4 — L8 Track B: US Securities Counsel

**RFP:** `audits/rfp/L8-legal-counsel-rfp.md` Track B
**Budget:** $50K – $150K initial + ongoing retainer
**Window:** 8 weeks initial opinion + ongoing
**Recipients:** Cooley + Lowenstein Sandler + DLx Law

---

### To: **Cooley LLP** (route through firm intake: <https://www.cooley.com/contact> — preferably to a partner in the Digital Assets / Blockchain practice)

**Subject:** Engagement inquiry — PRSM Foundation (Cayman) + Prismatica (DE C-corp) US securities counsel, Reg D 506(c) raise + FTNS classification

Dear Cooley team,

I'm Ryne Schultz, founder of the PRSM Foundation, a Cayman Islands
nonprofit foundation operating the PRSM protocol — a decentralized
research / storage / modeling network with smart contracts on Base
mainnet.

We are seeking US securities counsel for two related matters:

1. **FTNS token classification.** Howey-test analysis of the FTNS
   utility token, given the protocol's two-entity design (Foundation
   nonprofit + Prismatica DE C-corp). We have an internal working
   memo and would value an outside-counsel opinion before mainnet
   public availability.

2. **Reg D 506(c) Prismatica equity raise.** Prismatica is the
   for-profit operating arm; the capital raise is structured as
   equity in Prismatica (NOT FTNS), with general-solicitation
   exemption under 506(c). We need form documents (PPM / sub
   docs / Form D / accredited verification flow) and counsel
   throughout.

**Full RFP (both tracks):**
<https://github.com/Ryno2390/PRSM/blob/main/audits/rfp/L8-legal-counsel-rfp.md>

Budget envelope: $50K-$150K initial + ongoing retainer. Target start:
within 4 weeks of engagement letter signed.

We are soliciting Lowenstein Sandler and DLx Law in parallel for
Track B (US securities), and a separate Track A panel (Mourant /
Walkers / Ogier) for Cayman counsel.

Cooley is on our shortlist for your depth in token-issuer matters
and the Foundation/operating-company two-entity structures common in
the space. Could we schedule a 30-minute intro call?

Best regards,
Ryne Schultz
Founder, PRSM Foundation
schultzryne@gmail.com

---

### To: **Lowenstein Sandler LLP** (route through Crypto & Digital Assets practice: <https://www.lowenstein.com/practices/blockchain-cryptocurrency-fintech>)

**Subject:** Engagement inquiry — PRSM Foundation (Cayman) + Prismatica (DE C-corp) US securities counsel, Reg D 506(c) raise + FTNS classification

Dear Lowenstein team,

I'm Ryne Schultz, founder of the PRSM Foundation (Cayman Islands
nonprofit). PRSM is a decentralized research / storage / modeling
protocol on Base mainnet, with a two-entity design: Foundation
nonprofit + Prismatica (Delaware C-corp) operating arm.

We are seeking US securities counsel for two matters:

1. **FTNS token classification** — Howey-test outside-counsel
   opinion before mainnet public availability.
2. **Reg D 506(c) Prismatica equity raise** — equity in Prismatica
   (not FTNS), 506(c) general-solicitation exemption, full form-doc
   support.

**Full RFP:**
<https://github.com/Ryno2390/PRSM/blob/main/audits/rfp/L8-legal-counsel-rfp.md>

Budget envelope: $50K-$150K initial + ongoing retainer. Target start:
within 4 weeks of engagement letter signed.

We are soliciting Cooley and DLx Law in parallel for Track B; a
separate Cayman panel handles Track A.

Lowenstein is on our shortlist for your blockchain practice's record
on Foundation / operating-company structures — your team's
familiarity with the patterns we're using would shorten the onboarding
significantly. Could we schedule a 30-minute intro call?

Best regards,
Ryne Schultz
Founder, PRSM Foundation
schultzryne@gmail.com

---

### To: **DLx Law** (intake: <https://www.dlxlaw.com/contact>)

**Subject:** Engagement inquiry — PRSM Foundation (Cayman) + Prismatica (DE C-corp) US securities counsel, Reg D 506(c) raise + FTNS classification

Dear DLx Law team,

I'm Ryne Schultz, founder of the PRSM Foundation (Cayman). PRSM is a
decentralized research / storage / modeling protocol on Base
mainnet, with a two-entity design: Foundation nonprofit + Prismatica
(Delaware C-corp) operating arm.

We are seeking US securities counsel for FTNS token classification
(Howey-test outside-counsel opinion) and a Reg D 506(c) Prismatica
equity raise.

**Full RFP:**
<https://github.com/Ryno2390/PRSM/blob/main/audits/rfp/L8-legal-counsel-rfp.md>

Budget envelope: $50K-$150K initial + ongoing retainer.

We are soliciting Cooley and Lowenstein Sandler in parallel.

DLx Law is on our shortlist for the firm's deep specialization in
blockchain securities matters — exactly the kind of focused practice
we want for the FTNS classification opinion. Could we schedule a
30-minute intro call?

Best regards,
Ryne Schultz
Founder, PRSM Foundation
schultzryne@gmail.com

---

# §5 — L8 Track A: Cayman Foundation Counsel

**RFP:** `audits/rfp/L8-legal-counsel-rfp.md` Track A
**Budget:** $30K – $80K initial + ongoing retainer
**Window:** 8 weeks initial + ongoing
**Recipients:** Mourant + Walkers + Ogier

---

### To: **Mourant Cayman** (intake: <https://www.mourant.com/contact-us> — Funds & Investment Management or Corporate)

**Subject:** Engagement inquiry — PRSM Foundation (Cayman Foundation Company) ongoing counsel

Dear Mourant team,

I'm Ryne Schultz, founder of the PRSM Foundation, a Cayman Islands
nonprofit Foundation Company under the Foundation Companies Act 2017.
PRSM operates a decentralized research / storage / modeling protocol
on Base mainnet.

We are seeking Cayman counsel for ongoing Foundation matters:

- Foundation Companies Act 2017 compliance (annual returns,
  beneficiary designations, council resolutions).
- Cross-jurisdictional review of governance amendments + treasury
  policy (PRSM-POL-1) for Cayman regulatory fit.
- Ongoing retainer for council operational questions.

**Full RFP (both tracks):**
<https://github.com/Ryno2390/PRSM/blob/main/audits/rfp/L8-legal-counsel-rfp.md>

Budget envelope: $30K-$80K initial + ongoing retainer.

We are soliciting Walkers and Ogier in parallel for Track A; a
separate Track B panel (Cooley / Lowenstein / DLx) handles US
securities.

Mourant is on our shortlist for your foundation-company practice's
record. Could we schedule a 30-minute intro call?

Best regards,
Ryne Schultz
Founder, PRSM Foundation
schultzryne@gmail.com

---

### To: **Walkers Cayman** (intake: <https://www.walkersglobal.com/index.php/contact-us>)

**Subject:** Engagement inquiry — PRSM Foundation (Cayman Foundation Company) ongoing counsel

Dear Walkers team,

I'm Ryne Schultz, founder of the PRSM Foundation, a Cayman
Foundation Company. PRSM operates a decentralized research / storage
/ modeling protocol on Base mainnet.

We are seeking Cayman counsel for Foundation Companies Act 2017
compliance, cross-jurisdictional review of governance + treasury
policy (PRSM-POL-1), and an ongoing retainer.

**Full RFP:**
<https://github.com/Ryno2390/PRSM/blob/main/audits/rfp/L8-legal-counsel-rfp.md>

Budget envelope: $30K-$80K initial + ongoing retainer.

We are soliciting Mourant and Ogier in parallel for Track A.

Walkers is on our shortlist for the firm's depth in Cayman fund and
foundation structures relevant to Web3 protocols. Could we schedule
a 30-minute intro call?

Best regards,
Ryne Schultz
Founder, PRSM Foundation
schultzryne@gmail.com

---

### To: **Ogier Cayman** (intake: <https://www.ogier.com/locations/cayman-islands>)

**Subject:** Engagement inquiry — PRSM Foundation (Cayman Foundation Company) ongoing counsel

Dear Ogier team,

I'm Ryne Schultz, founder of the PRSM Foundation, a Cayman
Foundation Company. PRSM operates a decentralized research / storage
/ modeling protocol on Base mainnet.

We are seeking Cayman counsel for Foundation Companies Act 2017
compliance, governance + treasury-policy review (PRSM-POL-1), and
ongoing retainer for council operational questions.

**Full RFP:**
<https://github.com/Ryno2390/PRSM/blob/main/audits/rfp/L8-legal-counsel-rfp.md>

Budget envelope: $30K-$80K initial + ongoing retainer.

We are soliciting Mourant and Walkers in parallel for Track A.

Ogier is on our shortlist for the firm's foundation-company
practice. Could we schedule a 30-minute intro call?

Best regards,
Ryne Schultz
Founder, PRSM Foundation
schultzryne@gmail.com

---

# §6 — L5 ML Supply-Chain Audit

**RFP:** `audits/rfp/L5-ml-supply-chain-audit-rfp.md`
**Budget:** $40K – $80K
**Window:** 4-6 weeks
**Recipients:** Trail of Bits ML + NCC Group ML + Berkeley RDI + CMU CyLab

---

### To: **Trail of Bits ML/AI Assurance practice** (intake form: <https://www.trailofbits.com/contact/> — note "Machine Learning Assurance" in routing)

**Subject:** RFP — PRSM off-chain ML pipeline audit (~28K LoC Tier 1 + ~7K LoC Tier 2, $40K-$80K, 4-6 weeks)

Hi Trail of Bits ML team,

I'm Ryne Schultz, founder of the PRSM Foundation. PRSM is a
decentralized inference / royalty / settlement protocol; the off-
chain side is a ~28K-LoC Python pipeline covering tensor-parallel
+ Parallax-style sharded inference, RPC + handoff tokens,
streaming generators, manifest DHT, and per-tier privacy controls
(Tier B/C TEE gating).

The audit-bundle smart contracts cover the on-chain settlement
side; this L5 engagement covers the off-chain ML supply-chain side
that on-chain auditors typically don't have ML systems-security
background to address.

**Full RFP (8 audit dimensions including model-supply-chain
integrity, activation extraction, KV-cache cross-tenant isolation,
Tier C constant-time properties, threshold-key distribution
integrity, receipt forgery, etc.):**
<https://github.com/Ryno2390/PRSM/blob/main/audits/rfp/L5-ml-supply-chain-audit-rfp.md>

Budget: $40K-$80K. Target window: 4-6 weeks, start within 4 weeks
of engagement.

We are soliciting NCC Group ML, Berkeley RDI, and CMU CyLab in
parallel.

Trail of Bits is on our shortlist for your team's published work on
ML-systems security. Could we schedule a scoping call?

Thanks,
Ryne Schultz
Founder, PRSM Foundation
schultzryne@gmail.com / security@prsm.network
PGP: see SECURITY.md

---

### To: **NCC Group ML/AI Security practice** (intake: <https://www.nccgroup.com/us/contact-us/>)

**Subject:** RFP — PRSM off-chain ML pipeline audit (~28K LoC Tier 1 + ~7K LoC Tier 2, $40K-$80K, 4-6 weeks)

Hi NCC Group ML/AI Security team,

I'm Ryne Schultz, founder of the PRSM Foundation. We're seeking an
ML-systems-security firm for an off-chain Python ML-pipeline audit
(~28K LoC Tier 1 + ~7K LoC Tier 2) covering tensor-parallel + sharded
inference, RPC + handoff tokens, streaming generators, manifest DHT,
and per-tier privacy controls.

**Full RFP (8 audit dimensions):**
<https://github.com/Ryno2390/PRSM/blob/main/audits/rfp/L5-ml-supply-chain-audit-rfp.md>

Budget: $40K-$80K. Target window: 4-6 weeks.

We are soliciting Trail of Bits ML, Berkeley RDI, and CMU CyLab in
parallel.

NCC Group is on our shortlist for the firm's ML/AI security
practice. Could we schedule a scoping call?

Thanks,
Ryne Schultz
Founder, PRSM Foundation
schultzryne@gmail.com / security@prsm.network
PGP: see SECURITY.md

---

### To: **UC Berkeley RDI (Responsible Decentralized Intelligence)** — verify current PI contact at <https://rdi.berkeley.edu>

**Subject:** Research collaboration inquiry — PRSM ML supply-chain audit (academic engagement, $40K-$80K)

Hi RDI team,

I'm Ryne Schultz, founder of the PRSM Foundation. PRSM operates a
decentralized inference / royalty protocol; we're seeking an academic
or industry research engagement for an ML supply-chain audit covering
~28K LoC of Python ML pipeline (tensor-parallel + Parallax-style
sharded inference, RPC + handoff tokens, threshold-key distribution,
Tier C constant-time properties).

This work overlaps directly with RDI's research areas
(decentralized inference, model integrity, privacy-preserving
computation). We'd be open to either a sponsored-research engagement
or a focused audit with formal deliverable.

**Full RFP (8 audit dimensions):**
<https://github.com/Ryno2390/PRSM/blob/main/audits/rfp/L5-ml-supply-chain-audit-rfp.md>

Budget envelope: $40K-$80K, flexible to academic engagement
structure.

We are also engaging Trail of Bits ML, NCC Group ML, and CMU CyLab
in parallel.

Could we schedule a 30-minute call to scope?

Thanks,
Ryne Schultz
Founder, PRSM Foundation
schultzryne@gmail.com

---

### To: **CMU CyLab** — verify current contact at <https://www.cylab.cmu.edu>

**Subject:** Research collaboration inquiry — PRSM ML supply-chain audit (academic engagement, $40K-$80K)

Hi CyLab team,

I'm Ryne Schultz, founder of the PRSM Foundation. PRSM operates a
decentralized inference / royalty protocol; we're seeking an
academic engagement for an ML supply-chain audit (~28K LoC Python
ML pipeline) covering integrity, isolation, and constant-time
properties.

This work overlaps with CyLab research themes (systems security,
ML security, privacy). We'd be open to a sponsored-research
engagement or a focused audit with formal deliverable.

**Full RFP:**
<https://github.com/Ryno2390/PRSM/blob/main/audits/rfp/L5-ml-supply-chain-audit-rfp.md>

Budget envelope: $40K-$80K, flexible to academic engagement
structure.

We are also engaging Trail of Bits ML, NCC Group ML, and Berkeley
RDI in parallel.

Could we schedule a 30-minute call to scope?

Thanks,
Ryne Schultz
Founder, PRSM Foundation
schultzryne@gmail.com

---

# §7 — L7 Economic / Game-Theory Audit

**RFP:** `audits/rfp/L7-economic-game-theory-audit-rfp.md`
**Budget:** $50K – $100K
**Window:** 4-8 weeks
**Recipients:** Gauntlet + Chaos Labs + BlockScience + MIT DCI

---

### To: **Gauntlet** (intake: <https://www.gauntlet.network/contact>)

**Subject:** RFP — PRSM economic / game-theory audit (royalty + emission + slashing parameters, $50K-$100K, 4-8 weeks)

Hi Gauntlet team,

I'm Ryne Schultz, founder of the PRSM Foundation. PRSM is a
decentralized research / storage / modeling protocol with smart
contracts on Base mainnet. We're seeking economic / game-theory
audit coverage for our token-mechanism design.

**Scope (7 specific quantitative questions in the RFP):**
- 2% network-fee + RoyaltyDistributor split parameters
- Halving emission schedule + epoch parameters
- Slashing parameters + provider incentive alignment
- Stake-bond required deposit thresholds
- Per-tier compensation rate vector calibration
- POL (protocol-owned-liquidity) intervention thresholds + kill-switch
- Stake-vs-storage-vs-compute reward allocation

**Full RFP:**
<https://github.com/Ryno2390/PRSM/blob/main/audits/rfp/L7-economic-game-theory-audit-rfp.md>

Budget: $50K-$100K. Target window: 4-8 weeks.

We are soliciting Chaos Labs, BlockScience, and MIT DCI in
parallel.

Gauntlet is on our shortlist for your simulation-driven approach to
parameter calibration. Could we schedule a 30-minute scoping call?

Thanks,
Ryne Schultz
Founder, PRSM Foundation
schultzryne@gmail.com

---

### To: **Chaos Labs** (intake: <https://chaoslabs.xyz/contact>)

**Subject:** RFP — PRSM economic / game-theory audit (royalty + emission + slashing parameters, $50K-$100K, 4-8 weeks)

Hi Chaos Labs team,

I'm Ryne Schultz, founder of the PRSM Foundation. PRSM is a
decentralized research / storage / modeling protocol on Base
mainnet. We're seeking economic / game-theory audit coverage for our
token mechanism — royalty + emission + slashing + stake-bond
parameters across 7 specific quantitative questions.

**Full RFP:**
<https://github.com/Ryno2390/PRSM/blob/main/audits/rfp/L7-economic-game-theory-audit-rfp.md>

Budget: $50K-$100K. Target window: 4-8 weeks.

We are soliciting Gauntlet, BlockScience, and MIT DCI in parallel.

Chaos Labs is on our shortlist for your real-time risk-monitoring
work — useful both for the parameter-calibration audit and as a
potential follow-on to ongoing post-launch monitoring. Could we
schedule a scoping call?

Thanks,
Ryne Schultz
Founder, PRSM Foundation
schultzryne@gmail.com

---

### To: **BlockScience** (intake: <https://block.science/contact>)

**Subject:** RFP — PRSM economic / game-theory audit (royalty + emission + slashing parameters, $50K-$100K, 4-8 weeks)

Hi BlockScience team,

I'm Ryne Schultz, founder of the PRSM Foundation. PRSM is a
decentralized research / storage / modeling protocol on Base
mainnet. We're seeking economic / game-theory audit coverage for
our token mechanism design.

The audit covers 7 specific quantitative questions — royalty +
emission + slashing + stake-bond parameters + POL intervention
thresholds + per-tier compensation calibration.

**Full RFP:**
<https://github.com/Ryno2390/PRSM/blob/main/audits/rfp/L7-economic-game-theory-audit-rfp.md>

Budget: $50K-$100K. Target window: 4-8 weeks.

We are soliciting Gauntlet, Chaos Labs, and MIT DCI in parallel.

BlockScience is on our shortlist for your cadCAD-style modeling
approach. Could we schedule a scoping call?

Thanks,
Ryne Schultz
Founder, PRSM Foundation
schultzryne@gmail.com

---

### To: **MIT Digital Currency Initiative (DCI)** — verify contact at <https://dci.mit.edu>

**Subject:** Research collaboration inquiry — PRSM economic / game-theory audit (academic engagement, $50K-$100K)

Hi DCI team,

I'm Ryne Schultz, founder of the PRSM Foundation. PRSM is a
decentralized research / storage / modeling protocol on Base
mainnet. We're seeking an academic engagement for an economic /
game-theory audit of our token mechanism — royalty + emission +
slashing + stake-bond parameters.

This work overlaps with DCI research themes (cryptoeconomic
mechanism design, decentralized protocols).

**Full RFP:**
<https://github.com/Ryno2390/PRSM/blob/main/audits/rfp/L7-economic-game-theory-audit-rfp.md>

Budget envelope: $50K-$100K, flexible to academic engagement
structure.

We are also engaging Gauntlet, Chaos Labs, and BlockScience in
parallel.

Could we schedule a 30-minute call to scope?

Thanks,
Ryne Schultz
Founder, PRSM Foundation
schultzryne@gmail.com

---

# §8 — L6f Network Infrastructure Pen-Test

**RFP:** `audits/rfp/L6f-infrastructure-pentest-rfp.md`
**Budget:** $20K – $50K
**Window:** 2-4 weeks
**Recipients:** NCC Group + Bishop Fox + Doyensec

---

### To: **NCC Group Network/Infrastructure practice** (intake: <https://www.nccgroup.com/us/contact-us/>)

**Subject:** RFP — PRSM bootstrap-node + protocol infrastructure pen-test ($20K-$50K, 2-4 weeks)

Hi NCC Group team,

I'm Ryne Schultz, founder of the PRSM Foundation. PRSM operates a
decentralized research / storage / modeling protocol on Base
mainnet, with a live bootstrap node at
`wss://bootstrap-us.prsm-network.com:8765` (DigitalOcean droplet).

We're seeking a network/infrastructure pen-test covering:

- Bootstrap-node availability + DDoS resilience
- Transport-layer adapter security (Direct + SOCKS, R9 surface)
- DHT poisoning resistance (Manifest DHT + Profile DHT under sybil
  + eclipse-attack scenarios)
- Storage-layer availability under partial network failure
- Rate-limiting + connection-liveness exhaustion
- TLS configuration audit (cert pinning, protocol versions)

The RFP includes a coordination clause for the live bootstrap node so
testing doesn't accidentally produce a real outage.

**Full RFP:**
<https://github.com/Ryno2390/PRSM/blob/main/audits/rfp/L6f-infrastructure-pentest-rfp.md>

Budget: $20K-$50K. Target window: 2-4 weeks.

We are soliciting Bishop Fox and Doyensec in parallel.

NCC Group is on our shortlist for your network-protocol pen-test
record. Could we schedule a scoping call?

Thanks,
Ryne Schultz
Founder, PRSM Foundation
schultzryne@gmail.com / security@prsm.network
PGP: see SECURITY.md

---

### To: **Bishop Fox** (intake: <https://bishopfox.com/contact>)

**Subject:** RFP — PRSM bootstrap-node + protocol infrastructure pen-test ($20K-$50K, 2-4 weeks)

Hi Bishop Fox team,

I'm Ryne Schultz, founder of the PRSM Foundation. PRSM operates a
decentralized protocol on Base mainnet with a live bootstrap node at
`wss://bootstrap-us.prsm-network.com:8765`.

We're seeking a network/infrastructure pen-test covering bootstrap-
node availability, DHT poisoning resistance, storage-layer
availability, rate-limiting exhaustion, and TLS configuration. The
RFP includes a coordination clause for live-node testing.

**Full RFP:**
<https://github.com/Ryno2390/PRSM/blob/main/audits/rfp/L6f-infrastructure-pentest-rfp.md>

Budget: $20K-$50K. Target window: 2-4 weeks.

We are soliciting NCC Group and Doyensec in parallel.

Bishop Fox is on our shortlist for the firm's continuous-attack
methodology and depth in network-protocol assessments. Could we
schedule a scoping call?

Thanks,
Ryne Schultz
Founder, PRSM Foundation
schultzryne@gmail.com / security@prsm.network
PGP: see SECURITY.md

---

### To: **Doyensec** (intake: <https://doyensec.com/contact.html>)

**Subject:** RFP — PRSM bootstrap-node + protocol infrastructure pen-test ($20K-$50K, 2-4 weeks)

Hi Doyensec team,

I'm Ryne Schultz, founder of the PRSM Foundation. PRSM operates a
decentralized protocol on Base mainnet with a live bootstrap node at
`wss://bootstrap-us.prsm-network.com:8765`.

We're seeking a network/infrastructure pen-test covering bootstrap-
node DDoS resilience, transport-layer adapter security (Direct +
SOCKS), DHT poisoning resistance, and TLS configuration audit.

**Full RFP:**
<https://github.com/Ryno2390/PRSM/blob/main/audits/rfp/L6f-infrastructure-pentest-rfp.md>

Budget: $20K-$50K. Target window: 2-4 weeks.

We are soliciting NCC Group and Bishop Fox in parallel.

Doyensec is on our shortlist for the firm's specialization in
protocol security work. Could we schedule a scoping call?

Thanks,
Ryne Schultz
Founder, PRSM Foundation
schultzryne@gmail.com / security@prsm.network
PGP: see SECURITY.md

---

# §9 — Cadence + reply tracking

After sending: log each send into `audits/rfp/README.md` § "Status
update protocol" with date, recipient, and the engagement-letter
status. Suggested cadence:

- **Day 3:** light follow-up if no acknowledgment.
- **Day 7:** second follow-up if still no reply.
- **Day 14:** treat as no-response and proceed to the next firm in
  the recipient list (per the README's protocol).

When a proposal arrives, cross-reference budget + timeline against
the RFP envelope, then schedule the scoping call. Engagement letters
are signed at PRSM-POL-1 §5.1 disbursement authority (founder + 1
council signer on the Foundation Safe).

---

*Master plan: `audits/AUDIT_PLAN.md` v1.1*
*This packet: ready as of cumulative-audit-prep-20260505-i.*
