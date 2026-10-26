# L4 Code4rena Contest — Scoping Packet

**Engagement format:** Public competitive audit (Code4rena standard)
**Issuing organization:** PRSM Foundation (Cayman Islands nonprofit)
**Issued:** 2026-05-05
**Target submission window:** 2026-05-12 to 2026-05-19
**Target contest start:** 2026-06-09 (or first available C4 window)
**Target contest duration:** 14 days
**Prize pool target:** $40,000 USD (split per C4 standard distribution)

**Primary contact:** schultzryne@gmail.com / security@prsm.network
**PGP:** see SECURITY.md
**Repository:** https://github.com/Ryno2390/PRSM

---

## 1. Contest summary

PRSM is a decentralized inference + storage + royalty protocol on Base
mainnet. This contest covers the Phase 3.1 batched-settlement bundle plus
adjacent contracts that handle real fund flow:

- 9 in-scope contracts, ~2,300 production LoC
- Base mainnet (chainId 8453), Solidity 0.8.22 + OZ 5.x patterns
- Already passed: 488 hardhat tests; L1 static-tooling (Slither + Aderyn)
  clean; L2 AI multi-team adversarial review (4 teams) closed with 1 CRIT
  + 7 HIGH + 7 MEDIUM all remediated
- L3 cryptographic specialist engagement (Ed25519Lib + Sha512) running in
  parallel with this contest — those files are OUT OF SCOPE here
- Foundation Safe (`0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791`, 2-of-3 with
  Ledger + Trezor + OneKey diversity) deployed 2026-05-04 on Base mainnet

Wardens should expect a clean, well-documented codebase with most surface
hazards already closed. The hard problems live in cross-contract
composition, economic edge cases, and adversarial sequencing — not in
basic SAST patterns.

---

## 2. In-scope contracts

| File | LoC | Purpose |
|------|-----|---------|
| `contracts/EscrowPool.sol` | ~220 | Per-requester FTNS escrow; settlement target |
| `contracts/StakeBond.sol` | ~480 | Provider stake / unbond / slash lifecycle |
| `contracts/BatchSettlementRegistry.sol` | ~890 | Batched receipt commit + challenge + finalize |
| `contracts/RoyaltyDistributor.sol` | ~125 | 3-way pull-payment splitter (creator / treasury / node) |
| `contracts/FTNSTokenSimple.sol` | ~190 | UUPS-upgradeable ERC20 with role-based mint/pause |
| `contracts/EmissionController.sol` | ~270 | Halving emission schedule + minter |
| `contracts/CompensationDistributor.sol` | ~290 | Pull-based reward distributor |
| `contracts/StorageSlashing.sol` | ~280 | Storage-proof slasher (Phase 7-storage) |
| `contracts/KeyDistribution.sol` | ~220 | Threshold key distribution + heartbeat |

**Total: ~2,965 LoC. SLOC after removing comments + blanks: ~2,300.**

External dependencies (in-scope only insofar as they are USED — no review
of OZ itself required):

- `@openzeppelin/contracts@5.x`: Ownable2Step, ReentrancyGuard, Pausable,
  AccessControl, SafeERC20-equivalent (we use bool-returning ERC20 with
  explicit checks — note this in your review)
- `@openzeppelin/contracts-upgradeable@5.x`: UUPSUpgradeable,
  PausableUpgradeable, AccessControlUpgradeable, OwnableUpgradeable,
  Initializable

---

## 3. Out of scope

| File / Path | Why out of scope |
|-------------|------------------|
| `contracts/lib/Ed25519Lib.sol` (887 LoC) | L3 cryptographic specialist engagement (concurrent with this contest) — see `audits/rfp/L3-ed25519-crypto-rfp.md` |
| `contracts/lib/Sha512.sol` (328 LoC) | Same as above |
| `contracts/Ed25519Verifier.sol` | Wraps Ed25519Lib; covered by L3 |
| `contracts/test/*` | Test-only contracts (mocks, fixtures) |
| `contracts/test/MockSignatureVerifier.sol` | Test-only; not deployed |
| Off-chain Python (`prsm/`) | L5 off-chain ML supply-chain audit |
| Frontend / npm packages | Not deployed on-chain |
| Deployment scripts (`contracts/scripts/*`) | Operational tooling; reviewers may flag obvious issues but findings on script-level bugs will be classified MEDIUM at most |

If during the contest a warden identifies a defect in an out-of-scope file
that DIRECTLY enables an in-scope vulnerability, please submit it — we
will judge on the in-scope impact.

---

## 4. Prior-audit context

Wardens should know what prior layers have already covered, so they can
focus time on what's NEW.

### 4.1 Already remediated (don't re-find these)

**1 CRITICAL + 7 HIGH from L2 multi-team review (closed 2026-05-05):**

- CRIT-1 (C-INT-01): adversarial slashing via unbound `signingMessage` —
  fixed by binding `signingMessageHash` in `ReceiptLeaf` struct
- HIGH-1 (A-01): RoyaltyDistributor split deviates from PRSM-TOK-1 §8.1 —
  remediated by v2 redeploy (canonical RoyaltyDistributor v2
  `0xfEa9aeB99e02FDb799E2Df3C9195Dc4e5323df7e`), which adds Pausable + the
  D-04 pull-payment refactor. (The 20% burn-on-use was subsequently dropped
  from the design; v2 has no burn-on-use — `burnFrom` exists for the bridge
  only.)
- HIGH-2 (A-02 / D-01): slash-evasion race when
  `unbondDelay < challengeWindow` — closed via cross-contract invariant in
  `StakeBond.requestUnbond`
- HIGH-3 (D-02): treasury contracts had no pause surface — OZ Pausable
  added across audit-bundle
- HIGH-4 (B-VERIF-1): deploy-verification script called wrong getter
  names — fixed
- HIGH-5 (B-RENOUNCE-1): admin renounceRole could brick FTNSTokenSimple —
  closed by override blocking DEFAULT_ADMIN_ROLE renounce
- HIGH-6 (B-CROSS-1): `EscrowPool.settlementRegistry` re-pointable post-
  handoff — made immutable
- HIGH-7 (B-CROSS-3): `StakeBond.slasher` re-pointable post-handoff —
  made immutable

**7 MEDIUM (closed 2026-05-05):**

- B-TREASURY-1: pinned canonical Foundation Safe in deploy script
- C-INT-02: ISignatureVerifier.verify changed view → pure
- D-05: per-batch snapshot of `challengeWindowSeconds` (no retroactive
  shortening)
- B-OWNABLE-1: migrated all 7 Ownable contracts to Ownable2Step
- B-CROSS-2: `EscrowPool.setFtnsToken` enforces `totalEscrowedBalance == 0`
- D-03: per-batch snapshot of escrowPool/stakeBond/signatureVerifier
- D-04: RoyaltyDistributor pull-payment refactor

Full L2 audit report: `audits/findings/consolidated.md`
PoC suites: `contracts/test/audit-team-{a,b,c,d}/*.test.js`
(Each PoC was inverted post-fix into a REGRESSION test that asserts the
fix holds.)

### 4.2 Known issues / explicit non-bugs

**Don't submit findings on these — they're documented design choices:**

1. **`block.timestamp` for unbond delays.** ~7-day windows; miner
   deviation is bounded at ~15s on Base, irrelevant at this granularity.
2. **bool-returning IERC20 transfer/transferFrom.** We do explicit `if
   (!ok) revert TransferFailed()` checks; the canonical FTNS token is a
   conforming ERC20. We do NOT use SafeERC20 because the
   pre-deployed FTNS conforms and we want to avoid the OZ wrapper's
   gas overhead on the hot path. Submitter must demonstrate a real
   fund-loss path on a non-conforming token to qualify as a finding.
3. **Foundation owner is a 2-of-3 multi-sig (no DAO).** Governance is
   intentionally centralized at v1; transition to DAO is roadmap, not
   in-scope.
4. **`RoyaltyDistributor` v1 mainnet deploy uses push-payment.** v2
   redeploy ships the D-04 pull-payment fix (live canonical v2 is
   `0xfEa9aeB99e02FDb799E2Df3C9195Dc4e5323df7e`; the originally-planned
   20% burn-on-use was dropped — v2 has no burn-on-use). Contest-time
   source is v2.
5. **`EmissionController` halving uses right-shift.** Intentional;
   constant-time + gas-cheap.
6. **No timelock on the remaining mutable governance setters.** D-03 is
   solved via per-batch snapshot of the load-bearing cross-wires; the
   remaining setters (Registry.setEscrowPool, Registry.setStakeBond,
   Registry.setSignatureVerifier, StakeBond.setUnbondDelay,
   StakeBond.setFoundationReserveWallet, EscrowPool.setFtnsToken) only
   affect FUTURE batches. Findings claiming "owner can soft-brick
   in-flight batches" must demonstrate a path AROUND the per-batch
   snapshot.
7. **Mainnet v1 FTNSTokenSimple admin = hot key (not multi-sig yet).**
   Admin handoff to Foundation Safe is queued (CRIT-2 from L2 audit; not
   in this contest's scope because the source-side fix is operational,
   not a code change).

### 4.3 Most-promising attack surfaces (what to dig into)

This is the L2 + L3 + author's combined intuition for where wardens
should spend time:

1. **Cross-contract sequencing in BatchSettlementRegistry** — commit /
   challenge / finalize / slash interleaving across multiple batches,
   especially with `consensus_group_id` introduced in §8.7 sybil fix.
2. **Speculation in CompensationDistributor weighted split** — weighted
   pull-payment under different timing-of-claim sequences.
3. **EmissionController + minter authorization** — halving boundary
   conditions, mint cap enforcement at the slot transitions.
4. **StorageSlashing heartbeat + proof-failure interactions** —
   griefing via repeated heartbeat-miss patterns; bounty competition
   between concurrent challengers.
5. **KeyDistribution threshold + heartbeat** — boundary cases in the
   Shamir-style key reassembly + the heartbeat interaction.
6. **Reentrancy across EscrowPool.settle → ERC20.transfer → external
   recipient.** The receiver could be a contract; we use OZ
   ReentrancyGuard but composability bugs may exist.
7. **Edge cases in `RoyaltyDistributor.claim()` post-distribute-mid-
   ownership-transfer** (D-04 fix is recent; verify it doesn't introduce
   new bugs).
8. **Phase 7.1 multi-shard consensus dispatcher + CONSENSUS_MISMATCH
   reason code** — adversarial routing of consensus messages.

---

## 5. Threat model

**Attacker capabilities to assume:**

1. Full knowledge of all on-chain code (verifier-published).
2. Ability to deploy adversarial contracts on Base.
3. Ability to time transactions to specific blocks (within ~12s of
   miner control).
4. Ability to control multiple addresses / Sybil identities.
5. Ability to run multiple validators / providers / requesters
   simultaneously.
6. Ability to bribe at most 1 of the 3 Foundation multi-sig signers
   (single-signer compromise).
7. Ability to monitor mempool + frontrun submissions.

**Attacker capabilities to NOT assume:**

1. Compromise of 2+ Foundation Safe signers (out of contest scope).
2. Compromise of the FTNS token contract itself.
3. Compromise of L2 sequencer or chain reorgs > 1 block.
4. Off-chain key-material exfiltration.
5. Ed25519Lib / Sha512 cryptographic breaks (L3 scope).

---

## 6. Setup instructions

**Repository:** https://github.com/Ryno2390/PRSM
**Branch / tag at contest start:** `audit-prep-l4-contest-20260609` (will
be tagged at contest start)

```bash
# 1. Clone
git clone https://github.com/Ryno2390/PRSM.git
cd PRSM

# 2. Solidity setup
cd contracts
npm ci
npx hardhat compile     # Should succeed; uses Solidity 0.8.22 + viaIR

# 3. Run tests (488 should pass)
npx hardhat test

# 4. Run L1 static analysis locally (optional but recommended)
pip install slither-analyzer
slither . --config-file slither.config.json

# 5. (optional) Deploy to local hardhat node
npx hardhat node               # in one terminal
npx hardhat run scripts/deploy-audit-bundle.js --network localhost
```

**Reading order suggestion:**

1. `audits/AUDIT_PLAN.md` (master 11-layer plan)
2. `audits/findings/consolidated.md` (L2 multi-team review summary)
3. `contracts/contracts/BatchSettlementRegistry.sol` (the heart of the
   protocol)
4. `contracts/contracts/EscrowPool.sol` + `StakeBond.sol` (cross-wired
   counterparts)
5. `contracts/contracts/RoyaltyDistributor.sol` +
   `EmissionController.sol` + `CompensationDistributor.sol` (treasury
   layer)
6. `contracts/contracts/FTNSTokenSimple.sol` (token; UUPS upgrade
   surface)
7. `contracts/contracts/StorageSlashing.sol` +
   `KeyDistribution.sol` (Phase 7-storage)

---

## 7. Severity classification

We use the standard Code4rena severity model. For PRSM-specific
calibration:

- **CRITICAL:** Direct + provable theft of any FTNS, or permanent locking
  of any user balance, or ability to forge slashing against an honest
  provider.
- **HIGH:** Indirect or conditional fund loss; griefing that breaks the
  protocol's core promise (pay → settle); auth bypass on protected
  functions.
- **MEDIUM:** Operational disruption recoverable via governance action;
  edge cases that cost extra gas but not funds.
- **LOW:** Code quality / readability / minor optimization. Wardens
  should know that L1 SAST is wired in CI — duplicating Slither/Aderyn
  findings will not score.
- **INFORMATIONAL:** Style / typo / suggestion.

---

## 8. Prize pool

**Target pool:** $40,000 USD, distributed per Code4rena standard:

- ~80% to High/Med findings, weighted by severity + uniqueness
- ~20% to Judging / Q&A grant
- Custom bonuses for the first warden to find any CRITICAL (+$2K bonus)

We are open to Code4rena's recommendation if standard distribution
differs from above.

---

## 9. Engagement terms

- **Public report:** Yes — publish on Code4rena reports page after
  Foundation has remediated all High+ findings.
- **License of findings:** Wardens retain disclosure rights post-
  publication. Foundation publishes the consolidated report.
- **Embargo:** Standard Code4rena embargo (typically 30 days post-
  contest for fixes).
- **Bug bounty hand-off:** An Immunefi bounty (L9 layer of audit plan)
  was proposed to follow this contest for ongoing protection. It was
  not launched; PRSM does not run a bug bounty program.

---

## 10. Foundation contacts during contest

- **Lead reviewer / Q&A respondent:** founder (schultzryne@gmail.com)
- **Backup:** deputy-founder (per `docs/security/D7_DEPUTY_FOUNDER_SUCCESSION.md`)
- **Discord / Slack:** Code4rena's standard Discord channel for the
  contest
- **Response SLA:** ≤24h on weekdays for warden questions during the
  contest window

---

## 11. Complementary firm engagement (informational)

We are running this Code4rena contest in parallel with a private firm
engagement targeting cross-contract composition. The two reviews are
designed to find different bug classes:

- **This contest (L4 public):** broad crowd, creative findings, ~$40K
- **Firm pair-review (L4 private):** Trail of Bits / Spearbit /
  Cantina elite, structured methodology, ~$60K, scoped at integration
  surfaces

Combined L4 budget: ~$100K. Wardens should not feel they're competing
with a firm — the firm's scope intentionally avoids per-contract logic
that wardens are best positioned to find.

See `docs/2026-04-23-auditor-shortlist-and-rfp.md` for the firm RFP.

---

## 12. Submission path

To open a contest scoping conversation with Code4rena:

1. Email Code4rena's intake (typically `intake@code4rena.com`) with
   this packet attached.
2. They schedule a 30-min scoping call within ~3-5 business days.
3. After call, they propose a contest window + final pool size; we
   confirm / counter.
4. Contest starts on the agreed date; wardens have 14 days.
5. Judging period (1-2 weeks) finalizes findings.
6. Foundation triages + remediates.
7. Public report published.

**Total wall-clock from intake to published report: 6-8 weeks.**

---

## 13. Companion documents

- `audits/AUDIT_PLAN.md` — master 11-layer plan (where L4 sits)
- `audits/findings/consolidated.md` — L2 prior-audit context
- `audits/decisions/L3-ed25519-decision.md` — why crypto lib is
  out-of-scope here
- `audits/L1-static-tooling/README.md` — what L1 SAST has already
  caught (don't re-find)
- `docs/2026-04-23-auditor-shortlist-and-rfp.md` — firm-track RFP
- `SECURITY.md` — vulnerability disclosure policy + PGP key
- `docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md` — incident response
  posture

---

*Foundation reserves the right to adjust contest scope based on
Code4rena's intake recommendations.*
