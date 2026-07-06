# PRSM Lighthouse Design-Partner Pilot Kit

**Goal:** land 1–3 real external users with a real workload, and get them to a *repeated, real-value*
transaction on mainnet. One lab running private inference (and paying for it) is the sentence that
unlocks the audit budget, operator recruitment, liquidity, and any raise.

**Why now:** the product is mainnet-proven (trustless multi-stage settlement, verifiable receipts,
paid data + paid content all live) but has **zero external pull**. A pilot is the forcing function
that turns "it works" into "someone depends on it." This kit is everything needed except the
relationships — the assistant builds/operates the mechanics; you bring the humans.

---

## 1. Ideal design-partner profile

Three archetypes, each matching a proven PRSM capability. Start with the lowest-friction that you
have a warm relationship into.

| # | Archetype | Pilot workload | Value story | Friction |
|---|---|---|---|---|
| **A** | Researcher / data owner | Publish a proprietary dataset with verifiable provenance + creator attribution; or find + fetch verified datasets (`content publish-paid` / `content get` / `find_and_fetch`) | "Monetize/share data with cryptographic provenance; buyers get provably-intact, provably-attributed bytes" | Low–med |
| **B** | Privacy-sensitive inference (health / finance / legal) | Inference on sensitive data through the SEV-SNP **TEE** node with hardware attestation | "Run inference on a host you don't trust; the receipt proves it ran in a genuine TEE, node-bound" | **High** (marquee) |
| **C** | Cost / verifiability-sensitive compute (indie AI dev, small ML shop) | Paid big-model inference with a §7 verifiable receipt (`pay_and_infer` / `pay_and_infer_multistage`) | "Cheaper big-model inference with a receipt you can independently verify" | **Low** (fastest) |

**Recommended sequence:** land **C first** (fastest activation, proves the paid-compute loop with a
real outsider), use **A** for the clearest standalone value narrative, and hold **B** as the marquee
case study once the mechanics are smooth — it's the highest-value story but the highest-touch.

**What makes a *good* partner (screen for these):**
- Crypto-comfortable — can hold a wallet, handle a small amount of FTNS, sign a tx.
- A **recurring** workload, not a one-off (you're testing depth, not a demo).
- High-touch / reachable — you can talk to them weekly and watch them use it.
- Tolerant of rough edges — an early-adopter temperament.
- Ideally a **referenceable name or story** (a lab, a known indie dev, a recognizable use case).

---

## 2. Outreach

Principles: lead with the proof, name one specific value, ask for one tiny concrete first step, keep
it under 120 words. Not "check out our platform" — "run one verified inference this week."

**Template C (compute):**
> Subject: verifiable big-model inference — 15 min?
>
> We built PRSM: run large-model inference across independent GPU nodes and get a cryptographic
> receipt you can verify yourself — it's live and settling real payments on Base mainnet. I think it
> fits [their workload]. Could I get you running one paid, verified inference this week? I'll hand
> you a burner wallet + testnet credits so it costs nothing to try, and sit with you while you do it.

**Template A (data):**
> Subject: monetize [dataset] with provable provenance
>
> PRSM lets you publish a dataset so any consumer gets bytes that are provably intact AND provably
> from you (on-chain creator attribution), and pays you per access — live on Base mainnet. Want to
> put [dataset] up as a pilot? I'll walk you through publish → a test buyer unlock → the payment
> landing in your wallet, start to finish, this week.

**Template B (private inference):**
> Subject: inference on sensitive data, on a host you don't trust
>
> PRSM can run your inference inside a hardware TEE (AMD SEV-SNP) and hand back an attestation that
> proves it ran in a genuine enclave, bound to that node — so a compute provider can't see your data
> or fake the result. Hardware-validated and live. Would [their sensitive workload] be a fit for a
> bounded pilot? I'll run the whole confidential path with you.

---

## 3. Onboarding runbook (the partner's happy path)

Same shape for all three: **onboard cost-free on testnet → do the real thing on mainnet at bounded
value → verify → repeat.** The assistant sits with the partner for the first run.

**Step 0 — install (2 min):**
```bash
pip install prsm-network
```

**Step 1 — wallet + credits, cost-free (testnet):**
```bash
prsm join-testnet          # fresh burner wallet + onboarding
prsm wallet faucet         # testnet FTNS (on-chain)
prsm wallet balance        # confirm the on-chain FTNS landed
```

**Step 2 — do the workload (testnet first, then a small mainnet repeat):**
- **C — compute:** SDK `pay_and_infer(...)` (or `pay_and_infer_multistage` for a too-big model) → returns output + a §7 receipt.
- **A — data:** publish with `prsm content publish-paid ...`; a test buyer runs `prsm content get "<query>"` / SDK `find_and_fetch(query, verify_provenance=True)` → verified bytes + attribution; or `prsm content unlock` for paywalled content.
- **B — private inference:** route inference through the TEE node; capture the attestation in the receipt.

**Step 3 — verify (the whole point):**
- Compute: independently re-verify the receipt's signatures + activation-hash chain.
- Data: confirm `sha256(bytes) == content_hash` and that provenance resolves on-chain to the claimed creator (the client checks the ProvenanceRegistry itself).
- Private: confirm `vendor_verified == true` and the attestation is node-bound.

**Step 4 — settle on mainnet, bounded (the real value):**
Repeat Step 2 on `PRSM_NETWORK=mainnet` with a small real amount; confirm the on-chain settlement
(deposit → commit → finalize for compute; creator credited → claim for data). This is the moment the
pilot becomes real.

---

## 4. Bounded-value safety rails (pre-audit)

The money contracts have had multiple internal adversarial reviews and a live mainnet proof, but
**not yet a third-party audit** (started in parallel — see the initiatives spec). So run every pilot
bounded and honest:

- **Testnet-first, always.** Only move to mainnet after the partner has done the full loop on testnet.
- **Cap value.** Small per-transaction and total-pilot ceilings; nothing a partner can't afford to lose.
- **Informed partners only.** Disclose up front: early software, unaudited contracts, here's the
  honest risk and the honest trust model (link the trust/threat-model doc when it exists).
- **No custody.** You never hold partner funds; escrow is on-chain and failures auto-refund.
- **Prefer non-critical workloads** for the first mainnet runs; graduate to real workloads as trust builds.

---

## 5. Success metrics

Define the bar *before* starting so you know if it worked.

| Metric | Target for "pilot success" |
|---|---|
| **Activation** | Partner completes the full verify-and-settle loop on mainnet, unaided, once |
| **Depth** | ≥3 real transactions over a 2–4 week window (repeat, not a one-off) |
| **Value** | Any non-trivial real FTNS settled end to end (deposit → finalize / creator paid) |
| **Qualitative** | "Would you keep using this?" = yes, + a concrete reason; + the top-3 things that broke |
| **Reference** | Partner agrees to be named / quoted (even softly) |

A pilot that hits Activation + Depth + a usable quote is a fundable reference case, even at tiny value.

---

## 6. Feedback loop + timeline

**Cadence:** weekly 30-min check-in per partner; a shared issue list; one standing question — *"what
would make you keep using this?"* Every friction point becomes a code issue or a doc fix (route real
ones back into the repo; the operator/consumer surfaces are already deep, so most gaps will be
onboarding-shaped).

**5-week structure:**
- **Week 0** — screen + onboard (Steps 0–3 on testnet, assisted).
- **Weeks 1–4** — real usage on mainnet at bounded value; weekly check-ins; log + fix friction.
- **Week 5** — review against the success bar; capture the reference story; decide graduate vs. iterate.

---

## 7. Division of labor

**The assistant builds/operates:** this kit, the flagship demo (if wanted), any onboarding scripts a
partner needs, the per-partner runbook tailoring, the verification walkthroughs, and every code/doc
fix that pilot friction surfaces — plus it can drive the seed operator nodes that give a pilot real
capacity to run against.

**You bring:** the relationships and the conversations, the choice of first partner, any pilot
value/credit decisions, and the "yes" from the partner. Bring one warm intro and we can start Week 0
immediately.
