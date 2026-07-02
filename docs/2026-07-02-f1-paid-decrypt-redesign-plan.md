# F1 redesign plan — make the Tier B/C paid-decrypt gate binding

**Status:** DRAFT for owner approval. No code written. Blocks the ContentAccessVerifier deploy.
**Follows:** the B5 review (docs/2026-07-02-tier-bc-paid-decrypt-security-review.md).

## 1. The problem (why F1 is fundamental, not a patch)

The current design deposits the wrapped content key **in on-chain contract storage**
(`KeyDistribution.records[contentHash].encryptedKey`) and "releases" it by re-emitting it in an
event after payment. But **everything in contract storage is world-readable** via `eth_getStorageAt`
— the payment gate controls only the *event*, not the *bytes*. Since the wrapped key is sealed to
the buyer's X25519 pubkey, the one party who can decrypt it — the designated buyer — reads it
straight from storage for free and never pays. The paywall charges nobody who can use the content.

**Root cause (the principle):** *the releasable secret must never live in any public channel —
not on-chain storage, not an event, not an unauthenticated endpoint.* No amount of on-chain gating
fixes a secret that is already public.

Related confirmed findings this redesign should also resolve:
- **F9** (high): the fee routes to `ProvenanceRegistry.getCreatorAndRate(contentHash).creator` =
  first `registerContent` caller, decoupled from the publisher → squatter drains sales.
- **F10** (high): pay-but-can't-decrypt — no verification that the served key is *correct*, no
  recourse if the publisher serves garbage or withholds.
- **F2**: `depositKey` is permissionless + one-shot per `contentHash` → squatting bricks a paywall.

## 2. The core redesign (the part that fixes F1)

Move the secret **off-chain, behind a payment-gated endpoint**, and keep only a **commitment**
on-chain so the consumer can verify what they are served. This mirrors the existing §7 receipt-serve
data plane (`GET /settlement/receipt/leaf/{leaf_hash}`, sp1305).

**New flow:**

```
PUBLISH
  content_key      = AES-256-GCM key            (as today)
  ciphertext       = encrypt(plaintext, content_key)   → served FREELY (as today)
  wrapped_key      = seal(content_key, buyer_X25519_pub) (as today, B1)
  commitment       = keccak256(wrapped_key)     ← NEW: only this goes on-chain
  → register {contentHash, commitment, verifier, fee} on-chain (public, harmless)
  → the publisher's node RETAINS wrapped_key locally (never broadcast)

BUY
  1. consumer discovers the (publisher-signed) manifest via the content layer:
       {contentHash, commitment, verifierAddress, feeWei, publisher}
  2. consumer pays: ContentAccessVerifier.payForAccess(...) → records paid on-chain
  3. consumer requests the wrapped key from the publisher's node:
       GET /content/paid-key/{contentHash}  + a signature over a challenge by the consumer's ETH key
  4. the endpoint AUTHENTICATES the fetcher (recover signer) and GATES on-chain:
       require verifyPayment(signer, contentHash, feeWei) == true  → else 402
       then serves wrapped_key
  5. consumer verifies keccak256(wrapped_key) == commitment  ← defeats a lying publisher (F10 fraud)
  6. consumer unwraps with their X25519 privkey + AES-GCM-decrypts  (B1 reconstruct, unchanged)
```

**Why it now binds:** the wrapped key exists in serveable form ONLY on the publisher's node and is
released ONLY to a fetcher who (a) proves they are the payer (ETH signature) and (b) has an on-chain
`verifyPayment==true`. There is no public copy. The buyer cannot obtain it without paying.

## 3. Contract decision

Two viable shapes; **recommend Option A for v1** (smallest, reuses deployed contracts), with a
migration path to B if we want the anti-squat/escrow properties on-chain.

**Option A — reuse KeyDistribution + ContentAccessVerifier, deposit the COMMITMENT.**
- `KeyDistribution` treats `encryptedKey` as opaque bytes (verified: `EmptyEncryptedKey` is the only
  constraint). Deposit the 32-byte `commitment` there instead of the wrapped key. No contract change.
- `ContentAccessVerifier` (undeployed) unchanged for payment.
- All the new work is Python + one data-plane endpoint. **Deploys only ContentAccessVerifier** (as
  already planned) — nothing else new on-chain.
- Does NOT by itself fix F9/F2 (still `contentHash`-keyed, creator from ProvenanceRegistry). Mitigate
  off-chain: the consumer trusts the **publisher-signed manifest** (content advertisements are already
  signature-verified) and pays only against the deposit whose publisher matches the manifest signer.

**Option B — a purpose-built `ContentAccessRegistry` (supersedes standalone ContentAccessVerifier).**
- One contract keyed by `keccak(publisher, contentHash)` (fixes **F2** squat), storing
  `{commitment, feeWei}`; `payForAccess(publisher, contentHash)` credits **the deposit's publisher**
  (fixes **F9**), and optionally ESCROWS the fee with a timeout-refund (mitigates **F10** liveness).
- Since ContentAccessVerifier is not yet deployed, this is not a *net-new* contract — it replaces it.
- More design + audit, but fixes F1+F2+F9 (and partially F10) in one on-chain place.

**Recommendation:** ship **Option A** first (it fully fixes F1 — the critical — with minimal surface
and no new-contract risk), and treat **Option B** as a fast-follow if we want the squat/escrow fixes
enforced on-chain rather than by manifest-signature convention. F9/F2 in Option A are mitigated by
requiring the on-chain deposit's publisher to equal the signed-manifest publisher the consumer
discovered — a squatter's deposit simply won't match the authentic advertisement.

## 4. F10 (pay-but-can't-decrypt) — how far v1 goes

- **Wrong-key fraud (the exploitable part): FIXED** by the commitment check in step 5 — a publisher
  cannot serve a key that isn't the committed one without the consumer detecting it *before*
  trusting it. (Note: the consumer still paid; see below.)
- **Withholding after payment (liveness):** fair exchange is fundamentally unsolvable without a TTP,
  TEE, or zk-proof-of-delivery. v1 stance: (a) the commitment check means a paid consumer who
  *reaches* any honest holder gets the real key; (b) with Option B's escrow-timeout-refund, a
  consumer who never receives the key reclaims the fee after a window (the publisher must actively
  claim, so an offline publisher auto-refunds). Full dispute resolution (proving non-delivery) is
  explicitly OUT of scope for v1 and noted as a limitation.

## 5. Brick-by-brick change map

| Brick | Today | After redesign |
|-------|-------|----------------|
| B1 `paid_unlock` reconstruct | unchanged | unchanged — still unwrap+decrypt a wrapped key |
| B1 wrap | `wrap_content_key_for_deposit` → the deposited bytes | ADD `key_commitment(wrapped)=keccak256`; deposit the commitment, retain the wrapped key for serving |
| B2 `key_acquisition` | read the KeyReleased event to GET the key | REPLACE with: fetch the wrapped key from the payment-gated endpoint + verify `keccak==commitment` |
| B3 `pay_and_unlock` | settle → acquire(event) → retrieve ciphertext → reconstruct | settle → **fetch-key(endpoint,+sig)+verify-commitment** → retrieve ciphertext → reconstruct |
| B4 publisher `publish_paid_content` | deposit wrapped key | deposit **commitment**; retain wrapped key; register it for the gated serve |
| B4 node | — | NEW `GET /content/paid-key/{contentHash}` endpoint: authenticate fetcher sig → `verifyPayment` gate → serve retained wrapped key (mirrors sp1305 serve; PRSM_PAID_KEY_SERVE flag) |
| B4 client/SDK/CLI | fetch via event | fetch via endpoint; the CLI `content unlock` flow is otherwise unchanged (keys still from env) |
| ContentAccessVerifier | payment gate | unchanged (Option A) / folded into ContentAccessRegistry (Option B) |

Reused wholesale: the AES/X25519 crypto, ContentAccessVerifier payment + `verifyPayment`, the §7
serve-endpoint pattern, the signed content-advertisement machinery, the CLI/SDK surface shell.

## 6. Effort + sequencing (estimate)

1. **R1** — commitment helper + deposit the commitment (B1/B4 publisher change). Small.
2. **R2** — the payment-gated key-serve endpoint (`/content/paid-key/...`) with fetcher-signature
   auth + `verifyPayment` gate + retained-key store. Medium (mirrors sp1305).
3. **R3** — consumer key-fetch client + commitment verify; rewire B2/B3 to fetch-not-read-event.
   Medium.
4. **R4** — surface + end-to-end test (publish→pay→fetch-key→verify→decrypt) over a local EVM +
   fake serve. Small-medium.
5. **R5** — re-run the adversarial review on the redesigned flow before deploy. Small.
6. (Optional) **R6** — Option B `ContentAccessRegistry` for on-chain anti-squat/escrow (F9/F2/F10).
   Larger; only if we want those enforced on-chain.

The queued code-level review fixes (F3/F4/F12 fee-validation, F6 event-scan — now moot since we stop
reading the event, F5 deposit-order) fold into R1–R3.

## 7. Owner decisions needed

1. **Option A (reuse, ship F1 fix fast) vs Option B (new registry, on-chain F9/F2/F10).** Recommend
   A now, B as fast-follow.
2. **F10 appetite:** accept v1 (commitment-check fraud protection + optional escrow-refund; no full
   dispute resolution), or require on-chain escrow now (implies Option B).
3. **Deploy sequencing:** the F1 fix is Python + one endpoint + (Option A) the already-planned
   ContentAccessVerifier deploy. Confirm we still deploy ContentAccessVerifier, or wait for Option B.

Nothing here is built yet. On approval of Option A + the F10 stance, R1–R5 are autonomous
(contract deploy remains your gated ceremony); the re-review (R5) runs before any deploy.
