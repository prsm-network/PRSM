# Tier B/C paid-decrypt — adversarial security review (B5)

**Date:** 2026-07-02 · **Scope:** sp1349–1355 (paid_unlock, key_acquisition, paid_content,
content_access_verifier client, SDK/CLI surface) + `ContentAccessVerifier.sol` and its
`KeyDistribution.sol` / `ProvenanceRegistry.sol` interaction.
**Method:** 6 dimension finders → each finding adversarially verified by 3 independent
diverse-lens skeptics (exploitability / code-correctness / reproduce); only findings ≥2/3 REAL are
kept. Workflow `wpre3ev0y` (96 agents). **30 raw → 12 confirmed, 18 dismissed.**

## ★ CRITICAL — the on-chain release gate is non-binding (gates the deploy)

**F1.** `KeyDistribution.records` is `public` and `KeyRecord.encryptedKey` is stored in contract
storage (`contracts/contracts/KeyDistribution.sol:76,86`). Anything in contract storage is readable
via `eth_getStorageAt` (and the auto-getter for the struct). The deposited blob is the content key
**sealed to the buyer's X25519 pubkey**, and the ciphertext is served FREELY. So the *only party who
can decrypt it* — the designated buyer — can read the wrapped key straight from storage and decrypt,
**without ever calling `payForAccess`/`release`**. The payment gate (`verifyPayment` + `release`)
guards only the convenience `KeyReleased` *event*, not access to the secret. **The paywall is
economically non-binding for exactly the party it is meant to charge.**

**Implication:** the deploy must NOT proceed as-designed. The fix is a redesign of key *delivery*:
store only a commitment/hash on-chain, and deliver the actual wrapped key **off-chain via a
payment-gated data-plane endpoint** that checks `verifyPayment` (or TEE-gated release) before
serving — the §7 receipt-serve pattern. This is a design decision for the owner.

## HIGH — economic integrity

- **F9. Creator-squatting redirects 100% of fees.** The fee is credited to
  `ProvenanceRegistry.getCreatorAndRate(contentHash).creator` = whoever called `registerContent`
  first, which is DECOUPLED from the KeyDistribution `publisher`. `contentHash` is public
  (`sha256(freely-served ciphertext)`), so a squatter front-runs `registerContent`, becomes the
  on-chain creator, and drains every sale via `claim()`. **Fix:** bind the fee recipient to the
  KeyDistribution publisher (cross-check `creator == publisher` at deposit/register, or read the
  payee from the deposit).
- **F10. Pay-but-cannot-decrypt, no recourse.** `payForAccess` credits the creator immediately and
  irreversibly; there is no escrow/refund. A malicious publisher can `deauthorize` after payment,
  wrap to an X25519 key the payer doesn't hold, or deposit a key that doesn't decrypt the served
  ciphertext. **Fix:** escrow the fee against a proven `KeyReleased` / challenge window; bind the
  decrypt (X25519) identity to the paying address; block `deauthorize` while paid-but-unreleased.

## MEDIUM / LOW — code-level (in-arc, fixable now)

- **F3/F4/F12. Fee not validated against the deposit.** `payForAccess`/`pay_for_access` pull a
  *caller-supplied* `feeWei`; `release` checks the deposit's exact `releaseFeeFtnsWei`. A mismatch
  pulls + credits the fee but never unlocks, with no refund. **Fix:** drive both payment and the
  release check off the on-chain deposit fee; refuse to pay a mismatched fee client-side.
- **F8/F11. Double-charge on retry (no idempotency).** `payForAccess` pulls + credits on *every*
  call (no `if (paid[...]) return;`), and the client always calls `settle_fee` without a
  verify-before-pay. A benign retry (RPC lag, unsaved plaintext) double-charges. **Fix:** contract
  short-circuit when already paid + verify-before-pay in the settle path. → **FIXED (sp1356).**
- **F6. `acquire_released_key` scans from block 0, un-chunked, un-filtered.** Breaks on mainnet
  (eth_getLogs cap) and leaks a raw RPC exception (violates fail-loud). **Fix:** `argument_filters`
  on contentHash+recipient; default `from_block` to the deposit block + chunk; wrap errors.
- **F7. chain-id not pinned.** Clients sign with `web3.eth.chain_id` (RPC-reported); the
  authoritative `ep.chain_id` is discarded. A hostile/misconfigured RPC can get a signature bound to
  the wrong chain. **Fix:** thread + assert `expected_chain_id`. → **FIXED (sp1356).**
- **F5. Deposit-before-serve strands paid buyers** if `publish_ciphertext` fails after the deposit
  (and the docstring rationale is inverted — orphan ciphertext is harmless). **Fix:** serve first,
  or compensating `deauthorize` + loud partial-failure on publish failure.
- **F2. `depositKey` is permissionless + one-shot** → contentHash squatting bricks a paywall.
  (`KeyDistribution.sol`, deployed.) **Fix:** key records by `keccak(msg.sender, contentHash)` or
  gate to registered creators.

## Disposition

- **F1 / F9 / F10 / F2** — architectural and/or touch the deployed `KeyDistribution` /
  `ProvenanceRegistry`. **Surfaced to the owner; the deploy is gated on the F1 redesign decision.**
- **F8/F11 (double-charge), F7 (chain-id)** — **fixed in sp1356** (zero-regret, survive any redesign).
- **F3/F4/F12 (fee-validation), F6 (event scan), F5 (deposit order)** — code-level, fixable; queued
  pending the F1 redesign direction (some touch the to-be-redesigned delivery path).

18 findings were dismissed by the skeptics (plausible-but-not-real): e.g. SafeERC20/fee-on-transfer
hardening (FTNS is a standard OZ ERC20), envelope version/AAD hardening, and several
already-guarded paths.
