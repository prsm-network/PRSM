# Fable 5 Review — Domain 08: Core platform — crypto, auth, privacy, config & enterprise

Adversarial, cross-cutting review. See `00_INDEX.md` for shared framing + output format.

## Context

The foundational layer everything else depends on: cryptographic primitives (key generation,
signing, verification, encryption, the NodeIdentity = sha256(pubkey) used network-wide),
authentication/authorization, privacy mechanisms, config resolution, error handling, caching,
validation, monitoring, and third-party integrations. Plus the enterprise modules (TEE
policy, confidentiality mode, federated coordination, compliance/incident/insurance). A flaw
in a core primitive (a weak nonce, a verification that can be tricked, a key that leaks)
undermines every domain above it — so this is the highest-leverage place for a subtle bug.

### Read
- `prsm/core/cryptography/`, `prsm/core/auth/`, `prsm/core/security/`, `prsm/core/privacy/`
- `prsm/core/config/`, `prsm/core/errors/`, `prsm/core/validation/`, `prsm/core/caching/`,
  `prsm/core/monitoring/`, `prsm/core/integrations/`, `prsm/core/infrastructure/`
- `prsm/security/`, `prsm/observability/`
- `prsm/enterprise/` (TEE policy, confidentiality mode, federated coordination, compliance)
- The identity primitive (`prsm/node/identity.py`: `NodeIdentity`, `verify_signature`,
  `node_id_for_public_key`) — it's foundational even though it lives under `node/`

### Invariants — confirm or break
1. **Signing / verification soundness.** Every signature primitive verifies what it claims:
   the recovered/checked key is bound to the asserted identity (sha256(pubkey)==node_id),
   the signed bytes are the SAME canonical bytes on both sides (no sign-X-verify-Y), and a
   malformed/empty/None/wrong-curve signature is rejected (not silently accepted). Find a
   verification that can be tricked into returning success on unauthenticated input.
2. **Key + secret handling.** Private keys are never logged, persisted in plaintext, embedded
   in errors/receipts, or exposed via an endpoint; key material is zeroized/scoped where it
   matters; key generation uses a CSPRNG. Find a path where a secret escapes.
3. **Encryption correctness.** Symmetric encryption (AES-GCM etc.) uses unique nonces, verifies
   auth tags, and never reuses a nonce/key pair; asymmetric ops use vetted padding. Find a
   nonce-reuse, missing-tag-check, or rolled-your-own-crypto path.
4. **Config fails closed.** A missing/malformed security-relevant config value (auth key,
   trust anchor, policy floor, network selector) must fail CLOSED (disable the feature / refuse
   to start), never fail OPEN (silently weaken a control). Find an env-var or config path that
   silently downgrades security on a bad value.
5. **Authz consistency.** The authorization model is consistent across entry points (no
   endpoint/path that skips the check another enforces); privilege boundaries (operator vs
   admin vs Foundation) are crisp and enforced at the right layer.
6. **Enterprise confidentiality / TEE policy.** The enterprise confidentiality mode + TEE
   policy evaluation are consistent with the runtime attestation gate (Domain 01) — a policy
   that says "require hardware-verified" actually maps to `vendor_verified==true`, and the
   enterprise path can't be configured to accept a weaker tier than it advertises.

### Hunt list
- Custom crypto / hand-rolled constructions where a vetted library should be used.
- `==` instead of constant-time compare on secrets/MACs; timing side-channels on auth.
- A `try/except` that swallows a verification failure into a success/`True`.
- Global mutable security state (a default-permissive singleton that a caller can mutate).
- Logging of full request/response objects that include secrets; debug flags that disable auth.
- Dependency / integration trust: a third-party client that's trusted with more than it should
  be; deserialization of untrusted data in an integration.
- Randomness: `random` (non-CSPRNG) used for anything security-relevant (nonces, tokens, ids).

Follow the `00_INDEX.md` output format. Report only.
