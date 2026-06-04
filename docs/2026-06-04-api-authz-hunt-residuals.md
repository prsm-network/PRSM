# API authorization hunt — fixes and residual decisions

**Sprints 1011–1012.** The API-authz adversarial hunt (workflow `wt1tb4n3q`,
5 dimensions: admin-loopback-bypass / unauth-sensitive-mutation /
missing-authz-IDOR / info-leak / auth-mechanism, default-refute + reachability)
confirmed 11 findings about whether an unauthenticated or unauthorized caller can
bypass the gates, hit a sensitive mutation, act on another user's resource, or
read secrets. The systemic ones are fixed; this records the residuals.

## The two gates (context)

- **`/admin/*` loopback middleware** (sprint 734, api.py) — restricts admin paths
  to loopback clients. Uses an exact-match `_GATED_PATHS` tuple for the handful
  of non-`/admin/` sensitive paths.
- **`NodeAuthMiddleware.PROTECTED_PREFIXES`** (auth_middleware.py) — requires the
  operator's `PRSM_NODE_API_KEY` for sensitive prefixes, **but only when the key
  is set** (dev mode = no key = no enforcement).

## Shipped

- **sp1011 (finding 5, HIGH — the keystone):** the public-bind + no-key posture
  is now **fail-closed by default** (`should_refuse_insecure_public_bind`). A
  non-loopback bind with no `PRSM_NODE_API_KEY` refuses to start unless the
  operator acks `PRSM_ALLOW_INSECURE_PUBLIC_BIND=1`. This closes the entire
  "money + KYC endpoints unauthenticated because the operator forgot the key"
  class — the precondition for remotely exploiting nearly every other finding.
  The CLI binds loopback by default, so the common case is unaffected.
- **sp1012 (findings 1/3/4/7/10/11):** registered the endpoints that fell
  through `PROTECTED_PREFIXES` even with a key set —
  `/marketplace/creator-reputation/`, `/peers/connect`, `/billing/` (prefixes)
  and `/content/{cid}/pin` (a templated `PROTECTED_PATH_PATTERNS` entry, since a
  bare `/content/` prefix would wrongly gate public reads). Single-sourced in
  `is_protected_path()`.
- **Findings 6/8/9 (KYC / WaaS / balance reads):** already covered — those paths
  are under the existing `/wallet/` prefix, so they require the key when one is
  set, and sp1011 forces a key on any public bind. No code change needed.

## ★ RESIDUAL A — wallet-onboarding reads need ownership proof, not the operator key (finding 2)

`GET /api/v1/auth/wallet/{binding,bindings,devices/earnings,balance}` expose a
wallet→node binding, the device roster, and per-device earnings, keyed solely on
a `wallet_address` query param with no authorization (read-only IDOR). They are
NOT under `/wallet/` (they're under `/api/v1/auth/wallet/`), so the existing
prefix doesn't cover them. **But the correct gate is not the operator's API key**
— these endpoints are meant to be called by the *wallet owner* (a user), who does
not hold the operator key; gating them behind it would break the onboarding flow.
Severity is low (read-only; `balance` returns 0 in the wired daemon — only the
binding/earnings metadata leaks).

**Recommendation (decision):** require proof-of-ownership of `wallet_address` on
the read path — a SIWE-session token or a fresh EIP-191 signature — mirroring the
existing onboarding signature model (the mutations `/siwe/*`, `/bind` are already
EIP-191/4361-gated). A caller could then read only bindings/earnings for a wallet
it controls. This is a per-endpoint auth-model addition (a SIWE-session
dependency), best done as a focused follow-on rather than mis-gated behind the
operator key.

## ★ RESIDUAL B — per-resource authorization (IDOR) on keyed endpoints (findings 6, 7, 9 IDOR aspect)

Several endpoints authenticate (or are now key-gated) but do not AUTHORIZE the
specific resource: `/billing/{job_id}`, `/wallet/kyc/{user_id}`,
`/wallet/balance/{user_id}` operate on whatever id the caller supplies without
checking it belongs to the caller. sp1011/sp1012 close the *anonymous-remote*
access (the operator key is now required), so the residual is a *cross-resource*
read by an already-authenticated caller — relevant for multi-tenant operators.

**Recommendation:** add an owner/tenant check at each handler (the authenticated
principal must match the resource's owner), or scope these reads to the operator
(single-tenant) explicitly. Lower priority than the anonymous-access closure;
tracked for the multi-tenant hardening pass.

## Not-a-bug / backstopped (recorded for audit)

- **admin-loopback XFF spoof** — the hunt probed the `_is_loopback` /
  X-Forwarded-For handling for a remote spoof and the refuted analysis held: the
  middleware does not trust a remote-injected forwarded header to manufacture a
  loopback verdict on the default posture. (The `/peers/connect` and PII-read
  findings were path-COVERAGE gaps, now fixed, not loopback-logic bypasses.)
- **auth mechanism** — `NodeAuthMiddleware` compares the API-key hash, not the
  raw key; the dev-mode "no key = open" is exactly the posture sp1011 now
  fail-closes on a public bind.
