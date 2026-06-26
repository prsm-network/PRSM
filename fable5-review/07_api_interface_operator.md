# Fable 5 Review — Domain 07: API, interface, CLI & operator surface

Adversarial, cross-cutting review. See `00_INDEX.md` for shared framing + output format.

## Context

The node exposes a management/compute HTTP API (FastAPI), admin + health/observability
endpoints, a public/onboarding interface, dashboards, a CLI, and an SDK. This is the
externally-reachable attack surface. Money + attestation ingress logic inside the inference
endpoints is Domain 01 — **this domain owns the API surface as a whole**: authn/authz,
input validation, the network-exposure auth posture, rate-limiting, error/info leakage, admin
endpoint protection, and the CLI/operator UX.

### Read
- `prsm/interface/api/` (the bulk of the HTTP API, ~20k loc), `prsm/interface/public/`,
  `prsm/interface/onboarding/`, `prsm/interface/dashboard/`
- `prsm/node/api.py` (the endpoint surface beyond the Domain-01 settle sites: admin/incident/
  insurance/tee/federated/pipeline triage endpoints, `/health/detailed`, `/info`, `/metrics`,
  `/compute/*`, `/rings/status`), node lifecycle `start`/`stop`, the public-bind auth posture
  (`assess_public_bind_auth_posture`)
- `prsm/cli_modules/`, `prsm/cli_helpers/`, `prsm/api/`, `prsm/sdk/`, `prsm/dashboard/`

### Invariants — confirm or break
1. **Auth on protected endpoints.** Money/admin/mutating endpoints require the configured API
   key (`PRSM_NODE_API_KEY`); a bind to a non-loopback interface WITHOUT a key is flagged AND
   the protected endpoints are actually gated (not just warned about). Find a money/admin
   endpoint reachable unauthenticated when bound publicly.
2. **Input validation on every request body.** Sizes, types, ranges, and required fields are
   validated before use; no unbounded allocation from a request field; no injection (the
   request reaches SQL/shell/template/LLM-prompt boundaries safely). Find an endpoint that
   trusts a body field into a dangerous sink.
3. **No info leakage.** Error responses, `/health/detailed`, `/info`, logs, and admin views do
   not leak secrets (private keys, full RPC URLs, internal addresses, stack traces with
   sensitive state). The settlement/collateral status surfaces are read-only + redacted.
4. **Admin/triage endpoints are read-only or properly gated.** The incident/insurance/TEE/
   federated/pipeline triage endpoints are read-only (mutating ops gated to the right
   authority; e.g. the insurance recovery composer PRODUCES a multisig tx but does NOT
   execute). Confirm none silently mutate or execute privileged actions.
5. **Rate-limiting / abuse.** Per-requester buckets (or equivalent) prevent a single caller
   from exhausting the node; the rate-limit key can't be trivially spoofed to evade it.
6. **CLI safety.** The CLI never logs/persists secrets; commands that produce multisig-
   uploadable txs do NOT execute them; destructive/irreversible commands require explicit
   confirmation.

### Hunt list
- FastAPI dependency/auth wiring: an endpoint that forgot the auth dependency; a path that
  bypasses the global gate; CORS/allowed-origins too permissive.
- `eval`/`exec`/format-string/SQL built from request input; SSRF via a user-supplied URL the
  server fetches.
- Unbounded request body / streaming without a size cap; a slowloris-style stream.
- Health/metrics endpoints that expose internal topology or secrets to an unauthenticated
  scraper.
- Inconsistent authz between the unary and streaming variants of the same operation.

Follow the `00_INDEX.md` output format. Report only.
