# Security Policy

**Last updated:** 2026-10-26
**Related documents:** [`audits/AUDIT_PLAN.md`](audits/AUDIT_PLAN.md) (master audit plan, §12 disclosure policy) · [`docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md`](docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md) (incident response) · [`docs/2026-05-22-parallax-inference-audit-readiness.md`](docs/2026-05-22-parallax-inference-audit-readiness.md) (wire-protocol hardening audit, F30-F65)

## What we protect

PRSM (Protocol for Research, Storage, and Modeling) defends nine attack
surfaces, listed in `audits/AUDIT_PLAN.md` §3. The on-chain attack surface
includes live treasury infrastructure on Base mainnet:

| Contract | Address | Role |
|----------|---------|------|
| Foundation Safe (2-of-3 Gnosis Safe) | `0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791` | Treasury custodian |
| ProvenanceRegistry | `0xdF470BFa9eF310B196801D5105468515d0069915` | Content registration |
| RoyaltyDistributor v2 (canonical) | `0xfEa9aeB99e02FDb799E2Df3C9195Dc4e5323df7e` | Payment-split executor |
| FTNSTokenSimple | `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` | Native protocol token |

Off-chain we protect: bootstrap and relay nodes, the inference pipeline
(`prsm/inference/`, `prsm/chain_rpc/`, `prsm/streaming/`), the marketplace
listing surface, and the storage/proofs layer.

The audit-bundle stack (EscrowPool, BatchSettlementRegistry, StakeBond,
SignatureVerifier) is **pre-mainnet** pending external sign-off; reports
against pre-deploy code are still in scope.

## Reporting a vulnerability

### Channels

- **GitHub Security Advisory** (preferred): [Report a vulnerability](https://github.com/prsm-network/PRSM/security/advisories/new)
- **Email** (for time-sensitive issues): `security@prsm-network.com`
  - **PGP key:** *Pending publication.* Until the foundation PGP key is
    published, encrypt sensitive details with the GitHub Security Advisory
    flow (which is end-to-end encrypted between reporter and maintainers).
- **DO NOT** open a public GitHub issue for security vulnerabilities.

### What to include

- Affected component (contract address, file path, or service).
- Attack scenario — what does the attacker control, what's the sequence
  of calls, what does the attacker gain?
- Proof of concept — runnable test, transaction trace, or call log.
- Impact assessment — funds at risk, scope of compromise, severity estimate.
- Suggested remediation if known.

## Disclosure timeline (post-mainnet)

We follow a **90-day coordinated disclosure window** as defined in
`audits/AUDIT_PLAN.md` §12.

| Day | Phase | Action |
|-----|-------|--------|
| 0 | Receipt | Reporter submits via GHSA or email |
| 0–2 | Triage | Acknowledgment + severity assignment |
| 2–14 | Reproduction | Foundation engineering reproduces and confirms |
| 14–60 | Remediation | Fix development + audit of the fix |
| 60–75 | Coordinated deploy | Fix deployed; integrators privately notified ≥72 hours pre-disclosure |
| 75–90 | Public disclosure | GHSA + blog post + researcher credit (with consent) |
| 90 | Hard cap | Public disclosure even if remediation is incomplete |

### Critical-severity exception

For findings with **active exploitation** OR **>50% TVL at imminent risk**:

- Hour 0–2: Pause-transaction templates from
  [`docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md`](docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md)
  executed by the Foundation Safe.
- Hour 2–24: Private notification to known integrators + Forta alert
  subscribers.
- Hour 24–48: Public disclosure on Foundation channels + GHSA.
- Day 7+: Post-mortem published.

The 90-day window does **not** apply to actively-exploited bugs.

## Current disclosure state (live mainnet + pre-deploy audit-bundle)

PRSM is live on Base mainnet (chain 8453); the Phase 1.3 treasury
contracts above are deployed and the full disclosure policy applies to
them. The audit-bundle stack remains pre-deploy. Until
`audits/AUDIT_PLAN.md` Gate B clears (audit-bundle stack + Phase 8
contracts + L4 contest + L3 specialist crypto audit), the disclosure model is:

- Findings on **live mainnet contracts** (Phase 1.3 Task 8): full 90-day
  policy above + critical-severity exception applies.
- Findings on **audit-bundle pre-deploy code** or **off-chain Python**:
  reported confidentially, fixed pre-deploy, disclosed at Gate B clearance + 30
  days alongside the consolidated audit report.
- **Self-discovered findings** (internal review L0 / static-analysis L1):
  triaged internally; no external disclosure required pre-mainnet.

## Recognition

PRSM does not currently run a bug bounty program and does not offer
payment for vulnerability reports. Valid reports are acknowledged and,
with the reporter's consent, credited in the published advisory and in
`audits/hall-of-fame.md` once that roster is published.

### Duplicate handling

Credit goes to the first valid report, determined by submission timestamp on
`security@prsm-network.com` inbox or GHSA timestamp. Subsequent identical
reports receive acknowledgment.

## Scope

### In scope

- Smart contracts under `contracts/contracts/` (live mainnet + audit-bundle
  pre-deploy).
- Deploy ceremony scripts under `contracts/scripts/` (deploy-provenance.js,
  transfer-ownership.js, verify-audit-bundle-deployment.js, sweep-deployer.py).
- Off-chain Python services in `prsm/inference/`, `prsm/chain_rpc/`,
  `prsm/streaming/`, `prsm/node/`, `prsm/storage/`.
- The bootstrap node infrastructure
  (`wss://bootstrap-us.prsm-network.com:8765`).
- The on-chain monitoring + alert routing
  (`ops/monitoring/forta-bots/`).
- Foundation Safe operational security — phishing surface targeting signers,
  ceremony manipulation paths, etc. (responsible-disclosure paths only;
  social-engineering tests against signers are **NOT** authorized).

### Out of scope

- Issues already documented in `audits/findings/` or in the threat models
  (`docs/2026-04-22-r3-threat-model.md`,
  `docs/2026-04-30-phase3.x.11-threat-model-addendum.md`).
- Deprecated branches (`v1.5.x` and earlier).
- Spam, low-impact informational issues already known.
- Issues requiring physical access to a hardware wallet, an active social
  engineering attack against a Foundation signer, or compromise of a third-
  party service we use (these are non-PRSM attack surfaces).
- Best-practice / code-quality suggestions without an exploit (these are
  welcome as PRs or discussions rather than security reports).

## Defense-in-depth posture

The full defense matrix is in `audits/AUDIT_PLAN.md`. Eleven audit layers
cover the nine attack surfaces. Key continuously-active layers:

- **L0** — internal review gates (per-phase merge-ready tags).
- **L1** — static + symbolic tooling (Slither, Aderyn, Mythril, Halmos,
  Echidna; CI integration in progress).
- **L10a** — Forta detection bots monitoring live mainnet contracts for
  anomalous transfers, role grants, pause events, and distribution failures.
- **L10b** — exploit-response playbook with pre-staged pause transactions.

Pending external layers (per `audits/AUDIT_PLAN.md` §6 Gate B):
- **L3** — specialist Ed25519 crypto audit
- **L4** — external smart-contract audit (Code4rena + solo firm)
- **L5** — off-chain ML supply-chain audit
- **L6f** — network infrastructure pen-test
- **L7** — economic / game-theory audit
- **L8** — legal / regulatory review
- **L9** — public bug bounty (proposed only, not launched)

## Recent hardening — wire-protocol audit arc (2026-05-22 to 2026-05-23)

Between sprints 711 and 753, an iterative audit closed **50
F-class production-blockers (F30 through F80)** across multiple
audit dimensions. Detailed write-up + per-fix evidence in
[`docs/2026-05-22-parallax-inference-audit-readiness.md`](docs/2026-05-22-parallax-inference-audit-readiness.md).

Summary of the security-grade fixes:

### Wire protocol (F40, F50-F62) — 13 fixes

- **F50 / F51 / F52 / F55 / F56 / F57 / F58 / F59 / F61 / F62**
  — wire-protocol DoS hardening (back-pressure races, identical-
  request collisions, payload-size limits, per-peer concurrency
  caps, hang-defense execution timeouts) on both streaming and
  unary paths with parity.
- **F53 / F60** — stream + unary response hijack protection via
  sender-binding. Pre-fix, a peered third party that learned a
  victim's stream_id or request_id could forge frames/responses
  into the victim's pending queue/future. Now `pending[]` stores
  `(state, expected_sender)` tuples + handlers verify
  `msg.sender_id` before routing.
- **F54** — server-side resource leak: server kept iterating
  inference generator after requester disconnect, burning GPU
  on tokens nobody would receive.

### Transport foundation (F63 / F64) — 2 fixes

Transport verified cryptographic signatures only at handshake;
subsequent `msg.sender_id` was wire-trusted but not crypto-bound.
A peer with valid handshake could spoof sender_id to defeat
per-peer caps (F56/F59) AND forge responses (F53/F60) —
undermining several preceding fixes. Sprint 730 bound sender_id
at chain-executor dispatch wrappers; sprint 731 generalized at
the transport layer (`WebSocketTransport._dispatch`) so EVERY
`MSG_DIRECT` handler in the codebase — including `ledger_sync`
FTNS transfers, `compute_provider`, `storage_provider`,
`content_provider`, `agent_registry` — now sees the handshake-
authenticated peer identity.

### Admin auth + reconnaissance defense (F65-F80) — 16 fixes

**Deployment-pattern defenses (F65-F71)**:

- **F65** — `/admin/*` was unauthenticated (`tags=["admin"]` was
  a swagger grouping, not access control). Middleware now
  restricts to loopback by default.
- **F66 / F67** — reverse-proxy bypass defense. `X-Forwarded-For`
  + `X-Real-IP` are inspected when immediate client is loopback;
  external upstream → 403. Closes the nginx/HAProxy-on-same-host
  pattern where daemon would have seen `client.host=127.0.0.1`
  for all proxied external traffic.
- **F68** — loopback representations widened to include IPv4-
  mapped IPv6 (`::ffff:127.0.0.1`, dual-stack Linux default) and
  the entire 127.0.0.0/8 block per RFC 1122.
- **F69** — HTTP rate limiter ("`*_PER_REQUESTER`" envs) was
  keyed by `node.identity.node_id` (a constant) — effectively
  global, not per-requester. Now keyed by HTTP-client IP via
  proxy-aware helper. False-confidence bug class.
- **F70** — HTTP body-size limit (HTTP-side analog of F55/F58).
  `PRSM_HTTP_MAX_BODY_BYTES` default 1 MiB.
- **F71** — DNS-rebinding defense. Browsers always set the
  `Origin` header on cross-origin requests; CLI tools don't.
  Middleware rejects gated requests with `Origin` set, blocking
  the canonical loopback-as-auth attack against locally-bound
  services.

**Reconnaissance-endpoint defenses (F72-F80)** — 17 endpoint
groups now loopback-gated:

- **F72** — `/docs` + `/openapi.json` hidden by default. Complete
  API surface map. `PRSM_API_DOCS_ENABLED=1` opt-in.
- **F73** — `/metrics` (Prometheus). Leaked `prsm_total_locked_ftns`
  (financial), peer counts, subsystem counters.
- **F74** — `/info` + `/health/detailed`. Leaked operator's
  on-chain EOA, node_id, wired contract addresses, software
  version (CVE-matching surface), per-subsystem state.
- **F75** — `/status` + `/rings/status`. The largest single leak:
  operator's literal `ftns_balance` + 13+ subsystem provider
  stats + bootstrap telemetry.
- **F76** — `/peers`. Complete network topology: peer_ids +
  IP:port + connected_at + last_seen.
- **F77** — `/balance` + `/bootstrap/status`. The worst recon
  leak: operator's balance PLUS last 20 transactions with
  counterparty wallet_ids + amounts + timestamps. Complete
  financial profile.
- **F78** — `/transactions` (200-tx history) + `/staking/status`
  (stake position + unstake-unlock timing) + `/settlement/*`
  (counts + amounts + schedules).
- **F79** — `/balance/onchain` + `/audit/summary` + `/audit/recent`
  + `/ledger/sync/stats`. Access-log + sync-state recon vectors.
- **F80** — `/agents/spending` + `/privacy/budget`. Per-agent
  FTNS spend + DP epsilon usage patterns. `/agents` (bare list)
  intentionally preserved as marketplace service-discovery.

**Operator-facing consequences** are documented in
[`docs/operations/parallax-inference-deploy.md`](docs/operations/parallax-inference-deploy.md).
**Existing operators upgrading past sprint 734 should expect ALL
of `/admin/*`, `/metrics`, `/info`, `/status`, `/peers`,
`/balance`, `/transactions`, `/staking/status`, `/settlement/*`,
`/audit/*`, `/ledger/sync/stats`, and others to default-deny
non-loopback access** — set `PRSM_ADMIN_REMOTE_ALLOWED=1` AND
add a real auth layer (reverse-proxy auth, VPN) if remote
access is required.

82/82 pin tests across the F65-F80 admin-auth + reconnaissance
arc. Per-deployment-pattern coverage: direct-network, XFF-
forwarding proxy, X-Real-IP-forwarding proxy, IPv4-mapped IPv6
+ 127/8, DNS rebinding via Origin. Minimal `/health` and
`/agents` intentionally preserved as public (LB probe + service
discovery).

## Supported versions

| Version | Status | Security support |
|---------|--------|------------------|
| `main` (current) | Active development | Full |
| `v1.6.x` (latest tagged) | Active maintenance | Full |
| `v1.5.x` and earlier | Superseded | None |

The on-chain treasury layer (deployed 2026-05-04, Phase 1.3 Task 8) is
non-Ownable and non-upgradeable; "version" semantics for those contracts
are governed by the manifest at
`contracts/deployments/provenance-base-1777917793612.json` and the milestone
tag `phase1.3-task8-complete-20260504`.

## Authorization clarification

This policy authorizes good-faith security research within scope. It does
**not** authorize:

- Attacks against the live infrastructure that disrupt service (DoS testing
  against bootstrap nodes, etc.) — please coordinate first.
- Social engineering against Foundation employees, signers, or contributors.
- Physical attacks against any party.
- Testing on user systems without explicit authorization.

Out-of-scope research conducted in good faith and reported responsibly will
not be pursued legally; coordinate with us before testing edge cases.

## Acknowledgments

PRSM thanks the security research community. Researchers who report valid
findings are credited (with consent) in the published advisory and, once
that roster is published, in `audits/hall-of-fame.md` after fix deployment.

---

## Operationalization status (operator-facing notes)

The following are pending engineering/operations actions, tracked in the
PRSM task list:

- [ ] `security@prsm-network.com` inbox setup (DNS/MX + email provider).
- [ ] Foundation PGP keypair generation + publication.
- [ ] `audits/hall-of-fame.md` initial commit (empty roster).
- [ ] L10a alert routing wired to PagerDuty + Foundation council Slack.

Until these complete, the GHSA channel is the primary reporting path and
all operational claims above should be read as "policy-target" rather than
"already-operational."
