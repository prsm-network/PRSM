# Multi-Replica Fiat + Compute Operator Runbook

**Audience:** operators running PRSM node API replicas behind a load balancer that serve the **fiat onramp/offramp** and/or **`/compute/forge`** surfaces.

**Last updated:** 2026-06-20 · Covers sprints 1175, 1177, 1179, 1180 (and the related governance change in 1181).

---

## 1. Why this runbook exists

Two money-safety controls were originally enforced against **per-process, in-memory state**. On a **single replica** they are correct. On **multiple replicas** behind a load balancer, each replica keeps its own copy of that state, so the control is enforced **per-pod, not globally**:

1. **AML tier limit** (per-user USD/day onramp+offramp cap). The rolling total lived in an in-memory ring per replica → a user whose requests fan across *N* replicas could transact up to *N×* their limit.
2. **`/compute/forge` idempotency** (the `Idempotency-Key` double-charge guard). The key→job index lived in an in-memory store per replica → two same-key requests routed to *different* replicas could each lock an escrow (an un-refunded 2× charge for one logical request).

Both now have **opt-in shared-state backends (Redis)** that make enforcement global. **Defaults are unchanged** — a single-replica deployment needs nothing. **If you run more than one replica serving these surfaces, you must act** (Section 3).

> **TL;DR for a multi-replica production deployment:** run a shared Redis, then set `PRSM_FIAT_TIER_LIMIT_MODE=shared_redis`, `PRSM_FIAT_COMPLIANCE_REDIS_URL`, `PRSM_FORGE_IDEMPOTENCY_REDIS_URL`, and `PRSM_FIAT_COMPLIANCE_LOG_DIR` (a shared/retained path) on **every** replica. See Section 8B.

---

## 2. Decision matrix

| Deployment | AML tier limit | Forge idempotency | What to do |
|---|---|---|---|
| **Single replica** | Correct as-is (`process_local`) | Correct as-is (in-process) | Nothing required. (Still set `PRSM_FIAT_COMPLIANCE_LOG_DIR` for audit retention — Section 6.) |
| **Multi-replica, fiat NOT served** | n/a | Set the forge Redis URL (Section 5) | Only the forge idempotency backend matters. |
| **Multi-replica, compliance-strict, Redis not ready yet** | `strict_shared` (fail-closed: deny fiat) | in-process only | Stopgap: deny rather than under-enforce. See Section 8A. |
| **Multi-replica, production** | `shared_redis` (global enforce) | shared (cross-replica) | Full setup. See Section 8B. |

---

## 3. The fiat AML tier limit (sprints 1175 + 1179)

### Modes — `PRSM_FIAT_TIER_LIMIT_MODE`

| Value | Behavior | Use when |
|---|---|---|
| `process_local` (**default**) | Rolling total from the **per-replica** in-memory ring. Correct for a single replica; **under-enforces** across replicas. Tier-check responses carry `tier_limit_scope: "process_local"`. | Single replica. |
| `strict_shared` | **Fail-closed.** Every KYC-verified user's gated fiat request is denied with **HTTP 503 `tier_limit_enforcement_unavailable`**. No Redis required. | Multi-replica where you would rather **deny fiat than risk under-enforcing** the AML limit, and you haven't wired Redis yet. |
| `shared_redis` | **Global enforcement.** The rolling total is read from / written to a **shared Redis** sorted-set, so the limit is enforced across all replicas. Responses carry `tier_limit_scope: "shared_redis"`. | Multi-replica production (with Redis). |

### Redis URL (only for `shared_redis`)

Resolved in this order (first non-empty wins):

```
PRSM_FIAT_COMPLIANCE_REDIS_URL   →   PRSM_REDIS_URL   →   REDIS_URL
```

Example: `PRSM_FIAT_COMPLIANCE_REDIS_URL=redis://10.0.0.5:6379/0`

### Fail-mode: **fail-CLOSED** (compliance-correct)

In `shared_redis` mode, if Redis is **unreachable** (read error) **or no URL is wired** (misconfiguration), the tier check **fails closed**: the gated fiat surface returns **HTTP 503 `tier_limit_enforcement_unavailable`** with `tier_limit_scope: "shared_redis_unavailable"`. It never silently falls back to a 0.0 / per-pod total. Rationale: an AML limit is a regulatory control — denying fiat during a Redis outage is correct; under-enforcing is not.

> **Operational implication:** in `shared_redis` mode, **Redis is a hard dependency of the fiat surface.** Run it HA (Sentinel / managed). A Redis outage = fiat denied (recoverable: it resumes when Redis returns).

### Tier limits (independent of mode)

| Env | Default | Meaning |
|---|---|---|
| `PRSM_KYC_TIER_LIMIT_BASIC_USD` | `1000` | basic-KYC USD/day cap |
| `PRSM_KYC_TIER_LIMIT_ENHANCED_USD` | `10000` | enhanced-KYC USD/day cap |

---

## 4. How the shared AML total works (for reviewers)

`shared_redis` mode does **not** replace the audit ring — the in-memory/disk compliance ring (Section 6) remains the regulator-facing audit log. It adds a **separate Redis sorted-set per user** used only for the limit check: each settled/reserved fiat-USD execute is a member scored by its timestamp; the read dedups by `intent_id` (so an onramp's PENDING reservation and its later CONFIRMED settle count once) and sums over the rolling window. This faithfully mirrors the single-process semantics — a differential test asserts byte-equal totals between the in-memory and Redis backends. Abandoned/expired reservations write a terminal `$0` entry so they stop counting.

---

## 5. The `/compute/forge` idempotency claim (sprints 1177 + 1180)

The `Idempotency-Key` header makes a retried `POST /compute/forge` safe (no double escrow / double compute).

- **In-process (always on, sp1177):** a synchronous re-check-and-reserve immediately before the escrow lock closes the **same-replica** concurrent-duplicate window (the common keepalive/affinity retry). No configuration.
- **Cross-replica (opt-in, sp1180):** set a Redis URL and the handler additionally claims each key across replicas via `SETNX` before locking any escrow. A duplicate that lands on another replica gets **HTTP 409 `idempotency_key_in_progress`** (no second escrow).

### Redis URL

Resolved in this order:

```
PRSM_FORGE_IDEMPOTENCY_REDIS_URL   →   PRSM_REDIS_URL   →   REDIS_URL
```

If unset → in-process behavior only (correct for single replica).

### Fail-mode: **fail-OPEN to the in-process guarantee**

If the claim-store Redis errors, forge **proceeds** using the sp1177 in-process guarantee (logged loudly) — it does **not** 503/409. Rationale: a forge double-charge is a refundable UX concern, so a Redis outage should degrade gracefully rather than break compute. (This is the **opposite** of the AML limit's fail-closed posture — see Section 9.)

---

## 6. Compliance audit ring (retention — required in production)

Independent of replica count, the fiat compliance audit log must persist for regulator retention (AUSTRAC / FinCEN / IRS, 5–7 years).

| Env | Required? | Meaning |
|---|---|---|
| `PRSM_FIAT_COMPLIANCE_LOG_DIR` | **Production: yes** | Directory for the per-entry JSON audit log. Unset → bounded in-memory only (lost on restart). For multi-replica, point each replica at a **shared/retained** path (e.g. an EFS/NFS mount) or ship the files to central storage. |
| `PRSM_OPERATOR_JURISDICTION` | recommended | Stamped on each entry for jurisdiction-aware reporting. |

---

## 7. Redis requirements

- **One shared Redis** reachable by every replica (a single instance/cluster; **not** a per-replica sidecar — that would defeat the purpose).
- You can use **one Redis for both** AML and forge (set `PRSM_REDIS_URL` once) or separate instances (set the two feature-specific URLs).
- **Run it HA in production** (managed Redis or Sentinel). The AML path fails **closed** on a Redis outage (fiat denied); the forge path fails **open** (degrades to per-replica).
- Keyspace used: `prsm:fiat:roll:<user_id>` (sorted-sets, self-expiring TTL) and `prsm:forge:idem:<key>` (SETNX, ~15-min TTL). Both are bounded/self-expiring; no operator cleanup needed.

---

## 8. Recommended configurations

### 8A. Multi-replica, Redis not ready — fail-closed stopgap

Safe but conservative: denies fiat rather than under-enforcing the AML limit. Forge stays in-process (same-replica safe).

```bash
PRSM_FIAT_TIER_LIMIT_MODE=strict_shared
PRSM_FIAT_COMPLIANCE_LOG_DIR=/var/lib/prsm/fiat-compliance   # shared/retained
PRSM_OPERATOR_JURISDICTION=AU
# (no Redis; forge idempotency is in-process only)
```

### 8B. Multi-replica, production — global enforcement (recommended)

Set on **every** replica:

```bash
# Shared Redis (one URL covers both features)
PRSM_REDIS_URL=redis://prsm-redis.internal:6379/0
# or set the two feature-specific URLs instead:
#   PRSM_FIAT_COMPLIANCE_REDIS_URL=...
#   PRSM_FORGE_IDEMPOTENCY_REDIS_URL=...

# AML tier limit: enforce globally (fail-closed if Redis is down)
PRSM_FIAT_TIER_LIMIT_MODE=shared_redis

# Audit retention (shared/retained path)
PRSM_FIAT_COMPLIANCE_LOG_DIR=/mnt/prsm-compliance
PRSM_OPERATOR_JURISDICTION=AU

# (optional) tier caps if not using the $1k/$10k defaults
# PRSM_KYC_TIER_LIMIT_BASIC_USD=1000
# PRSM_KYC_TIER_LIMIT_ENHANCED_USD=10000
```

> **Forge idempotency is wired automatically** by the `PRSM_REDIS_URL` fallback once it's set — no separate flag needed.

---

## 9. Fail-mode summary (and why they differ)

| Control | On Redis outage | Why |
|---|---|---|
| **AML tier limit** (`shared_redis`) | **Fail-CLOSED** — deny fiat (503) | Regulatory limit; under-enforcing is a compliance violation. Denying is recoverable. |
| **Forge idempotency** (cross-replica) | **Fail-OPEN** — degrade to in-process | A double-charge is refundable UX; breaking all compute on a Redis blip is worse. |

---

## 10. Verifying it's live

1. **Startup logs** — each replica logs the wiring:
   - AML: a warning on construction stating the rolling-total scope (process-local vs the strict/shared posture).
   - Forge (sp1180): `cross-replica forge idempotency claim store wired (SETNX)` when a Redis URL resolved.
2. **Tier-check response field** — call an onramp quote/execute for a KYC-verified user and inspect `tier_limit_scope`:
   - `process_local` → per-replica (expected only on single-replica).
   - `shared_redis` → global enforcement active.
   - `shared_redis_unavailable` → mode is `shared_redis` but Redis is down/unwired (you're failing closed — fix Redis).
3. **AML smoke test (staging)** — with `shared_redis`, drive onramp executes for one user against **different replicas** until the cap; the *(N+1)*th must be `403 tier_limit_exceeded` regardless of which replica serves it.
4. **Forge smoke test (staging)** — fire two `POST /compute/forge` with the same `Idempotency-Key` at **two replicas**; exactly one locks an escrow, the other returns `409 idempotency_key_in_progress`.

---

## 11. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| All fiat returns `503 tier_limit_enforcement_unavailable` | `shared_redis` but Redis unreachable/unwired, **or** `strict_shared` set deliberately | Check the Redis URL + connectivity; or confirm `strict_shared` is intended. `tier_limit_scope` distinguishes (`shared_redis_unavailable` = Redis problem). |
| Users exceed their AML cap across replicas | Mode left at `process_local` on multi-replica | Set `shared_redis` (+ Redis URL) or `strict_shared` on every replica. |
| Duplicate forge jobs / double charges across replicas | No forge Redis URL on multi-replica | Set `PRSM_FORGE_IDEMPOTENCY_REDIS_URL` (or `PRSM_REDIS_URL`) on every replica. |
| Compliance audit entries lost on restart | `PRSM_FIAT_COMPLIANCE_LOG_DIR` unset or not shared/retained | Point every replica at a shared/retained path. |
| Config differs between replicas | Env not applied uniformly | These vars **must be identical on every replica** — a mismatched replica reintroduces the gap. |

---

## 12. Quick reference

| Env var | Default | Purpose |
|---|---|---|
| `PRSM_FIAT_TIER_LIMIT_MODE` | `process_local` | `process_local` \| `strict_shared` \| `shared_redis` |
| `PRSM_FIAT_COMPLIANCE_REDIS_URL` | — | AML shared-total Redis (falls back to `PRSM_REDIS_URL` / `REDIS_URL`) |
| `PRSM_FORGE_IDEMPOTENCY_REDIS_URL` | — | Forge cross-replica claim Redis (falls back to `PRSM_REDIS_URL` / `REDIS_URL`) |
| `PRSM_REDIS_URL` / `REDIS_URL` | — | Shared fallback URL for both features |
| `PRSM_FIAT_COMPLIANCE_LOG_DIR` | in-memory | Audit-log retention directory (production-required) |
| `PRSM_OPERATOR_JURISDICTION` | — | Jurisdiction stamp on audit entries |
| `PRSM_KYC_TIER_LIMIT_BASIC_USD` | `1000` | basic-tier USD/day cap |
| `PRSM_KYC_TIER_LIMIT_ENHANCED_USD` | `10000` | enhanced-tier USD/day cap |

---

## 13. Related: governance voting requires a stake (sprint 1181)

Not multi-replica-specific, but operator-relevant if you enable governance: voting power is now computed from a user's **governance stake** (`stake_for_governance`, which locks tokens), not their liquid balance. Voters with **no stake have zero voting power**. Communicate the stake-to-vote requirement to participants before opening governance. (This closed a Sybil-vote-inflation exploit; the stake is shared ledger state, so it is already multi-replica-consistent.)
