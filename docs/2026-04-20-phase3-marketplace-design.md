# Phase 3: Compute Marketplace — Design

**Status:** Draft authored 2026-04-20. Supersedes the marketplace sketches in the project roadmap. Phase 3 ships on top of the Phase 2 remote-dispatch primitives and the Phase 2.1 addendum primitives (gated commits `bc8c951` / `phase2.1-merge-ready-20260420`).

**Dependencies:**
- Phase 2 merge gate passed (remote dispatch + signed receipts + escrow state machine).
- Phase 2.1 merge gate passed (TopologyRandomizer, ShardPreemptedError, tee_attestation schema field).
- Phase 1.3 mainnet deploy **not required** (Phase 3 uses local-ledger settlement — see §3.5).

**Non-dependencies:** hardware multi-sig not required; Phase 3 can ship during the Phase 1.3 Task 8 wait.

---

## 1. Context & Goals

Phase 2 built the wire protocol for one-requester → one-provider shard execution: signed receipts, escrow, preemption signaling, verification. **Phase 2 does not answer the question of how a requester finds providers, or how price is set.** A requester in Phase 2 has to already know a provider's `node_id` + the price they're willing to pay; dispatch just executes the already-negotiated trade.

Phase 3 closes that gap. Providers broadcast their available capacity + price; requesters discover eligible providers, filter by policy (TEE requirement, max price, min reputation), and dispatch. The MarketplaceOrchestrator wraps Phase 2's RemoteShardDispatcher — the dispatcher stays unchanged; Phase 3 is discovery + selection + handoff.

**Non-goals for Phase 3** (see §10 for full list):
- No auction mechanics (fixed-price only; Dutch/sealed-bid → Phase 6).
- No on-chain settlement per-dispatch (local ledger only; batch settlement → Phase 3.1).
- No stake-slashing reputation enforcement (lightweight score only; slash → Phase 7).
- No on-chain registry (gossip-only; on-chain anchor → Phase 3.1+).

---

## 2. Scope

**In scope:**
- Provider capacity advertisement over gossip (`GOSSIP_MARKETPLACE_LISTING`).
- Requester-side directory aggregation from observed listings.
- Eligibility filter (price ceiling, TEE requirement, min reputation).
- Price handshake (quote → ack) before dispatch.
- Reputation tracker: per-provider success rate, preempted rate, p50/p95 latency.
- MarketplaceOrchestrator: single entrypoint `orchestrate_sharded_inference(shards, input, job_id, policy) → result`. Internally composes directory, filter, randomizer (Phase 2.1 Line B), dispatcher (Phase 2), receipt verifier (Phase 2).
- Integration test: 3-node marketplace end-to-end with price-filtered dispatch.

**Out of scope (Phase 3.x or later):**
- On-chain settlement of any form.
- Node advertisement to non-gossip peers.
- Auction price discovery.
- SLA disputes with financial consequences.
- Time-varying dynamic pricing (providers can update their listings, but the update mechanism is "re-broadcast" not "bid").

---

## 3. Protocol

### 3.1 Provider capacity advertisement

Providers publish a signed `ProviderListing` via gossip. Fields:

```
ProviderListing {
    listing_id: str              # uuid4
    provider_id: str             # NodeIdentity.node_id
    provider_pubkey_b64: str     # for signature verification
    capacity_shards_per_sec: float
    max_shard_bytes: int
    supported_dtypes: List[str]  # ["float32", "float64"]
    price_per_shard_ftns: float  # fixed-price, flat
    tee_capable: bool            # advertises TEE quote support
    stake_tier: str              # "open" | "standard" | "premium" | "critical"
    advertised_at_unix: int
    ttl_seconds: int             # default 300 (5 min); listing expires if not refreshed
    signature: str               # Ed25519 over the canonical payload
}
```

**Canonical signing payload:**
```
keccak256(
    "{listing_id}||{provider_id}||{capacity_shards_per_sec}||"
    "{max_shard_bytes}||{price_per_shard_ftns}||{tee_capable}||"
    "{stake_tier}||{advertised_at_unix}||{ttl_seconds}"
)
```

Providers re-broadcast every 60-120s (jittered) while listing conditions hold. Stale listings (advertised_at + ttl < now) are dropped by directory.

**Gossip topic:** `GOSSIP_MARKETPLACE_LISTING` (new; register in `prsm/node/gossip.py`).

### 3.2 Requester-side directory

`MarketplaceDirectory` subscribes to `GOSSIP_MARKETPLACE_LISTING` and maintains an in-memory dict `{provider_id: ProviderListing}`. On each gossip event:
1. Verify signature against `provider_pubkey_b64`.
2. Verify `provider_id == hex(sha256(pubkey_bytes))[:32]` (same binding as Phase 2 receipts).
3. Replace any existing listing for this `provider_id` with the newer `advertised_at_unix`.
4. Evict entries where `advertised_at + ttl_seconds < now`.

Directory exposes:
- `list_active_providers() → List[ProviderListing]`
- `get_listing(provider_id) → Optional[ProviderListing]`
- `size() → int`

### 3.3 Eligibility filter

`EligibilityFilter` takes a `DispatchPolicy` + list of listings, returns the filtered subset.

```
DispatchPolicy {
    max_price_per_shard_ftns: float
    require_tee: bool              # if True, only tee_capable providers
    min_stake_tier: str            # listings below this tier rejected
    min_reputation_score: float    # 0.0 - 1.0; see §6
    required_dtype: str            # e.g., "float64"
    min_capacity_shards_per_sec: float
}
```

Filter order (short-circuit on first rejection per listing):
1. TTL not expired
2. `price_per_shard_ftns <= max_price_per_shard_ftns`
3. `tee_capable` iff `require_tee`
4. `stake_tier >= min_stake_tier` (ordinal comparison)
5. `dtype in supported_dtypes`
6. `capacity_shards_per_sec >= min_capacity_shards_per_sec`
7. Reputation score for `provider_id` from ReputationTracker `>= min_reputation_score`

Result: list of eligible `ProviderListing` objects, unordered.

### 3.4 Price handshake

Before dispatch, requester sends a `shard_price_quote_request` direct message to each candidate provider (one per shard, after eligibility+randomization):

```
shard_price_quote_request {
    subtype: "shard_price_quote_request"
    request_id: str
    listing_id: str
    shard_index: int
    shard_size_bytes: int
    max_acceptable_price_ftns: float
    deadline_unix: int
}
```

Provider responds with either:
- `shard_price_quote_ack` — `{quoted_price_ftns: float, quote_expires_unix: int, signature}`
- `shard_price_quote_reject` — `{reason: str}` (e.g., `"overloaded"`, `"withdrawn"`, `"shard_too_large"`)

The quoted price **must not exceed** the listing's advertised `price_per_shard_ftns` (requester verifies). If quote > listing, requester rejects and marks the listing stale.

On ack, requester proceeds to the Phase 2 dispatch flow with `escrow_amount_ftns = quoted_price_ftns`. Escrow job_id format: `marketplace:{request_id}:shard:{shard_index}` (namespaced to distinguish from direct-dispatch).

**Why a quote round-trip at all**, when the listing already contains price? Two reasons:
1. Capacity commit — the provider reserves a slot for this specific dispatch. Without it, 100 requesters could all claim the same advertised capacity simultaneously.
2. Price freshness — the listing's `price_per_shard_ftns` is a ceiling, not a lock. Providers may honor lower prices on a per-request basis (volume discount, off-peak, etc.). The quote is the binding number.

### 3.5 Settlement

Local-ledger-only for Phase 3. The existing `PaymentEscrow` + `LocalLedger` pair handles it exactly as Phase 2 did: requester's wallet → escrow holding wallet → provider's wallet on successful receipt verification.

Cross-node ledger consistency is out of scope. Phase 3 assumes all participating nodes share the same logical ledger (single-orchestrator deployments, or federated-node deployments with a gossip-reconciled ledger — see Phase 1.3 work). Phase 3.1 will add **on-chain batch settlement**: providers accumulate local receipts for N dispatches or M seconds, then post a single on-chain transaction redeeming the batch.

---

## 4. Price Model

**Phase 3 ships with per-shard fixed-price in FTNS, flat rate.**

Rationale:
- Matches Phase 2's per-shard escrow granularity exactly — no new escrow semantics.
- Simplest wire format: one float in the listing.
- No metering infrastructure required. No FLOP counting, no trusted benchmarks, no gaming surface.
- Providers express their cost basis in a single number they control.
- Requesters budget in a single number they understand.

**Known limitations** (Phase 6 addresses these):
- A 10 KB shard and a 10 MB shard pay the same price. Unfair when shard size variance is high.
- No congestion pricing. Provider can't charge more during peak load.
- No volume discount baked into the protocol. A requester sending 10k shards to one provider pays the same per-shard as someone sending one.

**Phase 3 workarounds for the limitations above** (documented; not enforced):
- Providers can run multiple listings with different prices, each suitable for a different shard size range.
- Providers can update their listing frequently (re-broadcast with new `price_per_shard_ftns`) to implement time-of-day pricing.
- Off-protocol volume-discount deals are possible — requester and provider negotiate out-of-band, provider issues a bespoke low-price listing visible only to the requester (via direct MSG_DIRECT).

Phase 6 will replace per-shard fixed with **size-tiered bands** (listing carries `price_per_shard_ftns` as a dict `{"≤1MB": X, "≤10MB": Y, "≤100MB": Z}`). Wire-format-compatible extension.

---

## 5. SLA + Preemption

### 5.1 Response-time SLA

Providers advertise `capacity_shards_per_sec` — an expected throughput. Not a hard commit. Violating it counts against reputation (see §6), but Phase 3 does not enforce it with payment penalties.

### 5.2 Preemption handling (Phase 2.1 Line A integration)

When a provider's runtime detects spot-instance termination (e.g., AWS 2-minute warning, GCP preemption notice, Kubernetes pod disruption), the provider invokes `ComputeProvider.preempt_inflight_shards()` — this sends `shard_execute_response` with `status="preempted"` for all in-flight shards.

The MarketplaceOrchestrator catches `ShardPreemptedError` from the dispatcher and:
1. Records a preemption event against the provider's reputation (distinct from a failure — no score penalty, but preempted rate is tracked).
2. Refunds the escrow (handled by dispatcher automatically per Phase 2.1 Line A contract).
3. Re-dispatches the preempted shards to a fresh eligible pool (excluding the preempted provider for this inference to avoid a thrash loop if the pod eviction is slow).

Provider-side SIGTERM → `preempt_inflight_shards()` wiring is **in scope for Phase 3** (was deferred from Phase 2.1). See plan Task 5.

### 5.3 Timeout SLA

Default per-dispatch timeout = 30 seconds (inherited from Phase 2 dispatcher default). Providers can advertise a custom `max_response_time_seconds` in the listing; requesters who set `DispatchPolicy.max_timeout_seconds` below a listing's value will skip that listing during eligibility filtering.

---

## 6. Reputation

**Lightweight, Phase 3 scope only. Slashing = Phase 7.**

The ReputationTracker maintains per-provider rolling counters:

```
ProviderReputation {
    provider_id: str
    successful_dispatches: int          # last 1000
    failed_dispatches: int               # last 1000 (verification failure, timeout after retries)
    preempted_dispatches: int           # last 1000 (honest-work failures — not a reputation penalty)
    latency_ms_p50: float               # rolling p50 over last 1000 successes
    latency_ms_p95: float               # rolling p95 over last 1000 successes
    first_seen_unix: int
    last_seen_unix: int
}
```

Reputation score ∈ [0.0, 1.0]:
```
score = successful / (successful + failed)   # preempted ignored in denominator
```

Special cases:
- Brand-new provider (no history) → score = 0.5 (neutral). Allows bootstrapping without a cold-start dead zone.
- After `successful + failed < 10`, the score is informational only — treated as 0.5 for filtering.

**Intentionally excluded:**
- No cross-node reputation aggregation (each node maintains its own observed reputation). Federated reputation gossip = Phase 6.
- No reputation signatures or on-chain anchoring. Reputation is advisory to the local requester, not a protocol commitment.
- No reputation-based price modulation by providers (they don't know their score).

---

## 7. Phase 2.1 Integration

### 7.1 TopologyRandomizer (Line B)

After `EligibilityFilter` returns the eligible pool, MarketplaceOrchestrator calls:
```python
assignments = randomizer.assign(
    eligible_node_ids=[l.provider_id for l in filtered],
    num_shards=len(shards),
)
```

- `assign` (with replacement) is the default for tensor-parallel where collocation is acceptable.
- `assign_unique` (without replacement) is used when `DispatchPolicy.require_unique_providers=True` — e.g., redundant-execution workloads (Phase 7 Tier B preview).

The randomizer's pool-size check is the dispatch-time gate that ensures eligibility didn't drop the pool below `num_shards`.

### 7.2 tee_attestation schema (Line C)

`DispatchPolicy.require_tee` filters listings by `tee_capable` flag in §3.3 step 3. On dispatch, the request payload (Phase 2) carries `"require_tee": True`; providers who claimed `tee_capable` in their listing are contractually obligated to return a `tee_attestation` block in their receipt.

Phase 3 does NOT verify the attestation quote chain — that's the deferred Phase 2.1 follow-up. Phase 3 only enforces presence/absence: a receipt from a `require_tee=True` dispatch without a `tee_attestation` field fails Phase 3-level acceptance (MarketplaceOrchestrator raises `MissingAttestationError`, refunds escrow, drops reputation of the lying provider).

### 7.3 ShardPreemptedError (Line A)

MarketplaceOrchestrator catches this distinctly — no reputation penalty (it's honest work), but does count the event for the `preempted_dispatches` metric. Re-dispatch to fresh pool is handled at orchestrator layer (§5.2).

---

## 8. Anti-Spam + Sybil

### 8.1 Advertisement rate limit

Each `provider_id` can broadcast at most 1 listing per 30 seconds. Directory drops duplicates. Enforces a floor on the re-broadcast cadence + limits gossip spam from misbehaving providers.

### 8.2 Minimum stake tier (advisory)

Requesters' default `DispatchPolicy.min_stake_tier="open"` accepts all stake tiers. Requesters with value-sensitive workloads set `min_stake_tier="standard"` (5000 FTNS bonded) or higher. Stake-bond implementation is deferred to Phase 7; Phase 3 trusts the `stake_tier` field in the listing as self-reported.

**Known gap:** in Phase 3, a sybil attacker can freely claim any stake tier. Requesters using stake tier as a quality signal must be aware until Phase 7 lands the real stake-bond verification.

### 8.3 Listing signature verification

Every listing is Ed25519-signed by the provider. Unsigned or badly-signed listings are dropped silently by the directory. Closes the "adversary publishes listings in someone else's name" attack.

### 8.4 Price-range sanity check

Requesters should reject listings with implausible prices (e.g., `price_per_shard_ftns < 0.001` suggests a provider running a loss-leader attack to attract requests and then degrade service). Phase 3 does NOT enforce this — it's a requester-side policy. Reasonable default: `DispatchPolicy.min_price_per_shard_ftns=0.01`.

---

## 9. Data Model

New files:

| File | Contents |
|---|---|
| `prsm/marketplace/__init__.py` | package marker |
| `prsm/marketplace/listing.py` | `ProviderListing` dataclass + signing payload helper + `sign_listing()` + `verify_listing()` |
| `prsm/marketplace/advertiser.py` | `MarketplaceAdvertiser` (provider-side broadcaster) |
| `prsm/marketplace/directory.py` | `MarketplaceDirectory` (requester-side pool aggregator) |
| `prsm/marketplace/policy.py` | `DispatchPolicy` dataclass |
| `prsm/marketplace/filter.py` | `EligibilityFilter` (pure function + class) |
| `prsm/marketplace/reputation.py` | `ReputationTracker` + `ProviderReputation` dataclass |
| `prsm/marketplace/price_handshake.py` | quote/ack helpers + `negotiate_price()` |
| `prsm/marketplace/orchestrator.py` | `MarketplaceOrchestrator` |
| `prsm/marketplace/errors.py` | `MissingAttestationError`, `NoEligibleProvidersError`, `PriceQuoteRejectedError` |

Modified files:
- `prsm/node/gossip.py` — add `GOSSIP_MARKETPLACE_LISTING` constant.
- `prsm/node/compute_provider.py` — add `preempt_inflight_shards()` method for Phase 2.1 Line A wiring. Add handler for `shard_price_quote_request` subtype.
- `prsm/node/node.py` — bootstrap `MarketplaceAdvertiser` + `MarketplaceDirectory` + `ReputationTracker` + `MarketplaceOrchestrator` in `initialize()`. Wire to existing `_payment_escrow`, `remote_shard_dispatcher`, `transport`, `gossip`.

New tests (all under `tests/unit/` and `tests/integration/`):
- `test_provider_listing.py` — sign/verify roundtrip, stale eviction, malformed rejection.
- `test_marketplace_directory.py` — gossip ingestion, TTL eviction, duplicate replacement.
- `test_eligibility_filter.py` — each filter short-circuit case.
- `test_price_handshake.py` — quote-ack flow, expired quote, ceiling violation.
- `test_reputation_tracker.py` — score math, rolling window, latency percentiles, new-provider neutral baseline.
- `test_marketplace_orchestrator.py` — happy path with mocked dispatcher, preemption re-routing, no-eligible-providers, missing-attestation.
- `test_phase3_marketplace_e2e.py` — 3-node integration, price-filtered dispatch end-to-end.

---

## 10. Non-Goals

Explicit exclusions for Phase 3 (with deferral targets):

| Concern | Deferred to |
|---|---|
| On-chain per-dispatch settlement | Phase 3.1 (batch settlement) |
| On-chain node registry | Phase 3.1+ |
| Auction price discovery | Phase 6 (tiered economies) |
| Size-tiered pricing | Phase 6 |
| Time-of-day dynamic pricing | Phase 6 |
| FLOP-metered pricing | Phase 6 (requires trusted benchmark infra) |
| Stake-slashing for SLA violations | Phase 7 (slashing contract) |
| Federated reputation gossip | Phase 6 |
| TEE quote verification (DCAP/KDS/revocation) | Phase 2.1 follow-up |
| Cross-inference requester anonymity | Phase 6 (onion-routed dispatch) |
| Multi-tenancy / resource isolation | Phase 5 (node hardening) |

---

## 11. Acceptance Criterion

**Roadmap-level:** "A 3-node local cluster, with one requester and two providers advertising different prices + capabilities, successfully runs a sharded inference end-to-end where:**

1. Requester's MarketplaceOrchestrator discovers both providers via gossip.
2. EligibilityFilter with `max_price_per_shard_ftns=0.05` correctly excludes the provider advertising 0.10 FTNS/shard and includes the one advertising 0.03 FTNS/shard.
3. Price handshake completes for all shards.
4. Shards execute via Phase 2 dispatcher.
5. All receipts verify (Phase 2 binding checks).
6. Reputation tracker correctly records success count, latency percentiles.
7. Assembled output matches local-baseline bit-identically.
8. Escrow state: N escrows RELEASED (for N shards), 0 REFUNDED, ledger balances correct.

This test lands as `tests/integration/test_phase3_marketplace_e2e.py` in Task 7.

---

## 12. Rollout Plan

Phase 3 ships as a single version bump (v0.36.0 on PyPI, similar to Phase 2's shipping pattern) after Task 8's codex gate passes.

No breaking changes to Phase 2 consumers — the MarketplaceOrchestrator is an *optional* caller of RemoteShardDispatcher. Direct-dispatch consumers (Phase 2 Task 6's `_tensor_remote_dispatch` in `PRSMNode.initialize`) continue to work unchanged; they can migrate to MarketplaceOrchestrator at their convenience.

After Phase 3 ships, the natural follow-ons are:
- **Phase 3.1**: on-chain batch settlement (reuses Phase 1.3 infrastructure once mainnet deploy completes).
- **Phase 3.2**: on-chain provider registry anchor for high-value long-lived T3 operators (opt-in).
- **Phase 4**: see roadmap — likely federation-layer work.

---

## 13. Open Questions

None P0. P1/P2 to resolve during plan drafting:

- **Q1** (P2): Should `MarketplaceAdvertiser` auto-shutdown a listing when `_current_jobs >= max_concurrent_jobs`? Proposed answer: yes, re-broadcast with `capacity_shards_per_sec=0` until capacity frees up. Plan Task 1 should cover.
- **Q2** (P2): Reputation bootstrapping — should new providers with 0 history be blocked from high-stake-tier listings until they prove themselves? Proposed answer: no; stake tier is a separate signal. A new provider with 50k FTNS bonded on "critical" is genuinely different from a new provider on "open."
- **Q3** (P3): Gossip topic versioning — add `"v": 1` field to ProviderListing for future schema evolution? Proposed answer: yes, cheap insurance.

---

**Design doc authored:** 2026-04-20. Plan doc: see `docs/2026-04-20-phase3-marketplace-plan.md`.

---

## 14. Enabling marketplace advertising (sp1459 — operator config)

`MarketplaceAdvertiser` had a full lifecycle from Phase 3 but was never constructed in prod
(`compute_provider._marketplace_advertiser` stayed `None`), so no node broadcast a listing and the
whole selection machinery — directory → candidate pool → stake-weighted aggregator selection (sp1457)
→ the shard price-quote handler — fired on an empty directory. sp1459 wires the node lifecycle
(`build_marketplace_advertiser_from_env` → constructed on a FULL/COMPUTE node, started in
`Node.start()`, stopped in `Node.stop()`). It is **opt-in and fail-safe**: unless the operator sets
`PRSM_MARKETPLACE_ADVERTISE`, the node is unchanged.

To offer a compute node as marketplace supply, set on the provider node:

| Env var | Default | Meaning |
| --- | --- | --- |
| `PRSM_MARKETPLACE_ADVERTISE` | *(off)* | `1`/`true`/`yes`/`on` enables broadcasting. Master switch. |
| `PRSM_MARKETPLACE_PRICE_PER_SHARD_FTNS` | `1.0` | Advertised price per shard. |
| `PRSM_MARKETPLACE_CAPACITY_SHARDS_PER_SEC` | `1.0` | Advertised throughput (auto-drops to 0 when at `max_concurrent_jobs`). |
| `PRSM_MARKETPLACE_MAX_SHARD_BYTES` | `8388608` | Largest shard this node accepts. |
| `PRSM_MARKETPLACE_DTYPES` | `float16,float32` | Comma-separated supported dtypes. |
| `PRSM_MARKETPLACE_STAKE_TIER` | `open` | Claimed tier. Note: an unbound tier claim gains **no** selection weight (sp1457) — bond stake + supply a binding below to be weighted by real stake, and it is required for pool admission under `PRSM_REQUIRE_STAKE_BINDING`. |
| `PRSM_MARKETPLACE_TEE_CAPABLE` | *(off)* | Advertise TEE capability. |
| `PRSM_MARKETPLACE_TTL_SECONDS` | `300` | Listing TTL in the directory (keep > 2× rebroadcast interval). |
| `PRSM_MARKETPLACE_REBROADCAST_INTERVAL_SEC` | `90` | Re-broadcast cadence (jittered ±25%). |

**Stake binding (sp1457, optional but needed for real-stake weighting).** Attach the provider↔stake
binding one of two ways:

- **Pre-produced (recommended — the stake key never touches the node):** produce the pair out of band
  with `sign_stake_binding(provider_id, stake_eth_private_key)` (provider_id = the node's `node_id`),
  then set `PRSM_STAKE_ETH_ADDRESS` + `PRSM_STAKE_BINDING_SIG`.
- **Derive at startup:** set `PRSM_STAKE_ETH_KEY` to the stake-holding eth private key and the node
  signs the binding on boot. Convenient, but places the stake key in the node's environment.

Without a binding the node still advertises a valid listing; it is simply weighted at the tier label
(and excluded from the aggregator pool if selectors run with `PRSM_REQUIRE_STAKE_BINDING=1`).
