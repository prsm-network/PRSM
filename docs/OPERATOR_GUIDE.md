# PRSM Node Operator Guide

A practical guide for DevOps engineers and system administrators running persistent PRSM nodes.

## Who This Guide Is For

This guide is for you if you:
- Are a DevOps engineer deploying PRSM infrastructure
- Run servers and want to contribute to the PRSM network
- Need to deploy a PRSM node for your organization
- Want to participate in the network as a compute provider

If you're looking to use PRSM as an end-user, see the [Participant Guide](PARTICIPANT_GUIDE.md) instead.

---

## Quick Start: Docker Compose (Recommended)

The fastest way to run a production PRSM node.

### Prerequisites

- Docker 20.10+ and Docker Compose v2
- 4GB+ RAM (8GB+ recommended)
- 50GB+ disk space (SSD recommended)
- Open ports: 8000 (API), 8765 (P2P)

### Step 1: Get the Code

```bash
git clone https://github.com/prsm-network/PRSM.git
cd PRSM
```

### Step 2: Configure Environment

```bash
# Copy the example environment file
cp config/secure.env.template .env

# Edit the configuration
nano .env
```

**Required settings:**

```bash
# Security (REQUIRED - generate your own!)
SECRET_KEY=<generate with: openssl rand -hex 32>
# Management-API key — REQUIRED whenever the API is network-exposed (Docker
# port-mapping / api_host=0.0.0.0). The node fail-closes a public API bind
# without it (sp1011). Not needed for a default loopback API.
PRSM_NODE_API_KEY=<generate with: python3 -c 'import secrets;print(secrets.token_urlsafe(32))'>

# Database
DATABASE_URL=sqlite:///./prsm_node.db

# Initial Admin
INITIAL_ADMIN_EMAIL=admin@example.com
INITIAL_ADMIN_PASSWORD=<secure-password>

# P2P Network — production bootstrap is hosted on DigitalOcean.
# `prsm/node/bootstrap.py:_DEFAULT_BOOTSTRAP` already points at this; you
# only need to set this env var if you want to override or pin secondaries.
P2P_BOOTSTRAP_NODES=wss://bootstrap-us.prsm-network.com:8765
```

### Step 3: Launch

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f prsm-api
```

### Step 4: Verify

```bash
# Health check
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "version": "1.7.0", ...}
```

---

## System Requirements

### Minimum Configuration

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| CPU | 2 cores | 8 cores | More cores = more concurrent queries |
| RAM | 4 GB | 32 GB | AI workloads benefit from more RAM |
| Storage | 50 GB SSD | 500 GB NVMe | Fast storage improves content-store + BitTorrent shard read throughput |
| Network | 10 Mbps | 100 Mbps | Low latency is critical for P2P |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS | Also supports macOS, Windows |

### Hardware Recommendations by Use Case

| Use Case | CPU | RAM | GPU | Storage |
|----------|-----|-----|-----|---------|
| Light Node | 4 cores | 8 GB | None | 100 GB SSD |
| Standard Node | 8 cores | 16 GB | Optional | 250 GB NVMe |
| Compute Provider | 16+ cores | 32+ GB | NVIDIA 8GB+ VRAM | 500 GB NVMe |
| Bootstrap Node | 8 cores | 16 GB | None | 100 GB SSD |

---

## Installation Methods

### Option 1: Docker Compose (Recommended for Production)

Best for: Most production deployments

```yaml
# docker-compose.yml
# Note: the `version:` key is obsolete in Docker Compose v2 — omit it.

services:
  prsm-api:
    image: prsm/api:latest
    ports:
      - "8000:8000"
      - "8765:8765"
    environment:
      - SECRET_KEY=${SECRET_KEY}
      - DATABASE_URL=postgresql://prsm:${POSTGRES_PASSWORD}@db:5432/prsm
      - REDIS_URL=redis://redis:6379
      # sp1017/sp1018 — the management API binds api_host (default 127.0.0.1 =
      # container-loopback, UNREACHABLE via the 8000:8000 port-map). Expose it
      # on the container interface AND authenticate it (a public API bind
      # without a key fail-closes at startup, sp1011).
      - PRSM_API_HOST=0.0.0.0
      - PRSM_NODE_API_KEY=${PRSM_NODE_API_KEY}
    volumes:
      - prsm-data:/data
      - ./config:/app/config
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=prsm
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}  # set in .env, never commit
      - POSTGRES_DB=prsm
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data
    restart: unless-stopped

volumes:
  prsm-data:
  postgres-data:
  redis-data:
```

```bash
# Launch
docker-compose up -d
```

### Option 2: Systemd (Bare Metal)

Best for: Servers where you want direct control

```bash
# Install dependencies
sudo apt update
sudo apt install python3.11 python3.11-venv

# NOTE: as of v1.7.0, IPFS is no longer used. Native content-store +
# BitTorrent transport replaced it during the 2026-05-07 migration.
# See `prsm/data/native_storage/`. Existing nodes upgrading from <v1.7
# can remove the `ipfs` daemon and its repo after the alembic 016
# migration runs (renames `ipfs_cid` columns → `content_cid`).

# Clone and setup
git clone https://github.com/prsm-network/PRSM.git /opt/prsm
cd /opt/prsm
python3.11 -m venv venv
source venv/bin/activate
pip install -e .

# Create systemd service
sudo nano /etc/systemd/system/prsm.service
```

**prsm.service:**
```ini
[Unit]
Description=PRSM Node
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=prsm
Group=prsm
WorkingDirectory=/opt/prsm
Environment="PATH=/opt/prsm/venv/bin"
ExecStart=/opt/prsm/venv/bin/prsm node start
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable prsm
sudo systemctl start prsm

# Check status
sudo systemctl status prsm
```

### Option 3: Kubernetes

Best for: Cloud-native deployments with auto-scaling

See `k8s/` directory for Helm charts and deployment manifests.

```bash
# Deploy with Helm
helm install prsm ./k8s/helm/prsm \
  --set secrets.secretKey=$(openssl rand -hex 32) \
  --set persistence.enabled=true
```

---

## Configuration Reference

> **Note:** post-2026-05 surface (QueryOrchestrator, Item 6 arbitration, DHT transport, mainnet contract overrides) is documented in dedicated sections below. The tables here cover the pre-existing baseline.

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `SECRET_KEY` | JWT signing key (32+ chars) | `openssl rand -hex 32` |
| `DATABASE_URL` | Database connection | `postgresql://user:pass@host/db` |

### Network Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `P2P_LISTEN_PORT` | 8765 | P2P network port |
| `P2P_BOOTSTRAP_NODES` | *none* | Comma-separated bootstrap addresses |
| `API_HOST` | 0.0.0.0 | API bind address |
| `API_PORT` | 8000 | API port |

### Performance Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONCURRENT_QUERIES` | 10 | Max parallel AI queries |
| `QUERY_TIMEOUT` | 300 | Query timeout (seconds) |
| `CACHE_TTL` | 3600 | Cache TTL (seconds) |
| `WORKER_THREADS` | 4 | Background worker threads |

### Security Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `RATE_LIMIT_REQUESTS` | 100 | Requests per minute per user |
| `RATE_LIMIT_WINDOW` | 60 | Window in seconds |
| `MAX_QUERY_COST` | 100 | Max FTNS per query |
| `CIRCUIT_BREAKER_THRESHOLD` | 5 | Failures before circuit opens |

---

## Mainnet On-chain Surface (v1.7.0+)

PRSM is live on **Base mainnet** as of 2026-05-04 (treasury / provenance) and 2026-05-07 (full audit-bundle + Phase 8 + Phase 7-storage). Operators do **not** need to set contract addresses manually for the canonical mainnet deploy — `prsm/config/networks.py` ships them pinned.

### Default mainnet contract addresses (informational)

| Contract | Address | Verified |
|---|---|---|
| FTNSTokenSimple | `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` | Basescan |
| Foundation Safe (2-of-3) | `0x91b0...5791` | Basescan |
| ProvenanceRegistry V2 | `0xe0cedDA354f99526c7fbb9b9651e12aDB2180dbf` | Basescan |
| RoyaltyDistributor (v2, canonical) | `0xfEa9aeB99e02FDb799E2Df3C9195Dc4e5323df7e` | Basescan |
| EmissionController + CompensationDistributor + StorageSlashing + KeyDistribution + audit-bundle (BSR / EscrowPool / StakeBond / Ed25519Verifier) | see `prsm/config/networks.py` MAINNET block | Basescan |

The Foundation Safe is sole owner of every contract above. Operator nodes hold no admin keys — see "On-chain Keypairs" below.

### Optional override env vars

Operators rarely need these; they exist for testnet pinning, post-incident migration, or operator-controlled deployments.

| Variable | Purpose |
|---|---|
| `FTNS_TOKEN_ADDRESS` (alias: `FTNS_CONTRACT_ADDRESS`) | Override FTNS token address |
| `PRSM_PROVENANCE_REGISTRY_ADDRESS` | Override ProvenanceRegistry; legacy V1 addr |
| `PRSM_PROVENANCE_REGISTRY_V2_ADDRESS` | **Set this to use V2 (post-2026-05-06)** |
| `PRSM_ROYALTY_DISTRIBUTOR_ADDRESS` | Override RoyaltyDistributor |
| `PRSM_FOUNDATION_SAFE` | Override Foundation Safe; rarely needed |
| `PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS` | Override PublisherKeyAnchor (DHT lane) |
| `PRSM_SETTLEMENT_REGISTRY_ADDRESS` | Override BatchSettlementRegistry |
| `PRSM_ESCROW_POOL_ADDRESS` | Override EscrowPool |
| `PRSM_STAKE_BOND_ADDRESS` | Override StakeBond |
| `PRSM_EMISSION_CONTROLLER_ADDRESS` | Override EmissionController |
| `PRSM_COMPENSATION_DISTRIBUTOR_ADDRESS` | Override CompensationDistributor |
| `PRSM_STORAGE_SLASHING_ADDRESS` | Override StorageSlashing |
| `PRSM_KEY_DISTRIBUTION_ADDRESS` | Override KeyDistribution (Tier C) |

### On-chain ingest opt-in

| Variable | Default | Description |
|---|---|---|
| `PRSM_ONCHAIN_PROVENANCE` | `0` (off) | Set to `1` to register uploads on-chain via ProvenanceRegistry. **Requires** `PRSM_PROVENANCE_REGISTRY_ADDRESS` (or V2) AND `PRSM_ROYALTY_DISTRIBUTOR_ADDRESS`. Without these the node will hard-fail at startup rather than silently skipping the on-chain step. |

---

## QueryOrchestrator + Aggregation (Ring 5 replacement)

As of 2026-05-08, the legacy `AgentForge` dispatcher has been replaced by the **QueryOrchestrator** stack. New deployments default to it; existing operators must opt in.

### Activation

| Variable | Default | Description |
|---|---|---|
| `PRSM_QUERY_ORCHESTRATOR_ENABLED` | `0` | Set to `1` to enable QueryOrchestrator on `/compute/forge`. When unset (or set to anything other than `1`/`true`/`yes`), the legacy AgentForge path is used. |

`/compute/forge` duck-type-dispatches based on `hasattr(node.agent_forge, "dispatch_query")`, so the swap is transparent to clients.

### Settlement split

When QueryOrchestrator handles a job, payment escrow is released across all swarm participants via `PaymentEscrow.release_escrow_split` (multi-recipient atomic distribution).

| Variable | Default | Description |
|---|---|---|
| `PRSM_AGGREGATOR_SHARE_BPS` | `500` (= 5%) | Aggregator's share in basis points. Remaining `10_000 - N` bps split evenly across compute participants. Operator-tunable; valid range `[0, 10_000]`. |

### Endpoints

| Endpoint | Description |
|---|---|
| `POST /compute/forge` | Submit a query. Returns 32-byte `query_id` and (in QO mode) an `AggregatedResult` block including `participants: [...]`. |
| `GET /compute/status/{job_id}` | Two-tier response: `{"history": {...}, "escrow": {...}}`. The `history` block reports IN_PROGRESS / COMPLETED / FAILED / CANCELLED from `JobHistoryStore`; the `escrow` block reports `PaymentEscrow` state. Either tier may be `null` if the job is unknown to that subsystem. |
| `POST /compute/cancel/{job_id}` | Cancel an in-progress job: marks history.status as CANCELLED + refunds any PENDING escrow. **v1 caveat:** in-flight Python coroutines are NOT interrupted — cancellation marks intent + refunds the budget. If the coroutine completes successfully later, its release_escrow_split call race-loses against the REFUNDED escrow. Returns 503 if neither subsystem wired, 404 if neither has the job, 409 if already terminal, 200 with `{job_id, history_cancelled, escrow_refunded, refund_amount_ftns}` on success. |
| `GET /compute/status/{job_id}/stream` | SSE-streaming sibling of `/compute/status`. Polls JobHistoryStore + PaymentEscrow at `PRSM_STATUS_STREAM_POLL_SEC` (default 0.5s); emits `event: status` on every snapshot change (de-duplicated by JSON equality); closes with `event: terminal` on terminal status (history COMPLETED/FAILED/CANCELLED OR escrow RELEASED/REFUNDED) OR after `PRSM_STATUS_STREAM_TIMEOUT_SEC` (default 1800s). Clients can re-subscribe past timeout. Same 503/404 semantics as the GET sibling. |
| `GET /compute/jobs` | Paginated operator-side job list. Query params: `status` (in_progress/completed/failed/cancelled), `limit` (1..100, default 50), `offset` (default 0). Returns `{jobs: [...], total: N, offset: X, limit: Y}` ordered most-recent-first by `started_at`. 503 if JobHistoryStore not wired; 422 on validation errors. Backs the `prsm_jobs_list` MCP tool. |
| `GET /wallet/escrows` | Operator-side escrow summary. Query params: `address` (optional 0x override; defaults to node's connected wallet), `include_terminal` (default false; when true returns RELEASED + REFUNDED for audit). Returns `{address, escrows: [...], total: N, total_locked_ftns: X, include_terminal: bool}`. 503 if PaymentEscrow or ftns_ledger not wired. Backs the `prsm_escrow_summary` MCP tool. |
| `GET /wallet/spend` | Sum FTNS spent on completed compute jobs over the last N days. Query params: `days` (1..365, default 30), `address` (optional 0x override). Counts RELEASED escrows only — REFUNDED + PENDING excluded. Returns `{address, days, total_spent_ftns, escrows_count}`. Backs the `prsm_spend_summary` MCP tool — useful for cost-tracking dashboards + budget reconciliation. |
| `POST /compute/cleanup-stale` | Manual trigger for `PaymentEscrow.cleanup_expired_escrows` — refunds PENDING escrows whose age exceeds `default_timeout`. Returns `{cleaned: N}`. Use after lowering `PRSM_ESCROW_TIMEOUT_SEC` for immediate effect, or to drain stuck escrows pre-maintenance. 503 if PaymentEscrow not wired; 502 if cleanup raises. Backs the `prsm_cleanup_stale_escrows` MCP tool. |
| `GET /content/arbitration/queue` | List pending content-attribution disputes from FilesystemArbitrationQueue (PRSM-PROV-1 Item 6). Returns `{pending: [...], total}` with each record containing new_cid, candidate_parent_cid, similarity, fingerprint_kind, flagged_at, proposal_id. 503 if queue not wired; 502 if list_pending raises. Backs the `prsm_arbitration_status` MCP tool. |
| `GET /content/arbitration/queue/{record_id}` | Detail view of a single arbitration record. Returns `{record, resolution, status}` where `status` is `"resolved"` (resolution present) or `"pending"`; resolution payload includes `decision` (upheld_parent/rejected_parent/insufficient) + `by_council` (council members who signed). Council members use this to fetch full context before signing on-chain governance proposals. 503 if queue not wired; 404 if record_id unknown. |
| `POST /content/arbitration/preview-resolution` | **Composer-only** preview of what `queue.resolve()` WOULD do for the given (record_id, decision, by_council). DOES NOT call resolve(). Returns `{status: "DRY_RUN", record, proposed, current_resolution, conflict_with_existing, note}`. Council members compose a resolution from the AI side panel, see whether it conflicts with any existing resolution (idempotent-equal vs conflicting-different), then sign the on-chain governance proposal separately. Local-resolve auth model is pending council ratification — the composer-only path is the safe surface until that's settled. 503/404/422 standard error codes. |

### Per-requester rate limits (DoS protection, ships 2026-05-09)

Two independent token-bucket caps, one per compute endpoint. Default unset → no limiting (v1 behavior preserved bit-identically). When set, each requester's bucket starts with N tokens; refills at N tokens/sec; capped at N (no burst beyond steady-state). Empty bucket → HTTP 429 with `Retry-After` header indicating seconds until next token. Non-numeric / zero / negative env values silently disable rate limiting (log WARN).

| Env var | Endpoint | Bucket name |
|---|---|---|
| `PRSM_FORGE_MAX_RPS_PER_REQUESTER` | `POST /compute/forge` | `forge` |
| `PRSM_INFERENCE_MAX_RPS_PER_REQUESTER` | `POST /compute/inference` | `inference` |

The two buckets are **independent** — draining the forge bucket has no effect on inference and vice versa. Operators can tune the two caps separately based on resource profiles (forge runs full Ring 1-10 pipeline; inference is single-model TEE-attested).

Use these when running a public node where a single misbehaving client could otherwise saturate the compute pipeline.

### Per-job duration cap (`PRSM_FORGE_MAX_DURATION_SEC`, ships 2026-05-09)

Production-reliability feature: caps how long a single `/compute/forge` job can stay IN_PROGRESS before being reaped + its escrow refunded. Without this, a runaway compute job holds operator FTNS escrowed indefinitely until the periodic_cleanup_escrows task fires (which only checks escrow age, not job age).

Implementation: separate background reaper (`JobReaper` async task) that scans `JobHistoryStore` every `PRSM_FORGE_REAPER_INTERVAL_SEC` (default 60s) for IN_PROGRESS records older than the cap. For each:
1. Marks the history record FAILED with error="duration cap exceeded ({N}s)"
2. Refunds the associated escrow (if PENDING)

Decoupled from the `/compute/forge` body — handler keeps existing semantics, reaper provides timeout safety net independently. Reaper crash visibility comes from the daemon-task probe surface (`/health/detailed.heartbeat_scheduler` pattern; reaper subsystem entry shipped in a follow-on commit).

Configuration:
- `PRSM_FORGE_MAX_DURATION_SEC` — duration cap in seconds (positive float; unset → reaper disabled, v1 behavior preserved)
- `PRSM_FORGE_REAPER_INTERVAL_SEC` — scan cadence (default 60s)

Race-condition robustness: refund_escrow raising EscrowAlreadyFinalizedError (legitimate release happened concurrently) is treated as race-loss; the history record is still marked FAILED. Future `/compute/forge` callers seeing FAILED status see the duration-cap reason in `error`.

### Per-job FTNS cap (`PRSM_MAX_FTNS_PER_JOB`, ships 2026-05-09)

Cost-control feature for operators worried about misbehaving AI agents draining their balance via a single oversized request. Set `PRSM_MAX_FTNS_PER_JOB=N` (any positive float) to enforce a per-job ceiling. Requests with `budget_ftns > N` return HTTP 422 with a clear message before any escrow is locked.

Behavior:
- Unset (default) → no cap, v1 behavior preserved bit-identically.
- Non-numeric / zero / negative values → cap silently disabled (log WARN).
- Idempotent replays via `Idempotency-Key` bypass the cap (the cached job already settled).

Tighten + restart to apply changes; existing IN_FLIGHT requests aren't re-checked.

### Idempotency-Key on `/compute/forge` (ships 2026-05-09)

Operators retrying a failed POST due to network blip can pass an `Idempotency-Key` HTTP header (any opaque string). On first request the key is registered against the resulting `job_id` in JobHistoryStore. Subsequent requests with the same key return the cached job's status (HTTP 200 with `status: "idempotent_replay"`) without locking a new escrow or re-running compute.

The mapping persists across node restarts when `PRSM_JOB_HISTORY_DIR` is set — operators get retry-safety even through restarts. When the store isn't wired, idempotency is silently disabled (the request proceeds normally).

Edge cases:
- **Unknown key:** falls through to normal forge logic + registers the new mapping on completion.
- **Lookup raises** (corrupt index): logged at WARN, request proceeds normally.
- **Dangling index entry** (job evicted from LRU + not on disk): treated as cache miss, proceeds normally.

The `Idempotency-Key` cache hit short-circuits BEFORE the `agent_forge`-not-wired 503, so operators can recover prior job results even after a forge subsystem failure.

### Prometheus metrics (`GET /metrics`, ships 2026-05-09)

Standard Prometheus text/plain exposition format for ops dashboards (Grafana, alertmanager, VictoriaMetrics, etc.). No new tracking infra — gauges read live from existing node state.

Emitted gauges:
- `prsm_pending_escrow_count` — count of PENDING escrows
- `prsm_total_locked_ftns` — sum of FTNS locked in PENDING escrows
- `prsm_job_history_size` — JobHistoryStore record count
- `prsm_claimable_royalties_wei` — pull-payment balance in wei
- `prsm_arbitration_pending_count` — pending content-attribution disputes
- `prsm_node_up` — always 1 (canonical "up" indicator for scrapers)

Per-metric fail-soft: a subsystem RPC error (e.g., royalty client hits an unreachable Base RPC) logs at WARN and omits that gauge rather than 500-ing the endpoint. Scrapes always succeed.

Sample scrape config:

```yaml
scrape_configs:
  - job_name: prsm
    metrics_path: /metrics
    static_configs:
      - targets: ['localhost:8000']
```

### Webhook delivery primitive (`prsm.node.webhook_delivery`, ships 2026-05-09)

`WebhookDeliverer` is the operator-facing event-hook primitive. Future commits subscribe it to specific event sources (daemon crashes, escrow leaks, etc.); this module ships the delivery + signing + retry primitives standalone.

Wire format on POST to operator-config'd URL:

```
POST <webhook_url>
Content-Type: application/json
X-PRSM-Event: <event_name>
X-PRSM-Signature: sha256=<hmac>     (when shared secret set)
User-Agent: prsm-node/<version>

<canonical-JSON body>
```

Receiving systems verify by recomputing HMAC-SHA256 over the raw body using the shared secret and comparing in constant time. Reference recipe (Python):

```python
import hmac, hashlib
expected = "sha256=" + hmac.new(SECRET.encode(), request.body, hashlib.sha256).hexdigest()
if not hmac.compare_digest(expected, request.headers["X-PRSM-Signature"]):
    abort(403)
```

Retry behavior:
- `max_attempts` (default 3) with exponential backoff (1s, 2s, 4s, ...)
- per-attempt `timeout_seconds` (default 10s)
- 2xx → immediate success
- 408/429/5xx → retry with backoff
- other 4xx → give up immediately (operator misconfiguration; retry-storming a wrong URL helps no one)
- exception (DNS / connect / etc.) → retry

Final failure returns `DeliveryResult(success=False, attempts, status_code, error)` for ops triage.

### CORS allowlist (`PRSM_ALLOWED_ORIGINS`, ships 2026-05-09)

Production-hardening for nodes serving browser-based clients (operator dashboards, prsm-ui, etc.). Operator declares the explicit list of origins permitted to make cross-origin requests; everything else gets blocked at the CORS layer before reaching any endpoint.

```bash
# Single origin
export PRSM_ALLOWED_ORIGINS="https://dash.example.com"

# Multiple origins (comma-separated; whitespace trimmed)
export PRSM_ALLOWED_ORIGINS="https://dash.example.com, https://ops.example.com"
```

Behavior:
- **Env unset / whitespace-only:** permissive `*` allowlist — preserves v1 dev-friendly behavior bit-identically. Acceptable for local dev; **production deploys MUST set this explicitly to restrict**.
- **Listed origin:** browser receives `Access-Control-Allow-Origin: <origin>` and the response is delivered to JS code.
- **Unlisted origin:** server still responds 200 but omits the ACAO header. Browser blocks the response from reaching the requesting JS, satisfying the same-origin policy.
- **Preflight (OPTIONS):** allowed origins receive 200 + ACAO + ACAM headers; unlisted origins receive 200 with no CORS headers.
- **Credentials:** when an explicit allowlist is set, `Access-Control-Allow-Credentials: true` is also set (allows browser-side `credentials: 'include'`).

This middleware applies to ALL endpoints (the security-hardening middleware shipped earlier sits behind it); CORS rejection happens before any other middleware runs.

### Security hardening configuration (ships 2026-06-04)

A network-/data-plane security sweep (sprints 1002–1014) added the knobs below. The
**fail-closed public-bind posture is the one most likely to affect an upgrade** — read it first.

**Management-API authentication (sp1011 + sp1017).** The protected money + KYC endpoints
are authenticated only when `PRSM_NODE_API_KEY` is set. The management API binds **loopback
(`127.0.0.1`) by default** (`api_host`, sp1017), so a default `prsm node start` is NOT
network-exposed and boots fine **with no key** — local dev and reverse-proxy-fronted
deployments are unaffected. (The `0.0.0.0` you may see in config is the separate **P2P**
transport bind, which must stay reachable.)

If you **intentionally expose the management API** (set `api_host=0.0.0.0` / `PRSM_API_HOST=0.0.0.0`,
e.g. Docker port-mapping) you MUST also authenticate it, or the node **refuses to start**
(sp1011 fail-closed). Do ONE of:

```bash
# Recommended — authenticate the money/KYC endpoints, then expose the API
export PRSM_NODE_API_KEY="$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')"
export PRSM_API_HOST=0.0.0.0
# OR keep the API on loopback (default) behind an authenticating reverse proxy (no key needed)
# OR (NOT recommended — only if fronted by a firewall/security group) expose unauthenticated:
export PRSM_API_HOST=0.0.0.0 && export PRSM_ALLOW_INSECURE_PUBLIC_BIND=1
```

**Content-retrieval + gossip data-plane bounds** (all have safe defaults; tune only if needed):

| Variable | Default | Purpose |
|----------|---------|---------|
| `PRSM_MAX_GATEWAY_FETCH_BYTES` | `2147483648` (2 GiB) | Hard ceiling on a single peer-supplied GATEWAY HTTP fetch (sp1002 — caps OOM/over-large bodies; the per-fetch bound is the advertised size). |
| `PRSM_MAX_KNOWN_PEERS` | `2048` | Cap on the peer routing table (sp1005/1009 — bounds memory + eclipse magnitude). |
| `PRSM_MAX_PROVIDERS_PER_CID` | `64` | Cap on providers tracked per content id (sp1007 — bounds advertise-flood/eclipse). A real CID has a handful of replicas. |
| `PRSM_GOSSIP_DEDUP_WINDOW_SEC` | `86400` (24 h) | Gossip-layer replay-barrier window (sp1008 — drops a signed frame replayed past the transport's ~300 s nonce window). |
| `PRSM_GOSSIP_DEDUP_MAX` | `100000` | LRU cap on the gossip replay-barrier set. |

**Bootstrap TLS (sp1006).** The bootstrap WSS client now **verifies the server certificate by
default** (the live `bootstrap-*.prsm-network.com` fleet uses valid public certs, so no action
is needed). For a **self-signed dev bootstrap**, pin its CA or disable verification explicitly:

```bash
export PRSM_BOOTSTRAP_TLS_CA_FILE=/path/to/bootstrap-ca.pem   # pin a custom CA (still verifies)
export PRSM_BOOTSTRAP_TLS_INSECURE=1                          # disable verification (dev only; MITM-able)
```

**libp2p transport guard (sp1010).** The default WebSocket transport carries full
origin-authentication. The alternative libp2p transport does **not** yet, so selecting
`PRSM_TRANSPORT_BACKEND=libp2p` logs a CRITICAL warning; set
`PRSM_FORBID_UNAUTHENTICATED_LIBP2P=1` to make the node refuse to start on that backend until
the auth is ported. Production should use `PRSM_TRANSPORT_BACKEND=websocket`.

**Wallet session tokens (sp1013/1014 — opt-in).** `/siwe/verify` now returns a wallet-bound
`session_token`. To require it on the wallet-owner read endpoints
(`/api/v1/auth/wallet/{binding,bindings,devices/earnings,balance}`) so a caller can only read
the wallet it proved control of:

```bash
export PRSM_WALLET_SESSION_REQUIRED=1
export PRSM_WALLET_SESSION_SECRET="0x$(python3 -c 'import secrets;print(secrets.token_hex(32))')"  # stable 32-byte hex; random per-process if unset
```

Enforcement defaults **off** so existing clients keep working until they send the token
(default-on is a coordinated frontend/CLI follow-on). Do **not** set the secret to an empty
`0x` placeholder — a configured-but-empty value is treated as unset (random fallback) rather
than collapsing to a guessable constant (sp1014).

### `GET /audit/recent` — operator audit log (ships 2026-05-09)

In-memory ring buffer of state-changing API requests (non-GET). Each entry: `{timestamp, method, path, requester, status_code, request_id}`. Default 1024-entry capacity (older entries auto-evicted). Returns most-recent-first; pagination via `?limit=50&offset=0`.

```bash
# 50 most recent state changes
curl http://localhost:8000/audit/recent

# Older entries (page 2)
curl 'http://localhost:8000/audit/recent?limit=50&offset=50'
```

Use cases:
- "Did some user just call /compute/cancel?" → grep recent for path=/compute/cancel
- "What happened around the time the alert fired?" → grep request_id from logs
- "Trace the full sequence of writes from a specific requester" → filter by requester

503 if `_audit_log` not wired (test fixtures, single-shot scripts); 422 on invalid `limit` (must be 1..1000) / `offset` (must be ≥ 0).

**Filesystem persistence (v2, ships 2026-05-09):** Set `PRSM_AUDIT_LOG_DIR=/path/to/dir` to enable disk-backed continuity. Each `append()` writes through to a timestamp-prefixed JSON file under the directory; on node startup, the ring scans the directory and rehydrates entries sorted by timestamp ascending (deque maxlen evicts oldest if disk has more than `max_entries`).

Filename format: `{timestamp:020.6f}-{hash8(request_id)}.json` — sortable by name, collision-resistant. Corrupt JSON files are logged + skipped (fail-soft) rather than crashing startup. When `PRSM_AUDIT_LOG_DIR` unset, v1 in-memory-only behavior is preserved bit-identically.

**Disk retention (ships 2026-05-09):** Set `PRSM_AUDIT_LOG_RETENTION_DAYS=N` (positive float) to delete on-disk entries older than N days at node startup. Without this, persisted logs grow unbounded over time. Pruning happens once at startup; restart the node to re-prune. Set both `PRSM_AUDIT_LOG_DIR` + `PRSM_AUDIT_LOG_RETENTION_DAYS` for full disk-backed audit log with bounded retention. Non-numeric / zero / negative retention values silently disable pruning (log WARN).

### `GET /info` — static node metadata (ships 2026-05-09)

Single read-only endpoint surfacing version, active network, chain_id, node identity, and canonical contract addresses for the active `PRSM_NETWORK`. Useful for operator triage + integration code needing to know "what network is this node on" without parsing `/health/detailed`.

Sample mainnet response:

```json
{
  "node_id": "node-prod-east-1",
  "api_version": "0.24.0",
  "network": "mainnet",
  "chain_id": 8453,
  "canonical_addresses": {
    "ftns_token": "0x5276a3756C85f2E9e46f6D34386167a209aa16e5",
    "provenance_registry": "0xdF470BFa9eF310B196801D5105468515d0069915",
    "provenance_registry_v2": "0xe0cedDA354f99526c7fbb9b9651e12aDB2180dbf",
    "royalty_distributor": "0xfEa9aeB99e02FDb799E2Df3C9195Dc4e5323df7e",
    "foundation_safe": "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791"
  }
}
```

Always returns 200 with at least `node_id` + `api_version`. Network + chain_id + canonical_addresses surfaced when the active PRSM_NETWORK has a known config; omitted on local / unknown networks rather than crashing.

### Health probes

- `GET /health` — minimal load-balancer probe; returns `{status: "ok", node_id}` without subsystem checks. Stays bit-identical to v1 to avoid breaking external monitors.
- `GET /health/detailed` (ships 2026-05-09) — structured per-subsystem readiness for ops monitoring. Returns `{status, node_id, subsystems: {...}}` where `status` is one of:
  - `healthy` — all wired subsystems operational
  - `degraded` — core (FTNS ledger + payment escrow) works but optional subsystems (job history / royalty distributor) unavailable or erroring
  - `unhealthy` — core subsystem missing or erroring
  
  Each subsystem entry has `{available, status, ...details, error?}`. Subsystem probes are fail-soft — an exception in one subsystem's check (e.g., royalty RPC down) surfaces in that entry's `error` field rather than 500-ing the endpoint. Use this against your ops alerting (PagerDuty, Grafana, etc.) — alarm on `status != "healthy"` for paging, `status == "unhealthy"` for high-sev.

  **Canonical-address match check (added 2026-05-09 post-A-08 ceremony):** for subsystems that wire on-chain contracts, the entry surfaces `wired_address`, `canonical_address` (from `prsm/config/networks.py` for the active `PRSM_NETWORK`), and `canonical_match: bool`. Operators get an instant verification signal post-migration — e.g., after the v1→v2 RoyaltyDistributor migration, an operator pinned to v1 via `PRSM_ROYALTY_DISTRIBUTOR_ADDRESS` env override sees `canonical_match: false` + the canonical v2 address surfaced for comparison. When `PRSM_NETWORK` is set to a value with no canonical addresses (e.g., `local`), `canonical_*` fields are omitted gracefully. Currently shipped for `royalty_distributor`; same pattern extends to other contract subsystems.

### PaymentEscrow timeouts (configurable, ships 2026-05-09)

Two env vars tune the auto-refund behavior for stale escrows:

- `PRSM_ESCROW_TIMEOUT_SEC` — escrow expiry duration. PENDING escrows older than this are auto-refunded by the periodic cleanup task. Default `3600` (1h). Operators with high-throughput workloads (jobs settling in seconds) want smaller values; long-running compute (multi-hour inference) wants larger.
- `PRSM_ESCROW_CLEANUP_INTERVAL_SEC` — how often the cleanup task wakes up to scan for expired escrows. Default `600` (10min).

Both env vars fail-soft: non-numeric / zero / negative values log a WARN and fall back to the default rather than crashing node startup. Constructor args (`default_timeout`, `cleanup_interval`) take precedence over env vars when PaymentEscrow is constructed programmatically (e.g. in tests).

### JobHistoryStore

In-memory LRU-bounded store (default 1024 entries). Records `IN_PROGRESS` → `COMPLETED` / `FAILED` transitions for `/compute/forge` jobs.

**Filesystem persistence (v2, ships 2026-05-09):** Set `PRSM_JOB_HISTORY_DIR=/path/to/dir` to enable disk-backed history. Each `put()` writes through to a SHA-256-named JSON file under the directory; on node startup, the store scans the directory and rehydrates the in-memory LRU sorted by `started_at` (most-recently-started ends up LRU-newest). On `get()` miss in memory, the store falls back to disk — even LRU-evicted records remain retrievable until the operator manually cleans up the directory.

Filename sanitization is hash-based (SHA-256 first 16 hex chars), so caller-supplied `job_id` cannot escape the persist directory regardless of content. Corrupt JSON files are logged + skipped (fail-soft) rather than crashing startup. Disk-write failures are logged at WARN; the in-memory record remains authoritative.

When `PRSM_JOB_HISTORY_DIR` is unset, the v1 in-memory-only behavior is preserved bit-identically — process restart clears history (`escrow` tier of `/compute/status` survives, `history` does not).

---

## Per-Content-Type Dedup + Arbitration (PRSM-PROV-1 Item 6)

Three-band attribution pipeline shipped 2026-05-08: clear-uphold (≥ derivative threshold) / disputed-band (between `arbitration_floor` and `derivative`) / clear-reject (< `arbitration_floor`). Disputed-band records land in an arbitration queue and (optionally) a token-weighted governance proposal.

### Activation tiers

The pipeline is layered. Operators choose how deep to go:

| Tier | What runs | Activation |
|---|---|---|
| 0 | Three-band routing only | Always on (no env vars) |
| 1 | + Filesystem queue persistence | `~/.prsm/arbitration_queue/` (created automatically on first disputed record) |
| 2 | + Governance proposal sink (token-weighted voting) | Set `PRSM_ARBITRATION_PROPOSER_ID=<your-proposer-id>` |

Set `PRSM_ARBITRATION_PROPOSER_ID=""` (empty string) to explicitly disable Tier 2 even if accidentally inherited from a parent shell.

### Failure isolation

The upload path uses three-tier failure isolation: `queue.enqueue` / `sink.create_proposal` / `queue.set_proposal_id` are independently wrapped in try/except. A failure in any tier logs and proceeds — uploads stay green even if governance is broken.

### Operator runbook

For full activation, monitoring, alert thresholds, council resolution flow, rollback levels, and troubleshooting see:

`docs/2026-05-08-prsm-prov-1-item-6-operator-activation-runbook.md`

Threat model: `docs/2026-05-08-prsm-prov-1-threat-model-addendum-item-6.md` (§3.18, A1–A8).

---

## DHT Embedding Transport (cross-node fingerprint dedup)

Optional cross-node binary fingerprint deduplication via Kademlia DHT.

| Variable | Default | Description |
|---|---|---|
| `PRSM_DHT_ENABLED` | `0` | Set to `1` to enable two-way DHT transport (clients ASK, servers ANSWER). Lockstep with embedding lane via `LocalFingerprintIndex`. Off by default — opt in deliberately. |

When enabled, the node spins up an asyncio-loop-backed `SyncDHTTransport` + `DHTListener` + `EmbeddingDHTServer`. Cross-node hits are Ed25519-signature-verified before being trusted.

---

## MCP Server (Tool Surface)

The MCP server exposes a curated subset of tools for agentic clients.

As of 2026-05-08, the previously-hidden tools (`prsm_analyze`, `prsm_dispatch_agent`, `prsm_agent_status`) have all been restored:

- `prsm_analyze` — exposes the QueryOrchestrator + AggregatorClient stack
- `prsm_dispatch_agent` — async dispatch with JobHistoryRecord-backed status tracking
- `prsm_agent_status` — surfaces the two-tier `/compute/status` response

A new tool `prsm_balance_check` ships in v1.7.0+ (Vision §13 Phase 5 stand-in closure):

- Reads on-chain FTNS balance via the node's `OnChainFTNSLedger` (gates on `PRSM_ONCHAIN_FTNS=1` + `FTNS_TOKEN_ADDRESS`).
- Converts FTNS → USD using `PRSM_FTNS_USD_RATE` env var (default `1.0`; placeholder until the Aerodrome USDC-FTNS pool is seeded per Vision gantt 2026-06-15, after which the rate sources from the live pool).
- Optional `address` arg overrides the node's connected wallet address.
- Backed by the new `GET /balance/onchain` endpoint.

**Aggregate-source quoting (v2, ships 2026-05-09):** When the node has the RoyaltyDistributor client + PaymentEscrow manager wired in addition to the FTNS ledger, `/balance/onchain` aggregates across three sources:

- `balance_ftns` — on-chain FTNS (legacy v1 field, preserved bit-identically)
- `claimable_royalties_ftns` — pending RoyaltyDistributor claimable amount for the address
- `escrowed_ftns` — sum of FTNS held in PENDING escrows for in-flight compute jobs by this requester
- `total_ftns` / `total_usd_equivalent` — aggregate sum across all three
- `sources` — per-source breakdown with `available` flag (lets clients distinguish "0 because empty" from "0 because source unwired/RPC-flaked")

Each source is fail-soft: if the underlying RPC or escrow store throws, the endpoint still returns 200 with that source marked `available: false` and contributing 0 to the aggregate. Clients reading only v1 fields keep working unchanged. The MCP `prsm_balance_check` handler renders a multi-line breakdown when extras are present, and falls back to the legacy single-line format when only on-chain is wired.

A companion tool `coinbase_offramp_initiate` ships v1 alongside (Vision §13 Phase 5 step 2 — pre-flight composer):

- Returns the transaction-summary artifact users see in their AI side-panel before authorizing a cash-out (FTNS → USDC via Aerodrome → USD via Coinbase CDP off-ramp).
- **Does NOT initiate any on-chain or fiat-side action in v1** — status is `PENDING_COMMISSION` until CDP commissioning completes (gates on Aerodrome pool seeding per Vision gantt 2026-06-15).
- Required arg: `usd_amount` (positive; 422 if exceeds available balance). Optional: `bank_account_alias` (default `"primary"`).
- Backed by the new `POST /wallet/offramp/quote` endpoint.

**Source-aware available-balance check (v2, ships 2026-05-09):** The quote endpoint now checks the request against `available_ftns = on-chain + claimable royalties` (escrowed FTNS does NOT count — locked in pending compute jobs). When on-chain alone is insufficient but claimable royalties bridge the gap, the endpoint returns 200 with `claim_required: true` + `claim_amount_ftns` (the FTNS shortfall the operator must claim before the eventual swap can execute). The MCP handler renders a `Prerequisite: Claim X FTNS in royalties before swap can execute` block ahead of the quote summary so the chain-of-actions is explicit. Royalty client RPC failures are fail-soft: claimable treated as 0, request validated against on-chain only.

A companion tool `prsm_royalty_claim` ships v1 alongside (closes the loop on the offramp claim_required path):

- Defaults to `dry_run=true` — returns the artifact + claimable amount without on-chain action; `status: "DRY_RUN"`.
- Pass `dry_run=false` to execute the on-chain `RoyaltyDistributor.claim()` call; returns `status: "EXECUTED"` + `tx_hash`.
- 0 claimable + `dry_run=false` returns `status: "SKIPPED_ZERO"` (avoids on-chain ZeroClaim revert + gas burn).
- Backed by the new `POST /wallet/royalty/claim` endpoint.
- v1 caveat: operator authorization is implicit via running the node with the configured FTNS private key. There is no per-call passkey/biometric prompt; if you don't trust auto-claim execution from your AI side-panel, default to `dry_run=true` and surface the artifact for confirmation in the chat first.

Suggested usage flow today: `prsm_balance_check` (read available funds) → `coinbase_offramp_initiate` (pre-flight quote) → operator inspects the summary, plans for the eventual execution path.

`BROKEN_TOOLS_HIDDEN` is now an empty frozenset. If you re-add a tool to the hide-list mid-incident, also pin `tests/unit/test_mcp_server_hidden_tools.py` to match — the test asserts the exact set.

---

## KeyDistribution Client (Tier C content)

Tier C ("encrypt-then-distribute") content uses on-chain `KeyDistribution.sol` to release decryption shards only after payment is verified.

`prsm/economy/web3/key_distribution.py` provides the `KeyDistributionClient`:

- `deposit_key()` — uploader deposits shards
- `release()` — operator releases on payment proof
- `deauthorize()` — revoke previously-released access
- `KeyReleasedEvent` — typed event for downstream listeners

Errors are typed: `KeyAlreadyDepositedError` / `KeyNotFoundError` / `PaymentNotVerifiedError`. All Web3 calls go through the per-keypair `TX_LOCK_REGISTRY` lock — see "On-chain Keypairs" for the one-keypair-per-process invariant.

**Limitation:** an event-watcher daemon that reacts to `KeyReleasedEvent` is on the deferred-follow-ons list. Until it ships, operators wanting end-to-end automation should poll the contract or wire their own Web3 event subscription. The synchronous client is correct on its own; only the daemon-side reactivity is missing.

---

## Phase 7-storage + Phase 8 Daemons

As of 2026-05-08, two async daemons ship for periodic on-chain invocation. Both are optional and opt-in.

### HeartbeatScheduler (storage providers)

Storage providers MUST heartbeat regularly to avoid permissionless slashing via `slash_for_missing_heartbeat`. The daemon calls `StorageSlashingClient.record_heartbeat()` on a configurable cadence.

| Variable | Default | Description |
|---|---|---|
| `PRSM_STORAGE_SLASHING_ADDRESS` | *unset* | StorageSlashing.sol address. Required to construct the client. |
| `FTNS_WALLET_PRIVATE_KEY` | *unset* | Provider's signing key. Required for the heartbeat tx. |
| `PRSM_HEARTBEAT_SCHEDULER_ENABLED` | `0` | Set to `1` to launch the daemon at node startup. |
| `PRSM_HEARTBEAT_SCHEDULER_INTERVAL_SECONDS` | *auto-tune* | Cadence override (seconds). When unset (default), the daemon **auto-tunes from `client.heartbeat_grace_seconds()`** at construction time — interval = `grace / 4` (4 heartbeats per grace window for missed-tick defense), floored at 60s. With grace=1h (contract minimum), auto-tune produces 900s (matches prior fixed default). With longer grace, the interval scales proportionally — no manual env-var tuning needed. Auto-tune falls back to fixed 900s when the client's RPC is unreachable at startup or returns invalid grace. |

The daemon swallows all transient errors and stays alive across them. `success_count` and `failure_count` are exposed for operator telemetry.

### PullAndDistributeScheduler (Phase 8)

Permissionless on-chain — anyone can run this. Calls `CompensationDistributorClient.pull_and_distribute()` to drain accrued emission across the three pools. The contract's source flags monitoring alerts on call-gap > 7 days; the daemon's constructor REJECTS interval > 7 days as a hard fail-fast.

| Variable | Default | Description |
|---|---|---|
| `PRSM_COMPENSATION_DISTRIBUTOR_ADDRESS` | *unset* | CompensationDistributor.sol address. Required to construct the client. |
| `FTNS_WALLET_PRIVATE_KEY` | *unset* | Caller's signing key (any address — this is a permissionless call). |
| `PRSM_COMPENSATION_SCHEDULER_ENABLED` | `0` | Set to `1` to launch the daemon at node startup. |
| `PRSM_COMPENSATION_SCHEDULER_INTERVAL_SECONDS` | `86400` | Cadence (24h default; values > 7 days fall back to default). |

### Operator-side disable

To disable either daemon mid-incident: unset the matching `*_ENABLED` env var and restart, OR unset the address env var (which also disables the underlying client). See `docs/security/EXPLOIT_RESPONSE_PLAYBOOK_ANNEX_2026_05.md` for the full operator-pause runbook.

### Event watchers (post-2026-05-08)

Three async event watchers ship alongside the schedulers. Each polls on-chain event logs at a configurable cadence and fires callbacks. Default callbacks log each event at INFO/WARNING level for out-of-the-box observability.

| Variable | Default | Description |
|---|---|---|
| `PRSM_KEY_DISTRIBUTION_WATCHER_ENABLED` | `0` | Set to `1` to launch `KeyDistributionWatcher`. **Material for Tier C publishers** — surfaces `KeyReleased` events in seconds for payment-verification reconciliation (annex §5.4 detection). |
| `PRSM_KEY_DISTRIBUTION_WATCHER_POLL_SECONDS` | `30` | Cadence. |
| `PRSM_STORAGE_SLASHING_WATCHER_ENABLED` | `0` | Set to `1` to launch `StorageSlashingWatcher`. **Material for storage providers** — surfaces `HeartbeatMissingSlashed` + `ProofFailureSlashed` against your provider address. |
| `PRSM_STORAGE_SLASHING_WATCHER_POLL_SECONDS` | `30` | Cadence. |
| `PRSM_COMPENSATION_DISTRIBUTOR_WATCHER_ENABLED` | `0` | Set to `1` to launch `CompensationDistributorWatcher`. Surfaces `Distributed` events for accounting-reconciliation purposes. |
| `PRSM_COMPENSATION_DISTRIBUTOR_WATCHER_POLL_SECONDS` | `30` | Cadence. |

**Activation requires the matching contract-address env var to be set** (so the underlying client can construct). For the KeyDistribution watcher, that's `PRSM_KEY_DISTRIBUTION_ADDRESS`. The address env vars are documented in the "Mainnet On-chain Surface" section above.

Operators wanting custom callbacks (instead of the default INFO-log) should construct the watchers programmatically rather than via env-var activation. See module docstrings in `prsm/economy/web3/*_watcher.py` for the API.

#### Watcher last_processed_block persistence (post-2026-05-09)

By default, watchers reset their `last_processed_block` baseline to the chain tip on every node restart — events that landed during downtime are silently skipped. To enable restart-resilient detection (recommended for any storage provider or Tier C publisher relying on the watchers for incident detection), opt in to filesystem-backed persistence:

| Variable | Default | Description |
|---|---|---|
| `PRSM_WATCHER_STATE_PERSISTENCE_ENABLED` | `0` | Set to `1` to enable shared `FilesystemLastProcessedBlockStore` for the 3 watchers. |
| `PRSM_WATCHER_STATE_DIR` | `~/.prsm/watchers/` | Override the default state-store directory. Auto-created on first save; one JSON file per watcher (`key_distribution.json` / `storage_slashing.json` / `compensation_distributor.json`). |

When enabled, each watcher loads its persisted block on first tick and polls the downtime-window range — recovering events that landed during the restart. Failures (corrupt JSON / IOError) fall back to chain-tip baseline + log a WARNING; the next successful baseline advance re-persists.

#### Watcher event filters (RPC-side narrowing, post-2026-05-09)

By default, watchers fire callbacks for ALL events on the contract — fleet-wide. For operators monitoring only their own provider/recipient/publisher address (the common case), this produces unnecessary network bandwidth, callback invocations, and log noise.

`KeyDistributionWatcher` and `StorageSlashingWatcher` accept an optional `event_filters` kwarg that propagates to web3.py's `event.get_logs(argument_filters=...)` for **RPC-side filtering**:

```python
from prsm.economy.web3.storage_slashing_watcher import StorageSlashingWatcher

# A storage provider monitoring only their own slashing events:
my_provider = "0x..."  # provider's own address
watcher = StorageSlashingWatcher(
    client=storage_slashing_client,
    on_heartbeat_missing_slashed=alert_callback,
    on_proof_failure_slashed=alert_callback,
    event_filters={
        "HeartbeatMissingSlashed": {"provider": my_provider},
        "ProofFailureSlashed": {"provider": my_provider},
    },
)
```

Filter keys must match the watcher's `KNOWN_EVENT_NAMES` set (`KeyReleased` / `KeyDeposited` / `KeyDeauthorized` for KeyDistributionWatcher; `HeartbeatRecorded` / `ProofFailureSlashed` / `HeartbeatMissingSlashed` for StorageSlashingWatcher). Filter values can be a single address or a list (web3.py supports OR-style matching). Unknown event names raise `ValueError` at constructor time.

`CompensationDistributorWatcher.Distributed` has NO indexed args — there's nothing to filter on at the RPC level — so the watcher does NOT accept an `event_filters` kwarg (passing it raises `TypeError`).

This is a programmatic-only feature today (env-var activation would need a way to express address values in env, which is operator-specific). Operators wanting filters construct watchers programmatically instead of via env-var activation.

---

## Monitoring

### Health Endpoints

```bash
# Basic health check
GET /health
# Response: {"status": "healthy", "version": "1.0.0"}

# Detailed metrics
GET /health/metrics
# Response: Prometheus-format metrics

# Readiness check (for load balancers)
GET /health/ready
# Response: 200 if ready, 503 if not
```

### Key Metrics to Monitor

| Metric | Healthy Range | Alert Threshold |
|--------|---------------|-----------------|
| `api_latency_p99` | < 2000ms | > 5000ms |
| `error_rate` | < 1% | > 5% |
| `active_connections` | Varies | > 80% max |
| `ftns_queue_depth` | < 100 | > 500 |
| `content_store_size` | < 80% disk | > 90% |
| `bittorrent_active_torrents` | varies | > 80% configured max |
| `p2p_peers_connected` | > 3 | < 2 |

### Grafana Dashboard

Import the provided dashboard: `monitoring/grafana-dashboard.json`

Key panels:
- Request rate and latency
- Error rate by endpoint
- FTNS transaction volume
- P2P network health
- Resource utilization

---

## Upgrading

### Safe Upgrade Procedure

```bash
# 1. Backup database
./scripts/backup.sh

# 2. Pull latest code
git pull origin main

# 3. Run database migrations
alembic upgrade head

# 4. Restart services
docker-compose restart

# 5. Verify
curl http://localhost:8000/health
```

### Zero-Downtime Upgrade (Kubernetes)

```bash
# Rolling update
kubectl set image deployment/prsm-api \
  prsm-api=prsm/api:v1.2.0 \
  --record

# Monitor rollout
kubectl rollout status deployment/prsm-api

# Rollback if needed
kubectl rollout undo deployment/prsm-api
```

---

## Backup and Recovery

### What to Back Up

| Component | Frequency | Retention |
|-----------|-----------|-----------|
| Database | Daily | 30 days |
| `node_identity.json` | Once | Forever |
| `.env` file | On change | Keep last 3 |
| Content store + BitTorrent shards (optional, derivable from peers) | Weekly | 4 weeks |
| `~/.prsm/arbitration_queue/` (if Item 6 governance hook enabled) | Daily | Until disputes resolved |

### Backup Commands

```bash
# SQLite backup
cp prsm_node.db prsm_node.db.backup

# PostgreSQL backup
pg_dump prsm > backup_$(date +%Y%m%d).sql

# Full backup script
./scripts/backup-system.sh backup --output /backups/
```

### Recovery Procedure

```bash
# 1. Stop services
docker-compose down

# 2. Restore database
cp prsm_node.db.backup prsm_node.db
# OR
psql prsm < backup_20250101.sql

# 3. Restart services
docker-compose up -d

# 4. Verify
curl http://localhost:8000/health
```

---

## Troubleshooting

### Common Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Node won't connect to bootstrap | Firewall blocking port 8765 | `sudo ufw allow 8765/tcp` |
| Bootstrap WSS handshake fails | Operator pinned legacy `prsm.io` host | Update `P2P_BOOTSTRAP_NODES` to `wss://bootstrap-us.prsm-network.com:8765` |
| Native content-store fails to write | Permissions on `~/.prsm/content_store/` | `chmod -R u+rwX ~/.prsm/content_store/` |
| "JWT secret too short" | SECRET_KEY missing or weak | Generate: `openssl rand -hex 32` |
| FTNS balance stuck | Database locked | Restart node; check for orphaned SQLite WAL |
| High memory usage | Too many concurrent queries | Reduce `MAX_CONCURRENT_QUERIES` |
| Slow queries | Model provider API issues | Check external API status |
| P2P not connecting | NAT traversal failed | Enable UPnP or configure port forwarding |

### Diagnostic Commands

```bash
# Check service status
docker-compose ps
systemctl status prsm

# View recent logs
docker-compose logs --tail=100 prsm-api
journalctl -u prsm -n 100

# Check port bindings
netstat -tlnp | grep -E '8000|8765'

# Test database connection
python -c "from prsm.core.database import get_db; print('OK')"

# Inspect native content-store
ls -lh ~/.prsm/content_store/ | head

# Check P2P peers
curl http://localhost:8000/api/v1/p2p/peers

# Inspect arbitration queue (if Item 6 governance hook enabled)
ls -lh ~/.prsm/arbitration_queue/

# Inspect Job history (in-memory; restart clears)
curl http://localhost:8000/compute/status/<job_id>
```

### Log Analysis

```bash
# Find errors
docker-compose logs prsm-api | grep -i error

# Find slow queries
docker-compose logs prsm-api | grep "duration_ms" | awk '$NF > 5000'

# Monitor in real-time
docker-compose logs -f prsm-api | grep -E "ERROR|WARN"
```

---

## On-chain Keypairs

Nodes that participate in on-chain economic activity (provenance, royalty distribution, stake bonding, batched settlement) hold at least one Ethereum-compatible keypair. The invariants below apply to ALL such keypairs.

### Invariant: one keypair per OS process

**Do not share a provider keypair across multiple processes or multiple machines.**

Every PRSM Web3 client (`ProvenanceRegistryClient`, `RoyaltyDistributorClient`, `StakeManagerClient`, etc.) serializes tx build-and-send via an in-process lock. That lock does NOT coordinate across processes. If two processes on the same host (or two hosts) sign transactions for the same keypair concurrently, they can read the same `pending` nonce from the RPC and both broadcast against it. One transaction will revert; the other lands.

This is a correctness-preserving failure (no double-spend, no slashable misbehavior), but it wastes gas and generates confusing operator telemetry.

**Practical rules:**

- One Python process = one provider keypair. If you run a compute node AND a content-hosting node, give them separate keypairs.
- If you run multiple replicas (Kubernetes deployment with `replicas > 1`), ensure each replica has its own keypair — do NOT mount the same keyfile across pods.
- If you operate both a Phase 1.1 content node and a Phase 7 compute-provider node, use a distinct keypair per role. They may share an owner (for accounting), but not a signing key.
- Rotation: when rotating keys, finalize the old key's pending txs BEFORE the new key begins signing.

### Invariant: key material stays local

Never upload keyfile contents to external services (pastebins, shared notes, CI logs, screenshot tools). Key material belongs on the host that uses it; if you need to back it up, encrypt first with a passphrase only you hold.

### Invariant: hardware wallets for Foundation / multi-sig roles

Treasury, governance, and slasher-admin roles (owner of `StakeBond`, owner of `BatchSettlementRegistry`) MUST run behind a 2-of-3 hardware-wallet multi-sig. See PRSM-GOV-1 §8 for the quorum spec. Operator nodes do NOT hold these keys — they belong exclusively to the Foundation.

---

## Redundant-Execution Dispatch (Tier B)

Jobs submitted with `DispatchPolicy.consensus_mode="majority"` run on k providers in parallel (default k=3). The orchestrator compares output hashes, returns the majority's result, and records the minority for later on-chain challenge.

### What operators need to know

**1. Minority receipts accumulate in-process.**

`MarketplaceOrchestrator.consensus_minority_queue` is a per-process list. Each dispatch that finds a disagreeing provider appends a record there. Nothing auto-submits these as on-chain `CONSENSUS_MISMATCH` challenges — that's the job of a separate **consensus-submitter service**.

If you run the orchestrator without a consensus-submitter:
- The job still returns the correct (majority) output to the caller.
- The minority provider gets a reputation penalty via `ReputationTracker`.
- **But their stake is NOT slashed on-chain.** The minority walks away unpunished economically.

This is acceptable for early bake-in (Phase 7.1 MVP) but unsafe at scale. If you're running consensus dispatch in production, either wire a consensus-submitter service or document that your deployment does not enforce the economic layer of Tier B.

**2. Crash recovery drops pending challenges.**

The queue is in-memory. A process restart between dispatch and submission drops whatever minority receipts haven't been drained. Phase 7.1x's consensus-submitter is the planned solution (SQLite/Redis-backed drain). Until then, treat the queue as best-effort — critical-tier jobs with consensus enabled should also pair with on-chain settlement audit (cross-check batch emissions vs orchestrator logs).

**3. Cost model.**

k=3 consensus means 3× the compute cost per shard. Requester pays all k providers from a single escrow. Partial responses (e.g., 2-of-3 with one preempted) still consense if the threshold is met; the non-respondent forfeits their escrow share per Phase 2's existing timeout semantics.

**4. Price discovery.**

Phase 7.1 MVP splits `policy.max_price_per_shard_ftns` evenly across k providers rather than running k parallel price handshakes. Per-provider price discovery in the consensus path is Phase 7.1x work. Set `max_price_per_shard_ftns` generously relative to the per-provider ceiling you'd accept; the effective per-provider budget is `max_price / k`.

### Operator checklist for Tier B

- [ ] `policy.consensus_mode` set to `"majority"` (or `"unanimous"` if you require full k-agreement)
- [ ] `policy.consensus_k` set appropriately (default 3; higher for critical)
- [ ] `multi_dispatcher` wired on the orchestrator (else RuntimeError at dispatch)
- [ ] `stake_manager_client` wired (else the on-chain tier gate skips, per Phase 7)
- [ ] `provider_address_resolver` wired (else the on-chain tier gate skips, per Phase 7)
- [ ] Consensus-submitter service deployed and draining `consensus_minority_queue`, OR you have a documented policy that your deployment doesn't enforce the economic layer
- [ ] Eligible pool sized ≥ consensus_k; smaller pools fail-fast with `NoEligibleProvidersError`

---

## Security Checklist

- [ ] Generate unique `SECRET_KEY` (never use default)
- [ ] Change default admin password
- [ ] Enable TLS/HTTPS (use nginx or certbot)
- [ ] Configure firewall (only expose 8000, 8765)
- [ ] Set up rate limiting
- [ ] Enable audit logging
- [ ] Configure log rotation
- [ ] Set up monitoring alerts
- [ ] Create backup schedule
- [ ] Document recovery procedures

---

## Getting Help

- **Documentation**: [docs.prsm.ai](https://docs.prsm.ai)
- **GitHub Issues**: [github.com/prsm-network/PRSM/issues](https://github.com/prsm-network/PRSM/issues)
- **Discord**: [discord.gg/prsm](https://discord.gg/prsm)
- **Email**: ops@prsm.ai

For production incidents, email security@prsm.ai for security issues.

---

## Appendix: Example Configurations

### Minimal Development Config

```bash
# .env (development)
SECRET_KEY=dev-secret-key-do-not-use-in-production
DATABASE_URL=sqlite:///./prsm_dev.db
DEBUG=true
LOG_LEVEL=DEBUG
```

### Production Config

```bash
# .env (production)
SECRET_KEY=<32+ character random string>
DATABASE_URL=postgresql://prsm:${POSTGRES_PASSWORD}@db:5432/prsm
REDIS_URL=redis://redis:6379
P2P_BOOTSTRAP_NODES=wss://bootstrap-us.prsm-network.com:8765
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
LOG_LEVEL=INFO
DEBUG=false

# QueryOrchestrator (opt in to Ring 5 replacement)
PRSM_QUERY_ORCHESTRATOR_ENABLED=1
PRSM_AGGREGATOR_SHARE_BPS=500   # 5% — operator-tunable

# On-chain ingest (uncomment to register uploads on Base mainnet)
# PRSM_ONCHAIN_PROVENANCE=1
# PRSM_PROVENANCE_REGISTRY_V2_ADDRESS=0xe0cedDA354f99526c7fbb9b9651e12aDB2180dbf

# DHT cross-node fingerprint dedup (off by default)
# PRSM_DHT_ENABLED=1

# Item 6 governance proposal sink (off by default)
# PRSM_ARBITRATION_PROPOSER_ID=foundation-proposer-1

# Phase 7-storage daemon: heartbeats provider liveness on cadence.
# Required for storage providers to avoid permissionless slashing.
# PRSM_STORAGE_SLASHING_ADDRESS=0x...
# PRSM_HEARTBEAT_SCHEDULER_ENABLED=1
# PRSM_HEARTBEAT_SCHEDULER_INTERVAL_SECONDS=900   # default 15 min

# Phase 8 daemon: pulls + distributes FTNS emission on cadence.
# Permissionless on-chain — anyone can run this; cadence < 7 days.
# PRSM_COMPENSATION_DISTRIBUTOR_ADDRESS=0x...
# PRSM_COMPENSATION_SCHEDULER_ENABLED=1
# PRSM_COMPENSATION_SCHEDULER_INTERVAL_SECONDS=86400   # default 24h
```

### High-Performance Compute Node

```bash
# .env (compute provider)
SECRET_KEY=<random>
DATABASE_URL=postgresql://prsm:${POSTGRES_PASSWORD}@db:5432/prsm
MAX_CONCURRENT_QUERIES=50
WORKER_THREADS=16
QUERY_TIMEOUT=600
GPU_ENABLED=true
GPU_MEMORY_FRACTION=0.9
```
