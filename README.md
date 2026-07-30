# PRSM — Protocol for Research, Storage, and Modeling

**The code goes to the data. Not the other way around.**

PRSM is a peer-to-peer protocol that unifies three resource markets — data, compute, economic settlement — under one token. Frontier LLMs (Claude, GPT, Gemini, or local MCP-compatible models) call PRSM via MCP as a retrieval and heavy-compute substrate whenever a query needs data the model doesn't have, compute beyond its context budget, or verifiable provenance. Contributors earn **FTNS** (*Fungible Token for Node Support*) for sharing storage, compute, and data from idle consumer devices. Reasoning happens in your LLM; execution happens on PRSM nodes with cryptographically-enforced creator royalties.

**Version 1.7.0** | [PyPI](https://pypi.org/project/prsm-network/) | [Getting Started](docs/GETTING_STARTED.md) | [Documentation Index](docs/INDEX.md)

[![PyPI version](https://badge.fury.io/py/prsm-network.svg)](https://pypi.org/project/prsm-network/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP Tools](https://img.shields.io/badge/MCP%20tools-18-blue.svg)](#mcp-integration)

> **Community** — Join us on [Discord](https://discord.gg/R8dhCBCUp3) for dev / node-operator / governance discussion. GitHub Discussions enabled on this repo for async Q&A + proposals.

---

## 🔐 Verify a PRSM inference receipt in 30 seconds

PRSM's §7 verifiable-inference promise — *a third party can verify
chain-of-custody on any inference receipt without trusting the
operator* — is operationally provable in one shell session:

```bash
pip install web3 cryptography
python3 scripts/verify_prsm_receipt.py docs/sample-receipts/tier-standard-dp-2026-05-22.json
# → ✓ VALID — receipt verifies cleanly
```

The verifier is a single ~300-line file with **zero imports from
the `prsm` package**. It looks up the settler's pubkey on the
on-chain `PublisherKeyAnchor` (Base mainnet), reconstructs the
canonical signing bytes, and verifies the Ed25519 signature
client-side. Tamper any byte → `✗ INVALID — settler_signature
does NOT verify`.

4 sample receipts cover the full architectural matrix
(unary, streaming, DP injection, multi-host 2-stage) — see
[`docs/sample-receipts/`](docs/sample-receipts/README.md).

For the full external-readiness summary (operational claims with
on-chain TX evidence, architecture flow, cost model, known limits),
read **[Verifiable Inference — Audit Readiness Summary](docs/2026-05-22-parallax-inference-audit-readiness.md)** (~15 min).

To stand up your own anchor-registered operator node, follow
**[Parallax Inference Deploy Runbook](docs/operations/parallax-inference-deploy.md)** (8 numbered steps).

---

## Why PRSM Exists

**The problem:** Frontier AI labs hoard data, compute, and models behind API walls. Every query you send to a centralized API is logged, stored, and potentially trained on. Meanwhile, billions of consumer devices sit idle — gaming PCs, laptops, phones, tablets — each with storage, compute, and sometimes proprietary data that never leave the device.

**The PRSM thesis:** Build an open-source commons for AI infrastructure. Aggregate the latent resources of consumer electronics into a P2P mesh. Let any LLM — local, OAuth, or API — use that mesh via MCP tools. Pay contributors in FTNS tokens so sharing beats hoarding.

| Centralized AI (today) | PRSM |
|------------------------|------|
| You upload data to their cloud | WASM agents travel to the data |
| They see your query | Zero-persistence sandbox — nothing logged |
| One datacenter processes everything | Thousands of edge nodes work in parallel |
| You pay per token | Hybrid pricing — commodity compute + market-rate data |
| Vendor lock-in | Works with any LLM via MCP |

**The result:** A network where the model is open but the computation is private, and where contributing your idle storage/compute/data earns you a share of every query it serves.

---

## Quick Start

```bash
# Install
pip install "prsm-network[ml]"   # [ml] ships the local inference engine

# Check your hardware
prsm node benchmark

# Start your node
prsm node start

# Expose PRSM tools to any MCP-compatible LLM
prsm mcp-server
```

> **FTNS tokens:** Providers earn FTNS for sharing storage, compute, and data through their node. New nodes receive a 100 FTNS welcome grant. Third-party LLMs invoke PRSM tools via MCP; reasoning happens in the LLM, execution happens on PRSM nodes.

See the full [Getting Started Guide](docs/GETTING_STARTED.md) for detailed setup.

---

## How It Works

### The 10-Ring Architecture

PRSM is built as concentric capability rings. Each ring wraps and enriches the ones inside it.

| Ring | Name | What It Does |
|------|------|-------------|
| 1 | **The Sandbox** | WASM runtime with Wasmtime — sandboxed execution with memory/time limits |
| 2 | **The Courier** | Mobile agent dispatch — agents travel to data via P2P gossip + bidding |
| 3 | **The Swarm** | Semantic sharding — data split by meaning, parallel map-reduce across nodes |
| 4 | **The Economy** | Hybrid pricing — deterministic compute rates + market-rate data + 80/15/5 revenue splits |
| 5 | **Agent Forge** | WASM mobile agent runtime — query decomposition is performed by the caller's third-party LLM; PRSM dispatches the resulting WASM agents |
| 6 | **The Polish** | Production hardening — dynamic gas, RPC failover, CLI commands |
| 7 | **The Vault** | Confidential compute — TEE abstraction + differential privacy noise |
| 8 | **The Shield** | Model sharding — tensor parallelism + randomized pipelines + collision detection |
| 9 | **The Mind** | NWTN training pipeline — collect traces, evaluate quality, deploy fine-tuned models (reserved for future work) |
| 10 | **The Fortress** | Security — integrity verification, privacy budgets, hash-chained audit logs |

### End-to-End Flow

```
Third-party LLM (Claude/GPT/local): calls prsm_analyze via MCP

  → LLM:     Decomposes the query into WASM agent instructions
  → Ring 3:  Finds relevant semantic shards by embedding similarity
  → Ring 4:  Quotes cost: compute + data + network fee
  → Ring 3:  Fans out parallel agents to shard-holding nodes
  → Ring 2:  Each agent dispatched via gossip bidding
  → Ring 1:  Executed in WASM sandbox on provider hardware
  → Ring 7:  Differential privacy noise applied
  → Ring 3:  Results aggregated when quorum met
  → Ring 4:  FTNS settled: 80% data owner / 15% compute / 5% treasury

  ← Result returned to the LLM for final synthesis
```

---

## MCP Integration

Any LLM can use PRSM as a compute backend via the Model Context Protocol. **18 tools** are exposed.

### Three install paths

Pick whichever your environment prefers — all three launch the same Python MCP server underneath.

**1. npm (recommended for MCP-client integration):**
```bash
npx prsm-mcp        # auto-detects Python; one-time prompt to install prsm-network
```

Add to `~/.claude/claude_desktop_config.json`:
```json
{"mcpServers": {"prsm": {"command": "npx", "args": ["prsm-mcp"]}}}
```

**2. Homebrew (macOS / Linuxbrew):**
```bash
brew install prsm/tap/prsm
prsm mcp-server
```

**3. PyPI (direct Python install — canonical):**
```bash
pip install prsm-network
prsm mcp-server
```

Then add to your MCP client config:
```json
{"mcpServers": {"prsm": {"command": "prsm", "args": ["mcp-server"]}}}
```

The `prsm-network` Python package is the source of truth; npm and Homebrew are convenience wrappers.

Then Claude (or any MCP-compatible LLM) can:

| Tool | What It Does |
|------|-------------|
| `prsm_analyze` | Full Ring 1-10 pipeline — query in, answer out |
| `prsm_inference` | TEE-attested model inference with verifiable signed receipts |
| `prsm_quote` | Cost estimate before committing (free) |
| `prsm_create_agent` | Build custom agent with 11 data operations |
| `prsm_dispatch_agent` | Execute agent on the network |
| `prsm_upload_dataset` | Publish data with pricing |
| `prsm_list_datasets` | Browse available datasets |
| `prsm_search_shards` | Find relevant data shards |
| `prsm_yield_estimate` | "What would I earn?" |
| `prsm_stake` | Staking tier info |
| `prsm_revenue_split` | Calculate 80/15/5 distribution |
| `prsm_hardware_benchmark` | GPU, TFLOPS, tier, TEE detection |
| `prsm_node_status` | Ring 1-10 health check |
| `prsm_agent_status` | Check running agent |
| `prsm_settlement_stats` | FTNS settlement queue |
| `prsm_billing_status` | Look up FTNS escrow state by job_id |
| `prsm_privacy_status` | Differential privacy budget |
| `prsm_training_status` | NWTN training corpus quality |

---

## For Data Providers

Publish data through your node's ContentStore and earn 80% of every query against it:

```bash
prsm storage upload ./my_dataset.parquet \
  --description "NADA NC Vehicle Registrations 2025" \
  --royalty-rate 0.05 \
  --replicas 5
```

Revenue split: **80% to you**, 15% to compute providers, 5% to PRSM treasury.

---

## For Compute Providers

Check what you'd earn and start providing:

```bash
prsm node benchmark                           # See your hardware tier
prsm ftns yield-estimate --hours 8 --stake 1000   # Monthly earnings estimate
prsm node start                                # Start earning
```

**Staking is utility-only** — locking FTNS confers a network-fee
discount and dispatch-priority boost. It pays **no token yield, APY, or
emission multiplier**. Benefits scale with lock duration:

| Lock period | Network-fee discount | Dispatch-priority boost |
|-------------|----------------------|-------------------------|
| No lock | 0% | none |
| 30 days | 2% | +10% |
| 90 days | 5% | +25% |
| 365 days | 10% | +50% |

---

## Privacy Architecture

PRSM provides three layers of privacy by construction — not by policy:

1. **WASM Zero-Persistence** — The sandbox has no filesystem, no network, no state after execution. The agent literally *cannot* persist data.
2. **Semantic Data Sharding** — No single node holds the full dataset. Each node sees only its assigned shard.
3. **Differential Privacy** — Calibrated Gaussian noise on all intermediate activations (configurable ε: 8.0 standard, 4.0 high, 1.0 maximum).

For proprietary models, **tensor-parallel model sharding** distributes weights across nodes so no single operator can reconstruct the model. **Randomized pipeline assignment** changes the topology per inference. **Collision detection** catches tampering.

See [Confidential Compute Spec](docs/CONFIDENTIAL_COMPUTE_SPEC.md) for details.

---

## Built on Ethereum + Base

PRSM's smart contracts target **Base**, an Ethereum Layer 2 operated by Coinbase. The FTNS token is implemented as a standard ERC-20 (`FTNSTokenSimple.sol`) and is **live on Base mainnet (chain 8453)** at `0x5276a3756C85f2E9e46f6D34386167a209aa16e5`. The Foundation's 2-of-3 Safe on Base mainnet (`0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791`) is the deployer of record for the full contract suite (FTNSToken + ProvenanceRegistry + RoyaltyDistributor v2 `0xfEa9aeB99e02FDb799E2Df3C9195Dc4e5323df7e` + EmissionController + CompensationDistributor).

**Why Ethereum:**

| Criterion | Ethereum delivers |
|---|---|
| Credible decentralization | ~1M validators, zero chain halts since PoS launch (Sept 2022) |
| Developer ecosystem | Largest by an order of magnitude; Hardhat, Foundry, OpenZeppelin are Ethereum-native |
| Auditor coverage | Dozens of reputable Solidity audit firms vs. 3-4 on alternatives |
| US regulatory posture | ETH classified as commodity by CFTC; spot and futures ETFs approved |
| Standards + composability | ERC-20 / ERC-721 / ERC-4337 — every wallet, exchange, indexer speaks EVM |

**Why Base (not Ethereum L1 directly, not Solana, not a new chain):**

- **~$0.01 transactions, 2-second blocks** — viable economics for micropayment royalty flow
- **Operated by Coinbase** with public uptime SLAs and a deep tooling ecosystem
- **Native USDC from Circle** — zero-friction fiat on-ramp for Phase 5 (Coinbase → Base in seconds)
- **OP Stack foundation** — most battle-tested L2 codebase, portable to any OP Stack chain if Base ever falters
- **Inherits Ethereum L1 security** — fraud proofs and trust-minimized withdrawals, not a trust-in-Coinbase promise

**Short answer to "why not Solana?":** *"We need Ethereum's decentralization and tooling, we don't need Solana's peak speed for our use case, and we get near-Solana performance via Base L2 without giving up any of Ethereum's other advantages."*

For the full argument (evaluation framework, honest comparison of major chains, responses to common pushback), see [Technology Choices](docs/TECH_CHOICES.md).

---

## SDKs

| Language | Install | Docs |
|----------|---------|------|
| **Python** | `pip install prsm-network` | [SDK Guide](docs/SDK_DEVELOPER_GUIDE.md) |
| **JavaScript** | `npm install prsm-sdk` | [sdks/javascript/](sdks/javascript/) |
| **Go** | `go get github.com/prsm-network/PRSM/sdks/go@v0.37.0` | [sdks/go/](sdks/go/) |

```python
from prsm.sdk import PRSMClient

client = PRSMClient("http://localhost:8000")
result = await client.query("EV trends in NC", budget=10.0, privacy="standard")
quote = await client.quote("EV trends", shards=5, tier="t2")
```

---

## Production Deployment

```bash
# Systemd service
sudo cp deploy/production/prsm-node.service /etc/systemd/system/
sudo cp deploy/production/prsm.env.template /opt/prsm/.env
sudo systemctl enable prsm-node && sudo systemctl start prsm-node

# Docker (2-node demo)
docker-compose -f docker/docker-compose.demo.yml up
```

See [Deployment Guide](deploy/production/DEPLOYMENT_GUIDE.md) for full instructions.

---

## Project Stats

| Metric | Value |
|--------|-------|
| Version | 1.7.0 |
| MCP Tools | 18 |
| SDKs | Python, JavaScript, Go |
| FTNS Token | Live on Base mainnet (chain 8453) — `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` |
| Bootstrap | `wss://bootstrap-us.prsm-network.com:8765` (159.203.129.218, NYC3 droplet, single-region today; Europe + Asia expansion planned) |
| License | MIT |

---

## Documentation

**Start here:** [`docs/INDEX.md`](docs/INDEX.md) — navigable map of the full docs surface with audience-oriented entry points (auditors / Foundation officers / research partners / new engineers).

**Commonly-referenced docs:**

| Document | Description |
|----------|------------|
| [Getting Started](docs/GETTING_STARTED.md) | Install → configure → first query in 5 minutes |
| [Technology Choices](docs/TECH_CHOICES.md) | Why Ethereum + Base — chain-choice rationale |
| [Sovereign-Edge AI Spec](docs/archive/SOVEREIGN_EDGE_AI_SPEC.md) | Phase 1 architecture (Rings 1-6) |
| [Confidential Compute Spec](docs/CONFIDENTIAL_COMPUTE_SPEC.md) | Phase 2 architecture (Rings 7-10) |
| [Audit-Gap Roadmap](docs/2026-04-10-audit-gap-roadmap.md) | Master roadmap — phase status + scope + planning-artifact pointers |
| [Implementation Status](docs/IMPLEMENTATION_STATUS.md) | Subsystem status and test coverage |
| [Deployment Guide](deploy/production/DEPLOYMENT_GUIDE.md) | Production deployment walkthrough |
| [SDK Developer Guide](docs/SDK_DEVELOPER_GUIDE.md) | Building on PRSM |

---

## Contributing

```bash
git clone https://github.com/prsm-network/PRSM.git
cd PRSM && pip install -e ".[dev]"
pytest --timeout=120    # Run test suite
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Reporting Security Issues

Found a vulnerability? See [SECURITY.md](SECURITY.md) for the disclosure
policy, scope, and bounty information. **Do not** file public issues for
security reports — use [GitHub Security Advisories](https://github.com/prsm-network/PRSM/security/advisories/new)
or `security@prsm-network.com`.

---

## Third-Party Components

PRSM's `prsm/compute/parallax_scheduling/` module vendors and modifies
the scheduler from [GradientHQ/parallax](https://github.com/GradientHQ/parallax)
(Apache License 2.0, upstream commit `c8c8ebdaaf2924b6d25e2d1caff61e27374cce0b`).
The PRSM-original delta is the four trust adapters layered on top
(anchor verification, tier gating, stake-weighted profile trust,
consensus-mismatch challenges). See:

- `licenses/PARALLAX-APACHE-2.0.txt` — verbatim upstream LICENSE
- `licenses/PARALLAX-NOTICE.txt` — derivative-works disclosure +
  pinned commit + modification log
- Per-file 6-line attribution headers on every vendored module under
  `prsm/compute/parallax_scheduling/`

---

**License:** MIT (PRSM-original code) + Apache-2.0 (vendored Parallax components — see above) | **Website:** [prsm-network.com](https://www.prsm-network.com) | **PyPI:** [prsm-network](https://pypi.org/project/prsm-network/)
