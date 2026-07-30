# Getting Started with PRSM

> PRSM — P2P infrastructure protocol for open-source collaboration. Third-party LLMs (Claude, GPT, Gemini) remain the reasoning layer; PRSM plugs in underneath via MCP as a compute, storage, and data-access substrate.

This guide has two tracks. Start with **Quick Start (5 minutes)** if you want to run a query as fast as possible. Read **Running as a Daemon** / **Configuration** / **Skill Packages** if you want to operate a long-running node.

## Quick Start (5 minutes)

### 1. Install

```bash
pip install "prsm-network[ml]"
```

Or from source:
```bash
git clone https://github.com/prsm-network/PRSM.git
cd PRSM
pip install -e ".[ml]"
```

### 2. Configure

**Connect a third-party LLM** (the LLM is the reasoning layer — PRSM gives it compute, storage, and data access via MCP tools). You can use any MCP-compatible client, or configure a local/remote LLM through OpenRouter:

```bash
# Option A: OpenRouter (recommended — many free models available)
export OPENROUTER_API_KEY="your-key-here"

# Option B: Store in PRSM config
mkdir -p ~/.prsm
echo 'OPENROUTER_API_KEY=your-key-here' >> ~/.prsm/.env
```

Get a free OpenRouter key at https://openrouter.ai/keys

**Optional: Set API key for node security:**
```bash
export PRSM_NODE_API_KEY="your-secret-key"
```

### 3. Check Your Hardware

```bash
prsm node benchmark
```

This shows your hardware supply tier (T1-T4), GPU detection, TFLOPS, and thermal classification.

### 4. Run Your First Query

> **FTNS Tokens Required:** PRSM's distributed compute network requires FTNS tokens
> to pay compute providers and data owners. Every query requires a non-zero budget.
> New nodes receive a welcome grant of 100 FTNS on first start. Use `prsm compute quote`
> to estimate costs before committing.

**Start your node:**
```bash
prsm node start
```

In another terminal, confirm the local model is loaded and serving:
```bash
prsm node inference-status      # READY when the model is loaded
prsm compute models             # lists what this node serves (distilgpt2 by default)
```

**Run a verifiable inference (works out of the box on the local model):**
```bash
prsm compute infer --prompt "The capital of France is" --max-tokens 8
```
This returns the model output, the FTNS cost, and a signed receipt you can verify. It uses the
node's default local model (`distilgpt2`); pass `--model <id>` for another model the node serves.

> `prsm compute run` and `prsm compute submit` route through the multi-agent "forge" pipeline, which
> needs an external LLM backend (e.g. an OpenRouter key) — configure one before using them. For a
> plain verifiable inference on the node's own model, use `prsm compute infer` above.

**Check earnings estimate (what would I earn as a provider?):**
```bash
prsm ftns yield-estimate --hours 8
```

## Interactive Setup (Alternative)

If you'd rather configure interactively:

```bash
prsm setup
```

This walks through system detection (CPU, RAM, GPUs, disk), role selection (Contribute / Consume / Both), resource allocation, network config, and AI assistant integration (MCP server for Claude Desktop / Hermes / OpenClaw).

Quick variants:

```bash
prsm setup --minimal   # smart defaults, no prompts
prsm setup --dry-run   # preview without saving
```

## Running as a Daemon

Once configured, you probably want the node running in the background rather than holding a terminal.

### Background daemon (recommended)

```bash
prsm daemon start               # start in background
prsm daemon status              # check status
prsm daemon logs -f             # follow logs live
prsm daemon stop                # stop gracefully
prsm daemon restart             # stop + start
```

`prsm daemon status --format json` is available for scripting.

### Foreground (interactive)

```bash
prsm node start                 # full node with live dashboard
prsm node start --no-dashboard  # full node, static console view
```

### Auto-start on boot

```bash
prsm daemon install             # install launchd (macOS) or systemd (Linux)
prsm daemon install --dry-run   # print service file without installing
prsm daemon uninstall           # remove service
```

## Configuration

Quick commands:

```bash
prsm config show                # display all settings (human-readable)
prsm config show --format json  # machine-readable
prsm config set cpu_pct 60      # change a single setting
prsm config get p2p_port        # get one value
prsm config path                # path to config file (~/.prsm/config.yaml)
prsm config validate            # check config validity
prsm config export              # export current config as YAML
prsm config import file.yaml    # load a config file
prsm config reset               # reset to defaults (confirms first)
```

### Key settings

| Setting | Range | Default | Description |
|---|---|---|---|
| `cpu_pct` | 10-90 | 50 | CPU allocation for compute jobs |
| `memory_pct` | 10-90 | 50 | RAM allocation |
| `gpu_pct` | 0-100 | 80 | GPU allocation (0 = disabled) |
| `storage_gb` | 1+ | 10.0 | Storage pledge in GB |
| `max_concurrent_jobs` | 1-20 | 3 | Max parallel compute jobs |
| `p2p_port` | 1024-65535 | 9001 | P2P network port |
| `api_port` | 1024-65535 | 8000 | REST API port |
| `mcp_server_enabled` | true/false | true | Enable MCP server for AI assistants |
| `mcp_server_port` | 1024-65535 | 9100 | MCP server port |
| `node_role` | full/contributor/consumer | full | Node role |

For the full configuration reference, see [`configuration.md`](configuration.md).

## Explore the Ring Pipeline

The `--query` flag routes through the full Ring 1-10 pipeline:

1. **Decompose** — third-party LLM analyzes what data and operations are needed
2. **Plan** — selects execution route (direct LLM / single agent / swarm)
3. **Quote** — calculates cost before execution
4. **Execute** — routes WASM agents (SPRKs) to the right ring
5. **Trace** — collects execution traces

## Architecture Overview

PRSM is built as 10 concentric "Capability Rings":

| Ring | Name | What It Does |
|---|---|---|
| 1 | The Sandbox | WASM runtime + hardware profiling |
| 2 | The Courier | Mobile agent dispatch + settlement |
| 3 | The Swarm | Semantic sharding + parallel execution |
| 4 | The Economy | Hybrid pricing + prosumer staking |
| 5 | Agent Forge | WASM mobile agent runtime + MCP tools |
| 6 | The Polish | Dynamic gas, CLI, signatures |
| 7 | The Vault | TEE + differential privacy |
| 8 | The Shield | Model sharding + collusion resistance |
| 9 | The Mind | Training pipeline (traces → JSONL export) |
| 10 | The Fortress | Security audit + integrity |

Each ring wraps and enriches the ones inside it. See [`SOVEREIGN_EDGE_AI_SPEC.md`](archive/SOVEREIGN_EDGE_AI_SPEC.md) for details.

> **Terminology note:** "Rings 1-10" is pre-v1.6 numbering for shipped work. Current active planning uses **Phase 1-8** numbering. See [`glossary.md`](glossary.md) for the disambiguation, and [`2026-04-10-audit-gap-roadmap.md`](2026-04-10-audit-gap-roadmap.md) for the current phase plan.

## For Data Providers

Publish a dataset through your node's ContentStore with pricing:

```bash
prsm storage upload ./my_dataset.parquet \
  --description "My Dataset 2025" \
  --royalty-rate 0.05 \
  --replicas 5
```

Revenue split: **80% to you** (data owner), 15% to compute providers, 5% to network.

## For Compute Providers

Check what you'd earn:

```bash
prsm ftns yield-estimate --hours 20
```

Staking is **utility-only** — locking FTNS buys service discounts and dispatch priority (and, at the top tier, aggregator eligibility). It does **not** pay token yield/APY:
- **Casual** (0 FTNS): no lock
- **Pledged** (100 FTNS): service discount + dispatch priority
- **Dedicated** (1,000 FTNS): larger discount + higher dispatch priority
- **Sentinel** (10,000 FTNS): maximum discount + dispatch priority + aggregator eligibility

## Skill Packages

PRSM ships with built-in skill packages that any MCP-compatible AI can ingest:

```bash
prsm skills list                # list installed skills
prsm skills search <query>      # search the network for skills
prsm skills install <pkg>       # install a skill package
prsm skills remove <pkg>        # remove a skill package
prsm skills info <pkg>          # show detailed skill info
```

## Python SDK

```python
from prsm.sdk import PRSMClient

client = PRSMClient("http://localhost:8000", api_key="your-key")

# Full forge pipeline
result = await client.query("EV trends in NC", budget=10.0, privacy="standard")

# Get cost estimate
quote = await client.quote("EV trends", shards=5, tier="t2")

# Upload dataset
await client.upload_dataset("my-data", "My Dataset", content=data_bytes, shard_count=4)
```

## Node Status API

```bash
# Check ring initialization
curl http://localhost:8000/rings/status

# Node health
curl http://localhost:8000/status
```

## Next Steps

- [Quickstart Guide](quickstart.md) — more detailed cross-node walkthrough
- [Sovereign-Edge AI Spec](archive/SOVEREIGN_EDGE_AI_SPEC.md) — architecture details
- [Confidential Compute Spec](CONFIDENTIAL_COMPUTE_SPEC.md) — privacy architecture
- [AI Integration Guide](ai-integration.md) — connecting AI assistants via MCP
- [Configuration Reference](configuration.md) — full config options
- [Implementation Status](IMPLEMENTATION_STATUS.md) — subsystem details
- [Glossary](glossary.md) — disambiguation reference (Tier systems, Ring/Phase, entity structure, acronyms)
- [Master Roadmap](2026-04-10-audit-gap-roadmap.md) — current phase plan

Running `prsm node start` connects you to the bootstrap network automatically.
