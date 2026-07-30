# PRSM Quickstart Guide

Get from zero to cross-node compute in 10 minutes. This guide walks you through setting up a PRSM node, running compute jobs, and connecting multiple nodes.

## Prerequisites

- **Python**: 3.11 or higher
- **Git**: For cloning the repository
- **Disk space**: ~500MB for the codebase and dependencies
- **Operating System**: macOS, Linux, or Windows with WSL

**Optional**:
- **API Keys**: OpenAI or Anthropic keys for real AI inference (mock responses work without them)

Storage uses PRSM's native content store (ContentStore + BitTorrent layer) — no external daemon required. IPFS is no longer a dependency (the native-storage migration completed 2026-05-07).

## Install

### Install from PyPI

```bash
pip install "prsm-network[ml]"
```

### Install from Source

Clone the repository and install PRSM in editable mode:

```bash
# Clone the repository
git clone https://github.com/prsm-network/PRSM.git
cd PRSM

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install PRSM
pip install -e ".[ml]"
```

This installs the `prsm` CLI command. Verify it works:

```bash
prsm --help
```

## Start Your Node

Start a PRSM node in local mode:

```bash
prsm node start --no-dashboard
```

**What happens on first start:**

1. **Identity generated**: An Ed25519 keypair is created and stored in `~/.prsm/identity.json`
2. **Welcome grant**: Your node receives an initial FTNS token balance (100 FTNS by default)
3. **Bootstrap attempt**: The node tries to connect to configured bootstrap peers
4. **Local mode**: If no bootstrap peers are reachable, the node operates in local-only mode

You'll see a status table showing your node ID, roles, and FTNS balance:

```
┌───────────────────────────────────────────────────────────────┐
│                      PRSM Node Status                         │
├─────────────────────┬───────────────────────────────────────┤
│ Node ID             │ node_abc123...                        │
│ Display Name        │ prsm-node                             │
│ Roles               │ full                                  │
│ P2P Address         │ ws://0.0.0.0:9001                     │
│ API Address         │ http://127.0.0.1:8000                 │
│ FTNS Balance        │ 100.00                                │
│ Bootstrap           │ none configured (first node/local mode)│
│ CPU                 │ 8 cores                               │
│ RAM                 │ 16.0 GB                               │
│ GPU                 │ none                                  │
└─────────────────────┴───────────────────────────────────────┘

Node is running. Press Ctrl+C to stop.
```

## Verify It Works

With your node running, open a new terminal and verify the API:

```bash
# Health check
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy", "version": "0.2.0"}
```

```bash
# Check your FTNS balance
curl http://localhost:8000/balance
```

Expected response:
```json
{
  "wallet_id": "node_abc123...",
  "balance": 100.0,
  "recent_transactions": [
    {
      "tx_id": "tx_welcome_...",
      "type": "welcome_grant",
      "amount": 100.0,
      "description": "Welcome grant for new node"
    }
  ]
}
```

## Run Your First Compute Job

Submit a benchmark job to test the compute pipeline:

```bash
# Submit a benchmark job
curl -X POST http://localhost:8000/compute/submit \
  -H "Content-Type: application/json" \
  -d '{"job_type": "benchmark", "payload": {"test": "latency"}, "ftns_budget": 1.0}'
```

Expected response:
```json
{
  "job_id": "job_xyz789...",
  "status": "pending",
  "job_type": "benchmark",
  "ftns_budget": 1.0
}
```

**Poll for the result:**

```bash
# Check job status (replace with your job_id)
curl http://localhost:8000/compute/job/job_xyz789...
```

When complete, you'll see:
```json
{
  "job_id": "job_xyz789...",
  "status": "completed",
  "job_type": "benchmark",
  "provider_id": "node_abc123...",
  "result": {
    "latency_ms": 42,
    "throughput_tokens_per_sec": 1250
  },
  "result_verified": true,
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:30:05Z"
}
```

## Enable Real AI Inference

By default, PRSM returns mock responses when no LLM API keys are configured. To enable real AI inference:

**Step 1: Create your environment file**

```bash
cp .env.example .env
```

**Step 2: Add your API key**

Edit `.env` and uncomment one of the following:

```bash
# For OpenAI models
OPENAI_API_KEY=sk-your_openai_api_key_here

# For Anthropic models
ANTHROPIC_API_KEY=sk-ant-your_anthropic_api_key_here
```

**Step 3: Restart your node**

Stop the running node with `Ctrl+C` and start it again:

```bash
prsm node start --no-dashboard
```

**Step 4: Submit an inference job**

```bash
curl -X POST http://localhost:8000/compute/submit \
  -H "Content-Type: application/json" \
  -d '{
    "job_type": "inference",
    "payload": {
      "prompt": "Explain quantum entanglement in one paragraph.",
      "model": "claude-3-sonnet",
      "max_tokens": 200
    },
    "ftns_budget": 5.0
  }'
```

**The difference between mock and real inference:**

| Feature | Mock Mode | Real Inference |
|---------|-----------|----------------|
| Response | Static placeholder | Actual model output |
| Latency | Instant | Model-dependent |
| Cost | Free | Consumes FTNS |
| Use case | Development/testing | Production workloads |

When you restart with API keys configured, you'll no longer see the warning:
```
⚠️   No LLM API keys detected — inference will return mock responses.
```

## Connect Two Nodes

Run a second node to test P2P networking and cross-node compute:

**Terminal 1 - Start the first node (default ports):**

```bash
prsm node start --no-dashboard
```

This starts Node 1 with:
- P2P port: 9001
- API port: 8000

**Terminal 2 - Start the second node with different ports:**

```bash
prsm node start --no-dashboard --p2p-port 9002 --api-port 8001 --bootstrap ws://127.0.0.1:9001
```

This starts Node 2 with:
- P2P port: 9002
- API port: 8001
- Bootstrap: Connects to Node 1

**Verify the connection:**

```bash
# Check peers on Node 1
curl http://localhost:8000/peers
```

Expected response:
```json
{
  "connected": [
    {
      "peer_id": "node_def456...",
      "address": "ws://127.0.0.1:9002",
      "display_name": "prsm-node",
      "outbound": true
    }
  ],
  "known_count": 1
}
```

**Submit a cross-node job:**

From Node 1, submit a job that can be executed by Node 2:

```bash
curl -X POST http://localhost:8000/compute/submit \
  -H "Content-Type: application/json" \
  -d '{
    "job_type": "inference",
    "payload": {
      "prompt": "What is the speed of light?",
      "model": "claude-3-sonnet",
      "max_tokens": 100
    },
    "ftns_budget": 2.0
  }'
```

The job will be broadcast to connected peers. If Node 2 has compute capability and accepts the job, it will execute and return results.

**Check which node processed your job:**

```bash
curl http://localhost:8000/compute/job/job_xyz789...
```

Look for `"provider_id"` in the response — it will show which node executed the compute.

## Troubleshooting

### `prsm: command not found`

Make sure you've activated the virtual environment and installed PRSM:

```bash
source .venv/bin/activate
pip install -e ".[ml]"
```

### `Address already in use`

The default ports (8000 for API, 9001 for P2P) may be in use. Use different ports:

```bash
prsm node start --no-dashboard --api-port 8001 --p2p-port 9002
```

### `No LLM API keys detected`

This is expected for first-time setup. The node runs in mock mode. To enable real inference, see [Enable Real AI Inference](#enable-real-ai-inference).

### `Bootstrap connection failed`

If you see `DEGRADED local mode`, the node couldn't reach bootstrap peers. This is normal for:
- First node on a network (no peers exist yet)
- Network connectivity issues
- Firewall blocking P2P ports

The node continues to function locally. Jobs will execute on the local node.

### `Insufficient FTNS balance`

Each job consumes FTNS. Check your balance:

```bash
curl http://localhost:8000/balance
```

If depleted, you can:
1. Restart with a fresh identity (new welcome grant)
2. Receive FTNS from another node
3. Contribute compute/storage to earn rewards

## Next Steps

- **API Reference**: Full API documentation at [`docs/api/`](api/) or `http://localhost:8000/docs` when running
- **SDK Documentation**: Python SDK for programmatic access — see [`sdks/`](../sdks/)
- **Contributor Onboarding**: Join development at [`docs/CONTRIBUTOR_ONBOARDING.md`](CONTRIBUTOR_ONBOARDING.md)
- **Architecture Guide**: Deep dive into PRSM internals at [`docs/DEVELOPMENT_GUIDE.md`](DEVELOPMENT_GUIDE.md)
- **Examples Cookbook**: Common patterns and recipes (see the SDK examples under `sdks/`)

## CLI Reference

```bash
# Node management
prsm node start                    # Start node with defaults
prsm node start --no-dashboard     # Start without live dashboard
prsm node start --wizard           # Interactive setup
prsm node start --p2p-port 9002    # Custom P2P port
prsm node start --api-port 8001    # Custom API port
prsm node start --bootstrap ws://peer:9001  # Connect to bootstrap
prsm node peers                    # List connected peers
prsm node info                     # Show node identity

# Compute jobs (via API)
curl -X POST http://localhost:8000/compute/submit -d '...'
curl http://localhost:8000/compute/job/{job_id}

# FTNS tokens
curl http://localhost:8000/balance  # Check balance

# System
prsm status                        # Show system status
prsm serve                         # Start API server (without node)
```

## API Endpoints

When your node is running, these endpoints are available:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/status` | GET | Node status and metrics |
| `/balance` | GET | FTNS balance and transaction history |
| `/peers` | GET | Connected and known peers |
| `/compute/submit` | POST | Submit a compute job |
| `/compute/job/{id}` | GET | Get job status and results |
| `/content/upload` | POST | Upload content to native storage |
| `/content/search` | GET | Search content index |

Interactive API documentation is available at `http://localhost:8000/docs` when the node is running.