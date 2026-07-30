# PRSM Participant Guide — Earn FTNS by Contributing

> **Related docs:** For technical installation / CLI workflow, see [`GETTING_STARTED.md`](GETTING_STARTED.md) and [`quickstart.md`](quickstart.md). For token economics detail, see `PRSM_Tokenomics.md`. For terminology disambiguation, see [`glossary.md`](glossary.md).

Welcome to PRSM (**Protocol for Research, Storage, and Modeling**), a decentralized peer-to-peer infrastructure protocol where anyone can participate in providing compute, storage, or data for AI workloads and earn rewards.

## What is PRSM?

PRSM is a peer-to-peer infrastructure protocol that connects people who have computing resources with users who need compute, storage, and data access for AI workloads. Instead of relying on a single company's datacenters, PRSM distributes work across thousands of consumer devices and independent operators worldwide — including yours.

**Important framing:** PRSM is **infrastructure**, not a model. Third-party LLMs (Claude, GPT, Gemini) do the reasoning; PRSM provides compute, storage, and data access underneath them via MCP (Model Context Protocol). You contribute to the substrate, not to building a model.

**Why does this matter?**

- **Democratized AI infrastructure**: Compute becomes accessible without depending on a handful of hyperscaler datacenters
- **Fair compensation**: Contributors earn FTNS for providing compute, storage, and data
- **Verifiable provenance**: On-chain royalty distribution means every content access pays the creator, enforced by smart contracts rather than platform policy
- **Transparency**: All transactions are publicly verifiable on-chain

PRSM uses a token called **FTNS** (pronounced "photons") to compensate contributors. FTNS is distributed **only as compensation for services rendered** to the network — it is never sold to investors by the foundation. See `PRSM_Tokenomics.md` §3 for the full distribution model.

## What is FTNS?

FTNS is the native token of the PRSM network. It serves three key purposes:

1. **Payment**: Users spend FTNS to run AI queries and access models
2. **Rewards**: Contributors earn FTNS by providing compute, data, or expertise
3. **Governance**: Token holders can vote on network decisions

### How Much Can I Earn?

Earnings depend on what you contribute and how much:

| Contribution Type | Typical Earnings | Requirements |
|-------------------|------------------|--------------|
| GPU Compute | 0.5-5 FTNS/hour | NVIDIA GPU with 8GB+ VRAM |
| CPU Compute | 0.1-1 FTNS/hour | Modern multi-core CPU |
| Data Sharing | 1-50 FTNS/dataset | Curated, verified datasets |
| Model Development | 10-100+ FTNS/model | Working AI models |

*Note: FTNS is **live on Base mainnet** as of 2026-05-04 (treasury / provenance) + 2026-05-07 (full audit-bundle + Phase 8 + Phase 7-storage). FTNS price is not set by the foundation. The foundation does not run continuous market-making, does not announce prices, and does not guarantee appreciation; secondary-market price emerges from third-party trading. Earnings above are illustrative targets that depend on network adoption and are not commitments.*

### Getting Your First FTNS

New users receive a **welcome grant of 100 FTNS** upon registration. This allows you to:
- Run AI queries immediately
- Test the marketplace
- Participate in governance

## How to Participate

### Option A: Contribute Compute Power

Share your computer's processing power to help run AI queries and earn FTNS automatically.

#### Step 1: Install PRSM

PRSM is a Python package. Install with `pip` on any platform (macOS, Linux, Windows with WSL):

```bash
pip install "prsm-network[ml]"
```

Requires Python 3.11+. For the full installation walkthrough including virtualenv setup and first-run configuration, see [`GETTING_STARTED.md`](GETTING_STARTED.md).

#### Step 2: Initialize Your Node

```bash
prsm setup --minimal   # smart defaults, fast path
# OR
prsm setup             # interactive wizard
```

On first run, PRSM generates an Ed25519 identity keypair (stored in `~/.prsm/identity.json`) and credits your node with a welcome grant of 100 FTNS. There is no separate "account" — your node identity *is* your account. Back up `~/.prsm/identity.json` the same way you would back up a crypto wallet: losing it means losing access to your FTNS balance.

Phase 4 wallet SDK has shipped (EIP-4361 SIWE backend verifier + identity-binding + JS SDK + Coinbase Wallet helper + USD-equivalent display wrapper). For the architectural roadmap of remaining work, see `PRSM_Vision.md` §13. The CLI remains the primary interface for node operation; the new MCP tools (below) are the user-facing surface for the most common workflows.

#### Step 3: Start Contributing

```bash
prsm node start --background   # run in the background
prsm node status               # check it's running
prsm node logs -f              # follow logs
```

> **First-time-user reality check (2026-05-14).** The canonical query
> workflow (`prsm_analyze` MCP tool / `POST /compute/forge`) requires the
> QueryOrchestrator to be wired at node-startup time. Set
> `PRSM_QUERY_ORCHESTRATOR_ENABLED=1` in your environment BEFORE starting
> the node. Without it, queries return `"Agent forge not initialized"`.
>
> Additionally, a brand-new node has **no content shards** and the default
> peer-discovery starts at zero — so the first query you submit will
> typically return `"No content shards above the similarity threshold"`.
> This is expected: either upload content yourself (see Option B) or join
> a content-populated peer network. See
> `docs/operations/2026-05-14-user-dogfood-findings.md` for the full
> first-time-user journey + known friction points.

#### Installing the BitTorrent layer (required for content hosting)

PRSM's content layer uses `libtorrent` (the Rasterbar implementation) for
P2P content distribution. **This is a SYSTEM-LEVEL dependency**, not a pip
package — `pip install libtorrent` does NOT work because the PyPI name
resolves to an unrelated package. Use your platform's native channel:

| Platform | Install command |
|---|---|
| macOS (Homebrew) | `brew install libtorrent-rasterbar` |
| Debian / Ubuntu | `sudo apt install python3-libtorrent` |
| Fedora / RHEL | `sudo dnf install rb_libtorrent-python3` |
| Arch | `sudo pacman -S libtorrent-rasterbar` |
| Windows | See [libtorrent docs](https://www.libtorrent.org/python_binding.html) — pre-built wheels not available; build-from-source required |

After installing, verify by running `python -c "import libtorrent;
print(libtorrent.version)"`. If you use `pipx` (the recommended PRSM
install path), libtorrent must be available to the pipx venv — on macOS
the Homebrew install drops it into the global Python site-packages
which the pipx venv inherits.

**Without libtorrent installed:**
- The PRSM node still runs and accepts queries
- `GET /balance`, `GET /info`, MCP query tools all work
- BUT: content uploads via `POST /content/upload` will fail with the
  error `"upload_text returned None ... content_publisher unwired"`
- AND: queries against an empty local content index will return
  `"No content shards above the similarity threshold"`

In other words, you can be a **query consumer** without libtorrent but
you cannot be a **content host** without it. Most contributors will want
both, so install libtorrent up front.

> **Note.** `prsm daemon` is deprecated in favor of `prsm node`. The old
> commands still work (deprecation warning) but will be removed in a
> future release.

Adjust resource allocation via `prsm config set`:

```bash
prsm config set cpu_pct 50     # share 50% of CPU
prsm config set memory_pct 40  # share 40% of RAM
prsm config set gpu_pct 80     # share 80% of GPU (0 = disabled)
prsm config set storage_gb 50  # pledge 50 GB for storage
```

See `prsm config show` for all options. Your node will automatically process compute jobs from the network and earn FTNS.

#### Step 4: Monitor Your Earnings

```bash
curl http://localhost:8000/balance         # FTNS balance + recent transactions
prsm ftns yield-estimate --hours 24        # projected daily earnings
```

For richer balance + USD-equivalent views (post-2026-05-08), use the `prsm_balance_check` MCP tool from Claude Code / Gemini CLI / Cursor / Antigravity — it surfaces FTNS + USD equivalent + node-connected address in one call. See "Cashing Out to Bank" below.

**Tips for Maximum Earnings:**
- Keep your node online during peak hours (typically 9 AM - 9 PM UTC)
- Share GPU resources if available (higher earnings)
- Maintain good internet connectivity (low latency = more queries)
- Keep your system updated for optimal performance

---

### Option B: Share Your Data

Contribute datasets to the PRSM marketplace and earn FTNS when researchers use your data.

#### What Kind of Data is Valuable?

PRSM specializes in scientific and research data:

- **Scientific Papers**: Published research with proper licensing
- **Experimental Results**: Lab data, measurements, observations
- **Code Repositories**: Well-documented research code
- **Datasets**: Curated collections for ML training
- **Model Weights**: Trained neural network parameters

#### Step 1: Prepare Your Data

1. **Organize**: Clean and document your dataset
2. **Format**: Use standard formats (CSV, JSON, Parquet, HDF5)
3. **Document**: Create a README describing:
   - What the data contains
   - How it was collected
   - Any usage restrictions
   - Citation requirements
4. **License**: Choose an appropriate license (MIT, CC-BY, Apache, etc.)

#### Step 2: Upload to PRSM

Using the PRSM interface:
1. Navigate to the "Storage" tab
2. Click "Upload Dataset"
3. Select your files
4. Fill in metadata:
   - Title
   - Description
   - Tags (for discoverability)
   - License type
5. Set pricing (FTNS per access)
6. Click "Publish"

#### Step 3: Earn from Usage

You'll earn FTNS automatically when:
- Users download your dataset
- Users cite your data in their research
- Your data is used in AI model training

**Tips for Data Success:**
- High-quality documentation attracts more users
- Unique datasets command higher prices
- Respond to user questions promptly
- Update your datasets with new data periodically

---

### Option C: Use PRSM via a Third-Party LLM

**PRSM does not host AI models.** The reasoning layer is always a third-party LLM (Claude, GPT, Gemini, local Ollama, etc.). PRSM provides the compute, storage, and data-access substrate *underneath* your LLM via MCP. You spend FTNS when your LLM invokes PRSM tools — not for the reasoning itself.

#### Using PRSM from an MCP-compatible LLM client (live)

The PRSM MCP server is **shipped and live** as of v1.7.0+. Any MCP client (Claude Code, Claude Desktop, Cursor, Gemini CLI, Antigravity, etc.) can invoke 20 PRSM tools. The most user-facing of those:

- `prsm_analyze` — submit a natural-language query to the PRSM distributed compute network (decompose → find shards → dispatch WASM agents → aggregate → settle FTNS)
- `prsm_quote` — get a cost estimate BEFORE committing to a paid query
- `prsm_balance_check` — read FTNS balance + USD equivalent (post-2026-05-08; see "Cashing Out to Bank" below)
- `coinbase_offramp_initiate` — pre-flight cash-out quote (post-2026-05-08; composer-only until CDP commission)
- `prsm_inference` — run TEE-attested model inference with verifiable receipts
- `prsm_dispatch_agent` — async dispatch with JobHistoryRecord-backed status tracking
- `prsm_agent_status` — surfaces job lifecycle (history + escrow tiers)
- `prsm_list_datasets` — browse available datasets with pricing
- `prsm_node_status` — local node health + capability rings
- `prsm_hardware_benchmark` — measure your node's compute tier (T1-T4)
- `prsm_billing_status` — escrow lifecycle for billing reconciliation
- `prsm_search_shards` / `prsm_upload_dataset` — content layer
- `prsm_yield_estimate` / `prsm_stake` / `prsm_revenue_split` / `prsm_settlement_stats` — economic surface
- `prsm_privacy_status` / `prsm_training_status` / `prsm_create_agent` — system surfaces

Your LLM does the reasoning; PRSM tools provide data, compute, and economic primitives the LLM can't hold in its own context. Every paid PRSM tool call debits FTNS from your node's balance.

#### Using PRSM directly via CLI

For terminal-native workflows or scripting, the CLI mirrors the MCP-tool surface:

```bash
# Get a cost estimate first (free)
prsm compute quote "analyze EV registration trends in NC" --shards 5 --tier t2

# Run with a budget
prsm compute run --query "analyze EV registration trends in NC" --budget 1.0
```

This routes through the Ring 1-10 pipeline — decompose → plan → quote → execute → trace. Under the hood, a third-party LLM (configured via `OPENROUTER_API_KEY` or equivalent) does the reasoning; PRSM dispatches WASM agents (SPRKs) to data-holding nodes.

#### Budget controls

- `--budget N` caps FTNS spend at N
- `prsm compute quote ...` estimates cost before committing
- Higher tier (T2, T3) = faster compute but higher cost
- More shards = more parallelism but higher aggregate cost

## Cashing Out to Bank

PRSM is designed so end-users never have to learn crypto-native skills (seed phrases, gas tokens, fragmented exchange/bridge UIs) to access their FTNS earnings. The architecture per `PRSM_Vision.md` §13 Phase 5:

> A user types into Claude Code: *"Cash out $500 of my FTNS to my primary bank account."* The AI invokes `prsm_balance_check` to confirm funds, then `coinbase_offramp_initiate` to compose the transaction. The user's hardware (Pixel + Titan-M2, iPhone + Secure Enclave) prompts for biometric authorization. FTNS swaps to USDC on the Aerodrome pool, USDC offramps via Coinbase to USD, and dollars land in the user's bank account.

### Today's status (2026-05-08)

The MCP-tool layer is **shipped end-to-end**:

- **`prsm_balance_check`** (MCP tool) — read FTNS balance + USD-equivalent + node-connected address. Live now. Backed by `GET /balance/onchain`.

- **`coinbase_offramp_initiate`** (MCP tool) — pre-flight transaction-summary composer. Live now. Returns a quote artifact: *"Swap 500 FTNS → 500 USDC via Aerodrome → $500 USD via Coinbase CDP → primary bank."* **Status: `PENDING_COMMISSION`.** Today's tool returns the quote summary; **it does NOT initiate any on-chain swap or fiat off-ramp.**

### Why "PENDING_COMMISSION"

The execution side gates on three external commissioning events scheduled per `PRSM_Vision.md` §13 gantt for 2026-06-15:

1. **Aerodrome USDC-FTNS pool seeding** — anchors the on-chain exchange rate.
2. **Coinbase Developer Platform (CDP) production credentials** — wires the fiat-edge.
3. **Initial liquidity bootstrap funding decision** — Foundation vs Prismatica balance sheet (under active discussion).

When all three commission, the same `coinbase_offramp_initiate` tool gains an `execute=true` argument that triggers the real flow — the response shape is forward-compatible, so MCP clients you build today against the v1 quote surface continue to work post-commission.

### What's in the box at commission

When CDP commissions, the cash-out flow becomes:

1. **Intent** — you say to your AI: *"Cash out $X to my bank."*
2. **Quote** — the AI invokes `coinbase_offramp_initiate` and shows you the summary (FTNS amount, swap rate, bank account, status).
3. **Hardware authorization** — Coinbase sends a Passkey challenge to your phone (Pixel + Titan-M2, iPhone + Secure Enclave, etc.). You approve via fingerprint or Face Unlock; the private key never leaves the device.
4. **Settlement** — FTNS swaps to USDC on Aerodrome, USDC offramps via Coinbase, USD lands in your bank via ACH or instant-cashout to debit card.
5. **Confirmation** — your AI confirms: *"$X is on its way to your bank."*

### Why no seed phrases / no gas tokens

- **Coinbase Smart Wallet via Passkeys** — wallets are created via FaceID / TouchID / Google or Apple sign-in at `npm install` time. No 12-word seed phrase. Recovery is platform-level (your Apple/Google account), not custodied by PRSM.
- **Paymasters (gasless transactions)** — Coinbase paymaster infrastructure sponsors gas for routine on-chain operations. You "Cash Out" without holding ETH; gas is either deducted from the FTNS being cashed out, or covered by the Foundation as a UX subsidy in the early bootstrap phase.
- **Zero-fee USDC offramp on Base** — Coinbase's recent zero-fee USDC offramp on Base means the Coinbase leg is free. You only pay the Aerodrome pool's swap fee (which is split across the LPs, not extracted by Coinbase or the foundation).

### Privacy + regulatory framing

- **Coinbase, not the foundation, performs KYC** — when your cumulative cash-out volume crosses regulatory thresholds, Coinbase's compliance flow triggers natively. PRSM does NOT build a KYC department, does NOT store sensitive banking data, does NOT make AML determinations. Coinbase is the regulated gateway; PRSM is the upstream protocol. See `PRSM_Tokenomics.md` §5.5.
- **PRSM never transmits banking PII over the wire.** The MCP tools accept a USD amount + an optional bank-account *nickname* (e.g. `"primary"`, `"savings"`); CDP's Offramp SDK resolves the nickname to your actual linked Coinbase account during the Passkey handshake. PRSM does NOT store routing numbers, account numbers, SSNs, or any banking PII.
- **You retain custody.** The Smart Wallet's private key lives in your phone's Secure Enclave / Titan-M2 chip. PRSM cannot move funds without your biometric authorization at execute-time.

### Topping up FTNS for high-cost queries

The reverse path also works — if you need more FTNS for a complex multi-day compute job:

1. Click "Buy Credits" in the dashboard (or invoke the future `prsm_topup_initiate` MCP tool when it ships).
2. Apple Pay / Google Pay / debit → CDP Onramp buys USDC → Aerodrome swap → FTNS deposited to your wallet.
3. Confirmation: "Purchase Successful."

Same biometric flow as cash-out, run in reverse.

---

## Frequently Asked Questions

### Do I need an API key?

No, you don't need an API key to use the basic features. When you create an account, you can:
- Use the PRSM application directly
- Run queries from the web interface
- Contribute compute without any keys

API keys are only needed if you're a developer building applications that connect to PRSM programmatically.

### How much can I earn?

Earnings vary based on:
- **Hardware**: Better GPUs earn more
- **Uptime**: More hours = more earnings
- **Network Demand**: Higher demand = higher rates
- **Data Quality**: Better data = more downloads

Typical contributors earn 5-50 FTNS per day. Top contributors with powerful hardware can earn 100+ FTNS daily.

### Is my data safe?

Yes. PRSM implements several protections:

1. **Sandboxing**: Your data is processed in isolated environments
2. **Encryption**: All data transfers are encrypted
3. **Provenance**: Every access is logged and verifiable
4. **Control**: You can delete your data at any time

When you contribute compute, your personal files are **never** accessed. PRSM only processes data specifically submitted to the network.

### What is FTNS worth?

PRSM is **live on Base mainnet** (since 2026-05-04 / 2026-05-07). FTNS now has real-world value once the bootstrap-phase liquidity venue (Aerodrome USDC-FTNS pool, scheduled per `PRSM_Vision.md` §13 gantt for 2026-06-15) is seeded and gauge-eligible for AERO emissions.

**Important framing:**

- **The foundation does not set ongoing FTNS price.** FTNS is distributed exclusively as compensation for services rendered (creator royalties, node operator compensation, contributor grants). It is never sold by the foundation to retail.
- **Initial pool seeding is a one-time bootstrap act, distinct from ongoing market-making.** The foundation (or Prismatica, depending on which entity holds the bootstrap inventory) deposits both tokens at the intended starting ratio when the Aerodrome USDC-FTNS pool launches — this is a discrete event that anchors price discovery, NOT continuous market-making operations. Post-seeding, the pool is permissionless infrastructure that any LP can deepen. Same shape as Helium / io.net / other DePIN-token pool launches. See `PRSM_Tokenomics.md` §3.7 for full pool architecture.
- **Day-to-day price discovery is third-party-driven** via the Aerodrome pool + future CEX listings. The foundation does NOT run continuous market-making, does NOT announce prices, does NOT guarantee appreciation.
- **The protocol includes value-trajectory mechanisms** (Bitcoin-style halving emission schedule toward a 1B hard cap, utility staking locks, organic demand growth) that collectively produce modest appreciation during bootstrap and steady-state stabilization as adoption matures. Appreciation is a consequence of protocol design, not a foundation promise. See `PRSM_Tokenomics.md` §4 for the full design.
- **USD-denominated services pricing.** Service prices on PRSM are denominated in USD and settled in FTNS at the live exchange rate. As FTNS appreciates, services become *cheaper in FTNS terms* — your earned FTNS purchasing power grows even when USD-denominated job costs stay stable. See `PRSM_Tokenomics.md` §4.10.
- **Bear case is a real possibility.** If PRSM fails to achieve adoption, FTNS remains low-value regardless of the halving emission mechanics. Contribute what you can afford to lose the USD-equivalent value of.

### How do I get started without any FTNS?

New users receive a **100 FTNS welcome grant** upon registration. This is enough to:
- Run 50-200 basic AI queries
- Test the marketplace
- Participate in governance votes

If you run low on FTNS, you can:
- Contribute compute to earn more (Option A above)
- Share data for ongoing earnings (Option B above)
- Top up via the Coinbase Onramp once CDP commission completes (post-2026-06-15 per Vision §13 gantt; Apple Pay / Google Pay / debit → USDC → FTNS via Aerodrome pool, all in the background)

### Can I run PRSM on multiple computers?

Yes! You can run PRSM on as many computers as you like. Each computer:
- Connects to your single account
- Earns FTNS independently
- Shows up in your dashboard

This is a great way to maximize earnings if you have access to multiple machines.

### What happens if my computer goes offline?

Nothing bad happens:
- Your node simply disconnects from the network
- Pending work is redistributed to other nodes
- When you reconnect, you automatically start earning again
- Your FTNS balance and history are preserved

PRSM is designed for intermittent connectivity — many contributors only run their nodes part-time.

## Safety and Privacy

### What PRSM Sees

PRSM collects only essential information:
- Account email (for notifications and recovery)
- FTNS transaction history (for transparency)
- Query metadata (for billing and quality)

### What PRSM Doesn't See

PRSM **cannot** access:
- Your personal files
- Data outside the PRSM storage system
- Your identity (unless you choose to share it)
- Your location or IP address (beyond what's needed for network routing)

### Data Ownership

When you upload data to PRSM:
- **You retain ownership** of your data
- You set the license and usage terms
- You can delete your data at any time
- PRSM cannot claim ownership of your contributions

### Security Best Practices

1. **Use a strong password** (12+ characters, mix of types)
2. **Save your recovery phrase** in a secure location
3. **Keep software updated** for security patches
4. **Review permissions** before running third-party tools
5. **Report suspicious activity** to security@prsm.ai

## Community and Support

### Getting Help

- **Documentation**: [docs.prsm.ai](https://docs.prsm.ai)
- **Community Forum**: [community.prsm.ai](https://community.prsm.ai)
- **GitHub Issues**: [github.com/prsm-network/PRSM/issues](https://github.com/prsm-network/PRSM/issues)
- **Email Support**: support@prsm.ai

### Reporting Problems

Found a bug or have a suggestion?
1. Check [GitHub Issues](https://github.com/prsm-network/PRSM/issues) for known issues
2. Submit a new issue with:
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - Screenshots if applicable

### Contributing to Development

PRSM is open source! We welcome contributions:
- **Code**: Submit pull requests on GitHub
- **Documentation**: Help improve guides and API docs
- **Testing**: Report bugs and test new releases
- **Translation**: Help translate PRSM into more languages

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### Staying Updated

- **Newsletter**: Subscribe at [prsm.ai/newsletter](https://prsm.ai/newsletter)
- **Blog**: [blog.prsm.ai](https://blog.prsm.ai)
- **Twitter/X**: [@prsm_protocol](https://twitter.com/prsm_protocol)
- **Discord**: [discord.gg/prsm](https://discord.gg/prsm)

---

*Welcome to the future of decentralized AI. We're excited to have you as part of the PRSM community!*
