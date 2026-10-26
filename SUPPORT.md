# Getting Help with PRSM

PRSM is a peer-to-peer protocol for distributed AI compute, storage, and data with on-chain creator royalties. If you're stuck or have a question, here's where to go.

## Choose your path

### 🟢 I want to use PRSM (run a node, publish data, query the network)

- **Quickstart:** [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) — install → first query in 5 minutes
- **Node operator guide:** [`docs/INDEX.md`](docs/INDEX.md) → node operator section
- **Discord:** [https://discord.gg/R8dhCBCUp3](https://discord.gg/R8dhCBCUp3) — `#node-operators` channel for live help

### 🟢 I want to build on PRSM (SDK integration, MCP tools, agents)

- **SDK guides:** [`docs/SDK_DEVELOPER_GUIDE.md`](docs/SDK_DEVELOPER_GUIDE.md)
- **Python:** `pip install prsm-network`
- **JavaScript:** `npm install prsm-sdk`
- **Go:** `go get github.com/prsm-network/PRSM/sdks/go`
- **Discord:** `#dev` channel — direct Q&A with maintainers
- **MCP integration:** [`docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md`](docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md) for security context; full MCP architecture in `PRSM_Vision.md` §1 + §4

### 🟢 I want to contribute to PRSM (code, docs, governance)

- **Contributing guide:** [`CONTRIBUTING.md`](CONTRIBUTING.md) — fork, dev environment, testing, PR flow
- **Code of conduct:** [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) — community standards
- **Good first issues:** [GitHub issues with `good first issue` label](https://github.com/prsm-network/PRSM/labels/good%20first%20issue)
- **Discord:** `#dev` channel for design discussion; `#general` for community
- **GitHub Discussions:** [https://github.com/prsm-network/PRSM/discussions](https://github.com/prsm-network/PRSM/discussions) for async proposals + Q&A

### 🟢 I think I found a security vulnerability

**Do not file a public issue.** See [`SECURITY.md`](SECURITY.md) for responsible disclosure process.

### 🟢 I have a governance proposal (parameter change, policy change)

- Use the **Governance Proposal** issue template
- Discuss in Discord `#governance` channel
- Foundation board review post-formation per [PRSM-GOV-1 Foundation Governance Charter](docs/2026-04-21-prsm-gov-1-foundation-governance-charter.md)

### 🟢 I'm a journalist, researcher, or potential partner

- **GitHub:** open a Discussion in the [Showcase category](https://github.com/prsm-network/PRSM/discussions)
- **Email:** Foundation contact info on [prsm-network.com](https://www.prsm-network.com)

---

## Response time expectations

| Channel | Typical response |
|---|---|
| Discord (general questions) | minutes to hours during US business hours |
| Discord (technical / dev) | hours to a day |
| GitHub Issues | 1-3 business days |
| GitHub Discussions | 1-7 days depending on category |
| Security disclosure | 24 hours for acknowledgment per SECURITY.md |
| Press / partnership inquiries | 3-5 business days |

PRSM is currently solo-maintained (founder Ryne Schultz) pending Foundation board formation in Q3 2026. Response times will improve as the team scales.

---

## Where help is NOT available

- **PRSM is not a hosted service.** We don't run "PRSM-as-a-service" — there's no cloud dashboard, no SaaS account, no centralized API key. Everyone runs their own node. (Decision deferred per `docs/2026-04-22-phase3-status-and-forward-plan.md` §3.1; reconsidered post-mainnet.)
- **PRSM is not an investment vehicle.** We do not sell FTNS to investors. Investor inquiries go to Prismatica (the for-profit operating company); FTNS is distributed only as compensation for network services. See [`PRSM_Tokenomics.md`](https://github.com/prsm-network/PRSM/blob/main/PRSM_Tokenomics.md) §3.
- **PRSM is not financial advice.** Documents marked "informational" are exactly that. Consult your own counsel for jurisdiction-specific tax / regulatory questions.

---

## Building this resource

If you spot a gap in our documentation, please open an issue with the **Documentation** template — improving onboarding directly increases the network's value to every future user.
