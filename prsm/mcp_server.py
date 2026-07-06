"""
PRSM MCP Server
================

Model Context Protocol server that exposes PRSM tools to any LLM.
Enables Claude, Gemini, or any MCP-compatible model to submit queries,
get cost estimates, browse datasets, and dispatch agents via PRSM.

Usage:
    prsm mcp-server                     # Start via CLI
    python -m prsm.mcp_server           # Start directly

Configure in Claude Desktop:
    ~/.claude/claude_desktop_config.json:
    {
        "mcpServers": {
            "prsm": {
                "command": "prsm",
                "args": ["mcp-server"]
            }
        }
    }
"""

import asyncio
import inspect
import json
import logging
import os
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
)


# Streaming helper type — callable that emits progress notifications when the
# MCP client supplied a `progressToken` in its request meta. Phase 3.x.1 Task 8.
#
# Signature: emit(message: str, progress: float, total: float | None = None) -> None
# Handlers accepting an `emit_progress` keyword argument receive a real emitter
# during streaming requests, or None for non-streaming requests. They use it
# inline like:
#
#     async def handle_prsm_inference(args, *, emit_progress=None):
#         if emit_progress: await emit_progress("Submitting...", 1, 4)
#         result = await _call_node_api(...)
#         if emit_progress: await emit_progress("Inference complete.", 4, 4)
#         return format_response(result)
#
# Progress notifications are SIDE-CHANNEL — they do NOT replace the final
# TextContent response. Non-streaming clients see only the final return value;
# streaming clients see both the progress updates and the final response.
ProgressEmitter = Callable[[str, float, Optional[float]], Awaitable[None]]

logger = logging.getLogger(__name__)

# ── Tool Definitions ─────────────────────────────────────────────────────

# 2026-05-07 (canonical-workflow gap-list delta): tools were hidden
# end-to-end because their backends depended on the deleted Agent
# Forge. The Tool definitions remain in TOOLS below (so the
# call_tool dispatch table still works for explicit invocations)
# but list_tools() filters them so client-side tool discovery does
# not surface them.
#
# 2026-05-08 (B8 unhide pass 1): prsm_analyze re-exposed.
# /compute/forge now duck-type-dispatches on
# QueryOrchestrator.dispatch_query (replacing the deleted Agent
# Forge surface) — operators with PRSM_QUERY_ORCHESTRATOR_ENABLED=1
# get a working analyze path end-to-end.
#
# 2026-05-08 (B8 unhide pass 2): prsm_dispatch_agent re-exposed.
# Its handler already routes through /compute/forge with
# manifest.query — that path now works via the same QO dispatch.
# Caveat: the user-supplied InstructionManifest is pre-validated
# locally (catches malformed manifests early) but the
# QueryOrchestrator re-decomposes server-side; the manifest's
# instruction list is currently advisory rather than executed.
# A separate sprint can wire end-to-end manifest pass-through
# (add manifest= kwarg to QueryOrchestrator.dispatch_query +
# extend /compute/forge body schema) when that becomes
# load-bearing.
#
# 2026-05-08 (B8 unhide pass 3): prsm_agent_status re-exposed.
# /compute/status/{job_id} now reads from node._payment_escrow
# (the only per-job persistent state in the synchronous-from-
# caller-view forge pipeline) and returns the escrow's lifecycle:
# pending / released / refunded / disputed + amount + timing +
# provider winner. Coverage limitation: only jobs that locked an
# escrow (budget > 0) are retrievable; budget=0 jobs (test
# fixtures, free-tier dev mode) are not.
#
# All three originally-hidden tools now functional. The
# BROKEN_TOOLS_HIDDEN set is empty; the constant remains in case
# future endpoints need temporary hiding.
BROKEN_TOOLS_HIDDEN = frozenset()

TOOLS = [
    Tool(
        name="prsm_analyze",
        description=(
            "Submit a natural language query to the PRSM distributed compute network. "
            "Automatically decomposes the query via LLM, finds relevant data shards, "
            "dispatches WASM mobile agents to edge nodes, aggregates results, and "
            "settles FTNS token payments. IMPORTANT: Execution requires FTNS tokens — "
            "the budget_ftns parameter must be greater than 0. Use prsm_quote first "
            "to estimate costs before committing. The minimum budget is 0.01 FTNS."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The analysis query in natural language",
                },
                "budget_ftns": {
                    "type": "number",
                    "description": "FTNS tokens to spend (REQUIRED, minimum 0.01). Use prsm_quote to estimate costs first.",
                    "minimum": 0.01,
                    "default": 10.0,
                },
                "privacy_level": {
                    "type": "string",
                    "description": "Privacy level: none, standard (e=8), high (e=4), maximum (e=1)",
                    "enum": ["none", "standard", "high", "maximum"],
                    "default": "standard",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="prsm_quote",
        description=(
            "Get a cost estimate for a PRSM query BEFORE committing. Returns compute cost, "
            "data access cost, network fee, and total in FTNS tokens. Use this to check "
            "costs before running an expensive analysis."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to estimate costs for",
                },
                "shard_count": {
                    "type": "integer",
                    "description": "Estimated number of data shards (default: 3)",
                    "default": 3,
                },
                "hardware_tier": {
                    "type": "string",
                    "description": "Target hardware tier: t1 (mobile), t2 (consumer), t3 (high-end), t4 (datacenter)",
                    "enum": ["t1", "t2", "t3", "t4"],
                    "default": "t2",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="prsm_list_datasets",
        description=(
            "Browse/search datasets on the PRSM network with pricing + creator provenance. "
            "Keyword search by default; set semantic=true to find CONCEPTUALLY related "
            "datasets (embedding similarity), not just keyword matches. Use prsm_get_dataset "
            "to actually retrieve one."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "search": {
                    "type": "string",
                    "description": "Keyword (or, with semantic=true, a natural-language topic)",
                    "default": "",
                },
                "max_price": {
                    "type": "number",
                    "description": "Maximum base access fee in FTNS",
                },
                "semantic": {
                    "type": "boolean",
                    "description": "Semantic (embedding-similarity) search instead of keyword",
                    "default": False,
                },
            },
        },
    ),
    Tool(
        name="prsm_get_dataset",
        description=(
            "Retrieve a dataset from the PRSM network and VERIFY it. Give a `query` to find + "
            "fetch the best match, or a specific `cid`. Returns a text preview plus an integrity "
            "check (the bytes hash to their content_hash) and the creator's on-chain provenance "
            "attribution — so you can trust the content is intact and know who created it. This "
            "is the one-call find -> fetch -> verify path (the flagship data-consumer action)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language description of the dataset to find + fetch",
                },
                "cid": {
                    "type": "string",
                    "description": "Exact content id to fetch (from prsm_list_datasets); "
                                   "overrides query",
                },
                "semantic": {
                    "type": "boolean",
                    "description": "Use semantic search to resolve `query` (default keyword)",
                    "default": False,
                },
                "max_preview_chars": {
                    "type": "integer",
                    "description": "Max characters of text content to preview (default 8000)",
                    "default": 8000,
                },
            },
        },
    ),
    Tool(
        name="prsm_node_status",
        description=(
            "Check the status of the local PRSM node, including which of the 10 "
            "capability rings are initialized and healthy."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_section7_readiness",
        description=(
            "Sprint 587 — check §7 production-readiness of the local node by "
            "probing PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS (PublisherKeyAnchorClient "
            "construction), PRSM_STAKE_BOND_ADDRESS (StakeManagerClient "
            "construction), and PRSM_BASE_RPC_URL (eth_chainId reachability). "
            "Returns per-component outcomes + overall ready/not_ready. AI "
            "triage equivalent of the sprint-585 `prsm node section7-readiness` "
            "CLI."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_hardware_benchmark",
        description=(
            "Run a hardware benchmark on the local node. Returns compute tier (T1-T4), "
            "GPU detection, TFLOPS, thermal classification, and TEE availability."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_create_agent",
        description=(
            "Create a PRSM mobile agent with a custom instruction manifest. "
            "The agent will execute the specified operations on target data shards.\n\n"
            "AVAILABLE OPERATIONS:\n"
            "- filter: Filter records by field value (params: field, value, operator)\n"
            "- aggregate: Compute sum/count/avg/min/max over records\n"
            "- group_by: Group records by a field before aggregating\n"
            "- sort: Sort records by field (params: field, ascending)\n"
            "- limit: Take first N records (params: n)\n"
            "- count: Count total records matching criteria\n"
            "- sum: Sum a numeric field (params: field)\n"
            "- average: Average a numeric field (params: field)\n"
            "- select: Select specific fields from records (params: fields[])\n"
            "- compare: Compare values across groups or time periods\n"
            "- time_series: Time-based trend analysis (params: date_field, metric_field)\n\n"
            "The instructions are composed as a pipeline: each operation feeds into the next. "
            "PRSM wraps these into a WASM mobile agent that executes securely on remote nodes.\n\n"
            "IMPORTANT: Requires FTNS budget > 0. Use prsm_quote to estimate costs first."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Human-readable description of what this agent does",
                },
                "instructions": {
                    "type": "array",
                    "description": "Ordered list of operations to execute on the data",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {
                                "type": "string",
                                "enum": ["filter", "aggregate", "group_by", "sort", "limit",
                                        "count", "sum", "average", "select", "compare", "time_series"],
                                "description": "The operation to perform",
                            },
                            "field": {
                                "type": "string",
                                "description": "The data field this operation targets",
                            },
                            "value": {
                                "description": "Filter value, threshold, or parameter",
                            },
                            "params": {
                                "type": "object",
                                "description": "Additional operation parameters",
                            },
                        },
                        "required": ["op"],
                    },
                },
                "target_shards": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "CIDs of data shards to process (leave empty for auto-discovery)",
                },
                "hardware_tier": {
                    "type": "string",
                    "enum": ["t1", "t2", "t3", "t4"],
                    "description": "Minimum hardware tier required (t1=mobile, t2=consumer, t3=high-end, t4=datacenter)",
                    "default": "t1",
                },
                "budget_ftns": {
                    "type": "number",
                    "description": "FTNS budget for execution (minimum 0.01)",
                    "minimum": 0.01,
                    "default": 5.0,
                },
            },
            "required": ["query", "instructions"],
        },
    ),
    Tool(
        name="prsm_dispatch_agent",
        description=(
            "Dispatch a previously created agent instruction manifest to the PRSM network. "
            "The agent will be sent to nodes holding the target data shards, executed in a "
            "WASM sandbox, and results aggregated.\n\n"
            "Use prsm_create_agent to build the instruction manifest, then prsm_dispatch_agent "
            "to execute it. Or use prsm_analyze for automatic end-to-end execution.\n\n"
            "IMPORTANT: Requires FTNS budget > 0 and a running PRSM node."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "instructions_json": {
                    "type": "string",
                    "description": "JSON instruction manifest from prsm_create_agent",
                },
                "budget_ftns": {
                    "type": "number",
                    "description": "FTNS budget for execution",
                    "minimum": 0.01,
                    "default": 5.0,
                },
            },
            "required": ["instructions_json"],
        },
    ),
    Tool(
        name="prsm_agent_status",
        description="Check the status of a dispatched mobile agent by its agent ID or job ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "The job or agent ID to check"},
            },
            "required": ["job_id"],
        },
    ),
    Tool(
        name="prsm_search_shards",
        description=(
            "Search for relevant data shards on the PRSM network by semantic similarity. "
            "Returns shards whose content is most relevant to your query, ranked by cosine similarity."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query to find relevant data shards"},
                "dataset_id": {"type": "string", "description": "Optional: limit search to a specific dataset"},
                "top_k": {"type": "integer", "description": "Number of results to return (default: 5)", "default": 5},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="prsm_upload_dataset",
        description=(
            "Upload a dataset to the PRSM network with semantic sharding and pricing. "
            "The dataset will be split into shards, distributed across nodes, and listed "
            "in the marketplace with your pricing terms. Revenue split: 80% to you (data owner), "
            "15% to compute providers, 5% to PRSM treasury."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "description": "Unique identifier for your dataset"},
                "title": {"type": "string", "description": "Human-readable title"},
                "description": {"type": "string", "description": "What this dataset contains"},
                "shard_count": {"type": "integer", "description": "Number of shards to split into", "default": 4},
                "base_access_fee": {"type": "number", "description": "FTNS fee per query against this dataset", "default": 1.0},
                "per_shard_fee": {"type": "number", "description": "Additional FTNS per shard accessed", "default": 0.1},
                "require_stake": {"type": "number", "description": "FTNS stake required for access (anti-scraping)", "default": 0},
            },
            "required": ["dataset_id", "title"],
        },
    ),
    Tool(
        name="prsm_yield_estimate",
        description=(
            "Estimate how much FTNS you would earn as a compute provider based on your hardware, "
            "hours of availability, and staking tier. Staking tiers: Casual (0 FTNS, 1x), "
            "Pledged (100 FTNS, 1.25x), Dedicated (1000 FTNS, 1.5x), Sentinel (10000 FTNS, 2x)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "hours_per_day": {"type": "number", "description": "Hours available for compute per day", "default": 8},
                "stake_amount": {"type": "number", "description": "FTNS staked (determines yield boost tier)", "default": 0},
            },
        },
    ),
    Tool(
        name="prsm_stake",
        description=(
            "Preview or submit a FTNS stake on the running node. "
            "By default returns a tier preview without submitting. Pass execute=true "
            "to actually call POST /staking/stake. "
            "Tiers: Casual (0), Pledged (100+), Dedicated (1000+), Sentinel (10000+). "
            "Higher tiers earn proportionally more per compute job."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "FTNS to stake", "minimum": 0},
                "execute": {
                    "type": "boolean",
                    "description": "If true, actually submit the stake. Default false (preview only).",
                    "default": False,
                },
                "stake_type": {
                    "type": "string",
                    "description": "Stake type — passed through to the staking manager.",
                    "default": "general",
                },
            },
            "required": ["amount"],
        },
    ),
    Tool(
        name="prsm_revenue_split",
        description=(
            "Calculate how revenue would be distributed for a given payment. "
            "Default split: 80% data owner, 15% compute providers, 5% PRSM treasury. "
            "When no proprietary data is involved: 95% compute, 5% treasury."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "total_payment": {"type": "number", "description": "Total FTNS payment to split"},
                "has_data_owner": {"type": "boolean", "description": "Whether proprietary data is involved", "default": True},
                "compute_providers": {"type": "integer", "description": "Number of compute providers", "default": 1},
            },
            "required": ["total_payment"],
        },
    ),
    Tool(
        name="prsm_settlement_stats",
        description="Get FTNS settlement queue statistics — pending transfers, total settled, gas usage.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_privacy_status",
        description=(
            "Check the differential privacy budget status. Shows total epsilon spent, "
            "remaining budget, and recent privacy-consuming operations."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_training_status",
        description=(
            "Check NWTN training pipeline status — traces collected, corpus quality score, "
            "route coverage, and readiness for fine-tuning."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_billing_status",
        description=(
            "Look up the FTNS billing state for any prior PRSM tool invocation by job_id. "
            "Returns escrow status (pending / released / refunded), amount locked, "
            "requester / provider identifiers, and on-chain transaction references "
            "if settlement reached the chain. Use this to reconcile costs across multiple "
            "tool calls or to investigate why a particular job's escrow did not release."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": (
                        "The job ID returned by a previous PRSM tool call "
                        "(e.g. forge-abc123, infer-def456). Found in the cost-reconciliation "
                        "footer of any FTNS-consuming tool response."
                    ),
                },
            },
            "required": ["job_id"],
        },
    ),
    Tool(
        name="prsm_balance_check",
        description=(
            "Check FTNS token balance + USD equivalent for a wallet "
            "address. V1 reads on-chain via the node's "
            "OnChainFTNSLedger and converts to USD using the "
            "PRSM_FTNS_USD_RATE env var as a static placeholder until "
            "the Aerodrome USDC-FTNS pool is seeded (Vision §13 "
            "Phase 5 gantt: 2026-06-15). Defaults to the node's "
            "connected wallet when no address is supplied."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": (
                        "Optional 0x-prefixed Ethereum address. When "
                        "omitted, returns balance for the node's "
                        "connected wallet."
                    ),
                },
            },
        },
    ),
    Tool(
        name="prsm_arbitration_preview_resolution",
        description=(
            "Compose a dispute-resolution preview from the AI side "
            "panel. Composer-only — DOES NOT call queue.resolve(). "
            "Returns the would-be-applied resolution + conflict-"
            "with-existing detection so council members can confirm "
            "intent before signing on-chain governance proposals "
            "separately. Local-resolve auth model is pending council "
            "ratification; this composer is the safe surface until "
            "that's settled."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "record_id": {
                    "type": "string",
                    "description": "Arbitration record ID.",
                },
                "decision": {
                    "type": "string",
                    "enum": [
                        "upheld_parent",
                        "rejected_parent",
                        "insufficient",
                    ],
                    "description": (
                        "Council decision on the disputed attribution."
                    ),
                },
                "by_council": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Non-empty list of council member identifiers "
                        "endorsing this decision."
                    ),
                    "minItems": 1,
                },
            },
            "required": ["record_id", "decision", "by_council"],
        },
    ),
    Tool(
        name="prsm_arbitration_record_detail",
        description=(
            "Fetch full context for a single content-attribution "
            "dispute record by ID, including its current resolution "
            "state. Council members reviewing a flagged record use "
            "this to gather context (similarity, kind, flagged_at, "
            "resolution if any) before signing on-chain governance "
            "proposals. Backed by GET /content/arbitration/queue/"
            "{record_id}."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "record_id": {
                    "type": "string",
                    "description": (
                        "The arbitration record ID returned by "
                        "prsm_arbitration_status (list view)."
                    ),
                },
            },
            "required": ["record_id"],
        },
    ),
    Tool(
        name="prsm_arbitration_status",
        description=(
            "List pending content-attribution disputes awaiting "
            "council adjudication. Surfaces records flagged in "
            "the disputed similarity band (PRSM-PROV-1 Item 6). "
            "Backed by GET /content/arbitration/queue. Operators "
            "watching for council-action items use this to track "
            "which content uploads are blocked pending arbitration."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_cleanup_stale_escrows",
        description=(
            "Force-cleanup expired PENDING escrows (refund to "
            "requester). Operators use this to immediately reclaim "
            "FTNS from stuck escrows without waiting for the "
            "10-min periodic cleanup loop. Backed by POST "
            "/compute/cleanup-stale. Returns the number of escrows "
            "refunded."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_spend_summary",
        description=(
            "Sum operator's FTNS spend on completed compute jobs "
            "over the last N days (default 30). Counts RELEASED "
            "escrows only — REFUNDED + PENDING are excluded. "
            "Backed by GET /wallet/spend. Useful for cost-tracking "
            "dashboards + budget reconciliation."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Window in days (1..365). Default 30.",
                    "minimum": 1,
                    "maximum": 365,
                    "default": 30,
                },
                "address": {
                    "type": "string",
                    "description": (
                        "Optional 0x address override; defaults "
                        "to node's connected wallet."
                    ),
                },
            },
        },
    ),
    Tool(
        name="prsm_audit_summary",
        description=(
            "Aggregated bucketed counts over the audit ring "
            "buffer for quick ops dashboards. Returns status "
            "buckets (2xx/3xx/4xx/5xx), method counts, and "
            "top-N most-frequent paths. Faster operator triage "
            "than scrolling prsm_audit_recent. Backed by GET "
            "/audit/summary."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "top_paths": {
                    "type": "integer",
                    "description": (
                        "Number of top paths to surface (1..100). "
                        "Default 10."
                    ),
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                },
            },
        },
    ),
    Tool(
        name="prsm_audit_recent",
        description=(
            "Show recent state-changing API requests (POST/PUT/"
            "PATCH/DELETE) on this node from the in-memory audit "
            "ring buffer. Each entry: timestamp, method, path, "
            "requester, status_code, request_id. Useful for "
            "operator triage: 'what just happened on my node?' "
            "Optionally paginate via limit/offset OR filter by "
            "status code (exact like '404' or range like '4xx'/'5xx')."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Page size (1..1000). Default 20.",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 20,
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset. Default 0.",
                    "minimum": 0,
                    "default": 0,
                },
                "status": {
                    "type": "string",
                    "description": (
                        "Optional status filter. Either exact code "
                        "('404') or HTTP range ('2xx', '3xx', '4xx', "
                        "'5xx'). Useful for drilling into errors."
                    ),
                },
                "requester": {
                    "type": "string",
                    "description": (
                        "Optional exact-match filter on requester "
                        "node/identity. Composes with status filter."
                    ),
                },
                "path_prefix": {
                    "type": "string",
                    "description": (
                        "Optional URL path-prefix filter. E.g. "
                        "'/compute/forge' matches both "
                        "/compute/forge AND /compute/forge/quote. "
                        "Composes with status + requester filters."
                    ),
                },
            },
        },
    ),
    Tool(
        name="prsm_webhook_history",
        description=(
            "Recent webhook dispatch attempts (success or failure). "
            "Useful for verifying webhook integration is firing as "
            "expected — operators see event names + URLs + success/"
            "failure + status_code + error. Backed by GET "
            "/admin/webhook-history."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Page size (1..1000). Default 20.",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 20,
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset. Default 0.",
                    "minimum": 0,
                    "default": 0,
                },
            },
        },
    ),
    Tool(
        name="prsm_forge_submit",
        description=(
            "Submit a query through the full Ring 1-10 Agent Forge "
            "pipeline. End-to-end sovereign-edge AI: AgentForge "
            "decomposes the query, finds shards, quotes via "
            "PricingEngine, routes (DIRECT_LLM / SINGLE_AGENT / "
            "SWARM), aggregates, settles FTNS. Returns job_id + "
            "initial status. Use prsm_quote first to estimate cost. "
            "Backed by POST /compute/forge."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to submit.",
                },
                "budget_ftns": {
                    "type": "number",
                    "description": "Max FTNS to spend (default 10.0).",
                    "default": 10.0,
                },
                "shard_cids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of content CIDs to ground "
                        "the query in. Empty = LLM-only response."
                    ),
                },
                "privacy_level": {
                    "type": "string",
                    "enum": ["none", "standard", "high", "maximum"],
                    "description": "Privacy budget tier.",
                    "default": "standard",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="prsm_content_info",
        description=(
            "Look up a specific content record by CID. Returns "
            "filename, size, content_hash, creator_id, providers, "
            "royalty_rate, parent_cids. Use to verify on-chain "
            "registration + see provider list. Backed by GET "
            "/content/{cid}."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "cid": {
                    "type": "string",
                    "description": "Content ID (CID).",
                },
            },
            "required": ["cid"],
        },
    ),
    Tool(
        name="prsm_my_content",
        description=(
            "Paginated list of content uploaded by this node. "
            "Each entry: content_id, filename, size, royalty_rate, "
            "access_count, total_royalties, provenance_tx_hash. "
            "Use to verify on-chain provenance registration + "
            "track royalty accruals. Backed by GET /content/mine."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1, "maximum": 1000, "default": 20,
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0, "default": 0,
                },
            },
        },
    ),
    Tool(
        name="prsm_distribution_trigger",
        description=(
            "Manually trigger pull_and_distribute on-chain. Use when "
            "the PullAndDistributeScheduler has crashed / paused, or "
            "to force an emission round (e.g., after weight "
            "ratification) without waiting for the next scheduled "
            "tick. Permissionless on-chain; caller pays gas. Backed "
            "by POST /admin/distribution/trigger."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_heartbeat_trigger",
        description=(
            "Manually record an on-chain heartbeat. Use when the "
            "HeartbeatScheduler has crashed / paused / been "
            "disabled and the operator wants to avoid the slashing "
            "window opening. Returns tx_hash + status. Backed by "
            "POST /admin/heartbeat/trigger."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_distribution_history",
        description=(
            "Recent on-chain Distributed events observed by the "
            "CompensationDistributorWatcher. Each entry: timestamp, "
            "to_creator, to_operator, to_grant, total_distributed "
            "(all FTNS wei). Operators verify emission rounds are "
            "landing + tracks the 3-pool split over time. Backed "
            "by GET /admin/distribution-history."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1, "maximum": 1000, "default": 20,
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0, "default": 0,
                },
            },
        },
    ),
    Tool(
        name="prsm_heartbeat_history",
        description=(
            "Recent on-chain HeartbeatRecorded events observed by "
            "the StorageSlashingWatcher. Operators verify their "
            "scheduler is landing transactions on-chain. Optional "
            "`provider` filter narrows to a single address. Backed "
            "by GET /admin/heartbeat-history."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Page size (1..1000). Default 20.",
                    "minimum": 1, "maximum": 1000, "default": 20,
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset. Default 0.",
                    "minimum": 0, "default": 0,
                },
                "provider": {
                    "type": "string",
                    "description": (
                        "Optional address filter."
                    ),
                },
            },
        },
    ),
    Tool(
        name="prsm_slash_history",
        description=(
            "Recent on-chain slash events observed by the "
            "StorageSlashingWatcher. Two kinds: "
            "proof_failure_slashed (verification failed) and "
            "heartbeat_missing_slashed (operator missed window). "
            "Optional `provider` filter narrows to a single "
            "address. Backed by GET /admin/slash-history."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Page size (1..1000). Default 20.",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 20,
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset. Default 0.",
                    "minimum": 0,
                    "default": 0,
                },
                "provider": {
                    "type": "string",
                    "description": (
                        "Optional address filter. Omit to see "
                        "fleet-wide events."
                    ),
                },
            },
        },
    ),
    Tool(
        name="prsm_earnings_summary",
        description=(
            "Operator earnings dashboard. Aggregates 3 streams: "
            "(1) royalty.claimable_wei (RoyaltyDistributor), "
            "(2) heartbeat.last_heartbeat + grace_remaining + "
            "at_risk flag (StorageSlashing), (3) distribution."
            "last_distribution + seconds_since "
            "(CompensationDistributor). Each stream isolated — "
            "RPC failure on one doesn't take down others. "
            "Backed by GET /admin/earnings-summary."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_webhook_test",
        description=(
            "Smoke-test the operator's configured webhook URL. "
            "Synthesizes a webhook.test event + dispatches via "
            "the same deliverer the DaemonWatchdog uses; returns "
            "delivery success/failure with attempts + error fields. "
            "Use after configuring PRSM_WEBHOOK_URL to verify "
            "without waiting for a real daemon crash. Backed by "
            "POST /admin/webhook-test."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_canonical_check",
        description=(
            "Verify operator's wired contract addresses match the "
            "canonical pins in networks.py for the active "
            "PRSM_NETWORK. Purpose-built for post-migration "
            "verification: after a contract redeploy ceremony "
            "(e.g., A-08 v2 RoyaltyDistributor 2026-05-09), "
            "operators run this to confirm their node picked up "
            "the new pins without manually inspecting each "
            "subsystem. Renders PASS/FAIL summary + flags any "
            "mismatch with the action hint. Backed by GET "
            "/health/detailed (filters for canonical-match fields)."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_metrics_summary",
        description=(
            "Render the node's Prometheus /metrics exposition as "
            "a human-readable summary for AI side-panel triage. "
            "Distinct from prsm_node_health (subsystem readiness) "
            "— this surfaces actual operational gauge values "
            "(escrow counts, locked FTNS, history size, claimable "
            "royalties, arbitration pending)."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_node_health",
        description=(
            "One-shot operator diagnostic surfacing per-subsystem "
            "readiness: ftns_ledger, payment_escrow, job_history, "
            "royalty_distributor. Backed by GET /health/detailed. "
            "Top-level status is healthy / degraded / unhealthy. "
            "Distinct from prsm_node_status (which focuses on Ring "
            "activation) — use this for ops/troubleshooting when a "
            "subsystem is suspected of misbehaving."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_escrow_lookup",
        description=(
            "Direct-lookup detail view of a single escrow by "
            "escrow_id. Companion to prsm_escrow_summary (list "
            "view); operators investigating a specific escrow "
            "from logs / on-chain tx receipts use this to fetch "
            "full lifecycle metadata. Backed by GET /wallet/escrows/"
            "{escrow_id}."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "escrow_id": {
                    "type": "string",
                    "description": (
                        "Escrow ID (the unique primary key, "
                        "distinct from job_id)."
                    ),
                },
            },
            "required": ["escrow_id"],
        },
    ),
    Tool(
        name="prsm_escrow_summary",
        description=(
            "List active FTNS escrows for the operator's wallet "
            "(or any address via override). Surfaces outstanding "
            "compute-budget commitments — the FTNS amounts locked "
            "in pending compute jobs awaiting settlement. Backed "
            "by GET /wallet/escrows. Default returns PENDING only; "
            "pass include_terminal=true for RELEASED + REFUNDED "
            "audit view."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": (
                        "Optional 0x-prefixed address override. "
                        "Defaults to the node's connected wallet."
                    ),
                },
                "include_terminal": {
                    "type": "boolean",
                    "description": (
                        "When true, returns RELEASED + REFUNDED "
                        "escrows in addition to PENDING. Default "
                        "false (PENDING only)."
                    ),
                    "default": False,
                },
            },
        },
    ),
    Tool(
        name="prsm_jobs_list",
        description=(
            "List recent compute jobs from JobHistoryStore — "
            "covers /compute/forge, /compute/inference, and "
            "/compute/inference/stream paths. Backed by GET "
            "/compute/jobs. Most-recent-first by started_at. "
            "Optional status filter (in_progress | completed | "
            "failed | cancelled). Optional route filter scopes to "
            "a single compute path (forge | inference | "
            "inference_stream | qo_swarm | direct_llm | swarm). "
            "Pagination via offset + limit (max 100/page). Useful "
            "for operator dashboards + 'find my last failed job' "
            "workflows."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": [
                        "in_progress", "completed", "failed", "cancelled",
                    ],
                    "description": "Optional filter by job status.",
                },
                "route": {
                    "type": "string",
                    "description": (
                        "Optional filter by compute route. "
                        "Common values: forge | inference | "
                        "inference_stream | qo_swarm | direct_llm "
                        "| swarm."
                    ),
                    "maxLength": 64,
                },
                "limit": {
                    "type": "integer",
                    "description": "Page size (1..100). Default 20.",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 20,
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset. Default 0.",
                    "minimum": 0,
                    "default": 0,
                },
            },
        },
    ),
    Tool(
        name="prsm_settler_admin",
        description=(
            "Settler write actions: register (POST /settler/"
            "register), unbond (POST /settler/unbond), sign batch "
            "(POST /settler/batch/sign), or slash (POST /settler/"
            "slash/propose). All four are sensitive ops with "
            "server-side auth enforcement. Useful for settlers "
            "managing their bond + multi-sig duties via MCP."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["register", "unbond", "sign", "slash"],
                },
                "settler_id": {"type": "string"},
                "address": {"type": "string"},
                "bond_amount": {"type": "number", "exclusiveMinimum": 0},
                "batch_id": {"type": "string"},
                "signature": {"type": "string"},
                "slash_amount": {"type": "number", "exclusiveMinimum": 0},
                "reason": {"type": "string"},
                "proposer_id": {"type": "string"},
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_inference_quote",
        description=(
            "Pre-flight cost quote for an inference request. "
            "Backed by POST /compute/inference/quote which returns "
            "InferenceExecutor.estimate_cost() WITHOUT executing "
            "or locking escrow. Pair with prsm_inference: quote "
            "first, then submit with budget_ftns ≥ cost_ftns. "
            "Closes the gap that pre-fix forced users to lock "
            "escrow just to discover inference cost."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Prompt text.",
                    "minLength": 1,
                },
                "model_id": {
                    "type": "string",
                    "description": "Model ID (use prsm_models to discover).",
                    "minLength": 1,
                },
                "privacy_tier": {
                    "type": "string",
                    "enum": ["none", "standard", "high", "maximum"],
                    "default": "standard",
                },
                "content_tier": {
                    "type": "string",
                    "enum": ["A", "B", "C"],
                    "default": "A",
                },
                "max_tokens": {"type": "integer"},
                "temperature": {"type": "number"},
            },
            "required": ["prompt", "model_id"],
        },
    ),
    Tool(
        name="prsm_forge_quote",
        description=(
            "Network-aware cost quote for a forge query. Backed "
            "by POST /compute/forge/quote (same path the JS + Go "
            "SDKs use). Distinct from prsm_quote which uses a "
            "purely-local PricingEngine — this tool reflects real "
            "network state including server-side validation. "
            "Pairs with prsm_forge_submit: quote first, then submit "
            "with budget ≥ Total."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The forge query text.",
                    "minLength": 1,
                },
                "shard_cids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional explicit shard CID list. When "
                        "present, server uses len(shard_cids) "
                        "instead of shard_count."
                    ),
                },
                "shard_count": {
                    "type": "integer",
                    "description": "Shard count (1..100). Default 3.",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 3,
                },
                "hardware_tier": {
                    "type": "string",
                    "enum": ["t1", "t2", "t3", "t4"],
                    "description": "Hardware tier. Default t2.",
                    "default": "t2",
                },
                "estimated_pcu_per_shard": {
                    "type": "number",
                    "description": (
                        "Per-shard compute estimate. Default 50."
                    ),
                    "exclusiveMinimum": 0,
                    "default": 50,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="prsm_content_provider_stats",
        description=(
            "Render ContentProvider runtime stats: local content "
            "count, pending requests, content discovery sub-stats, "
            "cumulative fetch telemetry (successful/failed). "
            "Symmetric pair to prsm_index_stats but for the "
            "provider-side fetch pipeline. Backed by GET "
            "/content/provider-stats."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="prsm_content_filter",
        description=(
            "Operator's content-filter CRUD via MCP. Single tool "
            "with `action` selector: list | add_cids | remove_cid "
            "| add_tags | remove_tag | set_action. Per "
            "R9-SCOPING-1 §8 this is operator-local — the "
            "blocklist is never propagated to other operators. "
            "Use to manage your own jurisdiction-specific content "
            "refusal list (CIDs, model tags, action mode). "
            "Backed by /admin/content-filter/* endpoints."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "list", "add_cids", "remove_cid",
                        "add_tags", "remove_tag", "set_action",
                    ],
                },
                "cids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "CID list for action=add_cids "
                        "(max 1000)."
                    ),
                    "maxItems": 1000,
                },
                "cid": {
                    "type": "string",
                    "description": "CID for action=remove_cid.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Tag list for action=add_tags "
                        "(max 100)."
                    ),
                    "maxItems": 100,
                },
                "tag": {
                    "type": "string",
                    "description": "Tag for action=remove_tag.",
                },
                "filter_action": {
                    "type": "string",
                    "enum": [
                        "refuse", "log_and_refuse",
                        "silent_refuse",
                    ],
                    "description": (
                        "Action mode for action=set_action."
                    ),
                },
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_takedown_notices",
        description=(
            "Foundation takedown-notice intake via MCP. Single "
            "tool with `action` selector: list | lookup | record "
            "| apply_to_filter. Per Vision §14 / R9-SCOPING-1 §8 "
            "this surface is information distribution only — "
            "Foundation records notices (DMCA, EU-DSA, etc.); "
            "operators VOLUNTARILY act. apply_to_filter is the "
            "operator's one-call bridge: adds the notice's "
            "target_cid to the local ContentFilterStore + marks "
            "the notice acknowledged. Never enforces, never "
            "propagates blocklists. Backed by "
            "/admin/takedown-notice(s) + "
            "/admin/content-filter/from-notice/{id}."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "list", "lookup", "record",
                        "apply_to_filter", "set_status",
                    ],
                },
                "notice_status": {
                    "type": "string",
                    "enum": [
                        "received", "acknowledged",
                        "disputed", "expired",
                    ],
                    "description": (
                        "New status for action=set_status."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1, "maximum": 1000,
                    "description": (
                        "Page size for action=list "
                        "(default 50)."
                    ),
                },
                "offset": {
                    "type": "integer", "minimum": 0,
                    "description": (
                        "Page offset for action=list "
                        "(default 0)."
                    ),
                },
                "status": {
                    "type": "string",
                    "enum": [
                        "received", "acknowledged",
                        "disputed", "expired",
                    ],
                    "description": (
                        "Status filter for action=list."
                    ),
                },
                "target_cid": {
                    "type": "string",
                    "description": (
                        "Target-CID filter for action=list, "
                        "OR target CID for action=record."
                    ),
                },
                "notice_id": {
                    "type": "string",
                    "description": (
                        "Notice id for action=lookup."
                    ),
                },
                "sender": {
                    "type": "string",
                    "description": (
                        "Notice sender for action=record."
                    ),
                },
                "jurisdiction": {
                    "type": "string",
                    "description": (
                        "Jurisdiction tag for action=record "
                        "(e.g. US-DMCA, EU-DSA)."
                    ),
                },
                "basis": {
                    "type": "string",
                    "description": (
                        "Short statutory basis for "
                        "action=record."
                    ),
                },
                "notice_text": {
                    "type": "string",
                    "description": (
                        "Full notice body for action=record "
                        "(capped 8KB server-side)."
                    ),
                },
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_fiat_surface_health",
        description=(
            "Operator safety check for the Phase 5 fiat "
            "surface. Returns findings keyed by severity "
            "(ERROR / WARN / INFO) for dangerous env-var "
            "combinations — e.g., KYC commissioned without "
            "PERSONA_WEBHOOK_SECRET (sprint-283 pass-through "
            "lets any HTTP caller flip status to VERIFIED). "
            "Each finding ships a remediation hint. "
            "PRSM_FIAT_HEALTH_CHECK_BYPASS=1 demotes ERRORs "
            "to INFO for dev/staging. Backed by GET "
            "/admin/fiat-surface/health."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="prsm_fiat_compliance",
        description=(
            "Fiat compliance audit log query surface. Single "
            "tool with `action` selector: list | summary | "
            "lookup. Records every onramp/offramp/gasless "
            "quote + execute + KYC event for AUSTRAC / FinCEN "
            "/ IRS reporting once Phase 5 ramps are live. "
            "Recording is automatic from the fiat surface "
            "handlers — no write paths via MCP. Set "
            "PRSM_FIAT_COMPLIANCE_LOG_DIR for 5-7yr disk "
            "retention and PRSM_OPERATOR_JURISDICTION to tag "
            "entries. Backed by /admin/fiat-compliance + "
            "/admin/fiat-compliance/summary + "
            "/admin/fiat-compliance/{entry_id}."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "summary", "lookup"],
                },
                "entry_id": {
                    "type": "string",
                    "description": (
                        "Entry id for action=lookup."
                    ),
                },
                "kind": {
                    "type": "string",
                    "enum": [
                        "onramp_quote", "onramp_execute",
                        "offramp_quote", "offramp_execute",
                        "gasless_transfer_quote",
                        "gasless_transfer_execute",
                        "kyc_initiate", "kyc_status_change",
                    ],
                    "description": (
                        "Filter for action=list."
                    ),
                },
                "user_id": {
                    "type": "string",
                    "description": (
                        "Filter for action=list."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1, "maximum": 10000,
                    "description": (
                        "Row cap for action=list (default 100)."
                    ),
                },
                "offset": {
                    "type": "integer", "minimum": 0,
                    "description": (
                        "Page offset for action=list (default 0)."
                    ),
                },
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_kyc",
        description=(
            "KYC vendor adapter inspection + session "
            "initiation. Single tool with `action` selector: "
            "initiate | lookup | list | status. Pluggable "
            "backend (Persona / Onfido / Plaid Identity) — "
            "swap vendors via KYC_VENDOR env var. "
            "PENDING_COMMISSION pattern: pre-commission "
            "returns preview records without vendor API "
            "calls. Per Vision §14 'Crypto-UX adoption "
            "barrier' mitigation: KYC flow is vendor-hosted "
            "(Persona modal, Onfido iframe), user installs "
            "nothing. Backed by /wallet/kyc + "
            "/wallet/kyc/{user_id} + /wallet/kyc/initiate + "
            "/wallet/kyc/status."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "initiate", "lookup", "list", "status",
                    ],
                },
                "user_id": {
                    "type": "string",
                    "description": (
                        "PRSM user id (initiate + lookup)."
                    ),
                },
                "email": {
                    "type": "string",
                    "description": (
                        "User email (initiate only); vendor "
                        "uses this for recovery / notices."
                    ),
                },
                "level": {
                    "type": "string",
                    "enum": ["basic", "enhanced"],
                    "description": (
                        "KYC level for initiate: 'basic' "
                        "(selfie + ID) or 'enhanced' (proof "
                        "of address + source of funds). "
                        "Defaults to 'basic'."
                    ),
                    "default": "basic",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1, "maximum": 10000,
                    "description": (
                        "Row cap for action=list (default 100)."
                    ),
                },
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_pool_quote",
        description=(
            "Aerodrome USDC-FTNS pool inspection. Read-only "
            "(no commission gate). Single tool with `action` "
            "selector: state | quote. state returns the pool's "
            "live reserves + fee tier + block number; quote "
            "computes an exact-amount-in swap quote with "
            "price-impact (slippage only, excludes fee) "
            "telemetry. NOT_CONFIGURED returned pre-Vision-"
            "gantt-2026-06-15 seeding ceremony — once "
            "BASE_RPC_URL + AERODROME_USDC_FTNS_POOL_ADDRESS "
            "are set, real pool state surfaces immediately. "
            "Backed by /wallet/pool/state + /wallet/pool/quote."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["state", "quote"],
                },
                "amount_in": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Exact-input amount for action=quote "
                        "(token base units)."
                    ),
                },
                "token_in": {
                    "type": "string",
                    "description": (
                        "Address of the input token for "
                        "action=quote. Must equal one of the "
                        "pool's token0/token1."
                    ),
                },
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_gasless_transfer",
        description=(
            "Gasless FTNS transfer via Coinbase paymaster — "
            "the user never sees gas or holds ETH. Single tool "
            "with `action` selector: quote | execute | status. "
            "quote returns an estimate-only dry-run artifact; "
            "execute submits the sponsored UserOperation; "
            "status returns paymaster commission state + "
            "cumulative spend telemetry. Per Vision §14 "
            "'Crypto-UX adoption barrier' mitigation: FTNS "
            "transfers feel like normal money. "
            "PENDING_COMMISSION pattern — when paymaster env "
            "keys are absent, quote/execute return preview "
            "records. Backed by /wallet/transfer/gasless + "
            "/wallet/paymaster/status. Sender must already have "
            "a WaaS wallet (use prsm_waas_wallet?action=provision)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["quote", "execute", "status"],
                },
                "from_user_id": {
                    "type": "string",
                    "description": (
                        "Sender's PRSM user id "
                        "(must have a WaaS wallet)."
                    ),
                },
                "to_address": {
                    "type": "string",
                    "description": "Recipient on-chain address.",
                },
                "ftns_amount": {
                    "type": "string",
                    "description": (
                        "FTNS amount as decimal string."
                    ),
                },
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_waas_wallet",
        description=(
            "Coinbase Wallet-as-a-Service (WaaS) — provisions "
            "MPC-secured embedded wallets for end users via "
            "Coinbase CDP. Single tool with `action` selector: "
            "provision | lookup | list | status. Per Vision §14 "
            "'Crypto-UX adoption barrier' mitigation: makes "
            "wallet creation invisible (email in, address out, "
            "no seed phrase). PENDING_COMMISSION pattern: when "
            "COINBASE_CDP_API_KEY_NAME + "
            "COINBASE_CDP_API_KEY_PRIVATE env vars are absent, "
            "returns preview records; real provisioning lands on "
            "Coinbase commission. Backed by /wallet/waas(/*) "
            "endpoints."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "provision", "lookup", "list", "status",
                    ],
                },
                "user_id": {
                    "type": "string",
                    "description": (
                        "PRSM user id (provision + lookup)."
                    ),
                },
                "email": {
                    "type": "string",
                    "description": (
                        "User email (provision only); used by "
                        "Coinbase CDP for recovery and notices."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1, "maximum": 10000,
                    "description": (
                        "Row cap for action=list (default 100)."
                    ),
                },
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_insurance_fund",
        description=(
            "Vision §14 mitigation item 2: Foundation "
            "reserves 5% of treasury as a dedicated "
            "insurance fund for exploit recovery. Single "
            "tool with `action` selector: status | "
            "compose_recovery. Status returns reserve ratio "
            "(actual vs target), fund + treasury balances, "
            "target_met flag — public, on-chain "
            "verification per the §14 promise. "
            "compose_recovery returns a Safe-uploadable "
            "ERC-20 transfer payload that moves insurance "
            "funds to a recovery wallet during post-exploit "
            "response. Composer-only — Foundation Safe "
            "multisig gates execution. Backed by "
            "/admin/insurance-fund/*."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "status", "compose_recovery",
                    ],
                },
                "recipient": {
                    "type": "string",
                    "description": (
                        "0x-prefixed 40-hex Ethereum "
                        "recovery wallet address (required "
                        "for compose_recovery)."
                    ),
                },
                "amount_wei": {
                    "type": "integer",
                    "description": (
                        "Recovery amount in FTNS wei "
                        "(required for compose_recovery; "
                        "must be > 0)."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Short statement of recovery "
                        "rationale for audit trail (required "
                        "for compose_recovery)."
                    ),
                },
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_emergency_pause",
        description=(
            "Vision §14 smart-contract exploit response. "
            "Single tool with `action` selector: status | "
            "compose_pause | compose_unpause. Status returns "
            "per-contract paused state for all Foundation-"
            "Safe-owned OZ Pausable contracts (FTNS token, "
            "RoyaltyDistributor, EscrowPool, StakeBond, "
            "EmissionController, etc.). Compose returns a "
            "Safe-uploadable tx payload for multi-sig "
            "signing — PRSM never executes pause directly. "
            "Purpose: SAVE OPERATORS TIME constructing pause "
            "calldata by hand during an active incident. "
            "Backed by /admin/emergency-pause/*."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "status", "compose_pause",
                        "compose_unpause",
                    ],
                },
                "contract_name": {
                    "type": "string",
                    "description": (
                        "Contract registry name "
                        "(ftns_token / royalty_distributor "
                        "/ escrow_pool / stake_bond / "
                        "settlement_registry / "
                        "signature_verifier / "
                        "emission_controller / "
                        "compensation_distributor / "
                        "storage_slashing / "
                        "key_distribution). Required for "
                        "compose_pause + compose_unpause."
                    ),
                },
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_pipeline_inference",
        description=(
            "Sprint 312 — federated/pipeline inference "
            "orchestrator. Coordinates multi-stage "
            "TEE-attested inference across a partitioned "
            "model. Each stage runs a subset of the model's "
            "layers; activations flow through stages with "
            "hash-chain integrity. Output is a "
            "verifiable PipelineInferenceReceipt — anyone "
            "with the orchestrator's pubkey can confirm "
            "the inference ran end-to-end without trusting "
            "any single stage operator. Single tool with "
            "`action` selector: propose | list | lookup | "
            "execute | get_round. v1 uses default stub "
            "stage runners; real PyTorch per-stage forward "
            "pass lands in sprint 314."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "propose", "list", "lookup",
                        "execute", "get_round",
                    ],
                },
                "job_id": {"type": "string"},
                "model_id": {"type": "string"},
                "total_layers": {"type": "integer"},
                "node_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "prompt_b64": {"type": "string"},
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_federated_train",
        description=(
            "Vision §7 capstone follow-on (sprint 308b): "
            "trigger the worker-side training shim for a "
            "given (job_id, round_index, dataset_cid). The "
            "worker runs its training strategy, signs the "
            "resulting gradient + its TEE attestation under "
            "its Ed25519 privkey, and returns the signed "
            "GradientUpdate. The caller then submits the "
            "returned update to /admin/federated/job/.../"
            "update via the prsm_federated_learning tool. "
            "Backed by POST /compute/train on this worker "
            "node (requires PRSM_FEDERATED_WORKER_PRIVKEY)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "round_index": {"type": "integer"},
                "dataset_cid": {"type": "string"},
                "sample_count": {"type": "integer"},
            },
            "required": [
                "job_id", "round_index", "dataset_cid",
                "sample_count",
            ],
        },
    ),
    Tool(
        name="prsm_federated_learning",
        description=(
            "Vision §7 Enterprise Confidentiality Mode "
            "capstone: federated-learning orchestrator. "
            "Coordinates round-by-round training across a "
            "fleet of TEE-attested PRSM workers that see "
            "only gradient updates, never plaintext. "
            "Single tool with `action` selector: propose | "
            "list | lookup | issue_round | aggregate. "
            "Aggregation strategies: 'fedavg' (weighted "
            "average by sample_count) or 'fedmedian' "
            "(element-wise median; Byzantine-robust). "
            "Composes onto sprint 304/307 (recipient + "
            "threshold encryption), 305/305a (TEE policy), "
            "306/306a ($CORP capability). Backed by "
            "/admin/federated/*."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "propose", "list", "lookup",
                        "issue_round", "aggregate",
                        "register_worker_key",
                        "list_worker_keys",
                    ],
                },
                "job_id": {"type": "string"},
                "node_id": {"type": "string"},
                "signing_pubkey_b64": {"type": "string"},
                "model_id": {"type": "string"},
                "dataset_cids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "worker_pool": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "rounds_target": {"type": "integer"},
                "min_workers_per_round": {
                    "type": "integer",
                },
                "aggregation": {"type": "string"},
                "round_index": {"type": "integer"},
                "status": {"type": "string"},
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_corp_capability",
        description=(
            "Vision §7 Enterprise Confidentiality Mode "
            "layer 2: soulbound $CORP authorization "
            "capability. Single tool with `action` "
            "selector: keypair_gen | register_issuer | "
            "list_issuers | redeem | get_ledger | "
            "get_consumed. Capabilities are dual-signed "
            "(issuer Ed25519 grant + subject Ed25519 "
            "redemption-time signature), making them "
            "soulbound in practice — a leaked capability "
            "without the subject's device key is useless. "
            "Not the security gate (encryption is, sprint "
            "304; TEE policy is, sprint 305/305a). This is "
            "the ergonomics + accounting + audit layer. "
            "keypair_gen runs fully offline. Backed by "
            "/admin/corp/*."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "keypair_gen", "register_issuer",
                        "list_issuers", "redeem",
                        "get_ledger", "get_consumed",
                    ],
                },
                "kind": {
                    "type": "string",
                    "enum": ["issuer", "subject"],
                },
                "issuer_id": {"type": "string"},
                "signing_pubkey_b64": {"type": "string"},
                "capability_id": {"type": "string"},
                "capability": {"type": "object"},
                "request": {"type": "object"},
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_tee_policy",
        description=(
            "Vision §7 Enterprise Confidentiality Mode "
            "layer 3 — TEE-only execution policy. Single "
            "tool with `action` selector: evaluate | "
            "node_status | list_tiers. Tiers: NONE < "
            "SOFTWARE < HARDWARE_UNVERIFIED < "
            "HARDWARE_VERIFIED. evaluate runs a policy "
            "against an attestation blob (base64). "
            "node_status surfaces THIS node's own effective "
            "attestation tier so enterprises can pre-screen "
            "eligible nodes before dispatching. list_tiers "
            "is a static enum readout (pure client-side). "
            "Backed by /admin/tee-policy/*."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "evaluate", "node_status",
                        "list_tiers",
                    ],
                },
                "attestation_b64": {"type": "string"},
                "min_attestation_tier": {"type": "string"},
                "allowed_vendors": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "require_signature_chain": {
                    "type": "boolean",
                },
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_enterprise_recipient",
        description=(
            "Vision §7 Enterprise Confidentiality Mode "
            "primitives. Single tool with `action` selector: "
            "keypair_gen | encrypt | decrypt | get_manifest. "
            "keypair_gen produces a fresh X25519 recipient "
            "keypair (purely client-side, no network call). "
            "encrypt seals a plaintext under a list of "
            "recipient public keys via hybrid X25519 + "
            "ChaCha20-Poly1305 — OR-decrypt semantics, any "
            "one designated recipient can decrypt. decrypt "
            "unseals an encrypted payload with a private "
            "key. get_manifest fetches the per-recipient "
            "sealed-key manifest for an encrypted CID. The "
            "security claim is rooted in math, not in "
            "token-balance gating — FTNS balance is "
            "irrelevant to the encryption. Backed by "
            "/content/recipient-manifest/* for get_manifest; "
            "other actions are pure client-side."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "keypair_gen", "encrypt", "decrypt",
                        "get_manifest",
                        "encrypt_threshold", "unseal_share",
                        "combine_decrypt",
                    ],
                },
                "plaintext_b64": {"type": "string"},
                "privkey_b64": {"type": "string"},
                "recipients": {
                    "type": "array",
                    "items": {"type": "object"},
                },
                "payload": {"type": "object"},
                "cid": {"type": "string"},
                "threshold": {"type": "integer"},
                "contributions": {
                    "type": "array",
                    "items": {"type": "object"},
                },
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_upgrade",
        description=(
            "Vision §14 mitigation item 7: UUPS upgrade "
            "orchestrator with pre-committed rollback "
            "escape. Single tool with `action` selector: "
            "propose | list | lookup | update | "
            "compose_upgrade | compose_rollback. propose "
            "captures rationale + target proxy + new and "
            "PREVIOUS implementations (the rollback "
            "destination is locked at propose time). "
            "Workflow: proposed → reviewed → safe_uploaded "
            "→ executed → rolled_back | rejected. "
            "compose_upgrade returns a Safe-uploadable "
            "upgradeToAndCall(newImpl, initData) payload "
            "(requires REVIEWED+); compose_rollback returns "
            "the equivalent payload but with the recorded "
            "previous_implementation as the target — "
            "requires EXECUTED status. Composer-only — "
            "Foundation Safe 2-of-3 hardware multisig gates "
            "all execution. Backed by /admin/upgrade/*."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "propose", "list", "lookup",
                        "update", "compose_upgrade",
                        "compose_rollback",
                    ],
                },
                "proposal_id": {"type": "string"},
                "target_proxy": {"type": "string"},
                "new_implementation": {"type": "string"},
                "previous_implementation": {
                    "type": "string",
                },
                "severity": {"type": "string"},
                "rationale": {"type": "string"},
                "init_calldata_hex": {"type": "string"},
                "reviewer_assignments": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "status": {"type": "string"},
                "new_status": {"type": "string"},
                "safe_tx_hash": {"type": "string"},
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_formal_verification",
        description=(
            "Vision §14 mitigation item 4: pinned formal-"
            "invariant registry + runtime probe. Single tool "
            "with `action` selector: list | check | check_one. "
            "list returns the PUBLIC invariant spec for a "
            "contract (no backend required — §14 transparency "
            "promise). check runs all invariants for a given "
            "contract against live on-chain state, returning "
            "per-invariant PASS / FAIL / SKIPPED with "
            "diagnostics. check_one runs a single invariant by "
            "id. RoyaltyDistributor v2 has five pinned "
            "invariants: NETWORK_FEE_BPS == 200 "
            "(anti-confiscation), networkTreasury immutable, "
            "owner == Foundation Safe, balance >= "
            "totalClaimable (THE solvency invariant), and "
            "paused() observability. Backed by "
            "/admin/formal-verification/*."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "list", "check", "check_one",
                    ],
                },
                "contract": {"type": "string"},
                "invariant_id": {"type": "string"},
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_incident",
        description=(
            "Vision §14 mitigation item 5: public exploit-"
            "response playbook + code hooks. Single tool "
            "with `action` selector: open | list | lookup | "
            "advance | event | recommend | comms | playbook. "
            "Severity tiers: s0 (catastrophic active drain) "
            "/ s1 (critical confirmed) / s2 (high suspected) "
            "/ s3 (low / informational). Phase machine: "
            "detected → triaged → contained → mitigated → "
            "postmortem_published → closed (one-way). The "
            "decision tree + comms templates are PRE-"
            "COMMITTED and PUBLIC — that's the §14 promise. "
            "recommend returns operator imperatives at "
            "current (severity, phase); comms returns a "
            "pre-committed markdown comms template with the "
            "summary interpolated. playbook returns the full "
            "decision tree + all comms templates (public "
            "transparency). Backed by /admin/incident/*."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "open", "list", "lookup",
                        "advance", "event",
                        "recommend", "comms", "playbook",
                    ],
                },
                "incident_id": {"type": "string"},
                "severity": {"type": "string"},
                "summary": {"type": "string"},
                "affected_contracts": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "related_disclosure_id": {"type": "string"},
                "phase": {"type": "string"},
                "new_phase": {"type": "string"},
                "note": {"type": "string"},
                "actor": {"type": "string"},
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_disclosure",
        description=(
            "Vision §14 mitigation item 3: responsible-"
            "disclosure intake + bounty payout composer. "
            "Single tool with `action` selector: "
            "submit | list | lookup | update | "
            "compose_payout | record_payout_tx. submit takes "
            "severity (critical/high/medium/low/informational) "
            "+ summary + affected_contracts + "
            "researcher_contact + details (anonymous OK). "
            "list paginates; lookup detail-views one record. "
            "update transitions workflow status (received → "
            "triaged → confirmed → awarded | rejected | "
            "duplicate | out_of_scope). compose_payout returns "
            "a Safe-uploadable ERC-20 transfer payload for "
            "an AWARDED disclosure — composer-only, "
            "Foundation Safe 2-of-3 hardware multisig gates "
            "execution. record_payout_tx closes the audit "
            "trail after Safe-executed payout. Backed by "
            "/admin/disclosure/*."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "submit", "list", "lookup", "update",
                        "compose_payout", "record_payout_tx",
                    ],
                },
                "disclosure_id": {"type": "string"},
                "severity": {"type": "string"},
                "summary": {"type": "string"},
                "affected_contracts": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "researcher_contact": {"type": "string"},
                "details": {"type": "string"},
                "status": {"type": "string"},
                "new_status": {"type": "string"},
                "triage_notes": {"type": "string"},
                "payout_ftns": {"type": "integer"},
                "recipient": {"type": "string"},
                "tx_hash": {"type": "string"},
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_verify_inference_privacy",
        description=(
            "Verify an InferenceReceipt's privacy claims "
            "(Vision §7 zero-trust compute layer). "
            "Independently checks the signature, the "
            "differential-privacy noise application "
            "(ε>0 iff privacy_tier != none), AND the "
            "hardware-attestation quality. Surfaces the "
            "truth about whether a receipt carries a real "
            "hardware-TEE attestation (Intel ASP / AMD KDS "
            "/ Apple SEP) or the current DEV-ONLY software "
            "fallback. Default posture is permissive; pass "
            "require_hardware_attestation=true to reject "
            "software-fallback receipts. Backed by "
            "POST /compute/receipt/verify."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "receipt": {
                    "type": "object",
                    "description": (
                        "Full InferenceReceipt payload from "
                        "POST /compute/inference."
                    ),
                },
                "public_key_b64": {
                    "type": "string",
                    "description": (
                        "Base64-encoded Ed25519 public key "
                        "of the settler node that signed "
                        "the receipt (fetch via GET "
                        "/node/identity/pubkey)."
                    ),
                },
                "require_hardware_attestation": {
                    "type": "boolean",
                    "description": (
                        "When true, fail (ok=false) on "
                        "DEV-ONLY software-fallback "
                        "attestations. Default false."
                    ),
                    "default": False,
                },
                "require_dp_noise": {
                    "type": "boolean",
                    "description": (
                        "When true, fail if the receipt's "
                        "epsilon_spent is 0 while "
                        "privacy_tier claims privacy. "
                        "Default false."
                    ),
                    "default": False,
                },
            },
            "required": ["receipt", "public_key_b64"],
        },
    ),
    Tool(
        name="prsm_content_fingerprint",
        description=(
            "Content fingerprint registry inspection "
            "(Vision §14 item 3 cryptographic dedup). "
            "Single tool with `action` selector: list | "
            "lookup. First-creator-wins semantics — a "
            "creator who re-uploads someone else's content "
            "doesn't get to claim royalties; canonical "
            "creator is whoever registered the SHA-256 "
            "fingerprint first. Backed by "
            "/marketplace/fingerprint(/{content_hash})."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "lookup"],
                },
                "content_hash": {
                    "type": "string",
                    "description": (
                        "SHA-256 content hash for "
                        "action=lookup."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1, "maximum": 10000,
                    "description": (
                        "Row cap for action=list "
                        "(default 100)."
                    ),
                },
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_creator_stake",
        description=(
            "Creator stake management (Vision §14 item 2). "
            "Single tool with `action` selector: balance | "
            "stake | slash. High-tier creator status requires "
            "bonded FTNS collateral that can be slashed on "
            "misbehavior — economic disincentive for spam. "
            "PENDING_COMMISSION pattern: in-memory mirror "
            "until CREATOR_STAKE_REGISTRY_ADDRESS + "
            "BASE_RPC_URL are set; real contract delegation "
            "post-deploy. Backed by "
            "/marketplace/creator-stake/*."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["balance", "stake", "slash"],
                },
                "creator_id": {
                    "type": "string",
                    "description": (
                        "Creator id (required for all "
                        "actions)."
                    ),
                },
                "amount_wei": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Amount in FTNS wei "
                        "(stake/slash actions)."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Audit-trail reason for action=slash "
                        "(e.g. 'confirmed spam', "
                        "'CSAM detection')."
                    ),
                },
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_creator_reputation",
        description=(
            "Creator-side reputation visibility (Vision §14 "
            "data quality / Sybil resistance). Distinct from "
            "prsm_marketplace_reputation which scores compute "
            "providers — this scores content uploaders based "
            "on access frequency + distinct-purchaser breadth "
            "+ repeat-purchase rate. Spam pattern (many "
            "uploads, no repeats) is discriminated from real "
            "value (return visits). Single tool with `action` "
            "selector: list | lookup. Score is 0..1 with "
            "cold-start neutral 0.5. Backed by "
            "/marketplace/creator-reputation(/{id})."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "lookup"],
                },
                "creator_id": {
                    "type": "string",
                    "description": (
                        "Creator id for action=lookup."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1, "maximum": 10000,
                    "description": (
                        "Row cap for action=list (default 100)."
                    ),
                },
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_marketplace_reputation",
        description=(
            "Operator visibility into marketplace "
            "ReputationTracker. Single tool with `action` "
            "selector: list | lookup. list returns the "
            "score-desc-sorted provider table (per-provider "
            "successes/failures/preempted/slashed counts + "
            "p50/p95 latency + slash markers). lookup returns "
            "single-provider detail incl on-chain slash event "
            "history (batch_id, reason, wei amount, tx_hash). "
            "Backed by /marketplace/reputation(/{id}). Cold-start "
            "neutral score (0.5) returned for unknown providers "
            "AND known-but-<10-sample providers per the "
            "ReputationTracker contract."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "lookup"],
                },
                "provider_id": {
                    "type": "string",
                    "description": (
                        "Provider id for action=lookup."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1, "maximum": 10000,
                    "description": (
                        "Row cap for action=list (default 100)."
                    ),
                },
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_pinned_stats",
        description=(
            "Render per-pinned-content storage challenge stats: "
            "cid, size, requester, last_verified timestamp, "
            "successful/failed challenge counts. Backed by GET "
            "/storage/pinned-stats. Useful for storage operators "
            "verifying their pinned data is actively being "
            "challenged + proven."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="prsm_provider_reputations",
        description=(
            "Render cross-provider reputation + challenge stats. "
            "Sorted by reputation desc (most-trusted first). "
            "Backed by GET /storage/provider-reputations. Useful "
            "for picking reliable seeding partners or "
            "investigating why a specific provider is being "
            "deprioritized."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="prsm_bootstrap_status",
        description=(
            "Render bootstrap connection state for operator "
            "triage: configured/attempted/failed nodes, "
            "connected_count, degraded_mode, retry attempts, "
            "fallback telemetry, BootstrapClient active flag. "
            "Inline ✓/⚠ marker indicates healthy / degraded / "
            "disconnected. Backed by GET /bootstrap/status. Pair "
            "with prsm_peers for full network-side visibility."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="prsm_bootstrap_server_status",
        description=(
            "Probe a bootstrap *server* you are running (your "
            "own droplet — not a canonical bootstrap node you "
            "are connecting to). Hits the server's HTTP "
            "control surface at host:port (default "
            "127.0.0.1:8000 — BootstrapConfig.api_port) and "
            "returns /health + /metrics rendered for triage. "
            "AI-assisted complement to the sprint-390 CLI "
            "`prsm bootstrap-server status`. Operator-"
            "trifecta third corner: prsm_bootstrap_status "
            "reports THIS node's registration state; "
            "prsm_bootstrap_test probes canonical fleet from "
            "MY vantage; prsm_bootstrap_server_status probes "
            "MY OWN bootstrap server. Use when operating a "
            "bootstrap droplet to verify the registration "
            "daemon is healthy + observability surface live."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": (
                        "Bootstrap server host. Default "
                        "127.0.0.1 — pass a hostname / IP "
                        "for remote probes."
                    ),
                },
                "port": {
                    "type": "integer",
                    "description": (
                        "Bootstrap server api_port. Default "
                        "8000."
                    ),
                    "minimum": 1,
                    "maximum": 65535,
                    "default": 8000,
                },
                "timeout": {
                    "type": "number",
                    "description": (
                        "HTTP timeout in seconds. Default 5."
                    ),
                    "minimum": 1,
                    "maximum": 60,
                    "default": 5,
                },
                "include_subsystems": {
                    "type": "boolean",
                    "description": (
                        "Sprint 393 — also fetch "
                        "/health/detailed (sprint 392) and "
                        "include per-subsystem readiness in "
                        "the rendered output. Default false "
                        "(v1 contract preserved)."
                    ),
                    "default": False,
                },
            },
        },
    ),
    Tool(
        name="prsm_bootstrap_test",
        description=(
            "Probe the canonical PRSM bootstrap fleet (US + "
            "EU + APAC) from THIS node's vantage point. "
            "Reports per-host TCP / TLS / WSS handshake "
            "success + latency + cert subject/issuer + "
            "aggregate reachability summary. Operator-"
            "trifecta complement to prsm_bootstrap_status "
            "(which reports THIS node's registration state) "
            "and prsm_peers (which reports the network "
            "graph). Use this to diagnose 'is my regional "
            "bootstrap up, or is something local blocking "
            "me?' Doesn't require a running PRSM node — "
            "probes directly from the MCP server host."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of bootstrap URLs to "
                        "test (wss://host:port). When unset, "
                        "uses the canonical fleet from "
                        "prsm/node/config.py "
                        "(DEFAULT_BOOTSTRAP_NODES + "
                        "FALLBACK_BOOTSTRAP_NODES)."
                    ),
                },
                "timeout": {
                    "type": "number",
                    "description": (
                        "Per-host probe timeout in seconds. "
                        "Default 10."
                    ),
                    "minimum": 1,
                    "maximum": 60,
                    "default": 10,
                },
            },
        },
    ),
    Tool(
        name="prsm_royalty_dispatch_summary",
        description=(
            "Aggregate view over the on-chain content-royalty "
            "dispatch audit ring: total entries, status counts "
            "(sent/failed/skipped_*), total_sent_wei, "
            "by_allocation_mode breakdown, earliest/latest "
            "timestamps. Symmetric to prsm_earnings_summary but "
            "for the OUTGOING royalty flow. Backed by GET "
            "/admin/royalty-dispatch-summary."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="prsm_royalty_dispatch_history",
        description=(
            "Audit trail for the on-chain content-access royalty "
            "dispatcher (sprint 248 activation block). One entry "
            "per shard per forge query when "
            "PRSM_ONCHAIN_CONTENT_ROYALTY_ENABLED=1. Status is one "
            "of: sent / skipped_no_record / skipped_bad_hash / "
            "failed. Optional status + job_id filters. Backed by "
            "GET /admin/royalty-dispatch-history."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Page size (1..1000). Default 20.",
                    "minimum": 1, "maximum": 1000, "default": 20,
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset. Default 0.",
                    "minimum": 0, "default": 0,
                },
                "status": {
                    "type": "string",
                    "enum": [
                        "sent", "skipped_no_record",
                        "skipped_bad_hash",
                        "skipped_zero_amount", "failed",
                    ],
                    "description": "Optional status filter.",
                },
                "job_id": {
                    "type": "string",
                    "description": "Optional job_id filter.",
                },
                "allocation_mode": {
                    "type": "string",
                    "enum": ["uniform", "rate_weighted"],
                    "description": (
                        "Optional filter — show only outcomes "
                        "produced by this allocation policy."
                    ),
                },
            },
        },
    ),
    Tool(
        name="prsm_receipts_list",
        description=(
            "Paginated enumeration of stored InferenceReceipts. "
            "Newest first. Optional model_id filter. Backed by "
            "GET /compute/receipts. Pair with prsm_receipt for "
            "deep-dive on a specific job_id, or prsm_verify_"
            "receipt for signature validation. Useful for "
            "auditors enumerating a node's signed outputs."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Page size (1..1000). Default 20.",
                    "minimum": 1, "maximum": 1000, "default": 20,
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset. Default 0.",
                    "minimum": 0, "default": 0,
                },
                "model_id": {
                    "type": "string",
                    "description": "Optional model_id filter.",
                },
            },
        },
    ),
    Tool(
        name="prsm_receipt",
        description=(
            "Fetch a stored InferenceReceipt by job_id. Backed by "
            "GET /compute/receipt/{job_id}. /compute/inference "
            "writes every signed receipt to an LRU-bounded "
            "ReceiptStore (in-memory by default; filesystem "
            "persistence via PRSM_RECEIPT_STORE_DIR). Useful for "
            "auditors verifying a node's outputs after the fact, "
            "or end-users who didn't save the original "
            "prsm_inference response. Pair with "
            "prsm_verify_receipt to cryptographically validate the "
            "settler_signature."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": (
                        "Job ID returned by prsm_inference."
                    ),
                },
            },
            "required": ["job_id"],
        },
    ),
    Tool(
        name="prsm_pubkey",
        description=(
            "Render the running node's Ed25519 public key for "
            "receipt verification. Backed by GET /node/identity/"
            "pubkey. Returns node_id + public_key_b64. Pair with "
            "prsm_verify_receipt when the receipt's settler_node_"
            "id matches this node; otherwise query the actual "
            "settler's /node/identity/pubkey directly."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="prsm_verify_receipt",
        description=(
            "Verify the Ed25519 signature on an InferenceReceipt. "
            "Caller supplies the receipt as a dict (the shape /"
            "compute/inference returns) plus an optional "
            "public_key_b64. When public_key_b64 is omitted, the "
            "tool fetches /node/identity/pubkey from the running "
            "node — useful when the running node IS the settler. "
            "Returns SIGNATURE VALID + readable receipt fields on "
            "success, or a structured failure diagnostic."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "receipt": {
                    "type": "object",
                    "description": (
                        "Full InferenceReceipt dict (e.g. the "
                        "`receipt` field returned by prsm_inference)."
                    ),
                },
                "public_key_b64": {
                    "type": "string",
                    "description": (
                        "Optional base64-encoded Ed25519 pubkey. "
                        "Omit to fetch from the running node."
                    ),
                },
            },
            "required": ["receipt"],
        },
    ),
    Tool(
        name="prsm_models",
        description=(
            "List inference model_ids the node's executor will "
            "accept. Backed by GET /compute/models which surfaces "
            "node.inference_executor.supported_models(). Use any "
            "returned model_id with prsm_inference or prsm_quote. "
            "Closes the discoverability gap: pre-fix end-users had "
            "to read the prsm_inference tool description's sample "
            "list and hope the operator hadn't customized the "
            "registry."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="prsm_ledger_sync",
        description=(
            "Render ledger gossip-sync statistics: messages "
            "broadcast/received, peers in sync, last sync "
            "timestamp. Backed by GET /ledger/sync/stats. Useful "
            "for verifying the node is participating in ledger "
            "gossip."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="prsm_node_resources",
        description=(
            "Get or update node resource configuration. Routes to "
            "GET /node/resources (view current + effective values) "
            "or PUT /node/resources (update at runtime). Update "
            "accepts any subset of: cpu_allocation_pct, memory_"
            "allocation_pct, storage_gb, max_concurrent_jobs, "
            "gpu_allocation_pct, upload_mbps_limit, download_mbps_"
            "limit, active_hours_start/end, active_days. All fields "
            "are server-side bounded (sprint 207)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get", "update"],
                },
                "cpu_allocation_pct": {"type": "integer", "minimum": 10, "maximum": 90},
                "memory_allocation_pct": {"type": "integer", "minimum": 10, "maximum": 90},
                "storage_gb": {"type": "number", "exclusiveMinimum": 0, "maximum": 1_000_000_000},
                "max_concurrent_jobs": {"type": "integer", "minimum": 1, "maximum": 1_000_000},
                "gpu_allocation_pct": {"type": "integer", "minimum": 10, "maximum": 100},
                "upload_mbps_limit": {"type": "number", "minimum": 0, "maximum": 1_000_000},
                "download_mbps_limit": {"type": "number", "minimum": 0, "maximum": 1_000_000},
                "active_hours_start": {"type": "integer", "minimum": 0, "maximum": 23},
                "active_hours_end": {"type": "integer", "minimum": 0, "maximum": 23},
                "active_days": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0, "maximum": 6},
                    "maxItems": 7,
                },
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_settlement_view",
        description=(
            "Batch-settlement read + flush. Routes to GET "
            "/settlement/pending (un-settled queue), GET "
            "/settlement/history (recent results), or POST "
            "/settlement/flush (manually trigger settlement) via "
            "`action` selector. prsm_settlement_stats covers the "
            "stats view."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["pending", "history", "flush"],
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "History page size (1..200). Default 10."
                    ),
                    "minimum": 1,
                    "maximum": 200,
                    "default": 10,
                },
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="prsm_bridge_history",
        description=(
            "Bridge read endpoints in one tool: status (overall "
            "health + supported chains), list (recent transactions), "
            "lookup (single tx by tx_id). Routes to GET /bridge/"
            "status, GET /bridge/transactions, GET /bridge/"
            "transactions/{tx_id} via `view` selector."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "view": {
                    "type": "string",
                    "enum": ["status", "list", "lookup"],
                },
                "tx_id": {
                    "type": "string",
                    "description": "Required for view=lookup.",
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "List page size (1..200). Default 20."
                    ),
                    "minimum": 1,
                    "maximum": 200,
                    "default": 20,
                },
            },
            "required": ["view"],
        },
    ),
    Tool(
        name="prsm_stake_lookup",
        description=(
            "Single-record lookup for a stake or an unstake "
            "request. Routes to GET /staking/stakes/{id} when "
            "kind='stake', GET /staking/unstake-requests/{id} when "
            "kind='unstake_request'. Use prsm_staking_status for "
            "the full dashboard, this tool for deep-inspecting a "
            "specific record."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["stake", "unstake_request"],
                    "description": "Record type to look up.",
                },
                "id": {
                    "type": "string",
                    "description": "Target stake_id or request_id.",
                },
            },
            "required": ["kind", "id"],
        },
    ),
    Tool(
        name="prsm_get_agent",
        description=(
            "Look up a single agent by id and render the full "
            "record (display_name + capabilities + status + "
            "current allowance from ledger). Backed by GET "
            "/agents/{agent_id}. Use prsm_agents for list/search."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Target agent ID.",
                },
            },
            "required": ["agent_id"],
        },
    ),
    Tool(
        name="prsm_agent_conversations",
        description=(
            "Render recent conversation threads for a single "
            "agent (with last-5-message preview per thread). "
            "Backed by GET /agents/{agent_id}/conversations. "
            "Useful for operators monitoring what their agents "
            "are doing."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Target agent ID.",
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Max conversations to return (1..100). "
                        "Default 10."
                    ),
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                },
            },
            "required": ["agent_id"],
        },
    ),
    Tool(
        name="prsm_index_stats",
        description=(
            "Render content-index statistics: total records, "
            "total bytes indexed, distinct providers, last update. "
            "Backed by GET /content/index/stats. Useful for "
            "operators triaging index health (search latency, "
            "fragmentation)."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="prsm_local_balance",
        description=(
            "Render local-ledger FTNS balance + 20 most-recent "
            "transactions. Backed by GET /balance. Distinct from "
            "prsm_balance_check which hits /balance/onchain "
            "(aggregates on-chain + claimable royalties + "
            "escrowed). Use this when you just want a quick "
            "local-ledger view."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="prsm_transfer",
        description=(
            "Send FTNS to another wallet via signed gossip-"
            "broadcast transfer. Backed by POST /ledger/transfer. "
            "Local-side validates positive + finite amount; server "
            "rejects NaN/Infinity (sprint 199). Returns tx_id + "
            "from/to + amount + timestamp on success. Use "
            "prsm_balance_check first to verify sufficient balance."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "to_wallet": {
                    "type": "string",
                    "description": "Destination wallet ID.",
                    "minLength": 1,
                },
                "amount": {
                    "type": "number",
                    "description": (
                        "Amount of FTNS to transfer. Must be "
                        "positive + finite."
                    ),
                    "exclusiveMinimum": 0,
                },
            },
            "required": ["to_wallet", "amount"],
        },
    ),
    Tool(
        name="prsm_faucet",
        description=(
            "Request testnet FTNS from the node faucet. Backed by "
            "POST /ftns/faucet. 100 FTNS max per request, 1000 "
            "FTNS max per wallet (rate-limited). Disabled in "
            "production via PRSM_FAUCET_ENABLED=0 — returns 403 "
            "with a friendly message if so."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": (
                        "Optional amount (default 100, capped at "
                        "100 server-side)."
                    ),
                    "exclusiveMinimum": 0,
                    "maximum": 100,
                },
                "wallet_id": {
                    "type": "string",
                    "description": (
                        "Optional target wallet (defaults to the "
                        "node's own identity)."
                    ),
                },
            },
        },
    ),
    Tool(
        name="prsm_bridge",
        description=(
            "Bridge FTNS between local balance and external chain. "
            "Routes to POST /bridge/deposit (local burn → remote "
            "mint on destination_chain) or POST /bridge/withdraw "
            "(remote burn → local mint from source_chain) via "
            "`direction` selector. Default chain is 137 (Polygon "
            "mainnet). Useful for users moving FTNS between Base "
            "and other supported chains."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["deposit", "withdraw"],
                    "description": "Bridge direction.",
                },
                "amount": {
                    "type": "number",
                    "description": (
                        "FTNS amount to bridge. Must be positive + "
                        "finite."
                    ),
                    "exclusiveMinimum": 0,
                },
                "chain_address": {
                    "type": "string",
                    "description": "External chain address.",
                    "minLength": 1,
                },
                "destination_chain": {
                    "type": "integer",
                    "description": (
                        "Destination chain ID (deposit only). "
                        "Default 137 (Polygon)."
                    ),
                    "default": 137,
                },
                "source_chain": {
                    "type": "integer",
                    "description": (
                        "Source chain ID (withdraw only). "
                        "Default 137 (Polygon)."
                    ),
                    "default": 137,
                },
            },
            "required": ["direction", "amount", "chain_address"],
        },
    ),
    Tool(
        name="prsm_agent_admin",
        description=(
            "Admin actions on a single agent: set_allowance, "
            "revoke, pause, resume. Routes to POST "
            "/agents/{id}/allowance, DELETE /agents/{id}/allowance, "
            "POST /agents/{id}/pause, or POST /agents/{id}/resume "
            "based on `action` selector. Useful for operators "
            "managing agent budgets + lifecycle without curl."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Target agent ID.",
                },
                "action": {
                    "type": "string",
                    "enum": [
                        "set_allowance", "revoke", "pause", "resume",
                    ],
                    "description": "Which admin action to perform.",
                },
                "amount": {
                    "type": "number",
                    "description": (
                        "Allowance amount (set_allowance only). "
                        "Must be positive + finite."
                    ),
                    "exclusiveMinimum": 0,
                },
                "epoch_hours": {
                    "type": "number",
                    "description": (
                        "Allowance refresh window (set_allowance "
                        "only). Default 24."
                    ),
                    "exclusiveMinimum": 0,
                    "default": 24,
                },
            },
            "required": ["agent_id", "action"],
        },
    ),
    Tool(
        name="prsm_settlers",
        description=(
            "List active Phase-6 settlers OR look up a specific "
            "settler by id. Without `settler_id`, calls GET "
            "/settler/list/active. With `settler_id`, calls GET "
            "/settler/{id}. Useful for verifying who's authorized "
            "to approve batch settlements."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "settler_id": {
                    "type": "string",
                    "description": (
                        "Optional settler ID for lookup. Omit to "
                        "list all active."
                    ),
                },
            },
        },
    ),
    Tool(
        name="prsm_settler_batches",
        description=(
            "List pending multi-sig settlement batches. Backed by "
            "GET /settler/batch/pending. Shows batch_id, transfer "
            "count, total amount, signature_count/threshold, and "
            "approved status per batch. Useful for tracking which "
            "batches still need additional settler signatures."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="prsm_unstake_finalize",
        description=(
            "Finalize an unstake request: withdraw (after the "
            "unlock period) or cancel (before unlock; restores "
            "tokens to active staking). Single tool covers POST "
            "/staking/withdraw/{request_id} and POST "
            "/staking/cancel-unstake/{request_id} via `action` "
            "selector. Use prsm_staking_status to find pending "
            "request_id values and their available_at timestamps."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "request_id": {
                    "type": "string",
                    "description": "ID of the unstake request.",
                },
                "action": {
                    "type": "string",
                    "enum": ["withdraw", "cancel"],
                    "description": (
                        "withdraw: finalize after unlock. cancel: "
                        "abort before unlock, restoring tokens."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Optional reason string (cancel only)."
                    ),
                },
            },
            "required": ["request_id", "action"],
        },
    ),
    Tool(
        name="prsm_claim_rewards",
        description=(
            "Claim accumulated staking rewards. Backed by POST "
            "/staking/claim-rewards. Without `stake_id`, claims "
            "across all of the node's stakes; with `stake_id`, "
            "scopes to that single stake. Returns total rewards "
            "claimed + stakes processed. Use prsm_staking_status "
            "first to see unclaimed reward balance."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "stake_id": {
                    "type": "string",
                    "description": (
                        "Optional specific stake ID. Omit to claim "
                        "across all stakes."
                    ),
                },
            },
        },
    ),
    Tool(
        name="prsm_unstake",
        description=(
            "Request to unstake FTNS tokens. Backed by POST "
            "/staking/unstake. Creates an unstake request that "
            "becomes available for withdrawal after the unstaking "
            "period (default 7 days). amount is optional — omit to "
            "unstake the full stake balance. Use prsm_staking_status "
            "first to find your stake_id."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "stake_id": {
                    "type": "string",
                    "description": "ID of the stake to unstake.",
                },
                "amount": {
                    "type": "number",
                    "description": (
                        "Optional amount to unstake. Omit to unstake "
                        "the full stake balance. Must be positive."
                    ),
                    "exclusiveMinimum": 0,
                },
            },
            "required": ["stake_id"],
        },
    ),
    Tool(
        name="prsm_subsystem_stats",
        description=(
            "Stats for a chosen operator subsystem. Backed by GET "
            "/settler/stats, /storage/stats, or /compute/stats "
            "depending on the `subsystem` selector. Useful for "
            "operators checking single-subsystem health without "
            "loading the full /health/detailed response."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "subsystem": {
                    "type": "string",
                    "enum": ["settler", "storage", "compute"],
                    "description": "Which subsystem to probe.",
                },
            },
            "required": ["subsystem"],
        },
    ),
    Tool(
        name="prsm_staking_status",
        description=(
            "Render the user's full staking dashboard. Backed by "
            "GET /staking/status. Shows active stakes (id, amount, "
            "type, status, rewards earned/claimed), pending unstake "
            "requests with their available_at timestamps, and "
            "totals (staked + earned + claimed). Useful for "
            "stakers tracking positions without grepping the local "
            "staking manager state."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="prsm_agents",
        description=(
            "List or search PRSM agents. Without `capability`, "
            "calls GET /agents (with optional `local_only` filter). "
            "With `capability`, routes to GET /agents/search filtered "
            "by that capability string. Useful for discovering "
            "which agents the operator can dispatch jobs to."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "capability": {
                    "type": "string",
                    "description": (
                        "Optional capability string. When provided, "
                        "calls /agents/search; otherwise lists all."
                    ),
                    "maxLength": 256,
                },
                "local_only": {
                    "type": "boolean",
                    "description": (
                        "When listing (no capability), restrict to "
                        "locally-registered agents. Default false."
                    ),
                    "default": False,
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Max results when searching (1..100). "
                        "Default 20."
                    ),
                    "minimum": 1,
                    "maximum": 100,
                    "default": 20,
                },
            },
        },
    ),
    Tool(
        name="prsm_agent_spending",
        description=(
            "Aggregate spending dashboard across all local agents. "
            "Backed by GET /agents/spending. Returns per-agent spent "
            "+ allowance plus totals. Useful for operators tracking "
            "agent budget burn before granting more FTNS via "
            "/agents/{agent_id}/allowance."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="prsm_peers",
        description=(
            "List currently-connected peers (outbound + inbound). "
            "Backed by GET /peers. Useful for verifying bootstrap "
            "connectivity — degraded mode is typically caused by "
            "no peers reaching the canonical wss:// bootstrap. "
            "Shows direction (outbound/inbound), peer_id, address, "
            "display_name per peer."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="prsm_transactions",
        description=(
            "Render the node's FTNS transaction history. Backed by "
            "GET /transactions. Returns tx_id, type, from/to wallet, "
            "amount, description, timestamp per record. Limit defaults "
            "to 50; capped server-side at 200. Useful for end-users "
            "tracking FTNS flows without grepping the local ledger."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of records (1..200). Default 50.",
                    "minimum": 1,
                    "maximum": 200,
                    "default": 50,
                },
            },
        },
    ),
    Tool(
        name="prsm_info",
        description=(
            "Render static node metadata: node_id, api_version, "
            "network, chain_id, rpc_host, operator_address, "
            "agent_forge_wired, query_orchestrator state/error, "
            "and the full canonical_addresses dict (FTNS token + "
            "ProvenanceRegistry V1/V2 + RoyaltyDistributor + "
            "Foundation Safe + audit-bundle + Phase 7/8 contracts). "
            "Useful for verifying which chain/contracts a node is "
            "pinned to without parsing /health/detailed."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="prsm_cancel_job",
        description=(
            "Cancel a submitted /compute/forge job by job_id. "
            "Backed by POST /compute/cancel/{job_id}. Marks the "
            "JobHistoryStore record CANCELLED and refunds the "
            "PENDING escrow. v1 caveat: in-flight Python "
            "coroutines are NOT interrupted — the release-side "
            "race loses against the now-REFUNDED escrow (correct "
            "outcome). Useful when prsm_status_stream shows a job "
            "stuck IN_PROGRESS beyond expected duration."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": (
                        "Job ID returned by prsm_forge_submit or "
                        "prsm_inference."
                    ),
                },
            },
            "required": ["job_id"],
        },
    ),
    Tool(
        name="prsm_status_stream",
        description=(
            "Stream live status transitions for a submitted job. "
            "Backed by GET /compute/status/{job_id}/stream (Server-"
            "Sent Events). Blocks until the server emits a terminal "
            "event (completed / history_terminal / escrow_terminal / "
            "timeout) OR max_wait_sec elapses (default 60, clamped to "
            "[1, 600]). Returns a rendered trajectory of unique "
            "status snapshots + the terminal reason + the final "
            "status. Closes the gap referenced in prsm_forge_submit's "
            "idempotent-replay hint — end-users no longer need to "
            "hand-poll prsm_agent_status to track progress."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": (
                        "Job ID returned by prsm_forge_submit or "
                        "prsm_inference."
                    ),
                },
                "max_wait_sec": {
                    "type": "number",
                    "description": (
                        "Max seconds to block waiting for terminal "
                        "event. Clamped to [1, 600]. Default 60."
                    ),
                    "default": 60,
                    "minimum": 1,
                    "maximum": 600,
                },
            },
            "required": ["job_id"],
        },
    ),
    Tool(
        name="prsm_royalty_claim",
        description=(
            "Claim accumulated FTNS royalties from RoyaltyDistributor. "
            "Closes the loop on the offramp claim_required path: when "
            "coinbase_offramp_initiate reports `Prerequisite: Claim X "
            "FTNS in royalties`, this tool executes the claim. "
            "Defaults to dry_run=true (returns the artifact + claimable "
            "amount without on-chain action). Pass dry_run=false to "
            "actually execute the on-chain claim() call. v1 caveat: "
            "operator authorization is implicit via running the node "
            "with the configured FTNS private key."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "dry_run": {
                    "type": "boolean",
                    "description": (
                        "When true (default), returns the claimable "
                        "amount + artifact without on-chain action. "
                        "When false, executes the claim() on-chain "
                        "and returns the tx hash."
                    ),
                    "default": True,
                },
            },
        },
    ),
    Tool(
        name="coinbase_onramp_initiate",
        description=(
            "Compose a pre-flight transaction summary for "
            "buying FTNS with USD via Coinbase CDP on-ramp + "
            "the Aerodrome USDC-FTNS pool. Mirror of "
            "coinbase_offramp_initiate in the opposite "
            "direction. V1 returns a PENDING_COMMISSION artifact; "
            "does NOT initiate any fiat charge or on-chain "
            "movement. Real execution gates on Coinbase CDP "
            "on-ramp commission (Vision gantt 2026-06-22). "
            "Either destination_user_id (resolved via WaaS — "
            "use prsm_waas_wallet?action=provision first) or "
            "destination_address is required."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "usd_amount": {
                    "type": "number",
                    "description": (
                        "USD amount to convert to FTNS. Must "
                        "be positive."
                    ),
                    "minimum": 0.01,
                },
                "destination_user_id": {
                    "type": "string",
                    "description": (
                        "PRSM user id; resolved to the user's "
                        "WaaS-managed wallet address. Mutually "
                        "exclusive with destination_address."
                    ),
                },
                "destination_address": {
                    "type": "string",
                    "description": (
                        "Explicit on-chain destination address. "
                        "Mutually exclusive with "
                        "destination_user_id."
                    ),
                },
                "payment_method_alias": {
                    "type": "string",
                    "description": (
                        "Optional payment-method nickname "
                        "(e.g. 'primary', 'savings'). "
                        "Defaults to 'primary'."
                    ),
                    "default": "primary",
                },
            },
            "required": ["usd_amount"],
        },
    ),
    Tool(
        name="coinbase_offramp_initiate",
        description=(
            "Compose a pre-flight transaction summary for cashing out "
            "FTNS to USD via the Aerodrome USDC-FTNS pool + Coinbase "
            "CDP off-ramp. V1 returns the artifact described in "
            "Vision §13 Phase 5 step 2 ('Gemini presents an Artifact "
            "in your side panel'); does NOT initiate any on-chain "
            "swap or fiat off-ramp. Status is PENDING_COMMISSION "
            "until the CDP integration commissions (gates on "
            "Aerodrome pool seeding per Vision gantt 2026-06-15). "
            "Use prsm_balance_check first to confirm sufficient "
            "balance before quoting larger amounts."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "usd_amount": {
                    "type": "number",
                    "description": (
                        "USD amount to off-ramp. Must be positive. "
                        "Tool returns 422 if exceeds available balance."
                    ),
                    "minimum": 0.01,
                },
                "bank_account_alias": {
                    "type": "string",
                    "description": (
                        "Optional bank-account nickname (e.g. 'primary', "
                        "'savings'). Defaults to 'primary'."
                    ),
                    "default": "primary",
                },
            },
            "required": ["usd_amount"],
        },
    ),
    Tool(
        name="prsm_inference",
        description=(
            "Run TEE-attested model inference on PRSM with verifiable receipts. "
            "Routes the prompt through PRSM's confidential-compute layer (Phase 2 TEE + "
            "Phase 7 content-tier gating) and returns the inference output along with a "
            "signed receipt that the caller can independently verify against the settling "
            "node's published Ed25519 public key.\n\n"
            "COSTS FTNS: needs a budget_ftns balance. If it fails with insufficient funds, "
            "get testnet FTNS with prsm_faucet, verify with prsm_local_balance, then retry. "
            "Estimate the cost first with prsm_inference_quote.\n\n"
            "TWO LAYERS OF PRIVACY (per PRSM_Vision.md §7):\n"
            "- content_tier — encryption status of data being queried:\n"
            "    A = public content (default; no encryption)\n"
            "    B = encrypted-before-sharding (Phase 7-storage)\n"
            "    C = Tier B + Reed-Solomon erasure coding + Shamir-split keys\n"
            "- privacy_tier — TEE attestation + DP noise on activations:\n"
            "    none     = no DP noise\n"
            "    standard = ε=8.0 (default)\n"
            "    high     = ε=4.0\n"
            "    maximum  = ε=1.0\n\n"
            "End-to-end privacy for data-sensitive workloads requires both layers configured.\n\n"
            "IMPORTANT: privacy_tier other than 'none' is intended to require a "
            "hardware-backed TEE (SGX / TDX / SEV-SNP / TrustZone / Apple Secure Enclave). "
            "Phase 3.x.1 ships the privacy-budget gate (DP-ε accounting) but defers the "
            "hardware-TEE enforcement gate to Phase 3.x.1 Task 3. On the current mock "
            "executor, software TEE accepts all privacy tiers and the receipt records "
            "tee_type=software — verify TEE type before relying on confidentiality "
            "guarantees.\n\n"
            "REQUIRES FTNS budget > 0. Use prsm_quote first to estimate cost. "
            "Minimum budget: 0.01 FTNS."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The text prompt to send to the model",
                },
                "model_id": {
                    "type": "string",
                    "description": (
                        "Identifier of the model to run. Defaults to the node's local model "
                        "(distilgpt2). Call prsm_models / `prsm compute models` to list what a "
                        "node actually serves before overriding."
                    ),
                    "default": "distilgpt2",
                },
                "budget_ftns": {
                    "type": "number",
                    "description": "FTNS tokens to spend (REQUIRED, minimum 0.01).",
                    "minimum": 0.01,
                    "default": 1.0,
                },
                "privacy_tier": {
                    "type": "string",
                    "description": (
                        "Inference-layer privacy. Default 'none' = usable output on any node. "
                        "standard/high/maximum add activation-DP (ε=8/4/1) but need a hardware-TEE "
                        "node and degrade quality (sp1234); real confidentiality is the TEE tier."
                    ),
                    "enum": ["none", "standard", "high", "maximum"],
                    "default": "none",
                },
                "content_tier": {
                    "type": "string",
                    "description": "Content encryption tier: A (public), B (encrypted), C (encrypted+sharded)",
                    "enum": ["A", "B", "C"],
                    "default": "A",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens to generate (model-dependent; default unbounded within budget)",
                },
                "temperature": {
                    "type": "number",
                    "description": "Sampling temperature 0.0-2.0 (default model-specific)",
                    "minimum": 0.0,
                    "maximum": 2.0,
                },
            },
            "required": ["prompt"],
        },
    ),
]


# ── Tool Handlers ────────────────────────────────────────────────────────

class NodeAPIError(Exception):
    """sp1348 — a node HTTP error (4xx/5xx) raised by _call_node_api so it can't be swallowed
    as data. ``_call_node_api`` used to return ``resp.json()`` regardless of status, so a 4xx
    (FastAPI HTTPException → {"detail": ...}) reached handlers as a plain dict; any handler that
    read success-path keys then rendered fake-success / blank (sp1346/1347 fixed a few by hand).
    Raising at the source fixes the WHOLE class at once — every handler's 4xx now surfaces via
    its own ``except`` or the call_tool dispatch. ``__str__`` folds in the sp1346 funding hint so
    an insufficient-funds error self-guides toward prsm_faucet wherever it's surfaced."""

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = str(message)
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"HTTP {self.status}: {self.message}" + _funding_hint(self.message)


def _raise_for_status(status: Any, body: Any) -> None:
    """sp1348 — raise NodeAPIError when ``status`` is an HTTP error (>= 400), carrying the
    parsed reason (error/detail/success-message via _api_error, else the raw text/status)."""
    if isinstance(status, int) and status >= 400:
        msg = _api_error(body) if isinstance(body, dict) else (str(body)[:300] if body else None)
        raise NodeAPIError(status, msg or f"HTTP {status}")


async def _get_node_api_url() -> str:
    """Get the PRSM node API URL."""
    return os.environ.get("PRSM_NODE_URL", "http://localhost:8000")


async def _call_node_api(
    method: str, path: str, data: Dict = None,
    *, raw_text: bool = False,
):
    """Call the PRSM node API.

    By default returns the response as parsed JSON (Dict).
    Pass ``raw_text=True`` for endpoints that emit text/plain
    bodies (e.g., /metrics Prometheus exposition).
    """
    import aiohttp
    url = await _get_node_api_url()
    api_key = os.environ.get("PRSM_NODE_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async def _read(resp):
        # sp1348 — read the body then RAISE on an HTTP error status so a 4xx/5xx can't be
        # silently returned as data (the fake-success bug class). 2xx returns unchanged.
        if raw_text:
            txt = await resp.text()
            _raise_for_status(resp.status, txt)
            return txt
        body = await resp.json()
        _raise_for_status(resp.status, body)
        return body

    async with aiohttp.ClientSession() as session:
        if method == "GET":
            async with session.get(
                f"{url}{path}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                return await _read(resp)
        elif method == "DELETE":
            # Sprint 221 — added DELETE for agent allowance revoke.
            async with session.delete(
                f"{url}{path}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                return await _read(resp)
        elif method == "PUT":
            # Sprint 232 — added PUT for node-resources update.
            async with session.put(
                f"{url}{path}",
                json=data or {},
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                return await _read(resp)
        else:
            async with session.post(
                f"{url}{path}",
                json=data or {},
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                return await _read(resp)


# ──────────────────────────────────────────────────────────────────────
# Phase 3.x.8.1 Task 3 — SSE streaming client for /compute/inference/stream
# ──────────────────────────────────────────────────────────────────────


class InferenceError(RuntimeError):
    """Structured error from a streaming inference dispatch.

    Raised by ``_call_node_api_streaming`` when the SSE stream
    terminates with an ``event: error`` frame. ``code`` carries the
    machine-readable error code (e.g. ``EXECUTION_FAILURE``,
    ``INTERNAL_ERROR``) and ``message`` is the human-readable
    description. The MCP tool handler maps this to an MCP-friendly
    error response — same surface as a non-success unary response.
    """

    def __init__(self, message: str, *, code: Optional[str] = None):
        super().__init__(message)
        self.code = code
        self.message = message


async def _parse_sse(response: Any) -> AsyncIterator[Tuple[str, str]]:
    """Minimal Server-Sent Events parser. Yields ``(event_type,
    data)`` tuples per W3C SSE spec.

    Frame structure:
      event: <type>\\n
      data: <payload>\\n
      \\n   <- blank line terminates the frame

    Multi-line ``data:`` is concatenated with literal newlines
    between lines. ``event:`` defaults to ``"message"`` when absent
    (per spec). Comment lines (starting with ``:``) and unknown
    fields are silently ignored. The parser does NOT try to handle
    every edge case in the SSE spec — it handles the framing PRSM
    emits, which is the strict ``event:``/``data:``/blank-line
    pattern. Anything more exotic from a peer is ignored at the
    field level rather than crashing the parser.
    """
    event_type = "message"
    data_lines: List[str] = []
    # aiohttp's response.content is an asyncio.StreamReader-shaped
    # object. iter_any() yields raw bytes chunks; we accumulate +
    # split on newlines so a chunk-boundary mid-frame doesn't break
    # parsing.
    buffer = ""
    async for chunk in response.content.iter_any():
        buffer += chunk.decode("utf-8", errors="replace")
        # Process complete lines; keep the trailing partial line in
        # the buffer for the next chunk.
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")  # tolerate CRLF
            if line == "":
                # Blank line — frame terminator.
                if data_lines:
                    yield event_type, "\n".join(data_lines)
                    data_lines = []
                    event_type = "message"
            elif line.startswith(":"):
                # SSE comment — ignored.
                continue
            elif line.startswith("event:"):
                # ``event: <type>`` — strip the prefix + optional
                # leading space (per SSE spec's "leading space
                # after the colon is consumed if present").
                value = line[len("event:"):]
                if value.startswith(" "):
                    value = value[1:]
                event_type = value
            elif line.startswith("data:"):
                value = line[len("data:"):]
                if value.startswith(" "):
                    value = value[1:]
                data_lines.append(value)
            # Other fields (id:, retry:) are silently ignored.
    # If the connection closed mid-line (no trailing newline at all),
    # process the remaining buffered text as a final line. Then if
    # data_lines accumulated anything (with or without a trailing
    # blank-line terminator), flush them as a final frame —
    # defensive against servers that forget the trailing blank line.
    if buffer:
        line = buffer.rstrip("\r")
        if line.startswith("data:"):
            value = line[len("data:"):]
            if value.startswith(" "):
                value = value[1:]
            data_lines.append(value)
        elif line.startswith("event:"):
            value = line[len("event:"):]
            if value.startswith(" "):
                value = value[1:]
            event_type = value
        # Other unterminated fields ignored (id:, retry:, comment).
    if data_lines:
        yield event_type, "\n".join(data_lines)


async def _call_node_api_streaming(
    path: str,
    data: Dict[str, Any],
    emit_progress: ProgressEmitter,
) -> Dict[str, Any]:
    """Open an SSE connection to a node-API endpoint, forward token
    events to ``emit_progress``, return the final result dict on
    terminal ``event: result``.

    Raises:
      ``InferenceError`` when the stream terminates with an
        ``event: error`` frame. The ``code`` attribute carries the
        server-side error code.
      ``RuntimeError`` when the stream closes without a terminal
        ``result`` or ``error`` event (server crash / connection
        drop / etc.).

    Phase 3.x.8.1 Task 3 — wires the
    ``POST /compute/inference/stream`` endpoint (Task 2) to the MCP
    progress-event surface. Caller is responsible for catching
    ``InferenceError`` and formatting an MCP-friendly error
    response.
    """
    import aiohttp
    import json as _json

    url = await _get_node_api_url()
    api_key = os.environ.get("PRSM_NODE_API_KEY", "")
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    sequence_count = 0

    # No total timeout — streaming inference can run for minutes.
    # Per-chunk timeout via sock_read keeps a hung server detectable
    # without bounding the overall stream length.
    timeout = aiohttp.ClientTimeout(total=None, sock_read=120)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            f"{url}{path}",
            json=data or {},
            headers=headers,
        ) as response:
            response.raise_for_status()
            async for event_type, event_data in _parse_sse(response):
                if event_type == "token":
                    try:
                        payload = _json.loads(event_data)
                    except _json.JSONDecodeError as exc:
                        # Malformed token event — surface as an
                        # InferenceError rather than crashing the
                        # iterator. Token frames carry user-visible
                        # text so a parse failure here is a server
                        # bug worth surfacing.
                        raise InferenceError(
                            f"malformed token event: {exc}",
                            code="MALFORMED_RESPONSE",
                        )
                    text_delta = payload.get("text_delta", "")
                    sequence_count += 1
                    await emit_progress(
                        text_delta,
                        float(sequence_count),
                        None,
                    )
                elif event_type == "result":
                    try:
                        return _json.loads(event_data)
                    except _json.JSONDecodeError as exc:
                        raise InferenceError(
                            f"malformed result event: {exc}",
                            code="MALFORMED_RESPONSE",
                        )
                elif event_type == "error":
                    try:
                        err = _json.loads(event_data)
                    except _json.JSONDecodeError:
                        # Even the error event is malformed — surface
                        # as InferenceError with the raw payload.
                        raise InferenceError(
                            f"malformed error event: {event_data!r}",
                            code="MALFORMED_RESPONSE",
                        )
                    raise InferenceError(
                        err.get("error", "unknown inference error"),
                        code=err.get("code"),
                    )
                # Unknown event types are ignored — forward-compat
                # with future server-side event additions.

    # Stream closed without a terminal result/error event.
    raise RuntimeError(
        "SSE stream ended without a 'result' or 'error' event "
        "(server crashed or connection dropped mid-stream)"
    )


MINIMUM_BUDGET_FTNS = 0.01


async def handle_prsm_analyze(
    arguments: Dict[str, Any],
    *,
    emit_progress: Optional[ProgressEmitter] = None,
) -> str:
    """Handle prsm_analyze tool call.

    Streaming-aware (Phase 3.x.1 Task 8): if the MCP client included a
    progressToken, intermediate stages emit as progress notifications.
    Non-streaming clients see only the final return value (unchanged
    backwards-compatible behavior).
    """
    query = arguments.get("query", "")
    budget = arguments.get("budget_ftns", 10.0)
    privacy = arguments.get("privacy_level", "standard")

    # Enforce minimum budget
    if budget <= 0:
        return (
            "PRSM requires an FTNS budget to execute queries. "
            "Set budget_ftns to at least 0.01 FTNS.\n\n"
            "Tip: Use the prsm_quote tool first to estimate costs, "
            "then call prsm_analyze with an appropriate budget."
        )
    if budget < MINIMUM_BUDGET_FTNS:
        return (
            f"Budget {budget} FTNS is below the minimum ({MINIMUM_BUDGET_FTNS} FTNS). "
            f"Use prsm_quote to estimate the required budget for your query."
        )

    if emit_progress:
        await emit_progress("Submitting query to PRSM gateway...", 1.0, 4.0)

    try:
        if emit_progress:
            await emit_progress("Dispatching agents to swarm...", 2.0, 4.0)

        result = await _call_node_api("POST", "/compute/forge", {
            "query": query,
            "budget_ftns": budget,
            "privacy_level": privacy,
        })

        if emit_progress:
            await emit_progress("Aggregating swarm results...", 3.0, 4.0)

        # sp1347 — surface a 4xx (e.g. 402 insufficient funds) instead of rendering an empty
        # "PRSM Analysis Result" with response="" (the /compute/forge money path can 402).
        _e = _api_error(result)
        if _e:
            return f"PRSM analysis rejected: {_e}" + _funding_hint(_e)

        response = result.get("response", "")
        route = result.get("route", "unknown")
        job_id = result.get("job_id", "")

        if emit_progress:
            await emit_progress("Analysis complete.", 4.0, 4.0)

        # Cost reconciliation footer (Phase 3.x.1 Task 7).
        # The /compute/forge response includes job_id but doesn't currently
        # surface the actual settled cost in the response shape — fall back
        # to budget_ftns until the API exposes settled cost as a top-level
        # field. (Recoverable separately via prsm_billing_status.)
        footer = _format_cost_footer(
            job_id=job_id or "unknown",
            cost_ftns=result.get("cost_ftns"),
            budget_ftns=budget,
            extra_fields={
                "Route": route,
                "Privacy level": privacy,
            },
        )

        return (
            f"PRSM Analysis Result\n"
            f"====================\n\n"
            f"{response}\n"
            f"{footer}"
        )
    except Exception as e:
        # sp1393 — the analysis (forge/Rings-1-10) pipeline needs an external LLM backend; when it's
        # absent, point the model at the tool that DOES work on the node's local model so it can
        # self-recover instead of dead-ending.
        _msg = str(e)
        _hint = ""
        if "forge" in _msg.lower() or "backend" in _msg.lower():
            _hint = (" The multi-agent analysis pipeline needs an external LLM backend (set "
                     "OPENROUTER_API_KEY on the node). For a single-shot inference on the node's "
                     "own local model, call the prsm_inference tool instead.")
        return f"PRSM analysis failed: {_msg}.{_hint} Is your PRSM node running? (prsm node start)"


async def handle_prsm_quote(arguments: Dict[str, Any]) -> str:
    """Handle prsm_quote tool call."""
    query = arguments.get("query", "")
    shards = arguments.get("shard_count", 3)
    tier = arguments.get("hardware_tier", "t2")

    try:
        from prsm.economy.pricing import PricingEngine
        engine = PricingEngine()
        quote = engine.quote_swarm_job(
            shard_count=shards,
            hardware_tier=tier,
            estimated_pcu_per_shard=50.0,
        )
        return (
            f"Cost Estimate for: {query}\n"
            f"  Compute: {quote.compute_cost} FTNS\n"
            f"  Data: {quote.data_cost} FTNS\n"
            f"  Network Fee: {quote.network_fee} FTNS\n"
            f"  Total: {quote.total} FTNS\n"
            f"  Hardware Tier: {tier.upper()}\n"
            f"  Shards: {shards}"
        )
    except Exception as e:
        return f"Quote failed: {str(e)}"


async def handle_prsm_list_datasets(arguments: Dict[str, Any]) -> str:
    """Handle prsm_list_datasets tool call.

    Hits /content/search on the running node, optionally filtered by query
    string. Filters by max_price if the dataset's per-shard royalty rate is
    available in the index record. Returns up to 20 results by default.
    """
    import urllib.parse as _url
    search = (arguments.get("search") or "").strip()
    max_price = arguments.get("max_price")
    limit = int(arguments.get("limit") or 20)
    semantic = bool(arguments.get("semantic"))
    _q = _url.quote(search)

    try:
        # sp1345 — semantic (embedding-similarity) browsing when requested; keyword otherwise.
        if semantic:
            result = await _call_node_api(
                "GET", f"/content/search/semantic?q={_q}&top_k={limit}")
            if result.get("semantic_available") is False:
                return ("Semantic search is unavailable on this node (no embedding function "
                        "wired). Retry without semantic=true for keyword search.")
        else:
            result = await _call_node_api("GET", f"/content/search?q={_q}&limit={limit}")
    except Exception:
        # Fallback: surface the index stats so the user knows the node is reachable
        try:
            stats = await _call_node_api("GET", "/content/index/stats")
            return (
                f"Dataset listing failed but content index is reachable.\n"
                f"  Total entries: {stats.get('total_entries', 0)}\n"
                f"  Total bytes: {stats.get('total_bytes', 0)}\n"
                f"  Tip: publish data with prsm_upload_dataset."
            )
        except Exception:
            return "PRSM node not running. Start with: prsm node start"

    records = result.get("results", []) or []
    if max_price is not None:
        try:
            mp = float(max_price)
            records = [
                r for r in records
                if r.get("royalty_rate") is None or float(r.get("royalty_rate") or 0) <= mp
            ]
        except (TypeError, ValueError):
            pass

    if not records:
        return (
            f"No datasets found"
            + (f" matching '{search}'" if search else "")
            + ".\n  Use prsm_upload_dataset to publish data, or prsm storage upload via CLI."
        )

    lines = [f"Datasets ({len(records)} of {result.get('count', len(records))}):"]
    for r in records[:limit]:
        size_mb = (r.get("size_bytes") or 0) / (1024 * 1024)
        royalty = r.get("royalty_rate")
        royalty_str = f" royalty={royalty}" if royalty is not None else ""
        lines.append(
            f"  • {r.get('cid', '?')[:16]}…  {r.get('filename', '(unnamed)')}"
            f"  {size_mb:.2f} MB  by {r.get('creator_id', '?')[:12]}{royalty_str}"
        )
    return "\n".join(lines)


async def handle_prsm_get_dataset(arguments: Dict[str, Any]) -> str:
    """Handle prsm_get_dataset — the one-call find -> fetch -> verify data-consumer action.

    Resolves a `cid` (given, or the top `query` match via keyword/semantic search), retrieves
    it, INTEGRITY-checks the bytes (sha256 == content_hash), surfaces creator + on-chain
    provenance attribution, and returns a text preview the model can reason over. Mirrors the
    SDK ``find_and_fetch`` but composed over the node HTTP API (the MCP server's transport).
    """
    import base64
    import hashlib
    import urllib.parse as _url

    query = (arguments.get("query") or "").strip()
    cid = (arguments.get("cid") or "").strip()
    semantic = bool(arguments.get("semantic"))
    max_preview = int(arguments.get("max_preview_chars") or 8000)
    if not cid and not query:
        return "Provide `query` (to find a dataset) or `cid` (to fetch a specific one)."

    matched: Dict[str, Any] = {}
    if not cid:
        _q = _url.quote(query)
        path = (f"/content/search/semantic?q={_q}&top_k=1" if semantic
                else f"/content/search?q={_q}&limit=1")
        try:
            sr = await _call_node_api("GET", path)
        except Exception as e:  # noqa: BLE001
            return f"Search failed: {e}. Is the node running? (prsm node start)"
        if semantic and sr.get("semantic_available") is False:
            return ("Semantic search unavailable on this node (no embedding function). Retry "
                    "with semantic=false.")
        rows = sr.get("results") or []
        if not rows:
            return (f"No dataset found matching '{query}'. Browse with prsm_list_datasets "
                    f"(try semantic=true for a broader concept match).")
        matched = rows[0]
        cid = matched.get("cid") or ""

    try:
        r = await _call_node_api("GET", f"/content/retrieve/{cid}?verify_hash=true")
    except Exception as e:  # noqa: BLE001
        return f"Retrieve failed for {cid}: {e}"
    if (r or {}).get("status") != "success":
        return (f"Content {cid} is not retrievable (status={r.get('status')}, "
                f"providers tried={r.get('providers_tried', 0)}).")

    raw = b""
    integrity = "unknown"
    content_hash = r.get("content_hash")
    try:
        raw = base64.b64decode(r.get("data") or "")
        if content_hash:
            integrity = ("VERIFIED" if hashlib.sha256(raw).hexdigest() == content_hash
                         else "FAILED")
    except Exception:  # noqa: BLE001
        integrity = "unknown"

    lines = [
        f"Dataset retrieved: {cid}",
        f"  filename: {r.get('filename') or matched.get('filename') or '?'}",
        f"  size_bytes: {r.get('size_bytes', len(raw))}",
        f"  integrity: {integrity}  (sha256 == content_hash)",
    ]
    creator = r.get("creator_eth_address") or matched.get("creator_eth_address")
    prov = r.get("provenance_hash") or matched.get("provenance_hash")
    if creator:
        lines.append(f"  creator: {creator}  (verifiable provenance)")
    if prov:
        lines.append(f"  provenance_hash: {prov}")
    try:
        text = raw.decode("utf-8")
        preview = text[:max_preview]
        lines.append(f"\n--- content preview ({len(preview)}/{len(text)} chars) ---\n{preview}")
        if len(text) > len(preview):
            lines.append("… (truncated)")
    except UnicodeDecodeError:
        lines.append(f"\n[binary content, {len(raw)} bytes — not previewable as text]")
    return "\n".join(lines)


async def handle_prsm_node_status(arguments: Dict[str, Any]) -> str:
    """Handle prsm_node_status tool call."""
    try:
        result = await _call_node_api("GET", "/rings/status")
        rings = result.get("rings", [])
        lines = [f"PRSM Node -- {result.get('rings_initialized', 0)}/10 Rings Active\n"]
        for r in rings:
            status = "[ok]" if r.get("initialized") else "[--]"
            lines.append(f"  {status} Ring {r['ring']}: {r['name']}")

        pricing = result.get("pricing", {})
        if pricing:
            lines.append(f"\n  Spot Multiplier: {pricing.get('spot_multiplier', '1.0')}x")
            lines.append(f"  Utilization: {pricing.get('utilization', 0):.0%}")

        forge = result.get("forge", {})
        if forge:
            lines.append(f"  Training Traces: {forge.get('traces_collected', 0)}")

        return "\n".join(lines)
    except Exception as e:
        return f"Cannot reach PRSM node: {str(e)}\nStart with: prsm node start"


async def handle_prsm_section7_readiness(arguments: Dict[str, Any]) -> str:
    """Sprint 587 — MCP wrapper around sprint-585 §7-readiness check.

    Runs the same three probes (anchor / stake-bond / rpc) in-process
    + returns a multiline summary suitable for AI-triage agents.
    Tolerates env-unset (most common dev state).
    """
    import os as _os
    lines = ["§7 production-readiness:"]

    # Anchor
    anchor_addr = (
        _os.environ.get("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", "") or ""
    ).strip()
    rpc_url = _os.environ.get(
        "PRSM_BASE_RPC_URL", "https://mainnet.base.org",
    )
    if not anchor_addr:
        anchor_outcome, anchor_err = "unset", None
    else:
        try:
            from prsm.security.publisher_key_anchor.client import (
                PublisherKeyAnchorClient,
            )
            PublisherKeyAnchorClient(
                contract_address=anchor_addr, rpc_url=rpc_url,
            )
            anchor_outcome, anchor_err = "ok", None
        except Exception as exc:  # noqa: BLE001
            anchor_outcome = "construction_failed"
            anchor_err = f"{type(exc).__name__}: {exc}"
    lines.append(
        f"  anchor:     {anchor_outcome}"
        + (f" ({anchor_err})" if anchor_err else "")
    )

    # Stake-bond
    stake_addr = (
        _os.environ.get("PRSM_STAKE_BOND_ADDRESS", "") or ""
    ).strip()
    if not stake_addr:
        stake_outcome, stake_err = "unset", None
    else:
        try:
            from prsm.economy.web3.stake_manager import StakeManagerClient
            StakeManagerClient(
                contract_address=stake_addr, rpc_url=rpc_url,
            )
            stake_outcome, stake_err = "ok", None
        except Exception as exc:  # noqa: BLE001
            stake_outcome = "construction_failed"
            stake_err = f"{type(exc).__name__}: {exc}"
    lines.append(
        f"  stake_bond: {stake_outcome}"
        + (f" ({stake_err})" if stake_err else "")
    )

    # RPC
    import httpx as _httpx
    try:
        resp = _httpx.post(
            rpc_url,
            json={
                "jsonrpc": "2.0", "method": "eth_chainId",
                "params": [], "id": 1,
            },
            timeout=10.0,
        )
        if resp.status_code != 200:
            rpc_outcome = "error"
            rpc_err = f"HTTP {resp.status_code}"
        else:
            body = resp.json()
            chain_id = body.get("result")
            if chain_id is None:
                rpc_outcome, rpc_err = "error", f"no result: {body!r}"[:100]
            else:
                rpc_outcome, rpc_err = "ok", None
    except _httpx.HTTPError as exc:
        rpc_outcome = "unreachable"
        rpc_err = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        rpc_outcome, rpc_err = "error", f"{type(exc).__name__}: {exc}"
    lines.append(
        f"  rpc:        {rpc_outcome}"
        + (f" ({rpc_err})" if rpc_err else "")
    )

    overall = "ready" if all(
        o == "ok" for o in (anchor_outcome, stake_outcome, rpc_outcome)
    ) else "not_ready"
    lines.append(f"\noverall: {overall}")
    if overall == "not_ready":
        lines.append(
            "Fix failing component(s) before flipping "
            "PRSM_PARALLAX_TRUST_STACK_KIND=production."
        )
    return "\n".join(lines)


async def handle_prsm_hardware_benchmark(arguments: Dict[str, Any]) -> str:
    """Handle prsm_hardware_benchmark tool call."""
    try:
        from prsm.compute.wasm import HardwareProfiler
        from prsm.compute.tee.platform_detect import get_tee_summary

        profiler = HardwareProfiler()
        profile = profiler.detect()
        tee = get_tee_summary()

        return (
            f"PRSM Hardware Benchmark\n"
            f"  CPU: {profile.cpu_cores} cores @ {profile.cpu_freq_mhz:.0f} MHz\n"
            f"  GPU: {profile.gpu_name or 'None detected'}\n"
            f"  VRAM: {profile.gpu_vram_gb:.1f} GB\n"
            f"  TFLOPS: {profile.tflops_fp32:.2f} FP32\n"
            f"  Compute Tier: {profile.compute_tier.value.upper()}\n"
            f"  Thermal: {profile.thermal_class.value}\n"
            f"  TEE: {tee['type']} (hardware: {tee['hardware_backed']})\n"
            f"  RAM: {profile.ram_total_gb:.1f} GB total, {profile.ram_available_gb:.1f} GB available"
        )
    except Exception as e:
        return f"Benchmark failed: {str(e)}"


async def handle_prsm_create_agent(arguments: Dict[str, Any]) -> str:
    """Handle prsm_create_agent — build an instruction manifest."""
    query = arguments.get("query", "")
    instructions_raw = arguments.get("instructions", [])
    target_shards = arguments.get("target_shards", [])
    hardware_tier = arguments.get("hardware_tier", "t1")
    budget = arguments.get("budget_ftns", 5.0)

    if budget <= 0:
        return "FTNS budget required (minimum 0.01). Use prsm_quote to estimate costs."

    try:
        from prsm.compute.agents.instruction_set import (
            AgentOp, AgentInstruction, InstructionManifest,
        )

        instructions = []
        for inst in instructions_raw:
            op_str = inst.get("op", "count")
            try:
                op = AgentOp(op_str)
            except ValueError:
                return f"Unknown operation: {op_str}. Available: {[o.value for o in AgentOp]}"

            instructions.append(AgentInstruction(
                op=op,
                field=inst.get("field", ""),
                value=inst.get("value"),
                params=inst.get("params", {}),
            ))

        if not instructions:
            return "At least one instruction is required."

        manifest = InstructionManifest(
            query=query,
            instructions=instructions,
        )

        manifest_json = manifest.to_json()

        lines = [
            f"Agent Manifest Created",
            f"  Query: {query}",
            f"  Operations: {len(instructions)}",
        ]
        for i, inst in enumerate(instructions):
            field_str = f" on '{inst.field}'" if inst.field else ""
            value_str = f" = {inst.value}" if inst.value is not None else ""
            lines.append(f"    {i+1}. {inst.op.value}{field_str}{value_str}")

        lines.append(f"  Target shards: {target_shards or '(auto-discover)'}")
        lines.append(f"  Hardware tier: {hardware_tier}")
        lines.append(f"")
        lines.append(f"  Manifest JSON (pass to prsm_dispatch_agent):")
        lines.append(f"  {manifest_json}")

        # Cost reconciliation footer (Phase 3.x.1 Task 7).
        # prsm_create_agent does not itself consume FTNS — it builds the
        # manifest. Spend happens at prsm_dispatch_agent time, so the footer
        # carries the planned budget rather than a settled cost. There is no
        # job_id yet (one will be allocated at dispatch).
        footer = _format_cost_footer(
            job_id="(none — assigned at dispatch)",
            budget_ftns=budget,
            extra_fields={
                "Hardware tier": hardware_tier,
                "Operations": str(len(instructions)),
            },
            note="ℹ️  Manifest only — no FTNS consumed yet. "
                 "Cost will be charged when dispatched via prsm_dispatch_agent.",
        )
        return "\n".join(lines) + "\n" + footer

    except Exception as e:
        return f"Agent creation failed: {str(e)}"


async def handle_prsm_dispatch_agent(arguments: Dict[str, Any]) -> str:
    """Handle prsm_dispatch_agent — dispatch an instruction manifest.

    Flow (post-B8 unhide pass 2):
      1. Parse the user-supplied InstructionManifest JSON locally
         (early validation — malformed manifests rejected without
         spending FTNS or hitting the node).
      2. Forward ``manifest.query`` to /compute/forge with the
         requested budget.
      3. /compute/forge duck-type-dispatches on
         ``node.agent_forge.dispatch_query`` (QueryOrchestrator) →
         decomposes the query server-side → finds shards →
         fans out → aggregates → returns.

    Honest scope: the QueryOrchestrator currently RE-DECOMPOSES
    the natural-language query rather than consuming the user's
    pre-built manifest verbatim. The local manifest serves as a
    structured precondition (validates op set, budget hint) but
    its instruction list is not executed verbatim. A future
    sprint may wire end-to-end manifest pass-through.
    """
    instructions_json = arguments.get("instructions_json", "")
    budget = arguments.get("budget_ftns", 5.0)

    if budget <= 0:
        return "FTNS budget required (minimum 0.01)."

    if not instructions_json:
        return "Missing instructions_json. Use prsm_create_agent first to build a manifest."

    try:
        from prsm.compute.agents.instruction_set import InstructionManifest

        manifest = InstructionManifest.from_json(instructions_json)

        # Try to dispatch via the node API
        try:
            result = await _call_node_api("POST", "/compute/forge", {
                "query": manifest.query,
                "budget_ftns": budget,
            })
            route = result.get("route", "unknown")
            response = result.get("response", str(result))
            job_id = result.get("job_id", "unknown")

            footer = _format_cost_footer(
                job_id=job_id,
                cost_ftns=result.get("cost_ftns"),
                budget_ftns=budget,
                extra_fields={
                    "Route": route,
                    "Operations": str(len(manifest.instructions)),
                },
            )
            return (
                f"Agent Dispatched\n"
                f"  Query: {manifest.query}\n\n"
                f"Result:\n{response}\n"
                f"{footer}"
            )
        except Exception as e:
            return (
                f"Agent manifest valid ({len(manifest.instructions)} operations) "
                f"but dispatch failed: {str(e)}\n"
                f"Is your PRSM node running? (prsm node start)"
            )

    except Exception as e:
        return f"Invalid instruction manifest: {str(e)}"


async def handle_prsm_agent_status(arguments: Dict[str, Any]) -> str:
    job_id = arguments.get("job_id", "")
    try:
        result = await _call_node_api("GET", f"/compute/status/{job_id}")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Could not check agent status: {e}. Is your PRSM node running?"


async def handle_prsm_search_shards(arguments: Dict[str, Any]) -> str:
    """Handle prsm_search_shards by querying the node's content index.

    Sprint 289 forwards optional creator-tier filters:
      min_tier    — "low" | "medium" | "high"
      exclude_new — bool (hide cold-start creators)
    """
    query = (arguments.get("query") or "").strip()
    top_k = int(arguments.get("top_k") or 5)

    if not query:
        return "Search requires a 'query' string."

    # Sprint 289 — defense-in-depth: validate min_tier client-
    # side so we don't waste an RPC roundtrip on bogus input.
    min_tier_raw = arguments.get("min_tier")
    if min_tier_raw is not None:
        min_tier_norm = str(min_tier_raw).strip().lower()
        if min_tier_norm not in {"low", "medium", "high"}:
            return (
                f"min_tier must be one of "
                f"['low', 'medium', 'high'], "
                f"got {min_tier_raw!r}."
            )
    else:
        min_tier_norm = None
    exclude_new = bool(arguments.get("exclude_new", False))

    path = f"/content/search?q={query}&limit={top_k}"
    if min_tier_norm:
        path += f"&min_tier={min_tier_norm}"
    if exclude_new:
        path += "&exclude_new=true"

    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return f"Shard search requires a running PRSM node ({e}). Start with: prsm node start"

    records = result.get("results", []) or []
    if not records:
        return (
            f"No shards found for: '{query}'\n"
            f"  Use prsm_upload_dataset to publish data, or check the index with "
            f"prsm_list_datasets."
        )

    filter_note = ""
    if min_tier_norm or exclude_new:
        bits = []
        if min_tier_norm:
            bits.append(f"min_tier={min_tier_norm}")
        if exclude_new:
            bits.append("exclude_new")
        filter_note = f" [filtered: {', '.join(bits)}]"
    lines = [
        f"Shard search results for '{query}' "
        f"(top {len(records)}){filter_note}:"
    ]
    for r in records:
        providers = r.get("providers") or []
        size_mb = (r.get("size_bytes") or 0) / (1024 * 1024)
        tier = r.get("creator_tier") or "?"
        lines.append(
            f"  • CID {r.get('cid', '?')}\n"
            f"      file={r.get('filename', '(unnamed)')}  size={size_mb:.2f} MB  "
            f"providers={len(providers)}  creator={r.get('creator_id', '?')[:12]}  "
            f"tier={tier}"
        )
    return "\n".join(lines)


async def handle_prsm_upload_dataset(arguments: Dict[str, Any]) -> str:
    dataset_id = arguments.get("dataset_id", "")
    title = arguments.get("title", "")
    description = arguments.get("description", "")
    shard_count = arguments.get("shard_count", 4)
    base_fee = arguments.get("base_access_fee", 1.0)
    per_shard = arguments.get("per_shard_fee", 0.1)
    require_stake = arguments.get("require_stake", 0)

    try:
        result = await _call_node_api("POST", "/content/upload/shard", {
            "dataset_id": dataset_id,
            "title": title,
            "description": description,
            "shard_count": shard_count,
            "base_access_fee": base_fee,
            "per_shard_fee": per_shard,
        })
        # sp1347 — a 4xx (413 too big, 503 no publisher, 422) returns a dict, not an exception;
        # without this guard the success message rendered with default fields = FAKE SUCCESS.
        _e = _api_error(result)
        if _e:
            return f"Dataset upload failed: {_e}"
        return (
            f"Dataset Published\n"
            f"  ID: {dataset_id}\n"
            f"  Title: {title}\n"
            f"  Shards: {result.get('shard_count', shard_count)}\n"
            f"  Base Fee: {base_fee} FTNS/query\n"
            f"  Per-Shard Fee: {per_shard} FTNS\n"
            f"  Revenue: 80% to you, 15% compute, 5% treasury"
        )
    except Exception as e:
        return f"Dataset upload failed: {e}. Is your PRSM node running?"


async def handle_prsm_yield_estimate(arguments: Dict[str, Any]) -> str:
    hours = arguments.get("hours_per_day", 8)
    stake = arguments.get("stake_amount", 0)
    try:
        from prsm.compute.wasm import HardwareProfiler
        from prsm.economy.pricing import PricingEngine, ProsumerTier
        profiler = HardwareProfiler()
        profile = profiler.detect()
        tier = ProsumerTier.from_stake(int(stake))
        engine = PricingEngine()
        est = engine.yield_estimate(
            hardware_tier=profile.compute_tier.value,
            tflops=profile.tflops_fp32,
            hours_per_day=hours,
            prosumer_tier=tier,
        )
        return (
            f"Yield Estimate\n"
            f"  Hardware: {profile.compute_tier.value.upper()} ({profile.tflops_fp32:.1f} TFLOPS)\n"
            f"  Stake: {stake:.0f} FTNS ({tier.name})\n"
            f"  Yield Boost: {est['yield_boost']}x\n"
            f"  Daily: {float(est['daily_ftns']):.2f} FTNS\n"
            f"  Monthly: {float(est['monthly_ftns']):.2f} FTNS"
        )
    except Exception as e:
        return f"Yield estimate failed: {e}"


async def handle_prsm_stake(arguments: Dict[str, Any]) -> str:
    """Handle prsm_stake — actually stake against the running node.

    If 'execute' is true, calls POST /staking/stake on the node. Otherwise
    returns a tier preview (the legacy info-only behavior). The execute flag
    defaults to False so naive callers can't accidentally lock tokens.
    """
    amount = arguments.get("amount", 0)
    execute = bool(arguments.get("execute", False))
    stake_type = arguments.get("stake_type", "general")
    lock_period_days = arguments.get("lock_period_days")

    # sp906 — staking confers UTILITY benefits (lock-based service
    # discounts + priority access), NOT token yield (sp904). Preview the
    # benefit tier the chosen lock unlocks.
    try:
        from prsm.economy.tokenomics.staking_manager import StakingConfig
        benefits = StakingConfig().benefits_for_lock_days(lock_period_days)
    except Exception as e:
        return f"Staking info failed: {e}"

    if not execute:
        if benefits.is_active:
            benefit_line = (
                f"  Lock: {benefits.tier_label} → "
                f"{benefits.discount_fraction * 100:.0f}% network-fee discount, "
                f"+{benefits.priority_boost * 100:.0f}% dispatch priority\n"
            )
        else:
            benefit_line = (
                "  Lock: none → no utility benefit. Pass "
                "lock_period_days=30/90/365 for service discounts + priority "
                "(staking pays no token yield).\n"
            )
        return (
            f"Staking Preview (no transaction submitted)\n"
            f"  Amount: {amount} FTNS\n"
            f"{benefit_line}"
            f"  To actually stake, call prsm_stake again with execute=true."
        )

    if int(amount) <= 0:
        return "Cannot stake 0 FTNS. Provide a positive 'amount'."

    try:
        result = await _call_node_api(
            "POST",
            "/staking/stake",
            {
                "amount": float(amount),
                "stake_type": stake_type,
                "metadata": {},
                "lock_period_days": lock_period_days,
            },
        )
    except Exception as e:
        return (
            f"Stake submission failed: {e}\n"
            f"  Is your node running? Tip: prsm node start"
        )

    return (
        f"Stake Submitted\n"
        f"  Stake ID: {result.get('stake_id', '?')}\n"
        f"  Amount: {result.get('amount', amount)} FTNS\n"
        f"  Type: {result.get('stake_type', stake_type)}\n"
        f"  Status: {result.get('status', 'unknown')}\n"
        f"  Lock: {benefits.tier_label}"
        + (
            f" ({benefits.discount_fraction * 100:.0f}% fee discount, "
            f"+{benefits.priority_boost * 100:.0f}% priority)\n"
            if benefits.is_active
            else " (no utility benefit — unlocked stake)\n"
        )
        + f"  Staked at: {result.get('staked_at', '')}"
    )


async def handle_prsm_revenue_split(arguments: Dict[str, Any]) -> str:
    total = arguments.get("total_payment", 0)
    has_data = arguments.get("has_data_owner", True)
    providers = arguments.get("compute_providers", 1)
    # sp906 — optional staking network-fee discount. Pass a fraction
    # directly, or a staker_user_id to look up the live tier on the node.
    fee_discount = float(arguments.get("network_fee_discount_fraction", 0.0) or 0.0)
    staker_user_id = arguments.get("staker_user_id")
    try:
        from decimal import Decimal
        from prsm.economy.pricing.revenue_split import RevenueSplitEngine
        if staker_user_id and fee_discount == 0.0:
            try:
                b = await _call_node_api(
                    "GET", f"/staking/benefits/{staker_user_id}",
                )
                fee_discount = float(b.get("discount_fraction", 0.0) or 0.0)
            except Exception:
                fee_discount = 0.0
        engine = RevenueSplitEngine()
        provider_dict = {f"provider-{i}": 100.0/max(providers,1) for i in range(max(providers,1))}
        split = engine.calculate_split(
            total_payment=Decimal(str(total)),
            data_owner_id="data-owner" if has_data else "",
            compute_providers=provider_dict,
            network_fee_discount_fraction=fee_discount,
        )
        lines = [f"Revenue Split for {total} FTNS"]
        if has_data:
            lines.append(f"  Data Owner: {split.data_owner_amount} FTNS (80%)")
        lines.append(f"  Compute ({providers} providers): {sum(split.compute_amounts.values())} FTNS")
        lines.append(f"  Treasury: {split.treasury_amount} FTNS (5%)")
        if split.fee_discount_amount > 0:
            lines.append(
                f"  Staking fee discount: -{split.fee_discount_amount} FTNS "
                f"({fee_discount * 100:.0f}% off the network fee)"
            )
            lines.append(f"  Payer pays: {split.effective_total_paid} FTNS")
        return "\n".join(lines)
    except Exception as e:
        return f"Split calculation failed: {e}"


async def handle_prsm_settlement_stats(arguments: Dict[str, Any]) -> str:
    try:
        result = await _call_node_api("GET", "/settlement/stats")
        return json.dumps(result, indent=2)
    except Exception:
        return "Settlement stats require a running PRSM node. Start with: prsm node start"


async def handle_prsm_privacy_status(arguments: Dict[str, Any]) -> str:
    """Handle prsm_privacy_status — fetches the live DP budget audit report."""
    try:
        report = await _call_node_api("GET", "/privacy/budget")
    except Exception as e:
        return (
            f"Privacy Budget: Cannot reach node ({e}).\n"
            f"  Start a node with: prsm node start\n"
            f"  The privacy budget tracks cumulative differential privacy (ε) "
            f"spending across forge queries with privacy_level != 'none'."
        )

    max_eps = report.get("max_epsilon", 0)
    spent = report.get("total_spent", 0)
    remaining = report.get("remaining", 0)
    num_ops = report.get("num_operations", 0)
    spends = report.get("spends", []) or []

    pct = (spent / max_eps * 100.0) if max_eps else 0.0
    lines = [
        f"Differential Privacy Budget",
        f"  Total budget: {max_eps:.1f} ε",
        f"  Spent:        {spent:.3f} ε  ({pct:.1f}%)",
        f"  Remaining:    {remaining:.3f} ε",
        f"  Operations:   {num_ops}",
    ]
    if spends:
        lines.append("  Recent spends:")
        for s in spends[-5:]:
            # Sprint 263 — surface job_id alongside operation +
            # model_id. Pre-fix the renderer showed model_id but
            # the field structurally contained the job_id due to
            # an api.py arg mis-binding (now fixed).
            jid = s.get("job_id") or ""
            mid = s.get("model_id") or ""
            extras = []
            if jid:
                extras.append(f"job={jid[:14]}")
            if mid:
                extras.append(f"model={mid}")
            extras_str = "  " + "  ".join(extras) if extras else ""
            lines.append(
                f"    - {s.get('operation', '?')}  "
                f"ε={s.get('epsilon', 0):.3f}{extras_str}"
            )
    return "\n".join(lines)


async def handle_prsm_training_status(arguments: Dict[str, Any]) -> str:
    try:
        result = await _call_node_api("GET", "/rings/status")
        forge = result.get("forge", {})
        traces = forge.get("traces_collected", 0)

        # Try to evaluate quality if we have traces
        from prsm.compute.nwtn.training.evaluation import TrainingEvaluator
        evaluator = TrainingEvaluator(min_traces=100)

        return (
            f"NWTN Training Pipeline Status\n"
            f"  Traces Collected: {traces}\n"
            f"  Minimum for Fine-tune: 100\n"
            f"  Ready: {'Yes' if traces >= 100 else 'No — need more queries'}\n"
            f"  Tip: Run diverse queries via prsm_analyze to build the training corpus."
        )
    except Exception:
        return (
            "NWTN Training Pipeline\n"
            "  The training pipeline collects AgentTrace data from every forge query.\n"
            "  Once enough traces are collected (100+), the NWTN model can be fine-tuned\n"
            "  for better task decomposition and WASM agent generation."
        )


def _api_error(result: Any) -> Optional[str]:
    """sp1347 — extract a node API error message from a response dict, or None if it's not an
    error. ``_call_node_api`` returns ``resp.json()`` REGARDLESS of HTTP status, so a 4xx
    (FastAPI HTTPException → ``{"detail": ...}``, or a handler's ``{"error": ...}`` /
    ``{"success": False}``) lands as a plain dict. Handlers that skip this and read success-path
    keys silently render fake-success/blank output on an error (the bug class sp1346 fixed in
    prsm_inference). Prefer explicit error/detail; an explicit ``success == False`` is an error."""
    if not isinstance(result, dict):
        return "unexpected non-dict response from the node"
    err = result.get("error") or result.get("detail")
    if err:
        return str(err)
    if result.get("success") is False:
        return str(result.get("message") or "request failed")
    return None


def _funding_hint(err_text: Any) -> str:
    """sp1346 — when an inference error looks like insufficient FTNS, return an actionable
    next-step that points the model at the funding tools; otherwise ''. Makes the
    faucet -> inference flow self-guiding through MCP: without this, a model that hits
    "insufficient FTNS balance to lock escrow" gets a dead-end error and has no idea it can
    prsm_faucet its way to funded (the compute-flagship analog of sp1345's data actionables)."""
    t = str(err_text or "").lower()
    if ("insufficient" in t or "402" in t or "not enough" in t
            or ("escrow" in t and "balance" in t) or "no ftns" in t):
        return ("\n\nOut of FTNS? Get testnet FTNS with prsm_faucet, confirm it landed with "
                "prsm_local_balance, then retry. Estimate the cost first with "
                "prsm_inference_quote.")
    return ""


async def handle_prsm_inference(
    arguments: Dict[str, Any],
    *,
    emit_progress: Optional[ProgressEmitter] = None,
) -> str:
    """Handle prsm_inference — TEE-attested model inference with verifiable receipt.

    Builds an InferenceRequest, calls the node API, and formats the
    response with cost-reconciliation footer.

    Phase 3.x.8.1 Task 4 — routing: streaming-capable MCP clients
    (those that supplied a ``progressToken``, surfaced as a non-None
    ``emit_progress`` callback) hit the SSE
    ``POST /compute/inference/stream`` endpoint and receive
    incremental token output as MCP progress events. Non-streaming
    clients hit the existing unary ``POST /compute/inference``
    endpoint and see only the final formatted response.

    Both paths produce the same final TextContent shape (output +
    cost-reconciliation footer with model / privacy tier / content
    tier / TEE backend / duration / settler / signature note). The
    only caller-observable difference is the per-token progress
    stream on the streaming path.
    """
    prompt = arguments.get("prompt", "")
    # sp1393 — default to the node's real local model (was "mock-llama-3-8b", which no node serves →
    # prsm_inference dead-ended with "Unknown model_id" out of the box, the same class as sp1386).
    model_id = arguments.get("model_id", "distilgpt2")
    budget = arguments.get("budget_ftns", 1.0)
    # sp1393 — default 'none' (usable output on any node). 'standard' required a hardware-TEE node,
    # so the flagship prsm_inference tool dead-ended with a tier-gate refusal on a software node.
    privacy_tier = arguments.get("privacy_tier", "none")
    content_tier = arguments.get("content_tier", "A")
    max_tokens = arguments.get("max_tokens")
    temperature = arguments.get("temperature")

    if not prompt:
        return "Missing required 'prompt' argument."

    # Enforce minimum budget — same pattern as handle_prsm_analyze
    if budget <= 0:
        return (
            "PRSM inference requires an FTNS budget. "
            "Set budget_ftns to at least 0.01 FTNS.\n\n"
            "Tip: Use prsm_quote first to estimate the cost for your model + prompt."
        )
    if budget < MINIMUM_BUDGET_FTNS:
        return (
            f"Budget {budget} FTNS is below minimum ({MINIMUM_BUDGET_FTNS} FTNS). "
            f"Use prsm_quote to estimate the required budget."
        )

    request_payload: Dict[str, Any] = {
        "prompt": prompt,
        "model_id": model_id,
        "budget_ftns": budget,
        "privacy_tier": privacy_tier,
        "content_tier": content_tier,
    }
    if max_tokens is not None:
        request_payload["max_tokens"] = int(max_tokens)
    if temperature is not None:
        request_payload["temperature"] = float(temperature)

    # Branch on streaming-capable client. The streaming path emits
    # one progress event per StreamToken (Phase 3.x.8.1 Task 3
    # _call_node_api_streaming forwards). The unary path keeps the
    # original Phase 3.x.1 Task 8 4-stage progress milestones.
    if emit_progress is not None:
        try:
            result = await _call_node_api_streaming(
                "/compute/inference/stream",
                request_payload,
                emit_progress,
            )
        except InferenceError as e:
            # Server-side rejection (budget, model, tier, etc.) —
            # surface the structured error directly.
            return f"Inference rejected: {e.message}" + _funding_hint(e.message)
        except Exception as e:  # noqa: BLE001
            return (
                f"PRSM streaming inference failed: {e}.\n"
                f"Possible causes:\n"
                f"  • PRSM node not running (start with: prsm node start)\n"
                f"  • /compute/inference/stream endpoint not deployed "
                f"(Phase 3.x.8.1 Task 2 — verify with `curl -N` against "
                f"the node)\n"
                f"  • Network connectivity issue between MCP server and "
                f"node API"
            )
    else:
        if emit_progress:  # pragma: no cover — defensive; emit_progress is None here
            await emit_progress(
                f"Building inference request for model {model_id}...",
                1.0, 4.0,
            )
        try:
            result = await _call_node_api(
                "POST", "/compute/inference", request_payload,
            )
        except Exception as e:  # noqa: BLE001
            return (
                f"PRSM inference failed: {e}.\n"
                f"Possible causes:\n"
                f"  • PRSM node not running (start with: prsm node start)\n"
                f"  • /compute/inference endpoint not yet deployed (Phase 3.x.1 Task 5 pending; "
                f"see docs/2026-04-26-phase3.x.1-mcp-server-completion-design-plan.md)\n"
                f"  • Network connectivity issue between MCP server and node API"
            )

    # Surface API-level errors with helpful context (only reachable
    # on the unary path — the streaming path raises InferenceError
    # for these cases, handled above).
    # sp1346 — a FastAPI HTTPException (e.g. 402 insufficient funds) lands as {"detail": ...},
    # NOT {"error": ...}; _call_node_api returns the body regardless of status. Pre-fix this
    # surfaced "Inference failed: None" (losing the real reason + skipping the funding hint).
    if isinstance(result, dict) and (result.get("error") or result.get("detail")):
        _err = result.get("error") or result.get("detail")
        return f"Inference rejected: {_err}" + _funding_hint(_err)
    if not isinstance(result, dict) or not result.get("success"):
        _err = (result.get("error") or result.get("detail") or "unknown error"
                if isinstance(result, dict) else "unknown error")
        return f"Inference failed: {_err}" + _funding_hint(_err)

    # Format successful response with cost reconciliation footer
    # (Phase 3.x.1 Task 7 — uses shared _format_cost_footer helper).
    output = result.get("output", "")
    receipt = result.get("receipt") or {}

    extra: Dict[str, str] = {
        "Model": str(receipt.get("model_id", model_id)),
        "Privacy tier": f"{receipt.get('privacy_tier', privacy_tier)} (ε={receipt.get('epsilon_spent', '?')})",
        "Content tier": str(receipt.get("content_tier", content_tier)),
        "TEE backend": str(receipt.get("tee_type", "unknown")),
        "Duration": f"{receipt.get('duration_seconds', '?')}s",
        "Settler": str(receipt.get("settler_node_id", "unknown")),
    }
    note = None
    if receipt.get("settler_signature"):
        note = (
            "Receipt is signed. Verify with: "
            "prsm.compute.inference.verify_receipt(receipt, "
            "public_key_b64=<settler_pubkey>)"
        )

    footer = _format_cost_footer(
        job_id=str(receipt.get("job_id", result.get("job_id", "unknown"))),
        cost_ftns=receipt.get("cost_ftns"),
        budget_ftns=budget,
        extra_fields=extra,
        note=note,
    )

    return (
        f"PRSM Inference Result\n"
        f"=====================\n\n"
        f"{output}\n"
        f"{footer}"
    )


def _format_cost_footer(
    *,
    job_id: str,
    cost_ftns: Optional[Any] = None,
    budget_ftns: Optional[Any] = None,
    extra_fields: Optional[Dict[str, str]] = None,
    note: Optional[str] = None,
) -> str:
    """Build a uniform cost-reconciliation footer for FTNS-consuming tool responses.

    Phase 3.x.1 Task 7 — extracts the pattern from prsm_inference (Task 6) into a
    shared helper applied to all four FTNS-consuming tools (prsm_analyze,
    prsm_inference, prsm_create_agent, prsm_dispatch_agent).

    Args:
        job_id: required — the FTNS job identifier the LLM should pass to
            prsm_billing_status if it wants to reconcile later.
        cost_ftns: actual settled cost (post-execution). Falls back to "?" when
            the underlying API didn't surface it.
        budget_ftns: prepaid budget from the call. Used when cost_ftns isn't
            available, to communicate the upper-bound spend.
        extra_fields: per-tool fields (route, model, privacy tier, etc.).
            Inserted between the standard rows.
        note: optional trailing line below the rule (e.g. "Manifest only — no
            FTNS consumed yet" for prsm_create_agent).

    Returns the footer block as a single string ready to append to the response.
    """
    rule = "—" * 60
    lines = ["", rule, f"Job ID:           {job_id or 'unknown'}"]

    if extra_fields:
        for label, value in extra_fields.items():
            lines.append(f"{(label + ':'):<18}{value}")

    if cost_ftns is not None:
        lines.append(f"Cost:             {cost_ftns} FTNS")
    elif budget_ftns is not None:
        lines.append(f"Budget reserved:  {budget_ftns} FTNS")

    lines.append(f"Reconcile via:    prsm_billing_status(job_id=\"{job_id}\")")
    lines.append(rule)

    if note:
        lines.append(note)
    return "\n".join(lines)


async def handle_prsm_billing_status(arguments: Dict[str, Any]) -> str:
    """Handle prsm_billing_status — query escrow state for a prior job_id.

    Phase 3.x.1 Task 7. Calls /billing/{job_id} on the node API; formats
    the response as a structured billing report.
    """
    job_id = (arguments.get("job_id") or "").strip()
    if not job_id:
        return "Missing required 'job_id' argument."

    try:
        result = await _call_node_api("GET", f"/billing/{job_id}")
    except Exception as e:
        return (
            f"Failed to query billing for job_id={job_id}: {e}\n"
            f"  • Is your PRSM node running? (prsm node start)"
        )

    if isinstance(result, dict) and result.get("detail"):
        # FastAPI surfaces 404 as {"detail": "..."} — pass through as-is for
        # the LLM to read and explain to the user.
        return f"Billing query for {job_id}: {result['detail']}"

    if not isinstance(result, dict):
        return f"Unexpected billing response shape for {job_id}: {result!r}"

    lines = [
        f"PRSM Billing Status — {result.get('job_id', job_id)}",
        "=" * 60,
        f"Escrow ID:        {result.get('escrow_id', 'unknown')}",
        f"Status:           {result.get('status', 'unknown')}",
        f"Amount locked:    {result.get('amount_ftns', '?')} FTNS",
        f"Requester:        {result.get('requester_id', 'unknown')}",
    ]
    if result.get("provider_winner"):
        lines.append(f"Provider:         {result['provider_winner']}")
    if result.get("tx_lock"):
        lines.append(f"Lock tx:          {result['tx_lock']}")
    if result.get("tx_release"):
        lines.append(f"Release tx:       {result['tx_release']}")
    if result.get("created_at"):
        lines.append(f"Created at:       {result['created_at']}")
    if result.get("completed_at"):
        lines.append(f"Completed at:     {result['completed_at']}")
    return "\n".join(lines)


async def handle_prsm_balance_check(arguments: Dict[str, Any]) -> str:
    """Handle prsm_balance_check tool call.

    V1 scope: GET /balance/onchain via the node API; format the
    response as user-facing text. Closes the explicit Vision §13
    Phase 5 stand-in.
    """
    address = arguments.get("address")
    path = "/balance/onchain"
    if address:
        path = f"{path}?address={address}"

    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    # 503 fallback path — endpoint returned a `detail` envelope rather
    # than a balance. Common cause: ftns_ledger not initialized
    # because PRSM_ONCHAIN_FTNS or FTNS_TOKEN_ADDRESS is unset.
    if "balance_ftns" not in result:
        detail = result.get("detail", "unknown error")
        return (
            f"On-chain FTNS not configured on this node.\n"
            f"  Detail: {detail}\n"
            f"  Set PRSM_ONCHAIN_FTNS=1 + FTNS_TOKEN_ADDRESS to enable."
        )

    addr = result["address"]
    balance_ftns = result["balance_ftns"]
    usd_rate = result["usd_rate"]
    usd_equivalent = result["usd_equivalent"]

    # Display: short address (first 10 chars) + full FTNS amount +
    # USD equivalent (with explicit rate so users see the conversion
    # they're trusting).
    short_addr = (
        addr[:10] + "…" + addr[-4:] if len(addr) > 14 else addr
    )

    # Aggregate-source breakdown (audit-prep §7.23 honest-scope
    # closure): when v2 fields are present in the response, render
    # the multi-source breakdown. Backwards-compat: if the endpoint
    # returns a v1-only response (no `total_ftns`), fall through to
    # the legacy single-line format.
    if "total_ftns" in result:
        claimable_ftns = result["claimable_royalties_ftns"]
        escrowed_ftns = result["escrowed_ftns"]
        total_ftns = result["total_ftns"]
        total_usd = result["total_usd_equivalent"]
        sources = result.get("sources", {})

        # Show source breakdown only when at least one extra source
        # has non-zero balance OR is wired (available=true). When
        # all extras are zero+unwired, the breakdown is just noise;
        # render legacy format.
        has_extras = (
            claimable_ftns > 0 or escrowed_ftns > 0
            or sources.get("claimable_royalties", {}).get("available", False)
            or sources.get("escrowed", {}).get("available", False)
        )
        if has_extras:
            claimable_avail = sources.get(
                "claimable_royalties", {},
            ).get("available", False)
            escrowed_avail = sources.get("escrowed", {}).get("available", False)
            claimable_marker = "" if claimable_avail else " (unavailable)"
            escrowed_marker = "" if escrowed_avail else " (unavailable)"
            return (
                f"PRSM Wallet Balance (aggregate)\n"
                f"  Address:               {short_addr}\n"
                f"  On-chain:              {balance_ftns:.6f} FTNS\n"
                f"  Claimable royalties:   "
                f"{claimable_ftns:.6f} FTNS{claimable_marker}\n"
                f"  Escrowed (pending):    "
                f"{escrowed_ftns:.6f} FTNS{escrowed_marker}\n"
                f"  ─────────────────────\n"
                f"  Total:                 {total_ftns:.6f} FTNS\n"
                f"  USD (total):           ${total_usd:,.2f}  "
                f"(@ {usd_rate} USD/FTNS)\n"
                f"  Source:                aggregate"
            )

    # Legacy single-line format (v1 response shape OR v2 with no
    # extra sources wired).
    return (
        f"PRSM Wallet Balance\n"
        f"  Address:  {short_addr}\n"
        f"  Balance:  {balance_ftns:.6f} FTNS\n"
        f"  USD:      ${usd_equivalent:,.2f}  "
        f"(@ {usd_rate} USD/FTNS)\n"
        f"  Source:   {result['source']}"
    )


async def handle_prsm_arbitration_preview_resolution(
    arguments: Dict[str, Any],
) -> str:
    """Handle prsm_arbitration_preview_resolution: composer-only
    dry-run of a resolution proposal."""
    record_id = arguments.get("record_id")
    decision = arguments.get("decision")
    by_council = arguments.get("by_council") or []
    if not record_id or not decision or not by_council:
        return (
            "Missing required arguments. Need record_id, decision, "
            "by_council. Use prsm_arbitration_status to list pending "
            "records first."
        )

    body = {
        "record_id": record_id,
        "decision": decision,
        "by_council": list(by_council),
    }
    try:
        result = await _call_node_api(
            "POST", "/content/arbitration/preview-resolution", body,
        )
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    if "status" not in result:
        detail = result.get("detail", "unknown error")
        if "404" in detail or "No arbitration record" in detail:
            return (
                f"Record not found: {record_id}\n  Detail: {detail}"
            )
        return f"Preview composer failed.\n  Detail: {detail}"

    record = result["record"]
    proposed = result["proposed"]
    current = result.get("current_resolution")
    conflict = result.get("conflict_with_existing", False)

    lines = [
        f"PRSM Arbitration Resolution Preview (DRY_RUN)",
        f"  Record ID:        {record_id}",
        f"",
        f"  Disputed CID:     {record.get('new_cid', '')}",
        f"  Candidate parent: {record.get('candidate_parent_cid', '')}",
        f"  Similarity:       {record.get('similarity', 0.0):.4f}",
        f"  Fingerprint kind: {record.get('fingerprint_kind', '?')}",
        f"",
        f"  Proposed:",
        f"    Decision:    {proposed['decision']}",
        f"    By council:  {', '.join(proposed['by_council'])}",
    ]
    if current is not None:
        lines.append("")
        lines.append("  Current resolution (already on record):")
        lines.append(f"    Decision:    {current.get('decision', '?')}")
        cur_council = current.get("by_council", [])
        lines.append(f"    By council:  {', '.join(cur_council) or '(none)'}")
    if conflict:
        lines.append("")
        lines.append(
            "  [!] CONFLICT: proposed decision differs from existing "
            "resolution. Re-applying via queue.resolve() would be "
            "rejected (the queue raises ValueError on conflicting "
            "re-resolve). Reconcile with council before signing the "
            "on-chain proposal."
        )
    elif current is not None:
        lines.append("")
        lines.append(
            "  [-] No conflict (proposed matches existing resolution). "
            "Re-applying via queue.resolve() would be a no-op."
        )
    lines.append("")
    lines.append(
        "  Note: composer-only artifact; does NOT call queue.resolve(). "
        "Council member signs on-chain governance proposal separately."
    )
    return "\n".join(lines)


async def handle_prsm_arbitration_record_detail(
    arguments: Dict[str, Any],
) -> str:
    """Handle prsm_arbitration_record_detail: fetch full context
    for a single record + its resolution state."""
    record_id = arguments.get("record_id")
    if not record_id:
        return (
            "Missing required argument: record_id.\n"
            "Use prsm_arbitration_status to list pending records first."
        )

    try:
        result = await _call_node_api(
            "GET", f"/content/arbitration/queue/{record_id}",
        )
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "record" not in result:
        detail = result.get("detail", "unknown error")
        if "No arbitration record" in detail or "404" in detail:
            return (
                f"Record not found: {record_id}\n"
                f"  Detail: {detail}\n"
                f"  List pending records via prsm_arbitration_status."
            )
        return f"Detail fetch failed.\n  Detail: {detail}"

    record = result["record"]
    resolution = result.get("resolution")
    status = result.get("status", "?")

    lines = [
        f"PRSM Arbitration Record Detail",
        f"  Record ID:        {record_id}",
        f"  Status:           {status.upper()}",
        f"",
        f"  Disputed CID:     {record.get('new_cid', '')}",
        f"  Disputing creator: {record.get('new_creator', '')}",
        f"  Candidate parent: {record.get('candidate_parent_cid', '')}",
        f"  Parent creator:   {record.get('candidate_parent_creator', '')}",
        f"  Similarity:       {record.get('similarity', 0.0):.4f}",
        f"  Fingerprint kind: {record.get('fingerprint_kind', '?')}",
        f"  Flagged at:       {record.get('flagged_at', 0)} (unix)",
    ]
    proposal_id = record.get("proposal_id")
    if proposal_id:
        lines.append(f"  Proposal ID:      {proposal_id}")
    if resolution is not None:
        lines.append("")
        lines.append("  Resolution:")
        lines.append(f"    Decision:    {resolution.get('decision', '?')}")
        signers = resolution.get("by_council", [])
        lines.append(f"    By council:  {', '.join(signers) or '(none)'}")
    return "\n".join(lines)


async def handle_prsm_arbitration_status(
    arguments: Dict[str, Any],
) -> str:
    """Handle prsm_arbitration_status: render pending arbitration
    records."""
    try:
        result = await _call_node_api(
            "GET", "/content/arbitration/queue",
        )
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "pending" not in result:
        detail = result.get("detail", "unknown error")
        return f"Arbitration query failed.\n  Detail: {detail}"

    pending = result["pending"]
    total = result["total"]
    if total == 0:
        return "No pending arbitration disputes."

    lines = [
        f"PRSM Arbitration Queue ({total} pending)",
        "  CID                    Similarity  Kind     Proposal",
        "  " + "-" * 60,
    ]
    for r in pending[:20]:  # cap at 20 for sanity
        cid = r.get("new_cid", "")[:20]
        sim = r.get("similarity", 0.0)
        kind = r.get("fingerprint_kind", "?")
        prop = r.get("proposal_id") or "(no-proposal)"
        lines.append(
            f"  {cid:<22} {sim:>6.4f}     {kind:<8} {prop}"
        )
    if total > 20:
        lines.append(f"  ... ({total - 20} more not shown)")
    return "\n".join(lines)


async def handle_prsm_cleanup_stale_escrows(
    arguments: Dict[str, Any],
) -> str:
    """Handle prsm_cleanup_stale_escrows: force-cleanup expired
    escrows + return refunded count."""
    try:
        result = await _call_node_api("POST", "/compute/cleanup-stale")
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    if "cleaned" not in result:
        detail = result.get("detail", "unknown error")
        return f"Cleanup failed.\n  Detail: {detail}"

    cleaned = result["cleaned"]
    if cleaned == 0:
        return "No stale escrows. Nothing to clean."
    return (
        f"Cleaned up {cleaned} stale escrow(s). "
        f"FTNS refunded to requester(s)."
    )


async def handle_prsm_spend_summary(arguments: Dict[str, Any]) -> str:
    """Handle prsm_spend_summary tool call: aggregate operator's
    FTNS spend over the last N days from RELEASED escrows."""
    params = []
    days = arguments.get("days", 30)
    params.append(f"days={days}")
    if "address" in arguments:
        params.append(f"address={arguments['address']}")
    path = "/wallet/spend?" + "&".join(params)

    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    if "total_spent_ftns" not in result:
        detail = result.get("detail", "unknown error")
        return f"Spend summary failed.\n  Detail: {detail}"

    addr = result.get("address", "")
    short = addr[:10] + "…" + addr[-4:] if len(addr) > 14 else addr
    days_v = result["days"]
    total = result["total_spent_ftns"]
    count = result["escrows_count"]

    return (
        f"PRSM Spend Summary\n"
        f"  Address:        {short}\n"
        f"  Window:         last {days_v} day(s)\n"
        f"  Total spent:    {total:.6f} FTNS\n"
        f"  Released jobs:  {count}\n"
        f"  Avg / job:      "
        f"{(total / count if count else 0):.6f} FTNS"
    )


async def handle_prsm_audit_summary(
    arguments: Dict[str, Any],
) -> str:
    """Handle prsm_audit_summary: render bucketed audit counts."""
    params = []
    if "top_paths" in arguments:
        params.append(f"top_paths={arguments['top_paths']}")
    path = "/audit/summary"
    if params:
        path += "?" + "&".join(params)
    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "status_buckets" not in result:
        detail = result.get("detail", "unknown error")
        return f"Audit summary failed.\n  Detail: {detail}"

    total = result.get("total", 0)
    status = result.get("status_buckets", {})
    methods = result.get("method_buckets", {})
    top = result.get("top_paths", [])

    lines = [
        f"PRSM Audit Summary (buffer total: {total}):",
        f"",
        f"  Status buckets:",
    ]
    for bucket in ("2xx", "3xx", "4xx", "5xx", "other"):
        if bucket in status:
            lines.append(f"    {bucket}:    {status[bucket]}")
    if not status:
        lines.append("    (empty)")

    lines.append("")
    lines.append("  Methods:")
    for method, count in sorted(
        methods.items(), key=lambda kv: kv[1], reverse=True,
    ):
        lines.append(f"    {method:<8}  {count}")
    if not methods:
        lines.append("    (empty)")

    lines.append("")
    lines.append("  Top paths:")
    for entry in top:
        lines.append(
            f"    {entry['count']:>4}  {entry['path']}"
        )
    if not top:
        lines.append("    (empty)")

    return "\n".join(lines)


async def handle_prsm_audit_recent(
    arguments: Dict[str, Any],
) -> str:
    """Render recent state-changing requests from the audit ring."""
    params = []
    limit = arguments.get("limit", 20)
    params.append(f"limit={limit}")
    if "offset" in arguments:
        params.append(f"offset={arguments['offset']}")
    if "status" in arguments and arguments["status"]:
        params.append(f"status={arguments['status']}")
    if "requester" in arguments and arguments["requester"]:
        params.append(f"requester={arguments['requester']}")
    if "path_prefix" in arguments and arguments["path_prefix"]:
        params.append(f"path_prefix={arguments['path_prefix']}")
    path = "/audit/recent?" + "&".join(params)

    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    if "entries" not in result:
        detail = result.get("detail", "unknown error")
        return f"Audit fetch failed.\n  Detail: {detail}"

    entries = result["entries"]
    total = result["total"]
    total_matched = result.get("total_matched")  # only present with filter
    status_filter = result.get("status_filter")
    requester_filter = result.get("requester_filter")
    path_prefix_filter = result.get("path_prefix_filter")
    filters_applied = []
    if status_filter:
        filters_applied.append(f"status={status_filter}")
    if requester_filter:
        filters_applied.append(f"requester={requester_filter}")
    if path_prefix_filter:
        filters_applied.append(f"path_prefix={path_prefix_filter}")
    filter_str = ", ".join(filters_applied) if filters_applied else None
    if not entries:
        if filter_str:
            return (
                f"No state-changing requests matched filter "
                f"({filter_str}) (buffer total: {total})."
            )
        return (
            f"No state-changing requests recorded in audit ring "
            f"(buffer total: {total})."
        )

    header_parts = [f"PRSM Audit Log (showing {len(entries)}"]
    if total_matched is not None:
        header_parts.append(
            f" of {total_matched} matched, {total} total"
        )
        if filter_str:
            header_parts.append(f", filter={filter_str}")
        header_parts.append(")")
    else:
        header_parts.append(f" of {total})")
    lines = [
        "".join(header_parts) + ":",
        f"  Time  Method  Status  Path",
        f"  " + "-" * 70,
    ]
    import datetime
    for e in entries:
        ts = e.get("timestamp", 0)
        try:
            t = datetime.datetime.fromtimestamp(
                ts,
            ).strftime("%H:%M:%S")
        except Exception:
            t = "????"
        lines.append(
            f"  {t}  {e.get('method', '?'):<6}  "
            f"{e.get('status_code', 0):>3}     "
            f"{e.get('path', '')}"
        )
    return "\n".join(lines)


async def handle_prsm_forge_submit(
    arguments: Dict[str, Any],
) -> str:
    """Submit a query through /compute/forge."""
    body: Dict[str, Any] = {"query": arguments["query"]}
    if "budget_ftns" in arguments:
        body["budget_ftns"] = arguments["budget_ftns"]
    if "shard_cids" in arguments:
        body["shard_cids"] = arguments["shard_cids"]
    if "privacy_level" in arguments:
        body["privacy_level"] = arguments["privacy_level"]
    try:
        result = await _call_node_api(
            "POST", "/compute/forge", data=body,
        )
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "detail" in result and "job_id" not in result:
        detail = result["detail"]
        if "agent_forge" in detail.lower() or "not available" in detail.lower():
            return (
                f"Agent Forge not enabled on this node.\n"
                f"  Detail: {detail}\n"
                f"  Operator must set "
                f"PRSM_QUERY_ORCHESTRATOR_ENABLED=1 to enable."
            )
        return f"Forge submit failed.\n  Detail: {detail}"
    if result.get("status") == "idempotent_replay":
        return (
            f"Idempotent replay (cached result).\n"
            f"  job_id: {result.get('job_id', '?')}\n"
            f"  Use prsm_jobs_list / prsm_status_stream to "
            f"observe progress."
        )
    return (
        f"Query submitted to Agent Forge.\n"
        f"  job_id: {result.get('job_id', '?')}\n"
        f"  status: {result.get('status', '?')}\n"
        f"  Use prsm_jobs_list to track progress."
    )


async def handle_prsm_content_info(
    arguments: Dict[str, Any],
) -> str:
    """Render content record by CID."""
    cid = arguments.get("cid", "").strip()
    if not cid:
        return "Missing required 'cid'."
    try:
        result = await _call_node_api("GET", f"/content/{cid}")
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "cid" not in result:
        detail = result.get("detail", "unknown error")
        if "not found" in detail.lower():
            return f"Content not found: {cid}"
        if "not initialized" in detail.lower():
            return f"Content index not configured.\n  Detail: {detail}"
        return f"Content lookup failed.\n  Detail: {detail}"

    providers = result.get("providers", [])
    parent_cids = result.get("parent_cids", [])
    lines = [
        f"PRSM Content: {result['cid']}",
        f"  Filename:     {result.get('filename', '?')}",
        f"  Size:         {result.get('size_bytes', 0)} bytes",
        f"  Hash:         {result.get('content_hash', '?')}",
        f"  Creator:      {result.get('creator_id', '?')}",
        f"  Royalty:      {result.get('royalty_rate', 0):.4f}",
        f"  Providers:    {len(providers)}"
        + (f" ({', '.join(providers[:3])}...)" if len(providers) > 3
           else f" ({', '.join(providers)})" if providers else ""),
    ]
    if parent_cids:
        lines.append(f"  Parents:      {len(parent_cids)} citation(s)")
    return "\n".join(lines)


async def handle_prsm_my_content(
    arguments: Dict[str, Any],
) -> str:
    """Render content uploaded by this node."""
    params = []
    limit = arguments.get("limit", 20)
    params.append(f"limit={limit}")
    if "offset" in arguments:
        params.append(f"offset={arguments['offset']}")
    path = "/content/mine?" + "&".join(params)
    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "entries" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in detail.lower():
            return (
                f"ContentUploader not configured.\n"
                f"  Detail: {detail}"
            )
        return f"My-content fetch failed.\n  Detail: {detail}"

    entries = result["entries"]
    total = result["total"]
    if not entries:
        return (
            f"No uploaded content (total: {total}). "
            f"Upload via /content/upload or /content/upload/shard."
        )

    lines = [
        f"PRSM My Content (showing {len(entries)} of {total}):",
        f"  Content ID                   File           Royalties  Hits",
        "  " + "-" * 70,
    ]
    for e in entries:
        cid = e.get("content_id", "?")
        if len(cid) > 28:
            cid = cid[:14] + ".." + cid[-12:]
        fn = e.get("filename", "?")
        if len(fn) > 14:
            fn = fn[:11] + ".."
        royalties = e.get("total_royalties", 0.0)
        hits = e.get("access_count", 0)
        prov_marker = (
            "[chain]" if e.get("provenance_tx_hash") else "[off]"
        )
        lines.append(
            f"  {cid:<28}  {fn:<14}  {royalties:>9.6f}  "
            f"{hits:>4}  {prov_marker}"
        )
    return "\n".join(lines)


async def handle_prsm_distribution_trigger(
    arguments: Dict[str, Any],
) -> str:
    """Manually trigger pull_and_distribute."""
    try:
        result = await _call_node_api(
            "POST", "/admin/distribution/trigger",
        )
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "tx_hash" not in result:
        detail = result.get("detail", "unknown error")
        if "not wired" in detail.lower():
            return (
                f"CompensationDistributorClient not wired.\n"
                f"  Detail: {detail}"
            )
        return f"Distribution trigger failed.\n  Detail: {detail}"
    if result.get("status") == "PENDING":
        # sp915 — broadcast OK but receipt unconfirmed. Do NOT re-trigger.
        return (
            f"pull_and_distribute broadcast but UNCONFIRMED — do NOT re-trigger.\n"
            f"  tx_hash: {result.get('tx_hash')}\n"
            f"  {result.get('detail', 'Reconcile via tx_hash; do not re-trigger.')}\n"
            f"  Use prsm_distribution_history to confirm the tx lands."
        )
    return (
        f"pull_and_distribute submitted on-chain.\n"
        f"  tx_hash: {result['tx_hash']}\n"
        f"  status:  {result.get('status', '?')}\n"
        f"  Use prsm_distribution_history to confirm landing."
    )


async def handle_prsm_heartbeat_trigger(
    arguments: Dict[str, Any],
) -> str:
    """Manually trigger an on-chain heartbeat record."""
    try:
        result = await _call_node_api(
            "POST", "/admin/heartbeat/trigger",
        )
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "tx_hash" not in result:
        detail = result.get("detail", "unknown error")
        if "not wired" in detail.lower():
            return (
                f"StorageSlashingClient not wired.\n"
                f"  Detail: {detail}"
            )
        return f"Heartbeat trigger failed.\n  Detail: {detail}"
    if result.get("status") == "PENDING":
        # sp915 — broadcast OK but receipt unconfirmed. Do NOT re-trigger.
        return (
            f"Heartbeat broadcast but UNCONFIRMED — do NOT re-trigger.\n"
            f"  tx_hash: {result.get('tx_hash')}\n"
            f"  {result.get('detail', 'Reconcile via tx_hash; do not re-trigger.')}"
        )
    return (
        f"Heartbeat recorded on-chain.\n"
        f"  tx_hash: {result['tx_hash']}\n"
        f"  status:  {result.get('status', '?')}"
    )


async def handle_prsm_distribution_history(
    arguments: Dict[str, Any],
) -> str:
    """Render recent Distributed events."""
    params = []
    limit = arguments.get("limit", 20)
    params.append(f"limit={limit}")
    if "offset" in arguments:
        params.append(f"offset={arguments['offset']}")
    path = "/admin/distribution-history?" + "&".join(params)
    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "entries" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in detail.lower():
            return (
                f"Distribution log not configured.\n"
                f"  Detail: {detail}\n"
                f"  Set PRSM_COMPENSATION_DISTRIBUTOR_WATCHER_ENABLED=1."
            )
        return f"Distribution history fetch failed.\n  Detail: {detail}"

    entries = result["entries"]
    total = result["total"]
    if not entries:
        return (
            f"No distributions recorded "
            f"(buffer total: {total})."
        )

    import datetime
    lines = [
        f"PRSM Distributions (showing {len(entries)} of {total}):",
        f"  Time      Creator       Operator       Grant         Total",
        "  " + "-" * 60,
    ]
    for e in entries:
        ts = e.get("timestamp", 0)
        try:
            t = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        except Exception:
            t = "????"
        creator = e.get("to_creator", 0) / 1e18
        operator = e.get("to_operator", 0) / 1e18
        grant = e.get("to_grant", 0) / 1e18
        total_d = e.get("total_distributed", 0) / 1e18
        lines.append(
            f"  {t}  {creator:>10.4f}  {operator:>10.4f}  "
            f"{grant:>10.4f}  {total_d:>10.4f}"
        )
    return "\n".join(lines)


async def handle_prsm_heartbeat_history(
    arguments: Dict[str, Any],
) -> str:
    """Render recent on-chain HeartbeatRecorded events."""
    params = []
    limit = arguments.get("limit", 20)
    params.append(f"limit={limit}")
    if "offset" in arguments:
        params.append(f"offset={arguments['offset']}")
    if "provider" in arguments and arguments["provider"]:
        params.append(f"provider={arguments['provider']}")
    path = "/admin/heartbeat-history?" + "&".join(params)
    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "entries" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in detail.lower():
            return (
                f"Heartbeat log not configured.\n"
                f"  Detail: {detail}\n"
                f"  Set PRSM_STORAGE_SLASHING_WATCHER_ENABLED=1 "
                f"to enable."
            )
        return f"Heartbeat history fetch failed.\n  Detail: {detail}"

    entries = result["entries"]
    total = result["total"]
    if not entries:
        return (
            f"No heartbeats recorded "
            f"(buffer total: {total}). "
            f"Either the watcher hasn't started or no heartbeats "
            f"have landed yet."
        )

    import datetime
    lines = [
        f"PRSM Heartbeats (showing {len(entries)} of {total}):",
        f"  Observed     On-chain TS    Provider",
        "  " + "-" * 60,
    ]
    for e in entries:
        ts = e.get("timestamp", 0)
        try:
            obs = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        except Exception:
            obs = "????"
        ots = e.get("onchain_timestamp", 0)
        try:
            on = datetime.datetime.fromtimestamp(ots).strftime("%H:%M:%S")
        except Exception:
            on = "????"
        provider = e.get("provider", "?")
        if len(provider) > 26:
            provider = provider[:8] + ".." + provider[-14:]
        lines.append(
            f"  {obs}     {on}      {provider}"
        )
    return "\n".join(lines)


async def handle_prsm_slash_history(
    arguments: Dict[str, Any],
) -> str:
    """Render recent on-chain slash events."""
    params = []
    limit = arguments.get("limit", 20)
    params.append(f"limit={limit}")
    if "offset" in arguments:
        params.append(f"offset={arguments['offset']}")
    if "provider" in arguments and arguments["provider"]:
        params.append(f"provider={arguments['provider']}")
    path = "/admin/slash-history?" + "&".join(params)
    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "entries" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in detail.lower():
            return (
                f"Slash event log not configured.\n"
                f"  Detail: {detail}\n"
                f"  Set PRSM_STORAGE_SLASHING_WATCHER_ENABLED=1 "
                f"+ slashing client env to enable."
            )
        return f"Slash history fetch failed.\n  Detail: {detail}"

    entries = result["entries"]
    total = result["total"]
    if not entries:
        return (
            f"No slash events recorded "
            f"(buffer total: {total})."
        )

    import datetime
    lines = [
        f"PRSM Slash Events (showing {len(entries)} of {total}):",
        f"  Time      Kind                          Provider           Slash ID",
        "  " + "-" * 80,
    ]
    for e in entries:
        ts = e.get("timestamp", 0)
        try:
            t = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        except Exception:
            t = "????"
        provider = e.get("provider", "?")
        if len(provider) > 18:
            provider = provider[:8] + ".." + provider[-6:]
        slash_id = e.get("slash_id", "?")
        if len(slash_id) > 18:
            slash_id = slash_id[:10] + "..."
        lines.append(
            f"  {t}  {e.get('kind', '?'):<28}  "
            f"{provider:<18}  {slash_id}"
        )
    return "\n".join(lines)


async def handle_prsm_earnings_summary(
    arguments: Dict[str, Any],
) -> str:
    """Render aggregated operator earnings dashboard."""
    try:
        result = await _call_node_api("GET", "/admin/earnings-summary")
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    lines = ["PRSM Operator Earnings Summary"]
    op_addr = result.get("operator_address")
    lines.append(f"  Operator: {op_addr or '(PRSM_OPERATOR_ADDRESS unset)'}")
    lines.append("")

    royalty = result.get("royalty", {})
    if royalty.get("available"):
        wei = royalty.get("claimable_wei", 0)
        ftns = wei / 1e18
        lines.append(f"  Royalty:      {ftns:.6f} FTNS claimable")
    else:
        err = royalty.get("error")
        if err:
            lines.append(f"  Royalty:      [!] error: {err}")
        else:
            lines.append(f"  Royalty:      not wired")

    hb = result.get("heartbeat", {})
    if hb.get("available"):
        if hb.get("never_recorded"):
            lines.append(
                f"  Heartbeat:    [!] never recorded — "
                f"node will be slashed at next epoch"
            )
        elif hb.get("expired"):
            lines.append(
                f"  Heartbeat:    [!] EXPIRED — last at "
                f"{hb['last_heartbeat']}, slashing window open"
            )
        elif hb.get("at_risk"):
            lines.append(
                f"  Heartbeat:    [!] at-risk — only "
                f"{hb['grace_remaining']}s grace remaining"
            )
        else:
            lines.append(
                f"  Heartbeat:    ok — {hb['grace_remaining']}s "
                f"grace remaining (of {hb['grace_seconds']}s)"
            )
    else:
        err = hb.get("error")
        if err:
            lines.append(f"  Heartbeat:    [!] error: {err}")
        else:
            lines.append(
                f"  Heartbeat:    not wired (set "
                f"PRSM_OPERATOR_ADDRESS + StorageSlashing env)"
            )

    dist = result.get("distribution", {})
    if dist.get("available"):
        if dist.get("never_distributed"):
            lines.append(f"  Distribution: never run yet")
        else:
            secs = dist.get("seconds_since", 0)
            hours = secs // 3600
            lines.append(
                f"  Distribution: last run {hours}h ago "
                f"(timestamp {dist['last_distribution']})"
            )
    else:
        err = dist.get("error")
        if err:
            lines.append(f"  Distribution: [!] error: {err}")
        else:
            lines.append(f"  Distribution: not wired")

    return "\n".join(lines)


async def handle_prsm_webhook_history(
    arguments: Dict[str, Any],
) -> str:
    """Render recent webhook dispatch attempts."""
    params = []
    limit = arguments.get("limit", 20)
    params.append(f"limit={limit}")
    if "offset" in arguments:
        params.append(f"offset={arguments['offset']}")
    path = "/admin/webhook-history?" + "&".join(params)
    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "entries" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in detail.lower():
            return (
                f"Webhook log not configured on this node.\n"
                f"  Detail: {detail}\n"
                f"  Set PRSM_WEBHOOK_URL env var to enable."
            )
        return f"Webhook history fetch failed.\n  Detail: {detail}"

    entries = result["entries"]
    total = result["total"]
    if not entries:
        return (
            f"No webhook dispatches recorded "
            f"(buffer total: {total})."
        )

    lines = [
        f"PRSM Webhook History (showing {len(entries)} of {total}):",
        f"  Time      Event                  Status  Result",
        f"  " + "-" * 60,
    ]
    import datetime
    for e in entries:
        ts = e.get("timestamp", 0)
        try:
            t = datetime.datetime.fromtimestamp(
                ts,
            ).strftime("%H:%M:%S")
        except Exception:
            t = "????"
        result_marker = "[ok]" if e.get("success") else "[!]"
        lines.append(
            f"  {t}  {e.get('event', '?'):<22}  "
            f"{e.get('status_code', '?'):<5}  "
            f"{result_marker} {e.get('error', '') if not e.get('success') else 'delivered'}"
        )
    return "\n".join(lines)


async def handle_prsm_webhook_test(
    arguments: Dict[str, Any],
) -> str:
    """Handle prsm_webhook_test: smoke-test the configured
    webhook URL via POST /admin/webhook-test."""
    try:
        result = await _call_node_api("POST", "/admin/webhook-test")
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    if "success" not in result:
        detail = result.get("detail", "unknown error")
        if "not configured" in detail.lower():
            return (
                f"Webhook not configured on this node.\n"
                f"  Detail: {detail}\n"
                f"  Set PRSM_WEBHOOK_URL env var to enable."
            )
        return f"Webhook test failed.\n  Detail: {detail}"

    success = result.get("success", False)
    status_code = result.get("status_code")
    attempts = result.get("attempts", 0)
    error = result.get("error")

    if success:
        return (
            f"PRSM Webhook Test\n"
            f"  Result:       PASS\n"
            f"  Status code:  {status_code}\n"
            f"  Attempts:     {attempts}\n"
            f"  webhook.test event delivered successfully."
        )
    return (
        f"PRSM Webhook Test\n"
        f"  Result:       FAIL\n"
        f"  Status code:  {status_code}\n"
        f"  Attempts:     {attempts}\n"
        f"  Error:        {error}\n"
        f"  Operator action: verify webhook URL reachable + "
        f"accepts POST + returns 2xx."
    )


async def handle_prsm_canonical_check(
    arguments: Dict[str, Any],
) -> str:
    """Filter /health/detailed for canonical-match fields and
    render a pass/fail summary. Designed for post-migration
    verification (e.g., after A-08 v2 RoyaltyDistributor deploy)."""
    try:
        result = await _call_node_api("GET", "/health/detailed")
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    subsystems = result.get("subsystems", {})
    if not subsystems:
        return "Endpoint returned no subsystems; cannot verify canonical pins."

    matched: list = []
    mismatched: list = []
    skipped: list = []
    for name, info in subsystems.items():
        if "canonical_match" not in info:
            # Subsystem either has no on-chain contract or the
            # canonical-match check isn't implemented for it.
            if info.get("available", False):
                skipped.append((name, "no canonical pin"))
            continue
        if info["canonical_match"]:
            matched.append((name, info.get("wired_address", "")))
        else:
            mismatched.append((
                name,
                info.get("wired_address", ""),
                info.get("canonical_address", ""),
            ))

    lines = ["PRSM Canonical-Pin Check"]
    if not mismatched:
        lines.append(f"  Result: ALL {len(matched)} PIN(S) MATCH")
    else:
        lines.append(
            f"  Result: {len(mismatched)} MISMATCH(ES) "
            f"({len(matched)} match)"
        )
    lines.append("")
    if matched:
        lines.append("  Matched:")
        for name, addr in matched:
            short = addr[:10] + "..." + addr[-4:] if len(addr) > 14 else addr
            lines.append(f"    [ok]  {name:<22}  {short}")
    if mismatched:
        lines.append("")
        lines.append("  Mismatched:")
        for name, wired, canonical in mismatched:
            lines.append(f"    [!]   {name}")
            lines.append(f"            wired:     {wired}")
            lines.append(f"            canonical: {canonical}")
        lines.append("")
        lines.append(
            "  Operator action: update PRSM_*_ADDRESS env override "
            "to canonical address(es), OR remove the env override "
            "to fall through to networks.py defaults."
        )
    if skipped:
        lines.append("")
        lines.append("  Skipped (no canonical pin available):")
        for name, reason in skipped:
            lines.append(f"    [-]   {name:<22}  ({reason})")
    return "\n".join(lines)


async def handle_prsm_metrics_summary(
    arguments: Dict[str, Any],
) -> str:
    """Handle prsm_metrics_summary: parse /metrics text and
    render gauges as a side-panel summary."""
    try:
        body = await _call_node_api("GET", "/metrics", raw_text=True)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    if not isinstance(body, str):
        return f"Unexpected metrics response type: {type(body).__name__}"

    gauges: list = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # "metric_name VALUE" — split on whitespace.
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        name, value = parts
        gauges.append((name, value))

    if not gauges:
        return "Metrics endpoint returned no parseable gauges."

    lines = ["PRSM Metrics Summary:"]
    for name, value in gauges:
        # Strip the "prsm_" prefix for readability.
        label = name[5:] if name.startswith("prsm_") else name
        lines.append(f"  {label:<32}  {value}")
    return "\n".join(lines)


async def handle_prsm_node_health(arguments: Dict[str, Any]) -> str:
    """Handle prsm_node_health tool call: render structured
    per-subsystem readiness from /health/detailed."""
    try:
        result = await _call_node_api("GET", "/health/detailed")
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    status = result.get("status", "unknown")
    node_id = result.get("node_id", "unknown")
    subsystems = result.get("subsystems", {})

    lines = [
        f"PRSM Node Health",
        f"  Node ID:     {node_id}",
        f"  Status:      {status.upper()}",
        f"",
        f"  Subsystems:",
    ]
    for name, info in subsystems.items():
        avail = info.get("available", False)
        marker = "[ok]" if avail else "[--]"
        # Sprint 404 — tick_status from sprint 399-401
        # daemon extensions takes priority on the marker so
        # silent-economic-failure modes (task running but
        # every tick failing) surface as loudly as
        # cleanup_task CRASHED. Stale → loud, degraded →
        # warning, otherwise existing behavior.
        tick_status = info.get("tick_status")
        if tick_status == "stale":
            marker = "[!]"
        elif tick_status == "degraded":
            marker = "[⚠]"
        sub_status = info.get("status", "?")
        line = f"    {marker} {name:<22}  {sub_status}"
        # Append tick-age annotation when tick_status is
        # present + non-healthy. Operators see both the
        # severity AND how stale things are.
        if tick_status in ("stale", "degraded"):
            age = info.get("last_tick_age_seconds")
            if isinstance(age, (int, float)):
                line += (
                    f"  [{tick_status}: tick {int(age)}s old]"
                )
            else:
                line += f"  [{tick_status}: no tick yet]"
        if not avail and "error" in info:
            line += f"  (error: {info['error']})"
        elif name == "payment_escrow" and "pending_count" in info:
            line += f"  (pending: {info['pending_count']})"
            # Surface cleanup-task crash explicitly with [!] marker
            # since it's a high-sev silent failure.
            if info.get("cleanup_task_running") is False:
                line += "  [!] cleanup_task CRASHED"
            elif info.get("cleanup_task_running") is True:
                line += "  cleanup_task: ok"
        elif name == "job_history" and "count" in info:
            persisted = info.get("persisted", False)
            line += (
                f"  (count: {info['count']}, "
                f"persisted: {'yes' if persisted else 'no'})"
            )
        elif name == "ftns_ledger" and info.get("connected_address"):
            addr = info["connected_address"]
            short = addr[:10] + "…" + addr[-4:] if len(addr) > 14 else addr
            line += f"  ({short})"
        elif name == "royalty_distributor" and "claimable_wei" in info:
            line += f"  (claimable: {info['claimable_wei']} wei)"
        elif name == "bootstrap_discovery" and (
            "client_state" in info
        ):
            # Sprint 331 — surface client_state + peers count
            # inline so operators triaging via MCP see the
            # load-bearing discovery fields without drilling
            # into /bootstrap/status. Mirrors the pattern of
            # payment_escrow surfacing pending_count.
            # Sprint 376 — also surface active_url when set,
            # so operators triaging "we're degraded" see at
            # a glance whether they're on primary, fallback,
            # or no host at all.
            cs = info.get("client_state", "?")
            peers = info.get("discovered_peer_count", 0)
            active_url = info.get("active_url")
            if active_url:
                # Compact rendering: extract host:port from
                # the wss:// URL — operators recognize
                # bootstrap1 / bootstrap-eu / bootstrap-apac
                # without the full URL noise.
                short_url = active_url
                if "://" in short_url:
                    short_url = short_url.split("://", 1)[1]
                line += (
                    f"  (client_state={cs}, peers={peers}, "
                    f"active={short_url})"
                )
            else:
                line += (
                    f"  (client_state={cs}, peers={peers})"
                )
        elif "jobs_count" in info:
            # Sprint 344 — sprint-342 orchestrators surface
            # jobs_count. FL + pipeline-inference both follow
            # the same shape; checking the field rather than
            # the name keeps this generic.
            line += f"  (jobs={info['jobs_count']})"
        elif "record_count" in info:
            # Sprint 344 — sprint-343 stores surface
            # record_count. Same generic field check; covers
            # content_filter_store / disclosure_intake /
            # incident_response / corp_capability_store /
            # upgrade_orchestrator without per-name branches.
            line += f"  (records={info['record_count']})"
        lines.append(line)
        # Canonical-match indicator (shipped post-A-08 ceremony):
        # surface mismatches loudly so operators see stale env
        # overrides without scrolling. Match=True is shown subtly;
        # Match=False is the load-bearing signal.
        if "canonical_match" in info:
            if info["canonical_match"]:
                lines.append(
                    f"      -> canonical pin matches "
                    f"({info.get('wired_address', '?')[:10]}...)"
                )
            else:
                wired = info.get("wired_address", "?")
                canon = info.get("canonical_address", "?")
                lines.append(
                    f"      [!] canonical MISMATCH: wired={wired}, "
                    f"canonical={canon}"
                )
                lines.append(
                    f"        (operator action: update "
                    f"PRSM_*_ADDRESS env override or accept canonical)"
                )
    return "\n".join(lines)


async def handle_prsm_escrow_lookup(arguments: Dict[str, Any]) -> str:
    """Handle prsm_escrow_lookup: direct lookup by escrow_id."""
    escrow_id = arguments.get("escrow_id")
    if not escrow_id:
        return (
            "Missing required argument: escrow_id.\n"
            "Use prsm_escrow_summary to list active escrows first."
        )
    try:
        result = await _call_node_api(
            "GET", f"/wallet/escrows/{escrow_id}",
        )
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "escrow_id" not in result:
        detail = result.get("detail", "unknown error")
        if "404" in detail or "No escrow record" in detail:
            return (
                f"Escrow not found: {escrow_id}\n"
                f"  Detail: {detail}"
            )
        return f"Escrow lookup failed.\n  Detail: {detail}"

    lines = [
        f"PRSM Escrow Detail",
        f"  Escrow ID:        {result['escrow_id']}",
        f"  Job ID:           {result.get('job_id', '?')}",
        f"  Requester:        {result.get('requester_id', '?')}",
        f"  Amount (FTNS):    {result.get('amount_ftns', 0):.6f}",
        f"  Status:           {result.get('status', '?').upper()}",
    ]
    if result.get("provider_winner"):
        lines.append(f"  Provider winner:  {result['provider_winner']}")
    if result.get("tx_lock"):
        lines.append(f"  Lock tx:          {result['tx_lock']}")
    if result.get("tx_release"):
        lines.append(f"  Release tx:       {result['tx_release']}")
    if result.get("created_at"):
        lines.append(f"  Created at:       {result['created_at']}")
    if result.get("completed_at"):
        lines.append(f"  Completed at:     {result['completed_at']}")
    return "\n".join(lines)


async def handle_prsm_escrow_summary(arguments: Dict[str, Any]) -> str:
    """Handle prsm_escrow_summary tool call: enumerate operator's
    active escrows."""
    params = []
    if "address" in arguments:
        params.append(f"address={arguments['address']}")
    if arguments.get("include_terminal"):
        params.append("include_terminal=true")
    path = "/wallet/escrows"
    if params:
        path += "?" + "&".join(params)

    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    if "escrows" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in detail.lower():
            return (
                f"PaymentEscrow not configured on this node.\n"
                f"  Detail: {detail}"
            )
        return f"prsm_escrow_summary failed.\n  Detail: {detail}"

    escrows = result["escrows"]
    total = result["total"]
    locked = result["total_locked_ftns"]
    addr = result.get("address", "")
    short_addr = (
        addr[:10] + "…" + addr[-4:] if len(addr) > 14 else addr
    )

    if not escrows:
        return (
            f"PRSM Escrow Summary\n"
            f"  Address:  {short_addr}\n"
            f"  No active escrows."
        )

    lines = [
        f"PRSM Escrow Summary",
        f"  Address:        {short_addr}",
        f"  Active escrows: {total}",
        f"  Locked (PENDING): {locked:.6f} FTNS",
        f"",
        f"  Job ID            Amount        Status",
        f"  " + "-" * 50,
    ]
    for e in escrows:
        lines.append(
            f"  {e['job_id']:<16}  "
            f"{e['amount_ftns']:>10.6f}  "
            f"{e['status']}"
        )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Sprint 209 — prsm_status_stream
# Consumes /compute/status/{job_id}/stream SSE feed and renders
# status-transition trajectory for end-users polling job progress
# via MCP.
# ──────────────────────────────────────────────────────────────────────


async def _consume_status_stream(
    job_id: str, *, max_wait_sec: float,
) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    """Open a GET-SSE connection to /compute/status/{job_id}/stream,
    collect unique status snapshots, return when terminal event or
    max_wait_sec elapses.

    Returns ``(snapshots, terminal_reason, last_status)``:
      - ``snapshots`` — ordered list of unique status-dict snapshots
      - ``terminal_reason`` — one of "completed"/"history_terminal"/
        "escrow_terminal"/"timeout" (server-side) or "client_timeout"
        (max_wait_sec elapsed before server emitted terminal)
      - ``last_status`` — best-effort final status string

    Raises ``RuntimeError`` on network failure (caller renders).
    """
    import aiohttp
    import asyncio as _asyncio

    url = await _get_node_api_url()
    api_key = os.environ.get("PRSM_NODE_API_KEY", "")
    headers = {"Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    snapshots: List[Dict[str, Any]] = []
    terminal_reason: str = "client_timeout"
    last_status: Optional[str] = None

    async def _read():
        nonlocal terminal_reason, last_status
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{url}/compute/status/{job_id}/stream",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=max_wait_sec + 10),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(
                        f"HTTP {resp.status}: {body[:200]}"
                    )
                async for event_type, data in _parse_sse(resp):
                    try:
                        payload = json.loads(data) if data else {}
                    except json.JSONDecodeError:
                        payload = {"raw": data}
                    if event_type == "status":
                        # Dedup by JSON equality against prior tail
                        # snapshot — same shape the SSE emitter uses.
                        if not snapshots or snapshots[-1] != payload:
                            snapshots.append(payload)
                            last_status = (
                                payload.get("status")
                                or payload.get("history", {}).get("status")
                                or last_status
                            )
                    elif event_type == "terminal":
                        terminal_reason = (
                            payload.get("reason") or "completed"
                        )
                        # Capture any final-status field if present.
                        last_status = (
                            payload.get("status") or last_status
                        )
                        return
                    elif event_type == "error":
                        terminal_reason = "error"
                        last_status = (
                            payload.get("error") or last_status
                        )
                        return

    try:
        await _asyncio.wait_for(_read(), timeout=max_wait_sec)
    except _asyncio.TimeoutError:
        terminal_reason = "client_timeout"
    return snapshots, terminal_reason, last_status


async def handle_prsm_status_stream(
    arguments: Dict[str, Any],
    *, emit_progress: Optional[Any] = None,
) -> str:
    """Stream job-status transitions for a given job_id.

    Blocks until the server emits a terminal SSE event OR
    max_wait_sec elapses. Returns a rendered trajectory of unique
    status snapshots + the terminal reason.
    """
    job_id = (arguments.get("job_id") or "").strip()
    if not job_id:
        return "Missing required 'job_id' (non-empty)."

    raw_wait = arguments.get("max_wait_sec", 60)
    try:
        max_wait_sec = float(raw_wait)
    except (TypeError, ValueError):
        max_wait_sec = 60.0
    # Clamp to [1, 600]. Sub-second polling wastes worker; 10-min
    # ceiling avoids accidentally hanging the LLM session forever.
    if max_wait_sec < 1:
        max_wait_sec = 1.0
    if max_wait_sec > 600:
        max_wait_sec = 600.0

    try:
        snapshots, terminal_reason, last_status = (
            await _consume_status_stream(
                job_id, max_wait_sec=max_wait_sec,
            )
        )
    except Exception as e:
        return (
            f"prsm_status_stream failed for job_id={job_id}: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )

    # Emit one progress per unique transition (optional).
    if emit_progress is not None:
        try:
            for i, snap in enumerate(snapshots):
                s = (
                    snap.get("status")
                    or snap.get("history", {}).get("status")
                    or "?"
                )
                await emit_progress(
                    progress=i + 1,
                    total=max(1, len(snapshots)),
                    message=f"status={s}",
                )
        except Exception:  # noqa: BLE001
            # Progress is best-effort; don't fail the call.
            pass

    lines = [
        f"Status stream for job_id={job_id}:",
    ]
    if not snapshots:
        lines.append(
            f"  (no status frames received before "
            f"{terminal_reason}; final={last_status or 'unknown'})"
        )
    else:
        for i, snap in enumerate(snapshots):
            s = (
                snap.get("status")
                or snap.get("history", {}).get("status")
                or "?"
            )
            lines.append(f"  {i+1}. status={s}")
    lines.append(
        f"  terminal_reason={terminal_reason}; "
        f"final_status={last_status or 'unknown'}"
    )
    return "\n".join(lines)


async def handle_prsm_settler_admin(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 234 — settler write actions: register, unbond,
    sign (batch), slash (propose)."""
    import math as _math
    import urllib.parse as _up
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return (
            "Missing required 'action' (register, unbond, sign, "
            "or slash)."
        )
    if action not in ("register", "unbond", "sign", "slash"):
        return (
            f"action must be one of register/unbond/sign/slash; "
            f"got {action!r}."
        )

    if action == "register":
        for req in ("settler_id", "address", "bond_amount"):
            if not arguments.get(req):
                return f"register requires '{req}'."
        try:
            bond = float(arguments["bond_amount"])
        except (TypeError, ValueError):
            return (
                f"bond_amount must be a finite positive number; "
                f"got {arguments['bond_amount']!r}."
            )
        if not _math.isfinite(bond) or bond <= 0:
            return (
                f"bond_amount must be a finite positive number; "
                f"got {bond}."
            )
        sid = _up.quote(arguments["settler_id"])
        addr = _up.quote(arguments["address"])
        path = (
            f"/settler/register?settler_id={sid}"
            f"&address={addr}&bond_amount={bond}"
        )
    elif action == "unbond":
        if not arguments.get("settler_id"):
            return "unbond requires 'settler_id'."
        sid = _up.quote(arguments["settler_id"])
        path = f"/settler/unbond?settler_id={sid}"
    elif action == "sign":
        for req in ("batch_id", "settler_id", "signature"):
            if not arguments.get(req):
                return f"sign requires '{req}'."
        bid = _up.quote(arguments["batch_id"])
        sid = _up.quote(arguments["settler_id"])
        sig = _up.quote(arguments["signature"])
        path = (
            f"/settler/batch/sign?batch_id={bid}"
            f"&settler_id={sid}&signature={sig}"
        )
    else:  # slash
        for req in (
            "settler_id", "slash_amount", "reason", "proposer_id",
        ):
            if arguments.get(req) is None or arguments.get(req) == "":
                return f"slash requires '{req}'."
        try:
            slash = float(arguments["slash_amount"])
        except (TypeError, ValueError):
            return (
                f"slash_amount must be a finite positive number; "
                f"got {arguments['slash_amount']!r}."
            )
        if not _math.isfinite(slash) or slash <= 0:
            return (
                f"slash_amount must be a finite positive number; "
                f"got {slash}."
            )
        sid = _up.quote(arguments["settler_id"])
        pid = _up.quote(arguments["proposer_id"])
        reason = _up.quote(arguments["reason"])
        path = (
            f"/settler/slash/propose?settler_id={sid}"
            f"&slash_amount={slash}&reason={reason}"
            f"&proposer_id={pid}"
        )

    try:
        result = await _call_node_api("POST", path)
    except Exception as e:
        return (
            f"prsm_settler_admin failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if not isinstance(result, dict):
        return f"Settler {action} returned: {result}"
    if "detail" in result and "status" not in result and "settler_id" not in result and "batch_id" not in result and "proposal_id" not in result:
        return f"Settler {action} refused: {result.get('detail', '?')}"
    lines = [f"Settler {action} executed:"]
    for k, v in result.items():
        lines.append(f"  {k:<20} {v}")
    return "\n".join(lines)


async def handle_prsm_inference_quote(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 237 — pre-flight inference cost quote via
    POST /compute/inference/quote. Pairs with prsm_inference."""
    prompt = (arguments.get("prompt") or "").strip()
    if not prompt:
        return "Missing required 'prompt' (non-empty)."
    model_id = (arguments.get("model_id") or "").strip()
    if not model_id:
        return (
            "Missing required 'model_id'. Use prsm_models to "
            "discover available IDs."
        )
    body: Dict[str, Any] = {
        "prompt": prompt, "model_id": model_id,
    }
    for opt in (
        "privacy_tier", "content_tier", "max_tokens", "temperature",
    ):
        if opt in arguments and arguments[opt] is not None:
            body[opt] = arguments[opt]
    try:
        result = await _call_node_api(
            "POST", "/compute/inference/quote", body,
        )
    except Exception as e:
        return (
            f"prsm_inference_quote failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "cost_ftns" not in result:
        detail = result.get("detail", "unknown error")
        return f"Quote refused: {detail}"
    lines = [
        "PRSM Inference Quote:",
        f"  model_id:        {result.get('model_id', '?')}",
        f"  cost_ftns:       {result.get('cost_ftns', '?')} FTNS",
        f"  privacy_tier:    {result.get('privacy_tier', '?')}",
        f"  content_tier:    {result.get('content_tier', '?')}",
    ]
    # Sprint 262 — surface projected ε spend + remaining budget
    # so the caller spots an "ε exhausted before FTNS" rejection
    # before submitting.
    eps = result.get("epsilon_estimated")
    if eps is not None:
        lines.append(f"  epsilon_spend:   {eps}")
    remaining = result.get("privacy_budget_remaining")
    if remaining is not None:
        lines.append(f"  privacy_budget_remaining: {remaining}")
        if eps not in (None, 0.0) and remaining < float(eps):
            lines.append(
                f"  ⚠  ε spend ({eps}) > remaining "
                f"({remaining}); request will reject at "
                f"the /compute/inference pre-flight gate."
            )
    lines.append(
        "  Run prsm_inference with budget_ftns ≥ cost_ftns "
        "to execute."
    )
    return "\n".join(lines)


async def handle_prsm_forge_quote(arguments: Dict[str, Any]) -> str:
    """Sprint 236 — network-aware forge quote via POST
    /compute/forge/quote. Pairs with prsm_forge_submit."""
    query = (arguments.get("query") or "").strip()
    if not query:
        return "Missing required 'query' (non-empty)."
    body: Dict[str, Any] = {"query": query}
    if "shard_cids" in arguments and arguments["shard_cids"]:
        body["shard_cids"] = list(arguments["shard_cids"])
    elif "shard_count" in arguments:
        try:
            sc = int(arguments["shard_count"])
        except (TypeError, ValueError):
            return f"shard_count must be an integer; got {arguments['shard_count']!r}."
        if sc < 1 or sc > 100:
            return f"shard_count must be in [1, 100]; got {sc}."
        body["shard_count"] = sc
    if "hardware_tier" in arguments:
        tier = str(arguments["hardware_tier"]).strip().lower()
        if tier not in ("t1", "t2", "t3", "t4"):
            return f"hardware_tier must be t1/t2/t3/t4; got {tier!r}."
        body["hardware_tier"] = tier
    if "estimated_pcu_per_shard" in arguments:
        try:
            pcu = float(arguments["estimated_pcu_per_shard"])
        except (TypeError, ValueError):
            return (
                f"estimated_pcu_per_shard must be a positive "
                f"number; got {arguments['estimated_pcu_per_shard']!r}."
            )
        import math as _math
        if not _math.isfinite(pcu) or pcu <= 0:
            return f"estimated_pcu_per_shard must be a positive finite number; got {pcu}."
        body["estimated_pcu_per_shard"] = pcu

    try:
        result = await _call_node_api("POST", "/compute/forge/quote", body)
    except Exception as e:
        return (
            f"prsm_forge_quote failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "total" not in result:
        detail = result.get("detail", "unknown error")
        return f"Quote refused: {detail}"
    return (
        f"PRSM Forge Quote:\n"
        f"  Compute:     {result.get('compute_cost', '?')} FTNS\n"
        f"  Data:        {result.get('data_cost', '?')} FTNS\n"
        f"  Network Fee: {result.get('network_fee', '?')} FTNS\n"
        f"  Total:       {result.get('total', '?')} FTNS\n"
        f"  Hardware:    {result.get('hardware_tier', '?').upper()}\n"
        f"  Shards:      {result.get('shard_count', '?')}\n"
        f"  Use prsm_forge_submit with budget ≥ Total to execute."
    )


async def handle_prsm_content_provider_stats(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 268 — render ContentProvider runtime stats."""
    try:
        result = await _call_node_api(
            "GET", "/content/provider-stats",
        )
    except Exception as e:
        return (
            f"prsm_content_provider_stats failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if not isinstance(result, dict):
        return f"Unexpected response shape: {result!r}"
    if "detail" in result and "local_content_count" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in str(detail).lower():
            return (
                f"Content provider not wired on this node.\n"
                f"  Detail: {detail}"
            )
        return f"prsm_content_provider_stats refused: {detail}"
    lines = ["PRSM Content Provider Stats:"]
    for k in sorted(result.keys()):
        v = result[k]
        if isinstance(v, dict):
            lines.append(f"  {k}:")
            for sk in sorted(v.keys()):
                lines.append(f"    {sk:<22} {v[sk]}")
        else:
            lines.append(f"  {k:<24} {v}")
    return "\n".join(lines)


_CONTENT_FILTER_ACTIONS = {
    "list", "add_cids", "remove_cid",
    "add_tags", "remove_tag", "set_action",
}


async def handle_prsm_content_filter(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 270 — operator's content-filter CRUD via MCP.

    action selector mirrors prsm_agent_admin / prsm_settler_admin
    patterns. Per R9-SCOPING-1 §8 this is operator-local — the
    blocklist is never propagated to other operators."""
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return (
            f"Missing required 'action' (must be one of "
            f"{sorted(_CONTENT_FILTER_ACTIONS)})."
        )
    if action not in _CONTENT_FILTER_ACTIONS:
        return (
            f"action must be one of "
            f"{sorted(_CONTENT_FILTER_ACTIONS)}; got {action!r}."
        )

    if action == "list":
        try:
            result = await _call_node_api(
                "GET", "/admin/content-filter",
            )
        except Exception as e:
            return (
                f"prsm_content_filter failed: {e}\n"
                f"Is your PRSM node running? (prsm node start)"
            )
        if "count_cids" not in result:
            detail = result.get("detail", "unknown error")
            if "not initialized" in str(detail).lower():
                return (
                    f"Content filter store not wired on this "
                    f"node.\n  Detail: {detail}\n"
                    f"  Set PRSM_CONTENT_FILTER_DIR for "
                    f"persistence."
                )
            return f"list refused: {detail}"
        cids = result.get("blocked_content_ids") or []
        tags = result.get("blocked_model_tags") or []
        patterns = result.get("blocked_input_patterns") or []
        lines = [
            "PRSM Content Filter — operator blocklist:",
            f"  action_on_match: {result.get('action_on_match', '?')}",
            f"  count_cids:      {len(cids)}",
            f"  count_tags:      {len(tags)}",
            f"  count_patterns:  {len(patterns)}",
        ]
        if cids:
            lines.append("  blocked_content_ids:")
            for c in cids[:20]:
                lines.append(f"    {c}")
            if len(cids) > 20:
                lines.append(f"    ...+{len(cids) - 20} more")
        if tags:
            lines.append("  blocked_model_tags:")
            for t in tags:
                lines.append(f"    {t}")
        if patterns:
            lines.append("  blocked_input_patterns:")
            for p in patterns[:10]:
                lines.append(f"    {p}")
            if len(patterns) > 10:
                lines.append(
                    f"    ...+{len(patterns) - 10} more"
                )
        return "\n".join(lines)

    if action == "add_cids":
        cids = arguments.get("cids")
        if not isinstance(cids, list):
            return "add_cids requires 'cids' as a list of strings."
        try:
            result = await _call_node_api(
                "POST", "/admin/content-filter/cids",
                {"cids": cids},
            )
        except Exception as e:
            return (
                f"prsm_content_filter failed: {e}\n"
                f"Is your PRSM node running? (prsm node start)"
            )
        if "added" not in result:
            detail = result.get("detail", "unknown error")
            if "not initialized" in str(detail).lower():
                return (
                    f"Content filter store not wired.\n"
                    f"  Detail: {detail}"
                )
            return f"add_cids refused: {detail}"
        return (
            f"Content filter updated: added={result['added']}, "
            f"total={result.get('total', '?')}."
        )

    if action == "remove_cid":
        cid = (arguments.get("cid") or "").strip()
        if not cid:
            return "remove_cid requires 'cid'."
        try:
            result = await _call_node_api(
                "DELETE",
                f"/admin/content-filter/cids/{cid}",
            )
        except Exception as e:
            return (
                f"prsm_content_filter failed: {e}\n"
                f"Is your PRSM node running? (prsm node start)"
            )
        if "removed" not in result:
            detail = result.get("detail", "unknown error")
            if "not in blocklist" in str(detail).lower():
                return (
                    f"cid={cid!r} not in blocklist; nothing to "
                    f"remove."
                )
            if "not initialized" in str(detail).lower():
                return (
                    f"Content filter store not wired.\n"
                    f"  Detail: {detail}"
                )
            return f"remove_cid refused: {detail}"
        return (
            f"Removed {result['removed']!s} from filter "
            f"(total now {result.get('total', '?')})."
        )

    if action == "add_tags":
        tags = arguments.get("tags")
        if not isinstance(tags, list):
            return "add_tags requires 'tags' as a list of strings."
        try:
            result = await _call_node_api(
                "POST", "/admin/content-filter/tags",
                {"tags": tags},
            )
        except Exception as e:
            return (
                f"prsm_content_filter failed: {e}\n"
                f"Is your PRSM node running? (prsm node start)"
            )
        if "added" not in result:
            detail = result.get("detail", "unknown error")
            return f"add_tags refused: {detail}"
        return (
            f"Content filter updated: added={result['added']}, "
            f"total={result.get('total', '?')} tags."
        )

    if action == "remove_tag":
        tag = (arguments.get("tag") or "").strip()
        if not tag:
            return "remove_tag requires 'tag'."
        try:
            result = await _call_node_api(
                "DELETE",
                f"/admin/content-filter/tags/{tag}",
            )
        except Exception as e:
            return (
                f"prsm_content_filter failed: {e}\n"
                f"Is your PRSM node running? (prsm node start)"
            )
        if "removed" not in result:
            detail = result.get("detail", "unknown error")
            if "not in blocklist" in str(detail).lower():
                return (
                    f"tag={tag!r} not in blocklist; nothing to "
                    f"remove."
                )
            return f"remove_tag refused: {detail}"
        return (
            f"Removed tag {result['removed']!s} (total now "
            f"{result.get('total', '?')})."
        )

    # action == "set_action" — meta-action; choose the actual
    # filter action via filter_action arg to avoid name
    # collision with the dispatch action selector.
    filter_action = (
        arguments.get("filter_action") or ""
    ).strip().lower()
    if not filter_action:
        return (
            "set_action requires 'filter_action' "
            "(refuse | log_and_refuse | silent_refuse)."
        )
    try:
        result = await _call_node_api(
            "POST", "/admin/content-filter/action",
            {"action": filter_action},
        )
    except Exception as e:
        return (
            f"prsm_content_filter failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "action_on_match" not in result:
        detail = result.get("detail", "unknown error")
        return f"set_action refused: {detail}"
    return (
        f"Content filter action set to "
        f"{result['action_on_match']!s}."
    )


_TAKEDOWN_NOTICE_ACTIONS = {
    "list", "lookup", "record", "apply_to_filter", "set_status",
}
_TAKEDOWN_NOTICE_VALID_STATUSES = {
    "received", "acknowledged", "disputed", "expired",
}


async def handle_prsm_takedown_notices(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 272 — Foundation takedown-notice intake via MCP.

    Per Vision §14 / R9-SCOPING-1 §8: Foundation records
    notices (information distribution). Operators VOLUNTARILY
    act on them by editing their own ContentFilterStore
    (sprint 269) — this surface neither enforces nor
    propagates."""
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return (
            f"Missing required 'action' (must be one of "
            f"{sorted(_TAKEDOWN_NOTICE_ACTIONS)})."
        )
    if action not in _TAKEDOWN_NOTICE_ACTIONS:
        return (
            f"action must be one of "
            f"{sorted(_TAKEDOWN_NOTICE_ACTIONS)}; got {action!r}."
        )

    if action == "list":
        limit = arguments.get("limit", 50)
        offset = arguments.get("offset", 0)
        status = arguments.get("status")
        target_cid = arguments.get("target_cid")
        params = [f"limit={int(limit)}", f"offset={int(offset)}"]
        if status:
            params.append(f"status={status}")
        if target_cid:
            params.append(f"target_cid={target_cid}")
        path = "/admin/takedown-notices?" + "&".join(params)
        try:
            result = await _call_node_api("GET", path)
        except Exception as e:
            return (
                f"prsm_takedown_notices failed: {e}\n"
                f"Is your PRSM node running? (prsm node start)"
            )
        if "notices" not in result:
            detail = result.get("detail", "unknown error")
            if "not initialized" in str(detail).lower():
                return (
                    f"Takedown notice ring not wired on this "
                    f"node.\n  Detail: {detail}\n"
                    f"  Set PRSM_TAKEDOWN_NOTICE_LOG_DIR for "
                    f"persistence."
                )
            return f"list refused: {detail}"
        notices = result.get("notices") or []
        total = result.get("total", 0)
        lines = [
            f"PRSM Takedown Notices — {len(notices)} of "
            f"{total} (offset={result.get('offset', 0)}):",
        ]
        if not notices:
            lines.append("  (none)")
        for n in notices:
            lines.append(
                f"  {n.get('notice_id', '?')[:8]}  "
                f"{n.get('status', '?'):>12}  "
                f"target={n.get('target_cid', '?')}  "
                f"jur={n.get('jurisdiction', '?')}  "
                f"basis={n.get('basis', '?')}"
            )
        return "\n".join(lines)

    if action == "lookup":
        notice_id = (arguments.get("notice_id") or "").strip()
        if not notice_id:
            return "lookup requires 'notice_id'."
        try:
            result = await _call_node_api(
                "GET", f"/admin/takedown-notices/{notice_id}",
            )
        except Exception as e:
            return (
                f"prsm_takedown_notices failed: {e}\n"
                f"Is your PRSM node running? (prsm node start)"
            )
        if "notice_id" not in result:
            detail = result.get("detail", "unknown error")
            if "no notice with id" in str(detail).lower():
                return f"No notice with id={notice_id!r}."
            if "not initialized" in str(detail).lower():
                return (
                    f"Takedown notice ring not wired.\n"
                    f"  Detail: {detail}"
                )
            return f"lookup refused: {detail}"
        return (
            f"Takedown Notice {result['notice_id']}:\n"
            f"  timestamp:    {result.get('timestamp', '?')}\n"
            f"  status:       {result.get('status', '?')}\n"
            f"  target_cid:   {result.get('target_cid', '?')}\n"
            f"  sender:       {result.get('sender', '?')}\n"
            f"  jurisdiction: {result.get('jurisdiction', '?')}\n"
            f"  basis:        {result.get('basis', '?')}\n"
            f"  notice_text:\n"
            f"    {result.get('notice_text', '')[:1024]}"
        )

    if action == "set_status":
        notice_id = (arguments.get("notice_id") or "").strip()
        if not notice_id:
            return "set_status requires 'notice_id'."
        new_status = (
            arguments.get("notice_status") or ""
        ).strip().lower()
        if not new_status:
            return (
                f"set_status requires 'notice_status' "
                f"(must be one of "
                f"{sorted(_TAKEDOWN_NOTICE_VALID_STATUSES)})."
            )
        if new_status not in _TAKEDOWN_NOTICE_VALID_STATUSES:
            return (
                f"notice_status must be one of "
                f"{sorted(_TAKEDOWN_NOTICE_VALID_STATUSES)}; "
                f"got {new_status!r}."
            )
        try:
            result = await _call_node_api(
                "POST",
                f"/admin/takedown-notices/{notice_id}/status",
                {"status": new_status},
            )
        except Exception as e:
            return (
                f"prsm_takedown_notices failed: {e}\n"
                f"Is your PRSM node running? (prsm node start)"
            )
        if "status" not in result:
            detail = result.get("detail", "unknown error")
            if "no notice with id" in str(detail).lower():
                return f"No notice with id={notice_id!r}."
            if "not initialized" in str(detail).lower():
                return (
                    f"Takedown notice ring not wired.\n"
                    f"  Detail: {detail}"
                )
            return f"set_status refused: {detail}"
        return (
            f"Notice {notice_id} status set to "
            f"{result['status']!s}."
        )

    if action == "apply_to_filter":
        notice_id = (arguments.get("notice_id") or "").strip()
        if not notice_id:
            return "apply_to_filter requires 'notice_id'."
        try:
            result = await _call_node_api(
                "POST",
                f"/admin/content-filter/from-notice/{notice_id}",
            )
        except Exception as e:
            return (
                f"prsm_takedown_notices failed: {e}\n"
                f"Is your PRSM node running? (prsm node start)"
            )
        if "added" not in result:
            detail = result.get("detail", "unknown error")
            if "no notice with id" in str(detail).lower():
                return f"No notice with id={notice_id!r}."
            if "not initialized" in str(detail).lower():
                return (
                    f"Required surface not wired.\n"
                    f"  Detail: {detail}"
                )
            return f"apply_to_filter refused: {detail}"
        added = result.get("added", 0)
        target = result.get("target_cid", "?")
        status = result.get("notice_status", "?")
        if added == 0:
            note = f" (already in operator blocklist)"
        else:
            note = ""
        return (
            f"Notice applied: target_cid={target} added to "
            f"operator content filter{note}; notice now "
            f"status={status}."
        )

    # action == "record"
    target_cid = (arguments.get("target_cid") or "").strip()
    sender = (arguments.get("sender") or "").strip()
    jurisdiction = (arguments.get("jurisdiction") or "").strip()
    basis = (arguments.get("basis") or "").strip()
    notice_text = arguments.get("notice_text") or ""
    missing = [
        k for k, v in (
            ("target_cid", target_cid),
            ("sender", sender),
            ("jurisdiction", jurisdiction),
            ("basis", basis),
        ) if not v
    ]
    if missing:
        return (
            f"record requires: {', '.join(missing)}."
        )
    try:
        result = await _call_node_api(
            "POST", "/admin/takedown-notice",
            {
                "target_cid": target_cid, "sender": sender,
                "jurisdiction": jurisdiction, "basis": basis,
                "notice_text": notice_text,
            },
        )
    except Exception as e:
        return (
            f"prsm_takedown_notices failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "notice_id" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in str(detail).lower():
            return (
                f"Takedown notice ring not wired.\n"
                f"  Detail: {detail}"
            )
        return f"record refused: {detail}"
    return (
        f"Takedown notice recorded: id={result['notice_id']}, "
        f"target={result.get('target_cid', '?')}, "
        f"status={result.get('status', '?')}."
    )


async def handle_prsm_fiat_surface_health(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 286 — operator inspection of fiat-surface
    health findings. No actions — single read-only endpoint
    hit. Renders ERROR/WARN findings with remediation hints
    so operators see dangerous combos before vendor traffic
    arrives."""
    try:
        result = await _call_node_api(
            "GET", "/admin/fiat-surface/health",
        )
    except Exception as e:
        return (
            f"prsm_fiat_surface_health failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "overall" not in result:
        detail = result.get("detail", "unknown error")
        return f"health check refused: {detail}"
    overall = result.get("overall", "?")
    error_count = result.get("error_count", 0)
    warn_count = result.get("warn_count", 0)
    info_count = result.get("info_count", 0)
    lines = [
        f"PRSM Fiat-Surface Health — overall={overall}",
        f"  ERROR={error_count}  WARN={warn_count}  "
        f"INFO={info_count}",
    ]
    findings = result.get("findings") or []
    if not findings:
        lines.append("  (no findings — surface is clean)")
        return "\n".join(lines)
    # Render in severity order: ERROR → WARN → INFO
    severity_order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    findings_sorted = sorted(
        findings,
        key=lambda f: severity_order.get(
            f.get("severity", "INFO"), 9,
        ),
    )
    for f in findings_sorted:
        sev = f.get("severity", "?")
        cause = f.get("cause", "?")
        remediation = f.get("remediation", "")
        marker = (
            "⚠ ERROR" if sev == "ERROR"
            else "△ WARN" if sev == "WARN"
            else "· INFO"
        )
        lines.append("")
        lines.append(f"  {marker}  {cause}")
        # Indent remediation text
        for line in remediation.split("\n"):
            lines.append(f"      {line}")
    return "\n".join(lines)


_FIAT_COMPLIANCE_ACTIONS = {"list", "summary", "lookup"}


async def handle_prsm_fiat_compliance(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 282 — operator query surface for the fiat
    compliance audit ring. action selector: list | summary |
    lookup. No write paths — recording is automatic from
    quote + execute handlers."""
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return (
            f"Missing required 'action' (must be one of "
            f"{sorted(_FIAT_COMPLIANCE_ACTIONS)})."
        )
    if action not in _FIAT_COMPLIANCE_ACTIONS:
        return (
            f"action must be one of "
            f"{sorted(_FIAT_COMPLIANCE_ACTIONS)}; "
            f"got {action!r}."
        )

    if action == "summary":
        try:
            result = await _call_node_api(
                "GET", "/admin/fiat-compliance/summary",
            )
        except Exception as e:
            return (
                f"prsm_fiat_compliance failed: {e}\n"
                f"Is your PRSM node running? (prsm node start)"
            )
        if "by_kind" not in result:
            detail = result.get("detail", "unknown error")
            if "not initialized" in str(detail).lower():
                return (
                    f"Fiat compliance ring not wired.\n"
                    f"  Detail: {detail}\n"
                    f"  Set PRSM_FIAT_COMPLIANCE_LOG_DIR to "
                    f"enable persistence."
                )
            return f"summary refused: {detail}"
        by_kind = result.get("by_kind") or {}
        total = result.get("total_entries", 0)
        lines = [
            f"Fiat Compliance — {total} total entries:",
        ]
        if not by_kind:
            lines.append("  (no events recorded yet)")
        for kind in sorted(by_kind.keys()):
            bucket = by_kind[kind]
            lines.append(
                f"  {kind:>28}  count={bucket.get('count', 0):>6}"
                f"  total_usd=${bucket.get('total_usd', 0):,.2f}"
            )
        return "\n".join(lines)

    if action == "lookup":
        entry_id = (arguments.get("entry_id") or "").strip()
        if not entry_id:
            return "lookup requires 'entry_id'."
        try:
            result = await _call_node_api(
                "GET", f"/admin/fiat-compliance/{entry_id}",
            )
        except Exception as e:
            return (
                f"prsm_fiat_compliance failed: {e}\n"
                f"Is your PRSM node running? (prsm node start)"
            )
        if "entry_id" not in result:
            detail = result.get("detail", "unknown error")
            if "no entry with id" in str(detail).lower():
                return f"No entry with id={entry_id!r}."
            if "not initialized" in str(detail).lower():
                return (
                    f"Fiat compliance ring not wired.\n"
                    f"  Detail: {detail}"
                )
            return f"lookup refused: {detail}"
        return "\n".join([
            f"Fiat Compliance Entry {result['entry_id']}:",
            f"  timestamp:    {result.get('timestamp', 0)}",
            f"  kind:         {result.get('kind', '?')}",
            f"  user_id:      {result.get('user_id', '?')}",
            f"  status:       {result.get('status', '?')}",
            f"  kyc_status:   {result.get('kyc_status', '?')}",
            f"  usd_amount:   "
            f"${result.get('usd_amount', 0):,.2f}",
            f"  ftns_amount:  "
            f"{result.get('ftns_amount', 0):.6f} FTNS",
            f"  address:      {result.get('address', '?')}",
            f"  tx_hash:      {result.get('tx_hash', '?')}",
            f"  vendor_ref:   {result.get('vendor_ref', '?')}",
            f"  jurisdiction: {result.get('jurisdiction', '?')}",
            f"  metadata:     {result.get('metadata', {})}",
        ])

    # action == "list"
    params = []
    limit = int(arguments.get("limit", 100))
    offset = int(arguments.get("offset", 0))
    params.append(f"limit={limit}")
    params.append(f"offset={offset}")
    kind = (arguments.get("kind") or "").strip()
    if kind:
        params.append(f"kind={kind}")
    user_id = (arguments.get("user_id") or "").strip()
    if user_id:
        params.append(f"user_id={user_id}")
    path = "/admin/fiat-compliance?" + "&".join(params)
    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"prsm_fiat_compliance failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "entries" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in str(detail).lower():
            return (
                f"Fiat compliance ring not wired.\n"
                f"  Detail: {detail}"
            )
        return f"list refused: {detail}"
    entries = result.get("entries") or []
    total = result.get("count", 0)
    lines = [
        f"Fiat Compliance Log — {len(entries)} of {total} "
        f"(newest first):",
    ]
    if not entries:
        lines.append("  (no events recorded yet)")
    for e in entries:
        eid = (e.get("entry_id") or "?")[:8]
        usd = e.get("usd_amount", 0)
        ftns = e.get("ftns_amount", 0)
        lines.append(
            f"  {eid}…  {e.get('kind', '?'):>28}  "
            f"user={e.get('user_id', '-') or '-':<12}  "
            f"${usd:,.2f}  {ftns:.4f} FTNS  "
            f"status={e.get('status', '?')}"
        )
    return "\n".join(lines)


_KYC_ACTIONS = {"initiate", "lookup", "list", "status"}


async def handle_prsm_kyc(arguments: Dict[str, Any]) -> str:
    """Sprint 280 — KYC vendor adapter inspection + initiation.

    LLM-facing surface. action selector: initiate | lookup |
    list | status. Webhook handler intentionally absent from
    MCP — vendor → operator callbacks go through the HTTP
    surface directly."""
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return (
            f"Missing required 'action' (must be one of "
            f"{sorted(_KYC_ACTIONS)})."
        )
    if action not in _KYC_ACTIONS:
        return (
            f"action must be one of {sorted(_KYC_ACTIONS)}; "
            f"got {action!r}."
        )

    if action == "status":
        try:
            result = await _call_node_api(
                "GET", "/wallet/kyc/status",
            )
        except Exception as e:
            return (
                f"prsm_kyc failed: {e}\n"
                f"Is your PRSM node running? (prsm node start)"
            )
        if "commissioned" not in result:
            detail = result.get("detail", "unknown error")
            if "not initialized" in str(detail).lower():
                return (
                    f"KYC client not wired.\n  Detail: {detail}"
                )
            return f"status refused: {detail}"
        commissioned = result.get("commissioned")
        vendor = result.get("vendor") or "(not set)"
        supported = result.get("supported_vendors") or []
        if commissioned:
            return "\n".join([
                "PRSM KYC — commissioned",
                f"  vendor:            {vendor}",
                f"  record_count:      "
                f"{result.get('record_count', 0)}",
                f"  supported_vendors: {', '.join(supported)}",
            ])
        return "\n".join([
            "PRSM KYC — PENDING_COMMISSION",
            f"  vendor:            {vendor}",
            f"  record_count:      "
            f"{result.get('record_count', 0)}",
            f"  supported_vendors: {', '.join(supported)}",
            "  Set KYC_VENDOR + KYC_VENDOR_API_KEY to "
            "commission.",
        ])

    if action == "initiate":
        user_id = (arguments.get("user_id") or "").strip()
        if not user_id:
            return "initiate requires 'user_id'."
        email = (arguments.get("email") or "").strip()
        if not email:
            return "initiate requires 'email'."
        level = (arguments.get("level") or "basic").strip()
        try:
            result = await _call_node_api(
                "POST", "/wallet/kyc/initiate",
                {"user_id": user_id, "email": email,
                 "level": level},
            )
        except Exception as e:
            return (
                f"prsm_kyc failed: {e}\n"
                f"Is your PRSM node running? (prsm node start)"
            )
        if "status" not in result:
            detail = result.get("detail", "unknown error")
            d_lower = str(detail).lower()
            if "not initialized" in d_lower:
                return (
                    f"KYC client not wired.\n  Detail: {detail}"
                )
            return f"initiate refused: {detail}"
        status = result.get("status", "?")
        vendor = result.get("vendor") or "(not set)"
        if status == "PENDING_COMMISSION":
            return (
                f"KYC initiation preview for user_id="
                f"{user_id!s} (status=PENDING_COMMISSION):\n"
                f"  email:    {email}\n"
                f"  level:    {level}\n"
                f"  vendor:   {vendor}\n"
                f"  Set KYC_VENDOR + KYC_VENDOR_API_KEY to "
                f"enable real vendor session creation."
            )
        return (
            f"KYC session {status} for user_id={user_id!s}:\n"
            f"  email:       {email}\n"
            f"  level:       {result.get('level', level)}\n"
            f"  vendor:      {vendor}\n"
            f"  vendor_ref:  {result.get('vendor_ref', '?')}\n"
            f"  session_url: "
            f"{result.get('session_url', '?')}"
        )

    if action == "lookup":
        user_id = (arguments.get("user_id") or "").strip()
        if not user_id:
            return "lookup requires 'user_id'."
        try:
            result = await _call_node_api(
                "GET", f"/wallet/kyc/{user_id}",
            )
        except Exception as e:
            return (
                f"prsm_kyc failed: {e}\n"
                f"Is your PRSM node running? (prsm node start)"
            )
        if "user_id" not in result:
            detail = result.get("detail", "unknown error")
            if "no kyc record" in str(detail).lower():
                return f"No KYC record for user_id={user_id!r}."
            if "not initialized" in str(detail).lower():
                return (
                    f"KYC client not wired.\n  Detail: {detail}"
                )
            return f"lookup refused: {detail}"
        return "\n".join([
            f"KYC record user_id={result['user_id']!s}:",
            f"  status:      {result.get('status', '?')}",
            f"  level:       {result.get('level', '?')}",
            f"  vendor:      {result.get('vendor', '?')}",
            f"  vendor_ref:  {result.get('vendor_ref', '?')}",
            f"  session_url: {result.get('session_url', '?')}",
            f"  email:       {result.get('email', '?')}",
            f"  created_at:  {result.get('created_at', 0)}",
            f"  verified_at: {result.get('verified_at', 0)}",
        ])

    # action == "list"
    limit = int(arguments.get("limit", 100))
    try:
        result = await _call_node_api(
            "GET", f"/wallet/kyc?limit={limit}",
        )
    except Exception as e:
        return (
            f"prsm_kyc failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "records" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in str(detail).lower():
            return f"KYC client not wired.\n  Detail: {detail}"
        return f"list refused: {detail}"
    records = result.get("records") or []
    total = result.get("count", 0)
    lines = [
        f"PRSM KYC Records — {len(records)} of {total} "
        f"(newest first):",
    ]
    if not records:
        lines.append("  (none)")
    for r in records:
        lines.append(
            f"  user_id={r.get('user_id', '?')}  "
            f"status={r.get('status', '?'):>20}  "
            f"level={r.get('level', '?')}  "
            f"vendor={r.get('vendor', '?')}  "
            f"email={r.get('email', '?')}"
        )
    return "\n".join(lines)


_POOL_QUOTE_ACTIONS = {"state", "quote"}


async def handle_prsm_pool_quote(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 279 — Aerodrome USDC-FTNS pool inspection.

    Read-only; no commission gate. action=state returns the
    pool's live reserves; action=quote computes an exact-amount-
    in swap quote. Pre-seeding-ceremony, NOT_CONFIGURED is
    surfaced so operators see the plumbing is wired and just
    waiting on the pool address."""
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return (
            f"Missing required 'action' (must be one of "
            f"{sorted(_POOL_QUOTE_ACTIONS)})."
        )
    if action not in _POOL_QUOTE_ACTIONS:
        return (
            f"action must be one of "
            f"{sorted(_POOL_QUOTE_ACTIONS)}; got {action!r}."
        )

    if action == "state":
        try:
            result = await _call_node_api(
                "GET", "/wallet/pool/state",
            )
        except Exception as e:
            return (
                f"prsm_pool_quote failed: {e}\n"
                f"Is your PRSM node running? (prsm node start)"
            )
        status = result.get("status")
        if status is None:
            detail = result.get("detail", "unknown error")
            if "not initialized" in str(detail).lower():
                return (
                    f"Aerodrome client not wired on this "
                    f"node.\n  Detail: {detail}"
                )
            return f"state refused: {detail}"
        if status == "NOT_CONFIGURED":
            return (
                f"Aerodrome USDC-FTNS pool — NOT_CONFIGURED\n"
                f"  {result.get('note', '')}"
            )
        if status == "POOL_UNAVAILABLE":
            return (
                f"Aerodrome USDC-FTNS pool — POOL_UNAVAILABLE\n"
                f"  pool_address: "
                f"{result.get('pool_address', '?')}\n"
                f"  {result.get('note', '')}"
            )
        return "\n".join([
            f"Aerodrome USDC-FTNS pool state (OK):",
            f"  pool_address:  {result.get('pool_address', '?')}",
            f"  token0:        {result.get('token0', '?')}",
            f"  token1:        {result.get('token1', '?')}",
            f"  reserve0:      {result.get('reserve0', 0)}",
            f"  reserve1:      {result.get('reserve1', 0)}",
            f"  total_supply:  {result.get('total_supply', 0)}",
            f"  stable:        {result.get('stable', False)}",
            f"  fee_bps:       {result.get('fee_bps', 0)}",
            f"  block_number:  {result.get('block_number', 0)}",
        ])

    # action == "quote"
    amount_in = arguments.get("amount_in")
    if amount_in is None:
        return "quote requires 'amount_in' (positive integer)."
    token_in = (arguments.get("token_in") or "").strip()
    if not token_in:
        return "quote requires 'token_in' (token address)."
    try:
        amount_int = int(amount_in)
    except (ValueError, TypeError):
        return f"amount_in must be an integer, got {amount_in!r}."
    if amount_int <= 0:
        return f"amount_in must be > 0, got {amount_int}."

    try:
        path = (
            f"/wallet/pool/quote?amount_in={amount_int}"
            f"&token_in={token_in}"
        )
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"prsm_pool_quote failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    status = result.get("status")
    if status is None:
        detail = result.get("detail", "unknown error")
        d_lower = str(detail).lower()
        if "not in pool" in d_lower:
            return (
                f"Token not in the USDC-FTNS pool.\n"
                f"  Detail: {detail}"
            )
        if "stable" in d_lower:
            return (
                f"Stable-pool curve not supported in v1 "
                f"quoter; volatile pools only.\n"
                f"  Detail: {detail}"
            )
        if "not initialized" in d_lower:
            return (
                f"Aerodrome client not wired.\n"
                f"  Detail: {detail}"
            )
        return f"quote refused: {detail}"
    if status == "NOT_CONFIGURED":
        return (
            f"Aerodrome pool NOT_CONFIGURED — quote unavailable "
            f"until seeding ceremony completes."
        )
    if status == "POOL_UNAVAILABLE":
        return (
            f"Aerodrome pool POOL_UNAVAILABLE — RPC may be "
            f"down or pool contract missing."
        )
    return "\n".join([
        f"Aerodrome quote (OK):",
        f"  amount_in:        {result.get('amount_in', 0)} "
        f"{result.get('token_in', '?')}",
        f"  amount_out:       {result.get('amount_out', 0)} "
        f"{result.get('token_out', '?')}",
        f"  price_impact_bps: "
        f"{result.get('price_impact_bps', 0)} "
        f"(slippage; excludes fee)",
        f"  fee_bps:          {result.get('fee_bps', 0)}",
        f"  route:            {result.get('route', '?')}",
    ])


_GASLESS_TRANSFER_ACTIONS = {"quote", "execute", "status"}


async def handle_prsm_gasless_transfer(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 277 — gasless FTNS transfer via Coinbase paymaster.

    Per Vision §14 "Crypto-UX adoption barrier" mitigation, this
    is the LLM-facing surface that makes FTNS feel like normal
    money: user says "send 10 FTNS to Bob," LLM calls quote (or
    execute), user never sees gas. PENDING_COMMISSION until
    paymaster env keys are set."""
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return (
            f"Missing required 'action' (must be one of "
            f"{sorted(_GASLESS_TRANSFER_ACTIONS)})."
        )
    if action not in _GASLESS_TRANSFER_ACTIONS:
        return (
            f"action must be one of "
            f"{sorted(_GASLESS_TRANSFER_ACTIONS)}; got {action!r}."
        )

    if action == "status":
        try:
            result = await _call_node_api(
                "GET", "/wallet/paymaster/status",
            )
        except Exception as e:
            return (
                f"prsm_gasless_transfer failed: {e}\n"
                f"Is your PRSM node running? (prsm node start)"
            )
        if "commissioned" not in result:
            detail = result.get("detail", "unknown error")
            if "not initialized" in str(detail).lower():
                return (
                    f"Paymaster client not wired.\n"
                    f"  Detail: {detail}"
                )
            return f"status refused: {detail}"
        commissioned = result.get("commissioned")
        lines = [
            f"PRSM Paymaster — "
            f"{'commissioned' if commissioned else 'PENDING_COMMISSION'}",
            f"  endpoint:            "
            f"{result.get('endpoint') or '(not set)'}",
            f"  policy_id:           "
            f"{result.get('policy_id') or '(not set)'}",
            f"  sponsorships:        {result.get('sponsorships', 0)}",
            f"  total_sponsored_wei: "
            f"{result.get('total_sponsored_wei', 0)}",
        ]
        if not commissioned:
            lines.append(
                "  Set COINBASE_CDP_PAYMASTER_ENDPOINT + "
                "COINBASE_CDP_PAYMASTER_API_KEY to commission."
            )
        return "\n".join(lines)

    # quote + execute share field validation
    from_user_id = (arguments.get("from_user_id") or "").strip()
    if not from_user_id:
        return f"{action} requires 'from_user_id'."
    to_address = (arguments.get("to_address") or "").strip()
    if not to_address:
        return f"{action} requires 'to_address'."
    ftns_amount = (arguments.get("ftns_amount") or "").strip()
    if not ftns_amount:
        return f"{action} requires 'ftns_amount' (decimal string)."

    dry_run = action == "quote"  # quote → dry_run; execute → submit
    try:
        result = await _call_node_api(
            "POST", "/wallet/transfer/gasless",
            {
                "from_user_id": from_user_id,
                "to_address": to_address,
                "ftns_amount": ftns_amount,
                "dry_run": dry_run,
            },
        )
    except Exception as e:
        return (
            f"prsm_gasless_transfer failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "status" not in result:
        detail = result.get("detail", "unknown error")
        d_lower = str(detail).lower()
        if "no waas wallet" in d_lower:
            return (
                f"No WaaS wallet for from_user_id="
                f"{from_user_id!r}. Provision one first with "
                f"prsm_waas_wallet?action=provision."
            )
        if "not initialized" in d_lower:
            return (
                f"Required client not wired.\n"
                f"  Detail: {detail}"
            )
        return f"{action} refused: {detail}"

    status = result.get("status", "?")
    base = (
        f"Gasless transfer — "
        f"from={result.get('from_user_id', '?')} → "
        f"to={result.get('to_address', '?')}  "
        f"amount={result.get('ftns_amount', '?')} FTNS"
    )
    if status == "PENDING_COMMISSION":
        return (
            f"{base}\n"
            f"  status: PENDING_COMMISSION\n"
            f"  Paymaster not commissioned yet — this is a "
            f"preview. Set COINBASE_CDP_PAYMASTER_ENDPOINT + "
            f"COINBASE_CDP_PAYMASTER_API_KEY to enable real "
            f"sponsored submission."
        )
    if status == "ESTIMATED":
        return (
            f"{base}\n"
            f"  status:           ESTIMATED (dry_run)\n"
            f"  sender_address:   "
            f"{result.get('sender_address', '?')}\n"
            f"  gas_estimate_wei: "
            f"{result.get('gas_estimate_wei', 0)}\n"
            f"  Re-run with action=execute to submit."
        )
    if status == "SUBMITTED":
        return (
            f"{base}\n"
            f"  status:             SUBMITTED ✅\n"
            f"  tx_hash:            {result.get('tx_hash', '?')}\n"
            f"  user_op_hash:       "
            f"{result.get('user_op_hash', '?')}\n"
            f"  sponsor_amount_wei: "
            f"{result.get('sponsor_amount_wei', 0)}"
        )
    if status == "FAILED":
        return (
            f"{base}\n"
            f"  status: FAILED ⚠\n"
            f"  error:  {result.get('error', '?')}"
        )
    return (
        f"{base}\n  status: {status}\n  {result!s}"
    )


_WAAS_WALLET_ACTIONS = {
    "provision", "lookup", "list", "status",
}


async def handle_prsm_waas_wallet(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 276 — Coinbase WaaS wallet provisioning + inspection.

    Per Vision §14 "Crypto-UX adoption barrier" mitigation, this
    is the LLM-facing surface that makes wallet provisioning
    invisible: user says "give me a wallet," LLM calls this with
    action=provision, user gets back an address (or a
    PENDING_COMMISSION preview until Coinbase CDP commissions).
    """
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return (
            f"Missing required 'action' (must be one of "
            f"{sorted(_WAAS_WALLET_ACTIONS)})."
        )
    if action not in _WAAS_WALLET_ACTIONS:
        return (
            f"action must be one of "
            f"{sorted(_WAAS_WALLET_ACTIONS)}; got {action!r}."
        )

    if action == "provision":
        user_id = (arguments.get("user_id") or "").strip()
        if not user_id:
            return "provision requires 'user_id'."
        email = (arguments.get("email") or "").strip()
        if not email:
            return "provision requires 'email'."
        try:
            result = await _call_node_api(
                "POST", "/wallet/waas/provision",
                {"user_id": user_id, "email": email},
            )
        except Exception as e:
            return (
                f"prsm_waas_wallet failed: {e}\n"
                f"Is your PRSM node running? (prsm node start)"
            )
        if "user_id" not in result:
            detail = result.get("detail", "unknown error")
            if "not initialized" in str(detail).lower():
                return (
                    f"WaaS client not wired on this node.\n"
                    f"  Detail: {detail}\n"
                    f"  Set COINBASE_CDP_API_KEY_NAME + "
                    f"COINBASE_CDP_API_KEY_PRIVATE to commission."
                )
            return f"provision refused: {detail}"
        status = result.get("status", "?")
        if status == "PENDING_COMMISSION":
            return (
                f"Wallet preview for user_id={result['user_id']!s} "
                f"(status=PENDING_COMMISSION):\n"
                f"  email:   {result.get('email', '?')}\n"
                f"  network: {result.get('network', '?')}\n"
                f"  Note: Coinbase CDP not commissioned yet — "
                f"this is a preview record. Real provisioning "
                f"lands when COINBASE_CDP_API_KEY_NAME + "
                f"COINBASE_CDP_API_KEY_PRIVATE are configured."
            )
        return (
            f"Wallet PROVISIONED for user_id="
            f"{result['user_id']!s}:\n"
            f"  email:     {result.get('email', '?')}\n"
            f"  wallet_id: {result.get('wallet_id', '?')}\n"
            f"  address:   {result.get('address', '?')}\n"
            f"  network:   {result.get('network', '?')}"
        )

    if action == "lookup":
        user_id = (arguments.get("user_id") or "").strip()
        if not user_id:
            return "lookup requires 'user_id'."
        try:
            result = await _call_node_api(
                "GET", f"/wallet/waas/{user_id}",
            )
        except Exception as e:
            return (
                f"prsm_waas_wallet failed: {e}\n"
                f"Is your PRSM node running? (prsm node start)"
            )
        if "user_id" not in result:
            detail = result.get("detail", "unknown error")
            if "no wallet for" in str(detail).lower():
                return (
                    f"No wallet exists for user_id="
                    f"{user_id!r}. Run action=provision first."
                )
            if "not initialized" in str(detail).lower():
                return (
                    f"WaaS client not wired.\n  Detail: {detail}"
                )
            return f"lookup refused: {detail}"
        return (
            f"WaaS wallet user_id={result['user_id']!s}:\n"
            f"  status:    {result.get('status', '?')}\n"
            f"  email:     {result.get('email', '?')}\n"
            f"  wallet_id: {result.get('wallet_id', '?')}\n"
            f"  address:   {result.get('address', '?')}\n"
            f"  network:   {result.get('network', '?')}\n"
            f"  created:   {result.get('created_at', 0)}"
        )

    if action == "list":
        limit = int(arguments.get("limit", 100))
        try:
            result = await _call_node_api(
                "GET", f"/wallet/waas?limit={limit}",
            )
        except Exception as e:
            return (
                f"prsm_waas_wallet failed: {e}\n"
                f"Is your PRSM node running? (prsm node start)"
            )
        if "wallets" not in result:
            detail = result.get("detail", "unknown error")
            if "not initialized" in str(detail).lower():
                return (
                    f"WaaS client not wired.\n  Detail: {detail}"
                )
            return f"list refused: {detail}"
        wallets = result.get("wallets") or []
        total = result.get("count", 0)
        lines = [
            f"PRSM WaaS Wallets — {len(wallets)} of {total} "
            f"(newest first):",
        ]
        if not wallets:
            lines.append("  (none)")
        for w in wallets:
            lines.append(
                f"  user_id={w.get('user_id', '?')}  "
                f"status={w.get('status', '?'):>20}  "
                f"address={w.get('address') or '(pending)'}  "
                f"email={w.get('email', '?')}"
            )
        return "\n".join(lines)

    # action == "status"
    try:
        result = await _call_node_api("GET", "/wallet/waas/status")
    except Exception as e:
        return (
            f"prsm_waas_wallet failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "commissioned" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in str(detail).lower():
            return (
                f"WaaS client not wired.\n  Detail: {detail}"
            )
        return f"status refused: {detail}"
    commissioned = result.get("commissioned")
    if commissioned:
        return (
            f"WaaS commissioned=True\n"
            f"  network:      {result.get('network', '?')}\n"
            f"  wallet_count: {result.get('wallet_count', 0)}"
        )
    return (
        f"WaaS commissioned=False (PENDING_COMMISSION)\n"
        f"  network:      {result.get('network', '?')}\n"
        f"  wallet_count: {result.get('wallet_count', 0)}\n"
        f"  Set COINBASE_CDP_API_KEY_NAME + "
        f"COINBASE_CDP_API_KEY_PRIVATE to enable real "
        f"provisioning."
    )


_INSURANCE_FUND_ACTIONS = {
    "status", "compose_recovery",
}


async def handle_prsm_insurance_fund(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 299 — insurance fund tracker (Vision §14
    mitigation item 2).

    action selector:
      status            — bulk reserve ratio + balances
      compose_recovery  — Safe-uploadable ERC-20 transfer
                          payload for exploit-recovery
                          disbursement (composer-only;
                          Foundation Safe multisig gates
                          execution)
    """
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return (
            f"Missing required 'action' (must be one of "
            f"{sorted(_INSURANCE_FUND_ACTIONS)})."
        )
    if action not in _INSURANCE_FUND_ACTIONS:
        return (
            f"action must be one of "
            f"{sorted(_INSURANCE_FUND_ACTIONS)}; "
            f"got {action!r}."
        )

    if action == "status":
        try:
            result = await _call_node_api(
                "GET", "/admin/insurance-fund/status",
            )
        except Exception as e:
            return (
                f"prsm_insurance_fund failed: {e}\n"
                f"Is your PRSM node running?"
            )
        if "target_bps" not in result:
            detail = result.get("detail", "unknown error")
            if "not initialized" in str(detail).lower():
                return (
                    f"Insurance fund tracker not wired.\n"
                    f"  Detail: {detail}"
                )
            return f"status refused: {detail}"
        if not result.get("commissioned"):
            return (
                "PRSM Insurance Fund — not configured\n"
                "  Set PRSM_INSURANCE_FUND_ADDRESS to a "
                "designated insurance-fund wallet to enable "
                "monitoring per Vision §14 mitigation item 2."
            )
        ratio_bps = result.get("reserve_ratio_bps")
        target_bps = result.get("target_bps", 500)
        target_met = result.get("target_met", False)
        marker = "✅ at/above target" if target_met else "⚠ BELOW target"
        ratio_pct = (
            f"{ratio_bps / 100:.2f}%"
            if ratio_bps is not None else "unknown"
        )
        target_pct = f"{target_bps / 100:.2f}%"
        # Render FTNS amounts as whole tokens (wei / 1e18)
        def _fmt_wei(w):
            if w is None:
                return "unknown"
            return f"{w / (10 ** 18):,.2f} FTNS"
        lines = [
            f"PRSM Insurance Fund — {marker}",
            "",
            f"  Reserve ratio:   {ratio_pct}  "
            f"(target: {target_pct})",
            f"  Fund balance:    "
            f"{_fmt_wei(result.get('fund_balance_wei'))}",
            f"  Treasury total:  "
            f"{_fmt_wei(result.get('treasury_balance_wei'))}",
            f"  Fund address:    "
            f"{result.get('fund_address', '?')}",
            f"  Treasury addr:   "
            f"{result.get('treasury_address', '?')}",
        ]
        if result.get("error"):
            lines.append(f"  ⚠ RPC error: {result['error']}")
        if not target_met and ratio_bps is not None:
            shortfall_bps = max(0, target_bps - ratio_bps)
            lines.append(
                f"  Shortfall:       "
                f"{shortfall_bps / 100:.2f}% below target — "
                f"top up to restore §14 promise."
            )
        return "\n".join(lines)

    # compose_recovery
    recipient = (arguments.get("recipient") or "").strip()
    if not recipient:
        return "compose_recovery requires 'recipient'."
    amount_wei = arguments.get("amount_wei")
    if amount_wei is None:
        return "compose_recovery requires 'amount_wei'."
    try:
        amount_int = int(amount_wei)
    except (ValueError, TypeError):
        return (
            f"amount_wei must be an integer, "
            f"got {amount_wei!r}."
        )
    if amount_int <= 0:
        return "amount_wei must be > 0."
    reason = (arguments.get("reason") or "").strip()
    if not reason:
        return (
            "compose_recovery requires 'reason' "
            "(short statement of recovery rationale for "
            "audit trail)."
        )
    try:
        result = await _call_node_api(
            "POST", "/admin/insurance-fund/compose-recovery",
            {
                "recipient": recipient,
                "amount_wei": amount_int,
                "reason": reason,
            },
        )
    except Exception as e:
        return f"prsm_insurance_fund failed: {e}"
    if "data" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in str(detail).lower():
            return (
                f"Insurance fund tracker not wired.\n"
                f"  Detail: {detail}"
            )
        return f"compose refused: {detail}"

    explorer = result.get("explorer_url") or "(no explorer)"
    amount_ftns = (
        f"{int(result.get('amount_wei', '0')) / (10 ** 18):,.2f}"
    )
    lines = [
        "⚠ INSURANCE FUND RECOVERY TRANSFER COMPOSED — "
        "Foundation Safe upload required ⚠",
        "",
        f"  WARNING: {result.get('warning', '')}",
        "",
        "  Transaction payload (paste into Safe UI):",
        f"    to:        {result.get('to', '?')}",
        f"    data:      {result.get('data', '?')}",
        f"    value:     {result.get('value', '0')}",
        f"    chain_id:  {result.get('chain_id', '?')}",
        "",
        f"  Recipient:   {result.get('recipient', '?')}",
        f"  Amount:      {amount_ftns} FTNS",
        f"  From fund:   {result.get('from_fund', '?')}",
        f"  Reason:      {result.get('reason', '?')}",
        f"  Verify on:   {explorer}",
        "",
        "  Instructions:",
        f"    {result.get('instructions', '')}",
    ]
    return "\n".join(lines)


_EMERGENCY_PAUSE_ACTIONS = {
    "status", "compose_pause", "compose_unpause",
}


_DISCLOSURE_ACTIONS = {
    "submit", "list", "lookup", "update",
    "compose_payout", "record_payout_tx",
}


_INCIDENT_ACTIONS = {
    "open", "list", "lookup", "advance", "event",
    "recommend", "comms", "playbook",
}


_FORMAL_VERIFICATION_ACTIONS = {
    "list", "check", "check_one",
    "symbolic_list", "symbolic_check",
}


_UPGRADE_ACTIONS = {
    "propose", "list", "lookup", "update",
    "compose_upgrade", "compose_rollback",
}


_ENTERPRISE_RECIPIENT_ACTIONS = {
    "keypair_gen", "encrypt", "decrypt", "get_manifest",
    # Sprint 307 — threshold mode
    "encrypt_threshold", "unseal_share", "combine_decrypt",
}


_TEE_POLICY_ACTIONS = {
    "evaluate", "node_status", "list_tiers",
}


_CORP_CAPABILITY_ACTIONS = {
    "keypair_gen", "register_issuer", "list_issuers",
    "redeem", "get_ledger", "get_consumed",
}


_FEDERATED_ACTIONS = {
    "propose", "list", "lookup", "issue_round", "aggregate",
    # Sprint 308a — hardening
    "register_worker_key", "list_worker_keys",
}


_PIPELINE_INFERENCE_ACTIONS = {
    "propose", "list", "lookup", "execute", "get_round",
}


async def handle_prsm_pipeline_inference(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 312 — pipeline inference orchestrator MCP
    wrapper. Wraps /admin/inference/pipeline/*."""
    action = (
        arguments.get("action") or ""
    ).strip().lower()
    if not action:
        return (
            f"Missing required 'action' (must be one of "
            f"{sorted(_PIPELINE_INFERENCE_ACTIONS)})."
        )
    if action not in _PIPELINE_INFERENCE_ACTIONS:
        return (
            f"action must be one of "
            f"{sorted(_PIPELINE_INFERENCE_ACTIONS)}; got "
            f"{action!r}."
        )

    if action == "propose":
        required = ("model_id", "total_layers", "node_ids")
        missing = [
            f for f in required
            if arguments.get(f) in (None, "", [])
        ]
        if missing:
            return (
                f"propose missing required field(s): "
                f"{missing}"
            )
        # Build an even partition by convention; operators
        # who want custom layer splits hit the HTTP
        # endpoint directly with their own partition.
        from prsm.compute.inference.pipeline_partition import (
            even_layer_partition,
        )
        try:
            partition = even_layer_partition(
                total_layers=int(
                    arguments["total_layers"],
                ),
                node_ids=list(arguments["node_ids"]),
            )
        except ValueError as e:
            return f"propose refused: {e}"
        body = {
            "model_id": arguments["model_id"],
            "partition": partition.to_dict(),
        }
        try:
            r = await _call_node_api(
                "POST", "/admin/inference/pipeline/job",
                body,
            )
        except Exception as e:
            return (
                f"prsm_pipeline_inference propose failed: "
                f"{e}"
            )
        if "job_id" not in r:
            detail = r.get("detail", "unknown error")
            return f"propose refused: {detail}"
        return (
            f"Pipeline job proposed\n"
            f"  job_id:        {r.get('job_id')}\n"
            f"  model_id:      {r.get('model_id')}\n"
            f"  status:        {r.get('status')}\n"
            f"  n_stages:      "
            f"{len((r.get('partition') or {}).get('stage_layer_ranges', []))}\n"
            f"  total_layers:  "
            f"{(r.get('partition') or {}).get('total_layers')}"
        )

    if action == "list":
        try:
            r = await _call_node_api(
                "GET", "/admin/inference/pipeline/job",
            )
        except Exception as e:
            return (
                f"prsm_pipeline_inference list failed: {e}"
            )
        if "jobs" not in r:
            detail = r.get("detail", "unknown error")
            return f"list refused: {detail}"
        jobs = r.get("jobs") or []
        if not jobs:
            return "No pipeline jobs."
        lines = [
            f"PRSM Pipeline Inference Jobs — {len(jobs)}:",
            "",
            f"  {'id':<10} {'status':<11} {'stages':<7}  "
            f"model_id",
        ]
        for j in jobs:
            jid = (j.get("job_id") or "")[:8]
            part = j.get("partition") or {}
            stages = len(
                part.get("stage_layer_ranges") or [],
            )
            lines.append(
                f"  {jid:<10} "
                f"{j.get('status', '?'):<11} "
                f"{stages:<7}  "
                f"{j.get('model_id', '?')}"
            )
        return "\n".join(lines)

    if action == "lookup":
        jid = (arguments.get("job_id") or "").strip()
        if not jid:
            return "lookup requires 'job_id'."
        try:
            r = await _call_node_api(
                "GET",
                f"/admin/inference/pipeline/job/{jid}",
            )
        except Exception as e:
            return (
                f"prsm_pipeline_inference lookup failed: "
                f"{e}"
            )
        if "job_id" not in r:
            detail = r.get("detail", "unknown error")
            return f"lookup refused: {detail}"
        part = r.get("partition") or {}
        return (
            f"Pipeline job {r.get('job_id')}\n"
            f"  model_id:      {r.get('model_id')}\n"
            f"  status:        {r.get('status')}\n"
            f"  total_layers:  {part.get('total_layers')}\n"
            f"  stages:        "
            f"{len(part.get('stage_layer_ranges') or [])}\n"
            f"  nodes:         "
            f"{', '.join(part.get('stage_node_ids') or [])}"
        )

    if action == "execute":
        jid = (arguments.get("job_id") or "").strip()
        if not jid:
            return "execute requires 'job_id'."
        prompt_b64 = arguments.get("prompt_b64") or ""
        if not prompt_b64:
            return "execute requires 'prompt_b64'."
        try:
            r = await _call_node_api(
                "POST",
                f"/admin/inference/pipeline/job/{jid}"
                f"/execute",
                {"prompt_b64": prompt_b64},
            )
        except Exception as e:
            return (
                f"prsm_pipeline_inference execute failed: "
                f"{e}"
            )
        if "status" not in r:
            detail = r.get("detail", "unknown error")
            return f"execute refused: {detail}"
        receipt = r.get("receipt") or {}
        return (
            f"Pipeline execution {r.get('status')}\n"
            f"  job_id:           "
            f"{r.get('job_id', '?')}\n"
            f"  round_id:         "
            f"{r.get('round_id', '?')}\n"
            f"  prompt_hash:      "
            f"{(receipt.get('prompt_hash') or '')[:24]}\n"
            f"  output_hash:      "
            f"{(receipt.get('output_hash') or '')[:24]}\n"
            f"  partition_hash:   "
            f"{(receipt.get('partition_hash') or '')[:24]}\n"
            f"  stages_recorded:  "
            f"{len(receipt.get('stage_receipts') or [])}\n"
            f"  signature:        "
            f"{(receipt.get('orchestrator_signature_b64') or '')[:24]}"
        )

    # get_round
    jid = (arguments.get("job_id") or "").strip()
    if not jid:
        return "get_round requires 'job_id'."
    try:
        r = await _call_node_api(
            "GET",
            f"/admin/inference/pipeline/job/{jid}/round",
        )
    except Exception as e:
        return (
            f"prsm_pipeline_inference get_round failed: {e}"
        )
    if "status" not in r:
        detail = r.get("detail", "unknown error")
        return f"get_round refused: {detail}"
    receipt = r.get("receipt") or {}
    return (
        f"Pipeline round\n"
        f"  job_id:       {r.get('job_id')}\n"
        f"  round_id:     {r.get('round_id')}\n"
        f"  status:       {r.get('status')}\n"
        f"  error:        {r.get('error') or '(none)'}\n"
        f"  receipt:      "
        f"{'present' if receipt else '(none)'}"
    )


async def handle_prsm_federated_train(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 308b — worker-side training shim. Triggers
    /compute/train on this worker; the caller then submits
    the returned signed update via prsm_federated_learning
    (action=update — TODO 308c wiring)."""
    required = (
        "job_id", "round_index", "dataset_cid",
        "sample_count",
    )
    missing = [
        f for f in required
        if arguments.get(f) in (None, "")
    ]
    if missing:
        return (
            f"prsm_federated_train missing required "
            f"field(s): {missing}"
        )
    body = {
        "job_id": arguments["job_id"],
        "round_index": int(arguments["round_index"]),
        "dataset_cid": arguments["dataset_cid"],
        "sample_count": int(arguments["sample_count"]),
    }
    try:
        r = await _call_node_api(
            "POST", "/compute/train", body,
        )
    except Exception as e:
        return f"prsm_federated_train failed: {e}"
    if "worker_signature_b64" not in r:
        detail = r.get("detail", "unknown error")
        return f"train refused: {detail}"
    return (
        f"Trained + signed gradient update produced\n"
        f"  job_id:        {r.get('job_id')}\n"
        f"  round_index:   {r.get('round_index')}\n"
        f"  worker:        {r.get('worker_node_id')}\n"
        f"  sample_count:  {r.get('sample_count')}\n"
        f"  attestation:   "
        f"{'yes' if r.get('worker_attestation_b64') else 'no'}\n"
        f"  signature:     "
        f"{(r.get('worker_signature_b64') or '')[:24]}...\n"
        f"\n"
        f"Submit via prsm_federated_learning (or POST the "
        f"full update to /admin/federated/job/"
        f"{r.get('job_id')}/update):\n\n"
        f"{r}"
    )


def _short_job_id(jid: str) -> str:
    return jid[:8] if len(jid) > 8 else jid


async def handle_prsm_federated_learning(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 308 — federated-learning orchestrator MCP
    wrapper (Vision §7 Enterprise Confidentiality Mode
    capstone)."""
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return (
            f"Missing required 'action' (must be one of "
            f"{sorted(_FEDERATED_ACTIONS)})."
        )
    if action not in _FEDERATED_ACTIONS:
        return (
            f"action must be one of "
            f"{sorted(_FEDERATED_ACTIONS)}; got {action!r}."
        )

    if action == "propose":
        required = (
            "model_id", "worker_pool", "rounds_target",
            "min_workers_per_round", "aggregation",
        )
        missing = [
            f for f in required
            if arguments.get(f) in (None, "", [])
        ]
        if missing:
            return (
                f"propose missing required field(s): "
                f"{missing}"
            )
        body = {
            "model_id": arguments["model_id"],
            "dataset_cids": (
                arguments.get("dataset_cids") or []
            ),
            "worker_pool": arguments["worker_pool"],
            "rounds_target": int(
                arguments["rounds_target"],
            ),
            "min_workers_per_round": int(
                arguments["min_workers_per_round"],
            ),
            "aggregation": arguments["aggregation"],
        }
        try:
            r = await _call_node_api(
                "POST", "/admin/federated/job", body,
            )
        except Exception as e:
            return (
                f"prsm_federated_learning propose failed: {e}"
            )
        if "job_id" not in r:
            detail = r.get("detail", "unknown error")
            return f"propose refused: {detail}"
        return (
            f"Federated job proposed\n"
            f"  job_id:        {r.get('job_id')}\n"
            f"  status:        {r.get('status')}\n"
            f"  model_id:      {r.get('model_id')}\n"
            f"  rounds:        "
            f"{r.get('current_round', 0)}/"
            f"{r.get('rounds_target')}\n"
            f"  aggregation:   {r.get('aggregation')}\n"
            f"  worker_pool:   "
            f"{len(r.get('worker_pool') or [])} nodes"
        )

    if action == "list":
        path = "/admin/federated/job"
        st = (
            arguments.get("status") or ""
        ).strip().lower()
        if st:
            path = f"{path}?status={st}"
        try:
            r = await _call_node_api("GET", path)
        except Exception as e:
            return (
                f"prsm_federated_learning list failed: {e}"
            )
        if "jobs" not in r:
            detail = r.get("detail", "unknown error")
            return f"list refused: {detail}"
        jobs = r.get("jobs") or []
        if not jobs:
            return "No federated jobs."
        lines = [
            f"PRSM Federated Jobs — {len(jobs)}:",
            "",
            f"  {'id':<10} {'status':<12} {'rounds':<10}  "
            f"model_id",
        ]
        for j in jobs:
            jid = _short_job_id(j.get("job_id", ""))
            rounds = (
                f"{j.get('current_round', 0)}/"
                f"{j.get('rounds_target', 0)}"
            )
            lines.append(
                f"  {jid:<10} "
                f"{j.get('status', '?'):<12} "
                f"{rounds:<10}  {j.get('model_id', '?')}"
            )
        return "\n".join(lines)

    if action == "lookup":
        jid = (arguments.get("job_id") or "").strip()
        if not jid:
            return "lookup requires 'job_id'."
        try:
            r = await _call_node_api(
                "GET", f"/admin/federated/job/{jid}",
            )
        except Exception as e:
            return (
                f"prsm_federated_learning lookup failed: {e}"
            )
        if "job_id" not in r:
            detail = r.get("detail", "unknown error")
            return f"lookup refused: {detail}"
        return (
            f"Federated job {r.get('job_id')}\n"
            f"  status:        {r.get('status')}\n"
            f"  model_id:      {r.get('model_id')}\n"
            f"  rounds:        "
            f"{r.get('current_round', 0)}/"
            f"{r.get('rounds_target')}\n"
            f"  aggregation:   {r.get('aggregation')}\n"
            f"  worker_pool:   "
            f"{len(r.get('worker_pool') or [])} nodes\n"
            f"  min_per_round: "
            f"{r.get('min_workers_per_round')}\n"
            f"  datasets:      "
            f"{len(r.get('dataset_cids') or [])}"
        )

    if action == "issue_round":
        jid = (arguments.get("job_id") or "").strip()
        if not jid:
            return "issue_round requires 'job_id'."
        try:
            r = await _call_node_api(
                "POST",
                f"/admin/federated/job/{jid}/issue-round",
            )
        except Exception as e:
            return (
                f"prsm_federated_learning issue_round "
                f"failed: {e}"
            )
        if "round_index" not in r:
            detail = r.get("detail", "unknown error")
            return f"issue_round refused: {detail}"
        assigns = r.get("worker_assignments") or []
        lines = [
            f"Round {r.get('round_index')} issued "
            f"(status={r.get('status')})",
            "",
            f"  Assignments ({len(assigns)} workers):",
        ]
        for a in assigns:
            lines.append(
                f"    · {a.get('node_id'):<20} "
                f"dataset={a.get('dataset_cid')}"
            )
        return "\n".join(lines)

    if action == "register_worker_key":
        node_id = (
            arguments.get("node_id") or ""
        ).strip()
        pub = (
            arguments.get("signing_pubkey_b64") or ""
        ).strip()
        if not node_id or not pub:
            return (
                "register_worker_key requires 'node_id' "
                "+ 'signing_pubkey_b64'."
            )
        try:
            r = await _call_node_api(
                "POST", "/admin/federated/worker-key",
                {
                    "node_id": node_id,
                    "signing_pubkey_b64": pub,
                },
            )
        except Exception as e:
            return (
                f"prsm_federated_learning "
                f"register_worker_key failed: {e}"
            )
        if "node_id" not in r:
            detail = r.get("detail", "unknown error")
            return f"register_worker_key refused: {detail}"
        return (
            f"Registered worker key\n"
            f"  node_id:    {r.get('node_id')}\n"
            f"  pubkey_b64: "
            f"{r.get('signing_pubkey_b64')[:16]}..."
        )

    if action == "list_worker_keys":
        try:
            r = await _call_node_api(
                "GET", "/admin/federated/worker-key",
            )
        except Exception as e:
            return (
                f"prsm_federated_learning "
                f"list_worker_keys failed: {e}"
            )
        if "worker_keys" not in r:
            detail = r.get("detail", "unknown error")
            return f"list_worker_keys refused: {detail}"
        keys = r.get("worker_keys") or []
        if not keys:
            return "No registered worker keys."
        lines = [
            f"PRSM Federated Worker Keys — {len(keys)}:",
            "",
        ]
        for k in keys:
            pub = k.get("signing_pubkey_b64") or ""
            lines.append(
                f"  · {k.get('node_id'):<20} "
                f"{pub[:16]}..."
            )
        return "\n".join(lines)

    # aggregate
    jid = (arguments.get("job_id") or "").strip()
    if not jid:
        return "aggregate requires 'job_id'."
    ridx = arguments.get("round_index")
    if ridx is None:
        return "aggregate requires 'round_index'."
    try:
        r = await _call_node_api(
            "POST",
            f"/admin/federated/job/{jid}/aggregate/"
            f"{int(ridx)}",
        )
    except Exception as e:
        return (
            f"prsm_federated_learning aggregate failed: {e}"
        )
    if "status" not in r:
        detail = r.get("detail", "unknown error")
        return f"aggregate refused: {detail}"
    updates = r.get("gradient_updates_received") or []
    return (
        f"Round {r.get('round_index')} aggregated\n"
        f"  status:           {r.get('status')}\n"
        f"  updates_pooled:   {len(updates)}\n"
        f"  aggregated_bytes: "
        f"{len(r.get('aggregated_update_b64') or '')} (b64)"
    )


async def handle_prsm_corp_capability(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 306 — $CORP authorization capability MCP
    wrapper (Vision §7 Enterprise Confidentiality Mode
    layer 2)."""
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return (
            f"Missing required 'action' (must be one of "
            f"{sorted(_CORP_CAPABILITY_ACTIONS)})."
        )
    if action not in _CORP_CAPABILITY_ACTIONS:
        return (
            f"action must be one of "
            f"{sorted(_CORP_CAPABILITY_ACTIONS)}; got "
            f"{action!r}."
        )

    if action == "keypair_gen":
        from prsm.enterprise.corp_capability import (
            generate_issuer_keypair,
            generate_subject_keypair,
        )
        kind = (arguments.get("kind") or "issuer").lower()
        if kind == "subject":
            priv, pub = generate_subject_keypair()
            label = "Subject (Device-Bound)"
        else:
            priv, pub = generate_issuer_keypair()
            label = "Issuer (Corporate Signer)"
        return (
            f"$CORP {label} Keypair (Ed25519)\n\n"
            f"  pubkey_b64:   {pub}\n"
            f"  privkey_b64:  {priv}\n\n"
            "Issuer privkeys live in the enterprise "
            "HSM / SSO signer. Subject privkeys are "
            "device-bound (Secure Enclave, TPM, "
            "hardware security key). NEVER co-locate the "
            "two on the same host."
        )

    if action == "register_issuer":
        issuer_id = (arguments.get("issuer_id") or "").strip()
        pub = (
            arguments.get("signing_pubkey_b64") or ""
        ).strip()
        if not issuer_id or not pub:
            return (
                "register_issuer requires 'issuer_id' + "
                "'signing_pubkey_b64'."
            )
        try:
            r = await _call_node_api(
                "POST", "/admin/corp/issuer",
                {
                    "issuer_id": issuer_id,
                    "signing_pubkey_b64": pub,
                },
            )
        except Exception as e:
            return f"prsm_corp_capability register_issuer failed: {e}"
        if "issuer_id" not in r:
            detail = r.get("detail", "unknown error")
            return f"register_issuer refused: {detail}"
        return (
            f"Registered $CORP issuer\n"
            f"  issuer_id:    {r.get('issuer_id')}\n"
            f"  pubkey_b64:   {r.get('signing_pubkey_b64')}"
        )

    if action == "list_issuers":
        try:
            r = await _call_node_api(
                "GET", "/admin/corp/issuer",
            )
        except Exception as e:
            return f"prsm_corp_capability list_issuers failed: {e}"
        if "issuers" not in r:
            detail = r.get("detail", "unknown error")
            return f"list_issuers refused: {detail}"
        issuers = r.get("issuers") or []
        if not issuers:
            return "No $CORP issuers registered."
        lines = [
            f"PRSM $CORP Issuers — {len(issuers)} registered:",
            "",
        ]
        for i in issuers:
            pub = i.get("signing_pubkey_b64", "")
            lines.append(
                f"  · {i.get('issuer_id', '?'):<20} "
                f"{pub[:16]}..."
            )
        return "\n".join(lines)

    if action == "redeem":
        cap = arguments.get("capability")
        req = arguments.get("request")
        if not isinstance(cap, dict) or not isinstance(
            req, dict,
        ):
            return (
                "redeem requires 'capability' + 'request' "
                "objects (both dicts)."
            )
        try:
            r = await _call_node_api(
                "POST", "/admin/corp/capability/redeem",
                {"capability": cap, "request": req},
            )
        except Exception as e:
            return f"prsm_corp_capability redeem failed: {e}"
        if "status" not in r:
            detail = r.get("detail", "unknown error")
            return f"redeem refused: {detail}"
        sym = {
            "pass": "✅",
            "fail": "⚠ FAIL",
        }.get(r.get("status"), "?")
        return (
            f"{sym} Redemption — "
            f"{r.get('status', '?').upper()}\n\n"
            f"  capability_id:   "
            f"{r.get('capability_id', '?')}\n"
            f"  consumed_units:  "
            f"{r.get('units_consumed_this_request', 0)}\n"
            f"  remaining_quota: "
            f"{r.get('remaining_quota', 0)}\n"
            f"  diagnostic:      "
            f"{r.get('diagnostic') or '(none)'}"
        )

    if action == "get_ledger":
        cap_id = (
            arguments.get("capability_id") or ""
        ).strip()
        if not cap_id:
            return "get_ledger requires 'capability_id'."
        try:
            r = await _call_node_api(
                "GET",
                f"/admin/corp/capability/{cap_id}/ledger",
            )
        except Exception as e:
            return f"prsm_corp_capability get_ledger failed: {e}"
        if "entries" not in r:
            detail = r.get("detail", "unknown error")
            return f"get_ledger refused: {detail}"
        entries = r.get("entries") or []
        lines = [
            f"$CORP Audit Ledger — {cap_id} "
            f"({len(entries)} entries):",
            "",
        ]
        for e in entries:
            lines.append(
                f"  · ts={e.get('timestamp', '?')}  "
                f"{e.get('action', '?'):<22}  "
                f"units={e.get('units_requested', '?')}  "
                f"nonce={(e.get('nonce') or '')[:12]}  "
                f"subject={e.get('subject_id', '?')}"
            )
        return "\n".join(lines)

    # get_consumed
    cap_id = (arguments.get("capability_id") or "").strip()
    if not cap_id:
        return "get_consumed requires 'capability_id'."
    try:
        r = await _call_node_api(
            "GET",
            f"/admin/corp/capability/{cap_id}/consumed",
        )
    except Exception as e:
        return f"prsm_corp_capability get_consumed failed: {e}"
    if "consumed" not in r:
        detail = r.get("detail", "unknown error")
        return f"get_consumed refused: {detail}"
    return (
        f"$CORP capability {cap_id}\n"
        f"  consumed units: {r.get('consumed', 0):,}"
    )

_TEE_TIER_VALUES = (
    "none",
    "software",
    "hardware_unverified",
    "hardware_verified",
)


async def handle_prsm_tee_policy(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 305 — TEE policy MCP wrapper (Vision §7
    Enterprise Confidentiality Mode layer 3)."""
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return (
            f"Missing required 'action' (must be one of "
            f"{sorted(_TEE_POLICY_ACTIONS)})."
        )
    if action not in _TEE_POLICY_ACTIONS:
        return (
            f"action must be one of "
            f"{sorted(_TEE_POLICY_ACTIONS)}; got "
            f"{action!r}."
        )

    if action == "list_tiers":
        lines = [
            "PRSM TEE Attestation Tiers "
            "(strictly increasing):",
            "",
            "  · none                — no attestation "
            "required (default; no gating)",
            "  · software            — software-fallback "
            "attestation OK (parseable + signed receipt)",
            "  · hardware_unverified — real hardware TEE "
            "(Intel SGX/TDX, AMD SEV-SNP, Apple SEP); "
            "structural parse only (current production state "
            "until DCAP wiring)",
            "  · hardware_verified   — real hardware TEE "
            "with full cryptographic verification chain "
            "(vendor_verified=True)",
            "",
            "Policy satisfaction: effective_tier >= "
            "min_attestation_tier AND (allowed_vendors is "
            "None OR vendor in allowed_vendors).",
        ]
        return "\n".join(lines)

    if action == "evaluate":
        tier = (
            arguments.get("min_attestation_tier") or ""
        ).strip().lower()
        if not tier:
            return (
                "evaluate requires 'min_attestation_tier' "
                f"(one of {list(_TEE_TIER_VALUES)})."
            )
        body: Dict[str, Any] = {
            "attestation_b64": arguments.get(
                "attestation_b64",
            ),
            "policy": {
                "min_attestation_tier": tier,
                "allowed_vendors": arguments.get(
                    "allowed_vendors",
                ),
                "require_signature_chain": bool(
                    arguments.get(
                        "require_signature_chain", False,
                    ),
                ),
            },
        }
        try:
            r = await _call_node_api(
                "POST", "/admin/tee-policy/evaluate", body,
            )
        except Exception as e:
            return f"prsm_tee_policy evaluate failed: {e}"
        if "status" not in r:
            detail = r.get("detail", "unknown error")
            return f"evaluate refused: {detail}"
        sym = {
            "pass": "✅",
            "fail": "⚠ FAIL",
            "skipped": "·",
        }.get(r.get("status"), "?")
        lines = [
            f"{sym} TEE Policy Evaluation — "
            f"{r.get('status', '?').upper()}",
            "",
            f"  effective_tier:   {r.get('effective_tier')}",
            f"  min_required:     "
            f"{r.get('min_required_tier')}",
            f"  vendor:           {r.get('vendor') or '(none)'}",
            f"  diagnostic:       "
            f"{r.get('diagnostic') or '(none)'}",
        ]
        if r.get("error"):
            lines.append(f"  error:            {r['error']}")
        return "\n".join(lines)

    # node_status
    try:
        r = await _call_node_api(
            "GET", "/admin/tee-policy/node-status",
        )
    except Exception as e:
        return f"prsm_tee_policy node_status failed: {e}"
    if "effective_tier" not in r:
        detail = r.get("detail", "unknown error")
        return f"node_status refused: {detail}"
    return (
        "PRSM Node TEE Attestation Status:\n\n"
        f"  effective_tier:   {r.get('effective_tier')}\n"
        f"  vendor:           "
        f"{r.get('vendor') or '(unknown)'}\n"
        f"  vendor_verified:  "
        f"{r.get('vendor_verified', False)}\n"
        f"  diagnostic:       "
        f"{r.get('diagnostic') or '(none)'}"
    )


async def handle_prsm_enterprise_recipient(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 304 — recipient-encryption primitives MCP
    wrapper (Vision §7 Enterprise Confidentiality Mode).

    Most actions are pure client-side (no network call); only
    get_manifest hits the node. This deliberate split lets
    enterprises encrypt offline + air-gap + then upload, and
    decrypt without exposing the privkey to any service."""
    import base64 as _b64
    import json as _json

    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return (
            f"Missing required 'action' (must be one of "
            f"{sorted(_ENTERPRISE_RECIPIENT_ACTIONS)})."
        )
    if action not in _ENTERPRISE_RECIPIENT_ACTIONS:
        return (
            f"action must be one of "
            f"{sorted(_ENTERPRISE_RECIPIENT_ACTIONS)}; "
            f"got {action!r}."
        )

    if action == "keypair_gen":
        from prsm.enterprise.recipient_encryption import (
            generate_recipient_keypair,
        )
        priv, pub = generate_recipient_keypair()
        return (
            "PRSM Enterprise Recipient Keypair "
            "(X25519)\n\n"
            f"  pubkey_b64  (share with PRSM): {pub}\n"
            f"  privkey_b64 (KEEP SECRET):    {priv}\n\n"
            "Distribute the pubkey to the data uploader; "
            "keep the privkey in a hardware wallet / HSM / "
            "secrets manager. Loss of the privkey "
            "permanently revokes this recipient's access "
            "to encrypted content; no recovery is possible."
        )

    if action == "encrypt":
        from prsm.enterprise.recipient_encryption import (
            EnterpriseRecipient,
            encrypt_for_recipients,
        )
        pt_b64 = arguments.get("plaintext_b64") or ""
        if not pt_b64:
            return "encrypt requires 'plaintext_b64'."
        try:
            plaintext = _b64.b64decode(pt_b64, validate=True)
        except Exception as e:
            return f"encrypt: plaintext_b64 not valid: {e}"
        recipients_raw = arguments.get("recipients") or []
        if not recipients_raw:
            return "encrypt requires non-empty 'recipients'."
        try:
            recipients = [
                EnterpriseRecipient(
                    identifier=r.get("identifier", ""),
                    x25519_pubkey_b64=r.get(
                        "x25519_pubkey_b64", "",
                    ),
                )
                for r in recipients_raw
            ]
            payload = encrypt_for_recipients(
                plaintext, recipients,
            )
        except ValueError as e:
            return f"encrypt failed: {e}"
        return (
            f"Encrypted payload for "
            f"{len(recipients)} recipient(s):\n\n"
            f"{_json.dumps(payload.to_dict())}\n"
        )

    if action == "decrypt":
        from prsm.enterprise.recipient_encryption import (
            EncryptedPayload, decrypt_for_recipient,
        )
        priv = arguments.get("privkey_b64") or ""
        if not priv:
            return "decrypt requires 'privkey_b64'."
        payload_dict = arguments.get("payload")
        if not isinstance(payload_dict, dict):
            return "decrypt requires 'payload' object."
        try:
            payload = EncryptedPayload.from_dict(
                payload_dict,
            )
            out = decrypt_for_recipient(payload, priv)
        except ValueError as e:
            return f"decrypt failed: {e}"
        try:
            decoded = out.decode("utf-8")
            return f"Decrypted plaintext:\n\n{decoded}"
        except UnicodeDecodeError:
            return (
                "Decrypted (binary; base64):\n\n"
                f"{_b64.b64encode(out).decode()}"
            )

    # ── Sprint 307 — threshold mode actions ──────────
    if action == "encrypt_threshold":
        from prsm.enterprise.recipient_encryption import (
            EnterpriseRecipient,
            encrypt_for_threshold,
        )
        pt_b64 = arguments.get("plaintext_b64") or ""
        if not pt_b64:
            return (
                "encrypt_threshold requires 'plaintext_b64'."
            )
        try:
            plaintext = _b64.b64decode(pt_b64, validate=True)
        except Exception as e:
            return f"plaintext_b64 not valid: {e}"
        recipients_raw = arguments.get("recipients") or []
        if not recipients_raw:
            return (
                "encrypt_threshold requires non-empty "
                "'recipients'."
            )
        threshold = arguments.get("threshold")
        if threshold is None:
            return (
                "encrypt_threshold requires 'threshold' "
                "(t in [1, n])."
            )
        try:
            recipients = [
                EnterpriseRecipient(
                    identifier=r.get("identifier", ""),
                    x25519_pubkey_b64=r.get(
                        "x25519_pubkey_b64", "",
                    ),
                )
                for r in recipients_raw
            ]
            payload = encrypt_for_threshold(
                plaintext, recipients,
                threshold=int(threshold),
            )
        except ValueError as e:
            return f"encrypt_threshold failed: {e}"
        return (
            f"Threshold-encrypted payload "
            f"(t={threshold}, n={len(recipients)}):\n\n"
            f"{_json.dumps(payload.to_dict())}\n"
        )

    if action == "unseal_share":
        from prsm.enterprise.recipient_encryption import (
            EncryptedPayload, unseal_share_for_recipient,
        )
        priv = arguments.get("privkey_b64") or ""
        if not priv:
            return "unseal_share requires 'privkey_b64'."
        payload_dict = arguments.get("payload")
        if not isinstance(payload_dict, dict):
            return (
                "unseal_share requires 'payload' object."
            )
        try:
            payload = EncryptedPayload.from_dict(
                payload_dict,
            )
            contrib = unseal_share_for_recipient(
                payload, priv,
            )
        except ValueError as e:
            return f"unseal_share failed: {e}"
        return (
            f"Unsealed share:\n\n"
            f'{{"share_index": {contrib.share_index}, '
            f'"y_values_b64": '
            f'"{_b64.b64encode(contrib.share_y_values).decode()}"}}'
        )

    if action == "combine_decrypt":
        from prsm.enterprise.recipient_encryption import (
            EncryptedPayload, ShareContribution,
            combine_shares_and_decrypt,
        )
        payload_dict = arguments.get("payload")
        if not isinstance(payload_dict, dict):
            return (
                "combine_decrypt requires 'payload' object."
            )
        contribs_raw = arguments.get("contributions") or []
        if not contribs_raw:
            return (
                "combine_decrypt requires non-empty "
                "'contributions'."
            )
        try:
            payload = EncryptedPayload.from_dict(
                payload_dict,
            )
            contribs = [
                ShareContribution(
                    share_index=int(c["share_index"]),
                    share_y_values=_b64.b64decode(
                        c["share_y_values_b64"],
                    ),
                )
                for c in contribs_raw
            ]
            out = combine_shares_and_decrypt(
                payload, contribs,
            )
        except (ValueError, KeyError) as e:
            return f"combine_decrypt failed: {e}"
        try:
            return (
                f"Reconstructed plaintext:\n\n"
                f"{out.decode('utf-8')}"
            )
        except UnicodeDecodeError:
            return (
                f"Reconstructed (binary; base64):\n\n"
                f"{_b64.b64encode(out).decode()}"
            )

    # get_manifest
    cid = (arguments.get("cid") or "").strip()
    if not cid:
        return "get_manifest requires 'cid'."
    try:
        r = await _call_node_api(
            "GET", f"/content/recipient-manifest/{cid}",
        )
    except Exception as e:
        return f"prsm_enterprise_recipient get_manifest failed: {e}"
    if "entries" not in r:
        detail = r.get("detail", "unknown error")
        return f"get_manifest refused: {detail}"
    entries = r.get("entries") or []
    lines = [
        f"Recipient manifest for {cid} "
        f"(version {r.get('version', '?')}, "
        f"{len(entries)} entries):",
        "",
    ]
    for e in entries:
        lines.append(
            f"  · {e.get('identifier', '?')}  "
            f"(eph_pub: {(e.get('ephemeral_pubkey_b64') or '')[:16]}...)"
        )
    return "\n".join(lines)


def _short_proposal_id(pid: str) -> str:
    return pid[:8] if len(pid) > 8 else pid


async def handle_prsm_upgrade(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 303 — UUPS upgrade orchestrator MCP wrapper
    (Vision §14 item 7). Composer-only — Foundation Safe is
    the execution gate."""
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return (
            f"Missing required 'action' (must be one of "
            f"{sorted(_UPGRADE_ACTIONS)})."
        )
    if action not in _UPGRADE_ACTIONS:
        return (
            f"action must be one of "
            f"{sorted(_UPGRADE_ACTIONS)}; got {action!r}."
        )

    if action == "propose":
        required = [
            "target_proxy", "new_implementation",
            "previous_implementation", "severity",
            "rationale",
        ]
        missing = [
            f for f in required if not arguments.get(f)
        ]
        if missing:
            return (
                f"propose missing required field(s): "
                f"{missing}"
            )
        body = {
            "target_proxy": arguments["target_proxy"],
            "new_implementation": arguments[
                "new_implementation"
            ],
            "previous_implementation": arguments[
                "previous_implementation"
            ],
            "severity": arguments["severity"],
            "rationale": arguments["rationale"],
            "init_calldata_hex": (
                arguments.get("init_calldata_hex") or "0x"
            ),
            "reviewer_assignments": (
                arguments.get("reviewer_assignments") or []
            ),
        }
        try:
            r = await _call_node_api(
                "POST", "/admin/upgrade/propose", body,
            )
        except Exception as e:
            return f"prsm_upgrade propose failed: {e}"
        if "proposal_id" not in r:
            detail = r.get("detail", "unknown error")
            return f"propose refused: {detail}"
        return (
            f"Upgrade proposed\n"
            f"  id:       {r.get('proposal_id')}\n"
            f"  severity: {r.get('severity')}\n"
            f"  status:   {r.get('status')}\n"
            f"  proxy:    {r.get('target_proxy')}"
        )

    if action == "list":
        path = "/admin/upgrade"
        params = []
        st = (arguments.get("status") or "").strip().lower()
        if st:
            params.append(f"status={st}")
        sv = (arguments.get("severity") or "").strip().lower()
        if sv:
            params.append(f"severity={sv}")
        if params:
            path = f"{path}?{'&'.join(params)}"
        try:
            r = await _call_node_api("GET", path)
        except Exception as e:
            return f"prsm_upgrade list failed: {e}"
        if "records" not in r:
            detail = r.get("detail", "unknown error")
            return f"list refused: {detail}"
        records = r.get("records") or []
        if not records:
            return "No upgrade proposals."
        lines = [
            f"PRSM Upgrade Proposals — {r.get('count', 0)}",
            "",
            f"  {'id':<10} {'sev':<11} {'status':<14}  proxy",
        ]
        for rec in records:
            pid = _short_proposal_id(
                rec.get("proposal_id", ""),
            )
            lines.append(
                f"  {pid:<10} "
                f"{rec.get('severity', '?'):<11} "
                f"{rec.get('status', '?'):<14}  "
                f"{rec.get('target_proxy', '?')}"
            )
        return "\n".join(lines)

    if action == "lookup":
        pid = (arguments.get("proposal_id") or "").strip()
        if not pid:
            return "lookup requires 'proposal_id'."
        try:
            r = await _call_node_api(
                "GET", f"/admin/upgrade/{pid}",
            )
        except Exception as e:
            return f"prsm_upgrade lookup failed: {e}"
        if "proposal_id" not in r:
            detail = r.get("detail", "unknown error")
            return f"lookup refused: {detail}"
        lines = [
            f"Upgrade {r.get('proposal_id')}",
            "",
            f"  severity: {r.get('severity')}",
            f"  status:   {r.get('status')}",
            f"  proxy:    {r.get('target_proxy')}",
            f"  new_impl: {r.get('new_implementation')}",
            f"  prev_impl: "
            f"{r.get('previous_implementation')}",
            f"  rationale: {r.get('rationale')}",
            f"  safe_tx:   "
            f"{r.get('safe_tx_hash') or '(none)'}",
        ]
        return "\n".join(lines)

    if action == "update":
        pid = (arguments.get("proposal_id") or "").strip()
        if not pid:
            return "update requires 'proposal_id'."
        new_status = (
            arguments.get("new_status") or ""
        ).strip().lower()
        if not new_status:
            return "update requires 'new_status'."
        body = {"new_status": new_status}
        if arguments.get("safe_tx_hash"):
            body["safe_tx_hash"] = arguments["safe_tx_hash"]
        try:
            r = await _call_node_api(
                "POST", f"/admin/upgrade/{pid}/update",
                body,
            )
        except Exception as e:
            return f"prsm_upgrade update failed: {e}"
        if "proposal_id" not in r:
            detail = r.get("detail", "unknown error")
            return f"update refused: {detail}"
        return (
            f"Upgrade {r.get('proposal_id')} updated\n"
            f"  status:  {r.get('status')}\n"
            f"  safe_tx: "
            f"{r.get('safe_tx_hash') or '(none)'}"
        )

    # compose_upgrade / compose_rollback share render
    pid = (arguments.get("proposal_id") or "").strip()
    if not pid:
        return f"{action} requires 'proposal_id'."
    suffix = (
        "compose-upgrade"
        if action == "compose_upgrade"
        else "compose-rollback"
    )
    label = (
        "UPGRADE" if action == "compose_upgrade"
        else "ROLLBACK"
    )
    try:
        r = await _call_node_api(
            "POST", f"/admin/upgrade/{pid}/{suffix}",
        )
    except Exception as e:
        return f"prsm_upgrade {action} failed: {e}"
    if "data" not in r:
        detail = r.get("detail", "unknown error")
        return f"{action} refused: {detail}"
    explorer = r.get("explorer_url") or "(no explorer)"
    lines = [
        f"⚠ {label} COMPOSED — Foundation Safe upload "
        f"required ⚠",
        "",
        f"  WARNING: {r.get('warning', '')}",
        "",
        "  Transaction payload (paste into Safe UI):",
        f"    to:        {r.get('to', '?')}",
        f"    data:      {r.get('data', '?')}",
        f"    value:     {r.get('value', '0')}",
        f"    chain_id:  {r.get('chain_id', '?')}",
        "",
        f"  Proposal:    {r.get('proposal_id', '?')}",
        f"  Severity:    {r.get('severity', '?')}",
    ]
    if action == "compose_upgrade":
        lines.extend([
            f"  Rationale:   {r.get('rationale', '?')}",
            f"  New impl:    "
            f"{r.get('new_implementation', '?')}",
        ])
    else:
        lines.extend([
            f"  Rollback to: "
            f"{r.get('rollback_target_implementation', '?')}",
            f"  Was on:      "
            f"{r.get('originally_upgraded_to', '?')}",
        ])
    lines.extend([
        f"  Verify on:   {explorer}",
        "",
        f"  Instructions:",
        f"    {r.get('instructions', '')}",
    ])
    return "\n".join(lines)


async def handle_prsm_formal_verification(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 302 — formal-invariant harness (Vision §14
    item 4). Wraps /admin/formal-verification/*. The
    invariant LIST surface is PUBLIC by design (the spec is
    published before any incident — §14 transparency)."""
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return (
            f"Missing required 'action' (must be one of "
            f"{sorted(_FORMAL_VERIFICATION_ACTIONS)})."
        )
    if action not in _FORMAL_VERIFICATION_ACTIONS:
        return (
            f"action must be one of "
            f"{sorted(_FORMAL_VERIFICATION_ACTIONS)}; "
            f"got {action!r}."
        )

    if action == "list":
        contract = (
            arguments.get("contract") or ""
        ).strip()
        path = "/admin/formal-verification/invariants"
        if contract:
            path = f"{path}?contract={contract}"
        try:
            r = await _call_node_api("GET", path)
        except Exception as e:
            return f"prsm_formal_verification list failed: {e}"
        invs = r.get("invariants") or []
        if not invs:
            return "No pinned invariants found."
        lines = [
            f"PRSM Formal Invariants — {len(invs)} pinned "
            "(public spec):",
            "",
        ]
        for inv in invs:
            lines.append(
                f"  [{inv['id']:<10}] "
                f"{inv['severity']:<8}  "
                f"{inv['contract_name']:<22}  "
                f"{inv['title']}"
            )
            lines.append(
                f"    spec: {inv['spec_text']}"
            )
        return "\n".join(lines)

    if action == "check":
        contract = (
            arguments.get("contract") or ""
        ).strip()
        if not contract:
            return (
                "check requires 'contract' "
                "(e.g., 'royalty_distributor')."
            )
        try:
            r = await _call_node_api(
                "GET",
                f"/admin/formal-verification/check"
                f"?contract={contract}",
            )
        except Exception as e:
            return f"prsm_formal_verification check failed: {e}"
        if "results" not in r:
            detail = r.get("detail", "unknown error")
            return f"check refused: {detail}"
        summary = r.get("summary") or {}
        results = r.get("results") or []
        total = (
            summary.get("pass", 0)
            + summary.get("fail", 0)
            + summary.get("skipped", 0)
        )
        marker = (
            "✅"
            if summary.get("fail", 0) == 0 else "⚠"
        )
        lines = [
            f"{marker} Formal-Invariant Check — "
            f"{contract} @ {r.get('address')}",
            "",
            f"  Summary: "
            f"{summary.get('pass', 0)} pass / "
            f"{summary.get('fail', 0)} fail / "
            f"{summary.get('skipped', 0)} skipped "
            f"(of {total})",
            "",
        ]
        for rec in results:
            stat = rec.get("status", "?")
            sym = {
                "pass": "✅",
                "fail": "⚠ FAIL",
                "skipped": "·",
            }.get(stat, "?")
            line = (
                f"  {sym}  {rec.get('invariant_id'):<10}  "
            )
            diag = rec.get("diagnostic") or ""
            err = rec.get("error") or ""
            if diag:
                line += diag
            elif err:
                line += f"({err})"
            else:
                v = rec.get("value")
                e = rec.get("expected")
                if e is not None:
                    line += f"value={v}, expected={e}"
                elif v is not None:
                    line += f"value={v}"
            lines.append(line)
        return "\n".join(lines)

    if action == "symbolic_list":
        try:
            r = await _call_node_api(
                "GET",
                "/admin/formal-verification/symbolic",
            )
        except Exception as e:
            return (
                f"prsm_formal_verification symbolic_list "
                f"failed: {e}"
            )
        specs = r.get("specs") or []
        if not specs:
            return "No symbolic-proof specs registered."
        lines = [
            f"PRSM Symbolic-Proof Catalog — {len(specs)} "
            "spec(s) (halmos-compatible):",
            "",
        ]
        for s in specs:
            lines.append(
                f"  [{s['name']}]  mirrors "
                f"{s.get('mirrors_runtime_contract')}"
            )
            invs = s.get("runtime_invariants") or []
            if invs:
                lines.append(
                    f"    runtime invariants: "
                    f"{', '.join(invs)}"
                )
            desc = s.get("description") or ""
            if desc:
                lines.append(f"    {desc[:200]}")
        return "\n".join(lines)

    if action == "symbolic_check":
        spec = (arguments.get("spec") or "").strip()
        if not spec:
            return (
                "symbolic_check requires 'spec' "
                "(e.g., 'FTNSSupplyCapSpec')."
            )
        try:
            r = await _call_node_api(
                "GET",
                f"/admin/formal-verification/symbolic/"
                f"check/{spec}",
            )
        except Exception as e:
            return (
                f"prsm_formal_verification symbolic_check "
                f"failed: {e}"
            )
        if "status" not in r:
            detail = r.get("detail", "unknown error")
            return f"symbolic_check refused: {detail}"
        status = r.get("status", "?")
        summary = r.get("summary") or {}
        proofs = r.get("proofs") or []
        marker = {
            "passed": "✅",
            "failed": "⚠ FAIL",
            "skipped": "·",
            "error": "❌",
        }.get(status, "?")
        invs = r.get("runtime_invariants") or []
        lines = [
            f"{marker} Symbolic Proof — {spec}  ({status})",
            "",
            f"  halmos: {r.get('halmos_version') or 'unknown'}",
            (
                f"  Summary: {summary.get('passed', 0)} "
                f"passed / {summary.get('failed', 0)} "
                f"failed / {summary.get('errored', 0)} "
                f"errored"
            ),
        ]
        if invs:
            lines.append(
                f"  Mirrors runtime invariants: "
                f"{', '.join(invs)}"
            )
        err = r.get("error")
        if err:
            lines.append(f"  error: {err}")
        lines.append("")
        for p in proofs:
            psym = {
                "passed": "✅",
                "failed": "⚠ FAIL",
                "error":  "❌",
            }.get(p.get("status"), "·")
            line = (
                f"  {psym}  {p.get('name')}  "
                f"(paths: {p.get('paths_explored', 0)}, "
                f"time: {p.get('time_seconds', 0):.2f}s)"
            )
            lines.append(line)
            cex = p.get("counterexample")
            if cex:
                lines.append(
                    f"     counterexample: {cex}"
                )
        return "\n".join(lines)

    # check_one
    iid = (arguments.get("invariant_id") or "").strip()
    if not iid:
        return (
            "check_one requires 'invariant_id' "
            "(e.g., 'INV-RD-1')."
        )
    try:
        r = await _call_node_api(
            "GET",
            f"/admin/formal-verification/check/{iid}",
        )
    except Exception as e:
        return f"prsm_formal_verification check_one failed: {e}"
    if "status" not in r:
        detail = r.get("detail", "unknown error")
        return f"check_one refused: {detail}"
    stat = r.get("status", "?")
    sym = {
        "pass": "✅",
        "fail": "⚠ FAIL",
        "skipped": "·",
    }.get(stat, "?")
    lines = [
        f"{sym} {r.get('invariant_id')} — {stat}",
        "",
    ]
    if r.get("diagnostic"):
        lines.append(f"  {r['diagnostic']}")
    if r.get("error"):
        lines.append(f"  error: {r['error']}")
    if r.get("value") is not None:
        lines.append(f"  value:    {r['value']}")
    if r.get("expected") is not None:
        lines.append(f"  expected: {r['expected']}")
    return "\n".join(lines)

_INCIDENT_SEVERITY_VALUES = {"s0", "s1", "s2", "s3"}


def _short_incident_id(iid: str) -> str:
    return iid[:8] if len(iid) > 8 else iid


async def handle_prsm_incident(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 301 — public exploit-response playbook +
    incident lifecycle tracker (Vision §14 item 5).

    Wraps /admin/incident/*. The decision tree + comms
    templates are PUBLIC by design — anyone may read what
    PRSM has pre-committed to in an incident."""
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return (
            f"Missing required 'action' (must be one of "
            f"{sorted(_INCIDENT_ACTIONS)})."
        )
    if action not in _INCIDENT_ACTIONS:
        return (
            f"action must be one of "
            f"{sorted(_INCIDENT_ACTIONS)}; got {action!r}."
        )

    if action == "open":
        severity = (
            arguments.get("severity") or ""
        ).strip().lower()
        if not severity:
            return (
                f"open requires 'severity' (one of "
                f"{sorted(_INCIDENT_SEVERITY_VALUES)})."
            )
        body = {
            "severity": severity,
            "summary": arguments.get("summary") or "",
            "affected_contracts": (
                arguments.get("affected_contracts") or []
            ),
            "related_disclosure_id": arguments.get(
                "related_disclosure_id",
            ),
            "actor": arguments.get("actor") or "",
        }
        try:
            r = await _call_node_api(
                "POST", "/admin/incident/open", body,
            )
        except Exception as e:
            return f"prsm_incident open failed: {e}"
        if "incident_id" not in r:
            detail = r.get("detail", "unknown error")
            return f"open refused: {detail}"
        return (
            f"Incident opened\n"
            f"  id:       {r.get('incident_id')}\n"
            f"  severity: {r.get('severity')}\n"
            f"  phase:    {r.get('current_phase')}\n"
            f"  summary:  {r.get('summary')}"
        )

    if action == "list":
        params = []
        sev = (
            arguments.get("severity") or ""
        ).strip().lower()
        if sev:
            params.append(f"severity={sev}")
        phase = (
            arguments.get("phase") or ""
        ).strip().lower()
        if phase:
            params.append(f"phase={phase}")
        path = "/admin/incident"
        if params:
            path = f"{path}?{'&'.join(params)}"
        try:
            r = await _call_node_api("GET", path)
        except Exception as e:
            return f"prsm_incident list failed: {e}"
        if "records" not in r:
            detail = r.get("detail", "unknown error")
            return f"list refused: {detail}"
        records = r.get("records") or []
        if not records:
            return "No incidents recorded."
        lines = [
            f"PRSM Incidents — {r.get('count', 0)} record"
            f"{'s' if r.get('count', 0) != 1 else ''}",
            "",
            f"  {'id':<10} {'sev':<5} {'phase':<22}  summary",
        ]
        for rec in records:
            iid = _short_incident_id(
                rec.get("incident_id", ""),
            )
            sev_v = rec.get("severity", "?")
            phase_v = rec.get("current_phase", "?")
            summary = (rec.get("summary") or "")[:60]
            lines.append(
                f"  {iid:<10} {sev_v:<5} {phase_v:<22}  "
                f"{summary}"
            )
        return "\n".join(lines)

    if action == "lookup":
        iid = (
            arguments.get("incident_id") or ""
        ).strip()
        if not iid:
            return "lookup requires 'incident_id'."
        try:
            r = await _call_node_api(
                "GET", f"/admin/incident/{iid}",
            )
        except Exception as e:
            return f"prsm_incident lookup failed: {e}"
        if "incident_id" not in r:
            detail = r.get("detail", "unknown error")
            return f"lookup refused: {detail}"
        timeline = r.get("timeline") or []
        lines = [
            f"Incident {r.get('incident_id')}",
            "",
            f"  severity: {r.get('severity')}",
            f"  phase:    {r.get('current_phase')}",
            f"  summary:  {r.get('summary')}",
            f"  affected: "
            f"{', '.join(r.get('affected_contracts') or []) or '(none)'}",
            "",
            f"  Timeline ({len(timeline)} events):",
        ]
        for ev in timeline:
            lines.append(
                f"    [{ev.get('phase'):<22}] "
                f"{ev.get('note') or '(no note)'}"
                + (
                    f"  by {ev['actor']}"
                    if ev.get("actor") else ""
                )
            )
        return "\n".join(lines)

    if action == "advance":
        iid = (
            arguments.get("incident_id") or ""
        ).strip()
        if not iid:
            return "advance requires 'incident_id'."
        new_phase = (
            arguments.get("new_phase") or ""
        ).strip().lower()
        if not new_phase:
            return "advance requires 'new_phase'."
        body = {
            "new_phase": new_phase,
            "note": arguments.get("note") or "",
            "actor": arguments.get("actor") or "",
        }
        try:
            r = await _call_node_api(
                "POST", f"/admin/incident/{iid}/advance",
                body,
            )
        except Exception as e:
            return f"prsm_incident advance failed: {e}"
        if "incident_id" not in r:
            detail = r.get("detail", "unknown error")
            return f"advance refused: {detail}"
        return (
            f"Incident {r.get('incident_id')} advanced\n"
            f"  phase: {r.get('current_phase')}"
        )

    if action == "event":
        iid = (
            arguments.get("incident_id") or ""
        ).strip()
        if not iid:
            return "event requires 'incident_id'."
        note = arguments.get("note") or ""
        if not note:
            return "event requires 'note'."
        body = {
            "note": note,
            "actor": arguments.get("actor") or "",
        }
        try:
            r = await _call_node_api(
                "POST", f"/admin/incident/{iid}/event",
                body,
            )
        except Exception as e:
            return f"prsm_incident event failed: {e}"
        if "incident_id" not in r:
            detail = r.get("detail", "unknown error")
            return f"event refused: {detail}"
        last = (r.get("timeline") or [])[-1:]
        last_note = (
            last[0].get("note") if last else "(none)"
        )
        return (
            f"Event recorded on "
            f"{r.get('incident_id')}\n"
            f"  phase: {r.get('current_phase')}\n"
            f"  note:  {last_note}"
        )

    if action == "recommend":
        iid = (
            arguments.get("incident_id") or ""
        ).strip()
        if not iid:
            return "recommend requires 'incident_id'."
        try:
            r = await _call_node_api(
                "GET",
                f"/admin/incident/{iid}/recommendations",
            )
        except Exception as e:
            return f"prsm_incident recommend failed: {e}"
        if "recommendations" not in r:
            detail = r.get("detail", "unknown error")
            return f"recommend refused: {detail}"
        recs = r.get("recommendations") or []
        if not recs:
            return (
                f"No pre-committed recommendations for "
                f"({r.get('severity')}, "
                f"{r.get('current_phase')})."
            )
        lines = [
            f"Playbook recommendations — "
            f"{r.get('severity')} / "
            f"{r.get('current_phase')}:",
            "",
        ]
        for i, rec in enumerate(recs, 1):
            lines.append(f"  {i}. {rec}")
        return "\n".join(lines)

    if action == "comms":
        iid = (
            arguments.get("incident_id") or ""
        ).strip()
        if not iid:
            return "comms requires 'incident_id'."
        try:
            r = await _call_node_api(
                "GET",
                f"/admin/incident/{iid}/comms-template",
            )
        except Exception as e:
            return f"prsm_incident comms failed: {e}"
        if "text" not in r:
            detail = r.get("detail", "unknown error")
            return f"comms refused: {detail}"
        text = r.get("text") or ""
        if not text.strip():
            return (
                f"No pre-committed comms template for "
                f"({r.get('severity')}, "
                f"{r.get('current_phase')})."
            )
        return (
            f"Comms template — {r.get('severity')} / "
            f"{r.get('current_phase')}:\n\n"
            f"{text}"
        )

    # playbook
    try:
        r = await _call_node_api(
            "GET", "/admin/incident/playbook",
        )
    except Exception as e:
        return f"prsm_incident playbook failed: {e}"
    if "decision_tree" not in r:
        detail = r.get("detail", "unknown error")
        return f"playbook refused: {detail}"
    tree = r.get("decision_tree") or []
    lines = [
        "PRSM Exploit-Response Playbook "
        "(Vision §14 — pre-committed, public)",
        "",
    ]
    for entry in tree:
        sev = entry.get("severity", "?")
        phase = entry.get("phase", "?")
        recs = entry.get("recommendations") or []
        lines.append(f"  [{sev} / {phase}]")
        for rec in recs:
            lines.append(f"    • {rec}")
        lines.append("")
    return "\n".join(lines)

_DISCLOSURE_SEVERITY_VALUES = {
    "critical", "high", "medium", "low", "informational",
}


def _short_disclosure_id(did: str) -> str:
    return did[:8] if len(did) > 8 else did


async def handle_prsm_disclosure(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 300 — responsible-disclosure intake + bounty
    payout composer (Vision §14 mitigation item 3).

    Wraps /admin/disclosure/* endpoints. The compose_payout
    action returns a Foundation-Safe-uploadable ERC-20
    transfer payload — PRSM never executes the bounty
    payment; the multisig is the gate."""
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return (
            f"Missing required 'action' (must be one of "
            f"{sorted(_DISCLOSURE_ACTIONS)})."
        )
    if action not in _DISCLOSURE_ACTIONS:
        return (
            f"action must be one of "
            f"{sorted(_DISCLOSURE_ACTIONS)}; got {action!r}."
        )

    if action == "submit":
        severity = (
            arguments.get("severity") or ""
        ).strip().lower()
        if not severity:
            return (
                f"submit requires 'severity' (one of "
                f"{sorted(_DISCLOSURE_SEVERITY_VALUES)})."
            )
        body = {
            "severity": severity,
            "summary": arguments.get("summary") or "",
            "affected_contracts": (
                arguments.get("affected_contracts") or []
            ),
            "researcher_contact": (
                arguments.get("researcher_contact") or ""
            ),
            "details": arguments.get("details") or "",
        }
        try:
            r = await _call_node_api(
                "POST", "/admin/disclosure/submit", body,
            )
        except Exception as e:
            return f"prsm_disclosure submit failed: {e}"
        if "disclosure_id" not in r:
            detail = r.get("detail", "unknown error")
            return f"submit refused: {detail}"
        return (
            f"Disclosure received\n"
            f"  id:        {r.get('disclosure_id')}\n"
            f"  severity:  {r.get('severity')}\n"
            f"  status:    {r.get('status')}\n"
            f"  summary:   {r.get('summary')}"
        )

    if action == "list":
        path = "/admin/disclosure"
        params = []
        sev = (
            arguments.get("severity") or ""
        ).strip().lower()
        if sev:
            params.append(f"severity={sev}")
        status_filter = (
            arguments.get("status") or ""
        ).strip().lower()
        if status_filter:
            params.append(f"status={status_filter}")
        if params:
            path = f"{path}?{'&'.join(params)}"
        try:
            r = await _call_node_api("GET", path)
        except Exception as e:
            return f"prsm_disclosure list failed: {e}"
        if "records" not in r:
            detail = r.get("detail", "unknown error")
            return f"list refused: {detail}"
        records = r.get("records") or []
        if not records:
            return "No disclosures recorded."
        lines = [
            f"PRSM Disclosures — {r.get('count', 0)} record"
            f"{'s' if r.get('count', 0) != 1 else ''}",
            "",
            f"  {'id':<10} {'sev':<14} {'status':<14}  summary",
        ]
        for rec in records:
            did = _short_disclosure_id(
                rec.get("disclosure_id", ""),
            )
            sev = rec.get("severity", "?")
            status_v = rec.get("status", "?")
            summary = (rec.get("summary") or "")[:60]
            lines.append(
                f"  {did:<10} {sev:<14} {status_v:<14}  "
                f"{summary}"
            )
        return "\n".join(lines)

    if action == "lookup":
        did = (
            arguments.get("disclosure_id") or ""
        ).strip()
        if not did:
            return "lookup requires 'disclosure_id'."
        try:
            r = await _call_node_api(
                "GET", f"/admin/disclosure/{did}",
            )
        except Exception as e:
            return f"prsm_disclosure lookup failed: {e}"
        if "disclosure_id" not in r:
            detail = r.get("detail", "unknown error")
            return f"lookup refused: {detail}"
        lines = [
            f"Disclosure {r.get('disclosure_id')}",
            "",
            f"  severity:           {r.get('severity')}",
            f"  status:             {r.get('status')}",
            f"  summary:            {r.get('summary')}",
            f"  affected_contracts: "
            f"{', '.join(r.get('affected_contracts') or []) or '(none)'}",
            f"  researcher_contact: "
            f"{r.get('researcher_contact')}",
            f"  triage_notes:       "
            f"{r.get('triage_notes') or '(none)'}",
            f"  payout_ftns:        "
            f"{r.get('payout_ftns') or 0:,}",
            f"  payout_tx_hash:     "
            f"{r.get('payout_tx_hash') or '(none)'}",
        ]
        return "\n".join(lines)

    if action == "update":
        did = (
            arguments.get("disclosure_id") or ""
        ).strip()
        if not did:
            return "update requires 'disclosure_id'."
        new_status = (
            arguments.get("new_status") or ""
        ).strip().lower()
        if not new_status:
            return "update requires 'new_status'."
        body = {"new_status": new_status}
        if arguments.get("triage_notes") is not None:
            body["triage_notes"] = arguments["triage_notes"]
        if arguments.get("payout_ftns") is not None:
            body["payout_ftns"] = arguments["payout_ftns"]
        try:
            r = await _call_node_api(
                "POST",
                f"/admin/disclosure/{did}/update", body,
            )
        except Exception as e:
            return f"prsm_disclosure update failed: {e}"
        if "disclosure_id" not in r:
            detail = r.get("detail", "unknown error")
            return f"update refused: {detail}"
        return (
            f"Disclosure {r.get('disclosure_id')} updated\n"
            f"  status:       {r.get('status')}\n"
            f"  triage_notes: {r.get('triage_notes') or '(none)'}\n"
            f"  payout_ftns:  {r.get('payout_ftns') or 0:,}"
        )

    if action == "compose_payout":
        did = (
            arguments.get("disclosure_id") or ""
        ).strip()
        if not did:
            return "compose_payout requires 'disclosure_id'."
        recipient = (
            arguments.get("recipient") or ""
        ).strip()
        if not recipient:
            return "compose_payout requires 'recipient'."
        try:
            r = await _call_node_api(
                "POST",
                f"/admin/disclosure/{did}/compose-payout",
                {"recipient": recipient},
            )
        except Exception as e:
            return f"prsm_disclosure compose_payout failed: {e}"
        if "data" not in r:
            detail = r.get("detail", "unknown error")
            return f"compose_payout refused: {detail}"
        explorer = r.get("explorer_url") or "(no explorer)"
        amt_ftns = r.get("amount_ftns", 0)
        lines = [
            f"⚠ BOUNTY PAYOUT COMPOSED — Foundation Safe "
            f"upload required ⚠",
            "",
            f"  WARNING: {r.get('warning', '')}",
            "",
            "  Transaction payload (paste into Safe UI):",
            f"    to:        {r.get('to', '?')}",
            f"    data:      {r.get('data', '?')}",
            f"    value:     {r.get('value', '0')}",
            f"    chain_id:  {r.get('chain_id', '?')}",
            "",
            f"  Disclosure:  {r.get('disclosure_id', '?')}",
            f"  Severity:    {r.get('severity', '?')}",
            f"  Summary:     {r.get('summary', '?')}",
            f"  Recipient:   {r.get('recipient', '?')}",
            f"  Amount:      {amt_ftns:,} FTNS",
            f"  Verify on:   {explorer}",
            "",
            f"  Instructions:",
            f"    {r.get('instructions', '')}",
        ]
        return "\n".join(lines)

    # record_payout_tx
    did = (arguments.get("disclosure_id") or "").strip()
    if not did:
        return "record_payout_tx requires 'disclosure_id'."
    tx_hash = (arguments.get("tx_hash") or "").strip()
    if not tx_hash:
        return "record_payout_tx requires 'tx_hash'."
    try:
        r = await _call_node_api(
            "POST",
            f"/admin/disclosure/{did}/record-payout-tx",
            {"tx_hash": tx_hash},
        )
    except Exception as e:
        return f"prsm_disclosure record_payout_tx failed: {e}"
    if "disclosure_id" not in r:
        detail = r.get("detail", "unknown error")
        return f"record_payout_tx refused: {detail}"
    return (
        f"Audit trail closed for "
        f"{r.get('disclosure_id')}\n"
        f"  payout_tx_hash: {r.get('payout_tx_hash')}\n"
        f"  status:         {r.get('status')}"
    )


async def handle_prsm_emergency_pause(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 298 — emergency pause composer (Vision §14
    smart-contract exploit response).

    action selector:
      status         — bulk paused-state query per contract
      compose_pause  — Safe-uploadable pause tx payload
      compose_unpause — Safe-uploadable unpause tx payload

    The composer outputs calldata for the Foundation Safe
    2-of-3 hardware multisig — PRSM never executes pause
    directly. This MCP surface exists to SAVE OPERATORS TIME
    constructing pause calldata by hand during an active
    incident; speed matters when an exploit is ongoing."""
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return (
            f"Missing required 'action' (must be one of "
            f"{sorted(_EMERGENCY_PAUSE_ACTIONS)})."
        )
    if action not in _EMERGENCY_PAUSE_ACTIONS:
        return (
            f"action must be one of "
            f"{sorted(_EMERGENCY_PAUSE_ACTIONS)}; "
            f"got {action!r}."
        )

    if action == "status":
        try:
            result = await _call_node_api(
                "GET", "/admin/emergency-pause/status",
            )
        except Exception as e:
            return (
                f"prsm_emergency_pause failed: {e}\n"
                f"Is your PRSM node running?"
            )
        if "contracts" not in result:
            detail = result.get("detail", "unknown error")
            if "not initialized" in str(detail).lower():
                return (
                    f"Emergency pause client not wired.\n"
                    f"  Detail: {detail}"
                )
            return f"status refused: {detail}"
        contracts = result.get("contracts") or {}
        chain_id = result.get("chain_id", "?")
        lines = [
            f"PRSM Emergency Pause — chain_id={chain_id}",
            "",
        ]
        for name, status in sorted(contracts.items()):
            paused = status.get("paused")
            commissioned = status.get("commissioned")
            err = status.get("error")
            if not commissioned:
                marker = "·  uncommissioned"
            elif err:
                marker = f"⚠ RPC error: {err[:40]}"
            elif paused is True:
                marker = "⚠ PAUSED"
            elif paused is False:
                marker = "✅ active"
            else:
                marker = "? unknown"
            addr = status.get("address") or "(unset)"
            short_addr = (
                addr[:10] + "…" + addr[-4:]
                if len(addr) > 14 else addr
            )
            lines.append(
                f"  {marker:30s}  {name:<28}  {short_addr}"
            )
        return "\n".join(lines)

    # compose_pause / compose_unpause
    contract_name = (
        arguments.get("contract_name") or ""
    ).strip()
    if not contract_name:
        return f"{action} requires 'contract_name'."
    pause_action = (
        "pause" if action == "compose_pause" else "unpause"
    )
    try:
        result = await _call_node_api(
            "POST", "/admin/emergency-pause/compose",
            {
                "action": pause_action,
                "contract_name": contract_name,
            },
        )
    except Exception as e:
        return (
            f"prsm_emergency_pause failed: {e}\n"
            f"Is your PRSM node running?"
        )
    if "data" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in str(detail).lower():
            return (
                f"Emergency pause client not wired.\n"
                f"  Detail: {detail}"
            )
        return f"compose refused: {detail}"

    explorer = result.get("explorer_url") or "(no explorer)"
    lines = [
        f"⚠ EMERGENCY {pause_action.upper()} "
        f"COMPOSED — Foundation Safe upload required ⚠",
        "",
        f"  WARNING: {result.get('warning', '')}",
        "",
        "  Transaction payload (paste into Safe UI):",
        f"    to:        {result.get('to', '?')}",
        f"    data:      {result.get('data', '?')}",
        f"    value:     {result.get('value', '0')}",
        f"    chain_id:  {result.get('chain_id', '?')}",
        "",
        f"  Target:      {result.get('contract_name', '?')}",
        f"  Description: {result.get('description', '?')}",
        f"  Verify on:   {explorer}",
        "",
        f"  Instructions:",
        f"    {result.get('instructions', '')}",
    ]
    return "\n".join(lines)


async def handle_prsm_verify_inference_privacy(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 292 — verify an InferenceReceipt's privacy
    claims (Vision §7). Wraps POST /compute/receipt/verify.

    Surfaces the truth about hardware-attestation quality
    even when the default permissive posture returns ok=True
    — every receipt produced by the local executor today
    carries a DEV-ONLY software-fallback attestation, and
    callers should know that.
    """
    receipt = arguments.get("receipt")
    if not isinstance(receipt, dict):
        return (
            "Missing required argument: receipt (dict). "
            "Pass the full InferenceReceipt payload returned "
            "by /compute/inference."
        )
    public_key_b64 = (
        arguments.get("public_key_b64") or ""
    ).strip()
    if not public_key_b64:
        return (
            "Missing required argument: public_key_b64. "
            "Fetch via GET /node/identity/pubkey on the "
            "node that signed the receipt."
        )
    body = {
        "receipt": receipt,
        "public_key_b64": public_key_b64,
        "require_hardware_attestation": bool(
            arguments.get("require_hardware_attestation", False),
        ),
        "require_dp_noise": bool(
            arguments.get("require_dp_noise", False),
        ),
    }
    try:
        result = await _call_node_api(
            "POST", "/compute/receipt/verify", body,
        )
    except Exception as e:
        return (
            f"prsm_verify_inference_privacy failed: {e}\n"
            f"Is your PRSM node running?"
        )
    if "ok" not in result:
        detail = result.get("detail", "unknown error")
        return (
            f"Verify refused: {detail}"
        )

    ok = result.get("ok")
    sig_valid = result.get("signature_valid")
    dp_applied = result.get("dp_noise_applied")
    hw_attested = result.get("hardware_attested")
    multi_stage = result.get(
        "multi_stage_envelope_present", False,
    )
    tier = result.get("privacy_tier", "?")
    eps_spent = result.get("epsilon_spent", 0)
    eps_expected = result.get("expected_epsilon", 0)
    reasons = result.get("reasons") or []
    # Sprint 293 — backend-supplied attestation detail
    attest_vendor = result.get(
        "attestation_vendor", "unknown",
    )
    attest_vendor_data = result.get(
        "attestation_vendor_data", {}
    ) or {}
    attest_vendor_verified = result.get(
        "attestation_vendor_verified", False,
    )

    verdict = (
        "✅ VALID" if ok
        else "❌ REJECTED (failed required check)"
    )
    lines = [
        f"Receipt privacy verdict: {verdict}",
        "",
        f"  Signature:        "
        f"{'✅ valid' if sig_valid else '❌ INVALID'}",
        f"  DP noise applied: "
        f"{'✅ yes' if dp_applied else '⚠ no'}  "
        f"(ε={eps_spent}; tier={tier}; "
        f"expected ε={eps_expected})",
        f"  Hardware TEE:     "
        f"{'✅ hardware-attested' if hw_attested else '⚠ SOFTWARE FALLBACK (DEV-ONLY)'}",
        f"  Multi-stage env:  "
        f"{'present' if multi_stage else 'single-host'}",
        f"  Vendor:           "
        f"{attest_vendor}  "
        f"({'cryptographically verified' if attest_vendor_verified else 'structural-parse only'})",
    ]
    # Surface parsed measurements when present (sprint 293
    # backends fill these for Intel SGX/TDX). Callers can
    # pin against expected MRENCLAVE / MRSIGNER values
    # out-of-band.
    if attest_vendor.startswith("intel-sgx") and "mrenclave_hex" in attest_vendor_data:
        lines.append(
            f"    MRENCLAVE: "
            f"{attest_vendor_data.get('mrenclave_hex')}"
        )
        lines.append(
            f"    MRSIGNER:  "
            f"{attest_vendor_data.get('mrsigner_hex')}"
        )
    elif attest_vendor == "intel-tdx" and "mrtd_hex" in attest_vendor_data:
        lines.append(
            f"    MRTD:      "
            f"{attest_vendor_data.get('mrtd_hex')}"
        )
        lines.append(
            f"    RTMR0:     "
            f"{attest_vendor_data.get('rtmr0_hex')}"
        )
    elif attest_vendor == "amd-sev-snp" and "measurement_hex" in attest_vendor_data:
        lines.append(
            f"    MEASUREMENT: "
            f"{attest_vendor_data.get('measurement_hex')}"
        )
        lines.append(
            f"    REPORT_DATA: "
            f"{attest_vendor_data.get('report_data_hex')}"
        )
        lines.append(
            f"    CHIP_ID:     "
            f"{attest_vendor_data.get('chip_id_hex')}"
        )
        lines.append(
            f"    guest_svn:   "
            f"{attest_vendor_data.get('guest_svn')}"
        )
    if not hw_attested:
        lines.append(
            "    Truth check: every local-executor receipt "
            "today carries a DEV-ONLY software-fallback "
            "attestation. Hardware TEE (Intel ASP, AMD KDS, "
            "Apple SEP) backends ship in a future sprint. "
            "Pass require_hardware_attestation=true to fail "
            "loudly on this until then."
        )
    if reasons:
        lines.append("")
        lines.append("  Reasons:")
        for r in reasons:
            lines.append(f"    • {r}")
    return "\n".join(lines)


_CONTENT_FINGERPRINT_ACTIONS = {"list", "lookup"}


async def handle_prsm_content_fingerprint(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 291 — content fingerprint inspection (Vision
    §14 item 3 cryptographic deduplication). action: list |
    lookup. First-creator-wins dedup; duplicate attempts are
    counted but don't reassign the canonical creator."""
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return (
            f"Missing required 'action' (must be one of "
            f"{sorted(_CONTENT_FINGERPRINT_ACTIONS)})."
        )
    if action not in _CONTENT_FINGERPRINT_ACTIONS:
        return (
            f"action must be one of "
            f"{sorted(_CONTENT_FINGERPRINT_ACTIONS)}; "
            f"got {action!r}."
        )

    if action == "lookup":
        content_hash = (
            arguments.get("content_hash") or ""
        ).strip()
        if not content_hash:
            return "lookup requires 'content_hash'."
        try:
            result = await _call_node_api(
                "GET",
                f"/marketplace/fingerprint/{content_hash}",
            )
        except Exception as e:
            return (
                f"prsm_content_fingerprint failed: {e}\n"
                f"Is your PRSM node running?"
            )
        if "content_hash" not in result:
            detail = result.get("detail", "unknown error")
            if "no fingerprint" in str(detail).lower():
                return (
                    f"Unknown fingerprint {content_hash!r} — "
                    f"either no upload has registered this "
                    f"content_hash yet, or the registry was "
                    f"reset."
                )
            if "not initialized" in str(detail).lower():
                return (
                    f"Fingerprint registry not wired.\n"
                    f"  Detail: {detail}"
                )
            return f"lookup refused: {detail}"
        return "\n".join([
            f"Content Fingerprint — {result['content_hash']}:",
            f"  canonical_creator:       "
            f"{result.get('canonical_creator', '?')}",
            f"  first_seen_unix:         "
            f"{result.get('first_seen_unix', 0)}",
            f"  duplicate_attempt_count: "
            f"{result.get('duplicate_attempt_count', 0)}",
        ])

    # action == "list"
    limit = int(arguments.get("limit", 100))
    path = f"/marketplace/fingerprint?limit={limit}"
    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"prsm_content_fingerprint failed: {e}"
        )
    if "fingerprints" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in str(detail).lower():
            return (
                f"Fingerprint registry not wired.\n"
                f"  Detail: {detail}"
            )
        return f"list refused: {detail}"
    fps = result.get("fingerprints") or []
    total = result.get("count", 0)
    lines = [
        f"Content Fingerprints — {len(fps)} of {total} "
        f"(newest first):"
    ]
    if not fps:
        lines.append("  (registry empty)")
    for f in fps:
        dup_count = f.get("duplicate_attempt_count", 0)
        dup_marker = (
            f" ⚠ {dup_count} dup attempts"
            if dup_count > 0 else ""
        )
        lines.append(
            f"  {f.get('content_hash', '?')[:24]}  "
            f"creator={f.get('canonical_creator', '?')[:16]}"
            f"{dup_marker}"
        )
    return "\n".join(lines)


_CREATOR_STAKE_ACTIONS = {"balance", "stake", "slash"}


async def handle_prsm_creator_stake(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 290 — creator-stake operator surface
    (Vision §14 item 2). action selector: balance | stake |
    slash. Defends spam pattern by making high-tier status
    require bonded FTNS that gets slashed on misbehavior."""
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return (
            f"Missing required 'action' (must be one of "
            f"{sorted(_CREATOR_STAKE_ACTIONS)})."
        )
    if action not in _CREATOR_STAKE_ACTIONS:
        return (
            f"action must be one of "
            f"{sorted(_CREATOR_STAKE_ACTIONS)}; "
            f"got {action!r}."
        )

    if action == "balance":
        creator_id = (
            arguments.get("creator_id") or ""
        ).strip()
        if not creator_id:
            return "balance requires 'creator_id'."
        try:
            result = await _call_node_api(
                "GET",
                f"/marketplace/creator-stake/{creator_id}",
            )
        except Exception as e:
            return (
                f"prsm_creator_stake failed: {e}\n"
                f"Is your PRSM node running?"
            )
        if "balance_wei" not in result:
            detail = result.get("detail", "unknown error")
            if "not initialized" in str(detail).lower():
                return (
                    f"Creator stake client not wired.\n"
                    f"  Detail: {detail}"
                )
            return f"balance refused: {detail}"
        return "\n".join([
            f"Creator Stake — {result['creator_id']}:",
            f"  balance_wei:             "
            f"{result.get('balance_wei', 0)}",
            f"  min_high_tier_stake_wei: "
            f"{result.get('min_high_tier_stake_wei', 0)}",
            f"  high_tier_eligible:      "
            f"{result.get('high_tier_eligible', False)}",
            f"  commissioned:            "
            f"{result.get('commissioned', False)}",
        ])

    if action == "stake":
        creator_id = (
            arguments.get("creator_id") or ""
        ).strip()
        if not creator_id:
            return "stake requires 'creator_id'."
        amount_wei = arguments.get("amount_wei")
        if amount_wei is None:
            return "stake requires 'amount_wei'."
        try:
            amount_wei = int(amount_wei)
        except (ValueError, TypeError):
            return (
                f"amount_wei must be an integer, "
                f"got {amount_wei!r}."
            )
        if amount_wei <= 0:
            return "amount_wei must be > 0."
        try:
            result = await _call_node_api(
                "POST",
                "/marketplace/creator-stake/stake",
                {
                    "creator_id": creator_id,
                    "amount_wei": amount_wei,
                },
            )
        except Exception as e:
            return (
                f"prsm_creator_stake failed: {e}"
            )
        if "balance_wei" not in result:
            detail = result.get("detail", "unknown error")
            return f"stake refused: {detail}"
        return (
            f"Staked {amount_wei} wei for {creator_id}.\n"
            f"  new balance:        "
            f"{result.get('balance_wei', 0)} wei\n"
            f"  high_tier_eligible: "
            f"{result.get('high_tier_eligible', False)}"
        )

    # action == "slash"
    creator_id = (arguments.get("creator_id") or "").strip()
    if not creator_id:
        return "slash requires 'creator_id'."
    amount_wei = arguments.get("amount_wei")
    if amount_wei is None:
        return "slash requires 'amount_wei'."
    try:
        amount_wei = int(amount_wei)
    except (ValueError, TypeError):
        return (
            f"amount_wei must be an integer, "
            f"got {amount_wei!r}."
        )
    if amount_wei <= 0:
        return "amount_wei must be > 0."
    reason = (arguments.get("reason") or "").strip()
    if not reason:
        return (
            "slash requires 'reason' "
            "(short description for audit trail)."
        )
    try:
        result = await _call_node_api(
            "POST",
            "/marketplace/creator-stake/slash",
            {
                "creator_id": creator_id,
                "amount_wei": amount_wei,
                "reason": reason,
            },
        )
    except Exception as e:
        return f"prsm_creator_stake failed: {e}"
    if "slashed_wei" not in result:
        detail = result.get("detail", "unknown error")
        return f"slash refused: {detail}"
    return (
        f"Slashed {amount_wei} wei from {creator_id} "
        f"(reason: {reason}).\n"
        f"  new balance: "
        f"{result.get('balance_wei', 0)} wei"
    )


_CREATOR_REPUTATION_ACTIONS = {"list", "lookup"}


async def handle_prsm_creator_reputation(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 287 — operator visibility into creator-side
    reputation (Vision §14 data quality / Sybil resistance).
    action selector: list | lookup. Read-only; recording is
    automatic from ContentStore retrieve paths (sprint 288
    wiring)."""
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return (
            f"Missing required 'action' (must be one of "
            f"{sorted(_CREATOR_REPUTATION_ACTIONS)})."
        )
    if action not in _CREATOR_REPUTATION_ACTIONS:
        return (
            f"action must be one of "
            f"{sorted(_CREATOR_REPUTATION_ACTIONS)}; "
            f"got {action!r}."
        )

    if action == "list":
        limit = int(arguments.get("limit", 100))
        path = (
            f"/marketplace/creator-reputation?limit={limit}"
        )
        try:
            result = await _call_node_api("GET", path)
        except Exception as e:
            return (
                f"prsm_creator_reputation failed: {e}\n"
                f"Is your PRSM node running? (prsm node start)"
            )
        if "creators" not in result:
            detail = result.get("detail", "unknown error")
            if "not initialized" in str(detail).lower():
                return (
                    f"Creator reputation tracker not wired.\n"
                    f"  Detail: {detail}\n"
                    f"  Tracker is built when "
                    f"QueryOrchestrator wires."
                )
            return f"list refused: {detail}"
        creators = result.get("creators") or []
        total = result.get("count", 0)
        lines = [
            f"PRSM Creator Reputation — {len(creators)} of "
            f"{total} known creators (score desc):",
        ]
        if not creators:
            lines.append("  (none)")
        for c in creators:
            score = c.get("score", 0.0)
            lines.append(
                f"  {c.get('creator_id', '?')}  "
                f"score={score:.3f}  "
                f"accesses={c.get('total_accesses', 0)}  "
                f"distinct={c.get('distinct_purchasers', 0)}  "
                f"repeats="
                f"{c.get('repeat_purchaser_count', 0)}"
            )
        return "\n".join(lines)

    # action == "lookup"
    creator_id = (arguments.get("creator_id") or "").strip()
    if not creator_id:
        return "lookup requires 'creator_id'."
    try:
        result = await _call_node_api(
            "GET",
            f"/marketplace/creator-reputation/{creator_id}",
        )
    except Exception as e:
        return (
            f"prsm_creator_reputation failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "creator_id" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in str(detail).lower():
            return (
                f"Creator reputation tracker not wired.\n"
                f"  Detail: {detail}"
            )
        return f"lookup refused: {detail}"
    known = result.get("known", False)
    total = result.get("total_accesses", 0)
    cold_start = (
        " (cold-start — < 10 access events)"
        if not known or total < 10
        else ""
    )
    return "\n".join([
        f"Creator Reputation — {result['creator_id']}:",
        f"  known:                  {known}",
        f"  score:                  "
        f"{result.get('score', 0.0):.3f}{cold_start}",
        f"  total_accesses:         "
        f"{result.get('total_accesses', 0)}",
        f"  distinct_purchasers:    "
        f"{result.get('distinct_purchasers', 0)}",
        f"  repeat_purchaser_count: "
        f"{result.get('repeat_purchaser_count', 0)}",
        f"  first_seen_unix:        "
        f"{result.get('first_seen_unix', 0)}",
        f"  last_seen_unix:         "
        f"{result.get('last_seen_unix', 0)}",
    ])


_MARKETPLACE_REPUTATION_ACTIONS = {"list", "lookup"}


async def handle_prsm_marketplace_reputation(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 275 — operator visibility into ReputationTracker
    state. action=list returns score-sorted provider table;
    action=lookup returns single-provider detail incl slash
    events."""
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return (
            f"Missing required 'action' (must be one of "
            f"{sorted(_MARKETPLACE_REPUTATION_ACTIONS)})."
        )
    if action not in _MARKETPLACE_REPUTATION_ACTIONS:
        return (
            f"action must be one of "
            f"{sorted(_MARKETPLACE_REPUTATION_ACTIONS)}; "
            f"got {action!r}."
        )

    if action == "list":
        limit = int(arguments.get("limit", 100))
        path = f"/marketplace/reputation?limit={limit}"
        try:
            result = await _call_node_api("GET", path)
        except Exception as e:
            return (
                f"prsm_marketplace_reputation failed: {e}\n"
                f"Is your PRSM node running? (prsm node start)"
            )
        if "providers" not in result:
            detail = result.get("detail", "unknown error")
            if "not initialized" in str(detail).lower():
                return (
                    f"Reputation tracker not wired on this "
                    f"node.\n  Detail: {detail}\n"
                    f"  Tracker is built when QueryOrchestrator "
                    f"wires; ensure marketplace path is active."
                )
            return f"list refused: {detail}"
        providers = result.get("providers") or []
        total = result.get("count", 0)
        lines = [
            f"PRSM Marketplace Reputation — "
            f"{len(providers)} of {total} known providers "
            f"(score desc):",
        ]
        if not providers:
            lines.append("  (none)")
        for p in providers:
            slash_marker = (
                "⚠" if p.get("has_been_slashed") else " "
            )
            score = p.get("score", 0.0)
            lines.append(
                f"  {slash_marker} {p.get('provider_id', '?')}  "
                f"score={score:.3f}  "
                f"ok/fail/pre/slash="
                f"{p.get('successes', 0)}/"
                f"{p.get('failures', 0)}/"
                f"{p.get('preempted', 0)}/"
                f"{p.get('slashed_count', 0)}  "
                f"p50={p.get('latency_p50_ms')}ms  "
                f"p95={p.get('latency_p95_ms')}ms"
            )
        return "\n".join(lines)

    # action == "lookup"
    provider_id = (arguments.get("provider_id") or "").strip()
    if not provider_id:
        return "lookup requires 'provider_id'."
    try:
        result = await _call_node_api(
            "GET", f"/marketplace/reputation/{provider_id}",
        )
    except Exception as e:
        return (
            f"prsm_marketplace_reputation failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "provider_id" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in str(detail).lower():
            return (
                f"Reputation tracker not wired.\n"
                f"  Detail: {detail}"
            )
        return f"lookup refused: {detail}"
    known = result.get("known", False)
    cold_start = (
        " (cold-start — < MIN_SAMPLES_FOR_SCORE observations)"
        if (not known) or
        (result.get("successes", 0) + result.get("failures", 0) < 10)
        else ""
    )
    lines = [
        f"Marketplace Reputation — {result['provider_id']}:",
        f"  known:           {known}",
        f"  score:           {result.get('score', 0.0):.3f}"
        f"{cold_start}",
        f"  successes:       {result.get('successes', 0)}",
        f"  failures:        {result.get('failures', 0)}",
        f"  preempted:       {result.get('preempted', 0)}",
        f"  slashed_count:   {result.get('slashed_count', 0)}",
        f"  has_been_slashed:{result.get('has_been_slashed')}",
        f"  latency_p50_ms:  {result.get('latency_p50_ms')}",
        f"  latency_p95_ms:  {result.get('latency_p95_ms')}",
        f"  first_seen_unix: {result.get('first_seen_unix', 0)}",
        f"  last_seen_unix:  {result.get('last_seen_unix', 0)}",
    ]
    slash_events = result.get("slash_events") or []
    if slash_events:
        lines.append(f"  slash_events ({len(slash_events)}):")
        for s in slash_events:
            lines.append(
                f"    batch={s.get('batch_id', '?')}  "
                f"reason={s.get('reason', '?')}  "
                f"wei={s.get('slash_amount_wei', 0)}  "
                f"tx={s.get('tx_hash', '?')}"
            )
    return "\n".join(lines)


async def handle_prsm_pinned_stats(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 267 — render per-pinned-content storage challenge
    stats. Backed by GET /storage/pinned-stats."""
    try:
        result = await _call_node_api("GET", "/storage/pinned-stats")
    except Exception as e:
        return (
            f"prsm_pinned_stats failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "pinned" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in str(detail).lower():
            return (
                f"Storage provider not wired on this node.\n"
                f"  Detail: {detail}"
            )
        return f"prsm_pinned_stats refused: {detail}"
    pinned = result.get("pinned") or []
    count = result.get("count", len(pinned))
    if not pinned:
        return f"No pinned content (count={count})."
    lines = [f"PRSM Pinned Content Stats (count={count}):"]
    for p in pinned:
        cid = (p.get("cid") or "?")[:18]
        succ = p.get("successful_challenges", 0)
        fail = p.get("failed_challenges", 0)
        last_ver = p.get("last_verified")
        verified = (
            f"verified={last_ver}" if last_ver else "verified=NEVER"
        )
        lines.append(
            f"  {cid:<18}  "
            f"size={p.get('size_bytes', 0)}B  "
            f"req={(p.get('requester_id') or '?')[:12]}  "
            f"challenges={succ}/{succ+fail}  "
            f"{verified}"
        )
    return "\n".join(lines)


async def handle_prsm_provider_reputations(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 267 — render cross-provider reputation + challenge
    counts. Backed by GET /storage/provider-reputations."""
    try:
        result = await _call_node_api(
            "GET", "/storage/provider-reputations",
        )
    except Exception as e:
        return (
            f"prsm_provider_reputations failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "providers" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in str(detail).lower():
            return (
                f"Storage provider not wired on this node.\n"
                f"  Detail: {detail}"
            )
        return f"prsm_provider_reputations refused: {detail}"
    providers = result.get("providers") or {}
    count = result.get("count", len(providers))
    if not providers:
        return f"No provider reputation data yet (count={count})."
    # Sort by reputation descending so most-trusted first.
    items = sorted(
        providers.items(),
        key=lambda kv: kv[1].get("reputation", 0),
        reverse=True,
    )
    lines = [f"PRSM Provider Reputations (count={count}):"]
    for pid, stats in items:
        rep = stats.get("reputation", 0.0)
        total = stats.get("total_challenges", 0)
        succ = stats.get("successful_proofs", 0)
        fail = stats.get("failed_proofs", 0)
        expired = stats.get("expired_challenges", 0)
        lines.append(
            f"  {pid[:16]:<16}  "
            f"rep={rep:.3f}  "
            f"challenges={total}  "
            f"({succ} ok / {fail} fail / {expired} expired)"
        )
    return "\n".join(lines)


async def handle_prsm_bootstrap_status(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 266 / Sprint 325 — render bootstrap status for triage.

    Auto-detects payload shape:
      - Libp2pDiscovery (sprint 164 canonical default) — uses
        `connected`/`degraded`/`client_state` + sprint-324
        cumulative counters
      - Legacy PeerDiscovery — uses `connected_count` /
        `degraded_mode` / `success_node` etc.

    Pre-sprint-325 the handler hit `"connected_count" not in
    result` and emitted "Peer discovery not wired" — confusingly
    wrong on a healthy canonical-wired node.

    Pairs with prsm_peers for full network visibility.
    """
    try:
        result = await _call_node_api("GET", "/bootstrap/status")
    except Exception as e:
        return (
            f"prsm_bootstrap_status failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )

    # Sprint 325 — detect not-initialized state first
    if "detail" in result and not (
        "connected_count" in result or "client_state" in result
        or "peer_join_events" in result
    ):
        detail = result.get("detail", "unknown error")
        if "not initialized" in str(detail).lower():
            return (
                f"Peer discovery not wired on this node.\n"
                f"  Detail: {detail}"
            )
        return f"prsm_bootstrap_status refused: {detail}"

    # Sprint 325 — Libp2pDiscovery shape detection. The
    # canonical default since sprint 164. Recognizes by
    # presence of `client_state` (sprint 324) or any of the
    # cumulative counters.
    is_libp2p_shape = (
        "client_state" in result
        or "peer_join_events" in result
        or "stale_evictions" in result
    )
    if is_libp2p_shape:
        connected = int(result.get("connected", 0) or 0)
        degraded = bool(result.get("degraded", False))
        client_state = result.get("client_state", "?")
        if connected > 0 and not degraded and (
            client_state == "connected"
        ):
            health_marker = "✓ healthy"
        elif degraded or client_state == "dead":
            health_marker = "⚠ degraded"
        else:
            health_marker = "⚠ disconnected"
        lines = [
            f"PRSM Bootstrap Status — {health_marker}",
            f"  client_state:           {client_state}",
            f"  connected:              {connected}",
            f"  degraded:               {degraded}",
            f"  attempted:              "
            f"{result.get('attempted', 0)}",
            f"  discovered_peer_count:  "
            f"{result.get('discovered_peer_count', 0)}",
            f"  peer_join_events:       "
            f"{result.get('peer_join_events', 0)}",
            f"  peer_leave_events:      "
            f"{result.get('peer_leave_events', 0)}",
            f"  stale_evictions:        "
            f"{result.get('stale_evictions', 0)}",
            f"  reconnect_attempts:     "
            f"{result.get('reconnect_attempts', 0)}",
            f"  reconnect_successes:    "
            f"{result.get('reconnect_successes', 0)}",
        ]
        # Sprint 375 — surface the active bootstrap URL +
        # fallback config so operators see SPOF posture at
        # a glance.
        active_url = result.get("active_url")
        if active_url is not None:
            lines.append(f"  active_url:             {active_url}")
        bnodes = result.get("bootstrap_nodes") or []
        if bnodes:
            lines.append(f"  bootstrap_nodes ({len(bnodes)}):")
            for n in bnodes:
                lines.append(f"    {n}")
        fb_enabled = result.get(
            "bootstrap_fallback_enabled",
        )
        fb_nodes = result.get(
            "bootstrap_fallback_nodes",
        ) or []
        if fb_enabled is not None:
            lines.append(
                f"  fallback_enabled:       {fb_enabled}"
            )
        if fb_nodes:
            lines.append(
                f"  bootstrap_fallback_nodes "
                f"({len(fb_nodes)}):"
            )
            for n in fb_nodes:
                lines.append(f"    {n}")
        return "\n".join(lines)

    # Legacy PeerDiscovery shape
    if "connected_count" not in result:
        detail = result.get("detail", "unknown error")
        return f"prsm_bootstrap_status refused: {detail}"
    connected = result.get("connected_count", 0)
    degraded = result.get("degraded_mode", False)
    success = result.get("success_node") or "(none)"
    health_marker = "✓ healthy" if (
        connected > 0 and not degraded
    ) else (
        "⚠ degraded" if degraded else "⚠ disconnected"
    )
    lines = [
        f"PRSM Bootstrap Status — {health_marker}",
        f"  connected_count:        {connected}",
        f"  degraded_mode:          {degraded}",
        f"  success_node:           {success}",
        f"  retry_attempts:         {result.get('retry_attempts', 0)}",
        f"  bootstrap_client_active: "
        f"{result.get('bootstrap_client_active', False)}",
        f"  fallback_enabled:       "
        f"{result.get('fallback_enabled', False)}",
        f"  fallback_activated:     "
        f"{result.get('fallback_activated', False)}",
        f"  fallback_succeeded:     "
        f"{result.get('fallback_succeeded', False)}",
        f"  addresses_rejected:     "
        f"{result.get('addresses_rejected', 0)}",
        f"  source_policy:          "
        f"{result.get('source_policy', '?')}",
    ]
    configured = result.get("configured_nodes") or []
    failed = result.get("failed_nodes") or []
    if configured:
        lines.append(f"  configured_nodes ({len(configured)}):")
        for n in configured:
            lines.append(f"    {n}")
    if failed:
        lines.append(f"  ⚠ failed_nodes ({len(failed)}):")
        for n in failed:
            lines.append(f"    {n}")
    return "\n".join(lines)


async def handle_prsm_bootstrap_server_status(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 391 — probe a bootstrap *server* you are
    running (your droplet, not a canonical fleet node).

    AI-assisted complement to sprint-390's `prsm
    bootstrap-server status` CLI. Same probe core
    (prsm.cli_helpers.bootstrap_server_probe.
    fetch_server_status), MCP-rendered.
    """
    from prsm.cli_helpers import bootstrap_server_probe as bsp_module

    host = arguments.get("host") or "127.0.0.1"
    port = int(arguments.get("port") or 8000)
    timeout = float(arguments.get("timeout", 5.0))
    include_subsystems = bool(
        arguments.get("include_subsystems", False)
    )

    try:
        probe = await bsp_module.fetch_server_status(
            host=host, port=port, timeout_seconds=timeout,
            include_subsystems=include_subsystems,
        )
    except Exception as e:  # noqa: BLE001
        return (
            f"prsm_bootstrap_server_status failed: "
            f"{type(e).__name__}: {e}"
        )

    status_markers = {
        bsp_module.ProbeStatus.OK: "✅ healthy",
        bsp_module.ProbeStatus.PARTIAL: (
            "⚠ partial — metrics unavailable"
        ),
        bsp_module.ProbeStatus.CONNECT_FAIL: (
            "❌ connect refused"
        ),
        bsp_module.ProbeStatus.TIMEOUT: "❌ timeout",
        bsp_module.ProbeStatus.HTTP_ERROR: "❌ http error",
        bsp_module.ProbeStatus.UNKNOWN: "❌ unknown",
    }
    marker = status_markers.get(probe.status, "❌ ?")

    lines = [
        f"PRSM Bootstrap Server Status — {marker}",
        f"  target: {probe.host}:{probe.port}",
    ]
    if probe.error:
        lines.append(f"  error: {probe.error}")

    if probe.health:
        lines.append("")
        lines.append("Health:")
        for k, v in probe.health.items():
            lines.append(f"  {k}: {v}")

    if probe.metrics:
        lines.append("")
        lines.append("Metrics:")
        flat = {
            k: v for k, v in probe.metrics.items()
            if not isinstance(v, dict)
        }
        labeled = {
            k: v for k, v in probe.metrics.items()
            if isinstance(v, dict)
        }
        for k, v in flat.items():
            lines.append(f"  {k}: {v}")
        for k, label_dict in labeled.items():
            if not label_dict:
                continue
            lines.append(f"  {k}:")
            for label, value in label_dict.items():
                lines.append(f"    {label}: {value}")

    if probe.health_detailed:
        agg = probe.health_detailed.get("status", "?")
        agg_marker = {
            "healthy": "✅",
            "degraded": "⚠",
            "unhealthy": "❌",
        }.get(agg, "?")
        lines.append("")
        lines.append(
            f"Subsystems — aggregate: {agg_marker} {agg}"
        )
        for sub_name, sub_data in (
            probe.health_detailed.get("subsystems") or {}
        ).items():
            sub_status = sub_data.get("status", "?")
            sub_marker = {
                "healthy": "✅",
                "degraded": "⚠",
                "stale": "❌",
                # Sprint 405 — "disabled" surfaces sprint-397
                # config-opt-out state (e.g., federation_sync
                # when federation_peers is empty). Distinct
                # from healthy (it's not running) AND from
                # alert states (operator chose this).
                # Non-alarming marker.
                "disabled": "○",
            }.get(sub_status, "?")
            age = sub_data.get("last_heartbeat_age_seconds")
            age_str = (
                f"{age:.0f}s"
                if isinstance(age, (int, float)) else "—"
            )
            lines.append(
                f"  {sub_marker} {sub_name}: "
                f"{sub_status} (age {age_str})"
            )

    return "\n".join(lines)


async def handle_prsm_bootstrap_test(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 387 — probe canonical bootstrap fleet from
    this node's vantage point.

    AI-assisted complement to sprint-385's `prsm node
    bootstrap-test` CLI. Same probe surface (prsm.cli_
    helpers.bootstrap_probe.probe_fleet), MCP-rendered.
    Doesn't require a running PRSM node — probes directly
    from the MCP server host.

    Diagnostic for: 'is my regional bootstrap up, or is
    something local blocking me?'
    """
    from prsm.cli_helpers.bootstrap_probe import (
        ProbeStatus,
        canonical_bootstrap_urls,
        probe_fleet,
    )

    urls = arguments.get("urls") or []
    timeout = float(arguments.get("timeout", 10.0))

    if urls:
        target_urls = list(urls)
    else:
        target_urls = canonical_bootstrap_urls()

    if not target_urls:
        return (
            "prsm_bootstrap_test: no URLs to probe. "
            "Pass urls=[\"wss://...\"] or check that "
            "BOOTSTRAP_PRIMARY / BOOTSTRAP_FALLBACK_* env "
            "vars resolve."
        )

    try:
        fleet = await probe_fleet(
            target_urls, timeout_seconds=timeout,
        )
    except Exception as e:  # noqa: BLE001
        return (
            f"prsm_bootstrap_test failed: "
            f"{type(e).__name__}: {e}"
        )

    # Header marker
    if fleet.all_healthy:
        marker = "✅ all healthy"
    elif fleet.any_healthy:
        marker = "⚠ partial"
    else:
        marker = "❌ all degraded"

    lines = [
        (
            f"PRSM Bootstrap Fleet Probe — {marker} "
            f"({fleet.healthy_count}/{fleet.total_count} "
            f"reachable)"
        ),
        "",
    ]

    for h in fleet.hosts:
        if h.status == ProbeStatus.OK:
            status_str = "✅ ok"
        elif h.status == ProbeStatus.TIMEOUT:
            status_str = "⏱ timeout"
        else:
            status_str = f"❌ {h.status.value}"
        url_short = h.url
        if "://" in url_short:
            url_short = url_short.split("://", 1)[1]
        latency_str = (
            f"{h.latency_ms:.0f}ms"
            if h.latency_ms is not None else "-"
        )
        lines.append(
            f"  {status_str}  {url_short}  ({latency_str})"
        )
        # Per-layer breakdown
        layer_marks = []
        for label, ok in (
            ("TCP", h.tcp_ok),
            ("TLS", h.tls_ok),
            ("WSS", h.wss_ok),
        ):
            layer_marks.append(
                f"✓{label}" if ok else f"·{label}"
            )
        layer_line = "      " + " ".join(layer_marks)
        if h.cert_subject:
            layer_line += (
                f"  (cert: {h.cert_subject}"
                f" / issued by {h.cert_issuer})"
            )
        lines.append(layer_line)
        if h.error:
            lines.append(f"      error: {h.error}")

    return "\n".join(lines)


async def handle_prsm_royalty_dispatch_summary(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 265 — render the aggregate on-chain royalty
    dispatch summary. Symmetric to prsm_royalty_dispatch_history
    (paginated rows) but at aggregate level. Backed by GET
    /admin/royalty-dispatch-summary."""
    try:
        result = await _call_node_api(
            "GET", "/admin/royalty-dispatch-summary",
        )
    except Exception as e:
        return (
            f"prsm_royalty_dispatch_summary failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "total" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in str(detail).lower():
            return (
                f"Royalty dispatch ring not wired on this node.\n"
                f"  Detail: {detail}\n"
                f"  Enable on-chain dispatch via "
                f"PRSM_ONCHAIN_CONTENT_ROYALTY_ENABLED=1."
            )
        return f"prsm_royalty_dispatch_summary refused: {detail}"
    total = result.get("total", 0)
    if total == 0:
        return (
            f"No royalty dispatch outcomes recorded "
            f"(total={total}).\n"
            f"  Enable PRSM_ONCHAIN_CONTENT_ROYALTY_ENABLED=1 + "
            f"run a forge query."
        )
    lines = [
        f"PRSM On-Chain Royalty Dispatch Summary "
        f"(total={total}):",
        f"  total_sent_wei:    {result.get('total_sent_wei', 0)}",
    ]
    earliest = result.get("earliest_ts")
    latest = result.get("latest_ts")
    if earliest is not None and latest is not None:
        lines.append(f"  earliest_ts:       {earliest}")
        lines.append(f"  latest_ts:         {latest}")
    sc = result.get("status_counts") or {}
    if sc:
        lines.append("  status_counts:")
        for k in sorted(sc.keys()):
            lines.append(f"    {k:<24} {sc[k]}")
    bm = result.get("by_allocation_mode") or {}
    if bm:
        lines.append("  by_allocation_mode:")
        for k in sorted(bm.keys()):
            lines.append(f"    {k:<16} {bm[k]}")
    return "\n".join(lines)


async def handle_prsm_royalty_dispatch_history(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 249 — render the on-chain content-royalty dispatch
    audit ring. Backed by GET /admin/royalty-dispatch-history."""
    raw_limit = arguments.get("limit", 20)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        return f"limit must be an integer; got {raw_limit!r}."
    if limit < 1 or limit > 1000:
        return f"limit must be in [1, 1000]; got {limit}."
    raw_offset = arguments.get("offset", 0)
    try:
        offset = int(raw_offset)
    except (TypeError, ValueError):
        return f"offset must be an integer; got {raw_offset!r}."
    if offset < 0:
        return f"offset must be >= 0; got {offset}."

    parts = [f"limit={limit}", f"offset={offset}"]
    if arguments.get("status"):
        parts.append(f"status={arguments['status']}")
    if arguments.get("job_id"):
        parts.append(f"job_id={arguments['job_id']}")
    if arguments.get("allocation_mode"):
        parts.append(
            f"allocation_mode={arguments['allocation_mode']}"
        )
    path = (
        "/admin/royalty-dispatch-history?" + "&".join(parts)
    )
    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"prsm_royalty_dispatch_history failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "entries" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in str(detail).lower():
            return (
                f"Royalty dispatch ring not wired on this node.\n"
                f"  Detail: {detail}\n"
                f"  Enable on-chain dispatch via "
                f"PRSM_ONCHAIN_CONTENT_ROYALTY_ENABLED=1."
            )
        return f"prsm_royalty_dispatch_history refused: {detail}"
    entries = result.get("entries") or []
    total = result.get("total", len(entries))
    if not entries:
        return (
            f"No royalty dispatch outcomes recorded "
            f"(total={total}).\n"
            f"  Enable on-chain dispatch via "
            f"PRSM_ONCHAIN_CONTENT_ROYALTY_ENABLED=1, then run a "
            f"forge query."
        )
    lines = [
        f"PRSM On-Chain Royalty Dispatch History "
        f"(showing {offset+1}–{offset+len(entries)} of {total}):",
    ]
    for e in entries:
        cid_disp = (e.get("cid") or "?")[:14]
        tx_disp = (e.get("tx_hash") or "—")[:14]
        err = e.get("error")
        err_part = f"  err={err}" if err else ""
        mode = e.get("allocation_mode")
        mode_part = f"  mode={mode}" if mode else ""
        lines.append(
            f"  job={e.get('job_id', '?')[:14]:<14}  "
            f"cid={cid_disp:<14}  "
            f"status={e.get('status', '?'):<22}  "
            f"tx={tx_disp:<14}  "
            f"wei={e.get('gross_wei', 0)}"
            f"{mode_part}"
            f"{err_part}"
        )
    return "\n".join(lines)


async def handle_prsm_receipts_list(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 250 — paginated list of stored InferenceReceipts.

    Backed by GET /compute/receipts. Pair with prsm_receipt for
    deep-dive on a specific job_id, or prsm_verify_receipt to
    validate signatures."""
    raw_limit = arguments.get("limit", 20)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        return f"limit must be an integer; got {raw_limit!r}."
    if limit < 1 or limit > 1000:
        return f"limit must be in [1, 1000]; got {limit}."
    raw_offset = arguments.get("offset", 0)
    try:
        offset = int(raw_offset)
    except (TypeError, ValueError):
        return f"offset must be an integer; got {raw_offset!r}."
    if offset < 0:
        return f"offset must be >= 0; got {offset}."

    parts = [f"limit={limit}", f"offset={offset}"]
    if arguments.get("model_id"):
        parts.append(f"model_id={arguments['model_id']}")
    path = "/compute/receipts?" + "&".join(parts)
    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"prsm_receipts_list failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "receipts" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in str(detail).lower():
            return (
                f"Receipt store not wired on this node.\n"
                f"  Detail: {detail}\n"
                f"  Set PRSM_RECEIPT_STORE_DIR to persist "
                f"receipts across restarts."
            )
        return f"prsm_receipts_list refused: {detail}"
    receipts = result.get("receipts") or []
    total = result.get("total", len(receipts))
    if not receipts:
        return (
            f"No stored receipts (total={total}). Run "
            f"prsm_inference to populate."
        )
    lines = [
        f"PRSM Stored InferenceReceipts "
        f"(showing {offset+1}–{offset+len(receipts)} of {total}):",
    ]
    for r in receipts:
        lines.append(
            f"  job={r.get('job_id', '?')[:16]:<16}  "
            f"model={r.get('model_id', '?'):<22}  "
            f"cost={r.get('cost_ftns', '?')} FTNS  "
            f"settler={r.get('settler_node_id', '?')[:14]}"
        )
    lines.append(
        "  Use prsm_receipt <job_id> for full record, "
        "prsm_verify_receipt to validate signature."
    )
    return "\n".join(lines)


async def handle_prsm_receipt(arguments: Dict[str, Any]) -> str:
    """Sprint 242 — fetch a stored InferenceReceipt by job_id.

    Returns the receipt as a formatted text block so end-users +
    auditors can verify what was signed even when they didn't
    save the original /compute/inference response."""
    job_id = (arguments.get("job_id") or "").strip()
    if not job_id:
        return "Missing required 'job_id' (non-empty)."
    try:
        result = await _call_node_api(
            "GET", f"/compute/receipt/{job_id}",
        )
    except Exception as e:
        return (
            f"prsm_receipt failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "job_id" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in str(detail).lower():
            return (
                f"Receipt store not wired on this node.\n"
                f"  Detail: {detail}\n"
                f"  Set PRSM_RECEIPT_STORE_DIR for filesystem "
                f"persistence."
            )
        if "no receipt" in str(detail).lower():
            return f"No receipt found for job_id={job_id!r}."
        return f"Receipt lookup refused: {detail}"
    return (
        f"PRSM InferenceReceipt:\n"
        f"  job_id:           {result.get('job_id', '?')}\n"
        f"  request_id:       {result.get('request_id', '?')}\n"
        f"  model_id:         {result.get('model_id', '?')}\n"
        f"  privacy_tier:     {result.get('privacy_tier', '?')}\n"
        f"  content_tier:     {result.get('content_tier', '?')}\n"
        f"  tee_type:         {result.get('tee_type', '?')}\n"
        f"  cost_ftns:        {result.get('cost_ftns', '?')}\n"
        f"  settler_node_id:  {result.get('settler_node_id', '?')}\n"
        f"  Use prsm_verify_receipt to validate the signature."
    )


async def handle_prsm_pubkey(arguments: Dict[str, Any]) -> str:
    """Sprint 241 — render GET /node/identity/pubkey."""
    try:
        result = await _call_node_api("GET", "/node/identity/pubkey")
    except Exception as e:
        return (
            f"prsm_pubkey failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "public_key_b64" not in result:
        return f"Pubkey lookup refused: {result.get('detail', '?')}"
    return (
        f"PRSM Node Pubkey:\n"
        f"  node_id:        {result.get('node_id', '?')}\n"
        f"  public_key_b64: {result.get('public_key_b64', '?')}"
    )


async def handle_prsm_verify_receipt(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 241 — verify an InferenceReceipt's Ed25519
    signature. Caller passes the receipt as a dict (the same
    shape /compute/inference returns). public_key_b64 may be
    supplied directly OR fetched from /node/identity/pubkey when
    the running node IS the settler."""
    receipt = arguments.get("receipt")
    if not isinstance(receipt, dict):
        return "Missing required 'receipt' (must be a dict)."

    # Determine pubkey: supplied → use that. Otherwise fetch from
    # /node/identity/pubkey and compare node_id to receipt
    # settler_node_id (sanity check).
    pubkey_b64 = arguments.get("public_key_b64")
    if not pubkey_b64:
        try:
            node_pub = await _call_node_api(
                "GET", "/node/identity/pubkey",
            )
        except Exception as e:
            return (
                f"prsm_verify_receipt failed: cannot fetch pubkey: "
                f"{e}\nSupply public_key_b64 explicitly OR ensure "
                f"the running node is the settler."
            )
        if "public_key_b64" not in node_pub:
            return (
                f"Pubkey fetch refused: "
                f"{node_pub.get('detail', '?')}"
            )
        if (
            node_pub.get("node_id") != receipt.get("settler_node_id")
        ):
            return (
                f"settler_node_id mismatch: receipt settler="
                f"{receipt.get('settler_node_id', '?')!r} but "
                f"running node="
                f"{node_pub.get('node_id', '?')!r}.\n"
                f"Supply public_key_b64 of the actual settler."
            )
        pubkey_b64 = node_pub["public_key_b64"]

    try:
        from prsm.compute.inference.models import InferenceReceipt
        from prsm.compute.inference.receipt import verify_receipt
        rec = InferenceReceipt.from_dict(receipt)
    except Exception as e:
        return f"Receipt parse failed: {e}"

    try:
        ok = verify_receipt(rec, public_key_b64=pubkey_b64)
    except Exception as e:  # noqa: BLE001
        return f"Verification error: {e}"

    if not ok:
        return (
            f"SIGNATURE INVALID for receipt job_id="
            f"{rec.job_id!r}.\nReceipt fields may have been "
            f"tampered, or the pubkey does not match the settler."
        )
    return (
        f"SIGNATURE VALID for receipt:\n"
        f"  job_id:           {rec.job_id}\n"
        f"  request_id:       {rec.request_id}\n"
        f"  model_id:         {rec.model_id}\n"
        f"  privacy_tier:     {rec.privacy_tier}\n"
        f"  content_tier:     {rec.content_tier}\n"
        f"  tee_type:         {rec.tee_type}\n"
        f"  cost_ftns:        {rec.cost_ftns}\n"
        f"  settler_node_id:  {rec.settler_node_id}"
    )


async def handle_prsm_models(arguments: Dict[str, Any]) -> str:
    """Sprint 235 — list inference model_ids the executor accepts."""
    try:
        result = await _call_node_api("GET", "/compute/models")
    except Exception as e:
        return (
            f"prsm_models failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    models = result.get("models") or []
    count = result.get("count", len(models))
    if not models:
        return (
            f"No models registered on this node (count={count}).\n"
            f"  Detail: {result.get('detail', 'executor wired but registry empty')}"
        )
    lines = [f"PRSM Available Models (count={count}):"]
    for m in models:
        lines.append(f"  • {m}")
    lines.append(
        "  Use any model_id above with prsm_inference / prsm_quote."
    )
    return "\n".join(lines)


async def handle_prsm_ledger_sync(arguments: Dict[str, Any]) -> str:
    """Sprint 233 — render GET /ledger/sync/stats."""
    try:
        result = await _call_node_api("GET", "/ledger/sync/stats")
    except Exception as e:
        return (
            f"prsm_ledger_sync failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if not isinstance(result, dict) or not result:
        return "Ledger sync stats: (empty)"
    lines = ["PRSM Ledger Sync Stats:"]
    for k, v in result.items():
        lines.append(f"  {k:<26} {v}")
    return "\n".join(lines)


_RESOURCE_UPDATE_FIELDS = (
    "cpu_allocation_pct", "memory_allocation_pct",
    "storage_gb", "max_concurrent_jobs",
    "gpu_allocation_pct",
    "upload_mbps_limit", "download_mbps_limit",
    "active_hours_start", "active_hours_end",
    "active_days",
)


async def handle_prsm_node_resources(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 232 — get or update node resource configuration."""
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return "Missing required 'action' (get or update)."
    if action not in ("get", "update"):
        return f"action must be get or update; got {action!r}."
    if action == "get":
        try:
            result = await _call_node_api("GET", "/node/resources")
        except Exception as e:
            return (
                f"prsm_node_resources failed: {e}\n"
                f"Is your PRSM node running? (prsm node start)"
            )
        if not isinstance(result, dict):
            return "Node resources: (unexpected response shape)"
        lines = ["PRSM Node Resources:"]
        for k, v in result.items():
            lines.append(f"  {k:<26} {v}")
        return "\n".join(lines)
    # update
    body = {
        k: arguments[k]
        for k in _RESOURCE_UPDATE_FIELDS
        if k in arguments and arguments[k] is not None
    }
    if not body:
        return (
            "update requires at least one field. Valid: "
            + ", ".join(_RESOURCE_UPDATE_FIELDS)
            + "."
        )
    try:
        result = await _call_node_api("PUT", "/node/resources", body)
    except Exception as e:
        return (
            f"prsm_node_resources failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if not isinstance(result, dict):
        return f"Update returned: {result}"
    lines = ["PRSM Node Resources Updated:"]
    for k, v in result.items():
        lines.append(f"  {k:<26} {v}")
    return "\n".join(lines)


async def handle_prsm_settlement_view(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 231 — settlement views: pending / history / flush."""
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return (
            "Missing required 'action' (pending, history, or flush)."
        )
    if action not in ("pending", "history", "flush"):
        return (
            f"action must be pending, history, or flush; "
            f"got {action!r}."
        )
    if action == "pending":
        method = "GET"
        path = "/settlement/pending"
    elif action == "history":
        raw_limit = arguments.get("limit", 10)
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            return f"limit must be an integer; got {raw_limit!r}."
        if limit < 1 or limit > 200:
            return f"limit must be in [1, 200]; got {limit}."
        method = "GET"
        path = f"/settlement/history?limit={limit}"
    else:
        method = "POST"
        path = "/settlement/flush"
    try:
        result = await _call_node_api(method, path)
    except Exception as e:
        return (
            f"prsm_settlement_view failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if action == "pending":
        items = result.get("pending") or []
        count = result.get("count", len(items))
        if not items:
            return f"No pending settlement transfers (count={count})."
        lines = [f"PRSM Pending Settlements (count={count}):"]
        for it in items:
            lines.append(f"  {it}")
        return "\n".join(lines)
    if action == "history":
        items = result.get("history") or []
        count = result.get("count", len(items))
        if not items:
            return f"No settlement history (count={count})."
        lines = [f"PRSM Settlement History (count={count}):"]
        for it in items:
            lines.append(f"  {it}")
        return "\n".join(lines)
    # flush
    if "settled_count" not in result:
        detail = result.get("detail", "unknown error")
        return f"Settlement flush refused: {detail}"
    return (
        f"Settlement flush executed.\n"
        f"  settled_count:    {result.get('settled_count', 0)}\n"
        f"  total_amount:     {result.get('total_amount', 0)} FTNS\n"
        f"  net_transfers:    {result.get('net_transfers', 0)}\n"
        f"  tx_hashes:        {result.get('tx_hashes', [])}\n"
        f"  errors:           {result.get('errors', [])}\n"
        f"  duration_seconds: {result.get('duration_seconds', 0)}"
    )


async def handle_prsm_bridge_history(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 230 — bridge read views: status, list, lookup."""
    view = (arguments.get("view") or "").strip().lower()
    if view not in ("status", "list", "lookup"):
        return (
            f"view must be one of status/list/lookup; got {view!r}."
        )
    if view == "status":
        path = "/bridge/status"
    elif view == "list":
        raw_limit = arguments.get("limit", 20)
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            return f"limit must be an integer; got {raw_limit!r}."
        if limit < 1 or limit > 200:
            return f"limit must be in [1, 200]; got {limit}."
        path = f"/bridge/transactions?limit={limit}"
    else:
        tx_id = (arguments.get("tx_id") or "").strip()
        if not tx_id:
            return "lookup requires 'tx_id' (non-empty)."
        path = f"/bridge/transactions/{tx_id}"
    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"prsm_bridge_history failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if view == "status":
        if not isinstance(result, dict):
            return "Bridge status: (unexpected response shape)"
        lines = ["PRSM Bridge Status:"]
        for k, v in result.items():
            lines.append(f"  {k:<24} {v}")
        return "\n".join(lines)
    if view == "list":
        txs = result.get("transactions") or []
        count = result.get("count", len(txs))
        if not txs:
            return f"No bridge transactions (count={count})."
        lines = [f"PRSM Bridge Transactions (count={count}):"]
        for t in txs:
            lines.append(
                f"  {t.get('transaction_id', '?')[:16]:<16}  "
                f"dir={t.get('direction', '?'):<8}  "
                f"amount={t.get('amount', '?')}  "
                f"status={t.get('status', '?')}"
            )
        return "\n".join(lines)
    # lookup
    if "transaction_id" not in result:
        detail = result.get("detail", "unknown error")
        if "not found" in str(detail).lower():
            return f"Bridge transaction {arguments.get('tx_id')} not found."
        return f"Bridge lookup refused: {detail}"
    lines = [f"PRSM Bridge Transaction {result.get('transaction_id', '?')}:"]
    for k, v in result.items():
        if k == "transaction_id":
            continue
        lines.append(f"  {k:<22} {v}")
    return "\n".join(lines)


async def handle_prsm_stake_lookup(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 229 — single stake or unstake-request lookup."""
    kind = (arguments.get("kind") or "").strip().lower()
    if kind not in ("stake", "unstake_request"):
        return (
            f"kind must be 'stake' or 'unstake_request'; "
            f"got {kind!r}."
        )
    ident = (arguments.get("id") or "").strip()
    if not ident:
        return "Missing required 'id' (non-empty)."
    if kind == "stake":
        path = f"/staking/stakes/{ident}"
    else:
        path = f"/staking/unstake-requests/{ident}"
    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"prsm_stake_lookup failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    key = "stake_id" if kind == "stake" else "request_id"
    if key not in result:
        detail = result.get("detail", "unknown error")
        if "not found" in detail.lower():
            return f"{kind} {ident} not found."
        return f"{kind} lookup refused: {detail}"
    title = "Stake" if kind == "stake" else "Unstake Request"
    lines = [f"PRSM {title} {result.get(key, ident)}:"]
    for k, v in result.items():
        if k == key:
            continue
        lines.append(f"  {k:<24} {v}")
    return "\n".join(lines)


async def handle_prsm_get_agent(arguments: Dict[str, Any]) -> str:
    """Sprint 228 — render full agent record for a given id.

    Backed by GET /agents/{agent_id} which includes allowance
    from the ledger (when wired). Distinct from prsm_agents
    (list/search short-form)."""
    agent_id = (arguments.get("agent_id") or "").strip()
    if not agent_id:
        return "Missing required 'agent_id' (non-empty)."
    try:
        result = await _call_node_api("GET", f"/agents/{agent_id}")
    except Exception as e:
        return (
            f"prsm_get_agent failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "agent_id" not in result:
        detail = result.get("detail", "unknown error")
        if "not found" in detail.lower():
            return f"Agent {agent_id} not found."
        return f"Agent lookup refused: {detail}"
    lines = [f"PRSM Agent {result.get('agent_id', agent_id)}:"]
    for k, v in result.items():
        if k == "agent_id":
            continue
        lines.append(f"  {k:<16} {v}")
    return "\n".join(lines)


async def handle_prsm_agent_conversations(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 227 — render recent conversation threads for an
    agent. Backed by GET /agents/{agent_id}/conversations."""
    agent_id = (arguments.get("agent_id") or "").strip()
    if not agent_id:
        return "Missing required 'agent_id' (non-empty)."
    raw_limit = arguments.get("limit", 10)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        return f"limit must be an integer; got {raw_limit!r}."
    if limit < 1 or limit > 100:
        return f"limit must be in [1, 100]; got {limit}."
    try:
        result = await _call_node_api(
            "GET",
            f"/agents/{agent_id}/conversations?limit={limit}",
        )
    except Exception as e:
        return (
            f"prsm_agent_conversations failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    convs = result.get("conversations") or []
    count = result.get("count", len(convs))
    if not convs:
        return (
            f"No conversations for agent_id={agent_id} "
            f"(count={count})."
        )
    lines = [
        f"PRSM Agent {agent_id} Conversations (count={count}):",
    ]
    for c in convs:
        cid = c.get("conversation_id", "?")
        n = c.get("message_count", 0)
        lines.append(f"  conversation_id={cid}  messages={n}")
        for m in (c.get("messages") or [])[:5]:
            role = m.get("role", "?")
            content = (m.get("content") or "")[:60]
            lines.append(f"    [{role}] {content}")
    return "\n".join(lines)


async def handle_prsm_index_stats(arguments: Dict[str, Any]) -> str:
    """Sprint 226 — render GET /content/index/stats."""
    try:
        result = await _call_node_api("GET", "/content/index/stats")
    except Exception as e:
        return (
            f"prsm_index_stats failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if not isinstance(result, dict) or not result:
        return "Content index stats: (empty response)"
    lines = ["PRSM Content Index Stats:"]
    for k, v in result.items():
        lines.append(f"  {k:<24} {v}")
    return "\n".join(lines)


async def handle_prsm_local_balance(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 225 — render GET /balance (local-ledger + recent 20).

    Distinct from prsm_balance_check which hits /balance/onchain
    (aggregates on-chain + claimable + escrowed). Use this for a
    quick local view without aggregate round-trip."""
    try:
        result = await _call_node_api("GET", "/balance")
    except Exception as e:
        return (
            f"prsm_local_balance failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    wallet = result.get("wallet_id", "?")
    balance = result.get("balance", 0)
    txs = result.get("recent_transactions") or []
    lines = [
        f"PRSM Local Balance (wallet_id={wallet}):",
        f"  balance: {balance} FTNS",
        f"  recent_transactions ({len(txs)}):",
    ]
    if not txs:
        lines.append("    (none)")
    else:
        for tx in txs:
            lines.append(
                f"    {tx.get('tx_id', '?')[:18]:<18} "
                f"{tx.get('type', '?'):<18} "
                f"{tx.get('amount', '?')!s:>10} FTNS"
            )
    return "\n".join(lines)


async def handle_prsm_transfer(arguments: Dict[str, Any]) -> str:
    """Sprint 224 — send FTNS to another wallet (signed +
    gossip-broadcast). Endpoint: POST /ledger/transfer."""
    import math as _math
    to_wallet = (arguments.get("to_wallet") or "").strip()
    if not to_wallet:
        return "Missing required 'to_wallet'."
    if "amount" not in arguments or arguments["amount"] is None:
        return "Missing required 'amount'."
    try:
        amount = float(arguments["amount"])
    except (TypeError, ValueError):
        return (
            f"amount must be a finite positive number; "
            f"got {arguments['amount']!r}."
        )
    if not _math.isfinite(amount) or amount <= 0:
        return f"amount must be a finite positive number; got {amount}."
    # /ledger/transfer takes query-params per the handler signature.
    path = (
        f"/ledger/transfer?to_wallet={to_wallet}&amount={amount}"
    )
    try:
        result = await _call_node_api("POST", path)
    except Exception as e:
        return (
            f"prsm_transfer failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "tx_id" not in result:
        detail = result.get("detail", "unknown error")
        if "insufficient" in detail.lower():
            return f"Transfer refused: {detail}"
        return f"Transfer refused: {detail}"
    return (
        f"Transfer broadcast.\n"
        f"  tx_id:     {result.get('tx_id', '?')}\n"
        f"  from:      {result.get('from', '?')}\n"
        f"  to:        {result.get('to', '?')}\n"
        f"  amount:    {result.get('amount', '?')} FTNS\n"
        f"  timestamp: {result.get('timestamp', '?')}"
    )


async def handle_prsm_faucet(arguments: Dict[str, Any]) -> str:
    """Sprint 223 — request testnet FTNS from /ftns/faucet.

    100 FTNS max per request, 1000 FTNS max per wallet. Disabled
    in production (PRSM_FAUCET_ENABLED=0)."""
    import math as _math
    body: Dict[str, Any] = {}
    if "amount" in arguments and arguments["amount"] is not None:
        try:
            amount = float(arguments["amount"])
        except (TypeError, ValueError):
            return (
                f"amount must be a finite positive number; "
                f"got {arguments['amount']!r}."
            )
        if not _math.isfinite(amount) or amount <= 0:
            return f"amount must be a finite positive number; got {amount}."
        body["amount"] = amount
    wallet_id = (arguments.get("wallet_id") or "").strip()
    if wallet_id:
        body["wallet_id"] = wallet_id
    try:
        result = await _call_node_api("POST", "/ftns/faucet", body)
    except Exception as e:
        return (
            f"prsm_faucet failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "granted" not in result:
        detail = result.get("detail", "unknown error")
        if "disabled" in detail.lower():
            return (
                f"Faucet disabled (production node).\n"
                f"  Detail: {detail}"
            )
        return f"Faucet refused: {detail}"
    return (
        f"Faucet grant succeeded.\n"
        f"  wallet_id:   {result.get('wallet_id', '?')}\n"
        f"  granted:     {result.get('granted', 0)} FTNS\n"
        f"  new_balance: {result.get('new_balance', '?')} FTNS"
    )


async def handle_prsm_bridge(arguments: Dict[str, Any]) -> str:
    """Sprint 222 — bridge FTNS between local + external chain.

    direction=deposit  → POST /bridge/deposit (local burn,
                          remote mint on destination_chain)
    direction=withdraw → POST /bridge/withdraw (remote burn,
                          local mint from source_chain)

    Defaults destination_chain/source_chain to 137 (Polygon).
    """
    import math as _math
    direction = (arguments.get("direction") or "").strip().lower()
    if not direction:
        return "Missing required 'direction' (deposit or withdraw)."
    if direction not in ("deposit", "withdraw"):
        return f"direction must be 'deposit' or 'withdraw'; got {direction!r}."
    if "amount" not in arguments or arguments["amount"] is None:
        return "Missing required 'amount'."
    try:
        amount = float(arguments["amount"])
    except (TypeError, ValueError):
        return (
            f"amount must be a finite positive number; "
            f"got {arguments['amount']!r}."
        )
    if not _math.isfinite(amount) or amount <= 0:
        return f"amount must be a finite positive number; got {amount}."
    chain_address = (arguments.get("chain_address") or "").strip()
    if not chain_address:
        return "Missing required 'chain_address'."

    body: Dict[str, Any] = {
        "amount": amount,
        "chain_address": chain_address,
    }
    if direction == "deposit":
        body["destination_chain"] = int(
            arguments.get("destination_chain", 137)
        )
        path = "/bridge/deposit"
    else:
        body["source_chain"] = int(
            arguments.get("source_chain", 137)
        )
        path = "/bridge/withdraw"

    try:
        result = await _call_node_api("POST", path, body)
    except Exception as e:
        return (
            f"prsm_bridge failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if not result.get("success"):
        detail = result.get("detail", "unknown error")
        return f"Bridge {direction} refused: {detail}"
    tx = result.get("transaction") or {}
    return (
        f"Bridge {direction} initiated.\n"
        f"  transaction_id:  {tx.get('transaction_id', '?')}\n"
        f"  amount:          {tx.get('amount', '?')} FTNS\n"
        f"  status:          {tx.get('status', '?')}\n"
        f"  source_chain:    {tx.get('source_chain', '?')}\n"
        f"  destination_chain: {tx.get('destination_chain', '?')}"
    )


async def handle_prsm_agent_admin(arguments: Dict[str, Any]) -> str:
    """Sprint 221 — agent admin actions: set allowance / revoke /
    pause / resume."""
    import math as _math
    agent_id = (arguments.get("agent_id") or "").strip()
    if not agent_id:
        return "Missing required 'agent_id' (non-empty)."
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return (
            "Missing required 'action' (set_allowance, revoke, "
            "pause, or resume)."
        )
    if action not in ("set_allowance", "revoke", "pause", "resume"):
        return (
            f"action must be one of set_allowance / revoke / pause "
            f"/ resume; got {action!r}."
        )

    if action == "set_allowance":
        if "amount" not in arguments or arguments["amount"] is None:
            return "set_allowance requires 'amount'."
        try:
            amount = float(arguments["amount"])
        except (TypeError, ValueError):
            return (
                f"amount must be a finite positive number; "
                f"got {arguments['amount']!r}."
            )
        if not _math.isfinite(amount) or amount <= 0:
            return f"amount must be a finite positive number; got {amount}."
        raw_eh = arguments.get("epoch_hours", 24.0)
        try:
            epoch_hours = float(raw_eh)
        except (TypeError, ValueError):
            return (
                f"epoch_hours must be a finite positive number; "
                f"got {raw_eh!r}."
            )
        if not _math.isfinite(epoch_hours) or epoch_hours <= 0:
            return (
                f"epoch_hours must be a finite positive number; "
                f"got {epoch_hours}."
            )
        path = (
            f"/agents/{agent_id}/allowance?"
            f"amount={amount}&epoch_hours={epoch_hours}"
        )
        method = "POST"
    elif action == "revoke":
        path = f"/agents/{agent_id}/allowance"
        method = "DELETE"
    else:
        path = f"/agents/{agent_id}/{action}"
        method = "POST"

    try:
        result = await _call_node_api(method, path)
    except Exception as e:
        return (
            f"prsm_agent_admin failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "agent_id" not in result:
        detail = result.get("detail", "unknown error")
        return f"{action} refused for {agent_id}: {detail}"
    lines = [
        f"Agent admin action={action} executed on "
        f"agent_id={result.get('agent_id', agent_id)}:",
    ]
    for k, v in result.items():
        if k == "agent_id":
            continue
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


async def handle_prsm_settlers(arguments: Dict[str, Any]) -> str:
    """Sprint 220 — list active settlers or look up specific by id.

    With `settler_id`, calls GET /settler/{id}. Without, calls
    GET /settler/list/active."""
    settler_id = (arguments.get("settler_id") or "").strip()
    if settler_id:
        path = f"/settler/{settler_id}"
    else:
        path = "/settler/list/active"
    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"prsm_settlers failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if settler_id:
        if not isinstance(result, dict) or "settler_id" not in result:
            detail = (
                result.get("detail", "unknown error")
                if isinstance(result, dict) else "unexpected response"
            )
            if "not found" in str(detail).lower():
                return f"Settler {settler_id} not found."
            return f"Settler lookup refused: {detail}"
        return (
            f"PRSM Settler {result.get('settler_id', '?')}:\n"
            f"  address:        {result.get('address', '?')}\n"
            f"  bond_amount:    {result.get('bond_amount', '?')} FTNS\n"
            f"  status:         {result.get('status', '?')}\n"
            f"  can_settle:     {result.get('can_settle', '?')}\n"
            f"  total_settled:  {result.get('total_settled', 0)}\n"
            f"  slashed_amount: {result.get('slashed_amount', 0)}"
        )
    # list path
    settlers = result if isinstance(result, list) else []
    if not settlers:
        return "No active settlers (Phase 6 registry empty)."
    lines = [f"PRSM Active Settlers (count={len(settlers)}):"]
    for s in settlers:
        lines.append(
            f"  {s.get('settler_id', '?'):<16}  "
            f"{(s.get('address') or '?')[:14]:<14}  "
            f"bond={s.get('bond_amount', '?')!s:>10}  "
            f"settled={s.get('total_settled', 0)}"
        )
    return "\n".join(lines)


async def handle_prsm_settler_batches(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 220 — list pending multi-sig settlement batches."""
    try:
        result = await _call_node_api(
            "GET", "/settler/batch/pending",
        )
    except Exception as e:
        return (
            f"prsm_settler_batches failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    batches = result if isinstance(result, list) else []
    if not batches:
        return "No pending settlement batches."
    lines = [f"PRSM Pending Settlement Batches (count={len(batches)}):"]
    for b in batches:
        sig = b.get("signature_count", 0)
        thr = b.get("threshold", "?")
        approved = b.get("approved", False)
        lines.append(
            f"  {b.get('batch_id', '?'):<16}  "
            f"transfers={b.get('transfer_count', 0):>3}  "
            f"total={b.get('total_amount', 0)} FTNS  "
            f"sigs={sig}/{thr}  "
            f"approved={approved}"
        )
    return "\n".join(lines)


async def handle_prsm_unstake_finalize(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 219 — finalize an unstake request: withdraw (after
    unlock) or cancel (before unlock). Single selector tool covers
    both endpoints since they operate on the same request_id."""
    import urllib.parse as _up
    request_id = (arguments.get("request_id") or "").strip()
    if not request_id:
        return "Missing required 'request_id' (non-empty)."
    action = (arguments.get("action") or "").strip().lower()
    if not action:
        return "Missing required 'action' (must be 'withdraw' or 'cancel')."
    if action not in ("withdraw", "cancel"):
        return (
            f"action must be 'withdraw' or 'cancel'; got {action!r}."
        )
    if action == "withdraw":
        path = f"/staking/withdraw/{request_id}"
    else:
        reason = (arguments.get("reason") or "").strip()
        path = f"/staking/cancel-unstake/{request_id}"
        if reason:
            path = f"{path}?reason={_up.quote(reason)}"
    try:
        result = await _call_node_api("POST", path)
    except Exception as e:
        return (
            f"prsm_unstake_finalize failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if action == "withdraw":
        if "amount_withdrawn" not in result:
            detail = result.get("detail", "unknown error")
            return f"Withdraw refused: {detail}"
        return (
            f"Withdraw {'succeeded' if result.get('success') else 'failed'}.\n"
            f"  request_id:        {result.get('request_id', '?')}\n"
            f"  amount_withdrawn:  {result.get('amount_withdrawn', 0)} FTNS"
        )
    # cancel
    if "cancelled" not in result:
        detail = result.get("detail", "unknown error")
        return f"Cancel refused: {detail}"
    return (
        f"Unstake request {'cancelled' if result.get('cancelled') else 'not cancelled'}.\n"
        f"  request_id: {result.get('request_id', '?')}\n"
        f"  reason:     {result.get('reason') or '(none)'}"
    )


async def handle_prsm_claim_rewards(arguments: Dict[str, Any]) -> str:
    """Sprint 218 — claim accumulated staking rewards.

    Without stake_id, claims across all of the node's stakes.
    With stake_id, scopes to that single stake.
    """
    stake_id = (arguments.get("stake_id") or "").strip()
    path = "/staking/claim-rewards"
    if stake_id:
        path = f"{path}?stake_id={stake_id}"
    try:
        result = await _call_node_api("POST", path)
    except Exception as e:
        return (
            f"prsm_claim_rewards failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "total_rewards_claimed" not in result:
        detail = result.get("detail", "unknown error")
        return f"Claim refused: {detail}"
    total = result.get("total_rewards_claimed", 0)
    n = result.get("stakes_processed", 0)
    if float(total) == 0:
        return (
            f"No rewards to claim (stakes processed: {n}).\n"
            f"  user_id: {result.get('user_id', '?')}"
        )
    return (
        f"Rewards claimed:\n"
        f"  user_id:               {result.get('user_id', '?')}\n"
        f"  total_rewards_claimed: {total} FTNS\n"
        f"  stakes_processed:      {n}"
    )


async def handle_prsm_unstake(arguments: Dict[str, Any]) -> str:
    """Sprint 217 — request to unstake FTNS tokens.

    Local-side validation: stake_id required, amount must be
    positive + finite when provided. Body-guard middleware on
    the api side catches Infinity at the wire layer too (sprint
    201), but we validate locally for friendlier UX.
    """
    import math as _math
    stake_id = (arguments.get("stake_id") or "").strip()
    if not stake_id:
        return "Missing required 'stake_id' (non-empty)."
    body: Dict[str, Any] = {"stake_id": stake_id}
    if "amount" in arguments and arguments["amount"] is not None:
        raw_amt = arguments["amount"]
        try:
            amount = float(raw_amt)
        except (TypeError, ValueError):
            return (
                f"amount must be a positive finite number; "
                f"got {raw_amt!r}."
            )
        if not _math.isfinite(amount) or amount <= 0:
            return f"amount must be a positive finite number; got {amount}."
        body["amount"] = amount
    try:
        result = await _call_node_api("POST", "/staking/unstake", body)
    except Exception as e:
        return (
            f"prsm_unstake failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "request_id" not in result:
        detail = result.get("detail", "unknown error")
        if "not found" in detail.lower():
            return f"Unstake refused: {detail}"
        return f"Unstake refused: {detail}"
    return (
        f"Unstake requested.\n"
        f"  request_id:    {result.get('request_id', '?')}\n"
        f"  stake_id:      {result.get('stake_id', '?')}\n"
        f"  amount:        {result.get('amount', '?')} FTNS\n"
        f"  status:        {result.get('status', '?')}\n"
        f"  requested_at:  {result.get('requested_at', '?')}\n"
        f"  available_at:  {result.get('available_at', '?')}\n"
        f"  Use prsm_unstake_finalize to withdraw when available_at "
        f"is reached, or to cancel before then."
    )


_SUBSYSTEM_STATS_PATHS = {
    "settler": "/settler/stats",
    "storage": "/storage/stats",
    "compute": "/compute/stats",
}


async def handle_prsm_subsystem_stats(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 216 — render stats for settler/storage/compute
    subsystems via a single MCP tool selector."""
    subsystem = (arguments.get("subsystem") or "").strip().lower()
    if not subsystem:
        return (
            f"Missing required 'subsystem'. Must be one of "
            f"{sorted(_SUBSYSTEM_STATS_PATHS)}."
        )
    if subsystem not in _SUBSYSTEM_STATS_PATHS:
        return (
            f"subsystem must be one of "
            f"{sorted(_SUBSYSTEM_STATS_PATHS)}; got {subsystem!r}."
        )
    path = _SUBSYSTEM_STATS_PATHS[subsystem]
    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"prsm_subsystem_stats({subsystem}) failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    lines = [f"PRSM {subsystem.title()} Stats:"]
    if not result:
        lines.append("  (empty response)")
    else:
        for k, v in result.items():
            lines.append(f"  {k:<24} {v}")
    return "\n".join(lines)


async def handle_prsm_staking_status(arguments: Dict[str, Any]) -> str:
    """Sprint 215 — render GET /staking/status user dashboard."""
    try:
        result = await _call_node_api("GET", "/staking/status")
    except Exception as e:
        return (
            f"prsm_staking_status failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    user_id = result.get("user_id", "?")
    total_staked = result.get("total_staked", 0)
    earned = result.get("total_rewards_earned", 0)
    claimed = result.get("total_rewards_claimed", 0)
    stakes = result.get("active_stakes") or []
    pending = result.get("pending_unstake_requests") or []

    lines = [
        f"PRSM Staking Status (user={user_id}):",
        f"  Total staked: {total_staked} FTNS",
        f"  Rewards earned: {earned} FTNS  "
        f"(claimed: {claimed}; "
        f"unclaimed: {float(earned) - float(claimed)})",
    ]
    if stakes:
        lines.append(f"  Active stakes ({len(stakes)}):")
        for s in stakes:
            lines.append(
                f"    {s.get('stake_id', '?'):<12}  "
                f"{s.get('amount', '?')!s:>10} FTNS  "
                f"type={s.get('stake_type', '?'):<12}  "
                f"rewards={s.get('rewards_earned', 0)}"
            )
    else:
        lines.append("  No active stakes.")
    if pending:
        lines.append(f"  Pending unstake requests ({len(pending)}):")
        for r in pending:
            lines.append(
                f"    {r.get('request_id', '?'):<12}  "
                f"amount={r.get('amount', '?')}  "
                f"available_at={r.get('available_at', '?')}"
            )
    return "\n".join(lines)


async def handle_prsm_agents(arguments: Dict[str, Any]) -> str:
    """Sprint 214 — list or search agents.

    If `capability` is provided, routes to GET /agents/search;
    otherwise GET /agents with optional `local_only` filter.
    """
    capability = (arguments.get("capability") or "").strip()
    if capability:
        if len(capability) > 256:
            return f"capability must be <= 256 chars; got {len(capability)}."
        raw_limit = arguments.get("limit", 20)
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            return f"limit must be an integer; got {raw_limit!r}."
        if limit < 1 or limit > 100:
            return f"limit must be in [1, 100]; got {limit}."
        path = (
            f"/agents/search?capability={capability}&limit={limit}"
        )
    else:
        local_only = bool(arguments.get("local_only", False))
        path = (
            f"/agents?local_only={'true' if local_only else 'false'}"
        )
    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"prsm_agents failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    agents = result.get("agents") or []
    count = result.get("count", len(agents))
    header = (
        f"PRSM Agents matching capability='{capability}' "
        f"(count={count}):"
        if capability
        else f"PRSM Agents (count={count}):"
    )
    if not agents:
        return f"{header}\n  (none)"
    lines = [header]
    for a in agents:
        lines.append(
            f"  {a.get('agent_id', '?'):<16}  "
            f"{(a.get('display_name') or ''):<24}  "
            f"status={a.get('status', '?')}"
        )
    return "\n".join(lines)


async def handle_prsm_agent_spending(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 214 — render GET /agents/spending aggregate dashboard."""
    try:
        result = await _call_node_api("GET", "/agents/spending")
    except Exception as e:
        return (
            f"prsm_agent_spending failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    agents = result.get("agents") or []
    total_spent = result.get("total_spent", 0)
    total_allow = result.get("total_allowance", 0)
    lines = [
        f"PRSM Agent Spending (total {total_spent} FTNS of "
        f"{total_allow} FTNS allowance):",
    ]
    if not agents:
        lines.append("  (no agents with spending records)")
    else:
        for a in agents:
            lines.append(
                f"  {a.get('agent_id', '?'):<16}  "
                f"spent={a.get('spent', '?')} / "
                f"allowance={a.get('allowance', '?')}"
            )
    return "\n".join(lines)


async def handle_prsm_peers(arguments: Dict[str, Any]) -> str:
    """Sprint 213 / Sprint 326 — render /peers connected + known
    peer lists. Pre-sprint-326 the handler only rendered the
    `connected` list (transport-layer outbound/inbound peers)
    and silently dropped the `known` list (bootstrap-discovered
    peers with capabilities). On canonical wss:// bootstrap-wired
    nodes the connected list is typically empty (libp2p direct
    dialing isn't wired to the bootstrap server), so operators
    saw "No peers connected" and missed the actual discovered
    peer set.

    Sprint 326 renders both lists. Known peers show
    capabilities — pairs with sprint 322's threading of caps
    from bootstrap into PeerInfo.
    """
    try:
        result = await _call_node_api("GET", "/peers")
    except Exception as e:
        return (
            f"prsm_peers failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    connected = result.get("connected") or []
    connected_count = result.get("connected_count", len(connected))
    known = result.get("known") or []
    known_count = result.get("known_count", len(known))

    if not connected and not known:
        return (
            f"No peers (connected={connected_count}, "
            f"known={known_count}). If degraded, check "
            f"PRSM_BOOTSTRAP_ENDPOINT + /info network."
        )

    lines: List[str] = []
    if connected:
        lines.append(
            f"PRSM Connected Peers (count={connected_count}):"
        )
        for p in connected:
            direction = (
                "outbound" if p.get("outbound") else "inbound "
            )
            peer_id = (p.get("peer_id") or "?")[:14]
            addr = (p.get("address") or "?")[:60]
            name = p.get("display_name") or ""
            lines.append(
                f"  [{direction}] {peer_id:<14}  {addr}  {name}"
            )
    if known:
        if lines:
            lines.append("")
        lines.append(
            f"PRSM Known Peers (count={known_count}):"
        )
        for p in known:
            pid = (p.get("node_id") or "?")[:20]
            addr = (p.get("address") or "?")[:30]
            caps = p.get("capabilities") or []
            caps_str = ", ".join(caps) if caps else "—"
            lines.append(
                f"  {pid:<20}  {addr:<30}  caps=[{caps_str}]"
            )
    return "\n".join(lines)


async def handle_prsm_transactions(arguments: Dict[str, Any]) -> str:
    """Sprint 212 — render GET /transactions history.

    Local-side limit validation: server caps at [1, 200]; reject
    out-of-range before round-trip so users get an instant error
    instead of a 422.
    """
    raw_limit = arguments.get("limit", 50)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        return f"limit must be an integer in [1, 200]; got {raw_limit!r}."
    if limit < 1 or limit > 200:
        return f"limit must be in [1, 200]; got {limit}."

    try:
        result = await _call_node_api(
            "GET", f"/transactions?limit={limit}",
        )
    except Exception as e:
        return (
            f"prsm_transactions failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    txs = result.get("transactions") or []
    count = result.get("count", len(txs))
    if not txs:
        return f"No transactions in history (count={count})."
    lines = [f"PRSM Transactions (count={count}, showing {len(txs)}):"]
    for tx in txs:
        ts = tx.get("timestamp")
        ts_str = (
            f"{int(ts) % 86400 // 3600:02d}:"
            f"{int(ts) % 3600 // 60:02d}:{int(ts) % 60:02d}"
            if isinstance(ts, (int, float)) else "????"
        )
        lines.append(
            f"  {tx.get('tx_id', '?')[:18]:<18} "
            f"{tx.get('type', '?'):<18} "
            f"{tx.get('amount', '?')!s:>10} FTNS  "
            f"{(tx.get('from') or '?')[:10]}→{(tx.get('to') or '?')[:10]}  "
            f"@~{ts_str}"
        )
    return "\n".join(lines)


async def handle_prsm_info(arguments: Dict[str, Any]) -> str:
    """Sprint 211 — render GET /info static node metadata.

    Useful for verifying which chain/contracts the node is pinned
    to without parsing /health/detailed."""
    try:
        result = await _call_node_api("GET", "/info")
    except Exception as e:
        return (
            f"prsm_info failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    lines = ["PRSM Node Info:"]
    lines.append(f"  node_id:     {result.get('node_id', '?')}")
    lines.append(f"  api_version: {result.get('api_version', '?')}")
    if "network" in result:
        lines.append(f"  network:     {result['network']}")
    if "chain_id" in result:
        lines.append(f"  chain_id:    {result['chain_id']}")
    if "rpc_host" in result:
        lines.append(f"  rpc_host:    {result['rpc_host']}")
    if "operator_address" in result:
        lines.append(
            f"  operator:    {result['operator_address']}"
        )
    if "agent_forge_wired" in result:
        lines.append(
            f"  agent_forge_wired: {result['agent_forge_wired']}"
        )
    if "query_orchestrator_state" in result:
        lines.append(
            f"  qo_state:    {result['query_orchestrator_state']}"
        )
    if "query_orchestrator_error" in result:
        lines.append(
            f"  qo_error:    {result['query_orchestrator_error']}"
        )
    canonical = result.get("canonical_addresses") or {}
    if canonical:
        lines.append("  canonical_addresses:")
        for fld, addr in canonical.items():
            lines.append(f"    {fld:<26} {addr}")
    return "\n".join(lines)


async def handle_prsm_cancel_job(arguments: Dict[str, Any]) -> str:
    """Sprint 210 — cancel a submitted job by job_id via
    POST /compute/cancel/{job_id}. Marks history CANCELLED and
    refunds PENDING escrow (v1 caveat: in-flight Python coroutines
    not interrupted but their release-side race-loses against the
    now-REFUNDED escrow)."""
    job_id = (arguments.get("job_id") or "").strip()
    if not job_id:
        return "Missing required 'job_id' (non-empty)."
    try:
        result = await _call_node_api(
            "POST", f"/compute/cancel/{job_id}",
        )
    except Exception as e:
        return (
            f"prsm_cancel_job failed for job_id={job_id}: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    # 404 path — node returns {"detail": "..."}.
    if "detail" in result and "status" not in result:
        return (
            f"Cancellation refused for job_id={job_id}: "
            f"{result.get('detail', '?')}"
        )
    return (
        f"Job {job_id} cancellation requested.\n"
        f"  status: {result.get('status', '?')}\n"
        f"  history_marked: {result.get('history_marked', '?')}\n"
        f"  escrow_refunded: {result.get('escrow_refunded', '?')}\n"
        f"  Note: in-flight Python coroutines are not interrupted; "
        f"the release-side race loses against the REFUNDED escrow."
    )


async def handle_prsm_jobs_list(arguments: Dict[str, Any]) -> str:
    """Handle prsm_jobs_list tool call: enumerate /compute/forge,
    /compute/inference, and /compute/inference/stream jobs with
    optional filter + pagination. Sprint 260 — `route` filter
    scopes results to a single compute path."""
    params = []
    if "status" in arguments:
        params.append(f"status={arguments['status']}")
    if "route" in arguments and arguments["route"]:
        params.append(f"route={arguments['route']}")
    if "limit" in arguments:
        params.append(f"limit={arguments['limit']}")
    else:
        params.append("limit=20")
    if "offset" in arguments:
        params.append(f"offset={arguments['offset']}")
    path = "/compute/jobs"
    if params:
        path += "?" + "&".join(params)

    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    if "jobs" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in detail.lower():
            return (
                f"JobHistoryStore not configured on this node.\n"
                f"  Detail: {detail}"
            )
        return f"prsm_jobs_list failed.\n  Detail: {detail}"

    jobs = result["jobs"]
    total = result.get("total", 0)
    offset = result.get("offset", 0)
    limit = result.get("limit", 0)

    if not jobs:
        return f"No jobs match the filter (total={total})."

    lines = [f"PRSM Jobs (showing {offset+1}–{offset+len(jobs)} of {total}):"]
    for j in jobs:
        ts = j.get("started_at")
        ts_str = (
            f"{int(ts) % 86400 // 3600:02d}:"
            f"{int(ts) % 3600 // 60:02d}:{int(ts) % 60:02d}"
            if isinstance(ts, (int, float)) else "????"
        )
        # Sprint 261 — surface route so the unified compute view
        # (forge / inference / inference_stream / qo_swarm /
        # direct_llm / swarm) is visually distinguishable.
        route_disp = (j.get("route") or "?")[:16]
        lines.append(
            f"  {j['job_id']:<16}  "
            f"{j['status']:<12}  "
            f"route={route_disp:<16}  "
            f"started @ ~{ts_str}  "
            f"{(j.get('query') or '')[:40]}"
        )
    if offset + len(jobs) < total:
        lines.append(
            f"  ... pass offset={offset+limit} to see next page"
        )
    return "\n".join(lines)


async def handle_prsm_royalty_claim(arguments: Dict[str, Any]) -> str:
    """Handle prsm_royalty_claim tool call.

    Closes the loop on the offramp claim_required path. Defaults
    to dry_run=true; pass dry_run=false to execute the on-chain
    claim() call.
    """
    dry_run = bool(arguments.get("dry_run", True))
    body = {"dry_run": dry_run}

    try:
        result = await _call_node_api("POST", "/wallet/royalty/claim", body)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    if "status" not in result:
        detail = result.get("detail", "unknown error")
        if "not wired" in detail.lower() or "distributor" in detail.lower():
            return (
                f"RoyaltyDistributor not configured on this node.\n"
                f"  Detail: {detail}\n"
                f"  Set PRSM_ROYALTY_DISTRIBUTOR_ADDRESS + "
                f"FTNS_TOKEN_ADDRESS to enable."
            )
        return f"Royalty claim failed.\n  Detail: {detail}"

    status = result["status"]
    claimable = result.get("claimable_ftns", 0.0)

    if status == "DRY_RUN":
        return (
            f"PRSM Royalty Claim (dry-run)\n"
            f"  Claimable:    {claimable:.6f} FTNS\n"
            f"  Status:       DRY_RUN  (no on-chain action)\n"
            f"\n"
            f"  Pass dry_run=false to execute the on-chain claim().\n"
            f"  Example: prsm_royalty_claim {{\"dry_run\": false}}"
        )
    if status == "SKIPPED_ZERO":
        return (
            f"PRSM Royalty Claim\n"
            f"  Claimable:    0.000000 FTNS\n"
            f"  Status:       SKIPPED_ZERO\n"
            f"  Note: {result.get('note', 'No claimable balance.')}"
        )
    if status == "EXECUTED":
        return (
            f"PRSM Royalty Claim (executed)\n"
            f"  Claimed:      {result['amount_claimed_ftns']:.6f} FTNS\n"
            f"  Tx hash:      {result['tx_hash']}\n"
            f"  Status:       EXECUTED  "
            f"({result.get('transfer_status', 'OK')})"
        )
    if status == "PENDING":
        # sp915 — claim broadcast OK but receipt unconfirmed. Do NOT re-claim.
        return (
            f"PRSM Royalty Claim (PENDING)\n"
            f"  Claimable:    {claimable:.6f} FTNS\n"
            f"  Tx hash:      {result.get('tx_hash')}\n"
            f"  Status:       PENDING — broadcast OK but UNCONFIRMED.\n"
            f"  Do NOT re-claim; reconcile via tx_hash."
        )
    return f"Royalty claim returned unknown status: {status}"


async def handle_coinbase_onramp_initiate(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 278 — composer-only USD → FTNS on-ramp artifact.

    Mirrors handle_coinbase_offramp_initiate in the opposite
    direction. Composer-only: no execute path. Status returned is
    always PENDING_COMMISSION until Coinbase CDP commissions per
    Vision gantt 2026-06-22.

    Either destination_user_id (WaaS-resolved) or
    destination_address is required; the endpoint enforces XOR.
    """
    if "usd_amount" not in arguments:
        return (
            "Missing required argument: usd_amount.\n"
            "Example: {\"usd_amount\": 100.0, "
            "\"destination_user_id\": \"alice\"}"
        )
    dest_user_id = arguments.get("destination_user_id")
    dest_address = arguments.get("destination_address")
    if not dest_user_id and not dest_address:
        return (
            "Missing destination. Supply either "
            "destination_user_id (WaaS-resolved) OR "
            "destination_address."
        )

    body: Dict[str, Any] = {
        "usd_amount": arguments["usd_amount"],
        "payment_method_alias": arguments.get(
            "payment_method_alias", "primary",
        ),
    }
    if dest_user_id:
        body["destination_user_id"] = dest_user_id
    if dest_address:
        body["destination_address"] = dest_address

    try:
        result = await _call_node_api(
            "POST", "/wallet/onramp/quote", body,
        )
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {e}\n"
            f"Start with: prsm node start"
        )

    if "quote" not in result:
        detail = str(result.get("detail", "unknown error"))
        d_lower = detail.lower()
        if "no waas wallet" in d_lower:
            return (
                f"No WaaS wallet for destination_user_id="
                f"{dest_user_id!r}. Provision one first with "
                f"prsm_waas_wallet?action=provision."
            )
        if "positive" in d_lower or "> 0" in detail:
            return (
                f"usd_amount must be positive (> 0).\n"
                f"  Detail: {detail}"
            )
        if (
            "destination_user_id" in d_lower
            or "destination_address" in d_lower
        ):
            return (
                f"Destination validation failed.\n"
                f"  Detail: {detail}"
            )
        if "not initialized" in d_lower:
            return (
                f"WaaS client not configured on this node.\n"
                f"  Detail: {detail}"
            )
        return f"On-ramp quote failed.\n  Detail: {detail}"

    quote = result["quote"]
    dest_addr = result.get("destination_address")
    dest_uid = result.get("destination_user_id")
    addr_display = dest_addr if dest_addr else "(pending — wallet not yet provisioned)"
    lines = [
        f"Coinbase On-Ramp — Transaction Summary "
        f"(status: {result.get('status', 'PENDING_COMMISSION')}):",
        "",
        f"  USD in:            ${quote.get('usd_in', 0):.2f}",
        f"  USDC acquired:     ${quote.get('usdc_acquired', 0):.2f}",
        f"  FTNS received:     {quote.get('ftns_received', 0):.6f} FTNS",
        f"  Rate:              ${result.get('usd_rate', 1.0)} USD/FTNS",
        "",
        f"  Destination:       {addr_display}",
    ]
    if dest_uid:
        lines.append(f"  Resolved from user_id: {dest_uid}")
    # Sprint 281 — KYC prerequisite block (mirrors the
    # claim_required block on the offramp side).
    if result.get("kyc_required"):
        kyc_status = result.get("kyc_status", "NOT_STARTED")
        kyc_url = result.get("kyc_session_url")
        lines += ["", "  Prerequisite: KYC required."]
        lines.append(f"    Current KYC status: {kyc_status}")
        if kyc_url:
            lines.append(
                f"    Complete vendor flow: {kyc_url}"
            )
        else:
            lines.append(
                "    Start a session with "
                "prsm_kyc?action=initiate"
            )
    # Sprint 285 — tier-limit prerequisite block.
    if result.get("tier_limit_exceeded"):
        tier_level = result.get("tier_level", "?")
        tier_limit = result.get("tier_limit_usd", 0.0)
        tier_remaining = result.get(
            "tier_limit_remaining_usd", 0.0,
        )
        lines += [
            "",
            f"  Prerequisite: Tier limit exceeded "
            f"(tier={tier_level}).",
            f"    Tier daily cap:    ${tier_limit:,.2f}",
            f"    Remaining today:   ${tier_remaining:,.2f}",
        ]
        if tier_level == "basic":
            lines.append(
                "    Upgrade to enhanced KYC to raise the "
                "limit: prsm_kyc?action=initiate&level=enhanced"
            )
    lines += [
        "",
        f"  Payment method:    {quote.get('payment_method_alias', 'primary')}",
        f"  On-ramp route:     {quote.get('onramp_route', 'coinbase-cdp')}",
        f"  Swap route:        {quote.get('swap_route', 'aerodrome')}",
        "",
        f"  Note: {result.get('note', '')}",
        "",
        "This is a preview artifact only. No on-chain or fiat "
        "rails movement has occurred. Real execution lands when "
        "Coinbase CDP commissions.",
    ]
    return "\n".join(lines)


async def handle_coinbase_offramp_initiate(arguments: Dict[str, Any]) -> str:
    """Handle coinbase_offramp_initiate tool call.

    V1 scope: pre-flight quote composer per Vision §13 Phase 5
    step 2. Calls POST /wallet/offramp/quote and formats the
    response as a transaction-summary artifact. Does NOT initiate
    any on-chain or fiat-side action — actual execution gates on
    CDP commission per Vision gantt 2026-06-15.
    """
    if "usd_amount" not in arguments:
        return (
            "Missing required argument: usd_amount.\n"
            "Example: {\"usd_amount\": 500.0}"
        )

    body = {
        "usd_amount": arguments["usd_amount"],
        "bank_account_alias": arguments.get("bank_account_alias", "primary"),
    }

    try:
        result = await _call_node_api("POST", "/wallet/offramp/quote", body)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    # 4xx/503 fallback path — endpoint returned a `detail` envelope
    # rather than a quote.
    if "quote" not in result:
        detail = result.get("detail", "unknown error")
        # Distinguish insufficient-balance (422) from misconfig (503).
        if "insufficient" in detail.lower() or "balance" in detail.lower():
            return (
                f"Insufficient balance for off-ramp.\n"
                f"  Detail: {detail}\n"
                f"  Use prsm_balance_check to verify available funds."
            )
        if "not initialized" in detail.lower() or "ftns_ledger" in detail.lower():
            return (
                f"On-chain FTNS not configured on this node.\n"
                f"  Detail: {detail}\n"
                f"  Set PRSM_ONCHAIN_FTNS=1 + FTNS_TOKEN_ADDRESS to enable."
            )
        return f"Off-ramp quote failed.\n  Detail: {detail}"

    quote = result["quote"]
    addr = result["source_address"]
    short_addr = (
        addr[:10] + "…" + addr[-4:] if len(addr) > 14 else addr
    )

    # Aggregate-source claim-required prerequisite block (v2 endpoint
    # field). When on-chain alone insufficient but claimable royalties
    # bridge the gap, surface the required claim before the quote so
    # the operator knows the eventual swap depends on it.
    prereq_block = ""
    if result.get("claim_required"):
        claim_amount = result.get("claim_amount_ftns", 0.0)
        available_ftns = result.get("available_ftns", 0.0)
        available_usd = result.get("available_usd", 0.0)
        claimable = result.get("claimable_royalties_ftns", 0.0)
        prereq_block = (
            f"\n"
            f"  Prerequisite: Claim {claim_amount:.6f} FTNS in royalties "
            f"before swap can execute\n"
            f"    Available (aggregate):  {available_ftns:.6f} FTNS  "
            f"(${available_usd:,.2f})\n"
            f"    On-chain:               "
            f"{result['source_balance_ftns']:.6f} FTNS\n"
            f"    Claimable royalties:    {claimable:.6f} FTNS\n"
        )

    # Sprint 281 — KYC prerequisite block (mirrors the
    # onramp side; surfaces when source_user_id was supplied
    # and the resolved user is not VERIFIED).
    kyc_block = ""
    if result.get("kyc_required"):
        kyc_status = result.get("kyc_status", "NOT_STARTED")
        kyc_url = result.get("kyc_session_url")
        kyc_lines = [
            "",
            "  Prerequisite: KYC required.",
            f"    Current KYC status: {kyc_status}",
        ]
        if kyc_url:
            kyc_lines.append(
                f"    Complete vendor flow: {kyc_url}"
            )
        else:
            kyc_lines.append(
                "    Start a session with "
                "prsm_kyc?action=initiate"
            )
        kyc_block = "\n".join(kyc_lines) + "\n"

    # Sprint 285 — tier-limit prerequisite block.
    tier_block = ""
    if result.get("tier_limit_exceeded"):
        tier_level = result.get("tier_level", "?")
        tier_limit = result.get("tier_limit_usd", 0.0)
        tier_remaining = result.get(
            "tier_limit_remaining_usd", 0.0,
        )
        tier_lines = [
            "",
            f"  Prerequisite: Tier limit exceeded "
            f"(tier={tier_level}).",
            f"    Tier daily cap:    ${tier_limit:,.2f}",
            f"    Remaining today:   ${tier_remaining:,.2f}",
        ]
        if tier_level == "basic":
            tier_lines.append(
                "    Upgrade to enhanced KYC to raise the "
                "limit: prsm_kyc?action=initiate&level=enhanced"
            )
        tier_block = "\n".join(tier_lines) + "\n"

    return (
        f"PRSM Cash-Out Pre-Flight\n"
        f"  Requested:    ${result['requested_usd']:,.2f} USD\n"
        f"  Source:       {short_addr}\n"
        f"  Balance:      {result['source_balance_ftns']:.6f} FTNS  "
        f"(${result['source_balance_usd']:,.2f} @ "
        f"{result['usd_rate']} USD/FTNS)\n"
        f"{prereq_block}"
        f"{kyc_block}"
        f"{tier_block}"
        f"\n"
        f"  Quote:\n"
        f"    Swap:       {quote['ftns_to_swap']:.6f} FTNS  "
        f"→  {quote['usdc_received']:,.2f} USDC  (via {quote['swap_route']})\n"
        f"    Off-ramp:   {quote['usdc_received']:,.2f} USDC  "
        f"→  ${quote['usd_settled']:,.2f} USD  "
        f"(via {quote['offramp_route']})\n"
        f"    Bank:       {quote['bank_account_alias']}\n"
        f"\n"
        f"  Status:       {result['status']}\n"
        f"\n"
        f"  Note: {result['commission_gate_note']}"
    )


# Tool dispatch map
TOOL_HANDLERS = {
    "prsm_analyze": handle_prsm_analyze,
    "prsm_quote": handle_prsm_quote,
    "prsm_list_datasets": handle_prsm_list_datasets,
    "prsm_get_dataset": handle_prsm_get_dataset,
    "prsm_node_status": handle_prsm_node_status,
    "prsm_section7_readiness": handle_prsm_section7_readiness,
    "prsm_hardware_benchmark": handle_prsm_hardware_benchmark,
    "prsm_create_agent": handle_prsm_create_agent,
    "prsm_dispatch_agent": handle_prsm_dispatch_agent,
    "prsm_agent_status": handle_prsm_agent_status,
    "prsm_search_shards": handle_prsm_search_shards,
    "prsm_upload_dataset": handle_prsm_upload_dataset,
    "prsm_yield_estimate": handle_prsm_yield_estimate,
    "prsm_stake": handle_prsm_stake,
    "prsm_revenue_split": handle_prsm_revenue_split,
    "prsm_settlement_stats": handle_prsm_settlement_stats,
    "prsm_privacy_status": handle_prsm_privacy_status,
    "prsm_training_status": handle_prsm_training_status,
    "prsm_inference": handle_prsm_inference,
    "prsm_billing_status": handle_prsm_billing_status,
    "prsm_balance_check": handle_prsm_balance_check,
    "prsm_arbitration_preview_resolution": handle_prsm_arbitration_preview_resolution,
    "prsm_arbitration_record_detail": handle_prsm_arbitration_record_detail,
    "prsm_arbitration_status": handle_prsm_arbitration_status,
    "prsm_audit_recent": handle_prsm_audit_recent,
    "prsm_audit_summary": handle_prsm_audit_summary,
    "prsm_canonical_check": handle_prsm_canonical_check,
    "prsm_forge_submit": handle_prsm_forge_submit,
    "prsm_content_info": handle_prsm_content_info,
    "prsm_my_content": handle_prsm_my_content,
    "prsm_distribution_trigger": handle_prsm_distribution_trigger,
    "prsm_heartbeat_trigger": handle_prsm_heartbeat_trigger,
    "prsm_distribution_history": handle_prsm_distribution_history,
    "prsm_heartbeat_history": handle_prsm_heartbeat_history,
    "prsm_slash_history": handle_prsm_slash_history,
    "prsm_earnings_summary": handle_prsm_earnings_summary,
    "prsm_webhook_history": handle_prsm_webhook_history,
    "prsm_webhook_test": handle_prsm_webhook_test,
    "prsm_metrics_summary": handle_prsm_metrics_summary,
    "prsm_cleanup_stale_escrows": handle_prsm_cleanup_stale_escrows,
    "prsm_node_health": handle_prsm_node_health,
    "prsm_spend_summary": handle_prsm_spend_summary,
    "prsm_escrow_lookup": handle_prsm_escrow_lookup,
    "prsm_escrow_summary": handle_prsm_escrow_summary,
    "prsm_jobs_list": handle_prsm_jobs_list,
    "prsm_status_stream": handle_prsm_status_stream,
    "prsm_cancel_job": handle_prsm_cancel_job,
    "prsm_info": handle_prsm_info,
    "prsm_transactions": handle_prsm_transactions,
    "prsm_peers": handle_prsm_peers,
    "prsm_agents": handle_prsm_agents,
    "prsm_staking_status": handle_prsm_staking_status,
    "prsm_subsystem_stats": handle_prsm_subsystem_stats,
    "prsm_unstake": handle_prsm_unstake,
    "prsm_claim_rewards": handle_prsm_claim_rewards,
    "prsm_unstake_finalize": handle_prsm_unstake_finalize,
    "prsm_settlers": handle_prsm_settlers,
    "prsm_agent_admin": handle_prsm_agent_admin,
    "prsm_bridge": handle_prsm_bridge,
    "prsm_faucet": handle_prsm_faucet,
    "prsm_transfer": handle_prsm_transfer,
    "prsm_local_balance": handle_prsm_local_balance,
    "prsm_index_stats": handle_prsm_index_stats,
    "prsm_agent_conversations": handle_prsm_agent_conversations,
    "prsm_get_agent": handle_prsm_get_agent,
    "prsm_stake_lookup": handle_prsm_stake_lookup,
    "prsm_bridge_history": handle_prsm_bridge_history,
    "prsm_settlement_view": handle_prsm_settlement_view,
    "prsm_node_resources": handle_prsm_node_resources,
    "prsm_ledger_sync": handle_prsm_ledger_sync,
    "prsm_models": handle_prsm_models,
    "prsm_pubkey": handle_prsm_pubkey,
    "prsm_verify_receipt": handle_prsm_verify_receipt,
    "prsm_receipt": handle_prsm_receipt,
    "prsm_receipts_list": handle_prsm_receipts_list,
    "prsm_royalty_dispatch_history": handle_prsm_royalty_dispatch_history,
    "prsm_royalty_dispatch_summary": handle_prsm_royalty_dispatch_summary,
    "prsm_bootstrap_status": handle_prsm_bootstrap_status,
    "prsm_bootstrap_test": handle_prsm_bootstrap_test,
    "prsm_bootstrap_server_status": handle_prsm_bootstrap_server_status,
    "prsm_pinned_stats": handle_prsm_pinned_stats,
    "prsm_content_filter": handle_prsm_content_filter,
    "prsm_takedown_notices": handle_prsm_takedown_notices,
    "prsm_marketplace_reputation": handle_prsm_marketplace_reputation,
    "prsm_creator_reputation": handle_prsm_creator_reputation,
    "prsm_creator_stake": handle_prsm_creator_stake,
    "prsm_content_fingerprint": handle_prsm_content_fingerprint,
    "prsm_verify_inference_privacy": handle_prsm_verify_inference_privacy,
    "prsm_emergency_pause": handle_prsm_emergency_pause,
    "prsm_insurance_fund": handle_prsm_insurance_fund,
    "prsm_disclosure": handle_prsm_disclosure,
    "prsm_incident": handle_prsm_incident,
    "prsm_formal_verification": handle_prsm_formal_verification,
    "prsm_upgrade": handle_prsm_upgrade,
    "prsm_enterprise_recipient": handle_prsm_enterprise_recipient,
    "prsm_tee_policy": handle_prsm_tee_policy,
    "prsm_corp_capability": handle_prsm_corp_capability,
    "prsm_federated_learning": handle_prsm_federated_learning,
    "prsm_federated_train": handle_prsm_federated_train,
    "prsm_pipeline_inference": handle_prsm_pipeline_inference,
    "prsm_waas_wallet": handle_prsm_waas_wallet,
    "prsm_gasless_transfer": handle_prsm_gasless_transfer,
    "prsm_pool_quote": handle_prsm_pool_quote,
    "prsm_kyc": handle_prsm_kyc,
    "prsm_fiat_compliance": handle_prsm_fiat_compliance,
    "prsm_fiat_surface_health": handle_prsm_fiat_surface_health,
    "prsm_content_provider_stats": handle_prsm_content_provider_stats,
    "prsm_provider_reputations": handle_prsm_provider_reputations,
    "prsm_forge_quote": handle_prsm_forge_quote,
    "prsm_inference_quote": handle_prsm_inference_quote,
    "prsm_settler_admin": handle_prsm_settler_admin,
    "prsm_settler_batches": handle_prsm_settler_batches,
    "prsm_agent_spending": handle_prsm_agent_spending,
    "prsm_royalty_claim": handle_prsm_royalty_claim,
    "coinbase_offramp_initiate": handle_coinbase_offramp_initiate,
    "coinbase_onramp_initiate": handle_coinbase_onramp_initiate,
}


# ── MCP Server ───────────────────────────────────────────────────────────

def create_server() -> Server:
    """Create and configure the PRSM MCP server.

    Server version reads from installed package metadata so it
    stays in sync with pyproject.toml across releases (parallel
    to /api-info + /openapi.json + prsm_build_info gauge).
    """
    try:
        from importlib.metadata import version as _pkg_version
        _server_version = _pkg_version("prsm-network")
    except Exception:  # noqa: BLE001
        _server_version = "unknown"
    server = Server(
        name="prsm",
        version=_server_version,
        instructions=(
            "PRSM is a decentralized AI compute network. Use these tools to "
            "submit analysis queries, run TEE-attested inference, estimate "
            "costs, browse datasets, and monitor node health. The network "
            "dispatches WASM mobile agents to edge nodes for distributed "
            "computation, and runs sharded inference under TEE attestation "
            "for verifiable confidential compute."
        ),
    )

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        # Hide currently-broken tools from client-side discovery.
        # See BROKEN_TOOLS_HIDDEN above for rationale + lift-gate
        # conditions. PRSM_EXPOSE_BROKEN_TOOLS=1 forces visibility for
        # operators reconstructing the data-query path.
        expose_broken = os.getenv(
            "PRSM_EXPOSE_BROKEN_TOOLS", "",
        ).lower() in ("1", "true", "yes")
        if expose_broken:
            return TOOLS
        return [t for t in TOOLS if t.name not in BROKEN_TOOLS_HIDDEN]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> Sequence[TextContent]:
        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        try:
            # Streaming opt-in: handlers that accept an `emit_progress` keyword
            # parameter receive a real emitter when the MCP client supplied a
            # progressToken in its request meta. Per Phase 3.x.1 Task 8.
            kwargs: Dict[str, Any] = {}
            if _handler_accepts_emit_progress(handler):
                emitter = _build_progress_emitter(server)
                # Pass the emitter even when None — handlers should treat None
                # as "client didn't ask for streaming." Passing it consistently
                # keeps the handler's kwargs surface stable across calls.
                kwargs["emit_progress"] = emitter

            result_text = await handler(arguments or {}, **kwargs)
            return [TextContent(type="text", text=result_text)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    return server


def _handler_accepts_emit_progress(handler: Callable) -> bool:
    """True iff `handler` accepts an `emit_progress` keyword parameter.

    Used to opt handlers into streaming without changing the dispatcher
    contract for the other 15 tools.
    """
    try:
        sig = inspect.signature(handler)
    except (TypeError, ValueError):
        return False
    return "emit_progress" in sig.parameters


def _build_progress_emitter(server: Server) -> Optional[ProgressEmitter]:
    """Build a ProgressEmitter from the current MCP request context.

    Returns None when:
      - No request_context is active (server isn't currently handling a request)
      - Client did not provide a progressToken in request meta

    Returns a callable when the client opted into progress streaming. The
    callable safely no-ops if the underlying session call raises (we don't
    want a failed progress notification to break the tool response).
    """
    try:
        ctx = server.request_context
    except Exception:
        return None

    if ctx is None or ctx.meta is None:
        return None
    progress_token = getattr(ctx.meta, "progressToken", None)
    if progress_token is None:
        return None

    session = ctx.session

    async def _emit(message: str, progress: float, total: Optional[float] = None) -> None:
        try:
            await session.send_progress_notification(
                progress_token=progress_token,
                progress=progress,
                total=total,
                message=message,
            )
        except Exception as exc:
            # Log but don't propagate — a dropped progress notification
            # should not break the tool response.
            logger.warning(f"send_progress_notification failed: {exc}")

    return _emit


async def run_server():
    """Run the MCP server over stdio."""
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main():
    """Entry point for the MCP server.

    CRITICAL: MCP stdio protocol requires that ONLY JSON-RPC messages
    go to stdout. All logging and print output must go to stderr.
    We capture stdout during PRSM imports to prevent structlog noise
    from corrupting the JSON-RPC stream.
    """
    import sys

    # Temporarily redirect stdout to stderr during imports
    # (structlog prints to stdout on module load)
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    # Suppress all logging to stderr
    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stderr,
        format="%(name)s: %(message)s",
    )
    for name in [
        "prsm", "prsm.core", "prsm.compute", "prsm.data", "prsm.economy",
        "prsm.node", "structlog", "httpx", "aiohttp",
    ]:
        logging.getLogger(name).setLevel(logging.ERROR)

    # Force-import ALL PRSM modules while stdout is captured
    # This ensures structlog's noisy output goes to stderr, not stdout
    _imports = [
        "prsm", "prsm.core", "prsm.core.config", "prsm.core.models",
        "prsm.compute.wasm.profiler", "prsm.compute.wasm.profiler_models",
        "prsm.compute.tee.platform_detect", "prsm.compute.tee.models",
        "prsm.economy.pricing", "prsm.economy.pricing.engine",
        "prsm.compute.nwtn.agent_forge",
    ]
    for mod in _imports:
        try:
            __import__(mod)
        except Exception:
            pass

    # Restore stdout for MCP protocol
    sys.stdout = real_stdout

    asyncio.run(run_server())


if __name__ == "__main__":
    main()
