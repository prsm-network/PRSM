"""Tests for the PRSM MCP Server."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from prsm.mcp_server import (
    create_server,
    TOOLS,
    TOOL_HANDLERS,
    handle_prsm_quote,
    handle_prsm_hardware_benchmark,
    handle_prsm_analyze,
    handle_prsm_node_status,
    handle_prsm_create_agent,
    handle_prsm_dispatch_agent,
    handle_prsm_agent_status,
    handle_prsm_search_shards,
    handle_prsm_upload_dataset,
    handle_prsm_yield_estimate,
    handle_prsm_stake,
    handle_prsm_revenue_split,
    handle_prsm_settlement_stats,
    handle_prsm_privacy_status,
    handle_prsm_training_status,
    handle_prsm_inference,
)


class TestMCPToolDefinitions:
    def test_at_least_twenty_tools_defined(self):
        # Tool count was 20 at the post-coinbase_offramp_initiate
        # baseline (2026-05-08). Subsequent sprints have added
        # more tools; the invariant we care about is that the
        # surface only grows from here.
        assert len(TOOLS) >= 20

    def test_tool_names(self):
        names = [t.name for t in TOOLS]
        assert "prsm_analyze" in names
        assert "prsm_quote" in names
        assert "prsm_list_datasets" in names
        assert "prsm_node_status" in names
        assert "prsm_hardware_benchmark" in names
        assert "prsm_create_agent" in names
        assert "prsm_dispatch_agent" in names
        assert "prsm_agent_status" in names
        assert "prsm_search_shards" in names
        assert "prsm_upload_dataset" in names
        assert "prsm_yield_estimate" in names
        assert "prsm_stake" in names
        assert "prsm_revenue_split" in names
        assert "prsm_settlement_stats" in names
        assert "prsm_privacy_status" in names
        assert "prsm_training_status" in names
        assert "prsm_inference" in names

    def test_all_tools_have_descriptions(self):
        for tool in TOOLS:
            assert len(tool.description) > 20, f"Tool {tool.name} has short description"

    def test_all_tools_have_input_schema(self):
        for tool in TOOLS:
            assert tool.inputSchema is not None
            assert tool.inputSchema.get("type") == "object"

    def test_analyze_requires_query(self):
        analyze = next(t for t in TOOLS if t.name == "prsm_analyze")
        assert "query" in analyze.inputSchema.get("required", [])

    def test_all_handlers_registered(self):
        for tool in TOOLS:
            assert tool.name in TOOL_HANDLERS, f"No handler for {tool.name}"


class TestMCPServer:
    def test_create_server(self):
        server = create_server()
        assert server is not None

    def test_server_name(self):
        server = create_server()
        assert server.name == "prsm"


class TestToolHandlers:
    @pytest.mark.asyncio
    async def test_quote_handler(self):
        result = await handle_prsm_quote({
            "query": "EV trends",
            "shard_count": 3,
            "hardware_tier": "t2",
        })
        assert "Cost Estimate" in result
        assert "FTNS" in result
        assert "Total" in result

    @pytest.mark.asyncio
    async def test_quote_handler_defaults(self):
        result = await handle_prsm_quote({"query": "test"})
        assert "Cost Estimate" in result

    @pytest.mark.asyncio
    async def test_benchmark_handler(self):
        result = await handle_prsm_hardware_benchmark({})
        assert "Benchmark" in result
        assert "CPU" in result
        assert "TFLOPS" in result
        assert "Tier" in result

    @pytest.mark.asyncio
    async def test_analyze_handler_no_node(self):
        """Analyze should gracefully handle no running node."""
        result = await handle_prsm_analyze({"query": "test", "budget_ftns": 1.0})
        # Should either return a result or a helpful error
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_analyze_rejects_zero_budget(self):
        """Analyze must reject zero budget with helpful message."""
        result = await handle_prsm_analyze({"query": "test", "budget_ftns": 0})
        assert "FTNS" in result
        assert "budget" in result.lower() or "prsm_quote" in result

    @pytest.mark.asyncio
    async def test_analyze_rejects_negative_budget(self):
        """Analyze must reject negative budget."""
        result = await handle_prsm_analyze({"query": "test", "budget_ftns": -5.0})
        assert "FTNS" in result

    @pytest.mark.asyncio
    async def test_node_status_no_node(self):
        """Status should gracefully handle no running node."""
        result = await handle_prsm_node_status({})
        assert len(result) > 0


class TestCLICommand:
    def test_mcp_server_command_exists(self):
        from click.testing import CliRunner
        from prsm.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["mcp-server", "--help"])
        assert result.exit_code == 0
        assert "MCP" in result.output or "mcp" in result.output

    def test_compute_run_rejects_zero_budget_query(self):
        """CLI should reject --query with --budget 0."""
        from click.testing import CliRunner
        from prsm.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["compute", "run", "--query", "test", "--budget", "0"])
        assert result.exit_code != 0
        assert "FTNS" in result.output or "budget" in result.output.lower()


class TestAgentCreationTools:
    @pytest.mark.asyncio
    async def test_create_agent_with_instructions(self):
        result = await handle_prsm_create_agent({
            "query": "Count EV registrations in NC",
            "instructions": [
                {"op": "filter", "field": "state", "value": "NC"},
                {"op": "filter", "field": "vehicle_type", "value": "EV"},
                {"op": "count"},
            ],
            "budget_ftns": 5.0,
        })
        assert "Agent Manifest Created" in result
        assert "3" in result  # 3 operations
        assert "filter" in result
        assert "count" in result
        assert "Manifest JSON" in result

    @pytest.mark.asyncio
    async def test_create_agent_rejects_unknown_op(self):
        result = await handle_prsm_create_agent({
            "query": "test",
            "instructions": [{"op": "explode_database"}],
            "budget_ftns": 1.0,
        })
        assert "Unknown operation" in result

    @pytest.mark.asyncio
    async def test_create_agent_rejects_zero_budget(self):
        result = await handle_prsm_create_agent({
            "query": "test",
            "instructions": [{"op": "count"}],
            "budget_ftns": 0,
        })
        assert "FTNS" in result

    @pytest.mark.asyncio
    async def test_create_agent_requires_instructions(self):
        result = await handle_prsm_create_agent({
            "query": "test",
            "instructions": [],
            "budget_ftns": 1.0,
        })
        assert "At least one instruction" in result

    @pytest.mark.asyncio
    async def test_dispatch_agent_rejects_empty_manifest(self):
        result = await handle_prsm_dispatch_agent({
            "instructions_json": "",
            "budget_ftns": 1.0,
        })
        assert "Missing" in result or "prsm_create_agent" in result

    @pytest.mark.asyncio
    async def test_dispatch_agent_validates_manifest(self):
        from prsm.compute.agents.instruction_set import InstructionManifest, AgentInstruction, AgentOp

        manifest = InstructionManifest(
            query="Count records",
            instructions=[AgentInstruction(op=AgentOp.COUNT)],
        )
        result = await handle_prsm_dispatch_agent({
            "instructions_json": manifest.to_json(),
            "budget_ftns": 1.0,
        })
        # Will fail to connect to node but should validate the manifest
        assert "1 operations" in result or "dispatch failed" in result

    @pytest.mark.asyncio
    async def test_dispatch_agent_routes_through_compute_forge_qo_path(self):
        """B8 unhide pass 2: confirm the handler forwards manifest.query
        to /compute/forge and renders the QO-shaped response (route =
        'qo_swarm'). The QO dispatch path was wired in B8 unhide pass 1."""
        from unittest.mock import patch
        from prsm.compute.agents.instruction_set import (
            InstructionManifest, AgentInstruction, AgentOp,
        )

        manifest = InstructionManifest(
            query="Count records",
            instructions=[AgentInstruction(op=AgentOp.COUNT)],
        )
        captured = {}

        async def _fake_call(method, path, body=None):
            captured["method"] = method
            captured["path"] = path
            captured["body"] = body
            return {
                "job_id": "forge-test-xyz",
                "query": "Count records",
                "route": "qo_swarm",
                "response": '{"count": 7}',
                "result": {
                    "status": "success",
                    "route": "qo_swarm",
                    "aggregator_node_id": "agg-test",
                },
                "budget_ftns": 1.0,
            }

        with patch("prsm.mcp_server._call_node_api", side_effect=_fake_call):
            result = await handle_prsm_dispatch_agent({
                "instructions_json": manifest.to_json(),
                "budget_ftns": 1.0,
            })

        # /compute/forge was called with the manifest's query string.
        assert captured["method"] == "POST"
        assert captured["path"] == "/compute/forge"
        assert captured["body"]["query"] == "Count records"
        # Handler-rendered output surfaces the QO response.
        assert "Agent Dispatched" in result
        assert "Count records" in result
        assert '{"count": 7}' in result
        # Route surfaced in the cost-footer extra fields.
        assert "qo_swarm" in result


class TestFullToolSuite:
    @pytest.mark.asyncio
    async def test_yield_estimate_handler(self):
        from prsm.mcp_server import handle_prsm_yield_estimate
        result = await handle_prsm_yield_estimate({"hours_per_day": 8, "stake_amount": 1000})
        assert "Yield" in result
        assert "FTNS" in result

    @pytest.mark.asyncio
    async def test_revenue_split_handler(self):
        from prsm.mcp_server import handle_prsm_revenue_split
        result = await handle_prsm_revenue_split({"total_payment": 100, "has_data_owner": True})
        assert "Data Owner" in result
        assert "80%" in result

    @pytest.mark.asyncio
    async def test_revenue_split_without_data(self):
        from prsm.mcp_server import handle_prsm_revenue_split
        result = await handle_prsm_revenue_split({"total_payment": 100, "has_data_owner": False})
        assert "Compute" in result
        assert "Data Owner" not in result

    @pytest.mark.asyncio
    async def test_stake_handler(self):
        # sp932 — refreshed for the sp904/sp906 staking model: staking confers
        # UTILITY benefits (lock-based service discount + dispatch priority), NOT
        # token yield, so the old ProsumerTier "DEDICATED"/"1.5x yield" preview is
        # gone. With no lock the preview says so; with a lock it shows the benefit.
        from prsm.mcp_server import handle_prsm_stake

        no_lock = await handle_prsm_stake({"amount": 1000})
        assert "Staking Preview" in no_lock
        assert "no utility benefit" in no_lock

        locked = await handle_prsm_stake({"amount": 1000, "lock_period_days": 365})
        assert "Staking Preview" in locked
        assert "discount" in locked
        assert "priority" in locked

    @pytest.mark.asyncio
    async def test_privacy_status_handler(self):
        from prsm.mcp_server import handle_prsm_privacy_status
        result = await handle_prsm_privacy_status({})
        assert "privacy" in result.lower() or "Privacy" in result

    @pytest.mark.asyncio
    async def test_training_status_handler(self):
        from prsm.mcp_server import handle_prsm_training_status
        result = await handle_prsm_training_status({})
        assert "Training" in result or "training" in result


# ── prsm_inference (Phase 3.x.1 Task 6) ─────────────────────────────────────


class TestPrsmInferenceToolDefinition:
    """The prsm_inference tool surfaces correctly via MCP list_tools."""

    def _tool(self):
        return next(t for t in TOOLS if t.name == "prsm_inference")

    def test_tool_exists(self):
        names = [t.name for t in TOOLS]
        assert "prsm_inference" in names

    def test_tool_description_mentions_tee(self):
        # Description must communicate the verifiable-receipt + TEE properties
        desc = self._tool().description.lower()
        assert "tee" in desc
        assert "receipt" in desc
        assert "privacy" in desc

    def test_tool_description_mentions_both_privacy_layers(self):
        desc = self._tool().description.lower()
        # Vision §7 two-layer privacy must be explicit in description
        assert "content_tier" in desc or "content tier" in desc
        assert "privacy_tier" in desc or "privacy tier" in desc

    def test_input_schema_required_fields(self):
        schema = self._tool().inputSchema
        assert schema["required"] == ["prompt"]

    def test_input_schema_privacy_tier_enum(self):
        schema = self._tool().inputSchema
        privacy = schema["properties"]["privacy_tier"]
        assert set(privacy["enum"]) == {"none", "standard", "high", "maximum"}

    def test_input_schema_content_tier_enum(self):
        schema = self._tool().inputSchema
        content = schema["properties"]["content_tier"]
        assert set(content["enum"]) == {"A", "B", "C"}

    def test_input_schema_minimum_budget(self):
        schema = self._tool().inputSchema
        budget = schema["properties"]["budget_ftns"]
        assert budget["minimum"] == 0.01

    def test_input_schema_temperature_bounds(self):
        schema = self._tool().inputSchema
        temp = schema["properties"]["temperature"]
        assert temp["minimum"] == 0.0
        assert temp["maximum"] == 2.0

    def test_handler_registered(self):
        assert TOOL_HANDLERS["prsm_inference"] is handle_prsm_inference


class TestPrsmInferenceHandler:
    """Handler behavior for prsm_inference."""

    @pytest.mark.asyncio
    async def test_missing_prompt_returns_error(self):
        result = await handle_prsm_inference({})
        assert "Missing required 'prompt'" in result

    @pytest.mark.asyncio
    async def test_zero_budget_rejected(self):
        result = await handle_prsm_inference({
            "prompt": "Hello",
            "budget_ftns": 0,
        })
        assert "FTNS budget" in result
        assert "0.01" in result

    @pytest.mark.asyncio
    async def test_negative_budget_rejected(self):
        result = await handle_prsm_inference({
            "prompt": "Hello",
            "budget_ftns": -1,
        })
        assert "FTNS budget" in result

    @pytest.mark.asyncio
    async def test_below_minimum_budget_rejected(self):
        result = await handle_prsm_inference({
            "prompt": "Hello",
            "budget_ftns": 0.001,  # below 0.01 minimum
        })
        assert "below minimum" in result
        assert "0.01" in result

    @pytest.mark.asyncio
    async def test_node_unreachable_returns_helpful_error(self):
        # When the node API is unreachable, return clear error mentioning Task 5
        with patch("prsm.mcp_server._call_node_api") as mock_call:
            mock_call.side_effect = Exception("Connection refused")
            result = await handle_prsm_inference({
                "prompt": "Hello",
                "budget_ftns": 1.0,
            })
        assert "PRSM inference failed" in result
        assert "Connection refused" in result
        # Helpful diagnostic mentions both common causes
        assert "node not running" in result.lower() or "prsm node start" in result
        assert "compute/inference" in result or "Task 5" in result

    @pytest.mark.asyncio
    async def test_api_error_response_surfaced(self):
        # API returns a structured error
        with patch("prsm.mcp_server._call_node_api") as mock_call:
            mock_call.return_value = {"error": "Unknown model_id: foo"}
            result = await handle_prsm_inference({
                "prompt": "Hello",
                "model_id": "foo",
                "budget_ftns": 1.0,
            })
        assert "Unknown model_id" in result
        assert "rejected" in result.lower()

    @pytest.mark.asyncio
    async def test_successful_response_formats_receipt(self):
        # Mock a complete successful API response
        mock_response = {
            "success": True,
            "output": "The answer is 42.",
            "receipt": {
                "job_id": "job-abc-123",
                "model_id": "mock-llama-3-8b",
                "privacy_tier": "standard",
                "content_tier": "A",
                "tee_type": "software",
                "epsilon_spent": 8.0,
                "cost_ftns": "0.5",
                "duration_seconds": 1.2,
                "settler_node_id": "node-xyz",
                "settler_signature": "deadbeef",
            },
        }
        with patch("prsm.mcp_server._call_node_api") as mock_call:
            mock_call.return_value = mock_response
            result = await handle_prsm_inference({
                "prompt": "What is 2+2?",
                "model_id": "mock-llama-3-8b",
                "budget_ftns": 1.0,
            })

        # Check core output present
        assert "The answer is 42." in result
        # Check cost reconciliation footer per design plan §3.4
        assert "job-abc-123" in result
        assert "mock-llama-3-8b" in result
        assert "0.5" in result
        assert "node-xyz" in result
        # Check signed-receipt verification hint surfaced
        assert "verify_receipt" in result

    @pytest.mark.asyncio
    async def test_handler_passes_all_args_to_api(self):
        """Verify the handler forwards every input field to the API."""
        captured: dict = {}

        async def capture_payload(method, path, data=None):
            captured["method"] = method
            captured["path"] = path
            captured["data"] = data
            return {"success": True, "output": "ok", "receipt": {}}

        with patch("prsm.mcp_server._call_node_api", side_effect=capture_payload):
            await handle_prsm_inference({
                "prompt": "test prompt",
                "model_id": "mock-mistral-7b",
                "budget_ftns": 2.5,
                "privacy_tier": "high",
                "content_tier": "B",
                "max_tokens": 200,
                "temperature": 0.7,
            })

        assert captured["method"] == "POST"
        assert captured["path"] == "/compute/inference"
        payload = captured["data"]
        assert payload["prompt"] == "test prompt"
        assert payload["model_id"] == "mock-mistral-7b"
        assert payload["budget_ftns"] == 2.5
        assert payload["privacy_tier"] == "high"
        assert payload["content_tier"] == "B"
        assert payload["max_tokens"] == 200
        assert payload["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_handler_omits_optional_args_when_unset(self):
        """Optional fields should not be included if not provided by caller."""
        captured: dict = {}

        async def capture_payload(method, path, data=None):
            captured["data"] = data
            return {"success": True, "output": "ok", "receipt": {}}

        with patch("prsm.mcp_server._call_node_api", side_effect=capture_payload):
            await handle_prsm_inference({
                "prompt": "test",
                "budget_ftns": 1.0,
            })

        payload = captured["data"]
        # Required + defaults present
        assert "prompt" in payload
        assert "model_id" in payload
        assert "budget_ftns" in payload
        # Optional absent
        assert "max_tokens" not in payload
        assert "temperature" not in payload
