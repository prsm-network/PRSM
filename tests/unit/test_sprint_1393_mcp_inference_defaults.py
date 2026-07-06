"""Sprint 1393 — the flagship MCP prsm_inference tool works out of the box.

Two defaults dead-ended a fresh LLM client: model_id="mock-llama-3-8b" (no node serves it →
"Unknown model_id") and privacy_tier="standard" (needs a hardware-TEE node → tier-gate refusal on a
software node). Both now default to servable values.
"""
from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import handle_prsm_inference


@pytest.mark.asyncio
async def test_inference_defaults_are_servable():
    with patch("prsm.mcp_server._call_node_api",
               new=AsyncMock(return_value={"output": " Paris", "success": True})) as mc:
        result = await handle_prsm_inference({"prompt": "The capital of France is"})
    args, _ = mc.await_args
    payload = args[2]                                  # _call_node_api(method, endpoint, payload)
    assert payload.get("model_id") == "distilgpt2"     # was mock-llama-3-8b
    assert payload.get("privacy_tier") == "none"       # was standard (TEE-gated)
    assert "Paris" in (result if isinstance(result, str) else str(result))


@pytest.mark.asyncio
async def test_explicit_overrides_still_pass_through():
    with patch("prsm.mcp_server._call_node_api",
               new=AsyncMock(return_value={"output": "x", "success": True})) as mc:
        await handle_prsm_inference(
            {"prompt": "hi", "model_id": "Qwen/Qwen2.5-14B-Instruct", "privacy_tier": "high"})
    payload = mc.await_args.args[2]
    assert payload.get("model_id") == "Qwen/Qwen2.5-14B-Instruct"
    assert payload.get("privacy_tier") == "high"       # explicit values untouched


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
