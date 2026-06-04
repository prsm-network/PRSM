"""Sprint 1016 — single-node mock streaming inference.

Makes /compute/inference/stream work end-to-end on ONE node without real GPUs /
model files / a multi-host swarm, via a RUNNABLE mock ParallaxScheduledExecutor
(opt-in: PRSM_INFERENCE_EXECUTOR=mock-streaming). The mock threads synthetic
tokens through the GENUINE streaming code path (pre-execute gates, trust-filtered
pool, Phase-1 allocation, the execute_chain_streaming drive loop, receipt
signing), so streaming is demonstrable + testable in dev/dogfood. Like sp438's
non-streaming MockInferenceExecutor, the crypto is zero-/software-tier honest
scope and MUST NOT be trusted by real verifiers.
"""
from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from prsm.compute.inference.mock_streaming import (
    MOCK_MODEL_ID,
    MockStreamingChainExecutor,
    build_mock_streaming_executor,
)
from prsm.compute.inference.models import (
    ContentTier,
    InferenceRequest,
    InferenceResult,
)
from prsm.compute.inference.parallax_executor import (
    ChainExecutionResult,
    InferenceTokenEvent,
    ParallaxScheduledExecutor,
)
from prsm.compute.chain_rpc.client import StreamToken
from prsm.compute.tee.models import PrivacyLevel
from prsm.node.identity import generate_node_identity


def _drain(executor, request):
    async def _run():
        out = []
        async for item in executor.execute_streaming(request):
            out.append(item)
        return out
    return asyncio.run(_run())


def _request(privacy=PrivacyLevel.NONE):
    return InferenceRequest(
        prompt="hello",
        model_id=MOCK_MODEL_ID,
        budget_ftns=Decimal("10.0"),
        privacy_tier=privacy,
        content_tier=ContentTier.A,
    )


# ── chain-executor streaming contract ───────────────────────────────────────


def test_chain_executor_streaming_contract():
    ce = MockStreamingChainExecutor(deltas=["a", "b", "c"], finish_reason="stop")
    items = list(ce.execute_chain_streaming(request=None, chain=None))
    tokens = [i for i in items if isinstance(i, StreamToken)]
    terminals = [i for i in items if isinstance(i, ChainExecutionResult)]
    assert [t.text_delta for t in tokens] == ["a", "b", "c"]
    # sequence_index 0-indexed, no gaps; finish_reason only on the last token
    assert [t.sequence_index for t in tokens] == [0, 1, 2]
    assert tokens[0].finish_reason is None and tokens[1].finish_reason is None
    assert tokens[-1].finish_reason == "stop"
    assert len(terminals) == 1
    # terminal output == joined deltas (the receipt-hash invariant)
    assert terminals[0].output == "abc"


def test_chain_executor_has_unary_too():
    ce = MockStreamingChainExecutor(deltas=["x", "y"])
    res = ce.execute_chain(request=None, chain=None)
    assert isinstance(res, ChainExecutionResult)
    assert res.output == "xy"


# ── factory ─────────────────────────────────────────────────────────────────


def test_factory_builds_parallax_executor():
    ex = build_mock_streaming_executor(generate_node_identity("mock-node"))
    assert isinstance(ex, ParallaxScheduledExecutor)
    assert hasattr(ex, "execute_streaming")


def test_factory_requires_identity():
    with pytest.raises(Exception):
        build_mock_streaming_executor(None)


def test_factory_catalog_has_mock_model():
    ex = build_mock_streaming_executor(generate_node_identity("mock-node"))
    assert MOCK_MODEL_ID in ex._catalog


# ── end-to-end through the REAL ParallaxScheduledExecutor streaming path ─────


def test_end_to_end_streaming_yields_tokens_then_success():
    ex = build_mock_streaming_executor(generate_node_identity("settler"))
    events = _drain(ex, _request(privacy=PrivacyLevel.NONE))

    tokens = [e for e in events if isinstance(e, InferenceTokenEvent)]
    terminals = [e for e in events if isinstance(e, InferenceResult)]

    assert len(tokens) >= 1, "no token events emitted"
    assert len(terminals) == 1, "expected exactly one terminal InferenceResult"
    result = terminals[0]
    assert result.success is True, f"streaming failed: {result.error}"
    # output is the concatenation of the synthetic token deltas
    assert result.output == "".join(t.text_delta for t in tokens)
    # a signed receipt was produced + flagged as streamed
    assert result.receipt is not None
    assert result.receipt.streamed_output is True
    assert result.receipt.settler_signature  # non-empty (signed by node identity)


def test_end_to_end_streaming_standard_tier_also_runs():
    # The mock GPU is hardware-attested (tier-sgx), so higher privacy tiers also
    # pass the tier gate + allocate + stream (synthetic crypto regardless).
    ex = build_mock_streaming_executor(generate_node_identity("settler2"))
    events = _drain(ex, _request(privacy=PrivacyLevel.STANDARD))
    terminals = [e for e in events if isinstance(e, InferenceResult)]
    assert len(terminals) == 1
    assert terminals[0].success is True, terminals[0].error


def test_unknown_model_fails_cleanly_not_crash():
    ex = build_mock_streaming_executor(generate_node_identity("settler3"))
    bad = InferenceRequest(
        prompt="x", model_id="no-such-model", budget_ftns=Decimal("10.0"),
        privacy_tier=PrivacyLevel.NONE, content_tier=ContentTier.A,
    )
    events = _drain(ex, bad)
    # gate failure → a single terminal failure, no token events, no crash
    tokens = [e for e in events if isinstance(e, InferenceTokenEvent)]
    terminals = [e for e in events if isinstance(e, InferenceResult)]
    assert tokens == []
    assert len(terminals) == 1 and terminals[0].success is False
