"""Sprint 1019 — real model inference by default on a single node (Tier 2).

Closes the biggest user-facing gap: a default install returned HTTP 503, and the
only one-env options emitted SYNTHETIC tokens (sp438 mock / sp1016 mock-streaming).
This wires a RUNNABLE single-node ParallaxScheduledExecutor whose chain executor
runs a REAL HuggingFace model (gpt2/distilgpt2) IN-PROCESS via transformers
`model.generate` — no RPC, no multi-host swarm, no ~8-env-var operator ceremony,
no Base-anchor identity. Opt-in: PRSM_INFERENCE_EXECUTOR=local.

Real (greedy/deterministic) tokens flow through the GENUINE executor path
(pre-execute gates, trust-filtered single-GPU pool, Phase-1 allocation, the
execute_chain / execute_chain_streaming drive loops, signed-receipt construction).
Honest scope: trust is software-tier (TEEType.SOFTWARE, vendor_verified=false) —
real hardware TEE is Tier 3. The MODEL output is real; the privacy attestation is
not hardware-backed.

These tests load distilgpt2 from the local HF cache (offline) — a few seconds.
"""
from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from prsm.compute.chain_rpc.client import StreamToken
from prsm.compute.inference.local_inference import (
    LocalHuggingFaceChainExecutor,
    build_local_inference_executor,
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
from prsm.compute.tee.models import PrivacyLevel
from prsm.node.identity import generate_node_identity

MODEL = "distilgpt2"  # 6-layer, cached locally, fast on CPU
_MOCK_STR = "Mock streaming inference from PRSM."


@pytest.fixture(autouse=True)
def _offline_cached_model(monkeypatch):
    """sp1184 — these tests load distilgpt2 from the LOCAL HF cache and must NOT hit the
    network. The day-one production default now allows a hub download (offline=False); pin
    this hermetic module to offline so it keeps using the cache deterministically."""
    monkeypatch.setenv("PRSM_LOCAL_INFERENCE_OFFLINE", "1")


def _req(prompt="The capital of France is", privacy=PrivacyLevel.NONE, max_tokens=6):
    return InferenceRequest(
        prompt=prompt,
        model_id=MODEL,
        budget_ftns=Decimal("100.0"),
        privacy_tier=privacy,
        content_tier=ContentTier.A,
        max_tokens=max_tokens,
    )


def _drain(executor, request):
    async def _run():
        out = []
        async for item in executor.execute_streaming(request):
            out.append(item)
        return out
    return asyncio.run(_run())


# ── the chain executor produces REAL model output ───────────────────────────


def test_streaming_yields_real_deterministic_tokens():
    ce = LocalHuggingFaceChainExecutor(model_id=MODEL, max_tokens=6)
    items = list(ce.execute_chain_streaming(request=_req(), chain=None))
    tokens = [i for i in items if isinstance(i, StreamToken)]
    terminals = [i for i in items if isinstance(i, ChainExecutionResult)]
    assert len(tokens) >= 1, "no real tokens generated"
    assert len(terminals) == 1
    out = terminals[0].output
    assert out and out != _MOCK_STR, "output is empty or the synthetic mock string"
    # receipt-hash invariant: terminal output == concatenation of token deltas
    assert "".join(t.text_delta for t in tokens) == out
    # sequence indices 0-indexed, no gaps; finish_reason only on the last
    assert [t.sequence_index for t in tokens] == list(range(len(tokens)))
    assert tokens[-1].finish_reason is not None
    assert all(t.finish_reason is None for t in tokens[:-1])
    # determinism (greedy decode) — a REAL model gives the same output twice,
    # distinguishing it from random/synthetic output.
    ce2 = LocalHuggingFaceChainExecutor(model_id=MODEL, max_tokens=6)
    out2 = [i for i in ce2.execute_chain_streaming(request=_req(), chain=None)
            if isinstance(i, ChainExecutionResult)][0].output
    assert out2 == out


def test_unary_execute_chain_real_output():
    ce = LocalHuggingFaceChainExecutor(model_id=MODEL, max_tokens=6)
    res = ce.execute_chain(request=_req(), chain=None)
    assert isinstance(res, ChainExecutionResult)
    assert res.output and res.output != _MOCK_STR


def test_different_prompts_give_different_output():
    ce = LocalHuggingFaceChainExecutor(model_id=MODEL, max_tokens=8)
    a = ce.execute_chain(request=_req(prompt="The ocean is", max_tokens=8), chain=None).output
    b = ce.execute_chain(request=_req(prompt="Quantum physics studies", max_tokens=8), chain=None).output
    assert a != b, "output did not vary with the prompt — not a real model"


# ── factory + end-to-end through the REAL ParallaxScheduledExecutor ──────────


def test_factory_builds_parallax_executor_with_model():
    ex = build_local_inference_executor(generate_node_identity("ln"), model_id=MODEL)
    assert isinstance(ex, ParallaxScheduledExecutor)
    assert MODEL in ex._catalog


def test_factory_requires_identity():
    with pytest.raises(Exception):
        build_local_inference_executor(None, model_id=MODEL)


def test_end_to_end_streaming_real_inference():
    ex = build_local_inference_executor(generate_node_identity("settler"), model_id=MODEL, max_tokens=6)
    events = _drain(ex, _req(privacy=PrivacyLevel.NONE))
    tokens = [e for e in events if isinstance(e, InferenceTokenEvent)]
    terminals = [e for e in events if isinstance(e, InferenceResult)]
    assert len(tokens) >= 1
    assert len(terminals) == 1
    result = terminals[0]
    assert result.success is True, f"streaming failed: {result.error}"
    assert result.output and result.output != _MOCK_STR
    assert result.output == "".join(t.text_delta for t in tokens)
    assert result.receipt is not None
    assert result.receipt.streamed_output is True
    assert result.receipt.settler_signature  # really signed by node identity


def test_end_to_end_unary_real_inference():
    ex = build_local_inference_executor(generate_node_identity("settler2"), model_id=MODEL, max_tokens=6)
    async def _run():
        return await ex.execute(_req(privacy=PrivacyLevel.NONE))
    result = asyncio.run(_run())
    assert isinstance(result, InferenceResult)
    assert result.success is True, f"unary failed: {result.error}"
    assert result.output and result.output != _MOCK_STR
    assert result.receipt is not None and result.receipt.settler_signature
