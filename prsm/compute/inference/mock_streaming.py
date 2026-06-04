"""Sprint 1016 — single-node mock streaming inference.

Assembles a RUNNABLE ``ParallaxScheduledExecutor`` from mock components so
``/compute/inference/stream`` yields synthetic tokens end-to-end on a single node
— no real GPUs, model files, or multi-host swarm. Opt-in via
``PRSM_INFERENCE_EXECUTOR=mock-streaming``.

The synthetic tokens flow through the GENUINE streaming code path (pre-execute
gates, trust-filtered pool, Phase-1 allocation, the ``execute_chain_streaming``
drive loop, signed-receipt construction), so streaming is demonstrable and
testable in dev / dogfood without production inference infrastructure. Like
sp438's non-streaming ``MockInferenceExecutor``, the cryptographic fields are
zero-/software-tier honest scope and MUST NOT be trusted by real verifiers — a
mock receipt advertises a privacy tier it did not actually deliver.

The lone mock GPU is THIS node (``node_identity.node_id``): staked, hardware-tier
attested, and anchor-registered so the trust stack admits it and the allocator
places all model layers on it as a single-stage pipeline.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Union

from prsm.compute.chain_rpc.client import StreamToken
from prsm.compute.inference.parallax_executor import (
    ChainExecutionResult,
    ParallaxScheduledExecutor,
)
from prsm.compute.parallax_scheduling.model_info import ModelInfo
from prsm.compute.parallax_scheduling.prsm_request_router import (
    InMemoryProfileSource,
    ProfileSnapshot,
)
from prsm.compute.parallax_scheduling.prsm_types import ParallaxGPU
from prsm.compute.parallax_scheduling.trust_adapter import (
    AnchorVerifyAdapter,
    ConsensusMismatchHook,
    StakeWeightedTrustAdapter,
    TierGateAdapter,
    TrustStack,
)
from prsm.compute.tee.models import TEEType

# The single model this mock executor advertises. Streaming requests MUST use
# this model_id (the catalog lookup in the pre-execute gates is exact-match).
MOCK_MODEL_ID = "mock-model"

_MOCK_NUM_LAYERS = 4
_MOCK_DELTAS = ["Mock", " streaming", " inference", " from", " PRSM", "."]


class _MockAnchor:
    """Anchor stub — the lone mock node is 'registered'."""

    def __init__(self, registered: Dict[str, str]) -> None:
        self._registered = dict(registered)

    def lookup(self, node_id: str) -> Optional[str]:
        return self._registered.get(node_id)


class _MockStakeLookup:
    def __init__(self, stakes: Dict[str, int]) -> None:
        self._stakes = dict(stakes)

    def get_stake(self, node_id: str) -> int:
        return self._stakes.get(node_id, 0)


class _NoopSubmitter:
    def __call__(self, record: Any) -> None:  # noqa: ARG002
        return None


class MockStreamingChainExecutor:
    """Synthetic streaming chain executor.

    ``execute_chain_streaming`` yields a fixed set of ``StreamToken``s (0-indexed,
    ``finish_reason`` only on the last) followed by exactly one terminal
    ``ChainExecutionResult`` whose ``output`` is the joined deltas (the
    receipt-hash invariant). Software-tier, zero-epsilon (honest mock scope).
    Also implements the unary ``execute_chain`` for the non-streaming path.
    """

    def __init__(
        self,
        *,
        deltas: Optional[List[str]] = None,
        finish_reason: str = "stop",
    ) -> None:
        self._deltas = list(deltas) if deltas is not None else list(_MOCK_DELTAS)
        self._finish_reason = finish_reason

    def _terminal(self) -> ChainExecutionResult:
        return ChainExecutionResult(
            output="".join(self._deltas),
            duration_seconds=0.01,
            tee_attestation=b"\x00" * 32,  # zero-filled — honest mock scope
            tee_type=TEEType.SOFTWARE,
            epsilon_spent=0.0,
        )

    def execute_chain(self, *, request: Any, chain: Any) -> ChainExecutionResult:  # noqa: ARG002
        return self._terminal()

    def execute_chain_streaming(
        self, *, request: Any, chain: Any,  # noqa: ARG002
    ) -> Iterator[Union[StreamToken, ChainExecutionResult]]:
        last = len(self._deltas) - 1
        for i, delta in enumerate(self._deltas):
            yield StreamToken(
                sequence_index=i,
                text_delta=delta,
                finish_reason=self._finish_reason if i == last else None,
            )
        yield self._terminal()


def _mock_model_info(num_layers: int = _MOCK_NUM_LAYERS) -> ModelInfo:
    return ModelInfo(
        model_name=MOCK_MODEL_ID,
        mlx_model_name=MOCK_MODEL_ID + "-mlx",
        head_size=64,
        hidden_dim=512,
        intermediate_dim=2048,
        num_attention_heads=8,
        num_kv_heads=8,
        vocab_size=32000,
        num_layers=num_layers,
    )


def build_mock_streaming_executor(node_identity: Any) -> ParallaxScheduledExecutor:
    """Assemble a runnable single-node mock ``ParallaxScheduledExecutor``.

    The lone mock GPU is this node; the trust stack admits it (registered +
    staked + hardware-tier attested so all privacy tiers pass the tier gate), and
    the allocator places all ``_MOCK_NUM_LAYERS`` layers on it as a single-stage
    pipeline. Receipts are signed by ``node_identity`` (real signature over
    zero-filled mock crypto fields — honest scope).
    """
    if node_identity is None or not hasattr(node_identity, "node_id"):
        raise RuntimeError(
            "build_mock_streaming_executor requires a NodeIdentity (.node_id)"
        )
    node_id = node_identity.node_id

    gpu = ParallaxGPU(
        node_id=node_id,
        region="mock-region",
        layer_capacity=max(8, _MOCK_NUM_LAYERS),  # ≥ model layers → fits 1 GPU
        stake_amount=10**18,
        tier_attestation="tier-sgx",  # hardware tier → admits NONE..MAXIMUM
        tflops_fp16=100.0,
        memory_gb=80.0,
        memory_bandwidth_gbps=2000.0,
    )
    pool = [gpu]

    trust = TrustStack(
        anchor_verify=AnchorVerifyAdapter(
            anchor=_MockAnchor({node_id: "pk-" + node_id}),
        ),
        tier_gate=TierGateAdapter(),
        profile_source=StakeWeightedTrustAdapter(
            inner=InMemoryProfileSource(
                snapshots={
                    node_id: ProfileSnapshot(
                        node_id=node_id,
                        layer_latency_ms=10.0,
                        rtt_to_peers={},
                        timestamp_unix=1000.0,
                    ),
                },
            ),
            stake_lookup=_MockStakeLookup({node_id: 10**18}),
        ),
        consensus_hook=ConsensusMismatchHook(
            submitter=_NoopSubmitter(), sample_rate=0.0,
        ),
    )

    catalog: Dict[str, ModelInfo] = {MOCK_MODEL_ID: _mock_model_info()}

    def _provider() -> List[ParallaxGPU]:
        return list(pool)

    return ParallaxScheduledExecutor(
        gpu_pool_provider=_provider,
        trust_stack=trust,
        model_catalog=catalog,
        chain_executor=MockStreamingChainExecutor(),
        node_identity=node_identity,
    )
