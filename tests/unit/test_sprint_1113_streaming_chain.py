"""Sprint 1113 — Domain-03 F1/F2 brick 7: per-stage activation proofs on the
streaming/sharded decode path.

The unary path (bricks 3-4) was covered; this extends the producer + assembly to the
sharded decode path. The sharded INCREMENTAL/VERIFY sign site now emits the same
self-securing per-stage proof (symmetric with unary), and _build_sharded_chain_result
assembles a StageActivationChain for the streamed receipt from the last complete
iteration's outcomes (a representative full stage pass; per-token tail signatures already
cover generation, and brick 6 binds the chain to request_id).
"""
from __future__ import annotations

import pytest

from prsm.compute.chain_rpc.client import (
    RpcChainExecutor,
    StageOutcome,
    _build_stage_activation_chain,
)
from prsm.compute.chain_rpc.protocol import DecodeMode
from prsm.compute.inference.stage_activation_proof import activation_hash
from prsm.compute.tee.models import TEEType


def _outcome(idx, node, hin, hout, sig="sig"):
    return StageOutcome(
        stage_index=idx, stage_node_id=node, duration_seconds=0.1,
        tee_attestation=b"a", tee_type=TEEType.SOFTWARE, epsilon_spent=0.0,
        stage_input_activation_hash=hin, stage_output_activation_hash=hout,
        stage_activation_proof_b64=sig)


def _iteration_with_proofs(seed):
    h0 = activation_hash(f"{seed}-0".encode())
    h1 = activation_hash(f"{seed}-1".encode())
    h2 = activation_hash(f"{seed}-2".encode())
    return [_outcome(0, "n0", h0, h1), _outcome(1, "n1", h1, h2)]


def _bare_executor():
    return RpcChainExecutor.__new__(RpcChainExecutor)


def test_sharded_result_carries_chain_from_last_iteration():
    ex = _bare_executor()
    iters = [_iteration_with_proofs("prefill"), _iteration_with_proofs("tok1")]
    modes = [DecodeMode.PREFILL, DecodeMode.INCREMENTAL]
    result = ex._build_sharded_chain_result(
        output_text="hi", per_iteration_outcomes=iters,
        per_iteration_decode_modes=modes, request_id="req-stream")
    assert result.stage_activation_chain is not None
    assert result.stage_activation_chain.request_id == "req-stream"
    assert result.stage_activation_chain.topology_node_ids() == ["n0", "n1"]


def test_sharded_result_falls_back_to_an_earlier_complete_iteration():
    """If the LAST iteration lacks proofs (e.g. a chunked-PREFILL-only tail) but an
    earlier one is complete, use the earlier complete iteration rather than dropping."""
    ex = _bare_executor()
    incomplete = [
        StageOutcome(0, "n0", 0.1, b"a", TEEType.SOFTWARE, 0.0),  # no proof
        StageOutcome(1, "n1", 0.1, b"a", TEEType.SOFTWARE, 0.0),
    ]
    iters = [_iteration_with_proofs("good"), incomplete]
    modes = [DecodeMode.PREFILL, DecodeMode.INCREMENTAL]
    result = ex._build_sharded_chain_result(
        output_text="hi", per_iteration_outcomes=iters,
        per_iteration_decode_modes=modes, request_id="req-x")
    assert result.stage_activation_chain is not None  # used the complete earlier one


def test_sharded_result_omits_chain_when_no_iteration_complete():
    ex = _bare_executor()
    incomplete = [StageOutcome(0, "n0", 0.1, b"a", TEEType.SOFTWARE, 0.0)]
    result = ex._build_sharded_chain_result(
        output_text="hi", per_iteration_outcomes=[incomplete],
        per_iteration_decode_modes=[DecodeMode.PREFILL], request_id="r")
    assert result.stage_activation_chain is None


def test_empty_iterations_yields_no_chain():
    ex = _bare_executor()
    result = ex._build_sharded_chain_result(
        output_text="", per_iteration_outcomes=[],
        per_iteration_decode_modes=[], request_id="r")
    assert result.stage_activation_chain is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
