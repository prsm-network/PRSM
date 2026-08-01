"""
Tensor Parallel Executor
========================

Coordinates parallel model shard execution across nodes
with ring all-reduce synchronization.

Execution modes per shard (decided per assignment):
- LOCAL — Executed in-process via numpy. Used for tests, single-node runs,
  and assignments with node_id == "local" or unset.
- REMOTE — Dispatched via the Ring 2 AgentDispatcher when (a) a dispatcher
  is wired into the executor and (b) the assignment names a non-local
  node_id. Requires a tensor-matmul WASM module to ship with PRSM; until
  that lands, REMOTE assignments raise NotImplementedError instead of
  silently falling back to LOCAL — this prevents the "looks distributed
  but isn't" failure mode.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Any, Awaitable, Callable, Dict, List, Optional

from prsm.compute.model_sharding.models import ShardedModel, PipelineConfig

logger = logging.getLogger(__name__)


#: sp1497 — shard_id suffix marking a REGISTRATION PLACEHOLDER: a shard that exists
#: only so a model appears in the registry, carrying no real weights. Explicit
#: marker rather than sniffing for all-zero bytes, because a genuine tensor may
#: legitimately be all zeros and must still execute.
PLACEHOLDER_SHARD_SUFFIX = "-placeholder"


class MalformedShardError(ValueError):
    """A shard's declared tensor_shape disagrees with its tensor_data length.

    Raised INSTEAD of letting numpy's reshape fail, so callers can tell a bad
    SHARD apart from a bad PROVIDER. Scoring this against the provider defames a
    node that did nothing wrong.
    """


class PlaceholderShardError(MalformedShardError):
    """The shard is a registration placeholder, not real model weights."""


def is_placeholder_shard(shard) -> bool:
    """Is this a registration placeholder rather than real weights?"""
    return str(getattr(shard, "shard_id", "")).endswith(PLACEHOLDER_SHARD_SUFFIX)


# A RemoteShardDispatcher takes (shard, input_data, assignment) and returns
# a dict in the same shape as TensorParallelExecutor._execute_local. It is
# the integration seam between Ring 8 (sharding) and Ring 2 (mobile-agent
# dispatch). The default executor has no dispatcher; tests/integrations
# install one when they need real network execution.
RemoteShardDispatcher = Callable[
    [Any, bytes, Dict[str, Any]],
    Awaitable[Dict[str, Any]],
]


def execute_shard_locally(shard, input_data: bytes) -> np.ndarray:
    """Execute a single tensor-parallel shard locally via numpy matmul.

    Pure function with no side effects. Used by:
      - TensorParallelExecutor._execute_local (the executor's local path)
      - ComputeProvider._on_shard_execute_request (the remote-serve path)

    Both call sites use this helper so local and remote execution
    produce bit-identical output for the same shard + input. The
    integration test in Phase 2 Task 7 asserts this equality.

    Args:
        shard: ModelShard with tensor_data (bytes) and tensor_shape.
        input_data: Raw input bytes (interpreted as float64 array).

    Returns:
        Output numpy array. Shape depends on the matmul compatibility:
        when tensor is 2D and input vector length matches tensor.shape[1],
        returns tensor @ input_array. Otherwise returns a truncated flat
        slice (legacy fallback preserved for existing test contracts).
    """
    # sp1497 — validate BEFORE reshaping. This reshape used to sit outside the try
    # below, so a shard whose declared shape disagreed with its bytes raised a bare
    # ValueError out of here; the provider caught it, returned status="failed", and
    # the requester scored a REPUTATION FAILURE against a provider that did nothing
    # wrong. Every production shard registration was such a shard (see
    # PLACEHOLDER_SHARD_SUFFIX below), so the marketplace dispatched guaranteed
    # failures and blamed honest providers for them.
    if is_placeholder_shard(shard):
        raise PlaceholderShardError(
            f"shard {shard.shard_id!r} is a registration PLACEHOLDER, not real "
            "model weights — refusing to execute it. Computing on it would return "
            "zeros and present them as a genuine inference result. Stage real "
            "weights (ModelSharder.shard_model) before dispatching this model."
        )
    expected = int(np.prod(shard.tensor_shape)) * 8 if shard.tensor_shape else 0
    if len(shard.tensor_data) != expected:
        raise MalformedShardError(
            f"shard {shard.shard_id!r} declares shape {tuple(shard.tensor_shape)} "
            f"({expected} bytes of float64) but carries "
            f"{len(shard.tensor_data)} bytes. The SHARD is malformed — this is not "
            "a provider failure and must not be scored as one."
        )
    tensor = np.frombuffer(shard.tensor_data, dtype=np.float64).reshape(shard.tensor_shape)

    try:
        input_array = np.frombuffer(input_data, dtype=np.float64)
        if input_array.size == 0:
            input_array = np.ones(tensor.shape[-1] if tensor.ndim > 1 else tensor.shape[0])

        if tensor.ndim == 2 and input_array.shape[0] == tensor.shape[1]:
            return tensor @ input_array
        return tensor.flatten()[:min(10, tensor.size)]
    except Exception:
        return tensor.flatten()[:min(10, tensor.size)]


class TensorParallelExecutor:
    """Executes model shards in parallel across assigned nodes."""

    def __init__(
        self,
        confidential_executor=None,
        pipeline_config: Optional[PipelineConfig] = None,
        remote_dispatcher: Optional[RemoteShardDispatcher] = None,
    ):
        """Create a tensor-parallel executor.

        Args:
            confidential_executor: Ring 7 ConfidentialExecutor (currently
                informational; reserved for DP noise wiring).
            pipeline_config: PipelineConfig instance or None for defaults.
            remote_dispatcher: Optional async callable that runs a single
                shard on a remote node. When provided, assignments that
                name a non-local node_id are routed through it. When None
                (the default), only local execution is supported and
                non-local assignments raise NotImplementedError so callers
                immediately see the gap instead of getting a fake "success"
                from a silent local-fallback.
        """
        self._confidential_executor = confidential_executor
        self.config = pipeline_config or PipelineConfig()
        self._remote_dispatcher = remote_dispatcher

    @property
    def supports_remote_execution(self) -> bool:
        return self._remote_dispatcher is not None

    async def execute_parallel(
        self,
        sharded_model: ShardedModel,
        input_data: bytes,
        node_assignments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute all shards in parallel and aggregate results.

        Each shard is executed independently (tensor parallelism).
        Results are combined via all-reduce (sum then average).
        """
        started_at = time.time()
        shard_results = []
        errors = []
        execution_modes: Dict[str, int] = {"local": 0, "remote": 0}

        # Fan out execution per shard
        tasks = []
        for assignment in node_assignments:
            shard_index = assignment.get("shard_index", 0)
            shard = sharded_model.get_shard_by_index(shard_index)
            if shard is None:
                errors.append(f"Shard {shard_index} not found")
                continue
            tasks.append(self._execute_shard(shard, input_data, assignment))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(str(result))
            elif result is not None:
                shard_results.append(result)
                mode = result.get("execution_mode", "local") if isinstance(result, dict) else "local"
                execution_modes[mode] = execution_modes.get(mode, 0) + 1

        # All-reduce aggregation
        aggregated = None
        if shard_results:
            arrays = []
            for r in shard_results:
                if isinstance(r, dict) and "output_array" in r:
                    arrays.append(np.array(r["output_array"]))
                elif isinstance(r, np.ndarray):
                    arrays.append(r)

            if arrays:
                aggregated = self.all_reduce(arrays)

        return {
            "status": "success" if shard_results else "failed",
            "shards_executed": len(shard_results),
            "shards_failed": len(errors),
            "errors": errors,
            "execution_modes": execution_modes,
            "aggregated_output": aggregated.tolist() if aggregated is not None else None,
            "execution_time_seconds": time.time() - started_at,
        }

    async def _execute_shard(
        self,
        shard,
        input_data: bytes,
        assignment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a single shard, routing to local or remote.

        Routing logic:
            * If the assignment has no node_id, or node_id == "local", run
              locally with numpy (tests + single-node).
            * Otherwise, if a remote_dispatcher is wired in, delegate to it.
            * Otherwise raise NotImplementedError so the caller sees the
              missing remote-execution capability instead of a silent fall
              back. This is the documented v1.6.x behavior — Ring 8 supports
              remote dispatch through the Ring 2 seam, but the WASM
              tensor-matmul agent module is still TODO.
        """
        node_id = assignment.get("node_id") or "local"

        if node_id == "local":
            return await self._execute_local(shard, input_data, assignment)

        if self._remote_dispatcher is None:
            raise NotImplementedError(
                f"Shard {shard.shard_index} assigned to remote node "
                f"{node_id!r}, but TensorParallelExecutor was constructed "
                f"without a remote_dispatcher. Wire one in via "
                f"TensorParallelExecutor(remote_dispatcher=...) — see "
                f"prsm.compute.model_sharding.executor docs."
            )

        try:
            result = await self._remote_dispatcher(shard, input_data, assignment)
        except Exception as e:
            logger.warning(
                f"Remote dispatch failed for shard {shard.shard_index} "
                f"on {node_id}: {e}"
            )
            raise

        # Defensive normalization — ensure the dispatcher returned the
        # contract shape and tag execution_mode for the parent aggregator.
        if not isinstance(result, dict) or "output_array" not in result:
            raise ValueError(
                f"remote_dispatcher returned malformed result for shard "
                f"{shard.shard_index}: {type(result).__name__}"
            )
        result.setdefault("shard_index", shard.shard_index)
        result.setdefault("node_id", node_id)
        result["execution_mode"] = "remote"
        return result

    async def _execute_local(
        self,
        shard,
        input_data: bytes,
        assignment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a shard's forward pass in-process via numpy.

        Thin wrapper around the module-level execute_shard_locally()
        helper — see that function's docstring for the numerics contract.
        """
        output = execute_shard_locally(shard, input_data)
        return {
            "shard_index": shard.shard_index,
            "node_id": assignment.get("node_id", "local"),
            "output_array": output.tolist(),
            "execution_mode": "local",
        }

    @staticmethod
    def all_reduce(shard_outputs: List[np.ndarray]) -> np.ndarray:
        """Ring all-reduce: average across all shard outputs.

        In tensor parallelism, each shard produces a partial result.
        All-reduce combines them (typically sum for column-parallel,
        concatenate for row-parallel). We use average as a general default.
        """
        if not shard_outputs:
            return np.array([])

        # Pad to same length if needed
        max_len = max(a.size for a in shard_outputs)
        padded = []
        for a in shard_outputs:
            flat = a.flatten()
            if flat.size < max_len:
                flat = np.pad(flat, (0, max_len - flat.size))
            padded.append(flat)

        stacked = np.stack(padded)
        return np.mean(stacked, axis=0)
