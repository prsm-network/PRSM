"""Sprint 1501 — the bridge: marketplace-selected providers -> a real inference chain.

WHY THIS EXISTS
---------------
PRSM has had two disjoint execution paths:

  * REAL inference — ``RpcChainExecutor`` (prsm/compute/chain_rpc/client.py) walks
    a ``GPUChain`` stage by stage, minting a handoff token per stage, verifying
    each stage's signature, and threading activations through. Proven cross-host
    (Qwen-1.5B, 14+14 layers on two A10s).
  * The MARKETPLACE — provider discovery, price quotes, escrow, reputation —
    which dispatched ``ModelShard``s to ``execute_shard_locally``, a numpy matmul
    on manifest bytes. That primitive is not inference, and a prompt never became
    a ModelShard anywhere in the tree.

So the marketplace was selecting and paying providers for the wrong kind of work.
The fix is not to give the matmul path real weights (the manifest sentinel is
deliberate — the HF runner loads weights from its own cache and ignores
``tensor_data``). It is to let the marketplace choose WHO runs a chain, and let
the proven chain executor do the running.

This module is that join, and only that join: listings in, a validated
``GPUChain`` out. It performs no I/O, so the part most likely to be subtly wrong
— the layer tiling — is fully testable without a network or a GPU.

THE INVARIANT THAT MATTERS
--------------------------
``layer_ranges`` must tile ``[0, num_layers)`` exactly: contiguous, no gap, no
overlap, first starting at 0 and last ending at num_layers. The router enforces
this on its own output because a chain that does not tile the model computes a
WRONG ANSWER rather than failing — some layers are skipped or applied twice, and
the result still decodes to plausible text. There is no downstream check that
catches it, so it must be impossible to construct here.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

from prsm.compute.parallax_scheduling.prsm_request_router import GPUChain

logger = logging.getLogger(__name__)


class ChainConstructionError(RuntimeError):
    """The selected providers cannot form a valid chain for this model.

    Raised rather than returning a partial or best-effort chain: a chain that
    does not tile the model produces a confidently wrong answer, and the
    requester has paid for it.
    """


def allocate_layer_ranges(num_layers: int, n_stages: int) -> List[Tuple[int, int]]:
    """Split ``num_layers`` across ``n_stages`` as evenly as possible.

    Returns half-open ``(start, end)`` pairs tiling ``[0, num_layers)``. Earlier
    stages take the extra layer when the split is uneven, which keeps the result
    deterministic — two nodes building the same chain must agree exactly, or the
    handoff tokens they mint will not line up.
    """
    if n_stages <= 0:
        raise ChainConstructionError(f"n_stages must be positive, got {n_stages}")
    if num_layers <= 0:
        raise ChainConstructionError(f"num_layers must be positive, got {num_layers}")
    if n_stages > num_layers:
        raise ChainConstructionError(
            f"cannot split {num_layers} layers across {n_stages} stages — a stage "
            "with zero layers would contribute nothing while still being paid"
        )
    base, extra = divmod(num_layers, n_stages)
    ranges: List[Tuple[int, int]] = []
    cursor = 0
    for i in range(n_stages):
        size = base + (1 if i < extra else 0)
        ranges.append((cursor, cursor + size))
        cursor += size
    assert cursor == num_layers, "allocation must consume every layer"
    return ranges


def verify_tiling(layer_ranges: Sequence[Tuple[int, int]], num_layers: int) -> None:
    """Raise unless ``layer_ranges`` exactly tiles ``[0, num_layers)``.

    Deliberately separate from the allocator so it can also be applied to ranges
    that came from somewhere else — an operator override, a resumed chain, a
    future capacity-weighted allocator. The allocator being correct today is not
    a reason to trust its callers tomorrow.
    """
    if not layer_ranges:
        raise ChainConstructionError("empty chain covers no layers")
    if layer_ranges[0][0] != 0:
        raise ChainConstructionError(
            f"chain starts at layer {layer_ranges[0][0]}, not 0 — the prompt would "
            "enter mid-model")
    if layer_ranges[-1][1] != num_layers:
        raise ChainConstructionError(
            f"chain ends at layer {layer_ranges[-1][1]}, not {num_layers} — the "
            "final layers would never run and the output would still decode")
    for i, (start, end) in enumerate(layer_ranges):
        if end <= start:
            raise ChainConstructionError(
                f"stage {i} covers [{start}, {end}) — empty or inverted")
    for i, (prev, nxt) in enumerate(zip(layer_ranges, layer_ranges[1:])):
        if prev[1] != nxt[0]:
            kind = "gap" if nxt[0] > prev[1] else "overlap"
            raise ChainConstructionError(
                f"{kind} between stage {i} (ends {prev[1]}) and stage {i+1} "
                f"(starts {nxt[0]}) — layers would be skipped or applied twice, "
                "and the result would still decode to plausible text")


def build_marketplace_chain(
    *,
    listings: Sequence,
    num_layers: int,
    request_id: str,
    region: str = "default",
    max_stages: Optional[int] = None,
) -> GPUChain:
    """Turn marketplace-selected providers into a chain the real executor can run.

    ``listings`` must already be filtered and ORDERED by the caller — the
    marketplace's eligibility filter plus sp1498's cheapest-first sort. This
    function does not re-price and does not re-filter; mixing selection policy
    into chain construction is how the two would drift apart.

    One provider appears at most once. A repeated provider would hold two stages
    of the same chain, which defeats the point of distributing trust and lets a
    single party observe activations entering and leaving its own slice.
    """
    if not listings:
        raise ChainConstructionError(
            f"no providers supplied for request {request_id!r} — cannot build a chain")

    seen = set()
    unique = []
    for l in listings:
        pid = str(getattr(l, "provider_id", "") or "").strip()
        if not pid:
            raise ChainConstructionError("listing with no provider_id")
        if pid in seen:
            continue
        seen.add(pid)
        unique.append(l)

    n = len(unique)
    if max_stages is not None:
        n = min(n, int(max_stages))
    # Never more stages than layers — a zero-layer stage is a provider paid for
    # nothing, and it also breaks the tiling invariant.
    n = min(n, int(num_layers))
    if n <= 0:
        raise ChainConstructionError(
            f"request {request_id!r}: no usable stages for a {num_layers}-layer model")

    chosen = unique[:n]
    ranges = allocate_layer_ranges(num_layers, n)
    verify_tiling(ranges, num_layers)      # belt and braces: never emit a bad tiling

    chain = GPUChain(
        request_id=request_id,
        region=region,
        stages=tuple(str(l.provider_id) for l in chosen),
        layer_ranges=tuple(ranges),
        total_latency_ms=0.0,      # marketplace selection is price-led, not latency-led
        stale_profile_count=0,
    )
    logger.info(
        "marketplace chain for %s: %d stage(s) over %d layers -> %s",
        request_id, n, num_layers,
        ", ".join(f"{l.provider_id[:8]}…[{s}:{e})"
                  for l, (s, e) in zip(chosen, ranges)))
    return chain
