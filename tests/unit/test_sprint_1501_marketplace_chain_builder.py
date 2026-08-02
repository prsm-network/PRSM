"""Sprint 1501 — the bridge from marketplace selection to real inference.

The marketplace had been selecting and paying providers to run
`execute_shard_locally` — a numpy matmul on manifest bytes, which is not
inference. Meanwhile `RpcChainExecutor` already runs REAL cross-host inference
over a `GPUChain` (proven: Qwen-1.5B, 14+14 layers on two A10s). The two were
never connected.

This module is the join: marketplace-ordered listings in, a validated `GPUChain`
out, no I/O.

THE INVARIANT THESE TESTS EXIST FOR: `layer_ranges` must tile `[0, num_layers)`
exactly. A chain that does not tile the model does not fail — it SKIPS or REPEATS
layers and still decodes to plausible text. Nothing downstream catches that, and
the requester has already paid. So a bad tiling must be unconstructable.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from prsm.marketplace.chain_builder import (
    ChainConstructionError,
    allocate_layer_ranges,
    build_marketplace_chain,
    verify_tiling,
)


@dataclass
class _L:
    provider_id: str


def _listings(n):
    return [_L(f"provider-{i:02d}" + "a" * 20) for i in range(n)]


# ── the tiling invariant ────────────────────────────────────────────

@pytest.mark.parametrize("num_layers,n_stages", [
    (28, 2), (28, 3), (28, 4), (32, 5), (1, 1), (7, 7), (100, 3), (13, 4),
])
def test_allocation_always_tiles_exactly(num_layers, n_stages):
    """★ THE property, across uneven splits: contiguous, no gap, no overlap,
    covering every layer once."""
    ranges = allocate_layer_ranges(num_layers, n_stages)
    assert len(ranges) == n_stages
    verify_tiling(ranges, num_layers)
    covered = [i for s, e in ranges for i in range(s, e)]
    assert covered == list(range(num_layers)), "every layer exactly once, in order"


def test_uneven_splits_put_the_extra_layers_FIRST_deterministically():
    """Two nodes building the same chain must agree exactly, or their handoff
    tokens will not line up."""
    assert allocate_layer_ranges(28, 3) == [(0, 10), (10, 19), (19, 28)]
    assert allocate_layer_ranges(28, 3) == allocate_layer_ranges(28, 3)


def test_more_stages_than_layers_is_refused():
    """A zero-layer stage is a provider paid for contributing nothing — and it
    breaks the tiling."""
    with pytest.raises(ChainConstructionError, match="zero layers"):
        allocate_layer_ranges(3, 5)


@pytest.mark.parametrize("bad_layers,bad_stages", [(0, 2), (-1, 2), (10, 0), (10, -3)])
def test_degenerate_inputs_are_refused(bad_layers, bad_stages):
    with pytest.raises(ChainConstructionError):
        allocate_layer_ranges(bad_layers, bad_stages)


# ── verify_tiling catches what an allocator might get wrong ─────────

def test_a_GAP_is_caught():
    """Skipped layers still decode to plausible text — nothing downstream notices."""
    with pytest.raises(ChainConstructionError, match="gap"):
        verify_tiling([(0, 10), (12, 28)], 28)


def test_an_OVERLAP_is_caught():
    """Layers applied twice — also silently wrong."""
    with pytest.raises(ChainConstructionError, match="overlap"):
        verify_tiling([(0, 14), (10, 28)], 28)


def test_not_starting_at_zero_is_caught():
    with pytest.raises(ChainConstructionError, match="not 0"):
        verify_tiling([(2, 28)], 28)


def test_not_ending_at_num_layers_is_caught():
    with pytest.raises(ChainConstructionError, match="final layers would never run"):
        verify_tiling([(0, 20)], 28)


def test_an_empty_or_inverted_stage_is_caught():
    with pytest.raises(ChainConstructionError, match="empty or inverted"):
        verify_tiling([(0, 10), (10, 10), (10, 28)], 28)
    with pytest.raises(ChainConstructionError, match="empty or inverted"):
        verify_tiling([(0, 14), (14, 12), (12, 28)], 28)


def test_an_empty_chain_is_caught():
    with pytest.raises(ChainConstructionError, match="covers no layers"):
        verify_tiling([], 28)


# ── building the chain ──────────────────────────────────────────────

def test_a_chain_preserves_the_callers_ORDER():
    """★ Selection policy belongs to the marketplace (sp1498 cheapest-first). This
    must not re-sort, or the two drift apart and price-ordering silently stops
    applying."""
    ls = _listings(3)
    chain = build_marketplace_chain(
        listings=ls, num_layers=28, request_id="req-1")
    assert list(chain.stages) == [l.provider_id for l in ls]
    verify_tiling(list(chain.layer_ranges), 28)


def test_a_provider_appears_at_most_ONCE():
    """★ A repeated provider holds two stages of one chain — it observes the
    activations entering and leaving its own slice, defeating the point of
    distributing the work."""
    dup = _L("same" + "a" * 20)
    chain = build_marketplace_chain(
        listings=[dup, dup, _L("other" + "b" * 20)], num_layers=28,
        request_id="req-1")
    assert len(set(chain.stages)) == len(chain.stages)
    assert len(chain.stages) == 2


def test_stages_are_capped_at_the_layer_count():
    """10 providers cannot each own a slice of a 3-layer model."""
    chain = build_marketplace_chain(
        listings=_listings(10), num_layers=3, request_id="req-1")
    assert len(chain.stages) == 3
    verify_tiling(list(chain.layer_ranges), 3)


def test_max_stages_is_honoured():
    chain = build_marketplace_chain(
        listings=_listings(8), num_layers=28, request_id="req-1", max_stages=2)
    assert len(chain.stages) == 2
    verify_tiling(list(chain.layer_ranges), 28)


def test_no_providers_is_refused():
    with pytest.raises(ChainConstructionError, match="no providers"):
        build_marketplace_chain(listings=[], num_layers=28, request_id="req-1")


def test_a_listing_without_a_provider_id_is_refused():
    with pytest.raises(ChainConstructionError, match="no provider_id"):
        build_marketplace_chain(
            listings=[_L("")], num_layers=28, request_id="req-1")


def test_the_chain_is_shaped_for_the_real_executor():
    """★ It must be a genuine GPUChain — the type RpcChainExecutor consumes."""
    from prsm.compute.parallax_scheduling.prsm_request_router import GPUChain

    chain = build_marketplace_chain(
        listings=_listings(2), num_layers=28, request_id="req-42", region="sfo")
    assert isinstance(chain, GPUChain)
    assert chain.request_id == "req-42"
    assert chain.region == "sfo"
    assert isinstance(chain.stages, tuple)
    assert isinstance(chain.layer_ranges, tuple)


def test_single_provider_takes_the_whole_model():
    chain = build_marketplace_chain(
        listings=_listings(1), num_layers=28, request_id="req-1")
    assert chain.stages == ("provider-00" + "a" * 20,)
    assert list(chain.layer_ranges) == [(0, 28)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
