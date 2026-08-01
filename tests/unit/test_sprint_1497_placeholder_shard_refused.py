"""Sprint 1497 — a placeholder shard must be REFUSED, and a malformed shard must
not be blamed on the provider.

Found by an adversarial assessment of the marketplace demand side, and reproduced
before fixing. Every production model registration built:

    tensor_data = b"\\x00" * 32     # 32 bytes == 4 float64 values
    tensor_shape = (32,)            # ...but declares 32

execute_shard_locally reshaped OUTSIDE its try block, so this raised:

    ValueError: cannot reshape array of size 4 into shape (32,)

The provider caught it, returned status="failed", and the requester scored a
REPUTATION FAILURE against a provider that did nothing wrong. So every marketplace
dispatch of a default node's registered model was a guaranteed failure that
defamed an honest counterparty.

Fixing only the shape would be worse: the shard is all zeros, so it would then
compute silently and return zeros as though they were a real inference result.
Placeholders are therefore marked explicitly and refused.
"""
from __future__ import annotations

import numpy as np
import pytest

from prsm.compute.model_sharding.executor import (
    PLACEHOLDER_SHARD_SUFFIX,
    MalformedShardError,
    PlaceholderShardError,
    execute_shard_locally,
    is_placeholder_shard,
)
from prsm.compute.model_sharding.models import ModelShard


def _shard(shard_id="m-shard-0", data=None, shape=(4,)):
    data = b"\x00" * 32 if data is None else data
    return ModelShard(shard_id=shard_id, model_id="m", shard_index=0, total_shards=1,
                      tensor_data=data, tensor_shape=shape,
                      layer_range=(0, 1), size_bytes=len(data))


# ── the crash that defamed providers ────────────────────────────────

def test_the_old_sentinel_shape_is_now_a_TYPED_error_not_a_raw_ValueError():
    """★ Was `ValueError: cannot reshape array of size 4 into shape (32,)` escaping
    from numpy. Now a typed MalformedShardError that says whose fault it is."""
    with pytest.raises(MalformedShardError, match="not\n?\\s*a provider failure|not a provider failure"):
        execute_shard_locally(_shard(shape=(32,)), b"")


def test_the_error_names_the_actual_mismatch():
    with pytest.raises(MalformedShardError, match=r"declares shape \(32,\).*carries 32 bytes"):
        execute_shard_locally(_shard(shape=(32,)), b"")


def test_a_malformed_shard_is_distinguishable_from_a_provider_failure():
    """★ The whole point: callers must be able to tell a bad SHARD from a bad
    PROVIDER, so a malformed request is never scored as misbehaviour."""
    assert issubclass(MalformedShardError, ValueError)
    assert issubclass(PlaceholderShardError, MalformedShardError)


# ── placeholders are refused, not silently computed ─────────────────

def test_a_placeholder_shard_is_REFUSED(caplog):
    """★★ Fixing only the shape would make this compute zeros and present them as
    a real inference result — worse than crashing."""
    s = _shard(shard_id="m-shard-0" + PLACEHOLDER_SHARD_SUFFIX)
    with pytest.raises(PlaceholderShardError, match="PLACEHOLDER"):
        execute_shard_locally(s, b"")


def test_the_refusal_explains_what_to_do():
    s = _shard(shard_id="m-shard-0" + PLACEHOLDER_SHARD_SUFFIX)
    with pytest.raises(PlaceholderShardError, match="Stage real"):
        execute_shard_locally(s, b"")


def test_placeholder_detection_is_by_MARKER_not_by_all_zero_bytes():
    """★ A genuine tensor may legitimately be all zeros and must still execute —
    so detection must not sniff for zero bytes."""
    real_zeros = _shard(shard_id="m-shard-0", data=b"\x00" * 32, shape=(4,))
    assert not is_placeholder_shard(real_zeros)
    out = execute_shard_locally(real_zeros, b"")
    assert isinstance(out, np.ndarray)


def test_marker_detection():
    assert is_placeholder_shard(_shard(shard_id="x" + PLACEHOLDER_SHARD_SUFFIX))
    assert not is_placeholder_shard(_shard(shard_id="x"))


# ── a real shard still works ────────────────────────────────────────

def test_a_wellformed_real_shard_executes():
    data = np.arange(4, dtype=np.float64).tobytes()
    out = execute_shard_locally(_shard(data=data, shape=(4,)), b"")
    assert isinstance(out, np.ndarray)


def test_a_wellformed_2d_shard_matmuls():
    tensor = np.arange(6, dtype=np.float64).reshape(2, 3)
    s = _shard(data=tensor.tobytes(), shape=(2, 3))
    out = execute_shard_locally(s, np.ones(3, dtype=np.float64).tobytes())
    assert np.allclose(out, tensor @ np.ones(3))


# ── the production registration sites are fixed ─────────────────────

@pytest.mark.parametrize("path", [
    "prsm/node/inference_wiring.py",
    "prsm/node/chain_executor_adapters.py",
    "scripts/stage_hf_model.py",
])
def test_no_production_site_still_registers_the_broken_sentinel(path):
    """★ Binding test. All three built b'\\x00'*32 with tensor_shape=(32,) — a
    shard guaranteed to raise on every dispatch."""
    from pathlib import Path
    src = Path(path).read_text()
    assert "tensor_shape=(32,)" not in src, f"{path} still registers the broken sentinel"
    if 'tensor_data=b"\\x00" * 32' in src:
        assert "PLACEHOLDER_SHARD_SUFFIX" in src, (
            f"{path} registers a zero-weight shard without marking it a placeholder")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
