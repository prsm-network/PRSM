"""Sprint 1100 — fold the Domain-04 review findings on the LIVE WASM untrusted-code
path (compute_provider._run_wasm → wasm/runtime.py). This is the only live
untrusted-code surface in the compute domain; everything else is scaffold/local-trust.

- HIGH #1 (resource exhaustion / accounting honesty): the requester chose its own
  WASM resource limits with NO operator ceiling (`payload.get(k, default)` supplied a
  default but never a cap) → a single authenticated peer could demand 64 GB / 1 hour
  and OOM/CPU-starve the host. Now clamped to operator-configured maxima.
- HIGH #2 (no wall-clock bound): the runtime applied only `consume_fuel` (an
  instruction count, not time) with no epoch deadline → no real timeout; `max_execution
  _seconds` was a misnomer. Now a wall-clock epoch deadline kills a runaway guest.
- MEDIUM (untrusted input → compile without validation): `_run_wasm` fed raw bytes to
  `runtime.load` skipping the `WASMModule` magic-byte + size check → a 500 MB blob is a
  compile-bomb before any limit binds. Now validated before compile.
- Latent classification bug surfaced while fixing #2: BOTH fuel exhaustion AND epoch
  interrupt raise `wasmtime.Trap` (NOT `WasmtimeError`), so the old `except
  WasmtimeError` fuel->TIMEOUT branch was DEAD — a timed-out job mis-reported as a
  generic ERROR. Now Trap is caught and classified.
"""
from __future__ import annotations

import time

import pytest

from prsm.compute.wasm.models import ExecutionStatus, ResourceLimits
from prsm.compute.wasm.runtime import WasmtimeRuntime

_HAS_WASMTIME = WasmtimeRuntime().available


def _wat(src: str) -> bytes:
    import wasmtime
    return wasmtime.wat2wasm(src)


# infinite loop with an i32 result — never returns; only a fuel/epoch trap stops it
_INFINITE_LOOP = '(module (func (export "run") (result i32) (loop $l (br $l)) (i32.const 0)))'
# minimal valid module (magic + version + a "run" returning 42)
_MINIMAL = bytes([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x00,
    0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2a, 0x0b,
])


# ── HIGH #1: requester-supplied limits clamped to operator ceilings ──────────────

def test_requester_limits_clamped_to_operator_max(monkeypatch):
    from prsm.node.compute_provider import clamp_wasm_resource_limits
    monkeypatch.setenv("PRSM_WASM_MAX_MEMORY_BYTES", str(512 * 1024 * 1024))
    monkeypatch.setenv("PRSM_WASM_MAX_EXECUTION_SECONDS", "60")
    monkeypatch.setenv("PRSM_WASM_MAX_OUTPUT_BYTES", str(10 * 1024 * 1024))
    # hostile requester demands 64 GB / 1 hour / 1 GB output
    payload = {"max_memory_bytes": 64 * 1024 ** 3,
               "max_execution_seconds": 3600,
               "max_output_bytes": 1024 ** 3}
    limits = clamp_wasm_resource_limits(payload)
    assert limits.max_memory_bytes == 512 * 1024 * 1024
    assert limits.max_execution_seconds == 60
    assert limits.max_output_bytes == 10 * 1024 * 1024


def test_requester_below_max_passes_through(monkeypatch):
    from prsm.node.compute_provider import clamp_wasm_resource_limits
    monkeypatch.setenv("PRSM_WASM_MAX_MEMORY_BYTES", str(512 * 1024 * 1024))
    monkeypatch.setenv("PRSM_WASM_MAX_EXECUTION_SECONDS", "60")
    payload = {"max_memory_bytes": 64 * 1024 * 1024, "max_execution_seconds": 5}
    limits = clamp_wasm_resource_limits(payload)
    assert limits.max_memory_bytes == 64 * 1024 * 1024
    assert limits.max_execution_seconds == 5


def test_missing_limits_use_safe_positive_defaults():
    from prsm.node.compute_provider import clamp_wasm_resource_limits
    limits = clamp_wasm_resource_limits({})
    assert limits.max_memory_bytes > 0
    assert limits.max_execution_seconds > 0
    assert limits.max_output_bytes > 0


def test_zero_or_negative_request_does_not_bypass_floor(monkeypatch):
    """A 0/negative request must not produce a non-positive limit (which would either
    crash ResourceLimits.__post_init__ or, worse, disable a cap)."""
    from prsm.node.compute_provider import clamp_wasm_resource_limits
    limits = clamp_wasm_resource_limits(
        {"max_memory_bytes": 0, "max_execution_seconds": -5})
    assert limits.max_memory_bytes > 0
    assert limits.max_execution_seconds > 0


# ── MEDIUM: module bytes validated (magic + size) before compile ─────────────────

def test_oversized_module_rejected_before_compile(monkeypatch):
    from prsm.node.compute_provider import validate_wasm_module_bytes
    monkeypatch.setenv("PRSM_WASM_MAX_MODULE_BYTES", "1024")
    with pytest.raises(ValueError):
        validate_wasm_module_bytes(b"\x00asm\x01\x00\x00\x00" + b"\x00" * 4096)


def test_non_wasm_magic_rejected():
    from prsm.node.compute_provider import validate_wasm_module_bytes
    with pytest.raises(ValueError):
        validate_wasm_module_bytes(b"this is definitely not a wasm module")


def test_valid_module_bytes_accepted():
    from prsm.node.compute_provider import validate_wasm_module_bytes
    validate_wasm_module_bytes(_MINIMAL)  # must not raise


# ── HIGH #2: real wall-clock deadline kills a runaway guest (epoch) ──────────────

@pytest.mark.skipif(not _HAS_WASMTIME, reason="wasmtime not installed")
def test_infinite_loop_killed_by_wall_clock_deadline():
    runtime = WasmtimeRuntime()
    module = runtime.load(_wat(_INFINITE_LOOP))
    t0 = time.monotonic()
    result = runtime.execute(
        module, b"",
        ResourceLimits(max_execution_seconds=1, max_memory_bytes=64 * 1024 * 1024),
    )
    elapsed = time.monotonic() - t0
    # killed near the 1s wall bound (NOT hung, NOT only-fuel which would let
    # generous-fuel work run far past the named seconds)
    assert result.status == ExecutionStatus.TIMEOUT
    assert elapsed < 6, f"guest ran {elapsed:.1f}s — wall-clock deadline did not fire"


@pytest.mark.skipif(not _HAS_WASMTIME, reason="wasmtime not installed")
def test_normal_module_still_succeeds():
    runtime = WasmtimeRuntime()
    module = runtime.load(_MINIMAL)
    result = runtime.execute(module, b"", ResourceLimits())
    assert result.status == ExecutionStatus.SUCCESS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
