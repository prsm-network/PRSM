"""
WASM Runtime Interface
======================

Abstract runtime interface and Wasmtime concrete implementation
for sandboxed WebAssembly execution of untrusted mobile agents.
"""

import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Tuple

from prsm.compute.wasm.models import ExecutionResult, ExecutionStatus, ResourceLimits

# sp1100 (Domain-04 review HIGH #2) — fuel bounds INSTRUCTION COUNT, not wall-clock
# time, so ``max_execution_seconds`` was a misnomer with no real timeout. We now make
# it a real WALL-CLOCK bound via an epoch deadline, and keep fuel as a generous
# SECONDARY instruction cap. This per-second figure is deliberately high (well above
# any realistic CPU's fuel/sec, ~10-50e9 measured) so that for time-bound work the
# wall-clock epoch is the binding limit; fuel only catches pathological throughput.
_FUEL_PER_SECOND_CEILING = 200_000_000_000


class WASMRuntime(ABC):
    """Abstract interface for a WASM execution runtime."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable runtime name."""
        ...

    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether the runtime backend is installed and usable."""
        ...

    @abstractmethod
    def load(self, wasm_bytes: bytes) -> Any:
        """Compile WASM bytes into a runnable module.

        Returns a runtime-specific handle that can be passed to ``execute``.
        """
        ...

    @abstractmethod
    def execute(
        self,
        module: Any,
        input_data: bytes,
        resource_limits: ResourceLimits,
    ) -> ExecutionResult:
        """Execute a compiled module inside a sandbox.

        Parameters
        ----------
        module:
            The handle returned by :meth:`load`.
        input_data:
            Opaque bytes made available to the module via WASI stdin.
        resource_limits:
            Memory, time, and output caps enforced during execution.

        Returns
        -------
        ExecutionResult
        """
        ...


class WasmtimeRuntime(WASMRuntime):
    """Concrete runtime backed by the *wasmtime* Python package."""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "wasmtime"

    @property
    def available(self) -> bool:
        try:
            import wasmtime as _wt  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, wasm_bytes: bytes) -> Tuple[Any, Any]:
        """Compile *wasm_bytes* into a ``wasmtime.Module``.

        Returns ``(engine, module)`` so that :meth:`execute` has access to
        the engine that produced the module.
        """
        import wasmtime

        try:
            config = wasmtime.Config()
            config.consume_fuel = True
            # sp1100 (HIGH #2) — enable epoch interruption so execute() can impose a
            # real wall-clock deadline on the guest (fuel alone is not a time bound).
            config.epoch_interruption = True
            engine = wasmtime.Engine(config)
            module = wasmtime.Module(engine, wasm_bytes)
            return (engine, module)
        except Exception as exc:
            raise ValueError(f"Failed to compile WASM module: {exc}") from exc

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(
        self,
        module: Any,
        input_data: bytes,
        resource_limits: ResourceLimits,
    ) -> ExecutionResult:
        import wasmtime

        engine, wasm_module = module

        # ---- Store with fuel + memory limits ----
        store = wasmtime.Store(engine)
        # sp1100 (HIGH #2) — fuel is a generous SECONDARY instruction cap; the real
        # time bound is the wall-clock epoch deadline below.
        fuel_budget = (
            resource_limits.max_execution_seconds * _FUEL_PER_SECOND_CEILING
        )
        store.set_fuel(fuel_budget)
        store.set_limits(memory_size=resource_limits.max_memory_bytes)

        # ---- Wall-clock kill switch (epoch) ----
        # A guest that does fuel-undercounted work (e.g. memory.grow churn) could run
        # far past max_execution_seconds with fuel alone. Arm a one-shot timer that
        # advances the engine epoch past the store's deadline → the guest traps.
        store.set_epoch_deadline(1)
        _deadline_timer = threading.Timer(
            max(0.001, float(resource_limits.max_execution_seconds)),
            engine.increment_epoch,
        )
        _deadline_timer.daemon = True
        _deadline_timer.start()

        # ---- Determine whether the module needs WASI ----
        export_names = [e.name for e in wasm_module.exports]
        uses_wasi = "_start" in export_names
        bare_run = "run" in export_names and not uses_wasi

        t0 = time.monotonic()

        try:
            if uses_wasi:
                # WASI command module — wire up stdin/stdout
                wasi_config = wasmtime.WasiConfig()
                wasi_config.stdin_bytes = input_data
                store.set_wasi(wasi_config)

                linker = wasmtime.Linker(engine)
                linker.define_wasi()
                instance = linker.instantiate(store, wasm_module)

                exports = instance.exports(store)
                start_fn = exports["_start"]
                start_fn(store)
                output = b""
            elif bare_run:
                # Bare module with a "run" export — no WASI needed
                instance = wasmtime.Instance(store, wasm_module, [])
                exports = instance.exports(store)
                run_fn = exports["run"]
                raw = run_fn(store)
                output = str(raw).encode() if raw is not None else b""
            else:
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    output=b"",
                    execution_time_seconds=0.0,
                    memory_used_bytes=0,
                    error="Module has no '_start' or 'run' export",
                )

            elapsed = time.monotonic() - t0
            remaining_fuel = store.get_fuel()
            fuel_used = fuel_budget - remaining_fuel

            # sp1100 — enforce the output cap that was previously stored-but-never-
            # checked: a guest must not return more than max_output_bytes.
            if len(output) > resource_limits.max_output_bytes:
                output = output[: resource_limits.max_output_bytes]

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                output=output,
                execution_time_seconds=elapsed,
                memory_used_bytes=fuel_used // 1000,  # rough proxy
            )

        # sp1100 — BOTH fuel exhaustion AND the epoch (wall-clock) deadline raise
        # wasmtime.Trap, which is NOT a subclass of WasmtimeError, so the prior
        # `except WasmtimeError` fuel->TIMEOUT branch was DEAD code (timeouts
        # mis-reported as generic ERROR). Catch both and classify on the message.
        except (wasmtime.WasmtimeError, wasmtime.Trap) as exc:
            elapsed = time.monotonic() - t0
            err_msg = str(exc).lower()

            if (
                "fuel" in err_msg
                or "interrupt" in err_msg
                or "epoch" in err_msg
                or "deadline" in err_msg
            ):
                status = ExecutionStatus.TIMEOUT
            elif "memory" in err_msg or "grow" in err_msg:
                status = ExecutionStatus.OOM
            else:
                status = ExecutionStatus.ERROR

            return ExecutionResult(
                status=status,
                output=b"",
                execution_time_seconds=elapsed,
                memory_used_bytes=0,
                error=str(exc),
            )

        except Exception as exc:
            elapsed = time.monotonic() - t0
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                output=b"",
                execution_time_seconds=elapsed,
                memory_used_bytes=0,
                error=str(exc),
            )

        finally:
            # Always stop the wall-clock timer so it can't fire on a recycled engine
            # nor leak a thread for up to max_execution_seconds after a fast return.
            _deadline_timer.cancel()
