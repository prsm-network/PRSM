"""
TEE Runtime Interface
=====================

Abstract interface for trusted execution environments.
SoftwareTEERuntime wraps the Ring 1 WASM sandbox as a fallback.
"""

import abc
import logging
from typing import Any

from prsm.compute.tee.models import TEEType
from prsm.compute.wasm.models import ExecutionResult, ResourceLimits

logger = logging.getLogger(__name__)


class TEERuntime(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Runtime name."""

    @property
    @abc.abstractmethod
    def tee_type(self) -> TEEType:
        """Type of TEE."""

    @property
    @abc.abstractmethod
    def available(self) -> bool:
        """Whether available on current hardware."""

    @abc.abstractmethod
    def load(self, wasm_bytes: bytes) -> Any:
        """Load a WASM module."""

    @abc.abstractmethod
    def execute(self, module: Any, input_data: bytes, resource_limits: ResourceLimits) -> ExecutionResult:
        """Execute in TEE sandbox."""

    @abc.abstractmethod
    def get_attestation_bytes(self) -> bytes:
        """Sprint 1239 (TEE Tier-3 roadmap D) — generate this runtime's
        attestation quote bytes for the current enclave/measurement. The
        quote-GENERATION half of the attestation contract (verification is
        attestation_backends / the sp1236 multi-stage chain). chain_rpc/server.py
        stamps the returned bytes onto each stage's §7 proof; verifiers route
        them through the AttestationBackendRegistry.

        ABSTRACT on purpose: a TEE runtime MUST declare how it attests — a
        hardware runtime that forgot to implement this would otherwise silently
        emit a software/dev-only blob, masking the bug. ``SoftwareTEERuntime``
        returns a DEV-ONLY-marked blob (NO confidentiality — verifiers MUST
        reject it as non-production); real ``SgxTEERuntime``/``SevTEERuntime``
        return a hardware quote (sprint E, needs hardware)."""


class SoftwareTEERuntime(TEERuntime):
    """Software-only TEE using Ring 1 WASM sandbox as fallback."""

    def __init__(self):
        self._wasm_runtime = None

    def _get_wasm_runtime(self):
        if self._wasm_runtime is None:
            from prsm.compute.wasm.runtime import WasmtimeRuntime
            self._wasm_runtime = WasmtimeRuntime()
        return self._wasm_runtime

    @property
    def name(self) -> str:
        return "software"

    @property
    def tee_type(self) -> TEEType:
        return TEEType.SOFTWARE

    @property
    def available(self) -> bool:
        return self._get_wasm_runtime().available

    def load(self, wasm_bytes: bytes) -> Any:
        return self._get_wasm_runtime().load(wasm_bytes)

    def execute(self, module: Any, input_data: bytes, resource_limits: ResourceLimits) -> ExecutionResult:
        return self._get_wasm_runtime().execute(module, input_data, resource_limits)

    def get_attestation_bytes(self) -> bytes:
        """A DEV-ONLY-marked blob (16-byte ``DEV-ONLY-SW-TEE:`` prefix + a
        48-byte sha384 tag = 64 bytes), matching the executor's software-stub
        format. Verifiers (DevOnlyBackend / is_dev_only) treat it as having NO
        confidentiality guarantee — it brands itself non-production so a
        software TEE can never be mistaken for a hardware one. Lazy imports keep
        this low-level runtime module decoupled from the heavier executor."""
        import hashlib
        from prsm.compute.inference.executor import (
            SOFTWARE_TEE_ATTESTATION_PREFIX,
        )
        return SOFTWARE_TEE_ATTESTATION_PREFIX + hashlib.sha384(b"sw-tee").digest()
