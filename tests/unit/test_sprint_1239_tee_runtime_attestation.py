"""Sprint 1239 (TEE Tier-3 roadmap D) — get_attestation_bytes() generation contract.

chain_rpc/server.py already called self._tee_runtime.get_attestation_bytes() to
stamp each §7 stage proof, but the TEERuntime ABC never declared it (production
runtimes duck-typed it). This sprint makes it an ABSTRACT method — so every TEE
runtime MUST declare how it attests (a hardware runtime that forgot to implement
it would otherwise silently emit a dev-only blob) — and implements it on
SoftwareTEERuntime as a DEV-ONLY-marked blob. Real Sgx/Sev runtimes (sprint E,
needs hardware) return a real quote.
"""

import hashlib

import pytest

from prsm.compute.inference.executor import SOFTWARE_TEE_ATTESTATION_PREFIX
from prsm.compute.tee.models import TEEType
from prsm.compute.tee.runtime import SoftwareTEERuntime, TEERuntime
from prsm.compute.wasm.models import ExecutionResult, ResourceLimits  # noqa: F401


def test_get_attestation_bytes_is_abstract():
    """A TEERuntime subclass that omits get_attestation_bytes cannot be
    instantiated — the generation contract is mandatory."""

    class _MissingAttest(TEERuntime):
        @property
        def name(self): return "x"
        @property
        def tee_type(self): return TEEType.SOFTWARE
        @property
        def available(self): return True
        def load(self, wasm_bytes): return None
        def execute(self, module, input_data, resource_limits): return None
        # NB: no get_attestation_bytes

    with pytest.raises(TypeError):
        _MissingAttest()


def test_software_runtime_returns_dev_only_blob():
    rt = SoftwareTEERuntime()                      # instantiates: all abstracts implemented
    blob = rt.get_attestation_bytes()
    assert isinstance(blob, bytes)
    assert blob.startswith(SOFTWARE_TEE_ATTESTATION_PREFIX)   # branded dev-only
    assert len(blob) == 16 + 48                              # prefix + sha384 tag
    assert blob == SOFTWARE_TEE_ATTESTATION_PREFIX + hashlib.sha384(b"sw-tee").digest()


def test_software_blob_is_recognized_dev_only_by_audit_dict():
    """Closes the loop with sp1238/sp1234: the software runtime's blob is
    detectably non-production wherever it flows."""
    from prsm.compute.shard_receipt import tee_attestation_audit_dict
    rt = SoftwareTEERuntime()
    d = tee_attestation_audit_dict(rt.get_attestation_bytes(), TEEType.SOFTWARE)
    assert d["is_dev_only"] is True
    assert d["is_multi_stage"] is False


def test_software_blob_is_stable():
    # deterministic (no per-call randomness) — useful for the no-arg contract
    assert SoftwareTEERuntime().get_attestation_bytes() == \
        SoftwareTEERuntime().get_attestation_bytes()
