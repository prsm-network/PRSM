"""
Unit tests — Phase 3.x.1 Task 4 — TensorParallelInferenceExecutor.

Acceptance per design plan §4 Task 4: end-to-end mock inference returns
correct shape + (unsigned) receipt + DP noise applied per privacy tier.

These tests build real ShardedModel objects with numpy float64 tensor
data and run the actual TensorParallelExecutor — no crypto/numerics
mocks. Per the project's testing rules: do not deprecate to simpler
testing; work through the problem with the real primitives.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal

import numpy as np
import pytest

from prsm.compute.inference import (
    InferenceRequest,
    InferenceResult,
    TensorParallelInferenceExecutor,
    UnsupportedModelError,
)
from prsm.compute.inference.executor import (
    SOFTWARE_TEE_ATTESTATION_PREFIX,
    SOFTWARE_TEE_PRIVACY_ENV,
)
from prsm.compute.inference.models import ContentTier
from prsm.compute.model_sharding.models import ModelShard, ShardedModel
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.compute.tee.runtime import SoftwareTEERuntime, TEERuntime


# ──────────────────────────────────────────────────────────────────────────
# Test helpers — real ShardedModel with deterministic numpy tensors
# ──────────────────────────────────────────────────────────────────────────


def _make_shard(model_id: str, index: int, total: int, rows: int, cols: int) -> ModelShard:
    """Build a real ModelShard with a deterministic 2D tensor.

    Tensor is `rows × cols` filled with `index/(rows*cols)` increments;
    different shards produce different but deterministic outputs so
    all-reduce is observable.
    """
    rng = np.random.default_rng(seed=1000 + index)
    tensor = rng.standard_normal(size=(rows, cols))
    data = tensor.tobytes()
    return ModelShard(
        shard_id=f"{model_id}-shard-{index}",
        model_id=model_id,
        shard_index=index,
        total_shards=total,
        tensor_data=data,
        tensor_shape=(rows, cols),
        layer_range=(0, 0),
        size_bytes=len(data),
        checksum=hashlib.sha256(data).hexdigest(),
    )


def _make_model(model_id: str, num_shards: int = 3, rows: int = 8, cols: int = 16) -> ShardedModel:
    shards = [_make_shard(model_id, i, num_shards, rows, cols) for i in range(num_shards)]
    return ShardedModel(
        model_id=model_id,
        model_name=f"test-{model_id}",
        total_shards=num_shards,
        shards=shards,
    )


@pytest.fixture
def model():
    return _make_model("test-llama")


@pytest.fixture
def registry(model):
    return {model.model_id: model}


@pytest.fixture
def executor(registry):
    # Phase 3.x.1 caveat #1 closed 2026-04-27 — default flipped to
    # safe-fail. Happy-path tests still want to exercise non-NONE
    # privacy tiers on the test's software TEE, so opt in here.
    # Production nodes serving Tier B/C confidential workloads on
    # hardware TEE don't need this kwarg.
    return TensorParallelInferenceExecutor(
        model_registry=registry, allow_software_tee_for_privacy=True
    )


def _make_request(
    *,
    model_id: str = "test-llama",
    prompt: str = "the quick brown fox",
    budget_ftns: Decimal = Decimal("1.0"),
    privacy_tier: PrivacyLevel = PrivacyLevel.STANDARD,
    content_tier: ContentTier = ContentTier.A,
) -> InferenceRequest:
    return InferenceRequest(
        prompt=prompt,
        model_id=model_id,
        budget_ftns=budget_ftns,
        privacy_tier=privacy_tier,
        content_tier=content_tier,
    )


# ──────────────────────────────────────────────────────────────────────────
# Construction + registry
# ──────────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_supported_models_lists_registry(self, registry):
        ex = TensorParallelInferenceExecutor(model_registry=registry)
        assert ex.supported_models() == ["test-llama"]

    def test_register_model_adds_to_registry(self, executor):
        m2 = _make_model("test-mistral", num_shards=2)
        executor.register_model(m2)
        assert "test-mistral" in executor.supported_models()

    def test_register_model_duplicate_raises(self, executor, registry):
        # Phase 3.x.2 changes the contract from silent-replace to
        # explicit-rejection. Registry-backed semantics: a duplicate
        # model_id is a publisher conflict, not an upsert. Callers
        # needing replacement must build a fresh registry.
        from prsm.compute.model_registry import ModelAlreadyRegisteredError
        m_new = _make_model("test-llama", num_shards=5)
        with pytest.raises(ModelAlreadyRegisteredError):
            executor.register_model(m_new)

    def test_default_tee_runtime_is_software(self, registry):
        ex = TensorParallelInferenceExecutor(model_registry=registry)
        assert ex.tee_runtime.tee_type == TEEType.SOFTWARE

    def test_custom_tee_runtime_used(self, registry):
        class FakeHardwareTEE(TEERuntime):
            @property
            def name(self): return "sgx-fake"
            @property
            def tee_type(self): return TEEType.SGX
            @property
            def available(self): return True
            def load(self, wasm_bytes): return None
            def execute(self, module, input_data, resource_limits): raise NotImplementedError
            def get_attestation_bytes(self): return b"sgx-fake-quote"  # sp1239

        ex = TensorParallelInferenceExecutor(
            model_registry=registry, tee_runtime=FakeHardwareTEE()
        )
        assert ex.tee_runtime.tee_type == TEEType.SGX


# ──────────────────────────────────────────────────────────────────────────
# Cost estimation
# ──────────────────────────────────────────────────────────────────────────


class TestEstimateCost:
    @pytest.mark.asyncio
    async def test_cost_scales_with_total_shards(self, executor):
        # 3 shards × 0.05 FTNS × 1.10 (standard overhead) = 0.165
        req = _make_request(privacy_tier=PrivacyLevel.STANDARD)
        cost = await executor.estimate_cost(req)
        assert cost == Decimal("0.05") * Decimal(3) * Decimal("1.10")

    @pytest.mark.asyncio
    async def test_overhead_increases_with_privacy_tier(self, executor):
        none_cost = await executor.estimate_cost(_make_request(privacy_tier=PrivacyLevel.NONE))
        std_cost = await executor.estimate_cost(_make_request(privacy_tier=PrivacyLevel.STANDARD))
        high_cost = await executor.estimate_cost(_make_request(privacy_tier=PrivacyLevel.HIGH))
        max_cost = await executor.estimate_cost(_make_request(privacy_tier=PrivacyLevel.MAXIMUM))
        assert none_cost < std_cost < high_cost < max_cost

    @pytest.mark.asyncio
    async def test_unknown_model_raises(self, executor):
        with pytest.raises(UnsupportedModelError, match="Unknown model_id"):
            await executor.estimate_cost(_make_request(model_id="does-not-exist"))


# ──────────────────────────────────────────────────────────────────────────
# Happy-path execution
# ──────────────────────────────────────────────────────────────────────────


class TestExecuteHappyPath:
    @pytest.mark.asyncio
    async def test_tier_a_standard_privacy_returns_signed_shape(self, executor):
        req = _make_request(privacy_tier=PrivacyLevel.STANDARD, content_tier=ContentTier.A)
        result = await executor.execute(req)
        assert result.success
        assert isinstance(result, InferenceResult)
        assert result.output  # non-empty
        assert result.receipt is not None

    @pytest.mark.asyncio
    async def test_receipt_fields_populated(self, executor):
        req = _make_request()
        result = await executor.execute(req)
        receipt = result.receipt
        assert receipt.job_id.startswith("infer-job-")
        assert receipt.request_id == req.request_id
        assert receipt.model_id == "test-llama"
        assert receipt.tee_type == TEEType.SOFTWARE
        # Tee attestation is the 64-byte sha512 marker
        assert len(receipt.tee_attestation) == 64
        # Output hash matches sha256 of output text
        assert receipt.output_hash == hashlib.sha256(result.output.encode("utf-8")).digest()
        # Settler signature is empty (signed at API layer Task 5)
        assert receipt.settler_signature == b"\x00" * 64
        assert receipt.settler_node_id == ""
        # Cost matches estimate
        expected_cost = await executor.estimate_cost(req)
        assert receipt.cost_ftns == expected_cost
        # Duration is positive
        assert receipt.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_receipt_carries_request_tiers(self, executor):
        req = _make_request(
            privacy_tier=PrivacyLevel.HIGH,
            content_tier=ContentTier.A,
        )
        result = await executor.execute(req)
        assert result.receipt.privacy_tier == PrivacyLevel.HIGH
        assert result.receipt.content_tier == ContentTier.A

    @pytest.mark.asyncio
    async def test_attestation_is_job_bound(self, executor):
        # Two separate executions produce distinct attestation blobs
        # because they have distinct job_ids.
        r1 = await executor.execute(_make_request())
        r2 = await executor.execute(_make_request())
        assert r1.receipt.tee_attestation != r2.receipt.tee_attestation
        assert r1.receipt.job_id != r2.receipt.job_id

    @pytest.mark.asyncio
    async def test_software_tee_attestation_has_dev_marker_prefix(self, executor):
        # Software-TEE attestations must start with the DEV-ONLY prefix so
        # a single byte-prefix check flags them as non-production proofs,
        # without requiring the verifier to inspect receipt.tee_type.
        result = await executor.execute(_make_request())
        att = result.receipt.tee_attestation
        assert att.startswith(SOFTWARE_TEE_ATTESTATION_PREFIX)
        assert len(att) == 64
        # The 16-byte prefix is human-readable ASCII so the marker is
        # visible in any hex/log dump of the receipt.
        assert SOFTWARE_TEE_ATTESTATION_PREFIX == b"DEV-ONLY-SW-TEE:"


# ──────────────────────────────────────────────────────────────────────────
# DP noise injection per privacy tier
# ──────────────────────────────────────────────────────────────────────────


class TestDPNoise:
    @pytest.mark.asyncio
    async def test_privacy_none_records_zero_epsilon(self, executor):
        result = await executor.execute(_make_request(privacy_tier=PrivacyLevel.NONE))
        assert result.success
        assert result.receipt.epsilon_spent == 0.0

    @pytest.mark.asyncio
    async def test_privacy_standard_records_eps_8(self, executor):
        result = await executor.execute(_make_request(privacy_tier=PrivacyLevel.STANDARD))
        assert result.receipt.epsilon_spent == 8.0

    @pytest.mark.asyncio
    async def test_privacy_high_records_eps_4(self, executor):
        result = await executor.execute(_make_request(privacy_tier=PrivacyLevel.HIGH))
        assert result.receipt.epsilon_spent == 4.0

    @pytest.mark.asyncio
    async def test_privacy_maximum_records_eps_1(self, executor):
        result = await executor.execute(_make_request(privacy_tier=PrivacyLevel.MAXIMUM))
        assert result.receipt.epsilon_spent == 1.0

    @pytest.mark.asyncio
    async def test_dp_noise_is_actually_applied(self, executor):
        # Same prompt + model_id + content_tier under NONE vs MAXIMUM
        # privacy must produce different output strings: NONE skips DP
        # entirely, MAXIMUM clips + adds calibrated Gaussian noise to
        # the aggregated tensor. The test holds regardless of RNG seed
        # because the privacy-tier branch decides whether noise runs at
        # all — there's no "noise == zero" coincidence to defend against.
        r_none = await executor.execute(_make_request(privacy_tier=PrivacyLevel.NONE))
        r_max = await executor.execute(
            _make_request(privacy_tier=PrivacyLevel.MAXIMUM, prompt="the quick brown fox")
        )
        assert r_none.output != r_max.output


# ──────────────────────────────────────────────────────────────────────────
# Failure paths
# ──────────────────────────────────────────────────────────────────────────


class TestFailurePaths:
    @pytest.mark.asyncio
    async def test_unknown_model_returns_failure(self, executor):
        result = await executor.execute(_make_request(model_id="not-a-model"))
        assert not result.success
        assert "Unknown model_id" in (result.error or "")
        assert result.receipt is None

    @pytest.mark.asyncio
    async def test_insufficient_budget_returns_failure(self, executor):
        # 3 shards × 0.05 × 1.10 = 0.165; budget=0.10 < cost.
        result = await executor.execute(_make_request(budget_ftns=Decimal("0.10")))
        assert not result.success
        assert "Insufficient budget" in (result.error or "")

    @pytest.mark.asyncio
    async def test_software_tee_blocked_when_disallowed_for_privacy(self, registry):
        ex = TensorParallelInferenceExecutor(
            model_registry=registry,
            allow_software_tee_for_privacy=False,
        )
        result = await ex.execute(_make_request(privacy_tier=PrivacyLevel.STANDARD))
        assert not result.success
        assert "hardware TEE" in (result.error or "")

    @pytest.mark.asyncio
    async def test_software_tee_disallowed_still_passes_for_privacy_none(self, registry):
        # privacy_tier=NONE doesn't need TEE — must succeed even when
        # software-TEE is gated for privacy levels.
        ex = TensorParallelInferenceExecutor(
            model_registry=registry,
            allow_software_tee_for_privacy=False,
        )
        result = await ex.execute(_make_request(privacy_tier=PrivacyLevel.NONE))
        assert result.success

    @pytest.mark.asyncio
    async def test_empty_model_returns_failure(self, registry):
        # Construct a model with zero shards → encoding raises → graceful failure
        empty = ShardedModel(
            model_id="empty", model_name="empty", total_shards=0, shards=[]
        )
        ex = TensorParallelInferenceExecutor(model_registry={"empty": empty})
        result = await ex.execute(_make_request(model_id="empty", privacy_tier=PrivacyLevel.NONE))
        assert not result.success
        assert "Prompt encoding failed" in (result.error or "")


# ──────────────────────────────────────────────────────────────────────────
# Determinism + numerics integration
# ──────────────────────────────────────────────────────────────────────────


class TestNumerics:
    @pytest.mark.asyncio
    async def test_privacy_none_is_deterministic(self, executor):
        # privacy_tier=NONE skips DP noise → same prompt should produce
        # identical output text for the deterministic portion.
        r1 = await executor.execute(_make_request(privacy_tier=PrivacyLevel.NONE))
        r2 = await executor.execute(_make_request(privacy_tier=PrivacyLevel.NONE))
        # Job IDs differ so receipts differ; the OUTPUT text must match.
        assert r1.output == r2.output

    @pytest.mark.asyncio
    async def test_different_prompts_produce_different_output(self, executor):
        r1 = await executor.execute(
            _make_request(privacy_tier=PrivacyLevel.NONE, prompt="prompt A")
        )
        r2 = await executor.execute(
            _make_request(privacy_tier=PrivacyLevel.NONE, prompt="prompt B different length")
        )
        assert r1.output != r2.output

    @pytest.mark.asyncio
    async def test_output_format_includes_diagnostics(self, executor):
        result = await executor.execute(
            _make_request(prompt="hello", privacy_tier=PrivacyLevel.STANDARD,
                          content_tier=ContentTier.A)
        )
        assert "test-llama" in result.output
        assert "prompt_len=5" in result.output
        assert "privacy=standard" in result.output
        assert "content_tier=A" in result.output
        assert "output_dim=" in result.output
        assert "sample=" in result.output


# ──────────────────────────────────────────────────────────────────────────
# Public-API contract: drop-in for MockInferenceExecutor
# ──────────────────────────────────────────────────────────────────────────


class TestSoftwareTEEPrivacyResolution:
    """Round 2 review LOW#3 — operator-facing controls for the
    software-TEE-accepts-privacy default."""

    def test_explicit_arg_takes_precedence_over_env(self, registry, monkeypatch):
        # Even when env says off, an explicit True wins.
        monkeypatch.setenv(SOFTWARE_TEE_PRIVACY_ENV, "0")
        ex = TensorParallelInferenceExecutor(
            model_registry=registry, allow_software_tee_for_privacy=True
        )
        assert ex._allow_software_tee_for_privacy is True

    def test_explicit_false_arg_overrides_env(self, registry, monkeypatch):
        monkeypatch.setenv(SOFTWARE_TEE_PRIVACY_ENV, "1")
        ex = TensorParallelInferenceExecutor(
            model_registry=registry, allow_software_tee_for_privacy=False
        )
        assert ex._allow_software_tee_for_privacy is False

    def test_env_off_disables_when_arg_unset(self, registry, monkeypatch):
        for value in ["0", "false", "FALSE", "no", "off", ""]:
            monkeypatch.setenv(SOFTWARE_TEE_PRIVACY_ENV, value)
            ex = TensorParallelInferenceExecutor(model_registry=registry)
            assert ex._allow_software_tee_for_privacy is False, (
                f"env={value!r} should disable"
            )

    def test_env_on_enables_when_arg_unset(self, registry, monkeypatch):
        for value in ["1", "true", "TRUE", "yes", "on"]:
            monkeypatch.setenv(SOFTWARE_TEE_PRIVACY_ENV, value)
            ex = TensorParallelInferenceExecutor(model_registry=registry)
            assert ex._allow_software_tee_for_privacy is True, (
                f"env={value!r} should enable"
            )

    def test_default_is_false_when_neither_arg_nor_env_set(self, registry, monkeypatch):
        # Phase 3.x.1 caveat #1 closed 2026-04-27 — default flipped from
        # True (dev-friendly) to False (safe-fail) as part of pre-mainnet
        # hardening. Operators serving Tier B/C confidential workloads
        # on software-only nodes must explicitly opt in via env or kwarg.
        monkeypatch.delenv(SOFTWARE_TEE_PRIVACY_ENV, raising=False)
        ex = TensorParallelInferenceExecutor(model_registry=registry)
        assert ex._allow_software_tee_for_privacy is False

    def test_constructor_warns_when_explicit_opt_in_on_software_tee(
        self, registry, monkeypatch, caplog
    ):
        # Operator who explicitly opts in to non-NONE privacy on a
        # software TEE gets a warning so the dev-vs-prod tradeoff is
        # surfaced in logs. Triggered by env=on (or arg=True); not the
        # safe-fail default.
        import logging
        monkeypatch.setenv(SOFTWARE_TEE_PRIVACY_ENV, "1")
        caplog.set_level(logging.WARNING, logger="prsm.compute.inference.executor")
        TensorParallelInferenceExecutor(model_registry=registry)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any(
            SOFTWARE_TEE_PRIVACY_ENV in r.message
            and "software TEE accepts" in r.message
            for r in warnings
        ), f"missing operator warning; got: {[r.message for r in warnings]}"

    def test_no_warning_on_safe_fail_default(
        self, registry, monkeypatch, caplog
    ):
        # The new safe-fail default is silent — no warning needed when
        # the safe configuration is selected.
        import logging
        monkeypatch.delenv(SOFTWARE_TEE_PRIVACY_ENV, raising=False)
        caplog.set_level(logging.WARNING, logger="prsm.compute.inference.executor")
        TensorParallelInferenceExecutor(model_registry=registry)
        assert not any(
            "software TEE accepts" in r.message
            for r in caplog.records
        )

    def test_no_warning_when_software_tee_rejects_privacy(
        self, registry, monkeypatch, caplog
    ):
        # Explicit False is also silent — same as the default.
        import logging
        caplog.set_level(logging.WARNING, logger="prsm.compute.inference.executor")
        TensorParallelInferenceExecutor(
            model_registry=registry, allow_software_tee_for_privacy=False
        )
        assert not any(
            "software TEE accepts" in r.message
            for r in caplog.records
        )


class TestModelRegistryIntegration:
    """Phase 3.x.2 Task 5 — executor accepts ModelRegistry directly,
    and registry-level verification (signature + shard sha256) gates
    every execute() call. Tampering surfaces as InferenceResult.failure,
    not silent corruption."""

    @pytest.fixture
    def real_identity(self):
        from prsm.node.identity import generate_node_identity
        return generate_node_identity(display_name="phase3.x.2-task5-publisher")

    @pytest.fixture
    def in_memory_registry(self, model, real_identity):
        from prsm.compute.model_registry import InMemoryModelRegistry
        reg = InMemoryModelRegistry()
        reg.register(model, identity=real_identity)
        return reg

    def test_constructor_accepts_model_registry(self, in_memory_registry):
        ex = TensorParallelInferenceExecutor(model_registry=in_memory_registry)
        assert ex.supported_models() == ["test-llama"]

    def test_constructor_accepts_dict_for_back_compat(self, model):
        # Phase 3.x.1 callers continue to work unchanged
        ex = TensorParallelInferenceExecutor(model_registry={model.model_id: model})
        assert ex.supported_models() == ["test-llama"]

    def test_constructor_rejects_other_types(self):
        with pytest.raises(TypeError, match="ModelRegistry"):
            TensorParallelInferenceExecutor(model_registry=["not", "a", "dict"])

    def test_dict_constructor_uses_scaffold_identity(self, model):
        # Dict-arg path generates a scaffold identity to sign in-memory
        # manifests. The scaffold publisher_node_id appears on the
        # manifest; never on inference receipts.
        ex = TensorParallelInferenceExecutor(model_registry={model.model_id: model})
        manifest = ex.registry.get_manifest(model.model_id)
        assert manifest.publisher_node_id == ex._scaffold_identity.node_id

    def test_registry_constructor_does_not_create_scaffold(self, in_memory_registry):
        # When the caller supplied a real registry, the executor MUST
        # NOT spawn a scaffold identity unprompted (it would be unused).
        ex = TensorParallelInferenceExecutor(model_registry=in_memory_registry)
        assert ex._scaffold_identity is None

    @pytest.mark.asyncio
    async def test_happy_path_via_explicit_registry(
        self, in_memory_registry
    ):
        # Test exercises STANDARD privacy on a software TEE — opt in
        # explicitly per the post-caveat#1-close safe-fail default.
        ex = TensorParallelInferenceExecutor(
            model_registry=in_memory_registry,
            allow_software_tee_for_privacy=True,
        )
        result = await ex.execute(_make_request())
        assert result.success
        assert result.receipt is not None

    @pytest.mark.asyncio
    async def test_unknown_model_via_registry(self, in_memory_registry):
        ex = TensorParallelInferenceExecutor(model_registry=in_memory_registry)
        result = await ex.execute(_make_request(model_id="nonexistent"))
        assert not result.success
        assert "Unknown model_id" in (result.error or "")

    @pytest.mark.asyncio
    async def test_shard_tamper_at_execute_returns_failure(
        self, model, real_identity
    ):
        # Build a registry, register, then mutate stored shard bytes
        # directly. The next execute() must fail with a verification
        # error in the result, NOT raise.
        from prsm.compute.model_registry import InMemoryModelRegistry
        reg = InMemoryModelRegistry()
        reg.register(model, identity=real_identity)
        # Reach into the registry's stored model and corrupt a shard
        from prsm.compute.model_sharding.models import ModelShard
        stored = reg._models[model.model_id]
        s0 = stored.shards[0]
        stored.shards[0] = ModelShard(
            shard_id=s0.shard_id, model_id=s0.model_id,
            shard_index=s0.shard_index, total_shards=s0.total_shards,
            tensor_data=bytes([s0.tensor_data[0] ^ 0xFF]) + s0.tensor_data[1:],
            tensor_shape=s0.tensor_shape, layer_range=s0.layer_range,
            size_bytes=s0.size_bytes, checksum=s0.checksum,
        )
        ex = TensorParallelInferenceExecutor(model_registry=reg)
        result = await ex.execute(_make_request())
        assert not result.success
        assert "verification failed" in (result.error or "").lower()
        assert result.receipt is None

    @pytest.mark.asyncio
    async def test_estimate_cost_via_get_manifest(self, in_memory_registry):
        # estimate_cost must use the cheaper get_manifest path. Even if
        # we corrupt shard bytes, the cost estimate should still work
        # because it doesn't read shard data.
        from prsm.compute.model_sharding.models import ModelShard
        stored = in_memory_registry._models["test-llama"]
        s0 = stored.shards[0]
        stored.shards[0] = ModelShard(
            shard_id=s0.shard_id, model_id=s0.model_id,
            shard_index=s0.shard_index, total_shards=s0.total_shards,
            tensor_data=bytes([s0.tensor_data[0] ^ 0xFF]) + s0.tensor_data[1:],
            tensor_shape=s0.tensor_shape, layer_range=s0.layer_range,
            size_bytes=s0.size_bytes, checksum=s0.checksum,
        )
        ex = TensorParallelInferenceExecutor(model_registry=in_memory_registry)
        # Cost should still be computable
        cost = await ex.estimate_cost(_make_request())
        assert cost > 0

    def test_register_model_with_explicit_identity(
        self, in_memory_registry, real_identity
    ):
        ex = TensorParallelInferenceExecutor(model_registry=in_memory_registry)
        m2 = _make_model("second-model")
        ex.register_model(m2, identity=real_identity)
        # Manifest carries the explicit publisher's node_id, not a
        # scaffold's
        manifest = ex.registry.get_manifest("second-model")
        assert manifest.publisher_node_id == real_identity.node_id

    def test_register_model_without_identity_uses_scaffold(
        self, in_memory_registry
    ):
        ex = TensorParallelInferenceExecutor(model_registry=in_memory_registry)
        # Real-registry construction left scaffold None; first
        # register_model() without identity generates one and stashes it.
        assert ex._scaffold_identity is None
        m2 = _make_model("second-model")
        ex.register_model(m2)
        assert ex._scaffold_identity is not None
        manifest = ex.registry.get_manifest("second-model")
        assert manifest.publisher_node_id == ex._scaffold_identity.node_id


class TestInterfaceContract:
    @pytest.mark.asyncio
    async def test_subclasses_inferenceexecutor(self, executor):
        from prsm.compute.inference.executor import InferenceExecutor
        assert isinstance(executor, InferenceExecutor)

    @pytest.mark.asyncio
    async def test_uses_request_id_from_request(self, executor):
        req = _make_request()
        result = await executor.execute(req)
        assert result.request_id == req.request_id

    @pytest.mark.asyncio
    async def test_signed_receipt_roundtrips_through_to_dict(self, executor):
        result = await executor.execute(_make_request())
        d = result.receipt.to_dict()
        # All the canonical fields are present + serializable
        for key in [
            "job_id", "request_id", "model_id", "content_tier",
            "privacy_tier", "epsilon_spent", "tee_type",
            "tee_attestation", "output_hash", "duration_seconds",
            "cost_ftns", "settler_signature", "settler_node_id",
        ]:
            assert key in d
