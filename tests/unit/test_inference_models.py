"""Tests for inference data models (Phase 3.x.1 Task 1 scaffold)."""

import asyncio
from decimal import Decimal

import pytest

from prsm.compute.inference import (
    ContentTier,
    InferenceExecutor,
    InferenceReceipt,
    InferenceRequest,
    InferenceResult,
    MockInferenceExecutor,
    UnsupportedModelError,
    default_mock_executor,
)
from prsm.compute.tee.models import PrivacyLevel, TEEType


# ── ContentTier ─────────────────────────────────────────────────────────────


class TestContentTier:
    def test_enum_values(self):
        assert ContentTier.A.value == "A"
        assert ContentTier.B.value == "B"
        assert ContentTier.C.value == "C"

    def test_str_enum(self):
        assert ContentTier.A == "A"
        assert isinstance(ContentTier.B, str)

    def test_from_string(self):
        assert ContentTier("A") is ContentTier.A
        assert ContentTier("C") is ContentTier.C

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            ContentTier("D")


# ── InferenceRequest ────────────────────────────────────────────────────────


class TestInferenceRequest:
    def test_minimal_construction(self):
        req = InferenceRequest(
            prompt="Hello",
            model_id="mock-llama-3-8b",
            budget_ftns=Decimal("1.0"),
        )
        assert req.prompt == "Hello"
        assert req.privacy_tier == PrivacyLevel.NONE  # sp1234 — honest default (DP is experimental)
        assert req.content_tier == ContentTier.A
        assert req.budget_ftns == Decimal("1.0")
        assert req.request_id.startswith("infer-")

    def test_request_id_unique(self):
        r1 = InferenceRequest(prompt="x", model_id="m", budget_ftns=Decimal("1"))
        r2 = InferenceRequest(prompt="x", model_id="m", budget_ftns=Decimal("1"))
        assert r1.request_id != r2.request_id

    def test_explicit_request_id_preserved(self):
        req = InferenceRequest(
            prompt="x",
            model_id="m",
            budget_ftns=Decimal("1"),
            request_id="my-fixed-id",
        )
        assert req.request_id == "my-fixed-id"

    def test_budget_coerced_to_decimal(self):
        # Pass a float; should coerce
        req = InferenceRequest(prompt="x", model_id="m", budget_ftns=1.5)
        assert isinstance(req.budget_ftns, Decimal)
        assert req.budget_ftns == Decimal("1.5")

    def test_privacy_tier_coerced_from_string(self):
        req = InferenceRequest(
            prompt="x",
            model_id="m",
            budget_ftns=Decimal("1"),
            privacy_tier="high",
        )
        assert req.privacy_tier == PrivacyLevel.HIGH

    def test_content_tier_coerced_from_string(self):
        req = InferenceRequest(
            prompt="x",
            model_id="m",
            budget_ftns=Decimal("1"),
            content_tier="B",
        )
        assert req.content_tier == ContentTier.B

    def test_optional_fields_default_none(self):
        req = InferenceRequest(prompt="x", model_id="m", budget_ftns=Decimal("1"))
        assert req.max_tokens is None
        assert req.temperature is None
        assert req.requester_node_id is None

    def test_to_dict_from_dict_roundtrip(self):
        original = InferenceRequest(
            prompt="What is the meaning of life?",
            model_id="mock-mistral-7b",
            budget_ftns=Decimal("2.5"),
            privacy_tier=PrivacyLevel.HIGH,
            content_tier=ContentTier.B,
            max_tokens=128,
            temperature=0.7,
            requester_node_id="node-abc",
            request_id="infer-test-123",
        )
        d = original.to_dict()
        # All values JSON-serializable (no Decimal/Enum bare in dict)
        assert isinstance(d["budget_ftns"], str)
        assert d["privacy_tier"] == "high"
        assert d["content_tier"] == "B"

        restored = InferenceRequest.from_dict(d)
        assert restored == original

    def test_from_dict_drops_unknown_keys(self):
        d = {
            "prompt": "x",
            "model_id": "m",
            "budget_ftns": "1.0",
            "schema_version_v9": "future-field",  # added in some future version
        }
        req = InferenceRequest.from_dict(d)
        assert req.prompt == "x"

    def test_immutable(self):
        req = InferenceRequest(prompt="x", model_id="m", budget_ftns=Decimal("1"))
        with pytest.raises((AttributeError, Exception)):
            req.prompt = "changed"  # frozen dataclass


# ── InferenceReceipt ────────────────────────────────────────────────────────


class TestInferenceReceipt:
    def _build(self, **overrides) -> InferenceReceipt:
        defaults = dict(
            job_id="job-abc",
            request_id="req-xyz",
            model_id="mock-llama-3-8b",
            content_tier=ContentTier.A,
            privacy_tier=PrivacyLevel.STANDARD,
            epsilon_spent=8.0,
            tee_type=TEEType.SOFTWARE,
            tee_attestation=b"\x01\x02\x03",
            output_hash=b"\xaa" * 32,
            duration_seconds=1.5,
            cost_ftns=Decimal("0.5"),
            settler_signature=b"\xbb" * 64,
            settler_node_id="node-settler-1",
        )
        defaults.update(overrides)
        return InferenceReceipt(**defaults)

    def test_construction(self):
        r = self._build()
        assert r.job_id == "job-abc"
        assert r.cost_ftns == Decimal("0.5")
        assert r.tee_type == TEEType.SOFTWARE

    def test_to_dict_serializable(self):
        r = self._build()
        d = r.to_dict()
        # All bytes hex-encoded; Decimal stringified
        assert isinstance(d["tee_attestation"], str)
        assert isinstance(d["output_hash"], str)
        assert isinstance(d["settler_signature"], str)
        assert d["cost_ftns"] == "0.5"
        # Hex round-trips
        assert bytes.fromhex(d["tee_attestation"]) == b"\x01\x02\x03"

    def test_to_dict_from_dict_roundtrip(self):
        original = self._build()
        restored = InferenceReceipt.from_dict(original.to_dict())
        assert restored == original

    def test_signing_payload_deterministic(self):
        r1 = self._build()
        r2 = self._build()
        assert r1.signing_payload() == r2.signing_payload()

    def test_signing_payload_excludes_signature(self):
        r1 = self._build(settler_signature=b"\x01" * 64)
        r2 = self._build(settler_signature=b"\x02" * 64)
        # Different signatures, but signing payload should be identical
        # (signing payload is what gets signed; including the signature would
        # be circular)
        assert r1.signing_payload() == r2.signing_payload()

    def test_signing_payload_changes_with_content(self):
        r1 = self._build(model_id="model-a")
        r2 = self._build(model_id="model-b")
        assert r1.signing_payload() != r2.signing_payload()

    def test_immutable(self):
        r = self._build()
        with pytest.raises((AttributeError, Exception)):
            r.cost_ftns = Decimal("999")  # frozen


# ── InferenceResult ─────────────────────────────────────────────────────────


class TestInferenceResult:
    def test_failure_helper(self):
        result = InferenceResult.failure("req-1", "budget too low")
        assert not result.success
        assert result.error == "budget too low"
        assert result.output == ""
        assert result.receipt is None
        assert result.request_id == "req-1"

    def test_success_construction(self):
        receipt = InferenceReceipt(
            job_id="j",
            request_id="r",
            model_id="m",
            content_tier=ContentTier.A,
            privacy_tier=PrivacyLevel.STANDARD,
            epsilon_spent=8.0,
            tee_type=TEEType.SOFTWARE,
            tee_attestation=b"",
            output_hash=b"\x00" * 32,
            duration_seconds=0.1,
            cost_ftns=Decimal("0.1"),
        )
        result = InferenceResult(
            request_id="r",
            success=True,
            output="Hello, world!",
            receipt=receipt,
        )
        assert result.success
        assert result.error is None
        assert result.output == "Hello, world!"

    def test_to_dict_from_dict_roundtrip_success(self):
        receipt = InferenceReceipt(
            job_id="j",
            request_id="r",
            model_id="m",
            content_tier=ContentTier.A,
            privacy_tier=PrivacyLevel.STANDARD,
            epsilon_spent=8.0,
            tee_type=TEEType.SOFTWARE,
            tee_attestation=b"\x11",
            output_hash=b"\x22" * 32,
            duration_seconds=0.5,
            cost_ftns=Decimal("0.25"),
        )
        original = InferenceResult(
            request_id="r",
            success=True,
            output="result",
            receipt=receipt,
        )
        restored = InferenceResult.from_dict(original.to_dict())
        assert restored == original

    def test_to_dict_from_dict_roundtrip_failure(self):
        original = InferenceResult.failure("req-1", "model not registered")
        restored = InferenceResult.from_dict(original.to_dict())
        assert restored == original


# ── MockInferenceExecutor ───────────────────────────────────────────────────


class TestMockInferenceExecutor:
    def test_supported_models_default(self):
        ex = MockInferenceExecutor()
        models = ex.supported_models()
        assert "mock-llama-3-8b" in models
        assert len(models) == 3

    def test_supported_models_custom(self):
        ex = MockInferenceExecutor(models=["my-model-1", "my-model-2"])
        assert ex.supported_models() == ["my-model-1", "my-model-2"]

    def test_estimate_cost_known_model(self):
        ex = MockInferenceExecutor(fixed_cost=Decimal("0.5"))
        req = InferenceRequest(
            prompt="x", model_id="mock-llama-3-8b", budget_ftns=Decimal("1")
        )
        cost = asyncio.run(ex.estimate_cost(req))
        assert cost == Decimal("0.5")

    def test_estimate_cost_unknown_model_raises(self):
        ex = MockInferenceExecutor()
        req = InferenceRequest(
            prompt="x", model_id="unregistered-model", budget_ftns=Decimal("1")
        )
        with pytest.raises(UnsupportedModelError):
            asyncio.run(ex.estimate_cost(req))

    def test_execute_success(self):
        ex = MockInferenceExecutor()
        req = InferenceRequest(
            prompt="What is 2+2?",
            model_id="mock-llama-3-8b",
            budget_ftns=Decimal("1.0"),
        )
        result = asyncio.run(ex.execute(req))
        assert result.success
        assert result.output  # non-empty
        assert result.receipt is not None
        assert result.receipt.request_id == req.request_id
        assert result.receipt.model_id == req.model_id

    def test_execute_deterministic(self):
        ex = MockInferenceExecutor()
        # Same prompt + model + fixed request_id → same output
        req1 = InferenceRequest(
            prompt="hello",
            model_id="mock-llama-3-8b",
            budget_ftns=Decimal("1"),
            request_id="r1",
        )
        req2 = InferenceRequest(
            prompt="hello",
            model_id="mock-llama-3-8b",
            budget_ftns=Decimal("1"),
            request_id="r2",
        )
        out1 = asyncio.run(ex.execute(req1)).output
        out2 = asyncio.run(ex.execute(req2)).output
        assert out1 == out2  # output is deterministic in prompt+model_id

    def test_execute_unknown_model_returns_failure(self):
        ex = MockInferenceExecutor()
        req = InferenceRequest(
            prompt="x", model_id="not-registered", budget_ftns=Decimal("1")
        )
        result = asyncio.run(ex.execute(req))
        assert not result.success
        assert "Unknown model_id" in (result.error or "")
        assert result.receipt is None

    def test_execute_budget_too_low_returns_failure(self):
        ex = MockInferenceExecutor(fixed_cost=Decimal("1.0"))
        req = InferenceRequest(
            prompt="x",
            model_id="mock-llama-3-8b",
            budget_ftns=Decimal("0.01"),
        )
        result = asyncio.run(ex.execute(req))
        assert not result.success
        assert "Insufficient budget" in (result.error or "")

    def test_default_factory(self):
        ex = default_mock_executor()
        assert isinstance(ex, MockInferenceExecutor)
        assert isinstance(ex, InferenceExecutor)


# ── Module-level acceptance criteria for Task 1 ─────────────────────────────


class TestTask1Acceptance:
    """Validates the explicit acceptance criteria from Phase 3.x.1 Task 1."""

    def test_can_import_inference_executor(self):
        # Acceptance: `from prsm.compute.inference import InferenceExecutor` works
        from prsm.compute.inference import InferenceExecutor  # noqa: F401

    def test_types_serializable(self):
        # Acceptance: types serializable
        req = InferenceRequest(prompt="x", model_id="m", budget_ftns=Decimal("1"))
        result = InferenceResult.failure(req.request_id, "test")
        # Roundtrip both
        assert InferenceRequest.from_dict(req.to_dict()) == req
        assert InferenceResult.from_dict(result.to_dict()) == result

    def test_executor_is_abstract(self):
        # InferenceExecutor itself cannot be instantiated
        with pytest.raises(TypeError):
            InferenceExecutor()  # type: ignore[abstract]

    def test_mock_executor_satisfies_interface(self):
        ex = default_mock_executor()
        assert isinstance(ex, InferenceExecutor)
