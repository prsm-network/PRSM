"""Phase 3.x.7 Task 2 — LayerStageServer unit tests.

Coverage matches design plan §4 Task 2 acceptance:
  - Happy path returns signed response
  - Each error code path: MALFORMED_REQUEST, INVALID_TOKEN,
    DEADLINE_EXCEEDED, MODEL_NOT_FOUND, SHARD_MISSING, TIER_GATE,
    ACTIVATION_INVALID, TIMEOUT, INTERNAL_ERROR, UNSUPPORTED_VERSION
  - Never-raises invariant: parametrized garbage inputs
  - Construction validation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from prsm.compute.chain_rpc.protocol import (
    CHAIN_RPC_PROTOCOL_VERSION,
    ChainRpcMessageType,
    HandoffToken,
    RunLayerSliceRequest,
    RunLayerSliceResponse,
    StageError,
    StageErrorCode,
    encode_message,
    parse_message,
)
from prsm.compute.chain_rpc.server import (
    LayerSliceResult,
    LayerSliceRunner,
    LayerStageServer,
)
from prsm.compute.inference.models import ContentTier
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import generate_node_identity


# ──────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────


class FakeAnchor:
    def __init__(self, registered: Optional[Dict[str, str]] = None):
        self.registered: Dict[str, str] = dict(registered or {})

    def lookup(self, node_id: str) -> Optional[str]:
        return self.registered.get(node_id)


def _register(anchor: FakeAnchor, identity) -> None:
    anchor.registered[identity.node_id] = identity.public_key_b64


@dataclass
class FakeShard:
    layer_range: Tuple[int, int]


@dataclass
class FakeModel:
    model_id: str
    shards: List[FakeShard] = field(default_factory=list)

    @classmethod
    def linear_chain(cls, model_id: str, total_layers: int) -> "FakeModel":
        """Build a model with one shard covering all layers — convenient
        when the test cares about coverage but not granular shard layout."""
        return cls(
            model_id=model_id,
            shards=[FakeShard(layer_range=(0, total_layers))],
        )


@dataclass
class FakeRegistry:
    models: Dict[str, FakeModel] = field(default_factory=dict)
    raise_on_get: Optional[Exception] = None

    def get(self, model_id: str) -> FakeModel:
        if self.raise_on_get is not None:
            raise self.raise_on_get
        if model_id not in self.models:
            # Simulate the real registry's NotFound semantics by class
            # name (the server pattern-matches on "NotFound" in the
            # exception class name).
            raise _ModelNotFoundError(f"unknown model {model_id!r}")
        return self.models[model_id]


class _ModelNotFoundError(Exception):
    pass


class _ManifestVerificationError(Exception):
    pass


class FakeTEERuntime:
    def __init__(self, tee_type: TEEType = TEEType.SOFTWARE):
        self.tee_type = tee_type


class FakeRunner(LayerSliceRunner):
    def __init__(
        self,
        *,
        output_factory=None,
        duration: float = 0.05,
        tee_attestation: bytes = b"\x01" * 32,
        tee_type: TEEType = TEEType.SOFTWARE,
        epsilon: float = 0.0,
        raise_on_call: Optional[Exception] = None,
    ):
        self.output_factory = output_factory
        self.duration = duration
        self.tee_attestation = tee_attestation
        self.tee_type = tee_type
        self.epsilon = epsilon
        self.raise_on_call = raise_on_call
        self.calls: List[Dict[str, Any]] = []

    def run_layer_range(
        self,
        *,
        model: Any,
        layer_range: Tuple[int, int],
        activation: np.ndarray,
        privacy_tier: PrivacyLevel,
        is_final_stage: bool,
    ) -> LayerSliceResult:
        self.calls.append({
            "model": model,
            "layer_range": layer_range,
            "activation_shape": activation.shape,
            "activation_dtype": str(activation.dtype),
            "privacy_tier": privacy_tier,
            "is_final_stage": is_final_stage,
        })
        if self.raise_on_call is not None:
            raise self.raise_on_call
        if self.output_factory is not None:
            output = self.output_factory(activation)
        else:
            # Default: identity transform — same shape + dtype as input.
            output = activation.copy()
        return LayerSliceResult(
            output=output,
            duration_seconds=self.duration,
            tee_attestation=self.tee_attestation,
            tee_type=self.tee_type,
            epsilon_spent=self.epsilon,
        )


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def stage_identity():
    return generate_node_identity("alice-stage")


@pytest.fixture
def settler_identity():
    return generate_node_identity("settler")


@pytest.fixture
def anchor(stage_identity, settler_identity):
    a = FakeAnchor()
    _register(a, stage_identity)
    _register(a, settler_identity)
    return a


@pytest.fixture
def registry():
    return FakeRegistry(
        models={"test-model": FakeModel.linear_chain("test-model", 4)}
    )


@pytest.fixture
def tee_runtime():
    return FakeTEERuntime(tee_type=TEEType.SOFTWARE)


@pytest.fixture
def runner():
    return FakeRunner()


@pytest.fixture
def server(stage_identity, registry, runner, tee_runtime, anchor):
    return LayerStageServer(
        identity=stage_identity,
        registry=registry,
        runner=runner,
        tee_runtime=tee_runtime,
        anchor=anchor,
        clock=lambda: 1000.0,
    )


def _make_request(
    *,
    settler_identity,
    request_id: str = "req-1",
    model_id: str = "test-model",
    layer_range: Tuple[int, int] = (0, 4),
    privacy_tier: PrivacyLevel = PrivacyLevel.NONE,
    deadline: float = 2000.0,
    activation: Optional[np.ndarray] = None,
    chain_stage_index: int = 0,
    chain_total: int = 2,
) -> RunLayerSliceRequest:
    if activation is None:
        activation = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    blob = activation.tobytes()
    token = HandoffToken.sign(
        identity=settler_identity,
        request_id=request_id,
        chain_stage_index=chain_stage_index,
        chain_total_stages=chain_total,
        deadline_unix=deadline,
    )
    return RunLayerSliceRequest(
        request_id=request_id,
        model_id=model_id,
        layer_range=layer_range,
        privacy_tier=privacy_tier,
        content_tier=ContentTier.A,
        activation_blob=blob,
        activation_shape=tuple(activation.shape),
        activation_dtype=str(activation.dtype),
        upstream_token=token,
        deadline_unix=deadline,
    )


# ──────────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────────


class TestConstruction:
    def _stub_args(self, stage_identity, registry, runner, tee_runtime, anchor):
        return dict(
            identity=stage_identity,
            registry=registry,
            runner=runner,
            tee_runtime=tee_runtime,
            anchor=anchor,
        )

    def test_rejects_missing_identity(
        self, registry, runner, tee_runtime, anchor
    ):
        with pytest.raises(RuntimeError, match="NodeIdentity"):
            LayerStageServer(
                identity=None,  # type: ignore[arg-type]
                registry=registry,
                runner=runner,
                tee_runtime=tee_runtime,
                anchor=anchor,
            )

    def test_rejects_missing_registry(
        self, stage_identity, runner, tee_runtime, anchor
    ):
        with pytest.raises(RuntimeError, match="ModelRegistry"):
            LayerStageServer(
                identity=stage_identity,
                registry=None,  # type: ignore[arg-type]
                runner=runner,
                tee_runtime=tee_runtime,
                anchor=anchor,
            )

    def test_rejects_missing_runner(
        self, stage_identity, registry, tee_runtime, anchor
    ):
        with pytest.raises(RuntimeError, match="LayerSliceRunner"):
            LayerStageServer(
                identity=stage_identity,
                registry=registry,
                runner=None,  # type: ignore[arg-type]
                tee_runtime=tee_runtime,
                anchor=anchor,
            )

    def test_rejects_missing_tee_runtime(
        self, stage_identity, registry, runner, anchor
    ):
        with pytest.raises(RuntimeError, match="tee_runtime"):
            LayerStageServer(
                identity=stage_identity,
                registry=registry,
                runner=runner,
                tee_runtime=None,  # type: ignore[arg-type]
                anchor=anchor,
            )

    def test_rejects_missing_anchor(
        self, stage_identity, registry, runner, tee_runtime
    ):
        with pytest.raises(RuntimeError, match="anchor"):
            LayerStageServer(
                identity=stage_identity,
                registry=registry,
                runner=runner,
                tee_runtime=tee_runtime,
                anchor=None,
            )


# ──────────────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────────────


class TestHappyPath:
    def test_returns_signed_response(
        self, server, settler_identity, anchor, stage_identity
    ):
        request = _make_request(settler_identity=settler_identity)
        response_bytes = server.handle(encode_message(request))
        response = parse_message(response_bytes)

        assert isinstance(response, RunLayerSliceResponse)
        assert response.request_id == "req-1"
        # Stage-signed response verifies under the EXPECTED stage's
        # pubkey (caller supplies the dispatched node_id explicitly to
        # close the substitution-attack hole — see Task 8 H2 finding).
        assert response.verify_with_anchor(
            anchor, expected_stage_node_id=stage_identity.node_id
        ) is True

    def test_runner_called_with_correct_inputs(
        self, server, runner, settler_identity
    ):
        request = _make_request(settler_identity=settler_identity)
        server.handle(encode_message(request))

        assert len(runner.calls) == 1
        call = runner.calls[0]
        assert call["layer_range"] == (0, 4)
        assert call["privacy_tier"] == PrivacyLevel.NONE
        assert call["activation_shape"] == (1, 4)
        assert call["activation_dtype"] == "float32"
        assert call["is_final_stage"] is True

    def test_response_activation_round_trips(
        self, server, runner, settler_identity
    ):
        # Runner doubles every value.
        runner.output_factory = lambda act: act * 2
        original = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        request = _make_request(
            settler_identity=settler_identity, activation=original
        )

        response = parse_message(server.handle(encode_message(request)))
        recovered = np.frombuffer(
            response.activation_blob, dtype=response.activation_dtype
        ).reshape(response.activation_shape)
        np.testing.assert_array_equal(recovered, original * 2)


# ──────────────────────────────────────────────────────────────────────────
# Error paths — one test per StageErrorCode
# ──────────────────────────────────────────────────────────────────────────


class TestErrorPaths:
    def _decode_error(self, response_bytes: bytes) -> StageError:
        msg = parse_message(response_bytes)
        assert isinstance(msg, StageError), (
            f"expected StageError, got {type(msg).__name__}"
        )
        return msg

    # MALFORMED_REQUEST -----------------------------------------------------

    def test_malformed_json(self, server):
        err = self._decode_error(server.handle(b"not-json"))
        assert err.code == StageErrorCode.MALFORMED_REQUEST.value

    def test_unknown_message_type(self, server):
        payload = json.dumps({
            "type": "made_up_type",
            "protocol_version": 1,
        }).encode("utf-8")
        err = self._decode_error(server.handle(payload))
        assert err.code == StageErrorCode.MALFORMED_REQUEST.value

    def test_oversize_payload(self, server):
        from prsm.compute.chain_rpc.protocol import MAX_HANDSHAKE_BYTES
        big = b"x" * (MAX_HANDSHAKE_BYTES + 1)
        err = self._decode_error(server.handle(big))
        assert err.code == StageErrorCode.MALFORMED_REQUEST.value

    def test_response_message_type_rejected(self, server, stage_identity):
        # A peer that sends us a Response instead of a Request should
        # be told to send Requests.
        response = RunLayerSliceResponse.sign(
            identity=stage_identity,
            request_id="r",
            activation_blob=b"x",
            activation_shape=(1,),
            activation_dtype="float32",
            duration_seconds=0.01,
            tee_attestation=b"\x01" * 16,
            tee_type=TEEType.SOFTWARE,
            epsilon_spent=0.0,
        )
        err = self._decode_error(server.handle(encode_message(response)))
        assert err.code == StageErrorCode.MALFORMED_REQUEST.value
        assert "RunLayerSliceRequest" in err.message

    # UNSUPPORTED_VERSION ---------------------------------------------------

    def test_version_mismatch(self, server):
        payload = json.dumps({
            "type": ChainRpcMessageType.RUN_LAYER_SLICE_REQUEST.value,
            "protocol_version": CHAIN_RPC_PROTOCOL_VERSION + 99,
        }).encode("utf-8")
        err = self._decode_error(server.handle(payload))
        assert err.code == StageErrorCode.UNSUPPORTED_VERSION.value

    # INVALID_TOKEN ---------------------------------------------------------

    def test_token_signed_by_unregistered_settler(
        self, server, registry
    ):
        # Mint a token from a settler the anchor does not recognize.
        rogue = generate_node_identity("rogue")
        request = _make_request(settler_identity=rogue)
        err = self._decode_error(server.handle(encode_message(request)))
        assert err.code == StageErrorCode.INVALID_TOKEN.value

    def test_token_with_wrong_signature(self, server, settler_identity):
        request = _make_request(settler_identity=settler_identity)
        # Replace the token's signature with a corrupted one.
        bad_token = HandoffToken(
            request_id=request.upstream_token.request_id,
            settler_node_id=request.upstream_token.settler_node_id,
            chain_stage_index=request.upstream_token.chain_stage_index,
            chain_total_stages=request.upstream_token.chain_total_stages,
            deadline_unix=request.upstream_token.deadline_unix,
            signature_b64="A" * len(request.upstream_token.signature_b64),
        )
        bad_request = RunLayerSliceRequest(
            request_id=request.request_id,
            model_id=request.model_id,
            layer_range=request.layer_range,
            privacy_tier=request.privacy_tier,
            content_tier=request.content_tier,
            activation_blob=request.activation_blob,
            activation_shape=request.activation_shape,
            activation_dtype=request.activation_dtype,
            upstream_token=bad_token,
            deadline_unix=request.deadline_unix,
        )
        err = self._decode_error(server.handle(encode_message(bad_request)))
        assert err.code == StageErrorCode.INVALID_TOKEN.value

    # DEADLINE_EXCEEDED -----------------------------------------------------

    def test_token_deadline_in_past(self, server, settler_identity):
        # Server clock is 1000.0; token deadline 500.0.
        request = _make_request(
            settler_identity=settler_identity,
            deadline=500.0,
        )
        err = self._decode_error(server.handle(encode_message(request)))
        assert err.code == StageErrorCode.DEADLINE_EXCEEDED.value

    # MODEL_NOT_FOUND -------------------------------------------------------

    def test_unknown_model(self, server, settler_identity):
        request = _make_request(
            settler_identity=settler_identity, model_id="ghost-model"
        )
        err = self._decode_error(server.handle(encode_message(request)))
        assert err.code == StageErrorCode.MODEL_NOT_FOUND.value

    def test_registry_verification_failure(
        self, stage_identity, runner, tee_runtime, anchor, settler_identity
    ):
        registry = FakeRegistry(raise_on_get=_ManifestVerificationError("bad sig"))
        server = LayerStageServer(
            identity=stage_identity,
            registry=registry,
            runner=runner,
            tee_runtime=tee_runtime,
            anchor=anchor,
            clock=lambda: 1000.0,
        )
        request = _make_request(settler_identity=settler_identity)
        msg = parse_message(server.handle(encode_message(request)))
        assert isinstance(msg, StageError)
        assert msg.code == StageErrorCode.MODEL_NOT_FOUND.value

    # SHARD_MISSING ---------------------------------------------------------

    def test_shard_does_not_cover_range(
        self, stage_identity, runner, tee_runtime, anchor, settler_identity
    ):
        # Local model only has shard for layers 0..2; request asks
        # for 0..4.
        registry = FakeRegistry(models={
            "test-model": FakeModel(
                model_id="test-model",
                shards=[FakeShard(layer_range=(0, 2))],
            )
        })
        server = LayerStageServer(
            identity=stage_identity,
            registry=registry,
            runner=runner,
            tee_runtime=tee_runtime,
            anchor=anchor,
            clock=lambda: 1000.0,
        )
        request = _make_request(
            settler_identity=settler_identity,
            layer_range=(0, 4),
        )
        msg = parse_message(server.handle(encode_message(request)))
        assert isinstance(msg, StageError)
        assert msg.code == StageErrorCode.SHARD_MISSING.value

    def test_shard_with_zero_zero_sentinel_treated_as_full_coverage(
        self, stage_identity, runner, tee_runtime, anchor, settler_identity
    ):
        # Phase 3.x.2 registry sentinel: every shard has (0, 0).
        # Server treats this as full-coverage fallback (single-stage
        # back-compat).
        registry = FakeRegistry(models={
            "test-model": FakeModel(
                model_id="test-model",
                shards=[FakeShard(layer_range=(0, 0))],
            )
        })
        server = LayerStageServer(
            identity=stage_identity,
            registry=registry,
            runner=runner,
            tee_runtime=tee_runtime,
            anchor=anchor,
            clock=lambda: 1000.0,
        )
        request = _make_request(
            settler_identity=settler_identity,
            layer_range=(0, 4),
        )
        msg = parse_message(server.handle(encode_message(request)))
        assert isinstance(msg, RunLayerSliceResponse)

    # TIER_GATE -------------------------------------------------------------

    def test_software_tee_rejects_high_privacy(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity
    ):
        # Server's tee_runtime is SOFTWARE; HIGH privacy demands hardware.
        server = LayerStageServer(
            identity=stage_identity,
            registry=registry,
            runner=runner,
            tee_runtime=tee_runtime,
            anchor=anchor,
            clock=lambda: 1000.0,
        )
        request = _make_request(
            settler_identity=settler_identity,
            privacy_tier=PrivacyLevel.HIGH,
        )
        msg = parse_message(server.handle(encode_message(request)))
        assert isinstance(msg, StageError)
        assert msg.code == StageErrorCode.TIER_GATE.value

    def test_hardware_tee_accepts_high_privacy(
        self, stage_identity, registry, runner, anchor, settler_identity
    ):
        hardware_tee = FakeTEERuntime(tee_type=TEEType.SGX)
        server = LayerStageServer(
            identity=stage_identity,
            registry=registry,
            runner=runner,
            tee_runtime=hardware_tee,
            anchor=anchor,
            clock=lambda: 1000.0,
        )
        request = _make_request(
            settler_identity=settler_identity,
            privacy_tier=PrivacyLevel.HIGH,
        )
        msg = parse_message(server.handle(encode_message(request)))
        assert isinstance(msg, RunLayerSliceResponse)

    # Sprint 1084 — advisory bypass of a confidential request must be LOUD ---

    def test_advisory_software_confidential_warns_and_allows(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity, monkeypatch, caplog
    ):
        """advisory + SOFTWARE runtime + HIGH privacy: the request is allowed (advisory's
        purpose) but a loud SECURITY warning fires so it can't run silently."""
        import logging
        monkeypatch.setenv("PRSM_PARALLAX_TIER_GATE", "advisory")
        server = LayerStageServer(
            identity=stage_identity, registry=registry, runner=runner,
            tee_runtime=tee_runtime, anchor=anchor, clock=lambda: 1000.0,
        )
        request = _make_request(
            settler_identity=settler_identity, privacy_tier=PrivacyLevel.HIGH)
        with caplog.at_level(logging.WARNING):
            msg = parse_message(server.handle(encode_message(request)))
        assert isinstance(msg, RunLayerSliceResponse)   # allowed, not TIER_GATE
        assert any("advisory" in r.message.lower() and "SECURITY" in r.message
                   for r in caplog.records)

    def test_advisory_software_none_privacy_no_warning(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity, monkeypatch, caplog
    ):
        """privacy_tier=NONE is never gated → advisory must not emit the confidential
        bypass warning for ordinary non-confidential traffic."""
        import logging
        monkeypatch.setenv("PRSM_PARALLAX_TIER_GATE", "advisory")
        server = LayerStageServer(
            identity=stage_identity, registry=registry, runner=runner,
            tee_runtime=tee_runtime, anchor=anchor, clock=lambda: 1000.0,
        )
        request = _make_request(
            settler_identity=settler_identity, privacy_tier=PrivacyLevel.NONE)
        with caplog.at_level(logging.WARNING):
            server.handle(encode_message(request))
        assert not any("attestation gate is DEFEATED" in r.message
                       for r in caplog.records)

    # ACTIVATION_INVALID ----------------------------------------------------

    def test_activation_size_mismatch(self, server, settler_identity):
        # Build request with mismatched shape + blob size.
        token = HandoffToken.sign(
            identity=settler_identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=2,
            deadline_unix=2000.0,
        )
        # Shape says (1, 8) (32 bytes float32) but blob is 16 bytes.
        bad_blob = b"\x00" * 16
        request = RunLayerSliceRequest(
            request_id="req-1",
            model_id="test-model",
            layer_range=(0, 4),
            privacy_tier=PrivacyLevel.NONE,
            content_tier=ContentTier.A,
            activation_blob=bad_blob,
            activation_shape=(1, 8),
            activation_dtype="float32",
            upstream_token=token,
            deadline_unix=2000.0,
        )
        msg = parse_message(server.handle(encode_message(request)))
        assert isinstance(msg, StageError)
        assert msg.code == StageErrorCode.ACTIVATION_INVALID.value

    def test_unsupported_dtype(self, server, settler_identity):
        token = HandoffToken.sign(
            identity=settler_identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=2,
            deadline_unix=2000.0,
        )
        request = RunLayerSliceRequest(
            request_id="req-1",
            model_id="test-model",
            layer_range=(0, 4),
            privacy_tier=PrivacyLevel.NONE,
            content_tier=ContentTier.A,
            activation_blob=b"\x00" * 16,
            activation_shape=(1, 4),
            activation_dtype="not-a-real-dtype",
            upstream_token=token,
            deadline_unix=2000.0,
        )
        msg = parse_message(server.handle(encode_message(request)))
        assert isinstance(msg, StageError)
        assert msg.code == StageErrorCode.ACTIVATION_INVALID.value

    # TIMEOUT ---------------------------------------------------------------

    def test_runner_timeout(
        self, stage_identity, registry, tee_runtime, anchor, settler_identity
    ):
        runner = FakeRunner(raise_on_call=TimeoutError("layer hung"))
        server = LayerStageServer(
            identity=stage_identity,
            registry=registry,
            runner=runner,
            tee_runtime=tee_runtime,
            anchor=anchor,
            clock=lambda: 1000.0,
        )
        request = _make_request(settler_identity=settler_identity)
        msg = parse_message(server.handle(encode_message(request)))
        assert isinstance(msg, StageError)
        assert msg.code == StageErrorCode.TIMEOUT.value

    # INTERNAL_ERROR --------------------------------------------------------

    def test_runner_unexpected_exception(
        self, stage_identity, registry, tee_runtime, anchor, settler_identity
    ):
        runner = FakeRunner(raise_on_call=ValueError("oops"))
        server = LayerStageServer(
            identity=stage_identity,
            registry=registry,
            runner=runner,
            tee_runtime=tee_runtime,
            anchor=anchor,
            clock=lambda: 1000.0,
        )
        request = _make_request(settler_identity=settler_identity)
        msg = parse_message(server.handle(encode_message(request)))
        assert isinstance(msg, StageError)
        assert msg.code == StageErrorCode.INTERNAL_ERROR.value

    def test_registry_unexpected_exception(
        self, stage_identity, runner, tee_runtime, anchor, settler_identity
    ):
        registry = FakeRegistry(raise_on_get=RuntimeError("disk full"))
        server = LayerStageServer(
            identity=stage_identity,
            registry=registry,
            runner=runner,
            tee_runtime=tee_runtime,
            anchor=anchor,
            clock=lambda: 1000.0,
        )
        request = _make_request(settler_identity=settler_identity)
        msg = parse_message(server.handle(encode_message(request)))
        assert isinstance(msg, StageError)
        assert msg.code == StageErrorCode.INTERNAL_ERROR.value


# ──────────────────────────────────────────────────────────────────────────
# Never-raises invariant — fuzz garbage inputs
# ──────────────────────────────────────────────────────────────────────────


class TestNeverRaises:
    @pytest.mark.parametrize("garbage", [
        b"",
        b"\x00\x01\x02",
        b"null",
        b"123",
        b'"a string"',
        b"[1, 2, 3]",
        b"{}",
        b'{"type": null}',
        b'{"type": 42}',
        b'{"type": "run_layer_slice_request", "protocol_version": "not-int"}',
        b'{"type": "stage_error", "protocol_version": 1}',  # missing fields
    ])
    def test_handle_never_raises(self, server, garbage):
        # Server must return bytes for every input; no exception leaks.
        result = server.handle(garbage)
        assert isinstance(result, bytes)
        # Returned bytes must parse as a valid StageError.
        parsed = parse_message(result)
        assert isinstance(parsed, StageError)


# ──────────────────────────────────────────────────────────────────────────
# Final-stage detection
# ──────────────────────────────────────────────────────────────────────────


class TestFinalStageDetection:
    def test_intermediate_stage_marked_not_final(
        self, stage_identity, runner, tee_runtime, anchor, settler_identity
    ):
        # Model has 8 layers; this stage covers 0..4 (not the tail).
        registry = FakeRegistry(models={
            "test-model": FakeModel(
                model_id="test-model",
                shards=[
                    FakeShard(layer_range=(0, 4)),
                    FakeShard(layer_range=(4, 8)),
                ],
            )
        })
        server = LayerStageServer(
            identity=stage_identity,
            registry=registry,
            runner=runner,
            tee_runtime=tee_runtime,
            anchor=anchor,
            clock=lambda: 1000.0,
        )
        request = _make_request(
            settler_identity=settler_identity,
            layer_range=(0, 4),
        )
        server.handle(encode_message(request))
        assert runner.calls[0]["is_final_stage"] is False

    def test_tail_stage_marked_final(
        self, stage_identity, runner, tee_runtime, anchor, settler_identity
    ):
        registry = FakeRegistry(models={
            "test-model": FakeModel(
                model_id="test-model",
                shards=[
                    FakeShard(layer_range=(0, 4)),
                    FakeShard(layer_range=(4, 8)),
                ],
            )
        })
        server = LayerStageServer(
            identity=stage_identity,
            registry=registry,
            runner=runner,
            tee_runtime=tee_runtime,
            anchor=anchor,
            clock=lambda: 1000.0,
        )
        request = _make_request(
            settler_identity=settler_identity,
            layer_range=(4, 8),
        )
        server.handle(encode_message(request))
        assert runner.calls[0]["is_final_stage"] is True


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.7.1 v2 streamed-path tests
# ──────────────────────────────────────────────────────────────────────────


import hashlib as _hashlib

from prsm.compute.chain_rpc.activation_codec import (
    ChunkedActivation as _CA,
    chunk_activation as _chunk_activation,
    reassemble_chunked as _reassemble,
)
from prsm.compute.chain_rpc.protocol import ActivationChunk as _ActivationChunk
from prsm.node.shard_streaming import (
    ShardChunk as _ShardChunk,
    ShardManifest,
)


def _make_streamed_request(
    *,
    settler_identity,
    activation: Optional[np.ndarray] = None,
    request_id: str = "req-1",
    model_id: str = "test-model",
    layer_range: Tuple[int, int] = (0, 4),
    privacy_tier: PrivacyLevel = PrivacyLevel.NONE,
    deadline: float = 2000.0,
    chunk_bytes: int = 64,
):
    """Build a streamed RunLayerSliceRequest manifest + ActivationChunk
    frames suitable for handle_streamed."""
    if activation is None:
        activation = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    chunked = _chunk_activation(
        activation,
        activation_id=f"{request_id}::stage-0::out-from-prev",
        chunk_bytes=chunk_bytes,
    )
    token = HandoffToken.sign(
        identity=settler_identity,
        request_id=request_id,
        chain_stage_index=0,
        chain_total_stages=2,
        deadline_unix=deadline,
    )
    request = RunLayerSliceRequest(
        request_id=request_id,
        model_id=model_id,
        layer_range=layer_range,
        privacy_tier=privacy_tier,
        content_tier=ContentTier.A,
        activation_blob=b"",
        activation_shape=chunked.shape,
        activation_dtype=chunked.dtype_str,
        upstream_token=token,
        deadline_unix=deadline,
        activation_manifest=chunked.manifest,
    )
    chunk_frames = [
        encode_message(_ActivationChunk(
            request_id=request_id,
            sequence=c.sequence,
            data=c.data,
            chunk_sha256=c.chunk_sha256,
        ))
        for c in chunked.chunks
    ]
    return encode_message(request), chunk_frames, activation


def _decode_streamed_response(
    response_manifest_bytes: bytes,
    response_chunk_iter: Iterable[bytes],
):
    """Mirror of the client-side reassembly: parse manifest, parse
    chunks, reassemble via ShardAssembler. Returns the recovered
    np.ndarray plus the parsed RunLayerSliceResponse."""
    response = parse_message(response_manifest_bytes)
    if isinstance(response, StageError):
        return None, response
    assert isinstance(response, RunLayerSliceResponse)
    assert response.activation_manifest is not None
    shard_chunks = []
    for raw in response_chunk_iter:
        msg = parse_message(raw)
        assert isinstance(msg, _ActivationChunk)
        shard_chunks.append(_ShardChunk(
            shard_id=response.activation_manifest.shard_id,
            sequence=msg.sequence,
            data=msg.data,
            chunk_sha256=msg.chunk_sha256,
        ))
    chunked = _CA(
        manifest=response.activation_manifest,
        chunks=shard_chunks,
        shape=response.activation_shape,
        dtype_str=response.activation_dtype,
    )
    activation = _reassemble(chunked, chunks=shard_chunks)
    return activation, response


# ──────────────────────────────────────────────────────────────────────────
# Imports for the v2 tests
# ──────────────────────────────────────────────────────────────────────────

from typing import Iterable


class TestStreamedHappyPath:
    def test_streamed_round_trip_returns_signed_response(
        self, server, settler_identity, anchor, stage_identity, runner
    ):
        manifest_bytes, chunk_frames, original_activation = (
            _make_streamed_request(settler_identity=settler_identity)
        )

        response_manifest_bytes, response_chunks_iter = server.handle_streamed(
            manifest_bytes, iter(chunk_frames)
        )
        recovered_activation, response = _decode_streamed_response(
            response_manifest_bytes, response_chunks_iter
        )

        assert response.request_id == "req-1"
        # Default identity-transform runner means recovered activation
        # equals the original.
        np.testing.assert_array_equal(recovered_activation, original_activation)
        # Stage-signed response verifies under the dispatched identity.
        assert response.verify_with_anchor(
            anchor, expected_stage_node_id=stage_identity.node_id
        ) is True

    def test_streamed_runner_called_with_correct_inputs(
        self, server, settler_identity, runner
    ):
        manifest_bytes, chunk_frames, _ = _make_streamed_request(
            settler_identity=settler_identity
        )
        server.handle_streamed(manifest_bytes, iter(chunk_frames))

        assert len(runner.calls) == 1
        call = runner.calls[0]
        assert call["layer_range"] == (0, 4)
        assert call["privacy_tier"] == PrivacyLevel.NONE
        assert call["activation_shape"] == (1, 4)


class TestStreamedValidationGates:
    """All gates from steps 2-6 must fire on the streamed path AND
    BEFORE chunk consumption (so a peer that blasts garbage chunks at
    a server doesn't waste assembly cost)."""

    def _decode_error(self, response_bytes: bytes) -> StageError:
        msg = parse_message(response_bytes)
        assert isinstance(msg, StageError), (
            f"expected StageError, got {type(msg).__name__}"
        )
        return msg

    def test_invalid_token_rejected_before_chunks_consumed(
        self, server, runner
    ):
        # Token signed by a settler NOT registered on the anchor.
        rogue = generate_node_identity("rogue")
        manifest_bytes, chunk_frames, _ = _make_streamed_request(
            settler_identity=rogue,
        )

        # Pass a generator that records consumption.
        consumed = []

        def chunk_gen():
            for f in chunk_frames:
                consumed.append(f)
                yield f

        manifest_resp, _ = server.handle_streamed(
            manifest_bytes, chunk_gen()
        )
        err = self._decode_error(manifest_resp)
        assert err.code == StageErrorCode.INVALID_TOKEN.value
        # No chunks consumed — token failure short-circuited.
        assert consumed == []
        # No layer execution.
        assert len(runner.calls) == 0

    def test_deadline_exceeded_rejected_before_chunks(
        self, server, settler_identity, runner
    ):
        manifest_bytes, chunk_frames, _ = _make_streamed_request(
            settler_identity=settler_identity,
            deadline=500.0,  # server clock is 1000.0
        )

        consumed = []

        def chunk_gen():
            for f in chunk_frames:
                consumed.append(f)
                yield f

        manifest_resp, _ = server.handle_streamed(
            manifest_bytes, chunk_gen()
        )
        err = self._decode_error(manifest_resp)
        assert err.code == StageErrorCode.DEADLINE_EXCEEDED.value
        assert consumed == []

    def test_unknown_model_rejected_before_chunks(
        self, server, settler_identity, runner
    ):
        manifest_bytes, chunk_frames, _ = _make_streamed_request(
            settler_identity=settler_identity,
            model_id="ghost",
        )
        consumed = []

        def chunk_gen():
            for f in chunk_frames:
                consumed.append(f)
                yield f

        manifest_resp, _ = server.handle_streamed(
            manifest_bytes, chunk_gen()
        )
        err = self._decode_error(manifest_resp)
        assert err.code == StageErrorCode.MODEL_NOT_FOUND.value
        assert consumed == []

    def test_tier_gate_rejects_high_privacy_on_software_tee(
        self, server, settler_identity,
    ):
        manifest_bytes, chunk_frames, _ = _make_streamed_request(
            settler_identity=settler_identity,
            privacy_tier=PrivacyLevel.HIGH,
        )
        manifest_resp, _ = server.handle_streamed(
            manifest_bytes, iter(chunk_frames)
        )
        msg = parse_message(manifest_resp)
        assert isinstance(msg, StageError)
        assert msg.code == StageErrorCode.TIER_GATE.value


class TestStreamedChunkValidation:
    """Errors during chunk reassembly map to ACTIVATION_INVALID."""

    def _decode_error(self, response_bytes: bytes) -> StageError:
        msg = parse_message(response_bytes)
        assert isinstance(msg, StageError)
        return msg

    def test_corrupted_chunk_rejected(self, server, settler_identity):
        manifest_bytes, chunk_frames, _ = _make_streamed_request(
            settler_identity=settler_identity,
            activation=np.arange(64, dtype=np.float32),
            chunk_bytes=64,
        )
        # Corrupt one chunk's data without updating its sha.
        bad_chunk = parse_message(chunk_frames[0])
        assert isinstance(bad_chunk, _ActivationChunk)
        forged = _ActivationChunk(
            request_id=bad_chunk.request_id,
            sequence=bad_chunk.sequence,
            data=b"\x00" * len(bad_chunk.data),
            chunk_sha256=bad_chunk.chunk_sha256,  # original — won't match
        )
        chunk_frames[0] = encode_message(forged)
        manifest_resp, _ = server.handle_streamed(
            manifest_bytes, iter(chunk_frames)
        )
        err = self._decode_error(manifest_resp)
        assert err.code == StageErrorCode.ACTIVATION_INVALID.value

    def test_chunk_request_id_mismatch_rejected(
        self, server, settler_identity
    ):
        manifest_bytes, chunk_frames, _ = _make_streamed_request(
            settler_identity=settler_identity,
        )
        # Splice a chunk that claims a different parent request.
        original = parse_message(chunk_frames[0])
        spliced = _ActivationChunk(
            request_id="DIFFERENT-REQUEST",
            sequence=original.sequence,
            data=original.data,
            chunk_sha256=original.chunk_sha256,
        )
        chunk_frames[0] = encode_message(spliced)
        manifest_resp, _ = server.handle_streamed(
            manifest_bytes, iter(chunk_frames)
        )
        err = self._decode_error(manifest_resp)
        assert err.code == StageErrorCode.ACTIVATION_INVALID.value

    def test_garbage_chunk_frame_rejected(self, server, settler_identity):
        manifest_bytes, chunk_frames, _ = _make_streamed_request(
            settler_identity=settler_identity,
        )
        chunk_frames[0] = b"not-json"
        manifest_resp, _ = server.handle_streamed(
            manifest_bytes, iter(chunk_frames)
        )
        err = self._decode_error(manifest_resp)
        assert err.code == StageErrorCode.ACTIVATION_INVALID.value

    def test_non_chunk_message_in_chunk_iter_rejected(
        self, server, settler_identity
    ):
        manifest_bytes, chunk_frames, _ = _make_streamed_request(
            settler_identity=settler_identity,
        )
        # Send a StageError as a "chunk" — wrong type.
        chunk_frames[0] = encode_message(
            StageError(request_id="req-1", code="X", message="")
        )
        manifest_resp, _ = server.handle_streamed(
            manifest_bytes, iter(chunk_frames)
        )
        err = self._decode_error(manifest_resp)
        assert err.code == StageErrorCode.ACTIVATION_INVALID.value


class TestStreamedManifestRouting:
    """The streamed handler rejects requests without an
    activation_manifest (caller should use handle() for inline)."""

    def test_inline_request_to_streamed_handler_rejected(
        self, server, settler_identity
    ):
        # Build an inline-formatted RunLayerSliceRequest (no manifest)
        # and try to dispatch it via handle_streamed.
        inline_request = _make_request(settler_identity=settler_identity)
        manifest_resp, _ = server.handle_streamed(
            encode_message(inline_request), iter([])
        )
        msg = parse_message(manifest_resp)
        assert isinstance(msg, StageError)
        assert msg.code == StageErrorCode.MALFORMED_REQUEST.value
        assert "lacks activation_manifest" in msg.message

    def test_streamed_request_to_inline_handler_rejected(
        self, server, settler_identity
    ):
        # Inverse: a streamed request reaching the inline handle()
        # should also be rejected with MALFORMED_REQUEST.
        manifest_bytes, _, _ = _make_streamed_request(
            settler_identity=settler_identity
        )
        msg = parse_message(server.handle(manifest_bytes))
        assert isinstance(msg, StageError)
        assert msg.code == StageErrorCode.MALFORMED_REQUEST.value
        assert "carries activation_manifest" in msg.message


class TestStreamedNeverRaises:
    """handle_streamed must never propagate exceptions through the
    streaming transport boundary. Garbage manifests + chunks both
    map to encoded StageError responses with empty chunk iter."""

    @pytest.mark.parametrize("garbage_manifest", [
        b"",
        b"not-json",
        b'{"type": "stage_error", "protocol_version": 1}',
        b'{"type": null}',
    ])
    def test_handle_streamed_garbage_manifest_never_raises(
        self, server, garbage_manifest
    ):
        manifest_resp, chunks = server.handle_streamed(
            garbage_manifest, iter([])
        )
        assert isinstance(manifest_resp, bytes)
        # The returned manifest must parse as a StageError.
        parsed = parse_message(manifest_resp)
        assert isinstance(parsed, StageError)
        # Empty chunk iter on failure path.
        assert list(chunks) == []

    def test_handle_streamed_runner_exception_maps_to_internal_error(
        self, stage_identity, registry, tee_runtime, anchor, settler_identity
    ):
        bad_runner = FakeRunner(raise_on_call=ValueError("oops"))
        server = LayerStageServer(
            identity=stage_identity,
            registry=registry,
            runner=bad_runner,
            tee_runtime=tee_runtime,
            anchor=anchor,
            clock=lambda: 1000.0,
        )
        manifest_bytes, chunk_frames, _ = _make_streamed_request(
            settler_identity=settler_identity,
        )
        manifest_resp, chunks = server.handle_streamed(
            manifest_bytes, iter(chunk_frames)
        )
        msg = parse_message(manifest_resp)
        assert isinstance(msg, StageError)
        assert msg.code == StageErrorCode.INTERNAL_ERROR.value
        assert list(chunks) == []


class TestStreamedConstructionValidation:
    def test_non_positive_chunk_bytes_rejected(
        self, stage_identity, registry, runner, tee_runtime, anchor
    ):
        with pytest.raises(ValueError, match="chunk_bytes"):
            LayerStageServer(
                identity=stage_identity,
                registry=registry,
                runner=runner,
                tee_runtime=tee_runtime,
                anchor=anchor,
                chunk_bytes=0,
            )

    def test_non_positive_max_streamed_payload_bytes_rejected(
        self, stage_identity, registry, runner, tee_runtime, anchor
    ):
        with pytest.raises(ValueError, match="max_streamed_payload_bytes"):
            LayerStageServer(
                identity=stage_identity,
                registry=registry,
                runner=runner,
                tee_runtime=tee_runtime,
                anchor=anchor,
                max_streamed_payload_bytes=0,
            )


class TestStreamedEnvelopeValidation:
    """H1 + M1 round-1 (Phase 3.x.7.1 Task 6): pre-consumption envelope
    sanity checks fire BEFORE any chunk consumption. Defense against
    network-level adversaries that ship inflated manifest envelopes to
    coerce server-side memory exhaustion during reassembly."""

    def _decode_error(self, response_bytes: bytes) -> StageError:
        msg = parse_message(response_bytes)
        assert isinstance(msg, StageError)
        return msg

    def test_envelope_payload_bytes_lt_shape_dtype_rejected(
        self, server, settler_identity
    ):
        """Manifest claims fewer bytes than shape × dtype implies."""
        activation = np.arange(64, dtype=np.float32)
        manifest_bytes, chunk_frames, _ = _make_streamed_request(
            settler_identity=settler_identity,
            activation=activation,
            chunk_bytes=64,
        )
        # Re-construct manifest with an inflated payload_bytes.
        original = parse_message(manifest_bytes)
        assert isinstance(original, RunLayerSliceRequest)
        bogus_manifest = ShardManifest(
            shard_id=original.activation_manifest.shard_id,
            payload_sha256=original.activation_manifest.payload_sha256,
            payload_bytes=original.activation_manifest.payload_bytes + 8,
            total_chunks=original.activation_manifest.total_chunks,
            chunk_bytes=original.activation_manifest.chunk_bytes,
        )
        forged = RunLayerSliceRequest(
            request_id=original.request_id,
            model_id=original.model_id,
            layer_range=original.layer_range,
            privacy_tier=original.privacy_tier,
            content_tier=original.content_tier,
            activation_blob=b"",
            activation_shape=original.activation_shape,
            activation_dtype=original.activation_dtype,
            upstream_token=original.upstream_token,
            deadline_unix=original.deadline_unix,
            activation_manifest=bogus_manifest,
        )
        manifest_resp, chunks = server.handle_streamed(
            encode_message(forged), iter(chunk_frames)
        )
        err = self._decode_error(manifest_resp)
        assert err.code == StageErrorCode.ACTIVATION_INVALID.value
        assert "payload_bytes" in err.message
        assert list(chunks) == []

    def test_envelope_total_chunks_inconsistent_rejected(
        self, server, settler_identity
    ):
        """Manifest declares a total_chunks that doesn't match
        ceil(payload_bytes / chunk_bytes)."""
        manifest_bytes, chunk_frames, _ = _make_streamed_request(
            settler_identity=settler_identity,
            activation=np.arange(64, dtype=np.float32),
            chunk_bytes=64,
        )
        original = parse_message(manifest_bytes)
        bogus_manifest = ShardManifest(
            shard_id=original.activation_manifest.shard_id,
            payload_sha256=original.activation_manifest.payload_sha256,
            payload_bytes=original.activation_manifest.payload_bytes,
            total_chunks=original.activation_manifest.total_chunks + 5,
            chunk_bytes=original.activation_manifest.chunk_bytes,
        )
        forged = RunLayerSliceRequest(
            request_id=original.request_id,
            model_id=original.model_id,
            layer_range=original.layer_range,
            privacy_tier=original.privacy_tier,
            content_tier=original.content_tier,
            activation_blob=b"",
            activation_shape=original.activation_shape,
            activation_dtype=original.activation_dtype,
            upstream_token=original.upstream_token,
            deadline_unix=original.deadline_unix,
            activation_manifest=bogus_manifest,
        )
        manifest_resp, _ = server.handle_streamed(
            encode_message(forged), iter(chunk_frames)
        )
        err = self._decode_error(manifest_resp)
        assert err.code == StageErrorCode.ACTIVATION_INVALID.value
        assert "total_chunks" in err.message

    def test_envelope_exceeds_max_streamed_payload_bytes_rejected(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity,
    ):
        """payload_bytes above the operator's configured cap."""
        # Configure server with a tight cap.
        capped_server = LayerStageServer(
            identity=stage_identity,
            registry=registry,
            runner=runner,
            tee_runtime=tee_runtime,
            anchor=anchor,
            clock=lambda: 1000.0,
            max_streamed_payload_bytes=100,
        )
        activation = np.arange(64, dtype=np.float32)  # 256 bytes > cap.
        manifest_bytes, chunk_frames, _ = _make_streamed_request(
            settler_identity=settler_identity,
            activation=activation,
            chunk_bytes=64,
        )
        manifest_resp, chunks = capped_server.handle_streamed(
            manifest_bytes, iter(chunk_frames)
        )
        err = self._decode_error(manifest_resp)
        assert err.code == StageErrorCode.ACTIVATION_INVALID.value
        assert "max_streamed_payload_bytes" in err.message
        assert list(chunks) == []

    def test_excess_chunks_after_envelope_rejected(
        self, server, settler_identity
    ):
        """Peer ships more frames than envelope.total_chunks promises —
        H1 round-1 fix bounds reassembly."""
        manifest_bytes, chunk_frames, activation = _make_streamed_request(
            settler_identity=settler_identity,
            activation=np.arange(64, dtype=np.float32),
            chunk_bytes=64,
        )
        # Append a duplicate frame so the iter yields one over the
        # manifest's total_chunks.
        chunk_frames.append(chunk_frames[-1])
        manifest_resp, _ = server.handle_streamed(
            manifest_bytes, iter(chunk_frames)
        )
        err = self._decode_error(manifest_resp)
        assert err.code == StageErrorCode.ACTIVATION_INVALID.value

    def test_envelope_dtype_unrecognized_rejected(
        self, server, settler_identity
    ):
        """A dtype string numpy can't parse — rejected at envelope
        validation, not later."""
        manifest_bytes, chunk_frames, _ = _make_streamed_request(
            settler_identity=settler_identity,
            chunk_bytes=64,
        )
        original = parse_message(manifest_bytes)
        # Build a forged request with a bogus dtype. activation_dtype
        # field validates only that it's a non-empty str — the dtype
        # parse failure surfaces inside envelope validation.
        forged = RunLayerSliceRequest(
            request_id=original.request_id,
            model_id=original.model_id,
            layer_range=original.layer_range,
            privacy_tier=original.privacy_tier,
            content_tier=original.content_tier,
            activation_blob=b"",
            activation_shape=original.activation_shape,
            activation_dtype="nope-not-a-dtype",
            upstream_token=original.upstream_token,
            deadline_unix=original.deadline_unix,
            activation_manifest=original.activation_manifest,
        )
        manifest_resp, _ = server.handle_streamed(
            encode_message(forged), iter(chunk_frames)
        )
        err = self._decode_error(manifest_resp)
        assert err.code == StageErrorCode.ACTIVATION_INVALID.value
        assert "dtype" in err.message


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.8 — Token-stream handler tests
# ──────────────────────────────────────────────────────────────────────────


from prsm.compute.chain_rpc.protocol import (
    StreamFinalFrame,
    TokenFrame,
)
from prsm.compute.inference.streaming_runner import (
    StreamingChunk,
    SyntheticStreamingRunner,
)


def _make_streaming_request(
    *,
    settler_identity,
    request_id: str = "req-stream-1",
    model_id: str = "test-model",
    layer_range: Tuple[int, int] = (0, 4),  # tail by default
    privacy_tier: PrivacyLevel = PrivacyLevel.NONE,
    deadline: float = 2000.0,
    activation: Optional[np.ndarray] = None,
    streaming: bool = True,
) -> RunLayerSliceRequest:
    """Streaming-flavor of _make_request."""
    if activation is None:
        activation = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    blob = activation.tobytes()
    token = HandoffToken.sign(
        identity=settler_identity,
        request_id=request_id,
        chain_stage_index=0,
        chain_total_stages=2,
        deadline_unix=deadline,
    )
    return RunLayerSliceRequest(
        request_id=request_id,
        model_id=model_id,
        layer_range=layer_range,
        privacy_tier=privacy_tier,
        content_tier=ContentTier.A,
        activation_blob=blob,
        activation_shape=tuple(activation.shape),
        activation_dtype=str(activation.dtype),
        upstream_token=token,
        deadline_unix=deadline,
        streaming=streaming,
    )


def _decode_stream_frames(
    iter_bytes: Iterable[bytes],
) -> Tuple[List[TokenFrame], Optional[StreamFinalFrame], Optional[StageError]]:
    """Decode an iterator of wire frames into (token_frames,
    stream_final_frame, error). Exactly one of stream_final_frame /
    error is set when the stream terminates cleanly."""
    tokens: List[TokenFrame] = []
    final: Optional[StreamFinalFrame] = None
    err: Optional[StageError] = None
    for raw in iter_bytes:
        msg = parse_message(raw)
        if isinstance(msg, TokenFrame):
            tokens.append(msg)
        elif isinstance(msg, StreamFinalFrame):
            final = msg
        elif isinstance(msg, StageError):
            err = msg
        else:
            raise AssertionError(
                f"unexpected wire type in stream: {type(msg).__name__}"
            )
    return tokens, final, err


@pytest.fixture
def streaming_runner_for_text():
    """Builds a SyntheticStreamingRunner that decodes activation bytes
    back into a fixed canonical text. Tests can override by passing
    different output_decoder via _make_streaming_server."""
    def output_decoder(activation: np.ndarray) -> str:
        return "hello world from synthetic"

    return output_decoder


def _make_streaming_server(
    *, stage_identity, registry, runner, tee_runtime, anchor,
    output_decoder=None,
    splitter=None,
) -> LayerStageServer:
    """LayerStageServer wired with a SyntheticStreamingRunner."""
    if output_decoder is None:
        def output_decoder(activation: np.ndarray) -> str:  # noqa: E306
            return "hello world from synthetic"
    kwargs = {"runner": runner, "output_decoder": output_decoder}
    if splitter is not None:
        kwargs["splitter"] = splitter
    streaming_runner = SyntheticStreamingRunner(**kwargs)
    return LayerStageServer(
        identity=stage_identity,
        registry=registry,
        runner=runner,
        tee_runtime=tee_runtime,
        anchor=anchor,
        clock=lambda: 1000.0,
        streaming_runner=streaming_runner,
    )


class TestTokenStreamHappyPath:
    def test_tail_stage_streaming_yields_token_frames_then_final(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity,
    ):
        server = _make_streaming_server(
            stage_identity=stage_identity, registry=registry,
            runner=runner, tee_runtime=tee_runtime, anchor=anchor,
        )
        req = _make_streaming_request(settler_identity=settler_identity)
        tokens, final, err = _decode_stream_frames(
            server.handle_token_stream(encode_message(req))
        )
        assert err is None
        assert final is not None
        assert len(tokens) >= 1
        # Joined invariant: concatenating text_deltas matches the
        # signed StreamFinalFrame.response.activation_blob.
        joined = "".join(t.text_delta for t in tokens)
        assert joined == "hello world from synthetic"
        assert final.response.activation_blob == joined.encode("utf-8")

    def test_terminal_token_frame_has_finish_reason_stop(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity,
    ):
        server = _make_streaming_server(
            stage_identity=stage_identity, registry=registry,
            runner=runner, tee_runtime=tee_runtime, anchor=anchor,
        )
        req = _make_streaming_request(settler_identity=settler_identity)
        tokens, final, _ = _decode_stream_frames(
            server.handle_token_stream(encode_message(req))
        )
        # Only the LAST token frame may carry finish_reason.
        for t in tokens[:-1]:
            assert t.finish_reason is None
        assert tokens[-1].finish_reason == "stop"
        assert final is not None

    def test_token_frames_have_strictly_increasing_sequence(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity,
    ):
        server = _make_streaming_server(
            stage_identity=stage_identity, registry=registry,
            runner=runner, tee_runtime=tee_runtime, anchor=anchor,
        )
        req = _make_streaming_request(settler_identity=settler_identity)
        tokens, _, _ = _decode_stream_frames(
            server.handle_token_stream(encode_message(req))
        )
        for i, t in enumerate(tokens):
            assert t.sequence_index == i

    def test_final_frame_response_signature_verifies_under_anchor(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity,
    ):
        server = _make_streaming_server(
            stage_identity=stage_identity, registry=registry,
            runner=runner, tee_runtime=tee_runtime, anchor=anchor,
        )
        req = _make_streaming_request(settler_identity=settler_identity)
        _, final, _ = _decode_stream_frames(
            server.handle_token_stream(encode_message(req))
        )
        assert final is not None
        # The H2 invariant: signature must verify under the EXPECTED
        # stage_node_id (caller-supplied), not the response self-claim.
        assert final.response.verify_with_anchor(
            anchor, expected_stage_node_id=stage_identity.node_id
        ) is True


class TestTokenStreamRouting:
    """Routing invariants: the token-stream handler is for tail
    streaming requests only. Mismatches are MALFORMED_REQUEST."""

    def test_streaming_false_rejected(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity,
    ):
        server = _make_streaming_server(
            stage_identity=stage_identity, registry=registry,
            runner=runner, tee_runtime=tee_runtime, anchor=anchor,
        )
        req = _make_streaming_request(
            settler_identity=settler_identity, streaming=False
        )
        _, final, err = _decode_stream_frames(
            server.handle_token_stream(encode_message(req))
        )
        assert final is None
        assert err is not None
        assert err.code == StageErrorCode.MALFORMED_REQUEST.value
        assert "streaming=True" in err.message

    def test_non_tail_stage_rejected(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity,
    ):
        # FakeModel has shards covering [0, 4); a layer_range of (0, 2)
        # is covered (passes shard gate) but is NOT the tail.
        server = _make_streaming_server(
            stage_identity=stage_identity, registry=registry,
            runner=runner, tee_runtime=tee_runtime, anchor=anchor,
        )
        req = _make_streaming_request(
            settler_identity=settler_identity, layer_range=(0, 2),
        )
        _, final, err = _decode_stream_frames(
            server.handle_token_stream(encode_message(req))
        )
        assert final is None
        assert err is not None
        assert err.code == StageErrorCode.MALFORMED_REQUEST.value
        assert "tail stage" in err.message

    def test_chunked_input_rejected_v1(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity,
    ):
        # v1 doesn't compose streaming-input + streaming-output.
        manifest = ShardManifest(
            shard_id="x",
            payload_sha256="0" * 64,
            payload_bytes=16,
            total_chunks=1,
            chunk_bytes=16,
        )
        token = HandoffToken.sign(
            identity=settler_identity,
            request_id="req-chunked-stream",
            chain_stage_index=0,
            chain_total_stages=2,
            deadline_unix=2000.0,
        )
        req = RunLayerSliceRequest(
            request_id="req-chunked-stream",
            model_id="test-model",
            layer_range=(0, 4),
            privacy_tier=PrivacyLevel.NONE,
            content_tier=ContentTier.A,
            activation_blob=b"",
            activation_shape=(1, 4),
            activation_dtype="float32",
            upstream_token=token,
            deadline_unix=2000.0,
            activation_manifest=manifest,
            streaming=True,
        )
        server = _make_streaming_server(
            stage_identity=stage_identity, registry=registry,
            runner=runner, tee_runtime=tee_runtime, anchor=anchor,
        )
        _, final, err = _decode_stream_frames(
            server.handle_token_stream(encode_message(req))
        )
        assert final is None
        assert err is not None
        assert err.code == StageErrorCode.MALFORMED_REQUEST.value
        assert "chunked-input" in err.message

    def test_no_streaming_runner_rejected_with_internal_error(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity,
    ):
        # Server constructed WITHOUT a streaming_runner — token-stream
        # requests must be rejected as INTERNAL_ERROR (operator
        # misconfig, not a malformed request).
        server = LayerStageServer(
            identity=stage_identity,
            registry=registry,
            runner=runner,
            tee_runtime=tee_runtime,
            anchor=anchor,
            clock=lambda: 1000.0,
            # streaming_runner intentionally not provided.
        )
        req = _make_streaming_request(settler_identity=settler_identity)
        _, _, err = _decode_stream_frames(
            server.handle_token_stream(encode_message(req))
        )
        assert err is not None
        assert err.code == StageErrorCode.INTERNAL_ERROR.value
        assert "streaming_runner" in err.message

    def test_non_request_payload_rejected(
        self, stage_identity, registry, runner, tee_runtime, anchor,
    ):
        # Send a StageError to the streaming handler — must surface
        # as a single MALFORMED_REQUEST frame.
        server = _make_streaming_server(
            stage_identity=stage_identity, registry=registry,
            runner=runner, tee_runtime=tee_runtime, anchor=anchor,
        )
        not_request = encode_message(
            StageError(request_id="r", code="X", message="")
        )
        _, _, err = _decode_stream_frames(
            server.handle_token_stream(not_request)
        )
        assert err is not None
        assert err.code == StageErrorCode.MALFORMED_REQUEST.value


class TestTokenStreamValidationGates:
    """Existing 8-step validation gates fire BEFORE the runner is
    invoked. Each error path emits a SOLE StageError frame."""

    def test_invalid_token_rejected(
        self, stage_identity, registry, runner, tee_runtime, anchor,
    ):
        # Use an unregistered settler — token verify fails at gate 2.
        rogue_settler = generate_node_identity("unregistered-settler")
        server = _make_streaming_server(
            stage_identity=stage_identity, registry=registry,
            runner=runner, tee_runtime=tee_runtime, anchor=anchor,
        )
        req = _make_streaming_request(settler_identity=rogue_settler)
        _, _, err = _decode_stream_frames(
            server.handle_token_stream(encode_message(req))
        )
        assert err is not None
        assert err.code == StageErrorCode.INVALID_TOKEN.value

    def test_deadline_exceeded_rejected(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity,
    ):
        server = _make_streaming_server(
            stage_identity=stage_identity, registry=registry,
            runner=runner, tee_runtime=tee_runtime, anchor=anchor,
        )
        req = _make_streaming_request(
            settler_identity=settler_identity, deadline=500.0,  # past
        )
        _, _, err = _decode_stream_frames(
            server.handle_token_stream(encode_message(req))
        )
        assert err is not None
        assert err.code == StageErrorCode.DEADLINE_EXCEEDED.value

    def test_unknown_model_rejected(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity,
    ):
        server = _make_streaming_server(
            stage_identity=stage_identity, registry=registry,
            runner=runner, tee_runtime=tee_runtime, anchor=anchor,
        )
        req = _make_streaming_request(
            settler_identity=settler_identity, model_id="not-real",
        )
        _, _, err = _decode_stream_frames(
            server.handle_token_stream(encode_message(req))
        )
        assert err is not None
        assert err.code == StageErrorCode.MODEL_NOT_FOUND.value


class TestTokenStreamRunnerErrors:
    """Runner-side failure modes. Either yields a clean StageError
    terminal frame OR yields some TokenFrames followed by a terminal
    StageError frame. Either way, NEVER raises."""

    def test_runner_raises_at_setup(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity,
    ):
        # A runner that raises BEFORE the for-loop starts (e.g. on
        # generator construction) must surface as INTERNAL_ERROR.
        class ExplodingRunner:
            def run_layer_slice_streaming(self, **kwargs):
                raise RuntimeError("boom at setup")

        server = LayerStageServer(
            identity=stage_identity, registry=registry, runner=runner,
            tee_runtime=tee_runtime, anchor=anchor,
            clock=lambda: 1000.0,
            streaming_runner=ExplodingRunner(),
        )
        req = _make_streaming_request(settler_identity=settler_identity)
        tokens, final, err = _decode_stream_frames(
            server.handle_token_stream(encode_message(req))
        )
        assert tokens == []
        assert final is None
        assert err is not None
        assert err.code == StageErrorCode.INTERNAL_ERROR.value

    def test_runner_raises_mid_stream(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity,
    ):
        class MidStreamCrasher:
            def run_layer_slice_streaming(self, **kwargs):
                yield StreamingChunk(sequence_index=0, text_delta="ok ")
                yield StreamingChunk(sequence_index=1, text_delta="ok ")
                raise RuntimeError("crashed mid-stream")

        server = LayerStageServer(
            identity=stage_identity, registry=registry, runner=runner,
            tee_runtime=tee_runtime, anchor=anchor,
            clock=lambda: 1000.0,
            streaming_runner=MidStreamCrasher(),
        )
        req = _make_streaming_request(settler_identity=settler_identity)
        tokens, final, err = _decode_stream_frames(
            server.handle_token_stream(encode_message(req))
        )
        assert len(tokens) == 2
        assert final is None
        assert err is not None
        assert err.code == StageErrorCode.INTERNAL_ERROR.value

    def test_runner_emits_out_of_order_sequence_index(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity,
    ):
        class OutOfOrderRunner:
            def run_layer_slice_streaming(self, **kwargs):
                yield StreamingChunk(sequence_index=0, text_delta="a")
                # BUG: should be 1.
                yield StreamingChunk(sequence_index=5, text_delta="b")

        server = LayerStageServer(
            identity=stage_identity, registry=registry, runner=runner,
            tee_runtime=tee_runtime, anchor=anchor,
            clock=lambda: 1000.0,
            streaming_runner=OutOfOrderRunner(),
        )
        req = _make_streaming_request(settler_identity=settler_identity)
        _, _, err = _decode_stream_frames(
            server.handle_token_stream(encode_message(req))
        )
        assert err is not None
        assert err.code == StageErrorCode.INTERNAL_ERROR.value
        assert "sequence_index" in err.message

    def test_runner_terminal_chunk_missing_aggregate_fields(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity,
    ):
        class IncompleteTerminal:
            def run_layer_slice_streaming(self, **kwargs):
                # finish_reason set but final-aggregate fields are None.
                yield StreamingChunk(
                    sequence_index=0,
                    text_delta="hi",
                    finish_reason="stop",
                    # Missing: full_output_text, duration_seconds, etc.
                )

        server = LayerStageServer(
            identity=stage_identity, registry=registry, runner=runner,
            tee_runtime=tee_runtime, anchor=anchor,
            clock=lambda: 1000.0,
            streaming_runner=IncompleteTerminal(),
        )
        req = _make_streaming_request(settler_identity=settler_identity)
        tokens, final, err = _decode_stream_frames(
            server.handle_token_stream(encode_message(req))
        )
        assert final is None
        assert err is not None
        assert err.code == StageErrorCode.INTERNAL_ERROR.value
        assert "final-aggregate" in err.message
        # M1 round-1 invariant: the terminal-chunk integrity check
        # MUST fail BEFORE any terminal TokenFrame hits the wire —
        # the docstring promises a SOLE error frame on failure.
        assert tokens == []

    def test_runner_joined_text_diverges_from_full_output_text(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity,
    ):
        class InconsistentRunner:
            def run_layer_slice_streaming(self, **kwargs):
                yield StreamingChunk(sequence_index=0, text_delta="hello ")
                yield StreamingChunk(
                    sequence_index=1,
                    text_delta="world",
                    finish_reason="stop",
                    # MISMATCH: deltas join to "hello world", but the
                    # full_output_text claims something else.
                    full_output_text="DIFFERENT TEXT",
                    duration_seconds=0.05,
                    tee_attestation=b"\x00" * 32,
                    tee_type=TEEType.SOFTWARE,
                    epsilon_spent=0.0,
                )

        server = LayerStageServer(
            identity=stage_identity, registry=registry, runner=runner,
            tee_runtime=tee_runtime, anchor=anchor,
            clock=lambda: 1000.0,
            streaming_runner=InconsistentRunner(),
        )
        req = _make_streaming_request(settler_identity=settler_identity)
        tokens, final, err = _decode_stream_frames(
            server.handle_token_stream(encode_message(req))
        )
        assert final is None
        assert err is not None
        assert err.code == StageErrorCode.INTERNAL_ERROR.value
        assert "joined text_deltas" in err.message
        # M1 round-1 invariant: the joined-text check fires BEFORE
        # the terminal TokenFrame is published — non-terminal
        # frames may already be on the wire (sequence_index < last)
        # but the LAST chunk's TokenFrame is held back until
        # validation passes. Here only the non-terminal "hello "
        # chunk is emitted; the terminal "world" chunk's
        # full_output_text mismatch fails before its TokenFrame
        # would be yielded.
        assert len(tokens) == 1
        assert tokens[0].text_delta == "hello "
        assert tokens[0].finish_reason is None

    def test_runner_yields_wrong_type_rejected(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity,
    ):
        class WrongType:
            def run_layer_slice_streaming(self, **kwargs):
                yield "not a StreamingChunk"  # type: ignore[misc]

        server = LayerStageServer(
            identity=stage_identity, registry=registry, runner=runner,
            tee_runtime=tee_runtime, anchor=anchor,
            clock=lambda: 1000.0,
            streaming_runner=WrongType(),
        )
        req = _make_streaming_request(settler_identity=settler_identity)
        _, _, err = _decode_stream_frames(
            server.handle_token_stream(encode_message(req))
        )
        assert err is not None
        assert err.code == StageErrorCode.INTERNAL_ERROR.value
        assert "StreamingChunk" in err.message

    def test_runner_exhausts_without_terminal_chunk(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity,
    ):
        class NoTerminal:
            def run_layer_slice_streaming(self, **kwargs):
                yield StreamingChunk(sequence_index=0, text_delta="a")
                yield StreamingChunk(sequence_index=1, text_delta="b")
                # No terminal — iterator just ends.

        server = LayerStageServer(
            identity=stage_identity, registry=registry, runner=runner,
            tee_runtime=tee_runtime, anchor=anchor,
            clock=lambda: 1000.0,
            streaming_runner=NoTerminal(),
        )
        req = _make_streaming_request(settler_identity=settler_identity)
        tokens, final, err = _decode_stream_frames(
            server.handle_token_stream(encode_message(req))
        )
        assert len(tokens) == 2
        assert final is None
        assert err is not None
        assert err.code == StageErrorCode.INTERNAL_ERROR.value
        assert "terminal chunk" in err.message


class TestTokenStreamNeverRaises:
    """``handle_token_stream`` must never propagate exceptions through
    the wire boundary. Garbage inputs all map to a single encoded
    StageError frame."""

    @pytest.mark.parametrize("garbage", [
        b"",
        b"not-json",
        b'{"type": null}',
        b'{"type": "stage_error", "protocol_version": 1}',
    ])
    def test_garbage_request_bytes_yield_single_stage_error(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        garbage,
    ):
        server = _make_streaming_server(
            stage_identity=stage_identity, registry=registry,
            runner=runner, tee_runtime=tee_runtime, anchor=anchor,
        )
        # The garbage parses or doesn't, but the iterator must yield
        # at most one StageError frame and terminate cleanly.
        frames = list(server.handle_token_stream(garbage))
        # We expect exactly one frame (the StageError) for truly
        # garbage payloads. The "stage_error" payload parses fine
        # but is the wrong message type — also produces a single
        # StageError.
        assert len(frames) == 1
        msg = parse_message(frames[0])
        assert isinstance(msg, StageError)


class TestTokenStreamStreamingRunnerConstruction:
    def test_invalid_streaming_runner_rejected_at_construction(
        self, stage_identity, registry, runner, tee_runtime, anchor,
    ):
        class NoMethod:
            pass

        with pytest.raises(RuntimeError, match="run_layer_slice_streaming"):
            LayerStageServer(
                identity=stage_identity,
                registry=registry,
                runner=runner,
                tee_runtime=tee_runtime,
                anchor=anchor,
                streaming_runner=NoMethod(),
            )

    def test_default_streaming_runner_is_none(
        self, stage_identity, registry, runner, tee_runtime, anchor,
    ):
        server = LayerStageServer(
            identity=stage_identity,
            registry=registry,
            runner=runner,
            tee_runtime=tee_runtime,
            anchor=anchor,
        )
        assert server._streaming_runner is None  # noqa: SLF001


class TestTokenStreamCancellation:
    """Phase 3.x.8 Task 6 — cancellation cleanup. v1 honest scope:
    caller .close() on the streaming generator deterministically
    closes the upstream runner generator. NO partial-output
    StreamFinalFrame is delivered (Python GeneratorExit semantics
    forbid yielding after .close()) — that's a Phase 3.x.8.x
    follow-up."""

    def _make_tracking_runner(self):
        """Build a runner whose generator records when .close() is
        called on it (Python forwards .close() through the
        generator protocol)."""

        class _TrackingRunner:
            def __init__(self):
                self.close_count = 0
                self.tokens_yielded = 0

            def run_layer_slice_streaming(self, **kwargs):
                outer = self  # capture for the inner generator

                def gen():
                    try:
                        # Emit many tokens so the caller has time
                        # to cancel mid-stream.
                        for i in range(10):
                            outer.tokens_yielded += 1
                            yield StreamingChunk(
                                sequence_index=i,
                                text_delta=f"t{i} ",
                            )
                        # Terminal — never reached if cancelled
                        # mid-stream.
                        outer.tokens_yielded += 1
                        yield StreamingChunk(
                            sequence_index=10,
                            text_delta="end",
                            finish_reason="stop",
                            full_output_text=(
                                "t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 end"
                            ),
                            duration_seconds=0.05,
                            tee_attestation=b"\x00" * 32,
                            tee_type=TEEType.SOFTWARE,
                            epsilon_spent=0.0,
                        )
                    except GeneratorExit:
                        outer.close_count += 1
                        raise

                return gen()

        return _TrackingRunner()

    def test_caller_close_propagates_to_runner_generator(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity,
    ):
        tracking_runner = self._make_tracking_runner()
        server = LayerStageServer(
            identity=stage_identity, registry=registry, runner=runner,
            tee_runtime=tee_runtime, anchor=anchor,
            clock=lambda: 1000.0,
            streaming_runner=tracking_runner,
        )
        req = _make_streaming_request(settler_identity=settler_identity)
        stream = server.handle_token_stream(encode_message(req))

        # Pull a few tokens, then close the stream early.
        first = next(stream)
        msg = parse_message(first)
        assert isinstance(msg, TokenFrame)
        next(stream)  # second token
        next(stream)  # third token

        # Closing the stream MUST propagate to the runner generator.
        stream.close()
        assert tracking_runner.close_count == 1

    def test_double_close_is_idempotent(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity,
    ):
        tracking_runner = self._make_tracking_runner()
        server = LayerStageServer(
            identity=stage_identity, registry=registry, runner=runner,
            tee_runtime=tee_runtime, anchor=anchor,
            clock=lambda: 1000.0,
            streaming_runner=tracking_runner,
        )
        req = _make_streaming_request(settler_identity=settler_identity)
        stream = server.handle_token_stream(encode_message(req))
        next(stream)
        stream.close()
        stream.close()  # second close — Python protocol: no-op
        # Runner gen .close() invoked at most once during the
        # pipeline cleanup; second close on an already-closed
        # generator is a Python no-op.
        assert tracking_runner.close_count == 1

    def test_close_before_any_tokens_consumed(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity,
    ):
        tracking_runner = self._make_tracking_runner()
        server = LayerStageServer(
            identity=stage_identity, registry=registry, runner=runner,
            tee_runtime=tee_runtime, anchor=anchor,
            clock=lambda: 1000.0,
            streaming_runner=tracking_runner,
        )
        req = _make_streaming_request(settler_identity=settler_identity)
        stream = server.handle_token_stream(encode_message(req))

        # Caller never iterated — close immediately.
        stream.close()
        # Runner generator was never advanced (no tokens yielded);
        # close on an unstarted generator is also a no-op for the
        # underlying chunk_iter, but our finally-clause still runs
        # the close call. Tokens yielded should be 0.
        assert tracking_runner.tokens_yielded == 0

    def test_natural_completion_also_closes_runner(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity,
    ):
        # Even on the happy path (full stream consumed), the
        # finally-clause runs close() on the runner — verifies the
        # cleanup is triggered uniformly, not only on cancellation.
        tracking_runner = self._make_tracking_runner()
        server = LayerStageServer(
            identity=stage_identity, registry=registry, runner=runner,
            tee_runtime=tee_runtime, anchor=anchor,
            clock=lambda: 1000.0,
            streaming_runner=tracking_runner,
        )
        req = _make_streaming_request(settler_identity=settler_identity)
        # Drain the stream completely.
        list(server.handle_token_stream(encode_message(req)))
        # Runner gen exhausted naturally; the finally clause's
        # close() call is a no-op on an exhausted gen, but it
        # ran without raising.
        assert tracking_runner.tokens_yielded == 11  # 10 + terminal

    def test_close_swallows_runner_close_exception(
        self, stage_identity, registry, runner, tee_runtime, anchor,
        settler_identity,
    ):
        # If the runner's .close() raises, the executor swallows
        # it (cleanup-time exceptions are non-actionable at the
        # wire boundary).
        class _NastyCloseRunner:
            def run_layer_slice_streaming(self, **kwargs):
                def gen():
                    try:
                        yield StreamingChunk(sequence_index=0, text_delta="a")
                        yield StreamingChunk(sequence_index=1, text_delta="b")
                    except GeneratorExit:
                        # Runner explodes on cleanup — must not
                        # propagate through the wire.
                        raise RuntimeError("close-time crash")

                return gen()

        server = LayerStageServer(
            identity=stage_identity, registry=registry, runner=runner,
            tee_runtime=tee_runtime, anchor=anchor,
            clock=lambda: 1000.0,
            streaming_runner=_NastyCloseRunner(),
        )
        req = _make_streaming_request(settler_identity=settler_identity)
        stream = server.handle_token_stream(encode_message(req))
        next(stream)
        # close() must not propagate the runner's crash.
        stream.close()  # no exception
