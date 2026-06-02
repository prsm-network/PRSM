"""Phase 3.x.8.1 Task 5 — E2E integration test for the streaming
HTTP endpoint.

Drives ``POST /compute/inference/stream`` via FastAPI TestClient
against a node that wires:
- A real ``NodeIdentity`` for receipt signing.
- A streaming-capable executor that yields realistic
  ``InferenceTokenEvent``s + a signed ``InferenceResult``.
- A fake escrow that records create / release / refund calls.

Headline guarantees verified end-to-end:
- Happy path: SSE token stream → joined deltas equal the
  result's output bit-for-bit; signed receipt verifies under the
  settler identity; ``streamed_output=True`` invariant from Phase
  3.x.8 Task 4 holds across the HTTP boundary.
- Pre-execute failure: sole ``event: error`` frame, no token
  events, escrow refunded.
- Back-compat: existing ``/compute/inference`` unary endpoint
  unchanged on the same node.

The deep-stack composition (RpcChainExecutor + LayerStageServer +
SyntheticStreamingRunner) is exhaustively validated in Phase 3.x.8
Task 7's ``test_chain_rpc_e2e.py::TestStreamingTokenOutput``. This
file focuses on what's NEW in Phase 3.x.8.1: the SSE wire path,
escrow integration, receipt re-signing under the API node identity,
and back-compat with the unary endpoint on the same FastAPI app.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from types import SimpleNamespace
from typing import AsyncIterator, Dict, List, Optional, Tuple, Union

import pytest
from fastapi.testclient import TestClient

from prsm.compute.inference.models import (
    ContentTier,
    InferenceReceipt,
    InferenceRequest,
    InferenceResult,
)
from prsm.compute.inference.parallax_executor import InferenceTokenEvent
from prsm.compute.inference.receipt import sign_receipt, verify_receipt
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.api import create_api_app
from prsm.node.identity import NodeIdentity, generate_node_identity


# ──────────────────────────────────────────────────────────────────────────
# Fakes — minimal shape for the endpoint's pre-execute gates
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeEscrow:
    create_count: int = 0
    release_count: int = 0
    refund_count: int = 0
    last_refund_reason: Optional[str] = None

    async def create_escrow(self, *, job_id, amount, requester_id):
        self.create_count += 1
        # sp952 — the real create_escrow returns an EscrowEntry OBJECT; the
        # streaming settle path reads getattr(escrow_entry, "amount", ...) to size
        # the credit-policy decision. A plain dict has no .amount attribute, so
        # getattr fell back to 0 -> the policy released nothing -> the escrow was
        # never settled (assert release_count==1 failed). Return an object.
        return SimpleNamespace(job_id=job_id, amount=amount, requester_id=requester_id)

    async def release_escrow(self, job_id, provider_id, consensus_reached=True, partial_amount=None, **_kw):
        # sp952 — match the REAL PaymentEscrow.release_escrow signature
        # (positional). settle_inference_receipt calls release_escrow(job_id,
        # operator_id) positionally; the prior keyword-only mock raised TypeError
        # on that path (caught upstream), leaving the escrow unreleased.
        self.release_count += 1

    async def release_escrow_split(self, job_id, splits, consensus_reached=True, **_kw):
        # sp952 — mirror the real split-release API (positional); the receipt-
        # aware settle uses it when a partial release+refund decision applies.
        self.release_count += 1

    async def refund_escrow(self, job_id, reason):
        self.refund_count += 1
        self.last_refund_reason = reason

    def get_escrow(self, job_id):
        return None


class _SignedStreamingExecutor:
    """Streaming executor whose terminal InferenceResult carries a
    real Ed25519-signed receipt under the supplied NodeIdentity.
    Mirrors ``ParallaxScheduledExecutor.execute_streaming``'s shape
    but bypasses the chain/router/runner stack — the deep-stack
    composition is validated separately in Phase 3.x.8 Task 7."""

    def __init__(
        self,
        *,
        identity: NodeIdentity,
        token_deltas: Optional[List[str]] = None,
        terminal_failure_message: Optional[str] = None,
        cost_ftns: Decimal = Decimal("0.04"),
        fail_after_n_tokens: Optional[int] = None,
    ):
        self._identity = identity
        self._token_deltas = (
            token_deltas if token_deltas is not None
            else ["Hello", " ", "world", "!"]
        )
        self._terminal_failure_message = terminal_failure_message
        self._cost = cost_ftns
        self._fail_after_n_tokens = fail_after_n_tokens
        self.calls = 0

    async def execute_streaming(
        self, request: InferenceRequest,
    ) -> AsyncIterator[Union[InferenceTokenEvent, InferenceResult]]:
        self.calls += 1
        if self._terminal_failure_message is not None:
            yield InferenceResult.failure(
                request.request_id, self._terminal_failure_message,
            )
            return
        last = len(self._token_deltas) - 1
        for i, delta in enumerate(self._token_deltas):
            # Phase 3.x.8.1 round-1 M2 test mode: emit N tokens
            # then yield a terminal failure (simulates a mid-stream
            # chain crash AFTER tokens have hit the wire). The
            # endpoint MUST settle (not refund) per design plan
            # §3.4 — caller paid for chain dispatch + the network
            # already burned compute on those N tokens.
            if (
                self._fail_after_n_tokens is not None
                and i == self._fail_after_n_tokens
            ):
                yield InferenceResult.failure(
                    request.request_id,
                    "simulated mid-stream crash after tokens emitted",
                )
                return
            yield InferenceTokenEvent(
                sequence_index=i,
                text_delta=delta,
                finish_reason="stop" if i == last else None,
            )
        joined = "".join(self._token_deltas)
        receipt = InferenceReceipt(
            job_id=f"parallax-stream-job-{request.request_id}",
            request_id=request.request_id,
            model_id=request.model_id,
            content_tier=request.content_tier,
            privacy_tier=request.privacy_tier,
            epsilon_spent=8.0,
            tee_type=TEEType.SOFTWARE,
            tee_attestation=b"\x01" * 32,
            output_hash=hashlib.sha256(joined.encode("utf-8")).digest(),
            duration_seconds=0.05,
            cost_ftns=self._cost,
            streamed_output=True,
        )
        signed = sign_receipt(receipt, self._identity)
        yield InferenceResult(
            request_id=request.request_id,
            success=True,
            output=joined,
            receipt=signed,
        )


class _UnaryShimExecutor:
    """Shim that satisfies the unary /compute/inference endpoint's
    ``execute()`` Protocol so the back-compat test can hit BOTH
    endpoints on the same node fixture. Returns a deterministic
    success result."""

    def __init__(self, *, identity: NodeIdentity):
        self._identity = identity

    async def execute(self, request: InferenceRequest) -> InferenceResult:
        text = "unary-output"
        receipt = InferenceReceipt(
            job_id=f"parallax-job-{request.request_id}",
            request_id=request.request_id,
            model_id=request.model_id,
            content_tier=request.content_tier,
            privacy_tier=request.privacy_tier,
            epsilon_spent=0.0,
            tee_type=TEEType.SOFTWARE,
            tee_attestation=b"\x02" * 32,
            output_hash=hashlib.sha256(text.encode("utf-8")).digest(),
            duration_seconds=0.05,
            cost_ftns=Decimal("0.01"),
            streamed_output=False,
        )
        return InferenceResult(
            request_id=request.request_id,
            success=True,
            output=text,
            receipt=sign_receipt(receipt, self._identity),
        )


class _DualExecutor:
    """Combines streaming + unary surfaces so a single node fixture
    can serve both /compute/inference and /compute/inference/stream
    in the same test."""

    def __init__(
        self,
        *,
        identity: NodeIdentity,
        streaming: _SignedStreamingExecutor,
        unary: _UnaryShimExecutor,
    ):
        self._streaming = streaming
        self._unary = unary

    async def execute(self, request: InferenceRequest) -> InferenceResult:
        return await self._unary.execute(request)

    def execute_streaming(self, request: InferenceRequest):
        return self._streaming.execute_streaming(request)


class _FakeNode:
    def __init__(
        self,
        *,
        identity: NodeIdentity,
        executor,
        escrow: Optional[_FakeEscrow] = None,
    ):
        self.identity = identity
        self.inference_executor = executor
        self._payment_escrow = escrow
        self.privacy_budget = None


def _parse_sse_events(body: bytes) -> List[Tuple[str, dict]]:
    """Minimal SSE parser used to assert event sequencing."""
    events: List[Tuple[str, dict]] = []
    current_event = "message"
    current_data: List[str] = []
    for line in body.decode("utf-8").split("\n"):
        if line == "":
            if current_data:
                payload = "\n".join(current_data)
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError:
                    parsed = payload  # type: ignore[assignment]
                events.append((current_event, parsed))
                current_data = []
                current_event = "message"
        elif line.startswith("event: "):
            current_event = line[len("event: "):]
        elif line.startswith("data: "):
            current_data.append(line[len("data: "):])
    return events


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def settler_identity() -> NodeIdentity:
    return generate_node_identity(display_name="e2e-settler")


@pytest.fixture
def escrow() -> _FakeEscrow:
    return _FakeEscrow()


@pytest.fixture
def streaming_executor(settler_identity):
    return _SignedStreamingExecutor(identity=settler_identity)


@pytest.fixture
def unary_executor(settler_identity):
    return _UnaryShimExecutor(identity=settler_identity)


@pytest.fixture
def node(settler_identity, streaming_executor, unary_executor, escrow):
    dual = _DualExecutor(
        identity=settler_identity,
        streaming=streaming_executor,
        unary=unary_executor,
    )
    return _FakeNode(
        identity=settler_identity, executor=dual, escrow=escrow,
    )


@pytest.fixture
def client(node):
    app = create_api_app(node, enable_security=False)
    return TestClient(app)


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────


class TestStreamingHttpHappyPath:
    def test_joined_deltas_equal_result_output(
        self, client, streaming_executor,
    ):
        with client.stream(
            "POST", "/compute/inference/stream",
            json={
                "prompt": "anything",
                "model_id": "mock-llama-3-8b",
                "budget_ftns": 1.0,
                "privacy_tier": "none",
                "content_tier": "A",
            },
            headers={"Accept": "text/event-stream"},
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith(
                "text/event-stream"
            )
            body = b"".join(response.iter_bytes())
        events = _parse_sse_events(body)

        # Sequence: token, token, token, token, result.
        assert [e[0] for e in events] == [
            "token", "token", "token", "token", "result",
        ]
        # Joined deltas equal the result's output bit-for-bit.
        token_deltas = [e[1]["text_delta"] for e in events if e[0] == "token"]
        result_event = [e[1] for e in events if e[0] == "result"][0]
        joined = "".join(token_deltas)
        assert joined == "Hello world!"
        assert joined == result_event["output"]
        # Streaming executor was invoked exactly once.
        assert streaming_executor.calls == 1

    def test_terminal_token_carries_finish_reason_stop(self, client):
        with client.stream(
            "POST", "/compute/inference/stream",
            json={
                "prompt": "x", "model_id": "mock-llama-3-8b",
                "budget_ftns": 1.0,
            },
        ) as response:
            body = b"".join(response.iter_bytes())
        events = _parse_sse_events(body)
        token_events = [e[1] for e in events if e[0] == "token"]
        assert token_events[-1]["finish_reason"] == "stop"
        for ev in token_events[:-1]:
            assert ev["finish_reason"] is None

    def test_signed_receipt_verifies_under_settler_identity(
        self, client, settler_identity,
    ):
        """Headline cryptographic invariant: the receipt extracted
        from the SSE result event verifies under the node's
        settling identity. Any tampering — including a relay
        flipping streamed_output — invalidates the signature
        (Phase 3.x.8 Task 4 downgrade-resistance verified
        end-to-end through the HTTP boundary)."""
        with client.stream(
            "POST", "/compute/inference/stream",
            json={
                "prompt": "verify-me", "model_id": "mock-llama-3-8b",
                "budget_ftns": 1.0,
            },
        ) as response:
            body = b"".join(response.iter_bytes())
        events = _parse_sse_events(body)
        result_payload = [e[1] for e in events if e[0] == "result"][0]
        receipt_dict = result_payload["receipt"]
        receipt = InferenceReceipt.from_dict(receipt_dict)
        assert receipt.streamed_output is True
        assert verify_receipt(receipt, identity=settler_identity)

        # Tamper test: flipping streamed_output → signature fails.
        import dataclasses
        downgraded = dataclasses.replace(receipt, streamed_output=False)
        assert not verify_receipt(downgraded, identity=settler_identity)

    def test_escrow_settled_on_success(self, client, escrow):
        with client.stream(
            "POST", "/compute/inference/stream",
            json={
                "prompt": "x", "model_id": "mock-llama-3-8b",
                "budget_ftns": 1.0,
            },
        ) as response:
            list(response.iter_bytes())
        assert escrow.create_count == 1
        assert escrow.release_count == 1
        assert escrow.refund_count == 0


class TestStreamingHttpBillingPolicy:
    """Phase 3.x.8.1 round-1 M1+M2 remediation: design plan §3.4
    settle-on-tokens-emitted policy.

    Pre-execute failure (zero tokens emitted) → refund. Mid-stream
    failure AFTER tokens have hit the wire → settle (caller paid
    for the work the network already did).
    """

    def test_mid_stream_failure_after_tokens_settles_not_refunds(
        self, settler_identity, escrow,
    ):
        # Executor emits 2 tokens, then yields InferenceResult.failure
        # — simulates a chain crash after partial output has reached
        # the wire. Per §3.4: settle (release escrow), not refund.
        failing_after_tokens = _SignedStreamingExecutor(
            identity=settler_identity,
            fail_after_n_tokens=2,
        )
        unary = _UnaryShimExecutor(identity=settler_identity)
        dual = _DualExecutor(
            identity=settler_identity,
            streaming=failing_after_tokens, unary=unary,
        )
        node = _FakeNode(
            identity=settler_identity, executor=dual, escrow=escrow,
        )
        app = create_api_app(node, enable_security=False)
        client = TestClient(app)
        with client.stream(
            "POST", "/compute/inference/stream",
            json={
                "prompt": "x", "model_id": "mock-llama-3-8b",
                "budget_ftns": 1.0,
            },
        ) as response:
            body = b"".join(response.iter_bytes())
        events = _parse_sse_events(body)

        # 2 token events + 1 error event (no result event because
        # the executor failed after the 2nd token).
        assert [e[0] for e in events] == ["token", "token", "error"]
        # Headline §3.4 invariant: escrow SETTLED (release_count=1),
        # not refunded.
        assert escrow.create_count == 1
        assert escrow.release_count == 1
        assert escrow.refund_count == 0


class TestStreamingHttpFailurePath:
    def test_pre_execute_failure_yields_sole_error_event_and_refunds(
        self, settler_identity, escrow,
    ):
        # Build a node whose streaming executor pre-emptively fails
        # (simulating budget gate / unknown model / etc.).
        failing = _SignedStreamingExecutor(
            identity=settler_identity,
            terminal_failure_message="Unknown model_id: bogus",
        )
        unary = _UnaryShimExecutor(identity=settler_identity)
        dual = _DualExecutor(
            identity=settler_identity, streaming=failing, unary=unary,
        )
        node = _FakeNode(
            identity=settler_identity, executor=dual, escrow=escrow,
        )
        app = create_api_app(node, enable_security=False)
        client = TestClient(app)
        with client.stream(
            "POST", "/compute/inference/stream",
            json={
                "prompt": "x", "model_id": "bogus", "budget_ftns": 1.0,
            },
        ) as response:
            body = b"".join(response.iter_bytes())
        events = _parse_sse_events(body)
        # Sole error event; no token events; no result event.
        assert [e[0] for e in events] == ["error"]
        assert "Unknown model_id" in events[0][1]["error"]
        # Escrow refunded.
        assert escrow.create_count == 1
        assert escrow.refund_count == 1
        assert escrow.release_count == 0


class TestStreamingHttpBackCompat:
    def test_unary_endpoint_unchanged_on_same_node(
        self, client,
    ):
        """The unary /compute/inference endpoint continues to work
        on a node that ALSO serves /compute/inference/stream. v1
        invariant: streaming is purely additive."""
        response = client.post(
            "/compute/inference",
            json={
                "prompt": "back-compat",
                "model_id": "mock-llama-3-8b",
                "budget_ftns": 1.0,
                "privacy_tier": "none",
                "content_tier": "A",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["output"] == "unary-output"
        # Unary receipt's streamed_output flag is False (Phase 3.x.8
        # Task 4 conditional encoding — preserves byte-equivalence
        # with pre-3.x.8 receipts).
        assert body["receipt"]["streamed_output"] is False

    def test_unary_and_streaming_receipts_distinguishable_by_flag(
        self, client,
    ):
        """Audit-log invariant: a receipt's ``streamed_output``
        flag tells operators whether the corresponding job ran
        unary or streamed. Both paths emit valid signed receipts
        on the same node; only the flag differs."""
        # Unary call.
        unary_resp = client.post(
            "/compute/inference",
            json={
                "prompt": "unary",
                "model_id": "mock-llama-3-8b",
                "budget_ftns": 1.0,
            },
        )
        unary_receipt = unary_resp.json()["receipt"]
        # Streaming call.
        with client.stream(
            "POST", "/compute/inference/stream",
            json={
                "prompt": "streamed",
                "model_id": "mock-llama-3-8b",
                "budget_ftns": 1.0,
            },
        ) as response:
            body = b"".join(response.iter_bytes())
        events = _parse_sse_events(body)
        streamed_receipt = [
            e[1] for e in events if e[0] == "result"
        ][0]["receipt"]

        assert unary_receipt["streamed_output"] is False
        assert streamed_receipt["streamed_output"] is True
        # Both are signed.
        assert unary_receipt["settler_signature"]
        assert streamed_receipt["settler_signature"]
