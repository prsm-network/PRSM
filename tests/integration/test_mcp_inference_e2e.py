"""End-to-end integration test for the MCP inference pipeline.

Phase 3.x.1 Task 13. Validates Tasks 1-2 + 5-8 wired together through the
full request flow:

  MCP client (test) calls handle_prsm_inference
       │
       ▼
  Handler builds request payload, calls _call_node_api
       │
       ▼  (patched here to route through FastAPI TestClient)
  POST /compute/inference endpoint
       │
       ▼
  Validates inputs, builds InferenceRequest
       │
       ▼
  PaymentEscrow.create_escrow (real escrow tracking)
       │
       ▼
  MockInferenceExecutor.execute → InferenceResult with mock receipt
       │
       ▼
  Endpoint replaces job_id, signs receipt with node.identity (Task 2)
       │
       ▼
  privacy_budget.record_spend (DP epsilon tracking)
       │
       ▼
  PaymentEscrow.release_escrow
       │
       ▼
  JSON response back to handler
       │
       ▼
  Handler formats with cost-reconciliation footer (Task 7)
       │
       ▼
  Test verifies: output, signed receipt, escrow lifecycle, footer

Plus follow-up prsm_billing_status query for the same job_id to verify
the billing endpoint can retrieve escrow state after settlement.

Hardhat-based on-chain settlement tests live separately (different setup
profile, skipped in fast unit-test runs). This test exercises the full
pipeline against the in-memory PaymentEscrow + MockInferenceExecutor
substrate — sufficient to catch wire-up bugs in the integration layer
without requiring a JSON-RPC node.
"""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.compute.inference import (
    ContentTier,
    InferenceReceipt,
    MockInferenceExecutor,
    verify_receipt,
)
from prsm.compute.tee.models import PrivacyLevel
from prsm.mcp_server import (
    handle_prsm_billing_status,
    handle_prsm_inference,
)
from prsm.node.api import create_api_app
from prsm.node.identity import generate_node_identity


# ── Stateful escrow fake ────────────────────────────────────────────────────


class _FakeEscrow:
    """In-memory PaymentEscrow stub that tracks lifecycle state.

    Mirrors the API surface needed by /compute/inference + /billing/{job_id}
    endpoints without depending on the real PaymentEscrow's persistence
    layer or chain interaction.
    """

    def __init__(self):
        self._entries = {}  # job_id -> dict
        self.create_calls = []
        self.release_calls = []
        self.refund_calls = []

    async def create_escrow(self, *, job_id, amount, requester_id, **_kw):
        self.create_calls.append({"job_id": job_id, "amount": amount, "requester_id": requester_id})
        entry = MagicMock()
        entry.job_id = job_id
        entry.escrow_id = f"escrow-{job_id}"
        entry.amount = amount
        entry.requester_id = requester_id
        entry.status = MagicMock()
        entry.status.value = "pending"
        entry.provider_winner = None
        entry.tx_lock = None
        entry.tx_release = None
        entry.created_at = 1745625600.0
        entry.completed_at = None
        entry.metadata = {}
        self._entries[job_id] = entry
        return entry

    async def release_escrow(self, *, job_id, provider_id, **_kw):
        self.release_calls.append({"job_id": job_id, "provider_id": provider_id})
        entry = self._entries.get(job_id)
        if entry:
            entry.status.value = "released"
            entry.provider_winner = provider_id
            entry.tx_release = f"0xrelease-{job_id}"
            entry.completed_at = 1745625610.0
        return True

    async def release_escrow_split(self, job_id, splits, consensus_reached=True, **_kw):
        # sp951 — the production receipt-aware inference settle (api.py) releases
        # via the multi-recipient split API, NOT release_escrow. This mock
        # predated that path, so the settle AttributeError'd and the escrow never
        # released (assert len(release_calls)==1 failed). Mirror release_escrow's
        # bookkeeping: splits is a list of (recipient_id, amount) tuples; in the
        # single-node case the sole recipient is the operator node.
        provider_id = splits[0][0] if splits else None
        self.release_calls.append({
            "job_id": job_id, "provider_id": provider_id, "splits": list(splits),
        })
        entry = self._entries.get(job_id)
        if entry:
            entry.status.value = "released"
            entry.provider_winner = provider_id
            entry.tx_release = f"0xrelease-{job_id}"
            entry.completed_at = 1745625610.0
        return True

    async def refund_escrow(self, job_id, reason, **_kw):
        self.refund_calls.append({"job_id": job_id, "reason": reason})
        entry = self._entries.get(job_id)
        if entry:
            entry.status.value = "refunded"
            entry.completed_at = 1745625610.0
        return True

    def get_escrow(self, job_id):
        return self._entries.get(job_id)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def real_identity():
    return generate_node_identity(display_name="e2e-test-node")


@pytest.fixture
def fake_escrow():
    return _FakeEscrow()


@pytest.fixture
def privacy_budget_recorder():
    """Tracks DP epsilon spending."""
    rec = MagicMock()
    rec.record_spend = MagicMock()
    return rec


@pytest.fixture
def mock_node(real_identity, fake_escrow, privacy_budget_recorder):
    """A mock PRSM node wired with real-enough subsystems for E2E coverage."""
    node = MagicMock()
    node.identity = real_identity
    node.inference_executor = MockInferenceExecutor(
        models=["mock-llama-3-8b", "mock-mistral-7b"],
        fixed_cost=Decimal("0.10"),
    )
    node._payment_escrow = fake_escrow
    node.privacy_budget = privacy_budget_recorder
    node.transport = MagicMock()
    node.transport.peers = {}
    node.transport.address = "127.0.0.1:8765"
    node.compute_provider = True
    node.storage_provider = False
    return node


@pytest.fixture
def api_client(mock_node):
    app = create_api_app(mock_node, enable_security=False)
    return TestClient(app)


@pytest.fixture
def routed_call_node_api(api_client):
    """Patches prsm.mcp_server._call_node_api to hit the FastAPI TestClient.

    This is the key wire-up: MCP handlers think they're talking to a remote
    node over HTTP; in this test, they route through the in-process FastAPI
    app via TestClient. End-to-end coverage without spinning up a real
    HTTP server or stdio subprocess.
    """
    async def _route(method, path, data=None):
        if method == "POST":
            response = api_client.post(path, json=data or {})
        elif method == "GET":
            response = api_client.get(path)
        else:
            raise ValueError(f"Unsupported method {method}")
        # Best-effort JSON parse — fall back to raw text if non-JSON
        try:
            return response.json()
        except Exception:
            return {"detail": response.text, "status_code": response.status_code}

    with patch("prsm.mcp_server._call_node_api", side_effect=_route):
        yield


# ── E2E happy path ──────────────────────────────────────────────────────────


class TestMcpInferenceE2eHappyPath:
    """Single-shot E2E: prsm_inference → API → executor → receipt → escrow → reply."""

    @pytest.mark.asyncio
    async def test_full_pipeline_success(
        self, routed_call_node_api, fake_escrow, privacy_budget_recorder, real_identity
    ):
        # Submit inference via the MCP handler (the LLM-facing entry point)
        result_text = await handle_prsm_inference({
            "prompt": "What is the meaning of life?",
            "model_id": "mock-llama-3-8b",
            "budget_ftns": 1.0,
            "privacy_tier": "standard",
            "content_tier": "A",
        })

        # ── Verify response shape ──
        assert "PRSM Inference Result" in result_text
        assert "mock-llama-3-8b" in result_text
        # Cost-reconciliation footer (Task 7)
        assert "Reconcile via:" in result_text
        assert "prsm_billing_status" in result_text
        # Receipt verification hint (Task 2 + Task 6)
        assert "verify_receipt" in result_text

        # ── Verify escrow lifecycle ──
        # Exactly one create + one release; no refund on the happy path
        assert len(fake_escrow.create_calls) == 1
        assert fake_escrow.create_calls[0]["amount"] == 1.0
        assert fake_escrow.create_calls[0]["requester_id"] == real_identity.node_id
        assert len(fake_escrow.release_calls) == 1
        assert fake_escrow.release_calls[0]["provider_id"] == real_identity.node_id
        assert len(fake_escrow.refund_calls) == 0

        # Same job_id flowed through create + release
        api_job_id = fake_escrow.create_calls[0]["job_id"]
        assert api_job_id == fake_escrow.release_calls[0]["job_id"]
        assert api_job_id.startswith("infer-")

        # ── Verify privacy budget recorded ──
        privacy_budget_recorder.record_spend.assert_called_once()
        eps_arg, op_arg, job_arg = privacy_budget_recorder.record_spend.call_args.args
        assert eps_arg == 8.0  # standard tier
        assert op_arg == "inference"
        assert job_arg == api_job_id

    @pytest.mark.asyncio
    async def test_signed_receipt_independently_verifiable(
        self, routed_call_node_api, real_identity
    ):
        """The receipt produced end-to-end must verify against ONLY the node's public key.

        This is the load-bearing security property — a third party with no
        trust relationship to PRSM can verify any receipt this pipeline
        produces.
        """
        # Patch the handler to capture the parsed JSON response (not the
        # formatted markdown text) so we can extract the receipt for
        # verification.
        captured = {}

        async def capture_response(method, path, data=None):
            from fastapi.testclient import TestClient
            # Re-route through the same TestClient — easiest is to keep the
            # outer fixture's patch and grab the endpoint response body
            # via a side-effect read of the API endpoint directly here.
            pass

        # Instead, call the API endpoint directly (it's the same wire we
        # exercise above), then verify the returned receipt.
        # But routed_call_node_api already patches the function — the
        # cleanest way to grab the JSON is to add a side-effect within
        # the patch that records the response. For test simplicity, we
        # instead do a fresh direct call here.
        from prsm.node.api import create_api_app
        from fastapi.testclient import TestClient as _TC

        # The fixture's TestClient is the same in-process app. We want the
        # raw JSON response, not the prsm_inference handler's formatted
        # text — so reach into the same node + app via the existing patch:
        from prsm.mcp_server import _call_node_api as _patched
        api_response = await _patched(
            "POST",
            "/compute/inference",
            {
                "prompt": "Verify me!",
                "model_id": "mock-llama-3-8b",
                "budget_ftns": 1.0,
            },
        )

        assert api_response["success"] is True
        receipt = InferenceReceipt.from_dict(api_response["receipt"])

        # Public-key-only verification — no node, no PRSM trust anchor
        assert verify_receipt(
            receipt, public_key_b64=real_identity.public_key_b64
        )

        # Tampering must invalidate
        import dataclasses
        tampered = dataclasses.replace(receipt, output_hash=b"\xff" * 32)
        assert not verify_receipt(
            tampered, public_key_b64=real_identity.public_key_b64
        )

    @pytest.mark.asyncio
    async def test_billing_status_queryable_after_inference(
        self, routed_call_node_api, fake_escrow
    ):
        """After inference completes, prsm_billing_status can fetch escrow state."""
        await handle_prsm_inference({
            "prompt": "x",
            "model_id": "mock-llama-3-8b",
            "budget_ftns": 1.0,
        })

        # Pull the API-side job_id from the escrow tracker
        job_id = fake_escrow.create_calls[0]["job_id"]

        # Now query billing status for that job_id
        billing_text = await handle_prsm_billing_status({"job_id": job_id})
        assert "PRSM Billing Status" in billing_text
        assert job_id in billing_text
        assert "released" in billing_text  # happy path = released
        # Escrow tracks the LOCKED BUDGET (1.0 here), not the executor's
        # internal cost estimate. At release, the full budget moves to the
        # provider — the executor's "fixed_cost" is internal accounting,
        # not a contract on what gets escrowed.
        assert "1.0" in billing_text

    @pytest.mark.asyncio
    async def test_billing_status_404_for_unknown_job(
        self, routed_call_node_api
    ):
        """prsm_billing_status returns user-readable 404 for unknown job_id."""
        billing_text = await handle_prsm_billing_status({"job_id": "infer-does-not-exist"})
        assert "No escrow found" in billing_text
        assert "infer-does-not-exist" in billing_text


# ── E2E failure paths ───────────────────────────────────────────────────────


class TestMcpInferenceE2eFailurePaths:
    @pytest.mark.asyncio
    async def test_unknown_model_returns_failure_and_refunds_escrow(
        self, routed_call_node_api, fake_escrow
    ):
        result_text = await handle_prsm_inference({
            "prompt": "x",
            "model_id": "not-registered",
            "budget_ftns": 1.0,
        })

        # Handler surfaces the rejection (note: response is success:false JSON,
        # which the handler turns into "Inference failed: ..." per its branch.)
        assert "failed" in result_text.lower() or "rejected" in result_text.lower()

        # Escrow flow on failure: create then refund (no release)
        assert len(fake_escrow.create_calls) == 1
        assert len(fake_escrow.refund_calls) == 1
        assert len(fake_escrow.release_calls) == 0

        # Refund reason mentions the unknown model
        refund_reason = fake_escrow.refund_calls[0]["reason"]
        assert "Unknown model_id" in refund_reason or "not-registered" in refund_reason

    @pytest.mark.asyncio
    async def test_validation_error_skips_escrow_entirely(
        self, routed_call_node_api, fake_escrow
    ):
        # Missing prompt → handler returns early with validation error
        result_text = await handle_prsm_inference({
            "model_id": "mock-llama-3-8b",
            "budget_ftns": 1.0,
        })

        assert "Missing required 'prompt'" in result_text
        # No escrow created when validation fails before API call
        assert len(fake_escrow.create_calls) == 0
        assert len(fake_escrow.release_calls) == 0
        assert len(fake_escrow.refund_calls) == 0


# ── E2E with privacy tier variations ────────────────────────────────────────


class TestMcpInferenceE2ePrivacyTiers:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("tier,expected_eps", [
        ("standard", 8.0),
        ("high", 4.0),
        ("maximum", 1.0),
    ])
    async def test_each_tier_records_correct_epsilon(
        self, routed_call_node_api, privacy_budget_recorder,
        tier, expected_eps,
    ):
        await handle_prsm_inference({
            "prompt": "x",
            "model_id": "mock-llama-3-8b",
            "budget_ftns": 1.0,
            "privacy_tier": tier,
        })
        privacy_budget_recorder.record_spend.assert_called_once()
        eps = privacy_budget_recorder.record_spend.call_args.args[0]
        assert eps == expected_eps

    @pytest.mark.asyncio
    async def test_privacy_none_skips_epsilon_tracking(
        self, routed_call_node_api, privacy_budget_recorder
    ):
        await handle_prsm_inference({
            "prompt": "x",
            "model_id": "mock-llama-3-8b",
            "budget_ftns": 1.0,
            "privacy_tier": "none",
        })
        privacy_budget_recorder.record_spend.assert_not_called()


# ── E2E with content tier variations ────────────────────────────────────────


class TestMcpInferenceE2eContentTiers:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("tier", ["A", "B", "C"])
    async def test_all_three_content_tiers_complete_pipeline(
        self, routed_call_node_api, fake_escrow, tier
    ):
        result_text = await handle_prsm_inference({
            "prompt": f"Test for tier {tier}",
            "model_id": "mock-llama-3-8b",
            "budget_ftns": 1.0,
            "content_tier": tier,
        })

        assert "PRSM Inference Result" in result_text
        assert tier in result_text  # appears in extra_fields footer
        # Each tier completed the full escrow lifecycle
        assert len(fake_escrow.release_calls) == 1


# ── Acceptance criterion ────────────────────────────────────────────────────


class TestTask13Acceptance:
    """Phase 3.x.1 Task 13 acceptance:
    Single-shot E2E test passes from mock MCP client through node API +
    escrow settlement + receipt verification."""

    @pytest.mark.asyncio
    async def test_single_shot_e2e_pipeline(
        self, routed_call_node_api, fake_escrow, privacy_budget_recorder,
        real_identity,
    ):
        """One test exercising every Phase 3.x.1 layer in a single call."""
        result_text = await handle_prsm_inference({
            "prompt": "Acceptance criterion check",
            "model_id": "mock-llama-3-8b",
            "budget_ftns": 1.0,
            "privacy_tier": "standard",
            "content_tier": "B",
        })

        # 1. Response was built (Task 6 MCP handler + Task 7 footer)
        assert "PRSM Inference Result" in result_text
        assert "Reconcile via:" in result_text
        assert "verify_receipt" in result_text

        # 2. API endpoint completed (Task 5)
        assert len(fake_escrow.create_calls) == 1
        assert len(fake_escrow.release_calls) == 1

        # 3. Privacy budget tracked (Task 5 + privacy integration)
        assert privacy_budget_recorder.record_spend.call_count == 1

        # 4. Same job_id consistent across escrow + footer
        api_job_id = fake_escrow.create_calls[0]["job_id"]
        assert api_job_id in result_text

        # 5. Receipt is independently verifiable (Task 2)
        from prsm.mcp_server import _call_node_api as _patched_call
        api_response = await _patched_call(
            "POST",
            "/compute/inference",
            {
                "prompt": "Verify",
                "model_id": "mock-llama-3-8b",
                "budget_ftns": 1.0,
            },
        )
        receipt = InferenceReceipt.from_dict(api_response["receipt"])
        assert verify_receipt(receipt, public_key_b64=real_identity.public_key_b64)

        # 6. Billing endpoint backfills lookup (Task 7)
        billing_text = await handle_prsm_billing_status({"job_id": api_job_id})
        assert api_job_id in billing_text
        assert "released" in billing_text
