"""Sprint 783 — settle_inference_receipt: thin wrapper over PaymentEscrow.

Sprint 782 shipped SettlementDecision (release/refund/slash
amounts pre-computed). Sprint 783 wires it into the actual
PaymentEscrow API.

  settle_inference_receipt(
      payment_escrow, receipt, operator_id, job_id,
  ) -> SettlementDecision

What it does:
  1. Looks up the escrow by job_id to get amount locked.
  2. Calls compute_settlement_for_receipt(receipt, escrow_amount).
  3. Branch on the decision:
     - release_to_operator > 0 + refund > 0  → release_escrow_split
       with the operator's release amount; PaymentEscrow auto-
       refunds remainder to requester (sprint-318 behavior).
     - release > 0 + refund == 0             → release_escrow
       (full payout; existing escrow-flow path)
     - release == 0 + refund > 0             → refund_escrow
       (no work credited; operator gets nothing)
  4. Returns the decision so callers can act on should_slash.

Slash-trigger emission is OUT OF SCOPE for sprint 783 — caller
inspects decision.should_slash and routes to whatever slash hook
exists in their flow (currently logged; on-chain slash emission
is a separate arc).

Pin tests:
- Wrapper exists.
- Full-success path → release_escrow called, not split, not refund.
- Partial-completion path → release_escrow_split called with
  proportional amount; split-API handles the remainder refund.
- Zero-completion path → refund_escrow called (full refund).
- escrow not found → raises a typed error or returns decision
  with zero release/refund (caller can detect).
- Returned SettlementDecision matches what sprint 782 computed.
- should_slash propagated unchanged from sprint 782 decision.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
import asyncio

import pytest


def _make_receipt(**overrides):
    from prsm.compute.inference.models import (
        InferenceReceipt,
        ContentTier,
    )
    from prsm.compute.tee.models import PrivacyLevel, TEEType
    defaults = dict(
        job_id="j1",
        request_id="r1",
        model_id="gpt2",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=b"att",
        output_hash=b"\xde\xad",
        duration_seconds=1.0,
        cost_ftns=Decimal("1.0"),
        settler_node_id="s1",
    )
    defaults.update(overrides)
    return InferenceReceipt(**defaults)


def _make_partial(reason="preempted", completed=7, requested=10):
    from prsm.compute.inference.partial_completion import (
        PartialCompletionInfo,
    )
    return PartialCompletionInfo(
        reason=reason,
        tokens_completed=completed,
        tokens_requested=requested,
        timestamp="2026-05-23T12:00:00Z",
    )


def _make_escrow_mock(amount: float):
    """Build a PaymentEscrow stand-in with the methods we call."""
    esc = MagicMock()
    esc.release_escrow = AsyncMock(return_value=MagicMock())
    esc.release_escrow_split = AsyncMock(return_value=[MagicMock()])
    esc.refund_escrow = AsyncMock(return_value=True)
    esc.get_escrow_amount = MagicMock(return_value=amount)
    return esc


# ---- Wrapper exists ---------------------------------------------


def test_wrapper_exists():
    from prsm.economy.credit_policy import settle_inference_receipt
    assert callable(settle_inference_receipt)


# ---- Branch dispatch --------------------------------------------


def test_full_success_calls_release_escrow_not_split():
    from prsm.economy.credit_policy import settle_inference_receipt
    esc = _make_escrow_mock(amount=1.0)
    r = _make_receipt(cost_ftns=Decimal("1.0"))
    d = asyncio.run(settle_inference_receipt(
        payment_escrow=esc, receipt=r,
        operator_id="op1", job_id="j1",
    ))
    esc.release_escrow.assert_awaited_once_with("j1", "op1")
    esc.release_escrow_split.assert_not_called()
    esc.refund_escrow.assert_not_called()
    assert d.release_to_operator == Decimal("1.0")
    assert d.refund_to_payer == Decimal("0")


def test_partial_completion_calls_release_escrow_split():
    """Partial = release proportional + auto-refund remainder via
    release_escrow_split's existing behavior."""
    from prsm.economy.credit_policy import settle_inference_receipt
    esc = _make_escrow_mock(amount=1.0)
    r = _make_receipt(
        cost_ftns=Decimal("1.0"),
        partial_completion=_make_partial(completed=7, requested=10),
    )
    d = asyncio.run(settle_inference_receipt(
        payment_escrow=esc, receipt=r,
        operator_id="op1", job_id="j1",
    ))
    esc.release_escrow_split.assert_awaited_once()
    call = esc.release_escrow_split.call_args
    # First positional or keyword: job_id
    assert call.args[0] == "j1" or call.kwargs.get("job_id") == "j1"
    # Splits arg
    splits = call.args[1] if len(call.args) > 1 else call.kwargs["splits"]
    assert len(splits) == 1
    recipient, amount = splits[0]
    assert recipient == "op1"
    assert Decimal(str(amount)) == Decimal("0.7")
    esc.release_escrow.assert_not_called()
    esc.refund_escrow.assert_not_called()
    assert d.release_to_operator == Decimal("0.7")
    assert d.refund_to_payer == Decimal("0.3")


def test_zero_completion_calls_refund_escrow():
    """No work done → full refund. Don't even call
    release_escrow_split with amount=0 (that's invalid)."""
    from prsm.economy.credit_policy import settle_inference_receipt
    esc = _make_escrow_mock(amount=1.0)
    r = _make_receipt(
        cost_ftns=Decimal("1.0"),
        partial_completion=_make_partial(completed=0, requested=10),
    )
    d = asyncio.run(settle_inference_receipt(
        payment_escrow=esc, receipt=r,
        operator_id="op1", job_id="j1",
    ))
    esc.refund_escrow.assert_awaited_once()
    esc.release_escrow.assert_not_called()
    esc.release_escrow_split.assert_not_called()
    assert d.release_to_operator == Decimal("0")
    assert d.refund_to_payer == Decimal("1.0")


# ---- Decision returned ------------------------------------------


def test_returns_decision_with_slash_propagated():
    """Decision returned to caller carries should_slash so the
    caller can dispatch slash-emission logic."""
    from prsm.economy.credit_policy import settle_inference_receipt
    esc = _make_escrow_mock(amount=1.0)
    r = _make_receipt(
        cost_ftns=Decimal("1.0"),
        partial_completion=_make_partial(
            reason="error", completed=4, requested=10,
        ),
    )
    d = asyncio.run(settle_inference_receipt(
        payment_escrow=esc, receipt=r,
        operator_id="op1", job_id="j1",
    ))
    assert d.should_slash is True
    assert d.release_to_operator == Decimal("0.4")
    assert d.refund_to_payer == Decimal("0.6")


def test_returns_decision_preempted_no_slash():
    from prsm.economy.credit_policy import settle_inference_receipt
    esc = _make_escrow_mock(amount=1.0)
    r = _make_receipt(
        cost_ftns=Decimal("1.0"),
        partial_completion=_make_partial(
            reason="preempted", completed=5, requested=10,
        ),
    )
    d = asyncio.run(settle_inference_receipt(
        payment_escrow=esc, receipt=r,
        operator_id="op1", job_id="j1",
    ))
    assert d.should_slash is False


# ---- Over-funded escrow (escrow > cost) -------------------------


def test_over_funded_escrow_amount_propagated():
    """When the escrow amount differs from receipt.cost_ftns
    (over-funded / promotional credit), the caller passes the
    actual locked amount via escrow_amount kwarg — the wrapper
    feeds it through to compute_settlement_for_receipt."""
    from prsm.economy.credit_policy import settle_inference_receipt
    esc = _make_escrow_mock(amount=2.0)
    r = _make_receipt(cost_ftns=Decimal("1.0"))
    d = asyncio.run(settle_inference_receipt(
        payment_escrow=esc, receipt=r,
        operator_id="op1", job_id="j1",
        escrow_amount=Decimal("2.0"),
    ))
    # Full success on a 1.0-priced receipt with 2.0 locked:
    # release 1.0, refund 1.0 → split path (release + refund
    # both > 0).
    assert d.release_to_operator == Decimal("1.0")
    assert d.refund_to_payer == Decimal("1.0")
    esc.release_escrow_split.assert_awaited_once()


# ---- sp1374: multi-stage defers the on-chain broadcast --------------------

def _make_multistage_receipt(operator_id):
    """A cross-host receipt whose signed topology names 2 distinct stage nodes
    (head=operator + a worker) — so split_release_across_stages fans out."""
    from prsm.compute.inference.topology_rotation import TopologyAssignment
    topo = TopologyAssignment(
        positions={(0, 0): operator_id, (1, 0): "worker2"},
        stage_count=2, slots_per_stage=1)
    return _make_receipt(
        settler_node_id=operator_id, topology_assignment=topo,
        cost_ftns=Decimal("1.0"))


def test_sp1374_multistage_on_defers_onchain_broadcast(monkeypatch):
    # PRSM_MULTISTAGE_SETTLEMENT on: the per-stage escrow commit (sp1324/1322)
    # settles the stage shares on-chain via the registry, so the off-chain split
    # here must NOT broadcast — else the stage nodes are DOUBLE-PAID.
    from prsm.economy.credit_policy import settle_inference_receipt
    monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", "1")
    esc = _make_escrow_mock(amount=1.0)
    r = _make_multistage_receipt("op1")
    asyncio.run(settle_inference_receipt(
        payment_escrow=esc, receipt=r, operator_id="op1", job_id="j1"))
    esc.release_escrow_split.assert_awaited_once()
    assert esc.release_escrow_split.call_args.kwargs.get("broadcast") is False


def test_sp1374_multistage_off_still_broadcasts(monkeypatch):
    # multistage OFF: the off-chain broadcast IS the on-chain settlement — unchanged.
    from prsm.economy.credit_policy import settle_inference_receipt
    monkeypatch.delenv("PRSM_MULTISTAGE_SETTLEMENT", raising=False)
    esc = _make_escrow_mock(amount=1.0)
    r = _make_multistage_receipt("op1")
    asyncio.run(settle_inference_receipt(
        payment_escrow=esc, receipt=r, operator_id="op1", job_id="j1"))
    esc.release_escrow_split.assert_awaited_once()
    assert esc.release_escrow_split.call_args.kwargs.get("broadcast", True) is True
