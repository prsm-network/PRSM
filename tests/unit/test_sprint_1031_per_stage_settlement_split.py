"""Sprint 1031 — per-stage settlement split across cross-host topology nodes.

The Tier-1 cross-host inference bench (2026-06-06) proved gpt2 splitting 6+6
across two A10 hosts with a signed receipt naming each stage node in
``topology_assignment``. But settlement (``settle_inference_receipt``) credited
only ONE payee — the local settler/head node — so the second host, which ran
half the model, was paid nothing. That is a money-correctness bug on the
cross-host path.

This sprint distributes the release amount EQUALLY across the distinct stage
node_ids carried (and cryptographically signed) in ``receipt.topology_assignment``,
reusing the existing lock-safe ``PaymentEscrow.release_escrow_split`` primitive.

Invariants pinned here:
- The split amounts sum EXACTLY to the SettlementDecision's release (no FTNS
  created or lost; the total released is unchanged — only the distribution).
- Single-node / no-topology receipts are UNCHANGED (single-payee path), so the
  default deployment sees zero behavior change.
- A receipt naming < 2 distinct nodes (or malformed/absent topology) falls back
  to single-payee.
- A zero-release (full-refund) job never splits — it refunds.
- The worker node (not just the head) is actually credited — the bug fix.

Equal-weight policy: layer-balanced stages do comparable work, so the release
splits evenly across the distinct nodes that filled a position. Per-node FLOP /
priced weighting + on-chain cross-node settlement are explicit follow-ons.
"""
from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock


def _make_receipt(**overrides):
    from prsm.compute.inference.models import InferenceReceipt, ContentTier
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
        settler_node_id="nodeA",
    )
    defaults.update(overrides)
    return InferenceReceipt(**defaults)


def _make_partial(reason="preempted", completed=7, requested=10):
    from prsm.compute.inference.partial_completion import PartialCompletionInfo
    return PartialCompletionInfo(
        reason=reason,
        tokens_completed=completed,
        tokens_requested=requested,
        timestamp="2026-06-06T12:00:00Z",
    )


def _topology(*node_ids_by_stage):
    """Build a TopologyAssignment with one slot per stage; arg i = stage i's
    node_id."""
    from prsm.compute.inference.topology_rotation import TopologyAssignment
    positions = {
        (stage, 0): node_id for stage, node_id in enumerate(node_ids_by_stage)
    }
    return TopologyAssignment(
        positions=positions,
        stage_count=len(node_ids_by_stage),
        slots_per_stage=1,
    )


def _make_escrow_mock(amount: float):
    esc = MagicMock()
    esc.release_escrow = AsyncMock(return_value=MagicMock())
    esc.release_escrow_split = AsyncMock(return_value=[MagicMock()])
    esc.refund_escrow = AsyncMock(return_value=True)
    esc.get_escrow_amount = MagicMock(return_value=amount)
    return esc


def _splits_from_call(mock_call):
    job_id = mock_call.args[0] if mock_call.args else mock_call.kwargs.get("job_id")
    splits = (
        mock_call.args[1]
        if len(mock_call.args) > 1
        else mock_call.kwargs["splits"]
    )
    return job_id, splits


# ── The pure helper ───────────────────────────────────────────────────────────

def test_helper_none_when_no_topology():
    from prsm.economy.credit_policy import split_release_across_stages
    r = _make_receipt()  # no topology_assignment
    assert split_release_across_stages(r, Decimal("1.0")) is None


def test_helper_none_when_single_distinct_node():
    from prsm.economy.credit_policy import split_release_across_stages
    # same node fills both stages → only 1 distinct payee → single-payee path
    r = _make_receipt(topology_assignment=_topology("nodeA", "nodeA"))
    assert split_release_across_stages(r, Decimal("1.0")) is None


def test_helper_none_when_malformed_positions():
    from prsm.economy.credit_policy import split_release_across_stages
    bad = MagicMock()
    bad.positions = "not-a-dict"
    r = _make_receipt(topology_assignment=bad)
    assert split_release_across_stages(r, Decimal("1.0")) is None


def test_helper_splits_two_nodes_equally_exact_sum():
    from prsm.economy.credit_policy import split_release_across_stages
    r = _make_receipt(topology_assignment=_topology("nodeA", "nodeB"))
    splits = split_release_across_stages(r, Decimal("1.0"))
    assert splits is not None
    assert {nid for nid, _ in splits} == {"nodeA", "nodeB"}
    assert sum((amt for _, amt in splits), Decimal("0")) == Decimal("1.0")
    assert all(amt == Decimal("0.5") for _, amt in splits)


def test_helper_three_nodes_sums_exactly_to_release():
    """1.0 / 3 does not terminate — the split must still sum EXACTLY to the
    release (drift absorbed, no FTNS created/lost)."""
    from prsm.economy.credit_policy import split_release_across_stages
    r = _make_receipt(topology_assignment=_topology("a", "b", "c"))
    splits = split_release_across_stages(r, Decimal("1.0"))
    assert splits is not None
    assert len(splits) == 3
    assert sum((amt for _, amt in splits), Decimal("0")) == Decimal("1.0")
    # every share is positive and within a hair of 1/3
    for _, amt in splits:
        assert amt > Decimal("0.33") and amt < Decimal("0.34")


def test_helper_dedups_distinct_nodes_preserving_order():
    from prsm.economy.credit_policy import split_release_across_stages
    # 4 positions, 2 distinct nodes (A on stages 0,2; B on 1,3)
    r = _make_receipt(topology_assignment=_topology("A", "B", "A", "B"))
    splits = split_release_across_stages(r, Decimal("1.0"))
    assert splits is not None
    assert [nid for nid, _ in splits] == ["A", "B"]
    assert sum((amt for _, amt in splits), Decimal("0")) == Decimal("1.0")


# ── settle_inference_receipt integration ───────────────────────────────────────

def test_multistage_full_success_splits_across_stage_nodes():
    from prsm.economy.credit_policy import settle_inference_receipt
    esc = _make_escrow_mock(amount=1.0)
    r = _make_receipt(
        cost_ftns=Decimal("1.0"),
        topology_assignment=_topology("nodeA", "nodeB"),
    )
    d = asyncio.run(settle_inference_receipt(
        payment_escrow=esc, receipt=r, operator_id="nodeA", job_id="j1",
    ))
    esc.release_escrow_split.assert_awaited_once()
    esc.release_escrow.assert_not_called()
    esc.refund_escrow.assert_not_called()
    job_id, splits = _splits_from_call(esc.release_escrow_split.call_args)
    assert job_id == "j1"
    recipients = {nid for nid, _ in splits}
    assert recipients == {"nodeA", "nodeB"}
    # the worker (nodeB) is actually paid — the bug fix
    assert "nodeB" in recipients
    total = sum((Decimal(str(amt)) for _, amt in splits), Decimal("0"))
    assert total == Decimal("1.0")
    assert d.release_to_operator == Decimal("1.0")


def test_multistage_partial_splits_proportional_release():
    from prsm.economy.credit_policy import settle_inference_receipt
    esc = _make_escrow_mock(amount=1.0)
    r = _make_receipt(
        cost_ftns=Decimal("1.0"),
        topology_assignment=_topology("nodeA", "nodeB"),
        partial_completion=_make_partial(completed=7, requested=10),
    )
    d = asyncio.run(settle_inference_receipt(
        payment_escrow=esc, receipt=r, operator_id="nodeA", job_id="j1",
    ))
    esc.release_escrow_split.assert_awaited_once()
    _, splits = _splits_from_call(esc.release_escrow_split.call_args)
    assert {nid for nid, _ in splits} == {"nodeA", "nodeB"}
    total = sum((Decimal(str(amt)) for _, amt in splits), Decimal("0"))
    # release is the proportional 0.7; split sums to it (0.3 auto-refunded)
    assert total == Decimal("0.7")
    assert d.release_to_operator == Decimal("0.7")
    assert d.refund_to_payer == Decimal("0.3")


def test_zero_release_with_topology_refunds_no_split():
    from prsm.economy.credit_policy import settle_inference_receipt
    esc = _make_escrow_mock(amount=1.0)
    r = _make_receipt(
        cost_ftns=Decimal("1.0"),
        topology_assignment=_topology("nodeA", "nodeB"),
        partial_completion=_make_partial(completed=0, requested=10),
    )
    d = asyncio.run(settle_inference_receipt(
        payment_escrow=esc, receipt=r, operator_id="nodeA", job_id="j1",
    ))
    esc.refund_escrow.assert_awaited_once()
    esc.release_escrow_split.assert_not_called()
    esc.release_escrow.assert_not_called()
    assert d.release_to_operator == Decimal("0")


def test_single_node_path_unchanged_release_escrow():
    """Regression: a no-topology receipt keeps the exact single-payee
    behavior (full success → release_escrow, not split)."""
    from prsm.economy.credit_policy import settle_inference_receipt
    esc = _make_escrow_mock(amount=1.0)
    r = _make_receipt(cost_ftns=Decimal("1.0"))  # no topology
    asyncio.run(settle_inference_receipt(
        payment_escrow=esc, receipt=r, operator_id="op1", job_id="j1",
    ))
    esc.release_escrow.assert_awaited_once_with("j1", "op1")
    esc.release_escrow_split.assert_not_called()


def test_single_distinct_node_falls_back_to_single_payee():
    """topology naming only the head node (1 distinct) → no fan-out."""
    from prsm.economy.credit_policy import settle_inference_receipt
    esc = _make_escrow_mock(amount=1.0)
    r = _make_receipt(
        cost_ftns=Decimal("1.0"),
        topology_assignment=_topology("nodeA", "nodeA"),
    )
    asyncio.run(settle_inference_receipt(
        payment_escrow=esc, receipt=r, operator_id="nodeA", job_id="j1",
    ))
    esc.release_escrow.assert_awaited_once_with("j1", "nodeA")
    esc.release_escrow_split.assert_not_called()


def test_topology_split_skipped_when_not_locally_settled():
    """Defense-in-depth: the topology fan-out only fires when the receipt was
    settled by THIS operator (settler_node_id == operator_id, i.e. locally
    built + signed). A receipt settled by a different node falls back to the
    single-payee path so a remote-supplied topology can't be paid out if a
    future caller ever feeds a remote receipt into settlement."""
    from prsm.economy.credit_policy import settle_inference_receipt
    esc = _make_escrow_mock(amount=1.0)
    r = _make_receipt(
        cost_ftns=Decimal("1.0"),
        settler_node_id="remoteX",  # NOT the operator
        topology_assignment=_topology("nodeA", "nodeB"),
    )
    asyncio.run(settle_inference_receipt(
        payment_escrow=esc, receipt=r, operator_id="nodeA", job_id="j1",
    ))
    esc.release_escrow.assert_awaited_once_with("j1", "nodeA")
    esc.release_escrow_split.assert_not_called()


def test_stage_splits_resolved_through_wallet_map(tmp_path, monkeypatch):
    """Parity with the swarm/forge split path: each stage node_id is resolved
    through ComputeWalletMap so a multi-agent operator's per-stage credits
    consolidate onto the registered FTNS wallet (unmapped ids fall through)."""
    import json
    from prsm.economy.credit_policy import settle_inference_receipt
    mapfile = tmp_path / "wallets.json"
    mapfile.write_text(json.dumps({"nodeB": "walletB"}))
    monkeypatch.setenv("PRSM_COMPUTE_WALLET_MAP_FILE", str(mapfile))
    esc = _make_escrow_mock(amount=1.0)
    r = _make_receipt(
        cost_ftns=Decimal("1.0"),
        settler_node_id="nodeA",
        topology_assignment=_topology("nodeA", "nodeB"),
    )
    asyncio.run(settle_inference_receipt(
        payment_escrow=esc, receipt=r, operator_id="nodeA", job_id="j1",
    ))
    _, splits = _splits_from_call(esc.release_escrow_split.call_args)
    recipients = {nid for nid, _ in splits}
    assert "walletB" in recipients   # nodeB resolved to its wallet
    assert "nodeA" in recipients     # nodeA unmapped → itself
    assert "nodeB" not in recipients
    total = sum((Decimal(str(amt)) for _, amt in splits), Decimal("0"))
    assert total == Decimal("1.0")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
