"""Sprint 1037 — accumulate_settled_inference_receipt (brick 1.5).

Wires the just-settled InferenceReceipt into the on-chain settlement accumulator
(node._onchain_settlement_client from sprint 1036) via the sprint-1035 adapter.
Called from the /compute/inference settle path AFTER the off-chain escrow
release. FAIL-OPEN: never raises, never unwinds the completed off-chain
settlement.

Self-escrow note: today's /compute/inference escrows the requester as the local
node, so requester_address defaults to provider_address (a self-settlement); a
distinct paying requester is a future API change.
"""
from __future__ import annotations

import asyncio
import hashlib
from decimal import Decimal
from unittest.mock import AsyncMock

_ADDR = "0x" + "11" * 20


def _ident(name="n1"):
    from prsm.node.identity import generate_node_identity
    return generate_node_identity(display_name=name)


def _make_ir(identity, **over):
    from prsm.compute.inference.models import InferenceReceipt, ContentTier
    from prsm.compute.tee.models import PrivacyLevel, TEEType
    defaults = dict(
        job_id="job-1", request_id="req-1", model_id="gpt2",
        content_tier=ContentTier.A, privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0, tee_type=TEEType.SOFTWARE, tee_attestation=b"att",
        output_hash=hashlib.sha256(b"out").digest(), duration_seconds=1.0,
        cost_ftns=Decimal("1.0"), settler_signature=b"sig",
        settler_node_id=identity.node_id,
    )
    defaults.update(over)
    return InferenceReceipt(**defaults)


def _client():
    c = AsyncMock()
    c.accumulate = AsyncMock()
    return c


def _run(**kw):
    from prsm.settlement.client_wiring import (
        accumulate_settled_inference_receipt,
    )
    return asyncio.run(accumulate_settled_inference_receipt(**kw))


def test_accumulates_a_batched_receipt():
    ident, c = _ident(), _client()
    status = _run(
        client=c, identity=ident, provider_address=_ADDR,
        receipt=_make_ir(ident), release_ftns=Decimal("1.0"), job_id="job-1",
    )
    assert status == "accumulated"
    c.accumulate.assert_awaited_once()
    br = c.accumulate.call_args.args[0]
    assert br.value_ftns == 10 ** 18                 # 1 FTNS -> wei
    assert br.requester_address == _ADDR             # self-escrow: requester==provider
    assert br.provider_address == _ADDR
    assert br.local_escrow_id == "job-1"
    assert br.receipt.provider_id == ident.node_id


def test_skips_when_no_client():
    ident = _ident()
    assert _run(
        client=None, identity=ident, provider_address=_ADDR,
        receipt=_make_ir(ident), release_ftns=Decimal("1"), job_id="j",
    ) == "skipped:no-client"


def test_skips_when_no_provider_address():
    ident, c = _ident(), _client()
    assert _run(
        client=c, identity=ident, provider_address="",
        receipt=_make_ir(ident), release_ftns=Decimal("1"), job_id="j",
    ) == "skipped:no-provider-address"
    c.accumulate.assert_not_awaited()


def test_skips_zero_release():
    ident, c = _ident(), _client()
    assert _run(
        client=c, identity=ident, provider_address=_ADDR,
        receipt=_make_ir(ident), release_ftns=Decimal("0"), job_id="j",
    ) == "skipped:zero-release"
    c.accumulate.assert_not_awaited()


def test_skips_sub_wei_release():
    """A release below 1 wei (1e-18 FTNS) rounds to 0 wei → don't post a 0-value
    receipt."""
    ident, c = _ident(), _client()
    assert _run(
        client=c, identity=ident, provider_address=_ADDR,
        receipt=_make_ir(ident), release_ftns=Decimal("1E-19"), job_id="j",
    ) == "skipped:sub-wei"
    c.accumulate.assert_not_awaited()


def test_value_wei_conversion_exact():
    ident, c = _ident(), _client()
    _run(
        client=c, identity=ident, provider_address=_ADDR,
        receipt=_make_ir(ident), release_ftns=Decimal("2.5"), job_id="j",
    )
    assert c.accumulate.call_args.args[0].value_ftns == 2_500_000_000_000_000_000


def test_distinct_requester_address_used():
    ident, c = _ident(), _client()
    req = "0x" + "99" * 20
    _run(
        client=c, identity=ident, provider_address=_ADDR, requester_address=req,
        receipt=_make_ir(ident), release_ftns=Decimal("1"), job_id="j",
    )
    br = c.accumulate.call_args.args[0]
    assert br.requester_address == req and br.provider_address == _ADDR


def test_fail_open_on_accumulate_error():
    """An accumulate failure must NOT raise (it must never unwind the completed
    off-chain settlement)."""
    ident = _ident()
    c = _client()
    c.accumulate = AsyncMock(side_effect=RuntimeError("boom"))
    status = _run(
        client=c, identity=ident, provider_address=_ADDR,
        receipt=_make_ir(ident), release_ftns=Decimal("1"), job_id="j",
    )
    assert status.startswith("error:")


def test_skips_multi_stage_receipt_deferred():
    """A cross-host (>=2 distinct topology nodes) receipt must NOT be
    single-payee accumulated: the off-chain settle split it N ways (sprint
    1031), so booking the FULL amount to one provider on-chain would
    over-attribute. Per-stage on-chain accumulation is brick 2."""
    from prsm.compute.inference.topology_rotation import TopologyAssignment
    ident, c = _ident(), _client()
    topo = TopologyAssignment(
        positions={(0, 0): ident.node_id, (1, 0): "otherworker0000000000000000000000"},
        stage_count=2, slots_per_stage=1,
    )
    status = _run(
        client=c, identity=ident, provider_address=_ADDR,
        receipt=_make_ir(ident, topology_assignment=topo),
        release_ftns=Decimal("1.0"), job_id="j",
    )
    assert status == "skipped:multi-stage-deferred"
    c.accumulate.assert_not_awaited()


def test_single_node_no_topology_still_accumulates():
    """Regression: a no-topology (single-node) receipt still accumulates."""
    ident, c = _ident(), _client()
    status = _run(
        client=c, identity=ident, provider_address=_ADDR,
        receipt=_make_ir(ident), release_ftns=Decimal("1.0"), job_id="j",
    )
    assert status == "accumulated"


def test_default_executed_at_unix_is_positive():
    ident, c = _ident(), _client()
    _run(
        client=c, identity=ident, provider_address=_ADDR,
        receipt=_make_ir(ident), release_ftns=Decimal("1"), job_id="j",
    )
    assert c.accumulate.call_args.args[0].receipt.executed_at_unix > 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
