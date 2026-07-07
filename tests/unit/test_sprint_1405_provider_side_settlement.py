"""Sprint 1405 — on-chain settlement is PROVIDER-side: the EARNER accumulates + commits.

Correctness fix caught during the ceremony §0 verify: commitBatch sets provider = msg.sender and the
commit client asserts committer_key_address == provider_address. So the node that SERVED the job (the
earner) must accumulate + commit — NOT the requester (payer). sp1403 had wired it on the requester,
which would set the payer as on-chain earner (and fail the assertion). Now compute_provider, on serving
a REMOTE job, accumulates a BatchedReceipt with provider_address = its OWN operator address and the
requester's payer address (carried in the offer).
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.compute_provider import ComputeJob, ComputeProvider, JobType
from prsm.node.gossip import GossipProtocol
from prsm.node.identity import generate_node_identity
from prsm.node.local_ledger import LocalLedger
from prsm.node.transport import WebSocketTransport


async def _provider(*, client=True, op_addr="0xPROV"):
    identity = generate_node_identity("prov")
    transport = WebSocketTransport(identity, host="127.0.0.1", port=19405)
    gossip = GossipProtocol(transport, fanout=3, heartbeat_interval=9999)
    ledger = LocalLedger(":memory:")
    await ledger.initialize()
    p = ComputeProvider(identity=identity, transport=transport, gossip=gossip, ledger=ledger,
                        max_concurrent_jobs=2)
    p.settlement_client = MagicMock() if client else None
    if client:
        p.settlement_client.accumulate = AsyncMock()
    p.operator_address = op_addr
    return p


def _served_job(p, *, requester="reqNODE", payer="0xPAYER"):
    job = ComputeJob(job_id="jx", job_type=JobType.INFERENCE, requester_id=requester,
                     payload={}, ftns_budget=2.0)
    job.requester_operator_address = payer
    job.result = {"response": "hi", "shard_receipt": p._signed_shard_receipt(job, "hi")}
    return job


@pytest.mark.asyncio
async def test_provider_accumulates_its_own_earning():
    p = await _provider()
    await p._maybe_accumulate_onchain_earning(_served_job(p))
    p.settlement_client.accumulate.assert_awaited_once()
    br = p.settlement_client.accumulate.call_args[0][0]
    assert br.provider_address == "0xPROV"        # THIS node earns (== its settler key)
    assert br.requester_address == "0xPAYER"      # the payer, from the offer
    assert br.value_ftns == 2 * 10 ** 18
    assert br.local_escrow_id == "job-jx"
    assert br.receipt.provider_id == p.identity.node_id


@pytest.mark.asyncio
async def test_noop_when_ungated():
    # no settlement client
    p = await _provider(client=False)
    await p._maybe_accumulate_onchain_earning(_served_job(p))     # must not raise
    # client present but no payer address in the offer → no accumulate
    p2 = await _provider()
    job = _served_job(p2, payer="")
    await p2._maybe_accumulate_onchain_earning(job)
    p2.settlement_client.accumulate.assert_not_called()
    # no operator_address on this node → no accumulate
    p3 = await _provider(op_addr="")
    await p3._maybe_accumulate_onchain_earning(_served_job(p3))
    p3.settlement_client.accumulate.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
