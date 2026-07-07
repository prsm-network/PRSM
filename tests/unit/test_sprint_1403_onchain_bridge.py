"""Sprint 1403 — bridge a cross-node gossip job to ON-CHAIN settlement.

The provider emits a signed ShardExecutionReceipt in its result; the requester, on settling the job,
hands a BatchedReceipt (receipt + eth addresses + escrowed value) to the on-chain ReceiptAccumulator
so a funded settler key commits the batch. Fully gated: no client / no operator addresses / no receipt
→ no-op (off-chain settlement from sp1401/1402 stands regardless).
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.compute.shard_receipt import build_receipt_signing_payload
from prsm.node.compute_provider import ComputeJob, ComputeProvider, JobType
from prsm.node.compute_requester import ComputeRequester, SubmittedJob
from prsm.node.gossip import GossipProtocol
from prsm.node.identity import generate_node_identity, verify_signature
from prsm.node.local_ledger import LocalLedger
from prsm.node.transport import WebSocketTransport


async def _provider():
    identity = generate_node_identity("prov")
    transport = WebSocketTransport(identity, host="127.0.0.1", port=19403)
    gossip = GossipProtocol(transport, fanout=3, heartbeat_interval=9999)
    ledger = LocalLedger(":memory:")
    await ledger.initialize()
    return ComputeProvider(identity=identity, transport=transport, gossip=gossip, ledger=ledger,
                           max_concurrent_jobs=2)


@pytest.mark.asyncio
async def test_provider_emits_verifiable_shard_receipt():
    p = await _provider()
    job = ComputeJob(job_id="j1", job_type=JobType.INFERENCE, requester_id="r",
                     payload={}, ftns_budget=1.0)
    rd = p._signed_shard_receipt(job, " Paris")
    assert rd["job_id"] == "j1" and rd["shard_index"] == 0
    assert rd["provider_id"] == p.identity.node_id
    # the signature verifies against the canonical signing payload → a settleable receipt
    payload = build_receipt_signing_payload(
        job_id="j1", shard_index=0, output_hash=rd["output_hash"],
        executed_at_unix=rd["executed_at_unix"])
    assert verify_signature(rd["provider_pubkey_b64"], payload, rd["signature"])


async def _requester():
    identity = generate_node_identity("req")
    transport = WebSocketTransport(identity, host="127.0.0.1", port=19404)
    gossip = GossipProtocol(transport, fanout=3, heartbeat_interval=9999)
    ledger = LocalLedger(":memory:")
    await ledger.initialize()
    return ComputeRequester(identity=identity, transport=transport, gossip=gossip, ledger=ledger)


def _wire(req, *, client=True, req_addr="0xREQ", prov_addr="0xPROV"):
    req.settlement_client = MagicMock() if client else None
    if client:
        req.settlement_client.accumulate = AsyncMock()
    req.operator_address = req_addr
    disc = MagicMock()
    peer = MagicMock()
    peer.hardware_profile = {"operator_address": prov_addr} if prov_addr else {}
    disc.known_peers = {"provP": peer}
    req.discovery = disc


_RESULT = {"response": "hi", "shard_receipt": {"job_id": "jx", "shard_index": 0,
           "provider_id": "provP", "provider_pubkey_b64": "PUB", "output_hash": "abc",
           "executed_at_unix": 1, "signature": "SIG"}}


def _job():
    j = SubmittedJob(job_id="jx", job_type=JobType.INFERENCE, payload={}, ftns_budget=2.0)
    j.escrow_id = "esc-1"
    return j


@pytest.mark.asyncio
async def test_requester_accumulates_onchain_when_wired():
    req = await _requester()
    _wire(req)
    await req._maybe_accumulate_onchain(_job(), "provP", _RESULT)
    req.settlement_client.accumulate.assert_awaited_once()
    br = req.settlement_client.accumulate.call_args[0][0]
    assert br.provider_address == "0xPROV" and br.requester_address == "0xREQ"
    assert br.value_ftns == 2 * 10 ** 18            # 2.0 FTNS in wei
    assert br.local_escrow_id == "esc-1"
    assert br.receipt.job_id == "jx"


@pytest.mark.asyncio
async def test_noop_when_ungated():
    # no settlement client
    req = await _requester()
    _wire(req, client=False)
    await req._maybe_accumulate_onchain(_job(), "provP", _RESULT)   # must not raise
    # client present but no provider operator address → no accumulate
    req2 = await _requester()
    _wire(req2, prov_addr="")
    await req2._maybe_accumulate_onchain(_job(), "provP", _RESULT)
    req2.settlement_client.accumulate.assert_not_called()
    # no shard_receipt in result → no accumulate
    req3 = await _requester()
    _wire(req3)
    await req3._maybe_accumulate_onchain(_job(), "provP", {"response": "hi"})
    req3.settlement_client.accumulate.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
