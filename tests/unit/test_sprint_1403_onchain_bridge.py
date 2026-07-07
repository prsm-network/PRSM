"""Sprint 1403 — the provider emits a signed ShardExecutionReceipt in its result (settlement metadata).

The receipt is carried in the local_inference result. On-chain accumulation moved PROVIDER-side in
sp1405 (the earner commits: commitBatch → provider=msg.sender); see test_sprint_1405. This keeps the
receipt-emission check.
"""
import pytest

from prsm.compute.shard_receipt import build_receipt_signing_payload
from prsm.node.compute_provider import ComputeJob, ComputeProvider, JobType
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
    payload = build_receipt_signing_payload(
        job_id="j1", shard_index=0, output_hash=rd["output_hash"],
        executed_at_unix=rd["executed_at_unix"])
    assert verify_signature(rd["provider_pubkey_b64"], payload, rd["signature"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
