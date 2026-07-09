"""Tests for single-node self-compute: a node executing its own jobs when no peers are connected."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.compute_provider import ComputeProvider, JobStatus, JobType


def _make_identity(node_id: str = "node-self"):
    identity = SimpleNamespace(
        node_id=node_id,
        public_key_b64="dGVzdHB1YmtleQ==",
    )
    identity.sign = lambda data: "fakesig"
    return identity


def _make_transport(peer_count: int = 0):
    transport = SimpleNamespace(peer_count=peer_count)
    return transport


def _make_gossip():
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    gossip.publish = AsyncMock()
    return gossip


def _make_ledger(balance: float = 100.0):
    ledger = MagicMock()
    ledger.get_balance = AsyncMock(return_value=balance)
    ledger.credit = AsyncMock(return_value=SimpleNamespace(
        tx_id="tx-test", from_wallet="system", to_wallet="node-self",
        amount=1.0, tx_type=SimpleNamespace(value="compute_earning"),
        description="test", timestamp=time.time(),
    ))
    return ledger


def _make_provider(
    node_id: str = "node-self",
    peer_count: int = 0,
    allow_self_compute: bool = True,
) -> ComputeProvider:
    identity = _make_identity(node_id)
    transport = _make_transport(peer_count)
    gossip = _make_gossip()
    ledger = _make_ledger()

    provider = ComputeProvider(
        identity=identity,
        transport=transport,
        gossip=gossip,
        ledger=ledger,
    )
    provider.allow_self_compute = allow_self_compute
    provider._running = True
    return provider


# ── Self-compute behavior tests ──────────────────────────────────


@pytest.mark.asyncio
async def test_self_compute_enabled_no_peers_accepts_own_job():
    """When self-compute is enabled and no peers, provider accepts its own job."""
    provider = _make_provider(node_id="node-self", peer_count=0, allow_self_compute=True)

    job_data = {
        "job_id": "job-001",
        "job_type": "benchmark",
        "ftns_budget": 1.0,
        "requester_id": "node-self",  # Same as provider identity
        "payload": {"iterations": 1000},
    }

    await provider._on_job_offer("job_offer", job_data, "node-self")

    # Job should be accepted and moved to active/completed
    # Give the background execution task time to run
    await asyncio.sleep(0.5)

    assert "job-001" in provider.active_jobs or "job-001" in provider.completed_jobs
    # Gossip should have published an accept and a result
    assert provider.gossip.publish.call_count >= 1


@pytest.mark.asyncio
async def test_self_compute_enabled_with_peers_rejects_own_job():
    """When self-compute is enabled but peers exist, provider rejects its own job."""
    provider = _make_provider(node_id="node-self", peer_count=3, allow_self_compute=True)

    job_data = {
        "job_id": "job-002",
        "job_type": "benchmark",
        "ftns_budget": 1.0,
        "requester_id": "node-self",
        "payload": {},
    }

    await provider._on_job_offer("job_offer", job_data, "node-self")

    # Job should NOT be accepted (peers available — let the network handle it)
    assert "job-002" not in provider.active_jobs
    assert "job-002" not in provider.completed_jobs


@pytest.mark.asyncio
async def test_self_compute_disabled_rejects_own_job():
    """When self-compute is disabled, provider always rejects its own jobs."""
    provider = _make_provider(node_id="node-self", peer_count=0, allow_self_compute=False)

    job_data = {
        "job_id": "job-003",
        "job_type": "benchmark",
        "ftns_budget": 1.0,
        "requester_id": "node-self",
        "payload": {},
    }

    await provider._on_job_offer("job_offer", job_data, "node-self")

    assert "job-003" not in provider.active_jobs
    assert "job-003" not in provider.completed_jobs


@pytest.mark.asyncio
async def test_other_node_jobs_always_accepted():
    """Jobs from other nodes are accepted regardless of self-compute setting."""
    provider = _make_provider(node_id="node-self", peer_count=0, allow_self_compute=True)

    job_data = {
        "job_id": "job-004",
        "job_type": "benchmark",
        "ftns_budget": 1.0,
        "requester_id": "node-other",  # Different from provider
        "payload": {"iterations": 1000},
    }

    await provider._on_job_offer("job_offer", job_data, "node-other")

    await asyncio.sleep(0.5)

    assert "job-004" in provider.active_jobs or "job-004" in provider.completed_jobs


@pytest.mark.asyncio
async def test_self_compute_benchmark_produces_result():
    """Self-executed benchmark job completes with a valid result."""
    provider = _make_provider(node_id="node-self", peer_count=0, allow_self_compute=True)

    job_data = {
        "job_id": "job-005",
        "job_type": "benchmark",
        "ftns_budget": 1.0,
        "requester_id": "node-self",
        "payload": {"iterations": 1000},
    }

    await provider._on_job_offer("job_offer", job_data, "node-self")

    # Wait for benchmark to complete
    for _ in range(20):
        await asyncio.sleep(0.2)
        if "job-005" in provider.completed_jobs:
            break

    assert "job-005" in provider.completed_jobs
    job = provider.completed_jobs["job-005"]
    assert job.status == JobStatus.COMPLETED
    assert job.result is not None
    assert "primes_found" in job.result or "elapsed_seconds" in job.result


@pytest.mark.asyncio
async def test_self_compute_records_earnings():
    """Self-executed job completes without self-crediting the ledger.

    v1.6.0: compute_provider._execute_job no longer self-credits FTNS.
    Payment flows through requester-side escrow release. For single-node
    self-compute, requester and provider are the same account — a self-credit
    would cause double-counting and ledger divergence.

    This test verifies that:
      1. The job completes successfully in single-node mode.
      2. The provider does NOT call ledger.credit directly.
    """
    provider = _make_provider(node_id="node-self", peer_count=0, allow_self_compute=True)

    job_data = {
        "job_id": "job-006",
        "job_type": "benchmark",
        "ftns_budget": 2.5,
        "requester_id": "node-self",
        "payload": {"iterations": 1000},
    }

    await provider._on_job_offer("job_offer", job_data, "node-self")

    for _ in range(20):
        await asyncio.sleep(0.2)
        if "job-006" in provider.completed_jobs:
            break

    assert "job-006" in provider.completed_jobs
    # Provider must NOT self-credit — payment flows through requester-side
    # escrow release. See compute_provider._execute_job docstring.
    provider.ledger.credit.assert_not_called()


async def _run_self_compute(provider, job_id="job-007"):
    job_data = {
        "job_id": job_id,
        "job_type": "benchmark",
        "ftns_budget": 1.0,
        "requester_id": "node-self",
        "payload": {"iterations": 1000},
    }
    await provider._on_job_offer("job_offer", job_data, "node-self")
    for _ in range(20):
        await asyncio.sleep(0.2)
        if job_id in provider.completed_jobs:
            break
    assert job_id in provider.completed_jobs
    return [call.args[0] for call in provider.gossip.publish.call_args_list]


@pytest.mark.asyncio
async def test_self_compute_with_no_peers_skips_the_result_gossip_and_delivers_locally():
    """sp1388 + sp1397 (this test replaced an obsolete one asserting a job_result gossip).

    ``_publish_job_result_bounded`` short-circuits on ``peer_count == 0`` — there is nobody to
    broadcast to, and awaiting that publish used to hang /compute/query forever. The result instead
    reaches the submitting requester through ``local_result_sink``, because ``transport.gossip`` has
    no loopback (a self-computed result could never have reached the only GOSSIP_JOB_RESULT
    subscriber, ``ComputeRequester._on_job_result``, on the same node).
    """
    provider = _make_provider(node_id="node-self", peer_count=0, allow_self_compute=True)
    delivered = []
    provider.local_result_sink = lambda job_id, result: delivered.append((job_id, result))

    subtypes = await _run_self_compute(provider)

    assert "job_accept" in subtypes
    assert "job_result" not in subtypes, "no peers → nothing to broadcast to"
    assert len(delivered) == 1 and delivered[0][0] == "job-007"
    assert delivered[0][1], "the self-computed result must reach the requester directly"


@pytest.mark.asyncio
async def test_result_gossip_is_skipped_only_when_there_are_no_peers():
    """The other side of the sp1388 branch, tested at its own unit.

    A node with peers never SELF-computes (``_on_job_offer``: self-accept requires
    ``peer_count == 0``), so the peers-present path can only be reached by driving
    ``_publish_job_result_bounded`` directly — which is the branch sp1388 actually added.
    """
    quiet = _make_provider(peer_count=0)
    await quiet._publish_job_result_bounded({"job_id": "job-009", "result": {"ok": True}})
    quiet.gossip.publish.assert_not_called()

    connected = _make_provider(peer_count=3)
    await connected._publish_job_result_bounded({"job_id": "job-009", "result": {"ok": True}})
    connected.gossip.publish.assert_awaited_once()
    assert connected.gossip.publish.call_args.args[0] == "job_result"
