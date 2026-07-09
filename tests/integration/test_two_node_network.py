"""
Integration test: Two PRSM nodes discover each other,
exchange a compute job, and settle payment.
"""

import asyncio
from unittest.mock import patch

import pytest

_REAL_SLEEP = asyncio.sleep


@pytest.fixture(autouse=True)
def real_asyncio_sleep():
    """Restore real asyncio.sleep for network tests."""
    with patch("asyncio.sleep", _REAL_SLEEP):
        yield

from types import SimpleNamespace

from prsm.node.compute_provider import ComputeProvider, JobType
from prsm.node.compute_requester import ComputeRequester, JobStatus
from prsm.node.config import NodeConfig, NodeRole
from prsm.node.discovery import PeerDiscovery
from prsm.node.gossip import GossipProtocol
from prsm.node.identity import generate_node_identity
from prsm.node.ledger_sync import LedgerSync
from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.node.transport import WebSocketTransport


async def _setup_node(name, p2p_port, bootstrap=None):
    """Create a minimal node stack for testing."""
    identity = generate_node_identity(name)
    transport = WebSocketTransport(identity, host="127.0.0.1", port=p2p_port)
    gossip = GossipProtocol(transport, fanout=3, heartbeat_interval=9999)
    ledger = LocalLedger(":memory:")
    await ledger.initialize()
    await ledger.create_wallet(identity.node_id, name)
    await ledger.create_wallet("system")
    await ledger.issue_welcome_grant(identity.node_id, 100.0)

    # LedgerSync handles cross-node FTNS transfers over gossip. Without it,
    # the requester's ledger.transfer debits the requester but the
    # recipient node's ledger never credits — payment doesn't cross
    # process boundaries. Production wiring (prsm.node.node) sets
    # ledger_sync on the requester after construction; the test must do
    # the same.
    ledger_sync = LedgerSync(
        identity=identity,
        gossip=gossip,
        ledger=ledger,
        transport=transport,
    )

    discovery = PeerDiscovery(
        transport,
        bootstrap_nodes=[bootstrap] if bootstrap else [],
    )
    provider = ComputeProvider(
        identity=identity,
        transport=transport,
        gossip=gossip,
        ledger=ledger,
    )
    requester = ComputeRequester(
        identity=identity,
        transport=transport,
        gossip=gossip,
        ledger=ledger,
    )
    # Production nodes set this attribute after construction.
    requester.ledger_sync = ledger_sync

    return {
        "identity": identity,
        "transport": transport,
        "gossip": gossip,
        "ledger": ledger,
        "ledger_sync": ledger_sync,
        "discovery": discovery,
        "provider": provider,
        "requester": requester,
    }


def _wire_embedding_backend(node, dimensions_ok=True):
    """Give a node a real embedding backend, the way a production node has one.

    ``ComputeProvider._run_embedding`` calls ``orchestrator.backend_registry
    .embed_with_fallback(...)`` and labels the result ``source: "backend_registry"``.
    Only when that is absent does it fabricate the ``source: "mock"`` sha256 pseudo-vector.
    """

    class _Registry:
        async def embed_with_fallback(self, text, model_id=None, dimensions=1536):
            # deterministic, but a genuine function of the text under a named model
            vec = [((i * 7 + len(text)) % 200 - 100) / 100.0 for i in range(dimensions)]
            return SimpleNamespace(
                embedding=vec,
                model_id=model_id or "test-embed-v1",
                provider=SimpleNamespace(value="local"),
                token_count=len(text.split()),
            )

    node["provider"].orchestrator = SimpleNamespace(backend_registry=_Registry())
    return node


async def _start_node(node):
    await node["transport"].start()
    await node["gossip"].start()
    node["ledger_sync"].start()
    await node["provider"].start()
    await node["requester"].start()
    await node["discovery"].start()


async def _stop_node(node):
    await node["discovery"].stop()
    await node["provider"].stop()
    await node["requester"].stop()
    await node["ledger_sync"].stop()
    await node["gossip"].stop()
    await node["transport"].stop()
    await node["ledger"].close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_two_nodes_discover_and_connect():
    """Two nodes find each other via bootstrap."""
    node_a = await _setup_node("node-A", 19400)
    node_b = await _setup_node("node-B", 19401, bootstrap="127.0.0.1:19400")

    try:
        await _start_node(node_a)
        await _start_node(node_b)

        await asyncio.sleep(0.5)

        assert node_a["transport"].peer_count >= 1
        assert node_b["transport"].peer_count >= 1
    finally:
        await _stop_node(node_b)
        await _stop_node(node_a)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_two_nodes_compute_job_and_payment():
    """Node A submits a job, Node B executes it, payment is settled."""
    node_a = await _setup_node("requester-A", 19410)
    node_b = await _setup_node("provider-B", 19411, bootstrap="127.0.0.1:19410")

    try:
        await _start_node(node_a)
        await _start_node(node_b)
        await asyncio.sleep(0.5)

        # Verify connection
        assert node_a["transport"].peer_count >= 1

        # Check initial balances
        balance_a_before = await node_a["ledger"].get_balance(node_a["identity"].node_id)
        balance_b_before = await node_b["ledger"].get_balance(node_b["identity"].node_id)
        assert balance_a_before == 100.0
        assert balance_b_before == 100.0

        # Node A submits a benchmark job
        submitted = await node_a["requester"].submit_job(
            job_type=JobType.BENCHMARK,
            payload={"iterations": 100},
            ftns_budget=5.0,
        )
        assert submitted.job_id is not None

        # Wait for job to be accepted and completed
        result = await node_a["requester"].get_result(submitted.job_id, timeout=10.0)

        assert result is not None, f"Job timed out. Status: {submitted.status.value}, error: {submitted.error}"
        assert "primes_found" in result
        assert submitted.result_verified  # signature verified

        # Wait for payment gossip to propagate A -> B. ComputeRequester
        # publishes GOSSIP_FTNS_TRANSACTION via LedgerSync after debiting
        # locally; B's LedgerSync subscription fires on receipt and
        # credits B's ledger. Give a few poll cycles for gossip fan-out.
        expected_b = 105.0
        for _ in range(30):  # up to ~3s
            balance_b_check = await node_b["ledger"].get_balance(node_b["identity"].node_id)
            if abs(balance_b_check - expected_b) < 0.01:
                break
            await asyncio.sleep(0.1)

        # Verify payment was recorded
        balance_a_after = await node_a["ledger"].get_balance(node_a["identity"].node_id)
        balance_b_after = await node_b["ledger"].get_balance(node_b["identity"].node_id)

        # A paid 5 FTNS, B earned 5 FTNS
        assert balance_a_after == pytest.approx(95.0, abs=0.01)
        assert balance_b_after == pytest.approx(105.0, abs=0.01)

    finally:
        await _stop_node(node_b)
        await _stop_node(node_a)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_two_nodes_embedding_job():
    """End-to-end embedding job between two nodes, with the worker actually backed.

    sp1408 — the worker must serve a REAL embedding. Previously this passed against
    ``_run_embedding``'s ``source: "mock"`` sha256 pseudo-vector fallback, i.e. the
    requester paid 2 FTNS for a hash. Wire the worker's backend registry so the job
    travels the production ``source: "backend_registry"`` path.
    """
    node_a = await _setup_node("client", 19420)
    node_b = await _setup_node("worker", 19421, bootstrap="127.0.0.1:19420")
    _wire_embedding_backend(node_b)

    try:
        await _start_node(node_a)
        await _start_node(node_b)
        await asyncio.sleep(0.5)

        submitted = await node_a["requester"].submit_job(
            job_type=JobType.EMBEDDING,
            payload={"text": "decentralized AI for science", "dimensions": 64},
            ftns_budget=2.0,
        )

        result = await node_a["requester"].get_result(submitted.job_id, timeout=10.0)

        assert result is not None
        assert len(result["embedding"]) == 64
        assert result["provider_node"] == node_b["identity"].node_id
        assert result["source"] == "backend_registry"     # a real answer, not a mock

    finally:
        await _stop_node(node_b)
        await _stop_node(node_a)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_two_nodes_unbacked_worker_mock_is_rejected_and_unpaid():
    """sp1408 — a worker with NO embedding backend fabricates a `source: "mock"` pseudo-vector
    and signs it. The signature is valid (it really is from that provider), so the requester
    used to accept it and pay. It must now refuse: no result, no payment."""
    node_a = await _setup_node("client", 19422)
    node_b = await _setup_node("worker", 19423, bootstrap="127.0.0.1:19422")
    # NOTE: node_b deliberately has no embedding backend → _run_embedding fabricates.

    try:
        await _start_node(node_a)
        await _start_node(node_b)
        await asyncio.sleep(0.5)

        worker_id = node_b["identity"].node_id
        before = await node_b["ledger"].get_balance(worker_id)

        submitted = await node_a["requester"].submit_job(
            job_type=JobType.EMBEDDING,
            payload={"text": "decentralized AI for science", "dimensions": 64},
            ftns_budget=2.0,
        )
        result = await node_a["requester"].get_result(submitted.job_id, timeout=10.0)

        assert result is None, "a self-declared mock must never be delivered as a result"
        job = node_a["requester"].submitted_jobs[submitted.job_id]
        assert job.status is JobStatus.FAILED
        assert "mock" in (job.error or "").lower()

        await asyncio.sleep(0.3)   # let any (erroneous) payment gossip land
        after = await node_b["ledger"].get_balance(worker_id)
        assert after == before, "the fabricating worker must not be paid"

    finally:
        await _stop_node(node_b)
        await _stop_node(node_a)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_insufficient_balance_rejected():
    """Submitting a job with insufficient balance should fail immediately."""
    node_a = await _setup_node("broke-node", 19430)

    try:
        await _start_node(node_a)

        # Try to submit a job worth more than our balance
        with pytest.raises(ValueError, match="Insufficient FTNS balance"):
            await node_a["requester"].submit_job(
                job_type=JobType.BENCHMARK,
                payload={},
                ftns_budget=999.0,
            )
    finally:
        await _stop_node(node_a)
