"""Tests for ComputeRequester reliability recording."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from prsm.node.compute_requester import ComputeRequester, JobType, SubmittedJob
from prsm.node.consensus_mismatch_log import ConsensusMismatchLog


def _make_requester():
    identity = MagicMock()
    identity.node_id = "requester_node"
    transport = MagicMock()
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    gossip.publish = AsyncMock(return_value=1)
    ledger = MagicMock()
    ledger.get_balance = AsyncMock(return_value=1000.0)
    ledger.transfer = AsyncMock()

    discovery = MagicMock()
    discovery.record_job_success = MagicMock()
    discovery.record_job_failure = MagicMock()

    req = ComputeRequester(
        identity=identity,
        transport=transport,
        gossip=gossip,
        ledger=ledger,
        discovery=discovery,
    )
    req.escrow = None
    req.ledger_sync = None
    return req


class TestReliabilityRecording:

    @pytest.mark.asyncio
    async def test_successful_result_records_success(self):
        req = _make_requester()
        req._running = True

        job = SubmittedJob(
            job_id="job_001",
            job_type=JobType.INFERENCE,
            payload={"prompt": "test"},
            ftns_budget=0.0,
        )
        req.submitted_jobs["job_001"] = job

        await req._on_job_result("job_result", {
            "job_id": "job_001",
            "provider_id": "requester_node",
            "status": "completed",
            "result": {"output": "hello"},
        }, "requester_node")

        req.discovery.record_job_success.assert_called_once_with("requester_node")

    @pytest.mark.asyncio
    async def test_failed_result_records_failure(self):
        req = _make_requester()
        req._running = True

        job = SubmittedJob(
            job_id="job_002",
            job_type=JobType.INFERENCE,
            payload={"prompt": "test"},
            ftns_budget=0.0,
        )
        req.submitted_jobs["job_002"] = job

        await req._on_job_result("job_result", {
            "job_id": "job_002",
            "provider_id": "provider_node",
            "status": "failed",
            "error": "GPU OOM",
        }, "provider_node")

        req.discovery.record_job_failure.assert_called_once_with("provider_node")

    @pytest.mark.asyncio
    async def test_no_discovery_is_safe(self):
        req = _make_requester()
        req.discovery = None
        req._running = True

        job = SubmittedJob(
            job_id="job_003",
            job_type=JobType.INFERENCE,
            payload={"prompt": "test"},
            ftns_budget=0.0,
        )
        req.submitted_jobs["job_003"] = job

        await req._on_job_result("job_result", {
            "job_id": "job_003",
            "provider_id": "requester_node",
            "status": "completed",
            "result": {"output": "ok"},
        }, "requester_node")


def _peer(node_id, *, success=0, failure=0):
    """Real PeerInfo so reliability_score works; capability='inference'."""
    from prsm.node.discovery import PeerInfo
    return PeerInfo(
        node_id=node_id, address=f"ws://{node_id}:9000",
        capabilities=["inference"],
        job_success_count=success, job_failure_count=failure,
    )


def _discovery_with_peers(peers):
    """A discovery double whose capability lookup returns `peers` and whose
    backend lookup returns [] (so the backend filter is a no-op keep-all)."""
    disc = MagicMock()
    disc.find_peers_with_capability = MagicMock(return_value=list(peers))
    disc.find_peers_with_backend = MagicMock(return_value=[])
    return disc


class TestDispatchEligibilityGate:
    """sp958 — a provider with >= PRSM_DISPATCH_MAX_CONSENSUS_MISMATCHES confirmed
    consensus-mismatch events (the sp957 evidence log) is EXCLUDED from compute
    dispatch selection — not merely sorted last. This is the sound, local,
    reversible enforcement of the sp928/957 evidence (no on-chain slash)."""

    def _req_with_log(self, peers, log, monkeypatch, *, max_mismatches=2):
        monkeypatch.setenv("PRSM_DISPATCH_MAX_CONSENSUS_MISMATCHES",
                           str(max_mismatches))
        req = _make_requester()
        req.discovery = _discovery_with_peers(peers)
        req.mismatch_log = log
        return req

    def test_confirmed_repeat_liar_excluded(self, monkeypatch):
        log = ConsensusMismatchLog()
        for i in range(2):  # 2 confirmed mismatches == threshold
            asyncio.run(log.record({
                "reason": "CONSENSUS_MISMATCH", "job_id": f"j{i}",
                "accused_provider_id": "prov-liar"}))
        req = self._req_with_log(
            [_peer("prov-liar", success=50, failure=2),
             _peer("prov-honest", success=10, failure=0)],
            log, monkeypatch)
        out = req._get_capable_peers(JobType.INFERENCE)
        assert "prov-liar" not in out, (
            "a provider with >= threshold confirmed mismatches must be excluded "
            "from dispatch, even though its reliability ratio looks fine (0.96)"
        )
        assert "prov-honest" in out

    def test_below_threshold_still_eligible(self, monkeypatch):
        log = ConsensusMismatchLog()
        asyncio.run(log.record({
            "reason": "CONSENSUS_MISMATCH", "job_id": "j1",
            "accused_provider_id": "prov-once"}))  # 1 < threshold 2
        req = self._req_with_log([_peer("prov-once")], log, monkeypatch)
        out = req._get_capable_peers(JobType.INFERENCE)
        assert "prov-once" in out  # one mismatch alone doesn't exclude

    def test_no_mismatch_log_keeps_all(self, monkeypatch):
        # mismatch_log unset (minimal wiring / legacy) → no exclusion.
        req = _make_requester()
        req.discovery = _discovery_with_peers([_peer("p1"), _peer("p2")])
        req.mismatch_log = None
        out = req._get_capable_peers(JobType.INFERENCE)
        assert set(out) == {"p1", "p2"}

    def test_window_lets_old_evidence_decay(self, monkeypatch):
        from prsm.node.consensus_mismatch_log import ConsensusMismatchEntry
        log = ConsensusMismatchLog()
        # Two OLD confirmed mismatches (timestamp 100), beyond the decay window.
        for i in range(2):
            log.append(ConsensusMismatchEntry(
                timestamp=100.0, job_id=f"old{i}",
                accused_provider_id="prov-reformed",
                accused_output_hash="0x1", majority_output_hash="0x2"))
        monkeypatch.setenv("PRSM_DISPATCH_MISMATCH_WINDOW_SEC", "3600")
        req = self._req_with_log([_peer("prov-reformed")], log, monkeypatch)
        out = req._get_capable_peers(JobType.INFERENCE)
        # now() is far past 100+3600 → old evidence outside window → eligible.
        assert "prov-reformed" in out
