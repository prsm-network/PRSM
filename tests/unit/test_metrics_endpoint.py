"""GET /metrics — Prometheus-format observability endpoint.

Operators running production nodes want Grafana dashboards +
alerting on PRSM-specific gauges. Standard Prometheus
text/plain exposition format makes the existing observability
stack just work.

Gauges emitted from live node state (no new tracking infra):
- prsm_pending_escrow_count
- prsm_total_locked_ftns
- prsm_job_history_size
- prsm_arbitration_pending_count
- prsm_claimable_royalties_wei
- prsm_node_health (1 healthy, 0.5 degraded, 0 unhealthy)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.payment_escrow import EscrowEntry, EscrowStatus, PaymentEscrow
from prsm.node.job_history import JobHistoryStore


def _ledger():
    led = MagicMock()
    led.get_balance = AsyncMock(return_value=100.0)
    led.transfer = AsyncMock()
    led.create_wallet = AsyncMock()
    return led


def _node(*, escrows=0, history_size=0, claimable_wei=0,
          arbitration_pending=0):
    node = MagicMock()
    node.identity.node_id = "test-node"

    ftns_ledger = MagicMock()
    ftns_ledger._is_initialized = True
    ftns_ledger._connected_address = "0x" + "11" * 20
    ftns_ledger._decimals = 18
    ftns_ledger.get_balance = AsyncMock(return_value=42.0)
    node.ftns_ledger = ftns_ledger

    escrow = PaymentEscrow(ledger=_ledger(), node_id="test-node")
    for i in range(escrows):
        e = EscrowEntry(
            escrow_id=f"e{i}", job_id=f"j{i}",
            requester_id="0x" + "11" * 20, amount=1.5,
            status=EscrowStatus.PENDING,
        )
        escrow._escrows[e.escrow_id] = e
    node._payment_escrow = escrow

    history = JobHistoryStore()
    from prsm.node.job_history import JobHistoryRecord, JobStatus
    import time
    for i in range(history_size):
        history.put(JobHistoryRecord(
            job_id=f"forge-{i}", query="q",
            status=JobStatus.COMPLETED,
            started_at=time.time(),
        ))
    node._job_history = history

    royalty = MagicMock()
    royalty.claimable = MagicMock(return_value=claimable_wei)
    node._royalty_distributor_client = royalty

    arb = MagicMock()
    arb.list_pending = AsyncMock(
        return_value=[MagicMock()] * arbitration_pending,
    )
    node._arbitration_queue = arb
    # Default no cleanup task wired — tests that need one set
    # _escrow_cleanup_task explicitly.
    node._escrow_cleanup_task = None
    # Default no daemon tasks wired — same rationale as cleanup_task.
    node._heartbeat_scheduler_task = None
    node._compensation_scheduler_task = None
    node._key_distribution_watcher_task = None
    node._storage_slashing_watcher_task = None
    node._compensation_distributor_watcher_task = None
    node._job_reaper_task = None
    node._daemon_watchdog_task = None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


# ──────────────────────────────────────────────────────────────────────
# Format
# ──────────────────────────────────────────────────────────────────────


class TestMetricsFormat:
    def test_returns_prometheus_content_type(self):
        node = _node()
        resp = _client(node).get("/metrics")
        assert resp.status_code == 200
        ct = resp.headers["content-type"]
        # Prometheus exposition is text/plain, optionally with version.
        assert ct.startswith("text/plain")

    def test_emits_help_and_type_lines(self):
        node = _node()
        body = _client(node).get("/metrics").text
        # Each metric should have a # HELP and # TYPE line per format.
        assert "# HELP prsm_pending_escrow_count" in body
        assert "# TYPE prsm_pending_escrow_count gauge" in body


# ──────────────────────────────────────────────────────────────────────
# Gauges
# ──────────────────────────────────────────────────────────────────────


class TestMetricsGauges:
    def test_pending_escrow_count(self):
        node = _node(escrows=3)
        body = _client(node).get("/metrics").text
        assert "prsm_pending_escrow_count 3" in body

    def test_total_locked_ftns(self):
        node = _node(escrows=4)  # each amount=1.5 → 6.0 total
        body = _client(node).get("/metrics").text
        # Should emit "prsm_total_locked_ftns 6.0" or "6"
        assert "prsm_total_locked_ftns 6" in body

    def test_job_history_size(self):
        node = _node(history_size=7)
        body = _client(node).get("/metrics").text
        assert "prsm_job_history_size 7" in body

    def test_claimable_royalties_wei(self):
        node = _node(claimable_wei=5_000_000_000_000_000_000)
        body = _client(node).get("/metrics").text
        assert "prsm_claimable_royalties_wei 5000000000000000000" in body

    def test_arbitration_pending_count(self):
        node = _node(arbitration_pending=2)
        body = _client(node).get("/metrics").text
        assert "prsm_arbitration_pending_count 2" in body


# ──────────────────────────────────────────────────────────────────────
# Fail-soft: subsystem missing → metric absent or zero
# ──────────────────────────────────────────────────────────────────────


class TestCleanupTaskGauge:
    """prsm_escrow_cleanup_task_running gauge — companion to the
    cleanup_task_running field on /health/detailed. Lets operators
    alert via Prometheus when the periodic_cleanup task crashes
    silently."""

    def test_gauge_emitted_when_task_running(self):
        node = _node()
        fake_task = MagicMock()
        fake_task.done.return_value = False
        node._escrow_cleanup_task = fake_task
        body = _client(node).get("/metrics").text
        assert "prsm_escrow_cleanup_task_running 1" in body

    def test_gauge_emitted_zero_when_task_done(self):
        """Done == crashed (the task is an infinite loop). Gauge
        flips to 0 to alarm."""
        node = _node()
        fake_task = MagicMock()
        fake_task.done.return_value = True
        node._escrow_cleanup_task = fake_task
        body = _client(node).get("/metrics").text
        assert "prsm_escrow_cleanup_task_running 0" in body

    def test_gauge_omitted_when_task_not_wired(self):
        """No task attached → omit the gauge rather than emit
        a misleading 0/1 value. Differentiates "we don't know"
        from "definitely crashed"."""
        node = _node()  # no _escrow_cleanup_task attr
        body = _client(node).get("/metrics").text
        assert "prsm_escrow_cleanup_task_running" not in body


class TestDaemonGauges:
    """Same task_running gauge pattern applied to the remaining 4
    daemons so operators get full Prometheus alerting coverage."""

    def _setup(self, node, attr_pair, running=True):
        scheduler_attr, task_attr = attr_pair
        setattr(node, scheduler_attr, MagicMock())
        fake_task = MagicMock()
        fake_task.done.return_value = not running
        setattr(node, task_attr, fake_task)

    def test_heartbeat_scheduler_gauge(self):
        node = _node()
        self._setup(node, ("_heartbeat_scheduler", "_heartbeat_scheduler_task"))
        body = _client(node).get("/metrics").text
        assert "prsm_heartbeat_scheduler_running 1" in body

    def test_compensation_scheduler_gauge_crash(self):
        node = _node()
        self._setup(
            node,
            ("_compensation_scheduler", "_compensation_scheduler_task"),
            running=False,
        )
        body = _client(node).get("/metrics").text
        assert "prsm_compensation_scheduler_running 0" in body

    def test_key_distribution_watcher_gauge(self):
        node = _node()
        self._setup(
            node,
            ("_key_distribution_watcher", "_key_distribution_watcher_task"),
        )
        body = _client(node).get("/metrics").text
        assert "prsm_key_distribution_watcher_running 1" in body

    def test_storage_slashing_watcher_gauge(self):
        node = _node()
        self._setup(
            node,
            ("_storage_slashing_watcher", "_storage_slashing_watcher_task"),
        )
        body = _client(node).get("/metrics").text
        assert "prsm_storage_slashing_watcher_running 1" in body

    def test_compensation_distributor_watcher_gauge(self):
        node = _node()
        self._setup(
            node,
            ("_compensation_distributor_watcher",
             "_compensation_distributor_watcher_task"),
        )
        body = _client(node).get("/metrics").text
        assert (
            "prsm_compensation_distributor_watcher_running 1"
            in body
        )

    def test_job_reaper_gauge(self):
        node = _node()
        self._setup(node, ("_job_reaper", "_job_reaper_task"))
        body = _client(node).get("/metrics").text
        assert "prsm_job_reaper_running 1" in body

    def test_daemon_watchdog_gauge(self):
        node = _node()
        self._setup(node, ("_daemon_watchdog", "_daemon_watchdog_task"))
        body = _client(node).get("/metrics").text
        assert "prsm_daemon_watchdog_running 1" in body


class TestMetricsFailSoft:
    def test_no_payment_escrow_emits_zero_or_absent(self):
        """Without PaymentEscrow wired, escrow gauges should still
        be safe — emit 0 or omit the metric, but never 500."""
        node = MagicMock()
        node.identity.node_id = "test-node"
        node.ftns_ledger = None
        node._payment_escrow = None
        node._job_history = None
        node._royalty_distributor_client = None
        node._arbitration_queue = None
        resp = _client(node).get("/metrics")
        assert resp.status_code == 200
        body = resp.text
        # Either 0 or omitted — both acceptable. Test asserts no crash.
        assert "prsm_" in body  # at least some metric still present

    def test_royalty_rpc_error_does_not_500(self):
        node = _node()
        node._royalty_distributor_client.claimable = MagicMock(
            side_effect=RuntimeError("rpc down"),
        )
        resp = _client(node).get("/metrics")
        # Endpoint must NOT 500 — emit 0 or omit royalties gauge.
        assert resp.status_code == 200


# ──────────────────────────────────────────────────────────────────────
# Sprint 402 — Prometheus labeled gauges for sprint-399-401
# tick-age tracking on operator-node daemons
# ──────────────────────────────────────────────────────────────────────


class TestSubsystemTickAgeGauges:
    """Sprint 402 — sprint-395's subsystem_status gauge now
    incorporates tick_status (from sprint 399-401 daemon
    extensions). A daemon whose task_running=True but
    tick_status=stale is observably bad — every tick is
    failing — but pre-sprint-402 it encoded as 0 (healthy)
    because the entry's top-level status was 'ok'.

    Sprint 402 also adds a dedicated tick-age gauge so
    PromQL can target heartbeat age directly. Mirrors
    sprint-394's bootstrap-side
    prsm_bootstrap_subsystem_heartbeat_age_seconds."""

    def _make_heartbeat_scheduler_node(
        self, *, age_seconds, interval=900,
    ):
        node = _node()
        scheduler = MagicMock()
        scheduler.interval_seconds = interval
        scheduler.last_tick_age_seconds = age_seconds
        node._heartbeat_scheduler = scheduler
        fake_task = MagicMock()
        fake_task.done.return_value = False
        node._heartbeat_scheduler_task = fake_task
        return node

    def test_stale_tick_status_encodes_subsystem_as_2(self):
        # 5000s old, 900s interval = 5.56× → stale
        node = self._make_heartbeat_scheduler_node(
            age_seconds=5000, interval=900,
        )
        body = _client(node).get("/metrics").text
        # Subsystem encoded as 2 (unhealthy) because tick is
        # stale, even though entry status is 'ok'
        assert (
            'prsm_node_subsystem_status'
            '{subsystem="heartbeat_scheduler"} 2'
        ) in body

    def test_degraded_tick_status_encodes_subsystem_as_1(self):
        # 2000s old, 900s interval = 2.22× → degraded
        node = self._make_heartbeat_scheduler_node(
            age_seconds=2000, interval=900,
        )
        body = _client(node).get("/metrics").text
        assert (
            'prsm_node_subsystem_status'
            '{subsystem="heartbeat_scheduler"} 1'
        ) in body

    def test_healthy_tick_status_keeps_subsystem_at_0(self):
        # 100s old, 900s interval = 0.11× → healthy
        node = self._make_heartbeat_scheduler_node(
            age_seconds=100, interval=900,
        )
        body = _client(node).get("/metrics").text
        assert (
            'prsm_node_subsystem_status'
            '{subsystem="heartbeat_scheduler"} 0'
        ) in body

    def test_tick_age_gauge_emitted(self):
        node = self._make_heartbeat_scheduler_node(
            age_seconds=42.5, interval=900,
        )
        body = _client(node).get("/metrics").text
        # Dedicated age gauge surfaces the raw number
        assert (
            'prsm_node_subsystem_tick_age_seconds'
            '{subsystem="heartbeat_scheduler"} 42.5'
        ) in body

    def test_tick_age_help_and_type_lines_present(self):
        node = self._make_heartbeat_scheduler_node(
            age_seconds=42.5,
        )
        body = _client(node).get("/metrics").text
        assert (
            "# HELP prsm_node_subsystem_tick_age_seconds"
            in body
        )
        assert (
            "# TYPE prsm_node_subsystem_tick_age_seconds gauge"
            in body
        )

    def test_no_tick_age_gauge_when_daemon_lacks_tick_age(self):
        """Daemons that haven't adopted the tick-age pattern
        do NOT get a tick_age_seconds gauge entry. Pinned to
        avoid noise from legacy daemons (e.g. event-watcher
        observability rings)."""
        # ftns_ledger doesn't have tick-age semantics
        node = _node()
        body = _client(node).get("/metrics").text
        # ftns_ledger appears in subsystem_status gauge
        # (sprint 395) but should NOT appear in
        # tick_age_seconds gauge (sprint 402 opt-in)
        assert (
            'prsm_node_subsystem_status'
            '{subsystem="ftns_ledger"}' in body
        )
        # Not in the tick-age gauge
        assert (
            'prsm_node_subsystem_tick_age_seconds'
            '{subsystem="ftns_ledger"}' not in body
        )


# ──────────────────────────────────────────────────────────────────────
# Sprint 395 — Per-subsystem Prometheus labeled gauges
# ──────────────────────────────────────────────────────────────────────


class TestSubsystemLabeledGauges:
    """Mirrors sprint 394's bootstrap-server-side labeled
    gauges on the operator-node side. /health/detailed
    exposes per-subsystem readiness as JSON; before sprint
    395 PromQL alerts could only see aggregate health.
    Now each subsystem has its own labeled gauge."""

    def test_subsystem_status_gauge_emitted_for_core(self):
        node = _node()
        body = _client(node).get("/metrics").text
        # ftns_ledger is core, available → status=0 (healthy)
        assert (
            'prsm_node_subsystem_status'
            '{subsystem="ftns_ledger"} 0'
        ) in body

    def test_help_and_type_lines_present(self):
        node = _node()
        body = _client(node).get("/metrics").text
        assert "# HELP prsm_node_subsystem_status" in body
        assert "# TYPE prsm_node_subsystem_status gauge" in body

    def test_not_wired_subsystem_encoded_as_1(self):
        node = _node()
        # job_history is optional; not-wired surfaces as 1
        # (= optional-opt-out, distinct from hard failure).
        node._job_history = None
        body = _client(node).get("/metrics").text
        assert (
            'prsm_node_subsystem_status'
            '{subsystem="job_history"} 1'
        ) in body

    def test_unhealthy_core_subsystem_encoded_as_2(self):
        node = _node()
        # ftns_ledger not initialized → status="uninitialized"
        # → encoded as 2 (= unhealthy/not-available core).
        node.ftns_ledger._is_initialized = False
        body = _client(node).get("/metrics").text
        assert (
            'prsm_node_subsystem_status'
            '{subsystem="ftns_ledger"} 2'
        ) in body

    def test_subsystem_block_does_not_500_endpoint(self):
        """Fail-soft per sprint-389 convention — if the
        /health/detailed call raises mid-iteration, the
        subsystem block omits but /metrics still 200s."""
        node = _node()
        # Break the closure by overriding ftns_ledger with
        # a property-raising mock.
        broken = MagicMock()
        type(broken)._is_initialized = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        node.ftns_ledger = broken
        resp = _client(node).get("/metrics")
        assert resp.status_code == 200


# ──────────────────────────────────────────────────────────────────────
# Sprint 1217 — opt-in runtime MetricsCollector exposition appended to /metrics
# ──────────────────────────────────────────────────────────────────────


class TestRuntimeMetricsAugmentation:
    """/metrics appends the opt-in MetricsCollector's bridged Prometheus
    exposition (NodeRuntimeMetrics) when wired; output is unchanged when off,
    and a render error never 500s the scrape."""

    def test_absent_when_collector_not_wired(self):
        node = _node()
        node._metrics_collector = None
        body = _client(node).get("/metrics").text
        assert "sp1217_sentinel_metric" not in body
        assert "prsm_node_up 1" in body  # base exposition intact

    def test_appended_when_collector_present(self):
        node = _node()
        mc = MagicMock()
        mc.get_prometheus_metrics.return_value = (
            "# HELP sp1217_sentinel_metric x\n"
            "# TYPE sp1217_sentinel_metric gauge\n"
            "sp1217_sentinel_metric 7\n"
        )
        node._metrics_collector = mc
        body = _client(node).get("/metrics").text
        assert "sp1217_sentinel_metric 7" in body
        assert "prsm_node_up 1" in body

    def test_non_string_render_is_ignored(self):
        """A mock/None render (non-str) is skipped, not appended — the
        join() must not see a non-str item."""
        node = _node()  # MagicMock node: _metrics_collector auto-mocks
        resp = _client(node).get("/metrics")
        assert resp.status_code == 200
        assert "prsm_node_up 1" in resp.text

    def test_collector_render_error_does_not_500(self):
        node = _node()
        mc = MagicMock()
        mc.get_prometheus_metrics.side_effect = RuntimeError("render boom")
        node._metrics_collector = mc
        resp = _client(node).get("/metrics")
        assert resp.status_code == 200
        assert "prsm_node_up 1" in resp.text
