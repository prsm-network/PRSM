"""Sprint 1217 — node-runtime alert rules + full observability-stack
integration (Brick 3). Verifies AlertManager.setup_node_runtime_rules registers
the expected rules over NodeRuntimeMetrics' metric names, the duration-gated
firing logic, and that the exact object graph the node wires
(MetricsCollector + NodeRuntimeMetrics + AlertManager) interoperates and
starts/stops cleanly — without booting a full PRSMNode.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from prsm.core.monitoring.alerts import AlertManager, AlertSeverity
from prsm.core.monitoring.metrics import MetricsCollector
from prsm.core.monitoring.node_runtime_metrics import NodeRuntimeMetrics


class _FakeSettlementClient:
    def __init__(self, status_dict):
        self._status = status_dict

    def status(self):
        return dict(self._status)


class _Node:
    def __init__(self, *, inference_executor=None, onchain_settlement_client=None):
        self.inference_executor = inference_executor
        self._onchain_settlement_client = onchain_settlement_client


# ── rule registration ───────────────────────────────────────────────────────

def test_setup_registers_node_runtime_rules():
    am = AlertManager()
    am.setup_node_runtime_rules()
    assert {
        "node_not_ready",
        "settlement_funds_in_flight_stuck",
        "settlement_committing_intents_unresolved",
    }.issubset(set(am.rules))

    r = am.rules["node_not_ready"]
    assert r.condition.metric_name == "prsm_node_ready"
    assert r.condition.operator == "lt"
    assert r.condition.threshold == 1
    assert r.severity == AlertSeverity.CRITICAL
    # no channel dependency (registers/logs without email/slack wired)
    assert r.channels == []

    fif = am.rules["settlement_funds_in_flight_stuck"]
    assert fif.condition.metric_name == "prsm_settlement_funds_in_flight"
    assert fif.condition.operator == "gt" and fif.condition.threshold == 0
    assert fif.severity == AlertSeverity.WARNING


# ── duration-gated firing ────────────────────────────────────────────────────

def test_node_not_ready_fires_only_after_sustained_duration():
    am = AlertManager()
    am.setup_node_runtime_rules()
    r = am.rules["node_not_ready"]
    t0 = datetime(2026, 6, 21, 12, 0, 0)
    # first breaching sample starts the window (not yet fired)
    assert r.evaluate(0.0, t0) is False
    # still breaching past the duration → fires
    assert r.evaluate(0.0, t0 + r.condition.duration + timedelta(seconds=1)) is True


def test_ready_node_never_fires():
    am = AlertManager()
    am.setup_node_runtime_rules()
    r = am.rules["node_not_ready"]
    t0 = datetime(2026, 6, 21, 12, 0, 0)
    assert r.evaluate(1.0, t0) is False
    assert r.evaluate(1.0, t0 + r.condition.duration + timedelta(seconds=1)) is False


# ── full-stack integration (the exact graph node.start() wires) ──────────────

def test_collector_alert_stack_interoperates_and_lifecycles():
    async def _run():
        node = _Node(
            inference_executor=None,
            onchain_settlement_client=_FakeSettlementClient({
                "tracked_batches": 2, "pending_commits": 1,
                "committing_intents": 0, "finalized_locally": 0,
                "funds_in_flight": True,
            }),
        )
        collector = MetricsCollector(collection_interval=0.05)
        collector.registry.register_metric(NodeRuntimeMetrics(node))

        # registry.collect_all is exactly what AlertManager's eval loop reads.
        values = await collector.registry.collect_all()
        by_name = {v.name: v.value for v in values}
        assert by_name.get("prsm_node_ready") == 0
        assert by_name.get("prsm_settlement_enabled") == 1
        assert by_name.get("prsm_settlement_pending_commits") == 1
        assert by_name.get("prsm_settlement_funds_in_flight") == 1

        alert_mgr = AlertManager(evaluation_interval=0.05)
        alert_mgr.set_metrics_collector(collector)
        alert_mgr.setup_node_runtime_rules()

        # Start both background loops (node lifecycle) and stop them cleanly.
        await collector.start_collection()
        await alert_mgr.start_monitoring()
        await asyncio.sleep(0.12)  # let a couple of cycles run
        assert collector.is_collecting is True
        assert alert_mgr.is_monitoring is True
        await alert_mgr.stop_monitoring()
        await collector.stop_collection()
        assert alert_mgr.is_monitoring is False
        assert collector.is_collecting is False

    asyncio.run(_run())


def test_rules_carry_provided_channels():
    """sp1218 — setup_node_runtime_rules(channels=[...]) routes fired rules to
    the named channels; default → no channels (register/log only)."""
    am = AlertManager()
    am.setup_node_runtime_rules(channels=["node-webhook"])
    for name in (
        "node_not_ready",
        "settlement_funds_in_flight_stuck",
        "settlement_committing_intents_unresolved",
    ):
        assert am.rules[name].channels == ["node-webhook"]

    am2 = AlertManager()
    am2.setup_node_runtime_rules()
    assert am2.rules["node_not_ready"].channels == []


def test_fired_alert_delivers_to_registered_channel():
    """sp1218 — a fired node-runtime rule actually pushes to its channel
    (the create_alert → _send_notifications path), not just active_alerts."""
    from prsm.core.monitoring.alerts import AlertChannel

    delivered = []

    class _StubChannel(AlertChannel):
        async def send_alert(self, alert):
            delivered.append(alert)
            return True

    am = AlertManager()
    am.add_channel(_StubChannel("node-webhook", {"enabled": True}))
    am.setup_node_runtime_rules(channels=["node-webhook"])
    rule = am.rules["node_not_ready"]

    asyncio.run(am.create_alert(rule, 0.0))
    assert len(delivered) == 1
    assert delivered[0].rule_name == "node_not_ready"
    # and it's tracked as active
    assert any(a.rule_name == "node_not_ready" for a in am.get_active_alerts())


def test_prometheus_exposition_includes_node_metrics():
    prom = pytest.importorskip("prometheus_client")  # noqa: F841
    node = _Node(inference_executor=object(), onchain_settlement_client=None)
    collector = MetricsCollector(collection_interval=1.0)
    collector.registry.register_metric(NodeRuntimeMetrics(node))

    async def _once():
        await collector.collect_once()

    asyncio.run(_once())
    text = collector.get_prometheus_metrics()
    # the bridged exposition should mention at least one node-runtime metric
    assert "prsm_node_ready" in text or "prsm_settlement_enabled" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
