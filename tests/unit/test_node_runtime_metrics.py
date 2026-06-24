"""Sprint 1217 — NodeRuntimeMetrics: a CustomMetric adapter that exposes a live
PRSM node's highest-value runtime signals (readiness + on-chain settlement
funds-in-flight) to the existing MetricsCollector/AlertManager stack.

Brick 1 of the node-observability arc (metrics source → AlertManager wiring).
Pure adapter: no node lifecycle change, fully offline-testable. Reuses
prsm.core.monitoring.metrics.{CustomMetric,MetricValue}, prsm.node.readiness
.compute_readiness, and prsm.settlement.client_wiring.get_settlement_status
(the never-raising operator-status helper). Reads the node defensively + is
fail-soft PER signal block (a raising subsystem omits its metric, never breaks
collection — the contract MetricsRegistry.collect_all relies on).
"""
from __future__ import annotations

import asyncio

import pytest

from prsm.core.monitoring.metrics import CustomMetric, MetricValue
from prsm.core.monitoring.node_runtime_metrics import (
    InferenceServingCounters,
    NodeRuntimeMetrics,
    record_inference_result,
)


class _FakeSettlementClient:
    """Stands in for the node's BatchSettlementClient — get_settlement_status
    calls .status()."""

    def __init__(self, status_dict):
        self._status = status_dict

    def status(self):
        return dict(self._status)


class _Node:
    """Minimal node-like object. compute_readiness reads inference_executor /
    settlement_client / ftns_ledger; the settlement adapter reads
    _onchain_settlement_client; the inference adapter reads
    _inference_serving_counters."""

    def __init__(self, *, inference_executor=None, onchain_settlement_client=None,
                 inference_serving_counters=None):
        self.inference_executor = inference_executor
        self._onchain_settlement_client = onchain_settlement_client
        self._inference_serving_counters = inference_serving_counters


class _Result:
    def __init__(self, success, duration=None):
        self.success = success
        self.receipt = None if duration is None else type(
            "_R", (), {"duration_seconds": duration},
        )()


def _collect(metric: CustomMetric) -> dict:
    values = asyncio.run(metric.collect())
    assert all(isinstance(v, MetricValue) for v in values)
    return {v.name: v.value for v in values}


def test_is_custom_metric_with_name():
    m = NodeRuntimeMetrics(_Node())
    assert isinstance(m, CustomMetric)
    assert m.name  # has a non-empty metric-family name


def test_ready_node_no_settlement():
    node = _Node(inference_executor=object(), onchain_settlement_client=None)
    m = _collect(NodeRuntimeMetrics(node))
    assert m["prsm_node_ready"] == 1
    assert m["prsm_inference_ready"] == 1
    # settlement off → enabled gauge 0, no detail gauges
    assert m["prsm_settlement_enabled"] == 0
    assert "prsm_settlement_pending_commits" not in m
    assert "prsm_settlement_funds_in_flight" not in m


def test_unready_node_with_active_settlement_funds_in_flight():
    client = _FakeSettlementClient({
        "provider_address": "0xabc",
        "write_capable": True,
        "durable_state": True,
        "tracked_batches": 3,
        "finalized_locally": 1,
        "pending_commits": 2,
        "committing_intents": 1,
        "funds_in_flight": True,
    })
    node = _Node(inference_executor=None, onchain_settlement_client=client)
    m = _collect(NodeRuntimeMetrics(node))
    # inference dark → not ready
    assert m["prsm_node_ready"] == 0
    assert m["prsm_inference_ready"] == 0
    # settlement on → all detail gauges present + correct
    assert m["prsm_settlement_enabled"] == 1
    assert m["prsm_settlement_pending_commits"] == 2
    assert m["prsm_settlement_committing_intents"] == 1
    assert m["prsm_settlement_tracked_batches"] == 3
    assert m["prsm_settlement_finalized_locally"] == 1
    assert m["prsm_settlement_funds_in_flight"] == 1


def test_per_block_fail_soft_does_not_raise():
    class _ExplodingNode:
        # readiness reads .inference_executor → raises here
        @property
        def inference_executor(self):
            raise RuntimeError("boom")

        # settlement block still works
        _onchain_settlement_client = _FakeSettlementClient({
            "tracked_batches": 5, "pending_commits": 0,
            "committing_intents": 0, "finalized_locally": 5,
            "funds_in_flight": False,
        })

    m = _collect(NodeRuntimeMetrics(_ExplodingNode()))
    # readiness block was skipped (raised) — but collection did NOT raise
    assert "prsm_node_ready" not in m
    # settlement block still produced its metrics
    assert m["prsm_settlement_enabled"] == 1
    assert m["prsm_settlement_tracked_batches"] == 5
    assert m["prsm_settlement_funds_in_flight"] == 0


def test_settlement_status_error_is_tolerated():
    class _RaisingClient:
        def status(self):
            raise ValueError("rpc down")

    node = _Node(inference_executor=object(), onchain_settlement_client=_RaisingClient())
    # get_settlement_status swallows the error → enabled True + status_error, no
    # numeric detail keys → adapter emits enabled=1 + funds_in_flight=0, never raises
    m = _collect(NodeRuntimeMetrics(node))
    assert m["prsm_settlement_enabled"] == 1
    assert "prsm_settlement_pending_commits" not in m


# ── sp1219: inference serving counters ───────────────────────────────────────

def test_counter_records_success_failure_latency():
    c = InferenceServingCounters()
    c.record(success=True, latency_seconds=0.5)
    c.record(success=True, latency_seconds=1.5)
    c.record(success=False, latency_seconds=0.0)
    snap = c.snapshot()
    assert snap["requests_total"] == 3
    assert snap["failures_total"] == 1
    assert snap["latency_seconds_sum"] == 2.0


def test_counter_ignores_bad_latency():
    c = InferenceServingCounters()
    c.record(success=True, latency_seconds=None)  # type: ignore[arg-type]
    c.record(success=True, latency_seconds=-3.0)  # negative ignored
    assert c.snapshot() == {
        "requests_total": 2, "failures_total": 0, "latency_seconds_sum": 0.0,
    }


def test_record_inference_result_helper():
    node = _Node(inference_serving_counters=InferenceServingCounters())
    record_inference_result(node, _Result(True, duration=0.25))
    record_inference_result(node, _Result(False))
    snap = node._inference_serving_counters.snapshot()
    assert snap["requests_total"] == 2
    assert snap["failures_total"] == 1
    assert snap["latency_seconds_sum"] == 0.25


def test_record_inference_result_noop_without_counter():
    node = _Node(inference_serving_counters=None)
    # must not raise when the node has no counter
    record_inference_result(node, _Result(True, duration=1.0))


def test_adapter_emits_inference_counters():
    c = InferenceServingCounters()
    c.record(success=True, latency_seconds=0.4)
    c.record(success=False)
    node = _Node(inference_executor=object(), inference_serving_counters=c)
    m = _collect(NodeRuntimeMetrics(node))
    assert m["prsm_inference_requests_total"] == 2
    assert m["prsm_inference_failures_total"] == 1
    assert m["prsm_inference_latency_seconds_sum"] == 0.4


def test_adapter_omits_inference_counters_when_absent():
    node = _Node(inference_executor=object(), inference_serving_counters=None)
    m = _collect(NodeRuntimeMetrics(node))
    assert "prsm_inference_requests_total" not in m


# ── sp1244: attestation-collateral freshness metrics ─────────────────────────
# Bridges the sp1090 horizon-aware health status into Prometheus so an operator
# can ALERT when the TEE revocation/recency collateral has silently gone stale.

import datetime as _dt  # noqa: E402

from cryptography import x509  # noqa: E402
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from cryptography.x509.oid import NameOID  # noqa: E402

from prsm.compute.inference.collateral_refresh import (  # noqa: E402
    INTEL_PCK_PLATFORM_CRL_FILE,
    amd_crl_cache_file,
)


def _crl_pem(*, next_update):
    """A signature-valid CRL whose nextUpdate the status helper parses (the
    horizon that decides fresh vs stale)."""
    priv = ec.generate_private_key(ec.SECP256R1())
    # anchor lastUpdate before nextUpdate so the builder accepts a past nextUpdate
    # (a stale CRL) as well as a future one (a fresh CRL).
    crl = (x509.CertificateRevocationListBuilder()
           .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "CA")]))
           .last_update(next_update - _dt.timedelta(days=5)).next_update(next_update)
           .sign(priv, hashes.SHA256()))
    return crl.public_bytes(serialization.Encoding.PEM)


def _collect_list(metric):
    """Full MetricValue list (the by-name _collect helper drops labels, which
    the per-item collateral metrics depend on)."""
    values = asyncio.run(metric.collect())
    assert all(isinstance(v, MetricValue) for v in values)
    return values


def _series(values, name):
    return [v for v in values if v.name == name]


def test_collateral_metrics_absent_without_cache_dir(monkeypatch):
    # The common (non-TEE) node has no collateral cache configured → the adapter
    # must emit NO collateral metrics at all (no per-node noise).
    monkeypatch.delenv("PRSM_ATTESTATION_COLLATERAL_DIR", raising=False)
    values = _collect_list(NodeRuntimeMetrics(_Node(inference_executor=object())))
    assert not [v for v in values if v.name.startswith("prsm_collateral")]


def _future(days):
    return _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(days=days)


def test_collateral_metrics_fresh_crl(monkeypatch, tmp_path):
    (tmp_path / INTEL_PCK_PLATFORM_CRL_FILE).write_bytes(_crl_pem(next_update=_future(20)))
    monkeypatch.setenv("PRSM_ATTESTATION_COLLATERAL_DIR", str(tmp_path))
    monkeypatch.setenv("PRSM_COLLATERAL_AUTO_REFRESH", "1")
    values = _collect_list(NodeRuntimeMetrics(_Node(inference_executor=object())))

    enabled = _series(values, "prsm_collateral_refresh_enabled")
    assert len(enabled) == 1 and enabled[0].value == 1

    # per-item detail (labeled)
    item = _series(values, "prsm_collateral_item_stale")
    assert len(item) == 1
    assert item[0].labels == {"item": "intel_pck_platform_crl"}
    assert item[0].value == 0  # within nextUpdate → not stale

    # the unlabeled alert aggregate
    agg = _series(values, "prsm_collateral_stale")
    assert len(agg) == 1 and agg[0].labels == {} and agg[0].value == 0

    age = _series(values, "prsm_collateral_age_seconds")
    assert len(age) == 1 and age[0].value >= 0


def test_collateral_metrics_stale_when_past_next_update(monkeypatch, tmp_path):
    # A CRL whose nextUpdate is in the PAST = the verifier freshness gate is now
    # rejecting → both the per-item and the aggregate must flag stale=1.
    (tmp_path / INTEL_PCK_PLATFORM_CRL_FILE).write_bytes(_crl_pem(next_update=_future(-1)))
    monkeypatch.setenv("PRSM_ATTESTATION_COLLATERAL_DIR", str(tmp_path))
    values = _collect_list(NodeRuntimeMetrics(_Node(inference_executor=object())))
    item = _series(values, "prsm_collateral_item_stale")
    assert len(item) == 1 and item[0].value == 1
    agg = _series(values, "prsm_collateral_stale")
    assert len(agg) == 1 and agg[0].value == 1


def test_collateral_aggregate_is_max_over_items(monkeypatch, tmp_path):
    # THE regression guard: a fresh item AND a stale item present. The AlertManager
    # reduces a metric to ONE series, so the alert MUST read the unlabeled aggregate
    # — which has to be max-over-items (=1) even though one item is fresh. (A naive
    # per-label gauge could let the fresh series mask the stale one.)
    (tmp_path / INTEL_PCK_PLATFORM_CRL_FILE).write_bytes(_crl_pem(next_update=_future(20)))
    (tmp_path / amd_crl_cache_file("milan")).write_bytes(_crl_pem(next_update=_future(-1)))
    monkeypatch.setenv("PRSM_ATTESTATION_COLLATERAL_DIR", str(tmp_path))
    values = _collect_list(NodeRuntimeMetrics(_Node(inference_executor=object())))

    item = {v.labels["item"]: v.value for v in _series(values, "prsm_collateral_item_stale")}
    assert item["intel_pck_platform_crl"] == 0
    assert item["amd_milan_crl"] == 1

    agg = _series(values, "prsm_collateral_stale")
    assert len(agg) == 1 and agg[0].labels == {} and agg[0].value == 1   # any stale → 1


def test_collateral_block_fail_soft(monkeypatch, tmp_path):
    # A raising status helper must not break the whole collection (the contract
    # MetricsRegistry.collect_all relies on) — and the other blocks still emit.
    monkeypatch.setenv("PRSM_ATTESTATION_COLLATERAL_DIR", str(tmp_path))

    def _boom(*a, **k):
        raise RuntimeError("status exploded")

    monkeypatch.setattr(
        "prsm.compute.inference.collateral_refresh.collateral_refresh_status", _boom)
    m = _collect(NodeRuntimeMetrics(_Node(inference_executor=object())))
    assert m["prsm_node_ready"] == 1                 # other blocks unaffected
    assert "prsm_collateral_stale" not in m           # collateral block skipped


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
