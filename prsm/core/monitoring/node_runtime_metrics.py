"""Sprint 1217 — NodeRuntimeMetrics: live-node runtime signals for the metrics
stack (Brick 1 of the node-observability arc).

A ``CustomMetric`` adapter that maps a running ``PRSMNode``'s highest-value,
already-available runtime signals into ``MetricValue``s the existing
``MetricsCollector``/``MetricsRegistry`` collects and the ``AlertManager`` can
alert on. It is a PURE adapter — registering it changes no node lifecycle; it
reads node state defensively and is FAIL-SOFT per signal block (a raising
subsystem omits its metric and never breaks ``collect()``, which
``MetricsRegistry.collect_all`` depends on).

Signals (Brick 1):
  - ``prsm_node_ready`` / ``prsm_inference_ready`` — from the sp1186 readiness
    probe (the primary request path).
  - ``prsm_settlement_enabled`` + (when on) ``pending_commits`` /
    ``committing_intents`` / ``tracked_batches`` / ``finalized_locally`` /
    ``funds_in_flight`` — from the sp1051 ``get_settlement_status`` helper, read
    off the node's ``_onchain_settlement_client`` (the authoritative attr;
    note readiness.py reads ``settlement_client`` which the node does NOT set,
    so the settlement signal MUST come from this helper, not the readiness
    detail).

Reuses ``prsm.node.readiness.compute_readiness`` and
``prsm.settlement.client_wiring.get_settlement_status`` — both never-raising —
so this module adds no new probing logic, only the metric mapping.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, List

from prsm.core.monitoring.metrics import CustomMetric, MetricValue


class InferenceServingCounters:
    """Sprint 1219 — process-local cumulative counters for the inference
    serving path (the PRIMARY request path of a day-one inference node).

    Monotonic totals + a latency sum so an operator's Prometheus can derive
    request rate / error rate / avg latency via ``rate()`` and
    ``…_sum / …_total`` — the correct way to alert on RATES (a cumulative
    counter must not be threshold-alerted directly). Cheap + always-on; the
    metrics only surface when the opt-in MetricsCollector reads them."""

    __slots__ = ("requests_total", "failures_total", "latency_seconds_sum")

    def __init__(self) -> None:
        self.requests_total = 0
        self.failures_total = 0
        self.latency_seconds_sum = 0.0

    def record(self, *, success: bool, latency_seconds: float = 0.0) -> None:
        self.requests_total += 1
        if not success:
            self.failures_total += 1
        try:
            lat = float(latency_seconds)
            if lat > 0:
                self.latency_seconds_sum += lat
        except (TypeError, ValueError):
            pass

    def snapshot(self) -> dict:
        return {
            "requests_total": self.requests_total,
            "failures_total": self.failures_total,
            "latency_seconds_sum": round(self.latency_seconds_sum, 6),
        }


def record_inference_result(node: Any, result: Any) -> None:
    """Sprint 1219 — record ONE inference outcome into the node's serving
    counters. Fail-soft no-op when the node has no counters (or anything
    raises) — instrumentation must never affect the request path. ``result``
    is an InferenceResult-like with ``.success`` and an optional
    ``.receipt.duration_seconds`` (used as the served latency on success)."""
    try:
        counters = getattr(node, "_inference_serving_counters", None)
        if counters is None:
            return
        success = bool(getattr(result, "success", False))
        latency = 0.0
        if success:
            receipt = getattr(result, "receipt", None)
            if receipt is not None:
                latency = float(getattr(receipt, "duration_seconds", 0.0) or 0.0)
        counters.record(success=success, latency_seconds=latency)
    except Exception:  # noqa: BLE001 — never affect the inference path
        pass


class NodeRuntimeMetrics(CustomMetric):
    """Expose a live node's readiness + on-chain-settlement runtime state."""

    def __init__(self, node: Any) -> None:
        super().__init__(
            "prsm_node_runtime",
            "PRSM live-node runtime metrics (readiness + on-chain settlement)",
        )
        self._node = node

    async def collect(self) -> List[MetricValue]:
        now = datetime.now()
        out: List[MetricValue] = []

        # ── readiness / inference (primary request path) ──
        # Each block is independently fail-soft: a raising subsystem omits its
        # metrics rather than failing the whole collection.
        try:
            from prsm.node.readiness import compute_readiness

            ready, detail = compute_readiness(self._node)
            out.append(MetricValue("prsm_node_ready", 1 if ready else 0, now))
            subs = detail.get("subsystems", {}) if isinstance(detail, dict) else {}
            out.append(
                MetricValue(
                    "prsm_inference_ready",
                    1 if subs.get("inference") else 0,
                    now,
                )
            )
        except Exception:  # noqa: BLE001 — a metrics collect must never raise
            pass

        # ── on-chain settlement (real money mid-flight) ──
        try:
            from prsm.settlement.client_wiring import get_settlement_status

            client = getattr(self._node, "_onchain_settlement_client", None)
            st = get_settlement_status(client)
            enabled = bool(st.get("enabled"))
            out.append(MetricValue("prsm_settlement_enabled", 1 if enabled else 0, now))
            if enabled:
                for key, name in (
                    ("pending_commits", "prsm_settlement_pending_commits"),
                    ("committing_intents", "prsm_settlement_committing_intents"),
                    ("tracked_batches", "prsm_settlement_tracked_batches"),
                    ("finalized_locally", "prsm_settlement_finalized_locally"),
                ):
                    val = st.get(key)
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        out.append(MetricValue(name, val, now))
                # funds_in_flight only emitted when the snapshot is healthy
                # (numeric detail present); a status_error snapshot lacks it.
                if "funds_in_flight" in st:
                    out.append(
                        MetricValue(
                            "prsm_settlement_funds_in_flight",
                            1 if st.get("funds_in_flight") else 0,
                            now,
                        )
                    )
        except Exception:  # noqa: BLE001
            pass

        # ── inference serving (the primary request path) ──
        try:
            counters = getattr(self._node, "_inference_serving_counters", None)
            if counters is not None:
                snap = counters.snapshot()
                out.append(MetricValue(
                    "prsm_inference_requests_total",
                    snap["requests_total"], now,
                ))
                out.append(MetricValue(
                    "prsm_inference_failures_total",
                    snap["failures_total"], now,
                ))
                out.append(MetricValue(
                    "prsm_inference_latency_seconds_sum",
                    snap["latency_seconds_sum"], now,
                ))
        except Exception:  # noqa: BLE001
            pass

        return out
