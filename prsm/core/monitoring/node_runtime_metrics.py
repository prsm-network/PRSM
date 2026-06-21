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

        return out
