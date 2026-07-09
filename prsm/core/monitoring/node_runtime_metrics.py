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
  - ``prsm_collateral_refresh_enabled`` + per-item ``prsm_collateral_item_stale``
    / ``prsm_collateral_age_seconds`` (label ``item``) + the unlabeled aggregate
    ``prsm_collateral_stale`` (max over items — the alert target) — sp1244, from
    the sp1090 ``collateral_refresh_status`` (horizon-aware: parses each cached
    item's real nextUpdate). Emitted ONLY when a collateral cache dir is
    configured, so non-TEE nodes stay silent. The machine-readable half of what
    was previously only on ``/health/detailed`` — lets the AlertManager warn when
    the TEE revocation/recency collateral has gone stale (auto-refresh silently
    failing). The aggregate exists because the AlertManager reduces a metric to a
    single series, so a per-label gauge cannot express "ANY item stale".

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
        # sp1382 — settlement challenge watcher: active challenges against this operator's batches
        # (the metric an operator alerts on) + the watcher's freshness.
        watcher = getattr(self._node, "_settlement_challenge_watcher", None)
        if watcher is not None:
            try:
                out.append(MetricValue(
                    "prsm_settlement_challenges_active",
                    int(watcher.active_challenge_count), now))
                age = watcher.last_tick_age_seconds()
                if age is not None:
                    out.append(MetricValue(
                        "prsm_settlement_challenge_watcher_last_tick_age_seconds", age, now))
            except Exception:  # noqa: BLE001 — metrics must never raise
                pass

        # sp1384/sp1411 — auto-defense verdict tallies, keyed on the challenge's ON-CHAIN reason
        # code (not on whether our retained receipt happened to verify — see challenge_auto_defense).
        #   legitimate    > 0 → a SLASHING challenge (DOUBLE_SPEND / INVALID_SIGNATURE /
        #                       CONSENSUS_MISMATCH) landed: stake at risk. THE ALERT TARGET.
        #   bad_faith     > 0 → NO_ESCROW: value invalidated, no slash. Requester-attestation
        #                       ground, so this is the griefing/harassment signal.
        #   expired       > 0 → EXPIRED: value invalidated, no slash (commit sooner).
        #   no_receipt    > 0 → could not self-assess (coverage, orthogonal to the three above).
        stats = getattr(self._node, "_challenge_defense_stats", None)
        if stats is not None:
            try:
                out.append(MetricValue(
                    "prsm_challenge_defense_bad_faith_total", int(stats.bad_faith), now))
                out.append(MetricValue(
                    "prsm_challenge_defense_legitimate_total", int(stats.legitimate), now))
                out.append(MetricValue(
                    "prsm_challenge_defense_no_receipt_total", int(stats.no_receipt), now))
                # sp1411 — appended LAST, read via getattr: these four share one try, so an
                # AttributeError on the newest field would otherwise silently drop the gauges
                # after it (exactly what a stats object predating this field would cause).
                out.append(MetricValue(
                    "prsm_challenge_defense_expired_total",
                    int(getattr(stats, "expired", 0)), now))
            except Exception:  # noqa: BLE001 — metrics must never raise
                pass

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

        # ── attestation collateral freshness (sp1244 — TEE revocation/recency) ──
        # Bridges the sp1090 health-JSON status (collateral_refresh_status, which is
        # horizon-aware: it parses each cached item's real nextUpdate) into Prometheus
        # + the AlertManager. Without this an operator scraping /metrics had NO signal
        # that the collateral auto-refresh (sp1081/1082) had silently stopped — once a
        # cached CRL/TCB-Info ages past its nextUpdate the verifiers' freshness gates
        # (sp1060/sp1089) start rejecting, i.e. revocation/recency enforcement degrades.
        # Gated on a configured collateral cache dir so non-TEE nodes emit nothing.
        try:
            from prsm.compute.inference.collateral_refresh import (
                collateral_refresh_status,
            )

            st = collateral_refresh_status()
            if st.get("cache_dir"):
                out.append(MetricValue(
                    "prsm_collateral_refresh_enabled",
                    1 if st.get("enabled") else 0, now,
                ))
                any_stale = 0
                for item, d in (st.get("items") or {}).items():
                    labels = {"item": str(item)}
                    age = d.get("age_seconds")
                    if isinstance(age, (int, float)) and not isinstance(age, bool):
                        out.append(MetricValue(
                            "prsm_collateral_age_seconds", age, now, labels,
                        ))
                    # item_stale=1 unless provably fresh (now <= nextUpdate). fresh
                    # is None when the horizon is unparseable/absent → the verifier
                    # can't trust it either, so flag it (conservative).
                    item_stale = 0 if d.get("fresh") is True else 1
                    out.append(MetricValue(
                        "prsm_collateral_item_stale", item_stale, now, labels,
                    ))
                    any_stale = max(any_stale, item_stale)
                # The UNLABELED max-over-items aggregate is the ALERT target: the
                # AlertManager reduces a metric to ONE series (latest by timestamp),
                # so a per-label gauge can't express "any item stale" — this scalar
                # can. Always emitted on a collateral-configured node (0 when nothing
                # is cached yet) so the alert rule always has a series to evaluate.
                out.append(MetricValue("prsm_collateral_stale", any_stale, now))
        except Exception:  # noqa: BLE001
            pass

        return out
