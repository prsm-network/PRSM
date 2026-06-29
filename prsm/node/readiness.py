"""Node readiness computation (sprint 1186, day-one-live blocker #9).

Liveness vs readiness: ``/health`` is LIVENESS (the process is up — restart me if it
isn't). This module computes READINESS — "should this node receive traffic?" — which the
``/health/ready`` + ``/readyz`` probes expose so a load balancer DRAINS traffic from a
node that can't serve, instead of routing requests to a broken one (and instead of k8s
RESTARTING a merely-not-ready pod, which a 503 on the liveness probe would wrongly cause).

Readiness is gated on the PRIMARY request path: can the node serve inference? (After
sp1184 a properly-installed node serves inference by default, so it is ready; a node
without the .[ml] extra, or with inference explicitly disabled, is correctly not-ready for
inference traffic.) The detail body carries only COARSE non-sensitive subsystem booleans,
so the probe is safe to expose ungated to the load balancer. Cheap: pure attribute reads,
no RPC — safe to hit at probe frequency.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple


def compute_readiness(node: Any) -> Tuple[bool, Dict[str, Any]]:
    """Return ``(ready, detail)``. ``ready`` is True iff the node can serve its primary
    path (an inference executor is wired). ``detail`` is a small JSON-able dict with a
    ``reason`` (when not ready) and coarse subsystem booleans. Never raises."""
    def _present(attr: str) -> bool:
        return getattr(node, attr, None) is not None

    inference = _present("inference_executor")
    subsystems = {
        "inference": inference,
        # sp1217 — the node stores the on-chain settlement client as
        # ``_onchain_settlement_client`` (node.py); reading ``settlement_client``
        # (which the node never sets) made this boolean ALWAYS False on a real
        # node — a latent readiness mis-report. Read the real attr.
        "settlement": _present("_onchain_settlement_client"),
        "ftns_ledger": _present("ftns_ledger"),
    }
    ready = inference
    detail: Dict[str, Any] = {"ready": ready, "subsystems": subsystems}
    # sp1303 — surface the local-inference model-LOADED state (sp1302 pre-warm). The
    # ``ready`` gate stays on WIRED (not loaded): a wired-but-warming node can still
    # serve (it just lazy-loads on the first request), so it must not drain traffic
    # mid-pre-warm. ``inference_detail.loaded`` lets an operator / smart LB see whether
    # the model is actually resident yet. Additive + fail-soft (never changes ``ready``).
    try:
        from prsm.compute.inference.local_inference import local_inference_readiness
        detail["inference_detail"] = local_inference_readiness(
            getattr(node, "inference_executor", None))
    except Exception:  # noqa: BLE001 — readiness must stay cheap + never raise
        pass
    if not ready:
        detail["reason"] = (
            "inference_executor_unavailable — the node cannot serve inference "
            "(install the .[ml] extra or set PRSM_INFERENCE_EXECUTOR), so it is not "
            "ready to receive inference traffic"
        )
    return ready, detail
