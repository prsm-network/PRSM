"""
Node Management API
===================

FastAPI endpoints for monitoring and controlling a running PRSM node.
This is the node-local API (not the main PRSM platform API).

Security Features (Phase 4.2):
- JWT authentication on protected endpoints
- Rate limiting with configurable limits
- WebSocket status updates
- OpenAPI specification
"""

import json
import logging
import os
import time as _time_for_history
import uuid as _uuid
from typing import Annotated, Any, Dict, List, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Header, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, StringConstraints

from prsm.node.api_hardening import (
    APIHardening,
    APISecurityConfig,
    StatusWebSocket,
    require_auth,
)

logger = logging.getLogger(__name__)


class JobSubmission(BaseModel):
    """Request body for submitting a compute job."""
    job_type: str  # inference, embedding, benchmark
    payload: Dict[str, Any] = {}
    # Sprint 204 — bound ftns_budget. Pre-fix the bare `float =
    # 1.0` accepted negative, zero, and absurdly-large values that
    # then propagated to compute_requester.submit_job with
    # undefined-behavior downstream. Upper bound is a sane absolute
    # ceiling; PRSM_MAX_FTNS_PER_JOB on /forge is the operator
    # cost-control knob, this is just garbage-rejection.
    ftns_budget: float = Field(
        default=1.0, gt=0, le=1e12, allow_inf_nan=False,
        description="FTNS budget for the job (must be > 0)",
    )


class ResourceUpdateRequest(BaseModel):
    """Request model for updating node resource settings."""
    cpu_allocation_pct: Optional[int] = Field(default=None, ge=10, le=90, description="CPU allocation percentage (10-90)")
    memory_allocation_pct: Optional[int] = Field(default=None, ge=10, le=90, description="Memory allocation percentage (10-90)")
    # Sprint 207 — upper bounds, allow_inf_nan=False, active_days
    # max_items. 1 PB storage / 1M-job concurrency / 1 Tbps cap is
    # well past any plausible single-node value but below any
    # value that would crash downstream schedulers.
    storage_gb: Optional[float] = Field(
        default=None, gt=0, le=1_000_000_000, allow_inf_nan=False,
        description="Storage pledge in GB",
    )
    max_concurrent_jobs: Optional[int] = Field(
        default=None, ge=1, le=1_000_000,
        description="Maximum concurrent jobs",
    )
    gpu_allocation_pct: Optional[int] = Field(default=None, ge=10, le=100, description="GPU allocation percentage (10-100)")
    upload_mbps_limit: Optional[float] = Field(
        default=None, ge=0, le=1_000_000, allow_inf_nan=False,
        description="Upload bandwidth limit in Mbps (0=unlimited)",
    )
    download_mbps_limit: Optional[float] = Field(
        default=None, ge=0, le=1_000_000, allow_inf_nan=False,
        description="Download bandwidth limit in Mbps (0=unlimited)",
    )
    active_hours_start: Optional[int] = Field(default=None, ge=0, le=23, description="Active hours start (0-23)")
    active_hours_end: Optional[int] = Field(default=None, ge=0, le=23, description="Active hours end (0-23)")
    active_days: Optional[List[int]] = Field(
        default=None, max_length=7,
        description="Active days (0=Mon...6=Sun, empty=every day)",
    )


class ResourceConfigResponse(BaseModel):
    """Response model for current resource configuration."""
    cpu_allocation_pct: int
    memory_allocation_pct: int
    storage_gb: float
    max_concurrent_jobs: int
    gpu_allocation_pct: int
    upload_mbps_limit: float
    download_mbps_limit: float
    active_hours_start: Optional[int]
    active_hours_end: Optional[int]
    active_days: List[int]
    # Computed values
    effective_cpu_cores: float
    effective_memory_gb: float
    effective_gpu_memory_gb: Optional[float]
    storage_available_gb: float


class ContentUploadRequest(BaseModel):
    """Request body for uploading text content."""
    # Sprint 208 added a 10MB Pydantic max_length as a DoS
    # guard. Sprint 333 bumped it to 100MB because the previous
    # static cap collided with the sprint 102 contract — when
    # an operator sets PRSM_MAX_UPLOAD_BYTES>10MB intending to
    # accept larger uploads, the Pydantic validator rejected
    # with 422 before the env-aware 413 path could fire. The
    # 100MB ceiling is the policy max: anything larger should
    # use /content/upload/shard. The runtime env-driven cap
    # still fires inside the handler (`text_bytes > _size_cap`)
    # so operators get the canonical 413 with
    # PRSM_MAX_UPLOAD_BYTES in the detail message.
    text: str = Field(..., max_length=100 * 1024 * 1024)
    filename: str = Field(default="document.txt", max_length=512)
    # Sprint 160 — Pydantic field constraints. Pre-fix royalty_rate
    # and replicas were unconstrained, so out-of-band values
    # (negative royalties, zero/negative replicas, 10000% rates)
    # silently passed model validation and either produced wrong
    # downstream behavior or hit generic errors.
    replicas: int = Field(
        default=3, ge=0, le=1000,
        description="Replication factor (0=local-only, max 1000)",
    )
    royalty_rate: Optional[float] = Field(
        default=None, ge=0.001, le=0.1,
        description="FTNS earned per access (0.001–0.1, default 0.01)",
    )
    parent_cids: List[Annotated[str, StringConstraints(max_length=256)]] = Field(
        default_factory=list,
        max_length=10_000,
        description="CIDs of source material this content derives from",
    )
    # Sprint 243 — capture creator's on-chain ETH address for the
    # eventual RoyaltyDistributor.distribute_royalty() destination.
    # Optional (default None) for backwards-compat with v1 uploads.
    # Validated upfront as 0x-prefixed 40-hex-char address.
    creator_eth_address: Optional[
        Annotated[str, StringConstraints(
            pattern=r"^0x[0-9a-fA-F]{40}$",
        )]
    ] = Field(
        default=None,
        description=(
            "Optional 0x-prefixed Ethereum address. Used as the "
            "destination for on-chain royalty distribution when "
            "the leg is wired."
        ),
    )
    # Sprint 304 — Enterprise Confidentiality Mode.
    # Optional. When present, `text` is encrypted with
    # X25519+ChaCha20-Poly1305 hybrid before sharding; only
    # the listed recipients can decrypt. FTNS balance is
    # irrelevant to the cryptography — encryption is the
    # security primitive, the payment gate is orthogonal.
    recipients: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description=(
            "Optional list of {identifier, x25519_pubkey_b64}"
            " dicts. When provided, the text is "
            "recipient-encrypted before sharding (Vision §7 "
            "Enterprise Confidentiality Mode)."
        ),
    )
    # Sprint 307 — threshold (t-of-n) encryption mode.
    # When set together with `recipients`, the symmetric
    # key is Shamir-split: any t of the n recipients must
    # cooperate to decrypt; t-1 reveal nothing.
    # Composes onto the same EncryptedPayload shape with
    # manifest.threshold + per-entry share_index.
    threshold: Optional[int] = Field(
        default=None, ge=1, le=255,
        description=(
            "Optional t-of-n threshold. Requires "
            "`recipients` to be set. When provided, the "
            "upload is encrypted in threshold mode: any t "
            "of the n recipients must cooperate to decrypt."
        ),
    )



# ──────────────────────────────────────────────────────────────────────
# Phase 3.x.8.1 — SSE helpers for /compute/inference/stream
# ──────────────────────────────────────────────────────────────────────


def _sse_event(event_type: str, data: Any) -> bytes:
    """Encode a single Server-Sent Events frame.

    Per the W3C SSE spec: ``event:`` + ``data:`` lines, terminated by
    a blank line. ``data`` is JSON-serialized; complex objects flow
    through ``json.dumps`` with ``default=str`` as a fallback for any
    Decimal / bytes / dataclass values that slip through the
    pre-encoders.
    """
    import json as _json

    if not isinstance(data, str):
        data = _json.dumps(data, default=str)
    return f"event: {event_type}\ndata: {data}\n\n".encode("utf-8")


def _token_event_to_dict(event: Any) -> Dict[str, Any]:
    """Encode an ``InferenceTokenEvent`` into the SSE ``data`` payload
    shape. Optional fields (token_id / finish_reason) are emitted as
    ``null`` when absent so consumer parsers see a stable schema."""
    return {
        "sequence_index": event.sequence_index,
        "text_delta": event.text_delta,
        "token_id": event.token_id,
        "finish_reason": event.finish_reason,
    }


def _result_to_dict(
    result: Any,
    *,
    job_id: str,
    identity: Optional[Any] = None,
) -> Dict[str, Any]:
    """Encode an ``InferenceResult`` into the SSE ``data`` payload
    shape. Mirrors the unary endpoint's success response with the
    job_id rebound to the API-side id (executor uses an internal
    parallax-stream-job-* id; the API is authoritative for billing
    correlation).

    When ``identity`` is provided, the rebound receipt is re-signed
    under that identity. The ``job_id`` is part of the signed
    payload — rebinding without re-signing would invalidate the
    settler signature. Identity-less callers (tests / dry-run
    encoding helpers) get a receipt with a stale signature; callers
    receiving over the wire MUST pass identity to preserve the
    cryptographic invariant.
    """
    import dataclasses

    receipt = result.receipt
    if receipt is not None:
        receipt = dataclasses.replace(receipt, job_id=job_id)
        if identity is not None:
            try:
                from prsm.compute.inference import sign_receipt
                receipt = sign_receipt(receipt, identity)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"Streaming result receipt re-sign failed for "
                    f"job_id={job_id}: {e}"
                )
    return {
        "success": result.success,
        "job_id": job_id,
        "request_id": result.request_id,
        "output": result.output,
        "receipt": receipt.to_dict() if receipt is not None else None,
    }


async def _settle_streaming_escrow(
    node: Any,
    job_id: str,
    escrow_entry: Any,
    request: Any,
    result: Any,
) -> None:
    """Settle escrow + record privacy-budget spend for a successful
    streaming inference.

    Phase 3.x.8.1 round-1 L1 remediation: the receipt re-sign
    previously done here was dead code — the wire-side re-sign in
    ``_result_to_dict(item, job_id=..., identity=node.identity)`` is
    the authoritative path. Re-signing here computed a value that
    was immediately discarded (the local ``rebound`` receipt object
    wasn't propagated anywhere). Removed.

    Privacy budget tracking still fires here on the success path —
    it's the right layer to record DP epsilon spend (the executor
    yielded a complete result with measured ``epsilon_spent``).
    """
    if not escrow_entry or not getattr(node, "_payment_escrow", None):
        return

    # Privacy budget tracking — same as unary path. Only fires on
    # the success path (a complete result with measured
    # epsilon_spent). Mid-stream failures don't produce a receipt,
    # so this code never sees them — they go through the
    # _resolve_post_token_billing settle-without-privacy-budget
    # path instead.
    if (
        hasattr(node, "privacy_budget")
        and node.privacy_budget
        and request.privacy_tier.value != "none"
        and result.receipt is not None
    ):
        try:
            node.privacy_budget.record_spend(
                result.receipt.epsilon_spent, "inference", job_id,
                model_id=getattr(request, "model_id", ""),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Streaming privacy budget tracking failed for job_id={job_id}: {e}"
            )

    operator_id = (
        node.identity.node_id if node.identity else "self"
    )
    # Sprint 784 — route through settle_inference_receipt when we
    # have an InferenceReceipt in scope so partial_completion
    # (sprint 777) drives proportional credit + refund + slash
    # signal. Fall back to release_escrow when no receipt is
    # present (legacy path / mid-stream failure that produced no
    # receipt — preserves pre-784 behavior).
    if (
        getattr(result, "receipt", None) is not None
        and hasattr(result.receipt, "cost_ftns")
    ):
        try:
            from decimal import Decimal as _D
            from prsm.economy.credit_policy import (
                settle_inference_receipt,
            )
            escrow_amount = _D(str(getattr(escrow_entry, "amount", 0)))
            decision = await settle_inference_receipt(
                payment_escrow=node._payment_escrow,
                receipt=result.receipt,
                operator_id=operator_id,
                job_id=job_id,
                escrow_amount=escrow_amount,
            )
            # Sprint 1037 (brick 1.5) — on-chain settlement accumulate
            # (opt-in; no-op when off). Fail-open (parity with the unary path).
            try:
                from prsm.settlement.client_wiring import (
                    accumulate_settled_inference_receipt,
                )
                _paid = _take_paid_requester(node, job_id)
                _acc = await accumulate_settled_inference_receipt(
                    client=getattr(node, "_onchain_settlement_client", None),
                    identity=node.identity,
                    provider_address=getattr(node, "_operator_address", None),
                    receipt=result.receipt,
                    release_ftns=decision.release_to_operator,
                    job_id=job_id,
                    requester_address=(_paid or {}).get("requester"),
                    max_spend_wei=(_paid or {}).get("max_spend_wei"),
                    # sp1144 — opt-in §7 retention (default None when audit OFF).
                    inference_receipt_store=getattr(
                        node, "_settlement_inference_receipt_store", None,
                    ),
                )
                if _acc != "skipped:no-client":
                    logger.info(
                        "Sprint 1037 on-chain settlement accumulate "
                        "(stream job=%s): %s", job_id, _acc,
                    )
                # sp1095/1096 — capture: release the unused remainder (or, when the
                # on-chain accumulate was skipped, the full reservation).
                await _capture_relayer_budget(
                    node, _paid, decision.release_to_operator, _acc)
            except Exception as _acc_exc:  # noqa: BLE001
                logger.debug(
                    "Sprint 1037 accumulate hook error: %s", _acc_exc,
                )
            if decision.should_slash:
                logger.warning(
                    "Sprint 784 — slash signal for job_id=%s: "
                    "partial_completion.reason=error (operator-"
                    "attributable). On-chain slash emission "
                    "deferred to a later sprint.",
                    job_id,
                )
                # Sprint 798 — persist to operator-visible ring.
                ring = getattr(
                    node, "_partial_completion_event_log", None,
                )
                if ring is not None:
                    try:
                        info = result.receipt.partial_completion
                        ring.append(
                            job_id=job_id,
                            operator_node_id=operator_id,
                            reason=getattr(info, "reason", "error"),
                            tokens_completed=int(
                                getattr(info, "tokens_completed", 0),
                            ),
                            tokens_requested=int(
                                getattr(info, "tokens_requested", 0),
                            ),
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "Sprint 798 — partial_completion ring "
                            "append failed: %s", exc,
                        )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Streaming receipt-aware settle failed for "
                f"job_id={job_id}: {e}"
            )
    else:
        try:
            await node._payment_escrow.release_escrow(
                job_id=job_id,
                provider_id=operator_id,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Streaming escrow release failed for job_id={job_id}: {e}"
            )


async def _resolve_post_token_billing(
    node: Any,
    job_id: str,
    escrow_entry: Any,
    tokens_emitted: int,
    reason: str,
) -> None:
    """Phase 3.x.8.1 round-1 M1 remediation: design plan §3.4
    settle-on-tokens-emitted policy.

    When the streaming pipeline fails AFTER one or more tokens have
    been emitted on the wire, the requester has consumed the
    network's compute work — the chain executors burned cycles on
    those tokens. Refunding the requester would mean the network
    eats the cost (billing griefing vector: a malicious node could
    emit N tokens then crash, paying nothing despite forcing real
    compute work).

    Policy: tokens emitted > 0 → settle (release escrow at full
    estimate, no privacy-budget recording — we don't have a
    complete receipt with measured epsilon for a partial stream).
    Tokens emitted == 0 → refund (pre-execute failure or
    immediately-failing dispatch — caller paid nothing for nothing).
    """
    if not escrow_entry or not getattr(node, "_payment_escrow", None):
        return
    if tokens_emitted > 0:
        try:
            await node._payment_escrow.release_escrow(
                job_id=job_id,
                provider_id=node.identity.node_id if node.identity else "self",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Streaming post-token escrow release failed for "
                f"job_id={job_id}: {e}"
            )
    else:
        try:
            await node._payment_escrow.refund_escrow(job_id, reason)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Streaming escrow refund failed for job_id={job_id}: {e}"
            )


async def _refund_streaming_escrow(
    node: Any,
    job_id: str,
    escrow_entry: Any,
    reason: str,
) -> None:
    """Refund escrow for a streaming inference that ended without a
    successful result. Swallows refund-time exceptions (logged) —
    they're non-actionable at the wire boundary."""
    if not escrow_entry or not getattr(node, "_payment_escrow", None):
        return
    try:
        await node._payment_escrow.refund_escrow(job_id, reason)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"Streaming escrow refund failed for job_id={job_id}: {e}"
        )


_inference_semaphore: Optional[Any] = None  # asyncio.Semaphore, lazy
_inference_semaphore_limit: Optional[int] = None


def _get_inference_semaphore():
    """Sprint 704 — global concurrency gate on /compute/inference and
    /compute/inference/stream. Returns None when PRSM_INFERENCE_
    CONCURRENCY_LIMIT is unset (default — preserves pre-704 behavior)
    or an asyncio.Semaphore(N) when set to a positive integer.

    Memory-tight droplets (NYC's 2GB tier) OOM'd under sprint 698's
    cross-host coordination because a second inference arrived while
    the first was still loading gpt2 (~500MB) + processing activations.
    A semaphore=1 serializes inference handling so the peak memory
    footprint is bounded by a single request's working set.

    Lazy single-process singleton. Re-reads the env on every call so
    operators can flip the limit at runtime via systemctl edit + reload
    (the existing systemd unit Reload signal won't recreate the
    asyncio Semaphore but the env reread + comparison ensures the new
    daemon process picks up the change on restart).
    """
    global _inference_semaphore, _inference_semaphore_limit
    import asyncio as _asyncio
    raw = os.environ.get("PRSM_INFERENCE_CONCURRENCY_LIMIT", "").strip()
    if not raw:
        _inference_semaphore = None
        _inference_semaphore_limit = None
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    if n <= 0:
        return None
    if _inference_semaphore is None or _inference_semaphore_limit != n:
        _inference_semaphore = _asyncio.Semaphore(n)
        _inference_semaphore_limit = n
    return _inference_semaphore


def register_parallax_pool_snapshot_endpoint(app: Any, node: Any) -> None:
    """Sprint 685 — /admin/parallax/pool/snapshot.

    Read-only introspection of the GPU pool the ParallaxScheduled-
    Executor would see for its next allocation. Operators use this
    to live-attest sprint-682's DHT-backed pool: after setting
    PRSM_PARALLAX_GPU_POOL_KIND=dht-backed on multiple nodes,
    confirm each daemon's snapshot reports both self + the other
    peer(s).

    Status codes:
      503 — no inference_executor wired or executor lacks
            _pool_provider (operator hasn't opted in).
      500 — provider raised at invocation time.
      200 — {pool_kind, gpu_count, gpus[]}.
    """
    import os as _os
    from fastapi import HTTPException

    @app.get("/admin/parallax/pool/snapshot", tags=["admin"])
    async def parallax_pool_snapshot() -> Dict[str, Any]:
        executor = getattr(node, "inference_executor", None)
        if executor is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Inference executor not initialized. Opt in "
                    "via PRSM_PARALLAX_GPU_POOL_KIND=dht-backed + "
                    "PRSM_PARALLAX_TRUST_STACK_KIND + "
                    "PRSM_PARALLAX_MODEL_CATALOG_FILE."
                ),
            )
        provider = getattr(executor, "_pool_provider", None)
        if provider is None or not callable(provider):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Inference executor lacks _pool_provider — "
                    "not a ParallaxScheduledExecutor instance."
                ),
            )
        try:
            pool = list(provider())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        gpus = []
        for g in pool:
            gpus.append({
                "node_id": getattr(g, "node_id", ""),
                "region": getattr(g, "region", ""),
                "layer_capacity": getattr(g, "layer_capacity", 0),
                "stake_amount": getattr(g, "stake_amount", 0),
                "tier_attestation": getattr(g, "tier_attestation", ""),
                "tflops_fp16": getattr(g, "tflops_fp16", 0.0),
                "memory_gb": getattr(g, "memory_gb", 0.0),
                "memory_bandwidth_gbps": getattr(
                    g, "memory_bandwidth_gbps", 0.0,
                ),
                "gpu_name": getattr(g, "gpu_name", ""),
                "device": getattr(g, "device", ""),
                "num_gpus": getattr(g, "num_gpus", 1),
            })
        # Sprint 686 — surface stake-eligibility mode so operators
        # can see which posture the daemon is running in without
        # SSH-ing to read systemd-show.
        _elig_raw = (
            _os.environ.get("PRSM_PARALLAX_STAKE_ELIGIBILITY", "")
            .strip().lower()
        )
        stake_eligibility = "advisory" if _elig_raw == "advisory" else "enforced"
        return {
            "pool_kind": _os.environ.get(
                "PRSM_PARALLAX_GPU_POOL_KIND", "",
            ) or None,
            "stake_eligibility": stake_eligibility,
            "gpu_count": len(gpus),
            "gpus": gpus,
        }


def _check_active_window_or_503() -> None:
    """Sprint 756 — gate inference dispatch on the operator's
    active-window schedule. If PRSM_ACTIVE_HOURS is set and we're
    currently OUTSIDE the window, raise 503 with an actionable
    error pointing the caller at the schedule.

    Backward-compat: env unset → always returns (no behavior
    change for operators who haven't set a schedule).

    Returns None on accept. Raises HTTPException(503) on reject.
    """
    from prsm.node.schedule import (
        is_currently_active, get_active_window,
    )
    if is_currently_active():
        return
    w = get_active_window()
    # Render only when we know we have a non-None schedule (the
    # is_currently_active guard above ensures this).
    window_desc = w.render() if w else "operator-configured schedule"
    raise HTTPException(
        status_code=503,
        detail=(
            f"Node is outside its active window ({window_desc}). "
            f"Try again during scheduled hours, or contact the "
            f"operator to adjust PRSM_ACTIVE_HOURS."
        ),
        # Retry-After is a rough hint; the real next-active time
        # would require computing the next window-start which is
        # tz-aware + DST-aware. 60s is a reasonable polling cadence
        # — clients can re-probe each minute.
        headers={"Retry-After": "60"},
    )


def _check_not_preempted_or_503() -> None:
    """Sprint 774 — gate inference dispatch on the cloud-spot
    preemption flag. If sprint-772's PreemptionDetector has been
    signaled, raise 503 so peers stop routing new work here while
    in-flight requests drain inside the ~2min preemption warning
    window.

    Backward-compat: detector unset → always returns (no behavior
    change for non-cloud nodes).

    Returns None on accept. Raises HTTPException(503) on reject.
    """
    from prsm.node.preemption import is_currently_preempted
    if not is_currently_preempted():
        return
    raise HTTPException(
        status_code=503,
        detail=(
            "Node has received a cloud-spot preemption notice "
            "and is draining in-flight work. New requests will "
            "succeed once this node exits and a sibling daemon "
            "picks up the route."
        ),
        # Cloud-spot preemption notices give ~2min warning. By the
        # time the client retries in 120s, this node is gone +
        # peers have routed past it. Retry-After is the operator-
        # facing hint.
        headers={"Retry-After": "120"},
    )


def _host_is_loopback(host: str) -> bool:
    """sp1103 — module-level loopback check (mirrors the admin-loopback middleware's
    nested _is_loopback): canonical loopback names, the whole 127/8 block, and the
    IPv4-mapped IPv6 loopback form a dual-stack daemon sees."""
    if not host:
        return False
    if host in ("127.0.0.1", "::1", "localhost", "testclient"):
        return True
    if host.lower().startswith("::ffff:127."):
        return True
    if host.startswith("127.") and host.count(".") == 3:
        parts = host.split(".")
        if all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
            return True
    return False


def _resolve_requester_key(request: Any) -> str:
    """Sprint 741 F69 — resolve a per-requester rate-limit bucket
    key from an HTTP request. Proxy-aware: prefers X-Forwarded-For
    last-hop, then X-Real-IP, then `request.client.host`. Mirrors
    the same precedence sprint 737/738 use for admin-loopback so
    behavior is consistent across the codebase.

    Returns "anonymous" if no identifiable source can be derived
    (defensive — shouldn't happen with FastAPI/Starlette but the
    fallback ensures the rate limiter still works rather than
    crashing).
    """
    client = getattr(request, "client", None)
    host = getattr(client, "host", "") if client else ""
    # sp1103 (Domain-07 review MEDIUM) — only trust the proxy-forwarded client headers
    # when the IMMEDIATE socket peer is loopback (i.e. a co-located trusted reverse
    # proxy). On a directly-exposed node these headers are attacker-controlled, so an
    # attacker could rotate X-Forwarded-For to get a fresh rate-limit bucket per request
    # and defeat the per-requester cap. Mirrors the admin-loopback middleware's
    # is_loopback_immediate precondition (sprints 737/738) which this helper previously
    # copied the precedence from but dropped the trust gate.
    headers = getattr(request, "headers", None)
    if headers is not None and _host_is_loopback(host):
        xff = headers.get("x-forwarded-for", "") if hasattr(
            headers, "get",
        ) else ""
        if xff and xff.strip():
            hops = [h.strip() for h in xff.split(",") if h.strip()]
            if hops:
                return hops[-1]
        x_real = headers.get("x-real-ip", "") if hasattr(
            headers, "get",
        ) else ""
        if x_real and x_real.strip():
            return x_real.strip()
    return host or "anonymous"


# Bound on the in-flight authenticated-requester carrier so a node taking paid
# traffic with a non-trivial failure rate (jobs that error/disconnect between
# ingress and the settle hook, never popping their entry) can't grow it without
# limit (review sp1058 MEDIUM #2). Oldest entries evict first (insertion-ordered
# dict); an evicted entry just means that long-pending job settles self-pay if it
# ever completes — rare + harmless (no fund loss).
_PAID_REQUESTER_CARRIER_CAP = 4096


async def _resolve_paid_requester_or_402(
    node: Any, body: Dict[str, Any], budget_ftns: float,
) -> Optional[Dict[str, Any]]:
    """Sprint 1056 (requester-payment brick 3) — verify an optional signed
    PaymentAuthorization in the inference body and return the authenticated
    requester info ``{"requester": <addr>, "max_spend_wei": <int>}`` (or None when
    no auth is supplied = self-pay).

    Computes the request-hash binding from the ACTUAL request body fields (the same
    canonical hash the requester signed) and uses ``budget_ftns`` as the quoted
    price ceiling. A rejected authorization raises HTTPException(402) so a paid job
    we can't honor is never served. Fail-CLOSED on the authorization; the on-chain
    settlement itself stays best-effort/fail-open at the settle site.

    NONCE SEMANTIC (review sp1058 HIGH): the verifier consumes the auth's job_nonce
    here, at ingress = single-use-on-acceptance. This deliberately protects the
    PROVIDER from doing duplicate free work for two concurrent same-nonce requests;
    consuming at settle-time would reopen that. A job that fails after acceptance
    does NOT touch the requester's escrow (no on-chain settle without a receipt), so
    the only cost of a retry is the requester re-signing with a fresh nonce (a free
    local op). Clients must therefore sign a new authorization per attempt."""
    payment_authorization = body.get("payment_authorization")
    payment_delegation = body.get("payment_delegation")   # sp1094 — relayer path
    if payment_authorization is None and payment_delegation is None:
        return None
    from decimal import Decimal as _D
    from prsm.settlement.payment_authorization import (
        canonical_request_hash, inference_request_fields,
    )
    from prsm.settlement.client_wiring import (
        resolve_paid_requester, resolve_relayer_requester,
    )
    from prsm.settlement.payment_authorization_verifier import AuthorizationRejected
    request_hash = canonical_request_hash(inference_request_fields(
        model_id=body.get("model_id", ""),
        prompt=body.get("prompt", ""),
        max_tokens=int(body.get("max_tokens") or 0),
        privacy_tier=str(body.get("privacy_tier", "none")),  # sp1234 — must match the inference default (else omitted-tier requests hash-mismatch)
        content_tier=str(body.get("content_tier", "A")),
    ))
    quoted_price_wei = int(_D(str(budget_ftns)) * (_D(10) ** 18))
    try:
        if payment_delegation is not None:
            # sp1094 — a gateway/relayer signed the per-request auth on a funder's
            # behalf; verify the funder's delegation + the relayer auth as one chain and
            # settle from the FUNDER's escrow. The returned address is the funder.
            requester = await resolve_relayer_requester(
                verifier=getattr(node, "_relayer_verifier", None),
                payment_authorization=payment_authorization,
                payment_delegation=payment_delegation,
                request_hash=request_hash,
                quoted_price_wei=quoted_price_wei,
            )
        else:
            requester = await resolve_paid_requester(
                verifier=getattr(node, "_payment_verifier", None),
                payment_authorization=payment_authorization,
                request_hash=request_hash,
                quoted_price_wei=quoted_price_wei,
            )
    except AuthorizationRejected as exc:
        raise HTTPException(
            status_code=402,
            detail=f"payment authorization rejected: {exc.reason}"
                   + (f" ({exc.detail})" if exc.detail else ""),
        )
    if requester is None:
        return None
    # Carry the signed ceiling so the settle site can enforce value <= max_spend
    # against the ACTUAL settled value (defense-in-depth; review sp1058 MEDIUM #3).
    try:
        max_spend_wei = int(payment_authorization["payload"]["max_spend_wei"])
    except (KeyError, TypeError, ValueError):
        max_spend_wei = None
    info = {"requester": requester, "max_spend_wei": max_spend_wei}
    # sp1095 — for a relayer job the verifier reserved the per-request ceiling
    # (max_spend_wei) against the funder's delegation budget; carry the delegation_nonce
    # + reserved amount so the settle site can CAPTURE (release the unused remainder).
    if payment_delegation is not None:
        try:
            info["delegation_nonce"] = payment_delegation["payload"]["delegation_nonce"]
            info["reserved_wei"] = max_spend_wei
        except (KeyError, TypeError):
            pass
    return info


def _record_paid_requester(node: Any, job_id: str, info: Dict[str, Any]) -> None:
    """Record the authenticated requester info for ``job_id`` in the bounded
    in-flight carrier (sprint 1056). Evicts the oldest entry when at capacity so a
    stream of non-settling paid jobs can't leak memory (review sp1058 MEDIUM #2)."""
    table = getattr(node, "_paid_requester_by_job", None)
    if not isinstance(table, dict):
        table = {}
        node._paid_requester_by_job = table
    while len(table) >= _PAID_REQUESTER_CARRIER_CAP:
        try:
            _evicted = table.pop(next(iter(table)))   # evict oldest (insertion order)
            # sp1096 (review A1) — an evicted relayer entry never settled, so release its
            # held delegation-budget reservation (else the funder's cap leaks until a
            # fresh delegation). No-op for self-pay / self-signed entries.
            _release_relayer_reservation(node, _evicted)
        except StopIteration:
            break
    table[job_id] = info


def _release_relayer_reservation(node: Any, info: Optional[Dict[str, Any]]) -> None:
    """Sprint 1096 — release a relayer job's full held delegation-budget reservation
    (used when the job did NOT settle: evicted, or failed before settle). No-op for
    non-relayer entries; fail-open."""
    try:
        if not info or not info.get("delegation_nonce"):
            return
        verifier = getattr(node, "_relayer_verifier", None)
        reserved = int(info.get("reserved_wei") or 0)
        if verifier is not None and reserved > 0:
            verifier.release_reservation_sync(info["delegation_nonce"], reserved)
    except Exception as exc:  # noqa: BLE001
        logger.debug("relayer reservation release skipped (non-fatal): %s", exc)


def _take_paid_requester(node: Any, job_id: str) -> Optional[Dict[str, Any]]:
    """Pop the authenticated requester info recorded at ingress for ``job_id``
    (sprint 1056). Returns None when the job was self-pay or the map is absent — the
    settle path then defaults to provider self-pay, unchanged. The returned dict has
    keys ``requester`` (eth address) and ``max_spend_wei`` (the signed ceiling)."""
    table = getattr(node, "_paid_requester_by_job", None)
    if not isinstance(table, dict):
        return None
    return table.pop(job_id, None)


async def _capture_relayer_budget(
    node: Any, paid_info: Optional[Dict[str, Any]], settled_ftns: Any,
    accumulate_result: Any = None,
) -> None:
    """Sprint 1095 — auth-and-capture for a relayer job: the verifier RESERVED the
    per-request ceiling against the funder's delegation budget at ingress; after the
    actual (<= ceiling) settle, release the unused remainder so a cheaper-than-ceiling
    request doesn't permanently consume the funder's cap. No-op for self-pay /
    self-signed jobs (no delegation_nonce). Fail-open — a capture error must never break
    the settle path.

    sp1096 (review A2): if the on-chain accumulate was SKIPPED (``accumulate_result``
    starts with ``skipped:`` — e.g. multi-stage-deferred, no-client, zero-release), then
    NOTHING settled against the funder's on-chain escrow for this job, so release the
    FULL reservation, not ``reserved - settled`` (the off-chain release didn't touch the
    funder's on-chain delegation budget)."""
    try:
        if not paid_info or not paid_info.get("delegation_nonce"):
            return
        verifier = getattr(node, "_relayer_verifier", None)
        if verifier is None:
            return
        reserved = int(paid_info.get("reserved_wei") or 0)
        if isinstance(accumulate_result, str) and accumulate_result.startswith("skipped:"):
            release_amount = reserved   # nothing booked on-chain → release the whole hold
        else:
            from decimal import Decimal as _D
            settled_wei = int(_D(str(settled_ftns or 0)) * (_D(10) ** 18))
            release_amount = reserved - settled_wei
        if release_amount > 0:
            await verifier.release_reservation(
                paid_info["delegation_nonce"], release_amount)
    except Exception as exc:  # noqa: BLE001
        logger.debug("relayer budget capture skipped (non-fatal): %s", exc)


def register_parallax_streams_endpoint(app: Any, node: Any) -> None:
    """Sprint 722 — /admin/parallax/streams.

    Read-only introspection of in-flight remote token-streams. After
    the sprint-711 wire protocol shipped + the sprint-713 bounded
    receive queue + sprint-719 sender-binding + sprint-720 disconnect
    cleanup, operators have NO direct visibility into what those
    machines are actually doing. This endpoint exposes:

      - active stream count
      - per-stream queue depth + maxsize (sprint 713 back-pressure)
      - expected_sender truncated id (sprint 719 hijack defense)
      - bounded-queue + max-bytes env values currently in effect

    Status codes:
      200 — {streams[], queue_maxsize, request_max_bytes, count}
      503 — node has no _chain_executor_pending_streams attr
            (daemon not fully started or wrong build).
    """
    import os as _os
    from fastapi import HTTPException
    from prsm.node.chain_executor_adapters import (
        _resolve_stream_queue_maxsize,
        _resolve_stream_request_max_bytes,
    )

    @app.get("/admin/parallax/streams", tags=["admin"])
    async def parallax_streams() -> Dict[str, Any]:
        pending = getattr(
            node, "_chain_executor_pending_streams", None,
        )
        if pending is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Node has no streaming-protocol state — "
                    "daemon not fully started or this build "
                    "predates sprint 711."
                ),
            )
        streams = []
        # Snapshot the dict keys so we iterate over a stable view —
        # the response handler may register new streams concurrently.
        for stream_id in list(pending.keys()):
            entry = pending.get(stream_id)
            if entry is None:
                continue
            # Sprint 719 tuple shape: (queue, expected_sender)
            if isinstance(entry, tuple) and len(entry) == 2:
                queue, expected_sender = entry
            else:  # legacy bare-queue
                queue, expected_sender = entry, ""
            try:
                qsize = queue.qsize()
                qmax = queue.maxsize
                qfull = queue.full()
            except Exception:  # noqa: BLE001 — defensive
                qsize, qmax, qfull = -1, -1, False
            streams.append({
                "stream_id_prefix": str(stream_id)[:16],
                "expected_sender_prefix": (
                    str(expected_sender)[:16] if expected_sender else ""
                ),
                "queue_depth": qsize,
                "queue_maxsize": qmax,
                "queue_full": qfull,
            })
        return {
            "count": len(streams),
            "queue_maxsize": _resolve_stream_queue_maxsize(),
            "request_max_bytes": _resolve_stream_request_max_bytes(),
            "streams": streams,
        }


def register_auto_claim_status_endpoint(app: Any, node: Any) -> None:
    """Sprint 769 — /admin/auto-claim/status.

    Surfaces the runtime AutoClaimWorker counters that the
    sprint-767 CLI couldn't see (it only inspects env config).
    Closes the carveout sprint 768's runbook flagged.

    Reported fields:
      - enabled: bool (worker constructed + config.enabled)
      - threshold_ftns / interval_seconds: from config
      - total_claimed_ftns: cumulative Decimal claimed this session
      - claim_attempts: number of times above-threshold check fired
      - claim_failures: number of those that raised

    Status codes:
      200 — worker exists; counters returned
      503 — no worker on this daemon (staking_manager + identity
            weren't both present at start)

    Loopback-gated by the sprint-734 admin middleware
    (`/admin/*` path prefix). Inherits F65-F73 defenses.
    """
    from fastapi import HTTPException

    @app.get("/admin/auto-claim/status", tags=["admin"])
    async def auto_claim_status() -> Dict[str, Any]:
        worker = getattr(node, "_auto_claim_worker", None)
        if worker is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "AutoClaimWorker not constructed on this "
                    "daemon. Requires staking_manager + identity. "
                    "If both should be present, check daemon "
                    "startup logs for a construction warning."
                ),
            )
        cfg = worker.config
        return {
            "enabled": cfg.enabled,
            "threshold_ftns": str(cfg.threshold_ftns),
            "interval_seconds": cfg.interval_seconds,
            "total_claimed_ftns": str(worker.total_claimed_ftns),
            "claim_attempts": worker.claim_attempts,
            "claim_failures": worker.claim_failures,
        }


def register_output_cache_stats_endpoint(
    app: Any, node: Any,
) -> None:
    """Sprint 815 — GET /admin/output-cache-stats.

    Returns the sprint-814 stats() snapshot from the executor's
    output cache. Lets operators audit hit-rate + eviction
    behavior without a full Prometheus stack.

    Status codes:
      200 — cache wired; stats returned
      503 — executor missing or no _output_cache wired
            (PRSM_INFERENCE_OUTPUT_CACHE_ENABLED unset or
             daemon-startup factory hit a fail-soft branch)

    Loopback-gated by sprint-734 admin middleware
    (`/admin/*` path prefix). Inherits F65-F73 defenses.
    """
    from fastapi import HTTPException

    @app.get(
        "/admin/output-cache-stats", tags=["admin"],
    )
    async def output_cache_stats() -> Dict[str, Any]:
        executor = getattr(node, "inference_executor", None)
        if executor is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Inference executor not initialized. "
                    "Output cache stats unavailable."
                ),
            )
        cache = getattr(executor, "_output_cache", None)
        if cache is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Output cache not configured. Set "
                    "PRSM_INFERENCE_OUTPUT_CACHE_ENABLED=1 + "
                    "restart daemon to enable."
                ),
            )
        return cache.stats()


def register_partial_completion_events_endpoint(
    app: Any, node: Any,
) -> None:
    """Sprint 799 — GET /admin/partial-completion-events.

    Surfaces sprint-798's PartialCompletionEventRing for
    operator audit. Pagination + operator_node_id filter mirror
    /admin/slash-history's shape (sprint 262).

    Status codes:
      200 — entries returned
      422 — limit out of [1, 1000] OR offset < 0
      503 — ring not initialized (daemon hasn't constructed
            it; expected during startup race or in test fixtures
            that don't set node._partial_completion_event_log)
    """
    from fastapi import HTTPException

    @app.get(
        "/admin/partial-completion-events",
        tags=["admin"],
    )
    async def get_partial_completion_events(
        limit: int = 50,
        offset: int = 0,
        operator_node_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if limit <= 0 or limit > 1000:
            raise HTTPException(
                status_code=422,
                detail=f"limit must be in [1, 1000], got {limit}",
            )
        if offset < 0:
            raise HTTPException(
                status_code=422,
                detail=f"offset must be >= 0, got {offset}",
            )
        ring = getattr(
            node, "_partial_completion_event_log", None,
        )
        if ring is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Partial-completion event log not "
                    "initialized on this daemon. Set "
                    "node._partial_completion_event_log at "
                    "startup (sprint 799 wire-up)."
                ),
            )
        entries = ring.recent(
            limit=limit, offset=offset,
            operator_node_id=operator_node_id,
        )
        return {
            "entries": entries,
            "total": ring.count(),
            "offset": offset,
            "limit": limit,
        }


def register_preemption_status_endpoint(app: Any, node: Any) -> None:
    """Sprint 776 — /admin/preemption/status.

    Surfaces sprint-775's PreemptionDetector state for operator
    observability:
      - enabled: detector wired
      - backend: aws | gcp (which metadata endpoint we're polling)
      - preempted: flag state (True = cloud signaled termination)
      - poll_interval_seconds: cadence configured

    Status codes:
      200 — detector wired; payload returned
      503 — no detector on this daemon (PRSM_PREEMPTION_DETECTOR
            unset OR construction failed at start)

    Loopback-gated by the sprint-734 admin middleware
    (`/admin/*` path prefix). Inherits F65-F73 defenses.
    """
    from fastapi import HTTPException

    @app.get("/admin/preemption/status", tags=["admin"])
    async def preemption_status() -> Dict[str, Any]:
        det = getattr(node, "_preemption_detector", None)
        if det is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "PreemptionDetector not wired on this daemon. "
                    "Set PRSM_PREEMPTION_DETECTOR=aws|gcp in the "
                    "systemd unit (or env) + restart. Safe default "
                    "for non-cloud nodes is unset."
                ),
            )
        backend_cls = det.backend.__class__.__name__
        # Map class name → short backend tag ("AWSPreemptionBackend" → "aws")
        if "AWS" in backend_cls:
            backend = "aws"
        elif "GCP" in backend_cls:
            backend = "gcp"
        else:
            backend = backend_cls.lower()
        return {
            "enabled": True,
            "backend": backend,
            "preempted": bool(det.is_preempted()),
            "poll_interval_seconds": float(det.poll_interval_s),
        }


def create_api_app(node: Any, enable_security: bool = True) -> FastAPI:
    """
    Create the node management FastAPI app with a reference to the running node.
    
    Args:
        node: The PRSM node instance
        enable_security: Whether to enable security hardening (default: True)
    
    Returns:
        FastAPI application with security hardening applied
    """

    # Read canonical version from package metadata so OpenAPI
    # spec stays in sync with pyproject.toml across releases.
    # Fallback to "unknown" when running without editable install.
    try:
        from importlib.metadata import version as _pkg_version
        _api_version = _pkg_version("prsm-network")
    except Exception:  # noqa: BLE001
        _api_version = "unknown"

    # Sprint 189 — declare default `servers` so openapi-generator
    # and similar tooling can prefill the API base URL instead of
    # leaving it null. Operators on non-default deploys can
    # override by setting PRSM_API_BASE_URL.
    _default_server = os.environ.get(
        "PRSM_API_BASE_URL", "http://127.0.0.1:8000",
    ).strip() or "http://127.0.0.1:8000"

    # Sprint 744 F72 — /openapi.json, /docs, /redoc are hidden by
    # default. Pre-744, any HTTP client could fetch /openapi.json
    # and get the complete API surface map — every endpoint URL,
    # every path parameter, every body schema — including the
    # /admin/* paths sprints 734-743 worked to defend. An attacker
    # gets a roadmap of what to probe + the exact body shape to
    # send. Hide by default; operators who genuinely need the
    # interactive docs in dev set `PRSM_API_DOCS_ENABLED=1`.
    _docs_raw = os.environ.get(
        "PRSM_API_DOCS_ENABLED", "",
    ).strip().lower()
    _docs_enabled = _docs_raw in ("1", "true", "yes")
    app = FastAPI(
        title="PRSM Node API",
        description="Management API for a PRSM network node",
        version=_api_version,
        docs_url="/docs" if _docs_enabled else None,
        redoc_url="/redoc" if _docs_enabled else None,
        openapi_url="/openapi.json" if _docs_enabled else None,
        servers=[{"url": _default_server, "description": "PRSM node"}],
    )

    # Sprint 547 — wire the interactive operator onboarding wizard
    # (prsm/interface/api/onboarding_router.py). User-perspective
    # dogfood arc sprint 424 surfaced F6: node-start log advertised
    # the /onboarding/ URL but the path returned 404 because the
    # router existed in the tree but was never included in the
    # FastAPI app. Include happens fail-soft — a missing template
    # dir or pydantic mismatch on import shouldn't break the rest
    # of the API surface.
    try:
        from prsm.interface.api.onboarding_router import (
            router as _onboarding_router,
        )
        app.include_router(_onboarding_router)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Sprint 547: onboarding router include failed (%s); "
            "/onboarding/* will return 404. Non-fatal — the rest "
            "of the API surface still works.", exc,
        )

    # Sprint 829 — F28 fix: wallet_api router was never mounted in
    # the daemon's create_api_app path. Sprint 793 wired the
    # SERVICES (set_services for shared state) but the multi-
    # device arc's HTTP routes (/api/v1/auth/wallet/bind,
    # /devices, /devices/earnings — sprints 786-794) lived in the
    # router and only the unused app_factory.py path mounted it.
    # Surfaced by live dogfood 2026-05-24: `prsm wallet devices
    # add --register` → 404 from the daemon. Same fail-soft
    # pattern as onboarding_router above.
    try:
        from prsm.interface.api.wallet_api import (
            router as _wallet_router,
        )
        app.include_router(_wallet_router)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Sprint 829: wallet router include failed (%s); "
            "/api/v1/auth/wallet/* will return 404. Non-fatal — "
            "operators using PRIVATE_KEY-only flows still work; "
            "only the multi-device registration path is impacted.",
            exc,
        )

    # Audit log middleware (ships 2026-05-09). Records every
    # non-GET request to the in-process ring buffer for operator
    # review via /audit/recent. GET excluded so the buffer stays
    # focused on writes.
    @app.middleware("http")
    async def audit_log_middleware(request, call_next):
        response = await call_next(request)
        try:
            method = request.method
            if method != "GET" and method != "HEAD":
                ring = getattr(node, "_audit_log", None)
                if ring is not None:
                    requester = (
                        node.identity.node_id
                        if node.identity else None
                    )
                    request_id = response.headers.get(
                        "X-Request-ID", "-",
                    )
                    ring.append(
                        method=method,
                        path=request.url.path,
                        requester=requester,
                        status_code=response.status_code,
                        request_id=request_id,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "audit log middleware failed: %s", exc,
            )
        return response

    # X-Request-ID correlation middleware (ships 2026-05-09).
    # Every response carries an X-Request-ID header for log
    # correlation across distributed systems. Client-supplied
    # IDs (e.g. from upstream LBs) are echoed back; missing /
    # empty IDs trigger server-side UUID generation. Cap at
    # 128 chars defends against log-poisoning via gigantic IDs.
    #
    # Also sets the request_id contextvar so log records produced
    # during the request's processing get tagged with the current
    # request_id (operators wiring %(request_id)s in their log
    # formatter see end-to-end correlation in log files).
    from prsm.node.request_id_logging import (
        set_request_id, clear_request_id,
    )

    @app.middleware("http")
    async def request_id_middleware(request, call_next):
        supplied = request.headers.get("x-request-id", "").strip()
        if supplied:
            request_id = supplied[:128]
        else:
            request_id = str(_uuid.uuid4())
        token = set_request_id(request_id)
        try:
            response = await call_next(request)
        finally:
            clear_request_id(token)
        response.headers["X-Request-ID"] = request_id
        return response

    # Sprint 188 — security response headers. Applied uniformly to
    # every response. Documented OWASP recommendations for HTTP-API
    # services that may be embedded in browsers (dashboard) or
    # consumed by third-party clients.
    #
    #   X-Content-Type-Options: nosniff
    #     Prevents browsers from MIME-sniffing past the declared
    #     Content-Type (defends against polyglot attacks where a
    #     response declared application/json is sniffed as
    #     text/html and executes inline script).
    #
    #   X-Frame-Options: DENY
    #     Prevents the API responses from being embedded in iframes
    #     (defends against clickjacking on the dashboard surface).
    #
    #   Referrer-Policy: strict-origin-when-cross-origin
    #     Limits Referer header leakage to other origins —
    #     defends against operator-IP / endpoint enumeration via
    #     outbound clickthroughs from the dashboard.
    #
    # Strict-Transport-Security is intentionally NOT set here —
    # PRSM runs on http://127.0.0.1 by default; HSTS belongs on
    # Sprint 201 — reject JSON bodies containing the non-standard
    # `Infinity` / `-Infinity` / `NaN` literals. Python's stdlib
    # `json.loads` accepts these by default; Pydantic v2 `gt=0`
    # accepts Infinity (`inf > 0` is True), and even
    # `allow_inf_nan=False` causes FastAPI's error renderer to
    # crash on the validation-error `input` field (cannot serialize
    # inf back to JSON). Intercept upstream of body parsing.
    #
    # Strategy: only apply to POST/PUT/PATCH with JSON content-type.
    # Use a regex on the raw bytes that matches the JSON token form
    # (unquoted whole-word `Infinity` / `NaN`, not embedded in a
    # string literal). Body is cached on `request._body` so handlers
    # downstream read the same bytes from Starlette's cache.
    import re as _re_inf
    _INF_NAN_RE = _re_inf.compile(
        rb'(?<![\w"])(-?Infinity|NaN)(?![\w"])',
    )

    @app.middleware("http")
    async def inf_nan_body_guard(request, call_next):
        if request.method in ("POST", "PUT", "PATCH"):
            ctype = request.headers.get("content-type", "")
            if "json" in ctype.lower():
                body = await request.body()
                if body and _INF_NAN_RE.search(body):
                    from starlette.responses import JSONResponse
                    return JSONResponse(
                        {"detail": (
                            "Request body contains NaN or Infinity "
                            "literal; only finite numbers are "
                            "accepted."
                        )},
                        status_code=422,
                    )
        return await call_next(request)

    # the operator's reverse proxy (Caddy/nginx/Cloudflare) where
    # the TLS termination actually happens.
    @app.middleware("http")
    async def security_headers_middleware(request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault(
            "Referrer-Policy", "strict-origin-when-cross-origin",
        )
        return response

    # Sprint 742 F70 — HTTP body-size limit (memory-DoS defense).
    # Pre-742, FastAPI accepted arbitrary JSON bodies — a malicious
    # POST with a 1GB body would allocate all of it before any
    # size check ran. HTTP-side analog of sprint-721/725's wire-
    # protocol size limits. Default 1 MiB (covers reasonable
    # prompts + metadata for /compute/inference + /compute/forge);
    # operators tune via PRSM_HTTP_MAX_BODY_BYTES. Setting <= 0
    # disables the gate (pre-742 behavior). Non-int safely
    # defaults to 1 MiB.
    from starlette.responses import JSONResponse as _BodyLimitJSON
    # sp947 — routes that enforce their OWN, larger body cap in-handler. The
    # generic 1 MiB guard below would SHADOW these (rejecting an over-1-MiB
    # upload at the middleware with the wrong PRSM_HTTP_MAX_BODY_BYTES message
    # before the handler's canonical PRSM_MAX_UPLOAD_BYTES / *_SHARD_* 413 — or
    # a legitimate 200 — can fire). For these paths the ceiling is raised to a
    # coarse multiple of the route's own configured cap: enough that an over-cap
    # request still REACHES the handler for its canonical error, while a gross
    # multi-GB body is still bounded here. Every other route keeps the 1 MiB
    # DoS guard (sprint 742 F70) unchanged.
    _UPLOAD_ROUTE_CAP_ENV = {
        "/content/upload": ("PRSM_MAX_UPLOAD_BYTES", 10 * 1024 * 1024),
        "/content/upload/shard": ("PRSM_MAX_SHARD_UPLOAD_BYTES", 100 * 1024 * 1024),
    }

    def _positive_int_env(_os, name, default):
        raw = _os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            v = int(raw)
        except (TypeError, ValueError):
            return default
        return v if v > 0 else default

    @app.middleware("http")
    async def http_body_size_limit_middleware(request, call_next):
        import os as _os
        raw = _os.environ.get("PRSM_HTTP_MAX_BODY_BYTES", "")
        if raw.strip():
            try:
                _max_bytes = int(raw)
            except ValueError:
                _max_bytes = 1024 * 1024  # 1 MiB safe default
        else:
            _max_bytes = 1024 * 1024
        if _max_bytes > 0:
            # sp947 — honor an upload route's own (larger) cap so the generic
            # guard never pre-empts the handler's canonical 413. Keeps the
            # larger of (operator's global cap, route budget); leaves the
            # global cap authoritative for all non-upload routes. The "0 =
            # disabled" semantics are preserved (this block is gated on >0).
            _route_cap_env = _UPLOAD_ROUTE_CAP_ENV.get(request.url.path)
            if _route_cap_env is not None:
                _route_budget = _positive_int_env(_os, *_route_cap_env) * 2 + 1024 * 1024
                if _route_budget > _max_bytes:
                    _max_bytes = _route_budget
            cl_header = request.headers.get("content-length", "")
            if cl_header:
                try:
                    cl = int(cl_header)
                except ValueError:
                    cl = -1
                if cl > _max_bytes:
                    return _BodyLimitJSON(
                        status_code=413,
                        content={
                            "detail": (
                                f"Request body exceeds max bytes "
                                f"({cl} > {_max_bytes}); tune via "
                                f"PRSM_HTTP_MAX_BODY_BYTES."
                            ),
                            "content_length": cl,
                            "limit": _max_bytes,
                        },
                    )
        return await call_next(request)

    # Sprint 734 F65 — restrict `/admin/*` endpoints to loopback by
    # default. Pre-734, any HTTP client on the same network as the
    # daemon could read endpoints like /admin/parallax/streams
    # (leaking expected_sender peer IDs — the very data sprint
    # 719/727's sender-binding protects against forging),
    # /admin/fiat-compliance (KYC/financial audit records),
    # /admin/content-filter (moderation state), etc. None of these
    # had auth checks; the `tags=["admin"]` was a swagger grouping
    # tag with no access-control semantics.
    #
    # Default: only allow connections from 127.0.0.1, ::1, and
    # localhost. Operators who need remote admin access set
    # PRSM_ADMIN_REMOTE_ALLOWED=1 (and accept the risk — they
    # should be behind reverse-proxy auth or a VPN). Empty/non-set
    # = default safe-deny.
    from starlette.responses import JSONResponse as _AdminJSON
    @app.middleware("http")
    async def admin_loopback_middleware(request, call_next):
        path = request.url.path
        # Sprint 745 F73 — /metrics also gated. Prometheus
        # exposition leaks internal financial state (pending
        # escrow totals, locked FTNS), counter values, peer
        # connection counts, and subsystem internals. Same
        # reconnaissance + financial-intel concern as the F65-F72
        # arc. Operators with remote Prometheus scrapers behind
        # reverse-proxy auth set PRSM_ADMIN_REMOTE_ALLOWED=1 (the
        # same env they already need for any remote admin tooling)
        # — sprint-740 runbook documents the 3 remediation paths.
        #
        # Sprint 747 F74 — /health/detailed and /info also gated.
        # Both leak reconnaissance-grade data:
        # - /info: operator_address (on-chain EOA), node_id, wired
        #   contract addresses, agent_forge wired status, software
        #   version (CVE-matching surface)
        # - /health/detailed: per-subsystem status including
        #   ftns_ledger.connected_address + wired token address +
        #   canonical-match boolean for 14+ subsystems
        # Minimal `/health` (load-balancer probe) is INTENTIONALLY
        # NOT gated — load balancers need it to stay reachable.
        #
        # Sprint 748 F75 — /status and /rings/status also gated.
        # /status is the heaviest leak yet: it returns the
        # operator's literal `ftns_balance` (financial value) plus
        # full subsystem stats, peer counts, bootstrap telemetry,
        # node_id, p2p_address, api_address, escrow stats,
        # consensus stats, batch_settlement stats, ftns_onchain
        # summary, and per-subsystem provider stats. A network
        # attacker can:
        # - Read operator's exact FTNS balance (financial intel)
        # - Map their peer topology (network attack surface)
        # - Learn batch-settlement schedules (timing attack)
        # /rings/status is the same recon-class as /health/detailed.
        #
        # Sprint 749 F76 — /peers also gated. It returns the
        # COMPLETE network topology: every peer_id this daemon
        # knows about + every IP:port address + connected_at +
        # last_seen timestamps. An attacker reading /peers gets
        # the entire P2P attack-surface map (which nodes to
        # target, when they connected, when they were last
        # active). Pure reconnaissance vector.
        #
        # Sprint 750 F77 — /balance + /bootstrap/status gated.
        # /balance is the WORST leak found yet: it returns the
        # operator's literal `balance` (FTNS value) PLUS the last
        # 20 transactions including counterparty wallet_ids +
        # amounts + descriptions + timestamps. Lets an attacker
        # build a complete financial profile (who the operator
        # pays, how much, when). /bootstrap/status leaks which
        # bootstrap servers this daemon is using — useful for an
        # attacker planning to attack the bootstrap fleet.
        #
        # Sprint 751 F78 — more financial recon endpoints:
        # - /transactions: returns up to 200 transactions (worse
        #   than /balance's 20). Same financial-profile concern.
        # - /staking/status: returns the node's active stakes,
        #   pending unstake requests, reward totals. Reveals the
        #   operator's staking position — useful for adversarial
        #   timing (when stake unlocks).
        # - /settlement/stats + /pending + /history: settlement
        #   counts/amounts/schedules. Same financial-intel class.
        #
        # Sprint 752 F79 — more recon-class endpoints:
        # - /balance/onchain (no address arg): operator's on-chain
        #   FTNS balance. Same financial-value concern as F77.
        # - /audit/summary + /audit/recent: HTTP access log
        #   aggregates + recent entries. Leaks endpoint-usage
        #   patterns, error rates, load intel — DoS reconnaissance.
        # - /ledger/sync/stats: ledger sync state with peer
        #   counts + last_sync timestamps. Network intel.
        #
        # Sprint 753 F80 — financial + privacy recon endpoints:
        # - /agents/spending: per-agent FTNS spend totals. Financial.
        # - /privacy/budget: differential-privacy budget audit
        #   report. Reveals usage patterns + remaining privacy
        #   budget (attacker can time queries to exhaust budget,
        #   or learn which queries the operator has run by inferring
        #   from epsilon spend trajectory).
        #
        # NOTE: /agents (bare list of services this node offers) is
        # INTENTIONALLY NOT gated — PRSM's marketplace model treats
        # agent listings as service-discovery surface. Operators
        # publishing services WANT them discoverable. Same posture
        # as /health (minimal) — public by design.
        _GATED_PATHS = (
            "/metrics", "/info", "/health/detailed",
            "/status", "/rings/status", "/peers",
            "/balance", "/bootstrap/status",
            "/transactions", "/staking/status",
            "/settlement/stats", "/settlement/pending",
            "/settlement/history",
            "/balance/onchain", "/audit/summary",
            "/audit/recent", "/ledger/sync/stats",
            "/agents/spending", "/privacy/budget",
        )
        if not (
            path.startswith("/admin/") or path in _GATED_PATHS
        ):
            return await call_next(request)
        import os as _os
        if _os.environ.get(
            "PRSM_ADMIN_REMOTE_ALLOWED", ""
        ).strip().lower() in ("1", "true", "yes"):
            return await call_next(request)
        client = getattr(request, "client", None)
        client_host = getattr(client, "host", "") if client else ""
        # IPv4 + IPv6 loopback. fastapi/starlette normalises ::1
        # and 127.0.0.1; "testclient" is starlette TestClient's
        # synthetic host (let through so test fixtures don't
        # need PRSM_ADMIN_REMOTE_ALLOWED).
        # Sprint 743 F71 — DNS-rebinding defense. A malicious web
        # page in the victim's browser can make requests to
        # `http://localhost:8000/admin/*`. The browser connects
        # from the victim's machine, so the daemon sees
        # `client.host=127.0.0.1` and the F65-F68 loopback gate
        # passes. Admin data (sprint-722 expected_sender peer IDs,
        # KYC records, moderation state) would leak to attacker JS.
        #
        # Defense: browsers always set the `Origin` header on
        # cross-origin requests (HTML form POSTs + fetch + XHR). CLI
        # tools (curl, prsm node CLI, python httpx) typically don't.
        # Reject /admin/* requests that carry an Origin header,
        # regardless of the loopback check result.
        #
        # An operator with a web dashboard genuinely needing to hit
        # admin endpoints can set PRSM_ADMIN_REMOTE_ALLOWED=1 (which
        # already bypasses the entire check chain — including this
        # one) AND add real auth at the proxy layer.
        origin_header = request.headers.get("origin", "")
        if origin_header.strip():
            return _AdminJSON(
                status_code=403,
                content={
                    "detail": (
                        "admin endpoints reject browser-origin "
                        "requests by default (sprint 743 F71 DNS-"
                        "rebinding defense). Origin header indicates "
                        "the request came from a browser, not a CLI. "
                        "Set PRSM_ADMIN_REMOTE_ALLOWED=1 + add real "
                        "auth at the proxy layer if you have a "
                        "legitimate web dashboard hitting /admin/*."
                    ),
                    "origin": origin_header,
                },
            )

        _LOOPBACK = ("127.0.0.1", "::1", "localhost", "testclient")

        def _is_loopback(host: str) -> bool:
            """Sprint 739 F68 — accept IPv4-mapped IPv6 loopback
            (`::ffff:127.0.0.1`) too. Dual-stack daemons on Linux
            see IPv4 loopback connections in this form by default;
            sprint-734's literal whitelist would reject them and
            block all admin CLI calls from operators running
            dual-stack.

            Also accepts the entire 127.0.0.0/8 IPv4 loopback block
            (per RFC 1122, all of 127/8 is loopback — though
            127.0.0.1 is the canonical address)."""
            if host in _LOOPBACK:
                return True
            # IPv4-mapped IPv6 loopback: ::ffff:127.0.0.1
            if host.lower().startswith("::ffff:127."):
                return True
            # Entire 127/8 IPv4 loopback block
            if host.startswith("127.") and host.count(".") == 3:
                # Validate the remaining 3 octets are numeric so we
                # don't accept "127.foo.bar.baz". Cheap split check.
                parts = host.split(".")
                if all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
                    return True
            return False

        is_loopback_immediate = _is_loopback(client_host)
        # Sprint 737 F66 — reverse-proxy bypass defense. If a
        # reverse proxy (nginx, HAProxy) on the same host
        # terminates external connections + forwards to
        # 127.0.0.1:8000, the daemon sees `client.host=127.0.0.1`
        # for what is in fact arbitrary external traffic. F65's
        # loopback gate would have passed those through.
        #
        # Defense: if the immediate client IS loopback AND an
        # X-Forwarded-For header is present, the request came
        # from a proxy. The last hop in X-Forwarded-For is the
        # real upstream client IP. If THAT is non-loopback, the
        # admin gate must deny.
        #
        # X-Forwarded-For spoofing requires LOCAL-PROCESS access
        # (the immediate connection has to be loopback for the
        # header to be examined at all), so attacker access is
        # already significantly bounded. A malicious local process
        # is a different threat model than a peered network
        # client.
        # Sprint 738 F67 — also inspect X-Real-IP (common in nginx
        # configurations that don't append to XFF). Proxies vary:
        # some set XFF, some set X-Real-IP, some set both. We
        # reject if EITHER indicates external upstream client.
        xff = request.headers.get("x-forwarded-for", "")
        x_real_ip = request.headers.get("x-real-ip", "")
        real_client = ""
        header_used = ""
        if is_loopback_immediate and xff.strip():
            # Take the LAST hop (rightmost) — that's the address
            # the immediate proxy saw as its client. Earlier hops
            # in the chain are less trusted (they may have been
            # set by the original requester).
            hops = [h.strip() for h in xff.split(",") if h.strip()]
            real_client = hops[-1] if hops else ""
            header_used = "x-forwarded-for"
        if (
            is_loopback_immediate
            and not real_client
            and x_real_ip.strip()
        ):
            real_client = x_real_ip.strip()
            header_used = "x-real-ip"
        if (
            is_loopback_immediate
            and real_client
            and not _is_loopback(real_client)
        ):
            return _AdminJSON(
                status_code=403,
                content={
                    "detail": (
                        "admin endpoints reject reverse-proxied "
                        "remote traffic by default (sprints 737-"
                        "738 F66/F67). Proxy header indicates a "
                        "non-loopback upstream client. Set "
                        "PRSM_ADMIN_REMOTE_ALLOWED=1 to allow "
                        "AND add real auth at your proxy layer."
                    ),
                    "client_host": client_host,
                    "proxy_header": header_used,
                    "upstream_client": real_client,
                },
            )
        if is_loopback_immediate:
            return await call_next(request)
        return _AdminJSON(
            status_code=403,
            content={
                "detail": (
                    "admin endpoints restricted to loopback by "
                    "default (sprint 734 F65). Set "
                    "PRSM_ADMIN_REMOTE_ALLOWED=1 to allow remote "
                    "access — only safe behind reverse-proxy auth "
                    "or VPN."
                ),
                "client_host": client_host,
            },
        )

    # Sprint 954 — server-side Origin check for the wallet onboarding endpoints
    # (/api/v1/auth/wallet/*), the last auth-review residual. DNS-rebinding
    # defense COMPLEMENTING CORS: CORS stops a cross-origin page from READING
    # wallet responses, but a rebinding attack makes the victim's browser send a
    # state-changing request to the loopback-bound node that the browser treats
    # as same-origin to the attacker's page — the Origin header still carries the
    # attacker's domain. Reject a wallet request whose Origin is present AND not
    # in the operator's PRSM_ALLOWED_ORIGINS allowlist. No Origin header (CLI /
    # MCP / server-to-server) is always allowed; an unset/empty allowlist stays
    # permissive (dev/unconfigured), matching the CORS default + the 127.0.0.1
    # default bind. Unlike /admin/* (which rejects ALL browser-origin requests),
    # the wallet endpoints are legitimately browser-facing, so allowlisted
    # origins pass. CORS (registered later → outermost) handles preflight
    # OPTIONS before this guard sees them.
    @app.middleware("http")
    async def wallet_origin_guard_middleware(request, call_next):
        if request.url.path.startswith("/api/v1/auth/wallet"):
            origin = request.headers.get("origin", "").strip()
            if origin:
                import os as _os_wog
                _raw = _os_wog.environ.get("PRSM_ALLOWED_ORIGINS", "").strip()
                if _raw:
                    _allowed = {o.strip() for o in _raw.split(",") if o.strip()}
                    if _allowed and origin not in _allowed:
                        from starlette.responses import JSONResponse as _WalletOriginJSON
                        return _WalletOriginJSON(
                            status_code=403,
                            content={
                                "detail": (
                                    "wallet endpoint rejected a cross-origin "
                                    "browser request (sprint 954 DNS-rebinding "
                                    "defense): Origin is not in "
                                    "PRSM_ALLOWED_ORIGINS. Add the wallet UI's "
                                    "origin to PRSM_ALLOWED_ORIGINS if it is "
                                    "legitimate."
                                ),
                                "origin": origin,
                            },
                        )
        return await call_next(request)

    # Sprint 187 — HEAD-rewriting middleware. The dashboard sub-app
    # mount at `""` (see ~line 6753) catches HEAD requests before
    # FastAPI's auto-HEAD-for-GET path runs, returning 404 for any
    # HEAD probe — broke Kubernetes liveness, AWS ELB, generic
    # monitoring (RFC 7231 designates HEAD as the canonical
    # probe verb). Sprint 186 explicitly registered HEAD on /health
    # but the gap remained for every other GET route.
    #
    # Strategy: intercept HEAD upstream of routing, rewrite to GET,
    # let the parent's GET route fire, then strip body per RFC 7231.
    # Failure-soft — exceptions inside dispatch fall back to the
    # original HEAD path (which still 404s from the mount, matching
    # pre-fix behavior).
    from starlette.responses import Response as _StarletteResponse
    @app.middleware("http")
    async def head_as_get_middleware(request, call_next):
        if request.method != "HEAD":
            return await call_next(request)
        # Rewrite the request scope to GET so the dispatcher picks
        # the GET route. Starlette exposes `scope` as a mutable dict.
        try:
            request.scope["method"] = "GET"
            response = await call_next(request)
        except Exception:  # noqa: BLE001
            request.scope["method"] = "HEAD"
            return await call_next(request)
        # Strip body for HEAD (RFC 7231 §4.3.2). Preserve all
        # headers including Content-Type + Content-Length so
        # probe-side tooling sees the same metadata as GET.
        return _StarletteResponse(
            content=b"",
            status_code=response.status_code,
            headers=dict(response.headers),
        )

    # CORS allowlist (PRSM_ALLOWED_ORIGINS env var, ships 2026-05-09).
    # Production-hardening for nodes serving browser-based clients.
    # Operator declares the explicit list of origins permitted to make
    # cross-origin requests; everything else gets blocked at the CORS
    # layer before reaching any endpoint.
    #
    # sp1123 (Domain-08 review LOW — CORS): with an explicit PRSM_ALLOWED_ORIGINS the node
    # restricts to that allowlist. With NONE set, the default is environment-aware: dev
    # keeps the convenient permissive `*`, but PRODUCTION (PRSM_ENV=production) DEFAULT-
    # DENIES cross-origin (empty allowlist) and warns — an operator must opt in by setting
    # PRSM_ALLOWED_ORIGINS rather than silently shipping `*`. Mirrors the schemas.py CORS
    # validator + the jwt/master-key production gates (same PRSM_ENV signal).
    import logging as _logging
    from fastapi.middleware.cors import CORSMiddleware
    _origins_raw = os.getenv("PRSM_ALLOWED_ORIGINS", "").strip()
    _origins = [o.strip() for o in _origins_raw.split(",") if o.strip()] if _origins_raw else []
    if _origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    elif os.getenv("PRSM_ENV", "development").lower() == "production":
        _logging.getLogger("prsm.node.api").warning(
            "PRSM_ALLOWED_ORIGINS is not set in production — CORS defaults to DENY all "
            "cross-origin requests. Set PRSM_ALLOWED_ORIGINS to permit browser clients."
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[],  # deny-all cross-origin (production fail-closed default)
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        # Dev / local / unconfigured → permissive default (unchanged).
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Initialize security hardening
    api_hardening: Optional[APIHardening] = None
    status_websocket: Optional[StatusWebSocket] = None
    
    if enable_security:
        # Create security configuration
        security_config = APISecurityConfig(
            enable_rate_limiting=True,
            enable_jwt_auth=False,
            enable_websocket=True,
            enable_openapi=True,
            rate_limit_requests_per_minute=100,
            rate_limit_requests_per_hour=1000,
        )
        
        # Initialize hardening
        api_hardening = APIHardening(app, security_config)
        
        # Apply middleware (must be done before adding routes)
        # Note: Middleware is applied in reverse order, so we add it after route definition
        # We'll apply it at the end of this function
        
        # Setup OpenAPI schema
        api_hardening.setup_openapi()
        
        # Get WebSocket manager for status updates
        status_websocket = api_hardening.get_status_websocket()

    # Store WebSocket manager in app state for access
    app.state.status_websocket = status_websocket
    app.state.api_hardening = api_hardening

    # ── Public Endpoints (no auth required) ─────────────────────────────────────

    @app.get("/api-info", tags=["status"])
    async def api_info() -> Dict[str, Any]:
        """API information endpoint (dashboard served at root).

        Version is read from installed package metadata
        (importlib.metadata) so it stays in sync with
        pyproject.toml across releases. Falls back to "unknown"
        if the package isn't installed (running from source
        without editable install).
        """
        try:
            from importlib.metadata import version as _pkg_version
            pkg_version = _pkg_version("prsm-network")
        except Exception:  # noqa: BLE001
            pkg_version = "unknown"
        return {
            "name": "PRSM Node API",
            "version": pkg_version,
            "docs": "/docs",
            "openapi": "/openapi.json",
            "websocket": "/ws/status",
        }

    @app.get("/status", tags=["status"])
    async def get_status() -> Dict[str, Any]:
        """Get comprehensive node status."""
        status = await node.get_status()
        
        # Broadcast status update via WebSocket if available
        if app.state.status_websocket:
            await app.state.status_websocket.broadcast_status(status)
        
        return status

    @app.get("/rings/status", tags=["status"])
    async def rings_status() -> Dict[str, Any]:
        """Get Ring 1-10 initialization and health status."""
        from prsm.observability.dashboard_metrics import DashboardMetrics
        metrics = DashboardMetrics(node=node)
        return metrics.get_summary()

    class _PeersConnectRequest(BaseModel):
        address: str

    @app.post("/peers/connect", tags=["network"])
    async def post_peers_connect(
        body: _PeersConnectRequest,
    ) -> Dict[str, Any]:
        """Sprint 569 — operator-facing trigger for
        ``transport.connect_to_peer(address)``. Closes sprint-567 gap 1
        (auto-dial): bootstrap-mediated discovery populates ``known[]``
        but never auto-connects; this endpoint lets operators turn a
        known peer into a connected one.

        Accepted address forms (per WebSocketTransport.connect_to_peer):
          - ``host:port`` (defaults to ws://; wss:// when port=443)
          - ``ws://host:port``
          - ``wss://host:port``

        Status:
          200 — {connected, peer_id, address}
          400 — empty address
          422 — body missing `address` (Pydantic)
          502 — transport.connect_to_peer returned None (remote
                unreachable, handshake rejected, etc.)
          503 — transport not initialized
          500 — transport raised an unexpected exception
        """
        transport = getattr(node, "transport", None)
        if transport is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Transport not initialized — daemon is still "
                    "starting up or transport failed to load."
                ),
            )
        addr = (body.address or "").strip()
        if not addr:
            raise HTTPException(
                status_code=400,
                detail=(
                    "address must be non-empty "
                    "(`host:port` / `ws://host:port` / `wss://host:port`)"
                ),
            )
        try:
            peer = await transport.connect_to_peer(addr)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=(
                    f"transport.connect_to_peer raised: "
                    f"{type(exc).__name__}: {exc!s}"[:300]
                ),
            )
        if peer is None:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"transport.connect_to_peer returned None for "
                    f"{addr!r} — remote unreachable, handshake "
                    f"rejected, or DO firewall payload-blocking. "
                    f"Check the daemon log for the underlying error."
                ),
            )
        return {
            "connected": True,
            "peer_id": getattr(peer, "peer_id", None),
            "address": getattr(peer, "address", addr),
        }

    @app.get("/peers")
    async def get_peers() -> Dict[str, Any]:
        """List connected and known peers.

        Pre-fix the `connected` list only contained OUTBOUND
        peers (those this node initiated via connect_to_peer).
        Inbound peers (connections initiated by remote nodes)
        live in the C/libp2p layer and weren't tracked in
        `transport._peers`. Result: /peers and /status reported
        different `connected_count` for the same node.

        Post-fix: the count comes from `transport.peer_count`
        (kernel-truth). The list still emits rich metadata for
        outbound peers (display_name, connected_at) but also
        synthesizes minimal entries for inbound peers, so
        operators can see ALL connected peers even if some
        fields are missing.
        """
        connected = []
        seen_addresses = set()
        if node.transport:
            # Outbound peers we initiated — full metadata available
            for pid, peer in node.transport.peers.items():
                connected.append({
                    "peer_id": pid,
                    "address": peer.address,
                    "display_name": peer.display_name,
                    "connected_at": peer.connected_at,
                    "last_seen": peer.last_seen,
                    "outbound": peer.outbound,
                })
                seen_addresses.add(peer.address)

            # Inbound peers — only kernel knows about them. Use the
            # libp2p peer list to backfill the connected[] list.
            try:
                for kernel_addr in node.transport.peer_addresses:
                    if not kernel_addr or kernel_addr in seen_addresses:
                        continue
                    connected.append({
                        "peer_id": None,  # unknown without deeper C-bridge work
                        "address": kernel_addr,
                        "display_name": "",
                        "connected_at": None,
                        "last_seen": None,
                        "outbound": False,  # we didn't track them, so likely inbound
                    })
            except Exception as exc:  # noqa: BLE001
                logger.debug("peer_addresses probe failed: %s", exc)

        known = []
        if node.discovery:
            for info in node.discovery.get_known_peers():
                # Sprint 326 — surface capabilities so operators
                # find compute / gpu / storage peers via /peers
                # without having to hit /bootstrap/status. Pairs
                # with sprint 322's threading of caps from
                # bootstrap-server peer_list into PeerInfo.
                known.append({
                    "node_id": info.node_id,
                    "address": info.address,
                    "display_name": info.display_name,
                    "last_seen": info.last_seen,
                    "capabilities": list(
                        getattr(info, "capabilities", []) or []
                    ),
                })

        # Truth count from the libp2p host (matches /status). The
        # connected[] list above is best-effort union (outbound rich,
        # inbound minimal); count is kernel-authoritative.
        truth_count = (
            node.transport.peer_count if node.transport else 0
        )
        return {
            "connected": connected,
            "known": known,
            "connected_count": truth_count,
            "known_count": len(known),
        }

    # Sprint 266 — expose PeerDiscovery.get_bootstrap_status() so
    # operators can triage "am I actually connected to bootstrap
    # or just random peers?" without scraping logs.
    @app.get("/bootstrap/status", tags=["network"])
    async def get_bootstrap_status_endpoint() -> Dict[str, Any]:
        """Return bootstrap connection state for operator triage:
        configured/attempted/failed nodes, connected_count,
        degraded_mode, retry attempts, fallback telemetry, and
        whether the BootstrapClient is currently active."""
        disco = getattr(node, "discovery", None)
        if disco is None:
            raise HTTPException(
                status_code=503,
                detail="Peer discovery not initialized.",
            )
        try:
            return disco.get_bootstrap_status()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=(
                    f"get_bootstrap_status() raised: {exc}"
                ),
            )

    # ── Sprint 540 Pattern A — daemon-mediated bridge ─────────────
    # Replaces the polygon_mumbai-era /bridge/* scaffold (deferred
    # in sprint 539). Deposit: user signs on-chain transfer to
    # operator's escrow address; InboundMonitor (sprint 514) detects
    # + credits off-chain balance via linked-address registry.
    # Withdraw: daemon debits off-chain + signs on-chain transfer
    # out (sprint 541 follow-on).

    @app.get("/wallet/deposit/info", tags=["wallet"])
    async def get_deposit_info() -> Dict[str, Any]:
        """Return the operator escrow address + linkage status.

        Users deposit on-chain FTNS by signing an ERC-20 transfer
        from THEIR wallet to the operator escrow address shown here.
        InboundMonitor detects the inbound + credits the off-chain
        balance of the wallet_id linked to the sender address.
        """
        ledger = getattr(node, "ftns_ledger", None)
        local_ledger = getattr(node, "ledger", None)
        identity = getattr(node, "identity", None)
        if ledger is None or local_ledger is None or identity is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Deposit flow not initialized. Daemon must be "
                    "started with FTNS_WALLET_PRIVATE_KEY set "
                    "(escrow address derives from operator wallet) "
                    "and LocalLedger initialized."
                ),
            )
        escrow_addr = getattr(
            ledger, "_connected_address", None,
        )
        if not escrow_addr:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Operator on-chain wallet not connected. "
                    "FTNS_WALLET_PRIVATE_KEY may be unset."
                ),
            )
        wallet_id = identity.node_id
        linked_eth = await local_ledger.eth_address_for_wallet(
            wallet_id,
        )
        # Sprint 557: surface sprint-554's user-sig fields so CLI
        # clients can read everything they need to build a signed
        # withdraw in one HTTP call. Missing/legacy ledgers return
        # the defaults (flag off, nonce 0) per the schema spec.
        try:
            requires_sig = await local_ledger.get_requires_user_signature(
                wallet_id,
            )
        except Exception:  # noqa: BLE001
            requires_sig = False
        try:
            next_nonce = await local_ledger.get_next_withdraw_nonce(
                wallet_id,
            )
        except Exception:  # noqa: BLE001
            next_nonce = 0
        return {
            "escrow_address": escrow_addr,
            "wallet_id": wallet_id,
            "linked_eth_address": linked_eth,
            "ftns_token_contract": ledger.contract_address,
            "chain_id": ledger.chain_id,
            "requires_user_signature": bool(requires_sig),
            "next_withdraw_nonce": int(next_nonce),
            "instructions": (
                "1. Link your sending ETH address via "
                "POST /wallet/deposit/link {wallet_id, "
                "eth_address}. "
                "2. From your linked address, sign an ERC-20 "
                f"transfer of FTNS to {escrow_addr}. "
                "3. Daemon's InboundMonitor will detect the "
                "transfer + credit your off-chain balance "
                "automatically."
            ),
        }

    class _LinkEthAddressRequest(BaseModel):
        wallet_id: str
        eth_address: str

    @app.post("/wallet/deposit/link", tags=["wallet"])
    async def link_deposit_address(
        body: _LinkEthAddressRequest,
    ) -> Dict[str, Any]:
        """Link an ETH address to a PRSM wallet_id for bridge deposits.

        Future on-chain transfers FROM `eth_address` TO the operator
        escrow address will credit `wallet_id`'s off-chain balance.
        """
        local_ledger = getattr(node, "ledger", None)
        if local_ledger is None:
            raise HTTPException(
                status_code=503,
                detail="LocalLedger not initialized.",
            )
        try:
            await local_ledger.link_eth_address(
                body.wallet_id, body.eth_address,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            )
        return {
            "wallet_id": body.wallet_id,
            "eth_address": body.eth_address.lower(),
            "status": "linked",
        }

    # ──────────────────────────────────────────────────────────────
    # Sprint 554 — per-wallet user-signature requirement toggle.
    # First step in the user-sig arc (554/555/556/557): groundwork
    # for sprint-556's EIP-712 enforcement at /wallet/withdraw.
    # ──────────────────────────────────────────────────────────────
    class _RequireSignatureRequest(BaseModel):
        wallet_id: str
        enabled: bool

    @app.post("/wallet/require-signature", tags=["wallet"])
    async def post_require_signature(
        body: _RequireSignatureRequest,
    ) -> Dict[str, Any]:
        """Toggle the per-wallet user-signature requirement.

        When enabled, sprint-556 enforces that POST /wallet/withdraw
        carries a valid EIP-712 signature recovered to the wallet's
        linked eth_address (sprint 540). When disabled (default),
        withdraws use the legacy daemon-mediated flow.

        Returns 200 with the new flag state; 404 when wallet_id is
        unknown; 503 when the ledger isn't initialized.
        """
        ledger = getattr(node, "ledger", None)
        if ledger is None:
            raise HTTPException(
                status_code=503,
                detail="LocalLedger not initialized.",
            )
        try:
            await ledger.set_requires_user_signature(
                body.wallet_id, body.enabled,
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=str(exc),
            )
        return {
            "wallet_id": body.wallet_id,
            "requires_user_signature": bool(body.enabled),
        }

    # Sprint 541 — Pattern A withdraw half. Symmetric inverse of
    # deposit. Atomicity strategy: debit off-chain FIRST (pre-flight
    # balance check + ledger entry), THEN broadcast on-chain. If
    # broadcast fails, immediately credit a refund entry so the
    # off-chain balance is restored. Debit-first ordering avoids the
    # double-spend window where a user submits two concurrent
    # withdraws and both pass the balance check.
    class _WithdrawRequest(BaseModel):
        amount_ftns: float
        wallet_id: Optional[str] = None
        to_eth_address: Optional[str] = None
        # Sprint 556 — optional EIP-712 user-sig fields. Required
        # when the wallet's requires_user_signature flag is on
        # (sprint 554); ignored otherwise.
        signature: Optional[str] = None
        nonce: Optional[int] = None
        expiry_unix: Optional[int] = None

    @app.post("/wallet/withdraw", tags=["wallet"])
    async def post_withdraw(
        body: _WithdrawRequest,
    ) -> Dict[str, Any]:
        """Withdraw off-chain FTNS to on-chain.

        Debits the off-chain wallet by `amount_ftns` (Pattern A
        BRIDGE_WITHDRAW tx type), then broadcasts an on-chain
        ERC-20 transfer from the operator escrow address to
        `to_eth_address`. If `to_eth_address` is omitted, defaults
        to the wallet's linked address.

        Failure modes:
          - amount_ftns <= 0 → 422
          - no recipient (unlinked + no to_eth_address) → 400
          - insufficient off-chain balance → 402
          - broadcast failure → off-chain credit refund + 502
          - ledger/onchain not wired → 503
        """
        ledger = getattr(node, "ftns_ledger", None)
        local_ledger = getattr(node, "ledger", None)
        identity = getattr(node, "identity", None)
        if (
            ledger is None
            or local_ledger is None
            or identity is None
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Withdraw flow not initialized — daemon must "
                    "be started with FTNS_WALLET_PRIVATE_KEY + "
                    "off-chain ledger ready."
                ),
            )
        if body.amount_ftns <= 0:
            raise HTTPException(
                status_code=422,
                detail="amount_ftns must be > 0",
            )
        wallet_id = body.wallet_id or identity.node_id
        # Resolve recipient: explicit > linked > fail
        to_addr = body.to_eth_address
        if not to_addr:
            to_addr = await local_ledger.eth_address_for_wallet(
                wallet_id,
            )
        if not to_addr or not to_addr.startswith("0x"):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No recipient address. Either pass "
                    f"to_eth_address explicitly or link an address "
                    f"to wallet_id={wallet_id!r} via "
                    f"POST /wallet/deposit/link first."
                ),
            )
        # ─── Sprint 556 — user-sig enforcement ────────────────────
        # When the wallet's requires_user_signature flag is True,
        # the request body must carry a valid EIP-712 signature
        # over {wallet_id, amount_ftns_wei, to_eth_address, nonce,
        # expiry_unix}. Each check below returns 401 with a
        # specific detail so callers can self-diagnose:
        #   - missing signature/nonce/expiry → "signature required"
        #   - unlinked wallet → "link eth address first"
        #   - expired payload → "payload expired"
        #   - wrong nonce → "nonce mismatch"
        #   - wrong signer → "signer address mismatch"
        # On success the nonce is consumed BEFORE the broadcast —
        # even if broadcast fails (502 + refund), the nonce stays
        # bumped so the captured signature can't be replayed.
        nonce_consumed: Optional[int] = None
        try:
            requires_sig = await local_ledger.get_requires_user_signature(
                wallet_id,
            )
        except Exception:  # noqa: BLE001
            requires_sig = False
        if requires_sig:
            if (
                body.signature is None
                or body.nonce is None
                or body.expiry_unix is None
            ):
                raise HTTPException(
                    status_code=401,
                    detail=(
                        "Wallet requires user signature: pass "
                        "`signature` (EIP-712 over WithdrawRequest), "
                        "`nonce` (current next_withdraw_nonce), and "
                        "`expiry_unix` (Unix epoch in the future) "
                        "in the body. See sprint-555's "
                        "`prsm.economy.withdraw_signature` module for "
                        "the canonical payload shape."
                    ),
                )
            linked = await local_ledger.eth_address_for_wallet(
                wallet_id,
            )
            if not linked:
                raise HTTPException(
                    status_code=401,
                    detail=(
                        "Wallet has requires_user_signature=True but "
                        "no linked eth_address — there is nothing to "
                        "verify the signer against. Link an address "
                        "via POST /wallet/deposit/link first."
                    ),
                )
            from prsm.economy.withdraw_signature import (
                verify_withdraw_signature,
                is_expired,
                InvalidSignatureFormat,
            )
            payload = {
                "wallet_id": wallet_id,
                "amount_ftns_wei": int(body.amount_ftns * 1e18),
                "to_eth_address": to_addr,
                "nonce": int(body.nonce),
                "expiry_unix": int(body.expiry_unix),
            }
            if is_expired(payload):
                raise HTTPException(
                    status_code=401,
                    detail=(
                        "Signed payload expired "
                        f"(expiry_unix={body.expiry_unix}). "
                        "Re-sign with a future expiry."
                    ),
                )
            expected_nonce = await local_ledger.get_next_withdraw_nonce(
                wallet_id,
            )
            if int(body.nonce) != int(expected_nonce):
                raise HTTPException(
                    status_code=401,
                    detail=(
                        f"Nonce mismatch: got {body.nonce}, expected "
                        f"{expected_nonce}. Read current nonce via "
                        "GET /wallet/deposit/info (sprint-556 field)."
                    ),
                )
            try:
                recovered = verify_withdraw_signature(
                    payload, body.signature,
                )
            except InvalidSignatureFormat as exc:
                raise HTTPException(
                    status_code=401,
                    detail=f"Signature format invalid: {exc!s}",
                )
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=401,
                    detail=f"Signature verification failed: {exc!s}"[:200],
                )
            if recovered.lower() != linked.lower():
                raise HTTPException(
                    status_code=401,
                    detail=(
                        f"Signer address mismatch: signature recovered "
                        f"to {recovered.lower()}, but wallet is linked "
                        f"to {linked.lower()}. Sign with the same key "
                        f"used to link via /wallet/deposit/link."
                    ),
                )
            # All checks passed — consume the nonce BEFORE broadcast
            # so replay isn't possible even if broadcast fails.
            nonce_consumed = await local_ledger.bump_withdraw_nonce(
                wallet_id,
            )
        # Pre-flight balance check
        balance = await local_ledger.get_balance(wallet_id)
        if balance < body.amount_ftns:
            raise HTTPException(
                status_code=402,
                detail=(
                    f"Insufficient off-chain balance: "
                    f"{balance:.6f} < {body.amount_ftns:.6f}"
                ),
            )
        # Debit off-chain FIRST. Withdraw the off-chain balance to
        # `system` (DAGLedger debit convention) BEFORE broadcast so
        # concurrent withdraws can't both pass the balance check.
        from prsm.node.dag_ledger import (
            TransactionType as _DAG_TT,
        )
        from prsm.node.local_ledger import (
            TransactionType as _LL_TT,
        )
        # Pick the right TransactionType enum based on ledger class
        if isinstance(local_ledger, __import__(
            "prsm.node.dag_ledger", fromlist=["DAGLedger"],
        ).DAGLedger):
            withdraw_tt = _DAG_TT.BRIDGE_WITHDRAW
            refund_tt_for_credit = _DAG_TT.BRIDGE_WITHDRAW
        else:
            withdraw_tt = _LL_TT.BRIDGE_WITHDRAW
            refund_tt_for_credit = _LL_TT.BRIDGE_WITHDRAW
        try:
            debit_tx = await local_ledger.debit(
                wallet_id=wallet_id,
                amount=body.amount_ftns,
                tx_type=withdraw_tt,
                description=(
                    f"bridge withdraw to {to_addr} "
                    f"(broadcast pending)"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=f"Off-chain debit failed: {exc!s}"[:300],
            )
        # Broadcast on-chain ERC-20 transfer from escrow.
        import uuid
        job_id = f"withdraw-{uuid.uuid4().hex[:12]}"
        tx_record = await ledger.transfer(
            job_id=job_id,
            to_address=to_addr,
            amount_ftns=body.amount_ftns,
        )
        if tx_record is None or getattr(
            tx_record, "status", None,
        ) == "rejected":
            # Broadcast failed → refund the off-chain debit so the
            # user is whole. The refund is its own ledger entry so
            # the audit trail shows: debit (pending) → refund
            # (broadcast failed). Future readers can reconstruct
            # via tx descriptions.
            try:
                await local_ledger.credit(
                    wallet_id=wallet_id,
                    amount=body.amount_ftns,
                    tx_type=refund_tt_for_credit,
                    description=(
                        f"bridge withdraw REFUND "
                        f"(broadcast failed) — original debit "
                        f"tx_id={debit_tx.tx_id}"
                    ),
                )
            except Exception as refund_exc:  # noqa: BLE001
                logger.error(
                    "CRITICAL: withdraw refund failed AFTER debit "
                    "succeeded — manual reconciliation needed. "
                    "wallet=%s amount=%.6f debit_tx=%s "
                    "refund_error=%s",
                    wallet_id, body.amount_ftns,
                    debit_tx.tx_id, refund_exc,
                )
            raise HTTPException(
                status_code=502,
                detail=(
                    f"On-chain broadcast failed; off-chain debit "
                    f"refunded. debit_tx_id={debit_tx.tx_id}"
                ),
            )
        # Success path. sp914 — report the on-chain ledger's ACTUAL status:
        # a broadcast that succeeded but is still unconfirmed comes back as
        # "pending" (not None), so we must NOT claim "confirmed" for it. The
        # off-chain debit is intentionally NOT refunded for a pending tx — it
        # is in the mempool and refunding would double-pay once it confirms.
        _tx_status = getattr(tx_record, "status", "confirmed") or "confirmed"
        if _tx_status == "pending":
            # sp916 — record the pending withdraw so the reconciler can refund
            # this off-chain debit IF the on-chain tx ultimately reverts (the
            # debit already happened above; without this record the refund-on-
            # revert path has no wallet_id/amount linkage). Confirmed withdraws
            # need no record. Best-effort: never fail the response on store I/O.
            _pw_store = getattr(node, "_pending_withdraw_store", None)
            if _pw_store is not None:
                try:
                    _pw_store.record(
                        job_id=job_id,
                        wallet_id=wallet_id,
                        amount=float(body.amount_ftns),
                        to_addr=to_addr,
                        tx_hash=getattr(tx_record, "tx_hash", "") or "",
                    )
                except Exception as _exc:  # noqa: BLE001
                    logger.warning(
                        "withdraw: failed to record pending intent for "
                        "job=%s: %s", job_id, _exc,
                    )
        return {
            "status": _tx_status,
            "wallet_id": wallet_id,
            "amount_ftns": body.amount_ftns,
            "to_eth_address": to_addr,
            "tx_hash": getattr(tx_record, "tx_hash", None),
            "block_number": getattr(
                tx_record, "block_number", None,
            ),
            "debit_tx_id": debit_tx.tx_id,
            "job_id": job_id,
            # Sprint 556 — only present when requires_user_signature
            # is on for this wallet. Echoes back which nonce was
            # consumed so callers can verify their bookkeeping.
            "nonce_consumed": nonce_consumed,
        }

    @app.get("/balance")
    async def get_balance() -> Dict[str, Any]:
        """Get FTNS balance and recent transactions."""
        if not node.ledger or not node.identity:
            raise HTTPException(status_code=503, detail="Node not initialized")

        balance = await node.ledger.get_balance(node.identity.node_id)
        history = await node.ledger.get_transaction_history(node.identity.node_id, limit=20)

        return {
            "wallet_id": node.identity.node_id,
            "balance": balance,
            "recent_transactions": [
                {
                    "tx_id": tx.tx_id,
                    "type": tx.tx_type.value,
                    "from": tx.from_wallet,
                    "to": tx.to_wallet,
                    "amount": tx.amount,
                    "description": tx.description,
                    "timestamp": tx.timestamp,
                }
                for tx in history
            ],
        }

    @app.get("/balance/onchain")
    async def get_balance_onchain(
        address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Read on-chain FTNS balance + USD equivalent.

        Backend for the prsm_balance_check MCP tool (v1 scope per
        Vision §13 Phase 5 stand-in closure). Reads via the node's
        already-initialized OnChainFTNSLedger; converts to USD using
        ``PRSM_FTNS_USD_RATE`` env var (default 1.0; placeholder
        until the Aerodrome USDC-FTNS pool is seeded per Vision
        gantt 2026-06-15).

        Query params:
            address: optional override; defaults to the ledger's
                connected address.
        """
        if not getattr(node, "ftns_ledger", None):
            raise HTTPException(
                status_code=503,
                detail=(
                    "On-chain ftns_ledger not initialized; "
                    "set PRSM_ONCHAIN_FTNS=1 + FTNS_TOKEN_ADDRESS "
                    "to enable."
                ),
            )

        target = address or node.ftns_ledger._connected_address
        balance_ftns = await node.ftns_ledger.get_balance(target)

        # USD-rate parsing — graceful fallback to 1.0 on misconfig.
        rate_raw = os.getenv("PRSM_FTNS_USD_RATE", "").strip()
        usd_rate = 1.0
        if rate_raw:
            try:
                parsed = float(rate_raw)
                if parsed > 0:
                    usd_rate = parsed
            except ValueError:
                pass  # keep default

        decimals = getattr(node.ftns_ledger, "_decimals", 18)
        balance_wei = int(balance_ftns * (10 ** decimals))
        usd_equivalent = balance_ftns * usd_rate

        # ── Aggregate-source quoting (audit-prep §7.23 honest-scope) ──
        # Read claimable royalties (RoyaltyDistributor) + escrowed
        # FTNS (PaymentEscrow). Each source independently fail-soft:
        # if reading raises, log + continue + report the source as
        # unavailable. Aggregate doesn't crash on partial sources.

        # Source 2: claimable royalties.
        claimable_ftns = 0.0
        claimable_available = False
        royalty_client = getattr(node, "_royalty_distributor_client", None)
        if royalty_client is not None:
            try:
                claimable_wei = royalty_client.claimable(target)
                claimable_ftns = float(claimable_wei) / (10 ** decimals)
                claimable_available = True
            except Exception as e:
                logger.warning(
                    "balance/onchain: claimable royalties read failed "
                    "(%s); reporting source unavailable", e,
                )

        # Source 3: escrowed FTNS in pending compute jobs.
        escrowed_ftns = 0.0
        escrowed_available = False
        payment_escrow = getattr(node, "_payment_escrow", None)
        if payment_escrow is not None:
            try:
                pending = payment_escrow.list_escrows_by_requester(target)
                # Sprint 162 — start=0.0 keeps the type float-stable when
                # `pending` is empty (sum([]) is int 0, breaking the
                # uniform-float contract sibling fields hold).
                escrowed_ftns = sum(
                    (e.amount for e in pending),
                    start=0.0,
                )
                escrowed_available = True
            except Exception as e:
                logger.warning(
                    "balance/onchain: escrow listing failed (%s); "
                    "reporting source unavailable", e,
                )

        total_ftns = balance_ftns + claimable_ftns + escrowed_ftns
        total_usd_equivalent = total_ftns * usd_rate

        return {
            # ── v1 fields (preserved bit-identically) ──
            "address": target,
            "balance_wei": balance_wei,
            "balance_ftns": balance_ftns,
            "usd_rate": usd_rate,
            "usd_equivalent": usd_equivalent,
            "source": "onchain",
            # ── v2 aggregate fields (additive) ──
            "claimable_royalties_ftns": claimable_ftns,
            "escrowed_ftns": escrowed_ftns,
            "total_ftns": total_ftns,
            "total_usd_equivalent": total_usd_equivalent,
            "sources": {
                "onchain": {
                    "ftns": balance_ftns,
                    "available": True,
                },
                "claimable_royalties": {
                    "ftns": claimable_ftns,
                    "available": claimable_available,
                },
                "escrowed": {
                    "ftns": escrowed_ftns,
                    "available": escrowed_available,
                },
            },
        }

    # Renamed from `_OfframpQuoteRequest` (sprint dogfood) so
    # OpenAPI schemas don't expose Python-private leading-
    # underscore convention to public API consumers. CI-enforced
    # composer-only invariant R-2026-05-08-1's source-string
    # marker is updated in the same commit per same-change-set
    # supersession protocol; the field-set invariant (no execute
    # tokens) is unchanged.
    class OfframpQuoteRequest(BaseModel):
        # Validation is in the handler so the 400 vs 422 boundary is
        # explicit (Pydantic's gt=0 returns 422; we want 400 for
        # operator-misconfig-class errors).
        usd_amount: float
        bank_account_alias: str = "primary"
        # Sprint 281 — optional source_user_id resolves the
        # source address via WaaS + drives KYC gating.
        # Backwards compat: when None, legacy `address` query
        # param path is used unchanged and no KYC gating applies.
        source_user_id: Optional[str] = None

    # Sprint 278 — Coinbase onramp quote (USD → FTNS).
    # Mirrors the offramp quote: composer-only
    # PENDING_COMMISSION artifact. Either destination_user_id
    # (resolved via WaaS) or destination_address (explicit)
    # required; XOR enforced in the handler.
    class OnrampQuoteRequest(BaseModel):
        usd_amount: float
        destination_user_id: Optional[str] = None
        destination_address: Optional[str] = None
        payment_method_alias: str = "primary"

    @app.post("/wallet/onramp/quote", tags=["wallet"])
    async def post_onramp_quote(
        body: OnrampQuoteRequest,
    ) -> Dict[str, Any]:
        """Pre-flight quote for USD → USDC → FTNS on-ramp via
        Coinbase CDP + Aerodrome.

        Composer-only PENDING_COMMISSION: returns artifact only.
        Does NOT initiate any swap or fiat on-ramp; actual
        execution gates on CDP commission per Vision gantt
        2026-06-22."""
        if body.usd_amount <= 0:
            raise HTTPException(
                status_code=400,
                detail="usd_amount must be positive (> 0)",
            )
        # XOR: exactly one destination must be supplied.
        has_user_id = bool(body.destination_user_id)
        has_address = bool(body.destination_address)
        if has_user_id and has_address:
            raise HTTPException(
                status_code=422,
                detail=(
                    "supply destination_user_id OR "
                    "destination_address, not both"
                ),
            )
        if not has_user_id and not has_address:
            raise HTTPException(
                status_code=422,
                detail=(
                    "either destination_user_id or "
                    "destination_address is required"
                ),
            )

        destination_user_id: Optional[str] = body.destination_user_id
        destination_address: Optional[str] = body.destination_address
        note: Optional[str] = None

        if has_user_id:
            waas = getattr(node, "_coinbase_waas_client", None)
            if waas is None:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "WaaS client not initialized; cannot "
                        "resolve destination_user_id."
                    ),
                )
            record = waas.get_wallet(body.destination_user_id)
            if record is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"no WaaS wallet for destination_user_id="
                        f"{body.destination_user_id!r}; run "
                        f"/wallet/waas/provision first"
                    ),
                )
            destination_address = record.address  # may be None pre-commission
            if not destination_address:
                note = (
                    f"Destination WaaS wallet is "
                    f"status={record.status} (no address yet). "
                    f"Quote returns PENDING_COMMISSION until "
                    f"both Coinbase CDP and the WaaS wallet "
                    f"are commissioned."
                )

        # Rate: PRSM_FTNS_USD_RATE env (default 1.0). Mirrors the
        # offramp side so a 1.0 default is round-trippable.
        rate_raw = os.getenv("PRSM_FTNS_USD_RATE", "").strip()
        usd_rate = 1.0
        if rate_raw:
            try:
                parsed = float(rate_raw)
                if parsed > 0:
                    usd_rate = parsed
            except ValueError:
                pass

        ftns_to_receive = body.usd_amount / usd_rate

        # Sprint 281 — KYC gating. When destination_user_id is
        # supplied AND a KYC client is wired, surface the
        # prerequisite (mirrors the claim_required pattern).
        # Explicit destination_address bypasses (caller-side
        # responsibility). When KYC client is unwired, fields
        # stay neutral so the artifact is still useful.
        kyc_required = False
        kyc_status = None
        kyc_session_url = None
        if destination_user_id:
            kyc = getattr(node, "_kyc_client", None)
            if kyc is not None:
                rec = kyc.get_status(destination_user_id)
                if rec is None:
                    kyc_required = True
                    kyc_status = "NOT_STARTED"
                elif rec.status != "VERIFIED":
                    kyc_required = True
                    kyc_status = rec.status
                    kyc_session_url = rec.session_url
                else:
                    kyc_status = "VERIFIED"

        # Sprint 285 — tier-limit check. Surface limit fields
        # in the artifact (neutral when no user_id / unverified).
        tier_block = _tier_check(
            user_id=destination_user_id,
            requested_usd=float(body.usd_amount),
            kyc_status=kyc_status,
        )

        # Sprint 282 — record to fiat compliance audit ring.
        # Best-effort; failures don't break the primary
        # surface.
        _record_fiat_compliance(
            kind="onramp_quote",
            user_id=destination_user_id or "",
            usd_amount=float(body.usd_amount),
            ftns_amount=float(ftns_to_receive),
            status="PENDING_COMMISSION",
            kyc_status=kyc_status,
            address=destination_address,
            metadata={
                "payment_method_alias":
                    body.payment_method_alias,
            },
        )

        return {
            "requested_usd": body.usd_amount,
            "destination_user_id": destination_user_id,
            "destination_address": destination_address,
            "ftns_to_receive": ftns_to_receive,
            "usd_rate": usd_rate,
            "kyc_required": kyc_required,
            "kyc_status": kyc_status,
            "kyc_session_url": kyc_session_url,
            **tier_block,
            "quote": {
                "usd_in": body.usd_amount,
                "usdc_acquired": body.usd_amount,
                "ftns_received": ftns_to_receive,
                "onramp_route": "coinbase-cdp",
                "swap_route": "aerodrome",
                "payment_method_alias": body.payment_method_alias,
            },
            "status": "PENDING_COMMISSION",
            "note": note or (
                "Coinbase CDP onramp commission gates on "
                "Aerodrome USDC-FTNS pool seeding (Vision "
                "gantt 2026-06-22). This is a preview "
                "artifact; nothing has moved on-chain or "
                "via fiat rails."
            ),
        }

    # Sp853 — onramp execute: builds the user-facing Coinbase Pay
    # widget URL that completes USD → USDC purchase. User opens
    # the URL, pays via Coinbase, and USDC lands in their WaaS
    # wallet on Base. From there a separate Aerodrome swap step
    # (sibling sprint, gated on pool ceremony) converts to FTNS.
    @app.post("/wallet/onramp/execute", tags=["wallet"])
    async def post_onramp_execute(
        body: OnrampQuoteRequest,
    ) -> Dict[str, Any]:
        if body.usd_amount <= 0:
            raise HTTPException(
                status_code=400,
                detail="usd_amount must be positive (> 0)",
            )
        # Same XOR destination validation as the quote endpoint.
        has_user_id = bool(body.destination_user_id)
        has_address = bool(body.destination_address)
        if has_user_id and has_address:
            raise HTTPException(
                status_code=422,
                detail=(
                    "supply destination_user_id OR "
                    "destination_address, not both"
                ),
            )
        if not has_user_id and not has_address:
            raise HTTPException(
                status_code=422,
                detail=(
                    "either destination_user_id or "
                    "destination_address is required"
                ),
            )

        # Sp884 — ENFORCE KYC + tier limits before minting a
        # real-money onramp session. The quote endpoint surfaces
        # kyc_required + tier_limit_exceeded as ADVISORY flags;
        # execute must BLOCK so a caller cannot skip the quote and
        # over-transact on an unverified / over-limit account. Only
        # the destination_user_id path is gated — a raw
        # destination_address has no PRSM identity to enforce against
        # (documented operator responsibility, mirroring the quote).
        # sp1185 (day-one-live blocker #6) — raw destination_address has NO PRSM identity
        # to KYC against; minting a real-money onramp session for one bypasses PRSM-side
        # KYC/tier/AML entirely. FAIL CLOSED: off unless the operator EXPLICITLY accepts
        # that responsibility via PRSM_FIAT_ALLOW_UNVERIFIED_ADDRESS_ONRAMP=1. (Was: raw
        # address silently minted with no checks.)
        if has_address:
            _allow_raw = (
                os.environ.get("PRSM_FIAT_ALLOW_UNVERIFIED_ADDRESS_ONRAMP", "")
                .strip().lower()
            ) in ("1", "true", "yes", "on")
            if not _allow_raw:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "raw_address_onramp_disabled",
                        "message": (
                            "Onramp to a raw destination_address is disabled because it "
                            "bypasses PRSM-side KYC/tier/AML enforcement. Use the "
                            "identity-gated destination_user_id path, or the operator "
                            "must set PRSM_FIAT_ALLOW_UNVERIFIED_ADDRESS_ONRAMP=1 to "
                            "accept off-PRSM KYC responsibility."
                        ),
                    },
                )
        if has_user_id:
            _kyc = getattr(node, "_kyc_client", None)
            # sp1185 — FAIL CLOSED when KYC enforcement is unavailable. Previously a node
            # with no _kyc_client wired (the default day-one state) SKIPPED the entire
            # KYC + tier + AML block and minted an unverified session — a regulatory
            # bypass masquerading as a gate. An identity-gated onramp must not proceed
            # without the KYC gate that is supposed to protect it.
            if _kyc is None:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "kyc_enforcement_unavailable",
                        "message": (
                            "Fiat onramp for an identity (destination_user_id) requires "
                            "KYC enforcement, but no KYC client is configured on this "
                            "node. Onramp is unavailable until the operator wires KYC "
                            "(fail-closed; refusing to mint an unverified session)."
                        ),
                    },
                )
            if _kyc is not None:  # always true here (None raised above); defensive
                _rec = _kyc.get_status(body.destination_user_id)
                if _rec is None or _rec.status != "VERIFIED":
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "error": "kyc_required",
                            "kyc_status": (
                                _rec.status if _rec
                                else "NOT_STARTED"
                            ),
                            "kyc_session_url": (
                                _rec.session_url if _rec else None
                            ),
                            "message": (
                                "Complete KYC verification before "
                                "onramping. Initiate via POST "
                                "/wallet/kyc/initiate."
                            ),
                        },
                    )
                _tier = _tier_check(
                    user_id=body.destination_user_id,
                    requested_usd=float(body.usd_amount),
                    kyc_status=_rec.status,
                )
                # Sp1175 — strict_shared fail-closed: the AML tier limit
                # can't be enforced globally on this deployment, so deny
                # rather than under-enforce. 503 (server-side capability
                # gap), NOT 403 (the caller is not over a known limit).
                if _tier.get("tier_limit_strict_block"):
                    raise HTTPException(
                        status_code=503,
                        detail={
                            "error": "tier_limit_enforcement_unavailable",
                            "tier_limit_scope": "process_local",
                            "message": (
                                "Fiat onramp is temporarily unavailable: "
                                "the per-user AML tier limit cannot be "
                                "enforced with global consistency on this "
                                "deployment (PRSM_FIAT_TIER_LIMIT_MODE="
                                "strict_shared, no shared real-time "
                                "compliance backend). Contact the operator."
                            ),
                        },
                    )
                if _tier["tier_limit_exceeded"]:
                    _lvl = _tier["tier_level"]
                    _upgrade = (
                        "Complete enhanced KYC (Tier 2) to raise "
                        "your limit: POST /wallet/kyc/initiate with "
                        "level=enhanced."
                        if _lvl == "basic"
                        else "Contact support to raise your limit."
                    )
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "error": "tier_limit_exceeded",
                            "tier_level": _lvl,
                            "tier_limit_usd": _tier["tier_limit_usd"],
                            "tier_limit_remaining_usd": _tier[
                                "tier_limit_remaining_usd"
                            ],
                            "requested_usd": float(body.usd_amount),
                            "message": (
                                f"This onramp would exceed your "
                                f"{_lvl}-tier limit of "
                                f"${_tier['tier_limit_usd']:.2f}. "
                                + _upgrade
                            ),
                        },
                    )

        destination_address: Optional[str] = body.destination_address
        if has_user_id:
            waas = getattr(node, "_coinbase_waas_client", None)
            if waas is None:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "WaaS client not initialized; cannot "
                        "resolve destination_user_id."
                    ),
                )
            record = waas.get_wallet(body.destination_user_id)
            if record is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"no WaaS wallet for destination_user_id="
                        f"{body.destination_user_id!r}; run "
                        f"/wallet/waas/provision first"
                    ),
                )
            destination_address = record.address

        if not destination_address:
            return {
                "status": "PENDING_COMMISSION",
                "session_url": None,
                "note": (
                    "Destination WaaS wallet has no address yet "
                    "(adapter_wired=False). Paste Ed25519 PEM "
                    "into COINBASE_CDP_API_KEY_PRIVATE so the "
                    "next provision call returns a real address."
                ),
            }

        # Sp855 — secure init: CDP projects with secure-init enabled
        # (the default on new projects) reject the legacy params-only
        # widget URL with "Missing or invalid parameters". We mint a
        # one-time 5-minute sessionToken server-side that embeds the
        # destination wallet + asset constraints, then redirect the
        # user to pay.coinbase.com?sessionToken=<token>.
        from prsm.economy.web3.coinbase_onramp_session import (
            from_env as _onramp_session_from_env,
        )
        onramp_client = _onramp_session_from_env()
        if onramp_client is None:
            return {
                "status": "PENDING_COMMISSION",
                "session_url": None,
                "destination_address": destination_address,
                "note": (
                    "Coinbase CDP API key not configured for onramp."
                    " Set COINBASE_CDP_API_KEY_NAME + "
                    "COINBASE_CDP_API_KEY_PRIVATE."
                ),
            }
        try:
            minted = onramp_client.build_buy_url(
                address=destination_address,
                usd_amount=body.usd_amount,
                partner_user_ref=body.destination_user_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "onramp session token mint failed: %s", exc,
            )
            raise HTTPException(
                status_code=502,
                detail=f"coinbase onramp /token call failed: {exc}",
            )
        finally:
            onramp_client.close()

        # Sp857 — record the funnel-entry event so the conversion
        # tracker can sweep on-chain truth + flag CONFIRMED arrivals.
        from prsm.economy.web3.onramp_funnel import OnrampFunnel
        funnel = getattr(node, "_onramp_funnel", None)
        if funnel is None:
            funnel = OnrampFunnel()
            setattr(node, "_onramp_funnel", funnel)
        intent = funnel.record_intent(
            user_id=body.destination_user_id,
            destination_address=destination_address,
            expected_usd=body.usd_amount,
            session_token=minted["token"],
        )

        # sp968 — record an onramp_execute PENDING reservation NOW (at session
        # mint), tagged with the intent_id, so the AML rolling total counts this
        # in-flight onramp immediately. Previously onramp_execute was recorded
        # only at CONFIRMED (the on-chain sweep, minutes-to-hours later), so
        # repeated /onramp/execute calls all saw a stale total → tier bypass.
        # total_usd_for_user dedups by intent_id, so this PENDING + the later
        # CONFIRMED settle (same intent_id) count once. Fail-soft: a ring error
        # must not break the session response.
        _cring = getattr(node, "_fiat_compliance_ring", None)
        if _cring is not None and body.destination_user_id:
            try:
                _cring.record(
                    kind="onramp_execute",
                    user_id=body.destination_user_id,
                    usd_amount=float(body.usd_amount),
                    ftns_amount=0.0,
                    status="PENDING",
                    address=destination_address,
                    metadata={"intent_id": intent.intent_id},
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "sp968: onramp reservation record failed for intent %s: %s",
                    intent.intent_id, exc,
                )

        return {
            "status": "SESSION_READY",
            "session_url": minted["session_url"],
            "session_token": minted["token"],
            "channel_id": minted["channel_id"],
            "intent_id": intent.intent_id,
            "destination_address": destination_address,
            "usd_amount": body.usd_amount,
            "asset": "USDC",
            "network": "base",
            "expires_in_seconds": 300,
            "note": (
                "Open session_url to complete USD → USDC purchase. "
                "Token is single-use + 5-min expiry. USDC lands at "
                "destination_address on Base. Conversion tracked "
                "via intent_id — sweep with GET /wallet/onramp/funnel."
                " Aerodrome USDC→FTNS swap gated on pool ceremony "
                "(Vision 2026-06-15)."
            ),
        }

    # Sp855 — Aerodrome USDC→FTNS swap surface.
    # Quote endpoint is live the moment the AerodromeClient's RPC +
    # pool address are configured (gated on Foundation Safe seeding
    # ceremony per Vision 2026-06-15). Execute endpoint returns
    # SESSION_READY with the prepared swap envelope (router address,
    # routes, amount_out_min, deadline) once everything's wired —
    # the actual signed-tx submit via CDP WaaS is sp856-class.
    # Sp890 — swap slippage ceiling (anti-sandwich). slippage_bps
    # near 10000 zeroes out amountOutMin (the swap accepts ANY
    # output = unbounded MEV/sandwich loss). Enforce a sane max:
    # PRSM_SWAP_MAX_SLIPPAGE_BPS (default 1000 = 10%), raisable by
    # operators for thin-liquidity launch conditions. 10000 is an
    # ABSOLUTE reject regardless of the env ceiling (it always
    # yields amountOutMin=0).
    def _check_swap_slippage(slippage_bps: int) -> None:
        if not (0 <= slippage_bps <= 10_000):
            raise HTTPException(
                status_code=422,
                detail=(
                    "slippage_bps must be in [0, 10000] (basis "
                    "points; 100 = 1%)"
                ),
            )
        max_raw = os.environ.get(
            "PRSM_SWAP_MAX_SLIPPAGE_BPS", "",
        ).strip()
        max_slip = 1000
        if max_raw:
            try:
                parsed = int(max_raw)
                if parsed > 0:
                    max_slip = parsed
            except ValueError:
                pass
        # 10000 always rejected (amountOutMin would be 0 → accept
        # any output); otherwise enforce the configured ceiling.
        if slippage_bps >= 10_000 or slippage_bps > max_slip:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"slippage_bps={slippage_bps} exceeds the "
                    f"maximum allowed ({min(max_slip, 9999)} bps). "
                    "High slippage enables sandwich/MEV loss; raise "
                    "PRSM_SWAP_MAX_SLIPPAGE_BPS only for thin-"
                    "liquidity launch conditions."
                ),
            )

    class SwapQuoteRequest(BaseModel):
        amount_in: float  # whole-token units (USDC or FTNS)
        token_in: str  # "USDC" or "FTNS"
        slippage_bps: int = 100  # 1% default — operator override

    @app.post("/wallet/swap/quote", tags=["wallet"])
    async def post_swap_quote(
        body: SwapQuoteRequest,
    ) -> Dict[str, Any]:
        """Quote a USDC↔FTNS swap via Aerodrome.

        Returns POOL_NOT_CONFIGURED when AERODROME_USDC_FTNS_POOL_ADDRESS
        is unset (pool seeding ceremony still pending). Returns
        POOL_UNAVAILABLE when configured but the RPC call returned
        no state. Returns OK with quote envelope otherwise.
        """
        if body.amount_in <= 0:
            raise HTTPException(
                status_code=400,
                detail="amount_in must be > 0",
            )
        token_in = (body.token_in or "").strip().upper()
        if token_in not in {"USDC", "FTNS"}:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"token_in must be USDC or FTNS, got "
                    f"{body.token_in!r}"
                ),
            )
        _check_swap_slippage(body.slippage_bps)  # sp890 ceiling
        pool = getattr(node, "_aerodrome_client", None)
        if pool is None:
            raise HTTPException(
                status_code=503,
                detail="Aerodrome client not initialized.",
            )
        if not pool.is_configured():
            return {
                "status": "POOL_NOT_CONFIGURED",
                "note": (
                    "Set BASE_RPC_URL + "
                    "AERODROME_USDC_FTNS_POOL_ADDRESS env vars "
                    "after the Foundation Safe seeding ceremony "
                    "(Vision gantt 2026-06-15)."
                ),
            }

        # Resolve token0/token1 addresses from pool state.
        state = pool.get_pool_state()
        if state is None:
            return {
                "status": "POOL_UNAVAILABLE",
                "note": (
                    "Pool RPC returned no state — Base RPC may be "
                    "unreachable or the pool contract may not exist "
                    "at the configured address yet."
                ),
            }

        # Map symbolic USDC/FTNS to canonical Base mainnet addresses.
        # USDC: canonical Circle deployment on Base.
        # FTNS: PRSM's mainnet token from networks.py.
        from prsm.config.networks import get_network_config
        net = get_network_config("mainnet")
        usdc_addr = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
        ftns_addr = net.ftns_token
        token_in_addr = (
            usdc_addr if token_in == "USDC" else ftns_addr
        )

        # Convert whole-token amount_in to base units.
        # USDC has 6 decimals; FTNS has 18.
        decimals = 6 if token_in == "USDC" else 18
        amount_in_units = int(body.amount_in * (10 ** decimals))

        try:
            quote = pool.quote_swap(
                amount_in=amount_in_units,
                token_in=token_in_addr,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Aerodrome quote_swap raised: %s", exc,
            )
            raise HTTPException(
                status_code=502,
                detail=f"pool quote failed: {exc}",
            )
        if quote is None:
            return {
                "status": "POOL_UNAVAILABLE",
                "note": "Pool quote returned None.",
            }

        # Whole-token denominate the output (USDC=6, FTNS=18).
        out_token = "FTNS" if token_in == "USDC" else "USDC"
        out_decimals = 18 if out_token == "FTNS" else 6
        amount_out_whole = quote.amount_out / (10 ** out_decimals)
        amount_out_min_units = (
            quote.amount_out
            * (10_000 - body.slippage_bps) // 10_000
        )

        return {
            "status": "OK",
            "token_in": token_in,
            "token_out": out_token,
            "amount_in": body.amount_in,
            "amount_in_units": amount_in_units,
            "amount_out": amount_out_whole,
            "amount_out_units": quote.amount_out,
            "amount_out_min_units": amount_out_min_units,
            "price_impact_bps": quote.price_impact_bps,
            "fee_bps": quote.fee_bps,
            "slippage_bps": body.slippage_bps,
            "route": "aerodrome",
            "pool_address": state.pool_address,
        }

    class SwapExecuteRequest(BaseModel):
        amount_in: float
        token_in: str
        slippage_bps: int = 100
        # XOR: explicit address, or look up by user_id via WaaS.
        from_user_id: Optional[str] = None
        from_address: Optional[str] = None

    @app.post("/wallet/swap/execute", tags=["wallet"])
    async def post_swap_execute(
        body: SwapExecuteRequest,
    ) -> Dict[str, Any]:
        """Prepare an Aerodrome swap envelope ready for CDP-signed
        submission.

        Returns SESSION_READY with the full swap payload (router
        call data, amount_out_min, deadline) once the pool is seeded
        and the user has a WaaS wallet with sufficient USDC. Actual
        on-chain submission via CDP WaaS signed-tx flow is sp856.
        """
        # Reuse the quote handler's pool/token/decimals logic via
        # a synthesized quote request.
        if body.amount_in <= 0:
            raise HTTPException(
                status_code=400,
                detail="amount_in must be > 0",
            )
        _check_swap_slippage(body.slippage_bps)  # sp890 ceiling
        has_user = bool(body.from_user_id)
        has_addr = bool(body.from_address)
        if has_user and has_addr:
            raise HTTPException(
                status_code=422,
                detail=(
                    "supply from_user_id OR from_address, not both"
                ),
            )
        if not has_user and not has_addr:
            raise HTTPException(
                status_code=422,
                detail=(
                    "either from_user_id or from_address is required"
                ),
            )

        # Resolve from_address via WaaS if user_id supplied.
        from_address: Optional[str] = body.from_address
        if has_user:
            waas = getattr(node, "_coinbase_waas_client", None)
            if waas is None:
                raise HTTPException(
                    status_code=503,
                    detail="WaaS client not initialized.",
                )
            record = waas.get_wallet(body.from_user_id)
            if record is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"no WaaS wallet for from_user_id="
                        f"{body.from_user_id!r}"
                    ),
                )
            from_address = record.address
            if not from_address:
                return {
                    "status": "PENDING_COMMISSION",
                    "note": (
                        "WaaS wallet has no address yet — provision "
                        "first (POST /wallet/waas/provision)."
                    ),
                }

        # Get a quote — this surfaces POOL_NOT_CONFIGURED /
        # POOL_UNAVAILABLE statuses cleanly.
        pool = getattr(node, "_aerodrome_client", None)
        if pool is None:
            raise HTTPException(
                status_code=503,
                detail="Aerodrome client not initialized.",
            )
        if not pool.is_configured():
            return {
                "status": "POOL_NOT_CONFIGURED",
                "from_address": from_address,
                "note": (
                    "Aerodrome pool not yet seeded; set "
                    "AERODROME_USDC_FTNS_POOL_ADDRESS after the "
                    "Foundation Safe ceremony (Vision 2026-06-15)."
                ),
            }
        # Reuse the quote endpoint logic for envelope assembly.
        quote_body = SwapQuoteRequest(
            amount_in=body.amount_in,
            token_in=body.token_in,
            slippage_bps=body.slippage_bps,
        )
        quote_resp = await post_swap_quote(quote_body)
        if quote_resp.get("status") != "OK":
            quote_resp["from_address"] = from_address
            return quote_resp

        # Build the CDP-WaaS-signable envelope.
        # 24-hour deadline by default — conservative; CDP submit
        # latency is sub-second so 1h is plenty but 24h gives
        # operators slack for batched flows.
        import time as _time
        deadline = int(_time.time()) + 86_400
        # Aerodrome Router v2 (Base mainnet, canonical).
        # Documented at github.com/aerodrome-finance/contracts.
        router_address = (
            "0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43"
        )
        return {
            "status": "READY_FOR_SUBMISSION",
            "from_address": from_address,
            "router_address": router_address,
            "function": "swapExactTokensForTokens",
            "args": {
                "amountIn": quote_resp["amount_in_units"],
                "amountOutMin": quote_resp["amount_out_min_units"],
                "routes": [
                    {
                        "from_token": quote_resp["token_in"],
                        "to_token": quote_resp["token_out"],
                        "stable": False,
                        "factory": (
                            # Aerodrome pool factory v2.
                            "0x420DD381b31aEf6683db6B902084cB0FFECe40Da"
                        ),
                    },
                ],
                "to": from_address,
                "deadline": deadline,
            },
            "quote": quote_resp,
            "note": (
                "Submission via CDP-signed transaction is sp856 "
                "(not yet wired). This envelope is ready to sign + "
                "submit — operator can use it via the CDP SDK "
                "directly until sp856 ships."
            ),
        }

    @app.get("/wallet/spend")
    async def get_wallet_spend(
        days: int = 30,
        address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sum FTNS spent on completed compute jobs over the last
        N days. RELEASED escrows count; REFUNDED + PENDING do not.

        Backs the ``prsm_spend_summary`` MCP tool.
        """
        if days <= 0 or days > 365:
            raise HTTPException(
                status_code=422,
                detail=f"days must be in [1, 365], got {days}",
            )

        escrow_svc = getattr(node, "_payment_escrow", None)
        ftns_ledger = getattr(node, "ftns_ledger", None)
        if escrow_svc is None or ftns_ledger is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "PaymentEscrow or ftns_ledger not initialized."
                ),
            )
        target = address or getattr(
            ftns_ledger, "_connected_address", None,
        )
        if not target:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No connected address; pass ?address=0x... explicitly."
                ),
            )

        try:
            entries = escrow_svc.list_escrows_by_requester(
                target, pending_only=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("list_escrows_by_requester raised: %s", exc)
            raise HTTPException(
                status_code=502,
                detail=f"escrow listing failed: {exc}",
            )

        from prsm.node.payment_escrow import EscrowStatus as _ES
        cutoff = _time_for_history.time() - days * 86400.0
        total = 0.0
        count = 0
        for e in entries:
            if e.status != _ES.RELEASED:
                continue
            if e.completed_at is None or e.completed_at < cutoff:
                continue
            total += e.amount
            count += 1

        return {
            "address": target,
            "days": days,
            "total_spent_ftns": total,
            "escrows_count": count,
        }

    @app.get("/wallet/escrows/{escrow_id}")
    async def get_wallet_escrow_detail(escrow_id: str) -> Dict[str, Any]:
        """Direct-lookup detail view of a single escrow by
        escrow_id. Operators investigating a specific escrow
        from logs / tx receipts use this rather than scanning
        the list view.

        Status:
          503 — PaymentEscrow not wired
          404 — escrow_id unknown
          200 — full record (escrow_id, job_id, amount_ftns,
                status, provider_winner, tx_lock, tx_release,
                created_at, completed_at, metadata)
        """
        escrow_svc = getattr(node, "_payment_escrow", None)
        if escrow_svc is None:
            raise HTTPException(
                status_code=503,
                detail="PaymentEscrow not initialized on this node.",
            )
        entry = escrow_svc.get_by_escrow_id(escrow_id)
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail=f"No escrow record for escrow_id={escrow_id!r}",
            )
        return {
            "escrow_id": entry.escrow_id,
            "job_id": entry.job_id,
            "requester_id": entry.requester_id,
            "amount_ftns": entry.amount,
            "status": entry.status.value,
            "provider_winner": entry.provider_winner,
            "tx_lock": entry.tx_lock,
            "tx_release": entry.tx_release,
            "created_at": entry.created_at,
            "completed_at": entry.completed_at,
            "metadata": dict(entry.metadata or {}),
        }

    @app.get("/wallet/escrows")
    async def get_wallet_escrows(
        address: Optional[str] = None,
        include_terminal: bool = False,
    ) -> Dict[str, Any]:
        """List active escrows for the operator's wallet (or any
        address via override). Backs the ``prsm_escrow_summary``
        MCP tool.

        Returns 503 if PaymentEscrow or ftns_ledger isn't wired,
        200 with `{escrows: [...], total: N, total_locked_ftns: X,
        address: 0x...}`. Default `pending_only=true`; pass
        `?include_terminal=true` for RELEASED + REFUNDED audit
        view.
        """
        escrow_svc = getattr(node, "_payment_escrow", None)
        ftns_ledger = getattr(node, "ftns_ledger", None)
        if escrow_svc is None or ftns_ledger is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "PaymentEscrow or ftns_ledger not initialized "
                    "on this node."
                ),
            )

        target = address or getattr(
            ftns_ledger, "_connected_address", None,
        )
        if not target:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No connected address available; pass "
                    "?address=0x... explicitly."
                ),
            )

        try:
            entries = escrow_svc.list_escrows_by_requester(
                target, pending_only=not include_terminal,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "list_escrows_by_requester raised: %s", exc,
            )
            raise HTTPException(
                status_code=502,
                detail=f"escrow listing failed: {exc}",
            )

        from prsm.node.payment_escrow import EscrowStatus as _EscStatus
        escrows_dicts = []
        total_locked = 0.0
        for e in entries:
            escrows_dicts.append({
                "escrow_id": e.escrow_id,
                "job_id": e.job_id,
                "amount_ftns": e.amount,
                "status": e.status.value,
                "provider_winner": e.provider_winner,
                "tx_lock": e.tx_lock,
                "tx_release": e.tx_release,
                "created_at": e.created_at,
                "completed_at": e.completed_at,
            })
            if e.status == _EscStatus.PENDING:
                total_locked += e.amount

        return {
            "address": target,
            "escrows": escrows_dicts,
            "total": len(escrows_dicts),
            "total_locked_ftns": total_locked,
            "include_terminal": include_terminal,
        }

    # ── Sprint 276 — Coinbase Wallet-as-a-Service (WaaS) ──
    # Operator-facing CRUD surface for embedded MPC wallets
    # via Coinbase CDP. Per Vision §14 "Crypto-UX adoption
    # barrier" mitigation: makes wallet provisioning
    # invisible to end users — email in, address out, no
    # seed phrase. PENDING_COMMISSION pattern: returns
    # preview records when CDP keys are absent.

    class _WaasProvisionRequest(BaseModel):
        user_id: str
        email: str

    @app.post("/wallet/waas/provision", tags=["wallet"])
    async def post_waas_provision(
        body: _WaasProvisionRequest,
    ) -> Dict[str, Any]:
        client = getattr(node, "_coinbase_waas_client", None)
        if client is None:
            raise HTTPException(
                status_code=503,
                detail="WaaS client not initialized.",
            )
        try:
            record = client.provision_wallet(
                user_id=body.user_id, email=body.email,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return record.to_dict()

    @app.get("/wallet/waas/status", tags=["wallet"])
    async def get_waas_status() -> Dict[str, Any]:
        client = getattr(node, "_coinbase_waas_client", None)
        if client is None:
            raise HTTPException(
                status_code=503,
                detail="WaaS client not initialized.",
            )
        return {
            "commissioned": client.is_commissioned(),
            "adapter_wired": client.adapter_wired(),
            "network": getattr(
                client, "_network", "base-mainnet",
            ),
            "wallet_count": len(client.list_wallets()),
        }

    # Sp862 — live Base mainnet balance reader for WaaS wallets.
    @app.get("/wallet/balance/{user_id}", tags=["wallet"])
    async def get_wallet_balance_by_user(
        user_id: str,
    ) -> Dict[str, Any]:
        """Live USDC + FTNS + native ETH balances for a WaaS user.

        Resolves user_id → WaaS address, then reads live Base
        mainnet balances via JSON-RPC. Uses BASE_RPC_URL when set
        or the free public Base RPC otherwise.
        """
        waas = getattr(node, "_coinbase_waas_client", None)
        if waas is None:
            raise HTTPException(
                status_code=503,
                detail="WaaS client not initialized.",
            )
        record = waas.get_wallet(user_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"no wallet for user_id={user_id!r}",
            )
        if not record.address:
            return {
                "user_id": user_id,
                "address": None,
                "status": "PENDING_COMMISSION",
                "note": (
                    "WaaS wallet has no address yet — provision "
                    "first."
                ),
            }
        from prsm.economy.web3.wallet_balance_reader import (
            from_env as _wbr_from_env,
        )
        reader = _wbr_from_env()
        try:
            balances = reader.get_balances(record.address)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "wallet_balance_reader failed for %s: %s",
                user_id, exc,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Base RPC balance read failed: {exc}",
            )
        finally:
            reader.close()
        out = balances.to_dict()
        out["user_id"] = user_id
        out["wallet_id"] = record.wallet_id
        out["network"] = record.network
        return out

    @app.get("/wallet/balance/by-address/{address}", tags=["wallet"])
    async def get_wallet_balance_by_address(
        address: str,
    ) -> Dict[str, Any]:
        """Live balances for an explicit address (not bound to a
        WaaS user). Useful for operator-side observability of
        non-WaaS wallets like Foundation Safe."""
        if not address.startswith("0x") or len(address) != 42:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"address must be 0x + 40 hex chars (42 total)"
                    f", got {address!r}"
                ),
            )
        from prsm.economy.web3.wallet_balance_reader import (
            from_env as _wbr_from_env,
        )
        reader = _wbr_from_env()
        try:
            balances = reader.get_balances(address)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "wallet_balance_reader failed for %s: %s",
                address, exc,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Base RPC balance read failed: {exc}",
            )
        finally:
            reader.close()
        return balances.to_dict()

    # Sp857 — onramp conversion funnel surfaces.
    @app.get("/wallet/onramp/funnel", tags=["wallet"])
    async def get_onramp_funnel(
        status: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Funnel summary + intent list. ``status`` filters by
        INTENT_RECORDED / PENDING_SETTLEMENT / CONFIRMED / EXPIRED."""
        if limit <= 0 or limit > 10_000:
            raise HTTPException(
                status_code=422,
                detail=f"limit must be in [1, 10000], got {limit}",
            )
        from prsm.economy.web3.onramp_funnel import OnrampFunnel
        funnel = getattr(node, "_onramp_funnel", None)
        if funnel is None:
            funnel = OnrampFunnel()
            setattr(node, "_onramp_funnel", funnel)
        intents = funnel.list_intents(status=status)
        return {
            "summary": funnel.summary(),
            "intents": [
                i.to_dict() for i in intents[:limit]
            ],
            "limit": limit,
            "filter_status": status,
        }

    @app.post("/wallet/onramp/sweep", tags=["wallet"])
    async def post_onramp_sweep() -> Dict[str, Any]:
        """Trigger an on-chain balance sweep across open intents.
        Transitions PENDING_SETTLEMENT → CONFIRMED when USDC
        arrives, and stale intents → EXPIRED after 24h.

        Sp871: on CONFIRMED transition, the onramp→swap
        orchestrator immediately builds an Aerodrome USDC→FTNS
        swap envelope and attaches it to the intent. User
        retrieves via GET /wallet/onramp/funnel/{intent_id}.
        Fail-soft when Aerodrome pool isn't yet seeded — envelope
        stays None, next sweep retries.
        """
        from prsm.economy.web3.onramp_funnel import OnrampFunnel
        from prsm.economy.web3.wallet_balance_reader import (
            from_env as _wbr_from_env,
        )
        from prsm.economy.web3.onramp_to_swap_orchestrator import (
            make_on_confirmed_callback,
            make_on_expired_callback,
        )
        from prsm.config.networks import get_network_config
        funnel = getattr(node, "_onramp_funnel", None)
        if funnel is None:
            funnel = OnrampFunnel()
            setattr(node, "_onramp_funnel", funnel)
        reader = _wbr_from_env()
        # Sp871 — wire the auto-orchestrator if Aerodrome client
        # is available. Callback fires per CONFIRMED intent.
        # Sp874 — also wire the outbound completion notifier
        # (no-op when PRSM_ONRAMP_COMPLETION_WEBHOOK_URL unset).
        from prsm.economy.web3.onramp_completion_notifier import (
            from_env as _notifier_from_env,
        )
        aero = getattr(node, "_aerodrome_client", None)
        net = get_network_config("mainnet")
        notifier = _notifier_from_env()
        # Sp885 — always build the callback (NOT gated on aero):
        # the swap-envelope build needs Aerodrome (None-safe inside
        # build_envelope_for_intent — deferred when unwired), but
        # the compliance-ring record (settled volume → tier limit)
        # and the completion webhook MUST fire on every CONFIRMED
        # regardless of pool status.
        on_confirmed = make_on_confirmed_callback(
            funnel=funnel,
            aerodrome_client=aero,
            ftns_address=net.ftns_token or "",
            completion_notifier=notifier,
            compliance_ring=getattr(
                node, "_fiat_compliance_ring", None,
            ),
        )
        # Sp1176 — write a terminal EXPIRED $0 compliance entry when an
        # intent is abandoned, so the AML rolling total stops counting
        # the sp968 PENDING reservation and the audit trail is complete.
        on_expired = make_on_expired_callback(
            compliance_ring=getattr(
                node, "_fiat_compliance_ring", None,
            ),
        )
        try:
            return funnel.sweep(
                balance_reader=reader,
                on_confirmed=on_confirmed,
                on_expired=on_expired,
            )
        finally:
            reader.close()
            notifier.close()

    @app.get(
        "/wallet/onramp/notifications", tags=["wallet"],
    )
    async def get_onramp_notification_deliveries(
        limit: int = 100,
    ) -> Dict[str, Any]:
        """List outbound completion-webhook delivery attempts.

        Persistent cross-restart audit trail of sp874 dispatches.
        Each entry records timestamp, intent_id, target URL,
        status_code, success bool, error message (if any),
        signature_attached bool.
        """
        if limit <= 0 or limit > 10_000:
            raise HTTPException(
                status_code=422,
                detail=f"limit must be in [1,10000], got {limit}",
            )
        from prsm.economy.web3.onramp_completion_notifier import (
            from_env as _notifier_from_env,
        )
        notifier = _notifier_from_env()
        try:
            records = notifier.list_deliveries(limit=limit)
        finally:
            notifier.close()
        successes = sum(
            1 for r in records if r.get("success")
        )
        return {
            "count": len(records),
            "success_count": successes,
            "failure_count": len(records) - successes,
            "configured": notifier.is_configured(),
            "deliveries": records,
        }

    @app.get(
        "/wallet/onramp/funnel/{intent_id}", tags=["wallet"],
    )
    async def get_onramp_intent(intent_id: str) -> Dict[str, Any]:
        from prsm.economy.web3.onramp_funnel import OnrampFunnel
        funnel = getattr(node, "_onramp_funnel", None)
        if funnel is None:
            funnel = OnrampFunnel()
            setattr(node, "_onramp_funnel", funnel)
        rec = funnel.get_intent(intent_id)
        if rec is None:
            raise HTTPException(
                status_code=404,
                detail=f"no intent for id={intent_id!r}",
            )
        return rec.to_dict()

    # Sp864 — fleet treasury aggregator: roll up balances across
    # every WaaS wallet.
    @app.get("/wallet/treasury", tags=["wallet"])
    async def get_wallet_treasury(
        max_wallets: int = 100,
    ) -> Dict[str, Any]:
        """Fleet-wide treasury view: aggregated USDC + FTNS + ETH
        across all PROVISIONED WaaS wallets, with per-wallet
        breakdown. ``max_wallets`` caps the per-call RPC load."""
        if max_wallets <= 0 or max_wallets > 10_000:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"max_wallets must be in [1, 10000], "
                    f"got {max_wallets}"
                ),
            )
        waas = getattr(node, "_coinbase_waas_client", None)
        from prsm.economy.web3.wallet_balance_reader import (
            from_env as _wbr_from_env,
        )
        from prsm.economy.web3.treasury_aggregator import (
            aggregate_treasury,
        )
        reader = _wbr_from_env()
        try:
            return aggregate_treasury(
                waas_client=waas,
                balance_reader=reader,
                max_wallets=max_wallets,
            )
        finally:
            reader.close()

    # Sp859 — Phase 5 readiness aggregator endpoint.
    @app.get("/wallet/phase5/status", tags=["wallet"])
    async def get_phase5_status() -> Dict[str, Any]:
        """One-shot Phase 5 readiness grid: KYC + WaaS + Paymaster
        + Onramp + Aerodrome status in a single canonical envelope.
        Powers operator dashboards + the `prsm node phase5-status`
        CLI surface."""
        from prsm.economy.web3.phase5_status import (
            aggregate_phase5_status,
        )
        return aggregate_phase5_status(
            kyc_client=getattr(node, "_kyc_client", None),
            waas_client=getattr(
                node, "_coinbase_waas_client", None,
            ),
            paymaster_client=getattr(
                node, "_paymaster_client", None,
            ),
            aerodrome_client=getattr(
                node, "_aerodrome_client", None,
            ),
        )

    @app.get("/wallet/waas", tags=["wallet"])
    async def list_waas_wallets(
        limit: int = 100,
    ) -> Dict[str, Any]:
        if limit <= 0 or limit > 10000:
            raise HTTPException(
                status_code=422,
                detail=f"limit must be in [1, 10000], got {limit}",
            )
        client = getattr(node, "_coinbase_waas_client", None)
        if client is None:
            raise HTTPException(
                status_code=503,
                detail="WaaS client not initialized.",
            )
        wallets = client.list_wallets()
        # Sort newest-first for operator UX.
        wallets.sort(key=lambda r: r.created_at, reverse=True)
        return {
            "wallets": [r.to_dict() for r in wallets[:limit]],
            "count": len(wallets),
            "limit": limit,
        }

    @app.get("/wallet/waas/{user_id}", tags=["wallet"])
    async def get_waas_wallet(user_id: str) -> Dict[str, Any]:
        client = getattr(node, "_coinbase_waas_client", None)
        if client is None:
            raise HTTPException(
                status_code=503,
                detail="WaaS client not initialized.",
            )
        record = client.get_wallet(user_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"no wallet for user_id={user_id!r}",
            )
        return record.to_dict()

    # ── Sprint 277 — Gasless FTNS transfer via paymaster ──
    # Per Vision §14 "Crypto-UX adoption barrier" mitigation:
    # users should never need to hold gas tokens. Composes a
    # UserOperation from a WaaS-managed sender and routes it
    # through the paymaster for sponsored submission.
    # dry_run=True (default) → estimate-only; False → submit.

    class _GaslessTransferRequest(BaseModel):
        from_user_id: str
        to_address: str
        ftns_amount: str  # decimal string to preserve precision
        dry_run: bool = True

    @app.post("/wallet/transfer/gasless", tags=["wallet"])
    async def post_gasless_transfer(
        body: _GaslessTransferRequest,
    ) -> Dict[str, Any]:
        waas = getattr(node, "_coinbase_waas_client", None)
        if waas is None:
            raise HTTPException(
                status_code=503,
                detail="WaaS client not initialized.",
            )
        paymaster = getattr(node, "_paymaster_client", None)
        if paymaster is None:
            raise HTTPException(
                status_code=503,
                detail="Paymaster client not initialized.",
            )
        # Validate amount
        try:
            amount_value = float(body.ftns_amount)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"ftns_amount must be a positive decimal, "
                    f"got {body.ftns_amount!r}"
                ),
            )
        if amount_value <= 0:
            raise HTTPException(
                status_code=422,
                detail="ftns_amount must be > 0",
            )
        if not body.to_address:
            raise HTTPException(
                status_code=422,
                detail="to_address must be non-empty",
            )

        sender = waas.get_wallet(body.from_user_id)
        if sender is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"no WaaS wallet for from_user_id="
                    f"{body.from_user_id!r}; run /wallet/waas/"
                    f"provision first"
                ),
            )
        # If the sender wallet is itself pre-commission, surface
        # that condition explicitly.
        if sender.status != "PROVISIONED" or not sender.address:
            return {
                "status": "PENDING_COMMISSION",
                "from_user_id": body.from_user_id,
                "to_address": body.to_address,
                "ftns_amount": body.ftns_amount,
                "tx_hash": None,
                "sponsor_amount_wei": None,
                "gas_estimate_wei": None,
                "note": (
                    "Sender wallet has status="
                    f"{sender.status} — provision must complete "
                    "before transfers."
                ),
            }

        user_op = {
            "sender": sender.address,
            "to": body.to_address,
            "ftns_amount": body.ftns_amount,
            "kind": "ftns_transfer",
        }
        result = paymaster.sponsor_user_op(
            user_op, dry_run=body.dry_run,
        )

        # Sprint 282 — record to fiat compliance audit ring.
        # dry_run → gasless_transfer_quote; execute →
        # gasless_transfer_execute.
        try:
            _ftns_amount_float = float(body.ftns_amount)
        except (ValueError, TypeError):
            _ftns_amount_float = 0.0
        _record_fiat_compliance(
            kind=(
                "gasless_transfer_execute"
                if not body.dry_run
                else "gasless_transfer_quote"
            ),
            user_id=body.from_user_id,
            usd_amount=0.0,
            ftns_amount=_ftns_amount_float,
            status=result.status,
            tx_hash=result.tx_hash,
            address=sender.address,
            metadata={
                "to_address": body.to_address,
                "sender_address": sender.address,
            },
        )

        out = result.to_dict()
        out["from_user_id"] = body.from_user_id
        out["to_address"] = body.to_address
        out["ftns_amount"] = body.ftns_amount
        out["sender_address"] = sender.address
        return out

    class _OnChainTransferRequest(BaseModel):
        to_address: str
        amount_ftns: float

    @app.post("/wallet/transfer/onchain", tags=["wallet"])
    async def post_onchain_transfer(
        body: _OnChainTransferRequest,
    ) -> Dict[str, Any]:
        ledger = getattr(node, "ftns_ledger", None)
        if ledger is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "On-chain FTNS ledger not initialized — "
                    "daemon must be started with "
                    "FTNS_WALLET_PRIVATE_KEY set."
                ),
            )
        if not body.to_address:
            raise HTTPException(
                status_code=422,
                detail="to_address must be non-empty",
            )
        if body.amount_ftns <= 0:
            raise HTTPException(
                status_code=422,
                detail="amount_ftns must be > 0",
            )
        import uuid
        job_id = f"manual-{uuid.uuid4().hex[:12]}"
        tx_record = await ledger.transfer(
            job_id=job_id,
            to_address=body.to_address,
            amount_ftns=body.amount_ftns,
        )
        if tx_record is None:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Ledger transfer returned None — wallet "
                    "may not be configured or amount invalid."
                ),
            )
        return {
            "tx_hash": getattr(tx_record, "tx_hash", None),
            "status": getattr(tx_record, "status", None),
            "block_number": getattr(
                tx_record, "block_number", None,
            ),
            "from_address": getattr(
                tx_record, "from_addr", None,
            ),
            "to_address": getattr(tx_record, "to_addr", None),
            "amount_ftns": getattr(
                tx_record, "amount_ftns", None,
            ),
            "job_id": job_id,
        }

    @app.get("/wallet/transactions/onchain", tags=["wallet"])
    async def get_onchain_transactions() -> Dict[str, Any]:
        ledger = getattr(node, "ftns_ledger", None)
        if ledger is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "On-chain FTNS ledger not initialized — "
                    "daemon must be started with "
                    "FTNS_WALLET_PRIVATE_KEY set."
                ),
            )
        txs = getattr(ledger, "_transactions", []) or []
        out = []
        for tx in txs:
            out.append({
                "tx_hash": getattr(tx, "tx_hash", ""),
                "status": getattr(tx, "status", None),
                "block_number": getattr(
                    tx, "block_number", None,
                ),
                "from_address": getattr(tx, "from_addr", None),
                "to_address": getattr(tx, "to_addr", None),
                "amount_ftns": getattr(tx, "amount_ftns", None),
                "created_at": getattr(tx, "created_at", None),
                "job_id": getattr(tx, "job_id", None),
            })
        if getattr(ledger, "is_persistent", False):
            scope = (
                f"persistent (sqlite: "
                f"{getattr(ledger, 'db_path', '?')})"
            )
        else:
            scope = (
                "in-memory (resets on daemon restart) — "
                "set OnChainFTNSLedger(db_path=…) to persist"
            )
        return {
            "count": len(out),
            "connected_address": getattr(
                ledger, "_connected_address", None,
            ),
            "scope": scope,
            "transactions": out,
        }

    @app.get("/wallet/transactions/onchain/inbound", tags=["wallet"])
    async def get_inbound_transactions(
        from_block: int = 0,
        to_block: str = "latest",
        lookback_blocks: int = 9000,
    ) -> Dict[str, Any]:
        """Sprint 512 / 542: scan inbound ERC-20 Transfer events.

        Query params:
          from_block (int): explicit start block
          to_block (str/int): "latest" or block number
          lookback_blocks (int): if from_block=0, scan
            current_block - lookback_blocks → current_block.
            Default 9_000 fits in a single public-RPC call
            (Base mainnet.base.org caps eth_getLogs near 10k
            blocks per request). For wider history, pass an
            explicit from_block — the server splits the range
            into 9k-block sub-windows automatically.
        """
        ledger = getattr(node, "ftns_ledger", None)
        if ledger is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "On-chain FTNS ledger not initialized — "
                    "daemon must be started with "
                    "FTNS_WALLET_PRIVATE_KEY set."
                ),
            )
        w3 = getattr(ledger, "w3", None)
        addr = getattr(ledger, "_connected_address", None)
        token = getattr(ledger, "_token", None)
        if w3 is None or addr is None or token is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Ledger not fully initialized — w3/token/"
                    "address missing."
                ),
            )
        try:
            from prsm.economy.ftns_onchain import (
                scan_inbound_transfers_chunked,
            )
            if from_block == 0:
                latest = w3.eth.block_number
                # sp932 — inclusive range, so subtract (lookback_blocks - 1) to
                # scan EXACTLY lookback_blocks. The prior `latest - lookback_blocks`
                # spanned lookback_blocks+1 blocks, which (at the 9000 default)
                # exceeded scan_inbound_transfers_chunked's 9000-block window and
                # forced a spurious extra boundary chunk.
                start = max(0, latest - lookback_blocks + 1)
                end = latest
            else:
                start = from_block
                end = (
                    w3.eth.block_number
                    if to_block == "latest"
                    else int(to_block)
                )
            transfers = scan_inbound_transfers_chunked(
                token,
                recipient=addr,
                from_block=start,
                to_block=end,
            )
            return {
                "recipient": addr,
                "from_block": start,
                "to_block": end,
                "count": len(transfers),
                "transfers": transfers,
            }
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=f"inbound scan failed: {exc!s}"[:300],
            )

    @app.get("/wallet/transactions/onchain/inbound/stats", tags=["wallet"])
    async def get_inbound_transaction_stats(
        from_block: int = 0,
        to_block: str = "latest",
        lookback_blocks: int = 9000,
    ) -> Dict[str, Any]:
        """Sprint 515 / 542 — aggregate inbound stats. Default
        lookback 9_000 fits in one public-RPC call; wider ranges
        are auto-chunked server-side."""
        ledger = getattr(node, "ftns_ledger", None)
        if ledger is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "On-chain FTNS ledger not initialized — "
                    "daemon must be started with "
                    "FTNS_WALLET_PRIVATE_KEY set."
                ),
            )
        w3 = getattr(ledger, "w3", None)
        addr = getattr(ledger, "_connected_address", None)
        token = getattr(ledger, "_token", None)
        if w3 is None or addr is None or token is None:
            raise HTTPException(
                status_code=503,
                detail="Ledger not fully initialized.",
            )
        try:
            from prsm.economy.ftns_onchain import (
                scan_inbound_transfers_chunked,
            )
            if from_block == 0:
                latest = w3.eth.block_number
                # sp932 — inclusive range, so subtract (lookback_blocks - 1) to
                # scan EXACTLY lookback_blocks. The prior `latest - lookback_blocks`
                # spanned lookback_blocks+1 blocks, which (at the 9000 default)
                # exceeded scan_inbound_transfers_chunked's 9000-block window and
                # forced a spurious extra boundary chunk.
                start = max(0, latest - lookback_blocks + 1)
                end = latest
            else:
                start = from_block
                end = (
                    w3.eth.block_number
                    if to_block == "latest"
                    else int(to_block)
                )
            transfers = scan_inbound_transfers_chunked(
                token, recipient=addr,
                from_block=start, to_block=end,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=f"inbound scan failed: {exc!s}"[:300],
            )
        total_ftns = sum(
            t.get("amount_ftns", 0.0) for t in transfers
        )
        blocks = [
            t.get("block_number")
            for t in transfers
            if t.get("block_number") is not None
        ]
        return {
            "recipient": addr,
            "from_block": start,
            "to_block": end,
            "count": len(transfers),
            "total_inbound_ftns": total_ftns,
            "first_inbound_block": (
                min(blocks) if blocks else None
            ),
            "last_inbound_block": (
                max(blocks) if blocks else None
            ),
        }

    @app.get("/wallet/transactions/onchain/stats", tags=["wallet"])
    async def get_onchain_transaction_stats() -> Dict[str, Any]:
        ledger = getattr(node, "ftns_ledger", None)
        if ledger is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "On-chain FTNS ledger not initialized — "
                    "daemon must be started with "
                    "FTNS_WALLET_PRIVATE_KEY set."
                ),
            )
        txs = getattr(ledger, "_transactions", []) or []
        confirmed_count = 0
        pending_count = 0
        rejected_count = 0
        total_sent = 0.0
        timestamps = []
        for t in txs:
            status = getattr(t, "status", None)
            amt = getattr(t, "amount_ftns", 0) or 0
            ts = getattr(t, "created_at", None)
            if ts is not None:
                timestamps.append(ts)
            if status == "confirmed":
                confirmed_count += 1
                total_sent += amt
            elif status == "pending":
                pending_count += 1
            elif status == "rejected":
                rejected_count += 1
        if getattr(ledger, "is_persistent", False):
            scope = (
                f"persistent (sqlite: "
                f"{getattr(ledger, 'db_path', '?')})"
            )
        else:
            scope = "in-memory (resets on daemon restart)"
        return {
            "address": getattr(
                ledger, "_connected_address", None,
            ),
            "total_count": len(txs),
            "confirmed_count": confirmed_count,
            "pending_count": pending_count,
            "rejected_count": rejected_count,
            "total_ftns_sent": total_sent,
            "first_tx_at": min(timestamps) if timestamps else None,
            "last_tx_at": max(timestamps) if timestamps else None,
            "scope": scope,
        }

    @app.get("/wallet/gas-status", tags=["wallet"])
    async def get_gas_status() -> Dict[str, Any]:
        ledger = getattr(node, "ftns_ledger", None)
        if ledger is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "On-chain FTNS ledger not initialized — "
                    "daemon must be started with "
                    "FTNS_WALLET_PRIVATE_KEY set."
                ),
            )
        # Base mainnet at ~0.0072 Gwei × 60k gas/TX ≈ 4.3e-7 ETH/TX
        # low      < 0.0005 ETH (~1150 TX runway)
        # critical < 0.0001 ETH (~230 TX runway)
        LOW = 0.0005
        CRITICAL = 0.0001
        addr = getattr(ledger, "_connected_address", None)
        w3 = getattr(ledger, "w3", None)
        if w3 is None or addr is None:
            return {
                "address": addr,
                "eth_balance_wei": None,
                "eth_balance": None,
                "low_threshold_eth": LOW,
                "critical_threshold_eth": CRITICAL,
                "status": "unavailable",
            }
        try:
            wei = w3.eth.get_balance(addr)
        except Exception as exc:
            return {
                "address": addr,
                "eth_balance_wei": None,
                "eth_balance": None,
                "low_threshold_eth": LOW,
                "critical_threshold_eth": CRITICAL,
                "status": "unavailable",
                "error": str(exc)[:200],
            }
        eth = wei / 1e18
        if eth < CRITICAL:
            status = "critical"
        elif eth < LOW:
            status = "low"
        else:
            status = "ok"
        return {
            "address": addr,
            "eth_balance_wei": wei,
            "eth_balance": eth,
            "low_threshold_eth": LOW,
            "critical_threshold_eth": CRITICAL,
            "status": status,
        }

    @app.get("/wallet/paymaster/status", tags=["wallet"])
    async def get_paymaster_status() -> Dict[str, Any]:
        paymaster = getattr(node, "_paymaster_client", None)
        if paymaster is None:
            raise HTTPException(
                status_code=503,
                detail="Paymaster client not initialized.",
            )
        return paymaster.spend_summary()

    # ── Sprint 279 — Aerodrome read-only pool quoter ──────
    # Operator-side surface for the AerodromeClient. Real
    # production code (no commission gate). Distinguishes
    # NOT_CONFIGURED (pool address not yet pasted into env;
    # seeding ceremony pending) from POOL_UNAVAILABLE (RPC
    # error tried-and-failed) — both fail-soft, 200.

    @app.get("/wallet/pool/state", tags=["wallet"])
    async def get_pool_state() -> Dict[str, Any]:
        pool = getattr(node, "_aerodrome_client", None)
        if pool is None:
            raise HTTPException(
                status_code=503,
                detail="Aerodrome client not initialized.",
            )
        if not pool.is_configured():
            return {
                "status": "NOT_CONFIGURED",
                "note": (
                    "Set BASE_RPC_URL + "
                    "AERODROME_USDC_FTNS_POOL_ADDRESS env "
                    "vars after the seeding ceremony "
                    "(Vision gantt 2026-06-15)."
                ),
            }
        state = pool.get_pool_state()
        if state is None:
            return {
                "status": "POOL_UNAVAILABLE",
                "pool_address": pool.pool_address,
                "note": (
                    "Pool RPC call returned no state — "
                    "Base RPC may be unreachable or the "
                    "pool contract may not exist at the "
                    "configured address yet."
                ),
            }
        out = state.to_dict()
        out["status"] = "OK"
        return out

    @app.get("/wallet/pool/quote", tags=["wallet"])
    async def get_pool_quote(
        amount_in: int,
        token_in: str,
    ) -> Dict[str, Any]:
        from prsm.economy.web3.aerodrome_client import (
            AerodromeQuoteError,
        )
        if amount_in <= 0:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"amount_in must be > 0, got {amount_in}"
                ),
            )
        if not token_in:
            raise HTTPException(
                status_code=422,
                detail="token_in is required",
            )
        pool = getattr(node, "_aerodrome_client", None)
        if pool is None:
            raise HTTPException(
                status_code=503,
                detail="Aerodrome client not initialized.",
            )
        if not pool.is_configured():
            return {
                "status": "NOT_CONFIGURED",
                "amount_in": amount_in,
                "token_in": token_in,
                "note": (
                    "Set BASE_RPC_URL + "
                    "AERODROME_USDC_FTNS_POOL_ADDRESS after "
                    "the seeding ceremony."
                ),
            }
        try:
            quote = pool.quote_swap(
                amount_in=amount_in, token_in=token_in,
            )
        except AerodromeQuoteError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        if quote is None:
            return {
                "status": "POOL_UNAVAILABLE",
                "amount_in": amount_in,
                "token_in": token_in,
            }
        out = quote.to_dict()
        out["status"] = "OK"
        return out

    # ── Sprint 286 — fiat-surface health check ────────────
    # Enumerates dangerous-combination env-var configs
    # (e.g., KYC commissioned without webhook secret) so
    # operators see safety issues before vendor traffic
    # arrives. Read-only; reads current process env.

    @app.get(
        "/admin/fiat-surface/health", tags=["admin"],
    )
    async def get_fiat_surface_health() -> Dict[str, Any]:
        from prsm.economy.web3.fiat_surface_health import (
            check_fiat_surface_health, FindingSeverity,
        )
        findings = check_fiat_surface_health(env=os.environ)
        error_count = sum(
            1 for f in findings
            if f.severity == FindingSeverity.ERROR
        )
        warn_count = sum(
            1 for f in findings
            if f.severity == FindingSeverity.WARN
        )
        info_count = sum(
            1 for f in findings
            if f.severity == FindingSeverity.INFO
        )
        if error_count > 0:
            overall = "ERROR"
        elif warn_count > 0:
            overall = "WARN"
        elif info_count > 0:
            overall = "INFO"
        else:
            overall = "OK"
        return {
            "overall": overall,
            "error_count": error_count,
            "warn_count": warn_count,
            "info_count": info_count,
            "findings": [f.to_dict() for f in findings],
        }

    # ── Sprint 285 — KYC-tier rolling-total enforcement ───
    # Per-tier USD/day limits. Defaults match FinCEN MSB +
    # vendor convention; tunable via env vars. Tier comes
    # from KYC record level (basic/enhanced); rolling total
    # comes from the sprint-282 FiatComplianceRing.

    _DEFAULT_TIER_LIMIT_USD = {
        "basic": 1000.0,
        "enhanced": 10000.0,
    }

    def _tier_limit_for_level(level: str) -> float:
        env_var = f"PRSM_KYC_TIER_LIMIT_{level.upper()}_USD"
        raw = os.environ.get(env_var)
        if raw:
            try:
                v = float(raw)
                if v > 0:
                    return v
            except (ValueError, TypeError):
                pass
        return _DEFAULT_TIER_LIMIT_USD.get(level, 1000.0)

    def _tier_check(
        user_id: Optional[str],
        requested_usd: float,
        kyc_status: Optional[str],
    ) -> Dict[str, Any]:
        """Compute tier_level, tier_limit_usd, remaining, and
        exceeded flag for the requested USD amount. Returns a
        neutral block when no user_id, no KYC client, or KYC
        not VERIFIED — those cases are gated elsewhere."""
        out = {
            "tier_level": None,
            "tier_limit_usd": 0.0,
            "tier_limit_remaining_usd": 0.0,
            "tier_limit_exceeded": False,
        }
        if not user_id:
            return out
        kyc = getattr(node, "_kyc_client", None)
        if kyc is None:
            return out
        rec = kyc.get_status(user_id)
        if rec is None or rec.status != "VERIFIED":
            return out
        level = rec.level
        limit_usd = _tier_limit_for_level(level)
        ring = getattr(node, "_fiat_compliance_ring", None)
        rolling = 0.0
        rolling_failed = False
        if ring is not None:
            try:
                rolling = float(
                    ring.total_usd_for_user(user_id),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "tier_check: total_usd_for_user raised: %s",
                    exc,
                )
                rolling = 0.0
                # Sp1179 — a read failure means the rolling total is
                # UNKNOWN. In shared_redis mode that must FAIL CLOSED
                # (deny) rather than treat 0.0 as "under the limit".
                rolling_failed = True
        remaining = max(0.0, limit_usd - rolling)
        exceeded = (rolling + float(requested_usd)) > limit_usd
        out["tier_level"] = level
        out["tier_limit_usd"] = limit_usd
        out["tier_limit_remaining_usd"] = remaining
        out["tier_limit_exceeded"] = exceeded
        # Sp1175 — the rolling total (total_usd_for_user) is read from
        # the PROCESS-LOCAL FiatComplianceRing (an in-memory deque per
        # node). On a multi-replica deployment each replica keeps its
        # OWN total, so a user whose requests fan across N replicas can
        # transact up to N× their AML tier limit — the limit is enforced
        # per-pod, NOT globally. `tier_limit_scope` makes that explicit
        # to every caller (and to an auditor reading the response).
        out["tier_limit_scope"] = "process_local"
        # Operators running fiat on multiple replicas WITHOUT a shared
        # real-time compliance backend can set
        # PRSM_FIAT_TIER_LIMIT_MODE=strict_shared to FAIL CLOSED: deny
        # the gated fiat surface rather than silently under-enforce the
        # cross-replica limit. Default (process_local) is correct for a
        # single-replica deployment and is a pure no-op here. The real
        # fix — a shared atomic rolling total (Redis/Postgres) — is a
        # follow-on requiring an infra decision (see the Vision
        # fiat-compliance note); this lever is the honest interim.
        mode = (
            os.environ.get(
                "PRSM_FIAT_TIER_LIMIT_MODE", "process_local",
            )
            .strip()
            .lower()
        )
        if mode == "strict_shared":
            out["tier_limit_exceeded"] = True
            out["tier_limit_strict_block"] = True
        elif mode == "shared_redis":
            # Sp1179 — the rolling total above came from the SHARED
            # cross-replica backend (the ring's _shared_total), so the
            # limit is enforced GLOBALLY: `exceeded` is already correct.
            # But if the shared store is unreachable (rolling_failed) OR
            # it wasn't wired (misconfig: mode set but no backend), we
            # CANNOT guarantee the limit → FAIL CLOSED, exactly like
            # strict_shared, rather than under-enforce off a 0.0 total.
            _shared = getattr(ring, "_shared_total", None) if ring else None
            if rolling_failed or _shared is None:
                out["tier_limit_exceeded"] = True
                out["tier_limit_strict_block"] = True
                out["tier_limit_scope"] = "shared_redis_unavailable"
            else:
                out["tier_limit_scope"] = "shared_redis"
        return out

    # ── Sprint 282 — Fiat compliance audit ring ───────────
    # Single queryable log across all fiat surfaces. Records
    # quotes + executes so operators have audit trail for
    # AUSTRAC / FinCEN / IRS reporting. Recording is best-
    # effort: ring exceptions are caught + logged so telemetry
    # failures never deny the primary fiat surface.

    def _record_fiat_compliance(**kwargs: Any) -> None:
        """Best-effort ring write. Swallows exceptions; logs
        warnings. Called from onramp/offramp/gasless handlers
        after they produce their artifact."""
        ring = getattr(node, "_fiat_compliance_ring", None)
        if ring is None:
            return
        try:
            ring.record(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "FiatComplianceRing.record failed: %s "
                "(kind=%s, user_id=%s)",
                exc, kwargs.get("kind"),
                kwargs.get("user_id"),
            )

    @app.get("/admin/fiat-compliance", tags=["admin"])
    async def list_fiat_compliance(
        limit: int = 100,
        offset: int = 0,
        kind: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if limit <= 0 or limit > 10000:
            raise HTTPException(
                status_code=422,
                detail=f"limit must be in [1, 10000], got {limit}",
            )
        if offset < 0:
            raise HTTPException(
                status_code=422,
                detail=f"offset must be >= 0, got {offset}",
            )
        ring = getattr(node, "_fiat_compliance_ring", None)
        if ring is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Fiat compliance ring not initialized."
                ),
            )
        try:
            entries = ring.recent(
                limit=limit, offset=offset,
                kind=kind, user_id=user_id,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return {
            "entries": [e.to_dict() for e in entries],
            "count": ring.count(),
            "limit": limit,
            "offset": offset,
        }

    @app.get(
        "/admin/fiat-compliance/summary", tags=["admin"],
    )
    async def get_fiat_compliance_summary() -> Dict[str, Any]:
        ring = getattr(node, "_fiat_compliance_ring", None)
        if ring is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Fiat compliance ring not initialized."
                ),
            )
        return {
            "by_kind": ring.summary_by_kind(),
            "total_entries": ring.count(),
        }

    # Sp872 — CSV export for AUSTRAC / FinCEN / IRS reporting.
    @app.get(
        "/admin/fiat-compliance/export.csv", tags=["admin"],
    )
    async def get_fiat_compliance_csv_export(
        since: Optional[float] = None,
        until: Optional[float] = None,
        user_id: Optional[str] = None,
        kind: Optional[str] = None,
        min_usd: Optional[float] = None,
    ):
        """Canonical CSV export of fiat compliance ring entries.

        Filters: since/until (Unix timestamp), user_id, kind,
        min_usd (FinCEN $10k CTR threshold typical). Returns
        text/csv with the canonical sp872 header schema.
        """
        from fastapi.responses import PlainTextResponse
        from prsm.economy.web3.compliance_csv_export import (
            export_to_csv,
        )
        ring = getattr(node, "_fiat_compliance_ring", None)
        if ring is None:
            raise HTTPException(
                status_code=503,
                detail="Fiat compliance ring not initialized.",
            )
        entries = ring.recent(limit=ring.count() or 1)
        kinds = {kind} if kind else None
        csv_text = export_to_csv(
            entries, since=since, until=until,
            user_id=user_id, kinds=kinds, min_usd=min_usd,
        )
        # Filename includes the time window if specified for
        # operator-side archive hygiene.
        from datetime import datetime, timezone
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = f"-since{int(since)}" if since else ""
        suffix += f"-until{int(until)}" if until else ""
        filename = f"prsm-compliance-{ts}{suffix}.csv"
        return PlainTextResponse(
            content=csv_text,
            media_type="text/csv",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{filename}"'
                ),
            },
        )

    @app.get(
        "/admin/fiat-compliance/{entry_id}", tags=["admin"],
    )
    async def get_fiat_compliance_entry(
        entry_id: str,
    ) -> Dict[str, Any]:
        ring = getattr(node, "_fiat_compliance_ring", None)
        if ring is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Fiat compliance ring not initialized."
                ),
            )
        entry = ring.get(entry_id)
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail=f"no entry with id={entry_id!r}",
            )
        return entry.to_dict()

    # ── Sprint 280 — KYC vendor adapter endpoints ─────────
    # Pluggable KYC surface (Persona / Onfido / Plaid Identity
    # via dependency-injected backend). PENDING_COMMISSION
    # pattern preserved when no vendor + API key configured.
    # Webhook endpoint accepts vendor-name in the URL for
    # routing — the path's vendor segment is informational
    # only in v1; the configured vendor is the authority.

    class _KYCInitiateRequest(BaseModel):
        user_id: str
        email: str
        level: str = "basic"

    class _KYCWebhookRequest(BaseModel):
        user_id: str
        status: str
        vendor_ref: Optional[str] = None

    @app.post("/wallet/kyc/initiate", tags=["wallet"])
    async def post_kyc_initiate(
        body: _KYCInitiateRequest,
    ) -> Dict[str, Any]:
        kyc = getattr(node, "_kyc_client", None)
        if kyc is None:
            raise HTTPException(
                status_code=503,
                detail="KYC client not initialized.",
            )
        try:
            record = kyc.initiate(
                user_id=body.user_id,
                email=body.email,
                level=body.level,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return record.to_dict()

    @app.get("/wallet/kyc/status", tags=["wallet"])
    async def get_kyc_status_summary() -> Dict[str, Any]:
        kyc = getattr(node, "_kyc_client", None)
        if kyc is None:
            raise HTTPException(
                status_code=503,
                detail="KYC client not initialized.",
            )
        return {
            "commissioned": kyc.is_commissioned(),
            "adapter_wired": kyc.adapter_wired(),
            "vendor": getattr(kyc, "_vendor", None),
            "supported_vendors": list(
                getattr(kyc, "SUPPORTED_VENDORS", []),
            ),
            "record_count": len(kyc.list_records()),
        }

    @app.get("/wallet/kyc", tags=["wallet"])
    async def list_kyc_records(
        limit: int = 100,
    ) -> Dict[str, Any]:
        if limit <= 0 or limit > 10000:
            raise HTTPException(
                status_code=422,
                detail=f"limit must be in [1, 10000], got {limit}",
            )
        kyc = getattr(node, "_kyc_client", None)
        if kyc is None:
            raise HTTPException(
                status_code=503,
                detail="KYC client not initialized.",
            )
        records = kyc.list_records()
        records.sort(key=lambda r: r.created_at, reverse=True)
        return {
            "records": [r.to_dict() for r in records[:limit]],
            "count": len(records),
            "limit": limit,
        }

    @app.get("/wallet/kyc/{user_id}", tags=["wallet"])
    async def get_kyc_record(user_id: str) -> Dict[str, Any]:
        kyc = getattr(node, "_kyc_client", None)
        if kyc is None:
            raise HTTPException(
                status_code=503,
                detail="KYC client not initialized.",
            )
        record = kyc.get_status(user_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"no KYC record for user_id={user_id!r}",
            )
        return record.to_dict()

    @app.post(
        "/wallet/kyc/webhook/{vendor}", tags=["wallet"],
    )
    async def post_kyc_webhook(
        vendor: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Vendor webhook callback with signature verification.

        Sprint 283: reads raw body bytes so the HMAC signature
        can be verified against the exact payload the vendor
        signed. When a vendor-specific webhook secret env var
        is set (PERSONA_WEBHOOK_SECRET / ONFIDO_WEBHOOK_TOKEN),
        signature verification is mandatory; without it,
        sprint-280 behavior is preserved (pass-through) so
        operators still wiring secrets aren't broken.

        PRSM_KYC_WEBHOOK_VERIFY_DISABLED=1 forces bypass for
        dev/staging.
        """
        from prsm.economy.web3.kyc_webhook_verifier import (
            KYCWebhookVerifier,
        )

        kyc = getattr(node, "_kyc_client", None)
        if kyc is None:
            raise HTTPException(
                status_code=503,
                detail="KYC client not initialized.",
            )

        raw_body = await request.body()

        # Resolve vendor-specific webhook secret. Sprint 283
        # ships env vars for Persona + Onfido; Plaid deferred.
        _vendor_lower = (vendor or "").strip().lower()
        secret_env = {
            "persona": "PERSONA_WEBHOOK_SECRET",
            "onfido": "ONFIDO_WEBHOOK_TOKEN",
            "plaid": "PLAID_WEBHOOK_SECRET",
        }.get(_vendor_lower)
        secret = (
            os.environ.get(secret_env) if secret_env else None
        )

        disable_raw = (
            os.environ.get(
                "PRSM_KYC_WEBHOOK_VERIFY_DISABLED", "",
            ).strip().lower()
        )
        verify_disabled = disable_raw in {"1", "true", "yes"}

        # Sp888 — FAIL CLOSED. This endpoint is the ONLY writer that
        # can mint a VERIFIED KYC record (→ auto-provision + raised
        # tier limits). The pre-sp888 behavior processed UNSIGNED
        # webhooks when no secret was configured ("keep operators
        # unblocked") — a fail-open authentication hole: anyone who
        # could reach the endpoint could forge inquiry.approved and
        # bypass KYC entirely. Now a webhook is accepted ONLY when
        # the signature verifies (secret set) OR the operator has
        # EXPLICITLY opted into the dev/test bypass. No secret + no
        # explicit bypass → refuse.
        if not verify_disabled and not secret:
            logger.warning(
                "KYC webhook REFUSED (vendor=%s, client=%s): no "
                "%s configured. Refusing to process an unsigned "
                "compliance webhook (fail-closed, sp888).",
                _vendor_lower,
                request.client.host if request.client else "?",
                secret_env or "<vendor>_WEBHOOK_SECRET",
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "KYC webhook verification is not configured. "
                    f"Set {secret_env or 'the vendor webhook secret'}"
                    " to enable signature verification, or "
                    "PRSM_KYC_WEBHOOK_VERIFY_DISABLED=1 for dev/test."
                    " Refusing to process an unsigned compliance "
                    "webhook."
                ),
            )

        # Enforce verification iff a secret is configured AND the
        # disable flag isn't set. (verify_disabled → explicit
        # dev/test bypass, handled by the fail-closed guard above.)
        if secret and not verify_disabled:
            ok, reason = KYCWebhookVerifier.verify(
                vendor=_vendor_lower,
                body=raw_body,
                headers=dict(request.headers),
                secret=secret,
            )
            if not ok:
                logger.warning(
                    "KYC webhook signature rejected "
                    "(vendor=%s, reason=%s, client=%s)",
                    _vendor_lower, reason,
                    request.client.host if request.client
                    else "?",
                )
                raise HTTPException(
                    status_code=401,
                    detail=(
                        "Webhook signature verification "
                        "failed."
                    ),
                )

            # Sprint 284 — replay defenses (only after
            # signature verifies).
            import time as _time
            from prsm.economy.web3.webhook_replay_defense import (
                is_timestamp_fresh,
            )

            # 1. Persona timestamp window check. Pull the
            # signed timestamp out of the header and reject
            # if outside tolerance.
            if _vendor_lower == "persona":
                persona_sig_header = (
                    request.headers.get("persona-signature")
                    or request.headers.get("Persona-Signature")
                    or ""
                )
                ts_value = ""
                for piece in persona_sig_header.split(","):
                    piece = piece.strip()
                    if piece.startswith("t="):
                        ts_value = piece[2:].strip()
                        break
                # Tunable tolerance via env.
                try:
                    tolerance_sec = int(os.environ.get(
                        "PRSM_KYC_WEBHOOK_TIMESTAMP_"
                        "TOLERANCE_SEC", "300",
                    ))
                except (ValueError, TypeError):
                    tolerance_sec = 300
                fresh_ok, fresh_reason = is_timestamp_fresh(
                    ts_str=ts_value,
                    current_time=_time.time(),
                    tolerance_sec=tolerance_sec,
                )
                if not fresh_ok:
                    logger.warning(
                        "KYC webhook timestamp rejected "
                        "(vendor=%s, reason=%s, client=%s)",
                        _vendor_lower, fresh_reason,
                        request.client.host if request.client
                        else "?",
                    )
                    raise HTTPException(
                        status_code=401,
                        detail=(
                            "Webhook timestamp outside "
                            "freshness window."
                        ),
                    )

            # 2. Signature-hash dedup. Vendor-agnostic. The
            # signature value itself is the perfect replay
            # token (cryptographically unique per body+ts+
            # secret). Reject duplicates.
            replay_ring = getattr(
                node, "_kyc_webhook_replay_ring", None,
            )
            # sp969 — FAIL CLOSED if the replay ring is unavailable. The ring is
            # constructed unconditionally at node init (node.py:2858); None here
            # means construction RAISED (swallowed to None at node.py:2873). The
            # signature-hash dedup is the vendor-agnostic replay defense — and
            # for Onfido (no timestamp) the ONLY one — so we cannot guarantee
            # replay protection on this money-adjacent KYC webhook. Reject (503)
            # rather than silently accept (mirrors sp888 fail-closed). Reached
            # only after the signature check, so a forged webhook is already
            # 401'd above and cannot probe this path.
            if replay_ring is None:
                logger.error(
                    "KYC webhook replay ring unavailable (construction "
                    "failed) — failing closed with 503 (vendor=%s)",
                    _vendor_lower,
                )
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Webhook replay defense temporarily unavailable; "
                        "retry later."
                    ),
                )
            if replay_ring is not None:
                # Persona: use the v1=<hex> portion as the
                # replay token (varies per body+ts).
                # Onfido: signature header IS the hex hmac.
                if _vendor_lower == "persona":
                    replay_token = ""
                    for piece in persona_sig_header.split(","):
                        piece = piece.strip()
                        if piece.startswith("v1="):
                            replay_token = piece[3:].strip()
                            break
                else:
                    replay_token = (
                        request.headers.get(
                            "x-sha2-signature",
                        )
                        or request.headers.get(
                            "X-SHA2-Signature",
                        )
                        or ""
                    )
                if replay_token:
                    fresh_record = replay_ring.record(
                        replay_token,
                    )
                    if not fresh_record:
                        logger.warning(
                            "KYC webhook replay rejected "
                            "(vendor=%s, client=%s, "
                            "token=%s…)",
                            _vendor_lower,
                            request.client.host
                            if request.client else "?",
                            replay_token[:16],
                        )
                        raise HTTPException(
                            status_code=409,
                            detail=(
                                "Webhook replay detected: "
                                "this signature was already "
                                "processed."
                            ),
                        )

        # Parse body AFTER signature verification (defense in
        # depth — never trust unsigned input).
        try:
            payload_json = (
                json.loads(raw_body) if raw_body else {}
            )
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=422,
                detail="invalid JSON body",
            )
        if not isinstance(payload_json, dict):
            raise HTTPException(
                status_code=422,
                detail="webhook body must be a JSON object",
            )

        # Sp852 — translate vendor-native JSON:API envelopes
        # (Persona, etc.) into PRSM canonical
        # {user_id, status, vendor_ref} shape. None return →
        # fall through to legacy flat parsing.
        from prsm.economy.web3.kyc_webhook_normalizer import (
            normalize_webhook_payload,
        )
        try:
            normalized = normalize_webhook_payload(
                vendor=_vendor_lower, body=payload_json,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        if normalized is not None:
            payload_json = normalized

        user_id_raw = payload_json.get("user_id")
        status_raw = payload_json.get("status")
        vendor_ref_raw = payload_json.get("vendor_ref")
        if not isinstance(user_id_raw, str) or not user_id_raw:
            raise HTTPException(
                status_code=422,
                detail="missing required field: user_id",
            )
        if not isinstance(status_raw, str) or not status_raw:
            raise HTTPException(
                status_code=422,
                detail="missing required field: status",
            )

        # Capture old status BEFORE update so the orchestrator
        # can detect a true transition (vs Persona re-firing the
        # same event on retry).
        old_record = kyc.get_status(user_id_raw)
        old_status = old_record.status if old_record else None
        try:
            updated = kyc.update_status(
                user_id=user_id_raw,
                new_status=status_raw,
                vendor_ref_update=vendor_ref_raw,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        if updated is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"no KYC record for user_id={user_id_raw!r}"
                ),
            )

        # Sp858 — auto-provision WaaS wallet on VERIFIED transition.
        # Fail-soft: webhook still returns 200 even if provision
        # errors so Persona doesn't keep retrying.
        from prsm.economy.web3.kyc_to_waas_orchestrator import (
            maybe_auto_provision_waas,
        )
        maybe_auto_provision_waas(
            waas_client=getattr(node, "_coinbase_waas_client", None),
            user_id=updated.user_id,
            email=updated.email,
            new_status=updated.status,
            old_status=old_status,
        )

        return updated.to_dict()

    # Renamed from `_RoyaltyClaimRequest` for OpenAPI hygiene.
    class RoyaltyClaimRequest(BaseModel):
        dry_run: bool = True

    @app.post("/wallet/royalty/claim")
    async def post_royalty_claim(
        body: RoyaltyClaimRequest,
    ) -> Dict[str, Any]:
        """Claim accumulated royalties from RoyaltyDistributor.

        Closes the loop on the offramp-quote claim_required path:
        when /wallet/offramp/quote returns claim_required=True,
        operators authorize the claim via this endpoint (or via
        the prsm_royalty_claim MCP composer that calls it).

        Behavior:
          - dry_run=True (default): read claimable + return artifact
            without on-chain action; status="DRY_RUN"
          - dry_run=False: call client.claim(); status="EXECUTED"
            with tx_hash + amount_claimed
          - claimable=0 + dry_run=False: skip the claim() call
            (avoids on-chain ZeroClaim revert + gas burn);
            status="SKIPPED_ZERO"
        """
        client = getattr(node, "_royalty_distributor_client", None)
        if client is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "RoyaltyDistributor client not wired on this "
                    "node. Either set PRSM_ROYALTY_DISTRIBUTOR_ADDRESS "
                    "explicitly, OR set PRSM_NETWORK=mainnet for "
                    "canonical-fallback wiring (sprint 144). "
                    "Note: testnet has no canonical RoyaltyDistributor "
                    "deployment."
                ),
            )

        ftns_ledger = getattr(node, "ftns_ledger", None)
        decimals = getattr(ftns_ledger, "_decimals", 18) if ftns_ledger else 18

        try:
            claimable_wei = client.claimable()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "RoyaltyDistributorClient.claimable raised: %s", exc,
            )
            raise HTTPException(
                status_code=502,
                detail=f"claimable() RPC failed: {exc}",
            )

        claimable_ftns = float(claimable_wei) / (10 ** decimals)

        if body.dry_run:
            return {
                "status": "DRY_RUN",
                "claimable_ftns": claimable_ftns,
                "amount_claimed_ftns": 0.0,
                "tx_hash": None,
            }

        if claimable_wei == 0:
            return {
                "status": "SKIPPED_ZERO",
                "claimable_ftns": 0.0,
                "amount_claimed_ftns": 0.0,
                "tx_hash": None,
                "note": (
                    "Skipped on-chain claim() call to avoid "
                    "ZeroClaim revert + gas burn."
                ),
            }

        from prsm.economy.web3.provenance_registry import OnChainPendingError
        from fastapi.responses import JSONResponse
        try:
            tx_hash, transfer_status = client.claim()
        except OnChainPendingError as exc:
            # sp915 — broadcast SUCCEEDED but the receipt is unconfirmed; the
            # claim tx is in the mempool. Do NOT flatten to a 502 that invites
            # a re-claim — the second claim() would race the first and revert
            # ZeroClaim once it settles, burning gas. Surface a distinct 202
            # carrying tx_hash so the operator reconciles via the receipt.
            logger.warning(
                "RoyaltyDistributorClient.claim PENDING (tx=%s); do not "
                "re-claim, reconcile via tx_hash",
                getattr(exc, "tx_hash", None),
            )
            return JSONResponse(
                status_code=202,
                content={
                    "status": "PENDING",
                    "tx_hash": getattr(exc, "tx_hash", None),
                    "claimable_ftns": claimable_ftns,
                    "detail": (
                        "claim() broadcast OK but UNCONFIRMED; do NOT "
                        "re-claim — reconcile via tx_hash."
                    ),
                },
            )
        except Exception as exc:  # noqa: BLE001
            # BroadcastFailedError / OnChainRevertedError land here too — both
            # are safe to retry (chain saw nothing / rolled back atomically).
            logger.warning(
                "RoyaltyDistributorClient.claim raised: %s", exc,
            )
            raise HTTPException(
                status_code=502,
                detail=f"claim() failed: {exc}",
            )

        return {
            "status": "EXECUTED",
            "claimable_ftns": claimable_ftns,
            "amount_claimed_ftns": claimable_ftns,
            "tx_hash": tx_hash,
            "transfer_status": (
                transfer_status.value
                if hasattr(transfer_status, "value")
                else str(transfer_status)
            ),
        }

    @app.post("/wallet/offramp/quote")
    async def post_offramp_quote(
        body: OfframpQuoteRequest,
        address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Pre-flight quote for FTNS → USDC → USD off-ramp via
        Aerodrome + Coinbase CDP.

        V1 scope: returns the transaction-summary artifact described
        in Vision §13 Phase 5 step 2 ("Gemini presents an Artifact in
        your side panel"). Does NOT initiate any on-chain swap or
        Coinbase off-ramp; actual execution gates on CDP commission
        per Vision gantt 2026-06-15. Until then status is
        ``PENDING_COMMISSION``.

        Validation:
          - ``usd_amount`` must be positive (Pydantic gt=0).
          - Source balance must be ≥ requested USD; otherwise 422.
          - ``ftns_ledger`` must be initialized; otherwise 503.
        """
        # Pydantic gt=0 catches usd_amount <= 0 with 422; the test
        # suite expects 400 for explicit negative/zero. Re-validate
        # to surface the simpler 400.
        if body.usd_amount <= 0:
            raise HTTPException(
                status_code=400,
                detail="usd_amount must be positive (> 0)",
            )

        if not getattr(node, "ftns_ledger", None):
            raise HTTPException(
                status_code=503,
                detail=(
                    "On-chain ftns_ledger not initialized; "
                    "cannot quote off-ramp without source balance."
                ),
            )

        # Sprint 281 — source_user_id resolution via WaaS.
        # Mutually exclusive with explicit `address` query
        # param (which legacy callers still use). When both
        # are absent, default to the operator's connected
        # ledger address.
        resolved_address = address
        if body.source_user_id:
            waas = getattr(node, "_coinbase_waas_client", None)
            if waas is None:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "WaaS client not initialized; cannot "
                        "resolve source_user_id."
                    ),
                )
            rec = waas.get_wallet(body.source_user_id)
            if rec is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"no WaaS wallet for source_user_id="
                        f"{body.source_user_id!r}"
                    ),
                )
            if rec.address:
                resolved_address = rec.address

        target = (
            resolved_address
            or node.ftns_ledger._connected_address
        )
        balance_ftns = await node.ftns_ledger.get_balance(target)

        rate_raw = os.getenv("PRSM_FTNS_USD_RATE", "").strip()
        usd_rate = 1.0
        if rate_raw:
            try:
                parsed = float(rate_raw)
                if parsed > 0:
                    usd_rate = parsed
            except ValueError:
                pass

        balance_usd = balance_ftns * usd_rate

        # Aggregate-source available: on-chain + claimable royalties.
        # Escrowed FTNS does NOT count (locked in pending compute jobs).
        # Each source fail-soft: RPC errors → treat as 0 contribution.
        decimals = getattr(node.ftns_ledger, "_decimals", 18)
        claimable_ftns = 0.0
        royalty_client = getattr(node, "_royalty_distributor_client", None)
        if royalty_client is not None:
            try:
                claimable_wei = royalty_client.claimable(target)
                claimable_ftns = float(claimable_wei) / (10 ** decimals)
            except Exception as e:
                logger.warning(
                    "offramp/quote: royalty claimable() raised — "
                    "treating as 0 (fail-soft): %s", e,
                )

        available_ftns = balance_ftns + claimable_ftns
        available_usd = available_ftns * usd_rate
        ftns_to_swap = body.usd_amount / usd_rate

        if available_usd < body.usd_amount:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Insufficient balance: requested ${body.usd_amount:.2f}, "
                    f"available ${available_usd:.2f} "
                    f"({available_ftns:.6f} FTNS @ {usd_rate} USD/FTNS"
                    f"; on-chain {balance_ftns:.6f} + claimable "
                    f"{claimable_ftns:.6f})"
                ),
            )

        # Claim-required path: on-chain alone insufficient but
        # aggregate covers. Operator must claim royalties before the
        # eventual swap can execute. Surface the claim amount so the
        # composer can chain a claim step.
        claim_required = balance_ftns < ftns_to_swap
        claim_amount_ftns = (
            max(0.0, ftns_to_swap - balance_ftns) if claim_required else 0.0
        )

        # Sprint 281 — KYC gating mirrors the onramp side.
        # Only triggers when source_user_id is supplied; legacy
        # `address`-based callers see neutral fields preserving
        # backwards-compat.
        offramp_kyc_required = False
        offramp_kyc_status = None
        offramp_kyc_session_url = None
        if body.source_user_id:
            kyc = getattr(node, "_kyc_client", None)
            if kyc is not None:
                rec = kyc.get_status(body.source_user_id)
                if rec is None:
                    offramp_kyc_required = True
                    offramp_kyc_status = "NOT_STARTED"
                elif rec.status != "VERIFIED":
                    offramp_kyc_required = True
                    offramp_kyc_status = rec.status
                    offramp_kyc_session_url = rec.session_url
                else:
                    offramp_kyc_status = "VERIFIED"

        # Sprint 285 — tier-limit check (symmetric with
        # onramp side; gates on source_user_id).
        offramp_tier_block = _tier_check(
            user_id=body.source_user_id,
            requested_usd=float(body.usd_amount),
            kyc_status=offramp_kyc_status,
        )

        # Sprint 282 — record to fiat compliance audit ring.
        _record_fiat_compliance(
            kind="offramp_quote",
            user_id=body.source_user_id or "",
            usd_amount=float(body.usd_amount),
            ftns_amount=float(ftns_to_swap),
            status="PENDING_COMMISSION",
            kyc_status=offramp_kyc_status,
            address=target,
            metadata={
                "bank_account_alias": body.bank_account_alias,
                "claim_required": claim_required,
            },
        )

        return {
            "requested_usd": body.usd_amount,
            "source_address": target,
            "source_user_id": body.source_user_id,
            **offramp_tier_block,
            "source_balance_ftns": balance_ftns,
            "source_balance_usd": balance_usd,
            # Aggregate-source mirror (additive — clients reading
            # legacy fields keep working).
            "available_ftns": available_ftns,
            "available_usd": available_usd,
            "claimable_royalties_ftns": claimable_ftns,
            "claim_required": claim_required,
            "claim_amount_ftns": claim_amount_ftns,
            "kyc_required": offramp_kyc_required,
            "kyc_status": offramp_kyc_status,
            "kyc_session_url": offramp_kyc_session_url,
            "quote": {
                "ftns_to_swap": ftns_to_swap,
                "usdc_received": body.usd_amount,
                "usd_settled": body.usd_amount,
                "swap_route": "aerodrome",
                "offramp_route": "coinbase-cdp",
                "bank_account_alias": body.bank_account_alias,
            },
            "usd_rate": usd_rate,
            "status": "PENDING_COMMISSION",
            "commission_gate_note": (
                "Coinbase CDP commission gates on Aerodrome USDC-FTNS "
                "pool seeding (Vision gantt 2026-06-15). The summary "
                "above shows what the transaction will look like once "
                "execution ships; it does NOT initiate any on-chain "
                "swap or fiat off-ramp."
            ),
        }

    @app.post("/compute/submit")
    async def submit_compute_job(job: JobSubmission) -> Dict[str, Any]:
        """Submit a compute job to the network."""
        # Sprint 197 — same payload-size cap as /api/jobs/submit
        # (gossip-DoS surface). PRSM_MAX_JOB_PAYLOAD_BYTES env
        # override; default 100KB.
        try:
            _payload_bytes = len(
                json.dumps(job.payload).encode("utf-8"),
            )
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail="payload must be JSON-serializable",
            )
        _payload_cap_raw = os.environ.get(
            "PRSM_MAX_JOB_PAYLOAD_BYTES", "",
        ).strip()
        try:
            _payload_cap = (
                int(_payload_cap_raw)
                if _payload_cap_raw else 100 * 1024
            )
            if _payload_cap <= 0:
                raise ValueError("non-positive")
        except (ValueError, TypeError):
            _payload_cap = 100 * 1024
        if _payload_bytes > _payload_cap:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"payload size {_payload_bytes} bytes exceeds "
                    f"PRSM_MAX_JOB_PAYLOAD_BYTES cap of "
                    f"{_payload_cap}. Trim the payload or have "
                    f"the operator raise the cap."
                ),
            )

        if not node.compute_requester:
            raise HTTPException(status_code=503, detail="Compute requester not initialized")

        from prsm.node.compute_provider import JobType
        try:
            job_type = JobType(job.job_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid job type: {job.job_type}. Valid types: inference, embedding, benchmark",
            )

        # Sprint 487 (F25) — concurrent submits race on the
        # dag_ledger's optimistic locking. Pre-fix:
        # ConcurrentModificationError propagated up as a raw
        # 500 with no body. Concurrent dogfood test hit 7/10
        # 5xx responses on a 10-caller burst. Now: bounded
        # retry (3 attempts, exponential backoff), then a
        # clean 503 with Retry-After-style hint so the client
        # can back off rather than the daemon crashing.
        import asyncio
        from prsm.node.dag_ledger import (
            ConcurrentModificationError, BalanceLockError,
            InsufficientBalanceError,
        )
        last_exc = None
        for attempt in range(3):
            try:
                submitted = await node.compute_requester.submit_job(
                    job_type=job_type,
                    payload=job.payload,
                    ftns_budget=job.ftns_budget,
                )
                return {
                    "job_id": submitted.job_id,
                    "status": submitted.status.value,
                    "job_type": submitted.job_type.value,
                    "ftns_budget": submitted.ftns_budget,
                }
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except InsufficientBalanceError as e:
                # Sprint 487 (F25) — InsufficientBalanceError
                # from dag_ledger must surface as 400 (client-
                # actionable) not 500 (server fault). Operator-
                # visible signal: "your wallet doesn't have
                # enough FTNS for this budget".
                raise HTTPException(
                    status_code=400,
                    detail=str(e),
                )
            except (
                ConcurrentModificationError, BalanceLockError,
            ) as e:
                last_exc = e
                # Exponential backoff: 10ms, 40ms, 90ms
                await asyncio.sleep(0.01 * (attempt + 1) ** 2)
                continue
        # All retries exhausted — return 503 with actionable
        # detail so the client knows to back off.
        raise HTTPException(
            status_code=503,
            detail=(
                "Compute submit failed after 3 retry attempts "
                "due to contention on the FTNS ledger. Last "
                f"error: {last_exc}. Back off and retry."
            ),
        )

    @app.get("/compute/job/{job_id}")
    async def get_job_status(job_id: str) -> Dict[str, Any]:
        """Get status of a submitted compute job."""
        if not node.compute_requester:
            raise HTTPException(status_code=503, detail="Compute requester not initialized")

        job = node.compute_requester.submitted_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "job_type": job.job_type.value,
            "provider_id": job.provider_id,
            "result": job.result,
            "result_verified": job.result_verified,
            "error": job.error,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
        }

    @app.post("/compute/query")
    async def compute_query(body: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Submit a compute job and wait for the result (blocking).

        POST body: {"prompt": "...", "model": "nwtn", "timeout": 120}
        Returns: {"job_id", "response", "result"}
        """
        if not node.compute_requester or not node.compute_provider:
            raise HTTPException(status_code=503, detail="Compute not initialized")

        prompt = body.get("prompt", "")
        model = body.get("model", "nwtn")
        # Sprint 196 — int/float casts were uncaught → 500 on
        # non-numeric input. Also no bounds: negative timeout
        # silently accepted (0 effective), negative budget silently
        # treated as 0 (free query) — operator-confusing.
        _raw_to = body.get("timeout", 120.0)
        try:
            timeout = float(_raw_to)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail=f"timeout must be a positive number; got {_raw_to!r}.",
            )
        if timeout <= 0 or timeout > 3600:
            raise HTTPException(
                status_code=422,
                detail=f"timeout must be in (0, 3600]; got {timeout}.",
            )
        _raw_b = body.get("budget", 0.0)
        try:
            budget = float(_raw_b)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail=f"budget must be a non-negative number; got {_raw_b!r}.",
            )
        if budget < 0:
            raise HTTPException(
                status_code=422,
                detail=f"budget must be >= 0; got {budget}.",
            )

        from prsm.node.compute_provider import JobType, ComputeJob

        import uuid as _uuid

        job_id = "compute-" + _uuid.uuid4().hex
        escrow_entry = None

        # --- Phase 1: Lock escrow (no gossip broadcast) ---
        if budget > 0 and hasattr(node, '_payment_escrow') and node._payment_escrow:
            escrow_entry = await node._payment_escrow.create_escrow(
                job_id=job_id,
                amount=budget,
                requester_id=node.identity.node_id,
            )
            if escrow_entry:
                logger.info(
                    f"compute_query: escrow locked {budget:.6f} FTNS "
                    f"for {job_id[:8]}"
                )
            else:
                logger.warning(
                    f"compute_query: escrow creation failed for {job_id[:8]}"
                )

        # --- Phase 2: Route the job ---
        result = None

        # Try Agent Forge first (Rings 1-10 pipeline) if available
        if hasattr(node, 'agent_forge') and node.agent_forge is not None:
            try:
                forge_result = await node.agent_forge.run(
                    query=prompt,
                    budget_ftns=budget,
                )
                if forge_result and forge_result.get("status") == "success":
                    # Extract response from forge result
                    route = forge_result.get("route", "unknown")
                    if route == "direct_llm":
                        result = {"response": forge_result.get("response", ""), "route": route}
                    elif route == "swarm":
                        output = forge_result.get("aggregated_output", {})
                        result = {"response": str(output), "route": route}
                    else:
                        result = {"response": str(forge_result), "route": route}
                    result["forge_result"] = forge_result
            except Exception as e:
                logger.debug(f"compute_query: forge path failed, falling back: {e}")

        # Try gossip-based peers if forge didn't produce a result
        has_peers = (
            node.transport
            and hasattr(node.transport, 'peer_count')
            and node.transport.peer_count > 0
        )
        if has_peers:
            try:
                # Dynamic timeout: allow enough time for remote inference
                # API timeout is the user-facing limit; gossip gets most of it
                gossip_timeout = max(timeout * 0.8, 30.0)

                # Submit via gossip for multi-node federation
                # Pass our job_id so escrow and gossip track the same ID
                submitted = await node.compute_requester.submit_job(
                    job_type=JobType.INFERENCE,
                    payload={"prompt": prompt, "model": model},
                    ftns_budget=budget,
                    use_escrow=False,  # escrow already created above
                    job_id=job_id,     # propagate API job_id
                )
                result = await node.compute_requester.get_result(
                    submitted.job_id, timeout=gossip_timeout
                )
                if result:
                    job_id = submitted.job_id
            except Exception as e:
                logger.warning(f"compute_query: gossip routing failed: {e}")

        # Fallback: direct self-compute (single-node mode)
        if result is None and node.compute_provider.allow_self_compute:
            self_job = ComputeJob(
                job_id=job_id,
                job_type=JobType.INFERENCE,
                payload={"prompt": prompt, "model": model},
                requester_id=node.identity.node_id,
                ftns_budget=budget,
            )
            logger.info(
                f"compute_query: self-compute for {job_id[:8]}"
            )
            await node.compute_provider._execute_job(self_job)
            completed = node.compute_provider.completed_jobs.get(job_id)
            if completed:
                result = completed.result

        if result is None:
            # Refund escrow on failure
            if escrow_entry and node._payment_escrow:
                try:
                    await node._payment_escrow.refund_escrow(job_id)
                except Exception:
                    pass
            raise HTTPException(
                status_code=504, detail="Compute timed out or no provider accepted"
            )

        # --- Phase 3: Release escrow to provider ---
        if budget > 0 and escrow_entry and node._payment_escrow:
            try:
                await node._payment_escrow.release_escrow(
                    job_id=job_id,
                    provider_id=node.identity.node_id,
                    consensus_reached=True,
                )
                logger.info(
                    f"api: escrow released {budget:.6f} FTNS for {job_id[:8]}"
                )
            except Exception as e:
                logger.warning(f"api: escrow release for {job_id[:8]} failed: {e}")

        return {
            "job_id": job_id,
            "response": result.get("response", result.get("text", str(result))),
            "result": result,
        }

    @app.get("/privacy/budget", tags=["privacy"])
    async def get_privacy_budget() -> Dict[str, Any]:
        """Return the current differential-privacy budget audit report.

        Used by the prsm_privacy_status MCP tool. The budget is enforced
        across all forge queries with privacy_level != 'none'.
        """
        if not hasattr(node, "privacy_budget") or node.privacy_budget is None:
            raise HTTPException(
                status_code=503,
                detail="Privacy budget not initialized (Ring 7 unavailable)",
            )
        return node.privacy_budget.get_audit_report()

    @app.post("/compute/forge/quote")
    async def compute_forge_quote(body: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Get a cost quote for a forge query without executing it (free).

        POST body: {
            "query": "...",
            "shard_cids": ["QmA", "QmB"],   // optional
            "shard_count": 3,                // optional, default 3
            "hardware_tier": "t2",           // optional, default t2
            "estimated_pcu_per_shard": 50.0  // optional, default 50.0
        }

        Returns the same CostQuote shape used by the prsm_quote MCP tool and
        the Python SDK's client.quote() helper. This endpoint is what the
        JavaScript and Go SDKs call.
        """
        from prsm.economy.pricing import PricingEngine

        query = body.get("query", "") or ""
        if not query.strip():
            # Sprint 536 F66 fix: include schema hint in error
            raise HTTPException(
                status_code=400,
                detail=(
                    "Missing 'query' field. Expected body: "
                    "{\"query\": \"<text>\", \"shard_cids\": [...], "
                    "\"budget_ftns\": <float>}"
                ),
            )

        shard_cids = body.get("shard_cids") or []
        # If caller passed shard_cids, use that count; otherwise honor explicit shard_count.
        # Sprint 195 — int/float coercion was uncaught and raised
        # ValueError → 500 on non-numeric input. Validate upfront.
        if shard_cids:
            shard_count = len(shard_cids)
        else:
            _raw_sc = body.get("shard_count", 3)
            try:
                shard_count = int(_raw_sc)
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"shard_count must be a positive integer; "
                        f"got {_raw_sc!r}."
                    ),
                )
        # Clamp to sane bounds: 1 ≤ shard_count ≤ 100 (matches
        # PRSM_MAX_FORGE_SHARDS default used elsewhere).
        if shard_count < 1 or shard_count > 100:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"shard_count must be in [1, 100]; got {shard_count}."
                ),
            )
        hardware_tier = str(body.get("hardware_tier", "t2"))
        # Hardware tier must match a known tier — t1/t2/t3/t4.
        # Pre-fix unknown tiers (e.g. "<script>") passed through to
        # PricingEngine which then hung trying to map them.
        _ALLOWED_TIERS = ("t1", "t2", "t3", "t4")
        if hardware_tier not in _ALLOWED_TIERS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"hardware_tier must be one of {list(_ALLOWED_TIERS)}; "
                    f"got {hardware_tier!r}."
                ),
            )
        _raw_pcu = body.get("estimated_pcu_per_shard", 50.0)
        try:
            estimated_pcu = float(_raw_pcu)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"estimated_pcu_per_shard must be a positive "
                    f"number; got {_raw_pcu!r}."
                ),
            )
        if estimated_pcu <= 0:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"estimated_pcu_per_shard must be > 0; "
                    f"got {estimated_pcu}."
                ),
            )

        engine = PricingEngine()
        quote = engine.quote_swarm_job(
            shard_count=shard_count,
            hardware_tier=hardware_tier,
            estimated_pcu_per_shard=estimated_pcu,
        )
        return {
            "query": query,
            "shard_count": shard_count,
            "hardware_tier": hardware_tier,
            **quote.to_dict(),
        }

    # Sprint 271 — operator content filter enforcement helper for
    # the 3 compute entry points (forge / inference / inference_
    # stream). Per ContentSelfFilter design (R9 Phase 6.4) the
    # filter must run BEFORE escrow lock / compute cost. HTTP 451
    # (Unavailable For Legal Reasons, RFC 7725) is the canonical
    # refusal. No filter wired → pre-271 pass-through.
    def _enforce_content_filter(
        *,
        prompt: str = "",
        shard_cids: Optional[List[str]] = None,
        model_id: str = "",
    ) -> None:
        store = getattr(node, "_content_filter_store", None)
        if store is None:
            return
        from prsm.node.content_self_filter import DispatchContext
        snapshot = store.current()
        # Check each shard_cid as a separate dispatch context;
        # ANY blocked CID refuses the whole dispatch.
        for cid in (shard_cids or []):
            decision = snapshot.evaluate(
                DispatchContext(content_id=cid)
            )
            if not decision.allow:
                logger.info(
                    "content-filter: refused (forge) "
                    "cid=%s reason=%s",
                    (cid or "")[:14], decision.reason,
                )
                raise HTTPException(
                    status_code=451,
                    detail=(
                        f"dispatch refused by operator's "
                        f"content filter: {decision.reason} "
                        f"(matched={decision.matched_value!r})"
                    ),
                )
        # Then check prompt + model tags. model_id passed as a
        # 1-element tag set so an operator-blocked model maps to
        # the same blocked_model_tag axis.
        if prompt or model_id:
            decision = snapshot.evaluate(DispatchContext(
                prompt_text=prompt or "",
                model_tags=frozenset(
                    [model_id] if model_id else []
                ),
            ))
            if not decision.allow:
                logger.info(
                    "content-filter: refused (inference) "
                    "model=%s reason=%s",
                    model_id[:20], decision.reason,
                )
                raise HTTPException(
                    status_code=451,
                    detail=(
                        f"dispatch refused by operator's "
                        f"content filter: {decision.reason}"
                    ),
                )

    def _enforce_corp_capability(
        cap_header: Optional[str],
        red_header: Optional[str],
    ) -> None:
        """Sprint 306a — Vision §7 Enterprise Confidentiality
        Mode layer-2 LIVE redemption.

        When BOTH `X-CORP-Capability` + `X-CORP-Redemption`
        headers are present (base64-JSON each), verify the
        dual-signature capability + redemption against this
        node's `_corp_capability_store` BEFORE doing any
        work. Refusal: 402 Payment Required (semantically:
        "this authorization isn't valid against your
        quota"). Distinct from 412 (TEE policy) and 451
        (content filter).

        Headers absent → no gating (opt-in for enterprises;
        existing inference requests unaffected). Either
        header alone → 422 (operator confusion).
        """
        import base64 as _b64
        import json as _json

        if cap_header is None and red_header is None:
            return
        if cap_header is None or red_header is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "X-CORP-Capability and X-CORP-Redemption "
                    "must both be present together; either "
                    "alone is operator confusion"
                ),
            )

        store = getattr(
            node, "_corp_capability_store", None,
        )
        if store is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "node received $CORP capability headers "
                    "but no capability store is wired; set "
                    "PRSM_CORP_CAPABILITY_DIR and restart"
                ),
            )

        from prsm.enterprise.corp_capability import (
            CorpCapability, RedemptionRequest,
        )

        # Decode both headers — base64 → JSON → dataclass.
        # Any decode/parse failure is 422 (the requester
        # gave us malformed input).
        try:
            cap_blob = _b64.b64decode(
                cap_header, validate=True,
            )
            red_blob = _b64.b64decode(
                red_header, validate=True,
            )
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"X-CORP-* header not valid base64: {e}"
                ),
            )
        try:
            cap_dict = _json.loads(cap_blob)
            red_dict = _json.loads(red_blob)
        except _json.JSONDecodeError as e:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"X-CORP-* header not valid JSON: {e}"
                ),
            )
        try:
            cap = CorpCapability.from_dict(cap_dict)
            req = RedemptionRequest.from_dict(red_dict)
        except (KeyError, ValueError, TypeError) as e:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"X-CORP-* header malformed: {e}"
                ),
            )

        # Redeem — pure verification + record-keeping.
        result = store.redeem(cap, req)
        if result.status.value != "pass":
            raise HTTPException(
                status_code=402,
                detail=(
                    f"$CORP capability redemption refused: "
                    f"{result.diagnostic}"
                ),
            )

    def _enforce_tee_policy(body: Dict[str, Any]) -> None:
        """Sprint 305a — Vision §7 Enterprise Confidentiality
        Mode layer-3 LIVE dispatch enforcement.

        If the request carries an optional `tee_policy`
        field, evaluate it against THIS node's attestation
        blob BEFORE doing any work. Refuse with 412
        Precondition Failed when the policy is not
        satisfied. Absent field → no gating (backwards-
        compatible). Malformed shape → 422.

        Composer-only on the operator side: we refuse, we
        don't decrypt or execute anything risky. The
        enterprise requester learns from the 412 detail
        which tier the node provides and which they asked
        for, then either reroutes to a compliant node or
        loosens the policy.
        """
        from prsm.enterprise.tee_policy import (
            TEEPolicy, evaluate_attestation_blob,
            operator_floor_policy_from_env,
        )
        # sp1114 (Domain-08 review MEDIUM-7) — the OPERATOR-mandated TEE floor. The gate
        # was requester-opt-in only (absent tee_policy → no gating), so an operator could
        # not require "this node only serves >= hardware-verified work". The floor (from
        # PRSM_MIN_ATTESTATION_TIER etc.) is now merged into EVERY request: a requester's
        # policy can only TIGHTEN it, never weaken it. None floor + no requester policy →
        # unchanged (no gating).
        try:
            floor = operator_floor_policy_from_env()
        except ValueError as e:
            # Misconfigured floor must fail CLOSED, not silently disable the control.
            raise HTTPException(
                status_code=412,
                detail=f"node TEE floor misconfigured (refusing dispatch): {e}",
            )
        raw_policy = body.get("tee_policy")
        if raw_policy is None and floor is None:
            return
        policy: Optional[TEEPolicy] = None
        if raw_policy is not None:
            if not isinstance(raw_policy, dict):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "tee_policy must be a JSON object with "
                        "at least 'min_attestation_tier'"
                    ),
                )
            try:
                policy = TEEPolicy.from_dict(raw_policy)
            except ValueError as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"invalid tee_policy: {e}",
                )
        # Merge: the effective policy must satisfy BOTH the requester and the operator
        # floor (stricter of the two on every dimension).
        if policy is None:
            policy = floor
        elif floor is not None:
            policy = policy.stricter_of(floor)
        blob = getattr(
            node, "_tee_node_attestation_blob", None,
        )
        if blob is not None and not isinstance(
            blob, (bytes, bytearray),
        ):
            blob = None
        result = evaluate_attestation_blob(
            bytes(blob) if blob else None, policy,
        )
        if result.status.value != "pass":
            raise HTTPException(
                status_code=412,
                detail=(
                    f"node refuses dispatch: TEE policy "
                    f"not satisfied. effective_tier="
                    f"{result.effective_tier.value}, "
                    f"required="
                    f"{result.min_required_tier.value}, "
                    f"vendor={result.vendor or '(none)'}. "
                    f"{result.diagnostic}"
                ),
            )

    @app.post("/compute/forge")
    async def compute_forge(
        request: Request,
        body: Dict[str, Any] = {},
        idempotency_key: Optional[str] = Header(
            default=None, alias="Idempotency-Key",
        ),
    ) -> Dict[str, Any]:
        """Submit a query through the full Ring 1-10 Agent Forge pipeline.

        This is the end-to-end sovereign-edge AI path:
        1. AgentForge decomposes the query via LLM
        2. Finds relevant data shards (if any)
        3. Quotes costs via PricingEngine
        4. Routes to appropriate execution path:
           - DIRECT_LLM: simple queries answered by LLM directly
           - SINGLE_AGENT: dispatch WASM agent to one node
           - SWARM: fan out agents across multiple nodes
        5. Collects and aggregates results
        6. Settles FTNS payments
        7. Returns the answer

        POST body: {
            "query": "...",
            "budget_ftns": 10.0,
            "shard_cids": ["QmA", "QmB"],  // optional
            "privacy_level": "standard"     // none, standard, high, maximum
        }
        """
        # Idempotency-Key handling FIRST (before any other checks):
        # if header present + we've seen this key, return the cached
        # job's status directly without locking a new escrow /
        # re-running compute. Retry-safe POST for clients that may
        # double-fire on network blips. Cache hit doesn't require
        # agent_forge to be wired.
        if idempotency_key and getattr(node, "_job_history", None) is not None:
            try:
                prior_job_id = node._job_history.lookup_by_idempotency_key(
                    idempotency_key,
                )
                if prior_job_id:
                    prior_record = node._job_history.get(prior_job_id)
                    if prior_record is not None:
                        return {
                            "status": "idempotent_replay",
                            "job_id": prior_job_id,
                            "history": prior_record.to_dict(),
                            "note": (
                                "Returning cached result from prior request "
                                "with same Idempotency-Key. No new escrow "
                                "locked; no compute re-run."
                            ),
                        }
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Idempotency lookup raised; proceeding with "
                    "fresh forge: %s", exc,
                )

        # Sprint 756 — active-window gate (same as /compute/inference).
        _check_active_window_or_503()
        # Sprint 774 — preemption gate (same as /compute/inference).
        _check_not_preempted_or_503()
        # PRSM_FORGE_MAX_RPS_PER_REQUESTER rate limiting (DoS
        # protection). Default unset → no limiting. Per-requester
        # token bucket. Named "forge" so /compute/inference's bucket
        # is independent.
        _rps_raw = os.getenv(
            "PRSM_FORGE_MAX_RPS_PER_REQUESTER", "",
        ).strip()
        if _rps_raw:
            try:
                _rps = float(_rps_raw)
                if _rps > 0:
                    from prsm.node.rate_limiter import (
                        get_or_build_bucket,
                    )
                    bucket = get_or_build_bucket(_rps, name="forge")
                    if bucket is not None:
                        # Sprint 741 F69 fix — same bug class as
                        # /compute/inference + /compute/inference/stream.
                        # Pre-741 `requester` was node.identity.node_id
                        # (LOCAL daemon constant) so the bucket was
                        # effectively global.
                        requester = _resolve_requester_key(request)
                        if not bucket.try_consume(requester):
                            retry = bucket.retry_after(requester)
                            raise HTTPException(
                                status_code=429,
                                detail=(
                                    f"Rate limit exceeded for "
                                    f"requester {requester[:24]}... "
                                    f"on /compute/forge "
                                    f"(cap {_rps}/sec). Retry "
                                    f"after {retry:.2f}s."
                                ),
                                headers={
                                    "Retry-After": f"{retry:.2f}",
                                },
                            )
            except ValueError:
                logger.warning(
                    "PRSM_FORGE_MAX_RPS_PER_REQUESTER=%r not "
                    "numeric; rate limiting disabled", _rps_raw,
                )

        # Sprint 157 — privacy_level enum validated upfront. Pre-fix
        # the endpoint accepted any string via epsilon_map.get(...,
        # 8.0) — bad values silently fell back to "standard" epsilon
        # which is BOTH wrong-by-construction (operator asked for a
        # tier that doesn't exist) and quietly downgrades privacy.
        _ALLOWED_PRIVACY = ("none", "standard", "high", "maximum")
        _privacy_raw = body.get("privacy_level", "standard")
        if _privacy_raw not in _ALLOWED_PRIVACY:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"privacy_level must be one of "
                    f"{list(_ALLOWED_PRIVACY)}; got {_privacy_raw!r}."
                ),
            )

        # Sprint 153 — validate budget_ftns FIELD upfront so a
        # malformed body returns 422 even when agent_forge is down.
        # Pre-fix the body's budget_ftns was only float()'d inside
        # the cap try/except, masking type errors as cap-disable
        # warnings and leaking through to a misleading 503.
        if "budget_ftns" in body:
            _raw = body["budget_ftns"]
            try:
                _budget_validated = float(_raw)
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"budget_ftns must be a positive number; "
                        f"got {_raw!r}."
                    ),
                )
            if _budget_validated <= 0:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"budget_ftns must be > 0; got "
                        f"{_budget_validated}."
                    ),
                )

        # PRSM_MAX_FTNS_PER_JOB cap enforcement (cost-control).
        # Default unlimited; non-numeric/zero/negative env values
        # silently disable the cap (log WARN). Operators tune this
        # when worried about misbehaving AI agents draining FTNS
        # via single oversized requests. Fires BEFORE agent_forge
        # availability check so an oversized request cleanly 422s
        # even when the forge subsystem is down.
        _cap_raw = os.getenv("PRSM_MAX_FTNS_PER_JOB", "").strip()
        if _cap_raw:
            try:
                _cap = float(_cap_raw)
                if _cap > 0:
                    requested = float(body.get("budget_ftns", 10.0))
                    if requested > _cap:
                        raise HTTPException(
                            status_code=422,
                            detail=(
                                f"budget_ftns {requested} exceeds "
                                f"PRSM_MAX_FTNS_PER_JOB cap of {_cap}. "
                                f"Either lower the budget or have the "
                                f"operator raise the cap."
                            ),
                        )
            except ValueError:
                logger.warning(
                    "PRSM_MAX_FTNS_PER_JOB=%r not numeric; cap disabled",
                    _cap_raw,
                )

        # Validate query BEFORE agent_forge availability so a
        # malformed request gets the right 4xx error code
        # regardless of whether the forge subsystem is wired.
        query = body.get("query", "")
        if not query or not query.strip():
            # Sprint 536 F66 fix: schema hint
            raise HTTPException(
                status_code=400,
                detail=(
                    "Missing 'query' field (or whitespace-only). "
                    "Expected body: {\"query\": \"<text>\", "
                    "\"budget_ftns\": <float>, \"privacy_tier\": "
                    "\"none|standard|high|maximum\"}"
                ),
            )
        # Cap query length to prevent prompt-injection DoS via
        # multi-MB queries that amplify through the LLM token
        # economy. Default 100KB covers any practical research
        # question; operators tune via PRSM_MAX_QUERY_BYTES.
        _query_cap_raw = os.getenv("PRSM_MAX_QUERY_BYTES", "").strip()
        try:
            _query_cap = (
                int(_query_cap_raw) if _query_cap_raw else 100 * 1024
            )
            if _query_cap <= 0:
                raise ValueError("non-positive")
        except (ValueError, TypeError):
            logger.warning(
                "PRSM_MAX_QUERY_BYTES=%r not a positive int; "
                "using 100KB default", _query_cap_raw,
            )
            _query_cap = 100 * 1024
        query_bytes = len(query.encode("utf-8"))
        if query_bytes > _query_cap:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"query size {query_bytes} bytes exceeds "
                    f"PRSM_MAX_QUERY_BYTES cap of {_query_cap}. "
                    f"Trim the query or have the operator raise "
                    f"the cap."
                ),
            )

        # Cap shard_cids list length. Each shard touches the
        # swarm dispatcher per the QueryOrchestrator canonical
        # workflow — unbounded list = unbounded dispatch.
        # Default 100 covers any practical query. Validation
        # before agent_forge check so 4xx is clean regardless
        # of forge state.
        _early_shard_cids = body.get("shard_cids", None)
        if _early_shard_cids is not None:
            _shard_cap_raw = os.getenv(
                "PRSM_MAX_FORGE_SHARDS", "",
            ).strip()
            try:
                _shard_cap = (
                    int(_shard_cap_raw) if _shard_cap_raw else 100
                )
                if _shard_cap <= 0:
                    raise ValueError("non-positive")
            except (ValueError, TypeError):
                logger.warning(
                    "PRSM_MAX_FORGE_SHARDS=%r not a positive int; "
                    "using 100 default", _shard_cap_raw,
                )
                _shard_cap = 100
            if len(_early_shard_cids) > _shard_cap:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"shard_cids count {len(_early_shard_cids)} "
                        f"exceeds PRSM_MAX_FORGE_SHARDS cap of "
                        f"{_shard_cap}. Trim the shard list or "
                        f"have the operator raise the cap."
                    ),
                )

        # Sprint 271 — operator content filter (BEFORE escrow
        # lock / forge availability / compute cost). Refuses with
        # 451 if query matches blocked_input_patterns OR any
        # shard_cid is in blocked_content_ids.
        _enforce_content_filter(
            prompt=query,
            shard_cids=_early_shard_cids,
        )

        if not hasattr(node, 'agent_forge') or node.agent_forge is None:
            raise HTTPException(
                status_code=503,
                detail="Agent forge not initialized. Check LLM backend configuration."
            )

        budget_ftns = float(body.get("budget_ftns", 10.0))
        shard_cids = body.get("shard_cids", None)
        privacy_level_str = body.get("privacy_level", "standard")

        # Enforce minimum budget — PRSM requires FTNS for execution
        if budget_ftns <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "FTNS budget is required for query execution. "
                    "Set budget_ftns to at least 0.01 FTNS. "
                    "Use GET /compute/quote or the prsm_quote MCP tool to estimate costs first."
                ),
            )

        # Lock escrow if budget > 0
        job_id = "forge-" + _uuid.uuid4().hex[:12]
        _job_started_at = _time_for_history.time()
        _have_history = (
            hasattr(node, "_job_history") and node._job_history is not None
        )

        # Sp1180 — CROSS-REPLICA idempotency claim (opt-in). sp1177
        # closed the same-replica double-charge (the common keepalive/
        # affinity retry), but on a multi-replica deployment two same-key
        # requests routed to DIFFERENT replicas each miss the other's
        # in-memory JobHistoryStore index and both lock an escrow (an
        # un-refunded double-charge for a single logical request). When a
        # shared claim store is wired (node._forge_idempotency_redis,
        # from PRSM_FORGE_IDEMPOTENCY_REDIS_URL), atomically claim the key
        # ACROSS replicas via SETNX BEFORE any local state / escrow: if
        # another replica already owns it, 409 (no second escrow). This
        # runs first so a cross-replica duplicate can't pollute the local
        # index. Fail-OPEN to the sp1177 in-process guarantee on a claim-
        # store error (a Redis outage must not break forge entirely; the
        # double-charge guard is a UX/refundable concern, unlike the AML
        # limit which fails closed).
        _idem_redis = getattr(node, "_forge_idempotency_redis", None)
        if idempotency_key and _idem_redis is not None:
            try:
                _claim_key = f"prsm:forge:idem:{idempotency_key}"
                _claimed = _idem_redis.set(
                    _claim_key, job_id, nx=True, ex=900,
                )
                if not _claimed:
                    _owner = None
                    try:
                        _owner = _idem_redis.get(_claim_key)
                    except Exception:  # noqa: BLE001
                        pass
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "error": "idempotency_key_in_progress",
                            "owner_job_id": _owner,
                            "message": (
                                "A request with this Idempotency-Key is "
                                "already in progress on another node; "
                                "retry to fetch the result. No new escrow "
                                "locked."
                            ),
                        },
                    )
            except HTTPException:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "sp1180: cross-replica forge idempotency claim "
                    "raised; proceeding with the sp1177 in-process "
                    "guarantee only: %s", exc,
                )

        # Sp1177 — close the Idempotency-Key double-charge TOCTOU. The
        # top-of-handler lookup (~6403) is an UNLOCKED fast-replay
        # optimization: two concurrent requests with the SAME key both
        # miss it (neither has registered the key yet) and both fall
        # through to the `await create_escrow` below — locking TWO
        # escrows and running the pipeline twice for one logical
        # request (a 2x charge; the pre-fix register-on-write happened
        # only AFTER create_escrow). Close the window with a re-lookup +
        # reserve performed SYNCHRONOUSLY here, immediately before the
        # escrow-locking await: there is no `await` between the
        # re-lookup and the put, so the single-threaded event loop
        # cannot interleave another forge coroutine in this window — the
        # reserve is atomic without a lock. A concurrent duplicate, when
        # it next runs, sees the reserved key (here or at the top
        # lookup) and replays instead of locking a second escrow.
        _idem_reserved = False
        if idempotency_key and _have_history:
            from prsm.node.job_history import (
                JobHistoryRecord as _JobRec,
                JobStatus as _JobStat,
            )
            try:
                _dup_job_id = node._job_history.lookup_by_idempotency_key(
                    idempotency_key,
                )
                if _dup_job_id:
                    _dup_rec = node._job_history.get(_dup_job_id)
                    if _dup_rec is not None:
                        return {
                            "status": "idempotent_replay",
                            "job_id": _dup_job_id,
                            "history": _dup_rec.to_dict(),
                            "note": (
                                "Returning the cached result from a "
                                "concurrent request with the same "
                                "Idempotency-Key. No new escrow locked; "
                                "no compute re-run."
                            ),
                        }
                # MISS → reserve the key NOW (synchronously, before the
                # create_escrow await) so a concurrent duplicate replays.
                node._job_history.put_with_idempotency(
                    _JobRec(
                        job_id=job_id,
                        query=query,
                        status=_JobStat.IN_PROGRESS,
                        started_at=_job_started_at,
                    ),
                    idempotency_key=idempotency_key,
                )
                _idem_reserved = True
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Idempotency reserve raised; proceeding with a "
                    "fresh forge (a concurrent duplicate may "
                    "double-charge): %s", exc,
                )

        escrow_entry = None
        if budget_ftns > 0 and hasattr(node, '_payment_escrow') and node._payment_escrow:
            escrow_entry = await node._payment_escrow.create_escrow(
                job_id=job_id,
                amount=budget_ftns,
                requester_id=node.identity.node_id,
            )

        # B8 async-dispatch follow-on: record IN_PROGRESS to
        # JobHistoryStore so /compute/status/{job_id} can surface
        # richer state than the escrow lifecycle alone. Best-effort —
        # store may be None on operators that haven't wired it. When the
        # sp1177 reserve above already registered the IN_PROGRESS record
        # + key, skip this (a re-put would be a redundant disk write);
        # this still runs for the no-key path and the reserve-failed
        # fallback.
        if not _idem_reserved and _have_history:
            try:
                from prsm.node.job_history import (
                    JobHistoryRecord as _JobRec,
                    JobStatus as _JobStat,
                )
                _ip_record = _JobRec(
                    job_id=job_id,
                    query=query,
                    status=_JobStat.IN_PROGRESS,
                    started_at=_job_started_at,
                )
                if idempotency_key:
                    node._job_history.put_with_idempotency(
                        _ip_record, idempotency_key=idempotency_key,
                    )
                else:
                    node._job_history.put(_ip_record)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "JobHistoryStore put (IN_PROGRESS) failed for "
                    "job_id=%s: %s",
                    job_id, exc,
                )

        try:
            # Dispatch on agent_forge type. The QueryOrchestrator
            # (Ring 5 replacement, wired via PRSM_QUERY_ORCHESTRATOR_ENABLED)
            # exposes ``dispatch_query`` instead of the legacy
            # ``run`` surface; we route both through this endpoint
            # so MCP clients see a single /compute/forge.
            if hasattr(node.agent_forge, "dispatch_query"):
                # PRSM-PROV-1 / QueryOrchestrator path. Build a 32-byte
                # query_id (binds A6/A9 invariants per the aggregator-
                # selector threat model) and call dispatch_query.
                import os as _os_for_qid
                query_id = _os_for_qid.urandom(32)
                qo_result = await node.agent_forge.dispatch_query(
                    query=query,
                    prompter_node_id=node.identity.node_id,
                    query_id=query_id,
                )
                # Marshal AggregatedResult → /compute/forge response shape.
                # payload is the aggregator's combined output (UTF-8 bytes
                # for COUNT op; opaque for other ops).
                try:
                    response_text = qo_result.payload.decode("utf-8")
                except UnicodeDecodeError:
                    response_text = qo_result.payload.hex()
                # sp998 — derive each worker's node_id from its pubkey so payout
                # legs are keyed by a payable identity (see participants below).
                from prsm.node.identity import (
                    node_id_for_public_key as _node_id_for_pubkey,
                )
                result = {
                    "status": "success",
                    "route": "qo_swarm",
                    "aggregator_node_id": qo_result.aggregator_node_id,
                    "contributing_shards": list(qo_result.contributing_shards),
                    "response": response_text,
                    # §4 step 6 settlement attribution. List of
                    # {shard_cid, source_agent_pubkey_hex, creator_id,
                    #  pcu_consumed} — settlement layer below
                    # builds the escrow split from this. Sprint 239
                    # added pcu_consumed: pre-fix the field was on
                    # ParticipantAttribution but stripped here, so
                    # compute_split_amounts always saw zero and fell
                    # back to uniform.
                    "participants": [
                        {
                            "shard_cid": pa.shard_cid,
                            "source_agent_pubkey_hex": pa.source_agent_pubkey.hex(),
                            # sp998 — the worker's node_id, derived from its pubkey
                            # the SAME way the worker self-assigns it. Settlement
                            # keys worker payout legs by THIS (not the raw pubkey
                            # hex, which is neither a 0x address nor a node_id →
                            # unspendable: it can't bridge on-chain and never
                            # matches a peer's to_wallet==node_id ledger-sync
                            # credit, so the worker's FTNS was credited to a
                            # local-only dead wallet). Matches the aggregator leg,
                            # which already keys by node_id.
                            "source_agent_node_id": _node_id_for_pubkey(
                                pa.source_agent_pubkey,
                            ),
                            "creator_id": pa.creator_id,
                            "pcu_consumed": pa.pcu_consumed,
                        }
                        for pa in qo_result.participants
                    ],
                }
            else:
                # Legacy AgentForge path — preserved for any operator
                # running a non-orchestrator backend (e.g. test fixture).
                result = await node.agent_forge.run(
                    query=query,
                    budget_ftns=budget_ftns,
                    shard_cids=shard_cids,
                )

                if result is None:
                    raise HTTPException(
                        status_code=500,
                        detail="Forge pipeline returned no result",
                    )

            # Track privacy budget if confidential compute is active
            if (
                hasattr(node, 'privacy_budget')
                and node.privacy_budget
                and privacy_level_str != "none"
            ):
                epsilon_map = {"standard": 8.0, "high": 4.0, "maximum": 1.0}
                epsilon = epsilon_map.get(privacy_level_str, 8.0)
                node.privacy_budget.record_spend(epsilon, "forge_query", job_id)

            # Release escrow on success. Two paths:
            #   (a) QO swarm path: §4 step 6 settlement — split the
            #       prompter's compute budget across the N compute
            #       participants (one per shard) + the aggregator
            #       coordination fee. Operator-tunable aggregator
            #       share via PRSM_AGGREGATOR_SHARE_BPS env var
            #       (default 500 = 5%).
            #   (b) Legacy path: single-provider release (prompter's
            #       own node) — preserves pre-QO behavior for
            #       AgentForge backends.
            if escrow_entry and node._payment_escrow and result.get("status") == "success":
                try:
                    qo_participants = result.get("participants") or []
                    if qo_participants:
                        # Sprint 238 — PCU-weighted split (uniform
                        # fallback when any participant lacks PCU).
                        # See prsm.economy.split_compute. Per-
                        # participant compute_share_total share is
                        # weighted by `pcu_consumed` when telemetry
                        # is complete, else uniform.
                        import os as _os_for_share_bps
                        try:
                            agg_share_bps = int(_os_for_share_bps.environ.get(
                                "PRSM_AGGREGATOR_SHARE_BPS", "500",
                            ))
                        except ValueError:
                            agg_share_bps = 500
                        if not (0 <= agg_share_bps <= 10000):
                            agg_share_bps = 500
                        from prsm.economy.split_compute import (
                            compute_split_amounts,
                        )
                        splits, split_mode = compute_split_amounts(
                            participants=qo_participants,
                            aggregator_node_id=result["aggregator_node_id"],
                            total_budget=budget_ftns,
                            aggregator_share_bps=agg_share_bps,
                        )
                        # Sprint 240 — resolve source_agent_pubkey_hex
                        # → operator FTNS wallet via the opt-in
                        # ComputeWalletMap. Operators running N
                        # compute agents map all N pubkeys to their
                        # single wallet. Empty map = pure pass-
                        # through (v1 backward-compat).
                        from prsm.node.compute_wallet_map import (
                            ComputeWalletMap, resolve_splits,
                        )
                        wallet_map = ComputeWalletMap.from_env()
                        splits = resolve_splits(splits, wallet_map)
                        logger.info(
                            "forge release split (mode=%s, n=%d, "
                            "agg_bps=%d, wallet_map_size=%d) "
                            "for job %s",
                            split_mode, len(qo_participants),
                            agg_share_bps, len(wallet_map),
                            job_id[:8],
                        )
                        # sp997 — capture the return: release_escrow_split returns
                        # None (NOT raise) when it cannot settle (a real shortfall
                        # beyond float tolerance, an already-finalized escrow, or a
                        # rolled-back partial release). The float-overshoot strand
                        # is now fixed at the source (epsilon tolerance + clamp in
                        # release_escrow_split), but a genuine None must not stay
                        # SILENT while the job reports COMPLETED — escalate to a
                        # visible ERROR naming the job so the stranded escrow can be
                        # reconciled (the inner warning was buried at WARNING and
                        # the forge handler swallows settlement exceptions).
                        _split_txs = (
                            await node._payment_escrow.release_escrow_split(
                                job_id=job_id,
                                splits=splits,
                            )
                        )
                        if _split_txs is None:
                            logger.error(
                                "forge settlement UNSETTLED for job %s: "
                                "release_escrow_split returned None (escrow "
                                "not released — providers unpaid). Escrow "
                                "remains PENDING for reconciliation/refund; "
                                "do NOT treat the job's COMPLETED status as "
                                "proof of payment.", job_id[:8],
                            )

                        # Sprint 248 — on-chain content-access
                        # royalty leg. Off by default. Operator
                        # opts in via PRSM_ONCHAIN_CONTENT_ROYALTY_
                        # ENABLED=1 + sets the per-shard wei
                        # amount + operator eth address. Best-
                        # effort: dispatch failures log + don't
                        # break the forge response.
                        try:
                            import os as _os_for_royalty
                            _en = _os_for_royalty.environ.get(
                                "PRSM_ONCHAIN_CONTENT_ROYALTY_ENABLED",
                                "0",
                            ).strip()
                            if _en == "1":
                                _client = getattr(
                                    node, "_royalty_distributor_client",
                                    None,
                                )
                                _op_addr = _os_for_royalty.environ.get(
                                    "PRSM_CONTENT_ROYALTY_OPERATOR_ADDRESS",
                                    "",
                                ).strip()
                                _wei_raw = _os_for_royalty.environ.get(
                                    "PRSM_CONTENT_ROYALTY_PER_SHARD_WEI",
                                    "1000000000000000",  # 0.001 FTNS
                                ).strip()
                                try:
                                    _wei = int(_wei_raw)
                                except ValueError:
                                    _wei = 1_000_000_000_000_000
                                _index = getattr(
                                    node, "content_index", None,
                                )
                                _contributing = list(
                                    result.get("contributing_shards") or ()
                                )
                                if (
                                    _client is not None
                                    and _op_addr
                                    and _wei > 0
                                    and _index is not None
                                    and _contributing
                                ):
                                    from prsm.economy.onchain_content_royalty import (
                                        allocate_royalty_amounts,
                                        dispatch_content_access_royalties,
                                        royalty_dispatch_key as _royalty_key,
                                        keys_to_release as _royalty_keys_to_release,
                                    )
                                    # sp911 — idempotent dispatch. Atomically
                                    # claim a per-(job, shard) key on the
                                    # durable ledger (sp898 record_nonce)
                                    # BEFORE the on-chain tx, so a re-entry /
                                    # re-delivery of THIS settlement does not
                                    # re-pay. settlement_key = job_id: stable
                                    # for one execution (catches re-dispatch)
                                    # yet distinct across separate accesses
                                    # (which correctly pay separately). Claim
                                    # in the async caller; pass a sync result
                                    # lookup to the (sync) dispatcher.
                                    _claimed_keys: Dict[str, bool] = {}
                                    if node.ledger is not None:
                                        for _shard_cid in _contributing:
                                            _ck = _royalty_key(str(job_id), _shard_cid)
                                            try:
                                                _claimed_keys[_ck] = (
                                                    await node.ledger.record_nonce(
                                                        _ck, _op_addr,
                                                    )
                                                )
                                            except Exception:  # noqa: BLE001
                                                # fail-closed: don't dispatch
                                                # if the claim couldn't be made
                                                _claimed_keys[_ck] = False
                                    _royalty_claim_fn = (
                                        (lambda _k: _claimed_keys.get(_k, False))
                                        if node.ledger is not None else None
                                    )
                                    _royalty_settle_key = (
                                        str(job_id) if node.ledger is not None
                                        else None
                                    )
                                    # Sprint 257 — choose allocation
                                    # mode. uniform = each shard gets
                                    # PRSM_CONTENT_ROYALTY_PER_SHARD_
                                    # WEI (sprint-248 behavior).
                                    # rate_weighted = interpret env
                                    # var as TOTAL pool, split by
                                    # ContentRecord.royalty_rate.
                                    _mode = _os_for_royalty.environ.get(
                                        "PRSM_CONTENT_ROYALTY_ALLOCATION_MODE",
                                        "uniform",
                                    ).strip().lower()
                                    if _mode not in (
                                        "uniform", "rate_weighted",
                                    ):
                                        _mode = "uniform"
                                    if _mode == "rate_weighted":
                                        # Treat per-shard env as the
                                        # TOTAL pool size when rate-
                                        # weighted: simpler operator
                                        # mental model than a separate
                                        # env var, since the per-shard
                                        # interpretation only made
                                        # sense in uniform mode.
                                        _pool = _wei * len(_contributing)
                                        _amounts = allocate_royalty_amounts(
                                            shards=_contributing,
                                            content_index=_index,
                                            total_pool_wei=_pool,
                                            mode="rate_weighted",
                                        )
                                        _royalty_results = (
                                            dispatch_content_access_royalties(
                                                shards=_contributing,
                                                content_index=_index,
                                                royalty_client=_client,
                                                serving_node_address=_op_addr,
                                                gross_amounts_wei=_amounts,
                                                settlement_key=_royalty_settle_key,
                                                claim_fn=_royalty_claim_fn,
                                            )
                                        )
                                    else:
                                        _royalty_results = (
                                            dispatch_content_access_royalties(
                                                shards=_contributing,
                                                content_index=_index,
                                                royalty_client=_client,
                                                serving_node_address=_op_addr,
                                                gross_per_shard_wei=_wei,
                                                settlement_key=_royalty_settle_key,
                                                claim_fn=_royalty_claim_fn,
                                            )
                                        )
                                    # sp918 — release the idempotency claim for
                                    # shards whose on-chain tx definitively did
                                    # NOT pay (reverted = atomic rollback;
                                    # failed = never broadcast), so a future
                                    # settlement can re-dispatch instead of the
                                    # royalty being permanently
                                    # skipped_already_dispatched. keys_to_release
                                    # NEVER returns a 'pending' shard (still in
                                    # the mempool — releasing would re-open the
                                    # double-pay window). Best-effort.
                                    if (
                                        node.ledger is not None
                                        and _royalty_settle_key is not None
                                    ):
                                        for _rk in _royalty_keys_to_release(
                                            _royalty_results,
                                            _royalty_settle_key,
                                        ):
                                            try:
                                                await node.ledger.release_nonce(
                                                    _rk,
                                                )
                                            except Exception:  # noqa: BLE001
                                                logger.warning(
                                                    "sp918: release_nonce "
                                                    "failed for %s", _rk,
                                                )
                                    _sent = sum(
                                        1 for r in _royalty_results
                                        if r.status == "sent"
                                    )
                                    _skipped = sum(
                                        1 for r in _royalty_results
                                        if r.status.startswith("skipped")
                                    )
                                    _failed = sum(
                                        1 for r in _royalty_results
                                        if r.status == "failed"
                                    )
                                    # sp918 — count the distinct non-success
                                    # outcomes so telemetry isn't blind to them
                                    # (the BroadcastFailedError/Exception split
                                    # moved generic errors out of "failed").
                                    _reverted = sum(
                                        1 for r in _royalty_results
                                        if r.status == "reverted"
                                    )
                                    _pending = sum(
                                        1 for r in _royalty_results
                                        if r.status == "pending"
                                    )
                                    _errored = sum(
                                        1 for r in _royalty_results
                                        if r.status == "error"
                                    )
                                    logger.info(
                                        "on-chain content royalty "
                                        "dispatch for job %s: sent=%d "
                                        "skipped=%d failed=%d reverted=%d "
                                        "pending=%d error=%d",
                                        job_id[:8], _sent, _skipped,
                                        _failed, _reverted, _pending,
                                        _errored,
                                    )
                                    # Sprint 249 — append each
                                    # per-shard outcome to the audit
                                    # ring so operators can inspect
                                    # via /admin/royalty-dispatch-
                                    # history. Best-effort: ring
                                    # failures stay invisible.
                                    _ring = getattr(
                                        node,
                                        "_royalty_dispatch_ring",
                                        None,
                                    )
                                    if _ring is not None:
                                        for _r in _royalty_results:
                                            try:
                                                # Sprint 258 — surface
                                                # which allocation
                                                # policy produced this
                                                # row so post-hoc audit
                                                # can distinguish a
                                                # 50/50 uniform-2-shard
                                                # split from an equally
                                                # weighted rate_weighted
                                                # outcome.
                                                _ring.append(
                                                    job_id=job_id,
                                                    cid=_r.cid,
                                                    status=_r.status,
                                                    tx_hash=_r.tx_hash,
                                                    gross_wei=_wei,
                                                    error=_r.error,
                                                    allocation_mode=_mode,
                                                )
                                            except Exception:  # noqa
                                                pass
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "on-chain content royalty dispatch "
                                "raised (job %s): %s — best-effort, "
                                "ignored", job_id[:8], exc,
                            )
                    else:
                        # Legacy path
                        await node._payment_escrow.release_escrow(
                            job_id=job_id,
                            provider_id=node.identity.node_id,
                        )
                except Exception as e:
                    logger.warning(f"Forge escrow release failed: {e}")

            # Extract response text based on route. The QO path already
            # set "response" + "route" above; legacy paths populate
            # "route" via AgentForge result keys.
            route = result.get("route", "unknown")
            if route == "qo_swarm":
                response_text = result.get("response", str(result))
            elif route == "direct_llm":
                response_text = result.get("response", str(result))
            elif route == "swarm":
                output = result.get("aggregated_output", {})
                response_text = str(output.get("shard_outputs", output))
            elif route == "single_agent":
                agent_result = result.get("result", {})
                response_text = str(agent_result)
            else:
                response_text = str(result)

            traces_count = len(getattr(node.agent_forge, "traces", []))

            # B8 async-dispatch follow-on: record COMPLETED to
            # JobHistoryStore so /compute/status surfaces the full
            # result. Best-effort.
            if hasattr(node, "_job_history") and node._job_history is not None:
                try:
                    from prsm.node.job_history import (
                        JobHistoryRecord as _JobRec,
                        JobStatus as _JobStat,
                    )
                    node._job_history.put(_JobRec(
                        job_id=job_id,
                        query=query,
                        status=_JobStat.COMPLETED,
                        started_at=_job_started_at,
                        completed_at=_time_for_history.time(),
                        route=route,
                        response=response_text,
                        aggregator_node_id=result.get("aggregator_node_id"),
                        contributing_shards=tuple(
                            result.get("contributing_shards") or ()
                        ),
                        participants=tuple(
                            result.get("participants") or ()
                        ),
                        traces_collected=traces_count,
                    ))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "JobHistoryStore put (COMPLETED) failed for "
                        "job_id=%s: %s",
                        job_id, exc,
                    )

            return {
                "job_id": job_id,
                "query": query,
                "route": route,
                "response": response_text,
                "result": result,
                "budget_ftns": budget_ftns,
                "traces_collected": traces_count,
            }

        except HTTPException:
            raise
        except Exception as e:
            # Refund escrow on failure
            if escrow_entry and node._payment_escrow:
                try:
                    await node._payment_escrow.refund_escrow(job_id, str(e))
                except Exception:
                    pass
            # B8 async-dispatch follow-on: record FAILED.
            if hasattr(node, "_job_history") and node._job_history is not None:
                try:
                    from prsm.node.job_history import (
                        JobHistoryRecord as _JobRec,
                        JobStatus as _JobStat,
                    )
                    node._job_history.put(_JobRec(
                        job_id=job_id,
                        query=query,
                        status=_JobStat.FAILED,
                        started_at=_job_started_at,
                        completed_at=_time_for_history.time(),
                        error=str(e),
                    ))
                except Exception as hist_exc:  # noqa: BLE001
                    logger.warning(
                        "JobHistoryStore put (FAILED) failed for "
                        "job_id=%s: %s",
                        job_id, hist_exc,
                    )
            # Sprint 175 — map "no shards above similarity threshold"
            # from the orchestrator to 404 instead of 500. That error
            # is a query/content mismatch (operator-fixable: upload
            # relevant content or refine query), not a server fault.
            _err_str = str(e)
            if "no shards above similarity threshold" in _err_str:
                logger.info(
                    "Forge query found no matching shards: %s", _err_str,
                )
                raise HTTPException(
                    status_code=404,
                    detail=(
                        "No content shards above the similarity "
                        "threshold for this query. Upload relevant "
                        "content to this node or refine the query."
                    ),
                )
            logger.error(f"Forge pipeline error: {e}")
            raise HTTPException(status_code=500, detail=f"Forge pipeline error: {str(e)}")

    # Renamed from `_ArbitrationPreviewRequest` for OpenAPI hygiene.
    class ArbitrationPreviewRequest(BaseModel):
        record_id: str
        decision: str
        by_council: List[str]

    @app.post("/content/arbitration/preview-resolution")
    async def post_arbitration_preview(
        body: ArbitrationPreviewRequest,
    ) -> Dict[str, Any]:
        """Composer-only preview of what queue.resolve() WOULD
        do for the given (record_id, decision, by_council).

        DOES NOT call queue.resolve(). Returns the would-be-applied
        resolution shape + conflict-with-existing detection +
        operator action hint to sign on-chain governance proposal
        separately.

        Status:
          503 — arbitration_queue not wired
          404 — record_id unknown
          422 — invalid decision (not in upheld_parent /
                rejected_parent / insufficient) OR empty by_council
          200 — DRY_RUN preview artifact
        """
        from prsm.data.dedup.arbitration import ArbitrationDecision

        # Validation
        try:
            decision_enum = ArbitrationDecision(body.decision)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"invalid decision {body.decision!r}; "
                    f"allowed: "
                    f"{[d.value for d in ArbitrationDecision]}"
                ),
            )
        if not body.by_council:
            raise HTTPException(
                status_code=422,
                detail="by_council must be a non-empty list",
            )

        queue = getattr(node, "_arbitration_queue", None)
        if queue is None:
            raise HTTPException(
                status_code=503,
                detail="ArbitrationQueue not initialized on this node.",
            )

        try:
            rec = await queue.get(body.record_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("arbitration get raised: %s", exc)
            raise HTTPException(status_code=502, detail=str(exc))
        if rec is None:
            raise HTTPException(
                status_code=404,
                detail=f"No arbitration record for id={body.record_id!r}",
            )

        try:
            current_resolution = await queue.get_resolution(body.record_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("arbitration get_resolution raised: %s", exc)
            current_resolution = None

        conflict = False
        if current_resolution is not None:
            existing_decision = current_resolution.get("decision")
            conflict = existing_decision != decision_enum.value

        return {
            "status": "DRY_RUN",
            "record": rec.to_dict(),
            "proposed": {
                "decision": decision_enum.value,
                "by_council": list(body.by_council),
            },
            "current_resolution": current_resolution,
            "conflict_with_existing": conflict,
            "note": (
                "Composer-only artifact; does NOT call queue.resolve(). "
                "Council member confirms intent + signs on-chain "
                "governance proposal separately. Local-resolve auth "
                "model pending council ratification."
            ),
        }

    @app.get("/content/arbitration/queue/{record_id}")
    async def get_arbitration_record(record_id: str) -> Dict[str, Any]:
        """Detail view of a single arbitration record + its
        resolution state. Council members use this to fetch full
        context before signing on-chain governance proposals.

        Status:
          503 — arbitration_queue not wired
          404 — record_id unknown
          502 — queue access raised
          200 — {record, resolution, status}
                where status is "resolved" if resolution present,
                else "pending"
        """
        queue = getattr(node, "_arbitration_queue", None)
        if queue is None:
            raise HTTPException(
                status_code=503,
                detail="ArbitrationQueue not initialized on this node.",
            )
        try:
            rec = await queue.get(record_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("arbitration get raised: %s", exc)
            raise HTTPException(status_code=502, detail=str(exc))
        if rec is None:
            raise HTTPException(
                status_code=404,
                detail=f"No arbitration record for id={record_id!r}",
            )
        try:
            resolution = await queue.get_resolution(record_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("arbitration get_resolution raised: %s", exc)
            resolution = None
        return {
            "record": rec.to_dict(),
            "resolution": resolution,
            "status": "resolved" if resolution is not None else "pending",
        }

    @app.get("/content/arbitration/queue")
    async def get_arbitration_queue() -> Dict[str, Any]:
        """List pending content-attribution disputes awaiting
        council adjudication. Closes the operator-UX gap from
        PRSM-PROV-1 Item 6: the FilesystemArbitrationQueue
        persists disputes at ``~/.prsm/arbitration_queue/`` but
        operators previously had no way to see them without
        scanning the directory by hand.

        Backs the ``prsm_arbitration_status`` MCP tool.

        Status:
          503 — arbitration_queue not wired
          502 — list_pending raised (disk error, lock timeout)
          200 — {pending: [...], total: N}
        """
        queue = getattr(node, "_arbitration_queue", None)
        if queue is None:
            raise HTTPException(
                status_code=503,
                detail="ArbitrationQueue not initialized on this node.",
            )
        try:
            records = await queue.list_pending()
        except Exception as exc:  # noqa: BLE001
            logger.warning("list_pending raised: %s", exc)
            raise HTTPException(
                status_code=502,
                detail=str(exc),
            )
        return {
            "pending": [r.to_dict() for r in records],
            "total": len(records),
        }

    @app.post("/compute/cleanup-stale")
    async def post_compute_cleanup_stale() -> Dict[str, Any]:
        """Manually trigger PaymentEscrow.cleanup_expired_escrows.

        Refunds any PENDING escrow whose age exceeds
        ``default_timeout``. Returns ``{cleaned: N}``. Use when
        operators need an immediate sweep without waiting for
        the 10-min periodic loop (e.g., after lowering
        PRSM_ESCROW_TIMEOUT_SEC, draining for maintenance).

        Status:
          503 — PaymentEscrow not wired
          502 — cleanup_expired_escrows raised
          200 — cleaned count returned
        """
        escrow_svc = getattr(node, "_payment_escrow", None)
        if escrow_svc is None:
            raise HTTPException(
                status_code=503,
                detail="PaymentEscrow not initialized on this node.",
            )
        try:
            cleaned = await escrow_svc.cleanup_expired_escrows()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "cleanup_expired_escrows raised: %s", exc,
            )
            raise HTTPException(
                status_code=502,
                detail=str(exc),
            )
        return {"cleaned": cleaned}

    @app.get("/compute/jobs")
    async def compute_jobs_list(
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        route: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Paginated operator-side job list. Backs the
        ``prsm_jobs_list`` MCP tool.

        Query params:
          - status: optional JobStatus filter (in_progress |
            completed | failed | cancelled).
          - route: optional route filter (forge | inference |
            inference_stream | qo_swarm | direct_llm | swarm).
            Sprint 260 — operators scope to a single compute path.
          - limit: page size (1..100, default 50).
          - offset: pagination offset, default 0.

        Returns 503 if JobHistoryStore not wired, 422 on
        validation errors. 200 returns
        {jobs: [...], total: N, offset: X, limit: Y}.
        """
        from prsm.node.job_history import JobStatus

        history = getattr(node, "_job_history", None)
        if history is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "JobHistoryStore is not initialized on this "
                    "node. Cannot list jobs."
                ),
            )

        # Validate query params.
        if offset < 0:
            raise HTTPException(
                status_code=422,
                detail=f"offset must be >= 0, got {offset}",
            )
        if limit <= 0 or limit > 100:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"limit must be in [1, 100], got {limit}. "
                    f"Use offset to paginate further."
                ),
            )
        status_filter = None
        if status is not None:
            try:
                status_filter = JobStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"invalid status {status!r}. Allowed: "
                        f"{[s.value for s in JobStatus]}"
                    ),
                )

        # Sprint 260 — route validation (sanity-only; doesn't
        # gate, since /compute/forge legacy paths emit route
        # values that aren't enumerable here).
        if route is not None and len(route) > 64:
            raise HTTPException(
                status_code=422,
                detail=f"route length must be <= 64, got {len(route)}",
            )

        records = history.list(
            status_filter=status_filter, limit=limit, offset=offset,
            route_filter=route,
        )
        total = history.count(
            status_filter=status_filter, route_filter=route,
        )
        return {
            "jobs": [r.to_dict() for r in records],
            "total": total,
            "offset": offset,
            "limit": limit,
        }

    @app.get("/compute/status/{job_id}")
    async def compute_status(job_id: str) -> Dict[str, Any]:
        """Look up the status of a /compute/forge job by its job_id.

        Backs the ``prsm_agent_status`` MCP tool. Two-tier lookup:

        1. **JobHistoryStore** (richer): pipeline state — route,
           response, aggregator, participants, traces, error if
           failed. Always preferred when present.
        2. **PaymentEscrow** (fallback): payment-leg lifecycle
           (pending / released / refunded / disputed + amount +
           timing + provider winner). Used when history doesn't
           know about the job (e.g. node-restart eviction, or
           budget=0 test fixtures that touched the escrow but
           weren't recorded in history).

        The two surfaces compose: a `compute` block (from history)
        and an `escrow` block (from payment_escrow) appear together
        when both are populated. At least one must be available or
        the endpoint returns 404.
        """
        history = getattr(node, "_job_history", None)
        escrow_svc = getattr(node, "_payment_escrow", None)

        if history is None and escrow_svc is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Neither JobHistoryStore nor PaymentEscrow "
                    "is initialized on this node."
                ),
            )

        history_record = None
        if history is not None:
            try:
                history_record = history.get(job_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "JobHistoryStore.get raised for job_id=%s: %s",
                    job_id, exc,
                )

        escrow = None
        if escrow_svc is not None:
            escrow = escrow_svc.get_escrow(job_id)

        if history_record is None and escrow is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No history or escrow record for job_id={job_id!r}. "
                    f"Either the job never ran on this node, or its "
                    f"history was evicted (LRU-bounded) and it ran "
                    f"without a locked escrow."
                ),
            )

        body: Dict[str, Any] = {"job_id": job_id}
        if history_record is not None:
            body["compute"] = history_record.to_dict()
        if escrow is not None:
            body["escrow"] = {
                "escrow_id": escrow.escrow_id,
                "requester_id": escrow.requester_id,
                "amount_ftns": escrow.amount,
                "status": escrow.status.value,
                "provider_winner": escrow.provider_winner,
                "tx_lock": escrow.tx_lock,
                "tx_release": escrow.tx_release,
                "created_at": escrow.created_at,
                "completed_at": escrow.completed_at,
                "metadata": dict(escrow.metadata or {}),
            }
        return body

    @app.get("/compute/status/{job_id}/stream")
    async def compute_status_stream(job_id: str):
        """SSE-streaming sibling of /compute/status/{job_id}.

        Closes the last B8 deferred sub-item: clients can render
        IN_PROGRESS → COMPLETED transitions live without polling
        the GET endpoint.

        Polling-based v1: server polls JobHistoryStore +
        PaymentEscrow at PRSM_STATUS_STREAM_POLL_SEC interval
        (default 0.5s), emits an SSE event whenever the snapshot
        changes (de-duplicated by JSON equality), and closes the
        connection on terminal status (history COMPLETED/FAILED/
        CANCELLED OR escrow RELEASED/REFUNDED) OR after
        PRSM_STATUS_STREAM_TIMEOUT_SEC (default 1800s = 30min).

        Wire format:
            event: status
            data: {"job_id": "...", "history": {...},
                   "escrow": {...}}

            event: terminal
            data: {"job_id": "...", "reason": "completed"|
                   "history_terminal"|"escrow_terminal"|"timeout"}

        `event: terminal` is the only terminal event — the server
        closes the connection after emitting it. Clients can
        re-subscribe to keep watching past a timeout.
        """
        from prsm.node.job_history import JobStatus
        from prsm.node.payment_escrow import EscrowStatus

        history = getattr(node, "_job_history", None)
        escrow_svc = getattr(node, "_payment_escrow", None)

        if history is None and escrow_svc is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Neither JobHistoryStore nor PaymentEscrow "
                    "is initialized on this node."
                ),
            )

        # Initial existence check — 404 fast if neither has the job.
        initial_history = history.get(job_id) if history else None
        initial_escrow = (
            escrow_svc.get_escrow(job_id) if escrow_svc else None
        )
        if initial_history is None and initial_escrow is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No history or escrow record for job_id={job_id!r}."
                ),
            )

        poll_raw = os.getenv("PRSM_STATUS_STREAM_POLL_SEC", "0.5").strip()
        try:
            poll_sec = float(poll_raw) if poll_raw else 0.5
            if poll_sec <= 0:
                poll_sec = 0.5
        except ValueError:
            poll_sec = 0.5

        timeout_raw = os.getenv(
            "PRSM_STATUS_STREAM_TIMEOUT_SEC", "1800",
        ).strip()
        try:
            timeout_sec = float(timeout_raw) if timeout_raw else 1800.0
            if timeout_sec <= 0:
                timeout_sec = 1800.0
        except ValueError:
            timeout_sec = 1800.0

        TERMINAL_HISTORY = {
            JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED,
        }
        TERMINAL_ESCROW = {
            EscrowStatus.RELEASED, EscrowStatus.REFUNDED,
        }

        def _snapshot():
            h = history.get(job_id) if history else None
            e = escrow_svc.get_escrow(job_id) if escrow_svc else None
            payload = {"job_id": job_id}
            if h is not None:
                payload["history"] = h.to_dict()
            if e is not None:
                payload["escrow"] = {
                    "escrow_id": e.escrow_id,
                    "status": e.status.value,
                    "amount_ftns": e.amount,
                    "completed_at": e.completed_at,
                }
            return payload, h, e

        def _terminal_reason(h, e):
            if h is not None and h.status in TERMINAL_HISTORY:
                return (
                    h.status.value if h.status == JobStatus.COMPLETED
                    else "history_terminal"
                )
            if e is not None and e.status in TERMINAL_ESCROW:
                return "escrow_terminal"
            return None

        # Pre-built initial payload from the existence-check reads
        # so the first snapshot doesn't double-poll.
        def _build_payload(h, e):
            payload = {"job_id": job_id}
            if h is not None:
                payload["history"] = h.to_dict()
            if e is not None:
                payload["escrow"] = {
                    "escrow_id": e.escrow_id,
                    "status": e.status.value,
                    "amount_ftns": e.amount,
                    "completed_at": e.completed_at,
                }
            return payload

        async def _generate():
            import asyncio as _asyncio
            last_payload_json = None
            start = _time_for_history.time()
            # Use the existence-check reads as the first snapshot
            # to avoid a redundant get() on the loop's first pass.
            h, e = initial_history, initial_escrow
            first_pass = True
            while True:
                if not first_pass:
                    try:
                        _payload, h, e = _snapshot()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "compute_status_stream snapshot raised for "
                            "job_id=%s: %s", job_id, exc,
                        )
                        yield (
                            "event: error\n"
                            f"data: {json.dumps({'error': str(exc)})}\n\n"
                        )
                        return
                payload = _build_payload(h, e)
                payload_json = json.dumps(payload, sort_keys=True)
                if payload_json != last_payload_json:
                    yield f"event: status\ndata: {payload_json}\n\n"
                    last_payload_json = payload_json
                reason = _terminal_reason(h, e)
                if reason is not None:
                    yield (
                        "event: terminal\n"
                        f"data: {json.dumps({'job_id': job_id, 'reason': reason})}\n\n"
                    )
                    return
                if _time_for_history.time() - start >= timeout_sec:
                    yield (
                        "event: terminal\n"
                        f"data: {json.dumps({'job_id': job_id, 'reason': 'timeout'})}\n\n"
                    )
                    return
                first_pass = False
                await _asyncio.sleep(poll_sec)

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
        )

    @app.post("/compute/cancel/{job_id}")
    async def compute_cancel(job_id: str) -> Dict[str, Any]:
        """Cancel a /compute/forge job: mark history.status as
        CANCELLED + refund any PENDING escrow.

        v1 caveat: in-flight Python coroutines are NOT interrupted.
        Cancellation marks intent + refunds the budget. If the
        coroutine completes successfully later, its
        release_escrow_split call will race-lose against the
        REFUNDED escrow and raise EscrowAlreadyFinalizedError —
        the correct race-loss outcome.

        Status:
          503 — neither JobHistoryStore nor PaymentEscrow wired.
          404 — neither has the job.
          409 — job already terminal (COMPLETED/FAILED/CANCELLED on
                history side, RELEASED/REFUNDED on escrow side).
          200 — cancelled. Response surfaces history_cancelled,
                escrow_refunded, refund_amount_ftns.
        """
        from prsm.node.job_history import (
            JobHistoryRecord, JobStatus,
        )
        from prsm.node.payment_escrow import (
            EscrowAlreadyFinalizedError, EscrowStatus,
        )

        history = getattr(node, "_job_history", None)
        escrow_svc = getattr(node, "_payment_escrow", None)

        if history is None and escrow_svc is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Neither JobHistoryStore nor PaymentEscrow "
                    "is initialized on this node."
                ),
            )

        # Look up both surfaces.
        history_record = None
        if history is not None:
            try:
                history_record = history.get(job_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "JobHistoryStore.get raised for job_id=%s: %s",
                    job_id, exc,
                )

        escrow = None
        if escrow_svc is not None:
            try:
                escrow = escrow_svc.get_escrow(job_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "PaymentEscrow.get_escrow raised for job_id=%s: %s",
                    job_id, exc,
                )

        if history_record is None and escrow is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No history or escrow record for job_id={job_id!r}; "
                    f"nothing to cancel."
                ),
            )

        # 409 — already terminal on either surface.
        if history_record is not None and history_record.status in {
            JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED,
        }:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Job {job_id!r} is already in terminal state "
                    f"{history_record.status.value!r}; cannot cancel."
                ),
            )
        if escrow is not None and escrow.status in {
            EscrowStatus.RELEASED, EscrowStatus.REFUNDED,
        }:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Escrow for job {job_id!r} is already "
                    f"{escrow.status.value!r}; cannot cancel."
                ),
            )

        # Mark history CANCELLED.
        history_cancelled = False
        if history_record is not None:
            try:
                cancelled_record = JobHistoryRecord(
                    job_id=history_record.job_id,
                    query=history_record.query,
                    status=JobStatus.CANCELLED,
                    started_at=history_record.started_at,
                    completed_at=_time_for_history.time(),
                    route=history_record.route,
                    response=history_record.response,
                    aggregator_node_id=history_record.aggregator_node_id,
                    contributing_shards=history_record.contributing_shards,
                    participants=history_record.participants,
                    traces_collected=history_record.traces_collected,
                    error=history_record.error,
                )
                history.put(cancelled_record)
                history_cancelled = True
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "JobHistoryStore.put (CANCELLED) failed for "
                    "job_id=%s: %s", job_id, exc,
                )

        # Refund the escrow if PENDING.
        escrow_refunded = False
        refund_amount_ftns = 0.0
        if escrow is not None and escrow.status == EscrowStatus.PENDING:
            try:
                ok = await escrow_svc.refund_escrow(
                    job_id, reason="operator-cancelled via /compute/cancel",
                )
                escrow_refunded = bool(ok)
                if escrow_refunded:
                    refund_amount_ftns = escrow.amount
            except EscrowAlreadyFinalizedError:
                # Race-loss: release_escrow_split landed between
                # our check + our refund. Treat as best-effort
                # cancel; history already CANCELLED.
                logger.info(
                    "Escrow for job_id=%s race-finalized during "
                    "cancel; treating as race-loss.", job_id,
                )
                escrow_refunded = False
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "PaymentEscrow.refund_escrow raised for "
                    "job_id=%s: %s", job_id, exc,
                )
                escrow_refunded = False

        return {
            "job_id": job_id,
            "history_cancelled": history_cancelled,
            "escrow_refunded": escrow_refunded,
            "refund_amount_ftns": refund_amount_ftns,
        }

    # Sprint 237 — inference cost-quote endpoint. Pre-fix
    # prsm_inference was the only path to discover cost, but
    # submitting locked escrow. Pre-flight cost discovery now
    # surfaces InferenceExecutor.estimate_cost() over HTTP
    # without executing.
    @app.post("/compute/inference/quote")
    async def compute_inference_quote(
        body: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        """Return the cost estimate for an inference request
        WITHOUT executing it. Same body shape as /compute/inference
        minus billing — caller doesn't need budget_ftns."""
        prompt = body.get("prompt", "")
        if not prompt:
            # Sprint 536 F66 fix: schema hint
            raise HTTPException(
                status_code=400,
                detail=(
                    "Missing 'prompt' field. Expected body: "
                    "{\"prompt\": \"<text>\", \"model_id\": "
                    "\"mock-llama-3-8b\", \"max_tokens\": <int>}"
                ),
            )
        # Sprint 198 cap inherited.
        _ip_raw = os.environ.get(
            "PRSM_MAX_INFERENCE_PROMPT_BYTES", "",
        ).strip()
        try:
            _ip_cap = int(_ip_raw) if _ip_raw else 100 * 1024
            if _ip_cap <= 0:
                raise ValueError("non-positive")
        except (ValueError, TypeError):
            _ip_cap = 100 * 1024
        _ip_bytes = len(prompt.encode("utf-8"))
        if _ip_bytes > _ip_cap:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"prompt size {_ip_bytes} bytes exceeds "
                    f"PRSM_MAX_INFERENCE_PROMPT_BYTES cap of "
                    f"{_ip_cap}."
                ),
            )
        model_id = body.get("model_id", "")
        if not model_id:
            raise HTTPException(
                status_code=400, detail="Missing 'model_id' field",
            )

        executor = getattr(node, "inference_executor", None)
        if executor is None:
            raise HTTPException(
                status_code=503,
                detail="Inference executor not initialized.",
            )

        from decimal import Decimal
        from prsm.compute.inference import (
            ContentTier,
            InferenceRequest,
        )
        from prsm.compute.inference.executor import (
            UnsupportedModelError,
        )
        from prsm.compute.tee.models import PrivacyLevel

        _privacy_raw = body.get("privacy_tier", "none")  # sp1234 — honest default (DP is experimental/best-effort)
        try:
            privacy_level = PrivacyLevel(_privacy_raw)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"privacy_tier must be one of "
                    f"{[lvl.value for lvl in PrivacyLevel]}; "
                    f"got {_privacy_raw!r}."
                ),
            )
        _content_raw = body.get("content_tier", "A")
        try:
            content_tier = ContentTier(_content_raw)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"content_tier must be one of "
                    f"{[t.value for t in ContentTier]}; "
                    f"got {_content_raw!r}."
                ),
            )

        # Quote doesn't need real budget; pass 0 since estimate_cost
        # ignores it.
        try:
            request = InferenceRequest(
                prompt=prompt,
                model_id=model_id,
                budget_ftns=Decimal("0"),
                privacy_tier=privacy_level,
                content_tier=content_tier,
                max_tokens=body.get("max_tokens"),
                temperature=body.get("temperature"),
                requester_node_id=(
                    node.identity.node_id if node.identity else None
                ),
            )
        except (ValueError, TypeError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid request: {e}",
            )
        try:
            cost = await executor.estimate_cost(request)
        except UnsupportedModelError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=f"estimate_cost() failed: {e}",
            )
        # Sprint 262 — surface projected ε spend so end-users can
        # plan against the privacy budget (the actual /compute/
        # inference pre-flight gate uses the same value). ε=0 for
        # PrivacyLevel.NONE.
        if privacy_level == PrivacyLevel.NONE:
            epsilon_estimated = 0.0
        else:
            try:
                epsilon_estimated = (
                    PrivacyLevel.config_for_level(privacy_level).epsilon
                )
            except Exception:  # noqa: BLE001
                epsilon_estimated = None
        # Surface remaining privacy-budget so the caller can spot
        # "I have enough FTNS but not enough ε" before submitting.
        privacy_budget_remaining: Optional[float] = None
        try:
            pb = getattr(node, "privacy_budget", None)
            if pb is not None:
                privacy_budget_remaining = float(pb.remaining())
        except Exception:  # noqa: BLE001
            privacy_budget_remaining = None

        return {
            "model_id": model_id,
            "cost_ftns": str(cost),
            "privacy_tier": privacy_level.value,
            "content_tier": content_tier.value,
            "epsilon_estimated": epsilon_estimated,
            "privacy_budget_remaining": privacy_budget_remaining,
        }

    @app.post("/compute/inference")
    async def compute_inference(
        request: Request,
        body: Dict[str, Any] = {},
        x_corp_capability: Optional[str] = Header(
            default=None, alias="X-CORP-Capability",
        ),
        x_corp_redemption: Optional[str] = Header(
            default=None, alias="X-CORP-Redemption",
        ),
    ) -> Dict[str, Any]:
        """Run TEE-attested model inference with verifiable signed receipts.

        Phase 3.x.1 Task 5 — wires the prsm.compute.inference module
        (Tasks 1-2) to the HTTP layer so the prsm_inference MCP tool
        (Task 6) can route through this endpoint.

        Mirrors /compute/forge: validate → lock escrow → run executor →
        sign receipt → track privacy budget → release/refund escrow.

        POST body:
        {
            "prompt": "...",                  // required
            "model_id": "mock-llama-3-8b",    // required
            "budget_ftns": 1.0,
            "privacy_tier": "standard",        // none|standard|high|maximum
            "content_tier": "A",               // A|B|C
            "max_tokens": 256,                 // optional
            "temperature": 0.7,                // optional
        }
        """
        # Sprint 756 — active-window gate. Fire BEFORE rate-limit
        # check so we don't even count the request against the
        # operator's bucket if the daemon shouldn't be serving.
        _check_active_window_or_503()
        # Sprint 774 — preemption gate: refuse new work when this
        # cloud-spot node has been signaled for termination so peers
        # route past us during the ~2min warning window.
        _check_not_preempted_or_503()
        # PRSM_INFERENCE_MAX_RPS_PER_REQUESTER rate limiting (DoS
        # protection). Independent bucket from /compute/forge so
        # operators can tune the two endpoints separately.
        _rps_raw = os.getenv(
            "PRSM_INFERENCE_MAX_RPS_PER_REQUESTER", "",
        ).strip()
        if _rps_raw:
            try:
                _rps = float(_rps_raw)
                if _rps > 0:
                    from prsm.node.rate_limiter import (
                        get_or_build_bucket,
                    )
                    bucket = get_or_build_bucket(_rps, name="inference")
                    if bucket is not None:
                        # Sprint 741 F69 fix: pre-741, `requester`
                        # was `node.identity.node_id` — the LOCAL
                        # daemon's own id, a constant. All callers
                        # shared one bucket → "per requester" was
                        # effectively GLOBAL. One legit client could
                        # be starved by another attacker's traffic
                        # hitting the same bucket. Now keyed by HTTP
                        # client identity (proxy-aware), so an
                        # attacker spamming from one IP can't deny
                        # service to legit callers from other IPs.
                        requester = _resolve_requester_key(request)
                        if not bucket.try_consume(requester):
                            retry = bucket.retry_after(requester)
                            raise HTTPException(
                                status_code=429,
                                detail=(
                                    f"Rate limit exceeded for "
                                    f"requester {requester[:24]}... "
                                    f"on /compute/inference "
                                    f"(cap {_rps}/sec). Retry "
                                    f"after {retry:.2f}s."
                                ),
                                headers={
                                    "Retry-After": f"{retry:.2f}",
                                },
                            )
            except ValueError:
                logger.warning(
                    "PRSM_INFERENCE_MAX_RPS_PER_REQUESTER=%r not "
                    "numeric; rate limiting disabled", _rps_raw,
                )

        # Sprint 154 — body validation BEFORE executor availability
        # check, mirroring sprint 153 /compute/forge fix. Pre-fix
        # all validation errors leaked through to 503 (executor
        # unwired) because the 503 fired first.

        prompt = body.get("prompt", "")
        if not prompt:
            # Sprint 536 F66 fix: schema hint
            raise HTTPException(
                status_code=400,
                detail=(
                    "Missing 'prompt' field. Expected body: "
                    "{\"prompt\": \"<text>\", \"model_id\": "
                    "\"mock-llama-3-8b\", \"budget_ftns\": "
                    "<float>, \"max_tokens\": <int>}"
                ),
            )

        # Sprint 198 — cap prompt size (DoS surface; sibling
        # /compute/query has the same cap via PRSM_MAX_QUERY_BYTES).
        # Default 100KB. Override: PRSM_MAX_INFERENCE_PROMPT_BYTES.
        _ip_raw = os.environ.get(
            "PRSM_MAX_INFERENCE_PROMPT_BYTES", "",
        ).strip()
        try:
            _ip_cap = int(_ip_raw) if _ip_raw else 100 * 1024
            if _ip_cap <= 0:
                raise ValueError("non-positive")
        except (ValueError, TypeError):
            _ip_cap = 100 * 1024
        _ip_bytes = len(prompt.encode("utf-8"))
        if _ip_bytes > _ip_cap:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"prompt size {_ip_bytes} bytes exceeds "
                    f"PRSM_MAX_INFERENCE_PROMPT_BYTES cap of "
                    f"{_ip_cap}. Trim the prompt or have the "
                    f"operator raise the cap."
                ),
            )

        model_id = body.get("model_id", "")
        if not model_id:
            raise HTTPException(status_code=400, detail="Missing 'model_id' field")

        # Sprint 271 — operator content filter (BEFORE budget
        # validation / escrow / privacy budget / executor).
        # Refuses with 451 if prompt matches blocked_input_patterns
        # OR model_id is in blocked_model_tags.
        _enforce_content_filter(prompt=prompt, model_id=model_id)

        # Sprint 305a — TEE policy enforcement (Vision §7
        # Enterprise Confidentiality Mode layer 3). Refuses
        # with 412 if the request's optional tee_policy is
        # not satisfied by this node's own attestation.
        # Absent field = no gating, fully backwards-compat.
        _enforce_tee_policy(body)

        # Sprint 306a — $CORP capability redemption
        # (Vision §7 Enterprise Confidentiality Mode layer
        # 2). Refuses with 402 Payment Required if the
        # X-CORP-* headers carry an invalid / over-quota /
        # expired / replayed capability. Headers absent =
        # no gating (opt-in for enterprises).
        _enforce_corp_capability(
            x_corp_capability, x_corp_redemption,
        )

        # Validate budget_ftns FIELD type/value upfront (422 for
        # well-formed body that fails semantic validation).
        if "budget_ftns" in body:
            _raw_b = body["budget_ftns"]
            try:
                budget_ftns = float(_raw_b)
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"budget_ftns must be a positive number; "
                        f"got {_raw_b!r}."
                    ),
                )
        else:
            budget_ftns = 1.0
        if budget_ftns <= 0:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"budget_ftns must be > 0; got {budget_ftns}. "
                    f"Use the prsm_quote MCP tool to estimate cost first."
                ),
            )

        # Lazy imports keep the inference module's dependencies (and any
        # future heavy ones like wasmtime/torch) out of the API import graph
        # for nodes that don't run inference.
        import dataclasses
        from decimal import Decimal

        from prsm.compute.inference import (
            ContentTier,
            InferenceRequest,
            sign_receipt,
        )
        from prsm.compute.tee.models import PrivacyLevel

        # Sprint 156 — enum body fields validated upfront, BEFORE
        # the executor 503 check. Pre-fix bad enum values leaked
        # through to a 503 ("Inference executor not initialized").
        _privacy_raw = body.get("privacy_tier", "none")  # sp1234 — honest default (DP is experimental/best-effort)
        try:
            privacy_level = PrivacyLevel(_privacy_raw)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"privacy_tier must be one of "
                    f"{[lvl.value for lvl in PrivacyLevel]}; "
                    f"got {_privacy_raw!r}."
                ),
            )
        _content_raw = body.get("content_tier", "A")
        try:
            content_tier = ContentTier(_content_raw)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"content_tier must be one of "
                    f"{[t.value for t in ContentTier]}; "
                    f"got {_content_raw!r}."
                ),
            )

        if not hasattr(node, 'inference_executor') or node.inference_executor is None:
            # Sprint 1033 — surface the SPECIFIC actionable cause (web3 missing,
            # catalog unset, executor not opted in, ...) instead of a generic
            # "not initialized" that forces operators to grep the daemon log.
            from prsm.node.inference_wiring import (
                diagnose_inference_executor_unavailable,
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "Inference executor not initialized. "
                    + diagnose_inference_executor_unavailable()
                ),
            )

        # Build the request — enums already validated above (sprint 156).
        try:
            request = InferenceRequest(
                prompt=prompt,
                model_id=model_id,
                budget_ftns=Decimal(str(budget_ftns)),
                privacy_tier=privacy_level,
                content_tier=content_tier,
                max_tokens=body.get("max_tokens"),
                temperature=body.get("temperature"),
                requester_node_id=node.identity.node_id if node.identity else None,
            )
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

        # Allocate API-side job_id; this becomes the canonical job_id for
        # both the escrow and the signed receipt.
        job_id = "infer-" + _uuid.uuid4().hex[:12]

        # Sprint 1056 (requester-payment brick 3) — if the body carries a signed
        # PaymentAuthorization, verify it FAIL-CLOSED before doing any work and
        # stash the authenticated requester for the settle path; absent → unchanged
        # self-pay. A rejected auth → 402 (do not serve a paid job we can't honor).
        _paid_requester = await _resolve_paid_requester_or_402(node, body, budget_ftns)
        if _paid_requester is not None:
            _record_paid_requester(node, job_id, _paid_requester)

        # Sprint 251 — record IN_PROGRESS in JobHistoryStore so
        # inference jobs surface in prsm_jobs_list +
        # /compute/status/{job_id} alongside forge jobs. Best-
        # effort: history failures don't block the request.
        import time as _time_for_history
        _job_started_at = _time_for_history.time()
        if (
            hasattr(node, "_job_history")
            and node._job_history is not None
        ):
            try:
                from prsm.node.job_history import (
                    JobHistoryRecord as _JobRec,
                    JobStatus as _JobStat,
                )
                node._job_history.put(_JobRec(
                    job_id=job_id,
                    query=prompt[:256],
                    status=_JobStat.IN_PROGRESS,
                    started_at=_job_started_at,
                    route="inference",
                ))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "JobHistoryStore put(IN_PROGRESS) failed for "
                    "inference job_id=%s: %s", job_id, exc,
                )

        # Lock escrow up-front (pre-pay billing pattern per Phase 3.x.1
        # design plan §6.2).
        escrow_entry = None
        if hasattr(node, '_payment_escrow') and node._payment_escrow:
            try:
                escrow_entry = await node._payment_escrow.create_escrow(
                    job_id=job_id,
                    amount=budget_ftns,
                    requester_id=node.identity.node_id if node.identity else "unknown",
                )
            except Exception as e:
                raise HTTPException(
                    status_code=402,
                    detail=f"Escrow creation failed (insufficient FTNS?): {e}",
                )
            # PaymentEscrow.create_escrow returns None (does NOT raise) when
            # the requester has insufficient FTNS balance. Without this guard
            # the request would proceed unbilled.
            if escrow_entry is None:
                raise HTTPException(
                    status_code=402,
                    detail="Insufficient FTNS balance to lock escrow for inference",
                )

        # Privacy budget pre-flight: reject before executor runs if the
        # cumulative DP epsilon would be exceeded. Post-hoc record_spend
        # below commits the spend; can_spend here gates entry.
        expected_epsilon: Optional[float] = None
        if request.privacy_tier != PrivacyLevel.NONE:
            # Route through PrivacyLevel.config_for_level so any future
            # tier↔ε rebalancing in prsm/compute/tee/models.py stays in
            # one place — duplicating the map here would silently desync
            # the pre-flight gate from the executor's actual ε spend.
            expected_epsilon = PrivacyLevel.config_for_level(
                request.privacy_tier
            ).epsilon
            if (
                hasattr(node, 'privacy_budget')
                and node.privacy_budget
                and not node.privacy_budget.can_spend(expected_epsilon)
            ):
                if escrow_entry and node._payment_escrow:
                    try:
                        await node._payment_escrow.refund_escrow(
                            job_id, "privacy budget exhausted"
                        )
                    except Exception as refund_exc:
                        logger.warning(
                            f"Escrow refund after privacy-budget rejection failed: {refund_exc}"
                        )
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Privacy budget exhausted: ε={expected_epsilon} would "
                        f"exceed remaining budget"
                    ),
                )

        try:
            # Sprint 704 — concurrency gate. When
            # PRSM_INFERENCE_CONCURRENCY_LIMIT is set, serialize
            # inference dispatch to prevent OOM under simultaneous
            # cold-load on tight-RAM operators.
            _sem = _get_inference_semaphore()
            if _sem is not None:
                async with _sem:
                    result = await node.inference_executor.execute(request)
            else:
                result = await node.inference_executor.execute(request)

            # sp1219 — record the served outcome into the node's inference
            # serving counters (observability; fail-soft no-op when off).
            try:
                from prsm.core.monitoring.node_runtime_metrics import (
                    record_inference_result,
                )
                record_inference_result(node, result)
            except Exception:  # noqa: BLE001 — never affect the request path
                pass

            if not result.success:
                # Inference rejected the request (unknown model, budget too
                # low for this executor's pricing, etc.). Refund the escrow.
                if escrow_entry and node._payment_escrow:
                    try:
                        await node._payment_escrow.refund_escrow(
                            job_id, result.error or "inference failed"
                        )
                    except Exception as refund_exc:
                        logger.warning(f"Inference escrow refund failed: {refund_exc}")
                return {
                    "success": False,
                    "error": result.error or "inference failed",
                    "job_id": job_id,
                    "request_id": request.request_id,
                }

            # Replace the executor's internal job_id with the API-side
            # job_id so the receipt's job_id matches the escrow id +
            # response payload, then sign with the node's identity.
            receipt = result.receipt
            if receipt is not None:
                receipt = dataclasses.replace(receipt, job_id=job_id)
                if node.identity:
                    try:
                        receipt = sign_receipt(receipt, node.identity)
                    except Exception as e:
                        logger.warning(f"Receipt signing failed: {e}")

            # Track DP epsilon spend (privacy budget) for non-none privacy
            # tiers — same accounting flow as /compute/forge.
            if (
                hasattr(node, 'privacy_budget')
                and node.privacy_budget
                and request.privacy_tier != PrivacyLevel.NONE
                and receipt is not None
            ):
                try:
                    node.privacy_budget.record_spend(
                        receipt.epsilon_spent,
                        "inference",
                        job_id,
                        model_id=request.model_id,
                    )
                except Exception as e:
                    logger.warning(f"Privacy budget tracking failed: {e}")

            # Sprint 785 — release escrow to the serving node, routing
            # through settle_inference_receipt when we have a receipt
            # so partial_completion (sprint 777) drives proportional
            # credit + refund + slash signal. Mirrors sprint 784's
            # streaming-path integration. Fall back to release_escrow
            # when no receipt (legacy mid-fail path; no regression).
            if escrow_entry and node._payment_escrow:
                operator_id = (
                    node.identity.node_id if node.identity else "self"
                )
                if receipt is not None and hasattr(receipt, "cost_ftns"):
                    try:
                        from decimal import Decimal as _D
                        from prsm.economy.credit_policy import (
                            settle_inference_receipt,
                        )
                        escrow_amount = _D(str(
                            getattr(escrow_entry, "amount", 0),
                        ))
                        decision = await settle_inference_receipt(
                            payment_escrow=node._payment_escrow,
                            receipt=receipt,
                            operator_id=operator_id,
                            job_id=job_id,
                            escrow_amount=escrow_amount,
                        )
                        # Sprint 1037 (brick 1.5) — feed the settled receipt
                        # into the on-chain settlement accumulator (opt-in;
                        # no-op when node._onchain_settlement_client is None).
                        # Fail-open: must never unwind the completed off-chain
                        # settlement above.
                        try:
                            from prsm.settlement.client_wiring import (
                                accumulate_settled_inference_receipt,
                            )
                            _paid = _take_paid_requester(node, job_id)
                            _acc = await accumulate_settled_inference_receipt(
                                client=getattr(
                                    node, "_onchain_settlement_client", None,
                                ),
                                identity=node.identity,
                                provider_address=getattr(
                                    node, "_operator_address", None,
                                ),
                                receipt=receipt,
                                release_ftns=decision.release_to_operator,
                                job_id=job_id,
                                requester_address=(_paid or {}).get("requester"),
                                max_spend_wei=(_paid or {}).get("max_spend_wei"),
                                # sp1144 — opt-in §7 retention (None when audit OFF).
                                inference_receipt_store=getattr(
                                    node,
                                    "_settlement_inference_receipt_store",
                                    None,
                                ),
                            )
                            if _acc != "skipped:no-client":
                                logger.info(
                                    "Sprint 1037 on-chain settlement "
                                    "accumulate (job=%s): %s", job_id, _acc,
                                )
                            # sp1095/1096 — capture the unused remainder (or full
                            # reservation when the on-chain accumulate was skipped).
                            await _capture_relayer_budget(
                                node, _paid, decision.release_to_operator, _acc)
                        except Exception as _acc_exc:  # noqa: BLE001
                            logger.debug(
                                "Sprint 1037 accumulate hook error: %s",
                                _acc_exc,
                            )
                        if decision.should_slash:
                            logger.warning(
                                "Sprint 785 — slash signal for "
                                "job_id=%s: partial_completion."
                                "reason=error (operator-attributable). "
                                "On-chain slash emission deferred to "
                                "a later sprint.",
                                job_id,
                            )
                            # Sprint 798 — persist to operator-visible ring.
                            ring = getattr(
                                node,
                                "_partial_completion_event_log",
                                None,
                            )
                            if ring is not None:
                                try:
                                    info = receipt.partial_completion
                                    ring.append(
                                        job_id=job_id,
                                        operator_node_id=operator_id,
                                        reason=getattr(
                                            info, "reason", "error",
                                        ),
                                        tokens_completed=int(
                                            getattr(
                                                info,
                                                "tokens_completed",
                                                0,
                                            ),
                                        ),
                                        tokens_requested=int(
                                            getattr(
                                                info,
                                                "tokens_requested",
                                                0,
                                            ),
                                        ),
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    logger.debug(
                                        "Sprint 798 — partial_completion "
                                        "ring append failed: %s", exc,
                                    )
                    except Exception as e:
                        logger.warning(
                            f"Inference receipt-aware settle failed "
                            f"for {job_id}: {e}"
                        )
                else:
                    try:
                        await node._payment_escrow.release_escrow(
                            job_id=job_id,
                            provider_id=operator_id,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Inference escrow release failed: {e}"
                        )

            # Sprint 251 — record COMPLETED in JobHistoryStore.
            # Inference jobs now appear in prsm_jobs_list +
            # /compute/status/{job_id} alongside forge jobs.
            # Best-effort: doesn't block response.
            if (
                hasattr(node, "_job_history")
                and node._job_history is not None
            ):
                try:
                    from prsm.node.job_history import (
                        JobHistoryRecord as _JobRec,
                        JobStatus as _JobStat,
                    )
                    node._job_history.put(_JobRec(
                        job_id=job_id,
                        query=prompt[:256],
                        status=_JobStat.COMPLETED,
                        started_at=_job_started_at,
                        completed_at=_time_for_history.time(),
                        route="inference",
                        response=result.output,
                    ))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "JobHistoryStore put(COMPLETED) failed "
                        "for inference job_id=%s: %s", job_id, exc,
                    )

            # Sprint 242 — persist the signed receipt for post-hoc
            # lookup via /compute/receipt/{job_id}. Best-effort.
            if (
                receipt is not None
                and hasattr(node, "_receipt_store")
                and node._receipt_store is not None
            ):
                try:
                    node._receipt_store.put(job_id, receipt.to_dict())
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "ReceiptStore put failed for job_id=%s: %s",
                        job_id[:8], e,
                    )

            return {
                "success": True,
                "job_id": job_id,
                "request_id": request.request_id,
                "output": result.output,
                "receipt": receipt.to_dict() if receipt is not None else None,
            }

        except HTTPException:
            raise
        except Exception as e:
            # Refund on unexpected failure (executor crashed, etc.)
            if escrow_entry and node._payment_escrow:
                try:
                    await node._payment_escrow.refund_escrow(job_id, str(e))
                except Exception:
                    pass
            # Sprint 251 — record FAILED in JobHistoryStore so the
            # operator can see the inference job failed via
            # prsm_jobs_list. Best-effort.
            if (
                hasattr(node, "_job_history")
                and node._job_history is not None
            ):
                try:
                    from prsm.node.job_history import (
                        JobHistoryRecord as _JobRec,
                        JobStatus as _JobStat,
                    )
                    node._job_history.put(_JobRec(
                        job_id=job_id,
                        query=prompt[:256],
                        status=_JobStat.FAILED,
                        started_at=_job_started_at,
                        completed_at=_time_for_history.time(),
                        route="inference",
                        error=str(e),
                    ))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "JobHistoryStore put(FAILED) failed for "
                        "inference job_id=%s: %s", job_id, exc,
                    )
            logger.error(f"Inference pipeline error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Inference pipeline error: {str(e)}",
            )

    @app.post("/compute/inference/stream")
    async def compute_inference_stream(
        request: Request,
        body: Dict[str, Any] = {},
        x_corp_capability: Optional[str] = Header(
            default=None, alias="X-CORP-Capability",
        ),
        x_corp_redemption: Optional[str] = Header(
            default=None, alias="X-CORP-Redemption",
        ),
    ):
        """SSE-streaming sibling of ``/compute/inference``.

        Phase 3.x.8.1 Task 2 — wires
        ``ParallaxScheduledExecutor.execute_streaming`` to a Server-
        Sent-Events HTTP surface so the ``prsm_inference`` MCP tool
        (and any other streaming-capable client) can render token-
        by-token output as the chain produces it.

        Wire format (text/event-stream):

          event: token
          data: {"sequence_index": N, "text_delta": "...",
                 "token_id": null|int, "finish_reason": null|"stop"|...}

          event: result
          data: {"success": true, "request_id": "...", "output": "...",
                 "receipt": {...}, "job_id": "..."}

          event: error
          data: {"error": "...", "code": "...", "job_id": "..."}

        Same input schema as ``/compute/inference``. Same escrow lock
        + refund/settle semantics: pre-execute failure refunds; ANY
        token emitted = settle at full estimate (caller paid for
        chain dispatch — see design plan §3.4 risk register).

        ``event: result`` and ``event: error`` are terminal — server
        closes the connection after them. ``event: token`` events
        are NEVER terminal.
        """
        # Sprint 756 — active-window gate (same as /compute/inference).
        _check_active_window_or_503()
        # Sprint 774 — preemption gate (same as /compute/inference).
        _check_not_preempted_or_503()
        # PRSM_INFERENCE_MAX_RPS_PER_REQUESTER rate limiting (DoS).
        # SHARES the "inference" bucket with /compute/inference so a
        # requester's combined RPS across both endpoints is capped
        # under one operator-tunable knob.
        _rps_raw = os.getenv(
            "PRSM_INFERENCE_MAX_RPS_PER_REQUESTER", "",
        ).strip()
        if _rps_raw:
            try:
                _rps = float(_rps_raw)
                if _rps > 0:
                    from prsm.node.rate_limiter import (
                        get_or_build_bucket,
                    )
                    bucket = get_or_build_bucket(_rps, name="inference")
                    if bucket is not None:
                        # Sprint 741 F69 fix: pre-741, `requester`
                        # was `node.identity.node_id` — the LOCAL
                        # daemon's id, a constant. All callers
                        # shared one bucket → effectively GLOBAL
                        # rate limit. Same bug as /compute/inference.
                        requester = _resolve_requester_key(request)
                        if not bucket.try_consume(requester):
                            retry = bucket.retry_after(requester)
                            raise HTTPException(
                                status_code=429,
                                detail=(
                                    f"Rate limit exceeded for "
                                    f"requester {requester[:24]}... "
                                    f"on /compute/inference/stream "
                                    f"(cap {_rps}/sec, shared with "
                                    f"/compute/inference). Retry "
                                    f"after {retry:.2f}s."
                                ),
                                headers={
                                    "Retry-After": f"{retry:.2f}",
                                },
                            )
            except ValueError:
                logger.warning(
                    "PRSM_INFERENCE_MAX_RPS_PER_REQUESTER=%r not "
                    "numeric; rate limiting disabled", _rps_raw,
                )

        # Sprint 155 — body validation BEFORE executor checks,
        # mirroring sprints 153 + 154 fixes for /compute/forge +
        # /compute/inference. Pre-fix all body errors leaked
        # through to 503 ("Inference executor not initialized").

        prompt = body.get("prompt", "")
        if not prompt:
            # Sprint 536 F66 fix: schema hint
            raise HTTPException(
                status_code=400,
                detail=(
                    "Missing 'prompt' field. Expected body: "
                    "{\"prompt\": \"<text>\", \"model_id\": "
                    "\"mock-llama-3-8b\", \"budget_ftns\": "
                    "<float>, \"max_tokens\": <int>}"
                ),
            )
        # Sprint 198 — cap prompt size; shares
        # PRSM_MAX_INFERENCE_PROMPT_BYTES with the unary sibling.
        _ip_raw = os.environ.get(
            "PRSM_MAX_INFERENCE_PROMPT_BYTES", "",
        ).strip()
        try:
            _ip_cap = int(_ip_raw) if _ip_raw else 100 * 1024
            if _ip_cap <= 0:
                raise ValueError("non-positive")
        except (ValueError, TypeError):
            _ip_cap = 100 * 1024
        _ip_bytes = len(prompt.encode("utf-8"))
        if _ip_bytes > _ip_cap:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"prompt size {_ip_bytes} bytes exceeds "
                    f"PRSM_MAX_INFERENCE_PROMPT_BYTES cap of "
                    f"{_ip_cap}. Trim the prompt or have the "
                    f"operator raise the cap."
                ),
            )
        model_id = body.get("model_id", "")
        if not model_id:
            raise HTTPException(status_code=400, detail="Missing 'model_id' field")

        # Sprint 271 — operator content filter (BEFORE budget
        # validation / escrow / privacy budget / executor stream).
        _enforce_content_filter(prompt=prompt, model_id=model_id)

        # Sprint 305a — TEE policy enforcement (Vision §7
        # Enterprise Confidentiality Mode layer 3). Mirrors
        # the /compute/inference gate so streamed inference
        # honors the same policy.
        _enforce_tee_policy(body)

        # Sprint 306a — $CORP capability redemption.
        # Mirrors the /compute/inference gate so streamed
        # inference honors the same authorization layer.
        _enforce_corp_capability(
            x_corp_capability, x_corp_redemption,
        )

        if "budget_ftns" in body:
            _raw_b = body["budget_ftns"]
            try:
                budget_ftns = float(_raw_b)
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"budget_ftns must be a positive number; "
                        f"got {_raw_b!r}."
                    ),
                )
        else:
            budget_ftns = 1.0
        if budget_ftns <= 0:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"budget_ftns must be > 0; got {budget_ftns}."
                ),
            )

        # Lazy imports — keep the inference module's dependencies out
        # of the API import graph for nodes that don't run inference.
        import dataclasses
        from decimal import Decimal

        from prsm.compute.inference import (
            ContentTier,
            InferenceRequest,
            sign_receipt,
        )
        from prsm.compute.inference.parallax_executor import (
            InferenceTokenEvent,
        )
        from prsm.compute.inference.models import InferenceResult
        from prsm.compute.tee.models import PrivacyLevel

        # Sprint 156 — enum body fields validated upfront, BEFORE
        # the executor 503 check.
        _privacy_raw = body.get("privacy_tier", "none")  # sp1234 — honest default (DP is experimental/best-effort)
        try:
            privacy_level = PrivacyLevel(_privacy_raw)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"privacy_tier must be one of "
                    f"{[lvl.value for lvl in PrivacyLevel]}; "
                    f"got {_privacy_raw!r}."
                ),
            )
        _content_raw = body.get("content_tier", "A")
        try:
            content_tier = ContentTier(_content_raw)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"content_tier must be one of "
                    f"{[t.value for t in ContentTier]}; "
                    f"got {_content_raw!r}."
                ),
            )

        if not hasattr(node, 'inference_executor') or node.inference_executor is None:
            # Sprint 1033 — actionable cause (parity with the unary path).
            from prsm.node.inference_wiring import (
                diagnose_inference_executor_unavailable,
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "Inference executor not initialized. "
                    + diagnose_inference_executor_unavailable()
                ),
            )
        # Streaming requires execute_streaming on the wired executor.
        # ParallaxScheduledExecutor (Phase 3.x.8.1) implements it; the
        # Phase 3.x.1 TensorParallelInferenceExecutor does not. Surface
        # as 503 — operator misconfig.
        if not hasattr(node.inference_executor, "execute_streaming"):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Inference executor does not support streaming. "
                    "Wire a ParallaxScheduledExecutor (Phase 3.x.8.1) to "
                    "enable /compute/inference/stream."
                ),
            )
        try:
            request = InferenceRequest(
                prompt=prompt,
                model_id=model_id,
                budget_ftns=Decimal(str(budget_ftns)),
                privacy_tier=privacy_level,
                content_tier=content_tier,
                max_tokens=body.get("max_tokens"),
                temperature=body.get("temperature"),
                requester_node_id=node.identity.node_id if node.identity else None,
            )
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

        job_id = "infer-stream-" + _uuid.uuid4().hex[:12]

        # Sprint 1056 (requester-payment brick 3) — verify an optional signed
        # PaymentAuthorization FAIL-CLOSED before serving the stream; stash the
        # authenticated requester for the settle path (parity with the unary path).
        _paid_requester = await _resolve_paid_requester_or_402(node, body, budget_ftns)
        if _paid_requester is not None:
            _record_paid_requester(node, job_id, _paid_requester)

        # Sprint 252 — JobHistoryStore wiring for the streaming
        # path. Mirror of sprint 251's /compute/inference write
        # sites: IN_PROGRESS at entry, COMPLETED on terminal
        # result success, FAILED on each failure/exhaustion path.
        # Helper consolidates the write logic so the 5 terminal
        # sites stay in sync.
        import time as _time_for_history
        _job_started_at = _time_for_history.time()

        def _record_history(
            status_val: str,
            *,
            response_text: Optional[str] = None,
            error_text: Optional[str] = None,
        ) -> None:
            if (
                not hasattr(node, "_job_history")
                or node._job_history is None
            ):
                return
            try:
                from prsm.node.job_history import (
                    JobHistoryRecord as _JobRec,
                    JobStatus as _JobStat,
                )
                completed = (
                    _time_for_history.time()
                    if status_val != _JobStat.IN_PROGRESS.value
                    else None
                )
                node._job_history.put(_JobRec(
                    job_id=job_id,
                    query=prompt[:256],
                    status=_JobStat(status_val),
                    started_at=_job_started_at,
                    completed_at=completed,
                    route="inference_stream",
                    response=response_text,
                    error=error_text,
                ))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "JobHistoryStore put(%s) failed for streaming "
                    "inference job_id=%s: %s",
                    status_val, job_id, exc,
                )

        _record_history("in_progress")

        # Sprint 253 — persist the signed receipt to ReceiptStore
        # on streaming success (mirrors sprint 242's unary
        # wiring). The receipt is signed inside _result_to_dict
        # for the wire payload; this helper re-applies the same
        # rebind+sign so we get the canonical signed receipt for
        # the audit store. Best-effort.
        def _record_receipt(item: Any) -> None:
            store = getattr(node, "_receipt_store", None)
            if store is None or item.receipt is None:
                return
            try:
                import dataclasses as _dc
                receipt = _dc.replace(item.receipt, job_id=job_id)
                if node.identity is not None:
                    from prsm.compute.inference import sign_receipt
                    receipt = sign_receipt(receipt, node.identity)
                store.put(job_id, receipt.to_dict())
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ReceiptStore put failed for streaming "
                    "job_id=%s: %s", job_id[:8], exc,
                )

        # Lock escrow — same pre-pay billing pattern as the unary
        # endpoint. Failure paths refund; success path settles after
        # the terminal result event.
        escrow_entry = None
        if hasattr(node, '_payment_escrow') and node._payment_escrow:
            try:
                escrow_entry = await node._payment_escrow.create_escrow(
                    job_id=job_id,
                    amount=budget_ftns,
                    requester_id=node.identity.node_id if node.identity else "unknown",
                )
            except Exception as e:
                raise HTTPException(
                    status_code=402,
                    detail=f"Escrow creation failed (insufficient FTNS?): {e}",
                )
            if escrow_entry is None:
                raise HTTPException(
                    status_code=402,
                    detail="Insufficient FTNS balance to lock escrow for inference",
                )

        # Privacy-budget pre-flight identical to /compute/inference.
        if request.privacy_tier != PrivacyLevel.NONE:
            expected_epsilon = PrivacyLevel.config_for_level(
                request.privacy_tier
            ).epsilon
            if (
                hasattr(node, 'privacy_budget')
                and node.privacy_budget
                and not node.privacy_budget.can_spend(expected_epsilon)
            ):
                if escrow_entry and node._payment_escrow:
                    try:
                        await node._payment_escrow.refund_escrow(
                            job_id, "privacy budget exhausted"
                        )
                    except Exception as refund_exc:
                        logger.warning(
                            f"Escrow refund after privacy-budget rejection failed: "
                            f"{refund_exc}"
                        )
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Privacy budget exhausted: ε={expected_epsilon} would "
                        f"exceed remaining budget"
                    ),
                )

        async def _event_generator():
            """Drives the executor's streaming generator and frames
            each yielded item as an SSE event. The terminal item
            (success or failure) drives escrow settle/refund + the
            terminal ``result`` or ``error`` event.

            Raises NEVER cross the SSE boundary — any unhandled
            exception encodes a final ``error`` event + refunds the
            escrow defensively."""
            tokens_emitted = 0
            try:
                # Sprint 704 — concurrency gate for streaming path.
                # Same semaphore as unary so a streaming inference
                # serializes against unary requests too (the model
                # cache is shared; both paths peak the same memory).
                _sem = _get_inference_semaphore()
                if _sem is not None:
                    await _sem.acquire()
                try:
                    _stream_iter = node.inference_executor.execute_streaming(
                        request,
                    )
                finally:
                    # Note: semaphore released AFTER the iterator is
                    # fully consumed below (in the finally of the outer
                    # try). Acquire-here-release-after-iter ensures
                    # cold-load + token-generation both hold the slot.
                    pass
                async for item in _stream_iter:
                    if isinstance(item, InferenceTokenEvent):
                        tokens_emitted += 1
                        yield _sse_event("token", _token_event_to_dict(item))
                    elif isinstance(item, InferenceResult):
                        # sp1219 — record the served outcome into the node's
                        # inference serving counters (observability; fail-soft).
                        try:
                            from prsm.core.monitoring.node_runtime_metrics import (
                                record_inference_result,
                            )
                            record_inference_result(node, item)
                        except Exception:  # noqa: BLE001 — never affect streaming
                            pass
                        if item.success:
                            await _settle_streaming_escrow(
                                node, job_id, escrow_entry, request,
                                item,
                            )
                            # Sprint 252 — COMPLETED to history.
                            _record_history(
                                "completed",
                                response_text=item.output,
                            )
                            # Sprint 253 — persist signed receipt.
                            _record_receipt(item)
                            yield _sse_event(
                                "result",
                                _result_to_dict(
                                    item,
                                    job_id=job_id,
                                    identity=node.identity,
                                ),
                            )
                        else:
                            # Phase 3.x.8.1 round-1 M1: settle on
                            # tokens emitted; refund on
                            # zero-tokens-and-failure (pre-execute
                            # gate trip).
                            await _resolve_post_token_billing(
                                node, job_id, escrow_entry,
                                tokens_emitted,
                                item.error or "inference failed",
                            )
                            # Sprint 252 — FAILED to history.
                            _record_history(
                                "failed",
                                error_text=(
                                    item.error or "inference failed"
                                ),
                            )
                            yield _sse_event("error", {
                                "error": item.error or "inference failed",
                                "code": "EXECUTION_FAILURE",
                                "job_id": job_id,
                                "request_id": request.request_id,
                            })
                        return
                    else:
                        # Defensive — execute_streaming's contract is
                        # InferenceTokenEvent | InferenceResult.
                        await _resolve_post_token_billing(
                            node, job_id, escrow_entry,
                            tokens_emitted,
                            f"unexpected event type {type(item).__name__}",
                        )
                        _record_history(
                            "failed",
                            error_text=(
                                f"unexpected event type "
                                f"{type(item).__name__}"
                            ),
                        )
                        yield _sse_event("error", {
                            "error": (
                                f"executor yielded unexpected type "
                                f"{type(item).__name__}"
                            ),
                            "code": "INTERNAL_ERROR",
                            "job_id": job_id,
                        })
                        return
                # Generator exhausted without a terminal — protocol
                # violation by the executor.
                await _resolve_post_token_billing(
                    node, job_id, escrow_entry,
                    tokens_emitted,
                    "executor exhausted without terminal result",
                )
                _record_history(
                    "failed",
                    error_text=(
                        "executor exhausted without terminal result"
                    ),
                )
                yield _sse_event("error", {
                    "error": (
                        "executor exhausted without yielding a terminal "
                        "InferenceResult"
                    ),
                    "code": "INTERNAL_ERROR",
                    "job_id": job_id,
                })
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Streaming inference pipeline error for job_id=%r",
                    job_id,
                )
                await _resolve_post_token_billing(
                    node, job_id, escrow_entry,
                    tokens_emitted, str(exc),
                )
                _record_history(
                    "failed",
                    error_text=f"{exc.__class__.__name__}: {exc}",
                )
                yield _sse_event("error", {
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "code": "INTERNAL_ERROR",
                    "job_id": job_id,
                })
            finally:
                # Sprint 704 — release the concurrency-gate semaphore
                # only AFTER the iterator finishes (success OR
                # exception). Acquired above before `_stream_iter`
                # construction; held across cold-load + all token
                # frames so the slot covers peak memory.
                if _sem is not None:
                    try:
                        _sem.release()
                    except (RuntimeError, ValueError):
                        # Defensive — already released by another
                        # path (shouldn't happen but don't crash
                        # the generator).
                        pass

        return StreamingResponse(
            _event_generator(),
            media_type="text/event-stream",
            headers={
                # Disable proxy buffering — critical for real-time
                # token delivery through nginx / cloudflare /
                # traefik / etc. SSE without these headers can be
                # buffered into a single chunk.
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.get("/billing/{job_id}")
    async def get_billing_status(job_id: str) -> Dict[str, Any]:
        """Return escrow + billing state for a given job_id.

        Phase 3.x.1 Task 7 — backs the prsm_billing_status MCP tool.
        Queries PaymentEscrow.get_escrow() for the job and returns a
        structured billing snapshot. Returns 404 if no escrow exists for
        the given job_id.
        """
        if not hasattr(node, '_payment_escrow') or node._payment_escrow is None:
            raise HTTPException(
                status_code=503,
                detail="Payment escrow not initialized on this node.",
            )

        entry = node._payment_escrow.get_escrow(job_id)
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail=f"No escrow found for job_id={job_id}",
            )

        return {
            "job_id": entry.job_id,
            "escrow_id": entry.escrow_id,
            "requester_id": entry.requester_id,
            "amount_ftns": entry.amount,
            "status": entry.status.value,
            "provider_winner": entry.provider_winner,
            "tx_lock": entry.tx_lock,
            "tx_release": entry.tx_release,
            "created_at": entry.created_at,
            "completed_at": entry.completed_at,
            "metadata": entry.metadata,
        }

    @app.post("/content/upload")
    async def upload_content(req: ContentUploadRequest) -> Dict[str, Any]:
        """Upload text content to ContentStore with provenance tracking."""
        if not node.content_uploader:
            raise HTTPException(status_code=503, detail="Content uploader not initialized")

        # Pre-flight check: content_publisher (BitTorrent layer)
        # must be wired or upload will produce a generic 502 from
        # the None-return path. Return a 503 with actionable hint
        # instead. Most common cause: libtorrent not installed.
        if getattr(node.content_uploader, "content_publisher", None) is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "ContentPublisher (BitTorrent layer) not wired — "
                    "uploads cannot proceed. Most common cause: "
                    "libtorrent not installed on this Python. Install "
                    "with `pip install libtorrent>=2.0.9` (macOS may "
                    "need `brew install libtorrent-rasterbar` first)."
                ),
            )

        # Cap parent_cids list length. Each parent gets royalty
        # per access — unbounded list = unbounded loop. Default
        # 100 covers any practical citation chain.
        _parents_cap_raw = os.getenv("PRSM_MAX_PARENT_CIDS", "").strip()
        try:
            _parents_cap = (
                int(_parents_cap_raw) if _parents_cap_raw else 100
            )
            if _parents_cap <= 0:
                raise ValueError("non-positive")
        except (ValueError, TypeError):
            logger.warning(
                "PRSM_MAX_PARENT_CIDS=%r not a positive int; using 100",
                _parents_cap_raw,
            )
            _parents_cap = 100
        if len(req.parent_cids) > _parents_cap:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"parent_cids count {len(req.parent_cids)} "
                    f"exceeds PRSM_MAX_PARENT_CIDS cap of "
                    f"{_parents_cap}. Trim citations or have the "
                    f"operator raise the cap."
                ),
            )

        # Cap upload size to prevent DoS via multi-GB payloads.
        # Default 10MB covers typical research papers; operators
        # tune for larger via PRSM_MAX_UPLOAD_BYTES.
        _size_cap_raw = os.getenv("PRSM_MAX_UPLOAD_BYTES", "").strip()
        try:
            _size_cap = int(_size_cap_raw) if _size_cap_raw else 10 * 1024 * 1024
            if _size_cap <= 0:
                raise ValueError("non-positive")
        except (ValueError, TypeError):
            logger.warning(
                "PRSM_MAX_UPLOAD_BYTES=%r not a positive int; "
                "using 10MB default",
                _size_cap_raw,
            )
            _size_cap = 10 * 1024 * 1024
        text_bytes = len(req.text.encode("utf-8"))
        if text_bytes > _size_cap:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"text size {text_bytes} bytes exceeds "
                    f"PRSM_MAX_UPLOAD_BYTES cap of {_size_cap}. "
                    f"Either upload via /content/upload/shard "
                    f"(supports sharding) or have the operator "
                    f"raise the cap."
                ),
            )

        # Cap replicas to prevent DoS via excessive replication
        # requests. Default 100 covers any practical use; operators
        # tune via PRSM_MAX_REPLICAS env (typo / non-numeric falls
        # back to default with WARNING log).
        _cap_raw = os.getenv("PRSM_MAX_REPLICAS", "").strip()
        try:
            _cap = int(_cap_raw) if _cap_raw else 100
            if _cap <= 0:
                raise ValueError("non-positive")
        except (ValueError, TypeError):
            logger.warning(
                "PRSM_MAX_REPLICAS=%r not a positive int; using 100",
                _cap_raw,
            )
            _cap = 100
        if req.replicas > _cap:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"replicas={req.replicas} exceeds "
                    f"PRSM_MAX_REPLICAS cap of {_cap}. Lower the "
                    f"replicas count or have the operator raise "
                    f"the cap."
                ),
            )

        # Sprint 304 — Enterprise Confidentiality Mode.
        # When recipients are specified, encrypt the text
        # client-style before handing off to the sharding
        # layer. The ciphertext bundle (with manifest) is
        # uploaded as the payload; only the listed recipients
        # can decrypt. FTNS balance is orthogonal — encryption
        # is the security primitive.
        upload_text = req.text
        upload_filename = req.filename
        encrypted = False
        if req.threshold is not None and req.recipients is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "`threshold` requires `recipients` to "
                    "be set; bare threshold on a plaintext "
                    "upload is operator confusion"
                ),
            )
        if req.recipients is not None:
            if not isinstance(req.recipients, list) or len(req.recipients) == 0:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "recipients must be a non-empty list "
                        "when provided (Vision §7 Enterprise "
                        "Confidentiality Mode)"
                    ),
                )
            from prsm.enterprise.recipient_encryption import (
                EnterpriseRecipient,
                encrypt_for_recipients,
                encrypt_for_threshold,
            )
            try:
                recipients = [
                    EnterpriseRecipient(
                        identifier=r.get("identifier", ""),
                        x25519_pubkey_b64=r.get(
                            "x25519_pubkey_b64", "",
                        ),
                    )
                    for r in req.recipients
                ]
                if req.threshold is not None:
                    # Sprint 307 — t-of-n mode
                    payload = encrypt_for_threshold(
                        req.text.encode("utf-8"),
                        recipients,
                        threshold=req.threshold,
                    )
                else:
                    payload = encrypt_for_recipients(
                        req.text.encode("utf-8"),
                        recipients,
                    )
            except ValueError as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"recipient encryption failed: {e}",
                )
            import json as _json
            upload_text = _json.dumps(payload.to_dict())
            # Tag the filename so the manifest endpoint and
            # retrieve consumers can identify encrypted blobs
            # without re-parsing — small ergonomics win.
            if not upload_filename.endswith(".enc.json"):
                upload_filename = (
                    f"{upload_filename}.enc.json"
                )
            encrypted = True

        try:
            result = await node.content_uploader.upload_text(
                text=upload_text,
                filename=upload_filename,
                replicas=req.replicas,
                royalty_rate=req.royalty_rate,
                parent_cids=req.parent_cids if req.parent_cids else None,
                creator_eth_address=req.creator_eth_address,
            )
        except NotImplementedError as exc:
            # Sprint 491 (F29 fix) — `upload_text` raises
            # NotImplementedError when content exceeds the
            # sharding threshold (the internal _upload_with_sharding
            # path references ContentSharder which doesn't exist).
            # Surface as 413 (Payload Too Large — matches the
            # earlier size-cap 413) with the actionable detail
            # already in the NotImplementedError message
            # (directs operators to /content/upload/shard).
            # Pre-fix: this raised a generic 502 with no path
            # forward.
            raise HTTPException(
                status_code=413,
                detail=str(exc),
            )
        except Exception as exc:
            # Sprint 179 — surface the underlying exception in the
            # 502 detail so operators see what really broke
            # (libtorrent API drift, IPv6 binding, disk full, etc.)
            # instead of a generic "content store unavailable?".
            logger.exception("upload_text raised")
            raise HTTPException(
                status_code=502,
                detail=f"Upload failed: {type(exc).__name__}: {exc}",
            )

        if not result:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Upload failed — upload_text returned None. "
                    "Common causes: content_publisher unwired, "
                    "_publish_content raised + was swallowed (check "
                    "logs for 'Content publish failed'), or BitTorrent "
                    "layer crashed mid-upload."
                ),
            )

        # Sprint 291 — fingerprint dedup. Decorate response
        # with canonical_creator + duplicate_of_creator so
        # the caller sees whether they're the original or a
        # duplicate-upload attempt. Bytes still get cached
        # for serving; royalty routing to the canonical
        # creator is enforced at the distribution layer.
        duplicate_of_creator = None
        canonical_creator = None
        _reg = getattr(
            node, "_content_fingerprint_registry", None,
        )
        if (
            _reg is not None
            and result.content_hash
            and req.creator_eth_address
        ):
            try:
                canonical_creator, is_new = _reg.register(
                    content_hash=result.content_hash,
                    creator_eth_address=req.creator_eth_address,
                )
                if (
                    not is_new
                    and canonical_creator
                    != req.creator_eth_address
                ):
                    duplicate_of_creator = canonical_creator
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ContentFingerprintRegistry.register "
                    "failed: %s (hash=%s)",
                    exc,
                    (result.content_hash or "?")[:16],
                )

        return {
            "cid": result.content_id,
            "filename": result.filename,
            "size_bytes": result.size_bytes,
            "content_hash": result.content_hash,
            "creator_id": result.creator_id,
            "royalty_rate": result.royalty_rate,
            "parent_cids": result.parent_cids,
            "duplicate_of_creator": duplicate_of_creator,
            "canonical_creator": canonical_creator,
            "encrypted": encrypted,
            # Sprint 524: surface provenance_tx_hash in the
            # immediate upload response (instead of forcing a
            # /content/mine round-trip). None when no on-chain
            # provenance write happened (no creator_address,
            # provenance_client unwired, etc).
            "provenance_tx_hash": getattr(
                result, "provenance_tx_hash", None,
            ),
        }

    # ── Sprint 304 — recipient-manifest endpoint ─────────
    # Vision §7 Enterprise Confidentiality Mode. Lets a
    # recipient check what sealed-key entries exist for a
    # given encrypted CID before they fetch + decrypt the
    # full ciphertext. Reads the same content blob the
    # /content/retrieve path does — no separate storage.

    @app.get("/content/recipient-manifest/{cid}")
    async def get_recipient_manifest(
        cid: str, timeout: float = 30.0,
    ) -> Dict[str, Any]:
        if not node.content_provider:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Content provider not initialized."
                ),
            )
        try:
            content_bytes = await (
                node.content_provider.request_content(
                    cid=cid, timeout=timeout,
                    verify_hash=True,
                )
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"cid {cid!r} not found",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"retrieve failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
        if content_bytes is None:
            raise HTTPException(
                status_code=404,
                detail=f"cid {cid!r} not found",
            )
        import json as _json
        try:
            parsed = _json.loads(content_bytes)
            if (
                not isinstance(parsed, dict)
                or "manifest" not in parsed
            ):
                raise ValueError("no manifest field")
            manifest = parsed["manifest"]
            if (
                not isinstance(manifest, dict)
                or "version" not in manifest
                or "entries" not in manifest
            ):
                raise ValueError("malformed manifest")
        except (ValueError, _json.JSONDecodeError) as e:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"cid {cid!r} is not an encrypted "
                    f"recipient bundle: {e}"
                ),
            )
        return manifest

    @app.post("/content/upload/shard")
    async def upload_shard_dataset(body: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Upload a dataset with semantic sharding.

        Each shard is published through the proprietary BitTorrent
        layer via ``ContentUploader.upload()`` — the same path the
        regular ``/content/upload`` endpoint uses — so the resulting
        ``cid`` for each shard is a real network-discoverable CID
        (no placeholders).

        POST body: {
            "dataset_id": "nada-nc-2025",
            "title": "NADA NC Vehicle Registrations 2025",
            "content_b64": "<base64-encoded content>",
            "shard_count": 4,
            "royalty_rate": 0.05,
            "base_access_fee": 5.0,
            "per_shard_fee": 0.5,
        }
        """
        import base64
        import re
        from prsm.data.shard_models import SemanticShard, SemanticShardManifest

        dataset_id = body.get("dataset_id", "")
        title = body.get("title", dataset_id)
        content_b64 = body.get("content_b64", "")
        # Sprint 196 — int/float casts uncaught → 500. Validate
        # upfront, bound royalty_rate to documented [0.001, 0.1].
        _raw_sc = body.get("shard_count", 4)
        try:
            shard_count = int(_raw_sc)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"shard_count must be a positive integer; "
                    f"got {_raw_sc!r}."
                ),
            )
        _raw_rr = body.get("royalty_rate", 0.01)
        try:
            royalty_rate = float(_raw_rr)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"royalty_rate must be a number in [0.001, 0.1]; "
                    f"got {_raw_rr!r}."
                ),
            )
        if royalty_rate < 0.001 or royalty_rate > 0.1:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"royalty_rate must be in [0.001, 0.1]; "
                    f"got {royalty_rate}."
                ),
            )
        # sp995 (fix A) — capture the creator's on-chain ETH address per shard,
        # mirroring /content/upload (ContentUploadRequest.creator_eth_address).
        # Without this, every shard fell back to the OPERATOR's wallet (or None),
        # so the §14 stake gate + on-chain royalty routing keyed on the hosting
        # operator, not the creator who earned HIGH. Optional + same 0x-40-hex
        # validation; omitted → unchanged (upload() falls back to the operator
        # address, preserving operator self-upload behavior).
        creator_eth_address = body.get("creator_eth_address") or None
        if creator_eth_address is not None and not re.fullmatch(
            r"0x[0-9a-fA-F]{40}", creator_eth_address,
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    "creator_eth_address must be a 0x-prefixed 40-hex-char "
                    f"address; got {creator_eth_address!r}."
                ),
            )

        if not dataset_id:
            # Sprint 536 F66 fix: schema hint
            raise HTTPException(
                status_code=400,
                detail=(
                    "Missing 'dataset_id'. Expected body: "
                    "{\"dataset_id\": \"<unique-id>\", "
                    "\"content_b64\": \"<base64>\", "
                    "\"title\": \"<optional>\", \"shard_count\": "
                    "<int default 4>, \"royalty_rate\": <0.001-0.1>}"
                ),
            )
        if shard_count < 1:
            raise HTTPException(
                status_code=400, detail="shard_count must be >= 1",
            )
        # Cap shard_count to prevent DoS via fragmentation —
        # millions of tiny shards balloon the manifest +
        # publishing overhead. PRSM_MAX_SHARD_COUNT (default
        # 1000) is plenty for any practical dataset.
        _shard_count_cap_raw = os.getenv(
            "PRSM_MAX_SHARD_COUNT", "",
        ).strip()
        try:
            _shard_count_cap = (
                int(_shard_count_cap_raw)
                if _shard_count_cap_raw else 1000
            )
            if _shard_count_cap <= 0:
                raise ValueError("non-positive")
        except (ValueError, TypeError):
            logger.warning(
                "PRSM_MAX_SHARD_COUNT=%r not a positive int; "
                "using 1000 default", _shard_count_cap_raw,
            )
            _shard_count_cap = 1000
        if shard_count > _shard_count_cap:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"shard_count {shard_count} exceeds "
                    f"PRSM_MAX_SHARD_COUNT cap of "
                    f"{_shard_count_cap}. Lower shard_count or "
                    f"have the operator raise the cap."
                ),
            )

        # Decode content. Empty payload is rejected — the previous
        # placeholder-CID behavior produced non-discoverable manifests
        # that broke `prsm_search_shards` → fetch → royalty-distribution
        # downstream. Per gap-list delta 2026-05-07.
        try:
            content = base64.b64decode(content_b64) if content_b64 else b""
        except Exception:
            raise HTTPException(
                status_code=400, detail="content_b64 is not valid base64",
            )
        if not content:
            raise HTTPException(
                status_code=400,
                detail=(
                    "content_b64 is empty or missing — refusing to "
                    "create a manifest with no payload"
                ),
            )

        # Cap shard payload size. PRSM_MAX_SHARD_UPLOAD_BYTES
        # (default 100MB) caps the decoded content. The shard
        # endpoint allows much higher than the regular /upload
        # endpoint because it natively chunks; but unbounded is
        # still a DoS vector.
        _shard_cap_raw = os.getenv(
            "PRSM_MAX_SHARD_UPLOAD_BYTES", "",
        ).strip()
        try:
            _shard_cap = (
                int(_shard_cap_raw) if _shard_cap_raw
                else 100 * 1024 * 1024
            )
            if _shard_cap <= 0:
                raise ValueError("non-positive")
        except (ValueError, TypeError):
            logger.warning(
                "PRSM_MAX_SHARD_UPLOAD_BYTES=%r not a positive int; "
                "using 100MB default", _shard_cap_raw,
            )
            _shard_cap = 100 * 1024 * 1024
        if len(content) > _shard_cap:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"decoded content size {len(content)} bytes "
                    f"exceeds PRSM_MAX_SHARD_UPLOAD_BYTES cap of "
                    f"{_shard_cap}. Either split the upload or "
                    f"have the operator raise the cap."
                ),
            )

        # ContentPublisher path requires a wired ContentUploader,
        # mirroring the regular /content/upload endpoint above.
        if not node.content_uploader:
            raise HTTPException(
                status_code=503, detail="Content uploader not initialized",
            )

        # Pre-flight check on content_publisher (BitTorrent layer)
        # — same diagnostic as /content/upload (sprint 124). Without
        # this, libtorrent-missing produced a downstream None-return
        # path with no clear remediation hint.
        if getattr(
            node.content_uploader, "content_publisher", None,
        ) is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "ContentPublisher (BitTorrent layer) not wired — "
                    "shard uploads cannot proceed. Most common cause: "
                    "libtorrent not installed on this Python. Install "
                    "with `pip install libtorrent>=2.0.9` (macOS may "
                    "need `brew install libtorrent-rasterbar` first)."
                ),
            )

        # Slice content into shards + publish each through the real
        # BitTorrent layer. Each upload() call returns an UploadedContent
        # with a network-discoverable CID + provenance record.
        chunk_size = max(len(content) // max(shard_count, 1), 1024)
        shards = []
        for i in range(shard_count):
            start = i * chunk_size
            end = min(start + chunk_size, len(content))
            chunk = content[start:end]
            if not chunk:
                # When shard_count > content size / 1024 the trailing
                # shards would be empty. Skip — we can't publish 0
                # bytes through the BT layer. The manifest reports the
                # actual shard count produced.
                continue

            shard_filename = f"{dataset_id}-shard-{i:04d}"
            try:
                uploaded = await node.content_uploader.upload(
                    content=chunk,
                    filename=shard_filename,
                    royalty_rate=royalty_rate,
                    creator_eth_address=creator_eth_address,
                )
            except NotImplementedError as exc:
                # sp921 (content data-plane review) — a single shard chunk that
                # still exceeds the uploader's internal sharding threshold makes
                # upload() raise NotImplementedError (sp491 F29 guard). Surface
                # as 413 with the actionable detail (mirrors /content/upload at
                # api.py:8906) instead of a generic 500. Operators should pass a
                # larger shard_count so each chunk falls under the threshold.
                raise HTTPException(status_code=413, detail=str(exc))
            if uploaded is None:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"Shard {i} upload failed — content publisher "
                        f"unavailable or rejected the payload"
                    ),
                )

            shards.append(SemanticShard(
                shard_id=shard_filename,
                parent_dataset=dataset_id,
                # Sprint 532 F45 fix: UploadedContent has
                # `content_id` (not `cid`) — same field-rename class
                # as F4 (sprint 425). Pre-fix: AttributeError on every
                # shard upload that produced > 1 shard.
                cid=uploaded.content_id,
                centroid=[float(i) / max(shard_count, 1)],
                record_count=len(chunk),
                size_bytes=len(chunk),
                keywords=[title, f"shard-{i}"],
            ))

        manifest = SemanticShardManifest(
            dataset_id=dataset_id,
            total_records=len(content),
            total_size_bytes=len(content),
            shards=shards,
        )

        # Register with data listing manager if available
        if hasattr(node, 'data_listing_manager') and node.data_listing_manager:
            from prsm.economy.pricing.data_listing import DataListing
            from decimal import Decimal
            listing = DataListing(
                dataset_id=dataset_id,
                owner_id=node.identity.node_id,
                title=title,
                shard_count=shard_count,
                total_size_bytes=len(content),
                base_access_fee=Decimal(str(body.get("base_access_fee", 0))),
                per_shard_fee=Decimal(str(body.get("per_shard_fee", 0))),
            )
            node.data_listing_manager.publish(listing)

        return {
            "dataset_id": dataset_id,
            "title": title,
            "shard_count": len(shards),
            "total_size_bytes": len(content),
            "manifest": manifest.to_dict() if hasattr(manifest, 'to_dict') else str(manifest),
            "shards": [s.to_dict() for s in shards],
            "listing_registered": hasattr(node, 'data_listing_manager') and node.data_listing_manager is not None,
        }

    @app.get("/content/search")
    async def search_content(
        q: str = "",
        limit: int = 20,
        min_tier: Optional[str] = None,
        exclude_new: bool = False,
    ) -> Dict[str, Any]:
        """Search the network content index by keyword.

        Sprint 289 added optional tier filtering:
          min_tier=low|medium|high  filter to creators ≥ tier
          exclude_new=true          hide cold-start creators

        Each result row carries `creator_tier` so callers see
        the tier even when not filtering. Default behavior
        (no filter args) returns everything — including
        TIER_NEW — preserving pre-sprint-289 contract.
        """
        # Sprint 194 — same bounds-validation as sprint 193 fixed
        # on the dashboard duplicate. Pre-fix `min(limit, 100)`
        # capped upper but accepted negative — limit=-1 returned
        # the entire content index.
        if limit < 1 or limit > 100:
            raise HTTPException(
                status_code=422,
                detail=f"limit must be in [1, 100], got {limit}",
            )
        if len(q) > 1024:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"q size {len(q)} chars exceeds cap of 1024. "
                    f"Trim the query."
                ),
            )
        # Sprint 289 — validate min_tier
        from prsm.marketplace.creator_reputation import (
            TIER_NEW, TIER_LOW, TIER_MEDIUM, TIER_HIGH,
        )
        _TIER_RANK = {
            TIER_NEW: -1,
            TIER_LOW: 1,
            TIER_MEDIUM: 2,
            TIER_HIGH: 3,
        }
        if min_tier is not None:
            min_tier = min_tier.strip().lower()
            if min_tier not in (TIER_LOW, TIER_MEDIUM, TIER_HIGH):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"min_tier must be one of "
                        f"{[TIER_LOW, TIER_MEDIUM, TIER_HIGH]}, "
                        f"got {min_tier!r}"
                    ),
                )

        if not node.content_index:
            raise HTTPException(status_code=503, detail="Content index not initialized")

        results = node.content_index.search(q, limit=limit)

        # Sprint 289 — decorate with tier + apply filters.
        # Sprint 290 — composed with stake-eligibility gate.
        from prsm.marketplace.creator_stake_client import (
            apply_stake_gate,
        )
        tracker = getattr(
            node, "_creator_reputation_tracker", None,
        )
        stake_client = getattr(
            node, "_creator_stake_client", None,
        )
        min_rank = (
            _TIER_RANK[min_tier] if min_tier else None
        )

        rendered = []
        # sp995 — per-request memo of the gated tier, keyed by the SAME identity
        # reputation is accumulated + gated under (the creator ETH address). A
        # prolific creator's many CIDs in one response share one lookup, so the
        # on-chain stake read fires at most once per distinct creator per request.
        _tier_memo = {}
        for r in results:
            # sp995 (fix C) — reputation is RECORDED under the creator ETH address
            # (auto-record keys on creator_eth_address), and the stake gate keys on
            # it too, so the tier must be READ under the same identity. Reading
            # under r.creator_id (the node_id) returned TIER_NEW for every creator
            # whose reputation was earned via the retrieve flow → legit HIGH
            # creators were wrongly surfaced as `new` and dropped by min_tier/
            # exclude_new. Fall back to creator_id only for legacy/no-eth records.
            tier_key = r.creator_eth_address or r.creator_id
            if tracker is not None and tier_key:
                if tier_key in _tier_memo:
                    tier = _tier_memo[tier_key]
                else:
                    try:
                        raw_tier = tracker.tier_for(tier_key)
                    except Exception:  # noqa: BLE001
                        raw_tier = TIER_NEW
                    # The stake gate keys on the creator ETH address; a record
                    # with no eth address can't have bonded stake → demote
                    # HIGH→MEDIUM (safe default).
                    tier = apply_stake_gate(
                        raw_tier, r.creator_eth_address, stake_client,
                    )
                    _tier_memo[tier_key] = tier
            else:
                tier = TIER_NEW
            if exclude_new and tier == TIER_NEW:
                continue
            if min_rank is not None:
                if _TIER_RANK.get(tier, -1) < min_rank:
                    continue
            rendered.append({
                "cid": r.cid,
                "filename": r.filename,
                "size_bytes": r.size_bytes,
                "content_hash": r.content_hash,
                "creator_id": r.creator_id,
                "creator_tier": tier,
                "providers": list(r.providers),
                "created_at": r.created_at,
                "metadata": r.metadata,
                "royalty_rate": r.royalty_rate,
                "parent_cids": r.parent_cids,
            })

        return {
            "query": q,
            "results": rendered,
            "count": len(rendered),
        }

    @app.get("/content/index/stats")
    async def content_index_stats() -> Dict[str, Any]:
        """Get content index statistics."""
        if not node.content_index:
            raise HTTPException(status_code=503, detail="Content index not initialized")
        return node.content_index.get_stats()

    # Sprint 268 — surface ContentProvider runtime stats:
    # local_content_count, pending_requests, discovery sub-stats,
    # fetch telemetry. Symmetric pair to /content/index/stats but
    # for the provider-side fetch pipeline. Pre-fix only used
    # internally by /content/retrieve.
    @app.get("/content/provider-stats", tags=["content"])
    async def content_provider_stats() -> Dict[str, Any]:
        """ContentProvider runtime statistics: local content
        count, pending request count, discovery sub-stats, +
        cumulative fetch telemetry."""
        cp = getattr(node, "content_provider", None)
        if cp is None:
            raise HTTPException(
                status_code=503,
                detail="Content provider not initialized.",
            )
        try:
            return cp.get_stats()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=f"get_stats() raised: {exc}",
            )

    @app.get("/content/mine")
    async def get_my_content(
        limit: int = 50, offset: int = 0,
    ) -> Dict[str, Any]:
        """Paginated list of content uploaded by this node.

        Surfaces ContentUploader.uploaded_content as a publisher-
        facing dashboard. Each entry: content_id, filename,
        size_bytes, content_hash, creator_id, royalty_rate,
        access_count, total_royalties, provenance_tx_hash,
        created_at.

        Status:
          503 — ContentUploader not initialized
          422 — limit out of [1, 1000] OR offset < 0
          200 — {entries, total, offset, limit}
        """
        if limit <= 0 or limit > 1000:
            raise HTTPException(
                status_code=422,
                detail=f"limit must be in [1, 1000], got {limit}",
            )
        if offset < 0:
            raise HTTPException(
                status_code=422,
                detail=f"offset must be >= 0, got {offset}",
            )
        uploader = getattr(node, "content_uploader", None)
        if uploader is None:
            raise HTTPException(
                status_code=503,
                detail="ContentUploader not initialized.",
            )
        store = getattr(uploader, "uploaded_content", None) or {}
        # Sort most-recent-first by created_at
        sorted_records = sorted(
            store.values(),
            key=lambda r: getattr(r, "created_at", 0),
            reverse=True,
        )
        page = sorted_records[offset:offset + limit]
        entries = []
        for r in page:
            entries.append({
                "content_id": r.content_id,
                "filename": r.filename,
                "size_bytes": r.size_bytes,
                "content_hash": r.content_hash,
                # Sprint 534 F59 fix: surface BOTH the raw
                # sha3_256 content_hash AND the on-chain
                # provenance_hash (keccak256(creator||sha3(bytes)))
                # so operators looking up their registration via
                # `prsm provenance info <hash>` get the right
                # value. Previously they tried content_hash and
                # got "NOT registered" because the registry keys
                # on provenance_hash.
                "provenance_hash": getattr(
                    r, "provenance_hash", None,
                ),
                "creator_id": r.creator_id,
                "creator_eth_address": getattr(
                    r, "creator_eth_address", None,
                ),
                "royalty_rate": r.royalty_rate,
                "access_count": r.access_count,
                "total_royalties": r.total_royalties,
                "provenance_tx_hash": getattr(r, "provenance_tx_hash", None),
                "created_at": r.created_at,
                "is_sharded": getattr(r, "is_sharded", False),
            })
        return {
            "entries": entries,
            "total": len(store),
            "offset": offset,
            "limit": limit,
        }

    @app.get("/content/{cid}")
    async def get_content_record(cid: str) -> Dict[str, Any]:
        """Look up a specific content record by CID."""
        if not node.content_index:
            raise HTTPException(status_code=503, detail="Content index not initialized")

        record = node.content_index.lookup(cid)
        if not record:
            raise HTTPException(status_code=404, detail="Content not found in index")

        return {
            "cid": record.cid,
            "filename": record.filename,
            "size_bytes": record.size_bytes,
            "content_hash": record.content_hash,
            "creator_id": record.creator_id,
            "providers": list(record.providers),
            "created_at": record.created_at,
            "metadata": record.metadata,
            "royalty_rate": record.royalty_rate,
            "parent_cids": record.parent_cids,
            # Sprint 244 — surface on-chain creator address (None
            # when not set at upload time).
            "creator_eth_address": getattr(
                record, "creator_eth_address", None,
            ),
        }

    class ContentRetrieveResponse(BaseModel):
        """Response model for content retrieval."""
        cid: str
        status: str = Field(description="Retrieval status: 'success', 'not_found', or 'error'")
        data: Optional[str] = Field(
            default=None,
            description="Base64-encoded content data (only if status is 'success')"
        )
        size_bytes: Optional[int] = Field(default=None, description="Size of retrieved content")
        content_hash: Optional[str] = Field(default=None, description="SHA-256 hash of content")
        filename: Optional[str] = Field(default=None, description="Original filename if available")
        providers_tried: int = Field(default=0, description="Number of providers attempted")
        error: Optional[str] = Field(default=None, description="Error message if status is 'error'")

    @app.get("/content/retrieve/{cid}", response_model=ContentRetrieveResponse)
    async def retrieve_content(cid: str, timeout: float = 30.0, verify_hash: bool = True) -> ContentRetrieveResponse:
        """
        Retrieve content from the network by CID.
        
        This endpoint retrieves content from the P2P network using the ContentProvider.
        It will:
        1. Check if content is available locally
        2. Query the content index for providers
        3. Request content from available providers
        4. Verify content hash if verification is enabled
        
        Args:
            cid: PRSM content identifier
            timeout: Seconds to wait for response (default: 30.0)
            verify_hash: Whether to verify SHA-256 hash (default: True)
        
        Returns:
            ContentRetrieveResponse with status and data (base64-encoded) or error
        """
        import base64

        # Sprint 203 — bound timeout. Pre-fix Infinity / NaN /
        # negative / 999999 all passed straight through to
        # request_content, tying up a worker indefinitely
        # (slow-loris DoS). Body-guard middleware doesn't catch
        # query params. Cap [0.1, PRSM_MAX_RETRIEVE_TIMEOUT_SEC]
        # (default 300s = 5min).
        import math as _math
        _to_cap_raw = os.environ.get(
            "PRSM_MAX_RETRIEVE_TIMEOUT_SEC", "",
        ).strip()
        try:
            _to_cap = float(_to_cap_raw) if _to_cap_raw else 300.0
            if _to_cap <= 0:
                raise ValueError("non-positive")
        except (ValueError, TypeError):
            _to_cap = 300.0
        if not _math.isfinite(timeout):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"timeout must be a finite positive number; "
                    f"got {timeout!r}."
                ),
            )
        if timeout < 0.1 or timeout > _to_cap:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"timeout must be in [0.1, {_to_cap}] seconds; "
                    f"got {timeout}."
                ),
            )

        if not node.content_provider:
            raise HTTPException(status_code=503, detail="Content provider not initialized")

        # Sprint 269 — operator content filter enforcement.
        # Refuses BEFORE any compute cost / network fetch, per
        # ContentSelfFilter design (R9-SCOPING-1 §7). Local-only
        # decision; doesn't affect what other operators serve.
        _filter_store = getattr(
            node, "_content_filter_store", None,
        )
        if (
            _filter_store is not None
            and _filter_store.is_cid_blocked(cid)
        ):
            logger.info(
                "content-filter: refused cid=%s (operator blocklist)",
                cid[:14],
            )
            raise HTTPException(
                status_code=451,  # Unavailable For Legal Reasons
                detail=(
                    f"content cid={cid!r} is blocked by this "
                    f"operator's content filter"
                ),
            )

        # Get provider stats before retrieval to determine providers tried
        stats_before = node.content_provider.get_stats()
        
        try:
            content_bytes = await node.content_provider.request_content(
                cid=cid,
                timeout=timeout,
                verify_hash=verify_hash,
            )
            
            if content_bytes is None:
                # Content not found or retrieval failed
                return ContentRetrieveResponse(
                    cid=cid,
                    status="not_found",
                    error="Content not found on any available provider",
                )
            
            # Get content metadata if available
            content_hash = None
            filename = None
            creator_eth_address = None
            provenance_hash = None
            if node.content_index:
                record = node.content_index.lookup(cid)
                if record:
                    content_hash = record.content_hash
                    filename = record.filename
                    creator_eth_address = getattr(
                        record, "creator_eth_address", None,
                    )
                    provenance_hash = getattr(
                        record, "provenance_hash", None,
                    )

            # Sprint 494 (F35 fix) — content_index is populated
            # via GOSSIP_CONTENT_ADVERTISE which is fanned out
            # to peers, with local-subscriber delivery only when
            # `sent == 0`. Even with bootstrap-only connectivity
            # the index may be empty for THIS node's own uploads
            # because the gossip wiring is multi-node-shaped.
            # For local content, fall back to the uploader's
            # `uploaded_content` dict which definitely has the
            # creator_eth_address from the upload-time payload.
            # Without this, the §14 creator-reputation
            # auto-record chain skipped every single-node retrieve
            # silently (sprint 494 chain test surfaced this).
            if creator_eth_address is None:
                _uploader = getattr(node, "content_uploader", None)
                if _uploader is not None:
                    _uploaded = getattr(
                        _uploader, "uploaded_content", {},
                    )
                    _local = _uploaded.get(cid)
                    if _local is not None:
                        if not content_hash:
                            content_hash = getattr(
                                _local, "content_hash", None,
                            )
                        if not filename:
                            filename = getattr(
                                _local, "filename", None,
                            )
                        creator_eth_address = getattr(
                            _local, "creator_eth_address", None,
                        )
                        if provenance_hash is None:
                            provenance_hash = getattr(
                                _local, "provenance_hash", None,
                            )

            # sp1078 (content-data-plane Gap B, creator leg) — the index/advertise-lane
            # creator_eth_address is forgeable (a malicious advertiser can set it). For
            # content with a REGISTERED provenance_hash, override it with the on-chain-
            # authoritative creator (the ProvenanceRegistry's `creator`) so the §14
            # reputation credit below can't be poisoned for a creator the advertiser
            # doesn't control. Unregistered / no economy → the existing value stands
            # (authentic for this node's own uploads; best-effort for advertise-lane).
            if provenance_hash:
                _economy = getattr(node, "content_economy", None)
                if _economy is not None and hasattr(_economy, "_authenticated_creator"):
                    try:
                        _auth_creator = await _economy._authenticated_creator(
                            {"provenance_hash": provenance_hash})
                        if _auth_creator:
                            if creator_eth_address and (
                                _auth_creator.lower() != str(creator_eth_address).lower()):
                                logger.info(
                                    "content %s: using on-chain-registered creator %s "
                                    "for reputation (advertise lane claimed %s)",
                                    cid[:12], _auth_creator[:14],
                                    str(creator_eth_address)[:14])
                            creator_eth_address = _auth_creator
                    except Exception as _ace:  # noqa: BLE001 - never block retrieve
                        logger.debug("authenticated-creator lookup failed for %s: %s",
                                     cid[:12], _ace)

            # Sprint 288 — auto-record creator access against
            # the marketplace reputation tracker. Best-effort;
            # tracker exceptions caught + logged so telemetry
            # failures never deny content retrieval. Skips
            # silently when:
            #   - tracker not wired
            #   - creator_eth_address absent from content record
            #     (content predates sprint-243 creator threading)
            #   - no operator address (ftns_ledger unwired) —
            #     better to skip than record an empty purchaser
            _creator_tracker = getattr(
                node, "_creator_reputation_tracker", None,
            )
            if _creator_tracker is not None and creator_eth_address:
                _op_addr = None
                _ledger = getattr(node, "ftns_ledger", None)
                if _ledger is not None:
                    _op_addr = getattr(
                        _ledger, "_connected_address", None,
                    )
                if _op_addr:
                    try:
                        _creator_tracker.record_access(
                            creator_id=creator_eth_address,
                            purchaser_id=_op_addr,
                            content_id=cid,
                        )
                    except Exception as _exc:  # noqa: BLE001
                        logger.warning(
                            "CreatorReputationTracker."
                            "record_access failed: %s "
                            "(creator=%s, cid=%s)",
                            _exc,
                            (creator_eth_address or "?")[:14],
                            cid[:14],
                        )

            # Encode content as base64 for JSON response
            data_b64 = base64.b64encode(content_bytes).decode('utf-8')
            
            return ContentRetrieveResponse(
                cid=cid,
                status="success",
                data=data_b64,
                size_bytes=len(content_bytes),
                content_hash=content_hash,
                filename=filename,
            )
            
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Content retrieval timed out after {timeout} seconds"
            )
        except Exception as e:
            logger.error(f"Error retrieving content {cid}: {e}")
            return ContentRetrieveResponse(
                cid=cid,
                status="error",
                error=str(e),
            )

    @app.get("/transactions")
    async def get_transactions(limit: int = 50) -> Dict[str, Any]:
        """Get transaction history."""
        # Sprint 172 — validate `limit` bounds. Pre-fix the handler
        # passed `min(limit, 200)` through, so a negative value
        # (e.g. limit=-1) became -1 and downstream interpreted it
        # as "unlimited", returning every transaction in history.
        # Real DoS vector for operators with deep transaction logs.
        if limit < 1 or limit > 200:
            raise HTTPException(
                status_code=422,
                detail=f"limit must be in [1, 200], got {limit}",
            )
        if not node.ledger or not node.identity:
            raise HTTPException(status_code=503, detail="Node not initialized")

        history = await node.ledger.get_transaction_history(
            node.identity.node_id, limit=limit,
        )
        return {
            "transactions": [
                {
                    "tx_id": tx.tx_id,
                    "type": tx.tx_type.value,
                    "from": tx.from_wallet,
                    "to": tx.to_wallet,
                    "amount": tx.amount,
                    "description": tx.description,
                    "timestamp": tx.timestamp,
                }
                for tx in history
            ],
            "count": len(history),
        }

    # ── Agent endpoints ─────────────────────────────────────────

    @app.get("/agents")
    async def list_agents(local_only: bool = False) -> Dict[str, Any]:
        """List known agents (local and/or remote)."""
        if not node.agent_registry:
            raise HTTPException(status_code=503, detail="Agent registry not initialized")

        if local_only:
            agents = node.agent_registry.get_local_agents()
        else:
            agents = node.agent_registry.get_all_agents()

        return {
            "agents": [a.to_dict() for a in agents],
            "count": len(agents),
        }

    @app.get("/agents/search")
    async def search_agents(capability: str, limit: int = 20) -> Dict[str, Any]:
        """Search agents by capability."""
        # Sprint 194 — bounds validation. Pre-fix limit=-1 passed
        # through to agent_registry.search returning all agents.
        if limit < 1 or limit > 100:
            raise HTTPException(
                status_code=422,
                detail=f"limit must be in [1, 100], got {limit}",
            )
        if len(capability) > 256:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"capability size {len(capability)} chars exceeds "
                    f"cap of 256. Trim the capability string."
                ),
            )
        if not node.agent_registry:
            raise HTTPException(status_code=503, detail="Agent registry not initialized")

        results = node.agent_registry.search(capability, limit=limit)
        return {
            "capability": capability,
            "agents": [a.to_dict() for a in results],
            "count": len(results),
        }

    @app.get("/agents/spending")
    async def agent_spending() -> Dict[str, Any]:
        """Aggregate spending dashboard across all local agents."""
        if not node.agent_registry or not node.ledger:
            raise HTTPException(status_code=503, detail="Not initialized")

        agents = node.agent_registry.get_local_agents()
        spending = []
        for agent in agents:
            allowance = await node.ledger.get_agent_allowance(agent.agent_id)
            spending.append({
                "agent_id": agent.agent_id,
                "agent_name": agent.agent_name,
                "allowance": allowance,
            })
        return {"agents": spending, "count": len(spending)}

    @app.get("/agents/{agent_id}")
    async def get_agent(agent_id: str) -> Dict[str, Any]:
        """Get agent details, spending, and status."""
        if not node.agent_registry:
            raise HTTPException(status_code=503, detail="Agent registry not initialized")

        record = node.agent_registry.lookup(agent_id)
        if not record:
            raise HTTPException(status_code=404, detail="Agent not found")

        result = record.to_dict()
        if node.ledger:
            result["allowance"] = await node.ledger.get_agent_allowance(agent_id)
        return result

    @app.get("/agents/{agent_id}/conversations")
    async def get_agent_conversations(agent_id: str, limit: int = 10) -> Dict[str, Any]:
        """Get recent conversation threads involving an agent."""
        if not node.agent_registry:
            raise HTTPException(status_code=503, detail="Agent registry not initialized")

        conv_ids = node.agent_registry.get_agent_conversations(agent_id, limit=limit)
        conversations = []
        for conv_id in conv_ids:
            messages = node.agent_registry.get_conversation(conv_id)
            conversations.append({
                "conversation_id": conv_id,
                "message_count": len(messages),
                "messages": messages[-5:],  # Last 5 messages per conversation
            })
        return {"conversations": conversations, "count": len(conversations)}

    @app.post("/agents/{agent_id}/allowance")
    async def set_agent_allowance(agent_id: str, amount: float, epoch_hours: float = 24.0) -> Dict[str, Any]:
        """Set or update an agent's spending allowance."""
        if not node.ledger or not node.identity:
            raise HTTPException(status_code=503, detail="Not initialized")

        # Sprint 200 — NaN/Infinity slip through `<= 0` (both
        # comparisons are False); guard upfront. Same goes for
        # epoch_hours since it controls a window calculation.
        import math
        if not math.isfinite(amount):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"amount must be a finite positive number; "
                    f"got {amount!r}."
                ),
            )
        if not math.isfinite(epoch_hours):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"epoch_hours must be a finite positive number; "
                    f"got {epoch_hours!r}."
                ),
            )
        if amount <= 0:
            raise HTTPException(status_code=400, detail="Amount must be positive")

        await node.ledger.grant_agent_allowance(
            principal_id=node.identity.node_id,
            agent_id=agent_id,
            amount=amount,
            epoch_hours=epoch_hours,
        )
        return await node.ledger.get_agent_allowance(agent_id)

    @app.delete("/agents/{agent_id}/allowance")
    async def revoke_agent_allowance(agent_id: str) -> Dict[str, Any]:
        """Revoke an agent's spending authority."""
        if not node.ledger or not node.identity:
            raise HTTPException(status_code=503, detail="Not initialized")

        # Sprint 182 — revoke_agent_allowance() may raise on any
        # malformed agent_id (the downstream DB rejects non-UUID
        # input with various DBAPI exception classes that we can't
        # all enumerate). Catch broadly and map to 404.
        try:
            revoked = await node.ledger.revoke_agent_allowance(
                principal_id=node.identity.node_id,
                agent_id=agent_id,
            )
        except Exception:  # noqa: BLE001
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Agent allowance not found (or agent_id "
                    f"malformed): {agent_id!r}"
                ),
            )
        if not revoked:
            raise HTTPException(status_code=404, detail="Agent allowance not found")
        return {"agent_id": agent_id, "revoked": True}

    @app.post("/agents/{agent_id}/pause")
    async def pause_agent(agent_id: str) -> Dict[str, Any]:
        """Temporarily suspend an agent."""
        if not node.agent_registry:
            raise HTTPException(status_code=503, detail="Agent registry not initialized")

        record = node.agent_registry.lookup(agent_id)
        if not record:
            raise HTTPException(status_code=404, detail="Agent not found")

        node.agent_registry.set_agent_status(agent_id, "paused")
        return {"agent_id": agent_id, "status": "paused"}

    @app.post("/agents/{agent_id}/resume")
    async def resume_agent(agent_id: str) -> Dict[str, Any]:
        """Resume a paused agent."""
        if not node.agent_registry:
            raise HTTPException(status_code=503, detail="Agent registry not initialized")

        record = node.agent_registry.lookup(agent_id)
        if not record:
            raise HTTPException(status_code=404, detail="Agent not found")

        node.agent_registry.set_agent_status(agent_id, "online")
        return {"agent_id": agent_id, "status": "online"}

    # ── Ledger endpoints ─────────────────────────────────────────

    @app.get("/ledger/sync/stats")
    async def ledger_sync_stats() -> Dict[str, Any]:
        """Get ledger sync statistics."""
        if not node.ledger_sync:
            raise HTTPException(status_code=503, detail="Ledger sync not initialized")
        return node.ledger_sync.get_stats()

    @app.post("/ledger/transfer")
    async def transfer_ftns(to_wallet: str, amount: float) -> Dict[str, Any]:
        """Transfer FTNS to another node (signed, gossip-broadcast)."""
        if not node.ledger_sync:
            raise HTTPException(status_code=503, detail="Ledger sync not initialized")

        # Sprint 199 — reject NaN/inf BEFORE the `<= 0` check.
        # `nan <= 0` is False and `inf <= 0` is False, so both
        # would silently pass through to signed_transfer.
        import math
        if not math.isfinite(amount):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"amount must be a finite positive number; "
                    f"got {amount!r} (NaN/Infinity rejected)."
                ),
            )
        if amount <= 0:
            raise HTTPException(status_code=400, detail="Amount must be positive")

        tx = await node.ledger_sync.signed_transfer(
            to_wallet=to_wallet,
            amount=amount,
            description=f"API transfer to {to_wallet[:12]}...",
        )
        if not tx:
            raise HTTPException(status_code=400, detail="Insufficient balance")

        return {
            "tx_id": tx.tx_id,
            "from": tx.from_wallet,
            "to": tx.to_wallet,
            "amount": tx.amount,
            "timestamp": tx.timestamp,
        }

    @app.get("/audit/summary")
    async def get_audit_summary(top_paths: int = 10) -> Dict[str, Any]:
        """Aggregated buckets over the audit ring buffer for ops
        dashboards. Bucketed by status range (2xx/3xx/4xx/5xx),
        by method, and the top-N most-frequent paths.

        Status:
          503 — _audit_log not wired
          422 — top_paths out of [1, 100]
          200 — {total, status_buckets, method_buckets, top_paths}
        """
        if top_paths <= 0 or top_paths > 100:
            raise HTTPException(
                status_code=422,
                detail=f"top_paths must be in [1, 100], got {top_paths}",
            )
        ring = getattr(node, "_audit_log", None)
        if ring is None:
            raise HTTPException(
                status_code=503,
                detail="Audit log not initialized on this node.",
            )
        sweep_limit = min(1000, ring.max_entries())
        all_entries = ring.recent(limit=sweep_limit, offset=0)

        status_buckets: Dict[str, int] = {}
        method_buckets: Dict[str, int] = {}
        path_counts: Dict[str, int] = {}
        for e in all_entries:
            code = e.status_code
            if 200 <= code < 300:
                bucket = "2xx"
            elif 300 <= code < 400:
                bucket = "3xx"
            elif 400 <= code < 500:
                bucket = "4xx"
            elif 500 <= code < 600:
                bucket = "5xx"
            else:
                bucket = "other"
            status_buckets[bucket] = status_buckets.get(bucket, 0) + 1
            method_buckets[e.method] = method_buckets.get(e.method, 0) + 1
            path_counts[e.path] = path_counts.get(e.path, 0) + 1

        # Sort paths by count descending; cap at top_paths.
        top = sorted(
            path_counts.items(), key=lambda kv: kv[1], reverse=True,
        )[:top_paths]
        return {
            "total": ring.count(),
            "status_buckets": status_buckets,
            "method_buckets": method_buckets,
            "top_paths": [
                {"path": p, "count": c} for p, c in top
            ],
        }

    @app.get("/audit/recent")
    async def get_audit_recent(
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        requester: Optional[str] = None,
        path_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Recent state-changing API requests for operator review.

        Returns most-recent-first; bounded ring buffer (default
        1024 entries). State-changing means non-GET; GET requests
        are not recorded to keep the buffer focused on writes.

        Optional ?status=N filter for exact status code, or
        ?status=4xx / ?status=5xx for HTTP range shortcuts.

        Status:
          503 — _audit_log not wired on this node
          422 — limit out of [1, 1000] OR offset < 0 OR invalid
                status filter
          200 — {entries, total, offset, limit, total_matched?}
        """
        if limit <= 0 or limit > 1000:
            raise HTTPException(
                status_code=422,
                detail=f"limit must be in [1, 1000], got {limit}",
            )
        if offset < 0:
            raise HTTPException(
                status_code=422,
                detail=f"offset must be >= 0, got {offset}",
            )

        # Parse optional status filter.
        status_predicate = None
        if status is not None:
            s = status.strip().lower()
            if s == "4xx":
                status_predicate = lambda c: 400 <= c < 500
            elif s == "5xx":
                status_predicate = lambda c: 500 <= c < 600
            elif s == "2xx":
                status_predicate = lambda c: 200 <= c < 300
            elif s == "3xx":
                status_predicate = lambda c: 300 <= c < 400
            else:
                try:
                    target = int(s)
                    status_predicate = lambda c, t=target: c == t
                except ValueError:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"invalid status filter {status!r}; "
                            f"expected exact code (e.g. '404') or "
                            f"range '2xx'/'3xx'/'4xx'/'5xx'"
                        ),
                    )

        ring = getattr(node, "_audit_log", None)
        if ring is None:
            raise HTTPException(
                status_code=503,
                detail="Audit log not initialized on this node.",
            )

        # Compose filters: status_predicate AND requester_match
        # AND path_prefix all optional. When all absent, no
        # filter — paginated ring directly.
        any_filter = (
            status_predicate is not None
            or requester is not None
            or path_prefix is not None
        )
        if not any_filter:
            entries = ring.recent(limit=limit, offset=offset)
            return {
                "entries": [e.to_dict() for e in entries],
                "total": ring.count(),
                "offset": offset,
                "limit": limit,
            }

        # With filter: pull a generous window then filter,
        # paginate the result. Cap sweep at 1000 (recent()'s
        # validation ceiling); good enough for practical filter
        # use cases on a 1024-entry ring.
        sweep_limit = min(1000, ring.max_entries())
        all_entries = ring.recent(limit=sweep_limit, offset=0)

        def _matches(e):
            if status_predicate is not None and not status_predicate(
                e.status_code
            ):
                return False
            if requester is not None and e.requester != requester:
                return False
            if path_prefix is not None and not e.path.startswith(
                path_prefix,
            ):
                return False
            return True

        matched = [e for e in all_entries if _matches(e)]
        page = matched[offset:offset + limit]
        result = {
            "entries": [e.to_dict() for e in page],
            "total": ring.count(),
            "total_matched": len(matched),
            "offset": offset,
            "limit": limit,
        }
        if status is not None:
            result["status_filter"] = status
        if requester is not None:
            result["requester_filter"] = requester
        if path_prefix is not None:
            result["path_prefix_filter"] = path_prefix
        return result

    @app.get("/admin/webhook-history")
    async def get_webhook_history(
        limit: int = 50, offset: int = 0,
    ) -> Dict[str, Any]:
        """Recent webhook dispatch attempts (success or failure).
        Each entry: timestamp, event, url, success, attempts,
        status_code, error.

        Status:
          503 — webhook log not wired
          422 — limit out of [1, 1000] OR offset < 0
          200 — {entries, total, offset, limit}
        """
        if limit <= 0 or limit > 1000:
            raise HTTPException(
                status_code=422,
                detail=f"limit must be in [1, 1000], got {limit}",
            )
        if offset < 0:
            raise HTTPException(
                status_code=422,
                detail=f"offset must be >= 0, got {offset}",
            )
        ring = getattr(node, "_webhook_log", None)
        if ring is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Webhook log not initialized "
                    "(set PRSM_WEBHOOK_URL to enable)."
                ),
            )
        entries = ring.recent(limit=limit, offset=offset)
        return {
            "entries": [e.to_dict() for e in entries],
            "total": ring.count(),
            "offset": offset,
            "limit": limit,
        }

    @app.post("/admin/distribution/trigger")
    async def admin_distribution_trigger() -> Dict[str, Any]:
        """Manually trigger pull_and_distribute on-chain.

        Operator action endpoint symmetric to heartbeat/trigger
        (sprint 81). Use when the PullAndDistributeScheduler has
        crashed / paused, or to force an emission round before
        the next scheduled tick.

        Permissionless on contract side; caller pays gas.

        Status:
          503 — CompensationDistributorClient not wired
          502 — chain call raised
          200 — {tx_hash, status}
        """
        client = getattr(node, "_compensation_distributor_client", None)
        if client is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "CompensationDistributorClient not wired. "
                    "FTNS_WALLET_PRIVATE_KEY is required for the write "
                    "client; address resolves from "
                    "PRSM_COMPENSATION_DISTRIBUTOR_ADDRESS or, if "
                    "PRSM_NETWORK is set, the canonical-fallback "
                    "address from networks.py (sprint 144)."
                ),
            )
        from prsm.economy.web3.provenance_registry import OnChainPendingError
        from fastapi.responses import JSONResponse
        try:
            tx_hash, status = client.pull_and_distribute()
        except OnChainPendingError as exc:
            # sp915 — broadcast SUCCEEDED but the receipt is unconfirmed; the
            # distribution tx is in the mempool. Do NOT flatten to a 502 that
            # invites a re-trigger (the re-broadcast races/reverts + burns
            # gas). Surface a distinct 202 carrying tx_hash for reconciliation.
            return JSONResponse(
                status_code=202,
                content={
                    "status": "PENDING",
                    "tx_hash": getattr(exc, "tx_hash", None),
                    "detail": (
                        "pull_and_distribute broadcast OK but UNCONFIRMED; "
                        "do NOT re-trigger — reconcile via tx_hash."
                    ),
                },
            )
        except Exception as exc:  # noqa: BLE001
            # BroadcastFailedError / OnChainRevertedError land here too — both
            # are safe to retry. Sprint 536 F65 fix: detect "insufficient
            # funds for gas"
            # specifically + return 402 (Payment Required) with
            # actionable top-up guidance. Other exceptions stay 502
            # but with cleaner message — strip Web3 internal dict
            # serialization that leaks RPC error codes to operators.
            exc_str = str(exc)
            if (
                "insufficient funds" in exc_str.lower()
                or "-32003" in exc_str
            ):
                raise HTTPException(
                    status_code=402,
                    detail=(
                        "CompensationDistributor TX would fail — "
                        "operator wallet has insufficient ETH for "
                        "gas. Top up the wallet (see `prsm wallet "
                        "gas-status` for current balance + "
                        "threshold). Underlying RPC error: "
                        f"{exc_str[:200]}"
                    ),
                )
            raise HTTPException(
                status_code=502,
                detail=(
                    f"pull_and_distribute raised "
                    f"{type(exc).__name__}: {exc_str[:300]}"
                ),
            )
        return {
            "tx_hash": tx_hash,
            "status": status.name if hasattr(status, "name") else str(status),
        }

    @app.post("/admin/heartbeat/trigger")
    async def admin_heartbeat_trigger() -> Dict[str, Any]:
        """Manually record a heartbeat on-chain.

        Operator action endpoint. Use when the
        HeartbeatScheduler has crashed / paused / been
        disabled and the operator wants to record a manual
        heartbeat to avoid the slashing window opening. The
        contract has no per-caller access control; the call
        succeeds even from non-providers (no-op on chain).

        Status:
          503 — StorageSlashingClient not wired
          502 — chain call raised (RPC unreachable / revert)
          200 — {tx_hash, status: TransferStatus name}
        """
        client = getattr(node, "_storage_slashing_client", None)
        if client is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "StorageSlashingClient not wired. "
                    "FTNS_WALLET_PRIVATE_KEY is required for the write "
                    "client; address resolves from "
                    "PRSM_STORAGE_SLASHING_ADDRESS or, if "
                    "PRSM_NETWORK is set, the canonical-fallback "
                    "address from networks.py (sprint 144)."
                ),
            )
        from prsm.economy.web3.provenance_registry import OnChainPendingError
        from fastapi.responses import JSONResponse
        try:
            tx_hash, status = client.record_heartbeat()
        except OnChainPendingError as exc:
            # sp915 — broadcast SUCCEEDED but the receipt is unconfirmed
            # (wait_for_transaction_receipt timed out); the tx is in the
            # mempool and will likely confirm. Do NOT flatten to a 502 that
            # invites a re-trigger — the re-broadcast races/reverts and burns
            # gas. Surface a distinct 202 carrying tx_hash so the operator
            # reconciles via the receipt instead of re-sending.
            return JSONResponse(
                status_code=202,
                content={
                    "status": "PENDING",
                    "tx_hash": getattr(exc, "tx_hash", None),
                    "detail": (
                        "Heartbeat broadcast OK but UNCONFIRMED; do NOT "
                        "re-trigger — reconcile via tx_hash."
                    ),
                },
            )
        except Exception as exc:  # noqa: BLE001
            # BroadcastFailedError / OnChainRevertedError land here too — both
            # are safe to retry (chain saw nothing / rolled back atomically).
            raise HTTPException(
                status_code=502,
                detail=f"record_heartbeat raised: {exc}",
            )
        return {
            "tx_hash": tx_hash,
            "status": status.name if hasattr(status, "name") else str(status),
        }

    @app.get("/admin/distribution-history")
    async def get_distribution_history(
        limit: int = 50, offset: int = 0,
    ) -> Dict[str, Any]:
        """Recent on-chain Distributed events observed by the
        CompensationDistributorWatcher. Each entry: timestamp,
        to_creator, to_operator, to_grant, total_distributed.

        Status:
          503 — distribution log not wired
          422 — limit out of [1, 1000] OR offset < 0
          200 — {entries, total, offset, limit}
        """
        if limit <= 0 or limit > 1000:
            raise HTTPException(
                status_code=422,
                detail=f"limit must be in [1, 1000], got {limit}",
            )
        if offset < 0:
            raise HTTPException(
                status_code=422,
                detail=f"offset must be >= 0, got {offset}",
            )
        ring = getattr(node, "_distribution_log", None)
        if ring is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Distribution log not initialized "
                    "(requires CompensationDistributor watcher wiring)."
                ),
            )
        entries = ring.recent(limit=limit, offset=offset)
        return {
            "entries": [e.to_dict() for e in entries],
            "total": ring.count(),
            "offset": offset,
            "limit": limit,
        }

    @app.get("/admin/watcher-event-dedup", tags=["admin"])
    async def get_watcher_event_dedup() -> Dict[str, Any]:
        """Sprint 552 — operator visibility for the watcher
        event-dedup state shipped in sprints 549/550/551.

        Per-watcher summary of how many on-chain events have been
        marked processed by the InboundMonitor + the 3 event
        watchers, and the most recently marked
        ``(tx_hash, log_index)`` tuple for each. Operators use this
        to verify the dedup is actually firing on their node post-
        restart (sprint 543's checkpoint persistence + sprint 549
        per-event dedup together close the restart-catch-up double-
        process gap).

        Status:
          503 — event dedup store not wired (operator hasn't set
                PRSM_WATCHER_STATE_PERSISTENCE_ENABLED=1; without
                it the watchers fall back to in-memory dedup and
                this surface has nothing to report).
          200 — {watchers: {<watcher_key>: {rows_processed,
                latest_tx_hash, latest_log_index}}}
        """
        store = getattr(node, "_watcher_event_dedup_store", None)
        if store is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Watcher event-dedup store not wired. Set "
                    "PRSM_WATCHER_STATE_PERSISTENCE_ENABLED=1 (and "
                    "optionally PRSM_WATCHER_EVENT_DEDUP_DB=<path> "
                    "to override the default "
                    "~/.prsm/watcher_event_dedup.db) and restart "
                    "the daemon. Without persistence, watchers fall "
                    "back to in-memory dedup and this surface has "
                    "nothing to report."
                ),
            )
        try:
            summary = store.summary()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=(
                    f"event-dedup summary failed: "
                    f"{type(exc).__name__}: {exc!s}"[:300]
                ),
            )
        return {"watchers": summary}

    @app.get("/admin/heartbeat-history")
    async def get_heartbeat_history(
        limit: int = 50,
        offset: int = 0,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Recent on-chain HeartbeatRecorded events observed by the
        StorageSlashingWatcher. Each entry: timestamp, provider,
        onchain_timestamp. Operators verify their heartbeat
        scheduler is actually landing transactions on-chain.

        Status:
          503 — heartbeat log not wired
          422 — limit out of [1, 1000] OR offset < 0
          200 — {entries, total, offset, limit}
        """
        if limit <= 0 or limit > 1000:
            raise HTTPException(
                status_code=422,
                detail=f"limit must be in [1, 1000], got {limit}",
            )
        if offset < 0:
            raise HTTPException(
                status_code=422,
                detail=f"offset must be >= 0, got {offset}",
            )
        ring = getattr(node, "_heartbeat_log", None)
        if ring is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Heartbeat log not initialized "
                    "(requires StorageSlashing watcher wiring)."
                ),
            )
        entries = ring.recent(
            limit=limit, offset=offset, provider=provider,
        )
        return {
            "entries": [e.to_dict() for e in entries],
            "total": ring.count(),
            "offset": offset,
            "limit": limit,
        }

    # ── Sprint 269 — operator-side content filter CRUD ────────
    # Vision §14 "Content moderation" mitigation. Per-operator
    # blocklist; never propagated. R9-SCOPING-1 §7-8 invariants
    # preserved: each operator manages their own list.

    @app.get("/admin/content-filter", tags=["admin"])
    async def get_content_filter() -> Dict[str, Any]:
        """Snapshot the operator's current content filter."""
        store = getattr(node, "_content_filter_store", None)
        if store is None:
            raise HTTPException(
                status_code=503,
                detail="Content filter store not initialized.",
            )
        return store.to_dict()

    @app.post("/admin/chain-exec-ping", tags=["admin"])
    async def chain_exec_ping(body: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Sprint 605 — live Phase 2 round-trip verification.

        Constructs a SendMessage adapter (sprint 596) against the
        live daemon, dispatches a chain_executor_rpc REQUEST to the
        specified peer with the operator-supplied payload, returns
        the response.

        Body: {"peer_id": "<hex>", "payload": "<utf8 string>",
               "timeout": 10.0}
        Returns:
          200 {response: "<utf8>", response_b64: "<b64>"} on success
          400 — missing peer_id / unknown peer
          500 — round-trip failed (timeout / send error / executor
                returned CHAIN_ERROR_KEY)

        For full end-to-end test: both sides set
        PRSM_PARALLAX_STAGE_EXECUTOR_KIND=echo so the server-side
        request handler (sprint 604) echoes the request back.
        """
        peer_id = (body.get("peer_id") or "").strip()
        if not peer_id:
            raise HTTPException(
                status_code=400, detail="peer_id is required",
            )
        # Sprint 624 — accept either `payload` (UTF-8 string) OR
        # `payload_b64` (base64-encoded raw bytes). The latter is
        # required for arbitrary binary wire payloads like
        # encode_message(RunLayerSliceRequest) which contain
        # non-UTF-8 bytes. Both paths feed identical bytes into
        # the SendMessage adapter.
        import base64 as _b64
        payload_b64_str = body.get("payload_b64")
        if payload_b64_str is not None:
            if not isinstance(payload_b64_str, str):
                raise HTTPException(
                    status_code=400,
                    detail="payload_b64 must be a base64 string",
                )
            try:
                request_bytes_override = _b64.b64decode(payload_b64_str)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=400,
                    detail=f"payload_b64 base64 decode failed: {exc}",
                )
        else:
            request_bytes_override = None
        payload_str = body.get("payload", "")
        if not isinstance(payload_str, str):
            raise HTTPException(
                status_code=400, detail="payload must be a string",
            )
        timeout = float(body.get("timeout", 10.0))
        loop = getattr(node, "_loop", None)
        if loop is None:
            raise HTTPException(
                status_code=503, detail="node._loop not initialized",
            )

        from prsm.node.chain_executor_adapters import (
            build_send_message_adapter,
        )
        import base64
        adapter = build_send_message_adapter(node, timeout=timeout)
        request_bytes = (
            request_bytes_override
            if request_bytes_override is not None
            else payload_str.encode("utf-8")
        )

        # The adapter is sync (drives loop via run_async_on_loop) +
        # we're already inside the loop's thread — must dispatch the
        # adapter call to a worker thread to avoid deadlock.
        import asyncio
        try:
            response_bytes = await asyncio.get_running_loop().run_in_executor(
                None, adapter, peer_id, request_bytes,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=(
                    f"chain-exec-ping failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        try:
            response_utf8 = response_bytes.decode("utf-8")
        except UnicodeDecodeError:
            response_utf8 = None
        return {
            "response_b64": base64.b64encode(response_bytes).decode(),
            "response": response_utf8,
            "size_bytes": len(response_bytes),
        }

    @app.post("/admin/content-filter/cids", tags=["admin"])
    async def add_filter_cids(
        body: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        """Add CIDs to the operator's content blocklist.

        Body: {"cids": ["bafy123", "Qm..."]}.
        Idempotent: existing CIDs are no-ops in the `added` count.
        """
        store = getattr(node, "_content_filter_store", None)
        if store is None:
            raise HTTPException(
                status_code=503,
                detail="Content filter store not initialized.",
            )
        cids = body.get("cids")
        if not isinstance(cids, list):
            raise HTTPException(
                status_code=422,
                detail="body.cids must be a list of strings",
            )
        if len(cids) > 1000:
            raise HTTPException(
                status_code=422,
                detail=f"max 1000 CIDs per request; got {len(cids)}",
            )
        added = store.add_cids(cids)
        return {
            "added": added,
            "total": store.to_dict()["count_cids"],
        }

    @app.delete(
        "/admin/content-filter/cids/{cid}", tags=["admin"],
    )
    async def remove_filter_cid(cid: str) -> Dict[str, Any]:
        """Remove a single CID from the blocklist."""
        store = getattr(node, "_content_filter_store", None)
        if store is None:
            raise HTTPException(
                status_code=503,
                detail="Content filter store not initialized.",
            )
        removed = store.remove_cid(cid)
        if not removed:
            raise HTTPException(
                status_code=404,
                detail=f"cid={cid!r} not in blocklist",
            )
        return {
            "removed": cid,
            "total": store.to_dict()["count_cids"],
        }

    @app.post("/admin/content-filter/tags", tags=["admin"])
    async def add_filter_tags(
        body: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        """Add model tags to the operator's content blocklist."""
        store = getattr(node, "_content_filter_store", None)
        if store is None:
            raise HTTPException(
                status_code=503,
                detail="Content filter store not initialized.",
            )
        tags = body.get("tags")
        if not isinstance(tags, list):
            raise HTTPException(
                status_code=422,
                detail="body.tags must be a list of strings",
            )
        if len(tags) > 100:
            raise HTTPException(
                status_code=422,
                detail=f"max 100 tags per request; got {len(tags)}",
            )
        added = store.add_tags(tags)
        return {
            "added": added,
            "total": store.to_dict()["count_tags"],
        }

    @app.delete(
        "/admin/content-filter/tags/{tag}", tags=["admin"],
    )
    async def remove_filter_tag(tag: str) -> Dict[str, Any]:
        store = getattr(node, "_content_filter_store", None)
        if store is None:
            raise HTTPException(
                status_code=503,
                detail="Content filter store not initialized.",
            )
        removed = store.remove_tag(tag)
        if not removed:
            raise HTTPException(
                status_code=404,
                detail=f"tag={tag!r} not in blocklist",
            )
        return {
            "removed": tag,
            "total": store.to_dict()["count_tags"],
        }

    @app.post("/admin/content-filter/action", tags=["admin"])
    async def set_filter_action(
        body: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        """Set the action mode: refuse / log_and_refuse /
        silent_refuse."""
        store = getattr(node, "_content_filter_store", None)
        if store is None:
            raise HTTPException(
                status_code=503,
                detail="Content filter store not initialized.",
            )
        action = body.get("action")
        if not isinstance(action, str):
            raise HTTPException(
                status_code=422,
                detail="body.action must be a string",
            )
        try:
            store.set_action(action)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return {"action_on_match": action}

    # ── Sprint 272 — Foundation takedown notice intake ────────
    # Per Vision §14 "Content moderation" mitigation: Foundation
    # operates a takedown process for DMCA / legal notices.
    # Per R9-SCOPING-1 §8 invariant: this is information
    # distribution only — never enforcement. Operators read
    # notices via /admin/takedown-notices and VOLUNTARILY
    # update their own ContentFilterStore (sprint 269) if they
    # decide to act on a given notice.

    @app.post("/admin/takedown-notice", tags=["admin"])
    async def record_takedown_notice(
        body: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        """Record a received takedown notice.

        Body fields (all required string except notice_text):
          - target_cid:    CID the notice cites
          - sender:        email / org / legal entity
          - jurisdiction:  e.g. "US-DMCA", "EU-DSA"
          - basis:         short statutory citation
          - notice_text:   full notice body (optional, capped 8KB)

        Returns the assigned notice_id + timestamp + status.
        """
        ring = getattr(node, "_takedown_notice_ring", None)
        if ring is None:
            raise HTTPException(
                status_code=503,
                detail="Takedown notice ring not initialized.",
            )
        required = ("target_cid", "sender", "jurisdiction", "basis")
        missing = [k for k in required if not body.get(k)]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"missing required field(s): "
                    f"{', '.join(missing)}"
                ),
            )
        try:
            entry = ring.record(
                target_cid=body["target_cid"],
                sender=body["sender"],
                jurisdiction=body["jurisdiction"],
                basis=body["basis"],
                notice_text=body.get("notice_text", ""),
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return entry.to_dict()

    @app.get("/admin/takedown-notices", tags=["admin"])
    async def list_takedown_notices(
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        target_cid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Paginated list of received takedown notices."""
        if limit <= 0 or limit > 1000:
            raise HTTPException(
                status_code=422,
                detail=f"limit must be in [1, 1000], got {limit}",
            )
        if offset < 0:
            raise HTTPException(
                status_code=422,
                detail=f"offset must be >= 0, got {offset}",
            )
        ring = getattr(node, "_takedown_notice_ring", None)
        if ring is None:
            raise HTTPException(
                status_code=503,
                detail="Takedown notice ring not initialized.",
            )
        try:
            entries = ring.recent(
                limit=limit, offset=offset,
                status=status, target_cid=target_cid,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return {
            "notices": [e.to_dict() for e in entries],
            "total": ring.count(),
            "offset": offset,
            "limit": limit,
        }

    # ── Sprint 273 — bridge: notice → operator filter ─────
    # One-call operator action that adds a notice's
    # target_cid to the operator's ContentFilterStore AND
    # marks the notice acknowledged. Operator EXPLICITLY
    # initiates — Foundation never auto-propagates.

    @app.post(
        "/admin/content-filter/from-notice/{notice_id}",
        tags=["admin"],
    )
    async def apply_takedown_notice_to_filter(
        notice_id: str,
    ) -> Dict[str, Any]:
        ring = getattr(node, "_takedown_notice_ring", None)
        if ring is None:
            raise HTTPException(
                status_code=503,
                detail="Takedown notice ring not initialized.",
            )
        store = getattr(node, "_content_filter_store", None)
        if store is None:
            raise HTTPException(
                status_code=503,
                detail="Content filter store not initialized.",
            )
        entry = ring.get(notice_id)
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail=f"no notice with id={notice_id!r}",
            )
        added = store.add_cids([entry.target_cid])
        updated = ring.set_status(notice_id, "acknowledged")
        return {
            "notice_id": notice_id,
            "target_cid": entry.target_cid,
            "added": added,
            "notice_status": (
                updated.status if updated else "received"
            ),
        }

    # ── Sprint 274 — notice status transitions ────────────
    # General-purpose status mutation (vs sprint-273 bridge
    # which is acknowledged-only with filter side effect).
    # Used for disputed/expired transitions where the
    # operator does NOT want to apply the notice but does
    # want to update its lifecycle status.

    @app.post(
        "/admin/takedown-notices/{notice_id}/status",
        tags=["admin"],
    )
    async def set_takedown_notice_status(
        notice_id: str,
        body: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        ring = getattr(node, "_takedown_notice_ring", None)
        if ring is None:
            raise HTTPException(
                status_code=503,
                detail="Takedown notice ring not initialized.",
            )
        status = body.get("status")
        if not status:
            raise HTTPException(
                status_code=422,
                detail="missing required field: status",
            )
        try:
            updated = ring.set_status(notice_id, status)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        if updated is None:
            raise HTTPException(
                status_code=404,
                detail=f"no notice with id={notice_id!r}",
            )
        return updated.to_dict()

    @app.get(
        "/admin/takedown-notices/{notice_id}", tags=["admin"],
    )
    async def get_takedown_notice(notice_id: str) -> Dict[str, Any]:
        """Fetch a single notice by notice_id."""
        ring = getattr(node, "_takedown_notice_ring", None)
        if ring is None:
            raise HTTPException(
                status_code=503,
                detail="Takedown notice ring not initialized.",
            )
        entry = ring.get(notice_id)
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail=f"no notice with id={notice_id!r}",
            )
        return entry.to_dict()

    # ── Sprint 299 — insurance fund tracker + recovery ────
    # Vision §14 mitigation item 2: 5% treasury reserve for
    # exploit recovery; public on-chain verification.
    # Recovery transfer composer-only — Safe-uploaded.

    class _InsuranceRecoveryRequest(BaseModel):
        recipient: str
        amount_wei: int
        reason: str

    @app.get(
        "/admin/insurance-fund/status", tags=["admin"],
    )
    async def get_insurance_fund_status() -> Dict[str, Any]:
        tracker = getattr(
            node, "_insurance_fund_tracker", None,
        )
        if tracker is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Insurance fund tracker not initialized."
                ),
            )
        status = tracker.status()
        return status.to_dict()

    @app.post(
        "/admin/insurance-fund/compose-recovery",
        tags=["admin"],
    )
    async def compose_insurance_recovery(
        body: _InsuranceRecoveryRequest,
    ) -> Dict[str, Any]:
        tracker = getattr(
            node, "_insurance_fund_tracker", None,
        )
        if tracker is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Insurance fund tracker not initialized."
                ),
            )
        try:
            tx = tracker.compose_recovery_transfer_tx(
                recipient=body.recipient,
                amount_wei=body.amount_wei,
                reason=body.reason,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return tx

    # ── Sprint 298 — emergency pause composer surface ─────
    # Vision §14 smart-contract exploit-response engineering.
    # PRSM never executes pause directly — Foundation Safe
    # holds the privilege. This endpoint composes the
    # multi-sig-uploadable tx payload + surfaces per-contract
    # paused state for operator monitoring.

    class _EmergencyPauseComposeRequest(BaseModel):
        action: str  # "pause" or "unpause"
        contract_name: str

    @app.get(
        "/admin/emergency-pause/status", tags=["admin"],
    )
    async def get_emergency_pause_status() -> Dict[str, Any]:
        pause_client = getattr(
            node, "_emergency_pause_client", None,
        )
        if pause_client is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Emergency pause client not initialized."
                ),
            )
        statuses = pause_client.status_all()
        return {
            "chain_id": getattr(
                pause_client, "_chain_id", None,
            ),
            "contracts": {
                name: status.to_dict()
                for name, status in statuses.items()
            },
        }

    @app.post(
        "/admin/emergency-pause/compose", tags=["admin"],
    )
    async def compose_emergency_pause(
        body: _EmergencyPauseComposeRequest,
    ) -> Dict[str, Any]:
        pause_client = getattr(
            node, "_emergency_pause_client", None,
        )
        if pause_client is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Emergency pause client not initialized."
                ),
            )
        action = (body.action or "").strip().lower()
        if action not in ("pause", "unpause"):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"action must be 'pause' or 'unpause', "
                    f"got {action!r}"
                ),
            )
        try:
            if action == "pause":
                tx = pause_client.compose_pause_tx(
                    body.contract_name,
                )
            else:
                tx = pause_client.compose_unpause_tx(
                    body.contract_name,
                )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return tx

    # ── Sprint 300 — responsible-disclosure intake ────────
    # Vision §14 mitigation item 3: bug bounty / coordinated
    # disclosure surface. /submit is intentionally open (no
    # auth) — security researchers may be anonymous. Workflow
    # transitions + payout composer are admin paths (gated by
    # the same security middleware as other /admin/* routes).
    # Payout itself is composer-only — Foundation Safe.

    class _DisclosureSubmitRequest(BaseModel):
        severity: str
        summary: str
        affected_contracts: List[str] = []
        researcher_contact: str
        details: str = ""

    class _DisclosureUpdateRequest(BaseModel):
        new_status: str
        triage_notes: Optional[str] = None
        payout_ftns: Optional[int] = None

    class _DisclosureComposePayoutRequest(BaseModel):
        recipient: str

    class _DisclosureRecordTxRequest(BaseModel):
        tx_hash: str

    def _require_disclosure_intake():
        intake = getattr(node, "_disclosure_intake", None)
        if intake is None:
            raise HTTPException(
                status_code=503,
                detail="Disclosure intake not initialized.",
            )
        return intake

    @app.post("/admin/disclosure/submit", tags=["admin"])
    async def disclosure_submit(
        body: _DisclosureSubmitRequest,
    ) -> Dict[str, Any]:
        from prsm.economy.web3.disclosure_intake import (
            DisclosureSeverity,
        )
        intake = _require_disclosure_intake()
        sev_raw = (body.severity or "").strip().lower()
        try:
            severity = DisclosureSeverity(sev_raw)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"severity must be one of "
                    f"{[s.value for s in DisclosureSeverity]}"
                ),
            )
        try:
            record = intake.submit(
                severity=severity,
                summary=body.summary,
                affected_contracts=body.affected_contracts,
                researcher_contact=body.researcher_contact,
                details=body.details,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return record.to_dict()

    @app.get("/admin/disclosure", tags=["admin"])
    async def disclosure_list(
        severity: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        from prsm.economy.web3.disclosure_intake import (
            DisclosureSeverity, DisclosureStatus,
        )
        intake = _require_disclosure_intake()
        sev_obj = None
        status_obj = None
        if severity:
            try:
                sev_obj = DisclosureSeverity(severity.lower())
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=f"invalid severity {severity!r}",
                )
        if status:
            try:
                status_obj = DisclosureStatus(status.lower())
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=f"invalid status {status!r}",
                )
        records = intake.list(
            severity=sev_obj, status=status_obj,
        )
        return {
            "records": [r.to_dict() for r in records],
            "count": len(records),
        }

    @app.get(
        "/admin/disclosure/{disclosure_id}", tags=["admin"],
    )
    async def disclosure_get(
        disclosure_id: str,
    ) -> Dict[str, Any]:
        intake = _require_disclosure_intake()
        record = intake.get(disclosure_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"disclosure {disclosure_id!r} not found"
                ),
            )
        return record.to_dict()

    @app.post(
        "/admin/disclosure/{disclosure_id}/update",
        tags=["admin"],
    )
    async def disclosure_update(
        disclosure_id: str,
        body: _DisclosureUpdateRequest,
    ) -> Dict[str, Any]:
        from prsm.economy.web3.disclosure_intake import (
            DisclosureStatus,
        )
        intake = _require_disclosure_intake()
        if intake.get(disclosure_id) is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"disclosure {disclosure_id!r} not found"
                ),
            )
        try:
            new_status = DisclosureStatus(
                (body.new_status or "").lower(),
            )
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"invalid new_status {body.new_status!r}"
                ),
            )
        try:
            record = intake.update_status(
                disclosure_id, new_status,
                triage_notes=body.triage_notes,
                payout_ftns=body.payout_ftns,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return record.to_dict()

    @app.post(
        "/admin/disclosure/{disclosure_id}/compose-payout",
        tags=["admin"],
    )
    async def disclosure_compose_payout(
        disclosure_id: str,
        body: _DisclosureComposePayoutRequest,
    ) -> Dict[str, Any]:
        from prsm.economy.web3.disclosure_intake import (
            compose_bounty_payout_tx,
        )
        intake = _require_disclosure_intake()
        if intake.get(disclosure_id) is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"disclosure {disclosure_id!r} not found"
                ),
            )
        ftns_token = getattr(
            node, "_disclosure_ftns_token_address", None,
        )
        if not ftns_token:
            raise HTTPException(
                status_code=503,
                detail=(
                    "FTNS token address not configured "
                    "(set PRSM_NETWORK or "
                    "FTNS_TOKEN_ADDRESS env)."
                ),
            )
        import re
        if not re.fullmatch(
            r"0x[0-9a-fA-F]{40}", body.recipient or "",
        ):
            raise HTTPException(
                status_code=422,
                detail="invalid recipient address format",
            )
        try:
            tx = compose_bounty_payout_tx(
                intake=intake,
                disclosure_id=disclosure_id,
                recipient=body.recipient,
                ftns_token_address=ftns_token,
                chain_id=getattr(
                    node, "_disclosure_chain_id", None,
                ),
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return tx

    @app.post(
        "/admin/disclosure/{disclosure_id}/record-payout-tx",
        tags=["admin"],
    )
    async def disclosure_record_tx(
        disclosure_id: str,
        body: _DisclosureRecordTxRequest,
    ) -> Dict[str, Any]:
        intake = _require_disclosure_intake()
        if intake.get(disclosure_id) is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"disclosure {disclosure_id!r} not found"
                ),
            )
        if not body.tx_hash:
            raise HTTPException(
                status_code=422,
                detail="tx_hash must be non-empty",
            )
        try:
            record = intake.record_payout_tx(
                disclosure_id, tx_hash=body.tx_hash,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return record.to_dict()

    # ── Sprint 318d — enterprise metrics endpoint ────────
    # Prometheus text exposition of the
    # prsm.enterprise.metrics REGISTRY. Operators scrape
    # for dashboards + alerts.

    @app.get(
        "/admin/enterprise/metrics",
        response_class=Response,
    )
    async def enterprise_metrics() -> Response:
        from prsm.enterprise.metrics import REGISTRY
        # Update gauges that snapshot live state on read
        # (if the orchestrators are wired into the node)
        try:
            from prsm.enterprise.metrics import (
                FL_JOBS_PENDING,
                PIPELINE_JOBS_PENDING,
            )
            fl_orch = getattr(
                node,
                "_federated_learning_orchestrator", None,
            )
            if fl_orch is not None:
                from prsm.enterprise.federated_learning import (
                    JobStatus,
                )
                FL_JOBS_PENDING.set(len(
                    fl_orch.list_jobs(
                        status=JobStatus.PROPOSED,
                    ),
                ))
            pipeline_orch = getattr(
                node,
                "_pipeline_inference_orchestrator", None,
            )
            if pipeline_orch is not None:
                from prsm.compute.inference.pipeline_orchestrator import (
                    PipelineJobStatus,
                )
                PIPELINE_JOBS_PENDING.set(sum(
                    1 for j in pipeline_orch.list_jobs()
                    if j.status == PipelineJobStatus.PROPOSED
                ))
        except Exception:  # noqa: BLE001
            # Snapshot is best-effort — counters from
            # observed events are authoritative
            pass
        return Response(
            content=REGISTRY.to_prometheus_text(),
            media_type=(
                "text/plain; version=0.0.4; "
                "charset=utf-8"
            ),
        )

    # ── Sprint 316a — TP worker shard endpoint ───────────
    # Each TP worker node holds its own weight shard
    # (node._tp_weight_shard) and computes the partial
    # matmul X @ W_local for any incoming input X.

    class _TPShardRequest(BaseModel):
        shard_id: int = Field(ge=0)
        input_activations_b64: str

    @app.post(
        "/compute/inference/tensor_parallel/shard",
    )
    async def tp_shard_forward(
        body: _TPShardRequest,
    ) -> Dict[str, Any]:
        weight_shard = getattr(
            node, "_tp_weight_shard", None,
        )
        if weight_shard is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "TP worker weight shard not "
                    "configured (set "
                    "node._tp_weight_shard or wire from "
                    "operator startup script)"
                ),
            )
        import base64 as _b64
        try:
            input_bytes = _b64.b64decode(
                body.input_activations_b64, validate=True,
            )
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"input_activations_b64 not valid: {e}"
                ),
            )
        from prsm.compute.inference.pytorch_stage_runner import (
            deserialize_activation,
            serialize_activation,
        )
        try:
            x = deserialize_activation(input_bytes)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=422,
                detail=(
                    f"input deserialization failed: "
                    f"{exc}"
                ),
            )
        try:
            import torch
            with torch.no_grad():
                partial = x @ weight_shard
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=502,
                detail=(
                    f"partial matmul failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
        worker_node_id = (
            node.identity.node_id if node.identity
            else "unknown"
        )
        return {
            "shard_id": body.shard_id,
            "worker_node_id": worker_node_id,
            "output_partial_b64": _b64.b64encode(
                serialize_activation(partial),
            ).decode("ascii"),
        }

    # ── Sprint 313 — pipeline stage worker endpoint ───────
    # Each "remote" stage node exposes this; the orchestrator
    # uses http_stage_runner (sprint 313) to call it. v1 is
    # orchestrator-driven (no worker-to-worker chaining).

    class _PipelineStageRequest(BaseModel):
        job_id: str
        round_id: str
        stage_id: int = Field(ge=0)
        layer_indices: List[int]
        input_activations_b64: str

    @app.post("/compute/inference/pipeline/stage")
    async def pipeline_stage_run(
        body: _PipelineStageRequest,
    ) -> Dict[str, Any]:
        runner = getattr(
            node, "_pipeline_stage_runner", None,
        )
        if runner is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "pipeline stage runner not configured "
                    "on this node (set "
                    "PRSM_PIPELINE_STAGE_RUNNER_ENABLED=1 "
                    "to enable default stub runner)"
                ),
            )
        if not body.layer_indices:
            raise HTTPException(
                status_code=422,
                detail="layer_indices must be non-empty",
            )
        import base64 as _b64
        try:
            input_bytes = _b64.b64decode(
                body.input_activations_b64, validate=True,
            )
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"input_activations_b64 not valid: {e}"
                ),
            )
        try:
            output_bytes = runner(
                input_activations=input_bytes,
                stage_id=body.stage_id,
                layer_indices=list(body.layer_indices),
            )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=502,
                detail=(
                    f"stage runner raised: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
        worker_node_id = (
            node.identity.node_id if node.identity
            else "unknown"
        )
        return {
            "job_id": body.job_id,
            "round_id": body.round_id,
            "stage_id": body.stage_id,
            "worker_node_id": worker_node_id,
            "output_activations_b64": _b64.b64encode(
                output_bytes,
            ).decode("ascii"),
        }

    # ── Sprint 312 — pipeline inference orchestrator ──────
    # Coordinates multi-stage TEE-attested inference across
    # a partitioned model. Each stage runs in-process for
    # v1 (cross-node activation streaming = sprint 313).

    class _PipelineProposeRequest(BaseModel):
        model_id: str
        partition: Dict[str, Any]

    class _PipelineExecuteRequest(BaseModel):
        prompt_b64: str

    def _require_pipeline_orchestrator():
        o = getattr(
            node,
            "_pipeline_inference_orchestrator",
            None,
        )
        if o is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Pipeline inference orchestrator not "
                    "initialized (set "
                    "PRSM_PIPELINE_ORCHESTRATOR_PRIVKEY env)"
                ),
            )
        return o

    @app.post(
        "/admin/inference/pipeline/job", tags=["admin"],
    )
    async def pipeline_propose_job(
        body: _PipelineProposeRequest,
    ) -> Dict[str, Any]:
        from prsm.compute.inference.pipeline_partition import (
            PipelinePartition,
        )
        orch = _require_pipeline_orchestrator()
        try:
            partition = PipelinePartition.from_dict(
                body.partition,
            )
        except (KeyError, ValueError, TypeError) as e:
            raise HTTPException(
                status_code=422,
                detail=f"malformed partition: {e}",
            )
        try:
            job = orch.propose_job(
                model_id=body.model_id, partition=partition,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=422, detail=str(e),
            )
        return job.to_dict()

    @app.get(
        "/admin/inference/pipeline/job", tags=["admin"],
    )
    async def pipeline_list_jobs() -> Dict[str, Any]:
        orch = _require_pipeline_orchestrator()
        return {
            "jobs": [j.to_dict() for j in orch.list_jobs()],
        }

    @app.get(
        "/admin/inference/pipeline/job/{job_id}",
        tags=["admin"],
    )
    async def pipeline_get_job(
        job_id: str,
    ) -> Dict[str, Any]:
        orch = _require_pipeline_orchestrator()
        job = orch.get_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail=f"job {job_id!r} not found",
            )
        return job.to_dict()

    @app.post(
        "/admin/inference/pipeline/job/{job_id}/execute",
        tags=["admin"],
    )
    async def pipeline_execute(
        job_id: str, body: _PipelineExecuteRequest,
    ) -> Dict[str, Any]:
        from prsm.compute.inference.pipeline_stage import (
            deterministic_stub_stage_runner,
        )
        import base64 as _b64
        orch = _require_pipeline_orchestrator()
        if orch.get_job(job_id) is None:
            raise HTTPException(
                status_code=404,
                detail=f"job {job_id!r} not found",
            )
        try:
            prompt = _b64.b64decode(
                body.prompt_b64, validate=True,
            )
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"prompt_b64 not valid: {e}",
            )
        job = orch.get_job(job_id)
        n_stages = job.partition.n_stages
        # v1: API uses default stub runners. Operators
        # wiring real runners do so by calling the
        # orchestrator's execute() directly from a
        # privileged code path.
        runners = [
            deterministic_stub_stage_runner()
            for _ in range(n_stages)
        ]
        try:
            rnd = orch.execute(
                job_id, prompt=prompt,
                stage_runners=runners,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=422, detail=str(e),
            )
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"pipeline execution failed: "
                    f"{type(e).__name__}: {e}"
                ),
            )
        return rnd.to_dict()

    @app.get(
        "/admin/inference/pipeline/job/{job_id}/round",
        tags=["admin"],
    )
    async def pipeline_get_round(
        job_id: str,
    ) -> Dict[str, Any]:
        orch = _require_pipeline_orchestrator()
        rnd = orch.get_round(job_id)
        if rnd is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"no round for job {job_id!r} (not "
                    f"executed yet?)"
                ),
            )
        return rnd.to_dict()

    # ── Sprint 308b — worker-side /compute/train shim ─────
    # Vision §7 Enterprise Confidentiality Mode capstone
    # follow-on. Workers receive a round assignment from the
    # orchestrator, run their training strategy here, sign
    # the gradient + their TEE attestation under their
    # Ed25519 privkey, and return the signed update. The
    # caller submits to /admin/federated/job/.../update.

    class _ComputeTrainRequest(BaseModel):
        job_id: str
        round_index: int = Field(ge=0)
        dataset_cid: str
        sample_count: int = Field(ge=0)
        # Sprint 308c — when set, the worker seals the
        # gradient to this orchestrator X25519 pubkey
        # before signing.
        transport_pubkey_b64: Optional[str] = None

    @app.post("/compute/train")
    async def compute_train(
        body: _ComputeTrainRequest,
    ) -> Dict[str, Any]:
        privkey = getattr(
            node, "_federated_worker_privkey_b64", None,
        )
        if not privkey:
            raise HTTPException(
                status_code=503,
                detail=(
                    "worker privkey not configured; set "
                    "PRSM_FEDERATED_WORKER_PRIVKEY env"
                ),
            )
        # Validate privkey format eagerly (loud-fail on
        # operator misconfig instead of silent signature
        # garbage)
        try:
            from prsm.enterprise.federated_learning import (
                _load_ed25519_priv,
            )
            _load_ed25519_priv(privkey)
        except ValueError as e:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"PRSM_FEDERATED_WORKER_PRIVKEY "
                    f"malformed: {e}"
                ),
            )

        # Surface this node's TEE attestation blob in the
        # signed payload. Workers without attestation
        # configured pass empty string — the orchestrator
        # can decide whether to accept that based on its
        # own policy.
        import base64 as _b64
        attestation_blob = getattr(
            node, "_tee_node_attestation_blob", None,
        )
        attestation_b64 = (
            _b64.b64encode(bytes(attestation_blob)).decode()
            if attestation_blob else ""
        )

        worker_node_id = (
            node.identity.node_id if node.identity
            else "unknown"
        )

        from prsm.compute.train import (
            compute_signed_gradient_update,
        )
        try:
            update = compute_signed_gradient_update(
                job_id=body.job_id,
                round_index=body.round_index,
                dataset_cid=body.dataset_cid,
                sample_count=body.sample_count,
                worker_node_id=worker_node_id,
                worker_privkey_b64=privkey,
                worker_attestation_b64=attestation_b64,
                transport_pubkey_b64=(
                    body.transport_pubkey_b64
                ),
            )
        except ValueError as e:
            raise HTTPException(
                status_code=422, detail=str(e),
            )
        return update.to_dict()

    # ── Sprint 308 — federated-learning orchestrator ──────
    # Vision §7 Enterprise Confidentiality Mode capstone.
    # Coordinates round-by-round training across a fleet
    # of TEE-attested workers that see only gradients,
    # never plaintext.

    class _FederatedProposeJob(BaseModel):
        model_id: str
        dataset_cids: List[str] = []
        worker_pool: List[str]
        rounds_target: int = Field(ge=1, le=10000)
        min_workers_per_round: int = Field(ge=1)
        aggregation: str
        # Sprint 308a — opt-in hardening fields
        require_signed_updates: bool = False
        dp_policy: Optional[Dict[str, float]] = None
        # Sprint 308c — opt-in transport encryption
        transport_pubkey_b64: Optional[str] = None

    class _FederatedWorkerKey(BaseModel):
        node_id: str
        signing_pubkey_b64: str

    class _FederatedGradientUpdate(BaseModel):
        round_index: int = Field(ge=0)
        worker_node_id: str
        gradient_b64: str
        sample_count: int = Field(ge=0)
        worker_attestation_b64: str = ""
        worker_signature_b64: str = ""
        timestamp: float = 0.0

    def _require_federated_orchestrator():
        o = getattr(
            node, "_federated_learning_orchestrator", None,
        )
        if o is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "FederatedLearningOrchestrator not "
                    "initialized."
                ),
            )
        return o

    @app.post("/admin/federated/job", tags=["admin"])
    async def federated_propose_job(
        body: _FederatedProposeJob,
    ) -> Dict[str, Any]:
        from prsm.enterprise.federated_learning import (
            AggregationStrategy,
        )
        orch = _require_federated_orchestrator()
        try:
            agg = AggregationStrategy(
                (body.aggregation or "").lower(),
            )
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"invalid aggregation "
                    f"{body.aggregation!r}; expected "
                    f"'fedavg' or 'fedmedian'"
                ),
            )
        from prsm.enterprise.federated_learning import (
            DPPolicy,
        )
        dp_policy = None
        if body.dp_policy is not None:
            try:
                dp_policy = DPPolicy.from_dict(
                    body.dp_policy,
                )
            except (KeyError, ValueError) as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"invalid dp_policy: {e}",
                )
        try:
            job = orch.propose_job(
                model_id=body.model_id,
                dataset_cids=body.dataset_cids,
                worker_pool=body.worker_pool,
                rounds_target=body.rounds_target,
                min_workers_per_round=(
                    body.min_workers_per_round
                ),
                aggregation=agg,
                require_signed_updates=(
                    body.require_signed_updates
                ),
                dp_policy=dp_policy,
                transport_pubkey_b64=(
                    body.transport_pubkey_b64
                ),
            )
        except ValueError as e:
            raise HTTPException(
                status_code=422, detail=str(e),
            )
        return job.to_dict()

    @app.get(
        "/admin/federated/transport-pubkey",
        tags=["admin"],
    )
    async def federated_transport_pubkey() -> Dict[str, Any]:
        """Sprint 308c — derive + surface the orchestrator's
        transport pubkey from its in-memory privkey.
        Operators distribute this pubkey to enterprises;
        enterprises pin it on each FederatedJob."""
        priv = getattr(
            node,
            "_federated_orchestrator_transport_privkey_b64",
            None,
        )
        if not priv:
            raise HTTPException(
                status_code=503,
                detail=(
                    "orchestrator transport privkey not "
                    "configured; set "
                    "PRSM_FEDERATED_ORCHESTRATOR_"
                    "TRANSPORT_PRIVKEY env"
                ),
            )
        try:
            from prsm.enterprise.federated_learning import (
                _load_x25519_priv,
            )
            from cryptography.hazmat.primitives import (
                serialization as _ser,
            )
            import base64 as _b64
            x_priv = _load_x25519_priv(priv)
            pub_raw = x_priv.public_key().public_bytes(
                encoding=_ser.Encoding.Raw,
                format=_ser.PublicFormat.Raw,
            )
            return {
                "transport_pubkey_b64": _b64.b64encode(
                    pub_raw,
                ).decode("ascii"),
            }
        except ValueError as e:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"orchestrator transport privkey "
                    f"malformed: {e}"
                ),
            )

    @app.post(
        "/admin/federated/worker-key", tags=["admin"],
    )
    async def federated_register_worker_key(
        body: _FederatedWorkerKey,
    ) -> Dict[str, Any]:
        from prsm.enterprise.federated_learning import (
            WorkerKey,
        )
        orch = _require_federated_orchestrator()
        try:
            orch.register_worker_key(WorkerKey(
                node_id=body.node_id,
                signing_pubkey_b64=body.signing_pubkey_b64,
            ))
        except ValueError as e:
            raise HTTPException(
                status_code=422, detail=str(e),
            )
        return {
            "node_id": body.node_id,
            "signing_pubkey_b64": body.signing_pubkey_b64,
        }

    @app.get(
        "/admin/federated/worker-key", tags=["admin"],
    )
    async def federated_list_worker_keys() -> Dict[str, Any]:
        orch = _require_federated_orchestrator()
        return {
            "worker_keys": [
                k.to_dict()
                for k in orch.list_worker_keys()
            ],
        }

    @app.get("/admin/federated/job", tags=["admin"])
    async def federated_list_jobs(
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        from prsm.enterprise.federated_learning import (
            JobStatus,
        )
        orch = _require_federated_orchestrator()
        status_obj = None
        if status:
            try:
                status_obj = JobStatus(status.lower())
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=f"invalid status {status!r}",
                )
        jobs = orch.list_jobs(status=status_obj)
        return {"jobs": [j.to_dict() for j in jobs]}

    @app.get(
        "/admin/federated/job/{job_id}", tags=["admin"],
    )
    async def federated_get_job(
        job_id: str,
    ) -> Dict[str, Any]:
        orch = _require_federated_orchestrator()
        job = orch.get_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail=f"job {job_id!r} not found",
            )
        return job.to_dict()

    @app.post(
        "/admin/federated/job/{job_id}/issue-round",
        tags=["admin"],
    )
    async def federated_issue_round(
        job_id: str,
    ) -> Dict[str, Any]:
        orch = _require_federated_orchestrator()
        if orch.get_job(job_id) is None:
            raise HTTPException(
                status_code=404,
                detail=f"job {job_id!r} not found",
            )
        try:
            rnd = orch.issue_round(job_id)
        except ValueError as e:
            raise HTTPException(
                status_code=422, detail=str(e),
            )
        return rnd.to_dict()

    @app.post(
        "/admin/federated/job/{job_id}/update",
        tags=["admin"],
    )
    async def federated_accept_update(
        job_id: str, body: _FederatedGradientUpdate,
    ) -> Dict[str, Any]:
        from prsm.enterprise.federated_learning import (
            GradientUpdate,
        )
        orch = _require_federated_orchestrator()
        if orch.get_job(job_id) is None:
            raise HTTPException(
                status_code=404,
                detail=f"job {job_id!r} not found",
            )
        u = GradientUpdate(
            job_id=job_id,
            round_index=body.round_index,
            worker_node_id=body.worker_node_id,
            gradient_b64=body.gradient_b64,
            sample_count=body.sample_count,
            worker_attestation_b64=(
                body.worker_attestation_b64
            ),
            worker_signature_b64=body.worker_signature_b64,
            timestamp=body.timestamp,
        )
        try:
            orch.accept_gradient_update(u)
        except ValueError as e:
            raise HTTPException(
                status_code=422, detail=str(e),
            )
        return {"status": "accepted"}

    @app.post(
        "/admin/federated/job/{job_id}/aggregate/{round_index}",
        tags=["admin"],
    )
    async def federated_aggregate(
        job_id: str, round_index: int,
    ) -> Dict[str, Any]:
        orch = _require_federated_orchestrator()
        if orch.get_job(job_id) is None:
            raise HTTPException(
                status_code=404,
                detail=f"job {job_id!r} not found",
            )
        try:
            rnd = orch.aggregate_round(
                job_id, round_index,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=422, detail=str(e),
            )
        return rnd.to_dict()

    @app.get(
        "/admin/federated/job/{job_id}/round/{round_index}",
        tags=["admin"],
    )
    async def federated_get_round(
        job_id: str, round_index: int,
    ) -> Dict[str, Any]:
        orch = _require_federated_orchestrator()
        rnd = orch.get_round(job_id, round_index)
        if rnd is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"round {round_index} not found for "
                    f"job {job_id!r}"
                ),
            )
        return rnd.to_dict()

    # ── Sprint 306 — $CORP authorization capability ───────
    # Vision §7 Enterprise Confidentiality Mode layer 2:
    # ergonomics + accounting + audit. Soulbound capability
    # via dual-signature (issuer + subject). NOT the security
    # gate — that's the encryption (304) + TEE policy (305).
    # Sprint 306a wires header-driven redemption into the
    # /compute/* dispatch path.

    class _CorpRegisterIssuerRequest(BaseModel):
        issuer_id: str
        signing_pubkey_b64: str

    class _CorpRedeemRequest(BaseModel):
        capability: Dict[str, Any]
        request: Dict[str, Any]

    def _require_corp_store():
        s = getattr(node, "_corp_capability_store", None)
        if s is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "$CORP capability store not initialized."
                ),
            )
        return s

    @app.post("/admin/corp/issuer", tags=["admin"])
    async def corp_register_issuer(
        body: _CorpRegisterIssuerRequest,
    ) -> Dict[str, Any]:
        from prsm.enterprise.corp_capability import (
            CorpIssuer,
        )
        store = _require_corp_store()
        try:
            store.register_issuer(CorpIssuer(
                issuer_id=body.issuer_id,
                signing_pubkey_b64=body.signing_pubkey_b64,
            ))
        except ValueError as e:
            raise HTTPException(
                status_code=422, detail=str(e),
            )
        return {
            "issuer_id": body.issuer_id,
            "signing_pubkey_b64": body.signing_pubkey_b64,
        }

    @app.get("/admin/corp/issuer", tags=["admin"])
    async def corp_list_issuers() -> Dict[str, Any]:
        store = _require_corp_store()
        return {
            "issuers": [
                i.to_dict() for i in store.list_issuers()
            ],
        }

    @app.post(
        "/admin/corp/capability/redeem", tags=["admin"],
    )
    async def corp_redeem(
        body: _CorpRedeemRequest,
    ) -> Dict[str, Any]:
        from prsm.enterprise.corp_capability import (
            CorpCapability, RedemptionRequest,
        )
        store = _require_corp_store()
        try:
            cap = CorpCapability.from_dict(body.capability)
            req = RedemptionRequest.from_dict(body.request)
        except (KeyError, ValueError) as e:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"malformed capability or request: {e}"
                ),
            )
        result = store.redeem(cap, req)
        return result.to_dict()

    @app.get(
        "/admin/corp/capability/{capability_id}/ledger",
        tags=["admin"],
    )
    async def corp_get_ledger(
        capability_id: str,
    ) -> Dict[str, Any]:
        store = _require_corp_store()
        return {
            "capability_id": capability_id,
            "entries": store.get_ledger(capability_id),
        }

    @app.get(
        "/admin/corp/capability/{capability_id}/consumed",
        tags=["admin"],
    )
    async def corp_get_consumed(
        capability_id: str,
    ) -> Dict[str, Any]:
        store = _require_corp_store()
        return {
            "capability_id": capability_id,
            "consumed": store.get_consumed(capability_id),
        }

    # ── Sprint 305 — TEE-only execution policy ────────────
    # Vision §7 Enterprise Confidentiality Mode layer 3.
    # Declarative attestation-quality gate. Evaluation is
    # pure; live dispatcher wiring is sprint 305a.

    class _TEEPolicyEvaluateRequest(BaseModel):
        attestation_b64: Optional[str] = None
        policy: Dict[str, Any]

    @app.post(
        "/admin/tee-policy/evaluate", tags=["admin"],
    )
    async def tee_policy_evaluate(
        body: _TEEPolicyEvaluateRequest,
    ) -> Dict[str, Any]:
        from prsm.enterprise.tee_policy import (
            TEEPolicy, evaluate_attestation_blob,
        )
        import base64 as _b64
        try:
            policy = TEEPolicy.from_dict(body.policy or {})
        except ValueError as e:
            raise HTTPException(
                status_code=422,
                detail=f"invalid policy: {e}",
            )
        blob: Optional[bytes] = None
        if body.attestation_b64:
            try:
                blob = _b64.b64decode(
                    body.attestation_b64, validate=True,
                )
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"attestation_b64 not valid base64: "
                        f"{e}"
                    ),
                )
        result = evaluate_attestation_blob(blob, policy)
        return result.to_dict()

    @app.get(
        "/admin/tee-policy/node-status", tags=["admin"],
    )
    async def tee_policy_node_status() -> Dict[str, Any]:
        """Snapshot of THIS node's own attestation tier.
        Enterprises use this to pre-screen which nodes are
        eligible to participate in a given workload BEFORE
        dispatching the job."""
        from prsm.compute.inference.attestation_backends import (
            AttestationVerificationResult, verify_attestation,
        )
        from prsm.enterprise.tee_policy import (
            effective_tier_from_result,
        )
        blob = getattr(
            node, "_tee_node_attestation_blob", None,
        )
        if not blob:
            result = AttestationVerificationResult(
                vendor="unknown",
                error="node has no attestation blob configured",
            )
        else:
            try:
                result = verify_attestation(bytes(blob))
            except Exception as exc:  # noqa: BLE001
                result = AttestationVerificationResult(
                    vendor="unknown",
                    error=(
                        f"verify_attestation raised: {exc}"
                    ),
                )
        tier = effective_tier_from_result(result)
        return {
            "effective_tier": tier.value,
            "vendor": result.vendor,
            "vendor_verified": result.vendor_verified,
            "diagnostic": result.error or (
                "node attestation parsed; effective "
                f"tier={tier.value}"
            ),
        }

    # ── Sprint 303 — UUPS upgrade orchestrator ────────────
    # Vision §14 mitigation item 7: UUPS upgrade pattern
    # with pre-committed rollback escape. Composer-only —
    # Foundation Safe is the execution gate.

    class _UpgradeProposeRequest(BaseModel):
        target_proxy: str
        new_implementation: str
        previous_implementation: str
        severity: str
        rationale: str
        init_calldata_hex: str = "0x"
        reviewer_assignments: List[str] = []

    class _UpgradeUpdateRequest(BaseModel):
        new_status: str
        safe_tx_hash: Optional[str] = None

    def _require_upgrade_orchestrator():
        o = getattr(node, "_upgrade_orchestrator", None)
        if o is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Upgrade orchestrator not initialized."
                ),
            )
        return o

    @app.post("/admin/upgrade/propose", tags=["admin"])
    async def upgrade_propose(
        body: _UpgradeProposeRequest,
    ) -> Dict[str, Any]:
        from prsm.economy.web3.upgrade_orchestrator import (
            UpgradeSeverity,
        )
        o = _require_upgrade_orchestrator()
        try:
            severity = UpgradeSeverity(
                (body.severity or "").lower(),
            )
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"invalid severity {body.severity!r}"
                ),
            )
        try:
            record = o.propose(
                target_proxy=body.target_proxy,
                new_implementation=body.new_implementation,
                previous_implementation=(
                    body.previous_implementation
                ),
                severity=severity,
                rationale=body.rationale,
                init_calldata_hex=body.init_calldata_hex,
                reviewer_assignments=(
                    body.reviewer_assignments
                ),
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return record.to_dict()

    @app.get("/admin/upgrade", tags=["admin"])
    async def upgrade_list(
        status: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> Dict[str, Any]:
        from prsm.economy.web3.upgrade_orchestrator import (
            UpgradeSeverity, UpgradeStatus,
        )
        o = _require_upgrade_orchestrator()
        status_obj = None
        sev_obj = None
        if status:
            try:
                status_obj = UpgradeStatus(status.lower())
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=f"invalid status {status!r}",
                )
        if severity:
            try:
                sev_obj = UpgradeSeverity(severity.lower())
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=f"invalid severity {severity!r}",
                )
        records = o.list(
            status=status_obj, severity=sev_obj,
        )
        return {
            "records": [r.to_dict() for r in records],
            "count": len(records),
        }

    @app.get(
        "/admin/upgrade/{proposal_id}", tags=["admin"],
    )
    async def upgrade_get(
        proposal_id: str,
    ) -> Dict[str, Any]:
        o = _require_upgrade_orchestrator()
        record = o.get(proposal_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"proposal {proposal_id!r} not found"
                ),
            )
        return record.to_dict()

    @app.post(
        "/admin/upgrade/{proposal_id}/update",
        tags=["admin"],
    )
    async def upgrade_update(
        proposal_id: str,
        body: _UpgradeUpdateRequest,
    ) -> Dict[str, Any]:
        from prsm.economy.web3.upgrade_orchestrator import (
            UpgradeStatus,
        )
        o = _require_upgrade_orchestrator()
        if o.get(proposal_id) is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"proposal {proposal_id!r} not found"
                ),
            )
        try:
            new_status = UpgradeStatus(
                (body.new_status or "").lower(),
            )
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"invalid new_status {body.new_status!r}"
                ),
            )
        try:
            record = o.update_status(
                proposal_id, new_status,
                safe_tx_hash=body.safe_tx_hash,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return record.to_dict()

    @app.post(
        "/admin/upgrade/{proposal_id}/compose-upgrade",
        tags=["admin"],
    )
    async def upgrade_compose_upgrade(
        proposal_id: str,
    ) -> Dict[str, Any]:
        from prsm.economy.web3.upgrade_orchestrator import (
            compose_upgrade_tx,
        )
        o = _require_upgrade_orchestrator()
        if o.get(proposal_id) is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"proposal {proposal_id!r} not found"
                ),
            )
        try:
            tx = compose_upgrade_tx(
                orchestrator=o,
                proposal_id=proposal_id,
                chain_id=getattr(
                    node, "_upgrade_chain_id", None,
                ),
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return tx

    @app.post(
        "/admin/upgrade/{proposal_id}/compose-rollback",
        tags=["admin"],
    )
    async def upgrade_compose_rollback(
        proposal_id: str,
    ) -> Dict[str, Any]:
        from prsm.economy.web3.upgrade_orchestrator import (
            compose_rollback_tx,
        )
        o = _require_upgrade_orchestrator()
        if o.get(proposal_id) is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"proposal {proposal_id!r} not found"
                ),
            )
        try:
            tx = compose_rollback_tx(
                orchestrator=o,
                proposal_id=proposal_id,
                chain_id=getattr(
                    node, "_upgrade_chain_id", None,
                ),
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return tx

    # ── Sprint 302 — formal-invariant harness ─────────────
    # Vision §14 mitigation item 4: pinned formal-spec
    # registry + runtime probe. /invariants is PUBLIC (the
    # spec is published BEFORE any incident — same posture
    # as the §14 item 5 playbook). /check requires both a
    # wired backend AND a known contract address.

    @app.get(
        "/admin/formal-verification/invariants",
        tags=["admin"],
    )
    async def formal_invariant_list(
        contract: Optional[str] = None,
    ) -> Dict[str, Any]:
        from prsm.economy.web3.formal_invariants import (
            INVARIANT_REGISTRY,
        )
        out: List[Dict[str, Any]] = []
        for name, invs in INVARIANT_REGISTRY.items():
            if contract and name != contract:
                continue
            for inv in invs:
                out.append(inv.to_dict())
        return {"invariants": out}

    @app.get(
        "/admin/formal-verification/check", tags=["admin"],
    )
    async def formal_invariant_check(
        contract: str,
    ) -> Dict[str, Any]:
        from prsm.economy.web3.formal_invariants import (
            INVARIANT_REGISTRY,
        )
        checker = getattr(
            node, "_formal_invariant_checker", None,
        )
        if checker is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Formal-invariant checker not wired "
                    "(no RPC backend available)."
                ),
            )
        if contract not in INVARIANT_REGISTRY:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"unknown contract {contract!r}; known: "
                    f"{sorted(INVARIANT_REGISTRY)}"
                ),
            )
        addresses = getattr(
            node, "_formal_invariant_addresses", {},
        ) or {}
        addr = addresses.get(contract)
        if not addr:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"contract address for {contract!r} not "
                    f"configured (set PRSM_NETWORK or the "
                    f"corresponding per-contract env var)."
                ),
            )
        results = checker.check_contract(
            contract, contract_address=addr,
        )
        summary = {"pass": 0, "fail": 0, "skipped": 0}
        for r in results:
            summary[r.status.value] = (
                summary.get(r.status.value, 0) + 1
            )
        return {
            "contract": contract,
            "address": addr,
            "summary": summary,
            "results": [r.to_dict() for r in results],
        }

    @app.get(
        "/admin/formal-verification/check/{invariant_id}",
        tags=["admin"],
    )
    async def formal_invariant_check_one(
        invariant_id: str,
    ) -> Dict[str, Any]:
        from prsm.economy.web3.formal_invariants import (
            INVARIANT_REGISTRY,
        )
        checker = getattr(
            node, "_formal_invariant_checker", None,
        )
        if checker is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Formal-invariant checker not wired."
                ),
            )
        found = None
        contract_for_inv = None
        for name, invs in INVARIANT_REGISTRY.items():
            for inv in invs:
                if inv.id == invariant_id:
                    found = inv
                    contract_for_inv = name
                    break
            if found is not None:
                break
        if found is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"invariant {invariant_id!r} not found"
                ),
            )
        addresses = getattr(
            node, "_formal_invariant_addresses", {},
        ) or {}
        addr = addresses.get(contract_for_inv)
        if not addr:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"contract address for "
                    f"{contract_for_inv!r} not configured."
                ),
            )
        result = checker.check_one(found, addr)
        return result.to_dict()

    # ── Sprint 364 — symbolic verification surface ────────
    # Halmos symbolic-execution results, complementing the
    # runtime probe above. /symbolic is PUBLIC (catalog list,
    # same posture as /invariants). /symbolic/check/{spec}
    # requires a wired HalmosRunner (sprint 360) — fail-soft
    # 503 if halmos/forge isn't installed.

    @app.get(
        "/admin/formal-verification/symbolic",
        tags=["admin"],
    )
    async def formal_verification_symbolic_list(
    ) -> Dict[str, Any]:
        from prsm.economy.web3.halmos_runner import (
            SYMBOLIC_PROOF_CATALOG,
        )
        specs: List[Dict[str, Any]] = []
        for name, entry in SYMBOLIC_PROOF_CATALOG.items():
            specs.append({
                "name": name,
                "mirrors_runtime_contract": entry.get(
                    "mirrors_runtime_contract",
                ),
                "runtime_invariants": list(
                    entry.get("runtime_invariants", []),
                ),
                "description": entry.get("description", ""),
            })
        return {"specs": specs}

    @app.get(
        "/admin/formal-verification/symbolic/check/{spec}",
        tags=["admin"],
    )
    async def formal_verification_symbolic_check(
        spec: str,
    ) -> Dict[str, Any]:
        from prsm.economy.web3.halmos_runner import (
            SYMBOLIC_PROOF_CATALOG,
        )
        if spec not in SYMBOLIC_PROOF_CATALOG:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"unknown symbolic spec {spec!r}; "
                    f"known: {sorted(SYMBOLIC_PROOF_CATALOG)}"
                ),
            )
        runner = getattr(node, "_halmos_runner", None)
        if runner is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Halmos symbolic runner not wired. "
                    "Install halmos (pip install halmos) + "
                    "forge (foundryup), then restart the "
                    "node."
                ),
            )
        if not runner.is_available():
            missing = runner.missing_tools()
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Halmos runner unavailable — missing "
                    f"tools: {', '.join(missing)}."
                ),
            )
        suite = runner.run(spec)
        body = suite.to_dict()
        # Attach catalog cross-reference so operators see
        # which runtime invariants this proof mirrors —
        # same join as the runtime-probe endpoint exposes.
        entry = SYMBOLIC_PROOF_CATALOG[spec]
        body["runtime_invariants"] = list(
            entry.get("runtime_invariants", []),
        )
        body["mirrors_runtime_contract"] = entry.get(
            "mirrors_runtime_contract",
        )
        return body

    # ── Sprint 301 — incident response playbook ───────────
    # Vision §14 mitigation item 5: public exploit-response
    # playbook + code hooks. /playbook is intentionally
    # accessible (public per §14 transparency promise). All
    # other paths are admin (mutations + per-incident reads).

    class _IncidentOpenRequest(BaseModel):
        severity: str
        summary: str
        affected_contracts: List[str] = []
        related_disclosure_id: Optional[str] = None
        actor: str = ""

    class _IncidentAdvanceRequest(BaseModel):
        new_phase: str
        note: str = ""
        actor: str = ""

    class _IncidentEventRequest(BaseModel):
        note: str
        actor: str = ""

    def _require_incident_response():
        ir = getattr(node, "_incident_response", None)
        if ir is None:
            raise HTTPException(
                status_code=503,
                detail="Incident response not initialized.",
            )
        return ir

    @app.post("/admin/incident/open", tags=["admin"])
    async def incident_open(
        body: _IncidentOpenRequest,
    ) -> Dict[str, Any]:
        from prsm.economy.web3.incident_response import (
            IncidentSeverity,
        )
        ir = _require_incident_response()
        sev_raw = (body.severity or "").strip().lower()
        try:
            severity = IncidentSeverity(sev_raw)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"severity must be one of "
                    f"{[s.value for s in IncidentSeverity]}"
                ),
            )
        try:
            record = ir.open(
                severity=severity,
                summary=body.summary,
                affected_contracts=body.affected_contracts,
                related_disclosure_id=(
                    body.related_disclosure_id
                ),
                actor=body.actor,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return record.to_dict()

    @app.post("/admin/escrow/recover-orphans", tags=["admin"])
    async def recover_orphan_escrows(
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """Sprint 489 (F27 recovery) — scan dag_ledger for
        escrow-* wallets with positive balance whose
        corresponding `_escrows` in-memory record has been
        lost (e.g., across daemon restart). For each, look
        up the original `Escrow for job X` transfer's
        `from_wallet` and refund the balance back.

        Without this endpoint, operators whose daemon crashed
        mid-test or restarted before all escrows resolved
        would have FTNS permanently locked in escrow-* wallets
        with no recovery path short of a database edit.

        Args:
            dry_run: If True (default), report what would be
                refunded without mutating state. Operators
                MUST review the dry-run output before passing
                dry_run=false.

        Returns: { dry_run, scanned, recoverable, refunded,
                   total_ftns_recovered, errors }
        """
        ledger = getattr(node, "ledger", None)
        if ledger is None or not hasattr(ledger, "_db"):
            raise HTTPException(
                status_code=503,
                detail="DAG ledger not initialized.",
            )

        scanned = 0
        recoverable: List[Dict[str, Any]] = []
        refunded = 0
        total_ftns = 0.0
        errors: List[str] = []

        try:
            cursor = await ledger._db.execute(
                "SELECT wallet_id, balance FROM wallet_balances "
                "WHERE wallet_id LIKE 'escrow-%' AND balance > 0"
            )
            rows = await cursor.fetchall()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=503,
                detail=f"orphan scan failed: {exc}",
            )

        from prsm.node.dag_ledger import TransactionType

        for escrow_wallet, balance in rows:
            scanned += 1
            try:
                cur2 = await ledger._db.execute(
                    "SELECT from_wallet FROM dag_transactions "
                    "WHERE to_wallet = ? AND description LIKE "
                    "'Escrow for%' ORDER BY timestamp ASC LIMIT 1",
                    (escrow_wallet,),
                )
                rr = await cur2.fetchone()
                if not rr or not rr[0]:
                    errors.append(
                        f"{escrow_wallet}: no requester found"
                    )
                    continue
                requester = rr[0]
                recoverable.append({
                    "escrow_wallet": escrow_wallet,
                    "balance": float(balance),
                    "requester": requester,
                })
                if not dry_run:
                    await ledger.submit_transaction(
                        tx_type=TransactionType.TRANSFER,
                        amount=float(balance),
                        from_wallet=escrow_wallet,
                        to_wallet=requester,
                        description=(
                            "Sprint 489 F27 admin recovery: "
                            "orphaned escrow refund"
                        ),
                    )
                    refunded += 1
                    total_ftns += float(balance)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{escrow_wallet}: {exc}")

        return {
            "dry_run": dry_run,
            "scanned": scanned,
            "recoverable": recoverable if dry_run else [],
            "refunded": refunded,
            "total_ftns_recovered": total_ftns,
            "errors": errors,
        }

    @app.get("/admin/incident/playbook", tags=["admin"])
    async def incident_playbook() -> Dict[str, Any]:
        """Public playbook surface — Vision §14 promise that
        the response plan is published BEFORE any incident.
        Returns decision tree + comms templates."""
        from prsm.economy.web3.incident_response import (
            COMMS_TEMPLATES, DECISION_TREE,
            IncidentPhase, IncidentSeverity,
        )
        decision_tree = []
        for sev in IncidentSeverity:
            for phase in IncidentPhase:
                recs = DECISION_TREE.get((sev, phase))
                if recs:
                    decision_tree.append({
                        "severity": sev.value,
                        "phase": phase.value,
                        "recommendations": list(recs),
                    })
        comms = []
        for sev in IncidentSeverity:
            for phase in IncidentPhase:
                text = COMMS_TEMPLATES.get((sev, phase))
                if text:
                    comms.append({
                        "severity": sev.value,
                        "phase": phase.value,
                        "text": text,
                    })
        return {
            "decision_tree": decision_tree,
            "comms_templates": comms,
        }

    @app.get("/admin/incident", tags=["admin"])
    async def incident_list(
        severity: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> Dict[str, Any]:
        from prsm.economy.web3.incident_response import (
            IncidentPhase, IncidentSeverity,
        )
        ir = _require_incident_response()
        sev_obj = None
        phase_obj = None
        if severity:
            try:
                sev_obj = IncidentSeverity(severity.lower())
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=f"invalid severity {severity!r}",
                )
        if phase:
            try:
                phase_obj = IncidentPhase(phase.lower())
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=f"invalid phase {phase!r}",
                )
        records = ir.list(
            severity=sev_obj, phase=phase_obj,
        )
        return {
            "records": [r.to_dict() for r in records],
            "count": len(records),
        }

    @app.get(
        "/admin/incident/{incident_id}", tags=["admin"],
    )
    async def incident_get(
        incident_id: str,
    ) -> Dict[str, Any]:
        ir = _require_incident_response()
        record = ir.get(incident_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"incident {incident_id!r} not found",
            )
        return record.to_dict()

    @app.post(
        "/admin/incident/{incident_id}/advance",
        tags=["admin"],
    )
    async def incident_advance(
        incident_id: str,
        body: _IncidentAdvanceRequest,
    ) -> Dict[str, Any]:
        from prsm.economy.web3.incident_response import (
            IncidentPhase,
        )
        ir = _require_incident_response()
        if ir.get(incident_id) is None:
            raise HTTPException(
                status_code=404,
                detail=f"incident {incident_id!r} not found",
            )
        try:
            new_phase = IncidentPhase(
                (body.new_phase or "").lower(),
            )
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"invalid new_phase {body.new_phase!r}"
                ),
            )
        try:
            record = ir.advance_phase(
                incident_id, new_phase,
                note=body.note, actor=body.actor,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return record.to_dict()

    @app.post(
        "/admin/incident/{incident_id}/event",
        tags=["admin"],
    )
    async def incident_event(
        incident_id: str,
        body: _IncidentEventRequest,
    ) -> Dict[str, Any]:
        ir = _require_incident_response()
        if ir.get(incident_id) is None:
            raise HTTPException(
                status_code=404,
                detail=f"incident {incident_id!r} not found",
            )
        try:
            record = ir.record_event(
                incident_id, note=body.note, actor=body.actor,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return record.to_dict()

    @app.get(
        "/admin/incident/{incident_id}/recommendations",
        tags=["admin"],
    )
    async def incident_recommendations(
        incident_id: str,
    ) -> Dict[str, Any]:
        from prsm.economy.web3.incident_response import (
            get_recommendations,
        )
        ir = _require_incident_response()
        record = ir.get(incident_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"incident {incident_id!r} not found",
            )
        return {
            "incident_id": record.incident_id,
            "severity": record.severity.value,
            "current_phase": record.current_phase.value,
            "recommendations": get_recommendations(
                record.severity, record.current_phase,
            ),
        }

    @app.get(
        "/admin/incident/{incident_id}/comms-template",
        tags=["admin"],
    )
    async def incident_comms_template(
        incident_id: str,
    ) -> Dict[str, Any]:
        from prsm.economy.web3.incident_response import (
            get_comms_template,
        )
        ir = _require_incident_response()
        record = ir.get(incident_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"incident {incident_id!r} not found",
            )
        return {
            "incident_id": record.incident_id,
            "severity": record.severity.value,
            "current_phase": record.current_phase.value,
            "text": get_comms_template(
                record.severity, record.current_phase,
                summary=record.summary,
            ),
        }

    # ── Sprint 292 — privacy-claim verification ───────────
    # Public API for the §7 promise: lets callers verify
    # signature + DP-noise + hardware-attestation quality
    # of an InferenceReceipt without trusting the executor's
    # claims. Defaults are permissive (returns ok=True with
    # diagnostic flags); strict callers pass
    # require_hardware_attestation=true to gate on real TEE
    # (currently every local executor uses DEV-ONLY software
    # fallback, so this gate currently fails — by design,
    # until hardware-attestation backends ship).

    @app.post(
        "/compute/receipt/verify", tags=["compute"],
    )
    async def verify_receipt_privacy(
        body: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        import base64
        import dataclasses as _dc
        from prsm.compute.inference.models import (
            ContentTier, InferenceReceipt,
        )
        from prsm.compute.inference.privacy_verification import (
            verify_receipt_privacy_claim,
        )
        from prsm.compute.tee.models import (
            PrivacyLevel as _PL, TEEType as _TT,
        )

        receipt_payload = body.get("receipt")
        if not isinstance(receipt_payload, dict):
            raise HTTPException(
                status_code=422,
                detail=(
                    "missing required field: receipt "
                    "(must be a JSON object)"
                ),
            )
        public_key_b64 = body.get("public_key_b64")
        if not public_key_b64:
            raise HTTPException(
                status_code=422,
                detail=(
                    "missing required field: public_key_b64 "
                    "(base64 Ed25519 public key of the "
                    "settler that signed the receipt)"
                ),
            )
        require_hw = bool(
            body.get("require_hardware_attestation", False),
        )
        require_dp = bool(
            body.get("require_dp_noise", False),
        )
        # Sprint 900 — expose the verifier's activation-DP-trace +
        # topology-rotation integrity gates over HTTP (the library
        # supported them; the endpoint never surfaced them).
        require_dp_trace = bool(
            body.get("require_activation_dp_trace", False),
        )
        require_topology = bool(
            body.get("require_topology_rotation", False),
        )

        # Reconstruct InferenceReceipt from JSON payload.
        # Fields containing bytes are base64-encoded on the
        # wire; decode upfront. Surface 422 on malformed
        # input rather than 500.
        try:
            tee_attestation = base64.b64decode(
                receipt_payload.get(
                    "tee_attestation_b64", "",
                ) or b""
            )
            output_hash = base64.b64decode(
                receipt_payload.get("output_hash_b64", "")
                or b""
            )
            settler_signature = base64.b64decode(
                receipt_payload.get(
                    "settler_signature_b64", "",
                ) or b""
            )
            # Sprint 900 — reconstruct the §7 capstone fields. The
            # default-wrapped live executor (make_rpc_chain_executor
            # wraps TopologyAware + ActivationDP by default) populates
            # activation_noise_trace + topology_assignment on EVERY
            # non-NONE-tier receipt, and signing_payload() folds their
            # hashes into the signed bytes. Dropping them here made the
            # reconstructed receipt's signing_payload diverge from what
            # was signed → signature_valid=False on honest private-tier
            # receipts, breaking the §7 independent-verification promise
            # for exactly the receipts it's meant to cover.
            _ant = receipt_payload.get("activation_noise_trace")
            if isinstance(_ant, dict):
                from prsm.compute.inference.activation_dp import (
                    ActivationNoiseTrace,
                )
                _ant = ActivationNoiseTrace.from_dict(_ant)
            elif _ant is not None:
                _ant = None  # malformed → treat as absent
            _topo = receipt_payload.get("topology_assignment")
            if isinstance(_topo, dict):
                from prsm.compute.inference.topology_rotation import (
                    TopologyAssignment,
                )
                _topo = TopologyAssignment.from_dict(_topo)
            elif _topo is not None:
                _topo = None
            # sp1098 (Domain-03 review F5) — streamed_output + partial_completion ALSO
            # fold into signing_payload() (Phase-3.x.8 + sprint-777), so dropping them on
            # reconstruction made streamed / partial-completion receipts falsely fail
            # verification — the same defect class as the sprint-900 fix above, left open
            # for these two fields. Restore them so honest streamed/partial receipts verify.
            _streamed = bool(receipt_payload.get("streamed_output", False))
            _partial = receipt_payload.get("partial_completion")
            if isinstance(_partial, dict):
                from prsm.compute.inference.partial_completion import (
                    PartialCompletionInfo,
                )
                _partial = PartialCompletionInfo.from_dict(_partial)
            elif _partial is not None:
                _partial = None
            # sp1099 (Domain-03 review F1) — prompt_hash folds into
            # signing_payload(); restore it on reconstruction or a prompt-bound
            # receipt falsely fails verification (same defect class as the F5
            # streamed/partial restoration above).
            _prompt_hash = receipt_payload.get("prompt_hash")
            if isinstance(_prompt_hash, str):
                _prompt_hash = bytes.fromhex(_prompt_hash)
            elif _prompt_hash is not None:
                _prompt_hash = None
            # sp1110 (Domain-03 F1/F2) — the per-stage signed activation chain folds into
            # signing_payload(); restore it on reconstruction or a chain-bearing receipt
            # falsely fails signature verification (same defect class as the sp1098/1099
            # streamed/partial/prompt_hash restorations).
            _stage_chain = receipt_payload.get("stage_activation_chain")
            if isinstance(_stage_chain, dict):
                from prsm.compute.inference.stage_activation_proof import (
                    StageActivationChain,
                )
                _stage_chain = StageActivationChain.from_dict(_stage_chain)
            elif _stage_chain is not None:
                _stage_chain = None
            receipt = InferenceReceipt(
                job_id=receipt_payload["job_id"],
                request_id=receipt_payload["request_id"],
                model_id=receipt_payload["model_id"],
                content_tier=_PL(
                    receipt_payload["content_tier"]
                ) if False else ContentTier(
                    receipt_payload["content_tier"]
                ),
                privacy_tier=_PL(
                    receipt_payload["privacy_tier"]
                ),
                epsilon_spent=float(
                    receipt_payload.get("epsilon_spent", 0)
                ),
                tee_type=_TT(
                    receipt_payload.get("tee_type", "software")
                ),
                tee_attestation=tee_attestation,
                output_hash=output_hash,
                duration_seconds=float(
                    receipt_payload.get("duration_seconds", 0)
                ),
                cost_ftns=str(
                    receipt_payload.get("cost_ftns", "0")
                ),
                settler_signature=settler_signature,
                settler_node_id=receipt_payload.get(
                    "settler_node_id", "",
                ),
                activation_noise_trace=_ant,
                topology_assignment=_topo,
                streamed_output=_streamed,
                partial_completion=_partial,
                prompt_hash=_prompt_hash,
                stage_activation_chain=_stage_chain,
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"malformed receipt: {exc}"
                ),
            )

        result = verify_receipt_privacy_claim(
            receipt,
            require_hardware_attestation=require_hw,
            require_dp_noise=require_dp,
            require_activation_dp_trace=require_dp_trace,
            require_topology_rotation=require_topology,
            public_key_b64=public_key_b64,
        )
        result_dict = result.to_dict()
        # sp1111 (Domain-03 F1/F2, brick 5) — surface the per-stage signed activation
        # chain verification. The settler signature (checked above) already commits to
        # the chain; this ADDS the independent per-stage view: internal links + derived
        # topology (always), per-stage signatures (when the caller supplies
        # stage_public_keys, mirroring the settler public_key_b64 it already supplies),
        # and a topology cross-check (F2) against the receipt's topology_assignment.
        if receipt.stage_activation_chain is not None:
            from prsm.compute.inference.stage_activation_proof import (
                summarize_stage_activation_chain,
            )
            result_dict["stage_activation_chain"] = summarize_stage_activation_chain(
                receipt.stage_activation_chain,
                stage_public_keys=body.get("stage_public_keys"),
                topology_assignment=receipt.topology_assignment,
                expected_request_id=receipt.request_id,
            )
        else:
            result_dict["stage_activation_chain"] = {"present": False}
        return result_dict

    @app.post("/settlement/challenge/verify", tags=["settlement"])
    async def settlement_challenge_verify(
        body: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        """Sprint 1116 (challenge/dispute mechanism, brick 2) — off-chain dispute verdict
        for a §7 inference receipt. A would-be challenger / auditor POSTs a receipt + the
        settler's public key (+ optionally the per-stage node keys) and gets back a
        structured verdict: is there PROVABLE fraud, on what grounds (invalid settler
        signature / broken activation chain / topology lie / spliced chain), and which
        grounds are on-chain-actionable (map to a BatchSettlementRegistry ReasonCode).
        Read-only + pure (no broadcast). The receipt is the canonical
        InferenceReceipt.to_dict() shape.
        """
        from prsm.compute.inference.models import InferenceReceipt
        from prsm.settlement.challenge_verifier import (
            verify_inference_receipt_for_challenge,
        )

        receipt_payload = body.get("receipt")
        if not isinstance(receipt_payload, dict):
            raise HTTPException(
                status_code=422,
                detail="missing required field: receipt (a JSON object)",
            )
        settler_key = body.get("settler_public_key_b64")
        if not settler_key:
            raise HTTPException(
                status_code=422,
                detail=(
                    "missing required field: settler_public_key_b64 (base64 Ed25519 "
                    "public key of the settler that signed the receipt)"
                ),
            )
        stage_keys = body.get("stage_public_keys")
        if stage_keys is not None and not isinstance(stage_keys, dict):
            raise HTTPException(
                status_code=422,
                detail="stage_public_keys must be a {node_id: pubkey_b64} object",
            )
        try:
            receipt = InferenceReceipt.from_dict(receipt_payload)
        except (KeyError, ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=422, detail=f"malformed receipt: {exc}",
            )
        report = verify_inference_receipt_for_challenge(
            receipt,
            settler_public_key_b64=settler_key,
            stage_public_keys=stage_keys,
        )
        out = report.to_dict()
        out["on_chain_actionable"] = [
            f.to_dict() for f in report.on_chain_actionable()
        ]
        return out

    # ── Sprint 287 — creator reputation operator surface ─
    # Per Vision §14 "Data quality and Sybil resistance"
    # mitigation item (1). Read paths surface aggregates
    # (no purchaser_counts — privacy + payload size); write
    # path is operator-internal (called by ContentStore
    # retrieve paths when a piece of content is accessed).

    def _creator_row(tracker, creator_id: str) -> Dict[str, Any]:
        from prsm.marketplace.creator_stake_client import (
            apply_stake_gate,
        )
        e = tracker.get_entry(creator_id)
        # Sprint 288 — tier always surfaces. Cold-start /
        # unknown → TIER_NEW.
        # Sprint 290 — apply stake-eligibility gate. Demotes
        # HIGH → MEDIUM when stake client wired AND creator
        # hasn't bonded the minimum stake. Pre-sprint-290
        # behavior preserved when stake_client is None.
        raw_tier = tracker.tier_for(creator_id)
        stake_client = getattr(
            node, "_creator_stake_client", None,
        )
        tier = apply_stake_gate(
            raw_tier, creator_id, stake_client,
        )
        if e is None:
            return {
                "creator_id": creator_id,
                "known": False,
                "score": tracker.score_for(creator_id),
                "tier": tier,
                "total_accesses": 0,
                "distinct_purchasers": 0,
                "repeat_purchaser_count": 0,
                "first_seen_unix": 0,
                "last_seen_unix": 0,
            }
        d = e.to_dict()
        d["known"] = True
        d["score"] = tracker.score_for(creator_id)
        d["tier"] = tier
        return d

    @app.get(
        "/marketplace/creator-reputation",
        tags=["marketplace"],
    )
    async def list_creator_reputation(
        limit: int = 100,
    ) -> Dict[str, Any]:
        if limit <= 0 or limit > 10000:
            raise HTTPException(
                status_code=422,
                detail=f"limit must be in [1, 10000], got {limit}",
            )
        tracker = getattr(
            node, "_creator_reputation_tracker", None,
        )
        if tracker is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Creator reputation tracker not "
                    "initialized."
                ),
            )
        creator_ids = tracker.known_creators()
        rows = [
            _creator_row(tracker, cid) for cid in creator_ids
        ]
        rows.sort(
            key=lambda r: (r["score"], r["total_accesses"]),
            reverse=True,
        )
        return {
            "creators": rows[:limit],
            "count": len(creator_ids),
            "limit": limit,
        }

    @app.get(
        "/marketplace/creator-reputation/{creator_id}",
        tags=["marketplace"],
    )
    async def get_creator_reputation(
        creator_id: str,
    ) -> Dict[str, Any]:
        tracker = getattr(
            node, "_creator_reputation_tracker", None,
        )
        if tracker is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Creator reputation tracker not "
                    "initialized."
                ),
            )
        return _creator_row(tracker, creator_id)

    class _CreatorAccessRequest(BaseModel):
        creator_id: str
        purchaser_id: str
        content_id: str

    @app.post(
        "/marketplace/creator-reputation/access",
        tags=["marketplace"],
    )
    async def record_creator_access(
        body: _CreatorAccessRequest,
    ) -> Dict[str, Any]:
        tracker = getattr(
            node, "_creator_reputation_tracker", None,
        )
        if tracker is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Creator reputation tracker not "
                    "initialized."
                ),
            )
        try:
            tracker.record_access(
                creator_id=body.creator_id,
                purchaser_id=body.purchaser_id,
                content_id=body.content_id,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return {
            "creator_id": body.creator_id,
            "total_accesses": tracker.access_count(
                body.creator_id,
            ),
            "score": tracker.score_for(body.creator_id),
        }

    # ── Sprint 291 — fingerprint registry inspection ──────
    @app.get(
        "/marketplace/fingerprint/{content_hash}",
        tags=["marketplace"],
    )
    async def get_fingerprint(
        content_hash: str,
    ) -> Dict[str, Any]:
        reg = getattr(
            node, "_content_fingerprint_registry", None,
        )
        if reg is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Content fingerprint registry not "
                    "initialized."
                ),
            )
        entry = reg.get_entry(content_hash)
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"no fingerprint for {content_hash!r}"
                ),
            )
        return entry.to_dict()

    @app.get(
        "/marketplace/fingerprint",
        tags=["marketplace"],
    )
    async def list_fingerprints(
        limit: int = 100,
    ) -> Dict[str, Any]:
        if limit <= 0 or limit > 10000:
            raise HTTPException(
                status_code=422,
                detail=f"limit must be in [1, 10000], got {limit}",
            )
        reg = getattr(
            node, "_content_fingerprint_registry", None,
        )
        if reg is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Content fingerprint registry not "
                    "initialized."
                ),
            )
        entries = reg.recent(limit=limit)
        return {
            "fingerprints": [e.to_dict() for e in entries],
            "count": reg.count(),
            "limit": limit,
        }

    # ── Sprint 290 — creator stake operator surface ────────
    # Operator-side surface for the CreatorStakeClient
    # (Vision §14 item 2). stake / slash are operator-
    # internal writes (real on-chain when commissioned, in-
    # memory otherwise per the PENDING_COMMISSION pattern).

    class _CreatorStakeRequest(BaseModel):
        creator_id: str
        amount_wei: int

    class _CreatorSlashRequest(BaseModel):
        creator_id: str
        amount_wei: int
        reason: str

    @app.get(
        "/marketplace/creator-stake/{creator_id}",
        tags=["marketplace"],
    )
    async def get_creator_stake(
        creator_id: str,
    ) -> Dict[str, Any]:
        from prsm.marketplace.creator_stake_client import (
            MIN_HIGH_TIER_STAKE_WEI,
        )
        stake_client = getattr(
            node, "_creator_stake_client", None,
        )
        if stake_client is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Creator stake client not initialized."
                ),
            )
        return {
            "creator_id": creator_id,
            "balance_wei": stake_client.stake_balance(
                creator_id,
            ),
            "high_tier_eligible": (
                stake_client.is_high_tier_eligible(
                    creator_id,
                )
            ),
            "min_high_tier_stake_wei": (
                MIN_HIGH_TIER_STAKE_WEI
            ),
            "commissioned": stake_client.is_commissioned(),
        }

    @app.post(
        "/marketplace/creator-stake/stake",
        tags=["marketplace"],
    )
    async def post_creator_stake(
        body: _CreatorStakeRequest,
    ) -> Dict[str, Any]:
        stake_client = getattr(
            node, "_creator_stake_client", None,
        )
        if stake_client is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Creator stake client not initialized."
                ),
            )
        try:
            stake_client.stake(
                creator_id=body.creator_id,
                amount_wei=body.amount_wei,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return {
            "creator_id": body.creator_id,
            "balance_wei": stake_client.stake_balance(
                body.creator_id,
            ),
            "high_tier_eligible": (
                stake_client.is_high_tier_eligible(
                    body.creator_id,
                )
            ),
        }

    @app.post(
        "/marketplace/creator-stake/slash",
        tags=["marketplace"],
    )
    async def post_creator_slash(
        body: _CreatorSlashRequest,
    ) -> Dict[str, Any]:
        stake_client = getattr(
            node, "_creator_stake_client", None,
        )
        if stake_client is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Creator stake client not initialized."
                ),
            )
        try:
            stake_client.slash(
                creator_id=body.creator_id,
                amount_wei=body.amount_wei,
                reason=body.reason,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return {
            "creator_id": body.creator_id,
            "balance_wei": stake_client.stake_balance(
                body.creator_id,
            ),
            "slashed_wei": body.amount_wei,
            "reason": body.reason,
        }

    # ── Sprint 275 — marketplace reputation operator surface ─
    # ReputationTracker has informed the marketplace candidate
    # pool since Phase 3 Task 6, but operators had no surface
    # to inspect it. These read-only endpoints close that gap.
    # Per `prsm/marketplace/reputation.py` contract: NEUTRAL_SCORE
    # (0.5) is returned for unknown providers AND for known
    # providers with < MIN_SAMPLES_FOR_SCORE total observations.

    def _reputation_row(tracker, provider_id: str) -> Dict[str, Any]:
        """Materialize one provider's row from the tracker."""
        rep = tracker.get_reputation(provider_id)
        if rep is None:
            return {
                "provider_id": provider_id,
                "known": False,
                "score": tracker.NEUTRAL_SCORE,
                "successes": 0,
                "failures": 0,
                "preempted": 0,
                "slashed_count": 0,
                "has_been_slashed": False,
                "latency_p50_ms": None,
                "latency_p95_ms": None,
                "first_seen_unix": 0,
                "last_seen_unix": 0,
            }
        return {
            "provider_id": provider_id,
            "known": True,
            "score": tracker.score_for(provider_id),
            "successes": len(rep.successful_dispatches),
            "failures": len(rep.failed_dispatches),
            "preempted": len(rep.preempted_dispatches),
            "slashed_count": tracker.slashed_count(provider_id),
            "has_been_slashed": tracker.has_been_slashed(provider_id),
            "latency_p50_ms": tracker.latency_p50(provider_id),
            "latency_p95_ms": tracker.latency_p95(provider_id),
            "first_seen_unix": rep.first_seen_unix,
            "last_seen_unix": rep.last_seen_unix,
        }

    @app.get("/marketplace/reputation", tags=["marketplace"])
    async def list_marketplace_reputation(
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Sorted (score desc) list of every provider the local
        ReputationTracker has observed."""
        if limit <= 0 or limit > 10000:
            raise HTTPException(
                status_code=422,
                detail=f"limit must be in [1, 10000], got {limit}",
            )
        tracker = getattr(node, "reputation_tracker", None)
        if tracker is None:
            # Sprint 535 F62 fix: actionable hint. Tracker is
            # currently wired only inside the QO-init block (node.py
            # line ~4304) — set PRSM_QUERY_ORCHESTRATOR_ENABLED=1
            # to unlock. Long-term fix: move out of QO-gated block
            # (same class as F34/F44).
            raise HTTPException(
                status_code=503,
                detail=(
                    "Reputation tracker not initialized — set "
                    "PRSM_QUERY_ORCHESTRATOR_ENABLED=1 to enable "
                    "marketplace surfaces."
                ),
            )
        provider_ids = tracker.known_providers()
        rows = [_reputation_row(tracker, pid) for pid in provider_ids]
        rows.sort(
            key=lambda r: (r["score"], r["successes"]),
            reverse=True,
        )
        return {
            "providers": rows[:limit],
            "count": len(provider_ids),
            "limit": limit,
        }

    @app.get(
        "/marketplace/reputation/{provider_id}",
        tags=["marketplace"],
    )
    async def get_marketplace_reputation(
        provider_id: str,
    ) -> Dict[str, Any]:
        """Single-provider detail incl slash event history."""
        tracker = getattr(node, "reputation_tracker", None)
        if tracker is None:
            # Sprint 535 F62 fix: actionable hint. Tracker is
            # currently wired only inside the QO-init block (node.py
            # line ~4304) — set PRSM_QUERY_ORCHESTRATOR_ENABLED=1
            # to unlock. Long-term fix: move out of QO-gated block
            # (same class as F34/F44).
            raise HTTPException(
                status_code=503,
                detail=(
                    "Reputation tracker not initialized — set "
                    "PRSM_QUERY_ORCHESTRATOR_ENABLED=1 to enable "
                    "marketplace surfaces."
                ),
            )
        row = _reputation_row(tracker, provider_id)
        # Add the slash event list (not surfaced in /list rows
        # to keep payloads small).
        row["slash_events"] = [
            {
                "batch_id": e.batch_id,
                "slash_amount_wei": e.slash_amount_wei,
                "reason": e.reason,
                "recorded_unix": e.recorded_unix,
                "tx_hash": e.tx_hash,
            }
            for e in tracker.get_slash_events(provider_id)
        ]
        return row

    @app.get("/admin/royalty-dispatch-summary")
    async def get_royalty_dispatch_summary() -> Dict[str, Any]:
        """Aggregate view over the sprint-249 royalty dispatch
        audit ring. Symmetric to /admin/earnings-summary but for
        the OUTGOING content-royalty flow.

        Returns:
          - total: ring entry count
          - status_counts: {sent, failed, skipped_*}
          - total_sent_wei: sum of gross_wei across sent entries
          - by_allocation_mode: {uniform, rate_weighted, unknown}
          - earliest_ts / latest_ts: timestamp bookends

        Status:
          503 — ring not wired
          200 — summary dict
        """
        ring = getattr(node, "_royalty_dispatch_ring", None)
        if ring is None:
            raise HTTPException(
                status_code=503,
                detail="Royalty dispatch ring not initialized.",
            )
        # Snapshot all entries by passing a large limit (ring caps
        # internally at 1024 by default, so this is bounded).
        entries = ring.recent(limit=1000, offset=0)
        total = len(entries)
        status_counts: Dict[str, int] = {}
        total_sent_wei = 0
        by_mode: Dict[str, int] = {}
        earliest_ts = None
        latest_ts = None
        for e in entries:
            status_counts[e.status] = (
                status_counts.get(e.status, 0) + 1
            )
            if e.status == "sent":
                total_sent_wei += int(e.gross_wei)
            mode = e.allocation_mode or "unknown"
            by_mode[mode] = by_mode.get(mode, 0) + 1
            if earliest_ts is None or e.timestamp < earliest_ts:
                earliest_ts = e.timestamp
            if latest_ts is None or e.timestamp > latest_ts:
                latest_ts = e.timestamp
        return {
            "total": total,
            "status_counts": status_counts,
            "total_sent_wei": total_sent_wei,
            "by_allocation_mode": by_mode,
            "earliest_ts": earliest_ts,
            "latest_ts": latest_ts,
        }

    @app.get("/admin/royalty-dispatch-history")
    async def get_royalty_dispatch_history(
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        job_id: Optional[str] = None,
        allocation_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Recent on-chain content-royalty dispatch outcomes
        (sprint 249 audit ring). Each entry: timestamp, job_id,
        cid, status (sent | skipped_no_record | skipped_bad_hash
        | failed), tx_hash, gross_wei, error. Operators verify
        their PRSM_ONCHAIN_CONTENT_ROYALTY_ENABLED=1 path is
        actually firing distribute_royalty transactions.

        Status:
          503 — ring not wired
          422 — limit out of [1, 1000] OR offset < 0
          200 — {entries, total, offset, limit}
        """
        if limit <= 0 or limit > 1000:
            raise HTTPException(
                status_code=422,
                detail=f"limit must be in [1, 1000], got {limit}",
            )
        if offset < 0:
            raise HTTPException(
                status_code=422,
                detail=f"offset must be >= 0, got {offset}",
            )
        ring = getattr(node, "_royalty_dispatch_ring", None)
        if ring is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Royalty dispatch ring not initialized."
                ),
            )
        entries = ring.recent(
            limit=limit, offset=offset,
            status=status, job_id=job_id,
            allocation_mode=allocation_mode,
        )
        return {
            "entries": [e.to_dict() for e in entries],
            "total": ring.count(),
            "offset": offset,
            "limit": limit,
        }

    @app.get("/admin/slash-history")
    async def get_slash_history(
        limit: int = 50,
        offset: int = 0,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Recent on-chain slash events observed by the
        StorageSlashingWatcher. Each entry: timestamp, kind
        (proof_failure_slashed | heartbeat_missing_slashed),
        provider, slash_id, amount, block_number, tx_hash.

        Optional `provider` filter narrows to a single address.

        Status:
          503 — slash event log not wired
          422 — limit out of [1, 1000] OR offset < 0
          200 — {entries, total, offset, limit}
        """
        if limit <= 0 or limit > 1000:
            raise HTTPException(
                status_code=422,
                detail=f"limit must be in [1, 1000], got {limit}",
            )
        if offset < 0:
            raise HTTPException(
                status_code=422,
                detail=f"offset must be >= 0, got {offset}",
            )
        ring = getattr(node, "_slash_event_log", None)
        if ring is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Slash event log not initialized "
                    "(requires StorageSlashing watcher wiring)."
                ),
            )
        entries = ring.recent(
            limit=limit, offset=offset, provider=provider,
        )
        return {
            "entries": [e.to_dict() for e in entries],
            "total": ring.count(),
            "offset": offset,
            "limit": limit,
        }

    @app.get("/admin/consensus-mismatch-evidence")
    async def get_consensus_mismatch_evidence(
        limit: int = 50,
        offset: int = 0,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Recent CONSENSUS_MISMATCH evidence from the single-provider compute
        pay path (sp928 optimistic verification / sp957 evidence capture).

        When the ComputeResultSampler catches a provider returning an output
        that disagreed with a strict majority of independent re-runs, the
        evidence lands here. Each entry: timestamp, job_id, accused_provider_id,
        accused_output_hash, majority_output_hash, witness_provider_ids,
        accused_bonded, accused_stake_wei, accused_operator_address.

        This is NOT an on-chain slash — an autonomous slash from one node's
        re-execution is unsound (StakeBond.slash is slasher-only; the open
        challengeReceipt rail re-verifies a Merkle proof the off-chain
        single-provider path never produces). It is the operator-reviewable
        corpus a future authority-gated on-chain bridge consumes.

        Optional `provider` filter narrows to one accused provider id.

        Status:
          503 — evidence log not wired
          422 — limit out of [1, 1000] OR offset < 0
          200 — {entries, total, offset, limit}
        """
        if limit <= 0 or limit > 1000:
            raise HTTPException(
                status_code=422,
                detail=f"limit must be in [1, 1000], got {limit}",
            )
        if offset < 0:
            raise HTTPException(
                status_code=422,
                detail=f"offset must be >= 0, got {offset}",
            )
        ring = getattr(node, "_consensus_mismatch_log", None)
        if ring is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Consensus-mismatch evidence log not initialized "
                    "(requires the compute requester / sp928 verification wiring)."
                ),
            )
        entries = ring.recent(limit=limit, offset=offset, provider=provider)
        return {
            "entries": [e.to_dict() for e in entries],
            "total": ring.count(),
            "offset": offset,
            "limit": limit,
        }

    @app.get("/admin/consensus-mismatch-evidence/summary")
    async def get_consensus_mismatch_summary() -> Dict[str, Any]:
        """sp959 — operator visibility into the sp958 dispatch-exclusion policy.

        sp958 makes this node silently stop routing paid jobs to a provider
        with enough confirmed consensus mismatches. This surfaces that policy:
        the active threshold + decay window, and a per-provider breakdown of
        confirmed-mismatch counts with an `excluded` flag — so an operator can
        answer "which providers am I refusing to pay, and why?" without reading
        logs.

        Status:
          503 — compute requester not wired
          200 — {threshold, window_sec, providers: [{provider_id, mismatch_count, excluded}]}
        """
        requester = getattr(node, "compute_requester", None)
        if requester is None or not hasattr(requester, "dispatch_exclusion_summary"):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Compute requester not initialized "
                    "(dispatch-exclusion policy unavailable)."
                ),
            )
        return requester.dispatch_exclusion_summary()

    @app.get("/admin/earnings-summary")
    async def get_earnings_summary() -> Dict[str, Any]:
        """Aggregate operator earnings view.

        Composes per-stream signals operators need to answer
        "is my node earning?":
          * royalty.claimable_wei (from RoyaltyDistributor)
          * heartbeat.last_heartbeat + grace_remaining
              (from StorageSlashing)
          * distribution.last_distribution + seconds_since
              (from CompensationDistributor)

        Each stream is per-stream-isolated: an RPC failure on one
        won't take down the others. Streams unwired return
        available=False so operators see which streams need env config.
        """
        import time as _time
        now = int(_time.time())
        out: Dict[str, Any] = {
            "operator_address": getattr(node, "_operator_address", None),
        }

        royalty_client = getattr(node, "_royalty_distributor_client", None)
        if royalty_client is None:
            # Sprint 152 — operators need to know WHY available=false.
            # client_not_wired → no PRSM_ROYALTY_DISTRIBUTOR_ADDRESS
            #   env AND no canonical addr in networks.py for this network.
            out["royalty"] = {
                "available": False,
                "reason": "client_not_wired",
            }
        else:
            try:
                claimable = royalty_client.claimable()
                out["royalty"] = {
                    "available": True,
                    "claimable_wei": int(claimable),
                    "address": getattr(royalty_client, "address", None),
                }
            except Exception as e:
                out["royalty"] = {
                    "available": False,
                    "error": str(e),
                }

        slash_client = getattr(node, "_storage_slashing_client", None)
        operator_addr = getattr(node, "_operator_address", None)
        if slash_client is None:
            out["heartbeat"] = {
                "available": False,
                "reason": "client_not_wired",
            }
        elif not operator_addr:
            # Sprint 152 — slash client wired but no operator addr.
            # Distinct from client_not_wired so the operator knows to
            # set FTNS_WALLET_PRIVATE_KEY (or PRSM_OPERATOR_ADDRESS).
            out["heartbeat"] = {
                "available": False,
                "reason": "operator_address_missing",
            }
        else:
            try:
                last_hb = int(slash_client.last_heartbeat(operator_addr))
                grace = int(slash_client.heartbeat_grace_seconds())
                hb: Dict[str, Any] = {
                    "available": True,
                    "last_heartbeat": last_hb,
                    "grace_seconds": grace,
                }
                if last_hb == 0:
                    hb["never_recorded"] = True
                    hb["grace_remaining"] = 0
                    hb["expired"] = True
                    hb["at_risk"] = True
                else:
                    elapsed = now - last_hb
                    remaining = max(0, grace - elapsed)
                    hb["grace_remaining"] = remaining
                    hb["expired"] = remaining == 0
                    # at-risk if <10% of grace window left
                    hb["at_risk"] = remaining < (grace * 0.1)
                out["heartbeat"] = hb
            except Exception as e:
                out["heartbeat"] = {
                    "available": False,
                    "error": str(e),
                }

        comp_client = getattr(node, "_compensation_distributor_client", None)
        if comp_client is None:
            out["distribution"] = {
                "available": False,
                "reason": "client_not_wired",
            }
        else:
            try:
                last_dist = int(comp_client.last_distribution_timestamp())
                dist: Dict[str, Any] = {
                    "available": True,
                    "last_distribution": last_dist,
                }
                if last_dist == 0:
                    dist["never_distributed"] = True
                else:
                    dist["seconds_since"] = now - last_dist
                out["distribution"] = dist
            except Exception as e:
                out["distribution"] = {
                    "available": False,
                    "error": str(e),
                }

        return out

    @app.post("/admin/webhook-test")
    async def post_webhook_test() -> Dict[str, Any]:
        """Smoke-test the configured webhook URL. Synthesizes a
        webhook.test event + dispatches; returns the
        DeliveryResult so operators verify config without
        waiting for a real daemon crash.

        Returns 200 even on delivery failure (failure detail is
        in the response body) so operators can distinguish
        "endpoint broken" from "webhook delivery failed."

        Status:
          503 — webhook not configured (PRSM_WEBHOOK_URL unset)
          200 — DeliveryResult shape (success, status_code,
                attempts, error)
        """
        deliverer = getattr(node, "_webhook_deliverer", None)
        watchdog = getattr(node, "_daemon_watchdog", None)
        if deliverer is None or watchdog is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Webhook not configured. Set "
                    "PRSM_WEBHOOK_URL env var to enable."
                ),
            )
        url = getattr(watchdog, "_webhook_url", None)
        secret = getattr(watchdog, "_webhook_secret", None)
        if not url:
            raise HTTPException(
                status_code=503,
                detail="Webhook URL not configured on watchdog.",
            )
        result = await deliverer.deliver(
            url=url,
            event="webhook.test",
            payload={
                "event": "webhook.test",
                "node_id": (
                    node.identity.node_id if node.identity else "unknown"
                ),
                "note": (
                    "Operator-triggered smoke test from "
                    "POST /admin/webhook-test"
                ),
            },
            secret=secret,
        )
        return {
            "success": result.success,
            "status_code": result.status_code,
            "attempts": result.attempts,
            "error": result.error,
        }

    @app.get("/info")
    async def get_info() -> Dict[str, Any]:
        """Static node metadata. Useful for operator triage +
        integration code needing to know which network this node
        is on without parsing /health/detailed.

        Always returns 200 with node_id + api_version; network +
        chain_id + canonical_addresses are surfaced when the
        active PRSM_NETWORK has a known config, else omitted.
        """
        body: Dict[str, Any] = {
            "node_id": (
                node.identity.node_id if node.identity else "unknown"
            ),
            "api_version": app.version,
        }
        # Sprint 169 — surface the derived on-chain operator_address
        # (`_derive_creator_address` priority: ftns_ledger's
        # _connected_address from FTNS_WALLET_PRIVATE_KEY → fallback
        # to PRSM_CREATOR_ADDRESS env). Operators need a quick way
        # to confirm the running node knows its on-chain identity
        # without hitting /admin/earnings-summary (which requires
        # auth in production).
        op_addr = getattr(node, "_operator_address", None)
        if op_addr:
            body["operator_address"] = op_addr
        # Sprint 1196 — advertise whether this operator ACCEPTS requester payment
        # (PRSM_REQUESTER_PAYMENT on + a payment/relayer verifier wired). A requester
        # needs this BEFORE signing: operator_address alone is the node's general
        # on-chain identity and can be set without requester-payment enabled, in
        # which case a paid /compute/inference is rejected (402 verifier-not-wired).
        # The `pay-infer --dry-run` preflight reads this to give an accurate verdict.
        body["requester_payment_accepted"] = bool(
            getattr(node, "_payment_verifier", None)
            or getattr(node, "_relayer_verifier", None)
        )
        # Sprint 173 — surface whether QueryOrchestrator (agent_forge)
        # is wired. Operators flipping PRSM_QUERY_ORCHESTRATOR_ENABLED=1
        # need a quick check that the env var actually produced a
        # wired adapter — vs silently falling back to None on a missing
        # primitive. Boolean: True iff agent_forge is set + non-None.
        body["agent_forge_wired"] = bool(getattr(node, "agent_forge", None))
        # Sprint 425 — surface BitTorrent / content_publisher
        # wiring status. Pre-fix, operators discovered the
        # gap only after attempting an upload + hitting a
        # cryptic 500 error. Now visible in /info so the
        # operator knows BEFORE attempting content workflows.
        # Three layers checked: libtorrent (system dep),
        # bt_client (initialized), content_publisher (attached
        # to ContentUploader). All three must be True for
        # content uploads to succeed.
        content_uploader = getattr(node, "content_uploader", None)
        body["content_publisher_wired"] = bool(
            content_uploader is not None
            and getattr(content_uploader, "content_publisher", None)
            is not None
        )
        # Diagnostic state + error reason (set by
        # `_build_query_orchestrator_or_none` in node.py). Lets operators
        # see WHY agent_forge_wired=False without scraping logs.
        qo_state = getattr(node, "_query_orchestrator_state", None)
        if qo_state:
            body["query_orchestrator_state"] = qo_state
        qo_err = getattr(node, "_query_orchestrator_error", None)
        if qo_err:
            body["query_orchestrator_error"] = qo_err
        # Sprint 327 — compact bootstrap connectivity digest so
        # operators get bootstrap state without a separate
        # /bootstrap/status hit. Fail-soft: if discovery isn't
        # wired or raises, omit the `bootstrap` key entirely so
        # absence is meaningful (no confusing zero/false values).
        # Cumulative counters from sprint 324 stay on
        # /bootstrap/status — /info keeps it tight.
        disco = getattr(node, "discovery", None)
        if disco is not None:
            try:
                full = disco.get_bootstrap_status()
                body["bootstrap"] = {
                    "client_state": full.get("client_state", "?"),
                    "connected": int(full.get("connected", 0) or 0),
                    "degraded": bool(full.get("degraded", False)),
                    "discovered_peer_count": int(
                        full.get("discovered_peer_count", 0) or 0
                    ),
                }
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "info: bootstrap summary skipped: %s", exc,
                )
        try:
            from prsm.config.networks import (
                get_network_config, _resolve_network_name,
                resolve_endpoints,
            )
            network_name = _resolve_network_name()
            cfg = get_network_config(network_name)
            body["network"] = network_name
            body["chain_id"] = cfg.chain_id
            # Sprint 170 — surface the RPC HOST (not full URL) so
            # operators can verify they're pointed at the right
            # endpoint without leaking Alchemy/Infura API keys that
            # live in the URL path.
            try:
                from urllib.parse import urlparse
                endpoints = resolve_endpoints(network_name)
                parsed = urlparse(endpoints.rpc_url)
                if parsed.hostname:
                    body["rpc_host"] = parsed.hostname
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "info: rpc_url host extraction skipped: %s", exc,
                )
            canonical: Dict[str, Optional[str]] = {}
            # Sprint 143 — surface ALL mainnet canonical pins,
            # not just the FTNS-cluster five. Operators use
            # /info to cross-check what their node's wired
            # clients should match. Missing entries here
            # silently skipped operators' validation.
            for fld in (
                # FTNS-cluster (pre-existing 5)
                "ftns_token", "provenance_registry",
                "provenance_registry_v2", "royalty_distributor",
                "foundation_safe",
                # Phase 7-storage + Phase 8 (sprint 142 set —
                # the contracts whose canonical-match check we
                # JUST fixed)
                "storage_slashing", "compensation_distributor",
                "key_distribution",
                # Audit-bundle + emission infrastructure
                "emission_controller", "escrow_pool",
                "stake_bond", "settlement_registry",
                "signature_verifier", "publisher_key_anchor",
            ):
                val = getattr(cfg, fld, None)
                if val is not None:
                    canonical[fld] = val
            if canonical:
                body["canonical_addresses"] = canonical
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "info: network config lookup skipped: %s", exc,
            )
        return body

    @app.get("/health")
    async def health() -> Dict[str, str]:
        """Simple health check.

        Sprint 186 + 187 — HEAD is supported via the generic
        head_as_get_middleware near the top of create_api_app.
        Both methods return 200 with the same headers; HEAD strips
        the body per RFC 7231.
        """
        return {"status": "ok", "node_id": node.identity.node_id if node.identity else "unknown"}

    # Sprint 1186 (day-one-live blocker #9) — READINESS probe (distinct from the /health
    # LIVENESS probe above). Returns 503 when the node can't serve its primary path
    # (inference dark) so a load balancer DRAINS traffic instead of routing to a broken
    # node — without the 503-on-liveness mistake that would make k8s RESTART a merely-
    # not-ready pod. Ungated (not in _GATED_PATHS) so the LB can reach it like /health;
    # body is coarse non-sensitive flags only. Cheap (no RPC).
    async def _readiness_response():
        from fastapi.responses import JSONResponse
        from prsm.node.readiness import compute_readiness
        ready, detail = compute_readiness(node)
        detail["node_id"] = node.identity.node_id if node.identity else "unknown"
        return JSONResponse(status_code=200 if ready else 503, content=detail)

    @app.get("/health/ready")
    async def health_ready():
        """Readiness probe — 200 when the node can serve inference, 503 otherwise."""
        return await _readiness_response()

    @app.get("/readyz")
    async def readyz():
        """Alias of /health/ready (the conventional k8s readiness path name)."""
        return await _readiness_response()

    @app.get("/metrics")
    async def get_metrics() -> Any:
        """Prometheus text/plain exposition for ops dashboards.
        Emits gauges from live node state without new tracking
        infra. Fail-soft per-metric: a subsystem RPC error logs
        + omits that gauge rather than 500-ing the endpoint."""
        from fastapi.responses import PlainTextResponse

        lines: list = []

        # prsm_pending_escrow_count + prsm_total_locked_ftns
        try:
            esc = getattr(node, "_payment_escrow", None)
            if esc is not None:
                pending_count = 0
                total_locked = 0.0
                for e in esc._escrows.values():
                    if e.status.value == "pending":
                        pending_count += 1
                        total_locked += e.amount
                lines.append(
                    "# HELP prsm_pending_escrow_count "
                    "Pending escrow count"
                )
                lines.append(
                    "# TYPE prsm_pending_escrow_count gauge"
                )
                lines.append(
                    f"prsm_pending_escrow_count {pending_count}"
                )
                lines.append(
                    "# HELP prsm_total_locked_ftns "
                    "Sum of FTNS locked in PENDING escrows"
                )
                lines.append("# TYPE prsm_total_locked_ftns gauge")
                lines.append(
                    f"prsm_total_locked_ftns {total_locked}"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("metrics escrow probe failed: %s", exc)

        # prsm_job_history_size
        try:
            hist = getattr(node, "_job_history", None)
            if hist is not None:
                lines.append(
                    "# HELP prsm_job_history_size "
                    "Job history record count"
                )
                lines.append("# TYPE prsm_job_history_size gauge")
                lines.append(
                    f"prsm_job_history_size {hist.size()}"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("metrics history probe failed: %s", exc)

        # Sprint 254 — prsm_receipt_store_size. Stored signed
        # InferenceReceipts (forge+inference+stream paths). Useful
        # for operators tracking long-term audit-trail growth.
        try:
            rstore = getattr(node, "_receipt_store", None)
            if rstore is not None:
                lines.append(
                    "# HELP prsm_receipt_store_size "
                    "Stored signed InferenceReceipt count"
                )
                lines.append(
                    "# TYPE prsm_receipt_store_size gauge"
                )
                lines.append(
                    f"prsm_receipt_store_size {rstore.count()}"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "metrics receipt_store probe failed: %s", exc,
            )

        # Sprint 816 — prsm_inference_output_cache_*. Cache
        # observability for operators tuning TTL + MAX_ENTRIES.
        # Counters use *_total suffix per Prometheus convention.
        # size is a gauge (snapshot).
        try:
            executor = getattr(node, "inference_executor", None)
            cache = (
                getattr(executor, "_output_cache", None)
                if executor is not None else None
            )
            if cache is not None:
                stats = cache.stats()
                # Counters
                for name, desc, value in [
                    (
                        "prsm_inference_output_cache_hits_total",
                        "OutputCache hit count (cumulative)",
                        stats.get("hits", 0),
                    ),
                    (
                        "prsm_inference_output_cache_misses_total",
                        "OutputCache miss count (cumulative)",
                        stats.get("misses", 0),
                    ),
                    (
                        "prsm_inference_output_cache_puts_total",
                        "OutputCache put count (cumulative)",
                        stats.get("puts", 0),
                    ),
                    (
                        "prsm_inference_output_cache_evictions_total",
                        "OutputCache LRU evictions (cumulative)",
                        stats.get("evictions", 0),
                    ),
                    (
                        "prsm_inference_output_cache_ttl_evictions_total",
                        "OutputCache TTL evictions (cumulative)",
                        stats.get("ttl_evictions", 0),
                    ),
                ]:
                    lines.append(f"# HELP {name} {desc}")
                    lines.append(f"# TYPE {name} counter")
                    lines.append(f"{name} {value}")
                # Gauge
                lines.append(
                    "# HELP prsm_inference_output_cache_size "
                    "OutputCache current entry count"
                )
                lines.append(
                    "# TYPE prsm_inference_output_cache_size gauge"
                )
                lines.append(
                    f"prsm_inference_output_cache_size "
                    f"{stats.get('size', 0)}"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "metrics output_cache probe failed: %s", exc,
            )

        # Sprint 254 — prsm_royalty_dispatch_ring_size. On-chain
        # content-royalty dispatch audit outcomes. Goes nonzero
        # only when PRSM_ONCHAIN_CONTENT_ROYALTY_ENABLED=1.
        try:
            ring = getattr(node, "_royalty_dispatch_ring", None)
            if ring is not None:
                lines.append(
                    "# HELP prsm_royalty_dispatch_ring_size "
                    "On-chain royalty dispatch outcome count"
                )
                lines.append(
                    "# TYPE prsm_royalty_dispatch_ring_size gauge"
                )
                lines.append(
                    f"prsm_royalty_dispatch_ring_size "
                    f"{ring.count()}"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "metrics royalty_dispatch probe failed: %s", exc,
            )

        # prsm_claimable_royalties_wei
        try:
            royalty = getattr(node, "_royalty_distributor_client", None)
            if royalty is not None:
                claimable = royalty.claimable()
                lines.append(
                    "# HELP prsm_claimable_royalties_wei "
                    "Pull-payment balance in wei"
                )
                lines.append(
                    "# TYPE prsm_claimable_royalties_wei gauge"
                )
                lines.append(
                    f"prsm_claimable_royalties_wei {claimable}"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("metrics royalty probe failed: %s", exc)

        # prsm_escrow_cleanup_task_running + per-daemon task_running
        # gauges. Same lifecycle-watch pattern as the
        # /health/detailed daemon subsystems.
        def _emit_task_gauge(metric_name: str, task_attr: str, help_text: str):
            task = getattr(node, task_attr, None)
            if task is None:
                return
            try:
                running = 0 if task.done() else 1
                lines.append(f"# HELP {metric_name} {help_text}")
                lines.append(f"# TYPE {metric_name} gauge")
                lines.append(f"{metric_name} {running}")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "metrics %s probe failed: %s", metric_name, exc,
                )

        _emit_task_gauge(
            "prsm_escrow_cleanup_task_running",
            "_escrow_cleanup_task",
            "1 if escrow auto-refund cleanup task is alive, 0 if crashed",
        )
        _emit_task_gauge(
            "prsm_heartbeat_scheduler_running",
            "_heartbeat_scheduler_task",
            "1 if HeartbeatScheduler task is alive, 0 if crashed",
        )
        _emit_task_gauge(
            "prsm_compensation_scheduler_running",
            "_compensation_scheduler_task",
            "1 if CompensationScheduler task is alive, 0 if crashed",
        )
        _emit_task_gauge(
            "prsm_key_distribution_watcher_running",
            "_key_distribution_watcher_task",
            "1 if KeyDistribution watcher task is alive, 0 if crashed",
        )
        _emit_task_gauge(
            "prsm_storage_slashing_watcher_running",
            "_storage_slashing_watcher_task",
            "1 if StorageSlashing watcher task is alive, 0 if crashed",
        )
        _emit_task_gauge(
            "prsm_compensation_distributor_watcher_running",
            "_compensation_distributor_watcher_task",
            "1 if CompensationDistributor watcher task is alive, 0 if crashed",
        )
        _emit_task_gauge(
            "prsm_job_reaper_running",
            "_job_reaper_task",
            "1 if JobReaper duration-cap task is alive, 0 if crashed",
        )
        _emit_task_gauge(
            "prsm_daemon_watchdog_running",
            "_daemon_watchdog_task",
            "1 if DaemonWatchdog crash-webhook task is alive, 0 if crashed",
        )

        # prsm_build_info — version label gauge, standard
        # Prometheus build-info pattern. Operators correlate
        # alerts to release boundaries.
        try:
            from importlib.metadata import version as _pkg_version
            try:
                pkg_version = _pkg_version("prsm-network")
            except Exception:  # noqa: BLE001
                pkg_version = "unknown"
            lines.append(
                "# HELP prsm_build_info "
                "Build info (always 1; version label)"
            )
            lines.append("# TYPE prsm_build_info gauge")
            lines.append(
                f'prsm_build_info{{version="{pkg_version}"}} 1'
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("metrics build_info probe failed: %s", exc)

        # In-memory ring buffer counts. 4 dashboard rings —
        # operators alert on growth/stagnation patterns. Each
        # gauge emitted only when the ring is wired (None when
        # unwired = gauge omitted, NOT zero).
        for ring_metric, ring_attr, help_text in (
            (
                "prsm_webhook_log_count", "_webhook_log",
                "WebhookDeliverer ring buffer entry count",
            ),
            (
                "prsm_slash_event_log_count", "_slash_event_log",
                "SlashEventRing entry count (proof_failure + missing_heartbeat)",
            ),
            (
                "prsm_heartbeat_log_count", "_heartbeat_log",
                "HeartbeatRecordedRing entry count",
            ),
            (
                "prsm_distribution_log_count", "_distribution_log",
                "DistributedEventRing entry count (Phase 8 emission rounds)",
            ),
            (
                "prsm_consensus_mismatch_log_count", "_consensus_mismatch_log",
                "ConsensusMismatchLog entry count (sp957 — confirmed compute "
                "re-execution mismatches; alert on growth)",
            ),
        ):
            try:
                ring = getattr(node, ring_attr, None)
                if ring is not None:
                    count = ring.count()
                    lines.append(f"# HELP {ring_metric} {help_text}")
                    lines.append(f"# TYPE {ring_metric} gauge")
                    lines.append(f"{ring_metric} {count}")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "metrics %s probe failed: %s", ring_metric, exc,
                )

        # prsm_dispatch_excluded_providers — sp962: how many providers this node
        # is currently EXCLUDING from compute dispatch (sp958 enforcement of the
        # sp957 evidence). Fleet operators alert on this rising. Emitted only
        # when the compute requester is wired.
        try:
            requester = getattr(node, "compute_requester", None)
            if requester is not None and hasattr(requester, "dispatch_exclusion_summary"):
                summary = requester.dispatch_exclusion_summary()
                excluded = sum(
                    1 for p in summary.get("providers", []) if p.get("excluded")
                )
                lines.append(
                    "# HELP prsm_dispatch_excluded_providers "
                    "Providers currently barred from compute dispatch "
                    "(confirmed consensus mismatches >= threshold)"
                )
                lines.append("# TYPE prsm_dispatch_excluded_providers gauge")
                lines.append(f"prsm_dispatch_excluded_providers {excluded}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "metrics prsm_dispatch_excluded_providers probe failed: %s", exc,
            )

        # prsm_arbitration_pending_count
        try:
            arb = getattr(node, "_arbitration_queue", None)
            if arb is not None:
                pending = await arb.list_pending()
                lines.append(
                    "# HELP prsm_arbitration_pending_count "
                    "Pending content-attribution disputes"
                )
                lines.append(
                    "# TYPE prsm_arbitration_pending_count gauge"
                )
                lines.append(
                    f"prsm_arbitration_pending_count {len(pending)}"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("metrics arbitration probe failed: %s", exc)

        # Sprint 328 — bootstrap discovery counters + gauges
        # for ops dashboards. Counters from sprint 324 are
        # cumulative-monotonic (process-restart reset); live
        # gauges reflect point-in-time snapshots. Fail-soft per
        # the rest of /metrics: omit block on any error.
        try:
            disco = getattr(node, "discovery", None)
            if disco is not None:
                bs = disco.get_bootstrap_status()
                # Counters (sprint 324 cumulative)
                counter_metrics = (
                    ("prsm_bootstrap_peer_join_events_total",
                     "Cumulative peer_join announcements consumed",
                     bs.get("peer_join_events", 0)),
                    ("prsm_bootstrap_peer_leave_events_total",
                     "Cumulative peer_leave announcements consumed",
                     bs.get("peer_leave_events", 0)),
                    ("prsm_bootstrap_stale_evictions_total",
                     "Cumulative peers swept by last_seen threshold",
                     bs.get("stale_evictions", 0)),
                    ("prsm_bootstrap_reconnect_attempts_total",
                     "Cumulative reconnect dispatches in poll loop",
                     bs.get("reconnect_attempts", 0)),
                    ("prsm_bootstrap_reconnect_successes_total",
                     "Cumulative reconnect successes",
                     bs.get("reconnect_successes", 0)),
                )
                for name, help_text, value in counter_metrics:
                    lines.append(f"# HELP {name} {help_text}")
                    lines.append(f"# TYPE {name} counter")
                    lines.append(f"{name} {int(value or 0)}")
                # Gauges (point-in-time)
                gauge_metrics = (
                    ("prsm_bootstrap_connected",
                     "Bootstrap nodes currently registered with",
                     int(bs.get("connected", 0) or 0)),
                    ("prsm_bootstrap_discovered_peer_count",
                     "Peers visible via last bootstrap poll",
                     int(bs.get("discovered_peer_count", 0) or 0)),
                    ("prsm_bootstrap_degraded",
                     "1 iff all bootstrap nodes unreachable",
                     1 if bs.get("degraded", False) else 0),
                )
                for name, help_text, value in gauge_metrics:
                    lines.append(f"# HELP {name} {help_text}")
                    lines.append(f"# TYPE {name} gauge")
                    lines.append(f"{name} {value}")
                # Sprint 377 — multi-bootstrap surface.
                # fallback_enabled as a plain gauge; active_url
                # as a labeled gauge with value 1 (canonical
                # Prometheus pattern for current-string-state
                # — operators do `count by (url)` across the
                # fleet to graph bootstrap distribution).
                fb_enabled = bs.get(
                    "bootstrap_fallback_enabled",
                )
                if fb_enabled is not None:
                    lines.append(
                        "# HELP prsm_bootstrap_fallback_enabled "
                        "1 iff multi-region bootstrap fallback "
                        "is enabled"
                    )
                    lines.append(
                        "# TYPE prsm_bootstrap_fallback_enabled "
                        "gauge"
                    )
                    lines.append(
                        f"prsm_bootstrap_fallback_enabled "
                        f"{1 if fb_enabled else 0}"
                    )
                active_url = bs.get("active_url")
                if active_url:
                    # Prometheus label-value escape rules: \"
                    # for ", \\ for \, \n for newline.
                    escaped = (
                        str(active_url)
                        .replace("\\", "\\\\")
                        .replace('"', '\\"')
                        .replace("\n", "\\n")
                    )
                    lines.append(
                        "# HELP prsm_bootstrap_active "
                        "Current bootstrap URL the node is "
                        "registered with (labeled gauge)"
                    )
                    lines.append(
                        "# TYPE prsm_bootstrap_active gauge"
                    )
                    lines.append(
                        f'prsm_bootstrap_active{{url="'
                        f'{escaped}"}} 1'
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("metrics bootstrap probe failed: %s", exc)

        # Sprint 345 — gauges for the §7/§14 orchestrators +
        # stores wired in sprints 342/343. Each probe is in its
        # own try/except so a single subsystem failure omits its
        # gauge without 500-ing the endpoint. Subsystem not
        # wired → gauge omitted entirely (Prometheus reads
        # absence as "feature disabled" instead of "wired but
        # zero").
        def _emit_count_gauge(
            attr: str, probe, metric_name: str, help_text: str,
        ) -> None:
            store = getattr(node, attr, None)
            if store is None:
                return
            try:
                value = int(probe(store))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "metrics %s probe failed: %s",
                    metric_name, exc,
                )
                return
            lines.append(f"# HELP {metric_name} {help_text}")
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name} {value}")

        _emit_count_gauge(
            "_federated_learning_orchestrator",
            lambda x: len(x.list_jobs() or []),
            "prsm_fl_jobs_count",
            "Federated-learning orchestrator job count",
        )
        _emit_count_gauge(
            "_pipeline_inference_orchestrator",
            lambda x: len(x.list_jobs() or []),
            "prsm_pipeline_jobs_count",
            "Pipeline-inference orchestrator job count",
        )
        _emit_count_gauge(
            "_content_filter_store",
            lambda x: x.count(),
            "prsm_content_filter_count",
            "Content-filter rule record count",
        )
        _emit_count_gauge(
            "_disclosure_intake",
            lambda x: x.count(),
            "prsm_disclosure_count",
            "Responsible-disclosure record count",
        )
        _emit_count_gauge(
            "_incident_response",
            lambda x: x.count(),
            "prsm_incident_count",
            "Security-incident record count",
        )
        _emit_count_gauge(
            "_corp_capability_store",
            lambda x: len(x.list_issuers()),
            "prsm_corp_issuer_count",
            "$CORP capability issuer count",
        )
        _emit_count_gauge(
            "_upgrade_orchestrator",
            lambda x: x.count(),
            "prsm_upgrade_count",
            "UUPS upgrade proposal record count",
        )

        # Sprint 395 — Per-subsystem labeled gauges sourced
        # from /health/detailed. Mirrors sprint-394 on the
        # bootstrap-server side. Encoding:
        #   0 = healthy / available (status="ok")
        #   1 = optional-opt-out (not_wired / disabled /
        #       uninitialized — distinct from hard failure)
        #   2 = error / unhealthy / unknown
        # Wrapped in try/except per the fail-soft per-metric
        # convention — a /health/detailed crash MUST NOT
        # 500 the /metrics endpoint.
        try:
            detailed = await health_detailed()
            subs = detailed.get("subsystems") or {}
            if subs:
                def _escape_label(s: str) -> str:
                    return (
                        s.replace("\\", "\\\\")
                         .replace('"', '\\"')
                         .replace("\n", "\\n")
                    )

                def _encode_status(sub: Dict[str, Any]) -> int:
                    status = sub.get("status")
                    available = sub.get("available")
                    # Sprint 402 — incorporate tick_status
                    # from sprint 399-401 daemon extensions.
                    # tick_status=stale wins over a basic
                    # "ok" because a daemon whose task is
                    # running but ticks are failing is
                    # observably unhealthy.
                    tick_status = sub.get("tick_status")
                    if tick_status == "stale":
                        return 2
                    if tick_status == "degraded":
                        return 1
                    if status == "ok" and available:
                        return 0
                    # Explicit operator-opt-out signals only.
                    # "uninitialized" is NOT opt-out — it
                    # means a core subsystem hasn't connected,
                    # which is a hard failure for that node.
                    if status in ("not_wired", "disabled"):
                        return 1
                    return 2

                lines.append(
                    "# HELP prsm_node_subsystem_status "
                    "Per-subsystem readiness "
                    "(0=healthy, 1=optional-opt-out, "
                    "2=unhealthy)"
                )
                lines.append(
                    "# TYPE prsm_node_subsystem_status gauge"
                )
                for sub_name, sub_data in subs.items():
                    if not isinstance(sub_data, dict):
                        continue
                    label = _escape_label(sub_name)
                    value = _encode_status(sub_data)
                    lines.append(
                        f'prsm_node_subsystem_status'
                        f'{{subsystem="{label}"}} {value}'
                    )

                # Sprint 402 — dedicated tick-age gauge for
                # daemons that adopted the sprint-399-401
                # pattern. Mirrors sprint-394's bootstrap-
                # side prsm_bootstrap_subsystem_heartbeat_
                # age_seconds. PromQL alerts on heartbeat
                # age directly:
                #   prsm_node_subsystem_tick_age_seconds
                #     {subsystem="heartbeat_scheduler"} > 1800
                tick_age_lines = []
                for sub_name, sub_data in subs.items():
                    if not isinstance(sub_data, dict):
                        continue
                    age = sub_data.get("last_tick_age_seconds")
                    if not isinstance(age, (int, float)):
                        continue
                    tick_age_lines.append(
                        f'prsm_node_subsystem_tick_age_seconds'
                        f'{{subsystem="{_escape_label(sub_name)}"}}'
                        f' {age}'
                    )
                if tick_age_lines:
                    lines.append(
                        "# HELP prsm_node_subsystem_tick_age_seconds "
                        "Seconds since last successful tick per "
                        "daemon subsystem"
                    )
                    lines.append(
                        "# TYPE prsm_node_subsystem_tick_age_seconds "
                        "gauge"
                    )
                    lines.extend(tick_age_lines)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "metrics subsystem block failed: %s", exc,
            )

        # Always emit at least one metric so probes have something
        # to scrape. The "up" gauge is canonical for this.
        lines.append("# HELP prsm_node_up Node-up indicator")
        lines.append("# TYPE prsm_node_up gauge")
        lines.append("prsm_node_up 1")

        # sp1217 — append the opt-in runtime MetricsCollector's bridged
        # Prometheus exposition (NodeRuntimeMetrics: readiness + on-chain
        # settlement). None unless PRSM_RUNTIME_METRICS_ENABLED, so when off the
        # output is byte-for-byte unchanged. Fail-soft: a render error never
        # breaks the scrape.
        try:
            _mc = getattr(node, "_metrics_collector", None)
            if _mc is not None:
                _extra = _mc.get_prometheus_metrics()
                # Only a genuine non-empty str is appended — guards the
                # join() below from a mock/None render (never 500 the scrape).
                if isinstance(_extra, str) and _extra.strip():
                    lines.append(_extra.rstrip("\n"))
        except Exception:  # noqa: BLE001 — a metrics scrape must never 500
            pass

        body = "\n".join(lines) + "\n"
        return PlainTextResponse(
            body, media_type="text/plain; version=0.0.4",
        )

    @app.get("/health/detailed")
    async def health_detailed() -> Dict[str, Any]:
        """Structured per-subsystem readiness probe for ops
        monitoring. Distinct from /health (which stays minimal
        for load-balancer probes).

        Top-level status:
          - healthy: all wired subsystems operational
          - degraded: optional subsystems unavailable but core
            (FTNS ledger + payment escrow) works
          - unhealthy: core subsystem missing or erroring

        Per-subsystem fields: {available, status, ...details, error?}.
        """
        subsystems: Dict[str, Dict[str, Any]] = {}

        # FTNS ledger (core).
        ftns_ledger = getattr(node, "ftns_ledger", None)
        if ftns_ledger is not None:
            try:
                addr = getattr(ftns_ledger, "_connected_address", None)
                init = getattr(ftns_ledger, "_is_initialized", False)
                entry = {
                    "available": bool(init),
                    "status": "ok" if init else "uninitialized",
                    "connected_address": addr,
                }
            except Exception as exc:  # noqa: BLE001
                entry = {
                    "available": False, "status": "error",
                    "error": str(exc),
                }
            # Canonical-match check on the FTNS token address.
            wired_token = getattr(ftns_ledger, "contract_address", None)
            if wired_token is not None:
                entry["wired_address"] = wired_token
                try:
                    from prsm.config.networks import (
                        get_network_config, _resolve_network_name,
                    )
                    cfg = get_network_config(_resolve_network_name())
                    canonical = cfg.ftns_token
                    if canonical is not None:
                        entry["canonical_address"] = canonical
                        entry["canonical_match"] = (
                            wired_token.lower() == canonical.lower()
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "canonical-match lookup failed for "
                        "ftns_ledger: %s", exc,
                    )
            subsystems["ftns_ledger"] = entry
        else:
            subsystems["ftns_ledger"] = {
                "available": False, "status": "not_wired",
            }

        # Sprint 503 — operator gas balance. Wired to ftns_ledger
        # but surfaced as its own subsystem so monitoring tools
        # picking up /health/detailed get the gas signal without
        # extra config. Mirrors /wallet/gas-status thresholds.
        if ftns_ledger is not None:
            try:
                gas_addr = getattr(
                    ftns_ledger, "_connected_address", None,
                )
                w3 = getattr(ftns_ledger, "w3", None)
                if w3 is not None and gas_addr is not None:
                    wei = w3.eth.get_balance(gas_addr)
                    eth = wei / 1e18
                    if eth < 0.0001:
                        gas_status = "critical"
                    elif eth < 0.0005:
                        gas_status = "low"
                    else:
                        gas_status = "ok"
                    subsystems["operator_gas"] = {
                        "available": True,
                        "status": gas_status,
                        "address": gas_addr,
                        "eth_balance": eth,
                        "low_threshold_eth": 0.0005,
                        "critical_threshold_eth": 0.0001,
                    }
                else:
                    subsystems["operator_gas"] = {
                        "available": False,
                        "status": "unavailable",
                    }
            except Exception as exc:  # noqa: BLE001
                subsystems["operator_gas"] = {
                    "available": False,
                    "status": "error",
                    "error": str(exc)[:200],
                }
        else:
            subsystems["operator_gas"] = {
                "available": False, "status": "not_wired",
            }

        # Sprint 515 — inbound_monitor subsystem. Exposes
        # sprint 514's poller state so monitoring tools see
        # whether the inbound scan is healthy + which block
        # was last scanned (no event-loss).
        inbound_mon = getattr(node, "_inbound_monitor", None)
        if inbound_mon is not None:
            try:
                subsystems["inbound_monitor"] = {
                    "available": True,
                    "status": "ok",
                    "last_scanned_block": getattr(
                        inbound_mon, "_last_scanned_block",
                        None,
                    ),
                    "interval_seconds": getattr(
                        inbound_mon, "interval_seconds",
                        None,
                    ),
                }
            except Exception as exc:  # noqa: BLE001
                subsystems["inbound_monitor"] = {
                    "available": False, "status": "error",
                    "error": str(exc)[:200],
                }
        else:
            subsystems["inbound_monitor"] = {
                "available": False, "status": "not_wired",
            }

        # Sprint 553 — watcher_event_dedup subsystem. Sprints
        # 549/550/551 ship persistent (watcher_key, tx_hash,
        # log_index) dedup for the 3 event watchers; sprint 552
        # exposed an /admin endpoint. This entry surfaces the same
        # state on the canonical operator probe surface so
        # monitoring tools see whether dedup is wired without
        # discovering /admin/watcher-event-dedup independently.
        dedup_store = getattr(
            node, "_watcher_event_dedup_store", None,
        )
        if dedup_store is not None:
            try:
                summary = dedup_store.summary()
                total = sum(
                    s.get("rows_processed", 0)
                    for s in summary.values()
                )
                subsystems["watcher_event_dedup"] = {
                    "available": True,
                    "status": "ok",
                    "total_rows_processed": total,
                    "watchers": summary,
                }
            except Exception as exc:  # noqa: BLE001
                subsystems["watcher_event_dedup"] = {
                    "available": False,
                    "status": "error",
                    "error": str(exc)[:200],
                }
        else:
            subsystems["watcher_event_dedup"] = {
                "available": False,
                "status": "not_wired",
                "hint": (
                    "Set PRSM_WATCHER_STATE_PERSISTENCE_ENABLED=1 "
                    "(and optionally PRSM_WATCHER_EVENT_DEDUP_DB="
                    "<path>) and restart the daemon to enable "
                    "persistent watcher event dedup."
                ),
            }

        # Payment escrow (core).
        escrow_svc = getattr(node, "_payment_escrow", None)
        if escrow_svc is not None:
            try:
                pending_count = sum(
                    1 for e in escrow_svc._escrows.values()
                    if e.status.value == "pending"
                )
                entry = {
                    "available": True, "status": "ok",
                    "pending_count": pending_count,
                    "default_timeout_sec": getattr(
                        escrow_svc, "default_timeout", None,
                    ),
                }
                # Cleanup-task health probe: the periodic_cleanup
                # task is an infinite asyncio loop; if .done() ever
                # returns True it's because the task crashed
                # (raised, was cancelled, or otherwise terminated).
                # Operators see cleanup_task_running == False as
                # the silent-failure signal.
                cleanup_task = getattr(
                    node, "_escrow_cleanup_task", None,
                )
                if cleanup_task is not None:
                    try:
                        entry["cleanup_task_running"] = (
                            not cleanup_task.done()
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "cleanup_task probe raised: %s", exc,
                        )
            except Exception as exc:  # noqa: BLE001
                entry = {
                    "available": False, "status": "error",
                    "error": str(exc),
                }
            subsystems["payment_escrow"] = entry
        else:
            subsystems["payment_escrow"] = {
                "available": False, "status": "not_wired",
            }

        # Sprint 482 — actionable operator hint when filesystem
        # persistence is opt-in but not configured. Each subsystem
        # has a canonical PRSM_*_DIR env var; surfacing the name
        # alongside `persisted: false` gives operators a one-step
        # remediation path instead of grepping the codebase.
        _PERSISTENCE_ENV_VAR = {
            "job_history": "PRSM_JOB_HISTORY_DIR",
            "receipt_store": "PRSM_RECEIPT_STORE_DIR",
            "royalty_dispatch_ring": (
                "PRSM_ROYALTY_DISPATCH_LOG_DIR"
            ),
            "webhook_log": "PRSM_WEBHOOK_LOG_DIR",
            "slash_event_log": "PRSM_SLASH_EVENT_LOG_DIR",
            "heartbeat_log": "PRSM_HEARTBEAT_LOG_DIR",
            "distribution_log": "PRSM_DISTRIBUTION_LOG_DIR",
        }

        def _add_persistence_hint(entry: Dict[str, Any], name: str) -> None:
            if entry.get("persisted") is False:
                hint = _PERSISTENCE_ENV_VAR.get(name)
                if hint:
                    entry["persistence_env_var"] = hint

        # Job history (optional).
        history = getattr(node, "_job_history", None)
        if history is not None:
            try:
                subsystems["job_history"] = {
                    "available": True, "status": "ok",
                    "count": history.size(),
                    "max_entries": history._max_entries,
                    "persisted": history._persist_dir is not None,
                }
                _add_persistence_hint(
                    subsystems["job_history"], "job_history",
                )
            except Exception as exc:  # noqa: BLE001
                subsystems["job_history"] = {
                    "available": False, "status": "error",
                    "error": str(exc),
                }
        else:
            subsystems["job_history"] = {
                "available": False, "status": "not_wired",
            }

        # Sprint 255 — Receipt store (optional). Holds the signed
        # InferenceReceipts persisted by /compute/inference +
        # /compute/inference/stream. Operators see persisted=True
        # when PRSM_RECEIPT_STORE_DIR is set.
        rstore = getattr(node, "_receipt_store", None)
        if rstore is not None:
            try:
                subsystems["receipt_store"] = {
                    "available": True, "status": "ok",
                    "count": rstore.count(),
                    "persisted": (
                        rstore._persist_dir is not None
                    ),
                }
                _add_persistence_hint(
                    subsystems["receipt_store"], "receipt_store",
                )
            except Exception as exc:  # noqa: BLE001
                subsystems["receipt_store"] = {
                    "available": False, "status": "error",
                    "error": str(exc),
                }
        else:
            subsystems["receipt_store"] = {
                "available": False, "status": "not_wired",
            }

        # Sprint 255 — Royalty dispatch ring (optional). Holds
        # per-shard on-chain dispatch outcomes. Goes nonzero only
        # when PRSM_ONCHAIN_CONTENT_ROYALTY_ENABLED=1 and a
        # RoyaltyDistributorClient is wired.
        royalty_ring = getattr(node, "_royalty_dispatch_ring", None)
        if royalty_ring is not None:
            try:
                subsystems["royalty_dispatch_ring"] = {
                    "available": True, "status": "ok",
                    "count": royalty_ring.count(),
                    "persisted": (
                        royalty_ring._persist_dir is not None
                    ),
                }
                _add_persistence_hint(
                    subsystems["royalty_dispatch_ring"],
                    "royalty_dispatch_ring",
                )
            except Exception as exc:  # noqa: BLE001
                subsystems["royalty_dispatch_ring"] = {
                    "available": False, "status": "error",
                    "error": str(exc),
                }
        else:
            subsystems["royalty_dispatch_ring"] = {
                "available": False, "status": "not_wired",
            }

        # Provenance registry (optional). Canonical pin is V2
        # (post-A-08 ceremony 2026-05-09); v1 callers will surface
        # canonical_match=False as a signal to migrate.
        provenance = getattr(node, "_provenance_client", None)
        if provenance is not None:
            entry = {"available": True, "status": "ok"}
            wired = getattr(provenance, "contract_address", None)
            if wired is not None:
                entry["wired_address"] = wired
                try:
                    from prsm.config.networks import (
                        get_network_config, _resolve_network_name,
                    )
                    cfg = get_network_config(_resolve_network_name())
                    canonical = cfg.provenance_registry_v2
                    if canonical is not None:
                        entry["canonical_address"] = canonical
                        entry["canonical_match"] = (
                            wired.lower() == canonical.lower()
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "canonical-match lookup failed for "
                        "provenance_registry: %s", exc,
                    )
            subsystems["provenance_registry"] = entry
        else:
            subsystems["provenance_registry"] = {
                "available": False, "status": "not_wired",
            }

        # Royalty distributor (optional).
        royalty = getattr(node, "_royalty_distributor_client", None)
        if royalty is not None:
            try:
                # Probe via claimable() — a read-only call that
                # exercises RPC connectivity.
                _claimable_wei = royalty.claimable()
                entry = {
                    "available": True, "status": "ok",
                    "claimable_wei": _claimable_wei,
                }
            except Exception as exc:  # noqa: BLE001
                entry = {
                    "available": False, "status": "error",
                    "error": str(exc),
                }
            # Canonical-match check (post-A-08 ceremony 2026-05-09).
            # Operators get an instant signal whether their wired
            # distributor address matches the canonical pin in
            # networks.py for the active PRSM_NETWORK. Mismatch
            # = stale env override (e.g., still pinned to v1
            # post-migration).
            wired = getattr(royalty, "distributor_address", None)
            if wired is not None:
                entry["wired_address"] = wired
                try:
                    from prsm.config.networks import (
                        get_network_config, _resolve_network_name,
                    )
                    cfg = get_network_config(_resolve_network_name())
                    canonical = cfg.royalty_distributor
                    if canonical is not None:
                        entry["canonical_address"] = canonical
                        entry["canonical_match"] = (
                            wired.lower() == canonical.lower()
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "canonical-match lookup failed for "
                        "royalty_distributor: %s", exc,
                    )
            subsystems["royalty_distributor"] = entry
        else:
            subsystems["royalty_distributor"] = {
                "available": False, "status": "not_wired",
            }

        # In-memory ring buffer counts. Surfaces the 4 dashboard
        # rings (webhook + slash + heartbeat + distribution) so
        # operators can see at a glance whether buffers are
        # populated + whether persistence is enabled. None when
        # the corresponding feature isn't wired.
        for ring_name, attr in (
            ("webhook_log", "_webhook_log"),
            ("slash_event_log", "_slash_event_log"),
            ("heartbeat_log", "_heartbeat_log"),
            ("distribution_log", "_distribution_log"),
        ):
            ring = getattr(node, attr, None)
            if ring is None:
                if hasattr(node, attr):
                    subsystems[ring_name] = {
                        "available": False, "status": "not_wired",
                    }
                continue
            entry: Dict[str, Any] = {
                "available": True, "status": "ok",
            }
            try:
                entry["count"] = ring.count()
            except Exception as exc:  # noqa: BLE001
                logger.debug("%s.count() raised: %s", ring_name, exc)
            persist_dir = getattr(ring, "_persist_dir", None)
            entry["persisted"] = persist_dir is not None
            _add_persistence_hint(entry, ring_name)
            subsystems[ring_name] = entry

        # Phase 7-storage + Phase 8 client canonical-match probes.
        # Each underlying client (StorageSlashing / Compensation
        # Distributor / KeyDistribution) exposes a `.address`
        # property; we surface wired_address + canonical_address +
        # canonical_match as a config-correctness signal for
        # operators on multi-network deployments.
        def _client_canonical_subsystem(
            name: str, client_attr: str, networks_field: str,
        ):
            client = getattr(node, client_attr, None)
            if client is None:
                if hasattr(node, client_attr):
                    subsystems[name] = {
                        "available": False, "status": "not_wired",
                    }
                return
            entry = {"available": True, "status": "ok"}
            # Sprint 142 fix: read CONTRACT address, not signer
            # address. `.address` on StorageSlashingClient /
            # CompensationDistributorClient returns the signing
            # account (operator wallet), not the contract.
            # Sprint 83 used the wrong attribute and produced
            # garbage canonical-match output (every node showed
            # MISMATCH because operator wallet != contract addr).
            # Try .contract_address first (matches FTNS ledger
            # convention); fall back to .address only if absent.
            wired = (
                getattr(client, "contract_address", None)
                or getattr(client, "address", None)
            )
            if wired is not None:
                entry["wired_address"] = wired
                try:
                    from prsm.config.networks import (
                        get_network_config, _resolve_network_name,
                    )
                    cfg = get_network_config(_resolve_network_name())
                    canonical = getattr(cfg, networks_field, None)
                    if canonical is not None:
                        entry["canonical_address"] = canonical
                        entry["canonical_match"] = (
                            wired.lower() == canonical.lower()
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "canonical-match lookup failed for %s: %s",
                        name, exc,
                    )
            subsystems[name] = entry

        _client_canonical_subsystem(
            "storage_slashing",
            "_storage_slashing_client",
            "storage_slashing",
        )
        _client_canonical_subsystem(
            "compensation_distributor",
            "_compensation_distributor_client",
            "compensation_distributor",
        )
        _client_canonical_subsystem(
            "key_distribution",
            "_key_distribution_client",
            "key_distribution",
        )

        # DRY helper for the 5 daemon subsystems sharing the
        # task-liveness pattern (heartbeat + compensation_scheduler
        # + 3 event watchers). Each daemon has a (daemon_attr,
        # task_attr) pair on Node; the helper renders a uniform
        # subsystem entry from them.
        def _daemon_subsystem(name: str, daemon_attr: str, task_attr: str):
            daemon = getattr(node, daemon_attr, None)
            if daemon is None:
                if hasattr(node, daemon_attr):
                    subsystems[name] = {
                        "available": False, "status": "not_wired",
                    }
                return
            entry = {"available": True, "status": "ok"}
            task = getattr(node, task_attr, None)
            if task is not None:
                try:
                    entry["task_running"] = not task.done()
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "%s task probe raised: %s", name, exc,
                    )
            # Sprint 400 — daemons that adopt the sprint-399
            # tick-age tracking pattern (last_tick_at +
            # last_tick_age_seconds + interval_seconds)
            # automatically surface tick_status in
            # /health/detailed. Thresholds match sprint 392:
            # age < 2× → healthy, 2-5× → degraded, ≥5× or
            # None → stale. Pure-additive: aggregate
            # top-level status NOT modified.
            interval = getattr(daemon, "interval_seconds", None)
            age = getattr(daemon, "last_tick_age_seconds", None)
            if isinstance(interval, (int, float)) and interval > 0 \
                    and hasattr(daemon, "last_tick_age_seconds"):
                try:
                    entry["last_tick_age_seconds"] = age
                    if age is None:
                        tick_status = "stale"
                    elif age < 2 * interval:
                        tick_status = "healthy"
                    elif age < 5 * interval:
                        tick_status = "degraded"
                    else:
                        tick_status = "stale"
                    entry["tick_status"] = tick_status
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "%s tick-age probe raised: %s", name, exc,
                    )
            subsystems[name] = entry

        _daemon_subsystem(
            "compensation_scheduler",
            "_compensation_scheduler", "_compensation_scheduler_task",
        )
        _daemon_subsystem(
            "key_distribution_watcher",
            "_key_distribution_watcher", "_key_distribution_watcher_task",
        )
        _daemon_subsystem(
            "storage_slashing_watcher",
            "_storage_slashing_watcher", "_storage_slashing_watcher_task",
        )
        _daemon_subsystem(
            "compensation_distributor_watcher",
            "_compensation_distributor_watcher",
            "_compensation_distributor_watcher_task",
        )
        _daemon_subsystem(
            "job_reaper",
            "_job_reaper",
            "_job_reaper_task",
        )
        _daemon_subsystem(
            "daemon_watchdog",
            "_daemon_watchdog",
            "_daemon_watchdog_task",
        )

        # HeartbeatScheduler (optional). Same task-liveness pattern
        # as payment_escrow's cleanup_task — operators detect
        # silent crash of the scheduler.
        heartbeat = getattr(node, "_heartbeat_scheduler", None)
        if heartbeat is not None:
            entry = {"available": True, "status": "ok"}
            interval = getattr(
                heartbeat, "interval_seconds", None,
            )
            entry["interval_seconds"] = interval
            hb_task = getattr(node, "_heartbeat_scheduler_task", None)
            if hb_task is not None:
                try:
                    entry["task_running"] = not hb_task.done()
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "heartbeat task probe raised: %s", exc,
                    )
            # Sprint 399 — surface heartbeat-scheduler's
            # last-tick age + status. Catches the silent-
            # economic-failure mode where the task is
            # running but the chain RPC has been failing
            # every tick (no compensation epoch credit).
            # Pure-additive — does NOT modify the
            # aggregate top-level status (operators set
            # their own alert thresholds on tick_status).
            try:
                age = getattr(
                    heartbeat, "last_tick_age_seconds", None,
                )
                entry["last_tick_age_seconds"] = age
                if age is None or not isinstance(interval, (int, float)) or interval <= 0:
                    tick_status = "stale"
                elif age < 2 * interval:
                    tick_status = "healthy"
                elif age < 5 * interval:
                    tick_status = "degraded"
                else:
                    tick_status = "stale"
                entry["tick_status"] = tick_status
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "heartbeat tick-age probe raised: %s", exc,
                )
            subsystems["heartbeat_scheduler"] = entry
        elif hasattr(node, "_heartbeat_scheduler"):
            # Explicitly None means "operator opted out / unwired"
            subsystems["heartbeat_scheduler"] = {
                "available": False, "status": "not_wired",
            }

        # Sprint 329 — bootstrap_discovery subsystem. Wired but
        # degraded/dead → status=degraded so ops alerts on
        # /health/detailed catch it. Not wired → opt-out per
        # the sprint-147 convention. Errors get a separate
        # status so operators see the raise reason.
        disco = getattr(node, "discovery", None)
        # Sprint 329 — guard against the "MagicMock attribute"
        # case from incomplete test fixtures: `node.discovery`
        # may auto-vivify a MagicMock that has a callable
        # `get_bootstrap_status` returning another MagicMock.
        # Treat anything that doesn't yield a real dict shape
        # as not_wired so opt-out semantics hold + we don't
        # falsely flag the node degraded.
        if disco is None or not hasattr(
            disco, "get_bootstrap_status",
        ):
            subsystems["bootstrap_discovery"] = {
                "available": False,
                "status": "not_wired",
            }
        else:
            try:
                bs = disco.get_bootstrap_status()
            except Exception as exc:  # noqa: BLE001
                bs = exc
            if not isinstance(bs, dict):
                # Non-dict return (incomplete test fixture or
                # raise) → treat as not_wired so opt-out
                # semantics hold + the node doesn't get falsely
                # marked degraded.
                if isinstance(bs, Exception):
                    subsystems["bootstrap_discovery"] = {
                        "available": False,
                        "status": "error",
                        "error": str(bs),
                    }
                else:
                    subsystems["bootstrap_discovery"] = {
                        "available": False,
                        "status": "not_wired",
                    }
                bs = None
            if isinstance(bs, dict):
                try:
                    degraded_flag = bool(bs.get("degraded", False))
                    connected = int(bs.get("connected", 0) or 0)
                    client_state = bs.get("client_state", "?")
                    entry = {
                        "available": (
                            connected > 0 and not degraded_flag
                        ),
                        "status": (
                            "degraded" if degraded_flag else "ok"
                        ),
                        "client_state": client_state,
                        "connected": connected,
                        "discovered_peer_count": int(
                            bs.get("discovered_peer_count", 0) or 0
                        ),
                        # Sprint 376 — surface sprint-375
                        # multi-bootstrap fields so ops alerting
                        # can distinguish "primary down, on
                        # fallback" from "ALL bootstraps down."
                        # Defaults to None when the upstream
                        # status dict omits them (pre-sprint-375
                        # discovery objects).
                        "active_url": bs.get("active_url"),
                        "fallback_enabled": bs.get(
                            "bootstrap_fallback_enabled",
                        ),
                    }
                    subsystems["bootstrap_discovery"] = entry
                except Exception as exc:  # noqa: BLE001
                    subsystems["bootstrap_discovery"] = {
                        "available": False,
                        "status": "error",
                        "error": str(exc),
                    }

        # Sprint 342 — federated_learning_orchestrator subsystem.
        # Sprint 343 — generalized to accept a custom probe
        # callable so subsystems with non-list_jobs APIs
        # (content_filter / disclosure_intake / incident_response
        # / $CORP capability registry / upgrade orchestrator)
        # share the same pattern.
        def _orchestrator_subsystem(
            name: str, attr: str,
            probe=None, count_field: str = "record_count",
        ) -> None:
            orch = getattr(node, attr, None)
            if orch is None:
                if hasattr(node, attr):
                    subsystems[name] = {
                        "available": False,
                        "status": "not_wired",
                    }
                return
            try:
                # Default probe: list_jobs() → len(list)
                if probe is None:
                    result = orch.list_jobs()
                    count = len(result) if result is not None else 0
                else:
                    count = probe(orch)
                subsystems[name] = {
                    "available": True,
                    "status": "ok",
                    count_field: count,
                }
            except Exception as exc:  # noqa: BLE001
                subsystems[name] = {
                    "available": False,
                    "status": "error",
                    "error": str(exc),
                }

        # §7 enterprise orchestrators (sprints 308 + 312)
        _orchestrator_subsystem(
            "federated_learning_orchestrator",
            "_federated_learning_orchestrator",
            count_field="jobs_count",
        )
        _orchestrator_subsystem(
            "pipeline_inference_orchestrator",
            "_pipeline_inference_orchestrator",
            count_field="jobs_count",
        )

        # Sprint 343 — five wired stores that lacked /health
        # visibility. Each uses a record-count probe.
        _orchestrator_subsystem(
            "content_filter_store",            # sprint 270
            "_content_filter_store",
            probe=lambda x: x.count(),
        )
        _orchestrator_subsystem(
            "disclosure_intake",               # sprint 300
            "_disclosure_intake",
            probe=lambda x: x.count(),
        )
        _orchestrator_subsystem(
            "incident_response",               # sprint 302
            "_incident_response",
            probe=lambda x: x.count(),
        )
        _orchestrator_subsystem(
            "corp_capability_store",           # sprint 304
            "_corp_capability_store",
            probe=lambda x: len(x.list_issuers()),
        )
        _orchestrator_subsystem(
            "upgrade_orchestrator",            # sprint 303
            "_upgrade_orchestrator",
            probe=lambda x: x.count(),
        )

        # Sprint 582 — trust-stack subsystem (read-only).
        # Surfaces the 4 Phase-1 env-driven kinds (sprints 558-562,
        # 576/577/578) at the REST layer so monitoring + MCP +
        # downstream tools see the same view as `prsm node
        # trust-stack` CLI (sprint 579). NEVER degrades health
        # status — purely informational.
        try:
            _ts_entries: Dict[str, Dict[str, Any]] = {}
            _ts_env = [
                ("trust_stack_kind", "PRSM_PARALLAX_TRUST_STACK_KIND",
                 ("mock", "production"), "mock"),
                ("profile_source_kind",
                 "PRSM_PARALLAX_PROFILE_SOURCE_KIND",
                 ("in_memory", "dht"), "in_memory"),
                ("consensus_submitter_kind",
                 "PRSM_PARALLAX_CONSENSUS_SUBMITTER_KIND",
                 ("logging", "onchain"), "logging"),
                ("chain_executor_kind",
                 "PRSM_PARALLAX_CHAIN_EXECUTOR_KIND",
                 ("stub", "rpc"), "stub"),
            ]
            for slug, envname, valid, default in _ts_env:
                raw = (os.environ.get(envname, "") or "").strip()
                env_val = raw or None
                val = (raw or default).lower()
                if val in valid:
                    if val == default:
                        status = "active_default"
                    else:
                        status = "phase_2_pending"
                else:
                    status = "unknown_fallback"
                    val = default
                _ts_entries[slug] = {
                    "kind": val,
                    "env_value": env_val,
                    "status": status,
                    "default": default,
                }
            subsystems["trust_stack"] = {
                "available": True,
                "status": "ok",
                "components": _ts_entries,
            }
        except Exception as exc:  # noqa: BLE001
            subsystems["trust_stack"] = {
                "available": False, "status": "error",
                "error": str(exc),
            }

        # Sprint 1051 — on-chain settlement rail observability. Read-only snapshot
        # of the live rail (tracked/finalized + quarantined/orphaned commit counts
        # = the funds-in-flight signal). An off-by-config rail is opt-out, NOT an
        # error → status 'not_wired' (per the sprint-147 convention below), so it
        # never drags the core aggregate. Not added to `core`.
        try:
            from prsm.settlement.client_wiring import get_settlement_status
            _settle = get_settlement_status(
                getattr(node, "_onchain_settlement_client", None))
            subsystems["settlement"] = {
                "available": bool(_settle.get("enabled")),
                "status": "ok" if _settle.get("enabled") else "not_wired",
                **_settle,
            }
        except Exception as exc:  # noqa: BLE001
            subsystems["settlement"] = {
                "available": False, "status": "error", "error": str(exc)}

        # Sprint 1090 — attestation-collateral auto-refresh status (sp1081-1089): per
        # cached item present? fresh (within nextUpdate)? age? — so an operator can SEE
        # that revocation + TCB-recency enforcement is being kept current. Opt-out/off
        # is 'not_wired' (per the sprint-147 convention); never in the core aggregate.
        try:
            from prsm.compute.inference.collateral_refresh import (
                collateral_refresh_status)
            _collat = collateral_refresh_status()
            _items = _collat.get("items") or {}
            _stale = [k for k, v in _items.items() if v.get("fresh") is False]
            subsystems["attestation_collateral"] = {
                "available": bool(_collat.get("enabled")),
                "status": ("degraded" if _stale else
                           "ok" if _collat.get("enabled") else "not_wired"),
                "stale_items": _stale,
                **_collat,
            }
        except Exception as exc:  # noqa: BLE001
            subsystems["attestation_collateral"] = {
                "available": False, "status": "error", "error": str(exc)}

        # Aggregate status.
        # Sprint 147 — `not_wired` / `disabled` is operator opt-out,
        # not a degradation. Only count an optional subsystem as
        # degraded if it's wired but reporting unavailable for a
        # genuine reason (status=error/crashed/uninitialized).
        core = ["ftns_ledger", "payment_escrow"]
        # Sprint 329 — bootstrap_discovery joins job_history /
        # royalty_distributor as optional. Degraded bootstrap
        # alone flips top-level to "degraded" (not unhealthy).
        # Sprint 342 — fl + pipeline orchestrators join as optional.
        # Sprint 343 — content_filter/disclosure/incident/$CORP/
        # upgrade orchestrators join as optional too.
        optional = [
            "job_history", "royalty_distributor",
            "bootstrap_discovery",
            "federated_learning_orchestrator",
            "pipeline_inference_orchestrator",
            "content_filter_store",
            "disclosure_intake",
            "incident_response",
            "corp_capability_store",
            "upgrade_orchestrator",
        ]
        _OPT_OUT_STATUSES = ("not_wired", "disabled")
        core_ok = all(
            subsystems[s]["available"] for s in core
        )

        def _optional_ok(name: str) -> bool:
            sub = subsystems[name]
            if sub.get("available"):
                return True
            return sub.get("status") in _OPT_OUT_STATUSES

        all_optional_ok = all(_optional_ok(s) for s in optional)
        if not core_ok:
            top_status = "unhealthy"
        elif not all_optional_ok:
            top_status = "degraded"
        else:
            top_status = "healthy"

        # sp963 — surface the sp960 network-exposure auth posture so operators
        # can re-check it any time (not just at the startup log line). A bind to
        # a non-loopback interface with no PRSM_NODE_API_KEY leaves the protected
        # money endpoints unauthenticated.
        security_posture: Dict[str, Any] = {"level": "ok", "message": ""}
        try:
            from prsm.node.node import assess_public_bind_auth_posture
            listen_host = getattr(getattr(node, "config", None), "listen_host", None)
            api_key_set = bool(os.environ.get("PRSM_NODE_API_KEY", "").strip())
            level, message = assess_public_bind_auth_posture(
                listen_host=listen_host, api_key_present=api_key_set,
            )
            security_posture = {
                "level": level,
                "listen_host": listen_host,
                "api_key_set": api_key_set,
                "message": message,
            }
        except Exception as exc:  # noqa: BLE001 — never break health on this
            logger.debug("security_posture probe failed: %s", exc)

        return {
            "status": top_status,
            "node_id": (
                node.identity.node_id if node.identity else "unknown"
            ),
            "subsystems": subsystems,
            "security_posture": security_posture,
        }

    # ── Batch Settlement Endpoints ─────────────────────────────────

    @app.get("/settlement/stats")
    async def settlement_stats() -> Dict[str, Any]:
        """Get batch settlement queue stats."""
        if not hasattr(node, '_batch_settlement') or not node._batch_settlement:
            return {"enabled": False}
        stats = node._batch_settlement.get_stats()
        stats["enabled"] = True
        return stats

    @app.get("/settlement/pending")
    async def settlement_pending() -> Dict[str, Any]:
        """List pending (un-settled) on-chain transfers."""
        if not hasattr(node, '_batch_settlement') or not node._batch_settlement:
            return {"pending": [], "count": 0}
        pending = node._batch_settlement.get_pending()
        return {"pending": pending, "count": len(pending)}

    @app.post("/settlement/flush")
    async def settlement_flush() -> Dict[str, Any]:
        """Manually trigger batch settlement (flush all pending transfers)."""
        if not hasattr(node, '_batch_settlement') or not node._batch_settlement:
            raise HTTPException(status_code=503, detail="Batch settlement not initialized")
        result = await node._batch_settlement.flush()
        return {
            "settled_count": result.settled_count,
            "total_amount": result.total_amount,
            "net_transfers": result.net_transfers,
            "tx_hashes": result.tx_hashes,
            "errors": result.errors,
            "duration_seconds": result.duration_seconds,
        }

    @app.get("/settlement/history")
    async def settlement_history(limit: int = 10) -> Dict[str, Any]:
        """Get recent settlement history."""
        if not hasattr(node, '_batch_settlement') or not node._batch_settlement:
            return {"history": [], "count": 0}
        history = node._batch_settlement.get_history(limit=limit)
        return {"history": history, "count": len(history)}


    # ── Staking Endpoints ─────────────────────────────────────────

    class StakeRequest(BaseModel):
        """Request body for staking FTNS tokens."""
        # Sprint 201 — `allow_inf_nan=False` closes the residual
        # Infinity-bypass on Pydantic gt=0 (NaN was already rejected
        # by gt=0; Infinity passed because `inf > 0` is True).
        amount: float = Field(
            ..., gt=0, le=1e12, allow_inf_nan=False,
            description="Amount of FTNS to stake",
        )
        stake_type: str = Field(default="general", description="Type of staking: governance, validation, compute, storage, liquidity, general")
        metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata for the stake")
        lock_period_days: Optional[int] = Field(
            default=None,
            description=(
                "Optional utility-benefit lock (30/90/365 days). Locking "
                "confers service discounts + priority access (no token "
                "yield); the stake cannot be unstaked until the lock "
                "expires. None = no lock, no benefit."
            ),
        )

    class StakeResponse(BaseModel):
        """Response model for a stake operation."""
        stake_id: str
        user_id: str
        amount: float
        stake_type: str
        status: str
        staked_at: str
        rewards_earned: float = 0.0

    class UnstakeRequest(BaseModel):
        """Request body for unstaking FTNS tokens."""
        stake_id: str = Field(..., description="ID of the stake to unstake")
        amount: Optional[float] = Field(
            default=None, gt=0, le=1e12, allow_inf_nan=False,
            description="Amount to unstake (None = full stake)",
        )

    class UnstakeResponse(BaseModel):
        """Response model for an unstake operation."""
        request_id: str
        stake_id: str
        user_id: str
        amount: float
        requested_at: str
        available_at: str
        status: str

    class StakingStatusResponse(BaseModel):
        """Response model for staking status."""
        user_id: str
        total_staked: float
        active_stakes: List[Dict[str, Any]]
        pending_unstake_requests: List[Dict[str, Any]]
        total_rewards_earned: float
        total_rewards_claimed: float

    class ClaimRewardsResponse(BaseModel):
        """Response model for claiming rewards."""
        user_id: str
        total_rewards_claimed: float
        stakes_processed: int

    @app.post("/staking/stake", response_model=StakeResponse, tags=["staking"])
    async def stake_tokens(req: StakeRequest) -> StakeResponse:
        """
        Stake FTNS tokens.
        
        Stakes the specified amount of FTNS tokens for the node's identity.
        The tokens will be locked and start earning rewards based on the
        configured annual reward rate.
        
        Args:
            req: StakeRequest containing amount, stake_type, and optional metadata
            
        Returns:
            StakeResponse with the created stake details
            
        Raises:
            HTTPException 503: If staking manager not initialized
            HTTPException 400: If stake validation fails
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")
        
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")
        
        # Validate stake_type
        from prsm.economy.tokenomics.staking_manager import StakeType
        try:
            stake_type = StakeType(req.stake_type.lower())
        except ValueError:
            valid_types = [t.value for t in StakeType]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stake_type: {req.stake_type}. Valid types: {valid_types}"
            )
        
        try:
            from decimal import Decimal
            stake = await node.staking_manager.stake(
                user_id=node.identity.node_id,
                amount=Decimal(str(req.amount)),
                stake_type=stake_type,
                metadata=req.metadata,
                lock_period_days=req.lock_period_days,
            )
            
            return StakeResponse(
                stake_id=stake.stake_id,
                user_id=stake.user_id,
                amount=float(stake.amount),
                stake_type=stake.stake_type.value,
                status=stake.status.value,
                staked_at=stake.staked_at.isoformat(),
                rewards_earned=float(stake.rewards_earned)
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error staking tokens: {e}")
            raise HTTPException(status_code=500, detail=f"Staking failed: {str(e)}")

    @app.post("/staking/unstake", response_model=UnstakeResponse, tags=["staking"])
    async def unstake_tokens(req: UnstakeRequest) -> UnstakeResponse:
        """
        Request to unstake FTNS tokens.
        
        Creates an unstake request that will be available for withdrawal
        after the configured unstaking period (default: 7 days).
        
        Args:
            req: UnstakeRequest containing stake_id and optional amount
            
        Returns:
            UnstakeResponse with the unstake request details
            
        Raises:
            HTTPException 503: If staking manager not initialized
            HTTPException 400: If unstake validation fails
            HTTPException 404: If stake not found
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")
        
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")
        
        try:
            from decimal import Decimal
            amount = Decimal(str(req.amount)) if req.amount else None
            
            unstake_request = await node.staking_manager.unstake(
                user_id=node.identity.node_id,
                stake_id=req.stake_id,
                amount=amount
            )
            
            return UnstakeResponse(
                request_id=unstake_request.request_id,
                stake_id=unstake_request.stake_id,
                user_id=unstake_request.user_id,
                amount=float(unstake_request.amount),
                requested_at=unstake_request.requested_at.isoformat(),
                available_at=unstake_request.available_at.isoformat(),
                status=unstake_request.status.value
            )
        except ValueError as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error unstaking tokens: {e}")
            raise HTTPException(status_code=500, detail=f"Unstake failed: {str(e)}")

    @app.get("/staking/benefits/{user_id}", tags=["staking"])
    async def get_staking_benefits(user_id: str) -> Dict[str, Any]:
        """Sprint 906 — the staking utility benefits currently in effect
        for a user.

        Returns the best (highest) active lock tier's service discount +
        priority boost. This is the single source of truth the pricing
        layer (network-fee discount) and dispatch layer (priority access)
        consume. A user with no active, sufficiently-funded, still-locked
        stake gets the inert "none" tier (0% discount, +0 priority).

        Staking pays NO token yield (sp904); the value is the utility
        benefits reported here.
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")
        try:
            benefits = await node.staking_manager.get_user_benefits(user_id)
            return {
                "user_id": user_id,
                "yield_model": "utility_only",
                **benefits.to_dict(),
                "service_discount_pct": round(benefits.discount_fraction * 100, 4),
                "priority_boost_pct": round(benefits.priority_boost * 100, 4),
            }
        except Exception as e:
            logger.error(f"Error getting staking benefits: {e}")
            raise HTTPException(status_code=500, detail=f"Benefits lookup failed: {str(e)}")

    @app.get("/staking/status", response_model=StakingStatusResponse, tags=["staking"])
    async def get_staking_status() -> StakingStatusResponse:
        """
        Get staking status for the current node identity.
        
        Returns information about all active stakes, pending unstake requests,
        and reward totals for the node's identity.
        
        Returns:
            StakingStatusResponse with comprehensive staking information
            
        Raises:
            HTTPException 503: If staking manager not initialized
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")
        
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")
        
        user_id = node.identity.node_id
        
        # Get all stakes for the user
        stakes = await node.staking_manager.get_user_stakes(user_id)
        
        # Get pending unstake requests
        pending_requests = await node.staking_manager.get_pending_unstake_requests(user_id)
        
        # Calculate totals
        total_staked = sum(float(s.amount) for s in stakes if s.status.value == "active")
        total_rewards_earned = sum(float(s.rewards_earned) for s in stakes)
        total_rewards_claimed = sum(float(s.rewards_claimed) for s in stakes)
        
        return StakingStatusResponse(
            user_id=user_id,
            total_staked=total_staked,
            active_stakes=[
                {
                    "stake_id": s.stake_id,
                    "amount": float(s.amount),
                    "stake_type": s.stake_type.value,
                    "status": s.status.value,
                    "staked_at": s.staked_at.isoformat(),
                    "rewards_earned": float(s.rewards_earned),
                    "rewards_claimed": float(s.rewards_claimed)
                }
                for s in stakes
            ],
            pending_unstake_requests=[
                {
                    "request_id": r.request_id,
                    "stake_id": r.stake_id,
                    "amount": float(r.amount),
                    "requested_at": r.requested_at.isoformat(),
                    "available_at": r.available_at.isoformat(),
                    "status": r.status.value
                }
                for r in pending_requests
            ],
            total_rewards_earned=total_rewards_earned,
            total_rewards_claimed=total_rewards_claimed
        )

    @app.post("/staking/claim-rewards", response_model=ClaimRewardsResponse, tags=["staking"])
    async def claim_staking_rewards(stake_id: Optional[str] = None) -> ClaimRewardsResponse:
        """
        Claim accumulated staking rewards.
        
        Claims all pending rewards for the node's stakes. If stake_id is provided,
        only claims rewards for that specific stake.
        
        Args:
            stake_id: Optional specific stake ID to claim rewards from
            
        Returns:
            ClaimRewardsResponse with total rewards claimed and stakes processed
            
        Raises:
            HTTPException 503: If staking manager not initialized
            HTTPException 400: If claim validation fails
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")
        
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")
        
        try:
            total_rewards = await node.staking_manager.claim_rewards(
                user_id=node.identity.node_id,
                stake_id=stake_id
            )
            
            # Count stakes processed
            stakes = await node.staking_manager.get_user_stakes(node.identity.node_id)
            stakes_processed = len(stakes) if not stake_id else 1
            
            return ClaimRewardsResponse(
                user_id=node.identity.node_id,
                total_rewards_claimed=float(total_rewards),
                stakes_processed=stakes_processed
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error claiming rewards: {e}")
            raise HTTPException(status_code=500, detail=f"Claim rewards failed: {str(e)}")

    @app.get("/staking/stakes/{stake_id}", tags=["staking"])
    async def get_stake(stake_id: str) -> Dict[str, Any]:
        """
        Get details of a specific stake.
        
        Args:
            stake_id: The ID of the stake to retrieve
            
        Returns:
            Detailed stake information
            
        Raises:
            HTTPException 503: If staking manager not initialized
            HTTPException 404: If stake not found
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")

        # Sprint 182 — get_stake() raises ValueError on malformed UUID
        # (sqlalchemy / underlying ORM rejects "nonexistent" as a UUID
        # before the None-not-found path). Map to 404 so operators see
        # "not found" instead of a 500.
        try:
            stake = await node.staking_manager.get_stake(stake_id)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=404,
                detail=f"Stake not found (or stake_id malformed): {stake_id!r}",
            )
        if not stake:
            raise HTTPException(status_code=404, detail="Stake not found")
        
        return {
            "stake_id": stake.stake_id,
            "user_id": stake.user_id,
            "amount": float(stake.amount),
            "stake_type": stake.stake_type.value,
            "status": stake.status.value,
            "staked_at": stake.staked_at.isoformat(),
            "rewards_earned": float(stake.rewards_earned),
            "rewards_claimed": float(stake.rewards_claimed),
            "last_reward_calculation": stake.last_reward_calculation.isoformat() if stake.last_reward_calculation else None,
            "lock_reason": stake.lock_reason,
            "metadata": stake.metadata
        }

    @app.get("/staking/unstake-requests/{request_id}", tags=["staking"])
    async def get_unstake_request(request_id: str) -> Dict[str, Any]:
        """
        Get details of a specific unstake request.
        
        Args:
            request_id: The ID of the unstake request to retrieve
            
        Returns:
            Detailed unstake request information
            
        Raises:
            HTTPException 503: If staking manager not initialized
            HTTPException 404: If request not found
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")

        # Sprint 182 — same malformed-UUID → 404 mapping as get_stake.
        try:
            request = await node.staking_manager.get_unstake_request(request_id)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Unstake request not found (or request_id "
                    f"malformed): {request_id!r}"
                ),
            )
        if not request:
            raise HTTPException(status_code=404, detail="Unstake request not found")
        
        return {
            "request_id": request.request_id,
            "stake_id": request.stake_id,
            "user_id": request.user_id,
            "amount": float(request.amount),
            "requested_at": request.requested_at.isoformat(),
            "available_at": request.available_at.isoformat(),
            "status": request.status.value,
            "completed_at": request.completed_at.isoformat() if request.completed_at else None,
            "cancellation_reason": request.cancellation_reason,
            "is_available": request.is_available
        }

    @app.post("/staking/withdraw/{request_id}", tags=["staking"])
    async def withdraw_unstaked_tokens(request_id: str) -> Dict[str, Any]:
        """
        Withdraw unstaked tokens after the unstaking period.
        
        Completes an unstake request and returns the tokens to the user's
        available balance.
        
        Args:
            request_id: The ID of the unstake request to withdraw
            
        Returns:
            Withdrawal confirmation with amount
            
        Raises:
            HTTPException 503: If staking manager not initialized
            HTTPException 400: If withdrawal validation fails
            HTTPException 404: If request not found
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")
        
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")
        
        try:
            success, amount = await node.staking_manager.withdraw(
                user_id=node.identity.node_id,
                request_id=request_id
            )
            
            return {
                "request_id": request_id,
                "success": success,
                "amount_withdrawn": float(amount)
            }
        except ValueError as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error withdrawing tokens: {e}")
            raise HTTPException(status_code=500, detail=f"Withdrawal failed: {str(e)}")

    @app.post("/staking/cancel-unstake/{request_id}", tags=["staking"])
    async def cancel_unstake_request(request_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancel a pending unstake request.
        
        Cancels an unstake request and restores the tokens to active staking.
        
        Args:
            request_id: The ID of the unstake request to cancel
            reason: Optional reason for cancellation
            
        Returns:
            Cancellation confirmation
            
        Raises:
            HTTPException 503: If staking manager not initialized
            HTTPException 400: If cancellation validation fails
            HTTPException 404: If request not found
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")
        
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")
        
        try:
            success = await node.staking_manager.cancel_unstake(
                user_id=node.identity.node_id,
                request_id=request_id,
                reason=reason
            )
            
            return {
                "request_id": request_id,
                "cancelled": success,
                "reason": reason
            }
        except ValueError as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error cancelling unstake: {e}")
            raise HTTPException(status_code=500, detail=f"Cancellation failed: {str(e)}")

    # ── Settler Registry (Phase 6: L2-style staking for batch security) ──

    @app.post("/settler/register", tags=["settler", "phase6"])
    async def register_settler(
        settler_id: str,
        address: str,
        bond_amount: float,
    ) -> Dict[str, Any]:
        """
        Register as a batch settler with staked bond.
        
        Settlers stake FTNS to earn the right to approve batch settlements.
        Requires minimum bond (default: 10K FTNS).
        
        Args:
            settler_id: Unique identifier (e.g., node ID)
            address: Ethereum address for on-chain operations
            bond_amount: FTNS to stake as bond
        """
        if not hasattr(node, "_settler_registry") or not node._settler_registry:
            raise HTTPException(503, "Settler registry not initialized")

        # Sprint 200 — guard NaN/Infinity bond_amount.
        import math
        if not math.isfinite(bond_amount):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"bond_amount must be a finite positive number; "
                    f"got {bond_amount!r}."
                ),
            )

        try:
            settler = await node._settler_registry.register_settler(
                settler_id=settler_id,
                address=address,
                bond_amount=bond_amount,
            )
            return {
                "settler_id": settler.settler_id,
                "address": settler.address,
                "bond_amount": settler.bond_amount,
                "status": settler.status.value,
                "staked_at": settler.staked_at.isoformat(),
            }
        except ValueError as e:
            raise HTTPException(400, str(e))

    # Sprint 534 F61 fix: route order matters in FastAPI — the
    # parameterized `/settler/{settler_id}` was declared BEFORE
    # the literal `/settler/stats`, causing every GET on /stats
    # to be matched as settler_id="stats" → 404 "Settler stats
    # not found". Moving the parameterized route to come AFTER
    # all literal 2-segment routes fixes the routing precedence.
    # FastAPI matches first-declared-wins so literals must come
    # before catch-all parameterized paths sharing the prefix.

    @app.get("/settler/list/active", tags=["settler"])
    async def list_active_settlers() -> List[Dict[str, Any]]:
        """List all active settlers."""
        if not hasattr(node, "_settler_registry") or not node._settler_registry:
            raise HTTPException(503, "Settler registry not initialized")
        
        return [
            {
                "settler_id": s.settler_id,
                "address": s.address,
                "bond_amount": s.bond_amount,
                "total_settled": s.total_settled,
            }
            for s in node._settler_registry.list_active_settlers()
        ]

    @app.post("/settler/unbond", tags=["settler"])
    async def unbond_settler(settler_id: str) -> Dict[str, Any]:
        """Initiate unbonding for a settler (30-day lock period)."""
        if not hasattr(node, "_settler_registry") or not node._settler_registry:
            raise HTTPException(503, "Settler registry not initialized")
        
        try:
            unbond_at = await node._settler_registry.unbond_settler(settler_id)
            return {
                "settler_id": settler_id,
                "status": "unbonding",
                "unbond_at": unbond_at.isoformat() if unbond_at else None,
            }
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/settler/batch/sign", tags=["settler"])
    async def sign_batch(
        batch_id: str,
        settler_id: str,
        signature: str,
    ) -> Dict[str, Any]:
        """
        Sign a pending batch for multi-sig approval.
        
        When signature threshold (default: 3) is reached,
        the batch is approved for on-chain settlement.
        """
        if not hasattr(node, "_settler_registry") or not node._settler_registry:
            raise HTTPException(503, "Settler registry not initialized")
        
        try:
            sig = await node._settler_registry.sign_batch(
                batch_id=batch_id,
                settler_id=settler_id,
                signature=signature,
            )
            batch = node._settler_registry.get_pending_batch(batch_id)
            return {
                "batch_id": batch_id,
                "settler_id": settler_id,
                "signed_at": sig.signed_at.isoformat(),
                "signature_count": batch.signature_count if batch else 1,
                "threshold": node._settler_registry.settlement_threshold,
                "approved": node._settler_registry.is_batch_approved(batch_id),
            }
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/settler/batch/pending", tags=["settler"])
    async def list_pending_batches() -> List[Dict[str, Any]]:
        """List all batches awaiting multi-sig approval."""
        if not hasattr(node, "_settler_registry") or not node._settler_registry:
            raise HTTPException(503, "Settler registry not initialized")
        
        return [
            {
                "batch_id": b.batch_id,
                "batch_hash": b.batch_hash,
                "transfer_count": len(b.transfers),
                "total_amount": b.total_amount,
                "signature_count": b.signature_count,
                "approved": b.signature_count >= node._settler_registry.settlement_threshold,
                "created_at": b.created_at.isoformat(),
            }
            for b in node._settler_registry.list_pending_batches()
            if not b.settled
        ]

    @app.get("/settler/ledger/export", tags=["settler"])
    async def export_ledger() -> Dict[str, Any]:
        """
        Export local ledger state for public audit (Challenge system).
        
        Enables anyone to compare local state against on-chain settlement.
        """
        if not hasattr(node, "_settler_registry") or not node._settler_registry:
            raise HTTPException(503, "Settler registry not initialized")
        
        # Gather ledger data
        ledger_data = {}
        if hasattr(node, "ledger"):
            ledger_data = {
                "balances": dict(node.ledger._balances) if hasattr(node.ledger, "_balances") else {},
                "total_supply": node.ledger.total_supply if hasattr(node.ledger, "total_supply") else 0,
            }
        
        return await node._settler_registry.export_ledger(ledger_data)

    @app.post("/settler/slash/propose", tags=["settler"])
    async def propose_slash(
        settler_id: str,
        slash_amount: float,
        reason: str,
        proposer_id: str,
    ) -> Dict[str, Any]:
        """
        Propose to slash a settler's bond.
        
        Creates a governance proposal that must be approved before execution.
        """
        if not hasattr(node, "_settler_registry") or not node._settler_registry:
            raise HTTPException(503, "Settler registry not initialized")

        # Sprint 200 — guard NaN/Infinity slash_amount.
        import math
        if not math.isfinite(slash_amount):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"slash_amount must be a finite positive number; "
                    f"got {slash_amount!r}."
                ),
            )

        try:
            proposal = await node._settler_registry.propose_slash(
                settler_id=settler_id,
                slash_amount=slash_amount,
                reason=reason,
                evidence={},
                proposer_id=proposer_id,
            )
            return {
                "proposal_id": proposal.proposal_id,
                "settler_id": settler_id,
                "slash_amount": slash_amount,
                "reason": reason,
                "status": "pending_vote",
            }
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/settler/stats", tags=["settler"])
    async def get_settler_stats() -> Dict[str, Any]:
        """Get settler registry statistics."""
        if not hasattr(node, "_settler_registry") or not node._settler_registry:
            raise HTTPException(503, "Settler registry not initialized")

        return node._settler_registry.get_stats()

    # Sprint 534 F61 fix: parameterized route DECLARED LAST so all
    # literal 2-segment /settler/<word> routes win route-matching.
    @app.get("/settler/{settler_id}", tags=["settler"])
    async def get_settler(settler_id: str) -> Dict[str, Any]:
        """Get details of a specific settler."""
        if not hasattr(node, "_settler_registry") or not node._settler_registry:
            raise HTTPException(503, "Settler registry not initialized")

        settler = node._settler_registry.get_settler(settler_id)
        if not settler:
            raise HTTPException(404, f"Settler {settler_id} not found")

        return {
            "settler_id": settler.settler_id,
            "address": settler.address,
            "bond_amount": settler.bond_amount,
            "status": settler.status.value,
            "can_settle": settler.can_settle,
            "total_settled": settler.total_settled,
            "slashed_amount": settler.slashed_amount,
        }


    # ── Storage endpoints ────────────────────────────────────────

    @app.get("/storage/stats")
    async def get_storage_stats() -> Dict[str, Any]:
        """Get storage provider statistics."""
        if not node.storage_provider:
            return {
                "available": False,
                "pledged_gb": 0,
                "used_gb": 0,
                "pinned_count": 0,
                "message": "Storage provider not initialized"
            }
        return node.storage_provider.get_stats()

    # Sprint 267 — surface per-content challenge stats for the
    # storage operator's pinned set. Closes the "is my pinned
    # data still being challenged?" triage gap.
    # Sprint 834 — F29 fix: `prsm storage pin <cid>` CLI hit
    # /api/v1/storage/{cid}/pin (legacy storage_api router,
    # never mounted; in fact storage_api.py doesn't exist).
    # Add inline POST that wires the existing
    # StorageProvider.pin_content(cid) → ContentStore.
    #
    # Sprint 835 (F30 fix): the operator-facing CID is the
    # BitTorrent infohash (40-char SHA-1) produced by the
    # ContentPublisher. But StorageProvider.pin_content expects
    # the ContentStore identifier (algo-prefixed SHA-256 hex,
    # 66 chars). Pre-835 every uploaded CID failed the
    # ContentHash.from_hex parse and pin_content silently
    # returned False — sprint 834's 404 path always fired
    # against the uploader's own content. Sprint 835 looks up
    # the SHA-256 from node.content_uploader.uploaded_content
    # then prefixes "01" (AlgorithmID.SHA256) before passing
    # to pin_content. Pinning is now reachable for content
    # this node uploaded itself.
    @app.post("/content/{cid}/pin", tags=["storage"])
    async def pin_content(cid: str) -> Dict[str, Any]:
        sp = getattr(node, "storage_provider", None)
        if sp is None:
            raise HTTPException(
                status_code=503,
                detail="Storage provider not initialized.",
            )

        # Sprint 835 — for content this node uploaded itself,
        # the operator-facing CID (torrent infohash) won't pass
        # ContentStore.exists_local because the manifest cache
        # is only populated for sharded-download content (sprint
        # 263 design). But we KNOW the content is locally
        # available — content_uploader tracks it. Short-circuit
        # the StorageProvider.pin_content path: record the pin
        # directly with the size from UploadedContent. Pinning
        # remains GC-protected by the same pinned_content dict
        # the rest of the StorageProvider machinery reads.
        uploader = getattr(node, "content_uploader", None)
        entry = None
        if uploader is not None:
            entry = getattr(
                uploader, "uploaded_content", {},
            ).get(cid)

        if entry is not None:
            from prsm.node.storage_provider import PinnedContent
            sp.pinned_content[cid] = PinnedContent(
                cid=cid, size_bytes=entry.size_bytes,
            )
            return {
                "pinned": True,
                "cid": cid,
                "size_bytes": entry.size_bytes,
            }

        # Content this node didn't upload itself — defer to
        # the legacy pin_content path. Will return False (404)
        # for content not present in the ContentStore manifest
        # cache.
        try:
            ok = await sp.pin_content(cid)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=f"pin_content raised: {exc}",
            )
        if not ok:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"CID {cid} not present locally — upload "
                    "or retrieve before pinning."
                ),
            )
        size = sp.pinned_content.get(cid)
        return {
            "pinned": True,
            "cid": cid,
            "size_bytes": size.size_bytes if size else 0,
        }

    @app.get("/storage/pinned-stats", tags=["storage"])
    async def get_storage_pinned_stats() -> Dict[str, Any]:
        """Per-pinned-content storage statistics: size, pinned_at,
        last_verified, successful/failed challenge counts."""
        sp = getattr(node, "storage_provider", None)
        if sp is None:
            raise HTTPException(
                status_code=503,
                detail="Storage provider not initialized.",
            )
        try:
            pinned = sp.get_pinned_content_stats()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=(
                    f"get_pinned_content_stats() raised: {exc}"
                ),
            )
        return {"pinned": pinned, "count": len(pinned)}

    # Sprint 267 — surface cross-provider reputation + challenge
    # stats so operators can see which OTHER providers in the
    # network are reliable, before choosing seeding partners.
    @app.get("/storage/provider-reputations", tags=["storage"])
    async def get_storage_provider_reputations() -> Dict[str, Any]:
        """Cross-provider reputation + challenge stats: per-
        provider reputation score + (total / successful / failed /
        expired) challenge counts."""
        sp = getattr(node, "storage_provider", None)
        if sp is None:
            raise HTTPException(
                status_code=503,
                detail="Storage provider not initialized.",
            )
        try:
            providers = sp.get_provider_stats_summary()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=(
                    f"get_provider_stats_summary() raised: {exc}"
                ),
            )
        return {"providers": providers, "count": len(providers)}

    # ── Compute endpoints ────────────────────────────────────────

    @app.get("/compute/stats", tags=["compute"])
    async def get_compute_stats() -> Dict[str, Any]:
        """Get compute provider statistics."""
        if not node.compute_provider:
            return {
                "available": False,
                "jobs_completed": 0,
                "jobs_queued": 0,
                "message": "Compute provider not initialized"
            }
        return node.compute_provider.get_stats()

    # Sprint 250 — paginated enumeration of all stored receipts
    # for audit + operator-side review. Optional model_id filter.
    @app.get("/compute/receipts", tags=["compute"])
    async def list_inference_receipts(
        limit: int = 50,
        offset: int = 0,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Paginated list of stored InferenceReceipts (newest
        first). Backed by ReceiptStore.list()."""
        if limit < 1 or limit > 1000:
            raise HTTPException(
                status_code=422,
                detail=f"limit must be in [1, 1000]; got {limit}",
            )
        if offset < 0:
            raise HTTPException(
                status_code=422,
                detail=f"offset must be >= 0; got {offset}",
            )
        store = getattr(node, "_receipt_store", None)
        if store is None:
            raise HTTPException(
                status_code=503,
                detail="Receipt store not initialized.",
            )
        entries = store.list(
            offset=offset, limit=limit, model_id=model_id,
        )
        return {
            "receipts": entries,
            "total": store.count(),
            "offset": offset,
            "limit": limit,
        }

    # Sprint 242 — post-hoc receipt lookup. /compute/inference
    # persists every signed receipt to node._receipt_store (when
    # wired). Audit-friendly: end-users + auditors can fetch a
    # receipt by job_id after the fact.
    @app.get("/compute/receipt/{job_id}", tags=["compute"])
    async def get_inference_receipt(job_id: str) -> Dict[str, Any]:
        """Return the signed InferenceReceipt for a job_id."""
        store = getattr(node, "_receipt_store", None)
        if store is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Receipt store not initialized. "
                    "PRSM_RECEIPT_STORE_DIR can opt in to "
                    "filesystem persistence."
                ),
            )
        receipt = store.get(job_id)
        if receipt is None:
            raise HTTPException(
                status_code=404,
                detail=f"No receipt for job_id={job_id!r}",
            )
        return receipt

    # Sprint 241 — surface the node's own Ed25519 public key so
    # end-users can verify signed InferenceReceipts. Returns
    # node_id + public_key_b64 for cross-node verification flows.
    @app.get("/node/identity/pubkey", tags=["identity"])
    async def get_node_pubkey() -> Dict[str, Any]:
        """Return this node's Ed25519 public key (base64) for
        receipt verification."""
        ident = getattr(node, "identity", None)
        if ident is None:
            raise HTTPException(
                status_code=503,
                detail="Node identity not initialized.",
            )
        return {
            "node_id": ident.node_id,
            "public_key_b64": ident.public_key_b64,
        }

    # Sprint 235 — surface the inference executor's registered
    # model_ids so end-users running prsm_inference can discover
    # what's available without reading the tool description's
    # hard-coded sample list.
    @app.get("/compute/models", tags=["compute"])
    async def get_compute_models() -> Dict[str, Any]:
        """List inference model_ids the executor accepts."""
        executor = getattr(node, "inference_executor", None)
        if executor is None:
            # Sprint 535 F63 fix: actionable hint pointing to the
            # mock executor for local dogfood. Real executor wiring
            # is daemon-side compute provider config (sprint 235+).
            raise HTTPException(
                status_code=503,
                detail=(
                    "Inference executor not initialized on this "
                    "node — set PRSM_INFERENCE_EXECUTOR=mock for "
                    "local testing, or wire a real executor via "
                    "compute_provider config."
                ),
            )
        try:
            models = list(executor.supported_models())
        except Exception as e:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=f"supported_models() failed: {e}",
            )
        return {"models": models, "count": len(models)}

    # ── Resource Configuration Endpoints ─────────────────────────────────

    @app.get("/node/resources", response_model=ResourceConfigResponse, tags=["resources"])
    async def get_resources() -> ResourceConfigResponse:
        """Get current node resource configuration.
        
        Returns the current resource allocation settings including:
        - CPU and memory allocation percentages
        - Storage pledge and availability
        - GPU allocation (if available)
        - Bandwidth limits
        - Active hours/days schedule
        - Computed effective values
        """
        if not node.config:
            raise HTTPException(status_code=503, detail="Node configuration not initialized")
        
        config = node.config
        
        # Get compute provider for effective values
        effective_cpu_cores = 0.0
        effective_memory_gb = 0.0
        effective_gpu_memory_gb = None
        
        if node.compute_provider:
            cp = node.compute_provider
            effective_cpu_cores = round(cp.resources.cpu_count * cp.cpu_allocation_pct / 100, 2)
            effective_memory_gb = round(cp.resources.memory_total_gb * cp.memory_allocation_pct / 100, 2)
            if cp.resources.gpu_available:
                effective_gpu_memory_gb = round(cp.resources.gpu_memory_gb * cp.gpu_allocation_pct / 100, 2)
        
        # Get storage available
        storage_available_gb = config.storage_gb
        if node.storage_provider:
            storage_available_gb = node.storage_provider.available_gb
        
        return ResourceConfigResponse(
            cpu_allocation_pct=config.cpu_allocation_pct,
            memory_allocation_pct=config.memory_allocation_pct,
            storage_gb=config.storage_gb,
            max_concurrent_jobs=config.max_concurrent_jobs,
            gpu_allocation_pct=config.gpu_allocation_pct,
            upload_mbps_limit=config.upload_mbps_limit,
            download_mbps_limit=config.download_mbps_limit,
            active_hours_start=config.active_hours_start,
            active_hours_end=config.active_hours_end,
            active_days=config.active_days,
            effective_cpu_cores=effective_cpu_cores,
            effective_memory_gb=effective_memory_gb,
            effective_gpu_memory_gb=effective_gpu_memory_gb,
            storage_available_gb=storage_available_gb,
        )

    @app.put("/node/resources", response_model=ResourceConfigResponse, tags=["resources"])
    async def update_resources(request: ResourceUpdateRequest) -> ResourceConfigResponse:
        """Update node resource allocation settings at runtime.
        
        Changes take effect immediately for storage and bandwidth limits.
        Compute changes (CPU, memory, jobs) take effect on next job acceptance.
        
        All fields are optional - only provided fields will be updated.
        
        Validation:
        - cpu_allocation_pct: 10-90
        - memory_allocation_pct: 10-90
        - gpu_allocation_pct: 10-100
        - active_hours_start/end: 0-23
        - storage_gb: must be positive
        - max_concurrent_jobs: at least 1
        - bandwidth limits: non-negative (0 = unlimited)
        """
        if not node.config:
            raise HTTPException(status_code=503, detail="Node configuration not initialized")
        
        config = node.config
        updates_applied = []
        
        # Validate active_hours consistency if both provided
        if request.active_hours_start is not None and request.active_hours_end is not None:
            if request.active_hours_start >= request.active_hours_end:
                raise HTTPException(
                    status_code=400,
                    detail=f"active_hours_start ({request.active_hours_start}) must be less than active_hours_end ({request.active_hours_end})"
                )
        
        # Validate active_days
        if request.active_days is not None:
            for day in request.active_days:
                if not 0 <= day <= 6:
                    raise HTTPException(
                        status_code=400,
                        detail=f"active_days must be 0-6 (Mon-Sun), got {day}"
                    )
        
        try:
            # Update config fields
            if request.cpu_allocation_pct is not None:
                config.cpu_allocation_pct = request.cpu_allocation_pct
                updates_applied.append(f"cpu_allocation_pct={request.cpu_allocation_pct}")
            
            if request.memory_allocation_pct is not None:
                config.memory_allocation_pct = request.memory_allocation_pct
                updates_applied.append(f"memory_allocation_pct={request.memory_allocation_pct}")
            
            if request.storage_gb is not None:
                config.storage_gb = request.storage_gb
                updates_applied.append(f"storage_gb={request.storage_gb}")
            
            if request.max_concurrent_jobs is not None:
                config.max_concurrent_jobs = request.max_concurrent_jobs
                updates_applied.append(f"max_concurrent_jobs={request.max_concurrent_jobs}")
            
            if request.gpu_allocation_pct is not None:
                config.gpu_allocation_pct = request.gpu_allocation_pct
                updates_applied.append(f"gpu_allocation_pct={request.gpu_allocation_pct}")
            
            if request.upload_mbps_limit is not None:
                config.upload_mbps_limit = request.upload_mbps_limit
                updates_applied.append(f"upload_mbps_limit={request.upload_mbps_limit}")
            
            if request.download_mbps_limit is not None:
                config.download_mbps_limit = request.download_mbps_limit
                updates_applied.append(f"download_mbps_limit={request.download_mbps_limit}")
            
            if request.active_hours_start is not None:
                config.active_hours_start = request.active_hours_start
                updates_applied.append(f"active_hours_start={request.active_hours_start}")
            
            if request.active_hours_end is not None:
                config.active_hours_end = request.active_hours_end
                updates_applied.append(f"active_hours_end={request.active_hours_end}")
            
            if request.active_days is not None:
                config.active_days = request.active_days
                updates_applied.append(f"active_days={request.active_days}")
            
            # Log schedule changes
            if request.active_hours_start is not None or request.active_hours_end is not None:
                start = config.active_hours_start
                end = config.active_hours_end
                if start is not None and end is not None:
                    logger.info(f"Active hours schedule updated: {start:02d}:00 - {end:02d}:00")
                else:
                    logger.info("Active hours schedule cleared: node is always on")
            
            # Update live provider settings
            if node.compute_provider:
                node.compute_provider.update_allocation(
                    cpu_allocation_pct=request.cpu_allocation_pct,
                    memory_allocation_pct=request.memory_allocation_pct,
                    max_concurrent_jobs=request.max_concurrent_jobs,
                    gpu_allocation_pct=request.gpu_allocation_pct,
                )
            
            if node.storage_provider:
                await node.storage_provider.update_limits(
                    pledged_gb=request.storage_gb,
                    upload_mbps_limit=request.upload_mbps_limit,
                    download_mbps_limit=request.download_mbps_limit,
                )
            
            # Persist config to disk
            config.save()
            
            logger.info(f"Resource configuration updated: {', '.join(updates_applied)}")
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to update resource configuration: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save configuration: {str(e)}")
        
        # Return updated configuration
        return await get_resources()

    # ── Bridge Endpoints ─────────────────────────────────────────

    # Sprint 161 — Pydantic chain_id constraints. Pre-fix any int
    # passed validation including 0, negative, and absurdly-large
    # values that would have produced opaque downstream errors on
    # a wired bridge.
    class BridgeDepositRequest(BaseModel):
        """Request body for bridge deposit operation."""
        amount: float = Field(
            ..., gt=0, le=1e12, allow_inf_nan=False,
            description="Amount of FTNS to deposit (in token units)",
        )
        chain_address: str = Field(..., min_length=1, description="Destination on-chain address")
        destination_chain: int = Field(
            default=137, ge=1, le=2147483647,
            description="Destination chain ID (default: Polygon mainnet)",
        )

    class BridgeWithdrawRequest(BaseModel):
        """Request body for bridge withdraw operation."""
        amount: float = Field(
            ..., gt=0, le=1e12, allow_inf_nan=False,
            description="Amount of FTNS to withdraw (in token units)",
        )
        chain_address: str = Field(..., min_length=1, description="Source on-chain address")
        source_chain: int = Field(
            default=137, ge=1, le=2147483647,
            description="Source chain ID (default: Polygon mainnet)",
        )

    class BridgeTransactionResponse(BaseModel):
        """Response model for bridge transactions."""
        transaction_id: str
        direction: str
        user_id: str
        chain_address: str
        amount: str
        source_chain: int
        destination_chain: int
        status: str
        source_tx_hash: Optional[str]
        destination_tx_hash: Optional[str]
        fee_amount: str
        created_at: str
        updated_at: str
        completed_at: Optional[str]
        error_message: Optional[str]

    # ──────────────────────────────────────────────────────────────
    # Sprint 548 — /bridge/* scaffold refresh.
    #
    # The 5 /bridge/* endpoints target a polygon_mumbai-era scaffold
    # with no Base-mainnet bridge contract deployed (sprint 539
    # investigation). Sprints 540 + 541 then shipped Pattern A —
    # daemon-mediated bridge — exposing exactly the missing
    # operations under /wallet/deposit/* + /wallet/withdraw.
    # Helper builds operation-specific 503 messages so operators
    # hitting the scaffold get the right working endpoint name.
    # ──────────────────────────────────────────────────────────────
    def _bridge_scaffold_503_detail(operation: str) -> str:
        """Build a Pattern-A-aware 503 detail body for a scaffold
        /bridge/* endpoint.

        ``operation`` selects the operation-specific working endpoint:
          "deposit"   → /wallet/deposit/link + /wallet/deposit/info
          "withdraw"  → /wallet/withdraw
          "status"    → /wallet/deposit/info + /transactions
          "tx_lookup" → /transactions
          "tx_list"   → /transactions
        """
        base = (
            "FTNS bridge endpoints are SCAFFOLD-ONLY in current PRSM "
            "builds. The FTNSBridge module "
            "(prsm/economy/blockchain/ftns_bridge.py) targets "
            "polygon_mumbai-era contracts; no bridge contract is "
            "deployed on Base mainnet. **Pattern A** "
            "(daemon-mediated bridge, sprints 540 + 541) is the "
            "current production-ready bridge surface — no separate "
            "contract; the daemon owns the on-chain wallet + the "
            "off-chain ledger and reconciles via InboundMonitor + "
            "_credit_deposit (deposits) and a debit-first / refund-"
            "on-failure broadcast path (withdrawals)."
        )
        per_op = {
            "deposit": (
                " For this operation use POST `/wallet/deposit/link` "
                "(one-time linkage of your eth address) followed by "
                "an on-chain Transfer to the daemon's escrow "
                "address shown in GET `/wallet/deposit/info` — the "
                "InboundMonitor auto-credits your off-chain wallet."
            ),
            "withdraw": (
                " For this operation use POST `/wallet/withdraw` "
                "(direct broadcast: off-chain debit FIRST, then "
                "on-chain Transfer; refund on broadcast failure)."
            ),
            "status": (
                " For linkage + escrow status use GET "
                "`/wallet/deposit/info`; for bridge transaction "
                "history use GET `/transactions` (rows with type "
                "`bridge_deposit` / `bridge_withdraw`)."
            ),
            "tx_lookup": (
                " For bridge transaction lookup use GET "
                "`/transactions` — Pattern A records each bridge "
                "leg in the off-chain ledger with type "
                "`bridge_deposit` / `bridge_withdraw` and includes "
                "the on-chain tx hash in the description."
            ),
            "tx_list": (
                " For bridge transaction history use GET "
                "`/transactions?limit=N` — Pattern A records each "
                "bridge leg as `bridge_deposit` / `bridge_withdraw` "
                "in the off-chain ledger."
            ),
        }
        return base + per_op.get(operation, "")

    @app.post("/bridge/deposit", tags=["bridge"])
    async def bridge_deposit(request: BridgeDepositRequest) -> Dict[str, Any]:
        """
        Deposit FTNS tokens from local balance to external chain.
        
        Burns local FTNS and initiates bridge transfer to mint tokens on the destination chain.
        
        Args:
            request: BridgeDepositRequest with amount, chain_address, and destination_chain
            
        Returns:
            Bridge transaction details including transaction_id and status
            
        Raises:
            HTTPException 503: If bridge not initialized
            HTTPException 400: If validation fails (insufficient balance, invalid address, etc.)
            HTTPException 500: If bridge operation fails
        """
        if not hasattr(node, 'ftns_bridge') or not node.ftns_bridge:
            raise HTTPException(
                status_code=503,
                detail=_bridge_scaffold_503_detail("deposit"),
            )

        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")

        try:
            # Convert amount to wei (assuming 18 decimals like ETH)
            amount_wei = int(request.amount * 10**18)

            # Execute deposit
            tx = await node.ftns_bridge.deposit_to_chain(
                user_id=node.identity.node_id,
                amount=amount_wei,
                chain_address=request.chain_address,
                destination_chain=request.destination_chain
            )
            
            return {
                "success": True,
                "transaction": tx.to_dict()
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Bridge deposit failed: {error_msg}")
            
            # Map specific errors to HTTP status codes
            if "Insufficient" in error_msg:
                raise HTTPException(status_code=400, detail=error_msg)
            elif "outside limits" in error_msg:
                raise HTTPException(status_code=400, detail=error_msg)
            elif "Invalid" in error_msg:
                raise HTTPException(status_code=400, detail=error_msg)
            else:
                raise HTTPException(status_code=500, detail=f"Bridge deposit failed: {error_msg}")

    @app.post("/bridge/withdraw", tags=["bridge"])
    async def bridge_withdraw(request: BridgeWithdrawRequest) -> Dict[str, Any]:
        """
        Withdraw FTNS tokens from external chain to local balance.
        
        Locks on-chain FTNS and initiates bridge transfer to mint local FTNS.
        
        Args:
            request: BridgeWithdrawRequest with amount, chain_address, and source_chain
            
        Returns:
            Bridge transaction details including transaction_id and status
            
        Raises:
            HTTPException 503: If bridge not initialized
            HTTPException 400: If validation fails (insufficient balance, invalid address, etc.)
            HTTPException 500: If bridge operation fails
        """
        if not hasattr(node, 'ftns_bridge') or not node.ftns_bridge:
            raise HTTPException(
                status_code=503,
                detail=_bridge_scaffold_503_detail("withdraw"),
            )

        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")

        try:
            # Convert amount to wei (assuming 18 decimals like ETH)
            amount_wei = int(request.amount * 10**18)

            # Execute withdraw
            tx = await node.ftns_bridge.withdraw_from_chain(
                chain_address=request.chain_address,
                amount=amount_wei,
                user_id=node.identity.node_id,
                source_chain=request.source_chain
            )
            
            return {
                "success": True,
                "transaction": tx.to_dict()
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Bridge withdraw failed: {error_msg}")
            
            # Map specific errors to HTTP status codes
            if "Insufficient" in error_msg:
                raise HTTPException(status_code=400, detail=error_msg)
            elif "outside limits" in error_msg:
                raise HTTPException(status_code=400, detail=error_msg)
            elif "Invalid" in error_msg:
                raise HTTPException(status_code=400, detail=error_msg)
            else:
                raise HTTPException(status_code=500, detail=f"Bridge withdraw failed: {error_msg}")

    @app.get("/bridge/status", tags=["bridge"])
    async def get_bridge_status() -> Dict[str, Any]:
        """
        Get bridge status and pending operations.
        
        Returns:
            Bridge statistics including:
            - total_deposited: Total FTNS deposited to chain
            - total_withdrawn: Total FTNS withdrawn from chain
            - total_fees_collected: Total fees collected
            - pending_transactions: Number of pending transactions
            - completed_transactions: Number of completed transactions
            - failed_transactions: Number of failed transactions
            - limits: Bridge limits (min/max amounts, fees)
            
        Raises:
            HTTPException 503: If bridge not initialized
        """
        if not hasattr(node, 'ftns_bridge') or not node.ftns_bridge:
            raise HTTPException(
                status_code=503,
                detail=_bridge_scaffold_503_detail("status"),
            )

        try:
            stats = await node.ftns_bridge.get_bridge_stats()
            limits = await node.ftns_bridge.get_bridge_limits()
            pending = await node.ftns_bridge.get_pending_transactions()
            
            return {
                "stats": stats.to_dict(),
                "limits": {
                    "min_amount": str(limits.min_amount) if limits else "0",
                    "max_amount": str(limits.max_amount) if limits else "0",
                    "daily_limit": str(limits.daily_limit) if limits else "0",
                    "fee_bps": limits.fee_bps if limits else 0,
                } if limits else None,
                "pending_transactions": [tx.to_dict() for tx in pending],
                "pending_count": len(pending),
            }
        except Exception as e:
            logger.error(f"Failed to get bridge status: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get bridge status: {str(e)}")

    @app.get("/bridge/transactions/{tx_id}", tags=["bridge"])
    async def get_bridge_transaction(tx_id: str) -> Dict[str, Any]:
        """
        Get status of a specific bridge transaction.
        
        Args:
            tx_id: Transaction ID to look up
            
        Returns:
            Bridge transaction details including status, amounts, and timestamps
            
        Raises:
            HTTPException 503: If bridge not initialized
            HTTPException 404: If transaction not found
        """
        if not hasattr(node, 'ftns_bridge') or not node.ftns_bridge:
            raise HTTPException(
                status_code=503,
                detail=_bridge_scaffold_503_detail("tx_lookup"),
            )

        tx = await node.ftns_bridge.get_bridge_status(tx_id)
        
        if not tx:
            raise HTTPException(status_code=404, detail=f"Transaction {tx_id} not found")
        
        return {
            "transaction": tx.to_dict()
        }

    @app.get("/bridge/transactions", tags=["bridge"])
    async def list_bridge_transactions(limit: int = 50) -> Dict[str, Any]:
        """Sprint 194 — limit bounds checked below."""
        """
        List bridge transactions for the current user.
        
        Args:
            limit: Maximum number of transactions to return (default: 50, max: 200)
            
        Returns:
            List of bridge transactions for the current user
            
        Raises:
            HTTPException 503: If bridge not initialized
        """
        if not hasattr(node, 'ftns_bridge') or not node.ftns_bridge:
            raise HTTPException(
                status_code=503,
                detail=_bridge_scaffold_503_detail("tx_list"),
            )

        # Sprint 194 — bounds validation. Pre-fix `min(limit, 200)`
        # capped upper but accepted negative — limit=-1 returned
        # all bridge transactions for the user.
        if limit < 1 or limit > 200:
            raise HTTPException(
                status_code=422,
                detail=f"limit must be in [1, 200], got {limit}",
            )

        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")

        transactions = await node.ftns_bridge.get_user_transactions(
            user_id=node.identity.node_id,
            limit=limit,
        )
        
        return {
            "transactions": [tx.to_dict() for tx in transactions],
            "count": len(transactions),
        }

    # ── WebSocket Endpoints ───────────────────────────────────────

    @app.websocket("/ws/status")
    async def websocket_status(websocket: WebSocket):
        """
        WebSocket endpoint for real-time status updates.

        Connect to receive live updates about:
        - Node status changes
        - Peer connections/disconnections
        - Job status updates
        - Transaction notifications

        Send JSON messages with type field to interact:
        - {"type": "heartbeat"} - Receive heartbeat acknowledgment
        - {"type": "get_status"} - Request current status
        """
        # Read api_hardening.status_websocket LIVE (not from
        # app.state.status_websocket snapshot). The earlier path
        # captured None because api_hardening.initialize() runs
        # via asyncio.create_task and hadn't completed when
        # app.state.status_websocket was set. Sprint 137 fix.
        status_ws = None
        api_hardening = getattr(app.state, "api_hardening", None)
        if api_hardening is not None:
            status_ws = api_hardening.get_status_websocket()
        if status_ws is None:
            # Fallback to legacy state attr for tests that wire it
            status_ws = getattr(
                app.state, "status_websocket", None,
            )
        if not status_ws:
            await websocket.close(code=1011, reason="WebSocket not initialized")
            return
        
        try:
            await status_ws.connect(websocket)

            while True:
                # Wait for messages from client. Sprint 184 — catch
                # malformed JSON specifically so a client typo doesn't
                # silently close the connection. The handler stays
                # open and sends an error frame back; client can
                # retry with valid JSON.
                try:
                    data = await websocket.receive_json()
                except json.JSONDecodeError:
                    await status_ws.send_personal_status(websocket, {
                        "type": "error",
                        "error": "malformed_json",
                        "message": (
                            "Received non-JSON frame. Send valid "
                            "JSON like {\"type\": \"heartbeat\"}."
                        ),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    continue

                msg_type = data.get("type") if isinstance(data, dict) else None

                # Handle heartbeat
                if msg_type == "heartbeat":
                    await status_ws.send_personal_status(websocket, {
                        "type": "heartbeat_ack",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })

                # Handle status request
                elif msg_type == "get_status":
                    status = await node.get_status()
                    await status_ws.send_personal_status(websocket, {
                        "type": "status_update",
                        "data": status,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })

                else:
                    # Sprint 184 — unknown commands now get an
                    # explicit error frame instead of silent
                    # ignore, so client can correct.
                    await status_ws.send_personal_status(websocket, {
                        "type": "error",
                        "error": "unknown_command",
                        "message": (
                            f"Unknown message type: {msg_type!r}. "
                            f"Supported: 'heartbeat', 'get_status'."
                        ),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

        except WebSocketDisconnect:
            status_ws.disconnect(websocket)
        except Exception as e:
            status_ws.disconnect(websocket)

    # ── Authentication Endpoints ───────────────────────────────────

    @app.get("/auth/verify", tags=["auth"])
    async def verify_token(request: Request, user: Dict[str, Any] = Depends(require_auth)) -> Dict[str, Any]:
        """
        Verify JWT token and return user information.
        
        Requires valid JWT token in Authorization header.
        """
        return {
            "valid": True,
            "user_id": user.get("user_id"),
            "username": user.get("username"),
            "role": user.get("role"),
            "permissions": user.get("permissions", []),
        }

    # ── FTNS Faucet (development / testnet) ──────────────────────────────

    @app.post("/ftns/faucet", tags=["ftns"])
    async def ftns_faucet(body: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Request FTNS tokens from the faucet (development/testnet only).

        Limited to 100 FTNS per request, max 1000 FTNS total per node.
        Disabled in production (set PRSM_FAUCET_ENABLED=0).
        """
        import os
        if os.environ.get("PRSM_FAUCET_ENABLED", "1") == "0":
            raise HTTPException(status_code=403, detail="Faucet disabled in production")

        # Sprint 264 — env-tunable caps (default 100/1000 preserves
        # sprint-181 behavior). Fail-soft to defaults on non-numeric
        # / zero / negative values: zero would brick the faucet, so
        # defensive clamp.
        def _resolve_pos_int(env_key: str, default: int) -> int:
            raw = os.environ.get(env_key, "").strip()
            if not raw:
                return default
            try:
                v = int(raw)
            except (TypeError, ValueError):
                return default
            return v if v > 0 else default

        per_request_cap = _resolve_pos_int(
            "PRSM_FAUCET_MAX_PER_REQUEST", 100,
        )
        per_wallet_cap = _resolve_pos_int(
            "PRSM_FAUCET_MAX_PER_WALLET", 1000,
        )

        # Sprint 181 — validate amount upfront. Pre-fix:
        #   amount = min(float(body.get("amount", 100)), 100)
        # capped at 100 max but had NO lower bound. amount=-1
        # returned 200 with "granted":-1.0 and DEBITED the wallet —
        # converting the faucet into an arbitrary-debit endpoint.
        _raw_amt = body.get("amount", per_request_cap)
        try:
            amount = float(_raw_amt)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"amount must be a positive number; "
                    f"got {_raw_amt!r}."
                ),
            )
        if amount <= 0:
            raise HTTPException(
                status_code=422,
                detail=f"amount must be > 0; got {amount}.",
            )
        # Cap per-request to PRSM_FAUCET_MAX_PER_REQUEST (default 100).
        amount = min(amount, per_request_cap)
        wallet_id = body.get("wallet_id", node.identity.node_id)

        try:
            balance = await node.ledger.get_balance(wallet_id)
            if balance >= per_wallet_cap:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Wallet already has {balance:.0f} FTNS "
                        f"(max {per_wallet_cap} from faucet)"
                    ),
                )

            from prsm.node.local_ledger import TransactionType
            await node.ledger.credit(
                wallet_id=wallet_id,
                amount=amount,
                tx_type=TransactionType.WELCOME_GRANT,
                description=f"Faucet grant: {amount} FTNS",
            )
            new_balance = await node.ledger.get_balance(wallet_id)
            return {
                "granted": amount,
                "new_balance": new_balance,
                "wallet_id": wallet_id,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ftns/faucet/onchain", tags=["ftns"])
    async def ftns_faucet_onchain(body: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Sprint 1190 (day-one blocker #3) — dispense REAL on-chain TESTNET FTNS to a
        new user's address (ERC-20 transfer from the faucet wallet), so the wired front
        door (real balance + deposit_escrow + pay_and_infer) works end to end on Base
        Sepolia. TESTNET-ONLY: the faucet is built only on a Base-Sepolia config and the
        transfer hard-refuses any other chainId — it never gives away real-value FTNS.

        Body: {destination_address: 0x-hex (required), amount: FTNS (optional, capped)}.
        Caps reuse PRSM_FAUCET_MAX_PER_REQUEST/_PER_WALLET (default 100/1000).
        """
        import os
        if os.environ.get("PRSM_FAUCET_ENABLED", "1") == "0":
            raise HTTPException(status_code=403, detail="Faucet disabled")

        faucet = getattr(node, "_onchain_faucet", None)
        if faucet is None:
            from prsm.economy.web3.ftns_faucet import build_onchain_faucet_or_none
            faucet = build_onchain_faucet_or_none()
        if faucet is None:
            raise HTTPException(
                status_code=403,
                detail=(
                    "on-chain FTNS faucet is not available: it is TESTNET-ONLY and "
                    "requires PRSM_FAUCET_PRIVATE_KEY on a Base-Sepolia (chainId 84532) "
                    "node. (It never dispenses on mainnet.)"
                ),
            )

        dest = (body.get("destination_address") or "").strip()
        if not (dest.startswith("0x") and len(dest) == 42):
            raise HTTPException(
                status_code=422,
                detail="destination_address must be a 0x-prefixed 20-byte hex address",
            )

        def _pos_int(env_key: str, default: int) -> int:
            raw = (os.environ.get(env_key, "") or "").strip()
            if not raw:
                return default
            try:
                v = int(raw)
            except (TypeError, ValueError):
                return default
            return v if v > 0 else default

        per_request_cap = _pos_int("PRSM_FAUCET_MAX_PER_REQUEST", 100)
        per_wallet_cap = _pos_int("PRSM_FAUCET_MAX_PER_WALLET", 1000)
        try:
            amount = float(body.get("amount", per_request_cap))
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="amount must be a positive number")
        if amount <= 0:
            raise HTTPException(status_code=422, detail="amount must be > 0")
        amount = min(amount, per_request_cap)

        try:
            decimals = faucet.decimals
            unit = 10 ** decimals
            # Per-wallet cap: refuse if the recipient already holds >= the cap (stateless,
            # bounds how much one address can pull from the faucet).
            current_wei = faucet.balance_of_wei(dest)
            if current_wei >= per_wallet_cap * unit:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"address already holds {current_wei / unit:.0f} FTNS "
                        f"(max {per_wallet_cap} from the faucet)"
                    ),
                )
            from decimal import Decimal
            amount_wei = int(Decimal(str(amount)) * Decimal(unit))
            tx_hash = faucet.dispense(dest, amount_wei)
            return {
                "status": "dispensed",
                "tx_hash": tx_hash,
                "dispensed_ftns": amount,
                "recipient": dest,
                "faucet_address": faucet.faucet_address,
                "network": "base-sepolia",
            }
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            # FaucetMainnetRefusedError / broadcast / revert → surface, never 200-with-error.
            raise HTTPException(status_code=502, detail=f"faucet dispense failed: {e}")

    # ── Web Dashboard (served at /, /static, /api/) ──────────────────────────────

    from pathlib import Path as _Path
    from fastapi.staticfiles import StaticFiles as _StaticFiles
    from fastapi.responses import FileResponse as _FileResponse

    _DASHBOARD_TEMPLATES = _Path(__file__).parent.parent / "dashboard" / "templates"
    _DASHBOARD_STATIC = _Path(__file__).parent.parent / "dashboard" / "static"

    # Serve static assets (JS, CSS) from /static/
    if _DASHBOARD_STATIC.exists():
        app.mount("/static", _StaticFiles(directory=str(_DASHBOARD_STATIC)), name="dashboard-static")

    # Serve the SPA shell at / and /dashboard
    @app.get("/", include_in_schema=False)
    @app.get("/dashboard", include_in_schema=False)
    async def serve_dashboard():
        html_file = _DASHBOARD_TEMPLATES / "dashboard.html"
        if html_file.exists():
            return _FileResponse(str(html_file))
        return {"message": "Dashboard assets not found. Run from PRSM source tree."}

    # Mount the dashboard's API routes. The dashboard sub-app
    # defines its routes WITH `/api/` prefix already (see
    # prsm/dashboard/app.py:173+). Mount at root so the prefix
    # isn't doubled — pre-fix this was mounted at "/api" which
    # made every dashboard route 404 with /api/api/<path>.
    # Dogfood (sprint 136) caught it: the entire dashboard
    # rendered blank because every JS request 404'd.
    # Sprint 685 — register DHT-backed-pool live-attest BEFORE the
    # dashboard mount. app.mount("", ...) installs a catch-all that
    # shadows any route added after it (sprint 685 live-attest
    # surfaced this — endpoint registered correctly in openapi.json
    # but returned 404 because the dashboard mount intercepted
    # /admin/parallax/pool/snapshot first).
    register_parallax_pool_snapshot_endpoint(app, node)
    # Sprint 722 — also BEFORE the dashboard catch-all mount.
    register_parallax_streams_endpoint(app, node)
    # Sprint 769 — auto-claim worker status endpoint. Register
    # BEFORE the dashboard catch-all mount, same as sprint 722
    # (F30 lesson from sprint 685 — catch-all mount swallows any
    # admin endpoint registered after it).
    register_auto_claim_status_endpoint(app, node)
    register_preemption_status_endpoint(app, node)
    register_partial_completion_events_endpoint(app, node)
    register_output_cache_stats_endpoint(app, node)

    try:
        from prsm.dashboard.app import create_dashboard_app as _create_dash_app
        _dash_app = _create_dash_app(node=node)
        app.mount("", _dash_app, name="dashboard-api")
        logger.info("Web dashboard mounted at /")
    except Exception as e:
        logger.warning(f"Dashboard not available: {e}")

    # ── Apply Security Hardening ───────────────────────────────────
    
    if enable_security and api_hardening:
        # Initialize the hardening components
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, schedule initialization
                asyncio.create_task(api_hardening.initialize())
            else:
                loop.run_until_complete(api_hardening.initialize())
        except RuntimeError:
            # No event loop, create one
            asyncio.run(api_hardening.initialize())

        # Re-store status_websocket in app.state AFTER init.
        # initialize() sets api_hardening.status_websocket but the
        # earlier assignment at app.state.status_websocket happened
        # BEFORE init ran (see line ~449), capturing None. The /ws/
        # status WebSocket handler reads app.state.status_websocket
        # and was getting None forever — pre-fix every WebSocket
        # handshake to /ws/status returned 403 because the handler
        # called close() before accept. Sprint 137 fix.
        app.state.status_websocket = api_hardening.get_status_websocket()

        # Apply middleware
        api_hardening.apply_middleware()

    # API key auth for protected endpoints
    from prsm.api.auth_middleware import get_node_auth_middleware, NodeAuthMiddleware
    auth_mw = get_node_auth_middleware(app)
    if auth_mw:
        app.add_middleware(NodeAuthMiddleware, api_key_hash=auth_mw._api_key_hash)

    return app
