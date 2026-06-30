"""Sprint 1319 (S3b-2c) — orchestrator-side SENDER for routed per-stage settlement tasks.

The settling node, having served a multi-stage cross-host inference, splits the settled §7
receipt into N per-node tasks (S3a) and must DELIVER each stage node its task + the full payee
set so that node can self-commit its share (Design A: ``msg.sender == provider``). This module is
the send half: it POSTs each ``(task, payees)`` to its node's ``POST /settlement/per-stage-task``
endpoint (sp1318).

FAIL-SOFT by design: a delivery never raises. A miss (transport error / non-200 / unmapped node)
is recorded + logged; the consequence is simply that that stage goes undelivered → unpaid
on-chain (the v1 per-stage-independence weakness, the SAFE failure — no double / wrong pay, and
the requester's escrow is not drawn for an unfinalized stage). The node→URL resolver +
``http_post`` are injected (tested without a network); the post-settlement hook that supplies the
real peer-registry resolver is wired with the on-chain commit brick (S3b-3).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

from prsm.settlement.per_stage_routing import (
    build_per_stage_settlement_tasks,
    payee_set_from_tasks,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeliveryResult:
    """The outcome of delivering ONE node's task. ``delivered`` = the POST reached the node and
    returned 200; ``accepted`` = the node's gate authorized + staged it (None when not delivered).
    ``reason`` carries the node's reason or the transport failure for operator surfacing."""

    node_id: str
    delivered: bool
    accepted: Optional[bool]
    reason: str
    local_escrow_id: Optional[str] = None
    status_code: Optional[int] = None


def deliver_per_stage_task(
    peer_url: str,
    task: Any,
    payees: List[Tuple[str, int]],
    *,
    http_post: Optional[Callable[..., Any]] = None,
    timeout: float = 10.0,
) -> DeliveryResult:
    """POST one ``(task, payees)`` to a node's ``/settlement/per-stage-task`` endpoint (sp1318).

    ``peer_url`` is the node base URL (e.g. ``http://host:8000``). ``http_post`` defaults to
    ``httpx.post`` (injected in tests). FAIL-SOFT: NEVER raises — a transport error / non-200 /
    non-JSON returns ``delivered=False`` with the reason. On 200 returns the node's
    ``{accepted, reason, local_escrow_id}``."""
    url = f"{peer_url.rstrip('/')}/settlement/per-stage-task"
    body = {"task": task.to_dict(), "payees": [[p, str(s)] for p, s in payees]}

    if http_post is None:
        import httpx
        http_post = httpx.post

    try:
        resp = http_post(url, json=body, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — fail-soft: a miss leaves the stage unpaid (safe)
        logger.warning("per-stage delivery to %s (node %s) transport error: %s",
                       url, task.node_id, exc)
        return DeliveryResult(task.node_id, False, None, f"transport error: {exc}")

    status = getattr(resp, "status_code", None)
    if status != 200:
        detail = ""
        try:
            detail = (resp.json() or {}).get("detail", "")
        except Exception:  # noqa: BLE001
            detail = getattr(resp, "text", "")
        logger.warning("per-stage delivery to %s (node %s) returned %s: %s",
                       url, task.node_id, status, detail)
        return DeliveryResult(task.node_id, False, None,
                              f"peer returned {status}: {detail}", status_code=status)

    try:
        data = resp.json() or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("per-stage delivery to %s (node %s) non-JSON body: %s",
                       url, task.node_id, exc)
        return DeliveryResult(task.node_id, True, None, f"non-JSON response: {exc}",
                              status_code=status)

    accepted = bool(data.get("accepted"))
    return DeliveryResult(
        node_id=task.node_id, delivered=True, accepted=accepted,
        reason=str(data.get("reason", "")),
        local_escrow_id=data.get("local_escrow_id"), status_code=status)


def deliver_settled_multistage_tasks(
    *,
    receipt: Any,
    total_value_wei: int,
    requester_address: str,
    endpoint_for_node: Callable[[str], Optional[str]],
    per_stage_authorization: Optional[dict] = None,
    wallet_map: Any = None,
    http_post: Optional[Callable[..., Any]] = None,
    timeout: float = 10.0,
    tier_slash_rate_bps: int = 0,
    consensus_group_id: bytes = b"\x00" * 32,
) -> List[DeliveryResult]:
    """Split a settled multi-stage receipt into per-node tasks (S3a) and DELIVER each to its node.

    Returns ``[]`` when the receipt is not a settleable multi-stage one (no per-stage signatures /
    single node / any node unmapped — the splitter fails closed, same as S3a). Otherwise derives
    the full payee set ONCE (``payee_set_from_tasks``) and POSTs each ``(task, payees)`` to its
    node via ``endpoint_for_node(node_id)``. A node with no resolvable endpoint is recorded as an
    undelivered ``DeliveryResult`` (logged) — NEVER raises. The on-chain commit is each node's own
    concern (S3b-3); this only routes the tasks."""
    tasks = build_per_stage_settlement_tasks(
        receipt=receipt, total_value_wei=total_value_wei,
        requester_address=requester_address,
        per_stage_authorization=per_stage_authorization, wallet_map=wallet_map,
        tier_slash_rate_bps=tier_slash_rate_bps, consensus_group_id=consensus_group_id)
    if not tasks:
        return []
    payees = payee_set_from_tasks(tasks)
    results: List[DeliveryResult] = []
    for task in tasks:
        url = endpoint_for_node(task.node_id)
        if not url:
            logger.warning("per-stage delivery: no endpoint for node %s — stage undelivered "
                           "(unpaid; safe failure)", task.node_id)
            results.append(DeliveryResult(
                task.node_id, False, None, "no resolvable endpoint for node"))
            continue
        results.append(deliver_per_stage_task(
            url, task, payees, http_post=http_post, timeout=timeout))
    return results
