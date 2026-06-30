"""Sprint 1315/S3a — per-stage settlement task routing (big-model paid settlement).

Turns a settled MULTI-STAGE §7 receipt into N routable per-node settlement TASKS — one per
stage node — each carrying that node's challenge-defensible ``BatchedReceipt`` (from the
brick-1 splitter) + the requester's per-stage PaymentAuthorization. In Design A each stage
node commits its OWN batch (``msg.sender == provider``), so the orchestrator must DELIVER each
node its task; this module produces the routable tasks (the split + wire-shape). The transport
delivery + the node-side accumulate/commit are S3b (the money path).

Reuses the existing, tested bricks: ``split_receipt_to_per_node_batched_receipts`` (brick 1,
conservation-guaranteed + per-node-signed leaves) and the sp1314-carried
``receipt.per_stage_settlement_signatures``. Pure: no I/O, no signing, no chain. Fail-closed
``None`` exactly where the splitter falls back to single-payee.

Sprint 1316 (S3b-1) adds the NODE-SIDE RECEIVER gate. The requester signs the per-stage
authorization over the WHOLE set's hash, but each node receives only its OWN task — so
``payee_set_from_tasks`` derives the full ``(payee, share)`` set (the orchestrator holds every
task) to route alongside each node's task, and ``verify_routed_settlement_task`` runs the tested
sp1172 verifier fail-closed for the node's own membership BEFORE it accumulates/commits on-chain.
Still pure (verify only — no transport, no commit; those are the rest of S3b).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.per_stage_payment_authorization import (
    DEFAULT_CHAIN_ID,
    InvalidSignatureFormat,
    PerStageAuthorizationVerdict,
    verify_per_stage_authorization,
)
from prsm.settlement.per_stage_settlement_split import (
    split_receipt_to_per_node_batched_receipts,
)
from prsm.settlement.published_batch_store import (
    _batched_receipt_from_dict,
    _batched_receipt_to_dict,
)


@dataclass(frozen=True)
class PerStageSettlementTask:
    """One stage node's settlement task: the node-signed ``BatchedReceipt`` it must commit
    (``msg.sender == node == provider``) + the requester's per-stage authorization the node
    verifies before committing (sp1172) + its share. Routable (JSON via ``to_dict``)."""

    node_id: str
    share_wei: int
    batched_receipt: BatchedReceipt
    # The requester's per-stage PaymentAuthorization {payload, signature} (sp1312). None for a
    # self-pay multi-stage settlement (the node's own multi-host inference).
    payment_authorization: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "share_wei": str(self.share_wei),   # str: may exceed JSON-safe int
            "batched_receipt": _batched_receipt_to_dict(self.batched_receipt),
            "payment_authorization": self.payment_authorization,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PerStageSettlementTask":
        return cls(
            node_id=str(d["node_id"]),
            share_wei=int(d["share_wei"]),
            batched_receipt=_batched_receipt_from_dict(d["batched_receipt"]),
            payment_authorization=d.get("payment_authorization"),
        )


def build_per_stage_settlement_tasks(
    *,
    receipt: Any,
    total_value_wei: int,
    requester_address: str,
    per_stage_authorization: Optional[Dict[str, Any]] = None,
    wallet_map: Any = None,
    tier_slash_rate_bps: int = 0,
    consensus_group_id: bytes = b"\x00" * 32,
) -> Optional[List[PerStageSettlementTask]]:
    """Split a settled MULTI-STAGE receipt into per-node routable settlement tasks.

    Reads the sp1314-carried ``receipt.per_stage_settlement_signatures`` (the per-node
    challenge-defensible material the RPC executor assembled) and runs the brick-1 splitter →
    one conserving, node-signed ``BatchedReceipt`` per distinct stage node, each wrapped with
    the requester's per-stage authorization into a routable ``PerStageSettlementTask``.

    Returns ``None`` (single-payee fallback) FAIL-CLOSED when: the receipt carries no per-node
    signatures, or the splitter declines (topology absent/malformed/<2 nodes, or any node
    unmapped/unsigned). Pure — no transport, no commit (those are S3b)."""
    node_signatures = getattr(receipt, "per_stage_settlement_signatures", None)
    if not node_signatures:
        return None
    shares = split_receipt_to_per_node_batched_receipts(
        receipt=receipt,
        total_value_wei=total_value_wei,
        node_signatures=node_signatures,
        requester_address=requester_address,
        wallet_map=wallet_map,
        tier_slash_rate_bps=tier_slash_rate_bps,
        consensus_group_id=consensus_group_id,
    )
    if shares is None:
        return None
    return [
        PerStageSettlementTask(
            node_id=s.node_id,
            share_wei=s.share_wei,
            batched_receipt=s.batched_receipt,
            payment_authorization=per_stage_authorization,
        )
        for s in shares
    ]


def payee_set_from_tasks(
    tasks: List[PerStageSettlementTask],
) -> List[tuple]:
    """The FULL ``(payee_address, share_wei)`` set a receiving node must verify against.

    The requester's per-stage authorization commits to a ``payee_set_hash`` over the WHOLE
    set of stage payees, but each node receives only its OWN task. To check its membership a
    node needs the full set (``verify_per_stage_authorization`` takes ``payees`` = full set +
    its own ``(payee, share)`` separately). The orchestrator holds every task, so it derives
    the set here and routes it alongside each node's task. Order is irrelevant — the set hash
    is order-independent."""
    return [
        (t.batched_receipt.provider_address, int(t.share_wei)) for t in tasks
    ]


def verify_routed_settlement_task(
    task: PerStageSettlementTask,
    *,
    payees: List[tuple],
    chain_id: int = DEFAULT_CHAIN_ID,
    now_unix: Optional[float] = None,
) -> PerStageAuthorizationVerdict:
    """Node-side FAIL-CLOSED gate (S3b receiver): may THIS node commit the share it was routed?

    A node, on receiving its ``PerStageSettlementTask`` (+ the full ``payees`` set from
    ``payee_set_from_tasks``), confirms the requester's signed per-stage authorization actually
    authorizes paying ITS OWN ``(provider_address, share_wei)`` as a member of the committed set
    — BEFORE it accumulates/commits anything on-chain (brick 4). Reuses the tested sp1172
    verifier so the on-chain commit and this pre-route check apply the IDENTICAL money-safety
    invariants (signer, set-hash, membership, cap, expiry, request binding).

    Returns an ``authorized=False`` verdict (NEVER raises) when: the task carries no
    authorization, the auth payload is malformed, the signature SHAPE is bad, or any invariant
    fails. The caller commits ONLY when ``verdict.authorized``."""
    auth = task.payment_authorization
    if not auth or "payload" not in auth or "signature" not in auth:
        return PerStageAuthorizationVerdict(
            False, "routed task carries no per-stage authorization (payload, signature) — "
            "refusing to commit an unauthorized share (fail-closed)")
    try:
        return verify_per_stage_authorization(
            auth["payload"], auth["signature"],
            payees=payees,
            payee=task.batched_receipt.provider_address,
            share_wei=int(task.share_wei),
            chain_id=chain_id, now_unix=now_unix,
        )
    except InvalidSignatureFormat as exc:
        # A malformed signature SHAPE on the wire is a reject at the receiver, not a crash —
        # the node must never raise its way past the gate.
        return PerStageAuthorizationVerdict(
            False, f"malformed per-stage authorization signature: {exc!s} (fail-closed)")
