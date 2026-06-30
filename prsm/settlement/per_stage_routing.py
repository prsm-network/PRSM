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
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from prsm.settlement.accumulator import BatchedReceipt
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
