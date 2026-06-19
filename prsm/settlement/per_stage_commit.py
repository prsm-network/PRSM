"""Sprint 1159 — per-node on-chain COMMIT orchestration (BRICK 4 of the
on-chain per-stage payee arc).

THE ARC (project_onchain_per_stage_payee_arc). On a multi-stage cross-host
inference we pay EACH topology node its share ON-CHAIN. The deployed
``BatchSettlementRegistry`` pays the WHOLE batch value to ONE provider —
``commitBatch`` sets ``b.provider = msg.sender`` and ``finalizeBatch`` settles
``finalValue`` from the requester's escrow to ``b.provider`` via
``EscrowPool.settleFromRequester(b.requester, b.provider, finalValue)``. There
is NO per-leaf payee. So on-chain per-stage payment WITHOUT a (gated,
redeploy+multisig+audit) contract change = one share-batch PER topology node,
each COMMITTED BY THAT NODE (``b.provider`` = that node's address, via that
node's own settler key), valued at its conserving share. N batches per
inference instead of 1.

BRICK SEQUENCE:
  1. per-node SPLITTER — DONE (``per_stage_settlement_split``): a multi-stage
     receipt -> one value-conserving, challenge-defensible per-node
     ``BatchedReceipt`` (one per distinct topology node).
  2. worker shard-payload signature EMISSION — DONE.
  2.5 deterministic per-stage leaf job_id — DONE.
  3. per-node-payee PaymentAuthorization — DONE.
  4. per-node COMMIT (THIS module): each node commits + finalizes its OWN
     share-batch on-chain, keyed by its OWN eth address (``msg.sender`` == that
     node => ``b.provider`` == that node). The requester (escrow funded with
     the TOTAL) is drained by the N shares; each node receives its share.

THIS module is PURE ORCHESTRATION — it does NOT reinvent the settlement client.
It orchestrates N existing ``BatchSettlementClient`` instances (one per node,
each bound to + keyed by that node's address). The single-payee path
(``inference_adapter``) is UNCHANGED; brick 4 is additive.

WHY a ``provider_address`` re-stamp (leaf-neutral, money-safe). The brick-1
splitter sets each per-node ``BatchedReceipt.provider_address`` to the requester
address as a deliberate placeholder (it has no node->eth-address map). For a
NODE to commit its own share-batch, two things must hold:

  - ``BatchSettlementClient.accumulate`` requires the receipt's
    ``provider_address`` OR ``requester_address`` to equal the client's bound
    address. A node-bound client would REJECT the placeholder unless the node
    happened to be the requester. So we re-stamp ``provider_address`` to the
    committing NODE's eth address before handing it to that node's client.
  - The on-chain payee is ``msg.sender`` (the node's settler key), NOT the
    ``provider_address`` audit string — so the re-stamp does not change WHO is
    paid on-chain (the node's key already determines that).

The re-stamp is provably LEAF-NEUTRAL: the canonical Merkle leaf
(``batched_receipt_to_leaf``) is built from the receipt's ``provider_id`` /
``provider_pubkey`` / ``output_hash`` / ``signature`` / ``value_ftns`` —
NOT from ``provider_address``. So re-stamping ``provider_address`` leaves the
leaf hash + root + challenge-defensibility BYTE-IDENTICAL. The share value
(``value_ftns``) is untouched, so conservation is preserved.

FAIL-CLOSED, never auto-broadcast beyond the explicit commit/finalize. This
module exposes two deliberate, caller-driven steps (commit, finalize). It never
polls a background loop and never broadcasts anything the caller did not ask
for. A per-node failure is isolated (reported), never silently retried into a
double-settle (the underlying client's quarantine/WAL handles that).
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from prsm.settlement.client import (
    BatchSettlementClient,
    CommittedBatch,
    FinalizedBatch,
)
from prsm.settlement.per_stage_payment_authorization import (
    DEFAULT_CHAIN_ID,
    verify_per_stage_authorization,
)
from prsm.settlement.per_stage_settlement_split import StageSettlementShare

logger = logging.getLogger(__name__)

# A requester-signed PerStagePaymentAuthorization: (payload_dict, signature). The payload
# commits to a canonical payee_set_hash + cumulative cap; the signature is the requester's
# EIP-712 signature over it (see per_stage_payment_authorization).
PerStageAuthorization = Tuple[Dict[str, Any], Union[bytes, str]]


class PerStageAuthorizationError(Exception):
    """sp1172 — raised when the requester's per-stage authorization does NOT cover the per-node
    payee set about to be committed (missing auth, unauthorized payee/share, set-hash mismatch,
    over-cap, expired). FAIL-CLOSED: NO share-batch is committed when this raises."""


@dataclass(frozen=True)
class NodeCommitResult:
    """The outcome of committing (and optionally finalizing) ONE node's
    share-batch. ``committed`` is None when the commit did not land (the
    underlying client kept the receipts for retry / quarantined them); a
    non-None ``error`` carries the failure reason for operator surfacing.
    Money-safety is preserved either way — a failed node never over/under-pays,
    it simply isn't settled this pass."""

    node_id: str
    committer_address: str
    share_wei: int
    committed: Optional[CommittedBatch]
    finalized: Optional[FinalizedBatch]
    error: Optional[str] = None


def _restamp_provider_address(
    share: StageSettlementShare, committer_address: str,
) -> StageSettlementShare:
    """Return a copy of ``share`` whose per-node ``BatchedReceipt`` is attributed
    (``provider_address``) to ``committer_address`` — the eth address the node
    commits with. LEAF-NEUTRAL (the Merkle leaf ignores ``provider_address``);
    the on-chain payee is ``msg.sender``, so this only satisfies the client's
    bound-address guard + audit attribution. ``value_ftns`` (the money) and every
    leaf-defining field are untouched."""
    br = share.batched_receipt
    restamped = dataclasses.replace(br, provider_address=committer_address)
    return dataclasses.replace(share, batched_receipt=restamped)


def _verify_authorization(
    shares: List[StageSettlementShare],
    committer_address_for_node: Callable[[str], str],
    authorization: Optional[PerStageAuthorization],
    chain_id: int,
    now_unix: Optional[float],
) -> None:
    """sp1172 FAIL-CLOSED money-safety gate. Bind the per-node payee set ABOUT TO BE COMMITTED
    to the requester's signed PerStagePaymentAuthorization BEFORE any commit. Verifies EVERY
    (committer_address, share_wei) is an authorized member of the signed set (which also
    confirms the supplied set hashes to the signed payee_set_hash + the cumulative cap holds +
    not expired). Raises ``PerStageAuthorizationError`` on a missing auth or ANY unauthorized
    node — so the caller commits NOTHING (no partial / unauthorized payout)."""
    if authorization is None:
        raise PerStageAuthorizationError(
            "per-stage commit requires the requester's signed PerStagePaymentAuthorization "
            "(payload, signature); refusing to commit an unauthorized payee set — the escrow "
            "draw must be bound to what the requester actually authorized"
        )
    payload, signature = authorization
    # the FINAL per-node (payee, share) pairs about to be committed.
    payees = [
        (committer_address_for_node(s.node_id), int(s.share_wei)) for s in shares
    ]
    for s in shares:
        committer = committer_address_for_node(s.node_id)
        verdict = verify_per_stage_authorization(
            payload, signature,
            payees=payees, payee=committer, share_wei=int(s.share_wei),
            chain_id=chain_id, now_unix=now_unix,
        )
        if not verdict.authorized:
            raise PerStageAuthorizationError(
                f"per-stage authorization does NOT cover node {s.node_id} "
                f"(payee={committer}, share={s.share_wei} wei): {verdict.reason} — "
                f"committing NOTHING (fail-closed)"
            )


async def commit_per_node_share_batches(
    *,
    shares: List[StageSettlementShare],
    client_for_node: Callable[[str], BatchSettlementClient],
    committer_address_for_node: Callable[[str], str],
    authorization: Optional[PerStageAuthorization],
    chain_id: int = DEFAULT_CHAIN_ID,
    now_unix: Optional[float] = None,
) -> Dict[str, CommittedBatch]:
    """Each node commits its OWN share-batch on-chain (``msg.sender`` == that
    node => ``b.provider`` == that node, valued at its conserving share).

    sp1172 MONEY-SAFETY GATE (fail-closed, runs FIRST): the per-node payee set about to be
    committed is bound to the requester's signed ``authorization`` (a ``PerStagePaymentAuthorization``
    payload + signature). EVERY (committer, share) must be an authorized member of the signed set
    (which also confirms the set hashes to the signed commitment + the cumulative cap holds + the
    auth is unexpired). If the auth is missing or does not cover ANY node, this raises
    ``PerStageAuthorizationError`` and commits NOTHING — the escrow draw can never exceed / diverge
    from what the requester authorized.

    PURE ORCHESTRATION of the existing per-node ``BatchSettlementClient``s — after the auth gate,
    for each share it: re-stamps ``provider_address`` to the node's committer address
    (leaf-neutral), ``accumulate``s the single share-receipt into THAT node's
    client, then drives ``commit_ready_batches`` on that client. Because each
    share is a single receipt whose value clears the accumulator's value
    threshold (the bench sets a small threshold) OR is force-flushed by the
    caller's accumulator config, the commit fires deterministically.

    Returns ``{node_id: CommittedBatch}`` for every node that committed. A node
    whose commit did not land is OMITTED (the underlying client retains/quarantines
    its receipts — never a double-settle). NEVER broadcasts beyond the deliberate
    ``commit_ready_batches`` call per node.

    Args:
      shares: the brick-1 per-node share-batches (each one node, one receipt,
        its conserving ``value_ftns``).
      client_for_node: node_id -> the ``BatchSettlementClient`` bound to + keyed
        by that node's eth address (the node IS its own committer).
      committer_address_for_node: node_id -> the eth address that node commits
        with (== ``client_for_node(node_id)``'s bound address).
      authorization: the requester's signed ``PerStagePaymentAuthorization`` (payload, signature)
        covering the per-node payee set. REQUIRED — fail-closed (a missing/insufficient auth
        commits nothing).
      chain_id / now_unix: the EIP-712 chain binding + the expiry clock for auth verification.
    """
    # FAIL-CLOSED: verify the requester authorized this exact payee set BEFORE committing anything.
    _verify_authorization(shares, committer_address_for_node, authorization, chain_id, now_unix)

    committed: Dict[str, CommittedBatch] = {}
    for share in shares:
        node_id = share.node_id
        committer = committer_address_for_node(node_id)
        client = client_for_node(node_id)
        restamped = _restamp_provider_address(share, committer)
        # Each node accumulates ONLY its own single share-receipt, then commits.
        await client.accumulate(restamped.batched_receipt)
        records = await client.commit_ready_batches()
        # A node's client holds only its own one share-batch this pass; the
        # commit-success list has at most one record for this node.
        if records:
            committed[node_id] = records[0]
            logger.info(
                "per-stage commit: node %s committed share %d wei as batch %s "
                "(msg.sender=%s => on-chain provider=node)",
                node_id, share.share_wei, records[0].batch_id.hex()[:12], committer,
            )
        else:
            logger.warning(
                "per-stage commit: node %s did NOT land a commit this pass "
                "(receipts retained/quarantined by its client) — share %d wei "
                "NOT settled; never re-committed blindly (no double-settle)",
                node_id, share.share_wei,
            )
    return committed


async def finalize_per_node_share_batches(
    *,
    committed: Dict[str, CommittedBatch],
    client_for_node: Callable[[str], BatchSettlementClient],
) -> Dict[str, FinalizedBatch]:
    """Each node finalizes its OWN committed share-batch (drives
    ``finalize_ready_batches`` on that node's client). Idempotent + driven by
    on-chain ``isFinalizable`` (the challenge window must have elapsed — the
    bench time-travels past it). Returns ``{node_id: FinalizedBatch}`` for the
    nodes whose finalize tx was submitted this pass. NEVER broadcasts beyond the
    deliberate ``finalize_ready_batches`` call per node."""
    finalized: Dict[str, FinalizedBatch] = {}
    for node_id in committed:
        client = client_for_node(node_id)
        results = await client.finalize_ready_batches()
        for fb in results:
            if fb.tx_submitted:
                finalized[node_id] = fb
                logger.info(
                    "per-stage finalize: node %s finalized batch %s — escrow "
                    "settles its share to the node",
                    node_id, fb.batch_id.hex()[:12],
                )
    return finalized
