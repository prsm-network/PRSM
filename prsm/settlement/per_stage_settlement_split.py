"""Sprint 1154 — per-node on-chain settlement SPLITTER (BRICK 1 of the
on-chain per-stage payee arc).

THE ARC (project_onchain_per_stage_payee_arc). On a multi-stage cross-host
inference we want to pay EACH topology node its share ON-CHAIN (real FTNS),
not just credit the off-chain sp1031 ledger split. The hard constraint: the
deployed ``BatchSettlementRegistry`` pays the WHOLE batch value to ONE
provider — ``commitBatch`` sets ``b.provider = msg.sender`` and
``finalizeBatch`` settles ``finalValue`` to ``b.provider``. Leaves carry
``providerPubkeyHash`` only for CHALLENGE verification, NOT payment. There is
NO per-leaf payee. So on-chain per-stage payment WITHOUT a (gated,
redeploy+multisig+audit) contract change = one share-batch PER topology node,
each committed BY that node (``b.provider`` = that node's address), valued at
its share. N batches per inference instead of 1.

BRICK SEQUENCE (this module is BRICK 1 only — the pure, money-safe,
offline-testable CORE):
  1. per-node SPLITTER (THIS module): value-conserving equal split of
     ``total_value_wei`` across the distinct topology node_ids, building one
     challenge-defensible per-node ``BatchedReceipt`` each (attributed to that
     node, valued at its share, leaf signed by that node's PROVIDED
     shard-payload signature). Mirrors sp1031 gating. CONSERVATION: sum of
     shares == total EXACTLY in integer wei (remainder deterministic). PURE:
     never signs (no keys), never broadcasts, never touches escrow/chain.
  2. worker shard-payload signature EMISSION (worker-side runtime) — produces,
     in production, the per-node signature material this splitter CONSUMES.
  3. per-node-payee PaymentAuthorization (the requester authorizes the
     share-SET).
  4. per-node COMMIT (each node commits + finalizes its own share-batch).
Bricks 2 + 4 are runtime/multi-node; do NOT build them here. The off-chain
sp1031 ledger split stays the immediate credit mechanism until the arc lands.

WHY each per-node leaf needs the NODE's OWN signature (challenge-defensibility).
The contract verifies a leaf sig via a pluggable Ed25519
``ISignatureVerifier.verify(leaf.signingMessageHash, sig, pubkey)`` (the
INVALID_SIGNATURE challenge succeeds iff ``verify`` returns FALSE). So a
per-node leaf attributed to node B is challenge-DEFENSIBLE only if node B's
pubkey + signature verify over the leaf's ``signingMessageHash``. The settler
convention (``prsm/settlement/inference_adapter.py`` +
``prsm/compute/shard_receipt.build_receipt_signing_payload``) signs a 32-BYTE
``keccak("{job_id}||{shard_index}||{output_hash}||{executed_at_unix}")`` —
that 32-byte value IS the leaf ``signingMessageHash``, so the sig verifies. The
per-stage ``StageActivationProof`` signature is NOT reusable (it signs a
variable-length payload, not a 32-byte hash). So this splitter CONSUMES, per
node, that node's challenge-defensible shard-payload signature (the
worker-emission brick will produce it in production; HERE the splitter takes
it as INPUT). The splitter NEVER signs (it has no keys).

GATING (mirror sp1031 ``split_release_across_stages``). Return ``None`` (caller
keeps the single-payee path) when the receipt carries no usable topology:
``topology_assignment`` absent, malformed (``positions`` not a non-empty dict),
or naming fewer than 2 DISTINCT node_ids. Additionally FAIL-CLOSED: return
``None`` when any required per-node signature is missing — NEVER fabricate or
guess a payee leaf, and NEVER return a partial payee set.
"""
from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from prsm.compute.shard_receipt import (
    ShardExecutionReceipt,
    build_receipt_signing_payload,
    per_stage_leaf_job_id,
)
from prsm.settlement.accumulator import BatchedReceipt

# uint128 ceiling — mirror the adapter / merkle leaf value bound. A single
# node's share is itself a uint128 value_ftns, and so is the total (the total
# is the sum of the shares), so the total must fit a uint128 too.
_MAX_UINT128 = 2 ** 128 - 1
_MAX_UINT64 = 2 ** 64 - 1


@dataclass(frozen=True)
class NodeSignatureMaterial:
    """A single topology node's challenge-defensible shard-payload signature
    material — the PUBLIC inputs the splitter consumes to build that node's
    leaf. The splitter holds NO private key; the worker-emission brick (brick
    2) produces this in production.

    Fields:
      pubkey_b64:        the node's base64 Ed25519 public key. The leaf binds
                         ``provider_id == sha256(b64decode(pubkey))[:32]``, so
                         this must derive the node's own node_id.
      signature:        base64 Ed25519 signature over
                         ``build_receipt_signing_payload(job_id, stage_index,
                         output_hash, executed_at_unix)`` — the 32-byte value
                         that IS the leaf ``signing_message_hash``.
      stage_index:      this node's logical shard/stage index (the
                         ``shard_index`` the leaf binds + the value signed).
      output_hash:      the 64-char hex sha256 output digest the leaf binds +
                         the value signed.
      executed_at_unix: absolute unix seconds bound into the signed payload +
                         the leaf.
    """

    pubkey_b64: str
    signature: str
    stage_index: int
    output_hash: str
    executed_at_unix: int


@dataclass(frozen=True)
class StageSettlementShare:
    """One topology node's settlement share + the per-node ``BatchedReceipt``
    that the downstream commit brick (brick 4) will commit BY that node."""

    node_id: str
    share_wei: int
    batched_receipt: BatchedReceipt


def compute_per_stage_shares(
    total_value_wei: int,
    node_ids: List[str],
) -> List[Tuple[str, int]]:
    """Value-conservation primitive: split ``total_value_wei`` EQUALLY across
    ``node_ids`` with EXACT integer-wei conservation.

    Each node gets ``base = total // N``; the remainder ``total % N`` is
    distributed deterministically as +1 wei to the FIRST ``total % N`` nodes in
    the given (stable) order. Guarantees:

      - ``sum(shares) == total_value_wei`` EXACTLY (no FTNS created or lost),
      - every share ``>= 0``,
      - no share exceeds ``ceil(total / N)`` (no node over-paid),
      - deterministic: same inputs → same output, every call.

    Rejects ``total < 0``, empty ``node_ids``, a total beyond the uint128 range
    (the leaf value bound), and a non-int total (``bool`` included — ``int``
    of a ``bool`` would silently coerce a True/False into a wei amount).

    Pure: no I/O, no signing, no side effects.
    """
    if isinstance(total_value_wei, bool) or not isinstance(total_value_wei, int):
        raise TypeError(
            "total_value_wei must be a non-negative int in wei; got "
            f"{type(total_value_wei).__name__}."
        )
    if total_value_wei < 0:
        raise ValueError(
            f"total_value_wei must be non-negative; got {total_value_wei}."
        )
    if total_value_wei > _MAX_UINT128:
        raise ValueError(
            f"total_value_wei out of range (need 0 <= v <= 2**128-1 wei): "
            f"{total_value_wei}."
        )
    if not node_ids:
        raise ValueError("node_ids must be a non-empty list.")

    n = len(node_ids)
    base = total_value_wei // n
    remainder = total_value_wei % n
    shares: List[Tuple[str, int]] = []
    for i, node_id in enumerate(node_ids):
        share = base + (1 if i < remainder else 0)
        shares.append((node_id, share))
    # Defensive money-safety assertion (cheap, pure): conservation MUST hold.
    assert sum(s for _, s in shares) == total_value_wei, (
        "compute_per_stage_shares broke conservation"
    )
    return shares


def _distinct_node_ids_with_stage(
    topology: Any,
) -> Optional[List[Tuple[str, int]]]:
    """Mirror sp1031 gating to extract the distinct node_ids in deterministic
    (stage, slot) order, pairing each with the FIRST stage_index it occupies.

    Returns ``None`` when the topology is absent / malformed / names fewer than
    2 distinct node_ids — the single-payee fallback signal.
    """
    if topology is None:
        return None
    positions = getattr(topology, "positions", None)
    if not isinstance(positions, dict) or not positions:
        return None

    # Deterministic order over the (stage, slot) keys so the remainder + stage
    # attribution land deterministically (matches sp1031's sorted-key walk).
    try:
        ordered_keys = sorted(positions.keys())
    except TypeError:
        ordered_keys = list(positions.keys())

    distinct: List[Tuple[str, int]] = []
    seen: set = set()
    for key in ordered_keys:
        node_id = positions[key]
        if not isinstance(node_id, str) or not node_id or node_id in seen:
            continue
        # The stage_index is the first element of the (stage, slot) key when
        # the key is a 2-tuple; fall back to the node's ordinal otherwise.
        if isinstance(key, tuple) and len(key) >= 1 and isinstance(key[0], int):
            stage_index = key[0]
        else:
            stage_index = len(distinct)
        seen.add(node_id)
        distinct.append((node_id, stage_index))

    if len(distinct) < 2:
        return None
    return distinct


def split_receipt_to_per_node_batched_receipts(
    *,
    receipt: Any,
    total_value_wei: int,
    node_signatures: Dict[str, NodeSignatureMaterial],
    requester_address: str,
    tier_slash_rate_bps: int = 0,
    consensus_group_id: bytes = b"\x00" * 32,
) -> Optional[List[StageSettlementShare]]:
    """Split ``total_value_wei`` across the receipt's distinct topology nodes,
    building one challenge-defensible per-node ``BatchedReceipt`` each.

    GATING — returns ``None`` (caller keeps the single-payee path) when:
      - ``receipt.topology_assignment`` is absent / malformed / names fewer
        than 2 distinct node_ids (mirror sp1031), OR
      - any required per-node signature is missing from ``node_signatures``
        (FAIL-CLOSED: never fabricate a payee leaf, never a partial set), OR
      - any provided signature material fails the provider binding
        (``provider_id == sha256(pubkey)[:32] == node_id``).

    For >= 2 distinct nodes with all signatures present + binding: computes the
    conserving per-node shares, then builds one ``BatchedReceipt`` per node
    attributed to that node (``provider_id = node_id``,
    ``provider_pubkey_b64`` from the sig material, ``value_ftns`` = its share,
    the leaf-defining fields — ``shard_index``, ``output_hash``,
    ``executed_at_unix``, ``signature`` — from that node's signed shard payload
    so the leaf is challenge-defensible). Returns the list of
    ``StageSettlementShare``.

    Args:
      receipt: a parallax ``InferenceReceipt`` carrying ``request_id`` (the
        per-node settlement leaf job_id is derived from it via the streamed-
        agnostic ``per_stage_leaf_job_id``) and a ``topology_assignment`` whose
        ``positions`` map (stage, slot) → node_id.
      total_value_wei: the whole-inference settled amount in wei (the sum the
        shares conserve). Validated to a non-negative uint128 int.
      node_signatures: node_id → ``NodeSignatureMaterial``. Each entry is that
        node's OWN challenge-defensible shard-payload signature material.
      requester_address: 0x-hex address that OWES (funds from escrow). The
        per-node ``provider_address`` audit field is set to the SAME address as
        a deliberate placeholder — the on-chain provider-of-record is
        ``msg.sender`` (the committing NODE in brick 4), NOT this string, and
        the splitter has no node→eth-address map (that resolution is brick 3/4).
      tier_slash_rate_bps / consensus_group_id: passed through to each
        per-node ``BatchedReceipt`` (validated by ``BatchedReceipt``).

    Returns ``Optional[List[StageSettlementShare]]``. PURE: never signs, never
    broadcasts, never touches escrow/chain — it only consumes the PROVIDED
    signature material.

    Raises ``ValueError`` only for an out-of-range / malformed
    ``total_value_wei`` (delegated to ``compute_per_stage_shares``); topology /
    signature problems are GATING conditions that return ``None``, not raises.
    """
    distinct = _distinct_node_ids_with_stage(
        getattr(receipt, "topology_assignment", None)
    )
    if distinct is None:
        return None

    # FAIL-CLOSED: every distinct node must have provided signature material,
    # and that material must satisfy the provider binding. Any miss → None
    # (single-payee fallback), NEVER a fabricated / partial payee set.
    if not isinstance(node_signatures, dict):
        return None
    for node_id, _stage in distinct:
        mat = node_signatures.get(node_id)
        if mat is None:
            return None
        if not isinstance(mat, NodeSignatureMaterial):
            return None
        if not mat.pubkey_b64 or not mat.signature:
            return None
        # Binding: provider_id == sha256(b64decode(pubkey))[:32] == node_id.
        try:
            pub_bytes = base64.b64decode(mat.pubkey_b64)
        except Exception:
            return None
        derived = hashlib.sha256(pub_bytes).hexdigest()[:32]
        if derived != node_id:
            return None

    # sp1158 — the per-node settlement leaf binds the STREAMED-AGNOSTIC
    # settlement job_id derived from request_id ALONE (NOT the receipt's
    # streamed-prefixed DISPLAY job_id). This is THE id the worker — which at
    # stage-execution time knows only request_id, not the receipt's eventual
    # output streamed flag — also binds when it emits its per-node signature, so
    # the worker signs EXACTLY the leaf this splitter rebuilds (a single source
    # of truth via per_stage_leaf_job_id). The receipt's own display job_id +
    # the single-payee adapter (inference_adapter, receipt.job_id) are unchanged.
    request_id = getattr(receipt, "request_id", None)
    if not isinstance(request_id, str) or not request_id:
        return None
    job_id = per_stage_leaf_job_id(request_id)

    node_ids = [node_id for node_id, _stage in distinct]
    # compute_per_stage_shares validates + raises on a bad total; that is a
    # caller programming error (a non-uint128 settled amount), not a gating
    # condition, so it propagates rather than silently falling back.
    shares = compute_per_stage_shares(total_value_wei, node_ids)
    share_by_node = dict(shares)

    out: List[StageSettlementShare] = []
    for node_id, _stage in distinct:
        mat = node_signatures[node_id]
        share_wei = share_by_node[node_id]
        shard_receipt = ShardExecutionReceipt(
            job_id=job_id,
            shard_index=mat.stage_index,
            provider_id=node_id,                 # binding holds (checked above)
            provider_pubkey_b64=mat.pubkey_b64,
            output_hash=mat.output_hash,
            executed_at_unix=mat.executed_at_unix,
            signature=mat.signature,             # the node's OWN signature
            tee_attestation=None,
        )
        # local_escrow_id namespaces the per-node share-batch under the job +
        # this node so the accumulator's sp973 idempotency key is stable +
        # distinct per payee (one job fans into N distinct share-batches).
        batched = BatchedReceipt(
            receipt=shard_receipt,
            requester_address=requester_address,
            provider_address=requester_address,
            value_ftns=int(share_wei),
            local_escrow_id=f"{job_id}::stage::{node_id}",
            tier_slash_rate_bps=tier_slash_rate_bps,
            consensus_group_id=consensus_group_id,
        )
        out.append(
            StageSettlementShare(
                node_id=node_id,
                share_wei=share_wei,
                batched_receipt=batched,
            )
        )

    # Money-safety: the per-node share-batch values conserve the total exactly.
    assert sum(s.share_wei for s in out) == total_value_wei, (
        "per-node split broke conservation"
    )
    return out
