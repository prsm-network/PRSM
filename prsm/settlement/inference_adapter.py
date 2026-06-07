"""Sprint 1035 — adapter: parallax InferenceReceipt → settlement BatchedReceipt.

First key-free brick of on-chain cross-node settlement (the remaining Tier-1
loop-closer). The parallax ``InferenceReceipt`` (prsm.compute.inference.models)
and the settlement-layer ``ShardExecutionReceipt`` (prsm.compute.shard_receipt)
are DISJOINT types; this pure adapter bridges them into a ``BatchedReceipt`` that
``ReceiptAccumulator.accumulate`` + the on-chain ``BatchSettlementRegistry``
consume. It does NOT touch the chain or any key beyond the local node's own
identity — committing/finalizing the accumulated batch (which needs a funded
settler key) is a deliberate follow-on.

Why this re-signs (load-bearing, scouted against the real code + contract):
the on-chain leaf (prsm.settlement.merkle.batched_receipt_to_leaf) binds a
``signature`` that an INVALID_SIGNATURE challenge verifies against the leaf's
``signing_message_hash`` = keccak(build_receipt_signing_payload preimage). The
parallax ``settler_signature`` signs ``InferenceReceipt.signing_payload`` — a
DIFFERENT preimage — so copying it verbatim would produce a leaf that fails that
challenge and gets the (honest) provider wrongfully slashed. The settling node
therefore RE-SIGNS the shard payload with its own identity, exactly as the live
shard-execution path does (prsm/node/compute_provider.py).

Single-payee: ``identity`` must be the receipt's settler, so the leaf's provider
binding (provider_id == sha256(pubkey)[:32]) holds and the node signs only its
OWN settlement receipt. Splitting one inference across per-stage payees (each
worker settling its own slice) is a follow-on, gated on each worker producing
its own signed shard receipt.

Invariants the eventual on-chain commit path (Task 6) MUST honor:
- One InferenceReceipt → one stable ``(shard_index, executed_at_unix,
  local_escrow_id)`` tuple. The on-chain DOUBLE_SPEND defense keys on the leaf
  hash (deterministic here — Ed25519 is deterministic), so re-adapting the same
  receipt with a DIFFERENT ``local_escrow_id`` would slip past the accumulator's
  escrow-id dedup yet collide on the leaf hash and revert at commit. Keep
  ``local_escrow_id`` stable across retries; the commit client should also dedup
  on ``hash_leaf`` before posting.
- The on-chain provider-of-record is ``msg.sender`` (the key that signs
  ``commitBatch``), NOT ``provider_address`` (an audit field). Before
  broadcasting, the committer MUST assert ``eth_address(committer_key) ==
  provider_address`` for every receipt in the batch, or funds settle to the
  wrong address.
"""
from __future__ import annotations

import re
from typing import Any

from prsm.compute.shard_receipt import (
    ShardExecutionReceipt,
    build_receipt_signing_payload,
)
from prsm.settlement.accumulator import BatchedReceipt

# 0x-prefixed 20-byte hex Ethereum address (format only — checksum-agnostic).
_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


def inference_receipt_to_batched_receipt(
    *,
    receipt: Any,
    identity: Any,
    requester_address: str,
    provider_address: str,
    value_ftns: int,
    local_escrow_id: str,
    executed_at_unix: int,
    shard_index: int = 0,
    tier_slash_rate_bps: int = 0,
    consensus_group_id: bytes = b"\x00" * 32,
) -> BatchedReceipt:
    """Build a settlement ``BatchedReceipt`` from a signed parallax
    ``InferenceReceipt``.

    Args:
        receipt: a parallax ``InferenceReceipt`` (needs ``job_id``,
            ``output_hash`` raw bytes, ``settler_node_id``).
        identity: the local node's ``NodeIdentity`` — MUST be the receipt's
            settler. Used to re-sign the shard payload (see module docstring).
        requester_address: 0x-hex address that OWES (funds from escrow on-chain).
        provider_address: 0x-hex address that EARNS (audit field; the on-chain
            provider-of-record is msg.sender = the settler key that later commits
            the batch, not this string).
        value_ftns: settled amount in FTNS base units (wei, 18 dec) — the caller
            converts the off-chain Decimal release to wei. Must be a non-negative
            int < 2**128.
        local_escrow_id: the job's escrow id; doubles as the accumulator's sp973
            idempotency key (must be non-empty to dedup).
        executed_at_unix: absolute unix seconds the inference completed
            (``InferenceReceipt`` carries only ``duration_seconds``, so the caller
            supplies this; it is bound into the signed shard payload + the leaf).
        shard_index: logical shard index (single-node inference is one shard → 0).
        tier_slash_rate_bps: 0..10000 (validated by ``BatchedReceipt``).
        consensus_group_id: exactly 32 bytes (validated by ``BatchedReceipt``).

    Returns a ``BatchedReceipt`` whose inner ``ShardExecutionReceipt`` carries a
    signature that verifies over ``build_receipt_signing_payload`` and whose leaf
    (``batched_receipt_to_leaf``) is reconstructable + challenge-defensible.

    Raises ``ValueError`` if ``identity`` is absent or is not the receipt's
    settler. Address-format + range validation is delegated to ``BatchedReceipt``
    / the merkle leaf builder.
    """
    if identity is None or not getattr(identity, "node_id", ""):
        raise ValueError(
            "identity is required: the settling node re-signs the settlement "
            "receipt over the shard payload (the parallax settler_signature "
            "signs a different preimage and is not reusable on-chain)."
        )
    settler = getattr(receipt, "settler_node_id", "") or ""
    if not settler:
        raise ValueError(
            "receipt.settler_node_id is empty: cannot determine the single "
            "payee for settlement."
        )
    if settler != identity.node_id:
        raise ValueError(
            f"identity.node_id {identity.node_id!r} != receipt.settler_node_id "
            f"{settler!r}: this single-payee adapter settles a node's OWN "
            f"receipt; per-stage payee mapping across topology nodes is a "
            f"follow-on (each worker must sign its own shard receipt)."
        )

    # ── Fail-fast boundary validations (sprint 1035 adversarial review) ──
    # Validate at the construction point so a malformed input errors at the call
    # site instead of late-failing in batched_receipt_to_leaf and poisoning the
    # whole accumulated batch (or silently mis-settling on-chain).
    output_hash = getattr(receipt, "output_hash", None)
    if not isinstance(output_hash, (bytes, bytearray)) or len(output_hash) != 32:
        _got = (
            f"len={len(output_hash)}"
            if isinstance(output_hash, (bytes, bytearray))
            else type(output_hash).__name__
        )
        raise ValueError(
            "receipt.output_hash must be a 32-byte sha256 digest (raw bytes); "
            f"got {_got}."
        )
    # value_ftns: an EXACT non-negative wei int. Reject bool (int(True)==1) and
    # float/Decimal (int() would SILENTLY TRUNCATE → wrong settled amount); the
    # caller must convert its Decimal release to a wei int explicitly.
    if isinstance(value_ftns, bool) or not isinstance(value_ftns, int):
        raise ValueError(
            "value_ftns must be a non-negative int in wei (18-dec FTNS base "
            "units) — convert the Decimal release to a wei int at the call "
            f"site; got {type(value_ftns).__name__}."
        )
    if value_ftns < 0 or value_ftns >= 2 ** 128:
        raise ValueError(
            f"value_ftns out of range (need 0 <= v < 2**128 wei): {value_ftns}."
        )
    if not isinstance(local_escrow_id, str) or not local_escrow_id:
        raise ValueError(
            "local_escrow_id must be a non-empty string (it is the sp973 "
            "accumulator idempotency key; it must be stable across retries)."
        )
    for _name, _addr in (
        ("requester_address", requester_address),
        ("provider_address", provider_address),
    ):
        if not isinstance(_addr, str) or not _ADDRESS_RE.match(_addr):
            raise ValueError(
                f"{_name} must be a 0x-prefixed 20-byte hex address "
                f"(^0x[0-9a-fA-F]{{40}}$); got {_addr!r}."
            )
    try:
        executed_at_unix = int(executed_at_unix)
    except (TypeError, ValueError):
        raise ValueError(
            "executed_at_unix must be an int unix timestamp; got "
            f"{type(executed_at_unix).__name__}."
        )
    if executed_at_unix < 0 or executed_at_unix >= 2 ** 64:
        raise ValueError(
            f"executed_at_unix out of range (uint64): {executed_at_unix}."
        )

    output_hash_hex = output_hash.hex()   # raw 32 bytes → 64-char hex
    sig_payload = build_receipt_signing_payload(
        job_id=receipt.job_id,
        shard_index=shard_index,
        output_hash=output_hash_hex,
        executed_at_unix=executed_at_unix,
    )
    # RE-SIGN over the shard payload (NOT receipt.settler_signature, which signs
    # InferenceReceipt.signing_payload — a different preimage). identity.sign
    # returns a base64 str, matching the live shard-execution path.
    signature = identity.sign(sig_payload)

    shard_receipt = ShardExecutionReceipt(
        job_id=receipt.job_id,
        shard_index=shard_index,
        provider_id=identity.node_id,          # == sha256(pubkey)[:32] → binding holds
        provider_pubkey_b64=identity.public_key_b64,
        output_hash=output_hash_hex,
        executed_at_unix=executed_at_unix,
        signature=signature,
        tee_attestation=None,                  # ir.tee_attestation is bytes, not a dict
    )
    return BatchedReceipt(
        receipt=shard_receipt,
        requester_address=requester_address,
        provider_address=provider_address,
        value_ftns=int(value_ftns),
        local_escrow_id=local_escrow_id,
        tier_slash_rate_bps=tier_slash_rate_bps,
        consensus_group_id=consensus_group_id,
    )
