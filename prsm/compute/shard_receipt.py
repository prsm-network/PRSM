"""Phase 2: ShardExecutionReceipt + VerificationStrategy.

Signed receipt for remote shard execution. The provider signs a
canonical payload after executing; the requester verifies the
signature matches the provider's advertised pubkey AND the
output_hash matches the actual bytes returned.

Tier-A verification (receipt-only) is the Phase 2 default. Tiers B
(redundant execution) and C (stake + slash) plug in at Phase 7 via
the same VerificationStrategy protocol without changing the receipt
format or the dispatch protocol.
"""
from __future__ import annotations

import base64
import hashlib
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Protocol

from eth_utils import keccak

logger = logging.getLogger(__name__)


def build_receipt_signing_payload(
    job_id: str,
    shard_index: int,
    output_hash: str,
    executed_at_unix: int,
) -> bytes:
    """Canonical bytes the provider signs. Requesters rebuild the same
    payload and verify the provider's signature over it.

    Format per design §136: keccak256(
        "{job_id}||{shard_index}||{output_hash}||{executed_at_unix}"
    ). Matches the on-chain-compatible hashing scheme for Phase 7 slashing.
    """
    raw = (
        f"{job_id}||{shard_index}||{output_hash}||{executed_at_unix}"
    ).encode("utf-8")
    return keccak(raw)


def tee_attestation_audit_dict(
    attestation: Any, tee_type: Any = None,
) -> Optional[Dict[str, Any]]:
    """Sprint 1238 (TEE Tier-3 roadmap C) — build the audit-only
    ``ShardExecutionReceipt.tee_attestation`` dict from a receipt's raw
    attestation BYTES (+ optional ``TEEType``), so the attestation survives to
    the settlement boundary for inspection / re-verification by a settler or
    auditor. Returns ``None`` for an empty attestation.

    PURELY ADDITIVE / audit-only: this field is NOT in the receipt's signing
    payload (``build_receipt_signing_payload``) nor the merkle leaf
    (``batched_receipt_to_leaf``) — it's a Phase-2.1 schema reservation — so
    threading it has ZERO signature / leaf / consensus impact.
    """
    if not attestation:
        return None
    import base64
    blob = bytes(attestation)
    try:
        from prsm.compute.inference.attestation_backends import (
            SOFTWARE_TEE_ATTESTATION_PREFIX,
        )
        is_dev_only = blob.startswith(SOFTWARE_TEE_ATTESTATION_PREFIX)
    except Exception:  # noqa: BLE001
        is_dev_only = False
    try:
        from prsm.compute.inference.multi_stage_attestation import (
            is_multi_stage_attestation,
        )
        is_multi_stage = bool(is_multi_stage_attestation(blob))
    except Exception:  # noqa: BLE001
        is_multi_stage = False
    d: Dict[str, Any] = {
        "attestation_b64": base64.b64encode(blob).decode("ascii"),
        "is_dev_only": is_dev_only,
        "is_multi_stage": is_multi_stage,
    }
    if tee_type is not None:
        d["tee_type"] = getattr(tee_type, "value", str(tee_type))
    return d


def attestation_tier_from_audit_dict(d: Any) -> Any:
    """Sprint 1241 — map a tee_attestation audit dict (sp1238) to an
    AttestationTier. The audit dict carries tee_type + is_dev_only + is_multi_stage
    but NOT vendor_verified, so a real hardware quote maps to HARDWARE_UNVERIFIED
    (the on-chain ceiling) — an auditor re-verifies off-chain against the committed
    measurement. Lazy import keeps compute decoupled from enterprise."""
    from prsm.enterprise.tee_policy import AttestationTier
    if not d or not isinstance(d, dict):
        return AttestationTier.NONE
    if d.get("is_dev_only"):
        return AttestationTier.SOFTWARE  # DEV-ONLY software stub
    tee_type = (d.get("tee_type") or "").lower()
    if tee_type in ("sgx", "tdx", "sev", "secure_enclave"):
        return AttestationTier.HARDWARE_UNVERIFIED
    if tee_type == "software":
        return AttestationTier.SOFTWARE
    return AttestationTier.NONE


def batch_attestation_commitment(audit_dicts: Any) -> bytes:
    """Sprint 1241 (TEE Tier-3 roadmap F off-chain) — deterministic bytes32
    commitment over a batch's per-receipt tee_attestation audit dicts, for the
    on-chain ``commitBatchWithAttestation``. WEAKEST-LINK: the batch tier is the
    LOWEST across receipts (one un-attested receipt collapses the batch). Returns
    ``bytes32(0)`` when no receipt carries an attestation (→ legacy commitBatch).

    Canonical encoding — auditors MUST reproduce it EXACTLY (the chain stores the
    bytes verbatim, never re-hashing):
      per-receipt triple = [tier_rank:int, tee_type:str, is_multi_stage:bool]
      measurement = keccak256(utf8(json.dumps(sorted(triples), separators=(',',':'))))
      commitment  = keccak256(abi.encode(['uint8','bytes32','bool'],
                              [min_tier_rank, measurement, all_multi_stage]))
    """
    import json
    from eth_abi import encode as _abi_encode
    from prsm.enterprise.tee_policy import tier_rank
    dicts = list(audit_dicts or [])
    if not dicts or all(not d for d in dicts):
        return b"\x00" * 32
    triples = sorted(
        [tier_rank(attestation_tier_from_audit_dict(d)),
         ((d or {}).get("tee_type") or ""),
         bool((d or {}).get("is_multi_stage"))]
        for d in dicts
    )
    measurement = keccak(
        json.dumps(triples, separators=(",", ":")).encode("utf-8"))
    min_tier_rank = min(t[0] for t in triples)
    all_multi_stage = all(t[2] for t in triples)
    return keccak(_abi_encode(
        ["uint8", "bytes32", "bool"],
        [min_tier_rank, measurement, all_multi_stage]))


def settlement_job_id(request_id: str, *, streamed: bool) -> str:
    """Deterministic settlement ``job_id`` for an inference, derived purely from
    its ``request_id``.

    Sprint 1157 (on-chain per-stage payee, brick 2.5). The settlement leaf binds
    ``job_id`` into ``build_receipt_signing_payload``; for a worker to sign its
    own per-stage settlement leaf at STAGE-execution time it must be able to
    derive the SAME ``job_id`` the receipt will carry. Previously the receipt
    minted ``job_id`` POST-HOC as ``f"{prefix}-{uuid4().hex[:12]}"`` (random) at
    receipt-build time — the worker, which knows only ``request.request_id`` when
    it executes its stage, could not reproduce it, so any leaf it signed over
    ``request_id`` would mismatch the splitter's rebuild over ``receipt.job_id``
    and be wrongly slashed on challenge. This helper closes that gap.

    Contract:
      * PURE + deterministic — same ``(request_id, streamed)`` -> same ``job_id``.
      * worker-derivable — depends ONLY on ``request_id`` (the inference identity,
        ``InferenceRequest.request_id``) + the ``streamed`` flag, both of which the
        worker/orchestrator has at stage-execution time.
      * uniqueness == ``request_id`` uniqueness — distinct ``request_id`` ->
        distinct ``job_id`` (``request_id`` is already the system's unique
        per-inference correlation key; we do not weaken that invariant).
      * prefix-preserving — ``parallax-job-`` (unary) / ``parallax-stream-job-``
        (streamed), so existing ``job_id.startswith("parallax-job-")`` audit
        filters / tests hold and a relay can't silently swap streamed<->unary.

    Form: the DIRECT form ``f"{prefix}-{request_id}"``. ``request_id`` defaults to
    ``infer-{uuid4().hex[:12]}`` (``InferenceRequest.request_id``) — bounded + URL
    safe — and the e2e path already uses exactly this direct form
    (``parallax-job-{request_id}``), so the direct form matches the established
    precedent and keeps the ``job_id`` human-traceable back to its inference.
    ``request_id`` is bound into the keccak signing preimage by value, so even an
    atypical (longer / punctuation-bearing) caller-supplied ``request_id`` stays
    deterministic + collision-free (keccak accepts any bytes and the preimage is
    1:1 with the unique ``request_id``); a hash form would only obscure the
    traceability without adding safety, so the direct form is preferred.

    Raises ``ValueError`` on an empty/``None`` (or non-``str``) ``request_id`` —
    a settlement leaf must never bind an empty job identity.
    """
    if not isinstance(request_id, str) or not request_id:
        raise ValueError(
            "request_id must be a non-empty string: the settlement job_id is "
            "derived deterministically from it so the worker (which knows only "
            "request_id) and the receipt agree on the same challenge-defensible "
            f"leaf; got {request_id!r}."
        )
    prefix = "parallax-stream-job" if streamed else "parallax-job"
    return f"{prefix}-{request_id}"


def per_stage_leaf_job_id(request_id: str) -> str:
    """Sprint 1158 (on-chain per-stage payee, brick 2 completion) — the
    STREAMED-AGNOSTIC settlement leaf ``job_id`` that BOTH the worker (at stage-
    execution time) AND the per-node splitter (``per_stage_settlement_split``)
    bind into ``build_receipt_signing_payload`` so the worker signs EXACTLY the
    leaf the splitter rebuilds.

    Single source of truth. A multi-stage worker, at the moment it executes its
    stage, has ``request.request_id`` + ``request.chain_stage_index`` but CANNOT
    reliably know the receipt's eventual OUTPUT ``streamed`` flag (that is set
    post-hoc by the orchestrator on the assembled receipt, and the worker's
    transfer streamed-ness — inline blob vs chunked manifest — is a DIFFERENT
    bit). If the per-node leaf bound the streamed-PREFIXED ``settlement_job_id``,
    the worker could pick the wrong prefix and its emitted signature would
    mismatch the splitter's rebuild → a wrongly-slashable leaf. So the per-node
    settlement leaf uses a settlement-INTERNAL, streamed-agnostic id derived from
    ``request_id`` ALONE — ``settlement_job_id(request_id, streamed=False)``.

    This is INTENTIONALLY distinct from the receipt's DISPLAY ``job_id`` (the
    streamed-prefixed sp1157 id the receipt carries and the single-payee adapter
    binds via ``receipt.job_id``). The per-node split and the single-payee path
    never coexist for one inference, so the two ids never collide on one leaf.
    """
    return settlement_job_id(request_id, streamed=False)


def _derive_node_id_from_pubkey_b64(pubkey_b64: str) -> Optional[str]:
    """Recompute NodeIdentity.node_id from an advertised pubkey.

    Must match generate_node_identity(): hex(sha256(public_key_bytes))[:32].
    Returns None on decode failure so callers can treat it as verification
    failure rather than raise.
    """
    try:
        pub_bytes = base64.b64decode(pubkey_b64)
    except Exception:
        return None
    return hashlib.sha256(pub_bytes).hexdigest()[:32]


@dataclass(frozen=True)
class ShardExecutionReceipt:
    """Signed proof that a provider executed a shard.

    Fields form the serialized `receipt` sub-object in a
    shard_execute_response MSG_DIRECT payload.

    The `tee_attestation` field is a Phase 2.1 schema reservation
    (Vision doc §7 Line Item C). When present, it carries a TEE quote
    bound to this specific execution. Phase 2.1 ships the field only;
    verification logic (quote chain validation, revocation check, tier
    gating) lands in a Phase 2.1 follow-up without breaking wire
    format. Current Phase 2 builds leave it None.
    """
    job_id: str
    shard_index: int
    provider_id: str
    provider_pubkey_b64: str
    output_hash: str
    executed_at_unix: int
    signature: str
    tee_attestation: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShardExecutionReceipt":
        return cls(
            job_id=data["job_id"],
            shard_index=data["shard_index"],
            provider_id=data["provider_id"],
            provider_pubkey_b64=data["provider_pubkey_b64"],
            output_hash=data["output_hash"],
            executed_at_unix=data["executed_at_unix"],
            signature=data["signature"],
            tee_attestation=data.get("tee_attestation"),
        )


class VerificationStrategy(Protocol):
    """Pluggable interface for receipt verification. Phase 2 implements
    Tier A (receipt-only). Tiers B (redundant execution) and C (stake +
    slash) implement this protocol at Phase 7."""

    async def verify(
        self,
        receipt: Dict[str, Any],
        output_bytes: bytes,
        expected_provider_id: Optional[str] = None,
    ) -> bool: ...


class ReceiptOnlyVerification:
    """Tier A: signature check only. Phase 2 default.

    Four checks (all must pass):
      1. output_hash == sha256(output_bytes) — the declared hash
         actually matches the bytes returned.
      2. node_id binding: receipt['provider_id'] ==
         hex(sha256(b64decode(provider_pubkey_b64)))[:32]. Closes the
         attack where a receipt claims 'provider A' but carries B's
         pubkey + signature — without this check, verification would
         be self-authenticating against whatever pubkey the receipt
         carries.
      3. expected match (optional): if expected_provider_id is supplied
         by the caller, it must equal receipt['provider_id']. Binds the
         receipt to the node the dispatcher actually sent the request to.
      4. Ed25519(signature) valid against provider_pubkey_b64 over
         build_receipt_signing_payload(...) — the receipt was signed
         by the claimed provider.

    Returns True iff all checks pass. Never raises on invalid input —
    logs a warning and returns False so dispatchers can uniformly
    refund escrow on verification failure.
    """

    async def verify(
        self,
        receipt: Dict[str, Any],
        output_bytes: bytes,
        expected_provider_id: Optional[str] = None,
    ) -> bool:
        declared_hash = receipt.get("output_hash")
        actual_hash = hashlib.sha256(output_bytes).hexdigest()
        if declared_hash != actual_hash:
            logger.warning(
                f"receipt verification failed: output_hash mismatch "
                f"(declared={str(declared_hash)[:16]}…, actual={actual_hash[:16]}…)"
            )
            return False

        try:
            provider_id = receipt["provider_id"]
            pubkey_b64 = receipt["provider_pubkey_b64"]
            signature = receipt["signature"]
            payload = build_receipt_signing_payload(
                job_id=receipt["job_id"],
                shard_index=receipt["shard_index"],
                output_hash=declared_hash,
                executed_at_unix=receipt["executed_at_unix"],
            )
        except KeyError as exc:
            logger.warning(f"receipt verification failed: missing field {exc}")
            return False

        derived_node_id = _derive_node_id_from_pubkey_b64(pubkey_b64)
        if derived_node_id is None or derived_node_id != provider_id:
            logger.warning(
                f"receipt verification failed: provider_id "
                f"{str(provider_id)[:12]}… does not match pubkey-derived "
                f"node_id {str(derived_node_id)[:12]}…"
            )
            return False

        if expected_provider_id is not None and expected_provider_id != provider_id:
            logger.warning(
                f"receipt verification failed: expected provider "
                f"{expected_provider_id[:12]}… but receipt claims "
                f"{str(provider_id)[:12]}…"
            )
            return False

        try:
            from prsm.node.identity import verify_signature
            if not verify_signature(pubkey_b64, payload, signature):
                logger.warning(
                    f"receipt verification failed: signature invalid for "
                    f"provider {str(provider_id)[:12]}…"
                )
                return False
        except ImportError:
            logger.warning(
                "receipt verification failed: verify_signature unavailable"
            )
            return False
        except Exception as exc:
            logger.warning(f"receipt verification raised: {exc}")
            return False

        return True
