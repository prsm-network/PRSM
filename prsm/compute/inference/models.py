"""
Inference data models.

Defines the request/response/receipt types for TEE-attested model inference
exposed via the `prsm_inference` MCP tool. Phase 3.x.1 Task 1 scaffold —
types and serialization only; actual inference execution lands in Tasks 4-5.

Two layers of privacy per `PRSM_Vision.md` §7:
- **Layer 1 — content tier** (A / B / C): encryption status of the data being
  queried. Tier A is public plaintext; B is encrypted-before-sharding;
  C is B plus Reed-Solomon erasure coding plus Shamir-split decryption keys.
- **Layer 2 — privacy tier** (none / standard / high / maximum): TEE attestation
  plus DP noise on activations. Reuses `prsm.compute.tee.PrivacyLevel`.

Both surface as explicit fields on `InferenceRequest`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional
import uuid

from prsm.compute.tee.models import PrivacyLevel, TEEType


class ContentTier(str, Enum):
    """Encryption status of content being queried.

    See `PRSM_Vision.md` §2 "Data layer" for full definitions of each tier.

    - ``A`` — public content; node operators see plaintext shards
    - ``B`` — encrypted-before-sharding; ciphertext-only on operators;
      decryption key released on payment per Phase 7-storage
    - ``C`` — Tier B plus Reed-Solomon erasure coding (K-of-N reconstruction)
      plus Shamir-split decryption keys (M-of-N reconstruction)
    """

    A = "A"
    B = "B"
    C = "C"


# --------------------------------------------------------------------------
# Request
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceRequest:
    """A request to run TEE-attested inference on PRSM.

    Constructed by `prsm_inference` MCP handler from caller arguments; passed
    to ``InferenceExecutor.execute()``. Frozen because requests should be
    immutable once accepted into the dispatch pipeline (any per-shard mutation
    happens on copies).
    """

    prompt: str
    model_id: str
    budget_ftns: Decimal
    privacy_tier: PrivacyLevel = PrivacyLevel.STANDARD
    content_tier: ContentTier = ContentTier.A
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    requester_node_id: Optional[str] = None
    request_id: str = field(default_factory=lambda: f"infer-{uuid.uuid4().hex[:12]}")

    def __post_init__(self) -> None:
        # Frozen dataclass — use object.__setattr__ to coerce types after init
        if not isinstance(self.budget_ftns, Decimal):
            object.__setattr__(self, "budget_ftns", Decimal(str(self.budget_ftns)))
        if not isinstance(self.privacy_tier, PrivacyLevel):
            object.__setattr__(self, "privacy_tier", PrivacyLevel(self.privacy_tier))
        if not isinstance(self.content_tier, ContentTier):
            object.__setattr__(self, "content_tier", ContentTier(self.content_tier))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "model_id": self.model_id,
            "budget_ftns": str(self.budget_ftns),
            "privacy_tier": self.privacy_tier.value,
            "content_tier": self.content_tier.value,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "requester_node_id": self.requester_node_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceRequest":
        d = dict(data)
        if "budget_ftns" in d:
            d["budget_ftns"] = Decimal(str(d["budget_ftns"]))
        if "privacy_tier" in d:
            d["privacy_tier"] = PrivacyLevel(d["privacy_tier"])
        if "content_tier" in d:
            d["content_tier"] = ContentTier(d["content_tier"])
        # Drop unknown keys gracefully so additive schema changes don't break
        # cross-version replays.
        accepted = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in accepted})


# --------------------------------------------------------------------------
# Receipt
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceReceipt:
    """TEE-attested receipt for a single inference execution.

    Per Phase 3.x.1 design plan §3.3, this receipt converts MCP inference
    from "trust the provider" to "verifiable inference":

    1. Caller receives receipt + output
    2. Caller verifies ``settler_signature`` against settling node identity
    3. Caller verifies ``tee_attestation`` against platform vendor's attestation
       service (Intel ASP for SGX/TDX, AMD KDS for SEV-SNP)
    4. Caller confirms ``output_hash`` matches received output

    All four steps run client-side; PRSM does not need to be trusted.
    """

    job_id: str
    request_id: str
    model_id: str
    content_tier: ContentTier
    privacy_tier: PrivacyLevel
    epsilon_spent: float
    tee_type: TEEType
    tee_attestation: bytes
    output_hash: bytes
    duration_seconds: float
    cost_ftns: Decimal
    settler_signature: bytes = b""
    settler_node_id: str = ""
    # Phase 3.x.8: True iff the inference output was streamed
    # token-by-token (``RpcChainExecutor.execute_chain_streaming``)
    # rather than returned in one shot. The flag is part of the
    # signed payload (downgrade-resistant — a relay can't claim a
    # streamed receipt was non-streamed or vice versa). Default
    # False preserves byte-equivalence with pre-3.x.8 receipts so
    # existing callers + verifiers keep working unchanged.
    streamed_output: bool = False
    # Sprint 297 — §7 capstone fields. Optional + default None
    # to preserve byte-equivalence with pre-sprint-297 signed
    # receipts (conditional encoding in signing_payload). When
    # populated, they're cryptographically bound: tampering
    # either field flips the signing bytes → signature fails.
    activation_noise_trace: Optional[Any] = None
    topology_assignment: Optional[Any] = None
    # Sprint 777 — partial-completion marker (Vision §4.5).
    # Attached when an inference completes only some requested
    # tokens (preemption / timeout / error). Optional + default
    # None preserves byte-equivalence with pre-777 signed
    # receipts via conditional encoding in signing_payload.
    partial_completion: Optional[Any] = None
    # Sprint 1099 (Domain-03 review F1, first brick) — sha256 of the INPUT
    # prompt. The receipt already commits to output_hash (the OUTPUT) but
    # nothing tied a signed receipt to the prompt the caller actually sent,
    # so a head node could return a (cheaper / canned) answer computed for a
    # DIFFERENT prompt and the receipt still verified. Binding prompt_hash
    # lets a caller recompute sha256(their prompt) and confirm the receipt
    # commits to THEIR request. Optional + default None preserves
    # byte-equivalence with pre-1099 signed receipts (conditional encoding in
    # signing_payload). (The deeper per-stage activation hash chain is a
    # follow-on brick; this closes the prompt-substitution half of F1.)
    prompt_hash: Optional[bytes] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "job_id": self.job_id,
            "request_id": self.request_id,
            "model_id": self.model_id,
            "content_tier": self.content_tier.value,
            "privacy_tier": self.privacy_tier.value,
            "epsilon_spent": self.epsilon_spent,
            "tee_type": self.tee_type.value,
            "tee_attestation": self.tee_attestation.hex(),
            "output_hash": self.output_hash.hex(),
            "duration_seconds": self.duration_seconds,
            "cost_ftns": str(self.cost_ftns),
            "settler_signature": self.settler_signature.hex(),
            "settler_node_id": self.settler_node_id,
            "streamed_output": self.streamed_output,
        }
        # Sprint 297 — only include new fields when populated
        # so pre-sprint-297 to_dict output is byte-identical.
        if self.activation_noise_trace is not None:
            d["activation_noise_trace"] = (
                self.activation_noise_trace.to_dict()
            )
        if self.topology_assignment is not None:
            d["topology_assignment"] = (
                self.topology_assignment.to_dict()
            )
        # Sprint 777 — omit when None for pre-777 byte-equivalence.
        if self.partial_completion is not None:
            d["partial_completion"] = (
                self.partial_completion.to_dict()
            )
        # Sprint 1099 — omit when None for pre-1099 byte-equivalence.
        if self.prompt_hash is not None:
            d["prompt_hash"] = self.prompt_hash.hex()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceReceipt":
        d = dict(data)
        if "content_tier" in d and not isinstance(d["content_tier"], ContentTier):
            d["content_tier"] = ContentTier(d["content_tier"])
        if "privacy_tier" in d and not isinstance(d["privacy_tier"], PrivacyLevel):
            d["privacy_tier"] = PrivacyLevel(d["privacy_tier"])
        if "tee_type" in d and not isinstance(d["tee_type"], TEEType):
            d["tee_type"] = TEEType(d["tee_type"])
        if "tee_attestation" in d and isinstance(d["tee_attestation"], str):
            d["tee_attestation"] = bytes.fromhex(d["tee_attestation"])
        if "output_hash" in d and isinstance(d["output_hash"], str):
            d["output_hash"] = bytes.fromhex(d["output_hash"])
        if "settler_signature" in d and isinstance(d["settler_signature"], str):
            d["settler_signature"] = bytes.fromhex(d["settler_signature"])
        if "cost_ftns" in d and not isinstance(d["cost_ftns"], Decimal):
            d["cost_ftns"] = Decimal(str(d["cost_ftns"]))
        # Phase 3.x.8 forward-compat: pre-3.x.8 serialized receipts
        # (no ``streamed_output`` key) parse cleanly with the default
        # False. A pre-3.x.8 verifier reading a 3.x.8 receipt still
        # works as long as ``streamed_output`` is False (the signed
        # bytes are byte-identical via the conditional-encoding rule
        # in ``signing_payload``).
        if "streamed_output" in d and not isinstance(
            d["streamed_output"], bool
        ):
            # Hostile peer ships {"streamed_output": 1} — reject as
            # malformed rather than silently coerce. Mirrors the M1
            # bool-rejection invariant on the chain-RPC wire layer.
            raise TypeError(
                f"streamed_output must be bool, got "
                f"{type(d['streamed_output']).__name__}"
            )
        # Sprint 297 — parse the new Optional fields into
        # their dataclass types. Lazy import to keep
        # models.py free of activation_dp / topology_rotation
        # dependencies when callers don't use those features.
        if d.get("activation_noise_trace") is not None and not isinstance(
            d["activation_noise_trace"], dict,
        ) is False:
            # Already a dict — parse it. If it's already an
            # ActivationNoiseTrace pass through.
            if isinstance(d["activation_noise_trace"], dict):
                from prsm.compute.inference.activation_dp import (
                    ActivationNoiseTrace,
                )
                d["activation_noise_trace"] = (
                    ActivationNoiseTrace.from_dict(
                        d["activation_noise_trace"],
                    )
                )
        if d.get("topology_assignment") is not None and isinstance(
            d["topology_assignment"], dict,
        ):
            from prsm.compute.inference.topology_rotation import (
                TopologyAssignment,
            )
            d["topology_assignment"] = (
                TopologyAssignment.from_dict(
                    d["topology_assignment"],
                )
            )
        # Sprint 777 — parse partial_completion when present.
        if d.get("partial_completion") is not None and isinstance(
            d["partial_completion"], dict,
        ):
            from prsm.compute.inference.partial_completion import (
                PartialCompletionInfo,
            )
            d["partial_completion"] = (
                PartialCompletionInfo.from_dict(
                    d["partial_completion"],
                )
            )
        # Sprint 1099 — parse prompt_hash hex → bytes when present.
        if "prompt_hash" in d and isinstance(d["prompt_hash"], str):
            d["prompt_hash"] = bytes.fromhex(d["prompt_hash"])
        accepted = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in accepted})

    def signing_payload(self) -> bytes:
        """Canonical bytes used for ``settler_signature`` generation/verification.

        Excludes ``settler_signature`` itself (would be circular). Field order
        is fixed; do not reorder without bumping a receipt-schema version.

        Phase 3.x.8 conditional encoding: when ``streamed_output`` is
        True, an extra trailing line ``streamed_output:true`` is
        appended. When False (the default), the field is OMITTED
        from the canonical bytes so a 3.x.8 receipt with
        ``streamed_output=False`` produces byte-identical signing
        bytes to a pre-3.x.8 receipt — preserving back-compat for
        callers + verifiers that don't know about the field.
        Tampering the flag (False↔True) flips the bytes →
        signature fails. Downgrade-resistant.
        """
        parts = [
            self.job_id,
            self.request_id,
            self.model_id,
            self.content_tier.value,
            self.privacy_tier.value,
            f"{self.epsilon_spent:.10f}",
            self.tee_type.value,
            self.tee_attestation.hex(),
            self.output_hash.hex(),
            f"{self.duration_seconds:.6f}",
            str(self.cost_ftns),
            self.settler_node_id,
        ]
        if self.streamed_output:
            parts.append("streamed_output:true")
        # Sprint 297 — conditional append for new fields.
        # Encoding uses a canonical hash of the field's
        # serialization so byte-equivalence is preserved
        # for pre-sprint-297 receipts (field absent →
        # nothing appended → bytes match exactly).
        if self.activation_noise_trace is not None:
            import hashlib as _h
            import json as _j
            trace_canon = _j.dumps(
                self.activation_noise_trace.to_dict(),
                sort_keys=True,
            )
            trace_hash = _h.sha256(
                trace_canon.encode("utf-8"),
            ).hexdigest()
            parts.append(
                f"activation_noise_trace:{trace_hash}"
            )
        if self.topology_assignment is not None:
            parts.append(
                f"topology_assignment:"
                f"{self.topology_assignment.stable_hash()}"
            )
        # Sprint 777 — conditional append for partial_completion.
        # stable_hash() = sha256 of canonical JSON. Tampering any
        # field flips the hash → signature fails.
        if self.partial_completion is not None:
            parts.append(
                f"partial_completion:"
                f"{self.partial_completion.stable_hash()}"
            )
        # Sprint 1099 — conditional append for prompt_hash. Binds the input
        # prompt to the signature so a substituted prompt flips the bytes →
        # signature fails. Absent (None) → nothing appended → byte-identical
        # to a pre-1099 receipt (back-compat for legacy receipts + verifiers).
        if self.prompt_hash is not None:
            parts.append(f"prompt_hash:{self.prompt_hash.hex()}")
        return "\n".join(parts).encode("utf-8")


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceResult:
    """Output of one inference execution, including the verifiable receipt.

    On success: ``output`` is the model's response and ``receipt`` carries the
    TEE attestation + signed settlement evidence. On failure: ``error`` is a
    human-readable description, ``output`` is empty, and ``receipt`` is None
    (no FTNS settled).
    """

    request_id: str
    success: bool
    output: str = ""
    receipt: Optional[InferenceReceipt] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "success": self.success,
            "output": self.output,
            "receipt": self.receipt.to_dict() if self.receipt else None,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceResult":
        d = dict(data)
        if d.get("receipt"):
            d["receipt"] = InferenceReceipt.from_dict(d["receipt"])
        else:
            d["receipt"] = None
        accepted = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in accepted})

    @classmethod
    def failure(cls, request_id: str, error: str) -> "InferenceResult":
        """Convenience constructor for failed inference."""
        return cls(request_id=request_id, success=False, error=error)
