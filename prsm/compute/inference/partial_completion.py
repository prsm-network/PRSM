"""Sprint 777 — partial-completion marker for InferenceReceipt.

Vision §4.5: PRSM's signed-receipt protocol must support
partial-completion credit + re-routing. Preemption should NOT
result in operator slashing; abandonment should.

This module ships the wire-format primitive: a `PartialCompletionInfo`
dataclass that an inference runner can attach to an InferenceReceipt
when the runner finishes only some of the requested tokens before
going down (or hitting a timeout, or any other partial-completion
condition).

Settlement-side credit policy is OUT OF SCOPE — that lives in
the settler / royalty-distributor logic and consults this field
when deciding how much to credit + whether to slash. Sprint 777
just ships the field + its byte-stable signing-payload contribution
so a downstream settlement contract can rely on a tamper-evident
record of what the operator claims.

Tamper resistance:
- All four fields (reason / tokens_completed / tokens_requested /
  timestamp) feed the receipt's signing_payload via a canonical
  JSON-sorted-keys hash (sprint 297's pattern). Tampering any
  field flips the bytes → signature fails verification.
- A preempted operator CAN'T relabel `reason="preempted"` as
  `reason="error"` (or vice versa) after signing.
- An operator CAN'T claim more tokens_completed than they
  actually produced and present a valid signature.

Reason values:
- "preempted"  — cloud-spot termination notice (sprint 772 detector)
- "timeout"    — operator's max-duration exceeded mid-inference
- "error"      — runtime fault (OOM, model load, etc.)

VALID_REASONS lists these; constructor accepts any string for
forward-compat (settlement-side validates).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet


VALID_REASONS: FrozenSet[str] = frozenset({
    "preempted",
    "timeout",
    "error",
})


@dataclass(frozen=True)
class PartialCompletionInfo:
    """Records that an inference completed only some of the
    requested tokens. Attached to an InferenceReceipt by the
    inference runner when partial completion happens.

    Frozen so receipts can be compared by value + the signing
    payload's hash stays stable through copy."""

    reason: str
    tokens_completed: int
    tokens_requested: int
    timestamp: str  # ISO-8601 UTC, e.g. "2026-05-23T12:00:00Z"

    def to_dict(self) -> Dict[str, Any]:
        # sp1098 (Domain-03 review F5) — coerce timestamp to str SYMMETRICALLY with
        # from_dict. `timestamp` is a str field (ISO-8601 UTC), but if a non-str ever
        # leaks in (e.g. a float epoch), to_dict-without-coercion + from_dict-with-coercion
        # diverge → the canonical signing payload is NOT byte-stable across a
        # to_dict→from_dict round-trip → an honest partial receipt falsely fails
        # signature verification. Coercing here keeps both ends symmetric.
        return {
            "reason": str(self.reason),
            "tokens_completed": int(self.tokens_completed),
            "tokens_requested": int(self.tokens_requested),
            "timestamp": str(self.timestamp),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PartialCompletionInfo":
        return cls(
            reason=str(data["reason"]),
            tokens_completed=int(data["tokens_completed"]),
            tokens_requested=int(data["tokens_requested"]),
            timestamp=str(data["timestamp"]),
        )

    def stable_hash(self) -> str:
        """Canonical JSON-sorted-keys SHA-256 hex digest used by
        InferenceReceipt.signing_payload (sprint 297 pattern).
        Stable across re-serialization."""
        canon = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()
