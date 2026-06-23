"""Phase 3.x.7 Task 5 — multi-stage TEE attestation aggregation.

Cross-host inference produces one TEE attestation per chain stage.
The Phase 3.x.1 ``InferenceReceipt`` was designed for single-host
inference: one ``tee_type`` + one ``tee_attestation`` blob. Extending
the schema directly would break backward compatibility with the
Phase 2 Ring 8 + Phase 3.x.1 callers that already produce single-stage
receipts.

This module's design:

  - Single-stage receipts continue to use ``tee_attestation`` as raw
    opaque bytes (whatever the platform vendor's attestation report
    contains). Existing callers see no change.

  - Multi-stage receipts encode a per-stage list inside the same
    ``tee_attestation`` field, prefixed with a recognizable magic
    string. Receivers detect the prefix; if absent, the bytes are
    treated as opaque single-stage attestation. If present, the
    JSON envelope is decoded and per-stage entries can be iterated.

  - The receipt's top-level ``tee_type`` is the **worst-case** TEE
    type across stages. SOFTWARE drags hardware down — one software
    stage in an SGX chain → receipt records SOFTWARE. This is the
    conservative policy: verifiers should treat the whole chain as
    only as trustworthy as its weakest link. Operators wanting
    per-stage trust can iterate ``decode_multi_stage_attestation``
    explicitly.

The encoding is intentionally backward-compatible — the
``InferenceReceipt`` dataclass needs no changes, and
``signing_payload()`` continues to hex-encode whatever bytes are in
``tee_attestation`` (so the Ed25519 signature commits the settler to
the per-stage list when the envelope is in use).

What this module does NOT do:
  - It does NOT verify per-stage attestations against platform-vendor
    services (Intel ASP, AMD KDS, etc.). That wiring is post-MVP;
    this module exposes structural validation hooks but the actual
    vendor RPC is out of scope for v1.
  - It does NOT enforce that all stages are present / contiguous;
    callers (the ``RpcChainExecutor``) own that contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from prsm.compute.tee.models import HARDWARE_TEE_TYPES, TEEType


# DecodeMode imported lazily inside ``IterationAttestation`` to avoid
# the circular import ``inference.__init__`` → ``multi_stage_attestation``
# → ``chain_rpc.protocol`` → ``inference.models`` (chain_rpc imports
# ContentTier from inference at module load).


# ──────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────


MULTI_STAGE_ATTESTATION_VERSION = 1
"""Wire-format version for the multi-stage envelope. Bump when any
field shape changes."""

MULTI_STAGE_MAGIC_PREFIX = b"PRSM-MS-ATT-V1:"
"""Recognizable byte sequence that marks the JSON envelope. Single-
stage attestations cannot start with this string by convention; if
they do (vanishingly unlikely for real platform-vendor reports), the
caller MUST add a randomization byte at the head — but no real TEE
report we've inspected starts with PRSM-prefixed ASCII."""

MULTI_ITERATION_ATTESTATION_VERSION = 1
"""Phase 3.x.11.x — wire-format version for the multi-iteration
envelope (per-token attestation chain for sharded autoregressive
decode). Bump when any field shape changes."""

MULTI_ITERATION_MAGIC_PREFIX = b"PRSM-MI-ATT-V1:"
"""Phase 3.x.11.x — recognizable byte sequence marking a
multi-iteration envelope. Distinct from MULTI_STAGE_MAGIC_PREFIX
so receipts produced by the non-sharded path stay byte-equivalent
with pre-3.x.11.x receipts (the discriminator at the magic-prefix
level is more robust than at the JSON-key level — old decoders
that read the existing PRSM-MS-ATT envelope return None on the
new prefix rather than raising, which is the intentional
back-compat path)."""


# ──────────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────────


class MultiStageAttestationError(Exception):
    """Base for multi-stage attestation envelope failures."""


class MultiStageMalformedError(MultiStageAttestationError):
    """Envelope bytes failed parse / structural validation."""


# ──────────────────────────────────────────────────────────────────────────
# Per-stage attestation record
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StageAttestation:
    """One stage's contribution to the multi-stage envelope.

    Fields:
      stage_index     0-based position in the chain. Receivers can
                      detect missing / out-of-order stages.
      stage_node_id   Identity that produced the attestation. Receivers
                      use this to look up the stage's pubkey on the
                      Phase 3.x.3 anchor when verifying the per-stage
                      RunLayerSliceResponse signature in addition to
                      the platform-vendor attestation.
      tee_type        Per-stage TEE type. Aggregate worst-case across
                      stages drives the receipt's top-level tee_type.
      attestation     Raw bytes — opaque to this layer; the platform
                      vendor's attestation service interprets them.
    """

    stage_index: int
    stage_node_id: str
    tee_type: TEEType
    attestation: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.stage_index, int) or self.stage_index < 0:
            raise MultiStageMalformedError(
                f"stage_index must be non-negative int, got {self.stage_index!r}"
            )
        if not isinstance(self.stage_node_id, str) or not self.stage_node_id:
            raise MultiStageMalformedError(
                f"stage_node_id must be non-empty str, got {self.stage_node_id!r}"
            )
        if not isinstance(self.tee_type, TEEType):
            raise MultiStageMalformedError(
                f"tee_type must be TEEType, got {type(self.tee_type).__name__}"
            )
        if not isinstance(self.attestation, (bytes, bytearray)):
            raise MultiStageMalformedError(
                f"attestation must be bytes, got {type(self.attestation).__name__}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_index": self.stage_index,
            "stage_node_id": self.stage_node_id,
            "tee_type": self.tee_type.value,
            "attestation_hex": bytes(self.attestation).hex(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageAttestation":
        if not isinstance(data, dict):
            raise MultiStageMalformedError(
                f"stage entry must be dict, got {type(data).__name__}"
            )
        try:
            stage_index = int(data["stage_index"])
            stage_node_id = str(data["stage_node_id"])
            tee_type = TEEType(data["tee_type"])
            attestation = bytes.fromhex(data["attestation_hex"])
        except (KeyError, ValueError, TypeError) as exc:
            raise MultiStageMalformedError(
                f"stage entry parse failed: {exc}"
            ) from exc
        return cls(
            stage_index=stage_index,
            stage_node_id=stage_node_id,
            tee_type=tee_type,
            attestation=attestation,
        )


# ──────────────────────────────────────────────────────────────────────────
# Envelope codec
# ──────────────────────────────────────────────────────────────────────────


def encode_multi_stage_attestation(stages: List[StageAttestation]) -> bytes:
    """Encode a per-stage list as a magic-prefixed JSON envelope.

    Empty list raises — a chain with zero stages is a caller bug, not
    a valid receipt state. Callers with single-host inference should
    continue to populate ``tee_attestation`` with raw opaque bytes
    (no envelope) — that's the back-compat path.
    """
    if not stages:
        raise MultiStageAttestationError(
            "encode_multi_stage_attestation requires at least one stage"
        )
    payload = {
        "version": MULTI_STAGE_ATTESTATION_VERSION,
        "stages": [s.to_dict() for s in stages],
    }
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    return MULTI_STAGE_MAGIC_PREFIX + body


def is_multi_stage_attestation(blob: bytes) -> bool:
    """True iff ``blob`` carries the multi-stage magic prefix.

    Cheap O(1) check the receipt-verification layer uses to decide
    whether to invoke ``decode_multi_stage_attestation`` or treat
    ``blob`` as opaque single-stage bytes.
    """
    if not isinstance(blob, (bytes, bytearray)):
        return False
    return bytes(blob).startswith(MULTI_STAGE_MAGIC_PREFIX)


def decode_multi_stage_attestation(
    blob: bytes,
    *,
    expected_stage_count: Optional[int] = None,
) -> Optional[List[StageAttestation]]:
    """Decode a multi-stage envelope into a list of stage attestations.

    Returns ``None`` if ``blob`` doesn't carry the magic prefix —
    indicates a single-stage receipt where the bytes are raw vendor
    attestation. Callers MUST handle the ``None`` case (typically by
    treating the bytes as one opaque attestation).

    Raises ``MultiStageMalformedError`` on a magic-prefixed but
    structurally invalid envelope. The boundary is intentional:
    "no envelope" is a normal back-compat path; "envelope present
    but broken" is a corruption signal worth raising.

    Structural validation enforced:
      - ``stages`` is a non-empty list
      - ``stage_index`` values are exactly 0..N-1 (contiguous, no
        gaps)
      - no duplicate ``stage_index``
      - if ``expected_stage_count`` is given, ``len(stages)`` must
        match it (defends against a settler omitting a SOFTWARE stage
        from the envelope to upgrade the receipt's apparent worst-
        case TEE type)
    """
    if not is_multi_stage_attestation(blob):
        return None
    body = bytes(blob)[len(MULTI_STAGE_MAGIC_PREFIX):]
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MultiStageMalformedError(
            f"envelope JSON parse failed: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise MultiStageMalformedError(
            f"envelope top-level must be dict, got {type(data).__name__}"
        )
    version = data.get("version")
    if version != MULTI_STAGE_ATTESTATION_VERSION:
        raise MultiStageMalformedError(
            f"envelope version {version!r} != local "
            f"{MULTI_STAGE_ATTESTATION_VERSION}"
        )
    raw_stages = data.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise MultiStageMalformedError(
            "envelope must contain non-empty 'stages' list"
        )
    stages = [StageAttestation.from_dict(s) for s in raw_stages]

    # Structural integrity: stage_index values must be exactly
    # 0..N-1 (contiguous, no gaps, no duplicates). Otherwise a
    # settler could omit a SOFTWARE stage to upgrade the apparent
    # worst-case TEE type seen by a receipt verifier.
    indices = [s.stage_index for s in stages]
    if len(set(indices)) != len(indices):
        raise MultiStageMalformedError(
            f"envelope contains duplicate stage_index values: {sorted(indices)}"
        )
    if sorted(indices) != list(range(len(stages))):
        raise MultiStageMalformedError(
            f"envelope stage_index values must be 0..{len(stages) - 1} "
            f"(contiguous); got {sorted(indices)}"
        )
    if expected_stage_count is not None and len(stages) != expected_stage_count:
        raise MultiStageMalformedError(
            f"envelope has {len(stages)} stages; "
            f"caller expected {expected_stage_count}"
        )

    # Return stages sorted by stage_index for deterministic iteration.
    return sorted(stages, key=lambda s: s.stage_index)


# ──────────────────────────────────────────────────────────────────────────
# Worst-case TEE type
# ──────────────────────────────────────────────────────────────────────────


_WORST_CASE_RANK: Dict[TEEType, int] = {
    TEEType.SOFTWARE: 0,
}
"""Lower rank = worse confidentiality. Only SOFTWARE has a non-default
rank; all hardware TEE types share rank 1 because at this level of
granularity (chain-level worst-case) we don't differentiate among
SGX/TDX/SEV/etc.

If a stage's ``tee_type`` is not in this table, it falls through to
rank 1 (treated as hardware-equivalent). NONE-tier or unknown types
should not appear in this list because they're filtered at the tier-
gate earlier in the chain (Phase 3.x.6 ``TierGateAdapter``)."""


def worst_case_tee_type(stages: List[StageAttestation]) -> TEEType:
    """Return the worst (lowest-confidentiality) TEE type across stages.

    Empty list returns SOFTWARE (the most conservative default). When
    multiple stages share the worst rank, the FIRST stage's type wins
    by tie-break — gives callers a deterministic top-level
    ``tee_type`` for receipts.
    """
    if not stages:
        return TEEType.SOFTWARE
    worst = stages[0].tee_type
    worst_rank = _WORST_CASE_RANK.get(worst, 1)
    for stage in stages[1:]:
        rank = _WORST_CASE_RANK.get(stage.tee_type, 1)
        if rank < worst_rank:
            worst = stage.tee_type
            worst_rank = rank
    return worst


def is_hardware_tee(tee_type: TEEType) -> bool:
    """Convenience: True iff ``tee_type`` is in the hardware-backed
    set (matches Phase 3.x.6 ``TierGateAdapter`` policy)."""
    return tee_type.value in HARDWARE_TEE_TYPES


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.11.x — per-iteration attestation chain
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class IterationAttestation:
    """Phase 3.x.11.x — one chain-iteration's worth of per-stage
    attestations.

    The Phase 3.x.7 single-iteration envelope (``StageAttestation[]``
    flat list) commits the receipt to "one record per chain stage,
    captured from the LAST iteration's per-stage outcomes". For
    sharded autoregressive decode, this loses signal: a malicious
    stage that swapped its KV-cache mid-stream would NOT be detected
    by the receipt's signature chain (only by R7 challenge-and-verify
    if the operator wires it).

    The multi-iteration envelope commits the receipt to one record
    per stage PER iteration. Verifiers reading the receipt can
    confirm "stage K served EVERY incremental dispatch, not just
    the last one".

    Fields:
      iteration_index   0-based; 0 = PREFILL, 1+ = INCREMENTAL or
                        VERIFY (Phase 3.x.11.y speculative
                        decoding). Receivers can detect missing /
                        out-of-order iterations.
      decode_mode       The ``DecodeMode`` for this iteration.
                        ``iteration_index == 0`` → PREFILL;
                        subsequent iterations are INCREMENTAL
                        (Phase 3.x.11) or VERIFY (Phase 3.x.11.y).
      stage_records     Per-stage attestations for this iteration.
                        Same shape as the existing single-iteration
                        envelope's ``StageAttestation[]``.
    """

    iteration_index: int
    decode_mode: Any  # DecodeMode — lazy import; runtime checked
    stage_records: List[StageAttestation]

    def __post_init__(self) -> None:
        from prsm.compute.chain_rpc.protocol import DecodeMode
        if (
            isinstance(self.iteration_index, bool)
            or not isinstance(self.iteration_index, int)
            or self.iteration_index < 0
        ):
            raise MultiStageMalformedError(
                f"iteration_index must be non-negative int, got "
                f"{self.iteration_index!r}"
            )
        if not isinstance(self.decode_mode, DecodeMode):
            raise MultiStageMalformedError(
                f"decode_mode must be DecodeMode, got "
                f"{type(self.decode_mode).__name__}"
            )
        if not isinstance(self.stage_records, list) or not self.stage_records:
            raise MultiStageMalformedError(
                "stage_records must be a non-empty list of "
                "StageAttestation"
            )
        for s in self.stage_records:
            if not isinstance(s, StageAttestation):
                raise MultiStageMalformedError(
                    f"stage_records entries must be StageAttestation, "
                    f"got {type(s).__name__}"
                )
        # Iteration 0 must be PREFILL; subsequent iterations must be
        # INCREMENTAL. Catches a settler bug where the decode_mode
        # gets out of sync with the iteration index.
        if self.iteration_index == 0 and self.decode_mode != DecodeMode.PREFILL:
            raise MultiStageMalformedError(
                f"iteration_index=0 requires decode_mode=PREFILL, "
                f"got {self.decode_mode.value!r}"
            )
        if self.iteration_index > 0 and self.decode_mode not in (
            DecodeMode.INCREMENTAL,
            DecodeMode.VERIFY,
        ):
            raise MultiStageMalformedError(
                f"iteration_index>{0} requires decode_mode in "
                f"{{INCREMENTAL, VERIFY}}, got "
                f"{self.decode_mode.value!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration_index": self.iteration_index,
            "decode_mode": self.decode_mode.value,
            "stage_records": [s.to_dict() for s in self.stage_records],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IterationAttestation":
        from prsm.compute.chain_rpc.protocol import DecodeMode
        if not isinstance(data, dict):
            raise MultiStageMalformedError(
                f"iteration entry must be dict, got {type(data).__name__}"
            )
        try:
            iteration_index = int(data["iteration_index"])
            decode_mode = DecodeMode(data["decode_mode"])
            raw_stages = data["stage_records"]
        except (KeyError, ValueError, TypeError) as exc:
            raise MultiStageMalformedError(
                f"iteration entry parse failed: {exc}"
            ) from exc
        if not isinstance(raw_stages, list):
            raise MultiStageMalformedError(
                f"stage_records must be list, got {type(raw_stages).__name__}"
            )
        stage_records = [StageAttestation.from_dict(s) for s in raw_stages]
        return cls(
            iteration_index=iteration_index,
            decode_mode=decode_mode,
            stage_records=stage_records,
        )


def encode_multi_iteration_attestation(
    iterations: List[IterationAttestation],
) -> bytes:
    """Encode a per-iteration list as a magic-prefixed JSON envelope.

    Empty list raises — a sharded receipt with zero iterations is a
    caller bug. Non-sharded callers (Phase 3.x.7+ unary, Phase
    3.x.8+ streaming-tail) MUST keep using
    ``encode_multi_stage_attestation`` — the existing envelope is
    unchanged and byte-equivalent with pre-3.x.11.x receipts.
    """
    if not iterations:
        raise MultiStageAttestationError(
            "encode_multi_iteration_attestation requires at least "
            "one iteration"
        )
    payload = {
        "version": MULTI_ITERATION_ATTESTATION_VERSION,
        "iterations": [it.to_dict() for it in iterations],
    }
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    return MULTI_ITERATION_MAGIC_PREFIX + body


def is_multi_iteration_attestation(blob: bytes) -> bool:
    """True iff ``blob`` carries the multi-iteration magic prefix."""
    if not isinstance(blob, (bytes, bytearray)):
        return False
    return bytes(blob).startswith(MULTI_ITERATION_MAGIC_PREFIX)


def decode_multi_iteration_attestation(
    blob: bytes,
    *,
    expected_iteration_count: Optional[int] = None,
    expected_stage_count: Optional[int] = None,
) -> Optional[List[IterationAttestation]]:
    """Decode a multi-iteration envelope into a list of iterations.

    Returns ``None`` if ``blob`` doesn't carry the multi-iteration
    magic prefix — indicates the receipt is non-sharded (or sharded
    but produced by a pre-3.x.11.x settler). Callers handle the
    None case by falling back to ``decode_multi_stage_attestation``.

    Raises ``MultiStageMalformedError`` on a magic-prefixed but
    structurally invalid envelope. The boundary is intentional:
    "no envelope" is a normal back-compat path; "envelope present
    but broken" is a corruption signal worth raising.

    Structural validation enforced:
      - ``iterations`` is a non-empty list
      - ``iteration_index`` values are exactly 0..N-1 (contiguous,
        no gaps, no duplicates) — defends against a settler omitting
        an iteration to upgrade the apparent worst-case TEE type
      - within each iteration, ``stage_records`` is a non-empty list
        and ``stage_index`` values are exactly 0..M-1 (same
        contract as ``decode_multi_stage_attestation``)
      - all iterations have the SAME stage count (a sharded chain
        doesn't change shape mid-stream)
      - if ``expected_iteration_count`` is given, must match
      - if ``expected_stage_count`` is given, every iteration's
        stage count must match
    """
    if not is_multi_iteration_attestation(blob):
        return None
    body = bytes(blob)[len(MULTI_ITERATION_MAGIC_PREFIX):]
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MultiStageMalformedError(
            f"iteration envelope JSON parse failed: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise MultiStageMalformedError(
            f"iteration envelope top-level must be dict, got "
            f"{type(data).__name__}"
        )
    version = data.get("version")
    if version != MULTI_ITERATION_ATTESTATION_VERSION:
        raise MultiStageMalformedError(
            f"iteration envelope version {version!r} != local "
            f"{MULTI_ITERATION_ATTESTATION_VERSION}"
        )
    raw_iterations = data.get("iterations")
    if not isinstance(raw_iterations, list) or not raw_iterations:
        raise MultiStageMalformedError(
            "iteration envelope must contain non-empty 'iterations' list"
        )
    iterations = [IterationAttestation.from_dict(it) for it in raw_iterations]

    # Iteration_index values must be exactly 0..N-1 (contiguous,
    # no gaps, no duplicates).
    indices = [it.iteration_index for it in iterations]
    if len(set(indices)) != len(indices):
        raise MultiStageMalformedError(
            f"iteration envelope contains duplicate iteration_index "
            f"values: {sorted(indices)}"
        )
    if sorted(indices) != list(range(len(iterations))):
        raise MultiStageMalformedError(
            f"iteration envelope iteration_index values must be "
            f"0..{len(iterations) - 1} (contiguous); "
            f"got {sorted(indices)}"
        )

    # All iterations must have the same stage count.
    stage_counts = {len(it.stage_records) for it in iterations}
    if len(stage_counts) != 1:
        raise MultiStageMalformedError(
            f"iteration envelope stage counts must be uniform across "
            f"iterations; got {sorted(stage_counts)}"
        )
    stage_count_actual = stage_counts.pop()

    # Within each iteration, stage_index values must be exactly
    # 0..M-1 (same contract as the single-iteration envelope).
    for it in iterations:
        si = [s.stage_index for s in it.stage_records]
        if len(set(si)) != len(si):
            raise MultiStageMalformedError(
                f"iteration {it.iteration_index} contains duplicate "
                f"stage_index values: {sorted(si)}"
            )
        if sorted(si) != list(range(len(it.stage_records))):
            raise MultiStageMalformedError(
                f"iteration {it.iteration_index} stage_index values "
                f"must be 0..{len(it.stage_records) - 1} "
                f"(contiguous); got {sorted(si)}"
            )

    if (
        expected_iteration_count is not None
        and len(iterations) != expected_iteration_count
    ):
        raise MultiStageMalformedError(
            f"iteration envelope has {len(iterations)} iterations; "
            f"caller expected {expected_iteration_count}"
        )
    if (
        expected_stage_count is not None
        and stage_count_actual != expected_stage_count
    ):
        raise MultiStageMalformedError(
            f"iteration envelope has {stage_count_actual} stages "
            f"per iteration; caller expected {expected_stage_count}"
        )

    # Return iterations sorted by iteration_index for deterministic
    # iteration; within each iteration, stage_records are sorted
    # by stage_index.
    iterations_sorted = sorted(
        iterations, key=lambda it: it.iteration_index,
    )
    return [
        IterationAttestation(
            iteration_index=it.iteration_index,
            decode_mode=it.decode_mode,
            stage_records=sorted(
                it.stage_records, key=lambda s: s.stage_index,
            ),
        )
        for it in iterations_sorted
    ]


def worst_case_tee_type_across_iterations(
    iterations: List[IterationAttestation],
) -> TEEType:
    """Return the worst (lowest-confidentiality) TEE type across
    ALL stages in ALL iterations. Sharded receipts use this to
    populate the receipt's top-level ``tee_type`` — a single
    SOFTWARE stage in any iteration drags the whole receipt to
    SOFTWARE.

    Empty list returns SOFTWARE (most conservative default).
    """
    if not iterations:
        return TEEType.SOFTWARE
    flat = [s for it in iterations for s in it.stage_records]
    return worst_case_tee_type(flat)


# ──────────────────────────────────────────────────────────────────────────
# Verification helpers
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StageVerificationResult:
    """Outcome of verifying one stage's attestation.

    Fields:
      stage_index       Echoed from the envelope, or ``-1`` for the
                        single-stage opaque-bytes back-compat path.
      stage_node_id     Echoed from the envelope, or empty string for
                        the back-compat path.
      tee_type          The stage's claimed TEE type. For
                        ``is_placeholder=True`` this is ``SOFTWARE``
                        as a conservative default — callers MUST NOT
                        treat this as a real software-tier event.
      structurally_ok   True iff the envelope entry parsed cleanly.
                        v1 stops here; v2 wires platform-vendor RPC.
      vendor_verified   None for v1 (no vendor service wired). v2:
                        True iff the platform-vendor attestation
                        service confirmed the report.
      is_placeholder    True iff this result represents the single-
                        stage opaque-bytes back-compat path (no
                        multi-stage envelope present). Monitoring
                        layers aggregating ``tee_type`` counts MUST
                        skip placeholder results.
      message           Free-form description of the verification
                        outcome (mainly for failure paths).
    """

    stage_index: int
    stage_node_id: str
    tee_type: TEEType
    structurally_ok: bool
    vendor_verified: Optional[bool]
    message: str
    is_placeholder: bool = False


def verify_stage_attestations(
    blob: bytes,
    *,
    registry: Any = None,
    expected_node_ids: Optional[List[str]] = None,
    expected_stage_count: Optional[int] = None,
) -> Tuple[bool, List[StageVerificationResult]]:
    """Iterate the multi-stage envelope and produce per-stage results.

    Returns ``(all_ok, results)``:
      - ``all_ok``: with NO ``registry`` (default), True iff every stage
        parsed cleanly (structural-only — byte-identical to pre-sp1236).
        With a ``registry`` supplied (sp1236), True iff EVERY stage is
        structurally valid AND its attestation is cryptographically
        ``vendor_verified`` by the registry AND the verified quote's
        REPORT_DATA binds to the stage's ``stage_node_id`` (anti-replay,
        the sp1083 single-stage binding now on the multi-stage chain)
        AND (when ``expected_node_ids`` is given) the stage's node matches
        the expected topology position (anti-substitution). Fail-closed.

    Args (sp1236, all opt-in — defaults preserve pre-sp1236 behavior):
      registry             AttestationBackendRegistry (or any object with
                           ``.verify(blob) -> result`` exposing
                           ``vendor_verified`` + ``vendor_data``). Build the
                           real one via ``configure_default_registry_from_env``.
      expected_node_ids    Caller's known per-stage node_ids (anti-substitution).
      expected_stage_count Reject a chain with a different stage count
                           (anti-omission); threaded to the decoder.
      - ``results`` is a list (one per stage) with the per-stage
        verification outcome.

    Single-stage opaque-bytes attestations return
    ``(True, [StageVerificationResult(stage_index=0, ...)])`` with the
    structural-ok flag True and ``vendor_verified=None``. Callers who
    need to distinguish single-stage from multi-stage should call
    ``is_multi_stage_attestation(blob)`` first.

    Malformed multi-stage envelopes (magic-prefixed but corrupt)
    return ``(False, [single error result])`` rather than raising —
    the verification API is non-throwing by contract, mirrors
    Phase 3.x.5 / 3.x.6 server-side "never raises" patterns.
    """
    if not is_multi_stage_attestation(blob):
        # Treat as single-stage opaque attestation. We can't infer the
        # stage_node_id or tee_type from raw bytes; report a placeholder
        # entry (is_placeholder=True) so callers can distinguish it
        # from a real multi-stage entry. Monitoring layers aggregating
        # tee_type counts MUST skip placeholder results.
        return True, [StageVerificationResult(
            stage_index=-1,
            stage_node_id="",
            tee_type=TEEType.SOFTWARE,
            structurally_ok=True,
            vendor_verified=None,
            message="single-stage opaque attestation (no envelope)",
            is_placeholder=True,
        )]

    try:
        stages = decode_multi_stage_attestation(
            blob, expected_stage_count=expected_stage_count,
        )
    except MultiStageAttestationError as exc:
        return False, [StageVerificationResult(
            stage_index=-1,
            stage_node_id="",
            tee_type=TEEType.SOFTWARE,
            structurally_ok=False,
            vendor_verified=None,
            message=f"envelope decode failed: {exc}",
        )]

    if stages is None:
        # Should be unreachable given the is_multi_stage_attestation
        # guard above, but treat defensively.
        return False, [StageVerificationResult(
            stage_index=-1,
            stage_node_id="",
            tee_type=TEEType.SOFTWARE,
            structurally_ok=False,
            vendor_verified=None,
            message="decode returned None despite magic prefix present",
        )]

    # sp1236 — NO registry: byte-identical to the pre-sp1236 structural-only
    # path (vendor_verified=None, all_ok = structural). Verification is strictly
    # opt-in so existing callers + the worst-case tee_type signing are unchanged.
    if registry is None:
        results = [StageVerificationResult(
            stage_index=s.stage_index,
            stage_node_id=s.stage_node_id,
            tee_type=s.tee_type,
            structurally_ok=True,
            vendor_verified=None,
            message="structurally valid; vendor verification not wired (no registry)",
        ) for s in stages]
        return True, results

    # sp1236 — with a registry: run each stage's attestation through the real
    # backend verifier (the same IntelDCAPBackend/AMDSEVSNPBackend the
    # single-stage path uses) AND bind the verified quote to the stage's node.
    from prsm.compute.inference.attestation_backends import (
        expected_attestation_report_data,
    )
    results = []
    all_ok = True
    for i, s in enumerate(stages):
        # anti-substitution: the stage's claimed node must match the caller's
        # expected topology position (a settler can't relabel a stage).
        node_ok = expected_node_ids is None or (
            i < len(expected_node_ids)
            and s.stage_node_id == expected_node_ids[i]
        )
        try:
            res = registry.verify(s.attestation)
            vendor_verified = bool(getattr(res, "vendor_verified", False))
            rdh = ((getattr(res, "vendor_data", None) or {}).get(
                "report_data_hex") or "").lower()
            vendor = getattr(res, "vendor", "unknown")
        except Exception as exc:  # noqa: BLE001 — a backend must never crash chain verify
            vendor_verified, rdh, vendor = False, "", f"verify-error:{type(exc).__name__}"
        # anti-replay: the verified quote must carry THIS node's commitment in the
        # first 32 bytes of REPORT_DATA (the remaining 32 are opaque to this layer).
        # Matches the sp1083 single-stage binding (.lower() + len(expected)). A backend
        # that returns vendor_verified=True MUST populate vendor_data["report_data_hex"]
        # (IntelDCAPBackend + AMDSEVSNPBackend do; structural backends return
        # vendor_verified=False and are gated out before this matters).
        expected_rd = expected_attestation_report_data(s.stage_node_id)
        bound = bool(rdh) and rdh[:len(expected_rd)] == expected_rd
        if not (vendor_verified and bound and node_ok):
            all_ok = False
        results.append(StageVerificationResult(
            stage_index=s.stage_index,
            stage_node_id=s.stage_node_id,
            tee_type=s.tee_type,
            structurally_ok=True,
            vendor_verified=vendor_verified,
            message=(
                f"vendor={vendor} vendor_verified={vendor_verified} "
                f"report_data_bound={bound} node_ok={node_ok}"
            ),
        ))
    return all_ok, results
