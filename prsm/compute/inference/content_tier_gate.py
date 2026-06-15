"""
Content Tier Gate — enforces Tier B/C decryption only inside a TEE.
==================================================================

Per Phase 3.x.1 design plan §4 Task 3 + ``PRSM_Vision.md`` §2 (data
layer confidentiality):

- **Tier A** — public content; pass-through, no TEE required.
- **Tier B** — encrypted-before-sharding (AES-256-GCM); decryption requires
  an attested TEE context plus the AES key.
- **Tier C** — Tier B + Reed-Solomon erasure coding (K-of-N) + Shamir-split
  decryption keys (M-of-N); decryption requires an attested TEE context
  plus enough erasure shards plus enough key shares.

The gate is a *pure verification + decryption* layer over the existing
``prsm.storage`` primitives (`encryption.py`, `erasure.py`, `key_sharing.py`).
It does **not** itself host a TEE — the caller (the inference executor)
asserts the TEE context it is running under via ``TEEContext`` and the
gate refuses Tier B/C work outside an attested context. Hardware-TEE
attestation is verified upstream when available; the gate's contract is
"if I am told the context is software-only, I will refuse Tier B/C
unless software-TEE has been explicitly allow-listed."

This means a malicious or buggy caller cannot bypass the gate by lying
about the context — but only because the gate is one layer in defense
in depth. The TEE attestation itself is the primary defense; the gate
is a fail-closed wrapper that surfaces violations as exceptions
instead of silently leaking plaintext.

WIRING STATUS (sp1126, Domain-03 review F4) — IMPORTANT: ``open_tier_a/b/c`` and
``open_content`` below are NOT yet wired into any live inference execution path (the
production executor currently serves Tier-A public content only; Tier B/C confidential
decryption is intended §7 scaffolding awaiting the swarm-side decrypt integration). So
this module currently enforces NOTHING at runtime — do NOT assume Tier B/C confidentiality
is active because this gate exists. It is kept (not deleted) because it is the planned,
already-fail-closed decrypt layer the executor will call once Tier B/C content dispatch
lands; ``open_tier_b/c`` already refuse outside an attested TEEContext
(``TEEContextRequiredError``), so wiring it cannot accidentally leak plaintext. When you
wire it, call it from the executor's post-fetch path and pass the real attested
``TEEContext``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from prsm.compute.inference.models import ContentTier
from prsm.compute.tee.models import HARDWARE_TEE_TYPES, TEEType
from prsm.storage.encryption import (
    AESKey,
    EncryptedPayload,
    EncryptionAuthError,
    EncryptionError,
    decrypt,
)
from prsm.storage.erasure import ErasureError, ErasureMetadata, ErasureShard
from prsm.storage.erasure import decode as erasure_decode
from prsm.storage.key_sharing import (
    InsufficientSharesError,
    KeyShare,
    ShamirError,
    combine_shares,
)


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class ContentTierGateError(Exception):
    """Base error from the content-tier gate."""


class TEEContextRequiredError(ContentTierGateError):
    """Raised when Tier B/C decryption is attempted outside an attested TEE."""


class MissingMaterialError(ContentTierGateError):
    """Required material (key / shares / shards / plaintext) absent for the tier."""


# --------------------------------------------------------------------------
# Context + materials
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TEEContext:
    """The TEE context under which decryption is being attempted.

    Tier B/C decryption requires ``tee_type`` to be a hardware-backed TEE
    (SGX/TDX/SEV/TrustZone/Secure Enclave). Software TEEs are accepted
    only when ``allow_software_tee=True`` (typical for dev/test runs;
    production nodes serving Tier B/C must run on attested hardware).
    """

    tee_type: TEEType
    allow_software_tee: bool = False

    @property
    def is_attested(self) -> bool:
        """Whether this context is acceptable for Tier B/C decryption."""
        if self.tee_type.value in HARDWARE_TEE_TYPES:
            return True
        if self.tee_type == TEEType.SOFTWARE and self.allow_software_tee:
            return True
        return False


@dataclass(frozen=True)
class TierBMaterial:
    """Everything needed to decrypt Tier B content: payload + key."""

    payload: EncryptedPayload
    key: AESKey


@dataclass(frozen=True)
class TierCMaterial:
    """Everything needed to decrypt Tier C content.

    Reconstruction order:
      1. ``combine_shares(key_shares)`` → ``AESKey`` (M-of-N Shamir)
      2. ``erasure_decode(erasure_metadata, erasure_shards)`` → ciphertext bytes
         (K-of-N Reed-Solomon)
      3. ``decrypt(EncryptedPayload(ciphertext, iv, auth_tag, key_id), key)`` → plaintext

    The Tier-C wire format stores the AEAD ``iv`` and ``auth_tag``
    alongside the erasure-coded ciphertext, since those are short fixed-size
    fields that don't benefit from sharding and are needed to authenticate
    the reconstructed payload.
    """

    erasure_metadata: ErasureMetadata
    erasure_shards: Sequence[ErasureShard]
    key_shares: Sequence[KeyShare]
    iv: bytes
    auth_tag: bytes
    key_id: str


# --------------------------------------------------------------------------
# Tier-specific entry points
# --------------------------------------------------------------------------


def open_tier_a(plaintext: bytes) -> bytes:
    """Tier A is public; pass-through. No TEE context required.

    Provided for symmetry so callers can dispatch on ``ContentTier``
    without a special-case branch.
    """
    return plaintext


def open_tier_b(material: TierBMaterial, ctx: TEEContext) -> bytes:
    """Decrypt Tier B content. Requires an attested TEE context."""
    if not ctx.is_attested:
        raise TEEContextRequiredError(
            f"Tier B decryption requires a hardware-attested TEE context "
            f"(got tee_type={ctx.tee_type.value}, "
            f"allow_software_tee={ctx.allow_software_tee})"
        )
    try:
        return decrypt(material.payload, material.key)
    except (EncryptionAuthError, EncryptionError) as exc:
        raise ContentTierGateError(f"Tier B decryption failed: {exc}") from exc


def open_tier_c(material: TierCMaterial, ctx: TEEContext) -> bytes:
    """Decrypt Tier C content: reconstruct key + erasure-decode + decrypt.

    Requires an attested TEE context. Returns plaintext bytes.

    Failure modes:
      - Outside a TEE → ``TEEContextRequiredError``
      - Insufficient key shares → ``MissingMaterialError`` (chained from
        ``InsufficientSharesError``)
      - Insufficient or corrupt erasure shards → ``MissingMaterialError``
      - Reconstructed key's ``key_id`` doesn't match the one stamped on
        the material (split/payload pairing error) → ``ContentTierGateError``
      - AEAD authentication failure → ``ContentTierGateError`` (chained
        from ``EncryptionAuthError``)
    """
    if not ctx.is_attested:
        raise TEEContextRequiredError(
            f"Tier C decryption requires a hardware-attested TEE context "
            f"(got tee_type={ctx.tee_type.value}, "
            f"allow_software_tee={ctx.allow_software_tee})"
        )

    # 1. Reconstruct the AES key from M-of-N Shamir shares.
    try:
        key = combine_shares(list(material.key_shares))
    except (InsufficientSharesError, ShamirError) as exc:
        raise MissingMaterialError(
            f"Tier C key reconstruction failed: {exc}"
        ) from exc

    # Pairing check: the reconstructed key must match the key_id stamped
    # on the material. Catches the case where someone mixes shares from
    # one split with metadata from another.
    if key.key_id != material.key_id:
        raise ContentTierGateError(
            f"Tier C key_id mismatch: reconstructed={key.key_id!r}, "
            f"expected={material.key_id!r}"
        )

    # 2. Reconstruct the encrypted payload from K-of-N erasure shards.
    try:
        ciphertext = erasure_decode(
            material.erasure_metadata, list(material.erasure_shards)
        )
    except ErasureError as exc:
        raise MissingMaterialError(
            f"Tier C erasure decode failed: {exc}"
        ) from exc

    # 3. Authenticated decryption under the reconstructed key.
    payload = EncryptedPayload(
        ciphertext=ciphertext,
        iv=material.iv,
        auth_tag=material.auth_tag,
        key_id=material.key_id,
    )
    try:
        return decrypt(payload, key)
    except (EncryptionAuthError, EncryptionError) as exc:
        raise ContentTierGateError(f"Tier C decryption failed: {exc}") from exc


# --------------------------------------------------------------------------
# Single-entry dispatch
# --------------------------------------------------------------------------


def open_content(
    tier: ContentTier,
    *,
    plaintext: Optional[bytes] = None,
    tier_b: Optional[TierBMaterial] = None,
    tier_c: Optional[TierCMaterial] = None,
    ctx: Optional[TEEContext] = None,
) -> bytes:
    """Single-entry dispatch: open content for the given tier.

    Acceptance per design plan §4 Task 3: Tier-A no-op, Tier-B/C correctly
    gated by TEE context flag.

    - Tier A → ``plaintext`` required; no ``ctx`` needed.
    - Tier B → ``tier_b`` material + ``ctx`` required.
    - Tier C → ``tier_c`` material + ``ctx`` required.
    """
    if tier == ContentTier.A:
        if plaintext is None:
            raise MissingMaterialError("Tier A requires plaintext bytes")
        return open_tier_a(plaintext)

    if tier == ContentTier.B:
        if tier_b is None:
            raise MissingMaterialError("Tier B requires tier_b material")
        if ctx is None:
            raise TEEContextRequiredError("Tier B requires TEE context")
        return open_tier_b(tier_b, ctx)

    if tier == ContentTier.C:
        if tier_c is None:
            raise MissingMaterialError("Tier C requires tier_c material")
        if ctx is None:
            raise TEEContextRequiredError("Tier C requires TEE context")
        return open_tier_c(tier_c, ctx)

    raise ValueError(f"unknown content tier: {tier!r}")
