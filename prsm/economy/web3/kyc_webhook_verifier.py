"""Sprint 283 — KYC webhook signature verification.

Vendor webhook handlers MUST verify the signature before
applying any status update — without it, anyone with the
operator's public webhook URL can flip a user's KYC status to
VERIFIED at will. This module ships the HMAC-SHA256 verifiers
for the two real-world vendors PRSM ships with (Persona +
Onfido); Plaid Identity uses JWT signatures and is deferred
to a follow-on sprint.

All comparisons use ``hmac.compare_digest`` for constant-time
equality — prevents timing oracles that could leak the secret
on a careless ``==`` comparison.

Empty-secret guard: if the operator's env var resolves to "",
verification ALWAYS fails regardless of body match. This
prevents the misconfig where an unset env var defaults to ""
and an attacker discovers they can forge signatures by
HMAC'ing with empty bytes.
"""
from __future__ import annotations

import hmac
import hashlib
import logging
from typing import Dict, Mapping, Tuple

logger = logging.getLogger(__name__)


def parse_persona_signature_header(signature_header: str) -> Dict[str, str]:
    """sp1453 — parse a Persona ``Persona-Signature`` header (``t=<unix_ts>,v1=<hex_hmac>``) into its
    ``{t, v1}`` parts, TOLERANT of surrounding whitespace (``t=123, v1 = <hex>``).

    This is the SINGLE source of truth for reading the header, shared by the signature verifier AND the
    node's replay-token extractor. They previously parsed it INDEPENDENTLY — the verifier via
    partition('=')+strip() (tolerant) and the replay-token extractor via a strict ``startswith('v1=')``
    — so a spaced variant verified fine yet left the replay token empty, silently skipping the replay
    ring (a defense that could not fire for that header). One parser can't drift from itself.
    """
    parts: Dict[str, str] = {}
    for piece in (signature_header or "").split(","):
        piece = piece.strip()
        if "=" not in piece:
            continue
        k, _, v = piece.partition("=")
        parts[k.strip()] = v.strip()
    return parts


def verify_persona_signature(
    body: bytes, signature_header: str, secret: str,
) -> Tuple[bool, str]:
    """Verify a Persona webhook signature.

    Header format: ``t=<unix_ts>,v1=<hex_hmac>``. The signed
    payload is ``{ts}.{raw_body}``.

    Returns (ok, reason). reason is "" on success; on failure
    it's a short human-readable explanation suitable for
    logging (NOT for return to the vendor).
    """
    if not secret:
        return (False, "empty webhook secret")
    if not signature_header:
        return (False, "missing Persona-Signature header")

    parts = parse_persona_signature_header(signature_header)
    ts = parts.get("t")
    v1 = parts.get("v1")
    if not ts or not v1:
        return (
            False,
            f"malformed Persona-Signature header "
            f"(t={bool(ts)}, v1={bool(v1)})",
        )

    payload = f"{ts}.".encode("utf-8") + body
    expected = hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()
    if hmac.compare_digest(expected, v1):
        return (True, "")
    return (False, "Persona signature mismatch")


def verify_onfido_signature(
    body: bytes, signature_header: str, secret: str,
) -> Tuple[bool, str]:
    """Verify an Onfido webhook signature.

    Header format: hex HMAC-SHA256 of raw body with the
    webhook token. Returns (ok, reason)."""
    if not secret:
        return (False, "empty webhook secret")
    if not signature_header:
        return (False, "missing X-SHA2-Signature header")
    if len(signature_header) < 32:
        # SHA-256 hex is 64 chars; anything dramatically
        # shorter is unusable.
        return (False, "X-SHA2-Signature too short")

    expected = hmac.new(
        secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()
    if hmac.compare_digest(expected, signature_header.strip()):
        return (True, "")
    return (False, "Onfido signature mismatch")


class KYCWebhookVerifier:
    """Dispatches signature verification by vendor name."""

    @staticmethod
    def _normalize_headers(
        headers: Mapping[str, str],
    ) -> Dict[str, str]:
        """HTTP headers are case-insensitive. Normalize all
        keys to lowercase for consistent lookup."""
        return {k.lower(): v for k, v in headers.items()}

    @classmethod
    def verify(
        cls,
        vendor: str,
        body: bytes,
        headers: Mapping[str, str],
        secret: str,
    ) -> Tuple[bool, str]:
        v = (vendor or "").strip().lower()
        h = cls._normalize_headers(headers)

        if v == "persona":
            return verify_persona_signature(
                body,
                h.get("persona-signature", ""),
                secret,
            )
        if v == "onfido":
            return verify_onfido_signature(
                body,
                h.get("x-sha2-signature", ""),
                secret,
            )
        if v == "plaid":
            # Plaid Identity uses JWT signatures (RS256 with
            # JWKS lookup). Deferred to a follow-on sprint.
            return (
                False,
                "Plaid JWT signature verification not "
                "implemented in v1 — set "
                "PRSM_KYC_WEBHOOK_VERIFY_DISABLED=1 in dev "
                "to bypass.",
            )
        return (False, f"unknown vendor {vendor!r}")
