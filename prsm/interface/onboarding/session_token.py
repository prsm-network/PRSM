"""Sprint 1013 — wallet session token (API-authz Residual A).

A stateless, HMAC-signed, wallet-bound, TTL'd token that carries the
proof-of-control established by a successful SIWE verify (EIP-4361) to later
wallet-OWNER read endpoints. The verified address is bound INTO the signed
payload, so a holder can only act for the wallet it proved control of — closing
the read-IDOR on /api/v1/auth/wallet/* without needing the operator's API key
(which the wallet owner does not hold) and without a server-side session store.

Token wire format (three dot-separated parts):

    v1.<b64url(payload_json)>.<b64url(hmac_sha256(secret, payload_json))>

where ``payload_json`` is canonical (sorted keys, compact) JSON:

    {"w": "<lowercased wallet address>", "iat": <unix>, "exp": <unix>}

Verification recomputes the HMAC (constant-time compare via hmac.compare_digest)
and checks expiry. The address is normalized to lowercase on mint + returned
lowercased on verify so EIP-55 checksum casing never causes a false mismatch.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Optional, Union

_VERSION = "v1"
_MIN_SECRET_BYTES = 16


class SessionTokenError(Exception):
    """Base class for all session-token failures."""


class SessionTokenMalformed(SessionTokenError):
    """The token is not the expected shape (parts / version / base64 / JSON)."""


class SessionTokenInvalid(SessionTokenError):
    """The HMAC signature does not verify (forged or wrong secret)."""


class SessionTokenExpired(SessionTokenError):
    """The token's ``exp`` is at or before the verification time."""


def _b64u_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64u_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _normalize_address(address: str) -> str:
    return (address or "").strip().lower()


def _coerce_secret(secret: Union[bytes, bytearray]) -> bytes:
    if not isinstance(secret, (bytes, bytearray)):
        raise SessionTokenError("secret must be bytes")
    if len(secret) < _MIN_SECRET_BYTES:
        raise SessionTokenError(
            f"secret must be at least {_MIN_SECRET_BYTES} bytes"
        )
    return bytes(secret)


def mint_session_token(
    wallet_address: str,
    *,
    secret: Union[bytes, bytearray],
    ttl_seconds: int = 3600,
    now: Optional[float] = None,
) -> str:
    """Mint a wallet-bound session token. Raises SessionTokenError on a bad
    secret / empty wallet / non-positive ttl."""
    key = _coerce_secret(secret)
    addr = _normalize_address(wallet_address)
    if not addr:
        raise SessionTokenError("wallet_address required")
    if not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
        raise SessionTokenError("ttl_seconds must be a positive int")
    iat = int(now if now is not None else time.time())
    payload = {"w": addr, "iat": iat, "exp": iat + ttl_seconds}
    payload_bytes = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    sig = hmac.new(key, payload_bytes, hashlib.sha256).digest()
    return f"{_VERSION}.{_b64u_encode(payload_bytes)}.{_b64u_encode(sig)}"


def verify_session_token(
    token: Any,
    *,
    secret: Union[bytes, bytearray],
    now: Optional[float] = None,
) -> str:
    """Verify a session token and return its bound (lowercased) wallet address.

    Raises:
      SessionTokenMalformed — wrong shape / version / base64 / JSON / missing field
      SessionTokenInvalid   — HMAC signature mismatch (forged or wrong secret)
      SessionTokenExpired   — token's exp is at or before ``now``
    """
    key = _coerce_secret(secret)
    if not isinstance(token, str):
        raise SessionTokenMalformed("token must be a string")
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != _VERSION:
        raise SessionTokenMalformed("unexpected token format")
    _, payload_b64, sig_b64 = parts
    if not payload_b64 or not sig_b64:
        raise SessionTokenMalformed("empty token segment")
    try:
        payload_bytes = _b64u_decode(payload_b64)
        sig = _b64u_decode(sig_b64)
    except Exception as exc:  # noqa: BLE001
        raise SessionTokenMalformed(f"invalid base64: {exc}") from exc

    expected = hmac.new(key, payload_bytes, hashlib.sha256).digest()
    # Constant-time compare — never branch on the signature bytes early.
    if not hmac.compare_digest(sig, expected):
        raise SessionTokenInvalid("signature mismatch")

    try:
        payload = json.loads(payload_bytes)
    except Exception as exc:  # noqa: BLE001
        raise SessionTokenMalformed(f"invalid payload json: {exc}") from exc
    if not isinstance(payload, dict):
        raise SessionTokenMalformed("payload is not an object")

    wallet = payload.get("w")
    exp = payload.get("exp")
    if not isinstance(wallet, str) or not wallet:
        raise SessionTokenMalformed("missing wallet binding")
    try:
        exp_i = int(exp)
    except (TypeError, ValueError) as exc:
        raise SessionTokenMalformed("missing/invalid exp") from exc

    now_i = int(now if now is not None else time.time())
    if now_i >= exp_i:
        raise SessionTokenExpired("session token expired")
    return _normalize_address(wallet)
