"""Sprint 1054 — PaymentAuthorization EIP-712 primitive (requester-payment, brick 1).

A paying requester signs a per-request ``PaymentAuthorization`` with the eth key
that controls their on-chain ``EscrowPool`` balance. The provider recovers the
signer, requires ``signer == requester`` + ``provider == this node``, checks
nonce/expiry/request-hash, pre-flights the escrow balance (brick 2), and only then
settles the inference on-chain against the authenticated requester (brick 3). See
docs/2026-06-10-requester-payment-model-proposal.md (self-signed end-user-wallet
path).

This module is PURE — it builds the canonical EIP-712 typed-data hash, signs it
(test helper / reference impl), and recovers the signer. No network, no escrow, no
API. It mirrors the sprint-555 ``withdraw_signature`` primitive deliberately so the
two signing surfaces stay consistent.

Domain spec (locked):
  Name:    "PRSM-Payment"
  Version: "v1"
  Chain:   8453 (Base mainnet) by default; overridable via ``chain_id`` for
           cross-chain testing only. The domain separator binds a signature to a
           specific chain to prevent cross-chain replay.

Typed-data shape:
  PaymentAuthorization {
    address requester      // EscrowPool depositor; settleFromRequester source
    address provider       // the node being paid; pins the auth to THIS provider
    uint256 max_spend_wei  // ceiling the requester authorizes for this job
    bytes32 job_nonce      // unique per job (anti-replay)
    uint256 expiry_unix    // auth invalid after this (anti-replay / staleness)
    bytes32 request_hash   // keccak of the canonical inference-request params
  }

NOTE: ``expiry_unix`` is EIP-712 ``uint256`` (matching the sprint-555
``withdraw_signature`` precedent). The encoded type string is part of the EIP-712
type hash, so an external wallet MUST sign it as uint256 (not uint64) or the
recovered signer won't match. The domain is network-locked — see DOMAIN_* below.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, Optional, Union

from eth_account import Account
from eth_account.messages import encode_typed_data
from eth_utils import keccak


# Default chain — Base mainnet. Operators on a different chain MUST pass chain_id
# explicitly; the domain separator binds signatures to one chain.
DEFAULT_CHAIN_ID = 8453

# Bump these if the typed-data shape changes — every signature on the network is
# keyed off this domain. DO NOT change without a coordinated migration.
DOMAIN_NAME = "PRSM-Payment"
DOMAIN_VERSION = "v1"

# Field names that carry the authorization.
_REQUIRED = (
    "requester", "provider", "max_spend_wei",
    "job_nonce", "expiry_unix", "request_hash",
)
_BYTES32 = ("job_nonce", "request_hash")


class InvalidSignatureFormat(ValueError):
    """Raised when the supplied signature isn't a parseable 65-byte
    (r || s || v) ECDSA secp256k1 signature."""


def _to_bytes32(name: str, value: Union[bytes, bytearray, str]) -> bytes:
    """Normalize a bytes32 field to exactly 32 raw bytes so the hex form
    ("0x..") and the raw-bytes form hash identically."""
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    elif isinstance(value, str):
        s = value[2:] if value.startswith("0x") else value
        try:
            raw = bytes.fromhex(s)
        except ValueError as exc:
            raise ValueError(f"{name} is not valid hex: {exc!s}")
    else:
        raise ValueError(
            f"{name} must be bytes or hex string, got {type(value).__name__}"
        )
    if len(raw) != 32:
        raise ValueError(f"{name} must be 32 bytes, got {len(raw)}")
    return raw


def _build_typed_data(payload: Dict[str, Any], chain_id: int) -> Dict[str, Any]:
    """Return the EIP-712 typed-data dict for ``encode_typed_data``.

    Validates field presence + normalizes types so a malformed input raises here
    rather than producing a junk hash.
    """
    missing = [k for k in _REQUIRED if k not in payload]
    if missing:
        raise ValueError(
            f"payment authorization missing required fields: {missing!r}"
        )
    if not isinstance(payload["requester"], str):
        raise ValueError("requester must be an address string")
    if not isinstance(payload["provider"], str):
        raise ValueError("provider must be an address string")
    return {
        "types": {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "chainId", "type": "uint256"},
            ],
            "PaymentAuthorization": [
                {"name": "requester", "type": "address"},
                {"name": "provider", "type": "address"},
                {"name": "max_spend_wei", "type": "uint256"},
                {"name": "job_nonce", "type": "bytes32"},
                {"name": "expiry_unix", "type": "uint256"},
                {"name": "request_hash", "type": "bytes32"},
            ],
        },
        "primaryType": "PaymentAuthorization",
        "domain": {
            "name": DOMAIN_NAME,
            "version": DOMAIN_VERSION,
            "chainId": int(chain_id),
        },
        "message": {
            "requester": str(payload["requester"]),
            "provider": str(payload["provider"]),
            "max_spend_wei": int(payload["max_spend_wei"]),
            "job_nonce": _to_bytes32("job_nonce", payload["job_nonce"]),
            "expiry_unix": int(payload["expiry_unix"]),
            "request_hash": _to_bytes32("request_hash", payload["request_hash"]),
        },
    }


def encode_payment_typed_data_hash(
    payload: Dict[str, Any], *, chain_id: int = DEFAULT_CHAIN_ID,
) -> bytes:
    """Canonical EIP-712 digest that gets signed:

      keccak256(0x19 0x01 || domain_separator || hashStruct(message))

    Using the full prefix + domain separator + struct hash binds the signature to
    the chain (via chainId in the domain). Tests assert hash stability + chain_id
    binding through this surface.
    """
    typed = _build_typed_data(payload, chain_id=chain_id)
    signable = encode_typed_data(full_message=typed)
    return keccak(b"\x19\x01" + bytes(signable.header) + bytes(signable.body))


def sign_payment_authorization(
    payload: Dict[str, Any],
    private_key: Union[bytes, str],
    *,
    chain_id: int = DEFAULT_CHAIN_ID,
) -> bytes:
    """Sign the canonical payload. TEST HELPER + reference implementation —
    production requesters sign with MetaMask / hardware wallets and submit the
    resulting signature directly. Returns the 65-byte (r || s || v) signature."""
    typed = _build_typed_data(payload, chain_id=chain_id)
    signable = encode_typed_data(full_message=typed)
    return bytes(Account.sign_message(signable, private_key=private_key).signature)


def verify_payment_authorization_signature(
    payload: Dict[str, Any],
    signature: Union[bytes, str],
    *,
    chain_id: int = DEFAULT_CHAIN_ID,
) -> str:
    """Recover the signer address from a signature over the canonical payload.
    Returns the EIP-55 checksum address. The CALLER must compare it
    case-insensitively against ``payload["requester"]`` (a shape-valid signature
    over a tampered payload returns a different but valid-looking address).

    Raises ``InvalidSignatureFormat`` if the signature isn't a valid 65-byte
    ECDSA secp256k1 signature shape.
    """
    if isinstance(signature, str):
        s = signature.strip()
        if not s.startswith("0x"):
            s = "0x" + s
        try:
            sig_bytes = bytes.fromhex(s[2:])
        except ValueError as exc:
            raise InvalidSignatureFormat(f"signature hex parse failed: {exc!s}")
    elif isinstance(signature, (bytes, bytearray)):
        sig_bytes = bytes(signature)
    else:
        raise InvalidSignatureFormat(
            f"signature must be bytes or hex string, got "
            f"{type(signature).__name__}"
        )
    if len(sig_bytes) != 65:
        raise InvalidSignatureFormat(
            f"signature must be 65 bytes (r||s||v), got {len(sig_bytes)} bytes"
        )
    typed = _build_typed_data(payload, chain_id=chain_id)
    signable = encode_typed_data(full_message=typed)
    return Account.recover_message(signable, signature=sig_bytes)


# Whitelist of inference-request fields the authorization binds to. Both the
# requester (signing) and the provider (verifying) build this from the SAME inputs
# so request_hash matches. Adding a field here is a coordinated change across both.
_REQUEST_FIELDS = ("model_id", "prompt", "max_tokens", "privacy_tier", "content_tier")


def inference_request_fields(
    *,
    model_id: str,
    prompt: str,
    max_tokens: int,
    privacy_tier: str,
    content_tier: str,
) -> Dict[str, Any]:
    """Build the canonical request-fields dict that ``canonical_request_hash``
    digests. Keyword-only so callers can't transpose positional args and silently
    produce a non-matching hash."""
    return {
        "model_id": str(model_id),
        "prompt": str(prompt),
        "max_tokens": int(max_tokens),
        "privacy_tier": str(privacy_tier),
        "content_tier": str(content_tier),
    }


def canonical_request_hash(fields: Dict[str, Any]) -> str:
    """keccak256 of the canonical JSON of the whitelisted request fields, as a
    0x-prefixed 32-byte hex string (bytes32). Order-independent (sorted keys), so a
    wallet and a python client that build the same fields agree on the hash that
    goes into ``request_hash``. Unknown extra keys are ignored — only the whitelist
    is bound."""
    canonical = {k: fields[k] for k in _REQUEST_FIELDS if k in fields}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return "0x" + keccak(text=encoded).hex()


def is_expired(
    payload: Dict[str, Any], *, now: Optional[Callable[[], float]] = None,
) -> bool:
    """True iff ``expiry_unix`` is strictly less than the current wall-clock time.
    ``now`` is an injection point for deterministic tests."""
    current = (now or time.time)()
    return int(payload["expiry_unix"]) < int(current)
