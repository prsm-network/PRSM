"""Sprint 555 — EIP-712 verification primitive for /wallet/withdraw.

Pure module — no API integration (that lands in sprint 556) and no
ledger access. Builds the canonical EIP-712 typed-data hash for a
withdraw request, signs it with a private key (test helper), and
verifies an ECDSA secp256k1 signature by recovering the signer
address.

Domain spec (locked per sprint-554's user-input decisions):
  Name:    "PRSM-Withdraw"
  Version: "v1"
  Chain:   8453 (Base mainnet) by default; overridable via the
           chain_id kwarg for cross-chain testing only.

Typed-data shape:
  WithdrawRequest {
    string  wallet_id
    uint256 amount_ftns_wei
    address to_eth_address
    uint256 nonce
    uint256 expiry_unix
  }

Production callers (sprint 556) will:

  1. Reconstruct the payload from the HTTP request body fields
     wallet_id / amount_ftns / to_eth_address / nonce / expiry_unix
     (converting amount_ftns to wei).
  2. Call ``verify_withdraw_signature(payload, signature)``.
  3. Compare the recovered address (case-insensitively) against the
     wallet's linked eth_address from sprint 540.
  4. Check ``is_expired(payload)`` against current wall-clock time.
  5. Check ``payload["nonce"]`` matches
     ``ledger.get_next_withdraw_nonce(wallet_id)``.
  6. Atomically bump nonce + debit as part of the same DB transaction
     (sprint 556 owns the integration).

Mismatch on any check → 401 from sprint 556's enforcement.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Optional, Union
import time

from eth_account import Account
from eth_account.messages import encode_typed_data
from eth_utils import keccak


# Default chain — Base mainnet. Operators deploying on a different
# chain MUST pass chain_id explicitly; the domain separator binds
# signatures to a specific chain to prevent cross-chain replay.
DEFAULT_CHAIN_ID = 8453

# Domain version bumped here if the typed-data shape changes.
# Every signature on the network is keyed off this string — DO NOT
# change without a coordinated migration.
DOMAIN_NAME = "PRSM-Withdraw"
DOMAIN_VERSION = "v1"


class InvalidSignatureFormat(ValueError):
    """Raised when the supplied signature isn't a parseable
    65-byte / 130-hex-char ECDSA secp256k1 signature."""


def _build_typed_data(
    payload: Dict[str, Any], chain_id: int,
) -> Dict[str, Any]:
    """Return the EIP-712 typed-data dict accepted by
    ``eth_account.messages.encode_typed_data``.

    Validates payload field presence + types so a malformed input
    raises here rather than producing a junk hash.
    """
    required = (
        "wallet_id", "amount_ftns_wei", "to_eth_address",
        "nonce", "expiry_unix",
    )
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(
            f"withdraw payload missing required fields: {missing!r}"
        )
    if not isinstance(payload["wallet_id"], str):
        raise ValueError("wallet_id must be a string")
    if not isinstance(payload["to_eth_address"], str):
        raise ValueError("to_eth_address must be a string")
    return {
        "types": {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "chainId", "type": "uint256"},
            ],
            "WithdrawRequest": [
                {"name": "wallet_id", "type": "string"},
                {"name": "amount_ftns_wei", "type": "uint256"},
                {"name": "to_eth_address", "type": "address"},
                {"name": "nonce", "type": "uint256"},
                {"name": "expiry_unix", "type": "uint256"},
            ],
        },
        "primaryType": "WithdrawRequest",
        "domain": {
            "name": DOMAIN_NAME,
            "version": DOMAIN_VERSION,
            "chainId": int(chain_id),
        },
        "message": {
            "wallet_id": str(payload["wallet_id"]),
            "amount_ftns_wei": int(payload["amount_ftns_wei"]),
            "to_eth_address": str(payload["to_eth_address"]),
            "nonce": int(payload["nonce"]),
            "expiry_unix": int(payload["expiry_unix"]),
        },
    }


def encode_withdraw_typed_data_hash(
    payload: Dict[str, Any],
    *,
    chain_id: int = DEFAULT_CHAIN_ID,
) -> bytes:
    """Canonical EIP-712 hash of the withdraw payload — the full
    digest that gets signed:

      keccak256(0x19 0x01 || domain_separator || hashStruct(message))

    eth_account's ``SignableMessage`` exposes the parts:
      - ``signable.header`` = 32-byte EIP-712 domain separator
      - ``signable.body``   = 32-byte hashStruct of the message

    The signed value is the keccak of the prefix + header + body.
    Using just ``body`` would lose the domain separator's chain_id
    binding and let signatures replay across chains.

    Production callers don't typically need this directly — sign/
    verify use it internally — but tests assert hash stability AND
    chain_id binding via this surface.
    """
    typed = _build_typed_data(payload, chain_id=chain_id)
    signable = encode_typed_data(full_message=typed)
    return keccak(
        b"\x19\x01" + bytes(signable.header) + bytes(signable.body)
    )


def sign_withdraw_payload(
    payload: Dict[str, Any],
    private_key: Union[bytes, str],
    *,
    chain_id: int = DEFAULT_CHAIN_ID,
) -> bytes:
    """Sign the canonical EIP-712 payload. TEST HELPER + reference
    implementation — production clients sign with MetaMask /
    hardware wallets and submit the resulting signature directly.

    Returns the 65-byte (r || s || v) signature.
    """
    typed = _build_typed_data(payload, chain_id=chain_id)
    signable = encode_typed_data(full_message=typed)
    signed = Account.sign_message(signable, private_key=private_key)
    return bytes(signed.signature)


def verify_withdraw_signature(
    payload: Dict[str, Any],
    signature: Union[bytes, str],
    *,
    chain_id: int = DEFAULT_CHAIN_ID,
) -> str:
    """Recover the signer address from a signature over the canonical
    EIP-712 payload. Returns the EIP-55 checksum address (hex
    "0x..."). Caller compares case-insensitively against the
    wallet's linked eth_address.

    Raises ``InvalidSignatureFormat`` if the signature isn't a
    valid ECDSA secp256k1 signature shape (length / hex parsing).
    A SHAPE-valid signature over a tampered payload returns a
    DIFFERENT (but still valid-looking) address — sprint 556's
    enforcement is responsible for checking that the recovered
    address matches the linked eth_address.
    """
    if isinstance(signature, str):
        s = signature.strip()
        if not s.startswith("0x"):
            s = "0x" + s
        try:
            sig_bytes = bytes.fromhex(s[2:])
        except ValueError as exc:
            raise InvalidSignatureFormat(
                f"signature hex parse failed: {exc!s}"
            )
    elif isinstance(signature, (bytes, bytearray)):
        sig_bytes = bytes(signature)
    else:
        raise InvalidSignatureFormat(
            f"signature must be bytes or hex string, got "
            f"{type(signature).__name__}"
        )
    if len(sig_bytes) != 65:
        raise InvalidSignatureFormat(
            f"signature must be 65 bytes (r||s||v), got "
            f"{len(sig_bytes)} bytes"
        )
    typed = _build_typed_data(payload, chain_id=chain_id)
    signable = encode_typed_data(full_message=typed)
    return Account.recover_message(signable, signature=sig_bytes)


def is_expired(
    payload: Dict[str, Any],
    *,
    now: Optional[Callable[[], float]] = None,
) -> bool:
    """True iff payload's expiry_unix is strictly less than the
    current wall-clock time. ``now`` is an injection point for
    deterministic tests."""
    current = (now or time.time)()
    return int(payload["expiry_unix"]) < int(current)


async def resolve_requires_signature_failclosed(
    read_flag: Callable[[], Awaitable[Any]],
) -> bool:
    """sp1454 — resolve a wallet's ``requires_user_signature`` flag, FAILING CLOSED.

    ``read_flag`` is an async callable that reads the flag (e.g. the ledger lookup). If it RAISES we
    CANNOT prove the wallet is signature-optional, so we REQUIRE a signature (return True) — the safe
    default for the /wallet/withdraw gate. Fail-OPEN here (defaulting False on a transient read error)
    would skip the whole EIP-712 signature check and let an UNSIGNED withdraw proceed against a
    signature-required wallet. The result is coerced to bool so a truthy non-bool still gates as
    required."""
    try:
        return bool(await read_flag())
    except Exception:  # noqa: BLE001 — a read error must REQUIRE a signature, never skip it
        return True
