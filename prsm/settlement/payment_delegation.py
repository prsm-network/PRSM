"""Sprint 1091 — PaymentDelegation EIP-712 primitive (relayer delegated-auth, brick 1).

The sp1054 self-signed model makes the paying requester sign every request. This adds the
relayer/gateway path: a funding requester signs ONE ``PaymentDelegation`` authorizing a
relayer to sign per-request ``PaymentAuthorization``s on its behalf, bounded by a
cumulative cap + expiry. End-users then transact through the gateway without holding a
wallet / signing each request, while the funder keeps custody (escrow) + caps.

  PaymentDelegation {
    address requester           // funds escrow; the delegating party (pays via settleFromRequester)
    address relayer             // authorized to sign PaymentAuthorizations on requester's behalf
    address provider            // sp1437 — the ONE provider node this delegation may be spent at
    uint256 max_total_spend_wei // cumulative ceiling across all auths under this delegation
    bytes32 delegation_nonce    // unique per delegation (anti-replay + revocation handle)
    uint256 expiry_unix         // delegation invalid after this
  }

sp1437 (money-path audit #3): the cumulative cap is enforced by a PER-NODE budget store keyed
by delegation_nonce, so a provider-AGNOSTIC delegation with cap C could be replayed at N
providers for a C×N drain (N× the signed cap). Binding the delegation to a single ``provider``
— checked here against the auth's provider, which the verifier in turn pins to the local node —
makes it spendable at exactly one node. To fund N providers a funder signs N delegations, each
with its own cap.

Same network-locked domain as PaymentAuthorization (PRSM-Payment / v1 / chainId) so both
live in one signature scheme. The verifier (``verify_delegated_authorization``) checks the
full two-signature chain: the relayer signed the per-request auth AND the funder signed a
delegation authorizing THAT relayer for THAT funder, neither expired, per-request within
the delegation cap. Cumulative spend tracking + ingress wiring are later bricks.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Union

from eth_account import Account
from eth_account.messages import encode_typed_data
from eth_utils import keccak

# Reuse the network-locked domain + helpers + signature-format error from sp1054 so the
# delegation and the authorization share one coherent scheme.
from prsm.settlement.payment_authorization import (
    DEFAULT_CHAIN_ID,
    DOMAIN_NAME,
    DOMAIN_VERSION,
    InvalidSignatureFormat,
    _to_bytes32,
    is_expired,
    verify_payment_authorization_signature,
)

_REQUIRED = (
    "requester", "relayer", "provider", "max_total_spend_wei",
    "delegation_nonce", "expiry_unix",
)


class DelegatedAuthError(ValueError):
    """The relayer delegated-authorization chain failed verification (bad signature,
    mis-binding, expiry, or per-request over the delegation cap). Raised fail-closed."""


def _build_typed_data(payload: Dict[str, Any], chain_id: int) -> Dict[str, Any]:
    missing = [k for k in _REQUIRED if k not in payload]
    if missing:
        raise ValueError(f"payment delegation missing required fields: {missing!r}")
    if not isinstance(payload["requester"], str):
        raise ValueError("requester must be an address string")
    if not isinstance(payload["relayer"], str):
        raise ValueError("relayer must be an address string")
    if not isinstance(payload["provider"], str):
        raise ValueError("provider must be an address string")
    return {
        "types": {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "chainId", "type": "uint256"},
            ],
            "PaymentDelegation": [
                {"name": "requester", "type": "address"},
                {"name": "relayer", "type": "address"},
                {"name": "provider", "type": "address"},
                {"name": "max_total_spend_wei", "type": "uint256"},
                {"name": "delegation_nonce", "type": "bytes32"},
                {"name": "expiry_unix", "type": "uint256"},
            ],
        },
        "primaryType": "PaymentDelegation",
        "domain": {
            "name": DOMAIN_NAME,
            "version": DOMAIN_VERSION,
            "chainId": int(chain_id),
        },
        "message": {
            "requester": str(payload["requester"]),
            "relayer": str(payload["relayer"]),
            "provider": str(payload["provider"]),
            "max_total_spend_wei": int(payload["max_total_spend_wei"]),
            "delegation_nonce": _to_bytes32("delegation_nonce", payload["delegation_nonce"]),
            "expiry_unix": int(payload["expiry_unix"]),
        },
    }


def encode_delegation_typed_data_hash(
    payload: Dict[str, Any], *, chain_id: int = DEFAULT_CHAIN_ID,
) -> bytes:
    """Canonical EIP-712 digest (0x1901 || domain_sep || hashStruct), chain-bound."""
    typed = _build_typed_data(payload, chain_id=chain_id)
    signable = encode_typed_data(full_message=typed)
    return keccak(b"\x19\x01" + bytes(signable.header) + bytes(signable.body))


def sign_payment_delegation(
    payload: Dict[str, Any], private_key: Union[bytes, str], *,
    chain_id: int = DEFAULT_CHAIN_ID,
) -> bytes:
    """Sign a delegation. TEST HELPER + reference — production funders sign with a
    wallet. Returns the 65-byte (r || s || v) signature."""
    typed = _build_typed_data(payload, chain_id=chain_id)
    signable = encode_typed_data(full_message=typed)
    return bytes(Account.sign_message(signable, private_key=private_key).signature)


def verify_payment_delegation_signature(
    payload: Dict[str, Any], signature: Union[bytes, str], *,
    chain_id: int = DEFAULT_CHAIN_ID,
) -> str:
    """Recover the signer (EIP-55) from a delegation signature. The CALLER compares it
    case-insensitively against ``payload['requester']``."""
    if isinstance(signature, str):
        s = signature.strip()
        s = s if s.startswith("0x") else "0x" + s
        try:
            sig_bytes = bytes.fromhex(s[2:])
        except ValueError as exc:
            raise InvalidSignatureFormat(f"signature hex parse failed: {exc!s}")
    elif isinstance(signature, (bytes, bytearray)):
        sig_bytes = bytes(signature)
    else:
        raise InvalidSignatureFormat(
            f"signature must be bytes or hex string, got {type(signature).__name__}")
    if len(sig_bytes) != 65:
        raise InvalidSignatureFormat(
            f"signature must be 65 bytes (r||s||v), got {len(sig_bytes)} bytes")
    typed = _build_typed_data(payload, chain_id=chain_id)
    signable = encode_typed_data(full_message=typed)
    return Account.recover_message(signable, signature=sig_bytes)


def is_delegation_expired(
    payload: Dict[str, Any], *, now: Optional[Callable[[], float]] = None,
) -> bool:
    """True iff the delegation's ``expiry_unix`` is strictly before now."""
    current = (now or time.time)()
    return int(payload["expiry_unix"]) < current


def verify_delegated_authorization(
    *,
    auth_payload: Dict[str, Any],
    auth_signature: Union[bytes, str],
    delegation_payload: Dict[str, Any],
    delegation_signature: Union[bytes, str],
    chain_id: int = DEFAULT_CHAIN_ID,
    now: Optional[Callable[[], float]] = None,
) -> str:
    """Verify the full relayer chain and return the authenticated FUNDING requester
    (the address whose escrow pays via settleFromRequester). Raises ``DelegatedAuthError``
    fail-closed on any failure.

    Checks:
      1. The per-request PaymentAuthorization is signed by the delegation's ``relayer``.
      2. The PaymentDelegation is signed by its own ``requester`` (the funder).
      3. The auth's ``requester`` == the delegation's ``requester`` (the auth charges the
         funder's escrow, not some third party).
      3b. sp1437 — the auth's ``provider`` == the delegation's ``provider`` (the delegation
         may be spent at exactly ONE provider node; the caller/verifier then pins that
         provider to the LOCAL node, so a delegation cannot be replayed across providers for
         a C×N drain past the signed cap).
      4. Neither the delegation nor the auth is expired.
      5. The per-request ``max_spend_wei`` does not exceed the delegation's
         ``max_total_spend_wei`` (cumulative tracking across requests is a later brick).
    """
    funder = str(delegation_payload.get("requester") or "")
    relayer = str(delegation_payload.get("relayer") or "")
    deleg_provider = str(delegation_payload.get("provider") or "")
    if not funder or not relayer:
        raise DelegatedAuthError("delegation missing requester/relayer")
    if not deleg_provider:
        # sp1437 — fail closed: a provider-less delegation is the pre-fix provider-AGNOSTIC
        # hole (spendable at any node → C×N drain). Never treat a missing provider as a
        # wildcard.
        raise DelegatedAuthError("delegation missing provider (sp1437 provider binding)")

    # 2. delegation signed by the funder it names.
    try:
        deleg_signer = verify_payment_delegation_signature(
            delegation_payload, delegation_signature, chain_id=chain_id)
    except InvalidSignatureFormat as exc:
        raise DelegatedAuthError(f"delegation signature malformed: {exc!s}")
    if deleg_signer.lower() != funder.lower():
        raise DelegatedAuthError(
            "delegation not signed by its requester (funder) — "
            f"signer {deleg_signer} != requester {funder}")

    # 1. per-request auth signed by the delegated relayer.
    try:
        auth_signer = verify_payment_authorization_signature(
            auth_payload, auth_signature, chain_id=chain_id)
    except InvalidSignatureFormat as exc:
        raise DelegatedAuthError(f"authorization signature malformed: {exc!s}")
    if auth_signer.lower() != relayer.lower():
        raise DelegatedAuthError(
            "authorization not signed by the delegated relayer — "
            f"signer {auth_signer} != relayer {relayer}")

    # 3. the auth charges the funder named in the delegation.
    auth_requester = str(auth_payload.get("requester") or "")
    if auth_requester.lower() != funder.lower():
        raise DelegatedAuthError(
            "authorization requester does not match the delegation funder — "
            f"{auth_requester} != {funder}")

    # 3b. sp1437 — the auth pays the provider named in the delegation. The auth's provider is
    # pinned to the local node by the verifier, so this ties the delegation to one node and
    # blocks the cross-provider C×N drain.
    auth_provider = str(auth_payload.get("provider") or "")
    if auth_provider.lower() != deleg_provider.lower():
        raise DelegatedAuthError(
            "authorization provider does not match the delegation provider — "
            f"{auth_provider} != {deleg_provider} (delegation is bound to one provider)")

    # 4. freshness — both must be unexpired.
    if is_delegation_expired(delegation_payload, now=now):
        raise DelegatedAuthError("delegation expired")
    if is_expired(auth_payload, now=now):
        raise DelegatedAuthError("authorization expired")

    # 5. per-request within the delegation's cumulative ceiling.
    if int(auth_payload["max_spend_wei"]) > int(delegation_payload["max_total_spend_wei"]):
        raise DelegatedAuthError(
            "authorization max_spend_wei exceeds the delegation's max_total_spend_wei")

    return funder
