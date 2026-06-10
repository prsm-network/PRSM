"""Sprint 1057 — requester-side payment helper (requester-payment, brick 4).

What a paying requester runs to participate in settled inference:

  1. Fund — deposit FTNS into the on-chain EscrowPool under your own address using
     the sprint-1041 ``EscrowPoolClient.deposit`` (self-custodied; withdraw any
     unspent balance). Not wrapped here — it's a one-liner against that client.
  2. Authorize — for each request, ``build_payment_authorization`` binds a signed
     EIP-712 PaymentAuthorization to the exact request body and returns the object
     to embed as ``{"payment_authorization": <this>}`` in the /compute/inference
     POST body.

This module is PURE (no network) so it's hermetic to test and usable from a CLI or
SDK. Production wallets (MetaMask / hardware) can produce the same object by signing
the typed data from ``prsm.settlement.payment_authorization``; this helper is the
reference + python-client path. See docs/2026-06-10-requester-payment-model-proposal.md.
"""
from __future__ import annotations

import secrets
from decimal import Decimal
from typing import Any, Dict, Optional, Union

from eth_account import Account

from prsm.settlement.payment_authorization import (
    DEFAULT_CHAIN_ID,
    canonical_request_hash,
    inference_request_fields,
    sign_payment_authorization,
)


def build_payment_authorization(
    *,
    requester_key: Union[str, bytes],
    provider_address: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    privacy_tier: str,
    content_tier: str,
    max_spend_ftns: Union[str, float, Decimal],
    expiry_unix: int,
    job_nonce: Optional[str] = None,
    chain_id: int = DEFAULT_CHAIN_ID,
) -> Dict[str, Any]:
    """Build + sign a PaymentAuthorization bound to one inference request.

    Returns ``{"payload": {...}, "signature": "0x.."}`` — drop it into the request
    body under ``payment_authorization``. ``max_spend_ftns`` is the ceiling the
    requester authorizes for this job (must be >= the node's quote, which is the
    request's ``budget_ftns``). ``job_nonce`` defaults to a fresh random 32-byte
    value (anti-replay); pass one only to reproduce a specific authorization. The
    request_hash is computed with the SAME canonicalization the provider uses, so
    the binding matches; ``max_tokens`` unset on the request should be passed as 0
    here (mirrors the provider's ``int(max_tokens or 0)``).
    """
    requester = Account.from_key(requester_key).address
    if job_nonce is None:
        job_nonce = "0x" + secrets.token_bytes(32).hex()
    request_hash = canonical_request_hash(inference_request_fields(
        model_id=model_id, prompt=prompt, max_tokens=int(max_tokens),
        privacy_tier=privacy_tier, content_tier=content_tier,
    ))
    max_spend_wei = int(Decimal(str(max_spend_ftns)) * (Decimal(10) ** 18))
    payload = {
        "requester": requester,
        "provider": provider_address,
        "max_spend_wei": max_spend_wei,
        "job_nonce": job_nonce,
        "expiry_unix": int(expiry_unix),
        "request_hash": request_hash,
    }
    signature = sign_payment_authorization(payload, requester_key, chain_id=chain_id)
    return {"payload": payload, "signature": "0x" + signature.hex()}
