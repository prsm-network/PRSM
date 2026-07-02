"""Sprint 1358 (F1 redesign, R2) — the payment-gated wrapped-key serve (data-plane core).

The F1 fix moves the wrapped content key OFF-chain: the publisher retains it locally and serves it
ONLY to a fetcher who proves (a) they control the paying ETH address and (b) they have an on-chain
``verifyPayment == true``. The consumer then checks it against the on-chain commitment (R1) before
decrypting. This module is the testable core (no FastAPI / no web3); the node route is a thin wrapper.

Flow the route drives:
    GET /content/paid-key/{contentHash}?nonce=...&signature=...
      → recover the signer of the paid-key challenge (EIP-191)
      → gate: verify_payment(signer, contentHash, fee) must be true
      → serve the retained wrapped key
The wrapped key is sealed to the buyer's X25519 pubkey, so even a mistaken serve to a non-buyer is
useless; the gate exists to stop the DESIGNATED buyer from fetching without paying (the F1 hole).
"""
from __future__ import annotations

from typing import Any, Callable


class PaidKeyServeError(Exception):
    """A paid-key serve was refused. ``status`` maps to the HTTP code the route returns
    (404 no such key, 401 bad signature, 402 unpaid, 503 can't verify on-chain)."""

    def __init__(self, status: int, message: str) -> None:
        self.status = int(status)
        self.message = str(message)
        super().__init__(self.message)


def paid_key_challenge(content_hash: bytes, nonce: str) -> str:
    """The message a fetcher signs (EIP-191 personal_sign) to prove control of the paying address.
    Bound to the content so a signature for one dataset can't be reused for another."""
    return f"PRSM-paid-key:{bytes(content_hash).hex()}:{nonce}"


def recover_paid_key_fetcher(content_hash: bytes, nonce: str, signature: str) -> str:
    """Recover the ETH address that signed the paid-key challenge. Raises on a malformed signature."""
    from eth_account import Account
    from eth_account.messages import encode_defunct
    msg = encode_defunct(text=paid_key_challenge(content_hash, nonce))
    return Account.recover_message(msg, signature=signature)


def serve_paid_key(
    content_hash: bytes,
    nonce: str,
    signature: str,
    *,
    key_store: Any,
    verify_payment: Callable[[str, bytes, int], bool],
) -> bytes:
    """R2 core — authenticate the fetcher, gate on on-chain payment, return the retained wrapped key.

    ``key_store.get(content_hash)`` → ``{"wrapped_key": bytes, "fee_wei": int}`` or ``None`` (the
    publisher retains this at publish time — the fee is the AUTHORITATIVE deposit fee, not
    fetcher-supplied). ``verify_payment(payer, content_hash, fee_wei) -> bool`` reads
    ContentAccessVerifier.verifyPayment on-chain. Raises PaidKeyServeError on every refusal path.
    """
    ch = bytes(content_hash)
    entry = key_store.get(ch)
    if entry is None:
        raise PaidKeyServeError(404, "no paid key retained for this content on this node")

    try:
        signer = recover_paid_key_fetcher(ch, nonce, signature)
    except Exception as exc:  # noqa: BLE001
        raise PaidKeyServeError(401, f"invalid fetcher signature: {exc}") from exc

    fee_wei = int(entry["fee_wei"])
    try:
        ok = bool(verify_payment(signer, ch, fee_wei))
    except Exception as exc:  # noqa: BLE001 — an RPC failure must not leak the key
        raise PaidKeyServeError(503, f"could not verify payment on-chain: {exc}") from exc
    if not ok:
        raise PaidKeyServeError(
            402, f"{signer} has not paid the release fee for this content — pay via "
                 f"ContentAccessVerifier.payForAccess first")

    return bytes(entry["wrapped_key"])
