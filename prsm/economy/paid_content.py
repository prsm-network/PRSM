"""Sprint 1351 (Tier B/C paid-decrypt consumer arc, brick 3) — the pay -> unlock orchestration.

Brick 1 (prsm.storage.paid_unlock) = offline reconstruct; brick 2
(prsm.economy.web3.key_acquisition) = on-chain key acquisition. This brick is the consumer glue
that composes them into the flagship paid-content action:

    settle the release fee  ->  acquire the released key (B2)  ->  retrieve the ciphertext
                            ->  reconstruct the plaintext (B1)

The fee-settlement step and the content retrieval are INJECTABLE, deliberately:
  * The royalty verifier (``IRoyaltyPaymentVerifier``) is chosen by the publisher at deposit
    time — there is no single production verifier contract yet, so HOW a consumer pays a fee
    that ``verifyPayment`` recognizes is publisher-specific. B3 takes a ``settle_fee`` callable
    (pass ``None`` if the fee is already settled) rather than hard-coding one payment path.
  * Retrieval is the existing content layer (``retrieve_content(content_hash) -> EncryptedPayload``).

Everything is FAIL-LOUD (KeyNotReleasedError from B2 on an unpaid fee → "pay first"; PaidUnlockError
from B1 on a wrong key / tampered content) — a consumer never proceeds on a missing/garbage key.
This is what brick 4's SDK/CLI/MCP surface wraps.
"""
from __future__ import annotations

from typing import Any, Callable, Optional


def pay_and_unlock(
    *,
    content_hash: bytes,
    recipient: str,
    recipient_privkey_b64: str,
    key_client: Any,
    retrieve_content: Callable[[bytes], Any],
    settle_fee: Optional[Callable[[], Any]] = None,
    from_block: int = 0,
) -> bytes:
    """Consumer pay -> unlock for a Tier B/C paid-access dataset. Returns the plaintext.

    Args:
      content_hash: the 32-byte KeyDistribution content id (the deposit key).
      recipient: the buyer's ON-CHAIN address — the payment/release identity (verifyPayment
                 binds to it).
      recipient_privkey_b64: the buyer's X25519 private key — the DECRYPT identity (the content
                 key was sealed to its pubkey at deposit).
      key_client: a KeyDistributionClient (or compatible) for release + event reads.
      retrieve_content: maps ``content_hash`` to the served ciphertext
                 (``prsm.storage.encryption.EncryptedPayload``).
      settle_fee: optional zero-arg callable that pays the release fee to the content's royalty
                 verifier so ``verifyPayment(recipient, content_hash, fee)`` becomes true. Pass
                 ``None`` when the fee is already settled (then release simply reads/triggers).

    Raises KeyNotReleasedError (unpaid / not-deposited / no-event) or PaidUnlockError (retrieval
    miss / wrong key / tampered ciphertext). Ordering: pay BEFORE release, so the release's
    verifyPayment check passes.
    """
    from prsm.economy.web3.key_acquisition import acquire_released_key
    from prsm.storage.paid_unlock import PaidUnlockError, reconstruct_paid_content

    # 1. Pay the release fee first (so the on-chain verifyPayment gate passes at release time).
    if settle_fee is not None:
        settle_fee()

    # 2. Acquire the released encrypted content-key (B2 — idempotent, fail-loud on unpaid).
    released_key = acquire_released_key(
        key_client, content_hash, recipient, from_block=from_block)

    # 3. Retrieve the freely-served ciphertext.
    content = retrieve_content(content_hash)
    if content is None:
        raise PaidUnlockError(
            f"paid for + released the key, but the ciphertext for content "
            f"{bytes(content_hash).hex()[:12]}… is not retrievable (no provider has it?).")

    # 4. Reconstruct the plaintext (B1 — unwrap key with the buyer's X25519 key + decrypt).
    return reconstruct_paid_content(released_key, recipient_privkey_b64, content)
