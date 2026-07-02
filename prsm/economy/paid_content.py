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

from typing import Any, Callable, Dict, List, Optional


def publish_paid_content(
    *,
    plaintext: bytes,
    recipients: List[Any],
    royalty_verifier_address: str,
    release_fee_ftns_wei: int,
    key_client: Any,
    publish_ciphertext: Callable[[bytes, bytes], Any],
    content_hash: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Sprint 1352 (arc brick 4, PUBLISHER side) — make a dataset Tier B/C paid-access.

    1. Encrypt the content with a fresh AES-256-GCM key (the ciphertext is served FREELY — a
       node holding it sees only random bytes).
    2. Wrap the content key to the designated buyer(s)' X25519 pubkeys
       (``wrap_content_key_for_deposit``) → the on-chain ``encrypted_key``.
    3. Deposit it on-chain (``KeyDistribution.deposit_key``), naming the royalty VERIFIER + the
       release fee — a consumer who pays that fee to the verifier gets the key released.
    4. Publish the ciphertext to the content layer (freely fetchable by content_hash).

    Deposit BEFORE publish (don't serve orphan ciphertext if the deposit reverts). Returns
    ``{content_hash, deposit_tx, deposit_status, release_fee_ftns_wei, ciphertext, num_recipients}``.

    Args:
      recipients: the buyers' ``EnterpriseRecipient`` (X25519 pubkeys — the DECRYPT identity).
      royalty_verifier_address: the ``IRoyaltyPaymentVerifier`` the release gates on (the
        publisher's choice; the consumer pays a fee this verifier confirms).
      key_client: a ``KeyDistributionClient`` (deposit_key).
      publish_ciphertext: ``(content_hash, ciphertext_bytes) -> ref`` — stores the freely-served
        ciphertext under content_hash (the existing content layer).
      content_hash: override the deposit key; default is sha256(ciphertext) (deterministic, ties
        the deposit to the exact served bytes; a fresh random IV makes each publish distinct).
    """
    import hashlib

    from prsm.storage.encryption import encrypt, generate_key
    from prsm.storage.paid_unlock import (
        serialize_encrypted_content,
        wrap_content_key_for_deposit,
    )

    if not recipients:
        raise ValueError("at least one recipient (buyer X25519 pubkey) is required")
    if int(release_fee_ftns_wei) <= 0:
        raise ValueError("release_fee_ftns_wei must be > 0")

    content_key = generate_key()
    ciphertext_bytes = serialize_encrypted_content(encrypt(plaintext, content_key))
    ch = content_hash if content_hash is not None else hashlib.sha256(ciphertext_bytes).digest()
    if len(ch) != 32:
        raise ValueError(f"content_hash must be 32 bytes, got {len(ch)}")

    wrapped = wrap_content_key_for_deposit(content_key, list(recipients))
    tx, status = key_client.deposit_key(
        ch, wrapped, royalty_verifier_address, int(release_fee_ftns_wei))
    publish_ciphertext(ch, ciphertext_bytes)   # serve the freely-fetchable ciphertext

    return {
        "content_hash": ch,
        "deposit_tx": tx,
        "deposit_status": status,
        "release_fee_ftns_wei": int(release_fee_ftns_wei),
        "ciphertext": ciphertext_bytes,
        "num_recipients": len(recipients),
    }


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
