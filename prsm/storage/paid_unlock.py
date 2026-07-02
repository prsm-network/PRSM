"""Sprint 1349 (Tier B/C paid-decrypt consumer arc, brick 1) — the offline reconstruct primitive.

The Vision's Tier B/C model: content ciphertext is served FREELY (a node holding it sees only
random bytes), and the decryption key is released to a consumer through the on-chain
``KeyDistribution`` contract ONLY on verified royalty payment. The crypto (recipient-wrap +
AES-GCM) and the KeyDistribution client (deposit/release/KeyReleased) already exist; what was
missing is the CONSUMER orchestration that ties *pay → key released → decrypt* together.

This brick is the pure, offline core of that orchestration — no chain, no payment — so the
money/chain wiring (later bricks) has a proven, tested crypto foundation to build on:

  * PUBLISHER: ``wrap_content_key_for_deposit(content_key, recipients)`` → the ``encrypted_key``
    bytes to pass to ``KeyDistributionClient.deposit_key``. The content itself is separately
    AES-GCM-encrypted with ``content_key`` (via prsm.storage.encryption) and served freely.
  * CONSUMER (after paying → the key surfaces in the KeyReleased event):
    ``reconstruct_paid_content(released_wrapped_key, buyer_privkey, content)`` → plaintext.
    Unwrap the content key with the buyer's X25519 private key, then AES-GCM-decrypt the served
    ciphertext. FAIL-LOUD: a wrong key, a corrupted wrap, or tampered ciphertext raises — never
    returns partial/garbage plaintext.

v1 scope = Tier B (recipient-wrapped key: the publisher deposits the content key sealed to the
designated buyer(s)). Tier C (threshold/Shamir-split key via combine_shares_and_decrypt) is a
sibling reconstruct handled in a follow-on brick.
"""
from __future__ import annotations

import json
from typing import Any, List


class PaidUnlockError(Exception):
    """Reconstruct failed — wrong buyer key, malformed wrapped key, or tampered content.
    Fail-loud so a consumer never mistakes garbage for the real dataset."""


def wrap_content_key_for_deposit(content_key: Any, recipients: List[Any]) -> bytes:
    """PUBLISHER side — wrap the content-encryption key to the buyer(s) as the on-chain
    ``encrypted_key`` for ``KeyDistributionClient.deposit_key``. The wrapped key is released
    VERBATIM (in the KeyReleased event) to a consumer who has paid; only a designated buyer's
    X25519 private key unwraps it. Returns JSON bytes (the on-wire encrypted_key)."""
    from prsm.enterprise.recipient_encryption import encrypt_for_recipients
    if not recipients:
        raise PaidUnlockError("at least one recipient (buyer) is required to wrap the key")
    key_bytes = getattr(content_key, "key_bytes", content_key)
    payload = encrypt_for_recipients(bytes(key_bytes), list(recipients))
    return json.dumps(payload.to_dict()).encode("utf-8")


def reconstruct_paid_content(
    released_wrapped_key: bytes,
    recipient_privkey_b64: str,
    content: Any,
) -> bytes:
    """CONSUMER side (post-payment) — recover the plaintext from the released wrapped key + the
    retrieved ciphertext.

    ``released_wrapped_key`` = the KeyReleased event's ``encrypted_key`` bytes (JSON of the
    recipient-wrapped content key, as produced by ``wrap_content_key_for_deposit``).
    ``content`` = the retrieved ``prsm.storage.encryption.EncryptedPayload`` (AES-GCM ciphertext).

    Unwraps the content key with the buyer's X25519 private key, then AES-GCM-decrypts the
    content. Raises ``PaidUnlockError`` on any failure (fail-loud)."""
    from prsm.enterprise.recipient_encryption import (
        EncryptedPayload as _RecipientPayload,
        decrypt_for_recipient,
    )
    from prsm.storage.encryption import AES_KEY_BYTES, AESKey
    from prsm.storage.encryption import decrypt as _aes_decrypt

    try:
        wrapped = _RecipientPayload.from_dict(json.loads(released_wrapped_key))
    except Exception as exc:  # noqa: BLE001
        raise PaidUnlockError(f"malformed released wrapped key: {exc}") from exc

    try:
        content_key_bytes = decrypt_for_recipient(wrapped, recipient_privkey_b64)
    except Exception as exc:  # noqa: BLE001 — wrong buyer key / not a designated recipient
        raise PaidUnlockError(
            f"could not unwrap the content key with this private key "
            f"(paid for the wrong content, or not a designated buyer?): {exc}") from exc

    if len(content_key_bytes) != AES_KEY_BYTES:
        raise PaidUnlockError(
            f"unwrapped key is {len(content_key_bytes)} bytes, expected {AES_KEY_BYTES} "
            f"(the deposited key was not a content-encryption key)")

    key = AESKey(key_id=getattr(content, "key_id", "paid-unlock"), key_bytes=content_key_bytes)
    try:
        return _aes_decrypt(content, key)
    except Exception as exc:  # noqa: BLE001 — AES-GCM auth failure = wrong key or tampered bytes
        raise PaidUnlockError(
            f"content decryption failed (wrong key or tampered ciphertext): {exc}") from exc
