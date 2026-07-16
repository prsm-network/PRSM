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


def key_commitment(wrapped_key: bytes) -> bytes:
    """sp1357 (F1 redesign, R1) — the on-chain COMMITMENT to a wrapped content key: sha256(wrapped).

    ONLY this 32-byte commitment is deposited on-chain — never the wrapped key itself, which would
    be world-readable in contract storage (the B5 F1 critical). The consumer fetches the wrapped key
    OFF-chain from a payment-gated endpoint and verifies it against this commitment (via
    ``verify_key_commitment``) before trusting it, so a lying publisher cannot serve a wrong key.
    sha256 (not Ethereum keccak) is fine: the contract stores the commitment as opaque bytes and
    never hashes it — only the Python publisher and consumer compute it, and both use sha256."""
    import hashlib
    return hashlib.sha256(bytes(wrapped_key)).digest()


def verify_key_commitment(wrapped_key: bytes, commitment: bytes) -> bool:
    """sp1357 — constant-time check that ``wrapped_key`` matches the on-chain ``commitment``."""
    import hmac
    return hmac.compare_digest(key_commitment(wrapped_key), bytes(commitment))


def serialize_encrypted_content(payload: Any) -> bytes:
    """sp1352 — serialize a ``prsm.storage.encryption.EncryptedPayload`` (the freely-served Tier
    B/C ciphertext) to JSON bytes for transport/retrieval. The publisher serves these bytes; the
    consumer parses them back with ``deserialize_encrypted_content`` before reconstruct."""
    import base64
    return json.dumps({
        "v": 1,
        "ciphertext_b64": base64.b64encode(bytes(payload.ciphertext)).decode("ascii"),
        "iv_b64": base64.b64encode(bytes(payload.iv)).decode("ascii"),
        "auth_tag_b64": base64.b64encode(bytes(payload.auth_tag)).decode("ascii"),
        "key_id": str(payload.key_id),
    }).encode("utf-8")


def deserialize_encrypted_content(data: bytes) -> Any:
    """sp1352 — parse the served Tier B/C ciphertext bytes back into a storage
    ``EncryptedPayload`` (for ``reconstruct_paid_content``). Raises PaidUnlockError on garbage."""
    import base64
    from prsm.storage.encryption import EncryptedPayload
    try:
        d = json.loads(data)
        return EncryptedPayload(
            ciphertext=base64.b64decode(d["ciphertext_b64"]),
            iv=base64.b64decode(d["iv_b64"]),
            auth_tag=base64.b64decode(d["auth_tag_b64"]),
            key_id=str(d["key_id"]))
    except Exception as exc:  # noqa: BLE001
        raise PaidUnlockError(f"malformed encrypted content envelope: {exc}") from exc


# ── sp1458: streaming Tier B/C ciphertext codec (LARGE paid content) ──────────────────────────────
# serialize_encrypted_content (JSON + base64 of the whole ciphertext) is in-memory by nature, so a
# multi-GiB paid dataset can't be published/consumed through it. This binary format streams a file with
# a bounded memory footprint, composing the proven StreamingEncryptor/StreamingDecryptor. It is ADDITIVE:
# a consumer detects the magic (is_streaming_ciphertext_file) and routes here; the JSON envelope path
# (serve small paid content) is unchanged, so the twice-audited paid-decrypt money gate is untouched.
#
# Layout:  _STREAM_MAGIC(8) | u8 key_id_len | key_id | iv(12) | ciphertext… | auth_tag(16 TRAILER)
# The GCM auth tag is a TRAILER because it is only known at finalize(); a seekable file reads it first.
_STREAM_MAGIC = b"PRSMSC1\x00"
_STREAM_CHUNK = 1024 * 1024  # 1 MiB read granularity
_STREAM_IV_LEN = 12
_STREAM_TAG_LEN = 16


def is_streaming_ciphertext_file(path: Any) -> bool:
    """True iff ``path`` begins with the streaming-ciphertext magic (cheap header read; never raises)."""
    try:
        with open(path, "rb") as fh:
            return fh.read(len(_STREAM_MAGIC)) == _STREAM_MAGIC
    except OSError:
        return False


def encrypt_content_to_file(src_path: Any, dest_path: Any, content_key: Any) -> None:
    """sp1458 — stream-encrypt the plaintext file at ``src_path`` to ``dest_path`` as a Tier B/C
    ciphertext (AES-256-GCM under ``content_key``) with NO full-content buffering. Byte-for-byte
    decryptable by ``decrypt_content_from_file``. Raises PaidUnlockError on an over-long key_id."""
    import struct
    from prsm.storage.encryption import StreamingEncryptor
    enc = StreamingEncryptor(content_key)
    key_id = str(content_key.key_id).encode("utf-8")
    if len(key_id) > 255:
        raise PaidUnlockError("key_id too long for the streaming ciphertext format (>255 bytes)")
    with open(src_path, "rb") as fin, open(dest_path, "wb") as fout:
        fout.write(_STREAM_MAGIC)
        fout.write(struct.pack("B", len(key_id)))
        fout.write(key_id)
        fout.write(enc.iv)
        for block in iter(lambda: fin.read(_STREAM_CHUNK), b""):
            fout.write(enc.encrypt_chunk(block))
        fout.write(enc.finalize())  # 16-byte GCM tag trailer


def decrypt_content_from_file(src_path: Any, dest_path: Any, content_key: Any) -> None:
    """sp1458 — inverse of ``encrypt_content_to_file``: stream-decrypt ``src_path`` → ``dest_path``
    (AES-256-GCM), FAIL-LOUD. Plaintext is written to a TEMP file and promoted to ``dest_path`` ONLY
    after the GCM tag verifies (GCM chunks are unauthenticated until finalize), so a tampered or
    wrong-key ciphertext never yields a plaintext file. Raises PaidUnlockError on any failure."""
    import os
    import struct
    from pathlib import Path as _Path
    from prsm.storage.encryption import StreamingDecryptor
    src = _Path(src_path)
    dest = _Path(dest_path)
    total = src.stat().st_size
    with open(src, "rb") as fin:
        if fin.read(len(_STREAM_MAGIC)) != _STREAM_MAGIC:
            raise PaidUnlockError("not a streaming Tier B/C ciphertext (bad magic)")
        kid_len = struct.unpack("B", fin.read(1))[0]
        _ = fin.read(kid_len)  # header key_id — informational only; the GCM tag is the integrity gate
        iv = fin.read(_STREAM_IV_LEN)
        header_len = len(_STREAM_MAGIC) + 1 + kid_len + _STREAM_IV_LEN
        ct_len = total - header_len - _STREAM_TAG_LEN
        if len(iv) != _STREAM_IV_LEN or ct_len < 0:
            raise PaidUnlockError("streaming ciphertext truncated (no room for iv/tag)")
        # NOTE: the caller's content_key need NOT carry the original key_id — the paid-decrypt
        # consumer recovers only the raw key BYTES from the wrapped key, not the key_id. Decryption
        # uses content_key.key_bytes and the file's iv/tag; a wrong key fails the GCM tag at finalize.
        fin.seek(total - _STREAM_TAG_LEN)      # the tag is the trailer; the decryptor needs it up front
        tag = fin.read(_STREAM_TAG_LEN)
        fin.seek(header_len)
        dec = StreamingDecryptor(content_key, iv, tag)
        tmp = _Path(str(dest) + ".dec.tmp")
        remaining = ct_len
        try:
            with open(tmp, "wb") as fout:
                while remaining > 0:
                    block = fin.read(min(_STREAM_CHUNK, remaining))
                    if not block:
                        break
                    remaining -= len(block)
                    fout.write(dec.decrypt_chunk(block))
            dec.finalize()                      # raises on tamper/wrong-key → temp discarded below
        except Exception as exc:  # noqa: BLE001 — any auth/IO failure must not leave a plaintext file
            tmp.unlink(missing_ok=True)
            raise PaidUnlockError(
                f"streaming ciphertext failed authentication (tampered or wrong key): {exc}") from exc
        os.replace(tmp, dest)                   # promote ONLY after the tag verifies


def reconstruct_paid_content_from_file(
    released_wrapped_key: bytes,
    recipient_privkey_b64: str,
    ciphertext_path: Any,
    dest_path: Any,
) -> None:
    """sp1458 — CONSUMER side, STREAMING: the large-file equivalent of ``reconstruct_paid_content``.

    Unwrap the content key from the payment-released ``released_wrapped_key`` with the buyer's X25519
    private key, then stream-decrypt the retrieved ciphertext FILE (``encrypt_content_to_file`` format,
    fetched via /content/retrieve-stream) to ``dest_path`` with a bounded memory footprint. FAIL-LOUD
    (PaidUnlockError) on a malformed/wrong wrapped key or a tampered ciphertext — no plaintext file is
    left behind (decrypt_content_from_file promotes only after the GCM tag verifies)."""
    from prsm.enterprise.recipient_encryption import (
        EncryptedPayload as _RecipientPayload,
        decrypt_for_recipient,
    )
    from prsm.storage.encryption import AES_KEY_BYTES, AESKey
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
    key = AESKey(key_id="paid-unlock", key_bytes=content_key_bytes)
    decrypt_content_from_file(ciphertext_path, dest_path, key)


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
