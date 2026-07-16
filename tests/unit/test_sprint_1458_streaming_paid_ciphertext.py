"""Sprint 1458 — streaming Tier B/C ciphertext codec (large paid content).

serialize_encrypted_content (JSON + base64 of the whole ciphertext) is in-memory by nature, so a
multi-GiB paid dataset can't be published/consumed through it. encrypt_content_to_file /
decrypt_content_from_file stream a file with a bounded footprint, composing StreamingEncryptor/
StreamingDecryptor. This is the first, self-contained increment of the Tier B/C large-file streaming
work — round-trip + tamper + wrong-key verified in isolation (no network, no wiring yet).
"""
from __future__ import annotations

import os

import pytest

from prsm.storage.encryption import generate_key
from prsm.storage.paid_unlock import (
    PaidUnlockError,
    decrypt_content_from_file,
    encrypt_content_to_file,
    is_streaming_ciphertext_file,
)

# A size spanning many 1 MiB chunks with a non-aligned tail — exercises chunk boundaries.
_PLAINTEXT = (b"proprietary paid dataset row; " * 200_000) + b"tail-bytes-unaligned!!"


def _write(path, data):
    path.write_bytes(data)
    return path


def test_streaming_ciphertext_round_trips_byte_identical(tmp_path):
    key = generate_key()
    src = _write(tmp_path / "plain.bin", _PLAINTEXT)
    ct = tmp_path / "cipher.bin"
    encrypt_content_to_file(src, ct, key)

    assert is_streaming_ciphertext_file(ct)                 # magic-detected
    assert ct.read_bytes()[:16] != _PLAINTEXT[:16]          # actually encrypted (not the plaintext)

    out = tmp_path / "recovered.bin"
    decrypt_content_from_file(ct, out, key)
    assert out.read_bytes() == _PLAINTEXT                    # byte-identical across chunk boundaries


def test_empty_plaintext_round_trips(tmp_path):
    key = generate_key()
    src = _write(tmp_path / "empty.bin", b"")
    ct = tmp_path / "empty.ct"
    encrypt_content_to_file(src, ct, key)
    out = tmp_path / "empty.out"
    decrypt_content_from_file(ct, out, key)
    assert out.read_bytes() == b""


def test_tampered_ciphertext_fails_loud_and_leaves_no_plaintext(tmp_path):
    key = generate_key()
    src = _write(tmp_path / "plain.bin", _PLAINTEXT)
    ct = tmp_path / "cipher.bin"
    encrypt_content_to_file(src, ct, key)

    # Flip a byte in the ciphertext BODY (past the header, before the tag trailer).
    raw = bytearray(ct.read_bytes())
    mid = len(raw) // 2
    raw[mid] ^= 0xFF
    ct.write_bytes(bytes(raw))

    out = tmp_path / "recovered.bin"
    with pytest.raises(PaidUnlockError):
        decrypt_content_from_file(ct, out, key)
    assert not out.exists()                                 # no plaintext promoted on auth failure
    assert not list(tmp_path.glob(".prsm-dec-*"))           # decrypt temp cleaned up (no leak)


def test_wrong_key_fails_loud(tmp_path):
    key = generate_key()
    other = generate_key()
    src = _write(tmp_path / "plain.bin", _PLAINTEXT)
    ct = tmp_path / "cipher.bin"
    encrypt_content_to_file(src, ct, key)
    with pytest.raises(PaidUnlockError):
        decrypt_content_from_file(ct, tmp_path / "out.bin", other)
    assert not (tmp_path / "out.bin").exists()


def test_magic_valid_but_truncated_header_raises_paid_unlock_error(tmp_path):
    # sp1458 hardening (audit LOW-1): a file that carries the magic but is truncated inside the
    # header (no key_id length byte) must raise the documented PaidUnlockError, NOT a raw struct.error
    # that would slip past a caller catching only PaidUnlockError. Fail-closed either way, but the
    # exception contract must hold.
    from prsm.storage.paid_unlock import _STREAM_MAGIC
    key = generate_key()
    truncated = _write(tmp_path / "just-magic.ct", _STREAM_MAGIC)   # 8 bytes: magic, nothing after
    assert is_streaming_ciphertext_file(truncated)
    with pytest.raises(PaidUnlockError):
        decrypt_content_from_file(truncated, tmp_path / "out.bin", key)
    assert not (tmp_path / "out.bin").exists()


def test_decrypt_temp_is_unpredictable_and_colocated(tmp_path, monkeypatch):
    # sp1458 hardening (audit LOW-2): the decrypt temp must NOT be the predictable "<dest>.dec.tmp"
    # (a co-located attacker could pre-create/observe it, and concurrent decrypts to one dest would
    # clobber). It must be an mkstemp file (0600, unique) in the SAME dir as dest (for atomic replace).
    import os as _os
    key = generate_key()
    src = _write(tmp_path / "plain.bin", _PLAINTEXT)
    ct = tmp_path / "cipher.bin"
    encrypt_content_to_file(src, ct, key)

    out = tmp_path / "recovered.bin"
    captured = {}
    real_replace = _os.replace

    def _cap_replace(a, b):
        captured["tmp"] = str(a)
        captured["mode"] = _os.stat(a).st_mode & 0o777
        return real_replace(a, b)

    monkeypatch.setattr(_os, "replace", _cap_replace)
    decrypt_content_from_file(ct, out, key)
    assert out.read_bytes() == _PLAINTEXT
    tmp = captured["tmp"]
    assert tmp != str(out) + ".dec.tmp"                     # NOT the predictable name
    assert _os.path.dirname(tmp) == str(tmp_path)           # co-located → atomic os.replace
    assert captured["mode"] == 0o600                        # not umask-dependent; plaintext window is private


def test_non_streaming_file_rejected(tmp_path):
    key = generate_key()
    junk = _write(tmp_path / "junk.bin", b"not a PRSM streaming ciphertext at all")
    assert not is_streaming_ciphertext_file(junk)
    with pytest.raises(PaidUnlockError):
        decrypt_content_from_file(junk, tmp_path / "out.bin", key)


def test_bounded_memory_no_full_read(tmp_path, monkeypatch):
    # Prove neither side reads the whole file at once: fail if anything calls .read() with no size
    # (a whole-file read). Both codecs must read in bounded blocks.
    import builtins
    real_open = builtins.open
    key = generate_key()
    src = _write(tmp_path / "plain.bin", _PLAINTEXT)
    ct = tmp_path / "cipher.bin"

    class _GuardedFile:
        def __init__(self, fh):
            self._fh = fh

        def read(self, n=-1):
            assert n is not None and n >= 0, "unbounded whole-file read is not allowed"
            return self._fh.read(n)

        def write(self, b):
            return self._fh.write(b)

        def seek(self, *a):
            return self._fh.seek(*a)

        def __enter__(self):
            self._fh.__enter__()
            return self

        def __exit__(self, *a):
            return self._fh.__exit__(*a)

    def _guarded_open(path, mode="r", *a, **k):
        fh = real_open(path, mode, *a, **k)
        if "b" in mode and "r" in mode:
            return _GuardedFile(fh)
        return fh

    monkeypatch.setattr(builtins, "open", _guarded_open)
    encrypt_content_to_file(src, ct, key)
    decrypt_content_from_file(ct, tmp_path / "out.bin", key)
    assert (tmp_path / "out.bin").read_bytes() == _PLAINTEXT


def test_build_paid_content_from_path_round_trips_via_buyer_unwrap(tmp_path):
    # ★ PRODUCER end-to-end: stream-encrypt a large file + wrap the key → a buyer unwraps the content
    # key with their X25519 privkey and stream-decrypts the ciphertext file back to the plaintext.
    import hashlib
    import json

    from prsm.economy.paid_content import build_paid_content_from_path
    from prsm.enterprise.recipient_encryption import (
        EncryptedPayload as _RecipientPayload,
        EnterpriseRecipient,
        decrypt_for_recipient,
        generate_recipient_keypair,
    )
    from prsm.storage.encryption import AESKey
    from prsm.storage.paid_unlock import decrypt_content_from_file

    plaintext = (b"proprietary paid dataset row; " * 100_000) + b"unaligned-tail"
    src = tmp_path / "plain.bin"
    src.write_bytes(plaintext)
    buyer_priv, buyer_pub = generate_recipient_keypair()
    ct = tmp_path / "cipher.bin"

    result = build_paid_content_from_path(
        src, ct, [EnterpriseRecipient(identifier="buyer", x25519_pubkey_b64=buyer_pub)])

    # content_hash is sha256 of the ciphertext FILE (the authoritative deposit/fetch key).
    assert result["content_hash"] == hashlib.sha256(ct.read_bytes()).digest()
    assert result["commitment"] == hashlib.sha256(result["wrapped_key"]).digest()
    assert ct.read_bytes()[:16] != plaintext[:16]      # actually encrypted

    # CONSUMER unwrap → content key bytes (key_id NOT recovered — proves the relaxed check) → decrypt.
    wrapped = _RecipientPayload.from_dict(json.loads(result["wrapped_key"]))
    content_key_bytes = decrypt_for_recipient(wrapped, buyer_priv)
    key = AESKey(key_id="placeholder-unknown-to-consumer", key_bytes=content_key_bytes)
    out = tmp_path / "recovered.bin"
    decrypt_content_from_file(ct, out, key)
    assert out.read_bytes() == plaintext


def test_build_paid_content_from_path_requires_recipient(tmp_path):
    import pytest as _pytest
    from prsm.economy.paid_content import build_paid_content_from_path
    src = tmp_path / "p.bin"
    src.write_bytes(b"data")
    with _pytest.raises(ValueError):
        build_paid_content_from_path(src, tmp_path / "ct.bin", [])


def test_reconstruct_paid_content_from_file_full_consumer_round_trip(tmp_path):
    # ★ CONSUMER end-to-end: producer publishes → consumer unwraps the payment-released key + streams
    # the ciphertext file back to plaintext, via the packaged reconstruct_paid_content_from_file.
    from prsm.economy.paid_content import build_paid_content_from_path
    from prsm.enterprise.recipient_encryption import (
        EnterpriseRecipient, generate_recipient_keypair)
    from prsm.storage.paid_unlock import reconstruct_paid_content_from_file

    plaintext = (b"paid dataset " * 90_000) + b"tail!"
    src = tmp_path / "plain.bin"
    src.write_bytes(plaintext)
    buyer_priv, buyer_pub = generate_recipient_keypair()
    ct = tmp_path / "cipher.bin"
    result = build_paid_content_from_path(
        src, ct, [EnterpriseRecipient(identifier="b", x25519_pubkey_b64=buyer_pub)])

    out = tmp_path / "recovered.bin"
    reconstruct_paid_content_from_file(result["wrapped_key"], buyer_priv, ct, out)
    assert out.read_bytes() == plaintext


def test_reconstruct_from_file_wrong_buyer_key_fails_loud(tmp_path):
    from prsm.economy.paid_content import build_paid_content_from_path
    from prsm.enterprise.recipient_encryption import (
        EnterpriseRecipient, generate_recipient_keypair)
    from prsm.storage.paid_unlock import reconstruct_paid_content_from_file

    src = tmp_path / "plain.bin"
    src.write_bytes(b"secret dataset " * 50_000)
    _buyer_priv, buyer_pub = generate_recipient_keypair()
    other_priv, _other_pub = generate_recipient_keypair()   # NOT a designated recipient
    ct = tmp_path / "cipher.bin"
    result = build_paid_content_from_path(
        src, ct, [EnterpriseRecipient(identifier="b", x25519_pubkey_b64=buyer_pub)])

    out = tmp_path / "recovered.bin"
    with pytest.raises(PaidUnlockError):
        reconstruct_paid_content_from_file(result["wrapped_key"], other_priv, ct, out)
    assert not out.exists()
