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
    assert not (tmp_path / "recovered.bin.dec.tmp").exists()  # temp cleaned up


def test_wrong_key_fails_loud(tmp_path):
    key = generate_key()
    other = generate_key()
    src = _write(tmp_path / "plain.bin", _PLAINTEXT)
    ct = tmp_path / "cipher.bin"
    encrypt_content_to_file(src, ct, key)
    with pytest.raises(PaidUnlockError):
        decrypt_content_from_file(ct, tmp_path / "out.bin", other)
    assert not (tmp_path / "out.bin").exists()


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
