"""Sprint 1121 — Domain-08 review HIGH-4 remainder: encrypt stored private keys at the
PKCS8 layer (BestAvailableEncryption) instead of NoEncryption.

Defense-in-depth: the Fernet wrapper still applies, but a Fernet-layer compromise no
longer yields plaintext private keys — the PKCS8 is password-encrypted under a passphrase
DERIVED from the master key with a distinct domain label. get_key transparently reveals
the plaintext PEM so the caller contract is unchanged; legacy NoEncryption PEM + symmetric
raw bytes pass through.
"""
from __future__ import annotations

import hashlib
import hmac

import pytest
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from prsm.core.cryptography.key_management import SecureKeyStorage


def _storage():
    return SecureKeyStorage(Fernet.generate_key())


def _encrypted_pkcs8(storage, key):
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.BestAvailableEncryption(
            storage.pkcs8_passphrase()),
    )


def test_pkcs8_passphrase_is_derived_not_the_master_key():
    s = _storage()
    pp = s.pkcs8_passphrase()
    assert isinstance(pp, bytes) and len(pp) == 32
    # distinct from the Fernet master key (so the two layers don't share the secret)
    assert pp != s.master_key
    assert pp == hmac.new(
        s.master_key, b"prsm-pkcs8-key-encryption-v1", hashlib.sha256).digest()


def test_pkcs8_passphrase_is_keyed_per_master():
    assert _storage().pkcs8_passphrase() != _storage().pkcs8_passphrase()


def test_encrypted_pkcs8_is_not_plaintext_and_reveals_round_trip():
    s = _storage()
    key = ed25519.Ed25519PrivateKey.generate()
    enc = _encrypted_pkcs8(s, key)
    # at rest it's an ENCRYPTED PKCS8 blob, not a plaintext private key
    assert b"ENCRYPTED PRIVATE KEY" in enc
    assert b"BEGIN PRIVATE KEY" not in enc[:40]
    # reveal returns the plaintext PKCS8 PEM, matching the original key
    revealed = s.reveal_private_pem(enc)
    assert b"BEGIN PRIVATE KEY" in revealed
    reloaded = serialization.load_pem_private_key(revealed, password=None)
    assert reloaded.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    ) == key.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
        serialization.NoEncryption())


def test_reveal_passes_through_legacy_noencryption_pem():
    """Back-compat: an old NoEncryption PEM (stored before this sprint) is returned
    unchanged."""
    s = _storage()
    key = ed25519.Ed25519PrivateKey.generate()
    legacy = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption())
    assert s.reveal_private_pem(legacy) == legacy


def test_reveal_passes_through_symmetric_raw_bytes():
    s = _storage()
    raw = b"\x00\x11\x22symmetric-key-material"
    assert s.reveal_private_pem(raw) == raw


def test_wrong_master_cannot_decrypt_the_pkcs8():
    s1 = _storage()
    key = ed25519.Ed25519PrivateKey.generate()
    enc = _encrypted_pkcs8(s1, key)
    s2 = _storage()  # different master → different passphrase
    with pytest.raises(Exception):
        s2.reveal_private_pem(enc)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
