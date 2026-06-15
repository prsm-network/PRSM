"""Sprint 1120 — Domain-08 review MEDIUM-6: keyed-HMAC + constant-time integrity tags.

The key-material integrity "tamper detection" was a bare unkeyed sha256 compared with a
plain == — forgeable (an attacker with DB write recomputes the sha256) and timing-leaky.
Now it's an HMAC under the master key, verified with hmac.compare_digest.
"""
from __future__ import annotations

import hashlib
import hmac

import pytest
from cryptography.fernet import Fernet

from prsm.core.cryptography.key_management import SecureKeyStorage


def _storage():
    return SecureKeyStorage(Fernet.generate_key())


def test_key_hash_is_keyed_not_bare_sha256():
    s = _storage()
    data = b"secret-key-material"
    assert s.generate_key_hash(data) != hashlib.sha256(data).hexdigest()
    assert s.generate_key_hash(data) == hmac.new(
        s.master_key, data, hashlib.sha256).hexdigest()


def test_key_hash_is_keyed_per_master():
    a, b = _storage(), _storage()
    data = b"x"
    assert a.generate_key_hash(data) != b.generate_key_hash(data)


def test_verify_key_hash_accepts_correct_tag():
    s = _storage()
    data = b"material"
    tag = s.generate_key_hash(data)
    assert s.verify_key_hash(data, tag) is True


def test_verify_key_hash_rejects_tampered_material():
    s = _storage()
    tag = s.generate_key_hash(b"material")
    # an attacker swaps the material; without the master key they can't forge a tag,
    # and they cannot reuse the old tag for new material
    assert s.verify_key_hash(b"different-material", tag) is False


def test_verify_key_hash_rejects_forged_sha256():
    """The whole point: a forger who knows only sha256 (not the master key) cannot pass."""
    s = _storage()
    data = b"material"
    forged = hashlib.sha256(data).hexdigest()  # the OLD, unkeyed scheme
    assert s.verify_key_hash(data, forged) is False


def test_verify_key_hash_handles_empty_expected():
    s = _storage()
    assert s.verify_key_hash(b"x", "") is False
    assert s.verify_key_hash(b"x", None) is False  # type: ignore[arg-type]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
