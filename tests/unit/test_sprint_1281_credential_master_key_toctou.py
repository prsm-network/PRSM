"""Sprint 1281 — integration credential master key written atomically 0o600 (round 7, MED).

credential_manager._initialize_encryption wrote the Fernet master key (which decrypts every
stored integration credential) via `open(...,'wb')` then `os.chmod(0o600)` — a world-readable
TOCTOU window the sp1266 sweep missed (it's binary, in a different module). Fix: reuse the
canonical write_owner_only_file, now generalized to accept bytes.
"""
from __future__ import annotations

import os
import stat

from prsm.node.identity import write_owner_only_file


def _mode(p):
    return stat.S_IMODE(os.stat(p).st_mode)


def test_write_owner_only_file_accepts_bytes(tmp_path):
    p = tmp_path / "master.key"
    payload = b"\x00\x01fernet-key-bytes\xfe\xff"
    write_owner_only_file(p, payload)
    assert p.read_bytes() == payload
    assert _mode(p) == 0o600


def test_write_owner_only_file_still_accepts_str(tmp_path):
    p = tmp_path / "s.txt"
    write_owner_only_file(p, "plain text")
    assert p.read_text() == "plain text"
    assert _mode(p) == 0o600


def test_credential_manager_master_key_is_0600(tmp_path, monkeypatch):
    from cryptography.fernet import Fernet
    # import the CLASS directly from the submodule (the package __init__ re-exports a
    # credential_manager INSTANCE under the same name, shadowing the module)
    from prsm.core.integrations.config.credential_manager import CredentialManager

    mgr = CredentialManager.__new__(CredentialManager)  # bypass heavy __init__
    mgr.master_key_file = tmp_path / "master.key"
    mgr.master_key = None
    mgr.cipher = None
    mgr._initialize_encryption()
    assert mgr.master_key_file.exists()
    assert _mode(mgr.master_key_file) == 0o600
    # the key round-trips + actually decrypts (real Fernet)
    token = Fernet(mgr.master_key).encrypt(b"secret")
    assert Fernet(mgr.master_key).decrypt(token) == b"secret"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
