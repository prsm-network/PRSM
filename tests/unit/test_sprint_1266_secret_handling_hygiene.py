"""Sprint 1266 — secret-handling hygiene (security audit round 4, findings #6-#10).

  - write_owner_only_file: a canonical atomic 0o600 secret-file writer with NO
    world-readable window (TOCTOU). Replaces write_text-then-chmod at the CLI/node sites.
  - _save_credentials (#6) and join_testnet's env file (#8/#10) now use it.
  - ensure_public_bind_api_key (#7) no longer logs the RAW auto-provisioned API key when it
    persisted to a 0o600 file (it points to the file + a short fingerprint instead).
  - enterprise decrypt (#9) no longer requires the recipient private key on argv (a
    process-listing / shell-history leak); it resolves from env / --privkey-file / --privkey.
"""
from __future__ import annotations

import logging
import os
import stat

import pytest

from prsm.node.identity import write_owner_only_file


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


# ── the canonical writer ─────────────────────────────────────────────────────────

def test_write_owner_only_file_is_0600_with_content(tmp_path):
    p = tmp_path / "sub" / "secret.txt"
    write_owner_only_file(p, "s3cr3t")
    assert p.read_text() == "s3cr3t"
    assert _mode(p) == 0o600
    assert _mode(p.parent) == 0o700


def test_write_owner_only_file_tightens_preexisting(tmp_path):
    p = tmp_path / "secret.txt"
    p.write_text("old")
    os.chmod(p, 0o644)            # pre-existing world-readable file
    write_owner_only_file(p, "new")
    assert p.read_text() == "new"
    assert _mode(p) == 0o600       # tightened


# ── #6 CLI credentials file ──────────────────────────────────────────────────────

def test_save_credentials_is_0600_and_roundtrips(tmp_path, monkeypatch):
    import prsm.cli as cli
    creds = tmp_path / ".prsm" / "credentials.json"
    monkeypatch.setattr(cli, "_CREDENTIALS_FILE", creds)
    cli._save_credentials({"token": "abc", "user": "alice"})
    assert _mode(creds) == 0o600
    assert cli._load_credentials() == {"token": "abc", "user": "alice"}


# ── #7 auto-provisioned node API key not logged in plaintext (persisted path) ─────

def test_public_bind_api_key_not_logged_raw(tmp_path, caplog):
    from prsm.node.node import ensure_public_bind_api_key
    key_path = tmp_path / "node_api.key"
    env: dict = {"PRSM_AUTO_PROVISION_API_KEY": "1"}   # opt into generation on a public bind
    with caplog.at_level(logging.WARNING):
        key = ensure_public_bind_api_key(api_host="0.0.0.0", key_path=key_path, env=env)
    assert key and env.get("PRSM_NODE_API_KEY") == key
    assert _mode(key_path) == 0o600                 # persisted owner-only
    # the raw key must NOT appear in any log record (it lives in the 0600 file)
    assert all(key not in rec.getMessage() for rec in caplog.records), "raw API key leaked to logs"


# ── #9 enterprise privkey not required on argv ───────────────────────────────────

def test_enterprise_privkey_resolves_from_env(monkeypatch):
    from prsm.enterprise.cli import _resolve_recipient_privkey
    import argparse
    monkeypatch.setenv("PRSM_ENTERPRISE_RECIPIENT_PRIVKEY", "envkey123")
    ns = argparse.Namespace(privkey=None, privkey_file=None)
    assert _resolve_recipient_privkey(ns) == "envkey123"


def test_enterprise_privkey_file_takes_over_argv(tmp_path, monkeypatch):
    from prsm.enterprise.cli import _resolve_recipient_privkey
    import argparse
    monkeypatch.delenv("PRSM_ENTERPRISE_RECIPIENT_PRIVKEY", raising=False)
    kf = tmp_path / "key.txt"
    kf.write_text("filekey456\n")
    ns = argparse.Namespace(privkey=None, privkey_file=str(kf))
    assert _resolve_recipient_privkey(ns) == "filekey456"


def test_enterprise_privkey_missing_raises(monkeypatch):
    from prsm.enterprise.cli import _resolve_recipient_privkey
    import argparse
    monkeypatch.delenv("PRSM_ENTERPRISE_RECIPIENT_PRIVKEY", raising=False)
    ns = argparse.Namespace(privkey=None, privkey_file=None)
    with pytest.raises(SystemExit):
        _resolve_recipient_privkey(ns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
