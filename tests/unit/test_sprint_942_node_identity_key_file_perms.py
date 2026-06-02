"""Sprint 942 — node identity private-key file permissions (auth/identity review).

save_node_identity wrote the identity JSON — which contains the node's ed25519
PRIVATE key — with `Path.write_text`, leaving it world-readable (default 0o644).
Any other local user or co-tenant container process could read the node's
signing key and impersonate it. Fix: write the file 0o600 (created with that
mode, no world-readable window) and its parent dir 0o700.
"""
from __future__ import annotations

import stat

import pytest

from prsm.node.identity import (
    generate_node_identity,
    load_node_identity,
    save_node_identity,
)


def test_saved_identity_file_is_owner_only(tmp_path):
    ident = generate_node_identity("test")
    p = tmp_path / "keys" / "identity.json"
    save_node_identity(ident, p)
    assert p.exists()
    assert stat.S_IMODE(p.stat().st_mode) == 0o600, oct(p.stat().st_mode)
    assert stat.S_IMODE(p.parent.stat().st_mode) == 0o700, oct(p.parent.stat().st_mode)


def test_save_tightens_a_preexisting_world_readable_file(tmp_path):
    p = tmp_path / "identity.json"
    p.write_text("{}")
    p.chmod(0o644)
    save_node_identity(generate_node_identity("test"), p)
    assert stat.S_IMODE(p.stat().st_mode) == 0o600


def test_save_load_roundtrip_preserves_key(tmp_path):
    ident = generate_node_identity("rt")
    p = tmp_path / "id.json"
    save_node_identity(ident, p)
    loaded = load_node_identity(p)
    assert loaded is not None
    assert loaded.node_id == ident.node_id
    assert loaded.private_key_bytes == ident.private_key_bytes
    assert loaded.public_key_bytes == ident.public_key_bytes
