"""Sprint 922 — type manifest decrypt/decode failures (content data-plane review #2).

ContentStore._decode_manifest let a raw cryptography.InvalidTag (AEAD tag
mismatch on a corrupt/tampered ciphertext or wrong/insufficient key shares) or a
UnicodeDecodeError/JSON error propagate. Downstream the broad `except` in
ContentProvider._fetch_local_via_bt logged that cryptic exception and returned
None → the failure was indistinguishable from "content not found". A
tampered/corrupt manifest looked like absence.

Fix: _decode_manifest now raises a TYPED ManifestError with a legible message.
Behaviour is otherwise unchanged (callers still fall back / return not-found for
availability) — only the surfaced/logged error is now clear enough to triage
(corruption/tampering vs genuine absence). NOT silent corruption: the crypto
correctly fails closed; this is about error legibility.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.storage.content_store import ContentStore
from prsm.storage.exceptions import ManifestError


@pytest.mark.asyncio
async def test_tampered_ciphertext_raises_manifest_error_not_invalidtag(tmp_path):
    s = ContentStore(data_dir=str(tmp_path), node_id="n")
    art = await s.store_local_with_artifacts(b"secret content payload")
    bad = bytearray(art.encrypted_manifest)
    bad[0] ^= 0xFF   # corrupt the ciphertext → AEAD tag mismatch
    with pytest.raises(ManifestError):
        s._decode_manifest(bytes(bad), art.key_shares)


def test_non_utf8_decrypt_raises_manifest_error(tmp_path):
    # decrypt succeeds but yields non-UTF-8 / non-JSON garbage → ManifestError,
    # not a raw UnicodeDecodeError.
    s = ContentStore(data_dir=str(tmp_path), node_id="n")
    s.key_manager.decrypt_manifest = MagicMock(
        return_value=b"\xff\xfe\x00not-valid-json",
    )
    with pytest.raises(ManifestError):
        s._decode_manifest(b"x", [])


@pytest.mark.asyncio
async def test_valid_artifacts_still_decode(tmp_path):
    # Regression: a valid round-trip still decodes (no false ManifestError).
    s = ContentStore(data_dir=str(tmp_path), node_id="n")
    art = await s.store_local_with_artifacts(b"valid content")
    manifest = s._decode_manifest(art.encrypted_manifest, art.key_shares)
    assert manifest is not None
    assert manifest.shard_hashes
