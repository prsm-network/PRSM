"""Sprint 304 — recipient-encrypted upload HTTP + MCP surface.

POST /content/upload  — extended with optional
                        `recipients` field. When present,
                        text is encrypted client-style
                        before sharding.
GET  /content/recipient-manifest/{cid} — returns the
                        per-recipient sealed-key manifest
                        for an encrypted content blob.

prsm_enterprise_recipient MCP tool — keypair_gen | encrypt
                        | decrypt | get_manifest.

The encryption layer is orthogonal to the existing FTNS
retrieval payment gate: encryption controls *can decrypt*;
FTNS controls *can fetch the ciphertext*. Both must succeed
for an outsider to access plaintext, and even all the FTNS
in the world won't bypass the encryption.
"""
from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.enterprise.recipient_encryption import (
    EncryptedPayload,
    EnterpriseRecipient,
    decrypt_for_recipient,
    encrypt_for_recipients,
    generate_recipient_keypair,
)
from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_enterprise_recipient,
)
from prsm.node.api import create_api_app


class _FakeUploadResult:
    """Mirrors the real ``UploadedContent`` dataclass field
    names at ``prsm/node/content_uploader.py:478`` — in
    particular ``content_id`` (NOT ``cid``). Sprint 425
    surfaced a production bug at ``api.py:5861`` that
    referenced ``result.cid`` and would have been caught by
    these tests had the fake been shape-correct."""

    def __init__(self, cid, filename, size_bytes):
        # Endpoint advertises the field as "cid" in JSON but
        # the dataclass attribute is ``content_id``. Mirror
        # the real shape so endpoint regressions surface.
        self.content_id = cid
        self.filename = filename
        self.size_bytes = size_bytes
        self.content_hash = None
        self.creator_id = "test-node"
        self.royalty_rate = 0.01
        self.parent_cids = []


class _FakeContentUploader:
    """Captures upload_text calls so we can assert the
    integration encrypts before handing off to the
    sharding layer."""

    def __init__(self):
        self.content_publisher = object()  # truthy
        self.last_text = None
        self.last_filename = None

    async def upload_text(
        self, *, text, filename, replicas,
        royalty_rate, parent_cids, creator_eth_address,
        metadata=None,  # sp1340 — real signature carries topic-search metadata
    ):
        self.last_text = text
        self.last_filename = filename
        return _FakeUploadResult(
            cid="Qm" + "a" * 44,
            filename=filename,
            size_bytes=len(text.encode("utf-8")),
        )


class _FakeContentProvider:
    """Stub provider for retrieve + manifest endpoints."""

    def __init__(self, blobs=None):
        self._blobs = blobs or {}

    def get_stats(self):
        return {}

    async def request_content(
        self, *, cid, timeout, verify_hash,
    ):
        if cid not in self._blobs:
            raise FileNotFoundError(cid)
        return self._blobs[cid]


def _client(uploader=None, provider=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node.content_uploader = uploader
    node.content_provider = provider
    node._content_filter_store = None
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


# ── Backwards-compat: plain upload still works ───────


def test_plain_upload_without_recipients_still_works():
    uploader = _FakeContentUploader()
    resp = _client(uploader=uploader).post(
        "/content/upload",
        json={
            "text": "hello world",
            "filename": "note.txt",
        },
    )
    assert resp.status_code == 200
    # The original text passed through unchanged
    assert uploader.last_text == "hello world"
    body = resp.json()
    assert body.get("encrypted") in (None, False)
    # Sprint 425 regression pin: the endpoint MUST source
    # the "cid" response value from
    # ``UploadedContent.content_id`` — not a phantom
    # ``.cid`` attribute. Pre-fix this raised
    # AttributeError → 500 in production.
    assert body["cid"] == "Qm" + "a" * 44
    assert body["filename"] == "note.txt"


# ── Encrypted upload ────────────────────────────────


def test_encrypted_upload_replaces_text_with_payload():
    uploader = _FakeContentUploader()
    priv, pub = generate_recipient_keypair()
    resp = _client(uploader=uploader).post(
        "/content/upload",
        json={
            "text": "proprietary research data",
            "filename": "trial.csv",
            "recipients": [
                {
                    "identifier": "alice@corp.com",
                    "x25519_pubkey_b64": pub,
                },
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["encrypted"] is True

    # The text the sharding layer saw is the encrypted JSON
    # bundle, not the original.
    assert uploader.last_text != "proprietary research data"
    assert uploader.last_filename.endswith(".enc.json")

    # Round-trip — the embedded payload must decrypt
    payload_dict = json.loads(uploader.last_text)
    payload = EncryptedPayload.from_dict(payload_dict)
    assert decrypt_for_recipient(payload, priv) == (
        b"proprietary research data"
    )


def test_encrypted_upload_multi_recipient():
    uploader = _FakeContentUploader()
    priv_a, pub_a = generate_recipient_keypair()
    priv_b, pub_b = generate_recipient_keypair()
    resp = _client(uploader=uploader).post(
        "/content/upload",
        json={
            "text": "shared sensitive dataset",
            "filename": "x.txt",
            "recipients": [
                {"identifier": "alice", "x25519_pubkey_b64": pub_a},
                {"identifier": "bob", "x25519_pubkey_b64": pub_b},
            ],
        },
    )
    assert resp.status_code == 200
    payload = EncryptedPayload.from_dict(
        json.loads(uploader.last_text),
    )
    assert decrypt_for_recipient(payload, priv_a) == (
        b"shared sensitive dataset"
    )
    assert decrypt_for_recipient(payload, priv_b) == (
        b"shared sensitive dataset"
    )


def test_encrypted_upload_422_malformed_recipient():
    uploader = _FakeContentUploader()
    resp = _client(uploader=uploader).post(
        "/content/upload",
        json={
            "text": "x",
            "filename": "y.txt",
            "recipients": [
                {
                    "identifier": "alice",
                    "x25519_pubkey_b64": "not-base64!",
                },
            ],
        },
    )
    assert resp.status_code == 422


def test_encrypted_upload_422_empty_recipients_list():
    """If `recipients` is given as an empty list, that's
    operator confusion — fail loud, don't silently fall
    through to plaintext upload."""
    uploader = _FakeContentUploader()
    resp = _client(uploader=uploader).post(
        "/content/upload",
        json={
            "text": "x",
            "filename": "y.txt",
            "recipients": [],
        },
    )
    assert resp.status_code == 422


# ── /content/recipient-manifest/{cid} ──────────────


def test_manifest_endpoint_503_when_unwired():
    resp = _client(provider=None).get(
        "/content/recipient-manifest/Qmabc",
    )
    assert resp.status_code == 503


def test_manifest_endpoint_404_unknown_cid():
    provider = _FakeContentProvider(blobs={})
    resp = _client(provider=provider).get(
        "/content/recipient-manifest/Qmnone",
    )
    assert resp.status_code == 404


def test_manifest_endpoint_422_not_encrypted():
    """CID exists but content isn't an encrypted bundle —
    fail loud, don't return a misleading empty manifest."""
    provider = _FakeContentProvider(blobs={
        "Qmplain": b"just plain text",
    })
    resp = _client(provider=provider).get(
        "/content/recipient-manifest/Qmplain",
    )
    assert resp.status_code == 422


def test_manifest_endpoint_happy_path():
    priv, pub = generate_recipient_keypair()
    payload = encrypt_for_recipients(
        b"secret",
        [EnterpriseRecipient(
            identifier="alice",
            x25519_pubkey_b64=pub,
        )],
    )
    blob = json.dumps(payload.to_dict()).encode("utf-8")
    provider = _FakeContentProvider(blobs={"Qmenc": blob})

    body = _client(provider=provider).get(
        "/content/recipient-manifest/Qmenc",
    ).json()
    assert body["version"] == "v1"
    assert len(body["entries"]) == 1
    assert body["entries"][0]["identifier"] == "alice"
    # Sealed key blob is present so a recipient can attempt
    # decryption without re-fetching the full ciphertext.
    assert body["entries"][0]["sealed_symmetric_key_b64"]


# ── MCP tool ────────────────────────────────────────


def test_mcp_tool_registered():
    assert "prsm_enterprise_recipient" in TOOL_HANDLERS


@pytest.mark.asyncio
async def test_mcp_missing_action():
    r = await handle_prsm_enterprise_recipient({})
    assert "action" in r.lower()


@pytest.mark.asyncio
async def test_mcp_unknown_action():
    r = await handle_prsm_enterprise_recipient(
        {"action": "explode"},
    )
    assert "must be" in r.lower()


@pytest.mark.asyncio
async def test_mcp_keypair_gen_runs_offline():
    """keypair_gen must NOT call the node — it's a pure
    client-side primitive."""
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(),
    ) as mock_call:
        r = await handle_prsm_enterprise_recipient({
            "action": "keypair_gen",
        })
    assert mock_call.await_count == 0
    # Output contains both keys
    assert "pubkey" in r.lower()
    assert "privkey" in r.lower()


@pytest.mark.asyncio
async def test_mcp_encrypt_runs_offline_and_round_trips():
    priv, pub = generate_recipient_keypair()
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(),
    ) as mock_call:
        r = await handle_prsm_enterprise_recipient({
            "action": "encrypt",
            "plaintext_b64": base64.b64encode(
                b"secret data",
            ).decode(),
            "recipients": [
                {
                    "identifier": "alice",
                    "x25519_pubkey_b64": pub,
                },
            ],
        })
    assert mock_call.await_count == 0
    # The render contains a JSON bundle the user can paste
    # into a follow-up decrypt call
    payload_blob = r[r.find("{"): r.rfind("}") + 1]
    payload = EncryptedPayload.from_dict(
        json.loads(payload_blob),
    )
    assert decrypt_for_recipient(payload, priv) == (
        b"secret data"
    )


@pytest.mark.asyncio
async def test_mcp_decrypt_runs_offline_and_round_trips():
    priv, pub = generate_recipient_keypair()
    payload = encrypt_for_recipients(
        b"target plaintext",
        [EnterpriseRecipient("alice", pub)],
    )
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(),
    ):
        r = await handle_prsm_enterprise_recipient({
            "action": "decrypt",
            "privkey_b64": priv,
            "payload": payload.to_dict(),
        })
    assert "target plaintext" in r


@pytest.mark.asyncio
async def test_mcp_get_manifest_calls_node():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "version": "v1",
            "entries": [{
                "identifier": "alice",
                "ephemeral_pubkey_b64": "AAAA",
                "nonce_b64": "BBBB",
                "sealed_symmetric_key_b64": "CCCC",
            }],
        }),
    ) as mock_call:
        r = await handle_prsm_enterprise_recipient({
            "action": "get_manifest",
            "cid": "Qmabc",
        })
    args = mock_call.await_args[0]
    assert args[1] == "/content/recipient-manifest/Qmabc"
    assert "alice" in r


@pytest.mark.asyncio
async def test_mcp_get_manifest_requires_cid():
    r = await handle_prsm_enterprise_recipient({
        "action": "get_manifest",
    })
    assert "cid" in r.lower()
