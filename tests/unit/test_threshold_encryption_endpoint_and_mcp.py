"""Sprint 307 — threshold encryption HTTP + MCP integration.

The /content/upload endpoint already accepts the `recipients`
field for OR-decrypt (sprint 304). This sprint extends the
request schema with an optional `threshold` field — when
present, the upload is encrypted in t-of-n mode instead.

prsm_enterprise_recipient MCP tool gains three actions:
  encrypt_threshold | unseal_share | combine_decrypt
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
    ShareContribution,
    ThresholdParams,
    combine_shares_and_decrypt,
    encrypt_for_threshold,
    generate_recipient_keypair,
    unseal_share_for_recipient,
)
from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_enterprise_recipient,
)
from prsm.node.api import create_api_app


class _FakeUploadResult:
    """Mirrors the real ``UploadedContent`` dataclass shape
    at ``prsm/node/content_uploader.py:478`` — including the
    ``content_id`` field name (NOT ``cid``). Sprint 425
    fixture-drift lesson."""

    def __init__(self, cid, filename, size_bytes):
        self.content_id = cid
        self.filename = filename
        self.size_bytes = size_bytes
        self.content_hash = None
        self.creator_id = "test-node"
        self.royalty_rate = 0.01
        self.parent_cids = []


class _FakeContentUploader:
    def __init__(self):
        self.content_publisher = object()
        self.last_text = None
        self.last_filename = None

    async def upload_text(
        self, *, text, filename, replicas,
        royalty_rate, parent_cids, creator_eth_address,
        metadata=None,  # sp1340 — descriptive topic-search metadata (real signature has it)
    ):
        self.last_text = text
        self.last_filename = filename
        self.last_metadata = metadata
        return _FakeUploadResult(
            cid="Qm" + "a" * 44,
            filename=filename,
            size_bytes=len(text.encode("utf-8")),
        )


def _client(uploader=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node.content_uploader = uploader
    node.content_provider = None
    node._content_filter_store = None
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def _gen_recipients(n):
    recipients = []
    privkeys = []
    for i in range(n):
        priv, pub = generate_recipient_keypair()
        privkeys.append(priv)
        recipients.append({
            "identifier": f"r-{i}",
            "x25519_pubkey_b64": pub,
        })
    return recipients, privkeys


# ── /content/upload threshold mode ───────────────────


def test_upload_threshold_produces_threshold_payload():
    uploader = _FakeContentUploader()
    recipients, _ = _gen_recipients(5)
    resp = _client(uploader=uploader).post(
        "/content/upload",
        json={
            "text": "high-stakes data",
            "filename": "x.txt",
            "recipients": recipients,
            "threshold": 3,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["encrypted"] is True

    payload = EncryptedPayload.from_dict(
        json.loads(uploader.last_text),
    )
    assert payload.manifest.threshold == (
        ThresholdParams(t=3, n=5)
    )
    # Every entry has a distinct share_index
    indices = {
        e.share_index for e in payload.manifest.entries
    }
    assert indices == {1, 2, 3, 4, 5}


def test_upload_threshold_round_trips_via_t_shares():
    uploader = _FakeContentUploader()
    recipients, privkeys = _gen_recipients(5)
    plaintext = "high-stakes round trip data" * 4
    resp = _client(uploader=uploader).post(
        "/content/upload",
        json={
            "text": plaintext,
            "filename": "x.txt",
            "recipients": recipients,
            "threshold": 3,
        },
    )
    assert resp.status_code == 200

    payload = EncryptedPayload.from_dict(
        json.loads(uploader.last_text),
    )
    contributions = [
        unseal_share_for_recipient(payload, privkeys[i])
        for i in (1, 3, 4)
    ]
    out = combine_shares_and_decrypt(
        payload, contributions,
    )
    assert out.decode("utf-8") == plaintext


def test_upload_threshold_over_n_422():
    uploader = _FakeContentUploader()
    recipients, _ = _gen_recipients(2)
    resp = _client(uploader=uploader).post(
        "/content/upload",
        json={
            "text": "x",
            "filename": "x.txt",
            "recipients": recipients,
            "threshold": 5,  # > n=2
        },
    )
    assert resp.status_code == 422


def test_upload_threshold_zero_422():
    uploader = _FakeContentUploader()
    recipients, _ = _gen_recipients(3)
    resp = _client(uploader=uploader).post(
        "/content/upload",
        json={
            "text": "x",
            "filename": "x.txt",
            "recipients": recipients,
            "threshold": 0,
        },
    )
    assert resp.status_code == 422


def test_upload_threshold_without_recipients_422():
    """`threshold` requires `recipients` — bare threshold
    on a plaintext upload is operator confusion."""
    uploader = _FakeContentUploader()
    resp = _client(uploader=uploader).post(
        "/content/upload",
        json={
            "text": "x",
            "filename": "x.txt",
            "threshold": 2,
        },
    )
    assert resp.status_code == 422


def test_upload_or_decrypt_path_unchanged():
    """No `threshold` field → OR-decrypt mode (sprint 304),
    unchanged behavior."""
    uploader = _FakeContentUploader()
    recipients, _ = _gen_recipients(2)
    resp = _client(uploader=uploader).post(
        "/content/upload",
        json={
            "text": "or-decrypt path",
            "filename": "x.txt",
            "recipients": recipients,
        },
    )
    assert resp.status_code == 200
    payload = EncryptedPayload.from_dict(
        json.loads(uploader.last_text),
    )
    assert payload.manifest.threshold is None


# ── MCP threshold actions ────────────────────────────


@pytest.mark.asyncio
async def test_mcp_encrypt_threshold_offline():
    priv, pub = generate_recipient_keypair()
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(),
    ) as mock_call:
        r = await handle_prsm_enterprise_recipient({
            "action": "encrypt_threshold",
            "plaintext_b64": base64.b64encode(
                b"hi",
            ).decode(),
            "recipients": [
                {"identifier": "alice",
                 "x25519_pubkey_b64": pub},
                {"identifier": "bob",
                 "x25519_pubkey_b64":
                     generate_recipient_keypair()[1]},
                {"identifier": "carol",
                 "x25519_pubkey_b64":
                     generate_recipient_keypair()[1]},
            ],
            "threshold": 2,
        })
    assert mock_call.await_count == 0  # fully offline
    payload_blob = r[r.find("{"): r.rfind("}") + 1]
    payload = EncryptedPayload.from_dict(
        json.loads(payload_blob),
    )
    assert payload.manifest.threshold == (
        ThresholdParams(t=2, n=3)
    )


@pytest.mark.asyncio
async def test_mcp_unseal_share_offline():
    priv, pub = generate_recipient_keypair()
    other_pub = generate_recipient_keypair()[1]
    payload = encrypt_for_threshold(
        b"x",
        [
            EnterpriseRecipient("alice", pub),
            EnterpriseRecipient("bob", other_pub),
        ],
        threshold=2,
    )
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(),
    ) as mock_call:
        r = await handle_prsm_enterprise_recipient({
            "action": "unseal_share",
            "privkey_b64": priv,
            "payload": payload.to_dict(),
        })
    assert mock_call.await_count == 0
    assert "share_index" in r.lower()
    assert "y_values_b64" in r.lower()


@pytest.mark.asyncio
async def test_mcp_combine_decrypt_round_trip():
    recipients = []
    privkeys = []
    for i in range(3):
        p, pub = generate_recipient_keypair()
        privkeys.append(p)
        recipients.append(
            EnterpriseRecipient(f"r-{i}", pub),
        )
    plaintext = b"combine roundtrip"
    payload = encrypt_for_threshold(
        plaintext, recipients, threshold=2,
    )
    contribs = [
        unseal_share_for_recipient(payload, privkeys[i])
        for i in (0, 1)
    ]
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(),
    ):
        r = await handle_prsm_enterprise_recipient({
            "action": "combine_decrypt",
            "payload": payload.to_dict(),
            "contributions": [
                {
                    "share_index": c.share_index,
                    "share_y_values_b64": base64.b64encode(
                        c.share_y_values,
                    ).decode(),
                }
                for c in contribs
            ],
        })
    assert "combine roundtrip" in r


@pytest.mark.asyncio
async def test_mcp_encrypt_threshold_requires_threshold():
    r = await handle_prsm_enterprise_recipient({
        "action": "encrypt_threshold",
        "plaintext_b64": base64.b64encode(b"x").decode(),
        "recipients": [],
    })
    assert (
        "threshold" in r.lower()
        or "recipient" in r.lower()
    )


@pytest.mark.asyncio
async def test_mcp_unseal_share_requires_privkey():
    r = await handle_prsm_enterprise_recipient({
        "action": "unseal_share",
        "payload": {},
    })
    assert "privkey" in r.lower()
