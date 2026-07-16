"""Sprint 1458 — POST /content/paid/publish-stream: streaming large Tier B/C paid publish.

Streams the plaintext body → stream-encrypts to a ciphertext file → serves it via the large-content
path → registers the creator on-chain with the PAID-PUBLISHER key (creator == the commitment depositor,
the sp1365 anti-squat invariant) → deposits the commitment + retains the wrapped key. FAIL CLOSED
without every publisher primitive (incl. the dedicated paid-publisher provenance client).
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from prsm.enterprise.recipient_encryption import generate_recipient_keypair
from prsm.node.api import create_api_app
from prsm.node.local_content_publisher import LocalContentPublisher
from prsm.node.paid_key_serve import PaidKeyStore
from prsm.storage.paid_unlock import reconstruct_paid_content_from_file

_CAV = "0x" + "ab" * 20
_PLAINTEXT = (b"large proprietary paid dataset row; " * 100_000) + b"tail"


def _node(tmp_path, *, with_prov=True, with_key=True):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._content_filter_store = None
    node.content_uploader = None
    node._paid_key_store = PaidKeyStore()
    if with_key:
        kc = MagicMock()
        kc.deposit_key.return_value = ("0xdep", None)
        kc.get_deposit.return_value = None          # fresh hash — anti-squat guard passes
        kc.address = "0x" + "11" * 20
        node._paid_publish_key_client = kc
    else:
        node._paid_publish_key_client = None
    if with_prov:
        pc = MagicMock()
        pc.register_content.return_value = ("0xreg", None)
        node._paid_publish_provenance_client = pc
    else:
        node._paid_publish_provenance_client = None
    publisher = LocalContentPublisher(tmp_path / "stage", node_id="test-node")
    node.content_provider.content_publisher = publisher
    node.content_provider.register_local_content = MagicMock()
    node._pub = publisher
    return node


def _client(node):
    fake_ep = MagicMock()
    fake_ep.content_access_verifier = _CAV
    patcher = patch("prsm.config.networks.resolve_endpoints", return_value=fake_ep)
    patcher.start()
    return TestClient(create_api_app(node, enable_security=False),
                      raise_server_exceptions=False), patcher


def test_paid_publish_stream_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("PRSM_PAID_KEY_SERVE", "1")
    node = _node(tmp_path)
    buyer_priv, buyer_pub = generate_recipient_keypair()
    client, patcher = _client(node)
    try:
        resp = client.post("/content/paid/publish-stream", content=_PLAINTEXT,
                           params={"fee_wei": 10 ** 18, "buyer_x25519_pubkeys": buyer_pub,
                                   "filename": "ds.bin"})
    finally:
        patcher.stop()
    assert resp.status_code == 200, resp.text
    j = resp.json()
    ch = bytes.fromhex(j["content_hash"][2:])

    # ★ creator == depositor: the on-chain register_content and the deposit operate on the SAME
    # content_hash (and the node wires BOTH clients with PRSM_PAID_PUBLISHER_KEY, so the signer/creator
    # equals the depositor — the sp1365 anti-squat invariant).
    node._paid_publish_provenance_client.register_content.assert_called_once()
    assert node._paid_publish_provenance_client.register_content.call_args[0][0] == ch
    assert node._paid_publish_key_client.deposit_key.call_args[0][0] == ch
    assert node._paid_publish_key_client.deposit_key.call_args[0][1] == bytes.fromhex(j["commitment"][2:])

    # wrapped key retained for the gated serve; ciphertext file served + decryptable by the buyer.
    retained = node._paid_key_store.get(ch)
    assert retained is not None and retained["fee_wei"] == 10 ** 18
    staged = node._pub.local_publish_path(j["cid"])
    assert staged is not None
    assert _PLAINTEXT[:32] not in staged.read_bytes()   # served bytes are encrypted
    out = tmp_path / "recovered.bin"
    reconstruct_paid_content_from_file(retained["wrapped_key"], buyer_priv, staged, out)
    assert out.read_bytes() == _PLAINTEXT               # ★ full publish→retain→decrypt round trip


def test_fail_closed_without_serve_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("PRSM_PAID_KEY_SERVE", raising=False)
    node = _node(tmp_path)
    client, patcher = _client(node)
    try:
        resp = client.post("/content/paid/publish-stream", content=b"x",
                           params={"fee_wei": 1, "buyer_x25519_pubkeys": "k"})
    finally:
        patcher.stop()
    assert resp.status_code == 503


def test_fail_closed_without_provenance_client(tmp_path, monkeypatch):
    # ★ the crux: without the dedicated paid-publisher provenance client, creator==depositor can't be
    # guaranteed → fail closed rather than register a mis-credited creator.
    monkeypatch.setenv("PRSM_PAID_KEY_SERVE", "1")
    node = _node(tmp_path, with_prov=False)
    client, patcher = _client(node)
    try:
        resp = client.post("/content/paid/publish-stream", content=_PLAINTEXT,
                           params={"fee_wei": 10 ** 18, "buyer_x25519_pubkeys": "k"})
    finally:
        patcher.stop()
    assert resp.status_code == 503


def test_fail_closed_without_key_client(tmp_path, monkeypatch):
    monkeypatch.setenv("PRSM_PAID_KEY_SERVE", "1")
    node = _node(tmp_path, with_key=False)
    client, patcher = _client(node)
    try:
        resp = client.post("/content/paid/publish-stream", content=_PLAINTEXT,
                           params={"fee_wei": 10 ** 18, "buyer_x25519_pubkeys": "k"})
    finally:
        patcher.stop()
    assert resp.status_code == 503


def test_base_exception_midstream_cleans_up_temps(tmp_path, monkeypatch):
    # sp1458 hardening (audit note): a mid-publish BaseException (e.g. asyncio.CancelledError on server
    # shutdown / client disconnect) must NOT leak the plaintext + ciphertext temp files. The route's
    # cleanup runs in a `finally`, so it fires even for a BaseException the `except Exception` misses.
    import glob
    import os as _os
    import tempfile

    monkeypatch.setenv("PRSM_PAID_KEY_SERVE", "1")
    _, buyer_pub = generate_recipient_keypair()
    node = _node(tmp_path)
    pat = _os.path.join(tempfile.gettempdir(), "prsm-paidpub-*")
    before = set(glob.glob(pat))

    def _boom(*a, **k):
        raise asyncio.CancelledError("shutdown mid-publish")

    client, patcher = _client(node)
    try:
        with patch("prsm.economy.paid_content.build_paid_content_from_path", side_effect=_boom):
            try:
                client.post("/content/paid/publish-stream", content=_PLAINTEXT,
                            params={"fee_wei": 10 ** 18, "buyer_x25519_pubkeys": buyer_pub})
            except BaseException:  # noqa: BLE001 — CancelledError may surface through the test client
                pass
    finally:
        patcher.stop()

    leaked = set(glob.glob(pat)) - before
    assert not leaked, f"streaming paid-publish temps leaked on BaseException: {leaked}"


def test_empty_body_and_missing_buyers_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("PRSM_PAID_KEY_SERVE", "1")
    _, buyer_pub = generate_recipient_keypair()
    node = _node(tmp_path)
    client, patcher = _client(node)
    try:
        r_empty = client.post("/content/paid/publish-stream", content=b"",
                              params={"fee_wei": 10 ** 18, "buyer_x25519_pubkeys": buyer_pub})
        r_nobuyer = client.post("/content/paid/publish-stream", content=_PLAINTEXT,
                                params={"fee_wei": 10 ** 18, "buyer_x25519_pubkeys": ""})
        r_nofee = client.post("/content/paid/publish-stream", content=_PLAINTEXT,
                              params={"fee_wei": 0, "buyer_x25519_pubkeys": buyer_pub})
    finally:
        patcher.stop()
    assert r_empty.status_code == 422
    assert r_nobuyer.status_code == 422
    assert r_nofee.status_code == 422
