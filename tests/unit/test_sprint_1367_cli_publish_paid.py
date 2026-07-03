"""Sprint 1367 — Tier B/C publisher surface, brick 3: `prsm content publish-paid` CLI.

Wraps POST /content/paid/publish. Tested with a mocked node call (happy path prints the
content_hash/commitment + the buyer's unlock command) and required-arg validation.
"""
from __future__ import annotations

import os
import tempfile

from click.testing import CliRunner

from prsm.cli import main


class _Resp:
    status_code = 200

    def json(self):
        return {"content_hash": "0x" + "ab" * 32, "commitment": "0x" + "cd" * 32,
                "verifier": "0x" + "ef" * 20, "deposit_tx": "0xdep", "num_recipients": 1}


class _Client:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None, headers=None):
        assert url.endswith("/content/paid/publish")
        assert json["buyer_x25519_pubkeys"] == ["buyer-key"]
        assert json["fee_wei"] == 10 ** 18
        return _Resp()


def _tmpfile():
    fh = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    fh.write(b"proprietary dataset bytes")
    fh.close()
    return fh.name


def test_cli_publish_paid_happy(monkeypatch):
    monkeypatch.setattr("httpx.Client", _Client)
    path = _tmpfile()
    try:
        r = CliRunner().invoke(main, [
            "content", "publish-paid", path, "--buyer-pubkey", "buyer-key", "--fee", "1",
            "--api-url", "http://localhost:8000"])
    finally:
        os.unlink(path)
    assert r.exit_code == 0, r.output
    assert "Published paid dataset" in r.output
    assert "0x" + "ab" * 32 in r.output                 # content_hash
    assert "content unlock" in r.output                 # the buyer-facing next step


def test_cli_publish_paid_requires_buyer_pubkey():
    path = _tmpfile()
    try:
        r = CliRunner().invoke(main, ["content", "publish-paid", path, "--fee", "1"])
    finally:
        os.unlink(path)
    assert r.exit_code != 0
    assert "buyer-pubkey" in r.output.lower() or "missing" in r.output.lower()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
