"""Sprint 1394 — `content get --verify-provenance` distinguishes UNREGISTERED from FAILED.

Free-published content has no on-chain provenance; that isn't an authenticity FAILURE (a registered
creator that mismatches) — it's just unregistered. Conflating them made a new user distrust
legitimately-free content.
"""
from click.testing import CliRunner


def _invoke_get(res, monkeypatch):
    monkeypatch.setattr("prsm.cli._api_url_from_creds", lambda o: "http://x")
    monkeypatch.setattr("prsm.cli._run_async", lambda coro: res)
    from prsm.cli import content as _content_group
    return CliRunner().invoke(_content_group, ["get", "EV", "--verify-provenance"])


def test_unregistered_labeled_distinctly(monkeypatch):
    res = {"cid": "abc", "filename": "f.txt", "tier": "new", "size_bytes": 86,
           "integrity_verified": True, "authenticity_verified": False,
           "authenticity_detail": "content carries no provenance_hash (unregistered)",
           "provenance_hash": None}
    r = _invoke_get(res, monkeypatch)
    assert "UNREGISTERED" in r.output
    assert "FAILED" not in r.output


def test_real_creator_mismatch_still_fails(monkeypatch):
    res = {"cid": "abc", "filename": "f.txt", "tier": "new", "size_bytes": 86,
           "integrity_verified": True, "authenticity_verified": False,
           "authenticity_detail": "creator mismatch", "provenance_hash": "0xdeadbeef"}
    r = _invoke_get(res, monkeypatch)
    assert "FAILED" in r.output
    assert "UNREGISTERED" not in r.output


def test_verified_shows_verified(monkeypatch):
    res = {"cid": "abc", "filename": "f.txt", "tier": "new", "size_bytes": 86,
           "integrity_verified": True, "authenticity_verified": True,
           "provenance_hash": "0xabc"}
    r = _invoke_get(res, monkeypatch)
    assert "VERIFIED" in r.output


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
