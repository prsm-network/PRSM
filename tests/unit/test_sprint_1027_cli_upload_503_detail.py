"""Sprint 1027 — `prsm storage upload` surfaces the server's real 503 detail (Tier-1 bench gap 4).

Live bench finding: the 503 branch printed a fixed "Content uploader not
initialized. Run prsm node start" for ANY 503, hiding the server's actual `detail`.
That cost a real debugging detour — the true cause was "ContentPublisher (BitTorrent
layer) not wired — libtorrent not installed", which the generic message buried.
"""
from __future__ import annotations

import pytest

from prsm.cli import _server_detail_or

_RAISE = object()


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        if self._payload is _RAISE:
            raise ValueError("not json")
        return self._payload


def test_surfaces_server_detail():
    msg = _server_detail_or(
        _Resp({"detail": "ContentPublisher (BitTorrent layer) not wired — "
                         "Most common cause: libtorrent not installed."}),
        "generic fallback",
    )
    assert "libtorrent" in msg
    assert "ContentPublisher" in msg
    assert "generic fallback" not in msg


def test_falls_back_when_no_detail():
    assert _server_detail_or(_Resp({}), "FB") == "FB"
    assert _server_detail_or(_Resp({"other": 1}), "FB") == "FB"


def test_falls_back_when_detail_blank_or_nonstring():
    assert _server_detail_or(_Resp({"detail": "   "}), "FB") == "FB"
    assert _server_detail_or(_Resp({"detail": 123}), "FB") == "FB"


def test_falls_back_when_not_json():
    assert _server_detail_or(_Resp(_RAISE), "FB") == "FB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
