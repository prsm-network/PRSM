"""Sprint 1362 — F1 redesign R5 HIGH fix: the retained-key store must survive a node restart.

The on-chain payment gate persists forever, so an in-memory-only key store stranded every buyer who
paid after a restart (404, no refund). PaidKeyStore is now durable when given a path.
"""
from __future__ import annotations

from prsm.node.paid_key_serve import PaidKeyStore

_CH = bytes.fromhex("cd" * 32)
_FEE = 10 ** 18


def test_persists_across_restart(tmp_path):
    path = str(tmp_path / "paid_keys.json")
    s1 = PaidKeyStore(path)
    s1.put(_CH, b"THE-WRAPPED-KEY", _FEE)
    # simulate a restart — a fresh store from the same path rehydrates the retained key
    s2 = PaidKeyStore(path)
    assert s2.get(_CH) == {"wrapped_key": b"THE-WRAPPED-KEY", "fee_wei": _FEE}
    assert len(s2) == 1


def test_in_memory_when_no_path():
    s = PaidKeyStore()
    s.put(_CH, b"WK", _FEE)
    assert s.get(_CH)["wrapped_key"] == b"WK"
    # a second in-memory store shares nothing (no persistence)
    assert PaidKeyStore().get(_CH) is None


def test_corrupt_file_does_not_crash_startup(tmp_path):
    path = str(tmp_path / "bad.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("}{ not json at all")
    s = PaidKeyStore(path)          # must not raise — a corrupt file just yields an empty store
    assert len(s) == 0


def test_multiple_entries_round_trip(tmp_path):
    path = str(tmp_path / "multi.json")
    s1 = PaidKeyStore(path)
    ch2 = bytes.fromhex("ab" * 32)
    s1.put(_CH, b"key-one", _FEE)
    s1.put(ch2, b"key-two", 5)
    s2 = PaidKeyStore(path)
    assert s2.get(_CH)["wrapped_key"] == b"key-one"
    assert s2.get(ch2) == {"wrapped_key": b"key-two", "fee_wei": 5}


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
