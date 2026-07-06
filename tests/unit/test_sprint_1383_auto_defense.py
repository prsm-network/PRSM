"""Sprint 1383 — challenge auto-defense: on a ReceiptChallenged, self-verify the retained §7 receipt
and classify BAD-FAITH (receipt verifies) vs LEGITIMATE (receipt fails) vs NO-RECEIPT. The verifier
is monkeypatched so no real receipt/crypto is needed.
"""
import prsm.settlement.challenge_auto_defense as ad
from prsm.settlement.challenge_auto_defense import build_challenge_auto_defense


class _Chal:
    def __init__(self, leaf="0x" + "ab" * 32, batch="0x" + "01" * 32):
        self.receipt_leaf_hash = leaf
        self.batch_id = batch


class _Record:
    inference_receipt = object()
    settler_public_key_b64 = "PUB="
    stage_public_keys = None


class _Store:
    def __init__(self, record):
        self._record = record
        self.asked = []

    def get(self, leaf):
        self.asked.append(leaf)
        return self._record


class _Report:
    def __init__(self, ok, findings=()):
        self.receipt_ok = ok
        self.findings = list(findings)


def _patch_verify(monkeypatch, report):
    import prsm.settlement.challenge_verifier as cv
    monkeypatch.setattr(cv, "verify_inference_receipt_for_challenge",
                        lambda *a, **k: report)


def test_bad_faith_receipt_verifies(monkeypatch):
    _patch_verify(monkeypatch, _Report(ok=True))
    store = _Store(_Record())
    verdicts = []
    defend = build_challenge_auto_defense(store, on_verdict=lambda c, r: verdicts.append(r))
    defend(_Chal())
    assert store.asked and bytes(store.asked[0]).hex() == "ab" * 32   # looked up by parsed leaf
    assert len(verdicts) == 1 and verdicts[0].receipt_ok is True       # defensible verdict surfaced


def test_legitimate_receipt_fails(monkeypatch):
    class _F:
        proven = True
        reason = "INVALID_SETTLER_SIGNATURE"
    _patch_verify(monkeypatch, _Report(ok=False, findings=[_F()]))
    verdicts = []
    build_challenge_auto_defense(_Store(_Record()),
                                 on_verdict=lambda c, r: verdicts.append(r))(_Chal())
    assert len(verdicts) == 1 and verdicts[0].receipt_ok is False


def test_no_retained_receipt(monkeypatch):
    verdicts = []
    build_challenge_auto_defense(_Store(None),
                                 on_verdict=lambda c, r: verdicts.append(r))(_Chal())
    assert verdicts == [None]                                          # verdict None → can't assess


def test_no_store_getter_returns_none():
    verdicts = []
    build_challenge_auto_defense(lambda: None,
                                 on_verdict=lambda c, r: verdicts.append(r))(_Chal())
    assert verdicts == []                                             # early return, no verdict


def test_unparseable_leaf_is_safe(monkeypatch):
    verdicts = []
    build_challenge_auto_defense(_Store(_Record()),
                                 on_verdict=lambda c, r: verdicts.append(r))(_Chal(leaf="not-hex"))
    assert verdicts == []                                            # skipped, no crash


def test_bad_verdict_callback_does_not_raise(monkeypatch):
    _patch_verify(monkeypatch, _Report(ok=True))

    def boom(_c, _r):
        raise RuntimeError("hook blew up")
    build_challenge_auto_defense(_Store(_Record()), on_verdict=boom)(_Chal())  # must not raise


# ── sp1384 — ChallengeDefenseStats ──
def test_defense_stats_buckets_by_verdict():
    from prsm.settlement.challenge_auto_defense import ChallengeDefenseStats
    s = ChallengeDefenseStats()
    s.record(None, _Report(ok=True))      # bad-faith
    s.record(None, _Report(ok=True))      # bad-faith
    s.record(None, _Report(ok=False))     # legitimate
    s.record(None, None)                  # no-receipt
    assert (s.bad_faith, s.legitimate, s.no_receipt) == (2, 1, 1)


def test_stats_plug_into_on_verdict(monkeypatch):
    from prsm.settlement.challenge_auto_defense import ChallengeDefenseStats
    _patch_verify(monkeypatch, _Report(ok=True))
    s = ChallengeDefenseStats()
    build_challenge_auto_defense(_Store(_Record()), on_verdict=s.record)(_Chal())
    assert s.bad_faith == 1                # the auto-defense verdict flowed into the tally


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
