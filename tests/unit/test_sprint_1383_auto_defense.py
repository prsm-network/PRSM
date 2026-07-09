"""Sprint 1383 — challenge auto-defense: on a ReceiptChallenged, self-verify the retained §7 receipt
and surface an actionable verdict. The verifier is monkeypatched so no real receipt/crypto is needed.

sp1411 UPDATED THIS SUITE. The verdict no longer comes from ``receipt_ok`` — it comes from the
challenge's on-chain reason code, and the §7 self-verify is EVIDENCE attached to the log line. See
``test_sprint_1411_auto_defense_reason_code.py`` for why (a DOUBLE_SPEND receipt verifies, yet the
registry slashes for it). What this suite still pins:

  - the retained receipt is looked up by the PARSED leaf,
  - the ChallengeReport is surfaced to ``on_verdict`` for programmatic consumers,
  - the callback never raises, whatever the store / leaf / verifier / hook does,
  - and a verdict is ALWAYS recorded — including when we cannot self-assess — so the operator's
    alert fires regardless (sp1411; these paths previously returned silently).
"""
import prsm.settlement.challenge_verifier as cv
from prsm.settlement.challenge_auto_defense import (
    REASON_DOUBLE_SPEND,
    REASON_EXPIRED,
    REASON_INVALID_SIGNATURE,
    REASON_NO_ESCROW,
    ChallengeDefenseStats,
    build_challenge_auto_defense,
)


class _Chal:
    def __init__(self, leaf="0x" + "ab" * 32, batch="0x" + "01" * 32,
                 reason_code=REASON_INVALID_SIGNATURE):
        self.receipt_leaf_hash = leaf
        self.batch_id = batch
        self.reason_code = reason_code


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
    monkeypatch.setattr(cv, "verify_inference_receipt_for_challenge",
                        lambda *a, **k: report)


def test_verifying_receipt_is_looked_up_and_surfaced(monkeypatch):
    _patch_verify(monkeypatch, _Report(ok=True))
    store = _Store(_Record())
    verdicts = []
    defend = build_challenge_auto_defense(store, on_verdict=lambda c, r: verdicts.append(r))
    defend(_Chal())
    assert store.asked and bytes(store.asked[0]).hex() == "ab" * 32   # looked up by parsed leaf
    assert len(verdicts) == 1 and verdicts[0].receipt_ok is True       # report surfaced as evidence


def test_failing_receipt_is_surfaced(monkeypatch):
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
    assert verdicts == [None]                                          # report None → can't assess


def test_no_store_getter_still_records_a_verdict():
    """sp1411 — previously an early return with NO verdict, so a slashing challenge landing before
    the receipt store was wired never reached the operator's alert."""
    verdicts = []
    build_challenge_auto_defense(lambda: None,
                                 on_verdict=lambda c, r: verdicts.append(r))(_Chal())
    assert verdicts == [None]


def test_unparseable_leaf_is_safe_and_still_records(monkeypatch):
    """A malformed leaf means we cannot self-assess. It does not mean the challenge isn't real."""
    verdicts = []
    build_challenge_auto_defense(_Store(_Record()),
                                 on_verdict=lambda c, r: verdicts.append(r))(_Chal(leaf="not-hex"))
    assert verdicts == [None]


def test_bad_verdict_callback_does_not_raise(monkeypatch):
    _patch_verify(monkeypatch, _Report(ok=True))

    def boom(_c, _r):
        raise RuntimeError("hook blew up")
    build_challenge_auto_defense(_Store(_Record()), on_verdict=boom)(_Chal())  # must not raise


# ── sp1384 — ChallengeDefenseStats (sp1411: bucketed by the on-chain reason) ──
def test_defense_stats_buckets_by_reason():
    s = ChallengeDefenseStats()
    s.record(_Chal(reason_code=REASON_DOUBLE_SPEND), _Report(ok=True))   # slashing — and it verifies
    s.record(_Chal(reason_code=REASON_INVALID_SIGNATURE), _Report(ok=False))
    s.record(_Chal(reason_code=REASON_NO_ESCROW), _Report(ok=True))      # griefing
    s.record(_Chal(reason_code=REASON_EXPIRED), _Report(ok=True))
    s.record(_Chal(reason_code=REASON_DOUBLE_SPEND), None)               # + a coverage miss
    assert (s.bad_faith, s.legitimate, s.expired, s.no_receipt) == (1, 3, 1, 1)


def test_stats_plug_into_on_verdict(monkeypatch):
    _patch_verify(monkeypatch, _Report(ok=True))
    s = ChallengeDefenseStats()
    build_challenge_auto_defense(_Store(_Record()), on_verdict=s.record)(
        _Chal(reason_code=REASON_DOUBLE_SPEND))
    assert s.legitimate == 1        # the slash alarm fires, even though the receipt verifies
    assert s.bad_faith == 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
