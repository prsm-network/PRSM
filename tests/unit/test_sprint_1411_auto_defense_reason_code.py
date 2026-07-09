"""Sprint 1411 — auto-defense must branch on the on-chain reason code, not on receipt_ok.

sp1383 classified a challenge purely on ``ChallengeReport.receipt_ok``. Two facts make that unsafe:

  1. ``BatchSettlementRegistry.challengeReceipt`` does ``if (!proven) revert ChallengeNotProven``
     BEFORE ``emit ReceiptChallenged``. Every challenge reaching this callback was already proven
     on-chain, and the registry has already invalidated the leaf's value.
  2. The §7 verifier only proves settler-signature and activation-chain grounds. It says nothing
     about DOUBLE_SPEND / NO_ESCROW / EXPIRED / CONSENSUS_MISMATCH. A DOUBLE_SPEND is a *validly
     signed* receipt replayed into two batches — so it verifies, and ``receipt_ok`` is True.

Result: DOUBLE_SPEND and CONSENSUS_MISMATCH — two of the three reasons the registry SLASHES on —
were logged "appears BAD-FAITH … no stake should be lost" and bucketed ``bad_faith``. The documented
alert target ``prsm_challenge_defense_legitimate_total`` stayed at 0 while the provider's bond
burned. An operator paging on it (as the sp1384 docstring instructs) was actively reassured.
"""
from __future__ import annotations

import pytest

import prsm.settlement.challenge_verifier as cv
from prsm.settlement.challenge_auto_defense import (
    REASON_CONSENSUS_MISMATCH,
    REASON_DOUBLE_SPEND,
    REASON_EXPIRED,
    REASON_INVALID_SIGNATURE,
    REASON_NO_ESCROW,
    ChallengeDefenseStats,
    DefenseVerdict,
    build_challenge_auto_defense,
    classify_challenge,
)


class _Chal:
    def __init__(self, reason_code, leaf="0x" + "ab" * 32, batch="0x" + "01" * 32, reason=None):
        self.reason_code = reason_code
        self.receipt_leaf_hash = leaf
        self.batch_id = batch
        if reason is not None:
            self.reason = reason


class _Record:
    inference_receipt = object()
    settler_public_key_b64 = "PUB="
    stage_public_keys = None


class _Store:
    def __init__(self, record):
        self._record = record

    def get(self, _leaf):
        return self._record


class _Report:
    def __init__(self, ok, findings=()):
        self.receipt_ok = ok
        self.findings = list(findings)


def _patch_verify(monkeypatch, report):
    monkeypatch.setattr(cv, "verify_inference_receipt_for_challenge", lambda *a, **k: report)


# ── classification ───────────────────────────────────────────────────────


@pytest.mark.parametrize("code", [REASON_DOUBLE_SPEND, REASON_INVALID_SIGNATURE,
                                  REASON_CONSENSUS_MISMATCH])
def test_the_three_slashing_reasons_classify_as_slashing(code):
    """Exactly the reasons BatchSettlementRegistry slashes the bond on."""
    assert classify_challenge(_Chal(code)) is DefenseVerdict.SLASHING


def test_no_escrow_is_griefable_not_slashing():
    """NO_ESCROW is the REQUESTER'S ATTESTATION, not a proof — the contract calls out the griefing
    risk and does not slash on it. Value is still invalidated."""
    assert classify_challenge(_Chal(REASON_NO_ESCROW)) is DefenseVerdict.GRIEFABLE


def test_expired_is_hygiene_not_slashing():
    assert classify_challenge(_Chal(REASON_EXPIRED)) is DefenseVerdict.EXPIRED


@pytest.mark.parametrize("code", [4, 99, None, "1", True, -1])
def test_unrecognised_reason_fails_safe_to_unknown(code):
    """Never reassure on a code we don't understand (incl. bool True, which == 1 in Python)."""
    assert classify_challenge(_Chal(code)) is DefenseVerdict.UNKNOWN


def test_missing_reason_code_attribute_fails_safe():
    assert classify_challenge(object()) is DefenseVerdict.UNKNOWN


# ── THE BUG: a validly-signed replayed receipt must still raise the slash alarm ──


def test_double_spend_with_a_verifying_receipt_alerts_as_slashing(monkeypatch):
    """A DOUBLE_SPEND receipt VERIFIES (it was validly signed, just replayed). The old code saw
    receipt_ok=True and reported bad-faith with no alert. Stake was being slashed."""
    _patch_verify(monkeypatch, _Report(ok=True))
    stats = ChallengeDefenseStats()
    build_challenge_auto_defense(_Store(_Record()), on_verdict=stats.record)(
        _Chal(REASON_DOUBLE_SPEND))

    assert stats.legitimate == 1, "the operator's slash alarm did not fire on a DOUBLE_SPEND"
    assert stats.bad_faith == 0, "a slashing challenge must never be logged as bad faith"


def test_consensus_mismatch_with_a_verifying_receipt_alerts_as_slashing(monkeypatch):
    _patch_verify(monkeypatch, _Report(ok=True))
    stats = ChallengeDefenseStats()
    build_challenge_auto_defense(_Store(_Record()), on_verdict=stats.record)(
        _Chal(REASON_CONSENSUS_MISMATCH))

    assert stats.legitimate == 1
    assert stats.bad_faith == 0


def test_slashing_verdict_never_says_no_stake_should_be_lost(monkeypatch, caplog):
    """The exact reassurance that made this dangerous."""
    _patch_verify(monkeypatch, _Report(ok=True))
    with caplog.at_level("ERROR"):
        build_challenge_auto_defense(_Store(_Record()))(_Chal(REASON_DOUBLE_SPEND))

    assert "no stake should be lost" not in caplog.text.lower()
    assert "stake is at risk" in caplog.text.lower()


def test_invalid_signature_with_failing_receipt_still_alerts(monkeypatch):
    """The one case the old code got right must keep working."""
    class _F:
        proven = True
        reason = "INVALID_SETTLER_SIGNATURE"
    _patch_verify(monkeypatch, _Report(ok=False, findings=[_F()]))
    stats = ChallengeDefenseStats()
    build_challenge_auto_defense(_Store(_Record()), on_verdict=stats.record)(
        _Chal(REASON_INVALID_SIGNATURE))
    assert stats.legitimate == 1


# ── the alarm fires even when we cannot self-assess ──────────────────────


def test_slashing_alarm_fires_with_no_retained_receipt():
    """A slashing challenge on a batch whose receipt we evicted must still page the operator.
    The old code bucketed this as no_receipt only — silent."""
    stats = ChallengeDefenseStats()
    build_challenge_auto_defense(_Store(None), on_verdict=stats.record)(_Chal(REASON_DOUBLE_SPEND))

    assert stats.legitimate == 1
    assert stats.no_receipt == 1      # coverage counter, orthogonal to the verdict


def test_slashing_alarm_fires_with_no_receipt_store_wired():
    stats = ChallengeDefenseStats()
    build_challenge_auto_defense(lambda: None, on_verdict=stats.record)(_Chal(REASON_DOUBLE_SPEND))
    assert (stats.legitimate, stats.no_receipt) == (1, 1)


def test_slashing_alarm_fires_on_an_unparseable_leaf():
    stats = ChallengeDefenseStats()
    build_challenge_auto_defense(_Store(_Record()), on_verdict=stats.record)(
        _Chal(REASON_DOUBLE_SPEND, leaf="not-hex"))
    assert stats.legitimate == 1


def test_self_verify_error_does_not_suppress_the_alarm(monkeypatch):
    def _boom(*_a, **_k):
        raise RuntimeError("crypto exploded")
    monkeypatch.setattr(cv, "verify_inference_receipt_for_challenge", _boom)
    stats = ChallengeDefenseStats()
    build_challenge_auto_defense(_Store(_Record()), on_verdict=stats.record)(
        _Chal(REASON_CONSENSUS_MISMATCH))
    assert stats.legitimate == 1


# ── stats buckets ────────────────────────────────────────────────────────


def test_stats_bucket_by_reason_not_by_receipt_ok():
    s = ChallengeDefenseStats()
    s.record(_Chal(REASON_DOUBLE_SPEND), _Report(ok=True))        # verifies, yet slashing
    s.record(_Chal(REASON_CONSENSUS_MISMATCH), _Report(ok=True))  # verifies, yet slashing
    s.record(_Chal(REASON_INVALID_SIGNATURE), _Report(ok=False))
    s.record(_Chal(REASON_NO_ESCROW), _Report(ok=True))           # griefing
    s.record(_Chal(REASON_EXPIRED), _Report(ok=True))
    s.record(_Chal(4), _Report(ok=True))                          # unknown → fail-safe

    assert s.legitimate == 4    # 3 slashing + 1 unknown
    assert s.bad_faith == 1     # NO_ESCROW only
    assert s.expired == 1
    assert s.no_receipt == 0


def test_no_receipt_is_a_coverage_counter_not_a_verdict():
    s = ChallengeDefenseStats()
    s.record(_Chal(REASON_DOUBLE_SPEND), None)
    assert (s.legitimate, s.no_receipt) == (1, 1)


def test_bad_hook_still_does_not_raise(monkeypatch):
    _patch_verify(monkeypatch, _Report(ok=True))

    def boom(_c, _r):
        raise RuntimeError("hook blew up")
    build_challenge_auto_defense(_Store(_Record()), on_verdict=boom)(_Chal(REASON_DOUBLE_SPEND))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
