"""Sprint 1129 — challenge/dispute brick 5: the ACTIVE challenge watcher.

The watcher ORCHESTRATES the three existing pure bricks:
  - challenge_verifier.verify_inference_receipt_for_challenge -> ChallengeReport
  - challenge_assembler.assemble_invalid_signature_challenge -> InvalidSignatureChallenge
  - invalid_signature_submitter.InvalidSignatureChallengeSubmitter.dry_run (READ-ONLY)

It pulls units from a PLUGGABLE async receipt source, verifies each, and for every
on-chain-actionable finding it ASSEMBLES + DRY-RUNS the challengeReceipt and surfaces an
ActionableChallenge. It MUST NOT broadcast/sign/slash (submit() is NEVER called), and a bad
unit must be surfaced as an error WITHOUT aborting the scan or raising.

Tested with fakes/AsyncMock — no network, no live chain.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.compute.inference.models import ContentTier, InferenceReceipt
from prsm.compute.inference.receipt import sign_receipt
from prsm.compute.shard_receipt import (
    ShardExecutionReceipt,
    build_receipt_signing_payload,
)
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import generate_node_identity
from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.invalid_signature_submitter import DryRunResult

from prsm.settlement.challenge_watcher import (
    ActionableChallenge,
    ChallengeWatcher,
    ScanResult,
    WatchUnit,
)


# ── builders ─────────────────────────────────────────────────────────────────────

def _inference_receipt(settler, *, request_id="req-1"):
    r = InferenceReceipt(
        job_id="job-1", request_id=request_id, model_id="gpt2",
        content_tier=ContentTier.A, privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0, tee_type=TEEType.SOFTWARE, tee_attestation=b"att",
        output_hash=b"\xab" * 32, duration_seconds=1.0, cost_ftns=Decimal("0.1"),
        settler_node_id="", stage_activation_chain=None, topology_assignment=None)
    return sign_receipt(r, settler)


def _shard_receipt(idn, *, shard, valid):
    out = "ab" * 32
    good = build_receipt_signing_payload(
        job_id="j", shard_index=shard, output_hash=out, executed_at_unix=1000)
    sig = idn.sign(good) if valid else idn.sign(build_receipt_signing_payload(
        job_id="j", shard_index=shard, output_hash="cd" * 32, executed_at_unix=1000))
    rec = ShardExecutionReceipt(
        job_id="j", shard_index=shard, provider_id=idn.node_id,
        provider_pubkey_b64=idn.public_key_b64, output_hash=out,
        executed_at_unix=1000, signature=sig)
    return BatchedReceipt(
        receipt=rec, requester_address="0x" + "1" * 40,
        provider_address="0x" + "2" * 40, value_ftns=10**18, local_escrow_id="e")


def _bad_unit(*, batch_id=b"\x11" * 32, target_index=1):
    """A unit whose inference receipt has a BAD settler signature (on-chain actionable
    INVALID_SIGNATURE) AND whose shard batch has a fraudulent receipt at target_index so
    the assembler will not fail-fast."""
    settler = generate_node_identity("settler")
    imposter = generate_node_identity("imposter")
    receipt = _inference_receipt(settler)
    a = generate_node_identity("a")
    b = generate_node_identity("b")
    batch = [_shard_receipt(a, shard=0, valid=True),
             _shard_receipt(b, shard=1, valid=False)]
    return WatchUnit(
        batch_id=batch_id,
        inference_receipt=receipt,
        settler_public_key_b64=imposter.public_key_b64,  # wrong key → bad signature
        batch_receipts=batch,
        target_index=target_index,
    )


def _clean_unit():
    settler = generate_node_identity("settler")
    receipt = _inference_receipt(settler)
    a = generate_node_identity("a")
    batch = [_shard_receipt(a, shard=0, valid=True)]
    return WatchUnit(
        batch_id=b"\x22" * 32,
        inference_receipt=receipt,
        settler_public_key_b64=settler.public_key_b64,  # correct key → valid
        batch_receipts=batch,
        target_index=0,
    )


def _source(units):
    async def gen():
        for u in units:
            yield u
    return gen


def _fake_submitter(*, would_succeed=True, revert_reason=None):
    """An AsyncMock-backed fake submitter. dry_run is a sync method returning a
    DryRunResult; submit() is an AsyncMock so we can assert it is NEVER called."""
    sub = MagicMock()
    sub.dry_run = MagicMock(
        return_value=DryRunResult(would_succeed=would_succeed,
                                  revert_reason=revert_reason))
    sub.submit = AsyncMock()  # the user-gated broadcast — MUST NOT be called
    return sub


# ── (a) known-bad receipt surfaces exactly one actionable challenge ────────────────

@pytest.mark.asyncio
async def test_known_bad_receipt_surfaces_one_actionable_challenge():
    sub = _fake_submitter(would_succeed=True)
    watcher = ChallengeWatcher(source=_source([_bad_unit()]), dry_run_client=sub)

    result = await watcher.scan()

    assert isinstance(result, ScanResult)
    assert len(result.actionable) == 1
    assert result.errors == []
    ac = result.actionable[0]
    assert isinstance(ac, ActionableChallenge)
    assert ac.batch_id == b"\x11" * 32
    assert ac.target_index == 1
    # the finding is the on-chain INVALID_SIGNATURE ground (reason code 1)
    assert ac.on_chain_reason == 1
    # the assembled challenge inputs are present + the dry-run confirmed acceptance
    assert ac.challenge is not None
    assert ac.challenge.reason_code == 1
    assert ac.dry_run_ok is True


# ── (b) a clean receipt surfaces nothing ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_clean_receipt_surfaces_nothing():
    sub = _fake_submitter(would_succeed=True)
    watcher = ChallengeWatcher(source=_source([_clean_unit()]), dry_run_client=sub)

    result = await watcher.scan()

    assert result.actionable == []
    assert result.errors == []
    sub.dry_run.assert_not_called()  # nothing actionable → no dry-run needed


# ── (c) NO-BROADCAST: submit()/sign is NEVER called ────────────────────────────────

@pytest.mark.asyncio
async def test_watcher_never_broadcasts():
    sub = _fake_submitter(would_succeed=True)
    watcher = ChallengeWatcher(
        source=_source([_bad_unit(), _clean_unit(), _bad_unit(batch_id=b"\x33" * 32)]),
        dry_run_client=sub)

    result = await watcher.scan()

    # two bad units → two actionable challenges, both dry-run only
    assert len(result.actionable) == 2
    # the user-gated broadcast path was NEVER touched
    sub.submit.assert_not_called()
    sub.submit.assert_not_awaited()
    # only the read-only dry_run was used
    assert sub.dry_run.call_count == 2


# ── (d) FAIL-CLOSED: a raising unit is surfaced as an error, scan continues ─────────

@pytest.mark.asyncio
async def test_fail_closed_records_error_and_continues():
    sub = _fake_submitter(would_succeed=True)

    # the dry-run RAISES on the first actionable challenge it sees
    boom_seen = {"n": 0}

    def _maybe_raise(challenge):
        boom_seen["n"] += 1
        if boom_seen["n"] == 1:
            raise RuntimeError("dry-run exploded")
        return DryRunResult(would_succeed=True)

    sub.dry_run = MagicMock(side_effect=_maybe_raise)

    watcher = ChallengeWatcher(
        source=_source([_bad_unit(batch_id=b"\x11" * 32),
                        _bad_unit(batch_id=b"\x44" * 32)]),
        dry_run_client=sub)

    # MUST NOT raise even though one unit's dry-run blew up
    result = await watcher.scan()

    # first unit recorded as an error, second still processed into an actionable
    assert len(result.errors) == 1
    assert result.errors[0].batch_id == b"\x11" * 32
    assert "dry-run exploded" in result.errors[0].message
    assert len(result.actionable) == 1
    assert result.actionable[0].batch_id == b"\x44" * 32


# ── (e) dry-run "would-revert" → surfaced with dry_run_ok=False ─────────────────────

@pytest.mark.asyncio
async def test_would_revert_dry_run_surfaces_with_dry_run_ok_false():
    sub = _fake_submitter(would_succeed=False, revert_reason="ChallengeNotProven")
    watcher = ChallengeWatcher(source=_source([_bad_unit()]), dry_run_client=sub)

    result = await watcher.scan()

    assert len(result.actionable) == 1
    ac = result.actionable[0]
    assert ac.dry_run_ok is False
    assert ac.dry_run_revert_reason == "ChallengeNotProven"
    # still surfaced (operator sees it) but NOT broadcast
    sub.submit.assert_not_called()


# ── (f) FAIL-CLOSED: an actionable finding whose on-chain reason is NOT ─────────────
#        INVALID_SIGNATURE must be recorded as an error, never mis-assembled.

@pytest.mark.asyncio
async def test_fail_closed_on_unsupported_on_chain_reason(monkeypatch):
    """The assembler only produces INVALID_SIGNATURE (reason 1) challenges. Today that's
    the sole on-chain-actionable ground, but if a future ChallengeReason gets a different
    on-chain reason code, the watcher must NOT blindly assemble it as INVALID_SIGNATURE —
    it surfaces a recorded error (fail-closed) instead of a wrong challenge."""
    from prsm.settlement import challenge_watcher as cw

    fake_finding = MagicMock()
    fake_finding.on_chain_reason = 2  # a hypothetical future code, NOT INVALID_SIGNATURE (1)
    fake_finding.reason = MagicMock(value="future-ground")
    fake_report = MagicMock()
    fake_report.on_chain_actionable.return_value = [fake_finding]
    monkeypatch.setattr(
        cw, "verify_inference_receipt_for_challenge", lambda *a, **k: fake_report)

    sub = _fake_submitter(would_succeed=True)
    watcher = ChallengeWatcher(source=_source([_bad_unit()]), dry_run_client=sub)

    result = await watcher.scan()

    # NOT mis-assembled/surfaced as an actionable challenge
    assert result.actionable == []
    # surfaced as a recorded, operator-visible error instead
    assert len(result.errors) == 1
    assert "unsupported on-chain reason 2" in result.errors[0].message
    # never reached assemble/dry-run, never broadcast
    sub.dry_run.assert_not_called()
    sub.submit.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
