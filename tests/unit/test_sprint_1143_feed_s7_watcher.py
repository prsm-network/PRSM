"""Sprint 1143 — §7-Brick-3: FEED the §7 InferenceReceipts to the observer watcher.

§7-Brick-1 (sp1141) retains the §7 ``InferenceReceipt`` keyed by committed-batch leaf
hash; §7-Brick-2 (sp1142) publishes it in the SAME canonical blob as the settlement
``BatchedReceipt`` set (``serialize_enriched_receipt_set`` / ``deserialize_enriched_
receipt_set``). Brick 3 is the CONSUMER side that completes the arc: the observer's
``VerifiedBatchCache`` retains the §7 receipts for a ROOT-VERIFIED batch, and the
``SettlementAuditEngine`` runs the §7 challenge verifier over them to surface
compute-integrity fraud — WITHOUT ever broadcasting.

THE CRITICAL SECURITY REQUIREMENT (ATTACK-A): the §7 receipt's
``settler_public_key_b64`` is supplied in the UNTRUSTED blob.
``verify_inference_receipt_for_challenge`` only attests the receipt is self-consistent
under WHATEVER key it is given — so a producer could publish a clean (fake) §7 receipt
under any key to MASK real fraud, or claim a wrong settler. Brick 3 therefore BINDS the
settler key to a CHAIN-ANCHORED source before trusting the verdict: for a §7 receipt
matched to a verified leaf (by leaf hash), it requires that hashing
``settler_public_key_b64`` the SAME WAY merkle builds the leaf ``provider_pubkey_hash``
(``keccak(b64decode(pubkey))``) yields EXACTLY the root-verified leaf
``provider_pubkey_hash``. (The adapter re-signs the shard payload with the settler
identity, so the committed leaf ``provider_pubkey_hash`` IS ``keccak(the settler
pubkey)`` — they MUST match.) If the binding fails, the §7 receipt is NOT bound to the
committed provider: its verdict is NOT run/trusted (recorded as unbound). With the
binding, an attacker cannot pass a receipt that both binds to the leaf AND verifies
without the real settler signature (which they lack).

These tests inject EVERYTHING — no network, no chain, no disk. They build REAL
InferenceReceipts via the real adapter + identities, and assert:
  (a) BINDING: a bound §7 receipt with a VALID settler signature -> no proven finding;
      a bound §7 receipt with a fraudulent chain / bad settler sig -> the finding
      surfaces (bound=True).
  (b) ATTACK-A REJECTION: a §7 receipt placed under a valid leaf but signed by an
      ATTACKER key (whose pubkey does NOT hash to the leaf provider_pubkey_hash) is
      marked UNBOUND and yields no clean OR fraud verdict.
  (c) only root-VERIFIED batches feed scan_inference_receipts (a rejected ingest never
      retains its §7 map).
  (d) NO-BROADCAST: submit/broadcast are never called (dry_run may be, for an
      on-chain-actionable finding).
  (e) FAIL-CLOSED: a §7 receipt that makes verify/bind raise is recorded+skipped; the
      scan does not crash; others still process.
  (f) LOOP e2e: run_once with an enriched blob root-verifies the batch, retains the §7
      receipts, surfaces a planted fraud, counts it, and never broadcasts.
"""
from __future__ import annotations

import dataclasses
import hashlib
from base64 import b64decode
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from eth_utils import keccak

from prsm.compute.inference.receipt import sign_receipt
from prsm.node.identity import generate_node_identity
from prsm.settlement.challenge_verifier import ChallengeReason
from prsm.settlement.inference_receipt_store import RetainedInferenceReceipt
from prsm.settlement.invalid_signature_submitter import DryRunResult
from prsm.settlement.merkle import (
    batched_receipt_to_leaf,
    build_tree_and_proofs,
    hash_leaf,
)
from prsm.settlement.published_batch_store import PublishedBatch
from prsm.settlement.receipt_set_blob import serialize_enriched_receipt_set
from prsm.settlement.settlement_audit_engine import (
    InferenceReceiptAuditFinding,
    ObservedBatch,
    OwnStakeSelector,
    SettlementAuditEngine,
    VerifiedBatchCache,
)
from prsm.settlement.settlement_audit_loop import (
    AuditRunResult,
    SettlementAuditLoop,
)

ZERO_GROUP = b"\x00" * 32
ME = "0x" + "a" * 40
PROVIDER = "0x" + "2" * 40
_WEI = 10 ** 18
_ADAPT_KW = dict(
    requester_address=ME,
    provider_address=PROVIDER,
    value_ftns=_WEI,
    local_escrow_id="job-1",
    executed_at_unix=1_700_000_000,
)


# ── builders: REAL InferenceReceipt -> REAL adapter -> BatchedReceipt ─────────


def _signed_ir(identity, **over):
    """A genuine, SIGNED §7 InferenceReceipt (settler_signature verifies under the
    identity's pubkey)."""
    from prsm.compute.inference.models import ContentTier, InferenceReceipt
    from prsm.compute.tee.models import PrivacyLevel, TEEType
    defaults = dict(
        job_id="job-1",
        request_id="req-1",
        model_id="gpt2",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=b"att-bytes",
        output_hash=hashlib.sha256(b"the output").digest(),
        duration_seconds=1.0,
        cost_ftns=Decimal("1.0"),
        settler_node_id=identity.node_id,
    )
    defaults.update(over)
    unsigned = InferenceReceipt(**defaults)
    return sign_receipt(unsigned, identity)


def _adapt(ir, identity, **over):
    from prsm.settlement.inference_adapter import (
        inference_receipt_to_batched_receipt,
    )
    kw = dict(_ADAPT_KW)
    kw.update(over)
    return inference_receipt_to_batched_receipt(receipt=ir, identity=identity, **kw)


def _root_for(receipts):
    return build_tree_and_proofs(
        [hash_leaf(batched_receipt_to_leaf(br)) for br in receipts]
    ).root


def _published(batch_id, receipts):
    return PublishedBatch(
        batch_id=batch_id,
        merkle_root=_root_for(receipts),
        commit_timestamp=1000,
        receipts=list(receipts),
    )


def _observed(batch_id, receipts, *, provider=PROVIDER, requester=ME,
              group=ZERO_GROUP, root=None, cid=None):
    return ObservedBatch(
        batch_id=batch_id,
        provider_address=provider,
        requester_address=requester,
        merkle_root=root if root is not None else _root_for(receipts),
        consensus_group_id=group,
        cid=cid,
    )


def _retained(ir, *, settler_public_key_b64, leaf_hash, stage_public_keys=None):
    return RetainedInferenceReceipt(
        leaf_hash=bytes(leaf_hash),
        inference_receipt=ir,
        settler_public_key_b64=settler_public_key_b64,
        stage_public_keys=stage_public_keys,
        retained_at=1_700_000_500,
    )


class _FakeDryRunClient:
    """Dry-run-only client. submit/broadcast MUST never fire."""

    def __init__(self, would_succeed=True):
        self._would_succeed = would_succeed
        self.dry_run_calls = []
        self.submit = MagicMock(name="submit")
        self.broadcast = AsyncMock(name="broadcast")

    def dry_run(self, challenge):
        self.dry_run_calls.append(challenge)
        return DryRunResult(would_succeed=self._would_succeed)


# ── (a) BINDING — bound + genuine -> clean; bound + fraud -> surfaced ─────────


def test_bound_genuine_receipt_no_finding():
    ident = generate_node_identity("settler")
    ir = _signed_ir(ident)
    br = _adapt(ir, ident)
    leaf_hash = hash_leaf(batched_receipt_to_leaf(br))

    pb = _published(b"\x01" * 32, [br])
    obs = _observed(b"\x01" * 32, [br])
    # The §7 receipt is published under the REAL settler pubkey (binds to the leaf).
    s7 = {leaf_hash: _retained(ir, settler_public_key_b64=ident.public_key_b64,
                               leaf_hash=leaf_hash)}

    cache = VerifiedBatchCache()
    assert cache.ingest(obs, pb, inference_receipts=s7) is True
    engine = SettlementAuditEngine(cache)
    findings = engine.scan_inference_receipts()

    # The settler key binds to the leaf AND the receipt verifies -> NO proven finding.
    assert all(not f.proven for f in findings)
    # The binding succeeded for the one receipt scanned.
    assert any(f.bound for f in findings) or findings == []


def test_bound_fraudulent_receipt_surfaces_finding():
    """A genuinely-fraudulent §7 receipt (signed field tampered AFTER signing, so the
    settler signature no longer verifies) — published under the REAL settler key so it
    BINDS — surfaces the INVALID_SETTLER_SIGNATURE finding (bound=True)."""
    ident = generate_node_identity("settler")
    ir = _signed_ir(ident)
    br = _adapt(ir, ident)
    leaf_hash = hash_leaf(batched_receipt_to_leaf(br))

    # Tamper a signed field AFTER signing -> settler signature no longer verifies.
    tampered_ir = dataclasses.replace(
        ir, output_hash=hashlib.sha256(b"a DIFFERENT output").digest()
    )

    pb = _published(b"\x01" * 32, [br])
    obs = _observed(b"\x01" * 32, [br])
    s7 = {leaf_hash: _retained(tampered_ir,
                               settler_public_key_b64=ident.public_key_b64,
                               leaf_hash=leaf_hash)}

    cache = VerifiedBatchCache()
    assert cache.ingest(obs, pb, inference_receipts=s7) is True
    engine = SettlementAuditEngine(cache)
    findings = engine.scan_inference_receipts()

    proven = [f for f in findings if f.proven]
    assert proven, "a bound fraudulent receipt must surface a proven finding"
    f = proven[0]
    assert f.bound is True
    assert f.batch_id == b"\x01" * 32
    assert f.leaf_hash == leaf_hash
    assert f.reason == ChallengeReason.INVALID_SETTLER_SIGNATURE
    # INVALID_SETTLER_SIGNATURE is on-chain-actionable (ReasonCode.INVALID_SIGNATURE==1).
    assert f.on_chain_reason == 1


# ── (b) ATTACK-A — substituted attacker key -> UNBOUND, no verdict ────────────


def test_attacker_key_substitution_is_unbound_no_false_clean_no_false_fraud():
    """A producer publishes a CLEAN (fake) §7 receipt under an ATTACKER key to MASK the
    real settler. The attacker key does NOT hash to the leaf provider_pubkey_hash, so the
    §7 receipt is UNBOUND: scan_inference_receipts records it unbound and surfaces NEITHER
    a (false) clean verdict NOR a (false) fraud verdict from it."""
    settler = generate_node_identity("settler")
    attacker = generate_node_identity("attacker")
    ir = _signed_ir(settler)
    br = _adapt(ir, settler)
    leaf_hash = hash_leaf(batched_receipt_to_leaf(br))

    # Sanity: the attacker pubkey does NOT hash to the leaf provider_pubkey_hash.
    leaf = batched_receipt_to_leaf(br)
    assert keccak(b64decode(attacker.public_key_b64)) != leaf.provider_pubkey_hash
    assert keccak(b64decode(settler.public_key_b64)) == leaf.provider_pubkey_hash

    # The attacker re-signs a clean receipt under THEIR OWN key so it self-verifies under
    # the attacker key — but the key isn't bound to the committed leaf.
    attacker_ir = _signed_ir(attacker, settler_node_id=attacker.node_id)
    pb = _published(b"\x01" * 32, [br])
    obs = _observed(b"\x01" * 32, [br])
    s7 = {leaf_hash: _retained(attacker_ir,
                               settler_public_key_b64=attacker.public_key_b64,
                               leaf_hash=leaf_hash)}

    cache = VerifiedBatchCache()
    assert cache.ingest(obs, pb, inference_receipts=s7) is True
    engine = SettlementAuditEngine(cache)
    findings = engine.scan_inference_receipts()

    # Exactly one record for the substituted receipt — and it is UNBOUND.
    unbound = [f for f in findings if not f.bound]
    assert len(unbound) == 1
    u = unbound[0]
    assert u.leaf_hash == leaf_hash
    assert u.bound is False
    # No verdict was trusted: no proven fraud AND no clean verdict from the unbound key.
    assert not any(f.proven for f in findings)
    assert not any(f.bound for f in findings)


# ── (c) only root-VERIFIED batches feed the §7 scan ───────────────────────────


def test_rejected_batch_s7_map_never_scanned():
    """A §7 map for a batch whose ingest is REJECTED (root mismatch) is never retained,
    so scan_inference_receipts never sees it."""
    ident = generate_node_identity("settler")
    ir = _signed_ir(ident)
    br = _adapt(ir, ident)
    leaf_hash = hash_leaf(batched_receipt_to_leaf(br))
    pb = _published(b"\x01" * 32, [br])

    # Observed root is WRONG -> cross-check fails -> ingest rejects, caches nothing.
    obs = _observed(b"\x01" * 32, [br], root=b"\x99" * 32)
    s7 = {leaf_hash: _retained(ir, settler_public_key_b64=ident.public_key_b64,
                               leaf_hash=leaf_hash)}

    cache = VerifiedBatchCache()
    assert cache.ingest(obs, pb, inference_receipts=s7) is False
    assert cache.inference_receipts_for(b"\x01" * 32) == {}
    engine = SettlementAuditEngine(cache)
    assert engine.scan_inference_receipts() == []


def test_s7_receipt_for_unknown_leaf_is_dropped():
    """A §7 entry keyed by a leaf hash NOT in the verified batch leaf set is dropped on
    ingest (only leaves of the verified batch are retained)."""
    ident = generate_node_identity("settler")
    ir = _signed_ir(ident)
    br = _adapt(ir, ident)
    leaf_hash = hash_leaf(batched_receipt_to_leaf(br))
    pb = _published(b"\x01" * 32, [br])
    obs = _observed(b"\x01" * 32, [br])

    bogus_leaf = b"\x55" * 32
    s7 = {
        leaf_hash: _retained(ir, settler_public_key_b64=ident.public_key_b64,
                             leaf_hash=leaf_hash),
        bogus_leaf: _retained(ir, settler_public_key_b64=ident.public_key_b64,
                              leaf_hash=bogus_leaf),
    }
    cache = VerifiedBatchCache()
    assert cache.ingest(obs, pb, inference_receipts=s7) is True
    retained = cache.inference_receipts_for(b"\x01" * 32)
    assert set(retained.keys()) == {leaf_hash}


# ── (d) NO-BROADCAST ──────────────────────────────────────────────────────────


def test_scan_inference_receipts_never_broadcasts():
    ident = generate_node_identity("settler")
    ir = _signed_ir(ident)
    br = _adapt(ir, ident)
    leaf_hash = hash_leaf(batched_receipt_to_leaf(br))
    # Fraudulent (tampered) -> on-chain-actionable INVALID_SETTLER_SIGNATURE -> dry_run.
    tampered_ir = dataclasses.replace(
        ir, output_hash=hashlib.sha256(b"DIFFERENT").digest()
    )
    pb = _published(b"\x01" * 32, [br])
    obs = _observed(b"\x01" * 32, [br])
    s7 = {leaf_hash: _retained(tampered_ir,
                               settler_public_key_b64=ident.public_key_b64,
                               leaf_hash=leaf_hash)}

    cache = VerifiedBatchCache()
    cache.ingest(obs, pb, inference_receipts=s7)
    fake = _FakeDryRunClient(would_succeed=True)
    engine = SettlementAuditEngine(cache, dry_run_client=fake)
    findings = engine.scan_inference_receipts()

    assert any(f.proven and f.on_chain_reason == 1 for f in findings)
    # dry_run MAY fire for the on-chain-actionable finding; submit/broadcast NEVER.
    fake.submit.assert_not_called()
    fake.broadcast.assert_not_called()
    # The actionable finding is surfaced with the on-chain reason code. Its dry-run is
    # advisory: here the COMMITTED leaf's shard signature is genuinely valid (the §7
    # fraud is the settler signature over the InferenceReceipt payload — a DIFFERENT
    # preimage), so the on-chain INVALID_SIGNATURE leaf challenge fail-fasts and the
    # dry-run verdict stays unknown (None). submit/broadcast still never fire.
    actionable = [f for f in findings if f.proven and f.on_chain_reason == 1]
    assert actionable[0].dry_run_ok is None


# ── (e) FAIL-CLOSED ─────────────────────────────────────────────────────────


def test_fail_closed_per_item_raising_receipt_recorded_and_skipped():
    """A §7 receipt whose verify/bind raises is recorded+skipped; the scan does not
    crash; the OTHER (genuine) receipt is still processed."""
    ident = generate_node_identity("settler")
    ir0 = _signed_ir(ident, job_id="j0", request_id="r0")
    ir1 = _signed_ir(ident, job_id="j1", request_id="r1")
    br0 = _adapt(ir0, ident, local_escrow_id="e0")
    br1 = _adapt(ir1, ident, local_escrow_id="e1")
    lh0 = hash_leaf(batched_receipt_to_leaf(br0))
    lh1 = hash_leaf(batched_receipt_to_leaf(br1))
    pb = _published(b"\x01" * 32, [br0, br1])
    obs = _observed(b"\x01" * 32, [br0, br1])

    # A retained record whose inference_receipt raises when the verifier touches it.
    class _Boom:
        def __getattr__(self, name):
            raise RuntimeError("boom on access")

    boom_rec = RetainedInferenceReceipt(
        leaf_hash=lh0,
        inference_receipt=_signed_ir(ident, job_id="j0", request_id="r0"),
        settler_public_key_b64=ident.public_key_b64,
        stage_public_keys=None,
        retained_at=1,
    )
    # Replace the inference_receipt with a landmine via object.__setattr__ (frozen).
    object.__setattr__(boom_rec, "inference_receipt", _Boom())

    s7 = {
        lh0: boom_rec,
        lh1: _retained(ir1, settler_public_key_b64=ident.public_key_b64,
                       leaf_hash=lh1),
    }
    cache = VerifiedBatchCache()
    assert cache.ingest(obs, pb, inference_receipts=s7) is True
    engine = SettlementAuditEngine(cache)
    # Must not raise.
    findings = engine.scan_inference_receipts()
    # The boom item is recorded (skipped, not crashed); the genuine item processes clean.
    assert any(f.leaf_hash == lh0 and not f.proven for f in findings)
    # No spurious proven fraud from either.
    assert not any(f.proven for f in findings)


# ── (f) LOOP e2e ───────────────────────────────────────────────────────────


class _FakeEnumerator:
    def __init__(self, observed):
        self._observed = list(observed)

    async def enumerate_committed_batches(self, from_block, to_block, *, provider=None):
        return list(self._observed)


class _FakeAd:
    def __init__(self, cid):
        self.cid = cid


class _FakeAdIndex:
    def __init__(self, mapping):
        self._ads = {bytes(b).hex(): _FakeAd(c) for b, c in mapping.items()}

    def get(self, batch_id):
        return self._ads.get(bytes(batch_id).hex())


class _FakeFetcher:
    def __init__(self, blobs):
        self._blobs = dict(blobs)
        self.fetched = []

    async def fetch(self, cid):
        self.fetched.append(cid)
        return self._blobs[cid]


@pytest.mark.asyncio
async def test_loop_e2e_enriched_blob_surfaces_planted_s7_fraud():
    ident = generate_node_identity("settler")
    ir = _signed_ir(ident)
    br = _adapt(ir, ident)
    leaf_hash = hash_leaf(batched_receipt_to_leaf(br))
    pb = _published(b"\x01" * 32, [br])

    # Planted §7 fraud: tampered receipt under the REAL settler key (binds + fraudulent).
    tampered_ir = dataclasses.replace(
        ir, output_hash=hashlib.sha256(b"forged").digest()
    )
    s7 = {leaf_hash: _retained(tampered_ir,
                               settler_public_key_b64=ident.public_key_b64,
                               leaf_hash=leaf_hash)}
    blob = serialize_enriched_receipt_set(pb, inference_receipts=s7)

    obs = [_observed(b"\x01" * 32, [br], cid="cid-1")]
    fetcher = _FakeFetcher({"cid-1": blob})
    ad_index = _FakeAdIndex({b"\x01" * 32: "cid-1"})
    cache = VerifiedBatchCache(max_batches=8)
    fake = _FakeDryRunClient(would_succeed=True)
    engine = SettlementAuditEngine(cache, dry_run_client=fake)
    loop = SettlementAuditLoop(
        enumerator=_FakeEnumerator(obs),
        selector=OwnStakeSelector(my_address=ME),
        ad_index=ad_index,
        fetcher=fetcher,
        cache=cache,
        engine=engine,
        max_fetches_per_run=100,
    )

    result = await loop.run_once(from_block=0, to_block=10)

    assert isinstance(result, AuditRunResult)
    assert result.verified == 1
    assert result.rejected == 0
    # §7 receipts retained for the verified batch.
    assert set(cache.inference_receipts_for(b"\x01" * 32).keys()) == {leaf_hash}
    # The planted §7 fraud surfaces.
    proven = [f for f in result.inference_receipt_findings if f.proven]
    assert proven and proven[0].bound is True
    assert proven[0].reason == ChallengeReason.INVALID_SETTLER_SIGNATURE
    assert result.inference_receipt_finding_count == len(result.inference_receipt_findings)
    assert result.inference_receipt_finding_count >= 1
    # Never broadcast.
    fake.submit.assert_not_called()
    fake.broadcast.assert_not_called()


@pytest.mark.asyncio
async def test_loop_e2e_genuine_enriched_blob_no_fraud():
    ident = generate_node_identity("settler")
    ir = _signed_ir(ident)
    br = _adapt(ir, ident)
    leaf_hash = hash_leaf(batched_receipt_to_leaf(br))
    pb = _published(b"\x01" * 32, [br])
    s7 = {leaf_hash: _retained(ir, settler_public_key_b64=ident.public_key_b64,
                               leaf_hash=leaf_hash)}
    blob = serialize_enriched_receipt_set(pb, inference_receipts=s7)

    obs = [_observed(b"\x01" * 32, [br], cid="cid-1")]
    cache = VerifiedBatchCache()
    fake = _FakeDryRunClient()
    engine = SettlementAuditEngine(cache, dry_run_client=fake)
    loop = SettlementAuditLoop(
        enumerator=_FakeEnumerator(obs),
        selector=OwnStakeSelector(my_address=ME),
        ad_index=_FakeAdIndex({b"\x01" * 32: "cid-1"}),
        fetcher=_FakeFetcher({"cid-1": blob}),
        cache=cache,
        engine=engine,
    )
    result = await loop.run_once(from_block=0, to_block=10)
    assert result.verified == 1
    assert not any(f.proven for f in result.inference_receipt_findings)
    fake.submit.assert_not_called()
    fake.broadcast.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
