"""sp1450 — the requester-side NO_ESCROW defense COVERS per-stage batches.

The requester's defense against an unauthorized on-chain escrow drain is scan_for_unauthorized_batches
-> match_unauthorized_batches against the IssuedAuthorizationStore: a committed batch naming the
requester with NO matching issued authorization is classified UNAUTHORIZED and challenged (NO_ESCROW,
which invalidates the leaf value so finalize won't draw it). This is the PRIMARY money boundary for
the per-stage path — commitBatch has no on-chain requester-auth check; the challenge window is the
defense, and escrow is fungible per-requester (no per-job cap).

BUG (this sprint): the requester signed a per-stage auth via build_per_stage_payment_authorization but
recorded NOTHING, so the store stayed EMPTY for per-stage and the matcher was BLIND to every per-stage
batch — the exact class of bug the single-payee path already fixed (build_and_record_payment_...).
A per-stage auth covers a SET of (payee, share) and each stage node commits its own batch
(provider=payee, value=share), so recording ONE entry per payee lets the existing conservative matcher
classify an HONEST per-stage batch AUTHORIZED and an INFLATED / FOREIGN one UNAUTHORIZED.

Money assertions — never weaken.
"""
from __future__ import annotations

import time

from eth_account import Account

from prsm.settlement.issued_authorization_store import (
    AUTHORIZED,
    UNAUTHORIZED,
    CommittedBatchView,
    IssuedAuthorizationStore,
    build_and_record_per_stage_payment_authorization,
    match_unauthorized_batches,
)

_REQ_KEY = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
_REQ_ADDR = Account.from_key(_REQ_KEY).address
_NODE_A = "0x" + "a1" * 20
_NODE_B = "0x" + "b2" * 20
_ATTACKER = "0x" + "cc" * 20
_WEI = 10 ** 18


def _store(tmp_path):
    return IssuedAuthorizationStore(tmp_path / "issued.json")


def _issue(store, tmp_path):
    """Requester signs + RECORDS a per-stage auth over {A: 3 FTNS, B: 2 FTNS}."""
    return build_and_record_per_stage_payment_authorization(
        store=store,
        requester_key=_REQ_KEY,
        payees=[(_NODE_A, 3), (_NODE_B, 2)],
        model_id="qwen2.5-72b", prompt="hi", max_tokens=16,
        privacy_tier="NONE", content_tier="A",
        expiry_unix=int(time.time()) + 86400,
    )


def _batch(provider, value_wei, *, requester=_REQ_ADDR, ts=None):
    return CommittedBatchView(
        batch_id=bytes.fromhex(f"{abs(hash((provider, value_wei))) % (16**64):064x}"),
        requester=requester, provider=provider, total_value_wei=int(value_wei),
        commit_timestamp=int(ts if ts is not None else time.time()),
        leaf_job_id_hashes=None)


def _classify(store, batch):
    return match_unauthorized_batches([batch], store, my_address=_REQ_ADDR)[0].classification


def test_honest_per_stage_batches_are_authorized(tmp_path):
    store = _store(tmp_path)
    _issue(store, tmp_path)
    # Each stage node commits exactly its authorized share → AUTHORIZED (not falsely griefed).
    assert _classify(store, _batch(_NODE_A, 3 * _WEI)) == AUTHORIZED
    assert _classify(store, _batch(_NODE_B, 2 * _WEI)) == AUTHORIZED


def test_inflated_per_stage_batch_is_unauthorized(tmp_path):
    store = _store(tmp_path)
    _issue(store, tmp_path)
    # Node A commits MORE than its authorized 3 FTNS share → over-draws the escrow → UNAUTHORIZED.
    assert _classify(store, _batch(_NODE_A, 5 * _WEI)) == UNAUTHORIZED


def test_foreign_provider_batch_is_unauthorized(tmp_path):
    store = _store(tmp_path)
    _issue(store, tmp_path)
    # An address that is NOT in the signed payee set names the requester → UNAUTHORIZED.
    assert _classify(store, _batch(_ATTACKER, 1 * _WEI)) == UNAUTHORIZED


def test_without_recording_the_matcher_is_blind(tmp_path):
    """The discriminator: sign the per-stage auth but DON'T record it (store=None → nothing stored).
    The matcher then cannot tell an honest per-stage batch from a foreign one — both UNAUTHORIZED —
    which griefs every honest stage node. Recording is what makes the defense fire correctly."""
    store = _store(tmp_path)
    build_and_record_per_stage_payment_authorization(
        store=None,  # NOT recorded — the pre-sp1450 behavior
        requester_key=_REQ_KEY, payees=[(_NODE_A, 3), (_NODE_B, 2)],
        model_id="qwen2.5-72b", prompt="hi", max_tokens=16,
        privacy_tier="NONE", content_tier="A", expiry_unix=int(time.time()) + 86400)
    assert _classify(store, _batch(_NODE_A, 3 * _WEI)) == UNAUTHORIZED  # honest batch griefed
