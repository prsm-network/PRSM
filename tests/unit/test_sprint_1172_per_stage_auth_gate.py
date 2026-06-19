"""Sprint 1172 — fail-closed per-stage authorization gate on the per-node commit (money-safety).

THE GAP (found by the money-rail holistic review): the per-stage PaymentAuthorization primitive
(sp1155) + payee-set builder (sp1162) had ZERO non-test callers — commit_per_node_share_batches
committed the per-node share-batches (which draw the requester's escrow at finalize) WITHOUT ever
verifying the requester authorized that payee set. The arc is not live yet, so it was a
must-fix-before-activation gap. THE FIX: commit_per_node_share_batches now REQUIRES the requester's
signed PerStagePaymentAuthorization and verifies EVERY (committer, share) against it BEFORE
committing anything — fail-closed (a missing/insufficient/expired/over-cap/wrong-set auth commits
NOTHING).
"""
from __future__ import annotations

import asyncio
import time

import pytest
from eth_account import Account
from eth_utils import keccak

from prsm.settlement.per_stage_commit import (
    PerStageAuthorizationError,
    _verify_authorization,
    commit_per_node_share_batches,
)
from prsm.settlement.per_stage_payment_authorization import (
    DEFAULT_CHAIN_ID,
    compute_payee_set_hash,
    sign_per_stage_authorization,
)

_REQ_KEY = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
_REQ_ADDR = Account.from_key(_REQ_KEY).address
_A, _B, _C = ("0x" + "a" * 40), ("0x" + "b" * 40), ("0x" + "c" * 40)
_ETH = {"n0": _A, "n1": _B, "n2": _C}
_SHARES_WEI = {"n0": 10, "n1": 20, "n2": 30}  # total 60


class _Share:
    def __init__(self, node_id, share_wei):
        self.node_id = node_id
        self.share_wei = share_wei
        self.batched_receipt = None  # never accessed before the gate raises


def _shares():
    return [_Share(nid, _SHARES_WEI[nid]) for nid in ("n0", "n1", "n2")]


def _committer(nid):
    return _ETH[nid]


def _signed_auth(*, total_cap=60, expiry_delta=86400, payees=None, requester=_REQ_ADDR,
                 key=_REQ_KEY, chain_id=DEFAULT_CHAIN_ID, request_bound=True):
    """Build + sign a PerStagePaymentAuthorization over ``payees`` (default = the real set)."""
    if payees is None:
        payees = [(_ETH[nid], _SHARES_WEI[nid]) for nid in ("n0", "n1", "n2")]
    payload = {
        "requester": requester,
        "payee_set_hash": compute_payee_set_hash(payees),
        "total_max_spend_wei": total_cap,
        "job_nonce": keccak(b"nonce"),
        "expiry_unix": int(time.time()) + expiry_delta,
        "request_hash": keccak(b"req") if request_bound else (b"\x00" * 32),
    }
    sig = sign_per_stage_authorization(payload, key, chain_id=chain_id)
    return payload, sig


# ── the gate AUTHORIZES the genuine, requester-signed payee set ──────────────────────


def test_gate_authorizes_the_real_signed_set():
    # no raise -> authorized
    _verify_authorization(_shares(), _committer, _signed_auth(), DEFAULT_CHAIN_ID, None)


# ── fail-closed: every way the auth fails to cover the set raises ────────────────────


def test_gate_refuses_missing_authorization():
    with pytest.raises(PerStageAuthorizationError):
        _verify_authorization(_shares(), _committer, None, DEFAULT_CHAIN_ID, None)


def test_gate_refuses_when_a_committed_share_is_not_in_the_signed_set():
    # the requester signed shares {10,20,30} but a node tries to commit a DIFFERENT share.
    bad_shares = [_Share("n0", 10), _Share("n1", 20), _Share("n2", 999)]  # n2 inflated
    with pytest.raises(PerStageAuthorizationError):
        _verify_authorization(bad_shares, _committer, _signed_auth(), DEFAULT_CHAIN_ID, None)


def test_gate_refuses_when_committed_set_does_not_hash_to_signed_commitment():
    # requester signed payees for A/B/C; the commit substitutes a DIFFERENT payee (D).
    rogue = {"n0": _A, "n1": _B, "n2": "0x" + "d" * 40}

    def _rogue_committer(nid):
        return rogue[nid]

    with pytest.raises(PerStageAuthorizationError):
        _verify_authorization(_shares(), _rogue_committer, _signed_auth(), DEFAULT_CHAIN_ID, None)


def test_gate_refuses_over_cap():
    # the signed cap (50) is LESS than the sum of shares (60) -> over-authorization refused.
    auth = _signed_auth(total_cap=50)
    with pytest.raises(PerStageAuthorizationError):
        _verify_authorization(_shares(), _committer, auth, DEFAULT_CHAIN_ID, None)


def test_gate_refuses_expired_auth():
    auth = _signed_auth(expiry_delta=-1)  # already expired
    with pytest.raises(PerStageAuthorizationError):
        _verify_authorization(_shares(), _committer, auth, DEFAULT_CHAIN_ID, None)


def test_gate_refuses_wrong_signer():
    # signed by a DIFFERENT key than the payload's requester -> signer mismatch.
    other = Account.create().key.hex()
    auth = _signed_auth(key=other)  # requester=_REQ_ADDR but signed by `other`
    with pytest.raises(PerStageAuthorizationError):
        _verify_authorization(_shares(), _committer, auth, DEFAULT_CHAIN_ID, None)


def test_gate_refuses_unbound_request_hash():
    auth = _signed_auth(request_bound=False)  # request_hash == 0 -> not request-bound
    with pytest.raises(PerStageAuthorizationError):
        _verify_authorization(_shares(), _committer, auth, DEFAULT_CHAIN_ID, None)


def test_gate_refuses_chain_id_mismatch():
    # signed on chain 8453 but verified on 84532 -> signer recovery mismatch (sp1127 class).
    auth = _signed_auth(chain_id=8453)
    with pytest.raises(PerStageAuthorizationError):
        _verify_authorization(_shares(), _committer, auth, 84532, None)


# ── the COMMIT is fail-closed: a bad auth commits NOTHING (never touches the clients) ─


def test_commit_refuses_and_touches_no_client_without_auth():
    class _Tripwire:
        async def accumulate(self, br):  # pragma: no cover - must never be reached
            raise AssertionError("commit MUST NOT accumulate when the auth gate fails")

        async def commit_ready_batches(self):  # pragma: no cover
            raise AssertionError("commit MUST NOT commit when the auth gate fails")

    with pytest.raises(PerStageAuthorizationError):
        asyncio.run(commit_per_node_share_batches(
            shares=_shares(), client_for_node=lambda nid: _Tripwire(),
            committer_address_for_node=_committer, authorization=None))
