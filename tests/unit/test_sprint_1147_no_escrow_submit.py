"""Sprint 1147 — NO_ESCROW submit (the requester self-dispute brick).

sp1146 built the requester-side matcher (match_unauthorized_batches -> per-batch
AUTHORIZED/UNAUTHORIZED). This brick adds the SUBMIT: assemble a NO_ESCROW
challengeReceipt input (leaf + merkle_proof + reason=2 + empty auxData) ONLY for
batches the matcher classified UNAUTHORIZED, extends the generalized
ChallengeSubmitter to accept reason NO_ESCROW (so a REQUESTER-keyed submitter can
dry_run + user-gated submit it), and a requester-side flow that assembles + (read-only)
dry-runs only — NEVER auto-submits/broadcasts.

Tests (NO network — injected fakes + locally-built batches):
  (a) assemble_no_escrow_challenge builds a valid challenge: to_call_args() has
      reason_code==2, aux_data==b"", and the merkle_proof verifies the leaf against the
      batch root (independently recomputed) — passes the on-chain caller-side check.
  (b) ONLY-UNAUTHORIZED: a mix of AUTHORIZED + UNAUTHORIZED classifications assembles
      challenges ONLY for the UNAUTHORIZED ones.
  (c) submitter extension: ChallengeSubmitter dry_run + submit ACCEPT a NoEscrowChallenge
      (reason 2) and forward to_call_args; an EXPIRED-reason challenge is still REJECTED.
  (d) NO-AUTO-SUBMIT: assemble_no_escrow_challenges with an injected dry_run_client whose
      submit/broadcast is a Mock -> never called (dry_run may be); fail-closed on missing
      receipts.
  (e) range/empty guards -> ValueError.
  (f) end-to-end: match_unauthorized_batches -> feed UNAUTHORIZED classifications +
      receipts to assemble_no_escrow_challenges -> exactly the unauthorized batch gets a
      NO_ESCROW challenge (dry-run ok), the authorized one does not.
"""
from __future__ import annotations

from unittest.mock import Mock

import pytest
from eth_account import Account

from prsm.compute.shard_receipt import (
    ShardExecutionReceipt,
    build_receipt_signing_payload,
)
from prsm.node.identity import generate_node_identity
from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.payment_client import build_payment_authorization
from prsm.settlement.merkle import (
    batched_receipt_to_leaf,
    build_merkle_root,
    hash_leaf,
    verify_merkle_proof,
)
from prsm.settlement.no_escrow_assembler import (
    REASON_NO_ESCROW,
    NoEscrowChallenge,
    assemble_no_escrow_challenge,
    assemble_no_escrow_challenges,
)
from prsm.settlement.invalid_signature_submitter import (
    ChallengeSubmitter,
    SUPPORTED_CHALLENGE_REASONS,
)
from prsm.settlement.issued_authorization_store import (
    AUTHORIZED,
    UNAUTHORIZED,
    BatchAuthorizationClassification,
    CommittedBatchView,
    IssuedAuthorizationStore,
    match_unauthorized_batches,
)

WEI = 10 ** 18


# ── helpers ──────────────────────────────────────────────────────────────────────

def _receipt(identity, *, job="job-1", shard=0, out=None, value=WEI,
             requester="0x" + "1" * 40, provider="0x" + "2" * 40):
    out = out or ("ab" * 32)
    payload = build_receipt_signing_payload(
        job_id=job, shard_index=shard, output_hash=out, executed_at_unix=1000)
    sig = identity.sign(payload)
    rec = ShardExecutionReceipt(
        job_id=job, shard_index=shard, provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64, output_hash=out,
        executed_at_unix=1000, signature=sig)
    return BatchedReceipt(
        receipt=rec, requester_address=requester,
        provider_address=provider, value_ftns=value, local_escrow_id="e1")


def _leaf_hashes(batch):
    return [hash_leaf(batched_receipt_to_leaf(br)) for br in batch]


def _auth_payload(*, requester_key, provider, max_spend_ftns, expiry_unix):
    return build_payment_authorization(
        requester_key=requester_key,
        provider_address=provider,
        model_id="m", prompt="p", max_tokens=10,
        privacy_tier="standard", content_tier="standard",
        max_spend_ftns=max_spend_ftns,
        expiry_unix=expiry_unix,
        job_nonce=None,
        chain_id=8453,
    )


# ── submitter fakes (mirror sp1119) ────────────────────────────────────────────────

class _FakeFn:
    def __init__(self, parent, call_ok=True, call_error="boom"):
        self._parent = parent
        self._call_ok = call_ok
        self._call_error = call_error

    def call(self, opts):
        self._parent.call_opts = opts
        if not self._call_ok:
            raise RuntimeError(self._call_error)
        return []

    def build_transaction(self, overrides):
        self._parent.built_overrides = overrides
        return {"to": "0xregistry", **overrides}


class _FakeFunctions:
    def __init__(self, parent):
        self._parent = parent

    def challengeReceipt(self, *args):
        self._parent.call_args = args
        return _FakeFn(self._parent, self._parent.call_ok, self._parent.call_error)


class _FakeRegistry:
    def __init__(self, parent):
        self.functions = _FakeFunctions(parent)


class _FakeEthAccount:
    def sign_transaction(self, tx, key):
        return type("Signed", (), {"raw_transaction": b"\xde\xad"})()


class _FakeReceipt:
    def __init__(self, status):
        self.status = status


class _FakeEth:
    def __init__(self, parent):
        self._parent = parent
        self.account = _FakeEthAccount()

    def get_transaction_count(self, addr, block):
        return 7

    @property
    def gas_price(self):
        return 1_000_000_000

    @property
    def chain_id(self):
        return 8453

    def send_raw_transaction(self, raw):
        return b"\xab\xcd"

    def wait_for_transaction_receipt(self, h, timeout=120):
        return _FakeReceipt(self._parent.tx_status)


class _FakeWeb3:
    def __init__(self, parent):
        self.eth = _FakeEth(parent)


class _Harness:
    def __init__(self, *, call_ok=True, call_error="reverted", tx_status=1):
        self.call_ok = call_ok
        self.call_error = call_error
        self.tx_status = tx_status
        self.call_args = None
        self.built_overrides = None
        self.call_opts = None
        # requester-keyed account: msg.sender == requester for NO_ESCROW
        self.account = type("A", (), {"address": "0xrequester", "key": b"k"})()
        self.web3 = _FakeWeb3(self)
        self.registry = _FakeRegistry(self)

    def submitter(self):
        return ChallengeSubmitter(
            web3=self.web3, registry=self.registry, account=self.account)


# ── (a) assemble builds a valid challenge ───────────────────────────────────────────

def test_assemble_builds_valid_no_escrow_challenge():
    a, b, c = (generate_node_identity(n) for n in ("a", "b", "c"))
    batch = [_receipt(a, shard=0), _receipt(b, shard=1), _receipt(c, shard=2)]
    batch_id = b"\x11" * 32

    ch = assemble_no_escrow_challenge(
        batch_id=batch_id, batch_receipts=batch, target_index=1)

    assert isinstance(ch, NoEscrowChallenge)
    args = ch.to_call_args()
    # (batch_id, leaf_tuple, merkle_proof, reason_code, aux_data)
    assert args[0] == batch_id
    assert args[3] == 2  # NO_ESCROW
    assert args[4] == b""  # auxData UNUSED for NO_ESCROW
    assert ch.reason_code == REASON_NO_ESCROW == 2

    # Independently recompute the root + verify the proof verifies the leaf — this is
    # exactly the on-chain caller-side MerkleProof.verify(...) at line 661.
    root = build_merkle_root(_leaf_hashes(batch))
    leaf_hash = hash_leaf(batched_receipt_to_leaf(batch[1]))
    assert verify_merkle_proof(list(args[2]), root, leaf_hash) is True


def test_assemble_single_leaf_batch_proof_is_empty_and_verifies():
    """The MVP 1-leaf batch: empty proof verifies iff root == leaf_hash."""
    a = generate_node_identity("a")
    batch = [_receipt(a, shard=0)]
    ch = assemble_no_escrow_challenge(
        batch_id=b"\x22" * 32, batch_receipts=batch, target_index=0)
    root = build_merkle_root(_leaf_hashes(batch))
    leaf_hash = hash_leaf(batched_receipt_to_leaf(batch[0]))
    assert ch.merkle_proof == []
    assert verify_merkle_proof(ch.merkle_proof, root, leaf_hash) is True


# ── (e) range / empty guards ────────────────────────────────────────────────────────

def test_assemble_empty_receipts_raises():
    with pytest.raises(ValueError):
        assemble_no_escrow_challenge(
            batch_id=b"\x11" * 32, batch_receipts=[], target_index=0)


def test_assemble_out_of_range_index_raises():
    a = generate_node_identity("a")
    batch = [_receipt(a, shard=0)]
    with pytest.raises(ValueError):
        assemble_no_escrow_challenge(
            batch_id=b"\x11" * 32, batch_receipts=batch, target_index=1)
    with pytest.raises(ValueError):
        assemble_no_escrow_challenge(
            batch_id=b"\x11" * 32, batch_receipts=batch, target_index=-1)


# ── (b) ONLY-UNAUTHORIZED ───────────────────────────────────────────────────────────

def _classification(batch_id, classification, *, provider="0x" + "2" * 40,
                    requester="0x" + "1" * 40, value=WEI):
    return BatchAuthorizationClassification(
        batch_id=batch_id, requester=requester, provider=provider,
        total_value_wei=value, classification=classification, match_basis="x")


def test_only_unauthorized_batches_are_assembled():
    a = generate_node_identity("a")
    auth_batch = [_receipt(a, job="authed", shard=0)]
    unauth_batch = [_receipt(a, job="unauthed", shard=0)]
    auth_id = b"\xaa" * 32
    unauth_id = b"\xbb" * 32

    classifications = [
        _classification(auth_id, AUTHORIZED),
        _classification(unauth_id, UNAUTHORIZED),
    ]
    receipts_by_batch = {auth_id: auth_batch, unauth_id: unauth_batch}

    results = assemble_no_escrow_challenges(classifications, receipts_by_batch)

    # exactly one challenge assembled, for the UNAUTHORIZED batch only
    assembled = [r for r in results if r.challenge is not None]
    assert len(assembled) == 1
    assert assembled[0].batch_id == unauth_id
    assert assembled[0].challenge.reason_code == 2
    # the AUTHORIZED batch is never assembled/disputed
    assert all(r.batch_id != auth_id or r.challenge is None for r in results)


# ── (c) submitter extension (requester-keyed) ───────────────────────────────────────

def test_submitter_accepts_no_escrow_reason_in_supported_set():
    assert REASON_NO_ESCROW in SUPPORTED_CHALLENGE_REASONS


def _no_escrow_challenge():
    a = generate_node_identity("a")
    batch = [_receipt(a, shard=0)]
    return assemble_no_escrow_challenge(
        batch_id=b"\x11" * 32, batch_receipts=batch, target_index=0)


def test_submit_forwards_no_escrow_args_and_succeeds():
    h = _Harness(tx_status=1)
    ch = _no_escrow_challenge()
    result = h.submitter().submit(ch)
    assert result.success is True
    assert h.call_args == ch.to_call_args()
    assert h.call_args[3] == 2  # NO_ESCROW
    assert h.call_args[4] == b""  # empty auxData


def test_dry_run_accepts_no_escrow():
    h = _Harness(call_ok=True)
    res = h.submitter().dry_run(_no_escrow_challenge())
    assert res.would_succeed is True
    assert h.built_overrides is None  # read-only — never built/broadcast


def test_submit_still_rejects_expired_reason():
    h = _Harness()
    ch = _no_escrow_challenge()
    expired = NoEscrowChallenge(
        batch_id=ch.batch_id, leaf=ch.leaf, merkle_proof=ch.merkle_proof,
        reason_code=3, aux_data=ch.aux_data)  # 3 = EXPIRED, unsupported
    result = h.submitter().submit(expired)
    assert result.success is False
    assert h.call_args is None  # never reached the contract
    # dry_run rejects it too
    assert h.submitter().dry_run(expired).would_succeed is False


# ── (d) NO-AUTO-SUBMIT + fail-closed ────────────────────────────────────────────────

def test_flow_never_submits_or_broadcasts():
    """An injected dry_run_client whose submit/broadcast are Mocks must NEVER be called;
    dry_run MAY be called (read-only)."""
    a = generate_node_identity("a")
    unauth_batch = [_receipt(a, job="unauthed", shard=0)]
    unauth_id = b"\xbb" * 32

    client = Mock()
    client.dry_run.return_value = type("DR", (), {"would_succeed": True,
                                                   "revert_reason": None})()

    classifications = [_classification(unauth_id, UNAUTHORIZED)]
    receipts_by_batch = {unauth_id: unauth_batch}

    results = assemble_no_escrow_challenges(
        classifications, receipts_by_batch, dry_run_client=client)

    client.dry_run.assert_called_once()
    client.submit.assert_not_called()
    client.broadcast.assert_not_called()
    assembled = [r for r in results if r.challenge is not None]
    assert len(assembled) == 1
    assert assembled[0].dry_run_would_succeed is True


def test_flow_fail_closed_on_missing_receipts():
    unauth_id = b"\xbb" * 32
    classifications = [_classification(unauth_id, UNAUTHORIZED)]
    # receipts missing for the UNAUTHORIZED batch
    results = assemble_no_escrow_challenges(classifications, {})
    assert len(results) == 1
    assert results[0].batch_id == unauth_id
    assert results[0].challenge is None
    assert results[0].error is not None


# ── (f) end-to-end with the sp1146 matcher ──────────────────────────────────────────

def test_end_to_end_matcher_to_no_escrow_assembly(tmp_path):
    requester_acct = Account.create()
    requester_addr = requester_acct.address
    honest_provider = "0x" + "a" * 40
    rogue_provider = "0x" + "b" * 40

    # R issues an auth ONLY to the honest provider.
    store = IssuedAuthorizationStore(tmp_path / "auths.json")
    store.record(_auth_payload(
        requester_key=requester_acct.key.hex(), provider=honest_provider,
        max_spend_ftns="5", expiry_unix=2_000_000_000))

    a = generate_node_identity("a")
    authed_batch = [_receipt(
        a, job="authed", shard=0, value=WEI,
        requester=requester_addr, provider=honest_provider)]
    rogue_batch = [_receipt(
        a, job="rogue", shard=0, value=WEI,
        requester=requester_addr, provider=rogue_provider)]
    authed_id = b"\xcc" * 32
    rogue_id = b"\xdd" * 32

    authed_view = CommittedBatchView(
        batch_id=authed_id, requester=requester_addr, provider=honest_provider,
        total_value_wei=WEI, commit_timestamp=1_000_000)
    rogue_view = CommittedBatchView(
        batch_id=rogue_id, requester=requester_addr, provider=rogue_provider,
        total_value_wei=WEI, commit_timestamp=1_000_000)

    classifications = match_unauthorized_batches(
        [authed_view, rogue_view], store, my_address=requester_addr)

    # matcher: authed -> AUTHORIZED, rogue -> UNAUTHORIZED
    by_id = {c.batch_id: c for c in classifications}
    assert by_id[authed_id].classification == AUTHORIZED
    assert by_id[rogue_id].classification == UNAUTHORIZED

    receipts_by_batch = {authed_id: authed_batch, rogue_id: rogue_batch}
    client = Mock()
    client.dry_run.return_value = type("DR", (), {"would_succeed": True,
                                                   "revert_reason": None})()

    results = assemble_no_escrow_challenges(
        classifications, receipts_by_batch, dry_run_client=client)

    assembled = [r for r in results if r.challenge is not None]
    # exactly the rogue (unauthorized) batch gets a NO_ESCROW challenge
    assert len(assembled) == 1
    assert assembled[0].batch_id == rogue_id
    assert assembled[0].challenge.reason_code == 2
    assert assembled[0].dry_run_would_succeed is True
    # never auto-submitted
    client.submit.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
