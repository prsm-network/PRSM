"""Sprint 1132 — challenge/dispute brick 8: bring DOUBLE_SPEND to full symmetry with
INVALID_SIGNATURE by reusing the SAME (generalized) on-chain submitter.

The brick-4 submitter was INVALID_SIGNATURE-only (its reason-guard rejected any reason
!= 1). Brick 8 generalizes it into a reason-agnostic ChallengeSubmitter that broadcasts
ANY assembled challenge via its reason-agnostic to_call_args() — accepting BOTH
INVALID_SIGNATURE (1) and DOUBLE_SPEND (0), while still uniformly REJECTING unsupported
reason codes (e.g. NO_ESCROW=2). InvalidSignatureChallengeSubmitter remains importable
as a backward-compat alias of the same class object.

Tested with injected fakes — no live chain (mirrors sp1119).
"""
from __future__ import annotations

import pytest

from prsm.node.identity import generate_node_identity
from prsm.compute.shard_receipt import (
    ShardExecutionReceipt,
    build_receipt_signing_payload,
)
from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.challenge_assembler import assemble_invalid_signature_challenge
from prsm.settlement.double_spend_assembler import (
    REASON_DOUBLE_SPEND,
    DoubleSpendChallenge,
)
from prsm.settlement.merkle import (
    batched_receipt_to_leaf,
    build_merkle_proof,
    hash_leaf,
)
from prsm.settlement.invalid_signature_submitter import (
    ChallengeSubmitter,
    InvalidSignatureChallengeSubmitter,
)


# ── fakes (identical injection shape to sp1119) ──────────────────────────────────────

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
        if self._parent.broadcast_raises:
            raise RuntimeError("rpc down")
        return b"\xab\xcd"

    def wait_for_transaction_receipt(self, h, timeout=120):
        return _FakeReceipt(self._parent.tx_status)


class _FakeWeb3:
    def __init__(self, parent):
        self.eth = _FakeEth(parent)


class _Harness:
    def __init__(self, *, call_ok=True, call_error="reverted", tx_status=1,
                 broadcast_raises=False):
        self.call_ok = call_ok
        self.call_error = call_error
        self.tx_status = tx_status
        self.broadcast_raises = broadcast_raises
        self.call_args = None
        self.built_overrides = None
        self.call_opts = None
        self.account = type("A", (), {"address": "0xchallenger", "key": b"k"})()
        self.web3 = _FakeWeb3(self)
        self.registry = _FakeRegistry(self)

    def submitter(self):
        return ChallengeSubmitter(
            web3=self.web3, registry=self.registry, account=self.account)


# ── a real DOUBLE_SPEND challenge to submit ─────────────────────────────────────────

def _double_spend_challenge(reason_code: int = REASON_DOUBLE_SPEND) -> DoubleSpendChallenge:
    """Build a DoubleSpendChallenge with a genuine ReceiptLeaf + merkle proof (so
    leaf_tuple() / the challengeReceipt ABI types line up). Built directly from receipt
    preimages (NOT via the INVALID_SIGNATURE assembler, whose fail-fast soundness check is
    INVALID_SIGNATURE-specific and irrelevant here). reason_code defaults to 0
    (DOUBLE_SPEND); a non-empty stand-in conflicting-batch auxData blob is attached."""
    idn = generate_node_identity("a")
    out = "ab" * 32
    payload = build_receipt_signing_payload(
        job_id="j", shard_index=0, output_hash=out, executed_at_unix=1000)
    rec = ShardExecutionReceipt(
        job_id="j", shard_index=0, provider_id=idn.node_id,
        provider_pubkey_b64=idn.public_key_b64, output_hash=out,
        executed_at_unix=1000, signature=idn.sign(payload))
    br = BatchedReceipt(receipt=rec, requester_address="0x" + "1" * 40,
                        provider_address="0x" + "2" * 40, value_ftns=10**18,
                        local_escrow_id="e")
    leaf = batched_receipt_to_leaf(br)
    proof = build_merkle_proof([hash_leaf(leaf)], 0)
    return DoubleSpendChallenge(
        batch_id=b"\x33" * 32,
        leaf=leaf,
        merkle_proof=proof,
        reason_code=reason_code,
        aux_data=b"\xaa" * 64,  # stand-in for abi.encode(conflictingBatchId, proof)
    )


# ── tests ────────────────────────────────────────────────────────────────────────────

def test_dry_run_double_spend_would_succeed_when_static_call_ok():
    """(a) RED against brick-4: the old reason-guard rejected reason 0. Generalized:
    a DOUBLE_SPEND dry_run whose static call succeeds → would_succeed=True, read-only."""
    h = _Harness(call_ok=True)
    res = h.submitter().dry_run(_double_spend_challenge())
    assert res.would_succeed is True
    assert h.built_overrides is None  # dry_run never builds/broadcasts a tx


def test_submit_double_spend_forwards_reason_zero_and_succeeds():
    """(b) submit(DoubleSpendChallenge) → success, and challengeReceipt was called with
    *challenge.to_call_args() — reason_code 0 forwarded verbatim."""
    h = _Harness(tx_status=1)
    ch = _double_spend_challenge()
    result = h.submitter().submit(ch)
    assert result.success is True
    assert result.tx_hash_hex == "0xabcd"
    assert h.call_args == ch.to_call_args()
    assert h.call_args[3] == REASON_DOUBLE_SPEND == 0  # DOUBLE_SPEND reason forwarded


def test_unsupported_reason_rejected_by_dry_run_and_submit():
    """(c) EXPIRED (reason 3) is NOT in the supported set → BOTH dry_run and submit
    reject it uniformly, never raise, never reach the contract / broadcast.

    (Sprint 1147 added NO_ESCROW=2 to the supported set — the requester self-dispute —
    so EXPIRED=3 is now the unsupported-reason exemplar. The rejection contract is
    unchanged: any reason outside SUPPORTED_CHALLENGE_REASONS is uniformly refused.)"""
    h = _Harness(tx_status=1)
    sub = h.submitter()
    bogus = _double_spend_challenge(reason_code=3)  # EXPIRED — unsupported

    dr = sub.dry_run(bogus)
    assert dr.would_succeed is False
    assert dr.revert_reason and "3" in dr.revert_reason
    assert h.call_args is None  # dry_run never reached the contract

    res = sub.submit(bogus)
    assert res.success is False
    assert res.error_type == "ValueError"
    assert h.call_args is None  # submit never reached the contract
    assert h.built_overrides is None  # never built/broadcast a tx


def test_invalid_signature_alias_is_same_class_object():
    """(d) backward-compat: InvalidSignatureChallengeSubmitter is still importable and is
    literally the same class object as the generalized ChallengeSubmitter."""
    assert InvalidSignatureChallengeSubmitter is ChallengeSubmitter


def test_invalid_signature_still_accepted_by_generalized_submitter():
    """Symmetry sanity: the generalized submitter still accepts INVALID_SIGNATURE (1)."""
    h = _Harness(tx_status=1)
    a, b = generate_node_identity("a"), generate_node_identity("b")

    def _rec(idn, shard, valid):
        out = "ab" * 32
        good = build_receipt_signing_payload(
            job_id="j", shard_index=shard, output_hash=out, executed_at_unix=1000)
        sig = idn.sign(good) if valid else idn.sign(build_receipt_signing_payload(
            job_id="j", shard_index=shard, output_hash="cd" * 32, executed_at_unix=1000))
        rec = ShardExecutionReceipt(
            job_id="j", shard_index=shard, provider_id=idn.node_id,
            provider_pubkey_b64=idn.public_key_b64, output_hash=out,
            executed_at_unix=1000, signature=sig)
        return BatchedReceipt(receipt=rec, requester_address="0x" + "1" * 40,
                              provider_address="0x" + "2" * 40, value_ftns=10**18,
                              local_escrow_id="e")

    batch = [_rec(a, 0, True), _rec(b, 1, False)]
    ch = assemble_invalid_signature_challenge(
        batch_id=b"\x11" * 32, batch_receipts=batch, target_index=1)
    result = h.submitter().submit(ch)
    assert result.success is True
    assert h.call_args[3] == 1  # INVALID_SIGNATURE reason forwarded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
