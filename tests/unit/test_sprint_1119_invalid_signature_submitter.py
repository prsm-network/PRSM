"""Sprint 1119 — challenge/dispute brick 4: the (inert-by-default) INVALID_SIGNATURE
challenge submitter.

dry_run() is a read-only pre-flight (no broadcast); submit() broadcasts the real
challengeReceipt tx (user-gated). Tested with injected fakes — no live chain.
"""
from __future__ import annotations

import pytest

from prsm.node.identity import generate_node_identity
from prsm.compute.shard_receipt import (
    ShardExecutionReceipt,
    build_receipt_signing_payload,
)
from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.challenge_assembler import (
    InvalidSignatureChallenge,
    assemble_invalid_signature_challenge,
)
from prsm.settlement.invalid_signature_submitter import (
    InvalidSignatureChallengeSubmitter,
)


# ── fakes ──────────────────────────────────────────────────────────────────────────

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
    """Holds the configurable fakes + the captured call args."""
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
        return InvalidSignatureChallengeSubmitter(
            web3=self.web3, registry=self.registry, account=self.account)


# ── a real assembled challenge to submit ──────────────────────────────────────────

def _fraudulent_challenge() -> InvalidSignatureChallenge:
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

    batch = [_rec(a, 0, True), _rec(b, 1, False)]  # index 1 fraudulent
    return assemble_invalid_signature_challenge(
        batch_id=b"\x11" * 32, batch_receipts=batch, target_index=1)


# ── tests ──────────────────────────────────────────────────────────────────────────

def test_submit_passes_assembled_args_and_succeeds():
    h = _Harness(tx_status=1)
    ch = _fraudulent_challenge()
    result = h.submitter().submit(ch)
    assert result.success is True
    assert result.tx_hash_hex == "0xabcd"
    # the contract was called with the challenge's exact assembled args
    assert h.call_args == ch.to_call_args()
    assert h.call_args[3] == 1  # reason code INVALID_SIGNATURE


def test_submit_classifies_reverted_tx_as_failure():
    h = _Harness(tx_status=0)  # receipt.status != 1 → OnChainRevertedError
    result = h.submitter().submit(_fraudulent_challenge())
    assert result.success is False
    assert result.error_type == "OnChainRevertedError"


def test_submit_classifies_broadcast_failure():
    h = _Harness(broadcast_raises=True)
    result = h.submitter().submit(_fraudulent_challenge())
    assert result.success is False
    assert result.error_type == "BroadcastFailedError"


def test_dry_run_would_succeed_when_static_call_ok():
    h = _Harness(call_ok=True)
    res = h.submitter().dry_run(_fraudulent_challenge())
    assert res.would_succeed is True
    # dry_run is read-only — it never built/broadcast a tx
    assert h.built_overrides is None


def test_dry_run_surfaces_revert_reason():
    h = _Harness(call_ok=False, call_error="ChallengeNotProven")
    res = h.submitter().dry_run(_fraudulent_challenge())
    assert res.would_succeed is False
    assert "ChallengeNotProven" in res.revert_reason


def test_submit_refuses_non_invalid_signature_reason():
    h = _Harness()
    ch = _fraudulent_challenge()
    wrong = InvalidSignatureChallenge(
        batch_id=ch.batch_id, leaf=ch.leaf, merkle_proof=ch.merkle_proof,
        reason_code=5, aux_data=ch.aux_data)  # 5 = CONSENSUS_MISMATCH
    result = h.submitter().submit(wrong)
    assert result.success is False
    assert h.call_args is None  # never reached the contract


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
