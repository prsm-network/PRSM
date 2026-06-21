"""Sprint 1200 — front-door e2e validation harness (the single-command proof).

prsm/node/frontdoor_e2e.py orchestrates faucet -> deposit -> pay-infer and asserts
each step against authoritative reads. These tests exercise the orchestration with
fully mocked clients (no chain), covering the mainnet kill-switch, each step's pass
+ fail paths, and the full run_all aggregation.
"""
from __future__ import annotations

import pytest

from prsm.node.frontdoor_e2e import (
    BASE_MAINNET_CHAIN_ID,
    BASE_SEPOLIA_CHAIN_ID,
    FrontDoorValidationError,
    assert_testnet_only,
    run_all,
    run_deposit_step,
    run_faucet_step,
    run_payinfer_step,
)

_WEI = 10 ** 18
_REQ = "0x" + "11" * 20


def _signed_receipt(sig=b"sig"):
    """A real InferenceReceipt dict (round-trips through from_dict) with a settler
    signature — what the harness now requires to count as a 'signed receipt' (review #2)."""
    from decimal import Decimal
    from prsm.compute.inference.models import (
        InferenceReceipt, ContentTier, PrivacyLevel, TEEType,
    )
    return InferenceReceipt(
        job_id="j1", request_id="r1", model_id="gpt2",
        content_tier=ContentTier.A, privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0, tee_type=TEEType.NONE, tee_attestation=b"",
        output_hash=b"\x01" * 32, duration_seconds=0.1, cost_ftns=Decimal("0.5"),
        settler_signature=sig, settler_node_id="node1").to_dict()


# ── fakes ────────────────────────────────────────────────────────────────────────────

class _SeqAsync:
    """async balance_of(addr) returning a sequence (last value repeats)."""
    def __init__(self, values):
        self._v = list(values)
        self._i = 0

    async def balance_of(self, addr):
        v = self._v[min(self._i, len(self._v) - 1)]
        self._i += 1
        return v


class _SeqSync:
    """sync balance_wei(addr) returning a sequence (last value repeats)."""
    def __init__(self, values, decimals=18):
        self._v = list(values)
        self._i = 0
        self.decimals = decimals

    def balance_wei(self, addr):
        v = self._v[min(self._i, len(self._v) - 1)]
        self._i += 1
        return v


class _Client:
    def __init__(self, *, deposit_tx="0xdep", pay_result=None):
        self._deposit_tx = deposit_tx
        self._pay_result = pay_result or {}
        self.closed = False

    async def deposit_escrow(self, **kw):
        return self._deposit_tx

    async def pay_and_infer(self, prompt, *, requester_key, verify_pubkey_b64=None, **kw):
        return self._pay_result

    async def close(self):
        self.closed = True


# ── mainnet kill-switch ───────────────────────────────────────────────────────────────

def test_assert_testnet_only_allows_sepolia():
    assert_testnet_only(chain_id=BASE_SEPOLIA_CHAIN_ID)  # no raise


def test_assert_testnet_only_refuses_mainnet():
    with pytest.raises(FrontDoorValidationError):
        assert_testnet_only(chain_id=BASE_MAINNET_CHAIN_ID)


def test_assert_testnet_only_refuses_eth_mainnet():
    with pytest.raises(FrontDoorValidationError):
        assert_testnet_only(chain_id=1)


# ── faucet step ───────────────────────────────────────────────────────────────────────

async def test_faucet_step_passes_when_balance_rises():
    reader = _SeqSync([0, 100 * _WEI])  # before 0, after 100
    posts = []

    def faucet_post(body):
        posts.append(body)
        return {"status": "dispensed", "tx_hash": "0xfaucet", "dispensed_ftns": 100}

    out = await run_faucet_step(faucet_post=faucet_post, balance_reader=reader,
                                recipient=_REQ, confirm_attempts=1)
    assert out["ok"] is True
    assert out["tx_hash"] == "0xfaucet"
    assert posts[0]["destination_address"] == _REQ


async def test_faucet_step_raises_when_no_tx():
    reader = _SeqSync([0, 0])

    def faucet_post(body):
        return {"detail": "faucet unavailable"}

    with pytest.raises(FrontDoorValidationError):
        await run_faucet_step(faucet_post=faucet_post, balance_reader=reader,
                              recipient=_REQ, confirm_attempts=1)


async def test_faucet_step_passes_amount_through():
    reader = _SeqSync([0, 50 * _WEI])
    seen = {}

    def faucet_post(body):
        seen.update(body)
        return {"tx_hash": "0x1", "dispensed_ftns": 50}

    await run_faucet_step(faucet_post=faucet_post, balance_reader=reader,
                          recipient=_REQ, amount_ftns=50, confirm_attempts=1)
    assert seen["amount"] == 50


# ── deposit step ──────────────────────────────────────────────────────────────────────

async def test_deposit_step_passes_when_escrow_funded():
    escrow = _SeqAsync([0, 5 * _WEI])  # before 0, after 5
    client = _Client(deposit_tx="0xdeposit")
    out = await run_deposit_step(client=client, requester_key="k", requester_address=_REQ,
                                 amount_ftns=5, escrow_client=escrow, confirm_attempts=1)
    assert out["ok"] is True
    assert out["deposit_tx"] == "0xdeposit"
    assert out["escrow_delta_wei"] == 5 * _WEI


async def test_deposit_step_not_ok_when_escrow_unchanged():
    escrow = _SeqAsync([0, 0])  # never credited (RPC lag / failure)
    client = _Client()
    out = await run_deposit_step(client=client, requester_key="k", requester_address=_REQ,
                                 amount_ftns=5, escrow_client=escrow, confirm_attempts=1)
    assert out["ok"] is False  # surfaced, not raised


async def test_deposit_step_raises_on_no_tx():
    escrow = _SeqAsync([0, 0])
    client = _Client(deposit_tx="")
    with pytest.raises(FrontDoorValidationError):
        await run_deposit_step(client=client, requester_key="k", requester_address=_REQ,
                               amount_ftns=5, escrow_client=escrow, confirm_attempts=1)


# ── pay-infer step ────────────────────────────────────────────────────────────────────

async def test_payinfer_step_passes_with_output_and_signed_receipt():
    escrow = _SeqAsync([5 * _WEI])  # unchanged (settle is later)
    client = _Client(pay_result={
        "success": True, "output": " world", "ftns_charged": 0.5,
        "receipt": _signed_receipt()})
    out = await run_payinfer_step(client=client, prompt="hi", requester_key="k",
                                  escrow_client=escrow, requester_address=_REQ)
    assert out["ok"] is True
    assert out["output"] == " world"
    assert out["has_receipt"] is True


async def test_payinfer_step_false_pass_guard_unparseable_receipt():
    # review #2: an output + a decorative non-receipt dict must NOT pass.
    escrow = _SeqAsync([5 * _WEI])
    client = _Client(pay_result={"success": True, "output": "x", "receipt": {"some": "junk"}})
    out = await run_payinfer_step(client=client, prompt="hi", requester_key="k",
                                  escrow_client=escrow, requester_address=_REQ)
    assert out["ok"] is False  # unparseable receipt → not a real proof


async def test_payinfer_step_surfaces_charge_from_receipt():
    # sp1203: the response has no ftns_charged; the cost is in the receipt's cost_ftns.
    escrow = _SeqAsync([5 * _WEI])
    rcpt = _signed_receipt()  # InferenceReceipt.to_dict() includes cost_ftns
    client = _Client(pay_result={"success": True, "output": "x", "receipt": rcpt})
    out = await run_payinfer_step(client=client, prompt="hi", requester_key="k",
                                  escrow_client=escrow, requester_address=_REQ)
    assert out["ok"] is True
    assert out["ftns_charged"] == rcpt["cost_ftns"]  # surfaced, not None


async def test_payinfer_step_unsigned_receipt_not_ok():
    # A parseable receipt with NO settler signature is not a settled receipt.
    escrow = _SeqAsync([5 * _WEI])
    client = _Client(pay_result={"output": "x", "receipt": _signed_receipt(sig=b"")})
    out = await run_payinfer_step(client=client, prompt="hi", requester_key="k",
                                  escrow_client=escrow, requester_address=_REQ)
    assert out["ok"] is False


async def test_payinfer_step_raises_on_402_rejection():
    escrow = _SeqAsync([5 * _WEI])
    client = _Client(pay_result={"detail": "payment authorization rejected: expired"})
    with pytest.raises(FrontDoorValidationError, match="payment authorization REJECTED"):
        await run_payinfer_step(client=client, prompt="hi", requester_key="k",
                                escrow_client=escrow, requester_address=_REQ)


async def test_payinfer_step_inference_error_is_not_called_a_payment_rejection():
    # sp1202: payment ACCEPTED (no 402) but the inference failed (e.g. unknown model)
    # — the harness must NOT mislabel it as a payment rejection.
    escrow = _SeqAsync([5 * _WEI])
    client = _Client(pay_result={
        "success": False, "error": "Unknown model_id: gpt2", "job_id": "infer-1"})
    with pytest.raises(FrontDoorValidationError, match="inference FAILED"):
        await run_payinfer_step(client=client, prompt="hi", requester_key="k",
                                escrow_client=escrow, requester_address=_REQ)


async def test_payinfer_step_requires_verified_receipt_when_pubkey_given():
    escrow = _SeqAsync([5 * _WEI])
    # a properly-signed receipt that the server says FAILED signature verification
    client = _Client(pay_result={
        "output": "x", "receipt": _signed_receipt(), "receipt_verified": False})
    out = await run_payinfer_step(client=client, prompt="hi", requester_key="k",
                                  escrow_client=escrow, requester_address=_REQ,
                                  verify_pubkey_b64="somekey")
    assert out["ok"] is False  # parsed+signed but signature didn't verify


async def test_payinfer_step_missing_receipt_not_ok():
    escrow = _SeqAsync([5 * _WEI])
    client = _Client(pay_result={"output": "x"})  # no receipt
    out = await run_payinfer_step(client=client, prompt="hi", requester_key="k",
                                  escrow_client=escrow, requester_address=_REQ)
    assert out["ok"] is False


# ── run_all aggregation ───────────────────────────────────────────────────────────────

async def test_run_all_deposit_then_payinfer_success():
    escrow = _SeqAsync([0, 5 * _WEI, 5 * _WEI, 5 * _WEI])
    client = _Client(deposit_tx="0xd", pay_result={
        "success": True, "output": "ok", "receipt": _signed_receipt(), "ftns_charged": 1})
    report = await run_all(
        chain_id=BASE_SEPOLIA_CHAIN_ID, client=client, escrow_client=escrow,
        balance_reader=_SeqSync([0]), requester_key="k", requester_address=_REQ,
        prompt="hi", deposit_amount_ftns=5, confirm_attempts=1)
    assert report["success"] is True
    assert [s["step"] for s in report["steps"]] == ["deposit", "pay-infer"]


async def test_run_all_includes_faucet_when_recipient_given():
    escrow = _SeqAsync([0, 5 * _WEI, 5 * _WEI])
    client = _Client(pay_result={"output": "ok", "receipt": _signed_receipt()})

    def faucet_post(body):
        return {"tx_hash": "0xf", "dispensed_ftns": 100}

    report = await run_all(
        chain_id=BASE_SEPOLIA_CHAIN_ID, client=client, escrow_client=escrow,
        balance_reader=_SeqSync([0, 100 * _WEI]), requester_key="k",
        requester_address=_REQ, prompt="hi", deposit_amount_ftns=5,
        faucet_post=faucet_post, faucet_recipient="0x" + "22" * 20, confirm_attempts=1)
    assert [s["step"] for s in report["steps"]] == ["faucet", "deposit", "pay-infer"]
    assert report["success"] is True


async def test_run_all_refuses_mainnet():
    with pytest.raises(FrontDoorValidationError):
        await run_all(
            chain_id=BASE_MAINNET_CHAIN_ID, client=_Client(), escrow_client=_SeqAsync([0]),
            balance_reader=_SeqSync([0]), requester_key="k", requester_address=_REQ,
            prompt="hi", confirm_attempts=1)


# ── EscrowPoolClient chain gate (review #1, defense-in-depth) ─────────────────────────

def test_escrow_client_refuses_wrong_chain():
    from prsm.economy.web3.escrow_pool_client import EscrowPoolClient
    c = EscrowPoolClient.__new__(EscrowPoolClient)
    c._expected_chain_id = BASE_SEPOLIA_CHAIN_ID

    class _Eth:
        chain_id = BASE_MAINNET_CHAIN_ID  # signer on the WRONG chain

    class _W3:
        eth = _Eth()
    c.web3 = _W3()
    with pytest.raises(RuntimeError, match="refuses to sign"):
        c._assert_chain()


def test_escrow_client_allows_matching_chain_and_none():
    from prsm.economy.web3.escrow_pool_client import EscrowPoolClient
    c = EscrowPoolClient.__new__(EscrowPoolClient)

    class _Eth:
        chain_id = BASE_SEPOLIA_CHAIN_ID

    class _W3:
        eth = _Eth()
    c.web3 = _W3()
    c._expected_chain_id = BASE_SEPOLIA_CHAIN_ID
    c._assert_chain()  # matching → no raise
    c._expected_chain_id = None
    c._assert_chain()  # unpinned → no gate, no raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
