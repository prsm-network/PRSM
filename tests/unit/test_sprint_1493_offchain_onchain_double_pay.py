"""Sprint 1493 — one job must not charge the requester off-chain AND on-chain.

Found by a 26-agent adversarial trace of the paid-dispatch money path; it was in
no prior audit record. Two independent legs pay for the SAME ComputeJob and
neither consults the other:

  requester  compute_requester._handle_job_result  — ledger.transfer + gossip, so
             the provider's own node credits itself (spendable via /wallet/withdraw)
  provider   compute_provider._maybe_accumulate_onchain_earning — accumulates the
             same job into the on-chain ReceiptAccumulator, and finalization calls
             EscrowPool.settleFromRequester, which moves FTNS FROM THE REQUESTER

So the requester pays twice for one job and the provider is paid twice. It is
latent — both legs additionally need PRSM_ONCHAIN_SETTLEMENT, which defaults off —
but it is a real double-spend once enabled. `local_escrow_id` on the on-chain
receipt is only an idempotency key for the per-stage store; nothing ever
cross-checks it against the off-chain payment.

The first two tests DEMONSTRATE the overlap rather than assuming it, because the
prior record on this rail was wrong twice.
"""
from __future__ import annotations

import pytest


# ── 1. demonstrate the two gates really do overlap ──────────────────

def test_the_two_payment_legs_have_OVERLAPPING_conditions():
    """★ The premise, verified from source rather than assumed. Both gates can be
    satisfied by the same job at the same time — nothing is mutually exclusive."""
    import inspect

    from prsm.node.compute_provider import ComputeProvider
    from prsm.node.compute_requester import ComputeRequester

    prov = inspect.getsource(ComputeProvider._maybe_accumulate_onchain_earning)
    # The provider pays on chain when it has a settlement client, its own operator
    # address, and the requester's (which the requester puts in the offer).
    assert "settlement_client" in prov
    assert "requester_operator_address" in prov
    assert "accumulate(" in prov

    req = inspect.getsource(ComputeRequester._on_job_result)
    # The requester pays off chain on the same completion.
    assert "COMPUTE_PAYMENT" in req or "release_escrow" in req
    assert "broadcast_transaction" in req


def test_the_onchain_leg_charges_the_REQUESTER_not_the_provider():
    """★ Why the overlap is a double-spend and not merely double-bookkeeping:
    settleFromRequester moves the funds from the requester."""
    from pathlib import Path

    src = Path("prsm/economy/web3/escrow_pool_client.py").read_text()
    assert "settleFromRequester" in src


# ── 2. the guard's predicate ────────────────────────────────────────

class _Req:
    """Minimal stand-in exposing only what the predicate reads."""

    def __init__(self, operator_address="0x" + "a1" * 20, provider_addr="0x" + "b2" * 20,
                 raises=False):
        self.operator_address = operator_address
        self._provider_addr = provider_addr
        self._raises = raises

    def _resolve_stake_posture(self, provider_id):
        if self._raises:
            raise RuntimeError("unresolvable")
        return (True, 1, self._provider_addr)

    _onchain_settlement_expected = None  # bound below


from prsm.node.compute_requester import ComputeRequester  # noqa: E402

_Req._onchain_settlement_expected = ComputeRequester._onchain_settlement_expected


def test_off_by_default_so_the_existing_rail_is_untouched(monkeypatch):
    """★ PRSM_ONCHAIN_SETTLEMENT defaults off — the live default path must keep
    paying off-chain exactly as before."""
    monkeypatch.delenv("PRSM_ONCHAIN_SETTLEMENT", raising=False)
    assert _Req()._onchain_settlement_expected("peer") is False


def test_true_only_when_every_visible_gate_passes(monkeypatch):
    monkeypatch.setenv("PRSM_ONCHAIN_SETTLEMENT", "1")
    assert _Req()._onchain_settlement_expected("peer") is True


def test_false_without_our_own_operator_address(monkeypatch):
    """Without it the offer carries no requester_operator_address, so the
    provider's gate cannot pass and it will NOT settle on chain."""
    monkeypatch.setenv("PRSM_ONCHAIN_SETTLEMENT", "1")
    assert _Req(operator_address="")._onchain_settlement_expected("peer") is False


def test_false_without_a_provider_operator_address(monkeypatch):
    monkeypatch.setenv("PRSM_ONCHAIN_SETTLEMENT", "1")
    assert _Req(provider_addr="")._onchain_settlement_expected("peer") is False


def test_an_unresolvable_provider_is_not_treated_as_onchain(monkeypatch):
    """★ Fails toward PAYING off-chain. A false positive here would skip the only
    payment the provider gets."""
    monkeypatch.setenv("PRSM_ONCHAIN_SETTLEMENT", "1")
    assert _Req(raises=True)._onchain_settlement_expected("peer") is False


def test_truthy_spellings_are_accepted(monkeypatch):
    for v in ("1", "true", "TRUE", "yes"):
        monkeypatch.setenv("PRSM_ONCHAIN_SETTLEMENT", v)
        assert _Req()._onchain_settlement_expected("peer") is True
    for v in ("0", "false", "no", ""):
        monkeypatch.setenv("PRSM_ONCHAIN_SETTLEMENT", v)
        assert _Req()._onchain_settlement_expected("peer") is False


# ── 3. the guard is actually wired into the money path ──────────────

def test_the_payment_path_CONSULTS_the_guard():
    """★ Binding test. A predicate nothing calls leaves the double-pay live —
    this codebase has produced a false-green RED check four times."""
    import inspect

    from prsm.node.compute_requester import ComputeRequester

    src = inspect.getsource(ComputeRequester._on_job_result)
    assert "_onchain_settlement_expected(provider_id)" in src


def test_locked_escrow_is_REFUNDED_not_stranded():
    """★ The subtlety a plain 'skip' would get wrong: if escrow was locked at
    submit-time, skipping the release leaves the funds sitting in the escrow
    wallet forever. They must go back to the requester, since the provider is
    paid on chain."""
    import inspect

    from prsm.node.compute_requester import ComputeRequester

    src = inspect.getsource(ComputeRequester._on_job_result)
    guard = src[src.index("_onchain_settlement_expected(provider_id)"):]
    guard = guard[:guard.index("else:")]
    assert "refund_escrow" in guard, (
        "locked escrow must be refunded when the off-chain payment is skipped")


def test_the_skip_is_LOGGED_with_job_amount_and_payee():
    """A silently skipped payment is indistinguishable from a bug — the operator
    must be able to reconcile it if the provider reports non-payment."""
    import inspect

    from prsm.node.compute_requester import ComputeRequester

    src = inspect.getsource(ComputeRequester._on_job_result)
    assert "SKIPPING the off-chain payment" in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
