"""Sprint 1161 — offline TDD for HARDENING the per-stage Base-Sepolia bench
(scripts/per_stage_sepolia_bench.py) against the SHARED-ESCROW under-funding bug
that bit the operator.

THE BUG (live run): phase-1 printed "escrow already funded (1e18 >= 1e18); skipping
deposit" and committed 2 per-node share-batches (0.5 FTNS each, total 1 FTNS). BUT
the requester A already had a PENDING UNRELATED two-party batch (1 FTNS) claiming
the SAME escrow. EscrowPool balance is a SHARED POOL per requester — so escrow=1
FTNS was SHORT of the 2 FTNS of total pending claims; the per-stage finalize would
have REVERTED on insufficient escrow. The deposit-skip wrongly assumed the escrow
was DEDICATED to the per-stage run.

THE FIX (3 parts, all offline-tested here, mocked chain — NO real network/keys):
  (a) NO SILENT UNDER-FUND: by DEFAULT phase-1 deposits the per-stage total
      ADDITIVELY even when escrow already holds >= the amount (it funds ITS OWN
      claims regardless of a pre-existing balance that may back OTHER batches),
      and emits a loud shared-escrow WARNING. --assume-dedicated-escrow restores
      the old skip-if-sufficient behavior.
  (b) VERIFY SUFFICIENCY: --phase verify reports escrow_balance vs the bench-known
      per-stage claims (SUFFICIENT/SHORT) + an honest NOTE that EXTERNAL pending
      batches also draw on the same shared escrow and are NOT counted here.
  (c) DEPOSIT-TOPUP: --phase deposit-topup --amount-ftns N is a clean STANDALONE
      additive deposit by the requester; key-gated, echoes the requester ADDRESS +
      amount + resulting balance, never the key.

PRESERVED (regression): the mainnet hard-guard, key-gating, and no-broadcast in the
read-only verify are unchanged.
"""
from __future__ import annotations

import asyncio
import hashlib
from unittest.mock import AsyncMock

import pytest
from eth_account import Account
from web3 import Web3

import scripts.per_stage_sepolia_bench as mod
from scripts.per_stage_sepolia_bench import build_arg_parser
from prsm.settlement.e2e_proof import (
    BASE_MAINNET_CHAIN_ID,
    BASE_SEPOLIA_CHAIN_ID,
    assert_chain_allowed,
)

ONE_FTNS = 10 ** 18

# Hardhat well-known PUBLIC zero-value keys (OFFLINE test only; never funded).
_REQUESTER_KEY = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
_NODE_KEYS = [
    "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",
    "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6",
]
_REQUESTER_ADDR = Web3.to_checksum_address(Account.from_key(_REQUESTER_KEY).address)


class _EP:
    network_name = "testnet"
    rpc_url = "https://sepolia.base.org"
    escrow_pool = "0xaa28b5818242608e04C1773c3e34bF7bFfb96248"
    ftns_token = "0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a"
    settlement_registry = "0xF8BEEb4362222b50109b6034767322B31aA92449"


def _fake_commit_factory():
    async def _fake_commit(*, shares, client_for_node, committer_address_for_node, **_):
        from prsm.settlement.client import CommittedBatch
        from prsm.settlement.accumulator import TriggerReason
        out = {}
        for s in shares:
            out[s.node_id] = CommittedBatch(
                batch_id=hashlib.sha256(s.node_id.encode()).digest(),
                tx_hash="0xc", provider_address=committer_address_for_node(s.node_id),
                requester_address=_REQUESTER_ADDR, merkle_root=b"\x00" * 32,
                receipt_count=1, total_value_ftns=s.share_wei,
                commit_timestamp=1_700_000_000, leaf_hashes=(b"\x01" * 32,),
                trigger_reason=TriggerReason.VALUE,
            )
        return out
    return _fake_commit


def _patch_phase1_chain(monkeypatch, escrow):
    """Patch the chain-touching symbols _run_phase1 imports so no RPC is touched."""
    monkeypatch.setattr(
        mod, "_build_node_clients",
        lambda **kw: {nid: AsyncMock() for nid in kw["assembly"].node_ids},
    )
    import prsm.economy.web3.escrow_pool_client as escrow_mod
    monkeypatch.setattr(escrow_mod, "EscrowPoolClient", lambda **kw: escrow)
    import prsm.settlement.per_stage_commit as psc_mod
    monkeypatch.setattr(psc_mod, "commit_per_node_share_batches", _fake_commit_factory())


# ── (a) NO SILENT UNDER-FUND ────────────────────────────────────────────────────


def test_phase1_default_deposits_additively_when_escrow_already_holds_amount(
        monkeypatch, capsys, tmp_path):
    """The BUG fix: with escrow ALREADY holding the per-stage --amount-ftns
    (simulating a shared/earmarked balance backing OTHER batches), the DEFAULT
    phase-1 STILL deposits the per-stage total additively + emits the WARNING."""
    total = 1 * ONE_FTNS  # per-stage amount
    pre_existing = 1 * ONE_FTNS  # escrow already at the amount (would have skipped!)

    escrow = AsyncMock()
    # balance_of: first the pre-deposit read, then post-deposit reads reflect +total.
    escrow.balance_of.side_effect = [pre_existing, pre_existing + total,
                                     pre_existing + total, pre_existing + total]
    escrow.deposit.return_value = "0xdeposit"
    _patch_phase1_chain(monkeypatch, escrow)

    monkeypatch.setenv("PRSM_REQUESTER_KEY", _REQUESTER_KEY)
    monkeypatch.setenv("PRSM_NODE_KEYS", ",".join(_NODE_KEYS))
    args = build_arg_parser().parse_args(
        ["--phase", "deposit-commit", "--amount-ftns", "1",
         "--state-file", str(tmp_path / "state.json")])

    rc = asyncio.run(mod._run_phase1(args, _EP(), BASE_SEPOLIA_CHAIN_ID))
    out = capsys.readouterr().out
    assert rc == 0
    # The default NO LONGER skips: deposit WAS called with the per-stage total.
    escrow.deposit.assert_awaited_once_with(total)
    # Loud shared-escrow WARNING surfaced (the operator must know).
    assert "WARNING" in out
    assert "additively" in out.lower()
    assert "--assume-dedicated-escrow" in out
    assert "skipping deposit" not in out.lower()


def test_phase1_assume_dedicated_escrow_restores_skip(monkeypatch, capsys, tmp_path):
    """--assume-dedicated-escrow opts back into the old skip-if-sufficient behavior
    (only safe for a KNOWN-dedicated escrow)."""
    total = 1 * ONE_FTNS
    escrow = AsyncMock()
    escrow.balance_of.return_value = total  # already >= amount
    escrow.deposit.return_value = "0xdeposit"
    _patch_phase1_chain(monkeypatch, escrow)

    monkeypatch.setenv("PRSM_REQUESTER_KEY", _REQUESTER_KEY)
    monkeypatch.setenv("PRSM_NODE_KEYS", ",".join(_NODE_KEYS))
    args = build_arg_parser().parse_args(
        ["--phase", "deposit-commit", "--amount-ftns", "1",
         "--assume-dedicated-escrow",
         "--state-file", str(tmp_path / "state.json")])

    rc = asyncio.run(mod._run_phase1(args, _EP(), BASE_SEPOLIA_CHAIN_ID))
    out = capsys.readouterr().out
    assert rc == 0
    escrow.deposit.assert_not_awaited()  # skipped (old behavior, opt-in)
    assert "skipping deposit" in out.lower()


def test_phase1_default_deposits_when_escrow_empty(monkeypatch, capsys, tmp_path):
    """Sanity: with an EMPTY escrow the default still deposits the per-stage total
    (and no shared-escrow warning, since there is no pre-existing balance)."""
    total = 1 * ONE_FTNS
    escrow = AsyncMock()
    escrow.balance_of.side_effect = [0, total, total, total]
    escrow.deposit.return_value = "0xdeposit"
    _patch_phase1_chain(monkeypatch, escrow)

    monkeypatch.setenv("PRSM_REQUESTER_KEY", _REQUESTER_KEY)
    monkeypatch.setenv("PRSM_NODE_KEYS", ",".join(_NODE_KEYS))
    args = build_arg_parser().parse_args(
        ["--phase", "deposit-commit", "--amount-ftns", "1",
         "--state-file", str(tmp_path / "state.json")])

    rc = asyncio.run(mod._run_phase1(args, _EP(), BASE_SEPOLIA_CHAIN_ID))
    out = capsys.readouterr().out
    assert rc == 0
    escrow.deposit.assert_awaited_once_with(total)
    # No pre-existing balance => no "may back OTHER pending batches" warning.
    assert "may back OTHER" not in out


# ── (b) VERIFY SUFFICIENCY ──────────────────────────────────────────────────────


def _verify_state(tmp_path):
    """A phase-1 state file with two committed bench batches summing to 1 FTNS."""
    import json
    n0 = Web3.to_checksum_address(Account.from_key(_NODE_KEYS[0]).address)
    n1 = Web3.to_checksum_address(Account.from_key(_NODE_KEYS[1]).address)
    half = ONE_FTNS // 2
    state = {
        "network": "testnet", "chain_id": BASE_SEPOLIA_CHAIN_ID,
        "request_id": "req-sp1161-verify", "total_wei": ONE_FTNS,
        "requester_address": _REQUESTER_ADDR,
        "registry": _EP.settlement_registry,
        "expected_payout_by_eth": {n0: half, n1: half},
        "committed": {
            "node-a": {"batch_id": (b"\x11" * 32).hex(),
                       "committer_address": n0, "share_wei": half},
            "node-b": {"batch_id": (b"\x22" * 32).hex(),
                       "committer_address": n1, "share_wei": half},
        },
    }
    p = tmp_path / "state.json"
    p.write_text(json.dumps(state))
    return str(p)


def _patch_verify_chain(monkeypatch, *, escrow_balance, pending_by_batch):
    """Mock the escrow + contract clients the verify path imports. pending_by_batch
    maps batch_id hex -> on-chain PENDING total_value_ftns."""
    escrow = AsyncMock()
    escrow.balance_of.return_value = escrow_balance
    escrow.ftns_balance_of.return_value = 0

    contract = AsyncMock()
    contract.get_batch_status.return_value = 1  # PENDING

    async def _get_batch(batch_id):
        return {"total_value_ftns": pending_by_batch[batch_id.hex()], "status": 1}
    contract.get_batch.side_effect = _get_batch
    contract.is_finalizable.return_value = False

    import prsm.economy.web3.escrow_pool_client as escrow_mod
    monkeypatch.setattr(escrow_mod, "EscrowPoolClient", lambda **kw: escrow)
    import prsm.economy.web3.batch_settlement_contract_client as bs_mod
    monkeypatch.setattr(bs_mod, "Web3SettlementContractClient", lambda **kw: contract)


def test_verify_reports_short_when_escrow_below_bench_claims(
        monkeypatch, capsys, tmp_path):
    """The diagnostic that would have caught the bug: escrow (1 FTNS) < bench-known
    claims (1 FTNS bench + an EXTERNAL claim means the operator is short) — here we
    drive the bench-known-claims path: escrow 1 FTNS but bench claims sum 1 FTNS is
    SUFFICIENT, so to show SHORT we set escrow BELOW the bench claims."""
    half = ONE_FTNS // 2
    state_file = _verify_state(tmp_path)
    # escrow holds only 0.5 FTNS but bench-known claims total 1.0 FTNS -> SHORT.
    _patch_verify_chain(
        monkeypatch, escrow_balance=half,
        pending_by_batch={(b"\x11" * 32).hex(): half, (b"\x22" * 32).hex(): half})

    args = build_arg_parser().parse_args(
        ["--phase", "verify", "--state-file", state_file])
    rc = asyncio.run(mod._run_verify(args, _EP(), BASE_SEPOLIA_CHAIN_ID))
    out = capsys.readouterr().out
    assert rc == 0
    assert "SHORT" in out
    assert "SUFFICIENT" not in out.split("SHORT")[0][-40:]  # SHORT is the verdict
    # The honest external-batches NOTE is always printed.
    assert "EXTERNAL" in out.upper()
    assert "NOT counted" in out or "not counted" in out.lower()


def test_verify_reports_sufficient_when_escrow_covers_bench_claims(
        monkeypatch, capsys, tmp_path):
    half = ONE_FTNS // 2
    state_file = _verify_state(tmp_path)
    # escrow holds 2 FTNS, bench-known claims total 1 FTNS -> SUFFICIENT.
    _patch_verify_chain(
        monkeypatch, escrow_balance=2 * ONE_FTNS,
        pending_by_batch={(b"\x11" * 32).hex(): half, (b"\x22" * 32).hex(): half})

    args = build_arg_parser().parse_args(
        ["--phase", "verify", "--state-file", state_file])
    rc = asyncio.run(mod._run_verify(args, _EP(), BASE_SEPOLIA_CHAIN_ID))
    out = capsys.readouterr().out
    assert rc == 0
    assert "SUFFICIENT" in out
    # The external-batches NOTE is still printed (always honest about shared escrow).
    assert "EXTERNAL" in out.upper()


def test_verify_broadcasts_nothing(monkeypatch, tmp_path):
    """Read-only verify must construct clients with NO private key (no broadcast)."""
    captured = {}

    class _RecordingEscrow(AsyncMock):
        pass

    def _escrow_factory(**kw):
        captured["escrow_key"] = kw.get("private_key", "MISSING")
        e = AsyncMock()
        e.balance_of.return_value = 2 * ONE_FTNS
        e.ftns_balance_of.return_value = 0
        return e

    def _contract_factory(**kw):
        captured["contract_key"] = kw.get("private_key", "MISSING")
        c = AsyncMock()
        c.get_batch_status.return_value = 1
        c.is_finalizable.return_value = False

        async def _gb(batch_id):
            return {"total_value_ftns": ONE_FTNS // 2, "status": 1}
        c.get_batch.side_effect = _gb
        return c

    import prsm.economy.web3.escrow_pool_client as escrow_mod
    monkeypatch.setattr(escrow_mod, "EscrowPoolClient", _escrow_factory)
    import prsm.economy.web3.batch_settlement_contract_client as bs_mod
    monkeypatch.setattr(bs_mod, "Web3SettlementContractClient", _contract_factory)

    state_file = _verify_state(tmp_path)
    args = build_arg_parser().parse_args(
        ["--phase", "verify", "--state-file", state_file])
    asyncio.run(mod._run_verify(args, _EP(), BASE_SEPOLIA_CHAIN_ID))
    assert captured["escrow_key"] is None
    assert captured["contract_key"] is None


# ── (c) DEPOSIT-TOPUP ───────────────────────────────────────────────────────────


def test_deposit_topup_calls_deposit_once_and_echoes_address_not_key(
        monkeypatch, capsys):
    """--phase deposit-topup --amount-ftns N: standalone additive deposit; calls
    deposit(N*1e18) ONCE, echoes the requester ADDRESS + amount + resulting balance,
    never the key."""
    topup = 1 * ONE_FTNS
    escrow = AsyncMock()
    # before-deposit read = 3 FTNS; post-deposit reads reflect the +1 FTNS top-up.
    escrow.balance_of.side_effect = [3 * ONE_FTNS] + [4 * ONE_FTNS] * 6
    escrow.deposit.return_value = "0xtopup"
    import prsm.economy.web3.escrow_pool_client as escrow_mod
    monkeypatch.setattr(escrow_mod, "EscrowPoolClient", lambda **kw: escrow)

    monkeypatch.setenv("PRSM_REQUESTER_KEY", _REQUESTER_KEY)
    args = build_arg_parser().parse_args(["--phase", "deposit-topup", "--amount-ftns", "1"])
    rc = asyncio.run(mod._run_deposit_topup(args, _EP(), BASE_SEPOLIA_CHAIN_ID))
    out = capsys.readouterr().out
    assert rc == 0
    escrow.deposit.assert_awaited_once_with(topup)
    assert _REQUESTER_ADDR in out
    assert _REQUESTER_KEY not in out  # never the key


def test_deposit_topup_key_gated_errors_without_requester_key(monkeypatch):
    """deposit-topup is key-gated: with no requester key in env it errors clearly."""
    monkeypatch.delenv("PRSM_REQUESTER_KEY", raising=False)
    monkeypatch.delenv("REQUESTER_PRIVATE_KEY", raising=False)
    args = build_arg_parser().parse_args(["--phase", "deposit-topup", "--amount-ftns", "1"])
    with pytest.raises(SystemExit) as ei:
        asyncio.run(mod._run_deposit_topup(args, _EP(), BASE_SEPOLIA_CHAIN_ID))
    assert ei.value.code == 2


def test_deposit_topup_rejects_nonpositive_amount(monkeypatch):
    monkeypatch.setenv("PRSM_REQUESTER_KEY", _REQUESTER_KEY)
    args = build_arg_parser().parse_args(["--phase", "deposit-topup", "--amount-ftns", "0"])
    rc = asyncio.run(mod._run_deposit_topup(args, _EP(), BASE_SEPOLIA_CHAIN_ID))
    assert rc == 2


# ── (d) PRESERVED (regression) ──────────────────────────────────────────────────


def test_mainnet_guard_still_refuses_8453_without_flag():
    with pytest.raises(SystemExit):
        assert_chain_allowed(chain_id=BASE_MAINNET_CHAIN_ID, mainnet_acknowledged=False)


def test_deposit_topup_is_a_valid_phase_choice():
    args = build_arg_parser().parse_args(["--phase", "deposit-topup"])
    assert args.phase == "deposit-topup"
    # default phase is still the safe read-only verify.
    assert build_arg_parser().parse_args([]).phase == "verify"


def test_assume_dedicated_escrow_defaults_false():
    assert build_arg_parser().parse_args([]).assume_dedicated_escrow is False
