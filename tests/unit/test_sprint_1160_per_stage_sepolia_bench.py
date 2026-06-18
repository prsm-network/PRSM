"""Sprint 1160 — offline TDD for the PER-STAGE Base-Sepolia bench harness
(scripts/per_stage_sepolia_bench.py — the operator-run counterpart to the sp1159
local-hardhat bench).

ALL OFFLINE — no real network, no real keys. Drives the script's importable,
composable functions (key-gating, the mainnet guard, the phase-1 receipt+split
assembly, the verify-logic) directly, and the chain-touching paths against a
mocked web3 / mocked clients, exactly like tests/unit/test_sprint_1042_settlement_e2e_proof.py
does for e2e_proof. The LIVE Sepolia run is the operator's.

Coverage:
  (a) mainnet hard-guard refuses chainId 8453 without --i-understand-mainnet, and
      allows testnet 84532 (reuses e2e_proof.assert_chain_allowed, the same guard).
  (b) every key is read from env — with no key env set the readers error CLEARLY
      (SystemExit 2); no embedded/default key.
  (c) phase-1 assembly produces conserving, challenge-defensible per-node
      share-batches (reuses brick-1 / sp1159 assertions) — no chain.
  (d) the read-only verify-logic computes the expected per-node payouts (sum==total).
  (e) the script echoes the signer ADDRESS, not the key (sp1046).
"""
from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock

import pytest
from eth_account import Account
from web3 import Web3

from scripts.per_stage_sepolia_bench import (
    assemble_phase1,
    build_arg_parser,
    compute_expected_payouts,
    read_node_keys,
    read_requester_key,
)
from prsm.settlement.e2e_proof import (
    BASE_MAINNET_CHAIN_ID,
    BASE_SEPOLIA_CHAIN_ID,
    assert_chain_allowed,
)

ONE_FTNS = 10 ** 18

# Hardhat well-known testnet keys (PUBLIC, zero-value — fine for an OFFLINE test;
# never funded on any real network). One requester + 3 nodes.
_REQUESTER_KEY = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
_NODE_KEYS = [
    "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",
    "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6",
    "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a",
]
_REQUESTER_ADDR = Web3.to_checksum_address(Account.from_key(_REQUESTER_KEY).address)


# ── (a) mainnet hard-guard ────────────────────────────────────────────────────


def test_mainnet_guard_refuses_8453_without_flag():
    with pytest.raises(SystemExit):
        assert_chain_allowed(chain_id=BASE_MAINNET_CHAIN_ID, mainnet_acknowledged=False)


def test_mainnet_guard_allows_8453_with_flag():
    # Explicit acknowledgement permits mainnet (no raise).
    assert_chain_allowed(chain_id=BASE_MAINNET_CHAIN_ID, mainnet_acknowledged=True)


def test_guard_allows_testnet_84532():
    assert BASE_SEPOLIA_CHAIN_ID == 84532
    assert_chain_allowed(chain_id=BASE_SEPOLIA_CHAIN_ID, mainnet_acknowledged=False)


def test_default_phase_is_the_safe_readonly_verify():
    args = build_arg_parser().parse_args([])
    assert args.phase == "verify"
    assert args.i_understand_mainnet is False


# ── (b) key gating — every key from env, no embedded/default ───────────────────


def test_requester_key_required_no_default():
    with pytest.raises(SystemExit) as ei:
        read_requester_key(env={})
    assert ei.value.code == 2


def test_node_keys_required_no_default():
    with pytest.raises(SystemExit) as ei:
        read_node_keys(env={})
    assert ei.value.code == 2


def test_requester_key_read_from_env_and_0x_normalized():
    bare = _REQUESTER_KEY[2:]  # no 0x prefix
    assert read_requester_key(env={"PRSM_REQUESTER_KEY": bare}) == _REQUESTER_KEY
    # alias
    assert read_requester_key(env={"REQUESTER_PRIVATE_KEY": _REQUESTER_KEY}) == _REQUESTER_KEY


def test_node_keys_read_csv_and_indexed():
    csv = read_node_keys(env={"PRSM_NODE_KEYS": ",".join(_NODE_KEYS)})
    assert csv == _NODE_KEYS
    indexed = read_node_keys(env={
        "PRSM_NODE_KEY_0": _NODE_KEYS[0],
        "PRSM_NODE_KEY_1": _NODE_KEYS[1],
        "PRSM_NODE_KEY_2": _NODE_KEYS[2],
    })
    assert indexed == _NODE_KEYS


def test_no_real_key_string_embedded_in_source():
    """Guard against a key ever being hard-coded into the harness: the script source
    must contain no 0x-prefixed 64-hex literal (the shape of a private key)."""
    import re
    from pathlib import Path
    import scripts.per_stage_sepolia_bench as mod

    src = Path(mod.__file__).read_text()
    # 0x followed by exactly 64 hex chars (a private key / 32-byte literal).
    assert not re.search(r"0x[0-9a-fA-F]{64}", src), (
        "no private-key-shaped literal may appear in the bench source"
    )


# ── (c) phase-1 assembly: conserving, challenge-defensible per-node share-batches ─


def test_assemble_phase1_conserving_and_challenge_defensible():
    total = 30 * ONE_FTNS + 1  # non-divisible by 3 -> exercises the remainder
    assembly = assemble_phase1(
        node_keys=_NODE_KEYS, requester_address=_REQUESTER_ADDR, total_wei=total,
        request_id="req-sp1160-unit", executed_at_unix=1_700_000_000,
    )
    assert len(assembly.shares) == 3
    assert len(assembly.node_ids) == 3
    # Conservation: the per-node shares sum to the total EXACTLY (no FTNS made/lost).
    assert sum(s.share_wei for s in assembly.shares) == total
    assert sum(assembly.expected_share_by_node.values()) == total
    # Each committer eth address is the node's OWN key's address.
    expected_addrs = {
        Web3.to_checksum_address(Account.from_key(k).address) for k in _NODE_KEYS
    }
    assert set(assembly.node_eth.values()) == expected_addrs

    # Each per-node leaf is challenge-defensible: its embedded signature verifies
    # against its embedded pubkey over the canonical shard-payload preimage (reuses
    # the brick-1 / sp1159 leaf-defensibility property).
    from prsm.compute.shard_receipt import (
        build_receipt_signing_payload,
        per_stage_leaf_job_id,
    )
    from prsm.node.identity import verify_signature
    from prsm.settlement.merkle import batched_receipt_to_leaf, hash_leaf

    job_id = per_stage_leaf_job_id("req-sp1160-unit")
    for share in assembly.shares:
        r = share.batched_receipt.receipt
        payload = build_receipt_signing_payload(
            job_id=job_id, shard_index=r.shard_index,
            output_hash=r.output_hash, executed_at_unix=r.executed_at_unix,
        )
        assert verify_signature(r.provider_pubkey_b64, payload, r.signature), (
            "per-node leaf signature must verify (challenge-defensible)"
        )
        # The leaf is well-formed + hashable (raises if the receipt is malformed).
        assert len(hash_leaf(batched_receipt_to_leaf(share.batched_receipt))) == 32
        # provider binding: node_id == sha256(pubkey)[:32].
        import base64
        derived = hashlib.sha256(
            base64.b64decode(r.provider_pubkey_b64)).hexdigest()[:32]
        assert derived == r.provider_id == share.node_id


def test_assemble_phase1_requires_two_nodes():
    with pytest.raises(ValueError):
        assemble_phase1(node_keys=[_NODE_KEYS[0]], requester_address=_REQUESTER_ADDR,
                        total_wei=ONE_FTNS)


# ── (d) read-only verify-logic: expected per-node payouts (sum == total) ────────


def test_compute_expected_payouts_sums_to_total():
    total = 7 * ONE_FTNS + 2  # remainder 2 across 3 nodes
    assembly = assemble_phase1(
        node_keys=_NODE_KEYS, requester_address=_REQUESTER_ADDR, total_wei=total,
        request_id="req-sp1160-payouts", executed_at_unix=1_700_000_000,
    )
    payouts = compute_expected_payouts(assembly)
    # One payout per distinct committer eth address.
    assert set(payouts.keys()) == set(assembly.node_eth.values())
    assert sum(payouts.values()) == total
    # Matches the brick-1 share each node was assigned.
    for nid in assembly.node_ids:
        assert payouts[assembly.node_eth[nid]] == assembly.expected_share_by_node[nid]


# ── (e) the harness echoes the signer ADDRESS, not the key (sp1046) ─────────────


def test_phase1_echoes_address_not_key(monkeypatch, capsys, tmp_path):
    """Drive _run_phase1 with a MOCKED chain (no real RPC): assert it prints each
    signer ADDRESS and NEVER prints any private key."""
    import scripts.per_stage_sepolia_bench as mod

    # Mocked escrow + per-node clients so no chain is touched.
    escrow = AsyncMock()
    escrow.balance_of.return_value = 30 * ONE_FTNS + 1
    escrow.deposit.return_value = "0xdeposit"
    monkeypatch.setattr(mod, "EscrowPoolClient" if hasattr(mod, "EscrowPoolClient") else "_unused",
                        escrow, raising=False)

    # Patch the imported symbols inside _run_phase1 by patching the modules they
    # come from is heavier; instead patch the two helpers the chain step uses.
    monkeypatch.setattr(
        mod, "_build_node_clients",
        lambda **kw: {nid: AsyncMock() for nid in kw["assembly"].node_ids},
    )

    async def _fake_commit(*, shares, client_for_node, committer_address_for_node):
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

    monkeypatch.setattr(mod, "commit_per_node_share_batches", _fake_commit, raising=False)

    # EscrowPoolClient + run_deposit_and_commit are imported INSIDE _run_phase1 from
    # their home modules; patch those home modules so the inner imports pick up mocks.
    import prsm.economy.web3.escrow_pool_client as escrow_mod
    monkeypatch.setattr(escrow_mod, "EscrowPoolClient", lambda **kw: escrow)
    import prsm.settlement.per_stage_commit as psc_mod
    monkeypatch.setattr(psc_mod, "commit_per_node_share_batches", _fake_commit)

    monkeypatch.setenv("PRSM_REQUESTER_KEY", _REQUESTER_KEY)
    monkeypatch.setenv("PRSM_NODE_KEYS", ",".join(_NODE_KEYS))

    args = build_arg_parser().parse_args(
        ["--phase", "deposit-commit", "--amount-ftns", "1",
         "--state-file", str(tmp_path / "state.json")])

    # Fake endpoints object.
    class _EP:
        network_name = "testnet"
        rpc_url = "https://sepolia.base.org"
        escrow_pool = "0xaa28b5818242608e04C1773c3e34bF7bFfb96248"
        ftns_token = "0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a"
        settlement_registry = "0xF8BEEb4362222b50109b6034767322B31aA92449"

    import asyncio
    rc = asyncio.run(mod._run_phase1(args, _EP(), BASE_SEPOLIA_CHAIN_ID))
    out = capsys.readouterr().out
    assert rc == 0

    # The requester + every node ADDRESS is echoed.
    assert _REQUESTER_ADDR in out
    for k in _NODE_KEYS:
        assert Web3.to_checksum_address(Account.from_key(k).address) in out
    # NO private key (requester or node) ever appears in the output.
    assert _REQUESTER_KEY not in out
    for k in _NODE_KEYS:
        assert k not in out


def test_phase1_rejects_requester_equal_to_a_node(monkeypatch, tmp_path):
    """A node cannot also be the paying requester (self-pay, not a per-stage proof)."""
    import asyncio
    import scripts.per_stage_sepolia_bench as mod

    monkeypatch.setenv("PRSM_REQUESTER_KEY", _NODE_KEYS[0])  # == node[0]
    monkeypatch.setenv("PRSM_NODE_KEYS", ",".join(_NODE_KEYS))
    args = build_arg_parser().parse_args(
        ["--phase", "deposit-commit", "--state-file", str(tmp_path / "s.json")])

    class _EP:
        network_name = "testnet"
        rpc_url = "https://sepolia.base.org"
        escrow_pool = "0xaa28b5818242608e04C1773c3e34bF7bFfb96248"
        ftns_token = "0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a"
        settlement_registry = "0xF8BEEb4362222b50109b6034767322B31aA92449"

    rc = asyncio.run(mod._run_phase1(args, _EP(), BASE_SEPOLIA_CHAIN_ID))
    assert rc == 2
