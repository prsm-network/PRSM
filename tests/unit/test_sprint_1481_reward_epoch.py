"""Sprint 1481 — reward-epoch builder ↔ OperatorRewardPool.sol parity + safety.

The pool → earner rail only works if the Python builder's leaf/root convention is
byte-identical to the contract's. A mismatch does not fail loudly — it produces
roots whose proofs simply never verify, i.e. an epoch nobody can claim, with the
funds reserved on chain.

So the parity vectors below were generated FROM THE EVM (an actual
``OperatorRewardPool.leafHash`` call against a deployed contract on the hardhat
network), not from a second Python implementation of the same idea. Regenerate with
``contracts/scripts/gen_reward_epoch_goldens.js`` if the encoding ever changes
deliberately — and change the contract in the same commit.
"""
from __future__ import annotations

import pytest
from eth_utils import to_checksum_address

from prsm.settlement.reward_epoch import (
    RewardEntry,
    aggregate_entitlements,
    build_reward_epoch,
    reward_leaf_hash,
    verify_reward_proof,
)

# ── Golden vectors: emitted by OperatorRewardPool.leafHash on the hardhat EVM ──
EVM_LEAF_VECTORS = [
    # (epoch_id, account, amount_wei, expected_leaf_hex)
    (1, "0x1111111111111111111111111111111111111111", 1,
     "0xa5acae06065170f6011fdcb9a415a258bf2d471029397024dfcead7cbe030592"),
    (1, "0xabc0000000000000000000000000000000000001", 10**18,
     "0xda920222bf0ae706794e3680ace7dec451ad9b15c748f4559a51c3e458ae6df6"),
    (42, "0xdead000000000000000000000000000000000000", 123456789012345678901,
     "0x408b1339d72848ee74b4915b167cd65e8bd44b5ff8a4c23c3a49fbac56b854ee"),
    (0, "0x0000000000000000000000000000000000000001", 2**255,
     "0x390ad53ea7529a9f5c0a7062128d5e77820a8521ffabedf4128f586f9c07c4ee"),
]

# A 3-entry epoch (odd layer -> exercises OZ odd-node promotion), root computed
# on chain over on-chain leaf hashes.
EVM_TREE_EPOCH_ID = 7
EVM_TREE_ENTRIES = [
    ("0x1111111111111111111111111111111111111111", 100),
    ("0x2222222222222222222222222222222222222222", 250),
    ("0x3333333333333333333333333333333333333333", 50),
]
EVM_TREE_ROOT = "0x450d44b4dd738d852387f0e66d70bdbb4cebd12dd65f36caddecfcbd69130354"


# ─────────────────────────── EVM parity ───────────────────────────

@pytest.mark.parametrize("epoch_id,account,amount,expected", EVM_LEAF_VECTORS)
def test_leaf_hash_matches_evm(epoch_id, account, amount, expected):
    """★ The single most load-bearing assertion in this sprint: the Python leaf
    equals what the CONTRACT computes. Drift here = silently unclaimable epochs."""
    assert "0x" + reward_leaf_hash(epoch_id, account, amount).hex() == expected


def test_root_matches_evm_including_odd_layer_promotion():
    """★ Root parity over a 3-leaf tree — the odd-count layer is where a
    duplicate-last-node convention (a common alternative) would diverge."""
    epoch = build_reward_epoch(
        EVM_TREE_EPOCH_ID,
        [RewardEntry(account=a, amount_wei=v) for a, v in EVM_TREE_ENTRIES],
    )
    assert epoch.root_hex == EVM_TREE_ROOT


def test_leaf_hash_is_case_insensitive_on_address():
    """Address casing must not change the leaf — abi.encode takes 20 raw bytes."""
    lower = reward_leaf_hash(1, "0xabc0000000000000000000000000000000000001", 10**18)
    checksummed = reward_leaf_hash(
        1, to_checksum_address("0xabc0000000000000000000000000000000000001"), 10**18)
    assert lower == checksummed


def test_epoch_id_is_bound_into_the_leaf():
    """★ Cross-epoch replay defence: same account+amount, different epoch -> a
    different leaf, so a proof cannot be reused against another epoch."""
    a, amt = "0x1111111111111111111111111111111111111111", 100
    assert reward_leaf_hash(1, a, amt) != reward_leaf_hash(2, a, amt)


# ─────────────────────────── proofs ───────────────────────────

@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 9, 17])
def test_every_entry_gets_a_verifying_proof(n):
    """Every earner in an epoch of any size must be able to prove membership —
    including odd sizes, where the tree has promoted nodes."""
    entries = [
        RewardEntry(account="0x" + f"{i+1:040x}", amount_wei=(i + 1) * 10**18)
        for i in range(n)
    ]
    epoch = build_reward_epoch(3, entries)
    assert epoch.total_amount_wei == sum(e.amount_wei for e in entries)
    for e in epoch.entries:
        assert verify_reward_proof(
            3, e.account, e.amount_wei,
            epoch.proofs[e.account], epoch.merkle_root,
        ), f"proof failed for {e.account} in an epoch of {n}"


def test_proof_rejects_wrong_amount_and_wrong_account():
    entries = [
        RewardEntry(account="0x" + f"{i+1:040x}", amount_wei=(i + 1) * 10**18)
        for i in range(4)
    ]
    epoch = build_reward_epoch(3, entries)
    victim = epoch.entries[0]
    proof = epoch.proofs[victim.account]
    # Inflated amount with a valid proof must not verify.
    assert not verify_reward_proof(
        3, victim.account, victim.amount_wei + 1, proof, epoch.merkle_root)
    # Someone else's address with this proof must not verify.
    assert not verify_reward_proof(
        3, "0x" + "9" * 40, victim.amount_wei, proof, epoch.merkle_root)
    # Right leaf, wrong epoch.
    assert not verify_reward_proof(
        4, victim.account, victim.amount_wei, proof, epoch.merkle_root)


def test_root_is_deterministic_regardless_of_input_order():
    """An epoch must be independently rebuildable — same entitlements, same root."""
    a = [
        RewardEntry("0x" + "1" * 40, 5),
        RewardEntry("0x" + "2" * 40, 7),
        RewardEntry("0x" + "3" * 40, 9),
    ]
    b = list(reversed(a))
    assert build_reward_epoch(1, a).merkle_root == build_reward_epoch(1, b).merkle_root


# ─────────────────────── input safety ───────────────────────

def test_duplicate_account_is_rejected():
    """★ The contract allows ONE claim per (epoch, account). A duplicate leaf would
    be unclaimable while still inflating the reserved total — reject at build time."""
    with pytest.raises(ValueError, match="duplicate account"):
        build_reward_epoch(1, [
            RewardEntry("0x" + "1" * 40, 100),
            RewardEntry("0x" + "1" * 40, 200),
        ])


def test_zero_and_negative_amounts_rejected():
    """The contract reverts a zero-amount claim, so such a leaf is dead weight."""
    for bad in (0, -1):
        with pytest.raises(ValueError, match="non-positive amount"):
            build_reward_epoch(1, [RewardEntry("0x" + "1" * 40, bad)])


def test_empty_epoch_rejected():
    with pytest.raises(ValueError, match="no entries"):
        build_reward_epoch(1, [])


def test_amount_out_of_uint256_range_rejected():
    with pytest.raises(ValueError, match="uint256 range"):
        build_reward_epoch(1, [RewardEntry("0x" + "1" * 40, 2**256)])


# ─────────────────────── aggregation ───────────────────────

def test_aggregate_sums_repeat_rows_per_account():
    """★ An operator appears many times per epoch; rows MUST be summed before the
    tree is built or every occurrence after the first is unclaimable."""
    acct = "0x" + "1" * 40
    other = "0x" + "2" * 40
    rows = [
        {"account": acct, "amount_wei": 100},
        {"account": other, "amount_wei": 5},
        {"account": acct.upper().replace("0X", "0x"), "amount_wei": 250},
    ]
    out = {e.account: e.amount_wei for e in aggregate_entitlements(rows)}
    assert out[to_checksum_address(acct)] == 350   # summed, not last-write-wins
    assert out[to_checksum_address(other)] == 5


def test_aggregate_drops_accounts_summing_to_zero():
    rows = [
        {"account": "0x" + "1" * 40, "amount_wei": 100},
        {"account": "0x" + "1" * 40, "amount_wei": -100},
        {"account": "0x" + "2" * 40, "amount_wei": 7},
    ]
    out = aggregate_entitlements(rows)
    assert [e.account for e in out] == [to_checksum_address("0x" + "2" * 40)]


def test_aggregated_epoch_total_matches_sum_of_rows():
    """The published totalAmount must equal the sum of the leaves, or the contract's
    solvency reserve is wrong in one direction or the other."""
    rows = [
        {"account": "0x" + "1" * 40, "amount_wei": 100},
        {"account": "0x" + "2" * 40, "amount_wei": 250},
        {"account": "0x" + "1" * 40, "amount_wei": 50},
    ]
    epoch = build_reward_epoch(9, aggregate_entitlements(rows))
    assert epoch.total_amount_wei == 400
    assert sum(e.amount_wei for e in epoch.entries) == epoch.total_amount_wei


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
