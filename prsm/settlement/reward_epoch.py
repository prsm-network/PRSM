"""Sprint 1481 — off-chain builder for OperatorRewardPool epochs (the pool → earner rail).

EmissionController mints protocol FTNS and CompensationDistributor splits it across
three pool ADDRESSES, but nothing routed those funds onward to the operators who
earned them. ``contracts/contracts/OperatorRewardPool.sol`` is the on-chain half:
publish a per-epoch Merkle root of ``(account, amount)`` entitlements, and each
earner claims permissionlessly with a proof. This module is the off-chain half that
builds that root and the per-earner proofs.

CANONICAL LEAF — must match OperatorRewardPool.leafHash EXACTLY::

    leaf = keccak256(keccak256(abi.encode(uint256 epochId, address account, uint256 amount)))

The DOUBLE hash is OpenZeppelin's standard second-preimage defence (a 64-byte
internal node can otherwise be presented as a leaf). ``epochId`` is bound INTO the
leaf so a proof for one epoch can never be replayed against another — the contract
test suite proves this by publishing the same root under two epoch ids.

Pair hashing reuses :mod:`prsm.settlement.merkle` (OZ sorted-pair keccak256, odd
nodes promoted), which already locks Python↔Solidity parity for the settlement
registry. Do NOT use :mod:`prsm.core.merkle` here — that one is sha256 and
index-ordered, and its roots will not verify against ``MerkleProof.sol``.

Parity is pinned by golden vectors generated FROM THE EVM (see
``tests/unit/test_sprint_1481_reward_epoch.py``); a change to the encoding on either
side breaks that test rather than silently producing unverifiable proofs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

from eth_abi import encode as abi_encode
from eth_utils import keccak, to_checksum_address

from prsm.settlement.merkle import build_merkle_proof, build_merkle_root

# uint256 ceiling — an amount at/over this cannot be abi-encoded as uint256.
_UINT256_MAX = (1 << 256) - 1


@dataclass(frozen=True)
class RewardEntry:
    """One earner's entitlement for an epoch.

    ``amount`` is in WEI (18-decimal FTNS base units), never a float — float FTNS
    would round and make the on-chain sum disagree with the published total.
    """
    account: str        # 0x EVM address
    amount_wei: int


@dataclass(frozen=True)
class RewardEpoch:
    """A built epoch: the root to publish, the total to reserve, and per-earner proofs."""
    epoch_id: int
    merkle_root: bytes
    total_amount_wei: int
    entries: List[RewardEntry]
    leaf_hashes: List[bytes]
    proofs: Dict[str, List[bytes]]     # checksummed account -> proof

    @property
    def root_hex(self) -> str:
        return "0x" + self.merkle_root.hex()

    def proof_hex(self, account: str) -> List[str]:
        return ["0x" + h.hex() for h in self.proofs[to_checksum_address(account)]]


def reward_leaf_hash(epoch_id: int, account: str, amount_wei: int) -> bytes:
    """The canonical leaf hash. Mirrors OperatorRewardPool.leafHash."""
    if epoch_id < 0 or epoch_id > _UINT256_MAX:
        raise ValueError(f"epoch_id out of uint256 range: {epoch_id}")
    if amount_wei < 0 or amount_wei > _UINT256_MAX:
        raise ValueError(f"amount_wei out of uint256 range: {amount_wei}")
    inner = keccak(abi_encode(
        ["uint256", "address", "uint256"],
        [int(epoch_id), to_checksum_address(account), int(amount_wei)],
    ))
    return keccak(inner)


def build_reward_epoch(
    epoch_id: int, entries: Iterable[RewardEntry],
) -> RewardEpoch:
    """Build the root + per-earner proofs for an epoch.

    Rejects the inputs that would produce a broken or unsafe epoch:

    * **empty entry set** — there is nothing to publish, and the contract rejects a
      zero root anyway.
    * **zero/negative amounts** — the contract rejects a zero-amount claim, so such a
      leaf is permanently unclaimable dead weight inside the reserved total.
    * **duplicate accounts** — the contract enforces ONE claim per (epoch, account),
      so a second leaf for the same account is silently unclaimable while still
      inflating the declared total. Aggregate upstream instead.

    Entries are sorted by account for a deterministic root: the same entitlement set
    must always produce the same root regardless of input ordering, so an epoch can
    be independently rebuilt and verified by anyone.
    """
    entry_list = list(entries)
    if not entry_list:
        raise ValueError("build_reward_epoch: no entries")

    seen = set()
    normalized: List[RewardEntry] = []
    for e in entry_list:
        acct = to_checksum_address(e.account)
        amount = int(e.amount_wei)
        if amount <= 0:
            raise ValueError(
                f"build_reward_epoch: non-positive amount for {acct} "
                f"({amount}) — the contract rejects a zero-amount claim, so this "
                "leaf would be unclaimable but still reserved"
            )
        if amount > _UINT256_MAX:
            raise ValueError(f"amount_wei out of uint256 range for {acct}")
        if acct in seen:
            raise ValueError(
                f"build_reward_epoch: duplicate account {acct} — only ONE claim per "
                "(epoch, account) is possible on chain; aggregate before building"
            )
        seen.add(acct)
        normalized.append(RewardEntry(account=acct, amount_wei=amount))

    normalized.sort(key=lambda e: int(e.account, 16))

    leaves = [reward_leaf_hash(epoch_id, e.account, e.amount_wei) for e in normalized]
    root = build_merkle_root(leaves)
    total = sum(e.amount_wei for e in normalized)

    proofs = {
        e.account: build_merkle_proof(leaves, i)
        for i, e in enumerate(normalized)
    }
    return RewardEpoch(
        epoch_id=int(epoch_id),
        merkle_root=root,
        total_amount_wei=total,
        entries=normalized,
        leaf_hashes=leaves,
        proofs=proofs,
    )


def aggregate_entitlements(
    rows: Iterable[Mapping[str, object]],
    *,
    account_key: str = "account",
    amount_key: str = "amount_wei",
) -> List[RewardEntry]:
    """Collapse many per-work rows into one entry per account.

    The epoch source is a set of FINALIZED settlement records, and one operator
    normally appears many times in an epoch. The contract permits a single claim per
    (epoch, account), so the rows MUST be summed before the tree is built —
    otherwise every occurrence after the first is unclaimable while still counting
    toward the reserved total. Rows summing to zero are dropped rather than becoming
    dead leaves.
    """
    totals: Dict[str, int] = {}
    for row in rows:
        acct = to_checksum_address(str(row[account_key]))
        amount = int(row[amount_key])  # type: ignore[call-overload]
        totals[acct] = totals.get(acct, 0) + amount
    return [
        RewardEntry(account=a, amount_wei=v)
        for a, v in sorted(totals.items(), key=lambda kv: int(kv[0], 16))
        if v > 0
    ]


def verify_reward_proof(
    epoch_id: int,
    account: str,
    amount_wei: int,
    proof: Sequence[bytes],
    merkle_root: bytes,
) -> bool:
    """Verify a proof exactly as ``MerkleProof.verify`` does on chain.

    Lets the CLI tell an operator "your claim will succeed" before spending gas.
    """
    from prsm.settlement.merkle import verify_merkle_proof
    leaf = reward_leaf_hash(epoch_id, account, amount_wei)
    # Argument order mirrors Solidity MerkleProof.verify(proof, root, leaf).
    return verify_merkle_proof(list(proof), merkle_root, leaf)
