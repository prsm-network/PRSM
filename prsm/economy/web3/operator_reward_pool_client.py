"""Sprint 1481 — client for OperatorRewardPool (the pool → earner claim surface).

The last link of the rail: an operator discovers what they are owed and claims it.

THE TRUST PROBLEM THIS SOLVES
-----------------------------
Merkle proofs are NOT on chain — only the root is. So an operator must obtain their
``(amount, proof)`` from an off-chain epoch manifest published by whoever runs the
epoch job. That manifest is untrusted input: a wrong or hostile one can hand an
operator a proof for the wrong amount, a stale epoch, or a root that was never
published.

``verify_manifest_entry_against_chain`` therefore re-derives the leaf and folds the
proof locally, and compares the result to the root READ FROM THE CONTRACT — never
to the root asserted by the manifest. A manifest that disagrees with the chain is
rejected before any gas is spent. The claim itself is also safe by construction
(the contract verifies the proof and pays the leaf's ``account``, not the sender),
so the worst a bad manifest can do is waste gas — but failing locally is cheaper
and tells the operator exactly what is wrong.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:
    from web3 import Web3
    from eth_account import Account
    HAS_WEB3 = True
except ImportError:  # pragma: no cover - exercised on installs without web3
    HAS_WEB3 = False
    Web3 = None  # type: ignore
    Account = None  # type: ignore

OPERATOR_REWARD_POOL_ABI = [
    {"name": "epochs", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "", "type": "uint256"}],
     "outputs": [{"name": "merkleRoot", "type": "bytes32"},
                 {"name": "totalAmount", "type": "uint256"},
                 {"name": "claimedAmount", "type": "uint256"},
                 {"name": "publishedAt", "type": "uint64"},
                 {"name": "reclaimed", "type": "bool"}]},
    {"name": "hasClaimed", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "", "type": "uint256"}, {"name": "", "type": "address"}],
     "outputs": [{"type": "bool"}]},
    {"name": "leafHash", "type": "function", "stateMutability": "pure",
     "inputs": [{"name": "epochId", "type": "uint256"},
                {"name": "account", "type": "address"},
                {"name": "amount", "type": "uint256"}],
     "outputs": [{"type": "bytes32"}]},
    {"name": "isClaimable", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "epochId", "type": "uint256"},
                {"name": "account", "type": "address"},
                {"name": "amount", "type": "uint256"},
                {"name": "proof", "type": "bytes32[]"}],
     "outputs": [{"type": "bool"}]},
    {"name": "claim", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "epochId", "type": "uint256"},
                {"name": "account", "type": "address"},
                {"name": "amount", "type": "uint256"},
                {"name": "proof", "type": "bytes32[]"}],
     "outputs": []},
    {"name": "totalReserved", "type": "function", "stateMutability": "view",
     "inputs": [], "outputs": [{"type": "uint256"}]},
    {"name": "paused", "type": "function", "stateMutability": "view",
     "inputs": [], "outputs": [{"type": "bool"}]},
]


class ManifestMismatchError(RuntimeError):
    """The manifest's proof does not verify against the ON-CHAIN root."""


@dataclass(frozen=True)
class ClaimableEpoch:
    """One epoch this operator can claim from."""
    epoch_id: int
    amount_wei: int
    proof: List[bytes]
    on_chain_root: bytes
    already_claimed: bool

    @property
    def amount_ftns(self) -> float:
        return self.amount_wei / 1e18


def _to_bytes32(v: Any) -> bytes:
    if isinstance(v, bytes):
        return v
    s = str(v)
    return bytes.fromhex(s[2:] if s.startswith("0x") else s)


def load_epoch_manifest(source: str) -> Dict[str, Any]:
    """Load an epoch manifest from a local path or an http(s) URL.

    The manifest is UNTRUSTED — every entry it yields is re-verified against the
    on-chain root before use (see verify_manifest_entry_against_chain).
    """
    if source.startswith(("http://", "https://")):
        import urllib.request
        with urllib.request.urlopen(source, timeout=30) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    with open(source, "r", encoding="utf-8") as fh:
        return json.load(fh)


def manifest_entry_for(manifest: Dict[str, Any], account: str) -> Optional[Dict[str, Any]]:
    """Find this account's entry in a manifest (case-insensitive on the address)."""
    target = account.lower()
    for e in manifest.get("entries", []):
        if str(e.get("account", "")).lower() == target:
            return e
    return None


def verify_manifest_entry_against_chain(
    *,
    epoch_id: int,
    account: str,
    amount_wei: int,
    proof: Sequence[bytes],
    on_chain_root: bytes,
) -> bool:
    """Fold the proof locally and compare to the CONTRACT's root.

    Deliberately does NOT consult the manifest's own ``merkle_root`` field: a
    manifest that lies about the root would otherwise validate against itself.
    """
    from prsm.settlement.reward_epoch import verify_reward_proof
    return verify_reward_proof(
        epoch_id, account, amount_wei, list(proof), bytes(on_chain_root)
    )


class OperatorRewardPoolClient:
    """Read entitlements and submit claims against OperatorRewardPool."""

    def __init__(
        self,
        rpc_url: str,
        pool_address: str,
        private_key: Optional[str] = None,
        expected_chain_id: Optional[int] = None,
    ) -> None:
        if not HAS_WEB3:
            raise RuntimeError("web3 package required")
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        # sp1356 — pin the chain the signer commits to; a hostile/misconfigured RPC
        # must not get a claim signed against an unintended network.
        if expected_chain_id is not None:
            actual = int(self.web3.eth.chain_id)
            if actual != int(expected_chain_id):
                raise RuntimeError(
                    f"RPC chainId {actual} != expected {expected_chain_id} — "
                    "refusing to sign against an unintended chain"
                )
        self.pool_address = Web3.to_checksum_address(pool_address)
        self.pool = self.web3.eth.contract(
            address=self.pool_address, abi=OPERATOR_REWARD_POOL_ABI)

        self._account = Account.from_key(private_key) if private_key else None
        if self._account is not None:
            # sp931 — share the process-wide per-account nonce lock so a claim
            # cannot collide with another client signing from the same key.
            from prsm.economy.web3.tx_lock_registry import TX_LOCK_REGISTRY
            self._tx_lock = TX_LOCK_REGISTRY.get_lock(self._account.address)
        else:
            self._tx_lock = threading.Lock()

    @property
    def address(self) -> Optional[str]:
        return self._account.address if self._account else None

    # ── Reads ───────────────────────────────────────────────────────────

    def get_epoch(self, epoch_id: int) -> Dict[str, Any]:
        root, total, claimed, published_at, reclaimed = self.pool.functions.epochs(
            int(epoch_id)).call()
        return {
            "epoch_id": int(epoch_id),
            "merkle_root": bytes(root),
            "total_amount_wei": int(total),
            "claimed_amount_wei": int(claimed),
            "published_at": int(published_at),
            "reclaimed": bool(reclaimed),
            "published": int(published_at) != 0,
        }

    def has_claimed(self, epoch_id: int, account: str) -> bool:
        return bool(self.pool.functions.hasClaimed(
            int(epoch_id), Web3.to_checksum_address(account)).call())

    def is_claimable(
        self, epoch_id: int, account: str, amount_wei: int, proof: Sequence[bytes],
    ) -> bool:
        return bool(self.pool.functions.isClaimable(
            int(epoch_id), Web3.to_checksum_address(account), int(amount_wei),
            [bytes(p) for p in proof]).call())

    def leaf_hash(self, epoch_id: int, account: str, amount_wei: int) -> bytes:
        return bytes(self.pool.functions.leafHash(
            int(epoch_id), Web3.to_checksum_address(account), int(amount_wei)).call())

    def paused(self) -> bool:
        return bool(self.pool.functions.paused().call())

    def resolve_claimable(
        self, manifest: Dict[str, Any], account: str,
    ) -> Optional[ClaimableEpoch]:
        """Resolve this account's entitlement from a manifest, verified on chain.

        Returns None when the manifest has no entry for the account. Raises
        ManifestMismatchError when the manifest's proof does not verify against the
        on-chain root — the case that must never be silently claimed against.
        """
        epoch_id = int(manifest["epoch_id"])
        entry = manifest_entry_for(manifest, account)
        if entry is None:
            return None
        amount_wei = int(entry["amount_wei"])
        proof = [_to_bytes32(p) for p in entry.get("proof", [])]

        chain = self.get_epoch(epoch_id)
        if not chain["published"]:
            raise ManifestMismatchError(
                f"epoch {epoch_id} is not published on chain — the manifest is "
                "ahead of (or unrelated to) the deployed pool"
            )
        if not verify_manifest_entry_against_chain(
            epoch_id=epoch_id, account=account, amount_wei=amount_wei,
            proof=proof, on_chain_root=chain["merkle_root"],
        ):
            raise ManifestMismatchError(
                f"epoch {epoch_id}: the manifest's proof for {account} does NOT "
                f"verify against the on-chain root "
                f"0x{chain['merkle_root'].hex()} — refusing to submit. The "
                "manifest is stale, for another deployment, or tampered with."
            )
        return ClaimableEpoch(
            epoch_id=epoch_id,
            amount_wei=amount_wei,
            proof=proof,
            on_chain_root=chain["merkle_root"],
            already_claimed=self.has_claimed(epoch_id, account),
        )

    # ── Write ───────────────────────────────────────────────────────────

    def claim(
        self, epoch_id: int, account: str, amount_wei: int, proof: Sequence[bytes],
    ) -> str:
        """Submit a claim. Returns the tx hash.

        The recipient is ``account`` (the leaf's address) regardless of who signs,
        so a funded relayer can pay gas for an operator without being able to
        redirect the reward.
        """
        if self._account is None:
            raise RuntimeError("no private key configured — cannot submit a claim")
        acct = Web3.to_checksum_address(account)
        with self._tx_lock:
            tx = self.pool.functions.claim(
                int(epoch_id), acct, int(amount_wei), [bytes(p) for p in proof]
            ).build_transaction({
                "from": self._account.address,
                "nonce": self.web3.eth.get_transaction_count(
                    self._account.address, "pending"),
                "gasPrice": self.web3.eth.gas_price,
                "chainId": self.web3.eth.chain_id,
            })
            signed = self.web3.eth.account.sign_transaction(tx, self._account.key)
            raw = (getattr(signed, "raw_transaction", None)
                   or getattr(signed, "rawTransaction", None))
            tx_hash = self.web3.eth.send_raw_transaction(raw)
        return "0x" + tx_hash.hex()
