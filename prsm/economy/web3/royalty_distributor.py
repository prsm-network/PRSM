"""RoyaltyDistributor Web3 Client.

Wraps the RoyaltyDistributor contract and the FTNS ERC-20 approval flow:
when a payer wants to distribute `gross` FTNS for content X, this client
first checks the existing allowance and only sends an `approve` tx when
needed, then calls `distributeRoyalty`.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    from web3 import Web3
    from eth_account import Account
    HAS_WEB3 = True
except ImportError:  # pragma: no cover
    HAS_WEB3 = False
    Web3 = None  # type: ignore
    Account = None  # type: ignore

# Re-export the broadcast/settle types from the registry module so tests
# and downstream callers can import them from either client uniformly.
from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError,
    OnChainPendingError,
    OnChainRevertedError,
    TransferStatus,
)

logger = logging.getLogger(__name__)


ROYALTY_DISTRIBUTOR_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
            {"internalType": "address", "name": "servingNode", "type": "address"},
            {"internalType": "uint256", "name": "gross", "type": "uint256"},
        ],
        "name": "distributeRoyalty",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
            {"internalType": "uint256", "name": "gross", "type": "uint256"},
        ],
        "name": "preview",
        "outputs": [
            {"internalType": "uint256", "name": "creatorAmount", "type": "uint256"},
            {"internalType": "uint256", "name": "networkAmount", "type": "uint256"},
            {"internalType": "uint256", "name": "servingNodeAmount", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
            {"indexed": True, "internalType": "address", "name": "payer", "type": "address"},
            {"indexed": True, "internalType": "address", "name": "creator", "type": "address"},
            {"indexed": False, "internalType": "address", "name": "servingNode", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "creatorAmount", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "networkAmount", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "servingNodeAmount", "type": "uint256"},
        ],
        "name": "RoyaltyPaid",
        "type": "event",
    },
    # T6.2 (2026-05-05): pull-payment surface (D-04 refactor).
    # `claimable(address)` is the auto-generated getter for the public
    # `mapping(address => uint256) public claimable` field; `claim()`
    # withdraws msg.sender's accumulated balance.
    {
        "inputs": [{"internalType": "address", "name": "", "type": "address"}],
        "name": "claimable",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "claim",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "recipient", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "amount", "type": "uint256"},
        ],
        "name": "RoyaltyClaimed",
        "type": "event",
    },
]

# Minimal ERC-20 ABI subset we need (allowance + approve)
_ERC20_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "owner", "type": "address"},
            {"internalType": "address", "name": "spender", "type": "address"},
        ],
        "name": "allowance",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "spender", "type": "address"},
            {"internalType": "uint256", "name": "value", "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]


@dataclass
class SplitPreview:
    creator_amount: int
    network_amount: int
    serving_node_amount: int


class RoyaltyDistributorClient:
    """Sync Web3 client for RoyaltyDistributor + FTNS approval flow."""

    def __init__(
        self,
        rpc_url: str,
        distributor_address: str,
        ftns_token_address: str,
        private_key: Optional[str] = None,
    ) -> None:
        if not HAS_WEB3:
            raise RuntimeError("web3 package required")

        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self.distributor_address = Web3.to_checksum_address(distributor_address)
        self.ftns_address = Web3.to_checksum_address(ftns_token_address)

        self.distributor = self.web3.eth.contract(
            address=self.distributor_address, abi=ROYALTY_DISTRIBUTOR_ABI
        )
        self.token = self.web3.eth.contract(
            address=self.ftns_address, abi=_ERC20_ABI
        )

        self._account = None
        if private_key:
            self._account = Account.from_key(private_key)

        # Phase 1.1 Task 5: lock makes the approve+distribute pair atomic
        # per client, and serializes nonce reads under concurrent callers.
        # sp931 — use the PROCESS-WIDE per-account lock so every client signing
        # from this same account (OnChainFTNSLedger, StakeManager, …) serializes
        # its nonce-fetch→sign→send through the SAME lock; two independent locks
        # let both read the same "pending" nonce and broadcast a colliding tx
        # that is silently dropped ("nonce too low"). Falls back to a private
        # lock only when no signing account is configured.
        if self._account is not None:
            from prsm.economy.web3.tx_lock_registry import TX_LOCK_REGISTRY
            self._tx_lock = TX_LOCK_REGISTRY.get_lock(self._account.address)
        else:
            self._tx_lock = threading.Lock()

    @property
    def address(self) -> Optional[str]:
        return self._account.address if self._account else None

    # ── Reads ──────────────────────────────────────────────────

    def preview_split(self, content_hash: bytes, gross: int) -> SplitPreview:
        if len(content_hash) != 32:
            raise ValueError("content_hash must be 32 bytes")
        creator_amt, network_amt, node_amt = (
            self.distributor.functions.preview(content_hash, gross).call()
        )
        return SplitPreview(
            creator_amount=int(creator_amt),
            network_amount=int(network_amt),
            serving_node_amount=int(node_amt),
        )

    def allowance(self) -> int:
        if not self._account:
            raise RuntimeError("private_key required")
        return int(
            self.token.functions.allowance(
                self._account.address, self.distributor_address
            ).call()
        )

    def claimable(self, address: Optional[str] = None) -> int:
        """T6.2: read accumulated pull-payment balance for `address`.

        Returns the FTNS amount (in wei) the address can withdraw via
        claim(). Defaults to the configured signer's address.
        """
        if address is None:
            if not self._account:
                raise RuntimeError("address required when no signer configured")
            address = self._account.address
        return int(
            self.distributor.functions.claimable(
                Web3.to_checksum_address(address)
            ).call()
        )

    # ── Writes ─────────────────────────────────────────────────

    def distribute_royalty(
        self, content_hash: bytes, serving_node: str, gross: int
    ) -> Tuple[str, TransferStatus]:
        """Distribute `gross` FTNS via the on-chain RoyaltyDistributor.

        Returns (tx_hash_hex, TransferStatus). May raise BroadcastFailedError
        (safe fallback), OnChainPendingError (do NOT fall back), or
        OnChainRevertedError (safe fallback).
        """
        if not self._account:
            raise RuntimeError("private_key required")
        if len(content_hash) != 32:
            raise ValueError("content_hash must be 32 bytes")
        if gross <= 0:
            raise ValueError("gross must be positive")

        with self._tx_lock:
            # Approve only if needed. The lock makes this allowance check
            # plus the subsequent approve+distribute pair atomic — no
            # other thread can interleave a competing approve.
            current_allowance = int(
                self.token.functions.allowance(
                    self._account.address, self.distributor_address
                ).call()
            )
            if current_allowance < gross:
                approve_tx = self.token.functions.approve(
                    self.distributor_address, gross
                ).build_transaction(self._tx_overrides())
                self._sign_and_send(approve_tx)

            tx = self.distributor.functions.distributeRoyalty(
                content_hash, Web3.to_checksum_address(serving_node), gross
            ).build_transaction(self._tx_overrides())
            return self._sign_and_send(tx)

    def claim(self) -> Tuple[str, TransferStatus]:
        """T6.2: withdraw accumulated pull-payment balance for the signer.

        Calls `RoyaltyDistributor.claim()` which transfers the signer's
        full `claimable[msg.sender]` balance to msg.sender as FTNS,
        zeroing the mapping entry. No-op (reverts ZeroClaim) if the
        balance is 0.

        Returns (tx_hash_hex, TransferStatus).
        """
        if not self._account:
            raise RuntimeError("private_key required for claim()")
        with self._tx_lock:
            tx = self.distributor.functions.claim().build_transaction(
                self._tx_overrides()
            )
            return self._sign_and_send(tx)

    # ── Internals ──────────────────────────────────────────────

    def _tx_overrides(self) -> dict:
        return {
            "from": self._account.address,
            # Phase 1.1 Task 5: pending nonce so the approve+distribute pair
            # under one lock acquisition sees its own pending state.
            "nonce": self.web3.eth.get_transaction_count(
                self._account.address, "pending"
            ),
            "gasPrice": self.web3.eth.gas_price,
            "chainId": self.web3.eth.chain_id,
        }

    def _sign_and_send(self, tx: dict) -> Tuple[str, TransferStatus]:
        """Sign and broadcast a tx. Returns (tx_hash_hex, status).

        Same broadcast/settle distinction as ProvenanceRegistryClient:
          - BroadcastFailedError: safe to fall back
          - OnChainPendingError: DO NOT fall back (chain may settle)
          - OnChainRevertedError: safe to fall back (chain rolled it back)
        """
        signed = self.web3.eth.account.sign_transaction(tx, self._account.key)
        raw = getattr(signed, "raw_transaction", None) or getattr(
            signed, "rawTransaction", None
        )

        try:
            tx_hash_bytes = self.web3.eth.send_raw_transaction(raw)
        except Exception as exc:
            raise BroadcastFailedError(f"broadcast failed: {exc}") from exc

        tx_hash_hex = "0x" + tx_hash_bytes.hex()

        try:
            receipt = self.web3.eth.wait_for_transaction_receipt(
                tx_hash_bytes, timeout=120
            )
        except Exception as exc:
            raise OnChainPendingError(
                f"broadcast OK but receipt unknown: {exc}", tx_hash=tx_hash_hex
            ) from exc

        if receipt.status != 1:
            raise OnChainRevertedError(f"tx reverted: {tx_hash_hex}")

        return tx_hash_hex, TransferStatus.CONFIRMED
