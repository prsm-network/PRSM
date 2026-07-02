"""ContentAccessVerifier Web3 client (Tier B/C paid-decrypt arc, brick 4 consumer side — sp1354).

The consumer-facing driver for the ``ContentAccessVerifier`` contract (sp1353) — the production
``IRoyaltyPaymentVerifier`` that gates ``KeyDistribution.release``. A consumer calls
``pay_for_access(content_hash, fee_wei)`` here (approve FTNS + payForAccess); that records the
payment so the key can be released, and credits the fee to the content's registered creator.

Mirrors the patterns of ``royalty_distributor.py``:
  - Web3.py 7.x synchronous calls, explicit private-key signing
  - process-wide TX_LOCK_REGISTRY per-account lock (makes the approve+pay pair atomic + serializes
    nonce reads under concurrent callers)
  - three-tier error model: BroadcastFailedError / OnChainPendingError / OnChainRevertedError
"""
from __future__ import annotations

import threading
from typing import Optional, Tuple

try:
    from web3 import Web3
    from eth_account import Account
    HAS_WEB3 = True
except ImportError:  # pragma: no cover
    HAS_WEB3 = False
    Web3 = None  # type: ignore
    Account = None  # type: ignore

from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError,
    OnChainPendingError,
    OnChainRevertedError,
    TransferStatus,
)

CONTENT_ACCESS_VERIFIER_ABI = [
    {
        "type": "function", "name": "verifyPayment", "stateMutability": "view",
        "inputs": [
            {"name": "payer", "type": "address"},
            {"name": "contentHash", "type": "bytes32"},
            {"name": "feeWei", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
    },
    {
        "type": "function", "name": "payForAccess", "stateMutability": "nonpayable",
        "inputs": [
            {"name": "contentHash", "type": "bytes32"},
            {"name": "feeWei", "type": "uint256"},
        ],
        "outputs": [],
    },
    {
        "type": "function", "name": "claim", "stateMutability": "nonpayable",
        "inputs": [], "outputs": [],
    },
    {
        "type": "function", "name": "claimable", "stateMutability": "view",
        "inputs": [{"name": "", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
]

_ERC20_ABI = [
    {
        "type": "function", "name": "approve", "stateMutability": "nonpayable",
        "inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}],
        "outputs": [{"name": "", "type": "bool"}],
    },
    {
        "type": "function", "name": "allowance", "stateMutability": "view",
        "inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "type": "function", "name": "balanceOf", "stateMutability": "view",
        "inputs": [{"name": "account", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
]


class ContentAccessVerifierClient:
    def __init__(
        self,
        rpc_url: str,
        verifier_address: str,
        ftns_token_address: str,
        private_key: Optional[str] = None,
        expected_chain_id: Optional[int] = None,
    ) -> None:
        if not HAS_WEB3:
            raise RuntimeError("web3 package required")

        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        # sp1356 (review F7): pin the chain the signer commits to. A hostile/misconfigured RPC can
        # report (or drift to) a different chainId; without this, approve/payForAccess would be
        # signed against whatever the RPC claims. Assert the intended network up front, fail loud.
        if expected_chain_id is not None:
            actual = int(self.web3.eth.chain_id)
            if actual != int(expected_chain_id):
                raise RuntimeError(
                    f"RPC chainId {actual} != expected {expected_chain_id} — refusing to sign "
                    f"against an unintended chain")
        self.verifier_address = Web3.to_checksum_address(verifier_address)
        self.ftns_address = Web3.to_checksum_address(ftns_token_address)

        self.verifier = self.web3.eth.contract(
            address=self.verifier_address, abi=CONTENT_ACCESS_VERIFIER_ABI)
        self.token = self.web3.eth.contract(address=self.ftns_address, abi=_ERC20_ABI)

        self._account = None
        if private_key:
            self._account = Account.from_key(private_key)

        if self._account is not None:
            from prsm.economy.web3.tx_lock_registry import TX_LOCK_REGISTRY
            self._tx_lock = TX_LOCK_REGISTRY.get_lock(self._account.address)
        else:
            self._tx_lock = threading.Lock()

    @property
    def address(self) -> Optional[str]:
        return self._account.address if self._account else None

    # ── Reads ──────────────────────────────────────────────────

    def verify_payment(self, payer: str, content_hash: bytes, fee_wei: int) -> bool:
        """True iff ``payer`` has paid ``fee_wei`` for ``content_hash`` (the KeyDistribution gate)."""
        if len(content_hash) != 32:
            raise ValueError("content_hash must be 32 bytes")
        return bool(self.verifier.functions.verifyPayment(
            Web3.to_checksum_address(payer), content_hash, int(fee_wei)).call())

    def claimable(self, address: Optional[str] = None) -> int:
        """Accumulated FTNS (wei) a creator can withdraw via claim(). Defaults to the signer."""
        if address is None:
            if not self._account:
                raise RuntimeError("address required when no signer configured")
            address = self._account.address
        return int(self.verifier.functions.claimable(
            Web3.to_checksum_address(address)).call())

    def allowance(self) -> int:
        if not self._account:
            raise RuntimeError("private_key required")
        return int(self.token.functions.allowance(
            self._account.address, self.verifier_address).call())

    # ── Writes ─────────────────────────────────────────────────

    def pay_for_access(self, content_hash: bytes, fee_wei: int) -> Tuple[str, TransferStatus]:
        """Pay the release fee for ``content_hash`` (approve FTNS if needed, then payForAccess).

        Records the payment (so the key can be released to the signer) and credits the fee to the
        content's registered creator. Returns (tx_hash_hex, TransferStatus). May raise
        BroadcastFailedError / OnChainPendingError / OnChainRevertedError.
        """
        if not self._account:
            raise RuntimeError("private_key required")
        if len(content_hash) != 32:
            raise ValueError("content_hash must be 32 bytes")
        if int(fee_wei) <= 0:
            raise ValueError("fee_wei must be positive")

        with self._tx_lock:
            # Approve only if needed — the lock makes allowance-check + approve + payForAccess
            # atomic, so no competing approve interleaves.
            current = int(self.token.functions.allowance(
                self._account.address, self.verifier_address).call())
            if current < int(fee_wei):
                approve_tx = self.token.functions.approve(
                    self.verifier_address, int(fee_wei)).build_transaction(self._tx_overrides())
                self._sign_and_send(approve_tx)

            tx = self.verifier.functions.payForAccess(
                content_hash, int(fee_wei)).build_transaction(self._tx_overrides())
            return self._sign_and_send(tx)

    def claim(self) -> Tuple[str, TransferStatus]:
        """Creator withdraws accumulated access fees (reverts NothingToClaim if zero)."""
        if not self._account:
            raise RuntimeError("private_key required")
        with self._tx_lock:
            tx = self.verifier.functions.claim().build_transaction(self._tx_overrides())
            return self._sign_and_send(tx)

    # ── tx plumbing (mirrors royalty_distributor) ──────────────

    def _tx_overrides(self) -> dict:
        return {
            "from": self._account.address,
            "nonce": self.web3.eth.get_transaction_count(self._account.address, "pending"),
            "gasPrice": self.web3.eth.gas_price,
            "chainId": self.web3.eth.chain_id,
        }

    def _sign_and_send(self, tx: dict) -> Tuple[str, TransferStatus]:
        signed = self.web3.eth.account.sign_transaction(tx, self._account.key)
        raw = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction", None)
        try:
            tx_hash_bytes = self.web3.eth.send_raw_transaction(raw)
        except Exception as exc:
            raise BroadcastFailedError(f"broadcast failed: {exc}") from exc
        tx_hash_hex = "0x" + tx_hash_bytes.hex()
        try:
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash_bytes, timeout=120)
        except Exception as exc:
            raise OnChainPendingError(
                f"broadcast OK but receipt unknown: {exc}", tx_hash=tx_hash_hex) from exc
        if receipt.status != 1:
            raise OnChainRevertedError(f"tx reverted: {tx_hash_hex}")
        return tx_hash_hex, TransferStatus.CONFIRMED
