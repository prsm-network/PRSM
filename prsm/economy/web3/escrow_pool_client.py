"""Sprint 1041 — EscrowPoolClient (brick 3a): the requester-funding mechanism.

On-chain settlement (sprint 1021-1040) commits + finalizes batches, but
``finalizeBatch`` calls ``EscrowPool.settleFromRequester(requester, recipient,
amount)`` which reverts unless the requester holds a funded escrow balance. Until
this client shipped there was no way to deposit into EscrowPool, so the rail could
only self-pay-of-zero (mainnet EscrowPool balance == 0 → a live finalize reverts).

Mirrors ``Web3SettlementContractClient`` (sync web3 + ``Account`` signing +
``TX_LOCK_REGISTRY`` nonce serialization + the three-tier error model), with each
blocking call wrapped in ``asyncio.to_thread``. ``EscrowPool.deposit(amount)``
requires the caller to have approved FTNS for the pool first, so ``deposit()``
ensures ``allowance >= amount`` (approving only when short) before depositing.

Scope: deposit/withdraw move the CALLER's own FTNS into/out of the CALLER's own
escrow balance (recoverable), so the blast radius is the operator's own funds —
narrower than the settlement client. ABIs are minimal inline subsets verified
against ``contracts/contracts/EscrowPool.sol`` + the ERC-20 standard.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

try:
    from web3 import Web3
    from eth_account import Account
    HAS_WEB3 = True
except ImportError:  # pragma: no cover - environment without web3
    Web3 = None  # type: ignore[assignment]
    Account = None  # type: ignore[assignment]
    HAS_WEB3 = False

from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError,
    OnChainPendingError,
    OnChainRevertedError,
)

logger = logging.getLogger(__name__)

# Minimal ABIs — verified against contracts/contracts/EscrowPool.sol + ERC-20.
ESCROW_POOL_ABI = [
    {"type": "function", "name": "deposit", "stateMutability": "nonpayable",
     "inputs": [{"name": "amount", "type": "uint256"}], "outputs": []},
    {"type": "function", "name": "withdraw", "stateMutability": "nonpayable",
     "inputs": [{"name": "amount", "type": "uint256"}], "outputs": []},
    {"type": "function", "name": "balanceOf", "stateMutability": "view",
     "inputs": [{"name": "requester", "type": "address"}],
     "outputs": [{"name": "", "type": "uint256"}]},
]
_ERC20_ABI = [
    {"type": "function", "name": "approve", "stateMutability": "nonpayable",
     "inputs": [{"name": "spender", "type": "address"},
                {"name": "amount", "type": "uint256"}],
     "outputs": [{"name": "", "type": "bool"}]},
    {"type": "function", "name": "allowance", "stateMutability": "view",
     "inputs": [{"name": "owner", "type": "address"},
                {"name": "spender", "type": "address"}],
     "outputs": [{"name": "", "type": "uint256"}]},
    {"type": "function", "name": "balanceOf", "stateMutability": "view",
     "inputs": [{"name": "account", "type": "address"}],
     "outputs": [{"name": "", "type": "uint256"}]},
]

_RECEIPT_TIMEOUT_SECONDS = 120


class EscrowPoolClient:
    """web3.py client for EscrowPool deposit/withdraw/balance. Write calls need a
    private key; ``balance_of`` is a view and does not."""

    def __init__(
        self,
        rpc_url: str,
        escrow_pool_address: str,
        ftns_token_address: str,
        private_key: Optional[str] = None,
    ) -> None:
        if not HAS_WEB3:
            raise RuntimeError(
                "web3 package is required (pip install web3 eth-account)"
            )
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self.pool_address = Web3.to_checksum_address(escrow_pool_address)
        self.ftns_address = Web3.to_checksum_address(ftns_token_address)
        self.pool = self.web3.eth.contract(
            address=self.pool_address, abi=ESCROW_POOL_ABI)
        self.ftns = self.web3.eth.contract(
            address=self.ftns_address, abi=_ERC20_ABI)
        self._account = Account.from_key(private_key) if private_key else None

        from prsm.economy.web3.tx_lock_registry import TX_LOCK_REGISTRY
        self._tx_lock = (
            TX_LOCK_REGISTRY.get_lock(self._account.address)
            if self._account is not None else threading.Lock()
        )

    @property
    def address(self) -> Optional[str]:
        return self._account.address if self._account else None

    # ── Async surface ────────────────────────────────────────────────────────

    async def balance_of(self, requester: str) -> int:
        """The requester's ESCROW balance held inside the pool."""
        addr = Web3.to_checksum_address(requester)
        return await asyncio.to_thread(
            lambda: int(self.pool.functions.balanceOf(addr).call())
        )

    async def ftns_balance_of(self, account: str) -> int:
        """The account's WALLET FTNS balance (the token's balanceOf, distinct from
        the in-pool escrow balance). After a settlement, the provider's wallet
        balance is what rises — the e2e proof asserts on this (sp1042)."""
        addr = Web3.to_checksum_address(account)
        return await asyncio.to_thread(
            lambda: int(self.ftns.functions.balanceOf(addr).call())
        )

    async def deposit(self, amount: int) -> str:
        """Approve (only if allowance is short) then deposit ``amount`` FTNS into
        the caller's own escrow balance. Returns the deposit tx hash."""
        return await asyncio.to_thread(self._deposit_sync, int(amount))

    async def withdraw(self, amount: int) -> str:
        return await asyncio.to_thread(self._withdraw_sync, int(amount))

    # ── Sync implementations ─────────────────────────────────────────────────

    def _deposit_sync(self, amount: int) -> str:
        if not self._account:
            raise RuntimeError("private_key required for write calls (deposit)")
        if amount <= 0:
            raise ValueError(f"deposit amount must be positive (got {amount})")
        with self._tx_lock:
            owner = self._account.address
            allowance = int(
                self.ftns.functions.allowance(owner, self.pool_address).call())
            if allowance < amount:
                # Approve exactly what's needed (overwrites any short allowance).
                approve_tx = self.ftns.functions.approve(
                    self.pool_address, amount,
                ).build_transaction(self._tx_overrides())
                self._sign_send_wait(approve_tx)
            deposit_tx = self.pool.functions.deposit(amount).build_transaction(
                self._tx_overrides())
            receipt = self._sign_send_wait(deposit_tx)
        return "0x" + receipt_tx_hash(receipt)

    def _withdraw_sync(self, amount: int) -> str:
        if not self._account:
            raise RuntimeError("private_key required for write calls (withdraw)")
        if amount <= 0:
            raise ValueError(f"withdraw amount must be positive (got {amount})")
        with self._tx_lock:
            tx = self.pool.functions.withdraw(amount).build_transaction(
                self._tx_overrides())
            receipt = self._sign_send_wait(tx)
        return "0x" + receipt_tx_hash(receipt)

    # ── Helpers (mirror Web3SettlementContractClient) ─────────────────────────

    def _tx_overrides(self) -> dict:
        return {
            "from": self._account.address,
            "nonce": self.web3.eth.get_transaction_count(
                self._account.address, "pending"),
            "gasPrice": self.web3.eth.gas_price,
            "chainId": self.web3.eth.chain_id,
        }

    def _sign_send_wait(self, tx: dict):
        """Sign → broadcast → wait. Three-tier errors: BroadcastFailedError (never
        landed — safe to retry), OnChainPendingError (broadcast OK, receipt
        unknown), OnChainRevertedError (mined but reverted)."""
        signed = self.web3.eth.account.sign_transaction(tx, self._account.key)
        raw = getattr(signed, "raw_transaction", None) or getattr(
            signed, "rawTransaction", None)
        try:
            tx_hash_bytes = self.web3.eth.send_raw_transaction(raw)
        except Exception as exc:
            raise BroadcastFailedError(f"broadcast failed: {exc}") from exc

        tx_hash_hex = "0x" + tx_hash_bytes.hex()
        try:
            receipt = self.web3.eth.wait_for_transaction_receipt(
                tx_hash_bytes, timeout=_RECEIPT_TIMEOUT_SECONDS)
        except Exception as exc:
            raise OnChainPendingError(
                f"broadcast OK but receipt unknown: {exc}", tx_hash=tx_hash_hex,
            ) from exc

        if receipt.status != 1:
            raise OnChainRevertedError(f"tx reverted: {tx_hash_hex}")
        return receipt


def receipt_tx_hash(receipt) -> str:
    """Best-effort hex (no 0x) of a receipt's transactionHash; '' if absent
    (some fakes/receipts omit it — the hash is informational for the caller)."""
    th = getattr(receipt, "transactionHash", None)
    if th is None and isinstance(receipt, dict):
        th = receipt.get("transactionHash")
    if th is None:
        return ""
    return th.hex() if hasattr(th, "hex") else str(th)
