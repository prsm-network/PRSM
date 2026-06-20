"""On-chain TESTNET FTNS faucet (sprint 1190, day-one blocker #3).

Dispenses REAL on-chain testnet FTNS (an ERC-20 ``transfer`` from a faucet wallet to a
new user's address) so the day-one front door works end to end on Base Sepolia: the user
sees a real balance (sp1183) and can deposit + pay_and_infer (sp1189).

HARD SAFETY: the faucet is TESTNET-ONLY. ``dispense`` refuses (raises
FaucetMainnetRefusedError) on any chainId other than Base Sepolia (84532), so it can NEVER
give away real-value mainnet FTNS — fail-closed, defense-in-depth with the build-time
config gate. Mirrors the EscrowPoolClient web3 pattern (explicit gas per the sp1047
replica-lag lesson; three-tier broadcast/pending/reverted errors).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

try:
    from web3 import Web3
    from eth_account import Account
    HAS_WEB3 = True
except ImportError:  # pragma: no cover
    Web3 = None  # type: ignore[assignment]
    Account = None  # type: ignore[assignment]
    HAS_WEB3 = False

from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError,
    OnChainPendingError,
    OnChainRevertedError,
)

logger = logging.getLogger(__name__)

BASE_SEPOLIA_CHAIN_ID = 84532
_TRANSFER_GAS = 120_000          # ERC-20 transfer headroom (explicit → no estimateGas)
_RECEIPT_TIMEOUT_SECONDS = 120
_DEFAULT_DECIMALS = 18

_ERC20_FAUCET_ABI = [
    {"type": "function", "name": "balanceOf", "stateMutability": "view",
     "inputs": [{"name": "account", "type": "address"}],
     "outputs": [{"name": "", "type": "uint256"}]},
    {"type": "function", "name": "decimals", "stateMutability": "view",
     "inputs": [], "outputs": [{"name": "", "type": "uint8"}]},
    {"type": "function", "name": "transfer", "stateMutability": "nonpayable",
     "inputs": [{"name": "to", "type": "address"}, {"name": "amount", "type": "uint256"}],
     "outputs": [{"name": "", "type": "bool"}]},
]


class FaucetMainnetRefusedError(RuntimeError):
    """Raised when a dispense is attempted on a non-Base-Sepolia chain — the kill-switch
    that guarantees the faucet never gives away real-value FTNS."""


class OnChainFTNSFaucet:
    """Signs an ERC-20 ``transfer`` of testnet FTNS from the faucet wallet to a recipient.
    web3/account/contract are injectable for tests."""

    def __init__(
        self,
        *,
        rpc_url: str,
        token_address: str,
        faucet_private_key: str,
        expected_chain_id: int = BASE_SEPOLIA_CHAIN_ID,
        _web3: Any = None,
        _contract: Any = None,
        _account: Any = None,
    ) -> None:
        self._expected_chain_id = int(expected_chain_id)
        self._decimals_cache: Optional[int] = None
        if _web3 is not None and _contract is not None and _account is not None:
            self.web3 = _web3
            self._contract = _contract
            self._account = _account
            return
        if not HAS_WEB3:
            raise RuntimeError("web3 package is required (pip install web3 eth-account)")
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self._contract = self.web3.eth.contract(
            address=Web3.to_checksum_address(token_address), abi=_ERC20_FAUCET_ABI)
        self._account = Account.from_key(faucet_private_key)

    @property
    def faucet_address(self) -> str:
        return self._account.address

    @property
    def chain_id(self) -> int:
        return int(self.web3.eth.chain_id)

    @property
    def decimals(self) -> int:
        if self._decimals_cache is None:
            try:
                self._decimals_cache = int(self._contract.functions.decimals().call())
            except Exception:  # noqa: BLE001
                self._decimals_cache = _DEFAULT_DECIMALS
        return self._decimals_cache

    def balance_of_wei(self, address: str) -> int:
        return int(self._contract.functions.balanceOf(address).call())

    def dispense(self, recipient: str, amount_wei: int) -> str:
        """Transfer ``amount_wei`` FTNS to ``recipient``; returns the tx hash. HARD GATE:
        refuses unless the LIVE chainId is Base Sepolia (the kill-switch)."""
        live = self.chain_id
        if live != self._expected_chain_id:
            raise FaucetMainnetRefusedError(
                f"faucet refuses to dispense on chainId {live} (testnet-only; expected "
                f"{self._expected_chain_id} = Base Sepolia) — never gives away real FTNS"
            )
        if amount_wei <= 0:
            raise ValueError(f"amount_wei must be positive (got {amount_wei})")
        tx = self._contract.functions.transfer(recipient, int(amount_wei)).build_transaction({
            "from": self._account.address,
            "nonce": self.web3.eth.get_transaction_count(self._account.address, "pending"),
            "gasPrice": self.web3.eth.gas_price,
            "chainId": live,
            "gas": _TRANSFER_GAS,  # explicit → skip estimateGas (sp1047 replica-lag lesson)
        })
        signed = self.web3.eth.account.sign_transaction(tx, self._account.key)
        raw = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction", None)
        try:
            tx_hash_bytes = self.web3.eth.send_raw_transaction(raw)
        except Exception as exc:  # noqa: BLE001
            raise BroadcastFailedError(f"faucet broadcast failed: {exc}") from exc
        tx_hash_hex = "0x" + tx_hash_bytes.hex()
        try:
            receipt = self.web3.eth.wait_for_transaction_receipt(
                tx_hash_bytes, timeout=_RECEIPT_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001
            raise OnChainPendingError(
                f"faucet broadcast OK but receipt unknown: {exc}", tx_hash=tx_hash_hex,
            ) from exc
        if receipt.status != 1:
            raise OnChainRevertedError(f"faucet transfer reverted: {tx_hash_hex}")
        return tx_hash_hex


def build_onchain_faucet_or_none(env: Optional[dict] = None) -> Optional[OnChainFTNSFaucet]:
    """Build the testnet faucet from PRSM_FAUCET_PRIVATE_KEY + the active network, or None.

    Returns None (no faucet) when: no faucet key is set, the configured network is NOT
    Base Sepolia (NEVER build a faucet on mainnet — the config-time half of the
    kill-switch), or web3 is unavailable. Fail-closed by construction."""
    import os
    e = env if env is not None else os.environ
    key = (e.get("PRSM_FAUCET_PRIVATE_KEY") or "").strip()
    if not key:
        return None
    if not HAS_WEB3:
        logger.warning("PRSM_FAUCET_PRIVATE_KEY set but web3 is unavailable; faucet off.")
        return None
    try:
        from prsm.config.networks import resolve_endpoints
        ep = resolve_endpoints()
        if ep.chain_id != BASE_SEPOLIA_CHAIN_ID:
            logger.warning(
                "FTNS faucet is TESTNET-ONLY; configured network chainId=%s is not Base "
                "Sepolia (%s) — faucet disabled (never dispenses real FTNS).",
                ep.chain_id, BASE_SEPOLIA_CHAIN_ID,
            )
            return None
        if not ep.ftns_token:
            logger.warning("no FTNS token address resolved for the faucet; disabled.")
            return None
        return OnChainFTNSFaucet(
            rpc_url=ep.rpc_url, token_address=ep.ftns_token, faucet_private_key=key,
            expected_chain_id=BASE_SEPOLIA_CHAIN_ID,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("on-chain faucet construction failed (%s); faucet off.", exc)
        return None
