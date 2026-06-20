"""Read-only on-chain FTNS balance reader (sprint 1183).

A no-key ERC-20 ``balanceOf`` + ``decimals`` reader for the wallet API's balance display.
Reads over a default public RPC with NO private key — purely a view client. Kept separate
from ``EscrowPoolClient`` (which couples to the escrow contract + signing) so the
balance-display path has no escrow dependency.

Day-one context: the wallet API previously hardcoded a $0 balance (``_ZeroBalanceLookup``);
this is the on-chain reader that backs the real ``OnChainBalanceLookup`` in
``wallet_api``. Returns RAW wei (the caller converts to whole FTNS by ``decimals``).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

try:
    from web3 import Web3
    HAS_WEB3 = True
except ImportError:  # pragma: no cover
    Web3 = None  # type: ignore[assignment]
    HAS_WEB3 = False

logger = logging.getLogger(__name__)

# Minimal ERC-20 view ABI — balanceOf + decimals only (read-only display client).
_ERC20_VIEW_ABI = [
    {"type": "function", "name": "balanceOf", "stateMutability": "view",
     "inputs": [{"name": "account", "type": "address"}],
     "outputs": [{"name": "", "type": "uint256"}]},
    {"type": "function", "name": "decimals", "stateMutability": "view",
     "inputs": [],
     "outputs": [{"name": "", "type": "uint8"}]},
]

_DEFAULT_DECIMALS = 18


class ReadOnlyFTNSBalanceReader:
    """Reads ``balanceOf(address)`` (raw wei) + ``decimals`` for an ERC-20 token over a
    public RPC, no signing key. ``decimals`` is lazily read once + cached; on any read
    error it falls back to 18 (FTNS's actual precision). Construction never performs a
    network call — the first ``balance_wei`` / ``decimals`` access does."""

    def __init__(self, rpc_url: str, token_address: str) -> None:
        if not HAS_WEB3:
            raise RuntimeError(
                "web3 package is required (pip install web3)"
            )
        if not rpc_url or not token_address:
            raise ValueError("rpc_url and token_address are required")
        self._web3 = Web3(Web3.HTTPProvider(rpc_url))
        self._contract = self._web3.eth.contract(
            address=Web3.to_checksum_address(token_address),
            abi=_ERC20_VIEW_ABI,
        )
        self._decimals: Optional[int] = None

    @property
    def decimals(self) -> int:
        if self._decimals is None:
            try:
                self._decimals = int(self._contract.functions.decimals().call())
            except Exception as exc:  # noqa: BLE001 — degrade to FTNS's known precision
                logger.debug(
                    "FTNS decimals() read failed (%s); defaulting to %d",
                    exc, _DEFAULT_DECIMALS,
                )
                self._decimals = _DEFAULT_DECIMALS
        return self._decimals

    def balance_wei(self, address: str) -> int:
        """Raw on-chain token balance (wei) for ``address``. Raises on RPC failure — the
        caller (OnChainBalanceLookup) owns the fail-soft policy."""
        checksummed = Web3.to_checksum_address(address)
        return int(self._contract.functions.balanceOf(checksummed).call())
