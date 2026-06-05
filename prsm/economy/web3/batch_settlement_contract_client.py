"""Sprint 1021 — Web3SettlementContractClient.

Concrete web3.py implementation of the settlement ``SettlementContractClient``
Protocol (``prsm/settlement/client.py``) against the deployed
``BatchSettlementRegistry`` on Base. Until this shipped, the settlement
orchestration (``BatchSettlementClient`` + ``ReceiptAccumulator`` + Merkle) drove
commit→finalize only against a Protocol/AsyncMock — there was no concrete client,
so a signed inference receipt could never reach the chain. This closes that gap.

Mirrors the ``CompensationDistributorClient`` pattern (sync web3, ``Account``
signing, ``TX_LOCK_REGISTRY`` nonce serialization, the three-tier error model). The
Protocol is async, so each method wraps the blocking sync web3 calls in
``asyncio.to_thread`` — it satisfies the async surface without stalling the caller's
event loop.

Correctness note: the on-chain ``commitBatch`` takes ``requester`` and derives the
provider from ``msg.sender`` — so ``provider_address`` from the Protocol signature is
NOT forwarded into the contract call (it's retained only for the Protocol contract +
audit logging). The deployed registry address is in ``prsm/config/networks.py``
(``settlement_registry``); a write-capable client needs a funded settler key, which
is the only piece that can't be exercised without a live chain.

ABI is a minimal inline subset (the 4 methods + the ``BatchCommitted`` event),
verified against ``contracts/artifacts/contracts/BatchSettlementRegistry.sol/``.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional, Tuple

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

# Minimal ABI — verified against the in-repo BatchSettlementRegistry artifact.
# commitBatch takes `requester` (provider = msg.sender on-chain); batches() is the
# struct getter whose field index 7 is the uint8 status.
BATCH_SETTLEMENT_REGISTRY_ABI = [
    {
        "type": "function", "name": "commitBatch", "stateMutability": "nonpayable",
        "inputs": [
            {"name": "requester", "type": "address"},
            {"name": "merkleRoot", "type": "bytes32"},
            {"name": "receiptCount", "type": "uint256"},
            {"name": "totalValueFTNS", "type": "uint256"},
            {"name": "tierSlashRateBps", "type": "uint16"},
            {"name": "consensusGroupId", "type": "bytes32"},
            {"name": "metadataURI", "type": "string"},
        ],
        "outputs": [{"name": "batchId", "type": "bytes32"}],
    },
    {
        "type": "function", "name": "finalizeBatch", "stateMutability": "nonpayable",
        "inputs": [{"name": "batchId", "type": "bytes32"}], "outputs": [],
    },
    {
        "type": "function", "name": "isFinalizable", "stateMutability": "view",
        "inputs": [{"name": "batchId", "type": "bytes32"}],
        "outputs": [{"name": "", "type": "bool"}],
    },
    {
        "type": "function", "name": "batches", "stateMutability": "view",
        "inputs": [{"name": "batchId", "type": "bytes32"}],
        "outputs": [
            {"name": "provider", "type": "address"},
            {"name": "requester", "type": "address"},
            {"name": "merkleRoot", "type": "bytes32"},
            {"name": "receiptCount", "type": "uint256"},
            {"name": "totalValueFTNS", "type": "uint256"},
            {"name": "invalidatedValueFTNS", "type": "uint256"},
            {"name": "commitTimestamp", "type": "uint64"},
            {"name": "status", "type": "uint8"},  # index 7
            {"name": "tier_slash_rate_bps", "type": "uint16"},
            {"name": "consensus_group_id", "type": "bytes32"},
            {"name": "lookbackWindowSecondsAtCommit", "type": "uint64"},
            {"name": "totalPausedAtBatchOrigin", "type": "uint64"},
            {"name": "challengeWindowSecondsAtCommit", "type": "uint64"},
            {"name": "escrowPoolAtCommit", "type": "address"},
            {"name": "stakeBondAtCommit", "type": "address"},
            {"name": "signatureVerifierAtCommit", "type": "address"},
            {"name": "metadataURI", "type": "string"},
        ],
    },
    {
        "type": "event", "name": "BatchCommitted", "anonymous": False,
        "inputs": [
            {"name": "batchId", "type": "bytes32", "indexed": True},
            {"name": "provider", "type": "address", "indexed": True},
            {"name": "merkleRoot", "type": "bytes32", "indexed": False},
            {"name": "receiptCount", "type": "uint256", "indexed": False},
            {"name": "totalValueFTNS", "type": "uint256", "indexed": False},
            {"name": "commitTimestamp", "type": "uint64", "indexed": False},
            {"name": "metadataURI", "type": "string", "indexed": False},
        ],
    },
]

_BATCH_STATUS_FIELD_INDEX = 7  # uint8 status within the batches() struct getter
_RECEIPT_TIMEOUT_SECONDS = 120


class Web3SettlementContractClient:
    """web3.py implementation of the settlement Protocol. Write calls need a
    private key; view calls (is_finalizable / get_batch_status) do not."""

    def __init__(
        self,
        rpc_url: str,
        contract_address: str,
        private_key: Optional[str] = None,
    ) -> None:
        if not HAS_WEB3:
            raise RuntimeError(
                "web3 package is required (pip install web3 eth-account)"
            )
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self.contract_address = Web3.to_checksum_address(contract_address)
        self.contract = self.web3.eth.contract(
            address=self.contract_address,
            abi=BATCH_SETTLEMENT_REGISTRY_ABI,
        )
        self._account = Account.from_key(private_key) if private_key else None

        from prsm.economy.web3.tx_lock_registry import TX_LOCK_REGISTRY
        if self._account is not None:
            self._tx_lock = TX_LOCK_REGISTRY.get_lock(self._account.address)
        else:
            self._tx_lock = threading.Lock()

    @property
    def address(self) -> Optional[str]:
        return self._account.address if self._account else None

    # ── Async Protocol surface (wraps the blocking sync web3 calls) ──────────

    async def commit_batch(
        self,
        provider_address: str,
        requester_address: str,
        merkle_root: bytes,
        receipt_count: int,
        total_value_ftns: int,
        tier_slash_rate_bps: int,
        consensus_group_id: bytes,
        metadata_uri: str,
    ) -> Tuple[bytes, int]:
        # provider_address is intentionally NOT forwarded — the contract uses
        # msg.sender for the provider. Kept in the signature for the Protocol +
        # audit logging only.
        return await asyncio.to_thread(
            self._commit_batch_sync,
            requester_address, merkle_root, int(receipt_count),
            int(total_value_ftns), int(tier_slash_rate_bps),
            consensus_group_id, metadata_uri,
        )

    async def is_finalizable(self, batch_id: bytes) -> bool:
        return await asyncio.to_thread(
            lambda: bool(self.contract.functions.isFinalizable(batch_id).call())
        )

    async def finalize_batch(self, batch_id: bytes) -> None:
        await asyncio.to_thread(self._finalize_batch_sync, batch_id)

    async def get_batch_status(self, batch_id: bytes) -> int:
        return await asyncio.to_thread(
            lambda: int(self.contract.functions.batches(batch_id).call()[_BATCH_STATUS_FIELD_INDEX])
        )

    # ── Sync implementations ─────────────────────────────────────────────────

    def _commit_batch_sync(
        self, requester_address, merkle_root, receipt_count,
        total_value_ftns, tier_slash_rate_bps, consensus_group_id, metadata_uri,
    ) -> Tuple[bytes, int]:
        if not self._account:
            raise RuntimeError("private_key required for write calls (commitBatch)")
        with self._tx_lock:
            tx = self.contract.functions.commitBatch(
                requester_address, merkle_root, receipt_count, total_value_ftns,
                tier_slash_rate_bps, consensus_group_id, metadata_uri,
            ).build_transaction(self._tx_overrides())
            receipt = self._sign_send_wait(tx)

        logs = self.contract.events.BatchCommitted().process_receipt(receipt)
        if not logs:
            # Tx succeeded but the registry emitted no BatchCommitted — treat as a
            # revert-equivalent (safe fallback: the caller retains the receipts).
            raise OnChainRevertedError(
                "commitBatch confirmed but no BatchCommitted event was emitted"
            )
        args = logs[0]["args"]
        return bytes(args["batchId"]), int(args["commitTimestamp"])

    def _finalize_batch_sync(self, batch_id: bytes) -> None:
        if not self._account:
            raise RuntimeError("private_key required for write calls (finalizeBatch)")
        with self._tx_lock:
            tx = self.contract.functions.finalizeBatch(batch_id).build_transaction(
                self._tx_overrides()
            )
            self._sign_send_wait(tx)  # raises OnChainRevertedError on revert

    # ── Helpers (mirror CompensationDistributorClient) ───────────────────────

    def _tx_overrides(self) -> dict:
        return {
            "from": self._account.address,
            "nonce": self.web3.eth.get_transaction_count(
                self._account.address, "pending",
            ),
            "gasPrice": self.web3.eth.gas_price,
            "chainId": self.web3.eth.chain_id,
        }

    def _sign_send_wait(self, tx: dict):
        """Sign → broadcast → wait for receipt. Three-tier errors:
        BroadcastFailedError (tx never landed — safe to retry),
        OnChainPendingError (broadcast OK, receipt unknown — do NOT blind-retry, a
        commit could double-settle), OnChainRevertedError (mined but reverted)."""
        signed = self.web3.eth.account.sign_transaction(tx, self._account.key)
        raw = getattr(signed, "raw_transaction", None) or getattr(
            signed, "rawTransaction", None,
        )
        try:
            tx_hash_bytes = self.web3.eth.send_raw_transaction(raw)
        except Exception as exc:
            raise BroadcastFailedError(f"broadcast failed: {exc}") from exc

        tx_hash_hex = "0x" + tx_hash_bytes.hex()
        try:
            receipt = self.web3.eth.wait_for_transaction_receipt(
                tx_hash_bytes, timeout=_RECEIPT_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise OnChainPendingError(
                f"broadcast OK but receipt unknown: {exc}", tx_hash=tx_hash_hex,
            ) from exc

        if receipt.status != 1:
            raise OnChainRevertedError(f"tx reverted: {tx_hash_hex}")
        return receipt
