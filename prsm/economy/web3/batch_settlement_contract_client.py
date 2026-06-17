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
        "type": "function", "name": "secondsUntilFinalizable", "stateMutability": "view",
        "inputs": [{"name": "batchId", "type": "bytes32"}],
        "outputs": [{"name": "", "type": "uint256"}],
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
# sp1040 — keep each recovery getLogs window under Base public RPC's ~10k cap
# (the same ceiling sprint 542 hit; see ftns_onchain.scan_inbound_transfers_chunked).
_SCAN_MAX_WINDOW = 9_000


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

    async def seconds_until_finalizable(self, batch_id: bytes) -> int:
        """Seconds remaining in the challenge window (0 if already finalizable).
        sp1042 — gives the operator a real countdown between the two proof phases
        instead of a bare 'not finalizable yet'."""
        return await asyncio.to_thread(
            lambda: int(self.contract.functions.secondsUntilFinalizable(batch_id).call())
        )

    async def finalize_batch(self, batch_id: bytes) -> None:
        await asyncio.to_thread(self._finalize_batch_sync, batch_id)

    async def get_batch_status(self, batch_id: bytes) -> int:
        return await asyncio.to_thread(
            lambda: int(self.contract.functions.batches(batch_id).call()[_BATCH_STATUS_FIELD_INDEX])
        )

    async def get_batch(self, batch_id: bytes) -> dict:
        """sp1043 — the batch struct fields the state-file-independent finalize
        needs (provider/requester/status/value/escrow-pool snapshot). Lets a
        committed batch be recovered + finalized from chain alone."""
        return await asyncio.to_thread(self._get_batch_sync, batch_id)

    def _get_batch_sync(self, batch_id: bytes) -> dict:
        b = self.contract.functions.batches(batch_id).call()
        return {
            "provider": b[0],
            "requester": b[1],
            "merkle_root": bytes(b[2]),
            "receipt_count": int(b[3]),
            "total_value_ftns": int(b[4]),
            "commit_timestamp": int(b[6]),
            "status": int(b[7]),               # 0=NONEXISTENT 1=PENDING 2=FINALIZED 3=VOIDED
            "escrow_pool_at_commit": b[13],
        }

    async def get_committed_batch_for_tx(
        self, tx_hash: str
    ) -> Optional[Tuple[bytes, int]]:
        return await asyncio.to_thread(self._get_committed_batch_for_tx_sync, tx_hash)

    async def find_committed_batch_by_root(
        self, provider_address: str, merkle_root: bytes,
    ) -> Optional[Tuple[bytes, int]]:
        return await asyncio.to_thread(
            self._find_committed_batch_by_root_sync, provider_address, merkle_root,
        )

    def _find_committed_batch_by_root_sync(
        self, provider_address: str, merkle_root: bytes,
    ) -> Optional[Tuple[bytes, int]]:
        """sp1040 (brick 2.6) — recover a commit that LANDED on chain but whose
        local post-commit persist was lost to a crash. Scans BatchCommitted logs
        (``provider`` is an indexed topic) for one whose ``merkleRoot`` matches,
        returning that batch's (batch_id, commit_timestamp) or None.

        sp1040 review fixes:
        - #1/#6/#12 — the scan is CHUNKED into ``_SCAN_MAX_WINDOW``-block windows
          (Base public RPC caps getLogs around 10k blocks; an unbounded
          genesis-to-head request 413s and was silently swallowed, defeating the
          whole recovery mechanism). The range is bounded: from
          ``PRSM_SETTLEMENT_SCAN_FROM_BLOCK`` if set, else ``head -
          PRSM_SETTLEMENT_SCAN_LOOKBACK_BLOCKS`` (default 300k ≈ days on Base,
          comfortably longer than the challenge window — a freshly-orphaned commit
          is recovered on the first post-restart poll while it is minutes old).
        - On ANY scan error we LOG (not silently swallow) and return None so the
          caller keeps the intent and retries — distinguishable from not-landed.
        - #2/#5/#11 — if MORE THAN ONE distinct batchId matches (provider, root)
          we REFUSE to adopt (return None + warn): the root does not bind the
          requester, so an ambiguous match could adopt the wrong batch. A single
          unambiguous match is safe (the event is authoritative for what this
          provider committed)."""
        import os
        provider = Web3.to_checksum_address(provider_address)
        target = bytes(merkle_root)
        try:
            head = self.web3.eth.block_number
        except Exception as exc:  # noqa: BLE001
            logger.warning("settlement recovery scan: block_number failed (%s) "
                           "— keeping intent", exc)
            return None
        env_from = (os.environ.get("PRSM_SETTLEMENT_SCAN_FROM_BLOCK", "") or "").strip()
        if env_from:
            from_block = int(env_from)
        else:
            lookback = int(
                os.environ.get("PRSM_SETTLEMENT_SCAN_LOOKBACK_BLOCKS", "") or 300_000
            )
            from_block = max(0, head - lookback)

        matches: dict = {}  # batch_id -> commit_ts (dedup distinct batchIds)
        event = self.contract.events.BatchCommitted()
        start = from_block
        while start <= head:
            end = min(start + _SCAN_MAX_WINDOW - 1, head)
            try:
                flt = event.create_filter(
                    from_block=start, to_block=end,
                    argument_filters={"provider": provider},
                )
                entries = flt.get_all_entries()
            except Exception as exc:  # noqa: BLE001 - a capped/failed window is inconclusive
                logger.warning(
                    "settlement recovery scan failed on window %d-%d (%s) — "
                    "keeping intent, will retry", start, end, exc,
                )
                return None
            for e in entries:
                if bytes(e["args"]["merkleRoot"]) == target:
                    matches[bytes(e["args"]["batchId"])] = int(
                        e["args"]["commitTimestamp"])
            if end == head:
                break
            start = end + 1

        if not matches:
            return None
        if len(matches) > 1:
            logger.warning(
                "settlement recovery: %d distinct batchIds share root %s for "
                "provider %s — refusing ambiguous adoption (keeping intent for "
                "operator review)", len(matches), target.hex()[:12], provider,
            )
            return None
        (batch_id, commit_ts), = matches.items()
        return batch_id, commit_ts

    async def enumerate_committed_batches(
        self, from_block: int, to_block: int, *, provider: Optional[str] = None,
    ) -> "list":
        """sp1139 (Brick E.1) — READ-ONLY enumerate the committed-batch worklist an
        observer audits. Scans ``BatchCommitted`` logs over [from_block, to_block] in
        ``_SCAN_MAX_WINDOW``-block windows (the same Base public-RPC getLogs cap the
        recovery scan respects), optionally filtered to one ``provider`` (an indexed
        topic). For each distinct ``batchId`` it reads the on-chain ``Batch`` struct
        (``batches(batch_id).call()``) for requester (idx 1) / merkleRoot (idx 2) /
        consensus_group_id (idx 9) and builds an ``ObservedBatch`` — the TRUSTED anchor
        the observer cross-checks fetched receipt sets against.

        ``cid`` is left None (the loop fills it from the untrusted ad index). This is
        pure-read: no tx, no signing, no state change. The blocking web3 calls run in
        ``asyncio.to_thread`` like the sibling read methods."""
        return await asyncio.to_thread(
            self._enumerate_committed_batches_sync, int(from_block), int(to_block),
            provider,
        )

    def _enumerate_committed_batches_sync(
        self, from_block: int, to_block: int, provider: Optional[str],
    ) -> "list":
        # Imported here (not at module top) to keep this client importable in the
        # pure/offline test path where settlement_audit_engine's deps may differ.
        from prsm.settlement.settlement_audit_engine import ObservedBatch

        provider_cs = Web3.to_checksum_address(provider) if provider else None
        argument_filters = {"provider": provider_cs} if provider_cs else None

        # Distinct batchId -> (provider from the log topic). Dedup so a batch that emits
        # twice (it won't, but defensively) is read once.
        seen: "dict[bytes, str]" = {}
        event = self.contract.events.BatchCommitted()
        start = int(from_block)
        end_bound = int(to_block)
        while start <= end_bound:
            end = min(start + _SCAN_MAX_WINDOW - 1, end_bound)
            flt = event.create_filter(
                from_block=start, to_block=end,
                argument_filters=argument_filters,
            )
            for e in flt.get_all_entries():
                args = e["args"]
                bid = bytes(args["batchId"])
                if bid not in seen:
                    seen[bid] = args["provider"]
            if end == end_bound:
                break
            start = end + 1

        observed = []
        for bid, log_provider in seen.items():
            b = self.contract.functions.batches(bid).call()
            observed.append(ObservedBatch(
                batch_id=bid,
                provider_address=b[0] if b[0] else log_provider,
                requester_address=b[1],
                merkle_root=bytes(b[2]),
                consensus_group_id=bytes(b[9]),
                cid=None,
            ))
        return observed

    def _get_committed_batch_for_tx_sync(self, tx_hash: str) -> Optional[Tuple[bytes, int]]:
        """Recover (batch_id, commit_timestamp) from a commitBatch tx by parsing the
        BatchCommitted log in its receipt. Returns None when the tx is unknown / not
        yet mined (get_transaction_receipt raises or returns None), reverted
        (status != 1), or emitted no BatchCommitted — i.e. anything other than a
        confirmed landed commit. The reconcile caller leaves those quarantined."""
        try:
            receipt = self.web3.eth.get_transaction_receipt(tx_hash)
        except Exception:  # noqa: BLE001 - TransactionNotFound etc. → not landed
            return None
        if receipt is None or getattr(receipt, "status", 0) != 1:
            return None
        logs = self.contract.events.BatchCommitted().process_receipt(receipt)
        if not logs:
            return None
        args = logs[0]["args"]
        return bytes(args["batchId"]), int(args["commitTimestamp"])

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
