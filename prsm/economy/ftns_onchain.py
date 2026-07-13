"""
On-Chain FTNS Ledger Adapter
============================

Mirrors the LocalLedger to the real FTNS contract on Base mainnet.
When the node executes compute jobs and escrow transactions, this
adapter broadcasts real FTNS transfers on-chain.

Usage in PRSMNode:
    self.ftns_ledger = OnChainFTNSLedger(
        node_id=self.identity.node_id,
        wallet_private_key=os.getenv("FTNS_WALLET_PRIVATE_KEY"),
        contract_address="0x5276...",
        rpc_url=Base_RPC,
    )
    await self.ftns_ledger.initialize()

This creates a single Web3 connection that:
- Reads real on-chain FTNS balances
- Transfers FTNS on escrow lock/release/refund
- Tracks tx hashes for auditability
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

try:
    from web3 import Web3
    from web3.exceptions import TimeExhausted
    from eth_account import Account
    from eth_utils import to_checksum_address
    HAS_WEB3 = True
except ImportError:
    HAS_WEB3 = False
    Web3 = None
    Account = None
    to_checksum_address = None
    TimeExhausted = Exception

logger = logging.getLogger(__name__)

# ── Dynamic Gas Pricing (Ring 6) ──────────────────────────────────
DEFAULT_GAS_GWEI = 5
MAX_GAS_GWEI = 50


class RPCFailover:
    """Rotates through RPC endpoints on failure."""

    def __init__(self, urls: list):
        self._urls = urls if urls else ["https://mainnet.base.org"]
        self._index = 0

    @property
    def current_url(self) -> str:
        return self._urls[self._index % len(self._urls)]

    def mark_failed(self) -> None:
        self._index = (self._index + 1) % len(self._urls)

    def mark_success(self) -> None:
        pass


def estimate_gas_price(w3, multiplier: float = 1.2, max_gwei: int = MAX_GAS_GWEI) -> int:
    """Get dynamic gas price from network, with fallback and cap. Returns wei."""
    try:
        network_gas = w3.eth.gas_price
        adjusted = int(network_gas * multiplier)
        cap = max_gwei * 1_000_000_000
        return min(adjusted, cap)
    except Exception:
        return DEFAULT_GAS_GWEI * 1_000_000_000


# ── Minimal ERC20 ABI ──────────────────────────────────────────────
_ERC20_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "name",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [{"name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": False,
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "value", "type": "uint256"},
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
    },
    # Sprint 512 — Transfer event for inbound-scan endpoint.
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "from", "type": "address"},
            {"indexed": True, "name": "to", "type": "address"},
            {"indexed": False, "name": "value", "type": "uint256"},
        ],
        "name": "Transfer",
        "type": "event",
    },
]

# ── Network config (resolved at module load via PRSM_NETWORK) ─────────────
# T6 (post-2026-05-07): historically these were hardcoded mainnet
# defaults, which meant `prsm join-testnet`'s env-var bundle silently
# bypassed the constructor defaults (testnet env sets
# BASE_SEPOLIA_RPC_URL + FTNS_TOKEN_ADDRESS, not BASE_RPC_URL +
# FTNS_CONTRACT_ADDRESS). Now resolved through `prsm.config.networks`
# which honors PRSM_NETWORK + per-field overrides.
from prsm.config.networks import resolve_endpoints

_resolved = resolve_endpoints()
FTNS_CONTRACT_ADDRESS = _resolved.ftns_token or "0x5276a3756C85f2E9e46f6D34386167a209aa16e5"
BASE_RPC_URL = _resolved.rpc_url
BASE_CHAIN_ID = _resolved.chain_id


def scan_inbound_transfers(
    contract,
    recipient: str,
    from_block: int,
    to_block,
) -> list:
    """Sprint 512: scan ERC-20 Transfer events where to=recipient.

    Uses Web3 events get_logs (stateless) rather than
    create_filter (stateful) because public RPC providers
    like base.org reject stateful filters with
    "filter not found".

    Returns list of dicts:
      {block_number, tx_hash, from_address, to_address,
       amount_ftns}
    """
    from eth_utils import to_checksum_address
    recipient_checksum = to_checksum_address(recipient)
    events_obj = contract.events.Transfer
    # Prefer get_logs (stateless) — falls back to
    # create_filter for code paths that mock the older API.
    if hasattr(events_obj, "get_logs"):
        logs = events_obj.get_logs(
            from_block=from_block,
            to_block=to_block,
            argument_filters={"to": recipient_checksum},
        )
    else:
        filt = events_obj.create_filter(
            from_block=from_block,
            to_block=to_block,
            argument_filters={"to": recipient_checksum},
        )
        logs = filt.get_all_entries()
    out = []
    for log in logs:
        tx_hash_bytes = getattr(log, "transactionHash", b"")
        if hasattr(tx_hash_bytes, "hex"):
            tx_hash = tx_hash_bytes.hex()
        else:
            tx_hash = str(tx_hash_bytes)
        if not tx_hash.startswith("0x"):
            tx_hash = "0x" + tx_hash
        args = log.args
        try:
            from_addr = args["from"]
            to_addr = args["to"]
            amount_wei = args["value"]
        except (KeyError, TypeError):
            from_addr = getattr(args, "from", None) or ""
            to_addr = getattr(args, "to", None) or ""
            amount_wei = getattr(args, "value", 0) or 0
        out.append({
            "block_number": getattr(log, "blockNumber", 0),
            "tx_hash": tx_hash,
            "from_address": from_addr,
            "to_address": to_addr,
            "amount_ftns": amount_wei / 1e18,
        })
    return out


def scan_inbound_transfers_chunked(
    contract,
    recipient: str,
    from_block: int,
    to_block: int,
    max_window: int = 9_000,
) -> list:
    """Sprint 542 — F20 fix: chunk a wide block range into sub-windows
    so each underlying ``eth_getLogs`` call fits inside the public
    RPC's per-request payload limit (Base mainnet.base.org caps
    around 10k blocks).

    ``InboundMonitor`` doesn't hit the limit because it scans the
    per-tick delta only. The user-facing endpoint defaulted to a
    100k-block lookback that exceeded the cap on every call, producing
    ``413 Client Error: Payload Too Large``.

    Windows are inclusive on both ends; tail window may be smaller
    than ``max_window``. Results are returned in ascending block order
    (the underlying provider already returns each chunk ordered).
    """
    if to_block < from_block:
        return []
    if max_window < 1:
        raise ValueError("max_window must be >= 1")
    out: list = []
    start = from_block
    while start <= to_block:
        end = min(start + max_window - 1, to_block)
        out.extend(scan_inbound_transfers(
            contract,
            recipient=recipient,
            from_block=start,
            to_block=end,
        ))
        if end == to_block:
            break
        start = end + 1
    return out


# Base Sepolia's public RPC (sepolia.base.org) caps eth_getLogs at 2000 blocks per
# request — far tighter than Base mainnet's ~10k. sp1257 — the inbound-transfer scan
# must pick a window that fits the ACTIVE chain's cap or it 400s with
# "query exceeds max block range 2000" (seen in `wallet info` on testnet).
_BASE_SEPOLIA_CHAIN_ID = 84532
_GETLOGS_WINDOW_SEPOLIA = 1_900   # safety margin under the 2000-block cap
_GETLOGS_WINDOW_DEFAULT = 9_000   # Base mainnet (~10k cap)


def getlogs_window_for_chain(chain_id) -> int:
    """sp1257 — the safe ``eth_getLogs`` block-window for a chain's public-RPC cap.

    Base Sepolia (chainId 84532) caps at 2000 blocks/request; everything else uses
    the wider ~9000 default. Pass to ``scan_inbound_transfers_chunked(max_window=...)``
    so the inbound scan never over-runs the cap. Fails SAFE (smallest window) on a
    non-int chain_id."""
    try:
        if int(chain_id) == _BASE_SEPOLIA_CHAIN_ID:
            return _GETLOGS_WINDOW_SEPOLIA
        return _GETLOGS_WINDOW_DEFAULT
    except (TypeError, ValueError):
        return _GETLOGS_WINDOW_SEPOLIA


def _gas_status_for_eth(eth: float) -> str:
    """Pure helper — shared between startup log, endpoint,
    /health/detailed, and the periodic monitor. One source
    of truth for threshold boundaries."""
    if eth < 0.0001:
        return "critical"
    if eth < 0.0005:
        return "low"
    return "ok"


class GasStatusMonitor:
    """Periodic background sampler that logs ONLY on
    status transitions (ok ↔ low ↔ critical) so operators
    get continuous signal without log spam.
    """

    SEVERITY = {"ok": 0, "low": 1, "critical": 2}

    def __init__(self, ledger: "OnChainFTNSLedger",
                 interval_seconds: float = 60.0,
                 webhook_deliverer: Optional[Any] = None,
                 webhook_url: Optional[str] = None,
                 webhook_secret: Optional[str] = None):
        self.ledger = ledger
        self.interval_seconds = interval_seconds
        self._last_status: Optional[str] = None
        self._webhook_deliverer = webhook_deliverer
        self._webhook_url = webhook_url
        self._webhook_secret = webhook_secret

    async def _fire_webhook(
        self, prev: str, new: str, eth: float,
    ) -> None:
        if (
            self._webhook_deliverer is None
            or not self._webhook_url
        ):
            return
        import time as _time
        payload = {
            "event": "gas.transition",
            "node_id": getattr(
                self.ledger, "node_id", "unknown",
            ),
            "address": self.ledger._connected_address,
            "previous_status": prev,
            "new_status": new,
            "eth_balance": eth,
            "timestamp": _time.time(),
        }
        try:
            await self._webhook_deliverer.deliver(
                url=self._webhook_url,
                event="gas.transition",
                payload=payload,
                secret=self._webhook_secret,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "gas-monitor webhook delivery raised "
                "(non-fatal): %s", exc,
            )

    async def _tick_async(self) -> None:
        """Async tick variant that supports webhook firing.

        Used by run_forever() (which already runs in an event
        loop). The sync _tick_sync() remains for unit tests
        that need synchronous semantics.
        """
        prev_status = self._last_status
        self._tick_sync()
        new_status = self._last_status
        if (
            prev_status is not None
            and new_status is not None
            and prev_status != new_status
        ):
            w3 = getattr(self.ledger, "w3", None)
            addr = getattr(
                self.ledger, "_connected_address", None,
            )
            if w3 is not None and addr is not None:
                try:
                    wei = w3.eth.get_balance(addr)
                    await self._fire_webhook(
                        prev_status, new_status, wei / 1e18,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "gas-monitor webhook prep failed "
                        "(non-fatal): %s", exc,
                    )

    def _tick_sync(self) -> None:
        """Single tick. Synchronous helper for unit tests
        + reused inside the async run_forever loop."""
        w3 = getattr(self.ledger, "w3", None)
        addr = getattr(self.ledger, "_connected_address", None)
        if w3 is None or addr is None:
            return
        try:
            wei = w3.eth.get_balance(addr)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "gas-monitor tick failed (non-fatal): %s",
                exc,
            )
            return
        eth = wei / 1e18
        new_status = _gas_status_for_eth(eth)
        if self._last_status is None:
            # First tick is the baseline; don't log it
            # since sprint-504's startup log already did.
            self._last_status = new_status
            return
        if new_status == self._last_status:
            return
        prev = self._last_status
        addr_short = addr[:10] + "…"
        if new_status == "critical":
            logger.error(
                "Operator gas transition %s → critical: "
                "%.10f ETH on %s. Top up NOW — broadcasts "
                "will start failing.",
                prev, eth, addr_short,
            )
        elif new_status == "low" and (
            self.SEVERITY[new_status]
            > self.SEVERITY[prev]
        ):
            logger.warning(
                "Operator gas transition %s → low: %.10f "
                "ETH on %s. Plan to top up soon.",
                prev, eth, addr_short,
            )
        elif (
            self.SEVERITY[new_status]
            < self.SEVERITY[prev]
        ):
            logger.info(
                "Operator gas recovered %s → %s: %.10f "
                "ETH on %s.",
                prev, new_status, eth, addr_short,
            )
        else:
            logger.info(
                "Operator gas transition %s → %s: %.10f "
                "ETH on %s.",
                prev, new_status, eth, addr_short,
            )
        self._last_status = new_status

    async def run_forever(self) -> None:
        """Async loop suitable for asyncio.create_task()."""
        import asyncio as _aio
        while True:
            try:
                await self._tick_async()
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "gas-monitor outer exception "
                    "(non-fatal): %s", exc,
                )
            try:
                await _aio.sleep(self.interval_seconds)
            except _aio.CancelledError:
                raise


class InboundCheckpointStore:
    """Sprint 543 — SQLite K/V keyed by recipient address that
    persists ``InboundMonitor`` 's last-scanned-block across daemon
    restart. Pre-sprint, an in-memory-only checkpoint meant any
    inbound FTNS arriving during a daemon outage was never credited
    to the off-chain wallet (Pattern A bridge correctness gap).

    Schema is intentionally minimal: ``(address TEXT PRIMARY KEY,
    last_scanned_block INTEGER NOT NULL)`` — one row per recipient.
    File-mode (``db_path != ":memory:"``) creates the parent dir if
    needed, matching the sprint-501 ``onchain_tx.db`` policy.
    """

    _SCHEMA = (
        "CREATE TABLE IF NOT EXISTS inbound_checkpoints ("
        "  address TEXT PRIMARY KEY,"
        "  last_scanned_block INTEGER NOT NULL"
        ")"
    )

    # Sprint 544: persistent credit-side dedup. Pre-sprint, the
    # InboundMonitor's _credited_tx_hashes set was in-memory only.
    # Sprint 543's checkpoint persistence made this a regression
    # surface: restart catch-up scans can re-emit a Transfer event
    # that was already credited in the previous run. Persistent
    # (address, tx_hash) row gives correctness across restart.
    _SCHEMA_CREDITED = (
        "CREATE TABLE IF NOT EXISTS credited_deposits ("
        "  address TEXT NOT NULL,"
        "  tx_hash TEXT NOT NULL,"
        "  PRIMARY KEY (address, tx_hash)"
        ")"
    )

    def __init__(self, db_path: str):
        import sqlite3
        from pathlib import Path
        self._db_path = db_path
        if db_path != ":memory:":
            parent = Path(db_path).parent
            if str(parent) and not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)
        # ``check_same_thread=False`` so the monitor's async loop and
        # tests can share the connection without sqlite's default
        # thread-affinity check biting us. We serialize through a
        # single connection — no concurrent writes from this class.
        self._conn = sqlite3.connect(
            db_path, check_same_thread=False,
        )
        self._conn.execute(self._SCHEMA)
        self._conn.execute(self._SCHEMA_CREDITED)
        self._conn.commit()

    def get_last_scanned_block(self, address: str) -> Optional[int]:
        row = self._conn.execute(
            "SELECT last_scanned_block FROM inbound_checkpoints "
            "WHERE address = ?",
            (address,),
        ).fetchone()
        return row[0] if row else None

    def set_last_scanned_block(
        self, address: str, block_number: int,
    ) -> None:
        self._conn.execute(
            "INSERT INTO inbound_checkpoints "
            "(address, last_scanned_block) VALUES (?, ?) "
            "ON CONFLICT(address) DO UPDATE SET "
            "last_scanned_block = excluded.last_scanned_block",
            (address, int(block_number)),
        )
        self._conn.commit()

    def has_credited_tx(self, address: str, tx_hash: str) -> bool:
        """Sprint 544: True iff this (address, tx_hash) tuple was
        previously passed to ``mark_credited``. Survives restart."""
        row = self._conn.execute(
            "SELECT 1 FROM credited_deposits "
            "WHERE address = ? AND tx_hash = ? LIMIT 1",
            (address, tx_hash),
        ).fetchone()
        return row is not None

    def mark_credited(self, address: str, tx_hash: str) -> None:
        """Sprint 544: idempotent insert — calling twice is a no-op."""
        self._conn.execute(
            "INSERT OR IGNORE INTO credited_deposits "
            "(address, tx_hash) VALUES (?, ?)",
            (address, tx_hash),
        )
        self._conn.commit()


class InboundMonitor:
    """Sprint 514: periodic background poller for inbound FTNS.

    Pull complement (sprint 512 endpoint + sprint 513 CLI) gives
    operators on-demand visibility. This adds the push signal:
    on each tick scan for new Transfer events in
    (last_scanned_block, current_block], log each + fire webhook
    (event: "ftns.inbound").

    First tick records the current block as baseline without
    scanning — sprint-512's pull surface covers historical scan.
    """

    def __init__(self, ledger, interval_seconds: float = 60.0,
                 webhook_deliverer=None,
                 webhook_url=None, webhook_secret=None,
                 local_ledger=None,
                 checkpoint_store: Optional["InboundCheckpointStore"] = None,
                 max_catchup_blocks: int = 100_000):
        self.ledger = ledger
        self.interval_seconds = interval_seconds
        self._webhook_deliverer = webhook_deliverer
        self._webhook_url = webhook_url
        self._webhook_secret = webhook_secret
        # Sprint 540 Pattern A: local_ledger reference so detected
        # inbound transfers credit the linked off-chain wallet
        # automatically. Optional — when None, monitor still logs
        # + fires webhook but doesn't credit anywhere.
        self._local_ledger = local_ledger
        # Sprint 540: track already-credited tx_hashes within this
        # process to dedup on retries. Persistence across restart
        # is provided by sprint-501's onchain_tx.db (we check the
        # status before crediting).
        self._credited_tx_hashes: set = set()
        # Sprint 543: optional persistent checkpoint store. When
        # provided, ``_last_scanned_block`` is loaded from it on
        # startup so catch-up scans recover deposits that arrived
        # during daemon downtime. ``max_catchup_blocks`` clamps the
        # restart catch-up window — chunking (sprint 542) handles
        # the still-wide range RPC-side.
        self._checkpoint_store = checkpoint_store
        self._max_catchup_blocks = max_catchup_blocks
        self._last_scanned_block: Optional[int] = None

    async def _fire_webhook(self, transfer: dict) -> None:
        if (
            self._webhook_deliverer is None
            or not self._webhook_url
        ):
            return
        import time as _time
        payload = {
            "event": "ftns.inbound",
            "node_id": getattr(
                self.ledger, "node_id", "unknown",
            ),
            "recipient": self.ledger._connected_address,
            "from_address": transfer.get("from_address"),
            "tx_hash": transfer.get("tx_hash"),
            "block_number": transfer.get("block_number"),
            "amount_ftns": transfer.get("amount_ftns"),
            "timestamp": _time.time(),
        }
        try:
            await self._webhook_deliverer.deliver(
                url=self._webhook_url,
                event="ftns.inbound",
                payload=payload,
                secret=self._webhook_secret,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "inbound-monitor webhook raised "
                "(non-fatal): %s", exc,
            )

    async def _tick_async(self) -> None:
        w3 = getattr(self.ledger, "w3", None)
        token = getattr(self.ledger, "_token", None)
        addr = getattr(
            self.ledger, "_connected_address", None,
        )
        if w3 is None or token is None or addr is None:
            return
        try:
            current_block = w3.eth.block_number
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "inbound-monitor block_number failed "
                "(non-fatal): %s", exc,
            )
            return
        # Sprint 543: lazy-load persisted checkpoint on first tick.
        # When the store has a value for our address, we resume scan
        # coverage from there (catching up on deposits that arrived
        # while the daemon was down). Otherwise behave like sprint
        # 514: record current_block as the baseline + no scan.
        if (
            self._last_scanned_block is None
            and self._checkpoint_store is not None
        ):
            persisted = self._checkpoint_store.get_last_scanned_block(
                addr,
            )
            if persisted is not None:
                self._last_scanned_block = persisted
        if self._last_scanned_block is None:
            self._last_scanned_block = current_block
            if self._checkpoint_store is not None:
                self._checkpoint_store.set_last_scanned_block(
                    addr, current_block,
                )
            return
        if current_block <= self._last_scanned_block:
            return
        # Sprint 543: clamp the catch-up window so a very-stale
        # checkpoint doesn't trigger an unbounded scan on restart.
        # Sprint 542's chunker further splits the (still-large)
        # window into RPC-sized sub-windows.
        scan_from = self._last_scanned_block + 1
        scan_to = current_block
        if scan_to - scan_from + 1 > self._max_catchup_blocks:
            clamped_from = scan_to - self._max_catchup_blocks + 1
            logger.warning(
                "inbound-monitor catch-up clamped: "
                "would have scanned %d blocks since checkpoint %d, "
                "scanning last %d blocks instead (max_catchup_blocks=%d). "
                "Older missed deposits require manual /wallet/"
                "transactions/onchain/inbound replay.",
                scan_to - scan_from + 1, self._last_scanned_block,
                self._max_catchup_blocks, self._max_catchup_blocks,
            )
            scan_from = clamped_from
        try:
            transfers = scan_inbound_transfers_chunked(
                token,
                recipient=addr,
                from_block=scan_from,
                to_block=scan_to,
            )
        except Exception as exc:  # noqa: BLE001
            # Sprint 545: do NOT advance the checkpoint on scan
            # failure. The pre-sprint behavior (set _last_scanned_block
            # = current_block then return) silently skipped the
            # failed window forever, dropping any Transfer events in
            # blocks 101..current that the scan was trying to cover.
            # By leaving the checkpoint at its pre-tick value, the
            # next tick retries the same window — idempotent under
            # sprint 544's persistent dedup. Sustained RPC outage
            # keeps retrying; transient outage recovers on next tick.
            logger.warning(
                "inbound-monitor scan failed for blocks %d-%d "
                "(checkpoint preserved at %d, next tick retries): %s",
                scan_from, scan_to, self._last_scanned_block, exc,
            )
            return
        for t in transfers:
            logger.info(
                "Inbound FTNS detected: %.6f from %s "
                "(block %s, tx %s)",
                t.get("amount_ftns", 0),
                (t.get("from_address") or "?")[:10] + "…",
                t.get("block_number"),
                (t.get("tx_hash") or "?")[:18] + "…",
            )
            # Sprint 540 Pattern A: credit off-chain wallet if the
            # sender is a linked address (deposit flow). Runs BEFORE
            # webhook so listeners see consistent state.
            try:
                await self._credit_deposit(t)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "inbound-monitor credit failed for tx %s "
                    "(non-fatal — webhook still fires): %s",
                    (t.get("tx_hash") or "?")[:18], exc,
                )
            try:
                await self._fire_webhook(t)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "inbound-monitor webhook prep "
                    "raised (non-fatal): %s", exc,
                )
        self._last_scanned_block = current_block
        # Sprint 543: persist after every tick that scanned anything.
        # If the daemon crashes mid-tick the deposit-side dedup
        # (sprint 540 _credited_tx_hashes + sprint 501 onchain_tx.db
        # status check) still prevents double-credit on the next boot
        # — the checkpoint just advances coverage forward.
        if self._checkpoint_store is not None:
            try:
                self._checkpoint_store.set_last_scanned_block(
                    addr, current_block,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "inbound-monitor checkpoint persist failed "
                    "(non-fatal — next tick retries): %s", exc,
                )

    async def _credit_deposit(self, transfer: dict) -> None:
        """Sprint 540 Pattern A: credit off-chain wallet on detected
        inbound from a linked ETH address. No-op if local_ledger
        wasn't passed at construction time."""
        if self._local_ledger is None:
            return
        from_addr = transfer.get("from_address")
        amount = transfer.get("amount_ftns", 0) or 0
        tx_hash = transfer.get("tx_hash") or ""
        if not from_addr or amount <= 0 or not tx_hash:
            return
        # Sprint 544: persistent dedup. In-memory fast-path first
        # (covers retries within a process); SQLite-backed check
        # second (covers restart catch-up scans introduced by
        # sprint 543's checkpoint persistence — without this, a
        # Transfer event already credited in v1 would be re-credited
        # by v2 because the in-memory set is empty on boot).
        if tx_hash in self._credited_tx_hashes:
            return
        # Determine the recipient address for the dedup key. Use the
        # ledger's connected address (= the monitor's scan target) so
        # the dedup tuple matches the scan-window scope.
        recipient_addr = getattr(
            self.ledger, "_connected_address", "",
        )
        if (
            self._checkpoint_store is not None
            and recipient_addr
        ):
            try:
                if self._checkpoint_store.has_credited_tx(
                    recipient_addr, tx_hash,
                ):
                    # Cache the result in-memory so subsequent ticks
                    # within this process skip the SQLite lookup.
                    self._credited_tx_hashes.add(tx_hash)
                    return
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "inbound-monitor dedup-store lookup failed "
                    "(non-fatal — falling back to in-memory dedup, "
                    "double-credit risk on restart): %s", exc,
                )
        # Lookup linked wallet
        wallet_id = await self._local_ledger.wallet_for_eth_address(
            from_addr,
        )
        if wallet_id is None:
            logger.info(
                "Inbound from unlinked address %s — %.6f FTNS not "
                "credited (link via /wallet/deposit/link). tx %s",
                from_addr[:10] + "…",
                amount,
                tx_hash[:18] + "…",
            )
            return
        # Credit. Description includes tx hash so the off-chain
        # ledger entry is traceable to the on-chain TX.
        from prsm.node.local_ledger import TransactionType
        try:
            # sp1101 (Domain-05 review F1) — make the credit IDEMPOTENT at the durable
            # ledger, BELOW the checkpoint-store dedup. The credit fires before the
            # mark is persisted, so a crash in that window + the restart catch-up scan
            # would re-present this Transfer and double-credit (withdrawable as real
            # FTNS). A deterministic idempotency key keyed on the on-chain identity
            # (recipient + tx_hash) makes a replay a PRIMARY-KEY no-op → exactly-once.
            await self._local_ledger.credit(
                wallet_id=wallet_id,
                amount=amount,
                tx_type=TransactionType.BRIDGE_DEPOSIT,
                description=(
                    f"bridge deposit from {from_addr} tx={tx_hash}"
                ),
                idempotency_key=f"bridge-deposit:{recipient_addr}:{tx_hash}",
            )
            self._credited_tx_hashes.add(tx_hash)
            # Sprint 544: persist the dedup record alongside the
            # in-memory cache so the next restart's catch-up scan
            # skips this tx instead of re-crediting.
            if self._checkpoint_store is not None and recipient_addr:
                try:
                    self._checkpoint_store.mark_credited(
                        recipient_addr, tx_hash,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "inbound-monitor dedup-store mark failed "
                        "(non-fatal — in-memory dedup still active, "
                        "but restart could re-credit): %s", exc,
                    )
            logger.info(
                "Bridge deposit credited: %.6f FTNS to wallet=%s "
                "(from %s, tx %s)",
                amount, wallet_id[:18] + "…",
                from_addr[:10] + "…", tx_hash[:18] + "…",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "bridge deposit credit raised — off-chain balance "
                "NOT updated; manual reconciliation may be needed: "
                "%s. tx=%s, from=%s, amount=%.6f, wallet=%s",
                exc, tx_hash, from_addr, amount, wallet_id,
            )
            raise

    async def run_forever(self) -> None:
        import asyncio as _aio
        while True:
            try:
                await self._tick_async()
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "inbound-monitor outer exception "
                    "(non-fatal): %s", exc,
                )
            try:
                await _aio.sleep(self.interval_seconds)
            except _aio.CancelledError:
                raise


@dataclass
class FTNSTransaction:
    """Record of an on-chain FTNS transfer."""
    job_id: str
    from_addr: str
    to_addr: str
    amount_ftns: float
    tx_hash: str = ""
    status: str = "pending"  # pending | confirmed | rejected
    created_at: float = field(default_factory=lambda: __import__("time").time())
    block_number: Optional[int] = None
    # sp1439 (audit) — the on-chain nonce this tx was signed at. The pending-withdraw
    # reconciler uses it as the DETERMINISTIC "this tx can never mine" signal: once the
    # escrow's CONFIRMED nonce advances past it (a different tx took the slot), a dropped
    # tx is provably dead and its off-chain debit can be safely refunded without ever
    # double-paying a tx that later mines.
    nonce: Optional[int] = None


class OnChainFTNSLedger:
    """Manages FTNS token transfers on Base mainnet.

    Wraps a Web3 connection to the FTNS contract and provides:
    - get_balance(address) → real on-chain FTNS balance
    - create_escrow(job_id, amount) → approve + lock tokens
    - release_escrow(job_id, to_addr) → transfer tokens
    - record all tx hashes for audit

    The LocalLedger keeps its own internal state for speed;
    OnChainFTNSLedger mirrors the economic events to blockchain.
    """

    def __init__(
        self,
        node_id: str,
        wallet_private_key: Optional[str] = None,
        contract_address: str = FTNS_CONTRACT_ADDRESS,
        rpc_url: str = BASE_RPC_URL,
        chain_id: int = BASE_CHAIN_ID,
        db_path: Optional[str] = None,
    ):
        self.node_id = node_id
        self.wallet_private_key = wallet_private_key or os.getenv(
            "FTNS_WALLET_PRIVATE_KEY"
        )
        self.contract_address = contract_address
        self.rpc_url = rpc_url
        self.chain_id = chain_id
        self.db_path = db_path

        self.w3: Optional[Web3] = None
        self._account = None
        self._connected_address: Optional[str] = None
        self._token = None
        self._decimals = 18
        self._transactions: List[FTNSTransaction] = []
        self._is_initialized = False
        self._lock = asyncio.Lock()
        self._db = None  # aiosqlite.Connection if persistent

        # Phase 1.3 Task 3a: populate _connected_address synchronously
        # from the wallet private key. Account.from_key(key).address is
        # a pure-local elliptic curve derivation — no network call — so
        # it's safe to do in __init__. This makes the address available
        # at PRSMNode.initialize() time (before the async initialize()
        # runs in start()), which is when ContentUploader is constructed
        # and _derive_creator_address reads this field. Without the sync
        # population, ContentUploader.creator_address would always be
        # None in production and every upload would silently skip
        # provenance_hash computation and on-chain royalty routing.
        #
        # The async initialize() still runs later for RPC/balance/
        # contract-state setup; it idempotently re-derives the same
        # _account and _connected_address (no-op if already set).
        if self.wallet_private_key and Account is not None:
            try:
                key = (
                    self.wallet_private_key
                    if self.wallet_private_key.startswith("0x")
                    else "0x" + self.wallet_private_key
                )
                self._account = Account.from_key(key)
                self._connected_address = self._account.address
            except Exception as exc:
                # Bad key format or missing eth_account — log and leave
                # _connected_address=None so the upload path falls back
                # to PRSM_CREATOR_ADDRESS env var or local royalties.
                logger.warning(
                    f"Could not derive connected_address from "
                    f"FTNS_WALLET_PRIVATE_KEY: {exc}. On-chain routing "
                    f"will require PRSM_CREATOR_ADDRESS env var or fall "
                    f"back to local royalties."
                )

    @property
    def is_persistent(self) -> bool:
        return self.db_path is not None

    async def _init_persistence(self) -> None:
        if not self.is_persistent:
            return
        import aiosqlite
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS onchain_transactions (
                job_id        TEXT PRIMARY KEY,
                tx_hash       TEXT,
                from_addr     TEXT,
                to_addr       TEXT,
                amount_ftns   REAL NOT NULL,
                status        TEXT NOT NULL,
                block_number  INTEGER,
                created_at    REAL NOT NULL
            )
        """)
        # Sprint 510 F39 fix: ensure chain_id column exists.
        # ALTER TABLE adds it with NULL default for pre-sprint-510
        # rows. Wrapped in try/except since the column may already
        # exist on databases initialized after this sprint.
        try:
            await self._db.execute(
                "ALTER TABLE onchain_transactions "
                "ADD COLUMN chain_id INTEGER"
            )
        except Exception:
            pass  # column already exists
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_octx_created "
            "ON onchain_transactions(created_at)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_octx_tx_hash "
            "ON onchain_transactions(tx_hash)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_octx_chain_id "
            "ON onchain_transactions(chain_id)"
        )
        await self._db.commit()
        # Replay rows — sprint 510 F39 fix: filter by this
        # ledger's chain_id so a daemon on Sepolia doesn't
        # see mainnet TX (and vice versa). NULL-chain_id
        # rows (legacy pre-sprint-510) are excluded — they
        # have ambiguous provenance and shouldn't be
        # trusted as belonging to any specific network.
        cur = await self._db.execute(
            "SELECT job_id, tx_hash, from_addr, to_addr, "
            "amount_ftns, status, block_number, created_at "
            "FROM onchain_transactions "
            "WHERE chain_id = ? "
            "ORDER BY created_at ASC",
            (self.chain_id,),
        )
        rows = await cur.fetchall()
        for r in rows:
            self._transactions.append(FTNSTransaction(
                job_id=r[0],
                from_addr=r[2] or "",
                to_addr=r[3] or "",
                amount_ftns=r[4],
                tx_hash=r[1] or "",
                status=r[5],
                block_number=r[6],
                created_at=r[7],
            ))

    def _emit_startup_gas_log(self) -> None:
        """Push-signal counterpart to /wallet/gas-status.

        Called at end of initialize(). Logs WARNING for low gas
        and ERROR for critical so operators see it in startup
        output even without polling /health/detailed.
        """
        if self.w3 is None or self._connected_address is None:
            return
        try:
            wei = self.w3.eth.get_balance(self._connected_address)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "startup gas check failed (non-fatal): %s", exc,
            )
            return
        eth = wei / 1e18
        addr_short = self._connected_address[:10] + "…"
        if eth < 0.0001:
            logger.error(
                "Operator gas CRITICAL: %.10f ETH on %s — "
                "broadcasts will start failing soon. Top up now.",
                eth, addr_short,
            )
        elif eth < 0.0005:
            logger.warning(
                "Operator gas low: %.10f ETH on %s — plan to "
                "top up to avoid mid-job failures.",
                eth, addr_short,
            )
        else:
            logger.info(
                "Operator gas ok: %.10f ETH on %s",
                eth, addr_short,
            )

    async def _close_persistence(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def _record_tx(self, tx: FTNSTransaction) -> None:
        if self._db is None:
            return
        await self._db.execute(
            "INSERT OR REPLACE INTO onchain_transactions "
            "(job_id, tx_hash, from_addr, to_addr, amount_ftns, "
            "status, block_number, created_at, chain_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                tx.job_id, tx.tx_hash, tx.from_addr, tx.to_addr,
                tx.amount_ftns, tx.status, tx.block_number,
                tx.created_at, self.chain_id,
            ),
        )
        await self._db.commit()

    async def _reconcile_pending_transactions(self) -> None:
        """Sprint 511: rebuild pending status from chain receipts.

        If the daemon was killed during wait_for_transaction_receipt
        (e.g. SIGKILL, container restart, OOM), TX broadcast
        successfully on-chain but the local audit trail stayed
        'pending'. At init time, check the chain for any
        outstanding pending rows and update status if a receipt
        now exists.
        """
        if self.w3 is None:
            return
        import asyncio as _aio
        loop = _aio.get_running_loop()
        for tx in self._transactions:
            if tx.status != "pending":
                continue
            if not tx.tx_hash:
                continue
            try:
                tx_hash_bytes = tx.tx_hash
                if not tx_hash_bytes.startswith("0x"):
                    tx_hash_bytes = "0x" + tx_hash_bytes
                receipt = await loop.run_in_executor(
                    None,
                    lambda: self.w3.eth.get_transaction_receipt(
                        tx_hash_bytes,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "reconcile: receipt lookup for %s failed "
                    "(leaving pending): %s",
                    tx.tx_hash[:18], exc,
                )
                continue
            if receipt is None:
                continue
            if receipt.get("status") == 1:
                tx.status = "confirmed"
                tx.block_number = receipt.get("blockNumber")
                logger.info(
                    "reconcile: promoted %s pending → "
                    "confirmed (block %s)",
                    tx.tx_hash[:18], tx.block_number,
                )
            else:
                tx.status = "rejected"
                logger.warning(
                    "reconcile: marked %s pending → rejected",
                    tx.tx_hash[:18],
                )
            try:
                await self._update_tx_status(tx)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "reconcile: persistence update failed "
                    "for %s: %s", tx.tx_hash[:18], exc,
                )

    async def _update_tx_status(self, tx: FTNSTransaction) -> None:
        if self._db is None:
            return
        await self._db.execute(
            "UPDATE onchain_transactions "
            "SET status = ?, block_number = ?, tx_hash = ? "
            "WHERE job_id = ?",
            (tx.status, tx.block_number, tx.tx_hash, tx.job_id),
        )
        await self._db.commit()

    async def initialize(self) -> bool:
        """Connect to Base mainnet and load the FTNS contract."""
        if self._is_initialized:
            return True

        if not HAS_WEB3:
            logger.warning("web3 not installed — FTNS on-chain mode disabled")
            return False

        try:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url, request_kwargs={"timeout": 15}))
            if not self.w3.is_connected():
                logger.error(f"Cannot connect to Base RPC: {self.rpc_url}")
                return False

            # Connect wallet
            if self.wallet_private_key:
                key = self.wallet_private_key if self.wallet_private_key.startswith("0x") else "0x" + self.wallet_private_key
                self._account = Account.from_key(key)
                self._connected_address = self._account.address
                logger.info(f"FTNS wallet connected: {self._connected_address}")
            else:
                # Public read-only mode — can check balances but not send
                logger.info("FTNS ledger in read-only mode (no private key)")

            # Load the FTNS contract
            self._token = self.w3.eth.contract(
                address=to_checksum_address(self.contract_address),
                abi=_ERC20_ABI,
            )

            # Get decimals
            if self.w3.is_connected():
                try:
                    self._decimals = self._token.functions.decimals().call()
                    name = self._token.functions.name().call()
                    symbol = self._token.functions.symbol().call()
                    logger.info(f"FTNS token loaded: {name} ({symbol})")
                except Exception as e:
                    logger.warning(f"Could not read token metadata: {e}")

            await self._init_persistence()

            self._is_initialized = True
            self._emit_startup_gas_log()
            # Sprint 511: reconcile any pending TX whose receipt
            # may have been missed due to daemon interruption.
            try:
                await self._reconcile_pending_transactions()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "pending-TX reconciliation raised "
                    "(non-fatal): %s", exc,
                )
            return True

        except Exception as e:
            logger.error(f"FTNS ledger init failed: {e}")
            return False

    async def get_balance(self, address: Optional[str] = None) -> float:
        """Get FTNS balance for an address from the blockchain."""
        if not self._is_initialized:
            ok = await self.initialize()
            if not ok:
                return 0.0

        addr = address or self._connected_address
        if not addr or not self._token:
            return 0.0

        try:
            bal_wei = self._token.functions.balanceOf(to_checksum_address(addr)).call()
            return bal_wei / (10 ** self._decimals)
        except Exception as e:
            logger.warning(f"Failed to read FTNS balance for {addr[:12]}…: {e}")
            return 0.0

    async def transfer(
        self,
        job_id: str,
        to_address: str,
        amount_ftns: float,
    ) -> Optional[FTNSTransaction]:
        """Send FTNS tokens on-chain.

        Only works if a wallet private key is configured.
        """
        if not self._is_initialized:
            await self.initialize()
        if not self._account or not self._token:
            logger.error("Cannot transfer FTNS — no wallet configured")
            return None
        if amount_ftns <= 0:
            return None

        tx_record = FTNSTransaction(
            job_id=job_id,
            from_addr=self._connected_address,
            to_addr=to_address,
            amount_ftns=amount_ftns,
        )

        async with self._lock:
            try:
                amount_wei = int(amount_ftns * (10 ** self._decimals))

                # All web3 calls are synchronous and block the event loop.
                # Run them in a thread to keep the API responsive.
                import asyncio
                loop = asyncio.get_running_loop()

                # sp931 — serialize the nonce-fetch→sign→send on the PROCESS-WIDE
                # per-account lock. The "pending" block alone does NOT prevent a
                # collision: a co-resident client signing from the same
                # FTNS_WALLET_PRIVATE_KEY (RoyaltyDistributorClient, StakeManager,
                # …) locking independently reads the SAME "pending" nonce and
                # broadcasts a colliding tx that is silently dropped ("nonce too
                # low"). The registry hands every such client the SAME
                # threading.Lock for this account. Acquire it OFF the event loop
                # (it's a blocking lock) and hold it ONLY across nonce→send — NOT
                # the multi-second receipt wait below.
                from prsm.economy.web3.tx_lock_registry import TX_LOCK_REGISTRY
                tx_lock = TX_LOCK_REGISTRY.get_lock(self._connected_address)
                await loop.run_in_executor(None, tx_lock.acquire)
                try:
                    nonce = await loop.run_in_executor(
                        None,
                        lambda: self.w3.eth.get_transaction_count(
                            self._connected_address, "pending"
                        ),
                    )

                    # Build tx
                    tx = {
                        "chainId": self.chain_id,
                        "nonce": nonce,
                        "gasPrice": estimate_gas_price(self.w3),
                        "gas": 100000,
                        "to": self.contract_address,
                        "value": 0,
                        "data": self._token.functions.transfer(
                            to_checksum_address(to_address), amount_wei
                        )._encode_transaction_data(),
                    }

                    signed = self.w3.eth.account.sign_transaction(tx, self._account.key)
                    tx_hash = await loop.run_in_executor(
                        None, self.w3.eth.send_raw_transaction, signed.raw_transaction
                    )
                finally:
                    tx_lock.release()

                tx_record.tx_hash = tx_hash.hex()
                tx_record.status = "pending"
                tx_record.nonce = int(nonce)   # sp1439 — deterministic dropped-tx signal
                self._transactions.append(tx_record)
                await self._record_tx(tx_record)

                logger.info(
                    f"FTNS transfer sent: {amount_ftns:.6f} -> {to_address[:12]}… "
                    f"(tx: {tx_record.tx_hash[:16]}…)"
                )

                # Wait for confirmation in a thread (can take 2-30s on Base)
                receipt = await loop.run_in_executor(
                    None,
                    lambda: self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60),
                )
                if receipt["status"] == 1:
                    tx_record.status = "confirmed"
                    tx_record.block_number = receipt["blockNumber"]
                    logger.info(
                        f"FTNS transfer confirmed in block {tx_record.block_number}"
                    )
                else:
                    tx_record.status = "rejected"
                    logger.warning(f"FTNS transfer reverted: {tx_hash.hex()}")
                await self._update_tx_status(tx_record)

                return tx_record

            except Exception as e:
                # sp914 — distinguish NEVER-BROADCAST (safe to retry/refund)
                # from BROADCAST-BUT-UNCONFIRMED. If we already have a tx_hash,
                # `send_raw_transaction` SUCCEEDED and the tx is in the mempool
                # (this except is almost always `wait_for_transaction_receipt`
                # timing out on a congested Base). Returning None here made
                # callers treat an in-flight tx as a clean failure → withdraw
                # refunded the off-chain debit (DOUBLE-PAY when the tx later
                # confirmed) and batch-settlement silently DROPPED the owed
                # payout. Surface it as "pending" instead; only a tx that was
                # never broadcast (no tx_hash) returns None.
                if getattr(tx_record, "tx_hash", ""):
                    tx_record.status = "pending"
                    logger.warning(
                        f"FTNS transfer broadcast but unconfirmed "
                        f"(receipt wait failed: {e}); tx={tx_record.tx_hash[:16]}… "
                        f"is pending — NOT a clean failure"
                    )
                    await self._update_tx_status(tx_record)
                    return tx_record
                tx_record.status = "rejected"
                logger.error(f"FTNS transfer failed (never broadcast): {e}")
                self._transactions.append(tx_record)
                await self._record_tx(tx_record)
                return None

    def token_info_sync(self) -> Dict[str, Any]:
        """Get on-chain FTNS token info (name, symbol, total supply)."""
        if not self._token:
            return {}

        try:
            name = self._token.functions.name().call()
            symbol = self._token.functions.symbol().call()
            decimals = self._token.functions.decimals().call()
            total_supply = self._token.functions.totalSupply().call()
            return {
                "name": name,
                "symbol": symbol,
                "decimals": decimals,
                "total_supply": total_supply / (10 ** decimals),
                "contract": self.contract_address,
            }
        except Exception:
            return {}

    def get_summary(self) -> Dict[str, Any]:
        """Summary of all on-chain FTNS transactions."""
        statuses = {}
        for tx in self._transactions:
            statuses[tx.status] = statuses.get(tx.status, 0) + 1
        total_sent = sum(tx.amount_ftns for tx in self._transactions if tx.status == "confirmed")
        return {
            "total_transactions": len(self._transactions),
            "by_status": statuses,
            "total_confirmed_ftns": round(total_sent, 6),
            "wallet": self._connected_address,
        }
