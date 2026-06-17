"""Sprint 1069 — AerodromeSwapClient: the USDC→FTNS swap EXECUTION (Tier 4).

The fiat onramp loop had every piece except the last mile. ``aerodrome_client.py`` is
quote-only (pool state + constant-product quote) and ``onramp_to_swap_orchestrator``
builds a swap ENVELOPE, but nothing builds/signs/broadcasts the on-chain swap, so a
funded user's USDC dead-ends in their wallet. This client is the write side: it
ensures the USDC allowance to the Aerodrome router, then calls
``swapExactTokensForTokens`` with the USDC→FTNS Route.

Mirrors ``EscrowPoolClient`` (sp1041): sync web3 + ``Account`` signing +
``TX_LOCK_REGISTRY`` nonce serialization + the three-tier error model
(BroadcastFailed / OnChainPending / OnChainReverted), each blocking call wrapped in
``asyncio.to_thread``, with explicit gas limits so ``build_transaction`` skips
``eth_estimateGas`` (sp1047 — a just-set approve can be invisible to a lagging RPC
read replica and spuriously revert the swap's estimate).

Aerodrome is a Velodrome-fork DEX on Base. Its router takes a list of ``Route``
structs ``(from, to, stable, factory)``; a USDC→FTNS swap is a single volatile-pool
hop ``(USDC, FTNS, stable=False, factory)``.

Scope/gating: like the settlement rail, the live mainnet swap is gated on a SEEDED
pool (Foundation ceremony) — without liquidity the swap reverts (caught as
OnChainRevertedError). This builds + proves the execution so it activates the moment
the pool has liquidity. The caller is responsible for the slippage bound
(``amount_out_min``) — the onramp orchestrator already computes it from the quote.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass
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

# Aerodrome Router v2 (Velodrome fork) — swapExactTokensForTokens with Route[] tuples.
AERODROME_ROUTER_ABI = [
    {"type": "function", "name": "swapExactTokensForTokens",
     "stateMutability": "nonpayable",
     "inputs": [
         {"name": "amountIn", "type": "uint256"},
         {"name": "amountOutMin", "type": "uint256"},
         {"name": "routes", "type": "tuple[]", "components": [
             {"name": "from", "type": "address"},
             {"name": "to", "type": "address"},
             {"name": "stable", "type": "bool"},
             {"name": "factory", "type": "address"}]},
         {"name": "to", "type": "address"},
         {"name": "deadline", "type": "uint256"}],
     "outputs": [{"name": "amounts", "type": "uint256[]"}]},
    # sp1151 — VIEW used ONLY by the read-only dry-run to verify pool liquidity +
    # quote the expected FTNS out before any broadcast. Never written to.
    {"type": "function", "name": "getAmountsOut",
     "stateMutability": "view",
     "inputs": [
         {"name": "amountIn", "type": "uint256"},
         {"name": "routes", "type": "tuple[]", "components": [
             {"name": "from", "type": "address"},
             {"name": "to", "type": "address"},
             {"name": "stable", "type": "bool"},
             {"name": "factory", "type": "address"}]}],
     "outputs": [{"name": "amounts", "type": "uint256[]"}]},
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

_RECEIPT_TIMEOUT_SECONDS = 180
# Explicit gas (sp1047) — skip estimateGas, which can revert against a lagging replica
# that hasn't seen the just-set approve. Generous; only gasUsed is charged.
_APPROVE_GAS = 100_000
_SWAP_GAS = 400_000   # a single-hop router swap; generous headroom


@dataclass(frozen=True)
class AerodromeSwapDryRun:
    """sp1151 — verdict of a READ-ONLY swap pre-flight. Mirrors the challenge
    submitters' ``DryRunResult`` shape: a verdict object the operator inspects
    BEFORE broadcasting, never an exception.

    ``would_succeed`` is the bottom line: liquidity_ok AND sufficient_balance AND
    slippage_ok. A short allowance is NOT a hard fail — ``_swap_sync`` auto-approves —
    so ``approve_needed`` is reported but does not by itself flip ``would_succeed``.
    """
    would_succeed: bool
    amount_in: int
    amount_out_min: int
    expected_out: Optional[int] = None
    sufficient_balance: Optional[bool] = None
    sufficient_allowance: Optional[bool] = None
    approve_needed: bool = False
    revert_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "would_succeed": self.would_succeed,
            "amount_in": self.amount_in,
            "amount_out_min": self.amount_out_min,
            "expected_out": self.expected_out,
            "sufficient_balance": self.sufficient_balance,
            "sufficient_allowance": self.sufficient_allowance,
            "approve_needed": self.approve_needed,
            "revert_reason": self.revert_reason,
        }

    def render(self) -> str:
        verdict = "WOULD SUCCEED" if self.would_succeed else "WOULD REVERT"
        lines = [
            f"swap dry-run: {verdict}",
            f"  amount_in (USDC units):   {self.amount_in}",
            f"  amount_out_min (FTNS):    {self.amount_out_min}",
            f"  expected_out (FTNS):      {self.expected_out}",
            f"  sufficient_balance:       {self.sufficient_balance}",
            f"  sufficient_allowance:     {self.sufficient_allowance}",
            f"  approve_needed:           {self.approve_needed}",
        ]
        if self.revert_reason:
            lines.append(f"  reason:                   {self.revert_reason}")
        return "\n".join(lines)


class AerodromeSwapClient:
    """web3.py client that executes a USDC→FTNS swap on the Aerodrome router.
    Write calls require a private key controlling the USDC-holding wallet."""

    def __init__(
        self,
        rpc_url: str,
        router_address: str,
        usdc_address: str,
        ftns_address: str,
        factory_address: str,
        private_key: Optional[str] = None,
        *,
        stable: bool = False,
    ) -> None:
        if not HAS_WEB3:
            raise RuntimeError("web3 package is required (pip install web3 eth-account)")
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self.router_address = Web3.to_checksum_address(router_address)
        self.usdc_address = Web3.to_checksum_address(usdc_address)
        self.ftns_address = Web3.to_checksum_address(ftns_address)
        self.factory_address = Web3.to_checksum_address(factory_address)
        self._stable = bool(stable)
        self.router = self.web3.eth.contract(
            address=self.router_address, abi=AERODROME_ROUTER_ABI)
        self.usdc = self.web3.eth.contract(
            address=self.usdc_address, abi=_ERC20_ABI)
        self._account = Account.from_key(private_key) if private_key else None

        from prsm.economy.web3.tx_lock_registry import TX_LOCK_REGISTRY
        self._tx_lock = (
            TX_LOCK_REGISTRY.get_lock(self._account.address)
            if self._account is not None else threading.Lock()
        )

    @classmethod
    def from_env(cls) -> Optional["AerodromeSwapClient"]:
        """Build from env, or None when not fully configured (so a caller can fail
        soft until the pool ceremony populates the addresses)."""
        rpc = (os.environ.get("BASE_RPC_URL", "") or "").strip()
        router = (os.environ.get("AERODROME_ROUTER_ADDRESS", "") or "").strip()
        usdc = (os.environ.get("BASE_USDC_ADDRESS", "")
                or os.environ.get("USDC_TOKEN_ADDRESS", "") or "").strip()
        ftns = (os.environ.get("FTNS_TOKEN_ADDRESS", "") or "").strip()
        factory = (os.environ.get("AERODROME_FACTORY_ADDRESS", "") or "").strip()
        key = (os.environ.get("FTNS_WALLET_PRIVATE_KEY", "")
               or os.environ.get("PRSM_SWAP_PRIVATE_KEY", "") or "").strip() or None
        if not (rpc and router and usdc and ftns and factory):
            return None
        return cls(rpc_url=rpc, router_address=router, usdc_address=usdc,
                   ftns_address=ftns, factory_address=factory, private_key=key)

    @property
    def address(self) -> Optional[str]:
        return self._account.address if self._account else None

    # ── Async surface ────────────────────────────────────────────────────────

    async def usdc_allowance(self) -> int:
        """The signer's current USDC allowance to the router."""
        if not self._account:
            raise RuntimeError("private_key required to read the signer's allowance")
        owner = self._account.address
        return await asyncio.to_thread(
            lambda: int(self.usdc.functions.allowance(owner, self.router_address).call()))

    async def usdc_balance_of(self, account: str) -> int:
        addr = Web3.to_checksum_address(account)
        return await asyncio.to_thread(
            lambda: int(self.usdc.functions.balanceOf(addr).call()))

    # ── read-only pre-flight (NOT a broadcast — sp1151) ───────────────────────

    async def dry_run_swap_usdc_for_ftns(
        self,
        amount_in: int,
        amount_out_min: int,
        *,
        owner: Optional[str] = None,
    ) -> AerodromeSwapDryRun:
        """READ-ONLY verify-before-broadcast: would a USDC→FTNS swap of ``amount_in``
        units (slippage floor ``amount_out_min``) succeed against current chain state?

        Mirrors the challenge submitters' ``dry_run`` (static ``.call()`` + a verdict
        object, a revert → ``would_succeed=False`` rather than a propagated exception)
        and the three-tier checks of ``_swap_sync`` — but NEVER builds/signs/sends a tx
        and NEVER approves. All chain reads go through ``asyncio.to_thread(... .call())``.

        Checks, in order:
          (a) ``getAmountsOut(amount_in, [route])`` → ``expected_out`` (= amounts[-1])
              and proves the pool has LIQUIDITY. A revert here (e.g. pre-seeding, no
              pool) is caught → ``would_succeed=False`` + a clear liquidity reason.
          (b) ``balanceOf(owner) >= amount_in`` → ``sufficient_balance`` (a HARD fail).
          (c) ``allowance(owner, router) >= amount_in`` → ``sufficient_allowance`` /
              ``approve_needed`` (NOT a hard fail — the real swap auto-approves).
          (d) SLIPPAGE: ``expected_out >= amount_out_min`` (a HARD fail).

        ``owner`` defaults to the signer address (``self.address``). With neither a
        signer nor an ``owner`` the balance/allowance checks are skipped (liquidity +
        slippage stay verifiable) and noted in ``revert_reason`` — never raised.
        Raises ``ValueError`` on ``amount_in <= 0`` / ``amount_out_min < 0`` (mirrors
        ``_swap_sync``).
        """
        amount_in = int(amount_in)
        amount_out_min = int(amount_out_min)
        if amount_in <= 0:
            raise ValueError(f"amount_in must be positive (got {amount_in})")
        if amount_out_min < 0:
            raise ValueError(f"amount_out_min must be >= 0 (got {amount_out_min})")

        route = (self.usdc_address, self.ftns_address, self._stable,
                 self.factory_address)

        # (a) Liquidity + quote. A revert here is the pre-seeding "no pool" case —
        # caught, NOT propagated. READ-ONLY: a VIEW .call(), never a broadcast.
        try:
            amounts = await asyncio.to_thread(
                lambda: self.router.functions.getAmountsOut(
                    amount_in, [route]).call())
            expected_out = int(amounts[-1])
        except Exception as exc:  # noqa: BLE001 — surface as a verdict, don't raise
            return AerodromeSwapDryRun(
                would_succeed=False,
                amount_in=amount_in,
                amount_out_min=amount_out_min,
                expected_out=None,
                revert_reason=(
                    "insufficient liquidity / no pool (getAmountsOut reverted: "
                    f"{exc})"),
            )

        # Resolve the owner whose balance/allowance we'd spend.
        resolved_owner = owner or self.address
        sufficient_balance: Optional[bool] = None
        sufficient_allowance: Optional[bool] = None
        approve_needed = False
        owner_note: Optional[str] = None

        if resolved_owner is not None:
            addr = Web3.to_checksum_address(resolved_owner)
            # (b) balance — READ-ONLY .call().
            usdc_bal = await asyncio.to_thread(
                lambda: int(self.usdc.functions.balanceOf(addr).call()))
            sufficient_balance = usdc_bal >= amount_in
            # (c) allowance — READ-ONLY .call(). Short allowance is NOT a hard fail.
            allowance = await asyncio.to_thread(
                lambda: int(self.usdc.functions.allowance(
                    addr, self.router_address).call()))
            sufficient_allowance = allowance >= amount_in
            approve_needed = not sufficient_allowance
        else:
            owner_note = ("no signer and no owner supplied — balance/allowance not "
                          "checked (liquidity + slippage verified)")

        # (d) slippage.
        slippage_ok = expected_out >= amount_out_min

        would_succeed = (
            slippage_ok
            and (sufficient_balance is not False)
        )

        revert_reason: Optional[str] = None
        if not slippage_ok:
            revert_reason = (
                f"slippage: expected_out {expected_out} < amount_out_min "
                f"{amount_out_min} (would revert on the router's slippage check)")
        elif sufficient_balance is False:
            revert_reason = (
                f"insufficient USDC balance for owner {resolved_owner}: "
                f"< amount_in {amount_in}")
        elif owner_note is not None:
            revert_reason = owner_note

        return AerodromeSwapDryRun(
            would_succeed=would_succeed,
            amount_in=amount_in,
            amount_out_min=amount_out_min,
            expected_out=expected_out,
            sufficient_balance=sufficient_balance,
            sufficient_allowance=sufficient_allowance,
            approve_needed=approve_needed,
            revert_reason=revert_reason,
        )

    async def swap_usdc_for_ftns(
        self,
        amount_in: int,
        amount_out_min: int,
        deadline_unix: int,
        recipient: Optional[str] = None,
    ) -> str:
        """Approve (only if the allowance is short) then swap ``amount_in`` USDC units
        for FTNS via the router, requiring at least ``amount_out_min`` FTNS units (the
        caller's slippage floor). FTNS goes to ``recipient`` (default: the signer).
        Returns the swap tx hash. Raises the three-tier on-chain errors on failure."""
        return await asyncio.to_thread(
            self._swap_sync, int(amount_in), int(amount_out_min),
            int(deadline_unix), recipient)

    # ── Sync implementation ──────────────────────────────────────────────────

    def _swap_sync(self, amount_in: int, amount_out_min: int, deadline_unix: int,
                   recipient: Optional[str]) -> str:
        if not self._account:
            raise RuntimeError("private_key required for write calls (swap)")
        if amount_in <= 0:
            raise ValueError(f"amount_in must be positive (got {amount_in})")
        if amount_out_min < 0:
            raise ValueError(f"amount_out_min must be >= 0 (got {amount_out_min})")
        if deadline_unix <= 0:
            raise ValueError(f"deadline_unix must be positive (got {deadline_unix})")
        to = Web3.to_checksum_address(recipient) if recipient else self._account.address
        route = (self.usdc_address, self.ftns_address, self._stable, self.factory_address)
        with self._tx_lock:
            owner = self._account.address
            # sp1069 review MEDIUM-3 — fail fast if the wallet can't cover the swap,
            # rather than landing an approve (→ residual allowance) then reverting.
            usdc_bal = int(self.usdc.functions.balanceOf(owner).call())
            if usdc_bal < amount_in:
                raise ValueError(
                    f"insufficient USDC: balance {usdc_bal} < amount_in {amount_in}")
            allowance = int(
                self.usdc.functions.allowance(owner, self.router_address).call())
            did_approve = False
            if allowance < amount_in:
                approve_tx = self.usdc.functions.approve(
                    self.router_address, amount_in,
                ).build_transaction(self._tx_overrides(gas=_APPROVE_GAS))
                self._sign_send_wait(approve_tx)
                did_approve = True
            swap_tx = self.router.functions.swapExactTokensForTokens(
                amount_in, amount_out_min, [route], to, deadline_unix,
            ).build_transaction(self._tx_overrides(gas=_SWAP_GAS))
            try:
                receipt = self._sign_send_wait(swap_tx)
            except (BroadcastFailedError, OnChainRevertedError):
                # sp1069 review HIGH-2 — the approve landed but the swap did NOT
                # consume it (never broadcast, or mined-and-reverted). Best-effort
                # revoke the residual router allowance we just created so it can't
                # stand as a pull-right on the user's USDC. Only when WE approved
                # (don't touch a pre-existing user allowance). OnChainPending is
                # deliberately NOT revoked — that swap may still land + consume it.
                if did_approve:
                    try:
                        revoke = self.usdc.functions.approve(
                            self.router_address, 0,
                        ).build_transaction(self._tx_overrides(gas=_APPROVE_GAS))
                        self._sign_send_wait(revoke)
                    except Exception as rexc:  # noqa: BLE001
                        logger.warning("could not revoke residual USDC allowance "
                                       "after a failed swap: %s", rexc)
                raise
        return "0x" + _receipt_tx_hash(receipt)

    # ── Helpers (mirror EscrowPoolClient) ─────────────────────────────────────

    def _tx_overrides(self, gas: Optional[int] = None) -> dict:
        ov = {
            "from": self._account.address,
            "nonce": self.web3.eth.get_transaction_count(
                self._account.address, "pending"),
            "gasPrice": self.web3.eth.gas_price,
            "chainId": self.web3.eth.chain_id,
        }
        if gas is not None:
            ov["gas"] = int(gas)
        return ov

    def _sign_send_wait(self, tx: dict):
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
            raise OnChainRevertedError(f"swap reverted: {tx_hash_hex}")
        return receipt


_MAX_ENVELOPE_AGE_SECONDS = 600   # a quote older than this is too stale to trust


async def execute_swap_envelope(
    envelope: dict,
    client: "AerodromeSwapClient",
    *,
    now: Optional[float] = None,
    max_age_seconds: int = _MAX_ENVELOPE_AGE_SECONDS,
) -> dict:
    """Execute a sp871 onramp→swap envelope via ``client``. The bridge from the
    orchestrator's READY_FOR_SUBMISSION envelope to an on-chain swap.

    Money-path guards (fail closed BEFORE broadcasting; sp1069 review):
      - single-hop USDC→FTNS ``swapExactTokensForTokens`` only;
      - ``amountOutMin >= 1`` (non-zero slippage floor — no accept-any-output);
      - the envelope's ``router_address`` + route ``factory`` MUST match the client's
        configured addresses (HIGH-1: the quote was computed for THIS router/pool; a
        client misconfigured to a different router/factory would swap against the
        wrong venue the floor wasn't quoted for);
      - a non-empty recipient (``to``) — never silently default to the swap wallet;
      - the quote must be FRESH (``envelope_built_at`` within ``max_age_seconds``) and
        the deadline still in the future — a stale floor on a thin pool defeats the
        slippage protection.

    Returns ``{status:'SUBMITTED', tx_hash, ...}``; raises ``ValueError`` on a
    malformed/unsafe/stale envelope or propagates the client's on-chain errors."""
    import time as _time
    now = _time.time() if now is None else now
    if not isinstance(envelope, dict):
        raise ValueError("envelope must be a dict")
    if envelope.get("function") != "swapExactTokensForTokens":
        raise ValueError(f"unsupported swap function {envelope.get('function')!r}")
    args = envelope.get("args")
    if not isinstance(args, dict):
        raise ValueError("envelope.args must be an object")
    routes = args.get("routes") or []
    if len(routes) != 1 or not isinstance(routes[0], dict):
        raise ValueError("envelope route must be a single hop")
    route0 = routes[0]
    if route0.get("from_token") != "USDC" or route0.get("to_token") != "FTNS":
        raise ValueError("envelope route must be USDC→FTNS")
    # HIGH-1 — the envelope's venue must be the client's configured venue.
    env_router = str(envelope.get("router_address", "")).lower()
    if env_router and env_router != client.router_address.lower():
        raise ValueError(
            f"envelope router {envelope.get('router_address')} != client router "
            f"{client.router_address} — refusing to swap against a different venue")
    env_factory = str(route0.get("factory", "")).lower()
    if env_factory and env_factory != client.factory_address.lower():
        raise ValueError(
            f"envelope factory {route0.get('factory')} != client factory "
            f"{client.factory_address} — refusing to swap against a different pool")
    try:
        amount_in = int(args["amountIn"])
        amount_out_min = int(args["amountOutMin"])
        deadline = int(args["deadline"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"envelope args malformed: {exc}")
    if amount_out_min < 1:
        raise ValueError(
            "refusing to execute a swap with no slippage floor (amountOutMin < 1) — "
            "would accept any output (sandwich/MEV risk)")
    recipient = args.get("to")
    if not recipient:
        raise ValueError("envelope must specify a recipient (to) — never default to "
                         "the swap wallet")
    # MEDIUM-1/2 — freshness + future deadline.
    built_at = envelope.get("envelope_built_at")
    if built_at is not None and (now - float(built_at)) > max_age_seconds:
        raise ValueError(
            f"envelope is stale ({int(now - float(built_at))}s > {max_age_seconds}s) "
            f"— re-quote before executing (a stale slippage floor is unsafe)")
    if deadline <= now:
        raise ValueError(f"envelope deadline {deadline} is in the past (now {int(now)})")
    tx_hash = await client.swap_usdc_for_ftns(
        amount_in=amount_in, amount_out_min=amount_out_min,
        deadline_unix=deadline, recipient=recipient)
    return {
        "status": "SUBMITTED",
        "tx_hash": tx_hash,
        "intent_id": envelope.get("intent_id"),
        "amount_in_units": amount_in,
        "amount_out_min_units": amount_out_min,
        "recipient": recipient,
    }


def _receipt_tx_hash(receipt) -> str:
    th = getattr(receipt, "transactionHash", None)
    if th is None and isinstance(receipt, dict):
        th = receipt.get("transactionHash")
    if th is None:
        return ""
    return th.hex() if hasattr(th, "hex") else str(th)
