"""Sprint 793 — production wire-up of WalletApiServices.

Sprint 792 shipped the /devices/earnings endpoint with a
`ReceiptLookup` Protocol on WalletApiServices. The default
`_NoOpReceiptLookup` returns empty so the endpoint doesn't
500 on unwired deployments — but production daemons need a
real adapter backed by `node._receipt_store` for the endpoint
to actually return per-device earnings.

This module ships:
  _ReceiptStoreAdapter — implements ReceiptLookup over the
    node's ReceiptStore. In-memory filter by settler_node_id.
  wire_wallet_api_services(node) — daemon-startup helper that
    builds + registers WalletApiServices via wallet_api.set_services.
    Idempotent (resets first, then sets), fail-soft (wraps in
    try/except so daemon doesn't crash on this peripheral surface).

PRSMNode.start calls wire_wallet_api_services(self) so the
endpoint just works on a running daemon. No env vars or operator
action required — the wire-up is automatic.

Honest scope:
- Adapter scans up to 1000 most-recent receipts from the store's
  in-memory cache (existing ReceiptStore.list contract). Operators
  with very large receipt histories may need to migrate to a
  persistent indexed store — but that's a follow-on (most
  operators have << 1000 receipts in any practical window).
- The binding service still uses InMemoryWalletBindingStore by
  default. A production daemon with multi-process workers needs
  to swap to SqliteWalletBindingStore (or Postgres) so bindings
  survive a worker restart. Sprint 794 candidate.
"""
from __future__ import annotations

import logging
import hashlib
import os
import secrets
from pathlib import Path
from typing import Any, Iterable, List, Dict, Optional


logger = logging.getLogger(__name__)


def _wallet_session_config_from_env() -> tuple:
    """sp1013 — wallet session-token config from env (API-authz Residual A).

    Returns ``(session_secret: bytes, session_required: bool)``.

    Secret: PRSM_WALLET_SESSION_SECRET — a hex string (0x-optional) or, failing
    hex parse, raw UTF-8 bytes; a short configured value is stretched via
    SHA-256 to a 32-byte key. Unset → a per-process random 32-byte secret
    (sessions are invalidated on restart, which is fine for short-lived tokens).

    Enforcement: PRSM_WALLET_SESSION_REQUIRED — when truthy, the wallet-OWNER
    read endpoints require a session token bound to the requested wallet. Default
    off so the mint is additive and existing clients keep working until they
    adopt the token (default-on is a coordinated follow-on with the frontend).
    """
    raw = (os.environ.get("PRSM_WALLET_SESSION_SECRET") or "").strip()
    if raw:
        try:
            secret = bytes.fromhex(raw[2:] if raw.lower().startswith("0x") else raw)
        except ValueError:
            secret = raw.encode("utf-8")
        if len(secret) < 16:
            secret = hashlib.sha256(secret).digest()
    else:
        secret = secrets.token_bytes(32)
    required = (os.environ.get("PRSM_WALLET_SESSION_REQUIRED") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    return secret, required


def _resolve_wallet_bindings_db_path() -> Optional[Path]:
    """Sprint 794 — resolve the daemon-local SQLite path for the
    binding store.

    Order:
      1. PRSM_WALLET_BINDINGS_DB env (operator-set absolute path)
      2. ~/.prsm/wallet_bindings.db (default, matches existing
         daemon-local convention used by credentials + config)

    Returns a Path. Caller validates writability before using it
    + falls back to InMemoryWalletBindingStore on any IO failure.
    """
    env_path = (os.environ.get("PRSM_WALLET_BINDINGS_DB") or "").strip()
    if env_path:
        return Path(env_path)
    return Path.home() / ".prsm" / "wallet_bindings.db"


# Cap the in-memory scan window. ReceiptStore.list enforces an
# upper bound of 1000 itself; we use that explicitly so the
# adapter's behavior is operator-readable from the source.
_MAX_RECEIPTS_SCAN = 1000


class _ReceiptStoreAdapter:
    """Sprint 793 — ReceiptLookup over PRSMNode._receipt_store.

    Implements the sprint-792 Protocol. Scans the store's most-
    recent receipts and returns only those whose settler_node_id
    matches one of the requested node_ids.
    """

    def __init__(self, *, receipt_store: Optional[Any]) -> None:
        self._receipt_store = receipt_store

    def list_receipts_for_node_ids(
        self, node_ids: Iterable[str],
    ) -> List[Dict[str, Any]]:
        if self._receipt_store is None:
            return []
        ids = list(node_ids)
        if not ids:
            return []
        # Sprint 801 — delegate to ReceiptStore.list_for_node_ids
        # so persistence-backed stores return the FULL history
        # (uncapped by the 1024-entry LRU cap). For cache-only
        # stores the method scans the cache; either way the
        # adapter doesn't need its own filter loop.
        try:
            return self._receipt_store.list_for_node_ids(ids)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "_ReceiptStoreAdapter scan failed: %s", exc,
            )
            return []


def wire_wallet_api_services(node: Any) -> None:
    """Sprint 793 — register a production WalletApiServices on
    `wallet_api`.

    Idempotent: resets the existing services slot first, then
    set_services. Safe to call from daemon restart paths.

    Fail-soft: any exception during wiring is logged + swallowed.
    The /devices/earnings endpoint then falls back to the
    `_NoOpReceiptLookup` default — operators see empty earnings
    instead of a 500.
    """
    try:
        from prsm.interface.api import wallet_api
        from prsm.interface.onboarding.wallet_binding import (
            InMemoryWalletBindingStore,
            SqliteWalletBindingStore,
            WalletBindingService,
        )
        receipt_store = getattr(node, "_receipt_store", None)
        adapter = _ReceiptStoreAdapter(receipt_store=receipt_store)

        # Sprint 794 — prefer SQLite for the binding store so
        # bindings survive daemon restart. Fail-soft fallback to
        # in-memory on ANY init error (permission denied, parent
        # dir not creatable, schema incompatibility on an old
        # pre-786 DB, etc.).
        db_path = _resolve_wallet_bindings_db_path()
        binding_store: Any
        try:
            if db_path is None:
                raise RuntimeError("no DB path resolved")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            binding_store = SqliteWalletBindingStore(db_path)
            logger.info(
                "Sprint 794 — using SqliteWalletBindingStore at "
                "%s", db_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Sprint 794 — SqliteWalletBindingStore init failed "
                "(%s); falling back to InMemoryWalletBindingStore "
                "(bindings will NOT survive daemon restart).", exc,
            )
            binding_store = InMemoryWalletBindingStore()

        wallet_api.reset_services_for_tests()
        _sess_secret, _sess_required = _wallet_session_config_from_env()
        services = wallet_api.WalletApiServices(
            settings=wallet_api.WalletApiSettings(
                expected_domain="prsm-network.com",
                expected_chain_id=8453,
                # sp1013 — wallet session token (API-authz Residual A).
                session_secret=_sess_secret,
                session_required=_sess_required,
            ),
            nonce_store=wallet_api.InMemoryNonceStore(),
            binding_service=WalletBindingService(binding_store),
            price_source=wallet_api.StaticPriceSource(
                price_usd=0,  # type: ignore[arg-type]
            ),
            balance_lookup=wallet_api._ZeroBalanceLookup(),
            receipt_lookup=adapter,
        )
        wallet_api.set_services(services)
        logger.info(
            "Sprint 793 — WalletApiServices wired with "
            "_ReceiptStoreAdapter (receipt_store=%s)",
            "present" if receipt_store is not None else "None",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Sprint 793 — wire_wallet_api_services failed (the "
            "/devices/earnings endpoint will fall back to the "
            "no-op default): %s", exc,
        )
