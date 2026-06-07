"""Sprint 1036 — assemble the on-chain settlement client (brick 1).

``build_onchain_settlement_client_or_none`` hand-assembles a
``prsm.settlement.client.BatchSettlementClient`` (there is no factory): a default
``ReceiptAccumulator`` + a ``Web3SettlementContractClient`` (against the deployed
``BatchSettlementRegistry``, resolved from ``prsm.config.networks``) + this node's
``provider_address``. It is the on-chain cross-node settlement on-ramp — feeding
it adapter output (sprint 1035) + a commit/finalize/reconcile poll loop are the
next bricks; the actual ``commitBatch`` needs a funded settler key (a ceremony).

Gating (OFF by default):
- ``PRSM_ONCHAIN_SETTLEMENT`` must be truthy (1/true/yes) — the real opt-in.
- ``provider_address`` (the node's eth address, resolve_operator_address) must be
  given — it is the client's bound identity for the accumulate() address gate.
- the settlement_registry address must resolve (it does on mainnet by default).
- ``FTNS_WALLET_PRIVATE_KEY`` is OPTIONAL: absent → VIEW-ONLY client (local
  accumulation works; commit/finalize raise until a key exists). Present → its
  eth address MUST equal ``provider_address`` (else the eventual commitBatch,
  which settles to msg.sender == the key's address, would pay the wrong party →
  refuse, return None).

Brick 2 (the commit/finalize poll loop) MUST re-verify the signing key controls
provider_address AT WRITE TIME — a view-only build binding does not prove key
control, so before commitBatch the commit path must assert the key's eth address
== provider_address again.

Fail-open: any error (missing web3, malformed key, unresolvable network) returns
None so daemon startup never crashes; the caller logs that settlement is off.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def build_onchain_settlement_client_or_none(
    *,
    provider_address: Optional[str],
    env: Optional[dict] = None,
) -> Optional[Any]:
    """Return a ``BatchSettlementClient`` or ``None`` (see module docstring)."""
    try:
        environ = env if env is not None else os.environ
        if (environ.get("PRSM_ONCHAIN_SETTLEMENT", "") or "").strip().lower() \
                not in ("1", "true", "yes"):
            return None
        if not provider_address:
            return None

        from prsm.config.networks import resolve_endpoints
        endpoints = resolve_endpoints()
        registry = getattr(endpoints, "settlement_registry", None)
        rpc_url = getattr(endpoints, "rpc_url", None)
        if not registry or not rpc_url:
            return None

        key = (environ.get("FTNS_WALLET_PRIVATE_KEY", "") or "").strip()
        if key and not key.startswith("0x"):
            key = "0x" + key

        # Funds-safety: if a key is supplied it MUST control provider_address,
        # because commitBatch settles to msg.sender == this key's address. A
        # mismatch (or a malformed key) → refuse rather than risk paying the
        # wrong party. View-only (no key) is allowed — accumulation only.
        if key:
            from eth_account import Account
            try:
                signer = Account.from_key(key).address
            except Exception:
                logger.warning(
                    "on-chain settlement: malformed FTNS_WALLET_PRIVATE_KEY; "
                    "settlement OFF."
                )
                return None
            prov = str(provider_address).strip()
            if prov and not prov.startswith("0x"):
                prov = "0x" + prov   # tolerate a 0x-less operator address
            # .lower() compare is checksum-correct (EIP-55 is case-only).
            if signer.lower() != prov.lower():
                logger.warning(
                    "on-chain settlement: FTNS_WALLET_PRIVATE_KEY address %s "
                    "does not control provider_address %s; settlement OFF "
                    "(commit would settle to the wrong party).",
                    signer, provider_address,
                )
                return None

        from prsm.economy.web3.batch_settlement_contract_client import (
            Web3SettlementContractClient,
        )
        from prsm.settlement.accumulator import ReceiptAccumulator
        from prsm.settlement.client import BatchSettlementClient

        contract_client = Web3SettlementContractClient(
            rpc_url=rpc_url,
            contract_address=registry,
            private_key=key or None,   # None → view-only (commit/finalize defer)
        )
        return BatchSettlementClient(
            accumulator=ReceiptAccumulator(),
            contract_client=contract_client,
            provider_address=provider_address,
        )
    except Exception as exc:  # never crash daemon start on a wiring problem
        # Log so an operator can tell "off by config" from "off due to error".
        logger.debug(
            "on-chain settlement client build failed (off): %s: %s",
            type(exc).__name__, exc,
        )
        return None
