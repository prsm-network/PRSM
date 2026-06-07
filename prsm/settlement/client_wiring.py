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

from prsm.settlement.state_store import (
    SettlementStateCorruptError,
    SettlementStateStore,
)

logger = logging.getLogger(__name__)


def _resolve_state_store(environ) -> Optional[SettlementStateStore]:
    """Durable-state store for the settlement client, ON by default when
    settlement is opted in. ``PRSM_SETTLEMENT_STATE_FILE`` overrides the path;
    the literal ``:memory:`` disables durability (in-memory only)."""
    from pathlib import Path
    configured = (environ.get("PRSM_SETTLEMENT_STATE_FILE", "") or "").strip()
    if configured == ":memory:":
        return None
    path = Path(configured) if configured else (
        Path.home() / ".prsm" / "settlement_state.json"
    )
    return SettlementStateStore(path)


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
        # sp1039 (brick 2.5) — durable post-commit state, ON by default whenever
        # settlement is opted in: committed batches + the broadcast-but-unconfirmed
        # quarantine represent escrow already locked on chain, so they MUST survive
        # a restart or the money strands. PRSM_SETTLEMENT_STATE_FILE overrides the
        # path; ":memory:" disables durability (in-memory only — for tests/diagnostics).
        state_store = _resolve_state_store(environ)
        return BatchSettlementClient(
            accumulator=ReceiptAccumulator(),
            contract_client=contract_client,
            provider_address=provider_address,
            state_store=state_store,
        )
    except SettlementStateCorruptError as exc:
        # A corrupt money-state file: refuse to start settlement with empty state
        # (that would silently forget — and strand — every committed batch). Loud
        # so an operator restores/inspects the file. Settlement stays OFF.
        logger.error(
            "on-chain settlement OFF: settlement state file is corrupt (%s). "
            "Restore it from backup or inspect before re-enabling — starting with "
            "empty state would strand committed escrow.", exc,
        )
        return None
    except Exception as exc:  # never crash daemon start on a wiring problem
        # Log so an operator can tell "off by config" from "off due to error".
        logger.debug(
            "on-chain settlement client build failed (off): %s: %s",
            type(exc).__name__, exc,
        )
        return None


async def accumulate_settled_inference_receipt(
    *,
    client: Any,
    identity: Any,
    provider_address: Optional[str],
    receipt: Any,
    release_ftns: Any,
    job_id: str,
    requester_address: Optional[str] = None,
    executed_at_unix: Optional[int] = None,
) -> str:
    """Sprint 1037 (brick 1.5) — feed a just-settled InferenceReceipt into the
    on-chain settlement accumulator.

    Called from the /compute/inference settle path AFTER the off-chain escrow
    release: adapts the receipt (sprint 1035) to a BatchedReceipt and calls
    ``client.accumulate``. No-op when settlement is off (``client`` is None).

    FAIL-OPEN: never raises and never unwinds the completed off-chain
    settlement — an accumulation problem must not fail the inference response.
    Returns a status string for the caller to log: 'accumulated' |
    'skipped:<reason>' | 'error:<Type>'.

    Self-escrow note: today's /compute/inference escrows the requester as the
    local node, so ``requester_address`` defaults to ``provider_address`` (a
    self-settlement). Pass a distinct ``requester_address`` once a paying
    requester supplies its eth address (a future API change).
    """
    try:
        if client is None:
            return "skipped:no-client"
        if not provider_address:
            return "skipped:no-provider-address"
        from decimal import Decimal
        rel = (
            release_ftns if isinstance(release_ftns, Decimal)
            else Decimal(str(release_ftns))
        )
        if rel <= 0:
            return "skipped:zero-release"
        # Cross-host multi-stage: the off-chain settle split `release_ftns`
        # across N stage nodes (sprint 1031). Booking the FULL amount to one
        # provider on-chain would over-attribute (a conservation mismatch once
        # the commit path goes live). Skip until per-stage on-chain accumulation
        # (brick 2). Uses the SAME split helper the off-chain path uses, so the
        # two ledgers stay consistent.
        from prsm.economy.credit_policy import split_release_across_stages
        if split_release_across_stages(receipt, rel) is not None:
            return "skipped:multi-stage-deferred"
        value_wei = int(rel * (Decimal(10) ** 18))   # FTNS -> wei (18 dec)
        if value_wei <= 0:
            return "skipped:sub-wei"   # release below 1 wei rounds to 0
        if executed_at_unix is None:
            import time as _time
            executed_at_unix = int(_time.time())

        from prsm.settlement.inference_adapter import (
            inference_receipt_to_batched_receipt,
        )
        batched = inference_receipt_to_batched_receipt(
            receipt=receipt,
            identity=identity,
            requester_address=requester_address or provider_address,
            provider_address=provider_address,
            value_ftns=value_wei,
            local_escrow_id=job_id,
            executed_at_unix=executed_at_unix,
        )
        await client.accumulate(batched)
        return "accumulated"
    except Exception as exc:  # noqa: BLE001 — never unwind the off-chain settle
        logger.warning(
            "on-chain settlement accumulate failed (non-fatal; off-chain "
            "settlement already stands): %s: %s", type(exc).__name__, exc,
        )
        return f"error:{type(exc).__name__}"


# Phase order: adopt broadcast-but-unconfirmed commits that landed (so a
# restart-or-blip doesn't re-commit and double-settle) -> commit ready batches
# -> finalize those past the challenge window -> mark on-chain-finalized.
_POLL_PHASES = (
    ("reconcile_pending", "reconcile_pending_commits"),
    ("commit", "commit_ready_batches"),
    ("finalize", "finalize_ready_batches"),
    ("reconcile_finalized", "reconcile_finalized"),
)


async def run_settlement_poll_cycle(client: Any) -> dict:
    """Sprint 1038 (brick 2) — drive ONE commit/finalize/reconcile cycle of the
    on-chain settlement client.

    Each phase is ISOLATED (one failing does not block the others) and the cycle
    NEVER raises — it runs on a detached background task (see the node poll
    loop). Returns a per-phase status dict {'reconcile_pending'|'commit'|
    'finalize'|'reconcile_finalized': 'ok'|<repr>|'error:<Type>'} for logging.

    With the default VIEW-ONLY client (no funded settler key) the commit/finalize
    phases raise (private_key required) and are recorded as errors — the cycle is
    inert until the funded-key ceremony. DURABLE batch state (brick 2.5) is
    required BEFORE that ceremony: today's in-memory _tracked/_pending_commits do
    not survive a restart, so a crash mid-quarantine could strand a
    broadcast-but-unconfirmed commit (the sp1022 double-settle guard relies on
    that quarantine surviving).
    """
    results: dict = {}
    for status_key, method_name in _POLL_PHASES:
        try:
            method = getattr(client, method_name)
            res = await method()
            results[status_key] = "ok" if res is None else str(res)
        except Exception as exc:  # noqa: BLE001 — phase isolation; never raise
            results[status_key] = f"error:{type(exc).__name__}"
    return results
