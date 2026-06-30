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


def _commit_with_attestation_selector_hex() -> str:
    """The 4-byte selector for ``commitBatchWithAttestation`` (sp1241), DERIVED
    from the registry ABI so it can never drift from the contract — the same
    ABI-independent existence check the on-chain
    ``verify-attestation-commitment-deployment.js`` gate uses. Lowercase, no 0x.
    """
    from eth_utils import function_abi_to_4byte_selector

    from prsm.economy.web3.batch_settlement_contract_client import (
        BATCH_SETTLEMENT_REGISTRY_ABI,
    )
    for entry in BATCH_SETTLEMENT_REGISTRY_ABI:
        if (
            entry.get("type") == "function"
            and entry.get("name") == "commitBatchWithAttestation"
        ):
            return function_abi_to_4byte_selector(entry).hex().lower()
    raise KeyError("commitBatchWithAttestation missing from the registry ABI")


def _bytecode_exposes_selector(code_hex: str, selector_hex: str) -> bool:
    """Solidity's external-function dispatcher embeds each 4-byte selector as a
    PUSH4 literal in the deployed runtime bytecode, so the selector hex appearing
    in the code is a reliable, ABI-independent existence check (pure helper for
    testability — no network)."""
    return bool(selector_hex) and selector_hex.lower() in (code_hex or "").lower()


def _registry_supports_attestation(rpc_url: str, address: str) -> bool:
    """FAIL-SAFE probe: does the DEPLOYED registry expose
    ``commitBatchWithAttestation``? ANY error → ``False`` — we must NEVER enable
    the attestation-commit path against a registry we can't confirm supports it,
    because a missing function would make every ``commitBatch`` write revert and
    stall the settlement cycle (escrow already locked, commit never lands)."""
    try:
        from web3 import Web3  # local import: web3 is an optional dependency
        sel = _commit_with_attestation_selector_hex()
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        code = w3.eth.get_code(Web3.to_checksum_address(address))
        return _bytecode_exposes_selector(code.hex(), sel)
    except Exception:  # noqa: BLE001 — unconfirmable → fail safe to OFF
        return False


def build_onchain_settlement_client_or_none(
    *,
    provider_address: Optional[str],
    env: Optional[dict] = None,
    published_batch_store: Optional[Any] = None,
) -> Optional[Any]:
    """Return a ``BatchSettlementClient`` or ``None`` (see module docstring).

    ``published_batch_store`` (sp1140 Brick E.2) is OPT-IN: passed only when the audit
    data plane is enabled (``PRSM_SETTLEMENT_AUDIT``). When None (the default) the client
    is constructed exactly as before — the retention put on the commit-success path is a
    strict no-op (``client.py`` only puts when a store was injected), so default-off
    behavior is byte-for-byte unchanged."""
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

        # sp1299 — make the TEE Tier-3 attestation-commit path (sp1241) a CONFIG
        # flip, not a code change: the Phase-4 cutover's final + most consequential
        # step ("flip supports_attestation") is now PRSM_SETTLEMENT_SUPPORTS_ATTESTATION.
        # Default OFF → legacy commitBatch, byte-for-byte unchanged on the live path.
        # When requested, FAIL-SAFE verify the deployed registry actually exposes
        # commitBatchWithAttestation BEFORE enabling — flipping it against an
        # un-upgraded registry would make every commit tx revert and stall settlement.
        supports_attestation = False
        if (environ.get("PRSM_SETTLEMENT_SUPPORTS_ATTESTATION", "") or "").strip()\
                .lower() in ("1", "true", "yes", "on"):
            supports_attestation = _registry_supports_attestation(rpc_url, registry)
            if supports_attestation:
                logger.info(
                    "on-chain settlement: attestation-commit ENABLED — registry %s "
                    "exposes commitBatchWithAttestation (TEE Tier-3 roadmap F).",
                    registry,
                )
            else:
                logger.warning(
                    "PRSM_SETTLEMENT_SUPPORTS_ATTESTATION is set, but registry %s "
                    "does not expose commitBatchWithAttestation (or it could not be "
                    "confirmed); attestation-commit stays OFF (legacy commitBatch). "
                    "Point PRSM_SETTLEMENT_REGISTRY_ADDRESS at the sp1240 bundle first.",
                    registry,
                )

        contract_client = Web3SettlementContractClient(
            rpc_url=rpc_url,
            contract_address=registry,
            private_key=key or None,   # None → view-only (commit/finalize defer)
            supports_attestation=supports_attestation,
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
            published_batch_store=published_batch_store,
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
    max_spend_wei: Optional[int] = None,
    inference_receipt_store: Any = None,
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
        # sp1058 review MEDIUM #3 — defense-in-depth price ceiling: when a paying
        # requester supplied a signed max_spend, the value we book on-chain against
        # THEIR escrow must not exceed what they authorized. The escrow clamp makes
        # this hold today, but enforce it against the ACTUAL settled value so a
        # future re-pricing/top-up path can't silently overrun the signed ceiling.
        if max_spend_wei is not None and value_wei > int(max_spend_wei):
            logger.warning(
                "on-chain settlement skipped: settled value %s wei exceeds the "
                "requester's signed max_spend %s wei (job=%s)",
                value_wei, max_spend_wei, job_id,
            )
            return "skipped:exceeds-max-spend"
        if executed_at_unix is None:
            import time as _time
            executed_at_unix = int(_time.time())

        # §7-Brick-1 (sp1141): ADDITIVE + opt-in. When an InferenceReceiptStore is
        # provided (default None => behavior byte-for-byte unchanged), retain the ORIGINAL
        # §7 InferenceReceipt keyed by the produced batch's leaf hash
        # (hash_leaf(batched_receipt_to_leaf(batched)) — IDENTICAL to what the committed
        # batch carries), so an observer holding a committed batch leaf can look the §7
        # receipt up to drive verify_inference_receipt_for_challenge. The retention is
        # wrapped (in adapt_and_retain_inference_receipt) so a §7 retention failure NEVER
        # unwinds the produced BatchedReceipt — the money path is sacred + the adapter math
        # is unchanged.
        from prsm.settlement.inference_receipt_store import (
            adapt_and_retain_inference_receipt,
        )
        batched = adapt_and_retain_inference_receipt(
            receipt=receipt,
            identity=identity,
            store=inference_receipt_store,
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


# Phase order: FIRST recover commit-intents orphaned by a crash in the
# commit→persist window (sp1040 — adopt any that landed so they get finalized,
# before we commit anything new); then adopt broadcast-but-unconfirmed commits
# that landed (so a restart-or-blip doesn't re-commit and double-settle) ->
# commit ready batches -> finalize those past the challenge window -> mark
# on-chain-finalized.
_POLL_PHASES = (
    ("recover", "recover_committing_intents"),
    ("reconcile_pending", "reconcile_pending_commits"),
    ("commit", "commit_ready_batches"),
    ("finalize", "finalize_ready_batches"),
    ("reconcile_finalized", "reconcile_finalized"),
)


async def resolve_paid_requester(
    *,
    verifier: Optional[Any],
    payment_authorization: Optional[dict],
    request_hash: Any,
    quoted_price_wei: int,
) -> Optional[str]:
    """Sprint 1056 (brick 3) — turn an optional signed PaymentAuthorization from
    the inference request body into an authenticated requester eth address.

    Fail-CLOSED on the authorization (a bad/paid request must not silently serve):
      - ``payment_authorization`` None → return None (unchanged self-pay path).
      - present but ``verifier`` None → AuthorizationRejected('verifier-not-wired')
        (the node can't verify a paid job; do not downgrade to self-pay silently).
      - present but missing payload/signature → 'malformed-authorization'.
      - present + verifier → ``verifier.verify`` returns the authenticated
        requester, or raises AuthorizationRejected (propagated to the caller).

    The caller (api.py) computes ``request_hash`` from the ACTUAL request body via
    ``canonical_request_hash`` so the request-hash binding covers this exact work,
    and passes the provider's quote as ``quoted_price_wei``.
    """
    from prsm.settlement.payment_authorization_verifier import AuthorizationRejected

    if payment_authorization is None:
        return None
    if verifier is None:
        raise AuthorizationRejected(
            "verifier-not-wired",
            "a payment_authorization was supplied but on-chain settlement / the "
            "payment verifier is not enabled on this node",
        )
    if not isinstance(payment_authorization, dict):
        raise AuthorizationRejected("malformed-authorization", "not an object")
    payload = payment_authorization.get("payload")
    signature = payment_authorization.get("signature")
    if not isinstance(payload, dict) or not signature:
        raise AuthorizationRejected(
            "malformed-authorization",
            "expected {payload: {...}, signature: '0x..'}",
        )
    return await verifier.verify(
        payload, signature, request_hash=request_hash,
        quoted_price_wei=quoted_price_wei,
    )


async def resolve_relayer_requester(
    *,
    verifier: Optional[Any],
    payment_authorization: Optional[dict],
    payment_delegation: Optional[dict],
    request_hash: Any,
    quoted_price_wei: int,
) -> Optional[str]:
    """Sprint 1094 (relayer brick 4) — turn a relayer-signed PaymentAuthorization plus
    the funder's PaymentDelegation from the inference body into the authenticated FUNDING
    requester eth address (whose escrow pays). Fail-CLOSED, mirroring
    ``resolve_paid_requester``:
      - no ``payment_delegation`` → return None (caller falls back to the self-signed path).
      - delegation present but ``verifier`` None → AuthorizationRejected('verifier-not-wired').
      - either part malformed → 'malformed-authorization' / 'malformed-delegation'.
      - else → ``verifier.verify`` returns the funder, or raises AuthorizationRejected.
    """
    from prsm.settlement.payment_authorization_verifier import AuthorizationRejected

    if payment_delegation is None:
        return None
    if verifier is None:
        raise AuthorizationRejected(
            "verifier-not-wired",
            "a payment_delegation was supplied but the relayer payment verifier is "
            "not enabled on this node")
    if not isinstance(payment_authorization, dict):
        raise AuthorizationRejected("malformed-authorization", "not an object")
    if not isinstance(payment_delegation, dict):
        raise AuthorizationRejected("malformed-delegation", "not an object")
    auth_payload = payment_authorization.get("payload")
    auth_sig = payment_authorization.get("signature")
    deleg_payload = payment_delegation.get("payload")
    deleg_sig = payment_delegation.get("signature")
    if not isinstance(auth_payload, dict) or not auth_sig:
        raise AuthorizationRejected(
            "malformed-authorization", "expected {payload: {...}, signature: '0x..'}")
    if not isinstance(deleg_payload, dict) or not deleg_sig:
        raise AuthorizationRejected(
            "malformed-delegation", "expected {payload: {...}, signature: '0x..'}")
    return await verifier.verify(
        auth_payload, auth_sig, deleg_payload, deleg_sig,
        request_hash=request_hash, quoted_price_wei=quoted_price_wei)


def _resolve_verifier_chain_id(environ) -> Optional[int]:
    """Sprint 1201 — the EIP-712 domain chainId the requester-payment verifier MUST use
    to recover the signer.

    The payment authorization is signed over a domain that includes chainId; the
    verifier must recompute the digest with the SAME chainId or recovery yields a
    different address ("signer-mismatch"). Pre-fix, the verifier passed no chainId and
    ``verify_payment_authorization_signature`` fell back to DEFAULT_CHAIN_ID=8453
    (Base mainnet) — so on a TESTNET node (chainId 84532) every testnet-signed auth was
    rejected. Resolve the node's active network chain so signer and verifier agree.
    Fail-soft to None (the verifier then uses its 8453 default, the prior behavior)."""
    try:
        from prsm.config.networks import resolve_endpoints
        return int(resolve_endpoints(environ.get("PRSM_NETWORK")).chain_id)
    except Exception:  # noqa: BLE001
        return None


def build_relayer_verifier_or_none(
    environ=None, *, provider_address: Optional[str] = None,
    balance_reader: Optional[Any] = None,
) -> Optional[Any]:
    """Sprint 1094 (relayer brick 4) — build a RelayerAuthorizationVerifier for the node,
    or None. Same gate as the self-signed verifier (``PRSM_REQUESTER_PAYMENT`` truthy +
    a known ``provider_address``) — the relayer path is part of requester-payment. Uses
    durable nonce + budget stores (``PRSM_RELAYER_NONCE_FILE`` /
    ``PRSM_DELEGATION_BUDGET_FILE``, default under ~/.prsm; ``:memory:`` → in-memory) so a
    restart can't reopen a replay window or already-spent delegation budget. Fail-open:
    any error returns None + logs."""
    import os as _os
    from pathlib import Path

    environ = _os.environ if environ is None else environ
    try:
        flag = str(environ.get("PRSM_REQUESTER_PAYMENT", "")).strip().lower()
        if flag not in ("1", "true", "yes", "on"):
            return None
        if not provider_address:
            logger.warning(
                "PRSM_REQUESTER_PAYMENT set but no provider_address resolved; "
                "relayer payment verification stays OFF")
            return None
        from prsm.settlement.payment_authorization_verifier import (
            DurableNonceStore, InMemoryNonceStore, RelayerAuthorizationVerifier,
        )
        from prsm.settlement.delegation_budget import (
            DurableDelegationBudgetStore, InMemoryDelegationBudgetStore,
        )
        nonce_file = str(environ.get("PRSM_RELAYER_NONCE_FILE", "")).strip()
        if nonce_file == ":memory:":
            nonce_store: Any = InMemoryNonceStore()
        else:
            nonce_store = DurableNonceStore(
                nonce_file or str(Path.home() / ".prsm" / "relayer_nonces.json"))
        budget_file = str(environ.get("PRSM_DELEGATION_BUDGET_FILE", "")).strip()
        if budget_file == ":memory:":
            budget_store: Any = InMemoryDelegationBudgetStore()
        else:
            budget_store = DurableDelegationBudgetStore(
                budget_file or str(Path.home() / ".prsm" / "delegation_budgets.json"))
        return RelayerAuthorizationVerifier(
            provider_address=provider_address,
            nonce_store=nonce_store, budget_store=budget_store,
            balance_reader=balance_reader,
            chain_id=_resolve_verifier_chain_id(environ),  # sp1201 — match the signer
        )
    except Exception as exc:  # noqa: BLE001 - never crash startup over this
        logger.warning(
            "relayer payment verifier build failed (staying OFF): %s: %s",
            type(exc).__name__, exc)
        return None


def build_payment_verifier_or_none(
    environ=None, *, provider_address: Optional[str] = None,
    balance_reader: Optional[Any] = None,
) -> Optional[Any]:
    """Sprint 1056 (brick 3) — build a PaymentAuthorizationVerifier for the node, or
    None when the requester-payment path is not enabled.

    Gated ON only when ``PRSM_REQUESTER_PAYMENT`` is truthy AND a
    ``provider_address`` is known (the verifier pins this provider). The nonce store
    is durable (``PRSM_PAYMENT_NONCE_FILE``, default ~/.prsm/payment_nonces.json) so
    a restart can't reopen a replay window; ``:memory:`` selects the in-memory store.
    ``balance_reader`` (an async callable like EscrowPoolClient.balance_of) is wired
    by the caller when an RPC is available; absent → pre-flight skipped (finalizeBatch
    still enforces funding). Fail-open: any error returns None + logs."""
    import os as _os
    from pathlib import Path

    environ = _os.environ if environ is None else environ
    try:
        flag = str(environ.get("PRSM_REQUESTER_PAYMENT", "")).strip().lower()
        if flag not in ("1", "true", "yes", "on"):
            return None
        if not provider_address:
            logger.warning(
                "PRSM_REQUESTER_PAYMENT set but no provider_address resolved; "
                "requester-payment verification stays OFF")
            return None
        from prsm.settlement.payment_authorization_verifier import (
            DurableNonceStore, InMemoryNonceStore, PaymentAuthorizationVerifier,
        )
        nonce_file = str(environ.get("PRSM_PAYMENT_NONCE_FILE", "")).strip()
        if nonce_file == ":memory:":
            store: Any = InMemoryNonceStore()
        else:
            path = nonce_file or str(Path.home() / ".prsm" / "payment_nonces.json")
            store = DurableNonceStore(path)
        return PaymentAuthorizationVerifier(
            provider_address=provider_address,
            nonce_store=store,
            balance_reader=balance_reader,
            chain_id=_resolve_verifier_chain_id(environ),  # sp1201 — match the signer
        )
    except Exception as exc:  # noqa: BLE001 - never crash startup over this
        logger.warning(
            "requester-payment verifier build failed (staying OFF): %s: %s",
            type(exc).__name__, exc)
        return None


def get_settlement_status(client: Optional[Any]) -> dict:
    """sp1051 — operator-facing on-chain-settlement status for /health surfaces.
    ``client`` is the node's built BatchSettlementClient (or None when settlement
    is off). When off, returns {enabled: False, reason}; when on, wraps the
    client's status() snapshot under enabled=True."""
    if client is None:
        return {
            "enabled": False,
            "reason": "on-chain settlement not active (PRSM_ONCHAIN_SETTLEMENT "
                      "off, no resolvable network, key/provider mismatch, or a "
                      "corrupt state file turned it off — see daemon startup logs)",
        }
    try:
        st = {"enabled": True}
        st.update(client.status())
        return st
    except Exception as exc:  # noqa: BLE001 - status must never raise on a health path
        return {"enabled": True, "status_error": f"{type(exc).__name__}: {exc}"}


def multistage_settlement_enabled(environ=os.environ) -> bool:
    """sp1318 (S3b-2b) — is the big-model multi-stage paid-settlement data plane ON?

    DEFAULT-OFF: the proven single-stage settlement path is byte-for-byte unchanged unless
    ``PRSM_MULTISTAGE_SETTLEMENT`` is explicitly truthy (1/true/yes/on). Until the on-chain
    commit brick (S3b-3) lands + a testnet 2-node proof passes, this gates only the receive
    plumbing (the routed-task delivery endpoint + receiver store)."""
    return str(environ.get("PRSM_MULTISTAGE_SETTLEMENT", "")).strip().lower() in (
        "1", "true", "yes", "on")


def resolve_per_stage_receiver_store(node: Any, environ=os.environ) -> Optional[Any]:
    """sp1318 (S3b-2b) — the node's staged-task receiver store, lazily built + cached, gated.

    Returns ``None`` when ``PRSM_MULTISTAGE_SETTLEMENT`` is off (the delivery endpoint then 503s
    — a node that hasn't opted into multi-stage settlement accepts no routed tasks). When on,
    builds a ``PerStageReceiverStore`` once and caches it on
    ``node._settlement_per_stage_receiver_store`` (path from
    ``PRSM_MULTISTAGE_RECEIVER_STORE_FILE`` or ``~/.prsm/per_stage_receiver.json``). Never raises
    — a build failure logs + returns None (the endpoint then 503s, never a silent accept)."""
    if not multistage_settlement_enabled(environ):
        return None
    existing = getattr(node, "_settlement_per_stage_receiver_store", None)
    if existing is not None:
        return existing
    try:
        from pathlib import Path

        from prsm.settlement.per_stage_receiver_store import PerStageReceiverStore
        path = (
            str(environ.get("PRSM_MULTISTAGE_RECEIVER_STORE_FILE", "")).strip()
            or str(Path.home() / ".prsm" / "per_stage_receiver.json")
        )
        store = PerStageReceiverStore(path)
    except Exception as exc:  # noqa: BLE001 — never crash the delivery path on a store build error
        logger.warning(
            "resolve_per_stage_receiver_store: build failed (%s: %s); routed-task delivery "
            "unavailable this run", type(exc).__name__, exc)
        return None
    node._settlement_per_stage_receiver_store = store
    return store


def build_per_stage_endpoint_resolver(node: Any, environ=os.environ):
    """sp1321 (S3b-3b) — a ``node_id -> Optional[base_url]`` resolver for delivering per-stage
    settlement tasks to their stage nodes.

    Reuses the existing endpoint-resolver backends (sp's aggregate_endpoint_resolver): an
    operator-curated static map (``PRSM_MULTISTAGE_ENDPOINT_MAP``, JSON ``{node_id: base_url}``)
    wins over the transport-peer fallback (host:port from the live WS connection; scheme/port
    overridable via ``PRSM_MULTISTAGE_ENDPOINT_SCHEME``/``_PORT``). Returns a callable that yields
    ``None`` on an unresolved node (so the FAIL-SOFT sender records it undelivered rather than
    raising). When neither a static map nor a transport exists, every lookup is ``None``."""
    import json

    from prsm.compute.query_orchestrator.aggregate_endpoint_resolver import (
        AggregateEndpointUnresolvedError,
        ChainedEndpointResolver,
        StaticMapEndpointResolver,
        TransportPeerEndpointResolver,
    )

    resolvers = []
    static_raw = str(environ.get("PRSM_MULTISTAGE_ENDPOINT_MAP", "")).strip()
    if static_raw:
        try:
            resolvers.append(StaticMapEndpointResolver(json.loads(static_raw)))
        except (ValueError, TypeError) as exc:
            logger.warning(
                "PRSM_MULTISTAGE_ENDPOINT_MAP malformed (%s) — ignoring static map", exc)
    transport = getattr(node, "transport", None)
    if transport is not None:
        scheme = str(environ.get("PRSM_MULTISTAGE_ENDPOINT_SCHEME", "https")).strip() or "https"
        port_raw = str(environ.get("PRSM_MULTISTAGE_ENDPOINT_PORT", "")).strip()
        try:
            resolvers.append(TransportPeerEndpointResolver(
                transport, scheme=scheme,
                aggregate_port=int(port_raw) if port_raw else None))
        except ValueError as exc:
            logger.warning("per-stage transport endpoint resolver build failed (%s)", exc)
    if not resolvers:
        return lambda node_id: None
    chained = ChainedEndpointResolver(resolvers)

    def _resolve(node_id: str) -> Optional[str]:
        try:
            return chained(node_id)
        except AggregateEndpointUnresolvedError:
            return None

    return _resolve


def deliver_for_settled_receipt(
    node: Any,
    *,
    receipt: Any,
    total_value_wei: int,
    requester_address: str,
    per_stage_authorization: Optional[dict] = None,
    wallet_map: Any = None,
    http_post=None,
    environ=os.environ,
    timeout: float = 10.0,
) -> list:
    """sp1321 (S3b-3b) — node-aware orchestrator entrypoint: split a settled multi-stage receipt
    + DELIVER each per-node task to its stage node (the send half, S3b-2c) using this node's peer
    registry to resolve endpoints.

    GATED: returns ``[]`` when ``PRSM_MULTISTAGE_SETTLEMENT`` is off (the proven single-stage path
    is untouched). Resolves the endpoint resolver from ``node`` + builds the wallet map from env
    when not supplied, then fans out via ``deliver_settled_multistage_tasks`` (FAIL-SOFT — a miss
    is a recorded undelivered result, never a raise; the on-chain commit is each node's own
    concern via the poll-loop commit driver). Returns the per-node ``DeliveryResult`` list."""
    if not multistage_settlement_enabled(environ):
        return []
    from prsm.settlement.per_stage_delivery_client import (
        deliver_settled_multistage_tasks,
    )
    if wallet_map is None:
        from prsm.node.compute_wallet_map import ComputeWalletMap
        wallet_map = ComputeWalletMap.from_env()
    resolver = build_per_stage_endpoint_resolver(node, environ)
    return deliver_settled_multistage_tasks(
        receipt=receipt, total_value_wei=total_value_wei,
        requester_address=requester_address, endpoint_for_node=resolver,
        per_stage_authorization=per_stage_authorization, wallet_map=wallet_map,
        http_post=http_post, timeout=timeout)


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
