"""Front-door end-to-end validation harness (sprint 1200).

A single-command proof of the day-one-live bar — *can a REAL user, on REAL infra,
get REAL results WITHOUT becoming an operator?* — exercised live on Base Sepolia:

    faucet  ->  deposit  ->  pay-infer

asserting each step against authoritative reads (on-chain balances + the signed
receipt), not hopeful wallet deltas. Mirrors ``prsm/settlement/e2e_proof.py`` (the
settlement activation proof): pure composition, every client injected as an
``Any`` parameter so the orchestration is fully unit-testable without a chain, and
a HARD mainnet kill-switch — this is a TESTNET validation tool and refuses to run
against Base mainnet (it would dispense from a real faucet / move real-value FTNS).

What each step proves:
  - faucet (optional): the on-chain testnet faucet (sp1190) dispenses real testnet
    FTNS — the recipient's on-chain balance rises. The new-user on-ramp.
  - deposit (sp1189 deposit_escrow): the requester self-funds the EscrowPool — its
    on-chain escrow balance rises by the deposit.
  - pay-infer (sp1189 pay_and_infer): a signed EIP-712 PaymentAuthorization is
    ACCEPTED (not 402), and the node returns REAL inference output + a signed §7
    receipt. The on-chain A->B settle itself runs later over the 3-day challenge
    window via the settlement poll loop (already proven end-to-end on mainnet), so
    this step asserts the IMMEDIATE deliverables (output + receipt + acceptance) and
    only REPORTS the escrow balance — it does not gate on an instant debit.
"""
from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

BASE_MAINNET_CHAIN_ID = 8453
BASE_SEPOLIA_CHAIN_ID = 84532
_WEI = Decimal(10) ** 18


class FrontDoorValidationError(RuntimeError):
    """A front-door validation step failed an assertion (or the chain is wrong)."""


def assert_testnet_only(*, chain_id: int) -> None:
    """HARD kill-switch. This harness validates the TESTNET front door and must never
    run against mainnet — it dispenses from a faucet and moves FTNS. Refuses any chain
    other than Base Sepolia (84532). Defense-in-depth with the faucet's own sp1190
    kill-switch and the deposit/settle clients."""
    if int(chain_id) != BASE_SEPOLIA_CHAIN_ID:
        raise FrontDoorValidationError(
            f"front-door e2e is TESTNET-ONLY: refusing chainId {chain_id} "
            f"(expected {BASE_SEPOLIA_CHAIN_ID} = Base Sepolia). This tool dispenses "
            f"from a faucet and moves FTNS — it never runs against mainnet."
        )


async def _poll_async(
    read: Callable[[], Awaitable[int]],
    predicate: Callable[[int], bool],
    *,
    attempts: int = 8,
    delay_s: float = 2.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> int:
    """Poll an async integer read until ``predicate`` holds (rides past Base public-RPC
    read-replica lag — the sp1047/sp1048 lesson) or attempts exhaust. Returns the last
    value read either way (the caller decides pass/fail)."""
    value = await read()
    for _ in range(max(0, attempts - 1)):
        if predicate(value):
            return value
        await sleep(delay_s)
        value = await read()
    return value


def _poll_sync(
    read: Callable[[], int],
    predicate: Callable[[int], bool],
    *,
    attempts: int = 8,
    delay_s: float = 2.0,
    sleep: Callable[[float], None] = None,
) -> int:
    import time as _time
    _sleep = sleep or _time.sleep
    value = read()
    for _ in range(max(0, attempts - 1)):
        if predicate(value):
            return value
        _sleep(delay_s)
        value = read()
    return value


def _verify_receipt_safe(receipt_dict: Optional[dict], public_key_b64: Optional[str]):
    """Three-state receipt check (never raises):
      - ``False`` — absent, doesn't parse as an InferenceReceipt, OR carries no
        settler signature (an unsigned/decorative dict is NOT a "signed receipt").
      - ``None``  — parses AND is signed, but we can't verify the signature without
        the operator's public key.
      - ``True/False`` — a real Ed25519 signature check when ``public_key_b64`` is given.

    Review sp1200 (medium): a raw ``bool(receipt)`` truthiness check let a node return
    ``output`` + any non-empty dict and falsely PASS the "front door works" proof. The
    parse + signature-presence gate below makes a receipt a real assertion even on the
    default (no-pubkey) run."""
    if not receipt_dict:
        return False
    try:
        from prsm.compute.inference.models import InferenceReceipt
        from prsm.compute.inference.receipt import verify_receipt
        receipt = InferenceReceipt.from_dict(receipt_dict)
        if not getattr(receipt, "settler_signature", b""):
            return False  # parsed but unsigned → not a settled receipt
        if public_key_b64:
            return bool(verify_receipt(receipt, public_key_b64=public_key_b64))
        return None  # parsed + signed; signature unverifiable without the operator pubkey
    except Exception as exc:  # noqa: BLE001
        logger.warning("receipt verification raised: %s", exc)
        return False


async def run_faucet_step(
    *,
    faucet_post: Callable[[dict], dict],
    balance_reader: Any,
    recipient: str,
    amount_ftns: Optional[float] = None,
    confirm_attempts: int = 8,
    confirm_delay_s: float = 2.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Dict[str, Any]:
    """POST the on-chain testnet faucet for ``recipient`` and assert its on-chain FTNS
    balance rises. ``faucet_post(body)`` returns the endpoint JSON; ``balance_reader``
    has a sync ``balance_wei(addr)``."""
    before = int(balance_reader.balance_wei(recipient))
    body: Dict[str, Any] = {"destination_address": recipient}
    if amount_ftns is not None:
        body["amount"] = amount_ftns
    resp = faucet_post(body) or {}
    tx = resp.get("tx_hash")
    if not tx:
        raise FrontDoorValidationError(
            f"faucet returned no tx_hash (rejected?): {resp.get('detail') or resp}")
    after = await _poll_async(
        lambda: _as_awaitable(int(balance_reader.balance_wei(recipient))),
        lambda v: v > before, attempts=confirm_attempts, delay_s=confirm_delay_s, sleep=sleep)
    rose = after > before
    return {
        "step": "faucet", "ok": rose, "tx_hash": tx,
        "dispensed_ftns": resp.get("dispensed_ftns"),
        "recipient": recipient,
        "balance_before_wei": before, "balance_after_wei": after,
        "note": "on-chain FTNS balance rose" if rose else
                "balance not yet risen (RPC lag) — re-check the tx",
    }


async def _as_awaitable(value):
    return value


async def run_deposit_step(
    *,
    client: Any,
    requester_key: str,
    requester_address: str,
    amount_ftns: float,
    escrow_client: Any,
    network: str = "testnet",
    expected_chain_id: Optional[int] = None,
    confirm_attempts: int = 8,
    confirm_delay_s: float = 2.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Dict[str, Any]:
    """Deposit ``amount_ftns`` into the EscrowPool via the SDK and assert the
    requester's on-chain escrow balance rises by (at least) the deposit. ``client`` is
    a PRSMClient; ``escrow_client.balance_of(addr)`` is an async read (no key needed).
    ``expected_chain_id`` (when set) pins the deposit signer to that chain (review #1)."""
    before = int(await escrow_client.balance_of(requester_address))
    tx = await client.deposit_escrow(
        requester_key=requester_key, amount_ftns=amount_ftns, network=network,
        expected_chain_id=expected_chain_id)
    if not tx:
        raise FrontDoorValidationError("deposit returned no tx hash")
    expected_wei = int(Decimal(str(amount_ftns)) * _WEI)

    async def _read_escrow() -> int:
        return int(await escrow_client.balance_of(requester_address))

    after = await _poll_async(
        _read_escrow, lambda v: v >= before + expected_wei,
        attempts=confirm_attempts, delay_s=confirm_delay_s, sleep=sleep)
    delta = after - before
    ok = delta >= expected_wei
    if not ok:
        # surface but don't hard-raise: RPC lag could still resolve; the report shows it
        logger.warning("deposit escrow delta %s < expected %s (RPC lag?)", delta, expected_wei)
    return {
        "step": "deposit", "ok": ok, "deposit_tx": tx,
        "amount_ftns": amount_ftns,
        "escrow_before_wei": before, "escrow_after_wei": after,
        "escrow_delta_wei": delta, "expected_wei": expected_wei,
        "note": "escrow funded" if ok else "escrow not yet credited (RPC lag) — re-check",
    }


async def run_payinfer_step(
    *,
    client: Any,
    prompt: str,
    requester_key: str,
    escrow_client: Any,
    requester_address: str,
    verify_pubkey_b64: Optional[str] = None,
    payinfer_kwargs: Optional[dict] = None,
) -> Dict[str, Any]:
    """Pay for one inference. Asserts the IMMEDIATE deliverables: the auth was ACCEPTED
    (not 402), the node returned real output, and a signed receipt is present (verified
    when a pubkey is supplied). The on-chain A->B settle runs later over the challenge
    window, so escrow balance is REPORTED, not gated."""
    kwargs = dict(payinfer_kwargs or {})
    escrow_before = int(await escrow_client.balance_of(requester_address))
    result = await client.pay_and_infer(
        prompt, requester_key=requester_key, verify_pubkey_b64=verify_pubkey_b64, **kwargs)
    result = result or {}
    output = result.get("output")
    accepted = bool(result.get("success")) or (output is not None)
    if not accepted:
        # Distinguish WHERE it failed (review-of-live-run sp1202): a 402 means the
        # PAYMENT was rejected; a success:False/error body with a job_id means the
        # payment was ACCEPTED but the inference itself failed (e.g. unknown model).
        # Conflating them ("payment REJECTED") misleads — payment may have worked.
        detail = result.get("detail")
        err = result.get("error")
        if detail and ("payment" in str(detail).lower()
                       or "authorization" in str(detail).lower()):
            raise FrontDoorValidationError(
                "payment authorization REJECTED (the node did not honor the paid "
                "request): " + str(detail))
        if err or result.get("job_id"):
            raise FrontDoorValidationError(
                "payment was ACCEPTED but the inference FAILED (not a payment problem): "
                + str(err or detail or result))
        raise FrontDoorValidationError(
            "pay-infer returned no output: " + str(detail or result))
    receipt = result.get("receipt")
    # parsed: False = absent/unparseable/UNSIGNED; None = parsed+signed (no pubkey to
    # verify); True/False = real signature check when a pubkey is given. The gate
    # requires a genuinely-signed receipt even on the default no-pubkey run (review #2).
    parsed = _verify_receipt_safe(receipt, verify_pubkey_b64)
    # Prefer the server's inline verification when it actually ran one (pubkey supplied).
    if verify_pubkey_b64 is not None and "receipt_verified" in result:
        receipt_verified = bool(result["receipt_verified"])
    else:
        receipt_verified = parsed
    escrow_after = int(await escrow_client.balance_of(requester_address))
    ok = (output is not None) and (parsed is not False)
    if verify_pubkey_b64 is not None:
        ok = ok and bool(receipt_verified)
    return {
        "step": "pay-infer", "ok": ok,
        "output": output, "ftns_charged": result.get("ftns_charged"),
        "has_receipt": bool(receipt), "receipt_verified": receipt_verified,
        "escrow_before_wei": escrow_before, "escrow_after_wei": escrow_after,
        "note": "real output + signed receipt; on-chain settle runs over the challenge "
                "window (settlement poll loop)",
    }


async def run_all(
    *,
    chain_id: int,
    client: Any,
    escrow_client: Any,
    balance_reader: Any,
    requester_key: str,
    requester_address: str,
    prompt: str,
    deposit_amount_ftns: float = 5.0,
    faucet_post: Optional[Callable[[dict], dict]] = None,
    faucet_recipient: Optional[str] = None,
    faucet_amount_ftns: Optional[float] = None,
    verify_pubkey_b64: Optional[str] = None,
    payinfer_kwargs: Optional[dict] = None,
    confirm_attempts: int = 8,
    confirm_delay_s: float = 2.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Dict[str, Any]:
    """Run the full front-door proof (faucet? -> deposit -> pay-infer), asserting each
    step. TESTNET-ONLY (hard guard). Returns ``{network, success, steps:[...]}``;
    ``success`` is True iff every executed step's ``ok`` is True. A step that fails an
    assertion raises FrontDoorValidationError (caught by the caller for a clean report)."""
    assert_testnet_only(chain_id=chain_id)
    steps: List[Dict[str, Any]] = []

    if faucet_post is not None and faucet_recipient:
        steps.append(await run_faucet_step(
            faucet_post=faucet_post, balance_reader=balance_reader,
            recipient=faucet_recipient, amount_ftns=faucet_amount_ftns,
            confirm_attempts=confirm_attempts, confirm_delay_s=confirm_delay_s, sleep=sleep))

    steps.append(await run_deposit_step(
        client=client, requester_key=requester_key, requester_address=requester_address,
        amount_ftns=deposit_amount_ftns, escrow_client=escrow_client,
        expected_chain_id=chain_id,  # the harness already validated this == Base Sepolia
        confirm_attempts=confirm_attempts, confirm_delay_s=confirm_delay_s, sleep=sleep))

    steps.append(await run_payinfer_step(
        client=client, prompt=prompt, requester_key=requester_key,
        escrow_client=escrow_client, requester_address=requester_address,
        verify_pubkey_b64=verify_pubkey_b64, payinfer_kwargs=payinfer_kwargs))

    success = all(s.get("ok") for s in steps)
    return {"network": "base-sepolia", "chain_id": chain_id,
            "success": success, "steps": steps}
