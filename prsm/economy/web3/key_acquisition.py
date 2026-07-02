"""Sprint 1350 (Tier B/C paid-decrypt consumer arc, brick 2) — consumer key acquisition.

Brick 1 (prsm.storage.paid_unlock) gave the offline reconstruct (unwrap the released key +
decrypt the content). This brick gets the released key ON-CHAIN: a consumer who has PAID the
release fee triggers ``KeyDistribution.release(content_hash, recipient)`` (permissionless — the
contract binds payment verification to ``recipient``) and reads the resulting ``KeyReleased``
event to obtain the deposited ``encrypted_key``. Feeding that into brick 1 yields the plaintext.

PREREQUISITE: the recipient must have PAID the release fee to the deposit's royalty verifier
(``IRoyaltyPaymentVerifier.verifyPayment(recipient, content_hash, feeWei)`` → true). That
fee-settlement is the caller's / B3's responsibility — until it's true, ``release`` reverts
``PaymentNotVerified`` and this surfaces a clear KeyNotReleasedError telling the consumer to pay.

IDEMPOTENT: if the key was already released (a prior KeyReleased event exists for this
(content_hash, recipient)), it's returned WITHOUT re-triggering release (no wasted gas, no
double-charge). Everything is fail-loud — a consumer never proceeds on a missing/ambiguous key.
"""
from __future__ import annotations

from typing import Any, Optional


class KeyNotReleasedError(Exception):
    """The encrypted content-key could not be acquired — not yet released (unpaid fee), never
    deposited, or the release tx produced no matching event. Fail-loud: the consumer must not
    attempt to decrypt without the real key."""


class KeyCommitmentMismatchError(Exception):
    """sp1357 (F1 redesign) — the wrapped key fetched from the payment-gated endpoint does NOT
    match the on-chain commitment: the server served the WRONG key (malicious or corrupted).
    Fail-loud — never decrypt with an unverified key."""


def fetch_and_verify_wrapped_key(fetch_fn: Any, content_hash: bytes, commitment: bytes) -> bytes:
    """sp1357 (F1 redesign, R1) — the consumer's key acquisition under the binding-gate design.

    Replaces reading the world-readable KeyReleased event (the F1 critical): fetch the wrapped key
    OFF-chain via ``fetch_fn(content_hash) -> bytes`` (the payment-gated endpoint client, which
    proves payment to the server) and VERIFY it against the on-chain ``commitment`` before returning
    it. So neither an unpaid party (the endpoint gates on verifyPayment) nor a lying publisher
    (the commitment check) can hand the consumer a usable-but-wrong key.

    Raises KeyNotReleasedError if the fetch yields nothing; KeyCommitmentMismatchError if the served
    bytes don't match the commitment. Pure/offline — ``fetch_fn`` and the commitment are injected.
    """
    from prsm.storage.paid_unlock import verify_key_commitment

    if len(bytes(commitment)) != 32:
        raise KeyCommitmentMismatchError(
            f"on-chain commitment must be 32 bytes, got {len(bytes(commitment))}")
    wrapped = fetch_fn(content_hash)
    if not wrapped:
        raise KeyNotReleasedError(
            f"payment-gated key endpoint returned no key for content "
            f"{bytes(content_hash).hex()[:12]}… (unpaid, not served, or endpoint down)")
    wrapped = bytes(wrapped)
    if not verify_key_commitment(wrapped, commitment):
        raise KeyCommitmentMismatchError(
            f"served wrapped key does not match the on-chain commitment for content "
            f"{bytes(content_hash).hex()[:12]}… — the server served the WRONG key; refusing to use it")
    return wrapped


def _matching_key(events: Any, content_hash: bytes, recipient: str) -> Optional[bytes]:
    ch = bytes(content_hash)
    rl = str(recipient).lower()
    for ev in events or []:
        try:
            if bytes(ev.content_hash) == ch and str(ev.recipient).lower() == rl:
                return bytes(ev.encrypted_key)
        except Exception:  # noqa: BLE001 — a malformed event never derails the scan
            continue
    return None


def acquire_released_key(
    client: Any,
    content_hash: bytes,
    recipient: str,
    *,
    from_block: int = 0,
    to_block: Optional[int] = None,
    trigger_release: bool = True,
) -> bytes:
    """Acquire the released ``encrypted_key`` for ``(content_hash, recipient)`` from the
    KeyDistribution contract via ``client`` (a KeyDistributionClient or compatible).

    1. Read existing ``KeyReleased`` events — if already released, return it (IDEMPOTENT).
    2. Else, if ``trigger_release``, call ``release(content_hash, recipient)`` (reverts if the
       recipient hasn't paid → surfaced as KeyNotReleasedError), then re-read.
    Raises ``KeyNotReleasedError`` on unpaid / not-deposited / no-event-after-release, or when
    ``trigger_release=False`` and it isn't released yet.
    """
    from prsm.economy.web3.key_distribution import (
        KeyNotFoundError,
        PaymentNotVerifiedError,
    )
    from prsm.economy.web3.provenance_registry import OnChainRevertedError

    top = to_block if to_block is not None else client.latest_block()
    key = _matching_key(client.get_key_released_events(from_block, top), content_hash, recipient)
    if key is not None:
        return key  # already released — no re-trigger (idempotent)

    if not trigger_release:
        raise KeyNotReleasedError(
            f"key for content {bytes(content_hash).hex()[:12]}… not yet released to {recipient} "
            f"(trigger_release=False)")

    try:
        client.release(content_hash, recipient)
    except PaymentNotVerifiedError as exc:
        raise KeyNotReleasedError(
            "release reverted (PaymentNotVerified) — the recipient must PAY the release fee to "
            "the content's royalty verifier before the key can be released.") from exc
    except KeyNotFoundError as exc:
        raise KeyNotReleasedError(
            "no key deposited for this content_hash — the publisher never called deposit_key "
            "(this content is not Tier B/C paid-access).") from exc
    except OnChainRevertedError as exc:
        raise KeyNotReleasedError(
            f"release reverted — most likely the release fee is unpaid, or the key was "
            f"deauthorized: {exc}") from exc

    top = client.latest_block()
    key = _matching_key(client.get_key_released_events(from_block, top), content_hash, recipient)
    if key is None:
        raise KeyNotReleasedError(
            "release tx was sent but no matching KeyReleased event was found (RPC lag or a "
            "recipient/content mismatch) — retry the read shortly.")
    return key
