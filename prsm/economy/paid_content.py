"""Sprint 1351 (Tier B/C paid-decrypt consumer arc, brick 3) — the pay -> unlock orchestration.

Brick 1 (prsm.storage.paid_unlock) = offline reconstruct; brick 2
(prsm.economy.web3.key_acquisition) = on-chain key acquisition. This brick is the consumer glue
that composes them into the flagship paid-content action:

    settle the release fee  ->  acquire the released key (B2)  ->  retrieve the ciphertext
                            ->  reconstruct the plaintext (B1)

The fee-settlement step and the content retrieval are INJECTABLE, deliberately:
  * The royalty verifier (``IRoyaltyPaymentVerifier``) is chosen by the publisher at deposit
    time — there is no single production verifier contract yet, so HOW a consumer pays a fee
    that ``verifyPayment`` recognizes is publisher-specific. B3 takes a ``settle_fee`` callable
    (pass ``None`` if the fee is already settled) rather than hard-coding one payment path.
  * Retrieval is the existing content layer (``retrieve_content(content_hash) -> EncryptedPayload``).

Everything is FAIL-LOUD (KeyNotReleasedError from B2 on an unpaid fee → "pay first"; PaidUnlockError
from B1 on a wrong key / tampered content) — a consumer never proceeds on a missing/garbage key.
This is what brick 4's SDK/CLI/MCP surface wraps.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


def publish_paid_content(
    *,
    plaintext: bytes,
    recipients: List[Any],
    royalty_verifier_address: str,
    release_fee_ftns_wei: int,
    key_client: Any,
    publish_ciphertext: Callable[[bytes, bytes], Any],
    retain_wrapped_key: Optional[Callable[[bytes, bytes, int], Any]] = None,
    content_hash: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Sprint 1352/1359 (arc brick 4, PUBLISHER side) — make a dataset Tier B/C paid-access.

    1. Encrypt the content with a fresh AES-256-GCM key (the ciphertext is served FREELY — a
       node holding it sees only random bytes).
    2. Wrap the content key to the designated buyer(s)' X25519 pubkeys.
    3. sp1359 (F1 redesign): deposit ONLY the sha256 COMMITMENT to the wrapped key on-chain (never
       the wrapped key itself, which would be world-readable in contract storage — the B5 F1
       critical), naming the royalty VERIFIER + release fee. Retain the wrapped key locally for the
       payment-gated serve endpoint.
    4. Publish the ciphertext to the content layer (freely fetchable by content_hash).

    Deposit BEFORE publish. Returns ``{content_hash, commitment, wrapped_key, deposit_tx,
    deposit_status, release_fee_ftns_wei, ciphertext, num_recipients}``.

    Args:
      recipients: the buyers' ``EnterpriseRecipient`` (X25519 pubkeys — the DECRYPT identity).
      royalty_verifier_address: the ``IRoyaltyPaymentVerifier`` the release gates on.
      key_client: a ``KeyDistributionClient`` (deposit_key).
      publish_ciphertext: ``(content_hash, ciphertext_bytes) -> ref`` — stores the freely-served
        ciphertext under content_hash.
      retain_wrapped_key: ``(content_hash, wrapped_key, fee_wei) -> ref`` — stores the wrapped key
        for the payment-gated serve endpoint (sp1358). If None, the caller retains it from the
        returned ``wrapped_key`` itself.
      content_hash: override the deposit key; default is sha256(ciphertext).
    """
    import hashlib

    from prsm.storage.encryption import encrypt, generate_key
    from prsm.storage.paid_unlock import (
        key_commitment,
        serialize_encrypted_content,
        wrap_content_key_for_deposit,
    )

    if not recipients:
        raise ValueError("at least one recipient (buyer X25519 pubkey) is required")
    if int(release_fee_ftns_wei) <= 0:
        raise ValueError("release_fee_ftns_wei must be > 0")

    content_key = generate_key()
    ciphertext_bytes = serialize_encrypted_content(encrypt(plaintext, content_key))
    ch = content_hash if content_hash is not None else hashlib.sha256(ciphertext_bytes).digest()
    if len(ch) != 32:
        raise ValueError(f"content_hash must be 32 bytes, got {len(ch)}")

    wrapped = wrap_content_key_for_deposit(content_key, list(recipients))
    commitment = key_commitment(wrapped)       # sp1359: ONLY the commitment goes on-chain
    tx, status = key_client.deposit_key(
        ch, commitment, royalty_verifier_address, int(release_fee_ftns_wei))
    if retain_wrapped_key is not None:
        retain_wrapped_key(ch, wrapped, int(release_fee_ftns_wei))   # for the gated serve endpoint
    publish_ciphertext(ch, ciphertext_bytes)   # serve the freely-fetchable ciphertext

    return {
        "content_hash": ch,
        "commitment": commitment,
        "wrapped_key": wrapped,
        "deposit_tx": tx,
        "deposit_status": status,
        "release_fee_ftns_wei": int(release_fee_ftns_wei),
        "ciphertext": ciphertext_bytes,
        "num_recipients": len(recipients),
    }


def build_content_access_settle_fee(
    verifier_client: Any,
    content_hash: bytes,
    fee_wei: int,
) -> Callable[[], Any]:
    """Sprint 1354 (brick 4, CONSUMER side) — turn a ContentAccessVerifierClient into the
    ``settle_fee`` callable ``pay_and_unlock`` expects.

    The returned zero-arg callable calls ``verifier_client.pay_for_access(content_hash, fee_wei)``
    (approve FTNS + payForAccess), which records the payment so ``KeyDistribution.release`` — driven
    by ``acquire_released_key`` inside ``pay_and_unlock`` — passes its verifyPayment gate. This is
    the real, live ``settle_fee`` (vs. the injectable/mock used while the verifier was undecided).

    Usage:
        settle = build_content_access_settle_fee(verifier_client, content_hash, fee_wei)
        plaintext = pay_and_unlock(..., settle_fee=settle)
    """
    def _settle() -> Any:
        # sp1356 (review F8/F11): verify-before-pay — if this payer already settled this
        # (content, fee), skip payment (no double-charge on a retry). Best-effort read; the
        # contract-side short-circuit in payForAccess is the hard guarantee.
        try:
            payer = getattr(verifier_client, "address", None)
            if payer and verifier_client.verify_payment(payer, content_hash, fee_wei):
                return None
        except Exception:  # noqa: BLE001 — a failed read must not block a legitimate payment
            pass
        return verifier_client.pay_for_access(content_hash, fee_wei)
    return _settle


def pay_and_unlock(
    *,
    content_hash: bytes,
    recipient_privkey_b64: str,
    commitment: bytes,
    fetch_wrapped_key: Callable[[bytes], Any],
    retrieve_content: Callable[[bytes], Any],
    settle_fee: Optional[Callable[[], Any]] = None,
) -> bytes:
    """Consumer pay -> unlock for a Tier B/C paid-access dataset. Returns the plaintext.

    sp1359 (F1 redesign, R3): the wrapped key is fetched OFF-chain from a payment-gated endpoint
    and VERIFIED against the on-chain commitment — it no longer lives in world-readable on-chain
    storage (the B5 F1 critical). Ordering: pay BEFORE fetching, so the endpoint's verifyPayment
    gate passes.

    Args:
      content_hash: the 32-byte content id (the deposit key / retrieval id).
      recipient_privkey_b64: the buyer's X25519 private key — the DECRYPT identity (the content
                 key was sealed to its pubkey at deposit).
      commitment: the on-chain 32-byte commitment to the wrapped key (sha256(wrapped)); the served
                 key is checked against it (defeats a lying publisher).
      fetch_wrapped_key: ``content_hash -> wrapped_key bytes`` — fetches the wrapped key from the
                 payment-gated serve endpoint (the caller bakes in the authenticated request).
      retrieve_content: maps ``content_hash`` to the served ciphertext
                 (``prsm.storage.encryption.EncryptedPayload``).
      settle_fee: optional zero-arg callable that pays the release fee (so verifyPayment becomes
                 true). Pass ``None`` when the fee is already settled.

    Raises KeyNotReleasedError / KeyCommitmentMismatchError (unpaid / wrong served key) or
    PaidUnlockError (retrieval miss / wrong X25519 key / tampered ciphertext).
    """
    from prsm.economy.web3.key_acquisition import fetch_and_verify_wrapped_key
    from prsm.storage.paid_unlock import PaidUnlockError, reconstruct_paid_content

    # 1. Pay the release fee first (so the endpoint's on-chain verifyPayment gate passes).
    if settle_fee is not None:
        settle_fee()

    # 2. Fetch the wrapped key off-chain + verify it against the on-chain commitment (B2 redesign).
    wrapped_key = fetch_and_verify_wrapped_key(fetch_wrapped_key, content_hash, commitment)

    # 3. Retrieve the freely-served ciphertext.
    content = retrieve_content(content_hash)
    if content is None:
        raise PaidUnlockError(
            f"paid + fetched the key, but the ciphertext for content "
            f"{bytes(content_hash).hex()[:12]}… is not retrievable (no provider has it?).")

    # 4. Reconstruct the plaintext (B1 — unwrap key with the buyer's X25519 key + decrypt).
    return reconstruct_paid_content(wrapped_key, recipient_privkey_b64, content)
