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

    # sp1365 (review F2): don't publish into a content_hash another party already deposited.
    assert_content_hash_unclaimed(key_client, ch, getattr(key_client, "address", None))

    wrapped = wrap_content_key_for_deposit(content_key, list(recipients))
    commitment = key_commitment(wrapped)       # sp1359: ONLY the commitment goes on-chain

    # sp1361 (review F5): retain the key + serve the ciphertext BEFORE the on-chain deposit, so a
    # LIVE gate always implies the content is deliverable. If the deposit reverts, the retained key
    # + served ciphertext are harmless (no gate exists → no one can pay). The dangerous order is the
    # reverse (gate live but content undeliverable → paid buyers stranded).
    if retain_wrapped_key is not None:
        retain_wrapped_key(ch, wrapped, int(release_fee_ftns_wei))   # for the gated serve endpoint
    publish_ciphertext(ch, ciphertext_bytes)   # serve the freely-fetchable ciphertext
    tx, status = key_client.deposit_key(
        ch, commitment, royalty_verifier_address, int(release_fee_ftns_wei))

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


def build_paid_content(plaintext: bytes, recipients: List[Any]) -> Dict[str, Any]:
    """sp1367 — the pure crypto step of paid publish (no chain, no serve): encrypt with a fresh
    AES-256-GCM key + wrap that key to the buyer(s)' X25519 pubkeys. Returns
    ``{ciphertext, wrapped_key, commitment}`` (commitment = sha256(wrapped_key)).

    Split out from publish_paid_content so an ASYNC caller (the node route) can upload the
    ciphertext to the content layer FIRST — getting the authoritative content_hash =
    sha256(ciphertext), which is also where the upload registers the creator on-chain — and THEN
    deposit the commitment under that same hash (via ``deposit_commitment_and_retain``)."""
    from prsm.storage.encryption import encrypt, generate_key
    from prsm.storage.paid_unlock import (
        key_commitment,
        serialize_encrypted_content,
        wrap_content_key_for_deposit,
    )
    if not recipients:
        raise ValueError("at least one recipient (buyer X25519 pubkey) is required")
    content_key = generate_key()
    ciphertext = serialize_encrypted_content(encrypt(plaintext, content_key))
    wrapped = wrap_content_key_for_deposit(content_key, list(recipients))
    return {"ciphertext": ciphertext, "wrapped_key": wrapped, "commitment": key_commitment(wrapped)}


def deposit_commitment_and_retain(
    *,
    key_client: Any,
    paid_key_store: Any,
    content_hash: bytes,
    commitment: bytes,
    wrapped_key: bytes,
    fee_wei: int,
    verifier_address: str,
) -> Any:
    """sp1367 — the on-chain + retain step of paid publish, given the content_hash the content
    layer assigned (sha256(ciphertext)). Anti-squat guard (F2) → deposit the commitment naming the
    CAV → retain the wrapped key in the node's store. Returns (tx_hash, status)."""
    if int(fee_wei) <= 0:
        raise ValueError("fee_wei must be > 0")
    assert_content_hash_unclaimed(key_client, content_hash, getattr(key_client, "address", None))
    tx, status = key_client.deposit_key(
        content_hash, commitment, verifier_address, int(fee_wei))
    paid_key_store.put(content_hash, wrapped_key, int(fee_wei))
    return tx, status


class SquatMismatchError(Exception):
    """sp1365 (review F9/F2) — the party that would EARN the access fee (the ProvenanceRegistry
    creator the CAV credits) is not the party that CONTROLS the content (the KeyDistribution
    publisher who deposited the key). Since contentHash is a public, squattable identifier and
    neither deployed contract binds authority to the publisher, this mismatch is the signature of
    squatting (F9 fee-redirect / F2 deposit squat). Fail-loud: never fund a squatter, never publish
    into a squatted slot."""


def assert_publisher_controls_payee(
    key_client: Any, get_creator: Any, content_hash: bytes,
) -> None:
    """sp1365 — CONSUMER anti-squat guard. Verify the fee payee == the key depositor before paying.

    ``get_creator(content_hash) -> address`` reads the CAV's registry (getCreatorAndRate.creator —
    whom payForAccess credits). ``key_client.get_deposit(content_hash).publisher`` is who deposited
    the key (controls the content). If they diverge, a squatter registered the creator OR squatted
    the deposit without controlling both under one identity → refuse to pay. Best-effort skip only
    when the reads aren't available (no client)."""
    if key_client is None or get_creator is None or not hasattr(key_client, "get_deposit"):
        return
    ch12 = bytes(content_hash).hex()[:12]
    try:
        dep = key_client.get_deposit(content_hash)
        creator = get_creator(content_hash)
    except Exception:  # noqa: BLE001 — an unreadable check must not block a legitimate unlock
        return
    if dep is None:
        return  # assert_fee_matches_deposit already fails-closed on a missing deposit
    if not creator or str(creator).lower() in ("0x" + "0" * 40, ""):
        raise SquatMismatchError(
            f"content {ch12}… has no registered creator — the fee payee is unknown; refusing to pay")
    if str(creator).lower() != str(dep.publisher).lower():
        raise SquatMismatchError(
            f"fee payee (registry creator {creator}) != key depositor (publisher {dep.publisher}) "
            f"for content {ch12}… — possible squatting; refusing to pay a fee that wouldn't reach "
            f"the content owner")


def assert_content_hash_unclaimed(
    key_client: Any, content_hash: bytes, publisher: str,
) -> None:
    """sp1365 — PUBLISHER anti-squat guard. Before depositing, ensure this content_hash isn't
    already deposited by ANOTHER party (a front-run/squat). If it is, fail so the publisher
    re-publishes for a fresh content_hash (a new random IV yields a new sha256(ciphertext))."""
    if key_client is None or not hasattr(key_client, "get_deposit") or not publisher:
        return
    try:
        dep = key_client.get_deposit(content_hash)
    except Exception:  # noqa: BLE001
        return
    if dep is not None and str(dep.publisher).lower() != str(publisher).lower():
        raise SquatMismatchError(
            f"content_hash {bytes(content_hash).hex()[:12]}… is already deposited by "
            f"{dep.publisher} (not you) — squatted or a stale re-run; re-publish to get a fresh "
            f"content_hash (a new IV changes sha256(ciphertext))")


class FeeMismatchError(Exception):
    """sp1361 (review F3/F4/F12) — the fee about to be paid does not match the on-chain deposit
    fee. Paying it would pull + credit FTNS that can never satisfy the gate (no unlock, no refund).
    Fail-loud BEFORE paying."""


def assert_fee_matches_deposit(key_client: Any, content_hash: bytes, fee_wei: int) -> None:
    """sp1361/1362 — refuse to pay a fee that doesn't match the on-chain deposit. FAIL-CLOSED
    (R5 low): the ContentAccessVerifier pulls FTNS for ANY fee, but the key-serve gate checks the
    AUTHORITATIVE deposit fee — so a wrong/unconfirmable fee is pulled with no unlock and no refund.
    Therefore, when a key_client IS provided, an unreadable deposit (RPC error) or a missing deposit
    now RAISES rather than silently paying. Only skips when NO client was provided (caller opted
    out — offline/tests)."""
    if key_client is None or not hasattr(key_client, "get_deposit"):
        return
    ch12 = bytes(content_hash).hex()[:12]
    try:
        deposit = key_client.get_deposit(content_hash)
    except Exception as exc:  # noqa: BLE001 — can't confirm ⇒ don't pay (fail-closed)
        raise FeeMismatchError(
            f"could not confirm the on-chain deposit fee for content {ch12}… ({exc}) — refusing "
            f"to pay a fee that may not unlock") from exc
    if deposit is None:
        raise FeeMismatchError(
            f"no on-chain deposit for content {ch12}… — nothing to unlock; refusing to pay")
    on_chain = int(getattr(deposit, "release_fee_ftns_wei", -1))
    if on_chain != int(fee_wei):
        raise FeeMismatchError(
            f"fee {fee_wei} != on-chain deposit fee {on_chain} for content {ch12}… — refusing to "
            f"pay a fee that won't unlock")


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
    fetch_wrapped_key: Callable[[bytes], Any],
    retrieve_content: Callable[[bytes], Any],
    commitment: Optional[bytes] = None,
    fetch_commitment: Optional[Callable[[], Any]] = None,
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
      fetch_wrapped_key: ``content_hash -> wrapped_key bytes`` — fetches the wrapped key from the
                 payment-gated serve endpoint (the caller bakes in the authenticated request).
      retrieve_content: maps ``content_hash`` to the served ciphertext
                 (``prsm.storage.encryption.EncryptedPayload``).
      commitment: the 32-byte commitment to verify the served key against. Provide it ONLY from an
                 authoritative source; otherwise pass ``fetch_commitment``.
      fetch_commitment: sp1363 (R5 MEDIUM) — a zero-arg callable that reads the commitment from the
                 AUTHORITATIVE on-chain source (the gated KeyReleased event), called AFTER settle so
                 the release passes verifyPayment. Used when ``commitment`` is not supplied, so the
                 consumer doesn't trust a publisher-/MITM-supplied value.
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

    # 2. Resolve the commitment from an AUTHORITATIVE source (on-chain) if not supplied directly.
    resolved_commitment = commitment
    if resolved_commitment is None:
        if fetch_commitment is None:
            raise PaidUnlockError("commitment or fetch_commitment is required")
        resolved_commitment = fetch_commitment()

    # 3. Fetch the wrapped key off-chain + verify it against the commitment (B2 redesign).
    wrapped_key = fetch_and_verify_wrapped_key(
        fetch_wrapped_key, content_hash, resolved_commitment)

    # 3. Retrieve the freely-served ciphertext.
    content = retrieve_content(content_hash)
    if content is None:
        raise PaidUnlockError(
            f"paid + fetched the key, but the ciphertext for content "
            f"{bytes(content_hash).hex()[:12]}… is not retrievable (no provider has it?).")

    # 4. Reconstruct the plaintext (B1 — unwrap key with the buyer's X25519 key + decrypt).
    return reconstruct_paid_content(wrapped_key, recipient_privkey_b64, content)
