"""Sprint 1366 (Tier B/C paid-decrypt PUBLISHER surface, brick 1) — node-side paid publish.

The consumer side of Tier B/C shipped end to end (SDK/CLI `content unlock`) and the verifier is
deployed on mainnet, but `publish_paid_content` had NO surface — nobody could actually create paid
content, so the deployed contract was only half-usable. This is the node-side orchestration that
makes it reachable: it composes the paid layer onto the EXISTING upload path (which already serves
the ciphertext AND registers the creator on-chain in the ProvenanceRegistry, so the CAV's
getCreatorAndRate resolves the publisher — satisfying the sp1365 anti-squat invariant creator ==
key-depositor).

What it adds on top of a normal upload:
  * deposit the sha256 COMMITMENT on-chain (KeyDistribution, naming the ContentAccessVerifier as the
    royalty verifier) — never the wrapped key (F1),
  * retain the wrapped key in the node's PaidKeyStore for the payment-gated serve endpoint (sp1358).

A FastAPI route + `prsm content publish --paid` CLI wrap this in follow-on bricks.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


def run_paid_publish(
    *,
    plaintext: bytes,
    recipients: List[Any],
    fee_wei: int,
    verifier_address: str,
    key_client: Any,
    serve_ciphertext: Callable[[bytes, bytes], Any],
    paid_key_store: Any,
    content_hash: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Publish a Tier B/C paid dataset from the operator's node. Returns the publish_paid_content
    result (content_hash, commitment, wrapped_key, deposit_tx, …).

    Args:
      plaintext: the content to gate.
      recipients: the buyers' ``EnterpriseRecipient`` (X25519 pubkeys — the decrypt identity).
      fee_wei: the release fee (must match what the publisher advertises + what a consumer pays).
      verifier_address: the deployed ContentAccessVerifier (from network config).
      key_client: a signing ``KeyDistributionClient`` (the publisher deposits the commitment).
      serve_ciphertext: ``(content_hash, ciphertext_bytes) -> ref`` — stores the freely-served
        ciphertext under ``content_hash`` in the content layer (the SAME path that registers the
        creator on-chain, so the fee payee resolves to the publisher).
      paid_key_store: the node's ``PaidKeyStore`` (the retained wrapped key is served, per request,
        only to a paid + authenticated fetcher).
      content_hash: override; default sha256(ciphertext).
    """
    from prsm.economy.paid_content import publish_paid_content

    def _retain(ch: bytes, wrapped: bytes, fee: int) -> None:
        paid_key_store.put(ch, wrapped, fee)

    return publish_paid_content(
        plaintext=plaintext,
        recipients=list(recipients),
        royalty_verifier_address=verifier_address,
        release_fee_ftns_wei=int(fee_wei),
        key_client=key_client,
        publish_ciphertext=serve_ciphertext,
        retain_wrapped_key=_retain,
        content_hash=content_hash,
    )


def run_paid_publish_from_path(
    *,
    plaintext_path: Any,
    ciphertext_path: Any,
    recipients: List[Any],
    fee_wei: int,
    verifier_address: str,
    key_client: Any,
    serve_ciphertext_from_path: Callable[[bytes, Any], Any],
    paid_key_store: Any,
) -> Dict[str, Any]:
    """sp1458 — the STREAMING large-file equivalent of ``run_paid_publish``. Publishes a Tier B/C
    paid dataset too large to hold in memory, with a bounded footprint end to end.

    Steps (preserving the sp1361 F5 + sp1365 anti-squat + sp1438 retain-before-deposit invariants):
      1. ``build_paid_content_from_path`` stream-encrypts ``plaintext_path`` → ``ciphertext_path``
         (AES-256-GCM binary streaming format) and wraps the fresh content key to the buyer(s);
         content_hash = sha256 of the ciphertext FILE.
      2. ``serve_ciphertext_from_path(content_hash, ciphertext_path)`` serves the ciphertext FILE
         under content_hash (the SAME large-content upload path — POST /content/upload-stream — that
         registers the creator on-chain, so the CAV's fee payee resolves to the publisher). Serve is
         BEFORE deposit so a live payment gate never predates a serveable ciphertext.
      3. ``deposit_commitment_and_retain`` retains the wrapped key in the PaidKeyStore BEFORE
         depositing the commitment on-chain (naming the CAV) — the retain-first order that stops a
         buyer paying into a gate with no retained key.

    Returns ``{content_hash, commitment, wrapped_key, ciphertext_path, deposit_tx, deposit_status,
    serve_ref}``. The consumer fetches the ciphertext via /content/retrieve-stream and decrypts with
    ``reconstruct_paid_content_from_file`` after the payment-gated key release."""
    from prsm.economy.paid_content import (
        build_paid_content_from_path,
        deposit_commitment_and_retain,
    )
    built = build_paid_content_from_path(plaintext_path, ciphertext_path, list(recipients))
    content_hash = built["content_hash"]
    serve_ref = serve_ciphertext_from_path(content_hash, ciphertext_path)
    tx, status = deposit_commitment_and_retain(
        key_client=key_client,
        paid_key_store=paid_key_store,
        content_hash=content_hash,
        commitment=built["commitment"],
        wrapped_key=built["wrapped_key"],
        fee_wei=int(fee_wei),
        verifier_address=verifier_address,
    )
    return {
        "content_hash": content_hash,
        "commitment": built["commitment"],
        "wrapped_key": built["wrapped_key"],
        "ciphertext_path": built["ciphertext_path"],
        "deposit_tx": tx,
        "deposit_status": status,
        "serve_ref": serve_ref,
    }
