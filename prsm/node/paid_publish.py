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
