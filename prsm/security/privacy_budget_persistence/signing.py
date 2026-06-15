"""
Privacy-budget entry signing and verification.

Phase 3.x.4 Task 2 — wires :func:`PrivacyBudgetEntry.signing_payload` to
the existing :class:`prsm.node.identity.NodeIdentity` Ed25519 keypair so
every journal event carries a node-verifiable signature.

Verification path: a regulator or third-party auditor holding only the
node's public key can call :func:`verify_entry` against a serialized
entry and confirm:

1. The signature was produced by the holder of the matching private key.
2. None of the entry fields have been altered since signing.
3. (Combined with the store's chain-walk on read) every prior entry's
   ``prev_entry_hash`` is consistent — historical tampering invalidates
   every subsequent signature, not just the touched entry.

Same crypto trust-anchor as Phase 3.x.1 ``InferenceReceipt`` and
Phase 3.x.2 ``ModelManifest``: ``NodeIdentity.sign / verify``. One
keypair per node, used for identity, receipts, manifests, and now
privacy-budget entries.
"""

from __future__ import annotations

import base64
import dataclasses
from typing import Optional

from prsm.node.identity import NodeIdentity, verify_signature
from prsm.security.privacy_budget_persistence.models import PrivacyBudgetEntry


def sign_entry(
    entry: PrivacyBudgetEntry, identity: NodeIdentity
) -> PrivacyBudgetEntry:
    """Return a copy of ``entry`` with ``signature`` populated.

    The original entry is unchanged (frozen dataclass). The returned
    entry has:
    - ``node_id`` set to ``identity.node_id`` (overwrites any caller-
      supplied placeholder, since the signer's node_id is part of the
      signing payload — verifiers can't rely on a manifest field that
      wasn't authenticated by the same key)
    - ``signature`` set to the Ed25519 signature of ``signing_payload()``

    The signing payload excludes ``signature`` itself (Task 1
    ``test_payload_excludes_signature``), so the two-step replace is
    safe: stamp ``node_id`` first, compute payload bytes, sign, then
    stamp ``signature``.
    """
    # Step 1: stamp node_id BEFORE computing the signing payload so
    # the payload reflects the actual signer's identity.
    intermediate = dataclasses.replace(entry, node_id=identity.node_id)

    # Step 2: sign the canonical bytes
    payload = intermediate.signing_payload()
    signature_b64 = identity.sign(payload)
    signature_bytes = base64.b64decode(signature_b64)

    # Step 3: produce the signed entry
    return dataclasses.replace(intermediate, signature=signature_bytes)


def verify_entry(
    entry: PrivacyBudgetEntry,
    public_key_b64: Optional[str] = None,
    identity: Optional[NodeIdentity] = None,
) -> bool:
    """Verify ``entry.signature`` against ``signing_payload()``.

    Provide one of:
    - ``public_key_b64`` — base64-encoded Ed25519 public key matching
      the signer's published key (offline-verifier path; auditor only
      needs the pubkey, not a live ``NodeIdentity``)
    - ``identity`` — a :class:`NodeIdentity` instance whose public key
      matches the signer

    Returns False (does NOT raise) if:
    - Signature is empty (entry was never signed)
    - Neither verification credential supplied
    - Signature doesn't match the canonical signing payload
    - Public key doesn't correspond to the signer
    - Any underlying crypto step raises

    Tampering with any field in ``signing_payload()`` invalidates the
    signature — verified exhaustively by the Task 2 tampering tests.
    Cross-artifact replay (e.g., a ``ModelManifest`` signature against
    a budget-entry payload) also fails — the domain separator differs.
    """
    if not entry.signature:
        return False
    if public_key_b64 is None and identity is None:
        return False

    # Sprint 1123 (Domain-08 review LOW-10) — BIND the verifying key to the entry's
    # claimed node_id (node_id == hex(sha256(pubkey))[:32]). The signature already covers
    # node_id, but without this check a caller that fetched the pubkey from the SAME
    # untrusted source as the entry could verify an attacker's (pubkey, signature,
    # node_id) triple that is internally consistent yet signed by the attacker's key.
    # Asserting the key derives the claimed node_id makes that impossible — fail-closed.
    from prsm.node.identity import node_id_for_public_key
    try:
        if identity is not None:
            verifier_node_id = identity.node_id
        else:
            verifier_node_id = node_id_for_public_key(
                base64.b64decode(public_key_b64),  # type: ignore[arg-type]
            )
    except Exception:  # noqa: BLE001 — malformed key → can't bind → fail closed
        return False
    if verifier_node_id != entry.node_id:
        return False

    payload = entry.signing_payload()
    signature_b64 = base64.b64encode(entry.signature).decode()

    try:
        if identity is not None:
            return identity.verify(payload, signature_b64)
        # public_key_b64 is not None per the early-return above; the
        # explicit None-check survives `python -O` (a bare assert wouldn't).
        if public_key_b64 is None:
            return False
        return verify_signature(public_key_b64, payload, signature_b64)
    except Exception:
        # NodeIdentity.verify / verify_signature should already return
        # False on malformed input, but this defensive catch mirrors
        # the receipt + manifest verifiers' "fail closed, never raise"
        # contract.
        return False


def is_signed(entry: PrivacyBudgetEntry) -> bool:
    """Return True iff ``entry`` carries a non-empty signature + node_id.

    NOT a cryptographic check — only confirms both fields are populated.
    Use :func:`verify_entry` to validate the signature; an entry whose
    signature was tampered post-signing passes ``is_signed`` but fails
    ``verify_entry``.
    """
    return bool(entry.signature) and bool(entry.node_id)
