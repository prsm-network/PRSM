"""Sprint 1306 — challenger-side fetch + verify for the §7 settlement data plane.

sp1305 added the SERVE side: a producer exposes its retained §7 InferenceReceipt by
committed-batch leaf hash at ``GET /settlement/receipt/leaf/{leaf_hash}``. This module
is the CONSUME side that makes that endpoint actionable end to end: given a peer URL +
a leaf hash (which a challenger reads off the on-chain committed batch), fetch the
retained receipt and run the §7 compute-integrity verifier
(``verify_inference_receipt_for_challenge``) on it.

Fail-LOUD on the fetch (a challenger MUST distinguish "couldn't obtain the receipt"
from "obtained it and it verified clean") — transport errors, non-200, and malformed
payloads raise ``ReceiptFetchError``. The verification result itself is returned as a
``ChallengeReport`` (``receipt_ok=False`` iff a fraud ground is PROVEN). This module
NEVER signs or submits an on-chain challenge — that stays user-gated (mirrors the
ChallengeWatcher dry-run contract).

Kept separate from the pure ``challenge_verifier`` so that module stays free of any
network/HTTP dependency; ``http_get`` is injectable for offline tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from prsm.compute.inference.models import InferenceReceipt
from prsm.settlement.challenge_verifier import (
    verify_inference_receipt_for_challenge,
)


class ReceiptFetchError(Exception):
    """Raised when the §7 receipt could not be fetched/parsed from a peer (transport
    error, non-200 status, or malformed payload). Distinct from a clean verification —
    a challenger must not treat a fetch failure as 'no fraud'."""


@dataclass(frozen=True)
class FetchedReceiptVerification:
    """Result of fetch+verify: the §7 ``ChallengeReport`` plus the served metadata."""

    leaf_hash: str
    report: Any                       # ChallengeReport (receipt_ok / findings)
    settler_node_id: Optional[str]
    retained_at: Optional[int]

    @property
    def receipt_ok(self) -> bool:
        return bool(getattr(self.report, "receipt_ok", False))


def _normalize_leaf(leaf_hash: str) -> str:
    """Lowercase, strip an optional 0x prefix. Raises ValueError on non-32-byte hex."""
    h = leaf_hash[2:] if leaf_hash.startswith("0x") else leaf_hash
    h = h.lower()
    if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
        raise ValueError("leaf_hash must be 32 bytes (64 hex chars)")
    return h


def fetch_and_verify_receipt_for_leaf(
    peer_url: str,
    leaf_hash: str,
    *,
    http_get: Optional[Callable[..., Any]] = None,
    timeout: float = 10.0,
) -> FetchedReceiptVerification:
    """Fetch the §7 receipt a peer retained for ``leaf_hash`` (sp1305 endpoint) and run
    the §7 verifier on it.

    ``peer_url`` is the producer node base URL (e.g. ``http://host:8000``). ``http_get``
    defaults to ``httpx.get`` (injected in tests). Raises ``ReceiptFetchError`` on any
    transport/HTTP/parse failure; otherwise returns a ``FetchedReceiptVerification`` whose
    ``receipt_ok`` is False iff a fraud ground is proven. Does NOT submit a challenge.
    """
    leaf = _normalize_leaf(leaf_hash)
    url = f"{peer_url.rstrip('/')}/settlement/receipt/leaf/{leaf}"

    if http_get is None:
        import httpx
        http_get = httpx.get

    try:
        resp = http_get(url, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — surface as a typed fetch error
        raise ReceiptFetchError(f"transport error fetching {url}: {exc}") from exc

    status = getattr(resp, "status_code", None)
    if status != 200:
        # include the served detail when present (503 audit-off / 404 not-retained / 422)
        detail = ""
        try:
            detail = (resp.json() or {}).get("detail", "")
        except Exception:  # noqa: BLE001
            detail = getattr(resp, "text", "")
        raise ReceiptFetchError(
            f"peer returned {status} for leaf {leaf}: {detail}")

    try:
        body: Dict[str, Any] = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise ReceiptFetchError(f"peer returned non-JSON body: {exc}") from exc

    if not isinstance(body, dict) or "inference_receipt" not in body \
            or "settler_public_key_b64" not in body:
        raise ReceiptFetchError(
            "served payload missing inference_receipt / settler_public_key_b64")

    # Integrity: the served leaf_hash must match what we asked for (a peer must not
    # answer a different leaf than the on-chain one the challenger is investigating).
    served_leaf = str(body.get("leaf_hash", "")).lower()
    if served_leaf and served_leaf != leaf:
        raise ReceiptFetchError(
            f"served leaf_hash {served_leaf} != requested {leaf}")

    try:
        receipt = InferenceReceipt.from_dict(body["inference_receipt"])
    except Exception as exc:  # noqa: BLE001
        raise ReceiptFetchError(f"malformed inference_receipt: {exc}") from exc

    stage_keys = body.get("stage_public_keys")
    if stage_keys is not None and not isinstance(stage_keys, dict):
        raise ReceiptFetchError("stage_public_keys must be an object or null")

    report = verify_inference_receipt_for_challenge(
        receipt,
        settler_public_key_b64=body["settler_public_key_b64"],
        stage_public_keys=stage_keys,
    )
    return FetchedReceiptVerification(
        leaf_hash=leaf,
        report=report,
        settler_node_id=getattr(receipt, "settler_node_id", None),
        retained_at=body.get("retained_at"),
    )
