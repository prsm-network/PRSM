"""Sprint 1358 (F1 redesign, R2) — the payment-gated wrapped-key serve (data-plane core).

The F1 fix moves the wrapped content key OFF-chain: the publisher retains it locally and serves it
ONLY to a fetcher who proves (a) they control the paying ETH address and (b) they have an on-chain
``verifyPayment == true``. The consumer then checks it against the on-chain commitment (R1) before
decrypting. This module is the testable core (no FastAPI / no web3); the node route is a thin wrapper.

Flow the route drives:
    GET /content/paid-key/{contentHash}?nonce=...&signature=...
      → recover the signer of the paid-key challenge (EIP-191)
      → gate: verify_payment(signer, contentHash, fee) must be true
      → serve the retained wrapped key
The wrapped key is sealed to the buyer's X25519 pubkey, so even a mistaken serve to a non-buyer is
useless; the gate exists to stop the DESIGNATED buyer from fetching without paying (the F1 hole).
"""
from __future__ import annotations

from typing import Any, Callable, Optional


class PaidKeyStore:
    """sp1360/1362 (F1 redesign) — the publisher's retained wrapped-key store for the payment-gated
    serve endpoint. Maps content_hash → ``{"wrapped_key": bytes, "fee_wei": int}``.

    DURABLE (sp1362, R5 HIGH): the on-chain payment gate persists forever, so the key MUST outlive a
    node restart — otherwise a buyer who pays after a restart gets 404 and loses funds with no
    refund. When constructed with a ``path``, every ``put`` is flushed to a JSON file and the store
    rehydrates from it on startup. The wrapped key is already ciphertext (sealed to the buyer's
    X25519 pubkey), so file persistence adds negligible exposure; still, treat the file as
    operator-protected. Without a ``path`` it stays in-memory (tests / ephemeral use)."""

    def __init__(self, path: "Optional[str]" = None) -> None:
        self._d: dict = {}
        self._path = (path or "").strip() or None
        if self._path:
            self._load()

    def _load(self) -> None:
        import base64
        import json
        import os
        if not (self._path and os.path.exists(self._path)):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for ch_hex, entry in (raw or {}).items():
                self._d[bytes.fromhex(ch_hex)] = {
                    "wrapped_key": base64.b64decode(entry["wrapped_key_b64"]),
                    "fee_wei": int(entry["fee_wei"]),
                }
        except Exception:  # noqa: BLE001 — a corrupt file must not crash startup (serve returns 404)
            pass

    def _flush(self) -> None:
        import base64
        import json
        import os
        if not self._path:
            return
        out = {
            ch.hex(): {
                "wrapped_key_b64": base64.b64encode(e["wrapped_key"]).decode("ascii"),
                "fee_wei": int(e["fee_wei"]),
            }
            for ch, e in self._d.items()
        }
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(out, fh)
        os.replace(tmp, self._path)   # atomic

    def put(self, content_hash: bytes, wrapped_key: bytes, fee_wei: int) -> None:
        self._d[bytes(content_hash)] = {
            "wrapped_key": bytes(wrapped_key), "fee_wei": int(fee_wei)}
        self._flush()

    def get(self, content_hash: bytes):
        return self._d.get(bytes(content_hash))

    def __len__(self) -> int:
        return len(self._d)


def resolve_paid_key_store_path(environ: Any) -> str:
    """sp1438 (audit B5 #1) — the DURABLE store path the node wires PaidKeyStore with.

    Returns the operator override ``PRSM_PAID_KEY_STORE_FILE`` if set, else a durable per-user
    default (``~/.prsm/paid_key_store.json``). The on-chain payment gate persists forever, so the
    retained wrapped key MUST be durable: an unset env var must NOT silently degrade the store to
    in-memory, or a buyer who pays after a node restart gets a 404 with no refund (the live
    ContentAccessVerifier has no refund path). Mirrors the settlement verified-batch store's durable
    default in node.py — NEVER returns None/empty on the production path."""
    from pathlib import Path
    explicit = (environ.get("PRSM_PAID_KEY_STORE_FILE") or "").strip()
    return explicit or str(Path.home() / ".prsm" / "paid_key_store.json")


def build_paid_key_verify_payment(verifier_client: Any) -> Callable[[str, bytes, int], bool]:
    """sp1360 — adapt a ContentAccessVerifierClient into the ``verify_payment(payer, content_hash,
    fee_wei) -> bool`` callable ``serve_paid_key`` gates on (reads verifyPayment on-chain)."""
    def _vp(payer: str, content_hash: bytes, fee_wei: int) -> bool:
        return bool(verifier_client.verify_payment(payer, content_hash, fee_wei))
    return _vp


class PaidKeyServeError(Exception):
    """A paid-key serve was refused. ``status`` maps to the HTTP code the route returns
    (404 no such key, 401 bad signature, 402 unpaid, 503 can't verify on-chain)."""

    def __init__(self, status: int, message: str) -> None:
        self.status = int(status)
        self.message = str(message)
        super().__init__(self.message)


def paid_key_challenge(content_hash: bytes, nonce: str) -> str:
    """The message a fetcher signs (EIP-191 personal_sign) to prove control of the paying address.
    Bound to the content so a signature for one dataset can't be reused for another."""
    return f"PRSM-paid-key:{bytes(content_hash).hex()}:{nonce}"


def recover_paid_key_fetcher(content_hash: bytes, nonce: str, signature: str) -> str:
    """Recover the ETH address that signed the paid-key challenge. Raises on a malformed signature."""
    from eth_account import Account
    from eth_account.messages import encode_defunct
    msg = encode_defunct(text=paid_key_challenge(content_hash, nonce))
    return Account.recover_message(msg, signature=signature)


def serve_paid_key(
    content_hash: bytes,
    nonce: str,
    signature: str,
    *,
    key_store: Any,
    verify_payment: Callable[[str, bytes, int], bool],
) -> bytes:
    """R2 core — authenticate the fetcher, gate on on-chain payment, return the retained wrapped key.

    ``key_store.get(content_hash)`` → ``{"wrapped_key": bytes, "fee_wei": int}`` or ``None`` (the
    publisher retains this at publish time — the fee is the AUTHORITATIVE deposit fee, not
    fetcher-supplied). ``verify_payment(payer, content_hash, fee_wei) -> bool`` reads
    ContentAccessVerifier.verifyPayment on-chain. Raises PaidKeyServeError on every refusal path.
    """
    ch = bytes(content_hash)
    entry = key_store.get(ch)
    if entry is None:
        raise PaidKeyServeError(404, "no paid key retained for this content on this node")

    try:
        signer = recover_paid_key_fetcher(ch, nonce, signature)
    except Exception as exc:  # noqa: BLE001
        raise PaidKeyServeError(401, f"invalid fetcher signature: {exc}") from exc

    fee_wei = int(entry["fee_wei"])
    try:
        ok = bool(verify_payment(signer, ch, fee_wei))
    except Exception as exc:  # noqa: BLE001 — an RPC failure must not leak the key
        raise PaidKeyServeError(503, f"could not verify payment on-chain: {exc}") from exc
    if not ok:
        raise PaidKeyServeError(
            402, f"{signer} has not paid the release fee for this content — pay via "
                 f"ContentAccessVerifier.payForAccess first")

    return bytes(entry["wrapped_key"])
