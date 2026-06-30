"""Sprint 1155 — PerStagePaymentAuthorization EIP-712 primitive + canonical
payee-set hash + fail-closed verifier (BRICK 3 of the on-chain per-stage payee
arc, project_onchain_per_stage_payee_arc).

THE GAP. The sp1054 ``PaymentAuthorization`` (``payment_authorization.py``) pins a
SINGLE provider address — it authorizes paying ONE node. On a multi-stage
cross-host inference, brick 1 (``per_stage_settlement_split.py``) fans the settled
value into one challenge-defensible share-batch PER topology node, and brick 4 has
each node commit its OWN share-batch. So the requester can no longer authorize one
provider — they must authorize the SET of (payee, share) pairs in ONE signature,
with a CUMULATIVE ceiling. This module is that authorization primitive + its
fail-closed verifier. It is a NEW SIBLING of sp1054; ``payment_authorization.py``
is left byte-for-byte unchanged.

PARITY WITH sp1054 (DRY — imported, not redefined):
  - DOMAIN: reuse ``DEFAULT_CHAIN_ID`` (8453, Base), ``DOMAIN_NAME``
    ("PRSM-Payment"), ``DOMAIN_VERSION`` ("v1"). The domain is network-locked:
    the chainId in the domain separator binds every signature to one chain, so a
    signature cannot be replayed cross-chain.
  - HELPERS: reuse ``_to_bytes32`` (bytes32 normalization) + ``InvalidSignatureFormat``.
  - PATTERN: ``_build_typed_data`` → ``encode_typed_data`` → keccak(0x1901 ||
    domainSep || hashStruct) digest; ``sign_*`` (test helper / reference impl);
    ``verify_*`` recovers the signer. A malformed input raises in the builder; a
    malformed signature SHAPE raises ``InvalidSignatureFormat``.

NEW EIP-712 type (NEW primaryType on the SAME domain):
  PerStagePaymentAuthorization {
    address requester            // EscrowPool depositor; funding source
    bytes32 payee_set_hash       // canonical, order-independent commitment to the
                                 // SET of (payee, max_share_wei) pairs
    uint256 total_max_spend_wei  // cumulative ceiling across ALL payees
    bytes32 job_nonce            // unique per job (anti-replay)
    uint256 expiry_unix          // auth invalid after this (anti-replay/staleness)
    bytes32 request_hash         // binds the canonical inference request
  }

MONEY-SAFETY (fail-closed verifier). ``verify_per_stage_authorization`` returns a
verdict (never a false authorize):
  (a) recovered signer == payload["requester"] (case-insensitive),
  (b) compute_payee_set_hash(payees) == payload["payee_set_hash"] (supplied set ==
      signed commitment),
  (c) (payee, share_wei) is a MEMBER of payees (exact case-insensitive address +
      EXACT share),
  (d) sum(shares) <= total_max_spend_wei (NO over-authorization — the cumulative
      cap holds),
  (e) not expired (now < expiry_unix),
  (f) request_hash present + bound (non-zero).
ANY failure → ``authorized=False`` + a reason; NEVER a false authorize, and NEVER
raises on a logical reject — only a malformed-signature SHAPE raises (matching
sp1054). The per-node commit (brick 4) gates on ``authorized == True``.

PURE / offline: no network, no escrow, no chain. The splitter never signs; this
module signs only as a test-helper / reference (production requesters sign with
MetaMask / hardware wallets and submit the signature directly).
"""
from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

from eth_abi import encode as abi_encode
from eth_account import Account
from eth_account.messages import encode_typed_data
from eth_utils import is_address, keccak, to_checksum_address

# Reuse sp1054's network-locked domain + helpers verbatim (DRY — the domain is
# shared across both signing surfaces; redefining it would risk drift + a
# cross-surface signature mismatch). payment_authorization.py is UNCHANGED.
from prsm.settlement.payment_authorization import (
    DEFAULT_CHAIN_ID,
    DOMAIN_NAME,
    DOMAIN_VERSION,
    InvalidSignatureFormat,
    _to_bytes32,
)

__all__ = [
    "DEFAULT_CHAIN_ID",
    "DOMAIN_NAME",
    "DOMAIN_VERSION",
    "InvalidSignatureFormat",
    "PerStageAuthorizationVerdict",
    "compute_payee_set_hash",
    "encode_per_stage_typed_data_hash",
    "recover_per_stage_signer",
    "sign_per_stage_authorization",
    "verify_per_stage_authorization",
]

# uint256 ceiling — an EIP-712 ``uint256`` share / total bound.
_MAX_UINT256 = 2 ** 256 - 1
_ZERO_BYTES32 = b"\x00" * 32

# Fields that carry the per-stage authorization.
_REQUIRED = (
    "requester", "payee_set_hash", "total_max_spend_wei",
    "job_nonce", "expiry_unix", "request_hash",
)


# --------------------------------------------------------------------------- #
# 1. Canonical, order-independent payee-set hash                              #
# --------------------------------------------------------------------------- #

def _normalize_share(share: Any) -> int:
    """Validate a share is a plain non-negative int that fits a uint256.

    ``bool`` is rejected explicitly — ``int(True)`` would silently coerce a
    True/False into a wei amount (mirrors the sp1154 splitter's bool guard).
    """
    if isinstance(share, bool) or not isinstance(share, int):
        raise ValueError(
            f"share must be a non-negative int in wei; got "
            f"{type(share).__name__}: {share!r}"
        )
    if share < 0:
        raise ValueError(f"share must be non-negative; got {share}")
    if share > _MAX_UINT256:
        raise ValueError(f"share out of uint256 range: {share}")
    return share


def compute_payee_set_hash(
    payees: Sequence[Tuple[str, int]],
) -> bytes:
    """Canonical, deterministic, ORDER-INDEPENDENT keccak commitment to the SET of
    ``(payee_address, max_share_wei)`` pairs.

    Each address is checksum-normalized (so "0xAbC..", "0xabc.." and the
    checksum form all collapse to one canonical form), each share is validated as
    a uint256, and the pairs are SORTED by ``(checksum_address, share)``. The hash
    is ``keccak(abi_encode([address[], uint256[]], (addrs, shares)))`` over the
    sorted pairs.

    Guarantees:
      - SAME set, ANY input order → SAME hash (sorted before hashing),
      - DIFFERENT set (changed payee, changed share, added/removed entry) →
        DIFFERENT hash,
      - addresses compared case-insensitively (checksum-normalized).

    Rejects (raises ``ValueError``): an empty set, a duplicate payee
    (case-insensitive), a malformed address, a negative / >uint256 / non-int
    (incl. ``bool``) share. A malformed input raises HERE (fail-closed in the
    builder) rather than producing a junk commitment.
    """
    if payees is None:
        raise ValueError("payees must be a non-empty sequence of (address, share)")
    pairs = list(payees)
    if not pairs:
        raise ValueError("payees set must be non-empty")

    normalized: List[Tuple[str, int]] = []
    seen: set = set()
    for entry in pairs:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError(
                f"each payee entry must be a (address, share) pair; got {entry!r}"
            )
        addr_raw, share_raw = entry
        if not isinstance(addr_raw, str) or not is_address(addr_raw):
            raise ValueError(f"invalid payee address: {addr_raw!r}")
        addr = to_checksum_address(addr_raw)
        key = addr.lower()
        if key in seen:
            raise ValueError(f"duplicate payee in set: {addr}")
        seen.add(key)
        share = _normalize_share(share_raw)
        normalized.append((addr, share))

    # SORT for order-independence: same set in any input order → same ordering
    # here → same abi encoding → same hash.
    normalized.sort(key=lambda p: (p[0].lower(), p[1]))
    addrs = [p[0] for p in normalized]
    shares = [p[1] for p in normalized]
    # Canonical abi.encode of two parallel arrays binds BOTH the address list and
    # the share list (and their pairing-by-index) into one digest. The length is
    # encoded by abi, so an added/removed entry changes the encoding → the hash.
    encoded = abi_encode(["address[]", "uint256[]"], [addrs, shares])
    return keccak(encoded)


# --------------------------------------------------------------------------- #
# 2. EIP-712 typed-data build / digest / sign                                 #
# --------------------------------------------------------------------------- #

def _build_typed_data(payload: Dict[str, Any], chain_id: int) -> Dict[str, Any]:
    """Return the EIP-712 typed-data dict for ``encode_typed_data``.

    Validates field presence + normalizes types so a malformed input raises here
    rather than producing a junk hash. The domain is the SAME network-locked
    sp1054 domain; only the primaryType differs.
    """
    missing = [k for k in _REQUIRED if k not in payload]
    if missing:
        raise ValueError(
            f"per-stage payment authorization missing required fields: {missing!r}"
        )
    if not isinstance(payload["requester"], str):
        raise ValueError("requester must be an address string")
    return {
        "types": {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "chainId", "type": "uint256"},
            ],
            "PerStagePaymentAuthorization": [
                {"name": "requester", "type": "address"},
                {"name": "payee_set_hash", "type": "bytes32"},
                {"name": "total_max_spend_wei", "type": "uint256"},
                {"name": "job_nonce", "type": "bytes32"},
                {"name": "expiry_unix", "type": "uint256"},
                {"name": "request_hash", "type": "bytes32"},
            ],
        },
        "primaryType": "PerStagePaymentAuthorization",
        "domain": {
            "name": DOMAIN_NAME,
            "version": DOMAIN_VERSION,
            "chainId": int(chain_id),
        },
        "message": {
            "requester": str(payload["requester"]),
            "payee_set_hash": _to_bytes32("payee_set_hash", payload["payee_set_hash"]),
            "total_max_spend_wei": int(payload["total_max_spend_wei"]),
            "job_nonce": _to_bytes32("job_nonce", payload["job_nonce"]),
            "expiry_unix": int(payload["expiry_unix"]),
            "request_hash": _to_bytes32("request_hash", payload["request_hash"]),
        },
    }


def encode_per_stage_typed_data_hash(
    payload: Dict[str, Any], *, chain_id: int = DEFAULT_CHAIN_ID,
) -> bytes:
    """Canonical EIP-712 digest that gets signed:

      keccak256(0x19 0x01 || domain_separator || hashStruct(message))

    Chain-locked via chainId in the domain separator. Mirrors
    ``encode_payment_typed_data_hash`` exactly (only the primaryType differs), so
    the per-stage digest is stable across runs + distinct from the single-payee
    digest (different primaryType → different type hash).
    """
    typed = _build_typed_data(payload, chain_id=chain_id)
    signable = encode_typed_data(full_message=typed)
    return keccak(b"\x19\x01" + bytes(signable.header) + bytes(signable.body))


def sign_per_stage_authorization(
    payload: Dict[str, Any],
    private_key: Union[bytes, str],
    *,
    chain_id: int = DEFAULT_CHAIN_ID,
) -> bytes:
    """Sign the canonical per-stage payload. TEST HELPER + reference impl —
    production requesters sign with MetaMask / hardware wallets and submit the
    resulting signature directly. Returns the 65-byte (r || s || v) signature."""
    typed = _build_typed_data(payload, chain_id=chain_id)
    signable = encode_typed_data(full_message=typed)
    return bytes(Account.sign_message(signable, private_key=private_key).signature)


# --------------------------------------------------------------------------- #
# 3. Fail-closed verifier                                                     #
# --------------------------------------------------------------------------- #

class PerStageAuthorizationVerdict(NamedTuple):
    """A fail-closed verdict. ``authorized`` is True ONLY when every money-safety
    invariant holds; otherwise it is False with a human-readable ``reason``.
    NEVER raises for a logical reject — only a malformed signature SHAPE raises
    ``InvalidSignatureFormat`` (raised before a verdict can be formed)."""

    authorized: bool
    reason: str


def recover_per_stage_signer(
    payload: Dict[str, Any],
    signature: Union[bytes, str],
    *,
    chain_id: int = DEFAULT_CHAIN_ID,
) -> str:
    """Recover the EIP-712 signer of a per-stage authorization WITHOUT the payee set.

    For INGRESS authentication (sp1324): a paid multi-stage request can be attributed to its
    requester by confirming the per-stage auth's signer == ``payload["requester"]`` before the
    full money gate (set-hash / membership / cap) runs at the node-side commit (which needs the
    served payee set). Raises ``InvalidSignatureFormat`` on a malformed signature shape;
    ``ValueError`` on a malformed payload."""
    return _recover_signer(payload, signature, chain_id)


def _recover_signer(
    payload: Dict[str, Any],
    signature: Union[bytes, str],
    chain_id: int,
) -> str:
    """Recover the EIP-712 signer. Raises ``InvalidSignatureFormat`` on a malformed
    signature SHAPE (mirrors sp1054's verify); the typed-data builder raises
    ``ValueError`` on a malformed payload."""
    if isinstance(signature, str):
        s = signature.strip()
        if not s.startswith("0x"):
            s = "0x" + s
        try:
            sig_bytes = bytes.fromhex(s[2:])
        except ValueError as exc:
            raise InvalidSignatureFormat(f"signature hex parse failed: {exc!s}")
    elif isinstance(signature, (bytes, bytearray)):
        sig_bytes = bytes(signature)
    else:
        raise InvalidSignatureFormat(
            f"signature must be bytes or hex string, got "
            f"{type(signature).__name__}"
        )
    if len(sig_bytes) != 65:
        raise InvalidSignatureFormat(
            f"signature must be 65 bytes (r||s||v), got {len(sig_bytes)} bytes"
        )
    typed = _build_typed_data(payload, chain_id=chain_id)
    signable = encode_typed_data(full_message=typed)
    return Account.recover_message(signable, signature=sig_bytes)


def verify_per_stage_authorization(
    payload: Dict[str, Any],
    signature: Union[bytes, str],
    *,
    payees: Sequence[Tuple[str, int]],
    payee: str,
    share_wei: int,
    chain_id: int = DEFAULT_CHAIN_ID,
    now_unix: Optional[float] = None,
) -> PerStageAuthorizationVerdict:
    """Fail-closed verdict: authorize paying ``(payee, share_wei)`` under this
    requester-signed per-stage authorization.

    Confirms ALL of (any failure → ``authorized=False`` + a reason; NEVER a false
    authorize, and NEVER raises on a logical reject):
      (a) recovered signer == ``payload["requester"]`` (case-insensitive),
      (b) ``compute_payee_set_hash(payees) == payload["payee_set_hash"]`` (the
          supplied set matches the signed commitment),
      (c) ``(payee, share_wei)`` is a MEMBER of ``payees`` (exact case-insensitive
          address + EXACT share),
      (d) ``sum(shares in payees) <= total_max_spend_wei`` (NO over-authorization),
      (e) not expired (``now_unix < expiry_unix``; ``now_unix`` defaults to
          ``time.time()``),
      (f) ``request_hash`` present + bound (non-zero).

    Raises ``InvalidSignatureFormat`` ONLY for a malformed-signature SHAPE
    (matching sp1054) — that is an unparseable input, not a logical reject.

    The per-node commit (brick 4) gates on ``authorized == True``.
    """
    # Step 1: recover the signer. A malformed signature SHAPE raises (sp1054
    # parity); a malformed payload also raises in the builder. Both are
    # caller-input-shape errors, not logical authorization rejects.
    recovered = _recover_signer(payload, signature, chain_id)

    # (a) signer == requester (case-insensitive).
    requester = payload.get("requester")
    if not isinstance(requester, str) or recovered.lower() != requester.lower():
        return PerStageAuthorizationVerdict(
            False, "recovered signer does not match payload requester"
        )

    # (b) supplied set must hash to the signed commitment. A malformed supplied
    # set (empty / dup / bad addr / bad share) is a logical reject here — it
    # cannot match the signed commitment — so we catch the builder ValueError and
    # return a verdict rather than raising.
    try:
        computed = compute_payee_set_hash(payees)
    except (ValueError, TypeError) as exc:
        return PerStageAuthorizationVerdict(
            False, f"supplied payees set is invalid: {exc!s}"
        )
    try:
        signed_set_hash = _to_bytes32("payee_set_hash", payload["payee_set_hash"])
    except (ValueError, KeyError) as exc:
        return PerStageAuthorizationVerdict(
            False, f"payload payee_set_hash is invalid: {exc!s}"
        )
    if computed != signed_set_hash:
        return PerStageAuthorizationVerdict(
            False, "supplied payees do not match signed payee_set_hash (swapped set)"
        )

    # (f) request_hash present + bound (non-zero). Done before membership so a
    # missing binding fails fast regardless of payee.
    try:
        request_hash = _to_bytes32("request_hash", payload["request_hash"])
    except (ValueError, KeyError) as exc:
        return PerStageAuthorizationVerdict(
            False, f"payload request_hash is invalid: {exc!s}"
        )
    if request_hash == _ZERO_BYTES32:
        return PerStageAuthorizationVerdict(
            False, "request_hash is not bound (zero) — authorization not request-bound"
        )

    # (d) cumulative cap. Sum of ALL shares in the signed set must not exceed
    # total_max_spend_wei. NO over-authorization, even though each pair is itself
    # within the set. Use the same normalization the set-hash used.
    try:
        total_cap = int(payload["total_max_spend_wei"])
    except (ValueError, TypeError, KeyError) as exc:
        return PerStageAuthorizationVerdict(
            False, f"payload total_max_spend_wei is invalid: {exc!s}"
        )
    if total_cap < 0 or total_cap > _MAX_UINT256:
        return PerStageAuthorizationVerdict(
            False, "total_max_spend_wei out of uint256 range"
        )
    share_sum = sum(int(s) for _, s in payees)
    if share_sum > total_cap:
        return PerStageAuthorizationVerdict(
            False,
            f"share sum {share_sum} exceeds total_max_spend_wei cap {total_cap}"
            " (over-authorization refused)",
        )

    # (c) membership: exact case-insensitive address + EXACT share. The payee
    # MUST be inside the signed set with the EXACT share — no payee or share
    # outside the signed set is ever authorized.
    if not isinstance(payee, str) or not is_address(payee):
        return PerStageAuthorizationVerdict(
            False, f"queried payee is not a valid address: {payee!r}"
        )
    try:
        queried_share = _normalize_share(share_wei)
    except (ValueError, TypeError) as exc:
        return PerStageAuthorizationVerdict(
            False, f"queried share is invalid: {exc!s}"
        )
    payee_key = payee.lower()
    member = False
    for addr_raw, share_raw in payees:
        if str(addr_raw).lower() == payee_key and int(share_raw) == queried_share:
            member = True
            break
    if not member:
        return PerStageAuthorizationVerdict(
            False,
            f"({payee}, {queried_share}) is not a member of the signed payee set",
        )

    # (e) expiry: now < expiry_unix (strict). now defaults to wall-clock.
    if now_unix is None:
        import time
        now_unix = time.time()
    try:
        expiry = int(payload["expiry_unix"])
    except (ValueError, TypeError, KeyError) as exc:
        return PerStageAuthorizationVerdict(
            False, f"payload expiry_unix is invalid: {exc!s}"
        )
    if int(now_unix) >= expiry:
        return PerStageAuthorizationVerdict(
            False, f"authorization expired (now {int(now_unix)} >= expiry {expiry})"
        )

    return PerStageAuthorizationVerdict(True, "authorized")
