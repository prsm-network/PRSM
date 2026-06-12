"""Sprint 1079 — Intel SGX QE-Identity pinning (Tier 3 attestation hardening).

The DCAP chain (sp1044) proves: the ISV report was signed by the Attestation Key, the
AK is bound to *a* Quoting Enclave (SHA256(ak_pub||qe_auth) == QE report_data[:32]),
the QE report was signed by the PCK leaf, and the PCK chain verifies to Intel's root.
What it does NOT prove is that the QE is a *genuine Intel* Quoting Enclave — a
different enclave with a valid-looking structure could mint the AK→QE binding. This
closes that gap: verify the QE report's MRSIGNER / ISVPRODID / ISVSVN against Intel's
signed QE Identity collateral (the same `enclaveIdentity`+`signature` shape as the
TCB-Info, signed by the same Intel SGX TCB Signing cert).

SGX report body offsets (the QE report is a 384-byte SGX report body):
  MRSIGNER @ 0x80 (32) · ISVPRODID @ 0x100 (u16 LE) · ISVSVN @ 0x102 (u16 LE).

Mirrors the sp1061-1063 TCB-recency design: parse the signed JSON, verify the
ECDSA-P256 signature over the EXACT enclaveIdentity bytes (no re-serialize), reject
duplicate keys, then enforce MRSIGNER+ISVPRODID equality and an ISVSVN status floor.
"""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import List

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

# SGX report body offsets (same layout as the ISV report).
_MRSIGNER_OFF = 0x80
_ISVPRODID_OFF = 0x100
_ISVSVN_OFF = 0x102


class QeIdentityError(ValueError):
    """The QE Identity is unparseable, its signature is invalid, or the QE report's
    identity (MRSIGNER/ISVPRODID/ISVSVN) doesn't match it. Raised (not swallowed) so
    the verifier fails closed — an unverifiable QE must NOT be treated as genuine."""


@dataclass(frozen=True)
class QeReportIdentity:
    mrsigner: bytes      # 32
    isvprodid: int
    isvsvn: int


@dataclass(frozen=True)
class QeTcbLevel:
    isvsvn: int
    status: str


@dataclass(frozen=True)
class QeIdentity:
    mrsigner: bytes
    isvprodid: int
    tcb_levels: List[QeTcbLevel]   # Intel orders these descending by isvsvn


def parse_qe_report_identity(qe_report: bytes) -> QeReportIdentity:
    """Extract MRSIGNER/ISVPRODID/ISVSVN from a 384-byte QE report body."""
    if len(qe_report) < _ISVSVN_OFF + 2:
        raise QeIdentityError(
            f"QE report too short ({len(qe_report)} bytes) for the identity fields")
    mrsigner = qe_report[_MRSIGNER_OFF:_MRSIGNER_OFF + 32]
    (isvprodid,) = struct.unpack_from("<H", qe_report, _ISVPRODID_OFF)
    (isvsvn,) = struct.unpack_from("<H", qe_report, _ISVSVN_OFF)
    return QeReportIdentity(mrsigner=mrsigner, isvprodid=isvprodid, isvsvn=isvsvn)


def _no_duplicate_keys(pairs):
    seen = {}
    for k, v in pairs:
        if k in seen:
            raise QeIdentityError(f"duplicate JSON key {k!r} in QE Identity")
        seen[k] = v
    return seen


def _balance_object(blob: bytes, j: int) -> int:
    depth = 0
    in_str = False
    esc = False
    for k in range(j, len(blob)):
        c = blob[k]
        if in_str:
            if esc:
                esc = False
            elif c == 0x5C:
                esc = True
            elif c == 0x22:
                in_str = False
            continue
        if c == 0x22:
            in_str = True
        elif c == 0x7B:
            depth += 1
        elif c == 0x7D:
            depth -= 1
            if depth == 0:
                return k + 1
    raise QeIdentityError("enclaveIdentity object is not brace-balanced")


def _extract_enclave_identity_bytes(blob: bytes) -> bytes:
    """The EXACT bytes of the top-level ``enclaveIdentity`` object (the signed
    preimage). Matches the top-level key only (depth-1) so a string-embedded or
    duplicate occurrence can't divert the signed-vs-parsed pair (sp1063 lesson)."""
    n = len(blob)
    i = 0
    while i < n and blob[i] in (0x20, 0x09, 0x0A, 0x0D):
        i += 1
    if i >= n or blob[i] != 0x7B:
        raise QeIdentityError("QE Identity is not a JSON object")
    depth = 0
    in_str = False
    esc = False
    str_start = -1
    found = None
    k = i
    while k < n:
        c = blob[k]
        if in_str:
            if esc:
                esc = False
            elif c == 0x5C:
                esc = True
            elif c == 0x22:
                in_str = False
                if depth == 1:
                    token = blob[str_start + 1:k]
                    if token == b"enclaveIdentity":
                        m = k + 1
                        while m < n and blob[m] in (0x20, 0x09, 0x0A, 0x0D):
                            m += 1
                        if m < n and blob[m] == 0x3A:
                            m += 1
                            while m < n and blob[m] in (0x20, 0x09, 0x0A, 0x0D):
                                m += 1
                            if m < n and blob[m] == 0x7B:
                                if found is not None:
                                    raise QeIdentityError("duplicate enclaveIdentity key")
                                found = (m, _balance_object(blob, m))
            k += 1
            continue
        if c == 0x22:
            in_str = True
            str_start = k
        elif c == 0x7B:
            depth += 1
        elif c == 0x7D:
            depth -= 1
            if depth == 0:
                break
        k += 1
    if found is None:
        raise QeIdentityError("QE Identity has no top-level enclaveIdentity object")
    return blob[found[0]:found[1]]


def parse_and_verify_qe_identity(blob: bytes, signer_cert) -> QeIdentity:
    """Parse + ECDSA-P256 signature-verify a signed Intel QE Identity JSON against the
    operator-pinned TCB Signing cert (the same signer as the TCB-Info). The signature
    covers the literal ``enclaveIdentity`` bytes. Fail-closed on any error."""
    if isinstance(blob, str):
        blob = blob.encode()
    try:
        envelope = json.loads(blob, object_pairs_hook=_no_duplicate_keys)
    except QeIdentityError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise QeIdentityError(f"QE Identity is not valid JSON: {exc}")
    sig_hex = envelope.get("signature")
    if not sig_hex:
        raise QeIdentityError("QE Identity has no signature")
    try:
        raw = bytes.fromhex(sig_hex)
        if len(raw) != 64:
            raise ValueError(f"expected 64-byte r||s, got {len(raw)}")
        der_sig = encode_dss_signature(
            int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big"))
    except Exception as exc:  # noqa: BLE001
        raise QeIdentityError(f"QE Identity signature malformed: {exc}")

    signed_bytes = _extract_enclave_identity_bytes(blob)
    pub = signer_cert.public_key()
    if not isinstance(pub, ec.EllipticCurvePublicKey):
        raise QeIdentityError("QE-Identity signer cert key is not EC (expected ECDSA-P256)")
    try:
        pub.verify(der_sig, signed_bytes, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature:
        raise QeIdentityError("QE Identity signature invalid (not signed by the TCB Signing cert)")

    try:
        ei = json.loads(signed_bytes, object_pairs_hook=_no_duplicate_keys)
    except QeIdentityError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise QeIdentityError(f"verified enclaveIdentity bytes are not valid JSON: {exc}")
    try:
        mrsigner = bytes.fromhex(ei["mrsigner"])
        isvprodid = int(ei["isvprodid"])
    except (KeyError, ValueError, TypeError) as exc:
        raise QeIdentityError(f"QE Identity missing/invalid mrsigner/isvprodid: {exc}")
    levels = []
    for lvl in ei.get("tcbLevels", []):
        tcb = lvl.get("tcb", {})
        isvsvn = tcb.get("isvsvn")
        status = lvl.get("tcbStatus")
        if isvsvn is None or not status:
            raise QeIdentityError("QE Identity tcbLevel missing isvsvn/tcbStatus")
        levels.append(QeTcbLevel(isvsvn=int(isvsvn), status=str(status)))
    if not levels:
        raise QeIdentityError("QE Identity has no tcbLevels")
    return QeIdentity(mrsigner=mrsigner, isvprodid=isvprodid, tcb_levels=levels)


def qe_tcb_status(qe_rep: QeReportIdentity, identity: QeIdentity) -> str:
    """The QE's tcbStatus = the status of the HIGHEST tcbLevel whose isvsvn the QE
    meets (levels are descending). Below all → conservative 'OutOfDate'."""
    for level in identity.tcb_levels:
        if qe_rep.isvsvn >= level.isvsvn:
            return level.status
    return "OutOfDate"


# Default-acceptable QE statuses (mirror the TCB policy default).
_DEFAULT_ALLOWED = frozenset({
    "UpToDate", "SWHardeningNeeded", "ConfigurationNeeded",
    "ConfigurationAndSWHardeningNeeded",
})


def verify_qe_identity(qe_rep: QeReportIdentity, identity: QeIdentity,
                       allowed_statuses=None) -> None:
    """Raise ``QeIdentityError`` unless the QE report's identity matches Intel's QE
    Identity: MRSIGNER + ISVPRODID equal, and the ISVSVN-derived status is acceptable.
    Returns None on success."""
    if qe_rep.mrsigner != identity.mrsigner:
        raise QeIdentityError(
            f"QE MRSIGNER {qe_rep.mrsigner.hex()[:16]}… != Intel QE Identity "
            f"{identity.mrsigner.hex()[:16]}… (not a genuine Intel QE)")
    if qe_rep.isvprodid != identity.isvprodid:
        raise QeIdentityError(
            f"QE ISVPRODID {qe_rep.isvprodid} != Intel QE Identity {identity.isvprodid}")
    allowed = _DEFAULT_ALLOWED if allowed_statuses is None else frozenset(allowed_statuses)
    status = qe_tcb_status(qe_rep, identity)
    if status not in allowed:
        raise QeIdentityError(
            f"QE TCB status '{status}' is not acceptable (downlevel/revoked QE; "
            f"isvsvn={qe_rep.isvsvn})")
