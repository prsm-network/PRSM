"""Sprint 1061+ — Intel SGX TCB-recency enforcement (Tier 3 hardening).

A validly-SIGNED DCAP quote can still come from a TEE running a DOWNLEVEL (known-
vulnerable) microcode/TCB. The signature chain (sp1044) proves authenticity; it does
NOT prove the platform is up to date. This module derives the platform's TCB status
by comparing the PCK cert's TCB level against Intel's signed TCB-Info, so the
verifier can reject an out-of-date / revoked TEE.

Brick 1 (this file, sprint 1061): ``parse_pck_sgx_extension`` reads the platform TCB
level (16 SGX TCB component SVNs + PCESVN + FMSPC) out of the PCK leaf's Intel SGX
extension (OID 1.2.840.113741.1.13.1). Later bricks parse + verify Intel's TCB-Info
JSON and run the tcbLevels comparison to a configurable status floor.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Tuple

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from cryptography.x509.oid import ObjectIdentifier
from pyasn1.codec.der.decoder import decode as der_decode

# Intel SGX PCK certificate extension OIDs.
SGX_EXT_OID = "1.2.840.113741.1.13.1"
_TCB_OID = SGX_EXT_OID + ".2"
_PCESVN_OID = SGX_EXT_OID + ".2.17"
_FMSPC_OID = SGX_EXT_OID + ".4"
# the 16 component-SVN OIDs are SGX_EXT_OID + ".2.<1..16>"
_COMP_OIDS = tuple(f"{SGX_EXT_OID}.2.{i}" for i in range(1, 17))


@dataclass(frozen=True)
class PckTcb:
    """The platform TCB level read off a PCK leaf certificate."""
    fmspc: bytes            # 6-byte platform family (keys the TCB-Info lookup)
    comp_svns: List[int]    # 16 SGX TCB component SVNs (0-255 each)
    pcesvn: int             # PCE SVN


def _iter_pairs(seq):
    """Yield (oid_str, value) for each (OID, value) pair in a decoded SEQUENCE.
    Skips malformed (non-2-element / non-OID-first) members defensively."""
    for i in range(len(seq)):
        item = seq[i]
        try:
            if len(item) < 2:
                continue
            oid = str(item[0])
        except Exception:  # noqa: BLE001
            continue
        yield oid, item[1]


def parse_pck_sgx_extension(cert: x509.Certificate) -> PckTcb:
    """Parse the Intel SGX extension out of a PCK leaf certificate.

    Raises ``ValueError`` if the extension is absent, unparseable, or missing the
    TCB / FMSPC / a full set of 16 component SVNs + PCESVN — a missing TCB level
    must fail closed (we can't assert recency without it), not default to "ok"."""
    try:
        ext = cert.extensions.get_extension_for_oid(ObjectIdentifier(SGX_EXT_OID))
    except x509.ExtensionNotFound:
        raise ValueError("PCK certificate has no Intel SGX extension "
                         f"({SGX_EXT_OID}) — cannot determine TCB level")
    raw = ext.value.value if isinstance(ext.value, x509.UnrecognizedExtension) else ext.value
    try:
        decoded, _ = der_decode(bytes(raw))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"SGX extension is not valid DER: {exc}")

    fmspc = None
    tcb_seq = None
    for oid, val in _iter_pairs(decoded):
        if oid == _FMSPC_OID:
            fmspc = bytes(val)
        elif oid == _TCB_OID:
            tcb_seq = val
    if fmspc is None:
        raise ValueError("SGX extension missing FMSPC")
    if tcb_seq is None:
        raise ValueError("SGX extension missing TCB sequence")

    comp_by_oid = {}
    pcesvn = None
    for oid, val in _iter_pairs(tcb_seq):
        if oid in _COMP_OIDS:
            comp_by_oid[oid] = int(val)
        elif oid == _PCESVN_OID:
            pcesvn = int(val)
    comp_svns = [comp_by_oid.get(o) for o in _COMP_OIDS]
    if any(c is None for c in comp_svns):
        missing = [i + 1 for i, c in enumerate(comp_svns) if c is None]
        raise ValueError(f"SGX extension missing TCB component SVN(s): {missing}")
    if pcesvn is None:
        raise ValueError("SGX extension missing PCESVN")
    return PckTcb(fmspc=fmspc, comp_svns=comp_svns, pcesvn=pcesvn)


# ── bricks 2+3: Intel TCB-Info parse/verify + status comparison ────────────────

class TcbInfoError(ValueError):
    """The TCB-Info is unparseable, its signature is invalid, or it doesn't apply to
    the platform (FMSPC mismatch). Raised (not swallowed) so the verifier fails
    closed — an unverifiable TCB-Info must NOT be treated as 'up to date'."""


@dataclass(frozen=True)
class TcbLevel:
    comp_svns: List[int]   # 16 component SVNs the platform must meet/exceed
    pcesvn: int
    status: str            # e.g. "UpToDate", "OutOfDate", "Revoked", ...


@dataclass(frozen=True)
class TcbInfo:
    fmspc: bytes
    tcb_levels: List[TcbLevel]   # as published — Intel orders these descending


def _balance_object(blob: bytes, j: int) -> int:
    """Given ``blob[j] == '{'``, return the index just past the matching '}'."""
    depth = 0
    in_str = False
    esc = False
    for k in range(j, len(blob)):
        c = blob[k]
        if in_str:
            if esc:
                esc = False
            elif c == 0x5C:   # backslash
                esc = True
            elif c == 0x22:   # quote
                in_str = False
            continue
        if c == 0x22:
            in_str = True
        elif c == 0x7B:       # {
            depth += 1
        elif c == 0x7D:       # }
            depth -= 1
            if depth == 0:
                return k + 1
    raise TcbInfoError("TCB-Info tcbInfo object is not brace-balanced")


def _extract_tcb_info_bytes(blob: bytes) -> bytes:
    """Return the EXACT bytes of the TOP-LEVEL ``tcbInfo`` object as they appear in
    the signed envelope. The signature is over these literal bytes, so we must not
    re-serialize (key order / whitespace would differ).

    sp1063 review (CRITICAL): a naive ``blob.find('"tcbInfo"')`` matches a
    string-embedded occurrence or the FIRST of duplicate keys, which can diverge from
    what ``json.loads`` parses (it keeps the LAST duplicate) → a signed decoy + an
    unsigned duplicate would verify-then-parse-different. So scan for ``"tcbInfo"``
    only as a DEPTH-1 KEY (inside the root object, followed by ``:`` then ``{``), and
    reject a second one. The caller additionally parses the returned bytes directly
    (not the whole envelope) and rejects duplicate keys."""
    n = len(blob)
    i = 0
    # advance to the root object's opening brace
    while i < n and blob[i] in (0x20, 0x09, 0x0A, 0x0D):
        i += 1
    if i >= n or blob[i] != 0x7B:
        raise TcbInfoError("TCB-Info is not a JSON object")
    root_open = i
    depth = 0
    in_str = False
    esc = False
    str_start = -1
    found = None
    k = root_open
    while k < n:
        c = blob[k]
        if in_str:
            if esc:
                esc = False
            elif c == 0x5C:
                esc = True
            elif c == 0x22:
                in_str = False
                # a string token just closed at depth 1 → candidate key
                if depth == 1:
                    token = blob[str_start + 1:k]
                    if token == b"tcbInfo":
                        m = k + 1
                        while m < n and blob[m] in (0x20, 0x09, 0x0A, 0x0D):
                            m += 1
                        if m < n and blob[m] == 0x3A:  # ':'
                            m += 1
                            while m < n and blob[m] in (0x20, 0x09, 0x0A, 0x0D):
                                m += 1
                            if m < n and blob[m] == 0x7B:
                                if found is not None:
                                    raise TcbInfoError("duplicate top-level tcbInfo key")
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
        raise TcbInfoError("TCB-Info JSON has no top-level tcbInfo object")
    return blob[found[0]:found[1]]


def _no_duplicate_keys(pairs):
    """json object hook that rejects duplicate keys (a signed-vs-parsed divergence
    vector — sp1063 review CRITICAL)."""
    seen = {}
    for k, v in pairs:
        if k in seen:
            raise TcbInfoError(f"duplicate JSON key {k!r} in TCB-Info")
        seen[k] = v
    return seen


def parse_and_verify_tcb_info(blob: bytes, signer_cert: x509.Certificate) -> TcbInfo:
    """Parse + signature-verify a signed Intel TCB-Info JSON.

    The ECDSA-P256 signature (raw r||s, hex) covers the literal ``tcbInfo`` bytes;
    we verify it against ``signer_cert`` (the operator-pinned Intel TCB Signing cert
    — the trust anchor for the TCB dimension). Any failure raises ``TcbInfoError``
    (fail-closed). Returns the parsed FMSPC + tcbLevels."""
    if isinstance(blob, str):
        blob = blob.encode()
    try:
        envelope = json.loads(blob, object_pairs_hook=_no_duplicate_keys)
    except TcbInfoError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise TcbInfoError(f"TCB-Info is not valid JSON: {exc}")
    sig_hex = envelope.get("signature")
    if not sig_hex:
        raise TcbInfoError("TCB-Info has no signature")
    try:
        raw_sig = bytes.fromhex(sig_hex)
        if len(raw_sig) != 64:
            raise ValueError(f"expected 64-byte r||s, got {len(raw_sig)}")
        der_sig = encode_dss_signature(
            int.from_bytes(raw_sig[:32], "big"), int.from_bytes(raw_sig[32:], "big"))
    except Exception as exc:  # noqa: BLE001
        raise TcbInfoError(f"TCB-Info signature malformed: {exc}")

    signed_bytes = _extract_tcb_info_bytes(blob)
    pub = signer_cert.public_key()
    if not isinstance(pub, ec.EllipticCurvePublicKey):
        raise TcbInfoError("TCB Signing cert key is not EC (expected ECDSA-P256)")
    try:
        pub.verify(der_sig, signed_bytes, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature:
        raise TcbInfoError("TCB-Info signature invalid (not signed by the TCB Signing cert)")

    # sp1063 review (CRITICAL) — derive the TCB object from the EXACT bytes we just
    # signature-verified, NOT from envelope["tcbInfo"] (which, with duplicate keys,
    # could be a different, unsigned object than the one signed).
    try:
        tcb_obj = json.loads(signed_bytes, object_pairs_hook=_no_duplicate_keys)
    except TcbInfoError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise TcbInfoError(f"verified tcbInfo bytes are not valid JSON: {exc}")
    fmspc_hex = tcb_obj.get("fmspc")
    if not fmspc_hex:
        raise TcbInfoError("TCB-Info missing fmspc")
    levels = []
    for lvl in tcb_obj.get("tcbLevels", []):
        tcb = lvl.get("tcb", {})
        comps = [c.get("svn") for c in tcb.get("sgxtcbcomponents", [])]
        status = lvl.get("tcbStatus")
        pcesvn = tcb.get("pcesvn")
        if len(comps) != 16 or any(c is None for c in comps) or pcesvn is None or not status:
            raise TcbInfoError("TCB-Info tcbLevel malformed (need 16 components, "
                               "pcesvn, tcbStatus)")
        levels.append(TcbLevel(comp_svns=[int(c) for c in comps],
                               pcesvn=int(pcesvn), status=str(status)))
    if not levels:
        raise TcbInfoError("TCB-Info has no tcbLevels")
    return TcbInfo(fmspc=bytes.fromhex(fmspc_hex), tcb_levels=levels)


def _platform_meets(pck: PckTcb, level: TcbLevel) -> bool:
    """The platform satisfies a TCB level iff EVERY one of its 16 component SVNs is
    >= the level's AND its PCESVN is >= the level's. A single downlevel component
    disqualifies the level (matches Intel's procedure)."""
    if pck.pcesvn < level.pcesvn:
        return False
    return all(p >= l for p, l in zip(pck.comp_svns, level.comp_svns))


def evaluate_tcb_status(pck: PckTcb, tcb_info: TcbInfo) -> str:
    """Return the platform's tcbStatus: the status of the HIGHEST tcbLevel the
    platform meets (Intel publishes tcbLevels descending, so the first match is the
    highest). A platform below every known level → conservative "OutOfDate".

    Raises ``TcbInfoError`` on FMSPC mismatch — TCB-Info for a different platform
    family must never be applied (it would assert recency against the wrong baseline)."""
    if pck.fmspc != tcb_info.fmspc:
        raise TcbInfoError(
            f"FMSPC mismatch: PCK {pck.fmspc.hex()} != TCB-Info {tcb_info.fmspc.hex()}")
    for level in tcb_info.tcb_levels:
        if _platform_meets(pck, level):
            return level.status
    return "OutOfDate"


# ── brick 4: configurable TCB status policy ────────────────────────────────────

# The statuses the default policy accepts. UpToDate is fully current; the *Needed
# variants are accepted-with-caveat (a BIOS config / SW mitigation is advised but the
# microcode TCB itself is current). OutOfDate*/Revoked are rejected — those are a
# downlevel or revoked TCB (the exact recency gap this brick exists to close).
_DEFAULT_ALLOWED = frozenset({
    "UpToDate",
    "SWHardeningNeeded",
    "ConfigurationNeeded",
    "ConfigurationAndSWHardeningNeeded",
})


@dataclass(frozen=True)
class TcbPolicy:
    """Decides whether a derived tcbStatus is acceptable. ``allowed_statuses`` is the
    allow-list; anything not in it (incl. an unrecognized status) is rejected, so the
    policy fails closed."""
    allowed_statuses: frozenset

    def is_acceptable(self, status: str) -> bool:
        return status in self.allowed_statuses


def default_tcb_policy() -> TcbPolicy:
    return TcbPolicy(allowed_statuses=_DEFAULT_ALLOWED)


def tcb_policy_from_env(environ) -> TcbPolicy:
    """Build a policy from ``PRSM_INTEL_SGX_TCB_ALLOWED_STATUSES`` (comma-separated
    status names); unset → the default policy."""
    raw = (environ.get("PRSM_INTEL_SGX_TCB_ALLOWED_STATUSES", "") or "").strip()
    if not raw:
        return default_tcb_policy()
    return TcbPolicy(allowed_statuses=frozenset(
        s.strip() for s in raw.split(",") if s.strip()))
