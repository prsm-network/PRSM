"""Sprint 1062 — parse+verify Intel TCB-Info and derive the platform TCB status (bricks 2+3).

Brick 2: ``parse_and_verify_tcb_info`` extracts the exact ``tcbInfo`` bytes from the
signed TCB-Info JSON (Intel PCS "Get SGX TCB Info" v3), verifies the ECDSA-P256
signature over those bytes against the TCB Signing cert's key (raw r||s), and parses
the FMSPC + tcbLevels. Brick 3: ``evaluate_tcb_status`` walks tcbLevels (descending)
and returns the tcbStatus of the highest level the platform's PCK TCB satisfies (all
16 component SVNs >= the level's AND pcesvn >=). A platform below every known level
→ the conservative "OutOfDate".
"""
from __future__ import annotations

import json

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

from prsm.compute.inference.sgx_tcb import (
    PckTcb,
    TcbInfoError,
    evaluate_tcb_status,
    parse_and_verify_tcb_info,
)

FMSPC = "00906ed50000"


def _raw_sig(priv, msg):
    der = priv.sign(msg, ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der)
    return r.to_bytes(32, "big") + s.to_bytes(32, "big")


def _tcb_level(svns, pcesvn, status):
    return {"tcb": {"sgxtcbcomponents": [{"svn": v} for v in svns], "pcesvn": pcesvn},
            "tcbStatus": status}


def _signed_tcb_info(priv, *, fmspc=FMSPC, levels=None):
    """Assemble a signed TCB-Info: sign the EXACT tcbInfo bytes, then splice them
    into the envelope verbatim (so the verifier recovers byte-identical input)."""
    if levels is None:
        levels = [
            _tcb_level([10] * 16, 12, "UpToDate"),
            _tcb_level([8] * 16, 10, "OutOfDate"),
        ]
    tcb_info_obj = {"id": "SGX", "version": 3, "fmspc": fmspc, "pceId": "0000",
                    "tcbType": 0, "tcbEvaluationDataNumber": 15, "tcbLevels": levels}
    tcb_bytes = json.dumps(tcb_info_obj, separators=(",", ":")).encode()
    sig_hex = _raw_sig(priv, tcb_bytes).hex()
    return b'{"tcbInfo":' + tcb_bytes + b',"signature":"' + sig_hex.encode() + b'"}'


def _signer_cert(priv):
    import datetime as _dt
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    return (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Intel TCB Signing")]))
            .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Intel TCB Signing")]))
            .public_key(priv.public_key()).serial_number(1)
            .not_valid_before(_dt.datetime(2025, 1, 1))
            .not_valid_after(_dt.datetime(2035, 1, 1))
            .sign(priv, hashes.SHA256()))


def test_parse_and_verify_good_signature():
    priv = ec.generate_private_key(ec.SECP256R1())
    blob = _signed_tcb_info(priv)
    info = parse_and_verify_tcb_info(blob, _signer_cert(priv))
    assert info.fmspc.hex() == FMSPC
    assert len(info.tcb_levels) == 2


def test_tampered_tcb_info_fails_signature():
    priv = ec.generate_private_key(ec.SECP256R1())
    blob = _signed_tcb_info(priv)
    tampered = blob.replace(b'"UpToDate"', b'"Revoked!!"')  # changes signed bytes
    with pytest.raises(TcbInfoError):
        parse_and_verify_tcb_info(tampered, _signer_cert(priv))


def test_wrong_signer_fails():
    priv = ec.generate_private_key(ec.SECP256R1())
    other = ec.generate_private_key(ec.SECP256R1())
    blob = _signed_tcb_info(priv)
    with pytest.raises(TcbInfoError):
        parse_and_verify_tcb_info(blob, _signer_cert(other))


def test_status_uptodate_when_platform_meets_top_level():
    priv = ec.generate_private_key(ec.SECP256R1())
    info = parse_and_verify_tcb_info(_signed_tcb_info(priv), _signer_cert(priv))
    pck = PckTcb(fmspc=bytes.fromhex(FMSPC), comp_svns=[10] * 16, pcesvn=12)
    assert evaluate_tcb_status(pck, info) == "UpToDate"


def test_status_outofdate_when_platform_meets_only_lower_level():
    priv = ec.generate_private_key(ec.SECP256R1())
    info = parse_and_verify_tcb_info(_signed_tcb_info(priv), _signer_cert(priv))
    pck = PckTcb(fmspc=bytes.fromhex(FMSPC), comp_svns=[8] * 16, pcesvn=10)
    assert evaluate_tcb_status(pck, info) == "OutOfDate"


def test_status_one_component_below_drops_to_lower_level():
    """All-but-one component meets UpToDate, but one is below → does NOT qualify for
    UpToDate; falls to the lower level (a single downlevel component matters)."""
    priv = ec.generate_private_key(ec.SECP256R1())
    info = parse_and_verify_tcb_info(_signed_tcb_info(priv), _signer_cert(priv))
    svns = [10] * 16
    svns[5] = 8  # one component below the UpToDate level (10) but >= lower level (8)
    pck = PckTcb(fmspc=bytes.fromhex(FMSPC), comp_svns=svns, pcesvn=12)
    assert evaluate_tcb_status(pck, info) == "OutOfDate"


def test_status_below_all_levels_is_outofdate():
    priv = ec.generate_private_key(ec.SECP256R1())
    info = parse_and_verify_tcb_info(_signed_tcb_info(priv), _signer_cert(priv))
    pck = PckTcb(fmspc=bytes.fromhex(FMSPC), comp_svns=[1] * 16, pcesvn=1)
    assert evaluate_tcb_status(pck, info) == "OutOfDate"


def test_duplicate_tcbinfo_key_signed_decoy_plus_unsigned_evil_is_rejected():
    """sp1063 review CRITICAL — a validly-signed decoy tcbInfo + an unsigned
    duplicate 'tcbInfo' (which json would keep as the last value) must NOT verify:
    the signed bytes and the parsed object must be the SAME object."""
    priv = ec.generate_private_key(ec.SECP256R1())
    decoy = {"id": "SGX", "version": 3, "fmspc": FMSPC, "pceId": "0000", "tcbType": 0,
             "tcbEvaluationDataNumber": 15,
             "tcbLevels": [_tcb_level([10] * 16, 12, "UpToDate")]}
    decoy_bytes = json.dumps(decoy, separators=(",", ":")).encode()
    sig = _raw_sig(priv, decoy_bytes).hex().encode()
    evil = {"id": "SGX", "version": 3, "fmspc": FMSPC, "pceId": "0000", "tcbType": 0,
            "tcbEvaluationDataNumber": 15,
            "tcbLevels": [_tcb_level([1] * 16, 1, "UpToDate")]}  # grants UpToDate to junk
    evil_bytes = json.dumps(evil, separators=(",", ":")).encode()
    blob = (b'{"tcbInfo":' + decoy_bytes + b',"tcbInfo":' + evil_bytes
            + b',"signature":"' + sig + b'"}')
    with pytest.raises(TcbInfoError):
        parse_and_verify_tcb_info(blob, _signer_cert(priv))


def test_embedded_tcbinfo_string_does_not_steal_the_match():
    """A 'tcbInfo' substring inside an earlier string value must not be extracted as
    the signed object (it wouldn't verify → fail-closed, not a wrong-object parse)."""
    priv = ec.generate_private_key(ec.SECP256R1())
    obj = {"note": 'see "tcbInfo": below', "id": "SGX", "version": 3, "fmspc": FMSPC,
           "pceId": "0000", "tcbType": 0, "tcbEvaluationDataNumber": 15,
           "tcbLevels": [_tcb_level([10] * 16, 12, "UpToDate")]}
    tcb_bytes = json.dumps(obj, separators=(",", ":")).encode()
    blob = b'{"tcbInfo":' + tcb_bytes + b',"signature":"' + _raw_sig(priv, tcb_bytes).hex().encode() + b'"}'
    info = parse_and_verify_tcb_info(blob, _signer_cert(priv))  # must still verify the real object
    assert info.fmspc.hex() == FMSPC


def test_fmspc_mismatch_raises():
    """TCB-Info for a DIFFERENT platform family must not be applied to this PCK."""
    priv = ec.generate_private_key(ec.SECP256R1())
    info = parse_and_verify_tcb_info(_signed_tcb_info(priv), _signer_cert(priv))
    pck = PckTcb(fmspc=bytes.fromhex("ffffffffffff"), comp_svns=[10] * 16, pcesvn=12)
    with pytest.raises(TcbInfoError):
        evaluate_tcb_status(pck, info)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
