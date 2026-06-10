"""Sprint 1065 — AMD SEV-SNP TCB enforcement wired into AMDSEVSNPBackend (brick 2).

End-to-end: a validly-SIGNED SEV-SNP report from a DOWNLEVEL TCB (below the floor) or
a SPOOFED TCB (report REPORTED_TCB != the VCEK's issued TCB) is rejected
(vendor_verified=False) when a TCB policy is configured; the same report passes when
it meets the floor and binds. Without a policy, behavior is unchanged.
"""
from __future__ import annotations

import datetime as _dt
import struct

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.x509.oid import NameOID, ObjectIdentifier
from pyasn1.type import univ
from pyasn1.codec.der.encoder import encode

from prsm.compute.inference.amd_sev_snp import AMDSEVSNPBackend
from prsm.compute.inference.amd_tcb import SnpTcbPolicy

_SIG_OFF = 0x2A0
_REPORTED_TCB_OFF = 0x180
_OIDS = {"1.3.6.1.4.1.3704.1.3.1": 0, "1.3.6.1.4.1.3704.1.3.2": 1,
         "1.3.6.1.4.1.3704.1.3.3": 2, "1.3.6.1.4.1.3704.1.3.8": 3}  # bl,tee,snp,uc index


def _mk_cert(subj, issuer, subj_pub, issuer_priv, *, ca, exts=()):
    b = (x509.CertificateBuilder()
         .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subj)]))
         .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer)]))
         .public_key(subj_pub).serial_number(x509.random_serial_number())
         .not_valid_before(_dt.datetime(2025, 1, 1))
         .not_valid_after(_dt.datetime(2035, 1, 1))
         .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True))
    for oid, der in exts:
        b = b.add_extension(x509.UnrecognizedExtension(ObjectIdentifier(oid), der), critical=False)
    return b.sign(issuer_priv, hashes.SHA384())


def _envelope(*, report_tcb, vcek_tcb):
    """report_tcb / vcek_tcb are (bl, tee, snp, uc) tuples."""
    ark_priv = ec.generate_private_key(ec.SECP384R1())
    ask_priv = ec.generate_private_key(ec.SECP384R1())
    vcek_priv = ec.generate_private_key(ec.SECP384R1())
    ark = _mk_cert("AMD ARK", "AMD ARK", ark_priv.public_key(), ark_priv, ca=True)
    ask = _mk_cert("AMD ASK", "AMD ARK", ask_priv.public_key(), ark_priv, ca=True)
    spl_exts = [(oid, encode(univ.Integer(vcek_tcb[idx]))) for oid, idx in _OIDS.items()]
    vcek = _mk_cert("AMD VCEK", "AMD ASK", vcek_priv.public_key(), ask_priv, ca=False, exts=spl_exts)

    report = bytearray(1184)
    struct.pack_into("<I", report, 0, 2)
    tcb = bytearray(8)
    tcb[0], tcb[1], tcb[6], tcb[7] = report_tcb
    report[_REPORTED_TCB_OFF:_REPORTED_TCB_OFF + 8] = tcb
    der = vcek_priv.sign(bytes(report[:_SIG_OFF]), ec.ECDSA(hashes.SHA384()))
    r, s = decode_dss_signature(der)
    report[_SIG_OFF:_SIG_OFF + 72] = r.to_bytes(48, "little") + b"\x00" * 24
    report[_SIG_OFF + 72:_SIG_OFF + 144] = s.to_bytes(48, "little") + b"\x00" * 24
    chain_pem = (vcek.public_bytes(serialization.Encoding.PEM)
                 + ask.public_bytes(serialization.Encoding.PEM))
    env = b"PRSMSNP1" + struct.pack("<I", 1184) + bytes(report) + chain_pem
    return env, ark.public_bytes(serialization.Encoding.PEM)


def test_no_policy_is_chain_only_unchanged():
    env, ark = _envelope(report_tcb=(0, 0, 0, 0), vcek_tcb=(0, 0, 0, 0))
    res = AMDSEVSNPBackend(trusted_root_pem=ark).verify(env)  # no tcb_policy
    assert res.vendor_verified is True


def test_meets_floor_and_binds_passes():
    env, ark = _envelope(report_tcb=(3, 0, 20, 209), vcek_tcb=(3, 0, 20, 209))
    res = AMDSEVSNPBackend(trusted_root_pem=ark,
                           tcb_policy=SnpTcbPolicy(3, 0, 20, 209)).verify(env)
    assert res.vendor_verified is True
    assert res.vendor_data.get("reported_tcb") == {"bootloader": 3, "tee": 0, "snp": 20, "microcode": 209}


def test_below_floor_is_rejected():
    env, ark = _envelope(report_tcb=(3, 0, 20, 208), vcek_tcb=(3, 0, 20, 208))
    res = AMDSEVSNPBackend(trusted_root_pem=ark,
                           tcb_policy=SnpTcbPolicy(3, 0, 20, 209)).verify(env)
    assert res.vendor_verified is False
    assert "tcb" in (res.error or "").lower()


def test_binding_mismatch_is_rejected():
    """Report claims a higher TCB than the VCEK was issued for → spoof → rejected,
    even though the floor would be met."""
    env, ark = _envelope(report_tcb=(3, 0, 20, 209), vcek_tcb=(3, 0, 20, 100))
    res = AMDSEVSNPBackend(trusted_root_pem=ark,
                           tcb_policy=SnpTcbPolicy(0, 0, 0, 0)).verify(env)
    assert res.vendor_verified is False
    assert "tcb" in (res.error or "").lower()


def test_malformed_min_tcb_env_refuses_to_build_backend():
    """sp1065 review #5 fail-closed — a malformed PRSM_AMD_SEV_SNP_MIN_TCB must NOT
    build a real backend with enforcement silently off (a downlevel TEE would pass);
    it returns None → structural fallback (vendor_verified=false)."""
    from prsm.compute.inference.amd_sev_snp import build_amd_sev_snp_backend_or_none
    _, ark = _envelope(report_tcb=(0, 0, 0, 0), vcek_tcb=(0, 0, 0, 0))
    env = {"PRSM_AMD_SEV_SNP_ARK_PEM": ark.decode(),
           "PRSM_AMD_SEV_SNP_MIN_TCB": "3,0,20"}  # 3 fields, malformed
    assert build_amd_sev_snp_backend_or_none(env) is None


def test_valid_min_tcb_env_builds_enforcing_backend():
    from prsm.compute.inference.amd_sev_snp import build_amd_sev_snp_backend_or_none
    _, ark = _envelope(report_tcb=(0, 0, 0, 0), vcek_tcb=(0, 0, 0, 0))
    env = {"PRSM_AMD_SEV_SNP_ARK_PEM": ark.decode(),
           "PRSM_AMD_SEV_SNP_MIN_TCB": "3,0,20,209"}
    backend = build_amd_sev_snp_backend_or_none(env)
    assert backend is not None and backend._tcb_policy is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
