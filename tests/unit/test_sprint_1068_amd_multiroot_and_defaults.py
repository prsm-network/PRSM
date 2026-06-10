"""Sprint 1068 — AMD multi-ARK selection + bundled-default wiring (Tier 3, brick 3).

AMD has per-product ARKs (Milan/Genoa/Turin), so AMDSEVSNPBackend holds a SET of
roots and selects the one whose subject issued the chain's ASK. The build function
defaults to the bundled (fingerprint-pinned) ARK set so a daemon does real SEV-SNP
verification without sourcing the PEMs; opt out with PRSM_AMD_SEV_SNP_USE_BUNDLED_ROOT=0.
"""
from __future__ import annotations

import datetime as _dt
import struct

import pytest
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

from prsm.compute.inference.amd_sev_snp import (
    AMDSEVSNPBackend, build_amd_sev_snp_backend_or_none)

_SIG_OFF = 0x2A0


def _mk(subj, issuer, subj_pub, issuer_priv, *, ca):
    return (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subj)]))
            .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer)]))
            .public_key(subj_pub).serial_number(x509.random_serial_number())
            .not_valid_before(_dt.datetime(2025, 1, 1)).not_valid_after(_dt.datetime(2035, 1, 1))
            .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
            .sign(issuer_priv, hashes.SHA384()))


def _envelope(ark_name="ARK-Test"):
    ark_priv = ec.generate_private_key(ec.SECP384R1())
    ask_priv = ec.generate_private_key(ec.SECP384R1())
    vcek_priv = ec.generate_private_key(ec.SECP384R1())
    ark = _mk(ark_name, ark_name, ark_priv.public_key(), ark_priv, ca=True)
    ask = _mk("ASK-Test", ark_name, ask_priv.public_key(), ark_priv, ca=True)
    vcek = _mk("VCEK", "ASK-Test", vcek_priv.public_key(), ask_priv, ca=False)
    report = bytearray(1184)
    struct.pack_into("<I", report, 0, 2)
    der = vcek_priv.sign(bytes(report[:_SIG_OFF]), ec.ECDSA(hashes.SHA384()))
    r, s = decode_dss_signature(der)
    report[_SIG_OFF:_SIG_OFF + 72] = r.to_bytes(48, "little") + b"\x00" * 24
    report[_SIG_OFF + 72:_SIG_OFF + 144] = s.to_bytes(48, "little") + b"\x00" * 24
    chain = vcek.public_bytes(serialization.Encoding.PEM) + ask.public_bytes(serialization.Encoding.PEM)
    env = b"PRSMSNP1" + struct.pack("<I", 1184) + bytes(report) + chain
    return env, ark.public_bytes(serialization.Encoding.PEM)


def test_multiroot_selects_matching_ark():
    env, real_ark = _envelope("ARK-Test")
    decoy_priv = ec.generate_private_key(ec.SECP384R1())
    decoy = _mk("ARK-Other", "ARK-Other", decoy_priv.public_key(), decoy_priv, ca=True
                ).public_bytes(serialization.Encoding.PEM)
    backend = AMDSEVSNPBackend(trusted_root_pems=[decoy, real_ark])
    res = backend.verify(env)
    assert res.vendor_verified is True


def test_multiroot_no_matching_ark_fails():
    """With MULTIPLE roots and none issuing the chain → 'no configured ARK' (a single
    root is used directly, so that case reports the chain-verify error instead)."""
    env, _ = _envelope("ARK-Test")
    decoys = []
    for nm in ("ARK-Other1", "ARK-Other2"):
        p = ec.generate_private_key(ec.SECP384R1())
        decoys.append(_mk(nm, nm, p.public_key(), p, ca=True).public_bytes(serialization.Encoding.PEM))
    res = AMDSEVSNPBackend(trusted_root_pems=decoys).verify(env)
    assert res.vendor_verified is False
    assert "no configured amd ark" in (res.error or "").lower()


def test_single_root_still_works():
    env, ark = _envelope("AMD ARK")
    res = AMDSEVSNPBackend(trusted_root_pem=ark).verify(env)
    assert res.vendor_verified is True


def test_requires_at_least_one_root():
    with pytest.raises(ValueError):
        AMDSEVSNPBackend()


def test_build_defaults_to_bundled_arks():
    backend = build_amd_sev_snp_backend_or_none(env={})
    assert isinstance(backend, AMDSEVSNPBackend)
    assert len(backend._roots) == 3   # Milan/Genoa/Turin


def test_build_none_when_bundled_opted_out():
    assert build_amd_sev_snp_backend_or_none(
        env={"PRSM_AMD_SEV_SNP_USE_BUNDLED_ROOT": "0"}) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
