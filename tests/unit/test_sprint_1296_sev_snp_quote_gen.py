"""Sprint 1296 — AMD SEV-SNP quote GENERATION (roadmap E, code half).

The verifier (AMDSEVSNPBackend) was already complete + tested (sp1049). sp1243 left the
generation side (SevSnpTEERuntime._generate_quote) raising NotImplementedError pending
hardware. This implements it: SNP_GET_REPORT ioctl → AMD KDS VCEK/ASK fetch → PRSMSNP1
envelope. The two impure primitives (the /dev/sev-guest ioctl + the KDS HTTP fetch) are
hardware/network-validation-pending on a real SEV-SNP confidential VM (sprint E); these
tests cover everything else by round-tripping a generated envelope back through the REAL
verifier, with the ioctl + HTTP mocked.
"""
from __future__ import annotations

import struct

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization

from prsm.compute.tee.runtime import SevSnpTEERuntime, TEEHardwareUnavailableError, _SNP_REPORT_LEN
from tests.unit.test_sprint_1049_amd_sev_snp_verify import build_sev_snp_envelope


def _split(env):
    """Pull (report, vcek_pem, ask_pem) back out of a built envelope."""
    body = env[12:12 + _SNP_REPORT_LEN]            # 8 magic + 4 len header
    certs = x509.load_pem_x509_certificates(env[12 + _SNP_REPORT_LEN:])
    vcek_pem = certs[0].public_bytes(serialization.Encoding.PEM)
    ask_pem = certs[1].public_bytes(serialization.Encoding.PEM)
    return body, vcek_pem, ask_pem, certs


def test_assemble_envelope_matches_verifier_format():
    """_assemble_envelope must produce the byte-exact shape the verifier accepts."""
    from prsm.compute.inference.amd_sev_snp import AMDSEVSNPBackend
    env, ark_pem, _ = build_sev_snp_envelope()
    report, vcek_pem, ask_pem, _ = _split(env)

    my_env = SevSnpTEERuntime._assemble_envelope(report, vcek_pem, ask_pem)
    assert my_env == env, "assembled envelope must be byte-identical to the canonical shape"

    res = AMDSEVSNPBackend(trusted_root_pem=ark_pem).verify(my_env)
    assert res.vendor_verified is True and res.error is None


def test_generate_quote_orchestration_roundtrips_through_verifier(monkeypatch):
    """The full _generate_quote pipeline (ioctl + KDS mocked) yields an envelope the real
    verifier accepts, with REPORT_DATA == the node binding."""
    from prsm.compute.inference.amd_sev_snp import AMDSEVSNPBackend

    rt = SevSnpTEERuntime(node_id="node-rehearsal-1")
    binding = rt.report_data()                      # sha256(node_id) + 32 zero bytes
    env, ark_pem, _ = build_sev_snp_envelope(report_data=binding)
    report, vcek_pem, ask_pem, certs = _split(env)
    vcek_der = certs[0].public_bytes(serialization.Encoding.DER)
    chain_resp = ask_pem + ark_pem                  # KDS cert_chain returns ASK then ARK

    monkeypatch.setattr(rt, "_snp_get_report", lambda rd: report)
    monkeypatch.setattr(
        SevSnpTEERuntime, "_http_get",
        staticmethod(lambda url: chain_resp if "cert_chain" in url else vcek_der),
    )

    out = rt._generate_quote(binding)
    assert out == env
    # the node binding is embedded at REPORT_DATA (report offset 0x50, after the 12-byte header)
    assert out[12 + 0x50:12 + 0x50 + 64] == binding
    res = AMDSEVSNPBackend(trusted_root_pem=ark_pem).verify(out)
    assert res.vendor_verified is True and res.error is None


def test_parse_reported_tcb():
    report = bytearray(_SNP_REPORT_LEN)
    report[0x180:0x188] = bytes([7, 0, 0, 0, 0, 0, 21, 209])  # bl=7, tee=0, snp=21, ucode=209
    tcb = SevSnpTEERuntime._parse_reported_tcb(bytes(report))
    assert tcb == {"blSPL": 7, "teeSPL": 0, "snpSPL": 21, "ucodeSPL": 209}


def test_kds_urls():
    chip = bytes(range(64))
    tcb = {"blSPL": 7, "teeSPL": 0, "snpSPL": 21, "ucodeSPL": 209}
    vurl = SevSnpTEERuntime._vcek_url("Milan", chip, tcb)
    assert vurl.startswith("https://kdsintf.amd.com/vcek/v1/Milan/")
    assert chip.hex() in vurl
    assert "blSPL=7&teeSPL=0&snpSPL=21&ucodeSPL=209" in vurl
    assert SevSnpTEERuntime._chain_url("Milan") == "https://kdsintf.amd.com/vcek/v1/Milan/cert_chain"


def test_kds_product_env_override(monkeypatch):
    monkeypatch.delenv("PRSM_SEV_SNP_PRODUCT", raising=False)
    assert SevSnpTEERuntime._kds_product() == "Milan"
    monkeypatch.setenv("PRSM_SEV_SNP_PRODUCT", "Genoa")
    assert SevSnpTEERuntime._kds_product() == "Genoa"


@pytest.mark.skipif(
    any(__import__("os").path.exists(p) for p in SevSnpTEERuntime._DEVICE_PATHS),
    reason="a real SEV-SNP device is present — the gate would attempt a real ioctl",
)
def test_device_gate_raises_without_hardware():
    """No /dev/sev-guest → get_attestation_bytes fails loudly, never a dev-only blob."""
    rt = SevSnpTEERuntime(node_id="n")
    assert rt.available is False
    with pytest.raises(TEEHardwareUnavailableError):
        rt.get_attestation_bytes()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
