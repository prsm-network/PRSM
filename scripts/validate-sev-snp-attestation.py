#!/usr/bin/env python3
"""Sprint 1296 — on-VM validation harness for AMD SEV-SNP quote generation (roadmap E).

Run this ON a real AMD SEV-SNP confidential VM (with /dev/sev-guest). It generates a REAL
attestation quote via SevSnpTEERuntime (SNP_GET_REPORT ioctl + AMD KDS VCEK/ASK fetch) and
verifies it end-to-end through the PRODUCTION AMDSEVSNPBackend (bundled, fingerprint-pinned
AMD ARK roots — no config needed), confirming vendor_verified=True and the sp1083 node
binding (REPORT_DATA[:32] == sha256(node_id)).

This is the validation that flips roadmap E from "code-complete" to "hardware-validated."

Usage (on the VM, from the repo root):
    PRSM_NODE_ID=prsm-sev-snp-rehearsal python3 scripts/validate-sev-snp-attestation.py
Optional: PRSM_SEV_SNP_PRODUCT=Genoa  (default Milan; set per the VM's EPYC generation)

Exit codes: 0 = E validated; 1 no device; 2 quote-gen error; 3 backend off; 4 not verified.
"""
import hashlib
import os
import sys

NODE_ID = os.environ.get("PRSM_NODE_ID", "prsm-sev-snp-rehearsal")


def main() -> int:
    from prsm.compute.tee.runtime import SevSnpTEERuntime
    from prsm.compute.inference.amd_sev_snp import build_amd_sev_snp_backend_or_none

    rt = SevSnpTEERuntime(node_id=NODE_ID)
    print(f"node_id:                 {NODE_ID}")
    print(f"/dev/sev-guest present:  {rt.available}")
    if not rt.available:
        print("\n❌ FAIL: no SEV-SNP guest device — run this on a real AMD SEV-SNP confidential VM.")
        return 1
    print(f"KDS product:             {rt._kds_product()}  (override via PRSM_SEV_SNP_PRODUCT)")

    print("\nGenerating quote (SNP_GET_REPORT ioctl + AMD KDS fetch)…")
    try:
        quote = rt.get_attestation_bytes()
    except Exception as exc:  # noqa: BLE001 — report any hardware/network failure clearly
        print(f"❌ FAIL: quote generation errored: {type(exc).__name__}: {exc}")
        print("   (likely the ioctl struct packing or the KDS product/URL — paste this and I'll fix it.)")
        return 2
    print(f"  quote: {len(quote)} bytes, magic={quote[:8]!r}")

    backend = build_amd_sev_snp_backend_or_none()
    if backend is None:
        print("❌ FAIL: AMD backend not built (is PRSM_AMD_SEV_SNP_USE_BUNDLED_ROOT=0?).")
        return 3

    res = backend.verify(quote)
    print("\nVerification (production AMDSEVSNPBackend, bundled AMD ARK roots):")
    print(f"  vendor:              {res.vendor}")
    print(f"  structural_parse_ok: {res.structural_parse_ok}")
    print(f"  signature_chain_ok:  {res.signature_chain_ok}")
    print(f"  vendor_verified:     {res.vendor_verified}")
    print(f"  error:               {res.error}")

    report_data = quote[12 + 0x50: 12 + 0x50 + 32]
    binding_ok = report_data == hashlib.sha256(NODE_ID.encode()).digest()
    print(f"  node binding ok:     {binding_ok}  (REPORT_DATA[:32] == sha256(node_id))")

    if res.vendor_verified and binding_ok:
        print("\n✅ E VALIDATED — real SEV-SNP quote generated, verified to the genuine AMD ARK, "
              "and node-bound. vendor_verified=True.")
        return 0
    print("\n❌ NOT validated — see the fields above.")
    return 4


if __name__ == "__main__":
    sys.exit(main())
