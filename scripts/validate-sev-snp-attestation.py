#!/usr/bin/env python3
"""SEV-SNP + TEE Tier-B end-to-end validation harness (roadmap E → Tier-B).

Run this ON a real AMD SEV-SNP confidential VM (with /dev/sev-guest). It exercises the WHOLE
Tier-B code path deterministically from a real hardware quote — everything short of the funded
on-chain settlement tx (which is the separate go-live ceremony with a funded settler key):

  Phase 0 (sp1333 producer)  build_tee_runtime(PRSM_TEE_RUNTIME_KIND=sev-snp) selects the
                             hardware runtime, node-bound (the selector a real operator flips).
  Phase 1 (roadmap E)        SevSnpTEERuntime.get_attestation_bytes() → a REAL quote
                             (SNP_GET_REPORT ioctl + AMD KDS VCEK/ASK fetch).
  Phase 2 (roadmap E)        the production AMDSEVSNPBackend verifies it to the bundled,
                             fingerprint-pinned AMD ARK roots → vendor_verified=True + the
                             sp1083 node binding (REPORT_DATA[:32] == sha256(node_id)).
  Phase 3 (sp1334 binding)   a StageActivationProof binds sha256(real quote) into its
                             signature; the sig verifies, and SWAPPING the attestation breaks
                             it (the anti-swap property, proven with a real quote).
  Phase 4 (sp1241 roadmap F) the real quote → tee_attestation_audit_dict → the
                             batch_attestation_commitment bytes32 that commitBatchWithAttestation
                             would carry on-chain (NON-zero + not dev-only = a real Tier-B batch).

If every phase is green, Tier-B is hardware-validated end-to-end in code; the only remaining
step is committing that commitment on-chain via the funded-settler go-live ceremony (sp1301).

Usage (on the VM, from the repo root):
    PRSM_NODE_ID=prsm-sev-snp-rehearsal python3 scripts/validate-sev-snp-attestation.py
Optional: PRSM_SEV_SNP_PRODUCT=Genoa   (default Milan; set per the VM's EPYC generation)

Exit codes: 0 = Tier-B validated · 1 no device · 2 quote-gen error · 3 backend off ·
            4 a validation phase failed.
"""
import dataclasses
import hashlib
import os
import sys

NODE_ID = os.environ.get("PRSM_NODE_ID", "prsm-sev-snp-rehearsal")
_BIND_ON = {"PRSM_STAGE_PROOF_BIND_ATTESTATION": "1"}


def _ok(label, passed, detail=""):
    mark = "✅" if passed else "❌"
    print(f"  {mark} {label}" + (f"  ({detail})" if detail else ""))
    return passed


def main() -> int:
    from prsm.compute.tee.runtime import SevSnpTEERuntime, build_tee_runtime
    from prsm.compute.tee.models import TEEType
    from prsm.compute.inference.amd_sev_snp import build_amd_sev_snp_backend_or_none

    results = {}

    # ── Phase 0 — sp1333 producer selector ────────────────────────────────────
    print("Phase 0 — runtime selector (sp1333, PRSM_TEE_RUNTIME_KIND=sev-snp):")
    probe = SevSnpTEERuntime(node_id=NODE_ID)
    print(f"  node_id:                {NODE_ID}")
    print(f"  /dev/sev-guest present: {probe.available}")
    if not probe.available:
        print("\n❌ FAIL: no SEV-SNP guest device — run this on a real AMD SEV-SNP confidential VM.")
        return 1
    rt = build_tee_runtime(node_id=NODE_ID, environ={"PRSM_TEE_RUNTIME_KIND": "sev-snp"})
    results["selector picks SevSnpTEERuntime"] = _ok(
        "selector returns the hardware runtime", isinstance(rt, SevSnpTEERuntime),
        f"{type(rt).__name__}, tee_type={getattr(rt.tee_type, 'value', rt.tee_type)}")
    results["selector node-bound"] = _ok(
        "selected runtime is node-bound", rt.report_data()[:32] == hashlib.sha256(NODE_ID.encode()).digest(),
        "REPORT_DATA[:32] == sha256(node_id)")
    print(f"  KDS product:            {rt._kds_product()}  (override via PRSM_SEV_SNP_PRODUCT)")

    # ── Phase 1 — real quote generation (roadmap E) ───────────────────────────
    print("\nPhase 1 — real quote generation (SNP_GET_REPORT ioctl + AMD KDS fetch):")
    try:
        quote = rt.get_attestation_bytes()
    except Exception as exc:  # noqa: BLE001 — report any hardware/network failure clearly
        print(f"  ❌ FAIL: quote generation errored: {type(exc).__name__}: {exc}")
        print("     (likely the ioctl struct packing or the KDS product/URL — paste this and I'll fix it.)")
        return 2
    results["real quote generated"] = _ok(
        "quote generated", bool(quote), f"{len(quote)} bytes, magic={quote[:8]!r}")

    # ── Phase 2 — production backend verification (roadmap E) ──────────────────
    print("\nPhase 2 — verification (production AMDSEVSNPBackend, bundled AMD ARK roots):")
    backend = build_amd_sev_snp_backend_or_none()
    if backend is None:
        print("  ❌ FAIL: AMD backend not built (is PRSM_AMD_SEV_SNP_USE_BUNDLED_ROOT=0?).")
        return 3
    res = backend.verify(quote)
    print(f"     vendor={res.vendor} structural_parse_ok={res.structural_parse_ok} "
          f"signature_chain_ok={res.signature_chain_ok} error={res.error}")
    results["vendor_verified"] = _ok("vendor_verified (chains to genuine AMD ARK)", bool(res.vendor_verified))
    report_data = quote[12 + 0x50: 12 + 0x50 + 32]
    results["node binding"] = _ok(
        "node binding", report_data == hashlib.sha256(NODE_ID.encode()).digest(),
        "REPORT_DATA[:32] == sha256(node_id)")

    # ── Phase 3 — stage-proof attestation binding (sp1334) ─────────────────────
    print("\nPhase 3 — stage-proof binding (sp1334, PRSM_STAGE_PROOF_BIND_ATTESTATION=1):")
    from prsm.node.identity import generate_node_identity
    from prsm.compute.inference.stage_activation_proof import (
        StageActivationProof, activation_hash, tee_attestation_hash_for, verify_stage_proof,
    )

    class _Anchor:
        def __init__(self, m):
            self._m = m

        def lookup(self, nid):
            return self._m.get(nid)

    idn = generate_node_identity(display_name=NODE_ID)
    req_id = f"tierb-{NODE_ID}"
    att_hash = tee_attestation_hash_for(quote, environ=_BIND_ON)
    proof = StageActivationProof(
        stage_index=0, stage_node_id=idn.node_id,
        input_activation_hash=activation_hash(b"in"), output_activation_hash=activation_hash(b"out"),
        tee_attestation_hash=att_hash)
    proof = dataclasses.replace(proof, stage_signature_b64=idn.sign(proof.signing_bytes(req_id)))
    anchor = _Anchor({idn.node_id: idn.public_key_b64})
    results["binding non-empty"] = _ok(
        "the real quote produced a non-empty binding hash", bool(att_hash), att_hash[:16] + "…")
    results["bound proof verifies"] = _ok(
        "bound proof verifies", verify_stage_proof(proof, req_id, anchor=anchor) is True)
    swapped = dataclasses.replace(proof, tee_attestation_hash=activation_hash(quote + b"\x00"))
    results["swap breaks signature"] = _ok(
        "swapping the attestation breaks the signature",
        verify_stage_proof(swapped, req_id, anchor=anchor) is False)

    # ── Phase 4 — F off-chain on-chain commitment (sp1241) ─────────────────────
    print("\nPhase 4 — F on-chain commitment (sp1241 batch_attestation_commitment):")
    from prsm.compute.shard_receipt import (
        tee_attestation_audit_dict, batch_attestation_commitment,
    )
    audit = tee_attestation_audit_dict(quote, TEEType.SEV)
    results["audit not dev-only"] = _ok(
        "the real quote is recognized as NOT dev-only",
        bool(audit) and audit.get("is_dev_only") is False,
        f"tee_type={audit.get('tee_type') if audit else None}")
    commitment = batch_attestation_commitment([audit])
    nonzero = commitment != b"\x00" * 32
    results["commitment non-zero"] = _ok(
        "commitBatchWithAttestation commitment is non-zero (real Tier-B batch)", nonzero,
        "0x" + commitment.hex())

    # ── Summary ────────────────────────────────────────────────────────────────
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n{'='*72}\nTier-B validation: {passed}/{total} checks passed")
    if passed == total:
        print("✅ TIER-B HARDWARE-VALIDATED end-to-end in code — real SEV-SNP quote → backend-"
              "verified + node-bound → stage-proof-bound (swap-proof) → non-zero on-chain "
              "commitment.\n   Remaining: commit it on-chain via the funded-settler go-live "
              "ceremony (python -m prsm.settlement.go_live_preflight, then the Tier-B canary).")
        return 0
    print("❌ NOT fully validated — see the ❌ lines above.")
    return 4


if __name__ == "__main__":
    sys.exit(main())
