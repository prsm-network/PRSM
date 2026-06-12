# Attestation golden-vector fixtures

Real, **public** Intel SGX / AMD SEV-SNP trust anchors and signed collateral, used by
`tests/unit/test_sprint_1066_attestation_golden_vectors.py` to validate the Tier-3
attestation parsers/validators (sp1044–1065) against production data rather than only
self-built fixtures.

All files here are public trust material (root/intermediate CAs, signed TCB-Info, a
CRL) — **no secrets, no platform-identifying leaf certs**. Safe for an external audit.

| File | What | Source (fetched 2026-06-10) |
|------|------|------|
| `intel_tcb_info_00906ed50000.json` | Real signed Intel SGX TCB-Info (v3, fmspc `00906ed50000`, 21 tcbLevels) | `https://api.trustedservices.intel.com/sgx/certification/v4/tcb?fmspc=00906ed50000` |
| `intel_qe_identity.json` | Real signed Intel SGX QE Identity (`enclaveIdentity` v2, mrsigner/isvprodid/6 tcbLevels) — signed by the same TCB Signing cert | `https://api.trustedservices.intel.com/sgx/certification/v4/qe/identity` |
| `intel_tcb_signing.pem` | Intel SGX TCB Signing cert (ECDSA-P256) — signs the TCB-Info | `TCB-Info-Issuer-Chain` response header (above), first cert |
| `intel_sgx_root_ca.pem` | Intel SGX Root CA | `SGX-PCK-CRL-Issuer-Chain` header, root |
| `intel_sgx_platform_ca.pem` | Intel SGX PCK Platform CA | same issuer chain, intermediate |
| `intel_pck_platform.crl.pem` | Intel PCK Platform CRL (real revocations) | `…/sgx/certification/v4/pckcrl?ca=platform&encoding=pem` |
| `amd_milan_ask_ark.pem` | AMD Milan ASK + ARK chain (RSA-4096 / RSA-PSS) | `https://kdsintf.amd.com/vcek/v1/Milan/cert_chain` |

## What they validate

- **TCB-Info parse + signature** (`parse_and_verify_tcb_info`, sp1062): the real signed
  blob verifies against the real Intel TCB Signing cert; tampering breaks it.
- **Chain validation** (`verify_cert_chain`, sp1049): the real Intel PCK chain
  (Platform CA → Root CA, ECDSA) AND the real AMD chain (ASK → ARK, RSA-PSS).
- **CRL parsing** (sp1060): the real Intel PCK Platform CRL is accepted as
  signature-valid under the Platform CA.
- **sp1066 fix**: the AMD chain caught that the validator hardcoded `ec.ECDSA` and
  would reject real RSA-signed AMD certs; it now dispatches on issuer key type.

## Still TODO (narrower remaining caveat)

A real **leaf** cert with the vendor TCB extensions (an Intel PCK leaf with the SGX
extension, an AMD VCEK with the SPL extensions) requires platform secrets / a real
chip HWID and is not fetched here. The leaf-extension parsers
(`parse_pck_sgx_extension`, `parse_vcek_tcb`) remain validated only against
self-built fixtures — add a real-leaf golden vector to retire that last piece.

## Refreshing

The CRL and TCB-Info carry `nextUpdate` (~monthly). The test pins `now` inside the
committed fixtures' validity windows, so it stays deterministic and does NOT rot.
Re-fetch only if you want newer collateral; update the pinned `now` if you do.
