"""Sprint 1237 (TEE Tier-3 roadmap B) — requester-side receipt verification now
reflects the REAL multi-stage attestation verdict.

verify_receipt_privacy_claim() derived hardware_attested from a permissive
heuristic ("tee_attestation non-empty AND not DEV-ONLY") and detected the
multi-stage envelope only structurally. sp1236 made verify_stage_attestations
able to cryptographically verify + node-bind each stage; this sprint threads an
opt-in attestation_registry into the receipt check so a requester with the real
backends gets the crypto truth: multi_stage_verified (per-stage vendor_verified +
REPORT_DATA->node binding aggregate) drives hardware_attested. No registry → the
result is byte-identical to pre-sp1237 (multi_stage_verified=None, permissive).
"""

from decimal import Decimal

from prsm.compute.inference.attestation_backends import (
    AttestationVerificationResult,
    SOFTWARE_TEE_ATTESTATION_PREFIX,
    expected_attestation_report_data,
)
from prsm.compute.inference.models import ContentTier, InferenceReceipt
from prsm.compute.inference.multi_stage_attestation import (
    StageAttestation,
    encode_multi_stage_attestation,
)
from prsm.compute.inference.privacy_verification import verify_receipt_privacy_claim
from prsm.compute.tee.models import PrivacyLevel, TEEType


NODES = ["nodeA", "nodeB", "nodeC"]


def _attestation(node_id):
    return f"quote::{node_id}".encode()


def _envelope(node_ids):
    return encode_multi_stage_attestation([
        StageAttestation(stage_index=i, stage_node_id=nid, tee_type=TEEType.SGX,
                         attestation=_attestation(nid))
        for i, nid in enumerate(node_ids)
    ])


class _FakeRegistry:
    def __init__(self, mapping):
        self._mapping = mapping

    def verify(self, blob):
        return self._mapping.get(
            bytes(blob), AttestationVerificationResult(vendor="unknown"))


def _good_registry(node_ids):
    return _FakeRegistry({
        _attestation(nid): AttestationVerificationResult(
            vendor="intel-sgx", vendor_verified=True,
            vendor_data={"report_data_hex":
                         expected_attestation_report_data(nid) + "00" * 32})
        for nid in node_ids
    })


def _receipt(attestation_bytes):
    return InferenceReceipt(
        job_id="j1", request_id="r1", model_id="m1",
        content_tier=ContentTier.A, privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0, tee_type=TEEType.SGX,
        tee_attestation=attestation_bytes, output_hash=b"\x00" * 32,
        duration_seconds=1.0, cost_ftns=Decimal("0.1"),
    )


# ── back-compat: no registry → structural-only, permissive ─────────

def test_no_registry_is_structural_only():
    r = verify_receipt_privacy_claim(_receipt(_envelope(NODES)))
    assert r.multi_stage_verified is None              # not run
    assert r.multi_stage_envelope_present is True
    assert r.hardware_attested is True                 # permissive (non-dev-only)


# ── with registry: real verdict drives hardware_attested ───────────

def test_registry_verifies_multistage_chain():
    r = verify_receipt_privacy_claim(
        _receipt(_envelope(NODES)), attestation_registry=_good_registry(NODES))
    assert r.multi_stage_verified is True
    assert r.hardware_attested is True


def test_registry_binding_failure_flips_hardware_attested():
    reg = _good_registry(NODES)
    # nodeB's quote carries the wrong node commitment (replayed) → binding fails
    reg._mapping[_attestation("nodeB")] = AttestationVerificationResult(
        vendor="intel-sgx", vendor_verified=True,
        vendor_data={"report_data_hex": expected_attestation_report_data("nodeA")})
    r = verify_receipt_privacy_claim(
        _receipt(_envelope(NODES)), attestation_registry=reg)
    assert r.multi_stage_verified is False
    assert r.hardware_attested is False                # real verdict, not permissive


def test_registry_crypto_failure_flips_hardware_attested():
    reg = _good_registry(NODES)
    reg._mapping[_attestation("nodeC")] = AttestationVerificationResult(
        vendor="intel-sgx", vendor_verified=False,
        vendor_data={"report_data_hex":
                     expected_attestation_report_data("nodeC") + "00" * 32})
    r = verify_receipt_privacy_claim(
        _receipt(_envelope(NODES)), attestation_registry=reg)
    assert r.multi_stage_verified is False
    assert r.hardware_attested is False


# ── require_hardware_attestation surfaces a reason on failure ───────

def test_require_hardware_adds_reason_on_verification_failure():
    reg = _good_registry(NODES)
    reg._mapping[_attestation("nodeB")] = AttestationVerificationResult(
        vendor="intel-sgx", vendor_verified=False, vendor_data={})
    r = verify_receipt_privacy_claim(
        _receipt(_envelope(NODES)),
        attestation_registry=reg, require_hardware_attestation=True)
    assert r.multi_stage_verified is False
    assert any("multi-stage attestation failed" in x for x in r.reasons)


# ── a DEV-ONLY (software) attestation is never multi-stage-verified ─

def test_dev_only_attestation_not_verified():
    dev = SOFTWARE_TEE_ATTESTATION_PREFIX + b"\x00" * 32
    r = verify_receipt_privacy_claim(
        _receipt(dev), attestation_registry=_good_registry(NODES))
    assert r.multi_stage_envelope_present is False
    assert r.multi_stage_verified is None
    assert r.hardware_attested is False                # dev-only never hardware


def test_to_dict_exposes_multi_stage_verified():
    r = verify_receipt_privacy_claim(
        _receipt(_envelope(NODES)), attestation_registry=_good_registry(NODES))
    assert r.to_dict()["multi_stage_verified"] is True
