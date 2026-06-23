"""Sprint 1236 — wire real per-stage attestation verification + node-binding into
the multi-stage §7 chain verifier (TEE Tier-3 groundwork).

PRSM's proven production inference path is the cross-host MULTI-STAGE chain, but
verify_stage_attestations() hardcoded vendor_verified=None ("v2 wires platform-
vendor RPC here") — it never ran per-stage attestations through the real
IntelDCAPBackend/AMDSEVSNPBackend verifiers, and the sp1083 REPORT_DATA->node
binding that protects single-stage quotes was absent. So the verifiers existed
but were unreachable from the chain that matters.

This sprint makes verify_stage_attestations accept an opt-in registry +
expected_node_ids + expected_stage_count and, per stage: (a) verify the
attestation, (b) bind the verified quote's REPORT_DATA to the stage_node_id
(anti-replay), (c) match the stage to the expected topology (anti-substitution).
Default (no registry) stays byte-identical to pre-sp1236.

Tests use a FAKE registry (the real crypto is covered by test_sprint_1044/1049);
this isolates the new multi-stage WIRING.
"""

from prsm.compute.inference.attestation_backends import (
    AttestationVerificationResult,
    expected_attestation_report_data,
)
from prsm.compute.inference.multi_stage_attestation import (
    StageAttestation,
    encode_multi_stage_attestation,
    verify_stage_attestations,
)
from prsm.compute.tee.models import TEEType


def _report_data_hex(node_id: str) -> str:
    # full 64-byte REPORT_DATA hex: [:32]=sha256(node_id) commitment, rest pad
    return expected_attestation_report_data(node_id) + "00" * 32


class _FakeRegistry:
    """Maps a stage's attestation bytes -> AttestationVerificationResult."""

    def __init__(self, mapping):
        self._mapping = mapping

    def verify(self, blob):
        return self._mapping.get(
            bytes(blob), AttestationVerificationResult(vendor="unknown"))


def _attestation(node_id: str) -> bytes:
    return f"quote::{node_id}".encode()


def _envelope(node_ids, tee=TEEType.SGX):
    stages = [
        StageAttestation(stage_index=i, stage_node_id=nid, tee_type=tee,
                         attestation=_attestation(nid))
        for i, nid in enumerate(node_ids)
    ]
    return encode_multi_stage_attestation(stages)


def _good_registry(node_ids, *, vendor="intel-sgx"):
    """A registry that verifies + binds every node's quote correctly."""
    return _FakeRegistry({
        _attestation(nid): AttestationVerificationResult(
            vendor=vendor, vendor_verified=True,
            vendor_data={"report_data_hex": _report_data_hex(nid)})
        for nid in node_ids
    })


NODES = ["nodeA", "nodeB", "nodeC"]


# ── back-compat: no registry → structural-only, byte-identical ──────

def test_no_registry_is_structural_only():
    blob = _envelope(NODES)
    ok, results = verify_stage_attestations(blob)
    assert ok is True
    assert len(results) == 3
    assert all(r.vendor_verified is None for r in results)   # unchanged
    assert all(r.structurally_ok for r in results)


# ── happy path: every stage verified + bound ────────────────────────

def test_all_stages_verified_and_bound():
    blob = _envelope(NODES)
    ok, results = verify_stage_attestations(
        blob, registry=_good_registry(NODES), expected_node_ids=NODES)
    assert ok is True
    assert [r.vendor_verified for r in results] == [True, True, True]


def test_verified_without_expected_node_ids_still_binds():
    # expected_node_ids omitted → anti-substitution skipped, but per-stage
    # report_data binding (anti-replay) still enforced.
    blob = _envelope(NODES)
    ok, results = verify_stage_attestations(blob, registry=_good_registry(NODES))
    assert ok is True
    assert all(r.vendor_verified for r in results)


# ── anti-replay: a verified quote bound to the WRONG node fails ─────

def test_report_data_binding_failure_fails_chain():
    blob = _envelope(NODES)
    # registry returns vendor_verified=True but nodeB's quote carries nodeA's
    # commitment (replayed) → binding must fail for nodeB, chain not ok.
    reg = _good_registry(NODES)
    reg._mapping[_attestation("nodeB")] = AttestationVerificationResult(
        vendor="intel-sgx", vendor_verified=True,
        vendor_data={"report_data_hex": _report_data_hex("nodeA")})  # wrong node
    ok, results = verify_stage_attestations(blob, registry=reg, expected_node_ids=NODES)
    assert ok is False
    assert results[1].vendor_verified is True       # crypto passed
    assert "report_data_bound=False" in results[1].message  # but binding failed
    assert results[0].vendor_verified is True and results[2].vendor_verified is True


# ── crypto failure: one stage not vendor_verified fails chain ──────

def test_verified_but_missing_report_data_fails_binding():
    """Contract: a backend returning vendor_verified=True MUST populate
    report_data_hex; if it's missing/empty, the stage fails the binding
    (fail-closed) so an unbound 'verified' quote can't pass."""
    blob = _envelope(NODES)
    reg = _good_registry(NODES)
    reg._mapping[_attestation("nodeA")] = AttestationVerificationResult(
        vendor="intel-sgx", vendor_verified=True, vendor_data={})  # no report_data_hex
    ok, results = verify_stage_attestations(blob, registry=reg, expected_node_ids=NODES)
    assert ok is False
    assert results[0].vendor_verified is True
    assert "report_data_bound=False" in results[0].message


def test_one_stage_crypto_failure_fails_chain():
    blob = _envelope(NODES)
    reg = _good_registry(NODES)
    reg._mapping[_attestation("nodeC")] = AttestationVerificationResult(
        vendor="intel-sgx", vendor_verified=False,
        vendor_data={"report_data_hex": _report_data_hex("nodeC")})
    ok, results = verify_stage_attestations(blob, registry=reg, expected_node_ids=NODES)
    assert ok is False
    assert results[2].vendor_verified is False


# ── anti-substitution: stage relabeled vs expected topology ────────

def test_anti_substitution_expected_node_mismatch_fails():
    blob = _envelope(NODES)  # envelope claims nodeA/B/C
    ok, results = verify_stage_attestations(
        blob, registry=_good_registry(NODES),
        expected_node_ids=["nodeA", "EVIL", "nodeC"])  # caller expected EVIL at idx1
    assert ok is False
    assert "node_ok=False" in results[1].message


# ── anti-omission: stage count mismatch rejected ───────────────────

def test_expected_stage_count_mismatch_rejected():
    blob = _envelope(NODES)  # 3 stages
    ok, results = verify_stage_attestations(
        blob, registry=_good_registry(NODES), expected_stage_count=4)
    assert ok is False
    assert results[0].structurally_ok is False  # decode rejected the count


# ── a misbehaving backend must not crash the chain verify ──────────

def test_backend_exception_is_contained():
    blob = _envelope(NODES)

    class _Boom:
        def verify(self, blob):
            raise RuntimeError("backend exploded")

    ok, results = verify_stage_attestations(blob, registry=_Boom(), expected_node_ids=NODES)
    assert ok is False
    assert all(r.vendor_verified is False for r in results)
    assert "verify-error:RuntimeError" in results[0].message


# ── a software (unverified) stage drags the chain to not-ok ────────

def test_software_stage_in_otherwise_verified_chain():
    blob = _envelope(NODES)
    reg = _good_registry(NODES)
    # nodeB is a software/dev stage the registry can't vendor-verify
    reg._mapping[_attestation("nodeB")] = AttestationVerificationResult(
        vendor="software-fallback", vendor_verified=False, vendor_data={})
    ok, results = verify_stage_attestations(blob, registry=reg, expected_node_ids=NODES)
    assert ok is False
    assert results[1].vendor_verified is False
