"""Sprint 1115 — Domain-08 review MEDIUM-7 follow-on: wire TEEPolicy into the
federated-learning gradient-update gate.

The FL path checked signatures (sp308a) but never evaluated the worker's
worker_attestation_b64 against any TEE policy, despite the module docstring claiming TEE
gating. A FederatedJob can now declare a tee_policy; accept_gradient_update refuses any
update whose worker attestation doesn't satisfy it (fail-closed).
"""
from __future__ import annotations

import base64

import pytest

from prsm.compute.inference.executor import SOFTWARE_TEE_ATTESTATION_PREFIX
from prsm.enterprise.federated_learning import (
    AggregationStrategy,
    FederatedLearningOrchestrator,
    GradientUpdate,
    encode_gradient,
)


def _software_attestation_b64() -> str:
    return base64.b64encode(SOFTWARE_TEE_ATTESTATION_PREFIX + b"\x00" * 32).decode()


def _orch():
    return FederatedLearningOrchestrator()


def _propose(orch, *, tee_policy=None, workers=("n1", "n2")):
    return orch.propose_job(
        model_id="m", dataset_cids=["cidA"], worker_pool=list(workers),
        rounds_target=1, min_workers_per_round=1,
        aggregation=AggregationStrategy.FEDAVG, tee_policy=tee_policy)


def _update(job_id, node, *, attestation_b64=""):
    return GradientUpdate(
        job_id=job_id, round_index=0, worker_node_id=node,
        gradient_b64=base64.b64encode(encode_gradient([1.0, 2.0])).decode(),
        sample_count=10, worker_attestation_b64=attestation_b64,
        worker_signature_b64="", timestamp=100.0)


def _assigned_node(rnd):
    return rnd.worker_assignments[0].node_id


# ── policy validation at propose time ───────────────────────────────────────────────

def test_propose_job_rejects_malformed_tee_policy():
    orch = _orch()
    with pytest.raises(ValueError, match="invalid tee_policy"):
        _propose(orch, tee_policy={"min_attestation_tier": "bogus-tier"})


def test_tee_policy_round_trips_on_job():
    orch = _orch()
    job = _propose(orch, tee_policy={"min_attestation_tier": "software"})
    from prsm.enterprise.federated_learning import FederatedJob
    rebuilt = FederatedJob.from_dict(job.to_dict())
    assert rebuilt.tee_policy == {"min_attestation_tier": "software"}


# ── enforcement at accept_gradient_update ─────────────────────────────────────────────

def test_no_policy_accepts_update_without_attestation():
    """Backwards-compatible: no tee_policy → attestation not gated."""
    orch = _orch()
    job = _propose(orch)
    rnd = orch.issue_round(job.job_id)
    node = _assigned_node(rnd)
    orch.accept_gradient_update(_update(job.job_id, node))  # must not raise
    assert len(orch._rounds[(job.job_id, 0)].gradient_updates_received) == 1


def test_software_policy_accepts_software_attestation():
    orch = _orch()
    job = _propose(orch, tee_policy={"min_attestation_tier": "software"})
    rnd = orch.issue_round(job.job_id)
    node = _assigned_node(rnd)
    orch.accept_gradient_update(
        _update(job.job_id, node, attestation_b64=_software_attestation_b64()))
    assert len(orch._rounds[(job.job_id, 0)].gradient_updates_received) == 1


def test_software_policy_refuses_missing_attestation():
    orch = _orch()
    job = _propose(orch, tee_policy={"min_attestation_tier": "software"})
    rnd = orch.issue_round(job.job_id)
    node = _assigned_node(rnd)
    with pytest.raises(ValueError, match="TEE policy"):
        orch.accept_gradient_update(_update(job.job_id, node, attestation_b64=""))


def test_hardware_policy_refuses_software_attestation():
    """A job requiring real hardware must REFUSE a worker presenting only a software
    fallback attestation — the gate is fail-closed."""
    orch = _orch()
    job = _propose(orch, tee_policy={"min_attestation_tier": "hardware_verified"})
    rnd = orch.issue_round(job.job_id)
    node = _assigned_node(rnd)
    with pytest.raises(ValueError, match="TEE policy"):
        orch.accept_gradient_update(
            _update(job.job_id, node, attestation_b64=_software_attestation_b64()))


def test_bad_base64_attestation_fails_closed():
    orch = _orch()
    job = _propose(orch, tee_policy={"min_attestation_tier": "software"})
    rnd = orch.issue_round(job.job_id)
    node = _assigned_node(rnd)
    with pytest.raises(ValueError, match="TEE policy"):
        orch.accept_gradient_update(
            _update(job.job_id, node, attestation_b64="!!!not-base64!!!"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
