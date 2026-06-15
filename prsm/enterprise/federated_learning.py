"""Sprint 308 — federated-learning orchestrator.

Vision §7 Enterprise Confidentiality Mode capstone: trains
a shared model across a fleet of TEE-attested PRSM worker
nodes that never see plaintext data. The enterprise:

  1. Recipient-encrypts the training data to the worker
     fleet (sprint 304 OR-decrypt or sprint 307 t-of-n
     threshold)
  2. Proposes a FederatedJob declaring the model,
     dataset CIDs, worker pool, rounds target, min-quorum,
     aggregation strategy
  3. The orchestrator issues each round to a subset of
     the worker pool with assignments
  4. Workers train locally INSIDE their TEE (gated by
     sprint 305a TEE policy + sprint 306a $CORP
     capability) on the decrypted shards and return ONLY
     gradient updates
  5. Orchestrator aggregates gradients (FedAvg or
     FedMedian) and surfaces the combined update
  6. Enterprise applies the update to their own model
     copy locally and proposes the next round

PRSM never holds the trained model — only the aggregated
gradient updates flow through. The plaintext data never
leaves a TEE.

This sprint ships the orchestration layer + aggregation
math + lifecycle + filesystem persistence. Worker-side
/compute/train integration is deferred to 308a.

Aggregation:
  FEDAVG     — weighted average by sample_count
  FEDMEDIAN  — element-wise median (Byzantine-robust;
               single outlier cannot swing the result)

Gradient encoding: little-endian packed float32 array,
base64-encoded for JSON transport. Pure Python; no numpy
dependency.
"""
from __future__ import annotations

import base64
import json
import logging
import math
import os
import random
import secrets
import statistics
import struct
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import (
    ChaCha20Poly1305,
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────


class AggregationStrategy(str, Enum):
    FEDAVG = "fedavg"
    FEDMEDIAN = "fedmedian"


class JobStatus(str, Enum):
    PROPOSED = "proposed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RoundStatus(str, Enum):
    ISSUED = "issued"
    COLLECTING = "collecting"
    AGGREGATED = "aggregated"
    FAILED = "failed"


_TERMINAL_JOB_STATUSES = {
    JobStatus.COMPLETED, JobStatus.FAILED,
    JobStatus.CANCELLED,
}


# ── Gradient encoding ───────────────────────────────


def encode_gradient(values: List[float]) -> bytes:
    """Pack a list of floats into little-endian float32
    bytes. Stable wire format for cross-language
    consumers."""
    if not values:
        return b""
    return struct.pack(f"<{len(values)}f", *values)


def decode_gradient(blob: bytes) -> List[float]:
    if not blob:
        return []
    if len(blob) % 4 != 0:
        raise ValueError(
            f"gradient blob length {len(blob)} not "
            f"divisible by 4 (float32 size)"
        )
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


# ── Dataclasses ──────────────────────────────────────


@dataclass
class WorkerAssignment:
    node_id: str
    dataset_cid: str
    assigned_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "dataset_cid": self.dataset_cid,
            "assigned_at": float(self.assigned_at),
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "WorkerAssignment":
        return cls(
            node_id=d["node_id"],
            dataset_cid=d["dataset_cid"],
            assigned_at=float(d["assigned_at"]),
        )


@dataclass
class GradientUpdate:
    job_id: str
    round_index: int
    worker_node_id: str
    gradient_b64: str
    sample_count: int
    worker_attestation_b64: str
    worker_signature_b64: str
    timestamp: float
    # Sprint 308c — encrypted-gradient transport. When
    # non-None, gradient_b64 is sealed ciphertext + this
    # field holds the seal envelope (ephemeral pubkey +
    # nonce). When None, gradient_b64 is plaintext.
    gradient_envelope_b64: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "round_index": int(self.round_index),
            "worker_node_id": self.worker_node_id,
            "gradient_b64": self.gradient_b64,
            "sample_count": int(self.sample_count),
            "worker_attestation_b64": (
                self.worker_attestation_b64
            ),
            "worker_signature_b64": (
                self.worker_signature_b64
            ),
            "timestamp": float(self.timestamp),
            "gradient_envelope_b64": (
                self.gradient_envelope_b64
            ),
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "GradientUpdate":
        return cls(
            job_id=d["job_id"],
            round_index=int(d["round_index"]),
            worker_node_id=d["worker_node_id"],
            gradient_b64=d["gradient_b64"],
            sample_count=int(d["sample_count"]),
            worker_attestation_b64=d.get(
                "worker_attestation_b64", "",
            ),
            worker_signature_b64=d.get(
                "worker_signature_b64", "",
            ),
            timestamp=float(d["timestamp"]),
            gradient_envelope_b64=d.get(
                "gradient_envelope_b64",
            ),
        )


@dataclass
class FederatedRound:
    job_id: str
    round_index: int
    status: RoundStatus
    worker_assignments: List[WorkerAssignment] = field(
        default_factory=list,
    )
    gradient_updates_received: List[GradientUpdate] = field(
        default_factory=list,
    )
    aggregated_update: bytes = b""
    issued_at: float = 0.0
    completed_at: Optional[float] = None
    # Sprint 308a — recorded σ of the DP noise actually
    # applied to the aggregated update (0.0 when no DP).
    dp_noise_sigma_applied: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "round_index": int(self.round_index),
            "status": self.status.value,
            "worker_assignments": [
                a.to_dict() for a in self.worker_assignments
            ],
            "gradient_updates_received": [
                u.to_dict()
                for u in self.gradient_updates_received
            ],
            "aggregated_update_b64": base64.b64encode(
                self.aggregated_update,
            ).decode("ascii"),
            "issued_at": float(self.issued_at),
            "completed_at": (
                float(self.completed_at)
                if self.completed_at is not None else None
            ),
            "dp_noise_sigma_applied": float(
                self.dp_noise_sigma_applied,
            ),
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "FederatedRound":
        return cls(
            job_id=d["job_id"],
            round_index=int(d["round_index"]),
            status=RoundStatus(d["status"]),
            worker_assignments=[
                WorkerAssignment.from_dict(a)
                for a in (
                    d.get("worker_assignments") or []
                )
            ],
            gradient_updates_received=[
                GradientUpdate.from_dict(u)
                for u in (
                    d.get("gradient_updates_received")
                    or []
                )
            ],
            aggregated_update=base64.b64decode(
                d.get("aggregated_update_b64") or "",
            ),
            issued_at=float(d.get("issued_at", 0.0)),
            completed_at=(
                float(d["completed_at"])
                if d.get("completed_at") is not None
                else None
            ),
            dp_noise_sigma_applied=float(
                d.get("dp_noise_sigma_applied", 0.0),
            ),
        )


@dataclass
class DPPolicy:
    """Sprint 308a — central differential privacy policy
    applied at the aggregator. Each element of the
    aggregated gradient is clipped to ±clip_norm; Gaussian
    noise is then added with σ = clip_norm * sqrt(2 *
    ln(1.25/δ)) / ε (standard Gaussian-mechanism formula
    for (ε, δ)-DP)."""

    epsilon: float
    delta: float
    clip_norm: float

    def __post_init__(self):
        if self.epsilon <= 0:
            raise ValueError(
                f"epsilon must be > 0; got {self.epsilon}"
            )
        if not (0.0 < self.delta < 1.0):
            raise ValueError(
                f"delta must be in (0, 1); got {self.delta}"
            )
        if self.clip_norm <= 0:
            raise ValueError(
                f"clip_norm must be > 0; got "
                f"{self.clip_norm}"
            )

    def sigma(self) -> float:
        return (
            self.clip_norm
            * math.sqrt(2.0 * math.log(1.25 / self.delta))
            / self.epsilon
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epsilon": float(self.epsilon),
            "delta": float(self.delta),
            "clip_norm": float(self.clip_norm),
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "DPPolicy":
        return cls(
            epsilon=float(d["epsilon"]),
            delta=float(d["delta"]),
            clip_norm=float(d["clip_norm"]),
        )


@dataclass
class WorkerKey:
    """Sprint 308a — registered worker Ed25519 signing
    pubkey. Used to verify gradient-update signatures when
    a job has require_signed_updates=True."""

    node_id: str
    signing_pubkey_b64: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "signing_pubkey_b64": self.signing_pubkey_b64,
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "WorkerKey":
        return cls(
            node_id=d["node_id"],
            signing_pubkey_b64=d["signing_pubkey_b64"],
        )


@dataclass
class FederatedJob:
    job_id: str
    model_id: str
    dataset_cids: List[str]
    worker_pool: List[str]
    rounds_target: int
    min_workers_per_round: int
    aggregation: AggregationStrategy
    status: JobStatus
    current_round: int = 0
    started_at: float = 0.0
    completed_at: Optional[float] = None
    # Sprint 308a — opt-in hardening
    require_signed_updates: bool = False
    dp_policy: Optional[DPPolicy] = None
    # Sprint 308c — encrypted-gradient transport. When
    # non-None, workers must seal their gradient to this
    # X25519 pubkey before submitting. Orchestrator unseals
    # via PRSM_FEDERATED_ORCHESTRATOR_TRANSPORT_PRIVKEY env.
    transport_pubkey_b64: Optional[str] = None
    # Sprint 1115 (Domain-08 review MEDIUM-7 follow-on) — TEE attestation policy for this
    # job, stored as the enterprise.tee_policy.TEEPolicy.to_dict() shape (min_attestation
    # _tier / allowed_vendors / require_signature_chain). When non-None,
    # accept_gradient_update evaluates each worker's worker_attestation_b64 against it and
    # REFUSES updates that don't satisfy it — closing the gap where the FL path checked
    # signatures but never the attestation despite the module docstring's TEE-gating
    # claim. None → no attestation gating (backwards-compatible).
    tee_policy: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "dataset_cids": list(self.dataset_cids),
            "worker_pool": list(self.worker_pool),
            "rounds_target": int(self.rounds_target),
            "min_workers_per_round": int(
                self.min_workers_per_round,
            ),
            "aggregation": self.aggregation.value,
            "status": self.status.value,
            "current_round": int(self.current_round),
            "started_at": float(self.started_at),
            "completed_at": (
                float(self.completed_at)
                if self.completed_at is not None else None
            ),
            "require_signed_updates": bool(
                self.require_signed_updates,
            ),
            "dp_policy": (
                self.dp_policy.to_dict()
                if self.dp_policy is not None else None
            ),
            "transport_pubkey_b64": (
                self.transport_pubkey_b64
            ),
            "tee_policy": (
                dict(self.tee_policy)
                if self.tee_policy is not None else None
            ),
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "FederatedJob":
        dp_raw = d.get("dp_policy")
        return cls(
            job_id=d["job_id"],
            model_id=d["model_id"],
            dataset_cids=list(d.get("dataset_cids") or []),
            worker_pool=list(d.get("worker_pool") or []),
            rounds_target=int(d["rounds_target"]),
            min_workers_per_round=int(
                d["min_workers_per_round"],
            ),
            aggregation=AggregationStrategy(
                d["aggregation"],
            ),
            status=JobStatus(d["status"]),
            current_round=int(d.get("current_round", 0)),
            started_at=float(d.get("started_at", 0.0)),
            completed_at=(
                float(d["completed_at"])
                if d.get("completed_at") is not None
                else None
            ),
            require_signed_updates=bool(
                d.get("require_signed_updates", False),
            ),
            dp_policy=(
                DPPolicy.from_dict(dp_raw)
                if dp_raw is not None else None
            ),
            transport_pubkey_b64=d.get(
                "transport_pubkey_b64",
            ),
            tee_policy=d.get("tee_policy"),
        )


# ── Aggregation ──────────────────────────────────────


def _validate_updates_consistent(
    updates: List[GradientUpdate],
) -> int:
    """Returns the common gradient length. Raises on
    empty list or length mismatch."""
    if not updates:
        raise ValueError(
            "aggregate requires at least one update"
        )
    import base64 as _b64
    first_len = None
    for u in updates:
        raw = _b64.b64decode(u.gradient_b64)
        if first_len is None:
            first_len = len(raw)
        elif len(raw) != first_len:
            raise ValueError(
                f"gradient length mismatch: expected "
                f"{first_len} bytes, got {len(raw)} on "
                f"worker {u.worker_node_id!r}"
            )
    assert first_len is not None
    return first_len


def aggregate_fedavg(
    updates: List[GradientUpdate],
) -> bytes:
    """Weighted-average aggregation: each worker's gradient
    is weighted by their `sample_count`. Returns packed
    float32 bytes."""
    import base64 as _b64

    _validate_updates_consistent(updates)
    for u in updates:
        if u.sample_count < 0:
            raise ValueError(
                f"negative sample_count on worker "
                f"{u.worker_node_id!r}: {u.sample_count}"
            )
    total_samples = sum(u.sample_count for u in updates)
    if total_samples <= 0:
        raise ValueError(
            "aggregate_fedavg requires total sample_count "
            "> 0 across all updates"
        )

    grads = [
        decode_gradient(_b64.b64decode(u.gradient_b64))
        for u in updates
    ]
    n = len(grads[0])
    out = [0.0] * n
    for u, g in zip(updates, grads):
        w = u.sample_count / total_samples
        for i in range(n):
            out[i] += w * g[i]
    return encode_gradient(out)


def aggregate_fedmedian(
    updates: List[GradientUpdate],
) -> bytes:
    """Element-wise median aggregation. Byzantine-robust:
    a single outlier worker cannot swing the result."""
    import base64 as _b64

    _validate_updates_consistent(updates)
    grads = [
        decode_gradient(_b64.b64decode(u.gradient_b64))
        for u in updates
    ]
    n = len(grads[0])
    out = [
        statistics.median(
            [g[i] for g in grads],
        )
        for i in range(n)
    ]
    return encode_gradient(out)


_AGGREGATORS = {
    AggregationStrategy.FEDAVG: aggregate_fedavg,
    AggregationStrategy.FEDMEDIAN: aggregate_fedmedian,
}


# ── Sprint 308a — worker signatures + central DP ─────


def _ed25519_b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _ed25519_b64d(s: str) -> bytes:
    if not isinstance(s, str):
        raise ValueError(
            f"expected base64 string, got {type(s).__name__}"
        )
    try:
        return base64.b64decode(s, validate=True)
    except Exception as e:
        raise ValueError(f"invalid base64: {e}")


def _load_ed25519_priv(b64: str) -> Ed25519PrivateKey:
    raw = _ed25519_b64d(b64)
    if len(raw) != 32:
        raise ValueError(
            f"Ed25519 privkey must be 32 bytes, got "
            f"{len(raw)}"
        )
    return Ed25519PrivateKey.from_private_bytes(raw)


def _load_ed25519_pub(b64: str) -> Ed25519PublicKey:
    raw = _ed25519_b64d(b64)
    if len(raw) != 32:
        raise ValueError(
            f"Ed25519 pubkey must be 32 bytes, got "
            f"{len(raw)}"
        )
    return Ed25519PublicKey.from_public_bytes(raw)


def generate_worker_keypair() -> tuple[str, str]:
    """Generate a fresh Ed25519 keypair for a worker.
    Returns (privkey_b64, pubkey_b64). The privkey stays
    device-bound on the worker; the pubkey is registered
    on the orchestrator."""
    priv = Ed25519PrivateKey.generate()
    priv_raw = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return _ed25519_b64e(priv_raw), _ed25519_b64e(pub_raw)


# ── Sprint 308c — encrypted-gradient transport ───────


_TRANSPORT_HKDF_INFO = b"prsm-fl-transport-seal-v1"
_TRANSPORT_NONCE_LEN = 12


def generate_transport_keypair() -> tuple[str, str]:
    """X25519 keypair for orchestrator gradient transport.
    Privkey lives ONLY on the orchestrator (loaded from
    PRSM_FEDERATED_ORCHESTRATOR_TRANSPORT_PRIVKEY env);
    pubkey is pinned on each FederatedJob.transport_pubkey_b64
    and distributed to workers."""
    priv = X25519PrivateKey.generate()
    priv_raw = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return _ed25519_b64e(priv_raw), _ed25519_b64e(pub_raw)


def _load_x25519_pub(b64: str) -> X25519PublicKey:
    raw = _ed25519_b64d(b64)
    if len(raw) != 32:
        raise ValueError(
            f"x25519 pubkey must be 32 bytes, got {len(raw)}"
        )
    return X25519PublicKey.from_public_bytes(raw)


def _load_x25519_priv(b64: str) -> X25519PrivateKey:
    raw = _ed25519_b64d(b64)
    if len(raw) != 32:
        raise ValueError(
            f"x25519 privkey must be 32 bytes, got "
            f"{len(raw)}"
        )
    return X25519PrivateKey.from_private_bytes(raw)


def _derive_transport_key(shared_secret: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32, salt=None,
        info=_TRANSPORT_HKDF_INFO,
    ).derive(shared_secret)


def seal_gradient_for_orchestrator(
    gradient_bytes: bytes,
    orchestrator_pubkey_b64: str,
) -> tuple[str, str]:
    """Seal a gradient to the orchestrator's transport
    pubkey. Returns (sealed_b64, envelope_b64).

    Envelope = ephemeral X25519 pubkey || nonce (44 bytes
    total raw; base64 ~60 chars). Sealed = ChaCha20-Poly1305
    AEAD ciphertext over the gradient bytes."""
    pub = _load_x25519_pub(orchestrator_pubkey_b64)
    ephemeral_priv = X25519PrivateKey.generate()
    ephemeral_pub = ephemeral_priv.public_key()
    shared = ephemeral_priv.exchange(pub)
    sealing_key = _derive_transport_key(shared)
    nonce = os.urandom(_TRANSPORT_NONCE_LEN)
    sealed = ChaCha20Poly1305(sealing_key).encrypt(
        nonce, gradient_bytes, None,
    )
    eph_pub_raw = ephemeral_pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    envelope_raw = eph_pub_raw + nonce
    return (
        _ed25519_b64e(sealed),
        _ed25519_b64e(envelope_raw),
    )


def unseal_gradient_from_worker(
    sealed_b64: str,
    envelope_b64: str,
    orchestrator_privkey_b64: str,
) -> bytes:
    """Unseal a gradient. Raises on tamper / wrong key /
    malformed envelope."""
    priv = _load_x25519_priv(orchestrator_privkey_b64)
    envelope = _ed25519_b64d(envelope_b64)
    if len(envelope) != 32 + _TRANSPORT_NONCE_LEN:
        raise ValueError(
            f"envelope must be {32 + _TRANSPORT_NONCE_LEN} "
            f"bytes (32 pubkey + 12 nonce); got "
            f"{len(envelope)}"
        )
    ephemeral_pub = X25519PublicKey.from_public_bytes(
        envelope[:32],
    )
    nonce = envelope[32:]
    shared = priv.exchange(ephemeral_pub)
    sealing_key = _derive_transport_key(shared)
    sealed = _ed25519_b64d(sealed_b64)
    return ChaCha20Poly1305(sealing_key).decrypt(
        nonce, sealed, None,
    )


def _canonical_update_bytes(
    u: GradientUpdate,
) -> bytes:
    """Stable signing payload for a gradient update.
    Excludes worker_signature_b64 (signature can't bind
    itself); INCLUDES worker_attestation_b64 as of sprint
    308b so the attestation that produced the gradient is
    cryptographically bound to the gradient itself. A
    relabeled / forged attestation breaks the signature."""
    payload = {
        "job_id": u.job_id,
        "round_index": int(u.round_index),
        "worker_node_id": u.worker_node_id,
        "gradient_b64": u.gradient_b64,
        "sample_count": int(u.sample_count),
        "timestamp": float(u.timestamp),
        "worker_attestation_b64": (
            u.worker_attestation_b64
        ),
        # Sprint 308c — bind envelope so a MITM can't
        # strip or replace it without breaking the
        # signature
        "gradient_envelope_b64": (
            u.gradient_envelope_b64
        ),
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def sign_gradient_update(
    update: GradientUpdate,
    *,
    worker_privkey_b64: str,
) -> GradientUpdate:
    """Sign a gradient update with the worker's Ed25519
    private key. Returns the same update with
    `worker_signature_b64` populated."""
    priv = _load_ed25519_priv(worker_privkey_b64)
    sig = priv.sign(_canonical_update_bytes(update))
    update.worker_signature_b64 = _ed25519_b64e(sig)
    return update


def verify_gradient_update_signature(
    update: GradientUpdate, pubkey_b64: str,
) -> bool:
    """Verify the worker signature on a gradient update."""
    if not update.worker_signature_b64:
        return False
    try:
        pub = _load_ed25519_pub(pubkey_b64)
        sig = _ed25519_b64d(update.worker_signature_b64)
    except ValueError:
        return False
    try:
        pub.verify(sig, _canonical_update_bytes(update))
        return True
    except InvalidSignature:
        return False


def _apply_central_dp(
    aggregated: bytes,
    policy: DPPolicy,
    rng: random.Random,
) -> bytes:
    """Clip each element of the aggregated gradient to
    ±clip_norm and add Gaussian noise N(0, σ²) where
    σ = clip_norm * sqrt(2 * ln(1.25/δ)) / ε.

    Pure Python; uses random.gauss for noise sampling so
    tests can pin the RNG for reproducibility."""
    values = decode_gradient(aggregated)
    sigma = policy.sigma()
    out = []
    for v in values:
        clipped = max(
            -policy.clip_norm,
            min(policy.clip_norm, v),
        )
        noised = clipped + rng.gauss(0.0, sigma)
        out.append(noised)
    return encode_gradient(out)


# ── Orchestrator ─────────────────────────────────────


class FederatedLearningOrchestrator:
    """In-memory + filesystem-persisted federated-learning
    coordinator. Jobs and rounds are stored as JSON files
    under persist_dir."""

    def __init__(
        self,
        *,
        persist_dir: Optional[Path] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._jobs: Dict[str, FederatedJob] = {}
        # rounds keyed by (job_id, round_index)
        self._rounds: Dict[
            tuple[str, int], FederatedRound,
        ] = {}
        # Sprint 308a — worker key registry (node_id →
        # WorkerKey). Used to verify gradient-update
        # signatures when a job has
        # require_signed_updates=True.
        self._worker_keys: Dict[str, WorkerKey] = {}
        self._persist_dir: Optional[Path] = (
            Path(persist_dir)
            if persist_dir is not None else None
        )
        # sp1105 (Domain-08 review HIGH-3) — DP noise must come from a CSPRNG. The
        # default was random.Random() (Mersenne Twister), which is reconstructible from
        # ~624 outputs, so an observer of enough noised gradients could predict and
        # partially subtract the noise → the advertised ε-guarantee degrades. Use
        # secrets.SystemRandom (a random.Random subclass backed by os.urandom, so .gauss
        # still works); the injectable rng remains for deterministic tests.
        self._rng = rng if rng is not None else secrets.SystemRandom()
        if self._persist_dir is not None:
            self._persist_dir.mkdir(
                parents=True, exist_ok=True,
            )
            self._load_from_disk()

    # ── Sprint 308a — worker key registry ─────────────

    def register_worker_key(
        self, key: WorkerKey,
    ) -> None:
        # Validate pubkey eagerly
        try:
            _load_ed25519_pub(key.signing_pubkey_b64)
        except ValueError as e:
            raise ValueError(
                f"invalid signing_pubkey_b64 for worker "
                f"{key.node_id!r}: {e}"
            )
        self._worker_keys[key.node_id] = key
        self._persist_worker_keys()

    def get_worker_key(
        self, node_id: str,
    ) -> Optional[WorkerKey]:
        return self._worker_keys.get(node_id)

    def list_worker_keys(self) -> List[WorkerKey]:
        return list(self._worker_keys.values())

    @classmethod
    def from_env(
        cls,
    ) -> "FederatedLearningOrchestrator":
        raw = os.environ.get("PRSM_FEDERATED_LEARNING_DIR")
        persist_dir = Path(raw) if raw else None
        return cls(persist_dir=persist_dir)

    # ── Job lifecycle ────────────────────────────────

    def propose_job(
        self,
        *,
        model_id: str,
        dataset_cids: List[str],
        worker_pool: List[str],
        rounds_target: int,
        min_workers_per_round: int,
        aggregation: AggregationStrategy,
        require_signed_updates: bool = False,
        dp_policy: Optional[DPPolicy] = None,
        transport_pubkey_b64: Optional[str] = None,
        tee_policy: Optional[Dict[str, Any]] = None,
    ) -> FederatedJob:
        if not worker_pool:
            raise ValueError(
                "worker_pool must be non-empty"
            )
        if rounds_target < 1:
            raise ValueError(
                f"rounds_target must be >= 1; got "
                f"{rounds_target}"
            )
        if min_workers_per_round < 1:
            raise ValueError(
                f"min_workers_per_round must be >= 1; got "
                f"{min_workers_per_round}"
            )
        if min_workers_per_round > len(worker_pool):
            raise ValueError(
                f"min_workers_per_round "
                f"({min_workers_per_round}) exceeds pool "
                f"size ({len(worker_pool)})"
            )
        # Sprint 308c — validate transport pubkey format
        # eagerly so propose-job fails loud on misconfig.
        if transport_pubkey_b64 is not None:
            try:
                _load_x25519_pub(transport_pubkey_b64)
            except ValueError as e:
                raise ValueError(
                    f"invalid transport_pubkey_b64: {e}"
                )
        # sp1115 — validate the TEE policy eagerly so propose-job fails loud on a
        # malformed policy (rather than at the first gradient-update evaluation).
        if tee_policy is not None:
            from prsm.enterprise.tee_policy import TEEPolicy
            try:
                TEEPolicy.from_dict(tee_policy)
            except ValueError as e:
                raise ValueError(f"invalid tee_policy: {e}")
        job = FederatedJob(
            job_id=str(uuid.uuid4()),
            model_id=model_id,
            dataset_cids=list(dataset_cids),
            worker_pool=list(worker_pool),
            rounds_target=int(rounds_target),
            min_workers_per_round=int(
                min_workers_per_round,
            ),
            aggregation=aggregation,
            status=JobStatus.PROPOSED,
            started_at=time.time(),
            require_signed_updates=bool(
                require_signed_updates,
            ),
            dp_policy=dp_policy,
            transport_pubkey_b64=transport_pubkey_b64,
            tee_policy=tee_policy,
        )
        self._jobs[job.job_id] = job
        self._persist_job(job)
        return job

    def get_job(
        self, job_id: str,
    ) -> Optional[FederatedJob]:
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        *,
        status: Optional[JobStatus] = None,
    ) -> List[FederatedJob]:
        out = list(self._jobs.values())
        out.sort(
            key=lambda j: j.started_at, reverse=True,
        )
        if status is not None:
            out = [j for j in out if j.status == status]
        return out

    # ── Round lifecycle ──────────────────────────────

    def issue_round(self, job_id: str) -> FederatedRound:
        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(
                f"job {job_id!r} not found"
            )
        if job.status in _TERMINAL_JOB_STATUSES:
            raise ValueError(
                f"job {job_id!r} is complete or terminal "
                f"(status={job.status.value!r}); cannot "
                f"issue more rounds"
            )
        if job.current_round >= job.rounds_target:
            raise ValueError(
                f"job {job_id!r} has issued all "
                f"{job.rounds_target} rounds; cannot "
                f"issue more"
            )
        # Pick the round-of-workers: a deterministic
        # subset of size >= min_workers_per_round. For v1,
        # sample exactly min_workers_per_round (smallest
        # quorum). Operators can extend later.
        pool = list(job.worker_pool)
        # Use the RNG seeded per-orchestrator for testability
        self._rng.shuffle(pool)
        assigned = pool[:job.min_workers_per_round]
        now = time.time()
        # Round-robin a dataset_cid for each assigned worker
        # (cycles if dataset count < worker count)
        assignments = [
            WorkerAssignment(
                node_id=node_id,
                dataset_cid=job.dataset_cids[
                    i % max(1, len(job.dataset_cids))
                ] if job.dataset_cids else "",
                assigned_at=now,
            )
            for i, node_id in enumerate(assigned)
        ]
        rnd = FederatedRound(
            job_id=job_id,
            round_index=job.current_round,
            status=RoundStatus.ISSUED,
            worker_assignments=assignments,
            issued_at=now,
        )
        self._rounds[(job_id, job.current_round)] = rnd
        # Transition job out of PROPOSED on first round
        if job.status == JobStatus.PROPOSED:
            job.status = JobStatus.RUNNING
            self._persist_job(job)
        self._persist_round(rnd)
        return rnd

    def get_round(
        self, job_id: str, round_index: int,
    ) -> Optional[FederatedRound]:
        return self._rounds.get((job_id, round_index))

    def accept_gradient_update(
        self, update: GradientUpdate,
    ) -> None:
        job = self._jobs.get(update.job_id)
        if job is None:
            raise ValueError(
                f"job {update.job_id!r} not found"
            )
        rnd = self._rounds.get(
            (update.job_id, update.round_index),
        )
        if rnd is None:
            raise ValueError(
                f"round {update.round_index} not issued "
                f"for job {update.job_id!r}"
            )
        if rnd.status == RoundStatus.AGGREGATED:
            raise ValueError(
                f"round {update.round_index} is already "
                f"aggregated; cannot accept more updates"
            )
        # Worker must be in the assignment list
        assigned_nodes = {
            a.node_id for a in rnd.worker_assignments
        }
        if update.worker_node_id not in assigned_nodes:
            raise ValueError(
                f"worker {update.worker_node_id!r} not "
                f"assigned to round {update.round_index}"
            )
        # No duplicate per worker
        for u in rnd.gradient_updates_received:
            if u.worker_node_id == update.worker_node_id:
                raise ValueError(
                    f"worker {update.worker_node_id!r} "
                    f"already submitted an update for "
                    f"this round (duplicate)"
                )
        # Sprint 308a — signature gate
        if job.require_signed_updates:
            worker_key = self._worker_keys.get(
                update.worker_node_id,
            )
            if worker_key is None:
                raise ValueError(
                    f"worker {update.worker_node_id!r} "
                    f"has no registered signing key; "
                    f"register a key before submitting "
                    f"updates"
                )
            if not update.worker_signature_b64:
                raise ValueError(
                    f"job {job.job_id!r} requires signed "
                    f"updates but this submission carries "
                    f"no signature"
                )
            if not verify_gradient_update_signature(
                update, worker_key.signing_pubkey_b64,
            ):
                raise ValueError(
                    f"worker {update.worker_node_id!r} "
                    f"signature does not verify against "
                    f"the registered pubkey"
                )
        # Sprint 1115 (Domain-08 review MEDIUM-7 follow-on) — TEE attestation gate. When
        # the job declares a tee_policy, every worker's attestation MUST satisfy it before
        # its gradient is accepted into the round. This is what makes the module's
        # TEE-gating claim real: a worker without a sufficient attestation (e.g. software
        # tier when the job requires hardware-verified) is REFUSED, fail-closed.
        if job.tee_policy is not None:
            import base64 as _b64
            from prsm.enterprise.tee_policy import (
                TEEPolicy, evaluate_attestation_blob,
            )
            policy = TEEPolicy.from_dict(job.tee_policy)
            blob: Optional[bytes] = None
            if update.worker_attestation_b64:
                try:
                    blob = _b64.b64decode(
                        update.worker_attestation_b64, validate=True,
                    )
                except Exception:  # noqa: BLE001 — bad b64 → no attestation → fail closed
                    blob = None
            result = evaluate_attestation_blob(blob, policy)
            if result.status.value != "pass":
                raise ValueError(
                    f"worker {update.worker_node_id!r} attestation does not satisfy the "
                    f"job's TEE policy: effective_tier={result.effective_tier.value}, "
                    f"required={result.min_required_tier.value}, "
                    f"vendor={result.vendor or '(none)'}. {result.diagnostic}"
                )
        rnd.gradient_updates_received.append(update)
        if rnd.status == RoundStatus.ISSUED:
            rnd.status = RoundStatus.COLLECTING
        self._persist_round(rnd)

    def aggregate_round(
        self, job_id: str, round_index: int,
    ) -> FederatedRound:
        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(
                f"job {job_id!r} not found"
            )
        rnd = self._rounds.get((job_id, round_index))
        if rnd is None:
            raise ValueError(
                f"round {round_index} not issued for job "
                f"{job_id!r}"
            )
        if rnd.status == RoundStatus.AGGREGATED:
            raise ValueError(
                f"round {round_index} already aggregated "
                f"(status={rnd.status.value!r})"
            )
        if (
            len(rnd.gradient_updates_received)
            < job.min_workers_per_round
        ):
            raise ValueError(
                f"round {round_index} below quorum: got "
                f"{len(rnd.gradient_updates_received)} "
                f"updates, need "
                f"{job.min_workers_per_round} "
                f"(min_workers_per_round)"
            )
        # Sprint 308c — if the job has transport encryption,
        # unseal each gradient before aggregation. The
        # received updates are preserved verbatim (sealed)
        # so the signed-bytes audit trail stays intact; we
        # produce plaintext copies just for aggregation.
        updates_for_aggregation = rnd.gradient_updates_received
        if job.transport_pubkey_b64 is not None:
            orchestrator_priv = os.environ.get(
                "PRSM_FEDERATED_ORCHESTRATOR_TRANSPORT_PRIVKEY",
                "",
            ).strip()
            if not orchestrator_priv:
                raise ValueError(
                    f"job {job_id!r} has transport "
                    f"encryption but orchestrator "
                    f"transport privkey is not configured "
                    f"(set PRSM_FEDERATED_ORCHESTRATOR_"
                    f"TRANSPORT_PRIVKEY env)"
                )
            unsealed_updates = []
            for u in rnd.gradient_updates_received:
                if u.gradient_envelope_b64 is None:
                    rnd.status = RoundStatus.FAILED
                    self._persist_round(rnd)
                    raise ValueError(
                        f"job {job_id!r} requires "
                        f"transport encryption but update "
                        f"from worker "
                        f"{u.worker_node_id!r} has no "
                        f"envelope"
                    )
                try:
                    plaintext = unseal_gradient_from_worker(
                        u.gradient_b64,
                        u.gradient_envelope_b64,
                        orchestrator_priv,
                    )
                except Exception as exc:  # noqa: BLE001
                    rnd.status = RoundStatus.FAILED
                    self._persist_round(rnd)
                    raise ValueError(
                        f"unsealing gradient from worker "
                        f"{u.worker_node_id!r} failed: "
                        f"{exc}"
                    )
                # Build a sibling unsealed update (don't
                # mutate the stored signed one)
                unsealed_updates.append(GradientUpdate(
                    job_id=u.job_id,
                    round_index=u.round_index,
                    worker_node_id=u.worker_node_id,
                    gradient_b64=base64.b64encode(
                        plaintext,
                    ).decode("ascii"),
                    sample_count=u.sample_count,
                    worker_attestation_b64=(
                        u.worker_attestation_b64
                    ),
                    worker_signature_b64=(
                        u.worker_signature_b64
                    ),
                    timestamp=u.timestamp,
                    gradient_envelope_b64=None,
                ))
            updates_for_aggregation = unsealed_updates

        aggregator = _AGGREGATORS[job.aggregation]
        try:
            rnd.aggregated_update = aggregator(
                updates_for_aggregation,
            )
        except ValueError as e:
            rnd.status = RoundStatus.FAILED
            self._persist_round(rnd)
            raise ValueError(
                f"aggregation failed for round "
                f"{round_index}: {e}"
            )
        # Sprint 308a — central DP: clip + Gaussian noise
        if job.dp_policy is not None:
            rnd.aggregated_update = _apply_central_dp(
                rnd.aggregated_update,
                job.dp_policy,
                self._rng,
            )
            rnd.dp_noise_sigma_applied = (
                job.dp_policy.sigma()
            )
        rnd.status = RoundStatus.AGGREGATED
        rnd.completed_at = time.time()
        self._persist_round(rnd)

        # Advance job state
        job.current_round += 1
        if job.current_round >= job.rounds_target:
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
        self._persist_job(job)
        return rnd

    # ── Persistence ──────────────────────────────────

    def _job_path(self, job_id: str) -> Path:
        assert self._persist_dir is not None
        safe = (
            job_id.replace("/", "_")
            .replace("\\", "_")
            .replace("..", "_")
        )
        return self._persist_dir / f"job-{safe}.json"

    def _round_path(
        self, job_id: str, round_index: int,
    ) -> Path:
        assert self._persist_dir is not None
        safe = (
            job_id.replace("/", "_")
            .replace("\\", "_")
            .replace("..", "_")
        )
        return (
            self._persist_dir
            / f"round-{safe}-{int(round_index)}.json"
        )

    def _persist_job(self, job: FederatedJob) -> None:
        if self._persist_dir is None:
            return
        path = self._job_path(job.job_id)
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(job.to_dict()))
            tmp.replace(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "FederatedLearningOrchestrator: job "
                "persist failed for %s: %s",
                job.job_id, exc,
            )

    def _persist_worker_keys(self) -> None:
        if self._persist_dir is None:
            return
        path = self._persist_dir / "worker-keys.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps([
                k.to_dict()
                for k in self._worker_keys.values()
            ]))
            tmp.replace(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "FederatedLearningOrchestrator: "
                "worker-keys persist failed: %s", exc,
            )

    def _persist_round(self, rnd: FederatedRound) -> None:
        if self._persist_dir is None:
            return
        path = self._round_path(rnd.job_id, rnd.round_index)
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(rnd.to_dict()))
            tmp.replace(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "FederatedLearningOrchestrator: round "
                "persist failed for %s-%d: %s",
                rnd.job_id, rnd.round_index, exc,
            )

    def _load_from_disk(self) -> None:
        assert self._persist_dir is not None
        # Worker keys
        wk_path = self._persist_dir / "worker-keys.json"
        if wk_path.exists():
            try:
                for r in json.loads(wk_path.read_text()):
                    k = WorkerKey.from_dict(r)
                    self._worker_keys[k.node_id] = k
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "FederatedLearningOrchestrator: "
                    "skipping corrupt worker-keys.json: %s",
                    exc,
                )
        # Jobs first so rounds find their parent
        for path in self._persist_dir.glob("job-*.json"):
            try:
                job = FederatedJob.from_dict(
                    json.loads(path.read_text()),
                )
                self._jobs[job.job_id] = job
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "FederatedLearningOrchestrator: "
                    "skipping corrupt job %s: %s",
                    path, exc,
                )
        for path in self._persist_dir.glob("round-*.json"):
            try:
                rnd = FederatedRound.from_dict(
                    json.loads(path.read_text()),
                )
                self._rounds[
                    (rnd.job_id, rnd.round_index)
                ] = rnd
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "FederatedLearningOrchestrator: "
                    "skipping corrupt round %s: %s",
                    path, exc,
                )
