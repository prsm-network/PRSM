"""
TEE data models for confidential compute.

Defines TEE types, capabilities, differential privacy configuration,
privacy levels, and confidential execution results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional


HARDWARE_TEE_TYPES = frozenset({"sgx", "tdx", "sev", "trustzone", "secure_enclave"})


class TEEType(str, Enum):
    """Supported Trusted Execution Environment types."""

    NONE = "none"
    SGX = "sgx"
    TDX = "tdx"
    SEV = "sev"
    TRUSTZONE = "trustzone"
    SECURE_ENCLAVE = "secure_enclave"
    SOFTWARE = "software"


class PrivacyLevel(str, Enum):
    """Privacy level presets mapping to DPConfig parameters.

    sp1234 — NONE is the production default. STANDARD/HIGH/MAXIMUM apply
    activation-DP noise and are EXPERIMENTAL / best-effort obfuscation, NOT
    strong privacy: per-element Gaussian noise on a high-dim activation has
    SNR ∝ ε²/D, so usable output quality requires a near-no-privacy ε (sp1233
    GPU sweep — these tiers corrupt output at their nominal ε). For real
    privacy guarantees use a TEE (Tier 3), not these DP tiers.
    """

    NONE = "none"
    STANDARD = "standard"   # experimental DP — see class docstring
    HIGH = "high"           # experimental DP — see class docstring
    MAXIMUM = "maximum"     # experimental DP — see class docstring

    @staticmethod
    def config_for_level(level: PrivacyLevel) -> DPConfig:
        """Return a DPConfig matching the given privacy level."""
        epsilon_map = {
            PrivacyLevel.NONE: float("inf"),
            PrivacyLevel.STANDARD: 8.0,
            PrivacyLevel.HIGH: 4.0,
            PrivacyLevel.MAXIMUM: 1.0,
        }
        return DPConfig(epsilon=epsilon_map[level])


@dataclass
class TEECapability:
    """Describes the TEE capabilities of a compute node."""

    tee_type: TEEType = TEEType.NONE
    max_enclave_memory_mb: int = 0
    max_threads: int = 0
    attestation_supported: bool = False

    @property
    def available(self) -> bool:
        """Whether any TEE is available."""
        return self.tee_type != TEEType.NONE

    @property
    def is_hardware_backed(self) -> bool:
        """Whether the TEE is backed by hardware."""
        return self.tee_type.value in HARDWARE_TEE_TYPES

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        d = asdict(self)
        d["tee_type"] = self.tee_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> TEECapability:
        """Deserialize from a plain dictionary."""
        data = dict(data)  # copy
        if "tee_type" in data:
            data["tee_type"] = TEEType(data["tee_type"])
        return cls(**data)


@dataclass
class DPConfig:
    """Differential privacy configuration."""

    epsilon: float = 8.0
    delta: float = 1e-5
    clip_norm: float = 1.0
    noise_mechanism: str = "gaussian"

    @property
    def noise_scale(self) -> float:
        """Compute the Gaussian noise standard deviation (sigma).

        sigma = clip_norm * sqrt(2 * ln(1.25 / delta)) / epsilon
        Returns inf when epsilon <= 0.
        """
        if self.epsilon <= 0:
            return float("inf")
        return self.clip_norm * math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.epsilon


@dataclass
class ConfidentialResult:
    """Result from a confidential TEE execution."""

    output: bytes = b""
    dp_applied: bool = False
    epsilon_spent: float = 0.0
    tee_type: TEEType = TEEType.SOFTWARE
    attestation_proof: Optional[str] = None
    execution_time_seconds: float = 0.0
    memory_used_bytes: int = 0

    @property
    def is_hardware_attested(self) -> bool:
        """Whether the result has a valid hardware attestation."""
        return (
            self.tee_type.value in HARDWARE_TEE_TYPES
            and self.attestation_proof is not None
            and len(self.attestation_proof) > 0
        )
