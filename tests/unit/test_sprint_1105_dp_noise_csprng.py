"""Sprint 1105 — fold the Domain-08 review HIGH-3: differential-privacy noise must be
sampled from a CSPRNG, not the Mersenne Twister.

FederatedLearningOrchestrator (wired into the live node via the FL endpoints in
prsm/node/api.py + node.py) sampled its central-DP Gaussian noise from
``random.Random()`` (Mersenne Twister) by default. DP's ε-guarantee assumes the noise
is unpredictable to an adversary; Mersenne Twister is fully reconstructible from ~624
outputs, so an observer of enough noised gradients can predict subsequent noise and
partially subtract it, degrading the advertised privacy. The production default must be
a CSPRNG (secrets.SystemRandom); the injectable rng stays for deterministic tests.
"""
from __future__ import annotations

import random
import secrets

from prsm.enterprise.federated_learning import FederatedLearningOrchestrator


def test_default_rng_is_csprng():
    """A default-constructed orchestrator must NOT use the bare Mersenne-Twister RNG for
    DP noise — it must use a CSPRNG (secrets.SystemRandom)."""
    orch = FederatedLearningOrchestrator()
    assert isinstance(orch._rng, secrets.SystemRandom), (
        "DP noise RNG must be a CSPRNG; a predictable Mersenne Twister degrades the "
        "ε-guarantee"
    )
    # secrets.SystemRandom is a random.Random subclass, so .gauss still works
    assert hasattr(orch._rng, "gauss")


def test_injected_rng_is_honored_for_test_determinism():
    """A test/caller may still inject a seeded RNG for deterministic behavior."""
    seeded = random.Random(1234)
    orch = FederatedLearningOrchestrator(rng=seeded)
    assert orch._rng is seeded


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
