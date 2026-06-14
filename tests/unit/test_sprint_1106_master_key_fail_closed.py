"""Sprint 1106 — fold the Domain-08 review HIGH-4 (the safe, non-breaking part): the
at-rest master-key 'hsm'/'kms' source must FAIL CLOSED, not fake hardware protection.

KeyManager._initialize_master_key, when configured with master_key_source != "environment"
(i.e. "hsm"/"kms" — an operator asserting HARDWARE key protection), silently generated an
ephemeral in-process Fernet key and only logged a warning. That gives a false sense of
hardware-backed protection while the key is actually software-only and lost on restart —
a fail-OPEN on a security-relevant config (invariant #4: config must fail closed). No
production caller selects that source (default is "environment"), so failing it closed is
non-breaking.

(The separate per-process key regeneration in the "environment" branch when PRSM_MASTER_KEY
is unset is the documented dev behavior; requiring the env var in production is a deploy-
policy hardening tracked as a follow-on, not folded here.)
"""
from __future__ import annotations

import pytest

from prsm.core.cryptography.key_management import KeyManager


def test_hsm_source_without_backend_fails_closed():
    """Selecting an HSM/KMS master-key source with no real backend must RAISE, not
    silently fall back to an ephemeral software key claiming hardware protection."""
    with pytest.raises(Exception) as exc:
        KeyManager(config={"master_key_source": "hsm"})
    # the error must name the missing hardware integration (fail-closed, not silent)
    assert "hsm" in str(exc.value).lower() or "kms" in str(exc.value).lower() \
        or "not implemented" in str(exc.value).lower() or "integrat" in str(exc.value).lower()


def test_kms_source_without_backend_fails_closed():
    with pytest.raises(Exception):
        KeyManager(config={"master_key_source": "kms"})


def test_environment_source_with_key_still_works():
    """The supported path (explicit PRSM_MASTER_KEY) must keep working."""
    import base64
    from cryptography.fernet import Fernet
    key_b64 = base64.b64encode(Fernet.generate_key()).decode()
    import os
    os.environ["PRSM_MASTER_KEY"] = key_b64
    try:
        km = KeyManager(config={"master_key_source": "environment"})
        assert km.master_key is not None
    finally:
        del os.environ["PRSM_MASTER_KEY"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
