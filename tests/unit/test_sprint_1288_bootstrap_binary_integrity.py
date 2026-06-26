"""Sprint 1288 — AWS bootstrap binary integrity (supply-chain audit #7).

scripts/deploy_bootstrap_aws.sh runs a cloud-init user-data script as root on every
bootstrap EC2 node. It used to fetch the docker-compose binary from a floating
`releases/latest/download/...` URL with NO integrity check, straight into
/usr/local/bin, then chmod +x — so a MITM or a compromised release asset would run
arbitrary root code on every node. The fix pins a specific version and verifies an
out-of-band hardcoded SHA256 (a checksum from the same release can't protect against
a compromised release) BEFORE the binary is made executable, failing closed on
mismatch.

This is the regression guard. (Sibling: #8 dependency hash-lockfile remains — disruptive,
deferred. #9 GitHub Action SHA pinning shipped in sp1287.)
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "deploy_bootstrap_aws.sh"


@pytest.fixture(scope="module")
def script_text() -> str:
    assert _SCRIPT.is_file(), f"missing {_SCRIPT}"
    return _SCRIPT.read_text()


def test_no_floating_latest_binary_download(script_text):
    """The docker-compose binary must NOT be pulled from a mutable releases/latest URL."""
    assert "releases/latest/download" not in script_text, (
        "docker-compose is fetched from a floating releases/latest URL (no integrity "
        "pin) — supply-chain audit #7 regression"
    )


def test_compose_version_is_pinned(script_text):
    """A concrete DOCKER_COMPOSE_VERSION must be pinned."""
    m = re.search(r'DOCKER_COMPOSE_VERSION="(v\d+\.\d+\.\d+)"', script_text)
    assert m, "DOCKER_COMPOSE_VERSION is not pinned to a concrete vX.Y.Z"


def test_checksum_verified_before_install(script_text):
    """The download must be SHA256-verified (sha256sum -c) and fail closed before install."""
    assert "sha256sum -c" in script_text, "no sha256sum -c verification of the download"
    # Fail-closed: a mismatch must abort (exit 1) rather than continue to install.
    assert re.search(r"sha256sum -c[^\n]*\n[^\n]*exit 1", script_text) or re.search(
        r"sha256sum -c.*exit 1", script_text, re.DOTALL
    ), "checksum verification does not fail closed (no exit on mismatch)"
    # Hardcoded out-of-band digests (64-hex) must be present for the pinned arches.
    digests = re.findall(r'DC_SHA256="([0-9a-f]{64})"', script_text)
    assert len(digests) >= 2, f"expected >=2 hardcoded arch digests, found {len(digests)}"


def test_install_happens_after_verification(script_text):
    """The binary must only be installed to /usr/local/bin AFTER the checksum check."""
    verify_idx = script_text.find("sha256sum -c")
    install_idx = script_text.find("install -m 0755 /tmp/docker-compose /usr/local/bin/docker-compose")
    assert verify_idx != -1 and install_idx != -1, "expected verify + install steps present"
    assert verify_idx < install_idx, "install precedes checksum verification (must be after)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
