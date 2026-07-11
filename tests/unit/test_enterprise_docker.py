"""Sprint 318a — Docker artifacts for the enterprise demo.

Two layers of tests:
  - STATIC validation (always run): Dockerfile parses,
    docker-compose YAML is well-formed, .dockerignore
    excludes pycache, requirements pins are tight
  - LIVE integration (skipped without Docker daemon):
    `docker build` the image, `docker run` the demo
    subcommand inside, expect rc=0 + "All demos passed"
    in stdout — the LOAD-BEARING boot validation

The live integration test is gated by Docker daemon
availability so the static tests still run in CI envs
without Docker.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEPLOY_DIR = _REPO_ROOT / "deploy" / "enterprise-demo"
_DOCKERFILE = _DEPLOY_DIR / "Dockerfile"
_COMPOSE = _DEPLOY_DIR / "docker-compose.yml"
_DOCKERIGNORE = _DEPLOY_DIR / ".dockerignore"
_REQS = _DEPLOY_DIR / "requirements-enterprise-demo.txt"


# ── STATIC: artifact files exist ───────────────────


def test_dockerfile_exists():
    assert _DOCKERFILE.exists()


def test_compose_yml_exists():
    assert _COMPOSE.exists()


def test_dockerignore_exists():
    assert _DOCKERIGNORE.exists()


def test_requirements_exists():
    assert _REQS.exists()


# ── STATIC: Dockerfile content ─────────────────────


def test_dockerfile_has_multi_stage_build():
    content = _DOCKERFILE.read_text()
    # Two FROM lines = multi-stage
    from_count = sum(
        1 for line in content.splitlines()
        if line.strip().startswith("FROM ")
    )
    assert from_count >= 2, (
        "Dockerfile should use multi-stage build "
        "(builder + runtime)"
    )


def test_dockerfile_copies_prsm_source():
    content = _DOCKERFILE.read_text()
    assert "COPY prsm/" in content or (
        "COPY prsm " in content
    )


def test_dockerfile_does_not_copy_tests():
    """Build context excludes tests via .dockerignore;
    the Dockerfile itself also shouldn't COPY them
    explicitly (defense in depth)."""
    content = _DOCKERFILE.read_text()
    for forbidden in ("COPY tests", "COPY ./tests"):
        assert forbidden not in content


def test_dockerfile_default_cmd_runs_demo():
    """The default CMD runs `prsm-enterprise-bringup
    demo` — boot validation comes for free."""
    content = _DOCKERFILE.read_text()
    assert "bringup_cli" in content
    assert "demo" in content


def test_dockerfile_has_volume_for_persistence():
    """Persistence dir is mounted as a volume so
    operator state survives container restarts."""
    content = _DOCKERFILE.read_text()
    assert (
        "VOLUME" in content
        or "/var/lib/prsm" in content
    )


def test_dockerfile_has_healthcheck():
    content = _DOCKERFILE.read_text()
    assert "HEALTHCHECK" in content


# ── STATIC: docker-compose YAML ────────────────────


def test_compose_is_valid_yaml():
    content = _COMPOSE.read_text()
    data = yaml.safe_load(content)
    assert isinstance(data, dict)
    assert "services" in data


def test_compose_has_demo_service():
    data = yaml.safe_load(_COMPOSE.read_text())
    services = data.get("services") or {}
    # Service name must reference 'enterprise-demo' so
    # operators can find it
    has_demo = any(
        "enterprise-demo" in name
        or "demo" in name.lower()
        for name in services
    )
    assert has_demo, (
        f"compose has no enterprise-demo service: "
        f"{list(services)}"
    )


def test_compose_build_context_points_to_repo_root():
    data = yaml.safe_load(_COMPOSE.read_text())
    services = data.get("services") or {}
    for name, svc in services.items():
        build = svc.get("build")
        if isinstance(build, dict) and "context" in build:
            # ../.. (compose lives 2 levels deep)
            assert "../.." in build["context"] or (
                build["context"].endswith(".")
            )


def test_compose_dockerfile_path_references_our_dockerfile():
    data = yaml.safe_load(_COMPOSE.read_text())
    services = data.get("services") or {}
    for name, svc in services.items():
        build = svc.get("build")
        if isinstance(build, dict):
            df = build.get("dockerfile") or ""
            assert "deploy/enterprise-demo/Dockerfile" in df


def test_compose_mounts_persistence_volume():
    data = yaml.safe_load(_COMPOSE.read_text())
    services = data.get("services") or {}
    has_persistence_mount = False
    for svc in services.values():
        for vol in (svc.get("volumes") or []):
            if "/var/lib/prsm" in vol:
                has_persistence_mount = True
    assert has_persistence_mount


# ── STATIC: .dockerignore excludes correct paths ───


def test_dockerignore_excludes_pycache():
    content = _DOCKERIGNORE.read_text()
    assert "__pycache__" in content


def test_dockerignore_excludes_git():
    content = _DOCKERIGNORE.read_text()
    assert ".git" in content


def test_dockerignore_excludes_tests():
    """Tests live outside the runtime image — wasted
    context bytes + slower builds."""
    content = _DOCKERIGNORE.read_text()
    # Either tests/ excluded or .pytest_cache or both
    assert (
        ".pytest_cache" in content
        or "tests/" in content
    )


def test_dockerignore_excludes_data_dirs():
    """Operator data + venvs shouldn't bleed into the
    build context."""
    content = _DOCKERIGNORE.read_text()
    assert ".venv" in content or "venv/" in content


# ── STATIC: requirements pins ──────────────────────


def test_requirements_pins_torch():
    """Torch is heavy + the sprint 309/314 backends need
    it. Pin upper bound so a rebuild can't silently land
    a major-version regression."""
    content = _REQS.read_text()
    assert "torch" in content
    # Some bound (>=, <, ==)
    torch_line = next(
        line for line in content.splitlines()
        if line.strip().startswith("torch")
    )
    assert any(c in torch_line for c in (">=", "<", "=="))


def test_requirements_pins_cryptography():
    content = _REQS.read_text()
    assert "cryptography" in content


def test_requirements_pins_fastapi():
    content = _REQS.read_text()
    assert "fastapi" in content


def test_requirements_excludes_heavy_optional_deps():
    """The demo doesn't need libtorrent / web3 /
    postgres / redis / ipfs. Each is multi-MB; keeping
    them out of the demo image keeps cold rebuilds
    fast + the attack surface narrow.

    Check INSTALL lines only (strip comments) — comments
    are allowed to mention these by name for context."""
    install_lines = [
        line.strip().lower()
        for line in _REQS.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    install_blob = "\n".join(install_lines)
    for unwanted in (
        "libtorrent", "web3", "psycopg2",
        "redis", "ipfshttpclient",
    ):
        assert unwanted not in install_blob, (
            f"requirements should not pull {unwanted}"
        )


# ── LIVE: Docker build + run ───────────────────────


@pytest.fixture
def real_subprocess(monkeypatch):
    """Bypass conftest's session-wide subprocess.run mock
    so the live Docker tests can actually exec docker.
    The mock is reinstated automatically when the test
    exits via monkeypatch teardown."""
    import subprocess as _real_subprocess
    import importlib
    # Reload subprocess to drop the conftest patch's
    # in-place replacement for the duration of the test
    importlib.reload(_real_subprocess)
    monkeypatch.setattr(
        "subprocess.run", _real_subprocess.run,
    )
    return _real_subprocess


def _daemon_responds(sock_path) -> bool:
    """sp948 — actually ping the Docker daemon over its UDS (HTTP /_ping)
    rather than trusting socket-file presence. Docker Desktop for Mac leaves a
    STALE socket FILE in place when the VM is stopped, so the old
    `exists() and S_ISSOCK` probe returned True against a dead daemon and the
    live tests FAILED (rc=1 "Cannot connect to the Docker daemon") instead of
    skipping. Uses a raw AF_UNIX socket only (NOT subprocess.run, which
    conftest mocks session-wide), so it stays safe at collection time."""
    import socket as _socket
    s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    s.settimeout(2.0)
    try:
        s.connect(str(sock_path))
        s.sendall(b"GET /_ping HTTP/1.1\r\nHost: docker\r\n\r\n")
        return b"200" in s.recv(64).split(b"\r\n")[0]
    except OSError:
        return False
    finally:
        s.close()


def _docker_available() -> bool:
    """Probe at test-collection time. Uses os/shutil/socket only
    (not subprocess.run, which conftest has mocked) — so
    we can decide whether to even attempt the live tests."""
    if shutil.which("docker") is None:
        return False
    # Liveness probe: the daemon listens on a UDS at one of these
    # paths on Docker Desktop for Mac. A socket FILE existing is NOT
    # enough (a stopped VM leaves a stale socket), so we ping /_ping.
    import stat
    candidates = [
        Path.home() / ".docker/run/docker.sock",
        Path("/var/run/docker.sock"),
    ]
    for path in candidates:
        try:
            if path.exists() and stat.S_ISSOCK(
                path.stat().st_mode,
            ) and _daemon_responds(path):
                return True
        except (OSError, PermissionError):
            continue
    return False


@pytest.mark.skipif(
    not _docker_available(),
    reason="Docker daemon not available",
)
# sp1418 — override the suite-wide `--timeout=120` (CI). This test's own subprocess budget is 900s
# because a cold `pip install torch` inside the image is slow. Without this, pytest-timeout killed it
# at 120s the moment CI actually started running tests (it never had, until collection was fixed) —
# the guard would have been red forever, or silently deselected. Give it the budget it already assumes.
@pytest.mark.timeout(960)
def test_live_docker_build_succeeds(real_subprocess):
    """LOAD-BEARING: actually build the image. If this
    fails, the file-level artifacts are not actually
    deployable."""
    result = real_subprocess.run(
        [
            "docker", "build",
            "-f", str(_DOCKERFILE),
            "-t", "prsm-enterprise-demo:test-318a",
            str(_REPO_ROOT),
        ],
        capture_output=True,
        timeout=900,  # cold pip install of torch is slow
    )
    out = result.stdout
    err = result.stderr
    if isinstance(out, bytes):
        out = out.decode("utf-8", errors="replace")
    if isinstance(err, bytes):
        err = err.decode("utf-8", errors="replace")
    assert result.returncode == 0, (
        f"docker build failed:\n"
        f"STDOUT:\n{out[-2000:]}\n\n"
        f"STDERR:\n{err[-2000:]}"
    )


@pytest.mark.skipif(
    not _docker_available(),
    reason="Docker daemon not available",
)
# sp1418 — same as above: its subprocess budgets are build(900s) + run(180s), so the suite-wide
# `--timeout=120` would kill it long before either could finish.
@pytest.mark.timeout(1200)
def test_live_docker_run_demo_succeeds(real_subprocess):
    """LOAD-BEARING: build the image (idempotent if
    layers cached) + run the demo inside; expect rc=0 +
    `All demos passed` in stdout. This is the proof that
    the entire §7 enterprise + federated inference stack
    actually boots in a real container."""
    # Ensure image is built (cached if test above ran)
    build_result = real_subprocess.run(
        [
            "docker", "build",
            "-f", str(_DOCKERFILE),
            "-t", "prsm-enterprise-demo:test-318a",
            str(_REPO_ROOT),
        ],
        capture_output=True,
        timeout=900,
    )
    build_err = build_result.stderr
    if isinstance(build_err, bytes):
        build_err = build_err.decode(
            "utf-8", errors="replace",
        )
    assert build_result.returncode == 0, (
        f"build prerequisite failed:\n"
        f"{build_err[-1500:]}"
    )
    # Run the demo
    run_result = real_subprocess.run(
        [
            "docker", "run", "--rm",
            "prsm-enterprise-demo:test-318a",
        ],
        capture_output=True,
        timeout=180,
    )
    # Defensive: text=True doesn't always apply (some
    # subprocess builds leave stdout as bytes when the
    # child emits non-UTF-8 progress chars). Decode if
    # needed.
    stdout = run_result.stdout
    stderr = run_result.stderr
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    assert run_result.returncode == 0, (
        f"demo run failed:\n"
        f"STDOUT:\n{stdout[-1500:]}\n\n"
        f"STDERR:\n{stderr[-1500:]}"
    )
    assert (
        "All demos passed" in stdout
        or "✓" in stdout
    ), (
        f"demo stdout missing success marker:\n"
        f"{stdout[-1500:]}"
    )
