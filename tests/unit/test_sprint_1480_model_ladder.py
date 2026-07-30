"""Sprint 1480 — hardware-aware default local model ("day-one useful answer").

User-readiness re-assessment (wf_58eae474): a first-time user who gets past the
packaging defect still lands on ``DEFAULT_LOCAL_MODEL = "distilgpt2"`` — a 6-layer
2019 BASE model — on a 64GB Mac or an A10 alike, because there was zero
hardware-aware selection anywhere in the repo. The only escape,
``PRSM_LOCAL_INFERENCE_MODEL``, appeared in exactly two places repo-wide (both
inside node.py) — not in the README, the CLI help, or the wizard. So "PRSM does
real inference" was technically true and practically useless.

``resolve_default_local_model`` is deliberately PURE so the whole ladder is
testable without hardware. Sizing is conservative on purpose: picking a model that
OOMs mid-generation is a worse first run than picking one size down.
"""
from __future__ import annotations

import pytest

from prsm.compute.inference.local_inference import (
    APPROX_DOWNLOAD_GB,
    DEFAULT_LOCAL_MODEL,
    LADDER_05B,
    LADDER_3B,
    LADDER_7B,
    _KNOWN_MODELS,
    detect_local_hardware,
    resolve_default_local_model,
)


@pytest.mark.parametrize(
    "device,vram,ram,expected",
    [
        # CUDA sizes off VRAM (fp16).
        ("cuda", 80.0, 512.0, LADDER_7B),    # A100
        ("cuda", 24.0, 64.0, LADDER_7B),     # A10 / 3090 / 4090
        ("cuda", 16.0, 64.0, LADDER_7B),     # exact boundary
        ("cuda", 12.0, 32.0, LADDER_3B),     # 3080 Ti
        ("cuda", 8.0, 16.0, LADDER_3B),      # exact boundary
        ("cuda", 6.0, 16.0, LADDER_05B),     # small GPU still beats CPU
        ("cuda", 2.0, 8.0, DEFAULT_LOCAL_MODEL),
        # MPS is unified memory -> size off system RAM, not VRAM.
        ("mps", None, 128.0, LADDER_3B),     # M3 Max
        ("mps", None, 32.0, LADDER_3B),      # exact boundary
        ("mps", None, 24.0, LADDER_05B),
        ("mps", None, 16.0, LADDER_05B),     # exact boundary
        ("mps", None, 8.0, DEFAULT_LOCAL_MODEL),
        # CPU.
        ("cpu", None, 64.0, LADDER_05B),
        ("cpu", None, 8.0, LADDER_05B),      # exact boundary
        ("cpu", None, 4.0, DEFAULT_LOCAL_MODEL),
    ],
)
def test_ladder_matrix(device, vram, ram, expected):
    assert resolve_default_local_model(
        device=device, vram_gb=vram, free_ram_gb=ram) == expected


def test_unknown_capacity_is_conservative_never_optimistic():
    """★ Detection failure must NOT guess big — an OOM on first run is the worst
    possible first impression. None/0 capacity falls to the last-resort model."""
    for device in ("cuda", "mps", "cpu"):
        assert resolve_default_local_model(
            device=device, vram_gb=None, free_ram_gb=None) == DEFAULT_LOCAL_MODEL
        assert resolve_default_local_model(
            device=device, vram_gb=0, free_ram_gb=0) == DEFAULT_LOCAL_MODEL


def test_no_hardware_gets_a_toy_model_when_it_could_hold_an_instruct_model():
    """★ The actual regression this sprint fixes: capable hardware must NOT land on
    distilgpt2. Every mainstream dev box should get a real instruct model."""
    capable = [
        ("cuda", 24.0, 64.0),   # cloud GPU
        ("mps", None, 32.0),    # 32GB Mac
        ("cpu", None, 16.0),    # ordinary laptop/server
    ]
    for device, vram, ram in capable:
        picked = resolve_default_local_model(
            device=device, vram_gb=vram, free_ram_gb=ram)
        assert picked != DEFAULT_LOCAL_MODEL, (
            f"{device}/vram={vram}/ram={ram} still defaults to the toy model")
        assert "Instruct" in picked, f"{picked} is not an instruct model"


def test_every_ladder_model_has_layer_count_and_download_size():
    """Each ladder model needs a _KNOWN_MODELS fallback (used by _derive_model_info
    when the HF config isn't present) and a download size for the startup log —
    otherwise a multi-GB fetch is a silent surprise."""
    for m in (LADDER_05B, LADDER_3B, LADDER_7B, DEFAULT_LOCAL_MODEL):
        assert m in _KNOWN_MODELS, f"{m} missing from _KNOWN_MODELS"
        assert _KNOWN_MODELS[m] > 0
        assert m in APPROX_DOWNLOAD_GB, f"{m} missing a download-size estimate"
        assert APPROX_DOWNLOAD_GB[m] > 0


def test_ladder_is_monotonic_in_capacity():
    """More capacity must never select a SMALLER model."""
    order = {DEFAULT_LOCAL_MODEL: 0, LADDER_05B: 1, LADDER_3B: 2, LADDER_7B: 3}
    prev = -1
    for vram in (0, 2, 4, 6, 8, 12, 16, 24, 48, 80):
        rank = order[resolve_default_local_model(device="cuda", vram_gb=vram, free_ram_gb=64)]
        assert rank >= prev, f"cuda vram={vram} selected a smaller model than less VRAM"
        prev = rank
    prev = -1
    for ram in (0, 4, 8, 16, 24, 32, 64, 128):
        rank = order[resolve_default_local_model(device="mps", free_ram_gb=ram)]
        assert rank >= prev, f"mps ram={ram} selected a smaller model than less RAM"
        prev = rank


def test_detect_local_hardware_never_raises_and_is_usable():
    """Detection must degrade, never explode — it runs on the node startup path."""
    hw = detect_local_hardware()
    assert hw["device"] in {"cuda", "mps", "cpu"}
    # Whatever it returns must be directly consumable by the resolver.
    picked = resolve_default_local_model(
        device=hw["device"], vram_gb=hw.get("vram_gb"), free_ram_gb=hw.get("ram_gb"))
    assert picked in _KNOWN_MODELS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
