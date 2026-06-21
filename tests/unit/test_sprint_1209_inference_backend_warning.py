"""Sprint 1209 — accurate 'no inference backend' startup warning + jinja2 in [ml].

The Lambda GPU bench surfaced two operator-UX papercuts:
  1. The node warned "inference will return mock responses" whenever no Anthropic/OpenAI
     key was set — even with PRSM_INFERENCE_EXECUTOR=local serving REAL inference. The
     warning now fires only when NEITHER a real LLM-API backend NOR a real executor exists.
  2. Instruct chat templates need jinja2>=3.1.0; a fresh box only had the distro's older
     jinja2 → inference failed. It's now declared in the [ml] extras.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from prsm.cli import _should_warn_no_inference_backend


class _RealLocalExecutor:  # name has no "mock"
    pass


class _MockExecutor:
    pass


def test_no_warn_when_real_llm_backend():
    assert _should_warn_no_inference_backend(True, None) is False


def test_no_warn_when_real_local_executor_wired():
    # the exact GPU-bench case: no API key, but a real local executor serves inference
    assert _should_warn_no_inference_backend(False, _RealLocalExecutor()) is False


def test_warn_when_no_backend_and_no_executor():
    assert _should_warn_no_inference_backend(False, None) is True


def test_warn_when_only_a_mock_executor():
    assert _should_warn_no_inference_backend(False, _MockExecutor()) is True


def test_ml_extras_declare_jinja2():
    # guard: chat-template rendering (sp1204/sp1208) needs jinja2>=3.1.0 installed.
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    ml_block = text.split("ml = [", 1)[1].split("]", 1)[0]
    assert "jinja2" in ml_block and "3.1" in ml_block


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
