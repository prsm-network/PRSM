"""Sprint 1457 — SDK receipt_verified must bind the prompt + output, not just the signature.

The inference-receipt-verification audit found the SDK's pay_and_infer set receipt_verified from
the settler SIGNATURE alone (verify_receipt), skipping the InferenceReceipt docstring's caller
steps 3-4: confirm the receipt commits to THIS prompt (sp1099 prompt_hash) and the OUTPUT received
(output_hash). So receipt_verified=True overclaimed — "the settler signed a receipt" is not "for MY
prompt and the output I got"; a head node could return a canned answer for a different prompt and the
signature still verifies. _attach_receipt_verified now runs all three checks (signature AND prompt_hash
AND output_hash). These tests mock verify_receipt (the signature layer) to isolate the new binding.
"""
from __future__ import annotations

import hashlib
from unittest.mock import patch

from prsm.compute.inference.models import (
    ContentTier, InferenceReceipt, PrivacyLevel, TEEType)
from prsm.sdk.client import PRSMClient

_PROMPT = "what is the capital of France?"
_OUTPUT = " The capital of France is Paris."


def _receipt_dict(*, prompt: str = _PROMPT, output: str = _OUTPUT):
    r = InferenceReceipt(
        job_id="j", request_id="r", model_id="gpt2",
        content_tier=ContentTier.A, privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0, tee_type=TEEType.NONE, tee_attestation=b"",
        output_hash=hashlib.sha256(output.encode("utf-8")).digest(),
        duration_seconds=0.1, cost_ftns=0.5,
        prompt_hash=hashlib.sha256(prompt.encode("utf-8")).digest(),
    )
    return r.to_dict()


def _attach(result):
    return PRSMClient._attach_receipt_verified(result, _PROMPT, "pk-b64")


def test_receipt_verified_true_when_signature_prompt_output_all_match():
    with patch("prsm.compute.inference.receipt.verify_receipt", return_value=True):
        res = {"output": _OUTPUT, "receipt": _receipt_dict()}
        _attach(res)
    assert res["receipt_verified"] is True


def test_receipt_verified_false_when_output_mismatches_even_if_signed():
    # ★ the core gap: the settler SIGNATURE verifies, but the served output does NOT match the
    # receipt's output_hash — a canned/wrong answer. Pre-fix this was reported as verified.
    with patch("prsm.compute.inference.receipt.verify_receipt", return_value=True):
        res = {"output": "a DIFFERENT canned answer", "receipt": _receipt_dict()}
        _attach(res)
    assert res["receipt_verified"] is False


def test_receipt_verified_false_when_prompt_mismatches():
    # The receipt commits (via prompt_hash) to a prompt other than the one the caller sent.
    with patch("prsm.compute.inference.receipt.verify_receipt", return_value=True):
        res = {"output": _OUTPUT, "receipt": _receipt_dict(prompt="an unrelated prompt")}
        _attach(res)
    assert res["receipt_verified"] is False


def test_receipt_verified_false_when_output_field_missing():
    with patch("prsm.compute.inference.receipt.verify_receipt", return_value=True):
        res = {"receipt": _receipt_dict()}   # no "output" to confirm against
        _attach(res)
    assert res["receipt_verified"] is False


def test_receipt_verified_false_when_signature_fails():
    with patch("prsm.compute.inference.receipt.verify_receipt", return_value=False):
        res = {"output": _OUTPUT, "receipt": _receipt_dict()}
        _attach(res)
    assert res["receipt_verified"] is False


def test_receipt_verified_binds_output_when_no_prompt_hash():
    # A receipt without prompt_hash (pre-sp1099) still gets its OUTPUT confirmed (step 4),
    # so a mismatched output is still rejected.
    with patch("prsm.compute.inference.receipt.verify_receipt", return_value=True):
        d = _receipt_dict()
        d.pop("prompt_hash", None)
        res = {"output": "wrong output", "receipt": d}
        _attach(res)
    assert res["receipt_verified"] is False
