"""Sprint 802 — `prsm compute infer` user-facing inference CLI.

Pre-802 users had to curl /compute/inference manually + parse
the response themselves. Existing `prsm compute run --prompt`
hits `/compute/query` (NWTN orchestrator), NOT the parallax
verifiable-inference path that produces signed receipts.

Sprint 802 closes the gap with a first-class CLI:

  prsm compute infer --prompt "..."
                    [--model gpt2]
                    [--max-tokens 100]
                    [--budget 1.0]
                    [--privacy-tier none|standard|high|maximum]
                    [--content-tier A|B|C]
                    [--api-url http://...]
                    [--format text|json]
                    [--verify-receipt]

POSTs the verifiable-inference body shape to /compute/inference,
displays output + cost + receipt summary. With --verify-receipt
the CLI re-runs the canonical signing-payload check locally
(sprint 706/707's `verify_receipt` standalone path) so the user
sees PASS/FAIL on the cryptographic chain.

Exit codes:
  0 — inference succeeded (and verify passed when requested)
  1 — daemon answered but inference failed (503, 400, etc.) OR
      verify failed
  2 — daemon unreachable

Pin tests:
- Command registered under `compute` group.
- Body schema matches /compute/inference's POST input.
- Default model_id = "gpt2"; default tier = "A"; default
  privacy = "none"; default budget = 1.0; default max_tokens
  in a reasonable small range.
- Text mode renders output + cost.
- JSON mode returns full server payload.
- 503 daemon error → exit 1 with detail.
- Unreachable → exit 2.
- --verify-receipt PASS path → exit 0 + "verified".
- --verify-receipt FAIL path (tampered receipt) → exit 1 +
  "verification failed".
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def _invoke(args):
    from prsm.cli import compute as _compute_group
    runner = CliRunner()
    return runner.invoke(_compute_group, ["infer"] + list(args))


# ---- Command registration --------------------------------------


def test_infer_command_registered():
    from prsm.cli import compute as _compute_group
    cmd_names = [c.name for c in _compute_group.commands.values()]
    assert "infer" in cmd_names


# ---- Body schema ------------------------------------------------


def _good_response():
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "success": True,
        "output": " the",
        "ftns_charged": "0.25",
        "receipt": {
            "job_id": "j1",
            "settler_signature": "0xabcd",
            "output_hash": "deadbeef",
        },
    }
    return r


def test_body_uses_correct_field_names():
    """The /compute/inference endpoint expects:
      prompt, model_id, budget_ftns, privacy_tier,
      content_tier, max_tokens
    Pin the exact field names so future endpoint refactors
    surface here first."""
    with patch("httpx.post", return_value=_good_response()) as mp:
        result = _invoke([
            "--prompt", "Hi", "--format", "json",
        ])
    assert result.exit_code == 0, result.output
    body = mp.call_args.kwargs.get("json") or {}
    assert "prompt" in body
    assert "model_id" in body
    assert "budget_ftns" in body
    assert "privacy_tier" in body
    assert "content_tier" in body
    assert "max_tokens" in body
    # Wrong field names (regression guard against typos)
    assert "model" not in body
    assert "budget" not in body
    assert "privacy" not in body


def _empty_output_response():
    # sp1386 — the node answers 200 but with no output (e.g. an unavailable model_id).
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {"success": False, "output": "", "error": "Unknown model_id: nope"}
    return r


def test_empty_output_fails_loud_text_mode():
    """sp1386 — an unavailable model used to print a blank 'Output:' and exit 0. Now it surfaces the
    error and exits 1 so a new user isn't silently dead-ended."""
    with patch("httpx.post", return_value=_empty_output_response()):
        result = _invoke(["--prompt", "Hi", "--model", "nope"])
    assert result.exit_code == 1
    assert "no output" in result.output.lower()
    assert "Unknown model_id" in result.output


def test_default_values():
    with patch("httpx.post", return_value=_good_response()) as mp:
        _invoke(["--prompt", "Hi", "--format", "json"])
    body = mp.call_args.kwargs.get("json") or {}
    assert body["model_id"] == "distilgpt2"   # sp1386 — the node's actual default local model (was gpt2, which isn't served)
    assert body["content_tier"] == "A"
    assert body["privacy_tier"] == "none"
    assert body["budget_ftns"] == 1.0
    assert isinstance(body["max_tokens"], int)
    assert body["max_tokens"] >= 1


def test_overrides_passed_through():
    with patch("httpx.post", return_value=_good_response()) as mp:
        _invoke([
            "--prompt", "Capital of France?",
            "--model", "llama-7b",
            "--max-tokens", "50",
            "--budget", "3.5",
            "--privacy-tier", "standard",
            "--content-tier", "B",
            "--format", "json",
        ])
    body = mp.call_args.kwargs.get("json") or {}
    assert body["prompt"] == "Capital of France?"
    assert body["model_id"] == "llama-7b"
    assert body["max_tokens"] == 50
    assert body["budget_ftns"] == 3.5
    assert body["privacy_tier"] == "standard"
    assert body["content_tier"] == "B"


# ---- Output modes ----------------------------------------------


def test_text_mode_shows_output_and_cost():
    with patch("httpx.post", return_value=_good_response()):
        result = _invoke([
            "--prompt", "Hi", "--format", "text",
        ])
    assert result.exit_code == 0
    assert "the" in result.output
    # Cost surface
    assert "0.25" in result.output or "FTNS" in result.output


def test_json_mode_returns_payload():
    with patch("httpx.post", return_value=_good_response()):
        result = _invoke([
            "--prompt", "Hi", "--format", "json",
        ])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["success"] is True
    assert data["output"] == " the"


# ---- Error paths -----------------------------------------------


def test_daemon_503_exits_1():
    fake = MagicMock()
    fake.status_code = 503
    fake.text = '{"detail":"Inference executor not initialized"}'
    with patch("httpx.post", return_value=fake):
        result = _invoke([
            "--prompt", "Hi", "--format", "text",
        ])
    assert result.exit_code == 1
    assert "503" in result.output or (
        "executor" in result.output.lower()
    )


def test_unreachable_daemon_exits_2():
    with patch(
        "httpx.post",
        side_effect=ConnectionError("connection refused"),
    ):
        result = _invoke([
            "--prompt", "Hi", "--format", "text",
        ])
    assert result.exit_code == 2
    assert "unreachable" in result.output.lower()


# ---- --verify-receipt --------------------------------------


def _good_signed_response(receipt_dict):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "success": True,
        "output": " the",
        "ftns_charged": "0.25",
        "receipt": receipt_dict,
    }
    return r


def _build_signed_receipt_dict():
    """Build a real signed receipt dict via sprint-413 path."""
    from decimal import Decimal
    from prsm.compute.inference.models import (
        InferenceReceipt, ContentTier,
    )
    from prsm.compute.tee.models import PrivacyLevel, TEEType
    from prsm.compute.inference.receipt import sign_receipt
    from prsm.node.identity import generate_node_identity

    identity = generate_node_identity("verify-test")
    receipt = InferenceReceipt(
        job_id="j1", request_id="r1", model_id="gpt2",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=b"att", output_hash=b"\xde\xad",
        duration_seconds=1.0, cost_ftns=Decimal("0.25"),
        settler_node_id="",
    )
    signed = sign_receipt(receipt, identity)
    pubkey_b64 = identity.public_key_b64
    return signed.to_dict(), pubkey_b64


def test_verify_receipt_pass(tmp_path):
    """--verify-receipt with a valid signed receipt → exit 0 +
    'verified'."""
    receipt_dict, pubkey_b64 = _build_signed_receipt_dict()
    with patch(
        "httpx.post",
        return_value=_good_signed_response(receipt_dict),
    ):
        result = _invoke([
            "--prompt", "Hi", "--verify-receipt",
            "--verify-pubkey-b64", pubkey_b64,
            "--format", "text",
        ])
    assert result.exit_code == 0, result.output
    assert "verified" in result.output.lower()


def test_verify_receipt_fail():
    """--verify-receipt against the WRONG pubkey → exit 1."""
    receipt_dict, _ = _build_signed_receipt_dict()
    # Generate an unrelated pubkey
    from prsm.node.identity import generate_node_identity
    wrong_pubkey = generate_node_identity("other").public_key_b64

    with patch(
        "httpx.post",
        return_value=_good_signed_response(receipt_dict),
    ):
        result = _invoke([
            "--prompt", "Hi", "--verify-receipt",
            "--verify-pubkey-b64", wrong_pubkey,
            "--format", "text",
        ])
    assert result.exit_code == 1
    assert (
        "verification failed" in result.output.lower()
        or "verify failed" in result.output.lower()
    )
