"""Sprint 1099 — fold Domain-03 review F1 (first brick): bind the INPUT prompt into
the signed InferenceReceipt.

The review found the production receipt commits to ``output_hash`` (the OUTPUT) but
NOTHING ties the signed receipt to the prompt the caller actually sent — so a head
node can compute the (cheaper / canned) output for a DIFFERENT prompt and the receipt
still verifies (signature valid, output_hash matches the substituted output). This
brick adds ``prompt_hash`` = sha256(prompt) to the receipt + folds it into the signing
payload (conditional-append, byte-equivalent when absent, per the sprint-297 pattern),
so a caller can recompute sha256(their prompt) and confirm the receipt commits to THEIR
request. (The deeper per-stage activation hash chain — F1's full fix — is a follow-on
brick; this closes the prompt-substitution half.)
"""
from __future__ import annotations

import hashlib
from decimal import Decimal

import pytest

from prsm.compute.inference.models import ContentTier, InferenceReceipt
from prsm.compute.tee.models import PrivacyLevel, TEEType


def _receipt(**over):
    base = dict(
        job_id="job-1", request_id="req-1", model_id="gpt2",
        content_tier=ContentTier.A, privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0, tee_type=TEEType.SOFTWARE,
        tee_attestation=b"PRSM-MS-ATT-V1:{}", output_hash=b"\xab" * 32,
        duration_seconds=1.0, cost_ftns=Decimal("0.12"), settler_node_id="a" * 32,
    )
    base.update(over)
    return InferenceReceipt(**base)


def test_prompt_hash_changes_signing_payload():
    """Binding the prompt MUST change the signed bytes — otherwise a substituted
    prompt would verify identically."""
    ph = hashlib.sha256(b"what is 2+2?").digest()
    bound = _receipt(prompt_hash=ph)
    unbound = _receipt()  # no prompt_hash
    assert bound.signing_payload() != unbound.signing_payload()


def test_different_prompts_produce_different_signed_bytes():
    """Two receipts identical except for the prompt they bind must NOT share signed
    bytes — this is the property that defeats prompt substitution."""
    a = _receipt(prompt_hash=hashlib.sha256(b"prompt A").digest())
    b = _receipt(prompt_hash=hashlib.sha256(b"prompt B").digest())
    assert a.signing_payload() != b.signing_payload()


def test_prompt_hash_absent_is_byte_identical_to_pre_1099():
    """Conditional encoding: a receipt with prompt_hash=None must produce the EXACT
    pre-1099 signing bytes so legacy receipts + verifiers keep working."""
    legacy_parts = _receipt().signing_payload()
    # nothing about prompt_hash leaks into the payload when it's None
    assert b"prompt_hash" not in legacy_parts


def test_prompt_hash_round_trips_through_dict():
    """to_dict / from_dict must restore prompt_hash so the verify endpoint
    reconstructs the EXACT signed bytes."""
    ph = hashlib.sha256(b"round trip me").digest()
    r = _receipt(prompt_hash=ph)
    rebuilt = InferenceReceipt.from_dict(r.to_dict())
    assert rebuilt.prompt_hash == ph
    assert rebuilt.signing_payload() == r.signing_payload()


def test_build_signed_receipt_binds_the_request_prompt():
    """The production build path must compute prompt_hash from the request prompt and
    the resulting receipt must verify AND commit to that exact prompt."""
    from prsm.compute.inference.models import InferenceRequest
    from prsm.compute.inference.parallax_executor import ChainExecutionResult
    from prsm.node.identity import generate_node_identity
    from prsm.compute.inference.receipt import verify_receipt

    identity = generate_node_identity("builder-node")

    # Minimal executor shell exposing _build_signed_receipt with a real identity.
    from prsm.compute.inference.parallax_executor import ParallaxScheduledExecutor
    executor = ParallaxScheduledExecutor.__new__(ParallaxScheduledExecutor)
    executor._identity = identity

    prompt = "the capital of France is"
    request = InferenceRequest(prompt=prompt, model_id="gpt2", budget_ftns=Decimal("1"))
    outcome = ChainExecutionResult(
        output=" Paris", duration_seconds=0.1,
        tee_attestation=b"att", tee_type=TEEType.SOFTWARE, epsilon_spent=0.0,
    )
    receipt = executor._build_signed_receipt(request=request, cost=Decimal("0.1"), outcome=outcome)

    assert receipt.prompt_hash == hashlib.sha256(prompt.encode("utf-8")).digest()
    assert verify_receipt(receipt, identity=identity)


def test_substituted_prompt_fails_verification():
    """The attack F1 closes: a signed receipt for prompt A must NOT verify once its
    prompt_hash is swapped to prompt B — the signature is over the bound prompt."""
    import dataclasses
    from prsm.node.identity import generate_node_identity
    from prsm.compute.inference.receipt import sign_receipt, verify_receipt

    identity = generate_node_identity("settler-node")
    honest = sign_receipt(
        _receipt(prompt_hash=hashlib.sha256(b"prompt A").digest()), identity)
    assert verify_receipt(honest, identity=identity)  # honest receipt verifies

    # attacker keeps the signature but swaps the bound prompt to B
    tampered = dataclasses.replace(
        honest, prompt_hash=hashlib.sha256(b"prompt B").digest())
    assert not verify_receipt(tampered, identity=identity)  # signature no longer matches


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
