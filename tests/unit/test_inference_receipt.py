"""Tests for InferenceReceipt Ed25519 sign/verify (Phase 3.x.1 Task 2)."""

import dataclasses
from decimal import Decimal

import pytest

from prsm.compute.inference import (
    ContentTier,
    InferenceReceipt,
    is_signed,
    sign_receipt,
    verify_receipt,
)
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import NodeIdentity, generate_node_identity


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def identity() -> NodeIdentity:
    """A freshly-generated node identity for signing tests."""
    return generate_node_identity(display_name="test-settler")


@pytest.fixture
def other_identity() -> NodeIdentity:
    """A second identity to verify cross-key rejection."""
    return generate_node_identity(display_name="other-node")


@pytest.fixture
def unsigned_receipt() -> InferenceReceipt:
    """A baseline unsigned receipt for sign/verify tests."""
    return InferenceReceipt(
        job_id="job-abc",
        request_id="req-xyz",
        model_id="mock-llama-3-8b",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.STANDARD,
        epsilon_spent=8.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=b"\x01\x02\x03\x04",
        output_hash=b"\xaa" * 32,
        duration_seconds=1.5,
        cost_ftns=Decimal("0.5"),
        # settler_signature + settler_node_id default to empty bytes / empty string
    )


# ── is_signed ───────────────────────────────────────────────────────────────


class TestIsSigned:
    def test_unsigned_receipt(self, unsigned_receipt):
        assert not is_signed(unsigned_receipt)

    def test_signed_receipt(self, unsigned_receipt, identity):
        signed = sign_receipt(unsigned_receipt, identity)
        assert is_signed(signed)

    def test_partial_signature_is_unsigned(self, unsigned_receipt):
        # Partially-populated (signature without node_id) — should NOT count as signed
        partial = dataclasses.replace(
            unsigned_receipt, settler_signature=b"\x01" * 64
        )
        assert not is_signed(partial)


# ── sign_receipt ────────────────────────────────────────────────────────────


class TestSignReceipt:
    def test_returns_new_receipt(self, unsigned_receipt, identity):
        signed = sign_receipt(unsigned_receipt, identity)
        # Original unchanged (frozen dataclass)
        assert unsigned_receipt.settler_signature == b""
        assert unsigned_receipt.settler_node_id == ""
        # New receipt has both fields populated
        assert signed.settler_signature
        assert signed.settler_node_id == identity.node_id

    def test_signature_is_64_bytes(self, unsigned_receipt, identity):
        signed = sign_receipt(unsigned_receipt, identity)
        # Ed25519 signatures are exactly 64 bytes
        assert len(signed.settler_signature) == 64

    def test_node_id_matches_identity(self, unsigned_receipt, identity):
        signed = sign_receipt(unsigned_receipt, identity)
        assert signed.settler_node_id == identity.node_id

    def test_other_fields_preserved(self, unsigned_receipt, identity):
        signed = sign_receipt(unsigned_receipt, identity)
        assert signed.job_id == unsigned_receipt.job_id
        assert signed.model_id == unsigned_receipt.model_id
        assert signed.content_tier == unsigned_receipt.content_tier
        assert signed.privacy_tier == unsigned_receipt.privacy_tier
        assert signed.epsilon_spent == unsigned_receipt.epsilon_spent
        assert signed.cost_ftns == unsigned_receipt.cost_ftns
        assert signed.tee_attestation == unsigned_receipt.tee_attestation
        assert signed.output_hash == unsigned_receipt.output_hash

    def test_signing_is_deterministic_with_node_id(self, unsigned_receipt, identity):
        # Sign the same receipt twice with the same key — Ed25519 is deterministic
        s1 = sign_receipt(unsigned_receipt, identity)
        s2 = sign_receipt(unsigned_receipt, identity)
        assert s1.settler_signature == s2.settler_signature

    def test_different_identities_produce_different_signatures(
        self, unsigned_receipt, identity, other_identity
    ):
        s1 = sign_receipt(unsigned_receipt, identity)
        s2 = sign_receipt(unsigned_receipt, other_identity)
        assert s1.settler_signature != s2.settler_signature
        assert s1.settler_node_id != s2.settler_node_id


# ── verify_receipt — happy path ─────────────────────────────────────────────


class TestVerifyReceiptHappyPath:
    def test_verify_with_identity(self, unsigned_receipt, identity):
        signed = sign_receipt(unsigned_receipt, identity)
        assert verify_receipt(signed, identity=identity)

    def test_verify_with_public_key_b64(self, unsigned_receipt, identity):
        signed = sign_receipt(unsigned_receipt, identity)
        assert verify_receipt(signed, public_key_b64=identity.public_key_b64)

    def test_verify_after_serialization_roundtrip(self, unsigned_receipt, identity):
        # Receipt is signed, serialized, deserialized — signature should still verify
        signed = sign_receipt(unsigned_receipt, identity)
        restored = InferenceReceipt.from_dict(signed.to_dict())
        assert verify_receipt(restored, public_key_b64=identity.public_key_b64)

    def test_verify_returns_bool(self, unsigned_receipt, identity):
        signed = sign_receipt(unsigned_receipt, identity)
        result = verify_receipt(signed, identity=identity)
        assert isinstance(result, bool)
        assert result is True


# ── verify_receipt — rejection cases ────────────────────────────────────────


class TestVerifyReceiptRejection:
    def test_unsigned_receipt_rejected(self, unsigned_receipt, identity):
        # No signature on receipt → verify returns False
        assert not verify_receipt(unsigned_receipt, identity=identity)

    def test_no_key_falls_back_to_embedded_else_rejected(self, unsigned_receipt, identity):
        # sp1255 — with no key supplied, verify falls back to the pubkey EMBEDDED in
        # the receipt (bound to settler_node_id), so a genuine receipt verifies offline.
        signed = sign_receipt(unsigned_receipt, identity)
        assert verify_receipt(signed) is True
        # ...but a receipt with NO embedded pubkey AND no key supplied is rejected.
        stripped = dataclasses.replace(signed, settler_pubkey_b64=None)
        assert not verify_receipt(stripped)

    def test_wrong_public_key_rejected(
        self, unsigned_receipt, identity, other_identity
    ):
        # Sign with identity, attempt verify with other_identity's public key
        signed = sign_receipt(unsigned_receipt, identity)
        assert not verify_receipt(
            signed, public_key_b64=other_identity.public_key_b64
        )

    def test_wrong_identity_rejected(
        self, unsigned_receipt, identity, other_identity
    ):
        signed = sign_receipt(unsigned_receipt, identity)
        assert not verify_receipt(signed, identity=other_identity)

    def test_garbage_public_key_rejected(self, unsigned_receipt, identity):
        signed = sign_receipt(unsigned_receipt, identity)
        # Malformed public key should not cause exception, just rejection
        assert not verify_receipt(signed, public_key_b64="not-a-real-key!!!")


# ── verify_receipt — tampering detection (every signed field) ───────────────


class TestVerifyReceiptTampering:
    """Each test mutates exactly one signed field and confirms verify() returns False.

    This is the load-bearing security property: if the verifier accepts a
    tampered receipt, the entire ``verifiable inference`` claim is broken.
    """

    def _signed(self, unsigned_receipt, identity) -> InferenceReceipt:
        return sign_receipt(unsigned_receipt, identity)

    def test_tamper_job_id(self, unsigned_receipt, identity):
        signed = self._signed(unsigned_receipt, identity)
        tampered = dataclasses.replace(signed, job_id="job-DIFFERENT")
        assert not verify_receipt(tampered, identity=identity)

    def test_tamper_request_id(self, unsigned_receipt, identity):
        signed = self._signed(unsigned_receipt, identity)
        tampered = dataclasses.replace(signed, request_id="req-DIFFERENT")
        assert not verify_receipt(tampered, identity=identity)

    def test_tamper_model_id(self, unsigned_receipt, identity):
        signed = self._signed(unsigned_receipt, identity)
        tampered = dataclasses.replace(signed, model_id="other-model")
        assert not verify_receipt(tampered, identity=identity)

    def test_tamper_content_tier(self, unsigned_receipt, identity):
        signed = self._signed(unsigned_receipt, identity)
        tampered = dataclasses.replace(signed, content_tier=ContentTier.B)
        assert not verify_receipt(tampered, identity=identity)

    def test_tamper_privacy_tier(self, unsigned_receipt, identity):
        signed = self._signed(unsigned_receipt, identity)
        tampered = dataclasses.replace(signed, privacy_tier=PrivacyLevel.HIGH)
        assert not verify_receipt(tampered, identity=identity)

    def test_tamper_epsilon_spent(self, unsigned_receipt, identity):
        signed = self._signed(unsigned_receipt, identity)
        tampered = dataclasses.replace(signed, epsilon_spent=4.0)
        assert not verify_receipt(tampered, identity=identity)

    def test_tamper_tee_type(self, unsigned_receipt, identity):
        signed = self._signed(unsigned_receipt, identity)
        tampered = dataclasses.replace(signed, tee_type=TEEType.SGX)
        assert not verify_receipt(tampered, identity=identity)

    def test_tamper_tee_attestation(self, unsigned_receipt, identity):
        signed = self._signed(unsigned_receipt, identity)
        tampered = dataclasses.replace(signed, tee_attestation=b"\xff" * 4)
        assert not verify_receipt(tampered, identity=identity)

    def test_tamper_output_hash(self, unsigned_receipt, identity):
        signed = self._signed(unsigned_receipt, identity)
        tampered = dataclasses.replace(signed, output_hash=b"\xff" * 32)
        assert not verify_receipt(tampered, identity=identity)

    def test_tamper_duration(self, unsigned_receipt, identity):
        signed = self._signed(unsigned_receipt, identity)
        tampered = dataclasses.replace(signed, duration_seconds=999.999)
        assert not verify_receipt(tampered, identity=identity)

    def test_tamper_cost(self, unsigned_receipt, identity):
        signed = self._signed(unsigned_receipt, identity)
        tampered = dataclasses.replace(signed, cost_ftns=Decimal("999"))
        assert not verify_receipt(tampered, identity=identity)

    def test_tamper_settler_node_id(self, unsigned_receipt, identity):
        signed = self._signed(unsigned_receipt, identity)
        tampered = dataclasses.replace(signed, settler_node_id="impersonator")
        assert not verify_receipt(tampered, identity=identity)

    def test_tamper_signature(self, unsigned_receipt, identity):
        signed = self._signed(unsigned_receipt, identity)
        # Flip exactly one bit in the signature
        bad_sig = bytearray(signed.settler_signature)
        bad_sig[0] ^= 0x01
        tampered = dataclasses.replace(signed, settler_signature=bytes(bad_sig))
        assert not verify_receipt(tampered, identity=identity)


# ── Acceptance criteria for Phase 3.x.1 Task 2 ──────────────────────────────


class TestTask2Acceptance:
    """Validates the explicit acceptance criteria from Phase 3.x.1 Task 2.

    Acceptance: "Sign and verify works for all field combinations; tampering
    caught."
    """

    def test_sign_and_verify_all_content_tiers(self, identity):
        """Tier A / B / C all sign + verify correctly."""
        for tier in (ContentTier.A, ContentTier.B, ContentTier.C):
            receipt = InferenceReceipt(
                job_id="j",
                request_id="r",
                model_id="m",
                content_tier=tier,
                privacy_tier=PrivacyLevel.STANDARD,
                epsilon_spent=8.0,
                tee_type=TEEType.SOFTWARE,
                tee_attestation=b"",
                output_hash=b"\x00" * 32,
                duration_seconds=0.1,
                cost_ftns=Decimal("0.1"),
            )
            signed = sign_receipt(receipt, identity)
            assert verify_receipt(signed, identity=identity), \
                f"Sign/verify failed for content_tier={tier.value}"

    def test_sign_and_verify_all_privacy_tiers(self, identity):
        """All four privacy levels sign + verify correctly."""
        for level in (
            PrivacyLevel.NONE,
            PrivacyLevel.STANDARD,
            PrivacyLevel.HIGH,
            PrivacyLevel.MAXIMUM,
        ):
            receipt = InferenceReceipt(
                job_id="j",
                request_id="r",
                model_id="m",
                content_tier=ContentTier.A,
                privacy_tier=level,
                epsilon_spent={
                    PrivacyLevel.NONE: float("inf"),
                    PrivacyLevel.STANDARD: 8.0,
                    PrivacyLevel.HIGH: 4.0,
                    PrivacyLevel.MAXIMUM: 1.0,
                }[level],
                tee_type=TEEType.SOFTWARE,
                tee_attestation=b"",
                output_hash=b"\x00" * 32,
                duration_seconds=0.1,
                cost_ftns=Decimal("0.1"),
            )
            signed = sign_receipt(receipt, identity)
            assert verify_receipt(signed, identity=identity), \
                f"Sign/verify failed for privacy_tier={level.value}"

    def test_sign_and_verify_all_tee_types(self, identity):
        """All TEE types sign + verify correctly."""
        for tee in (
            TEEType.NONE,
            TEEType.SOFTWARE,
            TEEType.SGX,
            TEEType.TDX,
            TEEType.SEV,
            TEEType.TRUSTZONE,
            TEEType.SECURE_ENCLAVE,
        ):
            receipt = InferenceReceipt(
                job_id="j",
                request_id="r",
                model_id="m",
                content_tier=ContentTier.A,
                privacy_tier=PrivacyLevel.STANDARD,
                epsilon_spent=8.0,
                tee_type=tee,
                tee_attestation=b"",
                output_hash=b"\x00" * 32,
                duration_seconds=0.1,
                cost_ftns=Decimal("0.1"),
            )
            signed = sign_receipt(receipt, identity)
            assert verify_receipt(signed, identity=identity), \
                f"Sign/verify failed for tee_type={tee.value}"

    def test_tampering_caught_on_all_signed_fields(self, unsigned_receipt, identity):
        """Tampering with any signed field → verify returns False.

        Comprehensive enumeration — ensures every field included in
        signing_payload() actually contributes to signature uniqueness.
        """
        signed = sign_receipt(unsigned_receipt, identity)
        tampered_versions = [
            ("job_id", "job-x"),
            ("request_id", "req-x"),
            ("model_id", "model-x"),
            ("content_tier", ContentTier.C),
            ("privacy_tier", PrivacyLevel.MAXIMUM),
            ("epsilon_spent", 1.0),
            ("tee_type", TEEType.SGX),
            ("tee_attestation", b"\xff"),
            ("output_hash", b"\xff" * 32),
            ("duration_seconds", 99.9),
            ("cost_ftns", Decimal("99")),
            ("settler_node_id", "impersonator"),
        ]
        for field_name, new_value in tampered_versions:
            tampered = dataclasses.replace(signed, **{field_name: new_value})
            assert not verify_receipt(tampered, identity=identity), \
                f"Tampering with '{field_name}' was not caught"


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.8 — streamed_output flag
# ──────────────────────────────────────────────────────────────────────────


class TestStreamedOutputFlag:
    """The Phase 3.x.8 ``streamed_output: bool`` field on
    ``InferenceReceipt``. Conditional encoding preserves byte-
    equivalence with pre-3.x.8 receipts so the existing back-compat
    invariant for non-streamed receipts still holds."""

    def test_default_streamed_output_is_false(self, unsigned_receipt):
        # Pre-3.x.8 callers don't pass streamed_output; default False
        # preserves their behavior.
        assert unsigned_receipt.streamed_output is False

    def test_signing_payload_byte_identical_when_flag_false(
        self, unsigned_receipt,
    ):
        """Conditional encoding: streamed_output=False produces
        signing bytes identical to the pre-3.x.8 layout. A pre-3.x.8
        verifier reading a 3.x.8 non-streamed receipt MUST still
        validate signatures correctly."""
        # The pre-3.x.8 layout (from the existing signing_payload
        # implementation) is the parts list joined with newlines.
        # Reconstruct what those bytes look like and assert.
        expected_parts = [
            unsigned_receipt.job_id,
            unsigned_receipt.request_id,
            unsigned_receipt.model_id,
            unsigned_receipt.content_tier.value,
            unsigned_receipt.privacy_tier.value,
            f"{unsigned_receipt.epsilon_spent:.10f}",
            unsigned_receipt.tee_type.value,
            unsigned_receipt.tee_attestation.hex(),
            unsigned_receipt.output_hash.hex(),
            f"{unsigned_receipt.duration_seconds:.6f}",
            str(unsigned_receipt.cost_ftns),
            unsigned_receipt.settler_node_id,
        ]
        expected_bytes = "\n".join(expected_parts).encode("utf-8")
        assert unsigned_receipt.signing_payload() == expected_bytes
        # And the marker text MUST NOT appear anywhere.
        assert b"streamed_output" not in unsigned_receipt.signing_payload()

    def test_signing_payload_includes_marker_when_flag_true(
        self, unsigned_receipt,
    ):
        streamed = dataclasses.replace(
            unsigned_receipt, streamed_output=True,
        )
        payload = streamed.signing_payload()
        assert b"streamed_output:true" in payload
        # Trailing line specifically — check it's at the END of the
        # canonical bytes (after settler_node_id).
        assert payload.endswith(b"streamed_output:true")

    def test_signing_payload_differs_when_flag_flipped(
        self, unsigned_receipt,
    ):
        non_streamed = unsigned_receipt
        streamed = dataclasses.replace(
            unsigned_receipt, streamed_output=True,
        )
        assert non_streamed.signing_payload() != streamed.signing_payload()

    def test_round_trip_preserves_streamed_output_true(
        self, unsigned_receipt, identity,
    ):
        streamed = dataclasses.replace(
            unsigned_receipt, streamed_output=True,
        )
        signed = sign_receipt(streamed, identity)
        as_dict = signed.to_dict()
        assert as_dict["streamed_output"] is True
        recovered = InferenceReceipt.from_dict(as_dict)
        assert recovered.streamed_output is True
        assert verify_receipt(recovered, identity=identity)

    def test_round_trip_preserves_streamed_output_false(
        self, unsigned_receipt, identity,
    ):
        signed = sign_receipt(unsigned_receipt, identity)
        as_dict = signed.to_dict()
        assert as_dict["streamed_output"] is False
        recovered = InferenceReceipt.from_dict(as_dict)
        assert recovered.streamed_output is False
        assert verify_receipt(recovered, identity=identity)

    def test_streamed_signed_receipt_verifies(
        self, unsigned_receipt, identity,
    ):
        streamed = dataclasses.replace(
            unsigned_receipt, streamed_output=True,
        )
        signed = sign_receipt(streamed, identity)
        assert verify_receipt(signed, identity=identity) is True

    def test_tampering_streamed_output_invalidates_signature(
        self, unsigned_receipt, identity,
    ):
        """Downgrade-resistance: a relay that flips streamed_output
        from True→False (or False→True) on a SIGNED receipt MUST be
        caught by signature verification. The flag is part of the
        signing payload via the conditional-marker rule."""
        # Sign a streamed receipt.
        streamed = dataclasses.replace(
            unsigned_receipt, streamed_output=True,
        )
        signed_streamed = sign_receipt(streamed, identity)
        assert verify_receipt(signed_streamed, identity=identity) is True

        # Relay flips the flag (downgrade attempt).
        downgraded = dataclasses.replace(
            signed_streamed, streamed_output=False,
        )
        assert verify_receipt(downgraded, identity=identity) is False

        # Inverse: a non-streamed receipt's flag flipped to True.
        signed_nonstreamed = sign_receipt(unsigned_receipt, identity)
        assert verify_receipt(signed_nonstreamed, identity=identity) is True
        upgraded = dataclasses.replace(
            signed_nonstreamed, streamed_output=True,
        )
        assert verify_receipt(upgraded, identity=identity) is False

    def test_from_dict_defaults_to_false_when_key_absent(
        self, identity,
    ):
        """Forward-compat: a pre-3.x.8 serialized receipt (no
        ``streamed_output`` key) parses cleanly on a 3.x.8 reader
        and signature still verifies (the signed bytes are byte-
        identical via the omit-when-False rule)."""
        # Build a pre-3.x.8 receipt as a plain dict (simulating an
        # older serialization that doesn't include the new key).
        receipt = InferenceReceipt(
            job_id="legacy-job",
            request_id="legacy-req",
            model_id="legacy-model",
            content_tier=ContentTier.A,
            privacy_tier=PrivacyLevel.STANDARD,
            epsilon_spent=8.0,
            tee_type=TEEType.SOFTWARE,
            tee_attestation=b"\x01\x02",
            output_hash=b"\xaa" * 32,
            duration_seconds=1.0,
            cost_ftns=Decimal("0.5"),
        )
        signed = sign_receipt(receipt, identity)
        legacy_dict = signed.to_dict()
        # Strip the new field (simulate a pre-3.x.8 wire payload).
        legacy_dict.pop("streamed_output")
        recovered = InferenceReceipt.from_dict(legacy_dict)
        assert recovered.streamed_output is False
        assert verify_receipt(recovered, identity=identity) is True

    def test_from_dict_rejects_int_streamed_output(self):
        """M1-style coercion guard: a hostile peer ships
        ``{"streamed_output": 1}`` — must reject as malformed
        rather than silently coerce to True."""
        bad = {
            "job_id": "j",
            "request_id": "r",
            "model_id": "m",
            "content_tier": ContentTier.A.value,
            "privacy_tier": PrivacyLevel.STANDARD.value,
            "epsilon_spent": 0.0,
            "tee_type": TEEType.SOFTWARE.value,
            "tee_attestation": "00",
            "output_hash": "aa" * 32,
            "duration_seconds": 1.0,
            "cost_ftns": "0.5",
            "settler_signature": "",
            "settler_node_id": "",
            "streamed_output": 1,  # int not bool
        }
        with pytest.raises(TypeError, match="streamed_output"):
            InferenceReceipt.from_dict(bad)
