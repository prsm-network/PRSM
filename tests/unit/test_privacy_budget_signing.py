"""
Unit tests — Phase 3.x.4 Task 2 — entry sign/verify.

Acceptance per design plan §4 Task 2: sign/verify works against a real
NodeIdentity; tampering caught for every signed field; cross-artifact
domain separation tested against ModelManifest signatures.

Real Ed25519 keypairs and real NodeIdentity instances — no crypto mocks,
per project testing rules.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib

import pytest

from prsm.security.privacy_budget_persistence.models import (
    ENTRY_SIGNING_DOMAIN,
    GENESIS_PREV_HASH,
    PrivacyBudgetEntry,
    PrivacyBudgetEntryType,
)
from prsm.security.privacy_budget_persistence.signing import (
    is_signed,
    sign_entry,
    verify_entry,
)
from prsm.node.identity import NodeIdentity, generate_node_identity


# ──────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ──────────────────────────────────────────────────────────────────────────


def _entry(
    *,
    sequence_number: int = 0,
    entry_type: PrivacyBudgetEntryType = PrivacyBudgetEntryType.SPEND,
    epsilon: float = 8.0,
    operation: str = "inference",
    model_id: str = "llama-3-8b",
    timestamp: float = 1714000000.0,
    node_id: str = "placeholder",  # overwritten by sign_entry
    prev_entry_hash: bytes = GENESIS_PREV_HASH,
) -> PrivacyBudgetEntry:
    return PrivacyBudgetEntry(
        sequence_number=sequence_number,
        entry_type=entry_type,
        node_id=node_id,
        epsilon=epsilon,
        operation=operation,
        model_id=model_id,
        timestamp=timestamp,
        prev_entry_hash=prev_entry_hash,
    )


@pytest.fixture
def identity() -> NodeIdentity:
    return generate_node_identity(display_name="phase3.x.4-task2-signer")


@pytest.fixture
def other_identity() -> NodeIdentity:
    return generate_node_identity(display_name="phase3.x.4-task2-impostor")


@pytest.fixture
def signed(identity) -> PrivacyBudgetEntry:
    return sign_entry(_entry(), identity)


# ──────────────────────────────────────────────────────────────────────────
# sign_entry
# ──────────────────────────────────────────────────────────────────────────


class TestSignEntry:
    def test_returns_new_entry_unchanged_original(self, identity):
        original = _entry()
        signed = sign_entry(original, identity)
        # Frozen dataclasses; sign_entry must produce a fresh instance.
        assert signed is not original
        assert original.signature == b""
        assert signed.signature != b""

    def test_stamps_node_id(self, identity):
        signed = sign_entry(_entry(), identity)
        assert signed.node_id == identity.node_id

    def test_overwrites_placeholder_node_id(self, identity):
        # Caller-supplied node_id must be replaced with the signer's
        # actual node_id. Otherwise verifiers couldn't trust the field.
        signed = sign_entry(_entry(node_id="placeholder"), identity)
        assert signed.node_id == identity.node_id

    def test_signature_is_64_bytes(self, identity):
        # Ed25519 signatures are always 64 bytes.
        signed = sign_entry(_entry(), identity)
        assert len(signed.signature) == 64

    def test_signature_deterministic_for_same_entry(self, identity):
        # Ed25519 is deterministic — same payload + same key → same sig.
        s1 = sign_entry(_entry(), identity)
        s2 = sign_entry(_entry(), identity)
        assert s1.signature == s2.signature

    def test_signature_differs_across_signers(self, identity, other_identity):
        s1 = sign_entry(_entry(), identity)
        s2 = sign_entry(_entry(), other_identity)
        assert s1.signature != s2.signature
        assert s1.node_id != s2.node_id

    def test_signature_differs_for_spend_vs_reset(self, identity):
        spend = sign_entry(
            _entry(entry_type=PrivacyBudgetEntryType.SPEND), identity
        )
        reset = sign_entry(
            _entry(
                entry_type=PrivacyBudgetEntryType.RESET,
                epsilon=0.0, operation="", model_id="",
            ),
            identity,
        )
        assert spend.signature != reset.signature


# ──────────────────────────────────────────────────────────────────────────
# verify_entry — happy path
# ──────────────────────────────────────────────────────────────────────────


class TestVerifyHappyPath:
    def test_verifies_against_signing_identity(self, identity, signed):
        assert verify_entry(signed, identity=identity) is True

    def test_verifies_against_public_key_b64(self, identity, signed):
        assert verify_entry(signed, public_key_b64=identity.public_key_b64) is True

    def test_verifies_after_dict_roundtrip(self, identity, signed):
        # FilesystemPrivacyBudgetStore (Task 4) writes JSON and reads
        # back; verification must survive that roundtrip.
        d = signed.to_dict()
        reloaded = PrivacyBudgetEntry.from_dict(d)
        assert verify_entry(reloaded, identity=identity) is True


# ──────────────────────────────────────────────────────────────────────────
# verify_entry — failure modes
# ──────────────────────────────────────────────────────────────────────────


class TestVerifyFailureModes:
    def test_unsigned_returns_false(self, identity):
        assert verify_entry(_entry(), identity=identity) is False

    def test_no_credential_returns_false(self, signed):
        assert verify_entry(signed) is False

    def test_wrong_public_key_returns_false(self, signed, other_identity):
        assert verify_entry(signed, identity=other_identity) is False
        assert verify_entry(
            signed, public_key_b64=other_identity.public_key_b64
        ) is False

    def test_malformed_public_key_returns_false(self, signed):
        # Garbage pubkey must NOT raise — fail closed.
        assert verify_entry(signed, public_key_b64="not-base64-!!!") is False

    def test_truncated_signature_returns_false(self, identity, signed):
        bad = dataclasses.replace(signed, signature=signed.signature[:32])
        assert verify_entry(bad, identity=identity) is False

    def test_zero_signature_returns_false(self, identity, signed):
        bad = dataclasses.replace(signed, signature=b"\x00" * 64)
        assert verify_entry(bad, identity=identity) is False


# ──────────────────────────────────────────────────────────────────────────
# Tamper detection — every signed field must invalidate the signature
# ──────────────────────────────────────────────────────────────────────────


class TestTamperDetection:
    """Confirms every load-bearing field in signing_payload is covered.

    If any of these fail, that field is missing from the payload and
    could be tampered without invalidating the signature.
    """

    def test_tamper_sequence_number(self, identity, signed):
        bad = dataclasses.replace(signed, sequence_number=signed.sequence_number + 1)
        assert verify_entry(bad, identity=identity) is False

    def test_tamper_entry_type(self, identity, signed):
        # Convert a SPEND to a RESET (and zero out the spend-specific
        # fields to keep the manifest coherent — what matters is the
        # signature is now invalid).
        bad = dataclasses.replace(
            signed,
            entry_type=PrivacyBudgetEntryType.RESET,
            epsilon=0.0, operation="", model_id="",
        )
        assert verify_entry(bad, identity=identity) is False

    def test_tamper_node_id(self, identity, signed):
        bad = dataclasses.replace(signed, node_id="different-signer")
        assert verify_entry(bad, identity=identity) is False

    def test_tamper_epsilon(self, identity, signed):
        # Lower the recorded ε spend → operator dodges budget. Must fail.
        bad = dataclasses.replace(signed, epsilon=signed.epsilon / 2)
        assert verify_entry(bad, identity=identity) is False

    def test_tamper_epsilon_subdigit(self, identity, signed):
        # 10-decimal-place precision means a tiny ε edit also breaks the sig.
        bad = dataclasses.replace(signed, epsilon=signed.epsilon + 1e-10)
        assert verify_entry(bad, identity=identity) is False

    def test_tamper_operation(self, identity, signed):
        bad = dataclasses.replace(signed, operation="malicious")
        assert verify_entry(bad, identity=identity) is False

    def test_tamper_model_id(self, identity, signed):
        bad = dataclasses.replace(signed, model_id="different-model")
        assert verify_entry(bad, identity=identity) is False

    def test_tamper_timestamp(self, identity, signed):
        # Backdate the entry so it looks like an older spend.
        bad = dataclasses.replace(signed, timestamp=signed.timestamp - 1000.0)
        assert verify_entry(bad, identity=identity) is False

    def test_tamper_prev_entry_hash(self, identity, signed):
        # Change the chain link → payload changes → sig invalid. This
        # is what makes historical-tamper-detection work end-to-end.
        bad_hash = hashlib.sha256(b"tampered predecessor").digest()
        bad = dataclasses.replace(signed, prev_entry_hash=bad_hash)
        assert verify_entry(bad, identity=identity) is False

    def test_tamper_schema_version(self, identity, signed):
        # Downgrade attack: re-stamp v2 entry as v1 to evade a v1-only
        # verifier. Must fail.
        bad = dataclasses.replace(signed, schema_version=signed.schema_version + 1)
        assert verify_entry(bad, identity=identity) is False


# ──────────────────────────────────────────────────────────────────────────
# Signing-payload-excludes-signature property (re-verified via signing flow)
# ──────────────────────────────────────────────────────────────────────────


class TestSigningExclusion:
    def test_resigning_yields_identical_signature(self, identity):
        # Sign once, then sign again. Because the payload doesn't depend
        # on the signature, both signatures must be identical (Ed25519
        # is deterministic).
        e1 = sign_entry(_entry(), identity)
        e2 = sign_entry(e1, identity)  # already-signed → re-sign anyway
        assert e1.signature == e2.signature

    def test_signature_field_doesnt_affect_payload(self, identity, signed):
        # The signed entry's signing_payload bytes must equal the
        # un-signed-but-stamped entry's signing_payload bytes.
        stamped_only = dataclasses.replace(_entry(), node_id=identity.node_id)
        assert stamped_only.signing_payload() == signed.signing_payload()


# ──────────────────────────────────────────────────────────────────────────
# Cross-artifact replay — domain separation MUST hold across all 3 PRSM
# signed-artifact types (InferenceReceipt, ModelManifest, PrivacyBudgetEntry)
# ──────────────────────────────────────────────────────────────────────────


class TestCrossArtifactDomainSeparation:
    """The three artifact types use distinct domain prefixes so a
    signature over one cannot be replayed against another."""

    def test_entry_signature_does_not_verify_against_other_domain(
        self, identity, signed
    ):
        # Reconstruct the canonical entry payload, then prepend a
        # different domain string and re-verify with the same signature.
        # MUST fail.
        from prsm.node.identity import verify_signature
        sig_b64 = base64.b64encode(signed.signature).decode()

        # Real payload verifies
        assert verify_signature(
            identity.public_key_b64, signed.signing_payload(), sig_b64
        ) is True

        # Same signature over a manifest-domain-prefixed payload must NOT verify
        original_bytes = signed.signing_payload()
        forged = b"prsm-model-manifest:v1" + original_bytes[len(ENTRY_SIGNING_DOMAIN):]
        assert verify_signature(
            identity.public_key_b64, forged, sig_b64
        ) is False

        # Same signature over a receipt-domain-prefixed payload must NOT verify
        forged2 = b"prsm-inference-receipt:v1" + original_bytes[len(ENTRY_SIGNING_DOMAIN):]
        assert verify_signature(
            identity.public_key_b64, forged2, sig_b64
        ) is False

    def test_manifest_signature_does_not_verify_as_entry(self, identity):
        # The other direction: a real ModelManifest signature must not
        # verify as a budget-entry signature, even when the byte layouts
        # happen to look similar.
        from prsm.compute.model_registry import (
            ManifestShardEntry,
            ModelManifest,
            sign_manifest,
        )
        manifest = ModelManifest(
            model_id="m", model_name="M",
            publisher_node_id="placeholder",
            total_shards=1,
            shards=(ManifestShardEntry(
                shard_id="s0", shard_index=0,
                tensor_shape=(1,),
                sha256="ab" * 32, size_bytes=1,
            ),),
            published_at=0.0,
        )
        signed_manifest = sign_manifest(manifest, identity)
        sig_b64 = base64.b64encode(signed_manifest.publisher_signature).decode()

        # Build an entry whose signing_payload is the exact bytes the
        # manifest signature was over, but with the entry domain prefix.
        # The signature is valid for the manifest payload but NOT for the
        # entry payload — different domain, different bytes.
        # Easier check: verify the manifest's signature against an entry
        # under the same identity. The bytes don't match (different
        # canonical layouts entirely), so verification fails.
        entry = sign_entry(_entry(), identity)
        # Swap the entry's signature for the manifest's. Verify must fail.
        forged_entry = dataclasses.replace(
            entry, signature=signed_manifest.publisher_signature
        )
        assert verify_entry(forged_entry, identity=identity) is False


# ──────────────────────────────────────────────────────────────────────────
# is_signed predicate
# ──────────────────────────────────────────────────────────────────────────


class TestIsSigned:
    def test_unsigned_returns_false(self):
        assert is_signed(_entry()) is False

    def test_signed_returns_true(self, signed):
        assert is_signed(signed) is True

    def test_signature_only_no_node_id_returns_false(self):
        # Bizarre construction: signature populated but node_id empty.
        # Treat as unsigned (no provenance trace).
        bad = dataclasses.replace(_entry(node_id=""), signature=b"\xff" * 64)
        assert is_signed(bad) is False

    def test_node_id_only_no_signature_returns_false(self, identity):
        e = dataclasses.replace(_entry(), node_id=identity.node_id)
        assert is_signed(e) is False

    def test_is_signed_is_NOT_a_crypto_check(self, identity, other_identity):
        # is_signed only checks field-population. A tampered post-signing
        # entry passes is_signed but fails verify_entry — confirms the
        # documented contract.
        signed_by_a = sign_entry(_entry(), identity)
        # Tamper node_id post-signing
        confused = dataclasses.replace(signed_by_a, node_id=other_identity.node_id)
        assert is_signed(confused) is True   # fields populated
        assert verify_entry(confused, identity=identity) is False   # crypto fails


class TestSprint1123KeyNodeIdBinding:
    """Sprint 1123 (Domain-08 review LOW-10) — verify_entry binds the verifying key to
    the entry's claimed node_id, so a caller can't be tricked into verifying an internally
    consistent (pubkey, signature, node_id) triple signed by the WRONG key."""

    def test_pubkey_not_matching_entry_node_id_rejected(self, identity, other_identity):
        # entry genuinely signed by `identity` (entry.node_id == identity.node_id)
        signed = sign_entry(_entry(), identity)
        # an auditor who mistakenly supplies a DIFFERENT node's pubkey must get False
        # (the key doesn't derive entry.node_id), not a silent accept of the wrong key.
        assert verify_entry(signed, public_key_b64=other_identity.public_key_b64) is False

    def test_internally_consistent_attacker_triple_rejected(self, identity, other_identity):
        # Attacker signs an entry with THEIR key but stamps the victim's node_id claim.
        forged = _entry(node_id="placeholder")
        forged = dataclasses.replace(forged, node_id=identity.node_id)  # claim victim
        # sign with the attacker's key over a payload claiming node_id=victim
        import base64 as _b64
        intermediate = dataclasses.replace(forged, node_id=identity.node_id)
        sig_b64 = other_identity.sign(intermediate.signing_payload())
        triple = dataclasses.replace(intermediate, signature=_b64.b64decode(sig_b64))
        # The attacker presents their OWN pubkey (which signed it) — but it doesn't derive
        # the claimed victim node_id, so the binding check rejects it.
        assert verify_entry(triple, public_key_b64=other_identity.public_key_b64) is False

    def test_correct_key_still_verifies(self, identity):
        signed = sign_entry(_entry(), identity)
        assert verify_entry(signed, public_key_b64=identity.public_key_b64) is True
        assert verify_entry(signed, identity=identity) is True
