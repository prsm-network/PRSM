"""Sprint 1368 — Tier B/C publisher surface, brick 4: `prsm content buyer-keygen` + operator guide.

A buyer needs an X25519 keypair (give the public key to the publisher, keep the private key for
unlock). This CLI generates one locally, and the generated pair actually pairs with the wrap/unwrap
crypto.
"""
from __future__ import annotations

import json

from click.testing import CliRunner

from prsm.cli import main


def test_buyer_keygen_json_emits_a_working_keypair():
    r = CliRunner().invoke(main, ["content", "buyer-keygen", "--format", "json"])
    assert r.exit_code == 0, r.output
    d = json.loads(r.output)
    assert set(d) == {"x25519_pubkey_b64", "x25519_privkey_b64"}

    # the generated pair must round-trip: wrap the content key to the PUBLIC key, unwrap with PRIVATE
    from prsm.enterprise.recipient_encryption import EnterpriseRecipient
    from prsm.storage.encryption import encrypt, generate_key
    from prsm.storage.paid_unlock import (
        reconstruct_paid_content,
        wrap_content_key_for_deposit,
    )
    ck = generate_key()
    content = encrypt(b"paywalled bytes", ck)
    wrapped = wrap_content_key_for_deposit(
        ck, [EnterpriseRecipient(identifier="b", x25519_pubkey_b64=d["x25519_pubkey_b64"])])
    assert reconstruct_paid_content(wrapped, d["x25519_privkey_b64"], content) == b"paywalled bytes"


def test_buyer_keygen_text_marks_public_vs_private():
    r = CliRunner().invoke(main, ["content", "buyer-keygen"])
    assert r.exit_code == 0
    assert "PUBLIC" in r.output and "PRIVATE" in r.output
    assert "PRSM_X25519_PRIVKEY" in r.output          # tells the buyer where the private key goes


def test_two_keygens_differ():
    a = json.loads(CliRunner().invoke(main, ["content", "buyer-keygen", "--format", "json"]).output)
    b = json.loads(CliRunner().invoke(main, ["content", "buyer-keygen", "--format", "json"]).output)
    assert a["x25519_privkey_b64"] != b["x25519_privkey_b64"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
