"""sp1453 (fiat-onramp audit waxla0c1x) — the KYC-webhook replay ring must not be bypassable by a
whitespace variant of the Persona-Signature header.

The fiat on-ramp money audit came back CLEAN (the non-custodial design holds — a phantom funnel
confirmation only builds a swap envelope for the user's OWN USDC to their OWN address; no PRSM custody,
no FTNS minted). But it surfaced a real "defense that cannot fire": the signature verifier parses the
Persona-Signature header (`t=<ts>,v1=<hex>`) TOLERANTLY (partition('=')+strip(), so `v1 = <hex>` with
spaces still verifies), while the replay-token extractor used a STRICT `startswith("v1=")`, leaving
replay_token='' for the spaced variant → replay_ring.record() was skipped → a captured, still-valid
webhook could be REPLAYED (bounded only by the ±300s timestamp-freshness window) without the replay
ring catching it. Two independent parsers on one security boundary drifted.

Fix: ONE shared parser (parse_persona_signature_header) used by BOTH the verifier and the replay-token
extractor, so a header the verifier accepts always yields the same replay token — the ring can never be
silently skipped for a signature-valid header.
"""
from __future__ import annotations

import hashlib
import hmac

from prsm.economy.web3.kyc_webhook_verifier import (
    parse_persona_signature_header,
    verify_persona_signature,
)

_SECRET = "whsec_test_secret"


def _sig(ts: str, body: bytes) -> str:
    return hmac.new(_SECRET.encode("utf-8"), f"{ts}.".encode("utf-8") + body, hashlib.sha256).hexdigest()


def test_parser_extracts_v1_from_compact_and_spaced_headers():
    ts, body = "1700000000", b'{"data":{"attributes":{"status":"approved"}}}'
    sig = _sig(ts, body)
    compact = f"t={ts},v1={sig}"
    spaced = f"t={ts}, v1 = {sig}"  # the whitespace variant the old strict extractor missed
    assert parse_persona_signature_header(compact).get("v1") == sig
    assert parse_persona_signature_header(spaced).get("v1") == sig
    assert parse_persona_signature_header(spaced).get("t") == ts


def test_replay_token_extraction_agrees_with_the_verifier_on_a_spaced_header():
    """The invariant that closes the bypass: for ANY header the verifier ACCEPTS, the replay-token
    extractor (now the SAME parser) yields a non-empty v1 — so the ring records it and a replay is
    caught. A parser that returned '' here would silently skip the ring for a signature-valid header."""
    ts, body = "1700000000", b'{"data":{"attributes":{"status":"approved"}}}'
    sig = _sig(ts, body)
    spaced = f"t={ts}, v1 = {sig}"

    ok, reason = verify_persona_signature(body, spaced, _SECRET)
    assert ok, f"verifier rejected the spaced header (it has always tolerated it): {reason}"

    replay_token = parse_persona_signature_header(spaced).get("v1", "")
    assert replay_token == sig, (
        "replay token empty/mismatched for a signature-VALID spaced header — the replay ring is "
        "silently skipped and the webhook can be replayed within the freshness window")


def test_malformed_header_yields_no_token_but_never_raises():
    assert parse_persona_signature_header("").get("v1") is None
    assert parse_persona_signature_header("garbage-no-equals").get("v1") is None
    assert parse_persona_signature_header("t=123").get("v1") is None  # no v1 → no token
