"""Sprint 972 — verify the provider's price-quote signature in request_quote().

Agent-collab/marketplace review finding #2 (MEDIUM). PriceNegotiator.request_quote
captured the provider's signature into PriceQuote (price_handshake.py:149) but
NEVER verified it — the "stored but never checked" class (cf. sp964/sp969). The
provider signs `request_id||listing_id||shard_index||quoted_price||
quote_expires_unix||provider_id` (compute_provider.py:1017-1021); an on-path
attacker could tamper the quote (lower price below the signed listing ceiling,
shift expiry, spoof provider_id) and the requester would trust it.

(Harm is bounded — the listing ceiling check at line 128 + the single-source
quote funding both escrow and settlement mean a tampered quote can only LOWER the
price, harming the provider not the requester — so this is an authenticity/
hardening fix, not requester fund-loss. Fix it anyway: make the captured
signature load-bearing.)

Fix: reconstruct the EXACT signed bytes from the RAW (un-coerced) response values
(so str() matches the provider's f-string byte-for-byte), verify against
provider_pubkey_b64, bind the pubkey to provider_id (node_id == sha256(pubkey)[:32])
and require provider_id == the signed listing's provider. Reject otherwise.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from prsm.marketplace.price_handshake import (
    PriceNegotiator, PriceQuote, PriceQuoteRejected,
)
from prsm.node.identity import generate_node_identity
from prsm.node.transport import MSG_DIRECT, P2PMessage


def _listing(provider_id, ceiling=1.0):
    l = MagicMock()
    l.listing_id = "L1"
    l.provider_id = provider_id
    l.price_per_shard_ftns = ceiling
    return l


def _ack(provider, request_id, *, price, expires, listing_id="L1", shard_index=0,
         tamper_price=None, provider_id_override=None, sign_with=None):
    """Build a provider ack. The signature covers the ORIGINAL values; pass
    tamper_price to mutate the advertised price AFTER signing (forgery)."""
    signer = sign_with or provider
    pid = provider.node_id
    sig_src = (
        f"{request_id}||{listing_id}||{shard_index}||{price}||{expires}||{pid}"
    ).encode("utf-8")
    return {
        "subtype": "shard_price_quote_ack",
        "request_id": request_id,
        "listing_id": listing_id,
        "shard_index": shard_index,
        "quoted_price_ftns": tamper_price if tamper_price is not None else price,
        "quote_expires_unix": expires,
        "provider_id": provider_id_override or pid,
        "provider_pubkey_b64": provider.public_key_b64,
        "signature": signer.sign(sig_src),
    }


async def _run(neg, listing, build_ack):
    """Drive request_quote, delivering build_ack(request_id) synchronously
    inside send_to_peer so the pending future is resolved before wait_for."""
    async def fake_send(peer_id, msg):
        rid = msg.payload["request_id"]
        ack = build_ack(rid)
        if ack is not None:
            await neg._on_direct_message(
                P2PMessage(msg_type=MSG_DIRECT, sender_id="x", payload=ack), None)
    neg.transport.send_to_peer = fake_send
    return await neg.request_quote(listing, shard_index=0, shard_size_bytes=100,
                                   max_acceptable_price_ftns=1.0)


def _negotiator():
    transport = MagicMock()
    neg = PriceNegotiator(identity=generate_node_identity(), transport=transport)
    return neg


@pytest.mark.asyncio
async def test_valid_signed_quote_accepted():
    provider = generate_node_identity()
    neg = _negotiator()
    expires = int(time.time()) + 30
    result = await _run(neg, _listing(provider.node_id),
                        lambda rid: _ack(provider, rid, price=0.5, expires=expires))
    assert isinstance(result, PriceQuote)
    assert result.quoted_price_ftns == 0.5


@pytest.mark.asyncio
async def test_tampered_price_rejected():
    provider = generate_node_identity()
    neg = _negotiator()
    expires = int(time.time()) + 30
    # Signed for 0.5, but the wire says 0.1 → signature won't verify.
    result = await _run(neg, _listing(provider.node_id),
                        lambda rid: _ack(provider, rid, price=0.5, expires=expires,
                                         tamper_price=0.1))
    assert isinstance(result, PriceQuoteRejected)


@pytest.mark.asyncio
async def test_unsigned_quote_rejected():
    provider = generate_node_identity()
    neg = _negotiator()
    expires = int(time.time()) + 30

    def build(rid):
        a = _ack(provider, rid, price=0.5, expires=expires)
        a["signature"] = ""
        return a
    result = await _run(neg, _listing(provider.node_id), build)
    assert isinstance(result, PriceQuoteRejected)


@pytest.mark.asyncio
async def test_wrong_signer_rejected():
    """An attacker signs with their OWN key but claims the listing provider's id
    + pubkey — pubkey won't match provider_id binding / signature won't verify."""
    provider = generate_node_identity()
    attacker = generate_node_identity()
    neg = _negotiator()
    expires = int(time.time()) + 30
    # provider_id claims the real provider, but signed by attacker; pubkey is the
    # real provider's (so binding passes) but the attacker's signature won't verify.
    result = await _run(neg, _listing(provider.node_id),
                        lambda rid: _ack(provider, rid, price=0.5, expires=expires,
                                         sign_with=attacker))
    assert isinstance(result, PriceQuoteRejected)


@pytest.mark.asyncio
async def test_provider_id_must_match_listing():
    """A validly-signed quote from a DIFFERENT provider than the listing's must
    be rejected (can't substitute another provider's quote)."""
    other = generate_node_identity()
    neg = _negotiator()
    expires = int(time.time()) + 30
    # listing.provider_id is someone else; the ack is validly signed by `other`.
    result = await _run(neg, _listing("some_other_provider_id"),
                        lambda rid: _ack(other, rid, price=0.5, expires=expires))
    assert isinstance(result, PriceQuoteRejected)


@pytest.mark.asyncio
async def test_integer_price_does_not_falsely_reject():
    """Guard against the int-vs-float str() mismatch: a provider that quotes an
    integer price must still verify (reconstruct from the RAW value, not float())."""
    provider = generate_node_identity()
    neg = _negotiator()
    expires = int(time.time()) + 30
    result = await _run(neg, _listing(provider.node_id, ceiling=5.0),
                        lambda rid: _ack(provider, rid, price=1, expires=expires))
    assert isinstance(result, PriceQuote)
