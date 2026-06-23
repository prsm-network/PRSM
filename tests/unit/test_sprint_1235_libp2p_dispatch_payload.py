"""Sprint 1235 — libp2p _dispatch normalizes the direct/gossip payload to a dict.

The Go daemon delivers a /prsm/direct/1.0.0 (or gossip) message over UDS as
{"msg_type":"direct","data":<inner-P2PMessage-JSON-STRING>,...} — `data` is the
full inner P2PMessage that the sender put on the wire via msg.to_json(). The old
_dispatch set payload = that raw STRING, so every direct handler doing
msg.payload.get("subtype") hit AttributeError ('str' has no .get) and inflated
dispatch_failure_total. The gossip handler json.loads'd it itself; direct
handlers didn't. WebSocketTransport (the default backend) parses via
P2PMessage.from_json → handlers get the inner payload DICT. This sprint makes the
libp2p path match: parse the inner JSON, hand handlers the inner payload dict +
metadata. Routing stays on the OUTER msg_type ("direct"/"gossip") — unchanged.

Per the scope risk note: assert dispatch COUNTERS, not just "no raise" (the
handler-error path is swallowed into dispatch_failure_total).
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

from prsm.node.libp2p_transport import Libp2pTransport


def _make_transport():
    ident = MagicMock()
    ident.node_id = "self-node-id"
    ident.private_key_bytes = b"\x01" * 32
    ident.public_key_bytes = b"\x02" * 32
    ident.sign.return_value = "fakesig"
    with patch.object(Libp2pTransport, "_load_library", return_value=MagicMock()):
        return Libp2pTransport(identity=ident, port=9001)


def _wrapped(inner_msg_type, payload, sender="nodeA"):
    """Build the Go UDS frame: outer 'direct' + inner P2PMessage JSON in data."""
    inner = json.dumps({
        "msg_type": inner_msg_type, "sender_id": sender,
        "payload": payload, "signature": "sig", "ttl": 5, "nonce": "n",
    })
    return {"msg_type": "direct", "sender_id": "libp2p-peer-xyz", "data": inner}


def test_direct_wrapper_yields_dict_payload_to_handler():
    t = _make_transport()
    got = []

    async def h(msg, peer):
        # the exact access pattern real direct handlers use
        got.append((msg.payload.get("subtype"), msg.payload.get("x"), msg.sender_id))

    t.on_message("direct", h)
    asyncio.run(t._dispatch(_wrapped("direct", {"subtype": "shard_execute_response", "x": 7})))

    assert got == [("shard_execute_response", 7, "nodeA")]  # inner payload dict + inner sender_id
    assert t._telemetry["dispatch_success_total"] >= 1
    assert t._telemetry["dispatch_failure_total"] == 0


def test_handler_does_not_attributeerror_on_payload_get():
    """The actual bug: payload.get(...) on a str raised AttributeError pre-fix."""
    t = _make_transport()
    ok = []

    async def h(msg, peer):
        ok.append(msg.payload.get("subtype"))

    t.on_message("direct", h)
    asyncio.run(t._dispatch(_wrapped("direct", {"subtype": "hello"})))
    assert ok == ["hello"]
    assert t._telemetry["dispatch_failure_total"] == 0


def test_keepalive_wrapped_still_short_circuits():
    t = _make_transport()
    called = []

    async def h(msg, peer):
        called.append(msg)

    t.on_message("direct", h)
    before = t._telemetry["dispatch_failure_total"]
    asyncio.run(t._dispatch(_wrapped("keepalive_ping", {"subtype": "keepalive_ping"})))
    assert called == []  # sp1229 short-circuit fires before fan-out
    assert t._telemetry["dispatch_failure_total"] == before


def test_malformed_inner_json_no_crash_no_failure():
    t = _make_transport()
    calls = []

    async def h(msg, peer):
        calls.append(msg.payload)  # must be a dict, never a str

    t.on_message("direct", h)
    asyncio.run(t._dispatch({"msg_type": "direct", "sender_id": "p", "data": "{not valid json"}))
    assert calls == [{}]  # graceful empty dict
    assert t._telemetry["dispatch_failure_total"] == 0


def test_gossip_string_payload_also_normalized():
    t = _make_transport()
    got = []

    async def h(msg, peer):
        got.append(msg.payload.get("subtype"))

    t.on_message("gossip", h)
    inner = json.dumps({"msg_type": "gossip", "sender_id": "nodeB",
                        "payload": {"subtype": "ann"}, "ttl": 5, "nonce": "n"})
    asyncio.run(t._dispatch({"msg_type": "gossip", "sender_id": "peerL", "data": inner}))
    assert got == ["ann"]
    assert t._telemetry["dispatch_failure_total"] == 0


def test_routing_stays_on_outer_label():
    """A direct-protocol frame whose INNER msg_type differs still routes to the
    'direct' handler (Go labels the protocol; routing is on the outer label)."""
    t = _make_transport()
    direct_hits, other_hits = [], []

    async def direct_h(msg, peer):
        direct_hits.append(msg)

    async def other_h(msg, peer):
        other_hits.append(msg)

    t.on_message("direct", direct_h)
    t.on_message("compute_announce", other_h)
    # inner msg_type is a semantic label, but the Go outer label is "direct"
    asyncio.run(t._dispatch(_wrapped("compute_announce", {"subtype": "announce"})))
    assert len(direct_hits) == 1 and other_hits == []


import pytest


@pytest.mark.parametrize("bad_data", [None, 5, ["a", "b"], 3.14, True])
def test_non_dict_non_string_data_yields_dict_payload(bad_data):
    """A JSON null/scalar/list in `data` must NOT reach a handler as a non-dict
    (it would skip the str-normalization and crash payload.get())."""
    t = _make_transport()
    seen = []

    async def h(msg, peer):
        seen.append(msg.payload.get("anything"))  # must not AttributeError

    t.on_message("direct", h)
    asyncio.run(t._dispatch({"msg_type": "direct", "sender_id": "p", "data": bad_data}))
    assert seen == [None]                     # got an (empty) dict
    assert t._telemetry["dispatch_failure_total"] == 0


def test_non_dict_inner_payload_coerced_to_dict():
    """If the inner P2PMessage's payload is itself non-dict (malformed), the
    handler still receives a dict, not the raw scalar/string."""
    t = _make_transport()
    seen = []

    async def h(msg, peer):
        seen.append(msg.payload.get("k"))

    t.on_message("direct", h)
    inner = json.dumps({"msg_type": "direct", "sender_id": "a", "payload": "not-a-dict"})
    asyncio.run(t._dispatch({"msg_type": "direct", "sender_id": "p", "data": inner}))
    assert seen == [None]
    assert t._telemetry["dispatch_failure_total"] == 0


def test_already_dict_payload_passthrough():
    """A frame that already carries a dict payload (no wrapping) is unchanged."""
    t = _make_transport()
    got = []

    async def h(msg, peer):
        got.append(msg.payload.get("subtype"))

    t.on_message("direct", h)
    asyncio.run(t._dispatch({"msg_type": "direct", "sender_id": "p",
                            "payload": {"subtype": "plain"}}))
    assert got == ["plain"]
    assert t._telemetry["dispatch_failure_total"] == 0
