"""Unit tests for the Phase-2 profile DHT.

Phase 3.x.6 Task 4 — exercises the (τ, ρ) profile DHT mirroring the
Phase 3.x.5 manifest DHT pattern.

Test surface (per design plan §4 Task 4):
  - Publish + lookup happy path
  - Stale entry expiry (TTL respected)
  - Anchor verify rejects unsigned profile (no signature / bad sig)
  - Anchor verify rejects profile signed by unregistered publisher
  - Concurrent updates: latest timestamp wins per-node
  - Wire-format protocol version mismatch handled
  - Server.handle never raises (parametrized fuzz)
  - SignedProfileEntry validation rejects garbage
"""

from __future__ import annotations

import time

import pytest

from prsm.compute.parallax_scheduling.profile_dht import (
    DEFAULT_PUBLISH_INTERVAL_SECONDS,
    DEFAULT_TTL_SECONDS,
    MAX_MESSAGE_BYTES,
    PROFILE_DHT_PROTOCOL_VERSION,
    ErrorResponse,
    FetchProfileRequest,
    FetchResponse,
    ProfileDHT,
    ProfileDHTServer,
    ProfileErrorCode,
    ProfileMalformedError,
    ProfileMessageType,
    ProfileUnknownTypeError,
    ProfileVersionMismatchError,
    PublishProfileRequest,
    PublishResponse,
    SignedProfileEntry,
    encode_message,
    parse_message,
)
from prsm.node.identity import generate_node_identity


# ──────────────────────────────────────────────────────────────────────────
# Test helpers — fake anchor + clock
# ──────────────────────────────────────────────────────────────────────────


class FakeAnchor:
    """Anchor stub: maps node_id → public_key_b64. Tests register
    publishers explicitly so we can exercise both registered and
    unregistered paths."""

    def __init__(self):
        self._registry: dict[str, str] = {}

    def register(self, node_id: str, public_key_b64: str) -> None:
        self._registry[node_id] = public_key_b64

    def lookup(self, node_id: str):
        return self._registry.get(node_id)


class FakeClock:
    """Manually-advanced clock for deterministic TTL tests."""

    def __init__(self, t: float = 1_700_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def alice():
    return generate_node_identity(display_name="phase3.x.6-task4-alice")


@pytest.fixture
def bob():
    return generate_node_identity(display_name="phase3.x.6-task4-bob")


@pytest.fixture
def anchor(alice, bob):
    a = FakeAnchor()
    a.register(alice.node_id, alice.public_key_b64)
    a.register(bob.node_id, bob.public_key_b64)
    return a


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def server(anchor, clock):
    return ProfileDHTServer(anchor=anchor, clock=clock)


# ──────────────────────────────────────────────────────────────────────────
# SignedProfileEntry validation
# ──────────────────────────────────────────────────────────────────────────


class TestSignedProfileEntryValidation:
    def test_basic_construction(self, alice, clock):
        entry = SignedProfileEntry.sign(
            identity=alice,
            layer_latency_ms=1.5,
            rtt_to_peers={"bob": 5.0},
            timestamp_unix=clock(),
        )
        assert entry.node_id == alice.node_id
        assert entry.layer_latency_ms == 1.5
        assert entry.rtt_to_peers == {"bob": 5.0}
        assert entry.signature_b64

    def test_negative_latency_rejected(self):
        with pytest.raises(ProfileMalformedError, match="layer_latency_ms"):
            SignedProfileEntry(
                node_id="x",
                layer_latency_ms=-1.0,
                rtt_to_peers={},
                timestamp_unix=0.0,
                signature_b64="sig",
            )

    def test_negative_rtt_rejected(self):
        with pytest.raises(ProfileMalformedError, match="rtt_to_peers"):
            SignedProfileEntry(
                node_id="x",
                layer_latency_ms=1.0,
                rtt_to_peers={"y": -5.0},
                timestamp_unix=0.0,
                signature_b64="sig",
            )

    def test_empty_node_id_rejected(self):
        with pytest.raises(ProfileMalformedError):
            SignedProfileEntry(
                node_id="",
                layer_latency_ms=1.0,
                rtt_to_peers={},
                timestamp_unix=0.0,
                signature_b64="sig",
            )

    def test_signing_payload_is_deterministic(self, alice, clock):
        # Same logical payload → same canonical bytes regardless of
        # rtt_to_peers iteration order.
        bytes_a = SignedProfileEntry.signing_payload(
            alice.node_id, 1.5, {"a": 1.0, "b": 2.0}, clock()
        )
        bytes_b = SignedProfileEntry.signing_payload(
            alice.node_id, 1.5, {"b": 2.0, "a": 1.0}, clock()
        )
        assert bytes_a == bytes_b

    def test_to_dict_from_dict_roundtrip(self, alice, clock):
        original = SignedProfileEntry.sign(
            identity=alice,
            layer_latency_ms=1.5,
            rtt_to_peers={"bob": 5.0, "carol": 3.0},
            timestamp_unix=clock(),
        )
        round_trip = SignedProfileEntry.from_dict(original.to_dict())
        assert round_trip == original


# ──────────────────────────────────────────────────────────────────────────
# Anchor verification
# ──────────────────────────────────────────────────────────────────────────


class TestAnchorVerification:
    def test_verifies_under_correct_pubkey(self, alice, anchor, clock):
        entry = SignedProfileEntry.sign(
            identity=alice,
            layer_latency_ms=1.0,
            rtt_to_peers={},
            timestamp_unix=clock(),
        )
        assert entry.verify_with_anchor(anchor) is True

    def test_rejects_unregistered_publisher(self, alice, clock):
        # Anchor that doesn't know alice.
        empty_anchor = FakeAnchor()
        entry = SignedProfileEntry.sign(
            identity=alice,
            layer_latency_ms=1.0,
            rtt_to_peers={},
            timestamp_unix=clock(),
        )
        assert entry.verify_with_anchor(empty_anchor) is False

    def test_rejects_tampered_signature(self, alice, anchor, clock):
        entry = SignedProfileEntry.sign(
            identity=alice,
            layer_latency_ms=1.0,
            rtt_to_peers={},
            timestamp_unix=clock(),
        )
        # Flip the first base64 char to a DIFFERENT one. Hardcoding "A" was a
        # NO-OP whenever the real signature already began with "A" — b64[0]=="A"
        # iff signature byte0 < 4, i.e. 4/256 = 1/64 of freshly-generated Ed25519
        # keys. The "tampered" entry was then byte-identical to the genuine one,
        # verified True (correctly!), and the test failed "assert True is False"
        # on ~1.6% of runs. Measured: of 3000 fresh identities, 59 signatures
        # started with "A" and all 59 "tampers" were no-ops; zero GENUINE tampers
        # were ever accepted. The verifier was never the problem — the test was
        # signing its own no-op.
        sig = entry.signature_b64
        flipped = ("B" if sig[0] == "A" else "A") + sig[1:]
        assert flipped != sig, "tamper must actually change the signature"
        tampered = SignedProfileEntry(
            node_id=entry.node_id,
            layer_latency_ms=entry.layer_latency_ms,
            rtt_to_peers=dict(entry.rtt_to_peers),
            timestamp_unix=entry.timestamp_unix,
            signature_b64=flipped,
        )
        assert tampered.verify_with_anchor(anchor) is False

    def test_rejects_tampered_payload(self, alice, anchor, clock):
        # Modify the payload but keep the original signature → mismatch.
        entry = SignedProfileEntry.sign(
            identity=alice,
            layer_latency_ms=1.0,
            rtt_to_peers={},
            timestamp_unix=clock(),
        )
        tampered = SignedProfileEntry(
            node_id=entry.node_id,
            layer_latency_ms=999.0,  # different from signed payload
            rtt_to_peers=dict(entry.rtt_to_peers),
            timestamp_unix=entry.timestamp_unix,
            signature_b64=entry.signature_b64,
        )
        assert tampered.verify_with_anchor(anchor) is False

    def test_rejects_wrong_publisher_pubkey(self, alice, bob, clock):
        # Anchor that maps alice's node_id to bob's pubkey — alice's
        # signature won't verify under bob's key.
        wrong_anchor = FakeAnchor()
        wrong_anchor.register(alice.node_id, bob.public_key_b64)
        entry = SignedProfileEntry.sign(
            identity=alice,
            layer_latency_ms=1.0,
            rtt_to_peers={},
            timestamp_unix=clock(),
        )
        assert entry.verify_with_anchor(wrong_anchor) is False

    def test_returns_false_when_anchor_is_none(self, alice, clock):
        entry = SignedProfileEntry.sign(
            identity=alice,
            layer_latency_ms=1.0,
            rtt_to_peers={},
            timestamp_unix=clock(),
        )
        assert entry.verify_with_anchor(None) is False


# ──────────────────────────────────────────────────────────────────────────
# Wire codec
# ──────────────────────────────────────────────────────────────────────────


class TestWireCodec:
    def test_roundtrip_publish_request(self, alice, clock):
        entry = SignedProfileEntry.sign(
            identity=alice, layer_latency_ms=1.0, rtt_to_peers={},
            timestamp_unix=clock(),
        )
        request = PublishProfileRequest(request_id="r1", entry=entry)
        wire = encode_message(request)
        parsed = parse_message(wire)
        assert isinstance(parsed, PublishProfileRequest)
        assert parsed.entry == entry

    def test_roundtrip_fetch_request(self):
        request = FetchProfileRequest(request_id="r1", node_id="x")
        parsed = parse_message(encode_message(request))
        assert isinstance(parsed, FetchProfileRequest)
        assert parsed.node_id == "x"

    def test_roundtrip_fetch_response_with_entry(self, alice, clock):
        entry = SignedProfileEntry.sign(
            identity=alice, layer_latency_ms=1.0, rtt_to_peers={},
            timestamp_unix=clock(),
        )
        response = FetchResponse(request_id="r1", entry=entry)
        parsed = parse_message(encode_message(response))
        assert isinstance(parsed, FetchResponse)
        assert parsed.entry == entry

    def test_roundtrip_fetch_response_with_null_entry(self):
        response = FetchResponse(request_id="r1", entry=None)
        parsed = parse_message(encode_message(response))
        assert isinstance(parsed, FetchResponse)
        assert parsed.entry is None

    def test_roundtrip_error_response(self):
        err = ErrorResponse(
            request_id="r1",
            code=ProfileErrorCode.NOT_FOUND.value,
            message="missing",
        )
        parsed = parse_message(encode_message(err))
        assert isinstance(parsed, ErrorResponse)
        assert parsed.code == "NOT_FOUND"

    def test_oversized_payload_rejected(self):
        oversized = b"a" * (MAX_MESSAGE_BYTES + 1)
        with pytest.raises(ProfileMalformedError, match="MAX_MESSAGE_BYTES"):
            parse_message(oversized)

    def test_protocol_version_mismatch_rejected(self):
        import json as _json
        body = _json.dumps({
            "type": ProfileMessageType.FETCH_PROFILE.value,
            "protocol_version": PROFILE_DHT_PROTOCOL_VERSION + 1,
            "request_id": "r1",
            "node_id": "x",
        }).encode("utf-8")
        with pytest.raises(ProfileVersionMismatchError):
            parse_message(body)

    def test_unknown_type_rejected(self):
        import json as _json
        body = _json.dumps({
            "type": "unknown_type",
            "protocol_version": PROFILE_DHT_PROTOCOL_VERSION,
            "request_id": "r1",
        }).encode("utf-8")
        with pytest.raises(ProfileUnknownTypeError):
            parse_message(body)


# ──────────────────────────────────────────────────────────────────────────
# Server: publish + fetch happy paths
# ──────────────────────────────────────────────────────────────────────────


class TestServerPublishFetch:
    def test_publish_then_fetch(self, server, alice, clock):
        entry = SignedProfileEntry.sign(
            identity=alice, layer_latency_ms=1.0,
            rtt_to_peers={"x": 2.0}, timestamp_unix=clock(),
        )
        publish_req = PublishProfileRequest(request_id="p1", entry=entry)
        response_bytes = server.handle(encode_message(publish_req))
        response = parse_message(response_bytes)
        assert isinstance(response, PublishResponse)
        assert response.accepted is True

        fetch_req = FetchProfileRequest(request_id="f1", node_id=alice.node_id)
        fetch_response = parse_message(server.handle(encode_message(fetch_req)))
        assert isinstance(fetch_response, FetchResponse)
        assert fetch_response.entry == entry

    def test_fetch_unknown_returns_null_entry(self, server):
        fetch_req = FetchProfileRequest(request_id="f1", node_id="never-published")
        response = parse_message(server.handle(encode_message(fetch_req)))
        assert isinstance(response, FetchResponse)
        assert response.entry is None

    def test_publish_unregistered_publisher_rejected(self, server, clock):
        # alice is registered, but new_node is not.
        new_node = generate_node_identity(display_name="ghost")
        entry = SignedProfileEntry.sign(
            identity=new_node, layer_latency_ms=1.0,
            rtt_to_peers={}, timestamp_unix=clock(),
        )
        request = PublishProfileRequest(request_id="p1", entry=entry)
        response = parse_message(server.handle(encode_message(request)))
        assert isinstance(response, ErrorResponse)
        assert response.code == ProfileErrorCode.UNREGISTERED_PUBLISHER.value

    def test_publish_tampered_signature_rejected(self, server, alice, clock):
        entry = SignedProfileEntry.sign(
            identity=alice, layer_latency_ms=1.0,
            rtt_to_peers={}, timestamp_unix=clock(),
        )
        tampered = SignedProfileEntry(
            node_id=entry.node_id,
            layer_latency_ms=999.0,  # mismatch with signed payload
            rtt_to_peers=dict(entry.rtt_to_peers),
            timestamp_unix=entry.timestamp_unix,
            signature_b64=entry.signature_b64,
        )
        request = PublishProfileRequest(request_id="p1", entry=tampered)
        response = parse_message(server.handle(encode_message(request)))
        assert isinstance(response, ErrorResponse)
        assert response.code == ProfileErrorCode.SIGNATURE_INVALID.value


# ──────────────────────────────────────────────────────────────────────────
# Server: TTL + staleness
# ──────────────────────────────────────────────────────────────────────────


class TestServerTTL:
    def test_stale_on_arrival_rejected(self, server, alice, clock):
        # Publish an entry whose timestamp is already past TTL.
        old_entry = SignedProfileEntry.sign(
            identity=alice, layer_latency_ms=1.0,
            rtt_to_peers={},
            timestamp_unix=clock() - DEFAULT_TTL_SECONDS - 1.0,
        )
        request = PublishProfileRequest(request_id="p1", entry=old_entry)
        response = parse_message(server.handle(encode_message(request)))
        assert isinstance(response, ErrorResponse)
        assert response.code == ProfileErrorCode.STALE_ENTRY.value

    def test_fresh_entry_becomes_stale_after_ttl(self, server, alice, clock):
        entry = SignedProfileEntry.sign(
            identity=alice, layer_latency_ms=1.0,
            rtt_to_peers={}, timestamp_unix=clock(),
        )
        publish_req = PublishProfileRequest(request_id="p1", entry=entry)
        server.handle(encode_message(publish_req))

        # Right after publish: cached.
        assert server.get_cached(alice.node_id) is not None

        # Advance past TTL: cache returns None.
        clock.advance(DEFAULT_TTL_SECONDS + 0.1)
        assert server.get_cached(alice.node_id) is None

    def test_evict_stale_drops_old_entries(self, server, alice, clock):
        entry = SignedProfileEntry.sign(
            identity=alice, layer_latency_ms=1.0,
            rtt_to_peers={}, timestamp_unix=clock(),
        )
        server.handle(encode_message(PublishProfileRequest(
            request_id="p1", entry=entry,
        )))
        clock.advance(DEFAULT_TTL_SECONDS + 0.1)
        evicted = server.evict_stale()
        assert evicted == 1


# ──────────────────────────────────────────────────────────────────────────
# Server: concurrent updates / latest-wins
# ──────────────────────────────────────────────────────────────────────────


class TestServerLatestWins:
    def test_newer_timestamp_overwrites(self, server, alice, clock):
        first = SignedProfileEntry.sign(
            identity=alice, layer_latency_ms=1.0,
            rtt_to_peers={}, timestamp_unix=clock(),
        )
        clock.advance(0.5)
        second = SignedProfileEntry.sign(
            identity=alice, layer_latency_ms=2.0,
            rtt_to_peers={}, timestamp_unix=clock(),
        )
        server.handle(encode_message(
            PublishProfileRequest(request_id="p1", entry=first)
        ))
        server.handle(encode_message(
            PublishProfileRequest(request_id="p2", entry=second)
        ))
        cached = server.get_cached(alice.node_id)
        assert cached == second

    def test_older_timestamp_not_overwriting(self, server, alice, clock):
        # Out-of-order delivery: newer publish first, then older.
        # Older does NOT overwrite the cached newer.
        clock.advance(1.0)
        newer = SignedProfileEntry.sign(
            identity=alice, layer_latency_ms=2.0,
            rtt_to_peers={}, timestamp_unix=clock(),
        )
        # Build older by hand-rolling timestamp.
        older = SignedProfileEntry.sign(
            identity=alice, layer_latency_ms=1.0,
            rtt_to_peers={}, timestamp_unix=clock() - 0.5,
        )
        server.handle(encode_message(
            PublishProfileRequest(request_id="p1", entry=newer)
        ))
        response_bytes = server.handle(encode_message(
            PublishProfileRequest(request_id="p2", entry=older)
        ))
        response = parse_message(response_bytes)
        assert isinstance(response, PublishResponse)
        assert response.accepted is False
        # Cached entry remains the newer one.
        assert server.get_cached(alice.node_id) == newer


# ──────────────────────────────────────────────────────────────────────────
# Server: never-raises invariant
# ──────────────────────────────────────────────────────────────────────────


class TestServerNeverRaises:
    @pytest.mark.parametrize("payload", [
        b"",
        b"\x00\x01\x02",
        b"{",
        b"null",
        b"42",
        b'"just a string"',
        b'{"type": null}',
        b'{"type": ""}',
        b'{"type": "publish_profile"}',  # missing other required fields
        b'{"type": "fetch_profile", "protocol_version": "not-an-int"}',
    ])
    def test_garbage_input_returns_encoded_error(self, server, payload):
        response_bytes = server.handle(payload)
        # Must decode as ErrorResponse — doesn't raise.
        response = parse_message(response_bytes)
        assert isinstance(response, ErrorResponse)


# ──────────────────────────────────────────────────────────────────────────
# Server construction guards
# ──────────────────────────────────────────────────────────────────────────


class TestServerConstruction:
    def test_anchor_required(self):
        with pytest.raises(RuntimeError, match="anchor"):
            ProfileDHTServer(anchor=None)

    def test_anchor_must_have_lookup(self):
        with pytest.raises(RuntimeError, match="lookup"):
            ProfileDHTServer(anchor=object())


# ──────────────────────────────────────────────────────────────────────────
# Client + ProfileSource integration
# ──────────────────────────────────────────────────────────────────────────


class FakeNetwork:
    """In-process synchronous bus: address → server. Exposes a
    send_message callable for client injection."""

    def __init__(self):
        self.servers: dict[str, ProfileDHTServer] = {}

    def register(self, address: str, server: ProfileDHTServer) -> None:
        self.servers[address] = server

    def send(self, address: str, request_bytes: bytes) -> bytes:
        srv = self.servers.get(address)
        if srv is None:
            raise ConnectionRefusedError(f"no peer at {address}")
        return srv.handle(request_bytes)


class TestProfileDHTClient:
    def test_publish_self_round_trip(self, alice, anchor, clock):
        network = FakeNetwork()
        # Bob runs a server; Alice publishes to Bob.
        bob_server = ProfileDHTServer(anchor=anchor, clock=clock)
        network.register("bob:8000", bob_server)

        alice_dht = ProfileDHT(
            identity=alice,
            anchor=anchor,
            send_message=network.send,
            peers=["bob:8000"],
            clock=clock,
        )
        accepted = alice_dht.publish_self(
            layer_latency_ms=1.5,
            rtt_to_peers={alice.node_id: 0.0, "bob": 5.0},
        )
        assert accepted == 1
        # Bob's cache now has alice's entry.
        cached = bob_server.get_cached(alice.node_id)
        assert cached is not None
        assert cached.layer_latency_ms == 1.5

    def test_publish_self_caches_locally(self, alice, anchor, clock):
        network = FakeNetwork()
        alice_dht = ProfileDHT(
            identity=alice, anchor=anchor, send_message=network.send,
            peers=[], clock=clock,
        )
        # No peers configured — publish_self still caches locally.
        alice_dht.publish_self(layer_latency_ms=2.5, rtt_to_peers={})
        snap = alice_dht.get_snapshot(alice.node_id)
        assert snap is not None
        assert snap.layer_latency_ms == 2.5

    def test_get_snapshot_falls_through_to_peer(
        self, alice, bob, anchor, clock
    ):
        network = FakeNetwork()
        bob_server = ProfileDHTServer(anchor=anchor, clock=clock)
        network.register("bob:8000", bob_server)
        # Bob caches alice's entry first.
        alice_entry = SignedProfileEntry.sign(
            identity=alice, layer_latency_ms=1.5, rtt_to_peers={},
            timestamp_unix=clock(),
        )
        bob_server.handle(encode_message(
            PublishProfileRequest(request_id="p1", entry=alice_entry)
        ))

        # Carol queries — has only bob as peer; bob serves alice's entry.
        carol = generate_node_identity(display_name="carol")
        anchor.register(carol.node_id, carol.public_key_b64)
        carol_dht = ProfileDHT(
            identity=carol, anchor=anchor, send_message=network.send,
            peers=["bob:8000"], clock=clock,
        )
        snap = carol_dht.get_snapshot(alice.node_id)
        assert snap is not None
        assert snap.layer_latency_ms == 1.5

    def test_get_snapshot_returns_none_when_no_peer_has_it(
        self, alice, anchor, clock
    ):
        network = FakeNetwork()
        bob_server = ProfileDHTServer(anchor=anchor, clock=clock)
        network.register("bob:8000", bob_server)

        carol = generate_node_identity(display_name="carol")
        anchor.register(carol.node_id, carol.public_key_b64)
        carol_dht = ProfileDHT(
            identity=carol, anchor=anchor, send_message=network.send,
            peers=["bob:8000"], clock=clock,
        )
        snap = carol_dht.get_snapshot(alice.node_id)
        assert snap is None

    def test_unreachable_peer_does_not_break_publish(
        self, alice, anchor, clock
    ):
        # Peer raises ConnectionRefusedError — publish_self skips
        # that peer, doesn't propagate the exception.
        network = FakeNetwork()  # no servers registered
        alice_dht = ProfileDHT(
            identity=alice, anchor=anchor, send_message=network.send,
            peers=["dead:8000"], clock=clock,
        )
        accepted = alice_dht.publish_self(
            layer_latency_ms=1.0, rtt_to_peers={},
        )
        assert accepted == 0  # nothing succeeded but no exception
        # Local cache still populated.
        assert alice_dht.get_snapshot(alice.node_id) is not None

    def test_fetch_peer_rejects_anchor_failed_response(
        self, alice, anchor, clock
    ):
        # Bob's server returns alice's entry, but in carol's anchor
        # alice is NOT registered → fetch_peer rejects the bytes.
        network = FakeNetwork()
        bob_server = ProfileDHTServer(anchor=anchor, clock=clock)
        network.register("bob:8000", bob_server)

        alice_entry = SignedProfileEntry.sign(
            identity=alice, layer_latency_ms=1.5, rtt_to_peers={},
            timestamp_unix=clock(),
        )
        bob_server.handle(encode_message(
            PublishProfileRequest(request_id="p1", entry=alice_entry)
        ))

        # Carol uses a DIFFERENT anchor that doesn't know alice.
        carol_anchor = FakeAnchor()
        carol = generate_node_identity(display_name="carol")
        carol_anchor.register(carol.node_id, carol.public_key_b64)
        carol_dht = ProfileDHT(
            identity=carol, anchor=carol_anchor,
            send_message=network.send,
            peers=["bob:8000"], clock=clock,
        )
        # Bob serves alice's entry, but carol's anchor rejects → None.
        snap = carol_dht.get_snapshot(alice.node_id)
        assert snap is None


# ──────────────────────────────────────────────────────────────────────────
# Constants sanity
# ──────────────────────────────────────────────────────────────────────────


class TestConstants:
    def test_protocol_version_is_one(self):
        assert PROFILE_DHT_PROTOCOL_VERSION == 1

    def test_max_message_bytes_reasonable(self):
        # 64 KB: large enough for ~64 small profile entries, small
        # enough to bound DoS amplification.
        assert MAX_MESSAGE_BYTES == 64 * 1024

    def test_default_publish_interval_matches_paper(self):
        # Paper §3.3: republish every 1-2 seconds.
        assert DEFAULT_PUBLISH_INTERVAL_SECONDS == 2.0

    def test_default_ttl_provides_one_missed_window(self):
        # 5s TTL with 2s publish = 1 missed publish + 1s grace.
        assert DEFAULT_TTL_SECONDS == 5.0
        assert DEFAULT_TTL_SECONDS > 2 * DEFAULT_PUBLISH_INTERVAL_SECONDS
