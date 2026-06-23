"""Sprint 1229 — application-level P2P keepalive for idle libp2p streams.

Surfaced by the 72B-across-10-hosts run (2026-06-23): a downstream chain
stage waits many minutes for its turn while upstream stages cold-load
multi-GB slices. With NO periodic traffic on that idle libp2p stream, the
connection is reaped (cloud TCP / libp2p idle timeout), so the head's
eventual dispatch fails with ``TRANSPORT_ERROR: TimeoutError`` (or the peer
falls out of ``transport.peers`` entirely). The fix: a background keepalive
task that periodically pings every connected peer so the bidirectional
stream stays warm through long idle/load windows.

These tests run WITHOUT the Go shared library — ctypes is mocked.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prsm.node.libp2p_transport import Libp2pTransport


def _make_mock_identity():
    ident = MagicMock()
    ident.node_id = "abcdef1234567890abcdef1234567890"
    ident.private_key_bytes = b"\x01" * 32
    ident.public_key_bytes = b"\x02" * 32
    ident.sign.return_value = "fakesig"
    return ident


def _make_transport():
    identity = _make_mock_identity()
    lib = MagicMock()
    with patch.object(Libp2pTransport, "_load_library", return_value=lib):
        t = Libp2pTransport(identity=identity, port=9001)
    return t


# ── interval resolution ────────────────────────────────────────────

def test_keepalive_interval_default(monkeypatch):
    monkeypatch.delenv("PRSM_P2P_KEEPALIVE_INTERVAL_S", raising=False)
    t = _make_transport()
    assert t._keepalive_interval == pytest.approx(30.0)


def test_keepalive_interval_env_override(monkeypatch):
    monkeypatch.setenv("PRSM_P2P_KEEPALIVE_INTERVAL_S", "5")
    t = _make_transport()
    assert t._keepalive_interval == pytest.approx(5.0)


def test_keepalive_interval_zero_disables(monkeypatch):
    monkeypatch.setenv("PRSM_P2P_KEEPALIVE_INTERVAL_S", "0")
    t = _make_transport()
    assert t._keepalive_interval == 0.0


def test_keepalive_interval_garbage_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("PRSM_P2P_KEEPALIVE_INTERVAL_S", "not-a-number")
    t = _make_transport()
    assert t._keepalive_interval == pytest.approx(30.0)


# ── ping behaviour ─────────────────────────────────────────────────

def test_ping_all_peers_pings_every_connected_peer():
    t = _make_transport()
    t._peers = {"peerA": object(), "peerB": object(), "peerC": object()}
    sent = []

    async def fake_send(peer_id, msg):
        sent.append((peer_id, msg))
        return True

    t.send_to_peer = AsyncMock(side_effect=fake_send)
    asyncio.run(t._ping_all_peers())

    assert {p for p, _ in sent} == {"peerA", "peerB", "peerC"}
    for _, msg in sent:
        assert msg.msg_type == "keepalive_ping"
        assert msg.sender_id == t._identity.node_id
    assert t._telemetry["keepalive_pings_sent"] == 3


def test_ping_all_peers_survives_a_single_send_failure():
    t = _make_transport()
    t._peers = {"good1": object(), "bad": object(), "good2": object()}
    attempted = []

    async def flaky_send(peer_id, msg):
        attempted.append(peer_id)
        if peer_id == "bad":
            raise RuntimeError("stream reset")
        return True

    t.send_to_peer = AsyncMock(side_effect=flaky_send)
    # Must NOT raise — one bad peer can't kill the keepalive loop.
    asyncio.run(t._ping_all_peers())

    assert set(attempted) == {"good1", "bad", "good2"}
    # the two good peers counted; the failed one did not
    assert t._telemetry["keepalive_pings_sent"] == 2


def test_ping_all_peers_no_peers_is_noop():
    t = _make_transport()
    t._peers = {}
    t.send_to_peer = AsyncMock()
    asyncio.run(t._ping_all_peers())
    t.send_to_peer.assert_not_called()
    assert t._telemetry["keepalive_pings_sent"] == 0


# ── lifecycle: task started on start(), cancelled on stop() ─────────

def test_keepalive_task_started_and_cancelled(monkeypatch):
    monkeypatch.setenv("PRSM_P2P_KEEPALIVE_INTERVAL_S", "0.01")
    t = _make_transport()

    async def scenario():
        # Stub the parts of start() that need the real Go daemon/UDS.
        t._lib.PrsmStart = MagicMock(return_value=7)
        pinged = asyncio.Event()

        async def fake_ping():
            pinged.set()

        t._ping_all_peers = fake_ping
        with patch.object(Libp2pTransport, "_uds_reader", new=AsyncMock()):
            await t.start()
            assert t._keepalive_task is not None
            assert not t._keepalive_task.done()
            # deterministic: wait for the first real ping (not wall-clock)
            await asyncio.wait_for(pinged.wait(), timeout=2.0)
            assert not t._keepalive_task.done()
            t._lib.PrsmStop = MagicMock(return_value=0)
            await t.stop()
            assert t._keepalive_task is None

    asyncio.run(scenario())


def test_keepalive_task_not_started_when_disabled(monkeypatch):
    monkeypatch.setenv("PRSM_P2P_KEEPALIVE_INTERVAL_S", "0")
    t = _make_transport()

    async def scenario():
        t._lib.PrsmStart = MagicMock(return_value=7)
        with patch.object(Libp2pTransport, "_uds_reader", new=AsyncMock()):
            await t.start()
            assert t._keepalive_task is None
            t._lib.PrsmStop = MagicMock(return_value=0)
            await t.stop()

    asyncio.run(scenario())


# ── incoming ping is harmless ───────────────────────────────────────

def test_incoming_keepalive_ping_is_noop():
    t = _make_transport()
    before_fail = t._telemetry["dispatch_failure_total"]
    before_recv = t._telemetry["messages_received"]

    async def scenario():
        await t._dispatch({
            "msg_type": "keepalive_ping",
            "sender_id": "peerX",
            "payload": {"subtype": "keepalive_ping"},
        })

    asyncio.run(scenario())
    # received counted, no handler registered -> no failure, no crash
    assert t._telemetry["messages_received"] == before_recv + 1
    assert t._telemetry["dispatch_failure_total"] == before_fail


def test_keepalive_pings_sent_in_telemetry_snapshot():
    t = _make_transport()
    snap = t.get_telemetry_snapshot()
    assert "keepalive_pings_sent" in snap
    assert snap["keepalive_pings_sent"] == 0


# ── libp2p direct-wrapped frame must NOT reach direct handlers ──────

def test_incoming_keepalive_libp2p_direct_frame_short_circuits():
    """The Go daemon delivers a /prsm/direct/1.0.0 message as
    {"msg_type":"direct","data":<inner-json-str>,...}. A keepalive in that
    wrapper must be recognized and dropped BEFORE handler fan-out — otherwise
    it reaches every registered 'direct' handler, errors on the string
    payload, and inflates dispatch_failure_total (the sp1217 metrics)."""
    import json

    t = _make_transport()
    calls = []

    async def direct_handler(msg, peer):
        # mirrors real direct handlers (compute_provider, ledger_sync, …)
        calls.append(msg)
        return msg.payload.get("subtype")  # would AttributeError on a str

    t.on_message("direct", direct_handler)

    inner = json.dumps({
        "msg_type": "keepalive_ping",
        "sender_id": "peerX",
        "payload": {"subtype": "keepalive_ping", "ts": 1.0},
        "signature": "sig", "ttl": 5, "nonce": "abc",
    })
    before_fail = t._telemetry["dispatch_failure_total"]
    before_ok = t._telemetry["dispatch_success_total"]

    asyncio.run(t._dispatch({
        "msg_type": "direct",
        "sender_id": "peerX",
        "data": inner,
    }))

    assert calls == []  # handler never invoked
    assert t._telemetry["dispatch_failure_total"] == before_fail
    assert t._telemetry["dispatch_success_total"] == before_ok


def test_non_keepalive_direct_frame_still_reaches_handlers():
    """Guard: the short-circuit must be keepalive-specific — a real direct
    message still fans out to its handlers."""
    import json

    t = _make_transport()
    seen = []

    async def direct_handler(msg, peer):
        seen.append(msg.sender_id)

    t.on_message("direct", direct_handler)
    asyncio.run(t._dispatch({
        "msg_type": "direct",
        "sender_id": "peerZ",
        "data": json.dumps({"msg_type": "direct", "payload": {"subtype": "real_work"}}),
    }))
    assert seen == ["peerZ"]


# ── inbound peers get warmed too (head-side fix) ────────────────────

def test_inbound_peer_recorded_and_pinged_even_with_empty_peers():
    """A node that only RECEIVES a dial (empty self._peers) must still ping
    that peer back — this is the head's case in the pipeline topology."""
    t = _make_transport()
    assert t._peers == {}

    async def scenario():
        # receive any message from peerH -> records it as an inbound peer
        await t._dispatch({"msg_type": "discovery", "sender_id": "peerH",
                           "payload": {"subtype": "hello"}})
        assert "peerH" in t._inbound_peer_ids

        sent = []
        async def fake_send(peer_id, msg):
            sent.append(peer_id)
            return True
        t.send_to_peer = AsyncMock(side_effect=fake_send)
        await t._ping_all_peers()
        assert sent == ["peerH"]  # pinged despite empty _peers
        assert t._telemetry["keepalive_pings_sent"] == 1

    asyncio.run(scenario())


def test_inbound_peer_pruned_when_send_fails():
    t = _make_transport()

    async def scenario():
        await t._dispatch({"msg_type": "discovery", "sender_id": "goneH",
                           "payload": {}})
        assert "goneH" in t._inbound_peer_ids
        t.send_to_peer = AsyncMock(return_value=False)  # unreachable
        await t._ping_all_peers()
        assert "goneH" not in t._inbound_peer_ids  # pruned
        assert t._telemetry["keepalive_pings_sent"] == 0

    asyncio.run(scenario())


# ── loop self-heals across a bad sweep ──────────────────────────────

def test_keepalive_loop_self_heals_after_sweep_error(monkeypatch):
    monkeypatch.setenv("PRSM_P2P_KEEPALIVE_INTERVAL_S", "0.01")
    t = _make_transport()

    async def scenario():
        t._running = True
        calls = {"n": 0}
        ok = asyncio.Event()

        async def flaky_sweep():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient sweep failure")
            ok.set()

        t._ping_all_peers = flaky_sweep
        task = asyncio.create_task(t._keepalive_loop())
        # loop must survive the first raise and fire a second sweep
        await asyncio.wait_for(ok.wait(), timeout=2.0)
        assert not task.done()  # NOT permanently disabled
        t._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())
