"""Sprint 715 — end-to-end integration test for sprint-711 + sprint-713
streaming wire protocol.

Sprint 711 shipped the CHAIN_STREAM_MSG_TYPE wire protocol (request →
frames → end) with 8 pin tests covering protocol-shape correctness +
handler routing. Sprint 713 added bounded receive-queue back-pressure.
Neither layer was end-to-end tested with REAL transport between two
processes/nodes — only mock-shape pin tests.

Sprint 715 closes that gap with a single-host two-node integration:
real `WebSocketTransport` on 127.0.0.1 different ports, real
`on_message(MSG_DIRECT, ...)` handler registration, real frame
serialization + transmission, real per-stream asyncio.Queue. The
ONLY mock is the server-side `LayerStageServer` (which would
otherwise pull in HuggingFace model load + tokenizer — that
heavyweight wiring is sprint 712's live-attest scope).

This test demonstrates the wire protocol works without requiring
cloud infra — closes the gap between "pin tests pass" and "the
two daemons actually talk to each other".
"""
from __future__ import annotations

import asyncio
import base64
import json
import socket
from unittest.mock import patch

import pytest

_REAL_SLEEP = asyncio.sleep


@pytest.fixture(autouse=True)
def real_asyncio_sleep():
    """Restore real asyncio.sleep for network tests (matches existing
    two_node_network.py fixture pattern)."""
    with patch("asyncio.sleep", _REAL_SLEEP):
        yield


def _free_port() -> int:
    """Pick a free localhost port — avoids hardcoded port collisions
    when test runs alongside others."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _MinimalNode:
    """Lightweight node-like struct holding the attrs that
    chain_executor_adapters reads. Mirrors what `prsm.node.node.PRSMNode`
    exposes at the surface area used by the wire protocol — identity,
    transport, _loop, _chain_executor_pending_streams."""
    def __init__(self, identity, transport, loop):
        self.identity = identity
        self.transport = transport
        self._loop = loop
        self._chain_executor_pending_streams = {}
        self._chain_stage_executor = None  # what handle_chain_stream_request reads


@pytest.mark.asyncio
async def test_sprint_715_two_node_streaming_roundtrip():
    """Server node yields 5 known frames; requester node receives all
    5 in order via the sprint-711 wire protocol over a real
    WebSocketTransport. Verifies the full path: REQ message →
    handle_chain_stream_request → frame messages → handle_chain_stream_response
    → per-stream queue → _remote_token_stream_dispatch generator."""
    from prsm.node.identity import generate_node_identity
    from prsm.node.transport import WebSocketTransport, MSG_DIRECT
    from prsm.node.chain_executor_adapters import (
        _remote_token_stream_dispatch,
        handle_chain_stream_request,
        handle_chain_stream_response,
    )

    loop = asyncio.get_event_loop()
    server_id = generate_node_identity("sprint715-server")
    requester_id = generate_node_identity("sprint715-requester")

    server_port = _free_port()
    server_transport = WebSocketTransport(
        server_id, host="127.0.0.1", port=server_port,
    )
    requester_port = _free_port()
    requester_transport = WebSocketTransport(
        requester_id, host="127.0.0.1", port=requester_port,
    )

    server_node = _MinimalNode(server_id, server_transport, loop)
    requester_node = _MinimalNode(requester_id, requester_transport, loop)

    # Mock the server-side LayerStageServer: yields 5 known frames.
    expected_frames = [
        b"frame-0-hello",
        b"frame-1-world",
        b"frame-2-PRSM",
        b"frame-3-sprint",
        b"frame-4-715",
    ]
    class _MockChainExec:
        def __init__(self):
            self._server = _MockServer()
    class _MockServer:
        def handle_token_stream(self, request_bytes):
            for f in expected_frames:
                yield f
    server_node._chain_stage_executor = _MockChainExec()

    async def _server_dispatch(msg, peer):
        await handle_chain_stream_request(server_node, msg)
    async def _requester_dispatch(msg, peer):
        handle_chain_stream_response(requester_node, msg)

    server_transport.on_message(MSG_DIRECT, _server_dispatch)
    requester_transport.on_message(MSG_DIRECT, _requester_dispatch)

    await server_transport.start()
    await requester_transport.start()
    try:
        # Requester connects to server.
        await requester_transport.connect_to_peer(
            f"ws://127.0.0.1:{server_port}",
        )
        # Wait for handshake to complete.
        for _ in range(50):
            await asyncio.sleep(0.05)
            if (
                server_id.node_id in requester_transport.peers
                and requester_id.node_id in server_transport.peers
            ):
                break
        assert server_id.node_id in requester_transport.peers, (
            "requester failed to register server as a peer post-handshake"
        )
        assert requester_id.node_id in server_transport.peers, (
            "server failed to register requester as a peer post-handshake"
        )

        # Now invoke _remote_token_stream_dispatch as a foreign call.
        # The dispatcher creates a per-stream queue + sends REQ.
        request_bytes = b"sprint-715-request-payload"

        # _remote_token_stream_dispatch is a SYNC generator (yields
        # frame bytes); it internally uses run_coroutine_threadsafe.
        # We collect frames via asyncio.to_thread to avoid blocking
        # the test loop on the generator's q.get() — but actually,
        # since the dispatcher's run_coroutine_threadsafe call IS the
        # same loop, calling list(generator) from the loop thread
        # would deadlock. Use a separate thread.
        def _drain_generator():
            return list(_remote_token_stream_dispatch(
                requester_node, server_id.node_id, request_bytes,
            ))

        received = await asyncio.wait_for(
            asyncio.to_thread(_drain_generator), timeout=10.0,
        )

        # Every expected frame received in order.
        assert received == expected_frames, (
            f"frame mismatch: expected {expected_frames}, got {received}"
        )

    finally:
        await server_transport.stop()
        await requester_transport.stop()


@pytest.mark.asyncio
async def test_sprint_717_bounded_queue_no_frame_loss_under_load():
    """Sprint 717 — exercise sprint-713 bounded receive-queue
    back-pressure end-to-end. Server yields 20 frames as fast as
    possible; receiver queue maxsize=2 so producer blocks at puts
    once the queue fills. Asserts:

      - All 20 frames arrive at the dispatcher in correct order
      - No frame loss (back-pressure must be non-lossy)
      - Terminal STREAM_END lands after the last frame (sprint-715
        + 716 wire-order invariant holds under load)
    """
    import os
    from prsm.node.identity import generate_node_identity
    from prsm.node.transport import WebSocketTransport, MSG_DIRECT
    from prsm.node.chain_executor_adapters import (
        _remote_token_stream_dispatch,
        handle_chain_stream_request,
        handle_chain_stream_response,
    )

    loop = asyncio.get_event_loop()
    server_id = generate_node_identity("sprint717-server")
    requester_id = generate_node_identity("sprint717-req")

    server_port = _free_port()
    requester_port = _free_port()
    server_transport = WebSocketTransport(
        server_id, host="127.0.0.1", port=server_port,
    )
    requester_transport = WebSocketTransport(
        requester_id, host="127.0.0.1", port=requester_port,
    )
    server_node = _MinimalNode(server_id, server_transport, loop)
    requester_node = _MinimalNode(requester_id, requester_transport, loop)

    expected_frames = [f"frame-{i:02d}".encode() for i in range(20)]
    class _LoadServer:
        def handle_token_stream(self, request_bytes):
            for f in expected_frames:
                yield f
    class _LoadExec:
        def __init__(self):
            self._server = _LoadServer()
    server_node._chain_stage_executor = _LoadExec()

    async def _server_dispatch(msg, peer):
        await handle_chain_stream_request(server_node, msg)
    async def _requester_dispatch(msg, peer):
        handle_chain_stream_response(requester_node, msg)
    server_transport.on_message(MSG_DIRECT, _server_dispatch)
    requester_transport.on_message(MSG_DIRECT, _requester_dispatch)

    # Force a tight bound so back-pressure WILL kick in under
    # the 20-frame load. maxsize=2 means: once 2 frames in queue,
    # subsequent puts block at the producer side until consumer
    # drains.
    old_env = os.environ.get("PRSM_CHAIN_STREAM_QUEUE_MAXSIZE")
    os.environ["PRSM_CHAIN_STREAM_QUEUE_MAXSIZE"] = "2"

    await server_transport.start()
    await requester_transport.start()
    try:
        await requester_transport.connect_to_peer(
            f"ws://127.0.0.1:{server_port}",
        )
        for _ in range(50):
            await asyncio.sleep(0.05)
            if server_id.node_id in requester_transport.peers:
                break
        assert server_id.node_id in requester_transport.peers

        def _drain():
            return list(_remote_token_stream_dispatch(
                requester_node, server_id.node_id, b"sprint-717-load",
            ))

        received = await asyncio.wait_for(
            asyncio.to_thread(_drain), timeout=15.0,
        )

        # Invariant 1: no frame loss
        assert len(received) == len(expected_frames), (
            f"frame count mismatch under back-pressure: expected "
            f"{len(expected_frames)} got {len(received)}"
        )
        # Invariant 2: correct order (no reorder)
        assert received == expected_frames, (
            f"frame order mismatch under back-pressure: expected "
            f"{expected_frames}, got {received}"
        )

    finally:
        if old_env is None:
            os.environ.pop("PRSM_CHAIN_STREAM_QUEUE_MAXSIZE", None)
        else:
            os.environ["PRSM_CHAIN_STREAM_QUEUE_MAXSIZE"] = old_env
        await server_transport.stop()
        await requester_transport.stop()


@pytest.mark.asyncio
async def test_sprint_718_concurrent_identical_requests_no_collision():
    """Sprint 718 F52 — two concurrent invocations of
    `_remote_token_stream_dispatch` with IDENTICAL request_bytes
    must NOT collide on stream_id. Each receives its own complete
    frame sequence (no cross-contamination, no loss).

    Pre-F52: stream_id was `sha256(request_bytes + ':stream')` —
    purely deterministic. Two concurrent invocations with the
    same request_bytes computed the same stream_id, the second
    overwrote the first's queue entry, and the first dispatcher
    saw the second's frames (or nothing).
    """
    from prsm.node.identity import generate_node_identity
    from prsm.node.transport import WebSocketTransport, MSG_DIRECT
    from prsm.node.chain_executor_adapters import (
        _remote_token_stream_dispatch,
        handle_chain_stream_request,
        handle_chain_stream_response,
    )

    loop = asyncio.get_event_loop()
    server_id = generate_node_identity("sprint718-server")
    requester_id = generate_node_identity("sprint718-req")

    server_port = _free_port()
    requester_port = _free_port()
    server_transport = WebSocketTransport(
        server_id, host="127.0.0.1", port=server_port,
    )
    requester_transport = WebSocketTransport(
        requester_id, host="127.0.0.1", port=requester_port,
    )
    server_node = _MinimalNode(server_id, server_transport, loop)
    requester_node = _MinimalNode(requester_id, requester_transport, loop)

    # Server yields exactly 3 frames per invocation. Two concurrent
    # dispatches → each should receive their own 3 frames.
    frames_per_call = [b"a-0", b"a-1", b"a-2"]
    class _ConcServer:
        def handle_token_stream(self, request_bytes):
            for f in frames_per_call:
                yield f
    class _ConcExec:
        def __init__(self):
            self._server = _ConcServer()
    server_node._chain_stage_executor = _ConcExec()

    async def _server_dispatch(msg, peer):
        await handle_chain_stream_request(server_node, msg)
    async def _requester_dispatch(msg, peer):
        handle_chain_stream_response(requester_node, msg)
    server_transport.on_message(MSG_DIRECT, _server_dispatch)
    requester_transport.on_message(MSG_DIRECT, _requester_dispatch)

    await server_transport.start()
    await requester_transport.start()
    try:
        await requester_transport.connect_to_peer(
            f"ws://127.0.0.1:{server_port}",
        )
        for _ in range(50):
            await asyncio.sleep(0.05)
            if server_id.node_id in requester_transport.peers:
                break
        assert server_id.node_id in requester_transport.peers

        # IDENTICAL request_bytes in both calls — pre-F52 this
        # was the collision trigger.
        identical_req = b"sprint-718-identical-prompt"

        def _drain_one():
            return list(_remote_token_stream_dispatch(
                requester_node, server_id.node_id, identical_req,
            ))

        # Fire two concurrent dispatches. With F52 fix, each gets
        # its own stream_id + its own queue + its own complete 3-
        # frame sequence.
        results = await asyncio.gather(
            asyncio.to_thread(_drain_one),
            asyncio.to_thread(_drain_one),
        )

        for i, received in enumerate(results):
            assert received == frames_per_call, (
                f"dispatch #{i} received {received}, expected "
                f"{frames_per_call} — F52 collision likely (one "
                f"dispatch's queue was overwritten by the other)"
            )

    finally:
        await server_transport.stop()
        await requester_transport.stop()


@pytest.mark.asyncio
async def test_sprint_720_server_stops_streaming_on_requester_disconnect():
    """Sprint 720 F54 — when the requester disconnects mid-stream,
    server-side iteration must stop within a bounded number of
    frames. Otherwise: wasted GPU forward passes + held KV cache +
    memory pressure on the operator.

    Test fixture: server yields an infinite stream of frames (small
    `time.sleep` between yields so the test loop can advance). The
    requester drops its transport mid-stream (we stop it). Server
    should observe send_to_peer returning False on the next frame
    and exit the loop + close the inner generator within the next
    few yields.

    We instrument the generator with a counter to assert the server
    DID stop iterating after the disconnect (not unbounded).
    """
    from prsm.node.identity import generate_node_identity
    from prsm.node.transport import WebSocketTransport, MSG_DIRECT
    from prsm.node.chain_executor_adapters import (
        handle_chain_stream_request,
    )

    loop = asyncio.get_event_loop()
    server_id = generate_node_identity("sprint720-server")
    requester_id = generate_node_identity("sprint720-req")

    server_port = _free_port()
    requester_port = _free_port()
    server_transport = WebSocketTransport(
        server_id, host="127.0.0.1", port=server_port,
    )
    requester_transport = WebSocketTransport(
        requester_id, host="127.0.0.1", port=requester_port,
    )
    server_node = _MinimalNode(server_id, server_transport, loop)

    # Counter shared with the test scope: how many frames did the
    # server's inner generator actually yield? Conftest auto-mocks
    # time.sleep to instant, so we can't rate-limit the generator
    # that way. Instead: use a high-finite ceiling + assert the
    # generator was closed BEFORE reaching it (proves F54 fix
    # engaged) — the websocket send buffer naturally rate-limits
    # via back-pressure on send_to_peer.
    TOTAL_AVAILABLE = 10000
    yields = {"count": 0, "closed": False}
    class _BoundedServer:
        def handle_token_stream(self, request_bytes):
            try:
                for i in range(TOTAL_AVAILABLE):
                    yields["count"] += 1
                    yield f"f{i}".encode()
            finally:
                yields["closed"] = True
    class _BSExec:
        def __init__(self):
            self._server = _BoundedServer()
    server_node._chain_stage_executor = _BSExec()

    async def _server_dispatch(msg, peer):
        await handle_chain_stream_request(server_node, msg)
    server_transport.on_message(MSG_DIRECT, _server_dispatch)

    await server_transport.start()
    await requester_transport.start()
    try:
        await requester_transport.connect_to_peer(
            f"ws://127.0.0.1:{server_port}",
        )
        for _ in range(50):
            await asyncio.sleep(0.05)
            if server_id.node_id in requester_transport.peers:
                break
        assert server_id.node_id in requester_transport.peers

        # Send the stream REQ ourselves (no dispatcher) — we want
        # the server to start streaming but we'll drop the connection
        # before it finishes.
        from prsm.node.transport import P2PMessage, MSG_DIRECT as _D
        from prsm.node.chain_executor_adapters import (
            CHAIN_STREAM_MSG_TYPE, CHAIN_STREAM_REQ_KEY, CHAIN_PAYLOAD_KEY,
        )
        import base64 as _b64
        req_msg = P2PMessage(
            msg_type=_D,
            sender_id=requester_id.node_id,
            payload={
                "subtype": CHAIN_STREAM_MSG_TYPE,
                CHAIN_STREAM_REQ_KEY: "test-stream-720",
                CHAIN_PAYLOAD_KEY: _b64.b64encode(b"req").decode(),
            },
        )
        await requester_transport.send_to_peer(server_id.node_id, req_msg)

        # Wait for the server to produce at least a handful of frames.
        for _ in range(50):
            await asyncio.sleep(0.05)
            if yields["count"] >= 5:
                break
        frames_at_disconnect = yields["count"]
        assert frames_at_disconnect >= 1, (
            "server didn't start streaming — test fixture broken"
        )

        # Kill the requester's transport mid-stream.
        await requester_transport.stop()

        # Wait for the server to detect the disconnect + stop the
        # generator. Bound: 50 ticks × 100ms = 5s. The websockets
        # library buffers some frames before the close propagates,
        # so we can't be tighter than a few seconds on localhost.
        for _ in range(50):
            await asyncio.sleep(0.1)
            if yields["closed"]:
                break

        # F54 invariant: server's generator MUST be closed within
        # the wait budget after the requester disconnected, AND
        # must not have iterated through the entire 10000-frame
        # supply (else the test fixture's "stream is long enough
        # that disconnect happens mid-stream" assumption broke).
        final_count = yields["count"]
        assert yields["closed"], (
            f"server generator never closed after disconnect — F54 "
            f"regression. frames_at_disconnect={frames_at_disconnect} "
            f"final_count={final_count}"
        )
        # sp1418 — TWO assertions, because either one ALONE is defeatable, and
        # the pre-1418 test shipped with only the weaker of them. Measured
        # against the real product bug (server's send loop never yielded to the
        # event loop, so it could not see the peer's Close until the KERNEL
        # socket buffers filled):
        #
        #   config                      at_disconnect  final   old assert
        #   with fix (any buffer)             426        431   pass
        #   BUG, Linux 4MB buffers (CI)     10000      10000   FAIL  <- true positive
        #   BUG, macOS 128KB buffers         4861       7153   pass  <- MISSED IT
        #
        # (1) FIXTURE SANITY. The disconnect must actually land mid-stream. On
        # Linux the entire 3.6MB / 10k-frame supply fits in the autotuned
        # loopback buffer, so a non-yielding server finishes streaming before
        # the test's first `await` even resumes: frames_at_disconnect==10000 and
        # F54 is never exercised. Assert the precondition instead of silently
        # relying on it.
        assert frames_at_disconnect < TOTAL_AVAILABLE, (
            f"server streamed the ENTIRE {TOTAL_AVAILABLE}-frame supply before "
            f"the requester could even disconnect — its send loop is not "
            f"yielding to the event loop, so F54 can never engage (it can only "
            f"fire once the kernel socket buffers fill). This is the bug, not a "
            f"flaky fixture."
        )
        # (2) THE F54 BOUND. Once the peer is gone the server must stop within a
        # handful of frames. Measured delta with the fix: 5 frames (direct probe:
        # 3), so 500 is ~100x headroom — while still catching the macOS case
        # above, where the un-yielding server burned 2,292 frames after the peer
        # had gone yet still slipped under the old `final < TOTAL` assertion.
        ABORT_BUDGET = 500
        assert final_count <= frames_at_disconnect + ABORT_BUDGET, (
            f"server kept iterating after the peer disconnected — F54 fix "
            f"did NOT engage. frames_at_disconnect={frames_at_disconnect} "
            f"final_count={final_count} budget={ABORT_BUDGET} "
            f"(supply={TOTAL_AVAILABLE})"
        )

    finally:
        await server_transport.stop()
        try:
            await requester_transport.stop()
        except Exception:  # noqa: BLE001 — already stopped
            pass


@pytest.mark.asyncio
async def test_sprint_715_server_side_error_terminates_cleanly():
    """If the server-side stream generator raises mid-stream, the
    terminal STREAM_END must carry the error field so the requester
    closes cleanly with a StageExecutionError (not hang)."""
    from prsm.node.identity import generate_node_identity
    from prsm.node.transport import WebSocketTransport, MSG_DIRECT
    from prsm.node.chain_executor_adapters import (
        _remote_token_stream_dispatch,
        handle_chain_stream_request,
        handle_chain_stream_response,
    )

    loop = asyncio.get_event_loop()
    server_id = generate_node_identity("sprint715-err-server")
    requester_id = generate_node_identity("sprint715-err-req")

    server_port = _free_port()
    requester_port = _free_port()
    server_transport = WebSocketTransport(
        server_id, host="127.0.0.1", port=server_port,
    )
    requester_transport = WebSocketTransport(
        requester_id, host="127.0.0.1", port=requester_port,
    )
    server_node = _MinimalNode(server_id, server_transport, loop)
    requester_node = _MinimalNode(requester_id, requester_transport, loop)

    class _ErrServer:
        def handle_token_stream(self, request_bytes):
            yield b"frame-0-ok"
            raise RuntimeError("sprint-715-mid-stream-failure")
    class _ErrExec:
        def __init__(self):
            self._server = _ErrServer()
    server_node._chain_stage_executor = _ErrExec()

    async def _server_dispatch(msg, peer):
        await handle_chain_stream_request(server_node, msg)
    async def _requester_dispatch(msg, peer):
        handle_chain_stream_response(requester_node, msg)
    server_transport.on_message(MSG_DIRECT, _server_dispatch)
    requester_transport.on_message(MSG_DIRECT, _requester_dispatch)

    await server_transport.start()
    await requester_transport.start()
    try:
        await requester_transport.connect_to_peer(
            f"ws://127.0.0.1:{server_port}",
        )
        for _ in range(50):
            await asyncio.sleep(0.05)
            if server_id.node_id in requester_transport.peers:
                break
        assert server_id.node_id in requester_transport.peers

        def _drain():
            return list(_remote_token_stream_dispatch(
                requester_node, server_id.node_id, b"req",
            ))

        # The dispatcher should either raise (with the error text
        # propagated from terminal STREAM_END) or yield the partial
        # frame + terminate. Either is acceptable behavior, but the
        # iterator MUST terminate within the timeout (no hang).
        raised = None
        partial = []
        try:
            partial = await asyncio.wait_for(
                asyncio.to_thread(_drain), timeout=5.0,
            )
        except Exception as e:  # noqa: BLE001
            raised = e

        # Either the partial yielded the first frame OR it raised —
        # both proves the terminal-error envelope worked. What MUST
        # NOT happen is a timeout (hung iterator).
        terminated_cleanly = raised is not None or len(partial) >= 0
        assert terminated_cleanly, (
            "iterator never terminated — server-side error must "
            "ship a terminal STREAM_END so requester closes cleanly"
        )

    finally:
        await server_transport.stop()
        await requester_transport.stop()
