"""Sprint 1257 — chain-aware eth_getLogs window for the inbound-transfer scan.

Found by the live front-door validation: `wallet info` on Base Sepolia showed
"Inbound: ⚠️ read failed: query exceeds max block range 2000". The inbound scan
chunked at the 9000-block default (sized for Base mainnet's ~10k cap), which over-runs
Base Sepolia's public-RPC 2000-block cap. getlogs_window_for_chain picks a window that
fits the active chain's cap; the node /wallet/transactions/onchain/inbound{,/stats}
endpoints now pass it (derived from w3.eth.chain_id).
"""
from __future__ import annotations

import pytest

import prsm.economy.ftns_onchain as fo
from prsm.economy.ftns_onchain import (
    getlogs_window_for_chain,
    scan_inbound_transfers_chunked,
)


def test_window_for_chain():
    assert getlogs_window_for_chain(84532) <= 2000        # Base Sepolia cap
    assert getlogs_window_for_chain(8453) == 9000         # Base mainnet (~10k cap)
    assert getlogs_window_for_chain("84532") <= 2000      # str-int chain id
    assert getlogs_window_for_chain(None) <= 2000         # fail-safe = smallest window
    assert getlogs_window_for_chain("nonsense") <= 2000   # garbage → fail-safe


def test_chunked_scan_respects_sepolia_window(monkeypatch):
    # the Sepolia window must produce only sub-2000-block getLogs requests, contiguous,
    # covering the full range — so the scan never trips the 2000-block cap.
    windows = []

    def _fake_scan(contract, recipient, from_block, to_block):
        windows.append((from_block, to_block))
        return []

    monkeypatch.setattr(fo, "scan_inbound_transfers", _fake_scan)
    scan_inbound_transfers_chunked(
        object(), recipient="0xabc", from_block=0, to_block=5000,
        max_window=getlogs_window_for_chain(84532),
    )

    assert windows, "expected at least one window"
    assert all((end - start + 1) <= 2000 for start, end in windows)   # each ≤ Sepolia cap
    assert windows[0][0] == 0 and windows[-1][1] == 5000              # full coverage
    for (s1, e1), (s2, e2) in zip(windows, windows[1:]):             # contiguous, no gaps
        assert s2 == e1 + 1


def test_mainnet_window_uses_wider_chunks(monkeypatch):
    windows = []
    monkeypatch.setattr(fo, "scan_inbound_transfers",
                        lambda contract, recipient, from_block, to_block:
                        windows.append((from_block, to_block)) or [])
    scan_inbound_transfers_chunked(
        object(), recipient="0xabc", from_block=0, to_block=20000,
        max_window=getlogs_window_for_chain(8453),
    )
    # mainnet uses ~9000-block windows → fewer requests than the Sepolia path would.
    assert max((e - s + 1) for s, e in windows) == 9000
    assert windows[-1][1] == 20000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
