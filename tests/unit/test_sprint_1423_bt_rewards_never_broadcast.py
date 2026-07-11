"""Sprint 1423 — BitTorrent seeding rewards were minted but NEVER broadcast to the network.

`BitTorrentProvider.__init__` says, in as many words:

    self.ledger_sync = None  # Set by node.py

node.py never did. Its wiring block hands `ledger_sync` to five subsystems —
content_uploader, compute_provider, compute_requester, storage_provider, agent_collaboration —
and bt_provider is the sole omission, even though it is constructed ~2,000 lines EARLIER in the
same `initialize()` and was assignable the whole time.

So in `_reward_loop`:

    tx = await self.ledger.credit(...)          # mints FTNS locally
    if self.ledger_sync:                        # ALWAYS False
        await self.ledger_sync.broadcast_transaction(tx)   # unreachable

A seeding node self-minted FTNS that no peer ever saw. LedgerSync gossips self-credits precisely
"for transparency", and `broadcast_transaction` also calls `ledger.record_nonce(...)` — both
skipped. It further poisons the sp1419 balance reconciliation, which compares our balance against
peers who never observed the mint. Every other reward path broadcasts; this was an omission, not
a design choice.

The second test is the load-bearing one: it pins the CLASS of bug. Any subsystem that declares a
`ledger_sync` for node.py to fill in must actually be filled in — so the next one cannot be
forgotten silently.
"""
from __future__ import annotations

import ast
import inspect

from prsm.node import node as node_mod


class TestTheRewardIsActuallyBroadcast:
    def test_node_wires_ledger_sync_into_the_bittorrent_provider(self):
        src = inspect.getsource(node_mod)
        assert "self.bt_provider.ledger_sync = self.ledger_sync" in src, (
            "bt_provider never receives ledger_sync, so its _reward_loop's "
            "`if self.ledger_sync:` is always False — seeding rewards are minted locally and "
            "the network never sees them"
        )

    def test_the_reward_loop_still_guards_and_broadcasts(self):
        from prsm.node.bittorrent_provider import BitTorrentProvider

        body = inspect.getsource(BitTorrentProvider._reward_loop)
        assert "self.ledger_sync.broadcast_transaction" in body, (
            "the broadcast was removed — seeding mints would go unannounced again"
        )


class TestEveryDeclaredLedgerSyncConsumerIsWired:
    """The durable fix: nobody who asks for ledger_sync gets silently skipped again."""

    # subsystem module -> the attribute node.py stores it under
    CONSUMERS = {
        "prsm/node/bittorrent_provider.py": "bt_provider",
        "prsm/node/compute_provider.py": "compute_provider",
        "prsm/node/compute_requester.py": "compute_requester",
        "prsm/node/storage_provider.py": "storage_provider",
    }

    def test_every_subsystem_declaring_ledger_sync_is_assigned_it_by_node(self):
        src = inspect.getsource(node_mod)
        missing = [
            attr
            for path, attr in sorted(self.CONSUMERS.items())
            if f"self.{attr}.ledger_sync = self.ledger_sync" not in src
        ]
        assert not missing, (
            f"these subsystems declare `self.ledger_sync = None  # Set by node.py` but node.py "
            f"never sets it, so every transaction they create is invisible to the network: "
            f"{missing}"
        )

    def test_the_consumer_list_has_not_gone_stale(self):
        """If a NEW subsystem starts expecting ledger_sync, force it into the list above.

        Otherwise this test passes while the new subsystem silently repeats sp1423.
        """
        import pathlib

        declared = {
            str(p)
            for p in pathlib.Path("prsm/node").rglob("*.py")
            if "self.ledger_sync = None" in p.read_text(errors="ignore")
            and p.name != "node.py"  # node.py OWNS ledger_sync; it is not a consumer
        }
        assert declared == set(self.CONSUMERS), (
            f"the set of ledger_sync consumers changed — update CONSUMERS above and confirm "
            f"node.py wires the new one.\n  now: {sorted(declared)}\n  known: "
            f"{sorted(self.CONSUMERS)}"
        )
