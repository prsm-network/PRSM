"""Off-chain static publisher-key anchor (sprint 1213).

A file-backed implementation of the publisher-key-anchor ``lookup`` contract for
dev / bench / airgapped parallax fleets that don't want on-chain anchor registration.
It resolves ``node_id -> base64 pubkey`` from a local JSON file, so the chain-RPC
per-stage HandoffToken verification (``verify_with_anchor``) can resolve each stage
settler's REAL pubkey without a Base RPC round-trip or a registration tx.

Same ``lookup(node_id) -> Optional[str]`` surface as
``PublisherKeyAnchorClient`` (returns the base64 pubkey, or None if absent).

SECURITY: this is NOT a production trust anchor — there is no on-chain binding, so
anyone who can write the file can claim any node_id. Use it only for dev/bench/
airgapped runs (opt-in via PRSM_PUBLISHER_KEY_ANCHOR_KIND=static); production uses
the on-chain PublisherKeyAnchorClient.

File format (either shape accepted):
    {"node_id_hex": "base64pubkey", ...}
    {"nodes": {"node_id_hex": "base64pubkey", ...}}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union


class StaticFileAnchor:
    """node_id -> base64 pubkey, loaded from a JSON file at construction."""

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(
                f"static anchor file {self._path} must contain a JSON object")
        nodes = raw.get("nodes") if isinstance(raw.get("nodes"), dict) else raw
        self._map = {
            str(nid).lower(): str(pk)
            for nid, pk in nodes.items()
            if isinstance(nid, str) and isinstance(pk, str) and pk
        }

    def lookup(self, node_id: str) -> Optional[str]:
        if not isinstance(node_id, str) or not node_id:
            return None
        return self._map.get(node_id.lower())

    def __len__(self) -> int:
        return len(self._map)
