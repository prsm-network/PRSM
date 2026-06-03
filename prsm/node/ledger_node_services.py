"""Sprint 975 — shared node-services ledger surface (gossip-log + provenance).

LocalLedger and DAGLedger both back a node's gossip-log persistence (catch-up /
digest replay) and durable provenance registry. Historically each carried its OWN
copy of these methods — and that DUPLICATION was the root cause of the sp961-966
dormancy: the methods were absent from the default DAG backend (sp966), and even
where present they could drift (a fix landing in one ledger but not the other).

This mixin is the SINGLE source of truth for that surface. Both ledgers inherit
it, so the two backends can never diverge by construction. The mixin operates on
``self._db`` (an aiosqlite connection) and on the ``gossip_log`` /
``provenance_chains`` tables — each ledger's own ``initialize()`` still creates
those tables (incl. the sp961 ``attestation`` + sp965 ``signed_record`` columns);
the mixin only provides the behaviour.

A pin test (test_sprint_975_*) asserts both ledgers resolve these methods to this
mixin, so re-duplication is caught at CI.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional


class LedgerNodeServicesMixin:
    """Gossip-log + provenance persistence shared by LocalLedger and DAGLedger.

    Requires the host class to provide ``self._db`` (aiosqlite connection) and to
    have created the ``gossip_log`` + ``provenance_chains`` tables in its schema.
    """

    # ── Provenance registry ───────────────────────────────────────

    async def upsert_provenance(self, data: Dict[str, Any]) -> None:
        """Persist or update a provenance record received from the network.

        Called whenever a GOSSIP_PROVENANCE_REGISTER message is received so
        that every node builds a durable local registry of known content
        provenance — independent of the ephemeral gossip_log.
        """
        cid = data.get("cid", "")
        if not cid:
            return
        # sp965 — preserve the VERBATIM received record (the exact dict that was
        # signed, plus the signature) so the cross-node provenance RESPONSE path
        # can re-serve a cryptographically verifiable form. The typed columns
        # below are a lossy view (registered_at != the signed created_at, and
        # is_sharded/provenance_hash are dropped); the verbatim blob is the
        # source of truth for re-verification.
        signed_record = (
            json.dumps(data) if data.get("signature") else None
        )
        await self._db.execute(
            """INSERT INTO provenance_chains
               (cid, content_hash, creator_id, creator_pubkey, filename,
                size_bytes, royalty_rate, parent_cids, signature,
                embedding_id, near_duplicate_of, metadata, registered_at,
                signed_record)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(cid) DO UPDATE SET
                   content_hash      = excluded.content_hash,
                   creator_id        = excluded.creator_id,
                   creator_pubkey    = excluded.creator_pubkey,
                   filename          = excluded.filename,
                   size_bytes        = excluded.size_bytes,
                   royalty_rate      = excluded.royalty_rate,
                   parent_cids       = excluded.parent_cids,
                   signature         = excluded.signature,
                   embedding_id      = excluded.embedding_id,
                   near_duplicate_of = excluded.near_duplicate_of,
                   metadata          = excluded.metadata,
                   signed_record     = excluded.signed_record""",
            (
                cid,
                data.get("content_hash", ""),
                data.get("creator_id", ""),
                data.get("creator_public_key", ""),
                data.get("filename", ""),
                data.get("size_bytes", 0),
                data.get("royalty_rate", 0.01),
                json.dumps(data.get("parent_cids", [])),
                data.get("signature", ""),
                data.get("embedding_id"),
                data.get("near_duplicate_of"),
                json.dumps(data.get("metadata", {})),
                time.time(),
                signed_record,
            ),
        )
        await self._db.commit()

    async def get_signed_provenance(self, cid: str) -> Optional[Dict[str, Any]]:
        """sp965 — return the VERBATIM signed provenance record for `cid` (the
        exact dict that was signed + its signature), or None if no verbatim
        record was stored (pre-sp965 rows / unsigned writes). Used by the query
        responder to re-serve a cryptographically verifiable form."""
        cursor = await self._db.execute(
            "SELECT signed_record FROM provenance_chains WHERE cid = ?", (cid,)
        )
        row = await cursor.fetchone()
        if not row or row[0] is None:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return None

    async def get_provenance(self, cid: str) -> Optional[Dict[str, Any]]:
        """Return the provenance record for a CID, or None if unknown."""
        cursor = await self._db.execute(
            """SELECT cid, content_hash, creator_id, creator_pubkey, filename,
                      size_bytes, royalty_rate, parent_cids, signature,
                      embedding_id, near_duplicate_of, metadata, registered_at
               FROM provenance_chains WHERE cid = ?""",
            (cid,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "cid": row[0],
            "content_hash": row[1],
            "creator_id": row[2],
            "creator_public_key": row[3],
            "filename": row[4],
            "size_bytes": row[5],
            "royalty_rate": row[6],
            "parent_cids": json.loads(row[7]),
            "signature": row[8],
            "embedding_id": row[9],
            "near_duplicate_of": row[10],
            "metadata": json.loads(row[11]),
            "registered_at": row[12],
        }

    # ── Gossip Log ────────────────────────────────────────────────

    async def log_gossip(
        self,
        nonce: str,
        subtype: str,
        origin: str,
        payload: Dict[str, Any],
        ttl: int = 5,
        attestation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a gossip message for catch-up replay.

        sp961: `attestation` (the origin's signed origin_time/origin_pubkey/
        origin_sig) is stored so the catch-up consumer can RE-verify authorship
        relayer-independently. None → stored NULL (unsigned / pre-sp961)."""
        await self._db.execute(
            """INSERT OR IGNORE INTO gossip_log
               (nonce, subtype, origin, payload, ttl, received_at, attestation)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                nonce, subtype, origin, json.dumps(payload), ttl, time.time(),
                json.dumps(attestation) if attestation is not None else None,
            ),
        )
        await self._db.commit()

    async def get_recent_gossip(
        self,
        since: float,
        subtypes: Optional[List[str]] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Retrieve gossip messages received after *since* timestamp."""
        if subtypes:
            placeholders = ",".join("?" for _ in subtypes)
            cursor = await self._db.execute(
                f"""SELECT nonce, subtype, origin, payload, ttl, received_at, attestation
                    FROM gossip_log
                    WHERE received_at > ? AND subtype IN ({placeholders})
                    ORDER BY received_at ASC LIMIT ?""",
                (since, *subtypes, limit),
            )
        else:
            cursor = await self._db.execute(
                """SELECT nonce, subtype, origin, payload, ttl, received_at, attestation
                   FROM gossip_log
                   WHERE received_at > ?
                   ORDER BY received_at ASC LIMIT ?""",
                (since, limit),
            )
        rows = await cursor.fetchall()
        return [
            {
                "nonce": r[0],
                "subtype": r[1],
                "origin": r[2],
                "payload": json.loads(r[3]),
                "ttl": r[4],
                "received_at": r[5],
                "attestation": json.loads(r[6]) if r[6] is not None else None,
            }
            for r in rows
        ]

    async def prune_gossip_log(self, max_age: float) -> int:
        """Delete gossip entries older than *max_age* seconds. Returns count deleted."""
        cutoff = time.time() - max_age
        cursor = await self._db.execute(
            "DELETE FROM gossip_log WHERE received_at < ?", (cutoff,)
        )
        await self._db.commit()
        return cursor.rowcount
