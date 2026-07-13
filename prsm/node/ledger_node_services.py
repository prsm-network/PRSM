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

    async def gossip_nonce_exists(self, nonce: str) -> bool:
        """sp1442 (P2P audit, Finding B) — O(1) primary-key seek: has this gossip nonce already
        been logged? gossip_log.nonce is a PRIMARY KEY, so this is an index lookup, replacing the
        catch-up dedup's old full-24h-window range scan (which json-parsed every row) — one seek per
        catch-up entry instead of a scan. Lives on the shared mixin so BOTH LocalLedger and
        DAGLedger carry it by construction (the sp1425 conformance discipline)."""
        if not nonce:
            return False
        cursor = await self._db.execute(
            "SELECT 1 FROM gossip_log WHERE nonce = ? LIMIT 1", (nonce,))
        return (await cursor.fetchone()) is not None

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

    # ── Agent allowances (delegated payments) ────────────────────
    #
    # sp1425 — moved here from LocalLedger. These were defined ONLY on LocalLedger, so on the
    # DEFAULT (dag) backend the /agents/{id}/allowance endpoints 500'd (grant/revoke) and
    # AgentCollaboration's persistence silently no-op'd (its hasattr(ledger, "save_task") probe
    # was always False on a raw DAGLedger). Same drift class as sp966/sp1419. Putting the surface
    # in the shared mixin makes the two backends carry it identically BY CONSTRUCTION. Each
    # ledger's own initialize() already creates agent_allowances + collab_* (verified schema-
    # identical). agent_debit STAYS on LocalLedger — it needs _debit_locked/_write_lock and has
    # zero production callers.

    async def grant_agent_allowance(
        self,
        principal_id: str,
        agent_id: str,
        amount: float,
        epoch_hours: float = 24.0,
    ) -> None:
        """Grant an agent a spending allowance from the principal's wallet.

        The allowance resets at each epoch boundary (epoch_hours interval).
        """
        now = time.time()
        await self._db.execute(
            """INSERT INTO agent_allowances
               (agent_id, principal_id, allowance, spent, epoch_hours, epoch_start, created_at, revoked)
               VALUES (?, ?, ?, 0, ?, ?, ?, 0)
               ON CONFLICT(agent_id) DO UPDATE SET
                   allowance = excluded.allowance,
                   spent = 0,
                   epoch_hours = excluded.epoch_hours,
                   epoch_start = excluded.epoch_start,
                   revoked = 0""",
            (agent_id, principal_id, amount, epoch_hours, now, now),
        )
        await self._db.commit()

    async def get_agent_allowance(self, agent_id: str) -> Optional[dict]:
        """Get an agent's current allowance state. Returns None if not found."""
        cursor = await self._db.execute(
            "SELECT principal_id, allowance, spent, epoch_hours, epoch_start, revoked "
            "FROM agent_allowances WHERE agent_id = ?",
            (agent_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        principal_id, allowance, spent, epoch_hours, epoch_start, revoked = row

        # Check if epoch has rolled over — if so, reset spent
        now = time.time()
        epoch_seconds = epoch_hours * 3600
        if now - epoch_start >= epoch_seconds:
            spent = 0.0
            epoch_start = now
            await self._db.execute(
                "UPDATE agent_allowances SET spent = 0, epoch_start = ? WHERE agent_id = ?",
                (now, agent_id),
            )
            await self._db.commit()

        return {
            "agent_id": agent_id,
            "principal_id": principal_id,
            "allowance": allowance,
            "spent": spent,
            "remaining": max(0.0, allowance - spent),
            "epoch_hours": epoch_hours,
            "epoch_start": epoch_start,
            "revoked": bool(revoked),
        }

    async def revoke_agent_allowance(self, principal_id: str, agent_id: str) -> bool:
        """Revoke an agent's spending authority. Returns True if found."""
        cursor = await self._db.execute(
            "UPDATE agent_allowances SET revoked = 1 "
            "WHERE agent_id = ? AND principal_id = ?",
            (agent_id, principal_id),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    # ── Collaboration Persistence ────────────────────────────────

    async def save_task(self, data: Dict[str, Any]) -> None:
        """Upsert a collaboration task record."""
        await self._db.execute(
            """INSERT OR REPLACE INTO collab_tasks
               (task_id, requester_agent_id, requester_node_id, title, description,
                required_capabilities, ftns_budget, deadline_seconds, status,
                assigned_agent_id, bids, result, created_at, escrow_tx_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data["task_id"],
                data["requester_agent_id"],
                data["requester_node_id"],
                data.get("title", ""),
                data.get("description", ""),
                json.dumps(data.get("required_capabilities", [])),
                data.get("ftns_budget", 0),
                data.get("deadline_seconds", 3600),
                data.get("status", "open"),
                data.get("assigned_agent_id"),
                json.dumps(data.get("bids", [])),
                json.dumps(data.get("result")) if data.get("result") else None,
                data["created_at"],
                data.get("escrow_tx_id"),
            ),
        )
        await self._db.commit()

    async def save_review(self, data: Dict[str, Any]) -> None:
        """Upsert a collaboration review record."""
        await self._db.execute(
            """INSERT OR REPLACE INTO collab_reviews
               (review_id, submitter_agent_id, submitter_node_id, content_cid,
                description, required_capabilities, ftns_per_review, max_reviewers,
                status, reviews, created_at, escrow_tx_id, paid_reviewers)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data["review_id"],
                data["submitter_agent_id"],
                data["submitter_node_id"],
                data.get("content_cid", ""),
                data.get("description", ""),
                json.dumps(data.get("required_capabilities", [])),
                data.get("ftns_per_review", 0.1),
                data.get("max_reviewers", 3),
                data.get("status", "pending"),
                json.dumps(data.get("reviews", [])),
                data["created_at"],
                data.get("escrow_tx_id"),
                json.dumps(data.get("paid_reviewers", [])),
            ),
        )
        await self._db.commit()

    async def save_query(self, data: Dict[str, Any]) -> None:
        """Upsert a collaboration query record."""
        await self._db.execute(
            """INSERT OR REPLACE INTO collab_queries
               (query_id, requester_agent_id, requester_node_id, topic, question,
                ftns_per_response, max_responses, responses, created_at,
                escrow_tx_id, paid_responders)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data["query_id"],
                data["requester_agent_id"],
                data["requester_node_id"],
                data.get("topic", ""),
                data.get("question", ""),
                data.get("ftns_per_response", 0.05),
                data.get("max_responses", 5),
                json.dumps(data.get("responses", [])),
                data["created_at"],
                data.get("escrow_tx_id"),
                json.dumps(data.get("paid_responders", [])),
            ),
        )
        await self._db.commit()

    async def load_active_tasks(self) -> List[Dict[str, Any]]:
        """Load non-terminal collaboration tasks."""
        cursor = await self._db.execute(
            """SELECT task_id, requester_agent_id, requester_node_id, title,
                      description, required_capabilities, ftns_budget,
                      deadline_seconds, status, assigned_agent_id, bids,
                      result, created_at, escrow_tx_id
               FROM collab_tasks WHERE status NOT IN ('completed', 'cancelled')"""
        )
        rows = await cursor.fetchall()
        return [
            {
                "task_id": r[0], "requester_agent_id": r[1],
                "requester_node_id": r[2], "title": r[3],
                "description": r[4],
                "required_capabilities": json.loads(r[5]),
                "ftns_budget": r[6], "deadline_seconds": r[7],
                "status": r[8], "assigned_agent_id": r[9],
                "bids": json.loads(r[10]),
                "result": json.loads(r[11]) if r[11] else None,
                "created_at": r[12], "escrow_tx_id": r[13],
            }
            for r in rows
        ]

    async def load_active_reviews(self) -> List[Dict[str, Any]]:
        """Load non-terminal collaboration reviews."""
        cursor = await self._db.execute(
            """SELECT review_id, submitter_agent_id, submitter_node_id,
                      content_cid, description, required_capabilities,
                      ftns_per_review, max_reviewers, status, reviews,
                      created_at, escrow_tx_id, paid_reviewers
               FROM collab_reviews WHERE status NOT IN ('accepted', 'rejected')"""
        )
        rows = await cursor.fetchall()
        return [
            {
                "review_id": r[0], "submitter_agent_id": r[1],
                "submitter_node_id": r[2], "content_cid": r[3],
                "description": r[4],
                "required_capabilities": json.loads(r[5]),
                "ftns_per_review": r[6], "max_reviewers": r[7],
                "status": r[8], "reviews": json.loads(r[9]),
                "created_at": r[10], "escrow_tx_id": r[11],
                "paid_reviewers": json.loads(r[12]),
            }
            for r in rows
        ]

    async def load_active_queries(self) -> List[Dict[str, Any]]:
        """Load active collaboration queries (not yet at max responses)."""
        cursor = await self._db.execute(
            """SELECT query_id, requester_agent_id, requester_node_id,
                      topic, question, ftns_per_response, max_responses,
                      responses, created_at, escrow_tx_id, paid_responders
               FROM collab_queries"""
        )
        rows = await cursor.fetchall()
        results = []
        for r in rows:
            responses = json.loads(r[7])
            max_responses = r[6]
            if len(responses) < max_responses:
                results.append({
                    "query_id": r[0], "requester_agent_id": r[1],
                    "requester_node_id": r[2], "topic": r[3],
                    "question": r[4], "ftns_per_response": r[5],
                    "max_responses": max_responses,
                    "responses": responses,
                    "created_at": r[8], "escrow_tx_id": r[9],
                    "paid_responders": json.loads(r[10]),
                })
        return results

    async def delete_collab_record(self, table: str, id_column: str, record_id: str) -> None:
        """Delete a completed/archived collaboration record from SQLite."""
        allowed = {"collab_tasks": "task_id", "collab_reviews": "review_id", "collab_queries": "query_id"}
        if table not in allowed or allowed[table] != id_column:
            return
        await self._db.execute(f"DELETE FROM {table} WHERE {id_column} = ?", (record_id,))
        await self._db.commit()
