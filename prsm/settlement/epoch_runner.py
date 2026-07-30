"""Sprint 1487 — the emission epoch job: plan, publish, persist, manifest.

sp1481 built the planner and sp1486 the durable watermark; nothing ran them. This
is the runnable job that closes the pool -> earner rail's off-chain half.

THE ORDER OF OPERATIONS IS THE SAFETY PROPERTY
----------------------------------------------
    1. load watermark            (refuse if it looks LOST rather than cold — below)
    2. plan epoch                (exactly-once via the watermark's consumed set)
    3. reconcile against chain   (adopt an epoch we published but failed to persist)
    4. publish root on chain     <-- the irreversible step
    5. persist watermark         (only now are those batches "paid")
    6. write claim manifest      (proofs earners need; derivable, so loss is benign)

Steps 4/5 cannot be atomic across a chain and a file, so the window between them is
handled by step 3 rather than wished away: the chain is authoritative about which
epoch ids exist, so a crash after publish is recovered by reading it back.

THE CHECK THAT MATTERS MOST: A LOST WATERMARK LOOKS LIKE A COLD START
--------------------------------------------------------------------
An absent watermark file is indistinguishable, from the file alone, from "this
node has never published". Treating a lost one as cold re-attributes every
historical batch into epoch 1 — over-paying whoever lands in it out of a shared
pot and silently under-paying everyone else.

The chain can tell them apart. If our watermark says "cold start, next epoch is 1"
but the pool already has epoch 1 published, the watermark is LOST, not cold, and
this job REFUSES to run. That is the whole reason the job reads the chain before
planning rather than only before sending.

WHY LATENESS IS NOT LOSS
------------------------
Every refusal here delays payment; none destroys entitlement. An unconsumed batch
stays unconsumed and the next successful epoch pays it. So refusing loudly is
always the cheaper error.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional, Protocol, runtime_checkable

from prsm.settlement.emission_epoch import (
    EmissionEpochPlan,
    FinalizedBatch,
    build_emission_epoch,
    plan_to_reward_epoch,
)
from prsm.settlement.epoch_watermark import EpochWatermarkStore, WatermarkIntegrityError

logger = logging.getLogger(__name__)


class EpochRunAborted(RuntimeError):
    """The run stopped BEFORE publishing. Nothing was paid; nothing was consumed."""


class WatermarkLostError(EpochRunAborted):
    """Our watermark claims a cold start but the chain already has that epoch.

    This is the double-pay precondition. Refuse rather than re-attribute history.
    """


@runtime_checkable
class PoolChain(Protocol):
    """The narrow slice of OperatorRewardPool this job needs.

    A Protocol rather than the concrete web3 client so the ordering and refusal
    logic is testable without a chain — the parts most likely to be wrong are the
    ones hardest to exercise against a real node. runtime_checkable so a test can
    assert the REAL client still satisfies it; isinstance only compares method
    names, so the conformance test checks signatures separately.
    """

    def epoch_exists(self, epoch_id: int) -> bool: ...
    def unreserved_balance_wei(self) -> int: ...
    def publish_epoch(self, epoch_id: int, merkle_root: bytes, total_amount_wei: int) -> str: ...


@dataclass
class EpochRunResult:
    epoch_id: int
    published: bool
    dry_run: bool
    tx_hash: Optional[str] = None
    merkle_root: Optional[str] = None
    total_amount_wei: int = 0
    recipients: int = 0
    consumed_batches: int = 0
    recovered_from_chain: bool = False
    manifest_path: Optional[str] = None
    notes: List[str] = field(default_factory=list)


def run_epoch(
    *,
    chain: PoolChain,
    watermark: EpochWatermarkStore,
    batches: Iterable[FinalizedBatch],
    pot_wei: Optional[int] = None,
    manifest_dir: Optional[str | os.PathLike] = None,
    dry_run: bool = True,
    window_start: Optional[int] = None,
    window_end: Optional[int] = None,
) -> EpochRunResult:
    """Plan and (unless ``dry_run``) publish one emission epoch.

    :param pot_wei: how much to distribute. Defaults to the pool's UNRESERVED
        balance — reserved funds already back earlier epochs' unclaimed leaves, so
        distributing them would publish an epoch the contract's solvency gate
        rejects, or worse, one it accepts while earlier claims silently become
        unpayable.
    :param dry_run: default TRUE. Publishing is irreversible and an epoch id can
        never be rewritten, so the safe default is to show the plan.
    """
    batches = list(batches)

    # ── 1. watermark ────────────────────────────────────────────────────
    try:
        wm_consumed = watermark.consumed
    except WatermarkIntegrityError:
        watermark.load()
        wm_consumed = watermark.consumed
    epoch_id = watermark.next_epoch_id()
    notes: List[str] = []

    # ── 2. is this a cold start, or a LOST watermark? ───────────────────
    # Asked BEFORE planning: if the watermark is lost, the plan itself would be the
    # dangerous artifact (every historical batch, attributed as if fresh).
    if watermark.last_epoch_id is None and chain.epoch_exists(epoch_id):
        raise WatermarkLostError(
            f"watermark {watermark.path} says this is a cold start (next epoch "
            f"{epoch_id}) but the pool ALREADY has epoch {epoch_id} published. The "
            "watermark is lost, not cold — running now would re-attribute every "
            "historical batch and double-pay out of the shared pot. Restore the "
            "watermark from backup, or rebuild it from the published epochs' "
            "manifests, then re-run."
        )

    # ── 3. crash recovery: published but not persisted ──────────────────
    recovered = False
    if watermark.last_epoch_id is not None and chain.epoch_exists(epoch_id):
        # We planned and published this id on a previous run, then died before
        # persisting. The chain is authoritative; adopt it and move to the next id.
        prior = _load_manifest_batches(manifest_dir, epoch_id)
        if prior is None:
            raise EpochRunAborted(
                f"epoch {epoch_id} is already published on chain but our watermark "
                f"has not advanced, and no manifest for it was found in "
                f"{manifest_dir} to tell us which batches it consumed. Refusing to "
                "plan a new epoch: without that list those batches would be paid a "
                "second time. Recover the manifest for epoch "
                f"{epoch_id} (it is reproducible from the same inputs) and re-run."
            )
        recovered = watermark.reconcile_from_chain(
            epoch_id, prior, published_on_chain=True)
        notes.append(
            f"recovered epoch {epoch_id} from chain ({len(prior)} batches adopted)")
        logger.warning(
            "epoch_runner: adopted already-published epoch %s from chain", epoch_id)
        epoch_id = watermark.next_epoch_id()
        wm_consumed = watermark.consumed
        if chain.epoch_exists(epoch_id):
            raise EpochRunAborted(
                f"epoch {epoch_id} is ALSO already published — more than one epoch "
                "ran without persisting. Reconcile manually before continuing.")

    # ── 4. pot ──────────────────────────────────────────────────────────
    available = chain.unreserved_balance_wei()
    if pot_wei is None:
        pot_wei = available
    elif pot_wei > available:
        raise EpochRunAborted(
            f"requested pot {pot_wei} wei exceeds the pool's UNRESERVED balance "
            f"{available} wei. The excess is reserved against earlier epochs' "
            "unclaimed leaves; distributing it would strand those claims."
        )
    if pot_wei <= 0:
        raise EpochRunAborted(
            "pool has no unreserved balance to distribute — fund the pool (or wait "
            "for emissions) before publishing an epoch. Nothing was consumed."
        )

    # ── 5. plan ─────────────────────────────────────────────────────────
    plan = build_emission_epoch(
        epoch_id=epoch_id, batches=batches, pot_wei=pot_wei,
        consumed_batch_ids=wm_consumed,
        window_start=window_start, window_end=window_end,
    )
    reward = plan_to_reward_epoch(plan)
    assert plan.total_allocated_wei == pot_wei, "apportionment must spend the pot exactly"

    result = EpochRunResult(
        epoch_id=plan.epoch_id, published=False, dry_run=dry_run,
        merkle_root=reward.root_hex,
        total_amount_wei=plan.total_allocated_wei,
        recipients=len(plan.entries),
        consumed_batches=len(plan.consumed_batch_ids),
        recovered_from_chain=recovered, notes=notes,
    )
    if dry_run:
        result.notes.append("DRY RUN — nothing published, watermark unchanged")
        return result

    # ── 6. publish (irreversible), THEN persist ─────────────────────────
    # The manifest is written BEFORE publishing on purpose: it is the record of
    # which batches this epoch id consumed, and step 3 above needs it to recover if
    # we die right after the broadcast. Writing it first can only produce a
    # manifest for an epoch that never published — harmless, since the recovery
    # path only consults it when the CHAIN confirms the epoch exists.
    manifest_path = _write_manifest(manifest_dir, plan, reward)
    result.manifest_path = manifest_path

    tx_hash = chain.publish_epoch(
        plan.epoch_id, reward.merkle_root, plan.total_allocated_wei)
    result.tx_hash = tx_hash
    result.published = True

    watermark.commit_epoch(plan.epoch_id, plan.consumed_batch_ids)
    logger.info(
        "epoch_runner: published epoch %s root=%s recipients=%s tx=%s",
        plan.epoch_id, result.merkle_root, result.recipients, tx_hash)
    return result


# ── manifest io ─────────────────────────────────────────────────────────

def _manifest_file(manifest_dir: Optional[str | os.PathLike], epoch_id: int) -> Optional[Path]:
    if manifest_dir is None:
        return None
    return Path(manifest_dir) / f"epoch-{epoch_id}.json"


def _load_manifest_batches(
    manifest_dir: Optional[str | os.PathLike], epoch_id: int
) -> Optional[List[str]]:
    """Return the batch ids a previously-written manifest says this epoch consumed."""
    p = _manifest_file(manifest_dir, epoch_id)
    if p is None or not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    ids = data.get("consumed_batch_ids")
    return [str(b) for b in ids] if isinstance(ids, list) else None


def _write_manifest(
    manifest_dir: Optional[str | os.PathLike],
    plan: EmissionEpochPlan,
    reward: Any,
) -> Optional[str]:
    """Write the claim manifest: root, per-account amounts and Merkle proofs.

    Earners fetch this to build a claim. It carries no authority — the client
    verifies every proof against the ON-CHAIN root (sp1481), so a tampered
    manifest produces a failed verification, not a bad payout.
    """
    p = _manifest_file(manifest_dir, plan.epoch_id)
    if p is None:
        return None
    payload = {
        "epoch_id": plan.epoch_id,
        "merkle_root": reward.root_hex,
        "total_amount_wei": str(plan.total_allocated_wei),
        "recipients": len(plan.entries),
        "consumed_batch_ids": list(plan.consumed_batch_ids),
        "entries": [
            {
                "account": e.account,
                "amount_wei": str(e.amount_wei),
                "proof": reward.proof_hex(e.account),
            }
            for e in plan.entries
        ],
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise
    return str(p)
