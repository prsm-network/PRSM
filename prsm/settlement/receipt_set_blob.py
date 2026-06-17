"""Sprint 1136 — settlement-receipt data plane BRICK B: canonical receipt-set
BLOB + the on-chain-root CROSS-CHECK primitive.

Brick A (sp1135, ``published_batch_store.py``) retains the ordered receipt set for
a committed batch as a ``PublishedBatch`` (batch_id, merkle_root, commit_timestamp,
ordered ``BatchedReceipt`` list). Brick B adds the two PURE primitives that turn
that retained set into something a producer can SERVE and an observer can SAFELY
consume (design doc 2026-06-16 §3 gate 2, §7 Brick B):

  (1) a canonical, CID-addressable receipt-set BLOB. ``serialize_receipt_set``
      produces DETERMINISTIC bytes (byte-identical for identical input);
      ``receipt_set_cid`` is the content CID of that blob via the EXISTING v1
      BitTorrent-infohash mechanism (``compute_v1_infohash_single_file``), so the
      existing ``ContentProvider`` can address + re-verify the blob on fetch;
      ``deserialize_receipt_set`` round-trips back to a ``PublishedBatch`` with a
      BOUNDED parse (size cap + structural validation) that raises ``ValueError``
      on oversized / malformed bytes rather than silently mis-parsing.

  (2) the ON-CHAIN-ROOT CROSS-CHECK — the observer trust gate.
      ``verify_receipt_set_against_root`` recomputes the merkle root from a
      (fetched, UNTRUSTED) receipt set via the EXACT producer pipeline
      (``client._build_leaves_root``: ``[hash_leaf(batched_receipt_to_leaf(br))
      for br in receipts]`` then ``build_tree_and_proofs(...).root``) and compares
      it to the on-chain ``merkleRoot``. A genuine set matches; a forged / altered
      / reordered set does NOT. ``verify_leaf_hashes_against_root`` is the cheaper
      leaf-only double-spend path (root straight from leaf hashes).

HARD CONSTRAINTS honoured: this is a PURE module — no web3, no network, no file
I/O. Bytes <-> objects only; CID + merkle are pure compute. It REUSES
``merkle.py`` (leaf + root), ``torrent_infohash`` (CID), and Brick A's
``PublishedBatch.to_dict/from_dict`` — none of that logic is reinvented here.

WHY THE ROOTS MATCH EXACTLY: ``build_tree_and_proofs(leaf_hashes).root`` is
``build_merkle_root(leaf_hashes)`` (see merkle.py: ``build_tree_and_proofs`` sets
``root=build_merkle_root(leaf_hashes)``), and the leaf pipeline here is the SAME
list-comprehension the client runs in ``_build_leaves_root``. So a set that was
genuinely committed by the producer recomputes to the on-chain root bit-for-bit.
"""
from __future__ import annotations

import json
from typing import Dict, Optional, Sequence, Tuple

from prsm.core.torrent_infohash import compute_v1_infohash_single_file
from prsm.settlement.inference_receipt_store import (
    InferenceReceiptStore,
    RetainedInferenceReceipt,
)
from prsm.settlement.merkle import (
    batched_receipt_to_leaf,
    build_tree_and_proofs,
    hash_leaf,
)
from prsm.settlement.published_batch_store import PublishedBatch

# The canonical content name embedded in the single-file torrent ``info`` dict
# whose bencode is SHA-1'd to produce the CID. Fixed so the CID is a pure function
# of the blob bytes (a stable name keeps the infohash determined solely by the
# receipt-set content). Producer + observer MUST use this same name to agree.
RECEIPT_SET_CONTENT_NAME = "receipt-set"

# Defensive parse bound (JSON-bomb / oversized-fetch defense). An observer fetches
# this blob from an UNTRUSTED peer, so the deserialize path must cap input before
# it hands bytes to json.loads. A real batch is ~1000 receipts (~few hundred KB);
# 32 MiB is a generous ceiling that still rejects an adversarial multi-GB blob.
MAX_BLOB_BYTES = 32 * 1024 * 1024

# The OPTIONAL §7 (compute-integrity) section key in the canonical blob dict (sp1142,
# §7-Brick-2). An enriched blob carries this key mapping leaf_hash hex ->
# RetainedInferenceReceipt.to_dict(); a plain blob (no §7 receipts) OMITS it entirely so
# the bytes stay byte-identical to serialize_receipt_set (backward-compat).
INFERENCE_RECEIPTS_KEY = "inference_receipts"

# Defensive cap on the §7 section ENTRY COUNT (JSON-bomb / oversized-section defense).
# The §7 section carries at most one receipt per committed leaf, and a real batch is
# ~1000 receipts; 200_000 matches the producer store's per-batch receipt cap — a generous
# ceiling that still rejects an adversarial map before reconstruction.
_MAX_INFERENCE_RECEIPTS = 200_000


def serialize_receipt_set(pb: PublishedBatch) -> bytes:
    """Canonical, DETERMINISTIC bytes for a ``PublishedBatch``.

    Reuses Brick A's ``PublishedBatch.to_dict`` (which hex-encodes all bytes
    fields and reuses the nested ``ShardExecutionReceipt`` serializer) and dumps
    it with ``sort_keys=True`` + compact separators so the output is byte-identical
    for identical input — a hard requirement for a stable content CID.
    """
    return json.dumps(
        pb.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def receipt_set_cid(blob: bytes) -> str:
    """The content CID (40-hex v1 BitTorrent infohash) for a serialized receipt-set
    blob, via the EXISTING ``compute_v1_infohash_single_file`` (not reinvented).

    Deterministic in ``blob`` (fixed content name + piece length), and byte-for-byte
    identical to what the PRSM ``ContentProvider`` computes for the same bytes — so
    an observer that fetches the blob by this CID can content-verify it on arrival.
    """
    if not isinstance(blob, (bytes, bytearray)):
        raise TypeError(f"blob: expected bytes, got {type(blob).__name__}")
    return compute_v1_infohash_single_file(bytes(blob), name=RECEIPT_SET_CONTENT_NAME)


def deserialize_receipt_set(blob: bytes) -> PublishedBatch:
    """Inverse of ``serialize_receipt_set``: bytes -> ``PublishedBatch``.

    BOUNDED parse for untrusted input:
      - rejects non-bytes / oversized blobs (> ``MAX_BLOB_BYTES``) BEFORE decoding;
      - rejects non-UTF-8 / non-JSON / structurally-invalid payloads;
      - delegates the structural reconstruction (and its own receipt-count cap) to
        Brick A's ``PublishedBatch.from_dict``.

    Always raises ``ValueError`` on malformed input (never silently mis-parses,
    never lets a non-ValueError escape from the parse).
    """
    if not isinstance(blob, (bytes, bytearray)):
        raise ValueError(f"blob: expected bytes, got {type(blob).__name__}")
    if len(blob) > MAX_BLOB_BYTES:
        raise ValueError(
            f"receipt-set blob too large: {len(blob)} bytes exceeds cap "
            f"{MAX_BLOB_BYTES}"
        )
    try:
        text = bytes(blob).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"receipt-set blob is not valid UTF-8: {exc}") from exc
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"receipt-set blob is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(
            f"receipt-set blob must decode to an object, got {type(obj).__name__}"
        )
    try:
        return PublishedBatch.from_dict(obj)
    except ValueError:
        raise
    except (KeyError, TypeError, AttributeError) as exc:
        # Normalize any structural shortfall (missing/wrong-typed field) into a
        # clear ValueError — the contract is "malformed bytes raise ValueError".
        raise ValueError(f"receipt-set blob is not a valid PublishedBatch: {exc}") from exc


def _recompute_root(receipts) -> bytes:
    """Recompute the merkle root from receipts via the EXACT producer pipeline.

    Mirrors ``client._build_leaves_root`` byte-for-byte:
        leaf_hashes = [hash_leaf(batched_receipt_to_leaf(br)) for br in receipts]
        root = build_tree_and_proofs(leaf_hashes).root
    """
    leaf_hashes = [hash_leaf(batched_receipt_to_leaf(br)) for br in receipts]
    return build_tree_and_proofs(leaf_hashes).root


def verify_receipt_set_against_root(
    pb: PublishedBatch, expected_root: bytes,
) -> bool:
    """The CROSS-CHECK gate (design §3 gate 2): recompute the merkle root from
    ``pb.receipts`` via the EXACT producer pipeline and return whether it equals
    ``expected_root`` (the on-chain ``merkleRoot``).

    A genuine, chain-anchored set returns True; any forged / altered / reordered
    set returns False. PURE compute — no chain access; the caller supplies the
    trusted on-chain root.
    """
    if not isinstance(expected_root, (bytes, bytearray)):
        raise TypeError(
            f"expected_root: expected bytes, got {type(expected_root).__name__}"
        )
    if not pb.receipts:
        # An empty set has no leaves; ``build_merkle_root`` rejects empty input, so
        # there is nothing that can match a real (non-empty) committed root.
        return False
    return _recompute_root(pb.receipts) == bytes(expected_root)


def verify_leaf_hashes_against_root(
    leaf_hashes: Sequence[bytes], expected_root: bytes,
) -> bool:
    """The cheap double-spend path: recompute the root straight from already-hashed
    leaves (no receipt preimages needed) and compare to ``expected_root``.

    Uses the SAME tree builder the producer uses, so the root matches exactly.
    Returns False on an empty leaf list (no leaves can match a real root).
    """
    if not isinstance(expected_root, (bytes, bytearray)):
        raise TypeError(
            f"expected_root: expected bytes, got {type(expected_root).__name__}"
        )
    if not leaf_hashes:
        return False
    return build_tree_and_proofs(list(leaf_hashes)).root == bytes(expected_root)


# ── §7-Brick-2 (sp1142): PUBLISH §7 InferenceReceipts in the SAME blob ──────
#
# The §7 receipts ride in the SAME canonical blob as the BatchedReceipts (one fetch), as
# an OPTIONAL section keyed by leaf hash. TRUST MODEL (why the UNTRUSTED §7 section is
# safe): the BatchedReceipt set is root-anchored by the UNCHANGED cross-check above; a §7
# receipt is matched to a leaf by hash but NOT trusted by inclusion — its content is
# RE-VERIFIED on use by ``challenge_verifier.verify_inference_receipt_for_challenge``
# (settler signature over the signing_payload + activation-chain checks). A forged §7
# receipt either fails that re-check OR (if the settler signed a fraudulent chain) is
# itself the provable fraud the verifier surfaces. So the §7 section is untrusted-but-safe:
# include it, re-verify on use. It does NOT enter the merkle-root cross-check (the §7
# settler signature binds a DIFFERENT preimage than the on-chain leaf — the sp1141/sp1035
# adapter re-signs the shard payload).


def serialize_enriched_receipt_set(
    pb: PublishedBatch,
    inference_receipts: Optional[Dict[bytes, RetainedInferenceReceipt]] = None,
) -> bytes:
    """Canonical, DETERMINISTIC bytes for a ``PublishedBatch`` PLUS an OPTIONAL §7
    section mapping leaf_hash hex -> ``RetainedInferenceReceipt.to_dict()``.

    BACKWARD-COMPAT (hard constraint): when ``inference_receipts`` is None/empty, the
    blob is BYTE-IDENTICAL to ``serialize_receipt_set(pb)`` — the ``inference_receipts``
    key is NOT emitted at all, so an old observer / old blob is unaffected. When present,
    the §7 section is keyed by ``leaf_hash.hex()`` (so the keys collide with the committed
    batch leaf hashes an observer derives) and dumped with the SAME canonical discipline
    (sort_keys=True + compact separators) so the output stays byte-identical for identical
    input — a hard requirement for a stable content CID.
    """
    obj = pb.to_dict()
    if inference_receipts:
        obj[INFERENCE_RECEIPTS_KEY] = {
            bytes(leaf_hash).hex(): rec.to_dict()
            for leaf_hash, rec in inference_receipts.items()
        }
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def deserialize_enriched_receipt_set(
    blob: bytes,
) -> Tuple[PublishedBatch, Dict[bytes, RetainedInferenceReceipt]]:
    """Inverse of ``serialize_enriched_receipt_set``: bytes -> (``PublishedBatch``,
    {leaf_hash bytes -> ``RetainedInferenceReceipt``}).

    The ``PublishedBatch`` is recovered EXACTLY as ``deserialize_receipt_set`` does (same
    bounded blob parse). The §7 section is OPTIONAL: an empty dict is returned when it is
    absent (so this reader works on a plain Brick-B blob too). Bounded parse for untrusted
    input — the overall blob size cap (``MAX_BLOB_BYTES``) plus a §7 ENTRY-COUNT cap
    (``_MAX_INFERENCE_RECEIPTS``); a malformed / oversized / wrong-typed §7 section raises
    ``ValueError`` (never a silent mis-parse, never lets a non-ValueError escape).
    """
    # Reuse the exact bounded parse + PublishedBatch reconstruction (size cap, UTF-8/JSON
    # validation, structural validation all live there).
    pb = deserialize_receipt_set(blob)

    # Re-decode just the §7 section. ``deserialize_receipt_set`` already proved the blob is
    # bytes <= MAX_BLOB_BYTES and valid JSON object, so this re-parse is bounded + safe.
    obj = json.loads(bytes(blob).decode("utf-8"))
    section = obj.get(INFERENCE_RECEIPTS_KEY)
    if section is None:
        return pb, {}
    if not isinstance(section, dict):
        raise ValueError(
            f"§7 inference_receipts section must be an object, got "
            f"{type(section).__name__}"
        )
    if len(section) > _MAX_INFERENCE_RECEIPTS:
        raise ValueError(
            f"§7 inference_receipts count {len(section)} exceeds cap "
            f"{_MAX_INFERENCE_RECEIPTS}"
        )

    out: Dict[bytes, RetainedInferenceReceipt] = {}
    for key_hex, entry in section.items():
        try:
            leaf_hash = bytes.fromhex(key_hex)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"§7 inference_receipts key is not valid hex: {key_hex!r}"
            ) from exc
        if not isinstance(entry, dict):
            raise ValueError(
                f"§7 inference_receipts entry for {key_hex!r} must be an object, "
                f"got {type(entry).__name__}"
            )
        try:
            rec = RetainedInferenceReceipt.from_dict(entry)
        except ValueError:
            raise
        except (KeyError, TypeError, AttributeError) as exc:
            # Normalize any structural shortfall into the contract's ValueError.
            raise ValueError(
                f"§7 inference_receipts entry for {key_hex!r} is not a valid "
                f"RetainedInferenceReceipt: {exc}"
            ) from exc
        out[leaf_hash] = rec
    return pb, out


def enrich_from_store(
    pb: PublishedBatch, store: InferenceReceiptStore,
) -> Dict[bytes, RetainedInferenceReceipt]:
    """Producer-side helper: for each ``BatchedReceipt`` in ``pb.receipts`` compute its
    committed leaf hash (``hash_leaf(batched_receipt_to_leaf(br))``, the SAME key the §7-1
    store retains under) and collect the present ``RetainedInferenceReceipt`` from
    ``store``. Returns the §7 map (leaf_hash bytes -> record) for the leaves that HAVE a
    retained §7 receipt; leaves with no §7 receipt are simply absent.

    The caller passes this map to ``serialize_enriched_receipt_set`` to publish the §7
    receipts alongside the settlement leaves in one blob.
    """
    out: Dict[bytes, RetainedInferenceReceipt] = {}
    for br in pb.receipts:
        leaf_hash = hash_leaf(batched_receipt_to_leaf(br))
        rec = store.get(leaf_hash)
        if rec is not None:
            out[leaf_hash] = rec
    return out
