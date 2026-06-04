"""Sprint 1004 — bound unauthenticated advertise-lane credit fields.

The content-data-plane integrity hunt (workflow w8gvkhelm) confirmed the
GOSSIP_CONTENT_ADVERTISE lane is unauthenticated: any peer can create or
overwrite an in-memory ContentRecord's money/credit fields (findings 3, 5,
6). The on-chain royalty leg is backstopped (sp996 keys on the registered
provenance_hash), but the in-memory royalty_rate weights off-chain
multi-shard pool splits and size_bytes feeds the sp1002 gateway-fetch cap.

This sprint ships the bounded, clearly-correct part: reject non-finite /
negative / absurd royalty_rate (falling back to the safe default) and reject
non-integer / negative size_bytes (→ 0 = "unknown"), on BOTH the new-record
and backfill paths. This prevents absolute-insanity values (NaN/inf, negative
rate, a 0.99 rate vs others' 0.01, a negative size) from entering the index.

The remaining within-bounds relative-skew + creator-binding residual needs
the design decision recorded in
docs/2026-06-04-content-data-plane-trust-anchors.md (Gap B).
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from prsm.node.content_index import (
    DEFAULT_ROYALTY_RATE,
    MAX_ADVERTISE_ROYALTY_RATE,
    ContentIndex,
)


def _idx():
    return ContentIndex(gossip=MagicMock())


def _advertise(cid, **fields):
    data = {"cid": cid, "provider_id": "peer-a"}
    data.update(fields)
    return data


@pytest.mark.asyncio
async def test_new_record_rejects_over_max_royalty_rate():
    idx = _idx()
    await idx._on_content_advertise(
        "content.advertise", _advertise("c1", royalty_rate=0.99), "origin"
    )
    assert idx.lookup("c1").royalty_rate == DEFAULT_ROYALTY_RATE


@pytest.mark.asyncio
async def test_new_record_rejects_nan_royalty_rate():
    idx = _idx()
    await idx._on_content_advertise(
        "content.advertise", _advertise("c1", royalty_rate=float("nan")), "origin"
    )
    rate = idx.lookup("c1").royalty_rate
    assert math.isfinite(rate)
    assert rate == DEFAULT_ROYALTY_RATE


@pytest.mark.asyncio
async def test_new_record_rejects_inf_royalty_rate():
    idx = _idx()
    await idx._on_content_advertise(
        "content.advertise", _advertise("c1", royalty_rate=float("inf")), "origin"
    )
    assert idx.lookup("c1").royalty_rate == DEFAULT_ROYALTY_RATE


@pytest.mark.asyncio
async def test_new_record_rejects_negative_royalty_rate():
    idx = _idx()
    await idx._on_content_advertise(
        "content.advertise", _advertise("c1", royalty_rate=-5.0), "origin"
    )
    assert idx.lookup("c1").royalty_rate == DEFAULT_ROYALTY_RATE


@pytest.mark.asyncio
async def test_new_record_accepts_legit_royalty_rate():
    """Non-regression: a legitimate in-range rate is preserved exactly."""
    idx = _idx()
    await idx._on_content_advertise(
        "content.advertise", _advertise("c1", royalty_rate=0.05), "origin"
    )
    assert idx.lookup("c1").royalty_rate == 0.05


@pytest.mark.asyncio
async def test_max_advertise_royalty_rate_is_finite_and_sane():
    assert math.isfinite(MAX_ADVERTISE_ROYALTY_RATE)
    assert 0.0 < MAX_ADVERTISE_ROYALTY_RATE <= 1.0


@pytest.mark.asyncio
async def test_backfill_rejects_absurd_royalty_rate_clobber():
    """An existing default-rate record must not be clobbered to an absurd
    value by a later (unauthenticated) advertise."""
    idx = _idx()
    # First advertise: minimal, leaves royalty_rate at the default.
    await idx._on_content_advertise(
        "content.advertise", _advertise("c1"), "origin"
    )
    assert idx.lookup("c1").royalty_rate == DEFAULT_ROYALTY_RATE
    # Second advertise tries to bump it to an out-of-range value.
    await idx._on_content_advertise(
        "content.advertise", _advertise("c1", royalty_rate=50.0), "attacker"
    )
    assert idx.lookup("c1").royalty_rate == DEFAULT_ROYALTY_RATE


@pytest.mark.asyncio
async def test_backfill_accepts_legit_rate_on_default_record():
    """Non-regression: the legitimate backfill of a non-default in-range
    rate onto a default record still works."""
    idx = _idx()
    await idx._on_content_advertise(
        "content.advertise", _advertise("c1"), "origin"
    )
    await idx._on_content_advertise(
        "content.advertise", _advertise("c1", royalty_rate=0.07), "origin"
    )
    assert idx.lookup("c1").royalty_rate == 0.07


@pytest.mark.asyncio
async def test_new_record_rejects_negative_size_bytes():
    idx = _idx()
    await idx._on_content_advertise(
        "content.advertise", _advertise("c1", size_bytes=-100), "origin"
    )
    assert idx.lookup("c1").size_bytes == 0


@pytest.mark.asyncio
async def test_new_record_rejects_non_numeric_size_bytes():
    idx = _idx()
    await idx._on_content_advertise(
        "content.advertise", _advertise("c1", size_bytes="not-a-number"), "origin"
    )
    assert idx.lookup("c1").size_bytes == 0


@pytest.mark.asyncio
async def test_new_record_accepts_legit_size_bytes():
    """Non-regression: a positive size is preserved."""
    idx = _idx()
    await idx._on_content_advertise(
        "content.advertise", _advertise("c1", size_bytes=4096), "origin"
    )
    assert idx.lookup("c1").size_bytes == 4096
