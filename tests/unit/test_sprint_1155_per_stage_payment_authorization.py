"""Sprint 1155 — PerStagePaymentAuthorization EIP-712 type + canonical payee-set
hash + fail-closed verifier (BRICK 3 of the on-chain per-stage payee arc).

The existing sp1054 ``PaymentAuthorization`` pins a SINGLE provider address — it
authorizes paying ONE node. For per-stage on-chain payment a multi-stage inference
pays N topology nodes (each commits its own share-batch, brick 4), so the requester
must authorize the SET of (payee, share) in ONE signature, with a cumulative
ceiling. This brick is that authorization primitive + its fail-closed verifier.

It MIRRORS sp1054 exactly: same network-locked domain ("PRSM-Payment" / "v1" /
Base 8453), same ``_to_bytes32`` helper, same EIP-712 sign/recover pattern, same
``InvalidSignatureFormat`` for a malformed signature SHAPE. The sp1054 module is
imported + reused (DRY) and left byte-for-byte unchanged.

Payload shape:
  PerStagePaymentAuthorization {
    address requester
    bytes32 payee_set_hash       // canonical, order-independent hash of the SET
    uint256 total_max_spend_wei  // cumulative ceiling across all payees
    bytes32 job_nonce            // anti-replay
    uint256 expiry_unix          // anti-replay / staleness
    bytes32 request_hash         // binds the inference request
  }
"""
from __future__ import annotations

import pytest
from eth_account import Account

# Reuse sp1054's domain + helpers — assert they are the SAME objects (DRY parity).
import prsm.settlement.payment_authorization as sp1054
from prsm.settlement.per_stage_payment_authorization import (
    PerStageAuthorizationVerdict,
    compute_payee_set_hash,
    encode_per_stage_typed_data_hash,
    sign_per_stage_authorization,
    verify_per_stage_authorization,
)
from prsm.settlement.per_stage_payment_authorization import (
    DEFAULT_CHAIN_ID,
    DOMAIN_NAME,
    DOMAIN_VERSION,
    InvalidSignatureFormat,
)

# Deterministic test keys (NEVER used on any live chain).
REQUESTER_KEY = "0x" + "11" * 32
OTHER_KEY = "0x" + "99" * 32
REQUESTER = Account.from_key(REQUESTER_KEY).address
OTHER = Account.from_key(OTHER_KEY).address

PAYEE_A = Account.from_key("0x" + "aa" * 32).address
PAYEE_B = Account.from_key("0x" + "bb" * 32).address
PAYEE_C = Account.from_key("0x" + "cc" * 32).address

SHARE_A = 600_000_000_000_000_000   # 0.6 FTNS
SHARE_B = 400_000_000_000_000_000   # 0.4 FTNS
TOTAL = SHARE_A + SHARE_B           # 1.0 FTNS (cap == sum)


def _payees():
    return [(PAYEE_A, SHARE_A), (PAYEE_B, SHARE_B)]


def _payload(**over):
    p = {
        "requester": REQUESTER,
        "payee_set_hash": "0x" + compute_payee_set_hash(_payees()).hex(),
        "total_max_spend_wei": TOTAL,
        "job_nonce": "0x" + "ab" * 32,
        "expiry_unix": 2_000_000_000,
        "request_hash": "0x" + "cd" * 32,
    }
    p.update(over)
    return p


# --------------------------------------------------------------------------- #
# (e) PARITY — domain + helpers reused from sp1054, chain-locked digest        #
# --------------------------------------------------------------------------- #

def test_domain_constants_reused_from_sp1054():
    """The signed domain MUST be the SAME network-locked domain as sp1054 —
    imported, not redefined."""
    assert DOMAIN_NAME == "PRSM-Payment"
    assert DOMAIN_VERSION == "v1"
    assert DEFAULT_CHAIN_ID == 8453
    assert DOMAIN_NAME is sp1054.DOMAIN_NAME
    assert DOMAIN_VERSION is sp1054.DOMAIN_VERSION
    assert DEFAULT_CHAIN_ID is sp1054.DEFAULT_CHAIN_ID
    assert InvalidSignatureFormat is sp1054.InvalidSignatureFormat


def test_encode_uses_sp1054_to_bytes32_helper():
    """We reuse sp1054._to_bytes32 (no redefinition)."""
    from prsm.settlement import per_stage_payment_authorization as mod
    assert mod._to_bytes32 is sp1054._to_bytes32


def test_digest_stable_and_chain_locked():
    a = encode_per_stage_typed_data_hash(_payload())
    b = encode_per_stage_typed_data_hash(_payload())
    assert a == b
    assert isinstance(a, bytes) and len(a) == 32
    other = encode_per_stage_typed_data_hash(_payload(), chain_id=1)
    assert other != a


def test_primary_type_distinct_from_sp1054_single_payee_digest():
    """Same domain, DIFFERENT primaryType → the per-stage digest must not collide
    with the single-payee sp1054 digest for an analogous payload."""
    single = sp1054.encode_payment_typed_data_hash({
        "requester": REQUESTER,
        "provider": PAYEE_A,
        "max_spend_wei": TOTAL,
        "job_nonce": "0x" + "ab" * 32,
        "expiry_unix": 2_000_000_000,
        "request_hash": "0x" + "cd" * 32,
    })
    multi = encode_per_stage_typed_data_hash(_payload())
    assert single != multi


# --------------------------------------------------------------------------- #
# (a) HAPPY — requester signs the 2-payee set; each payee verifies True        #
# --------------------------------------------------------------------------- #

def test_happy_each_payee_authorized():
    sig = sign_per_stage_authorization(_payload(), REQUESTER_KEY)
    for payee, share in _payees():
        v = verify_per_stage_authorization(
            _payload(), sig, payees=_payees(), payee=payee, share_wei=share,
            now_unix=1_000_000_000,
        )
        assert isinstance(v, PerStageAuthorizationVerdict)
        assert v.authorized is True, v.reason


def test_happy_case_insensitive_payee_match():
    sig = sign_per_stage_authorization(_payload(), REQUESTER_KEY)
    v = verify_per_stage_authorization(
        _payload(), sig, payees=_payees(),
        payee=PAYEE_A.lower(), share_wei=SHARE_A, now_unix=1_000_000_000,
    )
    assert v.authorized is True, v.reason


# --------------------------------------------------------------------------- #
# (b) SET-HASH CANONICAL — order-independent, collision-free, rejects bad sets #
# --------------------------------------------------------------------------- #

def test_set_hash_order_independent():
    h1 = compute_payee_set_hash([(PAYEE_A, SHARE_A), (PAYEE_B, SHARE_B)])
    h2 = compute_payee_set_hash([(PAYEE_B, SHARE_B), (PAYEE_A, SHARE_A)])
    assert h1 == h2
    assert isinstance(h1, bytes) and len(h1) == 32


def test_set_hash_case_insensitive_addresses():
    h1 = compute_payee_set_hash([(PAYEE_A, SHARE_A), (PAYEE_B, SHARE_B)])
    h2 = compute_payee_set_hash([(PAYEE_A.lower(), SHARE_A), (PAYEE_B.upper(), SHARE_B)])
    assert h1 == h2


def test_set_hash_changes_on_changed_payee():
    base = compute_payee_set_hash([(PAYEE_A, SHARE_A), (PAYEE_B, SHARE_B)])
    diff = compute_payee_set_hash([(PAYEE_A, SHARE_A), (PAYEE_C, SHARE_B)])
    assert base != diff


def test_set_hash_changes_on_changed_share():
    base = compute_payee_set_hash([(PAYEE_A, SHARE_A), (PAYEE_B, SHARE_B)])
    diff = compute_payee_set_hash([(PAYEE_A, SHARE_A + 1), (PAYEE_B, SHARE_B)])
    assert base != diff


def test_set_hash_changes_on_added_and_removed_entry():
    base = compute_payee_set_hash([(PAYEE_A, SHARE_A), (PAYEE_B, SHARE_B)])
    added = compute_payee_set_hash([(PAYEE_A, SHARE_A), (PAYEE_B, SHARE_B), (PAYEE_C, 1)])
    removed = compute_payee_set_hash([(PAYEE_A, SHARE_A)])
    assert base != added
    assert base != removed
    assert added != removed


def test_set_hash_rejects_duplicate_payee():
    with pytest.raises(ValueError):
        compute_payee_set_hash([(PAYEE_A, SHARE_A), (PAYEE_A, SHARE_B)])
    # duplicate is case-insensitive
    with pytest.raises(ValueError):
        compute_payee_set_hash([(PAYEE_A, SHARE_A), (PAYEE_A.lower(), SHARE_B)])


def test_set_hash_rejects_empty_set():
    with pytest.raises(ValueError):
        compute_payee_set_hash([])


def test_set_hash_rejects_negative_share():
    with pytest.raises(ValueError):
        compute_payee_set_hash([(PAYEE_A, -1), (PAYEE_B, SHARE_B)])


def test_set_hash_rejects_over_uint256_share():
    with pytest.raises(ValueError):
        compute_payee_set_hash([(PAYEE_A, 2 ** 256), (PAYEE_B, SHARE_B)])


def test_set_hash_rejects_bool_share():
    with pytest.raises((ValueError, TypeError)):
        compute_payee_set_hash([(PAYEE_A, True), (PAYEE_B, SHARE_B)])


def test_set_hash_rejects_malformed_address():
    with pytest.raises(ValueError):
        compute_payee_set_hash([("not-an-address", SHARE_A), (PAYEE_B, SHARE_B)])


# --------------------------------------------------------------------------- #
# (c) NO OVER-AUTHORIZATION — outside-set / wrong-share / cap / swapped set    #
# --------------------------------------------------------------------------- #

def test_payee_not_in_set_rejected():
    sig = sign_per_stage_authorization(_payload(), REQUESTER_KEY)
    v = verify_per_stage_authorization(
        _payload(), sig, payees=_payees(), payee=PAYEE_C, share_wei=SHARE_A,
        now_unix=1_000_000_000,
    )
    assert v.authorized is False
    assert "member" in v.reason.lower() or "not in" in v.reason.lower()


def test_wrong_share_for_real_payee_rejected():
    sig = sign_per_stage_authorization(_payload(), REQUESTER_KEY)
    v = verify_per_stage_authorization(
        _payload(), sig, payees=_payees(), payee=PAYEE_A, share_wei=SHARE_A + 1,
        now_unix=1_000_000_000,
    )
    assert v.authorized is False


def test_share_sum_exceeding_cap_rejected():
    """A payees list whose share-sum EXCEEDS total_max_spend_wei → not authorized.
    The set hash matches the supplied set, but the cumulative cap is violated."""
    big = [(PAYEE_A, SHARE_A), (PAYEE_B, SHARE_B + 1)]
    payload = _payload(payee_set_hash="0x" + compute_payee_set_hash(big).hex())
    sig = sign_per_stage_authorization(payload, REQUESTER_KEY)
    v = verify_per_stage_authorization(
        payload, sig, payees=big, payee=PAYEE_A, share_wei=SHARE_A,
        now_unix=1_000_000_000,
    )
    assert v.authorized is False
    assert "cap" in v.reason.lower() or "exceed" in v.reason.lower() or "spend" in v.reason.lower()


def test_swapped_set_not_matching_hash_rejected():
    """A payees list that does NOT hash to payload.payee_set_hash → not authorized."""
    sig = sign_per_stage_authorization(_payload(), REQUESTER_KEY)
    swapped = [(PAYEE_A, SHARE_A), (PAYEE_C, SHARE_B)]
    v = verify_per_stage_authorization(
        _payload(), sig, payees=swapped, payee=PAYEE_A, share_wei=SHARE_A,
        now_unix=1_000_000_000,
    )
    assert v.authorized is False
    assert "set" in v.reason.lower() or "hash" in v.reason.lower()


# --------------------------------------------------------------------------- #
# (d) FAIL-CLOSED — wrong signer / expired / tamper / malformed sig SHAPE      #
# --------------------------------------------------------------------------- #

def test_wrong_signer_rejected():
    sig = sign_per_stage_authorization(_payload(), OTHER_KEY)
    v = verify_per_stage_authorization(
        _payload(), sig, payees=_payees(), payee=PAYEE_A, share_wei=SHARE_A,
        now_unix=1_000_000_000,
    )
    assert v.authorized is False
    assert "signer" in v.reason.lower() or "requester" in v.reason.lower()


def test_expired_rejected():
    sig = sign_per_stage_authorization(_payload(), REQUESTER_KEY)
    v = verify_per_stage_authorization(
        _payload(), sig, payees=_payees(), payee=PAYEE_A, share_wei=SHARE_A,
        now_unix=2_000_000_000,  # now == expiry → expired (strict <)
    )
    assert v.authorized is False
    assert "expir" in v.reason.lower()


def test_not_yet_expired_just_under_boundary_ok():
    sig = sign_per_stage_authorization(_payload(), REQUESTER_KEY)
    v = verify_per_stage_authorization(
        _payload(), sig, payees=_payees(), payee=PAYEE_A, share_wei=SHARE_A,
        now_unix=1_999_999_999,
    )
    assert v.authorized is True, v.reason


def test_tampered_request_hash_rejected():
    sig = sign_per_stage_authorization(_payload(), REQUESTER_KEY)
    v = verify_per_stage_authorization(
        _payload(request_hash="0x" + "00" * 32), sig,
        payees=_payees(), payee=PAYEE_A, share_wei=SHARE_A, now_unix=1_000_000_000,
    )
    assert v.authorized is False


def test_tampered_job_nonce_rejected():
    sig = sign_per_stage_authorization(_payload(), REQUESTER_KEY)
    v = verify_per_stage_authorization(
        _payload(job_nonce="0x" + "ef" * 32), sig,
        payees=_payees(), payee=PAYEE_A, share_wei=SHARE_A, now_unix=1_000_000_000,
    )
    assert v.authorized is False


def test_empty_request_hash_rejected():
    """request_hash must be present + bound (non-zero)."""
    payload = _payload(request_hash="0x" + "00" * 32)
    sig = sign_per_stage_authorization(payload, REQUESTER_KEY)
    v = verify_per_stage_authorization(
        payload, sig, payees=_payees(), payee=PAYEE_A, share_wei=SHARE_A,
        now_unix=1_000_000_000,
    )
    assert v.authorized is False
    assert "request" in v.reason.lower()


def test_malformed_signature_shape_raises():
    """A malformed-SHAPE signature RAISES (matching sp1054) — it is not a logical
    reject, it is an unparseable input."""
    with pytest.raises(InvalidSignatureFormat):
        verify_per_stage_authorization(
            _payload(), b"\x00" * 10, payees=_payees(), payee=PAYEE_A,
            share_wei=SHARE_A, now_unix=1_000_000_000,
        )
    with pytest.raises(InvalidSignatureFormat):
        verify_per_stage_authorization(
            _payload(), "not-hex-zzz", payees=_payees(), payee=PAYEE_A,
            share_wei=SHARE_A, now_unix=1_000_000_000,
        )


def test_logical_rejects_never_raise():
    """EVERY logical-reject path returns a verdict, never raises (only a bad sig
    SHAPE raises). Exhaustive sweep over the tamper dimensions."""
    sig = sign_per_stage_authorization(_payload(), REQUESTER_KEY)
    cases = [
        dict(payees=_payees(), payee=PAYEE_C, share_wei=SHARE_A),          # outside set
        dict(payees=_payees(), payee=PAYEE_A, share_wei=SHARE_A + 1),      # wrong share
        dict(payees=[(PAYEE_A, SHARE_A), (PAYEE_C, SHARE_B)], payee=PAYEE_A, share_wei=SHARE_A),  # swapped
    ]
    for kw in cases:
        v = verify_per_stage_authorization(_payload(), sig, now_unix=1_000_000_000, **kw)
        assert isinstance(v, PerStageAuthorizationVerdict)
        assert v.authorized is False
        assert isinstance(v.reason, str) and v.reason


def test_brick1_share_alignment():
    """The shares this authorizes align with brick 1's value-conserving equal
    split: sum(shares) == total cap, and each authorized payee verifies."""
    from prsm.settlement.per_stage_settlement_split import compute_per_stage_shares
    node_ids = ["nodeA", "nodeB", "nodeC"]
    shares = compute_per_stage_shares(TOTAL, node_ids)
    payees = [
        (PAYEE_A, shares[0][1]),
        (PAYEE_B, shares[1][1]),
        (PAYEE_C, shares[2][1]),
    ]
    assert sum(s for _, s in payees) == TOTAL
    payload = _payload(payee_set_hash="0x" + compute_payee_set_hash(payees).hex())
    sig = sign_per_stage_authorization(payload, REQUESTER_KEY)
    for payee, share in payees:
        v = verify_per_stage_authorization(
            payload, sig, payees=payees, payee=payee, share_wei=share,
            now_unix=1_000_000_000,
        )
        assert v.authorized is True, v.reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
