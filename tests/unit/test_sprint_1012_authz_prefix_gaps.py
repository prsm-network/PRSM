"""Sprint 1012 — register the missed sensitive endpoints in the auth gate.

The API-authz hunt (workflow wt1tb4n3q) found several sensitive endpoints that
fall through NodeAuthMiddleware's PROTECTED_PREFIXES even when the operator sets
PRSM_NODE_API_KEY — so they were reachable by any unauthenticated caller on a
public bind (now key-required by sp1011, making these the residual gaps that the
key itself must close):

  - /marketplace/creator-reputation/access (findings 3, 10) — unauthenticated
    reputation mutation (Sybil-ish reputation poisoning; sibling of the sp970
    creator-stake gate).
  - /peers/connect (findings 1, 4) — unauthenticated outbound P2P dial
    (constrained SSRF / port-probe).
  - /billing/{job_id} (finding 7) — leaks other users' escrow/billing records.
  - /content/{cid}/pin (finding 11) — unauthenticated content pin (resource
    consumption). Parameterized path — needs templated matching, which a plain
    startswith prefix cannot express.

Fix: add the three trailing-param prefixes to PROTECTED_PREFIXES, add a
PROTECTED_PATH_PATTERNS list for the embedded-param pin path, and expose an
is_protected_path() helper used by dispatch. The /wallet/* family (KYC, WaaS,
balance — findings 6/8/9) is already covered by the existing /wallet/ prefix +
sp1011's fail-closed; finding 2 (/api/v1/auth/wallet/ reads) needs SIWE-session
gating, not the operator key (documented residual).
"""
from __future__ import annotations

from prsm.api.auth_middleware import is_protected_path


def test_creator_reputation_mutation_now_protected():
    assert is_protected_path("/marketplace/creator-reputation/access") is True


def test_peers_connect_now_protected():
    assert is_protected_path("/peers/connect") is True


def test_billing_record_now_protected():
    assert is_protected_path("/billing/job-abc-123") is True


def test_content_pin_templated_path_now_protected():
    assert is_protected_path("/content/QmAbc123XYZ/pin") is True
    assert is_protected_path("/content/QmAbc123XYZ/pin/") is True


# ── non-regression: public + unrelated paths must NOT become protected ──


def test_public_content_read_not_protected():
    # Content retrieval is intentionally public — must not be gated.
    assert is_protected_path("/content/retrieve/QmAbc123") is False


def test_other_content_subpath_not_protected():
    # Only the /pin action is gated, not arbitrary /content/{cid}/... paths.
    assert is_protected_path("/content/QmAbc/info") is False


def test_existing_protected_prefixes_still_protected():
    assert is_protected_path("/wallet/kyc/user1") is True            # /wallet/
    assert is_protected_path("/marketplace/creator-stake/stake") is True  # sp970
    assert is_protected_path("/admin/slash-history") is True
    assert is_protected_path("/ledger/transfer") is True


def test_unrelated_public_path_not_protected():
    assert is_protected_path("/health") is False
    assert is_protected_path("/marketplace/search") is False


def test_dispatch_uses_the_helper():
    """Structural pin: the middleware's protection decision routes through the
    shared helper (so the prefix + pattern logic is single-sourced)."""
    import inspect

    from prsm.api.auth_middleware import NodeAuthMiddleware

    src = inspect.getsource(NodeAuthMiddleware.dispatch)
    assert "is_protected_path(" in src
