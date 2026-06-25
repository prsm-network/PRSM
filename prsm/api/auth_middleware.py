"""
API Authentication Middleware
=============================

Lightweight API key authentication for node management endpoints.
Protects settler, content economy, and other sensitive routes.
"""

import hashlib
import hmac
import logging
import os
import re
import secrets
from typing import Optional, Set

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Endpoints that don't require authentication
PUBLIC_ENDPOINTS: Set[str] = {
    "/", "/health", "/status", "/docs", "/openapi.json", "/redoc",
    "/peers", "/node/info",
}

# Endpoints that require auth (when PRSM_NODE_API_KEY is set).
# Sprint 892 — paths that authenticate via their OWN mechanism
# (HMAC signature) and MUST be reachable by EXTERNAL callers that
# do not hold the operator's PRSM_NODE_API_KEY. The KYC vendor
# webhook is called by Persona/Onfido/Plaid and verified by its
# per-request signature (sp283 + sp888 fail-closed). It lives under
# the `/wallet/` protected prefix, so without this carve-out the
# node-API-key middleware would 401 the vendor BEFORE its signature
# check runs — breaking the entire KYC→VERIFIED→auto-provision loop
# in production (where the API key is set). Exempting it here does
# NOT weaken security: the endpoint still enforces signature auth
# (no secret → 503, bad signature → 401). Checked BEFORE the
# protected-prefix enforcement.
SIGNATURE_AUTHENTICATED_PREFIXES = [
    "/wallet/kyc/webhook/",
]

PROTECTED_PREFIXES = [
    "/settler/",
    "/content/upload",
    "/compute/forge",
    # Sprint 138 (dogfood). /admin/* surfaces include both
    # READ (history endpoints) and WRITE (heartbeat trigger,
    # distribution trigger) actions. The trigger endpoints
    # submit on-chain transactions — leaving them unprotected
    # when an operator HAS configured PRSM_NODE_API_KEY meant
    # any network-adjacent attacker could spend the operator's
    # gas. Now matches the operator's auth-required intent.
    "/admin/",
    # /wallet/* includes royalty claim (POST /wallet/royalty/
    # claim) which transfers FTNS on-chain. Same threat model
    # as /admin/*/trigger. Read-side wallet endpoints (balance,
    # spend) are also sensitive (financial PII).
    "/wallet/",
    # /content/arbitration/preview-resolution writes governance
    # decisions. Even though the disputed-band resolution is
    # gated on council ratification (council-commitment-only
    # rule), the preview endpoint touches arbitration state and
    # should respect operator-set auth.
    "/content/arbitration/",
    # Sprint 139 (continuation of dogfood audit). The broader
    # set of sensitive endpoints. Each cluster either spends
    # FTNS, modifies node-controlled state, or grants writes
    # that other operators can exploit. Prefix-protecting them
    # matches the operator's PRSM_NODE_API_KEY-set production
    # intent. Dev mode (no key) is unaffected — auth_enabled
    # gates the whole check at runtime.
    "/bridge/",         # cross-chain deposit / withdraw
    "/ftns/",           # faucet, transfers
    "/ledger/",         # /ledger/transfer (direct FTNS xfer)
    "/staking/",        # stake / unstake / claim / withdraw
    "/agents/",         # allowance / pause / resume actions
    "/node/resources",  # PUT modifies node config
    "/settlement/",     # flush
    "/compute/",        # forge already protected; broaden to
                        # cover submit / cancel / inference /
                        # cleanup-stale (all write-side)
    # Sprint 183 — these were unprotected pre-fix. /transactions
    # leaks the operator's full transaction history (financial
    # PII); /content/mine leaks the operator's uploaded-content
    # list (workflow PII + provenance fingerprinting risk).
    "/transactions",
    "/content/mine",
    # Sprint 970 — the creator-stake MUTATION endpoints
    # (POST /marketplace/creator-stake/stake | /slash) write the
    # balance that gates HIGH creator-tier eligibility (search
    # ranking + tier label). Unprotected, any reachable caller
    # could grant a creator free HIGH tier or grief a competitor's
    # tier. Joins /staking/ + /wallet/ in the protected set. (The
    # deeper gap — the gate is in-memory-only with no on-chain
    # CreatorStakeRegistry teeth — is a separate creator-stake
    # commissioning item; this closes the auth vector now.)
    "/marketplace/creator-stake/",
    # Sprint 1012 — API-authz hunt (wt1tb4n3q) gaps. These sensitive endpoints
    # fell through the protected set even with a key set:
    #  - creator-reputation MUTATION (record_access) writes the §14 reputation
    #    that gates creator tier — sibling of the sp970 creator-stake gate
    #    (findings 3/10).
    #  - /peers/connect forces an attacker-chosen outbound P2P dial (constrained
    #    SSRF / port-probe) — an operator action (findings 1/4).
    #  - /billing/{job_id} leaks other users' escrow/billing records; the
    #    trailing job_id keeps the /billing/ prefix valid (finding 7).
    "/marketplace/creator-reputation/",
    "/peers/connect",
    "/billing/",
    # Sprint 1103 — Domain-07 review HIGH. The /onboarding/* wizard WRITES the node's
    # signing identity (POST /onboarding/identity import → config/node_identity.json)
    # and config (POST /onboarding/launch → config/node_config.json). Mounted on the
    # daemon app, it was in NEITHER this set nor the loopback gate → an unauthenticated
    # attacker on a publicly-bound, KEYED node could overwrite the identity and take the
    # node over on the next restart. Now gated by the operator's API key (dev/no-key
    # mode is unaffected — auth_enabled gates the whole check, and a no-key public bind
    # is already refused at startup).
    "/onboarding/",
]

# Sprint 1012 — protected paths with an EMBEDDED parameter that a startswith
# prefix cannot express (a bare "/content/" prefix would wrongly gate public
# content reads). POST /content/{cid}/pin is an operator pin action that should
# respect the API key (finding 11).
PROTECTED_PATH_PATTERNS = [
    re.compile(r"^/content/[^/]+/pin/?$"),
]


def is_protected_path(path: str) -> bool:
    """sp1012 — single-sourced protection decision for NodeAuthMiddleware: True
    when ``path`` matches a PROTECTED_PREFIXES prefix OR a PROTECTED_PATH_PATTERNS
    template. Public-endpoint + signature-authenticated carve-outs are handled
    separately (earlier) in dispatch."""
    if any(path.startswith(prefix) for prefix in PROTECTED_PREFIXES):
        return True
    return any(pat.match(path) for pat in PROTECTED_PATH_PATTERNS)


def generate_api_key() -> str:
    """Generate a new API key."""
    return f"prsm_{secrets.token_urlsafe(32)}"


def hash_api_key(key: str) -> str:
    """Hash an API key for storage."""
    return hashlib.sha256(key.encode()).hexdigest()


class NodeAuthMiddleware(BaseHTTPMiddleware):
    """API key authentication for the node management API.

    Checks for 'Authorization: Bearer <api_key>' or 'X-API-Key: <key>' header.
    If PRSM_NODE_API_KEY env var is set, that key is required for protected endpoints.
    If not set, all endpoints are open (development mode).
    """

    def __init__(self, app, api_key_hash: str = ""):
        super().__init__(app)
        self._api_key_hash = api_key_hash

    @property
    def auth_enabled(self) -> bool:
        return bool(self._api_key_hash)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Public endpoints always allowed
        if path in PUBLIC_ENDPOINTS:
            return await call_next(request)

        # Sp892 — signature-authenticated endpoints (vendor webhooks)
        # are exempt from node-API-key auth: external callers can't
        # hold the operator's key, and the endpoint enforces its own
        # HMAC signature auth. Checked BEFORE protected-prefix
        # enforcement so the /wallet/ prefix doesn't block them.
        if any(
            path.startswith(prefix)
            for prefix in SIGNATURE_AUTHENTICATED_PREFIXES
        ):
            return await call_next(request)

        # Check if this is a protected endpoint (sp1012 — prefix OR templated).
        is_protected = is_protected_path(path)

        if is_protected and self.auth_enabled:
            # Extract API key from headers.
            #
            # Sprint 183 — return JSONResponse directly instead of
            # raising HTTPException. Starlette's BaseHTTPMiddleware
            # does NOT catch HTTPException in dispatch(); a raise
            # here propagates up to the ASGI error handler and
            # surfaces as 500 to the client. Pre-fix every
            # auth-failure (no key + wrong key) returned 500, which
            # monitoring tools mis-classify as outage and clients
            # mis-interpret as server-broken-please-retry instead
            # of access-denied.
            api_key = self._extract_key(request)
            if not api_key:
                return JSONResponse(
                    status_code=401,
                    content={
                        "detail": (
                            "Authentication required. Provide "
                            "'Authorization: Bearer <key>' or "
                            "'X-API-Key: <key>' header."
                        ),
                    },
                )

            # sp1271 — constant-time compare of the key hashes (audit round 5, LOW): avoids a
            # timing side-channel on the digest comparison. compare_digest needs equal-length
            # inputs; both are fixed-length hex SHA256 digests.
            if not hmac.compare_digest(hash_api_key(api_key), self._api_key_hash):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Invalid API key."},
                )

        return await call_next(request)

    @staticmethod
    def _extract_key(request: Request) -> Optional[str]:
        """Extract API key from Authorization or X-API-Key header."""
        # Try Bearer token
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:].strip()

        # Try X-API-Key
        return request.headers.get("x-api-key", "").strip() or None


def get_node_auth_middleware(app) -> Optional[NodeAuthMiddleware]:
    """Create auth middleware if PRSM_NODE_API_KEY is configured."""
    env_key = os.environ.get("PRSM_NODE_API_KEY", "")
    if env_key:
        key_hash = hash_api_key(env_key)
        middleware = NodeAuthMiddleware(app, api_key_hash=key_hash)
        logger.info("Node API authentication enabled (PRSM_NODE_API_KEY set)")
        return middleware
    else:
        logger.info("Node API authentication disabled (set PRSM_NODE_API_KEY to enable)")
        return None
