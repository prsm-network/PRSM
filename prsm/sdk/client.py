"""
PRSM Node Client
=================

Async Python client for PRSM node Ring 1-10 APIs.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# sp1421 — distinguishes "caller said nothing" (use the default store path) from an explicit
# `issued_auth_store_path=None` (deliberately disable retention). Cannot use None for both.
_UNSET = object()


def _fetch_paid_key_sync(base_url: str, content_hash: bytes, requester_key: str):
    """Sprint 1359 (F1 redesign) — SYNC fetch of the wrapped key from the payment-gated serve
    endpoint (run inside pay_and_unlock's worker thread AFTER the fee is settled). Signs the
    paid-key challenge with the payer's ETH key so the server can gate on verifyPayment. Returns
    the wrapped-key bytes, or None if the server refuses (fetch_and_verify_wrapped_key then raises
    KeyNotReleasedError)."""
    import base64

    import httpx
    from eth_account import Account
    from eth_account.messages import encode_defunct

    from prsm.node.paid_key_serve import paid_key_challenge

    ch = bytes(content_hash)
    nonce = ch.hex()[:16]
    sig = Account.from_key(requester_key).sign_message(
        encode_defunct(text=paid_key_challenge(ch, nonce))).signature.hex()
    try:
        with httpx.Client(timeout=30.0) as c:
            resp = c.get(f"{base_url}/content/paid-key/0x{ch.hex()}",
                         params={"nonce": nonce, "signature": sig})
    except Exception:  # noqa: BLE001 — endpoint down / network → treat as no key (fail-loud upstream)
        return None
    if resp.status_code != 200:
        return None
    return base64.b64decode(resp.json()["wrapped_key_b64"])


class PRSMClient:
    """Client for interacting with a PRSM node's API.

    Usage:
        client = PRSMClient("http://localhost:8000")

        # Full forge pipeline
        result = await client.query("EV adoption trends in NC", budget=10.0)

        # Get cost quote first
        quote = await client.quote("EV trends", shards=5, tier="t2")

        # Check node hardware
        status = await client.status()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "",
        issued_auth_store_path: Any = _UNSET,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._session = None
        # sp1421 — where THIS requester retains the PaymentAuthorizations it has issued.
        # Pass an explicit path to relocate it, or None to disable retention entirely
        # (see _issued_auth_store for why disabling is dangerous).
        self._issued_auth_store_path = issued_auth_store_path
        self._issued_auth_store_cached: Any = None

    def _issued_auth_store(self):
        """The requester's ledger of PaymentAuthorizations IT has issued. ON by default.

        sp1421 — this is not bookkeeping, it is the requester's ONLY evidence in a live
        escrow-drain path. On-chain, `BatchSettlementRegistry.commitBatch(address requester, …)`
        takes the requester as a PLAIN, UNVERIFIED argument — no signature, no authorization,
        and no bond — and `finalizeBatch` then calls
        `EscrowPool.settleFromRequester(requester, provider, value)`, which checks only that the
        caller is the registry and that the victim HAS the balance. So ANY address can commit a
        batch naming ANY requester and, after the challenge window, drain their escrow for the
        cost of gas.

        The requester's sole defense is a NO_ESCROW challenge — and `_handleNoEscrow` reverts
        unless `msg.sender == batch.requester`, so nobody else can defend them. Raising it means
        proving "I never authorized this batch", which requires knowing what you DID authorize.
        That is this store.

        It was built (sp1172 et al) and never wired: every SDK payment called the plain
        `build_payment_authorization`, so the store stayed EMPTY and the matcher
        (`match_unauthorized_batches`) had nothing to match against. The defense could not fire.

        Recording is best-effort and can never break a payment (the wrapper swallows and logs).
        Disable with PRSM_ISSUED_AUTH_STORE=0 or `issued_auth_store_path=None` ONLY for a
        stateless client that will never fund escrow — a client that pays from escrow with
        retention off is undefendable.
        """
        if self._issued_auth_store_cached is not None:
            return self._issued_auth_store_cached

        path = self._issued_auth_store_path
        if path is None:
            return None
        if os.environ.get("PRSM_ISSUED_AUTH_STORE", "1").strip().lower() in ("0", "false", "no"):
            return None
        if path is _UNSET:
            path = Path.home() / ".prsm" / "issued_authorizations.json"

        try:
            from prsm.settlement.issued_authorization_store import IssuedAuthorizationStore

            self._issued_auth_store_cached = IssuedAuthorizationStore(path)
        except Exception as exc:  # noqa: BLE001 — retention must NEVER break a payment
            logger.warning(
                "sp1421: could not open the issued-authorization store at %s (%s). Payments "
                "still work, but this requester cannot prove an unauthorized batch (NO_ESCROW) "
                "if one is committed against its escrow.", path, exc,
            )
            return None
        return self._issued_auth_store_cached

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _ensure_session(self):
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def _get(self, path: str) -> Dict[str, Any]:
        await self._ensure_session()
        async with self._session.get(
            f"{self.base_url}{path}",
            headers=self._headers(),
            timeout=__import__("aiohttp").ClientTimeout(total=30),
        ) as resp:
            return await resp.json()

    async def _post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        await self._ensure_session()
        async with self._session.post(
            f"{self.base_url}{path}",
            json=data,
            headers=self._headers(),
            timeout=__import__("aiohttp").ClientTimeout(total=120),
        ) as resp:
            return await resp.json()

    # ── Core Endpoints ────────────────────────────────────────────

    async def status(self) -> Dict[str, Any]:
        """Get node status."""
        return await self._get("/status")

    async def peers(self) -> Dict[str, Any]:
        """Get connected peers."""
        return await self._get("/peers")

    # ── Ring 5: Forge Pipeline ────────────────────────────────────

    async def query(
        self,
        query: str,
        budget: float = 10.0,
        privacy: str = "standard",
        shard_cids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Submit a query through the full Ring 1-10 forge pipeline.

        Args:
            query: Natural language query.
            budget: Maximum FTNS to spend.
            privacy: Privacy level (none, standard, high, maximum).
            shard_cids: Optional specific shard CIDs to target.

        Returns:
            Dict with route, response, result, traces_collected.
        """
        payload = {
            "query": query,
            "budget_ftns": budget,
            "privacy_level": privacy,
        }
        if shard_cids:
            payload["shard_cids"] = shard_cids
        return await self._post("/compute/forge", payload)

    async def prompt(self, prompt: str, budget: float = 0.0) -> Dict[str, Any]:
        """Submit a prompt via legacy NWTN path."""
        return await self._post("/compute/query", {"prompt": prompt, "budget": budget})

    # ── Sprint 819 — Verifiable Inference ─────────────────────────

    async def infer(
        self,
        prompt: str,
        *,
        model_id: str = "gpt2",
        max_tokens: int = 8,
        budget_ftns: float = 1.0,
        privacy_tier: str = "none",
        content_tier: str = "A",
        verify_pubkey_b64: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sprint 819 — POST /compute/inference for verifiable
        inference + signed receipt.

        Returns the parsed server payload:
          {success, output, ftns_charged, receipt, ...}

        When ``verify_pubkey_b64`` is set, the returned dict
        gains a ``receipt_verified`` boolean computed via
        sprint-706 verify_receipt against the supplied pubkey.
        Useful when a caller pins an operator's published pubkey
        and wants the verify check inline with the inference
        request (avoids a separate verify-receipt round-trip).

        Defaults match the `prsm compute infer` CLI (sprint 802)
        so users moving between CLI + SDK see consistent
        behavior.
        """
        body = {
            "prompt": prompt,
            "model_id": model_id,
            "budget_ftns": budget_ftns,
            "privacy_tier": privacy_tier,
            "content_tier": content_tier,
            "max_tokens": max_tokens,
        }
        result = await self._post("/compute/inference", body)
        if verify_pubkey_b64:
            try:
                from prsm.compute.inference.models import (
                    InferenceReceipt,
                )
                from prsm.compute.inference.receipt import (
                    verify_receipt,
                )
                receipt = InferenceReceipt.from_dict(
                    result.get("receipt") or {},
                )
                result["receipt_verified"] = bool(
                    verify_receipt(
                        receipt, public_key_b64=verify_pubkey_b64,
                    ),
                )
            except Exception:
                result["receipt_verified"] = False
        return result

    # ── Sprint 1189 — pay-for-inference (requester-payment UX) ────

    async def deposit_escrow(
        self,
        *,
        requester_key: str,
        amount_ftns,
        network: Optional[str] = None,
        rpc_url: Optional[str] = None,
        escrow_pool_address: Optional[str] = None,
        ftns_token_address: Optional[str] = None,
        expected_chain_id: Optional[int] = None,
        _client: Any = None,
    ) -> str:
        """Sprint 1189 — deposit ``amount_ftns`` of FTNS into the on-chain EscrowPool
        under your own (requester) address, so settled inference can draw from it. Returns
        the deposit tx hash. Self-custodied — withdraw any unspent balance later.

        Resolves the EscrowPool / FTNS token / RPC from the active network (PRSM_NETWORK,
        or ``network``) unless overridden. Requires web3 + the requester private key (this
        signs + broadcasts a real on-chain tx). ``_client`` is injectable for tests."""
        from decimal import Decimal
        amount_wei = int(Decimal(str(amount_ftns)) * (Decimal(10) ** 18))
        client = _client
        if client is None:
            from prsm.config.networks import resolve_endpoints
            from prsm.economy.web3.escrow_pool_client import EscrowPoolClient
            ep = resolve_endpoints(network)
            client = EscrowPoolClient(
                rpc_url or ep.rpc_url,
                escrow_pool_address or ep.escrow_pool,
                ftns_token_address or ep.ftns_token,
                private_key=requester_key,
                # sp1200 review (defense-in-depth): pin the signer to the resolved
                # network's chain unless the caller pins it explicitly, so the deposit
                # refuses to sign against a divergent chain (e.g. a round-robin RPC).
                expected_chain_id=(
                    expected_chain_id if expected_chain_id is not None else ep.chain_id),
            )
        return await client.deposit(amount_wei)

    async def pay_and_infer(
        self,
        prompt: str,
        *,
        requester_key: str,
        provider_address: Optional[str] = None,
        model_id: str = "gpt2",
        max_tokens: int = 8,
        budget_ftns: float = 1.0,
        max_spend_ftns=None,
        privacy_tier: str = "none",
        content_tier: str = "A",
        chain_id: int = 8453,
        expiry_unix: Optional[int] = None,
        verify_pubkey_b64: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sprint 1189 — pay for one inference: sign an EIP-712 PaymentAuthorization bound
        to this exact request and POST it so the provider settles A→B from your escrow.

        Discovers the operator's payee address from GET /info when ``provider_address``
        isn't supplied. ``max_spend_ftns`` is the ceiling you authorize (defaults to
        ``budget_ftns``); ``chain_id`` MUST match the network (8453 mainnet / 84532 Base
        Sepolia); ``expiry_unix`` defaults to now+5min. You must have escrow balance ≥ the
        charge (see ``deposit_escrow``). Returns the server payload; a rejected
        authorization surfaces as HTTP 402 in the response. ``verify_pubkey_b64`` runs the
        inline receipt verification like ``infer``."""
        import time
        # sp1421 — the RECORDING variant. Identical auth (byte-for-byte); additionally
        # retains it so this requester can later PROVE a committed batch was never
        # authorized (the NO_ESCROW defense). See _issued_auth_store.
        from prsm.settlement.issued_authorization_store import (
            build_and_record_payment_authorization,
        )

        if provider_address is None:
            info = await self._get("/info")
            provider_address = (info or {}).get("operator_address")
            if not provider_address:
                raise ValueError(
                    "operator published no payment address (operator_address absent from "
                    "/info); supply provider_address explicitly"
                )
        if max_spend_ftns is None:
            max_spend_ftns = budget_ftns
        if expiry_unix is None:
            expiry_unix = int(time.time()) + 300
        _max_tokens = int(max_tokens or 0)
        auth = build_and_record_payment_authorization(
            store=self._issued_auth_store(),
            requester_key=requester_key,
            provider_address=provider_address,
            model_id=model_id,
            prompt=prompt,
            max_tokens=_max_tokens,
            privacy_tier=privacy_tier,
            content_tier=content_tier,
            max_spend_ftns=max_spend_ftns,
            expiry_unix=int(expiry_unix),
            chain_id=chain_id,
        )
        body = {
            "prompt": prompt,
            "model_id": model_id,
            "budget_ftns": budget_ftns,
            "privacy_tier": privacy_tier,
            "content_tier": content_tier,
            "max_tokens": _max_tokens,
            "payment_authorization": auth,
        }
        result = await self._post("/compute/inference", body)
        if verify_pubkey_b64:
            try:
                from prsm.compute.inference.models import InferenceReceipt
                from prsm.compute.inference.receipt import verify_receipt
                receipt = InferenceReceipt.from_dict(result.get("receipt") or {})
                result["receipt_verified"] = bool(
                    verify_receipt(receipt, public_key_b64=verify_pubkey_b64))
            except Exception:
                result["receipt_verified"] = False
        return result

    async def pay_and_infer_multistage(
        self,
        prompt: str,
        *,
        requester_key: str,
        model_id: str,
        max_tokens: int = 8,
        budget_ftns: float = 1.0,
        privacy_tier: str = "none",
        content_tier: str = "A",
        chain_id: int = 8453,
        expiry_unix: Optional[int] = None,
        verify_pubkey_b64: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sprint 1330 (S5) — pay for a big-model MULTI-STAGE (cross-host sliced) inference,
        end-to-end. The client glue that makes the proven Design-A paid path callable.

        Runs the full flow: (1) POST ``/compute/inference/quote-multistage`` to learn the
        planned stage→node payee set + the DETERMINISTIC price (sp1328); (2) sign ONE per-stage
        PaymentAuthorization over the quoted payees (``build_per_stage_payment_authorization``,
        sp1312) so each stage node can self-settle its share from your escrow; (3) POST
        ``/compute/inference`` with that auth. The provider serves the cross-host chain and each
        node commits its own share on-chain (msg.sender == provider).

        Raises ``ValueError`` (clearly) when the request doesn't route multi-stage (use
        ``pay_and_infer`` for the single-node case) or isn't settleable (e.g. a stage node has no
        on-chain payee, or the quoted price exceeds ``budget_ftns``). ``chain_id`` MUST match the
        network (8453 mainnet / 84532 Base Sepolia). You must hold escrow ≥ the quoted price (see
        ``deposit_escrow``). Returns the server result with the quote attached under
        ``result["multistage_quote"]``; ``verify_pubkey_b64`` runs inline receipt verification."""
        import time
        from decimal import Decimal

        # sp1450 — the RECORDING variant (per-stage sibling of the single-payee sp1421 fix). Identical
        # auth (byte-for-byte); additionally RETAINS it requester-side so the NO_ESCROW matcher can
        # recognize each stage node's authorized batch (and flag an inflated/foreign one). Without
        # this the store stayed EMPTY for per-stage and the matcher was blind to every per-stage batch.
        from prsm.settlement.issued_authorization_store import (
            build_and_record_per_stage_payment_authorization,
        )

        _max_tokens = int(max_tokens or 0)
        quote = await self._post("/compute/inference/quote-multistage", {
            "model_id": model_id, "prompt": prompt,
            "max_tokens": _max_tokens, "budget_ftns": budget_ftns,
        })
        if not quote.get("multi_stage"):
            raise ValueError(
                "request does not route multi-stage (single node) — use pay_and_infer: "
                + str(quote.get("reason", "")))
        if not quote.get("settleable"):
            raise ValueError(
                "multi-stage request not settleable: " + str(quote.get("reason", "")))

        if expiry_unix is None:
            expiry_unix = int(time.time()) + 300
        # wei → FTNS for the auth builder; the quote's shares ARE the price-based shares the
        # serve will settle (sp1328), so the signed payee_set_hash matches the settle.
        payees_ftns = [
            (addr, Decimal(str(share)) / (Decimal(10) ** 18))
            for addr, share in quote.get("payees", [])
        ]
        auth = build_and_record_per_stage_payment_authorization(
            store=self._issued_auth_store(),
            requester_key=requester_key, payees=payees_ftns,
            model_id=model_id, prompt=prompt, max_tokens=_max_tokens,
            privacy_tier=privacy_tier, content_tier=content_tier,
            expiry_unix=int(expiry_unix), chain_id=chain_id,
        )
        result = await self._post("/compute/inference", {
            "prompt": prompt, "model_id": model_id, "budget_ftns": budget_ftns,
            "privacy_tier": privacy_tier, "content_tier": content_tier,
            "max_tokens": _max_tokens, "per_stage_payment_authorization": auth,
        })
        result["multistage_quote"] = {
            "price_ftns": quote.get("price_ftns"),
            "stage_count": quote.get("stage_count"),
            "payees": quote.get("payees"),
            "payee_set_hash": quote.get("payee_set_hash"),
        }
        if verify_pubkey_b64:
            try:
                from prsm.compute.inference.models import InferenceReceipt
                from prsm.compute.inference.receipt import verify_receipt
                receipt = InferenceReceipt.from_dict(result.get("receipt") or {})
                result["receipt_verified"] = bool(
                    verify_receipt(receipt, public_key_b64=verify_pubkey_b64))
            except Exception:
                result["receipt_verified"] = False
        return result

    async def relayed_infer(
        self,
        prompt: str,
        *,
        relayer_key: str,
        payment_delegation: Dict[str, Any],
        provider_address: Optional[str] = None,
        model_id: str = "gpt2",
        max_tokens: int = 8,
        budget_ftns: float = 1.0,
        max_spend_ftns=None,
        privacy_tier: str = "none",
        content_tier: str = "A",
        chain_id: int = 8453,
        expiry_unix: Optional[int] = None,
        verify_pubkey_b64: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sprint 1311 — SPONSORED (relayer) inference for a WALLET-LESS end-user.

        The relayer/gateway (holding ``relayer_key``) signs a per-request PaymentAuthorization
        and attaches the FUNDER's pre-signed ``payment_delegation`` (from
        ``build_payment_delegation``, issued out-of-band by the sponsor). The provider
        verifies the two-signature chain and settles A→B from the FUNDER's escrow — so the
        END-USER holds no wallet and signs nothing; the gateway sponsors the inference under
        the funder's cap. Returns the server payload; a rejected chain surfaces as HTTP 402.
        ``verify_pubkey_b64`` runs the inline receipt verification like ``pay_and_infer``."""
        import time
        # sp1421 — the RECORDING variant. Identical auth (byte-for-byte); additionally
        # retains it so this requester can later PROVE a committed batch was never
        # authorized (the NO_ESCROW defense). See _issued_auth_store.
        from prsm.settlement.issued_authorization_store import (
            build_and_record_payment_authorization,
        )

        if provider_address is None:
            info = await self._get("/info")
            provider_address = (info or {}).get("operator_address")
            if not provider_address:
                raise ValueError(
                    "operator published no payment address (operator_address absent from "
                    "/info); supply provider_address explicitly"
                )
        if max_spend_ftns is None:
            max_spend_ftns = budget_ftns
        if expiry_unix is None:
            expiry_unix = int(time.time()) + 300
        _max_tokens = int(max_tokens or 0)
        # The RELAYER signs the per-request auth (auth.requester == relayer); the funder's
        # delegation (signed separately) authorizes this relayer to spend from its escrow.
        auth = build_and_record_payment_authorization(
            store=self._issued_auth_store(),
            requester_key=relayer_key,
            provider_address=provider_address,
            model_id=model_id,
            prompt=prompt,
            max_tokens=_max_tokens,
            privacy_tier=privacy_tier,
            content_tier=content_tier,
            max_spend_ftns=max_spend_ftns,
            expiry_unix=int(expiry_unix),
            chain_id=chain_id,
        )
        body = {
            "prompt": prompt,
            "model_id": model_id,
            "budget_ftns": budget_ftns,
            "privacy_tier": privacy_tier,
            "content_tier": content_tier,
            "max_tokens": _max_tokens,
            "payment_authorization": auth,
            "payment_delegation": payment_delegation,
        }
        result = await self._post("/compute/inference", body)
        if verify_pubkey_b64:
            try:
                from prsm.compute.inference.models import InferenceReceipt
                from prsm.compute.inference.receipt import verify_receipt
                receipt = InferenceReceipt.from_dict(result.get("receipt") or {})
                result["receipt_verified"] = bool(
                    verify_receipt(receipt, public_key_b64=verify_pubkey_b64))
            except Exception:
                result["receipt_verified"] = False
        return result

    # ── Sprint 821 — Content publish + fetch ────────────────────

    async def publish_content(
        self,
        text: str,
        *,
        filename: str = "document.txt",
        replicas: int = 3,
        royalty_rate: Optional[float] = None,
        parent_cids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Sprint 821 — POST /content/upload. Returns server
        payload with the assigned CID + filename + size.

        Mirror of `prsm content publish` CLI (sprint 806).
        Defaults match the CLI + server-side defaults.

        For binary content or text > 100MB, use the shard
        endpoint instead (no SDK wrapper yet; see sprint 817's
        CLI for the canonical body shape).
        """
        body: Dict[str, Any] = {
            "text": text,
            "filename": filename,
            "replicas": replicas,
            "parent_cids": list(parent_cids) if parent_cids else [],
        }
        if royalty_rate is not None:
            body["royalty_rate"] = royalty_rate
        return await self._post("/content/upload", body)

    async def search_content(
        self,
        query: str,
        *,
        limit: int = 20,
        min_tier: Optional[str] = None,
        exclude_new: bool = False,
    ) -> Dict[str, Any]:
        """Sprint 822 — GET /content/search; return server payload.

        Mirror of `prsm content search` CLI (sprint 808).
        Defaults match the CLI.

        Returns ``{results, count}``. Each result row carries
        ``cid``, ``filename``, ``creator_tier`` (server-side
        sprint 289).
        """
        await self._ensure_session()
        import aiohttp as _aiohttp

        params: Dict[str, Any] = {"q": query, "limit": limit}
        if min_tier:
            params["min_tier"] = min_tier
        if exclude_new:
            params["exclude_new"] = "true"

        async with self._session.get(
            f"{self.base_url}/content/search",
            headers=self._headers(),
            timeout=_aiohttp.ClientTimeout(total=15.0),
            params=params,
        ) as resp:
            return await resp.json()

    async def search_content_semantic(
        self,
        query: str,
        *,
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> Dict[str, Any]:
        """Sprint 1344 — GET /content/search/semantic: find content CONCEPTUALLY related to
        ``query`` (embedding-similarity), not just keyword matches. The complement to
        ``search_content`` for topic/concept discovery.

        Returns ``{query, results, count, semantic_available}``. Each result row carries
        ``cid``, ``similarity`` (cosine), ``filename``, ``creator_eth_address``,
        ``provenance_hash``, ``metadata``. ``semantic_available`` is False when the serving
        node has no embedding function wired — then ``search_content`` (keyword) is the path.
        """
        await self._ensure_session()
        import aiohttp as _aiohttp

        params: Dict[str, Any] = {
            "q": query, "top_k": top_k, "min_similarity": min_similarity}
        async with self._session.get(
            f"{self.base_url}/content/search/semantic",
            headers=self._headers(),
            timeout=_aiohttp.ClientTimeout(total=20.0),
            params=params,
        ) as resp:
            return await resp.json()

    async def fetch_content(
        self,
        cid: str,
        *,
        timeout: float = 30.0,
        verify_hash: bool = True,
    ) -> Dict[str, Any]:
        """Sprint 821 — GET /content/retrieve/{cid}. Returns
        the server payload with `data` as base64-encoded bytes
        (caller decodes for binary content).

        Mirror of `prsm content fetch` CLI (sprint 805).

        Threads ``timeout`` + ``verify_hash`` as query params;
        the server-side timeout cap (PRSM_MAX_RETRIEVE_TIMEOUT_SEC,
        default 300) applies.

        sp1338 — on ``status == "success"`` the payload also carries
        ``content_hash`` (integrity), plus ``creator_eth_address`` and
        ``provenance_hash`` (verifiable provenance + creator attribution),
        so a consumer can confirm WHO created the fetched dataset and its
        on-chain provenance commitment. Both are ``None`` for content that
        predates creator threading.
        """
        await self._ensure_session()
        import aiohttp as _aiohttp

        params: Dict[str, Any] = {"timeout": timeout}
        # The fastapi server reads bool from "false" / "true"
        # lowercased strings — serialize accordingly.
        params["verify_hash"] = "true" if verify_hash else "false"

        async with self._session.get(
            f"{self.base_url}/content/retrieve/{cid}",
            headers=self._headers(),
            timeout=_aiohttp.ClientTimeout(total=timeout + 5.0),
            params=params,
        ) as resp:
            return await resp.json()

    async def fetch_content_to_file(
        self,
        cid: str,
        dest_path,
        *,
        timeout: float = 30.0,
        verify_hash: bool = True,
        chunk_size: int = 262_144,
    ) -> str:
        """Sprint 1457 — GET /content/retrieve-stream/{cid}, streaming the body straight to
        ``dest_path`` with NO full-content buffering. The SDK companion to ``fetch_content``
        for LARGE content (above the in-memory retrieve ceiling), mirroring
        ``prsm content fetch --stream``. The server fetches to a temp file (CID re-verified by
        a streaming read) and streams it back; here it is written chunk-by-chunk to disk, so a
        multi-GiB dataset never materializes in memory on either side.

        Returns the destination path (str) on success. Raises ``ValueError`` on a non-200
        response (404 not found, 451 blocked, 503 no provider, 504 timeout) with the server
        detail, matching the SDK's error convention."""
        await self._ensure_session()
        import aiohttp as _aiohttp

        params: Dict[str, Any] = {
            "timeout": timeout,
            "verify_hash": "true" if verify_hash else "false",
        }
        async with self._session.get(
            f"{self.base_url}/content/retrieve-stream/{cid}",
            headers=self._headers(),
            timeout=_aiohttp.ClientTimeout(total=timeout + 5.0),
            params=params,
        ) as resp:
            if resp.status != 200:
                detail = (await resp.text())[:300]
                raise ValueError(
                    f"stream fetch of {cid} failed ({resp.status}): {detail}")
            with open(dest_path, "wb") as fh:
                async for chunk in resp.content.iter_chunked(chunk_size):
                    fh.write(chunk)
        return str(dest_path)

    async def publish_content_from_file(
        self,
        src_path,
        *,
        filename: Optional[str] = None,
        chunk_size: int = 262_144,
    ) -> Dict[str, Any]:
        """Sprint 1457 — stream a LARGE Tier-A file to POST /content/upload-stream with NO
        in-memory buffering (the SDK companion to ``prsm content publish --stream``). Reads
        ``src_path`` in chunks and returns the server payload
        ``{cid, content_hash, size_bytes, status}``. Raises ``ValueError`` on a non-200
        response (501 no streaming publisher, 413 over cap, 451 blocked, 500)."""
        await self._ensure_session()
        import aiohttp as _aiohttp
        from pathlib import Path as _Path

        src = _Path(src_path)
        params: Dict[str, Any] = {"filename": filename or src.name}
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async def _body():
            with open(src, "rb") as fh:
                while True:
                    block = fh.read(chunk_size)
                    if not block:
                        break
                    yield block

        async with self._session.post(
            f"{self.base_url}/content/upload-stream",
            headers=headers,
            params=params,
            data=_body(),
        ) as resp:
            if resp.status != 200:
                detail = (await resp.text())[:300]
                raise ValueError(
                    f"stream publish of {src} failed ({resp.status}): {detail}")
            return await resp.json()

    async def find_and_fetch(
        self,
        query: str,
        *,
        min_tier: Optional[str] = None,
        exclude_new: bool = False,
        timeout: float = 30.0,
        verify_provenance: bool = False,
        provenance_verifier: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Sprint 1341/1342 — one-call find → fetch → verify: the flagship data-consumer path.

        Topic-searches the network (sp1339/1340), fetches the TOP hit, and INDEPENDENTLY
        re-verifies the retrieved bytes client-side (sha256 == the record's content_hash — a
        defense-in-depth check that doesn't trust the server's own verify), returning the
        content plus its verifiable creator/provenance in one shot.

        Raises ``FileNotFoundError`` when the search finds nothing or the top hit isn't
        retrievable. The returned dict carries:
          data (base64) · content_hash · integrity_verified (bool) · creator_eth_address ·
          provenance_hash · filename · size_bytes · creator_tier · matched (the search row).

        Two independent trust checks:
          * ``integrity_verified`` (always) — the bytes match their claimed content_hash.
          * ``authenticity_verified`` (only when ``verify_provenance=True``, sp1342) — the
            content's ``provenance_hash`` resolves ON-CHAIN (ProvenanceRegistry) to a creator
            that MATCHES the claimed ``creator_eth_address``. This is the trustless check that
            the provider isn't lying about who made the data: the client reads the chain
            itself, it does not ask the (untrusted) serving node. Also returns
            ``registered_creator`` (the on-chain creator, or None). ``provenance_verifier`` is
            an injectable ``provenance_hash -> Optional[creator_address]`` callable (sync or
            async); when omitted, a read-only ProvenanceRegistry client is built from
            env/network config (None → authenticity_verified stays False with a reason).
        """
        import base64
        import hashlib

        search = await self.search_content(
            query, limit=1, min_tier=min_tier, exclude_new=exclude_new)
        results = (search or {}).get("results") or []
        if not results:
            raise FileNotFoundError(f"no content found for query {query!r}")
        top = results[0]
        cid = top.get("cid")

        fetched = await self.fetch_content(cid, timeout=timeout, verify_hash=True)
        if (fetched or {}).get("status") != "success":
            raise FileNotFoundError(
                f"top match {cid!r} not retrievable "
                f"(status={(fetched or {}).get('status')!r})")

        content_hash = fetched.get("content_hash")
        data_b64 = fetched.get("data") or ""
        integrity_verified = False
        try:
            if content_hash and data_b64:
                computed = hashlib.sha256(base64.b64decode(data_b64)).hexdigest()
                integrity_verified = computed == content_hash
        except Exception:
            integrity_verified = False

        result: Dict[str, Any] = {
            "cid": cid,
            "data": data_b64,
            "content_hash": content_hash,
            "integrity_verified": integrity_verified,
            # provenance/attribution — prefer the retrieve values (authoritative), fall back
            # to the search row (both surface them since sp1338/1339).
            "creator_eth_address": (
                fetched.get("creator_eth_address") or top.get("creator_eth_address")),
            "provenance_hash": fetched.get("provenance_hash") or top.get("provenance_hash"),
            "filename": fetched.get("filename") or top.get("filename"),
            "size_bytes": fetched.get("size_bytes"),
            "creator_tier": top.get("creator_tier"),
            "matched": top,
        }

        # sp1342 — full ON-CHAIN AUTHENTICITY (opt-in). Resolve the provenance_hash against the
        # ProvenanceRegistry and confirm the claimed creator IS the on-chain-registered creator.
        # Trustless: the client reads the chain itself, never asking the untrusted serving node.
        if verify_provenance:
            import asyncio
            ph = result["provenance_hash"]
            claimed = result["creator_eth_address"]
            registered = None
            reason = None
            verifier = provenance_verifier
            if verifier is None:
                verifier = self._default_provenance_verifier()
            if not ph:
                reason = "content carries no provenance_hash (unregistered)"
            elif verifier is None:
                reason = ("no on-chain verifier — set PRSM_PROVENANCE_REGISTRY_ADDRESS + an RPC "
                          "URL, or pass provenance_verifier")
            else:
                try:
                    if asyncio.iscoroutinefunction(verifier):
                        registered = await verifier(ph)
                    else:
                        registered = await asyncio.to_thread(verifier, ph)
                except Exception as exc:  # noqa: BLE001 — a lookup failure is NOT authenticity
                    reason = f"provenance lookup failed: {type(exc).__name__}: {exc}"
            authentic = bool(
                registered and claimed
                and str(registered).lower() == str(claimed).lower())
            if not authentic and reason is None:
                if not registered:
                    reason = "provenance_hash is not registered on-chain"
                else:
                    reason = (f"on-chain creator {registered} does NOT match the claimed "
                              f"creator {claimed}")
            result["registered_creator"] = registered
            result["authenticity_verified"] = authentic
            result["authenticity_detail"] = "verified" if authentic else reason

        return result

    def _default_provenance_verifier(self):
        """sp1342 — a read-only ``provenance_hash -> Optional[creator_address]`` verifier backed
        by the on-chain ProvenanceRegistry, built from env/network config (no key needed). Returns
        None when web3/registry/RPC aren't configured (→ authenticity stays unverified, honestly)."""
        try:
            import os

            from prsm.config.networks import resolve_endpoints
            from prsm.economy.web3.provenance_registry import ProvenanceRegistryClient

            ep = resolve_endpoints()
            registry = (os.environ.get("PRSM_PROVENANCE_REGISTRY_ADDRESS", "").strip()
                        or getattr(ep, "provenance_registry", None))
            rpc = (os.environ.get("PRSM_BASE_RPC_URL", "").strip()
                   or os.environ.get("BASE_RPC_URL", "").strip()
                   or getattr(ep, "rpc_url", None))
            if not registry or not rpc:
                return None
            client = ProvenanceRegistryClient(rpc_url=rpc, contract_address=registry)

            def _verify(ph):
                ch = bytes.fromhex(str(ph).removeprefix("0x"))
                if len(ch) != 32:
                    return None
                rec = client.get_content(ch)
                return getattr(rec, "creator", None) if rec else None

            return _verify
        except Exception:  # noqa: BLE001 — unconfigured/unavailable → no verifier (honest None)
            return None

    # ── Sprint 1355 — Tier B/C paid content: buy + decrypt ──────

    async def pay_and_unlock_content(
        self,
        content_hash,
        *,
        requester_key: str,
        x25519_privkey_b64: str,
        fee_wei: int,
        verifier_address: str,
        commitment=None,
        cid: Optional[str] = None,
        network: Optional[str] = None,
        rpc_url: Optional[str] = None,
        ftns_token_address: Optional[str] = None,
        key_distribution_address: Optional[str] = None,
        _verifier_client: Any = None,
        _content: Any = None,
        _fetch_wrapped_key: Any = None,
        _key_client: Any = None,
        _creator_reader: Any = None,
    ) -> bytes:
        """Sprint 1355 (+1359 F1 redesign) — buy + decrypt a Tier B/C paid dataset.

        Pays the release fee to the ContentAccessVerifier, then FETCHES the wrapped key from the
        publisher's payment-gated endpoint (GET /content/paid-key/…, signing the challenge with
        your ETH key) and VERIFIES it against the on-chain ``commitment`` before decrypting with
        your X25519 key. The wrapped key never lives in world-readable on-chain storage (the B5 F1
        fix). FAIL-LOUD: unpaid / wrong served key / tampered bytes raise.

        ``commitment``: the 32-byte sha256 commitment to the wrapped key (hex or bytes) — obtain it
          from the publisher-signed manifest or the on-chain deposit; the served key is checked
          against it. ``requester_key``: your ETH key (signs payForAccess + the key-fetch challenge).
          ``x25519_privkey_b64``: your DECRYPT key. ``verifier_address``: the deployed
          ContentAccessVerifier. Injectables (``_verifier_client`` / ``_content`` /
          ``_fetch_wrapped_key``) are for tests; keys are never stored on the client.
        """
        import asyncio
        import base64

        from prsm.economy.paid_content import (
            assert_fee_matches_deposit,
            assert_publisher_controls_payee,
            build_content_access_settle_fee,
            pay_and_unlock,
        )
        from prsm.storage.paid_unlock import deserialize_encrypted_content

        ch = content_hash if isinstance(content_hash, (bytes, bytearray)) else bytes.fromhex(
            content_hash[2:] if str(content_hash).startswith("0x") else content_hash)
        if len(ch) != 32:
            raise ValueError("content_hash must be 32 bytes")
        if int(fee_wei) <= 0:
            raise ValueError("fee_wei must be positive")

        commit = commitment
        if commit is not None and not isinstance(commit, (bytes, bytearray)):
            commit = bytes.fromhex(commit[2:] if str(commit).startswith("0x") else commit)
        # sp1363 (R5 MEDIUM): if the caller doesn't supply an authoritative commitment, we read it
        # from the on-chain KeyReleased event below (never trust a publisher/MITM-supplied value).

        # Resolve chain config (RPC / FTNS / KeyDistribution) unless injected.
        if _verifier_client is None or _key_client is None:
            from prsm.config.networks import resolve_endpoints
            ep = resolve_endpoints(network)
            rpc = rpc_url or ep.rpc_url
            ftns = ftns_token_address or ep.ftns_token
            keydist = key_distribution_address or ep.key_distribution
            chain_id = ep.chain_id

        verifier_client = _verifier_client
        if verifier_client is None:
            from prsm.economy.web3.content_access_verifier import ContentAccessVerifierClient
            # sp1356 (review F7): pin the intended chain.
            verifier_client = ContentAccessVerifierClient(
                rpc, verifier_address, ftns, private_key=requester_key,
                expected_chain_id=chain_id)

        # The KeyDistribution client — SIGNING (sp1363: it also drives the gated release() that
        # surfaces the authoritative on-chain commitment) — unless injected.
        key_client = _key_client
        if key_client is None:
            from prsm.economy.web3.key_distribution import KeyDistributionClient
            key_client = KeyDistributionClient(rpc, keydist, private_key=requester_key)
        # sp1361 (review F3/F4/F12): confirm the fee matches the on-chain deposit BEFORE paying.
        assert_fee_matches_deposit(key_client, ch, int(fee_wei))

        # sp1365 (review F9/F2): refuse to pay if the fee payee (the CAV's registry creator) is not
        # the key depositor (KeyDistribution publisher) — the signature of squatting. Best-effort:
        # if the registry can't be read, skip (the fee check is the hard gate); a real mismatch
        # raises SquatMismatchError and aborts before any payment.
        # sp1417 — ``_creator_reader`` is injectable so the guard's INVOCATION is testable; without a
        # seam its construction failed under injected clients (rpc unresolved) and the guard silently
        # skipped, so nothing could verify pay_and_unlock_content actually calls it.
        creator_reader = _creator_reader
        if creator_reader is None:
            try:
                from prsm.economy.web3.provenance_registry_v2 import ProvenanceRegistryV2Client
                _reg = ProvenanceRegistryV2Client(
                    rpc_url=rpc, contract_address=verifier_client.registry_address())
                creator_reader = (
                    lambda c, _r=_reg: _r.contract.functions.getCreatorAndRate(c).call()[0])
            except Exception:  # noqa: BLE001 — unreadable registry ⇒ skip (not a mismatch)
                creator_reader = None
        assert_publisher_controls_payee(key_client, creator_reader, ch)

        # sp1363 (R5 MEDIUM): when no authoritative commitment was supplied, read it from the
        # gated KeyReleased event (acquire_released_key) AFTER payment — never from the untrusted
        # publisher/serve path.
        fetch_commitment = None
        if commit is None:
            from eth_account import Account as _Account

            from prsm.economy.web3.key_acquisition import acquire_released_key
            _payer = _Account.from_key(requester_key).address

            def fetch_commitment(_kc=key_client, _c=ch, _p=_payer):
                return bytes(acquire_released_key(_kc, _c, _p))

        # The freely-served ciphertext (fetched async, then decrypted in the worker thread).
        content = _content
        if content is None:
            row = await self.fetch_content(cid or ("0x" + ch.hex()))
            data_b64 = (row or {}).get("data")
            if not data_b64:
                from prsm.storage.paid_unlock import PaidUnlockError
                raise PaidUnlockError(
                    f"ciphertext for content {ch.hex()[:12]}… is not retrievable")
            content = deserialize_encrypted_content(base64.b64decode(data_b64))

        # The wrapped-key fetch is a SYNC blocking call run inside the worker thread AFTER settle_fee
        # (the endpoint gates on payment), so the pay→fetch ordering holds.
        base_url = self.base_url
        fetch_wrapped_key = _fetch_wrapped_key
        if fetch_wrapped_key is None:
            def fetch_wrapped_key(_c, _u=base_url, _k=requester_key):
                return _fetch_paid_key_sync(_u, _c, _k)

        settle_fee = build_content_access_settle_fee(verifier_client, ch, int(fee_wei))

        return await asyncio.to_thread(
            pay_and_unlock,
            content_hash=ch, recipient_privkey_b64=x25519_privkey_b64, commitment=commit,
            fetch_commitment=fetch_commitment, fetch_wrapped_key=fetch_wrapped_key,
            retrieve_content=lambda _c: content, settle_fee=settle_fee)

    # ── Sprint 820 — Streaming verifiable inference ─────────────

    async def _stream_events(self, body: Dict[str, Any]):
        """Sprint 820/1310 — POST ``body`` to /compute/inference/stream and yield the SSE
        token/result/error events. Shared by ``infer_stream`` (self-pay) and
        ``pay_and_infer_stream`` (paid) so the SSE parsing lives in one place. Non-200 →
        a single ``error`` event then stop (sp827: surface the server detail instead of an
        empty generator); iteration ends after the first ``result``/``error`` event (extra
        frames from a misbehaving server are ignored)."""
        await self._ensure_session()
        import aiohttp as _aiohttp

        async with self._session.post(
            f"{self.base_url}/compute/inference/stream",
            json=body,
            headers=self._headers(),
            timeout=_aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                try:
                    detail_text = (await resp.read()).decode("utf-8", "replace")
                except Exception:
                    detail_text = (
                        f"<unable to read response body for status {resp.status}>"
                    )
                yield {"type": "error", "status": resp.status, "detail": detail_text}
                return
            # Parse SSE event/data frames from line-streamed bytes.
            current_event = None
            buffer = b""
            async for chunk in resp.content:
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line_text = line.decode("utf-8", "replace").rstrip("\r")
                    if not line_text:
                        current_event = None  # blank line = end of an event
                        continue
                    if line_text.startswith("event: "):
                        current_event = line_text[len("event: "):].strip()
                    elif line_text.startswith("data: "):
                        if current_event is None:
                            continue
                        try:
                            import json as _json
                            payload = _json.loads(line_text[len("data: "):])
                        except Exception:
                            continue
                        ev = {"type": current_event, **payload}
                        yield ev
                        if current_event in ("result", "error"):
                            return

    async def infer_stream(
        self,
        prompt: str,
        *,
        model_id: str = "gpt2",
        max_tokens: int = 8,
        budget_ftns: float = 1.0,
        privacy_tier: str = "none",
        content_tier: str = "A",
    ):
        """Sprint 820 — async generator consuming SSE from
        /compute/inference/stream (self-pay).

        Yields events as dicts with a `type` discriminator:
          {"type": "token", "sequence_index": N, "text_delta",
           "token_id", "finish_reason"}
          {"type": "result", "success", "output",
           "ftns_charged", "receipt", ...}
          {"type": "error", "detail"}

        Iteration terminates after the first `result` or `error` event.
        Defaults match sprint 819 .infer() so users can swap between unary
        and streaming with one keyword.

        Usage:
            async for ev in client.infer_stream(prompt="..."):
                if ev["type"] == "token":
                    print(ev["text_delta"], end="", flush=True)
                elif ev["type"] == "result":
                    print("Receipt:", ev["receipt"])
        """
        body = {
            "prompt": prompt,
            "model_id": model_id,
            "budget_ftns": budget_ftns,
            "privacy_tier": privacy_tier,
            "content_tier": content_tier,
            "max_tokens": max_tokens,
        }
        async for ev in self._stream_events(body):
            yield ev

    async def pay_and_infer_stream(
        self,
        prompt: str,
        *,
        requester_key: str,
        provider_address: Optional[str] = None,
        model_id: str = "gpt2",
        max_tokens: int = 8,
        budget_ftns: float = 1.0,
        max_spend_ftns=None,
        privacy_tier: str = "none",
        content_tier: str = "A",
        chain_id: int = 8453,
        expiry_unix: Optional[int] = None,
        verify_pubkey_b64: Optional[str] = None,
    ):
        """Sprint 1310 — paid STREAMING inference: the streaming twin of ``pay_and_infer``.

        Signs an EIP-712 PaymentAuthorization bound to this exact request and sends it to
        /compute/inference/stream, which verifies it FAIL-CLOSED and settles A→B from your
        escrow (the streaming server path already verifies + records + settles the requester,
        sp1056). Yields the same token/result/error events as ``infer_stream``; when
        ``verify_pubkey_b64`` is set, the terminal ``result`` event gains a
        ``receipt_verified`` bool (parity with ``pay_and_infer``). Discovers the operator's
        payee from GET /info when ``provider_address`` is absent; you need escrow balance ≥
        the charge (see ``deposit_escrow``); ``chain_id`` MUST match the network."""
        import time
        # sp1421 — the RECORDING variant. Identical auth (byte-for-byte); additionally
        # retains it so this requester can later PROVE a committed batch was never
        # authorized (the NO_ESCROW defense). See _issued_auth_store.
        from prsm.settlement.issued_authorization_store import (
            build_and_record_payment_authorization,
        )

        if provider_address is None:
            info = await self._get("/info")
            provider_address = (info or {}).get("operator_address")
            if not provider_address:
                raise ValueError(
                    "operator published no payment address (operator_address absent from "
                    "/info); supply provider_address explicitly"
                )
        if max_spend_ftns is None:
            max_spend_ftns = budget_ftns
        if expiry_unix is None:
            expiry_unix = int(time.time()) + 300
        _max_tokens = int(max_tokens or 0)
        auth = build_and_record_payment_authorization(
            store=self._issued_auth_store(),
            requester_key=requester_key,
            provider_address=provider_address,
            model_id=model_id,
            prompt=prompt,
            max_tokens=_max_tokens,
            privacy_tier=privacy_tier,
            content_tier=content_tier,
            max_spend_ftns=max_spend_ftns,
            expiry_unix=int(expiry_unix),
            chain_id=chain_id,
        )
        body = {
            "prompt": prompt,
            "model_id": model_id,
            "budget_ftns": budget_ftns,
            "privacy_tier": privacy_tier,
            "content_tier": content_tier,
            "max_tokens": _max_tokens,
            "payment_authorization": auth,
        }
        async for ev in self._stream_events(body):
            if ev.get("type") == "result" and verify_pubkey_b64:
                try:
                    from prsm.compute.inference.models import InferenceReceipt
                    from prsm.compute.inference.receipt import verify_receipt
                    receipt = InferenceReceipt.from_dict(ev.get("receipt") or {})
                    ev["receipt_verified"] = bool(
                        verify_receipt(receipt, public_key_b64=verify_pubkey_b64))
                except Exception:
                    ev["receipt_verified"] = False
            yield ev

    async def pay_and_infer_multistage_stream(
        self,
        prompt: str,
        *,
        requester_key: str,
        model_id: str,
        max_tokens: int = 8,
        budget_ftns: float = 1.0,
        privacy_tier: str = "none",
        content_tier: str = "A",
        chain_id: int = 8453,
        expiry_unix: Optional[int] = None,
        verify_pubkey_b64: Optional[str] = None,
    ):
        """Sprint 1332 (S5) — paid big-model MULTI-STAGE STREAMING inference: the streaming twin
        of ``pay_and_infer_multistage``.

        Quotes the stage→node payee set + the deterministic price, signs ONE per-stage
        PaymentAuthorization over it, and streams ``/compute/inference/stream`` with it; each
        stage node self-settles its share (the streaming settle path fires the per-stage delivery,
        sp1332). Yields the same token/result/error events as ``infer_stream``; the terminal
        ``result`` event gains ``multistage_quote`` (price + payees) and, when ``verify_pubkey_b64``
        is set, ``receipt_verified``.

        NOTE: sliced big models CANNOT stream — the server disables the streaming runner under
        ``PRSM_PARALLAX_SLICE_LOAD`` (a sliced node can't run ``model.generate``). This serves
        multi-stage models that DON'T slice; for a sliced big model use the unary
        ``pay_and_infer_multistage``. Raises ``ValueError`` on not-multi-stage / not-settleable."""
        import time
        from decimal import Decimal

        # sp1450 — the RECORDING variant (per-stage sibling of the single-payee sp1421 fix). Identical
        # auth (byte-for-byte); additionally RETAINS it requester-side so the NO_ESCROW matcher can
        # recognize each stage node's authorized batch (and flag an inflated/foreign one). Without
        # this the store stayed EMPTY for per-stage and the matcher was blind to every per-stage batch.
        from prsm.settlement.issued_authorization_store import (
            build_and_record_per_stage_payment_authorization,
        )

        _max_tokens = int(max_tokens or 0)
        quote = await self._post("/compute/inference/quote-multistage", {
            "model_id": model_id, "prompt": prompt,
            "max_tokens": _max_tokens, "budget_ftns": budget_ftns,
        })
        if not quote.get("multi_stage"):
            raise ValueError(
                "request does not route multi-stage (single node) — use pay_and_infer_stream: "
                + str(quote.get("reason", "")))
        if not quote.get("settleable"):
            raise ValueError(
                "multi-stage request not settleable: " + str(quote.get("reason", "")))

        if expiry_unix is None:
            expiry_unix = int(time.time()) + 300
        payees_ftns = [
            (addr, Decimal(str(share)) / (Decimal(10) ** 18))
            for addr, share in quote.get("payees", [])
        ]
        auth = build_and_record_per_stage_payment_authorization(
            store=self._issued_auth_store(),
            requester_key=requester_key, payees=payees_ftns,
            model_id=model_id, prompt=prompt, max_tokens=_max_tokens,
            privacy_tier=privacy_tier, content_tier=content_tier,
            expiry_unix=int(expiry_unix), chain_id=chain_id,
        )
        body = {
            "prompt": prompt, "model_id": model_id, "budget_ftns": budget_ftns,
            "privacy_tier": privacy_tier, "content_tier": content_tier,
            "max_tokens": _max_tokens, "per_stage_payment_authorization": auth,
        }
        async for ev in self._stream_events(body):
            if ev.get("type") == "result":
                ev["multistage_quote"] = {
                    "price_ftns": quote.get("price_ftns"),
                    "stage_count": quote.get("stage_count"),
                    "payees": quote.get("payees"),
                    "payee_set_hash": quote.get("payee_set_hash"),
                }
                if verify_pubkey_b64:
                    try:
                        from prsm.compute.inference.models import InferenceReceipt
                        from prsm.compute.inference.receipt import verify_receipt
                        receipt = InferenceReceipt.from_dict(ev.get("receipt") or {})
                        ev["receipt_verified"] = bool(
                            verify_receipt(receipt, public_key_b64=verify_pubkey_b64))
                    except Exception:
                        ev["receipt_verified"] = False
            yield ev

    # ── Ring 4: Pricing ───────────────────────────────────────────

    async def quote(
        self,
        query: str,
        shards: int = 3,
        tier: str = "t2",
    ) -> Dict[str, Any]:
        """Get a cost estimate for a query.

        Note: This is a client-side estimate using the pricing engine.
        For server-side quotes, use query() with a small budget.
        """
        # Client-side pricing estimate
        from prsm.economy.pricing import PricingEngine
        engine = PricingEngine()
        q = engine.quote_swarm_job(
            shard_count=shards,
            hardware_tier=tier,
            estimated_pcu_per_shard=50.0,
        )
        return q.to_dict()

    # ── Ring 3: Data Upload ───────────────────────────────────────

    async def upload_dataset(
        self,
        dataset_id: str,
        title: str,
        content: bytes,
        shard_count: int = 4,
        base_access_fee: float = 0.0,
        per_shard_fee: float = 0.0,
    ) -> Dict[str, Any]:
        """Upload a dataset with semantic sharding."""
        import base64
        return await self._post("/content/upload/shard", {
            "dataset_id": dataset_id,
            "title": title,
            "content_b64": base64.b64encode(content).decode(),
            "shard_count": shard_count,
            "base_access_fee": base_access_fee,
            "per_shard_fee": per_shard_fee,
        })

    # ── Settlement ────────────────────────────────────────────────

    async def settlement_stats(self) -> Dict[str, Any]:
        """Get FTNS settlement queue stats."""
        return await self._get("/settlement/stats")
