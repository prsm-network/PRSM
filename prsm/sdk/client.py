"""
PRSM Node Client
=================

Async Python client for PRSM node Ring 1-10 APIs.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._session = None

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
        from prsm.settlement.payment_client import build_payment_authorization

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
        auth = build_payment_authorization(
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

        from prsm.settlement.payment_client import (
            build_per_stage_payment_authorization,
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
        auth = build_per_stage_payment_authorization(
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
        from prsm.settlement.payment_client import build_payment_authorization

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
        auth = build_payment_authorization(
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
        from prsm.settlement.payment_client import build_payment_authorization

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
        auth = build_payment_authorization(
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
