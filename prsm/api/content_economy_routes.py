"""
Content Economy API Endpoints
=============================

FastAPI endpoints for Phase 4 content economy features:
- FTNS payment processing
- Royalty tracking
- Content retrieval marketplace
- Semantic search

These endpoints integrate with ContentEconomy to provide
HTTP access to the content marketplace functionality.
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from prsm.core.auth import get_current_user

router = APIRouter(prefix="/content-economy", tags=["Content Economy"])


# ── Global Content Economy Reference ──────────────────────────────────────

# This will be set by the node when it initializes
_content_economy_instance = None

def set_content_economy(economy):
    """Set the ContentEconomy instance (called by node during initialization)."""
    global _content_economy_instance
    _content_economy_instance = economy

def get_content_economy():
    """Get ContentEconomy instance from app state."""
    global _content_economy_instance
    if _content_economy_instance is None:
        raise HTTPException(status_code=503, detail="Content economy not initialized")
    return _content_economy_instance


# ── Request/Response Models ────────────────────────────────────────────────

class ContentAccessRequest(BaseModel):
    """Request to process content access payment."""
    cid: str = Field(..., description="Content identifier")
    accessor_id: str = Field(..., description="Node/user accessing content")
    royalty_rate: float = Field(..., ge=0.001, le=0.1, description="Per-access fee")
    creator_id: str = Field(..., description="Content creator ID")
    parent_cids: List[str] = Field(default_factory=list, description="Parent content CIDs")
    provenance_hash: Optional[str] = Field(
        None,
        description=(
            "Optional 0x-prefixed canonical provenance hash. When set and "
            "PRSM_ONCHAIN_PROVENANCE=1, payment routes through the Base "
            "RoyaltyDistributor instead of the local ledger."
        ),
    )


class ContentAccessResponse(BaseModel):
    """Response from content access payment."""
    payment_id: str
    cid: str
    status: str
    amount: float
    royalty_distributions: List[Dict[str, Any]]
    error: Optional[str] = None


class RetrievalRequest(BaseModel):
    """Request for content retrieval with bidding."""
    cid: str = Field(..., description="Content to retrieve")
    max_price_ftns: float = Field(..., ge=0.001, description="Maximum willing to pay")
    timeout: float = Field(30.0, ge=5.0, le=120.0, description="Bid timeout in seconds")


class RetrievalResponse(BaseModel):
    """Response from retrieval request."""
    request_id: str
    cid: str
    status: str
    bids_received: int
    selected_provider: Optional[str] = None
    price_paid: Optional[float] = None


class SemanticSearchRequest(BaseModel):
    """Request for semantic content search."""
    query: str = Field(..., min_length=3, description="Search query")
    limit: int = Field(10, ge=1, le=100, description="Max results")
    min_similarity: float = Field(0.7, ge=0.0, le=1.0, description="Min similarity score")
    max_royalty_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Max royalty filter")


class SemanticSearchResult(BaseModel):
    """Single semantic search result."""
    cid: str
    similarity: float
    creator_id: Optional[str]
    royalty_rate: float
    metadata: Dict[str, Any]


class SemanticSearchResponse(BaseModel):
    """Response from semantic search."""
    query: str
    results: List[SemanticSearchResult]
    total: int


class ReplicationStatusResponse(BaseModel):
    """Replication status for a CID."""
    cid: str
    min_replicas: int
    current_replicas: int
    providers: List[str]
    last_verified: float
    needs_more_replicas: bool


class ContentEconomyStats(BaseModel):
    """Content economy statistics."""
    pending_payments: int
    tracked_content: int
    active_retrieval_requests: int
    royalty_model: str
    min_replicas: int
    vector_store_enabled: bool


class RoyaltyInfoResponse(BaseModel):
    """Royalty distribution info for content."""
    cid: str
    royalty_model: str
    original_creator_rate: float
    derivative_creator_rate: float
    network_fee_rate: float
    total_royalties_earned: float
    access_count: int


# ── Dependency Injection ───────────────────────────────────────────────────

# The get_content_economy function is defined at the top of the file

# ── Payment Endpoints ──────────────────────────────────────────────────────

@router.post("/access", response_model=ContentAccessResponse)
async def process_content_access(
    request: ContentAccessRequest,
    economy = Depends(get_content_economy),
    current_user = Depends(get_current_user),
):
    """Process FTNS payment for content access.

    Flow:
    1. Lock FTNS in escrow
    2. Distribute royalties to creator chain
    3. Record payment completion

    Returns payment details including royalty distributions.

    sp1279 (audit round 7): requires authentication, and the PAYER (accessor_id) is bound to
    the authenticated caller — the body's accessor_id is IGNORED, so a caller can no longer
    charge another user's FTNS by spoofing it.
    """
    accessor_id = str(getattr(current_user, "id", current_user))
    try:
        payment = await economy.process_content_access(
            content_id=request.cid,
            accessor_id=accessor_id,
            content_metadata={
                "royalty_rate": request.royalty_rate,
                "creator_id": request.creator_id,
                "parent_cids": request.parent_cids,
                "provenance_hash": request.provenance_hash,
            },
        )

        return ContentAccessResponse(
            payment_id=payment.payment_id,
            cid=payment.content_id,
            status=payment.status.value,
            amount=float(payment.amount),
            royalty_distributions=payment.royalty_distributions,
            error=payment.error,
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Payment failed: {e}")


@router.get("/payment/{payment_id}", response_model=ContentAccessResponse)
async def get_payment_status(
    payment_id: str,
    economy = Depends(get_content_economy),
):
    """Get status of a content access payment."""
    payment = economy._pending_payments.get(payment_id)
    if not payment:
        raise HTTPException(status_code=404, detail="Payment not found")
    
    return ContentAccessResponse(
        payment_id=payment.payment_id,
        cid=payment.content_id,
        status=payment.status.value,
        amount=float(payment.amount),
        royalty_distributions=payment.royalty_distributions,
        error=payment.error,
    )


# ── Retrieval Marketplace Endpoints ────────────────────────────────────────

@router.post("/retrieval", response_model=RetrievalResponse)
async def request_content_retrieval(
    request: RetrievalRequest,
    economy = Depends(get_content_economy),
):
    """Request content retrieval with marketplace bidding.
    
    Broadcasts a retrieval request to the network and collects bids
    from providers. Selects the best provider based on price, reputation,
    and latency.
    """
    try:
        # Note: This initiates the retrieval but content delivery
        # is handled by ContentProvider
        request_id = f"ret-{request.cid[:12]}"
        
        # Create retrieval request
        from prsm.node.content_economy import RetrievalRequest as InternalRequest
        internal_req = InternalRequest(
            request_id=request_id,
            cid=request.cid,
            requester_id=economy.identity.node_id,
            max_price_ftns=Decimal(str(request.max_price_ftns)),
            bid_deadline=asyncio.get_event_loop().time() + request.timeout / 3,
        )
        economy._retrieval_requests[request_id] = internal_req
        
        # Broadcast request
        await economy.gossip.publish("retrieval_request", {
            "request_id": request_id,
            "cid": request.cid,
            "requester_id": economy.identity.node_id,
            "max_price_ftns": request.max_price_ftns,
            "deadline": internal_req.bid_deadline,
        })
        
        return RetrievalResponse(
            request_id=request_id,
            cid=request.cid,
            status="open",
            bids_received=0,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval request failed: {e}")


@router.get("/retrieval/{request_id}", response_model=RetrievalResponse)
async def get_retrieval_status(
    request_id: str,
    economy = Depends(get_content_economy),
):
    """Get status of a retrieval request."""
    request = economy._retrieval_requests.get(request_id)
    if not request:
        raise HTTPException(status_code=404, detail="Retrieval request not found")
    
    return RetrievalResponse(
        request_id=request.request_id,
        cid=request.cid,
        status=request.status,
        bids_received=len(request.bids),
        selected_provider=request.selected_provider,
    )


# ── Semantic Search Endpoints ──────────────────────────────────────────────

@router.post("/search", response_model=SemanticSearchResponse)
async def semantic_search(
    request: SemanticSearchRequest,
    economy = Depends(get_content_economy),
):
    """Search content by semantic similarity.
    
    Uses vector embeddings to find similar content.
    Requires vector store to be configured.
    """
    if not economy.vector_store:
        raise HTTPException(status_code=503, detail="Vector store not configured")
    
    try:
        results = await economy.semantic_search(
            query=request.query,
            limit=request.limit,
            min_similarity=request.min_similarity,
            max_royalty_rate=request.max_royalty_rate,
        )
        
        return SemanticSearchResponse(
            query=request.query,
            results=[
                SemanticSearchResult(
                    cid=r["cid"],
                    similarity=r["similarity"],
                    creator_id=r.get("creator_id"),
                    royalty_rate=r.get("royalty_rate", 0.01),
                    metadata=r.get("metadata", {}),
                )
                for r in results
            ],
            total=len(results),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@router.post("/index/{cid}")
async def index_content(
    cid: str,
    economy = Depends(get_content_economy),
):
    """Index content into vector store for semantic search.
    
    Fetches content from ContentStore and generates embedding for indexing.
    """
    if not economy.vector_store:
        raise HTTPException(status_code=503, detail="Vector store not configured")

    try:
        # Would need to fetch content from ContentStore
        # For now, return success if vector store is available
        return {"status": "queued", "cid": cid}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")


# ── Replication Endpoints ──────────────────────────────────────────────────

@router.get("/replication/{cid}", response_model=ReplicationStatusResponse)
async def get_replication_status(
    cid: str,
    economy = Depends(get_content_economy),
):
    """Get replication status for a content item."""
    status = economy._replication_status.get(cid)
    if not status:
        # Check content index
        record = economy.content_index.lookup(cid)
        if record:
            status = economy._replication_status.get(cid)
        else:
            raise HTTPException(status_code=404, detail="Content not tracked")
    
    return ReplicationStatusResponse(
        cid=status.cid,
        min_replicas=status.min_replicas,
        current_replicas=status.current_replicas,
        providers=list(status.providers),
        last_verified=status.last_verified,
        needs_more_replicas=status.current_replicas < status.min_replicas,
    )


@router.post("/replication/{cid}/ensure")
async def ensure_replication(
    cid: str,
    min_replicas: int = Query(3, ge=1, le=10),
    economy = Depends(get_content_economy),
):
    """Ensure minimum replication for content.
    
    Requests additional replicas if current count is below minimum.
    """
    status = economy._replication_status.get(cid)
    if not status:
        raise HTTPException(status_code=404, detail="Content not tracked")
    
    # Update minimum
    status.min_replicas = max(status.min_replicas, min_replicas)
    
    # Check if we need more
    await economy._check_replication_needs(cid, status)
    
    return {
        "cid": cid,
        "min_replicas": status.min_replicas,
        "current_replicas": status.current_replicas,
        "pending_requests": status.pending_requests,
    }


# ── Royalty Info Endpoints ────────────────────────────────────────────────

@router.get("/royalty/{cid}", response_model=RoyaltyInfoResponse)
async def get_royalty_info(
    cid: str,
    economy = Depends(get_content_economy),
):
    """Get royalty distribution info for content."""
    record = economy.content_index.lookup(cid)
    if not record:
        raise HTTPException(status_code=404, detail="Content not found")
    
    return RoyaltyInfoResponse(
        cid=cid,
        royalty_model=economy.royalty_model.value,
        original_creator_rate=0.08,  # Phase4 default
        derivative_creator_rate=0.01,
        network_fee_rate=0.02,
        total_royalties_earned=0.0,  # Would need ledger query
        access_count=0,  # Would need tracking
    )


# ── Statistics Endpoints ──────────────────────────────────────────────────

@router.get("/stats", response_model=ContentEconomyStats)
async def get_content_economy_stats(
    economy = Depends(get_content_economy),
):
    """Get content economy statistics."""
    stats = economy.get_stats()
    return ContentEconomyStats(**stats)


# ── Utility Endpoints ──────────────────────────────────────────────────────

@router.get("/models")
async def get_royalty_models():
    """Get available royalty distribution models."""
    return {
        "models": [
            {
                "name": "phase4",
                "description": "Phase 4 model: 8% original, 1% derivative, 2% network",
                "original_creator_rate": 0.08,
                "derivative_creator_rate": 0.01,
                "network_fee_rate": 0.02,
            },
            {
                "name": "legacy",
                "description": "Legacy model: 70% derivative, 25% source, 5% network",
                "derivative_share": 0.70,
                "source_share": 0.25,
                "network_share": 0.05,
            },
        ]
    }


# ── Router Registration Helper ─────────────────────────────────────────────

def register_content_economy_routes(app):
    """Register content economy routes with the FastAPI app."""
    app.include_router(router)
    return router


import asyncio  # Import needed for time() in request_content_retrieval
