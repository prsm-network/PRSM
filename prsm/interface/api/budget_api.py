"""
PRSM FTNS Budget Management API

🎯 BUDGET CONTROL ENDPOINTS:
- Create session budgets with predictive cost estimation
- Real-time budget monitoring and spending analytics
- Budget expansion requests and authorization workflows
- Multi-resource budget tracking (models, agents, tools, datasets)
- Emergency controls and spending circuit breakers

This API provides comprehensive budget management for PRSM sessions,
enabling users to control FTNS spending with confidence and transparency.
"""

from decimal import Decimal
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field, field_validator

from ..auth import get_current_user
from prsm.core.models import UserInput, PRSMSession
from prsm.economy.tokenomics.ftns_budget_manager import (
    FTNSBudgetManager, SpendingCategory, get_ftns_budget_manager
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1/budget", tags=["Budget Management"])


# Request/Response Models

class CreateBudgetRequest(BaseModel):
    """Request to create a session budget"""
    session_id: Optional[UUID] = None
    prompt: str = Field(min_length=1)
    total_budget: Optional[Decimal] = Field(None, gt=0)
    auto_expand_enabled: bool = Field(default=True)
    max_auto_expand: Optional[Decimal] = Field(None, ge=0)
    expansion_increment: Optional[Decimal] = Field(None, gt=0)
    category_allocations: Optional[Dict[str, Dict[str, Any]]] = None
    alerts: Optional[List[Dict[str, Any]]] = None
    
    @field_validator('total_budget', 'max_auto_expand', 'expansion_increment', mode='before')
    @classmethod
    def convert_to_decimal(cls, v):
        if v is not None:
            return Decimal(str(v))
        return v


class BudgetResponse(BaseModel):
    """Budget information response"""
    budget_id: str
    session_id: str
    status: str
    total_budget: float
    total_spent: float
    available_budget: float
    utilization_percentage: float
    category_breakdown: Dict[str, Dict[str, float]]
    created_at: str
    updated_at: str


class PredictionResponse(BaseModel):
    """Cost prediction response"""
    prediction_id: str
    query_complexity: float
    estimated_total_cost: float
    category_estimates: Dict[str, float]
    confidence_score: float
    recommended_budget: float
    prediction_factors: Dict[str, Any]


class SpendingRequest(BaseModel):
    """Request to spend from budget"""
    amount: Decimal = Field(gt=0)
    category: SpendingCategory
    description: str = Field(default="")
    
    @field_validator('amount', mode='before')
    @classmethod
    def convert_to_decimal(cls, v):
        return Decimal(str(v))


class BudgetExpansionRequest(BaseModel):
    """Request for budget expansion"""
    requested_amount: Decimal = Field(gt=0)
    reason: str
    cost_breakdown: Optional[Dict[str, Decimal]] = None
    priority_level: str = Field(default="normal")
    
    @field_validator('requested_amount', mode='before')
    @classmethod
    def convert_to_decimal(cls, v):
        return Decimal(str(v))

    @field_validator('cost_breakdown', mode='before')
    @classmethod
    def convert_breakdown_to_decimal(cls, v):
        if v is not None:
            return {k: Decimal(str(val)) for k, val in v.items()}
        return v


class ExpansionApprovalRequest(BaseModel):
    """Approval for budget expansion"""
    approved: bool
    approved_amount: Optional[Decimal] = None
    reason: str = Field(default="")
    
    @field_validator('approved_amount', mode='before')
    @classmethod
    def convert_to_decimal(cls, v):
        if v is not None:
            return Decimal(str(v))
        return v


# Budget Management Endpoints

@router.post("/predict-cost", response_model=PredictionResponse)
async def predict_session_cost(
    prompt: str = Query(..., min_length=1),
    user_preferences: Optional[Dict[str, Any]] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Predict the FTNS cost for a session before execution
    
    This endpoint provides cost estimation to help users set appropriate
    budgets before starting resource-intensive queries.
    """
    try:
        budget_manager = get_ftns_budget_manager()
        
        # Create mock user input for prediction
        user_input = UserInput(
            user_id=current_user["user_id"],
            prompt=prompt,
            preferences=user_preferences or {}
        )
        
        # Create mock session for prediction
        session = PRSMSession(user_id=current_user["user_id"])
        
        # Generate prediction
        prediction = await budget_manager.predict_session_cost(user_input, session)
        
        # Convert to response format
        category_estimates = {
            category.value: float(amount) 
            for category, amount in prediction.category_estimates.items()
        }
        
        response = PredictionResponse(
            prediction_id=str(prediction.prediction_id),
            query_complexity=prediction.query_complexity,
            estimated_total_cost=float(prediction.estimated_total_cost),
            category_estimates=category_estimates,
            confidence_score=prediction.confidence_score,
            recommended_budget=float(prediction.get_recommended_budget()),
            prediction_factors=prediction.prediction_factors
        )
        
        logger.info("Cost prediction generated",
                   user_id=current_user["user_id"],
                   estimated_cost=float(prediction.estimated_total_cost),
                   confidence=prediction.confidence_score)
        
        return response
        
    except Exception as e:
        logger.error("Cost prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Cost prediction failed: {str(e)}")


@router.post("/create", response_model=BudgetResponse)
async def create_session_budget(
    request: CreateBudgetRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Create a comprehensive budget for a PRSM session
    
    This creates a budget with predictive allocations and monitoring
    capabilities for transparent FTNS spending control.
    """
    try:
        budget_manager = get_ftns_budget_manager()
        
        # Create user input and session
        user_input = UserInput(
            user_id=current_user["user_id"],
            prompt=request.prompt
        )
        
        session = PRSMSession(
            session_id=request.session_id or uuid4(),
            user_id=current_user["user_id"]
        )
        
        # Build budget configuration
        budget_config = {
            "auto_expand": request.auto_expand_enabled,
            "max_auto_expand": request.max_auto_expand,
            "expansion_increment": request.expansion_increment,
            "category_allocations": request.category_allocations,
            "alerts": request.alerts
        }
        
        if request.total_budget:
            budget_config["total_budget"] = request.total_budget
        
        # Create budget
        budget = await budget_manager.create_session_budget(
            session, user_input, budget_config
        )
        
        # Get budget status for response
        budget_status = await budget_manager.get_budget_status(budget.budget_id)
        
        if not budget_status:
            raise HTTPException(status_code=500, detail="Failed to retrieve budget status")
        
        response = BudgetResponse(
            budget_id=budget_status["budget_id"],
            session_id=budget_status["session_id"],
            status=budget_status["status"],
            total_budget=budget_status["total_budget"],
            total_spent=budget_status["total_spent"],
            available_budget=budget_status["available_budget"],
            utilization_percentage=budget_status["utilization_percentage"],
            category_breakdown=budget_status["category_breakdown"],
            created_at=budget_status["created_at"],
            updated_at=budget_status["updated_at"]
        )
        
        logger.info("Session budget created",
                   user_id=current_user["user_id"],
                   budget_id=str(budget.budget_id),
                   total_budget=float(budget.total_budget))
        
        return response
        
    except Exception as e:
        logger.error("Budget creation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Budget creation failed: {str(e)}")


async def _assert_budget_owner(budget_manager, budget_id, current_user) -> dict:
    """sp1282 (audit round 7) — IDOR guard: a budget may only be read/spent/expanded by the
    authenticated user who owns it. Returns the budget status (so callers can reuse it). 404
    when the budget doesn't exist; 403 when it belongs to another user."""
    status = await budget_manager.get_budget_status(budget_id)
    if not status:
        raise HTTPException(status_code=404, detail="Budget not found")
    if str(status.get("user_id")) != str((current_user or {}).get("user_id")):
        raise HTTPException(status_code=403, detail="Not authorized for this budget")
    return status


@router.get("/status/{budget_id}", response_model=BudgetResponse)
async def get_budget_status(
    budget_id: UUID,
    current_user: dict = Depends(get_current_user)
):
    """
    Get comprehensive budget status and analytics
    
    Provides real-time budget utilization, spending breakdown,
    and predictive analytics for budget management.
    """
    try:
        budget_manager = get_ftns_budget_manager()
        
        # sp1282 — enforce ownership (was a TODO comment → any authed user could read any
        # budget_id, an IDOR). Returns the status for reuse.
        budget_status = await _assert_budget_owner(budget_manager, budget_id, current_user)

        response = BudgetResponse(
            budget_id=budget_status["budget_id"],
            session_id=budget_status["session_id"],
            status=budget_status["status"],
            total_budget=budget_status["total_budget"],
            total_spent=budget_status["total_spent"],
            available_budget=budget_status["available_budget"],
            utilization_percentage=budget_status["utilization_percentage"],
            category_breakdown=budget_status["category_breakdown"],
            created_at=budget_status["created_at"],
            updated_at=budget_status["updated_at"]
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get budget status failed", budget_id=budget_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get budget status: {str(e)}")


@router.post("/spend/{budget_id}")
async def spend_from_budget(
    budget_id: UUID,
    request: SpendingRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Record spending against a budget
    
    This endpoint is used by the PRSM system to track spending
    as resources are consumed during session execution.
    """
    try:
        budget_manager = get_ftns_budget_manager()

        # sp1282 — IDOR guard: only the budget's owner may spend from it.
        await _assert_budget_owner(budget_manager, budget_id, current_user)

        # Spend from budget
        success = await budget_manager.spend_budget_amount(
            budget_id=budget_id,
            amount=request.amount,
            category=request.category,
            description=request.description
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Spending failed - insufficient budget or budget inactive")
        
        # Get updated status
        budget_status = await budget_manager.get_budget_status(budget_id)
        
        logger.info("Budget spending recorded",
                   budget_id=budget_id,
                   amount=float(request.amount),
                   category=request.category.value)
        
        return {
            "success": True,
            "amount_spent": float(request.amount),
            "category": request.category.value,
            "remaining_budget": budget_status["available_budget"] if budget_status else 0,
            "utilization": budget_status["utilization_percentage"] if budget_status else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Budget spending failed", budget_id=budget_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Budget spending failed: {str(e)}")


@router.post("/expand/{budget_id}")
async def request_budget_expansion(
    budget_id: UUID,
    request: BudgetExpansionRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Request expansion of a budget
    
    This creates a formal request for additional budget allocation
    that can be auto-approved or require manual authorization.
    """
    try:
        budget_manager = get_ftns_budget_manager()

        # sp1282 — IDOR guard: only the budget's owner may request expansion.
        await _assert_budget_owner(budget_manager, budget_id, current_user)

        # Create expansion request
        expand_request = await budget_manager.request_budget_expansion(
            budget_id=budget_id,
            requested_amount=request.requested_amount,
            reason=request.reason,
            cost_breakdown=request.cost_breakdown
        )
        
        response_data = {
            "request_id": str(expand_request.request_id),
            "budget_id": str(expand_request.budget_id),
            "requested_amount": float(expand_request.requested_amount),
            "current_utilization": expand_request.current_utilization,
            "remaining_budget": float(expand_request.remaining_budget),
            "auto_generated": expand_request.auto_generated,
            "approved": expand_request.approved,
            "expires_at": expand_request.expires_at.isoformat()
        }
        
        if expand_request.approved is not None:
            response_data.update({
                "approved_amount": float(expand_request.approved_amount) if expand_request.approved_amount else 0,
                "response_reason": expand_request.response_reason,
                "auto_approved": expand_request.auto_generated
            })
        
        logger.info("Budget expansion requested",
                   budget_id=budget_id,
                   request_id=str(expand_request.request_id),
                   amount=float(request.requested_amount),
                   auto_approved=expand_request.approved)
        
        return response_data
        
    except Exception as e:
        logger.error("Budget expansion request failed", budget_id=budget_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Budget expansion request failed: {str(e)}")


@router.post("/approve-expansion/{request_id}")
async def approve_budget_expansion(
    request_id: UUID,
    approval: ExpansionApprovalRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Approve or deny a budget expansion request
    
    This endpoint allows users to authorize additional budget
    allocations for ongoing sessions.
    """
    try:
        budget_manager = get_ftns_budget_manager()
        
        # Process approval
        success = await budget_manager.approve_budget_expansion(
            request_id=request_id,
            approved=approval.approved,
            approved_amount=approval.approved_amount,
            reason=approval.reason
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Expansion request not found")
        
        logger.info("Budget expansion processed",
                   request_id=request_id,
                   approved=approval.approved,
                   amount=float(approval.approved_amount) if approval.approved_amount else 0)
        
        return {
            "success": True,
            "request_id": str(request_id),
            "approved": approval.approved,
            "approved_amount": float(approval.approved_amount) if approval.approved_amount else 0,
            "reason": approval.reason
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Budget expansion approval failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Budget expansion approval failed: {str(e)}")


@router.get("/user-budgets")
async def get_user_budgets(
    status: Optional[str] = Query(None, description="Filter by budget status"),
    limit: int = Query(20, ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    """
    Get all budgets for the current user
    
    Provides a comprehensive view of user's budget history
    and current active budgets.
    """
    try:
        budget_manager = get_ftns_budget_manager()
        
        # Get user's budgets (simplified - in production would query database)
        user_budgets = []
        for budget in budget_manager.active_budgets.values():
            if budget.user_id == current_user["user_id"]:
                if status is None or budget.status.value == status:
                    budget_status = await budget_manager.get_budget_status(budget.budget_id)
                    if budget_status:
                        user_budgets.append(budget_status)
        
        # Add budget history
        for budget in budget_manager.budget_history.values():
            if budget.user_id == current_user["user_id"]:
                if status is None or budget.status.value == status:
                    budget_status = await budget_manager.get_budget_status(budget.budget_id)
                    if budget_status:
                        user_budgets.append(budget_status)
        
        # Sort by created date (most recent first)
        user_budgets.sort(key=lambda x: x["created_at"], reverse=True)
        
        # Apply limit
        user_budgets = user_budgets[:limit]
        
        return {
            "budgets": user_budgets,
            "total_count": len(user_budgets),
            "user_id": current_user["user_id"]
        }
        
    except Exception as e:
        logger.error("Get user budgets failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get user budgets: {str(e)}")


@router.get("/analytics/{budget_id}")
async def get_budget_analytics(
    budget_id: UUID,
    current_user: dict = Depends(get_current_user)
):
    """
    Get detailed budget analytics and insights
    
    Provides comprehensive analytics including spending patterns,
    category breakdowns, and predictive insights.
    """
    try:
        budget_manager = get_ftns_budget_manager()
        
        # Get budget status
        budget_status = await budget_manager.get_budget_status(budget_id)
        
        if not budget_status:
            raise HTTPException(status_code=404, detail="Budget not found")
        
        # Get budget object for detailed analytics
        budget = budget_manager.active_budgets.get(budget_id) or budget_manager.budget_history.get(budget_id)
        
        if not budget:
            raise HTTPException(status_code=404, detail="Budget data not found")
        
        # Calculate advanced analytics
        spending_history = budget.spending_history
        recent_spending = [entry for entry in spending_history[-10:] if entry["action"] == "spend"]
        
        analytics = {
            "budget_overview": budget_status,
            "spending_analytics": {
                "total_transactions": len([entry for entry in spending_history if entry["action"] == "spend"]),
                "avg_transaction_size": sum(entry["amount"] for entry in recent_spending) / max(len(recent_spending), 1),
                "spending_velocity": sum(entry["amount"] for entry in recent_spending) / max(len(recent_spending), 1),
                "largest_transaction": max([entry["amount"] for entry in recent_spending], default=0),
                "spending_trend": "increasing" if len(recent_spending) > 5 else "stable"
            },
            "category_analysis": {
                category: {
                    "total_spent": sum(entry["amount"] for entry in spending_history 
                                     if entry.get("category") == category),
                    "transaction_count": len([entry for entry in spending_history 
                                            if entry.get("category") == category and entry["action"] == "spend"]),
                    "avg_per_transaction": 0  # Calculate separately
                }
                for category in set(entry.get("category", "unknown") for entry in spending_history)
            },
            "budget_health": {
                "utilization_status": "healthy" if budget_status["utilization_percentage"] < 75 else "warning",
                "projected_completion": "within_budget" if budget_status["utilization_percentage"] < 90 else "may_exceed",
                "efficiency_score": min(100, (1 - budget_status["utilization_percentage"] / 100) * 100),
                "risk_level": "low" if budget_status["utilization_percentage"] < 75 else "medium"
            },
            "alerts_and_notifications": {
                "triggered_alerts": budget_status.get("triggered_alerts", []),
                "pending_expansions": budget_status.get("pending_expansions", 0),
                "recommendations": []
            }
        }
        
        # Add recommendations based on analytics
        if budget_status["utilization_percentage"] > 80:
            analytics["alerts_and_notifications"]["recommendations"].append(
                "Consider requesting budget expansion or optimizing resource usage"
            )
        
        if analytics["spending_analytics"]["spending_velocity"] > budget_status["available_budget"] / 2:
            analytics["alerts_and_notifications"]["recommendations"].append(
                "High spending velocity detected - monitor closely"
            )
        
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get budget analytics failed", budget_id=budget_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get budget analytics: {str(e)}")


@router.get("/spending-categories")
async def get_spending_categories():
    """
    Get all available spending categories
    
    Returns the list of spending categories for budget allocation
    and tracking purposes.
    """
    try:
        categories = {
            category.value: {
                "name": category.value.replace("_", " ").title(),
                "description": _get_category_description(category)
            }
            for category in SpendingCategory
        }
        
        return {
            "categories": categories,
            "total_count": len(categories)
        }
        
    except Exception as e:
        logger.error("Get spending categories failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get spending categories: {str(e)}")


def _get_category_description(category: SpendingCategory) -> str:
    """Get description for spending category"""
    descriptions = {
        SpendingCategory.MODEL_INFERENCE: "Costs for LLM API calls and model inference",
        SpendingCategory.AGENT_COORDINATION: "Costs for agent orchestration and coordination",
        SpendingCategory.AGENT_EXECUTION: "Costs for agent and MCP tool execution",
        SpendingCategory.CONTEXT_PROCESSING: "Costs for context compression and management",
        SpendingCategory.REASONING_OPERATION: "Costs for complex reasoning operations",
        SpendingCategory.BREAKTHROUGH_DETECTION: "Costs for breakthrough analysis and detection",
        SpendingCategory.SYNTHESIS_GENERATION: "Costs for content synthesis and generation",
        SpendingCategory.DATABASE_OPERATION: "Costs for database queries and operations",
        SpendingCategory.SYSTEM_OVERHEAD: "General PRSM system operational costs",
    }

    return descriptions.get(category, "General PRSM system costs")


# Global budget manager for dependency injection
def get_budget_manager() -> FTNSBudgetManager:
    """Get FTNS budget manager instance"""
    return get_ftns_budget_manager()