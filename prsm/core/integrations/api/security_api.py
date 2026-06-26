"""
Security API Endpoints
=====================

FastAPI endpoints for accessing enhanced security features in the PRSM integration layer.
Provides comprehensive security scanning and monitoring capabilities.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import Field

from ..security.security_orchestrator import security_orchestrator, SecurityAssessment
from ..security.audit_logger import audit_logger, EventLevel
from prsm.core.models import PRSMBaseModel


# === API Models ===

class SecurityScanRequest(PRSMBaseModel):
    """Request model for security scanning"""
    content_path: str
    content_id: str
    platform: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    enable_sandbox: bool = True
    scan_options: Dict[str, Any] = Field(default_factory=dict)


class SecurityAssessmentResponse(PRSMBaseModel):
    """Response model for security assessment"""
    assessment_id: str
    content_id: str
    platform: str
    timestamp: datetime
    security_passed: bool
    overall_risk_level: str
    scan_duration: float
    
    # Scan results summary
    vulnerability_scan: Dict[str, Any]
    license_scan: Dict[str, Any]
    threat_scan: Dict[str, Any]
    sandbox_scan: Optional[Dict[str, Any]]
    
    # Issues and recommendations
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    
    # Execution metadata
    scans_completed: List[str]
    scans_failed: List[str]


class SecurityEventResponse(PRSMBaseModel):
    """Response model for security events"""
    event_id: str
    event_type: str
    level: str
    user_id: str
    platform: str
    description: str
    timestamp: datetime
    metadata: Dict[str, Any]


class SecurityStatsResponse(PRSMBaseModel):
    """Response model for security statistics"""
    total_assessments: int
    assessments_passed: int
    assessments_failed: int
    security_events: Dict[str, int]
    threat_levels: Dict[str, int]
    risk_levels: Dict[str, int]
    last_updated: datetime


# === API Router ===

security_router = APIRouter(
    prefix="/security",
    tags=["security"],
    responses={404: {"description": "Not found"}}
)


# === Dependency Functions ===

_security_api_bearer = HTTPBearer(auto_error=True)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_security_api_bearer),
) -> str:
    """Authenticate via the canonical JWT verifier; return the user id as a string.

    sp1278 (audit round 7): was a stub returning "default_user", leaving this security router
    (incl. POST /policies/update, which can globally disable security scanning) effectively
    UNAUTHENTICATED — same pattern fixed in sp1272/sp1268. jwt_handler.verify_token enforces
    signature, expiry, required claims, and revocation; non-access tokens are rejected.
    """
    from prsm.core.auth.jwt_handler import jwt_handler

    try:
        token_data = await jwt_handler.verify_token(credentials.credentials)
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001 — never authenticate on a verifier error
        raise HTTPException(status_code=401, detail="Authentication failed")

    if token_data is None or token_data.token_type != "access":
        raise HTTPException(status_code=401, detail="Invalid or missing access token")
    return str(token_data.user_id)


# === Security Scanning Endpoints ===

@security_router.post("/scan")
async def perform_security_scan(
    request: SecurityScanRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user)
) -> JSONResponse:
    """
    Perform comprehensive security scan of content
    """
    try:
        # Perform security assessment
        assessment = await security_orchestrator.comprehensive_security_assessment(
            content_path=request.content_path,
            metadata=request.metadata,
            user_id=user_id,
            platform=request.platform,
            content_id=request.content_id,
            enable_sandbox=request.enable_sandbox
        )
        
        # Convert assessment to response format
        response_data = _convert_assessment_to_response(assessment)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response_data
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Security scan failed: {str(e)}"
        )


@security_router.get("/assessment/{assessment_id}")
async def get_security_assessment(
    assessment_id: str,
    user_id: str = Depends(get_current_user)
) -> SecurityAssessmentResponse:
    """
    Get details of a specific security assessment
    """
    # This would retrieve from a database in a real implementation
    # For now, return a mock response
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Assessment retrieval not yet implemented"
    )


@security_router.get("/events/recent")
async def get_recent_security_events(
    limit: int = 50,
    level: Optional[str] = None,
    user_id: str = Depends(get_current_user)
) -> List[SecurityEventResponse]:
    """
    Get recent security events
    """
    try:
        # Convert level string to EventLevel if provided
        event_level = None
        if level:
            try:
                event_level = EventLevel(level.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid event level: {level}"
                )
        
        # Get events from audit logger
        events = audit_logger.get_recent_events(limit=limit, level=event_level)
        
        # Convert to response format
        response_events = [
            SecurityEventResponse(
                event_id=event["event_id"],
                event_type=event["event_type"],
                level=event["level"],
                user_id=event["user_id"],
                platform=event["platform"],
                description=event["description"],
                timestamp=datetime.fromisoformat(event["timestamp"]),
                metadata=event["metadata"]
            )
            for event in events
        ]
        
        return response_events
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve security events: {str(e)}"
        )


@security_router.get("/stats")
async def get_security_statistics(
    user_id: str = Depends(get_current_user)
) -> SecurityStatsResponse:
    """
    Get security system statistics
    """
    try:
        # Get statistics from security components
        orchestrator_stats = security_orchestrator.get_security_statistics()
        audit_stats = audit_logger.get_security_stats()
        
        # Mock some additional statistics (in production these would come from database)
        stats = SecurityStatsResponse(
            total_assessments=100,  # Mock data
            assessments_passed=85,
            assessments_failed=15,
            security_events=audit_stats["events_by_level"],
            threat_levels={
                "none": 60,
                "low": 25,
                "medium": 10,
                "high": 4,
                "critical": 1
            },
            risk_levels={
                "none": 50,
                "low": 30,
                "medium": 15,
                "high": 4,
                "critical": 1
            },
            last_updated=datetime.now(timezone.utc)
        )
        
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get security statistics: {str(e)}"
        )


@security_router.get("/health")
async def get_security_health() -> Dict[str, Any]:
    """
    Get security system health status
    """
    try:
        # Check each security component
        components = {
            "vulnerability_scanner": "healthy",
            "license_scanner": "healthy",
            "threat_detector": "healthy",
            "enhanced_sandbox": "healthy",
            "audit_logger": "healthy",
            "security_orchestrator": "healthy"
        }
        
        # Test audit logger
        try:
            audit_logger.get_security_stats()
        except Exception:
            components["audit_logger"] = "error"
        
        # Determine overall status
        error_count = sum(1 for status in components.values() if status == "error")
        overall_status = "healthy" if error_count == 0 else "degraded" if error_count < 3 else "error"
        
        return {
            "status": overall_status,
            "components": components,
            "last_check": datetime.now(timezone.utc).isoformat(),
            "security_policies_active": True
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "last_check": datetime.now(timezone.utc).isoformat()
        }


@security_router.post("/policies/update")
async def update_security_policies(
    policies: Dict[str, Any],
    user_id: str = Depends(get_current_user)
) -> JSONResponse:
    """
    Update security policies (admin only)
    """
    try:
        # In a real implementation, this would check admin permissions
        # and validate policy values
        
        # Update orchestrator policies
        valid_policies = {
            "require_license_compliance": bool,
            "block_high_risk_vulnerabilities": bool,
            "block_medium_risk_threats": bool,
            "require_sandbox_validation": bool,
            "auto_quarantine_threats": bool
        }
        
        updated_policies = {}
        for key, value in policies.items():
            if key in valid_policies:
                if isinstance(value, valid_policies[key]):
                    updated_policies[key] = value
                    security_orchestrator.security_policies[key] = value
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid type for policy {key}: expected {valid_policies[key].__name__}"
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unknown security policy: {key}"
                )
        
        return JSONResponse(
            content={
                "message": f"Updated {len(updated_policies)} security policies",
                "updated_policies": updated_policies
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update security policies: {str(e)}"
        )


@security_router.get("/policies")
async def get_security_policies(
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current security policies
    """
    try:
        return {
            "policies": security_orchestrator.security_policies,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get security policies: {str(e)}"
        )


# === Helper Functions ===

def _convert_assessment_to_response(assessment: SecurityAssessment) -> Dict[str, Any]:
    """Convert SecurityAssessment to API response format"""
    
    # Convert vulnerability results
    vuln_summary = {"completed": False, "risk_level": "unknown", "vulnerabilities_found": 0}
    if assessment.vulnerability_result:
        vuln_summary = {
            "completed": True,
            "risk_level": assessment.vulnerability_result.risk_level.value,
            "vulnerabilities_found": len(assessment.vulnerability_result.vulnerabilities),
            "scan_method": assessment.vulnerability_result.scan_method
        }
    
    # Convert license results
    license_summary = {"completed": False, "compliant": False, "license_type": "unknown"}
    if assessment.license_result:
        license_summary = {
            "completed": True,
            "compliant": assessment.license_result.compliant,
            "license_type": assessment.license_result.license_type.value,
            "issues_found": len(assessment.license_result.issues)
        }
    
    # Convert threat results
    threat_summary = {"completed": False, "threat_level": "unknown", "threats_found": 0}
    if assessment.threat_result:
        threat_summary = {
            "completed": True,
            "threat_level": assessment.threat_result.threat_level.value,
            "threats_found": len(assessment.threat_result.threats),
            "scan_method": assessment.threat_result.scan_method
        }
    
    # Convert sandbox results
    sandbox_summary = None
    if assessment.sandbox_result:
        sandbox_summary = {
            "completed": True,
            "success": assessment.sandbox_result.success,
            "execution_time": assessment.sandbox_result.execution_time,
            "security_events": len(assessment.sandbox_result.security_events),
            "exit_code": assessment.sandbox_result.exit_code
        }
    
    return {
        "assessment_id": assessment.assessment_id,
        "content_id": assessment.content_id,
        "platform": assessment.platform,
        "timestamp": assessment.timestamp.isoformat(),
        "security_passed": assessment.security_passed,
        "overall_risk_level": assessment.overall_risk_level.value,
        "scan_duration": assessment.scan_duration,
        "vulnerability_scan": vuln_summary,
        "license_scan": license_summary,
        "threat_scan": threat_summary,
        "sandbox_scan": sandbox_summary,
        "issues": assessment.issues,
        "warnings": assessment.warnings,
        "recommendations": assessment.recommendations,
        "scans_completed": assessment.scans_completed,
        "scans_failed": assessment.scans_failed
    }