"""
API Hardening Module for PRSM Node
===================================

Comprehensive security hardening for the Node Management API including:
- JWT authentication middleware
- Rate limiting with configurable limits
- WebSocket status updates
- OpenAPI specification generation
- Security headers and request validation

This module integrates with existing auth infrastructure in prsm/core/auth/.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

import structlog
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.status import (
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_429_TOO_MANY_REQUESTS,
)

logger = structlog.get_logger(__name__)


# ── Rate Limit Configuration ─────────────────────────────────────────────────────

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: int = 10
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    burst_size: int = 20
    enabled: bool = True
    
    # Endpoint-specific overrides
    endpoint_limits: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "/auth/login": {"requests_per_minute": 10, "requests_per_hour": 50},
        "/auth/register": {"requests_per_minute": 5, "requests_per_hour": 20},
        "/compute/submit": {"requests_per_minute": 30, "requests_per_hour": 200},
        "/content/upload": {"requests_per_minute": 20, "requests_per_hour": 100},
    })


class RateLimitResult(BaseModel):
    """Result of a rate limit check."""
    allowed: bool
    limit: int
    remaining: int
    reset_at: float
    retry_after: Optional[int] = None
    rule_name: Optional[str] = None


class RateLimiter:
    """
    Rate limiter for API endpoints.
    
    Uses a sliding window algorithm with token bucket for burst handling.
    Integrates with Redis for distributed rate limiting when available.
    """
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.redis_client = None
        self._local_store: Dict[str, List[float]] = {}  # Fallback for non-Redis
        self._token_buckets: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self):
        """Initialize rate limiter with Redis connection if available."""
        try:
            from prsm.core.redis_client import get_redis_client
            self.redis_client = get_redis_client()
            logger.info("Rate limiter initialized with Redis backend")
        except Exception as e:
            logger.warning("Redis not available, using in-memory rate limiting", error=str(e))
            self.redis_client = None
    
    def _get_client_key(self, client_id: str, endpoint: str = "default") -> str:
        """Generate rate limit key for client/endpoint combination."""
        return f"rate_limit:{client_id}:{endpoint}"
    
    def _get_endpoint_config(self, endpoint: str) -> Dict[str, int]:
        """Get rate limit config for specific endpoint."""
        # Check for exact match first
        if endpoint in self.config.endpoint_limits:
            return self.config.endpoint_limits[endpoint]
        
        # Check for prefix match
        for pattern, limits in self.config.endpoint_limits.items():
            if endpoint.startswith(pattern.rstrip("*")):
                return limits
        
        # Default limits
        return {
            "requests_per_minute": self.config.requests_per_minute,
            "requests_per_hour": self.config.requests_per_hour,
        }
    
    async def check_rate_limit(
        self,
        client_id: str,
        endpoint: str = "default"
    ) -> RateLimitResult:
        """
        Check if client is within rate limits.
        
        Args:
            client_id: Unique client identifier (IP or user ID)
            endpoint: API endpoint being accessed
            
        Returns:
            RateLimitResult with allowance status and metadata
        """
        if not self.config.enabled:
            return RateLimitResult(
                allowed=True,
                limit=0,
                remaining=0,
                reset_at=time.time() + 60
            )
        
        endpoint_config = self._get_endpoint_config(endpoint)
        requests_per_minute = endpoint_config.get("requests_per_minute", self.config.requests_per_minute)
        
        key = self._get_client_key(client_id, endpoint)
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window
        
        if self.redis_client:
            return await self._check_rate_limit_redis(key, client_id, requests_per_minute)
        else:
            return self._check_rate_limit_local(key, client_id, requests_per_minute, current_time, window_start)
    
    async def _check_rate_limit_redis(
        self,
        key: str,
        client_id: str,
        limit: int
    ) -> RateLimitResult:
        """Check rate limit using Redis backend."""
        try:
            current_time = time.time()
            window_start = current_time - 60
            
            # Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            # Count current entries
            pipe.zcard(key)
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            # Set expiration
            pipe.expire(key, 120)
            
            results = await pipe.execute()
            current_count = results[1]  # Count after cleanup
            
            remaining = max(0, limit - current_count)
            allowed = current_count <= limit
            
            return RateLimitResult(
                allowed=allowed,
                limit=limit,
                remaining=remaining,
                reset_at=current_time + 60,
                retry_after=None if allowed else 60
            )
            
        except Exception as e:
            logger.error("Redis rate limit check failed", error=str(e), client_id=client_id)
            # Fail open
            return RateLimitResult(
                allowed=True,
                limit=limit,
                remaining=limit,
                reset_at=time.time() + 60
            )
    
    def _check_rate_limit_local(
        self,
        key: str,
        client_id: str,
        limit: int,
        current_time: float,
        window_start: float
    ) -> RateLimitResult:
        """Check rate limit using in-memory store."""
        if key not in self._local_store:
            self._local_store[key] = []
        
        # Clean old entries
        self._local_store[key] = [
            ts for ts in self._local_store[key] if ts > window_start
        ]
        
        current_count = len(self._local_store[key])
        remaining = max(0, limit - current_count)
        allowed = current_count < limit
        
        if allowed:
            self._local_store[key].append(current_time)
        
        return RateLimitResult(
            allowed=allowed,
            limit=limit,
            remaining=remaining if allowed else 0,
            reset_at=current_time + 60,
            retry_after=None if allowed else 60
        )
    
    def get_remaining_requests(self, client_id: str, endpoint: str = "default") -> int:
        """Get remaining requests for client."""
        # This is a simplified version - actual remaining is calculated in check_rate_limit
        return self.config.requests_per_minute
    
    def reset_limits(self, client_id: str) -> None:
        """Reset rate limits for client."""
        # Clear all keys for this client
        keys_to_remove = [k for k in self._local_store.keys() if client_id in k]
        for key in keys_to_remove:
            del self._local_store[key]
        
        if self.redis_client:
            # Async reset in Redis would require scan, skip for now
            pass


# ── JWT Authentication Middleware ───────────────────────────────────────────────

class JWTAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware for JWT authentication on protected endpoints.
    
    Features:
    - Token validation using existing JWTHandler
    - Public endpoint bypass
    - Request state injection with user info
    - Security headers
    """
    
    # Endpoints that don't require authentication
    PUBLIC_ENDPOINTS = {
        "/",
        "/health",
        "/status",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/metrics",
        "/auth/login",
        "/auth/register",
        "/auth/refresh",
        # /ws/status is the dashboard's status push channel.
        # The handler itself doesn't check auth (status info is
        # already exposed via /status which is public). The
        # middleware was inadvertently 403'ing the WS upgrade
        # request because /ws/status wasn't in this list. Sprint
        # 137 fix — found via dashboard dogfood.
        "/ws/status",
    }
    
    # Endpoints that allow optional authentication
    OPTIONAL_AUTH_ENDPOINTS = {
        "/peers",
        "/content/search",
    }
    
    def __init__(self, app, jwt_handler=None):
        super().__init__(app)
        self.jwt_handler = jwt_handler
        self.security = HTTPBearer(auto_error=False)
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through JWT authentication."""
        
        # Skip auth for public endpoints
        request_path = request.url.path
        if request_path in self.PUBLIC_ENDPOINTS:
            return await call_next(request)
        
        # Check for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Try to get token from Authorization header
        auth_header = request.headers.get("Authorization")
        token = None
        user_data = None
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            
            # Validate token
            if self.jwt_handler:
                try:
                    user_data = await self._validate_token(token)
                except Exception as e:
                    logger.warning("Token validation failed", error=str(e), path=request_path)
                    return JSONResponse(
                        status_code=HTTP_401_UNAUTHORIZED,
                        content={"detail": "Invalid or expired token"}
                    )
        
        # Check if endpoint requires auth
        if request_path not in self.OPTIONAL_AUTH_ENDPOINTS:
            if not user_data:
                return JSONResponse(
                    status_code=HTTP_401_UNAUTHORIZED,
                    content={"detail": "Authentication required"}
                )
        
        # Inject user data into request state
        if user_data:
            request.state.user = user_data
            request.state.user_id = user_data.get("user_id")
            request.state.user_role = user_data.get("role")
        else:
            request.state.user = None
            request.state.user_id = None
            request.state.user_role = None
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        self._add_security_headers(response)
        
        return response
    
    async def _validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token and return user data."""
        if not self.jwt_handler:
            return None
        
        try:
            # Use the JWT handler's validation
            token_data = await self.jwt_handler.verify_token(token)
            if token_data:
                return {
                    "user_id": str(token_data.user_id),
                    "username": token_data.username,
                    "email": token_data.email,
                    "role": token_data.role,
                    "permissions": token_data.permissions,
                }
        except Exception as e:
            logger.debug("Token validation error", error=str(e))
        
        return None
    
    def _add_security_headers(self, response: Response) -> None:
        """Add security headers to response."""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"


# ── Rate Limiting Middleware ────────────────────────────────────────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting API requests.
    
    Features:
    - Configurable rate limits per endpoint
    - Client identification via IP or auth token
    - Rate limit headers in responses
    - Graceful degradation when Redis unavailable
    """
    
    # Endpoints exempt from rate limiting
    RATE_LIMIT_EXEMPT = {
        "/health",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
    
    def __init__(self, app, rate_limiter: RateLimiter):
        super().__init__(app)
        self.rate_limiter = rate_limiter
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through rate limiting."""
        
        request_path = request.url.path
        
        # Skip rate limiting for exempt endpoints
        if request_path in self.RATE_LIMIT_EXEMPT:
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limit
        result = await self.rate_limiter.check_rate_limit(client_id, request_path)
        
        if not result.allowed:
            logger.warning(
                "Rate limit exceeded",
                client_id=client_id,
                path=request_path,
                limit=result.limit
            )
            
            # Return rate limit error
            response = JSONResponse(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "limit": result.limit,
                    "retry_after": result.retry_after
                }
            )
            self._add_rate_limit_headers(response, result)
            return response
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        self._add_rate_limit_headers(response, result)
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Try authenticated user first
        if hasattr(request.state, "user_id") and request.state.user_id:
            return f"user:{request.state.user_id}"
        
        # Fall back to IP address
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return f"ip:{forwarded_for.split(',')[0].strip()}"
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return f"ip:{real_ip}"
        
        if request.client:
            return f"ip:{request.client.host}"
        
        return "ip:unknown"
    
    def _add_rate_limit_headers(self, response: Response, result: RateLimitResult) -> None:
        """Add rate limit information headers to response."""
        response.headers["X-RateLimit-Limit"] = str(result.limit)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = str(int(result.reset_at))
        
        if result.retry_after:
            response.headers["Retry-After"] = str(result.retry_after)


# ── WebSocket Status Manager ────────────────────────────────────────────────────

class StatusWebSocket:
    """
    WebSocket manager for real-time status updates.
    
    Features:
    - Connection management with heartbeat
    - Broadcast status to all clients
    - Personal status messages
    - Automatic reconnection support
    """
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._client_info: Dict[WebSocket, Dict[str, Any]] = {}
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._status_cache: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None):
        """Accept WebSocket connection."""
        await websocket.accept()
        
        async with self._lock:
            self.active_connections.add(websocket)
            self._client_info[websocket] = {
                "client_id": client_id or str(uuid4()),
                "connected_at": datetime.now(timezone.utc).isoformat(),
                "last_heartbeat": time.time()
            }
        
        logger.info(
            "WebSocket connected",
            client_id=self._client_info[websocket]["client_id"],
            total_connections=len(self.active_connections)
        )
        
        # Send initial status
        await self.send_personal_status(
            websocket,
            {
                "type": "connection_established",
                "client_id": self._client_info[websocket]["client_id"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
        # Send cached status if available
        if self._status_cache:
            await self.send_personal_status(websocket, {
                "type": "status_update",
                "data": self._status_cache,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
    
    def disconnect(self, websocket: WebSocket):
        """Remove disconnected WebSocket."""
        client_id = None
        
        async def _disconnect():
            async with self._lock:
                if websocket in self._client_info:
                    client_id = self._client_info[websocket].get("client_id")
                    del self._client_info[websocket]
                self.active_connections.discard(websocket)
        
        # Run disconnect in event loop
        asyncio.create_task(_disconnect())
        
        logger.info(
            "WebSocket disconnected",
            client_id=client_id,
            total_connections=len(self.active_connections)
        )
    
    async def broadcast_status(self, status: Dict[str, Any]):
        """Broadcast status to all connected clients."""
        if not self.active_connections:
            return
        
        # Cache status for new connections
        self._status_cache = status
        
        message = {
            "type": "status_update",
            "data": status,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.debug("Failed to send to connection", error=str(e))
                disconnected.add(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            await self._safe_disconnect(conn)
    
    async def send_personal_status(self, websocket: WebSocket, status: Dict[str, Any]):
        """Send status to specific client."""
        try:
            await websocket.send_json(status)
        except Exception as e:
            logger.warning("Failed to send personal status", error=str(e))
    
    async def send_personal_message(self, client_id: str, message: Dict[str, Any]):
        """Send message to specific client by ID."""
        for ws, info in self._client_info.items():
            if info.get("client_id") == client_id:
                try:
                    await ws.send_json(message)
                    return True
                except Exception as e:
                    logger.warning("Failed to send personal message", error=str(e))
        return False
    
    async def _safe_disconnect(self, websocket: WebSocket):
        """Safely disconnect a WebSocket."""
        async with self._lock:
            if websocket in self._client_info:
                del self._client_info[websocket]
            self.active_connections.discard(websocket)
    
    async def start_heartbeat(self, interval: int = 30):
        """Start heartbeat task to check connection health."""
        async def heartbeat_loop():
            while True:
                try:
                    await asyncio.sleep(interval)
                    await self._check_connections()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Heartbeat error", error=str(e))
        
        self._heartbeat_task = asyncio.create_task(heartbeat_loop())
    
    async def stop_heartbeat(self):
        """Stop heartbeat task."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
    
    async def _check_connections(self):
        """Check connection health and remove stale connections."""
        current_time = time.time()
        stale_threshold = 120  # 2 minutes
        
        stale = set()
        for ws, info in self._client_info.items():
            last_heartbeat = info.get("last_heartbeat", current_time)
            if current_time - last_heartbeat > stale_threshold:
                stale.add(ws)
        
        for ws in stale:
            await self._safe_disconnect(ws)
    
    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)
    
    def get_client_ids(self) -> List[str]:
        """Get list of connected client IDs."""
        return [info.get("client_id") for info in self._client_info.values()]


# ── OpenAPI Specification Generator ─────────────────────────────────────────────

def generate_openapi_schema(app: FastAPI) -> Dict[str, Any]:
    """
    Generate comprehensive OpenAPI specification for the API.
    
    Args:
        app: FastAPI application instance
        
    Returns:
        OpenAPI schema dictionary
    """
    if app.openapi_schema:
        return app.openapi_schema

    # Read canonical version from package metadata so the
    # custom OpenAPI schema stays in sync with pyproject.toml.
    # Sister to sprints 110-114 + 130 which fixed the same
    # pattern at /api-info, FastAPI constructor, MCP Server,
    # interface API spec, and webhook User-Agent.
    try:
        from importlib.metadata import version as _pkg_version
        _spec_version = _pkg_version("prsm-network")
    except Exception:  # noqa: BLE001
        _spec_version = "unknown"
    openapi_schema = get_openapi(
        title="PRSM Node API",
        version=_spec_version,
        description="""
# Protocol for Research, Storage, and Modeling (PRSM) Node API

This API provides comprehensive management and monitoring capabilities for PRSM network nodes.

## Authentication

Most endpoints require JWT authentication. Include the `Authorization: Bearer <token>` header
with your requests after obtaining a token from `/auth/login`.

## Rate Limiting

API requests are rate limited to ensure fair usage:
- Default: 100 requests per minute per client
- Authentication endpoints: 10 requests per minute
- Compute endpoints: 30 requests per minute

Rate limit headers are included in all responses:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Unix timestamp when the window resets

## WebSocket Status

Connect to `/ws/status` for real-time node status updates.

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Authentication required |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource not found |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Node not initialized |
        """,
        routes=app.routes,
        tags=[
            {
                "name": "status",
                "description": "Node status and health monitoring"
            },
            {
                "name": "peers",
                "description": "Peer discovery and connection management"
            },
            {
                "name": "compute",
                "description": "Compute job submission and management"
            },
            {
                "name": "content",
                "description": "Content upload and search operations"
            },
            {
                "name": "agents",
                "description": "AI agent management and monitoring"
            },
            {
                "name": "ledger",
                "description": "FTNS transaction and balance operations"
            },
            {
                "name": "storage",
                "description": "Storage provider statistics"
            },
            {
                "name": "auth",
                "description": "Authentication endpoints"
            }
        ]
    )
    
    # Add security schemes - ensure components exists
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token obtained from /auth/login"
        },
        "apiKey": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for service-to-service authentication"
        }
    }
    
    # Add global security
    openapi_schema["security"] = [{"bearerAuth": []}]
    
    # Add common schemas - ensure schemas dict exists
    if "schemas" not in openapi_schema["components"]:
        openapi_schema["components"]["schemas"] = {}
    
    openapi_schema["components"]["schemas"].update({
        "ErrorResponse": {
            "type": "object",
            "properties": {
                "detail": {"type": "string", "description": "Error message"},
                "code": {"type": "string", "description": "Error code"},
                "timestamp": {"type": "string", "format": "date-time"}
            },
            "required": ["detail"]
        },
        "StatusResponse": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "node_id": {"type": "string"},
                "uptime_seconds": {"type": "number"},
                "connected_peers": {"type": "integer"},
                "ftns_balance": {"type": "number"},
                "active_jobs": {"type": "integer"},
                "timestamp": {"type": "string", "format": "date-time"}
            }
        },
        "RateLimitInfo": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer"},
                "remaining": {"type": "integer"},
                "reset": {"type": "integer"}
            }
        }
    })
    
    app.openapi_schema = openapi_schema
    return openapi_schema


# ── API Security Configuration ───────────────────────────────────────────────────

@dataclass
class APISecurityConfig:
    """Configuration for API security hardening."""
    enable_rate_limiting: bool = True
    enable_jwt_auth: bool = True
    enable_websocket: bool = True
    enable_openapi: bool = True
    
    # Rate limit settings
    rate_limit_requests_per_minute: int = 100
    rate_limit_requests_per_hour: int = 1000
    rate_limit_burst_size: int = 20
    
    # JWT settings
    jwt_required_endpoints: List[str] = field(default_factory=lambda: [
        "/compute/submit",
        "/content/upload",
        "/agents/",
        "/ledger/transfer",
        "/transactions",
    ])
    
    # WebSocket settings
    websocket_heartbeat_interval: int = 30
    
    # Security headers
    enable_security_headers: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


class APIHardening:
    """
    Main class for applying API security hardening.
    
    Integrates rate limiting, JWT authentication, WebSocket status,
    and OpenAPI specification generation.
    """
    
    def __init__(self, app: FastAPI, config: Optional[APISecurityConfig] = None):
        self.app = app
        self.config = config or APISecurityConfig()
        self.rate_limiter: Optional[RateLimiter] = None
        self.jwt_handler = None
        self.status_websocket: Optional[StatusWebSocket] = None
        self._initialized = False
    
    def _ensure_components_constructed(self):
        """Sprint 1187 — SYNCHRONOUSLY construct the hardening components (idempotent).

        Split out of initialize() so apply_middleware() can guarantee the components
        EXIST before it registers middleware. Previously the rate_limiter was built only
        inside the async initialize(), which create_api_app create_task's (does not await)
        when the event loop is already running — so apply_middleware() ran first, saw
        rate_limiter=None, and SILENTLY skipped RateLimitMiddleware (rate limiting / DDoS
        protection off in the async-server path). Construction is pure (no await); the
        async warm-up (redis/JWT) stays in initialize() and runs on these instances."""
        if self.config.enable_rate_limiting and self.rate_limiter is None:
            self.rate_limiter = RateLimiter(RateLimitConfig(
                requests_per_minute=self.config.rate_limit_requests_per_minute,
                requests_per_hour=self.config.rate_limit_requests_per_hour,
                burst_size=self.config.rate_limit_burst_size,
                enabled=True,
            ))
        if self.config.enable_jwt_auth and self.jwt_handler is None:
            try:
                from prsm.core.auth.jwt_handler import JWTHandler
                self.jwt_handler = JWTHandler()
            except Exception as e:  # noqa: BLE001
                logger.warning("JWT handler construction failed", error=str(e))
                self.jwt_handler = None
        if self.config.enable_websocket and self.status_websocket is None:
            self.status_websocket = StatusWebSocket()

    async def initialize(self):
        """Initialize all hardening components (the async warm-up: redis/JWT backends)."""
        if self._initialized:
            return

        # Construct the components synchronously first (idempotent — apply_middleware may
        # have already built them in the loop-running race path), then warm up the async
        # backends on those instances.
        self._ensure_components_constructed()

        if self.config.enable_rate_limiting and self.rate_limiter is not None:
            await self.rate_limiter.initialize()

        if self.config.enable_jwt_auth and self.jwt_handler is not None:
            try:
                await self.jwt_handler.initialize()
            except Exception as e:
                logger.warning("JWT handler initialization failed", error=str(e))
                self.jwt_handler = None

        self._initialized = True
        logger.info("API hardening initialized")

    def apply_middleware(self):
        """Apply all middleware to the FastAPI app."""
        # sp1187 — construct components synchronously so the rate-limit middleware is
        # ALWAYS registered here, even if the async initialize() hasn't completed yet.
        self._ensure_components_constructed()

        # Apply rate limiting middleware
        if self.config.enable_rate_limiting and self.rate_limiter:
            self.app.add_middleware(RateLimitMiddleware, rate_limiter=self.rate_limiter)
            logger.info("Rate limiting middleware applied")
        
        # Apply JWT authentication middleware
        if self.config.enable_jwt_auth:
            self.app.add_middleware(JWTAuthMiddleware, jwt_handler=self.jwt_handler)
            logger.info("JWT authentication middleware applied")
    
    def setup_openapi(self):
        """Setup OpenAPI specification."""
        if self.config.enable_openapi:
            self.app.openapi = lambda: generate_openapi_schema(self.app)
            logger.info("OpenAPI specification configured")
    
    def get_status_websocket(self) -> Optional[StatusWebSocket]:
        """Get WebSocket manager for status updates."""
        return self.status_websocket


# ── Dependency Injection Helpers ───────────────────────────────────────────────

async def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    """Get current authenticated user from request state."""
    return getattr(request.state, "user", None)


async def require_auth(request: Request) -> Dict[str, Any]:
    """Require authentication and return user data."""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return user


async def require_role(request: Request, required_role: str) -> Dict[str, Any]:
    """Require specific role and return user data."""
    user = await require_auth(request)
    user_role = user.get("role", "user")
    
    # Role hierarchy: admin > moderator > user
    role_hierarchy = {"admin": 3, "moderator": 2, "user": 1}
    
    if role_hierarchy.get(user_role, 0) < role_hierarchy.get(required_role, 0):
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail=f"Role '{required_role}' required"
        )
    
    return user


# ── WebSocket Endpoint Helper ──────────────────────────────────────────────────

async def websocket_status_endpoint(websocket: WebSocket, status_ws: StatusWebSocket):
    """
    WebSocket endpoint for status updates.
    
    Usage:
        @app.websocket("/ws/status")
        async def ws_status(websocket: WebSocket):
            await websocket_status_endpoint(websocket, status_websocket)
    """
    await status_ws.connect(websocket)
    
    try:
        while True:
            # Wait for messages from client (heartbeat, commands, etc.)
            data = await websocket.receive_json()
            
            # Handle heartbeat
            if data.get("type") == "heartbeat":
                await status_ws.send_personal_status(websocket, {
                    "type": "heartbeat_ack",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            
            # Handle status request
            elif data.get("type") == "get_status":
                # This would typically fetch from node
                await status_ws.send_personal_status(websocket, {
                    "type": "status_update",
                    "data": {"status": "ok"},
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
    
    except WebSocketDisconnect:
        status_ws.disconnect(websocket)
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        status_ws.disconnect(websocket)