"""
Standardized Middleware for PRSM API
Provides consistent request processing, security, and monitoring
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Callable, Dict, Any
from uuid import uuid4

from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from .standards import SECURITY_HEADERS, CORS_CONFIG


logger = logging.getLogger(__name__)


# Development-only CORS origins (never used in production)
DEV_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:8000",
]


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request IDs to all requests"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID if not provided
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        trace_id = request.headers.get("X-Trace-ID", str(uuid4()))
        
        # Store in request state
        request.state.request_id = request_id
        request.state.trace_id = trace_id
        
        # Process request
        response = await call_next(request)
        
        # Add to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Trace-ID"] = trace_id
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        for header, value in SECURITY_HEADERS.items():
            response.headers[header] = value
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log all API requests and responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Log request
        request_data = {
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "user_agent": request.headers.get("User-Agent"),
            "client_host": request.client.host if request.client else None,
            "request_id": getattr(request.state, 'request_id', None),
            "trace_id": getattr(request.state, 'trace_id', None),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra=request_data
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            response_data = {
                **request_data,
                "status_code": response.status_code,
                "process_time_seconds": round(process_time, 4),
                "response_size_bytes": len(response.body) if hasattr(response, 'body') else None
            }
            
            # Choose log level based on status code
            if response.status_code < 400:
                log_level = logging.INFO
                log_message = f"Request completed: {request.method} {request.url.path} - {response.status_code}"
            elif response.status_code < 500:
                log_level = logging.WARNING
                log_message = f"Client error: {request.method} {request.url.path} - {response.status_code}"
            else:
                log_level = logging.ERROR
                log_message = f"Server error: {request.method} {request.url.path} - {response.status_code}"
            
            logger.log(log_level, log_message, extra=response_data)
            
            # Add performance headers
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            # Calculate processing time for errors
            process_time = time.time() - start_time
            
            # Log error
            error_data = {
                **request_data,
                "error": str(e),
                "error_type": type(e).__name__,
                "process_time_seconds": round(process_time, 4)
            }
            
            logger.error(
                f"Request failed: {request.method} {request.url.path} - {type(e).__name__}",
                extra=error_data,
                exc_info=True
            )
            
            # Re-raise the exception
            raise


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Monitor API performance and add metrics"""
    
    def __init__(self, app, slow_request_threshold: float = 1.0):
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        process_time = time.time() - start_time
        
        # Log slow requests
        if process_time > self.slow_request_threshold:
            logger.warning(
                f"Slow request detected: {request.method} {request.url.path}",
                extra={
                    "process_time_seconds": process_time,
                    "threshold_seconds": self.slow_request_threshold,
                    "request_id": getattr(request.state, 'request_id', None),
                    "trace_id": getattr(request.state, 'trace_id', None)
                }
            )
        
        # Record into Prometheus metrics
        try:
            from prsm.compute.performance.metrics import increment_counter, observe_histogram
            increment_counter(
                "http_requests_total",
                method=request.method,
                endpoint=request.url.path,
                status_code=str(response.status_code)
            )
            observe_histogram(
                "http_request_duration_seconds",
                process_time,
                method=request.method,
                endpoint=request.url.path
            )
        except Exception:
            pass  # Never let metrics recording fail a request
        
        # Ingest event into real-time analytics processor
        try:
            from prsm.data.analytics.real_time_processor import (
                get_real_time_processor, StreamEvent, StreamEventType
            )
            import uuid

            processor = get_real_time_processor()
            event = StreamEvent(
                event_id=str(uuid.uuid4()),
                event_type=StreamEventType.PERFORMANCE_EVENT,
                timestamp=datetime.now(timezone.utc),
                data={
                    "latency_ms": round(process_time * 1000, 2),
                    "status_code": response.status_code,
                    "method": request.method,
                    "path": request.url.path,
                },
                source="api_middleware",
                tags=["http", request.method.lower()]
            )
            await processor.ingest_event(event)

        except RuntimeError:
            pass  # Processor not initialized — skip silently
        except Exception:
            pass  # Never let analytics ingestion fail a request
        
        # Add performance headers
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        response.headers["X-Timestamp"] = datetime.now(timezone.utc).isoformat()
        
        return response


class ContentValidationMiddleware(BaseHTTPMiddleware):
    """Validate request content and enforce limits"""
    
    def __init__(self, app, max_request_size: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__(app)
        self.max_request_size = max_request_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check content length
        content_length = request.headers.get("Content-Length")
        if content_length and int(content_length) > self.max_request_size:
            logger.warning(
                f"Request too large: {content_length} bytes (max: {self.max_request_size})",
                extra={
                    "request_size": content_length,
                    "max_size": self.max_request_size,
                    "url": str(request.url),
                    "request_id": getattr(request.state, 'request_id', None)
                }
            )
            from fastapi import HTTPException
            raise HTTPException(
                status_code=413,
                detail=f"Request too large. Maximum size: {self.max_request_size} bytes"
            )
        
        # Process request
        response = await call_next(request)
        
        return response


def setup_middleware(app):
    """Setup all middleware for the FastAPI app in correct order"""
    
    # Order matters! Middleware is applied in reverse order of addition
    
    # 1. CORS (should be first to handle preflight requests)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_CONFIG["allow_origins"],
        allow_credentials=CORS_CONFIG["allow_credentials"],
        allow_methods=CORS_CONFIG["allow_methods"],
        allow_headers=CORS_CONFIG["allow_headers"]
    )
    
    # 2. Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # 3. Content validation (early to reject large requests)
    app.add_middleware(ContentValidationMiddleware, max_request_size=10 * 1024 * 1024)
    
    # 4. Request ID assignment (needs to be early for logging)
    app.add_middleware(RequestIDMiddleware)
    
    # 5. Performance monitoring
    app.add_middleware(PerformanceMonitoringMiddleware, slow_request_threshold=1.0)
    
    # 6. Request/response logging
    app.add_middleware(LoggingMiddleware)
    
    # 7. Security headers (should be last to ensure they're added)
    app.add_middleware(SecurityHeadersMiddleware)
    
    logger.info("All middleware configured successfully")


def get_request_context(request: Request) -> Dict[str, Any]:
    """Extract standardized request context"""
    
    return {
        "request_id": getattr(request.state, 'request_id', None),
        "trace_id": getattr(request.state, 'trace_id', None),
        "method": request.method,
        "url": str(request.url),
        "path": request.url.path,
        "query_params": dict(request.query_params),
        "user_agent": request.headers.get("User-Agent"),
        "client_host": request.client.host if request.client else None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# Per-endpoint rate limit configuration
# 0 = unlimited
ENDPOINT_RATE_LIMITS = {
    "/health": {"limit": 0, "window": 60},           # Unlimited - health checks
    "/api/v1/query": {"limit": 10, "window": 60},    # Expensive AI calls
    "/api/v1/sessions": {"limit": 20, "window": 60}, # Session management
    "/api/v1/marketplace": {"limit": 100, "window": 60},  # Browse marketplace
    "/api/v1/ftns": {"limit": 50, "window": 60},     # FTNS operations
    "/docs": {"limit": 0, "window": 60},             # Unlimited - documentation
    "/redoc": {"limit": 0, "window": 60},            # Unlimited - documentation
    "/openapi.json": {"limit": 0, "window": 60},     # Unlimited - API spec
    "default": {"limit": 200, "window": 60},         # Default for all other endpoints
}


def _get_endpoint_limit(path: str) -> Dict[str, int]:
    """Get rate limit configuration for a given path."""
    # Check for exact match first
    if path in ENDPOINT_RATE_LIMITS:
        return ENDPOINT_RATE_LIMITS[path]

    # Check for prefix match (e.g., /api/v1/query/123 -> /api/v1/query)
    for endpoint, config in ENDPOINT_RATE_LIMITS.items():
        if endpoint != "default" and path.startswith(endpoint):
            return config

    return ENDPOINT_RATE_LIMITS["default"]


async def _extract_user_id_from_token(request: Request) -> str:
    """Extract user ID from JWT token in Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None

    token = auth_header[7:]  # Remove "Bearer " prefix

    try:
        # Import here to avoid circular imports

        # Sprint 1123 (Domain-08 review LOW-8) — VERIFY the signature before trusting the
        # `sub` claim to key the per-user rate-limit bucket. The prior verify_signature=
        # False let an attacker forge a `sub` to (a) evade their per-user limit and (b)
        # grief a victim by exhausting the victim's per-user allowance with a forged token.
        # Fail-safe: if verification fails for any reason (forged/unsigned token, or a
        # token signed with a different key than this middleware sees), return None — the
        # request then falls back to per-IP limiting, which always applies.
        import jwt
        from prsm.core.config import get_settings
        settings = get_settings()

        payload = jwt.decode(
            token,
            settings.secret_key if settings else "test-secret-key",
            algorithms=[settings.jwt_algorithm if settings else "HS256"],
            options={"verify_signature": True, "verify_exp": False}
        )
        return payload.get("sub")  # User ID (now from a signature-verified token)
    except Exception as e:
        logger.debug(f"Failed to extract user ID from token: {e}")
        return None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Global rate limiting middleware with per-IP, per-user, and per-endpoint limits.

    Features:
    - Per-IP rate limiting (existing behavior)
    - Per-user rate limiting for authenticated requests
    - Per-endpoint rate limiting with configurable limits
    - Unlimited access for health/documentation endpoints
    """

    def __init__(self, app, default_limit: int = 1000, window: int = 60):
        super().__init__(app)
        self.default_limit = default_limit
        self.window = window
        self._ip_requests: Dict[str, list] = {}      # IP-keyed request timestamps
        self._user_requests: Dict[str, list] = {}    # User-ID-keyed request timestamps

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        current_time = time.time()
        path = request.url.path

        # Get endpoint-specific rate limit configuration
        endpoint_config = _get_endpoint_limit(path)
        endpoint_limit = endpoint_config["limit"]
        endpoint_window = endpoint_config["window"]

        # Unlimited endpoints (limit = 0)
        if endpoint_limit == 0:
            return await call_next(request)

        # Get client identifier (IP address)
        client_ip = request.client.host if request.client else "unknown"

        # Extract user ID if authenticated
        user_id = await _extract_user_id_from_token(request)

        # === Per-IP Rate Limiting ===
        ip_key = f"ip:{client_ip}"
        self._clean_old_requests(ip_key, current_time, endpoint_window)

        if ip_key not in self._ip_requests:
            self._ip_requests[ip_key] = []

        # Check IP-based rate limit
        if len(self._ip_requests[ip_key]) >= endpoint_limit:
            logger.warning(
                f"IP rate limit exceeded",
                extra={
                    "client_ip": client_ip,
                    "request_count": len(self._ip_requests[ip_key]),
                    "limit": endpoint_limit,
                    "window": endpoint_window,
                    "path": path,
                    "request_id": getattr(request.state, 'request_id', None)
                }
            )
            from fastapi import HTTPException
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Limit: {endpoint_limit} requests per {endpoint_window} seconds.",
                headers={"Retry-After": str(endpoint_window)}
            )

        # === Per-User Rate Limiting (for authenticated requests) ===
        if user_id:
            user_key = f"user:{user_id}"
            # User gets a separate bucket with the same limit as the endpoint
            user_limit = endpoint_limit  # Same limit for user as for IP

            self._clean_old_requests(user_key, current_time, endpoint_window)

            if user_key not in self._user_requests:
                self._user_requests[user_key] = []

            # Check user-based rate limit
            if len(self._user_requests[user_key]) >= user_limit:
                logger.warning(
                    f"User rate limit exceeded",
                    extra={
                        "user_id": user_id,
                        "request_count": len(self._user_requests[user_key]),
                        "limit": user_limit,
                        "window": endpoint_window,
                        "path": path,
                        "request_id": getattr(request.state, 'request_id', None)
                    }
                )
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=429,
                    detail=f"User rate limit exceeded. Limit: {user_limit} requests per {endpoint_window} seconds.",
                    headers={"Retry-After": str(endpoint_window)}
                )

            # Record user request
            self._user_requests[user_key].append(current_time)

        # Record IP request
        self._ip_requests[ip_key].append(current_time)

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        ip_remaining = endpoint_limit - len(self._ip_requests.get(ip_key, []))
        response.headers["X-RateLimit-Limit"] = str(endpoint_limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, ip_remaining))
        response.headers["X-RateLimit-Reset"] = str(int(current_time + endpoint_window))
        response.headers["X-RateLimit-Window"] = str(endpoint_window)

        return response

    def _clean_old_requests(self, key: str, current_time: float, window: int):
        """Remove requests older than the window from the specified bucket."""
        if key in self._ip_requests:
            self._ip_requests[key] = [
                req_time for req_time in self._ip_requests[key]
                if current_time - req_time < window
            ]
        if key in self._user_requests:
            self._user_requests[key] = [
                req_time for req_time in self._user_requests[key]
                if current_time - req_time < window
            ]


def configure_middleware_stack(app) -> None:
    """
    Configure complete middleware stack for production PRSM API.

    This function sets up all middleware in the correct order for:
    - Security headers and CORS
    - Request validation and rate limiting
    - Logging and monitoring
    - Authentication

    Order is critical - middleware is applied in reverse order of addition.
    """
    from prsm.core.config import get_settings
    settings = get_settings()

    # === CORS Configuration ===
    try:
        from prsm.core.security.middleware import configure_cors
        cors_config = configure_cors()
        
        # Use development origins only in development mode
        env = os.getenv("PRSM_ENV", "development").lower()
        origins = DEV_ORIGINS if env == "development" else cors_config["allow_origins"]
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=cors_config["allow_credentials"],
            allow_methods=cors_config["allow_methods"],
            allow_headers=cors_config["allow_headers"],
            expose_headers=cors_config.get("expose_headers", []),
            max_age=cors_config.get("max_age", 600)
        )
        logger.info("CORS middleware configured")
    except ImportError:
        # Fallback CORS configuration
        app.add_middleware(
            CORSMiddleware,
            allow_origins=CORS_CONFIG["allow_origins"],
            allow_credentials=CORS_CONFIG["allow_credentials"],
            allow_methods=CORS_CONFIG["allow_methods"],
            allow_headers=CORS_CONFIG["allow_headers"]
        )
        logger.info("CORS middleware configured (fallback)")

    # === Gzip Compression ===
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # === Content Validation ===
    app.add_middleware(ContentValidationMiddleware, max_request_size=10 * 1024 * 1024)

    # === Request ID Assignment ===
    app.add_middleware(RequestIDMiddleware)

    # === Performance Monitoring ===
    app.add_middleware(PerformanceMonitoringMiddleware, slow_request_threshold=1.0)

    # === Request/Response Logging ===
    app.add_middleware(LoggingMiddleware)

    # === Security Headers ===
    app.add_middleware(SecurityHeadersMiddleware)

    # === Enhanced Security Middleware ===
    try:
        from prsm.core.security.middleware import (
            RequestValidationMiddleware,
            RateLimitingMiddleware,
            SecurityHeadersMiddleware as EnhancedSecurityHeaders
        )
        app.add_middleware(RequestValidationMiddleware)
        app.add_middleware(RateLimitingMiddleware)
        app.add_middleware(EnhancedSecurityHeaders)
        logger.info("Enhanced security middleware configured")
    except ImportError as e:
        logger.warning(f"Enhanced security middleware not available: {e}")

    # === Auth Middleware ===
    try:
        from prsm.core.auth.middleware import AuthMiddleware
        app.add_middleware(AuthMiddleware, rate_limit_requests=100, rate_limit_window=60)
        logger.info("Auth middleware configured")
    except ImportError as e:
        logger.warning(f"Auth middleware not available: {e}")

    # === Request Limits Middleware ===
    try:
        from prsm.core.security import RequestLimitsMiddleware, request_limits_config
        app.add_middleware(RequestLimitsMiddleware, config=request_limits_config)
        logger.info("Request limits middleware configured")
    except ImportError as e:
        logger.warning(f"Request limits middleware not available: {e}")

    # === API Versioning Middleware ===
    try:
        from prsm.interface.api.versioning import VersioningMiddleware, version_negotiator
        from prsm.interface.api.compatibility import CompatibilityMiddleware, compatibility_engine

        versioning_middleware = VersioningMiddleware(version_negotiator)
        compatibility_middleware = CompatibilityMiddleware(compatibility_engine)

        app.middleware("http")(versioning_middleware)
        app.middleware("http")(compatibility_middleware)
        logger.info("API versioning middleware configured")
    except ImportError as e:
        logger.warning(f"API versioning middleware not available: {e}")

    logger.info("Complete middleware stack configured successfully")