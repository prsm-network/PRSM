"""
Authentication Middleware
FastAPI middleware for authentication, rate limiting, and security headers
"""

import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timezone
import structlog

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.status import HTTP_429_TOO_MANY_REQUESTS, HTTP_401_UNAUTHORIZED

from prsm.core.redis_client import get_redis_client, resolve_async_redis
from prsm.core.integrations.security.audit_logger import audit_logger

logger = structlog.get_logger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive authentication and security middleware
    
    Features:
    - Rate limiting with Redis backend
    - Security headers injection
    - Request logging and audit trails
    - IP blocking and abuse prevention
    - Authentication bypass for public endpoints
    """
    
    def __init__(self, app, rate_limit_requests: int = 100, rate_limit_window: int = 60):
        super().__init__(app)
        self.rate_limit_requests = rate_limit_requests  # requests per window
        self.rate_limit_window = rate_limit_window      # window in seconds
        self.redis_client = None
        
        # Public endpoints that don't require authentication
        self.public_endpoints = {
            "/docs",
            "/redoc", 
            "/openapi.json",
            "/health",
            "/metrics",
            "/auth/login",
            "/auth/register",
            "/auth/refresh"
        }
        
        # Rate limit exempt endpoints (internal health checks)
        self.rate_limit_exempt = {
            "/health",
            "/metrics"
        }
        
        # Security headers to add to all responses
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request through middleware pipeline
        
        Args:
            request: FastAPI request object
            call_next: Next middleware/endpoint in chain
            
        Returns:
            Response with security enhancements
        """
        start_time = time.time()
        
        # Initialize Redis client if needed
        if not self.redis_client:
            try:
                self.redis_client = get_redis_client()
            except Exception as e:
                logger.warning("Redis not available for middleware", error=str(e))
        
        # Extract client information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        request_path = request.url.path
        request_method = request.method
        
        # Create request context for logging
        request_context = {
            "ip": client_ip,
            "user_agent": user_agent,
            "path": request_path,
            "method": request_method,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # 1. Check rate limiting (except for exempt endpoints)
            if request_path not in self.rate_limit_exempt:
                rate_limit_check = await self._check_rate_limit(client_ip, request_context)
                if rate_limit_check:
                    return rate_limit_check
            
            # 2. Check IP blocking (if implemented)
            ip_block_check = await self._check_ip_blocking(client_ip, request_context)
            if ip_block_check:
                return ip_block_check
            
            # 3. Process request
            response = await call_next(request)
            
            # 4. Add security headers
            self._add_security_headers(response)
            
            # 5. Log successful request
            processing_time = time.time() - start_time
            await self._log_request(request_context, response.status_code, processing_time)
            
            return response
            
        except Exception as e:
            # Log error and return generic error response
            processing_time = time.time() - start_time
            logger.error("Middleware error", error=str(e), **request_context)
            
            await audit_logger.log_security_event(
                "middleware_error",
                {"error": str(e), "processing_time": processing_time},
                request_context
            )
            
            # Return generic error to avoid information leakage
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )
            self._add_security_headers(response)
            return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request headers"""
        # Check for forwarded headers (behind reverse proxy)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to direct connection IP
        return request.client.host if request.client else "unknown"
    
    async def _check_rate_limit(self, client_ip: str, request_context: Dict[str, Any]) -> Optional[Response]:
        """
        Check rate limiting for client IP
        
        Args:
            client_ip: Client IP address
            request_context: Request context for logging
            
        Returns:
            Rate limit response if exceeded, None if OK
        """
        # sp1247 — resolve the live inner async Redis (self.redis_client is the
        # RedisClient wrapper, which has no .pipeline; calling it raised AttributeError
        # → the except below failed OPEN, silently disabling rate limiting). None →
        # no live Redis → skip (the documented no-Redis behavior).
        _redis = resolve_async_redis(self.redis_client)
        if _redis is None:
            # Skip rate limiting if Redis not available
            return None

        try:
            # Create rate limit key
            rate_limit_key = f"rate_limit:{client_ip}"
            current_time = int(time.time())
            window_start = current_time - (current_time % self.rate_limit_window)

            # Use Redis pipeline for atomic operations
            pipe = _redis.pipeline()
            
            # Sliding window rate limiting
            window_key = f"{rate_limit_key}:{window_start}"
            
            # Increment counter and set expiration
            pipe.incr(window_key)
            pipe.expire(window_key, self.rate_limit_window * 2)  # Keep for 2 windows
            
            # Execute pipeline
            results = await pipe.execute()
            current_requests = results[0]
            
            # Check if limit exceeded
            if current_requests > self.rate_limit_requests:
                await audit_logger.log_security_event(
                    "rate_limit_exceeded",
                    {
                        "requests": current_requests,
                        "limit": self.rate_limit_requests,
                        "window": self.rate_limit_window
                    },
                    request_context
                )
                
                logger.warning("Rate limit exceeded",
                              client_ip=client_ip,
                              requests=current_requests,
                              limit=self.rate_limit_requests)
                
                response = JSONResponse(
                    status_code=HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": "Rate limit exceeded",
                        "retry_after": self.rate_limit_window
                    },
                    headers={"Retry-After": str(self.rate_limit_window)}
                )
                self._add_security_headers(response)
                return response
            
            # Add rate limit headers to track usage
            remaining = max(0, self.rate_limit_requests - current_requests)
            logger.debug("Rate limit check passed",
                        client_ip=client_ip,
                        requests=current_requests,
                        remaining=remaining)
            
            return None
            
        except Exception as e:
            logger.error("Rate limit check error", error=str(e), client_ip=client_ip)
            # Allow request on error to avoid blocking legitimate traffic
            return None
    
    async def _check_ip_blocking(self, client_ip: str, request_context: Dict[str, Any]) -> Optional[Response]:
        """
        Check if IP is blocked
        
        Args:
            client_ip: Client IP address  
            request_context: Request context for logging
            
        Returns:
            Block response if IP is blocked, None if OK
        """
        # sp1247 — resolve the live inner async Redis (the wrapper has no .get;
        # calling it raised AttributeError → the block check failed OPEN, so a
        # blocked IP was admitted). None → no live Redis → skip.
        _redis = resolve_async_redis(self.redis_client)
        if _redis is None:
            return None

        try:
            # Check if IP is in block list
            blocked_key = f"blocked_ip:{client_ip}"
            is_blocked = await _redis.get(blocked_key)
            
            if is_blocked:
                await audit_logger.log_security_event(
                    "blocked_ip_access_attempt",
                    {"block_reason": is_blocked.decode() if isinstance(is_blocked, bytes) else str(is_blocked)},
                    request_context
                )
                
                logger.warning("Blocked IP access attempt", client_ip=client_ip)
                
                response = JSONResponse(
                    status_code=HTTP_401_UNAUTHORIZED,
                    content={"detail": "Access denied"}
                )
                self._add_security_headers(response)
                return response
            
            return None
            
        except Exception as e:
            logger.error("IP blocking check error", error=str(e), client_ip=client_ip)
            # Allow request on error
            return None
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response"""
        for header, value in self.security_headers.items():
            response.headers[header] = value
    
    async def _log_request(self, request_context: Dict[str, Any], status_code: int, processing_time: float):
        """Log request for audit and monitoring"""
        try:
            # Determine log level based on status code
            if status_code >= 500:
                log_level = "error"
            elif status_code >= 400:
                log_level = "warning"
            else:
                log_level = "info"
            
            # Log request with appropriate level
            log_data = {
                **request_context,
                "status_code": status_code,
                "processing_time": round(processing_time, 3)
            }
            
            if log_level == "error":
                logger.error("Request completed with error", **log_data)
            elif log_level == "warning":
                logger.warning("Request completed with client error", **log_data)
            else:
                logger.debug("Request completed successfully", **log_data)
            
            # Log to audit system for security monitoring
            await audit_logger.log_access_event(
                request_context["method"],
                request_context["path"],
                status_code,
                request_context
            )
            
        except Exception as e:
            logger.error("Request logging error", error=str(e))


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Simplified middleware focused only on security headers
    Use this if you don't need rate limiting or other auth features
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY", 
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to all responses"""
        response = await call_next(request)
        
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        return response


class CORSSecurityMiddleware(BaseHTTPMiddleware):
    """
    Enhanced CORS middleware with security considerations
    """
    
    def __init__(self, app, allowed_origins: list = None, allowed_methods: list = None):
        super().__init__(app)
        self.allowed_origins = allowed_origins or ["http://localhost:3000", "https://prsm.ai"]
        self.allowed_methods = allowed_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allowed_headers = ["Authorization", "Content-Type", "X-Requested-With"]
        self.max_age = 86400  # 24 hours
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle CORS with security validation"""
        origin = request.headers.get("origin")
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            if origin in self.allowed_origins:
                response = Response()
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
                response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
                response.headers["Access-Control-Max-Age"] = str(self.max_age)
                response.headers["Access-Control-Allow-Credentials"] = "true"
                return response
            else:
                return Response(status_code=403)
        
        # Process actual request
        response = await call_next(request)
        
        # Add CORS headers for allowed origins
        if origin in self.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response