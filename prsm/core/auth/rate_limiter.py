"""
Advanced Rate Limiting System
Comprehensive rate limiting with Redis backend and intelligent protection
"""

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import structlog

from prsm.core.redis_client import get_redis_client, resolve_async_redis
from prsm.core.integrations.security.audit_logger import audit_logger

logger = structlog.get_logger(__name__)


class RateLimitType(str, Enum):
    """Types of rate limits"""
    PER_IP = "per_ip"
    PER_USER = "per_user"
    PER_ENDPOINT = "per_endpoint"
    GLOBAL = "global"


class RateLimitRule:
    """Rate limit rule definition"""
    
    def __init__(
        self,
        name: str,
        limit_type: RateLimitType,
        requests: int,
        window: int,
        endpoint_pattern: str = "*",
        user_roles: List[str] = None,
        enabled: bool = True
    ):
        self.name = name
        self.limit_type = limit_type
        self.requests = requests
        self.window = window
        self.endpoint_pattern = endpoint_pattern
        self.user_roles = user_roles or []
        self.enabled = enabled


class RateLimiter:
    """
    Advanced rate limiting system with multiple strategies
    
    Features:
    - Multiple rate limit types (IP, user, endpoint, global)
    - Sliding window algorithm
    - Dynamic rate limit adjustment
    - Burst handling with token bucket
    - IP reputation scoring
    - Automatic threat detection
    - Whitelist/blacklist support
    """
    
    def __init__(self):
        self.redis_client = None
        
        # Default rate limit rules
        self.rules = [
            # Global API limits
            RateLimitRule("global_api", RateLimitType.GLOBAL, 10000, 3600),  # 10k/hour globally
            
            # Per-IP limits
            RateLimitRule("ip_general", RateLimitType.PER_IP, 100, 60),      # 100/min per IP
            RateLimitRule("ip_auth", RateLimitType.PER_IP, 20, 300, "/auth/*"),  # 20/5min for auth
            RateLimitRule("ip_registration", RateLimitType.PER_IP, 5, 3600, "/auth/register"),  # 5/hour registration
            
            # Per-user limits (authenticated users)
            RateLimitRule("user_general", RateLimitType.PER_USER, 1000, 3600),  # 1k/hour per user
            RateLimitRule("user_model_execution", RateLimitType.PER_USER, 100, 3600, "/models/execute/*"),
            
            # Endpoint-specific limits
            RateLimitRule("endpoint_expensive", RateLimitType.PER_ENDPOINT, 50, 3600, "/models/train/*"),
            RateLimitRule("endpoint_upload", RateLimitType.PER_ENDPOINT, 20, 3600, "/content/upload"),
        ]
        
        # IP reputation tracking
        self.ip_reputation_threshold = 0.3  # Below this = suspicious
        self.suspicious_behavior_window = 3600  # 1 hour
        
        # Whitelist for trusted IPs
        self.whitelisted_ips = set()
        
        # Blacklist for blocked IPs
        self.blacklisted_ips = set()
        
    async def initialize(self):
        """Initialize rate limiter with Redis connection"""
        try:
            self.redis_client = get_redis_client()
            
            # Load IP whitelist/blacklist from Redis
            await self._load_ip_lists()
            
            logger.info("Rate limiter initialized",
                       rules_count=len(self.rules),
                       whitelisted_ips=len(self.whitelisted_ips),
                       blacklisted_ips=len(self.blacklisted_ips))
                       
        except Exception as e:
            logger.error("Failed to initialize rate limiter", error=str(e))
            
    async def check_rate_limit(
        self,
        identifier: str,
        endpoint: str,
        user_id: Optional[str] = None,
        user_role: Optional[str] = None,
        client_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request should be rate limited
        
        Args:
            identifier: IP address or user identifier
            endpoint: Request endpoint
            user_id: Authenticated user ID (if any)
            user_role: User role (if authenticated)
            client_info: Additional client information
            
        Returns:
            Tuple of (is_allowed, limit_info)
        """
        if self._live_redis() is None:  # sp1249 — resolve the live inner client
            logger.warning("Rate limiter not initialized, allowing request")
            return True, {}
        
        client_ip = client_info.get("ip", identifier) if client_info else identifier
        
        try:
            # Check if IP is whitelisted
            if client_ip in self.whitelisted_ips:
                return True, {"whitelisted": True}
            
            # Check if IP is blacklisted
            if client_ip in self.blacklisted_ips:
                await self._log_blocked_request(client_ip, endpoint, "blacklisted", client_info)
                return False, {"blocked": True, "reason": "blacklisted"}
            
            # Check IP reputation
            reputation = await self._get_ip_reputation(client_ip)
            if reputation < self.ip_reputation_threshold:
                await self._log_blocked_request(client_ip, endpoint, "low_reputation", client_info)
                return False, {"blocked": True, "reason": "low_reputation", "reputation": reputation}
            
            # Apply rate limit rules
            for rule in self.rules:
                if not rule.enabled:
                    continue
                    
                # Check if rule applies to this request
                if not self._rule_applies(rule, endpoint, user_role):
                    continue
                
                # Determine the key for this rule
                key = self._get_rate_limit_key(rule, identifier, user_id, endpoint)
                
                # Check rate limit
                is_allowed, current_count, window_remaining = await self._check_sliding_window(
                    key, rule.requests, rule.window
                )
                
                if not is_allowed:
                    await self._log_rate_limit_exceeded(
                        rule, identifier, endpoint, current_count, client_info
                    )
                    
                    # Update IP reputation for rate limit violations
                    await self._update_ip_reputation(client_ip, -0.1)
                    
                    return False, {
                        "blocked": True,
                        "reason": "rate_limit_exceeded",
                        "rule": rule.name,
                        "current_count": current_count,
                        "limit": rule.requests,
                        "window": rule.window,
                        "retry_after": window_remaining
                    }
            
            # All checks passed
            return True, {"allowed": True}
            
        except Exception as e:
            logger.error("Rate limit check error", error=str(e), identifier=identifier)
            # Fail open to avoid blocking legitimate traffic
            return True, {"error": str(e)}
    
    async def increment_counter(
        self,
        identifier: str,
        endpoint: str,
        user_id: Optional[str] = None,
        user_role: Optional[str] = None
    ):
        """Increment rate limit counters after successful request"""
        if self._live_redis() is None:  # sp1249
            return
        
        try:
            # Increment counters for all applicable rules
            for rule in self.rules:
                if not rule.enabled or not self._rule_applies(rule, endpoint, user_role):
                    continue
                
                key = self._get_rate_limit_key(rule, identifier, user_id, endpoint)
                await self._increment_sliding_window(key, rule.window)
                
        except Exception as e:
            logger.error("Counter increment error", error=str(e))
    
    async def add_to_whitelist(self, ip: str, reason: str = "") -> bool:
        """Add IP to whitelist"""
        try:
            self.whitelisted_ips.add(ip)

            _redis = self._live_redis()  # sp1249
            if _redis is not None:
                await _redis.sadd("rate_limit:whitelist", ip)
                await _redis.hset(
                    "rate_limit:whitelist_reasons",
                    ip,
                    f"{reason} - {datetime.now(timezone.utc).isoformat()}"
                )
            
            logger.info("IP added to whitelist", ip=ip, reason=reason)
            return True
            
        except Exception as e:
            logger.error("Failed to add IP to whitelist", error=str(e), ip=ip)
            return False
    
    async def add_to_blacklist(self, ip: str, reason: str = "", duration: Optional[int] = None) -> bool:
        """Add IP to blacklist"""
        try:
            self.blacklisted_ips.add(ip)

            _redis = self._live_redis()  # sp1249
            if _redis is not None:
                if duration:
                    # Temporary blacklist with expiration
                    await _redis.setex(f"rate_limit:blacklist:{ip}", duration, reason)
                else:
                    # Permanent blacklist
                    await _redis.sadd("rate_limit:blacklist", ip)
                    await _redis.hset(
                        "rate_limit:blacklist_reasons",
                        ip,
                        f"{reason} - {datetime.now(timezone.utc).isoformat()}"
                    )
            
            logger.info("IP added to blacklist", ip=ip, reason=reason, duration=duration)
            return True
            
        except Exception as e:
            logger.error("Failed to add IP to blacklist", error=str(e), ip=ip)
            return False
    
    async def get_rate_limit_status(self, identifier: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current rate limit status for identifier"""
        if self._live_redis() is None:  # sp1249
            return {}
        
        try:
            status = {}
            
            for rule in self.rules:
                if not rule.enabled:
                    continue
                
                key = self._get_rate_limit_key(rule, identifier, user_id, "")
                current_count, window_remaining = await self._get_sliding_window_status(key, rule.window)
                
                status[rule.name] = {
                    "current_count": current_count,
                    "limit": rule.requests,
                    "window": rule.window,
                    "remaining": max(0, rule.requests - current_count),
                    "window_remaining": window_remaining
                }
            
            return status
            
        except Exception as e:
            logger.error("Failed to get rate limit status", error=str(e))
            return {}
    
    # Private helper methods

    def _live_redis(self):
        """sp1249 — the live inner async Redis, or None when disconnected.

        self.redis_client is the RedisClient WRAPPER (get_redis_client()), which
        exposes NO raw Redis commands — calling .zadd/.zcard/.get/.setex on it raised
        AttributeError, swallowed by each method's fail-open/fail-safe except, so this
        whole limiter silently degraded (reputation always 1.0, windows never recorded,
        blacklist writes lost). Centralizes the sp1247 resolve_async_redis() unwrap.
        """
        return resolve_async_redis(self.redis_client)

    def _rule_applies(self, rule: RateLimitRule, endpoint: str, user_role: Optional[str]) -> bool:
        """Check if rate limit rule applies to this request"""
        # Check endpoint pattern
        if rule.endpoint_pattern != "*":
            if not endpoint.startswith(rule.endpoint_pattern.replace("*", "")):
                return False
        
        # Check user role requirements
        if rule.user_roles and user_role not in rule.user_roles:
            return False
        
        return True
    
    def _get_rate_limit_key(
        self,
        rule: RateLimitRule,
        identifier: str,
        user_id: Optional[str],
        endpoint: str
    ) -> str:
        """Generate Redis key for rate limit rule"""
        if rule.limit_type == RateLimitType.GLOBAL:
            return f"rate_limit:global:{rule.name}"
        elif rule.limit_type == RateLimitType.PER_IP:
            return f"rate_limit:ip:{rule.name}:{identifier}"
        elif rule.limit_type == RateLimitType.PER_USER and user_id:
            return f"rate_limit:user:{rule.name}:{user_id}"
        elif rule.limit_type == RateLimitType.PER_ENDPOINT:
            return f"rate_limit:endpoint:{rule.name}:{endpoint}"
        else:
            # Fallback to IP-based
            return f"rate_limit:fallback:{rule.name}:{identifier}"
    
    async def _check_sliding_window(self, key: str, limit: int, window: int) -> Tuple[bool, int, int]:
        """Check sliding window rate limit"""
        try:
            _redis = self._live_redis()  # sp1249
            if _redis is None:
                return True, 0, 0  # no live Redis → allow (matches the no-redis guard)
            current_time = int(time.time())
            window_start = current_time - window

            # Remove old entries
            await _redis.zremrangebyscore(key, 0, window_start)

            # Count current entries
            current_count = await _redis.zcard(key)

            # Check if limit exceeded
            if current_count >= limit:
                # Calculate time remaining in window
                oldest_entry = await _redis.zrange(key, 0, 0, withscores=True)
                if oldest_entry:
                    oldest_time = int(oldest_entry[0][1])
                    window_remaining = max(0, oldest_time + window - current_time)
                else:
                    window_remaining = 0
                
                return False, current_count, window_remaining
            
            return True, current_count, 0
            
        except Exception as e:
            logger.error("Sliding window check error", error=str(e))
            return True, 0, 0  # Fail open
    
    async def _increment_sliding_window(self, key: str, window: int):
        """Increment sliding window counter"""
        try:
            _redis = self._live_redis()  # sp1249
            if _redis is None:
                return
            current_time = time.time()

            # Add current timestamp
            await _redis.zadd(key, {str(current_time): current_time})

            # Set expiration for cleanup
            await _redis.expire(key, window * 2)
            
        except Exception as e:
            logger.error("Sliding window increment error", error=str(e))
    
    async def _get_sliding_window_status(self, key: str, window: int) -> Tuple[int, int]:
        """Get current sliding window status"""
        try:
            _redis = self._live_redis()  # sp1249
            if _redis is None:
                return 0, 0
            current_time = int(time.time())
            window_start = current_time - window

            # Remove old entries
            await _redis.zremrangebyscore(key, 0, window_start)

            # Get current count
            current_count = await _redis.zcard(key)

            # Calculate window remaining
            oldest_entry = await _redis.zrange(key, 0, 0, withscores=True)
            if oldest_entry:
                oldest_time = int(oldest_entry[0][1])
                window_remaining = max(0, oldest_time + window - current_time)
            else:
                window_remaining = 0
            
            return current_count, window_remaining
            
        except Exception as e:
            logger.error("Sliding window status error", error=str(e))
            return 0, 0
    
    async def _get_ip_reputation(self, ip: str) -> float:
        """Get IP reputation score (0-1, higher is better)"""
        try:
            _redis = self._live_redis()  # sp1249
            if _redis is None:
                return 1.0

            reputation_key = f"ip_reputation:{ip}"
            score = await _redis.get(reputation_key)

            if score:
                return max(0.0, min(1.0, float(score)))
            else:
                # New IP starts with neutral reputation
                await _redis.setex(reputation_key, 86400, "0.7")  # 24 hour TTL
                return 0.7
            
        except Exception as e:
            logger.error("IP reputation check error", error=str(e))
            return 1.0  # Assume good reputation on error
    
    async def _update_ip_reputation(self, ip: str, delta: float):
        """Update IP reputation score"""
        try:
            _redis = self._live_redis()  # sp1249
            if _redis is None:
                return

            reputation_key = f"ip_reputation:{ip}"
            current_score = await self._get_ip_reputation(ip)
            new_score = max(0.0, min(1.0, current_score + delta))

            await _redis.setex(reputation_key, 86400, str(new_score))
            
            # Log significant reputation changes
            if abs(delta) >= 0.1:
                logger.info("IP reputation updated",
                           ip=ip,
                           old_score=current_score,
                           new_score=new_score,
                           delta=delta)
                           
        except Exception as e:
            logger.error("IP reputation update error", error=str(e))
    
    async def _load_ip_lists(self):
        """Load whitelist and blacklist from Redis"""
        try:
            _redis = self._live_redis()  # sp1249
            if _redis is None:
                return

            # Load whitelist
            whitelist = await _redis.smembers("rate_limit:whitelist")
            if whitelist:
                self.whitelisted_ips = {ip.decode() if isinstance(ip, bytes) else ip for ip in whitelist}

            # Load blacklist
            blacklist = await _redis.smembers("rate_limit:blacklist")
            if blacklist:
                self.blacklisted_ips = {ip.decode() if isinstance(ip, bytes) else ip for ip in blacklist}

            # Load temporary blacklisted IPs
            temp_blacklist_keys = await _redis.keys("rate_limit:blacklist:*")
            for key in temp_blacklist_keys:
                ip = key.decode().split(":")[-1] if isinstance(key, bytes) else key.split(":")[-1]
                self.blacklisted_ips.add(ip)
            
        except Exception as e:
            logger.error("Failed to load IP lists", error=str(e))
    
    async def _log_rate_limit_exceeded(
        self,
        rule: RateLimitRule,
        identifier: str,
        endpoint: str,
        current_count: int,
        client_info: Optional[Dict[str, Any]]
    ):
        """Log rate limit exceeded event"""
        await audit_logger.log_security_event(
            "rate_limit_exceeded",
            {
                "rule": rule.name,
                "limit_type": rule.limit_type.value,
                "identifier": identifier,
                "endpoint": endpoint,
                "current_count": current_count,
                "limit": rule.requests,
                "window": rule.window
            },
            client_info
        )
    
    async def _log_blocked_request(
        self,
        ip: str,
        endpoint: str,
        reason: str,
        client_info: Optional[Dict[str, Any]]
    ):
        """Log blocked request event"""
        await audit_logger.log_security_event(
            "request_blocked",
            {
                "ip": ip,
                "endpoint": endpoint,
                "reason": reason
            },
            client_info
        )


# Global rate limiter instance
rate_limiter = RateLimiter()