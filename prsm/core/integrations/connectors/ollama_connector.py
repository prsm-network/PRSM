"""
Ollama Connector
================

Integration connector for Ollama - local LLM runtime platform.
Provides access to locally hosted language models through Ollama API.

Features:
- Local model discovery and management
- Direct API communication with Ollama instance
- Model pulling and deployment
- Chat and completion interfaces
- Performance monitoring for local models

Note: Requires Ollama to be installed and running locally.
"""

import ipaddress
import json
import logging
import os
import socket
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin, urlparse

import aiohttp
from pydantic import BaseModel, Field

from prsm.core.integrations.core.base_connector import BaseConnector, ConnectorStatus
from ..models.integration_models import (
    IntegrationPlatform, ConnectorConfig, IntegrationSource
)

logger = logging.getLogger(__name__)

# sp1268 — cloud metadata endpoints (the classic SSRF target).
_METADATA_IPS = {"169.254.169.254", "fd00:ec2::254"}


def assert_safe_outbound_url(url: str) -> None:
    """SSRF guard for a user-supplied connector base URL. Raises ValueError if `url` is not a
    safe outbound target.

    Policy: scheme must be http/https; the host is RESOLVED and EVERY resolved address is
    checked. Link-local (incl. 169.254.169.254 cloud metadata), reserved, multicast and
    unspecified addresses are ALWAYS rejected; private (LAN) addresses are rejected unless the
    host is allowlisted via PRSM_CONNECTOR_ALLOW_INTERNAL_HOSTS (comma-separated); loopback is
    allowed (a local model server is the connector's legitimate default). Resolving on each
    call (callers re-invoke before requests) defeats DNS-rebinding.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"connector URL scheme must be http/https, got {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise ValueError("connector URL has no host")

    allowlist = {h.strip().lower() for h in
                 os.getenv("PRSM_CONNECTOR_ALLOW_INTERNAL_HOSTS", "").split(",") if h.strip()}
    host_allowed = host.lower() in allowlist

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise ValueError(f"connector URL host {host!r} does not resolve: {e}")

    for info in infos:
        ip_str = info[4][0].split("%", 1)[0]  # strip any IPv6 zone id
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            raise ValueError(f"connector URL host {host!r} resolved to a non-IP {ip_str!r}")
        # Loopback (127.0.0.0/8, ::1) is the connector's legitimate default (a local model
        # server) — allow it before any block check (IPv6 ::1 also classifies as reserved).
        if addr.is_loopback:
            continue
        # Always-blocked classes (never a legitimate outbound target).
        if (addr.is_link_local or addr.is_reserved or addr.is_multicast
                or addr.is_unspecified or ip_str in _METADATA_IPS):
            raise ValueError(
                f"connector URL host {host!r} resolves to a blocked address {ip_str} "
                f"(link-local/metadata/reserved)")
        # Private LAN: needs an explicit allowlist entry.
        if addr.is_private and not host_allowed:
            raise ValueError(
                f"connector URL host {host!r} resolves to a private address {ip_str}; add it "
                f"to PRSM_CONNECTOR_ALLOW_INTERNAL_HOSTS to permit an internal target")


class OllamaModelInfo(BaseModel):
    """Ollama model information"""
    name: str
    tag: str = "latest"
    size: int = 0
    digest: str = ""
    modified_at: Optional[datetime] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class OllamaConnector(BaseConnector):
    """
    Connector for Ollama local LLM platform
    
    Provides integration with locally running Ollama instance for:
    - Model discovery and management
    - Local model deployment
    - Direct inference capabilities
    """
    
    def __init__(self, config: ConnectorConfig):
        super().__init__(config)
        self.platform = IntegrationPlatform.OLLAMA
        
        # Ollama configuration
        self.base_url = config.custom_settings.get("base_url", "http://localhost:11434")
        # sp1268 — SSRF guard: reject a base_url that targets cloud metadata / internal hosts
        # at construction (the connector registration path is reachable by untrusted input).
        assert_safe_outbound_url(self.base_url)
        self.timeout = config.custom_settings.get("timeout", 30)
        self.api_version = config.custom_settings.get("api_version", "v1")
        
        # Connection state
        self.ollama_version = None
        self.available_models = []
        self.running_models = []
        
        # Session for HTTP requests
        self.session = None
        
        logger.info(f"Initialized Ollama connector for {self.base_url}")
    
    async def initialize(self) -> bool:
        """Initialize the Ollama connector"""
        try:
            self.status = ConnectorStatus.INITIALIZING
            
            # Create HTTP session
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"Content-Type": "application/json"}
            )
            
            # Test connection to Ollama
            if await self._test_connection():
                self.status = ConnectorStatus.CONNECTED
                await self._load_initial_data()
                logger.info("Ollama connector initialized successfully")
                return True
            else:
                self.status = ConnectorStatus.DISCONNECTED
                logger.error("Failed to connect to Ollama instance")
                return False
                
        except Exception as e:
            self.status = ConnectorStatus.ERROR
            self.last_error = str(e)
            logger.error(f"Ollama connector initialization failed: {e}")
            return False
    
    async def cleanup(self):
        """Clean up connector resources"""
        if self.session:
            await self.session.close()
            self.session = None
        
        logger.info("Ollama connector cleaned up")
    
    async def authenticate(self) -> bool:
        """
        Authenticate with Ollama (no auth required for local instance)
        """
        try:
            # Ollama typically doesn't require authentication for local access
            # Just verify we can reach the API
            if await self._test_connection():
                self.authenticated_user = "local"
                logger.info("Ollama authentication successful (local access)")
                return True
            else:
                logger.error("Failed to authenticate with Ollama")
                return False
                
        except Exception as e:
            logger.error(f"Ollama authentication error: {e}")
            return False
    
    async def _test_connection(self) -> bool:
        """Test connection to Ollama instance"""
        try:
            response = await self._make_api_request("/api/version")
            if response:
                self.ollama_version = response.get("version", "unknown")
                logger.info(f"Connected to Ollama version: {self.ollama_version}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Ollama connection test failed: {e}")
            return False
    
    async def _load_initial_data(self):
        """Load initial data from Ollama"""
        try:
            # Load available models
            await self._refresh_available_models()
            
            # Load running models
            await self._refresh_running_models()
            
            logger.info(f"Loaded {len(self.available_models)} available models")
            
        except Exception as e:
            logger.warning(f"Failed to load initial Ollama data: {e}")
    
    async def _refresh_available_models(self):
        """Refresh list of available models"""
        try:
            response = await self._make_api_request("/api/tags")
            if response and "models" in response:
                self.available_models = []
                for model_data in response["models"]:
                    model_info = OllamaModelInfo(
                        name=model_data.get("name", ""),
                        tag=model_data.get("name", "").split(":")[-1] if ":" in model_data.get("name", "") else "latest",
                        size=model_data.get("size", 0),
                        digest=model_data.get("digest", ""),
                        modified_at=self._parse_datetime(model_data.get("modified_at")),
                        details=model_data.get("details", {})
                    )
                    self.available_models.append(model_info)
                    
        except Exception as e:
            logger.error(f"Failed to refresh available models: {e}")
    
    async def _refresh_running_models(self):
        """Refresh list of currently running models"""
        try:
            response = await self._make_api_request("/api/ps")
            if response and "models" in response:
                self.running_models = response["models"]
                
        except Exception as e:
            logger.error(f"Failed to refresh running models: {e}")
    
    async def search_content(
        self,
        query: str,
        content_type: str = "model",
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[IntegrationSource]:
        """
        Search for Ollama models (local search only)
        """
        try:
            await self._refresh_available_models()
            
            results = []
            query_lower = query.lower()
            
            for model in self.available_models:
                # Simple text matching
                if (query_lower in model.name.lower() or 
                    query_lower in str(model.details).lower()):
                    
                    # Create integration source
                    source = IntegrationSource(
                        platform=IntegrationPlatform.OLLAMA,
                        external_id=model.name,
                        display_name=model.name.split(":")[0],  # Remove tag
                        owner_id="local",
                        url=f"{self.base_url}/api/show?name={model.name}",
                        description=f"Local Ollama model - {model.name}",
                        metadata={
                            "size": model.size,
                            "size_gb": round(model.size / (1024**3), 2) if model.size > 0 else 0,
                            "tag": model.tag,
                            "digest": model.digest[:12] + "..." if model.digest else "",
                            "modified_at": model.modified_at.isoformat() if model.modified_at else None,
                            "details": model.details,
                            "type": "local_model"
                        }
                    )
                    results.append(source)
                    
                    if len(results) >= limit:
                        break
            
            logger.info(f"Found {len(results)} Ollama models matching '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Ollama search failed: {e}")
            return []
    
    async def get_content_metadata(self, external_id: str) -> Dict[str, Any]:
        """
        Get detailed metadata for an Ollama model
        """
        try:
            # Get model info
            response = await self._make_api_request(f"/api/show", method="POST", data={"name": external_id})
            
            if not response:
                return {"error": f"Model {external_id} not found"}
            
            # Parse model information
            model_data = {
                "type": "local_model",
                "name": external_id,
                "platform": "ollama",
                "base_url": self.base_url,
                "template": response.get("template", ""),
                "parameters": response.get("parameters", {}),
                "model_info": response.get("model_info", {}),
                "details": response.get("details", {}),
                "license": response.get("license", ""),
                "system": response.get("system", ""),
                "size": 0,
                "modified_at": None
            }
            
            # Find model in available models for additional info
            for model in self.available_models:
                if model.name == external_id:
                    model_data.update({
                        "size": model.size,
                        "size_gb": round(model.size / (1024**3), 2) if model.size > 0 else 0,
                        "tag": model.tag,
                        "digest": model.digest,
                        "modified_at": model.modified_at.isoformat() if model.modified_at else None
                    })
                    break
            
            return model_data
            
        except Exception as e:
            logger.error(f"Failed to get Ollama model metadata: {e}")
            return {"error": str(e)}
    
    async def validate_license(self, external_id: str) -> Dict[str, Any]:
        """
        Validate license for Ollama model
        Note: Local models may have various licenses
        """
        try:
            metadata = await self.get_content_metadata(external_id)
            
            if "error" in metadata:
                return {
                    "type": "unknown",
                    "compliant": False,
                    "issues": [metadata["error"]]
                }
            
            license_text = metadata.get("license", "").lower()
            
            # Analyze license if available
            if not license_text:
                return {
                    "type": "unknown",
                    "compliant": True,  # Local models assumed OK
                    "issues": ["No license information available for local model"],
                    "note": "Local model - license compliance assumed"
                }
            
            # Simple license detection
            if any(term in license_text for term in ["mit", "apache", "bsd"]):
                license_type = "permissive"
                compliant = True
                issues = []
            elif any(term in license_text for term in ["gpl", "general public license", "copyleft"]):
                license_type = "copyleft"
                compliant = False
                issues = ["GPL/Copyleft license may have usage restrictions"]
            else:
                license_type = "custom"
                compliant = True  # Local models assumed OK
                issues = ["Custom license - review recommended"]
            
            return {
                "type": license_type,
                "compliant": compliant,
                "issues": issues,
                "license_text": metadata.get("license", ""),
                "note": "Local Ollama model"
            }
            
        except Exception as e:
            logger.error(f"Ollama license validation failed: {e}")
            return {
                "type": "unknown",
                "compliant": False,
                "issues": [str(e)]
            }
    
    async def download_content(
        self,
        external_id: str,
        target_path: str,
        progress_callback: Optional[callable] = None
    ) -> bool:
        """
        'Download' Ollama model (actually just ensure it's available locally)
        """
        try:
            logger.info(f"Ensuring Ollama model {external_id} is available locally")
            
            # Check if model is already available
            await self._refresh_available_models()
            
            model_exists = any(model.name == external_id for model in self.available_models)
            
            if model_exists:
                logger.info(f"Model {external_id} already available locally")
                if progress_callback:
                    progress_callback(100, "Model already available")
                return True
            
            # If model doesn't exist, try to pull it
            logger.info(f"Pulling model {external_id} from Ollama registry")
            
            if progress_callback:
                progress_callback(10, "Starting model pull...")
            
            # Use Ollama pull API
            pull_response = await self._stream_pull_request(external_id, progress_callback)
            
            if pull_response:
                # Refresh available models
                await self._refresh_available_models()
                
                # Verify model is now available
                model_exists = any(model.name == external_id for model in self.available_models)
                
                if model_exists:
                    logger.info(f"Successfully pulled model {external_id}")
                    if progress_callback:
                        progress_callback(100, "Model pull completed")
                    return True
                else:
                    logger.error(f"Model {external_id} not found after pull")
                    return False
            else:
                logger.error(f"Failed to pull model {external_id}")
                return False
            
        except Exception as e:
            logger.error(f"Ollama download failed: {e}")
            if progress_callback:
                progress_callback(-1, f"Error: {str(e)}")
            return False
    
    async def _stream_pull_request(self, model_name: str, progress_callback: Optional[callable] = None) -> bool:
        """Stream model pull request with progress updates"""
        try:
            assert_safe_outbound_url(self.base_url)  # sp1268 — SSRF guard (re-resolve)
            url = f"{self.base_url}/api/pull"
            data = {"name": model_name, "stream": True}
            
            async with self.session.post(url, json=data) as response:
                if response.status != 200:
                    logger.error(f"Pull request failed with status {response.status}")
                    return False
                
                total_size = 0
                downloaded_size = 0
                
                async for line in response.content:
                    try:
                        line_data = json.loads(line.decode().strip())
                        
                        # Parse progress information
                        if "total" in line_data:
                            total_size = line_data["total"]
                        
                        if "completed" in line_data:
                            downloaded_size = line_data["completed"]
                        
                        # Calculate progress percentage
                        if total_size > 0:
                            progress = int((downloaded_size / total_size) * 90) + 10  # 10-100%
                            status = line_data.get("status", "Downloading...")
                            
                            if progress_callback:
                                progress_callback(progress, status)
                        
                        # Check for completion
                        if line_data.get("status") == "success":
                            return True
                            
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.warning(f"Error parsing pull progress: {e}")
                        continue
                
                return False
                
        except Exception as e:
            logger.error(f"Stream pull request failed: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Ollama connector"""
        start_time = time.time()
        
        try:
            # Test basic connectivity
            version_response = await self._make_api_request("/api/version")
            response_time = (time.time() - start_time) * 1000
            
            if version_response:
                # Update version from response
                self.ollama_version = version_response.get("version", self.ollama_version)

                # Get model count
                await self._refresh_available_models()
                await self._refresh_running_models()

                health_data = {
                    "status": "healthy",
                    "platform": self.platform,
                    "response_time": round(response_time, 2),
                    "ollama_version": self.ollama_version,
                    "base_url": self.base_url,
                    "available_models": len(self.available_models),
                    "running_models": len(self.running_models),
                    "last_check": datetime.now(timezone.utc).isoformat(),
                    "capabilities": [
                        "local_inference",
                        "model_management", 
                        "streaming_responses",
                        "model_pulling"
                    ]
                }
                
                self.status = ConnectorStatus.CONNECTED
                return health_data
            else:
                self.status = ConnectorStatus.DISCONNECTED
                return {
                    "status": "unhealthy",
                    "platform": self.platform,
                    "error": "Failed to connect to Ollama API",
                    "base_url": self.base_url,
                    "last_check": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            self.status = ConnectorStatus.ERROR
            self.error_count += 1
            
            return {
                "status": "error",
                "platform": self.platform,
                "error": str(e),
                "error_count": self.error_count,
                "base_url": self.base_url,
                "last_check": datetime.now(timezone.utc).isoformat()
            }
    
    async def _make_api_request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Make API request to Ollama"""
        if not self.session:
            logger.error("HTTP session not initialized")
            return None

        # sp1268 — re-validate before each request (re-resolves the host → defeats DNS
        # rebinding between construction and now).
        try:
            assert_safe_outbound_url(self.base_url)
        except ValueError as exc:
            logger.error(f"Refusing Ollama request to unsafe base_url: {exc}")
            return None

        url = urljoin(self.base_url, endpoint)

        try:
            request_start = time.time()
            
            async with self.session.request(
                method=method,
                url=url,
                json=data if method != "GET" else None,
                params=params
            ) as response:
                
                request_time = (time.time() - request_start) * 1000
                self.total_requests += 1
                
                if response.status == 200:
                    self.successful_requests += 1
                    self.average_response_time = (
                        (self.average_response_time * (self.successful_requests - 1) + request_time) 
                        / self.successful_requests
                    )
                    
                    result = await response.json()
                    return result
                else:
                    self.failed_requests += 1
                    logger.error(f"Ollama API request failed: {response.status} - {await response.text()}")
                    return None
                    
        except Exception as e:
            self.failed_requests += 1
            logger.error(f"Ollama API request error: {e}")
            return None
    
    def _parse_datetime(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime string"""
        if not date_str:
            return None
        
        try:
            # Try different datetime formats
            for fmt in [
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ", 
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S"
            ]:
                try:
                    return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            
            logger.warning(f"Could not parse datetime: {date_str}")
            return None
            
        except Exception as e:
            logger.error(f"Datetime parsing error: {e}")
            return None
    
    # === Extended Ollama-specific Methods ===
    
    async def list_available_models(self) -> List[OllamaModelInfo]:
        """Get list of all available Ollama models"""
        await self._refresh_available_models()
        return self.available_models
    
    async def list_running_models(self) -> List[Dict[str, Any]]:
        """Get list of currently running models"""
        await self._refresh_running_models()
        return self.running_models
    
    async def pull_model(self, model_name: str, progress_callback: Optional[callable] = None) -> bool:
        """Pull a model from Ollama registry"""
        return await self.download_content(model_name, "", progress_callback)
    
    async def remove_model(self, model_name: str) -> bool:
        """Remove a model from local storage"""
        try:
            response = await self._make_api_request(
                "/api/delete",
                method="DELETE",
                data={"name": model_name}
            )
            
            if response is not None:  # 200 response with empty body is success
                logger.info(f"Successfully removed model {model_name}")
                await self._refresh_available_models()
                return True
            else:
                logger.error(f"Failed to remove model {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Model removal failed: {e}")
            return False
    
    async def generate_completion(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """Generate completion using Ollama model"""
        try:
            data = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                **kwargs
            }
            
            if stream:
                return self._stream_completion(data)
            else:
                response = await self._make_api_request("/api/generate", method="POST", data=data)
                return response or {}
                
        except Exception as e:
            logger.error(f"Completion generation failed: {e}")
            return {}
    
    async def _stream_completion(self, data: Dict[str, Any]):
        """Stream completion responses"""
        try:
            url = f"{self.base_url}/api/generate"
            
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    async for line in response.content:
                        try:
                            chunk = json.loads(line.decode().strip())
                            yield chunk
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                        
        except Exception as e:
            logger.error(f"Streaming completion failed: {e}")
            yield {"error": str(e)}
    
    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """Generate chat completion using Ollama model"""
        try:
            data = {
                "model": model,
                "messages": messages,
                "stream": stream,
                **kwargs
            }
            
            if stream:
                return self._stream_chat_completion(data)
            else:
                response = await self._make_api_request("/api/chat", method="POST", data=data)
                return response or {}
                
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            return {}
    
    async def _stream_chat_completion(self, data: Dict[str, Any]):
        """Stream chat completion responses"""
        try:
            url = f"{self.base_url}/api/chat"
            
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    async for line in response.content:
                        try:
                            chunk = json.loads(line.decode().strip())
                            yield chunk
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                        
        except Exception as e:
            logger.error(f"Streaming chat completion failed: {e}")
            yield {"error": str(e)}