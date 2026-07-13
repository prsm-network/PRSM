"""
Enhanced Security Sandbox Manager
=================================

Comprehensive security sandboxing for both external content validation and
MCP tool execution. Provides secure isolation environments with resource
limits, permission controls, and comprehensive monitoring.

Key Features:
- Isolated execution environment for untrusted content
- MCP tool execution sandboxing with resource limits
- License compliance validation
- Vulnerability scanning and risk assessment
- Real-time resource monitoring and threat detection
- Fine-grained permission management with user consent
- Integration with PRSM's circuit breaker system
- Performance monitoring and audit logging
- Automatic cleanup and resource management

Sandbox Types:
- Content scanning for external integrations
- Tool execution for MCP protocol tools
- Multi-level isolation (basic, container, VM)
"""

import asyncio
import atexit
import os
import re
import tempfile
import shutil
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from uuid import UUID, uuid4
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel, Field

from ..models.integration_models import SecurityRisk, LicenseType, SecurityScanResult
from prsm.core.config import settings
from .audit_logger import audit_logger

logger = structlog.get_logger(__name__)


class SandboxStatus(str, Enum):
    """Sandbox operational status"""
    IDLE = "idle"
    ACTIVE = "active"
    SCANNING = "scanning"
    QUARANTINED = "quarantined"
    ERROR = "error"


class SandboxResultStatus(str, Enum):
    """Sandbox operation result statuses"""
    APPROVED = "approved"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"
    ERROR = "error"


@dataclass
class SandboxResult:
    """Result from sandbox execution"""
    success: bool = True
    output: str = ""
    error_output: str = ""
    exit_code: int = 0
    status: str = "completed"


# ── sp1443 (compute-sandbox audit, Findings 1+2) — untrusted-execution safety ──
# execute_safely's subprocess backend runs the guest as a PLAIN child of the daemon user: no
# network namespace, no seccomp, no user namespace, no chroot. It provides NO confidentiality or
# network isolation (the block_network / allowed_domains flags are advisory JSON this code never
# reads). Executing untrusted downloaded content there is host-secret exfiltration + SSRF + RCE, so
# execution is FAIL-CLOSED: it runs ONLY when the operator explicitly opts in, and even then it is
# resource-bounded (rlimits + its own session so a runaway process GROUP can be reaped).

def _unisolated_exec_enabled() -> bool:
    """Opt-in gate for running untrusted content in the (un-isolated) subprocess backend. OFF by
    default — an operator must set PRSM_SANDBOX_ALLOW_UNISOLATED_EXEC only when this process is
    itself wrapped in REAL external isolation (a --net=none container / nsjail), since this backend
    gives no network or filesystem isolation on its own."""
    return (os.environ.get("PRSM_SANDBOX_ALLOW_UNISOLATED_EXEC", "") or "").strip().lower() in (
        "1", "true", "yes", "on")


def _sandbox_mem_limit_bytes() -> int:
    # RLIMIT_AS caps VIRTUAL address space; a modern interpreter maps several GB of virtual space at
    # startup (especially on macOS), so the cap must be generous enough to START yet still bound a
    # runaway allocation. 2 GiB default, env-tunable, floored at 512 MiB.
    try:
        return max(512, int(os.environ.get("PRSM_SANDBOX_MEM_LIMIT_MB", "2048"))) * 1024 * 1024
    except (ValueError, TypeError):
        return 2048 * 1024 * 1024


def _sandbox_preexec(cpu_seconds: int, mem_bytes: int):
    """Build a POSIX preexec_fn: start a NEW SESSION so os.killpg can reap the whole process GROUP on
    timeout — a forked grandchild otherwise survives as a runaway (Finding 2) — and clamp the
    PER-PROCESS rlimits (CPU time, address space) so a runaway CPU loop / alloc bomb is bounded.

    Deliberately does NOT set RLIMIT_NPROC / RLIMIT_NOFILE: those are per-USER / inherited limits, so
    an absolute low value breaks whenever the host already runs many of the user's processes or holds
    many fds (it capped the child before it could exec). The session + killpg-on-timeout is the real
    runaway defense; a fork bomb is bounded by the timeout reaping the whole group. (A hard pids cap
    belongs in the external container/cgroup this opt-in path must run behind anyway.)"""
    def _pre():  # runs in the forked child, before exec
        try:
            os.setsid()
        except OSError:
            pass
        try:
            import resource as _resource
        except ImportError:
            return
        for name, limit in (
            ("RLIMIT_CPU", (int(cpu_seconds) + 1, int(cpu_seconds) + 1)),
            ("RLIMIT_AS", (int(mem_bytes), int(mem_bytes))),
        ):
            r = getattr(_resource, name, None)
            if r is not None:
                try:
                    _resource.setrlimit(r, limit)
                except (ValueError, OSError):
                    pass
    return _pre


# ===== MCP Tool Execution Sandbox Classes =====

class ToolSandboxType(str, Enum):
    """Types of security sandboxes for tool execution"""
    NONE = "none"              # No sandboxing (for testing only)
    BASIC = "basic"            # Basic process isolation
    CONTAINER = "container"    # Docker container isolation
    VM = "vm"                  # Virtual machine isolation (future)


class ResourceType(str, Enum):
    """Types of system resources to monitor and limit"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    PROCESSES = "processes"
    FILES = "files"
    TIME = "time"


class ToolSecurityLevel(str, Enum):
    """Security levels for tool execution (duplicated here for self-contained module)"""
    SAFE = "safe"
    RESTRICTED = "restricted"
    PRIVILEGED = "privileged"
    DANGEROUS = "dangerous"


@dataclass
class ResourceLimits:
    """Enhanced resource limits for sandbox execution with strict security controls"""
    # CPU limits (more restrictive)
    cpu_percent: float = 25.0          # Maximum CPU usage percentage (reduced from 50%)
    cpu_time_seconds: float = 15.0     # Maximum CPU time (reduced from 30s)
    cpu_burst_limit: float = 50.0      # Maximum burst CPU for short periods
    cpu_burst_duration: float = 2.0    # Maximum burst duration in seconds
    
    # Memory limits (stricter enforcement)
    memory_mb: int = 256               # Maximum memory in MB (reduced from 512MB)
    virtual_memory_mb: int = 512       # Maximum virtual memory in MB (reduced from 1024MB)
    memory_swap_mb: int = 0            # Disable swap to prevent resource exhaustion
    memory_kill_threshold: float = 0.9 # Kill at 90% of limit to prevent OOM
    
    # Disk limits (enhanced controls)
    disk_read_mb: int = 50             # Maximum disk read in MB (reduced from 100MB)
    disk_write_mb: int = 25            # Maximum disk write in MB (reduced from 50MB)
    temp_space_mb: int = 50            # Maximum temporary space in MB (reduced from 100MB)
    disk_iops_limit: int = 1000        # Maximum disk IOPS per second
    disk_bandwidth_mbps: float = 10.0  # Maximum disk bandwidth in MB/s
    
    # Network limits (tighter restrictions)
    network_upload_mb: int = 5         # Maximum network upload in MB (reduced from 10MB)
    network_download_mb: int = 25      # Maximum network download in MB (reduced from 50MB)
    network_connections: int = 3       # Maximum concurrent connections (reduced from 5)
    network_bandwidth_kbps: int = 1024 # Maximum network bandwidth in KB/s
    network_timeout: float = 10.0      # Network operation timeout
    
    # Process limits (enhanced security)
    max_processes: int = 5             # Maximum number of processes (reduced from 10)
    max_threads: int = 10              # Maximum number of threads (reduced from 20)
    max_file_descriptors: int = 50     # Maximum file descriptors (reduced from 100)
    max_child_processes: int = 2       # Maximum child processes
    process_priority: int = 19         # Lowest process priority (nice value)
    
    # Time limits (stricter timeouts)
    execution_timeout: float = 30.0    # Maximum execution time in seconds (reduced from 60s)
    idle_timeout: float = 5.0          # Maximum idle time in seconds (reduced from 10s)
    startup_timeout: float = 10.0      # Maximum startup time
    cleanup_timeout: float = 5.0       # Maximum cleanup time
    
    # Enhanced security limits
    syscall_whitelist: List[str] = None        # Allowed system calls (None = all allowed)
    disable_network: bool = False              # Completely disable network access
    read_only_filesystem: bool = False         # Mount filesystem as read-only
    disable_ptrace: bool = True               # Disable process tracing
    disable_core_dumps: bool = True           # Disable core dump generation
    umask: int = 0o077                        # Restrictive file creation mask
    
    # Resource monitoring
    monitor_interval: float = 0.5              # Monitoring check interval in seconds
    violation_threshold: int = 3               # Violations before termination
    resource_check_frequency: int = 10         # Resource checks per second
    
    # Emergency limits (triggered during security incidents)
    emergency_mode: bool = False
    emergency_cpu_percent: float = 10.0       # Emergency CPU limit
    emergency_memory_mb: int = 128             # Emergency memory limit
    emergency_timeout: float = 15.0            # Emergency execution timeout


@dataclass
class SecurityPermissions:
    """Security permissions for tool execution"""
    # File system permissions
    file_read: Set[str] = field(default_factory=set)      # Allowed read paths
    file_write: Set[str] = field(default_factory=set)     # Allowed write paths
    file_execute: Set[str] = field(default_factory=set)   # Allowed execute paths
    
    # Network permissions
    network_outbound: bool = False     # Allow outbound network access
    network_inbound: bool = False      # Allow inbound network access
    allowed_hosts: Set[str] = field(default_factory=set)  # Allowed host patterns
    allowed_ports: Set[int] = field(default_factory=set)  # Allowed port numbers
    
    # System permissions
    system_calls: Set[str] = field(default_factory=set)   # Allowed system calls
    environment_vars: Set[str] = field(default_factory=set)  # Allowed env vars
    
    # Tool-specific permissions
    tool_permissions: Set[str] = field(default_factory=set)  # Tool-specific perms
    user_consent_required: bool = True  # Require explicit user consent


class ToolExecutionRequest(BaseModel):
    """Request for tool execution (simplified for sandbox)"""
    execution_id: UUID = Field(default_factory=uuid.uuid4)
    tool_id: str
    tool_action: str
    parameters: Dict[str, Any]
    user_id: str
    permissions: List[str] = Field(default_factory=list)
    sandbox_level: ToolSecurityLevel = ToolSecurityLevel.RESTRICTED
    timeout_seconds: float = 30.0


class ToolExecutionResult(BaseModel):
    """Result from tool execution"""
    execution_id: UUID
    tool_id: str
    success: bool
    result_data: Any = None
    execution_time: float
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    security_violations: List[str] = Field(default_factory=list)
    resource_violations: List[str] = Field(default_factory=list)


class ToolSandboxContext(BaseModel):
    """Context for tool sandbox execution"""
    sandbox_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_id: str
    user_id: str
    execution_request: ToolExecutionRequest
    
    # Security configuration
    sandbox_type: ToolSandboxType = ToolSandboxType.CONTAINER
    security_level: ToolSecurityLevel
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    permissions: SecurityPermissions = Field(default_factory=SecurityPermissions)
    
    # Execution metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Monitoring data
    resource_usage: Dict[str, Any] = Field(default_factory=dict)
    security_violations: List[str] = Field(default_factory=list)
    audit_events: List[Dict[str, Any]] = Field(default_factory=list)


class ToolSandboxResult(BaseModel):
    """Result of tool sandbox execution"""
    sandbox_id: str
    tool_execution_result: ToolExecutionResult
    
    # Security metrics
    security_violations: List[str] = Field(default_factory=list)
    resource_violations: List[str] = Field(default_factory=list)
    permission_violations: List[str] = Field(default_factory=list)
    
    # Resource usage
    peak_cpu_percent: float = 0.0
    peak_memory_mb: float = 0.0
    total_disk_read_mb: float = 0.0
    total_disk_write_mb: float = 0.0
    total_network_mb: float = 0.0
    execution_time: float = 0.0
    
    # Audit information
    audit_trail: List[Dict[str, Any]] = Field(default_factory=list)
    compliance_status: str = "compliant"


class ResourceMonitor:
    """Real-time resource monitoring for sandbox execution"""
    
    def __init__(self, context: ToolSandboxContext):
        self.context = context
        self.monitoring = False
        self.violations: List[str] = []
        self.usage_data: Dict[str, List[float]] = {
            "cpu": [],
            "memory": [],
            "disk_read": [],
            "disk_write": [],
            "network": []
        }
        self.start_time = time.time()
    
    async def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.start_time = time.time()
        
        logger.info("Starting resource monitoring",
                   sandbox_id=self.context.sandbox_id,
                   limits=self.context.resource_limits.__dict__)
        
        # Start monitoring loop
        monitoring_task = asyncio.create_task(self._monitoring_loop())
        return monitoring_task
    
    async def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return usage summary"""
        self.monitoring = False
        
        summary = {
            "execution_time": time.time() - self.start_time,
            "peak_cpu": max(self.usage_data["cpu"]) if self.usage_data["cpu"] else 0.0,
            "peak_memory": max(self.usage_data["memory"]) if self.usage_data["memory"] else 0.0,
            "total_disk_read": sum(self.usage_data["disk_read"]),
            "total_disk_write": sum(self.usage_data["disk_write"]),
            "total_network": sum(self.usage_data["network"]),
            "violations": self.violations
        }
        
        logger.info("Resource monitoring stopped",
                   sandbox_id=self.context.sandbox_id,
                   summary=summary)
        
        return summary
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self.monitoring:
                await self._check_resources()
                await asyncio.sleep(0.5)  # Monitor every 500ms
        except Exception as e:
            logger.error("Resource monitoring failed",
                        sandbox_id=self.context.sandbox_id,
                        error=str(e))
    
    async def _check_resources(self):
        """Enhanced resource checking with strict enforcement"""
        try:
            # Get current resource usage (in production, use psutil or similar)
            current_cpu = await self._get_cpu_usage()
            current_memory = await self._get_memory_usage()
            current_disk_read = await self._get_disk_read()
            current_disk_write = await self._get_disk_write()
            current_network = await self._get_network_usage()
            current_processes = await self._get_process_count()
            current_file_descriptors = await self._get_file_descriptor_count()
            
            # Store usage data for analysis
            self.usage_data["cpu"].append(current_cpu)
            self.usage_data["memory"].append(current_memory)
            self.usage_data["disk_read"].append(current_disk_read)
            self.usage_data["disk_write"].append(current_disk_write)
            self.usage_data["network"].append(current_network)
            
            limits = self.context.resource_limits
            violations_detected = []
            
            # Enhanced CPU monitoring with burst detection
            if current_cpu > limits.cpu_percent:
                violation = f"CPU usage {current_cpu:.1f}% exceeds limit {limits.cpu_percent}%"
                violations_detected.append(violation)
                
                # Check for sustained high CPU
                recent_cpu = self.usage_data["cpu"][-10:] if len(self.usage_data["cpu"]) >= 10 else self.usage_data["cpu"]
                if len(recent_cpu) >= 5 and all(cpu > limits.cpu_percent * 0.8 for cpu in recent_cpu):
                    violations_detected.append("Sustained high CPU usage detected")
            
            # Enhanced memory monitoring with kill threshold
            if current_memory > limits.memory_mb:
                violation = f"Memory usage {current_memory:.1f}MB exceeds limit {limits.memory_mb}MB"
                violations_detected.append(violation)
                
                # Emergency termination if approaching kill threshold
                if current_memory > limits.memory_mb * limits.memory_kill_threshold:
                    violations_detected.append(f"CRITICAL: Memory usage exceeds kill threshold")
                    await self._emergency_resource_action("memory_kill_threshold")
            
            # Enhanced disk monitoring
            total_disk_read = sum(self.usage_data["disk_read"])
            total_disk_write = sum(self.usage_data["disk_write"])
            
            if total_disk_read > limits.disk_read_mb:
                violations_detected.append(f"Total disk read {total_disk_read:.1f}MB exceeds limit {limits.disk_read_mb}MB")
            
            if total_disk_write > limits.disk_write_mb:
                violations_detected.append(f"Total disk write {total_disk_write:.1f}MB exceeds limit {limits.disk_write_mb}MB")
            
            # Enhanced network monitoring
            total_network = sum(self.usage_data["network"])
            if total_network > limits.network_download_mb + limits.network_upload_mb:
                violations_detected.append(f"Total network usage {total_network:.1f}MB exceeds limits")
            
            # Process and file descriptor limits
            if current_processes > limits.max_processes:
                violations_detected.append(f"Process count {current_processes} exceeds limit {limits.max_processes}")
                await self._emergency_resource_action("process_limit")
            
            if current_file_descriptors > limits.max_file_descriptors:
                violations_detected.append(f"File descriptor count {current_file_descriptors} exceeds limit {limits.max_file_descriptors}")
            
            # Enhanced timeout checking
            elapsed_time = time.time() - self.start_time
            if elapsed_time > limits.execution_timeout:
                violations_detected.append(f"Execution time {elapsed_time:.1f}s exceeds timeout {limits.execution_timeout}s")
                await self._emergency_resource_action("timeout")
                self.monitoring = False
            
            # Check for idle timeout
            if hasattr(self, 'last_activity_time'):
                idle_time = time.time() - self.last_activity_time
                if idle_time > limits.idle_timeout:
                    violations_detected.append(f"Idle time {idle_time:.1f}s exceeds limit {limits.idle_timeout}s")
                    await self._emergency_resource_action("idle_timeout")
            
            # Log violations and take action
            if violations_detected:
                self.violations.extend(violations_detected)
                
                for violation in violations_detected:
                    logger.warning("Enhanced resource violation detected", 
                                 sandbox_id=self.context.sandbox_id,
                                 violation=violation)
                
                # Escalate if too many violations
                if len(self.violations) >= limits.violation_threshold:
                    logger.critical("Violation threshold exceeded - terminating sandbox",
                                  sandbox_id=self.context.sandbox_id,
                                  violation_count=len(self.violations))
                    await self._emergency_resource_action("violation_threshold")
                    self.monitoring = False
            
            # Update activity timestamp for idle detection
            if current_cpu > 1.0 or current_memory > 10:  # Some activity detected
                self.last_activity_time = time.time()
            
        except Exception as e:
            logger.error("Enhanced resource check failed",
                        sandbox_id=self.context.sandbox_id,
                        error=str(e))
    
    async def _emergency_resource_action(self, violation_type: str) -> None:
        """Take emergency action when critical resource violations occur"""
        try:
            logger.critical("Taking emergency resource action",
                           sandbox_id=self.context.sandbox_id,
                           violation_type=violation_type)
            
            # Send immediate signal to sandbox manager to terminate this sandbox
            if hasattr(self.context, 'emergency_callback'):
                await self.context.emergency_callback(self.context.sandbox_id, violation_type)
            
            # For memory violations, try to force garbage collection first
            if violation_type == "memory_kill_threshold":
                import gc
                gc.collect()
                
                # If still over threshold after GC, terminate
                current_memory = await self._get_memory_usage()
                if current_memory > self.context.resource_limits.memory_mb * self.context.resource_limits.memory_kill_threshold:
                    self.monitoring = False
            
            # For process violations, attempt to kill excess processes
            elif violation_type == "process_limit":
                await self._cleanup_excess_processes()
            
            # For timeout violations, immediate termination
            elif violation_type in ["timeout", "idle_timeout", "violation_threshold"]:
                self.monitoring = False
            
        except Exception as e:
            logger.error("Emergency resource action failed",
                        sandbox_id=self.context.sandbox_id,
                        violation_type=violation_type,
                        error=str(e))
    
    async def _cleanup_excess_processes(self) -> None:
        """Attempt to clean up excess processes"""
        try:
            # In production, would use psutil to identify and terminate excess processes
            logger.warning("Attempting to cleanup excess processes",
                          sandbox_id=self.context.sandbox_id)
            
            # Placeholder for process cleanup logic
            # In real implementation:
            # 1. Identify processes spawned by this sandbox
            # 2. Terminate non-essential processes first
            # 3. Force kill if necessary
            
        except Exception as e:
            logger.error("Process cleanup failed", error=str(e))
    
    async def _get_process_count(self) -> int:
        """Get current process count for sandbox"""
        try:
            # Placeholder - in production use psutil or /proc
            return 3  # Mock value
        except Exception:
            return 0
    
    async def _get_file_descriptor_count(self) -> int:
        """Get current file descriptor count"""
        try:
            # Placeholder - in production check /proc/[pid]/fd
            return 15  # Mock value  
        except Exception:
            return 0
    
    async def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        # Simulate CPU usage (in production, use psutil.cpu_percent())
        return min(25.0 + len(self.usage_data["cpu"]) * 0.1, 100.0)
    
    async def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        # Simulate memory usage (in production, use psutil.virtual_memory())
        return min(100.0 + len(self.usage_data["memory"]) * 2.0, 1024.0)
    
    async def _get_disk_read(self) -> float:
        """Get disk read in MB since last check"""
        return 0.5  # 0.5 MB per check
    
    async def _get_disk_write(self) -> float:
        """Get disk write in MB since last check"""
        return 0.2  # 0.2 MB per check
    
    async def _get_network_usage(self) -> float:
        """Get network usage in MB since last check"""
        return 0.1  # 0.1 MB per check


def _cleanup_sandbox_dir(sandbox_dir: str) -> None:
    """Sprint 483 (F23) — atexit cleanup for SandboxManager
    sandbox dirs. Fires once per process at shutdown for each
    instance's mkdtemp'd sandbox dir. Safe under repeated calls
    (ignore_errors=True; missing dir is a no-op)."""
    try:
        if sandbox_dir and os.path.isdir(sandbox_dir):
            shutil.rmtree(sandbox_dir, ignore_errors=True)
    except Exception:  # noqa: BLE001
        # atexit handlers must never raise — Python ignores
        # them but stderr gets noisy. Defensive swallow.
        pass


class SandboxManager:
    """
    Enhanced Security Sandbox Manager
    
    Provides secure isolation environments for both external content validation
    and MCP tool execution. Features comprehensive resource monitoring, 
    permission controls, and audit logging.
    
    Capabilities:
    - External content security scanning and validation
    - MCP tool execution with resource limits and isolation
    - Real-time resource monitoring and threat detection
    - Fine-grained permission management
    - Comprehensive audit logging and compliance reporting
    """
    
    def __init__(self):
        """Initialize the enhanced sandbox manager"""
        
        # Content scanning state (existing functionality)
        self.status = SandboxStatus.IDLE
        self.active_scans: Dict[UUID, Dict[str, Any]] = {}
        self.scan_history: List[SecurityScanResult] = []
        
        # Tool execution state (new functionality)
        self.active_tool_sandboxes: Dict[str, ToolSandboxContext] = {}
        self.tool_execution_history: List[ToolSandboxResult] = []
        
        # Sandbox configuration
        self.sandbox_dir = tempfile.mkdtemp(prefix="prsm_sandbox_")
        self.tool_sandbox_dir = os.path.join(self.sandbox_dir, "tools")
        # Sprint 483 (F23) — register atexit cleanup so the
        # mkdtemp'd sandbox doesn't accumulate forever in
        # /var/folders/.../T/. Pre-fix: 3 SandboxManager
        # instances per daemon startup × N restarts =
        # thousands of leaked dirs (live-verified: 4416 stale
        # `prsm_sandbox_*` dirs on a dev workstation). Each
        # holds a `quarantine` + `tools` subdir, low disk but
        # high inode pressure on macOS /var/folders.
        atexit.register(_cleanup_sandbox_dir, self.sandbox_dir)
        self.max_file_size = int(getattr(settings, "PRSM_MAX_FILE_SIZE_MB", 100)) * 1024 * 1024
        self.scan_timeout = int(getattr(settings, "PRSM_SCAN_TIMEOUT_SECONDS", 300))
        self.quarantine_dir = os.path.join(self.sandbox_dir, "quarantine")
        
        # Tool execution configuration
        self.max_concurrent_tool_sandboxes = getattr(settings, "PRSM_MAX_CONCURRENT_TOOLS", 10)
        self.default_tool_timeout = 60.0
        self.tool_cleanup_interval = 300.0  # 5 minutes
        
        # Security settings
        self.enable_vulnerability_scan = getattr(settings, "PRSM_ENABLE_VULN_SCAN", True)
        self.enable_license_scan = getattr(settings, "PRSM_ENABLE_LICENSE_SCAN", True)
        self.enable_malware_scan = getattr(settings, "PRSM_ENABLE_MALWARE_SCAN", False)
        self.enable_tool_sandboxing = getattr(settings, "PRSM_ENABLE_TOOL_SANDBOX", True)
        
        # Risk thresholds
        self.risk_thresholds = {
            "max_vulnerabilities": 5,
            "max_high_risk_vulns": 1,
            "min_license_compliance": 0.8,
            "max_file_size_ratio": 2.0
        }
        
        # Enhanced security state
        self.emergency_mode = False
        self.blocked_patterns = set()

        # Initialize sandbox environment
        self._setup_sandbox()
        
        # Defer periodic cleanup task creation until needed
        self._cleanup_task = None
        if self.enable_tool_sandboxing:
            self._start_cleanup_task()
        
        logger.info("Enhanced Security Sandbox Manager initialized",
                   sandbox_dir=self.sandbox_dir,
                   tool_sandbox_dir=self.tool_sandbox_dir,
                   vulnerability_scanning=self.enable_vulnerability_scan,
                   license_scanning=self.enable_license_scan,
                   malware_scanning=self.enable_malware_scan,
                   tool_sandboxing=self.enable_tool_sandboxing)
    
    def _start_cleanup_task(self):
        """Start periodic cleanup task if event loop is available"""
        try:
            loop = asyncio.get_running_loop()
            self._cleanup_task = loop.create_task(self._periodic_tool_cleanup())
        except RuntimeError:
            # No running event loop, task will be started later when needed
            pass
    
    # === Public Sandbox Operations ===

    async def execute_safely(self, content_path: str, metadata: Dict[str, Any]) -> SandboxResult:
        """
        Execute content safely in the sandbox environment

        Args:
            content_path: Path to content to execute
            metadata: Execution metadata

        Returns:
            SandboxResult with execution details
        """
        try:
            import subprocess as sp
            import os

            if not os.path.exists(content_path):
                return SandboxResult(
                    success=False,
                    output="",
                    error_output=f"Content path not found: {content_path}",
                    exit_code=-1,
                    status="error"
                )

            # Non-Python files are only validated (never executed).
            if not content_path.endswith('.py'):
                return SandboxResult(
                    success=True,
                    output="Content validated (non-executable)",
                    error_output="",
                    exit_code=0,
                    status="completed"
                )

            # sp1443 (Finding 1, CRITICAL) — FAIL CLOSED. This backend has NO OS isolation, so
            # running untrusted content here is host-secret exfiltration / SSRF / RCE. Skip execution
            # unless the operator explicitly opts in (behind real external isolation). Static scans
            # still gate the import; we simply refuse to RUN attacker-controlled code by default.
            if not _unisolated_exec_enabled():
                logger.warning(
                    "sandbox: refusing to EXECUTE untrusted content (no OS isolation on this "
                    "backend). Static analysis only. Set PRSM_SANDBOX_ALLOW_UNISOLATED_EXEC=1 to "
                    "enable — ONLY behind a real --net=none container / nsjail.",
                    content_path=content_path,
                )
                return SandboxResult(
                    success=True,
                    output="",
                    error_output="execution skipped: no OS isolation on this sandbox backend",
                    exit_code=0,
                    status="skipped_no_isolation",
                )

            import sys
            import signal
            cmd = [sys.executable, content_path]
            timeout = min(metadata.get("timeout", 30), self.scan_timeout)

            # sp1443 (Finding 2, HIGH) — run in a NEW SESSION with rlimits so a forked grandchild
            # can't survive the timeout as a runaway (killpg reaps the whole group) and a fork/alloc
            # bomb is bounded. Drop the host PATH/PYTHONPATH. NB: this still gives no network/fs
            # isolation — that is why the whole path is opt-in behind external containment above.
            proc = sp.Popen(
                cmd,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
                text=True,
                cwd=self.sandbox_dir,
                env={"PATH": "", "HOME": self.sandbox_dir, "PYTHONPATH": ""},
                preexec_fn=_sandbox_preexec(timeout, _sandbox_mem_limit_bytes()),
            )
            try:
                out, err = proc.communicate(timeout=timeout)
                return SandboxResult(
                    success=proc.returncode == 0,
                    output=(out or "")[:10000],
                    error_output=(err or "")[:10000],
                    exit_code=proc.returncode if proc.returncode is not None else -1,
                    status="completed",
                )
            except sp.TimeoutExpired:
                # Reap the ENTIRE process group, not just the tracked child.
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
                try:
                    proc.communicate(timeout=5)
                except Exception:  # noqa: BLE001 — best-effort reap after the kill
                    pass
                return SandboxResult(
                    success=False,
                    output="",
                    error_output=f"Execution timed out after {timeout} seconds",
                    exit_code=-1,
                    status="timeout",
                )

        except Exception as e:
            logger.error("Sandbox execution failed", error=str(e))
            return SandboxResult(
                success=False,
                output="",
                error_output=f"Sandbox execution error: {str(e)}",
                exit_code=-2,
                status="error"
            )

    async def scan_content(self, content_path: str, metadata: Dict[str, Any],
                         scan_options: Optional[Dict[str, Any]] = None) -> SecurityScanResult:
        """
        Perform comprehensive security scan on content
        
        Args:
            content_path: Path to content to scan
            metadata: Content metadata from platform
            scan_options: Optional scan configuration
            
        Returns:
            SecurityScanResult with comprehensive security assessment
        """
        scan_id = uuid4()
        start_time = datetime.now(timezone.utc)
        
        try:
            print(f"🔍 Starting security scan: {scan_id}")
            print(f"   - Content: {os.path.basename(content_path)}")
            print(f"   - Size: {os.path.getsize(content_path) if os.path.exists(content_path) else 0} bytes")
            
            # Update status
            self.status = SandboxStatus.SCANNING
            
            # Initialize scan tracking
            self.active_scans[scan_id] = {
                "content_path": content_path,
                "metadata": metadata,
                "start_time": start_time,
                "status": "running"
            }
            
            # Create scan result
            scan_result = SecurityScanResult(
                scan_id=scan_id,
                request_id=metadata.get("request_id", uuid4())
            )
            
            # Stage 1: Basic file validation
            basic_validation = await self._perform_basic_validation(content_path)
            if not basic_validation["passed"]:
                scan_result.risk_level = SecurityRisk.HIGH
                scan_result.vulnerabilities_found.extend(basic_validation["issues"])
                scan_result.approved_for_import = False
                return await self._finalize_scan(scan_id, scan_result)
            
            # Stage 2: License compliance scan
            if self.enable_license_scan:
                license_result = await self._scan_license_compliance(content_path, metadata)
                scan_result.license_compliance = license_result["type"]
                scan_result.compliance_issues.extend(license_result["issues"])
            
            # Stage 3: Vulnerability scanning
            if self.enable_vulnerability_scan:
                vuln_result = await self._scan_vulnerabilities(content_path)
                scan_result.vulnerabilities_found.extend(vuln_result["vulnerabilities"])
            
            # Stage 4: Malware scanning (if enabled)
            if self.enable_malware_scan:
                malware_result = await self._scan_malware(content_path)
                if malware_result["threats_found"]:
                    scan_result.vulnerabilities_found.extend(malware_result["threats"])
                    scan_result.risk_level = SecurityRisk.CRITICAL
            
            # Stage 5: Risk assessment
            risk_assessment = await self._assess_risk(scan_result, metadata)
            scan_result.risk_level = risk_assessment["risk_level"]
            scan_result.recommendations.extend(risk_assessment["recommendations"])
            scan_result.approved_for_import = risk_assessment["approved"]
            
            # Handle high-risk content
            if scan_result.risk_level in [SecurityRisk.HIGH, SecurityRisk.CRITICAL]:
                await self._quarantine_content(content_path, scan_result)
                
                # Log critical risks and trigger enhanced security measures
                if scan_result.risk_level == SecurityRisk.CRITICAL:
                    print(f"🚨 CRITICAL SECURITY RISK: {scan_result.vulnerabilities_found}")
                    await self._trigger_critical_security_response(scan_result)
                    # Enhanced circuit breaker integration for critical risks
                    await self._enforce_enhanced_resource_limits()
            
            print(f"🔍 Security scan completed: {scan_id}")
            print(f"   - Risk level: {scan_result.risk_level}")
            print(f"   - Approved: {scan_result.approved_for_import}")
            print(f"   - Vulnerabilities: {len(scan_result.vulnerabilities_found)}")
            
            return await self._finalize_scan(scan_id, scan_result)
            
        except Exception as e:
            print(f"❌ Security scan failed: {e}")
            
            # Create error result
            scan_result = SecurityScanResult(
                scan_id=scan_id,
                request_id=metadata.get("request_id", uuid4()),
                risk_level=SecurityRisk.HIGH,
                vulnerabilities_found=[f"Scan error: {str(e)}"],
                approved_for_import=False
            )
            
            return await self._finalize_scan(scan_id, scan_result)
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get current sandbox status and health
        
        Returns:
            Sandbox status and metrics
        """
        total_scans = len(self.scan_history)
        approved_scans = sum(1 for scan in self.scan_history if scan.approved_for_import)
        
        return {
            "status": self.status.value,
            "active_scans": len(self.active_scans),
            "total_scans": total_scans,
            "approved_scans": approved_scans,
            "approval_rate": (approved_scans / max(total_scans, 1)) * 100,
            "quarantine_count": len(os.listdir(self.quarantine_dir)) if os.path.exists(self.quarantine_dir) else 0,
            "sandbox_directory": self.sandbox_dir,
            "vulnerability_scanning": self.enable_vulnerability_scan,
            "license_scanning": self.enable_license_scan,
            "malware_scanning": self.enable_malware_scan
        }
    
    async def cleanup_sandbox(self) -> bool:
        """
        Clean up sandbox environment and temporary files
        
        Returns:
            True if cleanup successful
        """
        try:
            print("🧹 Cleaning up sandbox environment")
            
            # Clear active scans
            self.active_scans.clear()
            
            # Remove temporary files (preserve quarantine)
            for item in os.listdir(self.sandbox_dir):
                item_path = os.path.join(self.sandbox_dir, item)
                if item != "quarantine" and os.path.exists(item_path):
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
            
            self.status = SandboxStatus.IDLE
            print("✅ Sandbox cleanup completed")
            return True
            
        except Exception as e:
            print(f"❌ Sandbox cleanup failed: {e}")
            return False
    
    # === Private Security Scanning Methods ===
    
    async def _perform_basic_validation(self, content_path: str) -> Dict[str, Any]:
        """Perform basic file validation checks"""
        issues = []
        
        try:
            # Check file exists
            if not os.path.exists(content_path):
                issues.append("File does not exist")
                return {"passed": False, "issues": issues}
            
            # Check file size
            file_size = os.path.getsize(content_path)
            if file_size > self.max_file_size:
                issues.append(f"File size ({file_size} bytes) exceeds maximum ({self.max_file_size} bytes)")
            
            # Check file permissions
            if not os.access(content_path, os.R_OK):
                issues.append("File is not readable")
            
            # Check for suspicious file extensions
            suspicious_extensions = [".exe", ".bat", ".cmd", ".scr", ".pif"]
            if any(content_path.lower().endswith(ext) for ext in suspicious_extensions):
                issues.append("Suspicious file extension detected")
            
            # Basic content validation
            if os.path.isfile(content_path):
                with open(content_path, 'rb') as f:
                    header = f.read(1024)
                    
                    # Check for executable headers
                    if header.startswith(b'MZ') or header.startswith(b'\x7fELF'):
                        issues.append("Executable content detected")
                    
                    # Check for script headers
                    script_headers = [b'#!/bin/sh', b'#!/bin/bash', b'@echo off']
                    if any(header.startswith(h) for h in script_headers):
                        issues.append("Script content detected")
            
            return {"passed": len(issues) == 0, "issues": issues}
            
        except Exception as e:
            return {"passed": False, "issues": [f"Validation error: {str(e)}"]}
    
    async def _scan_license_compliance(self, content_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Scan for license compliance"""
        try:
            # Extract license information from metadata
            license_info = metadata.get("license", {})
            
            if isinstance(license_info, dict):
                license_type = license_info.get("key", "unknown").lower()
                license_name = license_info.get("name", "Unknown")
            else:
                license_type = str(license_info).lower()
                license_name = str(license_info)
            
            # Check against permissive licenses
            permissive_licenses = [
                "mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", 
                "unlicense", "cc0-1.0", "isc", "zlib"
            ]
            
            copyleft_licenses = [
                "gpl-2.0", "gpl-3.0", "lgpl-2.1", "lgpl-3.0", 
                "agpl-3.0", "cc-by-sa"
            ]
            
            issues = []
            
            if license_type in permissive_licenses:
                result_type = LicenseType.PERMISSIVE
            elif license_type in copyleft_licenses:
                result_type = LicenseType.COPYLEFT
                issues.append(f"Copyleft license detected: {license_name}")
            elif "proprietary" in license_type or "commercial" in license_type:
                result_type = LicenseType.PROPRIETARY
                issues.append(f"Proprietary license detected: {license_name}")
            else:
                result_type = LicenseType.UNKNOWN
                issues.append(f"Unknown or unrecognized license: {license_name}")
            
            # Additional file-based license detection
            if os.path.isfile(content_path) and content_path.endswith(('.py', '.js', '.java', '.cpp', '.c')):
                await self._scan_file_license_headers(content_path, issues)
            
            return {
                "type": result_type,
                "name": license_name,
                "issues": issues,
                "compliant": result_type == LicenseType.PERMISSIVE
            }
            
        except Exception as e:
            return {
                "type": LicenseType.UNKNOWN,
                "name": "Error",
                "issues": [f"License scan error: {str(e)}"],
                "compliant": False
            }
    
    async def _scan_file_license_headers(self, content_path: str, issues: List[str]) -> None:
        """Scan file headers for license information"""
        try:
            with open(content_path, 'r', encoding='utf-8', errors='ignore') as f:
                header = f.read(2000)  # Read first 2KB
                
                # Look for restrictive license indicators
                restrictive_indicators = [
                    "all rights reserved",
                    "proprietary",
                    "confidential",
                    "copyright.*not.*distribute",
                    "gpl.*license",
                    "copyleft"
                ]
                
                for indicator in restrictive_indicators:
                    if indicator.lower() in header.lower():
                        issues.append(f"Restrictive license indicator found in file: {indicator}")
                        break
                        
        except Exception:
            # Ignore file reading errors for license detection
            pass
    
    async def _scan_vulnerabilities(self, content_path: str) -> Dict[str, Any]:
        """Scan for known vulnerabilities"""
        vulnerabilities = []
        
        try:
            # This is a simplified vulnerability scanner
            # In production, this would integrate with tools like:
            # - Bandit (Python security)
            # - ESLint security plugins (JavaScript)
            # - SonarQube
            # - OWASP dependency check
            
            # Basic pattern-based vulnerability detection
            if os.path.isfile(content_path):
                await self._pattern_based_vuln_scan(content_path, vulnerabilities)
            
            # Simulated vulnerability database check
            # In production, this would query CVE databases
            await asyncio.sleep(0.1)  # Simulate scan time
            
            return {
                "vulnerabilities": vulnerabilities,
                "scan_method": "pattern_based",
                "database_version": "1.0.0"
            }
            
        except Exception as e:
            return {
                "vulnerabilities": [f"Vulnerability scan error: {str(e)}"],
                "scan_method": "error",
                "database_version": "unknown"
            }
    
    async def _pattern_based_vuln_scan(self, content_path: str, vulnerabilities: List[str]) -> None:
        """Pattern-based vulnerability detection"""
        try:
            with open(content_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(10000)  # Read first 10KB
                
                # Common vulnerability patterns
                vuln_patterns = {
                    "sql_injection": ["SELECT.*FROM.*WHERE.*=.*input", "exec.*SELECT", "query.*+.*user"],
                    "xss": ["innerHTML.*=.*input", "document.write.*input", "eval.*input"],
                    "command_injection": ["system.*input", "exec.*input", "shell_exec"],
                    "path_traversal": ["../", "..\\\\", "path.*input"],
                    "hardcoded_secrets": ["password.*=.*['\"]", "api_key.*=.*['\"]", "secret.*=.*['\"]"]
                }
                
                content_lower = content.lower()
                for vuln_type, patterns in vuln_patterns.items():
                    for pattern in patterns:
                        try:
                            if re.search(pattern, content_lower, re.IGNORECASE):
                                vulnerabilities.append(f"Potential {vuln_type.replace('_', ' ')}: {pattern}")
                                break
                        except re.error:
                            # Fall back to literal substring match for non-regex patterns
                            if pattern.lower() in content_lower:
                                vulnerabilities.append(f"Potential {vuln_type.replace('_', ' ')}: {pattern}")
                                break
                            
        except Exception:
            # Ignore file reading errors
            pass
    
    async def _scan_malware(self, content_path: str) -> Dict[str, Any]:
        """Scan for malware (simplified implementation)"""
        threats = []
        
        try:
            # This is a placeholder for malware scanning
            # In production, this would integrate with:
            # - ClamAV
            # - VirusTotal API
            # - Commercial antivirus engines
            
            # Basic suspicious content detection
            if os.path.isfile(content_path):
                with open(content_path, 'rb') as f:
                    content = f.read(1024)
                    
                    # Check for suspicious byte patterns
                    suspicious_patterns = [
                        b'\x4d\x5a',  # PE header
                        b'\x7f\x45\x4c\x46',  # ELF header
                        b'#!/bin/sh',  # Shell script
                        b'@echo off'  # Batch file
                    ]
                    
                    for pattern in suspicious_patterns:
                        if pattern in content:
                            threats.append(f"Suspicious binary pattern detected: {pattern.hex()}")
            
            return {
                "threats_found": len(threats) > 0,
                "threats": threats,
                "scanner_version": "1.0.0"
            }
            
        except Exception as e:
            return {
                "threats_found": True,
                "threats": [f"Malware scan error: {str(e)}"],
                "scanner_version": "error"
            }
    
    async def _assess_risk(self, scan_result: SecurityScanResult, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall security risk"""
        risk_factors = []
        recommendations = []
        
        # Vulnerability assessment
        vuln_count = len(scan_result.vulnerabilities_found)
        high_risk_vulns = sum(1 for v in scan_result.vulnerabilities_found 
                            if any(keyword in v.lower() for keyword in ["critical", "high", "injection", "exec"]))
        
        if vuln_count > self.risk_thresholds["max_vulnerabilities"]:
            risk_factors.append(f"High vulnerability count: {vuln_count}")
            recommendations.append("Manual security review required")
        
        if high_risk_vulns > self.risk_thresholds["max_high_risk_vulns"]:
            risk_factors.append(f"High-risk vulnerabilities detected: {high_risk_vulns}")
            recommendations.append("Immediate security remediation required")
        
        # License compliance assessment
        if scan_result.license_compliance != LicenseType.PERMISSIVE:
            risk_factors.append(f"Non-permissive license: {scan_result.license_compliance}")
            recommendations.append("License review required before use")
        
        # Compliance issues
        if len(scan_result.compliance_issues) > 0:
            risk_factors.append(f"Compliance issues: {len(scan_result.compliance_issues)}")
            recommendations.append("Address compliance issues before import")
        
        # Determine risk level
        if high_risk_vulns > 0 or "critical" in str(scan_result.vulnerabilities_found).lower():
            risk_level = SecurityRisk.CRITICAL
        elif vuln_count > 5 or scan_result.license_compliance == LicenseType.PROPRIETARY:
            risk_level = SecurityRisk.HIGH
        elif vuln_count > 2 or len(scan_result.compliance_issues) > 0:
            risk_level = SecurityRisk.MEDIUM
        elif vuln_count > 0:
            risk_level = SecurityRisk.LOW
        else:
            risk_level = SecurityRisk.NONE
        
        # Approval decision
        approved = risk_level in [SecurityRisk.NONE, SecurityRisk.LOW]
        
        if not approved:
            recommendations.append("Import blocked due to security concerns")
        
        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "recommendations": recommendations,
            "approved": approved
        }
    
    async def _quarantine_content(self, content_path: str, scan_result: SecurityScanResult) -> None:
        """Quarantine high-risk content"""
        try:
            os.makedirs(self.quarantine_dir, exist_ok=True)
            
            # Create quarantine entry
            quarantine_name = f"{scan_result.scan_id}_{os.path.basename(content_path)}"
            quarantine_path = os.path.join(self.quarantine_dir, quarantine_name)
            
            # Copy to quarantine
            if os.path.isfile(content_path):
                shutil.copy2(content_path, quarantine_path)
            elif os.path.isdir(content_path):
                shutil.copytree(content_path, quarantine_path)
            
            # Create quarantine metadata
            metadata_path = f"{quarantine_path}.metadata.json"
            metadata = {
                "scan_id": str(scan_result.scan_id),
                "quarantine_time": datetime.now(timezone.utc).isoformat(),
                "risk_level": scan_result.risk_level,
                "vulnerabilities": scan_result.vulnerabilities_found,
                "original_path": content_path
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"🚨 Content quarantined: {quarantine_name}")
            
        except Exception as e:
            print(f"❌ Failed to quarantine content: {e}")
    
    async def _trigger_critical_security_response(self, scan_result: SecurityScanResult) -> None:
        """Trigger enhanced security response for critical threats"""
        try:
            logger.critical("Critical security threat detected - activating enhanced measures",
                           scan_id=str(scan_result.scan_id),
                           risk_level=scan_result.risk_level,
                           vulnerabilities=scan_result.vulnerabilities_found[:3])  # Log first 3 for brevity
            
            # Activate emergency mode
            self.emergency_mode = True

            # Immediately terminate all active tool sandboxes
            for sandbox_id, context in list(self.active_tool_sandboxes.items()):
                logger.warning("Terminating sandbox due to critical security threat",
                             sandbox_id=sandbox_id,
                             tool_id=context.tool_id)
                await self._emergency_terminate_sandbox(sandbox_id)
            
            # Block similar content patterns
            await self._add_content_block_patterns(scan_result)
            
            # Notify security monitoring systems
            await self._notify_security_systems(scan_result)
            
            logger.info("Critical security response completed")
            
        except Exception as e:
            logger.error("Failed to execute critical security response", error=str(e))
    
    async def _enforce_enhanced_resource_limits(self) -> None:
        """Enforce enhanced resource limits across all active sandboxes"""
        try:
            logger.info("Enforcing enhanced resource limits due to security incident")
            
            # Apply emergency limits to all active sandboxes
            for sandbox_id, context in self.active_tool_sandboxes.items():
                # Update resource limits to emergency mode
                context.resource_limits.emergency_mode = True
                context.resource_limits.cpu_percent = min(
                    context.resource_limits.cpu_percent,
                    context.resource_limits.emergency_cpu_percent
                )
                context.resource_limits.memory_mb = min(
                    context.resource_limits.memory_mb,
                    context.resource_limits.emergency_memory_mb
                )
                context.resource_limits.execution_timeout = min(
                    context.resource_limits.execution_timeout,
                    context.resource_limits.emergency_timeout
                )
                
                logger.debug("Applied emergency limits to sandbox",
                           sandbox_id=sandbox_id,
                           cpu_limit=context.resource_limits.cpu_percent,
                           memory_limit=context.resource_limits.memory_mb)
            
            # Set global emergency state that affects new sandboxes
            self.emergency_mode = True
            
        except Exception as e:
            logger.error("Failed to enforce enhanced resource limits", error=str(e))
    
    async def _emergency_terminate_sandbox(self, sandbox_id: str) -> None:
        """Emergency termination of a sandbox"""
        try:
            if sandbox_id in self.active_tool_sandboxes:
                context = self.active_tool_sandboxes[sandbox_id]
                
                # Send termination signal
                if hasattr(context, 'process') and context.process:
                    try:
                        context.process.terminate()
                        # Give 2 seconds for graceful shutdown
                        await asyncio.wait_for(context.process.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        # Force kill if doesn't terminate gracefully
                        context.process.kill()
                        await context.process.wait()
                
                # Clean up resources
                await self._cleanup_sandbox_resources(context)
                
                # Remove from active sandboxes
                del self.active_tool_sandboxes[sandbox_id]
                
                logger.info("Emergency sandbox termination completed", sandbox_id=sandbox_id)
            
        except Exception as e:
            logger.error("Emergency sandbox termination failed", 
                        sandbox_id=sandbox_id, error=str(e))
    
    async def _add_content_block_patterns(self, scan_result: SecurityScanResult) -> None:
        """Add blocking patterns for similar malicious content"""
        try:
            # Extract patterns from vulnerabilities for future blocking
            block_patterns = []
            for vuln in scan_result.vulnerabilities_found:
                if "malware" in vuln.lower():
                    block_patterns.append("malware_pattern")
                elif "virus" in vuln.lower():
                    block_patterns.append("virus_pattern")
                elif "trojan" in vuln.lower():
                    block_patterns.append("trojan_pattern")
                elif "backdoor" in vuln.lower():
                    block_patterns.append("backdoor_pattern")
            
            # Store patterns for future reference (would integrate with threat intelligence)
            if not hasattr(self, 'blocked_patterns'):
                self.blocked_patterns = set()
            
            self.blocked_patterns.update(block_patterns)
            
            logger.info("Added content blocking patterns", patterns=block_patterns)
            
        except Exception as e:
            logger.error("Failed to add content block patterns", error=str(e))
    
    async def _notify_security_systems(self, scan_result: SecurityScanResult) -> None:
        """Notify external security monitoring systems"""
        try:
            # Prepare security alert payload
            alert_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "alert_type": "critical_security_threat",
                "scan_id": str(scan_result.scan_id),
                "risk_level": scan_result.risk_level,
                "vulnerabilities": scan_result.vulnerabilities_found,
                "system": "prsm_sandbox_manager",
                "action_taken": "quarantine_and_emergency_response"
            }

            # Log security alert (would also send to SIEM/monitoring systems)
            logger.critical("SECURITY ALERT", **alert_data)

            # Emit structured security event to audit logger
            await audit_logger.log_security_event(
                event_type="sandbox_security_threat",
                user_id="system",
                details={
                    "severity": scan_result.risk_level,
                    "scan_id": str(scan_result.scan_id),
                    "vulnerabilities_count": len(scan_result.vulnerabilities_found),
                    "vulnerabilities": scan_result.vulnerabilities_found[:10],  # Limit for log size
                    "recommendations": scan_result.recommendations[:5] if scan_result.recommendations else [],
                    "action_taken": "quarantine_and_emergency_response"
                },
                security_level="critical"
            )

        except Exception as e:
            logger.error("Failed to notify security systems", error=str(e))
    
    async def _cleanup_sandbox_resources(self, context) -> None:
        """Clean up resources for a terminated sandbox"""
        try:
            # Clean up temporary files
            if hasattr(context, 'temp_dir') and context.temp_dir and os.path.exists(context.temp_dir):
                shutil.rmtree(context.temp_dir, ignore_errors=True)
            
            # Close file descriptors
            if hasattr(context, 'file_descriptors'):
                for fd in context.file_descriptors:
                    try:
                        os.close(fd)
                    except OSError:
                        pass  # Already closed
            
            # Clean up network connections
            if hasattr(context, 'network_connections'):
                for conn in context.network_connections:
                    try:
                        conn.close()
                    except Exception:
                        pass  # Already closed or invalid
            
            logger.debug("Sandbox resource cleanup completed", 
                        sandbox_id=getattr(context, 'sandbox_id', 'unknown'))
            
        except Exception as e:
            logger.error("Sandbox resource cleanup failed", error=str(e))
    
    async def _finalize_scan(self, scan_id: UUID, scan_result: SecurityScanResult) -> SecurityScanResult:
        """Finalize scan and update tracking"""
        try:
            # Calculate scan duration
            if scan_id in self.active_scans:
                start_time = self.active_scans[scan_id]["start_time"]
                scan_result.scan_duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                # Remove from active scans
                del self.active_scans[scan_id]
            
            # Store in history
            self.scan_history.append(scan_result)
            
            # Update status
            if len(self.active_scans) == 0:
                self.status = SandboxStatus.IDLE
            
            return scan_result
            
        except Exception as e:
            print(f"❌ Failed to finalize scan: {e}")
            return scan_result
    
    def _setup_sandbox(self) -> None:
        """Initialize sandbox environment"""
        try:
            # Create sandbox directories
            os.makedirs(self.sandbox_dir, exist_ok=True)
            os.makedirs(self.quarantine_dir, exist_ok=True)
            
            # Set restrictive permissions
            os.chmod(self.sandbox_dir, 0o700)
            os.chmod(self.quarantine_dir, 0o700)
            
            print(f"🔒 Sandbox environment initialized at {self.sandbox_dir}")
            
        except Exception as e:
            print(f"❌ Failed to setup sandbox: {e}")
            raise
    
    # ===============================
    # MCP Tool Execution Methods
    # ===============================
    
    async def execute_tool_safely(self, tool_execution_request: ToolExecutionRequest) -> ToolSandboxResult:
        """
        Execute an MCP tool within a secure sandbox environment
        
        This is the main entry point for secure tool execution, providing:
        - Resource isolation and limits
        - Permission validation and enforcement
        - Real-time security monitoring
        - Comprehensive audit logging
        - Automatic cleanup and recovery
        
        Args:
            tool_execution_request: Complete tool execution request
            
        Returns:
            ToolSandboxResult: Complete execution result with security metrics
        """
        if not self.enable_tool_sandboxing:
            logger.warning("Tool sandboxing is disabled - executing without sandbox")
            return await self._execute_tool_without_sandbox(tool_execution_request)

        # Check concurrent execution limits
        if len(self.active_tool_sandboxes) >= self.max_concurrent_tool_sandboxes:
            logger.warning("Maximum concurrent tool sandboxes reached",
                         active_count=len(self.active_tool_sandboxes),
                         max_allowed=self.max_concurrent_tool_sandboxes)
            
            return ToolSandboxResult(
                sandbox_id="rate_limited",
                tool_execution_result=ToolExecutionResult(
                    execution_id=tool_execution_request.execution_id,
                    tool_id=tool_execution_request.tool_id,
                    success=False,
                    execution_time=0.0,
                    error_message="Maximum concurrent tool executions reached",
                    error_code="RATE_LIMITED"
                ),
                security_violations=["Rate limit exceeded"]
            )
        
        # Create sandbox context
        context = ToolSandboxContext(
            tool_id=tool_execution_request.tool_id,
            user_id=tool_execution_request.user_id,
            execution_request=tool_execution_request,
            sandbox_type=self._determine_sandbox_type(tool_execution_request.sandbox_level),
            security_level=tool_execution_request.sandbox_level,
            resource_limits=self._create_resource_limits(tool_execution_request),
            permissions=self._create_security_permissions(tool_execution_request)
        )
        
        # Register active sandbox
        self.active_tool_sandboxes[context.sandbox_id] = context
        
        try:
            logger.info("Starting secure tool execution",
                       sandbox_id=context.sandbox_id,
                       tool_id=tool_execution_request.tool_id,
                       user_id=tool_execution_request.user_id,
                       sandbox_type=context.sandbox_type.value)
            
            # Validate security permissions
            security_validation = await self._validate_tool_security(context)
            if not security_validation["valid"]:
                return await self._create_security_violation_result(context, security_validation)
            
            # Start resource monitoring
            resource_monitor = ResourceMonitor(context)
            monitoring_task = await resource_monitor.start_monitoring()
            
            try:
                # Execute tool based on sandbox type
                if context.sandbox_type == ToolSandboxType.CONTAINER:
                    execution_result = await self._execute_tool_in_container(context)
                elif context.sandbox_type == ToolSandboxType.BASIC:
                    execution_result = await self._execute_tool_in_basic_sandbox(context)
                else:
                    execution_result = await self._execute_tool_direct(context)
                
                # Stop monitoring and get resource usage
                resource_summary = await resource_monitor.stop_monitoring()
                
                # Create comprehensive result
                sandbox_result = await self._create_sandbox_result(context, execution_result, resource_summary)
                
                logger.info("Secure tool execution completed",
                           sandbox_id=context.sandbox_id,
                           success=execution_result.success,
                           execution_time=execution_result.execution_time,
                           security_violations=len(sandbox_result.security_violations))
                
                return sandbox_result
                
            finally:
                # Ensure monitoring is stopped
                try:
                    await resource_monitor.stop_monitoring()
                except Exception:
                    pass
        
        except Exception as e:
            logger.error("Tool execution failed in sandbox",
                        sandbox_id=context.sandbox_id,
                        error=str(e))
            
            # Create error result
            error_result = ToolExecutionResult(
                execution_id=tool_execution_request.execution_id,
                tool_id=tool_execution_request.tool_id,
                success=False,
                execution_time=0.0,
                error_message=str(e),
                error_code="SANDBOX_ERROR"
            )
            
            return ToolSandboxResult(
                sandbox_id=context.sandbox_id,
                tool_execution_result=error_result,
                security_violations=[f"Sandbox execution error: {str(e)}"]
            )
        
        finally:
            # Cleanup sandbox
            await self._cleanup_tool_sandbox(context.sandbox_id)
    
    async def _validate_tool_security(self, context: ToolSandboxContext) -> Dict[str, Any]:
        """Validate tool execution against security policies"""
        validation_result = {
            "valid": True,
            "violations": [],
            "warnings": []
        }
        
        try:
            # Check user consent requirements
            if context.permissions.user_consent_required:
                # In production, this would check actual user consent
                # For now, we'll assume consent is granted for valid users
                if not context.user_id or context.user_id == "anonymous":
                    validation_result["valid"] = False
                    validation_result["violations"].append("User consent required but user not authenticated")
            
            # Check security level compatibility
            if context.security_level == ToolSecurityLevel.DANGEROUS:
                if not context.permissions.user_consent_required:
                    validation_result["warnings"].append("Dangerous tool without explicit consent")
            
            # Validate file permissions
            if context.permissions.file_write:
                for path in context.permissions.file_write:
                    if not self._is_safe_file_path(path):
                        validation_result["valid"] = False
                        validation_result["violations"].append(f"Unsafe file write path: {path}")
            
            # Check network permissions
            if context.permissions.network_outbound:
                if not context.permissions.allowed_hosts:
                    validation_result["warnings"].append("Outbound network access without host restrictions")
            
            # Validate resource limits
            if context.resource_limits.execution_timeout > 300:  # 5 minutes max
                validation_result["valid"] = False
                validation_result["violations"].append("Execution timeout exceeds maximum allowed")
            
            if context.resource_limits.memory_mb > 2048:  # 2GB max
                validation_result["valid"] = False
                validation_result["violations"].append("Memory limit exceeds maximum allowed")
            
            return validation_result
            
        except Exception as e:
            logger.error("Security validation failed",
                        sandbox_id=context.sandbox_id,
                        error=str(e))
            return {
                "valid": False,
                "violations": [f"Security validation error: {str(e)}"],
                "warnings": []
            }
    
    async def _execute_tool_in_container(self, context: ToolSandboxContext) -> ToolExecutionResult:
        """Execute tool in Docker container (production-grade isolation)"""
        start_time = time.time()
        
        try:
            # Create container configuration
            container_config = {
                "image": "prsm/tool-sandbox:latest",
                "command": ["python", "-c", f"import json; print(json.dumps({{'tool_id': '{context.tool_id}', 'status': 'simulated_success'}})))"],
                "environment": {
                    "TOOL_ID": context.tool_id,
                    "EXECUTION_ID": str(context.execution_request.execution_id),
                    "SANDBOX_ID": context.sandbox_id
                },
                "resource_limits": {
                    "memory": f"{context.resource_limits.memory_mb}m",
                    "cpu_quota": int(context.resource_limits.cpu_percent * 1000),
                    "execution_timeout": context.resource_limits.execution_timeout
                },
                "network": "none" if not context.permissions.network_outbound else "bridge",
                "read_only_root": True,
                "no_new_privileges": True
            }
            
            # Simulate container execution (in production, use Docker API)
            logger.info("Simulating container execution",
                       sandbox_id=context.sandbox_id,
                       container_config=container_config)
            
            # Simulate execution time based on tool complexity
            await asyncio.sleep(min(2.0, context.resource_limits.execution_timeout / 30))
            
            execution_time = time.time() - start_time
            
            # Simulate successful execution
            return ToolExecutionResult(
                execution_id=context.execution_request.execution_id,
                tool_id=context.tool_id,
                success=True,
                result_data={
                    "status": "container_execution_success",
                    "sandbox_type": "container",
                    "simulated": True,
                    "container_config": container_config
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Container execution failed",
                        sandbox_id=context.sandbox_id,
                        error=str(e))
            
            return ToolExecutionResult(
                execution_id=context.execution_request.execution_id,
                tool_id=context.tool_id,
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                error_code="CONTAINER_ERROR"
            )
    
    async def _execute_tool_in_basic_sandbox(self, context: ToolSandboxContext) -> ToolExecutionResult:
        """Execute tool in basic process sandbox (lightweight isolation)"""
        start_time = time.time()
        
        try:
            # Create isolated process environment
            sandbox_env = os.environ.copy()
            
            # Restrict environment variables
            restricted_env = {
                "TOOL_ID": context.tool_id,
                "EXECUTION_ID": str(context.execution_request.execution_id),
                "SANDBOX_ID": context.sandbox_id,
                "PATH": "/usr/local/bin:/usr/bin:/bin",  # Minimal PATH
                "HOME": os.path.join(self.sandbox_dir, "home"),
                "TMPDIR": os.path.join(self.tool_sandbox_dir, context.sandbox_id)
            }
            
            # Create sandbox directory
            sandbox_dir = os.path.join(self.tool_sandbox_dir, context.sandbox_id)
            os.makedirs(sandbox_dir, exist_ok=True)
            os.chmod(sandbox_dir, 0o700)
            
            # Simulate tool execution with process limits
            logger.info("Executing tool in basic sandbox",
                       sandbox_id=context.sandbox_id,
                       sandbox_dir=sandbox_dir)
            
            # In production, this would execute the actual tool with resource limits
            # For now, simulate execution
            await asyncio.sleep(min(1.0, context.resource_limits.execution_timeout / 60))
            
            execution_time = time.time() - start_time
            
            # Check if execution exceeded limits
            if execution_time > context.resource_limits.execution_timeout:
                return ToolExecutionResult(
                    execution_id=context.execution_request.execution_id,
                    tool_id=context.tool_id,
                    success=False,
                    execution_time=execution_time,
                    error_message="Tool execution timeout",
                    error_code="TIMEOUT",
                    resource_violations=["Execution timeout exceeded"]
                )
            
            # Simulate successful execution
            return ToolExecutionResult(
                execution_id=context.execution_request.execution_id,
                tool_id=context.tool_id,
                success=True,
                result_data={
                    "status": "basic_sandbox_success",
                    "sandbox_type": "basic",
                    "sandbox_dir": sandbox_dir,
                    "simulated": True
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Basic sandbox execution failed",
                        sandbox_id=context.sandbox_id,
                        error=str(e))
            
            return ToolExecutionResult(
                execution_id=context.execution_request.execution_id,
                tool_id=context.tool_id,
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                error_code="SANDBOX_ERROR"
            )
        
        finally:
            # Cleanup sandbox directory
            try:
                if 'sandbox_dir' in locals() and os.path.exists(sandbox_dir):
                    shutil.rmtree(sandbox_dir, ignore_errors=True)
            except Exception:
                pass
    
    async def _execute_tool_direct(self, context: ToolSandboxContext) -> ToolExecutionResult:
        """Execute tool with minimal sandboxing (for testing/safe tools)"""
        start_time = time.time()
        
        try:
            logger.info("Executing tool with minimal sandboxing",
                       sandbox_id=context.sandbox_id,
                       tool_id=context.tool_id)
            
            # Simulate direct execution with basic monitoring
            await asyncio.sleep(0.1)  # Minimal execution time
            
            execution_time = time.time() - start_time
            
            return ToolExecutionResult(
                execution_id=context.execution_request.execution_id,
                tool_id=context.tool_id,
                success=True,
                result_data={
                    "status": "direct_execution_success",
                    "sandbox_type": "none",
                    "simulated": True
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Direct execution failed",
                        sandbox_id=context.sandbox_id,
                        error=str(e))
            
            return ToolExecutionResult(
                execution_id=context.execution_request.execution_id,
                tool_id=context.tool_id,
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                error_code="EXECUTION_ERROR"
            )
    
    async def _execute_tool_without_sandbox(self, tool_execution_request: ToolExecutionRequest) -> ToolSandboxResult:
        """Execute tool without sandboxing (fallback mode)"""
        logger.warning("Executing tool without sandbox - security disabled",
                      tool_id=tool_execution_request.tool_id)
        
        start_time = time.time()
        
        try:
            # Minimal execution simulation
            await asyncio.sleep(0.05)
            execution_time = time.time() - start_time
            
            execution_result = ToolExecutionResult(
                execution_id=tool_execution_request.execution_id,
                tool_id=tool_execution_request.tool_id,
                success=True,
                result_data={"status": "unsandboxed_execution", "warning": "No security sandbox"},
                execution_time=execution_time
            )
            
            return ToolSandboxResult(
                sandbox_id="no_sandbox",
                tool_execution_result=execution_result,
                security_violations=["Tool executed without sandboxing"]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            execution_result = ToolExecutionResult(
                execution_id=tool_execution_request.execution_id,
                tool_id=tool_execution_request.tool_id,
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                error_code="UNSANDBOXED_ERROR"
            )
            
            return ToolSandboxResult(
                sandbox_id="no_sandbox",
                tool_execution_result=execution_result,
                security_violations=["Tool execution failed without sandboxing"]
            )
    
    def _determine_sandbox_type(self, security_level: ToolSecurityLevel) -> ToolSandboxType:
        """Determine appropriate sandbox type based on security level"""
        if security_level == ToolSecurityLevel.SAFE:
            return ToolSandboxType.BASIC
        elif security_level == ToolSecurityLevel.RESTRICTED:
            return ToolSandboxType.CONTAINER
        elif security_level in [ToolSecurityLevel.PRIVILEGED, ToolSecurityLevel.DANGEROUS]:
            return ToolSandboxType.CONTAINER  # Always container for high-risk tools
        else:
            return ToolSandboxType.BASIC
    
    def _create_resource_limits(self, request: ToolExecutionRequest) -> ResourceLimits:
        """Create enhanced resource limits based on tool execution request and security level"""
        # Start with base enhanced limits
        limits = ResourceLimits()
        
        # Apply security-level specific adjustments
        if request.sandbox_level == ToolSecurityLevel.SAFE:
            # More generous limits for safe tools, but still secure
            limits.cpu_percent = 40.0
            limits.memory_mb = 512
            limits.execution_timeout = min(request.timeout_seconds, 60.0)
            limits.max_processes = 8
            limits.network_connections = 5
            limits.disk_read_mb = 100
            limits.disk_write_mb = 50
            
        elif request.sandbox_level == ToolSecurityLevel.RESTRICTED:
            # Standard enhanced limits - use defaults from ResourceLimits class
            limits.execution_timeout = min(request.timeout_seconds, 30.0)
            
        elif request.sandbox_level == ToolSecurityLevel.PRIVILEGED:
            # Stricter limits for privileged operations
            limits.cpu_percent = 20.0
            limits.memory_mb = 128
            limits.execution_timeout = min(request.timeout_seconds, 20.0)
            limits.max_processes = 3
            limits.max_threads = 5
            limits.network_connections = 2
            limits.network_upload_mb = 2
            limits.network_download_mb = 10
            limits.disable_ptrace = True
            limits.read_only_filesystem = True
            
        else:  # DANGEROUS
            # Maximum security restrictions for dangerous tools
            limits.cpu_percent = 15.0
            limits.memory_mb = 64
            limits.execution_timeout = min(request.timeout_seconds, 15.0)
            limits.max_processes = 2
            limits.max_threads = 3
            limits.network_connections = 1
            limits.network_upload_mb = 1
            limits.network_download_mb = 5
            limits.disable_network = True
            limits.read_only_filesystem = True
            limits.disable_ptrace = True
            limits.disable_core_dumps = True
            limits.syscall_whitelist = [
                "read", "write", "open", "close", "stat", "fstat", "lstat",
                "access", "getcwd", "getpid", "exit", "exit_group"
            ]
        
        # Apply emergency mode restrictions if active
        if hasattr(self, 'emergency_mode') and self.emergency_mode:
            limits.emergency_mode = True
            limits.cpu_percent = min(limits.cpu_percent, limits.emergency_cpu_percent)
            limits.memory_mb = min(limits.memory_mb, limits.emergency_memory_mb)
            limits.execution_timeout = min(limits.execution_timeout, limits.emergency_timeout)
        
        return limits
    
    def _create_security_permissions(self, request: ToolExecutionRequest) -> SecurityPermissions:
        """Create security permissions based on tool execution request"""
        permissions = SecurityPermissions()
        
        # Set permissions based on security level
        if request.sandbox_level == ToolSecurityLevel.SAFE:
            permissions.user_consent_required = False
            permissions.file_read.add(os.path.join(self.sandbox_dir, "*"))
        elif request.sandbox_level == ToolSecurityLevel.RESTRICTED:
            permissions.user_consent_required = True
            permissions.file_read.add(os.path.join(self.sandbox_dir, "*"))
            permissions.file_write.add(os.path.join(self.sandbox_dir, "*"))
        else:
            # Privileged/dangerous tools require explicit permissions
            permissions.user_consent_required = True
            # Permissions would be set based on specific tool requirements
        
        # Add tool-specific permissions from request
        permissions.tool_permissions.update(request.permissions)
        
        return permissions
    
    def _is_safe_file_path(self, path: str) -> bool:
        """Validate that file path is safe for tool access"""
        # Prevent path traversal attacks
        if ".." in path or path.startswith("/"):
            return False
        
        # Only allow access to sandbox directories
        safe_prefixes = [
            self.sandbox_dir,
            self.tool_sandbox_dir,
            self.quarantine_dir
        ]
        
        return any(path.startswith(prefix) for prefix in safe_prefixes)
    
    async def _create_security_violation_result(self, context: ToolSandboxContext, 
                                              validation: Dict[str, Any]) -> ToolSandboxResult:
        """Create result for security violation"""
        execution_result = ToolExecutionResult(
            execution_id=context.execution_request.execution_id,
            tool_id=context.tool_id,
            success=False,
            execution_time=0.0,
            error_message="Security validation failed",
            error_code="SECURITY_VIOLATION",
            security_violations=validation["violations"]
        )
        
        return ToolSandboxResult(
            sandbox_id=context.sandbox_id,
            tool_execution_result=execution_result,
            security_violations=validation["violations"],
            permission_violations=validation["violations"]
        )
    
    async def _create_sandbox_result(self, context: ToolSandboxContext, 
                                   execution_result: ToolExecutionResult,
                                   resource_summary: Dict[str, Any]) -> ToolSandboxResult:
        """Create comprehensive sandbox result"""
        return ToolSandboxResult(
            sandbox_id=context.sandbox_id,
            tool_execution_result=execution_result,
            security_violations=resource_summary.get("violations", []),
            resource_violations=resource_summary.get("violations", []),
            peak_cpu_percent=resource_summary.get("peak_cpu", 0.0),
            peak_memory_mb=resource_summary.get("peak_memory", 0.0),
            total_disk_read_mb=resource_summary.get("total_disk_read", 0.0),
            total_disk_write_mb=resource_summary.get("total_disk_write", 0.0),
            total_network_mb=resource_summary.get("total_network", 0.0),
            execution_time=execution_result.execution_time,
            audit_trail=[
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event": "tool_execution_completed",
                    "sandbox_id": context.sandbox_id,
                    "tool_id": context.tool_id,
                    "success": execution_result.success,
                    "security_level": context.security_level.value,
                    "sandbox_type": context.sandbox_type.value
                }
            ]
        )
    
    async def _cleanup_tool_sandbox(self, sandbox_id: str):
        """Clean up tool sandbox resources"""
        try:
            # Remove from active sandboxes
            if sandbox_id in self.active_tool_sandboxes:
                context = self.active_tool_sandboxes[sandbox_id]
                del self.active_tool_sandboxes[sandbox_id]
                
                # Update performance stats
                self.performance_stats["tool_usage"]["total_tool_requests"] += 1
                
                logger.debug("Tool sandbox cleaned up",
                           sandbox_id=sandbox_id,
                           tool_id=context.tool_id)
        
        except Exception as e:
            logger.error("Sandbox cleanup failed",
                        sandbox_id=sandbox_id,
                        error=str(e))
    
    async def _periodic_tool_cleanup(self):
        """Periodic cleanup of old tool sandboxes"""
        while True:
            try:
                await asyncio.sleep(self.tool_cleanup_interval)
                
                current_time = datetime.now(timezone.utc)
                expired_sandboxes = []
                
                for sandbox_id, context in self.active_tool_sandboxes.items():
                    # Check if sandbox has been running too long
                    if context.started_at:
                        running_time = (current_time - context.started_at).total_seconds()
                        max_runtime = context.resource_limits.execution_timeout * 2  # 2x timeout as max
                        
                        if running_time > max_runtime:
                            expired_sandboxes.append(sandbox_id)
                    elif (current_time - context.created_at).total_seconds() > 3600:  # 1 hour max
                        expired_sandboxes.append(sandbox_id)
                
                # Cleanup expired sandboxes
                for sandbox_id in expired_sandboxes:
                    logger.warning("Cleaning up expired tool sandbox",
                                  sandbox_id=sandbox_id)
                    await self._cleanup_tool_sandbox(sandbox_id)
                
                if expired_sandboxes:
                    logger.info("Periodic sandbox cleanup completed",
                               cleaned_count=len(expired_sandboxes))
            
            except Exception as e:
                logger.error("Periodic cleanup failed", error=str(e))
    
    def __del__(self):
        """Cleanup sandbox on destruction"""
        try:
            if hasattr(self, 'sandbox_dir') and os.path.exists(self.sandbox_dir):
                # Preserve quarantine directory
                quarantine_backup = None
                if os.path.exists(self.quarantine_dir):
                    quarantine_backup = self.quarantine_dir + "_backup"
                    shutil.move(self.quarantine_dir, quarantine_backup)
                
                # Remove sandbox
                shutil.rmtree(self.sandbox_dir, ignore_errors=True)
                
                # Restore quarantine
                if quarantine_backup:
                    os.makedirs(self.sandbox_dir, exist_ok=True)
                    shutil.move(quarantine_backup, self.quarantine_dir)
                    
        except Exception:
            pass


# Add json import for metadata handling
import json