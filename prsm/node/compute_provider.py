"""
Compute Provider
================

Accepts and executes compute jobs from the network.
Auto-detects available resources using psutil and the existing
ResourceCapabilityDetector. Jobs run in sandboxed subprocesses
with resource monitoring.
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from prsm.node.config import NodeConfig

from prsm.compute.model_sharding.executor import execute_shard_locally
from prsm.compute.model_sharding.models import ModelShard
from prsm.compute.shard_receipt import build_receipt_signing_payload
from prsm.node.config import is_active_now
from prsm.node.gossip import (
    GOSSIP_JOB_OFFER,
    GOSSIP_JOB_ACCEPT,
    GOSSIP_JOB_CONFIRM,
    GOSSIP_JOB_CANCEL,
    GOSSIP_JOB_RESULT,
    GossipProtocol,
)
from prsm.node.identity import NodeIdentity
from prsm.node.local_ledger import LocalLedger
from prsm.node.transport import MSG_DIRECT, P2PMessage, WebSocketTransport
from prsm.node.payment_escrow import PaymentEscrow
from prsm.node.result_consensus import ResultConsensus

logger = logging.getLogger(__name__)


class JobType(str, Enum):
    INFERENCE = "inference"
    EMBEDDING = "embedding"
    BENCHMARK = "benchmark"
    WASM_EXECUTE = "wasm_execute"


class JobStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ComputeJob:
    """A compute job offered to or accepted by this node."""
    job_id: str
    job_type: JobType
    requester_id: str
    payload: Dict[str, Any]
    ftns_budget: float
    status: JobStatus = JobStatus.PENDING
    accepted_by: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    result_signature: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error: Optional[str] = None


@dataclass
class SystemResources:
    """Detected system resources."""
    cpu_count: int = 1
    cpu_freq_mhz: float = 0.0
    memory_total_gb: float = 0.0
    memory_available_gb: float = 0.0
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0


def detect_resources() -> SystemResources:
    """Detect available system resources using psutil."""
    resources = SystemResources()
    try:
        import psutil
        import platform
        resources.cpu_count = psutil.cpu_count(logical=True) or 1
        freq = psutil.cpu_freq()
        if freq and freq.current and freq.current >= 100:
            # Sprint 533 F51 fix: psutil.cpu_freq() returns
            # nonsense (4 MHz) on Apple Silicon — the M-series
            # power-management exposes a "tbfrequency" that's
            # not the CPU clock. Sanity-check the reading;
            # anything below 100 MHz is broken (no modern CPU
            # runs that slow). Fall back to platform-specific
            # nominal clocks for known Apple chips, else leave
            # cpu_freq_mhz=0 + don't display.
            resources.cpu_freq_mhz = freq.current
        elif platform.system() == "Darwin":
            try:
                # Sprint 566 F25 fix: do NOT re-import `subprocess`
                # here. The module is already imported at the top
                # (line 17). Re-importing inside a conditional
                # branch makes Python treat `subprocess` as a
                # function-local variable for the WHOLE function
                # (scoping rule: any assignment makes the name
                # local). On non-Darwin systems the Darwin branch
                # is skipped, the local `subprocess` stays unbound,
                # and line 142 (`subprocess.TimeoutExpired`) raises
                # UnboundLocalError at module load — crashing every
                # Linux daemon startup. Mac dev hid the bug because
                # Darwin branch always runs there.
                brand = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    text=True, timeout=2,
                ).strip()
                # Nominal base clock for Apple M-series (Wikipedia)
                _APPLE_M_NOMINAL_MHZ = {
                    "Apple M1": 3200, "Apple M1 Pro": 3200,
                    "Apple M1 Max": 3200, "Apple M1 Ultra": 3200,
                    "Apple M2": 3500, "Apple M2 Pro": 3500,
                    "Apple M2 Max": 3700, "Apple M2 Ultra": 3700,
                    "Apple M3": 4000, "Apple M3 Pro": 4000,
                    "Apple M3 Max": 4000,
                    "Apple M4": 4400, "Apple M4 Pro": 4400,
                    "Apple M4 Max": 4400,
                }
                if brand in _APPLE_M_NOMINAL_MHZ:
                    resources.cpu_freq_mhz = float(
                        _APPLE_M_NOMINAL_MHZ[brand],
                    )
            except Exception:
                pass  # leave cpu_freq_mhz=0
        mem = psutil.virtual_memory()
        resources.memory_total_gb = round(mem.total / (1024**3), 2)
        resources.memory_available_gb = round(mem.available / (1024**3), 2)
    except ImportError:
        resources.cpu_count = os.cpu_count() or 1

    # GPU detection (best effort)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            resources.gpu_available = True
            resources.gpu_name = parts[0].strip()
            resources.gpu_memory_gb = round(float(parts[1].strip()) / 1024, 2)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    return resources


class ComputeProvider:
    """Accepts and executes compute jobs from the PRSM network.

    Supports both single-node mode (self-compute) and multi-node mode
    with consensus-verified results and escrowed payments.
    """

    def __init__(
        self,
        identity: NodeIdentity,
        transport: WebSocketTransport,
        gossip: GossipProtocol,
        ledger: LocalLedger,
        cpu_allocation_pct: int = 50,
        memory_allocation_pct: int = 50,
        max_concurrent_jobs: int = 3,
        gpu_allocation_pct: int = 80,
        config: Optional["NodeConfig"] = None,
    ):
        self.identity = identity
        self.transport = transport
        self.gossip = gossip
        self.ledger = ledger
        self.cpu_allocation_pct = cpu_allocation_pct
        self.memory_allocation_pct = memory_allocation_pct
        self.max_concurrent_jobs = max_concurrent_jobs
        self.gpu_allocation_pct = gpu_allocation_pct
        self.config = config  # NodeConfig for scheduling checks

        self.resources = detect_resources()
        self.active_jobs: Dict[str, ComputeJob] = {}
        self.completed_jobs: Dict[str, ComputeJob] = {}
        self._max_completed_jobs = 500  # Prevent unbounded memory growth
        self._running = False
        self.allow_self_compute = True  # Execute own jobs when no peers (single-node mode)
        self.ledger_sync = None  # Set by node.py after construction
        self.orchestrator = None  # NWTN orchestrator, set by node.py after construction

        # Cross-node infrastructure
        self.escrow = PaymentEscrow(
            ledger=self.ledger,
            node_id=self.identity.node_id,
        )
        self.consensus = ResultConsensus(
            epsilon=0.01,
            timeout_seconds=300.0,
        )
        self._escrow_task: Optional[asyncio.Task] = None

        self._current_jobs: int = 0

        # Phase 2.1 Line A / Phase 3 Task 5: track in-flight shard
        # requests so preempt_inflight_shards() can signal all of
        # them on SIGTERM / spot-instance eviction.
        # Keyed by request_id → (peer_id, shard_index).
        self._inflight_requests: Dict[str, tuple] = {}

        # Phase 3 Task 5: bound MarketplaceAdvertiser (set by node.py
        # after ComputeProvider is constructed). Lets the price-quote
        # handler answer with the currently-advertised price.
        self._marketplace_advertiser = None

    @property
    def available_capacity(self) -> Dict[str, Any]:
        active_count = len(self.active_jobs)
        capacity = {
            "cpu_cores_allocated": max(1, int(self.resources.cpu_count * self.cpu_allocation_pct / 100)),
            "memory_gb_allocated": round(self.resources.memory_total_gb * self.memory_allocation_pct / 100, 2),
            "gpu_available": self.resources.gpu_available,
            "concurrent_slots": max(0, self.max_concurrent_jobs - active_count),
            "active_jobs": active_count,
        }
        if self.resources.gpu_available:
            capacity["gpu_memory_gb_allocated"] = round(
                self.resources.gpu_memory_gb * self.gpu_allocation_pct / 100, 2
            )
        return capacity

    async def start(self) -> None:
        """Register gossip handlers and start provider."""
        self._running = True
        self.gossip.subscribe(GOSSIP_JOB_OFFER, self._on_job_offer)
        self.gossip.subscribe(GOSSIP_JOB_CONFIRM, self._on_job_confirm)
        self.gossip.subscribe(GOSSIP_JOB_CANCEL, self._on_job_cancel)
        self.transport.on_message(MSG_DIRECT, self._on_direct_message)
        # Jobs waiting for confirm before execution
        self._pending_confirm: Dict[str, ComputeJob] = {}
        logger.info(
            f"Compute provider started: {self.resources.cpu_count} CPUs, "
            f"{self.resources.memory_total_gb}GB RAM, "
            f"GPU: {'yes (' + self.resources.gpu_name + ')' if self.resources.gpu_available else 'no'}"
        )

    async def stop(self) -> None:
        self._running = False

    async def _on_job_offer(self, subtype: str, data: Dict[str, Any], origin: str) -> None:
        """Evaluate a job offer and accept if we have capacity.

        In multi-node mode, acceptance is a two-phase handshake:
          1. Provider publishes GOSSIP_JOB_ACCEPT (this method)
          2. Requester picks the winning provider and publishes GOSSIP_JOB_CONFIRM
          3. Provider starts execution only after receiving the confirm

        In single-node mode (no peers), the provider executes immediately
        since there's no competing provider.
        """
        if not self._running:
            return

        # Check if we're within active hours
        if self.config and not is_active_now(self.config):
            logger.debug("Node is outside active hours, declining job offer")
            return

        job_id = data.get("job_id", "")
        job_type_str = data.get("job_type", "")
        ftns_budget = data.get("ftns_budget", 0.0)
        requester_id = data.get("requester_id", origin)
        target_peers = data.get("target_peers", [])

        # Enforce target_peers: if the offer specifies target peers,
        # only accept if we're in the list.
        if target_peers and self.identity.node_id not in target_peers:
            logger.debug(
                f"Declining job {job_id[:8]}: not in target_peers list"
            )
            return

        # In network mode (peers connected), don't accept own jobs to avoid
        # double-counting.  In local mode (no peers), execute our own jobs
        # so a single-node setup is functional out of the box.
        is_single_node = not self.transport or self.transport.peer_count == 0
        if requester_id == self.identity.node_id:
            if not self.allow_self_compute:
                return
            if not is_single_node:
                return

        # Check if we already have this job
        if job_id in self.active_jobs or job_id in self.completed_jobs:
            return
        if hasattr(self, '_pending_confirm') and job_id in self._pending_confirm:
            return

        # Check capacity
        if len(self.active_jobs) >= self.max_concurrent_jobs:
            return

        # Validate job type
        try:
            job_type = JobType(job_type_str)
        except ValueError:
            return

        # Build the job object
        job = ComputeJob(
            job_id=job_id,
            job_type=job_type,
            requester_id=requester_id,
            payload=data.get("payload", {}),
            ftns_budget=ftns_budget,
            status=JobStatus.ACCEPTED,
            accepted_by=self.identity.node_id,
        )

        # Announce acceptance
        await self.gossip.publish(GOSSIP_JOB_ACCEPT, {
            "job_id": job_id,
            "provider_id": self.identity.node_id,
            "public_key": self.identity.public_key_b64,
            "capacity": self.available_capacity,
        })

        if is_single_node:
            # Single-node: no competing providers, execute immediately
            self.active_jobs[job_id] = job
            logger.info(
                f"Accepted job {job_id[:8]} ({job_type.value}) "
                f"from {requester_id[:8]} [single-node, executing]"
            )
            asyncio.create_task(self._execute_job(job))
        else:
            # Multi-node: wait for GOSSIP_JOB_CONFIRM before executing
            if hasattr(self, '_pending_confirm'):
                self._pending_confirm[job_id] = job
            else:
                self._pending_confirm = {job_id: job}
            logger.info(
                f"Accepted job {job_id[:8]} ({job_type.value}) "
                f"from {requester_id[:8]} [awaiting confirm]"
            )

    async def _on_job_confirm(self, subtype: str, data: Dict[str, Any], origin: str) -> None:
        """Handle job confirmation from the requester.

        The requester picks the winning provider and broadcasts
        GOSSIP_JOB_CONFIRM.  If we're the chosen provider, execute.
        If not, discard the pending job.
        """
        job_id = data.get("job_id", "")
        confirmed_provider = data.get("provider_id", "")

        pending = getattr(self, '_pending_confirm', {})
        job = pending.get(job_id)
        if not job:
            return

        # sp935 — only the job's REQUESTER may confirm it. `origin` is the
        # sp934-authenticated author of the confirm; without this check any peer
        # could confirm (or, pre-sp934, redirect) another requester's job.
        # Peek before popping so a forged confirm can't consume a pending job.
        if origin != job.requester_id:
            logger.warning(
                "Job %s: confirm from %s is not the requester %s — ignoring",
                job_id[:8], str(origin)[:8], str(job.requester_id)[:8],
            )
            return

        pending.pop(job_id, None)
        if confirmed_provider == self.identity.node_id:
            # We won — start executing
            self.active_jobs[job_id] = job
            logger.info(f"Job {job_id[:8]} confirmed — executing")
            asyncio.create_task(self._execute_job(job))
        else:
            # Another provider was chosen — drop the job
            logger.info(
                f"Job {job_id[:8]} assigned to {confirmed_provider[:8]}, "
                f"dropping"
            )

    async def _on_job_cancel(self, subtype: str, data: Dict[str, Any], origin: str) -> None:
        """Handle job cancellation from the requester."""
        job_id = data.get("job_id", "")

        pending = getattr(self, '_pending_confirm', {})
        job = pending.get(job_id) or self.active_jobs.get(job_id)
        if not job:
            return

        # sp935 — only the job's REQUESTER may cancel it. `origin` is the
        # sp934-authenticated author of the cancel; without this check any peer
        # could cancel another requester's in-flight job (free DoS on the
        # provider's work + payment).
        if origin != job.requester_id:
            logger.warning(
                "Job %s: cancel from %s is not the requester %s — ignoring",
                job_id[:8], str(origin)[:8], str(job.requester_id)[:8],
            )
            return

        # Remove from pending confirm queue
        pending.pop(job_id, None)

        # If the job is active, mark it cancelled (can't stop mid-execution
        # in async, but we can prevent result publishing)
        active = self.active_jobs.pop(job_id, None)
        if active:
            active.status = JobStatus.FAILED
            active.error = "Cancelled by requester"
            self.completed_jobs[job_id] = active
            logger.info(f"Job {job_id[:8]} cancelled by requester")

    async def _execute_job(self, job: ComputeJob) -> None:
        """Execute a compute job and publish the result."""
        job.status = JobStatus.RUNNING
        try:
            if job.job_type == JobType.BENCHMARK:
                result = await self._run_benchmark(job)
            elif job.job_type == JobType.INFERENCE:
                result = await self._run_inference(job)
            elif job.job_type == JobType.EMBEDDING:
                result = await self._run_embedding(job)
            elif job.job_type == JobType.WASM_EXECUTE:
                result = await self._run_wasm(job)
            else:
                raise ValueError(f"Unsupported job type: {job.job_type}")

            job.status = JobStatus.COMPLETED
            job.result = result
            job.completed_at = time.time()

            # Sign the result
            result_bytes = json.dumps(result, sort_keys=True).encode()
            job.result_signature = self.identity.sign(result_bytes)

            # NOTE: Provider does NOT self-credit FTNS here.  Payment flows
            # through the escrow system on the requester side:
            #   1. Requester locks escrow before broadcasting the offer.
            #   2. On receiving the result, the requester releases escrow
            #      to the provider (or refunds on failure).
            # Self-crediting would cause double-payment and ledger divergence
            # across nodes.

            # Publish result
            await self.gossip.publish(GOSSIP_JOB_RESULT, {
                "job_id": job.job_id,
                "provider_id": self.identity.node_id,
                "status": "completed",
                "result": result,
                "signature": job.result_signature,
                "public_key": self.identity.public_key_b64,
            })

            logger.info(f"Job {job.job_id[:8]} completed, result published")

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = time.time()

            await self.gossip.publish(GOSSIP_JOB_RESULT, {
                "job_id": job.job_id,
                "provider_id": self.identity.node_id,
                "status": "failed",
                "error": str(e),
            })

            logger.error(f"Job {job.job_id[:8]} failed: {e}")

        else:
            # Job completed successfully — escrow is released by the API
            # endpoint (compute_query) after returning the result.
            # This avoids a race with the concurrent API escrow release.
            pass

        finally:
            # Move from active to completed
            self.active_jobs.pop(job.job_id, None)
            self.completed_jobs[job.job_id] = job

            # Evict oldest completed jobs to prevent memory leak
            if len(self.completed_jobs) > self._max_completed_jobs:
                excess = len(self.completed_jobs) - self._max_completed_jobs
                oldest_keys = list(self.completed_jobs.keys())[:excess]
                for k in oldest_keys:
                    del self.completed_jobs[k]

    # ── Job type implementations ─────────────────────────────────

    async def _run_benchmark(self, job: ComputeJob) -> Dict[str, Any]:
        """Run a compute benchmark and return results."""
        iterations = job.payload.get("iterations", 1000000)

        start = time.monotonic()
        # CPU benchmark: compute primes (runs in thread to not block event loop)
        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(None, _cpu_benchmark, iterations)
        elapsed = time.monotonic() - start

        return {
            "benchmark_type": "cpu_prime_sieve",
            "iterations": iterations,
            "primes_found": count,
            "elapsed_seconds": round(elapsed, 4),
            "ops_per_second": round(iterations / elapsed, 2),
            "cpu_count": self.resources.cpu_count,
            "memory_gb": self.resources.memory_total_gb,
        }

    async def _run_inference(self, job: ComputeJob) -> Dict[str, Any]:
        """Run an inference job via NWTN orchestrator if available, else mock."""
        prompt = job.payload.get("prompt", "")
        model = job.payload.get("model", "local")

        # Use NWTN orchestrator if wired — with timeout to avoid hanging
        if self.orchestrator is not None:
            try:
                from prsm.core.models import UserInput
                user_input = UserInput(
                    user_id=job.requester_id,
                    prompt=prompt,
                    context_allocation=100,
                )
                response = await asyncio.wait_for(
                    self.orchestrator.process_query(user_input),
                    timeout=10.0,
                )
                return {
                    "model": model,
                    "prompt": prompt[:200],
                    "response": response.response,
                    "tokens_used": response.context_used,
                    "ftns_charged": response.ftns_charged,
                    "models_used": response.models_used,
                    "confidence": response.confidence_score,
                    "processing_time": response.processing_time,
                    "provider_node": self.identity.node_id,
                    "source": "nwtn_orchestrator",
                }
            except asyncio.TimeoutError:
                logger.warning(
                    f"NWTN orchestrator timed out after 10s, "
                    f"falling back to mock for {job.job_id[:8]}"
                )
            except Exception as e:
                logger.warning(f"NWTN inference failed, falling back to mock: {e}")

        # Fallback: mock response (alpha)
        return {
            "model": model,
            "prompt": prompt[:200],
            "response": f"[PRSM node {self.identity.node_id[:8]} processed inference]",
            "tokens_used": len(prompt.split()),
            "provider_node": self.identity.node_id,
            "source": "mock",
            "warning": "No LLM backend configured. This is a mock response.",
        }

    async def _run_embedding(self, job: ComputeJob) -> Dict[str, Any]:
        """Compute embeddings using the backend registry.
        
        Uses real embedding models when available (OpenAI, Local, etc.),
        falls back to deterministic mock embeddings when no backend is configured.
        """
        text = job.payload.get("text", "")
        dimensions = job.payload.get("dimensions", 1536)  # Default to OpenAI small dimensions
        model_id = job.payload.get("model_id")  # Optional specific model
        
        # Try to use backend registry if available (wired via orchestrator)
        if self.orchestrator is not None and hasattr(self.orchestrator, 'backend_registry'):
            try:
                registry = self.orchestrator.backend_registry
                if registry is not None:
                    result = await registry.embed_with_fallback(
                        text=text,
                        model_id=model_id,
                        dimensions=dimensions
                    )
                    return {
                        "text_length": len(text),
                        "dimensions": len(result.embedding),
                        "embedding": result.embedding,
                        "model_id": result.model_id,
                        "provider": result.provider.value,
                        "token_count": result.token_count,
                        "provider_node": self.identity.node_id,
                        "source": "backend_registry",
                    }
            except Exception as e:
                logger.warning(f"Backend registry embedding failed, falling back to mock: {e}")
        
        # Fallback: generate a deterministic pseudo-embedding (for testing/alpha)
        import hashlib
        text_hash = hashlib.sha256(text.encode()).digest()
        
        # Create embedding with requested dimensions
        embedding = []
        for i in range(dimensions):
            byte_val = text_hash[i % len(text_hash)]
            # Normalize to [-1, 1] range
            embedding.append((byte_val - 128) / 128.0)
        
        # Normalize to unit vector for cosine similarity
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return {
            "text_length": len(text),
            "dimensions": dimensions,
            "embedding": embedding,
            "model_id": "fallback-hash",
            "provider": "mock",
            "provider_node": self.identity.node_id,
            "source": "mock",
            "warning": "No embedding backend configured. Using pseudo-vectors.",
        }

    async def _run_wasm(self, job: ComputeJob) -> Dict[str, Any]:
        """Execute a WASM module in a sandboxed runtime."""
        import base64
        from prsm.compute.wasm.runtime import WasmtimeRuntime
        from prsm.compute.wasm.models import ResourceLimits

        payload = job.payload
        wasm_bytes = base64.b64decode(payload.get("wasm_bytes_b64", ""))
        input_data = base64.b64decode(payload.get("input_data_b64", ""))

        limits = ResourceLimits(
            max_memory_bytes=payload.get("max_memory_bytes", 256 * 1024 * 1024),
            max_execution_seconds=payload.get("max_execution_seconds", 30),
            max_output_bytes=payload.get("max_output_bytes", 10 * 1024 * 1024),
        )

        runtime = WasmtimeRuntime()
        if not runtime.available:
            raise RuntimeError(
                "WASM runtime not available. Install with: pip install prsm-network[wasm]"
            )

        module = runtime.load(wasm_bytes)
        result = runtime.execute(module, input_data, limits)

        return {
            "execution_status": result.status.value,
            "output_b64": base64.b64encode(result.output).decode(),
            "execution_time_seconds": result.execution_time_seconds,
            "memory_used_bytes": result.memory_used_bytes,
            "pcu": result.pcu(),
            "error": result.error,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return provider statistics."""
        return {
            "resources": {
                "cpu_count": self.resources.cpu_count,
                "cpu_freq_mhz": self.resources.cpu_freq_mhz,
                "memory_total_gb": self.resources.memory_total_gb,
                "gpu_available": self.resources.gpu_available,
                "gpu_name": self.resources.gpu_name,
            },
            "allocation": {
                "cpu_pct": self.cpu_allocation_pct,
                "memory_pct": self.memory_allocation_pct,
                "max_concurrent": self.max_concurrent_jobs,
            },
            "capacity": self.available_capacity,
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
        }

    def update_allocation(
        self,
        cpu_allocation_pct: Optional[int] = None,
        memory_allocation_pct: Optional[int] = None,
        max_concurrent_jobs: Optional[int] = None,
        gpu_allocation_pct: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update resource allocation settings at runtime.
        
        Changes take effect on next job acceptance. This method allows
        live tuning of compute resource allocation without restarting
        the node.
        
        Args:
            cpu_allocation_pct: Percentage of CPU to allocate (10-90)
            memory_allocation_pct: Percentage of memory to allocate (10-90)
            max_concurrent_jobs: Maximum number of concurrent jobs
            gpu_allocation_pct: Percentage of GPU memory to allocate (10-100)
        
        Returns:
            Dict with updated allocation values
        
        Raises:
            ValueError: If any value is out of valid range
        """
        # Validate ranges
        if cpu_allocation_pct is not None:
            if not 10 <= cpu_allocation_pct <= 90:
                raise ValueError(f"cpu_allocation_pct must be 10-90, got {cpu_allocation_pct}")
            self.cpu_allocation_pct = cpu_allocation_pct
            
        if memory_allocation_pct is not None:
            if not 10 <= memory_allocation_pct <= 90:
                raise ValueError(f"memory_allocation_pct must be 10-90, got {memory_allocation_pct}")
            self.memory_allocation_pct = memory_allocation_pct
            
        if max_concurrent_jobs is not None:
            if max_concurrent_jobs < 1:
                raise ValueError(f"max_concurrent_jobs must be at least 1, got {max_concurrent_jobs}")
            self.max_concurrent_jobs = max_concurrent_jobs
            
        if gpu_allocation_pct is not None:
            if not 10 <= gpu_allocation_pct <= 100:
                raise ValueError(f"gpu_allocation_pct must be 10-100, got {gpu_allocation_pct}")
            self.gpu_allocation_pct = gpu_allocation_pct
        
        logger.info(
            f"Updated compute allocation: CPU={self.cpu_allocation_pct}%, "
            f"Memory={self.memory_allocation_pct}%, Jobs={self.max_concurrent_jobs}, "
            f"GPU={self.gpu_allocation_pct}%"
        )
        
        return {
            "cpu_allocation_pct": self.cpu_allocation_pct,
            "memory_allocation_pct": self.memory_allocation_pct,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "gpu_allocation_pct": self.gpu_allocation_pct,
        }

    # ── Phase 2: remote shard-execute handler ─────────────────────

    MAX_SHARD_BYTES = 10 * 1024 * 1024

    async def _on_direct_message(self, msg: P2PMessage, peer) -> None:
        """Route MSG_DIRECT messages to subtype-specific handlers.

        Other subscribers may also receive this dispatch; each handler
        filters by subtype and ignores messages it doesn't own.
        """
        subtype = msg.payload.get("subtype", "")
        if subtype == "shard_execute_request":
            await self._on_shard_execute_request(msg, peer)
        elif subtype == "shard_price_quote_request":
            await self._on_shard_price_quote_request(msg, peer)

    async def _on_shard_execute_request(self, msg: P2PMessage, peer) -> None:
        """Accept a shard_execute_request, run execute_shard_locally,
        sign a receipt, and send a shard_execute_response.

        Uses the same execute_shard_locally helper the local executor
        uses, so local and remote execution produce bit-identical output.
        """
        payload = msg.payload
        request_id = payload.get("request_id", "")
        job_id = payload.get("job_id", "")
        shard_index = payload.get("shard_index", 0)

        try:
            decline_reason = self._can_accept_shard(payload)
            if decline_reason is not None:
                await self._send_shard_response(
                    peer.peer_id, request_id,
                    status="declined", shard_index=shard_index,
                    error=decline_reason,
                )
                return

            shard = self._deserialize_shard_from_payload(payload)
            input_bytes = base64.b64decode(payload.get("input_b64", ""))

            self._current_jobs += 1
            self._inflight_requests[request_id] = (peer.peer_id, shard_index)
            try:
                output = execute_shard_locally(shard, input_bytes)
            finally:
                self._current_jobs -= 1
                self._inflight_requests.pop(request_id, None)

            output_bytes = output.tobytes()
            output_hash = hashlib.sha256(output_bytes).hexdigest()
            executed_at = int(time.time())

            sig_payload = build_receipt_signing_payload(
                job_id=job_id,
                shard_index=shard_index,
                output_hash=output_hash,
                executed_at_unix=executed_at,
            )
            signature = self.identity.sign(sig_payload)

            receipt = {
                "job_id": job_id,
                "shard_index": shard_index,
                "provider_id": self.identity.node_id,
                "provider_pubkey_b64": self.identity.public_key_b64,
                "output_hash": output_hash,
                "executed_at_unix": executed_at,
                "signature": signature,
            }

            await self._send_shard_response(
                peer.peer_id, request_id,
                status="completed",
                shard_index=shard_index,
                output_bytes=output_bytes,
                output_shape=list(output.shape),
                output_dtype=str(output.dtype),
                receipt=receipt,
            )
        except Exception as exc:
            logger.warning(
                f"shard_execute_request failed for job {job_id!r} "
                f"shard {shard_index}: {exc}"
            )
            await self._send_shard_response(
                peer.peer_id, request_id,
                status="failed", shard_index=shard_index, error=str(exc),
            )

    def _can_accept_shard(self, payload: Dict[str, Any]) -> Optional[str]:
        """Return None if the shard request is acceptable, else a string
        reason (becomes the response's `error` field).

        Checks (in order):
          1. Capacity: _current_jobs < max_concurrent_jobs
          2. Deadline: deadline_unix > now
          3. Tensor decodable
          4. Payload size: decoded tensor bytes <= 10 MB
          5. Checksum: declared == sha256(tensor_bytes)
        """
        if self._current_jobs >= self.max_concurrent_jobs:
            return "resource_unavailable"

        now = int(time.time())
        deadline = payload.get("deadline_unix", 0)
        if deadline < now:
            return "deadline_past"

        try:
            tensor_bytes = base64.b64decode(payload.get("tensor_data_b64", ""))
        except Exception:
            return "tensor_decode_failed"

        if len(tensor_bytes) > self.MAX_SHARD_BYTES:
            return "shard_too_large"

        declared_checksum = payload.get("checksum", "")
        actual_checksum = hashlib.sha256(tensor_bytes).hexdigest()
        if declared_checksum != actual_checksum:
            return "checksum_mismatch"

        return None

    def _deserialize_shard_from_payload(self, payload: Dict[str, Any]) -> ModelShard:
        """Build a ModelShard from a shard_execute_request payload.

        The executor currently interprets tensor_data as float64 (see
        execute_shard_locally). Payload's tensor_dtype is carried for
        future-proofing but not yet enforced at the executor layer.
        """
        tensor_bytes = base64.b64decode(payload["tensor_data_b64"])
        return ModelShard(
            shard_id=f"{payload.get('job_id', '')}:{payload.get('shard_index', 0)}",
            model_id=payload.get("model_id", ""),
            shard_index=payload["shard_index"],
            total_shards=payload.get("total_shards", 1),
            tensor_data=tensor_bytes,
            tensor_shape=tuple(payload["tensor_shape"]),
            size_bytes=len(tensor_bytes),
            checksum=payload.get("checksum", ""),
        )

    async def _send_shard_response(
        self,
        peer_id: str,
        request_id: str,
        status: str,
        shard_index: int,
        output_bytes: Optional[bytes] = None,
        output_shape: Optional[list] = None,
        output_dtype: Optional[str] = None,
        receipt: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "subtype": "shard_execute_response",
            "request_id": request_id,
            "status": status,
            "shard_index": shard_index,
            "error": error,
        }
        if output_bytes is not None:
            payload["output_b64"] = base64.b64encode(output_bytes).decode()
            payload["output_shape"] = output_shape
            payload["output_dtype"] = output_dtype
        if receipt is not None:
            payload["receipt"] = receipt

        resp = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id=self.identity.node_id,
            payload=payload,
        )
        await self.transport.send_to_peer(peer_id, resp)

    # ── Phase 2.1 Line A wiring + Phase 3 Task 5 ────────────────────

    async def preempt_inflight_shards(self, reason: str = "spot_eviction") -> None:
        """Send status=preempted for every in-flight shard request.

        Called from a SIGTERM handler or cloud pod-eviction hook. The
        requester's RemoteShardDispatcher raises ShardPreemptedError
        (distinct from ShardDispatchError — no reputation penalty), the
        escrow refunds, and a MarketplaceOrchestrator can re-dispatch
        the preempted shards to a fresh eligible pool.

        Idempotent: clearing _inflight_requests as we go means a second
        call immediately returns without re-sending.
        """
        if not self._inflight_requests:
            return

        logger.warning(
            f"preempt_inflight_shards: signaling preemption for "
            f"{len(self._inflight_requests)} in-flight shard(s) "
            f"(reason={reason!r})"
        )

        # Copy + clear first so any late-arriving retries can't race.
        pending = list(self._inflight_requests.items())
        self._inflight_requests.clear()

        for request_id, (peer_id, shard_index) in pending:
            try:
                await self._send_shard_response(
                    peer_id, request_id,
                    status="preempted",
                    shard_index=shard_index,
                    error=reason,
                )
            except Exception as exc:
                logger.warning(
                    f"preempt_inflight_shards: failed to notify peer "
                    f"{peer_id[:12]}… for request {request_id}: {exc}"
                )

    async def _on_shard_price_quote_request(
        self, msg: P2PMessage, peer,
    ) -> None:
        """Handle a shard_price_quote_request and respond with ack or reject.

        Phase 3 Task 5 / design §3.4. Four reject paths:
          - no_active_listing: this ComputeProvider has no MarketplaceAdvertiser
            wired, so no advertised price exists.
          - overloaded: _current_jobs >= max_concurrent_jobs.
          - shard_too_large: requested shard exceeds MAX_SHARD_BYTES.
          - above_ceiling: advertised price exceeds requester's
            max_acceptable_price_ftns.

        On accept, returns an ack with quoted_price_ftns + an expiry
        30 seconds in the future. The ack is signed so the requester
        can bind the subsequent dispatch to this specific quote.
        """
        payload = msg.payload
        request_id = payload.get("request_id", "")
        listing_id = payload.get("listing_id", "")
        shard_index = payload.get("shard_index", 0)
        shard_size_bytes = int(payload.get("shard_size_bytes", 0))
        max_acceptable = float(payload.get("max_acceptable_price_ftns", 0.0))

        reject_reason: Optional[str] = None
        quoted_price: Optional[float] = None

        if self._marketplace_advertiser is None:
            reject_reason = "no_active_listing"
        elif self._current_jobs >= self.max_concurrent_jobs:
            reject_reason = "overloaded"
        elif shard_size_bytes > self.MAX_SHARD_BYTES:
            reject_reason = "shard_too_large"
        else:
            quoted_price = self._marketplace_advertiser.current_price_ftns()
            if quoted_price > max_acceptable:
                reject_reason = "above_ceiling"

        if reject_reason is not None:
            resp_payload = {
                "subtype": "shard_price_quote_reject",
                "request_id": request_id,
                "listing_id": listing_id,
                "shard_index": shard_index,
                "reason": reject_reason,
                "provider_id": self.identity.node_id,
            }
        else:
            quote_expires_unix = int(time.time()) + 30
            sig_src = (
                f"{request_id}||{listing_id}||{shard_index}||"
                f"{quoted_price}||{quote_expires_unix}||{self.identity.node_id}"
            ).encode("utf-8")
            signature = self.identity.sign(sig_src)
            resp_payload = {
                "subtype": "shard_price_quote_ack",
                "request_id": request_id,
                "listing_id": listing_id,
                "shard_index": shard_index,
                "quoted_price_ftns": quoted_price,
                "quote_expires_unix": quote_expires_unix,
                "provider_id": self.identity.node_id,
                "provider_pubkey_b64": self.identity.public_key_b64,
                "signature": signature,
            }

        resp = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id=self.identity.node_id,
            payload=resp_payload,
        )
        await self.transport.send_to_peer(peer.peer_id, resp)


def _cpu_benchmark(iterations: int) -> int:
    """Simple prime sieve for CPU benchmarking (runs in thread pool)."""
    if iterations < 2:
        return 0
    sieve = bytearray(b'\x01') * (iterations + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(iterations**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return sum(sieve)
