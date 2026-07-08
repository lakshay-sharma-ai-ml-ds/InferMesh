"""
InferMesh Core Data Models
==========================
Pydantic v2 models for all domain entities used across the orchestrator.
Provides strict type safety, serialization, and validation for inference
requests, GPU resources, scheduling decisions, and metrics.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class GPUType(str, Enum):
    H100_SXM5 = "H100_SXM5"
    H100_PCIe = "H100_PCIe"
    A100_SXM4_80G = "A100_SXM4_80G"
    A100_PCIe_40G = "A100_PCIe_40G"
    A10G = "A10G"
    L40S = "L40S"
    T4 = "T4"
    SIMULATED = "SIMULATED"


class GPUStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNREACHABLE = "UNREACHABLE"
    DRAINING = "DRAINING"
    MAINTENANCE = "MAINTENANCE"
    OFFLINE = "OFFLINE"


class RequestPriority(int, Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class RequestStatus(str, Enum):
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    PREFILLING = "PREFILLING"
    DECODING = "DECODING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    PREEMPTED = "PREEMPTED"


class SchedulingAlgorithm(str, Enum):
    PRIORITY = "priority"
    TOPOLOGY_AWARE = "topology_aware"
    GANG = "gang"
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"


class BatchingStrategy(str, Enum):
    CONTINUOUS = "continuous"        # vLLM-style iteration-level
    CHUNKED_PREFILL = "chunked_prefill"  # split prefill into chunks
    STATIC = "static"               # fixed-size batches
    DYNAMIC = "dynamic"             # size-adaptive


class EvictionPolicy(str, Enum):
    LRU = "lru"
    LFU = "lfu"
    ARC = "arc"    # Adaptive Replacement Cache
    FIFO = "fifo"


class CircuitBreakerState(str, Enum):
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Failing, reject all
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


# ---------------------------------------------------------------------------
# GPU Resource Models
# ---------------------------------------------------------------------------

class GPUSpecs(BaseModel):
    """Hardware specifications for a GPU type."""
    gpu_type: GPUType
    vram_gb: float
    compute_tflops_fp16: float
    compute_tflops_bf16: float
    memory_bandwidth_gbps: float
    nvlink_bandwidth_gbps: float = 0.0
    pcie_bandwidth_gbps: float = 64.0  # PCIe 4.0 x16
    max_batch_size: int = 256
    tensor_parallel_degree: int = 1


GPU_SPECS_REGISTRY: dict[GPUType, GPUSpecs] = {
    GPUType.H100_SXM5: GPUSpecs(
        gpu_type=GPUType.H100_SXM5, vram_gb=80, compute_tflops_fp16=989,
        compute_tflops_bf16=989, memory_bandwidth_gbps=3350,
        nvlink_bandwidth_gbps=900, pcie_bandwidth_gbps=128, max_batch_size=512,
        tensor_parallel_degree=8,
    ),
    GPUType.H100_PCIe: GPUSpecs(
        gpu_type=GPUType.H100_PCIe, vram_gb=80, compute_tflops_fp16=756,
        compute_tflops_bf16=756, memory_bandwidth_gbps=2000,
        pcie_bandwidth_gbps=128, max_batch_size=384, tensor_parallel_degree=4,
    ),
    GPUType.A100_SXM4_80G: GPUSpecs(
        gpu_type=GPUType.A100_SXM4_80G, vram_gb=80, compute_tflops_fp16=312,
        compute_tflops_bf16=312, memory_bandwidth_gbps=2000,
        nvlink_bandwidth_gbps=600, pcie_bandwidth_gbps=64, max_batch_size=256,
        tensor_parallel_degree=8,
    ),
    GPUType.A100_PCIe_40G: GPUSpecs(
        gpu_type=GPUType.A100_PCIe_40G, vram_gb=40, compute_tflops_fp16=312,
        compute_tflops_bf16=312, memory_bandwidth_gbps=1555,
        pcie_bandwidth_gbps=64, max_batch_size=128, tensor_parallel_degree=4,
    ),
    GPUType.A10G: GPUSpecs(
        gpu_type=GPUType.A10G, vram_gb=24, compute_tflops_fp16=125,
        compute_tflops_bf16=125, memory_bandwidth_gbps=600,
        pcie_bandwidth_gbps=64, max_batch_size=64, tensor_parallel_degree=1,
    ),
    GPUType.L40S: GPUSpecs(
        gpu_type=GPUType.L40S, vram_gb=48, compute_tflops_fp16=362,
        compute_tflops_bf16=362, memory_bandwidth_gbps=864,
        pcie_bandwidth_gbps=64, max_batch_size=128, tensor_parallel_degree=2,
    ),
    GPUType.T4: GPUSpecs(
        gpu_type=GPUType.T4, vram_gb=16, compute_tflops_fp16=65,
        compute_tflops_bf16=65, memory_bandwidth_gbps=320,
        pcie_bandwidth_gbps=32, max_batch_size=32, tensor_parallel_degree=1,
    ),
    GPUType.SIMULATED: GPUSpecs(
        gpu_type=GPUType.SIMULATED, vram_gb=40, compute_tflops_fp16=312,
        compute_tflops_bf16=312, memory_bandwidth_gbps=1555,
        pcie_bandwidth_gbps=64, max_batch_size=128, tensor_parallel_degree=4,
    ),
}


class GPUMemoryState(BaseModel):
    """Real-time GPU memory tracking."""
    total_bytes: int
    allocated_bytes: int = 0
    reserved_bytes: int = 0
    kvcache_bytes: int = 0
    model_weights_bytes: int = 0

    @property
    def free_bytes(self) -> int:
        return self.total_bytes - self.allocated_bytes

    @property
    def utilization_pct(self) -> float:
        return (self.allocated_bytes / self.total_bytes * 100) if self.total_bytes > 0 else 0.0

    @property
    def fragmentation_ratio(self) -> float:
        """Ratio of unusable free space due to fragmentation (0=none, 1=full)."""
        return self.reserved_bytes / self.total_bytes if self.total_bytes > 0 else 0.0


class GPUWorker(BaseModel):
    """Represents a single GPU worker in the cluster."""
    model_config = {"arbitrary_types_allowed": True}

    worker_id: str = Field(default_factory=lambda: f"gpu-{uuid.uuid4().hex[:8]}")
    gpu_type: GPUType = GPUType.SIMULATED
    specs: GPUSpecs = Field(default_factory=lambda: GPU_SPECS_REGISTRY[GPUType.SIMULATED])
    status: GPUStatus = GPUStatus.HEALTHY
    memory: GPUMemoryState = Field(default=None)
    compute_utilization_pct: float = 0.0
    active_requests: int = 0
    total_requests_served: int = 0
    numa_node: int = 0
    pcie_domain: int = 0
    nvlink_group: Optional[int] = None
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    circuit_breaker_state: CircuitBreakerState = CircuitBreakerState.CLOSED
    labels: dict[str, str] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.memory is None:
            vram_bytes = int(self.specs.vram_gb * 1024**3)
            self.memory = GPUMemoryState(total_bytes=vram_bytes)

    @property
    def is_available(self) -> bool:
        return self.status == GPUStatus.HEALTHY and \
               self.circuit_breaker_state != CircuitBreakerState.OPEN

    @property
    def load_score(self) -> float:
        """Composite load score [0, 1]. Lower = more available."""
        mem_load = self.memory.utilization_pct / 100.0
        comp_load = self.compute_utilization_pct / 100.0
        return 0.6 * mem_load + 0.4 * comp_load


# ---------------------------------------------------------------------------
# Inference Request Models
# ---------------------------------------------------------------------------

class SLAConfig(BaseModel):
    """Service Level Agreement parameters for a request class."""
    max_ttft_ms: float = 500.0        # Time To First Token
    max_tbt_ms: float = 100.0         # Time Between Tokens
    max_e2e_latency_ms: float = 5000.0
    max_queue_wait_ms: float = 2000.0


class InferenceRequest(BaseModel):
    """A single LLM inference request."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str
    prompt_tokens: int = Field(ge=1)
    max_new_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)
    priority: RequestPriority = RequestPriority.NORMAL
    sla: SLAConfig = Field(default_factory=SLAConfig)
    tenant_id: str = "default"
    model_id: str = "default"
    stream: bool = False
    arrived_at: datetime = Field(default_factory=datetime.utcnow)
    status: RequestStatus = RequestStatus.PENDING
    assigned_worker_id: Optional[str] = None

    @field_validator("prompt_tokens", mode="before")
    @classmethod
    def estimate_tokens_if_zero(cls, v: int, info: Any) -> int:
        return v if v > 0 else 1


class InferenceResponse(BaseModel):
    """Completed inference response with timing metrics."""
    request_id: str
    generated_text: str = ""
    generated_tokens: int = 0
    status: RequestStatus
    worker_id: Optional[str] = None
    # Timing
    queued_at: Optional[datetime] = None
    prefill_started_at: Optional[datetime] = None
    first_token_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    # Derived metrics (ms)
    queue_wait_ms: Optional[float] = None
    ttft_ms: Optional[float] = None       # Time to first token
    tbt_ms: Optional[float] = None        # Avg time between tokens
    e2e_latency_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    # SLA
    sla_violated: bool = False
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Batch Models
# ---------------------------------------------------------------------------

class BatchToken(BaseModel):
    """Token budget allocation for a batch slot."""
    request_id: str
    prompt_tokens: int
    decode_tokens_remaining: int
    kvcache_block_ids: list[int] = Field(default_factory=list)


class InferenceBatch(BaseModel):
    """A batch of requests scheduled for a single forward pass."""
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    worker_id: str
    requests: list[BatchToken]
    strategy: BatchingStrategy = BatchingStrategy.CONTINUOUS
    created_at: datetime = Field(default_factory=datetime.utcnow)
    prefill_tokens: int = 0   # Tokens being prefilled this step
    decode_tokens: int = 0    # Tokens being decoded this step
    is_chunked_prefill: bool = False
    chunk_index: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prefill_tokens + self.decode_tokens

    @property
    def batch_size(self) -> int:
        return len(self.requests)


# ---------------------------------------------------------------------------
# Scheduling Models
# ---------------------------------------------------------------------------

class SchedulingDecision(BaseModel):
    """Result of a scheduling algorithm."""
    request_id: str
    worker_id: str
    algorithm_used: SchedulingAlgorithm
    score: float = 0.0
    placement_reason: str = ""
    preempted_request_id: Optional[str] = None
    gang_group_id: Optional[str] = None
    latency_estimate_ms: float = 0.0
    decided_at: datetime = Field(default_factory=datetime.utcnow)


class SchedulerStats(BaseModel):
    """Scheduler telemetry snapshot."""
    total_scheduled: int = 0
    total_preempted: int = 0
    total_rejected: int = 0
    queue_depths: dict[str, int] = Field(default_factory=dict)
    avg_queue_wait_ms: float = 0.0
    scheduling_latency_us: float = 0.0  # microseconds
    algorithm_distribution: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# KV-Cache Models
# ---------------------------------------------------------------------------

class KVCacheBlock(BaseModel):
    """A single block in the KV-cache."""
    block_id: int
    worker_id: str
    size_bytes: int
    token_ids: list[int] = Field(default_factory=list)
    prefix_hash: Optional[str] = None
    ref_count: int = 0
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    is_shared: bool = False  # True if shared via prefix cache


class KVCacheStats(BaseModel):
    """KV-cache telemetry."""
    total_blocks: int
    used_blocks: int
    free_blocks: int
    hit_rate: float = 0.0
    prefix_hit_rate: float = 0.0
    evictions_total: int = 0
    bytes_saved_by_prefix_sharing: int = 0

    @property
    def utilization_pct(self) -> float:
        return (self.used_blocks / self.total_blocks * 100) if self.total_blocks > 0 else 0.0


# ---------------------------------------------------------------------------
# Health Models
# ---------------------------------------------------------------------------

class HealthCheckResult(BaseModel):
    """Result of a health probe."""
    worker_id: str
    is_healthy: bool
    latency_ms: float
    checked_at: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class ClusterHealth(BaseModel):
    """Cluster-wide health summary."""
    total_workers: int
    healthy_workers: int
    degraded_workers: int
    unreachable_workers: int
    overall_status: str = "HEALTHY"
    checked_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def availability_pct(self) -> float:
        return (self.healthy_workers / self.total_workers * 100) if self.total_workers > 0 else 0.0


# ---------------------------------------------------------------------------
# Autoscaler Models
# ---------------------------------------------------------------------------

class ScalingAction(str, Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_OP = "no_op"


class ScalingDecision(BaseModel):
    """Decision made by the autoscaler."""
    action: ScalingAction
    current_workers: int
    target_workers: int
    reason: str
    confidence: float = 1.0
    decided_at: datetime = Field(default_factory=datetime.utcnow)
    predicted_load: Optional[float] = None


# ---------------------------------------------------------------------------
# Metrics Snapshot
# ---------------------------------------------------------------------------

class OrchestratorMetrics(BaseModel):
    """Full metrics snapshot of the orchestrator."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    cluster_health: ClusterHealth
    kvcache_stats: KVCacheStats
    scheduler_stats: SchedulerStats
    total_requests_in_flight: int = 0
    total_tokens_per_second: float = 0.0
    avg_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    avg_e2e_latency_ms: float = 0.0
    p99_e2e_latency_ms: float = 0.0
    sla_violation_rate: float = 0.0
    gpu_utilization_avg: float = 0.0
    queue_wait_avg_ms: float = 0.0
