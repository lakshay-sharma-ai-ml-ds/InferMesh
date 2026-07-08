"""
InferMesh Metrics Registry
==========================
Prometheus metric definitions for the entire orchestrator.
All metrics are defined once here and imported throughout the codebase.
"""

from __future__ import annotations

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    Summary,
    CollectorRegistry,
    REGISTRY,
)

# ---------------------------------------------------------------------------
# Custom registry (avoids double-registration in tests)
# ---------------------------------------------------------------------------
INFERMESH_REGISTRY = REGISTRY


# ---------------------------------------------------------------------------
# Request Metrics
# ---------------------------------------------------------------------------
REQUEST_TOTAL = Counter(
    "infermesh_requests_total",
    "Total number of inference requests received",
    ["priority", "tenant_id", "status"],
)

REQUEST_IN_FLIGHT = Gauge(
    "infermesh_requests_in_flight",
    "Number of requests currently being processed",
    ["worker_id"],
)

REQUEST_QUEUE_DEPTH = Gauge(
    "infermesh_queue_depth",
    "Number of requests waiting in the priority queue",
    ["priority"],
)

REQUEST_QUEUE_WAIT_SECONDS = Histogram(
    "infermesh_queue_wait_seconds",
    "Time requests spend waiting in queue",
    ["priority", "tenant_id"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# ---------------------------------------------------------------------------
# Latency Metrics
# ---------------------------------------------------------------------------
TTFT_SECONDS = Histogram(
    "infermesh_ttft_seconds",
    "Time To First Token latency",
    ["worker_id", "model_id"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0),
)

TBT_SECONDS = Histogram(
    "infermesh_tbt_seconds",
    "Time Between Tokens (inter-token latency)",
    ["worker_id"],
    buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
)

E2E_LATENCY_SECONDS = Histogram(
    "infermesh_e2e_latency_seconds",
    "End-to-end request latency",
    ["priority", "model_id"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

SCHEDULING_LATENCY_MICROSECONDS = Histogram(
    "infermesh_scheduling_latency_microseconds",
    "Time taken by the scheduler to make a placement decision",
    buckets=(10, 50, 100, 250, 500, 1000, 2500, 5000, 10000),
)

# ---------------------------------------------------------------------------
# Throughput Metrics
# ---------------------------------------------------------------------------
TOKENS_GENERATED_TOTAL = Counter(
    "infermesh_tokens_generated_total",
    "Total tokens generated across all workers",
    ["worker_id"],
)

TOKENS_PER_SECOND = Gauge(
    "infermesh_tokens_per_second",
    "Current token generation throughput",
    ["worker_id"],
)

BATCH_SIZE = Histogram(
    "infermesh_batch_size",
    "Distribution of batch sizes used for inference",
    ["strategy", "worker_id"],
    buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
)

BATCH_TOKENS = Histogram(
    "infermesh_batch_tokens",
    "Total tokens (prefill + decode) per batch",
    ["worker_id"],
    buckets=(64, 128, 256, 512, 1024, 2048, 4096, 8192),
)

# ---------------------------------------------------------------------------
# GPU Resource Metrics
# ---------------------------------------------------------------------------
GPU_MEMORY_UTILIZATION = Gauge(
    "infermesh_gpu_memory_utilization_pct",
    "GPU VRAM utilization percentage",
    ["worker_id", "gpu_type"],
)

GPU_COMPUTE_UTILIZATION = Gauge(
    "infermesh_gpu_compute_utilization_pct",
    "GPU compute utilization percentage",
    ["worker_id", "gpu_type"],
)

GPU_MEMORY_BYTES = Gauge(
    "infermesh_gpu_memory_bytes",
    "GPU memory usage in bytes",
    ["worker_id", "memory_type"],  # memory_type: total, allocated, kvcache, weights
)

WORKER_STATUS = Gauge(
    "infermesh_worker_status",
    "Worker status (1=HEALTHY, 0.5=DEGRADED, 0=UNREACHABLE)",
    ["worker_id", "gpu_type"],
)

# ---------------------------------------------------------------------------
# KV-Cache Metrics
# ---------------------------------------------------------------------------
KVCACHE_HIT_RATE = Gauge(
    "infermesh_kvcache_hit_rate",
    "KV-cache hit rate (0-1)",
    ["worker_id"],
)

KVCACHE_PREFIX_HIT_RATE = Gauge(
    "infermesh_kvcache_prefix_hit_rate",
    "Prefix cache hit rate (0-1)",
    ["worker_id"],
)

KVCACHE_UTILIZATION = Gauge(
    "infermesh_kvcache_utilization_pct",
    "Percentage of KV-cache blocks in use",
    ["worker_id"],
)

KVCACHE_EVICTIONS_TOTAL = Counter(
    "infermesh_kvcache_evictions_total",
    "Total KV-cache block evictions",
    ["worker_id", "eviction_policy"],
)

KVCACHE_BYTES_SAVED = Gauge(
    "infermesh_kvcache_bytes_saved_by_prefix_sharing",
    "Memory saved by prefix sharing in bytes",
    ["worker_id"],
)

# ---------------------------------------------------------------------------
# Health & Circuit Breaker
# ---------------------------------------------------------------------------
CIRCUIT_BREAKER_STATE = Gauge(
    "infermesh_circuit_breaker_state",
    "Circuit breaker state (0=CLOSED, 1=OPEN, 0.5=HALF_OPEN)",
    ["worker_id"],
)

HEALTH_CHECK_LATENCY_SECONDS = Histogram(
    "infermesh_health_check_latency_seconds",
    "Health probe round-trip latency",
    ["worker_id"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0),
)

GPU_FAILURES_TOTAL = Counter(
    "infermesh_gpu_failures_total",
    "Total GPU failure events detected",
    ["worker_id", "failure_type"],
)

GPU_RECOVERY_TIME_SECONDS = Histogram(
    "infermesh_gpu_recovery_time_seconds",
    "Time from failure detection to recovery",
    ["worker_id"],
    buckets=(1, 5, 10, 30, 60, 120, 300),
)

# ---------------------------------------------------------------------------
# Autoscaler Metrics
# ---------------------------------------------------------------------------
AUTOSCALER_WORKER_COUNT = Gauge(
    "infermesh_autoscaler_worker_count",
    "Current number of active GPU workers",
)

AUTOSCALER_SCALING_EVENTS_TOTAL = Counter(
    "infermesh_autoscaler_scaling_events_total",
    "Total autoscaler scaling events",
    ["action"],  # action: scale_up, scale_down
)

AUTOSCALER_PREDICTED_LOAD = Gauge(
    "infermesh_autoscaler_predicted_load",
    "EWMA-predicted cluster load (0-1)",
)

# ---------------------------------------------------------------------------
# Scheduler Metrics
# ---------------------------------------------------------------------------
SCHEDULER_DECISIONS_TOTAL = Counter(
    "infermesh_scheduler_decisions_total",
    "Total scheduling decisions made",
    ["algorithm", "success"],
)

SCHEDULER_PREEMPTIONS_TOTAL = Counter(
    "infermesh_scheduler_preemptions_total",
    "Total request preemptions",
    ["preempted_priority"],
)

SCHEDULER_REJECTIONS_TOTAL = Counter(
    "infermesh_scheduler_rejections_total",
    "Total requests rejected by admission control",
    ["reason"],
)

# ---------------------------------------------------------------------------
# SLA Metrics
# ---------------------------------------------------------------------------
SLA_VIOLATIONS_TOTAL = Counter(
    "infermesh_sla_violations_total",
    "Total SLA violations",
    ["violation_type", "priority"],  # violation_type: ttft, tbt, e2e, queue
)

SLA_COMPLIANCE_RATE = Gauge(
    "infermesh_sla_compliance_rate",
    "SLA compliance rate over recent window (0-1)",
    ["priority"],
)

# ---------------------------------------------------------------------------
# System Info
# ---------------------------------------------------------------------------
ORCHESTRATOR_INFO = Info(
    "infermesh_orchestrator",
    "InferMesh orchestrator build information",
)
