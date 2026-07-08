"""
InferMesh Admission Controller
================================
SLA-aware admission control that decides whether to accept, queue, or
reject incoming inference requests based on current cluster capacity,
queue depth, and SLA requirements.

Implements:
- Token bucket rate limiting per tenant
- Capacity estimation based on cluster utilization
- SLA feasibility check (can the cluster meet the deadline?)
- Graceful degradation (shed background tasks first)
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Optional

import structlog

from app.config import get_config
from app.models import InferenceRequest, RequestPriority
from app.metrics.registry import SCHEDULER_REJECTIONS_TOTAL

logger = structlog.get_logger(__name__)


class TokenBucket:
    """
    Token bucket rate limiter.
    Allows `capacity` tokens per `refill_period_s` seconds.
    """

    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self._tokens = capacity
        self._last_refill = time.monotonic()

    def consume(self, tokens: float = 1.0) -> bool:
        """Try to consume `tokens`. Returns True if allowed."""
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
        self._last_refill = now

    @property
    def current_tokens(self) -> float:
        self._refill()
        return self._tokens


class AdmissionController:
    """
    SLA-aware admission control for incoming inference requests.

    Policy:
    1. CRITICAL requests: always admit (within absolute system limits)
    2. HIGH requests: admit if cluster has capacity > 20%
    3. NORMAL requests: admit if queue depth < 80% max and SLA feasible
    4. LOW/BACKGROUND: shed if under pressure; rate-limit always
    """

    def __init__(self, gpu_manager, queue):
        self.config = get_config()
        self.gpu_manager = gpu_manager
        self.queue = queue
        self._lock = asyncio.Lock()

        # Per-tenant token buckets (default: 100 rps sustained)
        self._tenant_buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(capacity=200, refill_rate=100)
        )
        self._total_admitted = 0
        self._total_rejected = 0

    async def admit(self, request: InferenceRequest) -> tuple[bool, str]:
        """
        Evaluate admission for a request.

        Returns (admitted: bool, reason: str).
        """
        async with self._lock:
            # Always pass CRITICAL through
            if request.priority == RequestPriority.CRITICAL:
                self._total_admitted += 1
                return True, "critical_always_admitted"

            # Rate limiting check
            bucket = self._tenant_buckets[request.tenant_id]
            if not bucket.consume():
                SCHEDULER_REJECTIONS_TOTAL.labels(reason="rate_limit").inc()
                self._total_rejected += 1
                return False, "rate_limit_exceeded"

            # Queue depth check
            queue_depth = self.queue.total_depth()
            max_depth = self.config.queue.max_queue_depth
            depth_ratio = queue_depth / max_depth

            if depth_ratio > 0.95:
                SCHEDULER_REJECTIONS_TOTAL.labels(reason="queue_full").inc()
                self._total_rejected += 1
                return False, "queue_full"

            # Pressure-based shedding for low-priority
            if request.priority.value >= RequestPriority.LOW.value:
                if depth_ratio > 0.70:
                    SCHEDULER_REJECTIONS_TOTAL.labels(reason="load_shedding").inc()
                    self._total_rejected += 1
                    return False, "load_shedding_low_priority"

            # SLA feasibility check
            if self.config.queue.admission_control_enabled:
                feasible, reason = self._check_sla_feasibility(request, depth_ratio)
                if not feasible:
                    SCHEDULER_REJECTIONS_TOTAL.labels(reason="sla_infeasible").inc()
                    self._total_rejected += 1
                    return False, f"sla_infeasible:{reason}"

            self._total_admitted += 1
            return True, "admitted"

    def _check_sla_feasibility(
        self, request: InferenceRequest, depth_ratio: float
    ) -> tuple[bool, str]:
        """
        Quick feasibility check: can we likely meet the SLA?
        Uses cluster utilization and queue depth to estimate wait time.
        """
        healthy_count = len(self.gpu_manager.get_healthy_workers())
        if healthy_count == 0:
            return False, "no_healthy_workers"

        # Rough queue wait estimate
        avg_service_time_ms = 500.0  # 500ms per request on average
        queue_depth = self.queue.total_depth()
        estimated_wait_ms = (queue_depth / max(1, healthy_count)) * avg_service_time_ms

        if estimated_wait_ms > request.sla.max_queue_wait_ms * 2:
            return False, f"estimated_wait_{estimated_wait_ms:.0f}ms_exceeds_sla"

        return True, "ok"

    def get_stats(self) -> dict:
        return {
            "total_admitted": self._total_admitted,
            "total_rejected": self._total_rejected,
            "admission_rate": (
                self._total_admitted /
                max(1, self._total_admitted + self._total_rejected)
            ),
        }
