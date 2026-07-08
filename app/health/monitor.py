"""
InferMesh Health Monitor
=========================
Heartbeat-based GPU liveness monitoring with circuit breaker integration.

Design:
- Each worker sends periodic heartbeats (simulated in sim mode)
- Health monitor tracks consecutive failures/successes
- Circuit breaker: CLOSED → OPEN (on failures) → HALF_OPEN → CLOSED (recovery)
- Dead worker detection triggers rescheduling of in-flight requests
- Exponential backoff for recovery probes
"""

from __future__ import annotations

import asyncio
import random
import time
from datetime import datetime
from typing import Optional

import structlog

from app.config import get_config
from app.models import (
    CircuitBreakerState,
    GPUStatus,
    GPUWorker,
    HealthCheckResult,
    ClusterHealth,
)
from app.metrics.registry import (
    GPU_FAILURES_TOTAL,
    GPU_RECOVERY_TIME_SECONDS,
    HEALTH_CHECK_LATENCY_SECONDS,
)

logger = structlog.get_logger(__name__)


class CircuitBreaker:
    """
    Per-worker circuit breaker implementing the three-state FSM:

    CLOSED → OPEN: After `failure_threshold` consecutive failures
    OPEN → HALF_OPEN: After `open_duration_ms` milliseconds
    HALF_OPEN → CLOSED: After `success_threshold` consecutive successes
    HALF_OPEN → OPEN: On any failure during probe
    """

    def __init__(self, worker_id: str, cfg):
        self.worker_id = worker_id
        self.cfg = cfg
        self.state = CircuitBreakerState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._opened_at: Optional[float] = None
        self._failure_start_at: Optional[float] = None

    def record_success(self) -> None:
        self._consecutive_failures = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self._consecutive_successes += 1
            if self._consecutive_successes >= self.cfg.circuit_breaker_success_threshold:
                self._transition(CircuitBreakerState.CLOSED)
                logger.info("Circuit breaker closing after successful probe", worker_id=self.worker_id)
        elif self.state == CircuitBreakerState.CLOSED:
            self._consecutive_successes += 1

    def record_failure(self) -> None:
        self._consecutive_successes = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self._transition(CircuitBreakerState.OPEN)
            logger.warning("Circuit breaker re-opening during probe", worker_id=self.worker_id)
            return

        self._consecutive_failures += 1
        if self._failure_start_at is None:
            self._failure_start_at = time.monotonic()

        if (self.state == CircuitBreakerState.CLOSED and
                self._consecutive_failures >= self.cfg.circuit_breaker_failure_threshold):
            self._transition(CircuitBreakerState.OPEN)
            logger.error(
                "Circuit breaker OPEN",
                worker_id=self.worker_id,
                failures=self._consecutive_failures,
            )

    def should_allow_probe(self) -> bool:
        """Check if OPEN circuit should transition to HALF_OPEN for a probe."""
        if self.state != CircuitBreakerState.OPEN:
            return False
        open_for_ms = (time.monotonic() - self._opened_at) * 1000
        return open_for_ms >= self.cfg.circuit_breaker_open_duration_ms

    def maybe_half_open(self) -> None:
        if self.should_allow_probe():
            self._transition(CircuitBreakerState.HALF_OPEN)
            logger.info("Circuit breaker half-opening for probe", worker_id=self.worker_id)

    def _transition(self, new_state: CircuitBreakerState) -> None:
        old = self.state
        self.state = new_state
        if new_state == CircuitBreakerState.OPEN:
            self._opened_at = time.monotonic()
        elif new_state == CircuitBreakerState.CLOSED:
            self._failure_start_at = None
            self._opened_at = None
            self._consecutive_failures = 0
            self._consecutive_successes = 0


# ---------------------------------------------------------------------------
# Health Monitor
# ---------------------------------------------------------------------------

class HealthMonitor:
    """
    Cluster-wide GPU health monitoring.

    Responsibilities:
    - Periodic heartbeat collection from each GPU worker
    - Failure detection and circuit breaker management
    - Triggering GPU status updates for the scheduler
    - Recovery tracking (MTTR metrics)
    """

    def __init__(self, gpu_manager):
        self.gpu_manager = gpu_manager
        self.config = get_config()
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._last_heartbeats: dict[str, float] = {}
        self._failure_detected_at: dict[str, float] = {}
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        # Initialize circuit breakers for existing workers
        for worker in self.gpu_manager.get_all_workers():
            self._ensure_circuit_breaker(worker.worker_id)
            self._last_heartbeats[worker.worker_id] = time.monotonic()

        self._monitor_task = asyncio.create_task(
            self._monitor_loop(), name="health-monitor"
        )
        logger.info("Health monitor started")

    async def stop(self) -> None:
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self) -> None:
        cfg = self.config.health
        interval_s = cfg.heartbeat_interval_ms / 1000.0

        while self._running:
            try:
                await self._check_all_workers()
            except Exception as exc:
                logger.warning("Health monitor error", error=str(exc))
            await asyncio.sleep(interval_s)

    async def _check_all_workers(self) -> None:
        tasks = [
            self._check_worker(worker)
            for worker in self.gpu_manager.get_all_workers()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_worker(self, worker: GPUWorker) -> None:
        cfg = self.config.health
        cb = self._ensure_circuit_breaker(worker.worker_id)

        # Check if circuit should attempt half-open probe
        cb.maybe_half_open()

        # Simulate heartbeat check
        t_start = time.monotonic()
        result = await self._probe_worker(worker, cb)
        latency_s = time.monotonic() - t_start

        HEALTH_CHECK_LATENCY_SECONDS.labels(worker_id=worker.worker_id).observe(latency_s)

        if result.is_healthy:
            cb.record_success()
            self._last_heartbeats[worker.worker_id] = time.monotonic()

            # Recovery tracking
            if worker.worker_id in self._failure_detected_at:
                recovery_time = time.monotonic() - self._failure_detected_at.pop(worker.worker_id)
                GPU_RECOVERY_TIME_SECONDS.labels(worker_id=worker.worker_id).observe(recovery_time)
                logger.info(
                    f"GPU {worker.worker_id} recovered -> HEALTHY",
                    recovery_time_s=f"{recovery_time:.2f}",
                )

            if worker.status != GPUStatus.HEALTHY:
                await self.gpu_manager.mark_worker_status(worker.worker_id, GPUStatus.HEALTHY)
        else:
            cb.record_failure()

            # Check heartbeat timeout
            last_hb = self._last_heartbeats.get(worker.worker_id, time.monotonic())
            timeout_ms = cfg.heartbeat_timeout_ms
            if (time.monotonic() - last_hb) * 1000 > timeout_ms:
                if worker.worker_id not in self._failure_detected_at:
                    self._failure_detected_at[worker.worker_id] = time.monotonic()
                    logger.warning(
                        f"GPU {worker.worker_id} missed heartbeat deadline; marking UNREACHABLE"
                    )
                    GPU_FAILURES_TOTAL.labels(
                        worker_id=worker.worker_id,
                        failure_type="heartbeat_timeout",
                    ).inc()

                if worker.status == GPUStatus.HEALTHY:
                    await self.gpu_manager.mark_worker_status(
                        worker.worker_id, GPUStatus.UNREACHABLE
                    )

        # Update worker circuit breaker state
        worker.circuit_breaker_state = cb.state

    async def _probe_worker(
        self, worker: GPUWorker, cb: CircuitBreaker
    ) -> HealthCheckResult:
        """
        Simulate or perform actual health probe.
        In simulation mode: workers fail intermittently per sim config.
        """
        cfg = self.config.health
        sim_cfg = self.config.simulation

        # If circuit is OPEN (not half-open), don't probe
        if cb.state == CircuitBreakerState.OPEN:
            return HealthCheckResult(
                worker_id=worker.worker_id,
                is_healthy=False,
                latency_ms=0.0,
                error_message="circuit_open",
            )

        # Simulate probe latency
        probe_latency_ms = random.gauss(5, 1)  # ~5ms ± 1ms
        await asyncio.sleep(probe_latency_ms / 1000)

        # Check worker status (simulation)
        is_healthy = worker.status != GPUStatus.OFFLINE

        # For UNREACHABLE workers, simulate recovery probabilistically
        if worker.status == GPUStatus.UNREACHABLE:
            # 80% chance of recovery per probe (simulating transient failures)
            is_healthy = random.random() > 0.2

        return HealthCheckResult(
            worker_id=worker.worker_id,
            is_healthy=is_healthy,
            latency_ms=probe_latency_ms,
            error_message=None if is_healthy else "probe_failed",
        )

    def _ensure_circuit_breaker(self, worker_id: str) -> CircuitBreaker:
        if worker_id not in self._circuit_breakers:
            self._circuit_breakers[worker_id] = CircuitBreaker(
                worker_id, self.config.health
            )
        return self._circuit_breakers[worker_id]

    def get_cluster_health(self) -> ClusterHealth:
        workers = self.gpu_manager.get_all_workers()
        healthy = sum(1 for w in workers if w.status == GPUStatus.HEALTHY)
        degraded = sum(1 for w in workers if w.status == GPUStatus.DEGRADED)
        unreachable = sum(1 for w in workers if w.status == GPUStatus.UNREACHABLE)

        overall = "HEALTHY"
        if healthy == 0:
            overall = "CRITICAL"
        elif unreachable > len(workers) // 2:
            overall = "DEGRADED"
        elif degraded > 0:
            overall = "WARNING"

        return ClusterHealth(
            total_workers=len(workers),
            healthy_workers=healthy,
            degraded_workers=degraded,
            unreachable_workers=unreachable,
            overall_status=overall,
        )
