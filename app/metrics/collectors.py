"""
InferMesh Metrics Collectors
============================
Background tasks that periodically collect and update Prometheus metrics
from all orchestrator subsystems.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import structlog

from app.metrics.registry import (
    AUTOSCALER_WORKER_COUNT,
    CIRCUIT_BREAKER_STATE,
    GPU_COMPUTE_UTILIZATION,
    GPU_MEMORY_BYTES,
    GPU_MEMORY_UTILIZATION,
    KVCACHE_HIT_RATE,
    KVCACHE_PREFIX_HIT_RATE,
    KVCACHE_UTILIZATION,
    ORCHESTRATOR_INFO,
    REQUEST_IN_FLIGHT,
    WORKER_STATUS,
)
from app.models import CircuitBreakerState, GPUStatus

if TYPE_CHECKING:
    from app.orchestrator import InferMeshOrchestrator

logger = structlog.get_logger(__name__)


class MetricsCollector:
    """
    Periodic metrics collector that snapshots all subsystem states
    and updates Prometheus gauges.
    """

    def __init__(self, orchestrator: "InferMeshOrchestrator", interval_s: float = 5.0):
        self.orchestrator = orchestrator
        self.interval_s = interval_s
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        ORCHESTRATOR_INFO.info({"version": "1.0.0", "env": self.orchestrator.config.env})
        self._running = True
        self._task = asyncio.create_task(self._collect_loop(), name="metrics-collector")
        logger.info("Metrics collector started", interval_s=self.interval_s)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _collect_loop(self) -> None:
        while self._running:
            try:
                await self._collect_once()
            except Exception as exc:
                logger.warning("Metrics collection error", error=str(exc))
            await asyncio.sleep(self.interval_s)

    async def _collect_once(self) -> None:
        """Snapshot all subsystems and push to Prometheus."""
        gpu_manager = self.orchestrator.gpu_manager
        kvcache_manager = self.orchestrator.kvcache_manager

        # GPU worker metrics
        for worker in gpu_manager.get_all_workers():
            wid = worker.worker_id
            gtype = worker.gpu_type.value

            GPU_MEMORY_UTILIZATION.labels(worker_id=wid, gpu_type=gtype).set(
                worker.memory.utilization_pct
            )
            GPU_COMPUTE_UTILIZATION.labels(worker_id=wid, gpu_type=gtype).set(
                worker.compute_utilization_pct
            )
            GPU_MEMORY_BYTES.labels(worker_id=wid, memory_type="total").set(
                worker.memory.total_bytes
            )
            GPU_MEMORY_BYTES.labels(worker_id=wid, memory_type="allocated").set(
                worker.memory.allocated_bytes
            )
            GPU_MEMORY_BYTES.labels(worker_id=wid, memory_type="kvcache").set(
                worker.memory.kvcache_bytes
            )
            GPU_MEMORY_BYTES.labels(worker_id=wid, memory_type="weights").set(
                worker.memory.model_weights_bytes
            )

            # Worker health status
            status_map = {
                GPUStatus.HEALTHY: 1.0,
                GPUStatus.DEGRADED: 0.5,
                GPUStatus.UNREACHABLE: 0.0,
                GPUStatus.DRAINING: 0.3,
                GPUStatus.MAINTENANCE: 0.2,
                GPUStatus.OFFLINE: 0.0,
            }
            WORKER_STATUS.labels(worker_id=wid, gpu_type=gtype).set(
                status_map.get(worker.status, 0.0)
            )

            # Circuit breaker
            cb_map = {
                CircuitBreakerState.CLOSED: 0.0,
                CircuitBreakerState.OPEN: 1.0,
                CircuitBreakerState.HALF_OPEN: 0.5,
            }
            CIRCUIT_BREAKER_STATE.labels(worker_id=wid).set(
                cb_map.get(worker.circuit_breaker_state, 0.0)
            )

            # In-flight requests
            REQUEST_IN_FLIGHT.labels(worker_id=wid).set(worker.active_requests)

        # KV-cache per worker
        for worker_id, stats in kvcache_manager.get_all_stats().items():
            KVCACHE_HIT_RATE.labels(worker_id=worker_id).set(stats.hit_rate)
            KVCACHE_PREFIX_HIT_RATE.labels(worker_id=worker_id).set(stats.prefix_hit_rate)
            KVCACHE_UTILIZATION.labels(worker_id=worker_id).set(stats.utilization_pct)

        # Autoscaler
        AUTOSCALER_WORKER_COUNT.set(len(gpu_manager.get_healthy_workers()))
