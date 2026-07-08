"""
InferMesh Horizontal Autoscaler
================================
Reactive autoscaler that adjusts GPU worker count based on:
- Cluster-wide compute utilization
- Queue depth and wait time
- SLA violation rate

Uses EWMA-smoothed signals to avoid oscillation (thrashing).
Implements scale-up / scale-down cooldowns to prevent rapid flapping.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Optional

import structlog

from app.config import get_config
from app.models import (
    GPU_SPECS_REGISTRY,
    GPUType,
    GPUWorker,
    GPUMemoryState,
    ScalingAction,
    ScalingDecision,
)
from app.metrics.registry import (
    AUTOSCALER_SCALING_EVENTS_TOTAL,
    AUTOSCALER_WORKER_COUNT,
)

logger = structlog.get_logger(__name__)


class HorizontalAutoscaler:
    """
    Horizontal Pod Autoscaler (HPA) analog for GPU workers.

    Scaling policy:
    - Scale UP when: avg_utilization > scale_up_threshold OR queue_depth_ratio > 0.7
    - Scale DOWN when: avg_utilization < scale_down_threshold AND queue is short
    - Cooldown: separate up/down cooldowns to prevent flapping
    - Step size: configurable (default +2 up, -1 down)
    """

    def __init__(self, gpu_manager, queue):
        self.gpu_manager = gpu_manager
        self.queue = queue
        self.config = get_config()
        self._last_scale_up_at: float = 0.0
        self._last_scale_down_at: float = 0.0
        self._task: Optional[asyncio.Task] = None
        self._running = False

        # EWMA state for smoothing
        self._ewma_util: float = 0.0
        self._ewma_queue: float = 0.0

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._autoscale_loop(), name="autoscaler-hpa")
        logger.info("Horizontal autoscaler started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _autoscale_loop(self) -> None:
        while self._running:
            try:
                await self._evaluate()
            except Exception as exc:
                logger.warning("Autoscaler error", error=str(exc))
            await asyncio.sleep(10.0)  # Evaluate every 10s

    async def _evaluate(self) -> Optional[ScalingDecision]:
        cfg = self.config.autoscaler
        if not cfg.enabled:
            return None

        now = time.monotonic()
        workers = self.gpu_manager.get_healthy_workers()
        current_count = len(workers)

        # Compute raw signals
        if workers:
            raw_util = sum(w.load_score for w in workers) / len(workers)
        else:
            raw_util = 1.0  # No workers = max load

        queue_depth = self.queue.total_depth()
        queue_ratio = queue_depth / max(1, self.config.queue.max_queue_depth)

        # EWMA smoothing
        alpha = cfg.ewma_alpha
        self._ewma_util = alpha * raw_util + (1 - alpha) * self._ewma_util
        self._ewma_queue = alpha * queue_ratio + (1 - alpha) * self._ewma_queue

        smoothed_util = self._ewma_util
        smoothed_queue = self._ewma_queue

        logger.debug(
            "Autoscaler eval",
            workers=current_count,
            avg_util=f"{smoothed_util * 100:.1f}%",
            queue_ratio=f"{smoothed_queue * 100:.1f}%",
        )

        AUTOSCALER_WORKER_COUNT.set(current_count)

        # Scale UP decision
        should_up = (
            smoothed_util > cfg.scale_up_threshold or
            smoothed_queue > 0.70
        )
        up_cooldown_ok = (now - self._last_scale_up_at) >= cfg.scale_up_cooldown_s
        at_max = current_count >= cfg.max_workers

        if should_up and up_cooldown_ok and not at_max:
            target = min(cfg.max_workers, current_count + cfg.scale_up_step)
            decision = ScalingDecision(
                action=ScalingAction.SCALE_UP,
                current_workers=current_count,
                target_workers=target,
                reason=f"util={smoothed_util:.2f}_queue={smoothed_queue:.2f}",
            )
            await self._execute_scale_up(current_count, target)
            self._last_scale_up_at = now
            return decision

        # Scale DOWN decision
        should_down = (
            smoothed_util < cfg.scale_down_threshold and
            smoothed_queue < 0.20
        )
        down_cooldown_ok = (now - self._last_scale_down_at) >= cfg.scale_down_cooldown_s
        at_min = current_count <= cfg.min_workers

        if should_down and down_cooldown_ok and not at_min:
            target = max(cfg.min_workers, current_count - cfg.scale_down_step)
            decision = ScalingDecision(
                action=ScalingAction.SCALE_DOWN,
                current_workers=current_count,
                target_workers=target,
                reason=f"avg_util={smoothed_util * 100:.1f}% below floor",
            )
            await self._execute_scale_down(current_count, target)
            self._last_scale_down_at = now
            return decision

        return ScalingDecision(
            action=ScalingAction.NO_OP,
            current_workers=current_count,
            target_workers=current_count,
            reason="within_thresholds",
        )

    async def _execute_scale_up(self, current: int, target: int) -> None:
        """Add (target - current) simulated GPU workers."""
        cfg = self.config.autoscaler
        to_add = target - current
        logger.info(f"Autoscaler scale_up: {current} -> {target} workers")
        AUTOSCALER_SCALING_EVENTS_TOTAL.labels(action="scale_up").inc()

        sim_cfg = self.config.simulation
        gpu_types = sim_cfg.gpu_types
        existing = len(self.gpu_manager.get_all_workers())

        for i in range(to_add):
            idx = existing + i
            gpu_type_str = gpu_types[idx % len(gpu_types)]
            try:
                gpu_type = GPUType(gpu_type_str)
            except ValueError:
                gpu_type = GPUType.SIMULATED

            specs = GPU_SPECS_REGISTRY.get(gpu_type, GPU_SPECS_REGISTRY[GPUType.SIMULATED])
            vram_bytes = int(specs.vram_gb * 1024**3)

            worker = GPUWorker(
                worker_id=f"autoscaled-gpu-{uuid.uuid4().hex[:6]}",
                gpu_type=gpu_type,
                specs=specs,
                memory=GPUMemoryState(
                    total_bytes=vram_bytes,
                    model_weights_bytes=int(vram_bytes * 0.20),
                    allocated_bytes=int(vram_bytes * 0.20),
                ),
                labels={"autoscaled": "true"},
            )
            await self.gpu_manager.add_worker(worker)

    async def _execute_scale_down(self, current: int, target: int) -> None:
        """Drain and remove (current - target) workers."""
        to_remove = current - target
        logger.info(f"Autoscaler scale_down: {current} -> {target} workers")
        AUTOSCALER_SCALING_EVENTS_TOTAL.labels(action="scale_down").inc()

        # Remove autoscaled workers first (prefer to keep static ones)
        workers = self.gpu_manager.get_all_workers()
        autoscaled = [w for w in workers if w.labels.get("autoscaled") == "true"]
        to_drain = autoscaled[:to_remove] or workers[-to_remove:]

        for worker in to_drain:
            logger.info(f"Autoscaler drained simulated GPU {worker.worker_id}")
            await self.gpu_manager.remove_worker(worker.worker_id)
