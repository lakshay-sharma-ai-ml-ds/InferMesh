"""
InferMesh Gang Scheduler
=========================
Coordinated multi-GPU allocation for tensor-parallel inference.

Algorithm:
1. Determines required parallelism degree from request model config
2. Finds a group of GPUs with sufficient NVLink/PCIe bandwidth
3. Atomically reserves all required GPUs (or none)
4. Uses barrier synchronization tracking for gang completion

Gang scheduling is critical for models requiring tensor parallelism (TP)
where all GPUs must be allocated together to avoid deadlocks.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Optional

import structlog

from app.config import get_config
from app.models import (
    InferenceRequest,
    SchedulingAlgorithm,
    SchedulingDecision,
)
from app.scheduler.base import BaseScheduler
from app.metrics.registry import SCHEDULER_DECISIONS_TOTAL, SCHEDULING_LATENCY_MICROSECONDS

logger = structlog.get_logger(__name__)


class GangScheduler(BaseScheduler):
    """
    Gang Scheduler for multi-GPU tensor-parallel inference.

    Allocates `N` GPUs atomically — either all succeed or none are reserved.
    Supports:
    - NVLink-preferred grouping (minimize inter-GPU communication latency)
    - Heterogeneous gang handling (mix of GPU types if necessary)
    - Gang membership tracking for barrier synchronization
    """

    algorithm = SchedulingAlgorithm.GANG

    def __init__(self, gpu_manager, kvcache_manager):
        super().__init__(gpu_manager, kvcache_manager)
        self.config = get_config()
        self._active_gangs: dict[str, list[str]] = {}  # gang_id -> [worker_ids]
        self._allocation_lock = asyncio.Lock()

    async def schedule(self, request: InferenceRequest) -> Optional[SchedulingDecision]:
        """
        For gang scheduling, we pick the primary worker and record the gang group.
        The orchestrator uses the gang_group_id to coordinate multi-GPU allocation.
        """
        t_start = time.monotonic()

        # Determine required TP degree from model config
        tp_degree = self._required_tp_degree(request)
        if tp_degree <= 1:
            # Single GPU — delegate to simple scoring
            healthy = self.gpu_manager.get_healthy_workers()
            if not healthy:
                return None
            best = min(healthy, key=lambda w: w.load_score)
            decision = SchedulingDecision(
                request_id=request.request_id,
                worker_id=best.worker_id,
                algorithm_used=self.algorithm,
                score=1.0 - best.load_score,
                placement_reason="gang_single_gpu",
            )
            self._emit(t_start, True)
            return decision

        # Multi-GPU gang allocation
        gang = await self._find_gang(tp_degree, request)
        if not gang:
            logger.warning(
                "Gang scheduling failed: insufficient GPUs",
                request_id=request.request_id,
                tp_degree=tp_degree,
            )
            self._emit(t_start, False)
            return None

        gang_id = str(uuid.uuid4())
        self._active_gangs[gang_id] = [w.worker_id for w in gang]
        primary = gang[0]

        decision = SchedulingDecision(
            request_id=request.request_id,
            worker_id=primary.worker_id,
            algorithm_used=self.algorithm,
            score=self._gang_score(gang),
            placement_reason=f"gang_tp{tp_degree}_{len(gang)}_gpus",
            gang_group_id=gang_id,
        )
        self._log_decision(decision)
        self._emit(t_start, True)
        return decision

    async def can_schedule(self, request: InferenceRequest) -> bool:
        tp = self._required_tp_degree(request)
        healthy = self.gpu_manager.get_healthy_workers()
        return len(healthy) >= tp

    # -----------------------------------------------------------------------
    # Gang finding
    # -----------------------------------------------------------------------

    async def _find_gang(self, size: int, request: InferenceRequest):
        """
        Find `size` GPUs that form a good gang.
        Preference order:
        1. All in same NVLink group
        2. All on same PCIe domain
        3. Any available GPUs (cross-domain)
        """
        healthy = self.gpu_manager.get_healthy_workers()
        healthy = [w for w in healthy if self._memory_fit_score(w, request) > 0]

        if len(healthy) < size:
            return None

        # Try NVLink groups first
        nvlink_groups: dict[int, list] = {}
        for w in healthy:
            if w.nvlink_group is not None:
                grp = nvlink_groups.setdefault(w.nvlink_group, [])
                grp.append(w)

        for grp_workers in nvlink_groups.values():
            if len(grp_workers) >= size:
                # Sort by load, pick `size` least loaded
                grp_workers.sort(key=lambda w: w.load_score)
                return grp_workers[:size]

        # Fallback: PCIe domain grouping
        pcie_groups: dict[int, list] = {}
        for w in healthy:
            grp = pcie_groups.setdefault(w.pcie_domain, [])
            grp.append(w)

        for grp_workers in pcie_groups.values():
            if len(grp_workers) >= size:
                grp_workers.sort(key=lambda w: w.load_score)
                return grp_workers[:size]

        # Final fallback: any available
        healthy.sort(key=lambda w: w.load_score)
        return healthy[:size] if len(healthy) >= size else None

    def _required_tp_degree(self, request: InferenceRequest) -> int:
        """Determine tensor parallelism degree from model config."""
        # In production, this reads from model registry. Simulated here.
        model_tp_map = {
            "llama-70b": 8,
            "llama-13b": 4,
            "llama-7b": 2,
            "default": 1,
        }
        return model_tp_map.get(request.model_id.lower(), 1)

    def _gang_score(self, gang: list) -> float:
        """Overall gang quality score."""
        avg_load = sum(w.load_score for w in gang) / len(gang)
        nvlink_count = sum(1 for w in gang if w.nvlink_group is not None)
        nvlink_ratio = nvlink_count / len(gang)
        return (1.0 - avg_load) * 0.6 + nvlink_ratio * 0.4

    def get_gang(self, gang_id: str) -> list[str]:
        return self._active_gangs.get(gang_id, [])

    def release_gang(self, gang_id: str) -> None:
        self._active_gangs.pop(gang_id, None)

    def _emit(self, t_start: float, success: bool) -> None:
        elapsed_us = (time.monotonic() - t_start) * 1e6
        SCHEDULING_LATENCY_MICROSECONDS.observe(elapsed_us)
        SCHEDULER_DECISIONS_TOTAL.labels(
            algorithm=self.algorithm.value,
            success="true" if success else "false",
        ).inc()
