"""
InferMesh Scheduler — Abstract Base
====================================
Defines the scheduling interface contract that all concrete scheduler
implementations must satisfy. Provides shared utilities including
placement scoring helpers and preemption logic.
"""

from __future__ import annotations

import abc
import time
from typing import Optional

import structlog

from app.models import (
    InferenceRequest,
    SchedulingAlgorithm,
    SchedulingDecision,
)

logger = structlog.get_logger(__name__)


class BaseScheduler(abc.ABC):
    """
    Abstract base class for all InferMesh scheduling algorithms.

    Subclasses implement `schedule()` which maps a request to a worker.
    All schedulers share:
    - Placement scoring utilities
    - Preemption hooks
    - Decision audit logging
    """

    algorithm: SchedulingAlgorithm

    def __init__(self, gpu_manager, kvcache_manager):
        self.gpu_manager = gpu_manager
        self.kvcache_manager = kvcache_manager
        self._decisions_count = 0
        self._preemptions_count = 0

    # -----------------------------------------------------------------------
    # Abstract interface
    # -----------------------------------------------------------------------

    @abc.abstractmethod
    async def schedule(self, request: InferenceRequest) -> Optional[SchedulingDecision]:
        """
        Schedule `request` onto a GPU worker.

        Returns a SchedulingDecision, or None if no placement is possible.
        """
        ...

    @abc.abstractmethod
    async def can_schedule(self, request: InferenceRequest) -> bool:
        """Return True if the scheduler can find a valid placement."""
        ...

    # -----------------------------------------------------------------------
    # Shared scoring utilities
    # -----------------------------------------------------------------------

    def _memory_fit_score(self, worker, request: InferenceRequest) -> float:
        """
        Score a worker by how well it can accommodate the request's memory.
        Returns 0-1 (1 = perfect fit, 0 = cannot fit).
        """
        # Estimate KV-cache bytes: 2 (K+V) * layers * heads * head_dim * 2 (fp16)
        # Using a simplified model: ~4 bytes per token per layer
        estimated_kv_bytes = request.prompt_tokens * 4 * 32  # 32 layers approx
        free = worker.memory.free_bytes
        if free < estimated_kv_bytes:
            return 0.0
        # Higher score = more free memory (but not too much waste)
        utilization = 1.0 - (worker.memory.utilization_pct / 100.0)
        return min(1.0, utilization)

    def _compute_score(self, worker) -> float:
        """Score based on compute availability (lower utilization = higher score)."""
        return 1.0 - (worker.compute_utilization_pct / 100.0)

    def _locality_score(self, worker, request: InferenceRequest) -> float:
        """
        Score based on prefix cache locality.
        If the worker has a cached prefix, strongly prefer it.
        """
        hit = self.kvcache_manager.has_prefix(worker.worker_id, request.prompt)
        return 1.0 if hit else 0.0

    def _composite_score(
        self,
        worker,
        request: InferenceRequest,
        locality_weight: float = 0.4,
        load_weight: float = 0.6,
    ) -> float:
        """
        Weighted composite placement score for a worker.
        Higher = better placement candidate.
        """
        mem = self._memory_fit_score(worker, request)
        if mem == 0.0:
            return 0.0  # Cannot fit — immediately disqualify

        compute = self._compute_score(worker)
        locality = self._locality_score(worker, request)

        load_score = (mem * 0.5 + compute * 0.5)
        return load_weight * load_score + locality_weight * locality

    # -----------------------------------------------------------------------
    # Audit
    # -----------------------------------------------------------------------

    def _log_decision(self, decision: SchedulingDecision) -> None:
        self._decisions_count += 1
        logger.debug(
            "Scheduling decision",
            algorithm=self.algorithm.value,
            request_id=decision.request_id,
            worker_id=decision.worker_id,
            score=f"{decision.score:.3f}",
            reason=decision.placement_reason,
        )

    def get_stats(self) -> dict:
        return {
            "algorithm": self.algorithm.value,
            "total_decisions": self._decisions_count,
            "total_preemptions": self._preemptions_count,
        }
