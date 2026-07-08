"""
InferMesh Topology-Aware Scheduler
====================================
NUMA-aware and NVLink-affinity-based scheduling.

Algorithm:
1. Computes a locality affinity score using the topology graph
2. Prefers NVLink-connected GPU groups for tensor-parallel requests
3. Respects NUMA boundaries to minimize PCIe traversal
4. Falls back to least-loaded when no topology preference applies

This scheduler is particularly effective for multi-GPU inference where
KV-cache migration cost dominates placement decisions.
"""

from __future__ import annotations

import time
from typing import Optional

import structlog

from app.config import get_config
from app.models import (
    InferenceRequest,
    SchedulingAlgorithm,
    SchedulingDecision,
)
from app.scheduler.base import BaseScheduler
from app.metrics.registry import (
    SCHEDULER_DECISIONS_TOTAL,
    SCHEDULING_LATENCY_MICROSECONDS,
)

logger = structlog.get_logger(__name__)


class TopologyAwareScheduler(BaseScheduler):
    """
    Topology-Aware Scheduler that uses the GPU interconnect graph to
    optimize placement for:
    - Prefix-cache locality (prefer workers with cached prompts)
    - NVLink affinity (group TP workers on high-bandwidth links)
    - NUMA-local placement (minimize cross-NUMA traffic)
    - KV-cache migration cost avoidance
    """

    algorithm = SchedulingAlgorithm.TOPOLOGY_AWARE

    def __init__(self, gpu_manager, kvcache_manager):
        super().__init__(gpu_manager, kvcache_manager)
        self.config = get_config()

    async def schedule(self, request: InferenceRequest) -> Optional[SchedulingDecision]:
        t_start = time.monotonic()

        healthy = self.gpu_manager.get_healthy_workers()
        if not healthy:
            SCHEDULER_DECISIONS_TOTAL.labels(
                algorithm=self.algorithm.value, success="false"
            ).inc()
            return None

        # Phase 1: Check prefix cache locality
        cache_hit_workers = [
            w for w in healthy
            if self.kvcache_manager.has_prefix(w.worker_id, request.prompt)
        ]

        if cache_hit_workers:
            # Among cache-hit workers, pick least loaded
            best = min(cache_hit_workers, key=lambda w: w.load_score)
            if self._memory_fit_score(best, request) > 0:
                decision = self._make_decision(
                    request, best,
                    score=1.0,
                    reason="prefix_cache_hit",
                )
                self._emit_metrics(t_start, success=True)
                return decision

        # Phase 2: Topology-scored placement
        scored = []
        for worker in healthy:
            mem_score = self._memory_fit_score(worker, request)
            if mem_score == 0:
                continue

            topo_score = self._topology_score(worker, request, healthy)
            compute_score = self._compute_score(worker)
            # NUMA locality bonus
            numa_bonus = self._numa_score(worker, request)

            composite = (
                0.35 * mem_score +
                0.25 * compute_score +
                0.25 * topo_score +
                0.15 * numa_bonus
            )
            scored.append((composite, worker))

        if not scored:
            self._emit_metrics(t_start, success=False)
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_worker = scored[0]

        decision = self._make_decision(
            request, best_worker,
            score=best_score,
            reason=f"topology_score={best_score:.3f}",
        )
        self._emit_metrics(t_start, success=True)
        return decision

    async def can_schedule(self, request: InferenceRequest) -> bool:
        return any(
            self._memory_fit_score(w, request) > 0
            for w in self.gpu_manager.get_healthy_workers()
        )

    # -----------------------------------------------------------------------
    # Scoring components
    # -----------------------------------------------------------------------

    def _topology_score(
        self, worker, request: InferenceRequest, all_workers
    ) -> float:
        """
        Score based on interconnect bandwidth from this worker to others.
        Workers with more NVLink peers score higher (better for TP).
        """
        nvlink_peers = self.gpu_manager.topology.get_nvlink_group(worker.worker_id)
        nvlink_ratio = len(nvlink_peers) / max(1, len(all_workers) - 1)
        return nvlink_ratio

    def _numa_score(self, worker, request: InferenceRequest) -> float:
        """
        NUMA locality score. Workers on NUMA node 0 get slight preference
        since most system memory is allocated there by default.
        We use a simple heuristic: prefer lower NUMA node IDs.
        """
        # Normalize: node 0 = score 1.0, higher nodes get lower scores
        return 1.0 / (1.0 + worker.numa_node)

    def _make_decision(
        self,
        request: InferenceRequest,
        worker,
        score: float,
        reason: str,
    ) -> SchedulingDecision:
        decision = SchedulingDecision(
            request_id=request.request_id,
            worker_id=worker.worker_id,
            algorithm_used=self.algorithm,
            score=score,
            placement_reason=reason,
            latency_estimate_ms=self._estimate_latency_topo(worker, request),
        )
        self._log_decision(decision)
        return decision

    def _estimate_latency_topo(self, worker, request: InferenceRequest) -> float:
        """Topology-aware latency estimate including NVLink transfer overhead."""
        sim = get_config().simulation
        base_ms = (
            request.prompt_tokens * sim.base_prefill_ms_per_token +
            request.max_new_tokens * sim.base_decode_ms_per_token
        )
        # NVLink reduces communication overhead for large models
        nvlink_peers = self.gpu_manager.topology.get_nvlink_group(worker.worker_id)
        comm_factor = 0.85 if nvlink_peers else 1.0
        load_factor = 1.0 + worker.load_score
        return base_ms * comm_factor * load_factor

    def _emit_metrics(self, t_start: float, success: bool) -> None:
        elapsed_us = (time.monotonic() - t_start) * 1e6
        SCHEDULING_LATENCY_MICROSECONDS.observe(elapsed_us)
        SCHEDULER_DECISIONS_TOTAL.labels(
            algorithm=self.algorithm.value,
            success="true" if success else "false",
        ).inc()
