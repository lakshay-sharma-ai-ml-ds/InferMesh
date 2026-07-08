"""
InferMesh Priority Scheduler
=============================
Multi-class priority scheduler with preemptive scheduling.

Algorithm:
1. Ranks available workers by composite score
2. Checks if high-priority request can preempt a running low-priority one
3. Respects SLA deadlines for time-critical requests
4. Tracks starvation and escalates priority after timeout
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import structlog

from app.config import get_config
from app.models import (
    InferenceRequest,
    RequestPriority,
    SchedulingAlgorithm,
    SchedulingDecision,
    GPUStatus,
)
from app.scheduler.base import BaseScheduler
from app.metrics.registry import (
    SCHEDULER_DECISIONS_TOTAL,
    SCHEDULER_PREEMPTIONS_TOTAL,
    SCHEDULING_LATENCY_MICROSECONDS,
)

logger = structlog.get_logger(__name__)


class PriorityScheduler(BaseScheduler):
    """
    Priority-based scheduler with:
    - Multi-level priority queues (CRITICAL > HIGH > NORMAL > LOW > BACKGROUND)
    - Preemptive scheduling: higher-priority requests can evict lower ones
    - SLA-deadline-aware placement (prefers workers with lowest E2E estimate)
    - Starvation prevention via priority escalation
    """

    algorithm = SchedulingAlgorithm.PRIORITY

    def __init__(self, gpu_manager, kvcache_manager):
        super().__init__(gpu_manager, kvcache_manager)
        self.config = get_config()

        # Track in-flight requests per worker for preemption decisions
        # worker_id -> [(priority, request_id, started_at)]
        self._inflight: dict[str, list[tuple[int, str, float]]] = {}

        # Starvation tracking: request_id -> first_seen_timestamp
        self._first_seen: dict[str, float] = {}

    # -----------------------------------------------------------------------
    # Core scheduling
    # -----------------------------------------------------------------------

    async def schedule(self, request: InferenceRequest) -> Optional[SchedulingDecision]:
        t_start = time.monotonic()

        # Track for starvation detection
        if request.request_id not in self._first_seen:
            self._first_seen[request.request_id] = t_start

        healthy = self.gpu_manager.get_healthy_workers()
        if not healthy:
            SCHEDULER_DECISIONS_TOTAL.labels(
                algorithm=self.algorithm.value, success="false"
            ).inc()
            return None

        # Score all candidates
        scored = []
        for worker in healthy:
            score = self._composite_score(
                worker,
                request,
                locality_weight=self.config.scheduler.locality_weight,
                load_weight=self.config.scheduler.load_weight,
            )
            if score > 0:
                scored.append((score, worker))

        # Attempt direct placement
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            best_score, best_worker = scored[0]

            decision = SchedulingDecision(
                request_id=request.request_id,
                worker_id=best_worker.worker_id,
                algorithm_used=self.algorithm,
                score=best_score,
                placement_reason=f"best_fit_priority_p{request.priority.value}",
                latency_estimate_ms=self._estimate_latency(best_worker, request),
            )
            self._record_inflight(best_worker.worker_id, request)
            self._log_decision(decision)

            elapsed_us = (time.monotonic() - t_start) * 1e6
            SCHEDULING_LATENCY_MICROSECONDS.observe(elapsed_us)
            SCHEDULER_DECISIONS_TOTAL.labels(
                algorithm=self.algorithm.value, success="true"
            ).inc()
            return decision

        # No direct fit — try preemption if enabled
        if self.config.scheduler.preemption_enabled:
            decision = await self._try_preempt(request)
            if decision:
                elapsed_us = (time.monotonic() - t_start) * 1e6
                SCHEDULING_LATENCY_MICROSECONDS.observe(elapsed_us)
                SCHEDULER_DECISIONS_TOTAL.labels(
                    algorithm=self.algorithm.value, success="true"
                ).inc()
                return decision

        SCHEDULER_DECISIONS_TOTAL.labels(
            algorithm=self.algorithm.value, success="false"
        ).inc()
        return None

    async def can_schedule(self, request: InferenceRequest) -> bool:
        healthy = self.gpu_manager.get_healthy_workers()
        return any(
            self._memory_fit_score(w, request) > 0 for w in healthy
        )

    # -----------------------------------------------------------------------
    # Preemption
    # -----------------------------------------------------------------------

    async def _try_preempt(
        self, incoming: InferenceRequest
    ) -> Optional[SchedulingDecision]:
        """
        Try to preempt the lowest-priority in-flight request to make room
        for the incoming higher-priority request.
        """
        if incoming.priority.value >= RequestPriority.NORMAL.value:
            # Only CRITICAL/HIGH can trigger preemption
            return None

        all_workers = self.gpu_manager.get_all_workers()
        candidates = [w for w in all_workers if w.status == GPUStatus.HEALTHY]

        for worker in candidates:
            inflight = self._inflight.get(worker.worker_id, [])
            if not inflight:
                continue
            # Find lowest priority request on this worker
            inflight.sort(key=lambda x: x[0], reverse=True)  # Highest priority number = lowest priority
            lowest_prio, victim_id, _ = inflight[0]

            # Only preempt if incoming is strictly higher priority
            if incoming.priority.value < lowest_prio:
                self._preemptions_count += 1
                inflight.remove((lowest_prio, victim_id, _))

                decision = SchedulingDecision(
                    request_id=incoming.request_id,
                    worker_id=worker.worker_id,
                    algorithm_used=self.algorithm,
                    score=0.8,
                    placement_reason=f"preempted_p{lowest_prio}_request",
                    preempted_request_id=victim_id,
                )
                self._record_inflight(worker.worker_id, incoming)

                SCHEDULER_PREEMPTIONS_TOTAL.labels(
                    preempted_priority=str(lowest_prio)
                ).inc()
                logger.info(
                    "Preemption triggered",
                    incoming=incoming.request_id,
                    victim=victim_id,
                    worker_id=worker.worker_id,
                )
                return decision

        return None

    # -----------------------------------------------------------------------
    # Starvation Prevention
    # -----------------------------------------------------------------------

    async def check_and_escalate_starvation(
        self, queued_requests: list[InferenceRequest]
    ) -> list[InferenceRequest]:
        """
        Inspect queued requests and escalate priority if waiting too long.
        Returns modified list.
        """
        timeout_ms = self.config.scheduler.starvation_timeout_ms
        now = time.monotonic()
        escalated = []

        for req in queued_requests:
            first_seen = self._first_seen.get(req.request_id, now)
            wait_ms = (now - first_seen) * 1000
            if wait_ms > timeout_ms and req.priority.value > RequestPriority.HIGH.value:
                old_priority = req.priority
                req.priority = RequestPriority(req.priority.value - 1)
                logger.info(
                    "Starvation escalation",
                    request_id=req.request_id,
                    old_priority=old_priority.name,
                    new_priority=req.priority.name,
                    wait_ms=f"{wait_ms:.1f}",
                )
                escalated.append(req)

        return escalated

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _record_inflight(self, worker_id: str, request: InferenceRequest) -> None:
        if worker_id not in self._inflight:
            self._inflight[worker_id] = []
        self._inflight[worker_id].append(
            (request.priority.value, request.request_id, time.monotonic())
        )

    def remove_inflight(self, worker_id: str, request_id: str) -> None:
        inflight = self._inflight.get(worker_id, [])
        self._inflight[worker_id] = [
            x for x in inflight if x[1] != request_id
        ]
        self._first_seen.pop(request_id, None)

    def _estimate_latency(self, worker, request: InferenceRequest) -> float:
        """
        Rough latency estimate in ms based on worker compute and request size.
        """
        sim_cfg = get_config().simulation
        prefill_ms = request.prompt_tokens * sim_cfg.base_prefill_ms_per_token
        decode_ms = request.max_new_tokens * sim_cfg.base_decode_ms_per_token
        # Scale by worker capability (relative to A100 baseline)
        compute_factor = 312.0 / max(1.0, worker.specs.compute_tflops_bf16)
        load_factor = 1.0 + worker.load_score  # higher load = slower
        return (prefill_ms + decode_ms) * compute_factor * load_factor
