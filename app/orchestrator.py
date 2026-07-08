"""
InferMesh Main Orchestrator
=============================
Central coordination layer that wires together all subsystems:
- GPU Resource Manager
- Priority Queue + Admission Controller
- Schedulers (Priority, Topology-Aware, Gang)
- Dynamic Batching Engine
- KV-Cache Manager
- Health Monitor
- Autoscaler (HPA + Predictive)
- Metrics Collector

The orchestrator runs the main scheduling loop that continuously:
1. Dequeues pending requests
2. Schedules them to GPU workers
3. Simulates inference execution
4. Tracks completion and metrics
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from datetime import datetime
from typing import Optional

import structlog

from app.autoscaler.horizontal import HorizontalAutoscaler
from app.autoscaler.predictive import PredictiveAutoscaler
from app.batching.dynamic_batcher import ContinuousBatcher
from app.config import InferMeshConfig, get_config
from app.health.monitor import HealthMonitor
from app.kvcache.kvcache_manager import KVCacheManager
from app.metrics.collectors import MetricsCollector
from app.models import (
    InferenceRequest,
    InferenceResponse,
    RequestStatus,
    SchedulingAlgorithm,
)
from app.queue.admission import AdmissionController
from app.queue.priority_queue import MultiLevelPriorityQueue
from app.resource.gpu_manager import GPUResourceManager
from app.scheduler.gang_scheduler import GangScheduler
from app.scheduler.priority_scheduler import PriorityScheduler
from app.scheduler.topology_scheduler import TopologyAwareScheduler
from app.metrics.registry import (
    E2E_LATENCY_SECONDS,
    REQUEST_TOTAL,
    SLA_VIOLATIONS_TOTAL,
    TTFT_SECONDS,
)

logger = structlog.get_logger(__name__)


class InferMeshOrchestrator:
    """
    The central orchestrator — the "brain" of InferMesh.

    Lifecycle:
    1. `start()` — initializes all subsystems
    2. `submit()` — accepts and queues inference requests
    3. `_scheduling_loop()` — continuously schedules and executes
    4. `stop()` — gracefully shuts down all subsystems
    """

    def __init__(self, config: Optional[InferMeshConfig] = None):
        self.config = config or get_config()

        # Core subsystems
        self.gpu_manager = GPUResourceManager()
        self.kvcache_manager = KVCacheManager()
        self.queue = MultiLevelPriorityQueue()
        self.admission = AdmissionController(self.gpu_manager, self.queue)

        # Schedulers
        self.priority_scheduler = PriorityScheduler(self.gpu_manager, self.kvcache_manager)
        self.topology_scheduler = TopologyAwareScheduler(self.gpu_manager, self.kvcache_manager)
        self.gang_scheduler = GangScheduler(self.gpu_manager, self.kvcache_manager)

        # Per-worker batchers
        self._batchers: dict[str, ContinuousBatcher] = {}

        # Health & autoscaling
        self.health_monitor = HealthMonitor(self.gpu_manager)
        self.hpa = HorizontalAutoscaler(self.gpu_manager, self.queue)
        self.predictive = PredictiveAutoscaler(self.gpu_manager, self.queue)

        # Metrics
        self.metrics_collector = MetricsCollector(self)

        # Request tracking
        self._pending_responses: dict[str, asyncio.Future] = {}
        self._completed_responses: dict[str, InferenceResponse] = {}
        self._running = False
        self._scheduling_task: Optional[asyncio.Task] = None
        self._execution_task: Optional[asyncio.Task] = None

        # Latency tracking
        self._recent_ttft: list[float] = []
        self._recent_e2e: list[float] = []

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def start(self) -> None:
        logger.info("Starting InferMesh orchestrator", version=self.config.version)
        self._running = True

        # Initialize GPU pool
        await self.gpu_manager.start()

        # Register KV-cache for each worker
        for worker in self.gpu_manager.get_all_workers():
            await self.kvcache_manager.register_worker(worker.worker_id)
            self._batchers[worker.worker_id] = ContinuousBatcher(
                worker.worker_id, self.gpu_manager, self.kvcache_manager
            )

        # Start subsystems
        await self.health_monitor.start()
        await self.hpa.start()
        await self.predictive.start()
        await self.metrics_collector.start()

        # Start scheduling loop
        self._scheduling_task = asyncio.create_task(
            self._scheduling_loop(), name="scheduler-loop"
        )
        self._execution_task = asyncio.create_task(
            self._execution_loop(), name="execution-loop"
        )

        logger.info(
            "InferMesh orchestrator ready",
            workers=len(self.gpu_manager.get_all_workers()),
            scheduler=self.config.scheduler.algorithm.value,
        )

    async def stop(self) -> None:
        logger.info("Stopping InferMesh orchestrator")
        self._running = False

        # Cancel loops
        for task in [self._scheduling_task, self._execution_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Stop subsystems
        await self.metrics_collector.stop()
        await self.predictive.stop()
        await self.hpa.stop()
        await self.health_monitor.stop()
        await self.gpu_manager.stop()

    # -----------------------------------------------------------------------
    # Request API
    # -----------------------------------------------------------------------

    async def submit(self, request: InferenceRequest) -> InferenceResponse:
        """
        Submit an inference request. Blocks until completion.
        """
        admitted, reason = await self.admission.admit(request)
        if not admitted:
            REQUEST_TOTAL.labels(
                priority=request.priority.name,
                tenant_id=request.tenant_id,
                status="rejected",
            ).inc()
            return InferenceResponse(
                request_id=request.request_id,
                status=RequestStatus.FAILED,
                error_message=f"Rejected: {reason}",
            )

        REQUEST_TOTAL.labels(
            priority=request.priority.name,
            tenant_id=request.tenant_id,
            status="admitted",
        ).inc()

        # Create future for response
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_responses[request.request_id] = future

        queued = await self.queue.enqueue(request)
        if not queued:
            future.cancel()
            return InferenceResponse(
                request_id=request.request_id,
                status=RequestStatus.FAILED,
                error_message="Queue full",
            )

        # Wait for completion with timeout
        try:
            response = await asyncio.wait_for(
                future,
                timeout=request.sla.max_e2e_latency_ms / 1000 * 2,
            )
            return response
        except asyncio.TimeoutError:
            self._pending_responses.pop(request.request_id, None)
            SLA_VIOLATIONS_TOTAL.labels(
                violation_type="e2e_timeout", priority=request.priority.name
            ).inc()
            return InferenceResponse(
                request_id=request.request_id,
                status=RequestStatus.FAILED,
                error_message="SLA timeout",
                sla_violated=True,
            )

    async def submit_nowait(self, request: InferenceRequest) -> bool:
        """Submit without waiting (fire-and-forget). Returns True if queued."""
        admitted, _ = await self.admission.admit(request)
        if not admitted:
            return False
        return await self.queue.enqueue(request)

    # -----------------------------------------------------------------------
    # Scheduling Loop
    # -----------------------------------------------------------------------

    async def _scheduling_loop(self) -> None:
        """
        Main scheduling loop: dequeue → schedule → assign to batcher.
        Runs at configurable interval (default 5ms).
        """
        interval_s = self.config.scheduler.scheduling_interval_ms / 1000.0
        algorithm = self.config.scheduler.algorithm

        while self._running:
            try:
                # Wait for items
                has_items = await self.queue.wait_for_items(timeout_s=interval_s)
                if not has_items:
                    continue

                # Starvation check on PRIORITY scheduler
                if algorithm == SchedulingAlgorithm.PRIORITY:
                    queued = self.queue.get_all_queued()
                    await self.priority_scheduler.check_and_escalate_starvation(queued)

                # Dequeue a batch of requests
                requests = await self.queue.dequeue_batch(max_size=16)
                if not requests:
                    continue

                # Schedule each request
                tasks = [self._schedule_one(req, algorithm) for req in requests]
                await asyncio.gather(*tasks, return_exceptions=True)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Scheduling loop error", error=str(exc))

    async def _schedule_one(
        self, request: InferenceRequest, algorithm: SchedulingAlgorithm
    ) -> None:
        """Schedule a single request and assign it to the appropriate batcher."""
        # Select scheduler
        if algorithm == SchedulingAlgorithm.TOPOLOGY_AWARE:
            scheduler = self.topology_scheduler
        elif algorithm == SchedulingAlgorithm.GANG:
            scheduler = self.gang_scheduler
        else:
            scheduler = self.priority_scheduler

        decision = await scheduler.schedule(request)

        if decision is None:
            # Re-queue with exponential backoff
            await asyncio.sleep(0.05)
            await self.queue.enqueue(request)
            return

        request.assigned_worker_id = decision.worker_id
        await self.gpu_manager.increment_active_requests(decision.worker_id)

        # Add to worker's batcher
        batcher = self._batchers.get(decision.worker_id)
        if batcher:
            batcher.add_request(request)
        else:
            # No batcher for new autoscaled worker, create one
            batcher = ContinuousBatcher(
                decision.worker_id, self.gpu_manager, self.kvcache_manager
            )
            self._batchers[decision.worker_id] = batcher
            batcher.add_request(request)

    # -----------------------------------------------------------------------
    # Execution Loop (Simulation)
    # -----------------------------------------------------------------------

    async def _execution_loop(self) -> None:
        """
        Simulates the transformer forward pass execution on each worker.
        In a real system, this communicates with the inference backend (vLLM, etc.)
        """
        sim_cfg = self.config.simulation

        while self._running:
            try:
                active_requests = []
                for worker_id, batcher in list(self._batchers.items()):
                    batch = batcher.schedule_step()
                    if batch:
                        # Simulate execution
                        task = asyncio.create_task(
                            self._simulate_batch_execution(worker_id, batcher, batch),
                            name=f"exec-{worker_id}"
                        )
                        active_requests.append(task)

                if active_requests:
                    await asyncio.gather(*active_requests, return_exceptions=True)
                else:
                    await asyncio.sleep(0.001)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Execution loop error", error=str(exc))
                await asyncio.sleep(0.1)

    async def _simulate_batch_execution(
        self, worker_id: str, batcher: ContinuousBatcher, batch
    ) -> None:
        """
        Simulate a single batch forward pass.
        Computes realistic latency based on GPU specs and batch size.
        """
        sim_cfg = self.config.simulation
        worker = self.gpu_manager.get_worker(worker_id)
        if not worker:
            return

        # Compute simulated execution time for the ENTIRE sequence
        # base_prefill_ms_per_token = 0.05ms, base_decode_ms_per_token = 0.5ms
        # This ensures total time is well within SLA (default 5000ms)
        prefill_time = batch.prefill_tokens * sim_cfg.base_prefill_ms_per_token
        decode_time = batch.decode_tokens * sim_cfg.base_decode_ms_per_token

        # Scale by GPU type relative to A100 (A100=312 TFLOPS)
        compute_factor = 312.0 / max(1.0, worker.specs.compute_tflops_bf16)
        noise = random.gauss(1.0, sim_cfg.latency_noise_std)
        total_ms = (prefill_time + decode_time) * compute_factor * max(0.5, noise)

        # Sleep to simulate the execution
        await asyncio.sleep(total_ms / 1000)

        # Complete ALL requests in this batch.
        # The batcher tracks token counts per step; here we advance each slot
        # to completion and resolve their futures.
        for bt in batch.requests:
            slot = batcher._active_slots.get(bt.request_id)
            if slot is None:
                # May be in prefill queue still; just skip
                continue

            # Advance tokens to completion for this simulated pass
            slot.tokens_generated = slot.request.max_new_tokens
            if slot.first_token_time is None:
                slot.first_token_time = time.monotonic() - total_ms / 1000

            await self._complete_request(slot, worker_id, total_ms)
            batcher.remove_request(bt.request_id)
            await self.gpu_manager.decrement_active_requests(worker_id)

        # Also handle prefill-only slots that completed prefill in this step
        # and moved to active_slots but have zero decode tokens in this batch
        # (they'll be picked up on the next step — no action needed here)

    async def _complete_request(self, slot, worker_id: str, e2e_ms: float) -> None:
        """Mark a request as complete and resolve its future."""
        request = slot.request
        request.status = RequestStatus.COMPLETED

        ttft_ms = (
            (time.monotonic() - slot.first_token_time) * 1000
            if slot.first_token_time
            else e2e_ms * 0.3
        )
        tbt_ms = e2e_ms / max(1, slot.tokens_generated) if slot.tokens_generated > 1 else 0

        # SLA check
        sla_violated = (
            ttft_ms > request.sla.max_ttft_ms or
            e2e_ms > request.sla.max_e2e_latency_ms
        )
        if sla_violated:
            SLA_VIOLATIONS_TOTAL.labels(
                violation_type="ttft" if ttft_ms > request.sla.max_ttft_ms else "e2e",
                priority=request.priority.name,
            ).inc()

        response = InferenceResponse(
            request_id=request.request_id,
            generated_text=f"[Simulated output: {slot.tokens_generated} tokens]",
            generated_tokens=slot.tokens_generated,
            status=RequestStatus.COMPLETED,
            worker_id=worker_id,
            ttft_ms=ttft_ms,
            tbt_ms=tbt_ms,
            e2e_latency_ms=e2e_ms,
            tokens_per_second=slot.tokens_generated / max(0.001, e2e_ms / 1000),
            sla_violated=sla_violated,
        )

        # Prometheus metrics
        TTFT_SECONDS.labels(worker_id=worker_id, model_id=request.model_id).observe(ttft_ms / 1000)
        E2E_LATENCY_SECONDS.labels(
            priority=request.priority.name, model_id=request.model_id
        ).observe(e2e_ms / 1000)

        # Track recent latencies
        self._recent_ttft.append(ttft_ms)
        self._recent_e2e.append(e2e_ms)
        if len(self._recent_ttft) > 1000:
            self._recent_ttft = self._recent_ttft[-1000:]
            self._recent_e2e = self._recent_e2e[-1000:]

        # Resolve future if waiting
        future = self._pending_responses.pop(request.request_id, None)
        if future and not future.done():
            future.set_result(response)

        self._completed_responses[request.request_id] = response

        # Prefix cache: store completed prompt prefix
        await self.kvcache_manager.cache_prefix(
            worker_id, request.prompt, block_id=hash(request.request_id) % 2048
        )

        if self.priority_scheduler:
            self.priority_scheduler.remove_inflight(worker_id, request.request_id)

    # -----------------------------------------------------------------------
    # Status API
    # -----------------------------------------------------------------------

    def get_metrics_snapshot(self) -> dict:
        """Return a JSON-serializable metrics snapshot."""
        cluster = self.health_monitor.get_cluster_health()
        workers = self.gpu_manager.get_all_workers()

        avg_util = 0.0
        if workers:
            avg_util = sum(w.compute_utilization_pct for w in workers) / len(workers)

        import numpy as np
        ttft_arr = np.array(self._recent_ttft) if self._recent_ttft else np.array([0.0])
        e2e_arr = np.array(self._recent_e2e) if self._recent_e2e else np.array([0.0])

        return {
            "cluster": {
                "total_workers": cluster.total_workers,
                "healthy_workers": cluster.healthy_workers,
                "overall_status": cluster.overall_status,
                "avg_gpu_utilization_pct": round(avg_util, 2),
            },
            "queue": self.queue.stats(),
            "latency": {
                "ttft_p50_ms": round(float(np.percentile(ttft_arr, 50)), 2),
                "ttft_p95_ms": round(float(np.percentile(ttft_arr, 95)), 2),
                "ttft_p99_ms": round(float(np.percentile(ttft_arr, 99)), 2),
                "e2e_p50_ms": round(float(np.percentile(e2e_arr, 50)), 2),
                "e2e_p95_ms": round(float(np.percentile(e2e_arr, 95)), 2),
                "e2e_p99_ms": round(float(np.percentile(e2e_arr, 99)), 2),
            },
            "kvcache": {
                wid: {
                    "hit_rate": round(s.hit_rate, 4),
                    "utilization_pct": round(s.utilization_pct, 2),
                }
                for wid, s in self.kvcache_manager.get_all_stats().items()
            },
            "admission": self.admission.get_stats(),
        }
