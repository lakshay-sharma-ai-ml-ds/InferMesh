"""Unit tests for scheduler components."""

from __future__ import annotations

import asyncio
import pytest

from app.models import InferenceRequest, RequestPriority, SchedulingAlgorithm


@pytest.mark.asyncio
async def test_priority_scheduler_basic(gpu_manager, kvcache_manager, sample_request):
    """Scheduler returns a decision for a valid request."""
    from app.scheduler.priority_scheduler import PriorityScheduler

    scheduler = PriorityScheduler(gpu_manager, kvcache_manager)
    decision = await scheduler.schedule(sample_request)

    assert decision is not None
    assert decision.worker_id in [w.worker_id for w in gpu_manager.get_all_workers()]
    assert decision.algorithm_used == SchedulingAlgorithm.PRIORITY


@pytest.mark.asyncio
async def test_topology_scheduler_basic(gpu_manager, kvcache_manager, sample_request):
    """Topology scheduler places request on a valid worker."""
    from app.scheduler.topology_scheduler import TopologyAwareScheduler

    scheduler = TopologyAwareScheduler(gpu_manager, kvcache_manager)
    decision = await scheduler.schedule(sample_request)

    assert decision is not None
    assert decision.algorithm_used == SchedulingAlgorithm.TOPOLOGY_AWARE


@pytest.mark.asyncio
async def test_gang_scheduler_single_gpu(gpu_manager, kvcache_manager, sample_request):
    """Gang scheduler handles single-GPU requests."""
    from app.scheduler.gang_scheduler import GangScheduler

    scheduler = GangScheduler(gpu_manager, kvcache_manager)
    decision = await scheduler.schedule(sample_request)

    assert decision is not None


@pytest.mark.asyncio
async def test_scheduler_composite_score(gpu_manager, kvcache_manager, sample_request):
    """Composite score returns valid range [0, 1]."""
    from app.scheduler.priority_scheduler import PriorityScheduler

    scheduler = PriorityScheduler(gpu_manager, kvcache_manager)
    workers = gpu_manager.get_healthy_workers()
    assert len(workers) > 0

    for worker in workers:
        score = scheduler._composite_score(worker, sample_request)
        assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_starvation_escalation(gpu_manager, kvcache_manager):
    """Low priority requests get escalated after starvation timeout."""
    from app.scheduler.priority_scheduler import PriorityScheduler
    from app.models import InferenceRequest, RequestPriority
    import time

    scheduler = PriorityScheduler(gpu_manager, kvcache_manager)
    # Force starvation timeout to 0
    scheduler.config.scheduler.starvation_timeout_ms = 0

    req = InferenceRequest(
        prompt="test",
        prompt_tokens=5,
        max_new_tokens=32,
        priority=RequestPriority.LOW,
    )
    scheduler._first_seen[req.request_id] = time.monotonic() - 999  # Old request

    escalated = await scheduler.check_and_escalate_starvation([req])
    assert len(escalated) == 1
    assert req.priority.value < RequestPriority.LOW.value  # Escalated


@pytest.mark.asyncio
async def test_can_schedule_returns_false_with_no_workers():
    """Can-schedule returns False when no healthy workers."""
    from app.resource.gpu_manager import GPUResourceManager
    from app.kvcache.kvcache_manager import KVCacheManager
    from app.scheduler.priority_scheduler import PriorityScheduler
    from app.models import InferenceRequest

    # Empty manager
    mgr = GPUResourceManager()
    mgr._workers = {}
    kv = KVCacheManager()

    scheduler = PriorityScheduler(mgr, kv)
    req = InferenceRequest(prompt="test", prompt_tokens=5, max_new_tokens=32)
    result = await scheduler.can_schedule(req)
    assert result is False
