"""Unit tests for priority queue and admission control."""

from __future__ import annotations

import asyncio
import pytest

from app.models import InferenceRequest, RequestPriority
from app.queue.priority_queue import MultiLevelPriorityQueue


@pytest.mark.asyncio
async def test_queue_enqueue_dequeue():
    q = MultiLevelPriorityQueue()
    req = InferenceRequest(prompt="test", prompt_tokens=5, max_new_tokens=32)
    queued = await q.enqueue(req)
    assert queued is True

    dequeued = await q.dequeue()
    assert dequeued is not None
    assert dequeued.request_id == req.request_id


@pytest.mark.asyncio
async def test_queue_priority_order():
    """CRITICAL requests should be dequeued before NORMAL."""
    q = MultiLevelPriorityQueue()

    normal_req = InferenceRequest(
        prompt="normal", prompt_tokens=5, max_new_tokens=32,
        priority=RequestPriority.NORMAL
    )
    critical_req = InferenceRequest(
        prompt="critical", prompt_tokens=5, max_new_tokens=32,
        priority=RequestPriority.CRITICAL
    )

    await q.enqueue(normal_req)
    await q.enqueue(critical_req)

    first = await q.dequeue()
    assert first.priority == RequestPriority.CRITICAL


@pytest.mark.asyncio
async def test_queue_depth():
    q = MultiLevelPriorityQueue()
    for _ in range(5):
        req = InferenceRequest(prompt="test", prompt_tokens=5, max_new_tokens=32)
        await q.enqueue(req)

    assert q.total_depth() == 5


@pytest.mark.asyncio
async def test_queue_tenant_quota():
    q = MultiLevelPriorityQueue()
    q.config.queue.per_tenant_quota = 3

    for _ in range(3):
        req = InferenceRequest(
            prompt="test", prompt_tokens=5, max_new_tokens=32, tenant_id="t1"
        )
        await q.enqueue(req)

    # 4th request should be rejected
    req4 = InferenceRequest(
        prompt="test", prompt_tokens=5, max_new_tokens=32, tenant_id="t1"
    )
    result = await q.enqueue(req4)
    assert result is False


@pytest.mark.asyncio
async def test_queue_dequeue_batch():
    q = MultiLevelPriorityQueue()
    for i in range(10):
        req = InferenceRequest(prompt=f"test{i}", prompt_tokens=5, max_new_tokens=32)
        await q.enqueue(req)

    batch = await q.dequeue_batch(max_size=4)
    assert len(batch) == 4
    assert q.total_depth() == 6


@pytest.mark.asyncio
async def test_admission_controller(gpu_manager):
    from app.queue.admission import AdmissionController
    q = MultiLevelPriorityQueue()
    controller = AdmissionController(gpu_manager, q)

    req = InferenceRequest(
        prompt="test", prompt_tokens=5, max_new_tokens=32,
        priority=RequestPriority.NORMAL,
        tenant_id="test_tenant",
    )
    admitted, reason = await controller.admit(req)
    assert admitted is True
    assert reason == "admitted"


@pytest.mark.asyncio
async def test_admission_always_admits_critical(gpu_manager):
    from app.queue.admission import AdmissionController
    q = MultiLevelPriorityQueue()
    controller = AdmissionController(gpu_manager, q)

    req = InferenceRequest(
        prompt="critical", prompt_tokens=5, max_new_tokens=32,
        priority=RequestPriority.CRITICAL,
    )
    admitted, reason = await controller.admit(req)
    assert admitted is True
    assert reason == "critical_always_admitted"
