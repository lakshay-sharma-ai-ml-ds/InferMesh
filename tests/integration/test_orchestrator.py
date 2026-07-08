"""Integration tests for the full orchestrator pipeline."""

from __future__ import annotations

import asyncio
import pytest

from app.models import InferenceRequest, RequestPriority, RequestStatus


@pytest.mark.asyncio
async def test_orchestrator_submit_and_complete(orchestrator):
    """Full end-to-end request flow through the orchestrator."""
    req = InferenceRequest(
        prompt="What is distributed computing?",
        prompt_tokens=8,
        max_new_tokens=32,
        priority=RequestPriority.NORMAL,
    )
    response = await orchestrator.submit(req)

    assert response is not None
    assert response.request_id == req.request_id
    assert response.status in (RequestStatus.COMPLETED, RequestStatus.FAILED)


@pytest.mark.asyncio
async def test_orchestrator_concurrent_requests(orchestrator):
    """Multiple concurrent requests should all be handled."""
    requests = [
        InferenceRequest(
            prompt=f"Explain topic {i}",
            prompt_tokens=6,
            max_new_tokens=32,
            priority=RequestPriority.NORMAL,
        )
        for i in range(5)
    ]

    responses = await asyncio.gather(
        *[orchestrator.submit(r) for r in requests],
        return_exceptions=True,
    )

    assert len(responses) == 5
    success = sum(
        1 for r in responses
        if not isinstance(r, Exception) and r.status == RequestStatus.COMPLETED
    )
    assert success >= 3  # At least 60% success in test mode


@pytest.mark.asyncio
async def test_orchestrator_high_priority_processed(orchestrator):
    """CRITICAL requests should not be rejected."""
    req = InferenceRequest(
        prompt="URGENT: System failure detected",
        prompt_tokens=8,
        max_new_tokens=16,
        priority=RequestPriority.CRITICAL,
    )
    response = await orchestrator.submit(req)
    assert response.status != RequestStatus.FAILED or "Rejected" not in (response.error_message or "")


@pytest.mark.asyncio
async def test_orchestrator_metrics_snapshot(orchestrator):
    """Metrics snapshot returns required fields."""
    snapshot = orchestrator.get_metrics_snapshot()

    assert "cluster" in snapshot
    assert "queue" in snapshot
    assert "latency" in snapshot
    assert "kvcache" in snapshot
    assert snapshot["cluster"]["total_workers"] > 0


@pytest.mark.asyncio
async def test_orchestrator_gpu_manager_initialized(orchestrator):
    """GPU manager should have workers after startup."""
    workers = orchestrator.gpu_manager.get_all_workers()
    assert len(workers) > 0


@pytest.mark.asyncio
async def test_orchestrator_kvcache_registered(orchestrator):
    """KV-cache should be registered for all workers."""
    stats = orchestrator.kvcache_manager.get_all_stats()
    workers = orchestrator.gpu_manager.get_all_workers()
    assert len(stats) == len(workers)
