"""
InferMesh API Routes
=====================
REST API endpoints for inference submission, cluster status,
scheduler control, and benchmark triggers.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.models import (
    InferenceRequest,
    InferenceResponse,
    RequestPriority,
    SLAConfig,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request/Response schemas (API-facing, simplified)
# ---------------------------------------------------------------------------

class InferenceRequestBody(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32768)
    max_new_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    priority: str = Field(default="NORMAL")
    tenant_id: str = Field(default="default")
    model_id: str = Field(default="default")
    stream: bool = False
    max_ttft_ms: float = Field(default=500.0)
    max_e2e_latency_ms: float = Field(default=5000.0)


def get_orchestrator(request: Request):
    return request.app.state.orchestrator


# ---------------------------------------------------------------------------
# Inference endpoints
# ---------------------------------------------------------------------------

@router.post("/v1/infer", response_model=InferenceResponse, tags=["inference"])
async def infer(body: InferenceRequestBody, orch=Depends(get_orchestrator)):
    """Submit a synchronous inference request."""

    try:
        priority = RequestPriority[body.priority.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid priority: {body.priority}")

    # Estimate prompt tokens (simplified: 1 token ≈ 4 chars)
    prompt_tokens = max(1, len(body.prompt) // 4)

    inference_req = InferenceRequest(
        prompt=body.prompt,
        prompt_tokens=prompt_tokens,
        max_new_tokens=body.max_new_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        priority=priority,
        tenant_id=body.tenant_id,
        model_id=body.model_id,
        stream=body.stream,
        sla=SLAConfig(
            max_ttft_ms=body.max_ttft_ms,
            max_e2e_latency_ms=body.max_e2e_latency_ms,
        ),
    )

    response = await orch.submit(inference_req)
    return response


@router.post("/v1/infer/batch", tags=["inference"])
async def infer_batch(bodies: list[InferenceRequestBody], orch=Depends(get_orchestrator)):
    """Submit multiple inference requests concurrently."""

    async def process(body: InferenceRequestBody):
        try:
            priority = RequestPriority[body.priority.upper()]
        except KeyError:
            priority = RequestPriority.NORMAL

        prompt_tokens = max(1, len(body.prompt) // 4)
        req = InferenceRequest(
            prompt=body.prompt,
            prompt_tokens=prompt_tokens,
            max_new_tokens=body.max_new_tokens,
            priority=priority,
            tenant_id=body.tenant_id,
            model_id=body.model_id,
        )
        return await orch.submit(req)

    results = await asyncio.gather(*[process(b) for b in bodies], return_exceptions=True)
    return [r if isinstance(r, InferenceResponse) else {"error": str(r)} for r in results]


# ---------------------------------------------------------------------------
# Cluster & scheduler status
# ---------------------------------------------------------------------------

@router.get("/v1/cluster/status", tags=["cluster"])
async def cluster_status(orch=Depends(get_orchestrator)):
    """Get real-time cluster health and metrics."""
    return orch.get_metrics_snapshot()


@router.get("/v1/cluster/workers", tags=["cluster"])
async def list_workers(orch=Depends(get_orchestrator)):
    """List all GPU workers with their current state."""
    workers = orch.gpu_manager.get_all_workers()
    return [
        {
            "worker_id": w.worker_id,
            "gpu_type": w.gpu_type.value,
            "status": w.status.value,
            "circuit_breaker": w.circuit_breaker_state.value,
            "memory": {
                "total_gb": round(w.memory.total_bytes / 1024**3, 1),
                "allocated_gb": round(w.memory.allocated_bytes / 1024**3, 1),
                "utilization_pct": round(w.memory.utilization_pct, 2),
            },
            "compute_utilization_pct": round(w.compute_utilization_pct, 2),
            "active_requests": w.active_requests,
            "total_served": w.total_requests_served,
            "numa_node": w.numa_node,
            "nvlink_group": w.nvlink_group,
        }
        for w in workers
    ]


@router.get("/v1/cluster/queue", tags=["cluster"])
async def queue_status(orch=Depends(get_orchestrator)):
    """Get priority queue depth and stats."""
    return orch.queue.stats()


@router.get("/v1/cluster/kvcache", tags=["cluster"])
async def kvcache_status(orch=Depends(get_orchestrator)):
    """Get KV-cache stats per worker."""
    stats = orch.kvcache_manager.get_all_stats()
    return {
        wid: {
            "total_blocks": s.total_blocks,
            "used_blocks": s.used_blocks,
            "utilization_pct": round(s.utilization_pct, 2),
            "hit_rate": round(s.hit_rate, 4),
            "prefix_hit_rate": round(s.prefix_hit_rate, 4),
            "evictions": s.evictions_total,
        }
        for wid, s in stats.items()
    }


@router.get("/v1/cluster/autoscaler", tags=["autoscaler"])
async def autoscaler_status(orch=Depends(get_orchestrator)):
    """Get autoscaler state and forecast."""
    forecast = orch.predictive.get_forecast()
    workers = orch.gpu_manager.get_healthy_workers()
    return {
        "current_workers": len(workers),
        "min_workers": orch.config.autoscaler.min_workers,
        "max_workers": orch.config.autoscaler.max_workers,
        "ewma_util": round(orch.hpa._ewma_util * 100, 2),
        "forecast": forecast,
    }


# ---------------------------------------------------------------------------
# Scheduler control
# ---------------------------------------------------------------------------

@router.post("/v1/scheduler/algorithm", tags=["scheduler"])
async def set_scheduler(algorithm: str, orch=Depends(get_orchestrator)):
    """Switch scheduling algorithm at runtime."""
    from app.models import SchedulingAlgorithm
    try:
        algo = SchedulingAlgorithm(algorithm)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm: {algorithm}")

    orch.config.scheduler.algorithm = algo
    return {"algorithm": algo.value, "status": "updated"}
