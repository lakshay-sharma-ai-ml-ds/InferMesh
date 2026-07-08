"""
InferMesh Priority Queue System
=================================
Multi-level weighted fair queue (WFQ) for inference requests.

Design:
- 5 priority levels: CRITICAL(0) → BACKGROUND(4)
- Per-priority virtual time tracking for WFQ fairness
- Per-tenant quota enforcement
- O(log n) insertion and extraction using sortedcontainers
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from sortedcontainers import SortedList

import structlog

from app.config import get_config
from app.models import InferenceRequest, RequestPriority, RequestStatus
from app.metrics.registry import REQUEST_QUEUE_DEPTH, REQUEST_QUEUE_WAIT_SECONDS

logger = structlog.get_logger(__name__)


@dataclass(order=True)
class QueueEntry:
    """
    Priority queue entry. Sorted by (virtual_finish_time, priority, arrival_time).
    Lower virtual_finish_time = higher scheduling priority.
    """
    virtual_finish_time: float
    priority: int         # lower = more urgent
    arrived_at: float
    request: InferenceRequest = field(compare=False)

    @classmethod
    def make(cls, request: InferenceRequest, weight: float) -> "QueueEntry":
        """Create entry with WFQ virtual finish time."""
        now = time.monotonic()
        size = request.prompt_tokens + request.max_new_tokens
        vft = now + size / weight  # WFQ: bigger requests have higher VFT
        return cls(
            virtual_finish_time=vft,
            priority=request.priority.value,
            arrived_at=now,
            request=request,
        )


class MultiLevelPriorityQueue:
    """
    Weighted Fair Queue with:
    - Per-priority SortedList for O(log n) ops
    - WFQ virtual time for fairness within a priority level
    - Per-tenant quotas
    - Async-safe access via asyncio.Lock
    """

    def __init__(self):
        self.config = get_config()
        self._queues: dict[int, SortedList] = {
            p.value: SortedList() for p in RequestPriority
        }
        self._tenant_counts: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Event()
        self._total_enqueued = 0
        self._total_dequeued = 0

    # -----------------------------------------------------------------------
    # Enqueue
    # -----------------------------------------------------------------------

    async def enqueue(self, request: InferenceRequest) -> bool:
        """
        Add request to queue. Returns False if rejected (quota exceeded).
        """
        async with self._lock:
            # Check total queue depth
            total = sum(len(q) for q in self._queues.values())
            if total >= self.config.queue.max_queue_depth:
                logger.warning(
                    "Queue full, rejecting request",
                    request_id=request.request_id,
                    total_depth=total,
                )
                return False

            # Per-tenant quota
            tenant_count = self._tenant_counts.get(request.tenant_id, 0)
            if tenant_count >= self.config.queue.per_tenant_quota:
                logger.warning(
                    "Tenant quota exceeded",
                    tenant_id=request.tenant_id,
                    count=tenant_count,
                )
                return False

            # WFQ weight for this priority
            weights = self.config.queue.priority_weights
            weight = weights.get(request.priority.name, 1.0)

            entry = QueueEntry.make(request, weight)
            self._queues[request.priority.value].add(entry)
            self._tenant_counts[request.tenant_id] = tenant_count + 1
            self._total_enqueued += 1
            request.status = RequestStatus.QUEUED

            # Update metrics
            REQUEST_QUEUE_DEPTH.labels(priority=request.priority.name).set(
                len(self._queues[request.priority.value])
            )

            self._not_empty.set()
            return True

    # -----------------------------------------------------------------------
    # Dequeue
    # -----------------------------------------------------------------------

    async def dequeue(
        self, timeout_s: float = 0.01
    ) -> Optional[InferenceRequest]:
        """
        Extract the highest-priority request (lowest virtual finish time).
        Processes CRITICAL first, then HIGH, etc.
        """
        async with self._lock:
            for priority_val in sorted(self._queues.keys()):
                q = self._queues[priority_val]
                if q:
                    entry = q.pop(0)  # Smallest vft = first scheduled
                    request = entry.request

                    # Update tenant count
                    tid = request.tenant_id
                    self._tenant_counts[tid] = max(0, self._tenant_counts.get(tid, 1) - 1)
                    self._total_dequeued += 1

                    # Observe wait time
                    wait_s = time.monotonic() - entry.arrived_at
                    REQUEST_QUEUE_WAIT_SECONDS.labels(
                        priority=request.priority.name,
                        tenant_id=request.tenant_id,
                    ).observe(wait_s)

                    # Update depth metric
                    REQUEST_QUEUE_DEPTH.labels(
                        priority=request.priority.name
                    ).set(len(q))

                    # Clear event if all queues empty
                    if all(len(q) == 0 for q in self._queues.values()):
                        self._not_empty.clear()

                    return request
            return None

    async def dequeue_batch(
        self, max_size: int = 32
    ) -> list[InferenceRequest]:
        """Dequeue up to `max_size` requests respecting priority order."""
        results = []
        for _ in range(max_size):
            req = await self.dequeue()
            if req is None:
                break
            results.append(req)
        return results

    # -----------------------------------------------------------------------
    # Inspection
    # -----------------------------------------------------------------------

    async def wait_for_items(self, timeout_s: float = 1.0) -> bool:
        """Wait until at least one item is in the queue."""
        try:
            await asyncio.wait_for(self._not_empty.wait(), timeout=timeout_s)
            return True
        except asyncio.TimeoutError:
            return False

    def depth(self) -> dict[str, int]:
        return {
            RequestPriority(pv).name: len(q)
            for pv, q in self._queues.items()
        }

    def total_depth(self) -> int:
        return sum(len(q) for q in self._queues.values())

    def peek_priority(self) -> Optional[RequestPriority]:
        """Return the priority of the next request to be dequeued."""
        for pv in sorted(self._queues.keys()):
            if self._queues[pv]:
                return RequestPriority(pv)
        return None

    def get_all_queued(self) -> list[InferenceRequest]:
        """Return all queued requests (for inspection / starvation check)."""
        result = []
        for pv in sorted(self._queues.keys()):
            for entry in self._queues[pv]:
                result.append(entry.request)
        return result

    def stats(self) -> dict:
        return {
            "depth": self.depth(),
            "total_depth": self.total_depth(),
            "total_enqueued": self._total_enqueued,
            "total_dequeued": self._total_dequeued,
            "tenant_counts": dict(self._tenant_counts),
        }
