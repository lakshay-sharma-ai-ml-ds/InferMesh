"""
InferMesh Memory Manager
========================
Tracks fine-grained GPU memory allocations across model weights,
KV-cache, activations, and intermediate tensors.
Provides fragmentation analysis and defragmentation hints.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MemoryAllocation:
    """Tracks a single memory allocation."""
    alloc_id: str
    worker_id: str
    size_bytes: int
    category: str  # "weights" | "kvcache" | "activations" | "other"
    request_id: Optional[str] = None
    created_at: float = field(default_factory=lambda: __import__("time").monotonic())


class MemoryManager:
    """
    Fine-grained memory allocation tracker per GPU worker.

    Tracks:
    - Model weight bytes (static, pre-allocated)
    - KV-cache block bytes (dynamic, per-request)
    - Activation bytes (transient, per-forward-pass)

    Provides:
    - Fragmentation ratio estimation
    - Memory pressure signals for eviction triggers
    - Allocation history for debugging
    """

    def __init__(self, worker_id: str, total_bytes: int):
        self.worker_id = worker_id
        self.total_bytes = total_bytes
        self._allocations: dict[str, MemoryAllocation] = {}
        self._lock = asyncio.Lock()
        self._by_category: dict[str, int] = {
            "weights": 0,
            "kvcache": 0,
            "activations": 0,
            "other": 0,
        }

    # -----------------------------------------------------------------------
    # Allocation API
    # -----------------------------------------------------------------------

    async def allocate(
        self,
        alloc_id: str,
        size_bytes: int,
        category: str = "other",
        request_id: Optional[str] = None,
    ) -> bool:
        """
        Try to allocate `size_bytes`. Returns True if successful.
        Thread-safe via asyncio lock.
        """
        async with self._lock:
            if self.used_bytes + size_bytes > self.total_bytes:
                return False
            alloc = MemoryAllocation(
                alloc_id=alloc_id,
                worker_id=self.worker_id,
                size_bytes=size_bytes,
                category=category,
                request_id=request_id,
            )
            self._allocations[alloc_id] = alloc
            self._by_category[category] = self._by_category.get(category, 0) + size_bytes
            return True

    async def deallocate(self, alloc_id: str) -> int:
        """Free an allocation. Returns freed bytes."""
        async with self._lock:
            alloc = self._allocations.pop(alloc_id, None)
            if alloc:
                self._by_category[alloc.category] -= alloc.size_bytes
                return alloc.size_bytes
            return 0

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def used_bytes(self) -> int:
        return sum(a.size_bytes for a in self._allocations.values())

    @property
    def free_bytes(self) -> int:
        return self.total_bytes - self.used_bytes

    @property
    def utilization_pct(self) -> float:
        return (self.used_bytes / self.total_bytes * 100) if self.total_bytes > 0 else 0.0

    @property
    def fragmentation_score(self) -> float:
        """
        Estimate fragmentation as ratio of small allocations.
        0 = no fragmentation, 1 = highly fragmented.
        """
        if not self._allocations:
            return 0.0
        sizes = [a.size_bytes for a in self._allocations.values()]
        avg = sum(sizes) / len(sizes)
        max_size = max(sizes)
        return 1.0 - (avg / max_size) if max_size > 0 else 0.0

    def category_breakdown(self) -> dict[str, int]:
        """Returns bytes used per category."""
        return dict(self._by_category)

    def is_under_pressure(self, threshold: float = 0.85) -> bool:
        return self.utilization_pct / 100 > threshold

    def allocations_for_request(self, request_id: str) -> list[MemoryAllocation]:
        return [a for a in self._allocations.values() if a.request_id == request_id]

    async def free_request_allocations(self, request_id: str) -> int:
        """Free all memory associated with a request. Returns total freed."""
        alloc_ids = [
            aid for aid, a in self._allocations.items()
            if a.request_id == request_id
        ]
        freed = 0
        for aid in alloc_ids:
            freed += await self.deallocate(aid)
        return freed
