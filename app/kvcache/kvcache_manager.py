"""
InferMesh KV-Cache Manager
============================
Radix-tree-based KV-cache with prefix sharing and adaptive eviction.

Architecture:
- Each worker maintains its own KV-cache block pool
- Prefix cache uses a hash-based trie to share cached prefix blocks
- ARC eviction policy: balances recency (LRU) and frequency (LFU)
- Cross-worker cache migration for load balancing (future work)

Key concepts:
- Block: Fixed-size unit (default 16 tokens) of KV-cache storage
- Prefix hash: SHA-256 of token sequence for cache lookup
- Ref count: Number of active requests using a block
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections import OrderedDict
from typing import Optional

import structlog

from app.config import get_config
from app.models import EvictionPolicy, KVCacheBlock, KVCacheStats
from app.metrics.registry import KVCACHE_EVICTIONS_TOTAL

logger = structlog.get_logger(__name__)


def _prefix_hash(prompt: str, length: int) -> str:
    """Deterministic hash for prefix cache lookups."""
    key = f"{prompt[:length]}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# ARC (Adaptive Replacement Cache) implementation
# ---------------------------------------------------------------------------

class ARCCache:
    """
    Adaptive Replacement Cache (ARC) for KV-cache block eviction.

    ARC maintains four lists:
    - T1: Recently used once (ghost: B1)
    - T2: Frequently used (ghost: B2)
    
    ARC dynamically adjusts the partition between T1 and T2 to
    adapt to access patterns — superior to pure LRU or LFU.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._p = 0  # Target size for T1
        self._t1: OrderedDict[str, int] = OrderedDict()  # Recent singles
        self._t2: OrderedDict[str, int] = OrderedDict()  # Recent multiples
        self._b1: OrderedDict[str, None] = OrderedDict()  # Ghost: evicted from T1
        self._b2: OrderedDict[str, None] = OrderedDict()  # Ghost: evicted from T2
        self.evictions = 0

    def access(self, key: str, value: int) -> None:
        """Record a cache access / insert."""
        if key in self._t1:
            self._t1.move_to_end(key)
            self._t2[key] = self._t1.pop(key)
        elif key in self._t2:
            self._t2.move_to_end(key)
        elif key in self._b1:
            # Adapt: favor T2
            self._p = min(self.capacity, self._p + max(1, len(self._b2) // len(self._b1)))
            self._evict()
            self._b1.pop(key)
            self._t2[key] = value
        elif key in self._b2:
            # Adapt: favor T1
            self._p = max(0, self._p - max(1, len(self._b1) // len(self._b2)))
            self._evict()
            self._b2.pop(key)
            self._t2[key] = value
        else:
            # New entry
            self._evict()
            self._t1[key] = value

    def _evict(self) -> Optional[str]:
        total = len(self._t1) + len(self._t2)
        if total < self.capacity:
            return None
        if len(self._t1) > self._p:
            evicted_key, _ = self._t1.popitem(last=False)
            self._b1[evicted_key] = None
        elif self._t2:
            evicted_key, _ = self._t2.popitem(last=False)
            self._b2[evicted_key] = None
        else:
            return None
        self.evictions += 1
        return evicted_key

    def evict_lru(self) -> Optional[str]:
        """Forcibly evict the LRU item from T1 or T2."""
        if self._t1:
            k, _ = self._t1.popitem(last=False)
            self._b1[k] = None
            self.evictions += 1
            return k
        if self._t2:
            k, _ = self._t2.popitem(last=False)
            self._b2[k] = None
            self.evictions += 1
            return k
        return None

    @property
    def size(self) -> int:
        return len(self._t1) + len(self._t2)

    def contains(self, key: str) -> bool:
        return key in self._t1 or key in self._t2


# ---------------------------------------------------------------------------
# Per-Worker KV-Cache
# ---------------------------------------------------------------------------

class WorkerKVCache:
    """KV-cache pool for a single GPU worker."""

    def __init__(self, worker_id: str, max_blocks: int, block_size_tokens: int):
        self.worker_id = worker_id
        self.max_blocks = max_blocks
        self.block_size_tokens = block_size_tokens

        self._blocks: dict[int, KVCacheBlock] = {}
        self._free_block_ids: list[int] = list(range(max_blocks))
        self._prefix_index: dict[str, int] = {}  # prefix_hash -> block_id
        self._arc = ARCCache(max_blocks)
        self._lock = asyncio.Lock()

        # Stats
        self._hits = 0
        self._misses = 0
        self._prefix_hits = 0
        self._evictions = 0
        self._bytes_saved = 0

    # -----------------------------------------------------------------------
    # Block allocation
    # -----------------------------------------------------------------------

    async def allocate_blocks(self, num_blocks: int) -> list[int]:
        """Allocate `num_blocks` free blocks. Returns block IDs or []."""
        async with self._lock:
            if len(self._free_block_ids) < num_blocks:
                # Try eviction
                await self._evict_until(num_blocks - len(self._free_block_ids))
            if len(self._free_block_ids) < num_blocks:
                return []
            allocated = self._free_block_ids[:num_blocks]
            self._free_block_ids = self._free_block_ids[num_blocks:]
            return allocated

    async def free_blocks(self, block_ids: list[int]) -> None:
        """Return blocks to the free pool."""
        async with self._lock:
            for bid in block_ids:
                block = self._blocks.pop(bid, None)
                if block:
                    if block.prefix_hash:
                        self._prefix_index.pop(block.prefix_hash, None)
                    self._free_block_ids.append(bid)

    # -----------------------------------------------------------------------
    # Prefix cache
    # -----------------------------------------------------------------------

    def check_prefix(self, prefix_hash: str) -> Optional[int]:
        """Return block_id if prefix is cached, else None."""
        bid = self._prefix_index.get(prefix_hash)
        if bid is not None and bid in self._blocks:
            self._arc.access(prefix_hash, bid)
            self._hits += 1
            self._prefix_hits += 1
            block = self._blocks[bid]
            block.ref_count += 1
            block.last_accessed = __import__("datetime").datetime.utcnow()
            return bid
        self._misses += 1
        return None

    async def cache_prefix(
        self, prefix_hash: str, token_ids: list[int], block_id: int
    ) -> None:
        """Register a computed prefix in the cache."""
        async with self._lock:
            block = KVCacheBlock(
                block_id=block_id,
                worker_id=self.worker_id,
                size_bytes=len(token_ids) * 4 * 32 * 2,  # simplified
                token_ids=token_ids,
                prefix_hash=prefix_hash,
                ref_count=1,
                is_shared=True,
            )
            self._blocks[block_id] = block
            self._prefix_index[prefix_hash] = block_id
            self._arc.access(prefix_hash, block_id)

    # -----------------------------------------------------------------------
    # Eviction
    # -----------------------------------------------------------------------

    async def _evict_until(self, needed: int) -> None:
        """Evict blocks until `needed` free blocks are available."""
        config = get_config()
        for _ in range(needed * 2):
            if len(self._free_block_ids) >= needed:
                break
            evicted_key = self._arc.evict_lru()
            if evicted_key is None:
                break
            bid = self._prefix_index.pop(evicted_key, None)
            if bid and bid in self._blocks:
                block = self._blocks.pop(bid)
                if block.ref_count <= 0:
                    self._free_block_ids.append(bid)
                    self._evictions += 1
                    KVCACHE_EVICTIONS_TOTAL.labels(
                        worker_id=self.worker_id,
                        eviction_policy=config.kvcache.eviction_policy.value,
                    ).inc()

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------

    def get_stats(self) -> KVCacheStats:
        used = self.max_blocks - len(self._free_block_ids)
        total_accesses = self._hits + self._misses
        return KVCacheStats(
            total_blocks=self.max_blocks,
            used_blocks=used,
            free_blocks=len(self._free_block_ids),
            hit_rate=self._hits / max(1, total_accesses),
            prefix_hit_rate=self._prefix_hits / max(1, total_accesses),
            evictions_total=self._evictions,
            bytes_saved_by_prefix_sharing=self._bytes_saved,
        )


# ---------------------------------------------------------------------------
# Global KV-Cache Manager
# ---------------------------------------------------------------------------

class KVCacheManager:
    """
    Cluster-wide KV-cache manager.
    Maintains per-worker caches and provides unified lookup/allocation API.
    """

    def __init__(self):
        self.config = get_config()
        self._caches: dict[str, WorkerKVCache] = {}
        self._lock = asyncio.Lock()

    async def register_worker(self, worker_id: str) -> None:
        """Initialize KV-cache for a new worker."""
        cfg = self.config.kvcache
        cache = WorkerKVCache(
            worker_id=worker_id,
            max_blocks=cfg.max_blocks_per_worker,
            block_size_tokens=cfg.block_size_tokens,
        )
        async with self._lock:
            self._caches[worker_id] = cache
        logger.info(
            "KV-cache registered",
            worker_id=worker_id,
            max_blocks=cfg.max_blocks_per_worker,
        )

    async def unregister_worker(self, worker_id: str) -> None:
        async with self._lock:
            self._caches.pop(worker_id, None)

    def has_prefix(self, worker_id: str, prompt: str) -> bool:
        """Quick check: does this worker have a cached prefix?"""
        cache = self._caches.get(worker_id)
        if not cache:
            return False
        cfg = self.config.kvcache
        if len(prompt.split()) < cfg.prefix_min_length_tokens:
            return False
        prefix_hash = _prefix_hash(prompt, cfg.prefix_min_length_tokens * 4)
        return prefix_hash in cache._prefix_index

    async def allocate_blocks(self, worker_id: str, num_blocks: int) -> list[int]:
        cache = self._caches.get(worker_id)
        if not cache:
            return []
        return await cache.allocate_blocks(num_blocks)

    async def free_blocks(self, worker_id: str, block_ids: list[int]) -> None:
        cache = self._caches.get(worker_id)
        if cache:
            await cache.free_blocks(block_ids)

    async def cache_prefix(
        self, worker_id: str, prompt: str, block_id: int
    ) -> None:
        cache = self._caches.get(worker_id)
        if not cache:
            return
        cfg = self.config.kvcache
        prefix_hash = _prefix_hash(prompt, cfg.prefix_min_length_tokens * 4)
        token_ids = list(range(min(32, len(prompt))))  # Simplified token IDs
        await cache.cache_prefix(prefix_hash, token_ids, block_id)

    def get_stats(self, worker_id: str) -> Optional[KVCacheStats]:
        cache = self._caches.get(worker_id)
        return cache.get_stats() if cache else None

    def get_all_stats(self) -> dict[str, KVCacheStats]:
        return {
            wid: c.get_stats()
            for wid, c in self._caches.items()
        }

    def cluster_hit_rate(self) -> float:
        stats = self.get_all_stats()
        if not stats:
            return 0.0
        return sum(s.hit_rate for s in stats.values()) / len(stats)
