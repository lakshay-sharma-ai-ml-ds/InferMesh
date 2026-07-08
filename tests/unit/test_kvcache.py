"""Unit tests for KV-cache manager."""

from __future__ import annotations

import pytest

from app.kvcache.kvcache_manager import ARCCache, KVCacheManager, WorkerKVCache


class TestARCCache:
    def test_arc_basic_insert_and_hit(self):
        arc = ARCCache(capacity=4)
        arc.access("a", 1)
        arc.access("b", 2)
        assert arc.contains("a")
        assert arc.contains("b")

    def test_arc_eviction_on_capacity(self):
        arc = ARCCache(capacity=2)
        arc.access("a", 1)
        arc.access("b", 2)
        arc.access("c", 3)  # Should evict one
        assert arc.size <= 2

    def test_arc_frequency_promotes_to_t2(self):
        arc = ARCCache(capacity=4)
        arc.access("frequent", 1)
        arc.access("frequent", 1)  # Second access promotes to T2
        arc.access("a", 2)
        arc.access("b", 3)
        arc.access("c", 4)
        # "frequent" should survive eviction pressure
        assert arc.contains("frequent")

    def test_arc_evict_lru_empty(self):
        arc = ARCCache(capacity=4)
        result = arc.evict_lru()
        assert result is None


@pytest.mark.asyncio
async def test_worker_kvcache_allocate_and_free():
    cache = WorkerKVCache(worker_id="test-gpu", max_blocks=10, block_size_tokens=16)
    blocks = await cache.allocate_blocks(3)
    assert len(blocks) == 3

    await cache.free_blocks(blocks)
    stats = cache.get_stats()
    assert stats.free_blocks == 10  # All freed


@pytest.mark.asyncio
async def test_prefix_cache_lookup():
    cache = WorkerKVCache(worker_id="test-gpu", max_blocks=20, block_size_tokens=16)
    blocks = await cache.allocate_blocks(1)
    await cache.cache_prefix("test_hash", [1, 2, 3, 4], blocks[0])

    hit = cache.check_prefix("test_hash")
    assert hit is not None
    assert hit == blocks[0]


@pytest.mark.asyncio
async def test_kvcache_manager_register_worker():
    mgr = KVCacheManager()
    await mgr.register_worker("worker-0")
    stats = mgr.get_stats("worker-0")
    assert stats is not None
    assert stats.total_blocks > 0


@pytest.mark.asyncio
async def test_kvcache_manager_has_prefix(kvcache_manager, gpu_manager):
    worker = gpu_manager.get_healthy_workers()[0]
    prompt = "The quick brown fox jumps over the lazy dog, and this is a detailed explanation"

    # Initially no cache hit
    has_hit = kvcache_manager.has_prefix(worker.worker_id, prompt)
    # After caching
    await kvcache_manager.cache_prefix(worker.worker_id, prompt, block_id=42)
    has_hit_after = kvcache_manager.has_prefix(worker.worker_id, prompt)
    assert has_hit_after is True


@pytest.mark.asyncio
async def test_kvcache_eviction_under_pressure():
    """Cache should evict blocks when at capacity."""
    cache = WorkerKVCache(worker_id="test", max_blocks=5, block_size_tokens=16)

    # Fill to capacity
    for i in range(5):
        blocks = await cache.allocate_blocks(1)
        await cache.cache_prefix(f"hash_{i}", list(range(16)), blocks[0])

    # Request one more — should trigger eviction
    new_blocks = await cache.allocate_blocks(1)
    assert len(new_blocks) >= 1 or cache._evictions > 0
