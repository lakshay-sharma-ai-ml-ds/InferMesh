"""
InferMesh Test Configuration
==============================
pytest fixtures for unit and integration tests.
"""

from __future__ import annotations

import asyncio
import pytest

from app.config import InferMeshConfig, SchedulerConfig, KVCacheConfig, BatchingConfig, SimulationConfig


@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
def test_config() -> InferMeshConfig:
    """Minimal config for fast tests."""
    return InferMeshConfig(
        env="test",
        log_level="WARNING",
        simulation=SimulationConfig(enabled=True, num_gpus=2, failure_injection_rate=0.0),
        scheduler=SchedulerConfig(scheduling_interval_ms=1),
        kvcache=KVCacheConfig(max_blocks_per_worker=64),
        batching=BatchingConfig(max_batch_size=16),
    )


@pytest.fixture
async def gpu_manager(test_config):
    from app.resource.gpu_manager import GPUResourceManager
    import app.config as cfg_module
    cfg_module._config = test_config

    mgr = GPUResourceManager()
    await mgr.start()
    yield mgr
    await mgr.stop()


@pytest.fixture
async def kvcache_manager(gpu_manager):
    from app.kvcache.kvcache_manager import KVCacheManager
    mgr = KVCacheManager()
    for worker in gpu_manager.get_all_workers():
        await mgr.register_worker(worker.worker_id)
    yield mgr


@pytest.fixture
async def orchestrator(test_config):
    import app.config as cfg_module
    cfg_module._config = test_config
    from app.orchestrator import InferMeshOrchestrator
    orch = InferMeshOrchestrator(test_config)
    await orch.start()
    yield orch
    await orch.stop()


@pytest.fixture
def sample_request():
    from app.models import InferenceRequest, RequestPriority
    return InferenceRequest(
        prompt="Explain distributed systems.",
        prompt_tokens=10,
        max_new_tokens=64,
        priority=RequestPriority.NORMAL,
        tenant_id="test",
    )
