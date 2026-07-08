"""
InferMesh Configuration System
==============================
Hierarchical Pydantic v2 settings with environment variable overrides,
YAML file loading, and per-component configuration blocks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models import (
    BatchingStrategy,
    EvictionPolicy,
    SchedulingAlgorithm,
)


class SchedulerConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERMESH_SCHEDULER_")

    algorithm: SchedulingAlgorithm = SchedulingAlgorithm.TOPOLOGY_AWARE
    preemption_enabled: bool = True
    gang_scheduling_enabled: bool = True
    max_preemptions_per_window: int = 10
    scheduling_interval_ms: int = 5       # How often to run scheduling loop
    starvation_timeout_ms: int = 30_000   # Promote starved LOW -> HIGH
    locality_weight: float = 0.4           # Weight for topology affinity
    load_weight: float = 0.6              # Weight for load balancing


class BatchingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERMESH_BATCHING_")

    strategy: BatchingStrategy = BatchingStrategy.CONTINUOUS
    max_batch_size: int = 128
    max_batch_tokens: int = 8192          # Max total tokens per forward pass
    prefill_chunk_size: int = 512         # Tokens per prefill chunk
    decode_token_budget: int = 4096       # Max decode tokens per step
    batch_wait_timeout_ms: float = 5.0    # Max wait to form a batch
    enable_sequence_packing: bool = True
    packing_efficiency_threshold: float = 0.8


class KVCacheConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERMESH_KVCACHE_")

    block_size_tokens: int = 16           # Tokens per KV block
    max_blocks_per_worker: int = 2048
    eviction_policy: EvictionPolicy = EvictionPolicy.ARC
    enable_prefix_caching: bool = True
    prefix_cache_max_entries: int = 10_000
    prefix_min_length_tokens: int = 8    # Minimum prefix to cache
    eviction_watermark_high: float = 0.90  # Start evicting at 90%
    eviction_watermark_low: float = 0.70   # Stop evicting at 70%
    cross_worker_migration: bool = True


class HealthConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERMESH_HEALTH_")

    heartbeat_interval_ms: int = 5_000
    heartbeat_timeout_ms: int = 15_000
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_success_threshold: int = 2
    circuit_breaker_open_duration_ms: int = 30_000
    probe_timeout_ms: int = 3_000
    max_consecutive_failures: int = 5


class AutoscalerConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERMESH_AUTOSCALER_")

    enabled: bool = True
    min_workers: int = 1
    max_workers: int = 32
    scale_up_threshold: float = 0.75      # GPU util or queue depth
    scale_down_threshold: float = 0.25
    scale_up_cooldown_s: int = 30
    scale_down_cooldown_s: int = 120
    scale_up_step: int = 2
    scale_down_step: int = 1
    predictive_enabled: bool = True
    ewma_alpha: float = 0.3               # EWMA smoothing factor
    forecast_horizon_s: int = 60          # How far ahead to predict


class QueueConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERMESH_QUEUE_")

    max_queue_depth: int = 10_000
    per_tenant_quota: int = 1_000
    admission_control_enabled: bool = True
    weighted_fair_queuing: bool = True
    # Priority weights for WFQ
    priority_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "CRITICAL": 8.0,
            "HIGH": 4.0,
            "NORMAL": 2.0,
            "LOW": 1.0,
            "BACKGROUND": 0.5,
        }
    )
    routing_consistent_hash: bool = True


class SimulationConfig(BaseSettings):
    """GPU simulation parameters (used when no real GPUs are present)."""
    model_config = SettingsConfigDict(env_prefix="INFERMESH_SIM_")

    enabled: bool = True
    num_gpus: int = Field(default=4, alias="INFERMESH_NUM_SIMULATED_GPUS")
    gpu_types: list[str] = Field(
        default_factory=lambda: ["A100_SXM4_80G", "A100_PCIe_40G", "A10G", "T4"]
    )
    # Latency simulation constants
    # prefill: 0.05ms/token (batch prefill is fast)
    # decode:  0.8ms/token  → 256 tokens = ~205ms total (well within 5s SLA)
    # At 20ms/token (old), 256 tokens = 5120ms → always exceeded timeout
    base_prefill_ms_per_token: float = 0.05   # ms per prompt token
    base_decode_ms_per_token: float = 0.8     # ms per generated token
    latency_noise_std: float = 0.10           # Gaussian noise std
    failure_injection_rate: float = 0.005     # 0.5% random GPU failures
    compute_utilization_model: str = "realistic"  # "realistic" | "random"

    model_config = SettingsConfigDict(
        env_prefix="INFERMESH_SIM_",
        populate_by_name=True,
    )


class RedisConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERMESH_REDIS_")

    url: str = Field(default="redis://localhost:6379/0", alias="INFERMESH_REDIS_URL")
    max_connections: int = 50
    socket_timeout: float = 5.0
    enabled: bool = True

    model_config = SettingsConfigDict(
        env_prefix="INFERMESH_REDIS_",
        populate_by_name=True,
    )


class InferMeshConfig(BaseSettings):
    """Root configuration for the entire InferMesh system."""
    model_config = SettingsConfigDict(
        env_prefix="INFERMESH_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
    )

    # Meta
    env: str = "development"
    log_level: str = "INFO"
    version: str = "1.0.0"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    grpc_port: int = 50051
    workers: int = 1

    # Sub-configs (loaded from env with prefix or defaults)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    batching: BatchingConfig = Field(default_factory=BatchingConfig)
    kvcache: KVCacheConfig = Field(default_factory=KVCacheConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    autoscaler: AutoscalerConfig = Field(default_factory=AutoscalerConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)

    # Paths
    results_dir: Path = Path("results")
    log_dir: Path = Path("logs")
    config_file: Optional[Path] = None

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return v.upper()

    @classmethod
    def from_yaml(cls, path: Path) -> "InferMeshConfig":
        """Load config from YAML file, overridden by env vars."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


# Global singleton
_config: Optional[InferMeshConfig] = None


def get_config() -> InferMeshConfig:
    """Get or create the global config singleton."""
    global _config
    if _config is None:
        config_file = os.environ.get("INFERMESH_CONFIG_FILE")
        if config_file and Path(config_file).exists():
            _config = InferMeshConfig.from_yaml(Path(config_file))
        else:
            _config = InferMeshConfig()
        _config.ensure_directories()
    return _config
