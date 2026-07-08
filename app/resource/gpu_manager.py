"""
InferMesh GPU Resource Manager
================================
Manages a pool of GPU workers (simulated or real), handles allocation,
de-allocation, VRAM tracking, and topology-aware placement.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import defaultdict
from typing import Optional

import structlog

from app.config import get_config
from app.models import (
    GPU_SPECS_REGISTRY,
    GPUMemoryState,
    GPUStatus,
    GPUType,
    GPUWorker,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# GPU Topology Graph
# ---------------------------------------------------------------------------

class TopologyEdge:
    """Represents interconnect between two GPU workers."""
    def __init__(self, bandwidth_gbps: float, is_nvlink: bool):
        self.bandwidth_gbps = bandwidth_gbps
        self.is_nvlink = is_nvlink
        self.latency_us = 1.0 if is_nvlink else 5.0  # μs


class TopologyGraph:
    """
    Undirected graph of GPU interconnects for NUMA/NVLink-aware placement.
    Nodes = worker_id, Edges = interconnect bandwidth + type.
    """

    def __init__(self):
        self._adj: dict[str, dict[str, TopologyEdge]] = defaultdict(dict)

    def add_edge(self, w1: str, w2: str, bandwidth_gbps: float, is_nvlink: bool) -> None:
        edge = TopologyEdge(bandwidth_gbps, is_nvlink)
        self._adj[w1][w2] = edge
        self._adj[w2][w1] = edge

    def get_bandwidth(self, w1: str, w2: str) -> float:
        return self._adj.get(w1, {}).get(w2, TopologyEdge(0, False)).bandwidth_gbps

    def get_nvlink_group(self, worker_id: str) -> list[str]:
        """Return all NVLink-connected peers of a worker."""
        return [
            peer for peer, edge in self._adj.get(worker_id, {}).items()
            if edge.is_nvlink
        ]

    def affinity_score(self, candidates: list[str], reference: str) -> dict[str, float]:
        """
        Score each candidate by bandwidth affinity to reference.
        Returns normalized scores (0-1).
        """
        bws = {c: self.get_bandwidth(reference, c) for c in candidates}
        max_bw = max(bws.values()) if bws else 1.0
        return {c: bw / max_bw for c, bw in bws.items()}


# ---------------------------------------------------------------------------
# GPU Resource Manager
# ---------------------------------------------------------------------------

class GPUResourceManager:
    """
    Central registry of all GPU workers with:
    - Allocation / deallocation of VRAM
    - Real-time utilization tracking
    - Topology graph maintenance
    - Heterogeneous GPU support
    """

    def __init__(self):
        self.config = get_config()
        self._workers: dict[str, GPUWorker] = {}
        self.topology = TopologyGraph()
        self._allocation_lock = asyncio.Lock()
        self._simulation_task: asyncio.Task | None = None
        self._running = False

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize GPU pool (simulate or discover real GPUs)."""
        if self.config.simulation.enabled:
            await self._initialize_simulated_gpus()
        self._running = True
        self._simulation_task = asyncio.create_task(
            self._simulation_loop(), name="gpu-sim-loop"
        )
        logger.info(
            "GPU Resource Manager started",
            num_workers=len(self._workers),
            simulated=self.config.simulation.enabled,
        )

    async def stop(self) -> None:
        self._running = False
        if self._simulation_task:
            self._simulation_task.cancel()
            try:
                await self._simulation_task
            except asyncio.CancelledError:
                pass

    # -----------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------

    async def _initialize_simulated_gpus(self) -> None:
        sim_cfg = self.config.simulation
        gpu_types = sim_cfg.gpu_types
        num_gpus = sim_cfg.num_gpus

        created: list[GPUWorker] = []
        for i in range(num_gpus):
            gpu_type_str = gpu_types[i % len(gpu_types)]
            try:
                gpu_type = GPUType(gpu_type_str)
            except ValueError:
                gpu_type = GPUType.SIMULATED

            specs = GPU_SPECS_REGISTRY.get(gpu_type, GPU_SPECS_REGISTRY[GPUType.SIMULATED])
            vram_bytes = int(specs.vram_gb * 1024**3)

            # Assign NUMA node and PCIe domain
            numa_node = i // 4
            pcie_domain = i // 2
            nvlink_group = i // 4 if specs.nvlink_bandwidth_gbps > 0 else None

            worker = GPUWorker(
                worker_id=f"sim-gpu-{i}",
                gpu_type=gpu_type,
                specs=specs,
                memory=GPUMemoryState(
                    total_bytes=vram_bytes,
                    # Simulate model weights pre-loaded (20% of VRAM)
                    model_weights_bytes=int(vram_bytes * 0.20),
                    allocated_bytes=int(vram_bytes * 0.20),
                ),
                numa_node=numa_node,
                pcie_domain=pcie_domain,
                nvlink_group=nvlink_group,
                labels={
                    "gpu_type": gpu_type.value,
                    "numa_node": str(numa_node),
                    "simulated": "true",
                },
            )
            self._workers[worker.worker_id] = worker
            created.append(worker)

        # Build topology graph
        self._build_simulated_topology(created)
        logger.info("Simulated GPU pool initialized", workers=[w.worker_id for w in created])

    def _build_simulated_topology(self, workers: list[GPUWorker]) -> None:
        """Build a realistic PCIe + NVLink topology for simulated GPUs."""
        for i, w1 in enumerate(workers):
            for j, w2 in enumerate(workers):
                if i >= j:
                    continue
                same_nvlink_group = (
                    w1.nvlink_group is not None and
                    w1.nvlink_group == w2.nvlink_group
                )
                same_pcie = w1.pcie_domain == w2.pcie_domain

                if same_nvlink_group:
                    # NVLink: up to 600 GB/s
                    bw = min(
                        w1.specs.nvlink_bandwidth_gbps,
                        w2.specs.nvlink_bandwidth_gbps,
                    ) or 100.0
                    self.topology.add_edge(w1.worker_id, w2.worker_id, bw, is_nvlink=True)
                elif same_pcie:
                    bw = min(w1.specs.pcie_bandwidth_gbps, w2.specs.pcie_bandwidth_gbps)
                    self.topology.add_edge(w1.worker_id, w2.worker_id, bw, is_nvlink=False)
                else:
                    # Cross-domain PCIe: limited to switch bandwidth
                    bw = min(w1.specs.pcie_bandwidth_gbps, w2.specs.pcie_bandwidth_gbps) * 0.5
                    self.topology.add_edge(w1.worker_id, w2.worker_id, bw, is_nvlink=False)

    # -----------------------------------------------------------------------
    # Simulation Loop
    # -----------------------------------------------------------------------

    async def _simulation_loop(self) -> None:
        """Continuously simulate realistic GPU utilization changes."""
        while self._running:
            try:
                await self._update_simulated_metrics()
            except Exception as exc:
                logger.warning("Simulation loop error", error=str(exc))
            await asyncio.sleep(1.0)

    async def _update_simulated_metrics(self) -> None:
        """Update simulated compute utilization with realistic patterns."""
        sim_cfg = self.config.simulation
        for worker in self._workers.values():
            if worker.status == GPUStatus.OFFLINE:
                continue

            # Compute utilization correlates with active requests
            base_util = min(100.0, worker.active_requests * 12.0)
            noise = random.gauss(0, sim_cfg.latency_noise_std * 10)
            new_util = max(0.0, min(100.0, base_util + noise))
            worker.compute_utilization_pct = new_util

            # Random failure injection
            if (sim_cfg.failure_injection_rate > 0 and
                    random.random() < sim_cfg.failure_injection_rate and
                    worker.status == GPUStatus.HEALTHY):
                worker.status = GPUStatus.UNREACHABLE
                logger.warning("Simulated GPU failure injected", worker_id=worker.worker_id)

    # -----------------------------------------------------------------------
    # Worker Queries
    # -----------------------------------------------------------------------

    def get_all_workers(self) -> list[GPUWorker]:
        return list(self._workers.values())

    def get_healthy_workers(self) -> list[GPUWorker]:
        return [w for w in self._workers.values() if w.is_available]

    def get_worker(self, worker_id: str) -> Optional[GPUWorker]:
        return self._workers.get(worker_id)

    def get_workers_by_status(self, status: GPUStatus) -> list[GPUWorker]:
        return [w for w in self._workers.values() if w.status == status]

    # -----------------------------------------------------------------------
    # Allocation
    # -----------------------------------------------------------------------

    async def allocate_vram(self, worker_id: str, bytes_needed: int) -> bool:
        """
        Attempt to allocate `bytes_needed` bytes of VRAM on `worker_id`.
        Returns True on success, False if insufficient memory.
        """
        async with self._allocation_lock:
            worker = self._workers.get(worker_id)
            if not worker or not worker.is_available:
                return False
            if worker.memory.free_bytes < bytes_needed:
                return False
            worker.memory.allocated_bytes += bytes_needed
            return True

    async def deallocate_vram(self, worker_id: str, bytes_freed: int) -> None:
        """Release previously allocated VRAM."""
        async with self._allocation_lock:
            worker = self._workers.get(worker_id)
            if worker:
                worker.memory.allocated_bytes = max(
                    worker.memory.model_weights_bytes,
                    worker.memory.allocated_bytes - bytes_freed,
                )

    async def allocate_kvcache(self, worker_id: str, bytes_needed: int) -> bool:
        """Allocate KV-cache bytes (tracked separately)."""
        async with self._allocation_lock:
            worker = self._workers.get(worker_id)
            if not worker or not worker.is_available:
                return False
            potential = worker.memory.allocated_bytes + bytes_needed
            if potential > worker.memory.total_bytes:
                return False
            worker.memory.allocated_bytes += bytes_needed
            worker.memory.kvcache_bytes += bytes_needed
            return True

    async def deallocate_kvcache(self, worker_id: str, bytes_freed: int) -> None:
        async with self._allocation_lock:
            worker = self._workers.get(worker_id)
            if worker:
                freed = min(worker.memory.kvcache_bytes, bytes_freed)
                worker.memory.kvcache_bytes -= freed
                worker.memory.allocated_bytes -= freed

    # -----------------------------------------------------------------------
    # Request Tracking
    # -----------------------------------------------------------------------

    async def increment_active_requests(self, worker_id: str) -> None:
        worker = self._workers.get(worker_id)
        if worker:
            worker.active_requests += 1
            worker.total_requests_served += 1

    async def decrement_active_requests(self, worker_id: str) -> None:
        worker = self._workers.get(worker_id)
        if worker:
            worker.active_requests = max(0, worker.active_requests - 1)

    # -----------------------------------------------------------------------
    # Status Management
    # -----------------------------------------------------------------------

    async def mark_worker_status(self, worker_id: str, status: GPUStatus) -> None:
        worker = self._workers.get(worker_id)
        if worker:
            old = worker.status
            worker.status = status
            if old != status:
                logger.info(
                    "Worker status changed",
                    worker_id=worker_id,
                    old_status=old.value,
                    new_status=status.value,
                )

    async def add_worker(self, worker: GPUWorker) -> None:
        """Dynamically add a new GPU worker (autoscaling)."""
        self._workers[worker.worker_id] = worker
        logger.info("Worker added to pool", worker_id=worker.worker_id)

    async def remove_worker(self, worker_id: str) -> Optional[GPUWorker]:
        """Remove a worker from the pool (scale-down)."""
        worker = self._workers.pop(worker_id, None)
        if worker:
            logger.info("Worker removed from pool", worker_id=worker_id)
        return worker

    # -----------------------------------------------------------------------
    # Cluster Stats
    # -----------------------------------------------------------------------

    def cluster_utilization(self) -> dict[str, float]:
        """Return per-worker utilization stats."""
        return {
            wid: w.load_score * 100
            for wid, w in self._workers.items()
            if w.status != GPUStatus.OFFLINE
        }

    def total_free_vram_bytes(self) -> int:
        return sum(
            w.memory.free_bytes for w in self._workers.values()
            if w.is_available
        )
